from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .config import ExperimentConfig
from .benchmarks import BenchmarkSystem
from .polynomial import PolynomialAbstraction, eval_gradient_basis, eval_monomials, monomial_exponents
from .sos import (
    add_poly,
    affine_substitute_poly,
    barrier_polynomial,
    constant_poly,
    evaluate_poly,
    fit_vector_polynomial,
    generator_polynomial,
    putinar_sos_constraints,
    putinar_sos_constraints_on_generators,
    scale_poly,
    vector_polynomial_from_coeffs,
)
from .utils import relu
from .transition import DiscreteTransitionAbstraction, DiscreteTransitionCell


@dataclass
class BarrierResult:
    coeffs: np.ndarray
    exponents: list[tuple[int, ...]]
    rho: float
    objective: float
    success: bool
    max_initial_violation: float
    max_unsafe_violation: float
    max_positive_violation: float
    max_generator_violation: float
    n_counterexamples: int
    solver_backend: str = "sample"
    solver_status: str = "unknown"
    drift_fit_error: float = 0.0
    abstraction_error_bound: float = 0.0
    abstraction_error_mode: str = "off"
    epsilon: float = 0.0
    generator_c: float = 0.0
    time_horizon: float = 0.0
    certificate_method: str = "fixed_epsilon_sbc"
    verification_level: str = "sampled_constraints"
    diffusion_generator_error_bound: float = 0.0
    sdp_max_constraint_violation: float = 0.0
    sdp_min_gram_eigenvalue: float = 0.0
    gamma: float | None = None
    formulation: str = "prajna_optimization"
    initial_condition_mode: str = "worst_case"
    initial_distribution: str | None = None
    initial_barrier_statistic: float = 0.0
    dynamics_semantics: str = "continuous_time_generator"
    transition_expectation_error_bound: float = 0.0
    transition_sample_count: int = 0

    def to_dict(self) -> dict[str, object]:
        risk_bound = max(0.0, float(self.rho))
        safety_lower = max(0.0, min(1.0, 1.0 - risk_bound)) if self.success else 0.0
        return {
            "solver_backend": self.solver_backend,
            "solver_status": self.solver_status,
            "certificate_method": self.certificate_method,
            "verification_level": self.verification_level,
            "drift_fit_error": float(self.drift_fit_error),
            "abstraction_error_bound": float(self.abstraction_error_bound),
            "abstraction_error_mode": self.abstraction_error_mode,
            "rho": float(risk_bound),
            "risk_bound": float(risk_bound),
            "gamma": None if self.gamma is None else float(self.gamma),
            "formulation": self.formulation,
            "initial_condition_mode": self.initial_condition_mode,
            "initial_distribution": self.initial_distribution,
            "initial_barrier_statistic": float(self.initial_barrier_statistic),
            "dynamics_semantics": self.dynamics_semantics,
            "transition_expectation_error_bound": float(self.transition_expectation_error_bound),
            "transition_sample_count": int(self.transition_sample_count),
            "epsilon": float(self.epsilon),
            "generator_c": float(self.generator_c),
            "time_horizon": float(self.time_horizon),
            "diffusion_generator_error_bound": float(self.diffusion_generator_error_bound),
            "sdp_max_constraint_violation": float(self.sdp_max_constraint_violation),
            "sdp_min_gram_eigenvalue": float(self.sdp_min_gram_eigenvalue),
            "safety_probability_lower_bound": float(safety_lower),
            "objective": float(self.objective),
            "success": bool(self.success),
            "coeffs": [float(v) for v in self.coeffs],
            "exponents": [list(e) for e in self.exponents],
            "max_initial_violation": float(self.max_initial_violation),
            "max_unsafe_violation": float(self.max_unsafe_violation),
            "max_positive_violation": float(self.max_positive_violation),
            "max_generator_violation": float(self.max_generator_violation),
            "n_counterexamples": int(self.n_counterexamples),
        }


def mosek_sos_available() -> bool:
    try:
        import cvxpy as cp  # type: ignore

        import mosek  # noqa: F401  # type: ignore
    except Exception:
        return False
    return "MOSEK" in cp.installed_solvers()


class SampledBarrierSynthesizer:
    """Sample-based fallback for stochastic barrier synthesis.

    This is not a full SOS solver. It mirrors the four certificate inequalities
    on sampled points so the prototype remains runnable without cvxpy/MOSEK.
    """

    def __init__(self, abstraction: PolynomialAbstraction, system: BenchmarkSystem, config: ExperimentConfig):
        self.abstraction = abstraction
        self.system = system
        self.config = config
        self.verify_box = system.barrier_box if system.barrier_box is not None else system.verify_box
        self.exponents = monomial_exponents(config.state_dim, config.barrier_degree)
        self.rng = np.random.default_rng(config.seed + 100)
        self._extra_drift_fit_points = np.empty((0, self.config.state_dim), dtype=float)

    def fit(self) -> BarrierResult:
        init_x = self.system.sample_initial_states(self.rng, self.config.barrier_samples)
        if bool(getattr(self.system, "absorbing_stop", False)) and self.system.barrier_box is not None:
            # For a continuous stopped process the generator is constrained on
            # the active domain; after exit into the safe stop set it is zero.
            verify_x = self._sample_box(self.verify_box, self.config.barrier_samples)
        else:
            verify_x = self.system.sample_verify_states(self.rng, self.config.barrier_samples)
        unsafe_x = self.system.sample_unsafe_states(self.rng, self.config.barrier_samples)
        init_counterexamples = np.empty((0, self.config.state_dim), dtype=float)
        unsafe_counterexamples = np.empty((0, self.config.state_dim), dtype=float)
        verify_counterexamples = np.empty((0, self.config.state_dim), dtype=float)

        best: BarrierResult | None = None
        for round_index in range(max(1, self.config.counterexample_rounds + 1)):
            init_aug = np.vstack([init_x, init_counterexamples]) if init_counterexamples.size else init_x
            unsafe_aug = np.vstack([unsafe_x, unsafe_counterexamples]) if unsafe_counterexamples.size else unsafe_x
            verify_aug = np.vstack([verify_x, verify_counterexamples]) if verify_counterexamples.size else verify_x
            result = self._solve(init_aug, unsafe_aug, verify_aug)
            best = result

            if round_index >= int(self.config.counterexample_rounds):
                break
            new_init, new_unsafe, new_verify, max_violation = self._find_counterexamples(result.coeffs, result.epsilon)
            if max_violation <= 1e-3:
                break
            if new_init.size:
                init_counterexamples = np.vstack([init_counterexamples, new_init])
            if new_unsafe.size:
                unsafe_counterexamples = np.vstack([unsafe_counterexamples, new_unsafe])
            if new_verify.size:
                verify_counterexamples = np.vstack([verify_counterexamples, new_verify])

        assert best is not None
        best.n_counterexamples = int(
            init_counterexamples.shape[0] + unsafe_counterexamples.shape[0] + verify_counterexamples.shape[0]
        )
        return best

    def evaluate_B(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self._basis_matrix(x) @ coeffs

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        return eval_monomials(x, self.exponents)

    def _sbc_epsilon(self) -> float:
        return max(0.0, float(getattr(self.config, "sbc_epsilon", 0.05)))

    def _barrier_formulation(self) -> str:
        mode = str(getattr(self.config, "barrier_formulation", "prajna_optimization")).strip().lower().replace("-", "_")
        aliases = {
            "prajna": "prajna_optimization",
            "optimize_gamma": "prajna_optimization",
            "fixed": "fixed_epsilon",
            "feasibility": "fixed_epsilon",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"prajna_optimization", "fixed_epsilon"}:
            raise ValueError("barrier_formulation must be 'prajna_optimization' or 'fixed_epsilon'")
        return mode

    def _optimizes_rho(self) -> bool:
        return self._barrier_formulation() == "prajna_optimization"

    def _rho_upper(self) -> float:
        return min(1.0, max(0.0, float(getattr(self.config, "sos_rho_upper", 1.0))))

    def _clamp_rho(self, value: float) -> float:
        return min(self._rho_upper(), max(0.0, float(value)))

    def _generator_limit(self) -> float:
        return 0.0 if self._optimizes_rho() else self._sbc_c()

    def _constraint_margin(self) -> float:
        # Theorem 7 admits the always-feasible B=1, gamma=1 certificate.
        # Artificial strict margins would remove that certificate.
        return 0.0 if self._optimizes_rho() else float(self.config.sos_margin)

    def _certificate_method(self) -> str:
        if self._optimizes_rho():
            return "prajna2004_rho_optimization"
        return "fixed_epsilon_sbc"

    def _initial_condition_mode(self) -> str:
        mode = str(getattr(self.config, "initial_condition_mode", "worst_case")).strip().lower().replace("-", "_")
        aliases = {
            "unknown": "worst_case",
            "set": "worst_case",
            "pointwise": "worst_case",
            "known": "uniform_box",
            "uniform": "uniform_box",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"worst_case", "uniform_box"}:
            raise ValueError("initial_condition_mode must be 'worst_case' or 'uniform_box'")
        return mode

    def _initial_distribution(self) -> str | None:
        return "uniform_on_initial_box" if self._initial_condition_mode() == "uniform_box" else None

    def _uniform_initial_moment_vector(self) -> np.ndarray:
        moments = np.ones(len(self.exponents), dtype=float)
        for term_index, exponent in enumerate(self.exponents):
            value = 1.0
            for power, (lo, hi) in zip(exponent, self.system.initial_box):
                p = int(power)
                if p == 0:
                    continue
                lo_value = float(lo)
                hi_value = float(hi)
                if abs(hi_value - lo_value) <= 1e-15:
                    value *= lo_value**p
                else:
                    value *= (hi_value ** (p + 1) - lo_value ** (p + 1)) / (
                        float(p + 1) * (hi_value - lo_value)
                    )
            moments[term_index] = value
        return moments

    def _initial_barrier_statistic(self, coeffs: np.ndarray, init_x: np.ndarray) -> float:
        coeff_arr = np.asarray(coeffs, dtype=float).reshape(-1)
        if self._initial_condition_mode() == "uniform_box":
            return float(self._uniform_initial_moment_vector() @ coeff_arr)
        values = self.evaluate_B(coeff_arr, init_x)
        return float(np.max(values)) if values.size else 0.0

    def _initial_violation(self, coeffs: np.ndarray, rho: float, init_x: np.ndarray, margin: float = 0.0) -> float:
        if self._initial_condition_mode() == "uniform_box":
            return float(relu(self._initial_barrier_statistic(coeffs, init_x) - (float(rho) - float(margin))))
        values = self.evaluate_B(coeffs, init_x)
        return float(np.max(relu(values - (float(rho) - float(margin))))) if values.size else 0.0

    def _sbc_c(self) -> float:
        return float(getattr(self.config, "sbc_c", 0.0))

    def _sbc_time_horizon(self) -> float:
        if self._optimizes_rho():
            return 0.0
        configured = float(getattr(self.config, "sbc_time_horizon", 0.0))
        if configured > 0.0:
            return configured
        if self._sbc_c() <= 0.0:
            return 0.0
        return float(self.config.diagnostic_horizon) * float(self.config.dt)

    def _sbc_risk_bound(self) -> float:
        return max(0.0, self._sbc_epsilon() + max(0.0, self._sbc_c()) * max(0.0, self._sbc_time_horizon()))

    def _solve(self, init_x: np.ndarray, unsafe_x: np.ndarray, verify_x: np.ndarray) -> BarrierResult:
        k = len(self.exponents)
        coeff0 = np.zeros(k, dtype=float)
        for idx, exp in enumerate(self.exponents):
            if exp == (2, 0) or exp == (0, 2):
                coeff0[idx] = 0.25
        optimize_rho = self._optimizes_rho()
        base = np.concatenate([coeff0, np.array([1.0])]) if optimize_rho else coeff0

        def objective(z: np.ndarray) -> float:
            coeffs = z[:k]
            margin = self._constraint_margin()
            initial_bound = float(z[k]) if optimize_rho else self._sbc_epsilon()
            generator_limit = self._generator_limit()
            b_unsafe = self.evaluate_B(coeffs, unsafe_x)
            b_verify = self.evaluate_B(coeffs, verify_x)
            gen = self.abstraction.generator(coeffs, self.exponents, verify_x, self.system)
            initial_violation = self._initial_violation(coeffs, initial_bound, init_x, margin)
            loss = 60.0 * initial_violation**2
            loss += 60.0 * np.mean(relu(1.0 + margin - b_unsafe) ** 2)
            loss += 20.0 * np.mean(relu(margin - b_verify) ** 2)
            loss += 50.0 * np.mean(relu(gen - generator_limit + margin) ** 2)
            loss += 0.01 * float(np.sum(coeffs**2))
            if optimize_rho:
                loss += initial_bound
            return float(loss)

        best_opt = None
        best_value = float("inf")
        for restart in range(4):
            jitter = self.rng.normal(0.0, 0.05, size=base.shape)
            z0 = base + jitter
            if optimize_rho:
                z0[k] = self._clamp_rho(z0[k])
            opt = minimize(
                objective,
                z0,
                method="Powell",
                bounds=([(None, None)] * k + [(0.0, self._rho_upper())]) if optimize_rho else None,
                options={"maxiter": self.config.barrier_max_iter, "disp": False},
            )
            if float(opt.fun) < best_value:
                best_value = float(opt.fun)
                best_opt = opt

        assert best_opt is not None
        solution = np.asarray(best_opt.x, dtype=float)
        coeffs = solution[:k]
        initial_bound = self._clamp_rho(solution[k]) if optimize_rho else self._sbc_epsilon()
        risk_bound = initial_bound if optimize_rho else self._sbc_risk_bound()
        metrics = self._metrics(coeffs, initial_bound, init_x, unsafe_x, verify_x)
        return BarrierResult(
            coeffs=coeffs,
            exponents=self.exponents,
            rho=risk_bound,
            objective=float(best_opt.fun),
            success=bool(best_opt.success and max(metrics.values()) <= 1e-1),
            n_counterexamples=0,
            solver_backend="sample",
            solver_status=str(best_opt.message),
            epsilon=initial_bound,
            generator_c=self._generator_limit(),
            time_horizon=self._sbc_time_horizon(),
            certificate_method=self._certificate_method(),
            verification_level="sampled_constraints_not_formal",
            gamma=initial_bound if optimize_rho else None,
            formulation=self._barrier_formulation(),
            initial_condition_mode=self._initial_condition_mode(),
            initial_distribution=self._initial_distribution(),
            initial_barrier_statistic=self._initial_barrier_statistic(coeffs, init_x),
            **metrics,
        )

    def _metrics(
        self,
        coeffs: np.ndarray,
        rho: float,
        init_x: np.ndarray,
        unsafe_x: np.ndarray,
        verify_x: np.ndarray,
    ) -> dict[str, float]:
        b_unsafe = self.evaluate_B(coeffs, unsafe_x)
        b_verify = self.evaluate_B(coeffs, verify_x)
        gen = self.abstraction.generator(coeffs, self.exponents, verify_x, self.system)
        return {
            "max_initial_violation": self._initial_violation(coeffs, rho, init_x),
            "max_unsafe_violation": float(np.max(relu(1.0 - b_unsafe))),
            "max_positive_violation": float(np.max(relu(-b_verify))),
            "max_generator_violation": float(np.max(relu(gen - self._generator_limit()))),
        }

    def _violation_score(
        self,
        coeffs: np.ndarray,
        rho: float,
        init_x: np.ndarray,
        unsafe_x: np.ndarray,
        verify_x: np.ndarray,
    ) -> np.ndarray:
        b_verify = self.evaluate_B(coeffs, verify_x)
        gen = self.abstraction.generator(coeffs, self.exponents, verify_x, self.system)
        return relu(-b_verify) + relu(gen - self._generator_limit())

    def _find_counterexamples(
        self,
        coeffs: np.ndarray,
        rho: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        dense_count = max(200, self.config.barrier_samples * 3)
        init_dense = self._probe_box(self.system.initial_box, dense_count)
        unsafe_dense = self._probe_box(self.system.unsafe_box, dense_count)
        verify_dense = self._probe_box(self.verify_box, dense_count)

        b_unsafe = self.evaluate_B(coeffs, unsafe_dense)
        b_verify = self.evaluate_B(coeffs, verify_dense)
        gen = self.abstraction.generator(coeffs, self.exponents, verify_dense, self.system)

        if self._initial_condition_mode() == "uniform_box":
            init_violation = np.empty((0,), dtype=float)
            initial_measure_violation = self._initial_violation(coeffs, rho, init_dense)
        else:
            init_violation = relu(self.evaluate_B(coeffs, init_dense) - rho)
            initial_measure_violation = 0.0
        unsafe_violation = relu(1.0 - b_unsafe)
        verify_violation = relu(-b_verify)
        generator_violation = relu(gen - self._generator_limit())
        if self._constraint_margin() > 1e-6:
            margin = self._constraint_margin()
            if self._initial_condition_mode() == "uniform_box":
                initial_measure_violation = self._initial_violation(coeffs, rho, init_dense, margin)
            else:
                init_violation = relu(self.evaluate_B(coeffs, init_dense) - (rho - margin))
            unsafe_violation = relu(1.0 + margin - b_unsafe)
            verify_violation = relu(margin - b_verify)
            generator_violation = relu(gen - self._generator_limit() + margin)
        verify_total = verify_violation + generator_violation

        new_init = np.empty((0, self.config.state_dim), dtype=float)
        new_unsafe = np.empty((0, self.config.state_dim), dtype=float)
        new_verify = np.empty((0, self.config.state_dim), dtype=float)
        max_violation = float(initial_measure_violation)

        init_idx = self._top_violation_indices(init_violation, limit=3)
        if init_idx.size:
            new_init = init_dense[init_idx]
            max_violation = max(max_violation, float(np.max(init_violation[init_idx])))

        unsafe_idx = self._top_violation_indices(unsafe_violation, limit=3)
        if unsafe_idx.size:
            new_unsafe = unsafe_dense[unsafe_idx]
            max_violation = max(max_violation, float(np.max(unsafe_violation[unsafe_idx])))

        verify_idx = self._top_violation_indices(verify_total, limit=5)
        if verify_idx.size:
            new_verify = verify_dense[verify_idx]
            max_violation = max(max_violation, float(np.max(verify_total[verify_idx])))

        return new_init, new_unsafe, new_verify, max_violation

    @staticmethod
    def _top_violation_indices(values: np.ndarray, limit: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == 0:
            return np.empty((0,), dtype=int)
        positive = np.where(arr > 1e-3)[0]
        if positive.size == 0:
            return np.empty((0,), dtype=int)
        order = positive[np.argsort(arr[positive])[::-1]]
        return order[: max(1, int(limit))]

    def _sample_box(self, box: tuple[tuple[float, float], ...], n: int) -> np.ndarray:
        lows = np.array([b[0] for b in box], dtype=float)
        highs = np.array([b[1] for b in box], dtype=float)
        return self.rng.uniform(lows, highs, size=(n, len(box)))

    def _probe_box(self, box: tuple[tuple[float, float], ...], n: int) -> np.ndarray:
        random_points = self._sample_box(box, n)
        corners = self._box_corners(box)
        if len(box) != 2:
            return np.vstack([random_points, corners])
        side = max(64, int(np.sqrt(max(1, n)) * 2))
        x0 = np.linspace(box[0][0], box[0][1], side)
        x1 = np.linspace(box[1][0], box[1][1], side)
        grid0, grid1 = np.meshgrid(x0, x1, indexing="ij")
        grid = np.column_stack([grid0.reshape(-1), grid1.reshape(-1)])
        edge_n = max(64, side)
        xs = np.linspace(box[0][0], box[0][1], edge_n)
        ys = np.linspace(box[1][0], box[1][1], edge_n)
        lower = np.column_stack([xs, np.full_like(xs, box[1][0])])
        upper = np.column_stack([xs, np.full_like(xs, box[1][1])])
        left = np.column_stack([np.full_like(ys, box[0][0]), ys])
        right = np.column_stack([np.full_like(ys, box[0][1]), ys])
        corner_grids = []
        local_side = max(24, side // 2)
        span_x = min((box[0][1] - box[0][0]) * 0.15, (box[0][1] - box[0][0]))
        span_y = min((box[1][1] - box[1][0]) * 0.15, (box[1][1] - box[1][0]))
        for cx in [box[0][0], box[0][1]]:
            for cy in [box[1][0], box[1][1]]:
                x_lo = cx
                x_hi = min(box[0][1], cx + span_x) if cx == box[0][0] else max(box[0][0], cx - span_x)
                y_lo = cy
                y_hi = min(box[1][1], cy + span_y) if cy == box[1][0] else max(box[1][0], cy - span_y)
                xs_local = np.linspace(min(x_lo, x_hi), max(x_lo, x_hi), local_side)
                ys_local = np.linspace(min(y_lo, y_hi), max(y_lo, y_hi), local_side)
                lx, ly = np.meshgrid(xs_local, ys_local, indexing="ij")
                corner_grids.append(np.column_stack([lx.reshape(-1), ly.reshape(-1)]))
        return np.vstack([random_points, grid, lower, upper, left, right, *corner_grids, corners])

    @staticmethod
    def _box_corners(box: tuple[tuple[float, float], ...]) -> np.ndarray:
        lows_highs = [[lo, hi] for lo, hi in box]
        mesh = np.meshgrid(*lows_highs, indexing="ij")
        return np.column_stack([m.reshape(-1) for m in mesh])


class MosekSampledBarrierSynthesizer(SampledBarrierSynthesizer):
    """MOSEK-backed convex solve for sampled barrier inequalities.

    This is a solver upgrade for the sampled constraints, not a full SOS
    certificate. Full SOS still requires polynomial multiplier/Gram encoding.
    """

    def _solve(self, init_x: np.ndarray, unsafe_x: np.ndarray, verify_x: np.ndarray) -> BarrierResult:
        try:
            import cvxpy as cp  # type: ignore
        except Exception as exc:
            raise RuntimeError("cvxpy is required for the MOSEK sampled barrier backend") from exc
        if "MOSEK" not in cp.installed_solvers():
            raise RuntimeError("MOSEK is not available in cvxpy.installed_solvers()")

        k = len(self.exponents)
        coeffs = cp.Variable(k)
        optimize_rho = self._optimizes_rho()
        rho = cp.Variable(name="rho") if optimize_rho else None

        phi_init = eval_monomials(init_x, self.exponents)
        phi_unsafe = eval_monomials(unsafe_x, self.exponents)
        phi_verify = eval_monomials(verify_x, self.exponents)
        gen_matrix = self._generator_matrix(verify_x)
        margin = self._constraint_margin()
        initial_bound = rho if optimize_rho else self._sbc_epsilon()
        generator_limit = self._generator_limit()

        initial_expression = (
            self._uniform_initial_moment_vector() @ coeffs
            if self._initial_condition_mode() == "uniform_box"
            else phi_init @ coeffs
        )
        constraints = [
            initial_expression <= initial_bound - margin,
            phi_unsafe @ coeffs >= 1.0 + margin,
            phi_verify @ coeffs >= margin,
            gen_matrix @ coeffs <= generator_limit - margin,
        ]
        if optimize_rho:
            assert rho is not None
            constraints.extend([rho >= 0.0, rho <= self._rho_upper()])
            objective = cp.Minimize(rho)
        else:
            if self._initial_condition_mode() == "uniform_box":
                initial_objective = self._uniform_initial_moment_vector() @ coeffs
            else:
                initial_objective = cp.sum(phi_init @ coeffs) / max(1, init_x.shape[0])
            primary_objective = float(self.config.sos_initial_objective_weight) * initial_objective
            objective = cp.Minimize(primary_objective + 1e-4 * cp.sum_squares(coeffs))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK, verbose=False)

        if coeffs.value is None:
            raise RuntimeError(f"MOSEK sampled barrier solve failed with status: {problem.status}")

        coeff_arr = np.asarray(coeffs.value, dtype=float).reshape(-1)
        optimized_rho = self._clamp_rho(float(rho.value)) if rho is not None and rho.value is not None else None
        initial_bound_value = optimized_rho if optimized_rho is not None else self._sbc_epsilon()
        risk_bound = initial_bound_value if optimize_rho else self._sbc_risk_bound()
        metrics = self._metrics(coeff_arr, initial_bound_value, init_x, unsafe_x, verify_x)
        max_constraint_violation = max(
            float(np.max(np.abs(np.asarray(constraint.violation(), dtype=float)))) for constraint in constraints
        )
        return BarrierResult(
            coeffs=coeff_arr,
            exponents=self.exponents,
            rho=risk_bound,
            objective=float(problem.value),
            success=bool(
                problem.status == "optimal"
                and max(metrics.values()) <= 1e-6
                and max_constraint_violation <= float(self.config.sos_constraint_tolerance)
            ),
            n_counterexamples=0,
            solver_backend="mosek_sampled",
            solver_status=str(problem.status),
            epsilon=initial_bound_value,
            generator_c=generator_limit,
            time_horizon=self._sbc_time_horizon(),
            certificate_method=self._certificate_method(),
            verification_level="sampled_constraints_not_formal",
            sdp_max_constraint_violation=max_constraint_violation,
            gamma=optimized_rho,
            formulation=self._barrier_formulation(),
            initial_condition_mode=self._initial_condition_mode(),
            initial_distribution=self._initial_distribution(),
            initial_barrier_statistic=self._initial_barrier_statistic(coeff_arr, init_x),
            **metrics,
        )

    def _generator_matrix(self, verify_x: np.ndarray) -> np.ndarray:
        cols = []
        for j in range(len(self.exponents)):
            basis_coeffs = np.zeros(len(self.exponents), dtype=float)
            basis_coeffs[j] = 1.0
            cols.append(self.abstraction.generator(basis_coeffs, self.exponents, verify_x, self.system))
        return np.column_stack(cols)


class SosBarrierSynthesizer(SampledBarrierSynthesizer):
    """Putinar SOS/SDP stochastic barrier synthesizer solved with MOSEK."""

    def __init__(
        self,
        abstraction: PolynomialAbstraction,
        system: BenchmarkSystem,
        config: ExperimentConfig,
        fixed_coeffs: np.ndarray | None = None,
    ) -> None:
        super().__init__(abstraction, system, config)
        self.fixed_coeffs = None if fixed_coeffs is None else np.asarray(fixed_coeffs, dtype=float).reshape(-1)
        if self.fixed_coeffs is not None and self.fixed_coeffs.shape[0] != len(self.exponents):
            raise ValueError("fixed_coeffs length does not match the barrier polynomial basis")

    def fit(self) -> BarrierResult:
        init_x = self.system.sample_initial_states(self.rng, self.config.barrier_samples)
        verify_x = self.system.sample_verify_states(self.rng, self.config.barrier_samples)
        unsafe_x = self.system.sample_unsafe_states(self.rng, self.config.barrier_samples)
        self._extra_drift_fit_points = np.empty((0, self.config.state_dim), dtype=float)

        best: BarrierResult | None = None
        for round_index in range(max(1, self.config.counterexample_rounds + 1)):
            result = self._solve_sos(init_x, unsafe_x, verify_x)
            best = result

            if round_index >= int(self.config.counterexample_rounds):
                break
            new_init, new_unsafe, new_verify, max_violation = self._find_counterexamples(result.coeffs, result.epsilon)
            if max_violation <= 1e-3:
                break
            if new_init.size:
                self._extra_drift_fit_points = np.vstack([self._extra_drift_fit_points, new_init])
            if new_unsafe.size:
                self._extra_drift_fit_points = np.vstack([self._extra_drift_fit_points, new_unsafe])
            if new_verify.size:
                self._extra_drift_fit_points = np.vstack([self._extra_drift_fit_points, new_verify])

        assert best is not None
        best.n_counterexamples = int(self._extra_drift_fit_points.shape[0])
        return best

    def _solve_sos(self, init_x: np.ndarray, unsafe_x: np.ndarray, verify_x: np.ndarray) -> BarrierResult:
        try:
            import cvxpy as cp  # type: ignore
        except Exception as exc:
            raise RuntimeError("cvxpy is required for the MOSEK SOS backend") from exc
        if "MOSEK" not in cp.installed_solvers():
            raise RuntimeError("MOSEK is not available in cvxpy.installed_solvers()")

        drift_cells, drift_fit_error = self._prepare_sos_dynamics()
        k = len(self.exponents)
        coeffs = cp.Variable(k, name="barrier_coeffs")
        optimize_rho = self._optimizes_rho()
        rho = cp.Variable(name="rho") if optimize_rho else None
        barrier_poly = self._barrier_polynomial(coeffs)
        phi_init = self._basis_matrix(init_x)
        error_mode = self._abstraction_error_mode()

        margin = self._constraint_margin()
        epsilon = self._sbc_epsilon()
        generator_limit = self._generator_limit()
        failed_initial_bound = 1.0 if optimize_rho else epsilon
        failed_risk_bound = 1.0 if optimize_rho else self._sbc_risk_bound()
        constraints = []
        # A covariance PSD upper bound alone is not conservative when Hess(B)
        # is indefinite. The absolute-coefficient envelope below bounds the
        # missing trace term for every covariance 0 <= G <= Gbar.
        coeff_abs = cp.Variable(k, nonneg=True, name="barrier_coeff_abs")
        constraints.extend([coeff_abs >= coeffs, coeff_abs >= -coeffs])
        if self.fixed_coeffs is not None:
            constraints.append(coeffs == self.fixed_coeffs)
        if optimize_rho:
            assert rho is not None
            constraints.extend([rho >= 0.0, rho <= self._rho_upper()])
        gram_matrices = []
        initial_bound = rho if rho is not None else epsilon
        conditions = [
            ("unsafe", add_poly(barrier_poly, constant_poly(self.config.state_dim, -1.0 - margin)), self.system.unsafe_generators(), self.system.unsafe_box),
            (
                "positive",
                add_poly(barrier_poly, constant_poly(self.config.state_dim, -margin)),
                self.system.domain_generators(self._positivity_box()),
                self._positivity_box(),
            ),
        ]
        if self._initial_condition_mode() == "uniform_box":
            constraints.append(self._uniform_initial_moment_vector() @ coeffs <= initial_bound - margin)
        else:
            conditions.insert(
                0,
                (
                    "initial",
                    add_poly(
                        constant_poly(self.config.state_dim, initial_bound - margin),
                        scale_poly(barrier_poly, -1.0),
                    ),
                    self.system.initial_generators(),
                    self.system.initial_box,
                ),
            )
        conditions.extend(
            self._dynamics_sos_conditions(barrier_poly, coeffs, coeff_abs, drift_cells, generator_limit, margin)
        )
        for name, poly, generators, box in conditions:
            center, scale, normalized_box = self._normalization_for_box(box)
            normalized_poly = affine_substitute_poly(poly, center, scale)
            normalized_generators = [affine_substitute_poly(generator, center, scale) for generator in generators]
            sos_constraints, grams = putinar_sos_constraints_on_generators(
                cp,
                normalized_poly,
                normalized_generators,
                relaxation_degree=self._relaxation_degree(poly),
                name=f"{name}_cert",
            )
            constraints.extend(sos_constraints)
            gram_matrices.extend(grams)

        regularizer = float(self.config.sos_regularization) * cp.sum_squares(coeffs)
        regularizer += float(self.config.sos_regularization) * cp.sum(coeff_abs)
        regularizer += float(self.config.sos_regularization) * sum(cp.trace(q) for q in gram_matrices)
        if optimize_rho:
            # The tiny secondary term selects a numerically conditioned point
            # on an otherwise highly non-unique minimum-rho SDP face.
            objective = cp.Minimize(rho + regularizer)
        else:
            if self._initial_condition_mode() == "uniform_box":
                initial_objective = self._uniform_initial_moment_vector() @ coeffs
            else:
                initial_objective = cp.sum(phi_init @ coeffs) / max(1, init_x.shape[0])
            primary_objective = float(self.config.sos_initial_objective_weight) * initial_objective
            objective = cp.Minimize(primary_objective + regularizer)
        problem = cp.Problem(objective, constraints)
        try:
            solve_kwargs: dict[str, Any] = {
                "solver": cp.MOSEK,
                "verbose": bool(self.config.sos_verbose),
            }
            mosek_params: dict[str, object] = {}
            if self.config.sos_mosek_tolerance is not None:
                tolerance = float(self.config.sos_mosek_tolerance)
                if not 0.0 < tolerance < 1.0:
                    raise ValueError("sos_mosek_tolerance must lie in (0, 1)")
                mosek_params.update(
                    {
                        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tolerance,
                        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tolerance,
                        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tolerance,
                    }
                )
            if mosek_params:
                solve_kwargs["mosek_params"] = mosek_params
            problem.solve(**solve_kwargs)
        except Exception as exc:
            return self._failed_sos_result(
                status=f"solver_error: {exc}",
                drift_fit_error=drift_fit_error,
                error_mode=error_mode,
                epsilon=failed_initial_bound,
                generator_c=generator_limit,
                risk_bound=failed_risk_bound,
                init_x=init_x,
                unsafe_x=unsafe_x,
                verify_x=verify_x,
            )

        status = str(problem.status)
        if coeffs.value is None:
            return self._failed_sos_result(
                status=status,
                drift_fit_error=drift_fit_error,
                error_mode=error_mode,
                epsilon=failed_initial_bound,
                generator_c=generator_limit,
                risk_bound=failed_risk_bound,
                init_x=init_x,
                unsafe_x=unsafe_x,
                verify_x=verify_x,
            )

        coeff_arr = np.asarray(coeffs.value, dtype=float).reshape(-1)
        optimized_rho = self._clamp_rho(float(rho.value)) if rho is not None and rho.value is not None else None
        initial_bound_value = optimized_rho if optimized_rho is not None else epsilon
        risk_bound = initial_bound_value if optimize_rho else self._sbc_risk_bound()
        metrics = self._metrics(coeff_arr, initial_bound_value, init_x, unsafe_x, verify_x)
        max_constraint_violation = self._max_constraint_violation(constraints)
        min_gram_eigenvalue = self._min_gram_eigenvalue(gram_matrices)
        status_ok = status == "optimal"
        metrics_ok = max(metrics.values()) <= float(self.config.sos_sample_tolerance)
        numerical_ok = (
            max_constraint_violation <= float(self.config.sos_constraint_tolerance)
            and min_gram_eigenvalue >= -float(self.config.sos_gram_eigenvalue_tolerance)
        )
        return BarrierResult(
            coeffs=coeff_arr,
            exponents=self.exponents,
            rho=risk_bound,
            objective=float(problem.value),
            success=bool(status_ok and metrics_ok and numerical_ok),
            n_counterexamples=0,
            solver_backend="mosek_sos",
            solver_status=status,
            drift_fit_error=drift_fit_error,
            abstraction_error_bound=self._abstraction_error_bound(drift_fit_error),
            abstraction_error_mode=error_mode,
            epsilon=initial_bound_value,
            generator_c=generator_limit,
            time_horizon=self._sbc_time_horizon(),
            certificate_method=self._certificate_method(),
            verification_level=self._verification_level(),
            diffusion_generator_error_bound=self._diffusion_generator_error_bound(coeff_arr, self.verify_box),
            sdp_max_constraint_violation=max_constraint_violation,
            sdp_min_gram_eigenvalue=min_gram_eigenvalue,
            gamma=optimized_rho,
            formulation=self._barrier_formulation(),
            initial_condition_mode=self._initial_condition_mode(),
            initial_distribution=self._initial_distribution(),
            initial_barrier_statistic=self._initial_barrier_statistic(coeff_arr, init_x),
            **metrics,
        )

    def _prepare_sos_dynamics(self) -> tuple[list[dict[str, object]], float]:
        drift_cells, drift_fit_error = self._fit_drift_polynomial_cells()
        self._sos_drift_cells = drift_cells
        return drift_cells, drift_fit_error

    def _barrier_polynomial(self, coeffs):
        return barrier_polynomial(coeffs, self.exponents)

    def _positivity_box(self):
        return self.verify_box

    def _dynamics_sos_conditions(
        self,
        barrier_poly,
        coeffs,
        coeff_abs,
        drift_cells,
        generator_limit: float,
        margin: float,
    ):
        conditions = []
        for cell_index, drift_cell in enumerate(drift_cells):
            cell_box = drift_cell["box"]
            drift_polys = drift_cell["polys"]
            assert isinstance(cell_box, tuple)
            assert isinstance(drift_polys, list)
            generator_poly = generator_polynomial(barrier_poly, drift_polys, self.abstraction.Gbar)
            robust_error: object = self._diffusion_generator_error_expression(cell_box, coeff_abs)
            error_bound = self._abstraction_error_bound(float(drift_cell["fit_error"]), cell_index=cell_index)
            if error_bound > 0.0:
                weights = self._gradient_l1_bound_weights(cell_box)
                robust_error = robust_error + float(error_bound) * (weights @ coeff_abs)
            generator_condition = add_poly(
                scale_poly(generator_poly, -1.0),
                constant_poly(self.config.state_dim, generator_limit - margin - robust_error),
            )
            conditions.append((f"generator_cell_{cell_index}", generator_condition, self.system.domain_generators(cell_box), cell_box))
        return conditions

    def _failed_sos_result(
        self,
        *,
        status: str,
        drift_fit_error: float,
        error_mode: str,
        epsilon: float,
        generator_c: float,
        risk_bound: float,
        init_x: np.ndarray,
        unsafe_x: np.ndarray,
        verify_x: np.ndarray,
    ) -> BarrierResult:
        fallback = np.zeros(len(self.exponents), dtype=float)
        metrics = self._metrics(fallback, epsilon, init_x, unsafe_x, verify_x)
        return BarrierResult(
            coeffs=fallback,
            exponents=self.exponents,
            rho=risk_bound,
            objective=1e30,
            success=False,
            n_counterexamples=0,
            solver_backend="mosek_sos",
            solver_status=status,
            drift_fit_error=drift_fit_error,
            abstraction_error_bound=self._abstraction_error_bound(drift_fit_error),
            abstraction_error_mode=error_mode,
            epsilon=epsilon,
            generator_c=generator_c,
            time_horizon=self._sbc_time_horizon(),
            verification_level=self._verification_level(),
            sdp_max_constraint_violation=1e30,
            sdp_min_gram_eigenvalue=-1e30,
            certificate_method=self._certificate_method(),
            gamma=None,
            formulation=self._barrier_formulation(),
            initial_condition_mode=self._initial_condition_mode(),
            initial_distribution=self._initial_distribution(),
            initial_barrier_statistic=self._initial_barrier_statistic(fallback, init_x),
            **metrics,
        )

    @staticmethod
    def _max_constraint_violation(constraints) -> float:
        maximum = 0.0
        for constraint in constraints:
            try:
                violation = constraint.violation()
            except Exception:
                return 1e30
            if violation is None:
                return 1e30
            arr = np.asarray(violation, dtype=float)
            if arr.size:
                maximum = max(maximum, float(np.max(np.abs(arr))))
        return maximum

    @staticmethod
    def _min_gram_eigenvalue(gram_matrices) -> float:
        minimum = float("inf")
        for gram in gram_matrices:
            if gram.value is None:
                return -1e30
            matrix = np.asarray(gram.value, dtype=float)
            symmetric = (matrix + matrix.T) / 2.0
            minimum = min(minimum, float(np.min(np.linalg.eigvalsh(symmetric))))
        return 0.0 if minimum == float("inf") else minimum

    def _fit_drift_polynomial_cells(self) -> tuple[list[dict[str, object]], float]:
        exact_provider = getattr(self.system, "exact_drift_polynomials", None)
        if callable(exact_provider) and not self.abstraction.partition_boxes:
            exact_polys = exact_provider()
            if exact_polys is not None:
                polys = [dict(poly) for poly in exact_polys]
                if len(polys) != int(self.config.state_dim):
                    raise ValueError("exact_drift_polynomials must return one polynomial per state dimension")
                return [
                    {
                        "box": self.verify_box,
                        "polys": polys,
                        "fit_error": 0.0,
                        "abstraction_error_bound": 0.0,
                    }
                ], 0.0

        boxes = self.abstraction.partition_boxes or [self.verify_box]
        cells: list[dict[str, object]] = []
        max_fit_error = 0.0
        for cell_index, cell_box in enumerate(boxes):
            fit_x = self._drift_fit_points(cell_box)
            fit_y = self.abstraction.drift(fit_x, self.system)
            exponents, coeffs, fit_error = fit_vector_polynomial(
                fit_x,
                fit_y,
                degree=int(self.config.sos_drift_degree),
            )
            max_fit_error = max(max_fit_error, float(fit_error))
            cells.append(
                {
                    "box": cell_box,
                    "polys": vector_polynomial_from_coeffs(exponents, coeffs),
                    "fit_error": float(fit_error),
                    "abstraction_error_bound": self._abstraction_error_bound(float(fit_error), cell_index=cell_index),
                }
            )
        return cells, max_fit_error

    def _fit_drift_polynomials(self) -> tuple[list[dict[tuple[int, ...], float]], float]:
        fit_x = self._drift_fit_points()
        fit_y = self.abstraction.drift(fit_x, self.system)
        exponents, coeffs, fit_error = fit_vector_polynomial(
            fit_x,
            fit_y,
            degree=int(self.config.sos_drift_degree),
        )
        return vector_polynomial_from_coeffs(exponents, coeffs), fit_error

    def _drift_fit_points(self, box: tuple[tuple[float, float], ...] | None = None) -> np.ndarray:
        target_box = self.verify_box if box is None else box
        n = max(int(self.config.sos_drift_fit_samples), len(self.verify_box) + 1)
        random_points = self._sample_box(target_box, n)
        corners = self._box_corners(target_box)
        grid_side = max(3, min(9, int(self.config.sos_drift_degree) + 2))
        grid = self._tensor_grid_points(target_box, grid_side)
        points = np.vstack([random_points, grid, corners])
        if self._extra_drift_fit_points.size:
            extra = self._extra_drift_fit_points[self._points_in_box(self._extra_drift_fit_points, target_box)]
            if extra.size:
                points = np.vstack([points, extra])
        return points

    def _metrics(
        self,
        coeffs: np.ndarray,
        rho: float,
        init_x: np.ndarray,
        unsafe_x: np.ndarray,
        verify_x: np.ndarray,
    ) -> dict[str, float]:
        b_unsafe = self.evaluate_B(coeffs, unsafe_x)
        b_verify = self.evaluate_B(coeffs, verify_x)
        gen = self._sos_generator_values(coeffs, verify_x)
        return {
            "max_initial_violation": self._initial_violation(coeffs, rho, init_x),
            "max_unsafe_violation": float(np.max(relu(1.0 - b_unsafe))),
            "max_positive_violation": float(np.max(relu(-b_verify))),
            "max_generator_violation": float(np.max(relu(gen - self._generator_limit()))),
        }

    def _find_counterexamples(
        self,
        coeffs: np.ndarray,
        rho: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        dense_count = max(200, self.config.barrier_samples * 3)
        init_dense = self._probe_box(self.system.initial_box, dense_count)
        unsafe_dense = self._probe_box(self.system.unsafe_box, dense_count)
        verify_dense = self._probe_box(self.verify_box, dense_count)

        b_unsafe = self.evaluate_B(coeffs, unsafe_dense)
        b_verify = self.evaluate_B(coeffs, verify_dense)
        gen = self._sos_generator_values(coeffs, verify_dense)

        if self._initial_condition_mode() == "uniform_box":
            init_violation = np.empty((0,), dtype=float)
            initial_measure_violation = self._initial_violation(coeffs, rho, init_dense)
        else:
            init_violation = relu(self.evaluate_B(coeffs, init_dense) - rho)
            initial_measure_violation = 0.0
        unsafe_violation = relu(1.0 - b_unsafe)
        verify_violation = relu(-b_verify)
        generator_violation = relu(gen - self._generator_limit())
        if self._constraint_margin() > 1e-6:
            margin = self._constraint_margin()
            if self._initial_condition_mode() == "uniform_box":
                initial_measure_violation = self._initial_violation(coeffs, rho, init_dense, margin)
            else:
                init_violation = relu(self.evaluate_B(coeffs, init_dense) - (rho - margin))
            unsafe_violation = relu(1.0 + margin - b_unsafe)
            verify_violation = relu(margin - b_verify)
            generator_violation = relu(gen - self._generator_limit() + margin)
        verify_total = verify_violation + generator_violation

        new_init = np.empty((0, self.config.state_dim), dtype=float)
        new_unsafe = np.empty((0, self.config.state_dim), dtype=float)
        new_verify = np.empty((0, self.config.state_dim), dtype=float)
        max_violation = float(initial_measure_violation)

        init_idx = self._top_violation_indices(init_violation, limit=3)
        if init_idx.size:
            new_init = init_dense[init_idx]
            max_violation = max(max_violation, float(np.max(init_violation[init_idx])))

        unsafe_idx = self._top_violation_indices(unsafe_violation, limit=3)
        if unsafe_idx.size:
            new_unsafe = unsafe_dense[unsafe_idx]
            max_violation = max(max_violation, float(np.max(unsafe_violation[unsafe_idx])))

        verify_idx = self._top_violation_indices(verify_total, limit=5)
        if verify_idx.size:
            new_verify = verify_dense[verify_idx]
            max_violation = max(max_violation, float(np.max(verify_total[verify_idx])))

        return new_init, new_unsafe, new_verify, max_violation

    def _sos_generator_values(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        drift_cells = getattr(self, "_sos_drift_cells", None)
        if drift_cells is None:
            drift_cells, _ = self._fit_drift_polynomial_cells()
            self._sos_drift_cells = drift_cells
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.zeros(x_arr.shape[0], dtype=float)
        assigned = np.zeros(x_arr.shape[0], dtype=bool)
        barrier_poly = barrier_polynomial(np.asarray(coeffs, dtype=float), self.exponents)
        for cell_index, cell in enumerate(drift_cells):
            cell_box = cell["box"]
            drift_polys = cell["polys"]
            assert isinstance(cell_box, tuple)
            assert isinstance(drift_polys, list)
            mask = self._points_in_box(x_arr, cell_box) & ~assigned
            if not np.any(mask):
                continue
            generator_poly = generator_polynomial(barrier_poly, drift_polys, self.abstraction.Gbar)
            center, scale, _ = self._normalization_for_box(cell_box)
            normalized_generator = affine_substitute_poly(generator_poly, center, scale)
            normalized_points = (x_arr[mask] - center) / scale
            cell_values = evaluate_poly(normalized_generator, normalized_points)
            error_bound = self._abstraction_error_bound(float(cell["fit_error"]), cell_index=cell_index)
            if error_bound > 0.0:
                cell_values = cell_values + error_bound * self._barrier_gradient_norm(coeffs, x_arr[mask])
            cell_values = cell_values + self._diffusion_generator_error_bound(coeffs, cell_box)
            values[mask] = cell_values
            assigned[mask] = True
        if np.any(~assigned):
            cell = drift_cells[0]
            drift_polys = cell["polys"]
            assert isinstance(drift_polys, list)
            generator_poly = generator_polynomial(barrier_poly, drift_polys, self.abstraction.Gbar)
            fallback_box = cell["box"]
            assert isinstance(fallback_box, tuple)
            center, scale, _ = self._normalization_for_box(fallback_box)
            normalized_generator = affine_substitute_poly(generator_poly, center, scale)
            normalized_points = (x_arr[~assigned] - center) / scale
            fallback = evaluate_poly(normalized_generator, normalized_points)
            error_bound = self._abstraction_error_bound(float(cell["fit_error"]), cell_index=0)
            if error_bound > 0.0:
                fallback = fallback + error_bound * self._barrier_gradient_norm(coeffs, x_arr[~assigned])
            fallback = fallback + self._diffusion_generator_error_bound(coeffs, self.verify_box)
            values[~assigned] = fallback
        return values

    def robust_generator_values(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the same robust generator bound used by the SOS program."""
        return self._sos_generator_values(coeffs, x)

    def candidate_counterexamples(
        self,
        coeffs: np.ndarray,
        *,
        per_condition: int = 8,
        probe_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return dense-probe violations for a fixed neural barrier candidate.

        These points are training feedback only. The subsequent SOS solve,
        rather than this probe, remains the certificate decision.
        """
        count = max(200, int(probe_count or self.config.barrier_samples * 4))
        init_x = self._probe_box(self.system.initial_box, count)
        unsafe_x = self._probe_box(self.system.unsafe_box, count)
        verify_x = self._probe_box(self.verify_box, count)
        margin = self._constraint_margin()
        if self._optimizes_rho() or self._initial_condition_mode() == "uniform_box":
            init_violation = np.zeros(init_x.shape[0], dtype=float)
        else:
            epsilon = self._sbc_epsilon()
            init_violation = relu(self.evaluate_B(coeffs, init_x) - (epsilon - margin))
        unsafe_violation = relu(1.0 + margin - self.evaluate_B(coeffs, unsafe_x))
        verify_violation = relu(margin - self.evaluate_B(coeffs, verify_x))
        generator_violation = relu(self._sos_generator_values(coeffs, verify_x) - self._generator_limit() + margin)
        return (
            init_x[self._top_violation_indices(init_violation, per_condition)],
            unsafe_x[self._top_violation_indices(unsafe_violation, per_condition)],
            verify_x[self._top_violation_indices(verify_violation + generator_violation, per_condition)],
        )

    def _abstraction_error_mode(self) -> str:
        mode = str(getattr(self.config, "sos_abstraction_error_mode", "off")).strip().lower()
        if mode not in {"off", "fit", "total"}:
            raise ValueError("sos_abstraction_error_mode must be 'off', 'fit', or 'total'")
        return mode

    def _verification_level(self) -> str:
        certificate = self.abstraction.abstraction_certificate
        if certificate.get("surrogate_error_semantics") == "paper_fixed":
            return "formal_sos_on_fixed_learned_surrogate_sde"
        if certificate.get("method") == "known_carla_closed_loop_drift_and_exact_additive_gaussian_covariance":
            return "sos_on_known_carla_gaussian_dynamics"
        if bool(certificate.get("terminal_pac_calibrated", False)):
            return "robust_sos_on_terminal_pac_calibrated_abstraction"
        if certificate.get("error_bound_mode") == "analytic" and certificate.get(
            "diffusion_bound_is_global_for_learned_sampler", False
        ):
            return "robust_sos_on_globally_bounded_learned_abstraction"
        return "robust_sos_on_empirical_abstraction"

    def _abstraction_error_bound(self, drift_fit_error: float, *, cell_index: int | None = None) -> float:
        mode = self._abstraction_error_mode()
        if mode == "off":
            return 0.0
        if mode == "fit":
            return max(0.0, float(drift_fit_error))
        residual_error = max(0.0, float(self.abstraction.epsilon_r))
        local_bounds = list(getattr(self.abstraction, "local_fit_errors", []))
        if cell_index is not None and 0 <= int(cell_index) < len(local_bounds):
            residual_error = max(0.0, float(local_bounds[int(cell_index)]))
            certificate = getattr(self.abstraction, "abstraction_certificate", {})
            if not bool(certificate.get("local_error_bounds_include_extra_margin", False)):
                residual_error += max(0.0, float(getattr(self.abstraction, "extra_error_margin", 0.0)))
        return max(0.0, float(drift_fit_error)) + residual_error

    def _barrier_gradient_norm(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        grad_basis = eval_gradient_basis(x, self.exponents)
        grad = np.einsum("kti,t->ki", grad_basis, np.asarray(coeffs, dtype=float))
        return np.linalg.norm(grad, axis=1)

    def _gradient_l1_bound_weights(self, box: tuple[tuple[float, float], ...]) -> np.ndarray:
        max_abs = np.asarray([max(abs(float(lo)), abs(float(hi))) for lo, hi in box], dtype=float)
        weights = np.zeros(len(self.exponents), dtype=float)
        for coeff_index, exp in enumerate(self.exponents):
            total = 0.0
            for var, power in enumerate(exp):
                if power == 0:
                    continue
                term = float(power)
                for idx, other_power in enumerate(exp):
                    effective = other_power - 1 if idx == var else other_power
                    if effective:
                        term *= float(max_abs[idx] ** effective)
                total += term
            weights[coeff_index] = total
        return weights

    def _diffusion_generator_error_expression(self, box: tuple[tuple[float, float], ...], coeff_abs):
        if bool(self.abstraction.abstraction_certificate.get("diffusion_covariance_is_exact", False)):
            return 0.0
        gamma = self._diffusion_covariance_upper_eigenvalue()
        dimension = float(self.config.state_dim)
        weights = self._hessian_l1_bound_weights(box)
        return 0.5 * gamma * np.sqrt(dimension) * (weights @ coeff_abs)

    def _diffusion_generator_error_bound(
        self,
        coeffs: np.ndarray,
        box: tuple[tuple[float, float], ...],
    ) -> float:
        if bool(self.abstraction.abstraction_certificate.get("diffusion_covariance_is_exact", False)):
            return 0.0
        gamma = self._diffusion_covariance_upper_eigenvalue()
        dimension = float(self.config.state_dim)
        weights = self._hessian_l1_bound_weights(box)
        return float(0.5 * gamma * np.sqrt(dimension) * (weights @ np.abs(coeffs)))

    def _diffusion_covariance_upper_eigenvalue(self) -> float:
        gbar = np.asarray(self.abstraction.Gbar, dtype=float)
        return max(0.0, float(np.max(np.linalg.eigvalsh((gbar + gbar.T) / 2.0))))

    def _hessian_l1_bound_weights(self, box: tuple[tuple[float, float], ...]) -> np.ndarray:
        max_abs = np.asarray([max(abs(float(lo)), abs(float(hi))) for lo, hi in box], dtype=float)
        weights = np.zeros(len(self.exponents), dtype=float)
        for coeff_index, exp in enumerate(self.exponents):
            total = 0.0
            for row, row_power in enumerate(exp):
                if row_power == 0:
                    continue
                for col, col_power in enumerate(exp):
                    if col_power == 0 or (row == col and row_power < 2):
                        continue
                    multiplier = float(row_power * (row_power - 1)) if row == col else float(row_power * col_power)
                    term = multiplier
                    for var, power in enumerate(exp):
                        reduced = power - int(var == row) - int(var == col)
                        if reduced:
                            term *= float(max_abs[var] ** reduced)
                    total += term
            weights[coeff_index] = total
        return weights

    @staticmethod
    def _normalization_for_box(
        box: tuple[tuple[float, float], ...],
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, float], ...]]:
        lows = np.asarray([b[0] for b in box], dtype=float)
        highs = np.asarray([b[1] for b in box], dtype=float)
        center = (lows + highs) / 2.0
        scale = np.maximum((highs - lows) / 2.0, 1e-12)
        normalized_box = tuple((-1.0, 1.0) for _ in box)
        return center, scale, normalized_box

    def _relaxation_degree(self, poly: dict[tuple[int, ...], object]) -> int:
        requested = int(self.config.sos_relaxation_degree)
        if requested > 0:
            degree = requested
        else:
            degree = max(sum(exp) for exp in poly.keys()) if poly else 0
        if degree % 2:
            degree += 1
        return max(2, degree)

    @staticmethod
    def _points_in_box(x: np.ndarray, box: tuple[tuple[float, float], ...], tol: float = 1e-10) -> np.ndarray:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        mask = np.ones(x_arr.shape[0], dtype=bool)
        for idx, (lo, hi) in enumerate(box):
            mask &= x_arr[:, idx] >= float(lo) - tol
            mask &= x_arr[:, idx] <= float(hi) + tol
        return mask

    @staticmethod
    def _tensor_grid_points(box: tuple[tuple[float, float], ...], count: int) -> np.ndarray:
        axes = [np.linspace(float(lo), float(hi), max(2, int(count))) for lo, hi in box]
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.reshape(-1) for m in mesh])

    @staticmethod
    def _box_corners(box: tuple[tuple[float, float], ...]) -> np.ndarray:
        lows_highs = [[lo, hi] for lo, hi in box]
        mesh = np.meshgrid(*lows_highs, indexing="ij")
        return np.stack([m.reshape(-1) for m in mesh], axis=1)


class DiscreteTransitionSosBarrierSynthesizer(SosBarrierSynthesizer):
    """SOS barrier for a learned one-step stochastic transition kernel.

    The dynamics constraint is

        E[B(X[k+1]) | X[k]=x] - B(x) <= 0,

    with an absolute-coefficient envelope for the fitted conditional-moment
    errors.  No continuous-time generator or reverse-time SDE is used here.
    """

    def __init__(
        self,
        abstraction: DiscreteTransitionAbstraction,
        system: BenchmarkSystem,
        config: ExperimentConfig,
        fixed_coeffs: np.ndarray | None = None,
    ) -> None:
        super().__init__(abstraction, system, config, fixed_coeffs=fixed_coeffs)  # type: ignore[arg-type]
        if list(abstraction.barrier_exponents) != list(self.exponents):
            raise ValueError("transition abstraction basis must match the configured barrier degree")
        self.verify_box = system.verify_box

    def fit(self) -> BarrierResult:
        result = super().fit()
        result.dynamics_semantics = "discrete_time_conditional_expectation"
        result.transition_expectation_error_bound = self.abstraction.max_total_error
        result.transition_sample_count = int(self.abstraction.transition_sample_count)
        result.diffusion_generator_error_bound = 0.0
        return result

    def _basis_matrix(self, x: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        normalized = (rows - self.abstraction.basis_center) / self.abstraction.basis_scale
        return eval_monomials(normalized, self.exponents)

    def _barrier_polynomial(self, coeffs):
        normalized_poly = barrier_polynomial(coeffs, self.exponents)
        return affine_substitute_poly(
            normalized_poly,
            -self.abstraction.basis_center / self.abstraction.basis_scale,
            1.0 / self.abstraction.basis_scale,
        )

    def _uniform_initial_moment_vector(self) -> np.ndarray:
        moments = np.ones(len(self.exponents), dtype=float)
        center = self.abstraction.basis_center
        scale = self.abstraction.basis_scale
        for term_index, exponent in enumerate(self.exponents):
            value = 1.0
            for coordinate, (power, (lo, hi)) in enumerate(zip(exponent, self.system.initial_box)):
                p = int(power)
                if p == 0:
                    continue
                z_lo = (float(lo) - float(center[coordinate])) / float(scale[coordinate])
                z_hi = (float(hi) - float(center[coordinate])) / float(scale[coordinate])
                if abs(z_hi - z_lo) <= 1e-15:
                    value *= z_lo**p
                else:
                    value *= (z_hi ** (p + 1) - z_lo ** (p + 1)) / (
                        float(p + 1) * (z_hi - z_lo)
                    )
            moments[term_index] = value
        return moments

    def _positivity_box(self):
        # The stopped state v=0 is reachable and absorbing, so B >= 0 must be
        # certified there as well as on the active transition cells.
        return self.system.verify_box

    def _prepare_sos_dynamics(self) -> tuple[list[DiscreteTransitionCell], float]:
        self._sos_transition_cells = self.abstraction.cells
        return self.abstraction.cells, self.abstraction.max_fit_error

    def _dynamics_sos_conditions(
        self,
        barrier_poly,
        coeffs,
        coeff_abs,
        transition_cells,
        generator_limit: float,
        margin: float,
    ):
        conditions = []
        error_mode = self._abstraction_error_mode()
        for cell_index, cell in enumerate(transition_cells):
            expected_poly = constant_poly(self.config.state_dim, 0.0)
            for basis_index, basis_poly in enumerate(cell.expected_basis_polys):
                expected_poly = add_poly(expected_poly, scale_poly(basis_poly, coeffs[basis_index]))
            difference_poly = add_poly(expected_poly, scale_poly(barrier_poly, -1.0))
            errors = self._transition_cell_errors(cell, error_mode)
            robust_error = errors @ coeff_abs
            condition = add_poly(
                scale_poly(difference_poly, -1.0),
                constant_poly(self.config.state_dim, generator_limit - margin - robust_error),
            )
            conditions.append(
                (
                    f"transition_expectation_cell_{cell_index}",
                    condition,
                    self.system.domain_generators(cell.box),
                    cell.box,
                )
            )
        return conditions

    def _sos_generator_values(self, coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.zeros(rows.shape[0], dtype=float)
        active = np.ones(rows.shape[0], dtype=bool)
        if bool(getattr(self.system, "absorbing_stop", False)) and rows.shape[1] >= 2:
            active &= rows[:, 1] > float(getattr(self.system, "stop_speed", 0.0))
        in_transition_domain = np.zeros(rows.shape[0], dtype=bool)
        for cell in self.abstraction.cells:
            in_transition_domain |= self._points_in_box(rows, cell.box)
        active &= in_transition_domain
        if np.any(active):
            values[active] = self.abstraction.barrier_difference(
                coeffs,
                rows[active],
                error_mode=self._abstraction_error_mode(),
            )
        return values

    def _abstraction_error_bound(self, drift_fit_error: float, *, cell_index: int | None = None) -> float:
        mode = self._abstraction_error_mode()
        if cell_index is not None and 0 <= int(cell_index) < len(self.abstraction.cells):
            return float(np.max(self._transition_cell_errors(self.abstraction.cells[int(cell_index)], mode)))
        return max(
            (float(np.max(self._transition_cell_errors(cell, mode))) for cell in self.abstraction.cells),
            default=0.0,
        )

    def _diffusion_generator_error_expression(self, box, coeff_abs):
        return 0.0

    def _diffusion_generator_error_bound(self, coeffs: np.ndarray, box) -> float:
        return 0.0

    def _verification_level(self) -> str:
        expectation = str(self.abstraction.abstraction_certificate.get("monte_carlo_bound", "unknown"))
        return f"sos_on_empirical_spatial_transition_abstraction_with_{expectation}"

    def _certificate_method(self) -> str:
        if self._optimizes_rho():
            return "discrete_time_expected_barrier_rho_optimization"
        return "discrete_time_expected_barrier_fixed_epsilon"

    @staticmethod
    def _transition_cell_errors(cell: DiscreteTransitionCell, mode: str) -> np.ndarray:
        if mode == "off":
            return np.zeros_like(cell.total_error_bounds)
        if mode == "fit":
            return cell.fit_error_bounds
        return cell.total_error_bounds
