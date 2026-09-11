from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

from .benchmarks import batch_control_policy


def monomial_exponents(n_vars: int, degree: int) -> list[tuple[int, ...]]:
    exps: list[tuple[int, ...]] = []
    for exp in product(range(degree + 1), repeat=n_vars):
        if sum(exp) <= degree:
            exps.append(tuple(int(v) for v in exp))
    exps.sort(key=lambda e: (sum(e), e))
    return exps


def eval_monomials(x: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    phi = np.ones((x_arr.shape[0], len(exponents)), dtype=float)
    for j, exp in enumerate(exponents):
        for k, power in enumerate(exp):
            if power:
                phi[:, j] *= x_arr[:, k] ** power
    return phi


def eval_gradient_basis(x: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    n_samples, n_vars = x_arr.shape
    grad = np.zeros((n_samples, len(exponents), n_vars), dtype=float)
    for j, exp in enumerate(exponents):
        for var in range(n_vars):
            power = exp[var]
            if power == 0:
                continue
            term = np.full(n_samples, float(power), dtype=float)
            for k, p in enumerate(exp):
                effective = p - 1 if k == var else p
                if effective:
                    term *= x_arr[:, k] ** effective
            grad[:, j, var] = term
    return grad


def eval_hessian_basis(x: np.ndarray, exponents: list[tuple[int, ...]]) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    n_samples, n_vars = x_arr.shape
    hess = np.zeros((n_samples, len(exponents), n_vars, n_vars), dtype=float)
    for j, exp in enumerate(exponents):
        for a in range(n_vars):
            for b in range(n_vars):
                if a == b:
                    coeff = exp[a] * (exp[a] - 1)
                    if coeff == 0:
                        continue
                    reduced = list(exp)
                    reduced[a] -= 2
                else:
                    coeff = exp[a] * exp[b]
                    if coeff == 0:
                        continue
                    reduced = list(exp)
                    reduced[a] -= 1
                    reduced[b] -= 1
                term = np.full(n_samples, float(coeff), dtype=float)
                for k, p in enumerate(reduced):
                    if p:
                        term *= x_arr[:, k] ** p
                hess[:, j, a, b] = term
    return hess


@dataclass
class PolynomialAbstraction:
    exponents: list[tuple[int, ...]]
    residual_coeffs: np.ndarray
    epsilon_r: float
    Gbar: np.ndarray
    gbar: np.ndarray
    dt: float
    source: str = "raw_residual"
    context_count: int = 1
    diffusion_upper_bound_type: str = "empirical_covariance"
    extra_error_margin: float = 0.0
    fit_error: float = 0.0
    partition_boxes: list[tuple[tuple[float, float], ...]] = field(default_factory=list)
    local_residual_coeffs: list[np.ndarray] = field(default_factory=list)
    local_fit_errors: list[float] = field(default_factory=list)
    abstraction_certificate: dict[str, Any] = field(default_factory=dict)

    def residual_drift(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        if not self.partition_boxes or not self.local_residual_coeffs:
            return eval_monomials(x_arr, self.exponents) @ self.residual_coeffs

        out = np.zeros((x_arr.shape[0], self.residual_coeffs.shape[1]), dtype=float)
        assigned = np.zeros(x_arr.shape[0], dtype=bool)
        for box, coeffs in zip(self.partition_boxes, self.local_residual_coeffs):
            mask = _points_in_box(x_arr, box) & ~assigned
            if not np.any(mask):
                continue
            out[mask] = eval_monomials(x_arr[mask], self.exponents) @ coeffs
            assigned[mask] = True
        if np.any(~assigned):
            out[~assigned] = eval_monomials(x_arr[~assigned], self.exponents) @ self.residual_coeffs
        return out

    def drift(self, x: np.ndarray, system: Any) -> np.ndarray:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        u = batch_control_policy(system, x_arr, step=0)
        return system.nominal_derivative(x_arr, u) + self.residual_drift(x_arr)

    def generator(self, barrier_coeffs: np.ndarray, barrier_exponents: list[tuple[int, ...]], x: np.ndarray, system: Any) -> np.ndarray:
        x_arr = np.atleast_2d(np.asarray(x, dtype=float))
        grad_basis = eval_gradient_basis(x_arr, barrier_exponents)
        hess_basis = eval_hessian_basis(x_arr, barrier_exponents)
        grad = np.einsum("kti,t->ki", grad_basis, barrier_coeffs)
        hess = np.einsum("ktij,t->kij", hess_basis, barrier_coeffs)
        drift = self.drift(x_arr, system)
        first_order = np.sum(grad * drift, axis=1)
        second_order = 0.5 * np.einsum("ij,kij->k", self.Gbar, hess)
        return first_order + second_order


def fit_polynomial_abstraction(
    x: np.ndarray,
    residual: np.ndarray,
    dt: float,
    degree: int = 2,
    diffusion_margin: float = 1.2,
) -> PolynomialAbstraction:
    x_arr = np.asarray(x, dtype=float)
    residual_arr = np.asarray(residual, dtype=float)
    target_drift = residual_arr / dt

    centered = residual_arr - residual_arr.mean(axis=0, keepdims=True)
    if centered.shape[0] > 1:
        cov = centered.T @ centered / (centered.shape[0] - 1)
    else:
        cov = np.eye(residual_arr.shape[1]) * 1e-6
    diffusion_covariances = cov[None, :, :] / max(dt, 1e-8)

    return fit_polynomial_abstraction_from_drift(
        x_arr,
        target_drift,
        diffusion_covariances,
        dt=dt,
        degree=degree,
        diffusion_margin=diffusion_margin,
        source="raw_residual",
        context_count=1,
        diffusion_upper_bound_type="empirical_residual_covariance",
    )


def fit_polynomial_abstraction_from_drift(
    x: np.ndarray,
    residual_drift: np.ndarray,
    diffusion_covariances: np.ndarray,
    dt: float,
    degree: int = 2,
    diffusion_margin: float = 1.2,
    source: str = "score_dynamics",
    context_count: int = 1,
    diffusion_upper_bound_type: str = "max_eigenvalue_psd_upper_bound",
    extra_error_margin: float = 0.0,
    diffusion_bound_mode: str = "sample",
    diffusion_confidence_delta: float = 0.05,
    diffusion_sample_count: int = 8,
    analytic_diffusion_eigenvalue_bound: float | None = None,
    analytic_residual_drift_norm_bound: float | None = None,
) -> PolynomialAbstraction:
    x_arr = np.asarray(x, dtype=float)
    drift_arr = np.asarray(residual_drift, dtype=float)
    exponents = monomial_exponents(x_arr.shape[1], degree)
    phi = eval_monomials(x_arr, exponents)
    coeffs, *_ = np.linalg.lstsq(phi, drift_arr, rcond=None)
    pred = phi @ coeffs
    fit_error = float(np.max(np.linalg.norm(pred - drift_arr, axis=1)))
    epsilon_r = float(fit_error + max(0.0, float(extra_error_margin)))

    Gbar = _robust_diffusion_upper_bound(
        np.asarray(diffusion_covariances, dtype=float),
        dim=drift_arr.shape[1],
        margin=diffusion_margin,
        mode=diffusion_bound_mode,
        confidence_delta=diffusion_confidence_delta,
        sample_count=diffusion_sample_count,
        analytic_eigenvalue_bound=analytic_diffusion_eigenvalue_bound,
    )
    gbar = _psd_square_root(Gbar)

    return PolynomialAbstraction(
        exponents=exponents,
        residual_coeffs=coeffs,
        epsilon_r=epsilon_r,
        Gbar=Gbar,
        gbar=gbar,
        dt=dt,
        source=source,
        context_count=int(context_count),
        diffusion_upper_bound_type=diffusion_upper_bound_type,
        extra_error_margin=float(extra_error_margin),
        fit_error=fit_error,
        abstraction_certificate={
            "method": "global_least_squares_probe_fit",
            "fit_points": int(x_arr.shape[0]),
            "fit_error": fit_error,
            "extra_error_margin": float(extra_error_margin),
            "epsilon_r": epsilon_r,
            "diffusion_bound_mode": diffusion_bound_mode,
            "diffusion_confidence_delta": float(diffusion_confidence_delta),
        },
    )


def fit_partitioned_polynomial_abstraction_from_drift(
    x: np.ndarray,
    residual_drift: np.ndarray,
    diffusion_covariances: np.ndarray,
    *,
    box: tuple[tuple[float, float], ...],
    partitions: int | tuple[int, ...],
    dt: float,
    degree: int = 2,
    diffusion_margin: float = 1.2,
    source: str = "score_dynamics",
    context_count: int = 1,
    extra_error_margin: float = 0.0,
    extra_error_margins: list[float] | None = None,
    error_bound_mode: str = "analytic",
    error_lipschitz_margin: float = 1.25,
    diffusion_bound_mode: str = "analytic",
    diffusion_confidence_delta: float = 0.05,
    diffusion_sample_count: int = 8,
    analytic_diffusion_eigenvalue_bound: float | None = None,
    analytic_residual_drift_norm_bound: float | None = None,
    partition_boxes: list[tuple[tuple[float, float], ...]] | None = None,
) -> PolynomialAbstraction:
    """Fit local residual polynomials on partitioned verification-grid data.

    The local models use an ordinary power basis. ``analytic`` mode combines a
    global learned-drift magnitude bound with a box-wise polynomial magnitude
    bound. ``sample`` and ``validation`` remain explicitly empirical modes.
    """

    x_arr = np.asarray(x, dtype=float)
    drift_arr = np.asarray(residual_drift, dtype=float)
    cov_arr = np.asarray(diffusion_covariances, dtype=float)
    exponents = monomial_exponents(x_arr.shape[1], degree)
    phi = eval_monomials(x_arr, exponents)
    coeffs, *_ = np.linalg.lstsq(phi, drift_arr, rcond=None)
    global_pred = phi @ coeffs
    global_fit_error = float(np.max(np.linalg.norm(global_pred - drift_arr, axis=1)))

    boxes = list(partition_boxes) if partition_boxes is not None else partition_box(box, partitions)
    if not boxes:
        raise ValueError("partition_boxes must contain at least one verification cell")
    local_coeffs: list[np.ndarray] = []
    local_errors: list[float] = []
    local_error_bounds: list[float] = []
    cell_reports: list[dict[str, object]] = []
    max_local_error = 0.0
    max_local_error_bound = 0.0
    max_cell_lambda = 0.0
    terms = len(exponents)
    normalized_error_mode = _normalize_error_bound_mode(error_bound_mode)

    for index, cell_box in enumerate(boxes):
        mask = _points_in_box(x_arr, cell_box)
        if int(np.sum(mask)) < terms:
            mask = _nearest_points_to_box(x_arr, cell_box, max(terms, int(np.sum(mask))))
        cell_x = x_arr[mask]
        cell_drift = drift_arr[mask]
        fit_idx, validation_idx = _split_fit_validation_indices(cell_x.shape[0], terms)
        fit_x = cell_x[fit_idx]
        fit_drift = cell_drift[fit_idx]
        fit_phi = eval_monomials(fit_x, exponents)
        cell_coeffs, *_ = np.linalg.lstsq(fit_phi, fit_drift, rcond=None)
        fit_error_vec = fit_phi @ cell_coeffs - fit_drift
        cell_error = float(np.max(np.linalg.norm(fit_error_vec, axis=1))) if fit_x.size else global_fit_error
        validation_x = cell_x[validation_idx]
        validation_drift = cell_drift[validation_idx]
        validation_error = cell_error
        if validation_x.size:
            validation_pred = eval_monomials(validation_x, exponents) @ cell_coeffs
            validation_error = float(np.max(np.linalg.norm(validation_pred - validation_drift, axis=1)))
        cell_error_bound = cell_error
        if normalized_error_mode == "validation":
            cell_error_bound = max(cell_error, validation_error) * max(1.0, float(error_lipschitz_margin))
        elif normalized_error_mode == "analytic":
            if analytic_residual_drift_norm_bound is None or not np.isfinite(analytic_residual_drift_norm_bound):
                raise ValueError("analytic error mode requires a finite analytic_residual_drift_norm_bound")
            polynomial_norm_bound = _polynomial_vector_norm_bound(cell_coeffs, exponents, cell_box)
            cell_error_bound = max(0.0, float(analytic_residual_drift_norm_bound)) + polynomial_norm_bound
        cell_extra_error = max(0.0, float(extra_error_margin))
        if extra_error_margins is not None:
            if len(extra_error_margins) != len(boxes):
                raise ValueError("extra_error_margins length must match partition_boxes")
            cell_extra_error = max(0.0, float(extra_error_margins[index]))
        full_cell_error_bound = cell_error_bound + cell_extra_error
        max_local_error = max(max_local_error, cell_error)
        max_local_error_bound = max(max_local_error_bound, full_cell_error_bound)
        local_coeffs.append(cell_coeffs)
        local_errors.append(cell_error)
        local_error_bounds.append(full_cell_error_bound)

        cell_cov = cov_arr[mask]
        cell_lambda = _max_covariance_eigenvalue(cell_cov, dim=drift_arr.shape[1])
        max_cell_lambda = max(max_cell_lambda, cell_lambda)
        cell_reports.append(
            {
                "index": int(index),
                "box": [[float(lo), float(hi)] for lo, hi in cell_box],
                "fit_points": int(fit_x.shape[0]),
                "validation_points": int(validation_x.shape[0]),
                "fit_error": cell_error,
                "validation_error": validation_error,
                "base_error_envelope": cell_error_bound,
                "extra_error_margin": cell_extra_error,
                "error_envelope": full_cell_error_bound,
                "error_bound_type": normalized_error_mode,
                "max_diffusion_eigenvalue": cell_lambda,
            }
        )

    epsilon_r = float(max_local_error_bound)
    Gbar = _robust_diffusion_upper_bound(
        cov_arr,
        dim=drift_arr.shape[1],
        margin=diffusion_margin,
        mode=diffusion_bound_mode,
        confidence_delta=diffusion_confidence_delta,
        sample_count=diffusion_sample_count,
        analytic_eigenvalue_bound=analytic_diffusion_eigenvalue_bound,
    )
    gbar = _psd_square_root(Gbar)
    normalized_diffusion_mode = _normalize_diffusion_bound_mode(diffusion_bound_mode)
    diffusion_factor = None
    if normalized_diffusion_mode == "inflated_sample":
        diffusion_factor = _covariance_confidence_factor(
            dim=drift_arr.shape[1],
            sample_count=diffusion_sample_count,
            n_conditions=max(1, cov_arr.reshape(-1, drift_arr.shape[1], drift_arr.shape[1]).shape[0]),
            delta=diffusion_confidence_delta,
        )

    partition_shape = _partition_shape_from_boxes(boxes, len(box))
    certificate: dict[str, Any] = {
        "method": "partitioned_grid_least_squares_with_empirical_envelopes",
        "error_bound_mode": normalized_error_mode,
        "validation_error_margin": float(error_lipschitz_margin),
        "box": [[float(lo), float(hi)] for lo, hi in box],
        "partition_shape": list(partition_shape),
        "partition_count": int(len(boxes)),
        "fit_points": int(x_arr.shape[0]),
        "polynomial_degree": int(degree),
        "global_fit_error": global_fit_error,
        "max_local_fit_error": max_local_error,
        "max_local_error_bound": max_local_error_bound,
        "extra_error_margin": float(max(extra_error_margins)) if extra_error_margins else float(extra_error_margin),
        "local_error_bounds_include_extra_margin": True,
        "analytic_residual_drift_norm_bound": analytic_residual_drift_norm_bound,
        "epsilon_r": epsilon_r,
        "max_cell_diffusion_eigenvalue": max_cell_lambda,
        "diffusion_bound_mode": normalized_diffusion_mode,
        "diffusion_bound_is_global_for_learned_sampler": normalized_diffusion_mode == "analytic",
        "analytic_diffusion_eigenvalue_bound": analytic_diffusion_eigenvalue_bound,
        "diffusion_confidence_delta": float(diffusion_confidence_delta),
        "diffusion_sample_count": int(diffusion_sample_count),
        "diffusion_confidence_factor": diffusion_factor,
        "diffusion_margin": float(diffusion_margin),
        "cell_reports": cell_reports,
        "note": (
            "Each local polynomial is assessed on a deterministic holdout subset. "
            "This is an empirical validation envelope, not a formal neural-network "
            "or distributional bound."
        ),
    }

    return PolynomialAbstraction(
        exponents=exponents,
        residual_coeffs=coeffs,
        epsilon_r=epsilon_r,
        Gbar=Gbar,
        gbar=gbar,
        dt=dt,
        source=source,
        context_count=int(context_count),
        diffusion_upper_bound_type=(
            "analytic_learned_sampler_second_moment_upper_bound"
            if normalized_diffusion_mode == "analytic"
            else "partitioned_verification_box_max_eigenvalue_psd_upper_bound"
        ),
        extra_error_margin=float(max(extra_error_margins)) if extra_error_margins else float(extra_error_margin),
        fit_error=max_local_error,
        partition_boxes=boxes,
        local_residual_coeffs=local_coeffs,
        local_fit_errors=local_error_bounds,
        abstraction_certificate=certificate,
    )


def partition_box(
    box: tuple[tuple[float, float], ...],
    partitions: int | tuple[int, ...],
    split_points: tuple[tuple[float, ...], ...] | None = None,
) -> list[tuple[tuple[float, float], ...]]:
    shape = _partition_shape(box, partitions)
    if split_points is not None and len(split_points) != len(box):
        raise ValueError("split_points length must match state dimension")
    axes = []
    for axis, ((lo, hi), count) in enumerate(zip(box, shape)):
        requested = () if split_points is None else tuple(float(v) for v in split_points[axis])
        interior = sorted({v for v in requested if float(lo) < v < float(hi)})
        if interior:
            edges = np.asarray([float(lo), *interior, float(hi)], dtype=float)
        else:
            edges = np.linspace(float(lo), float(hi), int(count) + 1)
        axes.append([(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)])
    return [tuple(cell) for cell in product(*axes)]


def _partition_shape_from_boxes(
    boxes: list[tuple[tuple[float, float], ...]],
    state_dim: int,
) -> tuple[int, ...]:
    return tuple(
        len({(float(cell[axis][0]), float(cell[axis][1])) for cell in boxes})
        for axis in range(state_dim)
    )


def _partition_shape(
    box: tuple[tuple[float, float], ...],
    partitions: int | tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(partitions, tuple):
        if len(partitions) != len(box):
            raise ValueError("partition tuple length must match state dimension")
        return tuple(max(1, int(v)) for v in partitions)
    return tuple(max(1, int(partitions)) for _ in box)


def _points_in_box(x: np.ndarray, box: tuple[tuple[float, float], ...], tol: float = 1e-10) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    mask = np.ones(x_arr.shape[0], dtype=bool)
    for idx, (lo, hi) in enumerate(box):
        mask &= x_arr[:, idx] >= float(lo) - tol
        mask &= x_arr[:, idx] <= float(hi) + tol
    return mask


def _nearest_points_to_box(x: np.ndarray, box: tuple[tuple[float, float], ...], count: int) -> np.ndarray:
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    lows = np.asarray([b[0] for b in box], dtype=float)
    highs = np.asarray([b[1] for b in box], dtype=float)
    clipped = np.minimum(np.maximum(x_arr, lows), highs)
    distance = np.linalg.norm(x_arr - clipped, axis=1)
    order = np.argsort(distance)
    mask = np.zeros(x_arr.shape[0], dtype=bool)
    mask[order[: max(1, min(int(count), x_arr.shape[0]))]] = True
    return mask


def _normalize_error_bound_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"sample", "sample_max", "grid"}:
        return "sample"
    if normalized in {"validation", "holdout"}:
        return "validation"
    if normalized in {"analytic", "global", "global_magnitude"}:
        return "analytic"
    raise ValueError("error_bound_mode must be 'analytic', 'sample', or 'validation'")


def _polynomial_vector_norm_bound(
    coeffs: np.ndarray,
    exponents: list[tuple[int, ...]],
    box: tuple[tuple[float, float], ...],
) -> float:
    """Bound a vector power-basis polynomial on an axis-aligned box."""

    coeff_arr = np.asarray(coeffs, dtype=float)
    max_abs = np.asarray([max(abs(float(lo)), abs(float(hi))) for lo, hi in box], dtype=float)
    component_bounds = np.zeros(coeff_arr.shape[1], dtype=float)
    for term_index, exp in enumerate(exponents):
        monomial_bound = 1.0
        for coordinate, power in enumerate(exp):
            if power:
                monomial_bound *= float(max_abs[coordinate] ** power)
        component_bounds += np.abs(coeff_arr[term_index]) * monomial_bound
    return float(np.linalg.norm(component_bounds, ord=2))


def _split_fit_validation_indices(n_points: int, terms: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically reserve a spatially mixed holdout set per cell."""
    indices = np.arange(n_points, dtype=int)
    if n_points <= terms + 1:
        return indices, indices
    validation_mask = indices % 5 == 0
    if int(np.sum(~validation_mask)) < terms:
        validation_mask[:] = False
        validation_mask[-max(1, n_points // 5) :] = True
    fit_idx = np.flatnonzero(~validation_mask)
    validation_idx = np.flatnonzero(validation_mask)
    return fit_idx, validation_idx if validation_idx.size else fit_idx


def _normalize_diffusion_bound_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"sample", "sample_max", "empirical"}:
        return "sample"
    if normalized in {"inflated_sample", "confidence", "high_confidence", "statistical"}:
        return "inflated_sample"
    if normalized in {"analytic", "analytic_sampler", "global_sampler"}:
        return "analytic"
    raise ValueError("diffusion_bound_mode must be 'analytic', 'sample', or 'inflated_sample'")


def _covariance_confidence_factor(
    *,
    dim: int,
    sample_count: int,
    n_conditions: int,
    delta: float,
) -> float:
    n_eff = max(2, int(sample_count) - 1)
    delta_eff = min(max(float(delta), 1e-12), 0.5)
    union = max(1, int(n_conditions))
    log_term = np.log(float(union) / delta_eff)
    radius = np.sqrt(max(0.0, (float(dim) + log_term) / float(n_eff)))
    return float((1.0 + radius) ** 2)


def _robust_diffusion_upper_bound(
    covariances: np.ndarray,
    dim: int,
    margin: float,
    mode: str = "sample",
    confidence_delta: float = 0.05,
    sample_count: int = 8,
    analytic_eigenvalue_bound: float | None = None,
) -> np.ndarray:
    normalized_mode = _normalize_diffusion_bound_mode(mode)
    if normalized_mode == "analytic":
        if analytic_eigenvalue_bound is None or not np.isfinite(analytic_eigenvalue_bound):
            raise ValueError("analytic diffusion mode requires a finite analytic_eigenvalue_bound")
        if float(analytic_eigenvalue_bound) < 0.0:
            raise ValueError("analytic_eigenvalue_bound must be nonnegative")
        return float(margin) * max(float(analytic_eigenvalue_bound), 1e-10) * np.eye(dim, dtype=float)
    cov_arr = np.asarray(covariances, dtype=float)
    if cov_arr.size == 0:
        return np.eye(dim, dtype=float) * 1e-8
    cov_arr = cov_arr.reshape(-1, dim, dim)
    max_lambda = 0.0
    for cov in cov_arr:
        sym = (cov + cov.T) / 2.0
        eigvals = np.linalg.eigvalsh(sym)
        max_lambda = max(max_lambda, float(np.max(eigvals)))
    factor = 1.0
    if normalized_mode == "inflated_sample":
        factor = _covariance_confidence_factor(
            dim=dim,
            sample_count=sample_count,
            n_conditions=cov_arr.shape[0],
            delta=confidence_delta,
        )
    return float(margin) * float(factor) * max(max_lambda, 1e-10) * np.eye(dim, dtype=float)


def _max_covariance_eigenvalue(covariances: np.ndarray, dim: int) -> float:
    cov_arr = np.asarray(covariances, dtype=float)
    if cov_arr.size == 0:
        return 0.0
    cov_arr = cov_arr.reshape(-1, dim, dim)
    max_lambda = 0.0
    for cov in cov_arr:
        sym = (cov + cov.T) / 2.0
        eigvals = np.linalg.eigvalsh(sym)
        max_lambda = max(max_lambda, float(np.max(eigvals)))
    return max_lambda


def _psd_square_root(matrix: np.ndarray) -> np.ndarray:
    sym = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals))
