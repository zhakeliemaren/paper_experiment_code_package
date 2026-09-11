from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt
from typing import Any

import numpy as np
from scipy.special import ndtr

from .benchmarks import BenchmarkSystem
from .polynomial import eval_monomials, monomial_exponents
from .sos import Poly, affine_substitute_poly, barrier_polynomial, evaluate_poly, vector_polynomial_from_coeffs


Box = tuple[tuple[float, float], ...]


@dataclass
class DiscreteTransitionCell:
    box: Box
    expected_basis_polys: list[Poly]
    fit_error_bounds: np.ndarray
    total_error_bounds: np.ndarray
    fit_points: int
    validation_points: int


@dataclass
class DiscreteTransitionAbstraction:
    """Polynomial abstraction of E[phi(X[k+1]) | X[k]=x].

    The basis ``phi`` is exactly the monomial basis used by the barrier.  This
    retains all estimated moments needed by a polynomial barrier instead of
    reducing the learned transition to only a drift and covariance.
    """

    barrier_exponents: list[tuple[int, ...]]
    cells: list[DiscreteTransitionCell]
    source: str
    transition_sample_count: int
    context_count: int
    confidence_delta: float
    basis_center: np.ndarray
    basis_scale: np.ndarray
    abstraction_certificate: dict[str, Any]

    @property
    def max_fit_error(self) -> float:
        return max((float(np.max(cell.fit_error_bounds)) for cell in self.cells), default=0.0)

    @property
    def max_total_error(self) -> float:
        return max((float(np.max(cell.total_error_bounds)) for cell in self.cells), default=0.0)

    def expected_basis(self, x: np.ndarray) -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        values = np.zeros((rows.shape[0], len(self.barrier_exponents)), dtype=float)
        assigned = np.zeros(rows.shape[0], dtype=bool)
        for cell in self.cells:
            mask = _points_in_box(rows, cell.box) & ~assigned
            if not np.any(mask):
                continue
            values[mask] = np.column_stack([evaluate_poly(poly, rows[mask]) for poly in cell.expected_basis_polys])
            assigned[mask] = True
        if np.any(~assigned):
            cell = self.cells[0]
            values[~assigned] = np.column_stack(
                [evaluate_poly(poly, rows[~assigned]) for poly in cell.expected_basis_polys]
            )
        return values

    def barrier_difference(self, coeffs: np.ndarray, x: np.ndarray, error_mode: str = "total") -> np.ndarray:
        rows = np.atleast_2d(np.asarray(x, dtype=float))
        coeff_arr = np.asarray(coeffs, dtype=float).reshape(-1)
        normalized = (rows - self.basis_center) / self.basis_scale
        values = self.expected_basis(rows) @ coeff_arr - eval_monomials(normalized, self.barrier_exponents) @ coeff_arr
        assigned = np.zeros(rows.shape[0], dtype=bool)
        for cell in self.cells:
            mask = _points_in_box(rows, cell.box) & ~assigned
            if np.any(mask):
                values[mask] += float(_cell_errors(cell, error_mode) @ np.abs(coeff_arr))
                assigned[mask] = True
        if np.any(~assigned):
            values[~assigned] += float(_cell_errors(self.cells[0], error_mode) @ np.abs(coeff_arr))
        return values


class DiffusionPhysicalStepModel:
    """Stateful online model that calls reverse diffusion once per vehicle step."""

    def __init__(
        self,
        diffusion: Any,
        context_encoder: Any,
        system: BenchmarkSystem,
        *,
        history_length: int,
        seed: int = 0,
    ) -> None:
        self.diffusion = diffusion
        self.context_encoder = context_encoder
        self.system = system
        self.history_length = max(1, int(history_length))
        self.seed = int(seed)
        self.step_index = 0
        self._past_pairs: list[np.ndarray] = []

    def reset(self) -> None:
        self.step_index = 0
        self._past_pairs.clear()

    def step(self, x: np.ndarray, u: np.ndarray | None = None) -> np.ndarray:
        state = np.asarray(x, dtype=float).reshape(self.system.state_dim)
        control = (
            np.asarray(u, dtype=float).reshape(self.system.control_dim)
            if u is not None
            else np.asarray(self.system.control_policy(state, step=self.step_index), dtype=float).reshape(self.system.control_dim)
        )
        pair = np.concatenate([state, control])
        available = [*self._past_pairs, pair][-self.history_length :]
        if len(available) < self.history_length:
            available = [available[0]] * (self.history_length - len(available)) + available
        history = np.asarray(available, dtype=float)[None, :, :]
        context = np.asarray(self.context_encoder.transform(history), dtype=float)
        next_state = sample_diffusion_next_states(
            self.diffusion,
            self.system,
            state[None, :],
            control[None, :],
            context,
            n_samples=1,
            seed=self.seed + self.step_index,
            condition_batch_size=1,
        )[0, 0]
        self._past_pairs.append(pair)
        self._past_pairs = self._past_pairs[-self.history_length :]
        self.step_index += 1
        return next_state


def sample_diffusion_next_states(
    diffusion: Any,
    system: BenchmarkSystem,
    x: np.ndarray,
    u: np.ndarray,
    contexts: np.ndarray,
    *,
    n_samples: int,
    seed: int,
    condition_batch_size: int = 128,
) -> np.ndarray:
    """Run one complete reverse chain for every requested physical-step sample."""

    states = np.atleast_2d(np.asarray(x, dtype=float))
    controls = np.atleast_2d(np.asarray(u, dtype=float))
    context_arr = np.atleast_2d(np.asarray(contexts, dtype=float))
    if not (states.shape[0] == controls.shape[0] == context_arr.shape[0]):
        raise ValueError("x, u, and contexts must contain the same number of conditions")
    if int(n_samples) < 1:
        raise ValueError("n_samples must be positive")

    chunks: list[np.ndarray] = []
    for start in range(0, states.shape[0], max(1, int(condition_batch_size))):
        stop = min(states.shape[0], start + max(1, int(condition_batch_size)))
        residual = diffusion.sample(
            states[start:stop],
            controls[start:stop],
            context_arr[start:stop],
            n_samples=int(n_samples),
            deterministic=False,
            seed=int(seed) + start,
        )
        residual = np.asarray(residual, dtype=float)
        physical_state_dim = int(getattr(diffusion, "physical_state_dim", residual.shape[-1]))
        physical_indices = tuple(
            getattr(diffusion, "physical_state_indices", tuple(range(residual.shape[-1])))
        )
        if residual.shape[-1] != physical_state_dim:
            embedded = np.zeros((*residual.shape[:-1], physical_state_dim), dtype=float)
            embedded[..., list(physical_indices)] = residual
            residual = embedded
        nominal = np.asarray(system.nominal_step(states[start:stop], controls[start:stop]), dtype=float)
        candidate = nominal[:, None, :] + residual
        chunks.append(_apply_physical_boundary_semantics(system, states[start:stop], candidate))
    return np.concatenate(chunks, axis=0)


def sample_gaussian_next_states(
    system: BenchmarkSystem,
    x: np.ndarray,
    u: np.ndarray,
    *,
    residual_mean: np.ndarray,
    residual_covariance: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample the Gaussian comparison through the same one-step state update."""

    states = np.atleast_2d(np.asarray(x, dtype=float))
    controls = np.atleast_2d(np.asarray(u, dtype=float))
    mean = np.asarray(residual_mean, dtype=float).reshape(-1)
    covariance = np.asarray(residual_covariance, dtype=float)
    rng = np.random.default_rng(seed)
    residual = rng.multivariate_normal(mean, covariance, size=(states.shape[0], int(n_samples)))
    nominal = np.asarray(system.nominal_step(states, controls), dtype=float)
    candidate = nominal[:, None, :] + residual
    return _apply_physical_boundary_semantics(system, states, candidate)


def fit_discrete_transition_abstraction(
    x: np.ndarray,
    next_states: np.ndarray | None,
    *,
    barrier_degree: int,
    fit_degree: int,
    partition_boxes: list[Box],
    verify_box: Box,
    source: str,
    context_count: int,
    confidence_delta: float = 0.05,
    validation_margin: float = 1.25,
    exact_expectations: bool = False,
    expectation_bound_mode: str = "hoeffding",
    expected_basis_targets: np.ndarray | None = None,
) -> DiscreteTransitionAbstraction:
    """Fit local polynomial maps for the next-state barrier basis.

    The Monte Carlo term uses Hoeffding's inequality because next states are
    clipped to ``verify_box``.  The spatial polynomial envelope is still a
    finite-grid validation envelope, and the certificate metadata says so.
    """

    states = np.atleast_2d(np.asarray(x, dtype=float))
    exponents = monomial_exponents(states.shape[1], int(barrier_degree))
    basis_center, basis_scale = _normalization_for_box(verify_box)
    if expected_basis_targets is not None:
        if not exact_expectations:
            raise ValueError("expected_basis_targets requires exact_expectations=True")
        targets = np.asarray(expected_basis_targets, dtype=float)
        if targets.shape != (states.shape[0], len(exponents)):
            raise ValueError("expected_basis_targets has the wrong shape")
        samples = 0
        basis_samples = np.empty((states.shape[0], 0, len(exponents)), dtype=float)
    else:
        next_arr = np.asarray(next_states, dtype=float)
        if next_arr.ndim != 3 or next_arr.shape[0] != states.shape[0]:
            raise ValueError("next_states must have shape (conditions, samples, state_dim)")
        samples = int(next_arr.shape[1])
        normalized_next = (next_arr - basis_center) / basis_scale
        flattened = normalized_next.reshape(-1, next_arr.shape[-1])
        basis_samples = eval_monomials(flattened, exponents).reshape(states.shape[0], samples, len(exponents))
        targets = basis_samples.mean(axis=1)
    mode = str(expectation_bound_mode).strip().lower().replace("-", "_")
    if mode not in {"hoeffding", "empirical_standard_error"}:
        raise ValueError("expectation_bound_mode must be 'hoeffding' or 'empirical_standard_error'")
    per_condition_mc_error = np.zeros_like(targets)
    global_mc_error = np.zeros(len(exponents), dtype=float)
    if not exact_expectations and mode == "hoeffding":
        union = max(1, states.shape[0] * len(exponents))
        factor = sqrt(log(2.0 * union / max(float(confidence_delta), 1e-12)) / (2.0 * samples))
        normalized_verify_box = tuple(
            (
                (float(lo) - float(basis_center[index])) / float(basis_scale[index]),
                (float(hi) - float(basis_center[index])) / float(basis_scale[index]),
            )
            for index, (lo, hi) in enumerate(verify_box)
        )
        global_mc_error = np.asarray(
            [_monomial_range(exp, normalized_verify_box) * factor for exp in exponents],
            dtype=float,
        )
    elif not exact_expectations:
        union = max(1, states.shape[0] * len(exponents))
        multiplier = sqrt(2.0 * log(2.0 * union / max(float(confidence_delta), 1e-12)))
        sample_std = np.std(basis_samples, axis=1, ddof=1)
        per_condition_mc_error = multiplier * sample_std / sqrt(float(samples))
    global_mc_error[0] = 0.0
    per_condition_mc_error[:, 0] = 0.0

    cells: list[DiscreteTransitionCell] = []
    reports: list[dict[str, Any]] = []
    fit_exponents = monomial_exponents(states.shape[1], int(fit_degree))
    for index, box in enumerate(partition_boxes):
        mask = _points_in_box(states, box)
        cell_x = states[mask]
        cell_y = targets[mask]
        if cell_x.shape[0] < len(fit_exponents):
            raise ValueError(f"transition cell {index} has too few fit conditions")
        fit_idx, validation_idx = _split_fit_validation_indices(cell_x.shape[0], len(fit_exponents), cell_x)
        center, scale = _normalization_for_box(box)
        normalized_x = (cell_x - center) / scale
        phi = eval_monomials(normalized_x, fit_exponents)
        coeffs, *_ = np.linalg.lstsq(phi[fit_idx], cell_y[fit_idx], rcond=None)
        predicted = phi @ coeffs
        fit_error = np.max(np.abs(predicted[fit_idx] - cell_y[fit_idx]), axis=0)
        validation_error = np.max(np.abs(predicted[validation_idx] - cell_y[validation_idx]), axis=0)
        spatial_error = max(1.0, float(validation_margin)) * np.maximum(fit_error, validation_error)
        spatial_error[0] = 0.0
        coeffs[:, 0] = 0.0
        coeffs[0, 0] = 1.0
        normalized_polys = vector_polynomial_from_coeffs(fit_exponents, coeffs)
        raw_polys = [affine_substitute_poly(poly, -center / scale, 1.0 / scale) for poly in normalized_polys]
        if exact_expectations:
            cell_mc_error = np.zeros(len(exponents), dtype=float)
        elif mode == "hoeffding":
            cell_mc_error = global_mc_error
        else:
            cell_mc_error = np.max(per_condition_mc_error[mask], axis=0)
        total_error = spatial_error + cell_mc_error
        cells.append(
            DiscreteTransitionCell(
                box=box,
                expected_basis_polys=raw_polys,
                fit_error_bounds=spatial_error,
                total_error_bounds=total_error,
                fit_points=int(fit_idx.size),
                validation_points=int(validation_idx.size),
            )
        )
        reports.append(
            {
                "index": index,
                "box": [[float(lo), float(hi)] for lo, hi in box],
                "fit_points": int(fit_idx.size),
                "validation_points": int(validation_idx.size),
                "max_spatial_error": float(np.max(spatial_error)),
                "max_expectation_sampling_error": float(np.max(cell_mc_error)),
                "max_total_error": float(np.max(total_error)),
            }
        )

    certificate = {
        "method": "conditional_expected_barrier_basis_polynomial_fit",
        "dynamics_semantics": "discrete_time_physical_step",
        "source": source,
        "barrier_degree": int(barrier_degree),
        "transition_polynomial_degree": int(fit_degree),
        "transition_sample_count": samples,
        "context_count": int(context_count),
        "confidence_delta": float(confidence_delta),
        "barrier_basis": "monomials_of_normalized_state",
        "basis_center": basis_center.tolist(),
        "basis_scale": basis_scale.tolist(),
        "monte_carlo_bound": (
            "none_exact_expectations"
            if exact_expectations
            else (
                "finite_condition_hoeffding_union_bound"
                if mode == "hoeffding"
                else "empirical_standard_error_union_indicator"
            )
        ),
        "spatial_bound": "finite_probe_holdout_envelope",
        "global_distribution_certificate": False,
        "cell_reports": reports,
        "note": "Conditional basis moments are fitted on probe states. Spatial generalization remains empirical between probe states.",
    }
    return DiscreteTransitionAbstraction(
        barrier_exponents=exponents,
        cells=cells,
        source=source,
        transition_sample_count=samples,
        context_count=int(context_count),
        confidence_delta=float(confidence_delta),
        basis_center=basis_center,
        basis_scale=basis_scale,
        abstraction_certificate=certificate,
    )


def exact_carla_gaussian_expected_basis(
    system: BenchmarkSystem,
    x: np.ndarray,
    u: np.ndarray,
    *,
    residual_mean: np.ndarray,
    residual_covariance: np.ndarray,
    barrier_degree: int,
    verify_box: Box,
) -> np.ndarray:
    """Evaluate CARLA's degree-two Gaussian one-step moments analytically.

    The calculation includes distance clipping, the absorbing stop rule, and
    upper speed clipping. It removes transition Monte Carlo error, while the
    later state-to-polynomial fit remains an empirical spatial abstraction.
    """

    if int(barrier_degree) > 2:
        raise ValueError("analytic CARLA Gaussian moments currently support barrier degree <= 2")
    states = np.atleast_2d(np.asarray(x, dtype=float))
    controls = np.atleast_2d(np.asarray(u, dtype=float))
    mean = np.asarray(residual_mean, dtype=float).reshape(-1)
    covariance = np.asarray(residual_covariance, dtype=float)
    if states.shape[1] != 2 or mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("analytic CARLA Gaussian moments require a two-state Gaussian residual")
    off_diagonal = covariance - np.diag(np.diag(covariance))
    if np.max(np.abs(off_diagonal)) > 1e-12:
        raise ValueError("analytic CARLA Gaussian moments require independent residual coordinates")

    exponents = monomial_exponents(2, int(barrier_degree))
    center, scale = _normalization_for_box(verify_box)
    nominal = np.asarray(system.nominal_step(states, controls), dtype=float) + mean
    sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    d_mean, d_second = _clipped_normal_moments(
        nominal[:, 0],
        float(sigma[0]),
        float(verify_box[0][0]),
        float(verify_box[0][1]),
    )
    stop_speed = float(getattr(system, "stop_speed", verify_box[1][0]))
    v_mean, v_second = _stopped_upper_clipped_normal_moments(
        nominal[:, 1],
        float(sigma[1]),
        stop_speed,
        float(verify_box[1][1]),
    )

    inactive = np.zeros(states.shape[0], dtype=bool)
    if bool(getattr(system, "absorbing_stop", False)):
        inactive = states[:, 1] <= stop_speed
        d_mean[inactive] = states[inactive, 0]
        d_second[inactive] = states[inactive, 0] ** 2
        v_mean[inactive] = states[inactive, 1]
        v_second[inactive] = states[inactive, 1] ** 2

    zd_mean = (d_mean - center[0]) / scale[0]
    zv_mean = (v_mean - center[1]) / scale[1]
    zd_second = (d_second - 2.0 * center[0] * d_mean + center[0] ** 2) / scale[0] ** 2
    zv_second = (v_second - 2.0 * center[1] * v_mean + center[1] ** 2) / scale[1] ** 2
    zd_zv = zd_mean * zv_mean
    if np.any(inactive):
        zd_zv[inactive] = (
            (states[inactive, 0] - center[0])
            * (states[inactive, 1] - center[1])
            / (scale[0] * scale[1])
        )

    lookup = {
        (0, 0): np.ones(states.shape[0], dtype=float),
        (1, 0): zd_mean,
        (0, 1): zv_mean,
        (2, 0): zd_second,
        (1, 1): zd_zv,
        (0, 2): zv_second,
    }
    return np.column_stack([lookup[exponent] for exponent in exponents])


def _standard_normal_pdf(value: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * np.asarray(value, dtype=float) ** 2) / sqrt(2.0 * np.pi)


def _normal_interval_raw_moments(
    mean: np.ndarray,
    sigma: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.asarray(mean, dtype=float)
    if sigma <= 1e-15:
        inside = (mu > lower) & (mu < upper)
        probability = inside.astype(float)
        return probability, np.where(inside, mu, 0.0), np.where(inside, mu**2, 0.0)
    a = (lower - mu) / sigma
    b = (upper - mu) / sigma
    phi_a = _standard_normal_pdf(a)
    phi_b = _standard_normal_pdf(b)
    probability = ndtr(b) - ndtr(a)
    first = mu * probability + sigma * (phi_a - phi_b)
    second = (
        (mu**2 + sigma**2) * probability
        + 2.0 * mu * sigma * (phi_a - phi_b)
        + sigma**2 * (a * phi_a - b * phi_b)
    )
    return probability, first, second


def _clipped_normal_moments(
    mean: np.ndarray,
    sigma: float,
    lower: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mean, dtype=float)
    if sigma <= 1e-15:
        clipped = np.clip(mu, lower, upper)
        return clipped, clipped**2
    probability, first, second = _normal_interval_raw_moments(mu, sigma, lower, upper)
    del probability
    lower_probability = ndtr((lower - mu) / sigma)
    upper_probability = 1.0 - ndtr((upper - mu) / sigma)
    return (
        lower * lower_probability + first + upper * upper_probability,
        lower**2 * lower_probability + second + upper**2 * upper_probability,
    )


def _stopped_upper_clipped_normal_moments(
    mean: np.ndarray,
    sigma: float,
    stop_speed: float,
    upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.asarray(mean, dtype=float)
    if sigma <= 1e-15:
        transformed = np.where(mu <= stop_speed, 0.0, np.minimum(mu, upper))
        return transformed, transformed**2
    probability, first, second = _normal_interval_raw_moments(mu, sigma, stop_speed, upper)
    del probability
    upper_probability = 1.0 - ndtr((upper - mu) / sigma)
    return first + upper * upper_probability, second + upper**2 * upper_probability


def _apply_physical_boundary_semantics(
    system: BenchmarkSystem,
    current: np.ndarray,
    candidate: np.ndarray,
) -> np.ndarray:
    result = np.asarray(candidate, dtype=float).copy()
    if bool(getattr(system, "absorbing_stop", False)) and result.shape[-1] >= 2:
        stop_speed = float(getattr(system, "stop_speed", 0.0))
        active = np.asarray(current, dtype=float)[:, 1] > stop_speed
        result = np.where(active[:, None, None], result, np.asarray(current, dtype=float)[:, None, :])
        entered = active[:, None] & (result[:, :, 1] <= stop_speed)
        result[:, :, 1] = np.where(entered, 0.0, result[:, :, 1])
    for coordinate, (lo, hi) in enumerate(system.verify_box):
        result[..., coordinate] = np.clip(result[..., coordinate], float(lo), float(hi))
    return result


def _cell_errors(cell: DiscreteTransitionCell, mode: str) -> np.ndarray:
    normalized = str(mode).strip().lower()
    if normalized == "off":
        return np.zeros_like(cell.total_error_bounds)
    if normalized == "fit":
        return cell.fit_error_bounds
    if normalized == "total":
        return cell.total_error_bounds
    raise ValueError("error_mode must be 'off', 'fit', or 'total'")


def _normalization_for_box(box: Box) -> tuple[np.ndarray, np.ndarray]:
    lows = np.asarray([lo for lo, _ in box], dtype=float)
    highs = np.asarray([hi for _, hi in box], dtype=float)
    return (lows + highs) / 2.0, np.maximum((highs - lows) / 2.0, 1e-12)


def _split_fit_validation_indices(
    n_points: int,
    terms: int,
    points: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(n_points, dtype=int)
    if n_points <= terms + 1:
        return indices, indices
    if points is None:
        group_ids = indices
    else:
        _, group_ids = np.unique(np.round(np.asarray(points, dtype=float), decimals=12), axis=0, return_inverse=True)
    validation = group_ids % 5 == 0
    if int(np.sum(~validation)) < terms:
        validation[:] = False
        validation[-max(1, n_points // 5) :] = True
    fit_idx = np.flatnonzero(~validation)
    validation_idx = np.flatnonzero(validation)
    return fit_idx, validation_idx if validation_idx.size else fit_idx


def _points_in_box(x: np.ndarray, box: Box, tol: float = 1e-10) -> np.ndarray:
    rows = np.atleast_2d(np.asarray(x, dtype=float))
    mask = np.ones(rows.shape[0], dtype=bool)
    for coordinate, (lo, hi) in enumerate(box):
        mask &= rows[:, coordinate] >= float(lo) - tol
        mask &= rows[:, coordinate] <= float(hi) + tol
    return mask


def _monomial_range(exp: tuple[int, ...], box: Box) -> float:
    lower, upper = 1.0, 1.0
    for power, (lo, hi) in zip(exp, box):
        if power == 0:
            continue
        values = [float(lo) ** power, float(hi) ** power]
        if power % 2 == 0 and float(lo) <= 0.0 <= float(hi):
            values.append(0.0)
        a, b = min(values), max(values)
        candidates = [lower * a, lower * b, upper * a, upper * b]
        lower, upper = min(candidates), max(candidates)
    return max(0.0, upper - lower)
