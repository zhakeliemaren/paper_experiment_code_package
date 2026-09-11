from __future__ import annotations

from typing import Any

import numpy as np


def maximum_order_statistic_pac_bound(
    values: np.ndarray,
    confidence_delta: float,
) -> dict[str, float | int | str]:
    """Return a distribution-free one-sided tolerance bound.

    For ``n`` IID calibration values and their maximum ``q``, the maximum
    order-statistic theorem gives

        P_data(P_Z(value(Z) > q) > epsilon) <= delta,

    with ``epsilon = 1 - delta ** (1 / n)``.  The guarantee is distributional
    for a fresh condition from the calibration distribution; it is not a
    deterministic supremum over a continuous box.
    """

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size < 1:
        raise ValueError("PAC calibration requires at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError("PAC calibration values must be finite")
    delta = float(confidence_delta)
    if not 0.0 < delta < 1.0:
        raise ValueError("confidence_delta must lie strictly between zero and one")
    violation_probability = float(1.0 - np.exp(np.log(delta) / float(arr.size)))
    return {
        "threshold": float(np.max(arr)),
        "sample_count": int(arr.size),
        "confidence_delta": delta,
        "confidence": float(1.0 - delta),
        "violation_probability_upper_bound": violation_probability,
        "theorem": "maximum_order_statistic_distribution_free_tolerance_bound",
    }


def certify_diffusion_context_lipschitz(diffusion: Any, context_radius: float, dt: float) -> dict[str, object]:
    """Certify a conservative context sensitivity bound for the current MLP diffusion model.

    The current diffusion implementation predicts noise with
    StandardScaler -> sklearn.MLPRegressor.  This routine uses products of
    spectral norms to upper-bound the network Lipschitz constant with respect
    to the context slice, then propagates that bound through the reverse sampler
    used by ConditionalResidualDiffusion.sample.
    """

    context_start = int(diffusion.state_dim) + int(diffusion.control_dim)
    context_stop = context_start + int(diffusion.context_dim)
    residual_start = context_stop
    residual_stop = residual_start + int(diffusion.residual_dim)

    context_slice = slice(context_start, context_stop)
    residual_slice = slice(residual_start, residual_stop)

    noise_context_lipschitz = mlp_pipeline_lipschitz(diffusion.model, input_slice=context_slice)
    noise_residual_lipschitz = mlp_pipeline_lipschitz(diffusion.model, input_slice=residual_slice)
    normalized_residual_lipschitz, reverse_steps = reverse_sampler_context_lipschitz(
        diffusion,
        noise_context_lipschitz=noise_context_lipschitz,
        noise_residual_lipschitz=noise_residual_lipschitz,
    )
    residual_scale = np.asarray(getattr(diffusion, "residual_scale", np.ones(diffusion.residual_dim)), dtype=float)
    residual_lipschitz = float(np.max(np.abs(residual_scale))) * normalized_residual_lipschitz
    drift_lipschitz = residual_lipschitz / max(float(dt), 1e-12)
    margin = drift_lipschitz * max(0.0, float(context_radius))

    return {
        "method": "spectral_norm_lipschitz",
        "target": "conditional_residual_diffusion_context_sensitivity",
        "model_type": type(diffusion.model).__name__,
        "activation": str(getattr(diffusion.model.named_steps["mlp"], "activation", "unknown")),
        "feature_layout": ["x", "u", "c", "residual_y_t", "t", "sin(pi*t)", "cos(pi*t)"],
        "context_slice": [context_start, context_stop],
        "residual_slice": [residual_start, residual_stop],
        "noise_context_lipschitz": float(noise_context_lipschitz),
        "noise_residual_lipschitz": float(noise_residual_lipschitz),
        "residual_normalization_scale": residual_scale.tolist(),
        "reverse_sampler_normalized_residual_lipschitz": float(normalized_residual_lipschitz),
        "reverse_sampler_residual_lipschitz": float(residual_lipschitz),
        "drift_lipschitz": float(drift_lipschitz),
        "context_radius": float(context_radius),
        "drift_error_margin": float(margin),
        "reverse_sampler_formula": "stochastic_reverse_update_used_by_ConditionalResidualDiffusion.sample",
        "reverse_step_count": len(reverse_steps),
        "reverse_step_lipschitz": reverse_steps,
    }


def certify_diffusion_sampler_covariance(diffusion: Any, dt: float) -> dict[str, object]:
    """Bound the covariance of the implemented reverse sampler for every condition.

    The current noise predictor has tanh hidden layers and a linear output, so
    its output is uniformly bounded independently of ``x``, ``u``, ``c`` and
    ``y_t``. Minkowski's inequality then propagates a second-moment bound from
    the Gaussian prior and every posterior-noise injection to the terminal
    normalized residual. This certifies the learned sampler, not the unknown
    real residual distribution.
    """

    eps_l2_bound = mlp_tanh_output_l2_bound(diffusion.model)
    dim = int(diffusion.residual_dim)
    root_second_moment = float(np.sqrt(dim))
    condition_stop = int(diffusion.state_dim) + int(diffusion.control_dim) + int(diffusion.context_dim)
    residual_slice = slice(condition_stop, condition_stop + dim)
    eps_residual_lipschitz = mlp_pipeline_lipschitz(diffusion.model, input_slice=residual_slice)
    gaussian_input_lipschitz = 1.0
    steps: list[dict[str, float]] = []

    for t in range(int(diffusion.schedule.timesteps) - 1, -1, -1):
        beta = float(diffusion.schedule.betas[t])
        alpha = float(diffusion.schedule.alphas[t])
        abar = float(diffusion.schedule.alpha_bars[t])
        state_gain = 1.0 / np.sqrt(max(alpha, 1e-12))
        noise_predictor_gain = beta / np.sqrt(max(alpha * (1.0 - abar), 1e-12))
        posterior_std = 0.0
        if t > 0:
            abar_prev = float(diffusion.schedule.alpha_bars[t - 1])
            posterior_variance = beta * (1.0 - abar_prev) / max(1.0 - abar, 1e-12)
            posterior_std = float(np.sqrt(max(posterior_variance, 0.0)))
        root_second_moment = (
            abs(state_gain) * root_second_moment
            + abs(noise_predictor_gain) * eps_l2_bound
            + posterior_std * np.sqrt(dim)
        )
        state_jacobian_bound = abs(state_gain) + abs(noise_predictor_gain) * eps_residual_lipschitz
        gaussian_input_lipschitz = float(
            np.sqrt((state_jacobian_bound * gaussian_input_lipschitz) ** 2 + posterior_std**2)
        )
        steps.append(
            {
                "t_index": int(t),
                "state_gain": float(state_gain),
                "noise_predictor_gain": float(noise_predictor_gain),
                "posterior_std": posterior_std,
                "root_second_moment": float(root_second_moment),
                "state_jacobian_bound": float(state_jacobian_bound),
                "gaussian_input_lipschitz": float(gaussian_input_lipschitz),
            }
        )

    residual_scale = np.asarray(diffusion.residual_scale, dtype=float)
    physical_root_second_moment = float(np.max(np.abs(residual_scale))) * root_second_moment
    scale_operator_norm = float(np.max(np.abs(residual_scale)))
    second_moment_covariance_bound = physical_root_second_moment**2 / max(float(dt), 1e-12)
    poincare_covariance_bound = (scale_operator_norm * gaussian_input_lipschitz) ** 2 / max(float(dt), 1e-12)
    covariance_eigenvalue_bound = min(second_moment_covariance_bound, poincare_covariance_bound)
    return {
        "method": "analytic_reverse_sampler_second_moment",
        "scope": "all_condition_inputs_for_the_fitted_tanh_sampler",
        "model_target": "learned_terminal_residual_sampler_not_real_system_distribution",
        "noise_predictor_l2_bound": float(eps_l2_bound),
        "noise_predictor_residual_lipschitz": float(eps_residual_lipschitz),
        "normalized_root_second_moment_bound": float(root_second_moment),
        "residual_scale_operator_norm": scale_operator_norm,
        "physical_root_second_moment_bound": physical_root_second_moment,
        "second_moment_covariance_eigenvalue_bound": float(second_moment_covariance_bound),
        "gaussian_input_lipschitz_bound": float(gaussian_input_lipschitz),
        "gaussian_poincare_covariance_eigenvalue_bound": float(poincare_covariance_bound),
        "physical_sde_covariance_eigenvalue_bound": float(covariance_eigenvalue_bound),
        "reverse_steps": steps,
    }


def mlp_tanh_output_l2_bound(model: Any) -> float:
    """Return a global L2 output bound for a fitted tanh-hidden sklearn MLP."""

    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline with named_steps.")
    mlp = model.named_steps.get("mlp")
    if mlp is None or not hasattr(mlp, "coefs_"):
        raise RuntimeError("A fitted MLPRegressor is required")
    if str(getattr(mlp, "activation", "")).lower() != "tanh":
        raise ValueError("Analytic sampler covariance certification requires tanh hidden activations")
    if len(mlp.coefs_) < 2:
        raise ValueError("Analytic sampler covariance certification requires at least one bounded hidden layer")
    output_weights = np.asarray(mlp.coefs_[-1], dtype=float)
    output_bias = np.asarray(mlp.intercepts_[-1], dtype=float)
    component_bounds = np.sum(np.abs(output_weights), axis=0) + np.abs(output_bias)
    return float(np.linalg.norm(component_bounds, ord=2))


def reverse_sampler_context_lipschitz(
    diffusion: Any,
    noise_context_lipschitz: float,
    noise_residual_lipschitz: float,
) -> tuple[float, list[dict[str, float]]]:
    """Propagate context Lipschitz constants through the implemented sampler."""

    alpha_bars = np.asarray(diffusion.schedule.alpha_bars, dtype=float)
    alphas = np.asarray(diffusion.schedule.alphas, dtype=float)
    y_lipschitz = 0.0
    steps: list[dict[str, float]] = []

    for t in range(int(diffusion.schedule.timesteps) - 1, -1, -1):
        abar = max(float(alpha_bars[t]), 1e-12)
        alpha = max(float(alphas[t]), 1e-12)
        beta = 1.0 - alpha
        sqrt_one_minus_abar = float(np.sqrt(max(1e-12, 1.0 - abar)))

        eps_lipschitz = float(noise_context_lipschitz) + float(noise_residual_lipschitz) * y_lipschitz
        # y_{t-1} = (y_t - beta_t eps_theta / sqrt(1-abar_t)) / sqrt(alpha_t)
        # The posterior noise is independent of context and does not affect the
        # context Lipschitz recurrence.
        a = float(1.0 / np.sqrt(alpha))
        b = float(beta / (np.sqrt(alpha) * sqrt_one_minus_abar))

        y_lipschitz = abs(a) * y_lipschitz + abs(b) * eps_lipschitz
        steps.append(
            {
                "t_index": int(t),
                "state_gain": float(abs(a)),
                "noise_gain": float(abs(b)),
                "eps_lipschitz": float(eps_lipschitz),
                "residual_lipschitz_after_step": float(y_lipschitz),
            }
        )

    return float(y_lipschitz), steps


def mlp_pipeline_lipschitz(model: Any, input_slice: slice | None = None) -> float:
    """Upper-bound StandardScaler -> MLPRegressor Lipschitz constant."""

    if not hasattr(model, "named_steps"):
        raise TypeError("Expected a sklearn Pipeline with named_steps.")
    scaler = model.named_steps.get("scale")
    mlp = model.named_steps.get("mlp")
    if scaler is None or mlp is None:
        raise TypeError("Expected Pipeline steps named 'scale' and 'mlp'.")
    if not hasattr(mlp, "coefs_"):
        raise RuntimeError("MLPRegressor must be fitted before certification.")

    coefs = [np.asarray(w, dtype=float) for w in mlp.coefs_]
    if not coefs:
        return 0.0

    scale = np.asarray(getattr(scaler, "scale_", np.ones(coefs[0].shape[0])), dtype=float)
    scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    first = coefs[0] / scale[:, None]
    if input_slice is not None:
        first = first[input_slice, :]

    activation_lip = activation_lipschitz(str(getattr(mlp, "activation", "relu")))
    lip = _spectral_norm(first)
    if len(coefs) > 1:
        lip *= activation_lip
    for layer_index, weights in enumerate(coefs[1:], start=1):
        lip *= _spectral_norm(weights)
        if layer_index < len(coefs) - 1:
            lip *= activation_lip
    return float(lip)


def activation_lipschitz(name: str) -> float:
    normalized = name.strip().lower()
    if normalized in {"identity", "relu", "tanh"}:
        return 1.0
    if normalized == "logistic":
        return 0.25
    return 1.0


def _spectral_norm(matrix: np.ndarray) -> float:
    arr = np.asarray(matrix, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr, ord=2))
