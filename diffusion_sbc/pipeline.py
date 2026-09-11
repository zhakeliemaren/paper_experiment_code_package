from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from .barrier import MosekSampledBarrierSynthesizer, SampledBarrierSynthesizer, SosBarrierSynthesizer, mosek_sos_available
from .benchmarks import BenchmarkSystem, build_configured_benchmark_system, build_samples, generate_trajectories
from .benchmarks import batch_control_policy
from .certification import (
    certify_diffusion_context_lipschitz,
    certify_diffusion_sampler_covariance,
    maximum_order_statistic_pac_bound,
)
from .config import ExperimentConfig, save_config
from .context import HistoryContextEncoder
from .diffusion import ConditionalResidualDiffusion, DiffusionSchedule
from .polynomial import fit_partitioned_polynomial_abstraction_from_drift, partition_box
from .score_dynamics import ConditionalMomentMatchedResidualSDE
from .utils import ensure_dir, grouped_train_test_split_indices, rng_from_seed, save_json, save_pickle, train_test_split_indices


@dataclass
class ContextSet:
    mode: str
    lower: np.ndarray
    upper: np.ndarray
    center: np.ndarray
    coverage_points: np.ndarray
    margin_fraction: float
    coverage_radius: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
            "center": self.center.tolist(),
            "coverage_points": self.coverage_points.tolist(),
            "coverage_count": int(self.coverage_points.shape[0]),
            "margin_fraction": float(self.margin_fraction),
            "coverage_radius": float(self.coverage_radius),
        }


def _normalize_diffusion_physical_role(value: str) -> str:
    role = str(value).strip().lower().replace("-", "_")
    if role not in {"diffusion_only", "residual_sde"}:
        raise ValueError("diffusion_physical_role must be 'diffusion_only' or 'residual_sde'")
    return role


def _restrict_abstraction_diffusion_support(
    abstraction,
    state_indices: tuple[int, ...],
    *,
    state_dim: int,
) -> None:
    """Project a covariance bound onto the known physical noise subspace."""

    projector = np.zeros((int(state_dim), int(state_dim)), dtype=float)
    projector[list(state_indices), list(state_indices)] = 1.0
    covariance = projector @ np.asarray(abstraction.Gbar, dtype=float) @ projector
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    abstraction.Gbar = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    abstraction.gbar = (eigenvectors * np.sqrt(eigenvalues)[None, :]) @ eigenvectors.T
    certificate = dict(getattr(abstraction, "abstraction_certificate", {}) or {})
    certificate.update(
        {
            "physical_stochastic_state_indices": list(state_indices),
            "physical_noise_support_is_known": True,
            "diffusion_covariance_projected_to_physical_noise_support": True,
        }
    )
    abstraction.abstraction_certificate = certificate


def generate_dataset(config: ExperimentConfig, out_npz: str | Path | None = None) -> dict[str, np.ndarray]:
    system = build_configured_benchmark_system(config)
    trajectories = generate_trajectories(system, config.n_trajectories, config.horizon, config.seed)
    samples = build_samples(trajectories, system, config.history)
    if out_npz is not None:
        p = Path(out_npz)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, **samples)
    return samples


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(path), allow_pickle=False)
    return {k: np.asarray(data[k]) for k in data.files}


def train_pipeline(
    samples: dict[str, np.ndarray],
    config: ExperimentConfig,
    out_dir: str | Path,
    dataset_path: str | Path | None = None,
    save_artifacts: bool = True,
    system: BenchmarkSystem | None = None,
) -> dict[str, object]:
    start = time.time()
    out = ensure_dir(out_dir)
    model_dir = ensure_dir(out / "model") if save_artifacts else None
    rng = rng_from_seed(config.seed)
    system = system or build_configured_benchmark_system(config)
    diffusion_physical_role = _normalize_diffusion_physical_role(config.diffusion_physical_role)
    stochastic_state_indices = tuple(int(index) for index in system.stochastic_state_indices())
    if not stochastic_state_indices:
        raise ValueError("at least one stochastic state coordinate is required")
    stochastic_increments = np.asarray(
        samples.get("stochastic_increment", samples["residual"]),
        dtype=float,
    )
    if "trajectory_id" in samples:
        train_idx, test_idx = grouped_train_test_split_indices(samples["trajectory_id"], 0.2, rng)
        split_scope = "trajectory"
    else:
        train_idx, test_idx = train_test_split_indices(samples["x"].shape[0], 0.2, rng)
        split_scope = "transition_legacy_dataset"

    encoder = HistoryContextEncoder(
        context_dim=config.context_dim,
        mode=config.encoder,
        random_state=config.seed,
        hidden_layer_sizes=config.context_mlp_hidden,
        max_iter=config.context_mlp_max_iter,
    )
    c_train = encoder.fit_transform(samples["history"][train_idx])
    c_all = encoder.transform(samples["history"])
    c_test = c_all[test_idx]

    schedule = DiffusionSchedule(
        timesteps=config.diffusion_steps,
        schedule_type=config.diffusion_schedule,
        beta_start=config.diffusion_beta_start,
        beta_end=config.diffusion_beta_end,
        max_beta=config.diffusion_max_beta,
        terminal_alpha_bar_max=config.diffusion_terminal_alpha_bar_max,
    )
    diffusion = ConditionalResidualDiffusion(
        state_dim=config.state_dim,
        control_dim=config.control_dim,
        context_dim=config.context_dim,
        residual_dim=len(stochastic_state_indices),
        schedule=schedule,
        hidden_layer_sizes=config.diffusion_hidden,
        max_iter=config.diffusion_max_iter,
        batch_size=config.diffusion_batch_size,
        learning_rate_init=config.diffusion_learning_rate,
        early_stopping=config.diffusion_early_stopping,
        random_state=config.seed,
    )
    diffusion_x = np.asarray(samples["x"][train_idx], dtype=float)
    diffusion_u = np.asarray(samples["u"][train_idx], dtype=float)
    diffusion_c = np.asarray(c_train, dtype=float)
    diffusion_targets = np.asarray(
        stochastic_increments[train_idx][:, stochastic_state_indices],
        dtype=float,
    )
    diffusion.physical_state_dim = int(config.state_dim)
    diffusion.physical_state_indices = stochastic_state_indices
    increments_symmetrized = bool(
        diffusion_physical_role == "diffusion_only" and config.diffusion_symmetrize_increments
    )
    if increments_symmetrized:
        # A Brownian increment g(x,c)dW is conditionally zero-mean and centrally
        # symmetric. Pairing eta with -eta prevents finite-data mean leakage
        # from being interpreted as a physical residual drift.
        diffusion_x = np.concatenate([diffusion_x, diffusion_x], axis=0)
        diffusion_u = np.concatenate([diffusion_u, diffusion_u], axis=0)
        diffusion_c = np.concatenate([diffusion_c, diffusion_c], axis=0)
        diffusion_targets = np.concatenate([diffusion_targets, -diffusion_targets], axis=0)
    diffusion_stats = diffusion.fit(
        diffusion_x,
        diffusion_u,
        diffusion_c,
        diffusion_targets,
        copies=config.diffusion_train_copies,
    )
    residual_dynamics = ConditionalMomentMatchedResidualSDE(
        diffusion,
        dt=config.dt,
        n_samples=config.score_residual_samples,
        random_state=config.seed,
        moment_method=config.diffusion_moment_method,
        polynomial_degree=config.reverse_polynomial_degree,
        fit_radius=config.reverse_polynomial_fit_radius,
        validation_radius=config.reverse_polynomial_validation_radius,
        validation_margin=config.reverse_polynomial_validation_margin,
        physical_state_dim=config.state_dim,
        physical_state_indices=stochastic_state_indices,
    )
    eval_count = min(96, len(test_idx))
    eval_idx = test_idx[:eval_count]
    eval_context = c_all[eval_idx]
    eval_moments = residual_dynamics.physical_sde_moments(
        samples["x"][eval_idx],
        samples["u"][eval_idx],
        eval_context,
        n_samples=min(8, config.residual_samples_for_mean),
        seed=config.seed + 10,
    )
    residual_mean = eval_moments.residual_mean
    residual_mae = float(np.mean(np.abs(residual_mean - stochastic_increments[eval_idx])))
    learned_increment_mean_norm = float(np.max(np.linalg.norm(residual_mean, axis=1), initial=0.0))

    context_certification = _normalize_context_certification(config.context_certification)
    if context_certification == "empirical":
        # Empirical runs evaluate representative contexts selected from the
        # observed training support. A box corner in a high-dimensional latent
        # context space is usually not an observed trajectory history and can
        # dominate the residual envelope.
        context_refs = _select_context_references(c_train, config.score_context_bound_samples, rng)
        context_set = _build_empirical_context_set(c_train, context_refs)
    else:
        context_set = _build_context_set(c_train, config, rng)
        context_refs = context_set.coverage_points
    abstraction_box = system.barrier_box if system.barrier_box is not None else system.verify_box
    partition_boxes = partition_box(
        abstraction_box,
        int(config.abstraction_partitions),
        split_points=system.abstraction_partition_breakpoints(),
    )
    abstraction_states = _partitioned_grid_probe_points(
        abstraction_box,
        degree=int(config.polynomial_degree),
        partitions=int(config.abstraction_partitions),
        validation_grid=int(config.abstraction_validation_grid),
        random_count=max(int(config.score_poly_samples), 0),
        rng=rng,
        partition_boxes=partition_boxes,
    )
    abstraction_controls = batch_control_policy(system, abstraction_states, step=0)
    moment_x, moment_u, moment_c = _expand_over_contexts(
        abstraction_states,
        abstraction_controls,
        context_refs,
    )
    moment_samples = residual_dynamics.physical_sde_moments(
        moment_x,
        moment_u,
        moment_c,
        n_samples=config.score_residual_samples,
        seed=config.seed + 20,
    )
    abstraction_residual_drift = (
        np.zeros_like(moment_samples.residual_drift)
        if diffusion_physical_role == "diffusion_only"
        else moment_samples.residual_drift
    )
    context_lipschitz_empirical = _estimate_context_lipschitz(
        abstraction_residual_drift,
        context_refs,
        n_states=len(abstraction_states),
    )
    context_error_margin_empirical = context_lipschitz_empirical * context_set.coverage_radius
    context_certification_report: dict[str, object]
    context_error_margin: float
    context_error_margin_source: str
    if diffusion_physical_role == "diffusion_only":
        context_certification_report = {
            "method": "not_applicable_to_fixed_nominal_drift",
            "target": "residual_drift",
            "drift_error_margin": 0.0,
            "scope": "diffusion_only_physical_sde",
        }
        context_error_margin = 0.0
        context_error_margin_source = "not_applicable_to_fixed_nominal_drift"
    elif context_certification == "empirical":
        context_certification_report = {
            "method": "empirical_training_support_validation",
            "target": "terminal_diffusion_moment_drift",
            "support_context_count": int(context_refs.shape[0]),
            "support_radius": float(context_set.coverage_radius),
            "drift_error_margin": 0.0,
            "scope": "selected_observed_training_contexts",
        }
        context_error_margin = 0.0
        context_error_margin_source = "included_in_empirical_multicontext_validation_envelope"
    else:
        context_certification_report = {
            "method": "empirical_pairwise_context_lipschitz",
            "target": "terminal_diffusion_moment_drift_context_sensitivity",
            "context_lipschitz": float(context_lipschitz_empirical),
            "context_radius": float(context_set.coverage_radius),
            "drift_error_margin": float(context_error_margin_empirical),
        }
        context_error_margin = context_error_margin_empirical
        context_error_margin_source = "empirical_pairwise_context_lipschitz"
        context_certification_report = certify_diffusion_context_lipschitz(
            diffusion,
            context_radius=context_set.coverage_radius,
            dt=config.dt,
        )
        certified_margin = float(context_certification_report["drift_error_margin"])
        context_error_margin = max(context_error_margin_empirical, certified_margin)
        context_error_margin_source = "max(empirical_pairwise_context_lipschitz, spectral_norm_lipschitz)"

    sampler_covariance_report = certify_diffusion_sampler_covariance(diffusion, config.dt)
    analytic_covariance_bound = float(sampler_covariance_report["physical_sde_covariance_eigenvalue_bound"])
    residual_location = np.asarray(diffusion.residual_location, dtype=float)
    analytic_residual_drift_norm_bound = 0.0 if diffusion_physical_role == "diffusion_only" else (
        float(sampler_covariance_report["physical_root_second_moment_bound"])
        + float(np.linalg.norm(residual_location))
    ) / max(config.dt, 1e-12)
    reverse_surrogate_error_margin = 0.0 if diffusion_physical_role == "diffusion_only" else float(
        moment_samples.diagnostics.get("physical_drift_validation_error", 0.0)
    )
    analytic_abstraction_error = str(config.abstraction_error_bound_mode).strip().lower() == "analytic"
    # Analytic abstraction mode already bounds the difference by
    # ||learned drift|| + ||polynomial drift|| over each cell. Adding the local
    # reverse-surrogate validation indicator again would double count it.
    abstraction_extra_error_margin = 0.0 if analytic_abstraction_error else (
        reverse_surrogate_error_margin + context_error_margin
    )
    moment_method_label = (
        "reverse_polynomial" if residual_dynamics.moment_method == "reverse_polynomial" else "terminal_sample"
    )
    moment_source = (
        f"diffusion_{moment_method_label}_covariance_sde"
        if diffusion_physical_role == "diffusion_only"
        else f"diffusion_{moment_method_label}_moment_matched_residual_sde"
    )
    abstraction = fit_partitioned_polynomial_abstraction_from_drift(
        moment_x,
        abstraction_residual_drift,
        moment_samples.diffusion_covariance,
        box=abstraction_box,
        partitions=int(config.abstraction_partitions),
        dt=config.dt,
        degree=config.polynomial_degree,
        diffusion_margin=config.diffusion_margin,
        source=moment_source,
        context_count=len(context_refs),
        extra_error_margin=abstraction_extra_error_margin,
        error_bound_mode=config.abstraction_error_bound_mode,
        error_lipschitz_margin=config.abstraction_error_lipschitz_margin,
        diffusion_bound_mode=config.diffusion_bound_mode,
        diffusion_confidence_delta=config.diffusion_confidence_delta,
        diffusion_sample_count=config.score_residual_samples,
        analytic_diffusion_eigenvalue_bound=analytic_covariance_bound,
        analytic_residual_drift_norm_bound=analytic_residual_drift_norm_bound,
        partition_boxes=partition_boxes,
    )
    abstraction.abstraction_certificate["reverse_moment_diagnostics"] = moment_samples.diagnostics
    abstraction.abstraction_certificate["reverse_surrogate_drift_error_margin"] = reverse_surrogate_error_margin
    abstraction.abstraction_certificate["reverse_surrogate_error_covered_by_analytic_envelope"] = bool(
        analytic_abstraction_error
    )
    abstraction.abstraction_certificate["moment_source"] = moment_source
    abstraction.abstraction_certificate["diffusion_physical_role"] = diffusion_physical_role
    abstraction.abstraction_certificate["nominal_drift_preserved"] = bool(
        diffusion_physical_role == "diffusion_only"
    )
    terminal_calibration_report: dict[str, object] | None = None
    terminal_calibration_mode = str(config.terminal_sde_calibration).strip().lower().replace("-", "_")
    if terminal_calibration_mode == "pac":
        terminal_calibration_report = _apply_terminal_pac_calibration(
            abstraction=abstraction,
            residual_dynamics=residual_dynamics,
            system=system,
            training_contexts=c_train,
            partition_boxes=partition_boxes,
            config=config,
        )
    elif terminal_calibration_mode != "off":
        raise ValueError("terminal_sde_calibration must be 'off' or 'pac'")
    _restrict_abstraction_diffusion_support(
        abstraction,
        stochastic_state_indices,
        state_dim=config.state_dim,
    )

    safety_diagnostic = _estimate_empirical_unsafe_reach(system, config)
    barrier_backend = _select_barrier_backend(config)
    barrier_solver_error: str | None = None
    try:
        barrier = _fit_barrier_with_backend(barrier_backend, abstraction, system, config)
    except Exception as exc:
        if config.barrier_solver.strip().lower() != "auto":
            raise
        barrier_solver_error = str(exc)
        barrier_backend = "sample"
        barrier = _fit_barrier_with_backend(barrier_backend, abstraction, system, config)

    save_config(config, out / "config.json")
    if model_dir is not None:
        save_pickle(encoder, model_dir / "context_encoder.pkl")
        save_pickle(diffusion, model_dir / "conditional_residual_diffusion.pkl")
        save_pickle(residual_dynamics, model_dir / "diffusion_residual_moments.pkl")
        save_pickle(abstraction, model_dir / "polynomial_abstraction.pkl")
    save_json(barrier.to_dict(), out / "barrier_certificate.json")
    save_json(context_certification_report, out / "context_certification.json")

    learned_abstraction_global = terminal_calibration_mode == "off" and (
        str(config.abstraction_error_bound_mode).strip().lower() == "analytic"
        and str(config.diffusion_bound_mode).strip().lower() == "analytic"
    )
    summary: dict[str, object] = {
        "samples": int(samples["x"].shape[0]),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_test_split_scope": split_scope,
        "benchmark": config.benchmark,
        "certificate_scope": (
            "robust_sos_on_globally_bounded_learned_moment_abstraction"
            if learned_abstraction_global
            else (
                "robust_sos_on_terminal_pac_calibrated_moment_abstraction"
                if terminal_calibration_mode == "pac"
                else "robust_sos_on_empirical_diffusion_moment_abstraction"
            )
        ),
        "learned_abstraction_global_bounds": learned_abstraction_global,
        "terminal_sde_calibration": terminal_calibration_mode,
        "terminal_pac_calibration": terminal_calibration_report,
        "real_system_distribution_certified": False,
        "real_system_distribution_gap": "No bound currently relates the fitted stochastic-increment distribution to the unknown real distribution.",
        "physical_model": moment_source,
        "diffusion_physical_role": diffusion_physical_role,
        "diffusion_latent_dimension": int(diffusion.residual_dim),
        "physical_stochastic_state_indices": list(stochastic_state_indices),
        "diffusion_symmetrize_increments": increments_symmetrized,
        "nominal_drift_preserved": bool(diffusion_physical_role == "diffusion_only"),
        "diffusion_moment_method": residual_dynamics.moment_method,
        "reverse_moment_diagnostics": moment_samples.diagnostics,
        "reverse_surrogate_drift_error_margin": reverse_surrogate_error_margin,
        "encoder": config.encoder,
        "context_dim": int(config.context_dim),
        "context_mlp_hidden": list(config.context_mlp_hidden),
        "context_mlp_max_iter": int(config.context_mlp_max_iter),
        "diffusion_noise_mse": float(diffusion_stats["noise_mse"]),
        "diffusion_n_iter": int(diffusion_stats["n_iter"]),
        "diffusion_training_examples": int(diffusion_stats["training_examples"]),
        "diffusion_schedule": schedule.schedule_type,
        "diffusion_terminal_alpha_bar": float(schedule.alpha_bars[-1]),
        "diffusion_train_copies": int(config.diffusion_train_copies),
        "diffusion_early_stopping": bool(config.diffusion_early_stopping),
        "residual_mean_mae": residual_mae,
        "stochastic_increment_mean_mae": residual_mae,
        "learned_increment_mean_norm": learned_increment_mean_norm,
        "score_residual_samples": int(config.score_residual_samples),
        "score_context_bound_samples": int(len(context_refs)),
        "score_poly_samples": int(abstraction_states.shape[0]),
        "context_certification": context_certification,
        "context_set": context_set.to_dict(),
        "context_reference_min": context_refs.min(axis=0).tolist(),
        "context_reference_max": context_refs.max(axis=0).tolist(),
        "context_lipschitz_estimate": float(context_lipschitz_empirical),
        "context_lipschitz_empirical": float(context_lipschitz_empirical),
        "context_error_margin_empirical": float(context_error_margin_empirical),
        "context_certification_report": context_certification_report,
        "context_error_margin_source": context_error_margin_source,
        "context_error_margin": float(context_error_margin),
        "context_error_margin_applied_to_abstraction": float(abstraction_extra_error_margin),
        "polynomial_degree": int(config.polynomial_degree),
        "polynomial_terms": int(len(abstraction.exponents)),
        "polynomial_source": abstraction.source,
        "abstraction_method": abstraction.abstraction_certificate.get("method"),
        "abstraction_partitions": int(config.abstraction_partitions),
        "abstraction_validation_grid": int(config.abstraction_validation_grid),
        "abstraction_error_bound_mode": config.abstraction_error_bound_mode,
        "abstraction_error_lipschitz_margin": float(config.abstraction_error_lipschitz_margin),
        "diffusion_bound_mode": config.diffusion_bound_mode,
        "diffusion_confidence_delta": float(config.diffusion_confidence_delta),
        "sampler_covariance_certification": sampler_covariance_report,
        "analytic_residual_drift_norm_bound": float(analytic_residual_drift_norm_bound),
        "abstraction_certificate": abstraction.abstraction_certificate,
        "abstraction_probe_states": int(abstraction_states.shape[0]),
        "residual_abstraction_fit_error": float(abstraction.fit_error),
        "residual_abstraction_epsilon": float(abstraction.epsilon_r),
        "diffusion_upper_bound_type": abstraction.diffusion_upper_bound_type,
        "Gbar": abstraction.Gbar.tolist(),
        "barrier": barrier.to_dict(),
        "mosek_available": mosek_sos_available(),
        "barrier_solver_requested": config.barrier_solver,
        "barrier_solver_used": barrier.solver_backend,
        "barrier_formulation": config.barrier_formulation,
        "barrier_solver_error": barrier_solver_error,
        "barrier_degree": int(config.barrier_degree),
        "barrier_samples": int(config.barrier_samples),
        "barrier_verify_box": [list(b) for b in ((system.barrier_box if system.barrier_box is not None else system.verify_box))],
        "counterexample_rounds": int(config.counterexample_rounds),
        "benchmark_simulation_diagnostic": safety_diagnostic,
        "sos_drift_degree": int(config.sos_drift_degree),
        "sos_drift_fit_samples": int(config.sos_drift_fit_samples),
        "sos_relaxation_degree": int(config.sos_relaxation_degree),
        "sos_margin": float(config.sos_margin),
        "sos_rho_upper": float(config.sos_rho_upper),
        "sos_initial_objective_weight": float(config.sos_initial_objective_weight),
        "sos_abstraction_error_mode": config.sos_abstraction_error_mode,
        "sbc_epsilon": float(config.sbc_epsilon),
        "sbc_c": float(config.sbc_c),
        "sbc_time_horizon": float(config.sbc_time_horizon),
        "paths": {
            "dataset": str(dataset_path) if dataset_path is not None else None,
            "model_dir": "model" if save_artifacts else None,
            "context_encoder": "model/context_encoder.pkl" if save_artifacts else None,
            "conditional_residual_diffusion": "model/conditional_residual_diffusion.pkl" if save_artifacts else None,
            "diffusion_residual_moments": "model/diffusion_residual_moments.pkl" if save_artifacts else None,
            "polynomial_abstraction": "model/polynomial_abstraction.pkl" if save_artifacts else None,
            "barrier_certificate": "barrier_certificate.json",
            "context_certification": "context_certification.json",
        },
        "runtime_sec": float(time.time() - start),
        "notes": [
            "Context encoder uses an MLP autoencoder by default; PCA and stats remain available as lightweight alternatives.",
            (
                "The nominal physical drift is preserved exactly; the learned diffusion model contributes only a calibrated covariance-rate bound to the SOS generator."
                if diffusion_physical_role == "diffusion_only"
                else "The certificate applies to a robust abstraction of the learned residual drift and diffusion covariance."
            ),
            (
                "Terminal PAC calibration uses independent condition draws and a distribution-free maximum-order-statistic tolerance theorem. Its guarantee is over a fresh condition from the declared calibration distribution, not a deterministic supremum over the continuous state/context domain or a trajectory-level real-system guarantee."
                if terminal_calibration_mode == "pac"
                else "Terminal PAC calibration is disabled; the configured abstraction and diffusion bound modes determine the learned-model envelope."
            ),
            (
                "The physical model uses deterministic quadratic reverse-update surrogates and Gaussian moment closure; "
                "it draws no terminal residual samples and is not the reverse-time latent SDE itself."
                if residual_dynamics.moment_method == "reverse_polynomial"
                else "The physical increment covariance is estimated from terminal samples of the reverse diffusion process; the reverse-time latent SDE is not used as the physical vehicle dynamics."
            ),
            (
                "The reverse-polynomial validation margin is included in the residual abstraction error. Its deterministic-node fit error and Gaussian moment-closure error are not global neural-network guarantees."
                if residual_dynamics.moment_method == "reverse_polynomial"
                else "Terminal reverse sampling estimates conditional residual moments by Monte Carlo."
            ),
            (
                "Context coverage is used when estimating the learned diffusion covariance; no context-dependent residual drift is added in diffusion-only mode."
                if diffusion_physical_role == "diffusion_only"
                else "Verification evaluates context coverage and propagates its margin into the residual-drift abstraction error."
            ),
            "context_certification='spectral_lipschitz' computes a spectral-norm Lipschitz upper bound for the sklearn MLP diffusion noise predictor and propagates it through the implemented reverse sampler.",
            "The default formulation jointly minimizes rho and B subject to B<=rho on the initial set, B>=1 on the unsafe set, B>=0 and AB<=0 on the stopped verification domain. This rho is the gamma bound in Prajna et al. (CDC 2004); fixed_epsilon remains available for compatibility.",
            "The benchmark_simulation_diagnostic uses the benchmark's separate simulator and is not comparable to the certificate unless the simulator has been formally related to the learned abstraction.",
            "Model artifacts are omitted when this run is part of a summary-only comparison sweep." if not save_artifacts else "Model artifacts are saved under model/.",
            "Default barrier_solver is mosek, so no-argument demo runs use the MOSEK SOS/SDP backend. Use --barrier-solver sample for an exploratory sampled fallback, or auto to try MOSEK and fall back to sample on solver failure.",
        ],
    }
    save_json(summary, out / "summary.json")
    return summary


def run_demo(out_dir: str | Path, config: ExperimentConfig | None = None) -> dict[str, object]:
    cfg = config or ExperimentConfig()
    out = ensure_dir(out_dir)
    dataset_path = Path("data") / "dataset.npz"
    samples = generate_dataset(cfg, out / dataset_path)
    return train_pipeline(samples, cfg, out, dataset_path=dataset_path)


def _apply_terminal_pac_calibration(
    *,
    abstraction,
    residual_dynamics: ConditionalMomentMatchedResidualSDE,
    system,
    training_contexts: np.ndarray,
    partition_boxes: list[tuple[tuple[float, float], ...]],
    config: ExperimentConfig,
) -> dict[str, object]:
    """Calibrate terminal SDE quantities after the complete reverse chain.

    The fitted polynomial abstraction is held fixed. Fresh states are sampled
    uniformly inside each partition and contexts are sampled independently
    from the empirical training-context distribution. The maximum order
    statistic supplies a finite-sample PAC tolerance statement. Diffusion-only
    runs calibrate covariance alone because their drift is the fixed nominal
    model; legacy residual-SDE runs also calibrate residual drift. This does not
    claim deterministic box coverage or certify the model-to-real gap.
    """

    context_arr = np.atleast_2d(np.asarray(training_contexts, dtype=float))
    if context_arr.shape[0] < 1:
        raise ValueError("terminal PAC calibration requires at least one training context")
    if not partition_boxes:
        raise ValueError("terminal PAC calibration requires at least one partition")
    samples_per_cell = int(config.terminal_calibration_samples_per_partition)
    if samples_per_cell < 2:
        raise ValueError("terminal_calibration_samples_per_partition must be at least two")
    total_delta = float(config.terminal_calibration_confidence_delta)
    if not 0.0 < total_delta < 1.0:
        raise ValueError("terminal_calibration_confidence_delta must lie strictly between zero and one")

    diffusion_physical_role = _normalize_diffusion_physical_role(config.diffusion_physical_role)
    calibrate_drift = diffusion_physical_role == "residual_sde"
    claims_per_partition = 2 if calibrate_drift else 1
    claim_count = claims_per_partition * len(partition_boxes)
    per_claim_delta = total_delta / float(claim_count)
    rng = np.random.default_rng(int(config.seed) + 20031)
    local_drift_bounds: list[float] = []
    local_covariance_bounds: list[float] = []
    cell_reports: list[dict[str, object]] = []

    for cell_index, cell_box in enumerate(partition_boxes):
        lows = np.asarray([bound[0] for bound in cell_box], dtype=float)
        highs = np.asarray([bound[1] for bound in cell_box], dtype=float)
        states = rng.uniform(lows, highs, size=(samples_per_cell, len(cell_box)))
        context_indices = rng.integers(0, context_arr.shape[0], size=samples_per_cell)
        contexts = context_arr[context_indices]
        controls = batch_control_policy(system, states, step=0)
        moments = residual_dynamics.physical_sde_moments(
            states,
            controls,
            contexts,
            n_samples=config.score_residual_samples,
            seed=int(config.seed) + 21000 + cell_index,
        )

        covariance = np.asarray(moments.diffusion_covariance, dtype=float)
        covariance = (covariance + np.swapaxes(covariance, 1, 2)) / 2.0
        covariance_eigenvalues = np.linalg.eigvalsh(covariance)[:, -1]

        if calibrate_drift:
            predicted_drift = abstraction.residual_drift(states)
            drift_errors = np.linalg.norm(
                np.asarray(moments.residual_drift, dtype=float) - predicted_drift,
                axis=1,
            )
            drift_report: dict[str, object] = maximum_order_statistic_pac_bound(
                drift_errors,
                per_claim_delta,
            )
            drift_bound = float(drift_report["threshold"])
        else:
            drift_report = {
                "theorem": "not_applicable",
                "reason": "diffusion_only_preserves_the_nominal_drift",
                "threshold": 0.0,
                "sample_count": int(samples_per_cell),
            }
            drift_bound = 0.0
        covariance_report = maximum_order_statistic_pac_bound(covariance_eigenvalues, per_claim_delta)
        covariance_bound = float(config.diffusion_margin) * max(
            0.0,
            float(covariance_report["threshold"]),
        )
        local_drift_bounds.append(drift_bound)
        local_covariance_bounds.append(covariance_bound)
        cell_reports.append(
            {
                "cell_index": int(cell_index),
                "box": [[float(lo), float(hi)] for lo, hi in cell_box],
                "calibration_distribution": "uniform_state_in_cell_x_empirical_training_context",
                "drift_error": drift_report,
                "covariance_max_eigenvalue": covariance_report,
                "covariance_margin": float(config.diffusion_margin),
                "calibrated_covariance_eigenvalue_bound": covariance_bound,
            }
        )

    global_covariance_bound = max(local_covariance_bounds)
    pre_calibration = {
        "epsilon_r": float(abstraction.epsilon_r),
        "local_error_bounds": [float(value) for value in abstraction.local_fit_errors],
        "Gbar": np.asarray(abstraction.Gbar, dtype=float).tolist(),
        "diffusion_upper_bound_type": str(abstraction.diffusion_upper_bound_type),
        "error_bound_mode": abstraction.abstraction_certificate.get("error_bound_mode"),
        "diffusion_bound_mode": abstraction.abstraction_certificate.get("diffusion_bound_mode"),
    }
    abstraction.local_fit_errors = local_drift_bounds
    abstraction.epsilon_r = max(local_drift_bounds)
    abstraction.extra_error_margin = 0.0
    abstraction.Gbar = np.eye(int(config.state_dim), dtype=float) * global_covariance_bound
    abstraction.gbar = np.eye(int(config.state_dim), dtype=float) * np.sqrt(global_covariance_bound)
    abstraction.diffusion_upper_bound_type = "terminal_pac_max_order_statistic"

    violation_probability = max(
        float(report["covariance_max_eigenvalue"]["violation_probability_upper_bound"])
        for report in cell_reports
    )
    certificate = dict(abstraction.abstraction_certificate)
    certificate.update(
        {
            "method": (
                "independent_terminal_pac_covariance_calibration"
                if not calibrate_drift
                else "partitioned_least_squares_with_independent_terminal_pac_calibration"
            ),
            "error_bound_mode": (
                "not_applicable_fixed_nominal_drift" if not calibrate_drift else "terminal_pac"
            ),
            "diffusion_bound_mode": "terminal_pac",
            "diffusion_bound_is_global_for_learned_sampler": False,
            "local_error_bounds_include_extra_margin": bool(calibrate_drift),
            "epsilon_r": float(abstraction.epsilon_r),
            "max_local_error_bound": float(abstraction.epsilon_r),
            "terminal_pac_calibrated": True,
            "terminal_pac_calibration": {
                "theorem": "maximum_order_statistic_distribution_free_tolerance_bound",
                "total_confidence_delta": total_delta,
                "joint_confidence_lower_bound": float(1.0 - total_delta),
                "claim_count": int(claim_count),
                "calibrated_quantities": (
                    ["residual_drift_error", "diffusion_covariance_max_eigenvalue"]
                    if calibrate_drift
                    else ["diffusion_covariance_max_eigenvalue"]
                ),
                "per_claim_confidence_delta": per_claim_delta,
                "samples_per_partition": samples_per_cell,
                "partition_count": int(len(partition_boxes)),
                "per_partition_condition_violation_probability_upper_bound": violation_probability,
                "condition_distribution": "uniform within each state partition and uniform over the empirical training-context support",
                "guarantee_scope": "fresh_condition_from_declared_distribution_for_fixed_learned_model_and_fixed_polynomial_abstraction",
                "deterministic_continuous_box_coverage": False,
                "trajectory_level_safety_transfer_proved": False,
                "gaussian_moment_closure_error_certified": False,
                "real_system_distribution_gap_certified": False,
                "global_covariance_eigenvalue_bound": global_covariance_bound,
                "cell_reports": cell_reports,
            },
            "pre_terminal_pac_bounds": pre_calibration,
        }
    )
    abstraction.abstraction_certificate = certificate
    return dict(certificate["terminal_pac_calibration"])


def _select_context_references(contexts: np.ndarray, n_contexts: int, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(contexts, dtype=float)
    count = min(max(1, int(n_contexts)), arr.shape[0])
    if count == arr.shape[0]:
        return arr.copy()
    mean_ref = arr.mean(axis=0, keepdims=True)
    if count == 1:
        return mean_ref
    # Greedy max-min selection covers observed histories more reliably than a
    # random subset while keeping the moment-evaluation cost bounded.
    selected = [mean_ref[0]]
    nearest_sq = np.sum((arr - mean_ref[0]) ** 2, axis=1)
    for _ in range(count - 1):
        index = int(np.argmax(nearest_sq))
        point = arr[index]
        selected.append(point)
        nearest_sq = np.minimum(nearest_sq, np.sum((arr - point) ** 2, axis=1))
    return np.vstack(selected)


def _normalize_context_certification(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"spectral_lipschitz", "empirical"}:
        return mode
    raise ValueError("context_certification must be 'spectral_lipschitz' or 'empirical'")


def _build_context_set(contexts: np.ndarray, config: ExperimentConfig, rng: np.random.Generator) -> ContextSet:
    arr = np.asarray(contexts, dtype=float)
    lower = arr.min(axis=0)
    upper = arr.max(axis=0)
    span = upper - lower
    fallback_span = np.maximum(np.abs(arr.mean(axis=0)), 1.0)
    span = np.where(span > 1e-10, span, 1e-6 * fallback_span)
    margin = float(config.context_box_margin) * span
    lower = lower - margin
    upper = upper + margin
    center = (lower + upper) / 2.0
    coverage = _context_box_coverage_points(
        lower,
        upper,
        max_points=max(1, int(config.score_context_bound_samples)),
        rng=rng,
    )
    radius = _estimate_context_coverage_radius(lower, upper, coverage, rng)
    return ContextSet(
        mode="axis_aligned_box",
        lower=lower,
        upper=upper,
        center=center,
        coverage_points=coverage,
        margin_fraction=float(config.context_box_margin),
        coverage_radius=radius,
    )


def _build_empirical_context_set(contexts: np.ndarray, references: np.ndarray) -> ContextSet:
    arr = np.asarray(contexts, dtype=float)
    refs = np.asarray(references, dtype=float)
    lower = arr.min(axis=0)
    upper = arr.max(axis=0)
    center = arr.mean(axis=0)
    nearest = np.linalg.norm(arr[:, None, :] - refs[None, :, :], axis=2)
    radius = float(np.max(np.min(nearest, axis=1)))
    return ContextSet(
        mode="empirical_training_support",
        lower=lower,
        upper=upper,
        center=center,
        coverage_points=refs,
        margin_fraction=0.0,
        coverage_radius=radius,
    )


def _probe_box_points(
    box: tuple[tuple[float, float], ...],
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lows = np.array([b[0] for b in box], dtype=float)
    highs = np.array([b[1] for b in box], dtype=float)
    random_points = rng.uniform(lows, highs, size=(n, len(box)))
    corners = _box_corners(box)
    if len(box) != 2:
        return np.vstack([random_points, corners])

    side = max(6, int(np.sqrt(max(1, n))))
    x0 = np.linspace(box[0][0], box[0][1], side)
    x1 = np.linspace(box[1][0], box[1][1], side)
    grid0, grid1 = np.meshgrid(x0, x1, indexing="ij")
    grid = np.column_stack([grid0.reshape(-1), grid1.reshape(-1)])

    edge_n = max(6, side)
    xs = np.linspace(box[0][0], box[0][1], edge_n)
    ys = np.linspace(box[1][0], box[1][1], edge_n)
    lower = np.column_stack([xs, np.full_like(xs, box[1][0])])
    upper = np.column_stack([xs, np.full_like(xs, box[1][1])])
    left = np.column_stack([np.full_like(ys, box[0][0]), ys])
    right = np.column_stack([np.full_like(ys, box[0][1]), ys])
    return np.vstack([random_points, grid, lower, upper, left, right, corners])


def _partitioned_grid_probe_points(
    box: tuple[tuple[float, float], ...],
    *,
    degree: int,
    partitions: int,
    validation_grid: int,
    random_count: int,
    rng: np.random.Generator,
    partition_boxes: list[tuple[tuple[float, float], ...]] | None = None,
) -> np.ndarray:
    cells = list(partition_boxes) if partition_boxes is not None else partition_box(box, max(1, int(partitions)))
    node_count = max(2, int(degree) + 1)
    validation_count = max(node_count, int(validation_grid))
    all_points: list[np.ndarray] = []

    for cell in cells:
        all_points.append(_tensor_grid_points(cell, node_count))
        all_points.append(_tensor_grid_points(cell, validation_count))

    random_per_cell = int(np.ceil(max(0, int(random_count)) / max(1, len(cells))))
    if random_per_cell > 0:
        for cell in cells:
            lows = np.array([b[0] for b in cell], dtype=float)
            highs = np.array([b[1] for b in cell], dtype=float)
            all_points.append(rng.uniform(lows, highs, size=(random_per_cell, len(cell))))

    points = np.vstack(all_points)
    rounded = np.round(points, decimals=12)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx]


def _tensor_grid_points(
    box: tuple[tuple[float, float], ...],
    count: int,
    max_points: int = 2048,
) -> np.ndarray:
    """Return the full tensor grid when feasible and a deterministic subset otherwise."""
    level_count = max(2, int(count))
    dimension = len(box)
    total = level_count**dimension
    axes = [np.linspace(float(lo), float(hi), level_count) for lo, hi in box]
    if total <= int(max_points):
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([m.reshape(-1) for m in mesh])

    lows = np.asarray([float(lo) for lo, _ in box], dtype=float)
    highs = np.asarray([float(hi) for _, hi in box], dtype=float)
    center = 0.5 * (lows + highs)
    points = [center]
    for axis, levels in enumerate(axes):
        for value in levels:
            point = center.copy()
            point[axis] = value
            points.append(point)

    remaining = max(0, int(max_points) - len(points))
    if remaining:
        indices = np.arange(1, remaining + 1, dtype=float)[:, None]
        multipliers = np.sqrt(np.arange(2, dimension + 2, dtype=float))[None, :]
        fractions = np.mod(indices * multipliers, 1.0)
        level_indices = np.minimum(
            level_count - 1,
            np.floor(fractions * level_count).astype(int),
        )
        sampled = np.column_stack(
            [axes[axis][level_indices[:, axis]] for axis in range(dimension)]
        )
        points.extend(sampled)
    return np.unique(np.asarray(points, dtype=float), axis=0)


def _box_corners(box: tuple[tuple[float, float], ...]) -> np.ndarray:
    lows_highs = [[lo, hi] for lo, hi in box]
    mesh = np.meshgrid(*lows_highs, indexing="ij")
    return np.column_stack([m.reshape(-1) for m in mesh])


def _context_box_coverage_points(
    lower: np.ndarray,
    upper: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    dim = lower.shape[0]
    center = ((lower + upper) / 2.0)[None, :]
    corners = np.array(list(product([0, 1], repeat=dim)), dtype=float)
    corners = lower + corners * (upper - lower)
    if max_points >= corners.shape[0] + 1:
        points = np.vstack([center, corners])
    elif max_points >= corners.shape[0]:
        points = corners
    elif max_points == 1:
        points = center
    else:
        selected = _farthest_context_subset(corners, max_points - 1, center[0])
        points = np.vstack([center, selected])
    remaining = max_points - points.shape[0]
    if remaining > 0:
        random_points = rng.uniform(lower, upper, size=(remaining, dim))
        points = np.vstack([points, random_points])
    return points[:max_points]


def _farthest_context_subset(points: np.ndarray, count: int, first: np.ndarray) -> np.ndarray:
    if count <= 0:
        return np.empty((0, points.shape[1]), dtype=float)
    chosen: list[np.ndarray] = []
    distances = np.linalg.norm(points - first[None, :], axis=1)
    idx = int(np.argmax(distances))
    chosen.append(points[idx])
    while len(chosen) < min(count, points.shape[0]):
        chosen_arr = np.vstack(chosen)
        min_dist = np.min(np.linalg.norm(points[:, None, :] - chosen_arr[None, :, :], axis=2), axis=1)
        idx = int(np.argmax(min_dist))
        if any(np.allclose(points[idx], c) for c in chosen):
            break
        chosen.append(points[idx])
    return np.vstack(chosen)


def _estimate_context_coverage_radius(
    lower: np.ndarray,
    upper: np.ndarray,
    coverage_points: np.ndarray,
    rng: np.random.Generator,
    n_probe: int = 512,
) -> float:
    probes = rng.uniform(lower, upper, size=(n_probe, lower.shape[0]))
    distances = np.linalg.norm(probes[:, None, :] - coverage_points[None, :, :], axis=2)
    return float(np.max(np.min(distances, axis=1)))


def _estimate_context_lipschitz(
    residual_drift: np.ndarray,
    contexts: np.ndarray,
    n_states: int,
) -> float:
    drift = np.asarray(residual_drift, dtype=float).reshape(n_states, contexts.shape[0], -1)
    c = np.asarray(contexts, dtype=float)
    max_ratio = 0.0
    for i in range(c.shape[0]):
        for j in range(i + 1, c.shape[0]):
            dc = float(np.linalg.norm(c[i] - c[j]))
            if dc <= 1e-12:
                continue
            dr = np.linalg.norm(drift[:, i, :] - drift[:, j, :], axis=1)
            max_ratio = max(max_ratio, float(np.max(dr) / dc))
    return max_ratio


def _expand_over_contexts(x: np.ndarray, u: np.ndarray, contexts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    u_arr = np.asarray(u, dtype=float)
    c_arr = np.asarray(contexts, dtype=float)
    n_contexts = c_arr.shape[0]
    return (
        np.repeat(x_arr, n_contexts, axis=0),
        np.repeat(u_arr, n_contexts, axis=0),
        np.tile(c_arr, (x_arr.shape[0], 1)),
    )


def _estimate_empirical_unsafe_reach(system, config: ExperimentConfig) -> dict[str, object]:
    rollouts = max(0, int(config.diagnostic_rollouts))
    horizon = max(0, int(config.diagnostic_horizon))
    if rollouts == 0 or horizon == 0:
        return {
            "unsafe_reach_rate": None,
            "mean_first_hit_step": None,
            "hit_count": 0,
            "rollouts": rollouts,
            "horizon": horizon,
        }

    rng = rng_from_seed(config.seed + 900)
    x = system.sample_initial_states(rng, rollouts)
    unsafe_lower = np.array([b[0] for b in system.unsafe_box], dtype=float)
    unsafe_upper = np.array([b[1] for b in system.unsafe_box], dtype=float)
    reached = np.zeros(rollouts, dtype=bool)
    first_hit = np.full(rollouts, -1, dtype=int)

    for step in range(horizon + 1):
        inside = np.all((x >= unsafe_lower) & (x <= unsafe_upper), axis=1)
        newly_reached = inside & ~reached
        first_hit[newly_reached] = step
        reached |= inside
        if step == horizon:
            break
        u = batch_control_policy(system, x, step=step)
        x = system.actual_step(x, u, rng, step=step)

    hit_count = int(np.sum(reached))
    mean_hit = float(np.mean(first_hit[first_hit >= 0])) if hit_count else None
    return {
        "unsafe_reach_rate": float(hit_count / max(1, rollouts)),
        "mean_first_hit_step": mean_hit,
        "hit_count": hit_count,
        "rollouts": rollouts,
        "horizon": horizon,
    }


def _select_barrier_backend(config: ExperimentConfig) -> str:
    requested = config.barrier_solver.strip().lower()
    if requested == "sample":
        return "sample"
    if requested == "mosek-sampled":
        if not mosek_sos_available():
            raise RuntimeError("barrier_solver='mosek-sampled' was requested, but cvxpy/MOSEK is not installed.")
        return "mosek_sampled"
    if requested == "mosek":
        if not mosek_sos_available():
            raise RuntimeError("barrier_solver='mosek' was requested, but cvxpy/MOSEK is not installed.")
        return "mosek_sos"
    if requested == "auto" and mosek_sos_available():
        return "mosek_sos"
    return "sample"


def _fit_barrier_with_backend(
    backend: str,
    abstraction,
    system,
    config: ExperimentConfig,
):
    if backend == "mosek_sos":
        return SosBarrierSynthesizer(abstraction, system, config).fit()
    if backend == "mosek_sampled":
        return MosekSampledBarrierSynthesizer(abstraction, system, config).fit()
    return SampledBarrierSynthesizer(abstraction, system, config).fit()
