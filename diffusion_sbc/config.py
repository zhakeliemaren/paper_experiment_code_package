from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Default settings for the runnable prototype.

    These defaults instantiate the X-Plane11 benchmark and run the complete
    learning plus SOS safety-verification pipeline end to end.
    """

    seed: int = 7
    benchmark: str = "xplane11"
    dt: float = 0.05
    state_dim: int = 2
    control_dim: int = 1

    n_trajectories: int = 32
    horizon: int = 60
    history: int = 8
    process_noise: float = 0.03
    # CARLA only: the nominal model uses dv/dt=-a, while the simulator uses
    # dv/dt=-mu*a with a bounded, state-dependent tire-road coefficient.
    carla_friction_enabled: bool = True
    carla_friction_mean: float = 0.82
    carla_friction_std: float = 0.12
    carla_friction_min: float = 0.45
    carla_friction_max: float = 1.12
    carla_friction_speed_gain: float = 0.10
    carla_friction_distance_gain: float = 0.08
    carla_absorbing_stop: bool = True
    carla_stop_speed: float = 0.5

    context_dim: int = 4
    encoder: str = "mlp"
    context_mlp_hidden: tuple[int, ...] = (32,)
    context_mlp_max_iter: int = 250

    diffusion_steps: int = 64
    diffusion_schedule: str = "cosine"
    # Each residual is paired with several randomly selected diffusion times.
    diffusion_train_copies: int = 8
    diffusion_hidden: tuple[int, ...] = (128, 128)
    diffusion_max_iter: int = 800
    diffusion_batch_size: int = 256
    diffusion_learning_rate: float = 5e-4
    diffusion_early_stopping: bool = False
    diffusion_beta_start: float = 1e-4
    diffusion_beta_end: float = 2e-2
    diffusion_max_beta: float = 0.5
    diffusion_terminal_alpha_bar_max: float = 1e-3
    residual_samples_for_mean: int = 16

    # ``diffusion_only`` learns the stochastic Euler-Maruyama increment
    # x[k+1]-x[k]-dt*f_nom(x[k],u[k]) and uses only its covariance rate in the
    # physical SDE. The nominal drift is kept unchanged. ``residual_sde`` is
    # the legacy mode that also interprets the learned conditional mean as an
    # additive residual drift.
    diffusion_physical_role: str = "diffusion_only"
    diffusion_symmetrize_increments: bool = True

    # ``reverse_polynomial`` avoids terminal Monte Carlo residual samples. At
    # each reverse DDPM step it fits a quadratic noise-predictor surrogate on
    # deterministic Gaussian nodes and propagates mean/covariance analytically
    # with Gaussian moment closure. ``terminal_sampling`` retains the original
    # stochastic reverse-sampling estimator as a comparison mode.
    diffusion_moment_method: str = "reverse_polynomial"
    reverse_polynomial_degree: int = 2
    reverse_polynomial_fit_radius: float = 2.0
    reverse_polynomial_validation_radius: float = 3.0
    reverse_polynomial_validation_margin: float = 1.25

    score_residual_samples: int = 8
    score_context_bound_samples: int = 17
    score_poly_samples: int = 80
    context_box_margin: float = 0.05
    context_certification: str = "spectral_lipschitz"
    diffusion_margin: float = 1.2

    polynomial_degree: int = 2
    abstraction_partitions: int = 2
    abstraction_validation_grid: int = 5
    abstraction_error_bound_mode: str = "analytic"
    abstraction_error_lipschitz_margin: float = 1.25
    diffusion_bound_mode: str = "analytic"
    diffusion_confidence_delta: float = 0.05
    # Calibrate the completed reverse chain, rather than propagating a global
    # worst-case neural bound through every reverse step. ``pac`` uses an
    # independent maximum-order-statistic calibration set. Its guarantee is
    # distributional over the declared condition distribution, not a
    # deterministic supremum over the continuous verification box.
    terminal_sde_calibration: str = "pac"
    terminal_calibration_samples_per_partition: int = 128
    terminal_calibration_confidence_delta: float = 0.05
    # Verification dynamics. The discrete transition mode calls a complete
    # conditional reverse diffusion at each physical sampling instant and
    # certifies E[B(X[k+1])|X[k]]-B(X[k]) <= 0.
    dynamics_verification_mode: str = "continuous_sde"
    transition_expectation_samples: int = 32
    # ``hoeffding`` is distribution-free but often dominates the certificate.
    # The empirical standard-error mode is tighter and explicitly reported as
    # an empirical finite-condition uncertainty indicator, not a global bound.
    transition_expectation_bound_mode: str = "empirical_standard_error"
    transition_polynomial_degree: int = 3
    transition_validation_margin: float = 1.25
    transition_condition_batch_size: int = 128
    barrier_degree: int = 2
    barrier_samples: int = 300
    barrier_max_iter: int = 250
    counterexample_rounds: int = 2
    barrier_solver: str = "mosek"
    # Jointly optimize B and the reachability bound rho. This is the gamma
    # optimization in Prajna et al. (CDC 2004), with project notation rho.
    barrier_formulation: str = "prajna_optimization"
    # ``worst_case`` certifies every state in X0. ``uniform_box`` certifies
    # the unconditional probability for x(0) uniformly distributed on X0.
    initial_condition_mode: str = "worst_case"

    sos_drift_degree: int = 3
    sos_drift_fit_samples: int = 80
    sos_relaxation_degree: int = 4
    sos_margin: float = 1e-6
    sos_rho_upper: float = 1.5
    sos_regularization: float = 1e-7
    sos_sample_tolerance: float = 1e-3
    sos_constraint_tolerance: float = 1e-7
    sos_gram_eigenvalue_tolerance: float = 1e-8
    # Optional MOSEK interior-point tolerances for numerically delicate SOS
    # programs. None preserves the solver defaults used by existing studies.
    sos_mosek_tolerance: float | None = None
    sos_initial_objective_weight: float = 0.0
    # ``total`` propagates both residual-abstraction and SOS drift-fit error.
    # Turning this off is useful only for numerical debugging, not reporting.
    sos_abstraction_error_mode: str = "total"
    sos_verbose: bool = False

    sbc_epsilon: float = 0.05
    sbc_c: float = 0.0
    sbc_time_horizon: float = 0.0

    diagnostic_rollouts: int = 512
    diagnostic_horizon: int = 120

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        clean = dict(data)
        if "context_mlp_hidden" in clean:
            clean["context_mlp_hidden"] = tuple(clean["context_mlp_hidden"])
        if "diffusion_hidden" in clean:
            clean["diffusion_hidden"] = tuple(clean["diffusion_hidden"])
        return cls(**clean)


def load_config(path: str | Path | None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        return ExperimentConfig.from_dict(json.load(f))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    import json

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
