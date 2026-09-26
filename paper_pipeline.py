"""Training and Generator-SOS synthesis for the paper's proposed method."""

from __future__ import annotations

import copy
import csv
import json
import pickle
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_sbc.barrier import SosBarrierSynthesizer
from diffusion_sbc.benchmarks import build_samples, generate_trajectories
from diffusion_sbc.config import ExperimentConfig
from diffusion_sbc.pipeline import train_pipeline
from diffusion_sbc.sos import Poly
from paper_benchmarks import (
    PAPER_CASE_NAMES,
    PAPER_SOURCE_LABELS,
    build_system,
    certificate_config,
    problem_size,
    relaxation_degree,
)


RESULT_PROTOCOL = "paper_main_diffusion_generator_sos_v1"
METHOD_NAME = "diffusion_direct_generator_sos"
# Display name used by the manuscript for the Gaussian discrete baseline.
# The serialized method key remains unchanged for compatibility with archived
# comparison results and is intentionally not used as a case identifier.
DIFFSBC_METHOD_NAME = "DiffSBC"


def resolve_stochastic_state_index(state_dim: int, requested: int) -> int:
    index = int(requested)
    if index < 0:
        index += int(state_dim)
    if index < 0 or index >= int(state_dim):
        raise ValueError(
            f"stochastic state index {requested} is invalid for dimension {state_dim}"
        )
    return index


def training_config(
    case_id: str, state_dim: int, drift_degree: int, args: Any
) -> ExperimentConfig:
    """Translate the documented paper parameters to the library configuration."""
    return replace(
        ExperimentConfig(),
        seed=int(args.seed),
        benchmark=f"paper_{case_id.lower()}",
        dt=0.05,
        state_dim=int(state_dim),
        control_dim=1,
        n_trajectories=int(args.n_trajectories),
        horizon=int(args.horizon),
        history=int(args.history),
        process_noise=float(args.data_noise_amplitude),
        diffusion_physical_role="diffusion_only",
        diffusion_symmetrize_increments=True,
        encoder="stats",
        context_dim=min(4, max(1, int(state_dim))),
        diffusion_steps=int(args.diffusion_steps),
        diffusion_terminal_alpha_bar_max=0.003,
        diffusion_train_copies=int(args.diffusion_train_copies),
        diffusion_hidden=(64, 64),
        diffusion_max_iter=int(args.diffusion_max_iter),
        diffusion_batch_size=256,
        diffusion_learning_rate=5e-4,
        diffusion_early_stopping=True,
        diffusion_moment_method="reverse_polynomial",
        score_residual_samples=int(args.moment_conditions),
        score_context_bound_samples=1,
        score_poly_samples=int(args.polynomial_conditions),
        context_certification="empirical",
        polynomial_degree=2,
        abstraction_partitions=1,
        abstraction_validation_grid=2,
        abstraction_error_bound_mode="validation",
        diffusion_bound_mode="sample",
        diffusion_confidence_delta=0.05,
        diffusion_margin=1.0,
        terminal_sde_calibration="off",
        barrier_solver="sample",
        barrier_degree=2,
        barrier_max_iter=1,
        counterexample_rounds=0,
        diagnostic_rollouts=0,
        diagnostic_horizon=0,
        sos_drift_degree=max(2, int(drift_degree)),
        sos_abstraction_error_mode="off",
    )


def verification_center_model(
    nominal_system: Any,
    nominal_polynomials: list[Poly],
    learned_abstraction: Any,
) -> tuple[Any, Any]:
    """Keep the exact nominal drift and use the learned model only for G."""
    system = replace(
        nominal_system,
        name=f"{nominal_system.name}_diffusion_noise",
        process_noise=0.0,
        drift_polynomials=tuple(nominal_polynomials),
    )
    abstraction = copy.deepcopy(learned_abstraction)
    abstraction.residual_coeffs = np.zeros_like(abstraction.residual_coeffs)
    abstraction.fit_error = 0.0
    abstraction.epsilon_r = 0.0
    abstraction.extra_error_margin = 0.0
    abstraction.partition_boxes = []
    abstraction.local_residual_coeffs = []
    abstraction.local_fit_errors = []
    certificate = dict(abstraction.abstraction_certificate)
    certificate.update(
        {
            "sos_center_drift": "exact_nominal_polynomial",
            "sos_center_refit_error": 0.0,
            "learned_physical_component": "diffusion_covariance_only",
            "nominal_drift_preserved": True,
            "surrogate_error_semantics": "paper_fixed",
            "diffusion_covariance_is_exact": True,
            "drift_polynomial_is_exact": True,
            "model_to_real_error_in_generator": False,
            "statistical_transfer_to_real_system_certified": False,
            "verification_claim": "formal_certificate_for_fixed_surrogate_model",
        }
    )
    abstraction.abstraction_certificate = certificate
    return system, abstraction


def train_case(
    case_id: str,
    example: Any,
    nominal_polynomials: list[Poly],
    drift_degree: int,
    args: Any,
    case_output: Path,
) -> tuple[Any, Any, float]:
    """Generate trajectories, fit the diffusion proxy, and prepare SOS dynamics."""
    stochastic_index = resolve_stochastic_state_index(
        int(example.n), int(args.stochastic_state_index)
    )
    nominal_system = replace(
        build_system(example, nominal_polynomials),
        noise_state_indices=(stochastic_index,),
    )
    noisy_system = replace(
        nominal_system, process_noise=float(args.data_noise_amplitude)
    )
    config = training_config(case_id, nominal_system.state_dim, drift_degree, args)

    trajectories = generate_trajectories(
        noisy_system, config.n_trajectories, config.horizon, config.seed
    )
    samples = build_samples(trajectories, noisy_system, config.history)
    minimum_count = max(16, config.n_trajectories)
    if int(samples["stochastic_increment"].shape[0]) < minimum_count:
        config = replace(config, history=1)
        samples = build_samples(trajectories, noisy_system, config.history)
    if int(samples["stochastic_increment"].shape[0]) < minimum_count:
        raise RuntimeError(
            f"{case_id} has only {samples['stochastic_increment'].shape[0]} "
            "unclipped training transitions"
        )

    training_output = case_output / "diffusion_training"
    model_output = training_output / "model"
    model_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(training_output / "dataset.npz", **samples)

    started = time.perf_counter()
    train_pipeline(
        samples,
        config,
        training_output,
        dataset_path=Path("dataset.npz"),
        save_artifacts=True,
        system=noisy_system,
    )
    with (model_output / "polynomial_abstraction.pkl").open("rb") as handle:
        learned_abstraction = pickle.load(handle)
    system, abstraction = verification_center_model(
        nominal_system, nominal_polynomials, learned_abstraction
    )
    abstraction.abstraction_certificate["physical_stochastic_state_indices"] = [
        stochastic_index
    ]
    with (model_output / "polynomial_abstraction.pkl").open("wb") as handle:
        pickle.dump(abstraction, handle)
    return system, abstraction, time.perf_counter() - started


def unavailable_row(
    *,
    case_id: str,
    source_case: str,
    state_dim: int,
    drift_degree: int,
    degree: int,
    relaxation: int,
    status: str,
    error: str,
    training_time: float = 0.0,
) -> dict[str, Any]:
    return {
        "example": case_id,
        "source_case": source_case,
        "method": METHOD_NAME,
        "state_dim": int(state_dim),
        "drift_degree": int(drift_degree),
        "barrier_degree": int(degree),
        "relaxation_degree": int(relaxation),
        **problem_size(state_dim, degree, relaxation),
        "status": status,
        "solver_status": "not_started" if status != "error" else "error",
        "success": False,
        "nontrivial": False,
        "raw_rho": 1.0,
        "reported_rho": 1.0,
        "safety_lower_bound": 0.0,
        "solve_time_seconds": 0.0,
        "training_time_seconds": float(training_time),
        "end_to_end_time_seconds": float(training_time),
        "error": str(error),
    }


def solve_candidate(
    *,
    case_id: str,
    source_case: str,
    system: Any,
    abstraction: Any,
    drift_degree: int,
    degree: int,
    training_time: float,
    refinement: bool,
    refinement_tolerance: float,
) -> dict[str, Any]:
    relaxation = relaxation_degree(degree, drift_degree)
    base_config = certificate_config(system.state_dim, degree, relaxation)
    config = replace(
        base_config,
        sos_constraint_tolerance=(
            float(refinement_tolerance)
            if refinement
            else base_config.sos_constraint_tolerance
        ),
        sos_gram_eigenvalue_tolerance=(
            float(refinement_tolerance)
            if refinement
            else base_config.sos_gram_eigenvalue_tolerance
        ),
    )
    started = time.perf_counter()
    result = SosBarrierSynthesizer(abstraction, system, config).fit()
    solve_time = time.perf_counter() - started
    result.verification_constraint_tolerance = float(config.sos_constraint_tolerance)
    result.verification_gram_tolerance = float(config.sos_gram_eigenvalue_tolerance)
    result.numerical_refinement_applied = bool(refinement)

    raw_rho = float(result.gamma if result.gamma is not None else result.rho)
    reported_rho = min(1.0, raw_rho + 1e-3) if result.success else 1.0
    return {
        "example": case_id,
        "source_case": source_case,
        "method": METHOD_NAME,
        "state_dim": int(system.state_dim),
        "drift_degree": int(drift_degree),
        "barrier_degree": int(degree),
        "relaxation_degree": int(relaxation),
        **problem_size(system.state_dim, degree, relaxation),
        "status": "verified" if result.success else "posterior_check_failed",
        "solver_status": str(result.solver_status),
        "success": bool(result.success),
        "nontrivial": bool(result.success and reported_rho < 1.0 - 1e-12),
        "raw_rho": raw_rho,
        "reported_rho": float(reported_rho),
        "safety_lower_bound": float(max(0.0, 1.0 - reported_rho)),
        "solve_time_seconds": float(solve_time),
        "training_time_seconds": float(training_time),
        "end_to_end_time_seconds": float(training_time + solve_time),
        "sdp_max_constraint_violation": float(result.sdp_max_constraint_violation),
        "sdp_min_gram_eigenvalue": float(result.sdp_min_gram_eigenvalue),
        "max_initial_violation": float(result.max_initial_violation),
        "max_unsafe_violation": float(result.max_unsafe_violation),
        "max_positive_violation": float(result.max_positive_violation),
        "max_generator_violation": float(result.max_generator_violation),
        "barrier_coefficients": np.asarray(result.coeffs, dtype=float).tolist(),
        "barrier_exponents": [list(exponent) for exponent in result.exponents],
        "epsilon_r": float(abstraction.epsilon_r),
        "Gbar": np.asarray(abstraction.Gbar, dtype=float).tolist(),
        "diffusion_upper_bound_type": str(abstraction.diffusion_upper_bound_type),
        "abstraction_certificate": dict(abstraction.abstraction_certificate),
        "numerical_refinement_applied": bool(refinement),
        "verification_constraint_tolerance": float(config.sos_constraint_tolerance),
        "verification_gram_tolerance": float(config.sos_gram_eigenvalue_tolerance),
    }


def best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    verified = [row for row in rows if bool(row.get("success", False))]
    if not verified:
        return None
    return min(
        verified,
        key=lambda row: (
            float(row["reported_rho"]), float(row["solve_time_seconds"])
        ),
    )


def write_results(output: Path, rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    """Write the authoritative JSON, complete candidates, and best-case table."""
    output.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows,
        key=lambda row: (int(str(row["example"])[1:]), int(row["barrier_degree"])),
    )
    payload = {
        "protocol": RESULT_PROTOCOL,
        "method": METHOD_NAME,
        "case_ids": list(settings["case_ids"]),
        "degrees": list(settings["degrees"]),
        "selection_rule": "minimum reported rho, then minimum solve time",
        "initial_condition_semantics": "worst_case",
        "verification_claim": "formal certificate for the fixed learned surrogate model",
        "settings": settings,
        "candidates": ordered,
    }
    (output / "results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    if ordered:
        fields = sorted({key for row in ordered for key in row})
        with (output / "candidate_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(ordered)

    lines = [
        "# Main-Method Paper Results",
        "",
        "| Bench | Internal ID | Source case | Degree | rho | Safety lower bound | Certificate time (s) | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for case_id in settings["case_ids"]:
        case_rows = [row for row in ordered if row["example"] == case_id]
        best = best_candidate(case_rows)
        source = PAPER_SOURCE_LABELS.get(case_id, str(case_rows[0]["source_case"])) if case_rows else "--"
        if best is None:
            statuses = ",".join(sorted({str(row["status"]) for row in case_rows}))
            lines.append(f"| {PAPER_CASE_NAMES.get(case_id, case_id)} | {case_id} | {source} | -- | 1 | 0 | -- | {statuses} |")
        else:
            lines.append(
                f"| {PAPER_CASE_NAMES.get(case_id, case_id)} | {case_id} | {source} | {best['barrier_degree']} | "
                f"{best['reported_rho']:.9g} | {best['safety_lower_bound']:.9g} | "
                f"{best['solve_time_seconds']:.6g} | {best['status']} |"
            )
    (output / "minimum_rho_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
