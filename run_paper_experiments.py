"""Reproduce the proposed-method experiments reported in the paper."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper_benchmarks import (
    DEFAULT_BENCHMARK_ROOT,
    PAPER_CASE_IDS,
    load_paper_examples,
    problem_size,
    relaxation_degree,
    symbolic_drift,
)
from paper_pipeline import (
    RESULT_PROTOCOL,
    solve_candidate,
    train_case,
    unavailable_row,
    write_results,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "paper_experiments"
DEFAULT_ARTIFACTS = PROJECT_ROOT / "paper_artifacts"


def _project_path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _portable_path(value: Path) -> str:
    try:
        return str(value.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(value)


def _package_versions() -> dict[str, str]:
    distributions = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit-learn": "scikit-learn",
        "sympy": "sympy",
        "cvxpy": "cvxpy",
        "Mosek": "Mosek",
        "Pillow": "Pillow",
    }
    versions = {"python": sys.version.split()[0]}
    missing: list[str] = []
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            missing.append(distribution)
    if missing:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + f". Install them with: {sys.executable} -m pip install -r requirements.txt"
        )
    return versions


def _artifact_readme(artifacts: Path) -> None:
    text = """# Generated Main-Method Artifacts

The files in this directory are generated from `outputs/paper_experiments/results.json`.

- `paper_main_table_rows.tex`: best verified result for each manuscript case.
- `paper_main_c6_sbc.tex`: selected C6 stochastic barrier polynomial.
- `paper_main_benchmark_summary.png`: safety lower bounds for C1--C9.
- `paper_main_c6_degree_sweep.png`: C6 results at barrier degrees 2, 4, and 6.
- `run_manifest.json`: runtime versions and experiment selection.
"""
    (artifacts / "README.md").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--examples", nargs="+", default=list(PAPER_CASE_IDS))
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--case-study", default="C6")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-noise-amplitude", type=float, default=0.03)
    parser.add_argument("--n-trajectories", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--diffusion-train-copies", type=int, default=8)
    parser.add_argument("--diffusion-max-iter", type=int, default=300)
    parser.add_argument("--moment-conditions", type=int, default=32)
    parser.add_argument("--polynomial-conditions", type=int, default=80)
    parser.add_argument("--stochastic-state-index", type=int, default=-1)
    parser.add_argument("--max-psd-scalars", type=int, default=12000)
    parser.add_argument("--refinement-case", default="C4")
    parser.add_argument("--refinement-degree", type=int, default=6)
    parser.add_argument("--refinement-tolerance", type=float, default=2e-6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run manuscript case C6 at degree 2 in separate smoke directories.",
    )
    return parser


def _settings(args: argparse.Namespace, case_ids: list[str], degrees: list[int]) -> dict[str, Any]:
    return {
        "case_ids": case_ids,
        "degrees": degrees,
        "seed": int(args.seed),
        "dt": 0.05,
        "data_noise_amplitude": float(args.data_noise_amplitude),
        "n_trajectories": int(args.n_trajectories),
        "horizon": int(args.horizon),
        "history": int(args.history),
        "train_test_split": "80/20 grouped by trajectory",
        "stochastic_state_index": int(args.stochastic_state_index),
        "stochastic_dimension": 1,
        "diffusion_steps": int(args.diffusion_steps),
        "diffusion_train_copies": int(args.diffusion_train_copies),
        "diffusion_max_iter": int(args.diffusion_max_iter),
        "diffusion_batch_size": 256,
        "diffusion_learning_rate": 5e-4,
        "diffusion_hidden_widths": [64, 64],
        "reverse_moment_method": "reverse_polynomial",
        "reverse_polynomial_degree": 2,
        "moment_conditions": int(args.moment_conditions),
        "polynomial_conditions": int(args.polynomial_conditions),
        "initial_condition_semantics": "worst_case",
        "reporting_rho_allowance": 1e-3,
        "max_psd_scalars": int(args.max_psd_scalars),
        "refinement_case": str(args.refinement_case).upper(),
        "refinement_degree": int(args.refinement_degree),
        "refinement_tolerance": float(args.refinement_tolerance),
    }


def main() -> int:
    args = build_parser().parse_args()
    versions = _package_versions()
    examples = load_paper_examples(DEFAULT_BENCHMARK_ROOT)

    if args.smoke:
        case_ids = ["C6"]
        degrees = [2]
        output = PROJECT_ROOT / "outputs" / "smoke"
        artifacts = PROJECT_ROOT / "paper_artifacts" / "smoke"
    else:
        case_ids = [str(value).upper() for value in args.examples]
        degrees = [int(value) for value in args.degrees]
        output = _project_path(args.output)
        artifacts = _project_path(args.artifacts)
    unknown = sorted(set(case_ids) - set(PAPER_CASE_IDS))
    if unknown:
        raise ValueError(f"Unknown manuscript cases: {unknown}")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Duplicate manuscript case identifiers are not allowed")
    if any(degree not in {2, 4, 6} for degree in degrees):
        raise ValueError("Barrier degrees must be selected from 2, 4, and 6")
    case_study = "C6" if args.smoke else str(args.case_study).upper()
    if case_study not in case_ids:
        raise ValueError("The case-study identifier must be included in --examples")

    settings = _settings(args, case_ids, degrees)
    print("Paper main-method experiment")
    print(f"  cases: {', '.join(case_ids)}")
    print(f"  barrier degrees: {', '.join(map(str, degrees))}")
    print(f"  output: {output}")
    if args.dry_run:
        print(json.dumps(settings, indent=2))
        return 0

    result_path = output / "results.json"
    rows: list[dict[str, Any]] = []
    if args.resume and result_path.exists():
        previous = json.loads(result_path.read_text(encoding="utf-8"))
        if str(previous.get("protocol", "")) != RESULT_PROTOCOL:
            raise RuntimeError("Cannot resume results from a different protocol")
        rows = list(previous.get("candidates", []))
    completed = {
        (str(row["example"]), int(row["barrier_degree"])) for row in rows
    }

    for case_id in case_ids:
        if all((case_id, degree) in completed for degree in degrees):
            print(f"skipping completed case {case_id}", flush=True)
            continue
        example = examples[case_id]
        source_case = str(getattr(example, "source_name", case_id))
        nominal_polynomials, drift_degree = symbolic_drift(example)
        print(
            f"training {case_id} (SynNBC {source_case}, n={example.n})",
            flush=True,
        )
        try:
            system, abstraction, training_time = train_case(
                case_id,
                example,
                nominal_polynomials,
                drift_degree,
                args,
                output / case_id,
            )
            training_error = ""
        except (RuntimeError, ValueError, FileNotFoundError) as exc:
            system = None
            abstraction = None
            training_time = 0.0
            training_error = f"{type(exc).__name__}: {exc}"
            print(f"training unavailable for {case_id}: {exc}", flush=True)

        for degree in degrees:
            if (case_id, degree) in completed:
                continue
            relaxation = relaxation_degree(degree, drift_degree)
            if system is None or abstraction is None:
                row = unavailable_row(
                    case_id=case_id,
                    source_case=source_case,
                    state_dim=int(example.n),
                    drift_degree=drift_degree,
                    degree=degree,
                    relaxation=relaxation,
                    status="training_data_unavailable",
                    error=training_error,
                )
            elif problem_size(system.state_dim, degree, relaxation)[
                "estimated_psd_scalar_variables"
            ] > int(args.max_psd_scalars):
                row = unavailable_row(
                    case_id=case_id,
                    source_case=source_case,
                    state_dim=system.state_dim,
                    drift_degree=drift_degree,
                    degree=degree,
                    relaxation=relaxation,
                    status="resource_limit",
                    error="continuous SOS PSD-variable precheck",
                    training_time=training_time,
                )
            else:
                print(f"solving {case_id}: barrier degree {degree}", flush=True)
                try:
                    row = solve_candidate(
                        case_id=case_id,
                        source_case=source_case,
                        system=system,
                        abstraction=abstraction,
                        drift_degree=drift_degree,
                        degree=degree,
                        training_time=training_time,
                        refinement=(
                            case_id == str(args.refinement_case).upper()
                            and degree == int(args.refinement_degree)
                        ),
                        refinement_tolerance=float(args.refinement_tolerance),
                    )
                except Exception as exc:
                    row = unavailable_row(
                        case_id=case_id,
                        source_case=source_case,
                        state_dim=system.state_dim,
                        drift_degree=drift_degree,
                        degree=degree,
                        relaxation=relaxation,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        training_time=training_time,
                    )
            rows.append(row)
            completed.add((case_id, degree))
            write_results(output, rows, settings)

    write_results(output, rows, settings)
    if not args.skip_export:
        artifacts.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "export_paper_artifacts.py"),
                "--results",
                str(result_path),
                "--case",
                case_study,
                "--benchmark-root",
                str(DEFAULT_BENCHMARK_ROOT),
                "--out",
                str(artifacts),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
        _artifact_readme(artifacts)
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": RESULT_PROTOCOL,
            "results": _portable_path(output),
            "artifacts": _portable_path(artifacts),
            "settings": settings,
            "package_versions": versions,
        }
        (artifacts / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
    print(f"results: {result_path}")
    print(f"summary: {output / 'minimum_rho_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
