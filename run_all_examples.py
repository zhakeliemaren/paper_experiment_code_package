"""Run the proposed-method pipeline on every SynNBC registry example.

This driver is a generalization of ``run_paper_experiments.py``.  Instead of the
nine manuscript cases C1--C9, it loads the *complete* SynNBC registry
(C1--C16, F1--F5, R1--R9, and the 7-dimensional test case) and runs training
plus the continuous Generator-SOS certificate pipeline for barrier degrees
2, 4, and 6.

For each (case, degree) pair it records one of the following statuses:

* ``verified`` -- a nontrivial or trivial certificate passed all posterior checks;
* ``resource_limit`` -- the SOS/Gram expansion for that dimension and degree
  exceeds ``--max-psd-scalars`` (this is the "dimension explosion" signal);
* ``training_data_unavailable`` -- too few unclipped transitions survived the
  common data protocol;
* ``error`` -- a runtime failure during synthesis.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from paper_benchmarks import (
    build_system,
    problem_size,
    relaxation_degree,
    symbolic_drift,
)
from paper_pipeline import (
    METHOD_NAME,
    RESULT_PROTOCOL,
    solve_candidate,
    train_case,
    unavailable_row,
)

PROJECT_ROOT = Path(__file__).resolve().parent
FULL_REGISTRY = PROJECT_ROOT / "outputs" / "_synnbc_full" / "Exampler_B_full.py"


def name_key(name: str) -> tuple[int, int]:
    """Sort names as C*, F*, R*, then any other identifier (e.g. test_7dim)."""
    letter = name[0].upper()
    rank = {"C": 0, "F": 1, "R": 2}.get(letter, 3)
    digits = int(re.sub(r"\D", "", name) or 0)
    return rank, digits


def load_full_examples() -> dict[str, Any]:
    if not FULL_REGISTRY.exists():
        raise FileNotFoundError(
            f"Full SynNBC registry not found: {FULL_REGISTRY}. "
            "Download it from https://github.com/tete0602/SynNBC "
            "(benchmarks/Exampler_B.py)."
        )
    spec = importlib.util.spec_from_file_location("synnbc_full_registry", FULL_REGISTRY)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load the full SynNBC registry")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = {str(example.name): example for example in module.examples.values()}
    return dict(sorted(examples.items(), key=lambda item: name_key(item[0])))


def build_args(argv: argparse.Namespace) -> argparse.Namespace:
    """Populate every attribute used by train_case / training_config."""
    argv.stochastic_state_index = -1
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "all_examples")
    parser.add_argument("--degrees", nargs="+", type=int, default=[2, 4, 6])
    parser.add_argument("--max-psd-scalars", type=int, default=12000)
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the dimension/PSD-size table without running training or solving.")
    return build_args(parser.parse_args())


def size_table(examples: dict[str, Any], degrees: list[int], budget: int) -> None:
    print(f"{'case':<10} {'n':>3} {'d_drift':>7} | " +
          " | ".join(f"deg {d} (psd)" for d in degrees))
    print("-" * 100)
    for name, example in examples.items():
        try:
            _, drift_degree = symbolic_drift(example)
        except Exception as exc:
            print(f"{name:<10} {example.n:>3} {'--':>7} | symbolic drift failed: {exc}")
            continue
        cells = []
        for degree in degrees:
            relaxation = relaxation_degree(degree, drift_degree)
            size = problem_size(example.n, degree, relaxation)
            psd = size["estimated_psd_scalar_variables"]
            marker = "OVER" if psd > budget else "ok"
            cells.append(f"{degree}({psd}){marker}")
        print(f"{name:<10} {example.n:>3} {drift_degree:>7} | " + " | ".join(cells))
    print("-" * 100)
    print(f"budget max_psd_scalars = {budget}; 'OVER' marks the dimension-explosion precheck.")


def write_output(output: Path, rows: list[dict[str, Any]], settings: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": RESULT_PROTOCOL,
        "method": METHOD_NAME,
        "case_ids": list(settings["case_ids"]),
        "degrees": list(settings["degrees"]),
        "selection_rule": "minimum reported rho, then minimum solve time",
        "initial_condition_semantics": "worst_case",
        "verification_claim": "formal certificate for the fixed learned surrogate model",
        "settings": settings,
        "candidates": rows,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if rows:
        fields = sorted({key for row in rows for key in row})
        with (output / "candidate_results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# All-SynNBC-Cases Results",
        "",
        "| Case | n | drift deg | Degree | est PSD vars | rho | Safety bound | Status |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for case_id in settings["case_ids"]:
        case_rows = [row for row in rows if row["example"] == case_id]
        for row in sorted(case_rows, key=lambda r: int(r["barrier_degree"])):
            status = str(row["status"])
            rho = "--" if not row.get("success") else f"{row['reported_rho']:.6g}"
            safety = "--" if not row.get("success") else f"{row['safety_lower_bound']:.6g}"
            lines.append(
                f"| {case_id} | {row['state_dim']} | {row['drift_degree']} | "
                f"{row['barrier_degree']} | {row['estimated_psd_scalar_variables']} | "
                f"{rho} | {safety} | {status} |"
            )
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    degrees = [int(d) for d in args.degrees]
    if any(d not in {2, 4, 6} for d in degrees):
        raise ValueError("Barrier degrees must be selected from 2, 4, and 6")
    examples = load_full_examples()
    case_ids = list(examples.keys())

    if args.dry_run:
        size_table(examples, degrees, int(args.max_psd_scalars))
        return 0

    output = args.output.resolve()
    print(f"Running pipeline over {len(case_ids)} cases: {', '.join(case_ids)}")
    print(f"  degrees: {degrees}; max_psd_scalars={args.max_psd_scalars}; output={output}")
    rows: list[dict[str, Any]] = []

    for case_id in case_ids:
        example = examples[case_id]
        n = int(example.n)
        print(f"\n[{case_id}] n={n} training ...", flush=True)
        try:
            nominal_polynomials, drift_degree = symbolic_drift(example)
        except Exception as exc:
            drift_degree = 0
            nominal_polynomials = []
            system = None
            abstraction = None
            training_time = 0.0
            training_error = f"symbolic_drift: {type(exc).__name__}: {exc}"
            print(f"[{case_id}] symbolic drift failed: {exc}", flush=True)
        else:
            try:
                system, abstraction, training_time = train_case(
                    case_id, example, nominal_polynomials, drift_degree,
                    args, output / case_id,
                )
                training_error = ""
            except (RuntimeError, ValueError, FileNotFoundError) as exc:
                system = None
                abstraction = None
                training_time = 0.0
                training_error = f"{type(exc).__name__}: {exc}"
                print(f"[{case_id}] training unavailable: {exc}", flush=True)

        for degree in degrees:
            relaxation = relaxation_degree(degree, drift_degree)
            size = problem_size(n, degree, relaxation)
            psd = size["estimated_psd_scalar_variables"]
            if system is None or abstraction is None:
                row = unavailable_row(
                    case_id=case_id, source_case=case_id, state_dim=n,
                    drift_degree=drift_degree, degree=degree, relaxation=relaxation,
                    status="training_data_unavailable", error=training_error,
                    training_time=training_time,
                )
            elif psd > int(args.max_psd_scalars):
                row = unavailable_row(
                    case_id=case_id, source_case=case_id, state_dim=n,
                    drift_degree=drift_degree, degree=degree, relaxation=relaxation,
                    status="resource_limit", error="continuous SOS PSD-variable precheck",
                    training_time=training_time,
                )
            else:
                print(f"[{case_id}] solving degree {degree} (psd={psd})", flush=True)
                started = time.perf_counter()
                try:
                    row = solve_candidate(
                        case_id=case_id, source_case=case_id, system=system,
                        abstraction=abstraction, drift_degree=drift_degree,
                        degree=degree, training_time=training_time,
                        refinement=False, refinement_tolerance=2e-6,
                    )
                except Exception as exc:
                    row = unavailable_row(
                        case_id=case_id, source_case=case_id, state_dim=n,
                        drift_degree=drift_degree, degree=degree, relaxation=relaxation,
                        status="error", error=f"{type(exc).__name__}: {exc}",
                        training_time=training_time,
                    )
                print(f"[{case_id}] degree {degree} -> {row['status']} "
                      f"rho={row.get('reported_rho', '--')} "
                      f"({time.perf_counter() - started:.2f}s)", flush=True)
            rows.append(row)

    settings = {
        "case_ids": case_ids,
        "degrees": degrees,
        "seed": int(args.seed),
        "max_psd_scalars": int(args.max_psd_scalars),
        "data_noise_amplitude": float(args.data_noise_amplitude),
        "n_trajectories": int(args.n_trajectories),
        "horizon": int(args.horizon),
        "history": int(args.history),
    }
    write_output(output, rows, settings)
    print(f"\nDone. Wrote {len(rows)} candidate rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
