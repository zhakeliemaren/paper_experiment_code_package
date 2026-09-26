"""Paper benchmark registry and exact polynomial-system adapters."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from diffusion_sbc.benchmarks import BenchmarkSystem
from diffusion_sbc.config import ExperimentConfig
from diffusion_sbc.sos import Poly, evaluate_poly


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_BENCHMARK_ROOT = PROJECT_ROOT / "third_party" / "synncb"
DEFAULT_EXTERNAL_BENCHMARK_ROOT = PROJECT_ROOT / "external_benchmarks"
PAPER_CASE_IDS = tuple(f"C{index}" for index in range(1, 10))

# Stable internal IDs are retained for result-file compatibility. These are
# the presentation names used by the manuscript and its nine-case tables.
PAPER_CASE_NAMES = {
    "C1": "C1", "C2": "C2", "C3": "C3",
    "C4": "C4", "C5": "C5", "C6": "C6", "C7": "C7",
    "C8": "C8", "C9": "C9",
}

PAPER_CASE_DESCRIPTIONS = {
    "C1": "two-dimensional quadratic Arch benchmark",
    "C2": "two-dimensional controlled Duffing Example 4.3",
    "C3": "two-dimensional quadratic oscillatory benchmark",
    "C4": "two-dimensional quadratic Van der Pol variant 1",
    "C5": "three-dimensional cubic Van der Pol variant 2",
    "C6": "seven-dimensional quadratic Lie-derivative benchmark",
    "C7": "nine-dimensional quadratic equilibrium benchmark",
    "C8": "three-dimensional cubic Lyapunov benchmark",
    "C9": "five-dimensional cubic Lotka benchmark",
}

PAPER_SOURCE_LABELS = {"C2": "Example 4.3 (Duffing_P1)"}


def _load_registry(root: Path, module_name: str) -> dict[str, Any]:
    """Load a benchmark registry indexed by its declared example names."""
    source = Path(root) / "benchmarks" / "Exampler_B.py"
    if not source.exists():
        raise FileNotFoundError(f"Benchmark registry not found: {source}")
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load paper benchmark registry: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {str(example.name): example for example in module.examples.values()}


def load_paper_examples(root: Path = DEFAULT_BENCHMARK_ROOT) -> dict[str, Any]:
    """Load paper cases, exposing external Duffing Example 4.3 as C2."""
    examples = _load_registry(root, "paper_benchmark_registry")
    missing = sorted(set(PAPER_CASE_IDS) - set(examples))
    if missing:
        raise ValueError(f"Paper benchmark cases are missing: {missing}")
    unexpected = sorted(set(examples) - set(PAPER_CASE_IDS))
    if unexpected:
        raise ValueError(f"Registry contains cases not used by the paper: {unexpected}")
    if Path(root).resolve() == DEFAULT_BENCHMARK_ROOT.resolve():
        external = _load_registry(DEFAULT_EXTERNAL_BENCHMARK_ROOT, "external_benchmark_registry")
        if "D1" not in external:
            raise ValueError("External benchmark registry does not define D1")
        examples["C2"] = external["D1"]
    return {case_id: examples[case_id] for case_id in PAPER_CASE_IDS}


def load_examples(root: Path = DEFAULT_BENCHMARK_ROOT) -> dict[str, Any]:
    """Load any bundled registry without imposing the manuscript C1--C9 set."""
    source = Path(root) / "benchmarks" / "Exampler_B.py"
    examples = _load_registry(root, "experiment_benchmark_registry")
    if not examples:
        raise ValueError(f"Benchmark registry is empty: {source}")
    return examples


def symbolic_drift(example: Any) -> tuple[list[Poly], int]:
    """Convert the source callable dynamics to exact polynomial dictionaries."""
    import sympy as sp

    symbols = sp.symbols(f"x0:{int(example.n)}")
    polynomials: list[Poly] = []
    maximum_degree = 0
    for component in example.f:
        polynomial = sp.Poly(sp.expand(component(symbols)), *symbols)
        maximum_degree = max(maximum_degree, int(polynomial.total_degree()))
        polynomials.append(
            {
                tuple(int(power) for power in monomial): float(coefficient)
                for monomial, coefficient in polynomial.terms()
            }
        )
    return polynomials, maximum_degree


@dataclass
class PaperPolynomialSystem(BenchmarkSystem):
    drift_polynomials: tuple[Poly, ...] = ()
    noise_state_indices: tuple[int, ...] | None = None

    def stochastic_state_indices(self) -> tuple[int, ...]:
        if self.noise_state_indices is None:
            return super().stochastic_state_indices()
        return tuple(int(index) for index in self.noise_state_indices)

    def sample_training_initial_states(
        self, rng: np.random.Generator, n: int
    ) -> np.ndarray:
        lows = np.asarray([bound[0] for bound in self.verify_box], dtype=float)
        highs = np.asarray([bound[1] for bound in self.verify_box], dtype=float)
        return rng.uniform(lows, highs, size=(int(n), self.state_dim))

    def control_policy(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        return np.zeros((1,), dtype=float)

    def control_policy_batch(self, x: np.ndarray, step: int = 0) -> np.ndarray:
        return np.zeros((np.atleast_2d(x).shape[0], 1), dtype=float)

    def exact_derivative(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        values = np.column_stack(
            [evaluate_poly(poly, x) for poly in self.drift_polynomials]
        )
        return values[0] if np.asarray(x).ndim == 1 else values

    def exact_drift_polynomials(self) -> list[Poly]:
        return [dict(poly) for poly in self.drift_polynomials]

    def accepts_training_transition(
        self, x: np.ndarray, nominal_next: np.ndarray, next_state: np.ndarray
    ) -> bool:
        """Exclude clipped transitions because clipping is not process noise."""
        del x
        nominal = np.asarray(nominal_next, dtype=float)
        observed = np.asarray(next_state, dtype=float)
        tolerance = 1e-10
        for coordinate, (lower, upper) in enumerate(self.verify_box):
            if nominal[coordinate] < lower or nominal[coordinate] > upper:
                return False
            if observed[coordinate] <= lower + tolerance:
                return False
            if observed[coordinate] >= upper - tolerance:
                return False
        return True


def build_system(example: Any, drift_polynomials: list[Poly]) -> PaperPolynomialSystem:
    return PaperPolynomialSystem(
        name=str(example.name),
        dt=0.05,
        state_dim=int(example.n),
        control_dim=1,
        process_noise=0.0,
        verify_box=tuple(
            (float(low), float(high))
            for low, high in np.asarray(example.D_zones, dtype=float)
        ),
        initial_box=tuple(
            (float(low), float(high))
            for low, high in np.asarray(example.I_zones, dtype=float)
        ),
        unsafe_box=tuple(
            (float(low), float(high))
            for low, high in np.asarray(example.U_zones, dtype=float)
        ),
        control_bounds=((0.0, 0.0),),
        barrier_box=None,
        drift_polynomials=tuple(drift_polynomials),
    )


def relaxation_degree(barrier_degree: int, drift_degree: int) -> int:
    generator_degree = max(0, int(barrier_degree) - 1) + int(drift_degree)
    target = max(int(barrier_degree), generator_degree)
    even_target = target if target % 2 == 0 else target + 1
    return max(2, even_target)


def problem_size(
    state_dim: int, barrier_degree: int, relaxation: int
) -> dict[str, int]:
    half = int(relaxation) // 2
    multiplier_half = max(0, (int(relaxation) - 2) // 2)
    gram_basis = math.comb(int(state_dim) + half, half)
    multiplier_basis = math.comb(int(state_dim) + multiplier_half, multiplier_half)
    per_condition = gram_basis * (gram_basis + 1) // 2
    per_condition += (
        int(state_dim) * multiplier_basis * (multiplier_basis + 1) // 2
    )
    return {
        "barrier_coefficients": math.comb(
            int(state_dim) + int(barrier_degree), int(barrier_degree)
        ),
        "gram_basis": gram_basis,
        "multiplier_gram_basis": multiplier_basis,
        "estimated_psd_scalar_variables": 4 * per_condition,
        "estimated_coefficient_equalities": 4
        * math.comb(int(state_dim) + int(relaxation), int(relaxation)),
    }


def certificate_config(
    state_dim: int, barrier_degree: int, relaxation: int
) -> ExperimentConfig:
    """Return the continuous-time Generator-SOS settings used in the paper."""
    return replace(
        ExperimentConfig(),
        benchmark="paper_main_experiment",
        state_dim=int(state_dim),
        control_dim=1,
        process_noise=0.0,
        barrier_degree=int(barrier_degree),
        barrier_samples=300,
        counterexample_rounds=0,
        barrier_formulation="prajna_optimization",
        initial_condition_mode="worst_case",
        sos_relaxation_degree=int(relaxation),
        sos_abstraction_error_mode="off",
        sos_margin=1e-6,
        sos_rho_upper=1.0,
        sos_regularization=1e-7,
        sos_sample_tolerance=1e-3,
        sos_constraint_tolerance=1e-7,
        sos_gram_eigenvalue_tolerance=1e-7,
        sbc_c=0.0,
        sbc_time_horizon=0.0,
    )
