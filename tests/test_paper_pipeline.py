from __future__ import annotations

import unittest

import numpy as np

from diffusion_sbc.polynomial import PolynomialAbstraction
from paper_benchmarks import PAPER_CASE_IDS, build_system, load_paper_examples, symbolic_drift
from paper_pipeline import best_candidate, resolve_stochastic_state_index, verification_center_model


class PaperPipelineTests(unittest.TestCase):
    def test_registry_contains_only_manuscript_cases(self) -> None:
        examples = load_paper_examples()
        self.assertEqual(tuple(examples), PAPER_CASE_IDS)
        self.assertEqual(examples["C4"].source_name, "C5")
        self.assertEqual(examples["C6"].source_name, "C12")
        self.assertEqual(examples["C9"].source_name, "R2")

    def test_symbolic_dynamics_preserve_declared_dimensions(self) -> None:
        for example in load_paper_examples().values():
            polynomials, degree = symbolic_drift(example)
            self.assertEqual(len(polynomials), int(example.n))
            self.assertGreaterEqual(degree, 1)

    def test_negative_stochastic_index_selects_last_coordinate(self) -> None:
        self.assertEqual(resolve_stochastic_state_index(7, -1), 6)
        with self.assertRaises(ValueError):
            resolve_stochastic_state_index(2, 2)

    def test_verification_model_keeps_nominal_drift_and_covariance(self) -> None:
        example = load_paper_examples()["C1"]
        drift, _ = symbolic_drift(example)
        system = build_system(example, drift)
        abstraction = PolynomialAbstraction(
            exponents=[(0, 0)],
            residual_coeffs=np.ones((1, 2)),
            epsilon_r=3.0,
            Gbar=np.diag([0.0, 0.02]),
            gbar=np.diag([0.0, np.sqrt(0.02)]),
            dt=0.05,
            source="test",
            diffusion_upper_bound_type="test",
            fit_error=2.0,
        )
        centered_system, centered = verification_center_model(system, drift, abstraction)
        self.assertEqual(centered_system.exact_drift_polynomials(), drift)
        np.testing.assert_allclose(centered.residual_coeffs, 0.0)
        np.testing.assert_allclose(centered.Gbar, abstraction.Gbar)
        self.assertEqual(centered.epsilon_r, 0.0)

    def test_best_candidate_uses_rho_then_time(self) -> None:
        rows = [
            {"success": True, "reported_rho": 0.2, "solve_time_seconds": 1.0},
            {"success": True, "reported_rho": 0.1, "solve_time_seconds": 5.0},
            {"success": False, "reported_rho": 0.0, "solve_time_seconds": 0.1},
        ]
        self.assertIs(best_candidate(rows), rows[1])


if __name__ == "__main__":
    unittest.main()
