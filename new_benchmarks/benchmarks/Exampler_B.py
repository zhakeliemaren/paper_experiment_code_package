"""Moderate-dimensional polynomial benchmarks inspired by control examples.

The systems are autonomous closed-loop surrogates.  They deliberately use
polynomial dynamics so the existing symbolic drift and Generator-SOS pipeline
can consume them without a trigonometric approximation layer.
"""

from __future__ import annotations

import numpy as np


class Example:
    def __init__(self, n, D_zones, I_zones, U_zones, f, name, source_name):
        if not (len(D_zones) == len(I_zones) == len(U_zones) == len(f) == n):
            raise ValueError("Benchmark dimensions are inconsistent")
        self.n = n
        self.D_zones = np.asarray(D_zones, dtype=float)
        self.I_zones = np.asarray(I_zones, dtype=float)
        self.U_zones = np.asarray(U_zones, dtype=float)
        self.f = f
        self.name = name
        self.source_name = source_name


examples = {
    # Linearized closed-loop CartPole-style oscillator.  The two second-order
    # modes are damped and the unsafe set is a large displacement region.
    17: Example(
        n=4,
        D_zones=[[-2, 2]] * 4,
        I_zones=[[-0.15, 0.15], [-0.15, 0.15], [-0.15, 0.15], [-0.15, 0.15]],
        U_zones=[[1.2, 2], [-2, 2], [1.2, 2], [-2, 2]],
        f=[
            lambda x: x[1],
            lambda x: -x[0] - 0.4 * x[1] + 0.15 * x[2],
            lambda x: x[3],
            lambda x: -x[2] - 0.4 * x[3],
        ],
        name="C17",
        source_name="CartPole4D_linearized",
    ),
    # Polynomial Tora-style pair of damped modes.  Cubic restoring terms keep
    # the vector field polynomial while making the example nonlinear.
    18: Example(
        n=4,
        D_zones=[[-2, 2]] * 4,
        I_zones=[[-0.12, 0.12], [-0.12, 0.12], [-0.12, 0.12], [-0.12, 0.12]],
        U_zones=[[1.1, 2], [-2, 2], [1.1, 2], [-2, 2]],
        f=[
            lambda x: x[1],
            lambda x: -x[0] - 0.35 * x[1] - 0.08 * x[0] ** 3 + 0.1 * x[2],
            lambda x: x[3],
            lambda x: -x[2] - 0.35 * x[3] - 0.08 * x[2] ** 3,
        ],
        name="C18",
        source_name="Tora4D_polynomial",
    ),
}
