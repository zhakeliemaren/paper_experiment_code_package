"""Nine benchmark systems used in the paper experiments.

The dynamics and sets originate from the SynNBC benchmark registry.  The
``source_name`` field records the original identifier, while ``name`` follows
the C1--C9 numbering used by the manuscript and this reproduction package.
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
    1: Example(
        n=2,
        D_zones=[[1, 5]] * 2,
        I_zones=[[4, 4.5], [0.9, 1.1]],
        U_zones=[[1, 2], [2, 3]],
        f=[
            lambda x: -5.5 * x[1] + x[1] ** 2,
            lambda x: 6 * x[0] - x[0] ** 2,
        ],
        name="C1",
        source_name="C1",
    ),
    2: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-1 / 5, 1 / 5], [3 / 10, 7 / 10]],
        U_zones=[[-2, -1], [-2, -1]],
        f=[
            lambda x: -x[0] + 2 * x[0] ** 3 * x[1] ** 2,
            lambda x: -x[1],
        ],
        name="C2",
        source_name="C2",
    ),
    3: Example(
        n=2,
        D_zones=[[-2, 2]] * 2,
        I_zones=[[-1, 0], [-1, 0]],
        U_zones=[[1, 2], [1, 2]],
        f=[
            lambda x: -1 + x[0] ** 2 + x[1] ** 2,
            lambda x: 5 * (-1 + x[0] * x[1]),
        ],
        name="C3",
        source_name="C3",
    ),
    4: Example(
        n=2,
        D_zones=[[-1, 1]] * 2,
        I_zones=[[-1 / 10, 1 / 10], [-1 / 10, 1 / 10]],
        U_zones=[[1 / 2, 1], [1 / 2, 1]],
        f=[
            lambda x: -2 * x[0] + x[0] ** 2 + x[1],
            lambda x: x[0] - 2 * x[1] + x[1] ** 2,
        ],
        name="C4",
        source_name="C5",
    ),
    5: Example(
        n=3,
        D_zones=[[-2, 2]] * 3,
        I_zones=[[-0.25, 0.75], [-0.25, 0.75], [-0.75, 0.25]],
        U_zones=[[1, 2], [-2, -1], [-2, -1]],
        f=[
            lambda x: -x[1],
            lambda x: -x[2],
            lambda x: -x[0] - 2 * x[1] - x[2] + x[0] ** 3,
        ],
        name="C5",
        source_name="C8",
    ),
    6: Example(
        n=7,
        D_zones=[[-2, 2]] * 7,
        I_zones=[[-1.01, -0.99]] * 7,
        U_zones=[[1.8, 2]] * 7,
        f=[
            lambda x: -0.4 * x[0] + 5 * x[2] * x[3],
            lambda x: 0.4 * x[0] - x[1],
            lambda x: x[1] - 5 * x[2] * x[3],
            lambda x: 5 * x[4] * x[5] - 5 * x[2] * x[3],
            lambda x: -5 * x[4] * x[5] + 5 * x[2] * x[3],
            lambda x: 0.5 * x[6] - 5 * x[4] * x[5],
            lambda x: -0.5 * x[6] + 5 * x[4] * x[5],
        ],
        name="C6",
        source_name="C12",
    ),
    7: Example(
        n=9,
        D_zones=[[-2, 2]] * 9,
        I_zones=[[0.99, 1.01]] * 9,
        U_zones=[[1.8, 2]] * 9,
        f=[
            lambda x: 3 * x[2] - x[0] * x[5],
            lambda x: x[3] - x[1] * x[5],
            lambda x: x[0] * x[5] - 3 * x[2],
            lambda x: x[1] * x[5] - x[3],
            lambda x: 3 * x[2] + 5 * x[0] - x[4],
            lambda x: 5 * x[4] + 3 * x[2] + x[3]
            - x[5] * (x[0] + x[1] + 2 * x[7] + 1),
            lambda x: 5 * x[3] + x[1] - 0.5 * x[6],
            lambda x: 5 * x[6] - 2 * x[5] * x[7] + x[8] - 0.2 * x[7],
            lambda x: 2 * x[5] * x[7] - x[8],
        ],
        name="C7",
        source_name="C14",
    ),
    8: Example(
        n=3,
        D_zones=[[-0.3, 0.3]] * 3,
        I_zones=[[-0.3, 0], [-0.2, 0.3], [-0.2, 0.3]],
        U_zones=[[-0.2, -0.15], [-0.3, -0.25], [-0.3, -0.25]],
        f=[
            lambda x: (x[1] + x[2]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
        ],
        name="C8",
        source_name="R1",
    ),
    9: Example(
        n=5,
        D_zones=[[-0.3, 0.3]] * 5,
        I_zones=[[-0.3, 0]] + [[-0.2, 0.3]] * 4,
        U_zones=[[-0.2, -0.15]] + [[-0.3, -0.25]] * 4,
        f=[
            lambda x: (x[1] + x[2] + x[2] + x[3]) / 100 + 1,
            lambda x: x[2],
            lambda x: -10 * (x[1] - x[1] ** 3 / 6) - x[1],
            lambda x: x[4],
            lambda x: -10 * (x[3] - x[3] ** 3 / 6) - x[1],
        ],
        name="C9",
        source_name="R2",
    ),
}
