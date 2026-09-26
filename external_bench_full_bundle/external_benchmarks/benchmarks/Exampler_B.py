"""Externally sourced polynomial benchmark systems in SynNBC format."""

from __future__ import annotations

import numpy as np


class Example:
    def __init__(self, n, D_zones, I_zones, U_zones, f, name, source_name,
                 source_citation, source_url, model_semantics, exact_sets=True):
        if not (len(D_zones) == len(I_zones) == len(U_zones) == len(f) == n):
            raise ValueError("Benchmark dimensions are inconsistent")
        self.n = int(n)
        self.D_zones = np.asarray(D_zones, dtype=float)
        self.I_zones = np.asarray(I_zones, dtype=float)
        self.U_zones = np.asarray(U_zones, dtype=float)
        self.f = list(f)
        self.name = str(name)
        self.source_name = str(source_name)
        self.source_citation = str(source_citation)
        self.source_url = str(source_url)
        self.model_semantics = str(model_semantics)
        self.exact_sets = bool(exact_sets)


examples = {
    # Zhu et al., PLDI 2019, Example 4.3. The published P1 controller
    # a=0.39*x-1.41*y is substituted into the Duffing equation.
    1: Example(
        2, [[-6, 6], [-6, 6]], [[-2.5, 2.5], [-2, 2]],
        [[5, 6], [5, 6]],
        [lambda x: x[1], lambda x: -0.6*x[1]-x[0]-x[0]**3+0.39*x[0]-1.41*x[1]],
        "D1", "Duffing_P1",
        "Zhu et al., An Inductive Synthesis Framework for Verifiable Reinforcement Learning, PLDI 2019, Example 4.3.",
        "https://doi.org/10.1145/3314221.3314635",
        "continuous Duffing system with published P1 feedback", False,
    ),
    # Lefringhausen et al., IEEE CDC 2025, Eq. (20). The paper gives a
    # discrete Euler map; the stored f is its continuous RHS.
    2: Example(
        2, [[-3, 3], [-3, 3]], [[-1, 1], [-1, 1]], [[3, 3], [3, 3]],
        [lambda x: x[1]-x[0]**3, lambda x: 0.01*x[0]-0.005*x[1]],
        "D2", "FitzHughNagumo_CDC2025",
        "Lefringhausen et al., Barrier Certificates for Unknown Systems with Latent States and Polynomial Dynamics using Bayesian Inference, IEEE CDC 2025, Eq. (20).",
        "https://doi.org/10.1109/CDC57313.2025.11312207",
        "continuous RHS corresponding to the paper's dt=0.1 Euler map", False,
    ),
    # Kong et al., arXiv:1303.6885, Example 2. This is mode 1 only; the
    # original guards, resets, and mode 2 are not representable by Example.
    3: Example(
        3, [[-4, 4]]*3, [[-0.1, 0.1]]*3, [[3.2, 4], [-4, 4], [-4, 4]],
        [lambda x: x[1], lambda x: -x[0]+x[2],
         lambda x: x[0]+(2*x[1]+3*x[2])*(1+x[2])],
        "D3", "HybridExample2_mode1",
        "Kong et al., Exponential-Condition-Based Barrier Certificate Generation for Safety Verification of Hybrid Systems, arXiv:1303.6885, Example 2.",
        "https://arxiv.org/abs/1303.6885",
        "mode-1 flow extracted from a published hybrid system", False,
    ),
}
