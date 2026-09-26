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
    # Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 1.
    # The published map is divided by dt=0.2 to obtain the nominal
    # continuous-time drift. The theta-dependent term is represented by the
    # learned diffusion residual in the paper pipeline.
    4: Example(
        2, [[-0.5, 0.5], [-0.5, 0.5]],
        [[-0.25, -0.15], [0.15, 0.25]],
        [[0.3, 0.5], [-0.5, 0.5]],
        [
            lambda x: -x[1],
            lambda x: x[0] + 0.5*x[1]*(x[0]**2 - 1.0),
        ],
        "D4", "vanderpol1_continuous",
        "Feng, An, and Xu, Runtime Safety and Reach-avoid Prediction of Stochastic Systems via Observation-aware Barrier Functions, arXiv:2511.09192v1, Appendix Example 1; source dynamics from Xue, Zhan, and Franzle (2022).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time nominal drift obtained from the published dt=0.2 Euler map; theta residual learned as diffusion",
        False,
    ),
    # Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 3.
    # The original reach-avoid target is retained in metadata only; this
    # adapter evaluates the same dynamics under the paper's safety pipeline.
    5: Example(
        2, [[-0.5, 0.5], [-0.5, 0.5]],
        [[0.1, 0.3], [0.1, 0.3]],
        [[0.34, 0.5], [-0.5, 0.5]],
        [
            lambda x: x[1],
            lambda x: -x[0] + x[0]**3/3.0 - x[1],
        ],
        "D5", "equil_continuous",
        "Feng, An, and Xu, Runtime Safety and Reach-avoid Prediction of Stochastic Systems via Observation-aware Barrier Functions, arXiv:2511.09192v1, Appendix Example 3; source system from Prajna, Jadbabaie, and Pappas (2007).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time nominal drift obtained from the published dt=0.1 Euler map; theta*x residual learned as diffusion; target disk omitted for safety-only adaptation",
        False,
    ),
    # Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 2.
    6: Example(
        2, [[-0.5, 0.5], [-0.5, 0.5]],
        [[-0.25, -0.15], [0.15, 0.25]],
        [[0.4, 0.5], [-0.5, 0.5]],
        [
            lambda x: -x[1],
            lambda x: x[0] + 0.5*x[1]*(x[0]**2 - 1.0),
        ],
        "D6", "vanderpol2_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 2; source dynamics from Xue, Zhan, and Franzle (2022).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time safety adapter from the published dt=0.2 Euler map; theta residual learned as diffusion",
        False,
    ),
    # Appendix Example 4 (arch), with the theta terms learned as diffusion.
    7: Example(
        2, [[-1, 1], [-1, 1]],
        [[0.8, 1.0], [0.8, 1.0]],
        [[-0.2, 0.2], [-0.2, 0.2]],
        [
            lambda x: x[0] - x[0]**3 - x[0]*x[1]**2,
            lambda x: -x[0] - x[0]**2*x[1] - x[1]**3,
        ],
        "D7", "arch_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 4; source system from Sogokon, Ghorbal, and Johnson (2016).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time nominal drift from the published dt=0.1 map; theta*y terms learned as diffusion",
        False,
    ),
    # Appendix Example 5 (descent) is reach-avoid in the source paper. This
    # adapter tests its unsafe-set component with the target omitted.
    8: Example(
        1, [[-3, 5]], [[0.8, 1.2]], [[2.9, 3.1]],
        [lambda x: -1.0],
        "D8", "descent_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 5.",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time safety-only adapter from the published dt=0.2 reach-avoid map; target omitted",
        False,
    ),
    # Appendix Example 6 (osc).
    9: Example(
        2, [[-4, 4], [-4, 4]],
        [[-0.1, 0.1], [0.65, 0.85]],
        [[-1, 1], [1, 3]],
        [
            lambda x: x[1],
            lambda x: -x[0] - 1.6*x[1],
        ],
        "D9", "osc_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 6; source system from Xue, Zhan, and Franzle (2022).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time nominal drift from the published dt=0.1 map; theta*y term learned as diffusion",
        False,
    ),
    # Appendix Example 7 (liederivative).
    10: Example(
        2, [[-2, 2], [-1, 1]],
        [[-0.1, 0.1], [-0.6, -0.4]],
        [[0.45, 0.55], [0.70, 0.80]],
        [
            lambda x: -x[1],
            lambda x: 0.5*x[0]**2,
        ],
        "D10", "liederivative_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 7; source system from Liu, Zhan, and Zhao (2011).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time nominal drift from the published mixed-step map; theta*x term learned as diffusion",
        False,
    ),
    # Appendix Example 8 (lyapunov). The theta-dependent linear terms are
    # omitted from the nominal drift and represented by learned diffusion.
    11: Example(
        3, [[-2, 2], [-2, 2], [-2, 2]],
        [[0.05, 0.45], [0.05, 0.45], [0.05, 0.45]],
        [[0.28, 0.72], [0.28, 0.72], [0.28, 0.72]],
        [
            lambda x: 0.0,
            lambda x: 0.0,
            lambda x: -x[0] - 2*x[1] - x[2] + x[0]**3,
        ],
        "D11", "lyapunov_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 8; source system from Ratschan and She (2010).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time adapter from the published stochastic map; theta-linear terms learned as diffusion",
        False,
    ),
    # Appendix Example 9 (lotka).
    12: Example(
        3, [[-1, 1], [-1, 1], [-1, 1]],
        [[0.1, 0.9], [0.1, 0.9], [-0.4, 0.4]],
        [[-0.2, 0.2], [-1, 1], [-1, 1]],
        [
            lambda x: -x[0]*x[2],
            lambda x: x[1] - 2*x[1]*x[2],
            lambda x: x[2]*(x[0] + x[1] - 1.0),
        ],
        "D12", "lotka_continuous",
        "Feng, An, and Xu, arXiv:2511.09192v1, Appendix Example 9; source system from Goubault et al. (2014).",
        "https://arxiv.org/abs/2511.09192",
        "continuous-time nominal drift from the published dt=0.1 map; theta*x term learned as diffusion",
        False,
    ),
}
