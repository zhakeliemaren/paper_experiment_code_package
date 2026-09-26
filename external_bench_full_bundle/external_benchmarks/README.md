# External Benchmarks

`benchmarks/Exampler_B.py` follows the SynNBC `Example` interface and stores
the citation and URL on every case.

- `D1`: Duffing with the published P1 feedback controller, Zhu et al., PLDI 2019, Example 4.3.
- `D2`: FitzHugh-Nagumo polynomial RHS corresponding to the Euler map, Lefringhausen et al., IEEE CDC 2025, Eq. (20).
- `D3`: mode-1 flow from Kong et al., arXiv:1303.6885, Example 2.

`D2` is a continuous representation of a discrete-time source model. `D3` is
only one mode and omits the source hybrid guards and resets. `exact_sets=False`
marks cases whose rectangular SynNBC sets are approximations of the source
sets; this must be reported in any paper experiment.
