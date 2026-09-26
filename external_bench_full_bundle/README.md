# External Benchmark Full Run Bundle

This folder keeps the external benchmark definitions and the corresponding
full-protocol experiment outputs together.

## Contents

- `external_benchmarks/benchmarks/Exampler_B.py`: SynNBC-format definitions of
  D1 Duffing, D2 FitzHugh-Nagumo, and D3 hybrid Example 2 mode 1, including
  citations and source URLs.
- `results/`: authoritative JSON/CSV/Markdown records and per-case datasets,
  learned models, abstractions, and candidate certificates.

## Reproduction command

Run from the parent `paper_experiment_code_package` directory:

```powershell
python run_paper_experiments.py `
  --benchmark-root external_benchmarks `
  --examples D1 D2 D3 `
  --degrees 2 4 6 `
  --case-study D1 `
  --output external_bench_full_bundle\results `
  --skip-export
```

The recorded run used 64 trajectories, horizon 40, diffusion maximum
iterations 300, and barrier degrees 2, 4, and 6. `D2` is a continuous RHS
corresponding to a discrete Euler source model. `D3` is only one mode of a
hybrid source system. The source and set-approximation caveats are documented
in `external_benchmarks/README.md` and `Exampler_B.py`.
