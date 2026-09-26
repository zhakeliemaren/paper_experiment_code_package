# Additional Benchmarks

This registry contains two four-dimensional polynomial closed-loop surrogate
systems used to exercise the existing diffusion and Generator-SOS pipeline:

- `C17`: a damped two-mode CartPole-style linearized model.
- `C18`: a nonlinear Tora-style model with cubic restoring terms.

They are deliberately polynomial and autonomous so that the current symbolic
drift loader can process them exactly. They are suitable for method and solver
tests, but they should not be described as exact reproductions of the original
non-polynomial CartPole or Tora environments from the referenced papers.

Example command:

```powershell
python run_paper_experiments.py `
  --benchmark-root new_benchmarks `
  --examples C17 C18 `
  --degrees 2 4 `
  --case-study C17 `
  --output outputs\new_bench_smoke `
  --skip-export
```
