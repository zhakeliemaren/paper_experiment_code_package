# Diffusion-Induced Probabilistic Safety Verification

This package reproduces the proposed-method experiments reported in the paper.
It contains the nine manuscript cases, the one-dimensional diffusion model,
continuous-time Generator-SOS barrier synthesis, posterior numerical checks,
and the main-result artifact exporter.

## Experiment scope

The package uses manuscript identifiers `C1` through `C9`. The source column
records the corresponding identifier in the SynNBC benchmark repository.

| Manuscript case | SynNBC source case | State dimension | Drift degree |
|---|---|---:|---:|
| C1 | C1 | 2 | 2 |
| C2 | C2 | 2 | 5 |
| C3 | C3 | 2 | 2 |
| C4 | C5 | 2 | 2 |
| C5 | C8 | 3 | 3 |
| C6 | C12 | 7 | 2 |
| C7 | C14 | 9 | 2 |
| C8 | R1 | 3 | 3 |
| C9 | R2 | 5 | 3 |

The bundled definitions are in
`third_party/synncb/benchmarks/Exampler_B.py`. They retain the dynamics and
sets from the source registry while using the manuscript numbering above.

## Installation

The reported environment uses Windows 11 and Python 3.14.5. Python 3.11 or
later is supported by the package dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

MOSEK requires a separate license. The installed solver can be checked with:

```powershell
python -c "import cvxpy as cp; print(cp.installed_solvers())"
```

The output must contain `MOSEK` before the SOS experiments are started.

## Reproduction commands

Validate the package configuration without training or solving:

```powershell
python run_paper_experiments.py --dry-run
```

Run a short C6 degree-2 installation test in `outputs/smoke`:

```powershell
python run_paper_experiments.py --smoke
```

Run all nine manuscript cases at barrier degrees 2, 4, and 6:

```powershell
python run_paper_experiments.py
```

Resume candidates already recorded under the same protocol identifier:

```powershell
python run_paper_experiments.py --resume
```

Run a subset using manuscript identifiers:

```powershell
python run_paper_experiments.py `
  --examples C1 C4 C6 `
  --degrees 2 4 6 `
  --case-study C6 `
  --output outputs\selected_cases `
  --artifacts paper_artifacts\selected_cases
```

`--case-study` must be included in `--examples`. Results produced under a
changed protocol or parameter set must use a new output directory unless all
existing candidates in the target directory have been removed.

## Method

### 1. Exact nominal dynamics

Each source vector field is converted symbolically to polynomial coefficient
dictionaries.

### 2. Stochastic trajectory generation

Training rollouts use

```text
x[k+1] = x[k] + dt*f_nom(x[k]) + 0.03*sqrt(dt)*xi[k],
xi[k] ~ Normal(0, I).
```

A new independent innovation is drawn at each time step. Domain-clipped
transitions are excluded because the clipping displacement is a simulator
artifact. The train/test split is grouped by trajectory.

### 3. One-dimensional diffusion training

Only the final state coordinate is retained as stochastic during learning and
verification. The conditional denoising model uses four history steps through
a statistics context encoder. Increment symmetrization imposes the intended
zero-mean noise role.

### 4. Reverse-moment construction

The 32-step reverse chain is represented by quadratic reverse-update
surrogates. Gaussian moments are propagated deterministically through the
complete chain. The resulting terminal covariance is converted to the
rank-one physical covariance rate

```text
G_diff = lambda_diff * e_j * e_j^T.
```

The fixed verification proxy is

```text
dX = f_nom(X) dt + sqrt(lambda_diff) * e_j dW.
```

The learned terminal mean is not added to the nominal drift.

### 5. Generator-SOS barrier synthesis

For each requested barrier degree, all coefficients of `B` and `rho` are
optimized under

```text
B(x) >= 0                                      on Psi,
B(x) <= rho                                    on Theta,
B(x) >= 1                                      on Xi,
grad(B)^T f_nom + 0.5*Tr(G_diff Hess(B)) <= 0  on Psi.
```

Putinar SOS multipliers encode the box constraints. Initial conditions use
pointwise worst-case semantics. The safety lower bound is `1 - rho`.

### 6. Posterior checks and selection

Each solver candidate is checked using polynomial samples, coefficient
identity residuals, and Gram-matrix eigenvalues. A candidate that fails a
posterior check is reported with safety lower bound zero. A common `1e-3`
reporting allowance is added to accepted raw `rho` values. For each case, the
accepted candidate with minimum reported `rho` is selected; certificate time
breaks ties.

## Active experiment parameters

### Data generation

| Parameter | Value |
|---|---:|
| Random seed | 7 |
| NumPy bit generator | PCG64 through `default_rng` |
| Time step | 0.05 |
| Trajectories per case | 64 |
| Steps per trajectory | 40 |
| Data-generation noise amplitude | 0.03 |
| History length | 4 |
| Train/test split | 80/20 by trajectory |
| Learned stochastic coordinate | final coordinate |
| Learned stochastic dimension | 1 |

### Diffusion model

| Parameter | Value |
|---|---:|
| Schedule | cosine |
| Reverse steps | 32 |
| Terminal cumulative signal target | at most 0.003 |
| Training copies per increment | 8 |
| Hidden widths | 64, 64 |
| Activation | tanh |
| Optimizer | Adam |
| L2 coefficient | 1e-4 |
| Batch size | 256 |
| Learning rate | 5e-4 |
| Maximum iterations | 300 |
| Early-stopping patience | 15 |
| Increment symmetrization | enabled |
| Reverse surrogate degree | 2 |
| Reverse-moment conditions | 32 |
| Polynomial abstraction conditions | 80 |
| Spatial partitions | 1 |
| Validation grid count | 2 |

### Barrier synthesis

| Parameter | Value |
|---|---:|
| Barrier degrees | 2, 4, 6 |
| Initial semantics | worst case |
| Objective | minimize rho |
| rho upper bound | 1 |
| SOS margin | 1e-6 |
| SOS regularization | 1e-7 |
| Sample tolerance | 1e-3 |
| Coefficient tolerance | 1e-7 |
| Gram eigenvalue tolerance | 1e-7 |
| Maximum estimated PSD scalars | 12000 |
| Reporting allowance | 1e-3 |
| C4 degree-6 refinement tolerance | 2e-6 |

All configurations are run once with seed 7. The degree sweep is a model
selection scan, not an independent statistical repetition. Reported times are
wall-clock measurements and vary with hardware, solver version, and system
load. No confidence interval or statistical significance test is claimed for
single-run wall-clock values.

## Code structure

| Path | Responsibility |
|---|---|
| `run_paper_experiments.py` | Main command, case scheduling, resume logic, and manifests |
| `paper_benchmarks.py` | Nine-case registry loader and exact polynomial system adapter |
| `paper_pipeline.py` | Diffusion training, fixed-surrogate construction, SOS solving, and result serialization |
| `export_paper_artifacts.py` | Main-result LaTeX fragments and figures |
| `diffusion_sbc/context.py` | History context encoding |
| `diffusion_sbc/diffusion.py` | Conditional denoising model |
| `diffusion_sbc/score_dynamics.py` | Reverse-polynomial moment propagation |
| `diffusion_sbc/pipeline.py` | Training and model artifact construction |
| `diffusion_sbc/barrier.py` | Continuous-time stochastic barrier synthesis |
| `diffusion_sbc/sos.py` | Polynomial and SOS utilities |
| `third_party/synncb/benchmarks/Exampler_B.py` | Nine benchmark definitions |

## Outputs

The complete run writes:

```text
outputs/paper_experiments/
  results.json
  candidate_results.csv
  minimum_rho_summary.md
  C1/ ... C9/

paper_artifacts/
  paper_main_table_rows.tex
  paper_main_c6_sbc.tex
  paper_main_benchmark_summary.png
  paper_main_c6_degree_sweep.png
  run_manifest.json
```

`results.json` is the authoritative numerical record. `candidate_results.csv`
contains every attempted degree. `minimum_rho_summary.md` contains one selected
main-method result per manuscript case.

## Verification scope

The SOS inequalities are global over the stated semialgebraic sets for the
fixed learned surrogate and the recorded numerical tolerances. The paper-fixed
protocol does not provide a statistical transfer theorem from finite training
data to the unknown real disturbance law. The generated safety lower bound
therefore applies to the fixed learned stochastic proxy used by the
Generator-SOS program.

## Benchmark attribution

The benchmark dynamics are selected from the SynNBC repository:

```text
https://github.com/tete0602/SynNBC
```

The bundled registry is limited to the nine cases required to reproduce the
paper table.
