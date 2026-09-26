# Diffusion-Induced Probabilistic Safety Verification

This archive contains the complete implementation used to generate the paper's
SynNBC experiments. It learns a one-dimensional diffusion disturbance from
noisy trajectories, constructs a fixed stochastic proxy, synthesizes stochastic
barrier certificates, and compares the proposed continuous Generator-SOS method
with two data-Gaussian certificate pipelines.

The archive is self-contained with respect to the benchmark definitions. The
required SynNBC registry is bundled under `third_party/synncb`; no external
SynNBC checkout or machine-specific path is needed.

## Quick start

The reported environment is Windows 11 with Python 3.14.5 and MOSEK 11.2.2.
Python 3.11 or later is recommended.

1. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install all Python dependencies.

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Install a valid MOSEK license. The Python package is included in
   `requirements.txt`, but the license is supplied separately by MOSEK. Verify
   that CVXPY can see the solver:

   ```powershell
   python -c "import cvxpy as cp; print(cp.installed_solvers())"
   ```

   `MOSEK` must appear in the printed list.

4. Validate the installation with manuscript case C6
   quadratic smoke experiment.

   ```powershell
   python run_paper_experiments.py --smoke
   ```

5. Run the complete paper experiment.

   ```powershell
   python run_paper_experiments.py
   ```

The complete run evaluates the nine manuscript benchmarks C1--C9 with the proposed
diffusion Generator-SOS method at barrier degrees 2,
4, and 6. It can take from tens of minutes to several hours depending on the
CPU, solver version, and the number of resource-limited cases.

## Resume an interrupted run

The result JSON is updated after each case and degree. Resume from the last
completed candidate with:

```powershell
python run_paper_experiments.py --resume
```

The resume operation validates the experiment protocol. Results created by an
older protocol are intentionally rejected; use a new output directory when the
model or certificate semantics have changed.

## Inspect commands without running solvers

```powershell
python run_paper_experiments.py --dry-run
```

The dry run checks installed package versions and the bundled benchmark
registry, then prints the exact training, verification, and export commands.

## Run a subset

Use manuscript IDs C1--C9 when selecting systems. Their source descriptions are
listed in the mapping table below:

```powershell
python run_paper_experiments.py `
  --examples C1 C3 C6 C8 `
  --degrees 2 4 6 `
  --output outputs\selected_cases `
  --artifacts paper_artifacts\selected_cases
```

The case selected by `--case-study` must also appear in `--examples` because
the exporter needs one verified case for the case-study figures.

## Regenerate an external paper directory

The archive does not require the LaTeX paper. If a paper checkout is available,
the same entry point can write directly to its `Figures` directory:

```powershell
python run_paper_experiments.py `
  --resume `
  --paper-root C:\path\to\paper
```

Add `--compile-paper` to run `latexmk` after exporting. The paper root must
contain `AnonymousSubmission2027.tex`, and `latexmk` must be on `PATH`.

## What the complete entry point runs

`run_paper_experiments.py` is the only recommended top-level entry point. It
performs the following operations in order:

1. Checks required package versions and the bundled benchmark registry.
2. Loads only the nine manuscript benchmarks (C1--C9), including the external
   Duffing Example 4.3 used for C2.
3. Generates shared noisy trajectories for each system.
4. Rejects transitions affected by domain clipping.
5. Trains a one-dimensional conditional diffusion model on the final state
   increment and fits a zero-mean Gaussian to the same accepted increments.
6. Runs the proposed continuous Generator-SOS certificate for degrees 2, 4,
   and 6 subject to the declared resource budget.
7. Applies posterior polynomial, SDP residual, and Gram-eigenvalue checks.
8. Selects the proposed-method candidate with minimum reported rho.
9. Exports JSON, CSV, Markdown, LaTeX fragments, and paper-ready figures.
10. Writes a run manifest containing the exact package versions.

## Experimental protocol

### Data generation

For every benchmark, the simulator uses

```text
x[k+1] = x[k] + dt*f_nom(x[k]) + 0.03*sqrt(dt)*xi[k]
xi[k] ~ Normal(0, I)
```

with `dt=0.05`, 64 requested trajectories, horizon 40, and seed 7. A fresh
innovation is drawn at every physical time step. The value `0.03` is used only
to generate trajectories; it is not inserted directly into either learned
verification model.

Transitions that leave the verification box or touch a clipping boundary are
removed. The train/test split is grouped by trajectory so adjacent transitions
from one rollout cannot appear on both sides.

### One-dimensional diffusion proxy

Only the final physical coordinate is retained as stochastic. The conditional
diffusion model uses a 32-step cosine schedule, eight noisy copies per training
increment, hidden widths `(64, 64)`, batch size 256, and learning rate
`5e-4`. Symmetric increments suppress an additional learned drift term, so the
exact benchmark drift is preserved.

Quadratic reverse-polynomial moment propagation maps the complete learned
reverse chain to a rank-one covariance rate

```text
G_diff = lambda_diff * e_j * e_j^T.
```

The proposed method verifies the fixed proxy

```text
dX = f_nom(X) dt + sqrt(lambda_diff) e_j dW.
```

### Data-Gaussian proxy

The comparison covariance is estimated from the same retained final-coordinate
increments:

```text
lambda_G = sum_i residual_i^2 / (N*dt).
```

The Gaussian model is therefore data-fitted rather than fixed to the trajectory
generator's `0.03` amplitude.

### Certificate methods

The paper compares the following method keys. The default entry point in this
archive recomputes the proposed method; archived comparison results use the
same keys and display names:

- `diffusion_direct_generator_sos`: continuous-time stochastic generator SOS
  on the learned diffusion proxy.
- `gaussian_neural_rsm_bernstein`: polynomial neural/RSM candidates with an
  independent Bernstein one-step verifier on the data-Gaussian proxy.
- `gaussian_c_sbc_multi_candidate_sos` (display name **DiffSBC**): a bank of
  polynomial candidates with discrete expectation SOS verification on the same
  Gaussian proxy.

For the proposed method, the optimized conditions are

```text
B(x) >= 0                on the verification domain
B(x) <= rho              on the initial set
B(x) >= 1                on the unsafe set
grad(B)^T f + 0.5 Tr(G Hess(B)) <= 0 on the domain
```

All paper results use pointwise worst-case initial semantics. The reported
lower bound is `1-rho`. Successful candidates receive a common reporting
allowance of `1e-3` on raw rho. The SynNBC C5 source case (manuscript case C4) receives the explicitly
declared `2e-6` numerical refinement only at barrier degree 6.

### Error semantics

The paper experiment uses `paper-fixed` surrogate semantics. The learned
covariance is treated as an exact parameter of a fixed proxy SDE. Terminal PAC
inflation, reverse-surrogate error, covariance transfer error, and finite-probe
coverage error are not added to the certificate constraints.

The resulting certificate is formal for the fixed proxy used by its verifier.
It is not a distribution-free certificate for every unknown physical process
consistent with the finite training set. See
`docs/dips_bc_error_handling_comparison.md` for the detailed scope.

## Output layout

The default numerical output is `outputs/paper_experiments`:

```text
outputs/paper_experiments/
  results.json
  candidate_results.csv
  minimum_rho_summary.md
  C1/
    diffusion_training/
    gaussian_transition_abstractions/
  ...
```

`results.json` is authoritative. The CSV and Markdown files are derived views.
Each case directory contains the data split, model configuration, training
summary, fitted model objects, and per-degree Gaussian transition abstractions.

The default paper output is `paper_artifacts`:

```text
paper_artifacts/
  README.md
  run_manifest.json
  selected_nine_case_results.md
  selected_nine_case_results.csv
  paper_main_table_rows.tex
  paper_main_c6_sbc.tex
  paper_main_c6_degree_sweep.png
  paper_main_benchmark_summary.png
```

## Manuscript case mapping

The manuscript renumbers nine selected source systems:

| Manuscript case | Benchmark description | Source | Registry ID |
|---|---|---|---|---:|
| C1 | 2D quadratic Arch system | SynNBC C1 | 4 |
| C2 | 2D controlled Duffing Example 4.3 | Zhu et al. PLDI 2019 | -- |
| C3 | 2D quadratic oscillatory system | SynNBC C3 | 7 |
| C4 | 2D quadratic Van der Pol variant | SynNBC C5 | 9 |
| C5 | 3D cubic Van der Pol variant | SynNBC C8 | 5 |
| C6 | 7D quadratic Lie-derivative system | SynNBC C12 | 12 |
| C7 | 9D quadratic equilibrium system | SynNBC C14 | 13 |
| C8 | 3D cubic Lyapunov system | SynNBC R1 | 23 |
| C9 | 5D cubic Lotka system | SynNBC R2 | 24 |

Generated result files use C1--C9 in tables and figures and retain source IDs
in metadata for reproducibility.

## Source layout

- `bobo_aaai/`: reusable data, diffusion, polynomial, SOS, and baseline code.
- `run_paper_experiments.py`: complete archive-level paper experiment entry.
- `run_synncb_certificate_methods.py`: method/degree scheduler and result writer.
- `run_synncb_diffusion_ablation.py`: shared diffusion and Gaussian fitting.
- `run_synncb_table3.py`: benchmark adapter and exact polynomial drift loader.
- `export_synncb_certificate_method_study.py`: table and figure exporter.
- `third_party/synncb/`: bundled benchmark registry.
- `tests/`: unit and regression tests.
- `docs/`: model assumptions, error semantics, and legacy experiment notes.
- `outputs/`: retained historical and generated numerical results.

Legacy scripts are kept for earlier CARLA, CDC 2004, discrete-transition, and
constant-noise experiments. They are not called by the current paper entry
point.

## Tests

Run all tests with the standard library test runner:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

The test suite covers benchmark loading, dataset grouping, diffusion moments,
Gaussian fitting, continuous and discrete barrier formulations, paper exports,
and the three certificate-method registrations.

## Reproducibility notes

- Solver times depend on hardware, MOSEK version, and system load.
- Boundary SDP residuals can move slightly across BLAS and solver versions.
- Random seed 7 controls trajectory and training sampling, but numerical solver
  ordering can still produce small coefficient differences.
- Resource-limited status means the declared pre-solve budget was exceeded; it
  does not prove that no certificate exists.
- A zero safety lower bound can be a verified but trivial certificate with
  reported rho equal to 1.
- Never edit paper numbers manually. Regenerate tables from `results.json`.

## Troubleshooting

### MOSEK is installed but solving fails

Confirm that the license is valid and visible to the current user. A package
installation alone does not include a license.

### A resume run reports a protocol mismatch

Use a new output directory. Resume is intentionally limited to results created
by exactly the same protocol version.

### A case reports `training_data_unavailable`

Too few unclipped transitions remained under the common data protocol. This is
reported consistently for all methods in that case.

### Paper figures cannot open a transition abstraction

Run the exporter through `run_paper_experiments.py`. New results store model
paths relative to the result root so the entire directory can be moved after
unpacking.
