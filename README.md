# Artifact for "Whiten Before You Audit: Privacy Auditing for Federated Learning with Correlated Noise"

This archive contains the simulator, the audit code, the scripts for every
experiment, and the per-run records behind every table and figure of the
paper. It is anonymized for double-blind review.

With the stored records, every number, table and figure of the paper can be
regenerated in about one minute on a laptop, without training a model.
Rerunning the experiments from scratch takes from minutes to hours per
experiment (see "Rerunning the experiments").

## Contents

| Path | What it is |
|---|---|
| `harness/` | Federated training simulator with differential privacy, the noise mechanisms, the privacy accountant and the audits |
| `experiments/` | One script per experiment, the script that computes all tables and numbers, and the script that draws the figures |
| `scripts/preprocess_stackoverflow.py` | Preparation of the StackOverflow data |
| `results/*.json` | One record per run: settings, noise multiplier, readings of the audits |
| `results/projections*/`, `results/side_release*/`, `results/retention*/` | Per-round projections of every run, from which all audits are recomputed |
| `paper/PoPETS/generated/` | The table bodies and numbers as they appear in the paper |
| `paper/PoPETS/figures/` | The result figures as they appear in the paper |
| `requirements.txt` | Python packages |

The main modules of `harness/`:

- `mechanisms.py`: DP-FedAvg, tree aggregation, BandMF, BLT and the
  square-root factorization, each with its strategy matrix, noise stream,
  sensitivity and calibration.
- `fedavg.py`: the training simulator (clipping, noise, canary clients).
- `audit_stats.py`: the five audits of the paper (final-model, max, sum,
  single-round and whitened), computed from stored per-round projections, and
  the two conversions to a privacy level.
- `auditor.py`, `auditor_whitened.py`: canary generation and the audits as
  they run during training.
- `accountant.py`: the privacy accountant.
- `side_release.py`: the adaptive-clipping experiment.
- `tasks.py`, `stackoverflow_data.py`, `decoder_task.py`: the learning tasks.

## Setup

Python 3.12 was used. Install the packages:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run every command below from the root of this archive.

## Regenerating the tables, numbers and figures

```
.venv/bin/python -m experiments.step48_v2_tables
.venv/bin/python -m experiments.plot_v2_figures
```

The first command recomputes all five audits of every run from the stored
projections and writes `paper/PoPETS/generated/numbers.tex` (every number
cited in the text), the table bodies `paper/PoPETS/generated/tab_*.tex`, and
`results/step48_v2_summary.json`. The second command writes the figures to
`paper/PoPETS/figures/`. Both overwrite the copies shipped in this archive
and reproduce them.

## Where each result of the paper comes from

| Result in the paper | Script | Run records |
|---|---|---|
| Main comparison on EMNIST (main table, readings figure, predicted-against-measured figure) | `experiments/step46_mechanism_arm_v2.py` | `results/step46_mechanism_arm_emnist.json`, `results/step46_mechanism_arm_emnist_production.json`, `results/projections_step46/`, `results/projections_step46_production/` |
| Sweep over the number of canary rounds (crossover figure, final-model figure, sweep table) | same script | `results/step46_mechanism_arm_emnist_ksweep.json`, `results/projections_step46_ksweep/` |
| Two stricter certificates | same script | `results/step46_mechanism_arm_emnist_eps4.json`, `..._eps8.json`, `results/projections_step46_eps4/`, `..._eps8/` |
| StackOverflow LSTM | same script | `results/step46_mechanism_arm_stackoverflow.json`, `results/projections_step46_so/` |
| Fault injection | `experiments/step47_fault_injection.py` | `results/step47_fault_injection_emnist.json`, `results/projections_step47/` |
| Adaptive clipping (hidden share) | `experiments/step44_side_release.py` | `results/step44_side_release_emnist*.json`, `results/side_release*/` |
| Check of the signal model | `experiments/step45_retention.py` | `results/step45_retention_emnist*.json`, `results/retention*/` |
| Misspecified covariance | `experiments/step42_mismatched_whitening.py` | `results/step42_mismatched_whitening.json` |
| Transformer adapters | `experiments/step20_mechanism_arm.py` | `results/step20_mechanism_arm_pythia-160m.json`, `results/projections/` |
| Gaussianity of the held-out projections | `experiments/step39_null_calibration.py` | `results/projections/`, `results/step39_null_calibration.json` |
| Forecasts registered before the runs (appendix) | `experiments/step17_decomposition.py` (runs), `experiments/step26_preregister_forecast.py` and `experiments/step29_preregister_gpt2.py` (registrations) | `results/prereg/` (forecast files and their SHA-256 digests), `results/step17_decomposition_*.json` (run records), `results/step28_armG_scored.json`, `results/step31_armH_scored.json`, `results/step35_armH_seeds_scored.json` (scoring) |
| All tables and numbers | `experiments/step48_v2_tables.py` | all of the above |
| All result figures | `experiments/plot_v2_figures.py` | `results/step48_v2_summary.json`, `results/step44_side_release_emnist_cpu.json` |

Files with the suffix `_cpu` are the repetition of an experiment on a CPU.

## Rerunning the experiments

The data are not part of this archive. EMNIST is downloaded automatically
into `data/` on first use. For StackOverflow, run

```
.venv/bin/python scripts/preprocess_stackoverflow.py --download
.venv/bin/python scripts/preprocess_stackoverflow.py --max-clients 5000
```

The download is about 8 GB, and the preparation needs the `tensorflow`
package. The transformer runs need the packages `transformers` and `peft`.

All settings are read from environment variables with the prefix `RL_`; each
script lists them at its top. `RL_DEVICE` is `cpu`, `mps` or `cuda`. A run
that is already in the output file is skipped, so an interrupted experiment
can be restarted with the same command.

A quick test of the installation (one run, a few minutes on a CPU):

```
RL_SEEDS=1 RL_MECHANISMS=tree RL_K_LEVELS=10 RL_E_GRID=1 RL_C_GRID=0.1 \
RL_OUT=results/smoke.json RL_PROJ_DIR=results/projections_smoke \
.venv/bin/python -m experiments.step46_mechanism_arm_v2
```

The experiments of the paper:

```
# Main comparison on EMNIST (160 runs)
.venv/bin/python -m experiments.step46_mechanism_arm_v2

# Square-root factorization and the optimized BandMF and BLT (60 runs)
RL_MECHANISMS=sqrtmf,bandmf_opt,blt_opt RL_K_LEVELS=10 \
RL_OUT=results/step46_mechanism_arm_emnist_production.json \
RL_PROJ_DIR=results/projections_step46_production \
.venv/bin/python -m experiments.step46_mechanism_arm_v2

# Sweep over the number of canary rounds (140 runs)
RL_MECHANISMS=fedavg,tree,bandmf_opt,blt_opt RL_K_LEVELS=1,2,4,5,20,25,50 \
RL_E_GRID=1 RL_C_GRID=0.1 \
RL_OUT=results/step46_mechanism_arm_emnist_ksweep.json \
RL_PROJ_DIR=results/projections_step46_ksweep \
.venv/bin/python -m experiments.step46_mechanism_arm_v2

# Two stricter certificates (20 runs each); repeat with 8 in place of 4
RL_TARGET_EPS=4 RL_MECHANISMS=fedavg,tree,bandmf_opt,blt_opt RL_K_LEVELS=10 \
RL_E_GRID=1 RL_C_GRID=0.1 \
RL_OUT=results/step46_mechanism_arm_emnist_eps4.json \
RL_PROJ_DIR=results/projections_step46_eps4 \
.venv/bin/python -m experiments.step46_mechanism_arm_v2

# StackOverflow LSTM (12 runs, about 50 minutes each on a laptop GPU)
RL_DATASET=stackoverflow RL_N_CLIENTS=500 RL_SEEDS=3 \
RL_MECHANISMS=fedavg,tree,bandmf_opt,blt_opt RL_K_LEVELS=10 \
RL_E_GRID=1 RL_C_GRID=1.0 \
RL_OUT=results/step46_mechanism_arm_stackoverflow.json \
RL_PROJ_DIR=results/projections_step46_so \
.venv/bin/python -m experiments.step46_mechanism_arm_v2

# Fault injection (180 runs)
.venv/bin/python -m experiments.step47_fault_injection

# Adaptive clipping, check of the signal model, misspecified covariance
.venv/bin/python -m experiments.step44_side_release
.venv/bin/python -m experiments.step45_retention
.venv/bin/python -m experiments.step42_mismatched_whitening
```

On a machine with little GPU memory, the LSTM runs can be made one per
process: add `RL_MAX_RUNS=1` and repeat the command until it exits with
code 0. The script exits with code 3 while runs remain.

## Reproducibility

The split of the data among clients, the canaries and the noise are seeded.
An EMNIST run is bit-for-bit reproducible on the same device. Runs on a GPU
and on a CPU differ in the last digits of the floating-point arithmetic; the
adaptive-clipping experiment and the check of the signal model were run on
both, and the two sets of records are included.

The stored records were produced with Python 3.12, PyTorch 2.15, NumPy 2.5
and SciPy 1.18.
