# FGRD Core Research Code

Core forecasting and evaluation code associated with **FGRD: Forecast-Guided
Proactive Taxi Dispatch With Regret-Calibrated Route Reranking**.

This repository is a provenance-preserving extraction of the latest source that
is currently available on the experiment machine. It contains the ST-GAPV
forecasting implementation, temporal split logic, physical/adaptive graph paths,
controlled GAP/V structural ablations, and the paper-exact checkpoint-composition
and speed-calibration utilities.

## Included

- GAP and speed (V) ST-GAPV model and trainer.
- Repeated within-month split: days 1-20 train, 21-24 validation, 25+ test, with
  complete history/target-window containment.
- Fixed node graph, directed line graph, learned sparse adaptive node/edge
  relations, and dual-path fusion.
- Two same-protocol Full controls and six NYC structural ablations.
- GAP validation-only checkpoint selection and V horizon-wise calibration tools.
- Small machine-readable copies of the paper-facing result tables.

See [docs/EXPERIMENT_INVENTORY.md](docs/EXPERIMENT_INVENTORY.md) for the
paper-to-code mapping and [docs/SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md)
for the known reproducibility boundary.

## Not included

- Raw or processed taxi/highway datasets.
- Generated adjacency, line-graph, edge-length, or superzone arrays.
- Checkpoints and prediction caches.
- The original formal baseline-adapter source used for Tables IV-V.
- The original 64-superzone builder, greedy dispatch, route-attention DDQN,
  stacked RCOG, and residual-safe training source used for Table IX. Those files
  were recorded under an older `D:\STDR` workspace that is not currently mounted.

No implementation from a different project has been substituted for missing
paper code.

## Environment

The historical runs used Python 3.10, PyTorch 2.1/CUDA 12.1, NumPy 1.26, and
pandas 2.2. Install a CUDA-compatible PyTorch build for GPU training, then run:

```powershell
python -m pip install -r requirements.txt
```

## Expected data layout

Place rebuilt experiment tensors under `data/nyc_dc/`. The loader expects the
schema documented in [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md). Dataset and
graph files are deliberately ignored by Git.

## Controlled NYC ablations

Run both Full controls and all six removals sequentially:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/controlled_ablation/01_run_all_8_controlled.ps1
```

The controller writes `controlled_ablation_results.csv` and JSON after all
selected jobs complete. Each removal must be compared only with its
same-configuration Full control.

## Paper-exact evaluation

The scripts under `scripts/paper_exact/` require the separately retained local
checkpoint/cache artifacts. They are not committed because they are large and
contain machine-specific provenance. After restoring those artifacts to the
paths listed by the preflight script, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/paper_exact/00_preflight.ps1
powershell -ExecutionPolicy Bypass -File scripts/paper_exact/01_evaluate_gap_paper_exact.ps1
powershell -ExecutionPolicy Bypass -File scripts/paper_exact/02_evaluate_v_paper_exact_from_cache.ps1
```

## Reproducibility note

The frozen V candidate matrix contains four neural checkpoint streams and six
causal historical candidates. At the three reported horizons, the retained
validation-selected weights assign zero aggregate weight to the neural streams.
This fact is preserved in `configs/PAPER_EXACT_METHOD_SPEC.json` and
`docs/PAPER_EXACT_AUDIT_NOTES.md`.

## Citation and license

Author and venue metadata can be added after the paper is accepted or publicly
posted. No open-source license has been selected yet; absent a license, reuse is
not granted automatically.
