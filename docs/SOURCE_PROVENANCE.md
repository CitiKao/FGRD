# Source provenance and availability

## Located source

The forecasting core and controlled-ablation source were recovered from the
local archive package:

```text
STGAPV_NYC_Controlled_Ablation_RTX4090_20260727
```

The paper-exact checkpoint-composition and calibration utilities were recovered
from the local archive package:

```text
STGAPV_PaperExact_GAP_V_NYC_20260727
```

The common core files in those two packages have matching SHA-256 hashes. The
controlled package is used as the canonical copy in this repository.

## Missing formal source

Historical manifests identify the original full workspace as `D:\STDR`.
Neither that path nor the earlier `E:\FGRD`, `E:\OK`, and
`E:\paper_experiment_results_20260616` archives are mounted at the time of this
release audit. Therefore the following exact source cannot presently be
included:

- raw NYC/OSRM preprocessing and graph-generation scripts;
- formal multi-dataset baseline adapters used in Tables IV-V;
- 64-superzone construction and dense/sparse dispatch graph builders;
- greedy fleet allocation;
- candidate-route generation and route-attention DDQN training;
- stacked RCOG and residual-safe model training/evaluation.

Only the corresponding small routing result table is retained. A future source
recovery should add these modules under their original provenance, not replace
them with similarly named code from another project.

## Result sources

- Controlled structural ablation: run
  `nyc_gap_v_controlled_ablation_20260727_112641`, completed 2026-07-29.
- Paper-exact GAP/V references: frozen package dated 2026-07-27.
- Final paper used for scope: `FGRD_latest_main.pdf`, compiled 2026-08-01.
