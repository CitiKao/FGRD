# Paper experiment inventory

This inventory was checked against the final 16-page PDF compiled on
2026-08-01. `Available` means the exact source was found on a currently mounted
drive. `Partial` means result artifacts or shared core code exist but a formal
experiment-specific implementation is missing.

| Paper item | Dataset / protocol | Code or result location | Status |
| --- | --- | --- | --- |
| ST-GAPV GAP main model | Shanghai, NYC, Chengdu; 15/30/60 min | `data_loader.py`, `stgat_model.py`, `tools/chengdu_gap_arch_smoke.py` | Available core; datasets absent |
| ST-GAPV speed main model | NYC, METR-LA, PEMS-BAY; 15/30/60 min | `train_predictor.py`, `stgat_model.py`, `tools/ensemble_*speed_predictions.py` | Available core; datasets absent |
| Within-month split | Days 1-20 / 21-24 / 25+ | `train_predictor.py` split helpers | Available |
| Physical graph and directed line graph | Zone nodes and road/sensor edges | `data_loader.py`, `stgat_model.py` | Available from generated inputs |
| Adaptive node/edge relations | Sparse top-k learned graphs | `stgat_model.py` | Available |
| Forecasting baselines in Tables IV-V | HA, ARIMA/XGBoost, LSTM, STGCN, DCRNN, Graph WaveNet and paper-specific adapters | Paper result table only | Formal adapter source unavailable |
| NYC GAP structure ablation | Full, no adaptive, one ST block, no fixed graph | `tools/run_nyc_gap_v_controlled_ablation_suite.py` | Available |
| NYC V structure ablation | Full, no adaptive speed matrix, one ST block, no fixed graph | Same controlled runner | Available |
| GAP checkpoint selection | Validation-selected full checkpoints | `tools/compose_paper_gap_results.py` | Available; checkpoints excluded |
| V checkpoint/candidate calibration | Four streams plus six causal candidates | `tools/verify_paper_v_from_cache.py`, calibration and ensemble tools | Available; cache/weights excluded |
| Greedy fleet rebalancing | 64 NYC superzones | Result/config descriptions only | Formal source unavailable |
| Candidate-route generation and DDQN | Six routes per OD pair | `results/paper_tables/PAPER_ROUTING_ABLATION.csv` | Formal source unavailable |
| Stacked RCOG and residual-safe scorer | Full-year routing evaluation | Same routing result table | Formal source unavailable |
| GAP map, volatile-speed plot, RCOG route case | Figures 6-8 | Final figure images exist outside this core repository | Figure-generation source not found |

## Experiments deliberately excluded

- Strict 1-8/9-10/11-12 month splitting and rolling-origin trials: performed
  locally but not reported in the final paper.
- Old two/three-epoch and seed-42-only exploratory ablations superseded by the
  final controlled table.
- Shenzhen and smoke/debug runs not present in the final paper.
- The newer FOIL-2013 ride-pooling project, which is a separate experiment and
  must not be presented as the FGRD routing implementation.
