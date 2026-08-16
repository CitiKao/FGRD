# Reproduction Audit Notes

This package has two distinct reproducibility targets:

1. Reproduce the numerical GAP and V results currently reported in the paper.
2. Preserve the source checkpoint and calibration provenance without inventing
   metadata that is absent from the original runs.

## GAP

The source artifacts support the paper description:

- Both candidates use the full GAP ST-GAPV architecture.
- The 15-minute checkpoint is selected from the low-learning-rate resumed run.
- The 30- and 60-minute checkpoint is selected from the high-learning-rate
  resumed run.
- The packaged selector reproduces this choice using validation metrics only.

## V

The four source checkpoints share the same recorded full ST-GAPV architecture:

- two spatio-temporal blocks;
- fixed line-graph and adaptive edge-relation paths;
- temporal features enabled;
- edge-domain adaptive graph with top-k 16.

However, the available `stgat_meta.json` files do not record a random seed. The
training program default is seed 42, so the source artifacts do not substantiate
the stronger statement that all four streams were trained with different random
seeds. In addition, stream 04 was originally trained with a contiguous split,
although its cached predictions were evaluated on the common monthly validation
and test windows used by the final ensemble.

For provenance-safe paper wording, describe these inputs as:

> four selected ST-GAPV checkpoint prediction streams

Do not describe them as four different seeds unless a new controlled four-seed
experiment is trained and reported.

## Frozen V weights

The validation-selected candidate matrix contains four neural streams and six
causal historical candidates. At report steps 1, 2, and 4 (15, 30, and 60 min),
the frozen neural-weight sum is exactly zero. Consequently, the current reported
V values are reproduced by the selected historical candidates at those horizons.
This is an implementation fact that should be considered when describing the
forecasting contribution and ablation results.
