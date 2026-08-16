# Data and artifact layout

Datasets are not included. Rebuilt tensors must be placed under `data/nyc_dc/`
for the supplied NYC launch scripts.

The historical NYC bundle contains the following generated inputs:

| File | Purpose |
| --- | --- |
| `adjacency_matrix.npy` | Fixed physical zone adjacency |
| `edge_index.npy` | Directed road-edge index |
| `edge_lengths.npy` or `edge_lengths_osrm.npy` | Road-edge lengths |
| `edge_speeds.npy` | Historical directed-edge speed tensor |
| `node_demand.npy` | Zone demand counts |
| `node_supply.npy` | Estimated vacant-taxi supply |
| `targets_dc.npy` | Demand/supply forecasting targets |
| `time_features.npy` | Calendar and temporal features |
| `time_meta.csv` | Timestamp metadata used for split assignment |
| `observed_time_mask.npy` | Missing-time/observability mask |
| `splits.json` | Historical split metadata; training code recomputes containment |
| `zone_info.csv` | Zone metadata |
| `manifest.json` | Dataset dimensions and provenance |

Paper-exact evaluation additionally expects `paper_checkpoints/`,
`paper_cache/`, and `paper_weights/` in the repository root. These directories
are ignored by Git and must be restored from the private experiment archive.

The loader also has an optional `superzone_graph` integration point for the
routing profile. Its original paper-specific builder is not currently available.
