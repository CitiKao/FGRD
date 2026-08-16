"""
data_loader.py — 資料載入與 PyTorch Dataset

提供：
  - load_nyc_real_graph_features() — 263 Zone 真實鄰接 + OSRM 邊長 + build_speed_features 產物
  - load_nyc_graph_for_rl()        — 僅路網與邊速統計，供 Double DQN 訓練
  - load_nyc_taxi_data()           — 舊版 CSV 流程（較少使用）

Dataset 以滑動窗口方式產生 (input, target) 組，供 STGAT 訓練使用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


NYC_TIME_FEATURE_MODE_CHOICES = (
    "baseline",
    "day_of_month",
    "week_of_month",
    "day_of_month_and_week_of_month",
)


def resolve_nyc_time_feature_mode(time_feature_mode: str) -> str:
    mode = str(time_feature_mode).strip().lower()
    if mode not in NYC_TIME_FEATURE_MODE_CHOICES:
        raise ValueError(
            "time_feature_mode must be one of "
            + ", ".join(NYC_TIME_FEATURE_MODE_CHOICES)
        )
    return mode


def edge_index_from_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    與 build_speed_features.build_edge_speeds 相同順序：列優先掃描 adj[i,j]>0。
    """
    adj = np.asarray(adj)
    edges: list[tuple[int, int]] = []
    n = adj.shape[0]
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                edges.append((i, j))
    return np.array(edges, dtype=np.int32)


def _load_nyc_adj_edge_lengths(
    root: Path,
    edge_length_source: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """載入 adjacency_matrix、edge_index、依邊展開的邊長 (km)。"""
    adj_path = root / "adjacency_matrix.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"找不到 {adj_path}，請先執行 build_adjacency.py")

    adj = np.load(adj_path).astype(np.float32)
    n = adj.shape[0]

    ei_path = root / "edge_index.npy"
    if ei_path.exists():
        edge_index = np.load(ei_path).astype(np.int32)
    else:
        edge_index = edge_index_from_adjacency(adj)

    if edge_length_source == "osrm":
        osrm_path = root / "edge_lengths_osrm.npy"
        cen_path = root / "edge_lengths.npy"
        if osrm_path.exists():
            len_mat = np.load(osrm_path).astype(np.float32)
        elif cen_path.exists():
            len_mat = np.load(cen_path).astype(np.float32)
        else:
            raise FileNotFoundError(
                f"需要 {osrm_path} 或 {cen_path}，請執行 update_edge_lengths_osrm.py 或 build_adjacency.py"
            )
    else:
        cen_path = root / "edge_lengths.npy"
        if not cen_path.exists():
            raise FileNotFoundError(f"找不到 {cen_path}")
        len_mat = np.load(cen_path).astype(np.float32)

    n_e = edge_index.shape[0]
    edge_lengths = np.zeros(n_e, dtype=np.float32)
    for e in range(n_e):
        i, j = int(edge_index[e, 0]), int(edge_index[e, 1])
        if i < n and j < n:
            edge_lengths[e] = float(len_mat[i, j])

    return adj, edge_index, edge_lengths


def _as_edge_slot_matrix(
    arr: np.ndarray,
    n_e: int,
    *,
    name: str,
) -> np.ndarray:
    """
    Normalize a speed array to shape (num_edges, num_time_slots).
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        if arr.shape[0] != n_e:
            raise ValueError(f"{name} 長度 {arr.shape[0]} 與邊數 {n_e} 不一致")
        return arr.reshape(n_e, 1).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"{name} 應為一維或二維，得到 {arr.shape}")
    if arr.shape[0] == n_e:
        return arr.astype(np.float32)
    if arr.shape[1] == n_e:
        return arr.T.astype(np.float32)
    raise ValueError(f"無法對齊邊數 n_e={n_e} 與 {name} 形狀 {arr.shape}")


def _resolve_default_shapefile() -> Path:
    for path in Path("NYCtaxizone").glob("*.shp"):
        return path
    raise FileNotFoundError("Could not find an NYC Taxi Zone shapefile under NYCtaxizone/")


def _build_temporal_features(
    root: Path,
    num_time_steps: int,
) -> Tuple[np.ndarray, list[str]]:
    """
    Build cyclical month / weekday / slot features from time_meta.csv.

    Returns
    -------
    temporal_features : (T, 6)
        [month_sin, month_cos, weekday_sin, weekday_cos, slot_sin, slot_cos]
    feature_names : list[str]
    """
    time_meta_path = root / "time_meta.csv"
    if not time_meta_path.exists():
        raise FileNotFoundError(f"找不到 {time_meta_path}，請先執行 build_speed_features.py")

    time_meta = pd.read_csv(time_meta_path)
    if len(time_meta) < num_time_steps:
        raise ValueError(
            f"time_meta 行數 {len(time_meta)} 少於時間步 {num_time_steps}"
        )
    time_meta = time_meta.iloc[:num_time_steps].copy()
    return build_temporal_features_from_time_meta(time_meta, time_feature_mode="baseline")


def build_temporal_features_from_time_meta(
    time_meta: "pd.DataFrame",
    *,
    time_feature_mode: str = "baseline",
) -> Tuple[np.ndarray, list[str]]:
    """
    Build cyclical calendar features from an in-memory time_meta table.

    Baseline mode preserves the original six features:
    [month_sin, month_cos, weekday_sin, weekday_cos, slot_sin, slot_cos].
    Optional modes append day-of-month and/or week-of-month position signals.
    """
    mode = resolve_nyc_time_feature_mode(time_feature_mode)
    if "date" not in time_meta.columns or "slot" not in time_meta.columns:
        raise ValueError("time_meta.csv 至少需要 date 與 slot 欄位")

    dates = pd.to_datetime(time_meta["date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("time_meta.csv 的 date 欄位存在無法解析的值")

    slot = pd.to_numeric(time_meta["slot"], errors="coerce")
    if slot.isna().any():
        raise ValueError("time_meta.csv 的 slot 欄位存在無法解析的值")

    if "day_of_week" in time_meta.columns:
        weekday = pd.to_numeric(time_meta["day_of_week"], errors="coerce")
    else:
        weekday = dates.dt.dayofweek.astype(np.float32)
    if weekday.isna().any():
        raise ValueError("time_meta.csv 的 day_of_week 欄位存在無法解析的值")

    month = dates.dt.month.astype(np.float32)
    weekday = weekday.astype(np.float32)
    slot = slot.astype(np.float32)
    slots_per_day = float(slot.max()) + 1.0
    if slots_per_day <= 0:
        raise ValueError("time_meta.csv 的 slot 欄位無法推導有效的每日時間槽數")

    month_angle = 2.0 * np.pi * (month - 1.0) / 12.0
    weekday_angle = 2.0 * np.pi * weekday / 7.0
    slot_angle = 2.0 * np.pi * slot / slots_per_day

    features = np.stack(
        [
            np.sin(month_angle),
            np.cos(month_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            np.sin(slot_angle),
            np.cos(slot_angle),
        ],
        axis=1,
    ).astype(np.float32)
    feature_names = [
        "month_sin",
        "month_cos",
        "weekday_sin",
        "weekday_cos",
        "slot_sin",
        "slot_cos",
    ]

    extra_features: list[np.ndarray] = []
    extra_names: list[str] = []
    if mode in {"day_of_month", "day_of_month_and_week_of_month"}:
        day_of_month = dates.dt.day.astype(np.float32)
        days_in_month = dates.dt.days_in_month.astype(np.float32)
        day_angle = 2.0 * np.pi * (day_of_month - 1.0) / days_in_month
        extra_features.extend([np.sin(day_angle), np.cos(day_angle)])
        extra_names.extend(["day_of_month_sin", "day_of_month_cos"])

    if mode in {"week_of_month", "day_of_month_and_week_of_month"}:
        day_of_month = dates.dt.day.astype(np.float32)
        days_in_month = dates.dt.days_in_month.astype(np.float32)
        week_of_month = np.floor((day_of_month - 1.0) / 7.0)
        weeks_in_month = np.floor((days_in_month - 1.0) / 7.0) + 1.0
        week_angle = 2.0 * np.pi * week_of_month / weeks_in_month
        extra_features.extend([np.sin(week_angle), np.cos(week_angle)])
        extra_names.extend(["week_of_month_sin", "week_of_month_cos"])

    if extra_features:
        features = np.concatenate(
            [features, np.stack(extra_features, axis=1).astype(np.float32)],
            axis=1,
        ).astype(np.float32)
        feature_names.extend(extra_names)
    return features, feature_names


def load_zone_metadata(
    data_dir: str | Path = "data",
    *,
    shapefile: str | Path | None = None,
) -> "pd.DataFrame":
    """
    Load zone metadata aligned with the graph node order.

    `zone_info.csv` generated by older runs may not contain `locationid`, so this
    helper backfills it from the shapefile while preserving row order.
    """
    import pandas as pd

    root = Path(data_dir)
    zone_info_path = root / "zone_info.csv"
    if not zone_info_path.exists():
        raise FileNotFoundError(f"Could not find {zone_info_path}")

    zone_info = pd.read_csv(zone_info_path)
    zone_info.columns = [str(col).strip().lower() for col in zone_info.columns]

    if "index" not in zone_info.columns:
        zone_info.insert(0, "index", np.arange(len(zone_info), dtype=np.int32))

    if "locationid" in zone_info.columns:
        zone_info["locationid"] = zone_info["locationid"].astype("Int64")
        return zone_info

    shp_path = Path(shapefile) if shapefile is not None else _resolve_default_shapefile()
    import geopandas as gpd

    gdf = gpd.read_file(shp_path)
    gdf.columns = [str(col).strip().lower() for col in gdf.columns]
    if "locationid" not in gdf.columns:
        raise ValueError(f"Shapefile {shp_path} does not contain a locationid column")
    if len(gdf) != len(zone_info):
        raise ValueError(
            f"zone_info rows ({len(zone_info)}) do not match shapefile rows ({len(gdf)})"
        )

    zone_info = zone_info.copy()
    zone_info["locationid"] = gdf["locationid"].astype("Int64").values
    if "zone_name" not in zone_info.columns and "zone" in gdf.columns:
        zone_info["zone_name"] = gdf["zone"].values
    if "borough" not in zone_info.columns and "borough" in gdf.columns:
        zone_info["borough"] = gdf["borough"].values
    return zone_info


def select_zone_indices_by_locationid_max(
    zone_info: "pd.DataFrame",
    locationid_max: int,
) -> np.ndarray:
    """Select node indices whose Taxi Zone LocationID is <= `locationid_max`."""
    if locationid_max <= 0:
        return zone_info["index"].to_numpy(dtype=np.int32)

    mask = zone_info["locationid"].astype("Int64") <= int(locationid_max)
    selected = zone_info.loc[mask, "index"].to_numpy(dtype=np.int32)
    if selected.size == 0:
        raise ValueError(
            f"No zones matched LocationID <= {locationid_max}; check zone metadata."
        )
    return selected


def build_induced_subgraph(
    num_nodes: int,
    edge_index: np.ndarray,
    edge_lengths: np.ndarray,
    selected_full_nodes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Build an induced directed subgraph from a full graph.

    Returns the subgraph edge list, edge lengths, adjacency, and node/edge mappings.
    """
    selected = np.asarray(selected_full_nodes, dtype=np.int32).reshape(-1)
    if selected.size == 0:
        raise ValueError("selected_full_nodes must not be empty")

    full_to_sub = np.full(num_nodes, -1, dtype=np.int32)
    full_to_sub[selected] = np.arange(selected.size, dtype=np.int32)

    src = edge_index[:, 0].astype(np.int32)
    dst = edge_index[:, 1].astype(np.int32)
    edge_mask = (full_to_sub[src] >= 0) & (full_to_sub[dst] >= 0)
    selected_edge_idx = np.flatnonzero(edge_mask).astype(np.int32)

    sub_edge_index = np.stack(
        [full_to_sub[src[edge_mask]], full_to_sub[dst[edge_mask]]],
        axis=1,
    ).astype(np.int32)
    sub_edge_lengths = edge_lengths[edge_mask].astype(np.float32)

    sub_num_nodes = int(selected.size)
    sub_adj = np.zeros((sub_num_nodes, sub_num_nodes), dtype=np.float32)
    if sub_edge_index.size > 0:
        sub_adj[sub_edge_index[:, 0], sub_edge_index[:, 1]] = 1.0

    return {
        "full_node_indices": selected,
        "full_to_sub": full_to_sub,
        "edge_mask": edge_mask,
        "full_edge_indices": selected_edge_idx,
        "adj": sub_adj,
        "edge_index": sub_edge_index,
        "edge_lengths": sub_edge_lengths,
    }


def load_nyc_real_graph_features(
    data_dir: str | Path = "data",
    *,
    max_time_steps: int = 0,
    edge_length_source: str = "osrm",
    add_time_features: bool = False,
) -> Dict[str, np.ndarray]:
    """
    載入紐約 263 Taxi Zone 真實路網與 `build_speed_features.py` 產生的時序特徵。

    必要檔案：
      - adjacency_matrix.npy
      - node_demand.npy, node_supply.npy  — (T, N)
      - edge_speeds.npy                   — (T, |E|)，與 edge_index 邊序一致
      - edge_index.npy（強烈建議；若缺則由 adj 重建）

    邊長（km，與速度 km/h 一致）：
      - edge_length_source=\"osrm\"  → edge_lengths_osrm.npy（優先），否則 edge_lengths.npy
      - edge_length_source=\"centroid\" → edge_lengths.npy

    Parameters
    ----------
    max_time_steps
        >0 時只取前 T 個時間步（除錯或減輕顯存）；0 表示使用全部。
    """
    root = Path(data_dir)
    adj, edge_index, edge_lengths = _load_nyc_adj_edge_lengths(root, edge_length_source)
    n = adj.shape[0]
    n_e = edge_index.shape[0]

    dem_path = root / "node_demand.npy"
    sup_path = root / "node_supply.npy"
    spd_path = root / "edge_speeds.npy"
    for p in (dem_path, sup_path, spd_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到 {p}，請先執行 build_speed_features.py")

    demand = np.load(dem_path).astype(np.float32)
    supply = np.load(sup_path).astype(np.float32)
    edge_speeds = np.load(spd_path).astype(np.float32)

    if demand.shape != supply.shape:
        raise ValueError(f"node_demand {demand.shape} 與 node_supply {supply.shape} 不一致")

    if edge_speeds.ndim != 2:
        raise ValueError(f"edge_speeds 應為二維，得到 {edge_speeds.shape}")

    # 標準為 (T, |E|)；若存成 (|E|, T) 則轉置
    if edge_speeds.shape[1] == n_e:
        pass
    elif edge_speeds.shape[0] == n_e:
        edge_speeds = edge_speeds.T
    else:
        raise ValueError(
            f"無法對齊邊數 n_e={n_e} 與 edge_speeds 形狀 {edge_speeds.shape}"
        )

    t_feat = demand.shape[0]
    if edge_speeds.shape[0] != t_feat:
        raise ValueError(
            f"時間步不一致: demand T={t_feat}, edge_speeds {edge_speeds.shape} "
            f"（預期 (T, |E|) 且 T={t_feat}）"
        )
    if edge_speeds.shape[1] != n_e:
        raise ValueError(
            f"邊數不一致: edge_index 有 {n_e} 條邊, edge_speeds 第二維為 {edge_speeds.shape[1]}"
        )

    if demand.shape[1] != n or supply.shape[1] != n:
        raise ValueError(f"節點數應為 {n}，得到 demand {demand.shape}")

    if max_time_steps > 0:
        t_use = min(t_feat, max_time_steps)
        demand = demand[:t_use]
        supply = supply[:t_use]
        edge_speeds = edge_speeds[:t_use]
    else:
        t_use = t_feat
    base_node_features = np.stack([demand, supply], axis=-1)
    time_feature_names: list[str] = []
    if add_time_features:
        temporal_features, time_feature_names = _build_temporal_features(root, t_use)
        temporal_node_features = np.broadcast_to(
            temporal_features[:, None, :],
            (t_use, n, temporal_features.shape[1]),
        )
        node_features = np.concatenate(
            [base_node_features, temporal_node_features],
            axis=-1,
        ).astype(np.float32)
    else:
        node_features = base_node_features.astype(np.float32)

    return {
        "adj": adj,
        "edge_index": edge_index,
        "edge_lengths": edge_lengths,
        "node_features": node_features,
        "edge_speeds": edge_speeds,
        "time_feature_names": time_feature_names,
        "source": "nyc_real",
    }


def load_nyc_graph_for_rl(
    data_dir: str | Path = "data",
    *,
    edge_length_source: str = "osrm",
    speed_seed: int = 42,
    routing_locationid_max: int = 63,
    routing_graph_mode: str = "legacy",
    superzone_dir: str | Path | None = None,
    shapefile: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """
    載入 Double DQN 用路網：不需 node_demand / node_supply，但需邊速檔。

    優先使用 ``edge_speeds_avg.npy`` 作為跨日平均時槽速度序列；若無則退回
    ``edge_speeds.npy``。回傳每邊的動態預測速度序列，以及由其微擾動得到的
    動態「真實」速度序列，供 RL 訓練 / 評估使用。
    """
    root = Path(data_dir)
    adj, edge_index, edge_lengths = _load_nyc_adj_edge_lengths(root, edge_length_source)
    n_e = edge_index.shape[0]

    spd_path = root / "edge_speeds.npy"
    avg_path = root / "edge_speeds_avg.npy"

    profile_source = ""
    if avg_path.exists():
        pred_speed_profile = _as_edge_slot_matrix(
            np.load(avg_path),
            n_e,
            name="edge_speeds_avg",
        )
        profile_source = "edge_speeds_avg"
    elif spd_path.exists():
        pred_speed_profile = _as_edge_slot_matrix(
            np.load(spd_path, mmap_mode="r"),
            n_e,
            name="edge_speeds",
        )
        profile_source = "edge_speeds"
    else:
        raise FileNotFoundError(
            f"需要 {spd_path} 或 {avg_path}（請先執行 build_speed_features.py）"
        )

    pred_speed_profile = np.maximum(pred_speed_profile.astype(np.float32), 1.0)
    avg_speeds = pred_speed_profile.mean(axis=1).astype(np.float32)

    if avg_speeds.shape[0] != n_e:
        raise ValueError(
            f"邊速長度 {avg_speeds.shape[0]} 與 edge_index 邊數 {n_e} 不一致"
        )

    num_time_slots = int(pred_speed_profile.shape[1])
    if profile_source == "edge_speeds_avg" and num_time_slots > 0 and 1440 % num_time_slots == 0:
        time_slot_minutes = int(1440 // num_time_slots)
    else:
        time_slot_minutes = 15

    mode = str(routing_graph_mode).strip().lower()
    if mode == "superzone":
        from superzone_graph import aggregate_speed_profile_to_rl_edges, load_superzone_artifacts

        artifacts = load_superzone_artifacts(root, superzone_dir)
        rl_edge_index = artifacts["rl_edge_index"].astype(np.int32)
        num_regions = int(artifacts["region_demand"].shape[1])
        rl_adj = np.zeros((num_regions, num_regions), dtype=np.float32)
        if rl_edge_index.size:
            rl_adj[rl_edge_index[:, 0], rl_edge_index[:, 1]] = 1.0

        pred_speed_profile = aggregate_speed_profile_to_rl_edges(
            pred_speed_profile,
            artifacts["rl_edge_speed_mapping_offsets"],
            artifacts["rl_edge_speed_mapping_indices"],
            artifacts["rl_edge_speeds_kmh"],
        )
        rng = np.random.RandomState(speed_seed)
        real_speed_profile = (
            pred_speed_profile * rng.uniform(0.7, 1.3, size=pred_speed_profile.shape)
        ).astype(np.float32)

        return {
            "adj": rl_adj,
            "edge_index": rl_edge_index,
            "edge_lengths": artifacts["rl_edge_lengths"].astype(np.float32),
            "avg_speeds": pred_speed_profile.mean(axis=1).astype(np.float32),
            "real_speeds": real_speed_profile.mean(axis=1).astype(np.float32),
            "pred_speed_profile": pred_speed_profile.astype(np.float32),
            "real_speed_profile": real_speed_profile.astype(np.float32),
            "num_time_slots": num_time_slots,
            "time_slot_minutes": int(time_slot_minutes),
            "full_node_indices": np.arange(num_regions, dtype=np.int32),
            "full_to_sub": {int(i): int(i) for i in range(num_regions)},
            "full_edge_indices": np.arange(rl_edge_index.shape[0], dtype=np.int32),
            "zone_info": artifacts["region_info"],
            "routing_locationid_max": 0,
            "routing_graph_mode": "superzone",
            "superzone_dir": str(artifacts["root"]),
            "speed_profile_source": profile_source,
        }

    if mode != "legacy":
        raise ValueError("routing_graph_mode must be either 'legacy' or 'superzone'")

    rng = np.random.RandomState(speed_seed)
    real_speed_profile = (
        pred_speed_profile * rng.uniform(0.7, 1.3, size=pred_speed_profile.shape)
    ).astype(np.float32)
    real_speeds = real_speed_profile.mean(axis=1).astype(np.float32)

    if routing_locationid_max > 0:
        zone_info = load_zone_metadata(data_dir, shapefile=shapefile)
        selected_nodes = select_zone_indices_by_locationid_max(zone_info, routing_locationid_max)
    else:
        import pandas as pd

        selected_nodes = np.arange(adj.shape[0], dtype=np.int32)
        zone_info = pd.DataFrame(
            {
                "index": selected_nodes,
                "locationid": selected_nodes + 1,
                "sub_index": selected_nodes,
            }
        )
    subgraph = build_induced_subgraph(adj.shape[0], edge_index, edge_lengths, selected_nodes)
    sub_edge_idx = subgraph["full_edge_indices"]

    selected_zone_info = zone_info.loc[
        zone_info["index"].isin(selected_nodes)
    ].copy()
    selected_zone_info["sub_index"] = selected_zone_info["index"].map(
        {int(full_idx): i for i, full_idx in enumerate(selected_nodes.tolist())}
    )
    selected_zone_info = selected_zone_info.sort_values("sub_index").reset_index(drop=True)

    return {
        "adj": subgraph["adj"],
        "edge_index": subgraph["edge_index"],
        "edge_lengths": subgraph["edge_lengths"],
        "avg_speeds": avg_speeds[sub_edge_idx].astype(np.float32),
        "real_speeds": real_speeds[sub_edge_idx].astype(np.float32),
        "pred_speed_profile": pred_speed_profile[sub_edge_idx].astype(np.float32),
        "real_speed_profile": real_speed_profile[sub_edge_idx].astype(np.float32),
        "num_time_slots": num_time_slots,
        "time_slot_minutes": int(time_slot_minutes),
        "full_node_indices": subgraph["full_node_indices"],
        "full_to_sub": subgraph["full_to_sub"],
        "full_edge_indices": subgraph["full_edge_indices"],
        "zone_info": selected_zone_info,
        "routing_locationid_max": int(routing_locationid_max),
        "routing_graph_mode": "legacy",
        "speed_profile_source": profile_source,
    }


# ════════════════════════════════════════════════════════════════
#  NYC Taxi 資料載入（可選）
# ════════════════════════════════════════════════════════════════

def load_nyc_taxi_data(
    trip_csv: str,
    num_zones: int = 63,
    time_slot_minutes: int = 15,
    adj_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    從 NYC TLC CSV 載入資料並聚合為時空特徵。

    需要的 CSV 欄位：
      tpep_pickup_datetime, tpep_dropoff_datetime,
      PULocationID, DOLocationID, trip_distance

    Parameters
    ----------
    trip_csv         : CSV 檔路徑
    num_zones        : 使用的區域數（取 LocationID < num_zones）
    time_slot_minutes: 時間槽長度
    adj_path         : 鄰接矩陣 .npy 路徑（若無則自動從行程推導）
    """
    import pandas as pd

    df = pd.read_csv(trip_csv, parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"])
    df = df[(df["PULocationID"] < num_zones) & (df["DOLocationID"] < num_zones)]
    df["duration_min"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60.0
    df = df[(df["duration_min"] > 1) & (df["duration_min"] < 180)]

    # 時間槽
    df["slot"] = (
        (df["tpep_pickup_datetime"] - df["tpep_pickup_datetime"].min())
        .dt.total_seconds()
        / (time_slot_minutes * 60)
    ).astype(int)
    T = df["slot"].max() + 1
    N = num_zones

    # 需求：每區域每時間槽的上車數
    demand = np.zeros((T, N), dtype=np.float32)
    grp = df.groupby(["slot", "PULocationID"]).size()
    for (t, z), cnt in grp.items():
        if t < T and z < N:
            demand[t, z] = cnt

    # 空車估計：下車數 − 上車數（累積 + 初始值）
    dropoff_cnt = np.zeros((T, N), dtype=np.float32)
    grp2 = df.groupby(["slot", "DOLocationID"]).size()
    for (t, z), cnt in grp2.items():
        if t < T and z < N:
            dropoff_cnt[t, z] = cnt

    supply = np.zeros((T, N), dtype=np.float32)
    supply[0] = 5.0  # 初始每區 5 台空車
    for t in range(1, T):
        supply[t] = np.maximum(supply[t - 1] + dropoff_cnt[t - 1] - demand[t - 1], 0)

    node_features = np.stack([demand, supply], axis=-1)

    # 鄰接矩陣
    if adj_path and Path(adj_path).exists():
        adj = np.load(adj_path).astype(np.float32)
    else:
        adj = np.zeros((N, N), dtype=np.float32)
        pairs = df.groupby(["PULocationID", "DOLocationID"]).size()
        for (i, j), cnt in pairs.items():
            if i < N and j < N and cnt > 5:
                adj[i, j] = 1.0
        np.fill_diagonal(adj, 0)

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols], axis=1)
    nE = edge_index.shape[0]

    # 邊速度與長度
    edge_lengths = np.ones(nE, dtype=np.float32) * 2.0  # 預設 2km
    edge_speeds = np.full((T, nE), 30.0, dtype=np.float32)  # 預設 30km/h

    for idx, (i, j) in enumerate(edge_index):
        sub = df[(df["PULocationID"] == i) & (df["DOLocationID"] == j)]
        if len(sub) > 0:
            mean_dist = sub["trip_distance"].mean() * 1.609  # mile → km
            edge_lengths[idx] = max(mean_dist, 0.1)
        for t in range(T):
            tsub = sub[sub["slot"] == t]
            if len(tsub) >= 2:
                avg_speed = (tsub["trip_distance"].mean() * 1.609) / (
                    tsub["duration_min"].mean() / 60.0 + 1e-5
                )
                edge_speeds[t, idx] = np.clip(avg_speed, 5.0, 130.0)

    return {
        "adj": adj,
        "edge_index": edge_index,
        "edge_lengths": edge_lengths,
        "node_features": node_features,
        "edge_speeds": edge_speeds,
    }


# ════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ════════════════════════════════════════════════════════════════

_DEFAULT_EDGE_SPEED_HISTORY_MASK = object()


class SpatioTemporalDataset(Dataset):
    """
    滑動窗口 Dataset。

    每筆樣本：
      input : (node_seq (N, h, C),  speed_seq (|E|, h))
      target: (demand (N, p),  supply (N, p),  speed (|E|, p))
    """

    def __init__(
        self,
        node_features: np.ndarray,    # (T, N, C)
        edge_speeds: np.ndarray,      # (T, |E|)
        *,
        edge_speed_valid_mask: np.ndarray | None = None,
        edge_speed_history_valid_mask: np.ndarray | None | object = _DEFAULT_EDGE_SPEED_HISTORY_MASK,
        history_imputation_enabled: bool = True,
        hist_len: int = 12,
        pred_horizon: int = 3,
    ) -> None:
        super().__init__()
        self.node_feat = node_features.astype(np.float32)
        self.edge_speed = edge_speeds.astype(np.float32)
        self.edge_speed_valid_mask = (
            None
            if edge_speed_valid_mask is None
            else np.asarray(edge_speed_valid_mask, dtype=bool)
        )
        if edge_speed_history_valid_mask is _DEFAULT_EDGE_SPEED_HISTORY_MASK:
            self.edge_speed_history_valid_mask = self.edge_speed_valid_mask
        elif edge_speed_history_valid_mask is None:
            self.edge_speed_history_valid_mask = None
        else:
            self.edge_speed_history_valid_mask = np.asarray(edge_speed_history_valid_mask, dtype=bool)
        if self.edge_speed_valid_mask is not None and self.edge_speed_valid_mask.shape != self.edge_speed.shape:
            raise ValueError(
                "edge_speed_valid_mask must have the same shape as edge_speeds "
                f"({self.edge_speed_valid_mask.shape} != {self.edge_speed.shape})."
            )
        if (
            self.edge_speed_history_valid_mask is not None
            and self.edge_speed_history_valid_mask.shape != self.edge_speed.shape
        ):
            raise ValueError(
                "edge_speed_history_valid_mask must have the same shape as edge_speeds "
                f"({self.edge_speed_history_valid_mask.shape} != {self.edge_speed.shape})."
            )
        self.history_imputation_enabled = bool(history_imputation_enabled)
        self.h = hist_len
        self.p = pred_horizon
        self.total = node_features.shape[0] - hist_len - pred_horizon + 1

    def __len__(self) -> int:
        return max(self.total, 0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t = idx + self.h  # 窗口結束位置

        # input
        node_seq = self.node_feat[idx: t]                # (h, N, C)
        speed_seq = self.edge_speed[idx: t]              # (h, |E|)
        speed_history_mask = (
            None
            if self.edge_speed_history_valid_mask is None
            else self.edge_speed_history_valid_mask[idx: t]
        )
        if self.history_imputation_enabled and speed_history_mask is not None:
            speed_seq = self._causal_forward_fill_speed_window(speed_seq, speed_history_mask)

        # target
        demand_target = self.node_feat[t: t + self.p, :, 0].T   # (N, p)
        supply_target = self.node_feat[t: t + self.p, :, 1].T   # (N, p)
        speed_target = self.edge_speed[t: t + self.p].T          # (|E|, p)
        speed_target_mask = (
            None
            if self.edge_speed_valid_mask is None
            else self.edge_speed_valid_mask[t: t + self.p].T
        )

        sample = {
            "node_seq": torch.from_numpy(node_seq.transpose(1, 0, 2)),   # (N, h, C)
            "speed_seq": torch.from_numpy(speed_seq.T),                   # (|E|, h)
            "demand_target": torch.from_numpy(demand_target),
            "supply_target": torch.from_numpy(supply_target),
            "speed_target": torch.from_numpy(speed_target),
        }
        if speed_history_mask is not None:
            sample["speed_history_mask"] = torch.from_numpy(speed_history_mask.T)
        if speed_target_mask is not None:
            sample["speed_target_mask"] = torch.from_numpy(speed_target_mask)
        return sample

    @staticmethod
    def _causal_forward_fill_speed_window(
        speed_seq: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        filled = np.asarray(speed_seq, dtype=np.float32).copy()
        mask = np.asarray(valid_mask, dtype=bool)
        last = np.zeros(filled.shape[1], dtype=np.float32)
        has_last = np.zeros(filled.shape[1], dtype=bool)
        for step in range(filled.shape[0]):
            valid = mask[step]
            filled[step, ~valid & has_last] = last[~valid & has_last]
            last[valid] = filled[step, valid]
            has_last[valid] = True
        return filled.astype(np.float32)
