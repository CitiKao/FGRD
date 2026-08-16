from __future__ import annotations

from typing import Any

import numpy as np

NORMALIZATION_EPS = 1e-6


def _ensure_nonnegative(name: str, values: np.ndarray) -> None:
    min_value = float(np.min(values))
    if min_value < 0:
        raise ValueError(
            f"{name} contains negative values (min={min_value:.6f}), "
            "but log1p normalization expects nonnegative inputs."
        )


def _safe_std(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    return np.where(arr < NORMALIZATION_EPS, 1.0, arr).astype(np.float32)


def build_normalization_stats(
    node_features: np.ndarray,
    edge_speeds: np.ndarray,
    train_time_mask: np.ndarray,
    *,
    speed_valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if node_features.ndim != 3 or node_features.shape[-1] < 2:
        raise ValueError(
            "node_features must have shape (T, N, C) with demand/supply in the first two channels."
        )
    if edge_speeds.ndim != 2:
        raise ValueError("edge_speeds must have shape (T, |E|).")

    train_time_mask = np.asarray(train_time_mask, dtype=bool)
    if train_time_mask.ndim != 1 or train_time_mask.shape[0] != node_features.shape[0]:
        raise ValueError("train_time_mask must be a 1D boolean array aligned with the time axis.")
    if not np.any(train_time_mask):
        raise ValueError("train_time_mask does not select any training time steps.")

    demand = np.asarray(node_features[..., 0], dtype=np.float32)
    supply = np.asarray(node_features[..., 1], dtype=np.float32)
    speed = np.asarray(edge_speeds, dtype=np.float32)

    _ensure_nonnegative("demand", demand)
    _ensure_nonnegative("supply", supply)

    demand_log = np.log1p(demand).astype(np.float32)
    supply_log = np.log1p(supply).astype(np.float32)

    demand_train = demand_log[train_time_mask]
    supply_train = supply_log[train_time_mask]
    speed_train = speed[train_time_mask]
    if speed_valid_mask is not None:
        valid_mask = np.asarray(speed_valid_mask, dtype=bool)
        if valid_mask.shape != speed.shape:
            raise ValueError(
                "speed_valid_mask must have the same shape as edge_speeds "
                f"({valid_mask.shape} != {speed.shape})."
            )
        speed_train_valid = valid_mask[train_time_mask]
        counts = speed_train_valid.sum(axis=0).astype(np.float32)
        safe_counts = np.maximum(counts, 1.0)
        speed_sum = (speed_train * speed_train_valid).sum(axis=0)
        speed_mean = (speed_sum / safe_counts).astype(np.float32)
        centered = np.where(speed_train_valid, speed_train - speed_mean, 0.0)
        speed_var = (centered * centered).sum(axis=0) / safe_counts
        missing_edges = counts < 1.0
        if np.any(missing_edges):
            fallback_mean = speed_train.mean(axis=0).astype(np.float32)
            fallback_std = _safe_std(speed_train.std(axis=0))
            speed_mean[missing_edges] = fallback_mean[missing_edges]
            speed_std = _safe_std(np.sqrt(speed_var).astype(np.float32))
            speed_std[missing_edges] = fallback_std[missing_edges]
        else:
            speed_std = _safe_std(np.sqrt(speed_var).astype(np.float32))
    else:
        speed_mean = speed_train.mean(axis=0).astype(np.float32)
        speed_std = _safe_std(speed_train.std(axis=0))

    return {
        "demand": {
            "transform": "log1p_zscore",
            "mean": float(demand_train.mean()),
            "std": float(max(float(demand_train.std()), NORMALIZATION_EPS)),
        },
        "supply": {
            "transform": "log1p_zscore",
            "mean": float(supply_train.mean()),
            "std": float(max(float(supply_train.std()), NORMALIZATION_EPS)),
        },
        "speed": {
            "transform": "per_edge_zscore",
            "mean": speed_mean,
            "std": speed_std,
        },
    }


def normalize_node_features(
    node_features: np.ndarray,
    normalization_stats: dict[str, Any] | None,
) -> np.ndarray:
    arr = np.asarray(node_features, dtype=np.float32).copy()
    if normalization_stats is None:
        return arr
    if arr.shape[-1] < 2:
        raise ValueError("node_features must include demand and supply channels in the last dimension.")

    demand_stats = normalization_stats["demand"]
    supply_stats = normalization_stats["supply"]
    arr[..., 0] = (
        np.log1p(arr[..., 0]) - float(demand_stats["mean"])
    ) / float(demand_stats["std"])
    arr[..., 1] = (
        np.log1p(arr[..., 1]) - float(supply_stats["mean"])
    ) / float(supply_stats["std"])
    return arr.astype(np.float32)


def normalize_speed_features(
    speed_values: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    *,
    edge_axis: int,
    speed_valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(speed_values, dtype=np.float32).copy()
    if normalization_stats is None:
        return arr

    speed_stats = normalization_stats["speed"]
    mean = np.asarray(speed_stats["mean"], dtype=np.float32)
    std = np.asarray(speed_stats["std"], dtype=np.float32)
    axis = edge_axis % arr.ndim
    shape = [1] * arr.ndim
    shape[axis] = mean.shape[0]
    normalized = ((arr - mean.reshape(shape)) / std.reshape(shape)).astype(np.float32)
    if speed_valid_mask is not None:
        valid_mask = np.asarray(speed_valid_mask, dtype=bool)
        if valid_mask.shape != arr.shape:
            raise ValueError(
                "speed_valid_mask must have the same shape as speed_values "
                f"({valid_mask.shape} != {arr.shape})."
            )
        normalized = np.where(valid_mask, normalized, 0.0).astype(np.float32)
    return normalized


def denormalize_count_values(
    normalized_values: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    *,
    task: str,
) -> np.ndarray:
    arr = np.asarray(normalized_values, dtype=np.float32)
    if normalization_stats is None:
        return arr

    task_stats = normalization_stats[task]
    return np.expm1(arr * float(task_stats["std"]) + float(task_stats["mean"])).astype(np.float32)


def denormalize_speed_values(
    normalized_values: np.ndarray,
    normalization_stats: dict[str, Any] | None,
    *,
    edge_axis: int,
) -> np.ndarray:
    arr = np.asarray(normalized_values, dtype=np.float32)
    if normalization_stats is None:
        return arr

    speed_stats = normalization_stats["speed"]
    mean = np.asarray(speed_stats["mean"], dtype=np.float32)
    std = np.asarray(speed_stats["std"], dtype=np.float32)
    axis = edge_axis % arr.ndim
    shape = [1] * arr.ndim
    shape[axis] = mean.shape[0]
    return (arr * std.reshape(shape) + mean.reshape(shape)).astype(np.float32)


def serialize_normalization_stats(
    normalization_stats: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if normalization_stats is None:
        return None
    return {
        "demand": {
            "transform": normalization_stats["demand"]["transform"],
            "mean": float(normalization_stats["demand"]["mean"]),
            "std": float(normalization_stats["demand"]["std"]),
        },
        "supply": {
            "transform": normalization_stats["supply"]["transform"],
            "mean": float(normalization_stats["supply"]["mean"]),
            "std": float(normalization_stats["supply"]["std"]),
        },
        "speed": {
            "transform": normalization_stats["speed"]["transform"],
            "mean": np.asarray(normalization_stats["speed"]["mean"], dtype=np.float32).tolist(),
            "std": np.asarray(normalization_stats["speed"]["std"], dtype=np.float32).tolist(),
        },
    }


def load_normalization_stats(
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not payload:
        return None
    return {
        "demand": {
            "transform": str(payload["demand"]["transform"]),
            "mean": float(payload["demand"]["mean"]),
            "std": float(payload["demand"]["std"]),
        },
        "supply": {
            "transform": str(payload["supply"]["transform"]),
            "mean": float(payload["supply"]["mean"]),
            "std": float(payload["supply"]["std"]),
        },
        "speed": {
            "transform": str(payload["speed"]["transform"]),
            "mean": np.asarray(payload["speed"]["mean"], dtype=np.float32),
            "std": _safe_std(np.asarray(payload["speed"]["std"], dtype=np.float32)),
        },
    }
