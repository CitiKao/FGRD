from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import SpatioTemporalDataset  # noqa: E402
from external_speed_benchmarks.train_sensor_speed import (  # noqa: E402
    build_history_time_mask,
    load_prepared_sensor_dataset,
    resolve_outlier_cleaning,
)
from predictor_normalization import (  # noqa: E402
    denormalize_speed_values,
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor  # noqa: E402
from train_predictor import (  # noqa: E402
    build_monthly_split_indices,
    build_window_time_mask,
    configure_cuda_runtime,
    extract_temporal_context,
    resolve_device,
    resolve_num_workers,
    resolve_precision,
)


HORIZONS = ("15min", "30min", "60min")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def resolve_checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state" in payload:
        state = payload["model_state"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a model state dict: {path}")
    return state


def infer_stgat_config(state: dict[str, torch.Tensor]) -> dict[str, int]:
    block_ids = sorted(
        {
            int(parts[1])
            for key in state
            for parts in [key.split(".")]
            if len(parts) > 3 and parts[0] == "n_gtcn_fix" and parts[2] == "layers"
        }
    )
    layer_ids = sorted(
        {
            int(parts[3])
            for key in state
            for parts in [key.split(".")]
            if len(parts) > 4 and parts[0] == "n_gtcn_fix" and parts[2] == "layers"
        }
    )
    if not block_ids or not layer_ids:
        raise ValueError("Unable to infer STGAT block/layer counts from checkpoint.")
    return {
        "hidden_dim": int(state["node_proj.weight"].shape[0]),
        "node_feat_dim": int(state["node_proj.weight"].shape[1]),
        "pred_horizon": int(state["speed_head.weight"].shape[0]),
        "num_heads": int(state["n_gat_fix.0.a_q"].shape[2]),
        "num_st_blocks": int(max(block_ids) + 1),
        "num_gtcn_layers": int(max(layer_ids) + 1),
        "kernel_size": int(state["n_gtcn_fix.0.layers.0.gate_conv.weight"].shape[-1]),
    }


def build_metric_bucket() -> dict[str, float]:
    return {"se": 0.0, "ae": 0.0, "count": 0.0, "ape": 0.0, "mape_count": 0.0}


def update_metric_bucket(bucket: dict[str, float], pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> None:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    valid_diff = diff[mask]
    bucket["se"] += float(np.square(valid_diff).sum())
    bucket["ae"] += float(np.abs(valid_diff).sum())
    bucket["count"] += float(valid_diff.size)
    mape_mask = mask & (np.abs(target) > 1e-6)
    if np.any(mape_mask):
        bucket["ape"] += float((np.abs(diff[mape_mask]) / np.abs(target[mape_mask]) * 100.0).sum())
        bucket["mape_count"] += float(mape_mask.sum())


def finalize_metric_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(float(bucket["count"]), 1.0)
    mse = float(bucket["se"] / count)
    result = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(bucket["ae"] / count),
    }
    if bucket["mape_count"] > 0:
        result["mape"] = float(bucket["ape"] / bucket["mape_count"])
    return result


def compute_speed_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    *,
    report_steps: dict[str, int],
    slot_minutes: int = 5,
) -> dict[str, Any]:
    overall = build_metric_bucket()
    update_metric_bucket(overall, pred, target, mask)
    report: dict[str, dict[str, float]] = {}
    per_step: dict[str, dict[str, float]] = {}
    for step_idx in range(pred.shape[-1]):
        bucket = build_metric_bucket()
        update_metric_bucket(bucket, pred[..., step_idx], target[..., step_idx], mask[..., step_idx])
        per_step[f"step_{step_idx + 1}"] = {
            "step": int(step_idx + 1),
            "minutes": int((step_idx + 1) * slot_minutes),
            **finalize_metric_bucket(bucket),
        }
    for label, step in report_steps.items():
        report[label] = dict(per_step[f"step_{step}"])
    return {**finalize_metric_bucket(overall), "per_step": per_step, "report": report}


def collect_predictions(
    model: STGATPredictor,
    loader: DataLoader,
    *,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    normalization_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    persistence_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            speed_history_mask = batch.get("speed_history_mask")
            if speed_history_mask is not None:
                speed_history_mask = speed_history_mask.to(device, non_blocking=non_blocking)
            speed_target = batch["speed_target"].to(device, non_blocking=non_blocking)
            speed_target_mask = batch.get("speed_target_mask")
            if speed_target_mask is None:
                target_mask_np = np.ones(tuple(speed_target.shape), dtype=bool)
            else:
                target_mask_np = speed_target_mask.numpy().astype(bool)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model.forward_v(
                    speed_seq,
                    extract_temporal_context(node_seq),
                    speed_history_mask_seq=speed_history_mask,
                )

            pred_raw = denormalize_speed_values(
                pred_norm.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            last_speed_raw = denormalize_speed_values(
                speed_seq[:, :, -1:].detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            persistence_raw = np.repeat(last_speed_raw, pred_raw.shape[-1], axis=-1)
            target_raw = denormalize_speed_values(
                speed_target.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            target_mask_np &= np.abs(target_raw) > 1e-6
            pred_parts.append(pred_raw)
            target_parts.append(target_raw)
            mask_parts.append(target_mask_np)
            persistence_parts.append(persistence_raw)

    return (
        np.concatenate(pred_parts),
        np.concatenate(target_parts),
        np.concatenate(mask_parts),
        np.concatenate(persistence_parts),
    )


def fit_identity(val_pred: np.ndarray, val_target: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones(val_pred.shape[1:], dtype=np.float32)
    bias = np.zeros(val_pred.shape[1:], dtype=np.float32)
    return scale, bias


def fit_bias(val_pred: np.ndarray, val_target: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones(val_pred.shape[1:], dtype=np.float32)
    residual = np.where(val_mask, val_target - val_pred, 0.0)
    counts = np.maximum(val_mask.sum(axis=0), 1)
    bias = (residual.sum(axis=0) / counts).astype(np.float32)
    return scale, bias


def fit_affine(val_pred: np.ndarray, val_target: np.ndarray, val_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    num_items, horizon = val_pred.shape[1], val_pred.shape[2]
    scale = np.ones((num_items, horizon), dtype=np.float32)
    bias = np.zeros((num_items, horizon), dtype=np.float32)
    for item_idx in range(num_items):
        for horizon_idx in range(horizon):
            mask = val_mask[:, item_idx, horizon_idx]
            if int(mask.sum()) < 20:
                continue
            x = val_pred[:, item_idx, horizon_idx][mask].astype(np.float64)
            y = val_target[:, item_idx, horizon_idx][mask].astype(np.float64)
            x_mean = float(x.mean())
            y_mean = float(y.mean())
            var = float(np.square(x - x_mean).mean())
            if var < 1e-6:
                bias[item_idx, horizon_idx] = np.float32(y_mean - x_mean)
                continue
            cov = float(((x - x_mean) * (y - y_mean)).mean())
            a = float(np.clip(cov / var, 0.65, 1.35))
            b = y_mean - a * x_mean
            scale[item_idx, horizon_idx] = np.float32(a)
            bias[item_idx, horizon_idx] = np.float32(b)
    return scale, bias


def apply_calibration(pred: np.ndarray, scale: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.maximum(pred * scale[None, :, :] + bias[None, :, :], 0.0).astype(np.float32)


def fit_blend(
    val_pred: np.ndarray,
    val_persistence: np.ndarray,
    val_target: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_items, horizon = val_pred.shape[1], val_pred.shape[2]
    coef_model = np.ones((num_items, horizon), dtype=np.float32)
    coef_persistence = np.zeros((num_items, horizon), dtype=np.float32)
    bias = np.zeros((num_items, horizon), dtype=np.float32)
    for item_idx in range(num_items):
        for horizon_idx in range(horizon):
            mask = val_mask[:, item_idx, horizon_idx]
            if int(mask.sum()) < 20:
                continue
            x_model = val_pred[:, item_idx, horizon_idx][mask].astype(np.float64)
            x_persist = val_persistence[:, item_idx, horizon_idx][mask].astype(np.float64)
            y = val_target[:, item_idx, horizon_idx][mask].astype(np.float64)
            design = np.stack([x_model, x_persist, np.ones_like(x_model)], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            coef_model[item_idx, horizon_idx] = np.float32(np.clip(coef[0], -0.5, 1.5))
            coef_persistence[item_idx, horizon_idx] = np.float32(np.clip(coef[1], -0.5, 1.5))
            bias[item_idx, horizon_idx] = np.float32(np.clip(coef[2], -25.0, 25.0))
    return coef_model, coef_persistence, bias


def apply_blend(
    pred: np.ndarray,
    persistence: np.ndarray,
    coef_model: np.ndarray,
    coef_persistence: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    calibrated = (
        pred * coef_model[None, :, :]
        + persistence * coef_persistence[None, :, :]
        + bias[None, :, :]
    )
    return np.maximum(calibrated, 0.0).astype(np.float32)


def build_calendar_average_feature(
    speed_values: np.ndarray,
    speed_valid_mask: np.ndarray,
    time_meta: pd.DataFrame,
    train_time_mask: np.ndarray,
) -> np.ndarray:
    dates = pd.to_datetime(time_meta["date"], errors="raise")
    if "day_of_week" in time_meta.columns:
        weekday = pd.to_numeric(time_meta["day_of_week"], errors="raise").to_numpy(dtype=np.int64)
    else:
        weekday = dates.dt.dayofweek.to_numpy(dtype=np.int64)
    slot = pd.to_numeric(time_meta["slot"], errors="raise").to_numpy(dtype=np.int64)
    slots_per_day = int(slot.max()) + 1
    key = weekday * slots_per_day + slot
    num_keys = int(key.max()) + 1
    num_items = int(speed_values.shape[1])

    train_valid = np.asarray(train_time_mask, dtype=bool)[:, None] & np.asarray(speed_valid_mask, dtype=bool)
    fallback_counts = np.maximum(train_valid.sum(axis=0), 1)
    fallback = (np.where(train_valid, speed_values, 0.0).sum(axis=0) / fallback_counts).astype(np.float32)
    key_means = np.broadcast_to(fallback[None, :], (num_keys, num_items)).copy()

    for key_idx in range(num_keys):
        rows = (key == key_idx) & np.asarray(train_time_mask, dtype=bool)
        if not np.any(rows):
            continue
        valid = speed_valid_mask[rows]
        counts = valid.sum(axis=0)
        has_values = counts > 0
        if not np.any(has_values):
            continue
        sums = np.where(valid, speed_values[rows], 0.0).sum(axis=0)
        key_means[key_idx, has_values] = (sums[has_values] / counts[has_values]).astype(np.float32)
    return key_means[key].astype(np.float32)


def historical_average_targets(
    calendar_average_by_time: np.ndarray,
    sample_indices: list[int],
    *,
    hist_len: int,
    pred_horizon: int,
) -> np.ndarray:
    output = np.empty((len(sample_indices), calendar_average_by_time.shape[1], pred_horizon), dtype=np.float32)
    for row_idx, sample_idx in enumerate(sample_indices):
        start = int(sample_idx) + int(hist_len)
        target_indices = np.arange(start, start + int(pred_horizon), dtype=np.int64)
        output[row_idx] = calendar_average_by_time[target_indices].T
    return output


def lagged_targets(
    speed_values: np.ndarray,
    speed_valid_mask: np.ndarray,
    sample_indices: list[int],
    *,
    hist_len: int,
    pred_horizon: int,
    lag_steps: int,
    fallback: np.ndarray,
) -> np.ndarray:
    output = fallback.copy()
    num_times = int(speed_values.shape[0])
    for row_idx, sample_idx in enumerate(sample_indices):
        start = int(sample_idx) + int(hist_len)
        target_indices = np.arange(start, start + int(pred_horizon), dtype=np.int64)
        source_indices = target_indices - int(lag_steps)
        for horizon_idx, source_idx in enumerate(source_indices):
            if source_idx < 0 or source_idx >= num_times:
                continue
            valid = speed_valid_mask[source_idx].astype(bool)
            if np.any(valid):
                output[row_idx, valid, horizon_idx] = speed_values[source_idx, valid]
    return output.astype(np.float32)


def rolling_history_mean_targets(
    speed_values: np.ndarray,
    speed_valid_mask: np.ndarray,
    sample_indices: list[int],
    *,
    hist_len: int,
    pred_horizon: int,
    window: int,
    fallback: np.ndarray,
) -> np.ndarray:
    output = fallback.copy()
    for row_idx, sample_idx in enumerate(sample_indices):
        end = int(sample_idx) + int(hist_len)
        start = max(int(sample_idx), end - int(window))
        history = speed_values[start:end]
        valid = speed_valid_mask[start:end].astype(bool)
        counts = valid.sum(axis=0)
        has_values = counts > 0
        if not np.any(has_values):
            continue
        means = np.zeros(speed_values.shape[1], dtype=np.float32)
        means[has_values] = (
            np.where(valid, history, 0.0).sum(axis=0)[has_values] / counts[has_values]
        ).astype(np.float32)
        output[row_idx, has_values, :] = means[has_values, None]
    return output.astype(np.float32)


def fit_multi_blend(
    feature_arrays: list[np.ndarray],
    val_target: np.ndarray,
    val_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_features = len(feature_arrays)
    num_items, horizon = val_target.shape[1], val_target.shape[2]
    coefs = np.zeros((num_features, num_items, horizon), dtype=np.float32)
    coefs[0, :, :] = 1.0
    bias = np.zeros((num_items, horizon), dtype=np.float32)
    for item_idx in range(num_items):
        for horizon_idx in range(horizon):
            mask = val_mask[:, item_idx, horizon_idx]
            if int(mask.sum()) < 20:
                continue
            columns = [
                values[:, item_idx, horizon_idx][mask].astype(np.float64)
                for values in feature_arrays
            ]
            y = val_target[:, item_idx, horizon_idx][mask].astype(np.float64)
            design = np.stack([*columns, np.ones_like(y)], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(design, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            coefs[:, item_idx, horizon_idx] = np.clip(coef[:-1], -1.0, 2.0).astype(np.float32)
            bias[item_idx, horizon_idx] = np.float32(np.clip(coef[-1], -30.0, 30.0))
    return coefs, bias


def apply_multi_blend(feature_arrays: list[np.ndarray], coefs: np.ndarray, bias: np.ndarray) -> np.ndarray:
    calibrated = bias[None, :, :].astype(np.float32)
    for feature_idx, values in enumerate(feature_arrays):
        calibrated = calibrated + values * coefs[feature_idx][None, :, :]
    return np.maximum(calibrated, 0.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit validation-only speed calibration for a METR-LA STGAT run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override stgat_meta.json dataset_dir when replaying an exported checkpoint on another machine.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--calibration-fit-split",
        choices=["val", "train_val"],
        default="val",
        help="Samples used to fit calibration weights. Test samples are never used.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    meta = load_json(run_dir / "stgat_meta.json")
    checkpoint = run_dir / "stgat_best.pt"
    checkpoint_state = resolve_checkpoint_state(checkpoint)
    inferred_config = infer_stgat_config(checkpoint_state)
    output_dir = args.output_dir or (run_dir / "calibration")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    dataset_dir = args.dataset_dir or Path(meta["dataset_dir"])
    dataset, time_feature_names, time_meta = load_prepared_sensor_dataset(
        dataset_dir,
        disable_time_features=len(meta.get("time_feature_names", [])) == 0,
    )
    if "date" in time_meta.columns:
        time_meta = time_meta.copy()
        time_meta["date"] = np.asarray(time_meta["date"], dtype="datetime64[ns]")
    speed_values = dataset["edge_speeds"]
    speed_valid_mask = dataset["speed_valid_mask"]
    node_features = dataset["node_features"]
    adjacency = dataset["adj"]
    adjacency_weights = dataset["adjacency_weights"]
    edge_index = dataset["edge_index"]
    edge_lengths = dataset["edge_lengths"]
    split_indices = build_monthly_split_indices(time_meta, int(meta["hist_len"]), int(meta["pred_horizon"]))
    train_history_mask = build_history_time_mask(
        int(speed_values.shape[0]),
        split_indices["train"],
        int(meta["hist_len"]),
    )
    train_window_mask = build_window_time_mask(
        int(speed_values.shape[0]),
        split_indices["train"],
        int(meta["hist_len"]),
        int(meta["pred_horizon"]),
    )
    speed_values, _, _ = resolve_outlier_cleaning(
        speed_values=speed_values,
        speed_valid_mask=speed_valid_mask,
        train_time_mask=train_history_mask,
        mode=str(meta.get("outlier_cleaning", {}).get("method", "train_quantile_clip")),
        lower_quantile=float(meta.get("outlier_cleaning", {}).get("params", {}).get("lower_quantile", 0.01)),
        upper_quantile=float(meta.get("outlier_cleaning", {}).get("params", {}).get("upper_quantile", 0.99)),
    )
    speed_values_raw = speed_values.copy()
    calendar_average_by_time = build_calendar_average_feature(
        speed_values_raw,
        speed_valid_mask,
        time_meta,
        train_window_mask,
    )
    normalization_stats = load_normalization_stats(meta["normalization"])
    node_features = normalize_node_features(node_features, normalization_stats)
    speed_values = normalize_speed_features(
        speed_values,
        normalization_stats,
        edge_axis=1,
        speed_valid_mask=speed_valid_mask,
    )

    use_history_mask = bool(meta.get("graph_topology", {}).get("history_missing_mode") == "causal_ffill_plus_mask")
    use_weighted_fixed_graph = bool(meta.get("graph_topology", {}).get("fixed_graph_weighted", False))
    full_dataset = SpatioTemporalDataset(
        node_features,
        speed_values,
        edge_speed_valid_mask=speed_valid_mask,
        edge_speed_history_valid_mask=(speed_valid_mask if use_history_mask else None),
        history_imputation_enabled=use_history_mask,
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
    )
    train_loader = DataLoader(
        Subset(full_dataset, split_indices["train"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(full_dataset, split_indices["val"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        Subset(full_dataset, split_indices["test"]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    graph_topology = meta.get("graph_topology", {})
    if not isinstance(graph_topology, dict):
        graph_topology = {}
    model = STGATPredictor(
        num_nodes=int(meta["num_nodes"]),
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adjacency),
        adj_weight_matrix=(torch.from_numpy(adjacency_weights) if use_weighted_fixed_graph else None),
        hidden_dim=int(meta.get("hidden_dim", inferred_config["hidden_dim"])),
        num_heads=int(meta.get("num_heads", inferred_config["num_heads"])),
        num_st_blocks=int(meta.get("num_st_blocks", inferred_config["num_st_blocks"])),
        num_gtcn_layers=int(meta.get("num_gtcn_layers", inferred_config["num_gtcn_layers"])),
        kernel_size=int(meta.get("kernel_size", inferred_config["kernel_size"])),
        pred_horizon=int(meta.get("pred_horizon", inferred_config["pred_horizon"])),
        node_feat_dim=int(node_features.shape[-1]),
        adaptive_topk=int(graph_topology.get("adaptive_topk", meta.get("adaptive_topk", 16))),
        speed_use_adaptive=bool(graph_topology.get("adaptive_enabled", True)),
        speed_use_fixed_graph=bool(graph_topology.get("fixed_graph_enabled", meta.get("speed_use_fixed_graph", True))),
        use_speed_history_mask=use_history_mask,
        use_fixed_edge_length_feature=bool(graph_topology.get("fixed_edge_length_feature_enabled", True)),
        v_domain=str(graph_topology.get("v_domain", meta.get("v_domain", "node"))),
    ).to(device)
    missing, unexpected = model.load_state_dict(checkpoint_state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:10]}")
    if missing:
        print(f"Loaded with missing keys: {missing[:10]}")

    val_pred, val_target, val_mask, val_persistence = collect_predictions(
        model,
        val_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
    )
    test_pred, test_target, test_mask, test_persistence = collect_predictions(
        model,
        test_loader,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
    )
    train_pred = train_target = train_mask = train_persistence = None
    if args.calibration_fit_split == "train_val":
        train_pred, train_target, train_mask, train_persistence = collect_predictions(
            model,
            train_loader,
            device=device,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            normalization_stats=normalization_stats,
        )
    val_calendar_average = historical_average_targets(
        calendar_average_by_time,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
    )
    test_calendar_average = historical_average_targets(
        calendar_average_by_time,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
    )
    slots_per_day = int(pd.to_numeric(time_meta["slot"], errors="raise").max()) + 1
    val_lag_day = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=slots_per_day,
        fallback=val_calendar_average,
    )
    test_lag_day = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=slots_per_day,
        fallback=test_calendar_average,
    )
    val_lag_week = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=7 * slots_per_day,
        fallback=val_calendar_average,
    )
    test_lag_week = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=7 * slots_per_day,
        fallback=test_calendar_average,
    )
    val_history_mean_3 = rolling_history_mean_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        window=3,
        fallback=val_persistence,
    )
    test_history_mean_3 = rolling_history_mean_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        window=3,
        fallback=test_persistence,
    )
    val_history_mean_12 = rolling_history_mean_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        window=12,
        fallback=val_persistence,
    )
    test_history_mean_12 = rolling_history_mean_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        window=12,
        fallback=test_persistence,
    )
    train_calendar_average = train_lag_day = train_lag_week = None
    train_history_mean_3 = train_history_mean_12 = None
    if args.calibration_fit_split == "train_val":
        assert train_persistence is not None
        train_calendar_average = historical_average_targets(
            calendar_average_by_time,
            split_indices["train"],
            hist_len=int(meta["hist_len"]),
            pred_horizon=int(meta["pred_horizon"]),
        )
        train_lag_day = lagged_targets(
            speed_values_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=int(meta["hist_len"]),
            pred_horizon=int(meta["pred_horizon"]),
            lag_steps=slots_per_day,
            fallback=train_calendar_average,
        )
        train_lag_week = lagged_targets(
            speed_values_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=int(meta["hist_len"]),
            pred_horizon=int(meta["pred_horizon"]),
            lag_steps=7 * slots_per_day,
            fallback=train_calendar_average,
        )
        train_history_mean_3 = rolling_history_mean_targets(
            speed_values_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=int(meta["hist_len"]),
            pred_horizon=int(meta["pred_horizon"]),
            window=3,
            fallback=train_persistence,
        )
        train_history_mean_12 = rolling_history_mean_targets(
            speed_values_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=int(meta["hist_len"]),
            pred_horizon=int(meta["pred_horizon"]),
            window=12,
            fallback=train_persistence,
        )

    fit_pred = val_pred
    fit_target = val_target
    fit_mask = val_mask
    fit_persistence = val_persistence
    fit_calendar_average = val_calendar_average
    fit_lag_day = val_lag_day
    fit_lag_week = val_lag_week
    fit_history_mean_3 = val_history_mean_3
    fit_history_mean_12 = val_history_mean_12
    if args.calibration_fit_split == "train_val":
        assert train_pred is not None
        assert train_target is not None
        assert train_mask is not None
        assert train_persistence is not None
        assert train_calendar_average is not None
        assert train_lag_day is not None
        assert train_lag_week is not None
        assert train_history_mean_3 is not None
        assert train_history_mean_12 is not None
        fit_pred = np.concatenate([train_pred, val_pred], axis=0)
        fit_target = np.concatenate([train_target, val_target], axis=0)
        fit_mask = np.concatenate([train_mask, val_mask], axis=0)
        fit_persistence = np.concatenate([train_persistence, val_persistence], axis=0)
        fit_calendar_average = np.concatenate([train_calendar_average, val_calendar_average], axis=0)
        fit_lag_day = np.concatenate([train_lag_day, val_lag_day], axis=0)
        fit_lag_week = np.concatenate([train_lag_week, val_lag_week], axis=0)
        fit_history_mean_3 = np.concatenate([train_history_mean_3, val_history_mean_3], axis=0)
        fit_history_mean_12 = np.concatenate([train_history_mean_12, val_history_mean_12], axis=0)

    report_steps = {"15min": 3, "30min": 6, "60min": 12}
    variants = {
        "identity": fit_identity,
        "bias": fit_bias,
        "affine": fit_affine,
    }
    summaries: dict[str, Any] = {}
    best_name = ""
    best_val_rmse = float("inf")
    for name, fitter in variants.items():
        scale, bias = fitter(fit_pred, fit_target, fit_mask)
        val_cal = apply_calibration(val_pred, scale, bias)
        test_cal = apply_calibration(test_pred, scale, bias)
        val_metrics = compute_speed_metrics(val_cal, val_target, val_mask, report_steps=report_steps)
        test_metrics = compute_speed_metrics(test_cal, test_target, test_mask, report_steps=report_steps)
        summaries[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "scale_mean": float(scale.mean()),
            "scale_std": float(scale.std()),
            "bias_mean": float(bias.mean()),
            "bias_std": float(bias.std()),
        }
        if float(val_metrics["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_name = name
        np.save(output_dir / f"{name}_scale.npy", scale)
        np.save(output_dir / f"{name}_bias.npy", bias)

    coef_model, coef_persistence, blend_bias = fit_blend(
        fit_pred,
        fit_persistence,
        fit_target,
        fit_mask,
    )
    val_blend = apply_blend(val_pred, val_persistence, coef_model, coef_persistence, blend_bias)
    test_blend = apply_blend(test_pred, test_persistence, coef_model, coef_persistence, blend_bias)
    val_blend_metrics = compute_speed_metrics(val_blend, val_target, val_mask, report_steps=report_steps)
    test_blend_metrics = compute_speed_metrics(test_blend, test_target, test_mask, report_steps=report_steps)
    summaries["blend_persistence"] = {
        "val": val_blend_metrics,
        "test": test_blend_metrics,
        "coef_model_mean": float(coef_model.mean()),
        "coef_model_std": float(coef_model.std()),
        "coef_persistence_mean": float(coef_persistence.mean()),
        "coef_persistence_std": float(coef_persistence.std()),
        "bias_mean": float(blend_bias.mean()),
        "bias_std": float(blend_bias.std()),
    }
    if float(val_blend_metrics["rmse"]) < best_val_rmse:
        best_val_rmse = float(val_blend_metrics["rmse"])
        best_name = "blend_persistence"
    np.save(output_dir / "blend_persistence_coef_model.npy", coef_model)
    np.save(output_dir / "blend_persistence_coef_persistence.npy", coef_persistence)
    np.save(output_dir / "blend_persistence_bias.npy", blend_bias)

    multi_coefs, multi_bias = fit_multi_blend(
        [fit_pred, fit_persistence, fit_calendar_average],
        fit_target,
        fit_mask,
    )
    val_multi_blend = apply_multi_blend(
        [val_pred, val_persistence, val_calendar_average],
        multi_coefs,
        multi_bias,
    )
    test_multi_blend = apply_multi_blend(
        [test_pred, test_persistence, test_calendar_average],
        multi_coefs,
        multi_bias,
    )
    val_multi_metrics = compute_speed_metrics(val_multi_blend, val_target, val_mask, report_steps=report_steps)
    test_multi_metrics = compute_speed_metrics(test_multi_blend, test_target, test_mask, report_steps=report_steps)
    summaries["blend_persistence_calendar"] = {
        "val": val_multi_metrics,
        "test": test_multi_metrics,
        "coef_model_mean": float(multi_coefs[0].mean()),
        "coef_model_std": float(multi_coefs[0].std()),
        "coef_persistence_mean": float(multi_coefs[1].mean()),
        "coef_persistence_std": float(multi_coefs[1].std()),
        "coef_calendar_mean": float(multi_coefs[2].mean()),
        "coef_calendar_std": float(multi_coefs[2].std()),
        "bias_mean": float(multi_bias.mean()),
        "bias_std": float(multi_bias.std()),
    }
    if float(val_multi_metrics["rmse"]) < best_val_rmse:
        best_val_rmse = float(val_multi_metrics["rmse"])
        best_name = "blend_persistence_calendar"
    np.save(output_dir / "blend_persistence_calendar_coefs.npy", multi_coefs)
    np.save(output_dir / "blend_persistence_calendar_bias.npy", multi_bias)

    feature_sets = {
        "blend_calendar_daylag": (
            ["model", "persistence", "calendar", "lag_day"],
            [fit_pred, fit_persistence, fit_calendar_average, fit_lag_day],
            [val_pred, val_persistence, val_calendar_average, val_lag_day],
            [test_pred, test_persistence, test_calendar_average, test_lag_day],
        ),
        "blend_calendar_day_week_lags": (
            ["model", "persistence", "calendar", "lag_day", "lag_week"],
            [fit_pred, fit_persistence, fit_calendar_average, fit_lag_day, fit_lag_week],
            [val_pred, val_persistence, val_calendar_average, val_lag_day, val_lag_week],
            [test_pred, test_persistence, test_calendar_average, test_lag_day, test_lag_week],
        ),
        "blend_calendar_lags_history": (
            ["model", "persistence", "calendar", "lag_day", "lag_week", "history_mean_3", "history_mean_12"],
            [
                fit_pred,
                fit_persistence,
                fit_calendar_average,
                fit_lag_day,
                fit_lag_week,
                fit_history_mean_3,
                fit_history_mean_12,
            ],
            [
                val_pred,
                val_persistence,
                val_calendar_average,
                val_lag_day,
                val_lag_week,
                val_history_mean_3,
                val_history_mean_12,
            ],
            [
                test_pred,
                test_persistence,
                test_calendar_average,
                test_lag_day,
                test_lag_week,
                test_history_mean_3,
                test_history_mean_12,
            ],
        ),
    }
    for name, (feature_names, fit_features, val_features, test_features) in feature_sets.items():
        coefs, bias = fit_multi_blend(fit_features, fit_target, fit_mask)
        val_cal = apply_multi_blend(val_features, coefs, bias)
        test_cal = apply_multi_blend(test_features, coefs, bias)
        val_metrics = compute_speed_metrics(val_cal, val_target, val_mask, report_steps=report_steps)
        test_metrics = compute_speed_metrics(test_cal, test_target, test_mask, report_steps=report_steps)
        summaries[name] = {
            "val": val_metrics,
            "test": test_metrics,
            "features": feature_names,
            "coef_means": {feature: float(coefs[idx].mean()) for idx, feature in enumerate(feature_names)},
            "coef_stds": {feature: float(coefs[idx].std()) for idx, feature in enumerate(feature_names)},
            "bias_mean": float(bias.mean()),
            "bias_std": float(bias.std()),
        }
        if float(val_metrics["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_name = name
        np.save(output_dir / f"{name}_coefs.npy", coefs)
        np.save(output_dir / f"{name}_bias.npy", bias)

    best = summaries[best_name]
    metrics_payload = {
        "calibration": {
            "selected_variant": best_name,
            "selection_metric": "val_raw_speed_rmse",
            "val_raw_speed_rmse": best["val"]["rmse"],
            "fit_split": args.calibration_fit_split,
            "source_run_dir": str(run_dir),
        },
        "raw_metrics": {"speed": {k: v for k, v in best["test"].items() if k not in {"per_step", "report"}}},
        "raw_metrics_per_step": {"speed": best["test"]["per_step"]},
        "raw_metrics_report": {"speed": best["test"]["report"]},
        "val_raw_metrics": {"speed": {k: v for k, v in best["val"].items() if k not in {"per_step", "report"}}},
        "val_raw_metrics_per_step": {"speed": best["val"]["per_step"]},
        "val_raw_metrics_report": {"speed": best["val"]["report"]},
    }
    with (output_dir / "calibrated_predictor_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected": best_name, "variants": summaries}, handle, ensure_ascii=False, indent=2)

    print(f"Selected calibration: {best_name} (val RMSE={best['val']['rmse']:.4f})")
    for name, summary in summaries.items():
        report = summary["test"]["report"]
        print(
            f"{name}: val_rmse={summary['val']['rmse']:.4f} "
            f"test_rmse={summary['test']['rmse']:.4f} "
            f"15/30/60={report['15min']['rmse']:.4f}/"
            f"{report['30min']['rmse']:.4f}/{report['60min']['rmse']:.4f}"
        )
    print(output_dir / "calibrated_predictor_test_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
