from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import minimize
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
from paper_speed_benchmarks.nyc_speed_prediction import filter_split_indices_by_target_mask  # noqa: E402
from predictor_normalization import (  # noqa: E402
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor  # noqa: E402
from tools.calibrate_metrla_speed_predictions import (  # noqa: E402
    build_calendar_average_feature,
    collect_predictions,
    compute_speed_metrics,
    historical_average_targets,
    infer_stgat_config,
    lagged_targets,
    load_json,
    rolling_history_mean_targets,
    resolve_checkpoint_state,
)
from tools.ensemble_paper_speed_predictions import (  # noqa: E402
    apply_convex_weights,
    apply_stepwise_convex_weights,
    convex_weight_grid,
    fit_stepwise_convex_weights,
    mse_grid_scores,
)
from train_predictor import (  # noqa: E402
    build_monthly_split_indices,
    build_window_time_mask,
    configure_cuda_runtime,
    resolve_device,
    resolve_num_workers,
    resolve_precision,
)


REPORT_STEPS = {"15min": 3, "30min": 6, "60min": 12}


def load_run_predictions(
    run_dir: Path,
    *,
    dataset_dir: Path | None,
    batch_size: int,
    device: torch.device,
    precision: str,
    num_workers_arg: int,
    filter_full_target_windows: bool,
    include_shared_features: bool,
) -> dict[str, Any]:
    meta = load_json(run_dir / "stgat_meta.json")
    checkpoint = run_dir / "stgat_best.pt"
    checkpoint_state = resolve_checkpoint_state(checkpoint)
    inferred_config = infer_stgat_config(checkpoint_state)

    configure_cuda_runtime(device)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(num_workers_arg, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    resolved_dataset_dir = dataset_dir or Path(meta["dataset_dir"])
    dataset, _time_feature_names, time_meta = load_prepared_sensor_dataset(
        resolved_dataset_dir,
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
    if filter_full_target_windows:
        split_indices = filter_split_indices_by_target_mask(
            split_indices,
            speed_valid_mask,
            int(meta["hist_len"]),
            int(meta["pred_horizon"]),
        )
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
    normalization_stats = load_normalization_stats(meta["normalization"])
    node_features = normalize_node_features(node_features, normalization_stats)
    speed_values = normalize_speed_features(
        speed_values,
        normalization_stats,
        edge_axis=1,
        speed_valid_mask=speed_valid_mask,
    )

    graph_topology = meta.get("graph_topology", {})
    if not isinstance(graph_topology, dict):
        graph_topology = {}
    use_history_mask = bool(graph_topology.get("history_missing_mode") == "causal_ffill_plus_mask")
    use_weighted_fixed_graph = bool(graph_topology.get("fixed_graph_weighted", False))
    full_dataset = SpatioTemporalDataset(
        node_features,
        speed_values,
        edge_speed_valid_mask=speed_valid_mask,
        edge_speed_history_valid_mask=(speed_valid_mask if use_history_mask else None),
        history_imputation_enabled=use_history_mask,
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
    )
    loaders = {
        split: DataLoader(
            Subset(full_dataset, split_indices[split]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split in ("val", "test")
    }
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
    model.load_state_dict(checkpoint_state, strict=False)

    val_pred, val_target, val_mask, val_persistence = collect_predictions(
        model,
        loaders["val"],
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
    )
    test_pred, test_target, test_mask, test_persistence = collect_predictions(
        model,
        loaders["test"],
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
    )
    result: dict[str, Any] = {
        "run_dir": str(run_dir),
        "val_pred": val_pred,
        "test_pred": test_pred,
    }
    if not include_shared_features:
        return result

    calendar_average = build_calendar_average_feature(
        speed_values_raw,
        speed_valid_mask,
        time_meta,
        train_window_mask,
    )
    val_calendar = historical_average_targets(
        calendar_average,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
    )
    test_calendar = historical_average_targets(
        calendar_average,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
    )
    slots_per_day = int(np.asarray(time_meta["slot"], dtype=np.int64).max()) + 1
    val_lag_day = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=slots_per_day,
        fallback=val_calendar,
    )
    test_lag_day = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=slots_per_day,
        fallback=test_calendar,
    )
    val_lag_week = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=7 * slots_per_day,
        fallback=val_calendar,
    )
    test_lag_week = lagged_targets(
        speed_values_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=int(meta["hist_len"]),
        pred_horizon=int(meta["pred_horizon"]),
        lag_steps=7 * slots_per_day,
        fallback=test_calendar,
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
    result.update(
        {
        "run_dir": str(run_dir),
        "val_target": val_target,
        "val_mask": val_mask,
        "val_persistence": val_persistence,
        "val_calendar": val_calendar,
        "val_lag_day": val_lag_day,
        "val_lag_week": val_lag_week,
        "val_history_mean_3": val_history_mean_3,
        "val_history_mean_12": val_history_mean_12,
        "test_target": test_target,
        "test_mask": test_mask,
        "test_persistence": test_persistence,
        "test_calendar": test_calendar,
        "test_lag_day": test_lag_day,
        "test_lag_week": test_lag_week,
        "test_history_mean_3": test_history_mean_3,
        "test_history_mean_12": test_history_mean_12,
        }
    )
    return result


SHARED_CACHE_KEYS = [
    "val_target",
    "val_mask",
    "val_persistence",
    "val_calendar",
    "val_lag_day",
    "val_lag_week",
    "val_history_mean_3",
    "val_history_mean_12",
    "test_target",
    "test_mask",
    "test_persistence",
    "test_calendar",
    "test_lag_day",
    "test_lag_week",
    "test_history_mean_3",
    "test_history_mean_12",
]


def progress_log(output_dir: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": event,
    }
    if payload:
        record.update(payload)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def cache_signature(
    run_dir: Path,
    *,
    dataset_dir: Path | None,
    filter_full_target_windows: bool,
    precision: str,
    include_shared_features: bool,
) -> dict[str, Any]:
    checkpoint = run_dir / "stgat_best.pt"
    stat = checkpoint.stat()
    resolved_dataset_dir = dataset_dir.resolve() if dataset_dir is not None else None
    return {
        "cache_schema": 1,
        "run_dir": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "dataset_dir": str(resolved_dataset_dir) if resolved_dataset_dir is not None else None,
        "filter_full_target_windows": bool(filter_full_target_windows),
        "precision": precision,
        "include_shared_features": bool(include_shared_features),
    }


def load_cache_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        return load_json(meta_path)
    except (OSError, json.JSONDecodeError):
        return None


def cache_is_valid(npz_path: Path, meta_path: Path, signature: dict[str, Any]) -> bool:
    meta = load_cache_meta(meta_path)
    return bool(npz_path.exists() and meta == signature)


def load_npz_payload(npz_path: Path, keys: list[str]) -> dict[str, Any]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in keys}


def save_npz_payload(npz_path: Path, arrays: dict[str, Any], keys: list[str]) -> None:
    np.savez_compressed(npz_path, **{key: arrays[key] for key in keys})


def checkpoint_cache_paths(cache_dir: Path, index: int, run_dir: Path) -> tuple[Path, Path]:
    stem = f"checkpoint_{index + 1:02d}_{run_dir.name}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.json"


def write_checkpoint_metrics(
    output_dir: Path,
    *,
    index: int,
    run_dir: Path,
    item: dict[str, Any],
    reference: dict[str, Any],
    cache_status: str,
    elapsed_seconds: float,
) -> None:
    summary = {
        "run_dir": str(run_dir),
        "cache_status": cache_status,
        "elapsed_seconds": float(elapsed_seconds),
        "val": compute_speed_metrics(item["val_pred"], reference["val_target"], reference["val_mask"], report_steps=REPORT_STEPS),
        "test": compute_speed_metrics(item["test_pred"], reference["test_target"], reference["test_mask"], report_steps=REPORT_STEPS),
    }
    metrics_path = output_dir / f"checkpoint_{index + 1:02d}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    report = summary["test"]["report"]
    progress_log(
        output_dir,
        "checkpoint_metrics_written",
        {
            "index": index + 1,
            "name": run_dir.name,
            "cache_status": cache_status,
            "elapsed_seconds": round(float(elapsed_seconds), 2),
            "val_rmse": round(float(summary["val"]["rmse"]), 6),
            "test_rmse": round(float(summary["test"]["rmse"]), 6),
            "test_15_30_60_rmse": [
                round(float(report["15min"]["rmse"]), 6),
                round(float(report["30min"]["rmse"]), 6),
                round(float(report["60min"]["rmse"]), 6),
            ],
        },
    )


def add_summary(
    summaries: dict[str, Any],
    name: str,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    val_target: np.ndarray,
    val_mask: np.ndarray,
    test_target: np.ndarray,
    test_mask: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "val": compute_speed_metrics(val_pred, val_target, val_mask, report_steps=REPORT_STEPS),
        "test": compute_speed_metrics(test_pred, test_target, test_mask, report_steps=REPORT_STEPS),
    }
    if extra:
        payload.update(extra)
    summaries[name] = payload


def convex_quadratic_terms(
    feature_arrays: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    num_features = len(feature_arrays)
    gram = np.zeros((num_features, num_features), dtype=np.float64)
    rhs = np.zeros(num_features, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    target_valid = target[valid].astype(np.float64)
    scale = max(int(target_valid.size), 1)
    flattened = [values[valid].astype(np.float64) for values in feature_arrays]
    for i in range(num_features):
        rhs[i] = float(np.dot(flattened[i], target_valid)) / scale
        for j in range(i, num_features):
            value = float(np.dot(flattened[i], flattened[j])) / scale
            gram[i, j] = value
            gram[j, i] = value
    return gram, rhs


def fit_convex_weights_exact(
    feature_arrays: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    gram, rhs = convex_quadratic_terms(feature_arrays, target, mask)
    num_features = len(feature_arrays)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ gram @ weights - 2.0 * (weights @ rhs))

    def gradient(weights: np.ndarray) -> np.ndarray:
        return (2.0 * gram @ weights - 2.0 * rhs).astype(np.float64)

    bounds = [(0.0, 1.0)] * num_features
    constraints = {
        "type": "eq",
        "fun": lambda weights: float(np.sum(weights) - 1.0),
        "jac": lambda weights: np.ones_like(weights, dtype=np.float64),
    }
    one_hot_best = int(np.argmin(np.diag(gram) - 2.0 * rhs))
    candidates = [
        np.full(num_features, 1.0 / num_features, dtype=np.float64),
        np.eye(num_features, dtype=np.float64)[one_hot_best],
    ]
    result = minimize(
        objective,
        candidates[0],
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        weights = np.clip(result.x.astype(np.float64), 0.0, 1.0)
        total = float(weights.sum())
        if total > 0:
            candidates.append(weights / total)

    best = min(candidates, key=objective)
    return best.astype(np.float32)


def fit_stepwise_convex_weights_exact(
    feature_arrays: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    pred_horizon = int(target.shape[-1])
    weights_by_step = np.zeros((pred_horizon, len(feature_arrays)), dtype=np.float32)
    for step_idx in range(pred_horizon):
        weights_by_step[step_idx] = fit_convex_weights_exact(
            [values[..., step_idx] for values in feature_arrays],
            target[..., step_idx],
            mask[..., step_idx],
        )
    return weights_by_step


def fit_item_step_convex_weights_exact(
    feature_arrays: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    num_items = int(target.shape[1])
    pred_horizon = int(target.shape[2])
    num_features = len(feature_arrays)
    weights = np.zeros((num_items, pred_horizon, num_features), dtype=np.float32)
    for item_idx in range(num_items):
        for step_idx in range(pred_horizon):
            item_mask = mask[:, item_idx, step_idx]
            if int(np.asarray(item_mask, dtype=bool).sum()) < 20:
                weights[item_idx, step_idx, 0] = 1.0
                continue
            weights[item_idx, step_idx] = fit_convex_weights_exact(
                [values[:, item_idx, step_idx] for values in feature_arrays],
                target[:, item_idx, step_idx],
                item_mask,
            )
    return weights


def apply_item_step_convex_weights(
    feature_arrays: list[np.ndarray],
    weights_by_item_step: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(feature_arrays[0], dtype=np.float32)
    for feature_idx, values in enumerate(feature_arrays):
        output += values.astype(np.float32) * weights_by_item_step[None, :, :, feature_idx].astype(np.float32)
    return np.maximum(output, 0.0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="STGAT-only validation-fit ensemble for METR-LA speed runs.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override stgat_meta.json dataset_dir when replaying exported checkpoints on another machine.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--include-causal-features", action="store_true")
    parser.add_argument("--include-advanced-causal-features", action="store_true")
    parser.add_argument("--filter-full-target-windows", action="store_true")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--skip-grid", action="store_true", help="Skip the coarse convex grid after exact convex fitting.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    precision = resolve_precision(device, args.precision)
    progress_log(
        output_dir,
        "ensemble_started",
        {
            "run_count": len(args.run_dirs),
            "output_dir": str(output_dir),
            "cache_dir": str(cache_dir),
            "dataset_dir": str(args.dataset_dir) if args.dataset_dir is not None else None,
            "device": str(device),
            "precision": precision,
            "batch_size": int(args.batch_size),
            "include_causal_features": bool(args.include_causal_features),
            "include_advanced_causal_features": bool(args.include_advanced_causal_features),
            "grid_step": float(args.grid_step),
            "skip_grid": bool(args.skip_grid),
        },
    )

    shared_npz = cache_dir / "shared_features.npz"
    shared_meta = cache_dir / "shared_features.json"
    shared_signature = cache_signature(
        args.run_dirs[0],
        dataset_dir=args.dataset_dir,
        filter_full_target_windows=bool(args.filter_full_target_windows),
        precision=precision,
        include_shared_features=True,
    )
    shared_payload: dict[str, Any] | None = None
    if cache_is_valid(shared_npz, shared_meta, shared_signature):
        shared_payload = load_npz_payload(shared_npz, SHARED_CACHE_KEYS)
        progress_log(output_dir, "shared_cache_loaded", {"path": str(shared_npz)})

    loaded: list[dict[str, Any]] = []
    for index, run_dir in enumerate(args.run_dirs):
        run_dir = Path(run_dir)
        include_shared_features = index == 0 and shared_payload is None
        pred_npz, pred_meta = checkpoint_cache_paths(cache_dir, index, run_dir)
        pred_signature = cache_signature(
            run_dir,
            dataset_dir=args.dataset_dir,
            filter_full_target_windows=bool(args.filter_full_target_windows),
            precision=precision,
            include_shared_features=False,
        )
        cache_status = "miss"
        start = time.perf_counter()
        if cache_is_valid(pred_npz, pred_meta, pred_signature) and not include_shared_features:
            item = {"run_dir": str(run_dir)}
            item.update(load_npz_payload(pred_npz, ["val_pred", "test_pred"]))
            cache_status = "hit"
            progress_log(output_dir, "checkpoint_cache_loaded", {"index": index + 1, "name": run_dir.name, "path": str(pred_npz)})
        else:
            progress_log(output_dir, "checkpoint_started", {"index": index + 1, "name": run_dir.name})
            full_item = load_run_predictions(
                run_dir,
                dataset_dir=args.dataset_dir,
                batch_size=int(args.batch_size),
                device=device,
                precision=precision,
                num_workers_arg=int(args.num_workers),
                filter_full_target_windows=bool(args.filter_full_target_windows),
                include_shared_features=include_shared_features,
            )
            item = {
                "run_dir": full_item["run_dir"],
                "val_pred": full_item["val_pred"],
                "test_pred": full_item["test_pred"],
            }
            save_npz_payload(pred_npz, item, ["val_pred", "test_pred"])
            with pred_meta.open("w", encoding="utf-8") as handle:
                json.dump(pred_signature, handle, ensure_ascii=False, indent=2)
            progress_log(output_dir, "checkpoint_cache_written", {"index": index + 1, "name": run_dir.name, "path": str(pred_npz)})
            if include_shared_features:
                shared_payload = {key: full_item[key] for key in SHARED_CACHE_KEYS}
                save_npz_payload(shared_npz, shared_payload, SHARED_CACHE_KEYS)
                with shared_meta.open("w", encoding="utf-8") as handle:
                    json.dump(shared_signature, handle, ensure_ascii=False, indent=2)
                progress_log(output_dir, "shared_cache_written", {"path": str(shared_npz)})
            del full_item

        if index == 0:
            if shared_payload is None:
                raise RuntimeError("Shared target and causal feature cache was not created for the first checkpoint.")
            item.update(shared_payload)
        loaded.append(item)
        write_checkpoint_metrics(
            output_dir,
            index=index,
            run_dir=run_dir,
            item=item,
            reference=loaded[0],
            cache_status=cache_status,
            elapsed_seconds=time.perf_counter() - start,
        )

    val_target = loaded[0]["val_target"]
    val_mask = loaded[0]["val_mask"]
    test_target = loaded[0]["test_target"]
    test_mask = loaded[0]["test_mask"]
    val_features = [item["val_pred"] for item in loaded]
    test_features = [item["test_pred"] for item in loaded]
    feature_names = [Path(item["run_dir"]).name for item in loaded]
    if args.include_causal_features:
        val_features.extend([loaded[0]["val_persistence"], loaded[0]["val_calendar"]])
        test_features.extend([loaded[0]["test_persistence"], loaded[0]["test_calendar"]])
        feature_names.extend(["persistence", "calendar_average"])
    if args.include_advanced_causal_features:
        val_features.extend(
            [
                loaded[0]["val_lag_day"],
                loaded[0]["val_lag_week"],
                loaded[0]["val_history_mean_3"],
                loaded[0]["val_history_mean_12"],
            ]
        )
        test_features.extend(
            [
                loaded[0]["test_lag_day"],
                loaded[0]["test_lag_week"],
                loaded[0]["test_history_mean_3"],
                loaded[0]["test_history_mean_12"],
            ]
        )
        feature_names.extend(["lag_day", "lag_week", "history_mean_3", "history_mean_12"])

    summaries: dict[str, Any] = {}
    best_name = ""
    best_val_rmse = float("inf")
    for idx, name in enumerate(feature_names):
        add_summary(
            summaries,
            f"component_{idx + 1}_{name}",
            val_features[idx],
            test_features[idx],
            val_target,
            val_mask,
            test_target,
            test_mask,
            {"feature": name},
        )
        val_rmse = float(summaries[f"component_{idx + 1}_{name}"]["val"]["rmse"])
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_name = f"component_{idx + 1}_{name}"

    progress_log(output_dir, "convex_exact_started", {"feature_count": len(feature_names)})
    exact_global_weights = fit_convex_weights_exact(val_features, val_target, val_mask)
    val_exact_global = apply_convex_weights(val_features, exact_global_weights)
    test_exact_global = apply_convex_weights(test_features, exact_global_weights)
    add_summary(
        summaries,
        "convex_exact_global_mse",
        val_exact_global,
        test_exact_global,
        val_target,
        val_mask,
        test_target,
        test_mask,
        {"features": feature_names, "weights": [float(value) for value in exact_global_weights]},
    )
    if float(summaries["convex_exact_global_mse"]["val"]["rmse"]) < best_val_rmse:
        best_val_rmse = float(summaries["convex_exact_global_mse"]["val"]["rmse"])
        best_name = "convex_exact_global_mse"

    exact_step_weights = fit_stepwise_convex_weights_exact(val_features, val_target, val_mask)
    val_exact_step = apply_stepwise_convex_weights(val_features, exact_step_weights)
    test_exact_step = apply_stepwise_convex_weights(test_features, exact_step_weights)
    add_summary(
        summaries,
        "convex_exact_per_step_mse",
        val_exact_step,
        test_exact_step,
        val_target,
        val_mask,
        test_target,
        test_mask,
        {"features": feature_names, "weights_by_step": exact_step_weights.astype(float).tolist()},
    )
    if float(summaries["convex_exact_per_step_mse"]["val"]["rmse"]) < best_val_rmse:
        best_val_rmse = float(summaries["convex_exact_per_step_mse"]["val"]["rmse"])
        best_name = "convex_exact_per_step_mse"
    np.save(output_dir / "convex_exact_per_step_mse_weights.npy", exact_step_weights)

    exact_item_step_weights = fit_item_step_convex_weights_exact(val_features, val_target, val_mask)
    val_exact_item_step = apply_item_step_convex_weights(val_features, exact_item_step_weights)
    test_exact_item_step = apply_item_step_convex_weights(test_features, exact_item_step_weights)
    add_summary(
        summaries,
        "convex_exact_per_item_step_mse",
        val_exact_item_step,
        test_exact_item_step,
        val_target,
        val_mask,
        test_target,
        test_mask,
        {"features": feature_names, "weights_shape": list(exact_item_step_weights.shape)},
    )
    if float(summaries["convex_exact_per_item_step_mse"]["val"]["rmse"]) < best_val_rmse:
        best_val_rmse = float(summaries["convex_exact_per_item_step_mse"]["val"]["rmse"])
        best_name = "convex_exact_per_item_step_mse"
    np.save(output_dir / "convex_exact_per_item_step_mse_weights.npy", exact_item_step_weights)
    progress_log(
        output_dir,
        "convex_exact_completed",
        {
            "global_val_rmse": round(float(summaries["convex_exact_global_mse"]["val"]["rmse"]), 6),
            "per_step_val_rmse": round(float(summaries["convex_exact_per_step_mse"]["val"]["rmse"]), 6),
            "per_item_step_val_rmse": round(float(summaries["convex_exact_per_item_step_mse"]["val"]["rmse"]), 6),
            "selected_so_far": best_name,
        },
    )

    if not args.skip_grid:
        progress_log(
            output_dir,
            "convex_grid_started",
            {"feature_count": len(feature_names), "grid_step": float(args.grid_step)},
        )
        grid = convex_weight_grid(len(val_features), step=float(args.grid_step))
        progress_log(output_dir, "convex_grid_built", {"weight_count": len(grid)})
        global_weights = grid[int(np.argmin(mse_grid_scores(val_features, val_target, val_mask, grid)))]
        val_global = apply_convex_weights(val_features, global_weights)
        test_global = apply_convex_weights(test_features, global_weights)
        add_summary(
            summaries,
            "convex_grid_global_mse",
            val_global,
            test_global,
            val_target,
            val_mask,
            test_target,
            test_mask,
            {"features": feature_names, "weights": [float(value) for value in global_weights]},
        )
        if float(summaries["convex_grid_global_mse"]["val"]["rmse"]) < best_val_rmse:
            best_val_rmse = float(summaries["convex_grid_global_mse"]["val"]["rmse"])
            best_name = "convex_grid_global_mse"

        step_weights = fit_stepwise_convex_weights(val_features, val_target, val_mask, grid, mode="mse")
        val_step = apply_stepwise_convex_weights(val_features, step_weights)
        test_step = apply_stepwise_convex_weights(test_features, step_weights)
        add_summary(
            summaries,
            "convex_grid_per_step_mse",
            val_step,
            test_step,
            val_target,
            val_mask,
            test_target,
            test_mask,
            {"features": feature_names, "weights_by_step": step_weights.astype(float).tolist()},
        )
        if float(summaries["convex_grid_per_step_mse"]["val"]["rmse"]) < best_val_rmse:
            best_val_rmse = float(summaries["convex_grid_per_step_mse"]["val"]["rmse"])
            best_name = "convex_grid_per_step_mse"
        np.save(output_dir / "convex_grid_per_step_mse_weights.npy", step_weights)
        progress_log(
            output_dir,
            "convex_grid_completed",
            {
                "global_val_rmse": round(float(summaries["convex_grid_global_mse"]["val"]["rmse"]), 6),
                "per_step_val_rmse": round(float(summaries["convex_grid_per_step_mse"]["val"]["rmse"]), 6),
                "selected_so_far": best_name,
            },
        )

    best = summaries[best_name]
    metrics_payload = {
        "ensemble": {
            "selected_variant": best_name,
            "selection_metric": "val_raw_speed_rmse",
            "val_raw_speed_rmse": best["val"]["rmse"],
            "source_run_dirs": [str(path) for path in args.run_dirs],
            "stgat_only": True,
            "causal_features_included": bool(args.include_causal_features),
            "advanced_causal_features_included": bool(args.include_advanced_causal_features),
            "filter_full_target_windows": bool(args.filter_full_target_windows),
            "grid_step": float(args.grid_step),
            "skip_grid": bool(args.skip_grid),
            "exact_convex_included": True,
        },
        "raw_metrics": {"speed": {k: v for k, v in best["test"].items() if k not in {"per_step", "report"}}},
        "raw_metrics_per_step": {"speed": best["test"]["per_step"]},
        "raw_metrics_report": {"speed": best["test"]["report"]},
        "val_raw_metrics": {"speed": {k: v for k, v in best["val"].items() if k not in {"per_step", "report"}}},
        "val_raw_metrics_per_step": {"speed": best["val"]["per_step"]},
        "val_raw_metrics_report": {"speed": best["val"]["report"]},
    }
    with (output_dir / "predictor_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected": best_name, "variants": summaries}, handle, ensure_ascii=False, indent=2)

    print(f"Selected STGAT-only ensemble: {best_name} (val RMSE={best['val']['rmse']:.4f})")
    for name, summary in summaries.items():
        report = summary["test"]["report"]
        print(
            f"{name}: val_rmse={summary['val']['rmse']:.4f} "
            f"test_rmse={summary['test']['rmse']:.4f} "
            f"15/30/60={report['15min']['rmse']:.4f}/"
            f"{report['30min']['rmse']:.4f}/{report['60min']['rmse']:.4f}"
        )
    print(output_dir / "predictor_test_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
