from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features  # noqa: E402
from predictor_normalization import (  # noqa: E402
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor  # noqa: E402
from tools.calibrate_metrla_speed_predictions import (  # noqa: E402
    apply_multi_blend,
    build_calendar_average_feature,
    compute_speed_metrics,
    fit_multi_blend,
    historical_average_targets,
    lagged_targets,
    load_json,
    resolve_checkpoint_state,
    rolling_history_mean_targets,
)
from tools.calibrate_nyc_stgat_speed_predictions import (  # noqa: E402
    collect_predictions,
    load_state_dict_shape_compatible,
    parse_report_steps,
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
    filter_split_indices_by_time_mask,
    infer_time_slot_minutes,
    load_observed_time_mask,
    load_time_meta_for_training,
    resolve_device,
    resolve_num_workers,
    resolve_precision,
)


def progress_log(output_dir: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {"time": datetime.now().isoformat(timespec="seconds"), "event": event}
    if payload:
        record.update(payload)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def build_model(
    meta: dict[str, Any],
    checkpoint_state: dict[str, torch.Tensor],
    nyc: dict[str, Any],
    node_features: np.ndarray,
    *,
    pred_horizon: int,
) -> STGATPredictor:
    time_feat_dim = max(int(node_features.shape[-1]) - 2, 0)
    checkpoint_edge_input_dim = int(checkpoint_state["edge_proj.weight"].shape[1])
    use_fixed_edge_length_feature = bool(
        meta.get(
            "use_fixed_edge_length_feature",
            checkpoint_edge_input_dim > 1 + time_feat_dim,
        )
    )
    return STGATPredictor(
        num_nodes=int(np.asarray(nyc["adj"]).shape[0]),
        edge_index=torch.from_numpy(np.asarray(nyc["edge_index"], dtype=np.int64)),
        edge_lengths=torch.from_numpy(np.asarray(nyc["edge_lengths"], dtype=np.float32)),
        adj_matrix=torch.from_numpy(np.asarray(nyc["adj"], dtype=np.float32)),
        hidden_dim=int(meta.get("hidden_dim", 32)),
        num_heads=int(meta.get("num_heads", 4)),
        num_st_blocks=int(meta.get("num_st_blocks", 2)),
        num_gtcn_layers=int(meta.get("num_gtcn_layers", 2)),
        kernel_size=int(meta.get("kernel_size", 3)),
        pred_horizon=pred_horizon,
        node_feat_dim=int(node_features.shape[-1]),
        adaptive_topk=int(meta.get("adaptive_topk", 20)),
        speed_adaptive_topk=int(meta.get("speed_adaptive_topk", meta.get("adaptive_topk", 20))),
        speed_use_adaptive=bool(meta.get("speed_use_adaptive", True)),
        speed_use_fixed_graph=bool(meta.get("speed_use_fixed_graph", True)),
        use_fixed_edge_length_feature=use_fixed_edge_length_feature,
        v_domain=str(meta.get("speed_adaptive_domain", "edge")),
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
    *,
    report_steps: dict[str, int],
    slot_minutes: int,
    extra: dict[str, Any] | None = None,
) -> None:
    summaries[name] = {
        "val": compute_speed_metrics(
            val_pred,
            val_target,
            val_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        ),
        "test": compute_speed_metrics(
            test_pred,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        ),
    }
    if extra:
        summaries[name].update(extra)


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


def write_prediction_cache(
    output_dir: Path,
    *,
    run_dirs: list[Path],
    val_model_features: list[np.ndarray],
    test_model_features: list[np.ndarray],
    shared_features: dict[str, np.ndarray],
    slot_minutes: int,
    report_steps: dict[str, int],
) -> None:
    cache_dir = output_dir / "prediction_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, run_dir in enumerate(run_dirs):
        stem = f"checkpoint_{index + 1:02d}_{run_dir.name}"
        np.savez_compressed(
            cache_dir / f"{stem}.npz",
            val_pred=val_model_features[index].astype(np.float32),
            test_pred=test_model_features[index].astype(np.float32),
        )
        with (cache_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "cache_schema": 1,
                    "run_dir": str(run_dir),
                    "checkpoint": str(run_dir / "stgat_best.pt"),
                    "slot_minutes": int(slot_minutes),
                    "report_steps": report_steps,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
    np.savez_compressed(cache_dir / "shared_features.npz", **shared_features)
    with (cache_dir / "shared_features.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "cache_schema": 1,
                "slot_minutes": int(slot_minutes),
                "report_steps": report_steps,
                "shared_keys": sorted(shared_features),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit a multi-checkpoint NYC STGAT speed ensemble with causal priors.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="auto", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--calibration-fit-split", choices=["val", "train_val"], default="val")
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--skip-grid", action="store_true", help="Skip the coarse convex grid after exact convex fitting.")
    parser.add_argument("--save-prediction-cache", action="store_true", help="Persist cached val/test predictions and causal features.")
    parser.add_argument("--cache-only", action="store_true", help="Stop after persisting prediction cache.")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log(
        output_dir,
        "ensemble_started",
        {
            "run_count": len(args.run_dirs),
            "data_dir": str(args.data_dir),
            "fit_split": args.calibration_fit_split,
            "grid_step": float(args.grid_step),
        },
    )

    first_meta = load_json(args.run_dirs[0] / "stgat_meta.json")
    pred_horizon = int(first_meta["pred_horizon"])
    hist_len = int(first_meta["hist_len"])
    for run_dir in args.run_dirs[1:]:
        meta = load_json(run_dir / "stgat_meta.json")
        if int(meta["pred_horizon"]) != pred_horizon or int(meta["hist_len"]) != hist_len:
            raise ValueError(
                "NYC ensemble currently requires matching hist_len and pred_horizon. "
                f"{run_dir} has hist_len={meta['hist_len']} pred_horizon={meta['pred_horizon']}."
            )

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    nyc = load_nyc_real_graph_features(
        args.data_dir,
        edge_length_source="osrm",
        add_time_features=bool(first_meta.get("time_feature_names", [])),
    )
    edge_speeds_raw = np.asarray(nyc["edge_speeds"], dtype=np.float32)
    time_meta = load_time_meta_for_training(args.data_dir, int(edge_speeds_raw.shape[0]))
    slot_minutes = int(first_meta.get("time_slot_minutes") or infer_time_slot_minutes(time_meta))
    split_indices = build_monthly_split_indices(time_meta, hist_len, pred_horizon)
    split_indices = filter_split_indices_by_time_mask(
        split_indices,
        load_observed_time_mask(args.data_dir, int(edge_speeds_raw.shape[0])),
        hist_len,
        pred_horizon,
    )
    train_window_mask = build_window_time_mask(
        int(edge_speeds_raw.shape[0]),
        split_indices["train"],
        hist_len,
        pred_horizon,
    )
    if "date" in time_meta.columns:
        time_meta = time_meta.copy()
        time_meta["date"] = np.asarray(time_meta["date"], dtype="datetime64[ns]")

    normalization_stats = load_normalization_stats(first_meta["normalization"])
    node_features = normalize_node_features(np.asarray(nyc["node_features"], dtype=np.float32), normalization_stats)
    edge_speeds = normalize_speed_features(edge_speeds_raw, normalization_stats, edge_axis=1)
    speed_valid_mask = np.ones_like(edge_speeds_raw, dtype=bool)
    full_dataset = SpatioTemporalDataset(
        node_features,
        edge_speeds,
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        history_imputation_enabled=False,
    )
    loaders = {
        split: DataLoader(
            Subset(full_dataset, split_indices[split]),
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for split in ("train", "val", "test")
    }
    report_steps = parse_report_steps(first_meta, pred_horizon, slot_minutes)

    val_model_features: list[np.ndarray] = []
    test_model_features: list[np.ndarray] = []
    fit_model_features: list[np.ndarray] = []
    feature_names: list[str] = []
    val_target = val_mask = test_target = test_mask = None
    val_persistence = test_persistence = fit_persistence = None
    fit_target = fit_mask = None
    checkpoint_loads: dict[str, Any] = {}

    for index, run_dir in enumerate(args.run_dirs):
        start = time.perf_counter()
        meta = load_json(run_dir / "stgat_meta.json")
        checkpoint_state = resolve_checkpoint_state(run_dir / "stgat_best.pt")
        model = build_model(meta, checkpoint_state, nyc, node_features, pred_horizon=pred_horizon).to(device)
        checkpoint_loads[run_dir.name] = load_state_dict_shape_compatible(model, checkpoint_state)
        progress_log(output_dir, "checkpoint_started", {"index": index + 1, "name": run_dir.name})
        val_pred, cur_val_target, cur_val_mask, cur_val_persistence = collect_predictions(
            model,
            loaders["val"],
            device=device,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            normalization_stats=normalization_stats,
        )
        test_pred, cur_test_target, cur_test_mask, cur_test_persistence = collect_predictions(
            model,
            loaders["test"],
            device=device,
            non_blocking=non_blocking,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            normalization_stats=normalization_stats,
        )
        if index == 0:
            val_target, val_mask = cur_val_target, cur_val_mask
            test_target, test_mask = cur_test_target, cur_test_mask
            val_persistence, test_persistence = cur_val_persistence, cur_test_persistence
        else:
            if not (np.array_equal(cur_val_target, val_target) and np.array_equal(cur_test_target, test_target)):
                raise RuntimeError(f"Target alignment mismatch for {run_dir}.")
        if args.calibration_fit_split == "train_val":
            train_pred, train_target, train_mask, train_persistence = collect_predictions(
                model,
                loaders["train"],
                device=device,
                non_blocking=non_blocking,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                normalization_stats=normalization_stats,
            )
            fit_pred = np.concatenate([train_pred, val_pred], axis=0)
            if index == 0:
                fit_target = np.concatenate([train_target, val_target], axis=0)
                fit_mask = np.concatenate([train_mask, val_mask], axis=0)
                fit_persistence = np.concatenate([train_persistence, val_persistence], axis=0)
        else:
            fit_pred = val_pred
            if index == 0:
                fit_target, fit_mask, fit_persistence = val_target, val_mask, val_persistence
        val_model_features.append(val_pred)
        test_model_features.append(test_pred)
        fit_model_features.append(fit_pred)
        feature_names.append(run_dir.name)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        progress_log(
            output_dir,
            "checkpoint_metrics_written",
            {
                "index": index + 1,
                "name": run_dir.name,
                "elapsed_seconds": round(time.perf_counter() - start, 2),
                "val_rmse": round(float(compute_speed_metrics(val_pred, val_target, val_mask, report_steps=report_steps, slot_minutes=slot_minutes)["rmse"]), 6),
                "test_rmse": round(float(compute_speed_metrics(test_pred, test_target, test_mask, report_steps=report_steps, slot_minutes=slot_minutes)["rmse"]), 6),
            },
        )

    assert val_target is not None and val_mask is not None and test_target is not None and test_mask is not None
    assert val_persistence is not None and test_persistence is not None
    assert fit_target is not None and fit_mask is not None and fit_persistence is not None

    calendar_average_by_time = build_calendar_average_feature(
        edge_speeds_raw,
        speed_valid_mask,
        time_meta,
        train_window_mask,
    )
    val_calendar = historical_average_targets(calendar_average_by_time, split_indices["val"], hist_len=hist_len, pred_horizon=pred_horizon)
    test_calendar = historical_average_targets(calendar_average_by_time, split_indices["test"], hist_len=hist_len, pred_horizon=pred_horizon)
    slots_per_day = int(pd.to_numeric(time_meta["slot"], errors="raise").max()) + 1
    val_lag_day = lagged_targets(edge_speeds_raw, speed_valid_mask, split_indices["val"], hist_len=hist_len, pred_horizon=pred_horizon, lag_steps=slots_per_day, fallback=val_calendar)
    test_lag_day = lagged_targets(edge_speeds_raw, speed_valid_mask, split_indices["test"], hist_len=hist_len, pred_horizon=pred_horizon, lag_steps=slots_per_day, fallback=test_calendar)
    val_lag_week = lagged_targets(edge_speeds_raw, speed_valid_mask, split_indices["val"], hist_len=hist_len, pred_horizon=pred_horizon, lag_steps=7 * slots_per_day, fallback=val_calendar)
    test_lag_week = lagged_targets(edge_speeds_raw, speed_valid_mask, split_indices["test"], hist_len=hist_len, pred_horizon=pred_horizon, lag_steps=7 * slots_per_day, fallback=test_calendar)
    val_history_mean_3 = rolling_history_mean_targets(edge_speeds_raw, speed_valid_mask, split_indices["val"], hist_len=hist_len, pred_horizon=pred_horizon, window=3, fallback=val_persistence)
    test_history_mean_3 = rolling_history_mean_targets(edge_speeds_raw, speed_valid_mask, split_indices["test"], hist_len=hist_len, pred_horizon=pred_horizon, window=3, fallback=test_persistence)
    val_history_mean_12 = rolling_history_mean_targets(edge_speeds_raw, speed_valid_mask, split_indices["val"], hist_len=hist_len, pred_horizon=pred_horizon, window=12, fallback=val_persistence)
    test_history_mean_12 = rolling_history_mean_targets(edge_speeds_raw, speed_valid_mask, split_indices["test"], hist_len=hist_len, pred_horizon=pred_horizon, window=12, fallback=test_persistence)

    if args.calibration_fit_split == "train_val":
        train_calendar = historical_average_targets(calendar_average_by_time, split_indices["train"], hist_len=hist_len, pred_horizon=pred_horizon)
        train_lag_day = lagged_targets(edge_speeds_raw, speed_valid_mask, split_indices["train"], hist_len=hist_len, pred_horizon=pred_horizon, lag_steps=slots_per_day, fallback=train_calendar)
        train_lag_week = lagged_targets(edge_speeds_raw, speed_valid_mask, split_indices["train"], hist_len=hist_len, pred_horizon=pred_horizon, lag_steps=7 * slots_per_day, fallback=train_calendar)
        train_history_mean_3 = rolling_history_mean_targets(edge_speeds_raw, speed_valid_mask, split_indices["train"], hist_len=hist_len, pred_horizon=pred_horizon, window=3, fallback=fit_persistence[: len(split_indices["train"])])
        train_history_mean_12 = rolling_history_mean_targets(edge_speeds_raw, speed_valid_mask, split_indices["train"], hist_len=hist_len, pred_horizon=pred_horizon, window=12, fallback=fit_persistence[: len(split_indices["train"])])
        fit_calendar = np.concatenate([train_calendar, val_calendar], axis=0)
        fit_lag_day = np.concatenate([train_lag_day, val_lag_day], axis=0)
        fit_lag_week = np.concatenate([train_lag_week, val_lag_week], axis=0)
        fit_history_mean_3 = np.concatenate([train_history_mean_3, val_history_mean_3], axis=0)
        fit_history_mean_12 = np.concatenate([train_history_mean_12, val_history_mean_12], axis=0)
    else:
        fit_calendar = val_calendar
        fit_lag_day = val_lag_day
        fit_lag_week = val_lag_week
        fit_history_mean_3 = val_history_mean_3
        fit_history_mean_12 = val_history_mean_12

    if args.save_prediction_cache or args.cache_only:
        shared_features = {
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
        write_prediction_cache(
            output_dir,
            run_dirs=list(args.run_dirs),
            val_model_features=val_model_features,
            test_model_features=test_model_features,
            shared_features=shared_features,
            slot_minutes=slot_minutes,
            report_steps=report_steps,
        )
        progress_log(output_dir, "prediction_cache_written", {"cache_dir": str(output_dir / "prediction_cache")})
        if args.cache_only:
            print(output_dir / "prediction_cache")
            return 0

    summaries: dict[str, Any] = {}
    for index, name in enumerate(feature_names):
        add_summary(
            summaries,
            f"component_{index + 1}_{name}",
            val_model_features[index],
            test_model_features[index],
            val_target,
            val_mask,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
            extra={"feature": name},
        )

    mean_val = np.mean(np.stack(val_model_features, axis=0), axis=0)
    mean_test = np.mean(np.stack(test_model_features, axis=0), axis=0)
    add_summary(
        summaries,
        "checkpoint_mean",
        mean_val,
        mean_test,
        val_target,
        val_mask,
        test_target,
        test_mask,
        report_steps=report_steps,
        slot_minutes=slot_minutes,
        extra={"features": feature_names},
    )

    feature_sets = {
        "blend_multickpt_persistence_calendar": (
            [*feature_names, "persistence", "calendar"],
            [*fit_model_features, fit_persistence, fit_calendar],
            [*val_model_features, val_persistence, val_calendar],
            [*test_model_features, test_persistence, test_calendar],
        ),
        "blend_multickpt_calendar_daylag": (
            [*feature_names, "persistence", "calendar", "lag_day"],
            [*fit_model_features, fit_persistence, fit_calendar, fit_lag_day],
            [*val_model_features, val_persistence, val_calendar, val_lag_day],
            [*test_model_features, test_persistence, test_calendar, test_lag_day],
        ),
        "blend_multickpt_calendar_day_week_lags": (
            [*feature_names, "persistence", "calendar", "lag_day", "lag_week"],
            [*fit_model_features, fit_persistence, fit_calendar, fit_lag_day, fit_lag_week],
            [*val_model_features, val_persistence, val_calendar, val_lag_day, val_lag_week],
            [*test_model_features, test_persistence, test_calendar, test_lag_day, test_lag_week],
        ),
        "blend_multickpt_calendar_lags_history": (
            [*feature_names, "persistence", "calendar", "lag_day", "lag_week", "history_mean_3", "history_mean_12"],
            [*fit_model_features, fit_persistence, fit_calendar, fit_lag_day, fit_lag_week, fit_history_mean_3, fit_history_mean_12],
            [*val_model_features, val_persistence, val_calendar, val_lag_day, val_lag_week, val_history_mean_3, val_history_mean_12],
            [*test_model_features, test_persistence, test_calendar, test_lag_day, test_lag_week, test_history_mean_3, test_history_mean_12],
        ),
    }

    best_name = ""
    best_val_rmse = float("inf")
    for name, summary in summaries.items():
        val_rmse = float(summary["val"]["rmse"])
        if val_rmse < best_val_rmse:
            best_name = name
            best_val_rmse = val_rmse

    for name, (names, fit_features, val_features, test_features) in feature_sets.items():
        coefs, bias = fit_multi_blend(fit_features, fit_target, fit_mask)
        val_cal = apply_multi_blend(val_features, coefs, bias)
        test_cal = apply_multi_blend(test_features, coefs, bias)
        add_summary(
            summaries,
            name,
            val_cal,
            test_cal,
            val_target,
            val_mask,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
            extra={"features": names},
        )
        val_rmse = float(summaries[name]["val"]["rmse"])
        if val_rmse < best_val_rmse:
            best_name = name
            best_val_rmse = val_rmse

    convex_feature_names = [
        *feature_names,
        "persistence",
        "calendar_average",
        "lag_day",
        "lag_week",
        "history_mean_3",
        "history_mean_12",
    ]
    convex_fit_features = [
        *fit_model_features,
        fit_persistence,
        fit_calendar,
        fit_lag_day,
        fit_lag_week,
        fit_history_mean_3,
        fit_history_mean_12,
    ]
    convex_val_features = [
        *val_model_features,
        val_persistence,
        val_calendar,
        val_lag_day,
        val_lag_week,
        val_history_mean_3,
        val_history_mean_12,
    ]
    convex_test_features = [
        *test_model_features,
        test_persistence,
        test_calendar,
        test_lag_day,
        test_lag_week,
        test_history_mean_3,
        test_history_mean_12,
    ]

    progress_log(output_dir, "convex_exact_started", {"feature_count": len(convex_feature_names)})
    exact_global_weights = fit_convex_weights_exact(convex_fit_features, fit_target, fit_mask)
    val_exact_global = apply_convex_weights(convex_val_features, exact_global_weights)
    test_exact_global = apply_convex_weights(convex_test_features, exact_global_weights)
    add_summary(
        summaries,
        "convex_exact_global_mse",
        val_exact_global,
        test_exact_global,
        val_target,
        val_mask,
        test_target,
        test_mask,
        report_steps=report_steps,
        slot_minutes=slot_minutes,
        extra={
            "features": convex_feature_names,
            "weights": [float(value) for value in exact_global_weights],
        },
    )
    val_rmse = float(summaries["convex_exact_global_mse"]["val"]["rmse"])
    if val_rmse < best_val_rmse:
        best_name = "convex_exact_global_mse"
        best_val_rmse = val_rmse

    exact_step_weights = fit_stepwise_convex_weights_exact(convex_fit_features, fit_target, fit_mask)
    val_exact_step = apply_stepwise_convex_weights(convex_val_features, exact_step_weights)
    test_exact_step = apply_stepwise_convex_weights(convex_test_features, exact_step_weights)
    add_summary(
        summaries,
        "convex_exact_per_step_mse",
        val_exact_step,
        test_exact_step,
        val_target,
        val_mask,
        test_target,
        test_mask,
        report_steps=report_steps,
        slot_minutes=slot_minutes,
        extra={
            "features": convex_feature_names,
            "weights_by_step": exact_step_weights.astype(float).tolist(),
        },
    )
    val_rmse = float(summaries["convex_exact_per_step_mse"]["val"]["rmse"])
    if val_rmse < best_val_rmse:
        best_name = "convex_exact_per_step_mse"
        best_val_rmse = val_rmse
    np.save(output_dir / "convex_exact_per_step_mse_weights.npy", exact_step_weights)

    exact_item_step_weights = fit_item_step_convex_weights_exact(convex_fit_features, fit_target, fit_mask)
    val_exact_item_step = apply_item_step_convex_weights(convex_val_features, exact_item_step_weights)
    test_exact_item_step = apply_item_step_convex_weights(convex_test_features, exact_item_step_weights)
    add_summary(
        summaries,
        "convex_exact_per_item_step_mse",
        val_exact_item_step,
        test_exact_item_step,
        val_target,
        val_mask,
        test_target,
        test_mask,
        report_steps=report_steps,
        slot_minutes=slot_minutes,
        extra={
            "features": convex_feature_names,
            "weights_shape": list(exact_item_step_weights.shape),
        },
    )
    val_rmse = float(summaries["convex_exact_per_item_step_mse"]["val"]["rmse"])
    if val_rmse < best_val_rmse:
        best_name = "convex_exact_per_item_step_mse"
        best_val_rmse = val_rmse
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
            {"feature_count": len(convex_feature_names), "grid_step": float(args.grid_step)},
        )
        grid = convex_weight_grid(len(convex_feature_names), step=float(args.grid_step))
        progress_log(output_dir, "convex_grid_built", {"weight_count": len(grid)})

        global_weights = grid[int(np.argmin(mse_grid_scores(convex_fit_features, fit_target, fit_mask, grid)))]
        val_global = apply_convex_weights(convex_val_features, global_weights)
        test_global = apply_convex_weights(convex_test_features, global_weights)
        add_summary(
            summaries,
            "convex_grid_global_mse",
            val_global,
            test_global,
            val_target,
            val_mask,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
            extra={
                "features": convex_feature_names,
                "weights": [float(value) for value in global_weights],
            },
        )
        val_rmse = float(summaries["convex_grid_global_mse"]["val"]["rmse"])
        if val_rmse < best_val_rmse:
            best_name = "convex_grid_global_mse"
            best_val_rmse = val_rmse

        step_weights = fit_stepwise_convex_weights(convex_fit_features, fit_target, fit_mask, grid, mode="mse")
        val_step = apply_stepwise_convex_weights(convex_val_features, step_weights)
        test_step = apply_stepwise_convex_weights(convex_test_features, step_weights)
        add_summary(
            summaries,
            "convex_grid_per_step_mse",
            val_step,
            test_step,
            val_target,
            val_mask,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
            extra={
                "features": convex_feature_names,
                "weights_by_step": step_weights.astype(float).tolist(),
            },
        )
        val_rmse = float(summaries["convex_grid_per_step_mse"]["val"]["rmse"])
        if val_rmse < best_val_rmse:
            best_name = "convex_grid_per_step_mse"
            best_val_rmse = val_rmse
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
            "fit_split": args.calibration_fit_split,
            "source_run_dirs": [str(path) for path in args.run_dirs],
            "stgat_only": True,
            "causal_features_included": True,
            "advanced_causal_features_included": True,
            "hist_len": hist_len,
            "pred_horizon": pred_horizon,
            "grid_step": float(args.grid_step),
            "skip_grid": bool(args.skip_grid),
            "checkpoint_loads": checkpoint_loads,
        },
        "raw_metrics": {"speed": {key: value for key, value in best["test"].items() if key not in {"per_step", "report"}}},
        "raw_metrics_per_step": {"speed": best["test"]["per_step"]},
        "raw_metrics_report": {"speed": best["test"]["report"]},
        "val_raw_metrics": {"speed": {key: value for key, value in best["val"].items() if key not in {"per_step", "report"}}},
        "val_raw_metrics_per_step": {"speed": best["val"]["per_step"]},
        "val_raw_metrics_report": {"speed": best["val"]["report"]},
    }
    with (output_dir / "predictor_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected": best_name, "variants": summaries}, handle, ensure_ascii=False, indent=2)

    print(f"Selected NYC STGAT ensemble: {best_name} (val RMSE={best['val']['rmse']:.4f})")
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
