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

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features  # noqa: E402
from predictor_normalization import (  # noqa: E402
    denormalize_speed_values,
    load_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor  # noqa: E402
from tools.calibrate_metrla_speed_predictions import (  # noqa: E402
    apply_blend,
    apply_calibration,
    apply_multi_blend,
    build_calendar_average_feature,
    compute_speed_metrics,
    fit_affine,
    fit_bias,
    fit_blend,
    fit_identity,
    fit_multi_blend,
    historical_average_targets,
    lagged_targets,
    load_json,
    resolve_checkpoint_state,
    rolling_history_mean_targets,
)
from train_predictor import (  # noqa: E402
    build_monthly_split_indices,
    build_window_time_mask,
    configure_cuda_runtime,
    extract_temporal_context,
    filter_split_indices_by_time_mask,
    infer_time_slot_minutes,
    load_observed_time_mask,
    load_time_meta_for_training,
    resolve_device,
    resolve_num_workers,
    resolve_precision,
)


def parse_report_steps(meta: dict[str, Any], pred_horizon: int, slot_minutes: int) -> dict[str, int]:
    report_horizons = meta.get("report_horizons", {})
    if isinstance(report_horizons, dict):
        minutes = report_horizons.get("resolved_minutes", [])
        steps = report_horizons.get("resolved_steps", [])
        if isinstance(minutes, list) and isinstance(steps, list) and len(minutes) == len(steps):
            return {f"{int(minute)}min": int(step) for minute, step in zip(minutes, steps)}
    defaults = (15, 30, 60)
    return {
        f"{minute}min": int(minute // slot_minutes)
        for minute in defaults
        if minute % slot_minutes == 0 and 1 <= int(minute // slot_minutes) <= pred_horizon
    }


def load_state_dict_shape_compatible(model: torch.nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        return {
            "mode": "strict_false",
            "loaded_keys": int(len(state)),
            "skipped_keys": 0,
            "missing_keys": list(missing),
            "unexpected_keys": list(unexpected),
        }
    except RuntimeError as exc:
        target_state = model.state_dict()
        compatible = {
            key: value
            for key, value in state.items()
            if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
        }
        if not compatible:
            raise
        incompatible = model.load_state_dict(compatible, strict=False)
        return {
            "mode": "shape_compatible",
            "loaded_keys": int(len(compatible)),
            "skipped_keys": int(len(state) - len(compatible)),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "strict_error": str(exc)[:2000],
        }


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
            speed_target = batch["speed_target"].to(device, non_blocking=non_blocking)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred_norm = model.forward_v(speed_seq, extract_temporal_context(node_seq))

            pred_raw = denormalize_speed_values(
                pred_norm.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            target_raw = denormalize_speed_values(
                speed_target.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            last_speed_raw = denormalize_speed_values(
                speed_seq[:, :, -1:].detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            persistence_raw = np.repeat(last_speed_raw, pred_raw.shape[-1], axis=-1)
            pred_parts.append(pred_raw)
            target_parts.append(target_raw)
            mask_parts.append(np.ones(tuple(target_raw.shape), dtype=bool))
            persistence_parts.append(persistence_raw)
    return (
        np.concatenate(pred_parts),
        np.concatenate(target_parts),
        np.concatenate(mask_parts),
        np.concatenate(persistence_parts),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit validation-only calibration for an exported NYC STGAT speed run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
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
    checkpoint_state = resolve_checkpoint_state(run_dir / "stgat_best.pt")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    data_dir = args.data_dir or Path(str(meta.get("data_dir", "data")))
    pred_horizon = int(meta["pred_horizon"])
    hist_len = int(meta["hist_len"])
    nyc = load_nyc_real_graph_features(
        data_dir,
        edge_length_source="osrm",
        add_time_features=bool(meta.get("time_feature_names", [])),
    )
    edge_speeds_raw = np.asarray(nyc["edge_speeds"], dtype=np.float32)
    time_meta = load_time_meta_for_training(data_dir, int(edge_speeds_raw.shape[0]))
    slot_minutes = int(meta.get("time_slot_minutes") or infer_time_slot_minutes(time_meta))
    split_indices = build_monthly_split_indices(time_meta, hist_len, pred_horizon)
    split_indices = filter_split_indices_by_time_mask(
        split_indices,
        load_observed_time_mask(data_dir, int(edge_speeds_raw.shape[0])),
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

    normalization_stats = load_normalization_stats(meta["normalization"])
    node_features = normalize_node_features(np.asarray(nyc["node_features"], dtype=np.float32), normalization_stats)
    edge_speeds = normalize_speed_features(
        edge_speeds_raw,
        normalization_stats,
        edge_axis=1,
    )
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

    time_feat_dim = max(int(node_features.shape[-1]) - 2, 0)
    checkpoint_edge_input_dim = int(checkpoint_state["edge_proj.weight"].shape[1])
    use_fixed_edge_length_feature = bool(
        meta.get(
            "use_fixed_edge_length_feature",
            checkpoint_edge_input_dim > 1 + time_feat_dim,
        )
    )
    model = STGATPredictor(
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
    ).to(device)
    load_summary = load_state_dict_shape_compatible(model, checkpoint_state)

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
    train_pred = train_target = train_mask = train_persistence = None
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

    calendar_average_by_time = build_calendar_average_feature(
        edge_speeds_raw,
        speed_valid_mask,
        time_meta,
        train_window_mask,
    )
    val_calendar_average = historical_average_targets(
        calendar_average_by_time,
        split_indices["val"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
    )
    test_calendar_average = historical_average_targets(
        calendar_average_by_time,
        split_indices["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
    )
    slots_per_day = int(pd.to_numeric(time_meta["slot"], errors="raise").max()) + 1
    val_lag_day = lagged_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        lag_steps=slots_per_day,
        fallback=val_calendar_average,
    )
    test_lag_day = lagged_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        lag_steps=slots_per_day,
        fallback=test_calendar_average,
    )
    val_lag_week = lagged_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        lag_steps=7 * slots_per_day,
        fallback=val_calendar_average,
    )
    test_lag_week = lagged_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        lag_steps=7 * slots_per_day,
        fallback=test_calendar_average,
    )
    val_history_mean_3 = rolling_history_mean_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        window=3,
        fallback=val_persistence,
    )
    test_history_mean_3 = rolling_history_mean_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        window=3,
        fallback=test_persistence,
    )
    val_history_mean_12 = rolling_history_mean_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["val"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        window=12,
        fallback=val_persistence,
    )
    test_history_mean_12 = rolling_history_mean_targets(
        edge_speeds_raw,
        speed_valid_mask,
        split_indices["test"],
        hist_len=hist_len,
        pred_horizon=pred_horizon,
        window=12,
        fallback=test_persistence,
    )

    if args.calibration_fit_split == "val":
        fit_pred = val_pred
        fit_target = val_target
        fit_mask = val_mask
        fit_persistence = val_persistence
        fit_calendar_average = val_calendar_average
        fit_lag_day = val_lag_day
        fit_lag_week = val_lag_week
        fit_history_mean_3 = val_history_mean_3
        fit_history_mean_12 = val_history_mean_12
    else:
        assert train_pred is not None
        assert train_target is not None
        assert train_mask is not None
        assert train_persistence is not None
        train_calendar_average = historical_average_targets(
            calendar_average_by_time,
            split_indices["train"],
            hist_len=hist_len,
            pred_horizon=pred_horizon,
        )
        train_lag_day = lagged_targets(
            edge_speeds_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            lag_steps=slots_per_day,
            fallback=train_calendar_average,
        )
        train_lag_week = lagged_targets(
            edge_speeds_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            lag_steps=7 * slots_per_day,
            fallback=train_calendar_average,
        )
        train_history_mean_3 = rolling_history_mean_targets(
            edge_speeds_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            window=3,
            fallback=train_persistence,
        )
        train_history_mean_12 = rolling_history_mean_targets(
            edge_speeds_raw,
            speed_valid_mask,
            split_indices["train"],
            hist_len=hist_len,
            pred_horizon=pred_horizon,
            window=12,
            fallback=train_persistence,
        )
        fit_pred = np.concatenate([train_pred, val_pred], axis=0)
        fit_target = np.concatenate([train_target, val_target], axis=0)
        fit_mask = np.concatenate([train_mask, val_mask], axis=0)
        fit_persistence = np.concatenate([train_persistence, val_persistence], axis=0)
        fit_calendar_average = np.concatenate([train_calendar_average, val_calendar_average], axis=0)
        fit_lag_day = np.concatenate([train_lag_day, val_lag_day], axis=0)
        fit_lag_week = np.concatenate([train_lag_week, val_lag_week], axis=0)
        fit_history_mean_3 = np.concatenate([train_history_mean_3, val_history_mean_3], axis=0)
        fit_history_mean_12 = np.concatenate([train_history_mean_12, val_history_mean_12], axis=0)

    report_steps = parse_report_steps(meta, pred_horizon, slot_minutes)
    summaries: dict[str, Any] = {}
    best_name = ""
    best_val_rmse = float("inf")
    for name, fitter in {"identity": fit_identity, "bias": fit_bias, "affine": fit_affine}.items():
        scale, bias = fitter(fit_pred, fit_target, fit_mask)
        val_cal = apply_calibration(val_pred, scale, bias)
        test_cal = apply_calibration(test_pred, scale, bias)
        val_metrics = compute_speed_metrics(
            val_cal,
            val_target,
            val_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        )
        test_metrics = compute_speed_metrics(
            test_cal,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        )
        summaries[name] = {"val": val_metrics, "test": test_metrics}
        if float(val_metrics["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_name = name

    coef_model, coef_persistence, blend_bias = fit_blend(
        fit_pred,
        fit_persistence,
        fit_target,
        fit_mask,
    )
    for name, (val_features, test_features) in {
        "blend_persistence": (
            [val_pred, val_persistence],
            [test_pred, test_persistence],
        ),
    }.items():
        val_cal = apply_blend(val_features[0], val_features[1], coef_model, coef_persistence, blend_bias)
        test_cal = apply_blend(test_features[0], test_features[1], coef_model, coef_persistence, blend_bias)
        val_metrics = compute_speed_metrics(
            val_cal,
            val_target,
            val_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        )
        test_metrics = compute_speed_metrics(
            test_cal,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        )
        summaries[name] = {"val": val_metrics, "test": test_metrics}
        if float(val_metrics["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_name = name

    feature_sets = {
        "blend_persistence_calendar": (
            ["model", "persistence", "calendar"],
            [fit_pred, fit_persistence, fit_calendar_average],
            [val_pred, val_persistence, val_calendar_average],
            [test_pred, test_persistence, test_calendar_average],
        ),
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
        val_metrics = compute_speed_metrics(
            val_cal,
            val_target,
            val_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        )
        test_metrics = compute_speed_metrics(
            test_cal,
            test_target,
            test_mask,
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        )
        summaries[name] = {"val": val_metrics, "test": test_metrics, "features": feature_names}
        if float(val_metrics["rmse"]) < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse"])
            best_name = name

    best = summaries[best_name]
    metrics_payload = {
        "calibration": {
            "selected_variant": best_name,
            "selection_metric": "val_raw_speed_rmse",
            "val_raw_speed_rmse": best["val"]["rmse"],
            "fit_split": args.calibration_fit_split,
            "source_run_dir": str(run_dir),
            "data_dir": str(data_dir),
            "checkpoint_load": load_summary,
        },
        "raw_metrics": {"speed": {key: value for key, value in best["test"].items() if key not in {"per_step", "report"}}},
        "raw_metrics_per_step": {"speed": best["test"]["per_step"]},
        "raw_metrics_report": {"speed": best["test"]["report"]},
        "val_raw_metrics": {"speed": {key: value for key, value in best["val"].items() if key not in {"per_step", "report"}}},
        "val_raw_metrics_per_step": {"speed": best["val"]["per_step"]},
        "val_raw_metrics_report": {"speed": best["val"]["report"]},
    }
    with (output_dir / "calibrated_predictor_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "calibration_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected": best_name, "variants": summaries}, handle, ensure_ascii=False, indent=2)

    print(f"Selected NYC STGAT calibration: {best_name} (val RMSE={best['val']['rmse']:.4f})")
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
