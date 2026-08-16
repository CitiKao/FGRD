from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_speed_benchmarks.nyc_speed_prediction import (  # noqa: E402
    build_model,
    infer_dataset_format,
    load_dataset_bundle,
    make_loaders,
    resolve_device,
    resolve_precision,
)
from predictor_normalization import denormalize_speed_values  # noqa: E402
from tools.calibrate_metrla_speed_predictions import (  # noqa: E402
    apply_multi_blend,
    compute_speed_metrics,
    fit_multi_blend,
)


REPORT_STEPS = {"15min": 3, "30min": 6, "60min": 12}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def args_from_run(run_dir: Path, *, device: str, precision: str, batch_size: int) -> argparse.Namespace:
    meta = load_json(run_dir / "benchmark_meta.json")
    raw_args = dict(meta["args"])
    raw_args["device"] = device if device != "from_meta" else raw_args.get("device", "auto")
    raw_args["precision"] = precision if precision != "from_meta" else raw_args.get("precision", "auto")
    raw_args["batch_size"] = int(batch_size or raw_args.get("batch_size", 32))
    raw_args["num_workers"] = 0
    raw_args["max_eval_batches"] = 0
    raw_args["max_train_batches"] = 0
    args = argparse.Namespace(**raw_args)
    args.device_resolved = resolve_device(args.device)
    args.amp_enabled, args.amp_dtype, args.precision_resolved = resolve_precision(
        args.precision,
        args.device_resolved,
    )
    args.non_blocking = bool(args.device_resolved.type == "cuda")
    args.dataset_format_resolved = infer_dataset_format(args.data_dir, args.dataset_format)
    args.horizon_loss_weights_tensor = None
    return args


def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    *,
    args: argparse.Namespace,
    normalization_stats: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    mask_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(args.device_resolved, non_blocking=args.non_blocking)
            speed_seq = batch["speed_seq"].to(args.device_resolved, non_blocking=args.non_blocking)
            speed_target = batch["speed_target"].to(args.device_resolved, non_blocking=args.non_blocking)
            speed_target_mask = batch.get("speed_target_mask")
            if speed_target_mask is None:
                mask_np = np.ones(tuple(speed_target.shape), dtype=bool)
            else:
                mask_np = speed_target_mask.numpy().astype(bool)
            with torch.autocast(
                device_type=args.device_resolved.type,
                dtype=args.amp_dtype,
                enabled=args.amp_enabled,
            ):
                pred_norm = model(node_seq, speed_seq)
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
            mask_np &= np.abs(target_raw) > 1e-6
            pred_parts.append(pred_raw)
            target_parts.append(target_raw)
            mask_parts.append(mask_np)
    return np.concatenate(pred_parts), np.concatenate(target_parts), np.concatenate(mask_parts)


def load_model_predictions(
    run_dir: Path,
    *,
    bundle: Any,
    loaders: dict[str, torch.utils.data.DataLoader],
    base_args: argparse.Namespace,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    model_args = args_from_run(
        run_dir,
        device=str(base_args.device),
        precision=str(base_args.precision),
        batch_size=int(base_args.batch_size),
    )
    model = build_model(model_args, bundle).to(model_args.device_resolved)
    checkpoint = run_dir / "best_model.pt"
    if not checkpoint.exists():
        meta = load_json(run_dir / "benchmark_meta.json")
        checkpoint = Path(meta.get("selected_checkpoint", checkpoint))
    state = torch.load(checkpoint, map_location=model_args.device_resolved, weights_only=False)
    model.load_state_dict(state)
    return {
        split: collect_predictions(
            model,
            loaders[split],
            args=model_args,
            normalization_stats=bundle.normalization_stats,
        )
        for split in ("val", "test")
    }


def convex_weight_grid(num_features: int, step: float = 0.05) -> list[np.ndarray]:
    units = int(round(1.0 / float(step)))
    weights: list[np.ndarray] = []

    def visit(prefix: list[int], remaining: int, slots_left: int) -> None:
        if slots_left == 1:
            weights.append(np.asarray([*prefix, remaining], dtype=np.float32) / float(units))
            return
        for value in range(remaining + 1):
            visit([*prefix, value], remaining - value, slots_left - 1)

    visit([], units, int(num_features))
    return weights


def apply_convex_weights(feature_arrays: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    output = np.zeros_like(feature_arrays[0], dtype=np.float32)
    for idx, values in enumerate(feature_arrays):
        output += np.float32(weights[idx]) * values.astype(np.float32)
    return np.maximum(output, 0.0).astype(np.float32)


def apply_stepwise_convex_weights(feature_arrays: list[np.ndarray], weights_by_step: np.ndarray) -> np.ndarray:
    output = np.zeros_like(feature_arrays[0], dtype=np.float32)
    for step_idx in range(output.shape[-1]):
        weights = weights_by_step[step_idx]
        for feature_idx, values in enumerate(feature_arrays):
            output[..., step_idx] += np.float32(weights[feature_idx]) * values[..., step_idx].astype(np.float32)
    return np.maximum(output, 0.0).astype(np.float32)


def score_prediction(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, *, mode: str) -> float:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    valid = diff[mask]
    if valid.size == 0:
        return float("inf")
    if mode == "mae":
        return float(np.abs(valid).mean())
    return float(np.square(valid).mean())


def mse_grid_scores(
    feature_arrays: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    grid: list[np.ndarray],
) -> np.ndarray:
    num_features = len(feature_arrays)
    gram = np.zeros((num_features, num_features), dtype=np.float64)
    rhs = np.zeros(num_features, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    target_valid = target[valid].astype(np.float64)
    flattened = [values[valid].astype(np.float64) for values in feature_arrays]
    for i in range(num_features):
        rhs[i] = float(np.dot(flattened[i], target_valid))
        for j in range(i, num_features):
            value = float(np.dot(flattened[i], flattened[j]))
            gram[i, j] = value
            gram[j, i] = value
    weights = np.stack(grid).astype(np.float64)
    return np.einsum("gf,fh,gh->g", weights, gram, weights) - 2.0 * (weights @ rhs)


def fit_stepwise_convex_weights(
    feature_arrays: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    grid: list[np.ndarray],
    *,
    mode: str,
) -> np.ndarray:
    pred_horizon = int(target.shape[-1])
    weights_by_step = np.zeros((pred_horizon, len(feature_arrays)), dtype=np.float32)
    for step_idx in range(pred_horizon):
        step_features = [values[..., step_idx] for values in feature_arrays]
        step_target = target[..., step_idx]
        step_mask = mask[..., step_idx]
        if mode != "mse":
            raise ValueError("Only MSE stepwise convex fitting is currently optimized.")
        scores = mse_grid_scores(step_features, step_target, step_mask, grid)
        best_weights = grid[int(np.argmin(scores))]
        weights_by_step[step_idx] = best_weights
    return weights_by_step


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-fit ensemble for paper speed benchmark runs.")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="bf16", choices=["auto", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_args = args_from_run(
        args.run_dirs[0],
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
    )
    bundle = load_dataset_bundle(base_args)
    loaders = make_loaders(base_args, bundle)

    by_run = [
        load_model_predictions(run_dir, bundle=bundle, loaders=loaders, base_args=base_args)
        for run_dir in args.run_dirs
    ]
    val_target = by_run[0]["val"][1]
    val_mask = by_run[0]["val"][2]
    test_target = by_run[0]["test"][1]
    test_mask = by_run[0]["test"][2]
    val_features = [item["val"][0] for item in by_run]
    test_features = [item["test"][0] for item in by_run]

    summaries: dict[str, Any] = {}
    best_name = ""
    best_score = float("inf")
    for idx, run_dir in enumerate(args.run_dirs):
        val_metrics = compute_speed_metrics(val_features[idx], val_target, val_mask, report_steps=REPORT_STEPS)
        test_metrics = compute_speed_metrics(test_features[idx], test_target, test_mask, report_steps=REPORT_STEPS)
        name = f"component_{idx + 1}_{run_dir.name}"
        summaries[name] = {"val": val_metrics, "test": test_metrics, "run_dir": str(run_dir)}
        if float(val_metrics["rmse"]) < best_score:
            best_score = float(val_metrics["rmse"])
            best_name = name

    coefs, bias = fit_multi_blend(val_features, val_target, val_mask)
    val_blend = apply_multi_blend(val_features, coefs, bias)
    test_blend = apply_multi_blend(test_features, coefs, bias)
    val_blend_metrics = compute_speed_metrics(val_blend, val_target, val_mask, report_steps=REPORT_STEPS)
    test_blend_metrics = compute_speed_metrics(test_blend, test_target, test_mask, report_steps=REPORT_STEPS)
    summaries["linear_blend"] = {
        "val": val_blend_metrics,
        "test": test_blend_metrics,
        "coef_means": [float(coefs[idx].mean()) for idx in range(coefs.shape[0])],
        "coef_stds": [float(coefs[idx].std()) for idx in range(coefs.shape[0])],
        "bias_mean": float(bias.mean()),
        "bias_std": float(bias.std()),
    }
    if float(val_blend_metrics["rmse"]) < best_score:
        best_score = float(val_blend_metrics["rmse"])
        best_name = "linear_blend"
    np.save(output_dir / "linear_blend_coefs.npy", coefs)
    np.save(output_dir / "linear_blend_bias.npy", bias)

    grid = convex_weight_grid(len(val_features), step=0.05)
    global_scores = mse_grid_scores(val_features, val_target, val_mask, grid)
    best_grid_weights = grid[int(np.argmin(global_scores))]
    val_grid = apply_convex_weights(val_features, best_grid_weights)
    test_grid = apply_convex_weights(test_features, best_grid_weights)
    val_grid_metrics = compute_speed_metrics(val_grid, val_target, val_mask, report_steps=REPORT_STEPS)
    test_grid_metrics = compute_speed_metrics(test_grid, test_target, test_mask, report_steps=REPORT_STEPS)
    summaries["convex_grid_global_mse"] = {
        "val": val_grid_metrics,
        "test": test_grid_metrics,
        "weights": [float(value) for value in best_grid_weights],
    }
    if float(val_grid_metrics["rmse"]) < best_score:
        best_score = float(val_grid_metrics["rmse"])
        best_name = "convex_grid_global_mse"

    weights_by_step = fit_stepwise_convex_weights(
        val_features,
        val_target,
        val_mask,
        grid,
        mode="mse",
    )
    val_stepwise = apply_stepwise_convex_weights(val_features, weights_by_step)
    test_stepwise = apply_stepwise_convex_weights(test_features, weights_by_step)
    val_stepwise_metrics = compute_speed_metrics(
        val_stepwise,
        val_target,
        val_mask,
        report_steps=REPORT_STEPS,
    )
    test_stepwise_metrics = compute_speed_metrics(
        test_stepwise,
        test_target,
        test_mask,
        report_steps=REPORT_STEPS,
    )
    name = "convex_grid_per_step_mse"
    summaries[name] = {
        "val": val_stepwise_metrics,
        "test": test_stepwise_metrics,
        "weights_by_step": weights_by_step.astype(float).tolist(),
    }
    if float(val_stepwise_metrics["rmse"]) < best_score:
        best_score = float(val_stepwise_metrics["rmse"])
        best_name = name
    np.save(output_dir / f"{name}_weights.npy", weights_by_step)

    best = summaries[best_name]
    metrics_payload = {
        "ensemble": {
            "selected_variant": best_name,
            "selection_metric": "val_raw_speed_rmse",
            "val_raw_speed_rmse": best["val"]["rmse"],
            "source_run_dirs": [str(path) for path in args.run_dirs],
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

    print(f"Selected ensemble: {best_name} (val RMSE={best['val']['rmse']:.4f})")
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
