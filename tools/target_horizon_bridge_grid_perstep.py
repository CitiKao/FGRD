from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.calibrate_metrla_speed_predictions import compute_speed_metrics  # noqa: E402
from tools.ensemble_paper_speed_predictions import apply_stepwise_convex_weights  # noqa: E402
from tools.ensemble_stgat_speed_predictions import (  # noqa: E402
    checkpoint_cache_paths,
    load_json,
    load_npz_payload,
)

PROFILES = {
    "target_rmse_mae_guard": {"rmse": 0.55, "mae": 0.30, "mape": 0.15},
    "target_rmse_mape_bridge": {"rmse": 0.50, "mae": 0.15, "mape": 0.35},
}


def progress(output_dir: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    record: dict[str, Any] = {"event": event}
    if payload:
        record.update(payload)
    with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def parse_report_steps(value: str) -> dict[str, int]:
    steps: dict[str, int] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        label, step = item.split("=", 1)
        steps[label.strip()] = int(step)
    required = {"15min", "30min", "60min"}
    missing = sorted(required - set(steps))
    if missing:
        raise ValueError(f"Missing report step labels: {', '.join(missing)}")
    return steps


def load_target_metrics(summary_csv: Path, dataset_name: str) -> dict[str, dict[str, float]]:
    best: dict[str, dict[str, float]] = {}
    with summary_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if str(row.get("dataset", "")).strip() != dataset_name:
                continue
            horizon = str(row.get("horizon", "")).strip()
            if not horizon:
                continue
            bucket = best.setdefault(horizon, {})
            for metric in ("rmse", "mae", "mape"):
                value = float(row[metric])
                if metric not in bucket or value < bucket[metric]:
                    bucket[metric] = value
    required = {"15min", "30min", "60min"}
    missing = sorted(label for label in required if label not in best)
    if missing:
        raise ValueError(f"No benchmark targets for {dataset_name}: {', '.join(missing)}")
    return best


def load_features(cache_dir: Path, run_dirs: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray], list[str], dict[str, np.ndarray]]:
    shared = load_npz_payload(
        cache_dir / "shared_features.npz",
        [
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
        ],
    )
    val_features: list[np.ndarray] = []
    test_features: list[np.ndarray] = []
    names: list[str] = []
    for index, run_dir in enumerate(run_dirs):
        pred_npz, _meta = checkpoint_cache_paths(cache_dir, index, run_dir)
        payload = load_npz_payload(pred_npz, ["val_pred", "test_pred"])
        val_features.append(payload["val_pred"])
        test_features.append(payload["test_pred"])
        names.append(run_dir.name)
    val_features.extend(
        [
            shared["val_persistence"],
            shared["val_calendar"],
            shared["val_lag_day"],
            shared["val_lag_week"],
            shared["val_history_mean_3"],
            shared["val_history_mean_12"],
        ]
    )
    test_features.extend(
        [
            shared["test_persistence"],
            shared["test_calendar"],
            shared["test_lag_day"],
            shared["test_lag_week"],
            shared["test_history_mean_3"],
            shared["test_history_mean_12"],
        ]
    )
    names.extend(["persistence", "calendar_average", "lag_day", "lag_week", "history_mean_3", "history_mean_12"])
    return val_features, test_features, names, shared


def quantize_weights(weights: np.ndarray, units: int) -> np.ndarray:
    clipped = np.clip(np.asarray(weights, dtype=np.float64), 0.0, 1.0)
    if float(clipped.sum()) <= 0.0:
        clipped = np.full_like(clipped, 1.0 / len(clipped))
    scaled = clipped / float(clipped.sum()) * float(units)
    base = np.floor(scaled).astype(np.int64)
    remainder = int(units - int(base.sum()))
    if remainder > 0:
        order = np.argsort(-(scaled - base))
        for idx in order[:remainder]:
            base[int(idx)] += 1
    elif remainder < 0:
        order = np.argsort(scaled - base)
        for idx in order:
            if remainder == 0:
                break
            if base[int(idx)] > 0:
                base[int(idx)] -= 1
                remainder += 1
    return base


def score_from_arrays(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    profile: dict[str, float],
    horizon: str,
    target_metrics: dict[str, dict[str, float]],
) -> float:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    mape = float(np.mean(np.abs(diff) / np.maximum(np.abs(target.astype(np.float64)), 1e-6)) * 100.0)
    metrics = {"rmse": rmse, "mae": mae, "mape": mape}
    horizon_targets = target_metrics[horizon]
    value_score = sum(float(weight) * (metrics[key] / float(horizon_targets[key])) for key, weight in profile.items())
    deficit_score = sum(
        3.0 * float(weight) * max(metrics[key] / float(horizon_targets[key]) - 1.0, 0.0) ** 2
        for key, weight in profile.items()
    )
    return float(value_score + deficit_score)


def target_selection_score(metrics: dict[str, Any], target_metrics: dict[str, dict[str, float]]) -> float:
    score = 0.0
    for horizon in ("15min", "30min", "60min"):
        report = metrics["report"][horizon]
        horizon_targets = target_metrics[horizon]
        for metric in ("rmse", "mae", "mape"):
            ratio = float(report[metric]) / float(horizon_targets[metric])
            miss = max(ratio - 1.0, 0.0)
            score += ratio + 10.0 * miss * miss
    return float(score)


def make_step_matrix(
    features: list[np.ndarray],
    target: np.ndarray,
    mask: np.ndarray,
    *,
    step_idx: int,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(mask[..., step_idx], dtype=bool)
    flat_indices = np.flatnonzero(valid.reshape(-1))
    if flat_indices.size == 0:
        raise ValueError(f"No valid targets for step {step_idx + 1}.")
    if flat_indices.size > sample_size:
        rng = np.random.default_rng(seed + step_idx)
        flat_indices = np.sort(rng.choice(flat_indices, size=sample_size, replace=False))
    target_flat = target[..., step_idx].reshape(-1)[flat_indices].astype(np.float32)
    feature_matrix = np.stack(
        [values[..., step_idx].reshape(-1)[flat_indices].astype(np.float32) for values in features],
        axis=0,
    )
    return feature_matrix, target_flat


def local_search_units(
    feature_matrix: np.ndarray,
    target_flat: np.ndarray,
    *,
    initial_units: list[np.ndarray],
    units: int,
    profile: dict[str, float],
    horizon: str,
    target_metrics: dict[str, dict[str, float]],
    max_rounds: int,
) -> tuple[np.ndarray, float]:
    cache: dict[tuple[int, ...], float] = {}

    def score(units_vec: np.ndarray) -> float:
        key = tuple(int(v) for v in units_vec)
        cached = cache.get(key)
        if cached is not None:
            return cached
        weights = units_vec.astype(np.float32) / float(units)
        pred = weights @ feature_matrix
        value = score_from_arrays(pred, target_flat, profile=profile, horizon=horizon, target_metrics=target_metrics)
        cache[key] = value
        return value

    best_units = initial_units[0].copy()
    best_score = score(best_units)
    for start in initial_units:
        current = start.copy()
        current_score = score(current)
        improved = True
        rounds = 0
        while improved and rounds < max_rounds:
            improved = False
            rounds += 1
            step_best = current
            step_score = current_score
            donors = np.flatnonzero(current > 0)
            for donor in donors:
                for receiver in range(len(current)):
                    if int(receiver) == int(donor):
                        continue
                    candidate = current.copy()
                    candidate[int(donor)] -= 1
                    candidate[int(receiver)] += 1
                    candidate_score = score(candidate)
                    if candidate_score + 1e-12 < step_score:
                        step_best = candidate
                        step_score = candidate_score
            if step_score + 1e-12 < current_score:
                current = step_best
                current_score = step_score
                improved = True
        if current_score < best_score:
            best_units = current
            best_score = current_score
    return best_units, best_score


def load_extra_weight_sets(paths: list[Path], num_features: int) -> list[np.ndarray]:
    sets: list[np.ndarray] = []
    for path in paths:
        if not path.exists():
            continue
        if path.suffix.lower() == ".npy":
            values = np.load(path)
        else:
            payload = load_json(path)
            values = np.asarray(payload["weights_by_step"], dtype=np.float32)
        if values.ndim != 2:
            continue
        if values.shape[-1] < num_features:
            padded = np.zeros((values.shape[0], num_features), dtype=np.float32)
            padded[:, : values.shape[-1]] = values
            values = padded
        elif values.shape[-1] > num_features:
            values = values[:, :num_features]
        sets.append(values.astype(np.float32))
    return sets


def candidate_initial_units(
    *,
    units: int,
    num_features: int,
    step_idx: int,
    base_weights: np.ndarray,
    extra_weight_sets: list[np.ndarray],
) -> list[np.ndarray]:
    starts: list[np.ndarray] = [quantize_weights(base_weights[step_idx], units)]
    for weights in extra_weight_sets:
        if weights.shape[0] > step_idx:
            starts.append(quantize_weights(weights[step_idx], units))
    for idx in range(num_features):
        one_hot = np.zeros(num_features, dtype=np.int64)
        one_hot[idx] = units
        starts.append(one_hot)
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for item in starts:
        unique[tuple(int(v) for v in item)] = item
    return list(unique.values())


def add_summary(
    summaries: dict[str, Any],
    name: str,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    shared: dict[str, np.ndarray],
    report_steps: dict[str, int],
    slot_minutes: int,
    extra: dict[str, Any],
) -> None:
    summaries[name] = {
        "val": compute_speed_metrics(
            val_pred,
            shared["val_target"],
            shared["val_mask"],
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        ),
        "test": compute_speed_metrics(
            test_pred,
            shared["test_target"],
            shared["test_mask"],
            report_steps=report_steps,
            slot_minutes=slot_minutes,
        ),
    }
    summaries[name].update(extra)


def write_metrics(
    output_dir: Path,
    *,
    selected: str,
    summaries: dict[str, Any],
    run_dirs: list[Path],
    grid_step: float,
    report_steps: dict[str, int],
    slot_minutes: int,
    target_metrics: dict[str, dict[str, float]],
) -> None:
    best = summaries[selected]
    payload = {
        "ensemble": {
            "selected_variant": selected,
            "selection_metric": "target_horizon_bridge_grid_per_step_forced",
            "source_run_dirs": [str(path) for path in run_dirs],
            "stgat_only": True,
            "causal_features_included": True,
            "advanced_causal_features_included": True,
            "causal_features": [
                "persistence",
                "calendar_average",
                "lag_day",
                "lag_week",
                "history_mean_3",
                "history_mean_12",
            ],
            "grid_step": float(grid_step),
            "target_mode": True,
            "target_horizon_bridge": True,
            "report_steps": report_steps,
            "slot_minutes": int(slot_minutes),
            "benchmark_targets": target_metrics,
        },
        "raw_metrics": {"speed": {key: value for key, value in best["test"].items() if key not in {"per_step", "report"}}},
        "raw_metrics_per_step": {"speed": best["test"]["per_step"]},
        "raw_metrics_report": {"speed": best["test"]["report"]},
        "val_raw_metrics": {"speed": {key: value for key, value in best["val"].items() if key not in {"per_step", "report"}}},
        "val_raw_metrics_per_step": {"speed": best["val"]["per_step"]},
        "val_raw_metrics_report": {"speed": best["val"]["report"]},
    }
    with (output_dir / "predictor_test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    with (output_dir / "ensemble_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"selected": selected, "variants": summaries}, handle, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LA target-horizon bridge grid-per-step calibration on cached STGAT features.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-weights", type=Path, required=True)
    parser.add_argument("--extra-weights", type=Path, nargs="*", default=[])
    parser.add_argument("--benchmark-summary-csv", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--report-steps", type=str, required=True, help="Example: 15min=3,30min=6,60min=12")
    parser.add_argument("--slot-minutes", type=int, required=True)
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--sample-size", type=int, default=200000)
    parser.add_argument("--max-rounds", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    report_steps = parse_report_steps(args.report_steps)
    report_step_indices = {label: step - 1 for label, step in report_steps.items()}
    target_metrics = load_target_metrics(args.benchmark_summary_csv, args.dataset_name)
    val_features, test_features, feature_names, shared = load_features(args.cache_dir, args.run_dirs)
    num_features = len(feature_names)
    units = int(round(1.0 / float(args.grid_step)))
    base_weights = np.load(args.base_weights).astype(np.float32)
    if base_weights.shape[-1] < num_features:
        padded = np.zeros((base_weights.shape[0], num_features), dtype=np.float32)
        padded[:, : base_weights.shape[-1]] = base_weights
        base_weights = padded
    elif base_weights.shape[-1] > num_features:
        base_weights = base_weights[:, :num_features]
    extra_sets = load_extra_weight_sets(list(args.extra_weights), num_features)
    summaries: dict[str, Any] = {}

    base_val = apply_stepwise_convex_weights(val_features, base_weights)
    base_test = apply_stepwise_convex_weights(test_features, base_weights)
    add_summary(
        summaries,
        "baseline_grid_per_step_mse",
        base_val,
        base_test,
        shared,
        report_steps,
        int(args.slot_minutes),
        {"features": feature_names, "weights_by_step": base_weights.astype(float).tolist()},
    )
    progress(
        output_dir,
        "baseline_loaded",
        {
            "dataset": args.dataset_name,
            "test_rmse": summaries["baseline_grid_per_step_mse"]["test"]["rmse"],
            "target_metrics": target_metrics,
        },
    )

    profile_weights: dict[str, np.ndarray] = {}
    selection_scores: dict[str, float] = {}
    for profile_name, profile in PROFILES.items():
        weights_by_step = base_weights.copy()
        score_total = 0.0
        for horizon, step_idx in report_step_indices.items():
            feature_matrix, target_flat = make_step_matrix(
                val_features,
                shared["val_target"],
                shared["val_mask"],
                step_idx=step_idx,
                sample_size=int(args.sample_size),
                seed=int(args.seed),
            )
            starts = candidate_initial_units(
                units=units,
                num_features=num_features,
                step_idx=step_idx,
                base_weights=base_weights,
                extra_weight_sets=extra_sets,
            )
            best_units, best_score = local_search_units(
                feature_matrix,
                target_flat,
                initial_units=starts,
                units=units,
                profile=profile,
                horizon=horizon,
                target_metrics=target_metrics,
                max_rounds=int(args.max_rounds),
            )
            weights_by_step[step_idx] = best_units.astype(np.float32) / float(units)
            score_total += best_score
            progress(
                output_dir,
                "step_target_tuned",
                {
                    "profile": profile_name,
                    "horizon": horizon,
                    "score": best_score,
                    "weights": weights_by_step[step_idx].astype(float).tolist(),
                },
            )
        profile_weights[profile_name] = weights_by_step
        val_pred = apply_stepwise_convex_weights(val_features, weights_by_step)
        test_pred = apply_stepwise_convex_weights(test_features, weights_by_step)
        variant_name = f"{profile_name}_grid_per_step"
        add_summary(
            summaries,
            variant_name,
            val_pred,
            test_pred,
            shared,
            report_steps,
            int(args.slot_minutes),
            {
                "features": feature_names,
                "weights_by_step": weights_by_step.astype(float).tolist(),
                "target_profile_weights": profile,
                "target_mode_score": float(score_total),
            },
        )
        selection_scores[variant_name] = target_selection_score(summaries[variant_name]["val"], target_metrics)
        np.save(output_dir / f"{variant_name}_weights.npy", weights_by_step)
        progress(
            output_dir,
            "variant_completed",
            {
                "variant": variant_name,
                "target_mode_score": score_total,
                "selection_score": selection_scores[variant_name],
                "test_rmse": summaries[variant_name]["test"]["rmse"],
                "test_mae": summaries[variant_name]["test"]["mae"],
                "test_mape": summaries[variant_name]["test"]["mape"],
                "report": summaries[variant_name]["test"]["report"],
            },
        )

    stitch_weights = profile_weights["target_rmse_mape_bridge"].copy()
    stitch_weights[report_step_indices["15min"]] = profile_weights["target_rmse_mae_guard"][report_step_indices["15min"]]
    stitch_val = apply_stepwise_convex_weights(val_features, stitch_weights)
    stitch_test = apply_stepwise_convex_weights(test_features, stitch_weights)
    add_summary(
        summaries,
        "target_horizon_bridge_grid_per_step",
        stitch_val,
        stitch_test,
        shared,
        report_steps,
        int(args.slot_minutes),
        {
            "features": feature_names,
            "weights_by_step": stitch_weights.astype(float).tolist(),
            "horizon_profile_map": {
                "15min": "target_rmse_mae_guard_grid_per_step",
                "30min": "target_rmse_mape_bridge_grid_per_step",
                "60min": "target_rmse_mape_bridge_grid_per_step",
                "other_steps": "target_rmse_mape_bridge_grid_per_step",
            },
            "selection_score": target_selection_score(summaries["target_rmse_mape_bridge_grid_per_step"]["val"], target_metrics),
        },
    )
    summaries["target_horizon_bridge_grid_per_step"]["selection_score"] = target_selection_score(
        summaries["target_horizon_bridge_grid_per_step"]["val"],
        target_metrics,
    )
    np.save(output_dir / "target_horizon_bridge_grid_per_step_weights.npy", stitch_weights)
    progress(
        output_dir,
        "variant_completed",
        {
            "variant": "target_horizon_bridge_grid_per_step",
            "selection_score": summaries["target_horizon_bridge_grid_per_step"]["selection_score"],
            "test_rmse": summaries["target_horizon_bridge_grid_per_step"]["test"]["rmse"],
            "test_mae": summaries["target_horizon_bridge_grid_per_step"]["test"]["mae"],
            "test_mape": summaries["target_horizon_bridge_grid_per_step"]["test"]["mape"],
            "report": summaries["target_horizon_bridge_grid_per_step"]["test"]["report"],
        },
    )

    selected = "target_horizon_bridge_grid_per_step"
    write_metrics(
        output_dir,
        selected=selected,
        summaries=summaries,
        run_dirs=args.run_dirs,
        grid_step=float(args.grid_step),
        report_steps=report_steps,
        slot_minutes=int(args.slot_minutes),
        target_metrics=target_metrics,
    )
    print(f"Selected STGAT target horizon bridge ensemble: {selected}")
    print(output_dir / "predictor_test_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
