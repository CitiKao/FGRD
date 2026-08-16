from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


REPORT_STEPS = {"15min": 1, "30min": 2, "60min": 4}
HISTORICAL_FEATURES = (
    ("persistence", "persistence"),
    ("calendar_average", "calendar"),
    ("lag_day", "lag_day"),
    ("lag_week", "lag_week"),
    ("history_mean_3", "history_mean_3"),
    ("history_mean_12", "history_mean_12"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_bucket(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    valid = diff[mask]
    if valid.size == 0:
        raise RuntimeError("Metric mask contains no valid values.")
    mse = float(np.square(valid).mean())
    mae = float(np.abs(valid).mean())
    mape_mask = mask & (np.abs(target) > 1e-6)
    mape = float(
        (
            np.abs(diff[mape_mask])
            / np.abs(target.astype(np.float64)[mape_mask])
            * 100.0
        ).mean()
    )
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": mae, "mape": mape}


def compute_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    per_step: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for step_idx in range(pred.shape[-1]):
        metrics = metric_bucket(
            pred[..., step_idx],
            target[..., step_idx],
            mask[..., step_idx],
        )
        per_step[f"step_{step_idx + 1}"] = {
            "step": step_idx + 1,
            "minutes": (step_idx + 1) * 15,
            **metrics,
        }
    for label, step in REPORT_STEPS.items():
        report[label] = dict(per_step[f"step_{step}"])
    return {**metric_bucket(pred, target, mask), "per_step": per_step, "report": report}


def load_feature_arrays(
    cache_dir: Path,
    split: str,
) -> tuple[list[np.ndarray], list[str], np.ndarray, np.ndarray]:
    checkpoint_paths = sorted(cache_dir.glob("checkpoint_*.npz"))
    if len(checkpoint_paths) != 4:
        raise RuntimeError(
            f"Expected exactly four checkpoint caches in {cache_dir}, found {len(checkpoint_paths)}."
        )

    features: list[np.ndarray] = []
    names: list[str] = []
    for checkpoint_path in checkpoint_paths:
        with np.load(checkpoint_path) as payload:
            features.append(payload[f"{split}_pred"].astype(np.float32))
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata = load_json(metadata_path)
        names.append(Path(metadata["run_dir"]).name)

    with np.load(cache_dir / "shared_features.npz") as shared:
        target = shared[f"{split}_target"].astype(np.float32)
        mask = shared[f"{split}_mask"].astype(bool)
        for name, cache_key in HISTORICAL_FEATURES:
            features.append(shared[f"{split}_{cache_key}"].astype(np.float32))
            names.append(name)
    return features, names, target, mask


def apply_weights(features: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if weights.shape != (features[0].shape[-1], len(features)):
        raise ValueError(
            f"Weight shape {weights.shape} does not match "
            f"{features[0].shape[-1]} steps x {len(features)} features."
        )
    output = np.zeros_like(features[0], dtype=np.float32)
    for step_idx in range(output.shape[-1]):
        for feature_idx, values in enumerate(features):
            output[..., step_idx] += (
                np.float32(weights[step_idx, feature_idx])
                * values[..., step_idx].astype(np.float32)
            )
    return np.maximum(output, 0.0).astype(np.float32)


def expected_report(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if "raw_metrics_report" in payload:
        report = payload["raw_metrics_report"]
        return report.get("speed", report)
    if "report" in payload:
        return payload["report"]
    raise KeyError(f"Cannot locate expected report metrics in {path}")


def compare_report(
    actual: dict[str, Any],
    expected: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    passed = True
    differences: dict[str, dict[str, float]] = {}
    for label in REPORT_STEPS:
        differences[label] = {}
        for metric in ("rmse", "mae", "mape"):
            delta = float(actual[label][metric]) - float(expected[label][metric])
            differences[label][metric] = delta
            passed = passed and abs(delta) <= tolerance
    return {
        "passed": passed,
        "absolute_tolerance": tolerance,
        "differences": differences,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# NYC V Paper-Exact Frozen Evaluation",
        "",
        "| Horizon | RMSE | MAE | MAPE | Neural weight sum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    report = payload["test"]["report"]
    for label, step in REPORT_STEPS.items():
        metrics = report[label]
        neural_sum = payload["weights"]["neural_weight_sum_by_step"][step - 1]
        lines.append(
            f"| {label} | {metrics['rmse']:.6f} | {metrics['mae']:.6f} | "
            f"{metrics['mape']:.6f} | {neural_sum:.6f} |"
        )
    lines.extend(
        [
            "",
            "The candidate pool contains four ST-GAPV streams followed by six causal "
            "historical candidates. The frozen validation-selected weights are applied "
            "without test-time fitting.",
            "",
            f"Reference check: **{'PASS' if payload['expected_comparison']['passed'] else 'FAIL'}**.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute the paper-facing NYC V metrics from frozen prediction caches and weights."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    weights = np.load(args.weights).astype(np.float32)
    if np.any(weights < -1e-7):
        raise ValueError("Frozen convex weights contain a negative value.")
    row_sums = weights.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ValueError(f"Frozen convex weights do not sum to one: {row_sums.tolist()}")

    split_metrics: dict[str, Any] = {}
    feature_names: list[str] | None = None
    for split in ("val", "test"):
        features, names, target, mask = load_feature_arrays(args.cache_dir, split)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("Feature order differs between validation and test caches.")
        pred = apply_weights(features, weights)
        split_metrics[split] = compute_metrics(pred, target, mask)

    expected = expected_report(args.expected_json)
    comparison = compare_report(
        split_metrics["test"]["report"],
        expected,
        args.tolerance,
    )
    payload = {
        "protocol": {
            "calibration": "frozen validation-selected horizon-wise convex weights",
            "test_time_fitting": False,
            "report_steps": REPORT_STEPS,
        },
        "features": feature_names,
        "weights": {
            "path": str(args.weights),
            "values_by_step": weights.tolist(),
            "row_sums": row_sums.tolist(),
            "neural_weight_sum_by_step": weights[:, :4].sum(axis=1).tolist(),
        },
        "val": split_metrics["val"],
        "test": split_metrics["test"],
        "expected_comparison": comparison,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "paper_exact_v_results.json"
    md_path = args.output_dir / "paper_exact_v_results.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, md_path)
    print(md_path.read_text(encoding="utf-8"), end="")
    print(f"JSON: {json_path}")
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
