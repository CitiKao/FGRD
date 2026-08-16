from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_LABELS = ("15min", "30min", "60min")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def metric_score(metrics: dict[str, Any], mape_weight: float) -> float:
    return (
        float(metrics["rmse"])
        + float(metrics["mae"])
        + mape_weight * float(metrics["mape"])
    )


def metric_source(result: dict[str, Any], split: str) -> dict[str, Any]:
    calibrated = result.get(f"{split}_gap_calibrated")
    if calibrated:
        return calibrated
    raw = result.get(f"{split}_gap")
    if raw:
        return raw
    raise KeyError(f"Missing {split} GAP metrics for {result.get('checkpoint')}")


def select_by_validation(
    results: list[dict[str, Any]],
    mape_weight: float,
) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for label in REPORT_LABELS:
        candidates: list[dict[str, Any]] = []
        for result in results:
            val_metrics = metric_source(result, "val")["report"][label]
            test_metrics = metric_source(result, "test")["report"][label]
            candidates.append(
                {
                    "checkpoint": str(result["checkpoint"]),
                    "epoch": result.get("epoch"),
                    "validation_score": metric_score(val_metrics, mape_weight),
                    "validation_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
            )
        selected[label] = min(candidates, key=lambda item: item["validation_score"])
    return selected


def compare_expected(
    selected: dict[str, dict[str, Any]],
    expected_path: Path | None,
    tolerance: float,
) -> dict[str, Any] | None:
    if expected_path is None:
        return None
    expected = load_json(expected_path)
    expected_report = expected["report"]
    differences: dict[str, dict[str, float]] = {}
    passed = True
    for label in REPORT_LABELS:
        actual_metrics = selected[label]["test_metrics"]
        expected_metrics = expected_report[label]
        differences[label] = {}
        for metric in ("rmse", "mae", "mape"):
            delta = float(actual_metrics[metric]) - float(expected_metrics[metric])
            differences[label][metric] = delta
            passed = passed and abs(delta) <= tolerance
    return {
        "expected_file": str(expected_path),
        "absolute_tolerance": tolerance,
        "passed": passed,
        "differences": differences,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# NYC GAP Paper-Exact Evaluation",
        "",
        "Checkpoint selection uses validation metrics only. Test metrics are read after selection.",
        "",
        "| Horizon | Selected checkpoint | Epoch | Val score | Test RMSE | Test MAE | Test MAPE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label in REPORT_LABELS:
        item = payload["selected"][label]
        metrics = item["test_metrics"]
        lines.append(
            f"| {label} | `{Path(item['checkpoint']).name}` | "
            f"{item.get('epoch', '')} | {item['validation_score']:.6f} | "
            f"{float(metrics['rmse']):.6f} | {float(metrics['mae']):.6f} | "
            f"{float(metrics['mape']):.6f} |"
        )
    comparison = payload.get("expected_comparison")
    if comparison:
        lines.extend(
            [
                "",
                f"Reference check: **{'PASS' if comparison['passed'] else 'FAIL'}** "
                f"(absolute tolerance {comparison['absolute_tolerance']}).",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the paper-facing NYC GAP checkpoint independently per report horizon."
    )
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-json", type=Path)
    parser.add_argument("--mape-weight", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    evaluation = load_json(args.evaluation_json)
    results = list(evaluation.get("results", []))
    if not results:
        raise RuntimeError(f"No checkpoint results found in {args.evaluation_json}")

    selected = select_by_validation(results, args.mape_weight)
    comparison = compare_expected(selected, args.expected_json, args.tolerance)
    payload = {
        "selection_protocol": {
            "split": "validation",
            "score": f"RMSE + MAE + {args.mape_weight:g} * MAPE",
            "test_used_for_selection": False,
        },
        "selected": selected,
        "expected_comparison": comparison,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "paper_exact_gap_results.json"
    md_path = args.output_dir / "paper_exact_gap_results.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(payload, md_path)

    print(md_path.read_text(encoding="utf-8"), end="")
    print(f"JSON: {json_path}")
    if comparison and not comparison["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
