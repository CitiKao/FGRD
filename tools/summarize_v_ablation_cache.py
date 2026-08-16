#!/usr/bin/env python3
"""Summarize neural-stream diagnostics for a paper-exact V ablation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.verify_paper_v_from_cache import compute_metrics, load_json


def load_checkpoint_name(path: Path) -> str:
    metadata = load_json(path.with_suffix(".json"))
    return Path(metadata["run_dir"]).name


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        f"# V Ablation Diagnostic: {payload['ablation_mode']}",
        "",
        "The paper-exact frozen output and the four-neural-stream mean are reported separately.",
        "",
        "| Output | Horizon | RMSE | MAE | MAPE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for output_name, report in (
        ("Frozen paper aggregate", payload["frozen_paper_output"]["test"]["report"]),
        ("Four neural streams mean", payload["neural_stream_mean"]["test"]["report"]),
    ):
        for horizon in ("15min", "30min", "60min"):
            metrics = report[horizon]
            lines.append(
                f"| {output_name} | {horizon} | {metrics['rmse']:.6f} | "
                f"{metrics['mae']:.6f} | {metrics['mape']:.6f} |"
            )
    lines.extend(
        [
            "",
            "At the paper-reported horizons the frozen aggregate assigns zero total "
            "weight to the four neural streams. The neural-stream mean is therefore "
            "included as an architecture-sensitive diagnostic, not as the paper-facing output.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--frozen-results", type=Path, required=True)
    parser.add_argument("--ablation-mode", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_paths = sorted(args.cache_dir.glob("checkpoint_*.npz"))
    if len(checkpoint_paths) != 4:
        raise RuntimeError(
            f"Expected four checkpoint caches in {args.cache_dir}, found {len(checkpoint_paths)}"
        )

    with np.load(args.cache_dir / "shared_features.npz") as shared:
        val_target = shared["val_target"].astype(np.float32)
        val_mask = shared["val_mask"].astype(bool)
        test_target = shared["test_target"].astype(np.float32)
        test_mask = shared["test_mask"].astype(bool)

    val_mean = np.zeros_like(val_target, dtype=np.float32)
    test_mean = np.zeros_like(test_target, dtype=np.float32)
    per_stream: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        with np.load(checkpoint_path) as cached:
            val_pred = cached["val_pred"].astype(np.float32)
            test_pred = cached["test_pred"].astype(np.float32)
        val_mean += val_pred / np.float32(len(checkpoint_paths))
        test_mean += test_pred / np.float32(len(checkpoint_paths))
        per_stream.append(
            {
                "name": load_checkpoint_name(checkpoint_path),
                "val": compute_metrics(val_pred, val_target, val_mask),
                "test": compute_metrics(test_pred, test_target, test_mask),
            }
        )

    frozen = load_json(args.frozen_results)
    payload = {
        "ablation_mode": args.ablation_mode,
        "protocol": {
            "checkpoint_stream_count": 4,
            "retrained": False,
            "paper_output_uses_frozen_validation_selected_weights": True,
            "test_time_fitting": False,
        },
        "frozen_paper_output": frozen,
        "neural_stream_mean": {
            "val": compute_metrics(val_mean, val_target, val_mask),
            "test": compute_metrics(test_mean, test_target, test_mask),
        },
        "per_stream": per_stream,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "v_ablation_diagnostic.json"
    md_path = args.output_dir / "v_ablation_diagnostic.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown(payload, md_path)
    print(md_path.read_text(encoding="utf-8"), end="")
    print(f"JSON: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
