#!/usr/bin/env python3
"""Replay all four packaged V checkpoint streams under one structural ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tools import ensemble_nyc_stgat_speed_predictions as ensemble


ABLATION_MODES = (
    "no_adaptive_speed_matrix",
    "single_st_block",
    "no_external_graph",
)


def find_option(arguments: list[str], option: str) -> str | None:
    try:
        index = arguments.index(option)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation-mode", required=True, choices=ABLATION_MODES)
    known, remaining = parser.parse_known_args()

    original_build_model = ensemble.build_model

    def build_ablation_model(
        meta: dict[str, Any],
        checkpoint_state: dict[str, Any],
        nyc: dict[str, Any],
        node_features: Any,
        *,
        pred_horizon: int,
    ) -> Any:
        overridden = dict(meta)
        if known.ablation_mode == "no_adaptive_speed_matrix":
            overridden["speed_use_adaptive"] = False
            overridden["speed_use_fixed_graph"] = True
        elif known.ablation_mode == "single_st_block":
            overridden["num_st_blocks"] = 1
        elif known.ablation_mode == "no_external_graph":
            overridden["speed_use_adaptive"] = True
            overridden["speed_use_fixed_graph"] = False
        return original_build_model(
            overridden,
            checkpoint_state,
            nyc,
            node_features,
            pred_horizon=pred_horizon,
        )

    ensemble.build_model = build_ablation_model
    output_dir_text = find_option(remaining, "--output-dir")
    print(
        json.dumps(
            {
                "event": "v_checkpoint_stream_ablation",
                "ablation_mode": known.ablation_mode,
                "protocol": (
                    "Replay all four packaged checkpoint streams with the structural "
                    "path disabled, then rebuild the common validation/test cache."
                ),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    sys.argv = [sys.argv[0], *remaining]
    return_code = int(ensemble.main())
    if output_dir_text:
        output_dir = Path(output_dir_text)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "ablation_protocol.json").write_text(
            json.dumps(
                {
                    "ablation_mode": known.ablation_mode,
                    "checkpoint_stream_count": 4,
                    "retrained": False,
                    "test_time_fitting": False,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
