#!/usr/bin/env python3
"""Run the controlled NYC GAP/V Full models and six structure ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"


@dataclass(frozen=True)
class RunSpec:
    key: str
    task: str
    paper_label: str
    role: str
    max_epochs: int
    args: list[str]


def local_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "calculating"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    index = result.index(option)
    result[index + 1] = value
    return result


def build_specs(
    repo: Path,
    python_exe: str,
    gap_epochs: int,
    v_epochs: int,
    patience: int,
    checkpoint_every: int,
) -> list[RunSpec]:
    data_dir = repo / "data" / "nyc_dc"
    gap_script = repo / "tools" / "chengdu_gap_arch_smoke.py"
    v_script = repo / "train_predictor.py"

    gap_common = [
        python_exe,
        "-u",
        str(gap_script),
        "--data-dir",
        str(data_dir),
        "--hist-len",
        "14",
        "--pred-horizon",
        "4",
        "--report-horizons-minutes",
        "15,30,60",
        "--max-train-samples",
        "0",
        "--max-val-samples",
        "0",
        "--max-test-samples",
        "0",
        "--epochs",
        str(gap_epochs),
        "--batch-size",
        "8",
        "--hidden-dim",
        "16",
        "--num-heads",
        "4",
        "--num-st-blocks",
        "2",
        "--num-gtcn-layers",
        "2",
        "--kernel-size",
        "3",
        "--adaptive-emb",
        "10",
        "--adaptive-topk",
        "16",
        "--lr",
        "0.001",
        "--score-mape-weight",
        "0.05",
        "--base-loss",
        "smooth_l1",
        "--smooth-l1-beta",
        "0.5",
        "--loss-mae-weight",
        "0.1",
        "--loss-mape-weight",
        "0.05",
        "--gap-target-transform",
        "signed_log",
        "--gap-log-scale",
        "10",
        "--loss-step-weights",
        "1,1,1,1.3",
        "--calibration-mode",
        "stepwise_shrink",
        "--calibration-mape-weight",
        "0.05",
        "--checkpoint-every",
        str(checkpoint_every),
        "--early-stopping-patience",
        str(patience),
        "--early-stopping-min-delta",
        "0",
        "--variants",
        "gap_v_to_gap",
        "--seed",
        "42",
        "--device",
        "cuda",
    ]
    v_common = [
        python_exe,
        "-u",
        str(v_script),
        "--data-dir",
        str(data_dir),
        "--edge-length-source",
        "osrm",
        "--hist-len",
        "12",
        "--pred-horizon",
        "4",
        "--report-horizons-minutes",
        "15,30,60",
        "--hidden-dim",
        "32",
        "--num-heads",
        "4",
        "--num-st-blocks",
        "2",
        "--adaptive-topk",
        "16",
        "--speed-use-adaptive",
        "--speed-adaptive-topk",
        "16",
        "--v-domain",
        "edge",
        "--num-gtcn-layers",
        "2",
        "--kernel-size",
        "3",
        "--epochs",
        str(v_epochs),
        "--batch-size",
        "32",
        "--lr",
        "0.001",
        "--lambda3",
        "1.0",
        "--seed",
        "42",
        "--device",
        "cuda",
        "--precision",
        "bf16",
        "--num-workers",
        "0",
        "--log-interval",
        "1",
        "--monitor-task",
        "v",
        "--train-task",
        "v",
        "--val-interval",
        "1",
        "--early-stopping-patience",
        str(patience),
        "--early-stopping-min-delta",
        "0",
    ]

    gap_single = replace_option(gap_common, "--num-st-blocks", "1")
    v_single = replace_option(v_common, "--num-st-blocks", "1")
    v_no_adaptive = [arg for arg in v_common if arg != "--speed-use-adaptive"]

    return [
        RunSpec(
            "gap_full_control",
            "GAP",
            "Full ST-GAPV (controlled)",
            "full",
            gap_epochs,
            gap_common,
        ),
        RunSpec(
            "gap_no_adaptive",
            "GAP",
            "Full - Adaptive Node-Relation Encoder",
            "ablation",
            gap_epochs,
            gap_common + ["--disable-node-adaptive"],
        ),
        RunSpec(
            "gap_single_st_block",
            "GAP",
            "Full - Second ST Block",
            "ablation",
            gap_epochs,
            gap_single,
        ),
        RunSpec(
            "gap_no_fixed_graph",
            "GAP",
            "Full - Fixed Physical-Graph Encoder",
            "ablation",
            gap_epochs,
            gap_common + ["--disable-node-fixed-graph"],
        ),
        RunSpec(
            "v_full_control",
            "V",
            "Full ST-GAPV (controlled)",
            "full",
            v_epochs,
            v_common,
        ),
        RunSpec(
            "v_no_adaptive",
            "V",
            "Full - Adaptive Edge-Relation Encoder",
            "ablation",
            v_epochs,
            v_no_adaptive,
        ),
        RunSpec(
            "v_single_st_block",
            "V",
            "Full - Second ST Block",
            "ablation",
            v_epochs,
            v_single,
        ),
        RunSpec(
            "v_no_fixed_graph",
            "V",
            "Full - Fixed Line-Graph Encoder",
            "ablation",
            v_epochs,
            v_common + ["--disable-speed-fixed-graph"],
        ),
    ]


def add_output_dir(spec: RunSpec, run_dir: Path) -> list[str]:
    option = "--output-dir" if spec.task == "GAP" else "--log-dir"
    return list(spec.args) + [option, str(run_dir)]


def add_warm_start(repo: Path, spec: RunSpec, command: list[str]) -> list[str]:
    checkpoint = repo / "resume_checkpoints" / spec.key / "checkpoint.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"No packaged continuation checkpoint for {spec.key}: {checkpoint}"
        )
    if spec.task == "GAP":
        return command + ["--resume-checkpoint", str(checkpoint)]
    return command + [
        "--init-checkpoint",
        str(checkpoint),
        "--init-load-mode",
        "strict",
    ]


def result_path(spec: RunSpec, run_dir: Path) -> Path:
    if spec.task == "GAP":
        return run_dir / "chengdu_gap_arch_smoke_results.json"
    return run_dir / "predictor_test_metrics.json"


def extract_epoch(spec: RunSpec, line: str) -> int | None:
    if spec.task == "GAP":
        match = re.search(r"\bepoch\s+(\d+)/(\d+)", line)
    else:
        match = re.search(r"\[Ep\s+(\d+)\]", line)
    return int(match.group(1)) if match else None


def extract_result(spec: RunSpec, run_dir: Path) -> dict[str, Any]:
    if spec.task == "GAP":
        payload = json.loads(result_path(spec, run_dir).read_text(encoding="utf-8"))
        result = payload["results"][0]
        metrics = result.get("test_gap_calibrated") or result["test_gap"]
        return {
            "best_epoch": result.get("best_epoch"),
            "completed_epochs": result.get("completed_epochs"),
            "stopped_early": result.get("stopped_early"),
            "best_checkpoint_path": result.get("best_checkpoint_path"),
            "last_checkpoint_path": result.get("last_checkpoint_path"),
            "metrics": metrics["report"],
        }

    payload = json.loads(result_path(spec, run_dir).read_text(encoding="utf-8"))
    report = payload["raw_metrics_report"]["speed"]
    per_step = payload.get("raw_metrics_per_step", {}).get("speed", {})
    for metrics in report.values():
        if metrics.get("mape") is None:
            step_metrics = per_step.get(f"step_{metrics.get('step')}", {})
            if step_metrics.get("mape") is not None:
                metrics["mape"] = step_metrics["mape"]
    return {
        "best_epoch": payload.get("best_monitor_epoch"),
        "completed_epochs": payload.get("completed_epochs"),
        "stopped_early": payload.get("stopped_early"),
        "best_checkpoint_path": str(run_dir / "stgat_best.pt"),
        "last_checkpoint_path": str(run_dir / "stgat_final.pt"),
        "metrics": report,
    }


def write_summary(root: Path, completed: list[dict[str, Any]]) -> None:
    full_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for item in completed:
        if item["role"] == "full":
            full_metrics[item["task"]] = item["result"]["metrics"]

    rows: list[dict[str, Any]] = []
    for item in completed:
        for horizon, metrics in item["result"]["metrics"].items():
            baseline = full_metrics.get(item["task"], {}).get(horizon, {})
            row = {
                "variant": item["key"],
                "task": item["task"],
                "role": item["role"],
                "paper_label": item["paper_label"],
                "horizon": horizon,
                "rmse": metrics.get("rmse"),
                "mae": metrics.get("mae"),
                "mape": metrics.get("mape"),
                "delta_rmse_vs_full": (
                    metrics.get("rmse") - baseline.get("rmse")
                    if metrics.get("rmse") is not None and baseline.get("rmse") is not None
                    else None
                ),
                "delta_mae_vs_full": (
                    metrics.get("mae") - baseline.get("mae")
                    if metrics.get("mae") is not None and baseline.get("mae") is not None
                    else None
                ),
                "delta_mape_vs_full": (
                    metrics.get("mape") - baseline.get("mape")
                    if metrics.get("mape") is not None and baseline.get("mape") is not None
                    else None
                ),
                "best_epoch": item["result"].get("best_epoch"),
                "completed_epochs": item["result"].get("completed_epochs"),
                "stopped_early": item["result"].get("stopped_early"),
            }
            rows.append(row)

    atomic_json(
        root / "controlled_ablation_results.json",
        {"generated_at": local_now(), "runs": completed, "rows": rows},
    )
    if rows:
        with (root / "controlled_ablation_results.csv").open(
            "w", newline="", encoding="utf-8-sig"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def selected_specs(specs: list[RunSpec], only: str) -> list[RunSpec]:
    if not only or only.lower() == "all":
        return specs
    requested = {item.strip() for item in only.split(",") if item.strip()}
    known = {spec.key for spec in specs}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown --only keys: {', '.join(unknown)}")
    return [spec for spec in specs if spec.key in requested]


def future_eta(
    remaining: list[RunSpec],
    seconds_per_epoch: dict[str, float],
) -> float | None:
    if not remaining:
        return 0.0
    estimates = []
    for spec in remaining:
        per_epoch = seconds_per_epoch.get(spec.task)
        if per_epoch is None:
            return None
        estimates.append(per_epoch * spec.max_epochs)
    return sum(estimates)


def stable_run_dir(root: Path, all_specs: list[RunSpec], spec: RunSpec) -> Path:
    index = next(i for i, item in enumerate(all_specs, 1) if item.key == spec.key)
    return root / f"{index:02d}_{spec.key}"


def load_existing_results(
    root: Path,
    all_specs: list[RunSpec],
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    for spec in all_specs:
        run_dir = stable_run_dir(root, all_specs, spec)
        if not result_path(spec, run_dir).exists():
            continue
        completed.append(
            {
                "key": spec.key,
                "task": spec.task,
                "role": spec.role,
                "paper_label": spec.paper_label,
                "run_dir": str(run_dir),
                "elapsed_seconds": 0.0,
                "skipped_existing": True,
                "result": extract_result(spec, run_dir),
            }
        )
    return completed


def upsert_completed(
    completed: list[dict[str, Any]],
    item: dict[str, Any],
    all_specs: list[RunSpec],
) -> None:
    completed[:] = [existing for existing in completed if existing["key"] != item["key"]]
    completed.append(item)
    order = {spec.key: index for index, spec in enumerate(all_specs)}
    completed.sort(key=lambda existing: order[existing["key"]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="")
    parser.add_argument("--gap-epochs", type=int, default=200)
    parser.add_argument("--v-epochs", type=int, default=610)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--only",
        default="all",
        help="Comma-separated run keys, or all.",
    )
    parser.add_argument(
        "--warm-start-from-bundle",
        action="store_true",
        help=(
            "Initialize selected runs from resume_checkpoints. "
            "Do not use for the clean controlled table."
        ),
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Rerun a job even when its final metrics already exist.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = (
        Path(args.output_root).resolve()
        if args.output_root
        else repo / "runs" / f"nyc_gap_v_controlled_ablation_{stamp}"
    )
    root.mkdir(parents=True, exist_ok=True)
    all_specs = build_specs(
        repo=repo,
        python_exe=args.python_exe,
        gap_epochs=args.gap_epochs,
        v_epochs=args.v_epochs,
        patience=args.patience,
        checkpoint_every=args.checkpoint_every,
    )
    specs = selected_specs(all_specs, args.only)
    status_path = root / "suite_status.json"
    console_path = root / "suite_console.log"
    runs_dir = repo / "runs"
    runs_dir.mkdir(exist_ok=True)
    (runs_dir / "latest_nyc_gap_v_controlled_ablation.txt").write_text(
        str(root), encoding="utf-8"
    )
    (root / "suite.pid").write_text(str(os.getpid()), encoding="ascii")

    manifest = {
        "experiment": "NYC GAP/V controlled Full plus six structure ablations",
        "started_at": local_now(),
        "python": args.python_exe,
        "gpu_required": True,
        "clean_controlled_run": not args.warm_start_from_bundle,
        "same_protocol_within_task": True,
        "gap_epochs": args.gap_epochs,
        "v_epochs": args.v_epochs,
        "early_stopping_patience": args.patience,
        "checkpoint_every_gap": args.checkpoint_every,
        "selected_runs": [asdict(spec) for spec in specs],
    }
    atomic_json(root / "suite_manifest.json", manifest)

    completed = load_existing_results(root, all_specs)
    suite_start = time.monotonic()
    seconds_per_epoch: dict[str, float] = {}

    with console_path.open("a", encoding="utf-8", buffering=1) as suite_log:
        for index, spec in enumerate(specs, 1):
            run_dir = stable_run_dir(root, all_specs, spec)
            run_dir.mkdir(parents=True, exist_ok=True)
            final_result = result_path(spec, run_dir)

            if final_result.exists() and not args.rerun_existing:
                result = extract_result(spec, run_dir)
                item = {
                    "key": spec.key,
                    "task": spec.task,
                    "role": spec.role,
                    "paper_label": spec.paper_label,
                    "run_dir": str(run_dir),
                    "elapsed_seconds": 0.0,
                    "skipped_existing": True,
                    "result": result,
                }
                upsert_completed(completed, item, all_specs)
                write_summary(root, completed)
                print(f"{YELLOW}[SKIP] Existing result: {spec.key}{RESET}", flush=True)
                continue

            command = add_output_dir(spec, run_dir)
            if args.warm_start_from_bundle:
                command = add_warm_start(repo, spec, command)
            (run_dir / "command.json").write_text(
                json.dumps(command, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            header = f"[{index}/{len(specs)}] {spec.task}: {spec.paper_label}"
            print(f"\n{BOLD}{CYAN}{header}{RESET}", flush=True)
            print(f"{CYAN}run_dir={run_dir}{RESET}", flush=True)
            suite_log.write(f"\n{local_now()} {header}\ncommand={json.dumps(command)}\n")

            run_start = time.monotonic()
            current_epoch = 0
            epoch_times: list[float] = []
            last_epoch_time = run_start
            status: dict[str, Any] = {
                "state": "running",
                "suite_started_at": manifest["started_at"],
                "updated_at": local_now(),
                "current_index": index,
                "total_runs": len(specs),
                "current_key": spec.key,
                "current_label": spec.paper_label,
                "current_task": spec.task,
                "current_epoch": 0,
                "max_epochs": spec.max_epochs,
                "completed": completed,
            }
            atomic_json(status_path, status)

            with (run_dir / "train.log").open(
                "w", encoding="utf-8", buffering=1
            ) as run_log:
                process = subprocess.Popen(
                    command,
                    cwd=repo,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"},
                )
                (run_dir / "train.pid").write_text(str(process.pid), encoding="ascii")
                assert process.stdout is not None
                for raw_line in process.stdout:
                    line = raw_line.rstrip("\r\n")
                    console_encoding = sys.stdout.encoding or "utf-8"
                    console_line = line.encode(
                        console_encoding, errors="replace"
                    ).decode(console_encoding, errors="replace")
                    print(console_line, flush=True)
                    run_log.write(raw_line)
                    suite_log.write(f"[{spec.key}] {line}\n")
                    parsed_epoch = extract_epoch(spec, line)
                    if parsed_epoch and parsed_epoch != current_epoch:
                        now = time.monotonic()
                        if current_epoch > 0:
                            epoch_times.append(now - last_epoch_time)
                            epoch_times = epoch_times[-10:]
                        current_epoch = parsed_epoch
                        last_epoch_time = now
                        avg_epoch = (
                            sum(epoch_times) / len(epoch_times)
                            if epoch_times
                            else seconds_per_epoch.get(spec.task)
                        )
                        current_eta = (
                            avg_epoch * max(0, spec.max_epochs - current_epoch)
                            if avg_epoch
                            else None
                        )
                        remaining_eta = future_eta(
                            specs[index:], seconds_per_epoch
                        )
                        total_eta = (
                            current_eta + remaining_eta
                            if current_eta is not None and remaining_eta is not None
                            else None
                        )
                        status.update(
                            {
                                "updated_at": local_now(),
                                "current_epoch": current_epoch,
                                "elapsed_seconds": now - suite_start,
                                "current_run_elapsed_seconds": now - run_start,
                                "current_run_eta_seconds": current_eta,
                                "total_eta_seconds": total_eta,
                            }
                        )
                        atomic_json(status_path, status)
                        print(
                            f"{YELLOW}[PROGRESS] {index}/{len(specs)} "
                            f"epoch {current_epoch}/{spec.max_epochs} "
                            f"elapsed={format_duration(now - run_start)} "
                            f"ETA={format_duration(current_eta)} "
                            f"TOTAL_ETA={format_duration(total_eta)}{RESET}",
                            flush=True,
                        )
                return_code = process.wait()

            elapsed = time.monotonic() - run_start
            if return_code != 0:
                status.update(
                    {
                        "state": "failed",
                        "updated_at": local_now(),
                        "return_code": return_code,
                        "failed_key": spec.key,
                    }
                )
                atomic_json(status_path, status)
                print(f"{RED}[FAILED] {header} exit={return_code}{RESET}", flush=True)
                return return_code

            result = extract_result(spec, run_dir)
            completed_epochs = result.get("completed_epochs") or current_epoch
            if completed_epochs:
                seconds_per_epoch[spec.task] = elapsed / float(completed_epochs)
            completed_item = {
                "key": spec.key,
                "task": spec.task,
                "role": spec.role,
                "paper_label": spec.paper_label,
                "run_dir": str(run_dir),
                "elapsed_seconds": elapsed,
                "skipped_existing": False,
                "result": result,
            }
            upsert_completed(completed, completed_item, all_specs)
            write_summary(root, completed)
            print(
                f"{GREEN}[DONE] {index}/{len(specs)} {spec.key} "
                f"epochs={completed_epochs} best={result.get('best_epoch')} "
                f"time={format_duration(elapsed)}{RESET}",
                flush=True,
            )

    final_status = {
        "state": "complete",
        "suite_started_at": manifest["started_at"],
        "completed_at": local_now(),
        "elapsed_seconds": time.monotonic() - suite_start,
        "total_runs": len(specs),
        "completed": completed,
    }
    atomic_json(status_path, final_status)
    write_summary(root, completed)
    print(f"\n{BOLD}{GREEN}Controlled suite complete: {root}{RESET}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
