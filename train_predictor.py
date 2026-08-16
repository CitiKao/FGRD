"""
train_predictor.py — STGAT 預測模型訓練腳本

訓練流程：
  1. 生成或載入時空資料
  2. 建構 SpatioTemporalDataset 與 DataLoader
  3. 初始化 STGATPredictor
  4. 多任務損失訓練：L = λ₁·L_demand + λ₂·L_supply + λ₃·L_speed
  5. 記錄各項指標並儲存最佳模型

使用方式（需先準備 data/ 真實路網與 build_speed_features 產物）：
    python train_predictor.py
    python train_predictor.py --epochs 100 --device auto --precision bf16
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data_loader import SpatioTemporalDataset, load_nyc_real_graph_features
from predictor_normalization import (
    build_normalization_stats,
    denormalize_count_values,
    denormalize_speed_values,
    normalize_node_features,
    normalize_speed_features,
    serialize_normalization_stats,
)
from stgat_model import STGATPredictor


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("指定了 CUDA，但目前環境沒有可用 GPU。")
    return device


def resolve_precision(device: torch.device, precision_arg: str) -> str:
    if device.type != "cuda":
        return "fp32"
    if precision_arg == "auto":
        return "bf16"
    return precision_arg


def resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return requested
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            available_cpus = max(int(slurm_cpus_per_task), 1)
        except ValueError:
            available_cpus = os.cpu_count() or 1
    else:
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 1
    if device.type == "cuda":
        return min(4, available_cpus)
    return 0


def load_init_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    mode: str,
) -> dict[str, object]:
    if mode == "strict":
        model.load_state_dict(state, strict=True)
        return {
            "mode": "strict",
            "loaded_keys": int(len(state)),
            "skipped_keys": 0,
            "missing_keys": [],
            "unexpected_keys": [],
        }

    target_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)
    }
    if not compatible:
        raise RuntimeError("No shape-compatible tensors found in init checkpoint.")
    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "mode": "compatible",
        "loaded_keys": int(len(compatible)),
        "skipped_keys": int(len(state) - len(compatible)),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }


def configure_cuda_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def load_time_meta_for_training(data_dir: str | Path, num_time_steps: int) -> pd.DataFrame:
    time_meta_path = Path(data_dir) / "time_meta.csv"
    if not time_meta_path.exists():
        raise FileNotFoundError(f"找不到 {time_meta_path}，請先執行 build_speed_features.py")
    time_meta = pd.read_csv(time_meta_path)
    if len(time_meta) < num_time_steps:
        raise ValueError(
            f"time_meta.csv 行數 {len(time_meta)} 少於目前資料時間步 {num_time_steps}"
        )
    time_meta = time_meta.iloc[:num_time_steps].copy()
    time_meta["date"] = pd.to_datetime(time_meta["date"], errors="coerce")
    if time_meta["date"].isna().any():
        raise ValueError("time_meta.csv 的 date 欄位有無法解析的值")
    return time_meta


def assign_calendar_split(time_meta: pd.DataFrame) -> pd.Series:
    day = time_meta["date"].dt.day
    return pd.Series(
        np.where(day <= 20, "train", np.where(day <= 24, "val", "test")),
        index=time_meta.index,
        dtype="object",
    )


CALENDAR_SPLIT_STRATEGY = (
    "per_month_day_1_20_train_21_24_val_25_plus_test_full_window_containment"
)
CALENDAR_SPLIT_DESCRIPTION = (
    "每月 1-20 train / 21-24 val / 25+ test，且整個 history+target window 必須完整落在同一 split"
)


def build_monthly_split_indices(
    time_meta: pd.DataFrame,
    hist_len: int,
    pred_horizon: int,
) -> dict[str, list[int]]:
    total = len(time_meta) - hist_len - pred_horizon + 1
    split_labels = assign_calendar_split(time_meta)
    splits = {"train": [], "val": [], "test": []}
    window_len = hist_len + pred_horizon

    for idx in range(max(total, 0)):
        window_labels = split_labels.iloc[idx: idx + window_len].unique()
        if len(window_labels) != 1:
            continue
        splits[str(window_labels[0])].append(idx)

    return splits


SPLIT_ALIGNMENT_MODES = ("none", "day", "week", "month")


def build_time_meta_timestamps(time_meta: pd.DataFrame) -> pd.Series:
    if "timestamp" in time_meta.columns:
        return pd.to_datetime(time_meta["timestamp"], errors="raise")
    if not {"date", "hour", "minute"}.issubset(time_meta.columns):
        raise ValueError("time_meta must contain timestamp or date/hour/minute columns")
    dates = pd.to_datetime(time_meta["date"], errors="raise")
    hours = pd.to_numeric(time_meta["hour"], errors="raise")
    minutes = pd.to_numeric(time_meta["minute"], errors="raise")
    return dates + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(minutes, unit="m")


def validate_split_alignment(mode: str) -> str:
    cleaned = str(mode).strip().lower()
    if cleaned not in SPLIT_ALIGNMENT_MODES:
        raise ValueError(
            f"Unsupported split alignment '{mode}'. Expected one of: {', '.join(SPLIT_ALIGNMENT_MODES)}."
        )
    return cleaned


def is_matching_split_boundary(timestamp: pd.Timestamp, mode: str) -> bool:
    if mode == "none":
        return True
    ts = pd.Timestamp(timestamp)
    is_day_start = bool(
        ts.hour == 0
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
        and ts.nanosecond == 0
    )
    if not is_day_start:
        return False
    if mode == "day":
        return True
    if mode == "week":
        return ts.weekday() == 0
    if mode == "month":
        return ts.day == 1
    raise ValueError(f"Unsupported split alignment '{mode}'.")


def align_split_boundary_index(
    timestamps: pd.Series,
    *,
    raw_index: int,
    mode: str,
    lower_bound: int,
    upper_bound: int,
) -> tuple[int, dict[str, object]]:
    cleaned_mode = validate_split_alignment(mode)
    clipped_index = int(min(max(raw_index, lower_bound), upper_bound))
    raw_ts = pd.Timestamp(timestamps.iloc[clipped_index])
    if cleaned_mode == "none":
        return clipped_index, {
            "mode": cleaned_mode,
            "raw_index": int(raw_index),
            "effective_index": clipped_index,
            "shift_steps": int(clipped_index - raw_index),
            "raw_timestamp": raw_ts.isoformat(),
            "effective_timestamp": raw_ts.isoformat(),
            "matched_boundary": True,
            "fallback_used": False,
        }

    candidates = [
        idx
        for idx in range(int(lower_bound), int(upper_bound) + 1)
        if is_matching_split_boundary(pd.Timestamp(timestamps.iloc[idx]), cleaned_mode)
    ]
    if not candidates:
        return clipped_index, {
            "mode": cleaned_mode,
            "raw_index": int(raw_index),
            "effective_index": clipped_index,
            "shift_steps": int(clipped_index - raw_index),
            "raw_timestamp": raw_ts.isoformat(),
            "effective_timestamp": raw_ts.isoformat(),
            "matched_boundary": False,
            "fallback_used": True,
        }

    effective_index = min(
        candidates,
        key=lambda idx: (abs(int(idx) - int(raw_index)), int(idx) < int(raw_index), int(idx)),
    )
    effective_ts = pd.Timestamp(timestamps.iloc[effective_index])
    return int(effective_index), {
        "mode": cleaned_mode,
        "raw_index": int(raw_index),
        "effective_index": int(effective_index),
        "shift_steps": int(effective_index - raw_index),
        "raw_timestamp": raw_ts.isoformat(),
        "effective_timestamp": effective_ts.isoformat(),
        "matched_boundary": True,
        "fallback_used": False,
    }


def build_split_boundary_summary(
    timestamps: pd.Series,
    *,
    alignment: str,
    train_end_raw: int,
    val_end_raw: int,
    train_end: int,
    val_end: int,
) -> dict[str, object]:
    ts = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)

    def summarize_segment(start_idx: int, end_idx: int) -> dict[str, object]:
        segment = ts.iloc[start_idx:end_idx]
        if segment.empty:
            return {
                "start": None,
                "end": None,
                "start_weekday": None,
                "end_weekday": None,
                "months": [],
                "weekday_counts": {},
            }
        return {
            "start": pd.Timestamp(segment.iloc[0]).isoformat(),
            "end": pd.Timestamp(segment.iloc[-1]).isoformat(),
            "start_weekday": str(pd.Timestamp(segment.iloc[0]).day_name()),
            "end_weekday": str(pd.Timestamp(segment.iloc[-1]).day_name()),
            "months": sorted(segment.dt.to_period("M").astype(str).unique().tolist()),
            "weekday_counts": {
                str(key): int(value)
                for key, value in segment.dt.day_name().value_counts().sort_index().items()
            },
        }

    return {
        "mode": "contiguous_ratio_70_10_20",
        "alignment": validate_split_alignment(alignment),
        "raw_boundaries": {
            "train_end_index": int(train_end_raw),
            "val_end_index": int(val_end_raw),
            "train_boundary_timestamp": pd.Timestamp(ts.iloc[min(max(train_end_raw, 0), len(ts) - 1)]).isoformat(),
            "val_boundary_timestamp": pd.Timestamp(ts.iloc[min(max(val_end_raw, 0), len(ts) - 1)]).isoformat(),
        },
        "effective_boundaries": {
            "train_end_index": int(train_end),
            "val_end_index": int(val_end),
            "train_last_timestamp": pd.Timestamp(ts.iloc[max(train_end - 1, 0)]).isoformat(),
            "val_first_timestamp": pd.Timestamp(ts.iloc[min(train_end, len(ts) - 1)]).isoformat(),
            "val_last_timestamp": pd.Timestamp(ts.iloc[max(val_end - 1, 0)]).isoformat(),
            "test_first_timestamp": pd.Timestamp(ts.iloc[min(val_end, len(ts) - 1)]).isoformat(),
        },
        "calendar": {
            "train": summarize_segment(0, train_end),
            "val": summarize_segment(train_end, val_end),
            "test": summarize_segment(val_end, len(ts)),
        },
    }


def resolve_split_boundaries(
    time_meta: pd.DataFrame,
    *,
    alignment: str,
) -> tuple[int, int, dict[str, object]]:
    timestamps = build_time_meta_timestamps(time_meta).reset_index(drop=True)
    num_time_steps = int(len(timestamps))
    if num_time_steps < 3:
        raise ValueError("Need at least 3 time steps to build contiguous splits")
    raw_train_end = round(num_time_steps * 0.7)
    raw_val_end = round(num_time_steps * 0.8)
    cleaned_alignment = validate_split_alignment(alignment)
    train_end, train_alignment = align_split_boundary_index(
        timestamps,
        raw_index=raw_train_end,
        mode=cleaned_alignment,
        lower_bound=1,
        upper_bound=max(num_time_steps - 2, 1),
    )
    val_end, val_alignment = align_split_boundary_index(
        timestamps,
        raw_index=raw_val_end,
        mode=cleaned_alignment,
        lower_bound=min(max(train_end + 1, 1), max(num_time_steps - 1, 1)),
        upper_bound=max(num_time_steps - 1, 1),
    )
    if val_end <= train_end:
        val_end = min(max(train_end + 1, 1), max(num_time_steps - 1, 1))
        val_alignment = {
            **val_alignment,
            "effective_index": int(val_end),
            "effective_timestamp": pd.Timestamp(timestamps.iloc[val_end]).isoformat(),
            "shift_steps": int(val_end - raw_val_end),
            "fallback_used": True,
            "matched_boundary": False,
        }

    summary = build_split_boundary_summary(
        timestamps,
        alignment=cleaned_alignment,
        train_end_raw=raw_train_end,
        val_end_raw=raw_val_end,
        train_end=train_end,
        val_end=val_end,
    )
    summary["alignment_details"] = {
        "train_boundary": train_alignment,
        "val_boundary": val_alignment,
    }
    return int(train_end), int(val_end), summary


def build_time_contained_split_indices(
    num_time_steps: int,
    *,
    hist_len: int,
    pred_horizon: int,
    train_end: int | None = None,
    val_end: int | None = None,
) -> dict[str, list[int]]:
    if train_end is None:
        train_end = round(num_time_steps * 0.7)
    if val_end is None:
        val_end = round(num_time_steps * 0.8)
    total_samples = num_time_steps - hist_len - pred_horizon + 1
    window_len = hist_len + pred_horizon
    splits = {"train": [], "val": [], "test": []}
    for idx in range(max(total_samples, 0)):
        end = idx + window_len
        if end <= train_end:
            splits["train"].append(idx)
        elif idx >= train_end and end <= val_end:
            splits["val"].append(idx)
        elif idx >= val_end and end <= num_time_steps:
            splits["test"].append(idx)
    return splits


def build_window_time_mask(
    num_time_steps: int,
    sample_indices: list[int],
    hist_len: int,
    pred_horizon: int,
) -> np.ndarray:
    mask = np.zeros(num_time_steps, dtype=bool)
    window_len = hist_len + pred_horizon
    for idx in sample_indices:
        start = int(idx)
        end = min(start + window_len, num_time_steps)
        if start < end:
            mask[start:end] = True
    return mask


OBSERVED_TIME_MASK_FILENAMES = ("observed_time_mask.npy", "valid_time_mask.npy")


def load_observed_time_mask(
    data_dir: str | Path,
    num_time_steps: int,
) -> np.ndarray | None:
    """Load optional observed-slot mask for datasets with missing raw days."""
    root = Path(data_dir)
    for filename in OBSERVED_TIME_MASK_FILENAMES:
        path = root / filename
        if not path.exists():
            continue
        mask = np.load(path).astype(bool)
        if mask.ndim != 1:
            raise ValueError(f"{path} must be a 1-D boolean mask, got shape {mask.shape}")
        if mask.shape[0] < num_time_steps:
            raise ValueError(
                f"{path} length {mask.shape[0]} is shorter than requested time steps {num_time_steps}"
            )
        return mask[:num_time_steps].copy()
    return None


def filter_split_indices_by_time_mask(
    splits: dict[str, list[int]],
    observed_time_mask: np.ndarray | None,
    hist_len: int,
    pred_horizon: int,
) -> dict[str, list[int]]:
    """Drop any sample whose history or target touches an unobserved slot."""
    if observed_time_mask is None:
        return {name: list(indices) for name, indices in splits.items()}
    mask = np.asarray(observed_time_mask, dtype=bool)
    window_len = int(hist_len) + int(pred_horizon)
    filtered: dict[str, list[int]] = {}
    for name, indices in splits.items():
        kept: list[int] = []
        for raw_idx in indices:
            idx = int(raw_idx)
            end = idx + window_len
            if idx < 0 or end > mask.shape[0]:
                continue
            if bool(mask[idx:end].all()):
                kept.append(idx)
        filtered[name] = kept
    return filtered


def init_loss_dict() -> dict[str, float]:
    return {"dc": 0.0, "v": 0.0, "demand": 0.0, "supply": 0.0, "speed": 0.0}


def skipped_loss_dict() -> dict[str, None]:
    return {"dc": None, "v": None, "demand": None, "supply": None, "speed": None}


def skipped_raw_metric_dict() -> dict[str, float | dict[str, float] | None]:
    return {
        "raw_dc": None,
        "demand": None,
        "supply": None,
        "speed": None,
        "report_horizons": None,
    }


def build_task_losses(
    loss_d: torch.Tensor,
    loss_c: torch.Tensor,
    loss_v: torch.Tensor,
    *,
    lam1: float,
    lam2: float,
    lam3: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    dc_loss = lam1 * loss_d + lam2 * loss_c
    v_loss = lam3 * loss_v
    return dc_loss, v_loss


def optimized_tasks_for_mode(train_task: str) -> list[str]:
    if train_task == "dc":
        return ["dc"]
    if train_task == "v":
        return ["v"]
    return ["dc", "v"]


def build_raw_dc_metric(raw_metrics: dict[str, dict[str, float]]) -> float:
    return float(raw_metrics["demand"]["rmse"] + raw_metrics["supply"]["rmse"])


DEFAULT_REPORT_PRED_HORIZON = 4
DEFAULT_NON_REPORT_PRED_HORIZON = 3
DEFAULT_REPORT_HORIZONS_MINUTES = (15, 30, 60)


def parse_report_horizons_minutes(raw_value: str) -> list[int]:
    value = raw_value.strip()
    if not value:
        return []

    report_minutes: list[int] = []
    seen: set[int] = set()
    for chunk in value.split(","):
        minute = int(chunk.strip())
        if minute <= 0:
            raise ValueError("report horizon minutes must be positive integers.")
        if minute in seen:
            continue
        seen.add(minute)
        report_minutes.append(minute)
    return report_minutes


def infer_time_slot_minutes(time_meta: pd.DataFrame) -> int:
    if time_meta.empty:
        raise ValueError("time_meta is empty; cannot infer slot length.")
    timestamps = (
        time_meta["date"]
        + pd.to_timedelta(time_meta["hour"], unit="h")
        + pd.to_timedelta(time_meta["minute"], unit="m")
    )
    diff_minutes = timestamps.diff().dt.total_seconds().dropna() / 60.0
    positive_diffs = diff_minutes[diff_minutes > 0]
    if positive_diffs.empty:
        raise ValueError("Unable to infer slot length from time_meta timestamps.")
    slot_minutes = int(round(float(positive_diffs.mode().iloc[0])))
    if slot_minutes <= 0:
        raise ValueError(f"Inferred invalid slot length: {slot_minutes} minutes.")
    return slot_minutes


def resolve_report_horizons(
    *,
    time_slot_minutes: int,
    pred_horizon: int,
    requested_minutes: list[int],
    strict: bool = True,
) -> dict[str, int | list[int]]:
    resolved_steps: list[int] = []
    resolved_minutes: list[int] = []
    missing_minutes: list[int] = []

    for minute in requested_minutes:
        if minute % time_slot_minutes != 0:
            raise ValueError(
                f"Requested report horizon {minute} min is not divisible by "
                f"time_slot_minutes={time_slot_minutes}."
            )
        step = minute // time_slot_minutes
        if step < 1 or step > pred_horizon:
            missing_minutes.append(minute)
            continue
        resolved_steps.append(int(step))
        resolved_minutes.append(int(minute))

    if strict and missing_minutes:
        missing = ", ".join(str(v) for v in missing_minutes)
        raise ValueError(
            "Requested report horizons exceed the available prediction horizon: "
            f"missing [{missing}] min for pred_horizon={pred_horizon} "
            f"with time_slot_minutes={time_slot_minutes}."
        )

    return {
        "slot_minutes": int(time_slot_minutes),
        "pred_horizon": int(pred_horizon),
        "requested_minutes": [int(v) for v in requested_minutes],
        "resolved_minutes": resolved_minutes,
        "resolved_steps": resolved_steps,
        "missing_minutes": missing_minutes,
    }


def init_raw_metric_bucket() -> dict[str, float]:
    return {"se": 0.0, "ae": 0.0, "count": 0.0, "ape": 0.0, "mape_count": 0.0}


def finalize_raw_metric_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(bucket["count"], 1.0)
    mse_value = bucket["se"] / count
    mae_value = bucket["ae"] / count
    return {
        "mse": float(mse_value),
        "rmse": float(np.sqrt(mse_value)),
        "mae": float(mae_value),
        **(
            {"mape": float(bucket["ape"] / bucket["mape_count"])}
            if bucket.get("mape_count", 0.0) > 0
            else {}
        ),
    }


def summarize_report_metrics(
    task_metrics: dict[str, float | dict[str, dict[str, float]] | None] | None,
    *,
    metric_name: str = "rmse",
) -> str:
    if task_metrics is None:
        return ""
    report_metrics = task_metrics.get("report") if isinstance(task_metrics, dict) else None
    if not isinstance(report_metrics, dict):
        return ""

    parts: list[str] = []
    for label, values in report_metrics.items():
        if not isinstance(values, dict) or metric_name not in values:
            continue
        parts.append(f"{label}:{values[metric_name]:.3f}")
    return " ".join(parts)


def build_history_record(
    *,
    epoch: int,
    train_losses: dict[str, float],
    val_losses: dict[str, float | None],
    val_raw_metrics: dict[str, float | dict[str, float] | None],
    lr: float,
    elapsed: float,
    train_task: str,
) -> dict:
    record = {
        "epoch": epoch,
        "lr": lr,
        "elapsed": round(elapsed, 1),
    }
    if train_task != "v":
        record.update(
            {
                "train_dc": round(train_losses["dc"], 5),
                "train_demand": round(train_losses["demand"], 5),
                "train_supply": round(train_losses["supply"], 5),
                "val_dc": round(val_losses["dc"], 5) if val_losses["dc"] is not None else None,
                "val_demand": round(val_losses["demand"], 5) if val_losses["demand"] is not None else None,
                "val_supply": round(val_losses["supply"], 5) if val_losses["supply"] is not None else None,
                "val_raw_dc": round(val_raw_metrics["raw_dc"], 5) if val_raw_metrics["raw_dc"] is not None else None,
                "val_raw_demand_rmse": (
                    round(val_raw_metrics["demand"]["rmse"], 5)
                    if val_raw_metrics["demand"] is not None
                    else None
                ),
                "val_raw_supply_rmse": (
                    round(val_raw_metrics["supply"]["rmse"], 5)
                    if val_raw_metrics["supply"] is not None
                    else None
                ),
            }
        )
        if val_raw_metrics["demand"] is not None:
            demand_report = val_raw_metrics["demand"].get("report", {})
            if isinstance(demand_report, dict):
                for label, values in demand_report.items():
                    if isinstance(values, dict) and "rmse" in values:
                        record[f"val_raw_demand_rmse_{label.lower()}"] = round(values["rmse"], 5)
        if val_raw_metrics["supply"] is not None:
            supply_report = val_raw_metrics["supply"].get("report", {})
            if isinstance(supply_report, dict):
                for label, values in supply_report.items():
                    if isinstance(values, dict) and "rmse" in values:
                        record[f"val_raw_supply_rmse_{label.lower()}"] = round(values["rmse"], 5)
    if train_task != "dc":
        record.update(
            {
                "train_v": round(train_losses["v"], 5),
                "train_speed": round(train_losses["speed"], 5),
                "val_v": round(val_losses["v"], 5) if val_losses["v"] is not None else None,
                "val_speed": round(val_losses["speed"], 5) if val_losses["speed"] is not None else None,
                "val_raw_speed_rmse": (
                    round(val_raw_metrics["speed"]["rmse"], 5)
                    if val_raw_metrics["speed"] is not None
                    else None
                ),
            }
        )
    return record


def format_banner(*, lam1: float, lam2: float, lam3: float, args: argparse.Namespace, val_interval: int) -> str:
    if args.train_task == "v":
        return (
            f"\n開始訓練 | Epochs={args.epochs} | "
            f"v=({lam3}*V) | val_interval={val_interval} | "
            f"monitor={args.monitor_task} | train_task={args.train_task}"
        )
    if args.train_task == "dc":
        return (
            f"\n開始訓練 | Epochs={args.epochs} | "
            f"dc=({lam1}*D + {lam2}*C) | val_interval={val_interval} | "
            f"monitor={args.monitor_task} | train_task={args.train_task}"
        )
    return (
        f"\n開始訓練 | Epochs={args.epochs} | "
        f"dc=({lam1}*D + {lam2}*C) | v=({lam3}*V) | "
        f"val_interval={val_interval} | monitor={args.monitor_task} | "
        f"train_task={args.train_task}"
    )


def format_val_message(
    *,
    val_losses: dict[str, float | None],
    val_raw_metrics: dict[str, float | dict[str, float] | None],
    train_task: str,
) -> str:
    if train_task == "v":
        if val_losses["v"] is None:
            return "Val=skip"
        if val_raw_metrics["speed"] is None:
            return f"ValV={val_losses['v']:.4f}"
        report_summary = summarize_report_metrics(val_raw_metrics["speed"])
        suffix = f" ValRawV={val_raw_metrics['speed']['rmse']:.3f}"
        if report_summary:
            suffix += f" ({report_summary})"
        return f"ValV={val_losses['v']:.4f}{suffix}"
    if val_losses["dc"] is None:
        return "Val=skip"
    if val_raw_metrics["raw_dc"] is not None:
        message = (
            f"ValDC={val_losses['dc']:.4f} ValV={val_losses['v']:.4f} "
            f"ValRawDC={val_raw_metrics['raw_dc']:.3f} "
            f"(D:{val_raw_metrics['demand']['rmse']:.3f} C:{val_raw_metrics['supply']['rmse']:.3f})"
        )
        demand_report = summarize_report_metrics(val_raw_metrics["demand"])
        supply_report = summarize_report_metrics(val_raw_metrics["supply"])
        if demand_report:
            message += f" D[{demand_report}]"
        if supply_report:
            message += f" C[{supply_report}]"
        return message
    return f"ValDC={val_losses['dc']:.4f} ValV={val_losses['v']:.4f}"


def filter_normalized_losses(losses: dict[str, float | None], train_task: str) -> dict[str, float | None]:
    if train_task == "v":
        return {
            "v": losses["v"],
            "speed": losses["speed"],
        }
    if train_task == "dc":
        return {
            "dc": losses["dc"],
            "demand": losses["demand"],
            "supply": losses["supply"],
        }
    return losses


def _extract_overall_task_metrics(task_metrics: dict[str, float | dict[str, dict[str, float]]]) -> dict[str, float]:
    extracted = {
        "mse": float(task_metrics["mse"]),
        "rmse": float(task_metrics["rmse"]),
        "mae": float(task_metrics["mae"]),
    }
    for optional_key in ("mape", "count", "mape_count"):
        if optional_key in task_metrics and task_metrics[optional_key] is not None:
            extracted[optional_key] = float(task_metrics[optional_key])
    return extracted


def filter_raw_metrics(
    metrics: dict[str, dict[str, float | dict[str, dict[str, float]]]],
    train_task: str,
) -> dict[str, dict[str, float]]:
    if train_task == "v":
        return {"speed": _extract_overall_task_metrics(metrics["speed"])}
    if train_task == "dc":
        filtered = {
            "demand": _extract_overall_task_metrics(metrics["demand"]),
            "supply": _extract_overall_task_metrics(metrics["supply"]),
        }
        if "gap" in metrics:
            filtered["gap"] = _extract_overall_task_metrics(metrics["gap"])
        return filtered
    return {
        "demand": _extract_overall_task_metrics(metrics["demand"]),
        "supply": _extract_overall_task_metrics(metrics["supply"]),
        "speed": _extract_overall_task_metrics(metrics["speed"]),
    }


def filter_raw_metrics_per_step(
    metrics: dict[str, dict[str, float | dict[str, dict[str, float]]]],
    train_task: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if train_task == "v":
        return {"speed": dict(metrics["speed"].get("per_step", {}))}
    if train_task == "dc":
        filtered = {
            "demand": dict(metrics["demand"].get("per_step", {})),
            "supply": dict(metrics["supply"].get("per_step", {})),
        }
        if "gap" in metrics:
            filtered["gap"] = dict(metrics["gap"].get("per_step", {}))
        return filtered
    return {
        "demand": dict(metrics["demand"].get("per_step", {})),
        "supply": dict(metrics["supply"].get("per_step", {})),
        "speed": dict(metrics["speed"].get("per_step", {})),
    }


def filter_raw_metrics_report(
    metrics: dict[str, dict[str, float | dict[str, dict[str, float]]]],
    train_task: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if train_task == "v":
        return {"speed": dict(metrics["speed"].get("report", {}))}
    if train_task == "dc":
        filtered = {
            "demand": dict(metrics["demand"].get("report", {})),
            "supply": dict(metrics["supply"].get("report", {})),
        }
        if "gap" in metrics:
            filtered["gap"] = dict(metrics["gap"].get("report", {}))
        return filtered
    return {
        "demand": dict(metrics["demand"].get("report", {})),
        "supply": dict(metrics["supply"].get("report", {})),
        "speed": dict(metrics["speed"].get("report", {})),
    }


def extract_report_horizons(
    metrics: dict[str, object],
) -> dict[str, int | list[int]] | None:
    report_horizons = metrics.get("report_horizons")
    return dict(report_horizons) if isinstance(report_horizons, dict) else None


def extract_temporal_context(node_seq: torch.Tensor) -> torch.Tensor | None:
    if node_seq.shape[-1] <= 2:
        return None
    return node_seq[:, 0, :, 2:]


def forward_for_task(
    model: nn.Module,
    node_seq: torch.Tensor,
    speed_seq: torch.Tensor,
    train_task: str,
    speed_history_mask_seq: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    if train_task == "v":
        v_pred = model.forward_v(
            speed_seq,
            extract_temporal_context(node_seq),
            speed_history_mask_seq=speed_history_mask_seq,
        )
        return None, None, v_pred
    d_pred, c_pred, v_pred = model(node_seq, speed_seq)
    return d_pred, c_pred, v_pred


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    train_task: str,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    mse: nn.Module,
    lam1: float,
    lam2: float,
    lam3: float,
) -> dict[str, float]:
    losses = init_loss_dict()
    n_batches = 0

    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = forward_for_task(model, node_seq, speed_seq, train_task)

                loss_v = mse(v_pred, v_tgt)
                if train_task == "v":
                    zero = loss_v.new_zeros(())
                    loss_d = zero
                    loss_c = zero
                else:
                    loss_d = mse(d_pred, d_tgt)
                    loss_c = mse(c_pred, c_tgt)
                loss_dc, loss_task_v = build_task_losses(
                    loss_d,
                    loss_c,
                    loss_v,
                    lam1=lam1,
                    lam2=lam2,
                    lam3=lam3,
                )

            losses["dc"] += loss_dc.item()
            losses["v"] += loss_task_v.item()
            losses["demand"] += loss_d.item()
            losses["supply"] += loss_c.item()
            losses["speed"] += loss_v.item()
            n_batches += 1

    for key in losses:
        losses[key] /= max(n_batches, 1)
    return losses


def evaluate_loader_raw_metrics(
    model: nn.Module,
    loader: DataLoader,
    *,
    train_task: str,
    device: torch.device,
    non_blocking: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype | None,
    normalization_stats: dict | None,
    time_slot_minutes: int,
    report_horizons: dict[str, int | list[int]],
    speed_metric_mask_zeros: bool = False,
) -> dict[str, object]:
    pred_horizon = int(report_horizons["pred_horizon"])
    accum = {
        "demand": {
            "overall": init_raw_metric_bucket(),
            "per_step": [init_raw_metric_bucket() for _ in range(pred_horizon)],
        },
        "supply": {
            "overall": init_raw_metric_bucket(),
            "per_step": [init_raw_metric_bucket() for _ in range(pred_horizon)],
        },
        "speed": {
            "overall": init_raw_metric_bucket(),
            "per_step": [init_raw_metric_bucket() for _ in range(pred_horizon)],
        },
    }

    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)
            speed_history_mask = batch.get("speed_history_mask")
            if speed_history_mask is not None:
                speed_history_mask = speed_history_mask.to(device, non_blocking=non_blocking)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)
            speed_target_mask = batch.get("speed_target_mask")

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = forward_for_task(
                    model,
                    node_seq,
                    speed_seq,
                    train_task,
                    speed_history_mask_seq=speed_history_mask,
                )

            v_pred_raw = denormalize_speed_values(
                v_pred.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )
            v_tgt_raw = denormalize_speed_values(
                v_tgt.detach().float().cpu().numpy(),
                normalization_stats,
                edge_axis=1,
            )

            speed_mask_np = None
            if speed_target_mask is not None:
                speed_mask_np = speed_target_mask.detach().cpu().numpy().astype(bool)
            if speed_metric_mask_zeros:
                nonzero_mask = np.abs(v_tgt_raw) > 1e-6
                speed_mask_np = nonzero_mask if speed_mask_np is None else (speed_mask_np & nonzero_mask)

            comparisons = [("speed", v_pred_raw, v_tgt_raw, speed_mask_np)]
            if train_task != "v":
                d_pred_raw = denormalize_count_values(
                    d_pred.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="demand",
                )
                d_tgt_raw = denormalize_count_values(
                    d_tgt.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="demand",
                )
                c_pred_raw = denormalize_count_values(
                    c_pred.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="supply",
                )
                c_tgt_raw = denormalize_count_values(
                    c_tgt.detach().float().cpu().numpy(),
                    normalization_stats,
                    task="supply",
                )
                comparisons = [
                    ("demand", d_pred_raw, d_tgt_raw, None),
                    ("supply", c_pred_raw, c_tgt_raw, None),
                    ("speed", v_pred_raw, v_tgt_raw, speed_mask_np),
                ]

            for name, pred_raw, tgt_raw, metric_mask in comparisons:
                diff = pred_raw.astype(np.float64) - tgt_raw.astype(np.float64)
                valid_metric_mask = (
                    np.ones(diff.shape, dtype=bool)
                    if metric_mask is None
                    else np.asarray(metric_mask, dtype=bool)
                )
                sq = np.square(diff)
                abs_diff = np.abs(diff)
                overall_sq = sq[valid_metric_mask]
                overall_abs = abs_diff[valid_metric_mask]
                accum[name]["overall"]["se"] += float(overall_sq.sum())
                accum[name]["overall"]["ae"] += float(overall_abs.sum())
                accum[name]["overall"]["count"] += float(overall_sq.size)
                mape_mask = valid_metric_mask & (np.abs(tgt_raw) > 1e-6)
                if np.any(mape_mask):
                    accum[name]["overall"]["ape"] += float(
                        (np.abs(diff[mape_mask]) / np.abs(tgt_raw[mape_mask]) * 100.0).sum()
                    )
                    accum[name]["overall"]["mape_count"] += float(mape_mask.sum())

                for step_idx in range(diff.shape[-1]):
                    step_sq = sq[..., step_idx]
                    step_abs = abs_diff[..., step_idx]
                    step_mask = valid_metric_mask[..., step_idx]
                    masked_step_sq = step_sq[step_mask]
                    masked_step_abs = step_abs[step_mask]
                    accum[name]["per_step"][step_idx]["se"] += float(masked_step_sq.sum())
                    accum[name]["per_step"][step_idx]["ae"] += float(masked_step_abs.sum())
                    accum[name]["per_step"][step_idx]["count"] += float(masked_step_sq.size)
                    step_tgt = tgt_raw[..., step_idx]
                    step_diff = diff[..., step_idx]
                    step_mape_mask = step_mask & (np.abs(step_tgt) > 1e-6)
                    if np.any(step_mape_mask):
                        accum[name]["per_step"][step_idx]["ape"] += float(
                            (
                                np.abs(step_diff[step_mape_mask])
                                / np.abs(step_tgt[step_mape_mask])
                                * 100.0
                            ).sum()
                        )
                        accum[name]["per_step"][step_idx]["mape_count"] += float(step_mape_mask.sum())

    metrics: dict[str, object] = {
        "report_horizons": dict(report_horizons),
    }
    for name, values in accum.items():
        overall = finalize_raw_metric_bucket(values["overall"])
        per_step: dict[str, dict[str, float]] = {}
        for step_idx, bucket in enumerate(values["per_step"], start=1):
            per_step[f"step_{step_idx}"] = {
                "step": int(step_idx),
                "minutes": int(step_idx * time_slot_minutes),
                **finalize_raw_metric_bucket(bucket),
            }

        report_metrics: dict[str, dict[str, float]] = {}
        resolved_steps = report_horizons.get("resolved_steps", [])
        resolved_minutes = report_horizons.get("resolved_minutes", [])
        if isinstance(resolved_steps, list) and isinstance(resolved_minutes, list):
            for minute, step in zip(resolved_minutes, resolved_steps):
                step_key = f"step_{int(step)}"
                step_metrics = per_step[step_key]
                report_metrics[f"{int(minute)}min"] = {
                    "step": int(step),
                    "minutes": int(minute),
                    "mse": float(step_metrics["mse"]),
                    "rmse": float(step_metrics["rmse"]),
                    "mae": float(step_metrics["mae"]),
                    **(
                        {"mape": float(step_metrics["mape"])}
                        if "mape" in step_metrics
                        else {}
                    ),
                }

        metrics[name] = {
            **overall,
            "per_step": per_step,
            "report": report_metrics,
        }
    return metrics


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.train_task == "dc" and args.monitor_task not in {"dc", "raw_dc"}:
        raise SystemExit("--train-task dc only supports --monitor-task dc/raw_dc.")
    if args.train_task == "v" and args.monitor_task != "v":
        raise SystemExit("--train-task v only supports --monitor-task v.")
    if args.pred_horizon < 1:
        raise SystemExit("--pred-horizon must be >= 1.")
    device = resolve_device(args.device)
    configure_cuda_runtime(device)
    precision = resolve_precision(device, args.precision)
    amp_enabled = device.type == "cuda" and precision == "bf16"
    amp_dtype = torch.bfloat16 if amp_enabled else None
    num_workers = resolve_num_workers(args.num_workers, device)
    pin_memory = device.type == "cuda"
    non_blocking = pin_memory

    # ── 資料準備 ──
    print("載入紐約真實路網與 build_speed_features 特徵 ...")
    data = load_nyc_real_graph_features(
        args.data_dir,
        max_time_steps=args.max_time_steps,
        edge_length_source=args.edge_length_source,
        add_time_features=not args.disable_time_features,
    )

    adj = data["adj"]
    edge_index = data["edge_index"]
    edge_lengths = data["edge_lengths"]
    node_feat = data["node_features"]
    edge_speeds = data["edge_speeds"]
    time_feature_names = data.get("time_feature_names", [])

    N = adj.shape[0]
    nE = edge_index.shape[0]
    t_steps = int(node_feat.shape[0])
    node_feat_dim = int(node_feat.shape[-1])
    print(f"  節點={N}, 邊={nE}, 時間步={t_steps}, node_feat_dim={node_feat_dim}")
    if time_feature_names:
        print(f"  時間特徵: {', '.join(time_feature_names)}")

    time_meta = load_time_meta_for_training(args.data_dir, t_steps)
    time_slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=time_slot_minutes,
        pred_horizon=args.pred_horizon,
        requested_minutes=args.report_horizons_minutes,
        strict=bool(args.report_horizons_minutes),
    )
    if report_horizons["resolved_minutes"]:
        print(
            "  report_horizons="
            f"{report_horizons['resolved_minutes']} min "
            f"(steps={report_horizons['resolved_steps']}, slot={time_slot_minutes} min)"
        )
    split_indices = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    observed_time_mask = load_observed_time_mask(args.data_dir, t_steps)
    if observed_time_mask is not None:
        raw_split_counts = {name: len(indices) for name, indices in split_indices.items()}
        split_indices = filter_split_indices_by_time_mask(
            split_indices,
            observed_time_mask,
            args.hist_len,
            args.pred_horizon,
        )
        filtered_split_counts = {name: len(indices) for name, indices in split_indices.items()}
        removed_split_counts = {
            name: raw_split_counts[name] - filtered_split_counts[name]
            for name in raw_split_counts
        }
        print(
            "  觀測時間遮罩: "
            f"observed={int(observed_time_mask.sum())}/{len(observed_time_mask)} slots；"
            f"已剔除碰到缺失日的 window={removed_split_counts}"
        )
    train_time_mask = build_window_time_mask(
        t_steps,
        split_indices["train"],
        args.hist_len,
        args.pred_horizon,
    )
    if not np.any(train_time_mask):
        raise ValueError("Train split is empty after applying observed_time_mask.")
    normalization_stats = build_normalization_stats(node_feat, edge_speeds, train_time_mask)
    node_feat = normalize_node_features(node_feat, normalization_stats)
    edge_speeds = normalize_speed_features(edge_speeds, normalization_stats, edge_axis=1)
    print("  正規化: D/C=log1p+zscore, V=per-edge zscore (僅用 train split 統計量)")

    # Dataset
    full_ds = SpatioTemporalDataset(
        node_feat, edge_speeds,
        hist_len=args.hist_len,
        pred_horizon=args.pred_horizon,
    )
    train_ds = Subset(full_ds, split_indices["train"])
    val_ds = Subset(full_ds, split_indices["val"])
    test_ds = Subset(full_ds, split_indices["test"])

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print(f"  切分方式: {CALENDAR_SPLIT_DESCRIPTION}")
    print(
        f"  訓練集={len(train_ds)}, 驗證集={len(val_ds)}, 測試集={len(test_ds)}"
    )
    print(
        f"  裝置={device} | precision={precision} | "
        f"num_workers={num_workers} | pin_memory={pin_memory}"
    )

    # ── 模型 ──
    model = STGATPredictor(
        num_nodes=N,
        edge_index=torch.from_numpy(edge_index),
        edge_lengths=torch.from_numpy(edge_lengths),
        adj_matrix=torch.from_numpy(adj),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_st_blocks=args.num_st_blocks,
        num_gtcn_layers=args.num_gtcn_layers,
        kernel_size=args.kernel_size,
        pred_horizon=args.pred_horizon,
        node_feat_dim=node_feat_dim,
        adaptive_topk=args.adaptive_topk,
        node_use_fixed_graph=not args.disable_node_fixed_graph,
        node_use_adaptive=not args.disable_node_adaptive,
        speed_adaptive_topk=args.speed_adaptive_topk,
        speed_use_adaptive=args.speed_use_adaptive,
        speed_use_fixed_graph=not args.disable_speed_fixed_graph,
        use_fixed_edge_length_feature=args.use_fixed_edge_length_feature,
        v_domain=args.v_domain,
    ).to(device)
    init_load_summary = None
    if args.init_checkpoint:
        init_checkpoint = Path(args.init_checkpoint)
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_checkpoint}")
        init_state = torch.load(init_checkpoint, map_location=device, weights_only=True)
        init_load_summary = load_init_state(model, init_state, mode=args.init_load_mode)
        print(f"Loaded init checkpoint: {init_checkpoint}")
        print(
            "  init_load="
            f"{init_load_summary['mode']} "
            f"loaded={init_load_summary['loaded_keys']} "
            f"skipped={init_load_summary['skipped_keys']}"
        )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型參數量: {total_params:,}")

    if args.compile:
        if hasattr(torch, "compile"):
            print(f"  啟用 torch.compile | mode={args.compile_mode}")
            try:
                model = torch.compile(model, mode=args.compile_mode)
            except Exception as exc:
                print(f"  torch.compile 啟用失敗，改用 eager mode: {exc}")
        else:
            print("  目前 torch 版本不支援 torch.compile，將改用 eager mode")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    mse = nn.MSELoss()

    # ── 訓練 ──
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val_losses = {"dc": float("inf"), "v": float("inf")}
    best_monitor_values = {"dc": float("inf"), "v": float("inf"), "raw_dc": float("inf")}
    selected_val_losses = skipped_loss_dict()
    selected_val_raw_metrics = skipped_raw_metric_dict()
    optimize_dc = args.train_task != "v"
    optimize_v = args.train_task != "dc"
    optimized_tasks = optimized_tasks_for_mode(args.train_task)

    lam1, lam2, lam3 = args.lambda1, args.lambda2, args.lambda3
    val_interval = max(args.val_interval, 1)
    early_stopping_best = float("inf")
    early_stopping_bad_checks = 0
    best_monitor_epoch = 0
    stopped_early = False
    completed_epochs = 0
    print(format_banner(lam1=lam1, lam2=lam2, lam3=lam3, args=args, val_interval=val_interval))
    print("-" * 70)

    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        completed_epochs = epoch
        # ── train ──
        model.train()
        train_losses = init_loss_dict()
        n_batches = 0

        for batch in train_loader:
            node_seq = batch["node_seq"].to(device, non_blocking=non_blocking)       # (B, N, h, C)
            speed_seq = batch["speed_seq"].to(device, non_blocking=non_blocking)     # (B, |E|, h)
            d_tgt = batch["demand_target"].to(device, non_blocking=non_blocking)
            c_tgt = batch["supply_target"].to(device, non_blocking=non_blocking)
            v_tgt = batch["speed_target"].to(device, non_blocking=non_blocking)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                d_pred, c_pred, v_pred = forward_for_task(model, node_seq, speed_seq, args.train_task)
                loss_v = mse(v_pred, v_tgt)
                if args.train_task == "v":
                    zero = loss_v.new_zeros(())
                    loss_d = zero
                    loss_c = zero
                else:
                    loss_d = mse(d_pred, d_tgt)
                    loss_c = mse(c_pred, c_tgt)
                loss_dc, loss_task_v = build_task_losses(
                    loss_d,
                    loss_c,
                    loss_v,
                    lam1=lam1,
                    lam2=lam2,
                    lam3=lam3,
                )

            optimizer.zero_grad(set_to_none=True)
            if optimize_dc and optimize_v:
                loss_dc.backward(retain_graph=True)
                loss_task_v.backward()
            elif optimize_v:
                loss_task_v.backward()
            else:
                loss_dc.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses["dc"] += loss_dc.item()
            train_losses["v"] += loss_task_v.item()
            train_losses["demand"] += loss_d.item()
            train_losses["supply"] += loss_c.item()
            train_losses["speed"] += loss_v.item()
            n_batches += 1

        for k in train_losses:
            train_losses[k] /= max(n_batches, 1)

        # ── val ──
        should_validate = (epoch % val_interval == 0) or (epoch == args.epochs)
        if should_validate:
            model.eval()
            val_losses = evaluate_loader(
                model,
                val_loader,
                train_task=args.train_task,
                device=device,
                non_blocking=non_blocking,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                mse=mse,
                lam1=lam1,
                lam2=lam2,
                lam3=lam3,
            )
            val_raw_metrics = skipped_raw_metric_dict()
            should_compute_raw_metrics = args.monitor_task == "raw_dc" or args.train_task != "v"
            if should_compute_raw_metrics:
                raw_metrics = evaluate_loader_raw_metrics(
                    model,
                    val_loader,
                    train_task=args.train_task,
                    device=device,
                    non_blocking=non_blocking,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    normalization_stats=normalization_stats,
                    time_slot_minutes=time_slot_minutes,
                    report_horizons=report_horizons,
                )
                val_raw_metrics = {
                    "raw_dc": (
                        build_raw_dc_metric(raw_metrics)
                        if args.train_task != "v"
                        else None
                    ),
                    "demand": raw_metrics["demand"] if args.train_task != "v" else None,
                    "supply": raw_metrics["supply"] if args.train_task != "v" else None,
                    "speed": raw_metrics["speed"] if args.train_task != "dc" else None,
                    "report_horizons": raw_metrics["report_horizons"],
                }
            if args.monitor_task == "raw_dc":
                monitor_value = val_raw_metrics["raw_dc"]
            else:
                monitor_value = val_losses[args.monitor_task]
            scheduler.step(monitor_value)
        else:
            val_losses = skipped_loss_dict()
            val_raw_metrics = skipped_raw_metric_dict()

        # ── 記錄 ──
        elapsed = time.time() - t0
        display_epoch = args.epoch_offset + epoch
        record = build_history_record(
            epoch=display_epoch,
            train_losses=train_losses,
            val_losses=val_losses,
            val_raw_metrics=val_raw_metrics,
            lr=optimizer.param_groups[0]["lr"],
            elapsed=elapsed,
            train_task=args.train_task,
        )
        history.append(record)

        val_msg = format_val_message(
            val_losses=val_losses,
            val_raw_metrics=val_raw_metrics,
            train_task=args.train_task,
        )
        if epoch % args.log_interval == 0 or epoch == 1:
            if args.train_task == "v":
                print(
                    f"[Ep {display_epoch:>4d}]  "
                    f"TrainV={train_losses['v']:.4f} "
                    f"(S={train_losses['speed']:.3f})  "
                    f"{val_msg}  "
                    f"({elapsed:.0f}s)"
                )
            elif args.train_task == "dc":
                print(
                    f"[Ep {display_epoch:>4d}]  "
                    f"TrainDC={train_losses['dc']:.4f} "
                    f"(D={train_losses['demand']:.3f} C={train_losses['supply']:.3f})  "
                    f"{val_msg}  "
                    f"({elapsed:.0f}s)"
                )
            else:
                print(
                    f"[Ep {display_epoch:>4d}]  "
                    f"TrainDC={train_losses['dc']:.4f} "
                    f"TrainV={train_losses['v']:.4f} "
                    f"(D={train_losses['demand']:.3f} C={train_losses['supply']:.3f} S={train_losses['speed']:.3f})  "
                    f"{val_msg}  "
                    f"({elapsed:.0f}s)"
                )

        # ── 儲存最佳 ──
        if optimize_dc and val_losses["dc"] is not None and val_losses["dc"] < best_val_losses["dc"]:
            best_val_losses["dc"] = val_losses["dc"]
            torch.save(model.state_dict(), log_dir / "stgat_best_dc.pt")
            if args.monitor_task == "dc":
                torch.save(model.state_dict(), log_dir / "stgat_best.pt")
                best_monitor_values["dc"] = val_losses["dc"]
                selected_val_losses = dict(val_losses)
                selected_val_raw_metrics = dict(val_raw_metrics)
        if optimize_v and val_losses["v"] is not None and val_losses["v"] < best_val_losses["v"]:
            best_val_losses["v"] = val_losses["v"]
            torch.save(model.state_dict(), log_dir / "stgat_best_v.pt")
            if args.monitor_task == "v":
                torch.save(model.state_dict(), log_dir / "stgat_best.pt")
                best_monitor_values["v"] = val_losses["v"]
                selected_val_losses = dict(val_losses)
                selected_val_raw_metrics = dict(val_raw_metrics)
        if val_raw_metrics["raw_dc"] is not None and val_raw_metrics["raw_dc"] < best_monitor_values["raw_dc"]:
            best_monitor_values["raw_dc"] = val_raw_metrics["raw_dc"]
            torch.save(model.state_dict(), log_dir / "stgat_best_raw_dc.pt")
            if args.monitor_task == "raw_dc":
                torch.save(model.state_dict(), log_dir / "stgat_best.pt")
                selected_val_losses = dict(val_losses)
                selected_val_raw_metrics = dict(val_raw_metrics)

        if should_validate and monitor_value is not None:
            if monitor_value < early_stopping_best - args.early_stopping_min_delta:
                early_stopping_best = float(monitor_value)
                early_stopping_bad_checks = 0
                best_monitor_epoch = display_epoch
            else:
                early_stopping_bad_checks += 1
            if (
                args.early_stopping_patience > 0
                and early_stopping_bad_checks >= args.early_stopping_patience
            ):
                stopped_early = True
                print(
                    "[EARLY STOP] "
                    f"epoch={display_epoch} best_epoch={best_monitor_epoch} "
                    f"best_{args.monitor_task}={early_stopping_best:.6f} "
                    f"no_improvement={early_stopping_bad_checks}"
                )
                break

    # ── 結束 ──
    torch.save(model.state_dict(), log_dir / "stgat_final.pt")
    best_monitor_value = best_monitor_values[args.monitor_task]
    if optimize_v:
        print(f"Best Val V = {best_val_losses['v']:.5f}")
    else:
        print("Best Val V = skipped (train_task=dc)")
    if optimize_dc and best_val_losses["dc"] < float("inf"):
        print(f"Best Val DC = {best_val_losses['dc']:.5f}")
    if args.train_task != "v" and best_monitor_values["raw_dc"] < float("inf"):
        print(f"Best Val Raw DC = {best_monitor_values['raw_dc']:.5f}")
    print(f"Default checkpoint is {args.monitor_task}-best: {log_dir / 'stgat_best.pt'}")
    print(f"\n訓練完成 | Best Monitor Metric = {best_monitor_value:.5f}")
    print(f"模型已儲存至 {log_dir / 'stgat_best.pt'}")

    best_state = torch.load(
        log_dir / "stgat_best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(best_state)
    model.eval()
    test_losses = evaluate_loader(
        model,
        test_loader,
        train_task=args.train_task,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        mse=mse,
        lam1=lam1,
        lam2=lam2,
        lam3=lam3,
    )
    test_raw_metrics = evaluate_loader_raw_metrics(
        model,
        test_loader,
        train_task=args.train_task,
        device=device,
        non_blocking=non_blocking,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        normalization_stats=normalization_stats,
        time_slot_minutes=time_slot_minutes,
        report_horizons=report_horizons,
    )
    if args.train_task == "v":
        print(
            "最終 Test="
            f"V={test_losses['v']:.4f} "
            f"(S={test_losses['speed']:.3f})"
        )
        print(
            "原始尺度 Test RMSE="
            f"V:{test_raw_metrics['speed']['rmse']:.3f}"
        )
    elif args.train_task == "dc":
        print(
            "最終 Test="
            f"DC={test_losses['dc']:.4f} "
            f"(D={test_losses['demand']:.3f} C={test_losses['supply']:.3f})"
        )
        print(
            "原始尺度 Test RMSE="
            f"D:{test_raw_metrics['demand']['rmse']:.3f} "
            f"C:{test_raw_metrics['supply']['rmse']:.3f}"
        )
        demand_report = summarize_report_metrics(test_raw_metrics["demand"])
        supply_report = summarize_report_metrics(test_raw_metrics["supply"])
        if demand_report:
            print(f"Paper-aligned demand RMSE={demand_report}")
        if supply_report:
            print(f"Paper-aligned supply RMSE={supply_report}")
    else:
        print(
            "最終 Test="
            f"DC={test_losses['dc']:.4f} "
            f"V={test_losses['v']:.4f} "
            f"(D={test_losses['demand']:.3f} C={test_losses['supply']:.3f} S={test_losses['speed']:.3f})"
        )
        print(
            "原始尺度 Test RMSE="
            f"D:{test_raw_metrics['demand']['rmse']:.3f} "
            f"C:{test_raw_metrics['supply']['rmse']:.3f} "
            f"V:{test_raw_metrics['speed']['rmse']:.3f}"
        )
        demand_report = summarize_report_metrics(test_raw_metrics["demand"])
        supply_report = summarize_report_metrics(test_raw_metrics["supply"])
        speed_report = summarize_report_metrics(test_raw_metrics["speed"])
        if demand_report:
            print(f"Paper-aligned demand RMSE={demand_report}")
        if supply_report:
            print(f"Paper-aligned supply RMSE={supply_report}")
        if speed_report:
            print(f"Paper-aligned speed RMSE={speed_report}")

    source_manifest_path = Path(args.data_dir) / "manifest.json"
    data_source = Path(args.data_dir).name
    source_dataset_manifest: dict[str, object] = {}
    if source_manifest_path.exists():
        try:
            source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
            data_source = str(source_manifest.get("dataset_name") or data_source)
            source_dataset_manifest = {
                "dataset_name": source_manifest.get("dataset_name"),
                "source_city": source_manifest.get("source_city"),
                "schema_version": source_manifest.get("schema_version"),
                "temporal_partition": source_manifest.get("temporal_partition", {}),
            }
        except (OSError, json.JSONDecodeError):
            source_dataset_manifest = {"manifest_path": str(source_manifest_path), "read_error": True}

    # 儲存訓練資料元資訊（供 pipeline 使用）
    meta = {
        "num_nodes": N,
        "num_edges": nE,
        "edge_index": edge_index.tolist(),
        "edge_lengths": edge_lengths.tolist(),
        "adj": adj.tolist(),
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_st_blocks": args.num_st_blocks,
        "num_gtcn_layers": args.num_gtcn_layers,
        "kernel_size": args.kernel_size,
        "adaptive_topk": args.adaptive_topk,
        "speed_adaptive_topk": args.speed_adaptive_topk,
        "speed_use_adaptive": args.speed_use_adaptive,
        "speed_use_fixed_graph": not args.disable_speed_fixed_graph,
        "speed_adaptive_domain": args.v_domain,
        "use_fixed_edge_length_feature": args.use_fixed_edge_length_feature,
        "pred_horizon": args.pred_horizon,
        "hist_len": args.hist_len,
        "epoch_offset": args.epoch_offset,
        "completed_epochs": completed_epochs,
        "stopped_early": stopped_early,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_monitor_epoch": best_monitor_epoch,
        "init_checkpoint": args.init_checkpoint,
        "init_load_mode": args.init_load_mode,
        "init_load_summary": init_load_summary,
        "time_slot_minutes": time_slot_minutes,
        "report_horizons": report_horizons,
        "node_feat_dim": node_feat_dim,
        "use_time_features": bool(time_feature_names),
        "time_feature_names": time_feature_names,
        "loss_space": "normalized",
        "loss_tasks": (
            {"v": {"formula": f"{lam3} * speed"}}
            if args.train_task == "v"
            else {
                "dc": {"formula": f"{lam1} * demand + {lam2} * supply"},
                "v": {"formula": f"{lam3} * speed"},
            }
        ),
        "train_task": args.train_task,
        "optimized_tasks": optimized_tasks,
        "checkpoint_selection": {
            "default_checkpoint": "stgat_best.pt",
            "default_monitor": args.monitor_task,
            "available_checkpoints": (
                {
                    "v": "stgat_best_v.pt",
                }
                if args.train_task == "v"
                else (
                {
                    "dc": "stgat_best_dc.pt",
                    "v": "stgat_best_v.pt",
                    "raw_dc": "stgat_best_raw_dc.pt",
                }
                if optimize_v
                else {
                    "dc": "stgat_best_dc.pt",
                    "raw_dc": "stgat_best_raw_dc.pt",
                }
                )
            ),
        },
        "normalization": serialize_normalization_stats(normalization_stats),
        "split_strategy": CALENDAR_SPLIT_STRATEGY,
        "split_description": CALENDAR_SPLIT_DESCRIPTION,
        "normalization_time_steps": int(train_time_mask.sum()),
        "split_counts": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
        "data_source": data_source,
        "data_dir": args.data_dir,
        "source_dataset_manifest": source_dataset_manifest,
    }
    with open(log_dir / "stgat_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(log_dir / "predictor_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"訓練日誌已儲存至 {log_dir / 'predictor_log.json'}")

    with open(log_dir / "predictor_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "loss_space": "normalized",
                "normalized_loss": filter_normalized_losses(test_losses, args.train_task),
                "val_normalized_loss": filter_normalized_losses(selected_val_losses, args.train_task),
                "raw_metrics": filter_raw_metrics(test_raw_metrics, args.train_task),
                "raw_metrics_per_step": filter_raw_metrics_per_step(test_raw_metrics, args.train_task),
                "raw_metrics_report": filter_raw_metrics_report(test_raw_metrics, args.train_task),
                "report_horizons": extract_report_horizons(test_raw_metrics),
                "val_raw_metrics": (
                    filter_raw_metrics(selected_val_raw_metrics, args.train_task)
                    if selected_val_raw_metrics["raw_dc"] is not None or selected_val_raw_metrics["speed"] is not None
                    else None
                ),
                "val_raw_metrics_per_step": (
                    filter_raw_metrics_per_step(selected_val_raw_metrics, args.train_task)
                    if selected_val_raw_metrics["raw_dc"] is not None or selected_val_raw_metrics["speed"] is not None
                    else None
                ),
                "val_raw_metrics_report": (
                    filter_raw_metrics_report(selected_val_raw_metrics, args.train_task)
                    if selected_val_raw_metrics["raw_dc"] is not None or selected_val_raw_metrics["speed"] is not None
                    else None
                ),
                "selected_checkpoint": "stgat_best.pt",
                "selected_checkpoint_task": args.monitor_task,
                "selected_checkpoint_metric": best_monitor_values[args.monitor_task],
                "completed_epochs": completed_epochs,
                "stopped_early": stopped_early,
                "early_stopping_patience": args.early_stopping_patience,
                "best_monitor_epoch": best_monitor_epoch,
                "train_task": args.train_task,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"測試指標已儲存至 {log_dir / 'predictor_test_metrics.json'}")


# ── CLI ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train STGAT Predictor")

    # 資料
    p.add_argument("--data-dir", type=str, default="data", help="真實資料目錄（adjacency、時序特徵）")
    p.add_argument(
        "--max-time-steps",
        type=int,
        default=0,
        help="截斷時間步（0=全部；除錯或減輕顯存時可設小於完整 T）",
    )
    p.add_argument(
        "--edge-length-source",
        type=str,
        default="osrm",
        choices=["osrm", "centroid"],
        help="邊長來源：osrm 優先讀 edge_lengths_osrm.npy，否則 centroid",
    )
    p.add_argument(
        "--disable-time-features",
        action="store_true",
        help="停用 month / weekday / slot 時間特徵，回到舊版 demand+supply 節點輸入",
    )
    p.add_argument("--hist-len", type=int, default=12, help="歷史窗口 h")
    p.add_argument(
        "--pred-horizon",
        type=int,
        default=None,
        help="Predicted steps p; DC/V defaults to 4 for 15/30/60 reporting, joint defaults to 3.",
    )
    p.add_argument(
        "--report-horizons-minutes",
        type=str,
        default=None,
        help="Comma-separated report horizons in minutes (for example 15,30,60). Empty disables report extraction.",
    )

    # 模型
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--num-st-blocks", type=int, default=2)
    p.add_argument(
        "--adaptive-topk",
        type=int,
        default=16,
        help="Top-k neighbors kept by the learned adaptive adjacency; 0 keeps the dense graph.",
    )
    p.add_argument(
        "--disable-node-adaptive",
        action="store_true",
        help="Disable the learned adaptive node-topology branch for DC demand/supply ablations.",
    )
    p.add_argument(
        "--disable-node-fixed-graph",
        action="store_true",
        help="Disable the fixed external graph node branch for DC demand/supply ablations.",
    )
    p.add_argument(
        "--speed-use-adaptive",
        action="store_true",
        help="Use the adaptive speed branch already implemented in STGATPredictor.",
    )
    p.add_argument(
        "--disable-speed-fixed-graph",
        action="store_true",
        help="Disable the fixed external/line-graph speed branch for V ablations.",
    )
    p.add_argument(
        "--speed-adaptive-topk",
        type=int,
        default=None,
        help="Top-k for the speed adaptive branch; defaults to --adaptive-topk.",
    )
    p.add_argument(
        "--v-domain",
        type=str,
        default="edge",
        choices=["edge", "node"],
        help="Speed prediction domain used by STGATPredictor.forward_v.",
    )
    p.add_argument(
        "--use-fixed-edge-length-feature",
        action="store_true",
        help="Include fixed edge length as an extra speed-domain feature.",
    )
    p.add_argument("--num-gtcn-layers", type=int, default=2)
    p.add_argument("--kernel-size", type=int, default=3)

    # 訓練
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda1", type=float, default=1.0, help="需求損失權重")
    p.add_argument("--lambda2", type=float, default=1.0, help="空車損失權重")
    p.add_argument("--lambda3", type=float, default=1.0, help="速度損失權重")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp32"],
        help="CUDA 上預設 auto->bf16；CPU 會自動退回 fp32",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader workers；-1 代表自動（CUDA 預設最多 8，CPU 預設 0）",
    )
    p.add_argument("--log-dir", type=str, default="runs")
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional model state_dict checkpoint used to initialize training.",
    )
    p.add_argument(
        "--init-load-mode",
        type=str,
        default="strict",
        choices=["strict", "compatible"],
        help="How to load --init-checkpoint; compatible loads shape-matched tensors only.",
    )
    p.add_argument(
        "--epoch-offset",
        type=int,
        default=0,
        help="Epoch number offset used for continuation logs, e.g. 100 prints epochs 101..",
    )
    p.add_argument(
        "--monitor-task",
        type=str,
        default="raw_dc",
        choices=["dc", "v", "raw_dc"],
        help="Validation metric used for the default scheduler/checkpoint (raw_dc = demand_rmse + supply_rmse).",
    )
    p.add_argument(
        "--train-task",
        type=str,
        default="joint",
        choices=["joint", "dc", "v"],
        help="Optimization mode: update both task heads, only the DC objective, or only the V objective.",
    )
    p.add_argument(
        "--val-interval",
        type=int,
        default=1,
        help="每隔多少個 epoch 跑一次 validation；最後一個 epoch 會強制驗證",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many validation checks without monitor improvement; 0 disables early stopping.",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum monitor decrease required to reset early-stopping patience.",
    )

    p.add_argument("--compile", action="store_true", help="啟用 torch.compile 加速")
    p.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile 的 mode",
    )

    args = p.parse_args()
    if args.speed_adaptive_topk is None:
        args.speed_adaptive_topk = args.adaptive_topk
    if args.pred_horizon is None:
        args.pred_horizon = (
            DEFAULT_REPORT_PRED_HORIZON
            if args.train_task in {"dc", "v"}
            else DEFAULT_NON_REPORT_PRED_HORIZON
        )

    if args.report_horizons_minutes is None:
        args.report_horizons_minutes = (
            ",".join(str(v) for v in DEFAULT_REPORT_HORIZONS_MINUTES)
            if args.train_task in {"dc", "v"}
            else ""
        )
    args.report_horizons_minutes = parse_report_horizons_minutes(args.report_horizons_minutes)
    return args


if __name__ == "__main__":
    train(parse_args())
