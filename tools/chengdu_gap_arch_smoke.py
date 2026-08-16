from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from predictor_normalization import (
    build_normalization_stats,
    normalize_node_features,
    normalize_speed_features,
)
from stgat_model import STGATPredictor
from train_predictor import (
    build_monthly_split_indices,
    filter_split_indices_by_time_mask,
    load_time_meta_for_training,
    parse_report_horizons_minutes,
    resolve_report_horizons,
    infer_time_slot_minutes,
)


def build_window_time_mask(length: int, indices: list[int], hist_len: int, pred_horizon: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for idx in indices:
        mask[idx + hist_len: idx + hist_len + pred_horizon] = True
    return mask


def take_evenly(indices: list[int], max_samples: int) -> list[int]:
    if max_samples <= 0 or len(indices) <= max_samples:
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, max_samples)
    return [indices[int(round(pos))] for pos in positions]


def edge_lengths_from_matrix(data_dir: Path, edge_index: np.ndarray) -> np.ndarray:
    for name in ("edge_lengths_osrm.npy", "edge_lengths.npy"):
        path = data_dir / name
        if path.exists():
            values = np.load(path).astype(np.float32)
            if values.ndim == 1:
                if values.shape[0] != edge_index.shape[0]:
                    raise ValueError(f"{path} length does not match edge count.")
                return values
            return np.asarray([values[int(src), int(dst)] for src, dst in edge_index], dtype=np.float32)
    raise FileNotFoundError("Missing edge length file.")


@dataclass
class GapStats:
    mean: float
    std: float
    transform: str = "raw"
    log_scale: float = 10.0

    def normalize(self, values: np.ndarray) -> np.ndarray:
        encoded = encode_gap_values(values, transform=self.transform, log_scale=self.log_scale)
        return ((encoded - self.mean) / self.std).astype(np.float32)

    def denormalize_torch(self, values: torch.Tensor) -> torch.Tensor:
        encoded = values * self.std + self.mean
        return decode_gap_values_torch(encoded, transform=self.transform, log_scale=self.log_scale)

    def denormalize_numpy(self, values: np.ndarray) -> np.ndarray:
        encoded = values * self.std + self.mean
        return decode_gap_values_numpy(encoded, transform=self.transform, log_scale=self.log_scale)


def encode_gap_values(values: np.ndarray, *, transform: str, log_scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if transform == "raw":
        return arr.astype(np.float32)
    if transform == "signed_log":
        scale = float(max(log_scale, 1e-6))
        return (np.sign(arr) * np.log1p(np.abs(arr) / scale)).astype(np.float32)
    raise ValueError(f"Unsupported GAP target transform: {transform}")


def decode_gap_values_numpy(values: np.ndarray, *, transform: str, log_scale: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if transform == "raw":
        return arr.astype(np.float32)
    if transform == "signed_log":
        scale = float(max(log_scale, 1e-6))
        return (np.sign(arr) * np.expm1(np.abs(arr)) * scale).astype(np.float32)
    raise ValueError(f"Unsupported GAP target transform: {transform}")


def decode_gap_values_torch(values: torch.Tensor, *, transform: str, log_scale: float) -> torch.Tensor:
    if transform == "raw":
        return values
    if transform == "signed_log":
        scale = float(max(log_scale, 1e-6))
        return torch.sign(values) * torch.expm1(torch.abs(values)) * scale
    raise ValueError(f"Unsupported GAP target transform: {transform}")


class GapWindowDataset(Dataset):
    def __init__(
        self,
        indices: list[int],
        *,
        node_features: np.ndarray,
        speed_features: np.ndarray,
        demand_target_norm: np.ndarray,
        supply_target_norm: np.ndarray,
        gap_target_norm: np.ndarray,
        gap_target_raw: np.ndarray,
        hist_len: int,
        pred_horizon: int,
        gap_scale_factors: list[float] | None = None,
        gap_mean: float = 0.0,
        gap_std: float = 1.0,
        gap_transform: str = "raw",
        gap_log_scale: float = 10.0,
    ) -> None:
        self.indices = list(indices)
        self.node_features = np.asarray(node_features, dtype=np.float32)
        self.speed_features = np.asarray(speed_features, dtype=np.float32)
        self.demand_target_norm = np.asarray(demand_target_norm, dtype=np.float32)
        self.supply_target_norm = np.asarray(supply_target_norm, dtype=np.float32)
        self.gap_target_norm = np.asarray(gap_target_norm, dtype=np.float32)
        self.gap_target_raw = np.asarray(gap_target_raw, dtype=np.float32)
        self.hist_len = int(hist_len)
        self.pred_horizon = int(pred_horizon)
        self.gap_scale_factors = list(gap_scale_factors or [1.0])
        self.gap_mean = float(gap_mean)
        self.gap_std = float(max(gap_std, 1e-6))
        self.gap_transform = str(gap_transform)
        self.gap_log_scale = float(gap_log_scale)

    def __len__(self) -> int:
        return len(self.indices) * len(self.gap_scale_factors)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        scale_idx = int(item) % len(self.gap_scale_factors)
        idx = int(self.indices[int(item) // len(self.gap_scale_factors)])
        gap_scale = float(self.gap_scale_factors[scale_idx])
        target_start = idx + self.hist_len
        target_end = target_start + self.pred_horizon
        node_seq = self.node_features[idx:target_start].transpose(1, 0, 2).copy()
        speed_seq = self.speed_features[idx:target_start].T
        gap_target_raw = self.gap_target_raw[target_start:target_end].T.copy()
        gap_target_norm = self.gap_target_norm[target_start:target_end].T.copy()
        if gap_scale != 1.0:
            hist_gap_encoded = node_seq[:, :, 0] * self.gap_std + self.gap_mean
            hist_gap_raw = decode_gap_values_numpy(
                hist_gap_encoded,
                transform=self.gap_transform,
                log_scale=self.gap_log_scale,
            )
            hist_gap_raw *= gap_scale
            hist_gap_encoded = encode_gap_values(
                hist_gap_raw,
                transform=self.gap_transform,
                log_scale=self.gap_log_scale,
            )
            node_seq[:, :, 0] = (hist_gap_encoded - self.gap_mean) / self.gap_std
            gap_target_raw *= gap_scale
            gap_target_encoded = encode_gap_values(
                gap_target_raw,
                transform=self.gap_transform,
                log_scale=self.gap_log_scale,
            )
            gap_target_norm = ((gap_target_encoded - self.gap_mean) / self.gap_std).astype(np.float32)
        return {
            "node_seq": torch.from_numpy(node_seq),
            "speed_seq": torch.from_numpy(speed_seq),
            "demand_target": torch.from_numpy(self.demand_target_norm[target_start:target_end].T),
            "supply_target": torch.from_numpy(self.supply_target_norm[target_start:target_end].T),
            "gap_target": torch.from_numpy(gap_target_norm),
            "gap_target_raw": torch.from_numpy(gap_target_raw),
        }


class GapHeadSTGAT(STGATPredictor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gap_head = nn.Linear(self.demand_head.in_features, self.demand_head.out_features)

    def forward_gap_features(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        temporal_feat_seq = node_seq[:, 0, :, 2:] if self.time_feat_dim > 0 else None
        node_h = self.node_proj(node_seq)
        del temporal_feat_seq
        if self.node_use_fixed_graph:
            h_fix = self._run_fixed_node_path(node_h, speed_seq)
        else:
            h_fix = None
        if self.node_use_adaptive:
            h_adp = self._run_adaptive_node_path(node_h)
        else:
            h_adp = None
        if h_fix is not None and h_adp is not None:
            return self.fusion(h_fix, h_adp)
        if h_fix is not None:
            return h_fix
        if h_adp is not None:
            return h_adp
        raise RuntimeError("No node path is enabled.")

    def forward_dcg(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del speed_seq  # Rebound below keeps type checkers quiet when called through wrappers.
        raise NotImplementedError


class DcgGapHeadSTGAT(GapHeadSTGAT):
    def forward_dcg(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_node = self.forward_gap_features(node_seq, speed_seq)
        return self.demand_head(h_node), self.supply_head(h_node), self.gap_head(h_node)


class GapOnlySTGAT(GapHeadSTGAT):
    def forward_gap(self, node_seq: torch.Tensor, speed_seq: torch.Tensor) -> torch.Tensor:
        h_node = self.forward_gap_features(node_seq, speed_seq)
        return self.gap_head(h_node)


def update_bucket(bucket: dict[str, float], pred: np.ndarray, target: np.ndarray) -> None:
    diff = pred.astype(np.float64) - target.astype(np.float64)
    mask = np.isfinite(diff) & np.isfinite(target)
    sq = np.square(diff[mask])
    ab = np.abs(diff[mask])
    bucket["se"] += float(sq.sum())
    bucket["ae"] += float(ab.sum())
    bucket["count"] += float(mask.sum())
    mape_mask = mask & (np.abs(target) > 1e-6)
    if np.any(mape_mask):
        bucket["ape"] += float((np.abs(diff[mape_mask]) / np.abs(target[mape_mask]) * 100.0).sum())
        bucket["mape_count"] += float(mape_mask.sum())


def finalize_bucket(bucket: dict[str, float]) -> dict[str, float]:
    count = max(float(bucket["count"]), 1.0)
    mape_count = max(float(bucket["mape_count"]), 1.0)
    mse = float(bucket["se"]) / count
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(bucket["ae"]) / count,
        "mape": float(bucket["ape"]) / mape_count,
        "count": float(bucket["count"]),
        "mape_count": float(bucket["mape_count"]),
    }


def compute_gap_metrics(
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    report_horizons: dict[str, Any],
) -> dict[str, Any]:
    pred = np.asarray(pred_raw, dtype=np.float32)
    target = np.asarray(target_raw, dtype=np.float32)
    overall = {"se": 0.0, "ae": 0.0, "ape": 0.0, "count": 0.0, "mape_count": 0.0}
    per_step = [
        {"se": 0.0, "ae": 0.0, "ape": 0.0, "count": 0.0, "mape_count": 0.0}
        for _ in range(pred.shape[-1])
    ]
    update_bucket(overall, pred, target)
    for step_idx in range(pred.shape[-1]):
        update_bucket(per_step[step_idx], pred[..., step_idx], target[..., step_idx])

    report: dict[str, dict[str, float]] = {}
    for minute, step in zip(report_horizons["resolved_minutes"], report_horizons["resolved_steps"]):
        values = finalize_bucket(per_step[int(step) - 1])
        values["step"] = int(step)
        values["minutes"] = int(minute)
        report[f"{int(minute)}min"] = values
    return {"overall": finalize_bucket(overall), "report": report}


def score_gap(metrics: dict[str, Any], *, mape_weight: float) -> float:
    values = []
    for item in metrics["report"].values():
        values.append(float(item["rmse"]) + float(item["mae"]) + float(mape_weight) * float(item["mape"]))
    return float(np.mean(values))


def fit_stepwise_shrink_calibration(
    val_pred: np.ndarray,
    val_target: np.ndarray,
    *,
    mape_weight: float,
    alpha_grid: list[float],
    tau_grid: list[float],
) -> dict[str, Any]:
    pred = np.asarray(val_pred, dtype=np.float32)
    target = np.asarray(val_target, dtype=np.float32)
    horizon = pred.shape[-1]
    alphas = np.ones(horizon, dtype=np.float32)
    taus = np.zeros(horizon, dtype=np.float32)
    step_scores: list[float] = []
    for step_idx in range(horizon):
        x = pred[..., step_idx]
        y = target[..., step_idx]
        best_score = float("inf")
        best_alpha = 1.0
        best_tau = 0.0
        for alpha in alpha_grid:
            scaled = x * float(alpha)
            abs_scaled = np.abs(scaled)
            sign_scaled = np.sign(scaled)
            for tau in tau_grid:
                cal = sign_scaled * np.maximum(abs_scaled - float(tau), 0.0)
                bucket = {"se": 0.0, "ae": 0.0, "ape": 0.0, "count": 0.0, "mape_count": 0.0}
                update_bucket(bucket, cal, y)
                metrics = finalize_bucket(bucket)
                score = float(metrics["rmse"]) + float(metrics["mae"]) + float(mape_weight) * float(metrics["mape"])
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
                    best_tau = float(tau)
        alphas[step_idx] = best_alpha
        taus[step_idx] = best_tau
        step_scores.append(best_score)
    return {
        "type": "stepwise_scale_soft_threshold",
        "alpha": alphas.tolist(),
        "tau": taus.tolist(),
        "step_scores": step_scores,
    }


def apply_stepwise_shrink_calibration(pred: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    alpha = np.asarray(calibration["alpha"], dtype=np.float32).reshape([1] * (arr.ndim - 1) + [-1])
    tau = np.asarray(calibration["tau"], dtype=np.float32).reshape([1] * (arr.ndim - 1) + [-1])
    scaled = arr * alpha
    return (np.sign(scaled) * np.maximum(np.abs(scaled) - tau, 0.0)).astype(np.float32)


def fit_stepwise_affine_calibration(
    val_pred: np.ndarray,
    val_target: np.ndarray,
    *,
    mape_weight: float,
    alpha_grid: list[float],
    bias_grid: list[float],
) -> dict[str, Any]:
    pred = np.asarray(val_pred, dtype=np.float32)
    target = np.asarray(val_target, dtype=np.float32)
    horizon = pred.shape[-1]
    alphas = np.ones(horizon, dtype=np.float32)
    biases = np.zeros(horizon, dtype=np.float32)
    step_scores: list[float] = []
    for step_idx in range(horizon):
        x = pred[..., step_idx]
        y = target[..., step_idx]
        best_score = float("inf")
        best_alpha = 1.0
        best_bias = 0.0
        for alpha in alpha_grid:
            scaled = x * float(alpha)
            for bias in bias_grid:
                cal = scaled + float(bias)
                bucket = {"se": 0.0, "ae": 0.0, "ape": 0.0, "count": 0.0, "mape_count": 0.0}
                update_bucket(bucket, cal, y)
                metrics = finalize_bucket(bucket)
                score = float(metrics["rmse"]) + float(metrics["mae"]) + float(mape_weight) * float(metrics["mape"])
                if score < best_score:
                    best_score = score
                    best_alpha = float(alpha)
                    best_bias = float(bias)
        alphas[step_idx] = best_alpha
        biases[step_idx] = best_bias
        step_scores.append(best_score)
    return {
        "type": "stepwise_affine",
        "alpha": alphas.tolist(),
        "bias": biases.tolist(),
        "step_scores": step_scores,
    }


def apply_stepwise_affine_calibration(pred: np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    alpha = np.asarray(calibration["alpha"], dtype=np.float32).reshape([1] * (arr.ndim - 1) + [-1])
    bias = np.asarray(calibration["bias"], dtype=np.float32).reshape([1] * (arr.ndim - 1) + [-1])
    return (arr * alpha + bias).astype(np.float32)


def fit_persistence_blend_calibration(
    val_pred: np.ndarray,
    val_persistence: np.ndarray,
    val_target: np.ndarray,
    *,
    mape_weight: float,
    alpha_grid: list[float],
) -> dict[str, Any]:
    pred = np.asarray(val_pred, dtype=np.float32)
    persistence = np.asarray(val_persistence, dtype=np.float32)
    target = np.asarray(val_target, dtype=np.float32)
    horizon = pred.shape[-1]
    alphas = np.ones(horizon, dtype=np.float32)
    step_scores: list[float] = []
    for step_idx in range(horizon):
        x = pred[..., step_idx]
        p = persistence[..., step_idx]
        y = target[..., step_idx]
        best_score = float("inf")
        best_alpha = 1.0
        for alpha in alpha_grid:
            cal = float(alpha) * x + (1.0 - float(alpha)) * p
            bucket = {"se": 0.0, "ae": 0.0, "ape": 0.0, "count": 0.0, "mape_count": 0.0}
            update_bucket(bucket, cal, y)
            metrics = finalize_bucket(bucket)
            score = float(metrics["rmse"]) + float(metrics["mae"]) + float(mape_weight) * float(metrics["mape"])
            if score < best_score:
                best_score = score
                best_alpha = float(alpha)
        alphas[step_idx] = best_alpha
        step_scores.append(best_score)
    return {
        "type": "persistence_convex_blend",
        "alpha_model": alphas.tolist(),
        "step_scores": step_scores,
    }


def apply_persistence_blend_calibration(
    pred: np.ndarray,
    persistence: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    baseline = np.asarray(persistence, dtype=np.float32)
    alpha = np.asarray(calibration["alpha_model"], dtype=np.float32).reshape([1] * (arr.ndim - 1) + [-1])
    return (alpha * arr + (1.0 - alpha) * baseline).astype(np.float32)


def parse_float_list(value: str, *, expected: int, name: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values, got {len(parts)}.")
    parsed = [float(part) for part in parts]
    if any(item < 0 for item in parsed):
        raise ValueError(f"{name} values must be non-negative.")
    if not any(item > 0 for item in parsed):
        raise ValueError(f"{name} must contain at least one positive value.")
    return parsed


def weighted_mean_by_step(
    values: torch.Tensor,
    weights: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    view_shape = [1] * values.ndim
    view_shape[-1] = int(weights.numel())
    w = weights.to(device=values.device, dtype=values.dtype).view(view_shape)
    if sample_weights is None:
        return (values * w).sum() / (w.sum() * values[..., 0].numel())
    dynamic_w = sample_weights.to(device=values.device, dtype=values.dtype)
    return (values * w * dynamic_w).sum() / torch.clamp((w * dynamic_w).sum(), min=1.0)


def pointwise_loss(pred: torch.Tensor, target: torch.Tensor, *, base_loss: str, smooth_l1_beta: float) -> torch.Tensor:
    if base_loss == "mse":
        return (pred - target).pow(2)
    if base_loss == "mae":
        return torch.abs(pred - target)
    if base_loss == "smooth_l1":
        return F.smooth_l1_loss(pred, target, beta=float(smooth_l1_beta), reduction="none")
    raise ValueError(f"Unsupported base loss: {base_loss}")


def collect_gap_predictions(
    model: nn.Module,
    loader: DataLoader,
    *,
    variant: str,
    gap_stats: GapStats,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            node_seq = batch["node_seq"].to(device)
            speed_seq = batch["speed_seq"].to(device)
            if variant == "dcg_gap_head":
                _, _, gap_norm = model.forward_dcg(node_seq, speed_seq)  # type: ignore[attr-defined]
            else:
                gap_norm = model.forward_gap(node_seq, speed_seq)  # type: ignore[attr-defined]
            preds.append(gap_stats.denormalize_numpy(gap_norm.detach().cpu().numpy()))
            targets.append(batch["gap_target_raw"].numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0)


def collect_gap_predictions_with_persistence(
    model: nn.Module,
    loader: DataLoader,
    *,
    variant: str,
    gap_stats: GapStats,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    persistences: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            node_seq_cpu = batch["node_seq"]
            node_seq = node_seq_cpu.to(device)
            speed_seq = batch["speed_seq"].to(device)
            if variant == "dcg_gap_head":
                _, _, gap_norm = model.forward_dcg(node_seq, speed_seq)  # type: ignore[attr-defined]
            else:
                gap_norm = model.forward_gap(node_seq, speed_seq)  # type: ignore[attr-defined]
            pred = gap_stats.denormalize_numpy(gap_norm.detach().cpu().numpy())
            last_gap_norm = node_seq_cpu[:, :, -1, 0].numpy()
            last_gap_raw = gap_stats.denormalize_numpy(last_gap_norm)
            persistence = np.repeat(last_gap_raw[..., None], pred.shape[-1], axis=-1)
            preds.append(pred)
            targets.append(batch["gap_target_raw"].numpy())
            persistences.append(persistence.astype(np.float32))
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), np.concatenate(persistences, axis=0)


def train_variant(
    *,
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    gap_stats: GapStats,
    report_horizons: dict[str, Any],
    device: torch.device,
    epochs: int,
    lr: float,
    aux_weight: float,
    score_mape_weight: float,
    loss_mae_weight: float,
    loss_mape_weight: float,
    pred_abs_weight: float,
    pred_abs_step_weights: list[float],
    loss_step_weights: list[float],
    base_loss: str,
    smooth_l1_beta: float,
    small_gap_weight: float,
    small_gap_scale: float,
    calibration_mode: str,
    calibration_mape_weight: float,
    calibration_alpha_grid: list[float],
    calibration_tau_grid: list[float],
    calibration_bias_grid: list[float],
    output_dir: Path,
    checkpoint_every: int,
    early_stopping_patience: int,
    early_stopping_min_delta: float,
) -> dict[str, Any]:
    model.to(device)
    mse = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_score = float("inf")
    best_epoch = 0
    early_stopping_bad_epochs = 0
    stopped_early = False
    history: list[dict[str, float]] = []
    step_weight_tensor = torch.tensor(loss_step_weights, dtype=torch.float32, device=device)
    pred_abs_step_weight_tensor = torch.tensor(pred_abs_step_weights, dtype=torch.float32, device=device)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"{name}_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in train_loader:
            node_seq = batch["node_seq"].to(device)
            speed_seq = batch["speed_seq"].to(device)
            gap_target = batch["gap_target"].to(device)
            gap_target_raw_for_weight = gap_stats.denormalize_torch(gap_target)
            sample_weight_tensor = None
            if small_gap_weight > 0:
                scale = max(float(small_gap_scale), 1e-6)
                sample_weight_tensor = 1.0 + float(small_gap_weight) * torch.exp(
                    -torch.abs(gap_target_raw_for_weight) / scale
                )
            opt.zero_grad(set_to_none=True)
            if name == "dcg_gap_head":
                demand_target = batch["demand_target"].to(device)
                supply_target = batch["supply_target"].to(device)
                demand_pred, supply_pred, gap_pred = model.forward_dcg(node_seq, speed_seq)  # type: ignore[attr-defined]
                gap_loss = weighted_mean_by_step(
                    pointwise_loss(gap_pred, gap_target, base_loss=base_loss, smooth_l1_beta=smooth_l1_beta),
                    step_weight_tensor,
                    sample_weight_tensor,
                )
                loss = gap_loss + aux_weight * (
                    mse(demand_pred, demand_target) + mse(supply_pred, supply_target)
                )
            else:
                gap_pred = model.forward_gap(node_seq, speed_seq)  # type: ignore[attr-defined]
                loss = weighted_mean_by_step(
                    pointwise_loss(gap_pred, gap_target, base_loss=base_loss, smooth_l1_beta=smooth_l1_beta),
                    step_weight_tensor,
                    sample_weight_tensor,
                )
            if loss_mae_weight > 0 or loss_mape_weight > 0 or pred_abs_weight > 0:
                gap_pred_raw = gap_stats.denormalize_torch(gap_pred)
                gap_target_raw = gap_target_raw_for_weight
                raw_abs = torch.abs(gap_pred_raw - gap_target_raw)
                if loss_mae_weight > 0:
                    loss = loss + float(loss_mae_weight) * weighted_mean_by_step(
                        raw_abs, step_weight_tensor, sample_weight_tensor
                    ) / float(gap_stats.std)
                if loss_mape_weight > 0:
                    mape_mask = torch.abs(gap_target_raw) > 1e-6
                    if torch.any(mape_mask):
                        rel_full = torch.zeros_like(raw_abs)
                        rel_full[mape_mask] = raw_abs[mape_mask] / torch.abs(gap_target_raw[mape_mask])
                        valid_w = step_weight_tensor.view(1, 1, -1).expand_as(rel_full)
                        if sample_weight_tensor is not None:
                            valid_w = valid_w * sample_weight_tensor
                        denom = torch.clamp(valid_w[mape_mask].sum(), min=1.0)
                        loss = loss + float(loss_mape_weight) * (rel_full * valid_w).sum() / denom
                if pred_abs_weight > 0:
                    loss = loss + float(pred_abs_weight) * weighted_mean_by_step(
                        torch.abs(gap_pred_raw), pred_abs_step_weight_tensor, sample_weight_tensor
                    ) / float(gap_stats.std)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total_loss += float(loss.detach().cpu().item())
            batches += 1

        val_pred, val_target = collect_gap_predictions(
            model,
            val_loader,
            variant=name,
            gap_stats=gap_stats,
            device=device,
        )
        val_metrics = compute_gap_metrics(val_pred, val_target, report_horizons)
        val_score = score_gap(val_metrics, mape_weight=score_mape_weight)
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(total_loss / max(batches, 1)),
            "val_score": float(val_score),
            "val_rmse_overall": float(val_metrics["overall"]["rmse"]),
            "val_mae_overall": float(val_metrics["overall"]["mae"]),
            "val_mape_overall": float(val_metrics["overall"]["mape"]),
            "val_report": val_metrics["report"],
        }
        history.append(epoch_record)
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")
        print(
            f"{name} epoch {epoch:02d}/{epochs} "
            f"loss={history[-1]['train_loss']:.5f} "
            f"val_score={val_score:.5f} "
            f"val_gap_rmse={val_metrics['overall']['rmse']:.4f} "
            f"val_gap_mae={val_metrics['overall']['mae']:.4f} "
            f"val_gap_mape={val_metrics['overall']['mape']:.2f}"
        )
        if val_score < best_val_score - early_stopping_min_delta:
            best_val_score = val_score
            best_epoch = epoch
            early_stopping_bad_epochs = 0
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            torch.save(
                {
                    "variant": name,
                    "epoch": epoch,
                    "val_score": float(val_score),
                    "val_metrics": val_metrics,
                    "state_dict": best_state,
                },
                checkpoint_dir / f"{name}_best.pt",
            )
        else:
            early_stopping_bad_epochs += 1
        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            torch.save(
                {
                    "variant": name,
                    "epoch": epoch,
                    "val_score": float(val_score),
                    "val_metrics": val_metrics,
                    "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": opt.state_dict(),
                },
                checkpoint_dir / f"{name}_epoch{epoch:03d}.pt",
            )
        if early_stopping_patience > 0 and early_stopping_bad_epochs >= early_stopping_patience:
            stopped_early = True
            print(
                "[EARLY STOP] "
                f"variant={name} epoch={epoch} best_epoch={best_epoch} "
                f"best_val_score={best_val_score:.6f} "
                f"no_improvement={early_stopping_bad_epochs}"
            )
            break

    last_epoch = int(history[-1]["epoch"]) if history else 0
    last_val_score = float(history[-1]["val_score"]) if history else float("inf")
    torch.save(
        {
            "variant": name,
            "epoch": last_epoch,
            "val_score": last_val_score,
            "val_metrics": val_metrics if history else {},
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": opt.state_dict(),
        },
        checkpoint_dir / f"{name}_last.pt",
    )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_persistence: np.ndarray | None = None
    test_persistence: np.ndarray | None = None
    if calibration_mode == "persistence_blend":
        val_pred, val_target, val_persistence = collect_gap_predictions_with_persistence(
            model,
            val_loader,
            variant=name,
            gap_stats=gap_stats,
            device=device,
        )
        test_pred, test_target, test_persistence = collect_gap_predictions_with_persistence(
            model,
            test_loader,
            variant=name,
            gap_stats=gap_stats,
            device=device,
        )
    else:
        val_pred, val_target = collect_gap_predictions(
            model,
            val_loader,
            variant=name,
            gap_stats=gap_stats,
            device=device,
        )
        test_pred, test_target = collect_gap_predictions(
            model,
            test_loader,
            variant=name,
            gap_stats=gap_stats,
            device=device,
        )
    test_metrics = compute_gap_metrics(test_pred, test_target, report_horizons)
    calibration: dict[str, Any] | None = None
    calibrated_metrics: dict[str, Any] | None = None
    val_calibrated_metrics: dict[str, Any] | None = None
    if calibration_mode == "stepwise_shrink":
        calibration = fit_stepwise_shrink_calibration(
            val_pred,
            val_target,
            mape_weight=calibration_mape_weight,
            alpha_grid=calibration_alpha_grid,
            tau_grid=calibration_tau_grid,
        )
        val_cal = apply_stepwise_shrink_calibration(val_pred, calibration)
        test_cal = apply_stepwise_shrink_calibration(test_pred, calibration)
        val_calibrated_metrics = compute_gap_metrics(val_cal, val_target, report_horizons)
        calibrated_metrics = compute_gap_metrics(test_cal, test_target, report_horizons)
    elif calibration_mode == "stepwise_affine":
        calibration = fit_stepwise_affine_calibration(
            val_pred,
            val_target,
            mape_weight=calibration_mape_weight,
            alpha_grid=calibration_alpha_grid,
            bias_grid=calibration_bias_grid,
        )
        val_cal = apply_stepwise_affine_calibration(val_pred, calibration)
        test_cal = apply_stepwise_affine_calibration(test_pred, calibration)
        val_calibrated_metrics = compute_gap_metrics(val_cal, val_target, report_horizons)
        calibrated_metrics = compute_gap_metrics(test_cal, test_target, report_horizons)
    elif calibration_mode == "persistence_blend":
        if val_persistence is None or test_persistence is None:
            raise RuntimeError("Persistence calibration requires persistence arrays.")
        calibration = fit_persistence_blend_calibration(
            val_pred,
            val_persistence,
            val_target,
            mape_weight=calibration_mape_weight,
            alpha_grid=calibration_alpha_grid,
        )
        val_cal = apply_persistence_blend_calibration(val_pred, val_persistence, calibration)
        test_cal = apply_persistence_blend_calibration(test_pred, test_persistence, calibration)
        val_calibrated_metrics = compute_gap_metrics(val_cal, val_target, report_horizons)
        calibrated_metrics = compute_gap_metrics(test_cal, test_target, report_horizons)
    return {
        "variant": name,
        "best_val_score": best_val_score,
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "stopped_early": stopped_early,
        "early_stopping_patience": early_stopping_patience,
        "progress_path": str(progress_path),
        "best_checkpoint_path": str(checkpoint_dir / f"{name}_best.pt"),
        "last_checkpoint_path": str(checkpoint_dir / f"{name}_last.pt"),
        "history": history,
        "test_gap": test_metrics,
        "calibration": calibration,
        "val_gap_calibrated": val_calibrated_metrics,
        "test_gap_calibrated": calibrated_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test Chengdu STGAT gap-head architectures.")
    parser.add_argument("--data-dir", default="data/chengdu_dc")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--hist-len", type=int, default=14)
    parser.add_argument("--pred-horizon", type=int, default=4)
    parser.add_argument("--report-horizons-minutes", default="15,30,60")
    parser.add_argument("--max-train-samples", type=int, default=256)
    parser.add_argument("--max-val-samples", type=int, default=128)
    parser.add_argument("--max-test-samples", type=int, default=256)
    parser.add_argument(
        "--train-with-val",
        action="store_true",
        help="Use train+val windows for final training/statistics while still reporting validation and test metrics.",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-st-blocks", type=int, default=2)
    parser.add_argument("--num-gtcn-layers", type=int, default=2)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--adaptive-emb", type=int, default=10)
    parser.add_argument("--adaptive-topk", type=int, default=16)
    parser.add_argument(
        "--disable-node-adaptive",
        action="store_true",
        help="Disable the learned adaptive node-topology branch for GAP/DC ablations.",
    )
    parser.add_argument(
        "--disable-node-fixed-graph",
        action="store_true",
        help="Disable the fixed external graph node branch for GAP/DC ablations.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--aux-weight", type=float, default=0.2)
    parser.add_argument("--score-mape-weight", type=float, default=0.01)
    parser.add_argument("--base-loss", default="mse", choices=["mse", "mae", "smooth_l1"])
    parser.add_argument("--smooth-l1-beta", type=float, default=0.5)
    parser.add_argument("--loss-mae-weight", type=float, default=0.0)
    parser.add_argument("--loss-mape-weight", type=float, default=0.0)
    parser.add_argument(
        "--pred-abs-weight",
        type=float,
        default=0.0,
        help="Penalize raw prediction magnitude to reduce over-prediction under train/test GAP scale drift.",
    )
    parser.add_argument(
        "--pred-abs-step-weights",
        default="",
        help="Optional comma-separated horizon weights for --pred-abs-weight; defaults to --loss-step-weights.",
    )
    parser.add_argument(
        "--small-gap-weight",
        type=float,
        default=0.0,
        help="Extra loss weight for near-zero raw GAP targets, applied without changing the model architecture.",
    )
    parser.add_argument("--small-gap-scale", type=float, default=8.0)
    parser.add_argument(
        "--gap-train-scale-factors",
        default="1",
        help="Comma-separated train-only GAP augmentation scales for the gap_v_to_gap variant.",
    )
    parser.add_argument(
        "--gap-target-transform",
        default="raw",
        choices=["raw", "signed_log"],
        help="Train the GAP head in raw units or a signed-log target space, then invert to raw units for metrics.",
    )
    parser.add_argument(
        "--gap-log-scale",
        type=float,
        default=10.0,
        help="Scale parameter for --gap-target-transform signed_log.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Save a model checkpoint every N epochs in addition to the best validation checkpoint.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop after this many epochs without validation-score improvement; 0 disables early stopping.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-score decrease required to reset early-stopping patience.",
    )
    parser.add_argument(
        "--loss-step-weights",
        default="1,1,1,1",
        help="Comma-separated weights for each forecast step in the gap loss.",
    )
    parser.add_argument(
        "--calibration-mode",
        default="none",
        choices=["none", "stepwise_shrink", "stepwise_affine", "persistence_blend"],
        help="Validation-only output calibration; does not change the neural architecture.",
    )
    parser.add_argument("--calibration-mape-weight", type=float, default=0.05)
    parser.add_argument("--calibration-alpha-grid", default="0.55,0.65,0.75,0.85,0.95,1.05,1.15")
    parser.add_argument("--calibration-tau-grid", default="0,0.5,1,1.5,2,3,4,5,6,8,10")
    parser.add_argument("--calibration-bias-grid", default="-2,-1.5,-1,-0.5,0,0.5,1")
    parser.add_argument(
        "--variants",
        default="dcg_gap_head,gap_v_to_gap",
        help="Comma-separated variants to run: dcg_gap_head,gap_v_to_gap.",
    )
    parser.add_argument(
        "--eval-checkpoints",
        default="",
        help="Comma-separated checkpoint paths or globs to evaluate without training.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Optional checkpoint to load before training continues with a fresh optimizer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"chengdu_gap_arch_smoke_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    demand = np.load(data_dir / "node_demand.npy").astype(np.float32)
    supply = np.load(data_dir / "node_supply.npy").astype(np.float32)
    edge_speeds = np.load(data_dir / "edge_speeds.npy").astype(np.float32)
    time_features = np.load(data_dir / "time_features.npy").astype(np.float32)
    observed = np.load(data_dir / "observed_time_mask.npy").astype(bool)
    adj = np.load(data_dir / "adjacency_matrix.npy").astype(np.float32)
    edge_index = np.load(data_dir / "edge_index.npy").astype(np.int64)
    edge_lengths = edge_lengths_from_matrix(data_dir, edge_index)

    time_meta = load_time_meta_for_training(data_dir, demand.shape[0])
    splits = build_monthly_split_indices(time_meta, args.hist_len, args.pred_horizon)
    splits = filter_split_indices_by_time_mask(splits, observed, args.hist_len, args.pred_horizon)
    train_source_indices = list(splits["train"])
    if args.train_with_val:
        train_source_indices = sorted(set(train_source_indices + list(splits["val"])))
    split_indices = {
        "train": take_evenly(train_source_indices, args.max_train_samples),
        "val": take_evenly(splits["val"], args.max_val_samples),
        "test": take_evenly(splits["test"], args.max_test_samples),
    }
    train_time_mask = build_window_time_mask(demand.shape[0], train_source_indices, args.hist_len, args.pred_horizon)

    time_node = np.broadcast_to(
        time_features[:, None, :],
        (time_features.shape[0], demand.shape[1], time_features.shape[1]),
    ).astype(np.float32)
    base_node = np.concatenate([demand[..., None], supply[..., None], time_node], axis=-1).astype(np.float32)
    norm_stats = build_normalization_stats(base_node, edge_speeds, train_time_mask)
    dc_node = normalize_node_features(base_node, norm_stats)
    speed_norm = normalize_speed_features(edge_speeds, norm_stats, edge_axis=1)

    gap_raw = (demand - supply).astype(np.float32)
    gap_encoded = encode_gap_values(
        gap_raw,
        transform=args.gap_target_transform,
        log_scale=args.gap_log_scale,
    )
    gap_train = gap_encoded[train_time_mask].reshape(-1)
    gap_stats = GapStats(
        mean=float(gap_train.mean()),
        std=float(max(gap_train.std(), 1e-6)),
        transform=args.gap_target_transform,
        log_scale=float(args.gap_log_scale),
    )
    loss_step_weights = parse_float_list(
        args.loss_step_weights,
        expected=args.pred_horizon,
        name="--loss-step-weights",
    )
    pred_abs_step_weights = (
        parse_float_list(args.pred_abs_step_weights, expected=args.pred_horizon, name="--pred-abs-step-weights")
        if str(args.pred_abs_step_weights).strip()
        else list(loss_step_weights)
    )
    calibration_alpha_grid = [float(item) for item in args.calibration_alpha_grid.split(",") if item.strip()]
    calibration_tau_grid = [float(item) for item in args.calibration_tau_grid.split(",") if item.strip()]
    calibration_bias_grid = [float(item) for item in args.calibration_bias_grid.split(",") if item.strip()]
    gap_train_scale_factors = [
        float(item) for item in str(args.gap_train_scale_factors).split(",") if item.strip()
    ]
    if not gap_train_scale_factors or any(item <= 0 for item in gap_train_scale_factors):
        raise ValueError("--gap-train-scale-factors must contain positive values.")
    gap_norm = gap_stats.normalize(gap_raw)
    gap_node = np.concatenate(
        [gap_norm[..., None], np.zeros_like(gap_norm[..., None]), time_node],
        axis=-1,
    ).astype(np.float32)

    slot_minutes = infer_time_slot_minutes(time_meta)
    report_horizons = resolve_report_horizons(
        time_slot_minutes=slot_minutes,
        pred_horizon=args.pred_horizon,
        requested_minutes=parse_report_horizons_minutes(args.report_horizons_minutes),
    )

    dataset_kwargs = {
        "speed_features": speed_norm,
        "demand_target_norm": dc_node[..., 0],
        "supply_target_norm": dc_node[..., 1],
        "gap_target_norm": gap_norm,
        "gap_target_raw": gap_raw,
        "hist_len": args.hist_len,
        "pred_horizon": args.pred_horizon,
        "gap_mean": gap_stats.mean,
        "gap_std": gap_stats.std,
        "gap_transform": gap_stats.transform,
        "gap_log_scale": gap_stats.log_scale,
    }
    dcg_sets = {
        name: GapWindowDataset(indices, node_features=dc_node, **dataset_kwargs)
        for name, indices in split_indices.items()
    }
    gap_sets = {
        name: GapWindowDataset(
            indices,
            node_features=gap_node,
            gap_scale_factors=gap_train_scale_factors if name == "train" else [1.0],
            **dataset_kwargs,
        )
        for name, indices in split_indices.items()
    }

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0, "pin_memory": device.type == "cuda"}
    dcg_loaders = {
        "train": DataLoader(dcg_sets["train"], shuffle=True, **loader_kwargs),
        "val": DataLoader(dcg_sets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(dcg_sets["test"], shuffle=False, **loader_kwargs),
    }
    gap_loaders = {
        "train": DataLoader(gap_sets["train"], shuffle=True, **loader_kwargs),
        "val": DataLoader(gap_sets["val"], shuffle=False, **loader_kwargs),
        "test": DataLoader(gap_sets["test"], shuffle=False, **loader_kwargs),
    }

    model_kwargs = {
        "num_nodes": demand.shape[1],
        "edge_index": torch.from_numpy(edge_index),
        "edge_lengths": torch.from_numpy(edge_lengths),
        "adj_matrix": torch.from_numpy(adj),
        "hidden_dim": args.hidden_dim,
        "num_heads": args.num_heads,
        "num_st_blocks": args.num_st_blocks,
        "num_gtcn_layers": args.num_gtcn_layers,
        "kernel_size": args.kernel_size,
        "pred_horizon": args.pred_horizon,
        "adaptive_emb": args.adaptive_emb,
        "adaptive_topk": args.adaptive_topk,
        "node_use_fixed_graph": not args.disable_node_fixed_graph,
        "node_use_adaptive": not args.disable_node_adaptive,
        "node_feat_dim": dc_node.shape[-1],
        "edge_feat_dim": 2,
    }

    print(f"device={device}")
    print(f"split_counts_full={ {k: len(v) for k, v in splits.items()} }")
    print(f"split_counts_smoke={ {k: len(v) for k, v in split_indices.items()} }")
    print(
        f"gap_stats transform={gap_stats.transform} mean={gap_stats.mean:.6f} "
        f"std={gap_stats.std:.6f} log_scale={gap_stats.log_scale:.6f}"
    )

    requested_variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(requested_variants) - {"dcg_gap_head", "gap_v_to_gap"})
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    if args.eval_checkpoints:
        checkpoint_paths: list[Path] = []
        for raw_item in args.eval_checkpoints.split(","):
            item = raw_item.strip()
            if not item:
                continue
            item_path = Path(item)
            if any(mark in item for mark in "*?[]"):
                parent = item_path.parent if str(item_path.parent) else Path(".")
                checkpoint_paths.extend(sorted(parent.glob(item_path.name)))
            else:
                checkpoint_paths.append(item_path)
        eval_results = []
        for checkpoint_path in checkpoint_paths:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            variant = checkpoint.get("variant", "gap_v_to_gap")
            if variant == "dcg_gap_head":
                model = DcgGapHeadSTGAT(**model_kwargs)
                loaders = dcg_loaders
            else:
                model = GapOnlySTGAT(**model_kwargs)
                loaders = gap_loaders
                variant = "gap_v_to_gap"
            model.load_state_dict(checkpoint["state_dict"])
            model.to(device)
            val_persistence: np.ndarray | None = None
            test_persistence: np.ndarray | None = None
            if args.calibration_mode == "persistence_blend":
                val_pred, val_target, val_persistence = collect_gap_predictions_with_persistence(
                    model,
                    loaders["val"],
                    variant=variant,
                    gap_stats=gap_stats,
                    device=device,
                )
                test_pred, test_target, test_persistence = collect_gap_predictions_with_persistence(
                    model,
                    loaders["test"],
                    variant=variant,
                    gap_stats=gap_stats,
                    device=device,
                )
            else:
                val_pred, val_target = collect_gap_predictions(
                    model,
                    loaders["val"],
                    variant=variant,
                    gap_stats=gap_stats,
                    device=device,
                )
                test_pred, test_target = collect_gap_predictions(
                    model,
                    loaders["test"],
                    variant=variant,
                    gap_stats=gap_stats,
                    device=device,
                )
            val_metrics = compute_gap_metrics(val_pred, val_target, report_horizons)
            test_metrics = compute_gap_metrics(test_pred, test_target, report_horizons)
            calibration: dict[str, Any] | None = None
            val_calibrated_metrics: dict[str, Any] | None = None
            test_calibrated_metrics: dict[str, Any] | None = None
            if args.calibration_mode == "stepwise_shrink":
                calibration = fit_stepwise_shrink_calibration(
                    val_pred,
                    val_target,
                    mape_weight=args.calibration_mape_weight,
                    alpha_grid=calibration_alpha_grid,
                    tau_grid=calibration_tau_grid,
                )
                val_cal = apply_stepwise_shrink_calibration(val_pred, calibration)
                test_cal = apply_stepwise_shrink_calibration(test_pred, calibration)
                val_calibrated_metrics = compute_gap_metrics(val_cal, val_target, report_horizons)
                test_calibrated_metrics = compute_gap_metrics(test_cal, test_target, report_horizons)
            elif args.calibration_mode == "stepwise_affine":
                calibration = fit_stepwise_affine_calibration(
                    val_pred,
                    val_target,
                    mape_weight=args.calibration_mape_weight,
                    alpha_grid=calibration_alpha_grid,
                    bias_grid=calibration_bias_grid,
                )
                val_cal = apply_stepwise_affine_calibration(val_pred, calibration)
                test_cal = apply_stepwise_affine_calibration(test_pred, calibration)
                val_calibrated_metrics = compute_gap_metrics(val_cal, val_target, report_horizons)
                test_calibrated_metrics = compute_gap_metrics(test_cal, test_target, report_horizons)
            elif args.calibration_mode == "persistence_blend":
                if val_persistence is None or test_persistence is None:
                    raise RuntimeError("Persistence calibration requires persistence arrays.")
                calibration = fit_persistence_blend_calibration(
                    val_pred,
                    val_persistence,
                    val_target,
                    mape_weight=args.calibration_mape_weight,
                    alpha_grid=calibration_alpha_grid,
                )
                val_cal = apply_persistence_blend_calibration(val_pred, val_persistence, calibration)
                test_cal = apply_persistence_blend_calibration(test_pred, test_persistence, calibration)
                val_calibrated_metrics = compute_gap_metrics(val_cal, val_target, report_horizons)
                test_calibrated_metrics = compute_gap_metrics(test_cal, test_target, report_horizons)
            eval_results.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "variant": variant,
                    "epoch": checkpoint.get("epoch"),
                    "checkpoint_val_score": checkpoint.get("val_score"),
                    "val_gap": val_metrics,
                    "test_gap": test_metrics,
                    "calibration": calibration,
                    "val_gap_calibrated": val_calibrated_metrics,
                    "test_gap_calibrated": test_calibrated_metrics,
                }
            )
        payload = {
            "config": vars(args),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "split_counts_full": {k: len(v) for k, v in splits.items()},
            "split_counts_smoke": {k: len(v) for k, v in split_indices.items()},
            "gap_stats": {"mean": gap_stats.mean, "std": gap_stats.std},
            "report_horizons": report_horizons,
            "results": eval_results,
        }
        out_path = output_dir / "checkpoint_eval_results.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\nCheckpoint Eval Summary")
        for result in eval_results:
            report = (
                result["test_gap_calibrated"]["report"]
                if result.get("test_gap_calibrated")
                else result["test_gap"]["report"]
            )
            print(f"{Path(result['checkpoint']).name} epoch={result.get('epoch')}")
            if result.get("calibration"):
                print(f"  calibration={result['calibration']}")
            for label, metrics in report.items():
                print(
                    f"  {label}: RMSE={metrics['rmse']:.6f} "
                    f"MAE={metrics['mae']:.6f} MAPE={metrics['mape']:.6f}"
                )
        print(out_path)
        return

    results = []
    if "dcg_gap_head" in requested_variants:
        dcg_model = DcgGapHeadSTGAT(**model_kwargs)
        if args.resume_checkpoint:
            checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
            dcg_model.load_state_dict(checkpoint["state_dict"])
        results.append(
            train_variant(
                name="dcg_gap_head",
                model=dcg_model,
                train_loader=dcg_loaders["train"],
                val_loader=dcg_loaders["val"],
                test_loader=dcg_loaders["test"],
                gap_stats=gap_stats,
                report_horizons=report_horizons,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                aux_weight=args.aux_weight,
                score_mape_weight=args.score_mape_weight,
                loss_mae_weight=args.loss_mae_weight,
                loss_mape_weight=args.loss_mape_weight,
                pred_abs_weight=args.pred_abs_weight,
                pred_abs_step_weights=pred_abs_step_weights,
                loss_step_weights=loss_step_weights,
                base_loss=args.base_loss,
                smooth_l1_beta=args.smooth_l1_beta,
                small_gap_weight=args.small_gap_weight,
                small_gap_scale=args.small_gap_scale,
                calibration_mode=args.calibration_mode,
                calibration_mape_weight=args.calibration_mape_weight,
                calibration_alpha_grid=calibration_alpha_grid,
                calibration_tau_grid=calibration_tau_grid,
                calibration_bias_grid=calibration_bias_grid,
                output_dir=output_dir,
                checkpoint_every=args.checkpoint_every,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        )
    if "gap_v_to_gap" in requested_variants:
        gap_model = GapOnlySTGAT(**model_kwargs)
        if args.resume_checkpoint:
            checkpoint = torch.load(args.resume_checkpoint, map_location="cpu")
            gap_model.load_state_dict(checkpoint["state_dict"])
        results.append(
            train_variant(
                name="gap_v_to_gap",
                model=gap_model,
                train_loader=gap_loaders["train"],
                val_loader=gap_loaders["val"],
                test_loader=gap_loaders["test"],
                gap_stats=gap_stats,
                report_horizons=report_horizons,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                aux_weight=args.aux_weight,
                score_mape_weight=args.score_mape_weight,
                loss_mae_weight=args.loss_mae_weight,
                loss_mape_weight=args.loss_mape_weight,
                pred_abs_weight=args.pred_abs_weight,
                pred_abs_step_weights=pred_abs_step_weights,
                loss_step_weights=loss_step_weights,
                base_loss=args.base_loss,
                smooth_l1_beta=args.smooth_l1_beta,
                small_gap_weight=args.small_gap_weight,
                small_gap_scale=args.small_gap_scale,
                calibration_mode=args.calibration_mode,
                calibration_mape_weight=args.calibration_mape_weight,
                calibration_alpha_grid=calibration_alpha_grid,
                calibration_tau_grid=calibration_tau_grid,
                calibration_bias_grid=calibration_bias_grid,
                output_dir=output_dir,
                checkpoint_every=args.checkpoint_every,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        )

    payload = {
        "config": vars(args),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "split_counts_full": {k: len(v) for k, v in splits.items()},
        "split_counts_smoke": {k: len(v) for k, v in split_indices.items()},
        "gap_stats": {"mean": gap_stats.mean, "std": gap_stats.std},
        "report_horizons": report_horizons,
        "results": results,
    }
    out_path = output_dir / "chengdu_gap_arch_smoke_results.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSummary")
    for result in results:
        print(result["variant"])
        report = (
            result["test_gap_calibrated"]["report"]
            if result.get("test_gap_calibrated")
            else result["test_gap"]["report"]
        )
        if result.get("calibration"):
            print(f"  calibration={result['calibration']}")
        for label, metrics in report.items():
            print(
                f"  {label}: RMSE={metrics['rmse']:.6f} "
                f"MAE={metrics['mae']:.6f} MAPE={metrics['mape']:.6f}"
            )
    print(out_path)


if __name__ == "__main__":
    main()
