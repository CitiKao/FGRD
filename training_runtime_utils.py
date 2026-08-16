from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F


OPTIMIZER_CHOICES = ("adam", "adamw")
V_LOSS_CHOICES = ("mse", "rmse", "huber", "charbonnier", "mse_huber")


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = list(parameters)
    if not params:
        raise ValueError("No parameters were provided to build_optimizer().")

    name = optimizer_name.strip().lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(
        f"Unsupported optimizer '{optimizer_name}'. Expected one of: {', '.join(OPTIMIZER_CHOICES)}."
    )


def maybe_apply_linear_warmup(
    optimizer: torch.optim.Optimizer,
    *,
    base_lr: float,
    global_step: int,
    warmup_steps: int,
) -> float:
    if warmup_steps <= 0 or global_step > warmup_steps:
        return float(optimizer.param_groups[0]["lr"])

    warmup_lr = float(base_lr) * float(global_step) / float(max(warmup_steps, 1))
    for group in optimizer.param_groups:
        group["lr"] = warmup_lr
    return warmup_lr


def masked_regression_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    target_mask: torch.Tensor | None = None,
    element_weight: torch.Tensor | None = None,
    huber_delta: float = 1.0,
    charbonnier_eps: float = 1e-3,
    hybrid_huber_weight: float = 0.2,
) -> torch.Tensor:
    name = loss_name.strip().lower()
    error = prediction - target

    if name in {"mse", "rmse"}:
        per_item = torch.square(error)
    elif name == "huber":
        per_item = F.huber_loss(
            prediction,
            target,
            reduction="none",
            delta=float(huber_delta),
        )
    elif name == "mse_huber":
        huber_weight = min(max(float(hybrid_huber_weight), 0.0), 1.0)
        mse_item = torch.square(error)
        huber_item = F.huber_loss(
            prediction,
            target,
            reduction="none",
            delta=float(huber_delta),
        )
        per_item = (1.0 - huber_weight) * mse_item + huber_weight * huber_item
    elif name == "charbonnier":
        eps = float(charbonnier_eps)
        per_item = torch.sqrt(torch.square(error) + eps * eps) - eps
    else:
        raise ValueError(
            f"Unsupported loss '{loss_name}'. Expected one of: {', '.join(V_LOSS_CHOICES)}."
        )

    weight = None
    if element_weight is not None:
        weight = element_weight.to(device=prediction.device, dtype=prediction.dtype)
        while weight.ndim < per_item.ndim:
            weight = weight.unsqueeze(0)
        weight = torch.broadcast_to(weight, per_item.shape)

    if target_mask is None:
        if weight is None:
            reduced = per_item.mean()
        else:
            reduced = torch.sum(per_item * weight) / torch.clamp(weight.sum(), min=1.0)
        return torch.sqrt(torch.clamp_min(reduced, 0.0)) if name == "rmse" else reduced

    mask = target_mask.to(dtype=prediction.dtype)
    if weight is not None:
        mask = mask * weight
    valid = torch.clamp(mask.sum(), min=1.0)
    reduced = torch.sum(per_item * mask) / valid
    return torch.sqrt(torch.clamp_min(reduced, 0.0)) if name == "rmse" else reduced
