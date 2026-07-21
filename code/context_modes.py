"""Context ablation (M0/M1/M2) configuration and u-label helpers.

グリッド一括（12 run + VAL/TEST）
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any

import numpy as np
import torch

# Default event/time config (matches DKP_RF_inf.py CONFIG)
DEFAULT_CONTEXT_CONFIG: dict[str, Any] = {
    "holidays": [20240928, 20240929, 20251122, 20251123],
    "night_start": 19,
    "night_end": 2,
    "events": [
        (20240929, 9, 16, [14]),
        (20251122, 10, 19, [2, 11]),
        (20251123, 10, 16, [2]),
    ],
}


class ConditionFeature(str, Enum):
    NONE = "none"
    EVENT = "event"
    TIMEZONE = "timezone"
    STAY = "stay"


class ConditionMode(str, Enum):
    M0 = "M0"
    M1 = "M1"
    M2 = "M2"


@dataclass
class ContextAblationConfig:
    target: str = "none"
    mode: str = "M1"
    stay_u_threshold: int = 3

    def __post_init__(self) -> None:
        self.target = str(self.target).lower()
        self.mode = str(self.mode).upper()
        if self.target not in {f.value for f in ConditionFeature}:
            raise ValueError(f"Unknown target feature: {self.target}")
        if self.mode not in {m.value for m in ConditionMode}:
            raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def is_active(self) -> bool:
        return self.target != ConditionFeature.NONE.value

    @property
    def uses_dual_A(self) -> bool:
        return self.is_active and self.mode == ConditionMode.M2.value

    def feature_mode(self, name: str) -> str:
        """Return effective mode for a context feature embedding."""
        if not self.is_active or name != self.target:
            return ConditionMode.M1.value
        return self.mode

    def should_zero_embed(self, name: str) -> bool:
        fm = self.feature_mode(name)
        return fm in (ConditionMode.M0.value, ConditionMode.M2.value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ContextAblationConfig:
        if d is None:
            return cls()
        return cls(
            target=d.get("target", "none"),
            mode=d.get("mode", "M1"),
            stay_u_threshold=int(d.get("stay_u_threshold", 3)),
        )


def _hour_from_timestamp(timestamp_int: int | torch.Tensor) -> int:
    if isinstance(timestamp_int, torch.Tensor):
        timestamp_int = int(timestamp_int.item())
    return (int(timestamp_int) // 100) % 100


def _date_from_timestamp(timestamp_int: int | torch.Tensor) -> int:
    if isinstance(timestamp_int, torch.Tensor):
        timestamp_int = int(timestamp_int.item())
    return int(timestamp_int) // 10000


def is_event_time_window(timestamp_int: int, config: dict[str, Any] | None = None) -> bool:
    """Return True if start_time falls in any configured event time window."""
    cfg = config or DEFAULT_CONTEXT_CONFIG
    date_int = _date_from_timestamp(timestamp_int)
    hour = _hour_from_timestamp(timestamp_int)
    for ev_date, ev_start, ev_end, _ev_nodes in cfg.get("events", []):
        if date_int == ev_date and ev_start <= hour < ev_end:
            return True
    return False


def is_night_time(timestamp_int: int, config: dict[str, Any] | None = None) -> bool:
    cfg = config or DEFAULT_CONTEXT_CONFIG
    hour = _hour_from_timestamp(timestamp_int)
    night_start = int(cfg.get("night_start", 19))
    night_end = int(cfg.get("night_end", 2))
    return hour >= night_start or hour < night_end


def compute_u_event(
    start_times: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """u=1 if start_time is in an event time window. Returns [B] float tensor."""
    u = torch.zeros(start_times.shape[0], dtype=torch.float32, device=start_times.device)
    for i in range(start_times.shape[0]):
        if is_event_time_window(int(start_times[i].item()), config):
            u[i] = 1.0
    return u


def compute_u_timezone(
    start_times: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """u=1 if night. Returns [B] float tensor."""
    u = torch.zeros(start_times.shape[0], dtype=torch.float32, device=start_times.device)
    for i in range(start_times.shape[0]):
        if is_night_time(int(start_times[i].item()), config):
            u[i] = 1.0
    return u


def compute_u_stay(
    prefix_stay_counts: torch.Tensor,
    prefix_mask: torch.Tensor | None,
    threshold: int = 3,
) -> torch.Tensor:
    """u=1 if max valid prefix stay count >= threshold. Returns [B] float tensor."""
    counts = prefix_stay_counts.float()
    if prefix_mask is not None:
        counts = counts.masked_fill(prefix_mask, 0.0)
    max_counts = counts.max(dim=1).values
    return (max_counts >= float(threshold)).float()


def compute_u(
    config: ContextAblationConfig,
    prefix_stay_counts: torch.Tensor,
    prefix_times: torch.Tensor | None = None,
    prefix_mask: torch.Tensor | None = None,
    context_config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute u labels [B] for M2; returns zeros for non-M2 modes."""
    batch_size = prefix_stay_counts.shape[0]
    device = prefix_stay_counts.device
    if not config.uses_dual_A:
        return torch.zeros(batch_size, dtype=torch.float32, device=device)

    target = config.target
    if target == ConditionFeature.EVENT.value:
        if prefix_times is None:
            raise ValueError("prefix_times required for event u computation")
        return compute_u_event(prefix_times, context_config)
    if target == ConditionFeature.TIMEZONE.value:
        if prefix_times is None:
            raise ValueError("prefix_times required for timezone u computation")
        return compute_u_timezone(prefix_times, context_config)
    if target == ConditionFeature.STAY.value:
        return compute_u_stay(prefix_stay_counts, prefix_mask, config.stay_u_threshold)

    return torch.zeros(batch_size, dtype=torch.float32, device=device)


def compute_u_numpy(
    config: ContextAblationConfig,
    prefix_stay_counts: np.ndarray,
    start_time: int,
    context_config: dict[str, Any] | None = None,
) -> int:
    """Scalar u for inference (single sample)."""
    if not config.uses_dual_A:
        return 0
    target = config.target
    if target == ConditionFeature.EVENT.value:
        return 1 if is_event_time_window(start_time, context_config) else 0
    if target == ConditionFeature.TIMEZONE.value:
        return 1 if is_night_time(start_time, context_config) else 0
    if target == ConditionFeature.STAY.value:
        max_count = int(np.max(prefix_stay_counts)) if prefix_stay_counts.size else 0
        return 1 if max_count >= config.stay_u_threshold else 0
    return 0


def apply_u_shuffle(
    u: torch.Tensor,
    sample_indices: torch.Tensor,
    u_perm: torch.Tensor,
) -> torch.Tensor:
    """Replace u[i] with u_perm[sample_indices[i]] (epoch-fixed permutation control)."""
    permuted = u_perm[sample_indices.to(u_perm.device)].to(u.device)
    return permuted
