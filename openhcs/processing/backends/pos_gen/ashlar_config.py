"""Shared request records for Ashlar position generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


@dataclass(frozen=True, slots=True)
class AshlarAlignmentConfig:
    """Alignment parameters shared by CPU and GPU Ashlar implementations."""

    pixel_size: float = 1.0
    max_shift: float = 30.0
    stitch_alpha: float = 0.05
    max_error: float | None = None
    randomize: bool = False
    verbose: bool = False
    upsample_factor: int = 50
    permutation_upsample: int = 1
    permutation_samples: int = 1000
    min_permutation_samples: int = 10
    max_permutation_tries: int = 100
    window_size_factor: float = 0.15


@dataclass(frozen=True, slots=True)
class AshlarPositionRequest:
    """Public Ashlar position-generation request normalized for implementation."""

    image_stack: Any
    grid_dimensions: Tuple[int, int]
    overlap_ratio: float
    alignment: AshlarAlignmentConfig
