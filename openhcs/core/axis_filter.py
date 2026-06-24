"""Nominal axis-filter records shared by runtime compilation phases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from openhcs.core.config import WellFilterMode


WellFilterValue: TypeAlias = list[str] | str | int | None


@dataclass(frozen=True, slots=True)
class StepAxisFilterResolution:
    """Resolved axis filter for one step-level filtering config root."""

    resolved_axis_values: frozenset[str]
    filter_mode: WellFilterMode
    original_filter: WellFilterValue


StepAxisFilterMap: TypeAlias = dict[int, dict[str, StepAxisFilterResolution]]
