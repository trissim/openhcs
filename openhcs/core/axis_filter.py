"""Nominal axis-filter records shared by runtime compilation phases."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from openhcs.core.config import WellFilterConfig, WellFilterMode


WellFilterValue: TypeAlias = list[str] | str | int | None


@dataclass(frozen=True, slots=True)
class StepAxisFilterResolution:
    """Resolved axis filter for one step-level filtering config root."""

    resolved_axis_values: frozenset[str]
    filter_mode: WellFilterMode
    original_filter: WellFilterValue


@dataclass(frozen=True, slots=True)
class StepAxisFilterSet:
    """Resolved axis filters keyed by the config type that declared them."""

    resolutions_by_config_type: Mapping[
        type["WellFilterConfig"], StepAxisFilterResolution
    ]

    @classmethod
    def empty(cls) -> "StepAxisFilterSet":
        return cls({})

    def __len__(self) -> int:
        return len(self.resolutions_by_config_type)

    def allows(self, config: "WellFilterConfig", axis_id: str | None) -> bool:
        resolution = self.resolution_for(config)
        if resolution is None:
            return True
        return axis_id in resolution.resolved_axis_values

    def resolution_for(
        self,
        config: "WellFilterConfig",
    ) -> StepAxisFilterResolution | None:
        from objectstate import get_base_type_for_lazy

        config_type = get_base_type_for_lazy(type(config)) or type(config)
        config_mro = config_type.mro()
        matches: list[tuple[int, StepAxisFilterResolution]] = []
        for owner_type, resolution in self.resolutions_by_config_type.items():
            owner_base = get_base_type_for_lazy(owner_type) or owner_type
            if not isinstance(owner_base, type):
                continue
            if owner_base not in config_mro:
                continue
            matches.append((config_mro.index(owner_base), resolution))
        if not matches:
            return None
        matches.sort(key=lambda item: item[0])
        return matches[0][1]


StepAxisFilterMap: TypeAlias = dict[int, StepAxisFilterSet]


def step_axis_allows_config(
    step_axis_filters: StepAxisFilterMap,
    *,
    step_index: int,
    config: "WellFilterConfig",
    axis_id: str | None,
) -> bool:
    """Return whether one resolved step config applies to the runtime axis."""
    return step_axis_filters.get(step_index, StepAxisFilterSet.empty()).allows(
        config,
        axis_id,
    )
