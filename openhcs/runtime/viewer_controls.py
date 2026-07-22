"""Nominal viewer control-message values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self, TypeAlias


ViewerScalar: TypeAlias = str | int | float | bool | None
ViewerControlWireValue: TypeAlias = (
    ViewerScalar
    | tuple["ViewerControlWireValue", ...]
    | list["ViewerControlWireValue"]
    | dict[str, "ViewerControlWireValue"]
)
ViewerPayloadAxisIndices: TypeAlias = tuple[int, ...] | dict[str, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerPayloadControlOptions:
    """Formal payload-inspection controls shared by agent and viewer runtimes."""

    route_key: str | None = None
    axis_indices: ViewerPayloadAxisIndices | None = None
    include_array_values: bool = False
    max_array_elements: int = 4096
    array_slices: tuple[tuple[int, int], ...] | None = None
    include_shape_payloads: bool = True
    max_shape_payloads: int = 256

    def __post_init__(self) -> None:
        if self.route_key is not None and (
            not isinstance(self.route_key, str) or not self.route_key
        ):
            raise ValueError("Viewer payload route_key must be a non-empty string.")
        if self.axis_indices is not None:
            self._validate_axis_indices(self.axis_indices)
        if not isinstance(self.include_array_values, bool):
            raise TypeError("Viewer payload include_array_values must be a bool.")
        self._validate_nonnegative_int(
            self.max_array_elements,
            "max_array_elements",
        )
        if self.array_slices is not None:
            self._validate_array_slices(self.array_slices)
        if not isinstance(self.include_shape_payloads, bool):
            raise TypeError("Viewer payload include_shape_payloads must be a bool.")
        self._validate_nonnegative_int(
            self.max_shape_payloads,
            "max_shape_payloads",
        )

    @classmethod
    def from_overrides(
        cls,
        *,
        route_key: str | None = None,
        axis_indices: tuple[int, ...] | Mapping[str, int] | None = None,
        include_array_values: bool | None = None,
        max_array_elements: int | None = None,
        array_slices: tuple[tuple[int, int], ...] | None = None,
        include_shape_payloads: bool | None = None,
        max_shape_payloads: int | None = None,
    ) -> Self:
        defaults = cls()
        return cls(
            route_key=route_key,
            axis_indices=(
                dict(axis_indices) if isinstance(axis_indices, Mapping) else axis_indices
            ),
            include_array_values=(
                defaults.include_array_values
                if include_array_values is None
                else include_array_values
            ),
            max_array_elements=(
                defaults.max_array_elements
                if max_array_elements is None
                else max_array_elements
            ),
            array_slices=array_slices,
            include_shape_payloads=(
                defaults.include_shape_payloads
                if include_shape_payloads is None
                else include_shape_payloads
            ),
            max_shape_payloads=(
                defaults.max_shape_payloads
                if max_shape_payloads is None
                else max_shape_payloads
            ),
        )

    @staticmethod
    def _validate_axis_indices(value: ViewerPayloadAxisIndices) -> None:
        if isinstance(value, tuple):
            for index in value:
                ViewerPayloadControlOptions._validate_nonnegative_int(
                    index,
                    "axis_indices",
                )
            return
        if not isinstance(value, Mapping):
            raise TypeError("Viewer payload axis_indices must be a tuple or mapping.")
        for axis_name, index in value.items():
            if not isinstance(axis_name, str) or not axis_name:
                raise ValueError(
                    "Viewer payload axis_indices keys must be non-empty strings."
                )
            ViewerPayloadControlOptions._validate_nonnegative_int(
                index,
                f"axis_indices[{axis_name!r}]",
            )

    @staticmethod
    def _validate_array_slices(value: tuple[tuple[int, int], ...]) -> None:
        if not isinstance(value, tuple):
            raise TypeError("Viewer payload array_slices must be a tuple.")
        for slice_pair in value:
            if not isinstance(slice_pair, tuple) or len(slice_pair) != 2:
                raise ValueError(
                    "Viewer payload array_slices entries require start and stop."
                )
            start, stop = slice_pair
            ViewerPayloadControlOptions._validate_nonnegative_int(
                start,
                "array_slices start",
            )
            ViewerPayloadControlOptions._validate_nonnegative_int(
                stop,
                "array_slices stop",
            )
            if stop < start:
                raise ValueError(
                    "Viewer payload array_slices stop must not precede start."
                )

    @staticmethod
    def _validate_nonnegative_int(value: object, field_name: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Viewer payload {field_name} must be an integer.")
        if value < 0:
            raise ValueError(f"Viewer payload {field_name} must be nonnegative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerStateControlOptions:
    """Formal state-inspection controls shared by agent and viewer runtimes."""

    route_key: str | None = None
    include_component_values: bool = True
    max_component_values_per_layer: int | None = None
    include_payload_summaries: bool = True
    max_payload_summaries_per_layer: int | None = None

    def __post_init__(self) -> None:
        if self.route_key is not None and (
            not isinstance(self.route_key, str) or not self.route_key
        ):
            raise ValueError("Viewer state route_key must be a non-empty string.")
        if not isinstance(self.include_component_values, bool):
            raise TypeError("Viewer state include_component_values must be a bool.")
        self._validate_optional_limit(
            self.max_component_values_per_layer,
            "max_component_values_per_layer",
        )
        if not isinstance(self.include_payload_summaries, bool):
            raise TypeError("Viewer state include_payload_summaries must be a bool.")
        self._validate_optional_limit(
            self.max_payload_summaries_per_layer,
            "max_payload_summaries_per_layer",
        )

    @classmethod
    def from_overrides(
        cls,
        *,
        route_key: str | None = None,
        include_component_values: bool | None = None,
        max_component_values_per_layer: int | None = None,
        include_payload_summaries: bool | None = None,
        max_payload_summaries_per_layer: int | None = None,
    ) -> Self:
        defaults = cls()
        return cls(
            route_key=route_key,
            include_component_values=(
                defaults.include_component_values
                if include_component_values is None
                else include_component_values
            ),
            max_component_values_per_layer=max_component_values_per_layer,
            include_payload_summaries=(
                defaults.include_payload_summaries
                if include_payload_summaries is None
                else include_payload_summaries
            ),
            max_payload_summaries_per_layer=max_payload_summaries_per_layer,
        )

    @staticmethod
    def _validate_optional_limit(value: object, field_name: str) -> None:
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Viewer state {field_name} must be an integer.")
        if value < 0:
            raise ValueError(f"Viewer state {field_name} must be nonnegative.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerNavigationControlOptions:
    """Formal viewer navigation controls shared by agent and viewer runtimes."""

    route_key: str
    axis_indices: Mapping[str, int] = field(default_factory=dict)
    visible: bool | None = None
    selected: bool | None = None
    data_index: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.route_key, str) or not self.route_key:
            raise ValueError("Viewer navigation route_key must be a non-empty string.")
        if not isinstance(self.axis_indices, Mapping):
            raise TypeError("Viewer navigation axis_indices must be a mapping.")
        for axis_name, index in self.axis_indices.items():
            if not isinstance(axis_name, str) or not axis_name:
                raise ValueError(
                    "Viewer navigation axis_indices keys must be non-empty strings."
                )
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(
                    f"Viewer navigation index for axis {axis_name!r} must be an integer."
                )
            if index < 0:
                raise ValueError(
                    f"Viewer navigation index for axis {axis_name!r} must be nonnegative."
                )
        for field_name, value in (("visible", self.visible), ("selected", self.selected)):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"Viewer navigation {field_name} must be a bool.")
        if self.data_index is not None:
            if isinstance(self.data_index, bool) or not isinstance(
                self.data_index,
                int,
            ):
                raise TypeError("Viewer navigation data_index must be an integer.")
            if self.data_index < 0:
                raise ValueError("Viewer navigation data_index must be nonnegative.")
            if self.visible is False:
                raise ValueError(
                    "Viewer navigation data_index cannot target a hidden layer."
                )
            if self.selected is False:
                raise ValueError(
                    "Viewer navigation data_index cannot target a deselected layer."
                )

    @classmethod
    def from_overrides(
        cls,
        *,
        route_key: str,
        axis_indices: Mapping[str, int] | None = None,
        visible: bool | None = None,
        selected: bool | None = None,
        data_index: int | None = None,
    ) -> Self:
        return cls(
            route_key=route_key,
            axis_indices=dict(axis_indices or {}),
            visible=visible,
            selected=selected,
            data_index=data_index,
        )
