"""Lightweight viewer control-message declarations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Self, TypeAlias


ViewerScalar: TypeAlias = str | int | float | bool | None
ViewerControlWireValue: TypeAlias = (
    ViewerScalar
    | tuple["ViewerControlWireValue", ...]
    | list["ViewerControlWireValue"]
    | dict[str, "ViewerControlWireValue"]
)
ViewerPayloadAxisIndices: TypeAlias = tuple[int, ...] | dict[str, int]


class ViewerPayloadControlField(Enum):
    """Shared fields for viewer payload-extraction control messages."""

    ROUTE_KEY = "route_key"
    AXIS_INDICES = "axis_indices"
    INCLUDE_ARRAY_VALUES = "include_array_values"
    MAX_ARRAY_ELEMENTS = "max_array_elements"
    ARRAY_SLICES = "array_slices"
    INCLUDE_SHAPE_PAYLOADS = "include_shape_payloads"
    MAX_SHAPE_PAYLOADS = "max_shape_payloads"


class ViewerStateControlField(Enum):
    """Shared fields for viewer state projection control messages."""

    ROUTE_KEY = "route_key"
    INCLUDE_COMPONENT_VALUES = "include_component_values"
    MAX_COMPONENT_VALUES_PER_LAYER = "max_component_values_per_layer"
    INCLUDE_PAYLOAD_SUMMARIES = "include_payload_summaries"
    MAX_PAYLOAD_SUMMARIES_PER_LAYER = "max_payload_summaries_per_layer"


class ViewerNavigationControlField(Enum):
    """Shared fields for viewer navigation and layer-state control messages."""

    ROUTE_KEY = "route_key"
    AXIS_INDICES = "axis_indices"
    VISIBLE = "visible"
    SELECTED = "selected"


VIEWER_PAYLOAD_INCLUDE_ARRAY_VALUES_DEFAULT = False
VIEWER_PAYLOAD_MAX_ARRAY_ELEMENTS_DEFAULT = 4096
VIEWER_PAYLOAD_INCLUDE_SHAPE_PAYLOADS_DEFAULT = True
VIEWER_PAYLOAD_MAX_SHAPE_PAYLOADS_DEFAULT = 256
VIEWER_STATE_INCLUDE_COMPONENT_VALUES_DEFAULT = True
VIEWER_STATE_INCLUDE_PAYLOAD_SUMMARIES_DEFAULT = True


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerPayloadControlOptions:
    """Formal payload-inspection controls shared by agent and viewer runtimes."""

    route_key: str | None = None
    axis_indices: ViewerPayloadAxisIndices | None = None
    include_array_values: bool = VIEWER_PAYLOAD_INCLUDE_ARRAY_VALUES_DEFAULT
    max_array_elements: int = VIEWER_PAYLOAD_MAX_ARRAY_ELEMENTS_DEFAULT
    array_slices: tuple[tuple[int, int], ...] | None = None
    include_shape_payloads: bool = VIEWER_PAYLOAD_INCLUDE_SHAPE_PAYLOADS_DEFAULT
    max_shape_payloads: int = VIEWER_PAYLOAD_MAX_SHAPE_PAYLOADS_DEFAULT

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
            axis_indices=cls._optional_axis_indices_value(
                axis_indices,
                ViewerPayloadControlField.AXIS_INDICES,
            ),
            include_array_values=cls._resolved_bool_override(
                ViewerPayloadControlField.INCLUDE_ARRAY_VALUES,
                include_array_values,
                defaults.include_array_values,
            ),
            max_array_elements=cls._resolved_nonnegative_int_override(
                ViewerPayloadControlField.MAX_ARRAY_ELEMENTS,
                max_array_elements,
                defaults.max_array_elements,
            ),
            array_slices=cls._optional_slice_tuple_value(
                array_slices,
                ViewerPayloadControlField.ARRAY_SLICES,
            ),
            include_shape_payloads=cls._resolved_bool_override(
                ViewerPayloadControlField.INCLUDE_SHAPE_PAYLOADS,
                include_shape_payloads,
                defaults.include_shape_payloads,
            ),
            max_shape_payloads=cls._resolved_nonnegative_int_override(
                ViewerPayloadControlField.MAX_SHAPE_PAYLOADS,
                max_shape_payloads,
                defaults.max_shape_payloads,
            ),
        )

    @classmethod
    def from_wire_payload(
        cls,
        payload: Mapping[str, ViewerControlWireValue],
    ) -> Self:
        return cls(
            route_key=cls._optional_str(
                payload,
                ViewerPayloadControlField.ROUTE_KEY,
            ),
            axis_indices=cls._optional_axis_indices(
                payload,
                ViewerPayloadControlField.AXIS_INDICES,
            ),
            include_array_values=cls._optional_bool(
                payload,
                ViewerPayloadControlField.INCLUDE_ARRAY_VALUES,
                VIEWER_PAYLOAD_INCLUDE_ARRAY_VALUES_DEFAULT,
            ),
            max_array_elements=cls._optional_nonnegative_int(
                payload,
                ViewerPayloadControlField.MAX_ARRAY_ELEMENTS,
                VIEWER_PAYLOAD_MAX_ARRAY_ELEMENTS_DEFAULT,
            ),
            array_slices=cls._optional_slice_tuple(
                payload,
                ViewerPayloadControlField.ARRAY_SLICES,
            ),
            include_shape_payloads=cls._optional_bool(
                payload,
                ViewerPayloadControlField.INCLUDE_SHAPE_PAYLOADS,
                VIEWER_PAYLOAD_INCLUDE_SHAPE_PAYLOADS_DEFAULT,
            ),
            max_shape_payloads=cls._optional_nonnegative_int(
                payload,
                ViewerPayloadControlField.MAX_SHAPE_PAYLOADS,
                VIEWER_PAYLOAD_MAX_SHAPE_PAYLOADS_DEFAULT,
            ),
        )

    def to_wire_payload(self) -> dict[str, ViewerControlWireValue]:
        payload: dict[str, ViewerControlWireValue] = {
            ViewerPayloadControlField.INCLUDE_ARRAY_VALUES.value: (
                self.include_array_values
            ),
            ViewerPayloadControlField.MAX_ARRAY_ELEMENTS.value: (
                self.max_array_elements
            ),
            ViewerPayloadControlField.INCLUDE_SHAPE_PAYLOADS.value: (
                self.include_shape_payloads
            ),
            ViewerPayloadControlField.MAX_SHAPE_PAYLOADS.value: (
                self.max_shape_payloads
            ),
        }
        if self.route_key is not None:
            payload[ViewerPayloadControlField.ROUTE_KEY.value] = self.route_key
        if self.axis_indices is not None:
            payload[ViewerPayloadControlField.AXIS_INDICES.value] = self.axis_indices
        if self.array_slices is not None:
            payload[ViewerPayloadControlField.ARRAY_SLICES.value] = (
                self.array_slices
            )
        return payload

    @staticmethod
    def _optional_str(
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerPayloadControlField,
    ) -> str | None:
        if field.value not in payload:
            return None
        value = payload[field.value]
        if value is None:
            return None
        if not isinstance(value, str):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a string."
            )
        return value

    @staticmethod
    def _optional_bool(
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerPayloadControlField,
        fallback: bool,
    ) -> bool:
        if field.value not in payload:
            return fallback
        value = payload[field.value]
        if not isinstance(value, bool):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a bool."
            )
        return value

    @classmethod
    def _optional_axis_indices(
        cls,
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerPayloadControlField,
    ) -> ViewerPayloadAxisIndices | None:
        if field.value not in payload:
            return None
        value = payload[field.value]
        if value is None:
            return None
        if isinstance(value, Mapping):
            return cls._semantic_axis_indices(value, field)
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a sequence or mapping."
            )
        return cls._optional_int_tuple_value(tuple(value), field)

    @classmethod
    def _optional_axis_indices_value(
        cls,
        value: tuple[int, ...] | Mapping[str, int] | None,
        field: ViewerPayloadControlField,
    ) -> ViewerPayloadAxisIndices | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            return cls._semantic_axis_indices(value, field)
        return cls._optional_int_tuple_value(value, field)

    @staticmethod
    def _optional_int_tuple_value(
        value: tuple[int, ...] | None,
        field: ViewerPayloadControlField,
    ) -> tuple[int, ...] | None:
        if value is None:
            return None
        if not isinstance(value, tuple):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a tuple."
            )
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise TypeError(
                    f"Viewer payload control field {field.value!r} must contain integers."
                )
            if item < 0:
                raise ValueError(
                    f"Viewer payload control field {field.value!r} must be nonnegative."
                )
        return value

    @staticmethod
    def _semantic_axis_indices(
        value: Mapping[object, object],
        field: ViewerPayloadControlField,
    ) -> dict[str, int]:
        axis_indices: dict[str, int] = {}
        for axis_name, axis_index in value.items():
            if not isinstance(axis_name, str) or not axis_name.strip():
                raise ValueError(
                    f"Viewer payload control field {field.value!r} keys must be non-empty strings."
                )
            if isinstance(axis_index, bool) or not isinstance(axis_index, int):
                raise TypeError(
                    f"Viewer payload control field {field.value!r} value for {axis_name!r} must be an integer."
                )
            if axis_index < 0:
                raise ValueError(
                    f"Viewer payload control field {field.value!r} value for {axis_name!r} must be nonnegative."
                )
            axis_indices[axis_name.strip()] = axis_index
        return axis_indices

    @classmethod
    def _optional_slice_tuple(
        cls,
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerPayloadControlField,
    ) -> tuple[tuple[int, int], ...] | None:
        if field.value not in payload:
            return None
        value = payload[field.value]
        if value is None:
            return None
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a sequence."
            )
        return cls._optional_slice_tuple_value(
            tuple(cls._slice_pair_from_wire(item, field) for item in value),
            field,
        )

    @classmethod
    def _slice_pair_from_wire(
        cls,
        value: ViewerControlWireValue,
        field: ViewerPayloadControlField,
    ) -> tuple[int, int]:
        if not isinstance(value, (list, tuple)):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must contain sequences."
            )
        return cls._slice_pair_from_value(tuple(value), field)

    @classmethod
    def _optional_slice_tuple_value(
        cls,
        value: tuple[tuple[int, int], ...] | None,
        field: ViewerPayloadControlField,
    ) -> tuple[tuple[int, int], ...] | None:
        if value is None:
            return None
        if not isinstance(value, tuple):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a tuple."
            )
        return tuple(cls._slice_pair_from_value(item, field) for item in value)

    @staticmethod
    def _slice_pair_from_value(
        value: tuple[int, ...],
        field: ViewerPayloadControlField,
    ) -> tuple[int, int]:
        if len(value) != 2:
            raise ValueError(
                f"Viewer payload control field {field.value!r} slice entries must have start and stop."
            )
        start, stop = value
        if (
            isinstance(start, bool)
            or isinstance(stop, bool)
            or not isinstance(start, int)
            or not isinstance(stop, int)
        ):
            raise TypeError(
                f"Viewer payload control field {field.value!r} slice entries must contain integers."
            )
        if start < 0 or stop < 0:
            raise ValueError(
                f"Viewer payload control field {field.value!r} slice entries must be nonnegative."
            )
        if stop < start:
            raise ValueError(
                f"Viewer payload control field {field.value!r} slice stop must be greater than or equal to start."
            )
        return (start, stop)

    @staticmethod
    def _resolved_bool_override(
        field: ViewerPayloadControlField,
        value: bool | None,
        fallback: bool,
    ) -> bool:
        if value is None:
            return fallback
        if not isinstance(value, bool):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be a bool."
            )
        return value

    @staticmethod
    def _optional_nonnegative_int(
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerPayloadControlField,
        fallback: int,
    ) -> int:
        if field.value not in payload:
            return fallback
        value = payload[field.value]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be an integer."
            )
        if value < 0:
            raise ValueError(
                f"Viewer payload control field {field.value!r} must be nonnegative."
            )
        return value

    @staticmethod
    def _resolved_nonnegative_int_override(
        field: ViewerPayloadControlField,
        value: int | None,
        fallback: int,
    ) -> int:
        if value is None:
            return fallback
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Viewer payload control field {field.value!r} must be an integer."
            )
        if value < 0:
            raise ValueError(
                f"Viewer payload control field {field.value!r} must be nonnegative."
            )
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerStateControlOptions:
    """Formal state-inspection controls shared by agent and viewer runtimes."""

    route_key: str | None = None
    include_component_values: bool = VIEWER_STATE_INCLUDE_COMPONENT_VALUES_DEFAULT
    max_component_values_per_layer: int | None = None
    include_payload_summaries: bool = VIEWER_STATE_INCLUDE_PAYLOAD_SUMMARIES_DEFAULT
    max_payload_summaries_per_layer: int | None = None

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
            route_key=cls._optional_str_value(
                route_key,
                ViewerStateControlField.ROUTE_KEY,
            ),
            include_component_values=cls._resolved_bool_override(
                ViewerStateControlField.INCLUDE_COMPONENT_VALUES,
                include_component_values,
                defaults.include_component_values,
            ),
            max_component_values_per_layer=cls._optional_nonnegative_int_value(
                max_component_values_per_layer,
                ViewerStateControlField.MAX_COMPONENT_VALUES_PER_LAYER,
            ),
            include_payload_summaries=cls._resolved_bool_override(
                ViewerStateControlField.INCLUDE_PAYLOAD_SUMMARIES,
                include_payload_summaries,
                defaults.include_payload_summaries,
            ),
            max_payload_summaries_per_layer=cls._optional_nonnegative_int_value(
                max_payload_summaries_per_layer,
                ViewerStateControlField.MAX_PAYLOAD_SUMMARIES_PER_LAYER,
            ),
        )

    @classmethod
    def from_wire_payload(
        cls,
        payload: Mapping[str, ViewerControlWireValue],
    ) -> Self:
        return cls(
            route_key=cls._optional_str(
                payload,
                ViewerStateControlField.ROUTE_KEY,
            ),
            include_component_values=cls._optional_bool(
                payload,
                ViewerStateControlField.INCLUDE_COMPONENT_VALUES,
                VIEWER_STATE_INCLUDE_COMPONENT_VALUES_DEFAULT,
            ),
            max_component_values_per_layer=cls._optional_nonnegative_int(
                payload,
                ViewerStateControlField.MAX_COMPONENT_VALUES_PER_LAYER,
            ),
            include_payload_summaries=cls._optional_bool(
                payload,
                ViewerStateControlField.INCLUDE_PAYLOAD_SUMMARIES,
                VIEWER_STATE_INCLUDE_PAYLOAD_SUMMARIES_DEFAULT,
            ),
            max_payload_summaries_per_layer=cls._optional_nonnegative_int(
                payload,
                ViewerStateControlField.MAX_PAYLOAD_SUMMARIES_PER_LAYER,
            ),
        )

    def to_wire_payload(self) -> dict[str, ViewerControlWireValue]:
        payload: dict[str, ViewerControlWireValue] = {
            ViewerStateControlField.INCLUDE_COMPONENT_VALUES.value: (
                self.include_component_values
            ),
            ViewerStateControlField.INCLUDE_PAYLOAD_SUMMARIES.value: (
                self.include_payload_summaries
            ),
        }
        if self.route_key is not None:
            payload[ViewerStateControlField.ROUTE_KEY.value] = self.route_key
        if self.max_component_values_per_layer is not None:
            payload[ViewerStateControlField.MAX_COMPONENT_VALUES_PER_LAYER.value] = (
                self.max_component_values_per_layer
            )
        if self.max_payload_summaries_per_layer is not None:
            payload[ViewerStateControlField.MAX_PAYLOAD_SUMMARIES_PER_LAYER.value] = (
                self.max_payload_summaries_per_layer
            )
        return payload

    @staticmethod
    def _optional_str(
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerStateControlField,
    ) -> str | None:
        if field.value not in payload:
            return None
        value = payload[field.value]
        return ViewerStateControlOptions._optional_str_value(value, field)

    @staticmethod
    def _optional_str_value(
        value: object,
        field: ViewerStateControlField,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Viewer state control field {field.value!r} must be a non-empty string."
            )
        return value

    @staticmethod
    def _optional_bool(
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerStateControlField,
        fallback: bool,
    ) -> bool:
        if field.value not in payload:
            return fallback
        value = payload[field.value]
        if not isinstance(value, bool):
            raise TypeError(
                f"Viewer state control field {field.value!r} must be a bool."
            )
        return value

    @staticmethod
    def _resolved_bool_override(
        field: ViewerStateControlField,
        value: bool | None,
        fallback: bool,
    ) -> bool:
        if value is None:
            return fallback
        if not isinstance(value, bool):
            raise TypeError(
                f"Viewer state control field {field.value!r} must be a bool."
            )
        return value

    @staticmethod
    def _optional_nonnegative_int(
        payload: Mapping[str, ViewerControlWireValue],
        field: ViewerStateControlField,
    ) -> int | None:
        if field.value not in payload:
            return None
        return ViewerStateControlOptions._optional_nonnegative_int_value(
            payload[field.value],
            field,
        )

    @staticmethod
    def _optional_nonnegative_int_value(
        value: object,
        field: ViewerStateControlField,
    ) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Viewer state control field {field.value!r} must be an integer."
            )
        if value < 0:
            raise ValueError(
                f"Viewer state control field {field.value!r} must be nonnegative."
            )
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerNavigationControlOptions:
    """Formal viewer navigation controls shared by agent and viewer runtimes."""

    route_key: str
    axis_indices: Mapping[str, int] = field(default_factory=dict)
    visible: bool | None = None
    selected: bool | None = None

    @classmethod
    def from_overrides(
        cls,
        *,
        route_key: str,
        axis_indices: Mapping[str, int] | None = None,
        visible: bool | None = None,
        selected: bool | None = None,
    ) -> Self:
        return cls(
            route_key=cls._required_str(
                route_key,
                ViewerNavigationControlField.ROUTE_KEY,
            ),
            axis_indices=cls._axis_indices(axis_indices or {}),
            visible=cls._optional_navigation_bool(
                visible,
                ViewerNavigationControlField.VISIBLE,
            ),
            selected=cls._optional_navigation_bool(
                selected,
                ViewerNavigationControlField.SELECTED,
            ),
        )

    @classmethod
    def from_wire_payload(
        cls,
        payload: Mapping[str, ViewerControlWireValue],
    ) -> Self:
        route_key = payload.get(ViewerNavigationControlField.ROUTE_KEY.value)
        if not isinstance(route_key, str) or not route_key:
            raise ValueError("Viewer navigation control requires a non-empty route_key.")
        return cls(
            route_key=route_key,
            axis_indices=cls._axis_indices(
                payload.get(ViewerNavigationControlField.AXIS_INDICES.value) or {}
            ),
            visible=cls._optional_navigation_bool(
                payload.get(ViewerNavigationControlField.VISIBLE.value),
                ViewerNavigationControlField.VISIBLE,
            ),
            selected=cls._optional_navigation_bool(
                payload.get(ViewerNavigationControlField.SELECTED.value),
                ViewerNavigationControlField.SELECTED,
            ),
        )

    def to_wire_payload(self) -> dict[str, ViewerControlWireValue]:
        payload: dict[str, ViewerControlWireValue] = {
            ViewerNavigationControlField.ROUTE_KEY.value: self.route_key,
            ViewerNavigationControlField.AXIS_INDICES.value: dict(self.axis_indices),
        }
        if self.visible is not None:
            payload[ViewerNavigationControlField.VISIBLE.value] = self.visible
        if self.selected is not None:
            payload[ViewerNavigationControlField.SELECTED.value] = self.selected
        return payload

    @staticmethod
    def _required_str(
        value: str,
        field: ViewerNavigationControlField,
    ) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Viewer navigation control field {field.value!r} must be a non-empty string."
            )
        return value

    @classmethod
    def _axis_indices(
        cls,
        value: Mapping[str, int] | ViewerControlWireValue,
    ) -> dict[str, int]:
        if not isinstance(value, Mapping):
            raise TypeError("Viewer navigation axis_indices must be a mapping.")
        return {
            cls._axis_name(axis_name): cls._axis_index(axis_index, str(axis_name))
            for axis_name, axis_index in value.items()
        }

    @staticmethod
    def _axis_name(value: object) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError("Viewer navigation axis_indices keys must be non-empty strings.")
        return value

    @staticmethod
    def _axis_index(value: object, axis_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"Viewer navigation index for axis {axis_name!r} must be an integer."
            )
        if value < 0:
            raise ValueError(
                f"Viewer navigation index for axis {axis_name!r} must be nonnegative."
            )
        return value

    @staticmethod
    def _optional_navigation_bool(
        value: object,
        field: ViewerNavigationControlField,
    ) -> bool | None:
        if value is None:
            return None
        if not isinstance(value, bool):
            raise TypeError(
                f"Viewer navigation control field {field.value!r} must be a boolean."
            )
        return value
