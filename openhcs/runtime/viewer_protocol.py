"""Shared nominal viewer protocol values for streaming visualizers."""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import ClassVar, Self, TypeAlias, cast

from openhcs.core.config import TransportMode as ViewerTransportMode
from openhcs.core.streaming_config_factory import (
    StreamingViewerRuntimeConfig,
)
from metaclass_registry import AutoRegisterMeta
from polystore.streaming_constants import StreamingDataType
from zmqruntime.config import TransportMode as ZMQTransportMode, ZMQConfig
from zmqruntime.streaming import VisualizerProcessManager
from zmqruntime.viewer_protocol import (
    ViewerBatchMessageType,
    ViewerBatchContextWireField,
    ViewerBatchWireField,
    ViewerControlResponseField,
    ViewerControlReplyHeader,
    ViewerControlReplyPayload,
    ViewerProtocolStatus,
    ViewerTransportEndpoint,
)


ViewerScalar: TypeAlias = str | int | float | bool | None
ViewerComponentValue: TypeAlias = ViewerScalar | tuple[ViewerScalar, ...]
NaturalTokenKey: TypeAlias = tuple[int, int | str]
NaturalTextKey: TypeAlias = tuple[NaturalTokenKey, ...]
ComponentValueSortKey: TypeAlias = tuple[int, int | float | NaturalTextKey, str, str]
ComponentTupleSortKey: TypeAlias = tuple[ComponentValueSortKey, ...]
ViewerHeartbeatValue: TypeAlias = str | bool | int | float | None
ViewerProcess: TypeAlias = BaseProcess | subprocess.Popen[bytes]
ViewerLaunchLiteral: TypeAlias = str | int | float | bool | None
ViewerControlWireValue: TypeAlias = (
    ViewerScalar
    | tuple["ViewerControlWireValue", ...]
    | list["ViewerControlWireValue"]
    | dict[str, "ViewerControlWireValue"]
)
ViewerPayloadAxisIndices: TypeAlias = tuple[int, ...] | dict[str, int]


class ViewerType(Enum):
    """Supported OpenHCS streaming viewer identities."""

    FIJI = "fiji"
    NAPARI = "napari"


class ViewerControlMessageType(Enum):
    """Shared control-message names consumed by viewer servers."""

    SCREENSHOT = "screenshot"
    SETTLE = "settle"
    STATE = "state"
    PAYLOADS = "payloads"
    NAVIGATE = "navigate"


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


@dataclass(frozen=True, slots=True)
class ViewerTypeIdentity:
    """Inherited viewer identity for runtime protocol records."""

    viewer_type: ViewerType


class ViewerPersistenceMode(Enum):
    """Viewer lifecycle ownership mode derived from streaming persistence."""

    PERSISTENT = "persistent"
    NON_PERSISTENT = "non-persistent"

    @classmethod
    def from_flag(cls, persistent: bool) -> "ViewerPersistenceMode":
        if persistent:
            return cls.PERSISTENT
        return cls.NON_PERSISTENT


@dataclass(frozen=True, slots=True)
class ViewerControlResponse:
    """Typed view of a viewer control-message response."""

    payload: Mapping[str, ViewerControlWireValue]

    @property
    def status(self) -> ViewerProtocolStatus:
        status_value = self.payload.get(ViewerControlResponseField.STATUS.value)
        if status_value is None:
            raise ValueError("Viewer control response is missing a status field.")
        return ViewerProtocolStatus(str(status_value))

    def succeeded(self) -> bool:
        return self.status is ViewerProtocolStatus.SUCCESS


class ViewerComponentValueOrdering:
    """Canonical ordering for viewer component values and stack coordinates."""

    NATURAL_TOKEN_PATTERN = re.compile(r"(\d+)")
    INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
    FLOAT_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")

    @classmethod
    def key(cls, value: ViewerComponentValue) -> ComponentValueSortKey:
        numeric_value = cls.numeric_value(value)
        if numeric_value is not None:
            return (0, numeric_value, type(value).__name__, str(value))

        text = str(value)
        return (1, cls.natural_text_key(text), type(value).__name__, text)

    @classmethod
    def tuple_key(
        cls, values: tuple[ViewerComponentValue, ...]
    ) -> ComponentTupleSortKey:
        return tuple(cls.key(value) for value in values)

    @classmethod
    def numeric_value(cls, value: ViewerComponentValue) -> int | float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return value
        if not isinstance(value, str):
            return None

        text = value.strip()
        if not text:
            return None
        if cls.INTEGER_PATTERN.fullmatch(text):
            return int(text)
        if cls.FLOAT_PATTERN.fullmatch(text):
            return float(text)
        return None

    @classmethod
    def natural_text_key(cls, text: str) -> NaturalTextKey:
        return tuple(
            (0, int(token)) if token.isdecimal() else (1, token.casefold())
            for token in cls.NATURAL_TOKEN_PATTERN.split(text)
            if token
        )


class QtPlatformName(Enum):
    """Qt platform plugin names used by detached viewer processes."""

    COCOA = "cocoa"
    XCB = "xcb"


class ViewerProcessPlatform(Enum):
    """Host platform family for detached viewer launch behavior."""

    WINDOWS = ("win32", None, None, {})
    DARWIN = ("Darwin", "Darwin", QtPlatformName.COCOA, {})
    LINUX = (
        "Linux",
        "Linux",
        QtPlatformName.XCB,
        {"QT_X11_NO_MITSHM": "1"},
    )
    OTHER = ("other", None, None, {})

    def __new__(
        cls,
        value: str,
        system_name: str | None,
        qpa_platform: QtPlatformName | None,
        always_set: Mapping[str, str],
    ) -> "ViewerProcessPlatform":
        member = object.__new__(cls)
        member._value_ = value
        member.system_name = system_name
        member.qpa_platform = qpa_platform
        member.always_set = dict(always_set)
        return member

    @classmethod
    def current(cls) -> "ViewerProcessPlatform":
        if sys.platform == cls.WINDOWS.value:
            return cls.WINDOWS
        system_name = platform.system()
        for platform_family in cls:
            if platform_family.system_name == system_name:
                return platform_family
        return cls.OTHER

    def qt_environment_policy(self) -> "ViewerQtPlatformEnvironmentPolicy":
        return ViewerQtPlatformEnvironmentPolicy(
            qpa_platform=self.qpa_platform,
            always_set=self.always_set,
        )


class NapariLayerKind(Enum):
    """Napari layer creation families used by streaming display."""

    IMAGE = "image"
    SHAPES = "shapes"
    POINTS = "points"
    LABELS = "labels"


class FijiPayloadKind(Enum):
    """Payload strings sent to the Fiji viewer process."""

    IMAGE = ("image", StreamingDataType.IMAGE, True)
    ROIS = ("rois", StreamingDataType.ROIS, False)

    def __init__(
        self,
        wire_value: str,
        streaming_data_type: StreamingDataType,
        uses_shared_memory: bool,
    ) -> None:
        self.wire_value = wire_value
        self.streaming_data_type = streaming_data_type
        self.uses_shared_memory = uses_shared_memory

    @classmethod
    def from_payload(cls, payload: str | None) -> "FijiPayloadKind | None":
        if payload is None:
            return None
        wire_value = str(payload)
        if wire_value in FIJI_PAYLOAD_KIND_BY_WIRE_VALUE:
            return FIJI_PAYLOAD_KIND_BY_WIRE_VALUE[wire_value]
        return None


FIJI_PAYLOAD_KIND_BY_WIRE_VALUE: Mapping[str, FijiPayloadKind] = {
    kind.wire_value: kind for kind in FijiPayloadKind
}


class ViewerHeartbeatField(Enum):
    """Fields owned by OpenHCS viewer heartbeat payloads."""

    VIEWER = "viewer"
    OPENHCS = "openhcs"
    SERVER = "server"
    MEMORY_MB = "memory_mb"
    CPU_PERCENT = "cpu_percent"


@dataclass(slots=True)
class ViewerHeartbeatPayload:
    """Nominal heartbeat payload builder around the ZMQ server pong mapping."""

    values: dict[str, ViewerHeartbeatValue] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        response: Mapping[str, ViewerHeartbeatValue],
    ) -> "ViewerHeartbeatPayload":
        return cls(dict(response))

    def set_field(
        self,
        field_name: ViewerHeartbeatField,
        value: ViewerHeartbeatValue,
    ) -> None:
        self.values[field_name.value] = value

    def mark_viewer(self, viewer_type: ViewerType, server_name: str) -> None:
        self.set_field(ViewerHeartbeatField.VIEWER, viewer_type.value)
        self.set_field(ViewerHeartbeatField.OPENHCS, True)
        self.set_field(ViewerHeartbeatField.SERVER, server_name)

    def add_process_metrics(self) -> None:
        import psutil

        process = psutil.Process(os.getpid())
        self.set_field(
            ViewerHeartbeatField.MEMORY_MB,
            process.memory_info().rss / 1024 / 1024,
        )
        self.set_field(
            ViewerHeartbeatField.CPU_PERCENT,
            process.cpu_percent(interval=0),
        )

    def to_dict(self) -> dict[str, ViewerHeartbeatValue]:
        return dict(self.values)


@dataclass(frozen=True, slots=True)
class ViewerHeartbeatDescriptor(ViewerTypeIdentity):
    """Viewer-specific fields added to a streaming server pong response."""

    server_name: str

    def apply_to(
        self,
        response: Mapping[str, ViewerHeartbeatValue],
    ) -> dict[str, ViewerHeartbeatValue]:
        heartbeat = ViewerHeartbeatPayload.from_mapping(response)
        heartbeat.mark_viewer(self.viewer_type, self.server_name)
        try:
            heartbeat.add_process_metrics()
        except Exception:
            pass
        return heartbeat.to_dict()


NAPARI_HEARTBEAT = ViewerHeartbeatDescriptor(ViewerType.NAPARI, "NapariViewer")
FIJI_HEARTBEAT = ViewerHeartbeatDescriptor(ViewerType.FIJI, "FijiViewerServer")


def viewer_lifecycle_registry_key(
    name: str,
    cls: type,
) -> str:
    """Derive the lifecycle registry key from the declared detached entrypoint."""
    del name
    if "detached_server_entrypoint" not in cls.__dict__:
        raise TypeError(
            f"{cls.__name__} must declare detached_server_entrypoint to register "
            "as a managed viewer lifecycle."
        )
    entrypoint = cls.__dict__["detached_server_entrypoint"]
    return entrypoint.viewer_type.value


@dataclass(frozen=True, slots=True, kw_only=True)
class ViewerServerLaunchRequest:
    """Shared launch request fields for OpenHCS viewer server processes."""

    port: int
    log_file_path: str | None = None
    transport_mode: ViewerTransportMode = ViewerTransportMode.IPC


@dataclass(frozen=True, slots=True)
class NapariViewerServerRequest(ViewerServerLaunchRequest):
    """Nominal launch request consumed by the Napari viewer server."""

    viewer_title: str
    replace_layers: bool = False


@dataclass(frozen=True, slots=True)
class ViewerRuntimeEndpoint:
    """OpenHCS viewer endpoint projected onto zmqruntime primitives."""

    transport: ViewerTransportEndpoint
    config: ZMQConfig

    @property
    def port(self) -> int:
        return self.transport.port

    @property
    def host(self) -> str:
        return self.transport.host

    @property
    def mode(self) -> ViewerTransportMode:
        return self.transport.transport_mode

    @property
    def zmq_transport_mode(self) -> ZMQTransportMode:
        from zmqruntime.transport import coerce_transport_mode

        zmq_mode = coerce_transport_mode(self.mode)
        if zmq_mode is None:
            raise ValueError(f"Unsupported viewer transport mode: {self.mode!r}")
        return zmq_mode

    @property
    def control_port(self) -> int:
        from zmqruntime.transport import get_control_port

        return get_control_port(self.port, self.config)

    def data_url(self) -> str:
        from zmqruntime.transport import get_zmq_transport_url

        return get_zmq_transport_url(
            self.port,
            host=self.host,
            mode=self.zmq_transport_mode,
            config=self.config,
        )

    def control_url(self) -> str:
        from zmqruntime.transport import get_control_url

        return get_control_url(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
        )

    def in_use(self) -> bool:
        from zmqruntime.transport import is_port_in_use

        return is_port_in_use(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
        )

    def wait_until_released(
        self,
        *,
        timeout: float,
        poll_interval: float = 0.1,
    ) -> bool:
        deadline = time.monotonic() + timeout
        while self.in_use() and time.monotonic() < deadline:
            time.sleep(poll_interval)
        return not self.in_use()

    def ping(
        self,
        *,
        timeout_ms: int,
        require_ready: bool,
    ) -> bool:
        from zmqruntime.transport import ping_control_port

        return ping_control_port(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
            timeout_ms=timeout_ms,
            require_ready=require_ready,
        )

    def wait_ready(self, *, timeout: float, require_ready: bool = True) -> bool:
        from zmqruntime.transport import wait_for_server_ready

        return wait_for_server_ready(
            self.port,
            self.zmq_transport_mode,
            host=self.host,
            config=self.config,
            timeout=timeout,
            require_ready=require_ready,
        )

    def release_bound_ports(self) -> None:
        if self.zmq_transport_mode is ZMQTransportMode.IPC:
            from zmqruntime.transport import remove_ipc_socket

            remove_ipc_socket(self.port, self.config)
            remove_ipc_socket(self.control_port, self.config)
            return

        from zmqruntime.server import ZMQServer

        ZMQServer.kill_processes_on_port(self.port)
        ZMQServer.kill_processes_on_port(self.control_port)


@dataclass(frozen=True, slots=True)
class DetachedViewerPythonExpression:
    """One expression allowed in generated detached-viewer Python."""

    source: str

    @classmethod
    def literal(cls, value: ViewerLaunchLiteral) -> "DetachedViewerPythonExpression":
        return cls(repr(value))

    @classmethod
    def symbol(cls, name: str) -> "DetachedViewerPythonExpression":
        if not name.isidentifier():
            raise ValueError(
                f"Detached viewer symbol is not a valid identifier: {name!r}"
            )
        return cls(name)


@dataclass(frozen=True, slots=True)
class DetachedViewerPythonArguments:
    """Nominal argument list for detached-viewer entrypoint code generation."""

    expressions: tuple[DetachedViewerPythonExpression, ...] = ()

    @classmethod
    def from_literals(
        cls,
        *values: ViewerLaunchLiteral,
    ) -> "DetachedViewerPythonArguments":
        return cls(
            tuple(DetachedViewerPythonExpression.literal(value) for value in values)
        )

    def append(
        self,
        *expressions: DetachedViewerPythonExpression,
    ) -> "DetachedViewerPythonArguments":
        return type(self)((*self.expressions, *expressions))

    def render(self) -> str:
        return ",\n".join(expression.source for expression in self.expressions)


@dataclass(frozen=True, slots=True)
class DetachedViewerLaunchRequest(ViewerTypeIdentity):
    """Authoritative detached launch request for a viewer process."""

    port: int
    python_code: str
    log_file: Path
    cwd: Path = field(default_factory=Path.cwd)
    env: Mapping[str, str] | None = None
    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    @classmethod
    def log_file_for(
        cls,
        *,
        viewer_type: ViewerType,
        port: int,
        log_dir: Path | None = None,
    ) -> Path:
        launch_log_dir = (
            Path.home() / ".local" / "share" / "openhcs" / "logs"
            if log_dir is None
            else log_dir
        )
        return launch_log_dir / f"{viewer_type.value}_detached_port_{port}.log"

    def command(self) -> list[str]:
        return [sys.executable, "-c", self.python_code]

    def launch(self) -> subprocess.Popen[bytes]:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        launch_env = dict(os.environ if self.env is None else self.env)
        ViewerQtEnvironmentPolicy(self.platform).apply_to(launch_env)
        log_handle = self.log_file.open("w")
        if self.platform is ViewerProcessPlatform.WINDOWS:
            return subprocess.Popen(
                self.command(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS,
                env=launch_env,
                cwd=str(self.cwd),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        return subprocess.Popen(
            self.command(),
            env=launch_env,
            cwd=str(self.cwd),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


@dataclass(frozen=True, slots=True)
class DetachedViewerServerEntrypointSpec(ViewerTypeIdentity):
    """Declared server function used to launch one detached viewer family."""

    module_name: str
    function_name: str
    extra_imports: tuple[str, ...] = ()

    def log_file_for(self, port: int) -> Path:
        return DetachedViewerLaunchRequest.log_file_for(
            viewer_type=self.viewer_type,
            port=port,
        )

    def python_code(
        self,
        python_path_root: Path,
        *,
        transport_mode: ViewerTransportMode,
        arguments: DetachedViewerPythonArguments,
    ) -> str:
        transport_name = transport_mode.name
        rendered_arguments = arguments.render()
        call_arguments = "\n".join(
            f"    {line}" for line in rendered_arguments.splitlines()
        )
        lines = [
            "import os",
            "import sys",
            "",
            'if os.name == "posix":',
            "    try:",
            "        os.setsid()",
            "    except OSError:",
            "        pass",
            "",
            f"sys.path.insert(0, {str(python_path_root)!r})",
            "",
            "try:",
            f"    from {self.module_name} import {self.function_name}",
            "    from openhcs.core.config import TransportMode",
        ]
        lines.extend(f"    {extra_import}" for extra_import in self.extra_imports)
        lines.extend(
            [
                "",
                f"    transport_mode = TransportMode.{transport_name}",
                f"    {self.function_name}(",
                call_arguments,
                "    )",
                "except Exception as error:",
                "    import logging",
                "    import traceback",
                "",
                '    logger = logging.getLogger("openhcs.runtime.detached_viewer")',
                '    logger.error("Detached viewer error: %s", error)',
                "    logger.error(traceback.format_exc())",
                "    sys.exit(1)",
            ]
        )
        return "\n".join(lines)

    def launch_request(
        self,
        *,
        port: int,
        transport_mode: ViewerTransportMode,
        arguments: DetachedViewerPythonArguments,
        log_file: Path,
        cwd: Path | None = None,
    ) -> DetachedViewerLaunchRequest:
        if cwd is None:
            cwd = Path.cwd()
        return DetachedViewerLaunchRequest(
            viewer_type=self.viewer_type,
            port=port,
            python_code=self.python_code(
                cwd,
                transport_mode=transport_mode,
                arguments=arguments,
            ),
            log_file=log_file,
            cwd=cwd,
        )


@dataclass(frozen=True, slots=True)
class ViewerQtPlatformEnvironmentPolicy:
    """Environment mutations for one viewer platform."""

    qpa_platform: QtPlatformName | None = None
    always_set: Mapping[str, str] = field(default_factory=dict)

    def apply_to(self, env: dict[str, str]) -> dict[str, str]:
        if self.qpa_platform is not None and "QT_QPA_PLATFORM" not in env:
            env["QT_QPA_PLATFORM"] = self.qpa_platform.value
        env.update(self.always_set)
        return env


@dataclass(frozen=True, slots=True)
class ViewerQtEnvironmentPolicy:
    """Apply viewer-safe Qt environment defaults for the current platform."""

    platform: ViewerProcessPlatform = field(
        default_factory=ViewerProcessPlatform.current
    )

    def apply_to(self, env: dict[str, str]) -> dict[str, str]:
        return self.platform.qt_environment_policy().apply_to(env)


@dataclass(frozen=True, slots=True)
class ViewerProcessHandle:
    """Nominal adapter over multiprocessing and subprocess viewer handles."""

    process: ViewerProcess

    @classmethod
    def from_process(cls, process: ViewerProcess) -> "ViewerProcessHandle":
        if isinstance(process, (BaseProcess, subprocess.Popen)):
            return cls(process)
        raise TypeError(f"Unsupported viewer process handle: {type(process)!r}")

    @property
    def pid(self) -> int | None:
        return self.process.pid

    @property
    def pid_label(self) -> str:
        if self.pid is None:
            return "unknown"
        return str(self.pid)

    def is_alive(self) -> bool:
        if isinstance(self.process, BaseProcess):
            return self.process.is_alive()
        return self.process.poll() is None

    def terminate(self, *, timeout: float = 5.0, kill_timeout: float = 2.0) -> bool:
        if not self.is_alive():
            return False
        self.process.terminate()
        if isinstance(self.process, BaseProcess):
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=kill_timeout)
                return True
            return False
        try:
            self.process.wait(timeout=timeout)
            return False
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=kill_timeout)
            return True


class ViewerControlPingMode(Enum):
    """Viewer control-port ping policy modes."""

    QUICK = ("quick", 200, False)
    EXISTING_VIEWER = ("existing_viewer", 500, True)

    def __new__(
        cls,
        value: str,
        timeout_ms: int,
        require_ready: bool,
    ) -> "ViewerControlPingMode":
        member = object.__new__(cls)
        member._value_ = value
        member.timeout_ms = timeout_ms
        member.require_ready = require_ready
        return member

    def policy(self) -> "ViewerControlPingPolicy":
        return ViewerControlPingPolicy(
            timeout_ms=self.timeout_ms,
            require_ready=self.require_ready,
        )


class ViewerLifecycleMode(Enum):
    """Runtime ownership state for a managed viewer."""

    STOPPED = "stopped"
    CONNECTED_EXTERNAL = "connected_external"
    OWNED_PROCESS = "owned_process"


@dataclass(slots=True)
class ViewerLifecycleState:
    """Nominal lifecycle state for viewer process managers."""

    mode: ViewerLifecycleMode = ViewerLifecycleMode.STOPPED

    @classmethod
    def stopped(cls) -> "ViewerLifecycleState":
        return cls()

    @property
    def is_active(self) -> bool:
        return self.mode is not ViewerLifecycleMode.STOPPED

    @property
    def is_connected_external(self) -> bool:
        return self.mode is ViewerLifecycleMode.CONNECTED_EXTERNAL

    def mark_connected_external(self) -> None:
        self.mode = ViewerLifecycleMode.CONNECTED_EXTERNAL

    def mark_owned_process(self) -> None:
        self.mode = ViewerLifecycleMode.OWNED_PROCESS

    def mark_stopped(self) -> None:
        self.mode = ViewerLifecycleMode.STOPPED


@dataclass(frozen=True, slots=True)
class ViewerControlPingPolicy:
    """Timeout/readiness coordinates for one control ping mode."""

    timeout_ms: int
    require_ready: bool


@dataclass(frozen=True, slots=True)
class ViewerControlPingRequest:
    """Typed control-port ping request for viewer readiness checks."""

    endpoint: ViewerRuntimeEndpoint
    timeout_ms: int = 500
    require_ready: bool = True

    @classmethod
    def from_mode(
        cls,
        *,
        mode: ViewerControlPingMode,
        endpoint: ViewerRuntimeEndpoint,
    ) -> "ViewerControlPingRequest":
        policy = mode.policy()
        return cls(
            endpoint=endpoint,
            timeout_ms=policy.timeout_ms,
            require_ready=policy.require_ready,
        )


@dataclass(frozen=True, slots=True)
class ViewerControlMessageRequest:
    """Typed REQ/REP control-message request shared by viewer visualizers."""

    endpoint: ViewerRuntimeEndpoint
    message_type: str
    timeout: float = 2.0

    def send(self) -> ViewerControlResponse:
        import pickle

        import zmq

        context = None
        socket = None
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.setsockopt(zmq.RCVTIMEO, int(self.timeout * 1000))
            socket.connect(self.endpoint.control_url())
            socket.send(
                pickle.dumps({ViewerControlResponseField.TYPE.value: self.message_type})
            )
            payload = pickle.loads(socket.recv())
            if not isinstance(payload, Mapping):
                raise TypeError(
                    "Viewer control response must be a mapping, "
                    f"got {type(payload).__name__}."
                )
            return ViewerControlResponse(
                cast(Mapping[str, ViewerControlWireValue], payload)
            )
        finally:
            if socket is not None:
                socket.close()
            if context is not None:
                context.term()


class ManagedViewerLifecycleMixin(
    VisualizerProcessManager,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Shared liveness property for viewer process managers."""

    __registry_key__ = "viewer_type"
    __key_extractor__ = staticmethod(viewer_lifecycle_registry_key)
    __skip_if_no_key__ = True

    __registry__: ClassVar[dict[str, type["ManagedViewerLifecycleMixin"]]]
    viewer_type: ClassVar[str | None] = None
    viewer_process_label: ClassVar[str] = "viewer"
    detached_server_entrypoint: ClassVar[DetachedViewerServerEntrypointSpec]

    def __init__(
        self,
        *,
        runtime_config: StreamingViewerRuntimeConfig,
        transport_config: ZMQConfig,
    ) -> None:
        super().__init__(port=runtime_config.transport_endpoint.port)
        self.persistent: bool = runtime_config.persistent
        self.lifecycle_presentation = runtime_config.presentation
        self.runtime_endpoint = ViewerRuntimeEndpoint(
            transport=runtime_config.transport_endpoint,
            config=transport_config,
        )
        self.lifecycle_state: ViewerLifecycleState = ViewerLifecycleState.stopped()

    @property
    def required_port(self) -> int:
        port = self.port
        if port is None:
            raise RuntimeError("OpenHCS streaming viewers require a configured port.")
        return port

    @property
    def persistence_mode(self) -> ViewerPersistenceMode:
        return ViewerPersistenceMode.from_flag(self.persistent)

    @property
    def persistence_label(self) -> str:
        return self.persistence_mode.value

    @property
    def viewer_title(self) -> str:
        return self.lifecycle_presentation.title

    @abstractmethod
    def start_viewer(self, async_mode: bool = False) -> None:
        """Start the concrete viewer server process."""

    @abstractmethod
    def detached_server_arguments(
        self,
        *,
        log_file: Path,
    ) -> DetachedViewerPythonArguments:
        """Return entrypoint arguments for this concrete viewer server."""

    def check_connected_viewer(self) -> bool:
        """Return whether an externally-owned viewer is still responsive."""
        request = ViewerControlPingRequest.from_mode(
            mode=ViewerControlPingMode.QUICK,
            endpoint=self.runtime_endpoint,
        )
        return request.endpoint.ping(
            timeout_ms=request.timeout_ms,
            require_ready=request.require_ready,
        )

    def request_bound_viewer_shutdown(self, timeout: float = 1.0) -> bool:
        """Ask the viewer currently bound to this endpoint to terminate."""
        response = ViewerControlMessageRequest(
            endpoint=self.runtime_endpoint,
            message_type="force_shutdown",
            timeout=timeout,
        ).send()
        return response.succeeded()

    def prepare_fresh_viewer_start(self) -> None:
        """Ensure this viewer endpoint is not backed by a previous run."""
        if not self.runtime_endpoint.in_use():
            return

        if self.check_connected_viewer():
            if not self.request_bound_viewer_shutdown():
                raise RuntimeError(
                    f"{self.viewer_process_label} viewer on port {self.required_port} "
                    "did not acknowledge shutdown before a fresh start."
                )
            if self.runtime_endpoint.wait_until_released(timeout=3.0):
                return

        self.runtime_endpoint.release_bound_ports()
        if not self.runtime_endpoint.wait_until_released(timeout=2.0):
            raise RuntimeError(
                f"{self.viewer_process_label} viewer on port {self.required_port} "
                "remained bound after forced endpoint release."
            )

    def existing_viewer_is_ready(self) -> bool:
        request = ViewerControlPingRequest.from_mode(
            mode=ViewerControlPingMode.EXISTING_VIEWER,
            endpoint=self.runtime_endpoint,
        )
        return request.endpoint.ping(
            timeout_ms=request.timeout_ms,
            require_ready=request.require_ready,
        )

    def wait_for_ready(self, timeout: float = 10.0) -> bool:
        """Satisfy zmqruntime's process-manager readiness contract."""
        return self.runtime_endpoint.wait_ready(
            timeout=timeout,
            require_ready=True,
        )

    def detached_launch_request(self) -> DetachedViewerLaunchRequest:
        port = self.required_port
        log_file = self.detached_server_entrypoint.log_file_for(port)
        return self.detached_server_entrypoint.launch_request(
            port=port,
            transport_mode=self.runtime_endpoint.mode,
            arguments=self.detached_server_arguments(log_file=log_file),
            log_file=log_file,
        )

    def launch_detached_viewer(self) -> subprocess.Popen[bytes]:
        launch_request = self.detached_launch_request()
        process = launch_request.launch()
        logging.getLogger(type(self).__module__).info(
            "%s detached process started (PID: %s), logging to %s",
            self.viewer_process_label,
            process.pid,
            launch_request.log_file,
        )
        return process

    def get_launch_command(self) -> list[str]:
        return self.detached_launch_request().command()

    def get_launch_env(self) -> dict[str, str]:
        return ViewerQtEnvironmentPolicy().apply_to(dict(os.environ))

    def cleanup_viewer_client(self) -> None:
        """Release client-side resources before forced viewer termination."""

    def force_stop(self, timeout: float = 5.0) -> None:
        """Terminate the viewer process regardless of persistence policy."""
        with self._lock:
            self.cleanup_viewer_client()
            if self.process is not None:
                killed = ViewerProcessHandle.from_process(self.process).terminate(
                    timeout=timeout,
                    kill_timeout=2.0,
                )
                if killed:
                    logging.getLogger(type(self).__module__).warning(
                        "%s viewer required force kill during shutdown",
                        self.viewer_process_label,
                    )
                self.process = None
            self.runtime_endpoint.release_bound_ports()
            self.lifecycle_state.mark_stopped()

    def start(self, detached: bool = True) -> subprocess.Popen[bytes]:
        self.start_viewer(async_mode=False)
        if self.process is None:
            raise RuntimeError(
                f"{self.viewer_process_label} viewer process failed to start."
            )
        return self.process

    @property
    def process_pid_label(self) -> str:
        process = self.process
        if process is None:
            return "unknown"
        return ViewerProcessHandle.from_process(process).pid_label

    def send_control_message(self, message_type: str, timeout: float = 2.0) -> bool:
        if not self.is_running:
            logging.getLogger(type(self).__module__).warning(
                "%s viewer cannot send %s - viewer not running",
                self.viewer_process_label,
                message_type,
            )
            return False

        try:
            response = ViewerControlMessageRequest(
                endpoint=self.runtime_endpoint,
                message_type=message_type,
                timeout=timeout,
            ).send()
            if response.succeeded():
                logging.getLogger(type(self).__module__).info(
                    "%s viewer acknowledged %s",
                    self.viewer_process_label,
                    message_type,
                )
                return True
            logging.getLogger(type(self).__module__).warning(
                "%s viewer failed %s: %s",
                self.viewer_process_label,
                message_type,
                response.payload,
            )
            return False
        except Exception as error:
            logging.getLogger(type(self).__module__).warning(
                "%s viewer failed to send %s: %s",
                self.viewer_process_label,
                message_type,
                error,
            )
            return False

    def clear_viewer_state(self) -> bool:
        """Clear accumulated viewer state for a new pipeline run."""

        return self.send_control_message("clear_state")

    def settle_viewer_state(self, timeout: float = 30.0) -> bool:
        """Wait for queued viewer layer updates before state/screenshot reads."""

        return self.send_control_message(
            ViewerControlMessageType.SETTLE.value,
            timeout=timeout,
        )

    @property
    def is_running(self) -> bool:
        lifecycle_state = self.lifecycle_state
        if not lifecycle_state.is_active:
            return False

        if lifecycle_state.is_connected_external:
            if not self.check_connected_viewer():
                logging.getLogger(self.__class__.__module__).debug(
                    "%s viewer on port %s is no longer responsive",
                    self.viewer_process_label,
                    self.port,
                )
                lifecycle_state.mark_stopped()
                return False
            return True

        if self.process is None:
            lifecycle_state.mark_stopped()
            return False

        try:
            alive = ViewerProcessHandle.from_process(self.process).is_alive()
            if not alive:
                logging.getLogger(self.__class__.__module__).debug(
                    "%s process on port %s is no longer alive",
                    self.viewer_process_label,
                    self.port,
                )
                lifecycle_state.mark_stopped()
            return alive
        except Exception as error:
            logging.getLogger(self.__class__.__module__).warning(
                "Error checking %s process status: %s",
                self.viewer_process_label,
                error,
            )
            lifecycle_state.mark_stopped()
            return False


@dataclass(frozen=True, slots=True)
class ChannelColormapPolicy:
    """Resolve channel-slice colors from component metadata."""

    colors_by_channel: Mapping[int, str] = field(
        default_factory=lambda: {1: "green", 2: "red"}
    )

    def colormap(self, channel_value: ViewerComponentValue) -> str | None:
        channel_number = self._channel_number(channel_value)
        if channel_number is None:
            return None
        return self.colors_by_channel.get(channel_number)

    @staticmethod
    def _channel_number(channel_value: ViewerComponentValue) -> int | None:
        if (
            channel_value is None
            or isinstance(channel_value, bool)
            or isinstance(channel_value, tuple)
        ):
            return None
        if isinstance(channel_value, int):
            return channel_value
        if isinstance(channel_value, float):
            if channel_value.is_integer():
                return int(channel_value)
            return None
        stripped = channel_value.strip()
        if stripped and stripped.lstrip("+-").isdigit():
            return int(stripped)
        return None
