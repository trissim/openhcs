"""
Fiji viewer server for OpenHCS.

ZMQ-based server that receives images from FijiStreamingBackend and displays them
via PyImageJ. Inherits from ZMQServer ABC for ping/pong handshake and dual-channel pattern.
"""

import logging
import time
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, TypeAlias

import numpy as np

from metaclass_registry import AutoRegisterMeta
from polystore.streaming_constants import StreamingDataType
from polystore.streaming.receivers.core import (
    DebouncedBatchEngine,
    GroupedWindowItems,
    WindowProjectionPayloadProvider,
)
from openhcs.core.config import (
    FijiDimensionMode,
    FijiDisplayConfig,
    FijiLUT,
    TransportMode as OpenHCSTransportMode,
)
from openhcs.runtime.viewer_protocol import (
    FIJI_HEARTBEAT,
    FijiPayloadKind,
    ViewerBatchMessageType,
    ViewerBatchContextWireField,
    ViewerBatchWireField,
    ViewerControlMessageType,
    ViewerControlReplyHeader,
    ViewerControlReplyPayload,
    ViewerComponentValueOrdering,
    ViewerProtocolStatus,
    ViewerServerLaunchRequest,
)
from openhcs.runtime.viewer_component_system import (
    ComponentValue,
    ViewerComponentAxisSemantics,
    ViewerBatchPayloadFields,
    ViewerComponentMetadataPayload,
    ViewerComponentNameMetadata,
    ViewerDimensionValueAuthority,
    ViewerDisplayBatchContext,
    ViewerObjectDisplayConfigInput,
    ViewerStreamingDataTypeHandler,
    ViewerStreamingDataTypeHandlerMeta,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from zmqruntime.config import ZMQConfig
from zmqruntime.transport import coerce_transport_mode
from zmqruntime.streaming import StreamingVisualizerServer

logger = logging.getLogger(__name__)
_ACK_ERROR = ViewerProtocolStatus.ERROR.value
_ACK_SUCCESS = ViewerProtocolStatus.SUCCESS.value

FijiWireScalar: TypeAlias = str | int | float | bool | None
FijiWireValue: TypeAlias = (
    FijiWireScalar
    | np.ndarray
    | np.dtype
    | tuple
    | list
    | dict
)
FijiDimensionComponents: TypeAlias = list[str]
FijiDimensionValues: TypeAlias = list[tuple]
FijiCoordinateKey: TypeAlias = tuple[tuple, tuple, tuple]
FijiDisplayRange: TypeAlias = tuple[float, float]
FijiDisplayRanges: TypeAlias = list[FijiDisplayRange]
FijiFixedLabels: TypeAlias = tuple[tuple[str, ComponentValue], ...]


@dataclass(frozen=True, slots=True)
class FijiDisplayItemContext(ViewerDisplayBatchContext[FijiDisplayConfig]):
    """Shared Fiji display context for item batches and deferred processing."""


@dataclass(frozen=True, slots=True)
class FijiBatchProcessingContext(FijiDisplayItemContext):
    """Semantic context carried through PolyStore's debounced batch engine."""

    images_dir: str | None


@dataclass(frozen=True, slots=True)
class FijiDimensionStorage:
    """Stored ImageJ C/Z/T domains for one Fiji window."""

    channel: FijiDimensionValues
    z_axis_values: FijiDimensionValues
    frame: FijiDimensionValues


@dataclass(frozen=True, slots=True)
class FijiSharedMemorySpec:
    """Shared-memory image payload fields required before local copying."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype


@dataclass(frozen=True, slots=True)
class FijiImagePayload:
    """Typed image payload used inside Fiji hyperstack construction."""

    data: np.ndarray
    metadata: Mapping[str, ComponentValue]
    image_id: str | None = None


@dataclass(frozen=True, slots=True)
class FijiWireItem(WindowProjectionPayloadProvider):
    """Nominal Fiji stream item with all raw wire-key semantics localized."""

    payload: dict[str, FijiWireValue]

    @classmethod
    def from_payload(cls, payload: FijiWireValue) -> "FijiWireItem":
        if not isinstance(payload, Mapping):
            raise TypeError("Fiji batch item must be a mapping.")
        return cls(dict(payload))

    @classmethod
    def from_payloads(cls, payloads: Sequence[FijiWireValue]) -> list["FijiWireItem"]:
        return [cls.from_payload(payload) for payload in payloads]

    def window_projection_payload(self) -> Mapping[str, FijiWireValue]:
        return self.payload

    @property
    def payload_kind(self) -> FijiPayloadKind | None:
        return FijiPayloadKind.from_payload(self.payload.get("data_type"))

    @property
    def payload_kind_text(self) -> str | None:
        value = self.payload.get("data_type")
        if value is None:
            return None
        return str(value)

    @property
    def image_id(self) -> str | None:
        value = self.payload.get("image_id")
        if value is None:
            return None
        return str(value)

    @property
    def metadata(self) -> Mapping[str, ComponentValue]:
        value = self.payload.get("metadata")
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("Fiji item metadata must be a mapping.")
        return ViewerComponentMetadataPayload.component_map(
            value,
            context="Fiji item metadata",
        )

    @property
    def data(self) -> np.ndarray:
        value = self.payload["data"]
        if not isinstance(value, np.ndarray):
            raise TypeError("Fiji image item data must be a numpy array.")
        return value

    @property
    def shared_memory(self) -> FijiSharedMemorySpec | None:
        kind = self.payload_kind
        if kind is None or not kind.uses_shared_memory or "shm_name" not in self.payload:
            return None
        return FijiSharedMemorySpec(
            name=str(self.payload["shm_name"]),
            shape=tuple(self.payload["shape"]),
            dtype=np.dtype(self.payload["dtype"]),
        )

    def with_local_data(self, data: np.ndarray) -> "FijiWireItem":
        copied = self.payload.copy()
        copied["data"] = data
        copied.pop("shm_name", None)
        copied.pop("shape", None)
        copied.pop("dtype", None)
        return FijiWireItem(copied)

    def image_payload(self) -> FijiImagePayload | None:
        if "data" not in self.payload:
            return None
        return FijiImagePayload(
            data=self.data,
            metadata=self.metadata,
            image_id=self.image_id,
        )

    @property
    def rois(self) -> Sequence[FijiWireValue]:
        value = self.payload.get("rois")
        if value is None:
            return ()
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise TypeError("Fiji ROI item rois must be a sequence.")
        return value

    @property
    def path(self) -> str:
        value = self.payload.get("path")
        if value is None:
            return "unknown"
        return str(value)


@dataclass(frozen=True, slots=True)
class FijiImageItemCollection:
    """Typed collection of Fiji image item payloads."""

    items: tuple[FijiImagePayload, ...]

    @classmethod
    def from_images(
        cls,
        images: Sequence[FijiImagePayload],
    ) -> "FijiImageItemCollection":
        return cls(tuple(images))

    def coordinate_payload_lookup(
        self,
        coordinates: "FijiHyperstackCoordinates",
    ) -> dict[FijiCoordinateKey, FijiImagePayload]:
        return {
            coordinates.key(item.metadata): item
            for item in self.items
        }

    def coordinate_data_lookup(
        self,
        coordinates: "FijiHyperstackCoordinates",
    ) -> dict[FijiCoordinateKey, np.ndarray]:
        return {
            coordinates.key(item.metadata): item.data
            for item in self.items
        }


@dataclass(frozen=True, slots=True)
class FijiImagePlaneLookup:
    """Image planes keyed by their shared OpenHCS C/Z/T coordinate tuple."""

    planes: dict[FijiCoordinateKey, np.ndarray]

    @classmethod
    def from_images(
        cls,
        images: Sequence[FijiImagePayload],
        coordinates: "FijiHyperstackCoordinates",
    ) -> "FijiImagePlaneLookup":
        return cls(
            FijiImageItemCollection.from_images(images).coordinate_data_lookup(
                coordinates
            )
        )

    def contains(self, key: FijiCoordinateKey) -> bool:
        return key in self.planes

    def data_for(self, key: FijiCoordinateKey) -> np.ndarray:
        return self.planes[key]


class FijiPlaneGeometry:
    """Canonical 2D plane extraction and spatial padding for Fiji stacks."""

    @staticmethod
    def extract_2d_plane(np_data: np.ndarray) -> np.ndarray:
        if np_data.ndim == 3:
            return np_data[np_data.shape[0] // 2]
        return np_data

    @staticmethod
    def pad_to_shape(
        np_data: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        current_height, current_width = np_data.shape[-2:]
        if current_height == target_height and current_width == target_width:
            return np_data

        if current_height > target_height or current_width > target_width:
            raise ValueError(
                f"Cannot pad plane {(current_height, current_width)} "
                f"to smaller target {(target_height, target_width)}"
            )

        padded = np.zeros((target_height, target_width), dtype=np_data.dtype)
        padded[:current_height, :current_width] = np_data
        return padded

    @classmethod
    def target_spatial_shape(
        cls,
        images: Sequence[FijiImagePayload],
    ) -> tuple[int, int]:
        max_height = 0
        max_width = 0
        for image in FijiImageItemCollection.from_images(images).items:
            plane = cls.extract_2d_plane(image.data)
            height, width = plane.shape[-2:]
            max_height = max(max_height, height)
            max_width = max(max_width, width)
        return max_height, max_width


@dataclass(frozen=True, slots=True)
class FijiImageIntensityRange:
    """Global display range across typed Fiji image payloads."""

    minimum: float
    maximum: float

    @classmethod
    def from_images(
        cls,
        images: Sequence[FijiImagePayload],
    ) -> "FijiImageIntensityRange":
        minimum = float("inf")
        maximum = float("-inf")
        for image in FijiImageItemCollection.from_images(images).items:
            plane = FijiPlaneGeometry.extract_2d_plane(image.data)
            minimum = min(minimum, float(np.min(plane)))
            maximum = max(maximum, float(np.max(plane)))
        return cls(minimum=minimum, maximum=maximum)


@dataclass(frozen=True, slots=True)
class FijiStackSliceLabelBuilder:
    """Build ImageJ slice labels from Fiji hyperstack coordinates."""

    coordinates: "FijiHyperstackCoordinates"

    def label_for(self, key: FijiCoordinateKey) -> str:
        c_key, z_key, t_key = key
        label_parts = []
        self._append_axis_label(label_parts, "C", c_key, self.coordinates.channel)
        self._append_axis_label(label_parts, "Z", z_key, self.coordinates.z_axis_coordinates)
        self._append_axis_label(label_parts, "T", t_key, self.coordinates.frame)
        if label_parts:
            return "_".join(label_parts)
        return "slice"

    @staticmethod
    def _append_axis_label(
        label_parts: list[str],
        prefix: str,
        axis_key: tuple,
        axis: "FijiDimensionAxis",
    ) -> None:
        if not axis.components:
            return
        label_parts.append(f"{prefix}{'_'.join(str(value) for value in axis_key)}")


@dataclass(frozen=True, slots=True)
class FijiImageStackBuilder:
    """Create an ImageJ ImageStack from typed Fiji image planes."""

    image_lookup: FijiImagePlaneLookup
    coordinates: "FijiHyperstackCoordinates"
    width: int
    height: int

    def build(self):
        import jpype
        import scyjava as sj

        ImageStack = sj.jimport("ij.ImageStack")
        stack = ImageStack(self.width, self.height)
        label_builder = FijiStackSliceLabelBuilder(self.coordinates)

        for t_key in self.coordinates.frame.values:
            for z_key in self.coordinates.z_axis_coordinates.values:
                for c_key in self.coordinates.channel.values:
                    key = (c_key, z_key, t_key)
                    if self.image_lookup.contains(key):
                        processor = self._processor_for_key(key, jpype, sj)
                        stack.addSlice(label_builder.label_for(key), processor)
                    else:
                        stack.addSlice("BLANK", self._blank_processor(sj))
        return stack

    def _processor_for_key(self, key: FijiCoordinateKey, jpype, sj):
        plane = FijiPlaneGeometry.extract_2d_plane(self.image_lookup.data_for(key))
        plane = FijiPlaneGeometry.pad_to_shape(plane, self.height, self.width)
        flattened = np.ascontiguousarray(plane).flatten()
        return self._processor_for_flattened_plane(flattened, jpype, sj)

    def _processor_for_flattened_plane(self, flattened: np.ndarray, jpype, sj):
        ByteProcessor = sj.jimport("ij.process.ByteProcessor")
        ShortProcessor = sj.jimport("ij.process.ShortProcessor")
        FloatProcessor = sj.jimport("ij.process.FloatProcessor")

        if flattened.dtype == np.uint8:
            java_array = jpype.JArray(jpype.JByte)(flattened.astype(np.int8))
            return ByteProcessor(self.width, self.height, java_array, None)
        if flattened.dtype in (np.uint16, np.int16):
            java_array = jpype.JArray(jpype.JShort)(flattened.astype(np.int16))
            return ShortProcessor(self.width, self.height, java_array, None)
        if flattened.dtype in (np.float32, np.float64):
            java_array = jpype.JArray(jpype.JFloat)(flattened.astype(np.float32))
            return FloatProcessor(self.width, self.height, java_array)

        java_array = jpype.JArray(jpype.JShort)(flattened.astype(np.int16))
        return ShortProcessor(self.width, self.height, java_array, None)

    def _blank_processor(self, sj):
        ShortProcessor = sj.jimport("ij.process.ShortProcessor")
        return ShortProcessor(self.width, self.height)


@dataclass(frozen=True, slots=True)
class FijiSharedMemoryItemCopier:
    """Copy shared-memory Fiji image payloads into local process memory."""

    send_error_ack: Callable[[str, str], None]

    def copy(self, items: Sequence[FijiWireItem]) -> list[FijiWireItem]:
        from multiprocessing import shared_memory

        copied_items = []
        for item in items:
            shared_memory_spec = item.shared_memory
            if shared_memory_spec is None:
                copied_items.append(item)
                continue

            try:
                shm = shared_memory.SharedMemory(name=shared_memory_spec.name)
                data = np.ndarray(
                    shared_memory_spec.shape,
                    dtype=shared_memory_spec.dtype,
                    buffer=shm.buf,
                ).copy()
                shm.close()
                shm.unlink()
                copied_items.append(item.with_local_data(data))
                logger.debug(
                    "📋 FIJI SERVER: Copied image data from shared memory %s",
                    shared_memory_spec.name,
                )
            except Exception as error:
                logger.error(
                    "📋 FIJI SERVER: Failed to copy from shared memory %s: %s",
                    shared_memory_spec.name,
                    error,
                )
                if item.image_id is not None:
                    self.send_error_ack(item.image_id, str(error))
        return copied_items


@dataclass(frozen=True)
class FijiDimensionAxis:
    """One ImageJ hyperstack axis backed by one or more OpenHCS components."""

    name: str
    components: FijiDimensionComponents
    values: FijiDimensionValues

    @classmethod
    def collect(
        cls,
        *,
        name: str,
        components: FijiDimensionComponents,
        items: Sequence[FijiWireItem],
    ) -> "FijiDimensionAxis":
        if not components:
            return cls(name=name, components=components, values=[()])
        values = {
            ViewerDimensionValueAuthority.value_tuple(item.metadata, components)
            for item in items
        }
        return cls(
            name=name,
            components=components,
            values=sorted(values, key=ViewerComponentValueOrdering.tuple_key),
        )

    @classmethod
    def collect_from_images(
        cls,
        *,
        name: str,
        components: FijiDimensionComponents,
        images: Sequence[FijiImagePayload],
    ) -> "FijiDimensionAxis":
        if not components:
            return cls(name=name, components=components, values=[()])
        values = {
            ViewerDimensionValueAuthority.value_tuple(image.metadata, components)
            for image in images
        }
        return cls(
            name=name,
            components=components,
            values=sorted(values, key=ViewerComponentValueOrdering.tuple_key),
        )

    def with_values(self, values: FijiDimensionValues) -> "FijiDimensionAxis":
        return FijiDimensionAxis(
            name=self.name,
            components=self.components,
            values=values,
        )

    def merge_values(self, values: FijiDimensionValues) -> "FijiDimensionAxis":
        return self.with_values(
            ViewerDimensionValueAuthority.merge(
                self.values,
                values,
            )
        )

    def value_tuple(self, metadata: Mapping[str, ComponentValue]) -> tuple:
        return ViewerDimensionValueAuthority.value_tuple(metadata, self.components)

    def index(self, metadata: Mapping[str, ComponentValue]) -> int:
        return ViewerDimensionValueAuthority.index(
            metadata,
            self.components,
            self.values,
        )

    def label_for_value(
        self,
        axis_value: Sequence[ComponentValue],
        component_names_metadata: ViewerComponentNameMetadata,
        *,
        fallback_label: str,
    ) -> str:
        labels = component_names_metadata.compact_tuple_labels(
            self.components,
            axis_value,
            context=f"Fiji {self.name!r} axis",
        )
        if labels:
            return " | ".join(labels)
        return fallback_label

    def labels_for_position(
        self,
        one_based_position: int,
        component_names_metadata: ViewerComponentNameMetadata,
    ) -> list[str]:
        if not self.components or not self.values:
            return []
        if one_based_position <= 0 or one_based_position > len(self.values):
            return []
        return component_names_metadata.compact_tuple_labels(
            self.components,
            self.values[one_based_position - 1],
            context=f"Fiji {self.name!r} axis",
        )


@dataclass(frozen=True)
class FijiHyperstackCoordinateComponents:
    """OpenHCS components assigned to ImageJ C/Z/T dimensions."""

    channel: FijiDimensionComponents
    z_axis_components: FijiDimensionComponents
    frame: FijiDimensionComponents

    def collect(self, items: Sequence[FijiWireItem]) -> "FijiHyperstackCoordinates":
        return FijiHyperstackCoordinates(
            channel=FijiDimensionAxis.collect(
                name="channel",
                components=self.channel,
                items=items,
            ),
            z_axis_coordinates=FijiDimensionAxis.collect(
                name="slice",
                components=self.z_axis_components,
                items=items,
            ),
            frame=FijiDimensionAxis.collect(
                name="frame",
                components=self.frame,
                items=items,
            ),
        )

    def collect_images(
        self,
        images: Sequence[FijiImagePayload],
    ) -> "FijiHyperstackCoordinates":
        return FijiHyperstackCoordinates(
            channel=FijiDimensionAxis.collect_from_images(
                name="channel",
                components=self.channel,
                images=images,
            ),
            z_axis_coordinates=FijiDimensionAxis.collect_from_images(
                name="z_axis",
                components=self.z_axis_components,
                images=images,
            ),
            frame=FijiDimensionAxis.collect_from_images(
                name="frame",
                components=self.frame,
                images=images,
            ),
        )


@dataclass(frozen=True)
class FijiHyperstackCoordinates:
    """Complete ImageJ C/Z/T coordinate domain for one Fiji window."""

    channel: FijiDimensionAxis
    z_axis_coordinates: FijiDimensionAxis
    frame: FijiDimensionAxis

    def merge_storage(
        self,
        stored: FijiDimensionStorage,
    ) -> "FijiHyperstackCoordinates":
        return FijiHyperstackCoordinates(
            channel=self.channel.merge_values(stored.channel),
            z_axis_coordinates=self.z_axis_coordinates.merge_values(stored.z_axis_values),
            frame=self.frame.merge_values(stored.frame),
        )

    def storage(self) -> FijiDimensionStorage:
        return FijiDimensionStorage(
            channel=self.channel.values,
            z_axis_values=self.z_axis_coordinates.values,
            frame=self.frame.values,
        )

    def dimensions(self) -> tuple[int, int, int]:
        return (
            len(self.channel.values),
            len(self.z_axis_coordinates.values),
            len(self.frame.values),
        )

    def key(self, metadata: Mapping[str, ComponentValue]) -> FijiCoordinateKey:
        return (
            self.channel.value_tuple(metadata),
            self.z_axis_coordinates.value_tuple(metadata),
            self.frame.value_tuple(metadata),
        )

    def imagej_position(
        self,
        metadata: Mapping[str, ComponentValue],
    ) -> tuple[int, int, int]:
        return (
            self.channel.index(metadata) + 1,
            self.z_axis_coordinates.index(metadata) + 1,
            self.frame.index(metadata) + 1,
        )

    def stack_index(self, key: FijiCoordinateKey) -> int:
        """Return the 1-based ImageJ stack position for a C/Z/T key."""
        c_key, z_key, t_key = key
        c_idx = self.channel.values.index(c_key)
        z_idx = self.z_axis_coordinates.values.index(z_key)
        t_idx = self.frame.values.index(t_key)
        n_channels, n_slices, _ = self.dimensions()
        return (t_idx * n_slices * n_channels) + (z_idx * n_channels) + c_idx + 1

    def contains_axis_values(self, key: FijiCoordinateKey) -> bool:
        c_key, z_key, t_key = key
        return (
            c_key in self.channel.values
            and z_key in self.z_axis_coordinates.values
            and t_key in self.frame.values
        )

    def label_text_for_position(
        self,
        *,
        imp,
        component_names_metadata: ViewerComponentNameMetadata,
        fixed_labels: FijiFixedLabels,
    ) -> str | None:
        labels = [
            component_names_metadata.compact_label(component, value)
            for component, value in fixed_labels
        ]
        labels.extend(
            self.channel.labels_for_position(
                imp.getChannel(),
                component_names_metadata,
            )
        )
        labels.extend(
            self.z_axis_coordinates.labels_for_position(
                imp.getSlice(),
                component_names_metadata,
            )
        )
        labels.extend(
            self.frame.labels_for_position(
                imp.getFrame(),
                component_names_metadata,
            )
        )
        if not labels:
            return None
        return " | ".join(labels)


@dataclass(frozen=True, slots=True)
class FijiWindowCoordinateResolution:
    """Coordinate domain resolved against stored per-window dimensions."""

    coordinates: FijiHyperstackCoordinates
    stored: FijiDimensionStorage | None = None

    @property
    def has_stored_dimensions(self) -> bool:
        return self.stored is not None

    def expanded_dimensions(self) -> bool:
        if self.stored is None:
            return False
        return (
            len(self.coordinates.channel.values) > len(self.stored.channel)
            or len(self.coordinates.z_axis_coordinates.values) > len(self.stored.z_axis_values)
            or len(self.coordinates.frame.values) > len(self.stored.frame)
        )


class FijiWindowState:
    """Mutable state for one Fiji window/hyperstack identity."""

    __slots__ = (
        "dimension_storage",
        "fixed_labels",
        "image_plus",
        "images",
        "roi_group_id",
        "window_key",
    )

    def __init__(
        self,
        window_key: str,
        image_plus=None,
        images: Sequence[FijiImagePayload] = (),
    ) -> None:
        self.window_key = window_key
        self.image_plus = image_plus
        self.images = list(images)
        self.dimension_storage: FijiDimensionStorage | None = None
        self.fixed_labels: FijiFixedLabels = ()
        self.roi_group_id: int | None = None

    def has_hyperstack(self) -> bool:
        return self.image_plus is not None

    def is_visible(self) -> bool:
        if self.image_plus is None:
            return False
        try:
            window = self.image_plus.getWindow()
            return window is not None and window.isVisible()
        except Exception:
            return False

    def close_hyperstack(self) -> None:
        if self.image_plus is None:
            return
        try:
            self.image_plus.close()
        except Exception:
            pass
        self.image_plus = None


class FijiWindowRegistry:
    """Own Fiji window state and keep window-scoped identities consistent."""

    def __init__(self) -> None:
        self._states: dict[str, FijiWindowState] = {}
        self._next_group_id = 1

    def state_for(self, window_key: str) -> FijiWindowState:
        if window_key not in self._states:
            self._states[window_key] = FijiWindowState(window_key)
        return self._states[window_key]

    def count_with_dimensions(self) -> int:
        return sum(
            1
            for state in self._states.values()
            if state.dimension_storage is not None
        )

    def clear_dimensions_and_labels(self) -> None:
        for state in self._states.values():
            state.dimension_storage = None
            state.fixed_labels = ()

    def resolve_coordinates(
        self,
        window_key: str,
        current_coordinates: FijiHyperstackCoordinates,
    ) -> FijiWindowCoordinateResolution:
        state = self._states.get(window_key)
        if state is None or state.dimension_storage is None:
            return FijiWindowCoordinateResolution(current_coordinates)
        return FijiWindowCoordinateResolution(
            current_coordinates.merge_storage(state.dimension_storage),
            state.dimension_storage,
        )

    def store_dimensions(
        self,
        window_key: str,
        coordinates: FijiHyperstackCoordinates,
    ) -> None:
        self.state_for(window_key).dimension_storage = coordinates.storage()

    def store_fixed_labels(
        self,
        window_key: str,
        fixed_labels: Sequence[tuple[str, ComponentValue]],
    ) -> None:
        self.state_for(window_key).fixed_labels = tuple(fixed_labels)

    def fixed_labels(self, window_key: str) -> FijiFixedLabels:
        state = self._states.get(window_key)
        if state is None:
            return ()
        return state.fixed_labels

    def open_hyperstack(self, window_key: str):
        state = self._states.get(window_key)
        if state is None or not state.has_hyperstack():
            return None
        if state.is_visible():
            return state.image_plus
        self.remove(window_key, close_hyperstack=False)
        return None

    def images(self, window_key: str) -> list[FijiImagePayload]:
        return list(self.state_for(window_key).images)

    def replace_images(
        self,
        window_key: str,
        images: Sequence[FijiImagePayload],
    ) -> None:
        self.state_for(window_key).images = list(images)

    def store_hyperstack(
        self,
        window_key: str,
        image_plus,
        images: Sequence[FijiImagePayload],
    ) -> None:
        state = self.state_for(window_key)
        if state.image_plus is not None and state.image_plus is not image_plus:
            state.close_hyperstack()
        state.image_plus = image_plus
        state.images = list(images)

    def remove(self, window_key: str, *, close_hyperstack: bool = True) -> None:
        state = self._states.pop(window_key, None)
        if state is not None and close_hyperstack:
            state.close_hyperstack()

    def group_id(self, window_key: str) -> int:
        state = self.state_for(window_key)
        if state.roi_group_id is None:
            state.roi_group_id = self._next_group_id
            self._next_group_id += 1
        return state.roi_group_id


@dataclass(frozen=True, slots=True)
class FijiItemBatch(FijiDisplayItemContext):
    """Shared Fiji item batch payload after display/config normalization."""

    items: list[FijiWireItem]


@dataclass(frozen=True, slots=True)
class FijiWindowProcessingRequest(FijiDisplayItemContext):
    """Inputs for processing one Fiji window projection."""

    window_key: str
    projection: "FijiWindowItemProjection"

    @property
    def items(self) -> list[FijiWireItem]:
        return self.projection.windows[self.window_key]

    @property
    def coordinate_components(self) -> FijiHyperstackCoordinateComponents:
        return self.projection.coordinate_components

    @property
    def fixed_labels(self) -> FijiFixedLabels:
        return self.projection.fixed_window_labels[self.window_key]


@dataclass(frozen=True, slots=True)
class FijiControlMessageResponse(ViewerControlReplyPayload):
    """Result of handling one Fiji control message."""

    shutdown_requested: bool = False


class FijiControlMessagePlan(ABC, metaclass=AutoRegisterMeta):
    """Executable behavior for one Fiji control message type."""

    __registry_key__ = "wire_value"
    __skip_if_no_key__ = True

    wire_value: ClassVar[str | None] = None

    @classmethod
    def for_message_type(cls, message_type: str | None) -> "FijiControlMessagePlan | None":
        if message_type is None:
            return None
        plan_type = cls.__registry__.get(message_type)
        if plan_type is None:
            return None
        return plan_type()

    @abstractmethod
    def response(self, windows: FijiWindowRegistry) -> FijiControlMessageResponse:
        """Return the response for one Fiji viewer control request."""


class FijiShutdownControlPlan(FijiControlMessagePlan):
    """Acknowledge shutdown and ask the server loop to stop."""

    def response(self, windows: FijiWindowRegistry) -> FijiControlMessageResponse:
        del windows
        logger.info(
            "🔬 FIJI SERVER: %s requested, will close after sending acknowledgment",
            self.wire_value,
        )
        return FijiControlMessageResponse(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type="shutdown_ack",
                message="Fiji viewer shutting down",
            ),
            shutdown_requested=True,
        )


class FijiGracefulShutdownControlPlan(FijiShutdownControlPlan):
    """Registered graceful Fiji shutdown control plan."""

    wire_value = "shutdown"


class FijiForceShutdownControlPlan(FijiShutdownControlPlan):
    """Registered force Fiji shutdown control plan."""

    wire_value = "force_shutdown"


class FijiClearStateControlPlan(FijiControlMessagePlan):
    """Clear Fiji dimension/window metadata without shutting down."""

    wire_value = "clear_state"

    def response(self, windows: FijiWindowRegistry) -> FijiControlMessageResponse:
        logger.info(
            "🔬 FIJI SERVER: Clearing dimension values (had %d windows)",
            windows.count_with_dimensions(),
        )
        windows.clear_dimensions_and_labels()
        return FijiControlMessageResponse(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type="clear_state_ack",
                message="Dimension values cleared",
            ),
        )


class FijiSettleControlPlan(FijiControlMessagePlan):
    """Acknowledge viewer-settle requests for the synchronous Fiji path."""

    wire_value = ViewerControlMessageType.SETTLE.value

    def response(self, windows: FijiWindowRegistry) -> FijiControlMessageResponse:
        del windows
        return FijiControlMessageResponse(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.SUCCESS,
                response_type="settle_ack",
                message="Fiji viewer has no queued debounced layer updates.",
            ),
        )


@dataclass(frozen=True, slots=True)
class FijiUnsupportedStateControlPlan(FijiControlMessagePlan):
    """Fail loudly for viewer-state polling until Fiji has a state projector."""

    wire_value = ViewerControlMessageType.STATE.value

    def response(self, windows: FijiWindowRegistry) -> FijiControlMessageResponse:
        del windows
        return FijiControlMessageResponse(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.ERROR,
                response_type="state_ack",
                message=(
                    "Fiji live viewer state polling is not implemented. "
                    "Use Napari state polling or add a Fiji state projection "
                    "before requesting layers, axes, labels, or payload summaries."
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class FijiUnsupportedPayloadsControlPlan(FijiControlMessagePlan):
    """Fail loudly for live payload extraction until Fiji has a state projector."""

    wire_value = ViewerControlMessageType.PAYLOADS.value

    def response(self, windows: FijiWindowRegistry) -> FijiControlMessageResponse:
        del windows
        return FijiControlMessageResponse(
            ViewerControlReplyHeader(
                ViewerProtocolStatus.ERROR,
                response_type="payloads_ack",
                message=(
                    "Fiji live viewer payload extraction is not implemented. "
                    "Add a Fiji state and payload projection before requesting "
                    "per-layer images, labels, shapes, or axis-coordinate payloads."
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class FijiControlMessageAuthority:
    """Handle Fiji control messages without leaking control literals into server."""

    windows: FijiWindowRegistry

    def response_for(
        self,
        message: Mapping[str, FijiWireValue],
    ) -> FijiControlMessageResponse:
        plan = FijiControlMessagePlan.for_message_type(
            None if "type" not in message else str(message["type"])
        )
        if plan is None:
            return FijiControlMessageResponse(
                ViewerControlReplyHeader(ViewerProtocolStatus.SUCCESS)
            )
        return plan.response(self.windows)


@dataclass(frozen=True, slots=True)
class FijiDisplayConfigWireAdapter:
    """Rehydrate OpenHCS FijiDisplayConfig from a serialized stream payload."""

    payload: Mapping[str, FijiWireValue]

    @classmethod
    def from_payload(cls, payload: FijiWireValue) -> "FijiDisplayConfigWireAdapter":
        if not isinstance(payload, Mapping):
            raise TypeError("Fiji batch message 'display_config' must be a mapping.")
        return cls(payload)

    def to_config(self) -> FijiDisplayConfig:
        component_modes = self._required_mapping("component_modes")
        self._validate_component_order(self._required_sequence("component_order"))
        return FijiDisplayConfig(
            lut=self._lut(self._required_value("lut")),
            auto_contrast=self._auto_contrast(self._required_value("auto_contrast")),
            **self._component_mode_fields(component_modes),
        )

    def _required_value(self, field_name: str) -> FijiWireValue:
        if field_name not in self.payload:
            raise ValueError(
                f"Fiji batch message 'display_config' missing {field_name!r}."
            )
        return self.payload[field_name]

    def _required_mapping(self, field_name: str) -> Mapping[str, FijiWireValue]:
        return self._required_typed_value(field_name, Mapping, "mapping")

    def _required_sequence(self, field_name: str) -> Sequence[FijiWireValue]:
        value = self._required_typed_value(field_name, Sequence, "sequence")
        if isinstance(value, str):
            raise TypeError(
                f"Fiji batch message display_config[{field_name!r}] must be a sequence."
            )
        return value

    def _required_typed_value(
        self,
        field_name: str,
        expected_type: type | tuple[type, ...],
        expected_name: str,
    ) -> FijiWireValue:
        value = self._required_value(field_name)
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Fiji batch message display_config[{field_name!r}] must be "
                f"a {expected_name}."
            )
        return value

    @staticmethod
    def _validate_component_order(component_order: Sequence[FijiWireValue]) -> None:
        received_order = tuple(str(component) for component in component_order)
        expected_order = tuple(
            str(component) for component in FijiDisplayConfig.COMPONENT_ORDER
        )
        if received_order != expected_order:
            raise ValueError(
                "Fiji display config component_order does not match the receiver "
                f"FijiDisplayConfig order: got {received_order!r}, expected "
                f"{expected_order!r}."
            )

    @staticmethod
    def _lut(value: FijiWireValue) -> FijiLUT:
        if isinstance(value, FijiLUT):
            return value
        try:
            return FijiLUT(str(value))
        except ValueError as error:
            raise ValueError(f"Unknown Fiji LUT value {value!r}.") from error

    @staticmethod
    def _auto_contrast(value: FijiWireValue) -> bool:
        if isinstance(value, bool):
            return value
        raise TypeError(f"Fiji auto_contrast must be bool, got {value!r}.")

    @staticmethod
    def _component_mode_fields(
        component_modes: Mapping[str, FijiWireValue],
    ) -> dict[str, FijiDimensionMode]:
        fields = FijiDisplayConfig.__dataclass_fields__
        mode_fields = {}
        for component, mode_value in component_modes.items():
            field_name = f"{component}_mode"
            if field_name not in fields:
                raise ValueError(
                    f"Fiji display config contains unknown component {component!r}."
                )
            try:
                mode_fields[field_name] = FijiDimensionMode(str(mode_value))
            except ValueError as error:
                raise ValueError(
                    f"Unknown Fiji dimension mode {mode_value!r} for "
                    f"component {component!r}."
                ) from error
        return mode_fields


@dataclass(frozen=True, slots=True)
class FijiPayloadHandlerRequest(FijiDisplayItemContext):
    """Typed execution request for one Fiji payload handler."""

    server: "FijiViewerServer"
    window_key: str
    items: list[FijiWireItem]
    coordinates: FijiHyperstackCoordinates


class FijiPayloadHandler(
    ViewerStreamingDataTypeHandler[FijiPayloadHandlerRequest],
    metaclass=ViewerStreamingDataTypeHandlerMeta,
):
    """Executable handler for one Fiji streaming payload kind."""


FIJI_BATCH_REQUIRED_FIELDS: tuple[ViewerBatchWireField | ViewerBatchContextWireField, ...] = (
    ViewerBatchWireField.IMAGES,
    ViewerBatchWireField.DISPLAY_CONFIG,
    ViewerBatchContextWireField.IMAGES_DIR,
    ViewerBatchWireField.COMPONENT_NAMES_METADATA,
    ViewerBatchWireField.COMPONENT_VALUE_DOMAIN,
)


@dataclass(frozen=True, slots=True)
class FijiBatchWireParser:
    """Validate and normalize one raw Fiji batch message."""

    payload: Mapping[str, FijiWireValue]

    def batch_message(self) -> "FijiBatchMessage":
        fields = ViewerBatchPayloadFields(self.payload, "Fiji batch message")
        fields.require_batch_message()
        fields.require_fields(FIJI_BATCH_REQUIRED_FIELDS)
        raw_items = fields.required_sequence(ViewerBatchWireField.IMAGES)
        display_config = FijiDisplayConfigWireAdapter.from_payload(
            fields.required_value(ViewerBatchWireField.DISPLAY_CONFIG)
        ).to_config()
        component_axis_semantics = fields.component_axis_semantics(
            ViewerObjectDisplayConfigInput(display_config),
            context="Fiji component value domain",
        )
        component_names_metadata = fields.required_component_names_metadata(
            context="Fiji batch component-name metadata",
        )
        return FijiBatchMessage(
            items=FijiWireItem.from_payloads(raw_items),
            viewer_display_config=display_config,
            images_dir=fields.required_optional_string(
                ViewerBatchContextWireField.IMAGES_DIR
            ),
            store=component_names_metadata.store,
            entries=component_axis_semantics.entries,
            layout=component_axis_semantics.layout,
        )


@dataclass(frozen=True)
class FijiBatchMessage(FijiItemBatch):
    """Validated Fiji stream batch payload."""

    images_dir: str | None

    def processing_context(self) -> "FijiBatchProcessingContext":
        return FijiBatchProcessingContext(
            viewer_display_config=self.viewer_display_config,
            images_dir=self.images_dir,
            store=self.store,
            entries=self.entries,
            layout=self.layout,
        )


@dataclass(frozen=True, slots=True)
class FijiPayloadKindItemGroup:
    """Items for one registered Fiji streaming handler type."""

    payload_stream_data_type: StreamingDataType
    items: list[FijiWireItem] = field(default_factory=list)

    def accepts(self, payload_kind: FijiPayloadKind) -> bool:
        return self.payload_stream_data_type is payload_kind.streaming_data_type

    def add(self, item: FijiWireItem) -> None:
        self.items.append(item)


@dataclass(frozen=True, slots=True)
class FijiPayloadKindItemBuckets:
    """Mutable buckets for Fiji items keyed by registered handler type."""

    groups: tuple[FijiPayloadKindItemGroup, ...]

    @classmethod
    def for_registered_handlers(cls) -> "FijiPayloadKindItemBuckets":
        return cls(
            tuple(
                FijiPayloadKindItemGroup(data_type)
                for data_type in FijiPayloadHandler.registered_data_types()
            )
        )

    def add(self, payload_kind: FijiPayloadKind, item: FijiWireItem) -> None:
        for group in self.groups:
            if group.accepts(payload_kind):
                group.add(item)
                return
        raise ValueError(f"No Fiji handler bucket for payload kind {payload_kind!r}.")

    def nonempty(self) -> tuple[FijiPayloadKindItemGroup, ...]:
        return tuple(group for group in self.groups if group.items)


@dataclass(frozen=True, slots=True)
class FijiPayloadKindGroups:
    """Fiji items grouped by registered streaming payload kind."""

    buckets: FijiPayloadKindItemBuckets

    @classmethod
    def from_items(cls, items: Sequence[FijiWireItem]) -> "FijiPayloadKindGroups":
        buckets = FijiPayloadKindItemBuckets.for_registered_handlers()
        for item in items:
            payload_kind = item.payload_kind
            if payload_kind is None:
                logger.warning(
                    "🔬 FIJI SERVER: Unknown data type string: %s",
                    item.payload_kind_text,
                )
                continue
            buckets.add(payload_kind, item)
        return cls(buckets)

    def dispatch(
        self,
        server: "FijiViewerServer",
        request: FijiWindowProcessingRequest,
        coordinates: FijiHyperstackCoordinates,
    ) -> None:
        for group in self.buckets.nonempty():
            FijiPayloadHandler.for_data_type(group.payload_stream_data_type).handle(
                FijiPayloadHandlerRequest(
                    server=server,
                    window_key=request.window_key,
                    items=group.items,
                    viewer_display_config=request.viewer_display_config,
                    coordinates=coordinates,
                    store=request.store,
                    entries=request.entries,
                    layout=request.layout,
                )
            )

@dataclass(frozen=True, slots=True)
class FijiWindowItemProjection(GroupedWindowItems[FijiWireItem]):
    """Fiji-owned bridge from component layout to windowed item processing."""

    coordinate_components: FijiHyperstackCoordinateComponents
    component_value_counts: tuple[tuple[str, int], ...]

    @classmethod
    def from_items(
        cls,
        items: Sequence[FijiWireItem],
        component_axis_semantics: ViewerComponentAxisSemantics,
    ) -> "FijiWindowItemProjection":
        projection = component_axis_semantics.layout.group_window_payload_providers(
            items
        )
        return cls(
            window_components=projection.window_components,
            channel_components=projection.channel_components,
            slice_components=projection.slice_components,
            frame_components=projection.frame_components,
            coordinate_components=FijiHyperstackCoordinateComponents(
                channel=projection.channel_components,
                z_axis_components=projection.slice_components,
                frame=projection.frame_components,
            ),
            windows=projection.windows,
            fixed_window_labels=projection.fixed_window_labels,
            component_value_counts=component_axis_semantics.component_value_counts(
                component_axis_semantics.layout.component_order
            ),
        )

    def log_summary(self) -> None:
        logger.info(
            "🔍 FIJI SERVER: Component cardinality: %s",
            list(self.component_value_counts),
        )
        logger.info("🗂️  FIJI SERVER: Dimension mapping:")
        logger.info("  WINDOW: %s", self.window_components)
        logger.info("  CHANNEL: %s", self.coordinate_components.channel)
        logger.info("  SLICE: %s", self.coordinate_components.z_axis_components)
        logger.info("  FRAME: %s", self.coordinate_components.frame)


@dataclass(slots=True)
class FijiBatchProcessingAuthority:
    """Own Fiji batch ingestion, debouncing, and window dispatch."""

    server: "FijiViewerServer"
    engine: DebouncedBatchEngine = field(init=False)
    hyperstack_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.engine = DebouncedBatchEngine(
            process_fn=self.process_items_with_context,
            debounce_delay_ms=self.server.DEBOUNCE_DELAY_MS,
            max_debounce_wait_ms=self.server.MAX_DEBOUNCE_WAIT_MS,
        )

    def copy_items_from_shared_memory(
        self,
        items: list[FijiWireItem],
    ) -> list[FijiWireItem]:
        """Copy shared-memory item payloads before acknowledging the sender."""
        return FijiSharedMemoryItemCopier(
            lambda image_id, error: self.server._send_ack(
                image_id,
                status=_ACK_ERROR,
                error=error,
            )
        ).copy(items)

    def queue(
        self,
        items: list[FijiWireItem],
        batch_context: FijiBatchProcessingContext,
    ) -> None:
        """Queue copied items for debounced batch processing."""
        self.engine.enqueue(
            items=items,
            context={"batch_context": batch_context},
        )

    def flush(self) -> None:
        """Force immediate processing of the pending batch."""
        self.engine.flush()

    def process_items_with_context(
        self,
        items: list[FijiWireItem],
        context: Mapping[str, FijiBatchProcessingContext],
    ) -> None:
        """Batch-engine callback that unpacks context into canonical arguments."""
        if not items:
            return
        batch_context = context.get("batch_context")
        if not isinstance(batch_context, FijiBatchProcessingContext):
            raise TypeError(
                "Fiji debounced batch context missing FijiBatchProcessingContext."
            )
        logger.info(f"🔄 FIJI SERVER: Processing debounced batch of {len(items)} items")
        self.process_items(items, batch_context)

    def process_image_message(self, message: bytes) -> dict:
        """Parse, copy, and queue one incoming Fiji batch message."""
        import json

        try:
            data = json.loads(message.decode("utf-8"))
            batch_message = FijiBatchWireParser(data).batch_message()

            logger.info(
                "📨 FIJI SERVER: Received batch message with %d items",
                len(batch_message.items),
            )

            if not batch_message.items:
                return {
                    "status": ViewerProtocolStatus.SUCCESS.value,
                    "message": "Empty batch",
                }

            copied_items = self.copy_items_from_shared_memory(batch_message.items)
            self.queue(
                copied_items,
                batch_message.processing_context(),
            )

            return {
                "status": ViewerProtocolStatus.SUCCESS.value,
                "message": "Data copied, queued for processing",
            }

        except Exception as e:
            logger.error(
                f"📨 FIJI SERVER: Error processing message: {e}", exc_info=True
            )
            return {"status": "error", "message": str(e)}

    def process_items(
        self,
        items: list[FijiWireItem],
        batch_context: FijiBatchProcessingContext,
    ) -> None:
        """Project a copied batch into Fiji window groups and dispatch handlers."""
        if not items:
            return

        projection = FijiWindowItemProjection.from_items(
            items,
            batch_context,
        )
        projection.log_summary()

        for window_key in projection.windows:
            self.process_window_group(
                FijiWindowProcessingRequest(
                    window_key=window_key,
                    viewer_display_config=batch_context.viewer_display_config,
                    projection=projection,
                    store=batch_context.store,
                    entries=batch_context.entries,
                    layout=batch_context.layout,
                )
            )

    def process_wire_items(
        self,
        *,
        items: Sequence[FijiWireValue],
        display_config: Mapping[str, FijiWireValue],
        images_dir: str | None,
        component_names_metadata: Mapping[str, FijiWireValue],
        component_value_domain: Mapping[str, Sequence[FijiWireValue]],
    ) -> None:
        """Convert raw receiver batch fields into nominal Fiji processing context."""
        batch_message = FijiBatchWireParser(
            {
                ViewerBatchWireField.TYPE.value: ViewerBatchMessageType.BATCH.value,
                ViewerBatchWireField.IMAGES.value: items,
                ViewerBatchWireField.DISPLAY_CONFIG.value: display_config,
                ViewerBatchContextWireField.IMAGES_DIR.value: images_dir,
                ViewerBatchWireField.COMPONENT_NAMES_METADATA.value: component_names_metadata,
                ViewerBatchWireField.COMPONENT_VALUE_DOMAIN.value: component_value_domain,
            }
        ).batch_message()
        self.process_items(
            batch_message.items,
            batch_message.processing_context(),
        )

    def process_window_group(
        self,
        request: FijiWindowProcessingRequest,
    ) -> None:
        """Process all items for one Fiji window group."""
        with self.hyperstack_lock:
            self.process_window_group_locked(request)

    def process_window_group_locked(
        self,
        request: FijiWindowProcessingRequest,
    ) -> None:
        """Process a Fiji window group while the hyperstack lock is held."""
        coordinate_resolution = self.server.windows.resolve_coordinates(
            request.window_key,
            request.coordinate_components.collect(request.items),
        )
        coordinates = coordinate_resolution.coordinates

        if coordinate_resolution.has_stored_dimensions:
            stored = coordinate_resolution.stored
            if coordinate_resolution.expanded_dimensions():
                logger.info(
                    f"🔬 FIJI SERVER: Expanded dimensions for window '{request.window_key}': "
                    f"{len(stored.channel)}→{len(coordinates.channel.values)}C, "
                    f"{len(stored.z_axis_values)}→{len(coordinates.z_axis_coordinates.values)}Z, "
                    f"{len(stored.frame)}→{len(coordinates.frame.values)}T"
                )
            else:
                logger.info(
                    f"🔬 FIJI SERVER: Reusing stored dimension values for window '{request.window_key}'"
                )
        else:
            logger.info(
                f"🔬 FIJI SERVER: First batch for window '{request.window_key}': "
                f"{len(coordinates.channel.values)}C x "
                f"{len(coordinates.z_axis_coordinates.values)}Z x "
                f"{len(coordinates.frame.values)}T"
            )

        self.server.windows.store_fixed_labels(request.window_key, request.fixed_labels)
        self.server.windows.store_dimensions(request.window_key, coordinates)

        FijiPayloadKindGroups.from_items(request.items).dispatch(
            self.server,
            request,
            coordinates,
        )


@dataclass(frozen=True, slots=True)
class FijiViewerServerLaunchConfig(ViewerServerLaunchRequest):
    """Nominal launch configuration for a Fiji viewer server process."""

    fiji_viewer_title: str
    fiji_display_config: FijiDisplayConfig | None
    zmq_config: ZMQConfig | None = None

    @property
    def resolved_zmq_config(self) -> ZMQConfig:
        if self.zmq_config is not None:
            return self.zmq_config
        return OPENHCS_ZMQ_CONFIG


@dataclass(frozen=True, slots=True)
class FijiInteractiveModeFailure:
    """Classify PyImageJ interactive-mode startup failures at the boundary."""

    error: OSError

    CANNOT_ENABLE_INTERACTIVE_MODE: ClassVar[str] = "Cannot enable interactive mode"

    def is_supported_headless_fallback(self) -> bool:
        return any(
            str(argument) == self.CANNOT_ENABLE_INTERACTIVE_MODE
            for argument in self.error.args
        )


class FijiViewerServer(StreamingVisualizerServer):
    """
    ZMQ server for Fiji viewer that receives images from clients.

    Inherits from ZMQServer ABC to get ping/pong, port management, etc.
    Uses SUB socket to receive images from pipeline clients.
    Displays images via PyImageJ.
    """

    _server_type = "fiji"  # Registration key for AutoRegisterMeta

    # Debouncing configuration
    DEBOUNCE_DELAY_MS = 500  # Collect items for 500ms before processing
    MAX_DEBOUNCE_WAIT_MS = (
        2000  # Maximum wait time before forcing batch processing (2s)
    )

    def __init__(self, launch_config: FijiViewerServerLaunchConfig):
        """
        Initialize Fiji viewer server.

        Args:
            launch_config: Nominal server launch configuration.
        """
        import zmq

        # Initialize with REP socket for receiving images (synchronous request/reply)
        # REP socket forces workers to wait for acknowledgment before closing shared memory
        super().__init__(
            launch_config.port,
            viewer_type="fiji",
            host="*",
            log_file_path=launch_config.log_file_path,
            data_socket_type=zmq.REP,
            transport_mode=coerce_transport_mode(launch_config.transport_mode),
            config=launch_config.resolved_zmq_config,
        )

        self.ij = None  # PyImageJ instance
        self._shutdown_requested = False
        self.windows = FijiWindowRegistry()
        self.batch_processor = FijiBatchProcessingAuthority(self)

    def _setup_ack_socket(self):
        """Setup PUSH socket for sending acknowledgments."""
        super()._setup_ack_socket()

    def _send_ack(self, image_id: str, status: str = "success", error: str = None):
        """Send acknowledgment that an image was processed.

        Args:
            image_id: UUID of the processed image
            status: 'success' or 'error'
            error: Error message if status='error'
        """
        self.send_ack(image_id, status=status, error=error)

    def _wait_for_swing_ui_ready(self, timeout: float = 5.0) -> bool:
        """Wait for Java Swing UI to be fully initialized.

        This is critical for IPC mode where messages arrive very fast.
        RoiManager and other Swing components require the EDT to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if UI is ready, False if timeout
        """
        import time
        import scyjava as sj

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try to access UIManager and verify UIDefaults are populated
                # This is critical because RoiManager needs JList UI components
                UIManager = sj.jimport("javax.swing.UIManager")
                look_and_feel = UIManager.getLookAndFeel()

                if look_and_feel is not None:
                    # Additional check: verify UIDefaults has JList UI class
                    # This is what RoiManager needs (it contains a JList)
                    ui_defaults = UIManager.getDefaults()
                    list_ui = ui_defaults.get("ListUI")

                    if list_ui is not None:
                        logger.info(
                            "🔬 FIJI SERVER: Java Swing UI is ready (UIDefaults populated)"
                        )
                        return True
                    else:
                        logger.debug(
                            "🔬 FIJI SERVER: Waiting for UIDefaults to populate..."
                        )

            except Exception as e:
                logger.debug(f"🔬 FIJI SERVER: Waiting for Swing UI: {e}")
            time.sleep(0.1)

        logger.warning("🔬 FIJI SERVER: Timeout waiting for Swing UI initialization")
        return False

    def start(self):
        """Start server and initialize PyImageJ."""
        super().start()

        # Initialize PyImageJ in this process
        try:
            import imagej

            logger.info("🔬 FIJI SERVER: Initializing PyImageJ...")

            # Try interactive mode first, fall back to headless mode on macOS
            try:
                self.ij = imagej.init(mode="interactive")
                # Show Fiji UI so users can interact with images and menus
                self.ij.ui().showUI()
                logger.info(
                    "🔬 FIJI SERVER: PyImageJ initialized in interactive mode with UI shown"
                )

                # Wait for Java Swing UI to be fully initialized
                # This is critical for IPC mode where messages arrive very fast
                # RoiManager creation requires the Swing event dispatch thread to be ready
                if not self._wait_for_swing_ui_ready(timeout=5.0):
                    logger.warning(
                        "🔬 FIJI SERVER: Swing UI may not be fully initialized, proceeding anyway"
                    )

            except OSError as e:
                if FijiInteractiveModeFailure(e).is_supported_headless_fallback():
                    logger.warning(
                        "🔬 FIJI SERVER: Interactive mode failed (likely macOS), using headless mode"
                    )
                    self.ij = imagej.init(mode="headless")
                    logger.info("🔬 FIJI SERVER: PyImageJ initialized in headless mode")
                else:
                    raise
        except ImportError:
            raise ImportError(
                "PyImageJ not available. Install with: pip install 'openhcs[viz]'"
            )

    def _create_pong_response(self) -> dict[str, FijiWireScalar]:
        """Override to add Fiji-specific fields and memory usage."""
        return FIJI_HEARTBEAT.apply_to(super()._create_pong_response())

    def handle_control_message(
        self,
        message: Mapping[str, FijiWireValue],
    ) -> dict[str, FijiWireScalar]:
        """Handle control messages beyond ping/pong."""
        response = FijiControlMessageAuthority(self.windows).response_for(message)
        if response.shutdown_requested:
            self._shutdown_requested = True
        return response.to_wire_mapping()

    def handle_data_message(self, message: Mapping[str, FijiWireValue]):
        """Handle incoming image data - called by process_messages()."""
        pass

    def display_image(self, image_data, metadata: dict) -> None:
        """Display a single image payload (no-op; Fiji uses batch processing)."""
        return

    def _add_slices_to_existing_hyperstack(
        self,
        existing_imp,
        new_images: list[FijiImagePayload],
        window_key: str,
        coordinates: FijiHyperstackCoordinates,
        display_config: FijiDisplayConfig,
        component_names_metadata: ViewerComponentNameMetadata,
    ):
        """
        Incrementally add new slices to an existing hyperstack WITHOUT rebuilding.

        This avoids the expensive min/max recalculation that happens when rebuilding.
        """
        # Get existing metadata
        existing_images = self.windows.images(window_key)
        existing_collection = FijiImageItemCollection.from_images(existing_images)
        new_collection = FijiImageItemCollection.from_images(new_images)

        # Build lookup of existing images by coordinates
        # CRITICAL: Use same key construction as new images to avoid false positives
        existing_lookup = existing_collection.coordinate_payload_lookup(coordinates)

        # Get existing stack and dimensions
        stack = existing_imp.getStack()
        stack_width = stack.getWidth()
        stack_height = stack.getHeight()
        old_nChannels = existing_imp.getNChannels()
        old_nSlices = existing_imp.getNSlices()
        old_nFrames = existing_imp.getNFrames()

        # Collect dimension values from existing images
        existing_coordinates = FijiHyperstackCoordinateComponents(
            channel=coordinates.channel.components,
            z_axis_components=coordinates.z_axis_coordinates.components,
            frame=coordinates.frame.components,
        ).collect_images(existing_images)

        # Process new images and check whether the coordinate domain changed.
        new_coords_added = []
        for item in new_collection.items:
            coord = coordinates.key(item.metadata)

            # Check if this is a new coordinate or replacement
            if coord not in existing_lookup:
                new_coords_added.append(coord)

            # Update lookup (new images override existing at same coordinates)
            existing_lookup[coord] = item

        # Check if dimensions actually changed
        # NEW LOGIC: Check if any coordinate has a dimension value not in existing dimension values
        # This is more accurate than checking if coordinates exist in existing_lookup
        dimensions_changed = False
        spatial_dimensions_changed = False
        for coord in new_coords_added:
            if not existing_coordinates.contains_axis_values(coord):
                dimensions_changed = True
                break

        if not dimensions_changed:
            for item in new_collection.items:
                np_data = FijiPlaneGeometry.extract_2d_plane(item.data)
                height, width = np_data.shape[-2:]
                if height > stack_height or width > stack_width:
                    spatial_dimensions_changed = True
                    break

        if not dimensions_changed and not spatial_dimensions_changed:
            # OPTIMIZATION: Only slice replacements - do INCREMENTAL UPDATE
            # Replace only changed slices in existing ImageStack WITHOUT rebuilding
            # This avoids recalculating contrast for ALL images
            logger.info(
                f"🔬 FIJI SERVER: ⚡ INCREMENTAL: Replacing {len(new_images)} slices in '{window_key}' (no rebuild, no recalc)"
            )

            # Map of new pixel data to replace
            new_pixel_data = {}
            for item in new_collection.items:
                np_data = FijiPlaneGeometry.extract_2d_plane(item.data)
                np_data = FijiPlaneGeometry.pad_to_shape(
                    np_data,
                    stack_height,
                    stack_width,
                )

                coord = coordinates.key(item.metadata)
                slice_idx = coordinates.stack_index(coord)

                new_pixel_data[slice_idx] = np_data

            # CRITICAL: Replace ALL changed slices in ONE call to avoid repeated min/max recalc
            # This is the key to avoiding fibonacci performance
            for slice_idx, np_data in new_pixel_data.items():
                stack.setPixels(slice_idx, np_data)

            # Update metadata
            self.windows.replace_images(window_key, list(existing_lookup.values()))

            # CRITICAL: Do NOT apply auto-contrast during incremental updates!
            # Only repaint window - auto-contrast will be applied on FINAL batch
            # This avoids O(n) auto-contrast for every single slice
            imp = existing_imp
            imp.updateAndRepaintWindow()

            logger.info(
                f"🔬 FIJI SERVER: ✅ Incremental update complete for '{window_key}'"
            )
        else:
            # Dimensions changed - need FULL REBUILD
            # ImageJ hyperstacks have fixed dimensions (C/Z/T and spatial width/height)
            # Preserve display ranges to avoid expensive min/max recalculation
            all_images = list(existing_lookup.values())
            logger.info(
                f"🔬 FIJI SERVER: 🔄 REBUILDING: Merging {len(new_images)} new images into "
                f"'{window_key}' (total: {len(all_images)} images, existing had {len(existing_images)}, "
                f"spatial_changed={spatial_dimensions_changed}, coord_changed={dimensions_changed})"
            )

            # Store display range before rebuilding
            display_ranges = []
            if old_nChannels > 0:
                for c in range(1, old_nChannels + 1):
                    try:
                        existing_imp.setC(c)
                        display_ranges.append(
                            (
                                existing_imp.getDisplayRangeMin(),
                                existing_imp.getDisplayRangeMax(),
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to get display range for channel {c}: {e}"
                        )
                        # Use default range if we can't get it
                        display_ranges.append((0, 255))

            # Close old hyperstack
            existing_imp.close()

            # Build new hyperstack with all images (old + new)
            # Pass is_new=False and preserved_display_ranges to avoid recalculation.
            self._build_new_hyperstack(
                all_images,
                window_key,
                coordinates,
                display_config,
                is_new=False,
                preserved_display_ranges=display_ranges,
                component_names_metadata=component_names_metadata,
            )

    def _build_single_hyperstack(
        self,
        window_key: str,
        images: list[FijiImagePayload],
        display_config: FijiDisplayConfig,
        coordinates: FijiHyperstackCoordinates,
        component_names_metadata: ViewerComponentNameMetadata | None = None,
    ):
        """
        Build or update a single ImageJ hyperstack from images.

        If a hyperstack already exists for this window_key, merge new images into it.
        Otherwise, create a new hyperstack.

        Args:
            window_key: Unique key for this window
            images: List of image data dicts (new images to add)
            display_config: Fiji display configuration
            coordinates: Shared ImageJ C/Z/T coordinate domain
            component_names_metadata: Optional component value labels for display
        """
        existing_imp = self.windows.open_hyperstack(window_key)
        is_new_hyperstack = existing_imp is None

        if not is_new_hyperstack:
            # INCREMENTAL UPDATE: Add only new slices to existing hyperstack
            logger.info(
                f"🔬 FIJI SERVER: ⚡ BATCH UPDATE: Adding {len(images)} new images to existing hyperstack '{window_key}'"
            )
            self._add_slices_to_existing_hyperstack(
                existing_imp,
                images,
                window_key,
                coordinates,
                display_config,
                component_names_metadata or ViewerComponentNameMetadata.empty(),
            )
            return

        # NEW HYPERSTACK: Build from scratch
        logger.info(
            f"🔬 FIJI SERVER: ✨ NEW HYPERSTACK: Creating '{window_key}' with {len(images)} images"
        )
        self._build_new_hyperstack(
            images,
            window_key,
            coordinates,
            display_config,
            is_new=True,
            component_names_metadata=component_names_metadata,
        )

    def _convert_to_hyperstack(self, imp, nChannels, nSlices, nFrames, window_key):
        """Convert ImagePlus to HyperStack with proper dimensions.

        Returns:
            ImagePlus or CompositeImage
        """
        import scyjava as sj

        # Set hyperstack dimensions
        imp.setDimensions(nChannels, nSlices, nFrames)

        # Convert to HyperStack to enable proper Z/T slider behavior
        if nSlices > 1 or nFrames > 1 or nChannels > 1:
            HyperStackConverter = sj.jimport("ij.plugin.HyperStackConverter")
            imp = HyperStackConverter.toHyperStack(
                imp, nChannels, nSlices, nFrames, "xyczt", "Composite"
            )
            imp.setTitle(window_key)

        # Convert to CompositeImage if multiple channels
        if nChannels > 1:
            CompositeImage = sj.jimport("ij.CompositeImage")
            if not isinstance(imp, CompositeImage):
                comp = CompositeImage(imp, CompositeImage.COMPOSITE)
                comp.setTitle(window_key)
                imp = comp

        return imp

    def _apply_display_settings(
        self,
        imp,
        window_key: str,
        lut_name,
        auto_contrast,
        nChannels,
        preserved_ranges=None,
        skip_auto_contrast=False,
    ):
        """Apply LUT and display settings to ImagePlus.

        Args:
            imp: ImagePlus to modify
            window_key: Hyperstack window key (for logging)
            lut_name: LUT name to apply
            auto_contrast: Whether to apply auto-contrast
            nChannels: Number of channels
            preserved_ranges: Optional list of (min, max) tuples per channel
            skip_auto_contrast: If True, skip auto-contrast even if auto_contrast=True
                                Used during loading to avoid O(n) recalc on every slice
        """
        if preserved_ranges:
            # Restore preserved display ranges
            for c in range(1, min(nChannels, len(preserved_ranges)) + 1):
                min_val, max_val = preserved_ranges[c - 1]
                imp.setC(c)
                imp.setDisplayRange(min_val, max_val)
        else:
            # Apply LUT and auto-contrast for new hyperstacks
            if lut_name not in ["Grays", "grays"] and nChannels == 1:
                try:
                    self.ij.IJ.run(imp, lut_name, "")
                except Exception as e:
                    logger.warning(
                        f"🔬 FIJI SERVER: Failed to apply LUT {lut_name}: {e}"
                    )

            # CRITICAL: Skip auto-contrast during incremental updates to avoid O(n) recalc on every slice
            # Only apply on final batch when skip_auto_contrast=False
            if auto_contrast and not skip_auto_contrast:
                try:
                    self.ij.IJ.run(imp, "Enhance Contrast", "saturated=0.35")
                    logger.info(
                        f"🔬 FIJI SERVER: ✅ Applied auto-contrast to '{window_key}' "
                        f"(nChannels={nChannels})"
                    )
                except Exception as e:
                    logger.warning(
                        f"🔬 FIJI SERVER: Failed to apply auto-contrast: {e}"
                    )

    def _create_dimension_label_overlay(
        self,
        window_key: str,
        imp,
        coordinates: FijiHyperstackCoordinates,
        component_names_metadata: ViewerComponentNameMetadata,
    ):
        """
        Create a text overlay showing current dimension labels (like napari's text_overlay).

        This creates an actual on-screen text overlay that updates when dimensions change,
        matching napari's behavior.

        Args:
            window_key: Window key for fixed window-component labels
            imp: ImagePlus instance
            coordinates: Shared ImageJ C/Z/T coordinate domain
            component_names_metadata: Dict mapping component names to {id: name} dicts
        """
        import scyjava as sj

        try:
            logger.info(f"🏷️  FIJI SERVER: Creating dimension label overlay")

            label_text = coordinates.label_text_for_position(
                imp=imp,
                component_names_metadata=component_names_metadata,
                fixed_labels=self.windows.fixed_labels(window_key),
            )
            if label_text is None:
                logger.info(
                    f"🏷️  FIJI SERVER: No dimensions to label (no channels/slices/frames)"
                )
                return
            logger.info(f"🏷️  FIJI SERVER: Creating overlay with text: '{label_text}'")

            # Create text overlay using ImageJ Overlay API
            TextRoi = sj.jimport("ij.gui.TextRoi")
            Overlay = sj.jimport("ij.gui.Overlay")
            Font = sj.jimport("java.awt.Font")
            Color = sj.jimport("java.awt.Color")

            # Position text in top-left corner
            x = 10
            y = 20

            # Create text ROI with white fill color
            text_roi = TextRoi(x, y, label_text)
            text_roi.setFont(Font("SansSerif", Font.BOLD, 16))

            # Set fill color to white (this is what shows)
            text_roi.setFillColor(Color.WHITE)
            # Set stroke (outline) color to black for contrast
            text_roi.setStrokeColor(Color.BLACK)
            text_roi.setStrokeWidth(2.0)

            # Create or get overlay
            overlay = imp.getOverlay()
            if overlay is None:
                overlay = Overlay()

            # Clear any existing overlays first
            overlay.clear()

            # Add text ROI to overlay
            overlay.add(text_roi)
            imp.setOverlay(overlay)

            logger.info(
                f"🏷️  FIJI SERVER: Text overlay created successfully with white text"
            )

            # Add listener to update overlay when hyperstack position changes
            self._add_dimension_change_listener(
                window_key,
                imp,
                coordinates,
                component_names_metadata,
            )

        except Exception as e:
            logger.error(
                f"🏷️  FIJI SERVER: Failed to create dimension label overlay: {e}",
                exc_info=True,
            )

    def _add_dimension_change_listener(
        self,
        window_key: str,
        imp,
        coordinates: FijiHyperstackCoordinates,
        component_names_metadata: ViewerComponentNameMetadata,
    ):
        """
        Add a listener to update the text overlay when hyperstack position changes.

        Uses AdjustmentListener on the StackWindow scrollbars to detect dimension changes.
        Based on ImageJ source: ij.gui.StackWindow implements AdjustmentListener for its scrollbars.
        """
        import scyjava as sj
        import jpype

        try:
            def build_label_text():
                return coordinates.label_text_for_position(
                    imp=imp,
                    component_names_metadata=component_names_metadata,
                    fixed_labels=self.windows.fixed_labels(window_key),
                )

            # Helper function to update overlay text
            def update_overlay():
                try:
                    label_text = build_label_text()
                    if label_text is None:
                        logger.info("🔄 FIJI SERVER: No overlay labels to update")
                        return
                    logger.info(
                        f"🔄 FIJI SERVER: Updating overlay text to: '{label_text}'"
                    )

                    TextRoi = sj.jimport("ij.gui.TextRoi")
                    Overlay = sj.jimport("ij.gui.Overlay")
                    Font = sj.jimport("java.awt.Font")
                    Color = sj.jimport("java.awt.Color")

                    text_roi = TextRoi(10, 20, label_text)
                    text_roi.setFont(Font("SansSerif", Font.BOLD, 16))
                    text_roi.setFillColor(Color.WHITE)
                    text_roi.setStrokeColor(Color.BLACK)
                    text_roi.setStrokeWidth(2.0)

                    overlay = Overlay()
                    overlay.add(text_roi)
                    imp.setOverlay(overlay)
                    # Force canvas repaint - this triggers the overlay redraw
                    canvas = imp.getCanvas()
                    if canvas is not None:
                        canvas.repaint()
                    logger.info(f"🔄 FIJI SERVER: Overlay updated successfully")
                except Exception as e:
                    logger.error(
                        f"🔄 FIJI SERVER: Error updating overlay: {e}", exc_info=True
                    )

            # Get the StackWindow and add AdjustmentListener to its scrollbars
            window = imp.getWindow()
            if window is not None:
                try:
                    # Define listener class using jpype @JImplements decorator
                    @jpype.JImplements("java.awt.event.AdjustmentListener")
                    class DimensionScrollbarListener:
                        @jpype.JOverride
                        def adjustmentValueChanged(self, event):
                            # Called when user scrolls through C/Z/T dimensions
                            update_overlay()

                    listener = DimensionScrollbarListener()

                    # StackWindow has cSelector, zSelector, tSelector scrollbars
                    # ImageJ only creates scrollbars when that dimension > 1
                    # JPype Java fields must be accessed directly.
                    added = []

                    logger.info(f"🏷️  FIJI SERVER: Window type: {type(window).__name__}")
                    logger.info(
                        f"🏷️  FIJI SERVER: Hyperstack: {imp.getNChannels()}C x {imp.getNSlices()}Z x {imp.getNFrames()}T"
                    )

                    # Scrollbars are AWT components, not fields
                    # Find all ScrollbarWithLabel components and attach listeners
                    import scyjava as sj

                    try:
                        ScrollbarWithLabel = sj.jimport("ij.gui.ScrollbarWithLabel")
                        components = window.getComponents()
                        logger.info(
                            f"🏷️  FIJI SERVER: Window has {len(components)} components"
                        )

                        scrollbar_count = 0
                        for i, comp in enumerate(components):
                            comp_type = type(comp).__name__
                            logger.info(f"🏷️  FIJI SERVER:   Component {i}: {comp_type}")

                            # ScrollbarWithLabel is the hyperstack dimension scrollbar
                            if isinstance(comp, ScrollbarWithLabel):
                                try:
                                    # Just attach listener to all scrollbars
                                    # The update_overlay() function will read current position from imp
                                    comp.addAdjustmentListener(listener)
                                    scrollbar_count += 1
                                    added.append(f"scrollbar_{i}")
                                    logger.info(
                                        f"🏷️  FIJI SERVER:     Added listener to scrollbar {i}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"🏷️  FIJI SERVER:     Could not attach listener: {e}"
                                    )

                        logger.info(
                            f"🏷️  FIJI SERVER: Attached listeners to {scrollbar_count} scrollbars"
                        )
                    except Exception as e:
                        logger.warning(
                            f"🏷️  FIJI SERVER: Could not enumerate components: {e}"
                        )

                    if added:
                        logger.info(
                            f"🏷️  FIJI SERVER: Added scrollbar listeners for: {', '.join(added)}"
                        )
                    else:
                        logger.warning(
                            f"🏷️  FIJI SERVER: No scrollbars found to attach listener (not a hyperstack?)"
                        )

                    # Add WindowListener to detect when user closes the window
                    # Capture closure variables explicitly
                    captured_window_key = window_key
                    captured_windows = self.windows
                    captured_lock = self.batch_processor.hyperstack_lock

                    @jpype.JImplements("java.awt.event.WindowListener")
                    class WindowCloseListener:
                        @jpype.JOverride
                        def windowClosing(self, event):
                            with captured_lock:
                                captured_windows.remove(
                                    captured_window_key,
                                    close_hyperstack=False,
                                )
                                logger.info(
                                    f"🔬 FIJI SERVER: Cleaned up hyperstack '{captured_window_key}' after window close"
                                )

                        @jpype.JOverride
                        def windowClosed(self, event):
                            pass

                        @jpype.JOverride
                        def windowOpened(self, event):
                            pass

                        @jpype.JOverride
                        def windowIconified(self, event):
                            pass

                        @jpype.JOverride
                        def windowDeiconified(self, event):
                            pass

                        @jpype.JOverride
                        def windowActivated(self, event):
                            pass

                        @jpype.JOverride
                        def windowDeactivated(self, event):
                            pass

                    window.addWindowListener(WindowCloseListener())
                    logger.info(
                        f"🏷️  FIJI SERVER: Added window close listener for '{window_key}'"
                    )

                except Exception as e:
                    logger.warning(
                        f"Could not add scrollbar listeners: {e}", exc_info=True
                    )
            else:
                logger.warning(f"🏷️  FIJI SERVER: No window found, cannot add listeners")

        except Exception as e:
            logger.error(
                f"🏷️  FIJI SERVER: Failed to add dimension change listener: {e}",
                exc_info=True,
            )

    def _set_dimension_labels(
        self,
        imp,
        coordinates: FijiHyperstackCoordinates,
        component_names_metadata: ViewerComponentNameMetadata,
    ):
        """
        Set dimension labels on ImagePlus using component metadata.

        Sets both:
        1. ImageJ property metadata (for channel selector UI)
        2. Text overlay (for on-screen display like napari)

        Args:
            imp: ImagePlus instance
            coordinates: Shared ImageJ C/Z/T coordinate domain
            component_names_metadata: Dict mapping component names to {id: name} dicts
        """
        try:
            logger.info(
                f"🏷️  FIJI SERVER: _set_dimension_labels called with {len(coordinates.channel.values)} channels"
            )
            logger.info(
                f"🏷️  FIJI SERVER: channel axis components = {coordinates.channel.components}"
            )
            logger.info(
                f"🏷️  FIJI SERVER: component_names_metadata = {component_names_metadata}"
            )

            # Set channel labels
            if coordinates.channel.components and coordinates.channel.values:
                logger.info(
                    f"🏷️  FIJI SERVER: Setting labels for {len(coordinates.channel.values)} channels"
                )
                for idx, channel_tuple in enumerate(coordinates.channel.values, start=1):
                    label = coordinates.channel.label_for_value(
                        channel_tuple,
                        component_names_metadata,
                        fallback_label=f"Ch{idx}",
                    )
                    imp.setProperty(f"Label{idx}", label)
                    logger.info(
                        f"🏷️  FIJI SERVER: Set channel {idx} label: '{label}' (property key: 'Label{idx}')"
                    )

                    # Verify the property was set
                    verified_label = imp.getProperty(f"Label{idx}")
                    logger.info(
                        f"🏷️  FIJI SERVER: Verified channel {idx} label: '{verified_label}'"
                    )

            # Note: ImageJ doesn't have built-in properties for slice/frame labels like it does for channels
            # Those would require custom overlays or other visualization methods

        except Exception as e:
            logger.warning(f"Failed to set dimension labels: {e}", exc_info=True)

    def _build_new_hyperstack(
        self,
        all_images: list[FijiImagePayload],
        window_key: str,
        coordinates: FijiHyperstackCoordinates,
        display_config: FijiDisplayConfig,
        is_new: bool,
        preserved_display_ranges: FijiDisplayRanges = None,
        component_names_metadata: ViewerComponentNameMetadata | None = None,
    ):
        """Build a new hyperstack from scratch."""
        import scyjava as sj

        nChannels, nSlices, nFrames = coordinates.dimensions()

        logger.info(
            f"🔬 FIJI SERVER: Building hyperstack '{window_key}': {nChannels}C x {nSlices}Z x {nFrames}T"
        )

        if not all_images:
            logger.error(f"🔬 FIJI SERVER: No images provided for '{window_key}'")
            return

        # Get target spatial dimensions (mixed shapes allowed by zero-padding smaller planes)
        height, width = FijiPlaneGeometry.target_spatial_shape(all_images)
        logger.info(
            f"🔬 FIJI SERVER: Target stack shape for '{window_key}' is {height}x{width}"
        )

        # Build image lookup
        image_lookup = FijiImagePlaneLookup.from_images(all_images, coordinates)

        # CRITICAL: Calculate global min/max ONCE from all images BEFORE creating stack
        # This avoids O(n) calculation per slice during stack creation
        logger.info(
            f"🔬 FIJI SERVER: Calculating global min/max from {len(all_images)} images"
        )
        intensity_range = FijiImageIntensityRange.from_images(all_images)
        logger.info(
            "🔬 FIJI SERVER: Global min/max: %s - %s",
            intensity_range.minimum,
            intensity_range.maximum,
        )

        # Create ImageStack
        stack = FijiImageStackBuilder(
            image_lookup=image_lookup,
            coordinates=coordinates,
            width=width,
            height=height,
        ).build()

        # Create ImagePlus
        ImagePlus = sj.jimport("ij.ImagePlus")
        imp = ImagePlus(window_key, stack)

        # Set display range using pre-calculated global min/max
        # This prevents ImageJ from scanning all pixels again
        imp.setDisplayRange(intensity_range.minimum, intensity_range.maximum)

        # Convert to hyperstack
        imp = self._convert_to_hyperstack(imp, nChannels, nSlices, nFrames, window_key)

        # Set dimension labels from metadata (e.g., channel names like "DAPI", "GFP")
        logger.info(
            f"🏷️  FIJI SERVER: component_names_metadata = {component_names_metadata}"
        )

        # Note: Text overlay will be created AFTER imp.show() so the window exists for listeners

        title_suffix = ""
        component_names_metadata = (
            component_names_metadata or ViewerComponentNameMetadata.empty()
        )
        self._set_dimension_labels(
            imp,
            coordinates,
            component_names_metadata,
        )

        if component_names_metadata:
            logger.info(f"🏷️  FIJI SERVER: Setting dimension labels for {window_key}")

            # For single-channel images, add channel name to window title since no slider appears
            if nChannels == 1 and coordinates.channel.components and coordinates.channel.values:
                first_comp = coordinates.channel.components[0]
                first_value_tuple = coordinates.channel.values[0]
                channel_name = component_names_metadata.display_name(
                    first_comp,
                    first_value_tuple[0],
                )
                if channel_name is not None:
                    title_suffix = f" [{channel_name}]"
                    logger.info(
                        f"🏷️  FIJI SERVER: Adding channel name to window title: {title_suffix}"
                    )
        else:
            logger.info(
                f"🏷️  FIJI SERVER: No component_names_metadata available for {window_key}"
            )

        # Update window title with suffix if present
        if title_suffix:
            imp.setTitle(f"{window_key}{title_suffix}")

        # Store BEFORE showing to prevent race condition where next batch arrives
        # before hyperstack is registered, causing duplicate window creation
        self.windows.store_hyperstack(window_key, imp, all_images)

        # Show after storing (imp.show() may be async on Swing thread)
        imp.show()

        if is_new:
            display_auto_contrast = display_config.auto_contrast
            display_ranges = None
        else:
            display_auto_contrast = False
            display_ranges = preserved_display_ranges

        # Apply display settings after the window is shown.
        # This keeps auto-contrast off the per-slice update path.
        self._apply_display_settings(
            imp,
            window_key,
            display_config.get_lut_name(),
            display_auto_contrast,
            nChannels,
            preserved_ranges=display_ranges,
            skip_auto_contrast=(not is_new) and (preserved_display_ranges is not None),
        )

        logger.info(
            f"🔬 FIJI SERVER: Displayed hyperstack '{window_key}' with {stack.getSize()} slices"
        )

        # NOW create text overlay AFTER window exists (so listeners can be attached)
        logger.info(
            f"🏷️  FIJI SERVER: Creating dimension label overlay for {window_key}"
        )
        self._create_dimension_label_overlay(
            window_key,
            imp,
            coordinates,
            component_names_metadata,
        )

        # Send acknowledgments
        for image in all_images:
            if image_id := image.image_id:
                self._send_ack(image_id, status=_ACK_SUCCESS)

    def request_shutdown(self):
        """Request graceful shutdown."""
        self._shutdown_requested = True
        self.stop()


@dataclass(frozen=True, slots=True)
class FijiImagePayloadHandler(FijiPayloadHandler):
    """Build or update Fiji hyperstacks from image payloads."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.IMAGE

    def handle(self, request: FijiPayloadHandlerRequest) -> None:
        image_data_list: list[FijiImagePayload] = []
        for item in request.items:
            image_payload = item.image_payload()
            if image_payload is not None:
                image_data_list.append(image_payload)
            elif item.shared_memory is not None:
                loaded = request.server.load_images_from_shared_memory(
                    [item.payload],
                    error_callback=request.server._send_ack,
                )
                image_data_list.extend(
                    payload
                    for loaded_item in loaded
                    if (
                        payload := FijiWireItem.from_payload(
                            loaded_item
                        ).image_payload()
                    ) is not None
                )

        if not image_data_list:
            return

        request.server._build_single_hyperstack(
            request.window_key,
            image_data_list,
            request.viewer_display_config,
            request.coordinates,
            request,
        )


@dataclass(frozen=True, slots=True)
class FijiRoiManagerProvider:
    """Create or retrieve the ImageJ ROI manager on the correct thread."""

    def manager(self):
        import scyjava as sj

        RoiManager = sj.jimport("ij.plugin.frame.RoiManager")
        roi_manager = RoiManager.getInstance()
        if roi_manager is not None:
            return roi_manager

        try:
            from jpype import JImplements, JOverride
        except ImportError:
            logger.warning(
                "JPype not available, creating RoiManager without EDT safety (may fail with IPC mode)"
            )
            return RoiManager()

        SwingUtilities = sj.jimport("javax.swing.SwingUtilities")
        roi_manager_holder = [None]

        @JImplements("java.lang.Runnable")
        class CreateRoiManagerRunnable:
            @JOverride
            def run(self):
                roi_manager_holder[0] = RoiManager()

        SwingUtilities.invokeAndWait(CreateRoiManagerRunnable())
        return roi_manager_holder[0]


@dataclass(frozen=True, slots=True)
class FijiRoiPayloadHandler(FijiPayloadHandler):
    """Add ROI payloads to ImageJ's ROI manager in hyperstack coordinates."""

    streaming_data_type: ClassVar[StreamingDataType] = StreamingDataType.ROIS
    roi_manager_provider: FijiRoiManagerProvider = field(
        default_factory=FijiRoiManagerProvider
    )

    def handle(self, request: FijiPayloadHandlerRequest) -> None:
        from pathlib import Path

        from polystore.roi_converters import FijiROIConverter
        import scyjava as sj

        roi_manager = self.roi_manager_provider.manager()
        group_id = request.server.windows.group_id(request.window_key)
        total_rois_added = 0

        for roi_item in request.items:
            rois_encoded = roi_item.rois
            if not rois_encoded:
                if image_id := roi_item.image_id:
                    request.server._send_ack(image_id, status=_ACK_SUCCESS)
                continue

            metadata = roi_item.metadata
            file_path = roi_item.path

            logger.info(f"🔬 FIJI SERVER: ROI metadata: {metadata}")
            logger.info(f"🔬 FIJI SERVER: Channel axis: {request.coordinates.channel}")
            logger.info(f"🔬 FIJI SERVER: Slice axis: {request.coordinates.z_axis_coordinates}")
            logger.info(f"🔬 FIJI SERVER: Frame axis: {request.coordinates.frame}")

            c_value, z_value, t_value = request.coordinates.imagej_position(metadata)

            logger.info(
                f"🔬 FIJI SERVER: ROI '{file_path}' position: C={c_value}, Z={z_value}, T={t_value}"
            )

            base_name = Path(file_path).stem

            java_rois = FijiROIConverter.transmission_to_java_rois(
                list(rois_encoded),
                sj,
            )
            for roi_idx, java_roi in enumerate(java_rois):
                java_roi.setName(f"{base_name}_{roi_idx:04d}")
                java_roi.setPosition(c_value, z_value, t_value)
                java_roi.setGroup(group_id)
                roi_manager.addRoi(java_roi)
                total_rois_added += 1

            if image_id := roi_item.image_id:
                request.server._send_ack(image_id, status=_ACK_SUCCESS)

        if not roi_manager.isVisible():
            roi_manager.setVisible(True)

        logger.info(
            f"🔬 FIJI SERVER: Added {total_rois_added} ROIs to group {group_id} ('{request.window_key}') with shared coordinate space"
        )


def fiji_viewer_server_process(
    port: int,
    viewer_title: str,
    display_config: FijiDisplayConfig | None,
    log_file_path: str = None,
    transport_mode: OpenHCSTransportMode = OpenHCSTransportMode.IPC,
    zmq_config: ZMQConfig | None = None,
):
    """
    Fiji viewer server process function.

    Runs in separate process to manage Fiji instance and handle incoming image data.

    Args:
        port: ZMQ port to listen on
        viewer_title: Title for the Fiji viewer window
        display_config: FijiDisplayConfig instance
        log_file_path: Path to log file (for client discovery via ping/pong)
        transport_mode: ZMQ transport mode (IPC or TCP)
        zmq_config: ZMQ configuration object (optional, uses default if None)
    """
    try:
        import zmq

        server = FijiViewerServer(
            FijiViewerServerLaunchConfig(
                port=port,
                fiji_viewer_title=viewer_title,
                fiji_display_config=display_config,
                log_file_path=log_file_path,
                transport_mode=transport_mode,
                zmq_config=zmq_config,
            )
        )

        # Start the server (binds sockets, initializes PyImageJ)
        server.start()

        logger.info(
            f"🔬 FIJI SERVER: Server started on port {port}, control port {port + 1000}"
        )
        logger.info("🔬 FIJI SERVER: Waiting for images...")

        # Message processing loop
        # REP socket requires sending reply after each receive (synchronous request/reply)
        while not server._shutdown_requested:
            # Process control messages (ping/pong handled by ABC)
            server.process_messages()

            # Process data messages (images) if ready
            if server._ready:
                # REP socket is synchronous - process one message at a time
                # Worker blocks until we send reply, ensuring no shared memory race conditions

                # CRITICAL: ZMQ REP sockets require strict recv->send->recv->send alternation
                # If recv() succeeds but send() doesn't happen, the socket enters an invalid
                # state and refuses all future recv() calls, causing the server to hang.

                # Step 1: Try to receive a message (non-blocking)
                try:
                    message = server.data_socket.recv(zmq.NOBLOCK)
                except zmq.Again:
                    # No messages available - this is normal
                    pass
                else:
                    # Step 2: We received a message, so we MUST send a response
                    try:
                        # Process the message and get acknowledgment
                        ack_response = server.batch_processor.process_image_message(
                            message
                        )
                    except Exception as e:
                        # ANY error during processing - send error response to maintain socket state
                        logger.error(
                            f"🔬 FIJI SERVER: Error processing image message: {e}",
                            exc_info=True,
                        )
                        ack_response = {"status": "error", "message": str(e)}

                    # Step 3: ALWAYS send response (even if it's an error response)
                    try:
                        server.data_socket.send_json(ack_response)
                        logger.info(
                            f"🔬 FIJI SERVER: Sent ack to worker: {ack_response['status']}"
                        )
                    except Exception as e:
                        # If send fails, the socket is likely broken - log and continue
                        logger.error(
                            f"🔬 FIJI SERVER: Failed to send ack on data socket: {e}",
                            exc_info=True,
                        )

            time.sleep(0.001)  # 1ms sleep - faster polling for multiprocessing

        logger.info("🔬 FIJI SERVER: Shutting down...")
        server.stop()

    except Exception as e:
        logger.error(f"🔬 FIJI SERVER: Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        logger.info("🔬 FIJI SERVER: Process terminated")
