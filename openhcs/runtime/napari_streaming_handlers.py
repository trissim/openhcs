"""Shared Napari streaming handler axis declarations."""

from __future__ import annotations

import colorsys
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, TypeAlias

import numpy as np
from napari.layers.shapes._shapes_constants import ShapeType

from polystore.streaming_constants import StreamingDataType
from polystore.streaming.identity import StreamProducerIdentity
from zmqruntime.viewer_protocol import ViewerComponentMode, ViewerWireField

from openhcs.core.artifacts import ObjectArtifactSubjectBinding
from openhcs.core.config import NapariDisplayConfig
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
)
from openhcs.runtime.viewer_protocol import (
    NapariLayerKind,
    ViewerComponentValueOrdering,
    ViewerSettlePhase,
    ViewerSettleProgress,
)
from openhcs.runtime.viewer_component_system import (
    ComponentMap,
    ComponentValue,
    ComponentValues,
    ViewerComponentAxisSemantics,
    ViewerComponentValueDomainPayload,
    ViewerLayerAxisProjection,
)

if TYPE_CHECKING:
    from polystore.streaming.receivers.napari import NapariBatchProcessor

    from openhcs.runtime.napari_viewer_server import NapariViewerServer


logger = logging.getLogger(__name__)


LayerKwargValue: TypeAlias = str | int | float | bool | tuple | list | dict | None
LayerData: TypeAlias = np.ndarray | list | tuple | str | int | float | bool | None
DimensionLabelMap: TypeAlias = dict[str, list[str]]
ShapePayloadValue: TypeAlias = LayerData | dict
ShapePayloadMap: TypeAlias = Mapping[str, ShapePayloadValue]


class VisualMetadataField(str, Enum):
    """Optional visual metadata fields attached to Napari shape payloads."""

    CENTROID = "centroid"
    LABEL = "label"
    AREA = "area"
    COMPONENT = "component"


class NapariLayerHandle(ABC):
    """Nominal marker for concrete layer objects returned by a Napari viewer."""


class NapariShapesLayerHandle(NapariLayerHandle):
    """Public mutation surface used for incremental native Shapes updates."""

    features: Mapping[str, Sequence[object]]
    edge_color: str
    face_color: str
    edge_color_mode: str
    face_color_mode: str
    edge_color_cycle: Sequence[tuple[float, float, float, float]]
    face_color_cycle: Sequence[tuple[float, float, float, float]]

    @abstractmethod
    def add(
        self,
        data: Sequence[np.ndarray],
        *,
        shape_type: Sequence[str],
        edge_color: Sequence[tuple[float, float, float, float]],
        face_color: Sequence[tuple[float, float, float, float]],
    ) -> None:
        """Append one bounded batch of Shapes data."""


class NapariLayerSelectionController(ABC):
    """Minimal layer-selection contract used by OpenHCS streaming."""

    @property
    @abstractmethod
    def active(self) -> NapariLayerHandle | None:
        """Return the currently selected Napari layer."""

    @active.setter
    @abstractmethod
    def active(self, layer: NapariLayerHandle | None) -> None:
        """Set the currently selected Napari layer."""


class NapariLayerCollection(ABC):
    """Minimal layer collection contract used by OpenHCS streaming."""

    selection: NapariLayerSelectionController

    @abstractmethod
    def remove(self, layer: NapariLayerHandle) -> None:
        """Remove a concrete Napari layer from the viewer."""

    @abstractmethod
    def __contains__(self, layer: NapariLayerHandle) -> bool:
        """Return whether the concrete Napari layer is still mounted."""

class NapariDimsController(ABC):
    """Subset of napari dims state mutated by streaming updates."""

    axis_labels: tuple[str, ...]
    ndim: int


class NapariViewerLayerCreator(ABC):
    """Explicit layer creation contract instead of reflective add_* lookup."""

    layers: NapariLayerCollection
    dims: NapariDimsController

    @abstractmethod
    def add_image(
        self,
        data: LayerData,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create an image layer."""

    @abstractmethod
    def add_shapes(
        self,
        data: LayerData,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create a shapes layer."""

    @abstractmethod
    def add_points(
        self,
        data: LayerData,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create a points layer."""

    @abstractmethod
    def add_labels(
        self,
        data: LayerData,
        *,
        name: str,
        **kwargs: LayerKwargValue,
    ) -> NapariLayerHandle:
        """Create a labels layer."""


class NapariTimerHandle(ABC):
    """Timer contract stored by layer state."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the pending layer update."""


@dataclass(frozen=True, slots=True)
class NapariPendingLayerUpdate(ViewerComponentAxisSemantics):
    """Queued debounced layer update with flush-local runtime residue."""

    timer: NapariTimerHandle
    data_type: StreamingDataType
    display_config: NapariDisplayConfig

    @classmethod
    def from_semantics(
        cls,
        *,
        timer: NapariTimerHandle,
        data_type: StreamingDataType,
        semantics: ViewerComponentAxisSemantics,
        display_config: NapariDisplayConfig,
    ) -> "NapariPendingLayerUpdate":
        return cls(
            entries=semantics.entries,
            layout=semantics.layout,
            timer=timer,
            data_type=data_type,
            display_config=display_config,
        )

    def stop_timer(self) -> None:
        """Stop the Qt timer that would otherwise execute this update later."""

        self.timer.stop()


@dataclass(slots=True)
class NapariLayerSettlementState:
    """Own one incremental drain of queued Napari layer updates."""

    updates: tuple[tuple[str, NapariPendingLayerUpdate], ...]
    completed_update_count: int = 0
    active_route: str | None = None
    active_route_work_unit_count: int = 0
    active_route_work_unit_active: bool = False
    failed: bool = False
    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    @property
    def phase(self) -> ViewerSettlePhase:
        with self._lock:
            if self.failed:
                return ViewerSettlePhase.FAILED
            if (
                self.completed_update_count == len(self.updates)
                and self.active_route is None
            ):
                return ViewerSettlePhase.COMPLETE
            return ViewerSettlePhase.RUNNING

    def begin_next(
        self,
    ) -> tuple[str, NapariPendingLayerUpdate] | None:
        """Claim the next exact update for one Qt callback."""

        with self._lock:
            if self.phase is not ViewerSettlePhase.RUNNING:
                return None
            if self.active_route is not None:
                raise RuntimeError(
                    f"Napari settlement route {self.active_route!r} is already active."
                )
            route_key, update = self.updates[self.completed_update_count]
            self.active_route = route_key
            self.active_route_work_unit_count = 0
            self.active_route_work_unit_active = False
            return route_key, update

    def begin_active_work_unit(self, route_key: str) -> None:
        """Declare entry into one native display mutation."""

        with self._lock:
            if self.active_route != route_key:
                raise RuntimeError(
                    f"Cannot begin Napari work for route {route_key!r}; active "
                    f"route is {self.active_route!r}."
                )
            if self.active_route_work_unit_active:
                raise RuntimeError(
                    f"Napari settlement route {route_key!r} already has active work."
                )
            self.active_route_work_unit_active = True

    def complete_active_work_unit(self, route_key: str) -> None:
        """Record one bounded unit of forward progress for the active route."""

        with self._lock:
            if self.active_route != route_key:
                raise RuntimeError(
                    f"Cannot advance Napari settlement route {route_key!r}; active "
                    f"route is {self.active_route!r}."
                )
            if not self.active_route_work_unit_active:
                raise RuntimeError(
                    f"Napari settlement route {route_key!r} has no active work "
                    "unit to complete."
                )
            self.active_route_work_unit_active = False
            self.active_route_work_unit_count += 1

    def complete_active(self, route_key: str) -> None:
        """Record successful completion of the claimed update."""

        with self._lock:
            if self.active_route != route_key:
                raise RuntimeError(
                    f"Cannot complete Napari settlement route {route_key!r}; active "
                    f"route is {self.active_route!r}."
                )
            if not self.active_route_work_unit_active:
                raise RuntimeError(
                    f"Napari settlement route {route_key!r} has no active work "
                    "unit to finish."
                )
            self.completed_update_count += 1
            self.active_route = None
            self.active_route_work_unit_count = 0
            self.active_route_work_unit_active = False

    def fail_active(self, route_key: str) -> None:
        """Record terminal failure of the claimed update."""

        with self._lock:
            if self.active_route != route_key:
                raise RuntimeError(
                    f"Cannot fail Napari settlement route {route_key!r}; active route "
                    f"is {self.active_route!r}."
                )
            self.failed = True
            self.active_route = None
            self.active_route_work_unit_count = 0
            self.active_route_work_unit_active = False

    def fail(self) -> None:
        """Record a terminal settlement failure without an active route."""

        with self._lock:
            if self.active_route is not None:
                raise RuntimeError(
                    "Napari settlement has an active route; use fail_active()."
                )
            self.failed = True

    def progress(self) -> ViewerSettleProgress:
        """Project current settlement state onto the shared wire contract."""

        with self._lock:
            return ViewerSettleProgress(
                phase=self.phase,
                completed_update_count=self.completed_update_count,
                total_update_count=len(self.updates),
                active_route=self.active_route,
                active_route_work_unit_count=self.active_route_work_unit_count,
                active_route_work_unit_active=self.active_route_work_unit_active,
            )


@dataclass(frozen=True, slots=True)
class NapariStreamLayerAddress:
    """Component/path/data-type identity for one staged stream payload."""

    components: ComponentMap
    path: str
    stream_layer_data_type: StreamingDataType

    def with_components(self, components: ComponentMap) -> "NapariStreamLayerAddress":
        """Return the same payload address after component normalization."""
        return NapariStreamLayerAddress(
            components=components,
            path=self.path,
            stream_layer_data_type=self.stream_layer_data_type,
        )

    def same_layer_slot(self, other: "NapariStreamLayerAddress") -> bool:
        """Return whether two payloads replace the same component/data-type slot."""
        return (
            self.components == other.components
            and self.stream_layer_data_type == other.stream_layer_data_type
        )


@dataclass(frozen=True, slots=True)
class NapariStreamLayerItem:
    """One component-addressed payload staged for a Napari layer update."""

    data: LayerData
    producer: StreamProducerIdentity
    address: NapariStreamLayerAddress
    image_metadata: ImagePayloadMetadata
    plane_component_domain: ViewerComponentValueDomainPayload


class NapariImagePayloadAxisLabelPolicy:
    """Labels payload-local image stack axes before the spatial y/x axes."""

    SPATIAL_AXIS_COUNT = 2

    @classmethod
    def axis_labels(
        cls,
        data: LayerData,
        image_metadata: ImagePayloadMetadata,
        consumed_payload_axes: tuple[int, ...] = (),
    ) -> tuple[str, ...]:
        local_axis_count = len(cls.local_axis_indices(data, image_metadata))
        consumed = frozenset(consumed_payload_axes)
        unbound_axes = tuple(
            index for index in range(local_axis_count) if index not in consumed
        )
        if unbound_axes:
            raise ValueError(
                "Napari image payload exposes non-spatial, non-color axes without "
                "an exact OpenHCS component-axis binding: "
                f"{unbound_axes!r}."
            )
        return ()

    @classmethod
    def local_axis_indices(
        cls,
        data: LayerData,
        image_metadata: ImagePayloadMetadata,
    ) -> tuple[int, ...]:
        ndim = int(np.ndim(data))
        spatial_axes = image_metadata.spatial_axes_yx(data)
        if spatial_axes is None:
            raise ValueError("Napari image payload requires two declared spatial axes.")
        channel_axis = image_metadata.normalized_source_channel_axis(data)
        return tuple(
            axis
            for axis in range(ndim)
            if axis not in spatial_axes and axis != channel_axis
        )


@dataclass(frozen=True, slots=True)
class NapariAggregateAxisBinding:
    """Bind one payload-local plane axis to one viewer component axis."""

    component: str
    payload_axis: int
    values: tuple[ComponentValue, ...]

    @property
    def extent(self) -> int:
        return len(self.values)

    def component_value_for_payload_index(self, index: int) -> ComponentValue:
        if index < 0 or index >= len(self.values):
            raise ValueError(
                f"Payload index {index} is outside aggregate axis "
                f"{self.component!r} extent {len(self.values)}."
            )
        return self.values[index]


@dataclass(frozen=True, slots=True)
class NapariAggregateAxisBindingSet:
    """Aggregate axis bindings for one Napari display route."""

    bindings: tuple[NapariAggregateAxisBinding, ...] = ()

    @property
    def component_values(self) -> ComponentValues:
        return {binding.component: list(binding.values) for binding in self.bindings}

    @property
    def payload_axes(self) -> tuple[int, ...]:
        return tuple(binding.payload_axis for binding in self.bindings)

    @property
    def scalar_component_values(self) -> ComponentMap:
        """Return payload-axis components whose exact domain is one value."""

        return {
            binding.component: binding.values[0]
            for binding in self.bindings
            if binding.extent == 1
        }

    def item_scalar_components(self, item: NapariStreamLayerItem) -> ComponentMap:
        """Return the exact scalar identity represented by one payload item."""

        return {
            **item.address.components,
            **self.scalar_component_values,
        }

    def item_component_values(
        self,
        item: NapariStreamLayerItem,
        payload_indices: tuple[int, ...],
    ) -> ComponentMap:
        components = dict(item.address.components)
        if len(payload_indices) != len(self.bindings):
            raise ValueError(
                "Aggregate payload index cardinality mismatch: "
                f"{payload_indices!r} for {len(self.bindings)} binding(s)."
            )
        for binding, payload_index in zip(self.bindings, payload_indices):
            components[binding.component] = binding.component_value_for_payload_index(
                payload_index
            )
        return components

    def shape_component_values(
        self,
        item: NapariStreamLayerItem,
        shape_dict: ShapePayloadMap,
    ) -> ComponentMap:
        if not self.bindings:
            return dict(item.address.components)
        plane_indices = NapariShapePlaneMetadata(shape_dict).indices()
        return self.item_component_values(item, plane_indices)


class NapariAggregateAxisBindingAuthority:
    """Bind payload-local axes through their exact declared component domains."""

    @classmethod
    def bindings(
        cls,
        items: Sequence[NapariStreamLayerItem],
        component_axis_semantics: ViewerComponentAxisSemantics,
    ) -> NapariAggregateAxisBindingSet:
        if not items:
            return NapariAggregateAxisBindingSet()
        axis_components = component_axis_semantics.layout.components_for_mode(
            ViewerComponentMode.STACK
        )
        declared_component_values = component_axis_semantics.required_component_values(
            axis_components
        )
        extents = cls._aggregate_extents(items)
        if not extents:
            if any(item.plane_component_domain for item in items):
                raise ValueError(
                    "Napari payload declares plane component values without a "
                    "payload-local plane axis."
                )
            return NapariAggregateAxisBindingSet()
        plane_domains = tuple(item.plane_component_domain.entries for item in items)
        present_domains = tuple(domain for domain in plane_domains if domain)
        if not present_domains:
            raise ValueError(
                "Napari aggregate payload axes require an exact "
                "plane_component_values declaration."
            )
        first_domain = present_domains[0]
        if len(present_domains) != len(items) or any(
            domain != first_domain for domain in present_domains[1:]
        ):
            raise ValueError(
                "Napari aggregate payload route has inconsistent plane component "
                f"domains: {plane_domains!r}."
            )
        if len(first_domain) > len(extents):
            raise ValueError(
                "Napari aggregate payload declares more component axes than its "
                f"payload shape exposes: {len(first_domain)} > {len(extents)}."
            )

        route_component_values = cls._route_component_values(items, axis_components)
        bindings: list[NapariAggregateAxisBinding] = []
        used_components: set[str] = set()
        for payload_axis, entry in enumerate(first_domain):
            component = entry.component
            values = tuple(entry.values)
            if component not in axis_components:
                raise ValueError(
                    "Napari aggregate payload component axis is not configured for "
                    f"stack display: {component!r}."
                )
            if component in used_components:
                raise ValueError(
                    "Napari aggregate payload declares component axis more than once: "
                    f"{component!r}."
                )
            if len(values) != extents[payload_axis]:
                raise ValueError(
                    "Napari aggregate payload component axis cardinality mismatch: "
                    f"{component!r} declares {len(values)} value(s) for extent "
                    f"{extents[payload_axis]}."
                )
            unknown_values = tuple(
                value
                for value in values
                if value not in declared_component_values[component]
            )
            if unknown_values:
                raise ValueError(
                    "Napari aggregate payload component axis contains values outside "
                    f"the viewer domain for {component!r}: {unknown_values!r}."
                )
            if len(route_component_values[component]) > 1:
                raise ValueError(
                    "Napari aggregate payload component axis is also varying across "
                    f"routed items: {component!r}."
                )
            bindings.append(
                NapariAggregateAxisBinding(
                    component=component,
                    payload_axis=payload_axis,
                    values=values,
                )
            )
            used_components.add(component)
        return NapariAggregateAxisBindingSet(tuple(bindings))

    @staticmethod
    def _route_component_values(
        items: Sequence[NapariStreamLayerItem],
        axis_components: Sequence[str],
    ) -> ComponentValues:
        return {
            component: sorted(
                {
                    item.address.components[component]
                    for item in items
                    if component in item.address.components
                },
                key=ViewerComponentValueOrdering.key,
            )
            for component in axis_components
        }

    @classmethod
    def _aggregate_extents(
        cls,
        items: Sequence[NapariStreamLayerItem],
    ) -> tuple[int, ...]:
        extents = tuple(cls._item_aggregate_extents(item) for item in items)
        present = tuple(value for value in extents if value)
        if not present:
            return ()
        first = present[0]
        if any(value != first for value in present):
            raise ValueError(
                "Napari aggregate payload route has inconsistent internal "
                f"axis extents: {present!r}."
            )
        return first

    @classmethod
    def _item_aggregate_extents(
        cls,
        item: NapariStreamLayerItem,
    ) -> tuple[int, ...]:
        if item.address.stream_layer_data_type is StreamingDataType.IMAGE:
            return cls._image_aggregate_extents(item)
        if item.address.stream_layer_data_type is StreamingDataType.SHAPES:
            return cls._shape_aggregate_extents(item.data)
        return ()

    @staticmethod
    def _image_aggregate_extents(
        item: NapariStreamLayerItem,
    ) -> tuple[int, ...]:
        shape = tuple(int(dimension) for dimension in np.shape(item.data))
        return tuple(
            shape[axis]
            for axis in NapariImagePayloadAxisLabelPolicy.local_axis_indices(
                item.data,
                item.image_metadata,
            )
        )

    @staticmethod
    def _shape_aggregate_extents(data: LayerData) -> tuple[int, ...]:
        if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
            return ()
        shapes: list[tuple[int, ...]] = []
        missing_plane_metadata = 0
        for shape_dict in data:
            if not isinstance(shape_dict, Mapping):
                continue
            plane_metadata = NapariShapePlaneMetadata(shape_dict)
            if plane_metadata.has_plane_metadata():
                shapes.append(plane_metadata.shape())
            else:
                missing_plane_metadata += 1
        if not shapes:
            return ()
        if missing_plane_metadata:
            raise ValueError(
                "Napari shape payload mixes plane-indexed and unindexed shapes; "
                "all shapes in an aggregate stack route must carry plane metadata."
            )
        first = shapes[0]
        if any(shape != first for shape in shapes):
            raise ValueError(
                "Napari shape payload has inconsistent plane_shape metadata: "
                f"{shapes!r}."
            )
        return first


@dataclass(frozen=True, slots=True)
class NapariShapePlaneMetadata:
    """Plane-index metadata carried by one serialized ROI shape."""

    shape_dict: ShapePayloadMap

    @property
    def metadata(self) -> Mapping[str, ShapePayloadValue]:
        metadata = self.shape_dict.get("metadata")
        if not isinstance(metadata, Mapping):
            return {}
        return metadata

    def has_plane_metadata(self) -> bool:
        return "plane_indices" in self.metadata and "plane_shape" in self.metadata

    def indices(self) -> tuple[int, ...]:
        return self._tuple_field("plane_indices")

    def shape(self) -> tuple[int, ...]:
        return self._tuple_field("plane_shape")

    def _tuple_field(self, field: str) -> tuple[int, ...]:
        value = self.metadata.get(field)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ValueError(
                f"Napari shape plane metadata field {field!r} must be a sequence."
            )
        return tuple(int(item) for item in value)


NapariLayerCreator: TypeAlias = Callable[
    [NapariViewerLayerCreator, LayerData, str, Mapping[str, LayerKwargValue]],
    NapariLayerHandle,
]
NapariLayerCreationLogger: TypeAlias = Callable[
    [NapariLayerKind, str, LayerData],
    None,
]


def _log_created_layer(
    layer_kind: NapariLayerKind,
    layer_name: str,
    data: LayerData,
) -> None:
    del data
    logger.info(
        "🔬 NAPARI PROCESS: Created %s layer %s",
        layer_kind.value,
        layer_name,
    )


def _log_created_item_layer(
    layer_kind: NapariLayerKind,
    layer_name: str,
    data: LayerData,
) -> None:
    logger.info(
        "🔬 NAPARI PROCESS: Created %s layer %s with %d %s",
        layer_kind.value,
        layer_name,
        len(data),
        layer_kind.value,
    )


def _complete_layer_kind_mapping(
    mapping: Mapping[NapariLayerKind, NapariLayerCreator],
) -> Mapping[NapariLayerKind, NapariLayerCreator]:
    missing = set(NapariLayerKind) - set(mapping)
    if missing:
        raise ValueError(f"Missing Napari layer creators for {missing!r}.")
    return mapping


def _complete_layer_log_mapping(
    mapping: Mapping[NapariLayerKind, NapariLayerCreationLogger],
) -> Mapping[NapariLayerKind, NapariLayerCreationLogger]:
    missing = set(NapariLayerKind) - set(mapping)
    if missing:
        raise ValueError(f"Missing Napari layer loggers for {missing!r}.")
    return mapping


NAPARI_LAYER_CREATORS = _complete_layer_kind_mapping(
    {
        NapariLayerKind.IMAGE: (
            lambda viewer, data, layer_name, layer_kwargs: viewer.add_image(
                data,
                name=layer_name,
                **dict(layer_kwargs),
            )
        ),
        NapariLayerKind.SHAPES: (
            lambda viewer, data, layer_name, layer_kwargs: viewer.add_shapes(
                data,
                name=layer_name,
                **dict(layer_kwargs),
            )
        ),
        NapariLayerKind.POINTS: (
            lambda viewer, data, layer_name, layer_kwargs: viewer.add_points(
                data,
                name=layer_name,
                **dict(layer_kwargs),
            )
        ),
        NapariLayerKind.LABELS: (
            lambda viewer, data, layer_name, layer_kwargs: viewer.add_labels(
                data,
                name=layer_name,
                **dict(layer_kwargs),
            )
        ),
    }
)
NAPARI_LAYER_CREATED_LOGGERS = _complete_layer_log_mapping(
    {
        NapariLayerKind.IMAGE: _log_created_layer,
        NapariLayerKind.SHAPES: _log_created_item_layer,
        NapariLayerKind.POINTS: _log_created_item_layer,
        NapariLayerKind.LABELS: _log_created_layer,
    }
)


class NapariImageLayerPresentationPolicy:
    """Formal defaults for streamed Napari image layer presentation."""

    DEFAULT_COLORMAP = "gray"
    DEFAULT_BLEND_MODE = "additive"

    @classmethod
    def colormap(cls, colormap: str | None) -> str:
        if colormap is None:
            return cls.DEFAULT_COLORMAP
        return colormap

    @classmethod
    def layer_kwargs(
        cls,
        image_data: LayerData,
        image_metadata: ImagePayloadMetadata,
        colormap: str | None,
    ) -> dict[str, LayerKwargValue]:
        kwargs: dict[str, LayerKwargValue] = {"blending": cls.DEFAULT_BLEND_MODE}
        if cls.is_rgb(image_data, image_metadata):
            kwargs["rgb"] = True
            return kwargs
        kwargs["colormap"] = cls.colormap(colormap)
        return kwargs

    @classmethod
    def is_rgb(
        cls,
        image_data: LayerData,
        image_metadata: ImagePayloadMetadata,
    ) -> bool:
        channel_axis = image_metadata.normalized_source_channel_axis(image_data)
        if channel_axis is None:
            return False
        shape = tuple(int(dimension) for dimension in np.shape(image_data))
        if channel_axis != len(shape) - 1:
            raise ValueError(
                "Napari RGB payload requires its declared source channel axis "
                f"to be last; got axis {channel_axis} for shape {shape!r}."
            )
        if shape[channel_axis] not in (3, 4):
            raise ValueError(
                "Napari RGB payload requires three or four values on its declared "
                f"source channel axis; got shape {shape!r}."
            )
        return True


@dataclass(frozen=True, slots=True)
class NapariLayerSelectionSnapshot:
    """Selection state before an automatic layer replacement."""

    active_layer: NapariLayerHandle | None
    replacing_active_layer: bool


class NapariLayerSelectionAuthority:
    """Preserve user layer selection across automatic streaming updates."""

    @classmethod
    def capture(
        cls,
        viewer: NapariViewerLayerCreator,
        replaced_layer: NapariLayerHandle | None,
    ) -> NapariLayerSelectionSnapshot:
        active_layer = cls._active_layer(viewer)
        return NapariLayerSelectionSnapshot(
            active_layer=active_layer,
            replacing_active_layer=active_layer is not None
            and active_layer is replaced_layer,
        )

    @classmethod
    def restore(
        cls,
        viewer: NapariViewerLayerCreator,
        snapshot: NapariLayerSelectionSnapshot,
        replacement_layer: NapariLayerHandle,
    ) -> None:
        if snapshot.active_layer is None:
            return
        if snapshot.replacing_active_layer:
            cls._set_active_layer(viewer, replacement_layer)
            return
        if snapshot.active_layer in viewer.layers:
            cls._set_active_layer(viewer, snapshot.active_layer)

    @staticmethod
    def _active_layer(
        viewer: NapariViewerLayerCreator,
    ) -> NapariLayerHandle | None:
        return viewer.layers.selection.active

    @staticmethod
    def _set_active_layer(
        viewer: NapariViewerLayerCreator,
        layer: NapariLayerHandle,
    ) -> None:
        viewer.layers.selection.active = layer


class NapariLayerUpdateAuthority:
    """Owns create-or-replace mechanics for Napari streaming layers."""

    def create_or_update(
        self,
        *,
        layer_kind: NapariLayerKind,
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        route_key: str,
        layer_name: str,
        data: LayerData,
        layer_kwargs: Mapping[str, LayerKwargValue],
    ) -> NapariLayerHandle:
        existing_layer = self._existing_layer(
            viewer=viewer,
            layers=layers,
            route_key=route_key,
        )
        selection = NapariLayerSelectionAuthority.capture(
            viewer,
            existing_layer,
        )
        if existing_layer is not None:
            viewer.layers.remove(existing_layer)
            layers.pop(route_key, None)
            logger.info(
                "🔬 NAPARI PROCESS: Removed existing %s layer %s for route %s",
                layer_kind.value,
                layer_name,
                route_key,
            )

        new_layer = NAPARI_LAYER_CREATORS[layer_kind](
            viewer,
            data,
            layer_name,
            layer_kwargs,
        )
        layers[route_key] = new_layer
        NapariLayerSelectionAuthority.restore(
            viewer,
            selection,
            new_layer,
        )
        NAPARI_LAYER_CREATED_LOGGERS[layer_kind](layer_kind, layer_name, data)
        return new_layer

    @staticmethod
    def _existing_layer(
        *,
        viewer: NapariViewerLayerCreator,
        layers: dict[str, NapariLayerHandle],
        route_key: str,
    ) -> NapariLayerHandle | None:
        if route_key not in layers:
            return None
        layer = layers[route_key]
        if layer in viewer.layers:
            return layer
        return None


@dataclass(slots=True)
class NapariDimensionLayerState:
    """Semantic dimension-label state for one streamed Napari layer."""

    labels: DimensionLabelMap
    scalar_labels: tuple[str, ...] = ()
    presentation: "NapariAxisPresentation | None" = None

    @classmethod
    def empty(cls) -> "NapariDimensionLayerState":
        return cls(labels={})

    @property
    def stack_axes(self) -> tuple[str, ...]:
        if self.presentation is None:
            return ()
        return self.presentation.projection.projected_axis_components

    @property
    def axis_labels(self) -> tuple[str, ...]:
        if self.presentation is None:
            return ()
        return self.presentation.axis_labels

    @property
    def axis_offsets(self) -> tuple[int, ...]:
        if self.presentation is None:
            return ()
        return self.presentation.projection.axis_offsets

    def label_parts_for_current_step(
        self,
        current_step: tuple[int, ...],
        viewer_axis_origins: tuple[int, ...] = (),
    ) -> tuple[str, ...] | None:
        """Return overlay labels for the current viewer position, or None if out of domain."""
        if self.presentation is None:
            return None

        label_parts = tuple(
            label
            for label in self.scalar_labels
            if label and str(label).lower() != "none"
        )
        stack_label_parts = []
        for axis_index, component in enumerate(self.stack_axes):
            viewer_axis_index = self.presentation.viewer_axis_index(axis_index)
            if viewer_axis_index >= len(current_step) or component not in self.labels:
                return None
            labels = self.labels[component]
            label_index = self.presentation.label_index(
                current_step[viewer_axis_index],
                axis_index,
                viewer_axis_origin=(
                    viewer_axis_origins[viewer_axis_index]
                    if viewer_axis_index < len(viewer_axis_origins)
                    else 0
                ),
            )
            if label_index < 0 or label_index >= len(labels):
                return None
            label = labels[label_index]
            if label and str(label).lower() != "none":
                stack_label_parts.append(label)
        return (*label_parts, *stack_label_parts)

    def describes_current_step(
        self,
        current_step: tuple[int, ...],
        viewer_axis_origins: tuple[int, ...] = (),
    ) -> bool:
        """Return whether this route has labels for the current viewer position."""
        return (
            self.label_parts_for_current_step(
                current_step,
                viewer_axis_origins,
            )
            is not None
        )


@dataclass(frozen=True, slots=True)
class NapariAxisPresentation(ViewerComponentAxisSemantics):
    """Layer-local axis semantics projected into the viewer coordinate space."""

    route_key: str
    projection: ViewerLayerAxisProjection
    payload_axis_labels: tuple[str, ...] = ()
    aggregate_axis_bindings: NapariAggregateAxisBindingSet = field(
        default_factory=NapariAggregateAxisBindingSet
    )

    def __post_init__(self) -> None:
        if len(set(self.display_axis_components)) != len(self.display_axis_components):
            raise ValueError(
                "Napari display-axis components must be unique; got "
                f"{self.display_axis_components!r}."
            )
        projected_indices = self.projected_display_axis_indices
        if projected_indices != tuple(sorted(projected_indices)):
            raise ValueError(
                "Napari route axes must preserve the declaration-owned display-axis "
                f"order; got route {self.projection.projected_axis_components!r} "
                f"within {self.display_axis_components!r}."
            )

    @property
    def display_axis_components(self) -> tuple[str, ...]:
        """Return the declaration-owned shared stack slots for this presentation."""

        return self.layout.components_for_mode(ViewerComponentMode.STACK)

    @property
    def projected_display_axis_indices(self) -> tuple[int, ...]:
        """Return the display slot owned by each route-local projected axis."""

        missing = tuple(
            component
            for component in self.projection.projected_axis_components
            if component not in self.display_axis_components
        )
        if missing:
            raise ValueError(
                "Napari projected axes are absent from the declared display axes: "
                f"{missing!r}."
            )
        return tuple(
            self.display_axis_components.index(component)
            for component in self.projection.projected_axis_components
        )

    @property
    def axis_labels(self) -> tuple[str, ...]:
        return tuple(
            [
                *self.display_axis_components,
                *self.payload_axis_labels,
                "y",
                "x",
            ]
        )

    def viewer_axis_index(self, projected_axis_index: int) -> int:
        """Return the shared-viewer slot for one route-local axis index."""

        try:
            return self.projected_display_axis_indices[projected_axis_index]
        except IndexError as error:
            raise IndexError(
                "Napari projected axis index is outside the route projection: "
                f"{projected_axis_index}."
            ) from error

    def axis_offset(self, display_axis_index: int) -> int:
        """Return a route offset in shared display-axis coordinates."""

        try:
            projected_axis_index = self.projected_display_axis_indices.index(
                display_axis_index
            )
        except ValueError:
            return 0
        return self.projection.axis_offset(projected_axis_index)

    def align_array(self, data: np.ndarray) -> np.ndarray:
        """Insert singleton slots for semantic display axes absent from this route."""

        projected_rank = len(self.projection.projected_axis_components)
        if data.ndim < projected_rank:
            raise ValueError(
                "Napari layer data rank is smaller than its projected component rank: "
                f"shape={data.shape!r}, axes="
                f"{self.projection.projected_axis_components!r}."
            )
        aligned_stack_shape = [1] * len(self.display_axis_components)
        for projected_index, display_index in enumerate(
            self.projected_display_axis_indices
        ):
            aligned_stack_shape[display_index] = data.shape[projected_index]
        return data.reshape((*aligned_stack_shape, *data.shape[projected_rank:]))

    def align_coordinates(self, coordinates: np.ndarray) -> np.ndarray:
        """Insert zero-valued columns for absent semantic display axes."""

        projected_rank = len(self.projection.projected_axis_components)
        if coordinates.ndim != 2 or coordinates.shape[1] < projected_rank:
            raise ValueError(
                "Napari coordinates must be a 2-D array containing every projected "
                f"axis column; got shape {coordinates.shape!r}."
            )
        payload_coordinates = coordinates[:, projected_rank:]
        aligned_stack_coordinates = np.zeros(
            (len(coordinates), len(self.display_axis_components)),
            dtype=coordinates.dtype,
        )
        for projected_index, display_index in enumerate(
            self.projected_display_axis_indices
        ):
            aligned_stack_coordinates[:, display_index] = coordinates[
                :, projected_index
            ]
        return np.concatenate(
            (aligned_stack_coordinates, payload_coordinates),
            axis=1,
        )

    def translate(
        self,
        payload_axis_labels: tuple[str, ...] = (),
    ) -> tuple[float, ...]:
        """Return translation in the same declaration-owned slots as layer data."""

        return (
            *(
                float(self.axis_offset(index))
                for index in range(len(self.display_axis_components))
            ),
            *(0.0 for _ in payload_axis_labels),
            0.0,
            0.0,
        )

    def label_index(
        self,
        viewer_step: int,
        axis_index: int,
        *,
        viewer_axis_origin: int = 0,
    ) -> int:
        """Project one shared-viewer coordinate into this route's local axis."""
        return (
            viewer_step + viewer_axis_origin - self.projection.axis_offset(axis_index)
        )

    def viewer_step(
        self,
        label_index: int,
        axis_index: int,
        *,
        viewer_axis_origin: int = 0,
    ) -> int:
        """Project one route-local label index into the shared viewer axis."""
        return (
            label_index + self.projection.axis_offset(axis_index) - viewer_axis_origin
        )


@dataclass(slots=True)
class NapariLayerRouteStateStore:
    """Own per-layer Napari runtime state that must stay keyed together."""

    layers: dict[str, NapariLayerHandle]
    layer_titles: dict[str, str]
    layer_dimension_states: dict[str, NapariDimensionLayerState]
    layer_pending_updates: dict[str, NapariPendingLayerUpdate]
    layer_update_errors: dict[str, str]
    active_dimension_label_route: str | None
    layer_settlement: NapariLayerSettlementState | None
    _settlement_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    @classmethod
    def empty(cls) -> "NapariLayerRouteStateStore":
        return cls(
            layers={},
            layer_titles={},
            layer_dimension_states={},
            layer_pending_updates={},
            layer_update_errors={},
            active_dimension_label_route=None,
            layer_settlement=None,
        )

    def set_title(self, layer_key: str, title: str) -> None:
        self.layer_titles[layer_key] = title

    def title_for(self, layer_key: str) -> str:
        return self.layer_titles[layer_key]

    def title_collides(self, layer_key: str, title: str) -> bool:
        return any(
            other_key != layer_key and other_title == title
            for other_key, other_title in self.layer_titles.items()
        )

    def purge_route(self, layer_key: str) -> None:
        self.layers.pop(layer_key, None)
        self.layer_titles.pop(layer_key, None)
        self.layer_dimension_states.pop(layer_key, None)
        self.layer_pending_updates.pop(layer_key, None)
        with self._settlement_lock:
            self.layer_update_errors.pop(layer_key, None)
        if self.active_dimension_label_route == layer_key:
            self.active_dimension_label_route = None

    def dimension_state_for(self, layer_key: str) -> NapariDimensionLayerState:
        if layer_key not in self.layer_dimension_states:
            return NapariDimensionLayerState.empty()
        return self.layer_dimension_states[layer_key]

    def set_dimension_state(
        self,
        layer_key: str,
        state: NapariDimensionLayerState,
    ) -> None:
        self.layer_dimension_states[layer_key] = state

    def axis_origins_for(self, axis_labels: tuple[str, ...]) -> tuple[int, ...]:
        """Return normalized-viewer origins derived from mounted route offsets."""
        compatible_states = tuple(
            state
            for state in self.layer_dimension_states.values()
            if state.axis_labels == axis_labels
        )
        return tuple(
            min(
                (
                    state.presentation.axis_offset(axis_index)
                    for state in compatible_states
                    if state.presentation is not None
                ),
                default=0,
            )
            for axis_index in range(len(axis_labels))
        )

    def set_active_dimension_label_route(self, layer_key: str) -> None:
        self.active_dimension_label_route = layer_key

    def cancel_pending_update(self, layer_key: str) -> bool:
        if layer_key not in self.layer_pending_updates:
            return False
        update = self.layer_pending_updates[layer_key]
        update.stop_timer()
        return True

    def set_pending_update(
        self,
        layer_key: str,
        update: NapariPendingLayerUpdate,
    ) -> None:
        with self._settlement_lock:
            if self.layer_settlement is not None:
                if self.layer_settlement.phase is ViewerSettlePhase.RUNNING:
                    raise RuntimeError(
                        "Cannot queue a Napari layer update while settlement is active."
                    )
                self.layer_settlement = None
        self.layer_pending_updates[layer_key] = update

    def pop_pending_update(self, layer_key: str) -> NapariPendingLayerUpdate | None:
        if layer_key not in self.layer_pending_updates:
            return None
        return self.layer_pending_updates.pop(layer_key)

    def pending_update_for(
        self,
        layer_key: str,
    ) -> NapariPendingLayerUpdate | None:
        """Return the exact queued update generation for one route."""

        return self.layer_pending_updates.get(layer_key)

    def drain_pending_updates(self) -> tuple[tuple[str, NapariPendingLayerUpdate], ...]:
        updates = tuple(self.layer_pending_updates.items())
        self.layer_pending_updates.clear()
        for _, update in updates:
            update.stop_timer()
        return updates

    def begin_settlement(self) -> NapariLayerSettlementState:
        """Return the active settlement or begin one from queued updates."""

        with self._settlement_lock:
            if self.layer_settlement is None:
                self.layer_settlement = NapariLayerSettlementState(
                    self.drain_pending_updates()
                )
            return self.layer_settlement

    def existing_settlement_progress(self) -> ViewerSettleProgress | None:
        """Return a thread-safe snapshot without starting Qt display work."""

        with self._settlement_lock:
            settlement = self.layer_settlement
        if settlement is None:
            return None
        return settlement.progress()

    def reset_settlement(self) -> None:
        """Discard terminal settlement state before a new stream cycle."""

        with self._settlement_lock:
            if (
                self.layer_settlement is not None
                and self.layer_settlement.phase is ViewerSettlePhase.RUNNING
            ):
                raise RuntimeError("Cannot reset an active Napari layer settlement.")
            self.layer_settlement = None

    def record_update_error(self, layer_key: str, error: Exception) -> None:
        with self._settlement_lock:
            self.layer_update_errors[layer_key] = str(error)

    def clear_update_error(self, layer_key: str) -> None:
        with self._settlement_lock:
            self.layer_update_errors.pop(layer_key, None)

    def clear_update_errors(self) -> None:
        with self._settlement_lock:
            self.layer_update_errors.clear()

    def require_updates_succeeded(self) -> None:
        failure_message = self.update_failure_message()
        if failure_message is None:
            return

        raise RuntimeError(f"Napari layer updates failed: {failure_message}")

    def update_failure_message(self) -> str | None:
        """Return the exact recorded route failures, if any."""

        with self._settlement_lock:
            if not self.layer_update_errors:
                return None
            return "; ".join(
                f"{layer_key}: {message}"
                for layer_key, message in self.layer_update_errors.items()
            )

    def has_layer(self, layer_key: str) -> bool:
        return layer_key in self.layers

    def layer(self, layer_key: str) -> NapariLayerHandle:
        return self.layers[layer_key]

    def set_layer(self, layer_key: str, layer: NapariLayerHandle) -> None:
        self.layers[layer_key] = layer

    def route_for_layer(self, layer: NapariLayerHandle) -> str | None:
        """Return the authoritative route owning ``layer``, if it is streamed."""

        return next(
            (
                layer_key
                for layer_key, candidate in self.layers.items()
                if candidate is layer
            ),
            None,
        )


@dataclass(slots=True)
class NapariComponentGroupStore:
    """Own accumulated stream items by Napari layer route."""

    groups: dict[str, list["NapariStreamLayerItem"]] = field(default_factory=dict)

    def items_for(self, layer_key: str) -> list["NapariStreamLayerItem"]:
        if layer_key not in self.groups:
            self.groups[layer_key] = []
        return self.groups[layer_key]

    def existing_items_for(
        self,
        layer_key: str,
    ) -> list["NapariStreamLayerItem"] | None:
        if layer_key not in self.groups:
            return None
        return self.groups[layer_key]

    def item_count(self, layer_key: str) -> int:
        items = self.existing_items_for(layer_key)
        if items is None:
            return 0
        return len(items)

    def purge(self, layer_key: str) -> None:
        self.groups.pop(layer_key, None)

    def clear(self) -> None:
        self.groups.clear()

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        return iter(self.groups)


@dataclass(slots=True)
class NapariLayerBatchDebouncePolicy:
    """Shared debounce policy for Napari layer updates and batch processors."""

    delay_ms: int = 1000

    def start_timer(self, timer: NapariTimerHandle) -> None:
        timer.start(self.delay_ms)

    def create_processor(
        self,
        *,
        napari_server: "NapariViewerServer",
    ) -> "NapariBatchProcessor":
        from polystore.streaming.receivers.napari import NapariBatchProcessor

        return NapariBatchProcessor(
            napari_server=napari_server,
        )


@dataclass(slots=True)
class NapariBatchProcessorStore:
    """Own lazy NapariBatchProcessor instances by layer key."""

    debounce_policy: NapariLayerBatchDebouncePolicy = field(
        default_factory=NapariLayerBatchDebouncePolicy
    )
    processors: dict[str, "NapariBatchProcessor"] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def get_or_create(
        self,
        *,
        layer_key: str,
        napari_server: "NapariViewerServer",
    ) -> "NapariBatchProcessor":
        with self.lock:
            if layer_key not in self.processors:
                self.processors[layer_key] = self.debounce_policy.create_processor(
                    napari_server=napari_server,
                )
                logger.info(
                    "NapariViewerServer: Created batch processor for layer '%s'",
                    layer_key,
                )
            return self.processors[layer_key]


class NapariShapeLabelAuthority:
    """Own validation and allocation of streamed ROI object identities."""

    MAX_LABEL: ClassVar[int] = int(np.iinfo(np.uint32).max)

    @staticmethod
    def declared_label(shape_dict: ShapePayloadMap) -> int | None:
        metadata = shape_dict.get(ViewerWireField.METADATA.value)
        if not isinstance(metadata, Mapping):
            return None
        label = metadata.get(VisualMetadataField.LABEL.value)
        if isinstance(label, np.integer):
            label = int(label)
        if isinstance(label, bool) or not isinstance(label, int):
            return None
        if label <= 0:
            raise ValueError(
                "Napari ROI shape metadata label must be a positive integer, "
                f"got {label!r}."
            )
        if label > NapariShapeLabelAuthority.MAX_LABEL:
            raise ValueError(
                "Napari ROI shape metadata label exceeds the uint32 ROI-label "
                f"domain: {label!r}."
            )
        return label

    @classmethod
    def declared_labels(
        cls,
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> set[int]:
        return {
            label_id
            for item in layer_items
            for shape_dict in item.data
            if (label_id := cls.declared_label(shape_dict)) is not None
        }


@dataclass(slots=True)
class NapariShapeLabelAllocator:
    """Allocate stable fallback identities without colliding with declared labels."""

    reserved_labels: set[int]
    next_fallback_label: int = 1

    @classmethod
    def for_items(
        cls,
        layer_items: Sequence[NapariStreamLayerItem],
    ) -> "NapariShapeLabelAllocator":
        return cls(NapariShapeLabelAuthority.declared_labels(layer_items))

    def label_for(self, shape_dict: ShapePayloadMap) -> int:
        declared_label = NapariShapeLabelAuthority.declared_label(shape_dict)
        if declared_label is not None:
            return declared_label
        while self.next_fallback_label in self.reserved_labels:
            self.next_fallback_label += 1
        label = self.next_fallback_label
        self.next_fallback_label += 1
        return label


@dataclass(slots=True)
class NapariShapeFeatureColumns:
    """Accumulate native feature columns without materializing row mirrors."""

    values: dict[str, list[object]] = field(default_factory=dict)
    row_count: int = 0

    def append(
        self,
        metadata: Mapping[str, ShapePayloadValue],
        *,
        label: int,
        path: str,
    ) -> None:
        """Append one metadata row while preserving first-seen column order."""

        for column_values in self.values.values():
            column_values.append(None)
        for name, value in metadata.items():
            if name in (
                ObjectArtifactSubjectBinding.SUBJECT_FEATURE,
                ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE,
            ):
                continue
            self._set_last(str(name), NapariShapeLayerPayload._feature_value(value))
        self._set_last(VisualMetadataField.LABEL.value, label)
        self._set_last(ViewerWireField.PATH.value, path)
        self.row_count += 1

    def _set_last(self, name: str, value: object) -> None:
        if name not in self.values:
            self.values[name] = [None] * self.row_count + [value]
            return
        self.values[name][-1] = value


@dataclass(frozen=True, slots=True)
class NapariShapeColorProjection:
    """One authoritative label-to-color projection for a Shapes payload."""

    cycle: list[tuple[float, float, float, float]]
    member_colors: list[tuple[float, float, float, float]]

    @classmethod
    def from_labels(
        cls,
        labels: Sequence[object],
    ) -> "NapariShapeColorProjection":
        """Build the cycle and member projection in one label-domain pass."""

        label_order = tuple(dict.fromkeys(labels))
        golden_ratio_conjugate = 0.618033988749895
        cycle = [
            (
                *colorsys.hsv_to_rgb(
                    (int(label) * golden_ratio_conjugate) % 1.0,
                    0.78,
                    1.0,
                ),
                1.0,
            )
            for label in label_order
        ]
        colors_by_label = dict(zip(label_order, cycle))
        return cls(
            cycle=cycle,
            member_colors=[colors_by_label[label] for label in labels],
        )


@dataclass(frozen=True, slots=True)
class NapariShapeLayerPayload:
    """Native N-D Napari Shapes data assembled from streamed ROI payloads."""

    data: list[np.ndarray]
    shape_types: list[ShapeType]
    features: dict[str, list[object]]
    ndim: int
    result_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def color_projection(self) -> NapariShapeColorProjection:
        """Project the payload's authoritative labels to one color domain."""

        labels = self.features.get(VisualMetadataField.LABEL.value, ())
        return NapariShapeColorProjection.from_labels(labels)

    def chunks(
        self,
        *,
        max_shape_count: int,
        max_vertex_count: int,
    ) -> tuple["NapariShapeLayerPayload", ...]:
        """Partition members in order into bounded native Shapes work units."""

        if max_shape_count <= 0 or max_vertex_count <= 0:
            raise ValueError("Napari Shapes chunk limits must be positive.")
        if not self.data:
            return (self,)

        bounds: list[tuple[int, int]] = []
        start = 0
        vertex_count = 0
        for index, coordinates in enumerate(self.data):
            member_vertex_count = max(1, len(coordinates))
            shape_limit_reached = index - start >= max_shape_count
            vertex_limit_reached = (
                index > start and vertex_count + member_vertex_count > max_vertex_count
            )
            if shape_limit_reached or vertex_limit_reached:
                bounds.append((start, index))
                start = index
                vertex_count = 0
            vertex_count += member_vertex_count
        bounds.append((start, len(self.data)))
        return tuple(self._slice(start, stop) for start, stop in bounds)

    def _slice(self, start: int, stop: int) -> "NapariShapeLayerPayload":
        result_metadata = dict(self.result_metadata)
        subject_ids = result_metadata.get(
            ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE
        )
        if isinstance(subject_ids, tuple):
            result_metadata[ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE] = (
                subject_ids[start:stop]
            )
        return NapariShapeLayerPayload(
            data=self.data[start:stop],
            shape_types=self.shape_types[start:stop],
            features={
                name: values[start:stop] for name, values in self.features.items()
            },
            ndim=self.ndim,
            result_metadata=result_metadata,
        )

    def align_display_axes(
        self,
        presentation: "NapariAxisPresentation",
    ) -> "NapariShapeLayerPayload":
        """Project every ROI into the viewer's declaration-owned axis slots."""

        return NapariShapeLayerPayload(
            data=[
                presentation.align_coordinates(coordinates) for coordinates in self.data
            ],
            shape_types=self.shape_types,
            features=self.features,
            ndim=len(presentation.display_axis_components) + 2,
            result_metadata=self.result_metadata,
        )

    @classmethod
    def build(
        cls,
        *,
        layer_items: Sequence[NapariStreamLayerItem],
        axis_projection: ViewerLayerAxisProjection,
        aggregate_axis_bindings: NapariAggregateAxisBindingSet | None = None,
    ) -> "NapariShapeLayerPayload":
        if aggregate_axis_bindings is None:
            aggregate_axis_bindings = NapariAggregateAxisBindingSet()

        shape_data: list[np.ndarray] = []
        shape_types: list[ShapeType] = []
        feature_columns = NapariShapeFeatureColumns()
        object_subject_tokens: list[object] = []
        object_subject_ids: list[object] = []
        subject_metadata_member_count = 0
        label_allocator = NapariShapeLabelAllocator.for_items(layer_items)

        for item in layer_items:
            if not isinstance(item.data, Sequence) or isinstance(
                item.data,
                (str, bytes),
            ):
                raise TypeError(
                    "Napari SHAPES payload data must be a sequence of shape mappings."
                )
            for shape_dict in item.data:
                if not isinstance(shape_dict, Mapping):
                    raise TypeError(
                        "Napari SHAPES payload entries must be shape mappings."
                    )
                shape_type_value = shape_dict.get(ViewerWireField.TYPE.value)
                if not isinstance(shape_type_value, str) or not shape_type_value:
                    raise TypeError(
                        "Napari SHAPES payload type must be a non-empty string."
                    )
                shape_type = ShapeType(shape_type_value)
                coordinates = np.asarray(shape_dict["coordinates"], dtype=float)
                if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                    raise ValueError(
                        "Napari streamed ROI coordinates must be an Nx2 YX array, "
                        f"got shape {coordinates.shape!r}."
                    )
                components = aggregate_axis_bindings.shape_component_values(
                    item,
                    shape_dict,
                )
                indices = axis_projection.coordinate_index(
                    components,
                    context="Napari shape item",
                )
                if indices:
                    prefix = np.broadcast_to(
                        np.asarray(indices, dtype=float),
                        (len(coordinates), len(indices)),
                    )
                    coordinates = np.concatenate((prefix, coordinates), axis=1)

                metadata = shape_dict.get(ViewerWireField.METADATA.value, {})
                if not isinstance(metadata, Mapping):
                    raise TypeError("Napari SHAPES payload metadata must be a mapping.")
                subject_token = metadata.get(
                    ObjectArtifactSubjectBinding.SUBJECT_FEATURE
                )
                subject_id = metadata.get(
                    ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE
                )
                if subject_token is not None or subject_id is not None:
                    if subject_token is None or subject_id is None:
                        raise ValueError(
                            "OpenHCS ROI subject metadata requires both subject and ID."
                        )
                    subject_metadata_member_count += 1
                    object_subject_tokens.append(cls._feature_value(subject_token))
                    object_subject_ids.append(cls._feature_value(subject_id))
                feature_columns.append(
                    metadata,
                    label=label_allocator.label_for(shape_dict),
                    path=item.address.path,
                )

                shape_data.append(coordinates)
                shape_types.append(shape_type)
        result_metadata: dict[str, object] = {}
        if subject_metadata_member_count:
            if subject_metadata_member_count != len(shape_data):
                raise ValueError(
                    "One Napari result layer cannot mix subject-bound and unbound ROIs."
                )
            subject_tokens = tuple(dict.fromkeys(object_subject_tokens))
            if len(subject_tokens) != 1:
                raise ValueError(
                    "One Napari result layer cannot mix multiple object subjects."
                )
            result_metadata = {
                ObjectArtifactSubjectBinding.SUBJECT_FEATURE: subject_tokens[0],
                ObjectArtifactSubjectBinding.SUBJECT_ID_FEATURE: tuple(
                    object_subject_ids
                ),
            }
        return cls(
            data=shape_data,
            shape_types=shape_types,
            features=feature_columns.values,
            ndim=len(axis_projection.projected_axis_components) + 2,
            result_metadata=result_metadata,
        )

    @staticmethod
    def _feature_value(value: object) -> object:
        """Keep Napari feature columns scalar while retaining metadata detail."""

        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)
