"""Typed runtime artifact values and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import InitVar, dataclass, replace
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Self, TypeVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactPayloadShape,
)
from openhcs.core.image_shapes import (
    is_color_image_slice,
    is_color_image_stack,
)
from openhcs.core.image_stack_layout import ImageStackLayout
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementScope,
    MeasurementSubject,
    ObjectLabelDomain,
    ObjectLabelDomainDeclaration,
    ObjectLabelDomainMetadata,
    ObjectLabelDomainScope,
    ObjectLabelIdDomainStrategy,
    ObjectLabelPlaneDomainStrategy,
    ObjectLabelRepresentation,
    ObjectLabelVariant,
    PreserveSourceObjectLabelDomainDeclaration,
    RelationshipEndpoint,
    RelationshipSemantics,
    RuntimePlaneAxis,
    SpatialGridOrigin,
    SpatialGridOrdering,
    aligned_dense_object_label_arrays,
    coerce_enum,
    measurement_table_row_layout,
    measurement_table_row_layout_from_fields,
    normalize_measurement_table_rows,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    NominalTypeKeyedStrategyMixin,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet


PhysicalBorderEdgesYX = tuple[bool, bool, bool, bool] | None

_TPayload = TypeVar("_TPayload", bound=type[Any])
_ARRAY_PAYLOAD_PREDICATES: list[Callable[[Any], bool]] = []
logger = logging.getLogger(__name__)


class RuntimeArrayPayload(ABC):
    """Nominal ABC for array payload types accepted by runtime artifacts."""

    __array_priority__ = 1000

    @property
    @abstractmethod
    def shape(self) -> Any:
        ...

    @abstractmethod
    def array_payload_data(self) -> Any:
        ...

    @abstractmethod
    def with_data(self, data: Any) -> Self:
        ...

    def array_operand(self) -> Any:
        return self.array_payload_data()

    def array_ufunc_result(self, result: Any) -> Any:
        if isinstance(result, tuple):
            return tuple(self.array_ufunc_result(item) for item in result)
        if isinstance(result, np.ndarray) and np.issubdtype(result.dtype, np.bool_):
            return result
        if isinstance(result, np.ndarray):
            return self.with_data(result)
        return result

    def compare_array_payload(self, other: Any, ufunc: Any) -> Any:
        return ufunc(np.asarray(self), runtime_array_operand(other))

    def max(self, *args: Any, **kwargs: Any) -> Any:
        """Return the maximum of the array payload for ndarray-like callers."""
        return np.asarray(self).max(*args, **kwargs)

    def __lt__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.less)

    def __le__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.less_equal)

    def __gt__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.greater)

    def __ge__(self, other: Any) -> Any:
        return self.compare_array_payload(other, np.greater_equal)

    def __array_ufunc__(
        self,
        ufunc: Any,
        method: str,
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        converted_inputs = tuple(runtime_array_operand(value) for value in inputs)
        if "out" in kwargs:
            kwargs = {
                **kwargs,
                "out": tuple(runtime_array_operand(value) for value in kwargs["out"]),
            }
        result = getattr(ufunc, method)(*converted_inputs, **kwargs)
        return self.array_ufunc_result(result)


@dataclass(frozen=True, slots=True)
class ImagePayloadMetadata:
    """Generic source-image metadata that should travel with runtime pixels."""

    intensity_scale: float | None = None
    source_dtype: str | None = None
    source_path: str | None = None
    unit_interval_intensity_scale: int | None = None
    channel_intensity_scales: tuple[float | None, ...] = ()
    channel_source_dtypes: tuple[str | None, ...] = ()
    channel_source_paths: tuple[str | None, ...] = ()
    channel_unit_interval_intensity_scales: tuple[int | None, ...] = ()
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None
    physical_border_edges_yx: PhysicalBorderEdgesYX = None
    mask_defines_border: bool | None = None

    @classmethod
    def for_array(
        cls,
        array: Any,
        *,
        source_path: str | None = None,
    ) -> "ImagePayloadMetadata":
        """Build metadata from an image array's source dtype."""
        import numpy as np

        dtype = np.asarray(array).dtype
        return cls(
            intensity_scale=image_intensity_scale_for_dtype(dtype),
            source_dtype=str(dtype),
            source_path=source_path,
        )

    @classmethod
    def for_array_payload(
        cls,
        array: Any,
        *,
        source_path: str | None = None,
    ) -> "ImagePayloadMetadata":
        """Build source metadata from an arraybridge-detectable payload."""
        from openhcs.core.memory import MEMORY_TYPE_NUMPY, detect_memory_type

        memory_type = detect_memory_type(array)
        dtype = getattr(array, "dtype", None)
        if dtype is None:
            raise TypeError(
                "ImagePayloadMetadata.for_array_payload requires an array payload "
                f"with dtype; got {type(array).__name__}."
            )
        return cls(
            intensity_scale=(
                image_intensity_scale_for_dtype(dtype)
                if memory_type == MEMORY_TYPE_NUMPY
                else None
            ),
            source_dtype=str(dtype),
            source_path=source_path,
        )

    @property
    def has_values(self) -> bool:
        """Return whether this metadata carries any semantic image facts."""
        return any(
            (
                self.intensity_scale is not None,
                self.source_dtype is not None,
                self.source_path is not None,
                self.unit_interval_intensity_scale is not None,
                bool(self.channel_intensity_scales),
                bool(self.channel_source_dtypes),
                bool(self.channel_source_paths),
                bool(self.channel_unit_interval_intensity_scales),
                self.spatial_origin_yx is not None,
                self.source_spatial_shape_yx is not None,
                self.physical_border_edges_yx is not None,
                self.mask_defines_border is not None,
            )
        )

    def intensity_scale_for_channel(self, channel_index: int) -> float | None:
        """Return the best available intensity scale for one channel/plane."""
        if 0 <= channel_index < len(self.channel_intensity_scales):
            channel_scale = self.channel_intensity_scales[channel_index]
            if channel_scale is not None:
                return channel_scale
        return self.intensity_scale

    def unit_interval_intensity_scale_for_channel(
        self,
        channel_index: int,
    ) -> int | None:
        """Return the scale proving current pixels are exact integer/scale values."""
        if 0 <= channel_index < len(self.channel_unit_interval_intensity_scales):
            channel_scale = self.channel_unit_interval_intensity_scales[channel_index]
            if channel_scale is not None:
                return int(channel_scale)
        return self.unit_interval_intensity_scale

    def for_channel(self, channel_index: int) -> "ImagePayloadMetadata":
        """Return single-channel metadata sliced from a stacked payload."""
        return ImagePayloadMetadata(
            intensity_scale=self.intensity_scale_for_channel(channel_index),
            source_dtype=_tuple_value(self.channel_source_dtypes, channel_index)
            or self.source_dtype,
            source_path=_tuple_value(self.channel_source_paths, channel_index)
            or self.source_path,
            unit_interval_intensity_scale=(
                self.unit_interval_intensity_scale_for_channel(channel_index)
            ),
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
            physical_border_edges_yx=self.physical_border_edges_yx,
            mask_defines_border=self.mask_defines_border,
        )

    def with_unit_interval_intensity_scale(
        self,
        scale: int | None,
    ) -> "ImagePayloadMetadata":
        """Return metadata with the current unit-interval pixel proof updated."""
        return replace(self, unit_interval_intensity_scale=scale)

    def without_unit_interval_intensity_scale(self) -> "ImagePayloadMetadata":
        """Return metadata after an arithmetic transform changed pixel values."""
        return replace(
            self,
            unit_interval_intensity_scale=None,
            channel_unit_interval_intensity_scales=(),
        )

    def without_spatial_domain(self) -> "ImagePayloadMetadata":
        """Return metadata with invalidated source-spatial placement removed."""
        return replace(
            self,
            spatial_origin_yx=None,
            source_spatial_shape_yx=None,
            physical_border_edges_yx=None,
            mask_defines_border=None,
        )

    def physical_border_edges_for_shape(
        self,
        image_shape_yx: Sequence[int],
    ) -> tuple[bool, bool, bool, bool]:
        """Return which local image edges are true source-image edges.

        Edge tuple order is ``(top, bottom, left, right)``. Missing spatial
        metadata means the current image is treated as the full physical image,
        preserving historical behavior for plain arrays.
        """
        if self.physical_border_edges_yx is not None:
            return tuple(bool(edge) for edge in self.physical_border_edges_yx)
        if self.spatial_origin_yx is None or self.source_spatial_shape_yx is None:
            return True, True, True, True

        height, width = _spatial_shape_pair(image_shape_yx, "image_shape_yx")
        origin_y, origin_x = self.spatial_origin_yx
        source_height, source_width = self.source_spatial_shape_yx
        return (
            origin_y <= 0,
            origin_y + height >= source_height,
            origin_x <= 0,
            origin_x + width >= source_width,
        )

    def with_spatial_crop(
        self,
        *,
        input_shape_yx: Sequence[int],
        output_shape_yx: Sequence[int],
        offset_yx: tuple[int, int],
        physical_border_edges_yx: PhysicalBorderEdgesYX = None,
    ) -> "ImagePayloadMetadata":
        """Return metadata for a crop of this image payload."""
        input_shape = _spatial_shape_pair(input_shape_yx, "input_shape_yx")
        output_shape = _spatial_shape_pair(output_shape_yx, "output_shape_yx")
        parent_origin = self.spatial_origin_yx or (0, 0)
        source_shape = self.source_spatial_shape_yx or input_shape
        origin = (
            int(parent_origin[0]) + int(offset_yx[0]),
            int(parent_origin[1]) + int(offset_yx[1]),
        )
        if physical_border_edges_yx is None:
            physical_border_edges_yx = (
                origin[0] <= 0,
                origin[0] + output_shape[0] >= source_shape[0],
                origin[1] <= 0,
                origin[1] + output_shape[1] >= source_shape[1],
            )
        return replace(
            self,
            spatial_origin_yx=origin,
            source_spatial_shape_yx=source_shape,
            physical_border_edges_yx=tuple(
                bool(edge) for edge in physical_border_edges_yx
            ),
        )


@dataclass(frozen=True, slots=True)
class ImageMetadataPayload(RuntimeArrayPayload):
    """Image data plus metadata, without requiring a validity mask."""

    data: Any
    metadata: ImagePayloadMetadata

    def __post_init__(self) -> None:
        if np.ndim(self.data) == 0:
            raise TypeError(
                "ImageMetadataPayload.data requires array-like image data, "
                f"got {type(self.data).__name__}."
            )
        if not self.metadata.has_values:
            raise ValueError("ImageMetadataPayload.metadata cannot be empty.")

    @property
    def shape(self) -> Any:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.data, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.data

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.data, name)

    def with_data(
        self,
        data: Any,
        *,
        metadata: ImagePayloadMetadata | None = None,
    ) -> "ImageMetadataPayload":
        """Return the same metadata attached to replacement data."""
        return type(self)(
            data=data,
            metadata=self.metadata if metadata is None else metadata,
        )


@dataclass(frozen=True, slots=True)
class MaskedImagePayload(RuntimeArrayPayload):
    """Image data plus an authoritative per-pixel validity mask."""

    data: Any
    mask: Any
    metadata: ImagePayloadMetadata = ImagePayloadMetadata()

    def __post_init__(self) -> None:
        if np.ndim(self.data) == 0:
            raise TypeError(
                "MaskedImagePayload.data requires array-like image data, "
                f"got {type(self.data).__name__}."
            )
        mask_shape = tuple(np.shape(self.mask))
        if not mask_shape:
            raise TypeError(
                "MaskedImagePayload.mask requires array-like mask data, "
                f"got {type(self.mask).__name__}."
            )
        data_shape = tuple(np.shape(self.data))
        if not ImageMaskDomain(data_shape).accepts(mask_shape):
            raise ValueError(
                "MaskedImagePayload.mask shape must match the image spatial "
                f"domain; got mask {mask_shape!r} for image {data_shape!r}."
            )

    @property
    def shape(self) -> Any:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> Any:
        return self.data.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.data, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.data

    def __getitem__(self, key: Any) -> Any:
        return self.data[key]

    def __len__(self) -> int:
        return len(self.data)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.data, name)

    def with_data(self, data: Any, mask: Any | None = None) -> "MaskedImagePayload":
        """Return the same semantic image mask attached to replacement data."""
        return type(self)(
            data=data,
            mask=self.mask if mask is None else mask,
            metadata=self.metadata,
        )


def runtime_array_operand(value: Any) -> Any:
    """Return the ndarray operand for nominal runtime array payloads."""
    if isinstance(value, RuntimeArrayPayload):
        return value.array_operand()
    return value


def image_payload_data(payload: Any) -> Any:
    """Return concrete image pixels from a runtime image payload."""
    if isinstance(payload, (MaskedImagePayload, ImageMetadataPayload)):
        return payload.data
    return payload


def image_payload_mask(payload: Any) -> Any | None:
    """Return a runtime image mask when present."""
    if isinstance(payload, MaskedImagePayload):
        return payload.mask
    return None


def image_payload_metadata(payload: Any) -> ImagePayloadMetadata:
    """Return runtime image metadata when present."""
    if isinstance(payload, (MaskedImagePayload, ImageMetadataPayload)):
        return payload.metadata
    return ImagePayloadMetadata()


def image_payload_with_context(
    data: Any,
    *,
    mask: Any | None = None,
    metadata: ImagePayloadMetadata = ImagePayloadMetadata(),
) -> Any:
    """Attach generic image context to pixels only when context is present."""
    if mask is not None:
        return MaskedImagePayload(data=data, mask=mask, metadata=metadata)
    if metadata.has_values:
        return ImageMetadataPayload(data=data, metadata=metadata)
    return data


def project_image_mask_to_data_domain(mask: Any, data: Any) -> Any | None:
    """Project a source mask into the concrete image data domain when possible."""
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    data_shape = tuple(np.asarray(data).shape)
    mask_shape = tuple(mask_array.shape)
    mask_domain = ImageMaskDomain(data_shape)
    if mask_domain.accepts(mask_shape):
        return mask_array
    if (
        mask_array.ndim >= 3
        and mask_domain.accepts(tuple(mask_array.shape[1:]))
    ):
        return np.all(mask_array, axis=0)
    if (
        mask_array.ndim >= 3
        and mask_domain.accepts(tuple(mask_array.shape[:-1]))
    ):
        return np.all(mask_array, axis=-1)
    if (
        len(mask_shape) == len(data_shape)
        and mask_shape[0] == 1
        and data_shape[0] != 1
        and mask_shape[1:] == data_shape[1:]
    ):
        return np.broadcast_to(mask_array, data_shape)
    return None


@dataclass(frozen=True, slots=True)
class DerivedImagePayloadContext:
    """Project source image context onto a derived image payload."""

    source_payload: Any
    data: Any

    def payload(self) -> Any:
        same_spatial_domain = self.same_spatial_domain()
        metadata = image_payload_metadata(self.source_payload)
        if not same_spatial_domain:
            metadata = metadata.without_spatial_domain()
        return image_payload_with_context(
            self.data,
            mask=self.projected_mask() if same_spatial_domain else None,
            metadata=metadata,
        )

    def same_spatial_domain(self) -> bool:
        source_shape_yx = image_payload_spatial_shape_yx(self.source_payload)
        output_shape_yx = image_payload_spatial_shape_yx(self.data)
        return (
            source_shape_yx is not None
            and output_shape_yx is not None
            and source_shape_yx == output_shape_yx
        )

    def projected_mask(self) -> Any | None:
        return project_image_mask_to_data_domain(
            image_payload_mask(self.source_payload),
            self.data,
        )


def with_image_payload_data(
    payload: Any,
    data: Any,
    *,
    mask: Any | None = None,
    metadata: ImagePayloadMetadata | None = None,
) -> Any:
    """Preserve image-mask and metadata semantics while replacing pixels."""
    resolved_mask = project_image_mask_to_data_domain(
        image_payload_mask(payload) if mask is None else mask,
        data,
    )
    resolved_metadata = (
        image_payload_metadata(payload) if metadata is None else metadata
    )
    return image_payload_with_context(
        data,
        mask=resolved_mask,
        metadata=resolved_metadata,
    )


def with_derived_image_payload_data(
    payload: Any,
    data: Any,
) -> Any:
    """Attach source image context valid for a derived image output.

    Raw CellProfiler image outputs should remain nominal image payloads so the
    runtime can preserve dtype/intensity metadata. Spatial-domain metadata is
    only valid when the derived pixels still occupy the same XY extent as the
    source payload; shape-changing transforms need an explicit crop/resize
    adapter to carry a new spatial domain.
    """
    return DerivedImagePayloadContext(payload, data).payload()


def image_payload_slice_context(
    payload: Any,
    data: Any,
    channel_index: int,
) -> Any:
    """Attach one channel/slice of a payload's image context to slice data."""
    mask = image_payload_mask(payload)
    return image_payload_with_context(
        data,
        mask=None if mask is None else image_payload_mask_slice(mask, channel_index),
        metadata=image_payload_metadata(payload).for_channel(channel_index),
    )


def image_payload_mask_slice(mask: Any, channel_index: int) -> Any:
    """Return the mask plane matching one image channel/slice."""
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        return mask_array[channel_index]
    return mask


@dataclass(frozen=True, slots=True)
class ImagePayloadChannelProjection:
    """Project one channel while preserving payload metadata and mask semantics."""

    source_payload: Any
    source_data: Any
    channel_index: int
    channel_data: Any

    @classmethod
    def from_channel(
        cls,
        source_payload: Any,
        source_data: Any,
        channel_index: int,
    ) -> "ImagePayloadChannelProjection":
        return cls(
            source_payload=source_payload,
            source_data=source_data,
            channel_index=channel_index,
            channel_data=source_data[channel_index : channel_index + 1],
        )

    def payload(self) -> Any:
        return image_payload_with_context(
            self.channel_data,
            mask=self.projected_mask(),
            metadata=image_payload_metadata(self.source_payload).for_channel(
                self.channel_index,
            ),
        )

    def projected_mask(self) -> Any | None:
        mask = image_payload_mask(self.source_payload)
        if mask is None:
            return None
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape == np.asarray(self.source_data).shape:
            return mask_array[self.channel_index : self.channel_index + 1]
        return mask_array


def image_payload_spatial_shape_yx(payload: Any) -> tuple[int, int] | None:
    """Return the XY image shape for a nominal image payload."""
    import numpy as np

    array = np.asarray(image_payload_data(payload))
    if array.ndim < 2:
        return None
    if is_color_image_slice(array) or is_color_image_stack(array):
        return tuple(int(value) for value in array.shape[-3:-1])
    return tuple(int(value) for value in array.shape[-2:])


def compose_image_payload_metadata(
    image_payloads: Sequence[Any],
) -> ImagePayloadMetadata:
    """Compose per-image metadata for a stacked image bundle."""
    metadata_by_payload = tuple(image_payload_metadata(payload) for payload in image_payloads)
    if not any(metadata.has_values for metadata in metadata_by_payload):
        return ImagePayloadMetadata()
    return ImagePayloadMetadata(
        channel_intensity_scales=tuple(
            metadata.intensity_scale_for_channel(0)
            for metadata in metadata_by_payload
        ),
        channel_source_dtypes=tuple(
            metadata.source_dtype
            for metadata in metadata_by_payload
        ),
        channel_source_paths=tuple(
            metadata.source_path
            for metadata in metadata_by_payload
        ),
        channel_unit_interval_intensity_scales=tuple(
            metadata.unit_interval_intensity_scale_for_channel(0)
            for metadata in metadata_by_payload
        ),
        spatial_origin_yx=_common_metadata_value(
            metadata.spatial_origin_yx for metadata in metadata_by_payload
        ),
        source_spatial_shape_yx=_common_metadata_value(
            metadata.source_spatial_shape_yx for metadata in metadata_by_payload
        ),
        physical_border_edges_yx=_common_metadata_value(
            metadata.physical_border_edges_yx for metadata in metadata_by_payload
        ),
        mask_defines_border=_common_metadata_value(
            metadata.mask_defines_border for metadata in metadata_by_payload
        ),
    )


def image_intensity_scale_for_dtype(dtype: Any) -> float | None:
    """Return the conventional full-scale intensity for a pixel dtype."""
    import numpy as np

    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.bool_):
        return 1.0
    if np.issubdtype(dtype, np.integer):
        return float(np.iinfo(dtype).max)
    return None


def image_payload_intensity_scale(
    payload: Any,
    *,
    channel_index: int = 0,
) -> float | None:
    """Return the best semantic intensity scale for an image payload."""
    import numpy as np

    metadata_scale = image_payload_metadata(payload).intensity_scale_for_channel(
        channel_index
    )
    if metadata_scale is not None and metadata_scale > 0:
        return float(metadata_scale)
    return image_intensity_scale_for_dtype(np.asarray(image_payload_data(payload)).dtype)


def normalize_image_payload_intensity(
    payload: Any,
    *,
    dtype: Any = None,
    channel_index: int = 0,
) -> Any:
    """Normalize image pixels by payload metadata while preserving context."""
    import numpy as np

    array = np.asarray(image_payload_data(payload))
    target_dtype = np.float32 if dtype is None else np.dtype(dtype)
    if np.issubdtype(array.dtype, np.bool_):
        normalized = array.astype(target_dtype)
    elif np.issubdtype(array.dtype, np.integer):
        intensity_scale = image_payload_intensity_scale(
            payload,
            channel_index=channel_index,
        )
        if intensity_scale is None or intensity_scale <= 1:
            normalized = array.astype(target_dtype)
        else:
            normalized = array.astype(target_dtype) / float(intensity_scale)
            metadata = image_payload_metadata(payload).with_unit_interval_intensity_scale(
                int(intensity_scale)
            )
            return with_image_payload_data(payload, normalized, metadata=metadata)
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(target_dtype, copy=False)
    else:
        return payload
    return with_image_payload_data(payload, normalized)


def image_payload_metadata_from_source(
    image: Any,
    *,
    source_path: str,
    read_backend: str | None = None,
    filemanager: Any | None = None,
) -> ImagePayloadMetadata:
    """Return generic source metadata for an image loaded through OpenHCS I/O."""
    resolved_source_path = resolve_image_payload_source_path(
        source_path=source_path,
        read_backend=read_backend,
        filemanager=filemanager,
    )
    source_metadata = image_file_source_metadata(resolved_source_path)
    source_dtype = source_metadata.source_dtype
    resolved_path_text = (
        str(resolved_source_path) if resolved_source_path is not None else source_path
    )
    if source_dtype is None:
        return ImagePayloadMetadata.for_array(
            image_payload_data(image),
            source_path=resolved_path_text,
        )
    return ImagePayloadMetadata(
        intensity_scale=source_metadata.intensity_scale,
        source_dtype=str(source_dtype),
        source_path=resolved_path_text,
    )


def resolve_image_payload_source_path(
    *,
    source_path: str,
    read_backend: str | None = None,
    filemanager: Any | None = None,
) -> Path | None:
    """Resolve a backend-specific image path to a physical file when possible."""
    if read_backend is not None and filemanager is not None:
        backend = getattr(filemanager, "registry", {}).get(read_backend)
        resolver = getattr(backend, "resolve_path", None) or getattr(
            backend,
            "_resolve_path",
            None,
        )
        if callable(resolver):
            try:
                return Path(resolver(source_path))
            except Exception:
                logger.debug(
                    "Could not resolve image source path %s via backend %s.",
                    source_path,
                    read_backend,
                    exc_info=True,
                )
    path = Path(source_path)
    return path if path.exists() else None


def image_file_source_dtype(path: Path | None) -> Any | None:
    """Return an image file's stored dtype without loading pixel data."""
    return image_file_source_metadata(path).source_dtype


@dataclass(frozen=True, slots=True)
class ImageFileSourceMetadata:
    """Image-file metadata relevant to runtime image semantics."""

    source_dtype: Any | None = None
    intensity_scale: float | None = None


def image_file_source_metadata(path: Path | None) -> ImageFileSourceMetadata:
    """Return image file metadata without loading pixel data when possible."""
    if path is None or not path.exists():
        return ImageFileSourceMetadata()
    try:
        import imageio.v3 as iio

        dtype = iio.improps(path).dtype
    except Exception:
        logger.debug("Could not read image dtype metadata for %s.", path, exc_info=True)
        return ImageFileSourceMetadata()
    return ImageFileSourceMetadata(
        source_dtype=dtype,
        intensity_scale=(
            _image_file_declared_intensity_scale(path)
            or image_intensity_scale_for_dtype(dtype)
        ),
    )


def _image_file_declared_intensity_scale(path: Path) -> float | None:
    """Return declared source-file max intensity when the format exposes one."""
    try:
        import tifffile

        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tag = page.tags.get("SMaxSampleValue") or page.tags.get("MaxSampleValue")
            if tag is None:
                return None
            value = tag.value
    except Exception:
        logger.debug(
            "Could not read declared image intensity scale for %s.",
            path,
            exc_info=True,
        )
        return None
    try:
        scale = float(value[0] if isinstance(value, (tuple, list)) else value)
    except (TypeError, ValueError):
        return None
    return scale if scale > 0 else None


def _spatial_shape_pair(value: Sequence[int], name: str) -> tuple[int, int]:
    if len(value) < 2:
        raise ValueError(f"{name} must have at least two spatial dimensions.")
    return int(value[0]), int(value[1])


@dataclass(frozen=True, slots=True)
class SpatialShapeYX:
    """Nominal two-dimensional spatial shape in row/column order."""

    height: int
    width: int

    @classmethod
    def from_sequence(
        cls,
        value: Sequence[int],
        *,
        field_name: str,
    ) -> "SpatialShapeYX":
        if len(value) < 2:
            raise ValueError(
                f"{field_name} must have at least two spatial dimensions."
            )
        return cls(height=int(value[0]), width=int(value[1]))

    @classmethod
    def optional_from_mapping(
        cls,
        data: Mapping[str, Any],
        field_name: str,
    ) -> "SpatialShapeYX | None":
        if field_name not in data or data[field_name] is None:
            return None
        return cls.from_sequence(data[field_name], field_name=field_name)

    def as_tuple(self) -> tuple[int, int]:
        return self.height, self.width


def _common_metadata_value(values: Any) -> Any | None:
    values_tuple = tuple(values)
    present = tuple(value for value in values_tuple if value is not None)
    if not present:
        return None
    first = present[0]
    if all(value == first for value in present):
        return first
    return None


def _tuple_value(values: tuple[Any, ...], index: int) -> Any | None:
    if 0 <= index < len(values):
        return values[index]
    return None


@dataclass(frozen=True, slots=True)
class ImageMaskDomain:
    """Accepted mask shapes for a concrete grayscale/color image data domain."""

    data_shape: tuple[int, ...]

    def accepts(self, mask_shape: tuple[int, ...]) -> bool:
        return mask_shape in self.valid_shapes()

    def valid_shapes(self) -> frozenset[tuple[int, ...]]:
        valid: set[tuple[int, ...]] = {self.data_shape}
        if len(self.data_shape) == 2:
            valid.add(self.data_shape)
        if len(self.data_shape) == 3:
            if self.data_shape[-1] in (3, 4):
                valid.add(self.data_shape[:2])
            else:
                valid.add(self.data_shape[1:])
        if len(self.data_shape) == 4:
            if self.data_shape[-1] in (3, 4):
                valid.add(self.data_shape[:3])
                valid.add(self.data_shape[1:3])
            else:
                valid.add(self.data_shape[1:])
                valid.add(self.data_shape[-2:])
                valid.add((self.data_shape[0], *self.data_shape[-2:]))
        if len(self.data_shape) == 5:
            valid.add(self.data_shape[:4])
            valid.add(self.data_shape[1:4])
            valid.add((self.data_shape[0], *self.data_shape[-3:-1]))
            valid.add(self.data_shape[-3:-1])
        return frozenset(valid)


@dataclass(frozen=True, slots=True)
class ObjectLabelPayload(RuntimeArrayPayload, ObjectLabelDomainMetadata):
    """Dense object labels plus optional semantic label variants."""

    labels: Any
    unedited_labels: Any | None = None
    small_removed_labels: Any | None = None
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()
    declared_object_id_domains: tuple[tuple[int, ...], ...] = ()
    domain_scope: ObjectLabelDomainScope = ObjectLabelDomainScope.PAYLOAD
    plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.declared_object_count is not None:
            count = int(self.declared_object_count)
            if count < 0:
                raise ValueError("ObjectLabelPayload.declared_object_count cannot be negative.")
            object.__setattr__(self, "declared_object_count", count)
        ids = tuple(int(object_id) for object_id in self.declared_object_ids)
        if any(object_id <= 0 for object_id in ids):
            raise ValueError("ObjectLabelPayload.declared_object_ids must be positive.")
        object.__setattr__(self, "declared_object_ids", tuple(sorted(dict.fromkeys(ids))))
        object.__setattr__(
            self,
            "declared_object_id_domains",
            tuple(
                ObjectLabelDomain._normalize_ids(domain, "declared_object_id_domains")
                for domain in self.declared_object_id_domains
            ),
        )
        object.__setattr__(
            self,
            "domain_scope",
            coerce_enum(ObjectLabelDomainScope, self.domain_scope, "ObjectLabelPayload.domain_scope"),
        )
        object.__setattr__(
            self,
            "plane_axis",
            coerce_enum(RuntimePlaneAxis, self.plane_axis, "ObjectLabelPayload.plane_axis"),
        )
        if self.spatial_origin_yx is not None:
            object.__setattr__(
                self,
                "spatial_origin_yx",
                _spatial_shape_pair(self.spatial_origin_yx, "spatial_origin_yx"),
            )
        if self.source_spatial_shape_yx is not None:
            object.__setattr__(
                self,
                "source_spatial_shape_yx",
                _spatial_shape_pair(
                    self.source_spatial_shape_yx,
                    "source_spatial_shape_yx",
                ),
            )

    @property
    def shape(self) -> Any:
        return self.labels.shape

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def dtype(self) -> Any:
        return self.labels.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.labels, dtype=dtype)

    def array_payload_data(self) -> Any:
        return self.labels

    def object_label_domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain(
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            scope=self.domain_scope,
        )

    def __getitem__(self, key: Any) -> Any:
        return self.labels[key]

    def labels_for_variant(
        self,
        variant: ObjectLabelVariant | str,
    ) -> Any:
        normalized = coerce_enum(
            ObjectLabelVariant,
            variant,
            "ObjectLabelPayload.variant",
        )
        if normalized is ObjectLabelVariant.UNEDITED:
            return (
                self.unedited_labels
                if self.unedited_labels is not None
                else self.labels
            )
        if normalized is ObjectLabelVariant.SMALL_REMOVED:
            return (
                self.small_removed_labels
                if self.small_removed_labels is not None
                else self.labels
            )
        return self.labels

    @property
    def variants(self) -> tuple[ObjectLabelVariant, ...]:
        variants = [ObjectLabelVariant.FINAL]
        if self.unedited_labels is not None:
            variants.append(ObjectLabelVariant.UNEDITED)
        if self.small_removed_labels is not None:
            variants.append(ObjectLabelVariant.SMALL_REMOVED)
        return tuple(variants)

    def with_labels(
        self,
        labels: object,
        *,
        unedited_labels: object | None = None,
        small_removed_labels: object | None = None,
    ) -> "ObjectLabelPayload":
        """Return this payload's domain metadata with replacement labels."""
        return ObjectLabelPayload(
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
        )

    def with_data(self, data: Any) -> "ObjectLabelPayload":
        return self.with_labels(data)


class ColumnarRows(ABC):
    """Nominal ABC for table payloads exposing named columns."""

    @property
    @abstractmethod
    def columns(self) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class SparseIJVLabelRows(ColumnarRows):
    """Sparse object-label table with CellProfiler-compatible y/x/label columns."""

    data: Any

    def __post_init__(self) -> None:
        array = self.as_array()
        if array.ndim != 2 or array.shape[1] not in (3, 4):
            raise ValueError(
                "SparseIJVLabelRows.data must be an N x 3 y/x/label table "
                "or an N x 4 slice/y/x/label table."
            )

    @property
    def columns(self) -> Mapping[str, Any]:
        array = self.as_array()
        columns = {
            "y": array[:, self.y_column],
            "x": array[:, self.x_column],
            "label": array[:, self.label_column],
        }
        if self.has_slice_index:
            columns = {"slice_index": array[:, self.slice_column], **columns}
        return MappingProxyType(columns)

    @property
    def has_slice_index(self) -> bool:
        return int(self.as_array().shape[1]) == 4

    @property
    def slice_column(self) -> int:
        if not self.has_slice_index:
            raise ValueError("SparseIJVLabelRows has no slice_index column.")
        return 0

    @property
    def y_column(self) -> int:
        return 1 if self.has_slice_index else 0

    @property
    def x_column(self) -> int:
        return 2 if self.has_slice_index else 1

    @property
    def label_column(self) -> int:
        return 3 if self.has_slice_index else 2

    @classmethod
    def from_yx_label(cls, data: Any) -> "SparseIJVLabelRows":
        return cls(data)

    @classmethod
    def from_dense_labels(cls, labels: Any) -> "SparseIJVLabelRows":
        import numpy as _np

        label_array = _np.asarray(labels)
        if label_array.ndim != 2:
            raise ValueError(
                "SparseIJVLabelRows.from_dense_labels requires a 2-D label image."
            )
        rows, columns = _np.nonzero(label_array > 0)
        if rows.size == 0:
            return cls(_np.zeros((0, 3), dtype=_np.int32))
        return cls(
            _np.column_stack((rows, columns, label_array[rows, columns])).astype(
                _np.int32,
                copy=False,
            )
        )

    @classmethod
    def from_dense_stack(cls, labels: Any) -> "SparseIJVLabelRows":
        """Build sparse rows from a 2-D label image or runtime-slice stack."""
        import numpy as _np

        label_array = _np.asarray(labels)
        if label_array.ndim == 2:
            return cls.from_dense_labels(label_array)
        if label_array.ndim != 3:
            raise ValueError(
                "SparseIJVLabelRows.from_dense_stack requires a 2-D label image "
                "or a 3-D runtime-slice label stack."
            )
        slices = tuple(cls.from_dense_labels(slice_labels) for slice_labels in label_array)
        return cls.from_slices(slices)

    @classmethod
    def from_slices(cls, values: Sequence["SparseIJVLabelRows"]) -> "SparseIJVLabelRows":
        import numpy as _np

        arrays = []
        for slice_index, value in enumerate(values):
            array = value.as_yx_label_array()
            if not array.size:
                continue
            arrays.append(
                _np.column_stack(
                    (
                        _np.full(array.shape[0], slice_index, dtype=_np.int32),
                        array,
                    )
                )
            )
        return cls(
            _np.vstack(arrays).astype(_np.int32, copy=False)
            if arrays
            else _np.zeros((0, 4), dtype=_np.int32)
        )

    def as_yx_label_array(self) -> Any:
        array = self.as_array()
        if not self.has_slice_index:
            return array
        return array[:, (self.y_column, self.x_column, self.label_column)]

    def slice_indices(self) -> tuple[int, ...]:
        if not self.has_slice_index:
            return (0,)
        import numpy as _np

        return tuple(int(index) for index in _np.unique(self.as_array()[:, self.slice_column]))

    def slice(self, slice_index: int) -> "SparseIJVLabelRows":
        if not self.has_slice_index:
            if slice_index != 0:
                import numpy as _np

                return type(self)(_np.zeros((0, 3), dtype=_np.int32))
            return self
        array = self.as_array()
        rows = array[array[:, self.slice_column] == int(slice_index)]
        return type(self)(rows[:, (self.y_column, self.x_column, self.label_column)])

    def to_dense(
        self,
        *,
        source_spatial_shape_yx: tuple[int, int] | None = None,
        dtype: object | None = None,
    ) -> np.ndarray:
        """Materialize sparse IJV rows as dense 2-D or slice-stacked labels."""
        array = self.as_array()
        if dtype is None:
            dtype = array.dtype if array.size else np.int32
        height, width = self._dense_spatial_shape(source_spatial_shape_yx)
        if not self.has_slice_index:
            dense = np.zeros((height, width), dtype=dtype)
            if array.size:
                dense[
                    array[:, self.y_column].astype(np.intp, copy=False),
                    array[:, self.x_column].astype(np.intp, copy=False),
                ] = array[:, self.label_column].astype(dtype, copy=False)
            return dense
        slice_count = self.label_data_runtime_slice_count()
        dense = np.zeros((slice_count, height, width), dtype=dtype)
        if array.size:
            dense[
                array[:, self.slice_column].astype(np.intp, copy=False),
                array[:, self.y_column].astype(np.intp, copy=False),
                array[:, self.x_column].astype(np.intp, copy=False),
            ] = array[:, self.label_column].astype(dtype, copy=False)
        return dense

    def label_data_runtime_slice_count(self) -> int:
        """Return the encoded runtime-slice count, including empty stacks."""
        if not self.has_slice_index:
            return 1
        slice_indices = self.slice_indices()
        return max(slice_indices) + 1 if slice_indices else 0

    def _dense_spatial_shape(
        self,
        source_spatial_shape_yx: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if source_spatial_shape_yx is not None:
            return _spatial_shape_pair(source_spatial_shape_yx, "source_spatial_shape_yx")
        array = self.as_array()
        if not array.size:
            return (0, 0)
        return (
            int(np.max(array[:, self.y_column])) + 1,
            int(np.max(array[:, self.x_column])) + 1,
        )

    def as_array(self) -> Any:
        _ensure_runtime_payload_integrations_registered()
        try:
            import numpy as _np
        except Exception as exc:  # pragma: no cover - numpy is a core runtime dep.
            raise TypeError("SparseIJVLabelRows requires an array-like payload.") from exc
        return _np.asarray(self.data)


class SparseIJVLabelRowsIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from sparse IJV label rows without densifying."""

    value_type = SparseIJVLabelRows

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsIdDomainStrategy requires SparseIJVLabelRows, "
                f"got {type(labels).__name__}."
            )
        array = labels.as_array()
        if array.size == 0:
            return ()
        label_column = array[:, labels.label_column]
        return self.positive_ids_from_array(label_column)


def register_array_payload_type(payload_type: _TPayload) -> _TPayload:
    """Declare an external type as a runtime array payload."""
    RuntimeArrayPayload.register(payload_type)
    return payload_type


def register_array_payload_predicate(
    predicate: Callable[[Any], bool],
) -> Callable[[Any], bool]:
    """Declare a semantic predicate for runtime array payload recognition."""
    if predicate not in _ARRAY_PAYLOAD_PREDICATES:
        _ARRAY_PAYLOAD_PREDICATES.append(predicate)
    return predicate


def register_columnar_rows_type(payload_type: _TPayload) -> _TPayload:
    """Declare an external type as a columnar rows payload."""
    ColumnarRows.register(payload_type)
    return payload_type


@dataclass(frozen=True, kw_only=True)
class SourceImageContext:
    """Shared source-image semantic context for values and schemas."""

    dimensions: tuple[str, ...] = ()
    source_image_name: str | None = None

    def _validate_source_image_context(self, owner_name: str) -> None:
        if self.source_image_name == "":
            raise ValueError(f"{owner_name}.source_image_name cannot be empty.")


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimeValueSchema(SourceImageContext):
    """Semantic schema attached to a runtime artifact value."""

    kind: ArtifactKind
    slice_aligned: bool = False
    fields: tuple[FieldSpec, ...] = ()
    label_representation: ObjectLabelRepresentation | None = None
    measurement_subject: MeasurementSubject | None = None
    relationship: RelationshipSemantics | None = None
    object_name: str | None = None
    object_id_field: str | None = None
    label_variants: tuple[ObjectLabelVariant, ...] = ()

    def __post_init__(self) -> None:
        self._validate_source_image_context("RuntimeValueSchema")
        object.__setattr__(
            self,
            "kind",
            coerce_enum(ArtifactKind, self.kind, "RuntimeValueSchema.kind"),
        )
        if self.label_representation is not None:
            object.__setattr__(
                self,
                "label_representation",
                coerce_enum(
                    ObjectLabelRepresentation,
                    self.label_representation,
                    "RuntimeValueSchema.label_representation",
                ),
        )
        if self.object_name == "":
            raise ValueError("RuntimeValueSchema.object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("RuntimeValueSchema.object_id_field cannot be empty.")
        object.__setattr__(
            self,
            "label_variants",
            tuple(
                coerce_enum(
                    ObjectLabelVariant,
                    variant,
                    "RuntimeValueSchema.label_variants",
                )
                for variant in self.label_variants
            ),
        )
        if (
            self.label_representation is not None
            and self.kind is not ArtifactKind.OBJECT_LABELS
        ):
            raise ValueError(
                "RuntimeValueSchema.label_representation requires "
                "OBJECT_LABELS kind."
            )
        if self.label_variants and self.kind is not ArtifactKind.OBJECT_LABELS:
            raise ValueError(
                "RuntimeValueSchema.label_variants requires OBJECT_LABELS kind."
            )
        if (
            self.measurement_subject is not None
            and self.kind is not ArtifactKind.MEASUREMENTS
        ):
            raise ValueError(
                "RuntimeValueSchema.measurement_subject requires "
                "MEASUREMENTS kind."
            )
        if (
            self.relationship is not None
            and self.kind is not ArtifactKind.RELATIONSHIPS
        ):
            raise ValueError(
                "RuntimeValueSchema.relationship requires RELATIONSHIPS kind."
            )


@dataclass(frozen=True, slots=True)
class RuntimeStoragePolicy:
    """Storage intent for a runtime value once stores/materializers consume it."""

    backend: str | None = None
    path: str | None = None
    materialize: bool = False

    @classmethod
    def from_output_plan(cls, output_plan: ArtifactOutputPlan) -> Self:
        return cls(
            backend=Backend.MEMORY.value,
            path=output_plan.path,
            materialize=output_plan.materialization is not None,
        )

    def __post_init__(self) -> None:
        if self.path and not self.backend:
            raise ValueError("RuntimeStoragePolicy.path requires a backend.")


@dataclass(frozen=True, slots=True)
class RuntimeValue:
    """Artifact payload validated against compiled runtime semantics."""

    key: ArtifactKey
    data: Any
    schema: RuntimeValueSchema
    storage: RuntimeStoragePolicy | None = None

    @classmethod
    def from_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
        data: Any,
        *,
        axis_id: str,
        schema: RuntimeValueSchema,
    ) -> Self:
        return cls(
            key=output_plan.artifact_key(axis_id=axis_id),
            data=data,
            schema=schema,
            storage=RuntimeStoragePolicy.from_output_plan(output_plan),
        )

    def __post_init__(self) -> None:
        if self.key.kind is not self.schema.kind:
            raise ValueError(
                f"RuntimeValue key kind {self.key.kind.value} does not match "
                f"schema kind {self.schema.kind.value}."
            )

    @property
    def name(self) -> str:
        return self.key.name

    @property
    def kind(self) -> ArtifactKind:
        return self.key.kind


@dataclass(frozen=True, slots=True, kw_only=True)
class NativeRuntimeValue(ABC):
    """Native OpenHCS value that can become a validated RuntimeValue."""

    name: str

    def __post_init__(self) -> None:
        _require_name(self.name, f"{type(self).__name__}.name")

    @abstractmethod
    def runtime_payload(self) -> Any:
        """Return the payload stored under the compiled artifact key."""

    @abstractmethod
    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        """Return the schema that validates the stored payload."""

    def to_runtime_value(
        self,
        output_plan: ArtifactOutputPlan,
        *,
        axis_id: str,
    ) -> RuntimeValue:
        payload = self.runtime_payload()
        return RuntimeValue.from_output_plan(
            output_plan,
            payload,
            axis_id=axis_id,
            schema=self.runtime_schema(payload),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceImageRuntimeValue(SourceImageContext, NativeRuntimeValue, ABC):
    """Native value derived from a source image coordinate system."""

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        self._validate_source_image_context(type(self).__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class NamedImage(SourceImageRuntimeValue):
    """Native OpenHCS named image value."""

    data: Any

    def __post_init__(self) -> None:
        SourceImageRuntimeValue.__post_init__(self)
        if not _is_array_like(self.data):
            raise TypeError(
                f"NamedImage '{self.name}' requires array-like data with "
                f"shape/ndim, got {type(self.data).__name__}."
            )

    def runtime_payload(self) -> Any:
        return self.data

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(
            kind=ArtifactKind.IMAGE,
            dimensions=self.dimensions,
            source_image_name=self.source_image_name,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectLabelSet(SourceImageRuntimeValue, ObjectLabelDomainMetadata):
    """Native OpenHCS object-label value."""

    labels: Any
    unedited_labels: Any | None = None
    small_removed_labels: Any | None = None
    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()
    declared_object_id_domains: tuple[tuple[int, ...], ...] = ()
    domain_scope: ObjectLabelDomainScope = ObjectLabelDomainScope.PAYLOAD
    plane_axis: RuntimePlaneAxis = RuntimePlaneAxis.RUNTIME_SLICE
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct the native object-label view from a stored runtime value."""
        if value.kind is not ArtifactKind.OBJECT_LABELS:
            raise TypeError(
                "ObjectLabelSet.from_runtime_value requires an OBJECT_LABELS "
                f"runtime value, got {value.kind.value}."
            )
        payload = value.data
        schema = value.schema
        if isinstance(payload, ObjectLabelPayload):
            return cls(
                name=value.name,
                labels=payload.labels,
                unedited_labels=payload.unedited_labels,
                small_removed_labels=payload.small_removed_labels,
                declared_object_count=payload.declared_object_count,
                declared_object_ids=payload.declared_object_ids,
                declared_object_id_domains=payload.declared_object_id_domains,
                domain_scope=payload.domain_scope,
                plane_axis=payload.plane_axis,
                spatial_origin_yx=payload.spatial_origin_yx,
                source_spatial_shape_yx=payload.source_spatial_shape_yx,
                dimensions=schema.dimensions,
                source_image_name=schema.source_image_name,
                representation=(
                    schema.label_representation
                    or ObjectLabelRepresentation.DENSE_LABELS
                ),
            )
        return cls(
            name=value.name,
            labels=payload,
            dimensions=schema.dimensions,
            source_image_name=schema.source_image_name,
            representation=(
                schema.label_representation
                or ObjectLabelRepresentation.DENSE_LABELS
            ),
        )

    def __post_init__(self) -> None:
        SourceImageRuntimeValue.__post_init__(self)
        representation = coerce_enum(
            ObjectLabelRepresentation,
            self.representation,
            "ObjectLabelSet.representation",
        )
        object.__setattr__(self, "representation", representation)
        if isinstance(self.labels, ObjectLabelPayload):
            payload = self.labels
            object.__setattr__(self, "labels", payload.labels)
            object.__setattr__(self, "unedited_labels", payload.unedited_labels)
            object.__setattr__(
                self,
                "small_removed_labels",
                payload.small_removed_labels,
            )
            if self.declared_object_count is None:
                object.__setattr__(
                    self,
                    "declared_object_count",
                    payload.declared_object_count,
                )
            if not self.declared_object_ids:
                object.__setattr__(
                    self,
                    "declared_object_ids",
                    payload.declared_object_ids,
                )
            if not self.declared_object_id_domains:
                object.__setattr__(
                    self,
                    "declared_object_id_domains",
                    payload.declared_object_id_domains,
                )
            object.__setattr__(self, "domain_scope", payload.domain_scope)
            object.__setattr__(self, "plane_axis", payload.plane_axis)
            if self.spatial_origin_yx is None:
                object.__setattr__(
                    self,
                    "spatial_origin_yx",
                    payload.spatial_origin_yx,
                )
            if self.source_spatial_shape_yx is None:
                object.__setattr__(
                    self,
                    "source_spatial_shape_yx",
                    payload.source_spatial_shape_yx,
                )
        if self.declared_object_count is not None:
            count = int(self.declared_object_count)
            if count < 0:
                raise ValueError("ObjectLabelSet.declared_object_count cannot be negative.")
            object.__setattr__(self, "declared_object_count", count)
        ids = tuple(int(object_id) for object_id in self.declared_object_ids)
        if any(object_id <= 0 for object_id in ids):
            raise ValueError("ObjectLabelSet.declared_object_ids must be positive.")
        object.__setattr__(self, "declared_object_ids", tuple(sorted(dict.fromkeys(ids))))
        object.__setattr__(
            self,
            "declared_object_id_domains",
            tuple(
                ObjectLabelDomain._normalize_ids(domain, "declared_object_id_domains")
                for domain in self.declared_object_id_domains
            ),
        )
        object.__setattr__(
            self,
            "domain_scope",
            coerce_enum(ObjectLabelDomainScope, self.domain_scope, "ObjectLabelSet.domain_scope"),
        )
        object.__setattr__(
            self,
            "plane_axis",
            coerce_enum(RuntimePlaneAxis, self.plane_axis, "ObjectLabelSet.plane_axis"),
        )
        if self.spatial_origin_yx is not None:
            object.__setattr__(
                self,
                "spatial_origin_yx",
                _spatial_shape_pair(self.spatial_origin_yx, "spatial_origin_yx"),
            )
        if self.source_spatial_shape_yx is not None:
            object.__setattr__(
                self,
                "source_spatial_shape_yx",
                _spatial_shape_pair(
                    self.source_spatial_shape_yx,
                    "source_spatial_shape_yx",
                ),
            )
        validator = _PAYLOAD_VALIDATORS[representation.payload_shape]
        if validator is not None and not validator(self.labels):
            raise TypeError(
                f"ObjectLabelSet '{self.name}' requires "
                f"{representation.value} payload, got "
                f"{type(self.labels).__name__}."
            )
        _validate_object_label_variant(
            self.name,
            "unedited_labels",
            self.labels,
            self.unedited_labels,
            validator,
        )
        _validate_object_label_variant(
            self.name,
            "small_removed_labels",
            self.labels,
            self.small_removed_labels,
            validator,
        )

    def runtime_payload(self) -> Any:
        if (
            self.unedited_labels is not None
            or self.small_removed_labels is not None
            or self.declared_object_count is not None
            or self.declared_object_ids
            or self.declared_object_id_domains
            or self.domain_scope is not ObjectLabelDomainScope.PAYLOAD
            or self.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE
            or self.spatial_origin_yx is not None
            or self.source_spatial_shape_yx is not None
        ):
            return ObjectLabelPayload(
                labels=self.labels,
                unedited_labels=self.unedited_labels,
                small_removed_labels=self.small_removed_labels,
                declared_object_count=self.declared_object_count,
                declared_object_ids=self.declared_object_ids,
                declared_object_id_domains=self.declared_object_id_domains,
                domain_scope=self.domain_scope,
                plane_axis=self.plane_axis,
                spatial_origin_yx=self.spatial_origin_yx,
                source_spatial_shape_yx=self.source_spatial_shape_yx,
            )
        return self.labels

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        label_variants = (
            payload.variants
            if isinstance(payload, ObjectLabelPayload)
            else (ObjectLabelVariant.FINAL,)
        )
        return RuntimeValueSchema(
            kind=ArtifactKind.OBJECT_LABELS,
            dimensions=self.dimensions,
            label_representation=self.representation,
            label_variants=label_variants,
            object_name=self.name,
            source_image_name=self.source_image_name,
        )

    def object_label_domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain(
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            scope=self.domain_scope,
        )

    @property
    def shape(self) -> Any:
        return self.labels.shape

    @property
    def ndim(self) -> int:
        return self.labels.ndim

    @property
    def dtype(self) -> Any:
        return self.labels.dtype

    def __array__(self, dtype: Any | None = None) -> Any:
        import numpy as np

        return np.asarray(self.labels, dtype=dtype)

    def labels_for_variant(
        self,
        variant: ObjectLabelVariant | str,
    ) -> Any:
        payload = ObjectLabelPayload(
            labels=self.labels,
            unedited_labels=self.unedited_labels,
            small_removed_labels=self.small_removed_labels,
        )
        return payload.labels_for_variant(variant)

    def with_labels(
        self,
        labels: object,
        *,
        unedited_labels: object | None = None,
        small_removed_labels: object | None = None,
    ) -> "ObjectLabelSet":
        """Return this runtime value's metadata with replacement labels."""
        return ObjectLabelSet(
            name=self.name,
            labels=ObjectLabelSetReplacementStrategy.for_source(self).replacement_labels(
                labels
            ),
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            representation=self.representation,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=self.plane_axis,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
            dimensions=self.dimensions,
            source_image_name=self.source_image_name,
        )


class RuntimePayloadDataStrategy(ABC):
    """Base contract for nominal payload-to-data extraction strategies."""

    value_type: ClassVar[type[object] | None] = None

    @abstractmethod
    def data(self, payload: object) -> object:
        """Return the concrete data represented by payload."""


class ObjectLabelDenseDataStrategy(
    NominalTypeKeyedStrategyMixin,
    RuntimePayloadDataStrategy,
    metaclass=AutoRegisterMeta,
):
    """Registered dense-label extractor for one nominal object-label runtime type."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    value_type_label: ClassVar[str | None] = None

    @classmethod
    def for_payload(cls, payload: object) -> "ObjectLabelDenseDataStrategy":
        strategy = cls.for_nominal_value(payload)
        return strategy if strategy is not None else RawObjectLabelDenseDataStrategy()

    @classmethod
    def dense_data(cls, payload: object) -> object:
        """Return dense label data through the registered object-label contract."""
        return cls.for_payload(payload).data(payload)

    @classmethod
    def spatial_rank(cls, payload: object) -> int | None:
        """Return object-label dense spatial rank when the payload can materialize it."""
        dense_data = cls.dense_data(payload)
        if isinstance(dense_data, np.ndarray):
            return int(dense_data.ndim)
        try:
            return int(np.asarray(dense_data).ndim)
        except ValueError:
            return None

    @abstractmethod
    def data(self, payload: object) -> object:
        """Return the dense label data represented by payload."""


class ObjectLabelContainerDenseDataStrategy(ObjectLabelDenseDataStrategy):
    """Dense-label extractor for payloads with label-container semantics."""

    value_type: ClassVar[type[object] | None] = None

    def data(self, payload: object) -> object:
        value_type = type(self).value_type
        if value_type is None:
            raise TypeError(
                f"{type(self).__name__} must declare a concrete value_type."
            )
        if not isinstance(payload, value_type):
            raise TypeError(
                f"{type(self).__name__} requires {value_type.__name__}, "
                f"got {type(payload).__name__}."
            )
        if isinstance(payload.labels, SparseIJVLabelRows):
            return payload.labels.to_dense(
                source_spatial_shape_yx=payload.source_spatial_shape_yx,
            )
        return payload.labels


class ObjectLabelPayloadDenseDataStrategy(ObjectLabelContainerDenseDataStrategy):
    """Extract dense labels from serialized object-label payloads."""

    value_type = ObjectLabelPayload


class ObjectLabelSetDenseDataStrategy(ObjectLabelContainerDenseDataStrategy):
    """Extract dense labels from native object-label runtime values."""

    value_type = ObjectLabelSet


class RawObjectLabelDenseDataStrategy(ObjectLabelDenseDataStrategy):
    """Pass through already-dense array payloads."""

    def data(self, payload: object) -> object:
        return payload


class ObjectLabelPayloadIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        if not isinstance(labels, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadIdDomainStrategy requires ObjectLabelPayload, "
                f"got {type(labels).__name__}."
            )
        return ObjectLabelIdDomainStrategy.for_value(labels.labels).present_ids(labels.labels)


class ObjectLabelSetIdDomainStrategy(ObjectLabelIdDomainStrategy):
    """Extract present object IDs from native object-label runtime values."""

    value_type = ObjectLabelSet

    def present_ids(self, labels: Any) -> tuple[int, ...]:
        if not isinstance(labels, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetIdDomainStrategy requires ObjectLabelSet, "
                f"got {type(labels).__name__}."
            )
        return ObjectLabelIdDomainStrategy.for_value(labels.labels).present_ids(labels.labels)


def object_label_dense_data(payload: object) -> object:
    """Return dense label data through the registered object-label strategy family."""
    return ObjectLabelDenseDataStrategy.dense_data(payload)


def object_label_dense_array(
    payload: object,
    *,
    dtype: object | None = None,
    copy: bool | None = None,
) -> np.ndarray:
    """Materialize object-label dense data as a NumPy array via nominal extraction."""
    dense_data = object_label_dense_data(payload)
    if copy is None:
        return np.asarray(dense_data, dtype=dtype)
    return np.array(dense_data, dtype=dtype, copy=copy)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelAggregation:
    """Vectorized reductions over dense object-label IDs."""

    labels: np.ndarray
    object_count: int

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        dtype: object = np.int32,
    ) -> "DenseObjectLabelAggregation":
        """Build reductions from any nominal object-label payload."""
        labels = object_label_dense_array(payload, dtype=dtype)
        return cls(
            labels=labels,
            object_count=int(np.max(labels)) if labels.size else 0,
        )

    def counts(self) -> np.ndarray:
        """Return per-object pixel counts excluding the background label."""
        return np.bincount(
            self.labels,
            minlength=self.object_count + 1,
        )[1:].astype(float, copy=False)

    def sum(self, values: object) -> np.ndarray:
        """Return per-object sums for values aligned with ``labels``."""
        return np.bincount(
            self.labels,
            weights=np.asarray(values, dtype=float),
            minlength=self.object_count + 1,
        )[1:].astype(float, copy=False)

    def maximum(self, values: object) -> np.ndarray:
        """Return per-object maxima for values aligned with ``labels``."""
        maxima = np.zeros(self.object_count + 1, dtype=float)
        np.maximum.at(maxima, self.labels, np.asarray(values, dtype=float))
        return maxima[1:]

    def subset(self, mask: object) -> "DenseObjectLabelAggregation":
        """Return reductions over a masked subset of the same object ID domain."""
        return DenseObjectLabelAggregation(
            labels=self.labels[np.asarray(mask, dtype=bool)],
            object_count=self.object_count,
        )


class ObjectLabelDataRuntimeSliceStackContract(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare runtime-slice preservation for a concrete label representation."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def preserves_runtime_slice_stack(
        cls,
        labels: object,
        *,
        plane_axis: RuntimePlaneAxis,
        slice_count: int,
    ) -> bool:
        if plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return False
        strategy = cls.for_nominal_value(labels)
        return (
            False
            if strategy is None
            else strategy.label_data_preserves_runtime_slice_stack(
                labels,
                slice_count=slice_count,
            )
        )

    @classmethod
    def runtime_slice_count(
        cls,
        labels: object,
        *,
        plane_axis: RuntimePlaneAxis,
    ) -> int | None:
        """Return the runtime-slice count encoded by this label data."""
        if plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return None
        strategy = cls.for_nominal_value(labels)
        if strategy is None:
            return None
        return strategy.label_data_runtime_slice_count(labels)

    @abstractmethod
    def label_data_preserves_runtime_slice_stack(
        self,
        labels: object,
        *,
        slice_count: int,
    ) -> bool:
        """Return whether this label representation carries one row per runtime slice."""

    @abstractmethod
    def label_data_runtime_slice_count(self, labels: object) -> int | None:
        """Return the runtime-slice count carried by this label representation."""


class SparseIJVLabelRowsRuntimeSliceStackContract(
    ObjectLabelDataRuntimeSliceStackContract
):
    """Sparse IJV labels preserve runtime slicing when they declare slice indexes."""

    value_type = SparseIJVLabelRows

    def label_data_preserves_runtime_slice_stack(
        self,
        labels: object,
        *,
        slice_count: int,
    ) -> bool:
        del slice_count
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsRuntimeSliceStackContract requires "
                f"SparseIJVLabelRows, got {type(labels).__name__}."
            )
        return labels.has_slice_index

    def label_data_runtime_slice_count(self, labels: object) -> int | None:
        if not isinstance(labels, SparseIJVLabelRows):
            raise TypeError(
                "SparseIJVLabelRowsRuntimeSliceStackContract requires "
                f"SparseIJVLabelRows, got {type(labels).__name__}."
            )
        if not labels.has_slice_index:
            return None
        return labels.label_data_runtime_slice_count()


class DenseArrayLabelRuntimeSliceStackContract(
    ObjectLabelDataRuntimeSliceStackContract
):
    """Dense array labels preserve runtime slicing when axis 0 is the slice axis."""

    value_type = np.ndarray

    def label_data_preserves_runtime_slice_stack(
        self,
        labels: object,
        *,
        slice_count: int,
    ) -> bool:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "DenseArrayLabelRuntimeSliceStackContract requires ndarray, "
                f"got {type(labels).__name__}."
            )
        return labels.ndim >= 3 and int(labels.shape[0]) == int(slice_count)

    def label_data_runtime_slice_count(self, labels: object) -> int | None:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "DenseArrayLabelRuntimeSliceStackContract requires ndarray, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim < 3:
            return None
        return int(labels.shape[0])


class ObjectLabelRuntimeSliceStackContract(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Declare whether an object-label payload carries the runtime slice axis."""

    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def preserves_runtime_slice_stack(
        cls,
        value: object,
        *,
        slice_count: int,
    ) -> bool:
        strategy = cls.for_nominal_value(value)
        return (
            False
            if strategy is None
            else strategy.value_preserves_runtime_slice_stack(
                value,
                slice_count=slice_count,
            )
        )

    @classmethod
    def runtime_slice_count(cls, value: object) -> int | None:
        """Return the runtime-slice count encoded by one object-label value."""
        strategy = cls.for_nominal_value(value)
        if strategy is None:
            return None
        return strategy.value_runtime_slice_count(value)

    @abstractmethod
    def value_preserves_runtime_slice_stack(
        self,
        value: object,
        *,
        slice_count: int,
    ) -> bool:
        """Return whether the payload must remain stack-shaped until execution."""

    @abstractmethod
    def value_runtime_slice_count(self, value: object) -> int | None:
        """Return the runtime-slice count carried by this object-label value."""


class ObjectLabelContainerRuntimeSliceStackContract(ObjectLabelRuntimeSliceStackContract):
    """Runtime-slice contract shared by nominal object-label containers."""

    value_type: ClassVar[type[Any] | None] = None

    def typed_value(self, value: object) -> Any:
        """Return value after validating it belongs to this registered family."""
        value_type = type(self).value_type
        if value_type is None:
            raise TypeError(
                f"{type(self).__name__} must declare a concrete value_type."
            )
        if not isinstance(value, value_type):
            raise TypeError(
                f"{type(self).__name__} requires {value_type.__name__}, "
                f"got {type(value).__name__}."
            )
        return value

    def value_preserves_runtime_slice_stack(
        self,
        value: object,
        *,
        slice_count: int,
    ) -> bool:
        payload = self.typed_value(value)
        if payload.domain_scope is not ObjectLabelDomainScope.PLANE:
            return False
        return ObjectLabelDataRuntimeSliceStackContract.preserves_runtime_slice_stack(
            payload.labels,
            plane_axis=payload.plane_axis,
            slice_count=slice_count,
        )

    def value_runtime_slice_count(self, value: object) -> int | None:
        payload = self.typed_value(value)
        if payload.domain_scope is not ObjectLabelDomainScope.PLANE:
            return None
        return ObjectLabelDataRuntimeSliceStackContract.runtime_slice_count(
            payload.labels,
            plane_axis=payload.plane_axis,
        )


class ObjectLabelPayloadRuntimeSliceStackContract(
    ObjectLabelContainerRuntimeSliceStackContract
):
    """Runtime-slice contract for dense object-label payloads."""

    value_type = ObjectLabelPayload


class ObjectLabelSetRuntimeSliceStackContract(
    ObjectLabelContainerRuntimeSliceStackContract
):
    """Runtime-slice contract for native object-label sets."""

    value_type = ObjectLabelSet


class ObjectLabelPure2DSliceAggregator(ABC, metaclass=AutoRegisterMeta):
    """Aggregate object-label slices while preserving label-domain semantics."""

    __registry_key__ = "label_type"
    __registry__: ClassVar[dict[Any, type["ObjectLabelPure2DSliceAggregator"]]] = {}
    label_type: ClassVar[type[Any] | None] = None

    def __init__(self, values: Sequence[Any], memory_type: str) -> None:
        self.values = tuple(values)
        self.memory_type = memory_type

    @classmethod
    def aggregate(
        cls,
        values: Sequence[Any],
        memory_type: str,
    ) -> Any:
        for aggregator_type in cls.__registry__.values():
            if aggregator_type.supports(values):
                return aggregator_type(values, memory_type).aggregate_values()
        raise TypeError("No object-label slice aggregator owns these values.")

    @classmethod
    def supports(cls, values: Sequence[Any]) -> bool:
        return cls.label_type is not None and bool(values) and all(
            isinstance(value, cls.label_type) for value in values
        )

    @property
    def first(self) -> Any:
        return self.values[0]

    @property
    def declared_object_count(self) -> int | None:
        return self.common_value(value.declared_object_count for value in self.values)

    @property
    def declared_object_ids(self) -> tuple[int, ...]:
        return self.common_value(value.declared_object_ids for value in self.values) or ()

    @property
    def declared_object_id_domains(self) -> tuple[tuple[int, ...], ...]:
        if len(self.values) == 1:
            return ObjectLabelDomain.explicit_plane_id_domains(
                value.object_label_domain() for value in self.values
            )
        return tuple(
            plane_domain
            for value in self.values
            for plane_domain in self.value_plane_id_domains(value)
        )

    def value_plane_id_domains(self, value: Any) -> tuple[tuple[int, ...], ...]:
        """Return the object-id domain represented by one PURE_2D output slice."""
        return ObjectLabelPlaneDomainStrategy.for_enum_member(
            value.domain_scope
        ).identity_domains(
            value.labels_for_variant(ObjectLabelVariant.FINAL),
            declared_object_count=value.declared_object_count,
            declared_object_ids=value.declared_object_ids,
            declared_object_id_domains=value.declared_object_id_domains,
        )

    @property
    def domain_scope(self) -> ObjectLabelDomainScope:
        if len(self.values) > 1:
            return ObjectLabelDomainScope.PLANE
        return ObjectLabelDomainScope.common(value.domain_scope for value in self.values)

    @property
    def source_spatial_shape_yx(self) -> tuple[int, int] | None:
        return self.common_value(value.source_spatial_shape_yx for value in self.values)

    @property
    def spatial_origin_yx(self) -> tuple[int, int] | None:
        if self.expands_to_source_domain:
            return None
        return self.common_value(value.spatial_origin_yx for value in self.values)

    @property
    def expands_to_source_domain(self) -> bool:
        if len(self.values) <= 1:
            return False
        domains = tuple(
            (value.spatial_origin_yx, value.source_spatial_shape_yx)
            for value in self.values
        )
        if any(origin is None or source_shape is None for origin, source_shape in domains):
            return False
        return len(set(domains)) > 1

    @staticmethod
    def common_value(values: Any) -> Any | None:
        unique_values = tuple(dict.fromkeys(values))
        if len(unique_values) == 1:
            return unique_values[0]
        return None

    def aggregate_values(self) -> Any:
        return self.output_value(
            labels=self.aggregate_variant(ObjectLabelVariant.FINAL),
            unedited_labels=(
                self.aggregate_variant(ObjectLabelVariant.UNEDITED)
                if self.has_variant(ObjectLabelVariant.UNEDITED)
                else None
            ),
            small_removed_labels=(
                self.aggregate_variant(ObjectLabelVariant.SMALL_REMOVED)
                if self.has_variant(ObjectLabelVariant.SMALL_REMOVED)
                else None
            ),
        )

    def has_variant(self, variant: ObjectLabelVariant) -> bool:
        if variant is ObjectLabelVariant.UNEDITED:
            return any(value.unedited_labels is not None for value in self.values)
        if variant is ObjectLabelVariant.SMALL_REMOVED:
            return any(value.small_removed_labels is not None for value in self.values)
        return True

    def aggregate_variant(self, variant: ObjectLabelVariant) -> Any:
        return stack_runtime_object_label_slices(
            [self.slice_labels(value, variant) for value in self.values],
            self.memory_type,
        )

    def slice_labels(self, value: Any, variant: ObjectLabelVariant) -> Any:
        if not self.expands_to_source_domain:
            return value.labels_for_variant(variant)
        domain_value = self.domain_value_for_variant(value, variant)
        aligned, _ = aligned_dense_object_label_arrays(domain_value, domain_value)
        return aligned

    @abstractmethod
    def domain_value_for_variant(
        self,
        value: Any,
        variant: ObjectLabelVariant,
    ) -> Any:
        """Return a typed label value carrying the selected variant and domain."""

    @abstractmethod
    def output_value(
        self,
        *,
        labels: Any,
        unedited_labels: Any | None,
        small_removed_labels: Any | None,
    ) -> Any:
        """Build the aggregated object-label value."""


class ObjectLabelPayloadPure2DSliceAggregator(ObjectLabelPure2DSliceAggregator):
    """Aggregate dense object-label payload slices."""

    label_type = ObjectLabelPayload

    def domain_value_for_variant(
        self,
        value: ObjectLabelPayload,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelPayload:
        return ObjectLabelPayload(
            labels=value.labels_for_variant(variant),
            declared_object_count=value.declared_object_count,
            declared_object_ids=value.declared_object_ids,
            declared_object_id_domains=value.declared_object_id_domains,
            domain_scope=value.domain_scope,
            plane_axis=value.plane_axis,
            spatial_origin_yx=value.spatial_origin_yx,
            source_spatial_shape_yx=value.source_spatial_shape_yx,
        )

    def output_value(
        self,
        *,
        labels: Any,
        unedited_labels: Any | None,
        small_removed_labels: Any | None,
    ) -> ObjectLabelPayload:
        return ObjectLabelPayload(
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
        )


class ObjectLabelSetPure2DSliceAggregator(ObjectLabelPure2DSliceAggregator):
    """Aggregate native object-label set slices."""

    label_type = ObjectLabelSet

    @property
    def representation(self) -> ObjectLabelRepresentation:
        representations = {value.representation for value in self.values}
        if len(representations) != 1:
            raise ValueError(
                "Cannot aggregate mixed object-label representations across PURE_2D slices."
            )
        return representations.pop()

    def aggregate_values(self) -> ObjectLabelSet:
        if self.representation is ObjectLabelRepresentation.SPARSE_IJV:
            return self.aggregate_sparse_ijv()
        return super().aggregate_values()

    def aggregate_sparse_ijv(self) -> ObjectLabelSet:
        return ObjectLabelSet(
            name=self.first.name,
            labels=SparseIJVLabelRows.from_slices(
                tuple(
                    value.labels
                    if isinstance(value.labels, SparseIJVLabelRows)
                    else SparseIJVLabelRows.from_yx_label(value.labels)
                    for value in self.values
                )
            ),
            representation=self.representation,
            dimensions=self.first.dimensions,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_name=self.first.source_image_name,
        )

    def domain_value_for_variant(
        self,
        value: ObjectLabelSet,
        variant: ObjectLabelVariant,
    ) -> ObjectLabelSet:
        return ObjectLabelSet(
            name=value.name,
            labels=value.labels_for_variant(variant),
            representation=value.representation,
            declared_object_count=value.declared_object_count,
            declared_object_ids=value.declared_object_ids,
            declared_object_id_domains=value.declared_object_id_domains,
            domain_scope=value.domain_scope,
            plane_axis=value.plane_axis,
            spatial_origin_yx=value.spatial_origin_yx,
            source_spatial_shape_yx=value.source_spatial_shape_yx,
            dimensions=value.dimensions,
            source_image_name=value.source_image_name,
        )

    def output_value(
        self,
        *,
        labels: Any,
        unedited_labels: Any | None,
        small_removed_labels: Any | None,
    ) -> ObjectLabelSet:
        return ObjectLabelSet(
            name=self.first.name,
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
            declared_object_id_domains=self.declared_object_id_domains,
            domain_scope=self.domain_scope,
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            representation=self.representation,
            dimensions=self.first.dimensions,
            source_image_name=self.first.source_image_name,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
        )


def stack_runtime_object_label_slices(values: Sequence[Any], memory_type: str) -> Any:
    """Stack dense runtime object-label slices using the declared memory backend."""
    slice_values = tuple(values)
    try:
        return ImageStackLayout.for_slices(slice_values).stack(
            slices=slice_values,
            memory_type=memory_type,
            gpu_id=0,
        )
    except ValueError:
        return np.stack(tuple(np.asarray(value) for value in slice_values), axis=0)


class ObjectLabelPayloadBuilderStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered constructor for preserving object-label metadata across transforms."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_source(cls, source: object) -> "ObjectLabelPayloadBuilderStrategy":
        strategy = cls.for_nominal_value(source)
        return strategy if strategy is not None else RawObjectLabelPayloadBuilderStrategy()

    @abstractmethod
    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        """Return transformed labels wrapped in the source object's semantic domain."""


class ObjectLabelPayloadBuilder(ObjectLabelPayloadBuilderStrategy):
    """Preserve metadata from serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        if not isinstance(source, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadBuilder requires ObjectLabelPayload, "
                f"got {type(source).__name__}."
            )
        return ObjectLabelPayload(
            labels=labels,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
            plane_axis=source.plane_axis,
            spatial_origin_yx=source.spatial_origin_yx,
            source_spatial_shape_yx=source.source_spatial_shape_yx,
        )


class ObjectLabelSetPayloadBuilder(ObjectLabelPayloadBuilderStrategy):
    """Preserve metadata from native object-label runtime values."""

    value_type = ObjectLabelSet

    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetPayloadBuilder requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return ObjectLabelPayload(
            labels=labels,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
            plane_axis=source.plane_axis,
            spatial_origin_yx=source.spatial_origin_yx,
            source_spatial_shape_yx=source.source_spatial_shape_yx,
        )


class RawObjectLabelPayloadBuilderStrategy(ObjectLabelPayloadBuilderStrategy):
    """Build a semantic payload for already-dense object labels."""

    def build(
        self,
        source: object,
        labels: object,
        *,
        declared_domain: ObjectLabelDomain,
    ) -> ObjectLabelPayload:
        return ObjectLabelPayload(
            labels=labels,
            declared_object_count=declared_domain.declared_object_count,
            declared_object_ids=declared_domain.declared_object_ids,
            declared_object_id_domains=declared_domain.declared_object_id_domains,
            domain_scope=declared_domain.scope,
        )


def object_label_payload_with_dense_labels(
    source: object,
    labels: object,
    *,
    domain_declaration: ObjectLabelDomainDeclaration = (
        PreserveSourceObjectLabelDomainDeclaration()
    ),
) -> ObjectLabelPayload:
    """Build a dense-label payload while preserving nominal object-label metadata."""
    declared_domain = domain_declaration.declared_domain(source, labels)
    return ObjectLabelPayloadBuilderStrategy.for_source(source).build(
        source,
        labels,
        declared_domain=declared_domain,
    )


def object_label_payload_from_source_image(
    image: object,
    labels: object,
    *,
    declared_object_count: int | None = None,
    declared_object_ids: tuple[int, ...] = (),
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelPayload:
    """Build labels in the source spatial domain carried by an image payload."""
    metadata = image_payload_metadata(image)
    return ObjectLabelPayload(
        labels=labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        declared_object_count=declared_object_count,
        declared_object_ids=declared_object_ids,
        spatial_origin_yx=metadata.spatial_origin_yx,
        source_spatial_shape_yx=metadata.source_spatial_shape_yx,
    )


def object_label_set_from_source_image(
    image: object,
    *,
    name: str,
    labels: object,
    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS,
    declared_object_count: int | None = None,
    declared_object_ids: tuple[int, ...] = (),
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelSet:
    """Build native object labels in the source spatial domain carried by an image."""
    metadata = image_payload_metadata(image)
    return ObjectLabelSet(
        name=name,
        labels=labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
        representation=representation,
        declared_object_count=declared_object_count,
        declared_object_ids=declared_object_ids,
        spatial_origin_yx=metadata.spatial_origin_yx,
        source_spatial_shape_yx=metadata.source_spatial_shape_yx,
    )


class ObjectLabelSetReplacementStrategy(
    EnumKeyedStrategyMixin[ObjectLabelRepresentation],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered replacement policy for ObjectLabelSet label representations."""

    representation: ClassVar[ObjectLabelRepresentation | None] = None
    representation_label: ClassVar[str | None] = None
    __registry_key__ = "representation_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "representation"
    __enum_label_attr__ = "representation_label"

    @classmethod
    def for_source(
        cls,
        source: "ObjectLabelSet",
    ) -> "ObjectLabelSetReplacementStrategy":
        return cls.for_enum_member(source.representation)

    @abstractmethod
    def replacement_labels(self, labels: object) -> object:
        """Return labels compatible with this representation."""


class DenseObjectLabelSetReplacementStrategy(ObjectLabelSetReplacementStrategy):
    """Dense labels are already the concrete replacement payload."""

    representation = ObjectLabelRepresentation.DENSE_LABELS

    def replacement_labels(self, labels: object) -> object:
        return labels


class SparseIJVObjectLabelSetReplacementStrategy(ObjectLabelSetReplacementStrategy):
    """Sparse-IJV replacements use the sparse rows carried by nominal label sets."""

    representation = ObjectLabelRepresentation.SPARSE_IJV

    def replacement_labels(self, labels: object) -> object:
        if isinstance(labels, SparseIJVLabelRows):
            return labels
        if isinstance(labels, ObjectLabelSet):
            if labels.representation is not ObjectLabelRepresentation.SPARSE_IJV:
                return SparseIJVLabelRows.from_dense_stack(
                    ObjectLabelDenseDataStrategy.for_payload(labels).data(labels)
                )
            return labels.labels
        if isinstance(labels, ObjectLabelPayload):
            if isinstance(labels.labels, SparseIJVLabelRows):
                return labels.labels
            return SparseIJVLabelRows.from_dense_stack(
                ObjectLabelDenseDataStrategy.for_payload(labels).data(labels)
            )
        return SparseIJVLabelRows.from_dense_stack(labels)


def object_label_set_with_replacement_labels(
    source: ObjectLabelSet,
    labels: object,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelSet:
    """Return an ObjectLabelSet with representation-compatible replacement labels."""
    return source.with_labels(
        labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
    )


def object_label_payload_with_replacement_labels(
    source: ObjectLabelPayload,
    labels: object,
    *,
    unedited_labels: object | None = None,
    small_removed_labels: object | None = None,
) -> ObjectLabelPayload:
    """Return an object-label payload with replacement labels and preserved domain."""
    return source.with_labels(
        labels,
        unedited_labels=unedited_labels,
        small_removed_labels=small_removed_labels,
    )


class ObjectLabelVariantCompatibilityStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered policy for retaining label variants after label replacement."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_variant(
        cls,
        variant: object,
    ) -> "ObjectLabelVariantCompatibilityStrategy":
        strategy = cls.for_nominal_value(variant)
        return strategy if strategy is not None else RawObjectLabelVariantCompatibilityStrategy()

    @abstractmethod
    def matching_labels(self, variant: object, labels: object) -> object | None:
        """Return variant when it is compatible with replacement labels."""


class RuntimeArrayObjectLabelVariantCompatibilityStrategy(
    ObjectLabelVariantCompatibilityStrategy
):
    """Runtime-array variants must inhabit the same dense label shape."""

    value_type = RuntimeArrayPayload

    def matching_labels(self, variant: object, labels: object) -> object | None:
        if not isinstance(variant, RuntimeArrayPayload):
            raise TypeError(
                "RuntimeArrayObjectLabelVariantCompatibilityStrategy requires "
                f"RuntimeArrayPayload, got {type(variant).__name__}."
            )
        if not isinstance(labels, RuntimeArrayPayload):
            return variant
        if tuple(variant.shape) == tuple(labels.shape):
            return variant
        return None


class NumpyObjectLabelVariantCompatibilityStrategy(
    ObjectLabelVariantCompatibilityStrategy
):
    """NumPy variants must inhabit the same dense label shape."""

    value_type = np.ndarray

    def matching_labels(self, variant: object, labels: object) -> object | None:
        if not isinstance(variant, np.ndarray):
            raise TypeError(
                "NumpyObjectLabelVariantCompatibilityStrategy requires ndarray, "
                f"got {type(variant).__name__}."
            )
        if not isinstance(labels, np.ndarray):
            return variant
        if tuple(variant.shape) == tuple(labels.shape):
            return variant
        return None


class RawObjectLabelVariantCompatibilityStrategy(
    ObjectLabelVariantCompatibilityStrategy
):
    """Unknown variants are metadata-only and remain attached."""

    def matching_labels(self, variant: object, labels: object) -> object | None:
        return variant


def object_label_variant_matching_labels(
    variant: object | None,
    labels: object,
) -> object | None:
    """Return a variant only when it is compatible with replacement labels."""
    if variant is None:
        return None
    return ObjectLabelVariantCompatibilityStrategy.for_variant(
        variant
    ).matching_labels(variant, labels)


class ObjectLabelMeasurementPayloadStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered policy for replacing labels used in measurement contexts."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_source(cls, source: object) -> "ObjectLabelMeasurementPayloadStrategy":
        strategy = cls.for_nominal_value(source)
        return strategy if strategy is not None else RawObjectLabelMeasurementPayloadStrategy()

    @abstractmethod
    def with_labels(self, source: object, labels: object) -> object:
        """Return source metadata over labels selected for measurement."""


class ObjectLabelSetMeasurementPayloadStrategy(ObjectLabelMeasurementPayloadStrategy):
    """Replace labels for native object-label runtime values."""

    value_type = ObjectLabelSet

    def with_labels(self, source: object, labels: object) -> object:
        if not isinstance(source, ObjectLabelSet):
            raise TypeError(
                "ObjectLabelSetMeasurementPayloadStrategy requires ObjectLabelSet, "
                f"got {type(source).__name__}."
            )
        return object_label_set_with_replacement_labels(
            source,
            labels,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )


class ObjectLabelPayloadMeasurementPayloadStrategy(
    ObjectLabelMeasurementPayloadStrategy
):
    """Replace labels for serialized object-label payloads."""

    value_type = ObjectLabelPayload

    def with_labels(self, source: object, labels: object) -> object:
        if not isinstance(source, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadMeasurementPayloadStrategy requires "
                f"ObjectLabelPayload, got {type(source).__name__}."
            )
        return object_label_payload_with_replacement_labels(
            source,
            labels,
            unedited_labels=object_label_variant_matching_labels(
                source.unedited_labels,
                labels,
            ),
            small_removed_labels=object_label_variant_matching_labels(
                source.small_removed_labels,
                labels,
            ),
        )


class RawObjectLabelMeasurementPayloadStrategy(ObjectLabelMeasurementPayloadStrategy):
    """Dense arrays have no nominal metadata to preserve."""

    def with_labels(self, source: object, labels: object) -> object:
        return labels


def object_label_payload_with_measurement_labels(
    source: object,
    labels: object,
) -> object:
    """Return object-label metadata over labels selected for measurement."""
    return ObjectLabelMeasurementPayloadStrategy.for_source(source).with_labels(
        source,
        labels,
    )


class SingletonObjectLabelStackCollapseStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered collapse policy for singleton object-label stacks."""

    value_type: ClassVar[type[object] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True

    @classmethod
    def for_labels(cls, labels: object) -> "SingletonObjectLabelStackCollapseStrategy":
        strategy = cls.for_nominal_value(labels)
        return strategy if strategy is not None else RawSingletonObjectLabelStackCollapseStrategy()

    @abstractmethod
    def collapse(self, labels: object) -> object:
        """Collapse singleton stacked labels when the strategy applies."""


class ObjectLabelPayloadStackCollapseStrategy(
    SingletonObjectLabelStackCollapseStrategy
):
    """Collapse singleton serialized object-label payload stacks."""

    value_type = ObjectLabelPayload

    def collapse(self, labels: object) -> object:
        if not isinstance(labels, ObjectLabelPayload):
            raise TypeError(
                "ObjectLabelPayloadStackCollapseStrategy requires ObjectLabelPayload, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim != 3 or labels.shape[0] != 1:
            return labels
        return object_label_payload_with_replacement_labels(
            labels,
            labels.labels[0],
            unedited_labels=(
                None if labels.unedited_labels is None else labels.unedited_labels[0]
            ),
            small_removed_labels=(
                None
                if labels.small_removed_labels is None
                else labels.small_removed_labels[0]
            ),
        )


class RuntimeArrayStackCollapseStrategy(SingletonObjectLabelStackCollapseStrategy):
    """Collapse singleton nominal array payload stacks."""

    value_type = RuntimeArrayPayload

    def collapse(self, labels: object) -> object:
        if not isinstance(labels, RuntimeArrayPayload):
            raise TypeError(
                "RuntimeArrayStackCollapseStrategy requires RuntimeArrayPayload, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim == 3 and labels.shape[0] == 1:
            return labels[0]
        return labels


class NumpyObjectLabelStackCollapseStrategy(SingletonObjectLabelStackCollapseStrategy):
    """Collapse singleton NumPy label stacks."""

    value_type = np.ndarray

    def collapse(self, labels: object) -> object:
        if not isinstance(labels, np.ndarray):
            raise TypeError(
                "NumpyObjectLabelStackCollapseStrategy requires ndarray, "
                f"got {type(labels).__name__}."
            )
        if labels.ndim == 3 and labels.shape[0] == 1:
            return labels[0]
        return labels


class RawSingletonObjectLabelStackCollapseStrategy(
    SingletonObjectLabelStackCollapseStrategy
):
    """Unknown payloads are not singleton object-label stacks."""

    def collapse(self, labels: object) -> object:
        return labels


def collapse_singleton_object_label_stack(labels: object) -> object:
    """Normalize singleton object-label stacks to one label plane."""
    return SingletonObjectLabelStackCollapseStrategy.for_labels(labels).collapse(labels)


@dataclass(frozen=True, slots=True)
class DenseObjectLabelSliceStack:
    """Dense object labels projected onto a fixed slice axis."""

    labels: np.ndarray

    @classmethod
    def from_payload(
        cls,
        payload: object,
        *,
        slice_count: int,
        dtype: object | None = None,
    ) -> "DenseObjectLabelSliceStack | None":
        label_array = object_label_dense_array(payload, dtype=dtype)
        if label_array.ndim == 3 and label_array.shape[0] == slice_count:
            return cls(np.ascontiguousarray(label_array))
        if label_array.ndim == 2:
            return cls(
                np.ascontiguousarray(
                    np.broadcast_to(label_array, (slice_count, *label_array.shape))
                )
            )
        return None

    def slice(self, slice_index: int) -> np.ndarray:
        return self.labels[slice_index]


class RuntimeSliceAlignedPayloadNormalizationStrategy(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Normalize nominal slice-aligned payloads before runtime storage."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "kind"

    kind: ClassVar[ArtifactKind]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_output_plan(
        cls,
        output_plan: ArtifactOutputPlan,
    ) -> "RuntimeSliceAlignedPayloadNormalizationStrategy | None":
        strategy_type = cls.__registry__.get(output_plan.kind.value)
        return strategy_type() if strategy_type is not None else None

    @abstractmethod
    def normalize(
        self,
        value: RuntimeSliceAlignedValueSet[Any],
    ) -> object | None:
        """Return an aggregate payload, or ``None`` when slices are not owned."""


class ObjectLabelSliceAlignedPayloadNormalizationStrategy(
    RuntimeSliceAlignedPayloadNormalizationStrategy
):
    """Aggregate object-label slice payloads into one plane-scoped label domain."""

    kind = ArtifactKind.OBJECT_LABELS

    def normalize(
        self,
        value: RuntimeSliceAlignedValueSet[Any],
    ) -> object | None:
        slices = tuple(value.value_for_slice(index) for index in range(value.slice_count))
        if not slices or not all(
            isinstance(item, (ObjectLabelPayload, ObjectLabelSet))
            for item in slices
        ):
            return None
        label_sets = tuple(
            item
            if isinstance(item, ObjectLabelSet)
            else ObjectLabelSet(name="slice", labels=item)
            for item in slices
        )
        labels = np.stack(
            [object_label_dense_array(label_set, dtype=np.int32) for label_set in label_sets],
            axis=0,
        )
        unedited_labels = (
            np.stack(
                [
                    object_label_dense_array(label_set.labels_for_variant("unedited"))
                    for label_set in label_sets
                ],
                axis=0,
            )
            if any(label_set.unedited_labels is not None for label_set in label_sets)
            else None
        )
        small_removed_labels = (
            np.stack(
                [
                    object_label_dense_array(label_set.labels_for_variant("small_removed"))
                    for label_set in label_sets
                ],
                axis=0,
            )
            if any(label_set.small_removed_labels is not None for label_set in label_sets)
            else None
        )
        declared_counts = {label_set.declared_object_count for label_set in label_sets}
        declared_ids = {label_set.declared_object_ids for label_set in label_sets}
        declared_id_domains = ObjectLabelDomain.explicit_plane_id_domains(
            label_set.object_label_domain() for label_set in label_sets
        )
        return ObjectLabelPayload(
            labels=labels,
            unedited_labels=unedited_labels,
            small_removed_labels=small_removed_labels,
            declared_object_count=(
                declared_counts.pop() if len(declared_counts) == 1 else None
            ),
            declared_object_ids=declared_ids.pop() if len(declared_ids) == 1 else (),
            declared_object_id_domains=declared_id_domains,
            domain_scope=(
                ObjectLabelDomainScope.PLANE
                if len(label_sets) > 1
                else ObjectLabelDomainScope.common(
                    label_set.domain_scope for label_set in label_sets
                )
            ),
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementTable(NativeRuntimeValue):
    """Native OpenHCS measurement table value."""

    rows: Any
    object_name: str | None = None
    fields: tuple[FieldSpec, ...] = ()
    object_id_field: str | None = None
    source_image_name: str | None = None
    subject: MeasurementSubject | None = None
    validated_runtime_schema: InitVar[bool] = False

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct the native measurement view from a stored runtime value."""
        if value.kind is not ArtifactKind.MEASUREMENTS:
            raise TypeError(
                "MeasurementTable.from_runtime_value requires a MEASUREMENTS "
                f"runtime value, got {value.kind.value}."
            )
        return cls(
            name=value.name,
            rows=value.data,
            object_name=value.schema.object_name,
            fields=value.schema.fields,
            object_id_field=value.schema.object_id_field,
            source_image_name=value.schema.source_image_name,
            subject=value.schema.measurement_subject,
            validated_runtime_schema=True,
        )

    def __post_init__(self, validated_runtime_schema: bool) -> None:
        NativeRuntimeValue.__post_init__(self)
        if self.object_name == "":
            raise ValueError("MeasurementTable.object_name cannot be empty.")
        if self.object_id_field == "":
            raise ValueError("MeasurementTable.object_id_field cannot be empty.")
        if self.source_image_name == "":
            raise ValueError("MeasurementTable.source_image_name cannot be empty.")
        subject = _resolve_measurement_subject(
            self.subject,
            artifact_name=self.name,
            object_name=self.object_name,
            object_id_field=self.object_id_field,
            source_image_name=self.source_image_name,
        )
        object.__setattr__(self, "subject", subject)
        if not _is_table_like(self.rows):
            raise TypeError(
                f"MeasurementTable '{self.name}' requires table-like rows, "
                f"got {type(self.rows).__name__}."
            )
        declared_layout = (
            measurement_table_row_layout_from_fields(self.fields)
            if validated_runtime_schema
            else None
        )
        normalized_rows = (
            self.rows
            if declared_layout is not None
            else normalize_measurement_table_rows(self.rows, fields=self.fields)
        )
        if normalized_rows is not self.rows:
            object.__setattr__(self, "rows", normalized_rows)
            object.__setattr__(self, "fields", ())
            declared_layout = None
        if declared_layout is None:
            measurement_table_row_layout(self.rows)

    def runtime_payload(self) -> Any:
        return self.rows

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        subject_resolver = MeasurementTableSubjectResolver(self)
        return RuntimeValueSchema(
            kind=ArtifactKind.MEASUREMENTS,
            fields=self.fields or RuntimePayloadFieldInference(payload).fields(),
            measurement_subject=self.subject,
            object_name=subject_resolver.object_name,
            source_image_name=subject_resolver.source_image_name,
            object_id_field=subject_resolver.object_id_field,
        )


@dataclass(frozen=True, slots=True)
class SpatialGridTopology:
    """Nominal object-number topology and centers for a spatial grid."""

    rows: int
    columns: int
    origin: SpatialGridOrigin
    ordering: SpatialGridOrdering
    x_spacing: float
    y_spacing: float
    x_origin: float
    y_origin: float
    x_locations: tuple[float, ...] | None = None
    y_locations: tuple[float, ...] | None = None
    spot_table: tuple[tuple[int, ...], ...] | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        rows: int,
        columns: int,
        origin: SpatialGridOrigin,
        ordering: SpatialGridOrdering,
        x_spacing: float,
        y_spacing: float,
        x_origin: float,
        y_origin: float,
    ) -> "SpatialGridTopology":
        x_locations_value = data.get("x_locations")
        y_locations_value = data.get("y_locations")
        spot_table_value = data.get("spot_table")
        return cls(
            rows=rows,
            columns=columns,
            origin=origin,
            ordering=ordering,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_origin=x_origin,
            y_origin=y_origin,
            x_locations=(
                None
                if x_locations_value is None
                else tuple(float(item) for item in x_locations_value)
            ),
            y_locations=(
                None
                if y_locations_value is None
                else tuple(float(item) for item in y_locations_value)
            ),
            spot_table=(
                None
                if spot_table_value is None
                else tuple(
                    tuple(int(item) for item in row)
                    for row in spot_table_value
                )
            ),
        )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "origin",
            coerce_enum(SpatialGridOrigin, self.origin, "SpatialGridTopology.origin"),
        )
        object.__setattr__(
            self,
            "ordering",
            coerce_enum(
                SpatialGridOrdering,
                self.ordering,
                "SpatialGridTopology.ordering",
            ),
        )
        if self.rows <= 0 or self.columns <= 0:
            raise ValueError("SpatialGridTopology dimensions must be positive.")
        if self.x_locations is None:
            object.__setattr__(
                self,
                "x_locations",
                tuple(self.x_origin + index * self.x_spacing for index in range(self.columns)),
            )
        elif len(self.x_locations) != self.columns:
            raise ValueError("SpatialGridTopology.x_locations must match columns.")
        else:
            object.__setattr__(
                self,
                "x_locations",
                tuple(float(value) for value in self.x_locations),
            )
        if self.y_locations is None:
            object.__setattr__(
                self,
                "y_locations",
                tuple(self.y_origin + index * self.y_spacing for index in range(self.rows)),
            )
        elif len(self.y_locations) != self.rows:
            raise ValueError("SpatialGridTopology.y_locations must match rows.")
        else:
            object.__setattr__(
                self,
                "y_locations",
                tuple(float(value) for value in self.y_locations),
            )
        if self.spot_table is None:
            object.__setattr__(self, "spot_table", self.derived_spot_table())
        elif len(self.spot_table) != self.rows or any(
            len(row) != self.columns for row in self.spot_table
        ):
            raise ValueError("SpatialGridTopology.spot_table must match rows x columns.")
        else:
            object.__setattr__(
                self,
                "spot_table",
                tuple(tuple(int(value) for value in row) for row in self.spot_table),
            )

    def derived_spot_table(self) -> tuple[tuple[int, ...], ...]:
        """Return the CellProfiler-compatible object-number topology."""
        object_ids = np.arange(1, self.rows * self.columns + 1, dtype=np.int32)
        if self.ordering is SpatialGridOrdering.BY_COLUMNS:
            table = object_ids.reshape(self.rows, self.columns)
        else:
            table = object_ids.reshape(self.columns, self.rows).T
        if self.origin.reverses_rows:
            table = table[::-1, :]
        if self.origin.reverses_columns:
            table = table[:, ::-1]
        return tuple(tuple(int(value) for value in row) for row in table)


@dataclass(frozen=True, slots=True, kw_only=True)
class SpatialGrid(NativeRuntimeValue):
    """Native OpenHCS rectangular spatial grid definition."""

    rows: int
    columns: int
    x_spacing: float
    y_spacing: float
    x_origin: float
    y_origin: float
    slice_index: int = 0
    total_width: float | None = None
    total_height: float | None = None
    origin: SpatialGridOrigin = SpatialGridOrigin.TOP_LEFT
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS
    x_locations: tuple[float, ...] | None = None
    y_locations: tuple[float, ...] | None = None
    spot_table: tuple[tuple[int, ...], ...] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct a spatial grid from a stored runtime value."""
        if value.kind is not ArtifactKind.SPATIAL_GRID:
            raise TypeError(
                "SpatialGrid.from_runtime_value requires a SPATIAL_GRID "
                f"runtime value, got {value.kind.value}."
            )
        if not isinstance(value.data, Mapping):
            raise TypeError(
                f"Spatial grid '{value.name}' payload must be mapping-backed, "
                f"got {type(value.data).__name__}."
            )
        return cls.from_mapping(value.name, value.data)

    @classmethod
    def from_mapping(cls, name: str, data: Mapping[str, Any]) -> Self:
        """Build a spatial grid from canonical or legacy grid field names."""
        rows = _required_int(data, "rows")
        columns = _required_int(data, "columns")
        x_spacing = _required_float(data, "x_spacing")
        y_spacing = _required_float(data, "y_spacing")
        x_origin = _required_float(
            data,
            "x_origin",
            aliases=("x_location_of_lowest_x_spot",),
        )
        y_origin = _required_float(
            data,
            "y_origin",
            aliases=("y_location_of_lowest_y_spot",),
        )
        origin = (
            SpatialGridOrigin.TOP_LEFT
            if data.get("origin") is None
            else coerce_enum(SpatialGridOrigin, data["origin"], "SpatialGrid.origin")
        )
        ordering = _optional_grid_ordering(data, "ordering")
        topology = SpatialGridTopology.from_mapping(
            data,
            rows=rows,
            columns=columns,
            origin=origin,
            ordering=ordering,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_origin=x_origin,
            y_origin=y_origin,
        )
        return cls(
            name=name,
            rows=rows,
            columns=columns,
            x_spacing=x_spacing,
            y_spacing=y_spacing,
            x_origin=x_origin,
            y_origin=y_origin,
            slice_index=_optional_int(data, "slice_index", default=0),
            total_width=_optional_float(data, "total_width"),
            total_height=_optional_float(data, "total_height"),
            origin=origin,
            ordering=ordering,
            x_locations=topology.x_locations,
            y_locations=topology.y_locations,
            spot_table=topology.spot_table,
            source_spatial_shape_yx=(
                None
                if (
                    shape := SpatialShapeYX.optional_from_mapping(
                        data,
                        "source_spatial_shape_yx",
                    )
                )
                is None
                else shape.as_tuple()
            ),
        )

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        object.__setattr__(
            self,
            "ordering",
            coerce_enum(SpatialGridOrdering, self.ordering, "SpatialGrid.ordering"),
        )
        object.__setattr__(
            self,
            "origin",
            coerce_enum(SpatialGridOrigin, self.origin, "SpatialGrid.origin"),
        )
        if self.rows <= 0:
            raise ValueError("SpatialGrid.rows must be positive.")
        if self.columns <= 0:
            raise ValueError("SpatialGrid.columns must be positive.")
        if self.x_spacing <= 0:
            raise ValueError("SpatialGrid.x_spacing must be positive.")
        if self.y_spacing <= 0:
            raise ValueError("SpatialGrid.y_spacing must be positive.")
        if self.total_width is None:
            object.__setattr__(self, "total_width", self.x_spacing * self.columns)
        if self.total_height is None:
            object.__setattr__(self, "total_height", self.y_spacing * self.rows)
        topology = SpatialGridTopology(
            rows=self.rows,
            columns=self.columns,
            origin=self.origin,
            ordering=self.ordering,
            x_spacing=self.x_spacing,
            y_spacing=self.y_spacing,
            x_origin=self.x_origin,
            y_origin=self.y_origin,
            x_locations=self.x_locations,
            y_locations=self.y_locations,
            spot_table=self.spot_table,
        )
        object.__setattr__(self, "x_locations", topology.x_locations)
        object.__setattr__(self, "y_locations", topology.y_locations)
        object.__setattr__(self, "spot_table", topology.spot_table)
        if self.source_spatial_shape_yx is not None:
            object.__setattr__(
                self,
                "source_spatial_shape_yx",
                SpatialShapeYX.from_sequence(
                    self.source_spatial_shape_yx,
                    field_name="SpatialGrid.source_spatial_shape_yx",
                ).as_tuple(),
            )

    def with_name(self, name: str) -> Self:
        """Return the same grid under a different artifact name."""
        return type(self)(
            name=name,
            rows=self.rows,
            columns=self.columns,
            x_spacing=self.x_spacing,
            y_spacing=self.y_spacing,
            x_origin=self.x_origin,
            y_origin=self.y_origin,
            slice_index=self.slice_index,
            total_width=self.total_width,
            total_height=self.total_height,
            origin=self.origin,
            ordering=self.ordering,
            x_locations=self.x_locations,
            y_locations=self.y_locations,
            spot_table=self.spot_table,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
        )

    @property
    def x_location_of_lowest_x_spot(self) -> float:
        return self.x_origin

    @property
    def y_location_of_lowest_y_spot(self) -> float:
        return self.y_origin

    def as_mapping(self) -> dict[str, Any]:
        """Return a JSON/metadata-compatible grid payload."""
        return {
            "slice_index": self.slice_index,
            "rows": self.rows,
            "columns": self.columns,
            "x_spacing": self.x_spacing,
            "y_spacing": self.y_spacing,
            "x_origin": self.x_origin,
            "y_origin": self.y_origin,
            "x_location_of_lowest_x_spot": self.x_origin,
            "y_location_of_lowest_y_spot": self.y_origin,
            "total_width": self.total_width,
            "total_height": self.total_height,
            "origin": self.origin.value,
            "ordering": self.ordering.value,
            "x_locations": self.x_locations,
            "y_locations": self.y_locations,
            "spot_table": self.spot_table,
            "source_spatial_shape_yx": self.source_spatial_shape_yx,
        }

    def runtime_payload(self) -> Any:
        return self.as_mapping()

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(kind=ArtifactKind.SPATIAL_GRID)

    def spot_table_array(self) -> np.ndarray:
        """Return the nominal grid object-number topology as a dense table."""
        return np.asarray(self.spot_table, dtype=np.int32)

    def x_locations_array(self) -> np.ndarray:
        """Return x center coordinates for grid columns."""
        return np.asarray(self.x_locations, dtype=np.float64)

    def y_locations_array(self) -> np.ndarray:
        """Return y center coordinates for grid rows."""
        return np.asarray(self.y_locations, dtype=np.float64)


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectRelationship(NativeRuntimeValue):
    """Native OpenHCS directed object relationship value."""

    source: RelationshipEndpoint
    target: RelationshipEndpoint
    source_ids: Any
    target_ids: Any
    relationship_type: str = "related"
    slice_indices: tuple[int, ...] = ()
    slice_count: int | None = None

    @classmethod
    def from_runtime_value(cls, value: RuntimeValue) -> Self:
        """Reconstruct the native relationship view from a runtime value."""
        if value.kind is not ArtifactKind.RELATIONSHIPS:
            raise TypeError(
                "ObjectRelationship.from_runtime_value requires a RELATIONSHIPS "
                f"runtime value, got {value.kind.value}."
            )
        if not isinstance(value.data, Mapping):
            raise TypeError(
                f"Relationship '{value.name}' payload must be mapping-backed, "
                f"got {type(value.data).__name__}."
            )
        relationship = value.schema.relationship
        if relationship is None:
            raise TypeError(
                f"Relationship '{value.name}' is missing typed relationship "
                "schema."
            )
        return cls(
            name=value.name,
            source=relationship.source,
            target=relationship.target,
            source_ids=value.data[relationship.source.id_field],
            target_ids=value.data[relationship.target.id_field],
            relationship_type=relationship.relationship_type,
            slice_indices=_optional_int_tuple(
                value.data,
                "slice_indices",
                aliases=("slice_index",),
            ),
            slice_count=_optional_nullable_int(value.data, "slice_count"),
        )

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        if not isinstance(self.source, RelationshipEndpoint):
            raise TypeError(
                "ObjectRelationship.source must be RelationshipEndpoint, "
                f"got {type(self.source).__name__}."
            )
        if not isinstance(self.target, RelationshipEndpoint):
            raise TypeError(
                "ObjectRelationship.target must be RelationshipEndpoint, "
                f"got {type(self.target).__name__}."
            )
        _require_name(self.relationship_type, "ObjectRelationship.relationship_type")
        _validate_relationship_ids(self.source_ids, self.target_ids, self.name)
        slice_indices = tuple(int(slice_index) for slice_index in self.slice_indices)
        if slice_indices and len(slice_indices) != len(self.source_ids):
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_indices must be empty "
                "or match source_ids/target_ids length, got "
                f"{len(slice_indices)} for {len(self.source_ids)} relationships."
            )
        if any(slice_index < 0 for slice_index in slice_indices):
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_indices cannot be negative."
            )
        slice_count = None if self.slice_count is None else int(self.slice_count)
        if slice_count is not None and slice_count < 0:
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_count cannot be negative."
            )
        if (
            slice_count is not None
            and slice_indices
            and max(slice_indices) >= slice_count
        ):
            raise ValueError(
                f"ObjectRelationship '{self.name}' slice_indices must be smaller "
                f"than slice_count {slice_count}."
            )
        object.__setattr__(self, "slice_indices", slice_indices)
        object.__setattr__(self, "slice_count", slice_count)

    @property
    def semantics(self) -> RelationshipSemantics:
        return RelationshipSemantics(
            source=self.source,
            target=self.target,
            relationship_type=self.relationship_type,
        )

    def as_table(self) -> dict[str, Any]:
        """Return table-like relationship columns for materialization."""
        table = {
            "relationship_type": self.relationship_type,
            "source_role": self.source.role,
            "target_role": self.target.role,
            "source_object": self.source.name,
            "target_object": self.target.name,
            self.source.id_field: self.source_ids,
            self.target.id_field: self.target_ids,
        }
        if self.slice_indices:
            table["slice_index"] = self.slice_indices
        if self.slice_count is not None:
            table["slice_count"] = self.slice_count
        return table

    def runtime_payload(self) -> Any:
        return self.as_table()

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(
            kind=ArtifactKind.RELATIONSHIPS,
            fields=RuntimePayloadFieldInference(payload).fields(),
            relationship=self.semantics,
        )


def normalize_artifact_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Normalize a raw function artifact return into a validated RuntimeValue."""
    if isinstance(value, RuntimeValue):
        return validate_runtime_value(value, output_plan, axis_id=axis_id)

    native_value = _normalize_native_value(output_plan, value, axis_id=axis_id)
    if native_value is not None:
        return validate_runtime_value(native_value, output_plan, axis_id=axis_id)

    runtime_value = RuntimeValue.from_output_plan(
        output_plan,
        value,
        axis_id=axis_id,
        schema=RuntimeValueSchema(kind=output_plan.kind),
    )
    return validate_runtime_value(runtime_value, output_plan, axis_id=axis_id)


def _normalize_native_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue | None:
    if _is_runtime_slice_aligned_values(value):
        return _normalize_slice_aligned_value(output_plan, value, axis_id=axis_id)
    if isinstance(value, NativeRuntimeValue):
        _validate_native_name(output_plan, value.name)
        return value.to_runtime_value(output_plan, axis_id=axis_id)
    return None


def _normalize_slice_aligned_value(
    output_plan: ArtifactOutputPlan,
    value: Any,
    *,
    axis_id: str,
) -> RuntimeValue:
    payload_strategy = RuntimeSliceAlignedPayloadNormalizationStrategy.for_output_plan(
        output_plan
    )
    if payload_strategy is not None:
        payload = payload_strategy.normalize(value)
        if payload is not None:
            return RuntimeValue.from_output_plan(
                output_plan,
                payload,
                axis_id=axis_id,
                schema=RuntimeValueSchema(kind=output_plan.kind),
            )

    slice_values: list[Any] = []
    slice_schemas: list[RuntimeValueSchema] = []
    for item in value.slices:
        if isinstance(item, NativeRuntimeValue):
            _validate_native_name(output_plan, item.name)
            runtime_item = item.to_runtime_value(output_plan, axis_id=axis_id)
            slice_values.append(runtime_item.data)
            slice_schemas.append(runtime_item.schema)
            continue
        if isinstance(item, RuntimeValue):
            slice_values.append(
                validate_runtime_value(item, output_plan, axis_id=axis_id).data
            )
            slice_schemas.append(item.schema)
            continue
        slice_values.append(item)

    schema = (
        _merge_slice_aligned_schemas(output_plan, slice_schemas)
        if slice_schemas
        else RuntimeValueSchema(kind=output_plan.kind, slice_aligned=True)
    )
    return RuntimeValue.from_output_plan(
        output_plan,
        tuple(slice_values),
        axis_id=axis_id,
        schema=schema,
    )


def _is_runtime_slice_aligned_values(value: Any) -> bool:
    return isinstance(value, RuntimeSliceAlignedValueSet)


def _merge_slice_aligned_schemas(
    output_plan: ArtifactOutputPlan,
    schemas: Sequence[RuntimeValueSchema],
) -> RuntimeValueSchema:
    first = schemas[0]
    for schema in schemas:
        if schema.kind is not output_plan.kind:
            raise ValueError(
                f"Slice-aligned artifact '{output_plan.name}' expected "
                f"{output_plan.kind.value}, got {schema.kind.value}."
            )
        if replace(schema, slice_aligned=False) != replace(first, slice_aligned=False):
            raise ValueError(
                f"Slice-aligned artifact '{output_plan.name}' has inconsistent "
                "per-slice runtime schemas."
            )
    return replace(first, slice_aligned=True)


def validate_runtime_value(
    value: RuntimeValue,
    output_plan: ArtifactOutputPlan,
    *,
    axis_id: str,
) -> RuntimeValue:
    """Validate a runtime value against the compiled output plan."""
    if value.key.name != output_plan.name:
        raise ValueError(
            f"RuntimeValue name '{value.key.name}' does not match planned "
            f"artifact '{output_plan.name}'."
        )
    if value.kind is not output_plan.kind:
        raise ValueError(
            f"Artifact '{output_plan.name}' expected {output_plan.kind.value}, "
            f"got {value.kind.value}."
        )
    if value.schema.kind is not output_plan.kind:
        raise ValueError(
            f"Artifact '{output_plan.name}' schema kind {value.schema.kind.value} "
            f"does not match planned kind {output_plan.kind.value}."
        )
    if value.key.scope.axis_id != axis_id:
        raise ValueError(
            f"Artifact '{output_plan.name}' belongs to axis "
            f"'{value.key.scope.axis_id}', not '{axis_id}'."
        )

    _validate_payload_kind(output_plan.name, value.kind, value.data, value.schema)
    return value


def _validate_payload_kind(
    name: str,
    kind: ArtifactKind,
    data: Any,
    schema: RuntimeValueSchema,
) -> None:
    ArtifactPayloadValidationStrategy.for_kind(kind).validate(name, data, schema)


class ArtifactPayloadValidationStrategy(
    EnumKeyedStrategyMixin[ArtifactKind],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Registered validation contract for artifact runtime payloads."""

    __registry_key__ = "kind_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "kind"
    __enum_label_attr__ = "kind_label"

    kind: ClassVar[ArtifactKind | None] = None
    kind_label: ClassVar[str | None] = None

    def __init__(self, kind: ArtifactKind | None = None) -> None:
        self._kind = kind

    @classmethod
    def for_kind(cls, kind: ArtifactKind) -> "ArtifactPayloadValidationStrategy":
        strategy_type = cls.__registry__.get(kind.value)
        strategy = strategy_type() if strategy_type is not None else None
        return (
            strategy
            if strategy is not None
            else GenericArtifactPayloadValidationStrategy(kind)
        )

    @property
    def artifact_kind(self) -> ArtifactKind:
        kind = self._kind or self.kind
        if kind is None:
            raise TypeError(
                f"{type(self).__name__} must declare an artifact kind or be "
                "constructed with one."
            )
        return kind

    def validate(
        self,
        name: str,
        data: Any,
        schema: RuntimeValueSchema,
    ) -> None:
        validator = self._validator(schema)
        if validator is None:
            return
        if schema.slice_aligned:
            if _is_slice_aligned_payload(data) and all(validator(item) for item in data):
                return
            raise TypeError(
                f"Artifact '{name}' expected slice-aligned "
                f"{self.artifact_kind.payload_description}, got {type(data).__name__}."
            )
        if validator(data):
            return
        raise TypeError(
            f"Artifact '{name}' expected {self.artifact_kind.payload_description}, "
            f"got {type(data).__name__}."
        )

    def _validator(
        self,
        schema: RuntimeValueSchema,
    ) -> Callable[[Any], bool] | None:
        return _PAYLOAD_VALIDATORS[_payload_shape_for(self.artifact_kind, schema)]


class GenericArtifactPayloadValidationStrategy(ArtifactPayloadValidationStrategy):
    """Validate artifacts by their declared payload shape."""

    kind = ArtifactKind.SPECIAL


class ObjectLabelsArtifactPayloadValidationStrategy(ArtifactPayloadValidationStrategy):
    """Validate object-label artifacts by nominal label payload semantics."""

    kind = ArtifactKind.OBJECT_LABELS

    def validate(
        self,
        name: str,
        data: Any,
        schema: RuntimeValueSchema,
    ) -> None:
        if isinstance(data, (ObjectLabelPayload, ObjectLabelSet)):
            return
        super().validate(name, data, schema)


def _payload_shape_for(
    kind: ArtifactKind,
    schema: RuntimeValueSchema,
) -> ArtifactPayloadShape:
    if kind.uses_label_representation_payload_shape:
        representation = (
            schema.label_representation or ObjectLabelRepresentation.DENSE_LABELS
        )
        return representation.payload_shape
    return kind.payload_shape


def _is_table_like(data: Any) -> bool:
    _ensure_runtime_payload_integrations_registered()
    return (
        isinstance(data, ColumnarRows)
        or isinstance(data, Mapping)
        or (
            isinstance(data, Sequence)
            and not isinstance(data, (str, bytes, bytearray))
        )
    )


def _is_array_like(data: Any) -> bool:
    _ensure_runtime_payload_integrations_registered()
    return isinstance(data, RuntimeArrayPayload) or any(
        predicate(data)
        for predicate in _ARRAY_PAYLOAD_PREDICATES
    )


def _is_mapping_like(data: Any) -> bool:
    return isinstance(data, Mapping)


def _is_slice_aligned_payload(data: Any) -> bool:
    return isinstance(data, Sequence) and not isinstance(
        data,
        (str, bytes, bytearray),
    )


def _validate_object_label_variant(
    object_name: str,
    variant_name: str,
    labels: Any,
    variant_labels: Any | None,
    validator: Callable[[Any], bool] | None,
) -> None:
    if variant_labels is None:
        return
    if validator is not None and not validator(variant_labels):
        raise TypeError(
            f"ObjectLabelSet '{object_name}' {variant_name} requires an "
            f"array-compatible payload, got {type(variant_labels).__name__}."
        )
    labels_shape = tuple(np.shape(labels))
    variant_labels_shape = tuple(np.shape(variant_labels))
    if labels_shape and variant_labels_shape and labels_shape != variant_labels_shape:
        raise ValueError(
            f"ObjectLabelSet '{object_name}' {variant_name} shape "
            f"{variant_labels_shape!r} does not match final labels "
            f"shape {labels_shape!r}."
        )


@dataclass(frozen=True, slots=True)
class _PayloadValidator:
    shape: ArtifactPayloadShape
    predicate: Callable[[Any], bool] | None


def _payload_validators(
    rows: tuple[_PayloadValidator, ...],
) -> Mapping[ArtifactPayloadShape, Callable[[Any], bool] | None]:
    validators = {row.shape: row.predicate for row in rows}
    if set(validators) != set(ArtifactPayloadShape):
        raise TypeError("Incomplete runtime payload validator table.")
    return MappingProxyType(validators)


_PAYLOAD_VALIDATORS = _payload_validators(
    (
        _PayloadValidator(ArtifactPayloadShape.ANY, None),
        _PayloadValidator(ArtifactPayloadShape.ARRAY, _is_array_like),
        _PayloadValidator(ArtifactPayloadShape.TABLE, _is_table_like),
        _PayloadValidator(ArtifactPayloadShape.MAPPING, _is_mapping_like),
    )
)


@dataclass(frozen=True, slots=True)
class RequiredMappingField:
    """Resolve a required canonical mapping field with legacy aliases."""

    data: Mapping[str, Any]
    name: str
    aliases: tuple[str, ...] = ()

    def value(self) -> Any:
        for key in (self.name, *self.aliases):
            if key in self.data:
                return self.data[key]
        names = ", ".join(repr(key) for key in (self.name, *self.aliases))
        raise KeyError(f"Missing required mapping field {names}.")


def _required_int(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> int:
    return int(RequiredMappingField(data, name, aliases).value())


def _required_float(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> float:
    return float(RequiredMappingField(data, name, aliases).value())


def _optional_int(
    data: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    if name not in data:
        return default
    return int(data[name])


def _optional_nullable_int(data: Mapping[str, Any], name: str) -> int | None:
    if name not in data or data[name] is None:
        return None
    return int(data[name])


def _optional_int_tuple(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> tuple[int, ...]:
    for key in (name, *aliases):
        if key not in data or data[key] is None:
            continue
        value = data[key]
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            return tuple(int(item) for item in value)
        raise TypeError(
            f"Optional integer tuple field '{key}' must be a sequence, "
            f"got {type(value).__name__}."
        )
    return ()


def _optional_float(
    data: Mapping[str, Any],
    name: str,
) -> float | None:
    if name not in data or data[name] is None:
        return None
    return float(data[name])


def _optional_grid_ordering(
    data: Mapping[str, Any],
    name: str,
) -> SpatialGridOrdering:
    if name not in data or data[name] is None:
        return SpatialGridOrdering.BY_ROWS
    return coerce_enum(SpatialGridOrdering, data[name], f"SpatialGrid.{name}")


def _require_name(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} cannot be empty.")


def _validate_native_name(output_plan: ArtifactOutputPlan, name: str) -> None:
    if name != output_plan.name:
        raise ValueError(
            f"Native runtime value '{name}' does not match planned artifact "
            f"'{output_plan.name}'."
        )


def _resolve_measurement_subject(
    subject: MeasurementSubject | None,
    *,
    artifact_name: str,
    object_name: str | None,
    object_id_field: str | None,
    source_image_name: str | None,
) -> MeasurementSubject:
    if subject is None:
        if object_name is not None:
            return MeasurementSubject(
                MeasurementScope.OBJECT,
                object_name,
                object_id_field,
            )
        if source_image_name is not None:
            return MeasurementSubject(MeasurementScope.IMAGE, source_image_name)
        return MeasurementSubject(MeasurementScope.ARTIFACT, artifact_name)

    if object_name is not None and (
        subject.scope is not MeasurementScope.OBJECT or subject.name != object_name
    ):
        raise ValueError(
            "MeasurementTable.object_name conflicts with "
            "MeasurementTable.subject."
        )
    if object_id_field is not None and subject.id_field != object_id_field:
        raise ValueError(
            "MeasurementTable.object_id_field conflicts with "
            "MeasurementTable.subject."
        )
    if (
        source_image_name is not None
        and subject.scope is MeasurementScope.IMAGE
        and subject.name != source_image_name
    ):
        raise ValueError(
            "MeasurementTable.source_image_name conflicts with "
            "MeasurementTable.subject."
        )
    return subject


@dataclass(frozen=True, slots=True)
class MeasurementTableSubjectResolver:
    """Resolve legacy measurement-table fields against the nominal subject."""

    table: MeasurementTable

    @property
    def object_name(self) -> str | None:
        if self.table.object_name is not None:
            return self.table.object_name
        if self.table.subject and self.table.subject.scope is MeasurementScope.OBJECT:
            return self.table.subject.name
        return None

    @property
    def object_id_field(self) -> str | None:
        if self.table.object_id_field is not None:
            return self.table.object_id_field
        if self.table.subject and self.table.subject.scope is MeasurementScope.OBJECT:
            return self.table.subject.id_field
        return None

    @property
    def source_image_name(self) -> str | None:
        if self.table.source_image_name is not None:
            return self.table.source_image_name
        if self.table.subject and self.table.subject.scope is MeasurementScope.IMAGE:
            return self.table.subject.source_image_name
        return None


@dataclass(frozen=True, slots=True)
class RuntimePayloadFieldInference:
    """Infer tabular field declarations from runtime payload row shapes."""

    rows: Any

    def fields(self) -> tuple[FieldSpec, ...]:
        _ensure_runtime_payload_integrations_registered()
        if isinstance(self.rows, ColumnarRows):
            return tuple(FieldSpec(str(column)) for column in self.rows.columns)
        if isinstance(self.rows, Mapping):
            return tuple(FieldSpec(str(column)) for column in self.rows)
        if (
            isinstance(self.rows, Sequence)
            and self.rows
            and isinstance(self.rows[0], Mapping)
        ):
            return tuple(FieldSpec(str(column)) for column in self.rows[0])
        return ()


def _ensure_runtime_payload_integrations_registered() -> None:
    """Load optional external payload capability registrations."""
    from openhcs.core.runtime_payload_integrations import (
        register_runtime_payload_integrations,
    )

    register_runtime_payload_integrations()


def _validate_relationship_ids(source_ids: Any, target_ids: Any, name: str) -> None:
    if isinstance(source_ids, Sequence) and isinstance(target_ids, Sequence):
        if (
            not isinstance(source_ids, (str, bytes, bytearray))
            and not isinstance(target_ids, (str, bytes, bytearray))
            and len(source_ids) != len(target_ids)
        ):
            raise ValueError(
                f"ObjectRelationship '{name}' source_ids and target_ids must "
                f"have equal length, got {len(source_ids)} and {len(target_ids)}."
            )
