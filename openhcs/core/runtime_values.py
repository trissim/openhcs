"""Typed runtime artifact values and validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self, TypeVar

from openhcs.constants.constants import Backend
from openhcs.core.artifacts import (
    ArtifactKey,
    ArtifactKind,
    ArtifactOutputPlan,
    ArtifactPayloadShape,
)
from openhcs.core.runtime_semantics import (
    FieldSpec,
    MeasurementScope,
    MeasurementSubject,
    ObjectLabelDomain,
    ObjectLabelDomainMetadata,
    ObjectLabelRepresentation,
    ObjectLabelVariant,
    RelationshipEndpoint,
    RelationshipSemantics,
    SpatialGridOrdering,
    coerce_enum,
)


_TPayload = TypeVar("_TPayload", bound=type[Any])
_ARRAY_PAYLOAD_PREDICATES: list[Callable[[Any], bool]] = []
logger = logging.getLogger(__name__)


class RuntimeArrayPayload(ABC):
    """Nominal ABC for array payload types accepted by runtime artifacts."""

    @property
    @abstractmethod
    def shape(self) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class ImagePayloadMetadata:
    """Generic source-image metadata that should travel with runtime pixels."""

    intensity_scale: float | None = None
    source_dtype: str | None = None
    source_path: str | None = None
    channel_intensity_scales: tuple[float | None, ...] = ()
    channel_source_dtypes: tuple[str | None, ...] = ()
    channel_source_paths: tuple[str | None, ...] = ()
    spatial_origin_yx: tuple[int, int] | None = None
    source_spatial_shape_yx: tuple[int, int] | None = None
    physical_border_edges_yx: tuple[bool, bool, bool, bool] | None = None
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

    @property
    def has_values(self) -> bool:
        """Return whether this metadata carries any semantic image facts."""
        return any(
            (
                self.intensity_scale is not None,
                self.source_dtype is not None,
                self.source_path is not None,
                bool(self.channel_intensity_scales),
                bool(self.channel_source_dtypes),
                bool(self.channel_source_paths),
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

    def for_channel(self, channel_index: int) -> "ImagePayloadMetadata":
        """Return single-channel metadata sliced from a stacked payload."""
        return ImagePayloadMetadata(
            intensity_scale=self.intensity_scale_for_channel(channel_index),
            source_dtype=_tuple_value(self.channel_source_dtypes, channel_index)
            or self.source_dtype,
            source_path=_tuple_value(self.channel_source_paths, channel_index)
            or self.source_path,
            spatial_origin_yx=self.spatial_origin_yx,
            source_spatial_shape_yx=self.source_spatial_shape_yx,
            physical_border_edges_yx=self.physical_border_edges_yx,
            mask_defines_border=self.mask_defines_border,
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
        physical_border_edges_yx: tuple[bool, bool, bool, bool] | None = None,
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
        if not hasattr(self.data, "shape") or not hasattr(self.data, "ndim"):
            raise TypeError(
                "ImageMetadataPayload.data requires array-like data with shape/ndim, "
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
        if not hasattr(self.data, "shape") or not hasattr(self.data, "ndim"):
            raise TypeError(
                "MaskedImagePayload.data requires array-like data with shape/ndim, "
                f"got {type(self.data).__name__}."
            )
        if not hasattr(self.mask, "shape"):
            raise TypeError(
                "MaskedImagePayload.mask requires array-like data with shape, "
                f"got {type(self.mask).__name__}."
            )
        data_shape = tuple(self.data.shape)
        mask_shape = tuple(self.mask.shape)
        if mask_shape not in _valid_image_mask_shapes(data_shape):
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


def with_image_payload_data(
    payload: Any,
    data: Any,
    *,
    mask: Any | None = None,
    metadata: ImagePayloadMetadata | None = None,
) -> Any:
    """Preserve image-mask and metadata semantics while replacing pixels."""
    resolved_mask = image_payload_mask(payload) if mask is None else mask
    resolved_metadata = (
        image_payload_metadata(payload) if metadata is None else metadata
    )
    return image_payload_with_context(
        data,
        mask=resolved_mask,
        metadata=resolved_metadata,
    )


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


def _valid_image_mask_shapes(data_shape: tuple[int, ...]) -> frozenset[tuple[int, ...]]:
    """Return accepted mask domains for common grayscale/color image layouts."""
    valid: set[tuple[int, ...]] = {data_shape}
    if len(data_shape) >= 2:
        valid.add(data_shape[:2])
    if len(data_shape) == 3:
        valid.add(data_shape[1:])
    if len(data_shape) >= 4:
        valid.add(data_shape[:3])
        valid.add(data_shape[1:3])
    return frozenset(valid)


@dataclass(frozen=True, slots=True)
class ObjectLabelPayload(RuntimeArrayPayload, ObjectLabelDomainMetadata):
    """Dense object labels plus optional semantic label variants."""

    labels: Any
    unedited_labels: Any | None = None
    small_removed_labels: Any | None = None
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()

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

    def object_label_domain(self) -> ObjectLabelDomain:
        return ObjectLabelDomain(
            declared_object_count=self.declared_object_count,
            declared_object_ids=self.declared_object_ids,
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


class ColumnarRows(ABC):
    """Nominal ABC for table payloads exposing named columns."""

    @property
    @abstractmethod
    def columns(self) -> Any:
        ...


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
class ObjectLabelSet(SourceImageRuntimeValue):
    """Native OpenHCS object-label value."""

    labels: Any
    unedited_labels: Any | None = None
    small_removed_labels: Any | None = None
    representation: ObjectLabelRepresentation = ObjectLabelRepresentation.DENSE_LABELS
    declared_object_count: int | None = None
    declared_object_ids: tuple[int, ...] = ()

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
        if self.declared_object_count is not None:
            count = int(self.declared_object_count)
            if count < 0:
                raise ValueError("ObjectLabelSet.declared_object_count cannot be negative.")
            object.__setattr__(self, "declared_object_count", count)
        ids = tuple(int(object_id) for object_id in self.declared_object_ids)
        if any(object_id <= 0 for object_id in ids):
            raise ValueError("ObjectLabelSet.declared_object_ids must be positive.")
        object.__setattr__(self, "declared_object_ids", tuple(sorted(dict.fromkeys(ids))))
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
        ):
            return ObjectLabelPayload(
                labels=self.labels,
                unedited_labels=self.unedited_labels,
                small_removed_labels=self.small_removed_labels,
                declared_object_count=self.declared_object_count,
                declared_object_ids=self.declared_object_ids,
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


@dataclass(frozen=True, slots=True, kw_only=True)
class MeasurementTable(NativeRuntimeValue):
    """Native OpenHCS measurement table value."""

    rows: Any
    object_name: str | None = None
    fields: tuple[FieldSpec, ...] = ()
    object_id_field: str | None = None
    source_image_name: str | None = None
    subject: MeasurementSubject | None = None

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
        )

    def __post_init__(self) -> None:
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

    def runtime_payload(self) -> Any:
        return self.rows

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(
            kind=ArtifactKind.MEASUREMENTS,
            fields=self.fields or _infer_fields(payload),
            measurement_subject=self.subject,
            object_name=_measurement_object_name(self),
            source_image_name=_measurement_source_image_name(self),
            object_id_field=_measurement_object_id_field(self),
        )


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
    ordering: SpatialGridOrdering = SpatialGridOrdering.BY_ROWS

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
        return cls(
            name=name,
            rows=_required_int(data, "rows"),
            columns=_required_int(data, "columns"),
            x_spacing=_required_float(data, "x_spacing"),
            y_spacing=_required_float(data, "y_spacing"),
            x_origin=_required_float(
                data,
                "x_origin",
                aliases=("x_location_of_lowest_x_spot",),
            ),
            y_origin=_required_float(
                data,
                "y_origin",
                aliases=("y_location_of_lowest_y_spot",),
            ),
            slice_index=_optional_int(data, "slice_index", default=0),
            total_width=_optional_float(data, "total_width"),
            total_height=_optional_float(data, "total_height"),
            ordering=_optional_grid_ordering(data, "ordering"),
        )

    def __post_init__(self) -> None:
        NativeRuntimeValue.__post_init__(self)
        object.__setattr__(
            self,
            "ordering",
            coerce_enum(SpatialGridOrdering, self.ordering, "SpatialGrid.ordering"),
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
            ordering=self.ordering,
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
            "ordering": self.ordering.value,
        }

    def runtime_payload(self) -> Any:
        return self.as_mapping()

    def runtime_schema(self, payload: Any) -> RuntimeValueSchema:
        return RuntimeValueSchema(kind=ArtifactKind.SPATIAL_GRID)


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
            fields=_infer_fields(payload),
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
    if isinstance(value, NativeRuntimeValue):
        _validate_native_name(output_plan, value.name)
        return value.to_runtime_value(output_plan, axis_id=axis_id)
    return None


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
    payload_shape = _payload_shape_for(kind, schema)
    validator = _PAYLOAD_VALIDATORS[payload_shape]
    if validator is None:
        return
    if validator(data):
        return
    raise TypeError(
        f"Artifact '{name}' expected {kind.payload_description}, "
        f"got {type(data).__name__}."
    )


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
    if (
        hasattr(labels, "shape")
        and hasattr(variant_labels, "shape")
        and tuple(labels.shape) != tuple(variant_labels.shape)
    ):
        raise ValueError(
            f"ObjectLabelSet '{object_name}' {variant_name} shape "
            f"{tuple(variant_labels.shape)!r} does not match final labels "
            f"shape {tuple(labels.shape)!r}."
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


def _required_mapping_value(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> Any:
    for key in (name, *aliases):
        if key in data:
            return data[key]
    names = ", ".join(repr(key) for key in (name, *aliases))
    raise KeyError(f"Missing required mapping field {names}.")


def _required_int(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> int:
    return int(_required_mapping_value(data, name, aliases=aliases))


def _required_float(
    data: Mapping[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> float:
    return float(_required_mapping_value(data, name, aliases=aliases))


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


def _measurement_object_name(value: MeasurementTable) -> str | None:
    if value.object_name is not None:
        return value.object_name
    if value.subject and value.subject.scope is MeasurementScope.OBJECT:
        return value.subject.name
    return None


def _measurement_object_id_field(value: MeasurementTable) -> str | None:
    if value.object_id_field is not None:
        return value.object_id_field
    if value.subject and value.subject.scope is MeasurementScope.OBJECT:
        return value.subject.id_field
    return None


def _measurement_source_image_name(value: MeasurementTable) -> str | None:
    if value.source_image_name is not None:
        return value.source_image_name
    if value.subject and value.subject.scope is MeasurementScope.IMAGE:
        return value.subject.name
    return None


def _infer_fields(rows: Any) -> tuple[FieldSpec, ...]:
    _ensure_runtime_payload_integrations_registered()
    if isinstance(rows, ColumnarRows):
        return tuple(FieldSpec(str(column)) for column in rows.columns)
    if isinstance(rows, Mapping):
        return tuple(FieldSpec(str(column)) for column in rows)
    if (
        isinstance(rows, Sequence)
        and rows
        and isinstance(rows[0], Mapping)
    ):
        return tuple(FieldSpec(str(column)) for column in rows[0])
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
