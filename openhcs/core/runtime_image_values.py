"""Nominal runtime image payload values and metadata."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
    Iterable,
    Sequence,
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeVar

import numpy as np

from openhcs.core.alias_property import AliasProperty
from openhcs.core.runtime_array_values import (
    DataBackedRuntimeArrayPayload,
    RuntimeArrayData,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisProjector,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.source_image_provenance import (
    RuntimeSourceImageProvenancePlane,
    SourceComponentMetadata,
    SourceImageIdentity,
    SourceImageProvenance,
    SourceImageProvenanceFields,
    SourceImageProvenancePlanes,
    SourcePlaneIndexedProvenanceExpansion,
    common_source_component_metadata,
)
from openhcs.core.source_metadata import (
    SourceVoxelSpacing,
    SourceVoxelSpacingFields,
    SourceMetadataValue,
)
from openhcs.core.source_spatial_domain import (
    SourceSpatialDomain,
    SourceSpatialDomainFields,
)
from openhcs.core.source_spatial_domain import (
    _spatial_shape_pair as _source_spatial_shape_pair,
)

PhysicalBorderEdgesYX = tuple[bool, bool, bool, bool] | None

OBJECT_LABEL_SOURCE_SPATIAL_VALUE_NAME = "Object-label"

MetadataValueT = TypeVar("MetadataValueT")


@dataclass(frozen=True, slots=True)
class ImageUnitIntervalIntensityMetadata:
    """Authored proof that image pixels retain exact unit-interval quantization."""

    scale: int | None = None
    source_plane_scales: tuple[int | None, ...] = ()

    def scale_for_source_plane(self, plane_index: int) -> int | None:
        """Return one plane's proof, falling back to the scalar proof."""

        plane_scale = _tuple_value(self.source_plane_scales, plane_index)
        return self.scale if plane_scale is None else int(plane_scale)

    def for_source_plane(
        self, plane_index: int
    ) -> "ImageUnitIntervalIntensityMetadata":
        """Project this proof to one source plane."""

        return type(self)(scale=self.scale_for_source_plane(plane_index))

    def for_source_planes(
        self,
        plane_indices: tuple[int, ...],
    ) -> "ImageUnitIntervalIntensityMetadata":
        """Project per-plane proofs to an ordered source-plane subset."""

        return type(self)(
            scale=self.scale,
            source_plane_scales=_tuple_values_at_indices(
                self.source_plane_scales,
                plane_indices,
            ),
        )

    def without_source_planes(self) -> "ImageUnitIntervalIntensityMetadata":
        """Discard per-plane proofs after removing their represented axis."""

        return type(self)(scale=self.scale)


@dataclass(slots=True)
class ImagePayloadMetadata(
    SourceImageProvenanceFields,
    SourceSpatialDomainFields,
    SourceVoxelSpacingFields,
):
    """Generic source-image metadata that should travel with runtime pixels."""

    intensity_scale: float | None = None
    source_dtype: str | None = None
    unit_interval_intensity: ImageUnitIntervalIntensityMetadata | None = None
    source_plane_intensity_scales: tuple[float | None, ...] = ()
    source_plane_dtypes: tuple[str | None, ...] = ()
    physical_border_edges_yx: PhysicalBorderEdgesYX = None
    mask_defines_border: bool | None = None
    source_channel_axis: int | None = None
    plane_axis: RuntimePlaneAxis | None = None

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
        dtype = array.dtype
        if memory_type == MEMORY_TYPE_NUMPY:
            intensity_scale = image_intensity_scale_for_dtype(dtype)
        else:
            intensity_scale = None
        return cls(
            intensity_scale=intensity_scale,
            source_dtype=str(dtype),
            source_path=source_path,
        )

    def __post_init__(self, *source_provenance_values: object) -> None:
        self.absorb_explicit_source_provenance(
            SourceImageProvenance.from_init_values(source_provenance_values)
        )
        self.source_voxel_spacing = self.source_voxel_spacing.with_missing_from(
            SourceVoxelSpacing.from_source_metadata(self.source_component_metadata)
        )
        self.normalize_source_provenance_fields()
        self.normalize_source_spatial_domain_fields()
        self.normalize_source_voxel_spacing_fields()
        if self.source_channel_axis is not None and (
            not isinstance(self.source_channel_axis, int)
            or isinstance(self.source_channel_axis, bool)
        ):
            raise TypeError(
                "ImagePayloadMetadata.source_channel_axis must be int or None."
            )
        if self.plane_axis is not None:
            self.plane_axis = RuntimePlaneAxis(
                self.plane_axis,
            )

    @property
    def has_values(self) -> bool:
        """Return whether this metadata carries any semantic image facts."""
        return any(
            (
                self.intensity_scale is not None,
                self.source_dtype is not None,
                self.source_path is not None,
                self.source_component_metadata is not None,
                self.unit_interval_intensity is not None,
                bool(self.source_plane_intensity_scales),
                bool(self.source_plane_dtypes),
                self.source_image_provenance_planes.has_values,
                self.source_spatial_domain.has_values,
                self.source_voxel_spacing.has_values,
                self.physical_border_edges_yx is not None,
                self.mask_defines_border is not None,
                bool(self.source_image_names),
                self.source_channel_axis is not None,
                self.plane_axis is not None,
            )
        )

    @property
    def unit_interval_intensity_scale(self) -> int | None:
        """Return the authored scalar unit-interval quantization proof."""

        if self.unit_interval_intensity is None:
            return None
        return self.unit_interval_intensity.scale

    @property
    def source_plane_unit_interval_intensity_scales(
        self,
    ) -> tuple[int | None, ...]:
        """Return authored per-plane unit-interval quantization proofs."""

        if self.unit_interval_intensity is None:
            return ()
        return self.unit_interval_intensity.source_plane_scales

    def attach_to(self, payload: Any) -> RuntimeArrayData:
        """Attach this metadata to an existing image payload."""
        return self.payload_with(
            image_payload_data(payload), image_payload_mask(payload)
        )

    def attach_source_context_to(self, payload: Any) -> RuntimeArrayData:
        """Attach source context while retaining the payload's declared array axes."""

        payload_metadata = image_payload_metadata(payload)
        return self.replace_fields(
            source_channel_axis=payload_metadata.source_channel_axis,
            plane_axis=payload_metadata.plane_axis,
        ).payload_with(
            image_payload_data(payload),
            image_payload_mask(payload),
        )

    def derive_payload(
        self,
        source_payload: RuntimeArrayData | None,
        data: RuntimeArrayData,
        *,
        plane_projection: RuntimePlaneAxisValueProjection | None = None,
    ) -> RuntimeArrayData:
        """Project this source metadata and mask onto derived image data."""
        source_shape_yx = self.spatial_shape_yx(source_payload)
        output_shape_yx = image_payload_metadata(data).spatial_shape_yx(data)
        same_spatial_domain = (
            source_shape_yx is not None
            and output_shape_yx is not None
            and source_shape_yx == output_shape_yx
        )
        output_metadata = image_payload_metadata(data)
        metadata = self._derived_metadata(data, plane_projection)
        output_declares_spatial_domain = (
            isinstance(data, ImagePayloadMetadataCarrier)
            and output_metadata.source_spatial_domain.has_values
        )
        if not same_spatial_domain and not output_declares_spatial_domain:
            metadata = metadata.without_spatial_domain()
        output_mask = image_payload_mask(data)
        if output_mask is None and same_spatial_domain:
            output_mask = project_image_mask_to_data_domain(
                image_payload_mask(source_payload),
                image_payload_data(data),
                metadata=metadata,
            )
        return metadata.payload_with(image_payload_data(data), output_mask)

    def _derived_metadata(
        self,
        data: RuntimeArrayData,
        plane_projection: RuntimePlaneAxisValueProjection | None,
    ) -> "ImagePayloadMetadata":
        output_metadata = image_payload_metadata(
            data
        ).with_indexed_source_plane_provenance(None)
        source_metadata = self.with_indexed_source_plane_provenance(None)
        output_metadata_is_authoritative = (
            isinstance(data, ImagePayloadMetadataCarrier)
            and output_metadata.has_values
        )
        declared_output_axis = (
            output_metadata.plane_axis
            if output_metadata_is_authoritative
            else source_metadata.plane_axis
        )
        if plane_projection is None and (
            declared_output_axis is not None
            or source_metadata.plane_axis is not None
        ):
            declared_axis = declared_output_axis or source_metadata.plane_axis
            raise ValueError(
                "Derived image payload carries a declared plane axis but the "
                "invocation supplied no plane projection: "
                f"{declared_axis.value!r}."
            )
        if plane_projection is not None and plane_projection.plane_index is None:
            for owner_name, metadata in (
                ("source", source_metadata),
                ("output", output_metadata),
            ):
                if (
                    metadata.plane_axis is not None
                    and metadata.plane_axis is not plane_projection.axis
                ):
                    raise ValueError(
                        f"Derived image {owner_name} metadata axis conflicts with "
                        "the invocation projection: "
                        f"{metadata.plane_axis.value!r} != "
                        f"{plane_projection.axis.value!r}."
                    )
            source_metadata = source_metadata.with_indexed_source_plane_provenance(
                plane_projection.axis_size
            )
            if declared_output_axis is not None:
                plane_projection.validate_shape(
                    np.asarray(image_payload_data(data)).shape,
                    value_name="Derived image payload",
                )
                output_metadata = (
                    output_metadata.with_indexed_source_plane_provenance(
                        plane_projection.axis_size
                    )
                )
        elif plane_projection is not None:
            for owner_name, metadata in (
                ("source", source_metadata),
                ("output", output_metadata),
            ):
                if metadata.plane_axis is not None:
                    raise ValueError(
                        f"Derived image {owner_name} metadata retains an unconsumed "
                        f"{metadata.plane_axis.value!r} axis after the invocation "
                        "selected one plane."
                    )
        source_context = source_metadata.replace_fields(plane_axis=None)
        if isinstance(data, ImagePayloadMetadataCarrier):
            source_context = source_context.replace_fields(source_channel_axis=None)
        metadata = output_metadata.with_source_context_from(source_context)
        if source_metadata.source_provenance.has_values:
            metadata = metadata.with_source_provenance(
                source_metadata.source_provenance.with_derived_source_image_names(
                    output_metadata.source_image_names
                    or source_metadata.source_image_names
                )
            )
        return metadata.with_missing_intensity_from(source_metadata).replace_fields(
            plane_axis=declared_output_axis,
        )

    def project_channel_payload(
        self,
        source_payload: RuntimeArrayData,
        source_data: Any,
        channel_index: int,
        *,
        channel_data: Any | None = None,
        channel_axis: int = 0,
    ) -> RuntimeArrayData:
        """Project one channel while preserving metadata and mask semantics."""
        if channel_data is None:
            channel_data = self._channel_axis_slice(
                source_data,
                channel_axis=channel_axis,
                channel_index=channel_index,
            )
        source_channel_axis = self.normalized_source_channel_axis(source_data)
        projected_axis = channel_axis % int(np.ndim(source_data))
        metadata = (
            self.for_source_plane(channel_index)
            if self.has_plane_specific_values
            else self
        )
        if source_channel_axis == projected_axis:
            metadata = metadata.without_source_channel_axis()
        mask = image_payload_mask(source_payload)
        if mask is not None:
            mask = self._projected_channel_mask(
                mask,
                source_data=source_data,
                channel_data=channel_data,
                channel_index=channel_index,
                channel_axis=channel_axis,
            )
        return metadata.payload_with(channel_data, mask)

    @staticmethod
    def _channel_axis_slice(
        value: Any,
        *,
        channel_axis: int,
        channel_index: int,
    ) -> Any:
        array = np.asarray(value)
        normalized_axis = channel_axis % array.ndim
        slices = [slice(None)] * array.ndim
        slices[normalized_axis] = slice(channel_index, channel_index + 1)
        return value[tuple(slices)]

    @classmethod
    def _projected_channel_mask(
        cls,
        mask: Any,
        *,
        source_data: Any,
        channel_data: Any,
        channel_index: int,
        channel_axis: int,
    ) -> Any:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != np.asarray(source_data).shape:
            return mask_array
        channel_mask = cls._channel_axis_slice(
            mask_array,
            channel_axis=channel_axis,
            channel_index=channel_index,
        )
        if np.shape(channel_mask) == np.shape(channel_data):
            return channel_mask
        squeezed_mask = np.squeeze(
            channel_mask,
            axis=channel_axis % channel_mask.ndim,
        )
        if np.shape(squeezed_mask) == np.shape(channel_data):
            return squeezed_mask
        return channel_mask

    def has_complete_source_identity(
        self,
        payload: RuntimeArrayData,
        plane_projection: RuntimePlaneAxisValueProjection | None = None,
    ) -> bool:
        """Validate source identity against declared runtime-plane semantics."""
        planes = self.source_image_provenance_planes
        if (
            plane_projection is not None
            and plane_projection.plane_index is None
            and plane_projection.axis_size > 1
            and self.plane_axis is None
        ):
            return False
        if plane_projection is not None and self.plane_axis is not None:
            if self.plane_axis is not plane_projection.axis:
                raise ValueError(
                    "Image payload plane declaration conflicts with the execution "
                    f"projection: {self.plane_axis!r} != {plane_projection.axis!r}."
                )
            plane_projection.validate_shape(
                np.asarray(image_payload_data(payload)).shape,
                value_name="Source-identified image payload",
            )
            if planes.count != plane_projection.axis_size:
                return False
        if planes.has_values:
            return all(plane.addressable for plane in planes.planes)
        return self.source_provenance.addressable

    @classmethod
    def compose(
        cls,
        payloads: Sequence[RuntimeArrayData],
        *,
        mode: "ImagePayloadMetadataCompositionMode | None" = None,
        source_metadata: Sequence["ImagePayloadMetadata"] | None = None,
    ) -> "ImagePayloadMetadata":
        """Compose metadata for payloads assembled on a new leading axis."""
        resolved_mode = (
            ImagePayloadMetadataCompositionMode.STACK if mode is None else mode
        )
        return _ImagePayloadMetadataComposer(
            payloads=tuple(payloads),
            mode=resolved_mode,
            source_metadata_override=source_metadata,
            metadata_type=cls,
        ).compose()

    def normalized_source_channel_axis(self, data: Any) -> int | None:
        """Return this declared channel axis normalized for ``data``."""
        if self.source_channel_axis is None:
            return None
        ndim = int(np.ndim(image_payload_data(data)))
        axis = self.source_channel_axis
        normalized = axis if axis >= 0 else ndim + axis
        if normalized < 0 or normalized >= ndim:
            raise ValueError(
                f"Source channel axis {axis} is invalid for payload rank {ndim}."
            )
        return normalized

    def spatial_axes_yx(self, data: Any) -> tuple[int, int] | None:
        """Return Y/X axes after excluding the declared channel axis."""
        ndim = int(np.ndim(image_payload_data(data)))
        channel_axis = self.normalized_source_channel_axis(data)
        candidate_axes = tuple(axis for axis in range(ndim) if axis != channel_axis)
        if len(candidate_axes) < 2:
            return None
        return candidate_axes[-2], candidate_axes[-1]

    def is_declared_source_channel_plane(self, data: Any) -> bool:
        """Return whether this payload declares one channel-bearing image plane."""
        if self.normalized_source_channel_axis(data) is None:
            return False
        return self.plane_axis is None

    def is_declared_source_channel_stack(self, data: Any) -> bool:
        """Return whether this payload declares a plane stack with a channel axis."""
        if self.normalized_source_channel_axis(data) is None:
            return False
        return self.plane_axis is not None

    def spatial_shape_yx(self, data: Any) -> tuple[int, int] | None:
        """Return Y/X shape using only declared channel-axis semantics."""
        axes = self.spatial_axes_yx(data)
        if axes is None:
            return None
        shape = tuple(
            int(axis_size) for axis_size in np.shape(image_payload_data(data))
        )
        return shape[axes[0]], shape[axes[1]]

    def mask_domain(self, data: Any) -> "ImageMaskDomain":
        """Return the mask domain declared for this payload."""
        spatial_axes_yx = (
            self.spatial_axes_yx(data)
            if self.source_spatial_domain.source_shape_yx is not None
            else None
        )
        return ImageMaskDomain(
            tuple(int(axis_size) for axis_size in np.shape(image_payload_data(data))),
            self.normalized_source_channel_axis(data),
            self.plane_axis,
            spatial_axes_yx,
        )

    def without_source_channel_axis(self) -> "ImagePayloadMetadata":
        """Return metadata after an operation collapses the source channel axis."""
        return self.replace_fields(source_channel_axis=None)

    def for_leading_source_plane(self, plane_index: int) -> "ImagePayloadMetadata":
        """Project metadata after explicitly removing its leading plane axis."""
        if self.plane_axis is None:
            raise ValueError(
                "Leading source-plane projection requires a declared plane axis."
            )
        return self.for_source_plane(plane_index).without_leading_plane_axis()

    def without_leading_plane_axis(self) -> "ImagePayloadMetadata":
        """Return metadata after an explicitly declared leading axis is removed."""
        if self.plane_axis is None:
            raise ValueError("Image metadata has no leading plane axis to remove.")
        source_channel_axis = self.source_channel_axis
        if source_channel_axis == 0:
            raise ValueError(
                "Image metadata cannot declare the same leading axis as both "
                "plane and channel."
            )
        if source_channel_axis is not None and source_channel_axis > 0:
            source_channel_axis -= 1
        return self.with_source_provenance(
            self.source_provenance.with_runtime_planes_as_contributors()
        ).replace_fields(
            plane_axis=None,
            source_channel_axis=source_channel_axis,
            source_plane_intensity_scales=(),
            source_plane_dtypes=(),
            unit_interval_intensity=(
                None
                if self.unit_interval_intensity is None
                else self.unit_interval_intensity.without_source_planes()
            ),
        )

    def collapse_leading_plane_axis(self) -> "ImagePayloadMetadata":
        """Return scalar metadata after reducing every plane of the leading axis."""

        collapsed = self.without_leading_plane_axis()
        return collapsed.with_source_provenance(
            collapsed.source_provenance.with_common_scalar_identity_from_planes()
        )

    @property
    def has_plane_specific_values(self) -> bool:
        """Return whether selecting a payload plane can change metadata."""
        return any(
            (
                bool(self.source_plane_intensity_scales),
                bool(self.source_plane_dtypes),
                bool(self.source_plane_unit_interval_intensity_scales),
                self.source_provenance.source_plane_count > 0,
                len(self.source_image_names) > 1,
            )
        )

    @property
    def source_image_paths(self) -> tuple[str, ...]:
        """Return paths from exact source identities represented by this payload."""
        return tuple(
            dict.fromkeys(
                str(identity.path)
                for identity in self.source_provenance.represented_source_identities
                if identity.path is not None and str(identity.path)
            )
        )

    def intensity_scale_for_source_plane(self, plane_index: int) -> float | None:
        """Return the best available intensity scale for one source plane."""
        if 0 <= plane_index < len(self.source_plane_intensity_scales):
            plane_scale = self.source_plane_intensity_scales[plane_index]
            if plane_scale is not None:
                return plane_scale
        return self.intensity_scale

    def unit_interval_intensity_scale_for_source_plane(
        self,
        plane_index: int,
    ) -> int | None:
        """Return the scale proving current pixels are exact integer/scale values."""
        if 0 <= plane_index < len(self.source_plane_unit_interval_intensity_scales):
            plane_scale = self.source_plane_unit_interval_intensity_scales[plane_index]
            if plane_scale is not None:
                return int(plane_scale)
        return self.unit_interval_intensity_scale

    @property
    def source_plane_metadata_count(self) -> int:
        """Return the cardinality represented by plane-specific metadata."""
        return max(
            1,
            self.source_provenance.source_plane_count,
            len(self.source_plane_intensity_scales),
            len(self.source_plane_dtypes),
            len(self.source_plane_unit_interval_intensity_scales),
        )

    def source_plane_metadata_records(self) -> tuple["ImagePayloadMetadata", ...]:
        """Return one scalar metadata record per represented source plane."""
        if self.source_plane_metadata_count == 1:
            return (self,)
        return tuple(
            self.for_source_plane(plane_index)
            for plane_index in range(self.source_plane_metadata_count)
        )

    def common_unit_interval_intensity_scale(self) -> int | None:
        """Return the common unit-interval quantization proof for this payload."""
        if self.source_plane_unit_interval_intensity_scales:
            present = tuple(
                int(scale)
                for plane_index in range(
                    len(self.source_plane_unit_interval_intensity_scales)
                )
                for scale in (
                    self.unit_interval_intensity_scale_for_source_plane(plane_index),
                )
                if scale is not None
            )
            if not present:
                return None
            first = present[0]
            if all(scale == first for scale in present):
                return first
            return None
        return self.unit_interval_intensity_scale

    def payload_with(self, data: Any, mask: Any | None = None) -> Any:
        """Return image payload data carrying this metadata."""
        if mask is None and self.has_values:
            return ImageMetadataPayload(data=data, metadata=self)
        if mask is None:
            return data
        return MaskedImagePayload(data=data, mask=mask, metadata=self)

    def for_source_plane(self, plane_index: int) -> "ImagePayloadMetadata":
        """Return metadata for one source plane sliced from a stacked payload."""
        source_provenance = self.source_provenance.for_source_plane(plane_index)
        return self.replace_fields(
            intensity_scale=self.intensity_scale_for_source_plane(plane_index),
            source_dtype=_tuple_value(self.source_plane_dtypes, plane_index)
            or self.source_dtype,
            source_provenance=source_provenance,
            unit_interval_intensity=(
                None
                if self.unit_interval_intensity is None
                else self.unit_interval_intensity.for_source_plane(plane_index)
            ),
            source_plane_intensity_scales=(),
            source_plane_dtypes=(),
        )

    def for_source_planes(
        self,
        plane_indices: Sequence[int],
    ) -> "ImagePayloadMetadata":
        """Return metadata projected to an ordered group of source planes."""
        normalized_indices = tuple(int(index) for index in plane_indices)
        if not normalized_indices:
            raise ValueError("Source-plane metadata projection cannot be empty.")
        if not self.has_plane_specific_values:
            return self
        invalid_indices = tuple(
            index
            for index in normalized_indices
            if index < 0 or index >= self.source_plane_metadata_count
        )
        if invalid_indices:
            raise IndexError(
                "Source-plane metadata projection indices must be within "
                f"0..{self.source_plane_metadata_count - 1}; got "
                f"{invalid_indices!r}."
            )
        if len(normalized_indices) == 1:
            return self.for_source_plane(normalized_indices[0])
        return self.with_source_provenance(
            self.source_provenance.for_source_planes(normalized_indices)
        ).replace_fields(
            source_plane_intensity_scales=_tuple_values_at_indices(
                self.source_plane_intensity_scales,
                normalized_indices,
            ),
            source_plane_dtypes=_tuple_values_at_indices(
                self.source_plane_dtypes,
                normalized_indices,
            ),
            unit_interval_intensity=(
                None
                if self.unit_interval_intensity is None
                else self.unit_interval_intensity.for_source_planes(normalized_indices)
            ),
        )

    def project_declared_source_image(
        self,
        payload: RuntimeArrayData,
        source_image_name: str,
    ) -> RuntimeArrayData:
        """Project pixels and metadata to one exact declared source image."""

        source_plane_selection = self.source_provenance.source_plane_selection(
            source_image_name
        )
        if source_plane_selection is None:
            raise ValueError(
                f"Image metadata does not represent declared source image "
                f"{source_image_name!r}; represented names are "
                f"{self.source_provenance.represented_source_image_names!r}, "
                "with provenance planes "
                f"{self.source_image_provenance_planes.identity!r}."
            )
        if not source_plane_selection:
            return self.attach_to(payload)
        complete_source_axis = tuple(range(self.source_provenance.source_plane_count))
        if self.plane_axis is None:
            if source_plane_selection == complete_source_axis:
                return self.attach_to(payload)
            channel_axis = self.normalized_source_channel_axis(payload)
            if channel_axis is not None:
                if len(source_plane_selection) != 1:
                    raise ValueError(
                        "Declared source-image channel projection requires exactly "
                        f"one channel; got source planes {source_plane_selection!r} "
                        f"for {source_image_name!r}."
                    )
                return self.project_channel_payload(
                    payload,
                    image_payload_data(payload),
                    source_plane_selection[0],
                    channel_axis=channel_axis,
                )
            raise ValueError(
                "Declared source-image payload projection requires a declared "
                "plane or channel axis."
            )
        if self.plane_axis not in (
            RuntimePlaneAxis.RUNTIME_SLICE,
            RuntimePlaneAxis.SOURCE_BINDING,
        ):
            raise ValueError(
                "Source-image identity projection requires a runtime-slice, "
                "source-binding, or channel axis."
            )
        axis_projection = RuntimePlaneAxisValueProjection.preserve(
            axis=self.plane_axis,
            axis_size=self.source_provenance.source_plane_count,
            source_aliases=self.source_image_names,
        )
        axis_projection.validate_shape(
            np.shape(image_payload_data(payload)),
            value_name="Declared source-image payload",
        )
        if source_plane_selection == complete_source_axis:
            return self.attach_to(payload)

        from openhcs.core.aligned_image_payload import stack_image_payloads
        from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

        projected_planes = tuple(
            RuntimeSliceProjection.value_for_slice(
                payload,
                axis_projection.selected_plane(plane_index),
            )
            for plane_index in source_plane_selection
        )
        if len(projected_planes) == 1:
            return projected_planes[0]
        return stack_image_payloads(
            projected_planes,
            metadata_mode=ImagePayloadMetadataCompositionMode.for_plane_axis(
                self.plane_axis
            ),
        )

    def with_indexed_source_plane_provenance(
        self,
        expected_plane_count: int | None,
    ) -> "ImagePayloadMetadata":
        """Expand scalar source-plane metadata into per-plane provenance."""
        source_provenance = SourcePlaneIndexedProvenanceExpansion(
            self.source_provenance,
            expected_plane_count=expected_plane_count,
        ).expanded()
        if (
            source_provenance is self.source_provenance
            or source_provenance == self.source_provenance
        ):
            return self
        return self.with_source_provenance(source_provenance)

    def for_grouped_source_plane_projection(
        self,
        *,
        source_plane_indices: tuple[int, ...] | None,
        runtime_plane_index: int,
        runtime_plane_count: int | None = None,
    ) -> "ImagePayloadMetadata":
        """Return metadata for a runtime plane that may group source planes."""
        metadata = self.with_indexed_source_plane_provenance(runtime_plane_count)
        if source_plane_indices is None:
            return metadata.for_source_plane(runtime_plane_index)
        if len(source_plane_indices) == 1:
            return metadata.for_source_plane(source_plane_indices[0])
        if not source_plane_indices:
            if metadata.has_plane_specific_values:
                raise ValueError(
                    "Cannot assign one image metadata record to a runtime plane "
                    "that represents multiple source planes: "
                    f"{source_plane_indices!r}."
                )
            return metadata
        return metadata.for_source_planes(source_plane_indices)

    def with_unit_interval_intensity_scale(
        self,
        scale: int | None,
    ) -> "ImagePayloadMetadata":
        """Return metadata with the current unit-interval pixel proof updated."""
        return self.replace_fields(
            unit_interval_intensity=ImageUnitIntervalIntensityMetadata(scale=scale)
        )

    def without_unit_interval_intensity_scale(self) -> "ImagePayloadMetadata":
        """Return metadata after an arithmetic transform changed pixel values."""
        return self.replace_fields(
            unit_interval_intensity=ImageUnitIntervalIntensityMetadata(),
        )

    def without_spatial_domain(self) -> "ImagePayloadMetadata":
        """Return metadata with invalidated source-spatial placement removed."""
        return self.replace_fields(
            source_spatial_domain=SourceSpatialDomain(),
            source_voxel_spacing=self.source_voxel_spacing,
            physical_border_edges_yx=None,
            mask_defines_border=None,
        )

    def object_label_source_spatial_domain(self) -> SourceSpatialDomain:
        """Return this metadata's object-label source-image coordinate domain."""
        return self.source_spatial_domain.with_value_name(
            OBJECT_LABEL_SOURCE_SPATIAL_VALUE_NAME,
        )

    def with_source_spatial_context_from(
        self,
        source: "ImagePayloadMetadata",
    ) -> "ImagePayloadMetadata":
        """Fill missing source-image geometry without changing provenance."""
        spatial_domain = self.source_spatial_domain.with_missing_from(
            source.source_spatial_domain
        )
        return self.replace_fields(
            source_spatial_domain=spatial_domain,
            source_voxel_spacing=self.source_voxel_spacing.with_missing_from(
                source.source_voxel_spacing
            ),
            physical_border_edges_yx=(
                self.physical_border_edges_yx
                if self.physical_border_edges_yx is not None
                else source.physical_border_edges_yx
            ),
            mask_defines_border=(
                self.mask_defines_border
                if self.mask_defines_border is not None
                else source.mask_defines_border
            ),
        )

    def with_source_context_from(
        self,
        source: "ImagePayloadMetadata",
    ) -> "ImagePayloadMetadata":
        """Fill missing source-image identity and spatial context from a source."""
        source_provenance = self.source_provenance.with_missing_from(
            source.source_provenance
        )
        source_channel_axis = self.source_channel_axis
        if source_channel_axis is None:
            source_channel_axis = source.source_channel_axis
        plane_axis = self.plane_axis
        if plane_axis is None:
            plane_axis = source.plane_axis
        elif source.plane_axis is not None and source.plane_axis is not plane_axis:
            raise ValueError(
                "Cannot combine image metadata with conflicting plane axes: "
                f"{plane_axis.value!r} != {source.plane_axis.value!r}."
            )
        return self.with_source_spatial_context_from(source).replace_fields(
            source_provenance=source_provenance,
            source_channel_axis=source_channel_axis,
            plane_axis=plane_axis,
        )

    def with_missing_intensity_from(
        self,
        source: "ImagePayloadMetadata",
    ) -> "ImagePayloadMetadata":
        """Fill missing pixel-type and intensity metadata from a source payload."""
        return self.replace_fields(
            intensity_scale=(
                self.intensity_scale
                if self.intensity_scale is not None
                else source.intensity_scale
            ),
            source_dtype=(
                self.source_dtype
                if self.source_dtype is not None
                else source.source_dtype
            ),
            unit_interval_intensity=(
                self.unit_interval_intensity
                if self.unit_interval_intensity is not None
                else source.unit_interval_intensity
            ),
            source_plane_intensity_scales=(
                self.source_plane_intensity_scales
                or source.source_plane_intensity_scales
            ),
            source_plane_dtypes=(
                self.source_plane_dtypes or source.source_plane_dtypes
            ),
        )

    def with_source_provenance(
        self,
        source_provenance: SourceImageProvenance,
    ) -> "ImagePayloadMetadata":
        """Return metadata with source-image provenance replaced atomically."""
        return self.replace_fields(source_provenance=source_provenance)

    def with_source_component_metadata(
        self,
        source_component_metadata: SourceComponentMetadata | None,
    ) -> "ImagePayloadMetadata":
        """Return metadata with only scalar source component metadata changed."""
        return self.with_source_provenance(
            self.source_provenance.with_source_component_metadata(
                source_component_metadata
            )
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
        return self.source_spatial_domain.physical_border_edges_for_shape(
            image_shape_yx
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
        output_shape = _source_spatial_shape_pair(output_shape_yx, "output_shape_yx")
        spatial_domain = self.source_spatial_domain.with_spatial_crop(
            input_shape_yx=input_shape_yx,
            output_shape_yx=output_shape_yx,
            offset_yx=offset_yx,
        )
        if physical_border_edges_yx is None:
            physical_border_edges_yx = spatial_domain.physical_border_edges_for_shape(
                output_shape
            )
        return self.replace_fields(
            source_spatial_domain=spatial_domain,
            physical_border_edges_yx=tuple(
                bool(edge) for edge in physical_border_edges_yx
            ),
        )

    def with_spatial_resize(
        self,
        output_shape_yx: Sequence[int],
    ) -> "ImagePayloadMetadata":
        """Return metadata in the local coordinate domain created by a resize."""

        return self.replace_fields(
            source_spatial_domain=self.source_spatial_domain.with_spatial_resize(
                output_shape_yx
            ),
            physical_border_edges_yx=(True, True, True, True),
            mask_defines_border=None,
        )

    def with_materialized_source_domain(
        self,
        target_domain: SourceSpatialDomain,
    ) -> "ImagePayloadMetadata":
        """Return metadata after pixels are expanded to source-image XY."""
        return self.replace_fields(
            source_spatial_domain=(
                self.source_spatial_domain.as_materialized_source_domain(target_domain)
            ),
            physical_border_edges_yx=(True, True, True, True),
        )


class ImagePayloadMetadataCarrier(ABC):
    """Nominal contract for image payloads that carry runtime metadata."""

    @property
    @abstractmethod
    def metadata(self) -> ImagePayloadMetadata:
        """Return the image metadata attached to this payload."""


@dataclass(frozen=True, slots=True)
class ImageMetadataPayload(DataBackedRuntimeArrayPayload, ImagePayloadMetadataCarrier):
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
class MaskedImagePayload(DataBackedRuntimeArrayPayload, ImagePayloadMetadataCarrier):
    """Image data plus an authoritative per-pixel validity mask."""

    data: Any
    mask: Any
    metadata: ImagePayloadMetadata = field(default_factory=ImagePayloadMetadata)

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
        if not self.metadata.mask_domain(self.data).accepts(mask_shape):
            raise ValueError(
                "MaskedImagePayload.mask shape must match the image spatial "
                f"domain; got mask {mask_shape!r} for image {data_shape!r}."
            )

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
    if isinstance(payload, ImagePayloadMetadataCarrier):
        return payload.metadata
    return ImagePayloadMetadata()


def preserved_image_plane_projection(
    payload: Any,
    projector: RuntimePlaneAxisProjector,
    source_aliases: tuple[str, ...] = (),
) -> RuntimePlaneAxisValueProjection | None:
    """Resolve the declared leading image axis against one runtime invocation."""

    from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

    metadata = image_payload_metadata(payload)
    if metadata.plane_axis is None:
        return RuntimeSliceProjection.preserved_context_for_value(payload)
    if metadata.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE:
        invocation_projection = RuntimePlaneAxisValueProjection.from_projector(
            projector,
            metadata.plane_axis,
            source_aliases,
        )
        if (
            invocation_projection is not None
            and invocation_projection.plane_index is not None
        ):
            return None
        payload_projection = RuntimeSliceProjection.preserved_context_for_value(payload)
        if payload_projection is None:
            raise ValueError(
                "Runtime-slice image metadata requires a payload with the same "
                "nominal leading axis."
            )
        return payload_projection

    shape = tuple(int(size) for size in np.shape(image_payload_data(payload)))
    if len(shape) < 3:
        raise ValueError(
            "Source-binding image metadata requires its declared plane axis as "
            f"the leading dimension, got shape {shape!r}."
        )
    axis_size = shape[0]
    return RuntimePlaneAxisValueProjection(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_aliases=source_aliases,
        plane_index=metadata.source_provenance.source_alias_plane_index(
            source_aliases,
            axis_size,
        ),
        axis_size=axis_size,
    )


def preserve_declared_image_payload_axis(
    projector: RuntimePlaneAxisProjector,
    output_payload: Any,
    *,
    source_payload: Any | None = None,
) -> RuntimePlaneAxisValueProjection | None:
    """Preserve the exact image axis declared by output or source ownership."""

    from openhcs.core.runtime_slice_projection import RuntimeSliceProjection

    output_metadata = image_payload_metadata(output_payload)
    output_projection = RuntimeSliceProjection.preserved_context_for_value(
        output_payload
    )
    output_axis = output_metadata.plane_axis or (
        None if output_projection is None else output_projection.axis
    )

    source_metadata = image_payload_metadata(source_payload)
    source_projection = RuntimeSliceProjection.preserved_context_for_value(
        source_payload
    )
    source_axis = source_metadata.plane_axis or (
        None if source_projection is None else source_projection.axis
    )
    if output_axis is not None:
        owner = output_payload
    elif source_axis is not None:
        owner = source_payload
    else:
        return None
    return preserved_image_plane_projection(
        owner,
        projector,
    )


def project_image_mask_to_data_domain(
    mask: Any,
    data: Any,
    *,
    metadata: ImagePayloadMetadata | None = None,
) -> Any | None:
    """Validate a mask against explicit image-domain metadata."""
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    mask_shape = tuple(mask_array.shape)
    resolved_metadata = image_payload_metadata(data) if metadata is None else metadata
    mask_domain = resolved_metadata.mask_domain(data)
    if mask_domain.accepts(mask_shape):
        return mask_array
    raise ValueError(
        f"Mask shape {mask_shape!r} does not match declared image mask domain "
        f"{tuple(sorted(mask_domain.valid_shapes()))!r}."
    )


def image_mask_for_data_domain(
    *,
    source_payload: Any,
    data: Any,
    explicit_mask: Any | None = None,
) -> Any | None:
    """Return the effective image mask projected into a concrete data domain."""
    source_mask = (
        image_payload_mask(source_payload)
        if explicit_mask is None
        else image_payload_data(explicit_mask)
    )
    metadata = image_payload_metadata(source_payload)
    projected_mask = project_image_mask_to_data_domain(
        source_mask,
        data,
        metadata=metadata,
    )
    if projected_mask is None:
        return None
    return metadata.mask_domain(data).broadcast_to_data(projected_mask)


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
        metadata=image_payload_metadata(payload) if metadata is None else metadata,
    )
    resolved_metadata = (
        image_payload_metadata(payload) if metadata is None else metadata
    )
    return resolved_metadata.payload_with(data, resolved_mask)


def image_payload_slice_context(
    payload: Any,
    data: Any,
    plane_index: int,
    *,
    plane_axis: RuntimePlaneAxis | None = None,
) -> Any:
    """Attach one source plane of a payload's image context to slice data."""
    metadata = image_payload_metadata(payload)
    if plane_axis is not None:
        plane_axis = RuntimePlaneAxis(
            plane_axis,
        )
        if metadata.plane_axis is not None and metadata.plane_axis is not plane_axis:
            raise ValueError(
                "Image slice projection axis conflicts with payload metadata: "
                f"{plane_axis.value!r} != {metadata.plane_axis.value!r}."
            )
        metadata = metadata.replace_fields(plane_axis=plane_axis)
    mask = image_payload_mask(payload)
    return metadata.for_leading_source_plane(plane_index).payload_with(
        data,
        image_payload_mask_for_slice(
            mask=mask,
            metadata=metadata,
            data_slice=data,
            plane_index=plane_index,
        ),
    )


def image_payload_mask_for_slice(
    *,
    mask: Any | None,
    metadata: ImagePayloadMetadata,
    data_slice: RuntimeArrayData,
    plane_index: int,
) -> RuntimeArrayData | None:
    """Project a shared or plane-specific mask into one declared image slice."""

    if mask is None:
        return None
    mask_array = np.asarray(mask)
    slice_metadata = metadata.for_leading_source_plane(plane_index)
    if metadata.plane_axis is None:
        if plane_index != 0:
            raise ValueError(
                "Image payload without a plane axis cannot select nonzero "
                f"slice index {plane_index}."
            )
        candidate = mask_array
    elif (
        metadata.plane_axis is RuntimePlaneAxis.SOURCE_BINDING
        and slice_metadata.mask_domain(data_slice).accepts(mask_array.shape)
    ):
        candidate = mask_array
    else:
        if mask_array.ndim == 0 or plane_index >= mask_array.shape[0]:
            raise ValueError(
                "Image payload mask does not carry the requested declared "
                f"slice index {plane_index}; got shape {mask_array.shape!r}."
            )
        candidate = mask_array[plane_index]
    if slice_metadata.mask_domain(data_slice).accepts(tuple(np.shape(candidate))):
        return candidate
    raise ValueError(
        "Image payload mask cannot be projected into slice domain; "
        f"got mask {mask_array.shape!r} for slice "
        f"{tuple(np.shape(data_slice))!r}."
    )


class ImagePayloadMetadataCompositionMode(Enum):
    """Source-provenance topology for composed image metadata."""

    def __new__(
        cls,
        value: str,
        plane_axis: RuntimePlaneAxis,
    ):
        member = object.__new__(cls)
        member._value_ = value
        member._plane_axis = plane_axis
        return member

    STACK = ("stack", RuntimePlaneAxis.RUNTIME_SLICE)
    BUNDLE = ("bundle", RuntimePlaneAxis.SOURCE_BINDING)
    plane_axis = AliasProperty[RuntimePlaneAxis]("_plane_axis")

    @classmethod
    def for_plane_axis(
        cls,
        plane_axis: RuntimePlaneAxis,
    ) -> "ImagePayloadMetadataCompositionMode":
        """Return the unique composition operation that creates an axis."""

        matches = tuple(mode for mode in cls if mode.plane_axis is plane_axis)
        if len(matches) != 1:
            raise ValueError(
                f"No unique image composition mode owns {plane_axis.value!r}."
            )
        return matches[0]

    def preserves_plane_topology(
        self,
        plane_axis: RuntimePlaneAxis | None,
    ) -> bool:
        """Return whether composition retains an existing runtime-plane axis."""

        return plane_axis is self.plane_axis or (
            plane_axis is None and self.plane_axis is RuntimePlaneAxis.RUNTIME_SLICE
        )


@dataclass(slots=True)
class _ImagePayloadMetadataComposer:
    """Stateful implementation for composing image metadata on a leading axis."""

    payloads: tuple[RuntimeArrayData, ...]
    mode: ImagePayloadMetadataCompositionMode = (
        ImagePayloadMetadataCompositionMode.STACK
    )
    source_metadata_override: Sequence[ImagePayloadMetadata] | None = None
    metadata_type: type[ImagePayloadMetadata] = ImagePayloadMetadata

    def __post_init__(self) -> None:
        self.payloads = tuple(self.payloads)
        if not self.payloads:
            raise ValueError("Image metadata composition payloads cannot be empty.")
        if self.source_metadata_override is None:
            return
        self.source_metadata_override = tuple(self.source_metadata_override)
        if len(self.source_metadata_override) != len(self.payloads):
            raise ValueError(
                "Image metadata composition source metadata must match payload count."
            )

    @property
    def source_metadata(self) -> tuple[ImagePayloadMetadata, ...]:
        if self.source_metadata_override is None:
            return tuple(image_payload_metadata(payload) for payload in self.payloads)
        return tuple(self.source_metadata_override)

    @staticmethod
    def source_plane_metadata_for_payload(
        metadata: ImagePayloadMetadata,
    ) -> ImagePayloadMetadata:
        """Return metadata for one payload on the newly composed leading axis."""
        if metadata.plane_axis is not None:
            return metadata
        if metadata.source_provenance.source_plane_count == 1:
            return metadata.for_source_plane(0)
        return metadata

    def compose(self) -> ImagePayloadMetadata:
        metadata_by_payload = self.source_metadata
        if not any(metadata.has_values for metadata in metadata_by_payload):
            return self.metadata_type(plane_axis=self.mode.plane_axis)
        source_metadata_by_payload = tuple(
            self.source_plane_metadata_for_payload(metadata)
            for metadata in metadata_by_payload
        )
        source_plane_metadata_records = source_metadata_by_payload
        composed_source_provenance_planes = self.composed_source_provenance_planes(
            source_plane_metadata_records
        )
        common_source_voxel_spacing = self.common_metadata_value(
            metadata.source_voxel_spacing
            for metadata in metadata_by_payload
            if metadata.source_voxel_spacing.has_values
        )
        if common_source_voxel_spacing is None:
            common_source_voxel_spacing = SourceVoxelSpacing()
        source_component_metadata_by_payload = tuple(
            (
                metadata
                if metadata.source_component_metadata is not None
                else source_metadata
            )
            for metadata, source_metadata in zip(
                metadata_by_payload,
                source_metadata_by_payload,
                strict=True,
            )
        )
        return self.metadata_type(
            source_path=self.common_metadata_value(
                metadata.source_path for metadata in source_metadata_by_payload
            ),
            source_component_metadata=self.common_source_component_metadata(
                source_component_metadata_by_payload
            ),
            source_plane_intensity_scales=tuple(
                metadata.intensity_scale for metadata in source_plane_metadata_records
            ),
            source_plane_dtypes=tuple(
                metadata.source_dtype for metadata in source_plane_metadata_records
            ),
            source_image_provenance_planes=composed_source_provenance_planes,
            unit_interval_intensity=self.composed_unit_interval_intensity(
                source_plane_metadata_records
            ),
            source_spatial_domain=SourceSpatialDomain(
                origin_yx=self.common_metadata_value(
                    metadata.spatial_origin_yx for metadata in metadata_by_payload
                ),
                source_shape_yx=self.common_metadata_value(
                    metadata.source_spatial_shape_yx for metadata in metadata_by_payload
                ),
            ),
            source_voxel_spacing=common_source_voxel_spacing,
            physical_border_edges_yx=self.common_metadata_value(
                metadata.physical_border_edges_yx for metadata in metadata_by_payload
            ),
            mask_defines_border=self.common_metadata_value(
                metadata.mask_defines_border for metadata in metadata_by_payload
            ),
            source_image_names=(
                composed_source_provenance_planes.runtime_source_image_names
            ),
            source_channel_axis=self.composed_source_channel_axis(metadata_by_payload),
            plane_axis=self.mode.plane_axis,
        )

    def composed_source_provenance_planes(
        self,
        metadata_records: tuple[ImagePayloadMetadata, ...],
    ) -> SourceImageProvenancePlanes:
        """Compose projectable planes without replacing compatible topology."""

        if len(metadata_records) == 1:
            metadata = metadata_records[0]
            provenance_planes = metadata.source_image_provenance_planes
            if provenance_planes.count > 1 and self.mode.preserves_plane_topology(
                metadata.plane_axis
            ):
                return provenance_planes
        return SourceImageProvenancePlanes(
            tuple(
                self.runtime_source_provenance_plane(metadata)
                for metadata in metadata_records
            )
        )

    @staticmethod
    def composed_unit_interval_intensity(
        metadata_records: Sequence[ImagePayloadMetadata],
    ) -> ImageUnitIntervalIntensityMetadata | None:
        """Compose only unit-interval proof state authored by an input payload."""

        if not any(
            metadata.unit_interval_intensity is not None
            for metadata in metadata_records
        ):
            return None
        return ImageUnitIntervalIntensityMetadata(
            source_plane_scales=tuple(
                metadata.unit_interval_intensity_scale for metadata in metadata_records
            )
        )

    @staticmethod
    def runtime_source_provenance_plane(
        metadata: ImagePayloadMetadata,
    ) -> RuntimeSourceImageProvenancePlane:
        """Return one projectable plane with nested non-projectable contributors."""
        source_image_names = metadata.source_image_names
        contributors = metadata.source_image_provenance_planes.as_contributors(
            source_image_names
        ).planes
        if len(source_image_names) > 1:
            raise ValueError(
                "Composed image payload provenance permits at most one "
                f"source alias per scalar plane, got {source_image_names!r}."
            )
        source_image_name = source_image_names[0] if source_image_names else None
        return RuntimeSourceImageProvenancePlane(
            SourceImageIdentity(
                metadata.source_path,
                metadata.source_component_metadata,
            ),
            contributors,
            source_image_name,
        )

    def common_source_component_metadata(
        self,
        values: Iterable[ImagePayloadMetadata],
    ) -> SourceComponentMetadata | None:
        """Return source metadata shared by the composed payload."""
        metadata_values = tuple(values)
        metadata_by_plane = tuple(
            dict(metadata.source_component_metadata)
            for metadata in metadata_values
            if metadata.source_component_metadata is not None
        )
        if self.mode is ImagePayloadMetadataCompositionMode.BUNDLE:
            field_names = set().union(
                *(metadata.keys() for metadata in metadata_by_plane)
            )
            common_metadata: dict[str, SourceMetadataValue] = {}
            for field_name in field_names:
                values_for_field = tuple(
                    metadata[field_name]
                    for metadata in metadata_by_plane
                    if field_name in metadata
                )
                if values_for_field and all(
                    value == values_for_field[0] for value in values_for_field
                ):
                    common_metadata[field_name] = values_for_field[0]
        else:
            common_metadata = dict(
                common_source_component_metadata(
                    tuple(
                        metadata.source_component_metadata
                        for metadata in metadata_values
                    )
                )
                or {}
            )
        extension = self.common_metadata_value(
            "".join(Path(metadata.source_path).suffixes)
            for metadata in metadata_values
            if metadata.source_path is not None
        )
        if extension:
            common_metadata.setdefault("extension", extension)
        if not common_metadata:
            return None
        return MappingProxyType(common_metadata)

    @staticmethod
    def common_metadata_value(
        values: Iterable[MetadataValueT | None],
    ) -> MetadataValueT | None:
        values_tuple = tuple(values)
        present = tuple(value for value in values_tuple if value is not None)
        if not present:
            return None
        first = present[0]
        if all(value == first for value in present):
            return first
        return None

    def composed_source_channel_axis(
        self,
        metadata_by_payload: tuple[ImagePayloadMetadata, ...],
    ) -> int | None:
        """Return the channel axis after this composition adds a leading axis."""
        normalized = tuple(
            metadata.normalized_source_channel_axis(payload)
            for payload, metadata in zip(
                self.payloads,
                metadata_by_payload,
                strict=True,
            )
        )
        present = tuple(value for value in normalized if value is not None)
        if not present:
            return None
        first = present[0]
        if any(value != first for value in present[1:]):
            raise ValueError(
                "Cannot compose image payloads with conflicting source channel axes: "
                f"{present!r}."
            )
        return first + 1


def image_intensity_scale_for_dtype(dtype: Any) -> float | None:
    """Return the conventional full-scale intensity for a pixel dtype."""
    normalized = np.dtype(dtype)
    if np.issubdtype(normalized, np.bool_):
        return 1.0
    if np.issubdtype(normalized, np.integer):
        return float(np.iinfo(normalized).max)
    return None


def image_payload_intensity_scale(
    payload: Any,
    *,
    channel_index: int = 0,
) -> float | None:
    """Return the best semantic intensity scale for an image payload."""
    import numpy as np

    metadata_scale = image_payload_metadata(payload).intensity_scale_for_source_plane(
        channel_index
    )
    if metadata_scale is not None and metadata_scale > 0:
        return float(metadata_scale)
    return image_intensity_scale_for_dtype(
        np.asarray(image_payload_data(payload)).dtype
    )


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
            metadata = image_payload_metadata(
                payload
            ).with_unit_interval_intensity_scale(int(intensity_scale))
            return with_image_payload_data(payload, normalized, metadata=metadata)
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(target_dtype, copy=False)
    else:
        return payload
    return with_image_payload_data(payload, normalized)


def _tuple_value(values: tuple[Any, ...], index: int) -> Any | None:
    if 0 <= index < len(values):
        return values[index]
    return None


def _tuple_values_at_indices(
    values: tuple[Any, ...],
    indices: tuple[int, ...],
) -> tuple[Any, ...]:
    """Project optional plane-specific tuple values without inventing records."""
    if not values:
        return ()
    return tuple(_tuple_value(values, index) for index in indices)


@dataclass(frozen=True, slots=True)
class ImageMaskDomain:
    """Accepted mask shapes for an explicitly declared image data domain."""

    data_shape: tuple[int, ...]
    channel_axis: int | None = None
    plane_axis: RuntimePlaneAxis | None = None
    spatial_axes_yx: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        data_shape = tuple(int(axis_size) for axis_size in self.data_shape)
        object.__setattr__(self, "data_shape", data_shape)
        channel_axis = self.channel_axis
        if channel_axis is not None:
            normalized = int(channel_axis)
            if normalized < 0:
                normalized += len(data_shape)
            if normalized < 0 or normalized >= len(data_shape):
                raise ValueError(
                    f"Image mask channel axis {channel_axis} is invalid for "
                    f"shape {data_shape!r}."
                )
            object.__setattr__(self, "channel_axis", normalized)
        spatial_axes_yx = self.spatial_axes_yx
        if spatial_axes_yx is None:
            return
        if len(set(spatial_axes_yx)) != 2 or any(
            axis < 0 or axis >= len(data_shape) for axis in spatial_axes_yx
        ):
            raise ValueError(
                "Image mask spatial axes must be two distinct data axes; "
                f"got {spatial_axes_yx!r} for shape {data_shape!r}."
            )

    @property
    def shared_spatial_mask_shape(self) -> tuple[int, int] | None:
        """Return a mask domain shared by declared source-binding planes."""
        if (
            self.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING
            or self.spatial_axes_yx is None
        ):
            return None
        return tuple(self.data_shape[axis] for axis in self.spatial_axes_yx)

    def accepts(self, mask_shape: tuple[int, ...]) -> bool:
        return mask_shape in self.valid_shapes()

    def valid_shapes(self) -> frozenset[tuple[int, ...]]:
        valid = {self.data_shape}
        if self.channel_axis is not None:
            valid.add(
                tuple(
                    axis_size
                    for axis, axis_size in enumerate(self.data_shape)
                    if axis != self.channel_axis
                )
            )
        if self.shared_spatial_mask_shape is not None:
            valid.add(self.shared_spatial_mask_shape)
        return frozenset(valid)

    def default_mask_shape(self) -> tuple[int, ...]:
        """Return the canonical mask shape for this declared image domain."""
        if self.shared_spatial_mask_shape is not None:
            return self.shared_spatial_mask_shape
        if self.channel_axis is None:
            return self.data_shape
        return tuple(
            axis_size
            for axis, axis_size in enumerate(self.data_shape)
            if axis != self.channel_axis
        )

    def broadcast_to_data(self, mask: Any) -> np.ndarray:
        """Broadcast a valid mask across its declared non-spatial axes."""
        mask_array = np.asarray(mask, dtype=bool)
        mask_shape = tuple(mask_array.shape)
        if mask_shape == self.data_shape:
            return mask_array
        if mask_shape == self.shared_spatial_mask_shape:
            if self.spatial_axes_yx is None:
                raise AssertionError("Shared spatial mask axes are missing.")
            broadcast_shape = [1] * len(self.data_shape)
            for mask_axis, data_axis in enumerate(self.spatial_axes_yx):
                broadcast_shape[data_axis] = mask_shape[mask_axis]
            return np.broadcast_to(
                mask_array.reshape(tuple(broadcast_shape)),
                self.data_shape,
            )
        channel_free_shape = (
            None
            if self.channel_axis is None
            else tuple(
                axis_size
                for axis, axis_size in enumerate(self.data_shape)
                if axis != self.channel_axis
            )
        )
        if mask_shape != channel_free_shape or self.channel_axis is None:
            raise ValueError(
                f"Mask shape {mask_shape!r} is not valid for image "
                f"shape {self.data_shape!r}."
            )
        return np.broadcast_to(
            np.expand_dims(mask_array, axis=self.channel_axis),
            self.data_shape,
        )
