"""Source image payload transforms implied by typed pipeline image semantics."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.image_shapes import is_color_image_slice, is_color_image_stack
from openhcs.core.pipeline_image_schema import (
    ImageTypeSourceRole,
    ImageStackSourceRole,
    MonochromeImageStackSourceRole,
    ObjectLabelsImageTypeSourceRole,
    SOURCE_IMAGE_TYPE_METADATA_FIELD,
)
from openhcs.core.registry_strategies import NominalTypeKeyedStrategyMixin
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_metadata_from_source,
    image_payload_with_context,
)
from openhcs.core.source_matching import source_metadata_value


def apply_source_image_loading_semantics(
    payload: Any,
    *,
    source_metadata: Mapping[str, str] | None,
    source_path: str | None,
    read_backend: str | None = None,
    filemanager: Any | None = None,
) -> Any:
    """Apply typed source image semantics to pixels loaded from storage."""

    return SourceImagePayloadSemantics.from_source_metadata(
        source_metadata=source_metadata,
        source_path=source_path,
        read_backend=read_backend,
        filemanager=filemanager,
    ).apply(payload)


@dataclass(frozen=True, slots=True)
class SourceImagePayloadSemantics:
    """Typed source-image role behavior applied to one loaded payload."""

    role: ImageTypeSourceRole | None
    source_path: str | None
    read_backend: str | None = None
    filemanager: Any | None = None

    @classmethod
    def from_source_metadata(
        cls,
        *,
        source_metadata: Mapping[str, str] | None,
        source_path: str | None,
        read_backend: str | None = None,
        filemanager: Any | None = None,
    ) -> "SourceImagePayloadSemantics":
        image_type = (
            None
            if source_metadata is None
            else source_metadata_value(
                source_metadata,
                SOURCE_IMAGE_TYPE_METADATA_FIELD,
            )
        )
        role = (
            None
            if image_type is None
            else ImageTypeSourceRole.for_image_type(image_type)
        )
        return cls(
            role=role,
            source_path=source_path,
            read_backend=read_backend,
            filemanager=filemanager,
        )

    def apply(self, payload: Any) -> Any:
        strategy = SourceImagePayloadRoleStrategy.for_role(self.role)
        data = strategy.source_data(payload)
        mask = strategy.source_mask(payload, data)
        metadata = self.source_metadata(payload, data)
        if (
            data is image_payload_data(payload)
            and mask is image_payload_mask(payload)
            and not metadata.has_values
        ):
            return payload
        return image_payload_with_context(
            data,
            mask=mask,
            metadata=metadata,
        )

    def source_metadata(self, payload: Any, data: Any) -> ImagePayloadMetadata:
        """Return source-file image metadata for transformed payload data."""
        existing_metadata = image_payload_metadata(payload)
        if self.source_path is None:
            if existing_metadata.has_values:
                return existing_metadata
            if self.role is None:
                return ImagePayloadMetadata()
            return ImagePayloadMetadata.for_array_payload(data)
        return image_payload_metadata_from_source(
            data,
            source_path=self.source_path,
            read_backend=self.read_backend,
            filemanager=self.filemanager,
        )


class SourceImagePayloadRoleStrategy(
    NominalTypeKeyedStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal payload behavior for declared source-image roles."""

    value_type: ClassVar[type[ImageTypeSourceRole] | None] = None
    value_type_label: ClassVar[str | None] = None
    __registry_key__ = "value_type_label"
    __skip_if_no_key__ = True
    role: ImageTypeSourceRole | None

    @classmethod
    def for_role(
        cls,
        role: ImageTypeSourceRole | None,
    ) -> "SourceImagePayloadRoleStrategy":
        if role is None:
            return UndeclaredSourceImagePayloadRoleStrategy()
        strategy = cls.for_nominal_value(role)
        if strategy is None:
            raise TypeError(
                f"No source-image payload strategy registered for {type(role).__name__}."
            )
        return strategy

    def source_data(self, payload: Any) -> Any:
        return image_payload_data(payload)

    def source_mask(self, payload: Any, data: Any) -> Any | None:
        return image_payload_mask(payload)


@dataclass(frozen=True, slots=True)
class UndeclaredSourceImagePayloadRoleStrategy(SourceImagePayloadRoleStrategy):
    """Payload behavior for sources without a pipeline image-type declaration."""

    value_type = None
    role: None = None


@dataclass(frozen=True, slots=True)
class DeclaredSourceImagePayloadRoleStrategy(SourceImagePayloadRoleStrategy):
    """Base payload behavior for declared pipeline image roles."""

    value_type = ImageTypeSourceRole
    role: ImageTypeSourceRole | None = None


@dataclass(frozen=True, slots=True)
class ImageStackSourcePayloadRoleStrategy(DeclaredSourceImagePayloadRoleStrategy):
    """Payload behavior for image roles entering the OpenHCS image stack."""

    value_type = ImageStackSourceRole

    def source_mask(self, payload: Any, data: Any) -> Any | None:
        mask = image_payload_mask(payload)
        if mask is not None:
            return mask
        return np.ones(self.source_mask_shape(data), dtype=bool)

    @staticmethod
    def source_mask_shape(data: Any) -> tuple[int, ...]:
        array = np.asarray(data)
        if is_color_image_slice(array):
            return tuple(int(value) for value in array.shape[:2])
        if is_color_image_stack(array):
            return tuple(int(value) for value in array.shape[:-1])
        return tuple(int(value) for value in array.shape)


@dataclass(frozen=True, slots=True)
class MonochromeImageStackSourcePayloadRoleStrategy(
    ImageStackSourcePayloadRoleStrategy
):
    """Payload behavior for CellProfiler monochrome source-image roles."""

    value_type = MonochromeImageStackSourceRole

    def source_data(self, payload: Any) -> Any:
        data = image_payload_data(payload)
        if is_color_image_slice(data):
            return self.cellprofiler_rgb_to_gray(np.asarray(data)[..., :3])
        if is_color_image_stack(data):
            return np.stack(
                tuple(
                    self.cellprofiler_rgb_to_gray(np.asarray(plane)[..., :3])
                    for plane in np.asarray(data)
                ),
                axis=0,
            )
        return data

    @staticmethod
    def cellprofiler_rgb_to_gray(rgb_data: Any) -> np.ndarray:
        from skimage.color import rgb2gray

        return rgb2gray(rgb_data)


@dataclass(frozen=True, slots=True)
class ObjectLabelsSourcePayloadRoleStrategy(DeclaredSourceImagePayloadRoleStrategy):
    """Payload behavior for externally supplied object-label source images."""

    value_type = ObjectLabelsImageTypeSourceRole

    def source_data(self, payload: Any) -> Any:
        data = np.asarray(image_payload_data(payload))
        if is_color_image_stack(data):
            return np.stack(
                tuple(self.color_label_plane_to_ids(plane) for plane in data),
                axis=0,
            )
        if is_color_image_slice(data):
            return self.color_label_plane_to_ids(data)
        return data

    @staticmethod
    def color_label_plane_to_ids(plane: Any) -> np.ndarray:
        """Convert CellProfiler color object-label images into dense label IDs."""
        rgb = np.asarray(plane)
        flat = rgb[..., :3].reshape(-1, 3)
        labels = np.zeros(flat.shape[0], dtype=np.int32)
        foreground = np.any(flat != 0, axis=1)
        if np.any(foreground):
            _colors, inverse = np.unique(flat[foreground], axis=0, return_inverse=True)
            labels[foreground] = inverse.astype(np.int32, copy=False) + 1
        return labels.reshape(rgb.shape[:2])
