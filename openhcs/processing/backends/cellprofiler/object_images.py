"""Object-label image rendering for CellProfiler-compatible processing."""

from __future__ import annotations

from enum import Enum

from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectInputCountAuthority,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    NoSpecialImageInputsMixin,
    SpecialInputBindingRequest,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


class ConvertObjectsToImageMode(Enum):
    """Object-label rendering modes exposed by ConvertObjectsToImage settings."""

    BINARY = "binary"
    GRAYSCALE = "grayscale"
    COLOR = "color"
    UINT16 = "uint16"


class ConvertObjectsToImageSpecialInputPolicy(
    NoSpecialImageInputsMixin,
):
    """Bind object labels as payloads so rendered images inherit label provenance."""


    def bind(
        self,
        request: SpecialInputBindingRequest,
    ) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name,
            object_inputs,
            1,
        )
        return {"labels": request.object_label_payload(object_inputs[0])}


class ConvertObjectsToImageModule(
    ConvertObjectsToImageSpecialInputPolicy,
    CellProfilerModule,
):
    module_name = 'ConvertObjectsToImage'
    function_name = 'convert_objects_to_image'
    validated = True
    contract = 'pure_3d'
    confidence = 1.0
    setting_bindings = (
        SettingToKeywordBinding(
            "Select the color format",
            "image_mode",
            cellprofiler_enum_value_setting_parser(ConvertObjectsToImageMode),
        ),
        SettingToKeywordBinding("Select the colormap", "colormap_value"),
    )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

        objects = builder.require_artifact(
            ArtifactSpec(required_setting_value(module, "Select the input objects"), ArtifactKind.OBJECT_LABELS),
            module,
        )
        output = builder.declare_artifact(
            ArtifactSpec(required_setting_value(module, "Name the output image"), ArtifactKind.IMAGE),
            module,
        )
        return assembler.assemble_contract(module, builder, inputs=[objects], outputs=[output])



from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_values import (
    image_payload_metadata,
    object_label_dense_array,
    with_image_payload_data,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois


class ImageMode(Enum):
    BINARY = "binary"
    GRAYSCALE = "grayscale"
    COLOR = "color"
    UINT16 = "uint16"


@dataclass(frozen=True, slots=True)
class ObjectConversionStats:
    """ConvertImageToObjects summary row."""

    slice_index: int
    object_count: int
    mean_area: float
    total_area: int


class ImageModeRenderer(ABC, metaclass=AutoRegisterMeta):
    """Render object labels for one closed ImageMode case."""

    __registry_key__ = "image_mode_label"
    __skip_if_no_key__ = True
    image_mode_label: ClassVar[str | None] = None
    image_mode: ClassVar[ImageMode | None] = None

    @classmethod
    def for_image_mode(cls, image_mode: ImageMode) -> "ImageModeRenderer":
        return cls.__registry__[image_mode.value]()

    @abstractmethod
    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        """Return one rendered image payload for the requested ImageMode."""


class BinaryImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.BINARY
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        del colormap_value
        return (labels > 0).astype(np.float32)


class GrayscaleImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.GRAYSCALE
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        del colormap_value
        max_label = labels.max()
        if max_label > 0:
            return labels.astype(np.float32) / max_label
        return np.zeros(labels.shape, dtype=np.float32)


class ColorImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.COLOR
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        max_label = labels.max()
        colors = object_label_colormap(colormap_value, max_label)
        pixel_data = colors[labels]
        return (
            np.float32(0.299) * pixel_data[..., 0]
            + np.float32(0.587) * pixel_data[..., 1]
            + np.float32(0.114) * pixel_data[..., 2]
        ).astype(np.float32, copy=False)


class Uint16ImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.UINT16
    image_mode_label = image_mode.value

    def render(
        self,
        labels: np.ndarray,
        *,
        colormap_value: str,
    ) -> np.ndarray:
        del colormap_value
        return labels.astype(np.int32, copy=False)


def object_label_colormap(colormap_name: str, num_labels: int) -> np.ndarray:
    """Generate colors for object labels using a matplotlib colormap."""
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(colormap_name)
    colors = np.zeros((num_labels + 1, 3), dtype=np.float32)
    for index in range(1, num_labels + 1):
        colors[index] = cmap(index / max(num_labels, 1))[:3]
    return colors


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "conversion_stats",
        csv_materializer(
            fields=["slice_index", "object_count", "mean_area", "total_area"],
            analysis_type="object_conversion",
        ),
    ),
    ("labels", segmentation_mask_rois()),
)
def convert_image_to_objects(
    image: np.ndarray,
    cast_to_bool: bool = False,
    preserve_label: bool = False,
    background: int = 0,
    connectivity: int = 1,
) -> tuple[np.ndarray, ObjectConversionStats, np.ndarray]:
    """Convert an image plane into CellProfiler-compatible object labels."""
    from skimage.measure import label

    working_image = image.copy()
    if cast_to_bool:
        working_image = (working_image != background).astype(np.uint8)

    if preserve_label:
        labels = working_image.astype(np.int32)
        labels[labels == background] = 0
    else:
        labels = label(
            working_image != background,
            connectivity=connectivity,
        ).astype(np.int32)

    props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(labels)
    object_count = int(props.label.size)
    if object_count > 0:
        mean_area = float(np.mean(props.area))
        total_area = int(np.sum(props.area))
    else:
        mean_area = 0.0
        total_area = 0
    return (
        image,
        ObjectConversionStats(
            slice_index=0,
            object_count=object_count,
            mean_area=mean_area,
            total_area=total_area,
        ),
        labels,
    )


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@special_inputs("labels")
def convert_objects_to_image(
    image: np.ndarray,
    labels: np.ndarray,
    image_mode: ImageMode = ImageMode.COLOR,
    colormap_value: str = "jet",
) -> np.ndarray:
    """Render object labels into the requested CellProfiler image mode."""
    del image
    label_array = object_label_dense_array(labels, dtype=np.int32)
    resolved_image_mode = coerce_cellprofiler_enum(ImageMode, image_mode)
    rendered = ImageModeRenderer.for_image_mode(resolved_image_mode).render(
        label_array,
        colormap_value=colormap_value,
    )
    return with_image_payload_data(
        labels,
        rendered,
        metadata=image_payload_metadata(labels).without_unit_interval_intensity_scale(),
    )


class ConvertImageToObjectsModule(CellProfilerModule):
    module_name = 'ConvertImageToObjects'
    function_name = 'convert_image_to_objects'
    validated = True
    contract = 'unknown'
    confidence = 1.0

__all__ = public_names_from_objects(
    BinaryImageModeRenderer,
    ColorImageModeRenderer,
    GrayscaleImageModeRenderer,
    ImageMode,
    ImageModeRenderer,
    ObjectConversionStats,
    Uint16ImageModeRenderer,
    convert_image_to_objects,
    convert_objects_to_image,
    object_label_colormap,
)
