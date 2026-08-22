"""Object-label image rendering for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.memory import numpy as numpy_decorator
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    special_inputs,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)


class ImageMode(Enum):
    """Object-label rendering modes exposed by ConvertObjectsToImage."""

    BINARY = "binary"
    GRAYSCALE = "grayscale"
    COLOR = "color"
    UINT16 = "uint16"


class ConvertObjectsToImageModule(
    ObjectArtifactInputModule,
    CellProfilerModule,
):
    module_name = "ConvertObjectsToImage"
    function_name = "convert_objects_to_image"
    validated = True
    confidence = 1.0
    input_objects_setting = SettingNameFamily("Select the input objects")
    output_image_setting = SettingNameFamily("Name the output image")
    setting_bindings = (
        SettingToKeywordBinding.input(
            input_objects_setting,
            ObjectLabelsArtifactType,
            runtime_parameter_name="labels",
        ),
        SettingToKeywordBinding.output(output_image_setting, ImageArtifactType),
        SettingToKeywordBinding(
            "Select the color format",
            "image_mode",
            cellprofiler_enum_setting_parser(ImageMode),
        ),
        SettingToKeywordBinding("Select the colormap", "colormap_value"),
    )


@dataclass(frozen=True, slots=True)
class ObjectConversionStats(MeasurementFeatureRecord):
    """ConvertImageToObjects summary row."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_count: int
    mean_area: float
    total_area: int


class ImageModeRenderer(
    EnumKeyedStrategyMixin[ImageMode],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Render object labels for one closed ImageMode case."""

    __enum_member_attr__ = "image_mode"
    image_mode: ClassVar[ImageMode | None] = None

    @abstractmethod
    def render(self, labels: np.ndarray, *, colormap_value: str) -> np.ndarray:
        """Return one rendered image payload for the requested ImageMode."""


class BinaryImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.BINARY

    def render(self, labels: np.ndarray, *, colormap_value: str) -> np.ndarray:
        del colormap_value
        return (labels > 0).astype(np.float32)


class GrayscaleImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.GRAYSCALE

    def render(self, labels: np.ndarray, *, colormap_value: str) -> np.ndarray:
        del colormap_value
        max_label = labels.max()
        if max_label > 0:
            return labels.astype(np.float32) / max_label
        return np.zeros(labels.shape, dtype=np.float32)


class ColorImageModeRenderer(ImageModeRenderer):
    image_mode = ImageMode.COLOR

    def render(self, labels: np.ndarray, *, colormap_value: str) -> np.ndarray:
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

    def render(self, labels: np.ndarray, *, colormap_value: str) -> np.ndarray:
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
def convert_image_to_objects(
    image: np.ndarray,
    cast_to_bool: bool = False,
    preserve_label: bool = False,
    background: int = 0,
    connectivity: int = 1,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Convert an image plane into CellProfiler-compatible object labels.

    Args:
        cast_to_bool: Treat every pixel unequal to ``background`` as foreground
            before labeling.
        preserve_label: Keep non-background pixel values as object IDs instead of
            relabeling connected foreground regions.
        background: Pixel value treated as background and mapped to object label 0.
        connectivity: Neighborhood connectivity used for connected-component
            labeling when ``preserve_label`` is false.
    """
    from skimage.measure import label

    working_image = image.copy()
    if cast_to_bool:
        working_image = (working_image != background).astype(np.uint8)
    if preserve_label:
        labels = working_image.astype(np.int32)
        labels[labels == background] = 0
    else:
        labels = label(working_image != background, connectivity=connectivity).astype(
            np.int32
        )
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
        DataclassMeasurementColumnarRows(
            (
                ObjectConversionStats(
                    slice_index=0,
                    object_count=object_count,
                    mean_area=mean_area,
                    total_area=total_area,
                ),
            ),
            row_type=ObjectConversionStats,
        ),
        SourceImageObjectLabelBuildRequest(
            image=image,
            labels=labels,
            declared_object_count=object_count,
            declared_object_ids=tuple(int(value) for value in props.label),
        ).payload(),
    )


@numpy_decorator(contract=ProcessingContract.PURE_3D)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
@special_inputs("labels")
def convert_objects_to_image(
    image: np.ndarray,
    labels: ObjectLabelValue,
    image_mode: ImageMode = ImageMode.COLOR,
    colormap_value: str = "jet",
) -> np.ndarray:
    """Render object labels into the requested CellProfiler image mode.

    Args:
        labels: Object-label image or volume to render in the selected image mode.
    """
    del image
    label_array = object_label_dense_array(labels, dtype=np.int32)
    rendered = ImageModeRenderer.for_enum_member(image_mode).render(
        label_array, colormap_value=colormap_value
    )
    label_metadata = image_payload_metadata(labels)
    output_metadata = (
        ImagePayloadMetadata(intensity_scale=1.0).with_source_context_from(
            label_metadata
        )
        if image_mode is ImageMode.UINT16
        else label_metadata
    )
    return with_image_payload_data(
        labels,
        rendered,
        metadata=output_metadata,
    )


class ConvertImageToObjectsModule(
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "ConvertImageToObjects"
    function_name = "convert_image_to_objects"
    validated = True
    confidence = 1.0
    input_image_setting = SettingNameFamily("Select the input image")
    output_objects_setting = SettingNameFamily("Name the output objects")
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.output(
            output_objects_setting,
            ObjectLabelsArtifactType,
        ),
        SettingToKeywordBinding(
            "Convert to boolean image",
            "cast_to_bool",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Preserve original labels",
            "preserve_label",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Background label",
            "background",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Connectivity",
            "connectivity",
            parse_cellprofiler_int,
        ),
    )


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
