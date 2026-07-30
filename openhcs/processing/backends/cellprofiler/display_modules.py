"""CellProfiler display module declarations and implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.artifacts import (
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    measurement_rows,
)
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    special_inputs,
    )
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    measurement_axis_integer_value,
)
from openhcs.core.runtime_tabular_values import (
    measurement_row_mapping,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    PriorMeasurementArtifactInputModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectMeasurementVectorBinding,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.parser import ModuleSetting
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    normalized_symbol_name,
    optional_setting_value,
    setting_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    MeasurementFeatureSettingBinding,
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler_setting_normalization import (
    normalize_cellprofiler_setting_name,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock


class DisplayMode(Enum):
    TEXT = "text"
    COLOR = "color"


class ObjectsOrImage(Enum):
    OBJECTS = "objects"
    IMAGE = "image"


class ColorMapScale(Enum):
    USE_MEASUREMENT_RANGE = "use_measurement_range"
    MANUAL = "manual"

    @property
    def cellprofiler_literals(self) -> tuple[str, ...]:
        if self is type(self).USE_MEASUREMENT_RANGE:
            return ("Use this image's measurement range",)
        return ()


class _DisplayMeasurementsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound object measurement vector rendered by DisplayDataOnImage."""

    parameter_name = "measurements"
    annotation_type = np.ndarray | None
    parameter_default = None


class _DisplayMeasurementTablesRuntimeParameter(KeywordRuntimeParameter):
    """Prior measurement tables selected by the module artifact contract."""

    parameter_name = "measurement_tables"
    annotation_type = tuple[MeasurementTable, ...]
    parameter_default = ()


class DisplayMeasurementInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind only declaration-selected prior measurements to display callables."""

    binds_without_declared_inputs = True
    supported_non_object_input_kinds = frozenset({MeasurementsArtifactType})

    @classmethod
    def prior_measurement_artifact_inputs(
        cls,
        module,
        *,
        step_context,
        direct_inputs,
    ):
        inputs = super().prior_measurement_artifact_inputs(
            module,
            step_context=step_context,
            direct_inputs=direct_inputs,
        )
        if not inputs:
            raise ValueError(
                f"{cls.module_name} requires at least one selected prior "
                "measurement feature."
            )
        return inputs

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        bound = super().bind_runtime_inputs(request)
        tables = request.declared_measurement_tables()
        if not tables:
            raise ValueError(
                f"{request.adapter.request.require_callable_contract().module_name} requires declared prior "
                "measurement artifacts."
            )
        bound[_DisplayMeasurementTablesRuntimeParameter.require_parameter_name()] = (
            tables
        )
        return bound


def _measurement_vector(
    measurement_tables: tuple[MeasurementTable, ...],
    *,
    feature_name: str | None,
    object_name: str | None,
) -> np.ndarray:
    """Resolve one exact numeric feature vector from declared measurement tables."""

    if feature_name is None:
        raise ValueError("Display modules require a selected measurement feature.")
    if not measurement_tables:
        raise ValueError(
            "Display modules require declared prior measurement artifacts."
        )
    values_by_label, positional_values = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    ).value_index(measurement_tables)
    values = (
        tuple(values_by_label[label] for label in sorted(values_by_label))
        if values_by_label
        else tuple(positional_values)
    )
    return np.asarray(values, dtype=float)


def _measurement_values_by_slice(
    measurement_tables: tuple[MeasurementTable, ...],
    *,
    feature_name: str,
    object_name: str | None,
) -> dict[int, tuple[object, ...]]:
    """Resolve feature values by their declared runtime slice domain."""

    query = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )
    values_by_slice: dict[int, list[object]] = {}
    axis = MeasurementRowAxisField.SLICE_INDEX
    for row in measurement_rows(measurement_tables):
        value = query.row_value(row)
        if value is None or value == "":
            continue
        row_mapping = measurement_row_mapping(row)
        slice_index = measurement_axis_integer_value(row_mapping.get(axis.value), axis)
        if slice_index is None:
            raise ValueError(
                f"DisplayPlatemap feature {feature_name!r} requires explicit "
                f"{axis.value!r} row ownership."
            )
        values_by_slice.setdefault(slice_index, []).append(value)
    if not values_by_slice:
        raise ValueError(
            f"Could not resolve measurement feature {feature_name!r} from "
            f"{query.table_summaries(measurement_tables)!r}."
        )
    return {
        slice_index: tuple(values) for slice_index, values in values_by_slice.items()
    }


def _single_slice_value(
    values_by_slice: Mapping[int, tuple[object, ...]],
    slice_index: int,
    feature_name: str,
) -> object:
    values = values_by_slice[slice_index]
    if len(values) != 1:
        raise ValueError(
            f"DisplayPlatemap feature {feature_name!r} has {len(values)} values "
            f"for slice {slice_index}; expected exactly one image-level value."
        )
    return values[0]


def _parse_float_range(value: str) -> tuple[float, float]:
    parts = tuple(part.strip() for part in value.split(","))
    if len(parts) != 2 or not all(parts):
        raise ValueError(
            "CellProfiler float ranges require exactly two comma-separated values, "
            f"got {value!r}."
        )
    return float(parts[0]), float(parts[1])


class CellProfilerFloatRangeSettingBinding(SettingToKeywordBinding):
    """One CellProfiler FloatRange setting serialized as one exact CSV row."""

    def records_from_kwargs(
        self,
        kwargs: Mapping[str, object],
    ) -> tuple[ModuleSetting, ...]:
        parameter_name = self.require_parameter_name()
        if parameter_name not in kwargs:
            return ()
        value = kwargs[parameter_name]
        if not isinstance(value, (tuple, list)) or len(value) != 2:
            raise ValueError(
                f"{parameter_name} requires exactly two float values, got {value!r}."
            )
        return (
            ModuleSetting(
                setting_names(self.setting_name)[0],
                f"{float(value[0])},{float(value[1])}",
            ),
        )


class SavedImageContents(Enum):
    IMAGE = "image"
    AXES = "axes"
    FIGURE = "figure"


@dataclass(frozen=True)
class DisplayDataOnImageRequest:
    """Typed request for rendering CellProfiler measurements onto an image."""

    image: np.ndarray
    labels: Optional[ObjectLabelValue]
    measurements: Optional[np.ndarray]
    objects_or_image: ObjectsOrImage
    display_mode: DisplayMode
    wants_background_image: bool
    text_color: Tuple[float, float, float]
    font_size: int
    decimals: int
    offset: int
    colormap: str
    color_map_scale_choice: ColorMapScale
    color_map_scale_min: float
    color_map_scale_max: float
    use_scientific_notation: bool
    image_measurement_value: Optional[float]
    center_x: Optional[np.ndarray]
    center_y: Optional[np.ndarray]


@dataclass(frozen=True)
class DisplayDataOnImageRenderer:
    """Render CellProfiler DisplayDataOnImage requests."""

    request: DisplayDataOnImageRequest

    @property
    def text_color_bgr(self) -> Tuple[int, int, int]:
        return (
            int(self.request.text_color[2] * 255),
            int(self.request.text_color[1] * 255),
            int(self.request.text_color[0] * 255),
        )

    def render_slice(self) -> np.ndarray:
        """Process a single 2D slice."""
        from skimage.measure import regionprops
        import cv2

        request = self.request
        image = request.image
        labels = request.labels
        measurements = request.measurements
        h, w = image.shape[:2]

        # Prepare background
        if request.wants_background_image:
            if image.ndim == 2:
                # Grayscale to RGB
                background = np.stack([image, image, image], axis=-1)
            else:
                background = image.copy()
        else:
            background = np.zeros((h, w, 3), dtype=np.float32)

        # Normalize to 0-1 range if needed
        if background.max() > 1.0:
            background = background / 255.0
        background = background.astype(np.float32)

        if request.objects_or_image == ObjectsOrImage.IMAGE:
            # Display single image measurement at center
            if request.image_measurement_value is not None:
                x = w // 2
                y = h // 2
                x_offset = np.random.uniform(-1.0, 1.0)
                y_offset = np.sqrt(1 - x_offset**2)
                x = int(x + request.offset * x_offset)
                y = int(y + request.offset * y_offset)

                if request.use_scientific_notation:
                    text = f"{request.image_measurement_value:.{request.decimals}e}"
                else:
                    text = f"{request.image_measurement_value:.{request.decimals}f}"

                # Convert to uint8 for cv2
                output = (background * 255).astype(np.uint8)
                font_scale = request.font_size / 20.0
                cv2.putText(
                    output,
                    text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    self.text_color_bgr,
                    1,
                    cv2.LINE_AA,
                )
                return output.astype(np.float32) / 255.0

        elif request.objects_or_image == ObjectsOrImage.OBJECTS and labels is not None:
            labels = object_label_dense_array(labels, dtype=np.int32)
            if labels.ndim != 2 or labels.shape != image.shape[:2]:
                raise ValueError(
                    "DisplayDataOnImage requires labels already projected into the "
                    f"display image plane; got labels {labels.shape!r} and image "
                    f"{image.shape!r}."
                )
            if request.display_mode == DisplayMode.COLOR and measurements is not None:
                # Color map mode
                from matplotlib import cm

                # Get colormap
                cmap = cm.get_cmap(request.colormap)

                # Determine scale
                valid_measurements = (
                    measurements[~np.isnan(measurements)]
                    if len(measurements) > 0
                    else np.array([0, 1])
                )
                if request.color_map_scale_choice == ColorMapScale.MANUAL:
                    vmin, vmax = (
                        request.color_map_scale_min,
                        request.color_map_scale_max,
                    )
                else:
                    vmin = (
                        valid_measurements.min() if len(valid_measurements) > 0 else 0
                    )
                    vmax = (
                        valid_measurements.max() if len(valid_measurements) > 0 else 1
                    )

                if vmax == vmin:
                    vmax = vmin + 1

                # Normalize measurements
                normalized = (measurements - vmin) / (vmax - vmin)
                normalized = np.clip(normalized, 0, 1)

                # Create colored output
                output = background.copy()
                if output.ndim == 2:
                    output = np.stack([output, output, output], axis=-1)

                # Apply colors to each labeled region
                for i, val in enumerate(normalized):
                    if not np.isnan(val):
                        color = cmap(val)[:3]
                        mask = labels == (i + 1)
                        for c in range(3):
                            output[:, :, c] = np.where(
                                mask,
                                output[:, :, c] * 0.5 + color[c] * 0.5,
                                output[:, :, c],
                            )

                return output

            else:
                # Text mode
                # Get object centers
                if request.center_x is None or request.center_y is None:
                    props = regionprops(labels)
                    centers = [(p.centroid[1], p.centroid[0]) for p in props]
                else:
                    centers = list(zip(request.center_x, request.center_y))

                # Convert to uint8 for cv2
                output = (background * 255).astype(np.uint8)
                font_scale = request.font_size / 20.0

                if measurements is not None:
                    for idx, (cx, cy) in enumerate(centers):
                        if idx < len(measurements):
                            val = measurements[idx]
                            if np.isnan(val):
                                continue

                            # Apply offset
                            x_off = np.random.uniform(-1.0, 1.0)
                            y_off = np.sqrt(1 - x_off**2)
                            x = int(cx + request.offset * x_off)
                            y = int(cy + request.offset * y_off)

                            if request.use_scientific_notation:
                                text = f"{val:.{request.decimals}e}"
                            else:
                                text = f"{val:.{request.decimals}f}"

                            cv2.putText(
                                output,
                                text,
                                (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                self.text_color_bgr,
                                1,
                                cv2.LINE_AA,
                            )

                return output.astype(np.float32) / 255.0

        return background


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@runtime_bound_parameters(_DisplayMeasurementsRuntimeParameter)
def display_data_on_image(
    image: np.ndarray,
    labels: Optional[ObjectLabelValue] = None,
    measurements: Optional[np.ndarray] = None,
    measurement_feature: Optional[str] = None,
    objects_or_image: ObjectsOrImage = ObjectsOrImage.OBJECTS,
    display_mode: DisplayMode = DisplayMode.TEXT,
    wants_background_image: bool = True,
    text_color: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    font_size: int = 10,
    decimals: int = 2,
    offset: int = 0,
    colormap: str = "viridis",
    color_map_scale_choice: ColorMapScale = ColorMapScale.USE_MEASUREMENT_RANGE,
    color_map_scale_min: float = 0.0,
    color_map_scale_max: float = 1.0,
    use_scientific_notation: bool = False,
    image_measurement_value: Optional[float] = None,
    center_x: Optional[np.ndarray] = None,
    center_y: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Display measurement data on top of an image.

    This function overlays measurement values on an image, either as text
    annotations at object centers or as a color map applied to object regions.

    Args:
        image: Input image, shape (D, H, W) or (H, W)
        labels: Optional label image for objects, shape matching image
        measurements: Optional array of measurement values per object
        measurement_feature: CellProfiler feature selected for runtime measurement lookup
        objects_or_image: Whether displaying object or image measurements
        display_mode: TEXT for numeric values, COLOR for colormap overlay
        wants_background_image: Whether to show background image or black
        text_color: RGB tuple for text color (0-1 range)
        font_size: Font size in points
        decimals: Number of decimal places to display
        offset: Pixel offset for text placement
        colormap: Name of matplotlib colormap
        color_map_scale_choice: Use measurement range or manual scale
        color_map_scale_min: Manual minimum for color scale
        color_map_scale_max: Manual maximum for color scale
        use_scientific_notation: Display values in scientific notation
        image_measurement_value: Single value for image-level measurement
        center_x: X coordinates of object centers
        center_y: Y coordinates of object centers

    Returns:
        RGB image with measurements displayed, shape (D, H, W, 3) or (H, W, 3)
    """
    request = DisplayDataOnImageRequest(
        image=np.asarray(image_payload_data(image)),
        labels=labels,
        measurements=measurements,
        objects_or_image=objects_or_image,
        display_mode=display_mode,
        wants_background_image=wants_background_image,
        text_color=text_color,
        font_size=font_size,
        decimals=decimals,
        offset=offset,
        colormap=colormap,
        color_map_scale_choice=color_map_scale_choice,
        color_map_scale_min=color_map_scale_min,
        color_map_scale_max=color_map_scale_max,
        use_scientific_notation=use_scientific_notation,
        image_measurement_value=image_measurement_value,
        center_x=center_x,
        center_y=center_y,
    )

    output = DisplayDataOnImageRenderer(request).render_slice()
    return with_image_payload_data(
        image,
        output,
        metadata=replace(image_payload_metadata(image), source_channel_axis=-1),
    )


class DensityPlotScaleType(Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class DensityPlotData:
    """Density plot histogram data for visualization."""

    slice_index: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_scale: str
    y_scale: str
    gridsize: int
    num_points: int
    colorbar_scale: str
    colormap: str
    title: str
    histogram_counts: str


@numpy(contract=ProcessingContract.FLEXIBLE)
@runtime_bound_parameters(_DisplayMeasurementTablesRuntimeParameter)
def display_density_plot(
    image: np.ndarray,
    x_object_name: str | None = None,
    x_measurement_feature: str | None = None,
    y_object_name: str | None = None,
    y_measurement_feature: str | None = None,
    gridsize: int = 100,
    x_scale: DensityPlotScaleType = DensityPlotScaleType.LINEAR,
    y_scale: DensityPlotScaleType = DensityPlotScaleType.LINEAR,
    colorbar_scale: DensityPlotScaleType = DensityPlotScaleType.LINEAR,
    colormap: str = "jet",
    title: str = "",
    *,
    measurement_tables: tuple[MeasurementTable, ...] = (),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """
    Compute 2D density histogram from two measurement arrays.

    This function takes two measurement arrays stacked along dimension 0
    and computes a 2D histogram (density plot) representation.

    Args:
        image: Shape (2, N) where image[0] contains X measurements and
               image[1] contains Y measurements. N is the number of objects.
        gridsize: Number of grid regions on each axis (1-1000). Higher values
                  increase resolution.
        x_scale: Scale for X-axis - linear or log (base 10).
        y_scale: Scale for Y-axis - linear or log (base 10).
        colorbar_scale: Scale for colorbar - linear or log (base 10).
        colormap: Colormap for the density plot visualization.
        title: Optional title for the plot.

    Returns:
        Tuple of:
        - 2D histogram array of shape (gridsize, gridsize) representing density
        - DensityPlotData with metadata about the plot"""
    x_data = _measurement_vector(
        measurement_tables,
        feature_name=x_measurement_feature,
        object_name=x_object_name,
    )
    y_data = _measurement_vector(
        measurement_tables,
        feature_name=y_measurement_feature,
        object_name=y_object_name,
    )
    pair_count = min(len(x_data), len(y_data))
    x_data = x_data[:pair_count]
    y_data = y_data[:pair_count]

    # Remove NaN and infinite values
    valid_mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]

    if len(x_data) == 0:
        histogram = np.zeros((gridsize, gridsize), dtype=np.float32)
        result = DensityPlotData(
            slice_index=0,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            x_scale=x_scale.value,
            y_scale=y_scale.value,
            gridsize=gridsize,
            num_points=0,
            colorbar_scale=colorbar_scale.value,
            colormap=colormap,
            title=title,
            histogram_counts=",".join(str(value) for value in histogram.ravel()),
        )
        return image, DataclassMeasurementColumnarRows(
            (result,), row_type=DensityPlotData
        )

    # Apply log transform if requested
    if x_scale == DensityPlotScaleType.LOG:
        # Filter out non-positive values for log scale
        pos_mask = x_data > 0
        x_data = x_data[pos_mask]
        y_data = y_data[pos_mask]
        if len(x_data) > 0:
            x_data = np.log10(x_data)

    if y_scale == DensityPlotScaleType.LOG:
        # Filter out non-positive values for log scale
        pos_mask = y_data > 0
        x_data = x_data[pos_mask]
        y_data = y_data[pos_mask]
        if len(y_data) > 0:
            y_data = np.log10(y_data)

    if len(x_data) == 0:
        histogram = np.zeros((gridsize, gridsize), dtype=np.float32)
        result = DensityPlotData(
            slice_index=0,
            x_min=0.0,
            x_max=1.0,
            y_min=0.0,
            y_max=1.0,
            x_scale=x_scale.value,
            y_scale=y_scale.value,
            gridsize=gridsize,
            num_points=0,
            colorbar_scale=colorbar_scale.value,
            colormap=colormap,
            title=title,
            histogram_counts=",".join(str(value) for value in histogram.ravel()),
        )
        return image, DataclassMeasurementColumnarRows(
            (result,), row_type=DensityPlotData
        )

    # Compute data ranges
    x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
    y_min, y_max = float(np.min(y_data)), float(np.max(y_data))

    # Handle edge case where min == max
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5

    # Compute 2D histogram
    histogram, _x_edges, _y_edges = np.histogram2d(
        x_data, y_data, bins=gridsize, range=[[x_min, x_max], [y_min, y_max]]
    )

    # Apply log transform to histogram counts if requested
    if colorbar_scale == DensityPlotScaleType.LOG:
        # Add 1 to avoid log(0), then take log
        histogram = np.log10(histogram + 1)

    # Normalize to 0-1 range for visualization
    if histogram.max() > 0:
        histogram = histogram / histogram.max()

    histogram = histogram.astype(np.float32)

    result = DensityPlotData(
        slice_index=0,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        x_scale=x_scale.value,
        y_scale=y_scale.value,
        gridsize=gridsize,
        num_points=len(x_data),
        colorbar_scale=colorbar_scale.value,
        colormap=colormap,
        title=title,
        histogram_counts=",".join(str(value) for value in histogram.ravel()),
    )
    return image, DataclassMeasurementColumnarRows((result,), row_type=DensityPlotData)


class AxisScale(Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class HistogramResult:
    """Histogram computation results."""

    slice_index: int
    bin_count: int
    data_min: float
    data_max: float
    data_mean: float
    data_std: float
    data_median: float
    total_count: int
    # Histogram bin edges and counts stored as comma-separated strings for CSV
    bin_edges: str
    bin_counts: str


@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(_DisplayMeasurementTablesRuntimeParameter)
def display_histogram(
    image: np.ndarray,
    object_name: str | None = None,
    measurement_feature: str | None = None,
    num_bins: int = 100,
    x_scale: AxisScale = AxisScale.LINEAR,
    y_scale: AxisScale = AxisScale.LINEAR,
    title: str = "",
    use_x_bounds: bool = False,
    x_bounds: tuple[float, float] = (0.0, 1.0),
    *,
    measurement_tables: tuple[MeasurementTable, ...] = (),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """
    Compute histogram statistics from object measurements.

    This function extracts measurements from labeled objects and computes
    histogram statistics. The actual histogram visualization is handled
    by the pipeline's visualization layer.

    Args:
        image: Input intensity image, shape (H, W)
        object_name: Object subject selected by the CellProfiler module
        measurement_feature: Prior object measurement selected for plotting
        num_bins: Number of histogram bins (1-1000)
        x_scale: Scale for X-axis (linear or log)
        y_scale: Scale for Y-axis (linear or log)
        use_x_bounds: Whether to apply min/max bounds to X-axis
        x_bounds: Minimum and maximum X-axis values

    Returns:
        Tuple of (original image, histogram results)"""
    del title
    values = _measurement_vector(
        measurement_tables,
        feature_name=measurement_feature,
        object_name=object_name,
    )

    # Apply log transform if needed for x-axis
    if x_scale == AxisScale.LOG:
        # Avoid log(0) by filtering out zeros and negatives
        values = values[values > 0]
        if len(values) > 0:
            values = np.log(values)

    # Apply X bounds if specified
    if use_x_bounds and len(values) > 0:
        x_min, x_max = x_bounds
        values = values[values >= x_min]
        values = values[values <= x_max]

    # Handle empty values after filtering
    if len(values) == 0:
        result = HistogramResult(
            slice_index=0,
            bin_count=num_bins,
            data_min=0.0,
            data_max=0.0,
            data_mean=0.0,
            data_std=0.0,
            data_median=0.0,
            total_count=0,
            bin_edges="",
            bin_counts="",
        )
        return image, DataclassMeasurementColumnarRows(
            (result,), row_type=HistogramResult
        )

    # Compute histogram
    counts, bin_edges = np.histogram(values, bins=num_bins)

    # Apply log transform to counts if y-scale is log
    if y_scale == AxisScale.LOG:
        counts = np.log1p(counts)  # log(1 + x) to handle zeros

    # Compute statistics
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    data_mean = float(np.mean(values))
    data_std = float(np.std(values))
    data_median = float(np.median(values))

    # Convert arrays to comma-separated strings for CSV storage
    bin_edges_str = ",".join([f"{x:.6f}" for x in bin_edges])
    bin_counts_str = ",".join([f"{x:.6f}" for x in counts])

    result = HistogramResult(
        slice_index=0,
        bin_count=num_bins,
        data_min=data_min,
        data_max=data_max,
        data_mean=data_mean,
        data_std=data_std,
        data_median=data_median,
        total_count=len(values),
        bin_edges=bin_edges_str,
        bin_counts=bin_counts_str,
    )

    return image, DataclassMeasurementColumnarRows((result,), row_type=HistogramResult)


class AggregationMethod(Enum):
    AVG = ("avg", np.mean)
    MEDIAN = ("median", np.median)
    STDEV = ("stdev", np.std)
    CV = ("cv%", None)

    def __init__(self, label: str, helper: object | None) -> None:
        self._value_ = label
        self._helper = helper

    def aggregate(self, values: np.ndarray) -> float:
        if self is AggregationMethod.CV:
            mean_value = np.mean(values)
            if mean_value == 0:
                return np.nan
            return float(np.std(values) / mean_value)
        if self._helper is None:
            raise NotImplementedError(
                f"{type(self).__name__}.{self.name} does not declare an aggregation helper."
            )
        return float(self._helper(values))


class PlateType(Enum):
    PLATE_96 = "96"
    PLATE_384 = "384"


class WellFormat(Enum):
    NAME = "Well name"
    ROWCOL = "Row & Column"


class ObjectOrImage(Enum):
    OBJECTS = "Object"
    IMAGE = "Image"


@dataclass(frozen=True, slots=True)
class PlatemapData:
    """Aggregated measurement data for plate map visualization."""

    plate: str
    well: str
    row: str
    column: str
    value: float
    measurement_name: str
    aggregation_method: str
    object_name: str


@dataclass(frozen=True, slots=True)
class PlatemapSummary:
    """Summary statistics for the entire plate map."""

    plate: str
    measurement_name: str
    aggregation_method: str
    min_value: float
    max_value: float
    mean_value: float
    well_count: int


class PlateDimensionStrategy(
    EnumKeyedStrategyMixin[PlateType],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal dimensions for one CellProfiler plate type."""

    __registry_key__ = "plate_type_label"
    __skip_if_no_key__ = True

    plate_type: ClassVar[PlateType | None] = None
    plate_type_label: ClassVar[str | None] = None
    __enum_member_attr__ = "plate_type"
    __enum_label_attr__ = "plate_type_label"

    @classmethod
    def for_plate_type(cls, plate_type: PlateType) -> "PlateDimensionStrategy":
        return cls.for_enum_member(plate_type)

    @abstractmethod
    def dimensions(self) -> Tuple[int, int]:
        """Return row and column count for this plate type."""


class Plate96DimensionStrategy(PlateDimensionStrategy):
    """96-well plate dimensions."""

    plate_type = PlateType.PLATE_96

    def dimensions(self) -> Tuple[int, int]:
        return 8, 12


class Plate384DimensionStrategy(PlateDimensionStrategy):
    """384-well plate dimensions."""

    plate_type = PlateType.PLATE_384

    def dimensions(self) -> Tuple[int, int]:
        return 16, 24


def _parse_well_name(well: str) -> Tuple[str, str]:
    """Parse well name like 'A01' into row 'A' and column '01'."""
    if len(well) >= 2:
        row = well[0].upper()
        col = well[1:]
        return row, col
    return "", ""


def _aggregate_values(values: np.ndarray, method: AggregationMethod) -> float:
    """Aggregate array of values using specified method."""
    if len(values) == 0:
        return np.nan
    return method.aggregate(values)


@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(_DisplayMeasurementTablesRuntimeParameter)
def display_platemap(
    image: np.ndarray,
    objects_or_image: ObjectOrImage = ObjectOrImage.OBJECTS,
    object_name: str | None = None,
    measurement_feature: str | None = None,
    plate_metadata_feature: str = "Metadata_Plate",
    plate_type: PlateType = PlateType.PLATE_96,
    well_format: WellFormat = WellFormat.NAME,
    well_metadata_feature: str = "Metadata_Well",
    well_row_metadata_feature: str = "Metadata_WellRow",
    well_column_metadata_feature: str = "Metadata_WellCol",
    agg_method: AggregationMethod = AggregationMethod.AVG,
    title: str = "",
    *,
    measurement_tables: tuple[MeasurementTable, ...] = (),
) -> tuple[np.ndarray, ColumnarRows]:
    """
    Aggregate measurements by well for plate map visualization.

    This function aggregates per-image or per-object measurements into
    per-well values suitable for plate map display. The actual visualization
    is handled by the OpenHCS frontend.

    Args:
        image: Input image array (D, H, W) - passed through unchanged
        objects_or_image: Whether measurements are from objects or images
        object_name: Name of object type being measured
        measurement_feature: Name of the measurement being displayed
        plate_type: Format of multiwell plate (96 or 384)
        well_format: How well location is specified (name or row/column)
        agg_method: How to aggregate multiple values per well
        title: Optional title for the plot

    Returns:
        Tuple of (image, one nominal platemap measurement table)"""
    del title
    measurement_values_by_slice = _measurement_values_by_slice(
        measurement_tables,
        feature_name=measurement_feature,
        object_name=(
            object_name if objects_or_image is ObjectOrImage.OBJECTS else None
        ),
    )
    plate_metadata_by_slice = _measurement_values_by_slice(
        measurement_tables,
        feature_name=plate_metadata_feature,
        object_name=None,
    )
    platemap_entries: list[PlatemapData] = []
    platemap_summaries: list[PlatemapSummary] = []

    # Construct well identifiers
    if well_format is WellFormat.NAME:
        well_metadata_by_slice = _measurement_values_by_slice(
            measurement_tables,
            feature_name=well_metadata_feature,
            object_name=None,
        )
    else:
        well_rows_by_slice = _measurement_values_by_slice(
            measurement_tables,
            feature_name=well_row_metadata_feature,
            object_name=None,
        )
        well_columns_by_slice = _measurement_values_by_slice(
            measurement_tables,
            feature_name=well_column_metadata_feature,
            object_name=None,
        )
        if well_rows_by_slice.keys() != well_columns_by_slice.keys():
            raise ValueError(
                "DisplayPlatemap well-row and well-column metadata must declare "
                "the same slice domain."
            )
        well_metadata_by_slice = {
            slice_index: (
                f"{_single_slice_value(well_rows_by_slice, slice_index, well_row_metadata_feature)}"
                f"{_single_slice_value(well_columns_by_slice, slice_index, well_column_metadata_feature)}",
            )
            for slice_index in well_rows_by_slice
        }

    slice_domain = measurement_values_by_slice.keys()
    if (
        slice_domain != plate_metadata_by_slice.keys()
        or slice_domain != well_metadata_by_slice.keys()
    ):
        raise ValueError(
            "DisplayPlatemap measurement, plate, and well features must declare "
            "the same slice domain."
        )

    # Build dictionary mapping plate -> well -> list of values
    pm_dict: Dict[str, Dict[str, List[float]]] = {}

    for slice_index in sorted(slice_domain):
        plate = str(
            _single_slice_value(
                plate_metadata_by_slice,
                slice_index,
                plate_metadata_feature,
            )
        )
        well = str(
            _single_slice_value(
                well_metadata_by_slice,
                slice_index,
                well_metadata_feature,
            )
        )
        values = tuple(
            float(value) for value in measurement_values_by_slice[slice_index]
        )

        if plate not in pm_dict:
            pm_dict[plate] = {}

        if well not in pm_dict[plate]:
            pm_dict[plate][well] = []

        pm_dict[plate][well].extend(values)

    # Aggregate values and create output entries
    for plate, well_dict in pm_dict.items():
        all_aggregated = []

        for well, values in well_dict.items():
            values_arr = np.array(values)
            aggregated = _aggregate_values(values_arr, agg_method)
            all_aggregated.append(aggregated)

            row, col = _parse_well_name(well)

            platemap_entries.append(
                PlatemapData(
                    plate=plate,
                    well=well,
                    row=row,
                    column=col,
                    value=aggregated,
                    measurement_name=measurement_feature,
                    aggregation_method=agg_method.value,
                    object_name=(
                        object_name
                        if objects_or_image == ObjectOrImage.OBJECTS
                        else "Image"
                    ),
                )
            )

        # Create summary for this plate
        if all_aggregated:
            valid_values = [v for v in all_aggregated if not np.isnan(v)]
            if valid_values:
                platemap_summaries.append(
                    PlatemapSummary(
                        plate=plate,
                        measurement_name=measurement_feature,
                        aggregation_method=agg_method.value,
                        min_value=float(np.min(valid_values)),
                        max_value=float(np.max(valid_values)),
                        mean_value=float(np.mean(valid_values)),
                        well_count=len(valid_values),
                    )
                )

    return image, ConcatenatedColumnarRows(
        (
            DataclassMeasurementColumnarRows(
                tuple(platemap_entries),
                row_type=PlatemapData,
            ),
            DataclassMeasurementColumnarRows(
                tuple(platemap_summaries),
                row_type=PlatemapSummary,
            ),
        )
    )


class MeasurementSource(Enum):
    IMAGE = "Image"
    OBJECT = "Object"


class ScatterPlotScaleType(Enum):
    LINEAR = "linear"
    LOG = "log"


@dataclass(frozen=True, slots=True)
class ScatterPlotData:
    """Data structure for scatter plot output."""

    slice_index: int
    x_values: str
    y_values: str
    x_label: str
    y_label: str
    x_scale: str
    y_scale: str
    title: str
    point_count: int


@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(_DisplayMeasurementTablesRuntimeParameter)
def display_scatter_plot(
    image: np.ndarray,
    x_source: MeasurementSource = MeasurementSource.IMAGE,
    x_object_name: str | None = None,
    x_measurement_feature: str | None = None,
    y_source: MeasurementSource = MeasurementSource.IMAGE,
    y_object_name: str | None = None,
    y_measurement_feature: str | None = None,
    x_scale: ScatterPlotScaleType = ScatterPlotScaleType.LINEAR,
    y_scale: ScatterPlotScaleType = ScatterPlotScaleType.LINEAR,
    title: str = "",
    *,
    measurement_tables: tuple[MeasurementTable, ...] = (),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """
    Extract scatter plot data from two measurement arrays.

    This function prepares data for scatter plot visualization by pairing
    corresponding measurements from two arrays. The actual visualization
    is handled by the OpenHCS frontend.

    Args:
        image: Input image array (H, W), passed through unchanged
        x_source: Source type for x measurements (Image or Object)
        y_source: Source type for y measurements (Image or Object)
        x_scale: Scale type for x-axis (linear or log)
        y_scale: Scale type for y-axis (linear or log)
        title: Plot title (empty string for auto-generated title)

    Returns:
        Tuple of (original image, scatter plot data)"""
    import json

    x_vals = _measurement_vector(
        measurement_tables,
        feature_name=x_measurement_feature,
        object_name=(x_object_name if x_source is MeasurementSource.OBJECT else None),
    )
    y_vals = _measurement_vector(
        measurement_tables,
        feature_name=y_measurement_feature,
        object_name=(y_object_name if y_source is MeasurementSource.OBJECT else None),
    )

    # Handle mismatched lengths - take minimum length
    min_len = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]

    # Filter out NaN and None values
    valid_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]

    # Apply log transform if needed (filter out non-positive values)
    if x_scale == ScatterPlotScaleType.LOG:
        positive_x = x_vals > 0
        x_vals = x_vals[positive_x]
        y_vals = y_vals[positive_x]

    if y_scale == ScatterPlotScaleType.LOG:
        positive_y = y_vals > 0
        x_vals = x_vals[positive_y]
        y_vals = y_vals[positive_y]

    # Generate title if not provided
    plot_title = (
        title if title else f"{x_measurement_feature} vs {y_measurement_feature}"
    )

    # Create scatter plot data
    scatter_data = ScatterPlotData(
        slice_index=0,
        x_values=json.dumps(x_vals.tolist()),
        y_values=json.dumps(y_vals.tolist()),
        x_label=x_measurement_feature,
        y_label=y_measurement_feature,
        x_scale=x_scale.value,
        y_scale=y_scale.value,
        title=plot_title,
        point_count=len(x_vals),
    )

    return image, DataclassMeasurementColumnarRows(
        (scatter_data,),
        row_type=ScatterPlotData,
    )


class DisplayModule(CellProfilerModule):
    """Parent for CellProfiler display/export debug sections."""


class DisplayDataOnImageSpecialInputPolicy:
    """Resolve display annotations from object labels and measurement tables."""

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        object_inputs = request.object_inputs
        object_spec = object_inputs[0]
        special_inputs = super().bind_runtime_inputs(request)
        if len(special_inputs) != 1:
            raise ValueError(
                f"{request.adapter.request.require_callable_contract().module_name} requires one special object-label "
                f"parameter, got {tuple(special_inputs)!r}."
            )
        labels_parameter, labels = next(iter(special_inputs.items()))
        feature_name = request.require_string_kwarg("measurement_feature")
        return {
            labels_parameter: labels,
            _DisplayMeasurementsRuntimeParameter.require_parameter_name(): CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=feature_name,
                labels=labels,
            )
            .vector()
            .runtime_value,
        }


class DisplayDataOnImageModule(
    DisplayDataOnImageSpecialInputPolicy,
    PriorMeasurementArtifactInputModule,
    ObjectArtifactInputModule,
    DisplayModule,
):
    module_name = "DisplayDataOnImage"
    function_name = "display_data_on_image"
    validated = True
    confidence = 1.0
    image_input_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the image on which to display the measurements"
    )
    object_input_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the input objects"
    )
    image_output_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Name the output image that has the measurements displayed"
    )
    object_or_image_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Display object or image measurements?"
    )
    measurement_feature_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Measurement to display"
    )
    display_mode_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Display mode"
    )
    background_image_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Display background image?"
    )
    text_color_setting: ClassVar[SettingNameFamily] = SettingNameFamily("Text color")
    font_size_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Font size (points)",
        aliases=("Font size",),
    )
    decimals_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Number of decimals"
    )
    annotation_offset_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Annotation offset (in pixels)",
        aliases=("Annotation offset",),
    )
    colormap_setting: ClassVar[SettingNameFamily] = SettingNameFamily("Color map")
    color_map_scale_choice_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Color map scale"
    )
    color_map_range_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Color map range"
    )
    scientific_notation_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Use scientific notation?"
    )
    font_setting: ClassVar[SettingNameFamily] = SettingNameFamily("Font")
    font_weight_setting: ClassVar[SettingNameFamily] = SettingNameFamily("Font weight")
    image_elements_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Image elements to save"
    )
    object_or_image_parameter: ClassVar[str] = "objects_or_image"
    measurement_feature_parameter: ClassVar[str] = "measurement_feature"
    default_object_or_image: ClassVar[str] = "Object"
    measurement_feature_binding = MeasurementFeatureSettingBinding(
        measurement_feature_setting,
        measurement_feature_parameter,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding.input(image_input_setting, ImageArtifactType),
        SettingToKeywordBinding.input(
            object_input_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
        ),
        SettingToKeywordBinding.output(image_output_setting, ImageArtifactType),
        SettingToKeywordBinding(
            object_or_image_setting,
            object_or_image_parameter,
        ),
        measurement_feature_binding,
        SettingToKeywordBinding(display_mode_setting, "display_mode"),
        SettingToKeywordBinding(
            background_image_setting,
            "wants_background_image",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(text_color_setting, "text_color"),
        SettingToKeywordBinding(
            font_size_setting,
            "font_size",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            decimals_setting,
            "decimals",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            annotation_offset_setting,
            "offset",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(colormap_setting, "colormap"),
        SettingToKeywordBinding(
            color_map_scale_choice_setting,
            "color_map_scale_choice",
        ),
        SettingToKeywordBinding(
            scientific_notation_setting,
            "use_scientific_notation",
            parse_cellprofiler_bool,
        ),
    )
    ignored_settings = (image_elements_setting,)

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        unsupported_font_settings = tuple(
            (setting_names(setting)[0], value)
            for setting in (cls.font_setting, cls.font_weight_setting)
            if (value := optional_setting_value(module, setting)) is not None
        )
        if unsupported_font_settings:
            raise NotImplementedError(
                "DisplayDataOnImage font settings are not supported by the public "
                "display_data_on_image callable: "
                f"{unsupported_font_settings!r}."
            )
        if cls.measurement_feature_parameter not in bound.kwargs:
            raise ValueError("DisplayDataOnImage requires a measurement feature.")
        kwargs = dict(bound.kwargs)
        color_map_range = optional_setting_value(module, cls.color_map_range_setting)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        if color_map_range is not None:
            range_parts = tuple(
                part.strip() for part in color_map_range.split(",") if part.strip()
            )
            if len(range_parts) != 2:
                raise ValueError(
                    "DisplayDataOnImage color-map range must contain exactly two "
                    f"numeric values, got {color_map_range!r}."
                )
            kwargs["color_map_scale_min"] = float(range_parts[0])
            kwargs["color_map_scale_max"] = float(range_parts[1])
            for setting_name in setting_names(cls.color_map_range_setting):
                unmapped_kwargs.pop(
                    normalize_cellprofiler_setting_name(setting_name),
                    None,
                )
        if "text_color" in kwargs:
            from openhcs.processing.backends.cellprofiler.color import coerce_rgb_color

            kwargs["text_color"] = coerce_rgb_color(kwargs["text_color"])
        if kwargs.get("colormap") == "Default":
            kwargs["colormap"] = "viridis"
        bound = BoundModuleSettings(
            kwargs,
            unmapped_kwargs,
            bound.setting_coverage,
        )
        if cls.object_or_image_parameter in bound.kwargs:
            return bound
        return bound.with_kwargs(
            {cls.object_or_image_parameter: cls.default_object_or_image}
        )


class DisplayDensityPlotModule(
    DisplayMeasurementInputPolicy,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    DisplayModule,
):
    module_name = "DisplayDensityPlot"
    function_name = "display_density_plot"
    validated = True
    confidence = 1.0
    x_object_setting = SettingNameFamily("Select the object to display on the X-axis")
    x_measurement_setting = SettingNameFamily(
        "Select the object measurement to plot on the X-axis"
    )
    y_object_setting = SettingNameFamily("Select the object to display on the Y-axis")
    y_measurement_setting = SettingNameFamily(
        "Select the object measurement to plot on the Y-axis"
    )
    x_measurement_binding = MeasurementFeatureSettingBinding(
        x_measurement_setting,
        "x_measurement_feature",
    )
    y_measurement_binding = MeasurementFeatureSettingBinding(
        y_measurement_setting,
        "y_measurement_feature",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            x_object_setting,
            "x_object_name",
            normalized_symbol_name,
        ),
        x_measurement_binding,
        SettingToKeywordBinding(
            y_object_setting,
            "y_object_name",
            normalized_symbol_name,
        ),
        y_measurement_binding,
        SettingToKeywordBinding(
            "Select the grid size", "gridsize", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            "How should the X-axis be scaled?",
            "x_scale",
            cellprofiler_enum_setting_parser(DensityPlotScaleType),
        ),
        SettingToKeywordBinding(
            "How should the Y-axis be scaled?",
            "y_scale",
            cellprofiler_enum_setting_parser(DensityPlotScaleType),
        ),
        SettingToKeywordBinding(
            "How should the colorbar be scaled?",
            "colorbar_scale",
            cellprofiler_enum_setting_parser(DensityPlotScaleType),
        ),
        SettingToKeywordBinding("Select the color map", "colormap"),
        SettingToKeywordBinding(
            "Enter a title for the plot, if desired",
            "title",
        ),
    )


class DisplayHistogramModule(
    DisplayMeasurementInputPolicy,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    DisplayModule,
):
    module_name = "DisplayHistogram"
    function_name = "display_histogram"
    validated = True
    confidence = 1.0
    object_setting = SettingNameFamily(
        "Select the object whose measurements will be displayed"
    )
    measurement_setting = SettingNameFamily("Select the object measurement to plot")
    measurement_binding = MeasurementFeatureSettingBinding(
        measurement_setting,
        "measurement_feature",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            object_setting,
            "object_name",
            normalized_symbol_name,
        ),
        measurement_binding,
        SettingToKeywordBinding("Number of bins", "num_bins", parse_cellprofiler_int),
        SettingToKeywordBinding(
            "How should the X-axis be scaled?",
            "x_scale",
            cellprofiler_enum_setting_parser(AxisScale),
        ),
        SettingToKeywordBinding(
            "How should the Y-axis be scaled?",
            "y_scale",
            cellprofiler_enum_setting_parser(AxisScale),
        ),
        SettingToKeywordBinding(
            "Enter a title for the plot, if desired",
            "title",
        ),
        SettingToKeywordBinding(
            "Specify min/max bounds for the X-axis?",
            "use_x_bounds",
            parse_cellprofiler_bool,
        ),
        CellProfilerFloatRangeSettingBinding(
            "Minimum/maximum values for the X-axis",
            "x_bounds",
            _parse_float_range,
        ),
    )


class DisplayPlatemapModule(
    DisplayMeasurementInputPolicy,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    DisplayModule,
):
    module_name = "DisplayPlatemap"
    function_name = "display_platemap"
    validated = True
    confidence = 1.0
    measurement_setting = SettingNameFamily("Select the measurement to plot")
    plate_metadata_setting = SettingNameFamily("Select your plate metadata")
    well_metadata_setting = SettingNameFamily("Select your well metadata")
    well_row_metadata_setting = SettingNameFamily("Select your well row metadata")
    well_column_metadata_setting = SettingNameFamily("Select your well column metadata")
    measurement_binding = MeasurementFeatureSettingBinding(
        measurement_setting,
        "measurement_feature",
    )
    plate_metadata_binding = MeasurementFeatureSettingBinding(
        plate_metadata_setting,
        "plate_metadata_feature",
    )
    well_metadata_binding = MeasurementFeatureSettingBinding(
        well_metadata_setting,
        "well_metadata_feature",
    )
    well_row_metadata_binding = MeasurementFeatureSettingBinding(
        well_row_metadata_setting,
        "well_row_metadata_feature",
    )
    well_column_metadata_binding = MeasurementFeatureSettingBinding(
        well_column_metadata_setting,
        "well_column_metadata_feature",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            "Display object or image measurements?",
            "objects_or_image",
            cellprofiler_enum_setting_parser(ObjectOrImage),
        ),
        SettingToKeywordBinding(
            "Select the object whose measurements will be displayed",
            "object_name",
            normalized_symbol_name,
        ),
        measurement_binding,
        plate_metadata_binding,
        SettingToKeywordBinding(
            "Multiwell plate format",
            "plate_type",
            cellprofiler_enum_setting_parser(PlateType),
        ),
        well_metadata_binding,
        well_row_metadata_binding,
        well_column_metadata_binding,
        SettingToKeywordBinding(
            "How should the values be aggregated?",
            "agg_method",
            cellprofiler_enum_setting_parser(AggregationMethod),
        ),
        SettingToKeywordBinding(
            "Enter a title for the plot, if desired",
            "title",
        ),
        SettingToKeywordBinding(
            "Well metadata format",
            "well_format",
            cellprofiler_enum_setting_parser(WellFormat),
        ),
    )


class DisplayScatterPlotModule(
    DisplayMeasurementInputPolicy,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    DisplayModule,
):
    module_name = "DisplayScatterPlot"
    function_name = "display_scatter_plot"
    validated = True
    confidence = 1.0
    x_measurement_setting = SettingNameFamily(
        "Select the measurement to plot on the X-axis"
    )
    y_measurement_setting = SettingNameFamily(
        "Select the measurement to plot on the Y-axis"
    )
    x_measurement_binding = MeasurementFeatureSettingBinding(
        x_measurement_setting,
        "x_measurement_feature",
    )
    y_measurement_binding = MeasurementFeatureSettingBinding(
        y_measurement_setting,
        "y_measurement_feature",
    )
    setting_bindings = (
        SettingToKeywordBinding(
            "Type of measurement to plot on X-axis",
            "x_source",
            cellprofiler_enum_setting_parser(MeasurementSource),
        ),
        SettingToKeywordBinding(
            "Select the object to plot on the X-axis",
            "x_object_name",
            normalized_symbol_name,
        ),
        x_measurement_binding,
        SettingToKeywordBinding(
            "Type of measurement to plot on Y-axis",
            "y_source",
            cellprofiler_enum_setting_parser(MeasurementSource),
        ),
        SettingToKeywordBinding(
            "Select the object to plot on the Y-axis",
            "y_object_name",
            normalized_symbol_name,
        ),
        y_measurement_binding,
        SettingToKeywordBinding(
            "How should the X-axis be scaled?",
            "x_scale",
            cellprofiler_enum_setting_parser(ScatterPlotScaleType),
        ),
        SettingToKeywordBinding(
            "How should the Y-axis be scaled?",
            "y_scale",
            cellprofiler_enum_setting_parser(ScatterPlotScaleType),
        ),
        SettingToKeywordBinding(
            "Enter a title for the plot, if desired",
            "title",
        ),
    )
