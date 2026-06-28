"""IdentifyPrimaryObjects execution policy for CellProfiler-compatible backends."""

import logging
import numpy as np
import time
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory import numpy
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_values import (
    ImagePayloadMetadata,
    SourceImageObjectLabelBuildRequest,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)

from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    required_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BoundModuleSettings,
    IdentifyPrimaryObjectsDebugViewModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    CurrentPayloadMeasurementRecordMixin,
    NoFieldsMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
    OutputObjectThresholdMeasurementRecordRowsMixin,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdMethod,
    CellProfilerThresholdRequest,
    CellProfilerThresholdScope,
    CellProfilerThresholdSettings,
    CellProfilerVarianceMethod,
    ThresholdSettingsModule,
    cellprofiler_threshold_diagnostics,
    normalize_cellprofiler_image,
    threshold_profile_sink,
    unit_interval_scale_for_threshold_diagnostics,
)
from openhcs.processing.backends.cellprofiler.perf_fixtures import capture_array_fixture
from openhcs.processing.backends.cellprofiler.watershed import (
    cellprofiler_legacy_watershed,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE,
    CellProfilerDeclumpMethod,
    DeclumpingMaximaGeometry,
    FillHolesOption,
    dense_label_area_statistics,
    filter_border_objects,
    filter_labels_by_area_numba,
    filter_labels_by_diameter_range,
    filter_physical_border_objects_numba,
    manual_declumping_size,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.shape import shape_measurement_backend
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
)
from openhcs.processing.backends.cellprofiler.enum_attributes import (
    CellProfilerEnumAttributeMixin,
)

logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)


class UnclumpMethod(Enum):
    INTENSITY = "intensity"
    SHAPE = "shape"
    NONE = "none"


class WatershedMethod(Enum):
    INTENSITY = "intensity"
    SHAPE = "shape"
    PROPAGATE = "propagate"
    NONE = "none"


class ExcessObjectHandling(CellProfilerEnumAttributeMixin, Enum):
    __cellprofiler_attribute_names__ = ("erase_excess",)

    CONTINUE = ("Continue", False)
    ERASE = ("Erase", True)


@dataclass
class PrimaryObjectStats:
    slice_index: int
    object_count: int
    mean_area: float
    median_area: float
    total_area: float
    threshold_used: float
    original_threshold: float = 0.0
    weighted_variance: float = 0.0
    sum_of_entropies: float = 0.0


@numpy
def identify_primary_objects(
    image: np.ndarray,
    min_diameter: int = 10,
    max_diameter: int = 40,
    exclude_size: bool = True,
    exclude_border_objects: bool = True,
    unclump_method: UnclumpMethod = UnclumpMethod.INTENSITY,
    watershed_method: WatershedMethod = WatershedMethod.INTENSITY,
    automatic_smoothing: bool = True,
    smoothing_filter_size: int = 10,
    automatic_suppression: bool = True,
    maxima_suppression_size: float = 7.0,
    low_res_maxima: bool = True,
    fill_holes: FillHolesOption = FillHolesOption.AFTER_BOTH,
    threshold_correction_factor: float = 1.0,
    threshold_min: float = 0.0,
    threshold_max: float = 1.0,
    use_advanced_settings: bool = True,
    threshold_scope: CellProfilerThresholdScope = CellProfilerThresholdScope.GLOBAL,
    threshold_method: CellProfilerThresholdMethod = CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
    threshold_smoothing_scale: float = 1.3488,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    assign_middle_to_foreground: CellProfilerThresholdAssignment = CellProfilerThresholdAssignment.FOREGROUND,
    log_transform: bool = False,
    adaptive_window_size: int = 10,
    lower_outlier_fraction: float = 0.05,
    upper_outlier_fraction: float = 0.05,
    averaging_method: CellProfilerAveragingMethod = CellProfilerAveragingMethod.MEAN,
    variance_method: CellProfilerVarianceMethod = CellProfilerVarianceMethod.STANDARD_DEVIATION,
    number_of_deviations: float = 2.0,
    manual_threshold: float = 0.0,
    maximum_object_count: int = 500,
    limit_erase: ExcessObjectHandling = ExcessObjectHandling.CONTINUE,
    morphology_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    watershed_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    respect_source_border_metadata: bool = False,
) -> Tuple[np.ndarray, PrimaryObjectStats, np.ndarray]:
    """
    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select the input image' -> (pipeline-handled)
        'Name the primary objects to be identified' -> (pipeline-handled)
        'Typical diameter of objects, in pixel units (Min,Max)' -> [min_diameter, max_diameter]
        'Discard objects outside the diameter range?' -> exclude_size
        'Discard objects touching the border of the image?' -> exclude_border_objects
        'Method to distinguish clumped objects' -> unclump_method
        'Method to draw dividing lines between clumped objects' -> watershed_method
        'Size of smoothing filter' -> smoothing_filter_size
        'Suppress local maxima that are closer than this minimum allowed distance' -> maxima_suppression_size
        'Speed up by using lower-resolution image to find local maxima?' -> low_res_maxima
        'Fill holes in identified objects?' -> fill_holes
        'Automatically calculate size of smoothing filter for declumping?' -> automatic_smoothing
        'Automatically calculate minimum allowed distance between local maxima?' -> automatic_suppression
        'Handling of objects if excessive number of objects identified' -> limit_erase
        'Maximum number of objects' -> maximum_object_count
        'Threshold correction factor' -> threshold_correction_factor
        'Lower and upper bounds on threshold' -> [threshold_min, threshold_max]
        'Use advanced settings?' -> use_advanced_settings
        'Threshold strategy' -> threshold_scope
        'Thresholding method' -> threshold_method
        'Threshold smoothing scale' -> threshold_smoothing_scale
        'Two-class or three-class thresholding?' -> otsu_class_count
        'Assign pixels in the middle intensity class to the foreground or the background?' -> assign_middle_to_foreground
        'Log transform before thresholding?' -> log_transform
        'Size of adaptive window' -> adaptive_window_size
        'Lower outlier fraction' -> lower_outlier_fraction
        'Upper outlier fraction' -> upper_outlier_fraction
        'Averaging method' -> averaging_method
        'Variance method' -> variance_method
        '# of deviations' -> number_of_deviations
        'Manual threshold' -> manual_threshold

    Identify primary objects in a grayscale image.
    
    Args:
        image: Input grayscale image (H, W)
        min_diameter: Minimum object diameter in pixels
        max_diameter: Maximum object diameter in pixels
        exclude_size: Discard objects outside diameter range
        exclude_border_objects: Discard objects touching image border
        unclump_method: Method to distinguish clumped objects
        watershed_method: Method to draw dividing lines between clumped objects
        automatic_smoothing: Auto-calculate smoothing filter size
        smoothing_filter_size: Size of smoothing filter for declumping
        automatic_suppression: Auto-calculate maxima suppression distance
        maxima_suppression_size: Minimum distance between local maxima
        low_res_maxima: Use lower resolution for finding maxima (faster)
        fill_holes: When to fill holes in objects
        threshold_correction_factor: Multiply threshold by this factor
        threshold_min: Minimum threshold value
        threshold_max: Maximum threshold value
        maximum_object_count: Max objects before erasing (if limit_erase=True)
        limit_erase: Erase all objects if count exceeds maximum
        respect_source_border_metadata: Treat only source-image edges as
            physical borders. The CellProfiler-compatible default filters the
            current image plane border.
    
    Returns:
        Tuple of (original image, object statistics, labeled image)
    """
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    profile_total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    morphology = MorphologyBackendStrategy.for_callable(
        identify_primary_objects,
        backend_provider=morphology_backend_provider,
    )
    unclump_method = coerce_cellprofiler_enum(UnclumpMethod, unclump_method)
    watershed_method = coerce_cellprofiler_enum(WatershedMethod, watershed_method)
    fill_holes = coerce_cellprofiler_enum(FillHolesOption, fill_holes)
    limit_erase = coerce_cellprofiler_enum(ExcessObjectHandling, limit_erase)
    threshold_method = coerce_cellprofiler_enum(
        CellProfilerThresholdMethod,
        threshold_method,
    )
    runtime_profiler.log(
        "ipo_prepare_inputs",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    phase_started_at = time.perf_counter()
    input_mask_payload = image_payload_mask(image)
    input_mask = (
        None
        if input_mask_payload is None
        else np.asarray(input_mask_payload, dtype=bool)
    )
    input_metadata = (
        image_payload_metadata(image)
        if respect_source_border_metadata
        else ImagePayloadMetadata()
    )
    raw_image_data = np.asarray(image_payload_data(image))
    proven_unit_interval_scale = unit_interval_scale_for_threshold_diagnostics(
        raw_image_data,
        image_payload_metadata(image),
    )
    img = normalize_cellprofiler_image(image)
    effective_threshold_smoothing = threshold_smoothing_scale
    runtime_profiler.log(
        "ipo_normalize_image",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    
    phase_started_at = time.perf_counter()
    threshold = CellProfilerThresholdRequest(
        image=img,
        image_mask=input_mask,
        settings=CellProfilerThresholdSettings(
            use_advanced_settings=use_advanced_settings,
            threshold_scope=threshold_scope,
            threshold_method=threshold_method,
            threshold_smoothing_scale=effective_threshold_smoothing,
            otsu_class_count=otsu_class_count,
            assign_middle_to_foreground=assign_middle_to_foreground,
            log_transform=log_transform,
            threshold_correction_factor=threshold_correction_factor,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            adaptive_window_size=adaptive_window_size,
            lower_outlier_fraction=lower_outlier_fraction,
            upper_outlier_fraction=upper_outlier_fraction,
            averaging_method=averaging_method,
            variance_method=variance_method,
            number_of_deviations=number_of_deviations,
            manual_threshold=manual_threshold,
            smooth_threshold_application=True,
        ),
        proven_unit_interval_scale=proven_unit_interval_scale,
        log_profile_function=threshold_profile_sink(),
    ).calculate()
    binary = threshold.mask
    runtime_profiler.log(
        "ipo_threshold",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
        threshold=float(threshold.final_threshold),
        original_threshold=float(threshold.original_threshold),
        mask_pixels=(
            int(np.asarray(input_mask, dtype=bool).sum())
            if input_mask is not None
            else None
        ),
        image_sources=image_payload_metadata(image).source_image_names,
        image_source_path=image_payload_metadata(image).source_path,
        image_source_components=image_payload_metadata(image).source_component_metadata,
    )
    # Fill holes if requested (before declumping)
    if fill_holes.before_declump_requested(use_advanced_settings=use_advanced_settings):
        phase_started_at = time.perf_counter()
        max_hole_size = max_diameter * max_diameter
        binary = morphology.fill_labeled_holes_below_size(
            binary,
            max_hole_size,
        )
        runtime_profiler.log(
            "ipo_fill_before_declump",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
    
    # Initial labeling
    phase_started_at = time.perf_counter()
    labeled_image, object_count = morphology.connected_components(
        binary,
        connectivity=2,
    )
    declump_backend_method = CellProfilerDeclumpMethod.SHAPE
    runtime_profiler.log(
        "ipo_initial_label",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
        object_count=object_count,
    )
    
    # Declumping and watershed
    if unclump_method != UnclumpMethod.NONE and watershed_method != WatershedMethod.NONE and object_count > 0:
        declump_started_at = time.perf_counter()
        if automatic_smoothing:
            smooth_size = 2.35 * min_diameter / 3.5
        else:
            smooth_size = manual_declumping_size(smoothing_filter_size)

        declump_backend_method = (
            CellProfilerDeclumpMethod.INTENSITY
            if unclump_method is UnclumpMethod.INTENSITY
            else CellProfilerDeclumpMethod.SHAPE
        )
        maxima_geometry = DeclumpingMaximaGeometry.from_cellprofiler_settings(
            min_diameter=min_diameter,
            low_res_maxima=low_res_maxima,
            automatic_suppression=automatic_suppression,
            maxima_suppression_size=maxima_suppression_size,
        )
        image_resize_factor = maxima_geometry.image_resize_factor
        suppress_size = maxima_geometry.suppress_size

        maxima_mask = np.asarray(
            morphology.declumping_suppression_footprint(
                suppress_size,
                min_diameter=min_diameter,
                declump_method=declump_backend_method,
            ),
            dtype=bool,
        )
        image_mask = (
            np.ones(img.shape, dtype=bool)
            if input_mask is None
            else np.asarray(input_mask, dtype=bool)
        )
        distance = None
        if unclump_method == UnclumpMethod.INTENSITY:
            phase_started_at = time.perf_counter()
            smoothed = morphology.smooth_image_for_declumping(
                img,
                image_mask,
                smooth_size,
                declump_method=declump_backend_method,
                suppress_size=suppress_size,
                min_diameter=min_diameter,
            )
            maxima_image = smoothed
            runtime_profiler.log(
                "ipo_declump_smooth",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
            )
        else:
            phase_started_at = time.perf_counter()
            distance = shape_measurement_backend().distance_to_edge(
                np.asarray(labeled_image, dtype=np.int32)
            )
            distance = distance + np.random.RandomState(0).uniform(
                0,
                0.001,
                distance.shape,
            )
            maxima_image = distance
            runtime_profiler.log(
                "ipo_declump_distance",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
            )

        phase_started_at = time.perf_counter()
        maxima = morphology.declumping_seed_points(
            maxima_image,
            labeled_image,
            maxima_mask,
            image_resize_factor,
        )
        runtime_profiler.log(
            "ipo_declump_seed_points",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
        phase_started_at = time.perf_counter()
        markers, object_count = morphology.connected_components(
            maxima,
            connectivity=2,
        )
        runtime_profiler.log(
            "ipo_declump_marker_label",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
            object_count=object_count,
        )
        if object_count > 0:
            phase_started_at = time.perf_counter()
            if watershed_method == WatershedMethod.SHAPE:
                if distance is None:
                    distance = shape_measurement_backend().distance_to_edge(
                        np.asarray(labeled_image, dtype=np.int32)
                    )
                watershed_image = -distance
                watershed_image = watershed_image - np.min(watershed_image)
            else:
                watershed_image = 1 - img
            watershed_markers = np.zeros(watershed_image.shape, np.int32)
            watershed_markers[markers > 0] = -markers[markers > 0]
            runtime_profiler.log(
                "ipo_watershed_prepare",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
            )
            phase_started_at = time.perf_counter()
            labeled_image = -cellprofiler_legacy_watershed(
                watershed_image,
                markers=watershed_markers,
                mask=labeled_image != 0,
                connectivity=np.ones((3, 3), bool),
                backend_provider=watershed_backend_provider,
            )
            object_count = int(labeled_image.max())
            runtime_profiler.log(
                "ipo_watershed_execute",
                time.perf_counter() - phase_started_at,
                function="identify_primary_objects",
                object_count=object_count,
            )
        runtime_profiler.log(
            "ipo_declump_total",
            time.perf_counter() - declump_started_at,
            function="identify_primary_objects",
        )
    
    phase_started_at = time.perf_counter()
    unedited_labels = labeled_image.copy()
    small_removed_labels = labeled_image.copy()
    runtime_profiler.log(
        "ipo_copy_label_variants",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )

    # Filter objects touching the image border, or the mask border when the
    # image has a crop/mask domain. This mirrors CellProfiler's legacy rule:
    # mask-border filtering is applied only when no labels touch the physical
    # image border.
    if exclude_border_objects and object_count > 0:
        phase_started_at = time.perf_counter()
        labeled_image = filter_border_objects(
            labeled_image,
            image_mask=input_mask,
            image_metadata=input_metadata,
        )
        runtime_profiler.log(
            "ipo_filter_border",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
    
    # Filter objects by size. Keep CellProfiler's small-removed variant: small
    # objects are removed, large objects are still present.
    if exclude_size and object_count > 0:
        phase_started_at = time.perf_counter()
        small_removed_labels, labeled_image = filter_labels_by_diameter_range(
            labeled_image,
            min_diameter,
            max_diameter,
        )
        runtime_profiler.log(
            "ipo_filter_size",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )

    # CellProfiler fills segmented-object holes after border and size filtering.
    if fill_holes.after_declump_requested(use_advanced_settings=use_advanced_settings):
        phase_started_at = time.perf_counter()
        capture_array_fixture("ipo_fill_after", labels=labeled_image)
        labeled_image = morphology.fill_labeled_holes(labeled_image)
        runtime_profiler.log(
            "ipo_fill_after_declump",
            time.perf_counter() - phase_started_at,
            function="identify_primary_objects",
        )
    
    # Relabel while preserving watershed boundaries between touching objects.
    phase_started_at = time.perf_counter()
    labeled_image, object_count = morphology.relabel_sequential(labeled_image)
    runtime_profiler.log(
        "ipo_relabel",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
        object_count=object_count,
    )
    
    # Check object count limit
    if limit_erase.erase_excess and object_count > maximum_object_count:
        labeled_image = np.zeros_like(labeled_image)
        object_count = 0
    
    # Calculate statistics
    phase_started_at = time.perf_counter()
    mean_area, median_area, total_area = dense_label_area_statistics(labeled_image)
    runtime_profiler.log(
        "ipo_statistics_diagnostics",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    
    stats = PrimaryObjectStats(
        slice_index=0,
        object_count=object_count,
        mean_area=mean_area,
        median_area=median_area,
        total_area=total_area,
        threshold_used=float(threshold.final_threshold),
        original_threshold=threshold.original_threshold,
        weighted_variance=threshold.weighted_variance,
        sum_of_entropies=threshold.sum_of_entropies,
    )
    phase_started_at = time.perf_counter()
    label_payload = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=labeled_image.astype(np.int32, copy=False),
        unedited_labels=unedited_labels.astype(np.int32, copy=False),
        small_removed_labels=small_removed_labels.astype(np.int32, copy=False),
        declared_object_count=object_count,
    ).payload()
    runtime_profiler.log(
        "ipo_build_label_payload",
        time.perf_counter() - phase_started_at,
        function="identify_primary_objects",
    )
    runtime_profiler.log(
        "ipo_total",
        time.perf_counter() - profile_total_started_at,
        function="identify_primary_objects",
    )

    return (
        image,
        stats,
        label_payload,
    )


class IdentifyPrimaryObjectsModule(
    OutputObjectThresholdMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    CurrentPayloadMeasurementRecordMixin,
    NoFieldsMeasurementRecordMixin,
    ThresholdSettingsModule,
    IdentifyPrimaryObjectsDebugViewModule,
):
    module_name = "IdentifyPrimaryObjects"
    function_name = "identify_primary_objects"
    validated = True
    confidence = 1.0
    include_threshold_advanced_setting = True
    input_image_setting = SettingNameFamily(
        "Select the input image",
        aliases=(
            "Select an input image",
            "Select the input binary image",
            "Input",
        ),
    )
    output_objects_setting = SettingNameFamily(
        "Name the primary objects to be identified",
        aliases=("Object",),
    )
    ignored_settings = (
        input_image_setting,
        output_objects_setting,
        "Display accepted local maxima?",
        "Select maxima color",
    )

    setting_bindings = (
        SettingToKeywordBinding(
            "Typical diameter of objects, in pixel units (Min,Max)",
            "diameter_range",
        ),
        SettingToKeywordBinding(
            "Discard objects outside the diameter range?",
            "exclude_size",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Discard objects touching the border of the image?",
            "exclude_border_objects",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Method to distinguish clumped objects",
            "unclump_method",
            cellprofiler_enum_value_setting_parser(UnclumpMethod),
        ),
        SettingToKeywordBinding(
            "Method to draw dividing lines between clumped objects",
            "watershed_method",
            cellprofiler_enum_value_setting_parser(WatershedMethod),
        ),
        SettingToKeywordBinding(
            "Size of smoothing filter",
            "smoothing_filter_size",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Suppress local maxima that are closer than this minimum allowed distance",
            "maxima_suppression_size",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Speed up by using lower-resolution image to find local maxima?",
            "low_res_maxima",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Fill holes in identified objects?",
            "fill_holes",
            cellprofiler_enum_value_setting_parser(FillHolesOption),
        ),
        SettingToKeywordBinding(
            "Automatically calculate size of smoothing filter for declumping?",
            "automatic_smoothing",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Automatically calculate minimum allowed distance between local maxima?",
            "automatic_suppression",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Handling of objects if excessive number of objects identified",
            "limit_erase",
            cellprofiler_enum_value_setting_parser(ExcessObjectHandling),
        ),
        SettingToKeywordBinding(
            "Maximum number of objects",
            "maximum_object_count",
            parse_cellprofiler_int,
        ),
    )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        kwargs = dict(bound.kwargs)
        diameter_range = kwargs.pop("diameter_range", None)
        if diameter_range is not None:
            if not isinstance(diameter_range, tuple) or len(diameter_range) != 2:
                raise ValueError(
                    f"{module.name} diameter range must contain two values, "
                    f"got {diameter_range!r}."
                )
            kwargs["min_diameter"], kwargs["max_diameter"] = diameter_range
        return BoundModuleSettings(
            kwargs,
            bound.unmapped_kwargs,
            bound.invocation_options,
            bound.setting_coverage,
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

        inputs = [
            builder.require_artifact(ArtifactSpec(required_setting_value(module, cls.input_image_setting), ArtifactKind.IMAGE), module)
        ]
        outputs = [
            builder.declare_artifact(ArtifactSpec(cls.measurement_artifact_name(module), ArtifactKind.MEASUREMENTS), module),
            builder.declare_artifact(ArtifactSpec(required_setting_value(module, cls.output_objects_setting), ArtifactKind.OBJECT_LABELS), module),
        ]
        return assembler.assemble_contract(module, builder, inputs=inputs, outputs=outputs)


def _prepare_identify_primary_objects() -> None:
    """Compile Numba/backend kernels used by IdentifyPrimaryObjects."""
    labels = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 2],
            [3, 3, 2, 2],
            [0, 0, 2, 2],
        ],
        dtype=np.int32,
    )
    areas = np.bincount(labels.ravel())
    filter_labels_by_area_numba(
        np.ascontiguousarray(labels),
        np.ascontiguousarray(areas),
        2.0,
        4.0,
    )
    filter_labels_by_diameter_range(labels, 2.0, 4.0)
    filter_physical_border_objects_numba(labels, True, True, True, True)
    stacked_labels = np.stack((labels, labels), axis=0)
    stacked_areas = np.bincount(stacked_labels.ravel())
    filter_labels_by_area_numba(
        np.ascontiguousarray(stacked_labels),
        np.ascontiguousarray(stacked_areas),
        2.0,
        4.0,
    )
    filter_labels_by_diameter_range(stacked_labels, 2.0, 4.0)
    image = np.zeros((96, 96), dtype=np.float32)
    yy, xx = np.ogrid[:96, :96]
    image[((yy - 32) ** 2 + (xx - 32) ** 2) <= 12 * 12] = 0.8
    image[((yy - 56) ** 2 + (xx - 56) ** 2) <= 12 * 12] = 0.75
    binary = image > np.float32(0.5)
    cellprofiler_threshold_diagnostics(
        image,
        binary,
        final_threshold=0.5,
        original_threshold=0.5,
        proven_unit_interval_scale=65535,
    )
    rectangular_mask = np.zeros_like(binary, dtype=bool)
    rectangular_mask[16:80, 16:80] = True
    cellprofiler_threshold_diagnostics(
        image,
        binary,
        final_threshold=0.5,
        original_threshold=0.5,
        mask=rectangular_mask,
        proven_unit_interval_scale=65535,
    )
    identify_primary_objects.__wrapped__(
        image,
        min_diameter=10,
        max_diameter=45,
        unclump_method=UnclumpMethod.SHAPE,
        watershed_method=WatershedMethod.SHAPE,
        low_res_maxima=True,
        use_advanced_settings=True,
        threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
    )
    identify_primary_objects.__wrapped__(
        image,
        min_diameter=3,
        max_diameter=15,
        unclump_method=UnclumpMethod.INTENSITY,
        watershed_method=WatershedMethod.INTENSITY,
        low_res_maxima=True,
        use_advanced_settings=True,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.BACKGROUND,
        threshold_smoothing_scale=1.3488,
    )
    identify_primary_objects.__wrapped__(
        image,
        min_diameter=3,
        max_diameter=15,
        unclump_method=UnclumpMethod.INTENSITY,
        watershed_method=WatershedMethod.INTENSITY,
        low_res_maxima=True,
        use_advanced_settings=True,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.BACKGROUND,
        threshold_smoothing_scale=1.3488,
    )


identify_primary_objects.__openhcs_prepare__ = _prepare_identify_primary_objects


__all__ = public_names_from_objects(
    ExcessObjectHandling,
    FillHolesOption,
    IdentifyPrimaryObjectsModule,
    PrimaryObjectStats,
    UnclumpMethod,
    WatershedMethod,
    identify_primary_objects,
)
