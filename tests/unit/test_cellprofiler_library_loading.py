import importlib
import sys
import types
from typing import get_type_hints
import numpy as np
import pytest
import skimage.morphology
import skimage.segmentation
from python_introspect import declared_enum_type
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadBundleContext,
    ImagePayloadExecutionMode,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.processing.backends.cellprofiler.alignment import (
    AlignModule,
    AlignShiftMeasurement,
    align,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    closing,
    mask_objects,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    CalculationScope as IlluminationCalculationScope,
    FilterSizeMethod as IlluminationFilterSizeMethod,
    IlluminationCorrectionMethod,
    IntensityChoice as IlluminationIntensityChoice,
    RescaleOption as IlluminationRescaleOption,
    SmoothingMethod as IlluminationSmoothingMethod,
    correct_illumination_apply,
    correct_illumination_calculate,
)
from openhcs.processing.backends.cellprofiler.crop import crop
from openhcs.processing.backends.cellprofiler.area_occupied import (
    AreaOccupiedRow,
    OperandChoice,
    measure_image_area_occupied,
)
from openhcs.processing.backends.cellprofiler.texture import (
    ObjectTextureCropBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation,
    image_math,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    MaskSource,
    mask_image,
)
from openhcs.processing.backends.cellprofiler.primary_objects import (
    ExcessObjectHandling,
    FillHolesOption,
    UnclumpMethod,
    WatershedMethod,
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE,
    DeclumpingMaximaGeometry,
    dilate_objects,
    filter_border_objects,
    filter_labels_by_diameter_range,
    manual_declumping_size,
    resize_object_labels_nearest,
)
import openhcs.processing.backends.cellprofiler.thresholding as thresholding_backend
from openhcs.processing.backends.cellprofiler.colocalization import (
    measure_colocalization,
    measure_colocalization_objects,
    _divide_costes_measurements,
)
from openhcs.processing.backends.cellprofiler.colocalization import (
    ColocalizationCostesThresholdBatch,
    ColocalizationCostesThresholds,
    ColocalizationImagePairContext,
    ColocalizationObjectLabelContext,
    MeasureColocalizationModule,
    ObjectColocalizationMetricArrays,
    costes_backend,
    measure_colocalization_objects_batch,
    object_colocalization_threshold_reductions,
    thresholded_colocalization_metrics,
)
from openhcs.processing.backends.cellprofiler.morphology import opening
from openhcs.processing.backends.cellprofiler.outlines import (
    LineMode,
    OverlayObjectsModule,
    OutlineSourceKind,
    overlay_objects,
)
from openhcs.processing.backends.cellprofiler.outlines import overlay_outlines
from openhcs.processing.backends.cellprofiler.image_geometry import (
    align_label_plane_to_shape,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    DistanceMethod,
    relate_objects,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    resize,
    resize_volumetric,
    FlipMethod,
    flip_and_rotate,
    mask_image_with_binary,
)
from openhcs.processing.backends.cellprofiler.edge import (
    EdgeDirection,
    EdgeMethod,
    enhance_edges,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
)
from openhcs.processing.backends.cellprofiler.smoothing import SmoothingMethod, smooth
from openhcs.processing.backends.cellprofiler.thresholding import (
    CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE,
    CellProfilerAveragingMethod,
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    CellProfilerThresholdMethod,
    CellProfilerThresholdScope,
    CellProfilerVarianceMethod,
    ThresholdModule,
    CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET,
    CELLPROFILER_LOG_MULTI_OTSU_BINS,
    cellprofiler_get_global_threshold,
    cellprofiler_threshold_diagnostics,
    cellprofiler_threshold,
    threshold_histogram_bin_width,
    threshold_multiotsu,
)
from openhcs.processing.backends.cellprofiler.thresholding import threshold
from openhcs.processing.backends.cellprofiler.color import StainType, unmix_colors
from openhcs.processing.backends.cellprofiler.crop import CropModule
from openhcs.core.config import DtypeConfig
from openhcs.core.runtime_batch_contracts import RuntimeBatchInvocationRequest
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    ImageMetadataPayload,
    MaskedImagePayload,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.morphology import (
    CellProfilerDeclumpMethod,
)
from openhcs.constants.constants import VariableComponents
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.settings_binder import (
    CellProfilerEnumSettingParser,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    SourceQualifiedMeasurementFeatureModule,
    SourceQualifiedWideMeasurementRowsModule,
)


def test_module_registry_resolves_every_declared_function():
    for module_type in CellProfilerModule.__registry__.values():
        if not module_type.emits_function_step():
            continue
        function_names = module_type.declared_function_names()
        assert function_names
        assert all(
            callable(module_type.require_callable(name)) for name in function_names
        )


def test_enum_setting_parsers_share_callable_annotation_owner() -> None:
    OpenHCSRegistry().get_modules_to_scan()
    failures: list[str] = []
    parser_count = 0
    for module_type in set(CellProfilerModule.__registry__.values()):
        if not module_type.emits_function_step():
            continue
        annotation_maps = tuple(
            get_type_hints(
                module_type.require_callable(function_name),
                include_extras=True,
            )
            for function_name in module_type.declared_function_names()
        )
        for binding in module_type.declared_setting_bindings():
            parser = binding.parse
            if not isinstance(parser, CellProfilerEnumSettingParser):
                continue
            parser_count += 1
            parameter_name = binding.require_parameter_name()
            callable_enum_types = tuple(
                declared_enum_type(annotations[parameter_name])
                for annotations in annotation_maps
                if parameter_name in annotations
            )
            if callable_enum_types and all(
                enum_type is parser.enum_type for enum_type in callable_enum_types
            ):
                continue
            failures.append(
                f"{module_type.__name__}.{parameter_name}: parser="
                f"{parser.enum_type.__name__}, callables="
                f"{tuple(getattr(enum_type, '__name__', None) for enum_type in callable_enum_types)!r}"
            )

    assert parser_count
    assert not failures, "\n".join(failures)


def test_measure_colocalization_declares_composed_stack_execution() -> None:
    assert (
        CallableContract.from_callable(
            measure_colocalization
        ).runtime_image_execution_mode
        is ImagePayloadExecutionMode.FULL_STACK
    )


def test_track_objects_callable_requires_timepoint():
    track_objects = CellProfilerModule.__registry__["TrackObjects"].require_callable()
    assert CallableContract.from_callable(
        track_objects
    ).required_variable_components == (VariableComponents.TIMEPOINT,)


def test_correct_illumination_apply_inherits_resolved_stack_grouping() -> None:
    assert (
        CallableContract.from_callable(
            correct_illumination_apply
        ).required_variable_components
        == ()
    )
    assert (
        CellProfilerModule.require_module("CorrectIlluminationApply").group_by is None
    )


def test_mask_objects_has_no_unconditional_image_stack_axis() -> None:
    assert (
        CallableContract.from_callable(mask_objects).required_variable_components == ()
    )


def test_all_cellprofiler_module_aliases_canonicalize():
    for module_type in CellProfilerModule.__registry__.values():
        for alias in module_type.aliases:
            assert CellProfilerModule.require_module(alias) is module_type
            assert (
                CellProfilerModule.require_module(alias).require_callable()
                is module_type.require_callable()
            )


def test_cellprofiler_module_rejects_primary_function_variant():
    with pytest.raises(ValueError, match="primary function"):

        class InvalidPrimaryVariantModule(CellProfilerModule):
            module_name = "__InvalidPrimaryVariantModule__"
            function_name = "invalid_primary_variant"
            validated = True
            function_variants = ("invalid_primary_variant",)


def test_cellprofiler_module_rejects_duplicate_alias_owner():
    with pytest.raises(ValueError, match="duplicates CellProfiler module"):

        class InvalidDuplicateAliasModule(CellProfilerModule):
            module_name = "__InvalidDuplicateAliasModule__"
            function_name = "invalid_duplicate_alias"
            validated = True
            aliases = ("MeasureCorrelation",)


def test_cellprofiler_module_discovers_measurement_features_from_mro():
    from openhcs.processing.backends.cellprofiler.area_occupied import (
        MeasureImageAreaOccupiedBinaryModule,
    )
    from openhcs.processing.backends.cellprofiler.intensity import (
        MeasureObjectIntensityModule,
    )

    assert MeasureObjectIntensityModule.measurement_feature_types() == (
        MeasureObjectIntensityModule.MeasurementFeature,
    )
    assert (
        MeasureObjectIntensityModule.source_qualified_measurement_feature_types()
        == (MeasureObjectIntensityModule.MeasurementFeature,)
    )
    assert MeasureImageAreaOccupiedBinaryModule.measurement_feature_types() == (
        MeasureImageAreaOccupiedBinaryModule.MeasurementFeature,
    )
    assert MeasureImageAreaOccupiedBinaryModule.source_qualified_measurement_feature_types() == (
        MeasureImageAreaOccupiedBinaryModule.MeasurementFeature,
    )


def test_source_qualified_wide_projection_has_one_nominal_owner() -> None:
    source_qualified_modules = tuple(
        module_type
        for module_type in CellProfilerModule.__registry__.values()
        if issubclass(module_type, SourceQualifiedMeasurementFeatureModule)
    )

    assert source_qualified_modules
    assert all(
        module_type.declared_source_qualified_measurement_feature_family_parts()
        for module_type in source_qualified_modules
    )
    wide_projection_modules = tuple(
        module_type
        for module_type in source_qualified_modules
        if issubclass(module_type, SourceQualifiedWideMeasurementRowsModule)
    )
    typed_projection_modules = tuple(
        module_type
        for module_type in source_qualified_modules
        if not issubclass(module_type, SourceQualifiedWideMeasurementRowsModule)
    )

    assert wide_projection_modules
    assert typed_projection_modules
    assert all(
        not module_type.measurement_row_projection_types()
        for module_type in wide_projection_modules
    )
    assert all(
        next(
            owner
            for owner in module_type.__mro__
            if "project_measurement_record_rows" in owner.__dict__
        )
        is SourceQualifiedWideMeasurementRowsModule
        for module_type in wide_projection_modules
    )
    assert all(
        next(
            owner
            for owner in module_type.__mro__
            if "project_measurement_record_rows" in owner.__dict__
        )
        is not SourceQualifiedWideMeasurementRowsModule
        for module_type in typed_projection_modules
    )


def test_cellprofiler_module_rejects_manual_measurement_feature_type_tuple():
    with pytest.raises(TypeError, match="RuntimeMeasurementFeature enum classes"):

        class InvalidMeasurementFeatureTupleModule(CellProfilerModule):
            measurement_feature_enum_types = ()


def test_cellprofiler_module_rejects_manual_source_qualified_feature_type_tuple():
    with pytest.raises(TypeError, match="SourceQualifiedMeasurementFeatureModule"):

        class InvalidSourceQualifiedFeatureTupleModule(CellProfilerModule):
            source_qualified_measurement_feature_enum_types = ()


def test_absorbed_watershed_accepts_grayscale_volumes() -> None:
    from openhcs.processing.backends.cellprofiler.watershed import watershed_library

    image = np.zeros((5, 12, 12), dtype=np.float32)
    image[:, 2:5, 2:5] = 1.0
    image[:, 7:10, 7:10] = 1.0
    output, stats, labels = watershed_library(image, footprint=3)
    label_array = object_label_dense_array(labels)
    assert output.shape == image.shape
    assert label_array.shape == image.shape
    assert label_array.dtype == np.int32
    assert stats.columns["object_count"][0] >= 1


def test_watershed_marker_mode_preserves_marker_label_identity() -> None:
    from openhcs.processing.backends.cellprofiler.watershed import (
        WatershedDeclumpMethod,
        WatershedMethod,
        watershed_library,
    )

    image = np.zeros((12, 12), dtype=np.float32)
    image[1:5, 1:5] = 1.0
    image[7:11, 7:11] = 1.0
    markers = np.zeros_like(image, dtype=np.int32)
    markers[2, 2] = 3
    markers[8, 8] = 9
    _output, _stats, labels = watershed_library(
        image,
        topology_inputs=(markers,),
        watershed_method=WatershedMethod.MARKERS,
        declump_method=WatershedDeclumpMethod.SHAPE,
        use_advanced_settings=False,
    )
    assert set(np.unique(object_label_dense_array(labels))) == {0, 1, 2}


def test_resize_objects_preserves_leading_axes_for_volume_stacks() -> None:
    from openhcs.processing.backends.cellprofiler.morphology import resize_objects

    image = np.zeros((2, 3, 4, 5), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[:, :, 1:3, 1:3] = 1
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    raw_resize_objects = resize_objects
    while hasattr(raw_resize_objects, "__wrapped__"):
        raw_resize_objects = raw_resize_objects.__wrapped__
    _output, stats, resized, relationship = raw_resize_objects(
        image,
        label_payload,
        method="factor",
        factor_x=2.0,
        factor_y=2.0,
        factor_z=1.0,
    )
    assert object_label_dense_array(resized).shape == (2, 3, 8, 10)
    assert relationship.source_ids == (1,)
    assert relationship.target_ids == (1,)
    assert stats.columns["original_height"][0] == 4
    assert stats.columns["original_width"][0] == 5
    assert stats.columns["new_height"][0] == 8
    assert stats.columns["new_width"][0] == 10


def test_resize_object_labels_nearest_matches_scipy_order_zero_zoom() -> None:
    from scipy.ndimage import zoom

    labels = np.arange(2 * 3 * 4 * 5, dtype=np.int32).reshape((2, 3, 4, 5)) % 17
    for zoom_factors in (
        (1.0, 1.0, 2.0, 2.0),
        (1.0, 1.0, 0.5, 0.5),
        (1.0, 1.0, 1.5, 2.5),
        (1.0, 2.0, 1.0, 0.75),
    ):
        observed = resize_object_labels_nearest(labels, zoom_factors)
        expected = zoom(labels, zoom_factors, order=0, mode="nearest").astype(np.int32)

        np.testing.assert_array_equal(observed, expected)


def test_resize_preserves_resized_image_mask() -> None:
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask = np.array(
        [
            [True, True, False, False],
            [True, True, False, False],
            [False, False, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )
    raw_resize = resize
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__
    resized = raw_resize(
        MaskedImagePayload(data=image, mask=mask),
        resizing_factor_x=0.5,
        resizing_factor_y=0.5,
    )
    assert isinstance(resized, MaskedImagePayload)
    assert resized.data.shape == (2, 2)
    np.testing.assert_array_equal(
        resized.mask, np.array([[True, False], [False, True]])
    )


def test_resize_volumetric_preserves_resized_image_mask() -> None:
    image = np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4)
    mask = np.zeros_like(image, dtype=bool)
    mask[:, :2, :2] = True
    raw_resize = resize_volumetric
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__
    resized = raw_resize(
        MaskedImagePayload(data=image, mask=mask),
        resizing_factor_x=0.5,
        resizing_factor_y=0.5,
        resizing_factor_z=1.0,
    )
    assert isinstance(resized, MaskedImagePayload)
    assert resized.data.shape == (2, 2, 2)
    np.testing.assert_array_equal(resized.mask, mask[:, ::2, ::2])


def test_resize_volumetric_projects_declared_volume_factors_onto_2d_slice() -> None:
    image = np.arange(4 * 4, dtype=np.float32).reshape(4, 4)
    mask = np.zeros_like(image, dtype=bool)
    mask[:2, :2] = True
    raw_resize = resize_volumetric
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__
    resized = raw_resize(
        MaskedImagePayload(data=image, mask=mask),
        resizing_factor_x=0.5,
        resizing_factor_y=0.5,
        resizing_factor_z=1.0,
    )
    assert isinstance(resized, MaskedImagePayload)
    assert resized.data.shape == (2, 2)
    np.testing.assert_array_equal(resized.mask, mask[::2, ::2])


def test_resize_volumetric_preserves_leading_channel_axis() -> None:
    image = np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4)
    mask = np.ones_like(image, dtype=bool)
    mask[:, :, 0, :] = False
    raw_resize = resize_volumetric
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__
    resized = raw_resize(
        MaskedImagePayload(data=image, mask=mask),
        resizing_factor_x=0.5,
        resizing_factor_y=0.5,
        resizing_factor_z=1.0,
    )
    assert isinstance(resized, MaskedImagePayload)
    assert resized.data.shape == (2, 3, 2, 2)
    assert resized.mask.shape == resized.data.shape


def test_resize_volumetric_projects_default_cellprofiler_validity_mask() -> None:
    image = np.ones((2, 4, 4), dtype=np.float32)
    raw_resize = resize_volumetric
    while hasattr(raw_resize, "__wrapped__"):
        raw_resize = raw_resize.__wrapped__
    downsampled = raw_resize(
        image, resizing_factor_x=0.5, resizing_factor_y=0.5, resizing_factor_z=1.0
    )
    upsampled = raw_resize(
        downsampled, resizing_factor_x=2.0, resizing_factor_y=2.0, resizing_factor_z=1.0
    )
    assert isinstance(upsampled, MaskedImagePayload)
    assert upsampled.data.shape == image.shape
    assert upsampled.mask.shape == image.shape
    assert not upsampled.mask[:, 0, :].any()
    assert not upsampled.mask[:, :, 0].any()
    assert not upsampled.mask[:, -1, :].any()
    assert not upsampled.mask[:, :, -1].any()
    assert upsampled.mask[:, 1:-1, 1:-1].all()


def test_erode_objects_preserves_leading_axes_for_volume_stacks() -> None:
    from openhcs.processing.backends.cellprofiler.morphology import erode_objects

    image = np.zeros((2, 3, 7, 7), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[:, :, 2:5, 2:5] = 1
    raw_erode_objects = erode_objects
    while hasattr(raw_erode_objects, "__wrapped__"):
        raw_erode_objects = raw_erode_objects.__wrapped__
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(declared_object_ids=(1,)),
    )
    _output, stats, eroded, relationship = raw_erode_objects(
        image, label_payload, structuring_element=StructuringElement.BALL, size=1
    )
    assert object_label_dense_array(eroded).shape == labels.shape
    assert relationship.source_ids == (1,)
    assert relationship.target_ids == (1,)
    assert stats.columns["input_object_count"][0] == 1
    assert stats.columns["output_object_count"][0] == 1


def test_erode_image_preserves_leading_axes_for_volume_stacks() -> None:
    from openhcs.processing.backends.cellprofiler.morphology import erode_image

    image = np.zeros((2, 3, 7, 7), dtype=np.float32)
    image[:, :, 2:5, 2:5] = 1
    raw_erode_image = erode_image
    while hasattr(raw_erode_image, "__wrapped__"):
        raw_erode_image = raw_erode_image.__wrapped__
    eroded = raw_erode_image(
        image,
        structuring_element=StructuringElement.BALL,
        size=1,
    )
    assert eroded.shape == image.shape
    assert np.count_nonzero(eroded) < np.count_nonzero(image)


def test_convert_objects_to_image_accepts_volume_label_stacks() -> None:
    from openhcs.processing.backends.cellprofiler.object_images import (
        ImageMode,
        convert_objects_to_image,
    )

    labels = np.zeros((2, 3, 4, 5), dtype=np.int32)
    labels[:, :, 1:3, 1:3] = 1
    raw_convert_objects_to_image = convert_objects_to_image
    while hasattr(raw_convert_objects_to_image, "__wrapped__"):
        raw_convert_objects_to_image = raw_convert_objects_to_image.__wrapped__
    converted = raw_convert_objects_to_image(
        np.zeros_like(labels, dtype=np.float32),
        labels,
        image_mode=ImageMode.COLOR,
    )
    assert converted.shape == labels.shape
    assert converted.dtype == np.float32
    assert np.all(converted[labels == 0] == 0.0)
    assert np.all(converted[labels == 1] > 0.0)


def test_convert_objects_to_image_uint16_preserves_integer_object_ids() -> None:
    from openhcs.processing.backends.cellprofiler.object_images import (
        ImageMode,
        convert_objects_to_image,
    )

    labels = np.array([[0, 1, 3]], dtype=np.int32)
    raw_convert_objects_to_image = convert_objects_to_image
    while hasattr(raw_convert_objects_to_image, "__wrapped__"):
        raw_convert_objects_to_image = raw_convert_objects_to_image.__wrapped__
    converted = raw_convert_objects_to_image(
        np.zeros_like(labels, dtype=np.float32),
        labels,
        image_mode=ImageMode.UINT16,
    )
    assert converted.dtype == np.int32
    np.testing.assert_array_equal(converted, labels)


def test_overlay_objects_rejects_mismatched_label_geometry() -> None:
    image = np.zeros((8, 10), dtype=np.float32)
    labels = np.zeros((4, 5), dtype=np.int32)
    labels[1:3, 2:4] = 1
    raw_overlay_objects = overlay_objects
    while hasattr(raw_overlay_objects, "__wrapped__"):
        raw_overlay_objects = raw_overlay_objects.__wrapped__
    with pytest.raises(ValueError, match="must exactly match"):
        raw_overlay_objects(
            image,
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        )


def test_overlay_objects_preserves_payload_scoped_volume() -> None:
    image = np.stack(
        (
            np.full((4, 5), 0.25, dtype=np.float32),
            np.full((4, 5), 0.75, dtype=np.float32),
        )
    )
    label_data = np.zeros((2, 4, 5), dtype=np.int32)
    label_data[0, 1:3, 1:3] = 1
    label_data[1, 1:3, 2:4] = 1
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=label_data),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )
    raw_overlay_objects = overlay_objects
    while hasattr(raw_overlay_objects, "__wrapped__"):
        raw_overlay_objects = raw_overlay_objects.__wrapped__

    result = raw_overlay_objects(image, labels, opacity=0.2)

    assert result.shape == (2, 4, 5, 3)
    assert (
        OverlayObjectsModule.execution_mode(
            ImagePayloadExecutionMode.NATURAL,
            image=image,
            kwargs={"labels": labels},
            variable_components=(),
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )


def test_threshold_executes_site_stacks_as_independent_planes() -> None:
    assert (
        ThresholdModule.execution_mode(
            ImagePayloadExecutionMode.FULL_STACK,
            image=np.zeros((2, 4, 5), dtype=np.float32),
            kwargs={},
            variable_components=(VariableComponents.SITE,),
        )
        is ImagePayloadExecutionMode.NATURAL
    )


def test_threshold_preserves_declared_z_stack_execution() -> None:
    assert (
        ThresholdModule.execution_mode(
            ImagePayloadExecutionMode.FULL_STACK,
            image=np.zeros((2, 4, 5), dtype=np.float32),
            kwargs={},
            variable_components=(VariableComponents.Z_INDEX,),
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )


def test_empty_label_plane_rejects_mismatched_target_geometry() -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        align_label_plane_to_shape(np.zeros((0, 0), dtype=np.int32), (8, 10))


def test_opening_default_backend_matches_skimage_grayscale_opening() -> None:
    image = np.arange(15 * 17, dtype=np.float32).reshape(15, 17)
    image[3:8, 4:9] = 2.0
    footprint = skimage.morphology.disk(3)
    expected = skimage.morphology.opening(image, footprint)
    raw_opening = opening
    while hasattr(raw_opening, "__wrapped__"):
        raw_opening = raw_opening.__wrapped__
    observed = raw_opening(
        image,
        structuring_element=StructuringElement.DISK,
        size=3,
    )
    np.testing.assert_array_equal(observed, expected)


def test_closing_default_backend_matches_skimage_grayscale_closing() -> None:
    image = np.arange(15 * 17, dtype=np.float32).reshape(15, 17)
    image[3:8, 4:9] = 2.0
    footprint = skimage.morphology.disk(3)
    expected = skimage.morphology.closing(image, footprint)
    raw_closing = closing
    while hasattr(raw_closing, "__wrapped__"):
        raw_closing = raw_closing.__wrapped__
    observed = raw_closing(
        image,
        structuring_element=StructuringElement.DISK,
        size=3,
    )
    np.testing.assert_array_equal(observed, expected)


def test_threshold_unwraps_image_metadata_payload() -> None:
    payload = ImageMetadataPayload(
        data=np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float32),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )
    binary, measurements = threshold(
        payload, predefined_threshold=0.5, dtype_config=DtypeConfig()
    )
    np.testing.assert_array_equal(
        binary, np.array([[False, True], [False, True]], dtype=np.float32)
    )
    assert measurements.row_mappings()[0]["final_threshold"] == 0.5


def test_threshold_uses_and_preserves_input_image_mask() -> None:
    payload = MaskedImagePayload(
        data=np.array([[0.0, 1.0, 1.0], [0.25, 0.75, 1.0]], dtype=np.float32),
        mask=np.array([[True, True, False], [True, True, False]], dtype=bool),
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )
    binary, measurements = threshold(
        payload, predefined_threshold=0.5, dtype_config=DtypeConfig()
    )
    np.testing.assert_array_equal(
        image_payload_data(binary),
        np.array([[False, True, False], [False, True, False]], dtype=np.float32),
    )
    np.testing.assert_array_equal(image_payload_mask(binary), payload.mask)
    assert measurements.row_mappings()[0]["final_threshold"] == 0.5


def test_threshold_rejects_unprojected_explicit_mask() -> None:
    image = np.array([[0.0, 1.0, 1.0], [0.25, 0.75, 1.0]], dtype=np.float32)
    mask = np.ones((2, *image.shape), dtype=bool)
    mask[1, :, 2] = False
    with pytest.raises(ValueError, match="does not match declared image mask domain"):
        threshold(
            image,
            mask=mask,
            predefined_threshold=0.5,
            dtype_config=DtypeConfig(),
        )


def test_smooth_accepts_nominal_smoothing_method():
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    result = smooth(
        image,
        smoothing_method=SmoothingMethod.GAUSSIAN_FILTER,
        auto_object_size=False,
        object_size=3.0,
        dtype_config=DtypeConfig(),
    )
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.max(result) > 0


def test_smooth_matches_cellprofiler_masked_gaussian():
    from scipy.ndimage import gaussian_filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    image[1, 1] = 1.0
    mask = np.ones(image.shape, dtype=bool)
    mask[:3, :3] = False
    payload = ImagePayloadMetadata().payload_with(image, mask)
    object_size = 3.0
    result = smooth(
        payload,
        smoothing_method=SmoothingMethod.GAUSSIAN_FILTER,
        auto_object_size=False,
        object_size=object_size,
        dtype_config=DtypeConfig(),
    )
    sigma = object_size / 2.35
    masked_image = np.zeros(image.shape, dtype=image.dtype)
    masked_image[mask] = image[mask]
    weights = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0)
    expected = gaussian_filter(masked_image, sigma, mode="constant", cval=0) / (
        weights + np.finfo(float).eps
    )
    assert np.allclose(image_payload_data(result), expected.astype(np.float32))


def test_smooth_matches_cellprofiler_unmasked_gaussian_edge_normalization():
    from scipy.ndimage import gaussian_filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[0, 0] = 1.0
    object_size = 3.0
    result = smooth(
        image,
        smoothing_method=SmoothingMethod.GAUSSIAN_FILTER,
        auto_object_size=False,
        object_size=object_size,
        dtype_config=DtypeConfig(),
    )
    sigma = object_size / 2.35
    mask = np.ones(image.shape, dtype=bool)
    weights = gaussian_filter(mask.astype(float), sigma, mode="constant", cval=0)
    expected = gaussian_filter(image, sigma, mode="constant", cval=0) / (
        weights + np.finfo(float).eps
    )
    assert np.allclose(image_payload_data(result), expected.astype(np.float32))


def test_enhance_edges_accepts_nominal_method_and_direction():
    image = np.zeros((9, 9), dtype=np.float32)
    image[:, 5:] = 1.0
    result = enhance_edges(
        image,
        method=EdgeMethod.SOBEL,
        direction=EdgeDirection.ALL,
        dtype_config=DtypeConfig(),
    )
    assert result.shape == image.shape
    assert result.dtype == np.float32
    assert np.max(result) > 0


def test_enhance_edges_uses_and_preserves_runtime_mask():
    import centrosome.filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[:, 5:] = 1.0
    mask = np.ones(image.shape, dtype=bool)
    mask[:, :4] = False
    payload = ImagePayloadMetadata().payload_with(image, mask)
    result = enhance_edges(
        payload,
        method=EdgeMethod.SOBEL,
        direction=EdgeDirection.ALL,
        dtype_config=DtypeConfig(),
    )
    assert np.allclose(
        image_payload_data(result),
        centrosome.filter.sobel(image, mask).astype(np.float32),
    )
    assert np.array_equal(image_payload_mask(result), mask)


def test_closing_preserves_runtime_mask_context():
    from skimage.morphology import closing as skimage_closing
    from skimage.morphology import disk

    image = np.zeros((9, 9), dtype=np.float32)
    image[3:6, 3:6] = 1.0
    mask = np.ones(image.shape, dtype=bool)
    mask[:2, :] = False
    payload = ImagePayloadMetadata().payload_with(image, mask)
    result = closing(
        payload,
        structuring_element=StructuringElement.DISK,
        size=1,
        dtype_config=DtypeConfig(),
    )
    assert np.array_equal(image_payload_data(result), skimage_closing(image, disk(1)))
    assert np.array_equal(image_payload_mask(result), mask)


def test_cellprofiler_disk_structuring_element_uses_radius_setting():
    from openhcs.processing.backends.cellprofiler.structuring_elements import (
        StructuringElement,
        build_structuring_element,
    )
    from skimage.morphology import disk

    np.testing.assert_array_equal(
        build_structuring_element(StructuringElement.DISK, 5), disk(5)
    )


def test_cellprofiler_ball_structuring_element_uses_radius_setting():
    from openhcs.processing.backends.cellprofiler.structuring_elements import (
        StructuringElement,
        build_structuring_element,
    )
    from skimage.morphology import ball

    np.testing.assert_array_equal(
        build_structuring_element(StructuringElement.BALL, 2), ball(2)
    )


def test_cellprofiler_structuring_element_rejects_rank_erasure():
    from openhcs.processing.backends.cellprofiler.structuring_elements import (
        StructuringElement,
        adapt_structuring_element_rank,
        build_structuring_element,
    )

    footprint = build_structuring_element(StructuringElement.BALL, 2)
    with pytest.raises(ValueError, match="rank exceeds"):
        adapt_structuring_element_rank(footprint, 2)


def test_dilate_objects_rejects_volumetric_structuring_element_for_2d_labels():
    image = np.zeros((7, 7), dtype=np.float32)
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 3] = 1
    with pytest.raises(ValueError, match="rank exceeds"):
        dilate_objects.__wrapped__(
            image,
            labels=labels,
            structuring_element_shape=StructuringElement.BALL,
            structuring_element_size=1,
        )


def test_measure_colocalization_object_costes_preserves_undefined_ratios():
    ratios = _divide_costes_measurements([0.0, 2.0], [0.0, 4.0])
    metrics = ObjectColocalizationMetricArrays.empty(1)
    metrics.costes_m1[0], metrics.costes_m2[0] = ratios
    row = next(iter(metrics.rows_for(np.asarray((1,), dtype=np.int32))))
    assert np.isnan(row.costes_m1)
    assert row.costes_m2 == 0.5


def test_object_costes_threshold_boundary_uses_native_operators():
    thresholds = ColocalizationCostesThresholds.from_thresholds(2.0, 2.0)
    reductions = object_colocalization_threshold_reductions(
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
        np.array([1, 1, 1], dtype=np.int64),
        np.array([0.0], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
        thresholds.first,
        thresholds.second,
        1,
    )
    total_first_costes, total_second_costes, costes_sum1, costes_sum2 = reductions[-4:]
    assert total_first_costes[0] == 5.0
    assert total_second_costes[0] == 5.0
    assert costes_sum1[0] == 3.0
    assert costes_sum2[0] == 3.0


def test_object_costes_thresholds_are_compared_in_pixel_dtype():
    raw_threshold = 25.0 / 255.0
    pixel_threshold = np.float32(raw_threshold)
    above_threshold = np.nextafter(
        pixel_threshold,
        np.float32(np.inf),
        dtype=np.float32,
    )
    image = np.asarray(
        (
            ((pixel_threshold, above_threshold),),
            ((1.0, 1.0),),
        ),
        dtype=np.float32,
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.ones((1, 2), dtype=np.int32))
    )

    _output, rows = measure_colocalization_objects.__wrapped__(
        image,
        labels,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        costes_thresholds=ColocalizationCostesThresholds.from_thresholds(
            raw_threshold,
            0.0,
        ),
        scale_max=65535,
    )

    row = next(iter(rows))
    assert row.costes_m1 == pytest.approx(
        float(above_threshold) / float(pixel_threshold + above_threshold)
    )
    assert row.costes_m2 == pytest.approx(0.5)


def test_measure_colocalization_objects_accepts_unmasked_finite_images():
    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 1], [2, 2]], dtype=np.int32)
        )
    )
    output, rows = measure_colocalization_objects.__wrapped__(
        image, labels, do_costes=False, do_manders=False, do_rwc=False, do_overlap=False
    )
    assert np.array_equal(output, image[0:1])
    assert [row["object_label"] for row in rows.row_mappings()] == [1, 2]
    assert all(np.isfinite(row["correlation"]) for row in rows.row_mappings())


def test_measure_colocalization_objects_emits_and_splits_both_scopes():
    from openhcs.interop.cellprofiler.measurement_scope import (
        CellProfilerMeasurementTargetScope,
    )
    from openhcs.processing.backends.cellprofiler.colocalization import (
        MeasureColocalizationObjectMeasurementRowPolicy,
    )

    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        )
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array([[1, 1], [2, 2]], dtype=np.int32)
        )
    )

    _output, rows = measure_colocalization_objects.__wrapped__(
        image,
        labels,
        measurement_scope=CellProfilerMeasurementTargetScope.BOTH,
        do_costes=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
    )
    object_rows, image_rows = (
        MeasureColocalizationObjectMeasurementRowPolicy().split_scoped_rows(rows)
    )

    assert image_rows.row_count() == 1
    assert object_rows.row_count() == 2
    assert [row["object_label"] for row in object_rows.row_mappings()] == [1, 2]


def test_colocalization_object_label_context_rejects_mask_stack_broadcast():
    labels = np.array([[1, 0], [2, 2]], dtype=np.int32)
    pair_valid_mask = np.stack(
        (
            np.array([[True, False], [True, True]]),
            np.array([[True, False], [False, True]]),
        )
    )
    with pytest.raises(ValueError, match="must share a shape"):
        ColocalizationObjectLabelContext.from_dense_labels(
            labels, pair_valid_mask=pair_valid_mask
        )


def test_colocalization_object_label_context_rejects_unprojected_label_stack():
    labels = np.stack(
        (
            np.array([[1, 0], [0, 2]], dtype=np.int32),
            np.array([[1, 0], [0, 2]], dtype=np.int32),
        )
    )
    with pytest.raises(ValueError, match="projected to one 2-D plane"):
        ColocalizationObjectLabelContext.from_dense_labels(
            labels, pair_valid_mask=None, measurement_shape=(2, 2)
        )


def test_colocalization_image_pair_context_accepts_aligned_image_stack() -> None:
    aligned = AlignedImageStack(
        (
            np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32),
            np.array([[1.0, 0.5], [0.0, 0.75]], dtype=np.float32),
        )
    )
    context = ColocalizationImagePairContext.from_request(
        aligned, channel_1=0, channel_2=1
    )
    assert context.image_float.shape == (2, 2, 2)
    np.testing.assert_array_equal(context.first_image, aligned.slices[0])
    np.testing.assert_array_equal(context.second_image, aligned.slices[1])


def test_colocalization_threshold_batch_uses_one_aligned_image_pair_context() -> None:
    aligned = AlignedImageStack(
        (np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32))
    )
    request = RuntimeBatchInvocationRequest(
        source_image_name=None,
        func=lambda image, **kwargs: (image, kwargs),
        image=aligned,
        kwargs={
            "labels": ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.ones((2, 2), dtype=np.int32)
                )
            ),
            "channel_1": 0,
            "channel_2": 1,
        },
        batch_index=0,
        batch_count=1,
    )
    kwargs = ColocalizationCostesThresholdBatch().request_kwargs(request)
    assert isinstance(kwargs["image_pair_context"], ColocalizationImagePairContext)
    assert isinstance(kwargs["object_label_context"], ColocalizationObjectLabelContext)
    assert kwargs["object_label_context"].slice_index == 0
    rows = ObjectColocalizationMetricArrays.empty(2).rows_for(
        np.asarray((1, 2), dtype=np.int32)
    )
    assert rows.columns["slice_index"].tolist() == [0, 0]


def test_runtime_batch_projects_singleton_aligned_axis_before_colocalization() -> None:
    channel_bundle = ImagePayloadBundleContext.from_payloads(
        (
            np.ones((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
        )
    ).compose()
    request = RuntimeBatchInvocationRequest(
        source_image_name=None,
        func=measure_colocalization_objects,
        image=AlignedImageStack((channel_bundle,)),
        kwargs={
            "labels": ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.ones((2, 2), dtype=np.int32)
                )
            ),
            "channel_1": 0,
            "channel_2": 1,
            "do_costes": False,
        },
        execution_mode=ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=1,
        ),
        batch_index=0,
        batch_count=1,
    )

    batch_request = request.batch_executor_request()

    assert batch_request is not None
    assert batch_request.execution_mode is ImagePayloadExecutionMode.FULL_STACK
    assert batch_request.plane_projection is None
    assert image_payload_data(batch_request.image).shape == (2, 2, 2)
    context = ColocalizationCostesThresholdBatch().image_pair_context(batch_request)
    np.testing.assert_array_equal(context.first_image, np.ones((2, 2)))
    np.testing.assert_array_equal(context.second_image, np.zeros((2, 2)))


def test_colocalization_threshold_batch_caches_semantic_label_context() -> None:
    image = np.stack(
        (np.ones((2, 2), dtype=np.float32), np.full((2, 2), 0.5, dtype=np.float32)),
        axis=0,
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(((1, 1), (0, 2)), dtype=np.int64)
        ),
        domain=ObjectLabelDomain(declared_object_count=2),
    )
    request = RuntimeBatchInvocationRequest(
        source_image_name=None,
        func=lambda image, **kwargs: (image, kwargs),
        image=image,
        kwargs={"labels": labels, "channel_1": 0, "channel_2": 1},
        batch_index=0,
        batch_count=2,
    )
    batch = ColocalizationCostesThresholdBatch()
    image_pair_context = batch.image_pair_context(request)
    first = batch.object_label_context(request, image_pair_context)
    second = batch.object_label_context(request, image_pair_context)
    assert second is first


def test_measure_colocalization_objects_batch_uses_contract_execution() -> None:
    image = np.stack(
        (
            np.array(((0.1, 0.2), (0.3, 0.4)), dtype=np.float32),
            np.array(((0.4, 0.3), (0.2, 0.1)), dtype=np.float32),
        ),
        axis=0,
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.array(((1, 1), (0, 2)), dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_count=2),
    )
    request = RuntimeBatchInvocationRequest(
        source_image_name=None,
        func=measure_colocalization_objects,
        image=image,
        kwargs={"labels": labels, "channel_1": 0, "channel_2": 1, "do_costes": False},
        batch_index=0,
        batch_count=1,
    )
    captured_requests: list[RuntimeBatchInvocationRequest] = []

    def execute_request(
        _func: object, executed_request: RuntimeBatchInvocationRequest
    ) -> object:
        captured_requests.append(executed_request)
        return (executed_request.image, executed_request.kwargs)

    output, executed_kwargs = measure_colocalization_objects_batch(
        measure_colocalization_objects, (request,), execute_request
    )[0]
    assert captured_requests
    assert output is image
    assert isinstance(
        executed_kwargs["image_pair_context"], ColocalizationImagePairContext
    )
    assert isinstance(
        executed_kwargs["object_label_context"], ColocalizationObjectLabelContext
    )


def test_measure_colocalization_costes_thresholds_preserve_backend_values():
    first = 0.06666672229766846
    second = 0.08594463765621185
    thresholds = ColocalizationCostesThresholds.from_thresholds(first, second)
    assert thresholds == ColocalizationCostesThresholds(
        first=first,
        second=second,
    )


def test_costes_backend_uses_compiled_forward_slope(monkeypatch):
    from openhcs.processing.backends.cellprofiler import colocalization

    monkeypatch.setattr(
        colocalization,
        "_correlation_slopes_numba",
        lambda _first, _second: (0.5, 0.25, 0.125),
    )

    observed = costes_backend().correlation_slopes(
        np.asarray((1.0, 2.0)),
        np.asarray((3.0, 5.0)),
    )

    assert observed == (0.5, 0.25, 0.125)


@pytest.mark.parametrize(
    ("first_codes", "second_codes", "expected_thresholds"),
    (
        (
            [0, 0, 0, 5, 10, 20, 50],
            [2, 4, 5, 6, 10, 20, 40],
            (0.011764705882352941, 0.02214064961986053),
        ),
        (
            [
                31,
                95,
                154,
                33,
                44,
                75,
                145,
                169,
                19,
                159,
                98,
                49,
                11,
                42,
                17,
                133,
                69,
                69,
                22,
                231,
                119,
                156,
                255,
                236,
                6,
                79,
                105,
                70,
                229,
                94,
                135,
                29,
            ],
            [
                64,
                250,
                218,
                48,
                48,
                97,
                99,
                174,
                0,
                149,
                113,
                54,
                48,
                0,
                0,
                123,
                102,
                97,
                151,
                110,
                84,
                70,
                236,
                188,
                36,
                42,
                87,
                82,
                191,
                46,
                70,
                0,
            ],
            (0.1803921568627451, 0.18013729476096402),
        ),
        (
            [
                61,
                46,
                192,
                25,
                144,
                138,
                177,
                144,
                2,
                240,
                238,
                89,
                0,
                238,
                44,
                3,
                122,
                26,
                98,
                206,
                191,
                6,
                115,
                81,
                208,
                236,
                166,
                147,
                216,
                103,
                121,
                82,
            ],
            [
                3,
                28,
                233,
                131,
                174,
                158,
                158,
                84,
                212,
                113,
                249,
                98,
                31,
                175,
                81,
                0,
                151,
                110,
                28,
                98,
                169,
                0,
                43,
                87,
                192,
                255,
                171,
                175,
                194,
                110,
                40,
                187,
            ],
            (0.37254901960784315, 0.3840162241575861),
        ),
    ),
)
def test_measure_colocalization_faster_costes_matches_native_extracted_vectors(
    first_codes: list[int],
    second_codes: list[int],
    expected_thresholds: tuple[float, float],
):
    observed_thresholds = costes_backend().scaled_second_channel_costes(
        np.asarray(first_codes, dtype=np.float64) / 255.0,
        np.asarray(second_codes, dtype=np.float64) / 255.0,
        255,
    )
    assert observed_thresholds == expected_thresholds


def test_measure_colocalization_respects_masked_payload_pixels():
    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[4.0, 3.0], [2.0, 100.0]], dtype=np.float32),
        )
    )
    mask = np.array([[True, True], [True, False]])
    output, measurements = measure_colocalization.__wrapped__(
        MaskedImagePayload(
            data=image,
            mask=np.stack((mask, mask)),
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            ),
        ),
        do_costes=False,
    )
    assert measurements.columns["correlation"][0] == -1.0
    assert isinstance(output, MaskedImagePayload)
    assert np.array_equal(image_payload_data(output), image[0:1])
    assert np.array_equal(image_payload_mask(output), mask[np.newaxis, ...])


def test_measure_colocalization_records_cellprofiler_emitted_slope_only():
    image = np.stack(
        (
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.array([[1.0, 2.0], [4.0, 9.0]], dtype=np.float32),
        )
    )
    _, measurements = measure_colocalization.__wrapped__(
        image, do_costes=False, do_manders=False, do_rwc=False, do_overlap=False
    )
    first = image[0].ravel().astype(np.float64)
    second = image[1].ravel().astype(np.float64)
    centered_first = first - first.mean()
    centered_second = second - second.mean()
    expected_forward = np.dot(centered_first, centered_second) / np.dot(
        centered_first,
        centered_first,
    )
    assert measurements.columns["slope"][0] == pytest.approx(expected_forward)
    assert "slope_reverse" not in {
        feature.measurement_row_field_name
        for feature in MeasureColocalizationModule.MeasurementFeature
    }


def test_measure_colocalization_threshold_metrics_match_numpy_semantics():
    first = np.array([0.0, 0.2, 0.2, 0.7, 0.9, 1.0], dtype=np.float64)
    second = np.array([0.1, 0.1, 0.4, 0.6, 0.95, 0.2], dtype=np.float64)
    threshold_percent = 15.0
    thr_fi = threshold_percent * np.max(first) / 100
    thr_si = threshold_percent * np.max(second) / 100
    thr_fi_out = first > thr_fi
    thr_si_out = second > thr_si
    combined = thr_fi_out & thr_si_out
    first_thresholded = first[combined]
    second_thresholded = second[combined]
    total_first = first[thr_fi_out].sum()
    total_second = second[thr_si_out].sum()
    rank1 = np.lexsort([first])
    rank2 = np.lexsort([second])
    rank1_u = np.hstack([[False], first[rank1[:-1]] != first[rank1[1:]]])
    rank2_u = np.hstack([[False], second[rank2[:-1]] != second[rank2[1:]]])
    rank1_s = np.cumsum(rank1_u)
    rank2_s = np.cumsum(rank2_u)
    rank_im1 = np.zeros(first.shape, dtype=int)
    rank_im2 = np.zeros(second.shape, dtype=int)
    rank_im1[rank1] = rank1_s
    rank_im2[rank2] = rank2_s
    rank_count = max(rank_im1.max(), rank_im2.max()) + 1
    weight = (rank_count - np.abs(rank_im1 - rank_im2)) / rank_count
    product_sum = (first_thresholded * second_thresholded).sum()
    expected = (
        first_thresholded.sum() / total_first,
        second_thresholded.sum() / total_second,
        (first_thresholded * weight[combined]).sum() / total_first,
        (second_thresholded * weight[combined]).sum() / total_second,
        product_sum
        / np.sqrt((first_thresholded**2).sum() * (second_thresholded**2).sum()),
        product_sum / (first_thresholded**2).sum(),
        product_sum / (second_thresholded**2).sum(),
    )
    observed = thresholded_colocalization_metrics(
        first, second, threshold_percent, True, True, True
    )
    np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_cellprofiler_multiotsu_threshold_ignores_robust_background_settings():
    image = np.tile(np.array([0.05, 0.2, 0.75, 0.95], dtype=np.float32), (16, 16))
    binary, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.2,
        upper_outlier_fraction=0.2,
        averaging_method=CellProfilerAveragingMethod.MEDIAN,
        variance_method=CellProfilerVarianceMethod.MEDIAN_ABSOLUTE_DEVIATION,
        number_of_deviations=4,
        manual_threshold=0.5,
    )
    assert binary.dtype == np.bool_
    assert 0.0 < final_threshold < 1.0
    assert final_threshold == original_threshold


def test_minimum_cross_entropy_threshold_uses_default_numba_primitive():
    image = np.tile(
        np.array([0.0, 0.03, 0.08, 0.16, 0.4, 0.75], dtype=np.float32), (8, 8)
    )
    mask = np.ones(image.shape, dtype=bool)
    from openhcs.constants.constants import MemoryType
    from openhcs.processing.backends.cellprofiler.thresholding import (
        ThresholdPrimitiveBackendStrategy,
    )

    expected = ThresholdPrimitiveBackendStrategy.for_memory_type(
        MemoryType.NUMPY
    ).minimum_cross_entropy_threshold(image, mask=mask)
    observed = cellprofiler_get_global_threshold(
        image,
        mask=mask,
        threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
    )
    assert observed == expected


def test_cellprofiler_three_class_otsu_honors_log_transform():
    from openhcs.processing.backends.cellprofiler.thresholding import (
        threshold_primitives,
    )

    image = np.array([[0.01, 0.05, 0.06, 0.2, 0.25, 0.75, 0.9]] * 8, dtype=np.float32)
    primitives = threshold_primitives()
    log_image, conversion = primitives.log_transform(image)
    log_values = log_image.ravel()
    expected_log_threshold = (
        threshold_multiotsu(log_values, nbins=CELLPROFILER_LOG_MULTI_OTSU_BINS)[0]
        + threshold_histogram_bin_width(log_values, CELLPROFILER_LOG_MULTI_OTSU_BINS)
        * CELLPROFILER_LOG_MULTI_OTSU_BIN_CENTER_OFFSET
    )
    expected_threshold = primitives.inverse_log_transform(
        expected_log_threshold, conversion
    )
    binary, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=True,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.2,
        upper_outlier_fraction=0.2,
        averaging_method=CellProfilerAveragingMethod.MEDIAN,
        variance_method=CellProfilerVarianceMethod.MEDIAN_ABSOLUTE_DEVIATION,
        number_of_deviations=4,
        manual_threshold=0.5,
    )
    np.testing.assert_allclose(final_threshold, expected_threshold)
    assert final_threshold == original_threshold
    np.testing.assert_array_equal(binary, image >= expected_threshold)


def test_identify_primary_objects_basic_mode_fills_holes_like_cellprofiler():
    assert FillHolesOption.AFTER_DECLUMP.before_declump_requested(
        use_advanced_settings=False
    )
    assert FillHolesOption.NEVER.after_declump_requested(use_advanced_settings=False)
    assert not FillHolesOption.AFTER_DECLUMP.before_declump_requested(
        use_advanced_settings=True
    )


def test_cellprofiler_basic_threshold_uses_native_default_smoothing():
    calls = {}

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_method"] = kwargs["threshold_method"]
        return 0.25

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_smoothing"] = smoothing
        return (np.asarray(pixel_data) >= threshold, 0.0)

    cellprofiler_threshold(
        np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3),
        use_advanced_settings=False,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=True,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        global_threshold_function=get_global_threshold,
        apply_threshold_function=apply_threshold,
    )
    assert calls == {
        "threshold_method": CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
        "application_smoothing": CELLPROFILER_BASIC_THRESHOLD_SMOOTHING_SCALE,
    }


def test_cellprofiler_threshold_passes_mask_to_native_thresholds():
    calls = []
    image = np.array([[0.0, 0.2], [0.8, 1.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]])

    def get_global_threshold(pixel_data, **kwargs):
        calls.append(("threshold", kwargs["mask"]))
        return 0.5 * kwargs["threshold_correction_factor"]

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls.append(("apply", mask))
        return (np.asarray(pixel_data) >= threshold, 0.0)

    binary, final_threshold, original_threshold = cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=0.7,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=0.0,
        adaptive_window_size=10,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        mask=mask,
        global_threshold_function=get_global_threshold,
        apply_threshold_function=apply_threshold,
    )
    assert final_threshold == 0.35
    assert original_threshold == 0.5
    np.testing.assert_array_equal(binary, np.array([[False, False], [True, False]]))
    assert [name for name, _mask in calls] == ["threshold", "apply"]
    assert all((np.array_equal(_mask, mask) for _name, _mask in calls))


def test_identify_primary_objects_applies_threshold_smoothing_to_binary_mask(
    monkeypatch,
) -> None:
    calls = {}

    def fake_threshold_tuple(self, **_kwargs):
        settings = self.settings.normalized()
        calls["threshold_smoothing_scale"] = settings.threshold_smoothing_scale
        calls["smooth_threshold_application"] = settings.smooth_threshold_application
        return (np.zeros_like(self.image, dtype=bool), 0.1, 0.1)

    monkeypatch.setattr(
        thresholding_backend.CellProfilerThresholdRequest,
        "threshold_tuple",
        fake_threshold_tuple,
    )
    identify_primary_objects(
        np.zeros((8, 8), dtype=np.float32),
        use_advanced_settings=True,
        smoothing_filter_size=10,
        threshold_smoothing_scale=1.3488,
        min_diameter=2,
        max_diameter=8,
        dtype_config=DtypeConfig(),
    )
    assert calls["threshold_smoothing_scale"] == pytest.approx(1.3488)
    assert calls["smooth_threshold_application"] is True


def test_identify_primary_objects_accepts_nominal_options_directly():
    image = np.zeros((8, 8), dtype=np.float32)
    image[2:6, 2:6] = 1.0
    _image, _measurements, labels = identify_primary_objects(
        image,
        min_diameter=2,
        max_diameter=8,
        exclude_size=False,
        exclude_border_objects=False,
        unclump_method=UnclumpMethod.NONE,
        watershed_method=WatershedMethod.NONE,
        fill_holes=FillHolesOption.AFTER_BOTH,
        limit_erase=ExcessObjectHandling.CONTINUE,
        threshold_method=CellProfilerThresholdMethod.MANUAL,
        manual_threshold=0.5,
        dtype_config=DtypeConfig(),
    )
    assert labels.labels.max() == 1


def test_identify_primary_objects_threshold_diagnostics_use_pre_fill_binary(
    monkeypatch,
):
    threshold_binary = np.zeros((7, 7), dtype=bool)
    threshold_binary[2:5, 2:5] = True
    threshold_binary[3, 3] = False
    captured = {}

    def fake_threshold_tuple(self, **_kwargs):
        return (threshold_binary.copy(), 0.5, 0.5)

    def fake_diagnostics(image, binary, **kwargs):
        captured["binary"] = np.asarray(binary, dtype=bool).copy()
        return types.SimpleNamespace(
            original_threshold=0.5, weighted_variance=0.0, sum_of_entropies=0.0
        )

    monkeypatch.setattr(
        thresholding_backend.CellProfilerThresholdRequest,
        "threshold_tuple",
        fake_threshold_tuple,
    )
    monkeypatch.setattr(
        thresholding_backend, "cellprofiler_threshold_diagnostics", fake_diagnostics
    )
    identify_primary_objects(
        np.ones((7, 7), dtype=np.float32),
        min_diameter=1,
        max_diameter=10,
        exclude_size=False,
        exclude_border_objects=False,
        unclump_method=UnclumpMethod.NONE,
        watershed_method=WatershedMethod.NONE,
        fill_holes=FillHolesOption.AFTER_BOTH,
        threshold_method=CellProfilerThresholdMethod.MANUAL,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(captured["binary"], threshold_binary)


def test_identify_primary_objects_does_not_size_filter_after_hole_fill() -> None:
    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 1] = 1.0
    image[1:6, 5] = 1.0
    image[1, 1:6] = 1.0
    image[5, 1:6] = 1.0
    _image, _measurements, labels = identify_primary_objects(
        image,
        min_diameter=1,
        max_diameter=5,
        exclude_size=True,
        exclude_border_objects=False,
        unclump_method=UnclumpMethod.NONE,
        watershed_method=WatershedMethod.NONE,
        fill_holes=FillHolesOption.AFTER_DECLUMP,
        threshold_method=CellProfilerThresholdMethod.MANUAL,
        threshold_smoothing_scale=0.0,
        manual_threshold=0.5,
        dtype_config=DtypeConfig(),
    )
    assert int(np.count_nonzero(labels.labels)) == 25


def test_watershed_xy_downsample_factors_preserve_leading_axes():
    from openhcs.processing.backends.cellprofiler.watershed import (
        watershed_connected_components,
        watershed_regionprops_stats,
        watershed_xy_downsample_factors,
    )

    assert watershed_xy_downsample_factors(2, 2) == (2.0, 2.0)
    assert watershed_xy_downsample_factors(3, 2) == (1.0, 2.0, 2.0)
    assert watershed_xy_downsample_factors(4, 2) == (1.0, 1.0, 2.0, 2.0)
    labels = watershed_connected_components(np.ones((2, 3, 4, 5), dtype=bool))
    assert labels.shape == (2, 3, 4, 5)
    assert labels.dtype == np.int32
    object_count, mean_area = watershed_regionprops_stats(labels)
    assert object_count == 2
    assert mean_area == 60.0


def test_identify_primary_objects_declumping_maxima_geometry_matches_public_semantics():

    def geometry_tuple(**settings):
        geometry = DeclumpingMaximaGeometry.from_cellprofiler_settings(**settings)
        return (geometry.image_resize_factor, geometry.suppress_size)

    assert geometry_tuple(
        min_diameter=8,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
    ) == (1.0, 8.0 / 1.5)
    assert geometry_tuple(
        min_diameter=20,
        low_res_maxima=True,
        automatic_suppression=True,
        maxima_suppression_size=7.0,
    ) == (0.5, CELLPROFILER_LOW_RES_AUTO_MAXIMA_SUPPRESSION_SIZE)
    assert geometry_tuple(
        min_diameter=20,
        low_res_maxima=True,
        automatic_suppression=False,
        maxima_suppression_size=7.0,
    ) == (0.5, 4.0)
    assert geometry_tuple(
        min_diameter=4,
        low_res_maxima=True,
        automatic_suppression=False,
        maxima_suppression_size=4.0,
    ) == (1.0, 4.0)
    assert manual_declumping_size(0) == 0.0
    assert manual_declumping_size(4) == 4.0


def test_identify_primary_objects_declumping_footprint_respects_min_diameter():

    class FakeMorphology:
        def __init__(self):
            self.calls = []

        def declumping_suppression_footprint(
            self, suppress_size, *, min_diameter, declump_method
        ):
            self.calls.append((suppress_size, min_diameter, declump_method))
            return np.ones((1, 1), dtype=bool)

    morphology = FakeMorphology()
    morphology.declumping_suppression_footprint(
        4, min_diameter=5, declump_method=CellProfilerDeclumpMethod.INTENSITY
    )
    morphology.declumping_suppression_footprint(
        4, min_diameter=4, declump_method=CellProfilerDeclumpMethod.INTENSITY
    )
    morphology.declumping_suppression_footprint(
        2, min_diameter=1, declump_method=CellProfilerDeclumpMethod.SHAPE
    )
    assert morphology.calls == [
        (4, 5, CellProfilerDeclumpMethod.INTENSITY),
        (4, 4, CellProfilerDeclumpMethod.INTENSITY),
        (2, 1, CellProfilerDeclumpMethod.SHAPE),
    ]


def test_cellprofiler_threshold_can_apply_unsmoothed_threshold():
    calls = {}

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_smoothing"] = smoothing
        return (np.asarray(pixel_data) >= threshold, 0.0)

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.TRIANGLE,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        smooth_threshold_application=False,
        global_threshold_function=get_global_threshold,
        apply_threshold_function=apply_threshold,
    )
    np.testing.assert_array_equal(calls["threshold_pixels"], image)
    assert calls["application_smoothing"] == 0.0


def test_cellprofiler_global_otsu_uses_raw_threshold_estimate():
    calls = {}

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_pixels"] = np.asarray(pixel_data).copy()
        calls["application_smoothing"] = smoothing
        return (np.asarray(pixel_data) >= threshold, 0.0)

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.OTSU,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        smooth_threshold_application=True,
        global_threshold_function=get_global_threshold,
        apply_threshold_function=apply_threshold,
    )
    np.testing.assert_array_equal(calls["threshold_pixels"], image)
    np.testing.assert_array_equal(calls["application_pixels"], image)
    assert calls["application_smoothing"] == 2.0


def test_cellprofiler_global_robust_background_uses_raw_threshold_estimate():
    calls = {}

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    def apply_threshold(pixel_data, *, threshold, mask, smoothing):
        calls["application_pixels"] = np.asarray(pixel_data).copy()
        calls["application_smoothing"] = smoothing
        return (np.asarray(pixel_data) >= threshold, 0.0)

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.ROBUST_BACKGROUND,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        smooth_threshold_application=True,
        global_threshold_function=get_global_threshold,
        apply_threshold_function=apply_threshold,
    )
    np.testing.assert_array_equal(calls["threshold_pixels"], image)
    np.testing.assert_array_equal(calls["application_pixels"], image)
    assert calls["application_smoothing"] == 2.0


def test_cellprofiler_minimum_cross_entropy_uses_unsmoothed_threshold_estimate():
    calls = {}

    def get_global_threshold(pixel_data, **kwargs):
        calls["threshold_pixels"] = np.asarray(pixel_data).copy()
        return 0.5

    image = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    cellprofiler_threshold(
        image,
        use_advanced_settings=True,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        threshold_method=CellProfilerThresholdMethod.MINIMUM_CROSS_ENTROPY,
        otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
        assign_middle_to_foreground=CellProfilerThresholdAssignment.FOREGROUND,
        log_transform=False,
        threshold_correction_factor=1.0,
        threshold_min=0.0,
        threshold_max=1.0,
        threshold_smoothing_scale=2.0,
        adaptive_window_size=50,
        lower_outlier_fraction=0.05,
        upper_outlier_fraction=0.05,
        averaging_method=CellProfilerAveragingMethod.MEAN,
        variance_method=CellProfilerVarianceMethod.STANDARD_DEVIATION,
        number_of_deviations=2,
        manual_threshold=0.5,
        global_threshold_function=get_global_threshold,
    )
    np.testing.assert_array_equal(calls["threshold_pixels"], image)


def test_cellprofiler_threshold_diagnostics_matches_reference_formula():
    import centrosome.threshold

    rng = np.random.default_rng(7)
    image = rng.random((16, 17), dtype=np.float32)
    mask = rng.random(image.shape) > 0.2
    binary = image > 0.45
    diagnostics = cellprofiler_threshold_diagnostics(
        image, binary, final_threshold=0.45, original_threshold=0.4, mask=mask
    )
    np.testing.assert_allclose(
        diagnostics.weighted_variance,
        centrosome.threshold.weighted_variance(image, mask, binary),
    )
    np.testing.assert_allclose(
        diagnostics.sum_of_entropies,
        centrosome.threshold.sum_of_entropies(image, mask, binary),
    )


def test_identify_primary_objects_filters_crop_mask_border_objects():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[1, 2] = 2
    labels[4, 4] = 3
    mask = np.zeros_like(labels, dtype=bool)
    mask[1:5, 1:5] = True
    filtered = filter_border_objects(labels, image_mask=mask)
    assert 1 in filtered
    assert 2 not in filtered
    assert 3 not in filtered


def test_identify_primary_objects_ignores_threshold_only_mask_border():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[1, 2] = 2
    mask = np.zeros_like(labels, dtype=bool)
    mask[1:5, 1:5] = True
    metadata = ImagePayloadMetadata(mask_defines_border=False)
    filtered = filter_border_objects(labels, image_mask=mask, image_metadata=metadata)
    assert 1 in filtered
    assert 2 in filtered


def test_identify_primary_objects_keeps_crop_local_nonphysical_edges():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[1:3, 0:2] = 2
    mask = np.ones_like(labels, dtype=bool)
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(12, 12),
        output_shape_yx=labels.shape,
        offset_yx=(3, 4),
        physical_border_edges_yx=(False, False, False, False),
    )
    filtered = filter_border_objects(labels, image_mask=mask, image_metadata=metadata)
    assert 1 in filtered
    assert 2 in filtered


def test_identify_primary_objects_removes_true_physical_edge_objects():
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[2:4, 2:4] = 1
    labels[0:2, 2:4] = 2
    metadata = ImagePayloadMetadata(source_dtype="float32").with_spatial_crop(
        input_shape_yx=(10, 10),
        output_shape_yx=labels.shape,
        offset_yx=(0, 2),
        physical_border_edges_yx=(True, False, False, False),
    )
    filtered = filter_border_objects(
        labels, image_mask=np.ones_like(labels, dtype=bool), image_metadata=metadata
    )
    assert 1 in filtered
    assert 2 not in filtered


def test_identify_primary_objects_filters_stacked_label_sizes_planewise():
    labels = np.zeros((2, 6, 6), dtype=np.int32)
    labels[0, 1:3, 1:3] = 1
    labels[0, 3:6, 3:6] = 2
    labels[1, 1:4, 1:4] = 10
    labels[1, 4:6, 4:6] = 20
    small_removed, final = filter_labels_by_diameter_range(
        labels, min_diameter=2.5, max_diameter=3.5
    )
    assert 1 not in small_removed[0]
    assert 2 in small_removed[0]
    assert 10 in small_removed[1]
    assert 20 not in small_removed[1]
    assert 1 not in final[0]
    assert 2 in final[0]
    assert 10 in final[1]
    assert 20 not in final[1]


def test_identify_primary_objects_border_filter_rejects_undeclared_label_stack():
    labels = np.zeros((2, 5, 5), dtype=np.int32)
    with pytest.raises(ValueError, match="PURE_2D processing contract"):
        filter_border_objects(labels, image_mask=np.ones_like(labels, dtype=bool))


def test_identify_primary_objects_border_filter_rejects_singleton_mask_stack():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:3, 1:3] = 1
    with pytest.raises(ValueError, match="shapes must match exactly"):
        filter_border_objects(
            labels, image_mask=np.ones((1, *labels.shape), dtype=bool)
        )


def test_cellprofiler_legacy_watershed_keeps_descending_pixel_priority():
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    image = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    markers = np.array([[1, 0, 2]], dtype=np.int32)
    labels = cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=np.ones_like(image, dtype=bool),
        connectivity=np.ones((1, 3), dtype=bool),
    )
    np.testing.assert_array_equal(
        object_label_dense_array(labels),
        np.array([[1, 1, 2]], dtype=np.int32),
    )


def test_cellprofiler4_marker_watershed_matches_cellprofiler_source_path():
    from openhcs.processing.backends.cellprofiler.watershed import (
        WatershedDeclumpMethod,
        WatershedMethod,
        watershed_cellprofiler4,
    )

    image = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    markers = np.array([[1, 0, 2]], dtype=np.int32)
    _image, _stats, labels = watershed_cellprofiler4(
        image,
        topology_inputs=(markers, np.ones_like(image, dtype=bool)),
        watershed_method=WatershedMethod.MARKERS,
        declump_method=WatershedDeclumpMethod.SHAPE,
        use_advanced_settings=False,
    )
    np.testing.assert_array_equal(
        object_label_dense_array(labels),
        np.array([[1, 1, 2]], dtype=np.int32),
    )


def test_cellprofiler_fast_legacy_watershed_matches_reference_path():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    rng = np.random.default_rng(123)
    for _case in range(20):
        image = rng.integers(0, 6, size=(8, 7)).astype(float) / 5.0
        mask = rng.random((8, 7)) > 0.15
        markers = np.zeros((8, 7), dtype=np.int32)
        coords = np.argwhere(mask)
        selected = coords[rng.choice(len(coords), size=4, replace=False)]
        for label, (y, x) in enumerate(selected, start=1):
            markers[y, x] = label
        reference = cellprofiler_legacy_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=np.ones((3, 3), dtype=bool),
            backend_provider=CellProfilerBackendProvider.NATIVE,
        )
        fast = cellprofiler_legacy_watershed(
            image,
            markers=markers,
            mask=mask,
            connectivity=np.ones((3, 3), dtype=bool),
            backend_provider=CellProfilerBackendProvider.NUMBA,
        )
        np.testing.assert_array_equal(fast, reference)


def test_cellprofiler_legacy_watershed_handles_stacked_planes_planewise():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    image = np.stack(
        (
            np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 0.5, 0.0]], dtype=np.float64),
        )
    )
    markers = np.stack(
        (np.array([[1, 0, 2]], dtype=np.int32), np.array([[10, 0, 20]], dtype=np.int32))
    )
    mask = np.ones_like(image, dtype=bool)
    labels = cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=np.ones((1, 3), dtype=bool),
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )
    expected = np.stack(
        (
            np.array([[1, 1, 2]], dtype=np.int32),
            np.array([[10, 10, 20]], dtype=np.int32),
        )
    )
    np.testing.assert_array_equal(labels, expected)


def test_cellprofiler_legacy_watershed_scalar_connectivity_is_volumetric():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.watershed import (
        cellprofiler_legacy_watershed,
    )

    image = np.zeros((2, 3, 3), dtype=np.float64)
    markers = np.zeros_like(image, dtype=np.int32)
    markers[0, 1, 1] = 1
    mask = np.ones_like(image, dtype=bool)
    labels = cellprofiler_legacy_watershed(
        image,
        markers=markers,
        mask=mask,
        connectivity=1,
        backend_provider=CellProfilerBackendProvider.NUMBA,
    )
    np.testing.assert_array_equal(labels, np.ones_like(markers))


def test_cellprofiler_fast_legacy_watershed_uses_required_numba_backend():
    from openhcs.constants.constants import MemoryType
    from openhcs.processing.backends.cellprofiler.watershed import (
        LegacyWatershedBackendStrategy,
        NumbaNumpyLegacyWatershedBackendStrategy,
    )

    watershed_backend = importlib.import_module(
        "openhcs.processing.backends.cellprofiler.watershed"
    )
    assert watershed_backend._legacy_watershed_raveled_numba is not None
    assert (
        type(LegacyWatershedBackendStrategy.for_memory_type(MemoryType.NUMPY))
        is NumbaNumpyLegacyWatershedBackendStrategy
    )


def test_measure_texture_uses_cellprofiler_haralick_backend(monkeypatch):
    from openhcs.processing.backends.cellprofiler.texture import (
        MeasureTextureModule,
        measure_texture,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    def haralick(pixel_data, distance, ignore_zeros=False):
        calls.append((pixel_data.copy(), distance, ignore_zeros))
        features = np.arange(52, dtype=float).reshape(4, 13)
        features[1, 2] = np.nan
        return features

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)
    image = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    _, measurements = measure_texture(
        image,
        scale=2,
        gray_levels=8,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )
    assert calls[0][1:] == (2, False)
    assert calls[0][0].dtype == np.uint8
    assert calls[0][0].max() <= 7
    assert tuple(field.name for field in measurements.fields) == (
        "slice_index",
        "scale",
        "direction",
        "gray_levels",
        *(
            feature.measurement_row_field_name
            for feature in MeasureTextureModule.MeasurementFeature
        ),
    )
    rows = measurements.row_mappings()
    assert rows[0]["contrast"] == 1.0
    assert rows[1]["correlation"] == 0.0
    row = rows[1]
    assert row["scale"] == 2
    assert row["direction"] == 1
    assert row["gray_levels"] == 8
    assert row["contrast"] == 14.0
    assert row["correlation"] == 0.0


def test_measure_texture_emits_all_requested_scales(monkeypatch):
    from openhcs.processing.backends.cellprofiler.texture import measure_texture
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    def haralick(pixel_data, distance, ignore_zeros=False):
        calls.append((distance, ignore_zeros))
        return np.full((4, 13), float(distance), dtype=float)

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)
    image = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)
    _, measurements = measure_texture(
        image,
        scale=(2, 4),
        gray_levels=8,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )
    assert calls == [(2, False), (4, False)]
    assert list(measurements.columns["scale"]) == [2] * 4 + [4] * 4


def test_measure_texture_objects_uses_cellprofiler_object_backend(monkeypatch):
    from openhcs.processing.backends.cellprofiler.texture import (
        MeasureTextureModule,
        measure_texture_objects,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    def haralick(pixel_data, distance, ignore_zeros=False):
        calls.append((pixel_data.copy(), distance, ignore_zeros))
        features = np.ones((4, 13), dtype=float)
        features[0, 3] = np.inf
        return features

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)
    image = np.full((5, 5), 0.5, dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1
    _, measurements = measure_texture_objects(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        scale=1,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )
    assert calls[0][1:] == (1, True)
    assert calls[0][0].dtype == np.uint8
    assert tuple(field.name for field in measurements.fields) == (
        "slice_index",
        "object_label",
        "scale",
        "direction",
        "gray_levels",
        *(
            feature.measurement_row_field_name
            for feature in MeasureTextureModule.MeasurementFeature
        ),
    )
    first_measurement = measurements.row_mappings()[0]
    assert first_measurement["object_label"] == 1
    assert first_measurement["variance"] == 0.0


def test_measure_texture_objects_emits_and_splits_both_scopes(monkeypatch):
    from openhcs.interop.cellprofiler.measurement_scope import (
        CellProfilerMeasurementTargetScope,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.texture import (
        MeasureTextureObjectMeasurementRowPolicy,
        measure_texture_objects,
    )

    def haralick(pixel_data, distance, ignore_zeros=False):
        del pixel_data, distance, ignore_zeros
        return np.ones((4, 13), dtype=float)

    mahotas_module = types.ModuleType("mahotas")
    features_module = types.ModuleType("mahotas.features")
    features_module.haralick = haralick
    mahotas_module.features = features_module
    monkeypatch.setitem(sys.modules, "mahotas", mahotas_module)
    monkeypatch.setitem(sys.modules, "mahotas.features", features_module)
    image = np.full((5, 5), 0.5, dtype=np.float32)
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1

    _output, rows = measure_texture_objects(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        measurement_scope=CellProfilerMeasurementTargetScope.BOTH,
        scale=1,
        haralick_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )
    object_rows, image_rows = (
        MeasureTextureObjectMeasurementRowPolicy().split_scoped_rows(rows)
    )

    assert image_rows.row_count() == 4
    assert object_rows.row_count() == 4


def test_measure_object_neighbors_prepare_uses_nominal_label_payload():
    from openhcs.core.callable_contract import prepare_processing_callable
    from openhcs.processing.backends.cellprofiler.neighbors import (
        measure_object_neighbors,
    )

    prepare_processing_callable(measure_object_neighbors)


def test_measure_texture_objects_rejects_unprojected_plane_domain_stack():
    from openhcs.processing.backends.cellprofiler.texture import measure_texture_objects

    image = np.ones((5, 5), dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    [
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [2, 2, 0, 0, 0],
                        [2, 2, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ),
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)), scope=ObjectLabelDomainScope.PLANE
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    with pytest.raises(ValueError, match="runtime-projected 2-D"):
        measure_texture_objects(image, labels, scale=1, dtype_config=DtypeConfig())


def test_measure_texture_objects_emits_only_present_object_labels():
    from openhcs.processing.backends.cellprofiler.texture import measure_texture_objects

    image = np.ones((5, 5), dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                [
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(declared_object_count=2),
    )
    _, measurements = measure_texture_objects(
        image, labels, scale=1, dtype_config=DtypeConfig()
    )
    assert {
        measurement["object_label"] for measurement in measurements.row_mappings()
    } == {1}


def test_measure_texture_objects_preserves_runtime_projected_label_domain():
    from openhcs.processing.backends.cellprofiler.texture import measure_texture_objects

    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    [
                        [1, 1, 0, 0, 0],
                        [1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                    [
                        [3, 3, 0, 0, 0],
                        [3, 3, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                    ],
                ),
                dtype=np.int32,
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (3, 4)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    results = []
    for slice_index in range(2):
        projected = RuntimeSliceProjection.value_for_slice(
            labels,
            RuntimePlaneAxisValueProjection.from_selected_plane(
                axis=RuntimePlaneAxis.RUNTIME_SLICE,
                plane_index=slice_index,
                axis_size=2,
            ),
        )
        results.append(
            measure_texture_objects.__wrapped__(
                np.ones((5, 5), dtype=np.float32),
                projected,
                scale=1,
            )
        )
    assert list(results[0][1].columns["object_label"][::4]) == [1]
    assert list(results[1][1].columns["object_label"][::4]) == [3]


def test_numba_haralick_backend_exactly_matches_mahotas_reference():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.texture import (
        HaralickTextureBackendStrategy,
    )

    rng = np.random.default_rng(7123)
    native_backend = HaralickTextureBackendStrategy.for_memory_type(
        backend_provider=CellProfilerBackendProvider.NATIVE
    )
    numba_backend = HaralickTextureBackendStrategy.for_memory_type(
        backend_provider=CellProfilerBackendProvider.NUMBA
    )
    for ignore_zeros in (False, True):
        for scale in (1, 2, 3):
            image = rng.integers(0, 16, size=(14, 13), dtype=np.uint8)
            image[0:2, 0:2] = 0
            expected = native_backend.haralick_features(
                image, scale=scale, ignore_zeros=ignore_zeros
            )
            actual = numba_backend.haralick_features(
                image, scale=scale, ignore_zeros=ignore_zeros
            )
            np.testing.assert_array_equal(actual, expected)


def test_object_texture_crop_backend_matches_regionprops_intensity_images():
    from skimage.measure import regionprops

    image = np.arange(36, dtype=np.uint8).reshape(6, 6)
    labels = np.zeros((6, 6), dtype=np.int32)
    labels[1:4, 1:4] = 3
    labels[2, 2] = 0
    labels[0:2, 4:6] = 7
    labels[5, 0] = 12
    backend = ObjectTextureCropBackendStrategy.for_callable(
        test_object_texture_crop_backend_matches_regionprops_intensity_images
    )
    object_labels, intensity_crops = backend.object_intensity_crops(image, labels)
    expected_props = regionprops(labels, intensity_image=image)
    assert object_labels.tolist() == [prop.label for prop in expected_props]
    assert len(intensity_crops) == len(expected_props)
    for intensity_crop, prop in zip(intensity_crops, expected_props, strict=True):
        np.testing.assert_array_equal(intensity_crop, prop.intensity_image)


def test_measure_object_intensity_uses_cellprofiler_mad_interpolation():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.array([[0.0, 10.0, 20.0]], dtype=np.float32)
    labels = np.array([[1, 1, 1]], dtype=np.int32)
    _, measurements = measure_object_intensity(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert measurements[0].median_intensity == 15.0
    assert measurements[0].mad_intensity == 10.0


def test_measure_object_intensity_rejects_unprojected_replicated_rgb_image():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    plane = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    image = np.repeat(plane[..., None], 3, axis=-1)
    labels = np.array([[1, 1], [0, 2]], dtype=np.int32)
    with pytest.raises(NotImplementedError, match="exact image domain"):
        measure_object_intensity(
            image,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=labels),
                domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
            ),
            dtype_config=DtypeConfig(),
        )


def test_measure_object_intensity_measures_3d_objects_as_single_volume_domain():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.ones((3, 4, 5), dtype=np.float32)
    labels = np.zeros((3, 4, 5), dtype=np.int32)
    labels[:, 1:3, 2:4] = 1
    _, measurements = measure_object_intensity(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert len(measurements) == 1
    assert measurements[0].integrated_intensity == 12.0
    assert measurements[0].center_mass_intensity_z == 1.0
    assert measurements[0].max_intensity_z == 2.0


def test_measure_object_intensity_preserves_sparse_object_ids_without_dense_lookup():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.ones((3, 4), dtype=np.float32)
    labels = np.zeros((3, 4), dtype=np.int32)
    labels[1, 1:3] = 100000
    _, measurements = measure_object_intensity(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(100000,)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert [measurement.object_label for measurement in measurements] == [100000]
    assert measurements[0].integrated_intensity == 2.0


def test_measure_object_intensity_honors_declared_object_count_domain():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.ones((2, 3), dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 3], [0, 0, 0]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_count=4),
    )
    _, measurements = measure_object_intensity(
        image, labels, dtype_config=DtypeConfig()
    )
    by_label = {measurement.object_label: measurement for measurement in measurements}
    assert tuple(by_label) == (1, 2, 3, 4)
    assert by_label[1].integrated_intensity == 1.0
    assert by_label[2].integrated_intensity == 0.0
    assert by_label[3].integrated_intensity == 1.0
    assert np.isnan(by_label[4].integrated_intensity)


def test_measure_object_intensity_preserves_sparse_declared_ids():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.ones((2, 3), dtype=np.float32)
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray([[1, 0, 3], [0, 0, 5]], dtype=np.int32)
        ),
        domain=ObjectLabelDomain(declared_object_ids=(1, 3, 5)),
    )
    _, measurements = measure_object_intensity(
        image, labels, dtype_config=DtypeConfig()
    )
    by_label = {measurement.object_label: measurement for measurement in measurements}
    assert tuple(by_label) == (1, 3, 5)


def test_measure_object_intensity_maximum_position_uses_cellprofiler_quicksort_order():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.zeros((1, 17), dtype=np.float32)
    image[0, (0, 8, 9)] = 1.0
    labels = np.ones(image.shape, dtype=np.int32)
    _, measurements = measure_object_intensity(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert len(measurements) == 1
    assert measurements[0].max_intensity_x == 8.0
    assert measurements[0].max_intensity_y == 0.0


def test_measure_object_intensity_rejects_unprojected_true_color_image():
    from openhcs.processing.backends.cellprofiler.intensity import (
        measure_object_intensity,
    )

    image = np.zeros((2, 2, 3), dtype=np.float32)
    image[..., 1] = 1.0
    labels = np.ones((2, 2), dtype=np.int32)
    with pytest.raises(NotImplementedError, match="exact image domain"):
        measure_object_intensity(
            image,
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=labels),
                domain=ObjectLabelDomain(declared_object_ids=(1,)),
            ),
            dtype_config=DtypeConfig(),
        )


def test_measure_image_quality_uses_openhcs_image_quality_backend():
    from openhcs.processing.backends.cellprofiler.image_quality import (
        image_quality_haralick_correlation,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(100, dtype=np.float32).reshape(10, 10) / 100.0
    default_value = image_quality_haralick_correlation(image, 2)
    centrosome_value = image_quality_haralick_correlation(
        image, 2, backend_provider=CellProfilerBackendProvider.CENTROSOME
    )
    assert np.isclose(default_value, centrosome_value, rtol=1e-06, atol=1e-06)


def test_measure_image_quality_uses_openhcs_power_spectrum_backend(monkeypatch):
    import openhcs.processing.backends.cellprofiler.image_quality as module
    from openhcs.processing.backends.cellprofiler.image_quality import (
        image_quality_power_spectrum_slope,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    calls = []

    class Backend:
        def radial_power_spectrum(self, pixel_data):
            calls.append(pixel_data.copy())
            radii = np.array([1.0, 2.0, 4.0])
            magnitude = np.array([1.0, 1.0, 1.0])
            power = radii ** (-2)
            return (radii, magnitude, power)

    def backend_factory(*, backend_provider=None):
        assert backend_provider is CellProfilerBackendProvider.NUMBA
        return Backend()

    monkeypatch.setattr(module, "image_quality_backend", backend_factory)
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.isclose(
        image_quality_power_spectrum_slope(
            image, backend_provider=CellProfilerBackendProvider.NUMBA
        ),
        -2.0,
    )
    np.testing.assert_array_equal(calls[0], image)


def test_measure_image_quality_uses_native_centrosome_otsu_semantics(monkeypatch):
    import centrosome.threshold

    from openhcs.processing.backends.cellprofiler.image_quality import (
        ImageQualityOtsuObjective,
        ImageQualityThresholdMethod,
        image_quality_threshold,
    )
    from openhcs.processing.backends.cellprofiler.thresholding import (
        CellProfilerOtsuMethod,
        CellProfilerThresholdAssignment,
    )

    calls = []

    def get_threshold(threshold_method, threshold_scope, image, **kwargs):
        calls.append((threshold_method, threshold_scope, image.copy(), kwargs))
        return (0.125, 0.25)

    monkeypatch.setattr(centrosome.threshold, "get_threshold", get_threshold)
    image = np.arange(16, dtype=np.float64).reshape(4, 4) / 16.0

    assert (
        image_quality_threshold(
            image,
            ImageQualityThresholdMethod.OTSU,
            object_fraction=0.2,
            otsu_class_count=CellProfilerOtsuMethod.THREE_CLASS,
            otsu_objective=ImageQualityOtsuObjective.ENTROPY,
            assign_middle_to_foreground=(CellProfilerThresholdAssignment.BACKGROUND),
        )
        == 0.25
    )
    assert len(calls) == 1
    threshold_method, threshold_scope, values, kwargs = calls[0]
    assert threshold_method == centrosome.threshold.TM_OTSU
    assert threshold_scope == centrosome.threshold.TM_GLOBAL
    np.testing.assert_array_equal(values, image.astype(np.float32))
    assert kwargs == {
        "object_fraction": 0.2,
        "two_class_otsu": False,
        "use_weighted_variance": False,
        "assign_middle_to_foreground": False,
    }


def test_measure_image_quality_constancy_check_matches_numpy_unique():
    from openhcs.processing.backends.cellprofiler.image_quality import (
        image_quality_has_multiple_unique_values,
    )

    cases = (
        np.array([[1.0, 1.0]], dtype=np.float32),
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[np.nan, np.nan]], dtype=np.float32),
        np.array([[np.nan, 1.0]], dtype=np.float32),
        np.array([[0.0, -0.0]], dtype=np.float32),
    )
    for image in cases:
        expected = len(np.unique(image)) > 1
        assert image_quality_has_multiple_unique_values(image) is expected


def test_measure_image_quality_log_log_slope_matches_lstsq():
    import scipy.linalg
    from openhcs.processing.backends.cellprofiler.image_quality import (
        _least_squares_log_log_slope_numba,
    )

    radii = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    power = radii ** (-1.75)
    idx = np.isfinite(np.log(power))
    design = np.hstack(
        (np.log(radii)[idx][:, np.newaxis], np.ones(radii.shape)[idx][:, np.newaxis])
    )
    expected = scipy.linalg.lstsq(design, np.log(power)[idx][:, np.newaxis])[0][0]
    assert np.isclose(
        _least_squares_log_log_slope_numba(radii, power),
        float(np.asarray(expected).ravel()[0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_measure_image_quality_local_focus_matches_grid_semantics():
    from scipy.ndimage import mean as ndimage_mean, sum as ndimage_sum
    from openhcs.processing.backends.cellprofiler.image_quality import (
        image_quality_local_focus_score,
    )

    image = np.arange(35, dtype=np.float32).reshape(5, 7) / 10.0
    scale = 3
    shape = image.shape
    i, j = np.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
    m, n = (np.array(shape) + scale - 1) // scale
    i = (i * float(m) / float(shape[0])).astype(int)
    j = (j * float(n) / float(shape[1])).astype(int)
    grid = i * n + j + 1
    grid_range = np.arange(0, m * n + 1, dtype=np.int32)
    local_means = np.nan_to_num(ndimage_mean(image, grid, grid_range), nan=0.0)
    local_squared_normalized = (image - local_means[grid]) ** 2
    grid_mask = (local_means != 0) & np.isfinite(local_means)
    nz_grid_range = grid_range[grid_mask]
    if nz_grid_range[0] == 0:
        nz_grid_range = nz_grid_range[1:]
        local_means = local_means[1:]
        grid_mask = grid_mask[1:]
    sums = ndimage_sum(local_squared_normalized, grid, nz_grid_range)
    pixel_counts = ndimage_sum(np.ones(shape), grid, nz_grid_range)
    valid_means = (
        local_means[grid_mask]
        if len(local_means) > len(nz_grid_range)
        else local_means[: len(nz_grid_range)]
    )
    expected_values = sums / (pixel_counts * valid_means[: len(sums)])
    expected_values = expected_values[np.isfinite(expected_values)]
    expected = float(np.var(expected_values) / np.median(expected_values))
    assert np.isclose(
        image_quality_local_focus_score(image, scale), expected, rtol=1e-12, atol=1e-12
    )


def test_measure_object_neighbors_accepts_explicit_centrosome_morphology(monkeypatch):
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.neighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    disk_calls = []
    outline_calls = []

    def strel_disk(radius):
        disk_calls.append(radius)
        return np.ones((3, 3), dtype=bool)

    def outline(labels):
        outline_calls.append(labels.copy())
        result = np.zeros_like(labels)
        result[labels > 0] = labels[labels > 0]
        return result

    def centers_of_labels(labels):
        centers = []
        for label in range(1, int(labels.max()) + 1):
            coords = np.argwhere(labels == label)
            centers.append(coords.mean(axis=0) if coords.size else (0.0, 0.0))
        return np.asarray(centers).T

    centrosome_module = types.ModuleType("centrosome")
    cpmorphology_module = types.ModuleType("centrosome.cpmorphology")
    outline_module = types.ModuleType("centrosome.outline")
    cpmorphology_module.strel_disk = strel_disk
    cpmorphology_module.centers_of_labels = centers_of_labels
    outline_module.outline = outline
    centrosome_module.cpmorphology = cpmorphology_module
    centrosome_module.outline = outline_module
    monkeypatch.setitem(sys.modules, "centrosome", centrosome_module)
    monkeypatch.setitem(sys.modules, "centrosome.cpmorphology", cpmorphology_module)
    monkeypatch.setitem(sys.modules, "centrosome.outline", outline_module)
    labels = np.zeros((7, 7), dtype=np.int32)
    labels[2, 2] = 1
    labels[2, 4] = 2
    _, _relationship, measurements = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=4,
        dtype_config=DtypeConfig(),
        morphology_backend_provider=CellProfilerBackendProvider.CENTROSOME,
        outline_backend_provider=CellProfilerBackendProvider.CENTROSOME,
    )
    assert disk_calls == [4, 4.5]
    np.testing.assert_array_equal(outline_calls[0], labels)
    assert len(measurements) == 2


def test_measure_object_neighbors_counts_small_removed_discarded_neighbors():
    from openhcs.processing.backends.cellprofiler.neighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 1] = 1
    small_removed = labels.copy()
    small_removed[3, 3] = 2
    label_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=labels, small_removed_labels=small_removed
        )
    )
    _, _with_discarded_relationship, with_discarded = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        label_payload,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=2,
        consider_discarded_objects=True,
        dtype_config=DtypeConfig(),
    )
    _, _without_discarded_relationship, without_discarded = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        label_payload,
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=2,
        consider_discarded_objects=False,
        dtype_config=DtypeConfig(),
    )
    assert with_discarded.row_mappings()[0]["number_of_neighbors"] == 1
    assert without_discarded.row_mappings()[0]["number_of_neighbors"] == 0


def test_measure_object_neighbors_returns_retained_count_image():
    from openhcs.processing.backends.cellprofiler.neighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    labels = np.zeros((7, 7), dtype=np.int32)
    labels[3, 2] = 1
    labels[3, 4] = 2
    count_image, relationship, measurements = measure_object_neighbors(
        np.zeros_like(labels, dtype=float),
        ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        distance_method=DistanceMethod.WITHIN,
        neighbor_distance=2,
        retain_neighbor_count_image=True,
        neighbor_count_colormap="hot",
        dtype_config=DtypeConfig(),
    )
    assert count_image.shape == (7, 7, 3)
    assert image_payload_metadata(count_image).source_channel_axis == -1
    np.testing.assert_array_equal(count_image[labels == 0], 0)
    assert np.any(count_image[labels > 0] > 0)
    assert relationship.source_ids == (1, 2)
    assert relationship.target_ids == (2, 1)
    assert [row["number_of_neighbors"] for row in measurements.iter_row_mappings()] == [
        1,
        1,
    ]


def test_measure_object_neighbors_rejects_singleton_label_stack_for_2d_image():
    from openhcs.processing.backends.cellprofiler.neighbors import (
        DistanceMethod,
        measure_object_neighbors,
    )

    labels = np.zeros((1, 7, 7), dtype=np.int32)
    labels[0, 3, 2] = 1
    labels[0, 3, 4] = 2
    with pytest.raises(ValueError, match="projected to one 2-D plane"):
        measure_object_neighbors(
            np.zeros((7, 7), dtype=float),
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
            distance_method=DistanceMethod.EXPAND,
            neighbor_distance=5,
            dtype_config=DtypeConfig(),
        )


def test_medianfilter_matches_cellprofiler_constant_default():
    from scipy.ndimage import median_filter as scipy_median_filter
    from openhcs.processing.backends.cellprofiler.median_filter import medianfilter

    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    image[1, 2] = 100.0
    image[3, 5] = -20.0
    observed = medianfilter(image, window_size=3, dtype_config=DtypeConfig())
    expected = scipy_median_filter(image, size=3, mode="constant").astype(image.dtype)
    np.testing.assert_array_equal(observed, expected)


def test_medianfilter_honors_explicit_reflect_mode():
    from scipy.ndimage import median_filter as scipy_median_filter
    from openhcs.processing.backends.cellprofiler.median_filter import medianfilter
    from openhcs.processing.backends.processors.method_axes import ScipyBoundaryMode

    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    image[1, 2] = 100.0
    image[3, 5] = -20.0
    observed = medianfilter(
        image,
        window_size=3,
        mode=ScipyBoundaryMode.REFLECT,
        dtype_config=DtypeConfig(),
    )
    expected = scipy_median_filter(image, size=3, mode="reflect").astype(image.dtype)
    np.testing.assert_array_equal(observed, expected)


def test_medianfilter_vectorized_volume_path_matches_scipy_constant():
    from scipy.ndimage import median_filter as scipy_median_filter
    from openhcs.processing.backends.cellprofiler.median_filter import (
        median_filter_backend,
    )
    from openhcs.processing.backends.processors.method_axes import ScipyBoundaryMode

    rng = np.random.default_rng(123)
    image = rng.random((5, 9, 7), dtype=np.float32)
    image[0, 0, 0] = 0.0
    image[-1, -1, -1] = 1.0
    observed = median_filter_backend().vectorized_window_filter(
        image,
        window_size=5,
        mode=ScipyBoundaryMode.CONSTANT,
    )
    expected = scipy_median_filter(image, size=5, mode="constant").astype(image.dtype)
    assert observed is not None
    np.testing.assert_array_equal(observed, expected)


def test_medianfilter_high_cardinality_volume_uses_exact_vector_path(monkeypatch):
    from scipy.ndimage import median_filter as scipy_median_filter
    from openhcs.processing.backends.cellprofiler.median_filter import (
        median_filter_backend,
    )
    from openhcs.processing.backends.processors.method_axes import ScipyBoundaryMode

    image = np.linspace(0.0, 1.0, 17 * 64 * 64, dtype=np.float32).reshape((17, 64, 64))
    expected = scipy_median_filter(image, size=3, mode="constant").astype(image.dtype)
    backend = median_filter_backend()

    def reject_scipy_fallback(*_args, **_kwargs):
        raise AssertionError("exact vector path unexpectedly fell through to SciPy")

    monkeypatch.setattr(backend, "scipy_filter", reject_scipy_fallback)

    assert np.unique(image).size > np.iinfo(np.uint16).max
    observed = backend.filter(
        image,
        window_size=3,
        mode=ScipyBoundaryMode.CONSTANT,
    )
    np.testing.assert_array_equal(observed, expected)


def test_medianfilter_declares_flexible_slice_by_slice_semantics():
    from scipy.ndimage import median_filter as scipy_median_filter

    function = CellProfilerModule.require_module("MedianFilter").require_callable(
        "medianfilter"
    )
    image = np.arange(3 * 5 * 5, dtype=np.float32).reshape((3, 5, 5))

    assert function.__processing_contract__ is ProcessingContract.FLEXIBLE
    assert "slice_by_slice" in function.__signature__.parameters

    volumetric = function(image, window_size=3, slice_by_slice=False)
    planar = function(image, window_size=3, slice_by_slice=True)
    expected_volumetric = scipy_median_filter(image, size=3, mode="constant").astype(
        image.dtype
    )
    expected_planar = np.stack(
        tuple(scipy_median_filter(plane, size=3, mode="constant") for plane in image)
    ).astype(image.dtype)

    np.testing.assert_array_equal(volumetric, expected_volumetric)
    np.testing.assert_array_equal(planar, expected_planar)
    assert not np.array_equal(volumetric, planar)


def test_image_math_accepts_nominal_operation():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    result = image_math(
        image,
        operation=ImageMathOperation.INVERT,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_allclose(result, 1 - image)


def test_image_math_preserves_or_ignores_masked_image_payload():
    image = np.array([[0.0, 0.25], [0.5, 1.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]])
    payload = MaskedImagePayload(data=image, mask=mask)
    preserved = image_math(
        payload,
        operation=ImageMathOperation.INVERT,
        dtype_config=DtypeConfig(),
    )
    ignored = image_math(
        payload,
        operation=ImageMathOperation.INVERT,
        ignore_masks=True,
        dtype_config=DtypeConfig(),
    )
    assert isinstance(preserved, MaskedImagePayload)
    np.testing.assert_array_equal(preserved.mask, mask)
    np.testing.assert_allclose(preserved.data, (1 - image) * mask)
    assert isinstance(ignored, np.ndarray)
    np.testing.assert_allclose(ignored, 1 - image)


def test_image_math_combines_operand_masks_without_reexpanding_single_output():
    image = np.stack(
        (
            np.full((2, 3, 4), 0.1, dtype=np.float32),
            np.full((2, 3, 4), 0.2, dtype=np.float32),
            np.full((2, 3, 4), 0.3, dtype=np.float32),
        )
    )
    mask = np.stack(
        (
            np.ones((2, 3, 4), dtype=bool),
            np.ones((2, 3, 4), dtype=bool),
            np.ones((2, 3, 4), dtype=bool),
        )
    )
    mask[0, 0, 0, 0] = False
    mask[1, 0, 0, 1] = False
    mask[2, 0, 0, 2] = False
    payload = ImagePayloadBundleContext.from_payloads(
        tuple(
            ImagePayloadMetadata(
                source_image_names=(f"Operand{index}",),
            ).payload_with(image[index], mask[index])
            for index in range(image.shape[0])
        )
    ).compose()
    result = image_math(
        payload,
        operation=ImageMathOperation.ADD,
        factors=(1.0, 1.0, 1.0),
        dtype_config=DtypeConfig(),
    )
    assert isinstance(result, MaskedImagePayload)
    expected_mask = mask[0] & mask[1] & mask[2]
    assert result.data.shape == image.shape[1:]
    np.testing.assert_array_equal(result.mask, expected_mask)
    np.testing.assert_allclose(result.data, image.sum(axis=0) * expected_mask)


def test_image_math_preserves_source_plane_stack_as_single_operand():
    image = np.linspace(0.0, 1.0, 3 * 2 * 2, dtype=np.float32).reshape((3, 2, 2))
    payload = ImagePayloadMetadata(
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/input/z{index}.tif" for index in range(image.shape[0]))
        )
    ).payload_with(image, None)

    result = image_math(
        payload,
        operation=ImageMathOperation.INVERT,
        dtype_config=DtypeConfig(),
    )

    assert isinstance(result, ImageMetadataPayload)
    assert result.data.shape == image.shape
    np.testing.assert_allclose(result.data, 1.0 - image)


def test_image_math_reduces_multi_volume_bundle_across_source_axis():
    payloads = tuple(
        ImagePayloadMetadata(
            source_image_names=(f"Channel{source_index}",),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=tuple(
                        f"/input/channel_{source_index}_z{z_index}.tif"
                        for z_index in (1, 2, 3)
                    ),
                    component_metadata=tuple(
                        {
                            "channel": str(source_index),
                            "z_index": str(z_index),
                        }
                        for z_index in (1, 2, 3)
                    ),
                )
            ),
        ).payload_with(np.full((3, 4, 5), source_index, dtype=np.float32), None)
        for source_index in (1, 2, 3)
    )
    payload = ImagePayloadBundleContext.from_payloads(payloads).compose()

    result = image_math(
        payload,
        operation=ImageMathOperation.ADD,
        factors=(1.0, 1.0, 1.0),
        truncate_high=False,
        dtype_config=DtypeConfig(),
    )

    assert image_payload_data(result).shape == (3, 4, 5)
    np.testing.assert_allclose(image_payload_data(result), 6.0)


def test_image_math_uses_only_the_declared_source_binding_axis():
    payloads = tuple(
        ImagePayloadMetadata(
            source_image_names=(f"Channel{source_index}",),
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=tuple(
                        f"/input/channel_{source_index}_z{z_index}.tif"
                        for z_index in (1, 2, 3)
                    )
                )
            ),
        ).payload_with(np.full((3, 4, 5), source_index, dtype=np.float32), None)
        for source_index in (1, 2, 3)
    )
    payload = ImagePayloadBundleContext.from_payloads(payloads).compose()

    result = image_math(
        payload,
        operation=ImageMathOperation.ADD,
        factors=(1.0, 1.0, 1.0),
        truncate_high=False,
        dtype_config=DtypeConfig(),
    )

    np.testing.assert_allclose(image_payload_data(result), 6.0)


def _declared_source_image_bundle(
    *images: np.ndarray,
    source_names: tuple[str, ...] | None = None,
):
    names = source_names or tuple(f"Source{index}" for index in range(len(images)))
    if len(names) != len(images):
        raise ValueError("source_names must align exactly with images")
    return ImagePayloadBundleContext.from_payloads(
        tuple(
            ImagePayloadMetadata(
                source_image_names=(names[index],),
            ).payload_with(image, None)
            for index, image in enumerate(images)
        )
    ).compose()


def test_correct_illumination_apply_preserves_source_image_metadata() -> None:
    payload = ImagePayloadMetadata(
        intensity_scale=65535.0,
        source_dtype="uint16",
        source_image_names=("OrigImage",),
    ).payload_with(np.full((2, 2), 0.5, dtype=np.float32), None)
    illumination = ImagePayloadMetadata(
        source_image_names=("IllumImage",)
    ).payload_with(np.full((2, 2), 0.25, dtype=np.float32), None)
    result = correct_illumination_apply(
        payload,
        illumination_function=illumination,
        dtype_config=DtypeConfig(),
    )
    assert result.metadata.intensity_scale == 65535.0
    assert result.metadata.source_dtype == "uint16"
    np.testing.assert_allclose(image_payload_data(result), np.ones((2, 2)))


def test_correct_illumination_apply_projects_runtime_slice_artifact_stack() -> None:
    image = ImageMetadataPayload(
        data=np.stack(
            (
                np.full((2, 2), 0.8, dtype=np.float32),
                np.full((2, 2), 0.6, dtype=np.float32),
            )
        ),
        metadata=ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    illumination = ImageMetadataPayload(
        data=np.stack(
            (
                np.full((2, 2), 0.2, dtype=np.float32),
                np.full((2, 2), 0.3, dtype=np.float32),
            )
        ),
        metadata=ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    result = correct_illumination_apply(
        image,
        illumination_function=illumination,
        method=IlluminationCorrectionMethod.SUBTRACT,
        truncate_low=False,
        truncate_high=True,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_allclose(
        image_payload_data(result),
        np.stack(
            (
                np.full((2, 2), 0.6, dtype=np.float32),
                np.full((2, 2), 0.3, dtype=np.float32),
            )
        ),
    )


def test_measure_colocalization_uses_payload_metadata_for_costes_scale() -> None:
    image = np.stack(
        (
            np.array([[0.0, 1.0], [0.3, 0.4]], dtype=np.float32),
            np.array([[0.1, 0.8], [0.2, 0.5]], dtype=np.float32),
        )
    )
    payload = MaskedImagePayload(
        data=image,
        mask=np.ones(image.shape, dtype=bool),
        metadata=ImagePayloadMetadata(
            source_plane_intensity_scales=(65535.0, 65535.0),
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        ),
    )
    _, metadata_measurements = measure_colocalization(
        payload,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        dtype_config=DtypeConfig(),
    )
    _, explicit_measurements = measure_colocalization(
        image,
        do_correlation=False,
        do_manders=False,
        do_rwc=False,
        do_overlap=False,
        scale_max=65535,
        dtype_config=DtypeConfig(),
    )
    assert (
        metadata_measurements.columns["costes_threshold_1"][0]
        == explicit_measurements.columns["costes_threshold_1"][0]
    )
    assert (
        metadata_measurements.columns["costes_threshold_2"][0]
        == explicit_measurements.columns["costes_threshold_2"][0]
    )


def test_legacy_cellprofiler_module_aliases_resolve_to_canonical_functions():
    assert CellProfilerModule.require_module(
        "MeasureCorrelation"
    ) is CellProfilerModule.require_module("MeasureColocalization")
    assert CellProfilerModule.require_module(
        "Erosion"
    ) is CellProfilerModule.require_module("ErodeImage")
    assert CellProfilerModule.require_module(
        "Dilation"
    ) is CellProfilerModule.require_module("DilateImage")


def test_absorbed_processing_contract_metadata_does_not_act_as_validator():
    image = np.ones((8, 8), dtype=np.float32)
    result = correct_illumination_calculate(image, dtype_config=DtypeConfig())
    assert result.shape == image.shape
    assert (
        correct_illumination_calculate.__processing_contract__
        is ProcessingContract.FLEXIBLE
    )
    assert opening.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_correct_illumination_returns_retained_images_in_declared_port_order():
    image = np.arange(64, dtype=np.float32).reshape((8, 8)) / 63.0

    retained_images = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.NONE,
        rescale_option=IlluminationRescaleOption.NO,
        retain_average=True,
        retain_dilated=True,
        dtype_config=DtypeConfig(),
    )
    assert isinstance(retained_images, AlignedImageStack)
    illumination, average, dilated = retained_images.slices

    np.testing.assert_array_equal(illumination, image)
    np.testing.assert_array_equal(average, image)
    np.testing.assert_array_equal(dilated, image)


def test_illumination_functions_accept_nominal_enums():
    image = np.ones((8, 8), dtype=np.float32)
    illumination = correct_illumination_calculate(
        image,
        intensity_choice=IlluminationIntensityChoice.REGULAR,
        rescale_option=IlluminationRescaleOption.NO,
        smoothing_method=IlluminationSmoothingMethod.NONE,
        dtype_config=DtypeConfig(),
    )
    corrected = correct_illumination_apply(
        image,
        illumination_function=np.full_like(image, 0.25),
        method=IlluminationCorrectionMethod.SUBTRACT,
        truncate_low=False,
        truncate_high=False,
        dtype_config=DtypeConfig(),
    )
    assert illumination.shape == image.shape
    np.testing.assert_array_equal(corrected, np.full((8, 8), 0.75, dtype=np.float32))


def test_correct_illumination_fit_polynomial_matches_dense_design_matrix():
    from openhcs.processing.backends.cellprofiler.illumination import (
        fit_polynomial_surface,
    )

    image = (np.arange(48, dtype=np.float32).reshape(6, 8) / 47.0) ** 2
    mask = np.ones(image.shape, dtype=bool)
    mask[1::3, 2::4] = False
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w].astype(float)
    y = y / h - 0.5
    x = x / w - 0.5
    valid = mask.flatten()
    design = np.column_stack(
        [
            (x**2).flatten()[valid],
            (y**2).flatten()[valid],
            (x * y).flatten()[valid],
            x.flatten()[valid],
            y.flatten()[valid],
            np.ones(valid.sum()),
        ]
    )
    coeffs, _, _, _ = np.linalg.lstsq(design, image.flatten()[valid], rcond=None)
    full_design = np.column_stack(
        [
            (x**2).flatten(),
            (y**2).flatten(),
            (x * y).flatten(),
            x.flatten(),
            y.flatten(),
            np.ones(h * w),
        ]
    )
    expected = (full_design @ coeffs).reshape(h, w)
    np.testing.assert_allclose(
        fit_polynomial_surface(image, mask), expected, rtol=1e-10, atol=1e-10
    )


def test_correct_illumination_background_uses_blockwise_minima():
    image = (np.arange(16, dtype=np.float32).reshape(4, 4) + 1) / 100
    illumination = correct_illumination_calculate(
        image,
        intensity_choice=IlluminationIntensityChoice.BACKGROUND,
        block_size=2,
        smoothing_method=IlluminationSmoothingMethod.NONE,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    expected = np.array(
        [
            [0.01, 0.01, 0.03, 0.03],
            [0.01, 0.01, 0.03, 0.03],
            [0.09, 0.09, 0.11, 0.11],
            [0.09, 0.09, 0.11, 0.11],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_gaussian_normalizes_implicit_mask_at_borders():
    from scipy.ndimage import gaussian_filter

    image = np.zeros((9, 9), dtype=np.float32)
    image[0, 0] = 1.0
    filter_size = 3.0
    sigma = filter_size / 2.35

    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.GAUSSIAN_FILTER,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=filter_size,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )

    implicit_mask = np.ones(image.shape, dtype=bool)
    weights = gaussian_filter(
        implicit_mask.astype(float), sigma, mode="constant", cval=0
    )
    expected = gaussian_filter(image, sigma, mode="constant", cval=0) / (
        weights + np.finfo(float).eps
    )
    np.testing.assert_allclose(image_payload_data(illumination), expected)


def test_correct_illumination_automatic_filter_size_matches_cellprofiler_source():
    from openhcs.processing.backends.cellprofiler.illumination import (
        AutomaticSmoothingFilterSizeStrategy,
        CalculationScope,
        FilterSizeMethod,
        IlluminationCalculationRequest,
        IntensityChoice,
        RescaleOption,
        SmoothingMethod,
        SplineBgMode,
        correct_illumination_calculate,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    request = IlluminationCalculationRequest(
        image_data=np.zeros((1116, 1112), dtype=np.float32),
        mask=None,
        intensity_choice=IntensityChoice.REGULAR,
        dilate_objects=False,
        object_dilation_radius=1,
        block_size=60,
        rescale_option=RescaleOption.YES,
        smoothing_method=SmoothingMethod.FIT_POLYNOMIAL,
        filter_size_method=FilterSizeMethod.AUTOMATIC,
        object_width=10,
        manual_filter_size=10,
        automatic_splines=True,
        spline_bg_mode=SplineBgMode.AUTO,
        spline_points=5,
        spline_threshold=2.0,
        spline_rescale=2.0,
        spline_max_iterations=40,
        spline_convergence=0.001,
        calculation_scope=CalculationScope.EACH,
        morphology=MorphologyBackendStrategy.for_callable(
            correct_illumination_calculate
        ),
        convex_hull_backend_provider=None,
        rank_median_backend_provider=None,
    )
    assert AutomaticSmoothingFilterSizeStrategy().calculate(request) == 27.9


def test_correct_illumination_background_respects_image_mask():
    image = np.array([[0.01, 0.99], [0.03, 0.04]], dtype=np.float32)
    mask = np.array([[True, False], [True, True]], dtype=bool)
    illumination = correct_illumination_calculate(
        MaskedImagePayload(data=image, mask=mask),
        intensity_choice=IlluminationIntensityChoice.BACKGROUND,
        block_size=2,
        smoothing_method=IlluminationSmoothingMethod.NONE,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(image_payload_mask(illumination), mask)
    np.testing.assert_array_equal(
        image_payload_data(illumination),
        np.array([[0.01, 0.0], [0.01, 0.01]], dtype=np.float32),
    )


def test_correct_illumination_all_scope_averages_stack_before_smoothing():
    stack = np.stack(
        (
            np.full((4, 4), 0.25, dtype=np.float32),
            np.full((4, 4), 0.75, dtype=np.float32),
        )
    )
    illumination = correct_illumination_calculate(
        stack,
        calculation_scope=IlluminationCalculationScope.ALL_FIRST_CYCLE,
        smoothing_method=IlluminationSmoothingMethod.NONE,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    assert illumination.shape == (4, 4)
    np.testing.assert_array_equal(illumination, np.full((4, 4), 0.5, dtype=np.float32))


def test_correct_illumination_median_smoothing_uses_skimage_for_wide_rank_domain(
    monkeypatch,
):
    import skimage.filters
    from openhcs.constants.constants import MemoryType
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    calls = []

    def median(image, positional_footprint=None, **kwargs):
        calls.append(("median", image.dtype, positional_footprint, kwargs))
        return image

    monkeypatch.setattr(skimage.filters, "median", median)
    image = np.linspace(0.0, 1.0, 20 * 20, dtype=np.float32).reshape((20, 20))
    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.MEDIAN_FILTER,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=2.35,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
        rank_median_backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    assert len(calls) == 1
    assert calls[0][0:2] == ("median", np.dtype("uint16"))
    np.testing.assert_array_equal(
        calls[0][2],
        MorphologyBackendStrategy.for_memory_type(MemoryType.NUMPY).disk_footprint(1),
    )
    assert calls[0][3] == {"behavior": "rank"}
    np.testing.assert_array_equal(
        illumination,
        ((image * 65535.0).astype(np.uint16)).astype(np.float32) / 65535.0,
    )


def test_correct_illumination_median_smoothing_default_uses_native_backend():
    from openhcs.processing.backends.cellprofiler.illumination import (
        NumbaNumpyRankMedianSmoothingBackendStrategy,
        RankMedianSmoothingBackendStrategy,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    backend = RankMedianSmoothingBackendStrategy.for_memory_type()
    assert backend.backend_provider is CellProfilerBackendProvider.NATIVE
    NumbaNumpyRankMedianSmoothingBackendStrategy().prepare_backend()


def test_correct_illumination_median_smoothing_fast_minimum_majority_path():
    image = np.ones((16, 16), dtype=np.float32)
    image[1::4, 1::4] = 0.25
    illumination = correct_illumination_calculate(
        image,
        intensity_choice=IlluminationIntensityChoice.BACKGROUND,
        block_size=4,
        smoothing_method=IlluminationSmoothingMethod.MEDIAN_FILTER,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=16,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    expected = np.full((16, 16), np.uint16(0.25 * 65535) / 65535, dtype=np.float32)
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_median_smoothing_hybrid_matches_rank_reference():
    import skimage.filters
    from openhcs.processing.backends.cellprofiler.morphology import (
        MorphologyBackendStrategy,
    )

    image = np.zeros((15, 15), dtype=np.float32)
    image[5:10, 5:10] = 0.75
    footprint = MorphologyBackendStrategy.for_memory_type().disk_footprint(2)
    scaled = (image * 65535.0).astype(np.uint16)
    expected = (
        skimage.filters.median(scaled, footprint, behavior="rank").astype(np.float32)
        / 65535.0
    )
    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.MEDIAN_FILTER,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=4.7,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_median_smoothing_falls_back_when_minimum_not_majority():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(25, dtype=np.float32).reshape((5, 5)) / 24
    accelerated = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.MEDIAN_FILTER,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=2.35,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    reference = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.MEDIAN_FILTER,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=2.35,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
        rank_median_backend_provider=CellProfilerBackendProvider.NATIVE,
    )
    np.testing.assert_array_equal(accelerated, reference)


def test_correct_illumination_convex_hull_smoothing_suppresses_sparse_spikes():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.zeros((7, 7), dtype=np.float32)
    image[1, 1] = 1.0
    image[1, 5] = 1.0
    image[5, 1] = 1.0
    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        rescale_option=IlluminationRescaleOption.NO,
        convex_hull_backend_provider=CellProfilerBackendProvider.NUMBA,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(illumination, np.zeros(image.shape, dtype=np.float32))
    assert illumination.dtype == np.float32


def test_correct_illumination_centrosome_convex_hull_preserves_input_dtype():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.linspace(0.0, 1.0, 12 * 16, dtype=np.float32).reshape((12, 16))

    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        rescale_option=IlluminationRescaleOption.NO,
        convex_hull_backend_provider=CellProfilerBackendProvider.CENTROSOME,
        dtype_config=DtypeConfig(),
    )

    assert illumination.dtype == image.dtype


def test_correct_illumination_exact_convex_hull_matches_native_reference():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    rng = np.random.default_rng(123)
    image = np.zeros((30, 40), dtype=np.float32)
    rows = rng.integers(0, image.shape[0], 20)
    columns = rng.integers(0, image.shape[1], 20)
    values = rng.choice(np.array([0.25, 0.5, 0.75, 1.0], dtype=np.float32), 20)
    image[rows, columns] = values
    accelerated = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        rescale_option=IlluminationRescaleOption.NO,
        convex_hull_backend_provider=CellProfilerBackendProvider.NUMBA,
        dtype_config=DtypeConfig(),
    )
    reference = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        rescale_option=IlluminationRescaleOption.NO,
        convex_hull_backend_provider=CellProfilerBackendProvider.NATIVE,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(accelerated, reference)


def test_correct_illumination_convex_hull_default_uses_cellprofiler_reference_backend():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(49, dtype=np.float32).reshape(7, 7) / 100
    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=3,
        rescale_option=IlluminationRescaleOption.NO,
        dtype_config=DtypeConfig(),
    )
    expected = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=3,
        rescale_option=IlluminationRescaleOption.NO,
        convex_hull_backend_provider=CellProfilerBackendProvider.CENTROSOME,
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_convex_hull_legacy_fast_backend_is_explicit():
    from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    image = np.arange(49, dtype=np.float32).reshape(7, 7) / 100
    illumination = correct_illumination_calculate(
        image,
        smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
        filter_size_method=IlluminationFilterSizeMethod.MANUALLY,
        manual_filter_size=3,
        rescale_option=IlluminationRescaleOption.NO,
        convex_hull_backend_provider=CellProfilerBackendProvider.LEGACY_FAST,
        dtype_config=DtypeConfig(),
    )
    expected = grey_dilation(
        maximum_filter(grey_erosion(image, size=3), size=3), size=3
    ).astype(np.float32)
    np.testing.assert_array_equal(illumination, expected)


def test_correct_illumination_convex_hull_unregistered_backend_is_explicit_error():
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
    )

    with pytest.raises(NotImplementedError, match="No CellProfiler"):
        correct_illumination_calculate(
            np.ones((4, 4), dtype=np.float32),
            smoothing_method=IlluminationSmoothingMethod.CONVEX_HULL,
            convex_hull_backend_provider=CellProfilerBackendProvider.CUCIM,
            dtype_config=DtypeConfig(),
        )


def test_correct_illumination_strategy_registries_use_json_stable_keys():
    from openhcs.processing.backends.cellprofiler.illumination import (
        FilterSizeMethod,
        SmoothingFilterSizeStrategy,
        SmoothingMethod,
        SmoothingPlaneStrategy,
    )

    assert set(SmoothingFilterSizeStrategy.__registry__) == {
        method.value for method in FilterSizeMethod
    }
    assert set(SmoothingPlaneStrategy.__registry__) == {
        method.value for method in SmoothingMethod
    }
    assert all(
        (isinstance(key, str) for key in SmoothingFilterSizeStrategy.__registry__)
    )
    assert all((isinstance(key, str) for key in SmoothingPlaneStrategy.__registry__))
    assert (
        type(SmoothingFilterSizeStrategy.for_enum_member(FilterSizeMethod.AUTOMATIC))
        is SmoothingFilterSizeStrategy.__registry__[FilterSizeMethod.AUTOMATIC.value]
    )
    assert (
        type(SmoothingPlaneStrategy.for_enum_member(SmoothingMethod.NONE))
        is SmoothingPlaneStrategy.__registry__[SmoothingMethod.NONE.value]
    )


def test_smooth_gaussian_default_provider_preserves_cellprofiler_semantics():
    from openhcs.processing.backends.cellprofiler.smoothing import (
        SmoothingBackendSelectionRequest,
        SmoothingBackendProviderPolicy,
        SmoothingMethod,
    )
    from openhcs.processing.backends.cellprofiler._backend import (
        CellProfilerBackendProvider,
        DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    )

    assert (
        SmoothingBackendProviderPolicy.resolve(
            SmoothingMethod.GAUSSIAN_FILTER, DEFAULT_CELLPROFILER_BACKEND_SELECTION
        )
        is CellProfilerBackendProvider.NATIVE
    )
    assert (
        SmoothingBackendProviderPolicy.resolve(
            SmoothingMethod.GAUSSIAN_FILTER,
            DEFAULT_CELLPROFILER_BACKEND_SELECTION,
            SmoothingBackendSelectionRequest(
                method=SmoothingMethod.GAUSSIAN_FILTER,
                auto_object_size=False,
                object_size=3.0,
                image_shape=(64, 64),
            ),
        )
        is CellProfilerBackendProvider.NATIVE
    )
    assert (
        SmoothingBackendProviderPolicy.resolve(
            SmoothingMethod.GAUSSIAN_FILTER,
            DEFAULT_CELLPROFILER_BACKEND_SELECTION,
            SmoothingBackendSelectionRequest(
                method=SmoothingMethod.GAUSSIAN_FILTER,
                auto_object_size=False,
                object_size=20.0,
                image_shape=(64, 64),
            ),
        )
        is CellProfilerBackendProvider.OPENCV
    )


def test_pure_2d_contract_wrapper_aggregates_illumination_outputs_per_slice():
    registry = OpenHCSRegistry()
    wrapped = registry.apply_contract_wrapper(
        correct_illumination_calculate, ProcessingContract.PURE_2D
    )
    image = np.stack(
        (np.full((8, 8), 1.0, dtype=np.float32), np.full((8, 8), 2.0, dtype=np.float32))
    )
    result = wrapped(
        ImageMetadataPayload(
            data=image,
            metadata=ImagePayloadMetadata(
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
        ),
        dtype_config=DtypeConfig(),
    )
    assert result.shape == image.shape


def test_unified_registry_strips_semantic_controls_from_non_flexible_contracts():
    registry = OpenHCSRegistry()

    def pure_2d(image, *, slice_by_slice: bool = False, dtype_config=None):
        if slice_by_slice:
            raise AssertionError("PURE_2D received flexible semantic control")
        return image

    pure_2d.output_memory_type = "numpy"
    wrapped = registry.apply_contract_wrapper(pure_2d, ProcessingContract.PURE_2D)

    assert "slice_by_slice" not in wrapped.__signature__.parameters
    result = wrapped(np.ones((2, 3, 3), dtype=np.float32), slice_by_slice=True)
    assert result.shape == (2, 3, 3)


def test_unified_registry_injects_semantic_controls_for_flexible_contracts():
    registry = OpenHCSRegistry()

    def flexible(image):
        return image

    wrapped = registry.apply_contract_wrapper(flexible, ProcessingContract.FLEXIBLE)

    assert "slice_by_slice" in wrapped.__signature__.parameters


def test_unmix_colors_returns_one_output_per_stain_row():
    image = np.full((8, 9, 3), 0.5, dtype=np.float32)
    outputs = unmix_colors(
        ImagePayloadMetadata(source_channel_axis=-1).payload_with(image, None),
        stain_names=(StainType.HEMATOXYLIN, StainType.EOSIN, StainType.CUSTOM),
        custom_absorbances=((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.1, 0.2, 0.3)),
        dtype_config=DtypeConfig(),
    )
    assert isinstance(outputs, AlignedImageStack)
    assert [output.shape for output in outputs.slices] == [(8, 9), (8, 9), (8, 9)]
    assert all((output.dtype == np.float32 for output in outputs.slices))
    assert all(
        image_payload_metadata(output).source_channel_axis is None
        for output in outputs.slices
    )
    assert unmix_colors.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_flip_and_rotate_preserves_declared_color_channel_axis() -> None:
    image = ImagePayloadMetadata(source_channel_axis=-1).payload_with(
        np.arange(8 * 9 * 3, dtype=np.float32).reshape((8, 9, 3)),
        None,
    )

    output, _rotation = flip_and_rotate(
        image,
        flip_method=FlipMethod.LEFT_TO_RIGHT,
        dtype_config=DtypeConfig(),
    )

    assert output.shape == (8, 9, 3)
    assert image_payload_metadata(output).source_channel_axis == -1


def test_binary_mask_output_preserves_declared_color_channel_axis() -> None:
    image = ImagePayloadMetadata(source_channel_axis=-1).payload_with(
        np.linspace(0.0, 1.0, 8 * 9 * 3, dtype=np.float32).reshape((8, 9, 3)),
        None,
    )

    output = mask_image_with_binary(image, dtype_config=DtypeConfig())

    assert output.shape == (8, 9, 3)
    assert image_payload_metadata(output).source_channel_axis == -1


def test_crop_preserves_hwc_color_image_domain() -> None:
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    image_payload = ImagePayloadMetadata(source_channel_axis=-1).payload_with(
        image, None
    )
    cropped, mask, measurements = crop(
        image_payload,
        removal_method=CropModule.RemovalMethod.ALL,
        left_right_rectangle_positions=(2, 7),
        top_bottom_rectangle_positions=(1, 6),
        dtype_config=DtypeConfig(),
    )
    assert isinstance(cropped, MaskedImagePayload)
    assert cropped.shape == (5, 5, 3)
    assert mask.shape == (8, 9)
    assert measurements.column_values("area_retained") == (25,)
    np.testing.assert_array_equal(cropped.data, image[1:6, 2:7])
    np.testing.assert_array_equal(cropped.mask, np.ones((5, 5), dtype=bool))
    assert cropped.metadata.spatial_origin_yx == (1, 2)
    assert cropped.metadata.source_spatial_shape_yx == (8, 9)
    assert cropped.metadata.physical_border_edges_for_shape(cropped.shape[:2]) == (
        False,
        False,
        False,
        False,
    )


def test_crop_no_removal_returns_masked_zeroed_image_domain() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    cropped, mask, measurements = crop(
        image,
        removal_method=CropModule.RemovalMethod.NO,
        left_right_rectangle_positions=(1, 4),
        top_bottom_rectangle_positions=(1, 3),
        dtype_config=DtypeConfig(),
    )
    assert isinstance(cropped, MaskedImagePayload)
    assert cropped.shape == image.shape
    assert measurements.column_values("area_retained") == (6,)
    assert cropped.metadata.mask_defines_border is False
    np.testing.assert_array_equal(
        mask,
        np.array(
            [
                [False, False, False, False, False],
                [False, True, True, True, False],
                [False, True, True, True, False],
                [False, False, False, False, False],
            ],
            dtype=bool,
        ),
    )
    np.testing.assert_array_equal(mask, cropped.mask)
    assert np.all(cropped.data[~mask] == 0)
    assert np.all(cropped.data[mask] == 1)
    assert cropped.metadata.spatial_origin_yx == (0, 0)
    assert cropped.metadata.source_spatial_shape_yx == image.shape
    assert cropped.metadata.physical_border_edges_for_shape(cropped.shape) == (
        True,
        True,
        True,
        True,
    )


def test_crop_previous_cropping_accepts_typed_mask_input() -> None:
    image = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    previous_mask = np.array(
        [
            [False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False],
        ],
        dtype=bool,
    )
    cropped, crop_mask, measurements = crop(
        image,
        topology_inputs=(previous_mask,),
        crop_shape=CropModule.Shape.CROPPING,
        removal_method=CropModule.RemovalMethod.EDGES,
        dtype_config=DtypeConfig(),
    )
    assert isinstance(cropped, MaskedImagePayload)
    np.testing.assert_array_equal(crop_mask, previous_mask)
    np.testing.assert_array_equal(cropped.data, image[1:3, 1:4])
    np.testing.assert_array_equal(cropped.mask, np.ones((2, 3), dtype=bool))
    assert measurements.column_values("area_retained") == (6,)


def test_crop_objects_rejects_unprojected_label_stack() -> None:
    image = np.ones((4, 5), dtype=np.float32)
    labels = np.zeros((2, 4, 5), dtype=np.int32)
    labels[0, 1, 1] = 1
    labels[1, 2, 3] = 2
    with pytest.raises(ValueError, match="projected to one 2-D plane"):
        crop(
            image,
            topology_inputs=(
                ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
            ),
            crop_shape=CropModule.Shape.OBJECTS,
            removal_method=CropModule.RemovalMethod.NO,
            dtype_config=DtypeConfig(),
        )


def test_measure_image_area_occupied_runs_mixed_rows():
    binary = np.zeros((5, 6), dtype=np.float32)
    binary[1:3, 1:4] = 1.0
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[2:4, 2:5] = 1
    output, measurements = measure_image_area_occupied(
        binary,
        operand_choices=(OperandChoice.BINARY_IMAGE, OperandChoice.OBJECTS),
        area_occupied_rows=(
            AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "DNA"),
            AreaOccupiedRow(OperandChoice.OBJECTS, "Nuclei"),
        ),
        object_labels=(
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        ),
        dtype_config=DtypeConfig(),
    )
    np.testing.assert_array_equal(output, binary)
    measurement_rows = measurements.row_mappings()
    assert [row["slice_index"] for row in measurement_rows] == [0, 0]
    assert all(row["area_occupied"] == 6.0 for row in measurement_rows)
    assert [row["source_image_name"] for row in measurement_rows] == [
        "DNA",
        "Nuclei",
    ]
    assert (
        measure_image_area_occupied.__processing_contract__
        is ProcessingContract.FLEXIBLE
    )


def test_measure_image_area_occupied_uses_runtime_slice_for_every_operand_row():
    image = _declared_source_image_bundle(
        np.ones((3, 4), dtype=np.float32),
        np.ones((3, 4), dtype=np.float32),
        source_names=("First", "Second"),
    )
    _retained, measurements = measure_image_area_occupied.__wrapped__(
        image,
        operand_choices=(
            OperandChoice.BINARY_IMAGE,
            OperandChoice.BINARY_IMAGE,
        ),
        area_occupied_rows=(
            AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "First"),
            AreaOccupiedRow(OperandChoice.BINARY_IMAGE, "Second"),
        ),
        slice_index=7,
    )

    assert [row["slice_index"] for row in measurements.iter_row_mappings()] == [7, 7]


def test_measure_image_area_occupied_rejects_unprojected_label_stacks():
    image = np.zeros((2, 5, 6), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0, 1:3, 1:4] = 1
    labels[1, 2:4, 2:5] = 1
    with pytest.raises(ValueError, match="already projected to one 2-D plane"):
        measure_image_area_occupied(
            image,
            operand_choices=(OperandChoice.OBJECTS,),
            area_occupied_rows=(
                AreaOccupiedRow(
                    OperandChoice.OBJECTS,
                    "Nuclei",
                ),
            ),
            object_labels=(
                ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
            ),
            dtype_config=DtypeConfig(),
        )


def test_mask_image_applies_2d_object_mask_to_projected_image_plane():
    image = np.ones((5, 6), dtype=np.float32)
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:4, 2:5] = 1
    masked = mask_image(
        image,
        labels,
        mask_source=MaskSource.OBJECTS,
        dtype_config=DtypeConfig(),
    )
    assert masked.shape == image.shape
    assert isinstance(masked, MaskedImagePayload)
    assert np.count_nonzero(image_payload_data(masked)) == 9
    assert np.all(image_payload_data(masked)[labels == 0] == 0)
    assert np.array_equal(image_payload_mask(masked), labels > 0)


def test_mask_image_accepts_object_label_payload_mask():
    image = np.ones((5, 6), dtype=np.float32)
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:4, 2:5] = 1
    masked = mask_image(
        image,
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1,)),
        ),
        mask_source=MaskSource.OBJECTS,
        dtype_config=DtypeConfig(),
    )
    assert isinstance(masked, MaskedImagePayload)
    assert np.count_nonzero(image_payload_data(masked)) == 9
    np.testing.assert_array_equal(image_payload_mask(masked), labels > 0)


def test_mask_image_accepts_source_backed_projected_image_plane():
    image = ImagePayloadMetadata(source_path="source.tif").payload_with(
        np.ones((5, 6), dtype=np.float32), None
    )
    labels = np.zeros((5, 6), dtype=np.int32)
    labels[1:4, 2:5] = 1
    masked = mask_image(
        image,
        labels,
        mask_source=MaskSource.OBJECTS,
        dtype_config=DtypeConfig(),
    )
    assert masked.shape == (5, 6)
    assert np.count_nonzero(image_payload_data(masked)) == 9
    assert np.array_equal(image_payload_mask(masked), labels > 0)


def test_mask_image_uses_aligned_mask_stack_planes():
    image = _declared_source_image_bundle(
        np.ones((5, 6), dtype=np.float32),
        np.ones((5, 6), dtype=np.float32),
        source_names=("First", "Second"),
    )
    first_mask = np.zeros((5, 6), dtype=np.float32)
    first_mask[1:3, 1:3] = 1.0
    second_mask = np.zeros((5, 6), dtype=np.float32)
    second_mask[2:5, 3:6] = 1.0
    mask = _declared_source_image_bundle(
        first_mask,
        second_mask,
        source_names=("First", "Second"),
    )
    masked_planes = tuple(
        mask_image(
            RuntimeSliceProjection.value_for_slice(
                image,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.SOURCE_BINDING,
                    plane_index=index,
                    axis_size=2,
                ),
            ),
            RuntimeSliceProjection.value_for_slice(
                mask,
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.SOURCE_BINDING,
                    plane_index=index,
                    axis_size=2,
                ),
            ),
            mask_source=MaskSource.IMAGE,
            dtype_config=DtypeConfig(),
        )
        for index in range(2)
    )
    assert np.count_nonzero(image_payload_data(masked_planes[0])) == 4
    assert np.count_nonzero(image_payload_data(masked_planes[1])) == 9


def test_mask_image_rejects_unprojected_object_label_stack():
    image = np.ones((5, 6), dtype=np.float32)
    mask = np.zeros((2, 5, 6), dtype=np.int32)
    mask[0, 1:3, 1:3] = 1
    mask[1, 2:5, 3:6] = 2
    label_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=mask),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    with pytest.raises(ValueError, match="cardinalities must exactly match"):
        mask_image(
            image,
            label_stack,
            mask_source=MaskSource.OBJECTS,
            dtype_config=DtypeConfig(),
        )


def test_relate_objects_rejects_mixed_object_label_domain_scopes():
    parent_plane = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
    child_plane = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
    parent_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.stack((parent_plane, parent_plane))
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1, 2), (1, 2)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    child_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=child_plane)
    )

    with pytest.raises(
        ValueError,
        match="Cannot merge object-label values with different domain scopes",
    ):
        relate_objects.__wrapped__(
            np.zeros_like(child_plane, dtype=np.float32),
            parent_payload,
            child_payload,
            calculate_distances=DistanceMethod.NONE,
        )


def test_relate_objects_preserves_child_object_label_domain_metadata():
    parent_plane = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
    child_plane = np.array([[1, 1, 0], [0, 2, 2]], dtype=np.int32)
    child_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=child_plane),
        domain=ObjectLabelDomain(declared_object_count=4),
    )
    output, _parent_relationship, _child_relationship, _measurements = (
        relate_objects.__wrapped__(
            np.zeros_like(child_plane, dtype=np.float32),
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=parent_plane)
            ),
            child_payload,
            calculate_distances=DistanceMethod.NONE,
        )
    )
    assert isinstance(output, ObjectLabelPayload)
    assert output.domain.declared_object_count == 4
    assert output.domain.scope is ObjectLabelDomainScope.PAYLOAD
    np.testing.assert_array_equal(output.labels, child_plane)


def test_relate_objects_uses_explicitly_projected_plane_identity():
    parent_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (((1, 1, 0), (0, 0, 0)), ((2, 2, 0), (0, 0, 0))), dtype=np.int32
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    child_stack = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (((1, 1, 0), (0, 0, 0)), ((1, 1, 0), (0, 0, 0))), dtype=np.int32
            )
        ),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    relationships = []
    for slice_index in range(2):
        axis = RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE, plane_index=slice_index, axis_size=2
        )
        _output, relationship, _reverse_relationship, _measurements = (
            relate_objects.__wrapped__(
                np.zeros((2, 3), dtype=np.float32),
                RuntimeSliceProjection.value_for_slice(parent_stack, axis),
                RuntimeSliceProjection.value_for_slice(child_stack, axis),
                calculate_distances=DistanceMethod.NONE,
                slice_index=slice_index,
            )
        )
        relationships.append(relationship)

    assert [relationship.source_ids for relationship in relationships] == [(1,), (2,)]
    assert [relationship.target_ids for relationship in relationships] == [(1,), (1,)]
    assert [relationship.slice_indices for relationship in relationships] == [
        (0,),
        (1,),
    ]


def test_relate_objects_raw_callable_leaves_measurements_to_output_recording():
    parent_labels = np.zeros((5, 5), dtype=np.int32)
    parent_labels[1:4, 1:4] = 1
    child_labels = np.zeros_like(parent_labels)
    child_labels[2, 2] = 1
    _output, relationships, reverse_relationship, measurements = (
        relate_objects.__wrapped__(
            np.zeros_like(parent_labels, dtype=np.float32),
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=parent_labels)
            ),
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(labels=child_labels)
            ),
            calculate_distances=DistanceMethod.BOTH,
        )
    )
    assert relationships.source_ids == (1,)
    assert relationships.target_ids == (1,)
    assert reverse_relationship.source_ids == (1,)
    assert reverse_relationship.target_ids == (1,)
    assert measurements.row_count() == 0


def test_mask_image_combines_existing_image_mask_with_mask_input():
    image = np.ones((5, 6), dtype=np.float32)
    existing_mask = np.zeros_like(image, dtype=bool)
    existing_mask[1:5, 1:5] = True
    mask = np.zeros_like(image, dtype=np.float32)
    mask[0:3, 2:6] = 1.0
    masked = mask_image(
        MaskedImagePayload(data=image, mask=existing_mask),
        mask,
        mask_source=MaskSource.IMAGE,
        dtype_config=DtypeConfig(),
    )
    expected_mask = existing_mask & (mask > 0)
    assert isinstance(masked, MaskedImagePayload)
    assert np.array_equal(image_payload_mask(masked), expected_mask)
    assert np.count_nonzero(image_payload_data(masked)) == int(expected_mask.sum())


def test_align_returns_two_registered_images_and_shift_measurements():
    first = np.zeros((8, 8), dtype=np.float32)
    first[2:5, 2:5] = 1.0
    second = np.zeros_like(first)
    second[3:6, 2:5] = 1.0
    image = np.stack((first, second))
    aligned_images, measurements = align(
        ImagePayloadMetadata(
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("/input/first.tif", "/input/second.tif")
                )
            ),
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        ).payload_with(image, None),
        crop_mode=AlignModule.CropMode.KEEP_SIZE,
        dtype_config=DtypeConfig(),
    )
    assert isinstance(aligned_images, AlignedImageStack)
    aligned_first, aligned_second = aligned_images.slices
    assert image_payload_data(aligned_first).shape == first.shape
    assert image_payload_data(aligned_second).shape == second.shape
    assert image_payload_mask(aligned_first) is None
    assert image_payload_mask(aligned_second) is not None
    assert measurements.rows[0] == AlignShiftMeasurement(
        slice_index=0, output_index=0, x_shift=0, y_shift=0
    )
    assert measurements.rows[1].output_index == 1
    assert measurements.rows[1].x_shift == 0
    assert measurements.rows[1].y_shift > 0
    assert type(measurements.rows[1].x_shift) is int
    assert type(measurements.rows[1].y_shift) is int
    assert align.__processing_contract__ is ProcessingContract.PURE_3D


def test_align_declares_native_integer_shift_fields():
    output_annotations = get_type_hints(AlignShiftMeasurement, include_extras=True)
    row_annotations = get_type_hints(
        AlignModule.MeasurementRecord,
        include_extras=True,
    )

    assert FieldSpec.annotation_dtype(output_annotations["x_shift"]) is int
    assert FieldSpec.annotation_dtype(output_annotations["y_shift"]) is int
    assert row_annotations["x_shift"] is int
    assert row_annotations["y_shift"] is int


def test_align_applies_similar_shift_to_additional_images():
    first = np.zeros((8, 8), dtype=np.float32)
    first[2:5, 2:5] = 1.0
    second = np.zeros_like(first)
    second[3:6, 2:5] = 1.0
    additional = np.zeros_like(first)
    additional[4:7, 4:7] = 2.0
    image = np.stack((first, second, additional))
    aligned_images, measurements = align(
        ImagePayloadMetadata(
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=(
                        "/input/first.tif",
                        "/input/second.tif",
                        "/input/additional.tif",
                    )
                )
            ),
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        ).payload_with(image, None),
        crop_mode=AlignModule.CropMode.KEEP_SIZE,
        additional_alignment_modes=(AlignModule.AdditionalMode.SIMILARLY,),
        dtype_config=DtypeConfig(),
    )
    assert isinstance(aligned_images, AlignedImageStack)
    aligned_first, aligned_second, aligned_additional = aligned_images.slices
    assert image_payload_data(aligned_first).shape == first.shape
    assert image_payload_data(aligned_second).shape == second.shape
    assert image_payload_data(aligned_additional).shape == additional.shape
    assert len(measurements) == 3
    assert measurements.rows[2].output_index == 2
    assert measurements.rows[2].x_shift == measurements.rows[1].x_shift
    assert measurements.rows[2].y_shift == measurements.rows[1].y_shift


def test_overlay_outlines_runs_mixed_image_and_object_rows():
    base = np.zeros((8, 8), dtype=np.float32)
    outline_image = np.zeros_like(base)
    outline_image[1:6, 1] = 1.0
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1
    output = overlay_outlines(
        np.stack((base, outline_image)),
        outline_source_kinds=(OutlineSourceKind.IMAGE, OutlineSourceKind.OBJECTS),
        outline_colors=("Red", "Green"),
        object_labels=(
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert output.shape == (8, 8, 3)
    assert output[..., 0].max() > 0
    assert output[..., 1].max() > 0
    assert image_payload_metadata(output).source_channel_axis == -1
    assert overlay_outlines.__processing_contract__ is ProcessingContract.FLEXIBLE


def test_overlay_outlines_accepts_hex_color_literals():
    base = np.zeros((8, 8), dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1
    output = overlay_outlines(
        base,
        outline_source_kinds=(OutlineSourceKind.OBJECTS,),
        outline_colors=("#0800F7",),
        object_labels=(
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert output.shape == (8, 8, 3)
    assert output[..., 2].max() > 0.9
    assert output[..., 0].max() < 0.1


def test_overlay_outlines_accepts_css_named_color_literals():
    base = np.zeros((8, 8), dtype=np.float32)
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1
    output = overlay_outlines(
        base,
        outline_source_kinds=(OutlineSourceKind.OBJECTS,),
        outline_colors=("DarkOrange",),
        object_labels=(
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        ),
        dtype_config=DtypeConfig(),
    )
    assert output.shape == (8, 8, 3)
    assert output[..., 0].max() == pytest.approx(1.0)
    assert output[..., 1].max() == pytest.approx(140.0 / 255.0)
    assert output[..., 2].max() == pytest.approx(0.0)


def test_overlay_outlines_uses_cellprofiler_mark_boundaries_semantics():
    base = np.zeros((8, 8), dtype=np.float32)
    base[1:7, 1:7] = 0.25
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[3:6, 3:6] = 1
    output = overlay_outlines(
        base,
        line_mode=LineMode.INNER,
        outline_source_kinds=(OutlineSourceKind.OBJECTS,),
        outline_colors=("Green",),
        object_labels=(
            ObjectLabelPayload(variant_data=ObjectLabelVariantData(labels=labels)),
        ),
        dtype_config=DtypeConfig(),
    )
    expected = skimage.segmentation.mark_boundaries(
        np.dstack((base, base, base)),
        labels,
        color=(0.0, 128.0 / 255.0, 0.0),
        mode="inner",
    ).astype(np.float32)
    np.testing.assert_array_equal(output, expected)


def test_overlay_outlines_rejects_unprojected_object_label_stack():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    labels[0, 2:5, 2:5] = 1
    labels[1, 3:6, 3:6] = 1
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (1,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )

    with pytest.raises(ValueError, match="runtime-projected 2-D"):
        overlay_outlines(
            image,
            outline_source_kinds=(OutlineSourceKind.OBJECTS,),
            outline_colors=("Green",),
            object_labels=(payload,),
            dtype_config=DtypeConfig(),
        )


def test_overlay_outlines_renders_exact_projected_empty_label_plane():
    image = np.zeros((2, 8, 8), dtype=np.float32)
    labels = np.zeros_like(image, dtype=np.int32)
    payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=labels),
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    projected = RuntimeSliceProjection.value_for_slice(
        payload,
        RuntimePlaneAxisValueProjection.from_selected_plane(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            plane_index=1,
            axis_size=2,
        ),
    )
    output = overlay_outlines(
        image[1],
        outline_source_kinds=(OutlineSourceKind.OBJECTS,),
        object_labels=(projected,),
        dtype_config=DtypeConfig(),
    )
    assert output.shape == (8, 8, 3)
    assert float(image_payload_data(output).max()) == 0.0
