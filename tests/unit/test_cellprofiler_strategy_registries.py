"""Regression tests for JSON-safe CellProfiler compatibility registries."""

import numpy as np

from openhcs.processing.backends.cellprofiler.alignment import AlignCropModeStrategy
from openhcs.processing.backends.cellprofiler.measurement_math import (
    MathOperationStrategy,
    RoundingStrategy,
)
from benchmark.cellprofiler_library.functions.convertobjectstoimage import (
    ImageModeRenderer,
)
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    IlluminationCorrectionStrategy,
)
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    SmoothingFilterSizeStrategy,
    SmoothingPlaneStrategy,
)
from benchmark.cellprofiler_library.functions.crop import CropShapeMaskStrategy
from benchmark.cellprofiler_library.functions.enhanceedges import (
    EdgeEnhancementStrategy,
)
from benchmark.cellprofiler_library.functions.filterobjects import (
    FilterSelectionStrategy,
    PerObjectAssignmentStrategy,
)
from benchmark.cellprofiler_library.functions.graytocolor import GrayToColorSchemeRunner
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperationStrategy,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.processing.backends.cellprofiler.grid import (
    DefineGridManualModule,
    IdentifyObjectsInGridModule,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    SecondarySegmentationStrategy,
    ThresholdCalculator,
)
from benchmark.cellprofiler_library.functions.measureobjectneighbors import (
    NeighborDistancePlanner,
)
from benchmark.cellprofiler_library.functions.smooth import SmoothingStrategy
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElementFactory,
)
from openhcs.processing.backends.cellprofiler.worms import (
    WormLabelOutputStrategy,
)
from openhcs.processing.backends.cellprofiler.watershed import (
    WatershedMethodStrategy,
    WatershedSeedStrategy,
    WatershedRuntimeStrategy,
)
from openhcs.interop.cellprofiler.runtime.module_execution import (
    CellProfilerObjectMeasurementLabelArgumentPolicy,
    CellProfilerObjectMeasurementLabelArgumentRequest,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    MeasurementLabelExecutionModeStrategy,
)
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelMeasurementExecution,
    object_label_measurement_execution,
)
from openhcs.core.runtime_semantics import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
    RuntimePlaneAxis,
)
from openhcs.core.runtime_values import ObjectLabelPayload


JSON_SAFE_REGISTRY_KEY_TYPES = (str, int, float, bool, type(None))


def test_cellprofiler_strategy_registry_keys_are_json_safe():
    registry_classes = (
        AlignCropModeStrategy,
        MathOperationStrategy,
        RoundingStrategy,
        ImageModeRenderer,
        ImageMathOperationStrategy,
        IlluminationCorrectionStrategy,
        SmoothingFilterSizeStrategy,
        SmoothingPlaneStrategy,
        CropShapeMaskStrategy,
        EdgeEnhancementStrategy,
        FilterSelectionStrategy,
        PerObjectAssignmentStrategy,
        GrayToColorSchemeRunner,
        SecondarySegmentationStrategy,
        ThresholdCalculator,
        NeighborDistancePlanner,
        SmoothingStrategy,
        StructuringElementFactory,
        WormLabelOutputStrategy,
        WatershedMethodStrategy,
        WatershedSeedStrategy,
        WatershedRuntimeStrategy,
        MeasurementLabelExecutionModeStrategy,
        CellProfilerObjectMeasurementLabelArgumentPolicy,
    )

    for registry_class in registry_classes:
        assert registry_class.__registry__
        assert all(
            isinstance(key, JSON_SAFE_REGISTRY_KEY_TYPES)
            for key in registry_class.__registry__
        ), registry_class.__name__


def test_grid_function_name_variants_resolve_from_module_declarations() -> None:
    define_grid = ModuleBlock(
        name="DefineGrid",
        module_num=1,
        settings={"Select the method to define the grid": "Automatic"},
    )
    identify_without_guides = ModuleBlock(
        name="IdentifyObjectsInGrid",
        module_num=2,
        settings={"Select the guiding objects": "None"},
    )
    identify_with_guides = ModuleBlock(
        name="IdentifyObjectsInGrid",
        module_num=3,
        settings={"Select the guiding objects": "Nuclei"},
    )

    assert DefineGridManualModule.resolve_function(define_grid).function_name == (
        "define_grid_automatic"
    )
    assert IdentifyObjectsInGridModule.resolve_function(
        identify_without_guides
    ).function_name == "identify_objects_in_grid"
    assert IdentifyObjectsInGridModule.resolve_function(
        identify_with_guides
    ).function_name == "identify_objects_in_grid_with_guides"


def test_measurement_label_execution_mode_follows_object_label_domain():
    def slice_aligned_measurement(image, labels):
        return image, labels

    @object_label_measurement_execution(ObjectLabelMeasurementExecution.FULL_STACK)
    def full_stack_measurement(image, labels):
        return image, labels

    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            slice_aligned_measurement,
            np.zeros((3, 8, 8), dtype=np.int32),
            ImagePayloadExecutionMode.NATURAL,
        )
        is ImagePayloadExecutionMode.NATURAL
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            np.zeros((3, 8, 8), dtype=np.int32),
            ImagePayloadExecutionMode.NATURAL,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            np.zeros((3, 8, 8), dtype=np.int32),
            ImagePayloadExecutionMode.FULL_STACK,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            np.zeros((8, 8), dtype=np.int32),
            ImagePayloadExecutionMode.NATURAL,
        )
        is ImagePayloadExecutionMode.NATURAL
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            ObjectLabelPayload(
                labels=np.zeros((3, 8, 8), dtype=np.int32),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
            ),
            ImagePayloadExecutionMode.FULL_STACK,
            runtime_slice_count=1,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            ObjectLabelPayload(
                labels=np.zeros((3, 8, 8), dtype=np.int32),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            ImagePayloadExecutionMode.NATURAL,
            runtime_slice_count=3,
        )
        is ImagePayloadExecutionMode.NATURAL
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            ObjectLabelPayload(
                labels=np.zeros((3, 8, 8), dtype=np.int32),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            ),
            ImagePayloadExecutionMode.FULL_STACK,
            runtime_slice_count=3,
        )
        is ImagePayloadExecutionMode.NATURAL
    )
    assert (
        MeasurementLabelExecutionModeStrategy.resolve(
            full_stack_measurement,
            ObjectLabelPayload(
                labels=np.zeros((3, 8, 8), dtype=np.int32),
                plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
                domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE),
            ),
            ImagePayloadExecutionMode.FULL_STACK,
        )
        is ImagePayloadExecutionMode.NATURAL
    )


def test_measurement_label_argument_policy_follows_execution_contract() -> None:
    dense_labels = np.zeros((8, 8), dtype=np.int32)
    payload = object()
    request = CellProfilerObjectMeasurementLabelArgumentRequest(
        dense_labels=dense_labels,
        label_payload=payload,
        measurement_image_payload=np.zeros((8, 8), dtype=np.float32),
    )

    assert (
        CellProfilerObjectMeasurementLabelArgumentPolicy.for_enum_member(
            ObjectLabelMeasurementExecution.SLICE_ALIGNED
        ).label_argument(request)
        is dense_labels
    )
    assert (
        CellProfilerObjectMeasurementLabelArgumentPolicy.for_enum_member(
            ObjectLabelMeasurementExecution.FULL_STACK
        ).label_argument(request)
        is payload
    )


def test_measurement_label_argument_policy_defers_aligned_stack_label_projection() -> (
    None
):
    dense_labels = np.zeros((8, 8), dtype=np.int32)
    label_payload = ObjectLabelPayload(
        labels=np.zeros((2, 8, 8), dtype=np.int32),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PLANE,
        ))
    request = CellProfilerObjectMeasurementLabelArgumentRequest(
        dense_labels=dense_labels,
        label_payload=label_payload,
        measurement_image_payload=AlignedImageStack(
            (
                np.zeros((8, 8), dtype=np.float32),
                np.ones((8, 8), dtype=np.float32),
            )
        ),
    )

    assert (
        CellProfilerObjectMeasurementLabelArgumentPolicy.for_enum_member(
            ObjectLabelMeasurementExecution.SLICE_ALIGNED
        ).label_argument(request)
        is label_payload
    )
