"""Regression tests for JSON-safe CellProfiler compatibility registries."""

import numpy as np

from openhcs.processing.backends.cellprofiler.alignment import AlignCropModeStrategy
from openhcs.processing.backends.cellprofiler.measurement_math import (
    MathOperationStrategy,
    RoundingStrategy,
)
from openhcs.processing.backends.cellprofiler.object_images import ImageModeRenderer
from openhcs.processing.backends.cellprofiler.illumination import (
    IlluminationCorrectionStrategy,
)
from openhcs.processing.backends.cellprofiler.illumination import (
    SmoothingFilterSizeStrategy,
    SmoothingPlaneStrategy,
)
from openhcs.processing.backends.cellprofiler.crop import CropShapeMaskStrategy
from openhcs.processing.backends.cellprofiler.edge import EdgeEnhancementStrategy
from openhcs.processing.backends.cellprofiler.object_filtering import (
    FilterSelectionStrategy,
    PerObjectAssignmentStrategy,
)
from openhcs.processing.backends.cellprofiler.color import GrayToColorSchemeRunner
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperationStrategy,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.core.callable_contract import CallableContract
from openhcs.core.source_bindings import EMPTY_SOURCE_BINDINGS
from openhcs.processing.backends.cellprofiler.grid import (
    DefineGridManualModule,
    IdentifyObjectsInGridModule,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    SecondarySegmentationStrategy,
    ThresholdCalculator,
)
from openhcs.processing.backends.cellprofiler.neighbors import NeighborDistancePlanner
from openhcs.processing.backends.cellprofiler.smoothing import SmoothingStrategy
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
from openhcs.interop.cellprofiler.runtime.object_measurement_execution import (
    CellProfilerObjectMeasurementExecutionPolicy,
)
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
)
from openhcs.core.runtime_object_label_domains import ObjectLabelDomain, ObjectLabelDomainScope
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)


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
        CellProfilerObjectMeasurementExecutionPolicy,
    )

    for registry_class in registry_classes:
        assert registry_class.__registry__
        assert all(
            isinstance(key, JSON_SAFE_REGISTRY_KEY_TYPES)
            for key in registry_class.__registry__
        ), registry_class.__name__


def test_grid_callable_selection_resolves_from_module_declarations() -> None:
    define_grid = ModuleBlock(
        name="DefineGrid",
        module_num=1,
        setting_records=[
            ModuleSetting("Select the method to define the grid", "Automatic")
        ],
    )
    identify_without_guides = ModuleBlock(
        name="IdentifyObjectsInGrid",
        module_num=2,
        setting_records=[
            ModuleSetting(
                "Select object shapes and locations", "Rectangle Forced Location"
            ),
            ModuleSetting("Select the guiding objects", "None"),
        ],
    )
    identify_with_guides = ModuleBlock(
        name="IdentifyObjectsInGrid",
        module_num=3,
        setting_records=[
            ModuleSetting(
                "Select object shapes and locations", "Natural Shape and Location"
            ),
            ModuleSetting("Select the guiding objects", "Nuclei"),
        ],
    )

    contract = CallableContract.from_callable(
        IdentifyObjectsInGridModule.require_callable()
    )
    assert DefineGridManualModule.resolve_function(
        define_grid,
        contract=contract,
        source_bindings=EMPTY_SOURCE_BINDINGS,
    ) is DefineGridManualModule.require_callable(
        DefineGridManualModule.function_variants[0]
    )
    assert (
        IdentifyObjectsInGridModule.resolve_function(
            identify_without_guides,
            contract=contract,
            source_bindings=EMPTY_SOURCE_BINDINGS,
        )
        is IdentifyObjectsInGridModule.require_callable()
    )
    assert (
        IdentifyObjectsInGridModule.resolve_function(
            identify_with_guides,
            contract=contract,
            source_bindings=EMPTY_SOURCE_BINDINGS,
        )
        is IdentifyObjectsInGridModule.require_callable()
    )


def test_measurement_execution_mode_follows_callable_declaration():
    slice_aligned_policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )
    full_stack_policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.FULL_STACK
    )

    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 8, 8), dtype=np.int32)),
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((), (), ()),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )

    assert (
        slice_aligned_policy.image_execution_mode(
            labels,
            ImagePayloadExecutionMode.NATURAL,
        )
        is ImagePayloadExecutionMode.NATURAL
    )
    assert (
        full_stack_policy.image_execution_mode(
            labels,
            ImagePayloadExecutionMode.NATURAL,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        full_stack_policy.image_execution_mode(
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((3, 8, 8), dtype=np.int32)
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(
                    declared_object_id_domains=((), (), ()),
                    scope=ObjectLabelDomainScope.PLANE,
                ),
            ),
            ImagePayloadExecutionMode.NATURAL,
            runtime_slice_count=3,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        full_stack_policy.image_execution_mode(
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((3, 8, 8), dtype=np.int32)
                ),
                plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
                domain=ObjectLabelDomain(
                    declared_object_id_domains=((), (), ()),
                    scope=ObjectLabelDomainScope.PLANE,
                ),
            ),
            ImagePayloadExecutionMode.FULL_STACK,
            runtime_slice_count=3,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
    assert (
        full_stack_policy.image_execution_mode(
            ObjectLabelPayload(
                variant_data=ObjectLabelVariantData(
                    labels=np.zeros((3, 8, 8), dtype=np.int32)
                ),
                plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
                domain=ObjectLabelDomain(
                    declared_object_id_domains=((), (), ()),
                    scope=ObjectLabelDomainScope.PLANE,
                ),
            ),
            ImagePayloadExecutionMode.FULL_STACK,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )


def test_measurement_execution_policy_preserves_nominal_label_payloads() -> None:
    source_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.zeros((8, 8), dtype=np.int32),
        ),
    )
    completion_payload = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.ones((8, 8), dtype=np.int32),
        ),
    )

    assert (
        CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
            ObjectLabelInputExecutionMode.SLICE_ALIGNED
        ).semantic_label_payload(source_payload, completion_payload)
        is completion_payload
    )
    assert (
        CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
            ObjectLabelInputExecutionMode.FULL_STACK
        ).semantic_label_payload(source_payload, completion_payload)
        is source_payload
    )


def test_slice_aligned_measurement_preserves_payload_scoped_volume() -> None:
    policy = CellProfilerObjectMeasurementExecutionPolicy.for_enum_member(
        ObjectLabelInputExecutionMode.SLICE_ALIGNED
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(labels=np.zeros((3, 8, 8), dtype=np.int32)),
        domain=ObjectLabelDomain(scope=ObjectLabelDomainScope.PAYLOAD),
    )

    assert (
        policy.image_execution_mode(
            labels,
            ImagePayloadExecutionMode.NATURAL,
            runtime_slice_count=3,
        )
        is ImagePayloadExecutionMode.FULL_STACK
    )
