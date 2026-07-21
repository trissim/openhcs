"""Adaptive guide-threshold contracts shared by CellProfiler modules."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.config import DtypeConfig
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    NormalizedFunctionItem,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.primary_objects import (
    FillHolesOption,
    IdentifyPrimaryObjectsModule,
    UnclumpMethod,
    WatershedMethod,
    identify_primary_objects,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    IdentifySecondaryObjectsModule,
    SecondaryMethod,
    identify_secondary_objects,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    AdaptiveObjectThresholdResult,
    CellProfilerThresholdRequest,
    CellProfilerThresholdResult,
    CellProfilerThresholdScope,
    ObjectThresholdResult,
    ThresholdModule,
    threshold,
)


THRESHOLD_MODULE_TYPES = (
    ThresholdModule,
    IdentifyPrimaryObjectsModule,
    IdentifySecondaryObjectsModule,
)
GLOBAL_RESULT_FIELDS = (
    "slice_index",
    "final_threshold",
    "original_threshold",
    "weighted_variance",
    "sum_of_entropies",
)
ADAPTIVE_RESULT_FIELDS = (
    "slice_index",
    "final_threshold",
    "original_threshold",
    "guide_threshold",
    "weighted_variance",
    "sum_of_entropies",
)


def _module_block(module_type: type, scope: CellProfilerThresholdScope) -> ModuleBlock:
    return ModuleBlock(
        name=str(module_type.module_name),
        module_num=1,
        setting_records=[ModuleSetting("Threshold strategy", scope.value)],
    )


def _public_invocation(
    module_type: type,
    scope: CellProfilerThresholdScope,
) -> NormalizedFunctionItem:
    step = FunctionStep(
        func=(module_type.require_callable(), {"threshold_scope": scope}),
    )
    return next(normalize_function_pattern(step.function_spec()).iter_items())


def _step_context(module_type: type) -> ArtifactDeclarationStepContext:
    image = ArtifactSpec.output("DNA", ImageArtifactType)
    objects = ArtifactSpec.output("Nuclei", ObjectLabelsArtifactType)
    return ArtifactDeclarationStepContext(
        step_name=str(module_type.module_name),
        step_index=0,
        available_artifact_producers=artifact_producers_for_outputs(
            (image, objects),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey("fixture_producer", DEFAULT_GROUP_KEY, 0),
            ),
        ),
        available_artifacts=ArtifactSpecCollection((image, objects)),
        main_flow_artifacts=ArtifactSpecCollection(
            (image.for_plan_type(ArtifactInputPlan),)
        ),
    )


def test_adaptive_result_schema_extends_shared_typed_threshold_fields() -> None:
    assert tuple(
        field.name for field in fields(ObjectThresholdResult)
    ) == GLOBAL_RESULT_FIELDS
    assert tuple(field.name for field in fields(AdaptiveObjectThresholdResult)) == (
        ADAPTIVE_RESULT_FIELDS
    )

    adaptive_rows = CellProfilerThresholdResult(
        final_threshold=0.5,
        original_threshold=0.5,
        guide_threshold=None,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        mask=np.ones((2, 2), dtype=bool),
    ).measurement_rows()
    assert adaptive_rows.row_type is AdaptiveObjectThresholdResult
    assert "guide_threshold" in adaptive_rows.columns


@pytest.mark.parametrize("module_type", THRESHOLD_MODULE_TYPES)
def test_all_threshold_module_declarations_select_scope_owned_row_schema(
    module_type: type,
) -> None:
    assert (
        module_type.threshold_measurement_row_type(
            _module_block(module_type, CellProfilerThresholdScope.GLOBAL)
        )
        is ObjectThresholdResult
    )
    assert (
        module_type.threshold_measurement_row_type(
            _module_block(module_type, CellProfilerThresholdScope.ADAPTIVE)
        )
        is AdaptiveObjectThresholdResult
    )


@pytest.mark.parametrize("module_type", THRESHOLD_MODULE_TYPES)
@pytest.mark.parametrize(
    ("scope", "guide_threshold", "expected_row_type", "expected_fields"),
    (
        (
            CellProfilerThresholdScope.GLOBAL,
            None,
            ObjectThresholdResult,
            GLOBAL_RESULT_FIELDS,
        ),
        (
            CellProfilerThresholdScope.ADAPTIVE,
            0.4,
            AdaptiveObjectThresholdResult,
            ADAPTIVE_RESULT_FIELDS,
        ),
    ),
)
def test_all_threshold_module_runtime_projections_preserve_conditional_schema(
    module_type: type,
    scope: CellProfilerThresholdScope,
    guide_threshold: float | None,
    expected_row_type: type,
    expected_fields: tuple[str, ...],
) -> None:
    source_rows = CellProfilerThresholdResult(
        final_threshold=0.5,
        original_threshold=0.45,
        guide_threshold=guide_threshold,
        threshold_scope=scope,
        mask=np.ones((3, 3), dtype=bool),
        weighted_variance=0.1,
        sum_of_entropies=-0.2,
    ).measurement_rows()

    assert source_rows.row_type is expected_row_type
    assert tuple(field.name for field in source_rows.fields) == expected_fields

    output_name = "Binary" if module_type is ThresholdModule else "Objects"
    projected = module_type.MeasurementRows(
        source_rows,
        module_type=module_type,
        object_name=output_name,
    ).rows()
    projected_fields = tuple(field.name for field in projected.fields)
    guide_feature = f"Threshold_GuideThreshold_{output_name}"

    assert (guide_feature in projected_fields) is (guide_threshold is not None)
    assert projected.row_mappings()[0].get(guide_feature) == guide_threshold


def test_threshold_runtime_emits_guide_only_for_adaptive_scope() -> None:
    image = np.arange(64, dtype=np.float32).reshape(8, 8) / 63

    _global_image, global_rows = threshold(
        image,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        smoothing=0.0,
        dtype_config=DtypeConfig(),
    )
    _adaptive_image, adaptive_rows = threshold(
        image,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        window_size=4,
        smoothing=0.0,
        dtype_config=DtypeConfig(),
    )

    assert global_rows.row_type is ObjectThresholdResult
    assert "guide_threshold" not in global_rows.columns
    assert adaptive_rows.row_type is AdaptiveObjectThresholdResult
    assert adaptive_rows.columns["guide_threshold"][0] == pytest.approx(0.490234375)


@pytest.mark.parametrize("module_type", THRESHOLD_MODULE_TYPES)
@pytest.mark.parametrize(
    ("scope", "expected_row_type"),
    (
        (CellProfilerThresholdScope.GLOBAL, ObjectThresholdResult),
        (CellProfilerThresholdScope.ADAPTIVE, AdaptiveObjectThresholdResult),
    ),
)
def test_public_function_step_threshold_scope_reconstructs_exact_contract(
    module_type: type,
    scope: CellProfilerThresholdScope,
    expected_row_type: type,
) -> None:
    invocation = _public_invocation(module_type, scope)
    step_context = _step_context(module_type)
    blocks, consumed = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )

    assert consumed == ()
    assert len(blocks) == 1
    assert module_type.setting_value(blocks[0], "Threshold strategy") == scope.value
    assert (
        module_type.threshold_measurement_row_type(blocks[0])
        is expected_row_type
    )
    (numbered_blocks,), _next_module_num = (
        module_type.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract, contract_consumed = module_type.invocation_callable_contract(
        invocation=invocation,
        numbered_module_blocks=numbered_blocks,
        consumed_kwarg_names=consumed,
        step_context=step_context,
    )
    assert contract_consumed == ()
    assert len(
        contract.artifact_outputs.of_artifact_type(
            MeasurementsArtifactType
        )
    ) == 1
    assert (
        module_type.resolve_function(
            numbered_blocks[0],
            contract=contract,
            source_bindings=StepSourceBindingsConfig(),
        )
        is module_type.require_callable()
    )


def _stub_threshold_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    def calculate(request: CellProfilerThresholdRequest) -> CellProfilerThresholdResult:
        return CellProfilerThresholdResult(
            final_threshold=0.5,
            original_threshold=0.45,
            guide_threshold=0.4,
            threshold_scope=request.settings.normalized().threshold_scope,
            mask=np.zeros_like(request.image, dtype=bool),
            weighted_variance=0.1,
            sum_of_entropies=-0.2,
        )

    monkeypatch.setattr(CellProfilerThresholdRequest, "calculate", calculate)


def test_identify_primary_objects_runtime_uses_scope_owned_threshold_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_threshold_calculation(monkeypatch)
    image = np.zeros((8, 8), dtype=np.float32)
    runtime_kwargs = {
        "min_diameter": 1,
        "max_diameter": 8,
        "exclude_size": False,
        "exclude_border_objects": False,
        "unclump_method": UnclumpMethod.NONE,
        "watershed_method": WatershedMethod.NONE,
        "fill_holes": FillHolesOption.NEVER,
        "threshold_smoothing_scale": 0.0,
        "dtype_config": DtypeConfig(),
    }

    _image, global_rows, _labels = identify_primary_objects(
        image,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        **runtime_kwargs,
    )
    _image, adaptive_rows, _labels = identify_primary_objects(
        image,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        **runtime_kwargs,
    )

    assert global_rows.row_type is ObjectThresholdResult
    assert "guide_threshold" not in global_rows.columns
    assert adaptive_rows.row_type is AdaptiveObjectThresholdResult
    assert adaptive_rows.columns["guide_threshold"] == (0.4,)


def test_identify_secondary_objects_runtime_uses_scope_owned_threshold_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_threshold_calculation(monkeypatch)
    image = np.zeros((8, 8), dtype=np.float32)
    primary_labels = SourceImageObjectLabelBuildRequest(
        image=image,
        labels=np.zeros_like(image, dtype=np.int32),
    ).payload()
    runtime_kwargs = {
        "primary_labels": primary_labels,
        "method": SecondaryMethod.DISTANCE_B,
        "fill_holes": False,
        "discard_edge_objects": False,
        "dtype_config": DtypeConfig(),
    }

    _image, global_rows, _relationships, _labels = identify_secondary_objects(
        image,
        threshold_scope=CellProfilerThresholdScope.GLOBAL,
        **runtime_kwargs,
    )
    _image, adaptive_rows, _relationships, _labels = identify_secondary_objects(
        image,
        threshold_scope=CellProfilerThresholdScope.ADAPTIVE,
        **runtime_kwargs,
    )

    assert global_rows.row_type is ObjectThresholdResult
    assert "guide_threshold" not in global_rows.columns
    assert adaptive_rows.row_type is AdaptiveObjectThresholdResult
    assert adaptive_rows.columns["guide_threshold"] == (0.4,)
