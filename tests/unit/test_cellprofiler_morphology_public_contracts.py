"""Public FunctionStep artifact contracts for CellProfiler morphology modules."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

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
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import DtypeConfig, GlobalPipelineConfig, PipelineConfig
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    FunctionPatternSyntax,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.pipeline.compilation_session import CompilationSession
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.pipeline.function_contracts import (
    artifact_outputs,
    special_input_names_from_callable,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.interop.cellprofiler.compile_time_contracts import (
    CellProfilerInvocationContractProviderFactory,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.processing.backends.cellprofiler.morphology import (
    fill_objects,
    morph,
    morphologicalskeleton,
    shrink_to_object_centers,
    split_or_merge_objects,
    split_or_merge_objects_per_parent,
    split_or_merge_objects_with_guide_image,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


@dataclass(frozen=True, slots=True)
class MorphologyPublicContractCase:
    function: FunctionPatternSyntax
    module_name: str
    source_bindings: tuple[NamedSourceBinding, ...]
    input_types: tuple[type, ...]
    output_types: tuple[type, ...]
    special_inputs: tuple[str, ...]


def _image(name: str) -> NamedSourceBinding:
    return NamedSourceBinding(alias=name, artifact_kind=ImageArtifactType)


def _objects(name: str) -> NamedSourceBinding:
    return NamedSourceBinding(
        alias=name,
        artifact_kind=ObjectLabelsArtifactType,
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
    )


MORPHOLOGY_PUBLIC_CONTRACT_CASES = (
    MorphologyPublicContractCase(
        fill_objects,
        "FillObjects",
        (_objects("InputObjects"),),
        (ObjectLabelsArtifactType,),
        (ObjectLabelsArtifactType,),
        ("labels",),
    ),
    MorphologyPublicContractCase(
        morph,
        "Morph",
        (_image("InputImage"),),
        (ImageArtifactType,),
        (ImageArtifactType,),
        (),
    ),
    MorphologyPublicContractCase(
        morphologicalskeleton,
        "Morphologicalskeleton",
        (_image("InputImage"),),
        (ImageArtifactType,),
        (ImageArtifactType,),
        (),
    ),
    MorphologyPublicContractCase(
        shrink_to_object_centers,
        "ShrinkToObjectCenters",
        (_objects("InputObjects"),),
        (ObjectLabelsArtifactType,),
        (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ("labels",),
    ),
    MorphologyPublicContractCase(
        split_or_merge_objects,
        "SplitOrMergeObjects",
        (_objects("InputObjects"),),
        (ObjectLabelsArtifactType,),
        (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ("labels",),
    ),
    MorphologyPublicContractCase(
        split_or_merge_objects_with_guide_image,
        "SplitOrMergeObjects",
        (_objects("InputObjects"), _image("GuideImage")),
        (ImageArtifactType, ObjectLabelsArtifactType),
        (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ("labels",),
    ),
    MorphologyPublicContractCase(
        split_or_merge_objects_per_parent,
        "SplitOrMergeObjects",
        (_objects("InputObjects"), _objects("ParentObjects")),
        (ObjectLabelsArtifactType, ObjectLabelsArtifactType),
        (MeasurementsArtifactType, ObjectLabelsArtifactType),
        ("labels", "parent_labels"),
    ),
)


def _compiled_contract(case: MorphologyPublicContractCase):
    source_invocation = next(normalize_function_pattern(case.function).iter_items())
    module_type = CellProfilerModule.require_callable_contract_owner(
        source_invocation.contract
    )
    object_sources = tuple(
        binding
        for binding in case.source_bindings
        if binding.artifact_kind is ObjectLabelsArtifactType
    )
    object_input_bindings = module_type.declared_artifact_bindings(
        plan_type=ArtifactInputPlan,
        artifact_type=ObjectLabelsArtifactType,
    )
    assert len(object_sources) <= len(object_input_bindings)
    invocation_kwargs = {
        **source_invocation.kwargs_dict,
        **{
            input_binding.require_parameter_name(): source.alias
            for input_binding, source in zip(
                object_input_bindings[: len(object_sources)],
                object_sources,
                strict=True,
            )
        },
    }
    step = FunctionStep(
        func=(source_invocation.func, invocation_kwargs),
        name=case.module_name,
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=case.source_bindings,
        ),
    )
    source = FunctionStepTransportAuthority.source_from_pipeline([step])
    namespace: dict[str, object] = {}
    exec(compile(source, "<morphology-public-contract>", "exec"), namespace)
    step = FunctionStepTransportAuthority.pipeline_steps_from_namespace(namespace)[0]
    available = tuple(
        ArtifactSpec.output(binding.alias, binding.artifact_kind)
        for binding in case.source_bindings
    )

    @artifact_outputs(*available)
    @numpy(contract=ProcessingContract.PURE_3D)
    def fixture_producer(image):
        return image

    producer_step = FunctionStep(func=fixture_producer, name="FixtureProducer")
    steps = (producer_step, step)
    snapshots = tuple(
        StepSnapshot(
            index=index,
            scope_id=f"test::morphology::{index}",
            step=current_step,
        )
        for index, current_step in enumerate(steps)
    )
    session = CompilationSession.from_context(
        context=ProcessingContext(
            step_plans={
                index: CompiledStepPlan(
                    step_index=index,
                    step_name=current_step.name,
                    step_type=type(current_step).__name__,
                    axis_id="A01",
                )
                for index, current_step in enumerate(steps)
            },
            axis_id="A01",
        ),
        steps=steps,
        orchestrator=SimpleNamespace(pipeline_config=PipelineConfig()),
        global_config=GlobalPipelineConfig(),
        step_state_map={index: object() for index in range(len(steps))},
        snapshots=snapshots,
    )
    provider = CellProfilerInvocationContractProviderFactory.provider_for_session(
        session
    )
    assert provider is not None
    invocation = next(normalize_function_pattern(step.func).iter_items())
    plan = provider(
        invocation,
        ArtifactDeclarationStepContext(
            step_index=1,
            source_bindings=step.source_bindings,
            available_artifacts=ArtifactSpecCollection(available),
            main_flow_artifacts=ArtifactSpecCollection(
                spec.for_plan_type(ArtifactInputPlan)
                for spec in available
                if spec.artifact_type is ImageArtifactType
            ),
            available_artifact_producers=artifact_producers_for_outputs(
                available,
                groups=(None,),
                invocation_keys=(
                    FunctionInvocationKey(
                        "fixture_producer",
                        invocation.key.group_key,
                        0,
                    ),
                ),
            ),
        ),
    )
    assert plan is not None
    return invocation, plan.contract


@pytest.mark.parametrize(
    "case",
    MORPHOLOGY_PUBLIC_CONTRACT_CASES,
    ids=lambda case: case.function.__name__,
)
def test_registry_morphology_owners_compile_exact_public_function_step_abi(
    case: MorphologyPublicContractCase,
) -> None:
    invocation, contract = _compiled_contract(case)
    module_type = CellProfilerModule.for_callable_contract(invocation.contract)

    assert module_type is not None
    assert module_type.module_name == case.module_name
    assert tuple(
        spec.artifact_type for spec in contract.artifact_inputs
    ) == case.input_types
    assert tuple(spec.artifact_type for spec in contract.artifact_outputs) == (
        case.output_types
    )
    assert special_input_names_from_callable(case.function) == case.special_inputs


def test_fill_objects_returns_its_declared_object_label_artifact() -> None:
    result = fill_objects(
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.int32),
        dtype_config=DtypeConfig(),
    )

    assert isinstance(result, ObjectLabelValue)


def test_split_or_merge_rejects_callable_topology_mismatch_at_compile_time() -> None:
    case = MorphologyPublicContractCase(
        (split_or_merge_objects, {"use_guide_image": True}),
        "SplitOrMergeObjects",
        (_objects("InputObjects"), _image("GuideImage")),
        (),
        (),
        (),
    )

    with pytest.raises(ValueError, match="does not match declared input topology"):
        _compiled_contract(case)
