"""Generic CellProfiler special-input binding for masks and object labels."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    ObjectLabelsArtifactType,
)
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_set import ComponentSet
from openhcs.core.function_patterns import (
    compile_function_pattern,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    InvocationContractPlan,
    InvocationContractProvider,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    runtime_bound_parameter_names_from_callable,
    special_input_names_from_callable,
    special_inputs,
)
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage
from openhcs.core.runtime_object_labels import ObjectLabelValue
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.processing.backends.cellprofiler import (
    convert_objects_to_image,
    mask_image,
)
from openhcs.processing.backends.cellprofiler.image_geometry import (
    MaskImageModule,
)
from openhcs.processing.backends.cellprofiler.object_images import (
    ConvertObjectsToImageModule,
)


def _source_binding(alias: str, artifact_type) -> NamedSourceBinding:
    if artifact_type is ImageArtifactType:
        return NamedSourceBinding(alias, artifact_kind=artifact_type)
    return NamedSourceBinding(
        alias,
        artifact_kind=artifact_type,
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
    )


class _ContractProvider(InvocationContractProvider):
    def __init__(self, plan: InvocationContractPlan) -> None:
        self.plan = plan

    def __call__(self, invocation, step_context):
        del invocation, step_context
        return self.plan


def test_registered_module_artifact_parameters_are_exact_special_inputs() -> None:
    declaration_mismatches: list[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
        ]
    ] = []
    for module_type in CellProfilerModule.__registry__.values():
        bound_parameters = {
            binding.runtime_parameter_name
            for binding in module_type.declared_artifact_bindings(
                plan_type=ArtifactInputPlan
            )
            if binding.runtime_parameter_name is not None
        }
        for function_name in module_type.declared_function_names():
            func = module_type.require_callable(function_name)
            signature = inspect.signature(func)
            artifact_parameters = tuple(
                parameter_name
                for parameter_name in signature.parameters
                if parameter_name in bound_parameters
            )
            special_parameters = special_input_names_from_callable(func)
            runtime_parameters = runtime_bound_parameter_names_from_callable(func)
            bound_without_special = tuple(
                parameter_name
                for parameter_name in artifact_parameters
                if parameter_name not in special_parameters
            )
            special_without_binding = tuple(
                parameter_name
                for parameter_name in special_parameters
                if parameter_name not in artifact_parameters
            )
            special_runtime_overlap = tuple(
                parameter_name
                for parameter_name in special_parameters
                if parameter_name in runtime_parameters
            )
            if (
                bound_without_special
                or special_without_binding
                or special_runtime_overlap
            ):
                declaration_mismatches.append(
                    (
                        module_type.require_module_name(),
                        function_name,
                        bound_without_special,
                        special_without_binding,
                        special_runtime_overlap,
                    )
                )

    assert declaration_mismatches == []


def _compile_public_step(step: FunctionStep):
    step_context = ArtifactDeclarationStepContext(
        step_name=step.name,
        step_index=0,
        source_bindings=step.source_bindings,
    )
    invocation = next(normalize_function_pattern(step.func).iter_items())
    module_type = CellProfilerModule.for_function_name(invocation.key.function_name)
    assert module_type is not None
    blocks, consumed_names = module_type.module_blocks_for_invocation(
        invocation=invocation,
        step_context=step_context,
    )
    (numbered_blocks,), _next_module_num = module_type.number_step_invocation_blocks(
        (blocks,),
        first_module_num=1,
    )
    assert len(numbered_blocks) == 1
    contract = module_type.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=step_context,
    )
    provider = _ContractProvider(
        InvocationContractPlan(contract, consumed_kwarg_names=consumed_names)
    )
    output_specs = contract.artifact_outputs
    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            spec.name,
            f"/tmp/{index}_{spec.name}",
            artifact_type=spec.artifact_type,
            sidecar_role=spec.sidecar_role,
            relations=spec.relations,
        )
        for index, spec in enumerate(output_specs)
    }
    compiled = compile_function_pattern(
        step.func,
        {},
        output_plans,
        invocation_contract_provider=provider,
        step_context=step_context,
    )
    compiled = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled,
        artifact_inputs={},
        relation_source_scopes={},
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
        source_bindings=step.source_bindings,
        available_artifacts=ArtifactSpecCollection(
            spec
            for invocation in compiled.iter_invocations()
            for spec in invocation.contract.artifact_inputs
        ),
    )
    return next(compiled.iter_invocations())


def _compile_runtime_input_edges(func, plans: tuple[ArtifactInputPlan, ...]):
    plans_by_ref = {plan.ref(): plan for plan in plans}
    if len(plans_by_ref) != len(plans):
        raise ValueError(
            "One semantic ArtifactSpecRef cannot resolve to multiple storage plans."
        )
    compiled = compile_function_pattern(func, plans_by_ref, {})
    return PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled,
        artifact_inputs=plans_by_ref,
        relation_source_scopes={
            plan.ref(): plan.producer_group_scope() for plan in plans
        },
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
    )


def test_unbound_special_input_fails_without_type_based_inference() -> None:
    source = ArtifactSpec.input("Input", ImageArtifactType)
    mask = ArtifactSpec.input("Mask", ImageArtifactType)

    @artifact_inputs(source, mask)
    @special_inputs("mask")
    def ambiguous_mask(
        image: np.ndarray,
        *,
        mask: np.ndarray,
    ) -> np.ndarray:
        return image

    plans = tuple(
        ArtifactInputPlan(
            spec.name,
            f"/memory/{spec.name}",
            artifact_type=spec.artifact_type,
        )
        for spec in (source, mask)
    )

    with pytest.raises(ValueError, match="mask.*no exact artifact declaration"):
        _compile_runtime_input_edges(ambiguous_mask, plans)


def test_repeated_special_input_occurrences_keep_exact_edge_identity() -> None:
    objects = ArtifactSpec.input(
        "Objects",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )

    @artifact_inputs(objects, objects)
    def repeated_labels(
        image: np.ndarray,
        *,
        labels: tuple[ObjectLabelValue, ...] = (),
    ) -> np.ndarray:
        return image

    plan = ArtifactInputPlan(
        objects.name,
        "/memory/Objects",
        artifact_type=objects.artifact_type,
    )
    compiled = _compile_runtime_input_edges(repeated_labels, (plan,))
    edges = next(compiled.iter_invocations()).artifact_input_edges

    assert tuple(edge.key.input_index for edge in edges) == (0, 1)
    assert tuple(edge.spec.ref() for edge in edges) == (objects.ref(), objects.ref())
    assert tuple(edge.spec.parameter_name for edge in edges) == ("labels", "labels")
    assert tuple(edge.storage_plan for edge in edges) == (plan, plan)


def test_mask_image_image_mask_uses_declared_broadcast_source() -> None:
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            _source_binding("Input", ImageArtifactType),
            _source_binding("Mask", ImageArtifactType),
        ),
    )
    step = FunctionStep(
        func=(
            mask_image,
            {
                MaskImageModule.input_image_binding.require_parameter_name(): "Input",
                MaskImageModule.masking_image_binding.require_parameter_name(): "Mask",
                MaskImageModule.output_image_binding.require_parameter_name(): "Masked",
                MaskImageModule.mask_source_binding.require_parameter_name(): "image",
            },
        ),
        name="MaskImage",
        source_bindings=source_bindings,
    )

    invocation = _compile_public_step(step)
    contract = invocation.contract

    source, mask_spec = contract.artifact_inputs
    assert contract.artifact_inputs.names() == ("Input", "Mask")
    assert source.parameter_name is None
    assert mask_spec.parameter_name == "mask"
    assert mask_spec.relations == (
        InputStackBroadcastSourceRelation(source=source.ref()),
    )
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        source,
        mask_spec,
    )
    assert all(edge.storage_plan is None for edge in invocation.artifact_input_edges)
    assert tuple(
        edge.spec.parameter_name for edge in invocation.artifact_input_edges
    ) == (
        None,
        "mask",
    )


def test_source_bound_object_mask_retains_special_parameter_ownership() -> None:
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            _source_binding("Input", ImageArtifactType),
            _source_binding("Objects", ObjectLabelsArtifactType),
        ),
    )
    step = FunctionStep(
        func=(
            mask_image,
            {
                MaskImageModule.input_image_binding.require_parameter_name(): "Input",
                MaskImageModule.masking_objects_binding.require_parameter_name(): "Objects",
                MaskImageModule.output_image_binding.require_parameter_name(): "Masked",
                MaskImageModule.mask_source_binding.require_parameter_name(): "objects",
            },
        ),
        name="MaskImage",
        source_bindings=source_bindings,
    )

    invocation = _compile_public_step(step)
    contract = invocation.contract

    source, object_spec = contract.artifact_inputs
    assert (source.name, object_spec.name) == (
        "Input",
        "Objects",
    )
    assert source.parameter_name is None
    assert object_spec.parameter_name == "mask"
    assert all(not spec.stack_broadcast_sources() for spec in contract.artifact_inputs)
    assert tuple(edge.spec for edge in invocation.artifact_input_edges) == (
        source,
        object_spec,
    )
    assert all(edge.storage_plan is None for edge in invocation.artifact_input_edges)
    assert tuple(
        edge.spec.parameter_name for edge in invocation.artifact_input_edges
    ) == (
        None,
        "mask",
    )


def test_convert_objects_to_image_uses_generic_positional_label_binding() -> None:
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(_source_binding("Objects", ObjectLabelsArtifactType),),
    )
    step = FunctionStep(
        func=(
            convert_objects_to_image,
            {
                ConvertObjectsToImageModule.declared_artifact_bindings(
                    plan_type=ArtifactInputPlan,
                    artifact_type=ObjectLabelsArtifactType,
                )[0].require_parameter_name(): "Objects",
                ConvertObjectsToImageModule.declared_artifact_bindings(
                    plan_type=ArtifactOutputPlan,
                    artifact_type=ImageArtifactType,
                )[0].require_parameter_name(): "Rendered",
            },
        ),
        name="ConvertObjectsToImage",
        source_bindings=source_bindings,
    )

    invocation = _compile_public_step(step)
    contract = invocation.contract

    (objects,) = contract.artifact_inputs
    assert objects.name == "Objects"
    assert objects.parameter_name == "labels"
    (edge,) = invocation.artifact_input_edges
    assert edge.spec == objects
    assert edge.spec.parameter_name == "labels"
    assert edge.storage_plan is None
