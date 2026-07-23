"""Registry-wide gates for CellProfiler artifact identity reconstruction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.constants.constants import AllComponents
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
from openhcs.core.pipeline.path_planner import PathPlanner, PathPlannerArtifactStage
from openhcs.core.source_bindings import (
    ComponentSelector,
    NamedSourceBinding,
    SourceProjectionRole,
    StepSourceBindingsConfig,
)
from openhcs.core.steps.function_step import FunctionStep
from openhcs.interop.cellprofiler.module_artifact_contracts import (
    CellProfilerModuleArtifactContracts,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.pipeline_import import import_cellprofiler_pipeline
from openhcs.interop.cellprofiler.settings_binder import SettingToKeywordBinding
from openhcs.processing.backends.cellprofiler.color import GrayToColorModule
from openhcs.processing.backends.cellprofiler.neighbors import (
    MeasureObjectNeighborsModule,
    measure_object_neighbors,
)
from openhcs.processing.backends.cellprofiler.relationships import RelateObjectsModule
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureObjectIntensityModule,
    measure_object_intensity,
)


class _ContractProvider(InvocationContractProvider):
    def __init__(self, plan: InvocationContractPlan) -> None:
        self.plan = plan

    def __call__(self, invocation, step_context):
        del invocation, step_context
        return self.plan


def _input_ref_occurrence(*names: str) -> tuple[ArtifactSpecRef, ...]:
    return tuple(
        ArtifactSpec.input(name, ImageArtifactType).ref() for name in names
    )


def test_binding_occurrence_equivalence_preserves_declared_cardinality() -> None:
    main_flow_binding = SettingToKeywordBinding.input(
        "Select the input image",
        ImageArtifactType,
    )
    runtime_binding = SettingToKeywordBinding.input(
        "Select the input image",
        ImageArtifactType,
        runtime_parameter_name="image",
    )
    ordered_occurrences = (
        _input_ref_occurrence("DNA", "RNA"),
        _input_ref_occurrence("Protein"),
    )

    assert CellProfilerModuleArtifactContracts.artifact_input_ref_occurrences_equivalent(
        binding=main_flow_binding,
        target=ordered_occurrences,
        candidate=tuple(reversed(ordered_occurrences)),
    )
    assert not CellProfilerModuleArtifactContracts.artifact_input_ref_occurrences_equivalent(
        binding=main_flow_binding,
        target=ordered_occurrences,
        candidate=(
            _input_ref_occurrence("RNA", "DNA"),
            _input_ref_occurrence("Protein"),
        ),
    )
    assert not CellProfilerModuleArtifactContracts.artifact_input_ref_occurrences_equivalent(
        binding=runtime_binding,
        target=tuple(_input_ref_occurrence(name) for name in ("DNA", "RNA")),
        candidate=(_input_ref_occurrence("DNA", "RNA"),),
    )


def test_registered_artifact_bindings_round_trip_structured_identity_rows() -> None:
    module_types = tuple(CellProfilerModule.__registry__.values())
    visited_bindings: list[SettingToKeywordBinding] = []

    assert module_types
    for module_type in module_types:
        for position, binding in enumerate(module_type.declared_artifact_bindings()):
            identities = (
                (f"Identity_{position}_0", f"Identity_{position}_1")
                if binding.repeated
                else (f"Identity_{position}",)
            )
            records = binding.records_from_kwargs(
                {
                    binding.require_parameter_name(): (
                        identities if binding.repeated else identities[0]
                    )
                }
            )
            block = ModuleBlock(
                name=module_type.require_module_name(),
                module_num=1,
                setting_records=list(records),
            )

            assert (
                CellProfilerModuleArtifactContracts.artifact_names_for_binding(
                    block,
                    binding,
                )
                == identities
            )
            if binding.require_artifact_plan_type() is ArtifactInputPlan:
                assert binding.artifact_input_domain_key() == (
                    binding.require_artifact_type(),
                    binding.sidecar_role,
                )
                assert binding.preserves_artifact_input_occurrence_partitions() is (
                    binding.runtime_parameter_name is not None
                )
            else:
                assert binding.require_artifact_plan_type() is ArtifactOutputPlan
                with pytest.raises(TypeError, match="not an artifact input"):
                    binding.artifact_input_domain_key()
            visited_bindings.append(binding)

    assert visited_bindings


def test_registered_reconstruction_overrides_stay_on_semantic_leaves() -> None:
    module_types = tuple(CellProfilerModule.__registry__.values())

    assert {
        module_type
        for module_type in module_types
        if "module_blocks_for_invocation" in module_type.__dict__
    } == {GrayToColorModule}
    assert {
        module_type
        for module_type in module_types
        if "_artifact_input_record_groups" in module_type.__dict__
    } == {MeasureObjectNeighborsModule}


def test_authored_identity_remains_visible_through_compiled_artifact_graph() -> None:
    object_scope = ComponentSelector(AllComponents.SITE, "A02")
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding("CorrDNA"),
            NamedSourceBinding(
                "Cells",
                artifact_kind=ObjectLabelsArtifactType,
                projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                component_identity=(object_scope,),
            ),
        ),
    )
    object_parameter = (
        MeasureObjectIntensityModule.object_measurement_binding.require_parameter_name()
    )
    image_parameter = (
        MeasureObjectIntensityModule.image_measurement_binding.require_parameter_name()
    )
    step = FunctionStep(
        func=(
            measure_object_intensity,
            {
                image_parameter: "CorrDNA",
                object_parameter: "Cells",
            },
        ),
        name="MeasureObjectIntensity",
        source_bindings=source_bindings,
    )
    step_context = ArtifactDeclarationStepContext(
        step_name=step.name,
        step_index=0,
        source_bindings=source_bindings,
    )
    authored_invocation = next(
        normalize_function_pattern(step.func).iter_items()
    )
    blocks, consumed_names = (
        MeasureObjectIntensityModule.module_blocks_for_invocation(
            invocation=authored_invocation,
            step_context=step_context,
        )
    )
    (numbered_blocks,), _next_module_num = (
        MeasureObjectIntensityModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = MeasureObjectIntensityModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=authored_invocation.key,
        step_context=step_context,
    )
    plan = InvocationContractPlan(
        contract,
        consumed_kwarg_names=consumed_names,
    )

    assert authored_invocation.kwargs_dict[object_parameter] == "Cells"
    assert object_parameter in plan.consumed_kwarg_names
    (object_input,) = contract.artifact_inputs.of_artifact_type(
        ObjectLabelsArtifactType
    )
    assert object_input.ref() == ArtifactSpec.input(
        "Cells",
        ObjectLabelsArtifactType,
    ).ref()
    assert object_input.parameter_name == "labels"

    output_plans = {
        spec.ref(): ArtifactOutputPlan(
            spec.name,
            f"/tmp/{spec.name}",
            artifact_type=spec.artifact_type,
            relations=spec.relations,
        )
        for spec in contract.artifact_outputs
    }
    compiled = compile_function_pattern(
        step.func,
        {},
        output_plans,
        invocation_contract_provider=_ContractProvider(plan),
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
        source_bindings=source_bindings,
        available_artifacts=ArtifactSpecCollection(contract.artifact_inputs),
    )
    compiled_invocation = next(compiled.iter_invocations())
    object_edge = next(
        edge
        for edge in compiled_invocation.artifact_input_edges
        if edge.spec.artifact_type is ObjectLabelsArtifactType
    )
    source_binding = source_bindings.binding_for_artifact_ref(object_edge.spec.ref())

    assert object_parameter not in compiled_invocation.kwargs_dict
    assert object_edge.spec == object_input
    assert source_binding is not None
    assert source_binding.alias == "Cells"
    assert source_binding.component_identity == (object_scope,)


def test_retained_binding_is_not_reproved_across_omitted_sibling_candidates(
    tmp_path,
) -> None:
    pipeline_path = tmp_path / "distinct-parent-child-sources.cppipe"
    pipeline_path.write_text(
        """CellProfiler Pipeline: https://cellprofiler.org
NamesAndTypes:[module_num:1|enabled:True]
    Assignments count:2
    Select the image type:Grayscale image
    Name to assign these images:DNA
    Select the rule criteria:and (file does contain "DNA")
    Select the image type:Grayscale image
    Name to assign these images:RNA
    Select the rule criteria:and (file does contain "RNA")
IdentifyPrimaryObjects:[module_num:2|enabled:True]
    Select the input image:RNA
    Name the primary objects to be identified:Cells
IdentifyPrimaryObjects:[module_num:3|enabled:True]
    Select the input image:DNA
    Name the primary objects to be identified:Nuclei
RelateObjects:[module_num:4|enabled:True]
    Select the parent objects:Nuclei
    Select the child objects:Cells
    Calculate child-parent distances?:None
    Calculate per-parent means for all child measurements?:No
    Calculate distances to other parents?:No
    Do you want to save the children with parents as a new object set?:No
""",
        encoding="utf-8",
    )

    steps, _pipeline_config = import_cellprofiler_pipeline(pipeline_path)
    relate_step = next(step for step in steps if step.name == "RelateObjects")
    (invocation,) = tuple(normalize_function_pattern(relate_step.func).iter_items())

    assert invocation.kwargs_dict[
        RelateObjectsModule.parent_objects_binding.require_parameter_name()
    ] == "Nuclei"
    assert invocation.kwargs_dict[
        RelateObjectsModule.child_objects_binding.require_parameter_name()
    ] == "Cells"


def test_dynamic_contract_composition_rejects_runtime_parameter_drift() -> None:
    contract = CallableContract.from_callable(measure_object_neighbors)
    artifact_inputs = (
        ArtifactSpec.input(
            "Cells",
            ObjectLabelsArtifactType,
            parameter_name="labels",
        ),
        ArtifactSpec.input(
            "Neighbors",
            ObjectLabelsArtifactType,
            parameter_name="neighbor_labels",
        ),
    )
    declared = replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_inputs=artifact_inputs,
        ),
    )
    drifted = replace(
        declared,
        metadata=replace(
            declared.metadata,
            artifact_inputs=(
                replace(artifact_inputs[0], parameter_name="neighbor_labels"),
                replace(artifact_inputs[1], parameter_name="labels"),
            ),
        ),
    )

    with pytest.raises(ValueError, match="conflicting dynamic artifact"):
        MeasureObjectNeighborsModule.combine_callable_contracts((declared, drifted))
