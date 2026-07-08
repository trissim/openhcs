from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from openhcs.constants.constants import AllComponents, GroupBy, VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecRef,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    MaterializedOutputPlan,
)
from openhcs.core.invocation_artifacts import (
    InvocationArtifactDeclarations,
    public_callable_invocation_contract,
)
from openhcs.core.pipeline.artifact_planning import (
    ArtifactConsumer,
    ArtifactGraph,
    ArtifactProducer,
    extract_artifact_declarations,
)
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.core.pipeline.path_planner import (
    ArtifactPlanMaps,
    MissingArtifactInputError,
    PathPlanner,
    PathPlannerComponentScopes,
    PathPlannerArtifactStage,
    PathPlannerExecutionGroups,
    PathPlannerGroupScope,
    PathPlannerMaterializationStage,
    PathPlannerPathAuthority,
    PathPlannerStepAssemblyStage,
    PathPlannerValidationStage,
)
from openhcs.core.runtime_adapters import runtime_adapter
from openhcs.core.source_bindings import (
    ComponentSelector,
    EMPTY_SOURCE_BINDINGS,
    NamedSourceBinding,
    SourceBindingsConfig,
    StepSourceBindingsConfig,
)
from openhcs.core.step_dependencies import StepInputDependency
from openhcs.core.step_dependencies import StepInputDependencyKind


@dataclass(frozen=True)
class PathConfigStub:
    sub_dir: str
    output_dir_suffix: str = "_processed"
    global_output_folder: str | None = None


def _artifact_planner_stub() -> PathPlanner:
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.cfg = PathConfigStub(sub_dir="images")
    planner.session = SimpleNamespace(
        global_config=SimpleNamespace(materialization_results_path="analysis"),
    )
    planner.ctx = SimpleNamespace(
        axis_id="A01",
    )
    planner.plans = {
        2: CompiledStepPlan(
            step_index=2,
            step_scope_id="plate::functionstep_2",
            step_name="identify",
            step_type="FunctionStep",
            axis_id="A01",
        ),
        3: CompiledStepPlan(
            step_index=3,
            step_scope_id="plate::functionstep_3",
            step_name="filter",
            step_type="FunctionStep",
            axis_id="A01",
        )
    }
    planner.declared = {}
    planner.source_bindings_defaults = SourceBindingsConfig()
    planner.step_source_bindings_defaults = StepSourceBindingsConfig()
    planner.invocation_contract_provider = public_callable_invocation_contract
    planner.main_flow_component_scopes = {}
    planner.execution_groups = PathPlannerExecutionGroups(planner)
    planner.paths = PathPlannerPathAuthority(planner)
    planner.artifacts = PathPlannerArtifactStage(planner)
    planner.materialization = PathPlannerMaterializationStage(planner)
    planner.validation = PathPlannerValidationStage(planner)
    planner.steps = PathPlannerStepAssemblyStage(planner)
    return planner


def _function_step_snapshot(
    name: str,
    *,
    source_bindings: StepSourceBindingsConfig = EMPTY_SOURCE_BINDINGS,
    group_by: GroupBy = GroupBy.CHANNEL,
    variable_components: tuple[VariableComponents, ...] = (VariableComponents.SITE,),
):
    return SimpleNamespace(
        name=name,
        source_bindings=source_bindings,
        group_by=group_by,
        variable_components=variable_components,
        input_source=InputSource.PREVIOUS_STEP,
    )


def test_materialization_collision_updates_results_dir_and_config():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plate_path = Path("/data/plate1")
    planner.plans = {
        3: CompiledStepPlan(
            step_index=3,
            step_name="materialize",
            step_type="FunctionStep",
            axis_id="A01",
            materialized_output=MaterializedOutputPlan(
                output_dir=Path("/data/plate1_processed/images"),
                backend="disk",
                plate_root="/data/plate1_processed",
                sub_dir="images",
                analysis_results_dir="/data/plate1_processed/images_results",
            ),
            materialization_config=PathConfigStub(sub_dir="images"),
        )
    }
    snapshot = SimpleNamespace(
        index=3,
        name="materialize",
        materialization_config=PathConfigStub(sub_dir="images"),
    )

    planner.paths = PathPlannerPathAuthority(planner)
    planner.validation = PathPlannerValidationStage(planner)

    planner.validation.resolve_and_update_paths(
        snapshot,
        3,
        Path("/data/plate1_processed/images"),
        "main flow",
    )

    assert snapshot.materialization_config.sub_dir == "images"
    materialized_output = planner.plans[3].materialized_output
    assert materialized_output.output_dir == Path("/data/plate1_processed/images_step3")
    assert materialized_output.sub_dir == "images_step3"
    assert materialized_output.analysis_results_dir == (
        "/data/plate1_processed/images_step3_results"
    )
    assert planner.plans[3].materialization_config.sub_dir == "images_step3"


def test_artifact_output_plans_preserve_declared_kind():
    planner = _artifact_planner_stub()

    outputs = planner.artifacts.process_artifact_outputs(
        {"nuclei": ArtifactSpec.output("nuclei", ObjectLabelsArtifactType)},
        sid=2,
        output_groups={"nuclei": PathPlannerGroupScope.ungrouped()},
        step_name="identify",
    )

    assert outputs["nuclei"].artifact_type is ObjectLabelsArtifactType
    assert planner.declared["nuclei"].artifact_type is ObjectLabelsArtifactType


def test_group_by_namespaces_compiler_owned_outputs():
    @artifact_outputs(ArtifactSpec.output("nuclei", ObjectLabelsArtifactType))
    def identify(image):
        return image

    planner = _artifact_planner_stub()
    declarations = extract_artifact_declarations(identify)

    namespaced = planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
        identify,
        declarations,
        PathPlannerGroupScope.from_raw(("1", "2")),
    )

    assert namespaced.output_groups["nuclei"] == {"1", "2"}


def test_group_by_namespaces_runtime_adapter_artifact_outputs():
    @runtime_adapter(
        "runtime",
        lambda _request: object(),
        manages_artifact_inputs=True,
    )
    def correct_illumination(image, *, runtime):
        return image

    def declarations_for_invocation(invocation, step_context):
        del invocation, step_context
        return InvocationArtifactDeclarations(
            artifacts=(
                ArtifactSpec.output("Hoechst", ImageArtifactType),
            ),
        )

    planner = _artifact_planner_stub()
    declarations = extract_artifact_declarations(
        correct_illumination,
        declaration_provider=declarations_for_invocation,
    )

    namespaced = planner.artifacts.namespace_grouped_outputs_for_runtime_consumers(
        correct_illumination,
        declarations,
        PathPlannerGroupScope.from_raw(("1", "2")),
    )

    assert declarations.output_groups["Hoechst"] == {None}
    assert namespaced.output_groups["Hoechst"] == {"1", "2"}


def test_declared_group_lineage_outputs_inherit_source_group_scope():
    planner = _artifact_planner_stub()
    planner.declared["Tile_of_grid"] = ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    )
    source = ArtifactSpecRef.output("Tile_of_grid", ObjectLabelsArtifactType)
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="Filtered_tiles",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Filtered_tiles",
                    ObjectLabelsArtifactType,
                    source,
                ),
                groups=("2",),
                invocation_keys=(),
            ),
            ArtifactProducer(
                name="FilterObjects_8_measurements",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "FilterObjects_8_measurements",
                    MeasurementsArtifactType,
                    source,
                ),
                groups=("2",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                name="Tile_of_grid",
                spec=ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _function_step_snapshot("FilterObjects"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("2",), component=AllComponents.CHANNEL),
    )

    assert maps.outputs["Filtered_tiles"].group_keys == ("1",)
    assert maps.outputs["FilterObjects_8_measurements"].group_keys == ("1",)
    assert maps.outputs["Filtered_tiles"].group_component is AllComponents.CHANNEL
    assert (
        maps.outputs["FilterObjects_8_measurements"].group_component
        is AllComponents.CHANNEL
    )
    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("1",),
        component=AllComponents.CHANNEL,
    )
    assert set(maps.outputs_by_group) == {"1"}


def test_declared_group_lineage_narrows_scalar_step_execution_scope():
    planner = _artifact_planner_stub()
    planner.declared["MembInvertRemoveHoles"] = ArtifactOutputPlan(
        name="MembInvertRemoveHoles",
        path="/memory/MembInvertRemoveHoles.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/MembInvertRemoveHoles_3.pkl"},
    )
    planner.declared["MonolayerMask"] = ArtifactOutputPlan(
        name="MonolayerMask",
        path="/memory/MonolayerMask.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/MonolayerMask_1.pkl"},
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="MembMasked",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "MembMasked",
                    ImageArtifactType,
                    ArtifactSpecRef.input(
                        "MembInvertRemoveHoles",
                        ImageArtifactType,
                    ),
                ),
                groups=("1", "3"),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                name="MembInvertRemoveHoles",
                spec=ArtifactSpec.input("MembInvertRemoveHoles", ImageArtifactType),
                invocation_keys=(),
            ),
            ArtifactConsumer(
                name="MonolayerMask",
                spec=ArtifactSpec.input("MonolayerMask", ImageArtifactType),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _function_step_snapshot("MaskImage"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("1", "3"), component=AllComponents.CHANNEL),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("3",),
        component=AllComponents.CHANNEL,
    )
    assert maps.outputs["MembMasked"].group_keys == ("3",)
    assert set(maps.inputs_by_group) == {"3"}
    assert set(maps.outputs_by_group) == {"3"}


def test_dict_pattern_output_groups_do_not_drive_scalar_scope_narrowing():
    planner = _artifact_planner_stub()
    planner.declared["source"] = ArtifactOutputPlan(
        name="source",
        path="/memory/source.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("3",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"3": "/memory/source_3.pkl"},
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="output",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "output",
                    ImageArtifactType,
                    ArtifactSpecRef.input("source", ImageArtifactType),
                ),
                groups=("1",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                name="source",
                spec=ArtifactSpec.input("source", ImageArtifactType),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _function_step_snapshot("dict_pattern"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("1", "3"), component=AllComponents.CHANNEL),
    )

    assert maps.group_scope == PathPlannerGroupScope.from_raw(
        ("1", "3"),
        component=AllComponents.CHANNEL,
    )


def test_source_binding_component_identity_narrows_declared_output_lineage():
    planner = _artifact_planner_stub()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="Cells",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Cells",
                    ObjectLabelsArtifactType,
                    ArtifactSpecRef.input("origMemb", ImageArtifactType),
                ),
                groups=("1", "2", "3"),
                invocation_keys=(),
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="origMemb",
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "3"),
                ),
            ),
        ),
        enabled=True,
    )

    maps = planner.artifacts.compile_plan_maps(
        _function_step_snapshot("Watershed", source_bindings=source_bindings),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(
            ("1", "2", "3"),
            component=AllComponents.CHANNEL,
        ),
    )

    assert maps.outputs["Cells"].group_keys == ("3",)
    assert maps.outputs["Cells"].group_component is AllComponents.CHANNEL
    assert set(maps.outputs_by_group) == {"3"}


def test_source_binding_identity_scopes_outputs_without_execution_fanout():
    planner = _artifact_planner_stub()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="Cells",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Cells",
                    ObjectLabelsArtifactType,
                    ArtifactSpecRef.input("origMemb", ImageArtifactType),
                ),
                groups=(None,),
                invocation_keys=(),
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        bindings=(
            NamedSourceBinding(
                alias="origMemb",
                component_identity=(
                    ComponentSelector(AllComponents.CHANNEL, "3"),
                ),
            ),
        ),
        enabled=True,
    )

    maps = planner.artifacts.compile_plan_maps(
        _function_step_snapshot("Watershed", source_bindings=source_bindings),
        3,
        declarations,
        PathPlannerGroupScope.ungrouped(),
    )

    assert maps.group_scope == PathPlannerGroupScope.ungrouped()
    assert maps.outputs["Cells"].group_keys == ("3",)
    assert maps.outputs["Cells"].group_component is AllComponents.CHANNEL
    assert set(maps.outputs_by_group) == {"3"}


def test_image_object_outputs_keep_declared_image_execution_group_scope():
    planner = _artifact_planner_stub()
    planner.ctx.microscope_handler = SimpleNamespace(
        can_resolve_metadata_artifact=lambda artifact_name: artifact_name == "DF_image",
    )
    planner.declared["Tile_of_grid"] = ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="MeasureObjectIntensity_7_measurements",
                spec=ArtifactSpec.output(
                    "MeasureObjectIntensity_7_measurements",
                    MeasurementsArtifactType,
                ),
                groups=("2",),
                invocation_keys=(),
            ),
        ),
        consumers=(
            ArtifactConsumer(
                name="DF_image",
                spec=ArtifactSpec.input("DF_image", ImageArtifactType),
                invocation_keys=(),
            ),
            ArtifactConsumer(
                name="Tile_of_grid",
                spec=ArtifactSpec.input("Tile_of_grid", ObjectLabelsArtifactType),
                invocation_keys=(),
            ),
        ),
    )

    maps = planner.artifacts.compile_plan_maps(
        _function_step_snapshot("MeasureObjectIntensity"),
        3,
        declarations,
        PathPlannerGroupScope.from_raw(("2",)),
    )

    assert maps.outputs["MeasureObjectIntensity_7_measurements"].group_keys == ("2",)
    assert set(maps.outputs_by_group) == {"2"}


def test_group_lineage_source_resolution_uses_full_artifact_ref():
    planner = _artifact_planner_stub()
    planner.declared["Tile_of_grid"] = ArtifactOutputPlan(
        name="Tile_of_grid",
        path="/memory/Tile_of_grid.pkl",
        artifact_type=ObjectLabelsArtifactType,
        group_keys=("1",),
        paths_by_group={"1": "/memory/Tile_of_grid_1.pkl"},
    )
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="Filtered_tiles",
                spec=ArtifactSpec.output_inheriting_group_scope(
                    "Filtered_tiles",
                    ObjectLabelsArtifactType,
                    ArtifactSpecRef.input("Tile_of_grid", ObjectLabelsArtifactType),
                ),
                groups=("2",),
                invocation_keys=(),
            ),
        ),
    )

    with pytest.raises(MissingArtifactInputError, match="Tile_of_grid"):
        planner.artifacts.compile_plan_maps(
            _function_step_snapshot("FilterObjects"),
            3,
            declarations,
            PathPlannerGroupScope.from_raw(("2",)),
        )


def test_planner_uses_invocation_aware_artifact_declaration_provider():
    def identify(image, artifact_name: str):
        return image

    def declarations_for_invocation(invocation, step_context):
        assert step_context.step_name == "identify_cells"
        artifact_name = dict(invocation.kwargs)["artifact_name"]
        return InvocationArtifactDeclarations(
            artifacts=(
                ArtifactSpec.output(artifact_name, ObjectLabelsArtifactType),
            ),
        )

    planner = _artifact_planner_stub()
    planner.declaration_provider = declarations_for_invocation
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=(identify, {"artifact_name": "cells"}),
        group_by=GroupBy.NONE,
        variable_components=(VariableComponents.SITE,),
        name="identify_cells",
        index=2,
        source_bindings=EMPTY_SOURCE_BINDINGS,
        processing_config=None,
        callable_runtime_config_bindings=(),
        input_source=InputSource.PREVIOUS_STEP,
    )

    declarations, _execution_groups, func_pattern = planner.artifacts.prepare_step_declarations(
        snapshot,
    )
    compiled = planner.artifacts.build_step_compiled_function_pattern(
        snapshot,
        True,
        func_pattern,
        {},
        {
            "cells": ArtifactOutputPlan(
                name="cells",
                path="/memory/cells.pkl",
                artifact_type=ObjectLabelsArtifactType,
            )
        },
    )

    assert list(declarations.outputs) == ["cells"]
    assert compiled.groups[0].invocations[0].artifact_output_keys == ("cells",)


def test_execution_groups_reject_variable_component_group_by_conflicts():
    planner = _artifact_planner_stub()
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE, VariableComponents.CHANNEL),
        name="source_bound_cellprofiler_step",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )

    with pytest.raises(
        ValueError,
        match=(
            "source_bound_cellprofiler_step.*group_by=CHANNEL cannot also appear "
            "in variable_components"
        ),
    ):
        planner.execution_groups.get_execution_groups(snapshot)


def test_non_dict_group_by_does_not_create_execution_scope_from_plate_keys():
    planner = _artifact_planner_stub()
    planner.orchestrator = SimpleNamespace(
        get_component_keys=lambda group_by: pytest.fail(
            "non-dict group_by must not request plate component keys"
        )
    )
    source_snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="enhance",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    source_scope = planner.execution_groups.get_execution_groups(
        source_snapshot,
        PathPlannerComponentScopes.empty(),
    )
    assert source_scope == PathPlannerGroupScope.ungrouped()


def test_non_dict_group_by_uses_dynamic_source_scope_for_pipeline_start():
    planner = _artifact_planner_stub()
    planner.orchestrator = SimpleNamespace(
        pipeline_config=SimpleNamespace(source_bindings_config=SourceBindingsConfig())
    )
    source_snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="source_loaded_channel_callable",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PIPELINE_START,
    )

    source_scope = planner.execution_groups.get_execution_groups(
        source_snapshot,
        PathPlannerComponentScopes.empty(),
    )
    assert source_scope == PathPlannerGroupScope.from_raw(
        (None,),
        component=AllComponents.CHANNEL,
    )


def test_dict_pattern_group_by_declares_execution_group_component():
    planner = _artifact_planner_stub()
    snapshot = SimpleNamespace(
        is_function_step=True,
        func={
            "1": lambda image: image,
            "2": lambda image: image,
        },
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="channel_dispatch",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )


def test_dict_pattern_rejects_group_by_none_execution_component():
    planner = _artifact_planner_stub()
    snapshot = SimpleNamespace(
        is_function_step=True,
        func={
            "1": lambda image: image,
            "2": lambda image: image,
        },
        group_by=GroupBy.NONE,
        variable_components=(VariableComponents.CHANNEL,),
        name="channel_dispatch",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )

    with pytest.raises(
        ValueError,
        match="dict function pattern without a concrete group_by component",
    ):
        planner.execution_groups.get_execution_groups(
            snapshot,
            PathPlannerComponentScopes.empty(),
        )


def test_execution_groups_reject_composite_group_by_axis_conflict():
    planner = _artifact_planner_stub()

    composite_snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.CHANNEL,),
        name="create_composite",
        source_bindings=EMPTY_SOURCE_BINDINGS,
    )
    with pytest.raises(
        ValueError,
        match="create_composite.*group_by=CHANNEL cannot also appear",
    ):
        planner.execution_groups.get_execution_groups(
            composite_snapshot,
            PathPlannerComponentScopes.empty(),
        )


def test_non_dict_group_by_uses_input_component_scope_before_plate_keys():
    planner = _artifact_planner_stub()
    input_scopes = PathPlannerComponentScopes(
        {
            VariableComponents.CHANNEL: PathPlannerGroupScope.ungrouped(),
            VariableComponents.SITE: PathPlannerGroupScope.from_raw(
                ("1", "2"),
                component=AllComponents.SITE,
            ),
        }
    )
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="measure_channel_named_artifacts_over_site_stack",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    scope = planner.execution_groups.get_execution_groups(snapshot, input_scopes)

    assert scope == PathPlannerGroupScope.ungrouped()


def test_non_dict_group_by_namespaces_artifact_outputs_with_dynamic_component():
    planner = _artifact_planner_stub()
    declarations = ArtifactGraph(
        producers=(
            ArtifactProducer(
                name="segmentation_masks",
                spec=ArtifactSpec.output(
                    "segmentation_masks",
                    ObjectLabelsArtifactType,
                ),
                groups=(None,),
                invocation_keys=(),
            ),
        )
    )
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="single_callable_channel_artifacts",
        source_bindings=EMPTY_SOURCE_BINDINGS,
        input_source=InputSource.PREVIOUS_STEP,
    )

    maps = planner.artifacts.compile_plan_maps(
        snapshot,
        2,
        declarations,
        PathPlannerGroupScope.from_raw(
            (None,),
            component=AllComponents.CHANNEL,
        ),
    )

    output_plan = maps.outputs["segmentation_masks"]
    assert output_plan.group_keys == (None,)
    assert output_plan.group_component is AllComponents.CHANNEL
    assert maps.outputs_by_group[None]["segmentation_masks"].group_component is (
        AllComponents.CHANNEL
    )


def test_non_dict_group_by_uses_source_binding_identity_for_pipeline_start_scope():
    planner = _artifact_planner_stub()
    planner.orchestrator = SimpleNamespace(
        pipeline_config=SimpleNamespace(source_bindings_config=SourceBindingsConfig())
    )
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="source_bound_channel_groups",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigStain1",
                    component_identity=(
                        ComponentSelector(AllComponents.CHANNEL, "1"),
                    ),
                ),
                NamedSourceBinding(
                    alias="OrigStain2",
                    component_identity=(
                        ComponentSelector(AllComponents.CHANNEL, "2"),
                    ),
                ),
            ),
        ),
        input_source=InputSource.PIPELINE_START,
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        ("1", "2"),
        component=AllComponents.CHANNEL,
    )


def test_non_dict_group_by_ignores_source_binding_identity_for_other_components():
    planner = _artifact_planner_stub()
    planner.orchestrator = SimpleNamespace(
        pipeline_config=SimpleNamespace(source_bindings_config=SourceBindingsConfig())
    )
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.SITE,
        variable_components=(VariableComponents.CHANNEL,),
        name="source_bound_site_groups",
        source_bindings=StepSourceBindingsConfig(
            enabled=True,
            bindings=(
                NamedSourceBinding(
                    alias="OrigStain1",
                    component_identity=(
                        ComponentSelector(AllComponents.CHANNEL, "1"),
                    ),
                ),
                NamedSourceBinding(
                    alias="OrigStain2",
                    component_identity=(
                        ComponentSelector(AllComponents.CHANNEL, "2"),
                    ),
                ),
            ),
        ),
        input_source=InputSource.PIPELINE_START,
    )

    scope = planner.execution_groups.get_execution_groups(
        snapshot,
        PathPlannerComponentScopes.empty(),
    )

    assert scope == PathPlannerGroupScope.from_raw(
        (None,),
        component=AllComponents.SITE,
    )


def test_compiled_group_by_preserves_identity_for_ungrouped_execution_scope():
    planner = _artifact_planner_stub()
    planner.cfg = PathConfigStub(sub_dir="images", output_dir_suffix="_generated")
    planner.plans[3].group_by = GroupBy.CHANNEL
    planner.plans[3].variable_components = (VariableComponents.SITE,)
    planner.plans[3].func = lambda image: image
    snapshot = SimpleNamespace(
        is_function_step=True,
        func=lambda image: image,
        group_by=GroupBy.CHANNEL,
        variable_components=(VariableComponents.SITE,),
        name="measure_after_channel_collapse",
        scope_id="plate::functionstep_3",
        input_source=InputSource.PREVIOUS_STEP,
    )
    artifact_maps = ArtifactPlanMaps(
        declarations=ArtifactGraph.empty(),
        group_scope=PathPlannerGroupScope.ungrouped(),
        inputs={},
        outputs={},
        inputs_by_group={None: {}},
        outputs_by_group={None: {}},
    )

    planner.steps.update_core_step_plan(
        snapshot,
        3,
        StepInputDependency.step_output(
            source_step_index=2,
            source_step_scope_id="plate::functionstep_2",
        ),
        Path("/input"),
        Path("/output"),
        artifact_maps,
        None,
    )

    assert planner.plans[3].group_by is GroupBy.CHANNEL
    assert planner.plans[3].execution_groups == [None]


def test_artifact_input_plan_rejects_producer_consumer_kind_mismatch():
    planner = _artifact_planner_stub()
    planner.declared["nuclei"] = ArtifactOutputPlan(
        name="nuclei",
        path="/memory/nuclei.pkl",
        artifact_type=ObjectLabelsArtifactType,
        producer_step_index=1,
        producer_step_name="identify",
    )

    with pytest.raises(ValueError, match="expects measurements"):
        planner.artifacts.process_artifact_inputs(
            {"nuclei": ArtifactSpec.input("nuclei", MeasurementsArtifactType)},
            {},
            consumer_scope=PathPlannerGroupScope.ungrouped(),
            sid=2,
            step_name="measure",
        )


def test_artifact_input_plan_preserves_single_grouped_producer_scope():
    planner = _artifact_planner_stub()
    planner.declared["illumination"] = ArtifactOutputPlan(
        name="illumination",
        path="/memory/illumination.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1",),
        group_component=AllComponents.CHANNEL,
        paths_by_group={"1": "/memory/illumination_channel_1.pkl"},
        producer_step_index=1,
        producer_step_name="calculate_illumination",
    )

    inputs = planner.artifacts.process_artifact_inputs(
        {"illumination": ArtifactSpec.input("illumination", ImageArtifactType)},
        {},
        consumer_scope=PathPlannerGroupScope.from_raw(("2", "3")),
        sid=2,
        step_name="apply_illumination",
    )

    plan = inputs["illumination"]
    assert plan.group_keys == ("1",)
    assert plan.group_component is AllComponents.CHANNEL
    assert plan.path == "/memory/illumination_channel_1.pkl"
    assert plan.paths_by_group == {"1": "/memory/illumination_channel_1.pkl"}
    inputs_by_group = planner.paths.artifact_inputs_by_group(
        inputs,
        PathPlannerGroupScope.from_raw(
            ("2", "3"),
            component=AllComponents.SITE,
        ),
    )
    assert tuple(inputs_by_group) == ("2", "3")
    assert (
        inputs_by_group["2"]["illumination"].path
        == "/memory/illumination_channel_1.pkl"
    )
    assert inputs_by_group["3"]["illumination"].group_keys == ("1",)


def test_artifact_input_plan_preserves_multi_grouped_producer_across_components():
    planner = _artifact_planner_stub()
    planner.declared["illumination"] = ArtifactOutputPlan(
        name="illumination",
        path="/memory/illumination_channel_1.pkl",
        artifact_type=ImageArtifactType,
        group_keys=("1", "2"),
        group_component=AllComponents.CHANNEL,
        paths_by_group={
            "1": "/memory/illumination_channel_1.pkl",
            "2": "/memory/illumination_channel_2.pkl",
        },
        producer_step_index=1,
        producer_step_name="calculate_illumination",
    )

    inputs = planner.artifacts.process_artifact_inputs(
        {"illumination": ArtifactSpec.input("illumination", ImageArtifactType)},
        {},
        consumer_scope=PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.SITE,
        ),
        sid=2,
        step_name="apply_illumination",
    )

    plan = inputs["illumination"]
    assert plan.group_keys == ("1", "2")
    assert plan.group_component is AllComponents.CHANNEL
    assert plan.paths_by_group == {
        "1": "/memory/illumination_channel_1.pkl",
        "2": "/memory/illumination_channel_2.pkl",
    }


def test_artifact_inputs_by_group_does_not_match_raw_keys_across_components():
    planner = _artifact_planner_stub()
    inputs = {
        "illumination": ArtifactInputPlan(
            name="illumination",
            path="/memory/illumination_channel_1.pkl",
            artifact_type=ImageArtifactType,
            group_keys=("1", "2"),
            group_component=AllComponents.CHANNEL,
            paths_by_group={
                "1": "/memory/illumination_channel_1.pkl",
                "2": "/memory/illumination_channel_2.pkl",
            },
        )
    }

    inputs_by_group = planner.paths.artifact_inputs_by_group(
        inputs,
        PathPlannerGroupScope.from_raw(
            ("1", "2"),
            component=AllComponents.SITE,
        ),
    )

    assert inputs_by_group == {}


def test_main_input_dependency_uses_scope_identity_for_step_output_edges():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        0: CompiledStepPlan(
            step_index=0,
            step_scope_id="plate::functionstep_0",
            step_name="load",
            step_type="FunctionStep",
            axis_id="A01",
            output_dir=Path("/data/plate1_processed/images"),
        ),
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="measure",
            step_type="FunctionStep",
            axis_id="A01",
        ),
    }
    snapshots_by_index = {
        0: SimpleNamespace(scope_id="plate::functionstep_0"),
        1: SimpleNamespace(scope_id="plate::functionstep_1"),
    }
    planner.session = SimpleNamespace(
        snapshot=lambda index: snapshots_by_index[index],
    )
    planner.steps = PathPlannerStepAssemblyStage(planner)

    dependency = planner.steps.main_input_dependency(
        SimpleNamespace(input_source=None),
        1,
    )

    assert dependency.kind is StepInputDependencyKind.STEP_OUTPUT
    assert dependency.source_step_index == 0
    assert dependency.source_step_scope_id == "plate::functionstep_0"

    input_dir, output_dir = planner.steps.step_io_dirs(dependency, 1)
    assert input_dir == Path("/data/plate1_processed/images")
    assert output_dir == Path("/data/plate1_processed/images")


def test_main_input_dependency_preserves_pipeline_start_edges():
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        1: CompiledStepPlan(
            step_index=1,
            step_scope_id="plate::functionstep_1",
            step_name="qc",
            step_type="FunctionStep",
            axis_id="A01",
        )
    }
    planner.initial_input = Path("/data/plate1/images")
    planner.session = SimpleNamespace(
        snapshot=lambda index: {1: SimpleNamespace(scope_id="plate::functionstep_1")}[index],
    )
    planner.paths = SimpleNamespace(
        build_output_path=lambda *_args, **_kwargs: Path(
            "/data/plate1_processed/images"
        )
    )
    planner.steps = PathPlannerStepAssemblyStage(planner)

    dependency = planner.steps.main_input_dependency(
        SimpleNamespace(input_source=InputSource.PIPELINE_START),
        1,
    )

    assert dependency.kind is StepInputDependencyKind.PIPELINE_START
    input_dir, output_dir = planner.steps.step_io_dirs(dependency, 1)
    assert input_dir == Path("/data/plate1/images")
    assert output_dir == Path("/data/plate1_processed/images")
