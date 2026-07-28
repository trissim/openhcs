"""Generic axis/plate FunctionStep execution-scope contracts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import fields, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import time

import numpy as np
import pytest
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    MeasurementsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import CallableContract, FunctionStepExecutionScope
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.component_set import ComponentSet
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.function_patterns import (
    compile_function_pattern,
    resolve_function_pattern_execution_scope,
)
from openhcs.core.invocation_artifacts import (
    ArtifactDeclarationStepContext,
    CompositeInvocationContractProvider,
)
from openhcs.core.orchestrator.compiled_plate_execution import (
    _plate_artifact_batch,
    execute_plate_scoped_steps,
    validate_plate_scoped_contexts,
)
from openhcs.core.progress import (
    ProgressEvent,
    ProgressExecutionContext,
    ProgressPhase,
)
from openhcs.core.pipeline.function_contracts import (
    artifact_inputs,
    artifact_outputs,
    execution_scope,
    runtime_bound_parameters,
)
from openhcs.core.pipeline.path_planner import (
    PathPlanner,
    PathPlannerArtifactStage,
    PathPlannerStepAssemblyStage,
    PathPlannerValidationStage,
)
from openhcs.core.pipeline.step_snapshot import StepSnapshot
from openhcs.core.runtime_artifact_values import RuntimeValue
from openhcs.core.runtime_batch_contracts import RuntimeBatchExecutionDomain
from openhcs.core.measurement_row_materialization import (
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_measurements import (
    MeasurementTable,
)
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.runtime_stores import (
    RuntimeArtifactBatch,
    RuntimeArtifactLocation,
    replace_runtime_artifact_payload,
)
from openhcs.core.source_bindings import CompiledSourceBindingPlan
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.step_dependencies import (
    StepInputDependency,
    StepInputDependencyKind,
)
from openhcs.core.steps.function_step import FunctionStep


def test_runtime_axis_scope_returns_exact_complete_plane_selection() -> None:
    scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value="1",
    )

    assert scope.matching_component_plane_indices(
        (
            {"site": "1", "channel": "1"},
            {"site": "2", "channel": "1"},
        )
    ) == (0, 1)


def test_runtime_axis_scope_returns_none_only_without_declared_component_axis() -> None:
    scope = RuntimeExecutionAxisScope.from_raw(
        "A01",
        component=AllComponents.CHANNEL,
        value="1",
    )

    assert (
        scope.matching_component_plane_indices(({"site": "1"}, {"site": "2"})) is None
    )


def test_function_step_execution_scope_is_exact_closed_enum() -> None:
    assert [(member.name, member.value) for member in FunctionStepExecutionScope] == [
        ("AXIS", "axis"),
        ("PLATE", "plate"),
    ]


def test_compiled_step_plan_does_not_store_execution_scope() -> None:
    assert "execution_scope" not in {field.name for field in fields(CompiledStepPlan)}


def test_plate_dependency_has_no_main_flow() -> None:
    assert StepInputDependencyKind.NO_MAIN_FLOW.value == "no_main_flow"
    dependency = StepInputDependency.no_main_flow()

    assert dependency.kind is StepInputDependencyKind.NO_MAIN_FLOW
    assert dependency.source_step_index is None
    assert dependency.source_step_scope_id is None
    assert dependency.is_resolved


def test_plate_scope_drives_no_main_flow_paths() -> None:
    planner = PathPlanner.__new__(PathPlanner)
    planner.plans = {
        1: CompiledStepPlan(
            step_index=1,
            step_name="export",
            step_type="FunctionStep",
            axis_id="A01",
        )
    }
    output_dir = Path("/data/plate_processed/images")
    planner.paths = SimpleNamespace(build_output_path=lambda: output_dir)
    planner.steps = PathPlannerStepAssemblyStage(planner)
    snapshot = StepSnapshot(
        index=1,
        scope_id="export",
        step=FunctionStep(func=lambda image: image, name="export"),
    )

    dependency = planner.steps.main_input_dependency(
        snapshot,
        1,
        execution_scope=FunctionStepExecutionScope.PLATE,
    )
    input_dir, resolved_output_dir = planner.steps.step_io_dirs(dependency, 1)

    assert dependency == StepInputDependency.no_main_flow()
    assert input_dir == output_dir
    assert resolved_output_dir == output_dir


def test_path_validation_accepts_explicit_no_main_flow_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = PathPlanner.__new__(PathPlanner)
    planner.session = SimpleNamespace(
        step_count=2,
        snapshot=lambda index: SimpleNamespace(name=("axis", "export")[index]),
    )
    planner.plans = {
        1: CompiledStepPlan(
            step_index=1,
            step_name="export",
            step_type="FunctionStep",
            axis_id="A01",
            main_input_dependency=StepInputDependency.no_main_flow(),
        )
    }
    validation = PathPlannerValidationStage(planner)
    monkeypatch.setattr(
        PathPlannerValidationStage,
        "validate_materialization_paths",
        lambda _self: None,
    )

    validation.validate()


def test_runtime_artifact_batch_is_exact_immutable_contract_selection() -> None:
    assert tuple(field.name for field in fields(RuntimeArtifactBatch)) == (
        "input_specs",
        "records_by_axis",
        "source_image_set_identity_policy",
        "source_binding_plan",
    )

    spec = ArtifactSpec.input("Measurements", MeasurementsArtifactType)
    batch = RuntimeArtifactBatch(
        input_specs=(spec,),
        records_by_axis={"A01": ()},
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(frozenset()),
    )

    assert batch.input_specs == (spec,)
    assert batch.source_binding_plan.is_empty
    assert isinstance(batch.records_by_axis, MappingProxyType)
    assert batch.require_parameter_name() == "artifact_batch"
    parameter = batch.parameter()
    assert parameter.name == "artifact_batch"
    assert parameter.kind.name == "KEYWORD_ONLY"
    assert parameter.default is parameter.empty
    assert batch.specs_of_type(MeasurementsArtifactType) == (spec,)
    assert dict(batch.records(spec.ref())) == {"A01": ()}
    assert dict(batch.records_of_type(MeasurementsArtifactType)) == {"A01": ()}

    with pytest.raises(TypeError, match="source_binding_plan"):
        RuntimeArtifactBatch(
            input_specs=(spec,),
            records_by_axis={"A01": ()},
            source_image_set_identity_policy=SourceImageSetIdentityPolicy(),
            source_binding_plan=object(),
        )

    with pytest.raises((KeyError, ValueError), match="Undeclared|declared"):
        batch.records(ArtifactSpec.input("Other", MeasurementsArtifactType).ref())


def test_mixed_axis_and_plate_pattern_fails_before_path_planning() -> None:
    def axis_callable(image):
        return image

    @execution_scope(FunctionStepExecutionScope.PLATE)
    def plate_callable(*, artifact_batch):
        return artifact_batch

    with pytest.raises(ValueError, match="mixed|uniform|scope"):
        resolve_function_pattern_execution_scope(
            [axis_callable, plate_callable],
            CompositeInvocationContractProvider(()),
            ArtifactDeclarationStepContext(step_index=0),
        )


def test_compiled_pattern_derives_uniform_scope_from_callable_contracts() -> None:
    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    def first(*, artifact_batch):
        return artifact_batch

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    def second(*, artifact_batch):
        return artifact_batch

    compiled = compile_function_pattern(
        [first, second],
        {},
        {},
    )

    assert compiled.execution_scope is FunctionStepExecutionScope.PLATE


def test_plate_scope_preserves_exact_callable_artifact_declarations() -> None:
    source_input = ArtifactSpec.input("Source", ImageArtifactType)
    main_flow_input = ArtifactSpec.input("MainFlow", ImageArtifactType)
    runtime_input = ArtifactSpec.input("Runtime", MeasurementsArtifactType)
    recorded_output = ArtifactSpec.output("Recorded", MeasurementsArtifactType)
    main_flow_output = ArtifactSpec.output("MainFlowOutput", ImageArtifactType)
    declared_output = ArtifactSpec.output("Declared", SpecialArtifactType)
    @execution_scope(FunctionStepExecutionScope.PLATE)
    @artifact_inputs(source_input, main_flow_input, runtime_input)
    @artifact_outputs(recorded_output, main_flow_output, declared_output)
    def plate_callable(*, artifact_batch: RuntimeArtifactBatch):
        return artifact_batch

    contract = CallableContract.from_callable(plate_callable)

    assert contract.execution_scope is FunctionStepExecutionScope.PLATE
    assert contract.artifact_inputs.specs == (
        source_input,
        main_flow_input,
        runtime_input,
    )
    assert contract.artifact_outputs.specs == (
        recorded_output,
        main_flow_output,
        declared_output,
    )


def _plate_step_plan(
    *,
    axis_id: str,
    step_index: int,
    func,
    artifact_inputs: tuple[ArtifactInputPlan, ...],
    artifact_output: ArtifactOutputPlan,
    metadata_writer: bool,
) -> CompiledStepPlan:
    input_plans = OrderedDict((plan.ref(), plan) for plan in artifact_inputs)
    output_plans = OrderedDict(((artifact_output.ref(), artifact_output),))
    compiled_pattern = compile_function_pattern(
        func,
        input_plans,
        output_plans,
    )
    compiled_pattern = PathPlannerArtifactStage(
        PathPlanner.__new__(PathPlanner)
    ).compile_invocation_input_edges(
        compiled_pattern,
        artifact_inputs=input_plans,
        relation_source_scopes={
            plan.ref(): plan.producer_group_scope() for plan in artifact_inputs
        },
        execution_group_scope=ComponentGroupScope.ungrouped(),
        consumer_variable_components=ComponentSet(),
    )
    return CompiledStepPlan(
        step_index=step_index,
        step_name=func.__name__,
        step_type="FunctionStep",
        axis_id=axis_id,
        func=func,
        input_dir=Path("/plate/images"),
        output_dir=Path("/plate/images"),
        output_plate_root="/plate",
        sub_dir="images",
        analysis_results_dir="/plate/analysis",
        pipeline_position=step_index + 1,
        variable_components=(),
        main_input_dependency=StepInputDependency.no_main_flow(),
        artifact_inputs=input_plans,
        artifact_outputs=output_plans,
        compiled_function_pattern=compiled_pattern,
        read_backend=Backend.MEMORY.value,
        write_backend=Backend.MEMORY.value,
        create_openhcs_metadata=metadata_writer,
    )


def _plate_context(
    axis_id: str,
    plans: tuple[CompiledStepPlan, ...],
) -> ProcessingContext:
    context = ProcessingContext(
        step_plans={plan.step_index: plan for plan in plans},
        axis_id=axis_id,
        filemanager=FileManager({Backend.MEMORY.value: MemoryStorageBackend()}),
    )
    context.microscope_handler = SimpleNamespace()
    return context


class _RecordingProgressQueue:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def put(self, event: dict) -> None:
        self.events.append(event)


def _execute_plate_steps(
    contexts: dict[str, ProcessingContext],
    *,
    heartbeat_interval_seconds: float = 2.0,
) -> list[ProgressEvent]:
    progress_queue = _RecordingProgressQueue()
    execute_plate_scoped_steps(
        contexts,
        progress_queue=progress_queue,
        progress_context=ProgressExecutionContext(
            execution_id="execution-1",
            plate_id="plate-1",
        ),
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )
    return [ProgressEvent.from_dict(event) for event in progress_queue.events]


def _record_measurements(
    context: ProcessingContext,
    *,
    name: str,
    path: str,
    count: int,
    object_name: str | None = None,
    group_component: AllComponents | None = None,
    group_key: str | None = None,
) -> None:
    output_plan = ArtifactOutputPlan(
        name=name,
        path=path,
        artifact_type=MeasurementsArtifactType,
        group_keys=(group_key,),
        group_component=group_component,
    )
    subject = (
        MeasurementSubject(MeasurementScope.ARTIFACT, name)
        if object_name is None
        else MeasurementSubject(MeasurementScope.OBJECT, object_name)
    )
    payload = MeasurementTable(
        name=name,
        rows=MeasurementSparseColumnarRows.from_rows(
            ({"image_number": 1, "count": count},),
            fields=(FieldSpec("image_number", int), FieldSpec("count", int)),
        ),
        subject=subject,
    )
    value = RuntimeValue.normalize(
        output_plan,
        payload,
        axis_id=str(context.axis_id),
    )
    location = RuntimeArtifactLocation(path, Backend.MEMORY.value)
    context.runtime_value_store.replace(
        value,
        path=location.path,
        backend=location.backend,
    )
    replace_runtime_artifact_payload(
        context.filemanager,
        value.data,
        location,
    )


def _record_image(
    context: ProcessingContext,
    *,
    name: str,
    path: str,
) -> None:
    output_plan = ArtifactOutputPlan(
        name=name,
        path=path,
        artifact_type=ImageArtifactType,
    )
    value = RuntimeValue.normalize(
        output_plan,
        np.zeros((2, 2), dtype=np.uint8),
        axis_id=str(context.axis_id),
    )
    location = RuntimeArtifactLocation(path, Backend.MEMORY.value)
    context.runtime_value_store.replace(
        value,
        path=location.path,
        backend=location.backend,
    )
    replace_runtime_artifact_payload(
        context.filemanager,
        value.data,
        location,
    )


def test_plate_artifact_batch_keeps_optional_source_declaration_without_record() -> None:
    measurement_spec = ArtifactSpec.input(
        "Measurements",
        MeasurementsArtifactType,
    )
    source_spec = ArtifactSpec.input(
        "InputImage",
        ImageArtifactType,
        required=False,
    )
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(source_spec, measurement_spec)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"Image.csv": b"ImageNumber,Count\n"}

    measurement_path = "/memory/A01/measurements"
    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(
            ArtifactInputPlan(
                name=measurement_spec.name,
                path=measurement_path,
                artifact_type=measurement_spec.artifact_type,
            ),
        ),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/A01/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )
    context = _plate_context("A01", (plan,))
    _record_measurements(
        context,
        name=measurement_spec.name,
        path=measurement_path,
        count=2,
    )
    records = tuple(context.runtime_value_store.values())
    assert {record.key.artifact_type for record in records} == {
        MeasurementsArtifactType,
    }
    batch = _plate_artifact_batch(
        compiled_contexts={"A01": context},
        step_index=1,
        invocation_position=0,
        contract=CallableContract.from_callable(export),
        records_by_axis={"A01": records},
        source_binding_plan=plan.source_binding_plan,
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(frozenset()),
    )

    assert batch.input_specs == (source_spec, measurement_spec)
    assert batch.source_binding_plan is plan.source_binding_plan
    assert len(batch.records(measurement_spec.ref())["A01"]) == 1
    assert batch.records(source_spec.ref())["A01"] == ()


def test_plate_scope_runs_once_from_exact_contract_selected_records() -> None:
    measurement_spec = ArtifactSpec.input(
        "Measurements",
        MeasurementsArtifactType,
    )
    source_spec = ArtifactSpec.input(
        "InputImage",
        ImageArtifactType,
        required=False,
    )
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)
    batches: list[RuntimeArtifactBatch] = []
    received_contexts: list[ProcessingContext] = []

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(source_spec, measurement_spec)
    @artifact_outputs(output_spec)
    def export(
        *,
        artifact_batch: RuntimeArtifactBatch,
        context: ProcessingContext,
    ):
        batches.append(artifact_batch)
        received_contexts.append(context)
        return {"Image.csv": b"ImageNumber,Count\n"}

    contexts = {}
    for axis_id, count in (("A01", 2), ("B01", 3)):
        input_path = f"/memory/{axis_id}/measurements"
        plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=1,
            func=export,
            artifact_inputs=(
                ArtifactInputPlan(
                    name=measurement_spec.name,
                    path=input_path,
                    artifact_type=measurement_spec.artifact_type,
                ),
            ),
            artifact_output=ArtifactOutputPlan(
                name=output_spec.name,
                path="/memory/plate/export",
                artifact_type=output_spec.artifact_type,
            ),
            metadata_writer=axis_id == "A01",
        )
        context = _plate_context(axis_id, (plan,))
        _record_measurements(
            context,
            name=measurement_spec.name,
            path=input_path,
            count=count,
        )
        contexts[axis_id] = context

    _execute_plate_steps(contexts)

    assert len(batches) == 1
    assert received_contexts == [contexts["A01"]]
    batch = batches[0]
    assert batch.input_specs == (source_spec, measurement_spec)
    assert (
        sum(len(records) for records in batch.records(measurement_spec.ref()).values())
        == 2
    )
    assert sum(len(records) for records in batch.records(source_spec.ref()).values()) == 0
    assert (
        len(
            contexts["A01"].runtime_value_store.find(
                name=output_spec.name,
                artifact_type=SpecialArtifactType,
            )
        )
        == 1
    )
    assert not contexts["B01"].runtime_value_store.find(
        name=output_spec.name,
        artifact_type=SpecialArtifactType,
    )


def test_plate_scope_progress_stays_live_until_materialization_completes() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        time.sleep(0.02)
        return {"export.txt": b"data"}

    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/plate/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )

    events = _execute_plate_steps(
        {"A01": _plate_context("A01", (plan,))},
        heartbeat_interval_seconds=0.001,
    )

    assert events[0].phase is ProgressPhase.STEP_STARTED
    assert events[-1].phase is ProgressPhase.STEP_COMPLETED
    assert any(event.phase is ProgressPhase.RUNNING for event in events[1:-1])
    assert all(event.axis_id == "" for event in events)
    assert all(event.step_name == "export" for event in events)
    assert events[0].completed == 1
    assert events[0].total == 2
    assert events[-1].completed == events[-1].total == 2


def test_plate_scope_observation_excludes_preexisting_runtime_history() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"export.txt": b"data"}

    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/plate/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )
    context = _plate_context("A01", (plan,))
    _record_measurements(
        context,
        name="prior_measurements",
        path="/memory/prior_measurements",
        count=99,
    )
    progress_queue = _RecordingProgressQueue()

    observation = execute_plate_scoped_steps(
        {"A01": context},
        progress_queue=progress_queue,
        progress_context=ProgressExecutionContext(
            execution_id="execution-1",
            plate_id="plate-1",
        ),
    )

    assert len(observation.contexts) == 1
    assert {
        record.key.name for record in observation.contexts[0].records
    } == {output_spec.name}
    events = [ProgressEvent.from_dict(event) for event in progress_queue.events]
    assert events[-1].phase is ProgressPhase.STEP_COMPLETED


def test_plate_scope_image_set_policy_includes_compiled_group_component() -> None:
    measurement_spec = ArtifactSpec.input(
        "Measurements",
        MeasurementsArtifactType,
    )
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)
    batches: list[RuntimeArtifactBatch] = []

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(measurement_spec)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        batches.append(artifact_batch)
        return {"Image.csv": b"ImageNumber,Count\n"}

    input_path = "/memory/A01/measurements"
    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(
            ArtifactInputPlan(
                name=measurement_spec.name,
                path=input_path,
                artifact_type=measurement_spec.artifact_type,
            ),
        ),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/A01/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )
    context = _plate_context("A01", (plan,))
    context.source_image_set_identity_policy = SourceImageSetIdentityPolicy(
        frozenset((AllComponents.CHANNEL,))
    )
    _record_measurements(
        context,
        name=measurement_spec.name,
        path=input_path,
        count=1,
    )

    _execute_plate_steps({"A01": context})

    assert len(batches) == 1
    assert batches[0].source_image_set_identity_policy.plane_member_components == (
        frozenset((AllComponents.CHANNEL,))
    )


def test_plate_scope_collects_runtime_discovered_component_groups() -> None:
    measurement_spec = ArtifactSpec.input(
        "Measurements",
        MeasurementsArtifactType,
    )
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)
    batches: list[RuntimeArtifactBatch] = []

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(measurement_spec)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        batches.append(artifact_batch)
        return {"Image.csv": b"ImageNumber,Count\n"}

    input_path = "/memory/A01/measurements"
    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(
            ArtifactInputPlan(
                name=measurement_spec.name,
                path=input_path,
                artifact_type=measurement_spec.artifact_type,
                group_component=AllComponents.CHANNEL,
                paths_by_group={None: input_path},
            ),
        ),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/A01/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )
    context = _plate_context("A01", (plan,))
    for group_key in ("1", "2"):
        _record_measurements(
            context,
            name=measurement_spec.name,
            path=f"/memory/A01/measurements_w{group_key}",
            count=int(group_key),
            group_component=AllComponents.CHANNEL,
            group_key=group_key,
        )

    _execute_plate_steps({"A01": context})

    assert len(batches) == 1
    selected = batches[0].records(measurement_spec.ref())["A01"]
    assert tuple(record.key.scope.value_text for record in selected) == ("1", "2")


def test_plate_scope_batch_preserves_semantic_variants_for_one_input() -> None:
    measurement_spec = ArtifactSpec.input(
        "Measurements",
        MeasurementsArtifactType,
    )
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)
    batches: list[RuntimeArtifactBatch] = []

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(measurement_spec)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        batches.append(artifact_batch)
        return {"Image.csv": b"ImageNumber,Count\n"}

    input_path = "/memory/A01/measurements"
    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(
            ArtifactInputPlan(
                name=measurement_spec.name,
                path=input_path,
                artifact_type=measurement_spec.artifact_type,
            ),
        ),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/A01/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )
    context = _plate_context("A01", (plan,))
    _record_measurements(
        context,
        name=measurement_spec.name,
        path=input_path,
        count=2,
        object_name="Nuclei",
    )
    _record_measurements(
        context,
        name=measurement_spec.name,
        path=input_path,
        count=3,
        object_name="Cells",
    )

    _execute_plate_steps({"A01": context})

    assert len(batches) == 1
    selected = batches[0].records(measurement_spec.ref())["A01"]
    assert len(selected) == 2
    tables = tuple(record.value.data for record in selected)
    assert all(isinstance(table, MeasurementTable) for table in tables)
    assert {table.subject.object_name for table in tables} == {
        "Nuclei",
        "Cells",
    }


def test_later_plate_step_consumes_prior_plate_output() -> None:
    first_output = ArtifactSpec.output("FirstBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(first_output)
    def first_export(*, artifact_batch: RuntimeArtifactBatch):
        assert not artifact_batch.input_specs
        return {"first.txt": b"first"}

    prior_input = ArtifactSpec.input("FirstBundle", SpecialArtifactType)
    second_output = ArtifactSpec.output("SecondBundle", SpecialArtifactType)
    consumed: list[int] = []

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_inputs(prior_input)
    @artifact_outputs(second_output)
    def second_export(*, artifact_batch: RuntimeArtifactBatch):
        record_count = sum(
            len(records)
            for records in artifact_batch.records(prior_input.ref()).values()
        )
        consumed.append(record_count)
        return {"second.txt": str(record_count)}

    contexts = {}
    for axis_id in ("A01", "B01"):
        first_path = "/memory/plate/first"
        first_plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=1,
            func=first_export,
            artifact_inputs=(),
            artifact_output=ArtifactOutputPlan(
                name=first_output.name,
                path=first_path,
                artifact_type=SpecialArtifactType,
            ),
            metadata_writer=axis_id == "A01",
        )
        second_plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=2,
            func=second_export,
            artifact_inputs=(
                ArtifactInputPlan(
                    name=prior_input.name,
                    path=first_path,
                    artifact_type=SpecialArtifactType,
                ),
            ),
            artifact_output=ArtifactOutputPlan(
                name=second_output.name,
                path="/memory/plate/second",
                artifact_type=SpecialArtifactType,
            ),
            metadata_writer=axis_id == "A01",
        )
        contexts[axis_id] = _plate_context(axis_id, (first_plan, second_plan))

    _execute_plate_steps(contexts)

    assert consumed == [1]
    assert (
        len(
            contexts["A01"].runtime_value_store.find(
                name=second_output.name,
                artifact_type=SpecialArtifactType,
            )
        )
        == 1
    )


def test_plate_batches_carry_each_invocation_owners_source_binding_plan() -> None:
    first_output = ArtifactSpec.output("FirstBundle", SpecialArtifactType)
    second_output = ArtifactSpec.output("SecondBundle", SpecialArtifactType)
    received_plans: list[CompiledSourceBindingPlan] = []

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(first_output)
    def first_export(*, artifact_batch: RuntimeArtifactBatch):
        received_plans.append(artifact_batch.source_binding_plan)
        return {"first.txt": b"first"}

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(second_output)
    def second_export(*, artifact_batch: RuntimeArtifactBatch):
        received_plans.append(artifact_batch.source_binding_plan)
        return {"second.txt": b"second"}

    first_plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=first_export,
        artifact_inputs=(),
        artifact_output=ArtifactOutputPlan(
            name=first_output.name,
            path="/memory/A01/first",
            artifact_type=first_output.artifact_type,
        ),
        metadata_writer=True,
    )
    second_plan = _plate_step_plan(
        axis_id="A01",
        step_index=2,
        func=second_export,
        artifact_inputs=(),
        artifact_output=ArtifactOutputPlan(
            name=second_output.name,
            path="/memory/A01/second",
            artifact_type=second_output.artifact_type,
        ),
        metadata_writer=True,
    )
    first_source_plan = CompiledSourceBindingPlan(
        metadata_fields=(FieldSpec("FirstPlate", str, required=False),),
    )
    second_source_plan = CompiledSourceBindingPlan(
        metadata_fields=(FieldSpec("SecondPlate", str, required=False),),
    )
    first_plan.source_binding_plan = first_source_plan
    second_plan.source_binding_plan = second_source_plan

    _execute_plate_steps(
        {"A01": _plate_context("A01", (first_plan, second_plan))}
    )

    assert received_plans == [first_source_plan, second_source_plan]


def test_plate_scope_rejects_context_invocation_drift() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch, prefix: str = ""):
        del artifact_batch
        return {f"{prefix}export.txt": b"data"}

    contexts = {}
    for axis_id, prefix in (("A01", "a"), ("B01", "b")):
        output_plan = ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/plate/export",
            artifact_type=SpecialArtifactType,
        )
        compiled = compile_function_pattern(
            (export, {"prefix": prefix}),
            {},
            {plan.ref(): plan for plan in (output_plan,)},
        )
        plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=1,
            func=export,
            artifact_inputs=(),
            artifact_output=output_plan,
            metadata_writer=axis_id == "A01",
        )
        plan.compiled_function_pattern = compiled
        contexts[axis_id] = _plate_context(axis_id, (plan,))

    with pytest.raises(ValueError, match="drifted"):
        validate_plate_scoped_contexts(contexts)


def test_plate_scope_rejects_context_output_plan_drift() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"export.txt": b"data"}

    contexts = {}
    for axis_id in ("A01", "B01"):
        output_plan = ArtifactOutputPlan(
            name=output_spec.name,
            path=f"/memory/{axis_id}/export",
            artifact_type=SpecialArtifactType,
        )
        plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=1,
            func=export,
            artifact_inputs=(),
            artifact_output=output_plan,
            metadata_writer=axis_id == "A01",
        )
        plan.compiled_function_pattern = compile_function_pattern(
            export,
            {},
            {plan.ref(): plan for plan in (output_plan,)},
        )
        contexts[axis_id] = _plate_context(axis_id, (plan,))

    with pytest.raises(ValueError, match="compiled invocations drifted"):
        validate_plate_scoped_contexts(contexts)


def test_plate_scope_rejects_mismatched_output_plan_key_before_invocation() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)
    calls = 0

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        nonlocal calls
        del artifact_batch
        calls += 1
        return {"export.txt": b"data"}

    output_plan = ArtifactOutputPlan(
        name=output_spec.name,
        path="/memory/A01/export",
        artifact_type=output_spec.artifact_type,
    )
    plan = _plate_step_plan(
        axis_id="A01",
        step_index=1,
        func=export,
        artifact_inputs=(),
        artifact_output=output_plan,
        metadata_writer=True,
    )
    mismatched_ref = ArtifactSpec.output(
        "DifferentExportBundle",
        SpecialArtifactType,
    ).ref()
    plan.artifact_outputs[mismatched_ref] = replace(
        output_plan,
        path="/memory/A01/wrong-export",
    )

    with pytest.raises(ValueError, match="received output plan key"):
        _execute_plate_steps({"A01": _plate_context("A01", (plan,))})

    assert calls == 0


def test_plate_scope_rejects_membership_present_only_after_first_context() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"export.txt": b"data"}

    plate_plan = _plate_step_plan(
        axis_id="B01",
        step_index=1,
        func=export,
        artifact_inputs=(),
        artifact_output=ArtifactOutputPlan(
            name=output_spec.name,
            path="/memory/B01/export",
            artifact_type=output_spec.artifact_type,
        ),
        metadata_writer=True,
    )
    contexts = {
        "A01": _plate_context("A01", ()),
        "B01": _plate_context("B01", (plate_plan,)),
    }

    with pytest.raises(ValueError, match="membership drifted"):
        validate_plate_scoped_contexts(contexts)


def test_plate_scope_rejects_source_binding_plan_drift() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"export.txt": b"data"}

    contexts = {}
    for axis_id in ("A01", "B01"):
        plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=1,
            func=export,
            artifact_inputs=(),
            artifact_output=ArtifactOutputPlan(
                name=output_spec.name,
                path="/memory/plate/export",
                artifact_type=output_spec.artifact_type,
            ),
            metadata_writer=axis_id == "A01",
        )
        if axis_id == "B01":
            plan.source_binding_plan = CompiledSourceBindingPlan(enabled=True)
        contexts[axis_id] = _plate_context(axis_id, (plan,))

    with pytest.raises(ValueError, match="source-binding plan drifted"):
        validate_plate_scoped_contexts(contexts)


def test_plate_scope_compares_complete_nominal_invocation_contract() -> None:
    output_spec = ArtifactSpec.output("ExportBundle", SpecialArtifactType)

    @execution_scope(FunctionStepExecutionScope.PLATE)
    @runtime_bound_parameters(RuntimeArtifactBatch)
    @artifact_outputs(output_spec)
    def export(*, artifact_batch: RuntimeArtifactBatch):
        del artifact_batch
        return {"export.txt": b"data"}

    def drifted_executor(request):
        return request

    contexts = {}
    for axis_id in ("A01", "B01"):
        plan = _plate_step_plan(
            axis_id=axis_id,
            step_index=1,
            func=export,
            artifact_inputs=(),
            artifact_output=ArtifactOutputPlan(
                name=output_spec.name,
                path="/memory/plate/export",
                artifact_type=output_spec.artifact_type,
            ),
            metadata_writer=axis_id == "A01",
        )
        if axis_id == "B01":
            pattern = plan.compiled_function_pattern
            group = pattern.default_group
            invocation = group.invocations[0]
            drifted_invocation = replace(
                invocation,
                contract=replace(
                    invocation.contract,
                    runtime_batch_executors=MappingProxyType(
                        {
                            RuntimeBatchExecutionDomain.MEASUREMENT_IMAGES: (
                                drifted_executor
                            )
                        }
                    ),
                ),
            )
            plan.compiled_function_pattern = replace(
                pattern,
                groups=(replace(group, invocations=(drifted_invocation,)),),
            )
        contexts[axis_id] = _plate_context(axis_id, (plan,))

    with pytest.raises(ValueError, match="compiled invocations drifted"):
        validate_plate_scoped_contexts(contexts)
