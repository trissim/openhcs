"""Compiled artifact inspection transport and UI projection tests."""

from __future__ import annotations

import os
import pickle
from collections import OrderedDict

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from openhcs.core.artifact_inspection import (
    CompiledArtifactInspection,
    CompiledArtifactInspectionControlPayload,
    CompiledArtifactInspectionRequest,
    CompiledArtifactInspectionResponse,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactInputProjectionPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.component_group_scope import (
    ComponentGroupScope,
    RuntimeExecutionAxisScope,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.debug import DebugArtifactRef, DebugCursor, DebugSnapshot
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    CompiledFunctionGroup,
    CompiledFunctionInvocation,
    CompiledFunctionPattern,
    FunctionInvocationKey,
    InvocationArtifactInputEdgePlan,
    InvocationArtifactInputProjectionKey,
)
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.progress import (
    ProgressEvent,
    ProgressIdentity,
    ProgressPhase,
    ProgressStatus,
)
from openhcs.core.progress.runtime_artifacts import RuntimeArtifactProgressPayload
from openhcs.core.runtime_artifact_values import ArtifactKey
from openhcs.core.runtime_stores import RuntimeArtifactAddress, RuntimeArtifactLocation
from openhcs.pyqt_gui.widgets.artifact_plan_view import (
    ArtifactPlanViewModel,
    ArtifactPlanViewWidget,
)
from openhcs.pyqt_gui.widgets.shared.services.runtime_artifact_progress_service import (
    RuntimeArtifactAvailableNotification,
)
from openhcs.runtime.zmq_compilation import (
    ZMQCompilationResult,
    ZMQCompileArtifactRecord,
)
from openhcs.runtime.zmq_control import (
    ZMQControlMessageRouter,
    ZMQControlRequestContext,
)


class QtApplication:
    app_instance: QApplication | None = None

    @classmethod
    def app(cls) -> QApplication:
        cls.app_instance = QApplication.instance() or QApplication([])
        return cls.app_instance


def _compiled_fixture() -> tuple[
    CompiledArtifactInspection,
    ArtifactOutputPlan,
    CompiledExecutionBundle,
]:
    input_spec = ArtifactSpec.input(
        "InputLabels",
        ObjectLabelsArtifactType,
        parameter_name="labels",
    )
    output_spec = ArtifactSpec.output("ResultImage", ImageArtifactType)

    @artifact_outputs(output_spec)
    @artifact_inputs(input_spec)
    def transform(image, labels):
        del labels
        return image

    input_plan = ArtifactInputPlan(
        name=input_spec.name,
        path="/memory/InputLabels.pkl",
        artifact_type=input_spec.artifact_type,
        source_step_id=0,
    )
    output_plan = ArtifactOutputPlan(
        name=output_spec.name,
        path="/memory/ResultImage.pkl",
        artifact_type=output_spec.artifact_type,
        producer_step_index=1,
        producer_step_name="Transform",
    )
    key = FunctionInvocationKey("transform", DEFAULT_GROUP_KEY, 0)
    edge = InvocationArtifactInputEdgePlan(
        key=InvocationArtifactInputProjectionKey(key, 0),
        spec=input_spec,
        storage_plan=input_plan,
        projection=ArtifactInputProjectionPlan(
            invocation_scope=ComponentGroupScope.ungrouped(),
            producer_selection_scope=ComponentGroupScope.ungrouped(),
        ),
    )
    invocation = CompiledFunctionInvocation(
        key=key,
        contract=CallableContract.from_callable(transform),
        artifact_input_edges=(edge,),
        artifact_output_plans=(output_plan,),
    )
    pattern = CompiledFunctionPattern(
        groups=(
            CompiledFunctionGroup(
                group_key=DEFAULT_GROUP_KEY,
                invocations=(invocation,),
            ),
        ),
        is_grouped=False,
    )
    step_plan = CompiledStepPlan(
        step_index=1,
        step_name="Transform",
        step_type="FunctionStep",
        axis_id="A01",
        artifact_inputs=OrderedDict(((input_plan.ref(), input_plan),)),
        artifact_outputs=OrderedDict(((output_plan.ref(), output_plan),)),
        compiled_function_pattern=pattern,
    )
    context = ProcessingContext(step_plans={1: step_plan}, axis_id="A01")
    bundle = CompiledExecutionBundle(
        pipeline_definition=(),
        runtime_contexts={"A01": context},
        transport_contexts={"A01": context},
        worker_assignments={},
        runtime_environment=object(),
    )
    inspection = CompiledArtifactInspection.from_execution_bundle(
        compile_artifact_id="compile-1",
        plate_id="/plates/one",
        bundle=bundle,
    )
    return inspection, output_plan, bundle


def _progress_event() -> ProgressEvent:
    return ProgressEvent(
        identity=ProgressIdentity(
            execution_id="run-1",
            plate_id="/plates/one",
            axis_id="A01",
            step_name="Transform",
        ),
        phase=ProgressPhase.STEP_COMPLETED,
        status=ProgressStatus.SUCCESS,
        percent=100.0,
        completed=1,
        total=1,
        timestamp=1.0,
        pid=1,
    )


def _runtime_notification(
    output_plan: ArtifactOutputPlan,
) -> RuntimeArtifactAvailableNotification:
    address = RuntimeArtifactAddress(
        key=ArtifactKey(
            name=output_plan.name,
            artifact_type=output_plan.artifact_type,
            scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
        ),
        location=RuntimeArtifactLocation(
            path=output_plan.path,
            backend="memory",
        ),
        value_type="ndarray",
    )
    return RuntimeArtifactAvailableNotification(
        event=_progress_event(),
        payload=RuntimeArtifactProgressPayload((address,)),
    )


def test_compiled_inspection_preserves_exact_contract_edges_and_plans() -> None:
    inspection, output_plan, _bundle = _compiled_fixture()

    step = inspection.steps_for_index(1)[0]

    assert step.axis_id == "A01"
    assert step.artifact_outputs == (output_plan,)
    assert step.invocations[0].input_edges[0].spec.name == "InputLabels"
    assert step.invocations[0].output_plans == (output_plan,)
    assert pickle.loads(pickle.dumps(inspection)) == inspection


def test_registered_control_router_reads_retained_compile_artifact() -> None:
    expected, _output_plan, bundle = _compiled_fixture()
    record = ZMQCompileArtifactRecord(
        execution_id="compile-1",
        plate_id="/plates/one",
        request_signature="request",
        debug_replay_signature="debug",
        compilation=ZMQCompilationResult(
            execution_bundle=bundle,
            compiled_axis_ids=["A01"],
        ),
    )
    request = CompiledArtifactInspectionRequest("compile-1")
    message = CompiledArtifactInspectionControlPayload.from_request(request).to_dict()

    response = ZMQControlMessageRouter.handle(
        message,
        ZMQControlRequestContext(compiled_artifacts={"compile-1": record}),
    )

    assert CompiledArtifactInspectionResponse.from_control_response(
        response
    ).inspection == expected


def test_view_model_uses_compiled_rows_and_exact_runtime_address_matching() -> None:
    inspection, output_plan, _bundle = _compiled_fixture()
    model = ArtifactPlanViewModel.from_inspection(inspection, step_index=1)

    enriched = model.with_runtime_notification(_runtime_notification(output_plan))

    assert tuple(row.artifact_name for row in model.rows) == (
        "InputLabels",
        "ResultImage",
    )
    output_row = enriched.rows[1]
    assert output_row.planned_path == "/memory/ResultImage.pkl"
    assert output_row.runtime_location == "memory:/memory/ResultImage.pkl"
    assert output_row.value_type == "ndarray"


def test_view_model_enriches_from_debug_identity_not_presentation_strings() -> None:
    inspection, output_plan, _bundle = _compiled_fixture()
    model = ArtifactPlanViewModel.from_inspection(inspection, step_index=1)
    cursor = DebugCursor(
        step_index=1,
        step_scope_id=None,
        group_key=DEFAULT_GROUP_KEY,
        invocation_key="default:0:transform",
    )
    ref = DebugArtifactRef.from_artifact_plan(plan=output_plan, cursor=cursor)
    snapshot = DebugSnapshot(
        snapshot_id="snapshot-1",
        cursor=cursor,
        step_name="Transform",
        callable_name="renamed presentation text",
        axis_id="A01",
        output_artifact_refs=(ref,),
    )

    enriched = model.with_debug_snapshot(snapshot)

    assert enriched.rows[1].runtime_location == "memory:/memory/ResultImage.pkl"


def test_widget_renders_compiled_static_and_runtime_columns() -> None:
    QtApplication.app()
    inspection, output_plan, _bundle = _compiled_fixture()
    widget = ArtifactPlanViewWidget(inspection=inspection, step_index=1)

    widget.apply_runtime_notification(_runtime_notification(output_plan))

    assert widget._table.rowCount() == 2
    assert widget._table.item(1, 4).text() == "ResultImage"
    assert widget._table.item(1, 7).text() == "/memory/ResultImage.pkl"
    assert widget._table.item(1, 8).text() == "memory:/memory/ResultImage.pkl"
    assert widget._message.text() == "Compiled artifact plan: step 2"


def test_declaration_only_preview_and_debug_only_router_are_deleted() -> None:
    from pathlib import Path

    root = Path(__file__).parents[2]
    assert not (
        root / "openhcs/pyqt_gui/widgets/artifact_contract_preview.py"
    ).exists()
    assert not (root / "openhcs/runtime/zmq_debug_control.py").exists()
