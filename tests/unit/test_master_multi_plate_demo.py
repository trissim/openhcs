from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from polystore.streaming.identity import StreamProducerIdentity

from openhcs.core.artifacts import ImageArtifactType, MeasurementsArtifactType
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.presets.demo_contribution import PipelineDemoContribution
from openhcs.ui.shared.plate_manager_code_document import (
    PlateManagerCodeDocumentAuthority,
)
from scripts import master_multi_plate_demo as master
from scripts import mcp_assay_showcase as showcase


def test_plate_manager_projects_the_current_scope_color_owner(monkeypatch):
    from openhcs.pyqt_gui.services import plate_manager_state_projection as projection

    class _Color:
        @staticmethod
        def name():
            return "#1A2B3C"

    class _Service:
        @staticmethod
        def get_accent_color(scope_id):
            assert scope_id == "/tmp/plate"
            return _Color()

    monkeypatch.setattr(
        projection.ScopeColorService,
        "instance",
        lambda: _Service(),
    )

    assert (
        projection.PlateManagerStateProjectionService.scope_accent_color("/tmp/plate")
        == "#1a2b3c"
    )


def test_master_composes_every_declared_showcase_without_copying_pipelines(tmp_path):
    definitions = master.built_in_demo_definitions(tmp_path)
    expected_ids = tuple(
        blueprint.scenario_id for blueprint in showcase.scenario_blueprints()
    )
    expected_titles = tuple(
        blueprint.title for blueprint in showcase.scenario_blueprints()
    )

    assert (
        tuple(definition.contribution.demo_id for definition in definitions)
        == expected_ids
    )
    assert tuple(
        definition.contribution.plate_path.name for definition in definitions
    ) == (tuple(master.demo_plate_directory_name(title) for title in expected_titles))

    schedule = master.build_demo_schedule(definitions, base_port=5900)
    assert tuple(item.port for item in schedule) == tuple(
        range(5900, 5900 + len(expected_ids))
    )

    payload = PlateManagerCodeDocumentAuthority.from_source(
        master.render_master_document(schedule)
    )
    assert payload.plate_paths == tuple(
        str(item.definition.contribution.plate_path) for item in schedule
    )
    for item in schedule:
        steps = payload.pipeline_data[str(item.definition.contribution.plate_path)]
        streamed_ports = {
            step.napari_streaming_config.port
            for step in steps
            if step.napari_streaming_config.enabled
        }
        assert streamed_ports == {item.port}
        assert item.definition.contribution.presentation_identity.step_name in {
            step.name for step in steps if step.napari_streaming_config.enabled
        }


def test_master_excludes_only_explicit_known_demo_ids(tmp_path):
    definitions = master.built_in_demo_definitions(tmp_path)

    selected = master.exclude_demo_definitions(
        definitions,
        ("primary_object_segmentation", "nuclear_morphology"),
    )

    assert {definition.contribution.demo_id for definition in definitions} - {
        definition.contribution.demo_id for definition in selected
    } == {"primary_object_segmentation", "nuclear_morphology"}
    with pytest.raises(ValueError, match="unknown"):
        master.exclude_demo_definitions(definitions, ("not-a-demo",))


def _compiled_plan(*steps):
    return {
        "axis_count": 1,
        "step_count": len(steps),
        "steps": list(steps),
        "source_workspace": {"file_count": 1, "truncated_file_count": 0},
    }


def _definition_with_identity(
    definition: master.MasterDemoDefinition,
    identity: StreamProducerIdentity,
) -> master.MasterDemoDefinition:
    return replace(
        definition,
        contribution=replace(
            definition.contribution,
            pipeline_steps=(FunctionStep(name=identity.step_name),),
            presentation_identity=identity,
            supporting_presentation_identities=(),
        ),
    )


def test_compiled_presentation_contract_validates_visual_and_all_measurements(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    identity = StreamProducerIdentity.pipeline_output(
        output_kind="artifact",
        output_key="neighbor_count",
        projection_key="neighbor_count",
        step_name="Measure neighbors",
        pipeline_position=None,
        artifact_kind=ImageArtifactType.require_value(),
    )
    definition = _definition_with_identity(definition, identity)
    plan = _compiled_plan(
        {
            "step_index": 0,
            "step_name": "Segment",
            "viewer_streaming": [],
            "artifact_outputs": [
                {"name": "counts", "kind": "measurements"},
            ],
        },
        {
            "step_index": 1,
            "step_name": "Measure neighbors",
            "viewer_streaming": [{"viewer_type": "napari"}],
            "artifact_outputs": [
                {"name": "neighbor_count", "kind": "image"},
                {"name": "topology", "kind": "measurements"},
            ],
        },
    )

    contract = master.compiled_presentation_contract(definition, plan)

    assert contract.visual_identity.pipeline_position == 1
    assert contract.visual_identity.output_key == "neighbor_count"
    assert contract.measurement_names == ("counts", "topology")


def test_compiled_presentation_contract_rejects_missing_visual_output(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    identity = StreamProducerIdentity.pipeline_output(
        output_kind="artifact",
        output_key="missing",
        projection_key="missing",
        step_name="Measure",
        pipeline_position=None,
        artifact_kind=ImageArtifactType.require_value(),
    )
    definition = _definition_with_identity(definition, identity)

    with pytest.raises(master.MasterDemoFailure, match="declared visual"):
        master.compiled_presentation_contract(
            definition,
            _compiled_plan(
                {
                    "step_index": 0,
                    "step_name": "Measure",
                    "viewer_streaming": [{"viewer_type": "napari"}],
                    "artifact_outputs": [
                        {
                            "name": "table",
                            "kind": MeasurementsArtifactType.require_value(),
                        }
                    ],
                }
            ),
        )


def test_compiled_presentation_contract_accepts_named_image_promoted_to_main_flow(
    tmp_path,
):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    identity = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="neighbor_count",
        projection_key="main",
        step_name="Measure neighbors",
        pipeline_position=None,
        artifact_kind=ImageArtifactType.require_value(),
    )
    definition = _definition_with_identity(definition, identity)

    contract = master.compiled_presentation_contract(
        definition,
        _compiled_plan(
            {
                "step_index": 1,
                "step_name": "Measure neighbors",
                "viewer_streaming": [{"viewer_type": "napari"}],
                "artifact_outputs": [
                    {"name": "neighbor_count", "kind": "image"},
                    {"name": "neighbor_rows", "kind": "measurements"},
                ],
            }
        ),
    )

    assert contract.visual_identity == replace(identity, pipeline_position=1)


class _FakeMasterOperations:
    def __init__(self, *, failed_demo_id: str | None = None):
        self.failed_demo_id = failed_demo_id
        self.events: list[tuple[str, str | int | None]] = []

    def prepare_all(self, schedule):
        self.events.append(("prepare_all", len(schedule)))

    def register_all(self, schedule):
        self.events.append(("register_all", len(schedule)))
        return {
            str(item.definition.contribution.plate_path): f"#{item.port - 5800:06x}"
            for item in schedule
        }

    def assert_port_available(self, item):
        self.events.append(("port", item.definition.contribution.demo_id))

    def select_plate(self, item):
        self.events.append(("select", item.definition.contribution.demo_id))

    def run_workflow(self, item, workflow):
        stage = workflow.name.casefold()
        demo_id = item.definition.contribution.demo_id
        self.events.append((stage, demo_id))
        if demo_id == self.failed_demo_id and stage == "compile":
            raise RuntimeError("deliberate compile failure")

    def wait_for_viewer(self, item):
        self.events.append(("viewer_ready", item.definition.contribution.demo_id))

    def present_visual_result(self, item):
        contribution = item.definition.contribution
        self.events.append(("present", contribution.demo_id))
        return master.FinalResultPresentation(
            route_key=f"route-{contribution.demo_id}",
            title=contribution.title,
            pipeline_position=2,
            feature_row_count=7,
            selected_data_indices=(0,),
        )

    def present_measurements(self, item):
        self.events.append(("measurements", item.definition.contribution.demo_id))
        return master.MeasurementPresentation(
            surface_id="live-results",
            measurement_names=("counts",),
            row_count=7,
            populated_preview_count=1,
            previews=({"row_count": 7},),
            human_results_table={"status": "completed"},
        )


def test_master_runs_sequentially_and_isolates_one_plate_failure(tmp_path):
    definitions = master.built_in_demo_definitions(tmp_path)[:3]
    schedule = master.build_demo_schedule(definitions, base_port=5900)
    failed_id = schedule[1].definition.contribution.demo_id
    operations = _FakeMasterOperations(failed_demo_id=failed_id)

    report = master.run_demo_schedule(schedule, operations)

    assert [result.status for result in report.results] == [
        "completed",
        "failed",
        "completed",
    ]
    assert report.results[1].failed_stage == "compile"
    assert report.results[1].error == "deliberate compile failure"
    assert operations.events[:2] == [("prepare_all", 3), ("register_all", 3)]
    for item in (schedule[0], schedule[2]):
        demo_id = item.definition.contribution.demo_id
        assert (
            operations.events.index(("init", demo_id))
            < operations.events.index(("compile", demo_id))
            < operations.events.index(("run", demo_id))
        )
    assert operations.events.index(("compile", failed_id)) < operations.events.index(
        ("port", schedule[2].definition.contribution.demo_id)
    )
    for item in (schedule[0], schedule[2]):
        demo_id = item.definition.contribution.demo_id
        assert ("viewer_ready", demo_id) in operations.events
        assert ("present", demo_id) in operations.events
        assert ("measurements", demo_id) in operations.events
    assert report.results[0].final_overlay_route == (
        f"route-{schedule[0].definition.contribution.demo_id}"
    )
    assert report.results[0].feature_row_count == 7
    assert report.results[0].selected_data_indices == (0,)
    assert report.results[0].measurement_names == ("counts",)
    assert report.results[0].measurement_row_count == 7


def _pipeline_layer(
    route_key: str,
    *,
    pipeline_position: int,
    feature_row_count: int,
    pending_update: bool = False,
    selected_data_indices: list[int] | None = None,
    output_kind: str = "artifact",
    artifact_kind: str | None = "object_labels",
    output_key: str | None = None,
    step_name: str | None = None,
):
    return {
        "route_key": route_key,
        "title": route_key,
        "mounted": True,
        "visible": False,
        "selected": False,
        "pending_update": pending_update,
        "feature_row_count": feature_row_count,
        "selected_data_indices": selected_data_indices or [],
        "producer_identities": [
            {
                "origin": "pipeline",
                "output_kind": output_kind,
                "output_key": output_key or route_key,
                "projection_key": output_key or route_key,
                "step_name": step_name or f"step-{pipeline_position}",
                "pipeline_position": pipeline_position,
                "artifact_kind": artifact_kind,
            }
        ],
    }


def _viewer_contract(
    route_key: str,
    *,
    pipeline_position: int,
    output_kind: str = "artifact",
    artifact_kind: str | None = "object_labels",
):
    return master.CompiledPresentationContract(
        visual_identity=StreamProducerIdentity.pipeline_output(
            output_kind=output_kind,
            output_key=route_key,
            projection_key=route_key,
            step_name=f"step-{pipeline_position}",
            pipeline_position=pipeline_position,
            artifact_kind=artifact_kind,
        ),
        measurement_names=("table",),
    )


def test_final_presentation_resolves_exact_declared_feature_overlay():
    state = {
        "observed": True,
        "layers": [
            _pipeline_layer("older", pipeline_position=0, feature_row_count=3),
            _pipeline_layer("final", pipeline_position=2, feature_row_count=5),
        ],
    }

    presentation = master.FinalResultPresentation.from_viewer_state(
        state,
        _viewer_contract("final", pipeline_position=2),
    )

    assert presentation.route_key == "final"
    assert presentation.pipeline_position == 2
    assert presentation.feature_row_count == 5


def test_final_presentation_accepts_declared_image_without_feature_rows():
    layer = _pipeline_layer(
        "image",
        pipeline_position=2,
        feature_row_count=0,
        output_kind="main",
        artifact_kind=None,
    )

    presentation = master.FinalResultPresentation.from_viewer_state(
        {"observed": True, "layers": [layer]},
        _viewer_contract(
            "image",
            pipeline_position=2,
            output_kind="main",
            artifact_kind=None,
        ),
    )

    assert presentation.feature_row_count == 0
    assert presentation.feature_selection_applied is False


def test_final_presentation_rejects_missing_or_duplicate_declared_route():
    for layers in (
        [_pipeline_layer("other", pipeline_position=2, feature_row_count=0)],
        [
            _pipeline_layer("target", pipeline_position=2, feature_row_count=3),
            _pipeline_layer("target", pipeline_position=2, feature_row_count=4),
        ],
    ):
        with pytest.raises(master.MasterDemoFailure, match="exactly one"):
            master.FinalResultPresentation.from_viewer_state(
                {"observed": True, "layers": layers},
                _viewer_contract("target", pipeline_position=2),
            )


def test_declared_visual_layer_uses_owner_declaration_matching():
    declaration = _viewer_contract(
        "target",
        pipeline_position=2,
    ).visual_identity
    observed = replace(
        declaration,
        step_scope_id="submission::step_2",
        invocation_key="axis::A01",
    )
    decoy = replace(observed, projection_key="different_projection")
    target_layer = _pipeline_layer(
        "target-route",
        pipeline_position=2,
        feature_row_count=4,
    )
    target_layer["producer_identities"] = [observed.to_payload()]
    decoy_layer = _pipeline_layer(
        "decoy-route",
        pipeline_position=2,
        feature_row_count=4,
    )
    decoy_layer["producer_identities"] = [decoy.to_payload()]

    selected = master.declared_visual_layer(
        {
            "observed": True,
            "layers": [decoy_layer, target_layer],
        },
        declaration,
    )

    assert selected["route_key"] == "target-route"


def test_master_aborts_schedule_when_persistent_session_is_invalidated(tmp_path):
    definitions = master.built_in_demo_definitions(tmp_path)[:2]
    schedule = master.build_demo_schedule(definitions, base_port=5900)

    class _InvalidatedOperations(_FakeMasterOperations):
        def run_workflow(self, item, workflow):
            raise master.MasterDemoSessionInvalidated("source changed")

    operations = _InvalidatedOperations()

    with pytest.raises(master.MasterDemoSessionInvalidated, match="source changed"):
        master.run_demo_schedule(schedule, operations)

    assert (
        "port",
        schedule[1].definition.contribution.demo_id,
    ) not in operations.events


def test_explicit_contributor_uses_same_port_and_document_path(tmp_path):
    built_in = master.built_in_demo_definitions(tmp_path)[:1]
    base = built_in[0]
    contribution = PipelineDemoContribution(
        demo_id="downloaded_assay",
        title="Downloaded assay",
        plate_path=tmp_path / "downloaded" / "plate",
        pipeline_config=base.contribution.pipeline_config,
        pipeline_steps=base.contribution.pipeline_steps,
        presentation_identity=base.contribution.presentation_identity,
        prepare=None,
    )

    definitions = master.demo_definitions_with_contributions(
        built_in,
        (contribution,),
    )
    schedule = master.build_demo_schedule(definitions, base_port=6100)
    payload = PlateManagerCodeDocumentAuthority.from_source(
        master.render_master_document(schedule)
    )

    assert tuple(item.definition.contribution.demo_id for item in schedule) == (
        base.contribution.demo_id,
        "downloaded_assay",
    )
    assert payload.plate_paths[-1] == str(contribution.plate_path)
    assert {
        step.napari_streaming_config.port
        for step in payload.pipeline_data[str(contribution.plate_path)]
        if step.napari_streaming_config.enabled
    } == {6101}


def test_contributor_loader_requires_the_production_owner(tmp_path, monkeypatch):
    module = SimpleNamespace(
        bad_factory=lambda *, session_root: SimpleNamespace(
            demo_id=session_root.name,
        )
    )
    monkeypatch.setattr(master.importlib, "import_module", lambda _name: module)

    with pytest.raises(TypeError, match="expected PipelineDemoContribution"):
        master.load_contributions(
            ("example.module:bad_factory",),
            session_root=tmp_path,
        )


def test_schedule_rejects_duplicate_demo_and_plate_identity(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]

    duplicate_id = replace(
        definition,
        contribution=replace(
            definition.contribution,
            plate_path=tmp_path / "other",
        ),
    )
    duplicate_plate = replace(
        definition,
        contribution=replace(definition.contribution, demo_id="other"),
    )
    for definitions in ((definition, duplicate_id), (definition, duplicate_plate)):
        try:
            master.build_demo_schedule(definitions, base_port=5900)
        except ValueError as error:
            assert "duplicate" in str(error).casefold()
        else:
            raise AssertionError("duplicate demo identity was accepted")


def test_schedule_rejects_data_control_endpoint_overlap(tmp_path):
    base = master.built_in_demo_definitions(tmp_path)[0]
    definitions = tuple(
        replace(
            base,
            contribution=replace(
                base.contribution,
                demo_id=f"demo_{index}",
                plate_path=tmp_path / f"plate_{index}",
            ),
        )
        for index in range(master.OPENHCS_ZMQ_CONFIG.control_port_offset + 1)
    )

    with pytest.raises(ValueError, match="data/control endpoint ranges overlap"):
        master.build_demo_schedule(definitions, base_port=5900)


def test_artifact_plan_axis_filter_uses_execution_filter_authority(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=5900)[0]

    assert master.artifact_plan_axis_filter(item) == ("A01",)

    unbounded = replace(
        definition,
        contribution=replace(
            definition.contribution,
            pipeline_config=replace(
                definition.contribution.pipeline_config,
                well_filter_config=replace(
                    definition.contribution.pipeline_config.well_filter_config,
                    well_filter=1,
                ),
            ),
        ),
    )
    with pytest.raises(master.MasterDemoFailure, match="exact axis labels"):
        master.artifact_plan_axis_filter(
            master.build_demo_schedule((unbounded,), base_port=5900)[0]
        )


class _FakePublicMcpClient:
    def __init__(
        self,
        plate_scope_id: str,
        *,
        async_apply: bool = False,
        async_select: bool = False,
        transient_state_error: bool = False,
        transient_widget_error: bool = False,
        transient_document_error: bool = False,
        stale_workflow: bool = False,
        viewer_feature_row_count: int = 9,
        viewer_output_kind: str = "artifact",
        viewer_artifact_kind: str | None = "object_labels",
        measurement_names: tuple[str, ...] = ("table",),
        include_supporting_layer: bool = False,
        async_results: bool = False,
    ):
        self.plate_scope_id = plate_scope_id
        self.async_apply = async_apply
        self.async_select = async_select
        self.transient_state_error = transient_state_error
        self.transient_widget_error = transient_widget_error
        self.transient_document_error = transient_document_error
        self.stale_workflow = stale_workflow
        self.viewer_feature_row_count = viewer_feature_row_count
        self.viewer_output_kind = viewer_output_kind
        self.viewer_artifact_kind = viewer_artifact_kind
        self.measurement_names = measurement_names
        self.include_supporting_layer = include_supporting_layer
        self.async_results = async_results
        self.operation_wait_calls = 0
        self.apply_calls = 0
        self.state_calls = 0
        self.widget_calls = 0
        self.document_calls = 0
        self.viewer_selection_applied = False
        self.viewer_data_index_applied: int | None = None
        self.supporting_visible = False
        self.argv: list[tuple[str, ...]] = []

    @staticmethod
    def _execution(tool: str, payload: dict):
        return SimpleNamespace(
            payload={
                "errors": [],
                "results": [{"tool": tool, "mcp_error": False, "payloads": [payload]}],
            },
            returncode=0,
            server_stderr_tail=None,
        )

    @staticmethod
    def _failed_execution():
        return SimpleNamespace(
            payload={"errors": [{"code": "ui_bridge_timeout"}], "results": []},
            returncode=1,
            server_stderr_tail=None,
        )

    def execute(self, argv, *, timeout_seconds):
        assert timeout_seconds > 0
        argv = tuple(argv)
        self.argv.append(argv)
        command = argv[0]
        if command == "code-document":
            self.document_calls += 1
            if self.transient_document_error and self.document_calls == 1:
                return self._failed_execution()
            return self._execution(
                "openhcs_ui_get_code_document",
                {"current_revision_token": "revision-1"},
            )
        if command == "validate-code-document":
            return self._execution(
                "openhcs_ui_validate_code_document",
                {"valid": True},
            )
        if command == "apply-code-document":
            if self.async_apply:
                self.apply_calls += 1
                return self._execution(
                    "openhcs_ui_apply_code_document",
                    {
                        "applied": False,
                        "outcome": "running",
                        "operation_id": f"op-{self.apply_calls}",
                    },
                )
            return self._execution(
                "openhcs_ui_apply_code_document",
                {"applied": True, "outcome": "applied"},
            )
        if command == "call":
            assert argv[1] == "openhcs_ui_wait_for_operation_receipt"
            self.operation_wait_calls += 1
            arguments = json.loads(argv[argv.index("--arguments") + 1])
            operation_id = arguments["operation_id"]
            if operation_id == "select-op":
                outcome = "invoked"
            elif operation_id == "results-op":
                outcome = "accepted"
            else:
                outcome = "applied"
            return self._execution(
                "openhcs_ui_wait_for_operation_receipt",
                {
                    "status": "completed",
                    "outcome": outcome,
                },
            )
        if command == "state-surface":
            self.state_calls += 1
            if self.transient_state_error and self.state_calls == 1:
                return self._failed_execution()
            if argv[1] == "live-results":
                return self._execution(
                    "openhcs_ui_get_state_surface",
                    {
                        "payload": {
                            "entries": [
                                {
                                    "plate_id": self.plate_scope_id,
                                    "preview": {
                                        "address": {"key": {"name": name}},
                                        "row_count": 2,
                                        "columns": ["value"],
                                        "rows": [{"value": 3.0}],
                                    },
                                }
                                for name in self.measurement_names
                            ]
                        }
                    },
                )
            selection_mode = argv[argv.index("--selection-mode") + 1]
            return self._execution(
                "openhcs_ui_get_state_surface",
                {
                    "payload": {
                        "rows": [
                            {
                                "plate_scope_id": self.plate_scope_id,
                                "scope_accent_color": "#0a64c8",
                            }
                        ],
                        "selected_scope_ids": [self.plate_scope_id],
                        "selection_mode": selection_mode,
                    }
                },
            )
        if command == "actions":
            return self._execution(
                "openhcs_ui_list_actions",
                {
                    "actions": [
                        {
                            "widget_id": master.PLATE_MANAGER_WIDGET_ID,
                            "action_id": master.PlateManagerAction.VIEW_RESULTS.value,
                            "related_state_surface_ids": [
                                master.PLATE_MANAGER_STATE_SURFACE_ID,
                                "live-results",
                            ],
                        }
                    ]
                },
            )
        if command == "state-surfaces":
            return self._execution(
                "openhcs_ui_list_state_surfaces",
                {
                    "surfaces": [
                        {"surface_id": master.PLATE_MANAGER_STATE_SURFACE_ID},
                        {"surface_id": "live-results"},
                    ]
                },
            )
        if command == "invoke-action":
            return self._execution(
                "openhcs_ui_invoke_action",
                (
                    {
                        "status": "accepted",
                        "receipt": {"bridge_operation_id": "results-op"},
                    }
                    if self.async_results
                    else {"status": "completed"}
                ),
            )
        if command == "widget-tree":
            self.widget_calls += 1
            if self.transient_widget_error and self.widget_calls == 1:
                return self._failed_execution()
            return self._execution(
                "openhcs_ui_get_widget_tree",
                {
                    "actionable_widgets": [
                        {
                            "path_id": "1.1.3",
                            "object_state_scope_id": self.plate_scope_id,
                            "action_role": "item_select",
                        }
                    ]
                },
            )
        if command == "invoke-widget-action":
            return self._execution(
                "openhcs_ui_invoke_widget_action",
                (
                    {
                        "invoked": False,
                        "receipt": {
                            "accepted": True,
                            "bridge_operation_id": "select-op",
                        },
                    }
                    if self.async_select
                    else {"invoked": True}
                ),
            )
        if command == "selected-workflow":
            if self.stale_workflow:
                return self._execution(
                    "mcp_dev_selected_workflow_poll",
                    {
                        "errors": [{"code": "mcp_server_stale"}],
                        "restart_required": True,
                    },
                )
            return self._execution(
                "mcp_dev_selected_workflow_poll",
                {"poll_completed": True, "poll_status": "completed"},
            )
        if command == "viewer-state":
            layer = _pipeline_layer(
                "final-route",
                pipeline_position=3,
                feature_row_count=self.viewer_feature_row_count,
                selected_data_indices=(
                    [self.viewer_data_index_applied]
                    if self.viewer_data_index_applied is not None
                    else []
                ),
                output_kind=self.viewer_output_kind,
                artifact_kind=self.viewer_artifact_kind,
            )
            layer["visible"] = self.viewer_selection_applied
            layer["selected"] = self.viewer_selection_applied
            layers = [layer]
            if self.include_supporting_layer:
                supporting = _pipeline_layer(
                    "supporting-route",
                    pipeline_position=4,
                    feature_row_count=0,
                    output_kind="main",
                    artifact_kind=None,
                    output_key="main",
                    step_name="Display raw context",
                )
                supporting["visible"] = self.supporting_visible
                layers.append(supporting)
            return self._execution(
                "openhcs_get_viewer_window_state",
                {"observed": True, "layers": layers},
            )
        if command == "navigate-viewer":
            route_key = argv[7]
            if route_key == "supporting-route":
                self.supporting_visible = True
            else:
                self.viewer_selection_applied = True
                self.viewer_data_index_applied = (
                    int(argv[argv.index("--data-index") + 1])
                    if "--data-index" in argv
                    else None
                )
            feature_row_count = (
                0 if route_key == "supporting-route" else self.viewer_feature_row_count
            )
            return self._execution(
                "openhcs_navigate_viewer_window",
                {
                    "observed": True,
                    "route_key": route_key,
                    "visible": True,
                    "selected": route_key != "supporting-route",
                    "data_index": self.viewer_data_index_applied,
                    "feature_row_count": feature_row_count,
                    "selected_data_indices": (
                        [self.viewer_data_index_applied]
                        if self.viewer_data_index_applied is not None
                        else []
                    ),
                },
            )
        raise AssertionError(f"unexpected command: {argv}")


def test_mcp_operations_register_select_and_run_through_public_ui_contract(tmp_path):
    definition = replace(
        master.built_in_demo_definitions(tmp_path)[0],
        preparation_argv=(),
    )
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(str(definition.contribution.plate_path))
    descriptor = tmp_path / "ui.json"
    descriptor.write_text("{}", encoding="utf-8")
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=descriptor,
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    assert operations.register_all((item,)) == {
        str(definition.contribution.plate_path): "#0a64c8"
    }
    operations.select_plate(item)
    operations.run_workflow(item, master.UiSelectedPlateWorkflowKind.INIT)

    commands = [argv[0] for argv in client.argv]
    assert commands == [
        "code-document",
        "apply-code-document",
        "state-surface",
        "code-document",
        "apply-code-document",
        "state-surface",
        "widget-tree",
        "invoke-widget-action",
        "state-surface",
        "selected-workflow",
    ]
    action_call = next(
        argv for argv in client.argv if argv[0] == "invoke-widget-action"
    )
    assert "--action-kind" in action_call
    assert action_call[action_call.index("--action-kind") + 1] == "item_select"
    assert client.argv[-1][1] == "init_plate"


def test_mcp_operations_wait_for_long_ui_apply_through_operation_owner(tmp_path):
    definition = replace(
        master.built_in_demo_definitions(tmp_path)[0],
        preparation_argv=(),
    )
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        async_apply=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    assert operations.register_all((item,)) == {
        str(definition.contribution.plate_path): "#0a64c8"
    }
    assert client.operation_wait_calls == 2
    wait_calls = [
        argv
        for argv in client.argv
        if argv[:2]
        == (
            "call",
            "openhcs_ui_wait_for_operation_receipt",
        )
    ]
    assert len(wait_calls) == 2
    for wait_call in wait_calls:
        arguments = json.loads(wait_call[wait_call.index("--arguments") + 1])
        request_timeout = arguments["timeout_seconds"]
        client_timeout = float(wait_call[wait_call.index("--timeout-seconds") + 1])
        assert client_timeout == request_timeout + 5.0


def test_mcp_operations_retry_read_only_document_without_replaying_apply(tmp_path):
    definition = replace(
        master.built_in_demo_definitions(tmp_path)[0],
        preparation_argv=(),
    )
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        transient_document_error=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    operations.register_all((item,))

    assert client.document_calls == 3
    assert sum(argv[0] == "apply-code-document" for argv in client.argv) == 2


def test_mcp_operations_wait_for_accepted_async_row_selection(tmp_path):
    definition = replace(
        master.built_in_demo_definitions(tmp_path)[0],
        preparation_argv=(),
    )
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        async_select=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    operations.select_plate(item)

    assert client.operation_wait_calls == 1


def test_mcp_operations_retry_read_only_widget_projection_without_reselecting(
    tmp_path,
):
    definition = replace(
        master.built_in_demo_definitions(tmp_path)[0],
        preparation_argv=(),
    )
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        transient_widget_error=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    operations.select_plate(item)

    assert client.widget_calls == 2
    assert sum(argv[0] == "invoke-widget-action" for argv in client.argv) == 1


def test_plate_manager_state_retries_a_transient_busy_projection(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        transient_state_error=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    state = operations._plate_manager_state("all")

    assert state["rows"][0]["plate_scope_id"] == str(definition.contribution.plate_path)
    assert client.state_calls == 2


def test_mcp_operations_surface_declared_restart_as_session_invalidation(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        stale_workflow=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    with pytest.raises(master.MasterDemoSessionInvalidated, match="invalidated"):
        operations.run_workflow(
            item,
            master.UiSelectedPlateWorkflowKind.COMPILE,
        )


def test_master_document_projects_exact_ui_color_into_runtime_viewer_config(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    scope_id = str(definition.contribution.plate_path)
    payload = PlateManagerCodeDocumentAuthority.from_source(
        master.render_master_document((item,), {scope_id: "#1a2b3c"})
    )
    config = payload.per_plate_configs[scope_id].napari_streaming_config
    steps = payload.pipeline_data[scope_id]

    assert config.scope_accent_color == "#1a2b3c"
    assert config.viewer_runtime_config().scope_accent_color == "#1a2b3c"
    assert {
        step.napari_streaming_config.scope_accent_color
        for step in steps
        if step.napari_streaming_config.enabled
    } == {"#1a2b3c"}


def test_mcp_operations_wait_for_runtime_owned_viewer(tmp_path, monkeypatch):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]

    class _Endpoint:
        @staticmethod
        def wait_ready(*, timeout, require_ready):
            assert timeout == 20.0
            assert require_ready is True
            return True

    monkeypatch.setattr(
        master.McpMasterDemoOperations,
        "_endpoint",
        staticmethod(lambda _item: _Endpoint()),
    )
    operations = master.McpMasterDemoOperations(
        client=object(),
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    operations.wait_for_viewer(item)


def test_mcp_operations_present_native_overlay_and_linked_feature_row(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(str(definition.contribution.plate_path))
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
        presentation_dwell_seconds=0,
    )
    operations.presentation_contracts[definition.contribution.demo_id] = (
        _viewer_contract(
            "final-route",
            pipeline_position=3,
        )
    )

    presentation = operations.present_visual_result(item)

    assert presentation.route_key == "final-route"
    assert presentation.pipeline_position == 3
    assert presentation.feature_row_count == 9
    assert presentation.selected_data_indices == (1,)
    assert [argv[0] for argv in client.argv] == [
        "viewer-state",
        "navigate-viewer",
        "navigate-viewer",
        "viewer-state",
    ]
    navigation = client.argv[1]
    assert navigation[navigation.index("--data-index") + 1] == "8"
    assert "--visible" in navigation
    assert "--selected" in navigation
    assert client.argv[2][client.argv[2].index("--data-index") + 1] == "1"


def test_mcp_operations_select_declared_image_without_data_index(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        viewer_feature_row_count=0,
        viewer_output_kind="main",
        viewer_artifact_kind=None,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
        presentation_dwell_seconds=0,
    )
    operations.presentation_contracts[definition.contribution.demo_id] = (
        _viewer_contract(
            "final-route",
            pipeline_position=3,
            output_kind="main",
            artifact_kind=None,
        )
    )

    presentation = operations.present_visual_result(item)

    assert presentation.feature_row_count == 0
    assert presentation.selected_data_indices == ()
    assert presentation.feature_selection_applied is False
    navigation = next(argv for argv in client.argv if argv[0] == "navigate-viewer")
    assert "--data-index" not in navigation


def test_mcp_operations_keep_declared_raw_context_visible_behind_final_graph(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        include_supporting_layer=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
        presentation_dwell_seconds=0,
    )
    supporting_identity = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="main",
        projection_key="main",
        step_name="Display raw context",
        pipeline_position=4,
    )
    target_contract = _viewer_contract("final-route", pipeline_position=3)
    operations.presentation_contracts[definition.contribution.demo_id] = replace(
        target_contract,
        supporting_visual_identities=(supporting_identity,),
    )

    presentation = operations.present_visual_result(item)

    assert presentation.route_key == "final-route"
    navigations = [argv for argv in client.argv if argv[0] == "navigate-viewer"]
    assert [argv[7] for argv in navigations] == [
        "supporting-route",
        "final-route",
        "final-route",
    ]
    assert "--visible" in navigations[0]
    assert "--selected" not in navigations[0]
    assert "--data-index" not in navigations[0]
    assert "--selected" in navigations[1]
    assert "--data-index" in navigations[1]
    assert "--selected" in navigations[2]
    assert "--data-index" in navigations[2]


def test_mcp_operations_present_every_compiled_live_measurement_table(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        measurement_names=("cell_counts", "cell_shapes"),
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )
    operations.presentation_contracts[definition.contribution.demo_id] = (
        master.CompiledPresentationContract(
            visual_identity=_viewer_contract(
                "final-route",
                pipeline_position=3,
            ).visual_identity,
            measurement_names=("cell_counts", "cell_shapes"),
        )
    )

    presentation = operations.present_measurements(item)

    assert presentation.surface_id == "live-results"
    assert presentation.measurement_names == ("cell_counts", "cell_shapes")
    assert presentation.row_count == 4
    assert presentation.populated_preview_count == 2
    assert presentation.previews[0]["rows"][0]["value"] == 3.0
    assert presentation.human_results_table["status"] == "completed"
    assert [argv[0] for argv in client.argv] == [
        "actions",
        "state-surfaces",
        "state-surface",
        "invoke-action",
    ]


def test_mcp_operations_accept_completed_results_dispatch_outcome(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]
    client = _FakePublicMcpClient(
        str(definition.contribution.plate_path),
        async_results=True,
    )
    operations = master.McpMasterDemoOperations(
        client=client,
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )
    operations.presentation_contracts[definition.contribution.demo_id] = (
        _viewer_contract(
            "final-route",
            pipeline_position=3,
        )
    )

    presentation = operations.present_measurements(item)

    assert presentation.human_results_table["status"] == "completed"
    assert presentation.human_results_table["outcome"] == "accepted"


def test_mcp_operations_reject_endpoint_collision_without_port_fallback(
    tmp_path,
    monkeypatch,
):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]

    class _Endpoint:
        control_port = 7200
        zmq_transport_mode = object()
        host = "localhost"
        config = object()
        stale_cleanup_calls = 0

        @classmethod
        def remove_stale_addresses(cls):
            cls.stale_cleanup_calls += 1
            return ()

        @staticmethod
        def in_use():
            return True

    monkeypatch.setattr(
        master.McpMasterDemoOperations,
        "_endpoint",
        staticmethod(lambda _item: _Endpoint()),
    )
    operations = master.McpMasterDemoOperations(
        client=object(),
        descriptor_path=tmp_path / "ui.json",
        session_root=tmp_path / "session",
        workflow_timeout_seconds=10.0,
    )

    try:
        operations.assert_port_available(item)
    except master.MasterDemoFailure as error:
        assert "6200" in str(error)
        assert "no fallback port" in str(error)
    else:
        raise AssertionError("occupied endpoint was accepted")
    assert _Endpoint.stale_cleanup_calls == 1


def test_master_viewer_endpoint_uses_the_openhcs_transport_owner(tmp_path):
    definition = master.built_in_demo_definitions(tmp_path)[0]
    item = master.build_demo_schedule((definition,), base_port=6200)[0]

    endpoint = master.McpMasterDemoOperations._endpoint(item)

    assert endpoint.config is master.OPENHCS_ZMQ_CONFIG
    assert endpoint.data_url().endswith("/.openhcs/ipc/openhcs-zmq-6200.sock")
    assert endpoint.control_url().endswith("/.openhcs/ipc/openhcs-zmq-7200.sock")
