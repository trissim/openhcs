from __future__ import annotations

import ast
import inspect
from multiprocessing import SimpleQueue
from pathlib import Path
from types import SimpleNamespace

import pytest
from objectstate import ObjectStateRegistry
from openhcs.config_framework.lazy_factory import ensure_global_config_context
from polystore.streaming.identity import StreamProducerOrigin

from openhcs.core.aligned_image_payload import AlignedImageSliceContext
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.progress import set_progress_queue
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)
from scripts import mcp_assay_showcase as showcase


def _assignment_names(source: str) -> set[str]:
    return {
        target.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def test_showcase_sources_are_seven_bounded_distinct_pipeline_documents(tmp_path):
    blueprints = showcase.scenario_blueprints()

    assert [blueprint.scenario_id for blueprint in blueprints] == [
        "primary_object_segmentation",
        "dual_channel_phenotype",
        "image_colocalization",
        "nuclear_morphology",
        "spatial_neighbors",
        "radial_intensity_distribution",
        "foreground_skeleton_topology",
    ]
    assert len({blueprint.biological_question for blueprint in blueprints}) == 7
    assert len({blueprint.title for blueprint in blueprints}) == 7
    assert all(blueprint.assay_budget.seconds > 0 for blueprint in blueprints)

    expected_visuals = {
        "primary_object_segmentation": (
            "Segment nucleus-like primary objects",
            "artifact",
            "segmentation_masks",
            ObjectLabelsArtifactType.require_value(),
        ),
        "dual_channel_phenotype": (
            "Classify dual-channel reporter phenotype",
            "artifact",
            "w2_stain",
            ObjectLabelsArtifactType.require_value(),
        ),
        "image_colocalization": (
            "Render shared-intensity colocalization",
            AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
            AlignedImageSliceContext.ANONYMOUS_MAIN_FLOW_OUTPUT_KEY,
            None,
        ),
        "nuclear_morphology": (
            "Segment nuclei for morphology",
            "artifact",
            "segmentation_masks",
            ObjectLabelsArtifactType.require_value(),
        ),
        "spatial_neighbors": (
            "Measure cell-neighbor topology",
            AlignedImageSliceContext.MAIN_FLOW_OUTPUT_KIND,
            "MeasureObjectNeighbors_2_image_1",
            ImageArtifactType.require_value(),
        ),
        "radial_intensity_distribution": (
            "Segment nuclei for radial intensity",
            "artifact",
            "segmentation_masks",
            ObjectLabelsArtifactType.require_value(),
        ),
        "foreground_skeleton_topology": (
            "Measure foreground skeleton topology",
            "artifact",
            "skeleton_rois",
            ObjectLabelsArtifactType.require_value(),
        ),
    }

    expected_callable = {
        "primary_object_segmentation": "count_cells_single_channel",
        "dual_channel_phenotype": "count_cells_simple_dual_channel",
        "image_colocalization": "measure_colocalization",
        "nuclear_morphology": "measure_object_size_shape",
        "spatial_neighbors": "measure_object_neighbors",
        "radial_intensity_distribution": "measure_object_intensity_distribution",
        "foreground_skeleton_topology": "skeletonize_and_save",
    }
    assert showcase.SHOWCASE_NAPARI_PORT == 5889
    for blueprint_index, blueprint in enumerate(blueprints):
        plate_path = tmp_path / blueprint.scenario_id / "plate"
        output_path = tmp_path / blueprint.scenario_id / "output"
        source = blueprint.pipeline_source(plate_path, output_path)
        pipeline_source = showcase.artifact_plan_source(source)

        compile(source, f"<{blueprint.scenario_id}>", "exec")
        pipeline_code = compile(
            pipeline_source, f"<{blueprint.scenario_id}-pipeline>", "exec"
        )
        namespace: dict[str, object] = {}
        exec(pipeline_code, namespace)
        visual = blueprint.presentation_identity
        assert visual.origin == StreamProducerOrigin.PIPELINE.value
        assert (
            visual.step_name,
            visual.output_kind,
            visual.output_key,
            visual.artifact_kind,
        ) == expected_visuals[blueprint.scenario_id]
        assert (
            namespace["pipeline_config"]
            is namespace["per_plate_configs"][namespace["plate_paths"][0]]
        )
        assert namespace["pipeline_steps"]
        streamed_steps = [
            step
            for step in namespace["pipeline_steps"]
            if step.napari_streaming_config.enabled
        ]
        assert len(streamed_steps) == 1
        assert streamed_steps[0].name == visual.step_name
        stream_config = streamed_steps[0].napari_streaming_config
        assert stream_config.port == showcase.SHOWCASE_NAPARI_PORT
        assert stream_config.persistent is True
        assert stream_config.well_filter is None
        assert _assignment_names(source) == {
            "plate_paths",
            "global_config",
            "per_plate_configs",
            "pipeline_data",
        }
        assert {"pipeline_config", "pipeline_steps"} <= _assignment_names(
            pipeline_source
        )
        assert expected_callable[blueprint.scenario_id] in source
        assert "well_filter='A01'" in source
        assert "well_filter=0" in source
        assert "materialize_runtime_artifacts=True" in source
        assert "SourceBinding" not in source
        assert ".cppipe" not in source
        assert str(plate_path) in source
        assert str(output_path) in source
        if blueprint_index >= 3:
            assert "export_to_spreadsheet" in source
            assert '"export_all_measurement_types": True' in source
        if blueprint.wavelengths == 1:
            if "VariableComponents" in source:
                assert (
                    "from openhcs.constants.constants import "
                    "GroupBy, VariableComponents" in source
                )
                assert "variable_components=[VariableComponents.SITE]" in source
                assert "group_by=GroupBy.CHANNEL" in source
            else:
                assert "GroupBy" not in source
        else:
            assert (
                "from openhcs.constants.constants import GroupBy, VariableComponents"
                in source
            )

        generation = blueprint.generation_arguments(plate_path)
        assert generation[0] == "generate-synthetic-plate"
        assert generation[generation.index("--grid-rows") + 1] == "1"
        assert generation[generation.index("--grid-cols") + 1] == "1"
        assert generation[generation.index("--z-stack-levels") + 1] == "1"
        assert generation.count("--well") == 1
        if blueprint.scenario_id == "spatial_neighbors":
            _callable, streamed_kwargs = streamed_steps[0].func
            assert streamed_kwargs["retain_neighbor_count_image"] is True
            assert streamed_kwargs["retain_percent_touching_image"] is False
        if blueprint.scenario_id == "image_colocalization":
            assert "create_projection" in source
            assert '"method": "min_projection"' in source
            assert "image_math" not in source
            assert len(namespace["pipeline_steps"]) == 3


def test_colocalization_presentation_compiles_from_public_native_source(
    tmp_path: Path,
) -> None:
    blueprint = next(
        blueprint
        for blueprint in showcase.scenario_blueprints()
        if blueprint.scenario_id == "image_colocalization"
    )
    plate_path = tmp_path / "plate"
    SyntheticMicroscopyGenerator(
        output_dir=str(plate_path),
        grid_size=(1, 1),
        tile_size=(32, 32),
        wavelengths=2,
        z_stack_levels=1,
        num_cells=2,
        wells=["A01"],
        random_seed=31,
    ).generate_dataset()
    source = showcase.artifact_plan_source(
        blueprint.pipeline_source(plate_path, tmp_path / "output")
    )
    namespace: dict[str, object] = {}
    exec(compile(source, "<image-colocalization-pipeline>", "exec"), namespace)

    ObjectStateRegistry.clear()
    set_progress_queue(SimpleQueue())
    try:
        ensure_global_config_context(
            GlobalPipelineConfig,
            namespace["global_config"],
        )
        orchestrator = PipelineOrchestrator(
            plate_path,
            pipeline_config=namespace["pipeline_config"],
        ).initialize()
        compilation = orchestrator.compile_pipelines(
            pipeline_definition=namespace["pipeline_steps"],
            well_filter=["A01"],
            is_zmq_execution=True,
        )
    finally:
        set_progress_queue(None)

    context = compilation["execution_bundle"].runtime_contexts["A01"]
    assert [plan.step_name for plan in context.step_plans.values()] == [
        "Measure two-channel colocalization",
        "Render shared-intensity colocalization",
        "Export colocalization measurements",
    ]


def test_showcase_keeps_one_blueprint_and_execution_harness_authority() -> None:
    module = ast.parse(inspect.getsource(showcase))
    classes = [node.name for node in module.body if isinstance(node, ast.ClassDef)]
    functions = [node.name for node in module.body if isinstance(node, ast.FunctionDef)]

    assert classes.count("ScenarioBlueprint") == 1
    assert functions.count("scenario_blueprints") == 1
    assert functions.count("run_showcase") == 1
    assert not any("registry" in name.casefold() for name in classes + functions)


def test_artifact_contracts_derive_outputs_from_compiled_plan():
    result = showcase.artifact_contracts(
        {
            "axis_count": 1,
            "step_count": 1,
            "source_workspace": {
                "file_count": 2,
                "truncated_file_count": 0,
            },
            "steps": [
                {
                    "step_name": "Segment cells",
                    "artifact_outputs": [
                        {"name": "cells", "kind": "object_labels"},
                        {"name": "cell_rows", "kind": "measurements"},
                    ],
                }
            ],
        }
    )

    assert result["measurement_names"] == ["cell_rows"]
    assert result["step_names"] == ["Segment cells"]
    assert result["source_file_count"] == 2
    assert result["final_biological_outputs"] == [
        {
            "step_name": "Segment cells",
            "name": "cells",
            "kind": "object_labels",
        },
        {
            "step_name": "Segment cells",
            "name": "cell_rows",
            "kind": "measurements",
        },
    ]


def test_artifact_contracts_require_a_measurement_output():
    with pytest.raises(showcase.ShowcaseFailure, match="measurement-table"):
        showcase.artifact_contracts(
            {
                "axis_count": 1,
                "step_count": 1,
                "source_workspace": {
                    "file_count": 1,
                    "truncated_file_count": 0,
                },
                "steps": [
                    {
                        "step_name": "Segment cells",
                        "artifact_outputs": [
                            {"name": "cells", "kind": "object_labels"}
                        ],
                    }
                ],
            }
        )


def test_artifact_contracts_reject_a_truncated_source_scope() -> None:
    with pytest.raises(showcase.ShowcaseFailure, match="non-empty and untruncated"):
        showcase.artifact_contracts(
            {
                "axis_count": 1,
                "step_count": 1,
                "source_workspace": {
                    "file_count": 1,
                    "truncated_file_count": 1,
                },
                "steps": [
                    {
                        "step_name": "Measure cells",
                        "artifact_outputs": [
                            {"name": "cell_rows", "kind": "measurements"}
                        ],
                    }
                ],
            }
        )


def test_live_measurement_surface_is_discovered_from_public_catalog():
    surface_id = showcase.discover_live_measurement_surface_id(
        {
            "actions": [
                {
                    "widget_id": "plate_manager",
                    "action_id": "view_results",
                    "related_state_surface_ids": [
                        "plate_manager.state",
                        "runtime-owned-id",
                    ],
                },
            ]
        },
        {
            "surfaces": [
                {"surface_id": "plate_manager.state"},
                {"surface_id": "runtime-owned-id"},
            ]
        },
    )

    assert surface_id == "runtime-owned-id"
    source = inspect.getsource(showcase.discover_live_measurement_surface_id)
    assert '.get("title")' not in source
    assert '.get("identity")' not in source
    assert "casefold" not in source


def test_result_records_require_the_current_query_envelope() -> None:
    expected = [{"full_path": "/tmp/result.csv", "preview": {}}]

    assert showcase._result_records({"query": {"records": expected}}) == expected
    with pytest.raises(showcase.ShowcaseFailure, match="structured records"):
        showcase._result_records({"records": expected})


def test_live_measurement_evidence_requires_owner_declared_populated_tables(tmp_path):
    plate_path = tmp_path / "plate"
    surface = {
        "payload": {
            "entries": [
                {
                    "plate_id": str(plate_path),
                    "step_name": "Count",
                    "preview": {
                        "address": {"key": {"name": "cell_rows"}},
                        "columns": ["object_number", "area"],
                        "rows": [{"object_number": 1, "area": 42.0}],
                        "row_count": 1,
                    },
                }
            ]
        }
    }

    evidence = showcase.live_measurement_evidence(
        surface,
        surface_id="runtime-owned-id",
        plate_path=plate_path,
        measurement_names=["cell_rows"],
    )

    assert evidence["surface_id"] == "runtime-owned-id"
    assert evidence["row_count"] == 1
    with pytest.raises(showcase.ShowcaseFailure, match="missing_rows"):
        showcase.live_measurement_evidence(
            surface,
            surface_id="runtime-owned-id",
            plate_path=plate_path,
            measurement_names=["missing_rows"],
        )


def test_materialized_measurement_evidence_requires_csv_rows():
    evidence = showcase._materialized_measurement_evidence(
        [
            {
                "full_path": "/tmp/results/cell_rows.csv",
                "preview": {
                    "csv_columns": ["object_number"],
                    "csv_rows": [{"object_number": "1"}],
                },
            }
        ],
        ["cell_rows"],
    )

    assert evidence["measurement_snapshot_count"] == 1
    with pytest.raises(showcase.ShowcaseFailure, match="no non-empty"):
        showcase._materialized_measurement_evidence(
            [
                {
                    "full_path": "/tmp/results/cell_rows.csv",
                    "preview": {
                        "csv_columns": ["object_number"],
                        "csv_rows": [],
                    },
                }
            ],
            ["other_rows"],
        )
    with pytest.raises(showcase.ShowcaseFailure, match="canonical full_path"):
        showcase._materialized_measurement_evidence(
            [
                {
                    "relative_path": "results/cell_rows.csv",
                    "preview": {
                        "csv_columns": ["object_number"],
                        "csv_rows": [{"object_number": "1"}],
                    },
                }
            ],
            ["cell_rows"],
        )


class _FakeMcpClient:
    def __init__(self, *args):
        del args
        self.argv: list[tuple[str, ...]] = []
        self.enter_count = 0
        self.plate_path: Path | None = None

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, *args):
        del args

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

    def execute(self, argv, *, timeout_seconds):
        del timeout_seconds
        argv = tuple(argv)
        self.argv.append(argv)
        command = argv[0]
        if command == "generate-synthetic-plate":
            self.plate_path = Path(argv[1])
            self.plate_path.mkdir(parents=True)
            return self._execution(
                "openhcs_generate_synthetic_plate",
                {"output_dir": str(self.plate_path), "generated_file_count": 1},
            )
        if command == "artifact-plan":
            return self._execution(
                "openhcs_inspect_pipeline_source_artifact_plan",
                {
                    "axis_count": 1,
                    "step_count": 1,
                    "source_workspace": {
                        "file_count": 1,
                        "truncated_file_count": 0,
                    },
                    "steps": [
                        {
                            "step_name": "Segment nucleus-like primary objects",
                            "artifact_outputs": [
                                {
                                    "name": "assay_measurements",
                                    "kind": "measurements",
                                },
                                {
                                    "name": "segmentation_masks",
                                    "kind": "object_labels",
                                },
                                {
                                    "name": "morphology",
                                    "kind": "spatial_graph",
                                },
                            ],
                        }
                    ],
                },
            )
        if command == "viewer-state":
            return self._execution(
                "openhcs_get_viewer_window_state",
                {
                    "observed": True,
                    "layers": [
                        {
                            "route_key": "segment-main",
                            "title": "Segment source",
                            "mounted": True,
                            "item_count": 1,
                            "producer_identities": [
                                {
                                    "origin": "pipeline",
                                    "output_kind": "main",
                                    "output_key": "main",
                                    "projection_key": "main",
                                    "step_name": "Segment nucleus-like primary objects",
                                    "pipeline_position": 0,
                                    "artifact_kind": "image",
                                }
                            ],
                        },
                        {
                            "route_key": "segment-labels",
                            "title": "Segmented objects",
                            "mounted": True,
                            "item_count": 1,
                            "producer_identities": [
                                {
                                    "origin": "pipeline",
                                    "output_kind": "artifact",
                                    "output_key": "segmentation_masks",
                                    "projection_key": "segmentation_masks",
                                    "step_name": "Segment nucleus-like primary objects",
                                    "pipeline_position": 0,
                                    "step_scope_id": "submission::step_0",
                                    "invocation_key": "axis::A01",
                                    "artifact_kind": "object_labels",
                                },
                                {
                                    "origin": "pipeline",
                                    "output_kind": "artifact",
                                    "output_key": "segmentation_masks",
                                    "projection_key": "different_projection",
                                    "step_name": "Segment nucleus-like primary objects",
                                    "pipeline_position": 0,
                                    "step_scope_id": "submission::step_0",
                                    "invocation_key": "axis::A01",
                                    "artifact_kind": "object_labels",
                                },
                            ],
                        },
                        {
                            "route_key": "prior-assay",
                            "title": "Prior assay",
                            "mounted": True,
                            "item_count": 1,
                            "producer_identities": [
                                {
                                    "origin": "pipeline",
                                    "output_kind": "main",
                                    "output_key": "main",
                                    "projection_key": "main",
                                    "step_name": "Prior assay",
                                    "pipeline_position": 0,
                                    "artifact_kind": "image",
                                }
                            ],
                        },
                        {
                            "route_key": "segment-morphology",
                            "title": "Final morphology",
                            "mounted": True,
                            "item_count": 1,
                            "producer_identities": [
                                {
                                    "origin": "pipeline",
                                    "output_kind": "artifact",
                                    "output_key": "morphology",
                                    "projection_key": "morphology",
                                    "step_name": "Segment nucleus-like primary objects",
                                    "pipeline_position": 0,
                                    "artifact_kind": "spatial_graph",
                                }
                            ],
                        },
                    ],
                },
            )
        if command == "isolate-viewer":
            return self._execution(
                "openhcs_isolate_viewer_window_layers",
                {
                    "observed": True,
                    "applied": True,
                    "visible_route_keys": [
                        "segment-main",
                        "segment-labels",
                        "segment-morphology",
                    ],
                    "hidden_route_keys": ["prior-assay"],
                },
            )
        if command == "code-document":
            return self._execution(
                "openhcs_ui_get_code_document",
                {"current_revision_token": "revision-1"},
            )
        if command == "validate-code-document":
            return self._execution("openhcs_ui_validate_code_document", {"valid": True})
        if command == "apply-code-document":
            return self._execution(
                "openhcs_ui_apply_code_document",
                {"applied": True, "outcome": "applied"},
            )
        if command == "selected-workflow":
            return self._execution(
                "mcp_dev_selected_workflow_poll",
                {"poll_completed": True, "poll_status": "completed"},
            )
        if command == "state-surfaces":
            return self._execution(
                "openhcs_ui_list_state_surfaces",
                {
                    "surfaces": [
                        {
                            "surface_id": "runtime-owned-id",
                            "widget_id": "plate_manager",
                            "title": "Live measurement results",
                        }
                    ]
                },
            )
        if command == "actions":
            return self._execution(
                "openhcs_ui_list_actions",
                {
                    "actions": [
                        {
                            "widget_id": "plate_manager",
                            "action_id": "view_results",
                            "related_state_surface_ids": [
                                "plate_manager.state",
                                "runtime-owned-id",
                            ],
                        }
                    ]
                },
            )
        if command == "state-surface":
            assert self.plate_path is not None
            return self._execution(
                "openhcs_ui_get_state_surface",
                {
                    "payload": {
                        "entries": [
                            {
                                "plate_id": str(self.plate_path),
                                "step_name": "Assay",
                                "preview": {
                                    "address": {"key": {"name": "assay_measurements"}},
                                    "columns": ["count"],
                                    "rows": [{"count": 3}],
                                    "row_count": 1,
                                },
                            }
                        ]
                    }
                },
            )
        if command == "invoke-action":
            return self._execution(
                "openhcs_ui_invoke_action",
                {
                    "status": "accepted",
                    "receipt": {"bridge_operation_id": "operation-1"},
                },
            )
        if command == "call":
            assert argv[1] == "openhcs_ui_wait_for_operation"
            return self._execution(
                "openhcs_ui_wait_for_operation",
                {"status": "completed", "outcome": "accepted"},
            )
        if command == "selected-plate-files":
            return self._execution(
                "openhcs_ui_query_selected_plate_files",
                {
                    "query": {
                        "records": [
                            {
                                "full_path": "/tmp/results/assay_measurements.csv",
                                "preview": {
                                    "csv_columns": ["count"],
                                    "csv_rows": [{"count": "3"}],
                                },
                            }
                        ]
                    }
                },
            )
        raise AssertionError(f"Unexpected command: {argv}")


def test_showcase_reuses_one_persistent_session_and_writes_timing_report(tmp_path):
    descriptor = tmp_path / "bridge.json"
    descriptor.write_text("{}", encoding="utf-8")
    fake_client = _FakeMcpClient()

    report = showcase.run_showcase(
        showcase.scenario_blueprints()[:1],
        session_dir=tmp_path / "session",
        descriptor_path=descriptor,
        budget_scale=10.0,
        client_factory=lambda *args: fake_client,
    )

    assert fake_client.enter_count == 1
    assert report["persistent_mcp_session"] is True
    assert report["scenario_count"] == 1
    scenario = report["scenarios"][0]
    assert scenario["live_measurements"]["measurement_names"] == ["assay_measurements"]
    assert scenario["materialized_measurements"]["measurement_snapshot_count"] == 1
    assert scenario["napari_presentation"] == {
        "port": 5889,
        "route_keys": [
            "segment-main",
            "segment-labels",
            "segment-morphology",
        ],
        "selected_route_key": "segment-labels",
        "visible_route_keys": [
            "segment-main",
            "segment-labels",
            "segment-morphology",
        ],
        "hidden_route_keys": ["prior-assay"],
        "layers": [
            {
                "route_key": "segment-main",
                "title": "Segment source",
                "producer_identities": [
                    {
                        "origin": "pipeline",
                        "output_kind": "main",
                        "output_key": "main",
                        "projection_key": "main",
                        "step_name": "Segment nucleus-like primary objects",
                        "pipeline_position": 0,
                        "artifact_kind": "image",
                    }
                ],
            },
            {
                "route_key": "segment-labels",
                "title": "Segmented objects",
                "producer_identities": [
                    {
                        "origin": "pipeline",
                        "output_kind": "artifact",
                        "output_key": "segmentation_masks",
                        "projection_key": "segmentation_masks",
                        "step_name": "Segment nucleus-like primary objects",
                        "pipeline_position": 0,
                        "step_scope_id": "submission::step_0",
                        "invocation_key": "axis::A01",
                        "artifact_kind": "object_labels",
                    },
                    {
                        "origin": "pipeline",
                        "output_kind": "artifact",
                        "output_key": "segmentation_masks",
                        "projection_key": "different_projection",
                        "step_name": "Segment nucleus-like primary objects",
                        "pipeline_position": 0,
                        "step_scope_id": "submission::step_0",
                        "invocation_key": "axis::A01",
                        "artifact_kind": "object_labels",
                    },
                ],
            },
            {
                "route_key": "segment-morphology",
                "title": "Final morphology",
                "producer_identities": [
                    {
                        "origin": "pipeline",
                        "output_kind": "artifact",
                        "output_key": "morphology",
                        "projection_key": "morphology",
                        "step_name": "Segment nucleus-like primary objects",
                        "pipeline_position": 0,
                        "artifact_kind": "spatial_graph",
                    }
                ],
            },
        ],
    }
    assert scenario["human_results_table"] == {
        "action_id": "view_results",
        "operation_id": "operation-1",
        "outcome": "accepted",
        "status": "completed",
    }
    assert {stage["name"] for stage in scenario["stages"]} == set(
        showcase.scenario_blueprints()[0].stage_budgets
    )
    assert all(stage["ok"] for stage in scenario["stages"])
    assert (tmp_path / "session" / "summary.json").is_file()
    assert any(argv[0] == "state-surfaces" for argv in fake_client.argv)
    assert any(argv[0] == "viewer-state" for argv in fake_client.argv)
    assert any(
        argv[0] == "isolate-viewer"
        and "segment-main" in argv
        and "segment-labels" in argv
        and "segment-morphology" in argv
        and "prior-assay" not in argv
        for argv in fake_client.argv
    )
    assert any(
        argv[:2] == ("state-surface", "runtime-owned-id") for argv in fake_client.argv
    )
    assert any(
        argv[:3] == ("invoke-action", "plate_manager", "view_results")
        for argv in fake_client.argv
    )
