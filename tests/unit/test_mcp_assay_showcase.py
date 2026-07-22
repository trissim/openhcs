from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import mcp_assay_showcase as showcase


def _assignment_names(source: str) -> set[str]:
    return {
        target.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }


def test_showcase_sources_are_three_bounded_distinct_pipeline_documents(tmp_path):
    blueprints = showcase.scenario_blueprints()

    assert [blueprint.scenario_id for blueprint in blueprints] == [
        "primary_object_segmentation",
        "dual_channel_phenotype",
        "image_colocalization",
    ]
    assert len({blueprint.biological_question for blueprint in blueprints}) == 3
    assert all(blueprint.assay_budget.seconds > 0 for blueprint in blueprints)

    expected_callable = {
        "primary_object_segmentation": "count_cells_single_channel",
        "dual_channel_phenotype": "count_cells_simple_dual_channel",
        "image_colocalization": "measure_colocalization",
    }
    for blueprint in blueprints:
        plate_path = tmp_path / blueprint.scenario_id / "plate"
        output_path = tmp_path / blueprint.scenario_id / "output"
        source = blueprint.pipeline_source(plate_path, output_path)
        pipeline_source = showcase.artifact_plan_source(source)

        compile(source, f"<{blueprint.scenario_id}>", "exec")
        compile(pipeline_source, f"<{blueprint.scenario_id}-pipeline>", "exec")
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
        assert str(plate_path) in source
        assert str(output_path) in source
        if blueprint.wavelengths == 1:
            assert "GroupBy" not in source
            assert "VariableComponents" not in source
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


def test_artifact_contracts_derive_outputs_from_compiled_plan():
    result = showcase._artifact_contracts(
        {
            "axis_count": 1,
            "step_count": 1,
            "source_workspace": {
                "file_count": 2,
                "truncated_file_count": 0,
            },
            "steps": [
                {
                    "artifact_outputs": [
                        {"name": "cells", "kind": "object_labels"},
                        {"name": "cell_rows", "kind": "measurements"},
                    ]
                }
            ],
        }
    )

    assert result["measurement_names"] == ["cell_rows"]
    assert result["source_file_count"] == 2
    assert result["final_biological_outputs"] == [
        {"name": "cell_rows", "kind": "measurements"},
        {"name": "cells", "kind": "object_labels"},
    ]


def test_artifact_contracts_require_a_measurement_output():
    with pytest.raises(showcase.ShowcaseFailure, match="measurement-table"):
        showcase._artifact_contracts(
            {
                "axis_count": 1,
                "step_count": 1,
                "source_workspace": {
                    "file_count": 1,
                    "truncated_file_count": 0,
                },
                "steps": [
                    {
                        "artifact_outputs": [
                            {"name": "cells", "kind": "object_labels"}
                        ]
                    }
                ],
            }
        )


def test_artifact_contracts_reject_a_truncated_source_scope() -> None:
    with pytest.raises(showcase.ShowcaseFailure, match="non-empty and untruncated"):
        showcase._artifact_contracts(
            {
                "axis_count": 1,
                "step_count": 1,
                "source_workspace": {
                    "file_count": 1,
                    "truncated_file_count": 1,
                },
                "steps": [
                    {
                        "artifact_outputs": [
                            {"name": "cell_rows", "kind": "measurements"}
                        ]
                    }
                ],
            }
        )


def test_live_measurement_surface_is_discovered_from_public_catalog():
    surface_id = showcase._discover_live_measurement_surface_id(
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
    source = inspect.getsource(showcase._discover_live_measurement_surface_id)
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

    evidence = showcase._live_measurement_evidence(
        surface,
        surface_id="runtime-owned-id",
        plate_path=plate_path,
        measurement_names=["cell_rows"],
    )

    assert evidence["surface_id"] == "runtime-owned-id"
    assert evidence["row_count"] == 1
    with pytest.raises(showcase.ShowcaseFailure, match="missing_rows"):
        showcase._live_measurement_evidence(
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
                "results": [
                    {"tool": tool, "mcp_error": False, "payloads": [payload]}
                ],
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
                            "artifact_outputs": [
                                {
                                    "name": "assay_measurements",
                                    "kind": "measurements",
                                },
                                {"name": "objects", "kind": "object_labels"},
                            ]
                        }
                    ],
                },
            )
        if command == "code-document":
            return self._execution(
                "openhcs_ui_get_code_document",
                {"current_revision_token": "revision-1"},
            )
        if command == "validate-code-document":
            return self._execution(
                "openhcs_ui_validate_code_document", {"valid": True}
            )
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
                                    "address": {
                                        "key": {"name": "assay_measurements"}
                                    },
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
    assert scenario["live_measurements"]["measurement_names"] == [
        "assay_measurements"
    ]
    assert scenario["materialized_measurements"]["measurement_snapshot_count"] == 1
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
    assert any(argv[:2] == ("state-surface", "runtime-owned-id") for argv in fake_client.argv)
    assert any(
        argv[:3] == ("invoke-action", "plate_manager", "view_results")
        for argv in fake_client.argv
    )
