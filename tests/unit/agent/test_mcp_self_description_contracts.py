"""Regressions for MCP contracts that must be usable without source inspection."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest

from openhcs.agent.path_policy import AgentPathPolicy, AgentPathPolicyError
from openhcs.agent.services.config_service import ConfigService
from openhcs.constants.constants import FileFormat
from openhcs.core.plate_file_inventory import (
    PlateFileInventoryQuery,
    PlateFileKind,
)
from openhcs.core.plate_image_inventory import (
    PlateFileInventory,
    PlateImageRecord,
    PlateResultFileRecord,
)
from openhcs.mcp import server
from openhcs.mcp.control_timeout import McpViewerTimeoutPolicy
from openhcs.microscopes.openhcs import OpenHCSMetadataHandler


def _tool_schemas() -> dict[str, dict]:
    built = server.build_server()
    listed_tools = built.list_tools()
    tools = (
        asyncio.run(listed_tools) if inspect.isawaitable(listed_tools) else listed_tools
    )
    return {tool.name: tool.inputSchema for tool in tools}


def test_plate_file_query_kind_is_derived_and_schema_constrained() -> None:
    assert PlateFileInventoryQuery.kind_choices() == (
        "all",
        *(member.value for member in PlateFileKind),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Plate file kind must be one of: 'all', 'image', 'result'; received 'roi'"
        ),
    ):
        PlateFileInventoryQuery.kind_from_value("roi")

    schemas = _tool_schemas()
    for tool_name in (
        "openhcs_query_plate_files",
        "openhcs_stream_plate_files_to_viewer",
        "openhcs_ui_query_selected_plate_files",
        "openhcs_ui_stream_selected_plate_files_to_viewer",
    ):
        query_schema = schemas[tool_name]
        assert "PlateFileQueryKind" not in query_schema["$defs"]
        kind_definition = query_schema["$defs"]["PlateFileKind"]
        assert kind_definition["enum"] == [
            member.value for member in PlateFileKind
        ]
        assert query_schema["properties"]["kind"]["anyOf"] == [
            {"$ref": "#/$defs/PlateFileKind"},
            {"const": "all", "type": "string"},
        ]


def test_exact_plate_file_path_reports_kind_exclusion() -> None:
    result_path = "/plate/results/A01_rois.roi.zip"
    inventory = PlateFileInventory(
        plate_path=Path("/plate"),
        image_records=(
            PlateImageRecord(
                virtual_path="A01.tif",
                full_virtual_path="/plate/A01.tif",
                backend="disk",
                source_path="/plate/A01.tif",
            ),
        ),
        result_records=(
            PlateResultFileRecord(
                relative_path="results/A01_rois.roi.zip",
                full_path=result_path,
                file_format=FileFormat.ROI,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "exists as kind 'result', but is excluded by the requested kind "
            r"filter \(image\)"
        ),
    ):
        inventory.require_file_record(
            result_path,
            kinds=(PlateFileKind.IMAGE,),
        )


def test_path_policy_error_includes_effective_writable_roots(
    tmp_path: Path,
) -> None:
    writable_root = tmp_path / "mcp_outputs"
    policy = AgentPathPolicy.with_roots(
        readable_roots=(tmp_path,),
        writable_roots=(writable_root,),
    )

    with pytest.raises(AgentPathPolicyError) as error:
        policy.assert_writable(tmp_path / "elsewhere" / "result.json")

    assert "Effective writable roots:" in str(error.value)
    assert str(writable_root.resolve()) in str(error.value)


def test_reflected_step_field_projects_nested_authoring_value_path() -> None:
    schema = ConfigService().describe_schema("step", "processing_config")
    fields_by_path = {field.path: field for field in schema.fields}

    assert fields_by_path[
        "processing_config.variable_components"
    ].authoring_value_path == (
        "processing_config",
        "variable_components",
    )
    assert fields_by_path["processing_config.group_by"].authoring_value_path == (
        "processing_config",
        "group_by",
    )

    binding_schema = ConfigService().describe_schema(
        "step",
        "source_bindings.bindings",
    )
    alias = next(
        field
        for field in binding_schema.fields
        if field.path == "source_bindings.bindings[].alias"
    )
    assert alias.authoring_value_path == (
        "source_bindings",
        "bindings",
        "[]",
        "alias",
    )


def test_mcp_schema_distinguishes_action_and_state_revisions_and_bounds_timeout() -> (
    None
):
    schemas = _tool_schemas()

    action_revision = schemas["openhcs_ui_invoke_action"]["properties"][
        "observed_selection_revision_token"
    ]
    assert "openhcs_ui_list_actions" in action_revision["description"]
    assert (
        "do not pass a state-surface revision token" in action_revision["description"]
    )

    state_revision = schemas["openhcs_ui_get_state_surface"]["properties"][
        "base_revision_token"
    ]
    assert "state/code-document" in state_revision["description"]
    assert "not an action selection token" in state_revision["description"]

    timeout = schemas["openhcs_get_viewer_window_state"]["properties"]["timeout_ms"]
    integer_schema = next(
        option for option in timeout["anyOf"] if option.get("type") == "integer"
    )
    assert integer_schema["minimum"] == McpViewerTimeoutPolicy.min_ms
    assert integer_schema["maximum"] == McpViewerTimeoutPolicy.max_ms
    assert str(McpViewerTimeoutPolicy.max_ms) in timeout["description"]


def test_openhcs_metadata_preserves_null_component_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        OpenHCSMetadataHandler,
        "_load_metadata",
        lambda _self, _plate_path: {
            "wells": {
                "A01": None,
                "A02": "Treatment",
            }
        },
    )
    handler = object.__new__(OpenHCSMetadataHandler)

    assert handler.get_well_values("/plate") == {
        "A01": None,
        "A02": "Treatment",
    }
