"""Focused compact rendering tests for paused runtime debug inspection."""

from openhcs.agent.dto.common import AgentError
from openhcs.agent.dto.execution import (
    ExecutionConnectionSpec,
    RuntimeDebugInspectionResult,
)
from openhcs.core.debug_views import (
    DebugViewModel,
    DebugViewSection,
    DebugViewSectionKind,
    DebugViewTable,
    DebugViewTableProjection,
)
from openhcs.mcp.dev_client import McpDevCommandSpec, _build_parser, _calls_from_args
from openhcs.mcp.dev_client_rendering import (
    CatalogRenderOptions,
    McpDevOutputRenderer,
)
from openhcs.serialization.json import to_jsonable


DEBUG_SESSION_ID = "debug-session-renderer"


def _runtime_debug_response() -> dict:
    result = RuntimeDebugInspectionResult(
        schema_version="openhcs.agent.v1",
        connection=ExecutionConnectionSpec(port=7787),
        debug_session_id=DEBUG_SESSION_ID,
        view_model=DebugViewModel(
            title="Paused worker values",
            sections=(
                DebugViewSection(
                    kind=DebugViewSectionKind.RUNTIME_VALUES,
                    title="Runtime Values",
                    table=DebugViewTable(
                        columns=("key", "location", "value_type"),
                        rows=(
                            ("image-a", "/memory/image-a.npy", "ImageArray"),
                            ("image-b", "/memory/image-b.npy", "ImageArray"),
                            ("labels", "/memory/labels.npy", "ObjectLabelSet"),
                        ),
                        projection=DebugViewTableProjection.RUNTIME_VALUE_RECORDS,
                    ),
                ),
                DebugViewSection(
                    kind=DebugViewSectionKind.ERROR,
                    title="Worker note",
                    text="Worker failure details are available for inspection.",
                ),
            ),
        ),
    )
    return {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_debug_runtime_values",
                "mcp_error": False,
                "payloads": [to_jsonable(result)],
            }
        ],
    }


def _runtime_debug_error_response() -> dict:
    result = RuntimeDebugInspectionResult(
        schema_version="openhcs.agent.v1",
        connection=ExecutionConnectionSpec(port=7787),
        debug_session_id=DEBUG_SESSION_ID,
        errors=(
            AgentError(
                code="runtime_debug_inspection_error",
                message="debug worker is not paused",
                hint="Pass a paused debug_session_id.",
            ),
        ),
    )
    return {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_inspect_debug_runtime_values",
                "mcp_error": False,
                "payloads": [to_jsonable(result)],
            }
        ],
    }


def test_runtime_debug_renderer_filters_and_bounds_declared_table_rows() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        (
            "runtime-debug-values",
            "7787",
            DEBUG_SESSION_ID,
            "--contains",
            "image",
            "--limit",
            "1",
        )
    )

    call = _calls_from_args(args)[0]
    assert call.name == "openhcs_inspect_debug_runtime_values"
    assert "contains" not in call.arguments
    assert "limit" not in call.arguments

    rendered = McpDevCommandSpec.for_name("runtime-debug-values").render_response(
        _runtime_debug_response(),
        args,
    )

    assert (
        "Runtime debug: session=debug-session-renderer "
        "endpoint=localhost:7787 transport=<none> persistent=True "
        'title="Paused worker values"'
    ) in rendered
    assert "Sections: total=2 matched=1 shown=1" in rendered
    assert "Items: total=4 matched=2 shown=1 truncated=1 limit=1" in rendered
    assert 'Section: kind=runtime_values title="Runtime Values"' in rendered
    assert "Columns (3): key | location | value_type" in rendered
    assert "Rows: total=3 matched=2 shown=1 truncated=1" in rendered
    assert "- image-a | /memory/image-a.npy | ImageArray" in rendered
    assert "image-b" not in rendered
    assert "Worker failure details" not in rendered


def test_runtime_debug_renderer_projects_section_text_without_kind_dispatch() -> None:
    binding = McpDevOutputRenderer.for_output_contract(RuntimeDebugInspectionResult)

    assert binding is not None
    assert binding.renderer_type.render_options_type is CatalogRenderOptions
    rendered = binding.render_with_options(
        _runtime_debug_response(),
        CatalogRenderOptions(contains="failure details", limit=1),
    )

    assert "Sections: total=2 matched=1 shown=1" in rendered
    assert 'Section: kind=error title="Worker note"' in rendered
    assert "Text: chars=52 shown=52 truncated=0" in rendered
    assert "- Worker failure details are available for inspection." in rendered
    assert "Columns" not in rendered


def test_runtime_debug_renderer_preserves_errors_and_raw_json_escape_hatch() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ("runtime-debug-values", "7787", DEBUG_SESSION_ID)
    )
    rendered = McpDevCommandSpec.for_name("runtime-debug-values").render_response(
        _runtime_debug_error_response(),
        args,
    )

    assert rendered.startswith("Runtime debug: unavailable\n")
    assert (
        "- runtime_debug_inspection_error: debug worker is not paused "
        'hint="Pass a paused debug_session_id."'
    ) in rendered
    assert '"view_model"' not in rendered

    json_args = parser.parse_args(
        ("runtime-debug-values", "7787", DEBUG_SESSION_ID, "--json")
    )
    raw_rendered = McpDevCommandSpec.for_name("runtime-debug-values").render_response(
        _runtime_debug_response(),
        json_args,
    )
    assert '"view_model"' in raw_rendered
    assert '"sections"' in raw_rendered
