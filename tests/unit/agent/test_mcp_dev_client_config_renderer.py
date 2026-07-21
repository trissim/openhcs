"""Focused dev-client rendering tests for reflected configuration schemas."""

from openhcs.agent.dto.config import ConfigSchema
from openhcs.mcp.dev_client import McpDevCommandSpec, _build_parser, _calls_from_args
from openhcs.mcp.dev_client_rendering import (
    CatalogRenderOptions,
    McpDevOutputRenderer,
)


def _config_schema_response() -> dict:
    return {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_describe_config_schema",
                "mcp_error": False,
                "payloads": [
                    {
                        "schema_version": "openhcs.agent.v1",
                        "config_type": "PipelineConfig",
                        "path_prefix": None,
                        "authoring_path": "ConfigPatch.values",
                        "fields": [
                            {
                                "path": "well_filter_config",
                                "type_repr": "LazyWellFilterConfig | None",
                                "default_repr": "None",
                                "required": False,
                                "description": "Filter wells before execution.",
                                "enum_values": [],
                                "registry_values": [
                                    "one",
                                    "two",
                                    "three",
                                    "four",
                                    "five",
                                    "six",
                                    "seven",
                                    "eight",
                                    "nine",
                                    "ten",
                                ],
                                "value_type_repr": "LazyWellFilterConfig",
                                "ui_hidden": False,
                                "lazy": True,
                                "inheritable": True,
                                "nested_schema_path": "well_filter_config",
                            },
                            {
                                "path": "fiji_streaming_config",
                                "type_repr": "LazyFijiStreamingConfig | None",
                                "default_repr": "None",
                                "required": False,
                                "description": "Stream selected results to Fiji.",
                                "enum_values": [],
                                "registry_values": [],
                                "value_type_repr": "LazyFijiStreamingConfig",
                                "ui_hidden": False,
                                "lazy": True,
                                "inheritable": True,
                                "nested_schema_path": "fiji_streaming_config",
                            },
                            {
                                "path": "default_component",
                                "type_repr": "AllComponents",
                                "default_repr": "AllComponents.CHANNEL",
                                "required": False,
                                "description": "Default source component.",
                                "enum_values": ["channel", "z_index"],
                                "registry_values": [],
                                "value_type_repr": None,
                                "ui_hidden": False,
                                "lazy": False,
                                "inheritable": False,
                                "nested_schema_path": None,
                            },
                        ],
                        "registries": [
                            {
                                "owner_type": "PipelineConfig",
                                "registered_types": [],
                            }
                        ],
                        "types": [
                            {
                                "type_repr": "LazyWellFilterConfig",
                                "description": "Well filtering configuration.",
                                "base_types": [],
                            }
                        ],
                    }
                ],
            }
        ],
    }


def _config_schema_error_response() -> dict:
    return {
        "errors": [],
        "results": [
            {
                "tool": "openhcs_describe_config_schema",
                "mcp_error": False,
                "payloads": [
                    {
                        "errors": [
                            {
                                "code": "mcp_server_stale",
                                "message": (
                                    "The OpenHCS MCP server source changed after "
                                    "this process started."
                                ),
                                "hint": "Restart the MCP client/server process.",
                            }
                        ],
                        "ok": False,
                        "restart_required": True,
                    }
                ],
            }
        ],
    }


def test_config_schema_renderer_is_keyed_by_nominal_output_contract() -> None:
    binding = McpDevOutputRenderer.for_output_contract(ConfigSchema)

    assert binding is not None
    assert binding.renderer_type.render_options_type is CatalogRenderOptions


def test_config_schema_renderer_preserves_shared_response_errors() -> None:
    binding = McpDevOutputRenderer.for_output_contract(ConfigSchema)

    assert binding is not None
    rendered = binding.render_with_options(
        _config_schema_error_response(),
        CatalogRenderOptions(),
    )

    assert rendered.startswith("Config schema: unavailable\n")
    assert (
        "- mcp_server_stale: The OpenHCS MCP server source changed after this "
        "process started. hint=\"Restart the MCP client/server process.\""
    ) in rendered
    assert "type=<none>" not in rendered

    transport_rendered = binding.render_with_options(
        {
            "errors": [
                {
                    "code": "mcp_transport_failed",
                    "message": "MCP stdio exchange ended during source reload.",
                    "hint": "Retry with a fresh server process.",
                }
            ],
            "results": [],
        },
        CatalogRenderOptions(),
    )
    assert transport_rendered.startswith("Config schema: unavailable\n")
    assert (
        "- mcp_transport_failed: MCP stdio exchange ended during source reload. "
        'hint="Retry with a fresh server process."'
    ) in transport_rendered


def test_config_schema_renderer_filters_and_bounds_reflected_fields() -> None:
    binding = McpDevOutputRenderer.for_output_contract(ConfigSchema)

    assert binding is not None
    rendered = binding.render_with_options(
        _config_schema_response(),
        CatalogRenderOptions(contains="lazy", limit=1),
    )

    assert (
        "Config schema: type=PipelineConfig path=<root> "
        "authoring=ConfigPatch.values"
    ) in rendered
    assert "Fields: total=3 matched=2 shown=1 registries=1 types=1" in rendered
    assert "Filter: contains=lazy" in rendered
    assert (
        "- well_filter_config: LazyWellFilterConfig | None "
        "value=LazyWellFilterConfig default=None "
        "flags=optional,lazy,inheritable nested=well_filter_config"
    ) in rendered
    assert "fiji_streaming_config" not in rendered
    assert "...<truncated 1 fields>" in rendered
    assert "Type inheritance (declaration-derived):" in rendered
    assert "- LazyWellFilterConfig extends=<none>" in rendered


def test_config_schema_renderer_declares_compact_catalog_defaults() -> None:
    binding = McpDevOutputRenderer.for_output_contract(ConfigSchema)

    assert binding is not None
    assert binding.default_cli_argument_values() == {
        "contains": None,
        "limit": 20,
    }
    rendered = binding.render_with_options(
        _config_schema_response(),
        CatalogRenderOptions(),
    )

    assert "Fields: total=3 matched=3 shown=3 registries=1 types=1" in rendered
    assert "enum=channel,z_index" in rendered
    assert "registry=one,two,three,four,five,six,seven,eight,+2" in rendered
    assert '"payloads"' not in rendered
    assert "Type inheritance (declaration-derived):" in rendered


def test_generated_config_schema_command_projects_request_and_render_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        (
            "config-schema",
            "pipeline",
            "--path-prefix",
            "fiji_streaming_config",
            "--contains",
            "lazy",
            "--limit",
            "1",
        )
    )

    call = _calls_from_args(args)[0]

    assert call.name == "openhcs_describe_config_schema"
    assert call.arguments == {
        "config_type": "pipeline",
        "path_prefix": "fiji_streaming_config",
    }
    rendered = McpDevCommandSpec.for_name("config-schema").render_response(
        _config_schema_response(),
        args,
    )
    assert "Fields: total=3 matched=2 shown=1 registries=1 types=1" in rendered
    assert "...<truncated 1 fields>" in rendered

    call_args = parser.parse_args(
        (
            "call",
            "openhcs_describe_config_schema",
            "--arguments",
            '{"config_type":"pipeline"}',
        )
    )
    call_rendered = McpDevCommandSpec.for_name("call").render_response(
        _config_schema_response(),
        call_args,
    )
    assert "Fields: total=3 matched=3 shown=3 registries=1 types=1" in call_rendered
    assert '"payloads"' not in call_rendered
