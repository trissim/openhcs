from __future__ import annotations

import asyncio

import pytest

from openhcs.agent.capabilities import (
    DescribeConfigSchemaCapability,
    DescribeInternalSymbolCapability,
    ExplainArchitectureCapability,
    ListArchitectureTopicsCapability,
)
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.services.architecture_projection_service import (
    ArchitectureProjectionService,
    ArchitectureSymbolNotCuratedError,
)
from openhcs.agent.services.config_service import ConfigService
from openhcs.mcp import server


def _structured_result(result) -> dict:
    return result[1] if isinstance(result, tuple) else result.structuredContent


def test_unknown_architecture_symbol_reports_curated_discovery_route() -> None:
    service = ArchitectureProjectionService()
    known = service.describe_internal_symbol("core.GlobalPipelineConfig")

    assert "curated architecture topics" in (
        DescribeInternalSymbolCapability.description
    )
    assert (
        ListArchitectureTopicsCapability.name
        in DescribeInternalSymbolCapability.description
    )
    assert (
        ExplainArchitectureCapability.name
        in DescribeInternalSymbolCapability.description
    )

    with pytest.raises(ArchitectureSymbolNotCuratedError) as error:
        service.describe_internal_symbol(known.import_path)

    projected = error.value.to_agent_error()
    assert projected.code == "architecture_symbol_not_curated"
    assert "curated architecture namespace" in projected.message
    assert "Arbitrary Python import paths are not accepted" in projected.message
    assert ListArchitectureTopicsCapability.name in (projected.hint or "")
    assert ExplainArchitectureCapability.name in (projected.hint or "")
    assert "core.GlobalPipelineConfig" in (projected.hint or "")
    assert "pipeline_model" in (projected.hint or "")


def test_config_patch_root_near_miss_uses_reflected_authoring_path() -> None:
    service = ConfigService()

    result = service.validate_patch(
        "global",
        ConfigPatch(config_type="global", values={"num_worker": 2}),
    )

    assert result.valid is False
    error = result.errors[0]
    assert error.code == "config_patch_unknown_field"
    assert error.exception_type == "ConfigPatchUnknownFieldError"
    assert error.path == 'ConfigPatch.values["num_worker"]'
    assert 'ConfigPatch.values["num_workers"]' in (error.hint or "")
    assert DescribeConfigSchemaCapability.name in (error.hint or "")
    assert "unexpected keyword argument" not in error.message


def test_config_patch_nested_collection_near_miss_uses_json_shape() -> None:
    service = ConfigService()

    result = service.validate_patch(
        "pipeline",
        ConfigPatch(
            config_type="pipeline",
            values={
                "source_bindings_config": {
                    "bindings": [
                        {
                            "alias": "DNA",
                            "artifct_kind": "image",
                        }
                    ]
                }
            },
        ),
    )

    assert result.valid is False
    error = result.errors[0]
    assert error.path == (
        'ConfigPatch.values["source_bindings_config"]["bindings"][]'
        '["artifct_kind"]'
    )
    assert (
        'ConfigPatch.values["source_bindings_config"]["bindings"][]'
        '["artifact_kind"]'
    ) in (error.hint or "")


def test_mcp_projects_typed_architecture_and_config_diagnostics() -> None:
    built = server.build_server()

    architecture_result = asyncio.run(
        built.call_tool(
            "openhcs_describe_internal_symbol",
            {"symbol_id": "openhcs.core.config.GlobalPipelineConfig"},
        )
    )
    config_result = asyncio.run(
        built.call_tool(
            "openhcs_validate_config_patch",
            {
                "config_type": "global",
                "values": {"num_worker": 2},
            },
        )
    )

    architecture_payload = _structured_result(architecture_result)
    config_payload = _structured_result(config_result)
    architecture_error = architecture_payload["errors"][0]
    config_error = config_payload["errors"][0]

    assert architecture_error["code"] == "architecture_symbol_not_curated"
    assert ListArchitectureTopicsCapability.name in architecture_error["hint"]
    assert architecture_error["exception_type"] == (
        "ArchitectureSymbolNotCuratedError"
    )
    assert config_payload["valid"] is False
    assert config_error["code"] == "config_patch_unknown_field"
    assert DescribeConfigSchemaCapability.name in config_error["hint"]
    assert 'ConfigPatch.values["num_workers"]' in config_error["hint"]
