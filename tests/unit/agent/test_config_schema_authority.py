from __future__ import annotations

import inspect
from dataclasses import fields
from pathlib import Path

import pytest

from openhcs.agent.dto.authoring import AuthoringContextRequest
from openhcs.agent.dto.config import ConfigPatch, ConfigSchemaRequest
from openhcs.agent.services.config_service import ConfigService
from openhcs.core.artifacts import ArtifactType, ImageArtifactType
from openhcs.core.config import (
    GlobalPipelineConfig,
    PathPlanningConfig,
    PipelineConfig,
    StepMaterializationConfig,
    StreamingConfig,
)
from openhcs.core.source_bindings import NamedSourceBinding
from openhcs.core.steps.abstract import AbstractStep
from openhcs.serialization.json import to_jsonable


def _field_by_path(schema, path: str):
    return next(field for field in schema.fields if field.path == path)


def _type_repr(value_type: type) -> str:
    return f"{value_type.__module__}.{value_type.__qualname__}"


def test_pipeline_schema_projects_effective_default_and_owner_provenance():
    service = ConfigService()

    root = service.describe_schema("pipeline")
    materialization = service.describe_schema(
        "pipeline",
        "step_materialization_config",
    )

    assert len(root.fields) == len(fields(PipelineConfig))
    assert root.authoring_path == "ConfigPatch.values"
    assert _field_by_path(root, "num_workers").default_repr == "1"
    assert _field_by_path(root, "materialization_results_path").default_repr == (
        "Path('results')"
    )
    assert _field_by_path(root, "microscope").default_repr == "Microscope.AUTO"
    assert _field_by_path(root, "num_workers").default_origin == "inherited_default"
    assert _field_by_path(root, "num_workers").inheritable is True
    assert _field_by_path(root, "num_workers").declaring_type == _type_repr(
        GlobalPipelineConfig
    )

    output_suffix = _field_by_path(
        materialization,
        "step_materialization_config.output_dir_suffix",
    )
    sub_dir = _field_by_path(
        materialization,
        "step_materialization_config.sub_dir",
    )
    assert output_suffix.default_repr == repr(PathPlanningConfig().output_dir_suffix)
    assert output_suffix.default_origin == "inherited_default"
    assert output_suffix.declaring_type == _type_repr(PathPlanningConfig)
    assert sub_dir.default_repr == repr(StepMaterializationConfig().sub_dir)
    assert sub_dir.default_origin == "field_default"
    assert sub_dir.declaring_type == _type_repr(StepMaterializationConfig)

    materialization_type = next(
        type_schema
        for type_schema in materialization.types
        if type_schema.type_repr == _type_repr(StepMaterializationConfig)
    )
    assert materialization_type.description.startswith(
        "Configuration for persistent copies of a step's ordinary main-flow result"
    )
    assert materialization_type.base_types == tuple(
        _type_repr(base_type)
        for base_type in StepMaterializationConfig.__bases__
        if base_type is not object
    )


def test_config_schema_request_owns_generated_cli_projection() -> None:
    request = ConfigSchemaRequest.from_fields(
        config_type="step",
        path_prefix="napari_streaming_config",
    )

    assert request.as_tool_arguments() == {
        "config_type": "step",
        "path_prefix": "napari_streaming_config",
    }
    specs = request.agent_cli_argument_specs()
    assert specs[0].field_name == "config_type"
    assert specs[0].positional is True
    assert specs[1].field_name == "path_prefix"
    assert specs[1].flags == ("--path-prefix",)


def test_ui_schema_is_reflected_but_not_draftable() -> None:
    service = ConfigService()

    schema = service.describe_schema("ui_config", "logging")

    assert schema.config_type == "UIConfig"
    assert schema.authoring_path == "ObjectState[openhcs.pyqt_gui.config.UIConfig]"
    assert _field_by_path(schema, "logging.level").default_repr == "GuiLogLevel.INFO"
    assert "root logging handlers" in (
        _field_by_path(schema, "logging.level").description or ""
    )

    with pytest.raises(ValueError, match="config_type must be one of"):
        service.create("ui")


def test_function_step_schema_is_the_exact_abstract_step_config_projection():
    service = ConfigService()

    schema = service.describe_schema("step")
    signature = inspect.signature(AbstractStep.__init__)

    assert schema.config_type == "FunctionStep"
    assert schema.authoring_path == "FunctionStepAddRequest.step_config_overrides"
    assert tuple(field.path for field in schema.fields) == tuple(
        AbstractStep.config_classes_by_field_name()
    )
    assert tuple(field.path for field in schema.fields) == tuple(
        name
        for name, parameter in signature.parameters.items()
        if name in AbstractStep.config_classes_by_field_name()
        and parameter.kind is inspect.Parameter.KEYWORD_ONLY
    )
    assert _field_by_path(schema, "source_bindings").nested_schema_path == (
        "source_bindings"
    )
    assert "Pipeline-level sequential processing" in (
        _field_by_path(schema, "processing_config").description or ""
    )
    assert "Shared viewer enablement" in (
        _field_by_path(schema, "streaming_defaults").description or ""
    )

    with pytest.raises(ValueError, match="config_type must be one of"):
        service.create("step")


def test_streaming_schema_projects_the_authoritative_registry_only_when_relevant():
    service = ConfigService()

    napari = service.describe_schema("pipeline", "napari_streaming_config")
    processing = service.describe_schema("pipeline", "processing_config")

    assert len(napari.registries) == 1
    registry = napari.registries[0]
    assert registry.owner_type == _type_repr(StreamingConfig)
    assert tuple(
        (registered.key, registered.type_repr)
        for registered in registry.registered_types
    ) == tuple(
        (
            key,
            _type_repr(StreamingConfig.config_type_for_key(key)),
        )
        for key in StreamingConfig.supported_config_keys()
    )
    assert processing.registries == ()


def test_source_binding_schema_and_patch_use_nominal_registry_and_json_types():
    service = ConfigService()
    schema = service.describe_schema(
        "pipeline",
        "source_bindings_config.bindings",
    )

    artifact_kind = _field_by_path(
        schema,
        "source_bindings_config.bindings[].artifact_kind",
    )
    assert artifact_kind.registry_values == tuple(
        str(key) for key in ArtifactType.__registry__
    )
    named_binding_type = next(
        type_schema
        for type_schema in schema.types
        if type_schema.type_repr == _type_repr(NamedSourceBinding)
    )
    assert named_binding_type.description.startswith("Name selected image planes")

    config_ref = service.create(
        "pipeline",
        ConfigPatch(
            config_type="PipelineConfig",
            values={
                "path_planning_config": {
                    "global_output_folder": "agent-results",
                },
                "source_bindings_config": {
                    "bindings": [
                        {
                            "alias": "nuclei",
                            "artifact_kind": "image",
                            "source_channel_axis": 2,
                            "source_channel_counts": [1, 3],
                        }
                    ]
                },
            },
        ),
    )
    config = service.resolve_ref(config_ref)
    binding = config.source_bindings_config.bindings[0]

    assert config.path_planning_config.global_output_folder == Path("agent-results")
    assert binding.artifact_kind is ImageArtifactType
    assert binding.source_channel_counts == frozenset((1, 3))


def test_authoring_context_request_uses_bounded_progressive_default():
    assert AuthoringContextRequest().max_chars == 16_000


def test_public_processing_schema_response_is_json_safe_and_self_routing():
    schema = ConfigService().describe_schema_request(
        ConfigSchemaRequest(
            config_type="pipeline",
            path_prefix="processing_config",
        )
    )

    payload = to_jsonable(schema)
    fields_by_path = {field["path"]: field for field in payload["fields"]}

    assert payload["config_type"] == "PipelineConfig"
    assert payload["path_prefix"] == "processing_config"
    assert payload["authoring_path"] == "ConfigPatch.values"
    assert fields_by_path["processing_config"]["nested_schema_path"] == (
        "processing_config"
    )
    assert fields_by_path["processing_config.variable_components"][
        "enum_values"
    ] == ["site", "channel", "z_index", "timepoint"]
    assert fields_by_path["processing_config.group_by"]["enum_values"][-1] == (
        "NONE"
    )
