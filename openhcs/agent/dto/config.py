"""Configuration DTOs for the OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

from openhcs.agent.dto.common import (
    AgentCliArgumentSpec,
    AgentCliRequest,
    AgentResultEnvelope,
    JsonObject,
)


@dataclass(frozen=True, slots=True)
class ConfigTypeRef:
    config_type: str


@dataclass(frozen=True, slots=True)
class ConfigSchemaRequest(ConfigTypeRef, AgentCliRequest):
    """Select one owner-derived config or FunctionStep schema subtree."""

    path_prefix: str | None = None

    @classmethod
    def agent_cli_argument_specs(cls) -> tuple[AgentCliArgumentSpec, ...]:
        return (
            AgentCliArgumentSpec(
                field_name="config_type",
                positional=True,
                help=(
                    "Configuration owner: global, pipeline, step, or read-only ui."
                ),
            ),
            AgentCliArgumentSpec(
                field_name="path_prefix",
                flags=("--path-prefix",),
                help=(
                    "Optional nested_schema_path returned by the top-level "
                    "schema, such as napari_streaming_config."
                ),
            ),
        )

    @classmethod
    def from_fields(
        cls,
        *,
        config_type: str,
        path_prefix: str | None = None,
    ) -> Self:
        return cls(config_type=config_type, path_prefix=path_prefix)

    def as_tool_arguments(self) -> JsonObject:
        return {
            "config_type": self.config_type,
            "path_prefix": self.path_prefix,
        }


@dataclass(frozen=True, slots=True)
class ConfigRef(ConfigTypeRef):
    config_id: str
    uri: str


@dataclass(frozen=True, slots=True)
class ConfigFieldSchema:
    """One reflected field and its exact nested authoring location.

    ``path`` is a catalog-navigation path. ``authoring_value_path`` is the
    sequence of object keys (plus ``[]`` collection markers) used to construct
    the nested JSON value accepted by the owning mutation request.
    """

    path: str
    type_repr: str
    default_repr: str | None
    required: bool
    description: str | None
    authoring_value_path: tuple[str, ...] = ()
    enum_values: tuple[str, ...] = ()
    registry_values: tuple[str, ...] = ()
    value_type_repr: str | None = None
    ui_hidden: bool = False
    lazy: bool = False
    inheritable: bool = False
    declaring_type: str | None = None
    default_origin: str | None = None
    nested_schema_path: str | None = None


@dataclass(frozen=True, slots=True)
class ConfigRegisteredType:
    """One concrete config registered by an authoritative nominal root."""

    key: str
    type_repr: str


@dataclass(frozen=True, slots=True)
class ConfigRegistrySchema:
    """JSON-safe projection of one authoritative config registry."""

    owner_type: str
    registered_types: tuple[ConfigRegisteredType, ...]


@dataclass(frozen=True, slots=True)
class ConfigTypeSchema:
    """Meaning and inheritance for one dataclass type referenced by fields."""

    type_repr: str
    description: str | None
    base_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfigSchema(ConfigTypeRef):
    schema_version: str
    fields: tuple[ConfigFieldSchema, ...]
    path_prefix: str | None = None
    authoring_path: str = "ConfigPatch.values"
    registries: tuple[ConfigRegistrySchema, ...] = ()
    types: tuple[ConfigTypeSchema, ...] = ()


@dataclass(frozen=True, slots=True)
class ConfigPatch(ConfigTypeRef):
    values: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConfigSourceRenderRequest:
    config_id: str
    clean: bool = True


@dataclass(frozen=True, slots=True)
class ConfigValidationResult(AgentResultEnvelope):
    valid: bool
    config_ref: ConfigRef | None = None
