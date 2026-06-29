"""Configuration DTOs for the OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.agent.dto.common import AgentResultEnvelope, JsonObject


@dataclass(frozen=True, slots=True)
class ConfigTypeRef:
    config_type: str


@dataclass(frozen=True, slots=True)
class ConfigRef(ConfigTypeRef):
    config_id: str
    uri: str


@dataclass(frozen=True, slots=True)
class ConfigFieldSchema:
    path: str
    type_repr: str
    default_repr: str | None
    required: bool
    description: str | None
    enum_values: tuple[str, ...] = ()
    ui_hidden: bool = False
    lazy: bool = False


@dataclass(frozen=True, slots=True)
class ConfigSchema(ConfigTypeRef):
    schema_version: str
    fields: tuple[ConfigFieldSchema, ...]


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
