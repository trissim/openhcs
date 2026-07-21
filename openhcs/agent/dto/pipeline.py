"""Pipeline authoring DTOs for the OpenHCS agent API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from openhcs.agent.dto.common import AgentResultEnvelope, JsonObject
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.functions import FunctionIdentity


@dataclass(frozen=True, slots=True)
class FunctionSpecRef(FunctionIdentity):
    kwargs: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FunctionStepSpec:
    step_id: str
    name: str
    functions: tuple[FunctionSpecRef, ...]
    description: str | None = None
    enabled: bool = True
    debug_pause: bool = False
    step_config_overrides: Mapping[str, ConfigPatch] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FunctionStepAddRequest:
    """Add one registry-backed FunctionStep to an in-memory pipeline draft."""

    pipeline_id: str
    function_id: str
    name: str | None = None
    kwargs: JsonObject = field(default_factory=dict)
    step_config_overrides: JsonObject = field(default_factory=dict)
    step_id: str | None = None
    description: str | None = None
    enabled: bool = True
    debug_pause: bool = False
    index: int | None = None

    @classmethod
    def from_fields(
        cls,
        *,
        pipeline_id: str,
        function_id: str,
        name: str | None = None,
        kwargs: dict | None = None,
        step_config_overrides: dict | None = None,
        step_id: str | None = None,
        description: str | None = None,
        enabled: bool = True,
        debug_pause: bool = False,
        index: int | None = None,
    ) -> "FunctionStepAddRequest":
        return cls(
            pipeline_id=pipeline_id,
            function_id=function_id,
            name=name,
            kwargs=dict(kwargs or {}),
            step_config_overrides=dict(step_config_overrides or {}),
            step_id=step_id,
            description=description,
            enabled=enabled,
            debug_pause=debug_pause,
            index=index,
        )


@dataclass(frozen=True, slots=True)
class PipelineRef:
    pipeline_id: str
    uri: str


@dataclass(frozen=True, slots=True)
class CreatePipelineRequest:
    """Create one in-memory pipeline document from an optional config draft."""

    pipeline_config_id: str | None = None


@dataclass(frozen=True, slots=True)
class PipelineValidationRequest:
    pipeline_id: str


@dataclass(frozen=True, slots=True)
class PipelineSourceRenderRequest:
    pipeline_id: str
    clean: bool = True


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    schema_version: str
    pipeline_id: str
    pipeline_config_id: str
    steps: tuple[FunctionStepSpec, ...]


@dataclass(frozen=True, slots=True)
class PipelineValidationResult(AgentResultEnvelope):
    valid: bool
    pipeline_ref: PipelineRef
