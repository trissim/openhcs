"""Pipeline authoring DTOs for the OpenHCS agent API."""

from __future__ import annotations

from dataclasses import dataclass, field

from openhcs.agent.dto.common import AgentResultEnvelope, JsonObject
from openhcs.agent.dto.config import ConfigPatch, ConfigRef
from openhcs.agent.dto.functions import FunctionIdentity


@dataclass(frozen=True, slots=True)
class FunctionSpecRef(FunctionIdentity):
    kwargs: JsonObject = field(default_factory=dict)
    runtime_options: JsonObject = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FunctionStepSpec:
    step_id: str
    name: str
    functions: tuple[FunctionSpecRef, ...]
    description: str | None = None
    enabled: bool = True
    debug_pause: bool = False
    dtype_config: ConfigPatch | None = None
    processing_config: ConfigPatch | None = None
    source_bindings: JsonObject | None = None
    step_well_filter_config: ConfigPatch | None = None
    step_materialization_config: ConfigPatch | None = None
    napari_streaming_config: ConfigPatch | None = None
    fiji_streaming_config: ConfigPatch | None = None


@dataclass(frozen=True, slots=True)
class PipelineRef:
    pipeline_id: str
    uri: str


@dataclass(frozen=True, slots=True)
class PipelineConfigRefs:
    global_ref: ConfigRef | None = None
    pipeline_ref: ConfigRef | None = None


@dataclass(frozen=True, slots=True)
class PipelineSpec:
    schema_version: str
    pipeline_id: str
    steps: tuple[FunctionStepSpec, ...]
    config_refs: PipelineConfigRefs = field(default_factory=PipelineConfigRefs)


@dataclass(frozen=True, slots=True)
class PipelineValidationResult(AgentResultEnvelope):
    valid: bool
    pipeline_ref: PipelineRef
