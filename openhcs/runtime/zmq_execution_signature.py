"""Typed request signatures for ZMQ execution and debug replay."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json

from metaclass_registry import AutoRegisterMeta

from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from zmqruntime.messages import ExecuteRequest, MessageFields

TransportValue = (
    str
    | int
    | float
    | bool
    | None
    | Mapping[str, "TransportValue"]
    | tuple["TransportValue", ...]
)
TransportRequestItems = tuple[tuple[str, TransportValue], ...]
EXECUTION_PLATE_ID_FIELD = "execution_plate_id"
SELECTED_PIPELINE_PATH_FIELD = "selected_pipeline_path"


@dataclass(frozen=True, slots=True)
class OpenHCSExecutionConfigBundle:
    """OpenHCS config pair shared by client submissions and server execution."""

    global_pipeline: GlobalPipelineConfig
    plate_pipeline: PipelineConfig | None = None

    def with_global_pipeline(
        self,
        global_pipeline: GlobalPipelineConfig,
    ) -> "OpenHCSExecutionConfigBundle":
        return OpenHCSExecutionConfigBundle(
            global_pipeline=global_pipeline,
            plate_pipeline=self.plate_pipeline,
        )


class OpenHCSExecutionConfigCarrier(ABC, metaclass=AutoRegisterMeta):
    """Mixin for records carrying OpenHCS execution config bundles."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @property
    @abstractmethod
    def execution_config_bundle(self) -> OpenHCSExecutionConfigBundle:
        raise NotImplementedError

    @property
    def global_config(self) -> GlobalPipelineConfig:
        return self.execution_config_bundle.global_pipeline

    @property
    def pipeline_config(self) -> PipelineConfig | None:
        return self.execution_config_bundle.plate_pipeline


@dataclass(frozen=True, slots=True)
class ZMQExecutionIdentity:
    """Plate and source-selection identity shared by client and server."""

    plate_id: str
    execution_plate_id: str | None = None
    selected_pipeline_path: str | None = None

    def request_items(self) -> TransportRequestItems:
        items: list[tuple[str, TransportValue]] = [
            (MessageFields.PLATE_ID, self.plate_id),
        ]
        if self.execution_plate_id is not None:
            items.append((EXECUTION_PLATE_ID_FIELD, self.execution_plate_id))
        if self.selected_pipeline_path is not None:
            items.append(
                (SELECTED_PIPELINE_PATH_FIELD, self.selected_pipeline_path)
            )
        return tuple(items)

    def signature_items(self) -> TransportRequestItems:
        return (
            (MessageFields.PLATE_ID, self.plate_id),
            (EXECUTION_PLATE_ID_FIELD, self.execution_plate_id),
            (SELECTED_PIPELINE_PATH_FIELD, self.selected_pipeline_path),
        )


@dataclass(frozen=True, slots=True)
class ZMQExecutionCompileControl:
    """Compile-mode controls shared by client and server requests."""

    compile_artifact_id: str | None = None
    compile_only: bool = False

    @classmethod
    def from_execute_request(
        cls,
        request: ExecuteRequest,
    ) -> "ZMQExecutionCompileControl":
        return cls(
            compile_artifact_id=request.compile_artifact_id,
            compile_only=request.compile_only,
        )

    def as_compile_request(self) -> "ZMQExecutionCompileControl":
        return ZMQExecutionCompileControl(
            compile_artifact_id=self.compile_artifact_id,
            compile_only=True,
        )

    def validate(self) -> None:
        if self.compile_only and self.compile_artifact_id:
            raise ValueError("compile_only and compile_artifact_id cannot both be set")

    def request_items(self) -> TransportRequestItems:
        items: list[tuple[str, TransportValue]] = []
        if self.compile_only:
            items.append((MessageFields.COMPILE_ONLY, True))
        if self.compile_artifact_id is not None:
            items.append((MessageFields.COMPILE_ARTIFACT_ID, self.compile_artifact_id))
        return tuple(items)


@dataclass(frozen=True, slots=True)
class ZMQExecutionConfigTransport:
    """Config transport authority for execution request signatures."""

    config_params: dict | None = None
    config_code: str | None = None
    pipeline_config_code: str | None = None

    @classmethod
    def from_execute_request(
        cls,
        request: ExecuteRequest,
    ) -> "ZMQExecutionConfigTransport":
        return cls(
            config_params=request.config_params,
            config_code=request.config_code,
            pipeline_config_code=request.pipeline_config_code,
        )

    def signature_items(self, config_params: dict | None) -> TransportRequestItems:
        return (
            (MessageFields.CONFIG_PARAMS, config_params),
            (MessageFields.CONFIG_CODE, self.config_code),
            (MessageFields.PIPELINE_CONFIG_CODE, self.pipeline_config_code),
        )


@dataclass(frozen=True, slots=True)
class ZMQExecutionRequestPayload:
    """Normalized execution request fields used by server execution phases."""

    identity: ZMQExecutionIdentity
    pipeline_code: str
    config_transport: ZMQExecutionConfigTransport
    compile_control: ZMQExecutionCompileControl
    client_address: str | None = None

    @classmethod
    def from_execute_request(
        cls,
        request: ExecuteRequest,
    ) -> "ZMQExecutionRequestPayload":
        return cls(
            identity=ZMQExecutionIdentity(
                plate_id=request.plate_id,
                execution_plate_id=request.execution_plate_id,
                selected_pipeline_path=request.selected_pipeline_path,
            ),
            pipeline_code=request.pipeline_code,
            config_transport=ZMQExecutionConfigTransport.from_execute_request(request),
            compile_control=ZMQExecutionCompileControl.from_execute_request(request),
            client_address=request.client_address,
        )

    @property
    def plate_id(self) -> str:
        return self.identity.plate_id

    @property
    def execution_plate_id(self) -> str | None:
        return self.identity.execution_plate_id

    @property
    def selected_pipeline_path(self) -> str | None:
        return self.identity.selected_pipeline_path

    @property
    def config_params(self) -> dict | None:
        return self.config_transport.config_params

    @property
    def config_code(self) -> str | None:
        return self.config_transport.config_code

    @property
    def pipeline_config_code(self) -> str | None:
        return self.config_transport.pipeline_config_code

    @property
    def compile_only(self) -> bool:
        return self.compile_control.compile_only

    @property
    def compile_artifact_id(self) -> str | None:
        return self.compile_control.compile_artifact_id

    @property
    def request_signature(self) -> str:
        return self.signature_for_config_params(self.config_params)

    @property
    def debug_replay_signature(self) -> str:
        from openhcs.core.debug import DebugExecutionConfig

        return self.signature_for_config_params(
            DebugExecutionConfig.compatibility_config_params(self.config_params)
        )

    @property
    def pipeline_sha(self) -> str:
        return hashlib.sha256(self.pipeline_code.encode("utf-8")).hexdigest()[:12]

    def signature_for_config_params(self, config_params: dict | None) -> str:
        payload = dict(
            self.identity.signature_items()
            + ((MessageFields.PIPELINE_CODE, self.pipeline_code),)
            + self.config_transport.signature_items(config_params)
        )
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
