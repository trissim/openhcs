"""Typed request signatures for ZMQ execution and debug replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from zmqruntime.messages import ExecuteRequest, MessageFields


@dataclass(frozen=True, slots=True)
class ZMQExecutionRequestPayload:
    """Normalized execution request fields used by server execution phases."""

    plate_id: str
    pipeline_code: str
    config_params: dict | None
    config_code: str | None
    pipeline_config_code: str | None
    client_address: str | None = None
    compile_only: bool = False
    compile_artifact_id: str | None = None

    @classmethod
    def from_execute_request(
        cls,
        request: ExecuteRequest,
    ) -> "ZMQExecutionRequestPayload":
        return cls(
            plate_id=request.plate_id,
            pipeline_code=request.pipeline_code,
            config_params=request.config_params,
            config_code=request.config_code,
            pipeline_config_code=request.pipeline_config_code,
            client_address=request.client_address,
            compile_only=request.compile_only,
            compile_artifact_id=request.compile_artifact_id,
        )

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
        payload = {
            MessageFields.PLATE_ID: self.plate_id,
            MessageFields.PIPELINE_CODE: self.pipeline_code,
            MessageFields.CONFIG_PARAMS: config_params,
            MessageFields.CONFIG_CODE: self.config_code,
            MessageFields.PIPELINE_CONFIG_CODE: self.pipeline_config_code,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

