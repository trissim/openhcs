"""Opaque execution sessions for OpenHCS agent integrations."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from itertools import count
from pathlib import Path
from typing import Self

from zmqruntime.execution import ExecutionProgressObservation
from zmqruntime.messages import ExecutionStatus

from openhcs.agent.dto.common import (
    SCHEMA_VERSION,
    AgentError,
    AgentWarning,
    JsonObject,
)
from openhcs.agent.dto.execution import (
    ArtifactInputPlanSummary,
    ArtifactMaterializationPathSummary,
    ArtifactMaterializationPlanSummary,
    ArtifactPlanInspection,
    ArtifactPlanSummary,
    CompiledStepPlanSummary,
    ExecutionConnectionSpec,
    ExecutionJobRef,
    ExecutionJobStatus,
    MainFlowMaterializationPlanSummary,
    OrchestratorSession,
    OrchestratorSessionCreationRequest,
    OrchestratorSessionRef,
    OrchestratorSessionRequest,
    PipelineSourceArtifactPlanInspectionRequest,
    PipelineSourceOrchestratorSessionRequest,
    SourceWorkspaceFileRecord,
    SourceWorkspaceSummary,
    ViewerStreamingPlanSummary,
    bounded_execution_status_response,
    execution_status_errors,
    execution_status_from_response,
    execution_status_warnings,
)
from openhcs.agent.exceptions import AgentFacingErrorMixin
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.pipeline_authoring_service import PipelineAuthoringService
from openhcs.core.compiled_execution import CompiledExecutionBundle
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.pipeline.path_planner import MissingArtifactInputError
from openhcs.core.pipeline_document import PipelineDocument, PipelineDocumentAuthority
from openhcs.core.progress import ProgressQueue
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
)
from openhcs.core.steps.function_artifact_materialization import (
    planned_materialization_preview,
)
from openhcs.microscopes.exceptions import MicroscopePixelSizeUnavailableError
from openhcs.microscopes.openhcs import OpenHCSMetadataHandler
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig
from openhcs.runtime.zmq_execution_client import (
    ExecutionSubmissionPreparationTimeoutError,
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity
from openhcs.serialization.json import to_jsonable

MAX_INSPECTION_AXES = 8
MAX_INSPECTION_STEPS = 24
MAX_INSPECTION_ARTIFACT_INPUTS_PER_STEP = 16
MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP = 16
MAX_INSPECTION_SOURCE_WORKSPACE_FILES = 64
logger = logging.getLogger(__name__)


class ExecutionSessionError(AgentFacingErrorMixin, ValueError):
    """Base class for execution-session failures intended for agents."""


class UnknownExecutionSessionIdError(ExecutionSessionError):
    """Raised when an execution session id is not present in this process."""

    agent_error_code = "unknown_execution_session_id"
    agent_error_hint = (
        "Create a session with openhcs_create_orchestrator_session or "
        "openhcs_create_orchestrator_session_from_pipeline_source in this same "
        "MCP server session, then reuse the returned session_id."
    )

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Unknown OpenHCS execution session_id: {session_id}")


class UnknownExecutionJobIdError(ExecutionSessionError):
    """Raised when an execution job id is not present in this process."""

    agent_error_code = "unknown_execution_job_id"
    agent_error_hint = (
        "Submit compile or run first, then pass the returned job_id to "
        "openhcs_get_execution_status."
    )

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        super().__init__(f"Unknown OpenHCS execution job_id: {job_id}")


class AgentProgressQueue(ProgressQueue):
    def __init__(self) -> None:
        self.events: list[JsonObject] = []

    def put(self, event) -> None:
        if isinstance(event, dict):
            self.events.append(dict(event))


@dataclass(frozen=True, slots=True)
class CompileInspectionInput:
    plate: Path
    pipeline_document: PipelineDocument
    axis_filter: tuple[str, ...]
    global_pipeline_config: GlobalPipelineConfig
    progress_queue: AgentProgressQueue


@dataclass(frozen=True, slots=True)
class CompileInspectionResult:
    """Compiler-owned bundle joined to its inspection-only source projection."""

    execution_bundle: CompiledExecutionBundle
    source_workspace_projection: VirtualWorkspaceSourceProjection


class CompileInspectionGatewayABC(ABC):
    @abstractmethod
    def compile(self, request: CompileInspectionInput) -> CompileInspectionResult:
        raise NotImplementedError


class InProcessCompileInspectionGateway(CompileInspectionGatewayABC):
    def compile(self, request: CompileInspectionInput) -> CompileInspectionResult:
        from objectstate.lazy_factory import ensure_global_config_context

        import openhcs.processing.func_registry as func_registry_module
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.core.progress import set_progress_queue

        ensure_global_config_context(
            GlobalPipelineConfig,
            request.global_pipeline_config,
        )
        with func_registry_module._registry_lock:
            if not func_registry_module._registry_initialized:
                func_registry_module._auto_initialize_registry()

        orchestrator = PipelineOrchestrator(
            plate_path=request.plate,
            pipeline_config=request.pipeline_document.pipeline_config,
            progress_callback=None,
        )
        orchestrator.initialize()
        set_progress_queue(request.progress_queue)
        try:
            execution_bundle = orchestrator.compile_pipelines(
                pipeline_definition=request.pipeline_document.pipeline_steps,
                well_filter=list(request.axis_filter) or None,
                is_zmq_execution=True,
            )
            return CompileInspectionResult(
                execution_bundle=execution_bundle,
                source_workspace_projection=(
                    orchestrator.source_workspace_projection()
                ),
            )
        finally:
            set_progress_queue(None)


@dataclass(frozen=True, slots=True)
class GlobalConfigSelection:
    config_id: str | None

    def resolve(self, config_service: ConfigService) -> GlobalPipelineConfig:
        if self.config_id is None:
            return GlobalPipelineConfig()
        config = config_service.resolve_ref(self.config_id)
        if not isinstance(config, GlobalPipelineConfig):
            raise TypeError("global_config_id must resolve to GlobalPipelineConfig")
        return config


class ExecutionClientABC(ABC):
    @abstractmethod
    def submit_compile(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def submit_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def wait_for_completion(self, execution_id: str) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def progress_observation(
        self,
        execution_id: str,
    ) -> ExecutionProgressObservation | None:
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        raise NotImplementedError


ExecutionSubmitter = Callable[
    [ExecutionClientABC, OpenHCSExecutionSubmission, int],
    JsonObject,
]


class ExecutionJobKind(Enum):
    COMPILE = (
        "compile",
        lambda client, submission, timeout_ms: client.submit_compile(
            submission,
            timeout_ms=timeout_ms,
        ),
    )
    EXECUTE = (
        "execute",
        lambda client, submission, timeout_ms: client.submit_pipeline(
            submission,
            timeout_ms=timeout_ms,
        ),
    )

    def __new__(
        cls,
        value: str,
        submitter: ExecutionSubmitter,
    ) -> Self:
        member = object.__new__(cls)
        member._value_ = value
        member._submitter = submitter
        return member

    def submit(
        self,
        client: ExecutionClientABC,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        """Submit through the operation owned by this job kind."""

        return self._submitter(client, submission, timeout_ms)


class ExecutionClientFactoryABC(ABC):
    @abstractmethod
    def create_client(
        self,
        connection: ExecutionConnectionSpec,
    ) -> ExecutionClientABC:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ZMQExecutionClientAdapter(ExecutionClientABC):
    client: ZMQExecutionClient

    def submit_compile(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
    ) -> JsonObject:
        return dict(self.client.submit_compile(submission, timeout_ms=timeout_ms))

    def submit_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
    ) -> JsonObject:
        return dict(self.client.submit_pipeline(submission, timeout_ms=timeout_ms))

    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> JsonObject:
        return dict(self.client.get_status(execution_id, timeout_ms=timeout_ms))

    def wait_for_completion(self, execution_id: str) -> JsonObject:
        return dict(self.client.wait_for_completion(execution_id))

    def progress_observation(
        self,
        execution_id: str,
    ) -> ExecutionProgressObservation | None:
        return self.client.progress_observation(execution_id)

    def disconnect(self) -> None:
        self.client.disconnect()


class ZMQExecutionClientFactory(ExecutionClientFactoryABC):
    def __init__(
        self,
        config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
    ) -> None:
        self._config = config

    def create_client(
        self,
        connection: ExecutionConnectionSpec,
    ) -> ZMQExecutionClientAdapter:
        return ZMQExecutionClientAdapter(connection.execution_client(self._config))


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionSessionCommonRequest:
    identity: ZMQExecutionIdentity
    global_config_id: str | None = None
    connection: ExecutionConnectionSpec = ExecutionConnectionSpec()


class ExecutionPipelineSessionRequest(
    ExecutionSessionCommonRequest,
    ABC,
):
    @abstractmethod
    def build_pipeline_definition(
        self,
        *,
        session_id: str,
        pipeline_service: PipelineAuthoringService,
    ) -> ExecutionPipelineDefinition:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ResolvedExecutionSessionInputs:
    plate: Path
    execution_plate: Path
    selected_pipeline: Path | None
    global_pipeline_config: GlobalPipelineConfig
    connection: ExecutionConnectionSpec

    @classmethod
    def from_request(
        cls,
        *,
        request: ExecutionSessionCommonRequest,
        path_policy: AgentPathPolicy,
        config_service: ConfigService,
    ) -> ResolvedExecutionSessionInputs:
        plate = path_policy.assert_readable(request.identity.plate_id)
        if request.identity.execution_plate_id is None:
            execution_plate = plate
        else:
            execution_plate = path_policy.assert_readable(
                request.identity.execution_plate_id
            )
        if request.identity.selected_pipeline_path is None:
            selected_pipeline = None
        else:
            selected_pipeline = path_policy.assert_readable(
                request.identity.selected_pipeline_path
            )
        return cls(
            plate=plate,
            execution_plate=execution_plate,
            selected_pipeline=selected_pipeline,
            global_pipeline_config=GlobalConfigSelection(
                request.global_config_id
            ).resolve(config_service),
            connection=request.connection,
        )


@dataclass(frozen=True, slots=True)
class ExecutionPipelineDefinition:
    pipeline_id: str
    pipeline_document: PipelineDocument
    pipeline_config_id: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DraftPipelineSessionRequest(ExecutionPipelineSessionRequest):
    pipeline_id: str

    def build_pipeline_definition(
        self,
        *,
        session_id: str,
        pipeline_service: PipelineAuthoringService,
    ) -> ExecutionPipelineDefinition:
        del session_id
        document = pipeline_service.to_pipeline_document(self.pipeline_id)
        spec = pipeline_service.get_pipeline(self.pipeline_id)
        return ExecutionPipelineDefinition(
            pipeline_id=self.pipeline_id,
            pipeline_document=document,
            pipeline_config_id=spec.pipeline_config_id,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PipelineSourceSessionRequest(ExecutionPipelineSessionRequest):
    pipeline_source: str

    def __post_init__(self) -> None:
        if self.identity.execution_plate_id is not None:
            raise ValueError(
                "Pipeline source sessions execute plate_id directly; "
                "execution_plate_id must be None."
            )
        if self.identity.selected_pipeline_path is not None:
            raise ValueError(
                "Pipeline source sessions use pipeline_source as the selected "
                "pipeline authority; selected_pipeline_path must be None."
            )

    def build_pipeline_definition(
        self,
        *,
        session_id: str,
        pipeline_service: PipelineAuthoringService,
    ) -> ExecutionPipelineDefinition:
        del pipeline_service
        logger.info("Parsing pipeline source for execution session %s", session_id)
        document = PipelineDocumentAuthority.from_source(self.pipeline_source)
        logger.info("Parsed pipeline source for execution session %s", session_id)
        return ExecutionPipelineDefinition(
            pipeline_id=f"pipeline-source:{session_id}",
            pipeline_document=document,
            pipeline_config_id=None,
        )


@dataclass(frozen=True, slots=True)
class ExecutionSessionRecord:
    session: OrchestratorSession
    pipeline_document: PipelineDocument
    global_pipeline_config: GlobalPipelineConfig

    def submission(
        self, compile_artifact_id: str | None = None
    ) -> OpenHCSExecutionSubmission:
        return OpenHCSExecutionSubmission(
            plate_id=self.session.plate_path,
            execution_plate_id=self.session.execution_plate_path,
            selected_pipeline_path=self.session.selected_pipeline_path,
            pipeline_document=self.pipeline_document,
            global_config=self.global_pipeline_config,
            compile_artifact_id=compile_artifact_id,
        )


@dataclass(frozen=True, slots=True)
class ExecutionJobRecord:
    ref: ExecutionJobRef
    response: JsonObject
    client: ExecutionClientABC | None

    def status(self, response: JsonObject | None = None) -> ExecutionJobStatus:
        payload = self.response if response is None else response
        bounded_payload = bounded_execution_status_response(payload)
        status = execution_status_from_response(
            bounded_payload,
            fallback=self.ref.status,
        )
        return ExecutionJobStatus(
            schema_version=SCHEMA_VERSION,
            job_id=self.ref.job_id,
            session_id=self.ref.session_id,
            kind=self.ref.kind,
            status=status,
            uri=self.ref.uri,
            server_execution_id=self.ref.server_execution_id,
            response=bounded_payload,
            progress=self.progress_observation(),
            errors=execution_status_errors(bounded_payload, status),
            warnings=execution_status_warnings(bounded_payload),
        )

    @property
    def is_terminal(self) -> bool:
        """Delegate terminality through the public job-status declaration."""

        return self.status().is_terminal

    def require_client(self) -> ExecutionClientABC:
        """Return the client that submitted this server-backed job."""

        if self.client is None:
            raise RuntimeError(
                f"Execution job {self.ref.job_id} has no submitting client."
            )
        return self.client

    def progress_observation(self) -> ExecutionProgressObservation | None:
        if self.client is None or self.ref.server_execution_id is None:
            return None
        return self.client.progress_observation(self.ref.server_execution_id)

    def release_client(self) -> None:
        """Release transport resources without replacing a terminal job result."""

        try:
            self.require_client().disconnect()
        except Exception:
            logger.exception(
                "Failed to release execution client for terminal job %s",
                self.ref.job_id,
            )


class ExecutionSessionStore:
    def __init__(self) -> None:
        self._counter = count(1)
        self._records: dict[str, ExecutionSessionRecord] = {}

    def next_id(self) -> str:
        return f"session-{next(self._counter)}"

    def store(self, record: ExecutionSessionRecord) -> OrchestratorSessionRef:
        self._records[record.session.session_id] = record
        return OrchestratorSessionRef(
            schema_version=SCHEMA_VERSION,
            session_id=record.session.session_id,
            uri=record.session.uri,
        )

    def session_record(self, session_id: str) -> ExecutionSessionRecord:
        try:
            return self._records[session_id]
        except KeyError as exc:
            raise UnknownExecutionSessionIdError(session_id) from exc

    @staticmethod
    def session_uri(session_id: str) -> str:
        return f"openhcs://execution/sessions/{session_id}"


class ExecutionJobStore:
    def __init__(self) -> None:
        self._counter = count(1)
        self._records: dict[str, ExecutionJobRecord] = {}

    def register(
        self,
        session_id: str,
        kind: ExecutionJobKind,
        response: JsonObject,
        client: ExecutionClientABC | None,
    ) -> ExecutionJobRef:
        job_id = f"job-{next(self._counter)}"
        ref = ExecutionJobRef(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            session_id=session_id,
            kind=kind.value,
            status=execution_status_from_response(response, fallback="submitted"),
            uri=self.job_uri(job_id),
            server_execution_id=_server_execution_id(response),
        )
        self._records[job_id] = ExecutionJobRecord(
            ref=ref,
            response=response,
            client=client,
        )
        return ref

    def job_record(self, job_id: str) -> ExecutionJobRecord:
        try:
            return self._records[job_id]
        except KeyError as exc:
            raise UnknownExecutionJobIdError(job_id) from exc

    def update_response(
        self,
        job_id: str,
        response: JsonObject,
    ) -> ExecutionJobRecord:
        record = replace(self.job_record(job_id), response=response)
        self._records[job_id] = record
        return record

    @staticmethod
    def job_uri(job_id: str) -> str:
        return f"openhcs://execution/jobs/{job_id}"


@dataclass(frozen=True, slots=True)
class ExecutionJobSubmission:
    """Accepted submission and the exact client that owns its progress stream."""

    client: ExecutionClientABC
    response: JsonObject


@dataclass(frozen=True, slots=True)
class ExecutionClientGateway:
    factory: ExecutionClientFactoryABC

    def submit(
        self,
        record: ExecutionSessionRecord,
        kind: ExecutionJobKind,
        compile_artifact_id: str | None = None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
    ) -> ExecutionJobSubmission:
        client = self.factory.create_client(record.session.connection)
        execution_request = record.submission(compile_artifact_id)
        try:
            response = kind.submit(
                client,
                execution_request,
                timeout_ms=timeout_ms,
            )
        except Exception:
            try:
                client.disconnect()
            except Exception:
                logger.exception("Failed to close rejected execution client")
            raise
        return ExecutionJobSubmission(client=client, response=dict(response))

    def status(
        self,
        client: ExecutionClientABC,
        server_execution_id: str,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> JsonObject:
        return dict(client.get_status(server_execution_id, timeout_ms=timeout_ms))

    def wait(
        self,
        client: ExecutionClientABC,
        server_execution_id: str,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> JsonObject:
        deadline = time.monotonic() + (max(0, timeout_ms) / 1000)
        last_response: JsonObject | None = None
        last_error: Exception | None = None
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                response = self.status(
                    client,
                    server_execution_id,
                    timeout_ms=min(
                        OPENHCS_ZMQ_CONFIG.control_timeout_ms,
                        remaining_ms,
                    ),
                )
            except Exception as exc:
                last_error = exc
                if time.monotonic() >= deadline:
                    return _execution_wait_timeout_response(
                        server_execution_id,
                        timeout_ms=timeout_ms,
                        last_response=last_response,
                        last_error=last_error,
                    )
                time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))
                continue
            last_response = response
            status = execution_status_from_response(response, fallback="unknown")
            lifecycle_status = ExecutionStatus.from_wire(status)
            if lifecycle_status is not None and lifecycle_status.is_terminal:
                return response
            if time.monotonic() >= deadline:
                return _execution_wait_timeout_response(
                    server_execution_id,
                    timeout_ms=timeout_ms,
                    last_response=last_response,
                    last_error=last_error,
                )
            time.sleep(min(0.5, max(0.0, deadline - time.monotonic())))


class ExecutionSessionService:
    """Create opaque ZMQ-backed execution sessions without exposing orchestrators."""

    def __init__(
        self,
        *,
        path_policy: AgentPathPolicy,
        pipeline_service: PipelineAuthoringService,
        config_service: ConfigService,
        client_factory: ExecutionClientFactoryABC | None = None,
        compile_inspection_gateway: CompileInspectionGatewayABC | None = None,
    ) -> None:
        self._path_policy = path_policy
        self._pipeline_service = pipeline_service
        self._config_service = config_service
        factory = client_factory or ZMQExecutionClientFactory()
        self._client_gateway = ExecutionClientGateway(factory)
        self._compile_inspection_gateway = (
            compile_inspection_gateway or InProcessCompileInspectionGateway()
        )
        self._session_store = ExecutionSessionStore()
        self._job_store = ExecutionJobStore()

    _DEFAULT_CONNECTION = ExecutionConnectionSpec()

    def create_session(
        self,
        *,
        plate_path: str,
        pipeline_id: str,
        execution_plate_path: str | None = None,
        selected_pipeline_path: str | None = None,
        global_config_id: str | None = None,
        connection: ExecutionConnectionSpec = _DEFAULT_CONNECTION,
    ) -> OrchestratorSessionRef:
        return self._create_session(
            DraftPipelineSessionRequest(
                identity=ZMQExecutionIdentity(
                    plate_id=plate_path,
                    execution_plate_id=execution_plate_path,
                    selected_pipeline_path=selected_pipeline_path,
                ),
                pipeline_id=pipeline_id,
                global_config_id=global_config_id,
                connection=connection,
            )
        )

    def create_session_from_request(
        self,
        request: OrchestratorSessionCreationRequest,
    ) -> OrchestratorSessionRef:
        return self._create_session(
            DraftPipelineSessionRequest(
                identity=ZMQExecutionIdentity(
                    plate_id=request.plate_path,
                    execution_plate_id=request.execution_plate_path,
                    selected_pipeline_path=request.selected_pipeline_path,
                ),
                pipeline_id=request.pipeline_id,
                global_config_id=request.global_config_id,
                connection=request.connection,
            )
        )

    def create_session_from_pipeline_source(
        self,
        request: PipelineSourceSessionRequest,
    ) -> OrchestratorSessionRef:
        return self._create_session(request)

    def create_session_from_pipeline_source_request(
        self,
        request: PipelineSourceOrchestratorSessionRequest,
    ) -> OrchestratorSessionRef:
        return self.create_session_from_pipeline_source(
            PipelineSourceSessionRequest(
                identity=ZMQExecutionIdentity(plate_id=request.plate_path),
                pipeline_source=request.pipeline_source,
                global_config_id=request.global_config_id,
                connection=request.connection,
            )
        )

    def inspect_pipeline_source_artifact_plan(
        self,
        request: PipelineSourceSessionRequest,
        *,
        axis_filter: tuple[str, ...] = (),
    ) -> ArtifactPlanInspection:
        progress_queue = AgentProgressQueue()
        plate = self._path_policy.assert_readable(request.identity.plate_id)
        metadata_path = _openhcs_metadata_path(plate)
        metadata_existed_before = metadata_path.exists()
        try:
            document = PipelineDocumentAuthority.from_source(request.pipeline_source)
        except Exception as exc:
            return ArtifactPlanInspection(
                schema_version=SCHEMA_VERSION,
                plate_path=str(plate),
                axis_filter=axis_filter,
                progress_event_count=0,
                warnings=_compile_inspection_workspace_warnings(
                    metadata_path,
                    metadata_existed_before,
                ),
                errors=(_pipeline_document_error(exc),),
            )

        try:
            compilation = self._compile_inspection_gateway.compile(
                CompileInspectionInput(
                    plate=plate,
                    pipeline_document=document,
                    axis_filter=axis_filter,
                    global_pipeline_config=GlobalConfigSelection(
                        request.global_config_id
                    ).resolve(self._config_service),
                    progress_queue=progress_queue,
                )
            )
        except Exception as exc:
            return ArtifactPlanInspection(
                schema_version=SCHEMA_VERSION,
                plate_path=str(plate),
                axis_filter=axis_filter,
                progress_event_count=len(progress_queue.events),
                warnings=_compile_inspection_workspace_warnings(
                    metadata_path,
                    metadata_existed_before,
                ),
                errors=(_compile_inspection_error(exc),),
            )

        return artifact_plan_inspection_from_compilation(
            plate_path=str(plate),
            axis_filter=axis_filter,
            compilation=compilation,
            progress_event_count=len(progress_queue.events),
            warnings=_compile_inspection_workspace_warnings(
                metadata_path,
                metadata_existed_before,
            ),
        )

    def inspect_pipeline_source_artifact_plan_request(
        self,
        request: PipelineSourceArtifactPlanInspectionRequest,
    ) -> ArtifactPlanInspection:
        return self.inspect_pipeline_source_artifact_plan(
            PipelineSourceSessionRequest(
                identity=ZMQExecutionIdentity(plate_id=request.plate_path),
                pipeline_source=request.pipeline_source,
                global_config_id=request.global_config_id,
                connection=ExecutionConnectionSpec(),
            ),
            axis_filter=request.axis_filter,
        )

    def _create_session(
        self,
        request: ExecutionPipelineSessionRequest,
    ) -> OrchestratorSessionRef:
        session_id = self._session_store.next_id()
        resolved = ResolvedExecutionSessionInputs.from_request(
            request=request,
            path_policy=self._path_policy,
            config_service=self._config_service,
        )
        pipeline_definition = request.build_pipeline_definition(
            session_id=session_id,
            pipeline_service=self._pipeline_service,
        )
        session = OrchestratorSession(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            uri=ExecutionSessionStore.session_uri(session_id),
            plate_path=str(resolved.plate),
            execution_plate_path=str(resolved.execution_plate),
            selected_pipeline_path=_optional_path_text(resolved.selected_pipeline),
            pipeline_id=pipeline_definition.pipeline_id,
            global_config_id=request.global_config_id,
            pipeline_config_id=pipeline_definition.pipeline_config_id,
            connection=resolved.connection,
        )
        return self._session_store.store(
            ExecutionSessionRecord(
                session=session,
                pipeline_document=pipeline_definition.pipeline_document,
                global_pipeline_config=resolved.global_pipeline_config,
            )
        )

    def get_session(self, session_id: str) -> OrchestratorSession:
        return self._session_store.session_record(session_id).session

    def get_session_from_request(
        self,
        request: OrchestratorSessionRequest,
    ) -> OrchestratorSession:
        return self.get_session(request.session_id)

    def submit_compile(
        self,
        session_id: str,
        *,
        wait: bool = False,
        submit_timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
        wait_timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> ExecutionJobRef | ExecutionJobStatus:
        return self._submit_job(
            session_id,
            ExecutionJobKind.COMPILE,
            wait=wait,
            submit_timeout_ms=submit_timeout_ms,
            wait_timeout_ms=wait_timeout_ms,
        )

    def submit_execution(
        self,
        session_id: str,
        *,
        compile_artifact_id: str | None = None,
        wait: bool = False,
        submit_timeout_ms: int = OPENHCS_ZMQ_CONFIG.execution_submission_timeout_ms,
        wait_timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> ExecutionJobRef | ExecutionJobStatus:
        return self._submit_job(
            session_id,
            ExecutionJobKind.EXECUTE,
            compile_artifact_id=compile_artifact_id,
            wait=wait,
            submit_timeout_ms=submit_timeout_ms,
            wait_timeout_ms=wait_timeout_ms,
        )

    def get_job_status(
        self,
        job_id: str,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ) -> ExecutionJobStatus:
        job = self._job_store.job_record(job_id)
        if job.ref.server_execution_id is None:
            return job.status()
        if job.is_terminal:
            return job.status()
        try:
            response = self._client_gateway.status(
                job.require_client(),
                job.ref.server_execution_id,
                timeout_ms=timeout_ms,
            )
        except Exception as exc:
            return replace(
                job.status(),
                status="status_error",
                errors=(AgentError.from_exception("execution_status_error", exc),),
            )
        updated = self._job_store.update_response(job_id, dict(response))
        if updated.is_terminal:
            updated.release_client()
        return updated.status()

    def _submit_job(
        self,
        session_id: str,
        kind: ExecutionJobKind,
        *,
        compile_artifact_id: str | None = None,
        wait: bool,
        submit_timeout_ms: int,
        wait_timeout_ms: int,
    ) -> ExecutionJobRef | ExecutionJobStatus:
        record = self._session_store.session_record(session_id)
        try:
            submission = self._client_gateway.submit(
                record,
                kind,
                compile_artifact_id,
                timeout_ms=submit_timeout_ms,
            )
        except Exception as exc:
            response = _execution_submit_error_response(
                kind,
                exc,
                timeout_ms=submit_timeout_ms,
            )
            ref = self._job_store.register(
                record.session.session_id,
                kind,
                response,
                client=None,
            )
            return replace(
                self._job_store.job_record(ref.job_id).status(),
                status="submit_error",
                errors=(_execution_submit_error(exc),),
            )
        ref = self._job_store.register(
            record.session.session_id,
            kind,
            submission.response,
            client=submission.client,
        )
        if wait and ref.server_execution_id is not None:
            wait_response = self._client_gateway.wait(
                submission.client,
                ref.server_execution_id,
                timeout_ms=wait_timeout_ms,
            )
            updated = self._job_store.update_response(ref.job_id, dict(wait_response))
            if updated.is_terminal:
                updated.release_client()
            return updated.status()
        return ref


def _server_execution_id(response: JsonObject) -> str | None:
    if "execution_id" not in response:
        return None
    execution_id = response["execution_id"]
    if execution_id is None:
        return None
    return str(execution_id)


def _execution_wait_timeout_response(
    server_execution_id: str,
    *,
    timeout_ms: int,
    last_response: JsonObject | None,
    last_error: Exception | None,
) -> JsonObject:
    status = "running"
    if last_response is not None:
        status = execution_status_from_response(last_response, fallback=status)
    response: dict[str, object] = {
        "status": status,
        "execution_id": server_execution_id,
        "wait_timed_out": True,
        "wait_timeout_ms": timeout_ms,
        "last_response": dict(last_response) if last_response is not None else {},
    }
    if last_error is not None:
        response["last_status_error"] = {
            "exception_type": type(last_error).__name__,
            "message": str(last_error),
        }
    return response


def _execution_submit_error_response(
    kind: ExecutionJobKind,
    exception: Exception,
    *,
    timeout_ms: int,
) -> JsonObject:
    return {
        "status": "submit_error",
        "kind": kind.value,
        "error": str(exception),
        "exception_type": type(exception).__name__,
        "submit_timeout_ms": timeout_ms,
    }


def _execution_submit_error(exception: Exception) -> AgentError:
    if isinstance(exception, ExecutionSubmissionPreparationTimeoutError):
        return AgentError.from_exception(
            "execution_submit_timeout",
            exception,
            hint=(
                "The submission budget expired before OpenHCS sent the execution "
                "request, so no execution request was sent. Retry with a larger "
                "submit_timeout_ms; slow first starts "
                "can include function-registry preparation."
            ),
        )
    if isinstance(exception, TimeoutError):
        return AgentError.from_exception(
            "execution_submit_timeout",
            exception,
            hint=(
                "The submit request timed out after it may have reached the "
                "runtime server, so the outcome is unknown and no execution_id "
                "is available. Check openhcs_get_runtime_server_execution_status "
                "or retry with a larger submit_timeout_ms."
            ),
        )
    return AgentError.from_exception(
        "execution_submit_error",
        exception,
        hint=(
            "The runtime server did not accept the job. Check runtime server info, "
            "the session connection, and plate/pipeline validity before retrying."
        ),
    )


def artifact_plan_inspection_from_compilation(
    *,
    plate_path: str,
    axis_filter: tuple[str, ...],
    compilation: CompileInspectionResult,
    progress_event_count: int,
    warnings: tuple[AgentWarning, ...] = (),
) -> ArtifactPlanInspection:
    execution_bundle = compilation.execution_bundle
    compiled_contexts = dict(execution_bundle.runtime_contexts)
    axes = tuple(sorted(str(axis_id) for axis_id in compiled_contexts))
    source_workspace_axes = axes or axis_filter
    step_summaries = tuple(
        _bounded_step_summaries(compiled_contexts, axes[:MAX_INSPECTION_AXES])
    )
    return ArtifactPlanInspection(
        schema_version=SCHEMA_VERSION,
        plate_path=plate_path,
        axis_filter=axis_filter,
        axis_count=len(axes),
        axes=axes[:MAX_INSPECTION_AXES],
        truncated_axis_count=max(0, len(axes) - MAX_INSPECTION_AXES),
        step_count=sum(
            len(context.step_plans) for context in compiled_contexts.values()
        ),
        steps=step_summaries,
        truncated_step_count=max(
            0,
            sum(len(context.step_plans) for context in compiled_contexts.values())
            - len(step_summaries),
        ),
        worker_assignments={
            str(worker): [str(axis_id) for axis_id in axis_ids]
            for worker, axis_ids in execution_bundle.worker_assignments.items()
        },
        source_workspace=_source_workspace_summary(
            compilation.source_workspace_projection,
            axes=source_workspace_axes,
        ),
        progress_event_count=progress_event_count,
        warnings=warnings,
    )


def _openhcs_metadata_path(plate: Path) -> Path:
    return plate / os.getenv(
        "OPENHCS_METADATA_FILENAME",
        OpenHCSMetadataHandler.METADATA_FILENAME,
    )


def _compile_inspection_workspace_warnings(
    metadata_path: Path,
    existed_before: bool,
) -> tuple[AgentWarning, ...]:
    if existed_before or not metadata_path.exists():
        return ()
    return (
        AgentWarning(
            code="compile_inspection_initialized_workspace",
            message=(
                "Compile inspection initialized OpenHCS workspace metadata at "
                f"{metadata_path}."
            ),
            hint=(
                "Raw microscope layouts can require virtual workspace metadata "
                "before compilation or execution. Run openhcs_inspect_plate_path "
                "first to preview whether workspace preparation is required."
            ),
        ),
    )


def _compile_inspection_error(exception: Exception) -> AgentError:
    if isinstance(exception, AgentFacingErrorMixin):
        return exception.to_agent_error()
    if isinstance(exception, MissingArtifactInputError):
        return AgentError.from_exception(
            "compile_inspection_missing_artifact_input",
            exception,
            hint=(
                f"Step {exception.step_id} requires artifact input "
                f"{exception.artifact_key!r}. Add an earlier FunctionStep that "
                "declares that artifact output, configure source bindings that "
                "provide it from the plate workspace, or inspect "
                "openhcs_function_patterns and "
                "openhcs_architecture_quick_start#compile-before-execution before "
                "retrying artifact-plan."
            ),
        )
    if isinstance(exception, MicroscopePixelSizeUnavailableError):
        return AgentError.from_exception(
            "compile_inspection_pixel_size_unavailable",
            exception,
            hint=(
                "Run openhcs_inspect_plate_path first and check its warnings. "
                "Use the true microscope plate root, initialize the workspace "
                "when required, or provide microscope metadata that includes "
                "physical pixel size before retrying compile inspection."
            ),
            path=str(exception.image_path),
        )
    return AgentError.from_exception(
        "compile_inspection_failed",
        exception,
        hint=(
            "Check plate inspection, the embedded pipeline config, and the external "
            "global config before retrying compile inspection."
        ),
    )


def _pipeline_document_error(exception: Exception) -> AgentError:
    code = (
        "pipeline_source_syntax_error"
        if isinstance(exception, SyntaxError)
        else "pipeline_source_invalid_document"
    )
    return AgentError.from_exception(
        code,
        exception,
        hint=(
            "Provide a pipeline document defining a typed pipeline_steps "
            "assignment. Omit pipeline_config only to use PipelineConfig(), or "
            "render an MCP draft with openhcs_render_pipeline_source."
        ),
    )


def _bounded_step_summaries(compiled_contexts, axes: tuple[str, ...]):
    emitted = 0
    for axis_id in axes:
        context = compiled_contexts[axis_id]
        for step_plan in context.step_plans.values():
            if emitted >= MAX_INSPECTION_STEPS:
                return
            emitted += 1
            artifact_inputs = tuple(
                _artifact_input_summary(plan)
                for _input_key, plan in tuple(step_plan.artifact_inputs.items())[
                    :MAX_INSPECTION_ARTIFACT_INPUTS_PER_STEP
                ]
            )
            artifact_outputs = tuple(
                _artifact_summary(context, step_plan, plan.name, plan)
                for plan in tuple(step_plan.artifact_outputs.values())[
                    :MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP
                ]
            )
            yield CompiledStepPlanSummary(
                step_index=int(step_plan.step_index),
                step_name=str(step_plan.step_name),
                axis_id=str(step_plan.axis_id),
                output_dir=_optional_path_text(step_plan.output_dir),
                main_flow_axis_persistence_enabled=(
                    step_plan.main_flow_axis_persistence_enabled
                ),
                execution_groups=step_plan.execution_group_scope.keys,
                main_flow_materialization=(
                    _main_flow_materialization_summary(step_plan)
                ),
                viewer_streaming=_viewer_streaming_summaries(step_plan),
                artifact_inputs=artifact_inputs,
                artifact_outputs=artifact_outputs,
                truncated_artifact_input_count=max(
                    0,
                    len(step_plan.artifact_inputs)
                    - MAX_INSPECTION_ARTIFACT_INPUTS_PER_STEP,
                ),
                truncated_artifact_output_count=max(
                    0,
                    len(step_plan.artifact_outputs)
                    - MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP,
                ),
            )


def _main_flow_materialization_summary(
    step_plan: CompiledStepPlan,
) -> MainFlowMaterializationPlanSummary | None:
    plan = step_plan.materialized_output
    if plan is None:
        return None
    return MainFlowMaterializationPlanSummary(
        output_dir=str(plan.output_dir),
        backend=str(plan.backend),
        plate_root=str(plan.plate_root),
        sub_dir=str(plan.sub_dir),
        analysis_results_dir=(
            None
            if plan.analysis_results_dir is None
            else str(plan.analysis_results_dir)
        ),
    )


def _viewer_streaming_summaries(
    step_plan: CompiledStepPlan,
) -> tuple[ViewerStreamingPlanSummary, ...]:
    summaries = []
    for config_key, config in step_plan.streaming_configs.items():
        effective_config = to_jsonable(config)
        if not isinstance(effective_config, dict):
            raise TypeError(
                "Compiled streaming config projection must be a JSON object; "
                f"got {type(effective_config).__name__}."
            )
        summaries.append(
            ViewerStreamingPlanSummary(
                config_key=str(config_key),
                viewer_type=config.viewer_type,
                backend=str(config.backend.value),
                effective_config=effective_config,
            )
        )
    return tuple(summaries)


def _artifact_input_summary(plan) -> ArtifactInputPlanSummary:
    return ArtifactInputPlanSummary(
        name=str(plan.name),
        kind=str(plan.artifact_type.value),
        path=str(plan.path),
        group_keys=tuple(plan.group_keys),
        paths_by_group=_artifact_paths_by_group(plan),
        source_step_id=plan.source_step_id,
        source_step_scope_id=plan.source_step_scope_id,
    )


def _artifact_summary(
    context,
    step_plan: CompiledStepPlan,
    output_key: str,
    plan,
) -> ArtifactPlanSummary:
    return ArtifactPlanSummary(
        name=str(plan.name),
        kind=str(plan.artifact_type.value),
        path=str(plan.path),
        group_keys=tuple(plan.group_keys),
        paths_by_group=_artifact_paths_by_group(plan),
        materialization=_artifact_materialization_summary(
            context,
            step_plan,
            output_key,
            plan,
        ),
    )


def _artifact_paths_by_group(plan) -> tuple[JsonObject, ...]:
    if plan.paths_by_group is None:
        return ()
    return tuple(
        {"group_key": group_key, "path": path}
        for group_key, path in plan.paths_by_group.items()
    )


def _artifact_materialization_summary(
    context,
    step_plan: CompiledStepPlan,
    output_key: str,
    output_plan,
) -> ArtifactMaterializationPlanSummary | None:
    persistent_plan = step_plan.runtime_artifact_materialization
    if output_plan.materialization is None:
        return None

    step_plan.require_function_execution_ready()

    preview = planned_materialization_preview(
        context=context,
        plan=step_plan,
        output_key=output_key,
        output_plan=output_plan,
    )
    paths = ()
    filename_uses_source_identity = (
        output_plan.materialization_uses_source_identity_filename()
    )
    runtime_metadata_can_refine_paths = False
    if preview is not None:
        paths = tuple(
            ArtifactMaterializationPathSummary(
                group_key=path.group_key,
                shared_output_stem=path.shared_output_stem,
                candidate_paths=path.candidate_paths,
            )
            for path in preview.paths
        )
        filename_uses_source_identity = preview.filename_uses_source_identity
        runtime_metadata_can_refine_paths = preview.runtime_metadata_can_refine_paths

    note = None
    if runtime_metadata_can_refine_paths:
        note = "Runtime payload metadata can split or refine candidate filenames."

    return ArtifactMaterializationPlanSummary(
        persistent_enabled=persistent_plan.persistent_enabled,
        persistent_backend=persistent_plan.persistent_backend,
        analysis_output_dir=str(step_plan.artifact_analysis_output_dir),
        paths=paths,
        runtime_resolved=False,
        filename_uses_source_identity=filename_uses_source_identity,
        runtime_metadata_can_refine_paths=runtime_metadata_can_refine_paths,
        note=note,
    )


def _source_workspace_summary(
    projection,
    *,
    axes: tuple[str, ...],
) -> SourceWorkspaceSummary:
    if not isinstance(projection, VirtualWorkspaceSourceProjection):
        return SourceWorkspaceSummary()

    full_virtual_paths = projection.pipeline_start_files()
    records = tuple(
        _source_workspace_record(projection, full_virtual_path)
        for full_virtual_path in full_virtual_paths[
            :MAX_INSPECTION_SOURCE_WORKSPACE_FILES
        ]
    )
    return SourceWorkspaceSummary(
        file_count=len(full_virtual_paths),
        files=records,
        truncated_file_count=max(0, len(full_virtual_paths) - len(records)),
        axis_file_counts={
            axis: len(projection.pipeline_start_files(axis_id=axis))
            for axis in axes[:MAX_INSPECTION_AXES]
        },
    )


def _source_workspace_record(
    projection: VirtualWorkspaceSourceProjection,
    full_virtual_path: str,
) -> SourceWorkspaceFileRecord:
    virtual_path = _relative_virtual_path(projection, full_virtual_path)
    lookup = VirtualWorkspacePathLookup.from_paths(virtual_path, full_virtual_path)
    metadata = projection.source_metadata_for(lookup)
    source_metadata = {} if metadata is None else to_jsonable(metadata)
    if not isinstance(source_metadata, dict):
        source_metadata = {}
    return SourceWorkspaceFileRecord(
        virtual_path=virtual_path,
        full_virtual_path=full_virtual_path,
        source_path=projection.source_path_for(lookup),
        source_metadata=source_metadata,
    )


def _relative_virtual_path(
    projection: VirtualWorkspaceSourceProjection,
    full_virtual_path: str,
) -> str:
    if projection.workspace_root is None:
        return full_virtual_path
    try:
        return str(Path(full_virtual_path).relative_to(projection.workspace_root))
    except ValueError:
        return full_virtual_path


def _optional_path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
