"""Opaque execution sessions for OpenHCS agent integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from itertools import count
import os
from pathlib import Path
import queue
import threading
from types import ModuleType
import sys
import time

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.common import (
    AgentError,
    AgentWarning,
    JsonObject,
    SCHEMA_VERSION,
)
from openhcs.agent.dto.execution import (
    ArtifactInputPlanSummary,
    ArtifactMaterializationPathSummary,
    ArtifactMaterializationPlanSummary,
    ArtifactPlanInspection,
    ArtifactPlanSummary,
    CompiledStepPlanSummary,
    DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
    DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    DEFAULT_EXECUTION_WAIT_TIMEOUT_MS,
    ExecutionConnectionSpec,
    ExecutionJobRef,
    ExecutionJobStatus,
    OrchestratorSession,
    OrchestratorSessionCreationRequest,
    OrchestratorSessionRequest,
    OrchestratorSessionRef,
    PipelineSourceArtifactPlanInspectionRequest,
    PipelineSourceOrchestratorSessionRequest,
    SourceWorkspaceFileRecord,
    SourceWorkspaceSummary,
    bounded_execution_status_response,
    execution_status_errors,
    execution_status_from_response,
    execution_status_warnings,
)
from openhcs.agent.exceptions import AgentFacingErrorMixin
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.serialization import to_jsonable
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.pipeline_authoring_service import PipelineAuthoringService
from openhcs.core.artifact_materialization_policy import NO_ARTIFACT_MATERIALIZATION
from openhcs.core.artifacts import ArtifactKind
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.pipeline.path_planner import MissingArtifactInputError
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjection,
)
from openhcs.core.steps.function_artifact_materialization import (
    planned_materialization_preview,
)
from openhcs.core.steps.function_plan import FunctionStepExecutionPlan
from openhcs.microscopes.exceptions import MicroscopePixelSizeUnavailableError
from openhcs.microscopes.openhcs import OpenHCSMetadataHandler
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity
from openhcs.runtime.zmq_pipeline_transport import (
    PipelineSourceExport,
    PipelineStepsNamespaceProjection,
    PipelineStepsBoundary,
    PipelineStepsCarrier,
)


MAX_INSPECTION_AXES = 8
MAX_INSPECTION_STEPS = 24
MAX_INSPECTION_ARTIFACT_INPUTS_PER_STEP = 16
MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP = 16
MAX_INSPECTION_SOURCE_WORKSPACE_FILES = 64
PENDING_EXECUTION_STATUSES = frozenset(
    ("accepted", "ok", "queued", "running", "submitted", "unknown")
)


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


class PipelineSourceSyntaxError(ExecutionSessionError):
    """Raised when pycodified pipeline source is not valid Python."""

    agent_error_code = "pipeline_source_syntax_error"
    agent_error_hint = (
        "Fix pipeline_source Python syntax. A valid source document should define "
        "pipeline_steps, usually by rendering an MCP pipeline draft with "
        "openhcs_render_pipeline_source."
    )

    def __init__(self, syntax_error: SyntaxError) -> None:
        self.syntax_error = syntax_error
        super().__init__(str(syntax_error))


class PipelineSourceMissingStepsError(ExecutionSessionError):
    """Raised when pycodified source does not define pipeline_steps."""

    agent_error_code = "pipeline_source_missing_steps"
    agent_error_hint = (
        "Define pipeline_steps in pipeline_source, or use "
        "openhcs_render_pipeline_source to generate a valid source document. "
        "This artifact-plan tool expects pipeline-only source, not a Plate "
        "Manager orchestrator config document that defines pipeline_data."
    )

    def __init__(self) -> None:
        super().__init__(
            f"Code must define {PipelineSourceExport.PIPELINE_STEPS.value!r}"
        )


class ExecutionJobKind(Enum):
    COMPILE = "compile"
    EXECUTE = "execute"


class AgentProgressQueue:
    def __init__(self) -> None:
        self.events: list[JsonObject] = []

    def put(self, event) -> None:
        if isinstance(event, dict):
            self.events.append(dict(event))


@dataclass(frozen=True, slots=True)
class CompileInspectionInput:
    plate: Path
    pipeline_source: str
    axis_filter: tuple[str, ...]
    configs: ExecutionConfigBundle
    progress_queue: AgentProgressQueue


class CompileInspectionGatewayABC(ABC):
    @abstractmethod
    def compile(self, request: CompileInspectionInput) -> JsonObject:
        raise NotImplementedError


class InProcessCompileInspectionGateway(CompileInspectionGatewayABC):
    def compile(self, request: CompileInspectionInput) -> JsonObject:
        module_name = "openhcs_agent_compile_inspection"
        module_file = f"<{module_name}>"
        try:
            code = compile(request.pipeline_source, module_file, "exec")
        except SyntaxError as exc:
            raise PipelineSourceSyntaxError(exc) from exc

        from openhcs.config_framework.lazy_factory import ensure_global_config_context
        from openhcs.core.config import GlobalPipelineConfig
        import openhcs.processing.func_registry as func_registry_module
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.core.progress import set_progress_queue

        ensure_global_config_context(
            GlobalPipelineConfig,
            request.configs.global_pipeline,
        )
        with func_registry_module._registry_lock:
            if not func_registry_module._registry_initialized:
                func_registry_module._auto_initialize_registry()

        module = ModuleType(module_name)
        module.__file__ = module_file
        sys.modules[module_name] = module
        try:
            exec(code, module.__dict__)
        finally:
            sys.modules.pop(module_name, None)

        pipeline_boundary = PipelineStepsNamespaceProjection(
            module.__dict__
        ).boundary_or_none()
        if pipeline_boundary is None:
            raise PipelineSourceMissingStepsError()

        orchestrator = PipelineOrchestrator(
            plate_path=request.plate,
            pipeline_config=request.configs.plate_pipeline,
            progress_callback=None,
        )
        orchestrator.initialize()
        set_progress_queue(request.progress_queue)
        try:
            compilation = dict(
                orchestrator.compile_pipelines(
                    pipeline_definition=pipeline_boundary.steps,
                    well_filter=list(request.axis_filter) or None,
                    is_zmq_execution=True,
                )
            )
            compilation["source_workspace_projection"] = (
                orchestrator.source_workspace_projection()
            )
            return compilation
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


@dataclass(frozen=True, slots=True)
class PipelineConfigSelection:
    config_id: str | None

    def resolve(self, config_service: ConfigService) -> PipelineConfig | None:
        if self.config_id is None:
            return None
        config = config_service.resolve_ref(self.config_id)
        if not isinstance(config, PipelineConfig):
            raise TypeError("pipeline_config_id must resolve to PipelineConfig")
        return config


class ExecutionClientABC(ABC):
    @abstractmethod
    def submit_compile(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def submit_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    ) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def wait_for_completion(self, execution_id: str) -> JsonObject:
        raise NotImplementedError


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
        timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
    ) -> JsonObject:
        return dict(self.client.submit_compile(submission, timeout_ms=timeout_ms))

    def submit_pipeline(
        self,
        submission: OpenHCSExecutionSubmission,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
    ) -> JsonObject:
        return dict(self.client.submit_pipeline(submission, timeout_ms=timeout_ms))

    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    ) -> JsonObject:
        return dict(self.client.get_status(execution_id, timeout_ms=timeout_ms))

    def wait_for_completion(self, execution_id: str) -> JsonObject:
        return dict(self.client.wait_for_completion(execution_id))


class ZMQExecutionClientFactory(ExecutionClientFactoryABC):
    def create_client(
        self,
        connection: ExecutionConnectionSpec,
    ) -> ZMQExecutionClientAdapter:
        return ZMQExecutionClientAdapter(
            ZMQExecutionClient(**connection.zmq_client_kwargs())
        )


@dataclass(frozen=True, slots=True)
class ExecutionConfigBundle:
    global_pipeline: GlobalPipelineConfig
    plate_pipeline: PipelineConfig | None


@dataclass(frozen=True, slots=True)
class PipelineIdBoundary:
    value: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionSessionCommonRequest:
    identity: ZMQExecutionIdentity
    global_config_id: str | None = None
    pipeline_config_id: str | None = None
    connection: ExecutionConnectionSpec = ExecutionConnectionSpec()


class ExecutionPipelineSessionRequest(
    ExecutionSessionCommonRequest,
    ABC,
    metaclass=AutoRegisterMeta,
):
    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @abstractmethod
    def pipeline_provider(self) -> ExecutionPipelineDefinitionProvider:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ResolvedExecutionSessionInputs:
    plate: Path
    execution_plate: Path
    selected_pipeline: Path | None
    configs: ExecutionConfigBundle
    connection: ExecutionConnectionSpec

    @classmethod
    def from_request(
        cls,
        *,
        request: ExecutionSessionCommonRequest,
        path_policy: AgentPathPolicy,
        config_service: ConfigService,
    ) -> "ResolvedExecutionSessionInputs":
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
            configs=ExecutionConfigBundle(
                global_pipeline=GlobalConfigSelection(
                    request.global_config_id
                ).resolve(config_service),
                plate_pipeline=PipelineConfigSelection(
                    request.pipeline_config_id
                ).resolve(config_service),
            ),
            connection=request.connection,
        )


@dataclass(frozen=True, slots=True)
class ExecutionPipelinePayload(PipelineStepsCarrier):
    registry_key = "execution_pipeline_payload"

    definition_pipeline: PipelineStepsBoundary
    pipeline_source: str | None

    @property
    def pipeline_steps_boundary(self) -> PipelineStepsBoundary:
        return self.definition_pipeline


@dataclass(frozen=True, slots=True)
class ExecutionPipelineDefinition(ExecutionPipelinePayload):
    registry_key = "execution_pipeline_definition"

    pipeline_identity: PipelineIdBoundary

    @property
    def pipeline_id(self) -> str:
        return self.pipeline_identity.value


class ExecutionPipelineDefinitionProvider(ABC, metaclass=AutoRegisterMeta):
    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key = None

    @abstractmethod
    def build(
        self,
        *,
        session_id: str,
        pipeline_service: PipelineAuthoringService,
    ) -> ExecutionPipelineDefinition:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class DraftPipelineDefinitionProvider(ExecutionPipelineDefinitionProvider):
    registry_key = "draft"
    request: DraftPipelineSessionRequest

    def build(
        self,
        *,
        session_id: str,
        pipeline_service: PipelineAuthoringService,
    ) -> ExecutionPipelineDefinition:
        return ExecutionPipelineDefinition(
            pipeline_identity=self.request.pipeline_identity,
            definition_pipeline=PipelineStepsBoundary(
                pipeline_service.to_function_steps(self.request.pipeline_id)
            ),
            pipeline_source=None,
        )


@dataclass(frozen=True, slots=True)
class PycodifiedSourcePipelineDefinitionProvider(ExecutionPipelineDefinitionProvider):
    registry_key = "pycodified_source"
    pipeline_source: str

    def build(
        self,
        *,
        session_id: str,
        pipeline_service: PipelineAuthoringService,
    ) -> ExecutionPipelineDefinition:
        return ExecutionPipelineDefinition(
            pipeline_identity=PipelineIdBoundary(f"pycodified-source:{session_id}"),
            definition_pipeline=PipelineStepsBoundary([]),
            pipeline_source=self.pipeline_source,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class DraftPipelineSessionRequest(ExecutionPipelineSessionRequest):
    registry_key = "draft"
    pipeline_identity: PipelineIdBoundary

    @property
    def pipeline_id(self) -> str:
        return self.pipeline_identity.value

    def pipeline_provider(self) -> DraftPipelineDefinitionProvider:
        return DraftPipelineDefinitionProvider(request=self)


@dataclass(frozen=True, slots=True, kw_only=True)
class PycodifiedPipelineSessionRequest(ExecutionPipelineSessionRequest):
    registry_key = "pycodified_source"
    pipeline_source: str

    def __post_init__(self) -> None:
        if self.identity.execution_plate_id is not None:
            raise ValueError(
                "Pycodified source sessions execute plate_id directly; "
                "execution_plate_id must be None."
            )
        if self.identity.selected_pipeline_path is not None:
            raise ValueError(
                "Pycodified source sessions use pipeline_source as the selected "
                "pipeline authority; selected_pipeline_path must be None."
            )

    def pipeline_provider(self) -> PycodifiedSourcePipelineDefinitionProvider:
        return PycodifiedSourcePipelineDefinitionProvider(self.pipeline_source)


@dataclass(frozen=True, slots=True)
class ExecutionSessionRecord(ExecutionPipelinePayload):
    registry_key = "execution_session_record"

    session: OrchestratorSession
    configs: ExecutionConfigBundle

    def submission(self, compile_artifact_id: str | None = None) -> OpenHCSExecutionSubmission:
        return OpenHCSExecutionSubmission(
            plate_id=self.session.plate_path,
            execution_plate_id=self.session.execution_plate_path,
            selected_pipeline_path=self.session.selected_pipeline_path,
            pipeline_steps=self.pipeline_steps,
            global_config=self.configs.global_pipeline,
            pipeline_config=self.configs.plate_pipeline,
            compile_artifact_id=compile_artifact_id,
            pipeline_source=self.pipeline_source,
        )


@dataclass(frozen=True, slots=True)
class ExecutionJobRecord:
    ref: ExecutionJobRef
    response: JsonObject

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
            errors=execution_status_errors(bounded_payload, status),
            warnings=execution_status_warnings(bounded_payload),
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
        self._records[job_id] = ExecutionJobRecord(ref=ref, response=response)
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
class ExecutionClientGateway:
    factory: ExecutionClientFactoryABC

    def submit(
        self,
        record: ExecutionSessionRecord,
        kind: ExecutionJobKind,
        compile_artifact_id: str | None = None,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
    ) -> JsonObject:
        return _call_zmq_control_with_timeout(
            lambda: self._submit_without_boundary_timeout(
                record,
                kind,
                compile_artifact_id,
                timeout_ms=timeout_ms,
            ),
            timeout_ms=timeout_ms,
            operation=f"{kind.value} submit",
        )

    def _submit_without_boundary_timeout(
        self,
        record: ExecutionSessionRecord,
        kind: ExecutionJobKind,
        compile_artifact_id: str | None = None,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        client = self.factory.create_client(record.session.connection)
        submission = record.submission(compile_artifact_id)
        if kind is ExecutionJobKind.COMPILE:
            return dict(client.submit_compile(submission, timeout_ms=timeout_ms))
        return dict(client.submit_pipeline(submission, timeout_ms=timeout_ms))

    def status(
        self,
        session: OrchestratorSession,
        server_execution_id: str,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    ) -> JsonObject:
        return _call_zmq_control_with_timeout(
            lambda: self._status_without_boundary_timeout(
                session,
                server_execution_id,
                timeout_ms=timeout_ms,
            ),
            timeout_ms=timeout_ms,
            operation="status poll",
        )

    def _status_without_boundary_timeout(
        self,
        session: OrchestratorSession,
        server_execution_id: str,
        *,
        timeout_ms: int,
    ) -> JsonObject:
        client = self.factory.create_client(session.connection)
        return dict(client.get_status(server_execution_id, timeout_ms=timeout_ms))

    def wait(
        self,
        session: OrchestratorSession,
        server_execution_id: str,
        *,
        timeout_ms: int = DEFAULT_EXECUTION_WAIT_TIMEOUT_MS,
    ) -> JsonObject:
        deadline = time.monotonic() + (max(0, timeout_ms) / 1000)
        last_response: JsonObject | None = None
        last_error: Exception | None = None
        while True:
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                response = self.status(
                    session,
                    server_execution_id,
                    timeout_ms=min(DEFAULT_EXECUTION_STATUS_TIMEOUT_MS, remaining_ms),
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
            if status not in PENDING_EXECUTION_STATUSES:
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
        pipeline_config_id: str | None = None,
        connection: ExecutionConnectionSpec = _DEFAULT_CONNECTION,
    ) -> OrchestratorSessionRef:
        return self._create_session(
            DraftPipelineSessionRequest(
                identity=ZMQExecutionIdentity(
                    plate_id=plate_path,
                    execution_plate_id=execution_plate_path,
                    selected_pipeline_path=selected_pipeline_path,
                ),
                pipeline_identity=PipelineIdBoundary(pipeline_id),
                global_config_id=global_config_id,
                pipeline_config_id=pipeline_config_id,
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
                pipeline_identity=PipelineIdBoundary(request.pipeline_id),
                global_config_id=request.global_config_id,
                pipeline_config_id=request.pipeline_config_id,
                connection=request.connection,
            )
        )

    def create_session_from_pipeline_source(
        self,
        request: PycodifiedPipelineSessionRequest,
    ) -> OrchestratorSessionRef:
        return self._create_session(request)

    def create_session_from_pipeline_source_request(
        self,
        request: PipelineSourceOrchestratorSessionRequest,
    ) -> OrchestratorSessionRef:
        return self.create_session_from_pipeline_source(
            PycodifiedPipelineSessionRequest(
                identity=ZMQExecutionIdentity(plate_id=request.plate_path),
                pipeline_source=request.pipeline_source,
                global_config_id=request.global_config_id,
                pipeline_config_id=request.pipeline_config_id,
                connection=request.connection,
            )
        )

    def inspect_pipeline_source_artifact_plan(
        self,
        request: PycodifiedPipelineSessionRequest,
        *,
        axis_filter: tuple[str, ...] = (),
    ) -> ArtifactPlanInspection:
        progress_queue = AgentProgressQueue()
        plate = self._path_policy.assert_readable(request.identity.plate_id)
        metadata_path = _openhcs_metadata_path(plate)
        metadata_existed_before = metadata_path.exists()
        try:
            compilation = self._compile_inspection_gateway.compile(
                CompileInspectionInput(
                    plate=plate,
                    pipeline_source=request.pipeline_source,
                    axis_filter=axis_filter,
                    configs=ExecutionConfigBundle(
                        global_pipeline=GlobalConfigSelection(
                            request.global_config_id
                        ).resolve(self._config_service),
                        plate_pipeline=PipelineConfigSelection(
                            request.pipeline_config_id
                        ).resolve(self._config_service),
                    ),
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
            PycodifiedPipelineSessionRequest(
                identity=ZMQExecutionIdentity(plate_id=request.plate_path),
                pipeline_source=request.pipeline_source,
                global_config_id=request.global_config_id,
                pipeline_config_id=request.pipeline_config_id,
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
        pipeline_definition = request.pipeline_provider().build(
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
            pipeline_config_id=request.pipeline_config_id,
            connection=resolved.connection,
        )
        return self._session_store.store(
            ExecutionSessionRecord(
                session=session,
                definition_pipeline=pipeline_definition.definition_pipeline,
                pipeline_source=pipeline_definition.pipeline_source,
                configs=resolved.configs,
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
        submit_timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
        wait_timeout_ms: int = DEFAULT_EXECUTION_WAIT_TIMEOUT_MS,
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
        submit_timeout_ms: int = DEFAULT_EXECUTION_SUBMIT_TIMEOUT_MS,
        wait_timeout_ms: int = DEFAULT_EXECUTION_WAIT_TIMEOUT_MS,
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
        timeout_ms: int = DEFAULT_EXECUTION_STATUS_TIMEOUT_MS,
    ) -> ExecutionJobStatus:
        job = self._job_store.job_record(job_id)
        if job.ref.server_execution_id is None:
            return job.status()
        try:
            session = self._session_store.session_record(job.ref.session_id).session
            response = self._client_gateway.status(
                session,
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
            response = self._client_gateway.submit(
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
            ref = self._job_store.register(record.session.session_id, kind, response)
            return replace(
                self._job_store.job_record(ref.job_id).status(),
                status="submit_error",
                errors=(_execution_submit_error(exc),),
            )
        ref = self._job_store.register(record.session.session_id, kind, response)
        if wait and ref.server_execution_id is not None:
            wait_response = self._client_gateway.wait(
                record.session,
                ref.server_execution_id,
                timeout_ms=wait_timeout_ms,
            )
            updated = self._job_store.update_response(ref.job_id, dict(wait_response))
            return updated.status()
        return ref

def _server_execution_id(response: JsonObject) -> str | None:
    if "execution_id" not in response:
        return None
    execution_id = response["execution_id"]
    if execution_id is None:
        return None
    return str(execution_id)


def _call_zmq_control_with_timeout(
    operation_fn,
    *,
    timeout_ms: int,
    operation: str,
) -> JsonObject:
    result_queue: queue.Queue[tuple[bool, JsonObject | Exception]] = queue.Queue(maxsize=1)

    def run_operation() -> None:
        try:
            result_queue.put((True, operation_fn()))
        except Exception as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=run_operation, daemon=True)
    thread.start()
    try:
        success, result = result_queue.get(timeout=max(timeout_ms / 1000, 0.001))
    except queue.Empty as exc:
        raise TimeoutError(
            f"Timed out waiting for ZMQ {operation} after {timeout_ms}ms."
        ) from exc
    if success:
        return result
    raise result


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
    compilation: JsonObject,
    progress_event_count: int,
    warnings: tuple[AgentWarning, ...] = (),
) -> ArtifactPlanInspection:
    execution_bundle = compilation["execution_bundle"]
    compiled_contexts = dict(execution_bundle.runtime_contexts)
    axes = tuple(sorted(str(axis_id) for axis_id in compiled_contexts.keys()))
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
        step_count=sum(len(context.step_plans) for context in compiled_contexts.values()),
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
            compilation.get("source_workspace_projection"),
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
                "openhcs_agent_mcp_overview#pipeline-input-routing before "
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
            "Check plate inspection, pipeline_source, and config IDs before retrying "
            "compile inspection."
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
                _artifact_summary(context, step_plan, output_key, plan)
                for output_key, plan in tuple(step_plan.artifact_outputs.items())[
                    :MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP
                ]
            )
            yield CompiledStepPlanSummary(
                step_index=int(step_plan.step_index),
                step_name=str(step_plan.step_name),
                axis_id=str(step_plan.axis_id),
                output_dir=_optional_path_text(step_plan.output_dir),
                execution_groups=tuple(step_plan.execution_groups),
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


def _artifact_input_summary(plan) -> ArtifactInputPlanSummary:
    return ArtifactInputPlanSummary(
        name=str(plan.name),
        kind=str(plan.kind.value),
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
        kind=str(plan.kind.value),
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
    if (
        output_plan.materialization is None
        and output_plan.kind is ArtifactKind.SPECIAL
    ):
        return None

    persistent_plan = step_plan.runtime_artifact_materialization
    analysis_output_dir = _analysis_output_dir(step_plan)
    if output_plan.materialization is NO_ARTIFACT_MATERIALIZATION:
        return ArtifactMaterializationPlanSummary(
            persistent_enabled=persistent_plan.persistent_enabled,
            persistent_backend=persistent_plan.persistent_backend,
            analysis_output_dir=analysis_output_dir,
            disabled=True,
            note="Artifact materialization is explicitly disabled.",
        )

    execution_plan = _function_step_execution_plan(context, step_plan)
    if execution_plan is None:
        runtime_resolved = output_plan.materialization is None
        return ArtifactMaterializationPlanSummary(
            persistent_enabled=persistent_plan.persistent_enabled,
            persistent_backend=persistent_plan.persistent_backend,
            analysis_output_dir=analysis_output_dir,
            runtime_resolved=runtime_resolved,
            filename_uses_source_identity=(
                output_plan.materialization_uses_source_identity_filename()
            ),
            note=(
                "Materialized path preview requires the compiled FunctionStep "
                "runtime plan."
            ),
        )

    preview = planned_materialization_preview(
        context=context,
        plan=execution_plan,
        output_key=output_key,
        output_plan=output_plan,
    )
    runtime_resolved = output_plan.materialization is None
    paths = ()
    filename_uses_source_identity = (
        output_plan.materialization_uses_source_identity_filename()
    )
    runtime_metadata_can_refine_paths = False
    if preview is not None:
        paths = tuple(
            ArtifactMaterializationPathSummary(
                group_key=path.group_key,
                base_path=path.base_path,
                candidate_paths=path.candidate_paths,
            )
            for path in preview.paths
        )
        filename_uses_source_identity = preview.filename_uses_source_identity
        runtime_metadata_can_refine_paths = preview.runtime_metadata_can_refine_paths

    note = None
    if runtime_resolved:
        note = (
            "Materialization spec is resolved from the runtime value schema "
            "during execution."
        )
    elif runtime_metadata_can_refine_paths:
        note = "Runtime payload metadata can split or refine candidate filenames."

    return ArtifactMaterializationPlanSummary(
        persistent_enabled=persistent_plan.persistent_enabled,
        persistent_backend=persistent_plan.persistent_backend,
        analysis_output_dir=str(execution_plan.artifact_analysis_output_dir),
        paths=paths,
        runtime_resolved=runtime_resolved,
        filename_uses_source_identity=filename_uses_source_identity,
        runtime_metadata_can_refine_paths=runtime_metadata_can_refine_paths,
        note=note,
    )


def _function_step_execution_plan(
    context,
    step_plan: CompiledStepPlan,
) -> FunctionStepExecutionPlan | None:
    try:
        return FunctionStepExecutionPlan.from_context(
            context,
            int(step_plan.step_index),
        )
    except (RuntimeError, ValueError):
        return None


def _analysis_output_dir(step_plan: CompiledStepPlan) -> str | None:
    if step_plan.materialized_output is not None:
        return step_plan.materialized_output.analysis_results_dir
    return step_plan.analysis_results_dir


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
        for full_virtual_path in full_virtual_paths[:MAX_INSPECTION_SOURCE_WORKSPACE_FILES]
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
