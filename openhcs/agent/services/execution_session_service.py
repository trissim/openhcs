"""Opaque execution sessions for OpenHCS agent integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from itertools import count
from pathlib import Path
from types import ModuleType
import sys

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.common import AgentError, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.execution import (
    ArtifactPlanInspection,
    ArtifactPlanSummary,
    CompiledStepPlanSummary,
    ExecutionConnectionSpec,
    ExecutionJobRef,
    ExecutionJobStatus,
    OrchestratorSession,
    OrchestratorSessionRef,
    execution_status_from_response,
)
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.pipeline_authoring_service import PipelineAuthoringService
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.core.steps.abstract import AbstractStep
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
MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP = 16


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

        module_name = "openhcs_agent_compile_inspection"
        module = ModuleType(module_name)
        module.__file__ = f"<{module_name}>"
        sys.modules[module_name] = module
        try:
            exec(
                compile(request.pipeline_source, module.__file__, "exec"),
                module.__dict__,
            )
        finally:
            sys.modules.pop(module_name, None)

        pipeline_boundary = PipelineStepsNamespaceProjection(
            module.__dict__
        ).boundary_or_none()
        if pipeline_boundary is None:
            raise ValueError(
                f"Code must define {PipelineSourceExport.PIPELINE_STEPS.value!r}"
            )

        orchestrator = PipelineOrchestrator(
            plate_path=request.plate,
            pipeline_config=request.configs.plate_pipeline,
            progress_callback=None,
        )
        orchestrator.initialize()
        set_progress_queue(request.progress_queue)
        try:
            return dict(
                orchestrator.compile_pipelines(
                    pipeline_definition=pipeline_boundary.steps,
                    well_filter=list(request.axis_filter) or None,
                    is_zmq_execution=True,
                )
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
    def submit_compile(self, submission: OpenHCSExecutionSubmission) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def submit_pipeline(self, submission: OpenHCSExecutionSubmission) -> JsonObject:
        raise NotImplementedError

    @abstractmethod
    def get_status(self, execution_id=None) -> JsonObject:
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

    def submit_compile(self, submission: OpenHCSExecutionSubmission) -> JsonObject:
        return dict(self.client.submit_compile(submission))

    def submit_pipeline(self, submission: OpenHCSExecutionSubmission) -> JsonObject:
        return dict(self.client.submit_pipeline(submission))

    def get_status(self, execution_id=None) -> JsonObject:
        return dict(self.client.get_status(execution_id))

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
        return ExecutionJobStatus(
            schema_version=SCHEMA_VERSION,
            job_id=self.ref.job_id,
            session_id=self.ref.session_id,
            kind=self.ref.kind,
            status=execution_status_from_response(payload, fallback=self.ref.status),
            uri=self.ref.uri,
            server_execution_id=self.ref.server_execution_id,
            response=payload,
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
            raise KeyError(f"Unknown OpenHCS execution session_id: {session_id}") from exc

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
            raise KeyError(f"Unknown OpenHCS execution job_id: {job_id}") from exc

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
    ) -> JsonObject:
        client = self.factory.create_client(record.session.connection)
        submission = record.submission(compile_artifact_id)
        if kind is ExecutionJobKind.COMPILE:
            return dict(client.submit_compile(submission))
        return dict(client.submit_pipeline(submission))

    def status(
        self,
        session: OrchestratorSession,
        server_execution_id: str,
    ) -> JsonObject:
        client = self.factory.create_client(session.connection)
        return dict(client.get_status(server_execution_id))

    def wait(
        self,
        session: OrchestratorSession,
        server_execution_id: str,
    ) -> JsonObject:
        client = self.factory.create_client(session.connection)
        return dict(client.wait_for_completion(server_execution_id))


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

    def create_session_from_pipeline_source(
        self,
        request: PycodifiedPipelineSessionRequest,
    ) -> OrchestratorSessionRef:
        return self._create_session(request)

    def inspect_pipeline_source_artifact_plan(
        self,
        request: PycodifiedPipelineSessionRequest,
        *,
        axis_filter: tuple[str, ...] = (),
    ) -> ArtifactPlanInspection:
        progress_queue = AgentProgressQueue()
        plate = self._path_policy.assert_readable(request.identity.plate_id)
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
                errors=(AgentError.from_exception("compile_inspection_failed", exc),),
            )

        return artifact_plan_inspection_from_compilation(
            plate_path=str(plate),
            axis_filter=axis_filter,
            compilation=compilation,
            progress_event_count=len(progress_queue.events),
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

    def submit_compile(
        self,
        session_id: str,
        *,
        wait: bool = False,
    ) -> ExecutionJobRef | ExecutionJobStatus:
        return self._submit_job(
            session_id,
            ExecutionJobKind.COMPILE,
            wait=wait,
        )

    def submit_execution(
        self,
        session_id: str,
        *,
        compile_artifact_id: str | None = None,
        wait: bool = False,
    ) -> ExecutionJobRef | ExecutionJobStatus:
        return self._submit_job(
            session_id,
            ExecutionJobKind.EXECUTE,
            compile_artifact_id=compile_artifact_id,
            wait=wait,
        )

    def get_job_status(self, job_id: str) -> ExecutionJobStatus:
        job = self._job_store.job_record(job_id)
        if job.ref.server_execution_id is None:
            return job.status()
        try:
            session = self._session_store.session_record(job.ref.session_id).session
            response = self._client_gateway.status(
                session,
                job.ref.server_execution_id
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
    ) -> ExecutionJobRef | ExecutionJobStatus:
        record = self._session_store.session_record(session_id)
        response = self._client_gateway.submit(
            record,
            kind,
            compile_artifact_id,
        )
        ref = self._job_store.register(record.session.session_id, kind, response)
        if wait and ref.server_execution_id is not None:
            wait_response = self._client_gateway.wait(
                record.session,
                ref.server_execution_id,
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


def artifact_plan_inspection_from_compilation(
    *,
    plate_path: str,
    axis_filter: tuple[str, ...],
    compilation: JsonObject,
    progress_event_count: int,
) -> ArtifactPlanInspection:
    execution_bundle = compilation["execution_bundle"]
    compiled_contexts = dict(execution_bundle.runtime_contexts)
    axes = tuple(sorted(str(axis_id) for axis_id in compiled_contexts.keys()))
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
            for worker, axis_ids in dict(compilation["worker_assignments"]).items()
        },
        progress_event_count=progress_event_count,
    )


def _bounded_step_summaries(compiled_contexts, axes: tuple[str, ...]):
    emitted = 0
    for axis_id in axes:
        context = compiled_contexts[axis_id]
        for step_plan in context.step_plans.values():
            if emitted >= MAX_INSPECTION_STEPS:
                return
            emitted += 1
            artifact_outputs = tuple(
                _artifact_summary(plan)
                for plan in tuple(step_plan.artifact_outputs.values())[
                    :MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP
                ]
            )
            yield CompiledStepPlanSummary(
                step_index=int(step_plan.step_index),
                step_name=str(step_plan.step_name),
                axis_id=str(step_plan.axis_id),
                output_dir=_optional_path_text(step_plan.output_dir),
                execution_groups=tuple(step_plan.execution_groups),
                artifact_outputs=artifact_outputs,
                truncated_artifact_output_count=max(
                    0,
                    len(step_plan.artifact_outputs)
                    - MAX_INSPECTION_ARTIFACT_OUTPUTS_PER_STEP,
                ),
            )


def _artifact_summary(plan) -> ArtifactPlanSummary:
    paths_by_group = ()
    if plan.paths_by_group is not None:
        paths_by_group = tuple(
            {"group_key": group_key, "path": path}
            for group_key, path in plan.paths_by_group.items()
        )
    return ArtifactPlanSummary(
        name=str(plan.name),
        kind=str(plan.kind.value),
        path=str(plan.path),
        group_keys=tuple(plan.group_keys),
        paths_by_group=paths_by_group,
    )


def _optional_path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
