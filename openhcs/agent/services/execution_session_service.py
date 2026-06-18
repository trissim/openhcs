"""Opaque execution sessions for OpenHCS agent integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
from itertools import count
from pathlib import Path

from openhcs.agent.dto.common import AgentError, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.execution import (
    ExecutionConnectionSpec,
    ExecutionJobRef,
    ExecutionJobStatus,
    OrchestratorSession,
    OrchestratorSessionRef,
)
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.pipeline_authoring_service import PipelineAuthoringService
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.runtime.zmq_execution_client import (
    OpenHCSExecutionSubmission,
    ZMQExecutionClient,
)


class ExecutionJobKind(Enum):
    COMPILE = "compile"
    EXECUTE = "execute"


class ExecutionResponseDefault(Enum):
    SUBMITTED = "submitted"


@dataclass(frozen=True, slots=True)
class ExecutionResponseView:
    payload: JsonObject

    def status(self, fallback: str) -> str:
        if "status" in self.payload:
            return str(self.payload["status"])
        return fallback


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
class ExecutionSessionRecord:
    session: OrchestratorSession
    pipeline_steps: list
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
        )


@dataclass(frozen=True, slots=True)
class ExecutionJobRecord:
    ref: ExecutionJobRef
    response: JsonObject

    def status(self, response: JsonObject | None = None) -> ExecutionJobStatus:
        payload = self.response if response is None else response
        response_view = ExecutionResponseView(payload)
        return ExecutionJobStatus(
            schema_version=SCHEMA_VERSION,
            job_id=self.ref.job_id,
            session_id=self.ref.session_id,
            kind=self.ref.kind,
            status=response_view.status(self.ref.status),
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
        response_view = ExecutionResponseView(response)
        ref = ExecutionJobRef(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            session_id=session_id,
            kind=kind.value,
            status=response_view.status(ExecutionResponseDefault.SUBMITTED.value),
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
    ) -> None:
        self._path_policy = path_policy
        self._pipeline_service = pipeline_service
        self._config_service = config_service
        factory = client_factory or ZMQExecutionClientFactory()
        self._client_gateway = ExecutionClientGateway(factory)
        self._session_store = ExecutionSessionStore()
        self._job_store = ExecutionJobStore()

    def create_session(
        self,
        *,
        plate_path: str,
        pipeline_id: str,
        execution_plate_path: str | None = None,
        selected_pipeline_path: str | None = None,
        global_config_id: str | None = None,
        pipeline_config_id: str | None = None,
        connection: ExecutionConnectionSpec | None = None,
    ) -> OrchestratorSessionRef:
        plate = self._path_policy.assert_readable(plate_path)
        execution_plate = self._execution_plate_path(plate, execution_plate_path)
        selected_pipeline = self._selected_pipeline_path(selected_pipeline_path)
        pipeline_steps = self._pipeline_service.to_function_steps(pipeline_id)
        global_config = GlobalConfigSelection(global_config_id).resolve(
            self._config_service
        )
        pipeline_config = PipelineConfigSelection(pipeline_config_id).resolve(
            self._config_service
        )
        session_id = self._session_store.next_id()
        session = OrchestratorSession(
            schema_version=SCHEMA_VERSION,
            session_id=session_id,
            uri=ExecutionSessionStore.session_uri(session_id),
            plate_path=str(plate),
            execution_plate_path=str(execution_plate),
            selected_pipeline_path=_optional_path_text(selected_pipeline),
            pipeline_id=pipeline_id,
            global_config_id=global_config_id,
            pipeline_config_id=pipeline_config_id,
            connection=connection or ExecutionConnectionSpec(),
        )
        return self._session_store.store(
            ExecutionSessionRecord(
                session=session,
                pipeline_steps=pipeline_steps,
                configs=ExecutionConfigBundle(
                    global_pipeline=global_config,
                    plate_pipeline=pipeline_config,
                ),
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
        job = self._job(job_id)
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

    def _job(self, job_id: str) -> ExecutionJobRecord:
        return self._job_store.job_record(job_id)

    def _execution_plate_path(
        self,
        plate: Path,
        execution_plate_path: str | None,
    ) -> Path:
        if execution_plate_path is None:
            return plate
        return self._path_policy.assert_readable(execution_plate_path)

    def _selected_pipeline_path(self, selected_pipeline_path: str | None) -> Path | None:
        if selected_pipeline_path is None:
            return None
        return self._path_policy.assert_readable(selected_pipeline_path)


def _server_execution_id(response: JsonObject) -> str | None:
    execution_id = response.get("execution_id")
    if execution_id is None:
        return None
    return str(execution_id)


def _optional_path_text(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)
