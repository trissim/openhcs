from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.services.execution_session_service import (
    ExecutionSessionService,
    PycodifiedPipelineSessionRequest,
)
from openhcs.agent.services.function_catalog_service import FunctionCatalogService
from openhcs.agent.services.llm_context_service import AgentAuthoringContextService
from openhcs.agent.services.pipeline_authoring_service import PipelineAuthoringService
from openhcs.agent.services.runtime_server_service import RuntimeServerService
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity


def sample_processing_function(image, sigma: float = 1.0):
    """Apply a small sample operation."""
    return image


def sample_large_signature_function(
    image,
    labels,
    object_name,
    measurement_name,
    bins: int = 256,
    threshold: float = 1.0,
):
    """Apply a sample operation with a long signature."""
    return image


def sample_gaussian_filter(image, sigma: float = 1.0):
    """Apply Gaussian smoothing."""
    return image


def sample_summary_function(image):
    """Summarize images that were produced by Gaussian filtering."""
    return image


@dataclass(frozen=True)
class _Metadata:
    func: Callable = sample_processing_function
    original_name: str = "sample_processing_function"
    name: str = "sample_processing_function"
    module: str = __name__
    doc: str = "Apply a small sample operation."
    tags: list[str] | None = None

    def get_registry_name(self) -> str:
        return "test"

    @classmethod
    def from_function(
        cls,
        func: Callable,
        doc: str,
        tags: list[str] | None = None,
    ) -> "_Metadata":
        return cls(
            func=func,
            original_name=func.__name__,
            name=func.__name__,
            doc=doc,
            tags=tags,
        )


def _catalog(monkeypatch):
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {"test:sample_processing_function": _Metadata(tags=[])},
    )
    return FunctionCatalogService()


class _ExecutionTestId:
    COMPILE = "compile-1"
    EXECUTE = "execute-1"


class _FakeExecutionClient:
    def __init__(self) -> None:
        self.compile_submissions = []
        self.execution_submissions = []

    def submit_compile(self, submission):
        self.compile_submissions.append(submission)
        return {"status": "accepted", "execution_id": _ExecutionTestId.COMPILE}

    def submit_pipeline(self, submission):
        self.execution_submissions.append(submission)
        return {"status": "accepted", "execution_id": _ExecutionTestId.EXECUTE}

    def get_status(self, execution_id=None):
        return {"status": "complete", "execution_id": execution_id}

    def wait_for_completion(self, execution_id: str):
        return {"status": "complete", "execution_id": execution_id}


class _FakeExecutionClientFactory:
    def __init__(self, client: _FakeExecutionClient) -> None:
        self.client = client

    def create_client(self, connection):
        return self.client


class _FakeRuntimeServerGateway:
    def __init__(self) -> None:
        self.server_info_connections = []
        self.execution_status_requests = []
        self.scan_requests = []

    def server_info(self, connection):
        self.server_info_connections.append(connection)
        return {
            "port": connection.port,
            "ready": True,
            "server": "OpenHCSExecutionServer",
            "control_port": 6555,
            "active_executions": 1,
            "running_executions": [{"execution_id": _ExecutionTestId.EXECUTE}],
            "queued_executions": [],
            "workers": [{"worker_id": "worker-1"}],
            "uptime": 12.5,
            "log_file_path": "/tmp/openhcs-runtime.log",
        }

    def execution_status(self, connection, execution_id=None):
        self.execution_status_requests.append((connection, execution_id))
        return {"status": "complete", "execution_id": execution_id}

    def scan(
        self,
        *,
        host: str,
        ports: tuple[int, ...],
        transport_mode: str | None,
        timeout_ms: int,
    ):
        self.scan_requests.append((host, ports, transport_mode, timeout_ms))
        return tuple(
            {
                "port": port,
                "ready": True,
                "server": "OpenHCSExecutionServer",
                "active_executions": 0,
                "running_executions": [],
                "queued_executions": [],
                "workers": [],
            }
            for port in ports
        )


def test_function_catalog_search_and_describe_use_registry_ids(monkeypatch):
    catalog = _catalog(monkeypatch)

    page = catalog.search(query="sample")
    detail = catalog.get("test:sample_processing_function")

    assert page.items[0].function_id == "test:sample_processing_function"
    assert detail.entry.signature == "sample_processing_function(image, sigma=1.0)"
    assert [parameter.name for parameter in detail.parameters] == ["image", "sigma"]


def test_function_catalog_search_can_return_compact_signatures(monkeypatch):
    metadata = _Metadata.from_function(
        sample_large_signature_function,
        "Apply a sample operation with a long signature.",
        [],
    )
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {"test:sample_large_signature_function": metadata},
    )
    catalog = FunctionCatalogService()

    page = catalog.search(query="large", compact_signatures=True)
    detail = catalog.get("test:sample_large_signature_function")

    assert page.items[0].signature == (
        "sample_large_signature_function(image, labels, object_name, measurement_name, ...)"
    )
    assert "bins=256" in detail.entry.signature


def test_function_catalog_search_ranks_name_matches_before_doc_matches(monkeypatch):
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self: {
            "test:sample_summary_function": _Metadata.from_function(
                sample_summary_function,
                "Summarize images that were produced by Gaussian filtering.",
                [],
            ),
            "test:sample_gaussian_filter": _Metadata.from_function(
                sample_gaussian_filter,
                "Apply Gaussian smoothing.",
                [],
            ),
        },
    )
    catalog = FunctionCatalogService()

    page = catalog.search(query="gaussian", compact_signatures=True)
    phrase_page = catalog.search(query="test gaussian filter", compact_signatures=True)

    assert page.items[0].function_id == "test:sample_gaussian_filter"
    assert phrase_page.items[0].function_id == "test:sample_gaussian_filter"


def test_config_service_reflects_pipeline_schema_without_materializing_lazy_values():
    service = ConfigService()

    schema = service.describe_schema("pipeline")
    well_filter = next(
        field for field in schema.fields if field.path == "well_filter_config"
    )

    assert schema.config_type == "PipelineConfig"
    assert well_filter.lazy is True
    assert well_filter.default_repr.endswith("LazyWellFilterConfig()")


def test_config_service_validates_and_renders_config_source():
    service = ConfigService()

    result = service.validate_patch(
        "global",
        ConfigPatch(config_type="GlobalPipelineConfig", values={"num_workers": 2}),
    )
    rendered = service.render_source(result.config_ref)

    assert result.valid is True
    assert "num_workers=2" in rendered.source


def test_pipeline_authoring_service_renders_function_step_source(monkeypatch):
    pipeline_service = PipelineAuthoringService(_catalog(monkeypatch))
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(
        function_id="test:sample_processing_function",
        kwargs={"sigma": 2.0},
    )

    pipeline_service.add_step(pipeline_ref, step)
    validation = pipeline_service.validate(pipeline_ref)
    rendered = pipeline_service.render_source(pipeline_ref)

    assert validation.valid is True
    assert "FunctionStep" in rendered.source
    assert "sample_processing_function" in rendered.source
    assert "sigma" in rendered.source


def test_execution_session_service_submits_compile_and_execution_jobs(
    monkeypatch,
    tmp_path: Path,
):
    pipeline_service = PipelineAuthoringService(_catalog(monkeypatch))
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(
        function_id="test:sample_processing_function",
        kwargs={"sigma": 2.0},
    )
    pipeline_service.add_step(pipeline_ref, step)
    fake_client = _FakeExecutionClient()
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=pipeline_service,
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(fake_client),
    )

    session_ref = execution_service.create_session(
        plate_path=str(tmp_path),
        pipeline_id=pipeline_ref.pipeline_id,
    )
    compile_ref = execution_service.submit_compile(session_ref.session_id)
    execute_ref = execution_service.submit_execution(
        session_ref.session_id,
        compile_artifact_id=_ExecutionTestId.COMPILE,
    )
    compile_status = execution_service.get_job_status(compile_ref.job_id)

    assert compile_ref.server_execution_id == _ExecutionTestId.COMPILE
    assert execute_ref.server_execution_id == _ExecutionTestId.EXECUTE
    assert compile_status.status == "complete"
    assert fake_client.compile_submissions[0].plate_id == str(tmp_path.resolve())
    assert fake_client.execution_submissions[0].compile_artifact_id == _ExecutionTestId.COMPILE


def test_execution_session_service_preserves_pycodified_pipeline_source(
    monkeypatch,
    tmp_path: Path,
):
    pipeline_source = "pipeline_steps = []\n"
    fake_client = _FakeExecutionClient()
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(fake_client),
    )

    session_ref = execution_service.create_session_from_pipeline_source(
        PycodifiedPipelineSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=pipeline_source,
            global_config_id=None,
            pipeline_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    execution_service.submit_compile(session_ref.session_id)

    submission = fake_client.compile_submissions[0]
    assert submission.pipeline_source == pipeline_source
    assert submission.pipeline_steps == []


def test_runtime_server_service_reads_runtime_server_state():
    gateway = _FakeRuntimeServerGateway()
    service = RuntimeServerService(gateway=gateway)

    server_info = service.server_info(port=5555)
    scan_result = service.scan(ports=(5555, 5555, 7777), timeout_ms=25)
    execution_status = service.execution_status(
        execution_id=_ExecutionTestId.EXECUTE,
        port=5555,
    )

    assert server_info.reachable is True
    assert server_info.server == "OpenHCSExecutionServer"
    assert server_info.running_executions[0]["execution_id"] == _ExecutionTestId.EXECUTE
    assert scan_result.ports == (5555, 7777)
    assert [server.port for server in scan_result.servers] == [5555, 7777]
    assert execution_status.status == "complete"
    assert gateway.execution_status_requests[0][1] == _ExecutionTestId.EXECUTE


def test_authoring_context_uses_function_catalog(monkeypatch):
    context = AgentAuthoringContextService(_catalog(monkeypatch)).get_authoring_context()

    assert context.kind == "pipeline"
    assert "CONFIG SCHEMA HINTS" in context.content
    assert "test:sample_processing_function" in context.content
