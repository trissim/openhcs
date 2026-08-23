from __future__ import annotations

import ast
import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from polystore.virtual_workspace import SourcePixelRef
from pyqt_reactive.services.parameter_help_service import (
    dataclass_parameter_descriptions,
)
from pyqt_reactive.services.window_snapshot import WindowSnapshotCaptureScope
from zmqruntime.config import TransportMode
from zmqruntime.execution import ExecutionProgressObservation
from zmqruntime.execution.server import ExecutionServer
from zmqruntime.messages import (
    PongResponse,
    RunningExecutionInfo,
    ServerRole,
    WorkerState,
)

from openhcs.agent.dto.authoring import AuthoringContextRequest
from openhcs.agent.dto.config import ConfigPatch
from openhcs.agent.dto.execution import (
    MAX_EXECUTION_STATUS_TRACEBACK_CHARS,
    ExecutionConnectionSpec,
    OrchestratorSessionCreationRequest,
    PipelineSourceArtifactPlanInspectionRequest,
    PipelineSourceOrchestratorSessionRequest,
)
from openhcs.agent.dto.pipeline import CreatePipelineRequest
from openhcs.agent.dto.viewer import (
    ViewerWindowLayerIsolationRequest,
    ViewerWindowLayerVisibilityRecord,
    ViewerWindowNavigationRequest,
    ViewerWindowPayloadRequest,
    ViewerWindowSnapshotRequest,
    ViewerWindowStateRequest,
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
)
from openhcs.agent.path_policy import AgentPathPolicy
from openhcs.agent.services import function_catalog_service as function_catalog_module
from openhcs.agent.services import viewer_window_service as viewer_window_service_module
from openhcs.agent.services.config_service import ConfigService
from openhcs.agent.services.execution_session_service import (
    ExecutionSessionService,
    PipelineSourceSessionRequest,
    artifact_plan_inspection_from_compilation,
)
from openhcs.agent.services.function_catalog_service import (
    AgentFunctionSearchPolicy,
    FunctionCatalogService,
    SignatureView,
    SummaryView,
)
from openhcs.agent.services.llm_context_service import AgentAuthoringContextService
from openhcs.agent.services.pipeline_authoring_service import (
    InvalidFunctionKwargsError,
    MissingFunctionKwargsError,
    PipelineAuthoringService,
)
from openhcs.agent.services.runtime_server_service import RuntimeServerService
from openhcs.agent.services.viewer_window_service import (
    ViewerWindowGatewayABC,
    ViewerWindowService,
    ZMQViewerWindowGateway,
)
from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ObjectLabelsArtifactType,
    SpecialArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.compiled_step_plan import CompiledStepPlan, MaterializedOutputPlan
from openhcs.core.component_group_scope import ComponentGroupScope
from openhcs.core.config import Backend, NapariStreamingConfig, PipelineConfig
from openhcs.core.config_document import ConfigDocumentAuthority
from openhcs.core.pipeline.function_contracts import artifact_inputs, artifact_outputs
from openhcs.core.pipeline_document import PipelineDocumentAuthority
from openhcs.core.source_bindings import (
    LazySourceBindingsConfig,
    MetadataExtractionRule,
    NamedSourceBinding,
    SourceFilterClause,
    SourceSelector,
)
from openhcs.core.source_workspace_projection import VirtualWorkspaceSourceProjection
from openhcs.core.streaming_config_declarations import ViewerType
from openhcs.microscopes.exceptions import MicroscopePixelSizeUnavailableError
from openhcs.runtime.viewer_protocol import (
    ViewerNavigationControlOptions,
    ViewerPayloadControlOptions,
    ViewerStateControlOptions,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_execution_client import (
    ExecutionSubmissionPreparationTimeoutError,
)
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity
from openhcs.serialization.json import to_jsonable


def sample_processing_function(image, sigma: float = 1.0):
    """Apply a small sample operation."""
    return image


def sample_runtime_context_function(*, context, scale: float = 1.0):
    """Apply an operation using runtime execution context."""
    return context, scale


@artifact_outputs(ArtifactSpec.output("objects", ObjectLabelsArtifactType))
@artifact_inputs(ArtifactSpec.input("positions", SpecialArtifactType))
def sample_artifact_contract_function(image, positions):
    """Apply a sample operation with canonical artifact declarations."""
    return image


def sample_required_parameter_function(image, threshold: float):
    """Apply a sample operation with a required agent parameter."""
    return image


def _viewer_connection(port: int = 5584) -> ExecutionConnectionSpec:
    return ExecutionConnectionSpec(port=port)


def test_execution_connection_spec_owns_zmq_endpoint_projection():
    connection = ExecutionConnectionSpec(
        host="127.0.0.1",
        port=5555,
        transport_mode=TransportMode.TCP,
    )

    assert connection.zmq_data_url(OPENHCS_ZMQ_CONFIG) == "tcp://127.0.0.1:5555"
    assert connection.zmq_control_port(OPENHCS_ZMQ_CONFIG) == 6555
    assert connection.zmq_control_url(OPENHCS_ZMQ_CONFIG) == "tcp://127.0.0.1:6555"
    with pytest.raises(ValueError, match="ZMQ data URL requires an explicit port"):
        ExecutionConnectionSpec().zmq_data_url(OPENHCS_ZMQ_CONFIG)


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


def sample_no_doc_function(image):
    return image


def identify_primary_objects(image):
    """Identify bright primary objects in a grayscale image."""
    return image


def measure_object_intensity(image, labels):
    """Measure intensity features for identified objects."""
    return image


def measure_colocalization(image):
    """Measure colocalization between fluorescence channels."""
    return image


def track_objects(image):
    """Track objects through time."""
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

    @property
    def display_name(self) -> str:
        if self.original_name:
            return self.original_name
        return self.name

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
        lambda self, **_kwargs: {"test:sample_processing_function": _Metadata(tags=[])},
    )
    return FunctionCatalogService()


def _pipeline_document_source(
    pipeline_config: PipelineConfig | None = None,
) -> str:
    config = pipeline_config if pipeline_config is not None else PipelineConfig()
    return PipelineDocumentAuthority.render(
        PipelineDocumentAuthority.from_values(
            pipeline_config=config,
            pipeline_steps=[],
        )
    )


def test_execution_session_request_contracts_do_not_mirror_pipeline_config_id():
    request_types = (
        OrchestratorSessionCreationRequest,
        PipelineSourceOrchestratorSessionRequest,
        PipelineSourceArtifactPlanInspectionRequest,
    )

    assert all(
        "pipeline_config_id" not in {field.name for field in fields(request_type)}
        for request_type in request_types
    )


class _ExecutionTestId:
    COMPILE = "compile-1"
    EXECUTE = "execute-1"


class _FakeExecutionClient:
    def __init__(self) -> None:
        self.compile_submissions = []
        self.execution_submissions = []
        self.status_requests = []
        self.wait_requests = []
        self.submit_timeout_requests = []
        self.progress_by_execution_id = {}
        self.disconnect_count = 0

    def submit_compile(
        self,
        submission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.submit_timeout_requests.append(("compile", timeout_ms))
        self.compile_submissions.append(submission)
        return {"status": "accepted", "execution_id": _ExecutionTestId.COMPILE}

    def submit_pipeline(
        self,
        submission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.submit_timeout_requests.append(("execute", timeout_ms))
        self.execution_submissions.append(submission)
        return {"status": "accepted", "execution_id": _ExecutionTestId.EXECUTE}

    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.status_requests.append((execution_id, timeout_ms))
        return {"status": "complete", "execution_id": execution_id}

    def wait_for_completion(self, execution_id: str):
        self.wait_requests.append(execution_id)
        return {"status": "complete", "execution_id": execution_id}

    def progress_observation(self, execution_id: str):
        return self.progress_by_execution_id.get(execution_id)

    def disconnect(self) -> None:
        self.disconnect_count += 1


class _EnvelopeStatusExecutionClient(_FakeExecutionClient):
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.status_requests.append((execution_id, timeout_ms))
        return {
            "status": "ok",
            "execution": {
                "status": "running",
                "execution_id": execution_id,
            },
        }


class _DisconnectFailureExecutionClient(_FakeExecutionClient):
    def disconnect(self) -> None:
        super().disconnect()
        raise RuntimeError("synthetic disconnect failure")


class _HeadlessCompleteExecutionClient(_FakeExecutionClient):
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.status_requests.append((execution_id, timeout_ms))
        return {
            "status": "ok",
            "execution": {
                "status": "complete",
                "execution_id": execution_id,
                "results_summary": {
                    "output_plate_root": "/tmp/source_openhcs",
                    "auto_add_output_plate_to_plate_manager": False,
                },
            },
        }


class _FailedStatusExecutionClient(_FakeExecutionClient):
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.status_requests.append((execution_id, timeout_ms))
        return {
            "status": "ok",
            "execution": {
                "status": "failed",
                "execution_id": execution_id,
                "error": "synthetic failure",
                "traceback": "T" * (MAX_EXECUTION_STATUS_TRACEBACK_CHARS + 200),
            },
        }


class _CustomFunctionImportFailedStatusExecutionClient(_FakeExecutionClient):
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.status_requests.append((execution_id, timeout_ms))
        message = (
            "cannot import name 'agent_threshold_mask' from "
            "'openhcs.processing.custom_functions'"
        )
        return {
            "status": "ok",
            "execution": {
                "status": "failed",
                "execution_id": execution_id,
                "error": message,
                "traceback": f"ImportError: {message}",
            },
        }


class _TimeoutStatusExecutionClient(_FakeExecutionClient):
    def get_status(
        self,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.status_requests.append((execution_id, timeout_ms))
        raise TimeoutError(f"status timed out after {timeout_ms}ms")


class _TimeoutSubmitExecutionClient(_FakeExecutionClient):
    def submit_compile(
        self,
        submission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.submit_timeout_requests.append(("compile", timeout_ms))
        raise TimeoutError(f"submit timed out after {timeout_ms}ms")


class _PreparationTimeoutSubmitExecutionClient(_FakeExecutionClient):
    def submit_compile(
        self,
        submission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.submit_timeout_requests.append(("compile", timeout_ms))
        raise ExecutionSubmissionPreparationTimeoutError("No execute request was sent.")


class _SlowSubmitExecutionClient(_FakeExecutionClient):
    def submit_compile(
        self,
        submission,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.submit_timeout_requests.append(("compile", timeout_ms))
        time.sleep(0.05)
        self.compile_submissions.append(submission)
        return {"status": "accepted", "execution_id": _ExecutionTestId.COMPILE}


class _FakeExecutionClientFactory:
    def __init__(self, client: _FakeExecutionClient) -> None:
        self.client = client
        self.creation_count = 0

    def create_client(self, connection):
        self.creation_count += 1
        return self.client


class _FakeCompileInspectionGateway:
    def __init__(self) -> None:
        self.requests = []

    def compile(self, request):
        self.requests.append(request)
        request.progress_queue.put({"phase": "compile", "status": "running"})
        step_plan = CompiledStepPlan(
            step_index=0,
            step_name="WriteArtifacts",
            step_type="FunctionStep",
            axis_id="A01",
            output_dir=Path("/tmp/out/A01"),
            main_flow_axis_persistence_enabled=False,
        )
        step_plan.execution_group_scope = ComponentGroupScope.from_raw(
            ("A01",),
            component=AllComponents.WELL,
        )
        step_plan.materialized_output = MaterializedOutputPlan(
            output_dir=Path("/tmp/out/A01/checkpoints"),
            backend="disk",
            plate_root="/tmp/out",
            sub_dir="checkpoints",
            analysis_results_dir="/tmp/out/A01/checkpoints/analysis",
        )
        step_plan.streaming_configs["napari_streaming_config"] = NapariStreamingConfig(
            enabled=True,
            persistent=True,
            well_filter=["A01"],
        )
        input_plan = ArtifactInputPlan(
            name="positions",
            artifact_type=SpecialArtifactType,
            path="/tmp/out/A01/positions.pkl",
            group_component=AllComponents.WELL,
            group_keys=("A01",),
            paths_by_group={"A01": "/tmp/out/A01/positions.pkl"},
            source_step_id=0,
            source_step_scope_id="step-find-positions",
        )
        step_plan.artifact_inputs[input_plan.ref()] = input_plan
        output_plan = ArtifactOutputPlan(
            name="objects",
            artifact_type=ObjectLabelsArtifactType,
            path="/tmp/out/A01/objects.zarr",
            group_component=AllComponents.WELL,
            group_keys=("A01",),
            paths_by_group={"A01": "/tmp/out/A01/objects.zarr"},
        )
        step_plan.artifact_outputs[output_plan.ref()] = output_plan
        context = SimpleNamespace(step_plans={0: step_plan})
        virtual_name = "A01_s001_w1_z001_t001.tif"
        full_virtual_path = str(request.plate / virtual_name)
        return {
            "execution_bundle": SimpleNamespace(
                runtime_contexts={"A01": context},
                worker_assignments={"worker-1": ["A01"]},
            ),
            "source_workspace_projection": VirtualWorkspaceSourceProjection(
                source_refs_by_virtual_path={
                    virtual_name: SourcePixelRef(
                        "disk", str(request.plate / "source.tif")
                    ),
                    full_virtual_path: SourcePixelRef(
                        "disk", str(request.plate / "source.tif")
                    ),
                },
                source_metadata_by_path={
                    virtual_name: {"well": "A01", "channel": "1"},
                    full_virtual_path: {"well": "A01", "channel": "1"},
                },
                workspace_root=str(request.plate),
            ),
        }


class _FailingCompileInspectionGateway:
    def __init__(self, exception: Exception) -> None:
        self.exception = exception

    def compile(self, request):
        request.progress_queue.put({"phase": "compile", "status": "running"})
        raise self.exception


class _WorkspacePreparingCompileInspectionGateway(_FakeCompileInspectionGateway):
    def compile(self, request):
        (request.plate / "openhcs_metadata.json").write_text(
            "{}",
            encoding="utf-8",
        )
        return super().compile(request)


class _FakeRuntimeServerGateway:
    def __init__(self) -> None:
        self.server_info_connections = []
        self.execution_status_requests = []
        self.scan_requests = []

    def server_info(self, connection, *, timeout_ms: int):
        self.server_info_connections.append((connection, timeout_ms))
        return PongResponse(
            port=connection.port,
            control_port=6555,
            ready=True,
            server="OpenHCSExecutionServer",
            server_type=ExecutionServer.server_type(),
            server_role=ServerRole.EXECUTION,
            active_executions=1,
            running_executions=(
                RunningExecutionInfo(
                    execution_id=_ExecutionTestId.EXECUTE,
                    plate_id="plate-1",
                    start_time=0.0,
                    elapsed=0.0,
                ),
            ),
            queued_executions=(),
            workers=(
                WorkerState(
                    pid=1,
                    status="running",
                    cpu_percent=0.0,
                    memory_mb=0.0,
                ),
            ),
            uptime=12.5,
            log_file_path="/tmp/openhcs-runtime.log",
        )

    def execution_status(
        self,
        connection,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.execution_status_requests.append((connection, execution_id, timeout_ms))
        return {"status": "complete", "execution_id": execution_id}

    def scan(
        self,
        *,
        host: str,
        ports: tuple[int, ...],
        transport_mode: TransportMode | None,
        timeout_ms: int,
    ):
        self.scan_requests.append((host, ports, transport_mode, timeout_ms))
        return tuple(
            PongResponse(
                port=port,
                control_port=port + 1000,
                ready=True,
                server="OpenHCSExecutionServer",
                server_type=ExecutionServer.server_type(),
                server_role=ServerRole.EXECUTION,
                active_executions=0,
                running_executions=(),
                queued_executions=(),
                workers=(),
            )
            for port in ports
        )


class _WrongKindRuntimeServerGateway(_FakeRuntimeServerGateway):
    def server_info(self, connection, *, timeout_ms: int):
        self.server_info_connections.append((connection, timeout_ms))
        return PongResponse(
            port=connection.port,
            control_port=6555,
            ready=True,
            server="NapariViewer",
            server_type="napari",
            server_role=ServerRole.VIEWER,
        )


class _MixedRoleRuntimeServerGateway(_FakeRuntimeServerGateway):
    def scan(
        self,
        *,
        host: str,
        ports: tuple[int, ...],
        transport_mode: TransportMode | None,
        timeout_ms: int,
    ):
        execution_servers = super().scan(
            host=host,
            ports=ports,
            transport_mode=transport_mode,
            timeout_ms=timeout_ms,
        )
        return (
            *execution_servers,
            PongResponse(
                port=5584,
                control_port=5585,
                ready=True,
                server="NapariViewer",
                server_type="napari",
                server_role=ServerRole.VIEWER,
            ),
        )


class _FailedRuntimeStatusGateway(_FakeRuntimeServerGateway):
    def execution_status(
        self,
        connection,
        execution_id=None,
        *,
        timeout_ms: int = OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    ):
        self.execution_status_requests.append((connection, execution_id, timeout_ms))
        return {
            "status": "ok",
            "execution": {
                "status": "failed",
                "execution_id": execution_id,
                "error": "runtime failure",
                "traceback": "R" * (MAX_EXECUTION_STATUS_TRACEBACK_CHARS + 17),
            },
        }


class _FakeViewerWindowGateway(ViewerWindowGatewayABC):
    def __init__(self) -> None:
        self.requests = []

    def snapshot_window(self, request):
        self.requests.append(request)
        return {
            "status": "success",
            "viewer": {
                "type": "napari",
                "title": "OpenHCS Napari Viewer",
            },
            "width": 640,
            "height": 480,
            "snapshot": request,
            "resource": {
                "uri": "file:///tmp/napari.png",
                "title": "OpenHCS Napari Viewer",
                "mime_type": "image/png",
                "path": "/tmp/napari.png",
                "size_bytes": 456,
                "sha256": "def456",
            },
        }

    def window_state(self, request):
        self.requests.append(request)
        return {
            "status": "success",
            "viewer": {
                "type": "napari",
                "title": "OpenHCS Napari Viewer",
            },
            "layer_count": 1,
            "layers": (
                {
                    "route_key": "IdentifyPrimaryObjects|image",
                    "producer_identities": (
                        {
                            "origin": "pipeline",
                            "output_kind": "main",
                            "output_key": "main",
                            "projection_key": "main",
                            "step_name": "IdentifyPrimaryObjects",
                            "pipeline_position": 3,
                            "step_scope_id": "identify-primary",
                            "invocation_key": None,
                            "artifact_kind": "image",
                        },
                    ),
                    "title": "IdentifyPrimaryObjects",
                    "mounted": True,
                    "item_count": 2,
                    "data_types": ("image",),
                    "component_values": (
                        {"well": "A14", "site": 1, "channel": 0},
                        {"well": "B13", "site": 1, "channel": 0},
                    ),
                    "component_value_count": 2,
                    "component_values_truncated": False,
                    "payload_summaries": (
                        {
                            "data_type": "image",
                            "path": "/tmp/A14.tif",
                            "components": {"well": "A14", "site": 1, "channel": 0},
                            "payload_type": "ndarray",
                            "shape": (16, 16),
                            "spatial_origin_yx": (0, 0),
                            "source_spatial_shape_yx": (16, 16),
                            "dtype": "uint16",
                            "size": 256,
                            "nonzero_count": 128,
                        },
                        {
                            "data_type": "image",
                            "path": "/tmp/B13.tif",
                            "components": {"well": "B13", "site": 1, "channel": 0},
                            "payload_type": "ndarray",
                            "shape": (16, 16),
                            "spatial_origin_yx": (0, 0),
                            "source_spatial_shape_yx": (16, 16),
                            "dtype": "uint16",
                            "size": 256,
                            "nonzero_count": 96,
                        },
                    ),
                    "payload_summary_count": 2,
                    "payload_summaries_truncated": False,
                    "axis_labels": ("well", "site", "channel", "y", "x"),
                    "stack_axes": ("well", "site", "channel"),
                    "axis_offsets": (0, 0, 0),
                    "scalar_labels": (),
                    "labels": {
                        "well": ("A14", "B13"),
                        "site": ("1",),
                        "channel": ("0",),
                    },
                    "axis_component_values": {
                        "well": ("A14", "B13"),
                        "site": (1,),
                        "channel": (0,),
                    },
                    "routed_component_values": {
                        "well": ("A14", "B13"),
                        "site": (1,),
                        "channel": (0,),
                    },
                    "data_shape": (2, 1, 1, 16, 16),
                    "translate": (0.0, 0.0, 0.0, 0.0, 0.0),
                    "visible": True,
                    "selected": True,
                    "pending_update": False,
                },
            ),
            "active_dimension_label_route": "IdentifyPrimaryObjects|image",
            "viewer_ndim": 5,
            "current_step": (0, 0, 0, 0, 0),
            "axis_labels": ("well", "site", "channel", "y", "x"),
            "component_group_count": 1,
            "component_item_count": 2,
        }

    def window_payloads(self, request):
        self.requests.append(request)
        return {
            "status": "success",
            "viewer": {
                "type": "napari",
                "title": "OpenHCS Napari Viewer",
            },
            "layer_count": 1,
            "layers": (
                {
                    "route_key": "IdentifyPrimaryObjects|image",
                    "producer_identities": (
                        {
                            "origin": "pipeline",
                            "output_kind": "main",
                            "output_key": "main",
                            "projection_key": "main",
                            "step_name": "IdentifyPrimaryObjects",
                            "pipeline_position": 3,
                            "step_scope_id": "identify-primary",
                            "invocation_key": None,
                            "artifact_kind": "image",
                        },
                    ),
                    "title": "IdentifyPrimaryObjects",
                    "mounted": True,
                    "item_count": 2,
                    "axis_labels": ("well", "site", "channel", "y", "x"),
                    "stack_axes": ("well", "site", "channel"),
                    "pending_update": False,
                    "payloads": (
                        {
                            "route_key": "IdentifyPrimaryObjects|image",
                            "data_type": "image",
                            "path": "/tmp/A14.tif",
                            "components": {
                                "well": "A14",
                                "site": 1,
                                "channel": 0,
                            },
                            "axis_indices": (0, 0, 0),
                            "aggregate_axis_indices": (),
                            "summary": {
                                "payload_type": "ndarray",
                                "shape": (16, 16),
                                "dtype": "uint16",
                                "size": 256,
                                "nonzero_count": 128,
                            },
                            "array_values": (1, 2, 3),
                            "array_value_summary": {
                                "requested": True,
                                "included": True,
                                "shape": (3,),
                            },
                            "shape_payloads": (
                                {
                                    "shape_type": "rectangle",
                                    "axis_indices": (0, 0, 0),
                                    "bounds": (2.0, 3.0, 8.0, 9.0),
                                },
                            ),
                        },
                    ),
                },
            ),
        }

    def navigate_window(self, request):
        self.requests.append(request)
        response = self.window_state(request)
        self.requests.pop()
        layer = dict(response["layers"][0])
        if request.navigation.visible is not None:
            layer["visible"] = request.navigation.visible
        if request.navigation.selected is not None:
            layer["selected"] = request.navigation.selected
        response["layers"] = (layer,)
        response["current_step"] = (
            request.navigation.axis_indices.get("well", 0),
            request.navigation.axis_indices.get("site", 0),
            request.navigation.axis_indices.get("channel", 0),
            0,
            0,
        )
        return response


class _UnmountedRouteViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        response = super().window_state(request)
        mounted_layer = response["layers"][0]
        response["layer_count"] = 2
        response["layers"] = (
            mounted_layer,
            {
                **mounted_layer,
                "route_key": "selected_rois",
                "title": "Selected ROIs",
                "mounted": False,
                "visible": False,
                "selected": False,
            },
        )
        return response


class _MalformedViewerWindowGateway(ViewerWindowGatewayABC):
    def snapshot_window(self, request):
        del request
        return {"status": "error"}

    def window_state(self, request):
        del request
        return {"status": "success", "layers": ()}

    def window_payloads(self, request):
        del request
        return {"status": "success", "layers": ()}

    def navigate_window(self, request):
        del request
        return {"status": "success", "layers": ()}


class _CompactStateViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["component_values"] = ()
        layer["component_value_count"] = 2
        layer["component_values_truncated"] = True
        layer["payload_summaries"] = ()
        layer["payload_summary_count"] = 2
        layer["payload_summaries_truncated"] = True
        state["layers"] = (layer,)
        return state


class _CoordinateGapViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["axis_component_values"] = {
            "well": ("A14", "B13"),
            "site": (1,),
            "channel": (0, 1, 2),
        }
        layer["routed_component_values"] = {
            "well": ("A14", "B13"),
            "site": (1,),
            "channel": (0, 2),
        }
        state["layers"] = (layer,)
        return state


class _UnroutedAxisPositionViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["axis_component_values"] = {
            "well": ("A14", "B13"),
            "site": (1,),
            "channel": (0, 1, 2),
        }
        state["layers"] = (layer,)
        return state


class _CollapsedComponentViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["item_count"] = 1
        layer["component_values"] = ({"well": "A14", "site": 1, "channel": 0},)
        layer["payload_summaries"] = (layer["payload_summaries"][0],)
        layer["payload_summary_count"] = 1
        layer["axis_labels"] = ("channel", "y", "x")
        layer["stack_axes"] = ("channel",)
        layer["axis_offsets"] = (0,)
        layer["labels"] = {
            "channel": ("0",),
        }
        layer["axis_component_values"] = {
            "channel": (0,),
        }
        layer["routed_component_values"] = {
            "channel": (0,),
        }
        layer["data_shape"] = (1, 16, 16)
        layer["translate"] = (0.0, 0.0, 0.0)
        state["layers"] = (layer,)
        state["viewer_ndim"] = 3
        state["current_step"] = (0, 0, 0)
        state["axis_labels"] = ("channel", "y", "x")
        state["component_item_count"] = 1
        return state


class _RgbViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["data_shape"] = (2, 1, 1, 16, 16, 3)
        layer["payload_summaries"] = tuple(
            {
                **payload_summary,
                "shape": (16, 16, 3),
                "spatial_origin_yx": (0, 0),
                "source_spatial_shape_yx": (16, 16),
                "size": 16 * 16 * 3,
            }
            for payload_summary in layer["payload_summaries"]
        )
        state["layers"] = (layer,)
        return state


class _PaddedVariableSizeViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["data_shape"] = (2, 1, 1, 20, 20)
        layer["payload_summaries"] = (
            {
                **layer["payload_summaries"][0],
                "shape": (16, 20),
                "spatial_origin_yx": (0, 0),
                "source_spatial_shape_yx": (16, 20),
            },
            {
                **layer["payload_summaries"][1],
                "shape": (20, 18),
                "spatial_origin_yx": (0, 0),
                "source_spatial_shape_yx": (20, 18),
            },
        )
        state["layers"] = (layer,)
        return state


class _AggregateStackViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["item_count"] = 1
        layer["component_values"] = ({"well": "A14", "site": 1, "channel": 1},)
        layer["payload_summaries"] = (
            {
                "data_type": "image",
                "path": "/tmp/A14_stack.tif",
                "components": {"well": "A14", "site": 1, "channel": 1},
                "aggregate_component_values": {"z_index": (1, 2, 3)},
                "payload_type": "ndarray",
                "shape": (3, 16, 16),
                "dtype": "uint16",
                "size": 3 * 16 * 16,
                "nonzero_count": 384,
            },
        )
        layer["axis_labels"] = ("z_index", "channel", "y", "x")
        layer["stack_axes"] = ("z_index", "channel")
        layer["labels"] = {
            "z_index": ("1", "2", "3"),
            "channel": ("1",),
        }
        layer["axis_component_values"] = {
            "z_index": (1, 2, 3),
            "channel": (1,),
        }
        layer["routed_component_values"] = {
            "z_index": (1, 2, 3),
            "channel": (1,),
        }
        layer["data_shape"] = (3, 1, 16, 16)
        state["layers"] = (layer,)
        state["axis_labels"] = ("z_index", "channel", "y", "x")
        state["viewer_ndim"] = 4
        state["component_item_count"] = 1
        return state


class _SilentZMQSocket:
    def __init__(self) -> None:
        self.closed = False
        self.sent_flags = []

    def setsockopt(self, option, value) -> None:
        del option, value

    def connect(self, control_url: str) -> None:
        self.control_url = control_url

    def send(self, payload: bytes, *, flags: int = 0) -> None:
        del payload
        self.sent_flags.append(flags)

    def recv(self, *, flags: int = 0):
        raise AssertionError(f"recv should not be called with flags={flags}")

    def close(self, *, linger: int = 0) -> None:
        self.closed = linger == 0


class _SilentZMQContext:
    def __init__(self, socket: _SilentZMQSocket) -> None:
        self.socket_instance = socket
        self.destroy_linger = None

    def socket(self, socket_type):
        del socket_type
        return self.socket_instance

    def destroy(self, *, linger: int = 0) -> None:
        self.destroy_linger = linger


class _SilentZMQPoller:
    def __init__(self) -> None:
        self.registered = []
        self.poll_timeouts = []

    def register(self, socket, flags) -> None:
        self.registered.append((socket, flags))

    def poll(self, timeout: int):
        self.poll_timeouts.append(timeout)
        return []


def test_viewer_window_zmq_gateway_times_out_without_blocking_context_teardown(
    monkeypatch,
):
    socket = _SilentZMQSocket()
    context = _SilentZMQContext(socket)
    poller = _SilentZMQPoller()
    monkeypatch.setattr(viewer_window_service_module.zmq, "Poller", lambda: poller)
    gateway = ZMQViewerWindowGateway(context_factory=lambda: context)
    service = ViewerWindowService(gateway=gateway)

    result = service.probe_window(
        ViewerWindowStateRequest(connection=_viewer_connection(), timeout_ms=25)
    )

    assert result.reachable is False
    assert result.errors[0].code == "viewer_window_state_failed"
    assert "timed out after 25ms" in result.errors[0].message
    assert poller.poll_timeouts == [25]
    assert socket.sent_flags == [viewer_window_service_module.zmq.DONTWAIT]
    assert socket.closed is True
    assert context.destroy_linger == 0


def test_function_catalog_search_and_describe_use_registry_ids(monkeypatch):
    catalog = _catalog(monkeypatch)

    page = catalog.search(query="sample")
    detail = catalog.get("test:sample_processing_function")

    assert page.items[0].function_id == "test:sample_processing_function"
    assert detail.entry.signature == "sample_processing_function(sigma=1.0)"
    assert [parameter.name for parameter in detail.parameters] == ["image", "sigma"]
    assert [parameter.supplied_by for parameter in detail.parameters] == [
        "runtime_primary_input",
        "agent",
    ]
    assert detail.parameters[0].required is False
    assert detail.parameters[1].required is False


def test_function_catalog_complete_route_is_owned_by_registry_service(monkeypatch):
    metadata = {
        "test:sample_processing_function": _Metadata.from_function(
            sample_processing_function,
            "Apply a sample image operation.",
            [],
        ),
        "test:sample_large_signature_function": _Metadata.from_function(
            sample_large_signature_function,
            "Apply an operation with a large signature.",
            [],
        ),
    }
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: metadata,
    )

    page = FunctionCatalogService().catalog(compact_signatures=True)

    assert page.total == len(metadata)
    assert page.limit == len(metadata)
    assert {entry.function_id for entry in page.items} == set(metadata)


def test_callable_contract_owns_primary_input_parameter_identity():
    contract = CallableContract.from_callable(sample_processing_function)

    assert contract.primary_input_parameter_name == "image"


def test_function_catalog_projects_runtime_context_from_callable_contract(
    monkeypatch,
):
    metadata = _Metadata.from_function(
        sample_runtime_context_function,
        "Apply an operation using runtime execution context.",
        [],
    )
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {"test:sample_runtime_context_function": metadata},
    )

    detail = FunctionCatalogService().get("test:sample_runtime_context_function")
    parameters = {parameter.name: parameter for parameter in detail.parameters}

    assert (
        CallableContract.from_callable(
            sample_runtime_context_function
        ).runtime_context_parameter
        == "context"
    )
    assert parameters["context"].supplied_by == "runtime_parameter"
    assert parameters["context"].required is False
    assert parameters["context"].description == (
        "Supplied by OpenHCS runtime execution infrastructure; "
        "do not pass this as a function kwarg."
    )
    assert detail.entry.signature == "sample_runtime_context_function(*, scale=1.0)"


def test_function_catalog_projects_canonical_callable_artifact_specs(monkeypatch):
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:sample_artifact_contract_function": _Metadata.from_function(
                sample_artifact_contract_function,
                "Apply a sample operation with canonical artifact declarations.",
                [],
            )
        },
    )

    detail = FunctionCatalogService().get("test:sample_artifact_contract_function")

    assert detail.runtime_contract is not None
    runtime_contract = detail.runtime_contract
    assert tuple(spec.name for spec in runtime_contract.artifact_inputs) == (
        "positions",
    )
    assert tuple(spec.name for spec in runtime_contract.artifact_outputs) == (
        "objects",
    )
    assert runtime_contract.source_binding_rule is not None
    assert "canonical CallableContract artifact_inputs" in (
        runtime_contract.source_binding_rule
    )
    assert runtime_contract.materialization_rule is not None


def test_function_catalog_resolves_detail_by_callable_import_path(monkeypatch):
    catalog = _catalog(monkeypatch)
    import_path = (
        f"{sample_processing_function.__module__}."
        f"{sample_processing_function.__qualname__}"
    )

    detail = catalog.get_by_import_path(import_path)

    assert detail is not None
    assert detail.entry.function_id == "test:sample_processing_function"
    assert detail.entry.signature == "sample_processing_function(sigma=1.0)"


def test_function_catalog_metadata_reconciles_custom_sources_before_projection(
    monkeypatch,
):
    from openhcs.processing import func_registry as func_registry_module

    lifecycle_calls = []
    monkeypatch.setattr(
        func_registry_module,
        "synchronize_custom_function_sources",
        lambda **_kwargs: lifecycle_calls.append("reconciled"),
    )
    monkeypatch.setattr(
        function_catalog_module.RegistryService,
        "get_all_functions_with_metadata",
        lambda **_kwargs: {},
    )
    catalog = FunctionCatalogService()

    page = catalog.search(query="custom")

    assert page.items == ()
    assert lifecycle_calls == ["reconciled"]


def test_function_catalog_search_can_return_compact_signatures(monkeypatch):
    metadata = _Metadata.from_function(
        sample_large_signature_function,
        "Apply a sample operation with a long signature.",
        [],
    )
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {"test:sample_large_signature_function": metadata},
    )
    catalog = FunctionCatalogService()

    page = catalog.search(query="large", compact_signatures=True)
    detail = catalog.get(
        "test:sample_large_signature_function",
        compact_signature=False,
    )

    assert page.items[0].signature == (
        "sample_large_signature_function(labels, object_name, measurement_name, bins, ...)"
    )
    assert "bins=256" in detail.entry.signature
    assert "image" not in detail.entry.signature


def test_function_catalog_describe_uses_compact_signature_by_default(monkeypatch):
    metadata = _Metadata.from_function(
        sample_large_signature_function,
        "Apply a sample operation with a long signature.",
        [],
    )
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {"test:sample_large_signature_function": metadata},
    )
    catalog = FunctionCatalogService()

    detail = catalog.get("test:sample_large_signature_function")

    assert detail.entry.signature == (
        "sample_large_signature_function(labels, object_name, measurement_name, bins, ...)"
    )


def test_function_catalog_describe_bounds_large_docs(monkeypatch):
    long_doc = "A" * 100
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:sample_no_doc_function": _Metadata.from_function(
                sample_no_doc_function,
                long_doc,
                [],
            )
        },
    )
    catalog = FunctionCatalogService()

    detail = catalog.get(
        "test:sample_no_doc_function",
        max_doc_chars=12,
    )
    full_detail = catalog.get(
        "test:sample_no_doc_function",
        max_doc_chars=None,
    )

    assert detail.doc == "A" * 12
    assert detail.doc_truncated is True
    assert detail.doc_chars == 100
    assert detail.max_doc_chars == 12
    assert detail.entry.summary is not None
    assert len(detail.entry.summary) <= 180
    assert full_detail.doc == long_doc
    assert full_detail.doc_truncated is False
    assert full_detail.max_doc_chars is None


def test_function_catalog_describe_projects_cellprofiler_module_contract(monkeypatch):
    from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule

    module_type = CellProfilerModule.require_module("TrackObjects")
    nominal_calls = []
    require_module_name = module_type.require_module_name

    def resolve_module(cls, contract):
        nominal_calls.append(("for_callable_contract", contract))
        return module_type

    def resolve_module_name(cls):
        nominal_calls.append(("require_module_name", cls))
        return require_module_name()

    monkeypatch.setattr(
        CellProfilerModule,
        "for_callable_contract",
        classmethod(resolve_module),
    )
    monkeypatch.setattr(
        module_type,
        "require_module_name",
        classmethod(resolve_module_name),
    )
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:cellprofiler_track_objects": _Metadata(
                func=track_objects,
                original_name="registry_track_alias",
                name="registry_track_alias",
                doc="Track CellProfiler object labels through time.",
                tags=["cellprofiler"],
            )
        },
    )
    catalog = FunctionCatalogService()

    detail = catalog.get("test:cellprofiler_track_objects")

    assert detail.runtime_contract is not None
    assert detail.runtime_contract.callable_kind == "cellprofiler_module"
    assert detail.runtime_contract.cellprofiler_module is not None
    assert detail.runtime_contract.cellprofiler_module.module_name == "TrackObjects"
    assert nominal_calls == [
        (
            "for_callable_contract",
            CallableContract.from_callable(track_objects),
        ),
        ("require_module_name", module_type),
    ]
    assert detail.runtime_contract.pattern_compatibility_rule is not None
    assert "one CP module contract per FunctionStep" in (
        detail.runtime_contract.pattern_compatibility_rule
    )


def test_function_catalog_does_not_classify_native_name_collision_as_cellprofiler(
    monkeypatch,
):
    from openhcs.processing.backends.processors.numpy_processor import crop

    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:numpy_crop": _Metadata(
                func=crop,
                original_name="crop",
                name="crop",
                module=crop.__module__,
                doc="Crop a NumPy image stack.",
                tags=["numpy"],
            )
        },
    )

    detail = FunctionCatalogService().get("test:numpy_crop")

    assert detail.runtime_contract is not None
    assert detail.runtime_contract.callable_kind == "regular"
    assert detail.runtime_contract.cellprofiler_module is None


def test_function_catalog_search_ranks_name_matches_before_doc_matches(monkeypatch):
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
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


def test_function_catalog_search_prefers_concise_exact_owner_text(monkeypatch):
    def concise_owner(image):
        """Apply white top-hat background subtraction to each image plane."""

        return image

    def verbose_incidental_match(image):
        """Apply a generic morphology filter.

        This imported operation has a long discussion containing background
        subtraction among many other possible applications and implementation
        details that do not define its primary purpose.
        """

        return image

    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:concise_owner": _Metadata.from_function(
                concise_owner,
                concise_owner.__doc__ or "",
                [],
            ),
            "test:verbose_incidental_match": _Metadata.from_function(
                verbose_incidental_match,
                verbose_incidental_match.__doc__ or "",
                [],
            ),
        },
    )

    page = FunctionCatalogService().search(
        query="background subtraction",
        compact_signatures=True,
    )

    assert page.items[0].function_id == "test:concise_owner"


def test_function_catalog_reuses_projection_until_registry_mapping_changes(monkeypatch):
    first_metadata = {
        "test:sample_processing_function": _Metadata.from_function(
            sample_processing_function,
            sample_processing_function.__doc__ or "",
            [],
        )
    }
    second_metadata = {
        "test:sample_gaussian_filter": _Metadata.from_function(
            sample_gaussian_filter,
            sample_gaussian_filter.__doc__ or "",
            [],
        )
    }
    current_metadata = [first_metadata]
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: current_metadata[0],
    )
    catalog = FunctionCatalogService()

    first = catalog.search(query="sample", compact_signatures=True)
    cache_key = (SignatureView.COMPACT, SummaryView.COMPACT)
    first_projection = catalog._projections[cache_key]
    repeated = catalog.search(query="operation", compact_signatures=True)

    assert first.items
    assert repeated.items
    assert catalog._projections[cache_key] is first_projection

    current_metadata[0] = second_metadata
    refreshed = catalog.search(query="gaussian", compact_signatures=True)

    assert tuple(item.function_id for item in refreshed.items) == (
        "test:sample_gaussian_filter",
    )
    assert catalog._projections[cache_key] is not first_projection


def test_function_catalog_search_ranks_complete_owner_text_over_incidental_name(
    monkeypatch,
):
    def segment_unrelated_objects(image):
        """Segment unrelated objects."""

        return image

    def identify_fluorescent_nuclei(image):
        """Segment fluorescent nuclei in microscopy images."""

        return image

    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:segment_unrelated_objects": _Metadata.from_function(
                segment_unrelated_objects,
                "Segment unrelated objects.",
                [],
            ),
            "test:identify_primary_objects": _Metadata.from_function(
                identify_fluorescent_nuclei,
                "Segment fluorescent nuclei in microscopy images.",
                [],
            ),
        },
    )

    page = FunctionCatalogService().search(
        query="segment fluorescent nuclei",
        limit=1,
        compact_signatures=True,
    )

    assert page.items[0].function_id == "test:identify_primary_objects"


def test_function_catalog_search_uses_full_callable_doc_not_cached_summary(
    monkeypatch,
):
    def background_parameter_probe(image, background_radius=10):
        """Adjust a background control."""

        return image

    def background_removal_probe(image, background_radius=10):
        """Process an image.

        Use white top-hat background subtraction for small bright foreground.
        """

        return image

    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:background_parameter_probe": _Metadata(
                func=background_parameter_probe,
                original_name=background_parameter_probe.__name__,
                name=background_parameter_probe.__name__,
                doc="Adjust a background control.",
                tags=[],
            ),
            "test:background_removal_probe": _Metadata(
                func=background_removal_probe,
                original_name=background_removal_probe.__name__,
                name=background_removal_probe.__name__,
                doc="Process an image.",
                tags=[],
            ),
        },
    )

    page = FunctionCatalogService().search(
        query="white top hat background subtraction",
        compact_signatures=True,
    )

    assert page.items[0].function_id == "test:background_removal_probe"


def test_function_catalog_search_splits_cellprofiler_module_camel_case(monkeypatch):
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:cellprofiler_export_to_spreadsheet": _Metadata(
                func=sample_processing_function,
                original_name="ExportToSpreadsheet",
                name="ExportToSpreadsheet",
                doc="Render a spreadsheet file bundle.",
                tags=["cellprofiler"],
            )
        },
    )

    page = FunctionCatalogService().search(query="export to spreadsheet")

    assert tuple(item.function_id for item in page.items) == (
        "test:cellprofiler_export_to_spreadsheet",
    )


def test_function_catalog_search_handles_broad_biology_workflow_query(monkeypatch):
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:cellprofiler_identify_primary_objects": _Metadata.from_function(
                identify_primary_objects,
                "Segment primary objects such as fluorescent nuclei.",
                ["cellprofiler"],
            ),
            "test:cellprofiler_measure_object_intensity": _Metadata.from_function(
                measure_object_intensity,
                "Measure object intensity.",
                ["cellprofiler"],
            ),
            "test:cellprofiler_measure_colocalization": _Metadata.from_function(
                measure_colocalization,
                "Measure colocalization.",
                ["cellprofiler"],
            ),
        },
    )
    catalog = FunctionCatalogService()

    page = catalog.search(
        query=(
            "I have fluorescence plate images. Segment nuclei, expand to cells, "
            "measure cell intensity, and maybe colocalization."
        ),
        limit=3,
        compact_signatures=True,
    )

    function_ids = {item.function_id for item in page.items}
    assert "test:cellprofiler_identify_primary_objects" in function_ids
    assert "test:cellprofiler_measure_object_intensity" in function_ids
    assert "test:cellprofiler_measure_colocalization" in function_ids


def test_function_catalog_search_policy_has_no_domain_alias_registry():
    policy_tree = ast.parse(inspect.getsource(AgentFunctionSearchPolicy))
    policy_class = policy_tree.body[0]
    assert isinstance(policy_class, ast.ClassDef)

    dictionary_assignments = [
        node
        for node in policy_class.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and isinstance(node.value, ast.Dict)
    ]

    assert dictionary_assignments == []
    assert AgentFunctionSearchPolicy.token_variants("nuclei") == ("nuclei",)
    assert AgentFunctionSearchPolicy.token_variants("cells") == ("cells", "cell")


def test_repository_catalog_finds_nucleus_segmentation_from_callable_owner():
    page = FunctionCatalogService().search(
        query="segment fluorescent nuclei",
        limit=10,
        compact_signatures=True,
    )

    assert "openhcs:cellprofiler_identify_primary_objects" in {
        item.function_id for item in page.items
    }


def test_repository_catalog_finds_background_subtraction_from_callable_owner():
    page = FunctionCatalogService().search(
        query="background subtraction",
        limit=5,
        compact_signatures=True,
    )

    assert "openhcs:processors_numpy_processor_tophat" in {
        item.function_id for item in page.items
    }


def test_function_catalog_search_finds_tile_assembler_by_stitch_vocabulary(monkeypatch):
    from openhcs.processing.backends.assemblers.assemble_stack_cpu import (
        assemble_stack_cpu,
    )

    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            "test:assemblers_assemble_stack_cpu": _Metadata.from_function(
                assemble_stack_cpu,
                assemble_stack_cpu.__doc__ or "",
                ["assemblers", "assemble_stack_cpu"],
            )
        },
    )
    catalog = FunctionCatalogService()

    page = catalog.search(query="stitch overlapping sites", compact_signatures=True)

    assert page.items
    assert page.items[0].function_id == "test:assemblers_assemble_stack_cpu"
    assert "Stitch/assemble overlapping image tiles" in (page.items[0].summary or "")


def test_viewer_window_service_snapshots_running_viewer():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.snapshot_window(
        ViewerWindowSnapshotRequest(
            connection=_viewer_connection(),
            output_dir_path="/tmp/openhcs-mcp-window-snapshots",
            capture_scope=WindowSnapshotCaptureScope.WINDOW,
        ),
    )

    assert result.captured is True
    assert result.connection.port == 5584
    assert result.viewer is not None
    assert result.viewer.viewer_type is ViewerType.NAPARI
    assert result.viewer.title == "OpenHCS Napari Viewer"
    assert result.resource is not None
    assert result.resource.mime_type == "image/png"
    assert result.width == 640
    assert result.height == 480
    assert result.capture_scope is WindowSnapshotCaptureScope.WINDOW
    assert gateway.requests[0].output_dir_path == "/tmp/openhcs-mcp-window-snapshots"
    assert gateway.requests[0].capture_scope is WindowSnapshotCaptureScope.WINDOW


def test_viewer_window_service_reports_malformed_viewer_response():
    result = ViewerWindowService(
        gateway=_MalformedViewerWindowGateway()
    ).snapshot_window(
        ViewerWindowSnapshotRequest(
            connection=_viewer_connection(),
            output_dir_path="/tmp/openhcs-mcp-window-snapshots",
            capture_scope=WindowSnapshotCaptureScope.WIDGET,
        ),
    )

    assert result.captured is False
    assert result.errors[0].code == "viewer_window_snapshot_response_invalid"


def test_viewer_window_service_reads_running_viewer_state():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.window_state(
        ViewerWindowStateRequest(connection=_viewer_connection())
    )

    assert result.observed is True
    assert result.connection.port == 5584
    assert result.viewer is not None
    assert result.viewer.viewer_type is ViewerType.NAPARI
    assert result.layer_count == 1
    assert result.component_group_count == 1
    assert result.component_item_count == 2
    assert result.active_dimension_label_route == "IdentifyPrimaryObjects|image"
    assert result.viewer_ndim == 5
    assert result.current_step == (0, 0, 0, 0, 0)
    assert result.axis_labels == ("well", "site", "channel", "y", "x")
    assert len(result.layers) == 1
    layer = result.layers[0]
    assert layer.route_key == "IdentifyPrimaryObjects|image"
    assert len(layer.producer_identities) == 1
    assert layer.producer_identities[0].step_scope_id == "identify-primary"
    assert layer.producer_identities[0].pipeline_position == 3
    assert layer.mounted is True
    assert layer.item_count == 2
    assert layer.data_types == ("image",)
    assert layer.stack_axes == ("well", "site", "channel")
    assert layer.axis_labels == ("well", "site", "channel", "y", "x")
    assert layer.data_shape == (2, 1, 1, 16, 16)
    assert layer.translate == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert layer.axis_component_values == {
        "well": ("A14", "B13"),
        "site": (1,),
        "channel": (0,),
    }
    assert layer.routed_component_values == {
        "well": ("A14", "B13"),
        "site": (1,),
        "channel": (0,),
    }
    assert layer.component_value_count == 2
    assert layer.component_values_truncated is False
    assert layer.payload_summary_count == 2
    assert layer.payload_summaries_truncated is False
    assert layer.visible is True
    assert layer.selected is True
    assert result.response["status"] == "success"
    assert gateway.requests[0].timeout_ms == 5000


def test_viewer_window_service_can_omit_raw_state_response():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.window_state(
        ViewerWindowStateRequest(
            connection=_viewer_connection(),
            include_response=False,
        )
    )

    assert result.observed is True
    assert result.response == {}


def test_viewer_window_service_reads_compact_running_viewer_state():
    gateway = _CompactStateViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.window_state(
        ViewerWindowStateRequest(
            connection=_viewer_connection(),
            state_controls=ViewerStateControlOptions.from_overrides(
                include_component_values=False,
                include_payload_summaries=False,
            ),
        )
    )

    assert result.observed is True
    layer = result.layers[0]
    assert layer.component_values == ()
    assert layer.component_value_count == 2
    assert layer.component_values_truncated is True
    assert layer.payload_summaries == ()
    assert layer.payload_summary_count == 2
    assert layer.payload_summaries_truncated is True
    assert gateway.requests[0].state_controls.include_component_values is False
    assert gateway.requests[0].state_controls.include_payload_summaries is False


def test_viewer_window_service_reads_payload_records():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.window_payloads(
        ViewerWindowPayloadRequest(
            connection=_viewer_connection(),
            payload_controls=ViewerPayloadControlOptions.from_overrides(
                route_key="IdentifyPrimaryObjects|image",
                include_array_values=True,
                max_array_elements=16,
            ),
        ),
    )

    assert result.observed is True
    assert result.connection.port == 5584
    assert result.viewer is not None
    assert result.viewer.viewer_type is ViewerType.NAPARI
    assert result.layer_count == 1
    assert len(result.layers) == 1
    layer = result.layers[0]
    assert layer.route_key == "IdentifyPrimaryObjects|image"
    assert layer.axis_labels == ("well", "site", "channel", "y", "x")
    assert len(layer.payloads) == 1
    payload = layer.payloads[0]
    assert payload.components["well"] == "A14"
    assert payload.axis_indices == (0, 0, 0)
    assert payload.summary["nonzero_count"] == 128
    assert payload.array_values == (1, 2, 3)
    assert payload.array_value_summary == {
        "requested": True,
        "included": True,
        "shape": (3,),
    }
    assert payload.shape_payloads == (
        {
            "shape_type": "rectangle",
            "axis_indices": (0, 0, 0),
            "bounds": (2.0, 3.0, 8.0, 9.0),
        },
    )
    assert (
        gateway.requests[0].payload_controls.route_key == "IdentifyPrimaryObjects|image"
    )
    assert gateway.requests[0].payload_controls.include_array_values is True
    assert gateway.requests[0].payload_controls.max_array_elements == 16
    assert gateway.requests[0].payload_controls.include_shape_payloads is True
    assert gateway.requests[0].payload_controls.max_shape_payloads == 256


def test_viewer_window_service_can_omit_raw_payload_response():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.window_payloads(
        ViewerWindowPayloadRequest(
            connection=_viewer_connection(),
            include_response=False,
            payload_controls=ViewerPayloadControlOptions.from_overrides(
                route_key="IdentifyPrimaryObjects|image",
            ),
        )
    )

    assert result.observed is True
    assert result.response == {}


def test_viewer_window_service_navigates_running_viewer_window():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.navigate_window(
        ViewerWindowNavigationRequest(
            connection=_viewer_connection(),
            timeout_ms=25,
            navigation=ViewerNavigationControlOptions.from_overrides(
                route_key="IdentifyPrimaryObjects|image",
                axis_indices={"well": 1, "channel": 0},
                visible=True,
                selected=True,
            ),
        )
    )

    assert result.observed is True
    assert result.connection.port == 5584
    assert result.route_key == "IdentifyPrimaryObjects|image"
    assert result.visible is True
    assert result.selected is True
    assert result.active_dimension_label_route == "IdentifyPrimaryObjects|image"
    assert result.current_step == (1, 0, 0, 0, 0)
    assert result.axis_labels == ("well", "site", "channel", "y", "x")
    assert result.available_layers == (
        ViewerWindowLayerVisibilityRecord(
            route_key="IdentifyPrimaryObjects|image",
            title="IdentifyPrimaryObjects",
            visible=True,
            selected=True,
        ),
    )
    assert gateway.requests[0].timeout_ms == 25
    assert gateway.requests[0].navigation.axis_indices == {"well": 1, "channel": 0}


def test_viewer_window_service_isolates_only_mounted_layers() -> None:
    gateway = _UnmountedRouteViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.isolate_layers(
        ViewerWindowLayerIsolationRequest.from_fields(
            connection=_viewer_connection(),
            visible_route_keys=("IdentifyPrimaryObjects|image",),
            selected_route_key="IdentifyPrimaryObjects|image",
        )
    )

    navigation_requests = tuple(
        request
        for request in gateway.requests
        if isinstance(request, ViewerWindowNavigationRequest)
    )
    assert result.applied is True
    assert result.changed_route_count == 1
    assert result.layer_count == 1
    assert result.visible_route_keys == ("IdentifyPrimaryObjects|image",)
    assert result.hidden_route_keys == ()
    assert result.available_layers == (
        ViewerWindowLayerVisibilityRecord(
            route_key="IdentifyPrimaryObjects|image",
            title="IdentifyPrimaryObjects",
            visible=True,
            selected=True,
        ),
    )
    assert tuple(request.navigation.route_key for request in navigation_requests) == (
        "IdentifyPrimaryObjects|image",
    )


def test_viewer_window_service_probes_running_viewer_endpoint():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.probe_window(
        ViewerWindowStateRequest(connection=_viewer_connection(), timeout_ms=25)
    )

    assert result.reachable is True
    assert result.observed is True
    assert result.connection.port == 5584
    assert result.viewer is not None
    assert result.viewer.viewer_type is ViewerType.NAPARI
    assert result.layer_count == 1
    assert result.component_group_count == 1
    assert result.component_item_count == 2
    assert result.errors == ()
    assert gateway.requests[0].timeout_ms == 25


def test_viewer_window_service_probe_reports_unreachable_viewer_endpoint():
    result = ViewerWindowService(gateway=_MalformedViewerWindowGateway()).probe_window(
        ViewerWindowStateRequest(connection=_viewer_connection())
    )

    assert result.reachable is False
    assert result.observed is False
    assert result.errors[0].code == "viewer_window_state_response_invalid"


def test_viewer_window_service_summarizes_viewer_state_validation():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.validation_summary(
        ViewerWindowValidationRequest(
            connection=_viewer_connection(),
            validation_policy=ViewerWindowValidationPolicy(
                expected_layer_count=1,
                required_axis_labels=("well", "site", "channel"),
            ),
        )
    )

    assert result.valid is True
    assert result.observed is True
    assert result.layer_count == 1
    assert result.mounted_layer_count == 1
    assert result.pending_update_count == 0
    assert result.active_dimension_label_route == "IdentifyPrimaryObjects|image"
    assert result.active_dimension_label_route_valid is True
    assert result.payload_count == 2
    assert result.nonzero_payload_count == 2
    assert result.zero_payload_count == 0
    assert result.missing_nonzero_count == 0
    assert result.missing_payload_coordinate_count == 0
    assert result.duplicate_payload_coordinate_count == 0
    assert result.payload_without_coordinate_count == 0
    assert result.required_axis_labels == ("well", "site", "channel")
    assert result.warnings == ()
    assert result.state is None
    assert len(result.layer_summaries) == 1
    layer_summary = result.layer_summaries[0]
    assert layer_summary.valid is True
    assert layer_summary.route_key == "IdentifyPrimaryObjects|image"
    assert layer_summary.coordinate_gap_count == 0
    assert layer_summary.expected_coordinate_count == 2
    assert layer_summary.payload_coordinate_count == 2
    assert layer_summary.axis_labels == ("well", "site", "channel", "y", "x")


class _ActiveDimensionLabelRouteViewerWindowGateway(_FakeViewerWindowGateway):
    def __init__(self, active_dimension_label_route: str | None) -> None:
        super().__init__()
        self.active_dimension_label_route = active_dimension_label_route

    def window_state(self, request):
        response = super().window_state(request)
        response["active_dimension_label_route"] = self.active_dimension_label_route
        return response


def test_viewer_window_validation_accepts_absent_active_dimension_label_route():
    result = ViewerWindowService(
        gateway=_ActiveDimensionLabelRouteViewerWindowGateway(None)
    ).validation_summary(ViewerWindowValidationRequest(connection=_viewer_connection()))

    assert result.valid is True
    assert result.active_dimension_label_route is None
    assert result.active_dimension_label_route_valid is True
    assert result.warnings == ()


def test_viewer_window_validation_rejects_unobserved_active_dimension_label_route():
    result = ViewerWindowService(
        gateway=_ActiveDimensionLabelRouteViewerWindowGateway("missing-route")
    ).validation_summary(ViewerWindowValidationRequest(connection=_viewer_connection()))

    assert result.valid is False
    assert result.active_dimension_label_route == "missing-route"
    assert result.active_dimension_label_route_valid is False
    assert [warning.code for warning in result.warnings] == [
        "viewer_active_dimension_label_route_missing"
    ]


def test_viewer_window_service_validation_applies_state_controls_when_requested():
    gateway = _FakeViewerWindowGateway()
    service = ViewerWindowService(gateway=gateway)

    result = service.validation_summary(
        ViewerWindowValidationRequest(
            connection=_viewer_connection(),
            state_controls=ViewerStateControlOptions.from_overrides(
                route_key="IdentifyPrimaryObjects|image",
                include_component_values=False,
                include_payload_summaries=True,
            ),
            include_state=True,
        )
    )

    assert result.valid is True
    assert result.state is not None
    assert (
        gateway.requests[0].state_controls.route_key == "IdentifyPrimaryObjects|image"
    )
    assert gateway.requests[0].state_controls.include_component_values is False
    assert gateway.requests[0].state_controls.include_payload_summaries is True


def test_viewer_window_service_validation_accepts_required_collapsed_components():
    result = ViewerWindowService(
        gateway=_CollapsedComponentViewerWindowGateway()
    ).validation_summary(
        ViewerWindowValidationRequest(
            connection=_viewer_connection(),
            validation_policy=ViewerWindowValidationPolicy(
                required_axis_labels=("channel", "y", "x"),
                required_component_labels=("well", "site"),
            ),
        )
    )

    assert result.valid is True
    assert result.required_component_labels == ("well", "site")
    assert result.warnings == ()
    layer_summary = result.layer_summaries[0]
    assert layer_summary.axis_labels == ("channel", "y", "x")
    assert layer_summary.component_labels == ("channel", "site", "well")
    assert layer_summary.missing_required_component_labels == ()


def test_viewer_window_service_validation_explains_collapsed_required_axes():
    result = ViewerWindowService(
        gateway=_CollapsedComponentViewerWindowGateway()
    ).validation_summary(
        ViewerWindowValidationRequest(
            connection=_viewer_connection(),
            validation_policy=ViewerWindowValidationPolicy(
                required_axis_labels=("well", "site", "channel"),
            ),
        )
    )

    assert result.valid is False
    layer_summary = result.layer_summaries[0]
    assert layer_summary.missing_required_axis_labels == ("well", "site")
    assert layer_summary.axis_labels_present_as_components == ("well", "site")
    assert result.warnings[0].code == "viewer_required_axis_labels_missing"
    assert "present as component metadata" in str(result.warnings[0].hint)


def test_viewer_window_service_validation_reports_axis_and_count_mismatch():
    result = ViewerWindowService(gateway=_FakeViewerWindowGateway()).validation_summary(
        ViewerWindowValidationRequest(
            connection=_viewer_connection(),
            validation_policy=ViewerWindowValidationPolicy(
                expected_layer_count=2,
                required_axis_labels=("source",),
            ),
        )
    )

    assert result.valid is False
    assert result.layer_count == 1
    assert result.expected_layer_count == 2
    assert result.layer_summaries[0].missing_required_axis_labels == ("source",)
    assert result.layer_summaries[0].valid is False
    assert [warning.code for warning in result.warnings] == [
        "viewer_layer_count_mismatch",
        "viewer_required_axis_labels_missing",
    ]


def test_viewer_window_service_validation_reports_coordinate_gaps():
    result = ViewerWindowService(
        gateway=_CoordinateGapViewerWindowGateway()
    ).validation_summary(ViewerWindowValidationRequest(connection=_viewer_connection()))

    assert result.valid is False
    assert result.layer_summaries[0].coordinate_gap_count == 2
    assert result.layer_summaries[0].missing_payload_coordinate_count == 2
    assert [warning.code for warning in result.warnings] == [
        "viewer_layer_coordinate_gaps",
        "viewer_payload_coordinates_missing",
    ]


def test_viewer_window_service_accepts_unrouted_axis_positions():
    result = ViewerWindowService(
        gateway=_UnroutedAxisPositionViewerWindowGateway()
    ).validation_summary(ViewerWindowValidationRequest(connection=_viewer_connection()))

    assert result.valid is True
    assert result.layer_summaries[0].valid is True
    assert result.layer_summaries[0].coordinate_gap_count == 4
    assert result.layer_summaries[0].missing_payload_coordinate_count == 0
    assert [warning.code for warning in result.warnings] == [
        "viewer_layer_coordinate_gaps",
    ]


def test_viewer_window_service_validation_accepts_channel_last_rgb_spatial_shape():
    result = ViewerWindowService(gateway=_RgbViewerWindowGateway()).validation_summary(
        ViewerWindowValidationRequest(connection=_viewer_connection())
    )

    assert result.valid is True


def test_viewer_window_service_validation_accepts_padded_variable_size_payloads():
    result = ViewerWindowService(
        gateway=_PaddedVariableSizeViewerWindowGateway()
    ).validation_summary(ViewerWindowValidationRequest(connection=_viewer_connection()))

    assert result.valid is True


def test_viewer_window_service_validation_expands_aggregate_payload_coordinates():
    result = ViewerWindowService(
        gateway=_AggregateStackViewerWindowGateway()
    ).validation_summary(ViewerWindowValidationRequest(connection=_viewer_connection()))

    assert result.valid is True
    assert result.layer_summaries[0].expected_coordinate_count == 3
    assert result.layer_summaries[0].payload_coordinate_count == 3
    assert result.layer_summaries[0].payload_without_coordinate_count == 0


def test_viewer_window_service_reports_malformed_state_response():
    result = ViewerWindowService(gateway=_MalformedViewerWindowGateway()).window_state(
        ViewerWindowStateRequest(connection=_viewer_connection())
    )

    assert result.observed is False
    assert result.errors[0].code == "viewer_window_state_response_invalid"


def test_config_service_reflects_pipeline_schema_without_materializing_lazy_values():
    service = ConfigService()

    schema = service.describe_schema("pipeline")
    shared_descriptions = dataclass_parameter_descriptions(PipelineConfig)
    well_filter = next(
        field for field in schema.fields if field.path == "well_filter_config"
    )
    source_bindings = next(
        field for field in schema.fields if field.path == "source_bindings_config"
    )
    path_planning = next(
        field for field in schema.fields if field.path == "path_planning_config"
    )

    assert schema.config_type == "PipelineConfig"
    assert well_filter.lazy is True
    assert well_filter.default_repr.endswith("LazyWellFilterConfig()")
    assert source_bindings.description == shared_descriptions["source_bindings_config"]
    assert path_planning.description == shared_descriptions["path_planning_config"]
    assert path_planning.description is not None
    assert path_planning.description.startswith(
        "Configuration for pipeline path planning"
    )
    assert "PathPlanningConfig(" not in path_planning.description
    assert len(schema.fields) == len(fields(PipelineConfig))
    nested_schema = service.describe_schema("pipeline", "processing_config")
    source_bindings_schema = service.describe_schema(
        "pipeline", "source_bindings_config"
    )
    napari_schema = service.describe_schema("pipeline", "napari_streaming_config")
    nested_paths = {
        field.path: field
        for selected_schema in (nested_schema, source_bindings_schema, napari_schema)
        for field in selected_schema.fields
    }
    assert nested_paths["processing_config.variable_components"].lazy is False
    assert nested_paths["processing_config.variable_components"].inheritable is True
    assert nested_paths["processing_config.group_by"].enum_values
    assert "assembled array axis" in (
        nested_paths["processing_config.variable_components"].description or ""
    )
    assert "partition already assembled values" in (
        nested_paths["processing_config.group_by"].description or ""
    )
    input_source_description = " ".join(
        (nested_paths["processing_config.input_source"].description or "").split()
    )
    assert "not another ``InputSource`` value" in input_source_description
    assert "source_bindings_config.bindings[].alias" in nested_paths
    assert "source_bindings_config.metadata_rules[].pattern" in nested_paths
    assert "napari_streaming_config.site_mode" in nested_paths
    assert nested_schema.path_prefix == "processing_config"

    with pytest.raises(ValueError, match="Unknown config schema path_prefix"):
        service.describe_schema("pipeline", "not_a_config")


def test_config_service_validates_and_renders_config_source():
    service = ConfigService()

    result = service.validate_patch(
        "global",
        ConfigPatch(config_type="GlobalPipelineConfig", values={"num_workers": 2}),
    )
    rendered = service.render_source(result.config_ref)
    config = service.resolve_ref(result.config_ref)

    assert result.valid is True
    assert "num_workers=2" in rendered.source
    assert (
        ConfigDocumentAuthority.from_source(
            rendered.source,
            expected_config_type=type(config),
        )
        == config
    )


def test_config_service_coerces_nested_pipeline_config_patch_values():
    service = ConfigService()

    config_ref = service.create(
        "pipeline",
        ConfigPatch(
            config_type="PipelineConfig",
            values={
                "well_filter_config": {"well_filter": 2},
                "path_planning_config": {
                    "output_dir_suffix": "_codex_mcp_contract_validation"
                },
                "vfs_config": {"read_backend": "disk"},
            },
        ),
    )

    config = service.resolve_ref(config_ref)
    rendered = service.render_source(config_ref)
    pipeline_field_types = {field.name: field.type for field in fields(PipelineConfig)}

    assert isinstance(
        config.well_filter_config,
        pipeline_field_types["well_filter_config"],
    )
    assert config.well_filter_config.well_filter == 2
    assert isinstance(
        config.path_planning_config,
        pipeline_field_types["path_planning_config"],
    )
    assert (
        config.path_planning_config.output_dir_suffix
        == "_codex_mcp_contract_validation"
    )
    assert isinstance(config.vfs_config, pipeline_field_types["vfs_config"])
    assert config.vfs_config.read_backend is Backend.DISK
    assert "well_filter_config=LazyWellFilterConfig" in rendered.source
    assert "path_planning_config=LazyPathPlanningConfig" in rendered.source


def test_config_service_coerces_generic_source_binding_patch_values():
    service = ConfigService()

    config_ref = service.create(
        "pipeline",
        ConfigPatch(
            config_type="PipelineConfig",
            values={
                "source_bindings_config": {
                    "source_filters": [
                        {
                            "subject": "extension",
                            "match_type": "is_image",
                        }
                    ],
                    "metadata_rules": [
                        {
                            "source": "file_name",
                            "pattern": (
                                r"^(?P<Well>[A-H][0-9]{2})_"
                                r"s(?P<Site>[0-9]+)_(?P<Stain>[^.]+)"
                            ),
                        }
                    ],
                    "bindings": [
                        {
                            "alias": "DNA",
                            "selector": {
                                "filters": [
                                    {
                                        "subject": "file",
                                        "match_type": "contains",
                                        "value": "DNA",
                                    }
                                ]
                            },
                            "component_identity": [
                                {"component": "channel", "value": "1"}
                            ],
                        }
                    ],
                }
            },
        ),
    )

    config = service.resolve_ref(config_ref)
    source_bindings = config.source_bindings_config
    rendered = service.render_source(config_ref)

    assert isinstance(source_bindings, LazySourceBindingsConfig)
    assert isinstance(source_bindings.source_filters[0], SourceFilterClause)
    assert isinstance(source_bindings.metadata_rules[0], MetadataExtractionRule)
    assert source_bindings.metadata_rules[0].capture_fields == (
        "Well",
        "Site",
        "Stain",
    )
    assert isinstance(source_bindings.bindings[0], NamedSourceBinding)
    assert isinstance(source_bindings.bindings[0].selector, SourceSelector)
    assert isinstance(
        source_bindings.bindings[0].selector.filters[0],
        SourceFilterClause,
    )
    assert source_bindings.bindings[0].component_identity[0].component.value == (
        "channel"
    )
    assert "source_bindings_config=LazySourceBindingsConfig" in rendered.source
    assert "NamedSourceBinding(" in rendered.source


def test_pipeline_authoring_service_renders_function_step_source(monkeypatch):
    config_service = ConfigService()
    pipeline_service = PipelineAuthoringService(
        _catalog(monkeypatch),
        config_service,
    )
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(
        function_id="test:sample_processing_function",
        kwargs={"sigma": 2.0},
    )

    pipeline_service.add_step(pipeline_ref, step)
    validation = pipeline_service.validate(pipeline_ref)
    pipeline_spec = pipeline_service.get_pipeline(pipeline_ref)
    document = pipeline_service.to_pipeline_document(pipeline_ref)
    rendered = pipeline_service.render_source(pipeline_ref)

    assert tuple(field.name for field in fields(type(step.functions[0]))) == (
        "function_id",
        "kwargs",
    )
    assert validation.valid is True
    assert pipeline_spec.pipeline_config_id
    assert document.pipeline_config is config_service.resolve_ref(
        pipeline_spec.pipeline_config_id
    )
    assert document.pipeline_steps
    assert "pipeline_config = PipelineConfig(" in rendered.source
    assert "pipeline_steps = [" in rendered.source
    assert "FunctionStep" in rendered.source
    assert "sample_processing_function" in rendered.source
    assert "sigma" in rendered.source


def test_pipeline_authoring_service_uses_referenced_pipeline_config(monkeypatch):
    config_service = ConfigService()
    config_ref = config_service.create(
        "pipeline",
        ConfigPatch(
            config_type="PipelineConfig",
            values={"well_filter_config": {"well_filter": 2}},
        ),
    )
    pipeline_service = PipelineAuthoringService(
        _catalog(monkeypatch),
        config_service,
    )

    pipeline_ref = pipeline_service.create_pipeline_from_request(
        CreatePipelineRequest(pipeline_config_id=config_ref.config_id)
    )
    pipeline_spec = pipeline_service.get_pipeline(pipeline_ref)
    document = pipeline_service.to_pipeline_document(pipeline_ref)
    rendered = pipeline_service.render_source(pipeline_ref)

    assert pipeline_spec.pipeline_config_id == config_ref.config_id
    assert document.pipeline_config is config_service.resolve_ref(config_ref)
    assert document.pipeline_config.well_filter_config.well_filter == 2
    assert "well_filter=2" in rendered.source


def test_pipeline_authoring_service_rejects_non_pipeline_config_reference(monkeypatch):
    config_service = ConfigService()
    global_config_ref = config_service.create("global")
    pipeline_service = PipelineAuthoringService(
        _catalog(monkeypatch),
        config_service,
    )

    with pytest.raises(TypeError, match="must resolve to PipelineConfig"):
        pipeline_service.create_pipeline_from_request(
            CreatePipelineRequest(
                pipeline_config_id=global_config_ref.config_id,
            )
        )


def test_pipeline_authoring_service_rejects_unknown_function_kwargs(monkeypatch):
    pipeline_service = PipelineAuthoringService(_catalog(monkeypatch))
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(
        function_id="test:sample_processing_function",
        kwargs={"not_a_parameter": 1},
    )

    pipeline_service.add_step(pipeline_ref, step)
    validation = pipeline_service.validate(pipeline_ref)

    assert validation.valid is False
    assert validation.errors[0].code == "invalid_function_kwargs"
    assert "not_a_parameter" in validation.errors[0].message
    assert "sigma" in validation.errors[0].hint
    with pytest.raises(InvalidFunctionKwargsError, match="not_a_parameter"):
        pipeline_service.render_source(pipeline_ref)


def test_pipeline_authoring_service_rejects_runtime_input_kwargs(monkeypatch):
    pipeline_service = PipelineAuthoringService(_catalog(monkeypatch))
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(
        function_id="test:sample_processing_function",
        kwargs={"image": "agent supplied image"},
    )

    pipeline_service.add_step(pipeline_ref, step)
    validation = pipeline_service.validate(pipeline_ref)

    assert validation.valid is False
    assert validation.errors[0].code == "invalid_function_kwargs"
    assert "image" in validation.errors[0].message
    assert "sigma" in validation.errors[0].hint
    assert "image" not in validation.errors[0].hint


def test_pipeline_authoring_service_rejects_missing_required_agent_kwargs(monkeypatch):
    function_id = "test:sample_required_parameter_function"
    monkeypatch.setattr(
        FunctionCatalogService,
        "_all_metadata",
        lambda self, **_kwargs: {
            function_id: _Metadata.from_function(
                sample_required_parameter_function,
                "Apply a sample operation with a required agent parameter.",
            )
        },
    )
    pipeline_service = PipelineAuthoringService(FunctionCatalogService())
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(function_id=function_id)

    pipeline_service.add_step(pipeline_ref, step)
    validation = pipeline_service.validate(pipeline_ref)

    assert validation.valid is False
    assert validation.errors[0].code == "missing_function_kwargs"
    assert "threshold" in validation.errors[0].message
    assert "openhcs_describe_function" in validation.errors[0].hint
    with pytest.raises(MissingFunctionKwargsError, match="threshold"):
        pipeline_service.render_source(pipeline_ref)

    valid_pipeline_ref = pipeline_service.create_pipeline()
    valid_step = pipeline_service.make_step_spec(
        function_id=function_id,
        kwargs={"threshold": 0.25},
    )
    pipeline_service.add_step(valid_pipeline_ref, valid_step)

    assert pipeline_service.validate(valid_pipeline_ref).valid is True


def test_pipeline_authoring_service_derives_step_config_overrides(monkeypatch):
    pipeline_service = PipelineAuthoringService(_catalog(monkeypatch))
    pipeline_ref = pipeline_service.create_pipeline()
    step = pipeline_service.make_step_spec(
        function_id="test:sample_processing_function",
        step_config_overrides={
            "processing_config": {
                "variable_components": ["site"],
                "group_by": "channel",
            },
        },
    )

    pipeline_service.add_step(pipeline_ref, step)
    validation = pipeline_service.validate(pipeline_ref)
    rendered = pipeline_service.render_source(pipeline_ref)

    processing_patch = step.step_config_overrides["processing_config"]
    assert processing_patch.config_type == "LazyProcessingConfig"
    assert validation.valid is True
    assert "processing_config=LazyProcessingConfig(" in rendered.source
    assert "VariableComponents.SITE" in rendered.source
    assert "group_by=GroupBy.CHANNEL" in rendered.source


def test_pipeline_authoring_service_warns_for_empty_pipeline(monkeypatch):
    pipeline_service = PipelineAuthoringService(_catalog(monkeypatch))
    pipeline_ref = pipeline_service.create_pipeline()

    validation = pipeline_service.validate(pipeline_ref)

    assert validation.valid is True
    assert validation.warnings[0].code == "pipeline_empty"
    assert "openhcs_add_function_step" in validation.warnings[0].hint


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
    client_factory = _FakeExecutionClientFactory(fake_client)
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=pipeline_service,
        config_service=ConfigService(),
        client_factory=client_factory,
    )

    session_ref = execution_service.create_session(
        plate_path=str(tmp_path),
        pipeline_id=pipeline_ref.pipeline_id,
    )
    session = execution_service.get_session(session_ref.session_id)
    compile_ref = execution_service.submit_compile(session_ref.session_id)
    execute_ref = execution_service.submit_execution(
        session_ref.session_id,
        compile_artifact_id=_ExecutionTestId.COMPILE,
    )
    compile_status = execution_service.get_job_status(compile_ref.job_id)

    assert compile_ref.server_execution_id == _ExecutionTestId.COMPILE
    assert execute_ref.server_execution_id == _ExecutionTestId.EXECUTE
    assert compile_status.status == "complete"
    assert fake_client.submit_timeout_requests == [
        ("compile", OPENHCS_ZMQ_CONFIG.control_timeout_ms),
        ("execute", OPENHCS_ZMQ_CONFIG.control_timeout_ms),
    ]
    assert fake_client.compile_submissions[0].plate_id == str(tmp_path.resolve())
    assert (
        session.pipeline_config_id
        == pipeline_service.get_pipeline(pipeline_ref).pipeline_config_id
    )
    assert (
        fake_client.compile_submissions[0].pipeline_document.pipeline_config
        == pipeline_service.to_pipeline_document(pipeline_ref).pipeline_config
    )
    assert fake_client.status_requests[0] == (
        _ExecutionTestId.COMPILE,
        OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    )
    assert client_factory.creation_count == 2
    assert fake_client.disconnect_count == 1
    repeated_status = execution_service.get_job_status(compile_ref.job_id)
    assert repeated_status.status == "complete"
    assert len(fake_client.status_requests) == 1
    assert fake_client.disconnect_count == 1
    assert (
        fake_client.execution_submissions[0].compile_artifact_id
        == _ExecutionTestId.COMPILE
    )
    assert type(fake_client.compile_submissions[0].pipeline_steps) is list
    assert len(fake_client.compile_submissions[0].pipeline_steps) == 1
    assert not hasattr(fake_client.compile_submissions[0], "submission_pipeline")


def test_execution_session_service_preserves_terminal_result_when_release_fails(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _DisconnectFailureExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    compile_ref = execution_service.submit_compile(session_ref.session_id)

    status = execution_service.get_job_status(compile_ref.job_id)

    assert status.status == "complete"
    assert fake_client.disconnect_count == 1


def test_execution_session_service_preserves_pipeline_source_document(
    monkeypatch,
    tmp_path: Path,
):
    pipeline_config = PipelineConfig(num_workers=7)
    pipeline_source = _pipeline_document_source(pipeline_config)
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=pipeline_source,
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    execution_service.submit_compile(session_ref.session_id)

    submission = fake_client.compile_submissions[0]
    assert submission.pipeline_document.original_source == pipeline_source
    assert submission.pipeline_code() == pipeline_source
    assert submission.pipeline_steps == []
    assert submission.pipeline_config == pipeline_config
    assert submission.pipeline_code() == pipeline_source
    assert not hasattr(submission, "pipeline_steps_boundary")


def test_execution_session_service_inspects_pipeline_source_artifact_plan(
    monkeypatch,
    tmp_path: Path,
):
    compile_gateway = _FakeCompileInspectionGateway()
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=compile_gateway,
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        ),
        axis_filter=("A01",),
    )

    assert compile_gateway.requests[0].plate == tmp_path.resolve()
    assert compile_gateway.requests[0].pipeline_document.pipeline_steps == []
    assert (
        compile_gateway.requests[0].pipeline_document.pipeline_config
        == PipelineConfig()
    )
    assert compile_gateway.requests[0].axis_filter == ("A01",)
    assert compile_gateway.requests[0].progress_queue.events == [
        {"phase": "compile", "status": "running"}
    ]
    assert inspection.errors == ()
    assert inspection.axis_count == 1
    assert inspection.axes == ("A01",)
    assert inspection.worker_assignments == {"worker-1": ["A01"]}
    assert inspection.progress_event_count == 1
    assert inspection.steps[0].step_name == "WriteArtifacts"
    assert inspection.steps[0].execution_groups == ("A01",)
    assert inspection.steps[0].main_flow_axis_persistence_enabled is False
    assert inspection.steps[0].main_flow_materialization is not None
    assert inspection.steps[0].main_flow_materialization.output_dir == (
        "/tmp/out/A01/checkpoints"
    )
    assert inspection.steps[0].main_flow_materialization.sub_dir == "checkpoints"
    assert inspection.steps[0].viewer_streaming[0].config_key == (
        "napari_streaming_config"
    )
    assert inspection.steps[0].viewer_streaming[0].viewer_type is ViewerType.NAPARI
    assert inspection.steps[0].viewer_streaming[0].effective_config["enabled"] is True
    assert inspection.steps[0].viewer_streaming[0].effective_config["well_filter"] == [
        "A01"
    ]
    assert inspection.steps[0].artifact_inputs[0].name == "positions"
    assert inspection.steps[0].artifact_inputs[0].kind == "special"
    assert inspection.steps[0].artifact_inputs[0].source_step_id == 0
    assert inspection.steps[0].artifact_inputs[0].source_step_scope_id == (
        "step-find-positions"
    )
    assert inspection.steps[0].artifact_inputs[0].paths_by_group == (
        {"group_key": "A01", "path": "/tmp/out/A01/positions.pkl"},
    )
    assert inspection.steps[0].artifact_outputs[0].name == "objects"
    assert inspection.steps[0].artifact_outputs[0].kind == "object_labels"
    assert inspection.steps[0].artifact_outputs[0].paths_by_group == (
        {"group_key": "A01", "path": "/tmp/out/A01/objects.zarr"},
    )
    assert inspection.source_workspace.file_count == 1
    assert inspection.source_workspace.axis_file_counts == {"A01": 1}
    assert inspection.source_workspace.files[0].virtual_path == (
        "A01_s001_w1_z001_t001.tif"
    )
    assert inspection.source_workspace.files[0].full_virtual_path == str(
        tmp_path.resolve() / "A01_s001_w1_z001_t001.tif"
    )
    assert inspection.source_workspace.files[0].source_path == str(
        tmp_path.resolve() / "source.tif"
    )
    assert inspection.source_workspace.files[0].source_metadata == {
        "well": "A01",
        "channel": "1",
    }


def test_compile_inspection_counts_source_workspace_for_empty_pipeline_axis_filter(
    tmp_path: Path,
):
    virtual_name = "A01_s001_w1_z001_t001.tif"
    other_virtual_name = "A02_s001_w1_z001_t001.tif"
    inspection = artifact_plan_inspection_from_compilation(
        plate_path=str(tmp_path),
        axis_filter=("A01",),
        compilation={
            "execution_bundle": SimpleNamespace(
                runtime_contexts={},
                worker_assignments={},
            ),
            "source_workspace_projection": VirtualWorkspaceSourceProjection(
                source_refs_by_virtual_path={
                    virtual_name: SourcePixelRef(
                        "disk", str(tmp_path / "source-a01.tif")
                    ),
                    other_virtual_name: SourcePixelRef(
                        "disk", str(tmp_path / "source-a02.tif")
                    ),
                },
                source_metadata_by_path={},
                workspace_root=str(tmp_path),
            ),
        },
        progress_event_count=0,
    )

    assert inspection.axis_count == 0
    assert inspection.axes == ()
    assert inspection.source_workspace.file_count == 2
    assert inspection.source_workspace.axis_file_counts == {"A01": 1}
    assert inspection.source_workspace.files[0].source_metadata == {
        "well": "A01",
        "site": 1,
        "channel": 1,
        "z_index": 1,
        "timepoint": 1,
        "extension": ".tif",
    }


def test_execution_session_service_warns_when_compile_inspection_initializes_workspace(
    monkeypatch,
    tmp_path: Path,
):
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=_WorkspacePreparingCompileInspectionGateway(),
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        ),
    )

    assert inspection.errors == ()
    assert inspection.warnings[0].code == "compile_inspection_initialized_workspace"
    assert str(tmp_path / "openhcs_metadata.json") in inspection.warnings[0].message
    assert "openhcs_inspect_plate_path" in inspection.warnings[0].hint


def test_execution_session_service_projects_invalid_pipeline_document_errors(
    monkeypatch,
    tmp_path: Path,
):
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=_FakeCompileInspectionGateway(),
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source="x = 1\n",
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        ),
    )

    assert inspection.errors[0].code == "pipeline_source_invalid_document"
    assert "openhcs_render_pipeline_source" in inspection.errors[0].hint


def test_execution_session_service_projects_missing_artifact_input_guidance(
    monkeypatch,
    tmp_path: Path,
):
    from openhcs.core.pipeline.path_planner import MissingArtifactInputError

    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=_FailingCompileInspectionGateway(
            MissingArtifactInputError(
                step_id=0,
                artifact_key="positions",
                step_name="Stitch overlapping sites",
            )
        ),
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        ),
    )

    error = inspection.errors[0]
    assert error.code == "compile_inspection_missing_artifact_input"
    assert error.exception_type == "MissingArtifactInputError"
    assert "positions" in error.message
    assert "source bindings" in error.hint
    assert "openhcs_architecture_quick_start#compile-before-execution" in error.hint


def test_execution_session_service_projects_pixel_size_compile_inspection_error(
    monkeypatch,
    tmp_path: Path,
):
    image_path = tmp_path / "A14_s001_w1_z001_t001.tif"
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=_FailingCompileInspectionGateway(
            MicroscopePixelSizeUnavailableError(image_path)
        ),
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        ),
    )

    error = inspection.errors[0]
    assert error.code == "compile_inspection_pixel_size_unavailable"
    assert error.exception_type == "MicroscopePixelSizeUnavailableError"
    assert error.path == str(image_path)
    assert "openhcs_inspect_plate_path" in error.hint
    assert "physical pixel size" in error.hint
    assert inspection.progress_event_count == 1


def test_compile_inspection_syntax_error_preflight_avoids_compile_gateway(
    monkeypatch,
    tmp_path: Path,
):
    compile_gateway = _FakeCompileInspectionGateway()
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=compile_gateway,
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source="not valid python !!!",
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )

    assert inspection.errors[0].code == "pipeline_source_syntax_error"
    assert compile_gateway.requests == []


def test_compile_inspection_requires_direct_pipeline_step_list(
    monkeypatch,
    tmp_path: Path,
) -> None:
    compile_gateway = _FakeCompileInspectionGateway()
    execution_service = ExecutionSessionService(
        path_policy=AgentPathPolicy.with_roots(
            readable_roots=(tmp_path,),
            writable_roots=(tmp_path,),
        ),
        pipeline_service=PipelineAuthoringService(_catalog(monkeypatch)),
        config_service=ConfigService(),
        client_factory=_FakeExecutionClientFactory(_FakeExecutionClient()),
        compile_inspection_gateway=compile_gateway,
    )

    inspection = execution_service.inspect_pipeline_source_artifact_plan(
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=(
                "from openhcs.core.config import PipelineConfig\n"
                "pipeline_config = PipelineConfig()\n"
                "pipeline_steps = ()\n"
            ),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )

    assert inspection.errors[0].code == "pipeline_source_invalid_document"
    assert "list[FunctionStep]" in inspection.errors[0].message
    assert compile_gateway.requests == []


def test_pipeline_source_session_rejects_execution_plate_path(tmp_path: Path):
    with pytest.raises(ValueError, match="execution_plate_id must be None"):
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(
                plate_id=str(tmp_path),
                execution_plate_id=str(tmp_path),
            ),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )


def test_execution_session_service_reports_nested_runtime_status(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _EnvelopeStatusExecutionClient()
    fake_client.progress_by_execution_id[_ExecutionTestId.COMPILE] = (
        ExecutionProgressObservation.first(
            {
                "execution_id": _ExecutionTestId.COMPILE,
                "phase": "compile",
                "status": "running",
                "percent": 25.0,
            }
        )
    )
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    compile_ref = execution_service.submit_compile(session_ref.session_id)
    status = execution_service.get_job_status(compile_ref.job_id)

    assert status.status == "running"
    assert status.response["status"] == "ok"
    assert status.progress is not None
    assert status.progress.sequence == 1
    assert status.progress.event["phase"] == "compile"
    serialized_status = to_jsonable(status)
    assert serialized_status["progress"]["event"]["phase"] == "compile"
    assert fake_client.status_requests[0] == (
        _ExecutionTestId.COMPILE,
        OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    )


def test_execution_session_service_warns_when_headless_run_skips_plate_manager(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _HeadlessCompleteExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    execute_ref = execution_service.submit_execution(session_ref.session_id)
    status = execution_service.get_job_status(execute_ref.job_id)

    assert status.status == "complete"
    assert status.warnings[0].code == "headless_execution_did_not_update_plate_manager"
    assert "plate_manager.orchestrator_config" in status.warnings[0].hint
    assert "openhcs_ui_selected_plate_workflow" in status.warnings[0].hint


def test_execution_session_service_wait_true_is_bounded_polling(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _EnvelopeStatusExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    status = execution_service.submit_compile(
        session_ref.session_id,
        wait=True,
        wait_timeout_ms=1,
    )

    assert status.status == "running"
    assert status.warnings[0].code == "execution_wait_timeout"
    assert status.response["wait_timed_out"] is True
    assert status.response["wait_timeout_ms"] == 1
    assert fake_client.status_requests[0][0] == _ExecutionTestId.COMPILE
    assert fake_client.status_requests[0][1] <= OPENHCS_ZMQ_CONFIG.control_timeout_ms
    assert fake_client.wait_requests == []


def test_execution_session_service_wait_true_converts_status_timeout_to_warning(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _TimeoutStatusExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    status = execution_service.submit_compile(
        session_ref.session_id,
        wait=True,
        wait_timeout_ms=1,
    )

    assert status.status == "running"
    assert status.errors == ()
    assert status.warnings[0].code == "execution_wait_timeout"
    assert status.response["wait_timed_out"] is True
    assert status.response["last_status_error"]["exception_type"] == "TimeoutError"
    assert fake_client.status_requests[0][0] == _ExecutionTestId.COMPILE
    assert fake_client.wait_requests == []


def test_execution_session_service_submit_timeout_returns_agent_error(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _TimeoutSubmitExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    status = execution_service.submit_compile(
        session_ref.session_id,
        submit_timeout_ms=7,
    )

    assert status.status == "submit_error"
    assert status.errors[0].code == "execution_submit_timeout"
    assert "unknown" in status.errors[0].hint
    assert status.response["submit_timeout_ms"] == 7
    assert fake_client.submit_timeout_requests == [("compile", 7)]


def test_execution_session_service_reports_known_preparation_timeout(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _PreparationTimeoutSubmitExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )

    status = execution_service.submit_compile(
        session_ref.session_id,
        submit_timeout_ms=7,
    )

    assert status.status == "submit_error"
    assert status.errors[0].code == "execution_submit_timeout"
    assert "no execution request was sent" in status.errors[0].hint
    assert "outcome is unknown" not in status.errors[0].hint


def test_execution_session_service_does_not_abandon_live_submit_operation(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _SlowSubmitExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    started = time.monotonic()
    ref = execution_service.submit_compile(
        session_ref.session_id,
        submit_timeout_ms=5,
    )
    elapsed = time.monotonic() - started

    assert elapsed >= 0.05
    assert ref.status == "accepted"
    assert ref.server_execution_id == _ExecutionTestId.COMPILE
    assert fake_client.submit_timeout_requests == [("compile", 5)]
    assert len(fake_client.compile_submissions) == 1


def test_execution_session_service_bounds_failed_execution_status(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _FailedStatusExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    execute_ref = execution_service.submit_execution(session_ref.session_id)
    status = execution_service.get_job_status(execute_ref.job_id)

    execution_payload = status.response["execution"]

    assert status.status == "failed"
    assert status.errors[0].code == "execution_failed"
    assert status.errors[0].message == "synthetic failure"
    assert len(execution_payload["traceback"]) == MAX_EXECUTION_STATUS_TRACEBACK_CHARS
    assert execution_payload["traceback_truncated"] is True
    assert (
        execution_payload["traceback_original_chars"]
        == MAX_EXECUTION_STATUS_TRACEBACK_CHARS + 200
    )
    assert fake_client.status_requests[0] == (
        _ExecutionTestId.EXECUTE,
        OPENHCS_ZMQ_CONFIG.control_timeout_ms,
    )


def test_execution_session_service_hints_custom_function_import_failures(
    monkeypatch,
    tmp_path: Path,
):
    fake_client = _CustomFunctionImportFailedStatusExecutionClient()
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
        PipelineSourceSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source=_pipeline_document_source(),
            global_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    execute_ref = execution_service.submit_execution(session_ref.session_id)
    status = execution_service.get_job_status(execute_ref.job_id)

    assert status.status == "failed"
    assert status.errors[0].code == "execution_failed"
    assert status.errors[0].message == (
        "cannot import name 'agent_threshold_mask' from "
        "'openhcs.processing.custom_functions'"
    )
    assert "custom function" in status.errors[0].hint
    assert "XDG_DATA_HOME" in status.errors[0].hint
    assert "custom_functions directory" in status.errors[0].hint
    assert "restart or reload the runtime server" in status.errors[0].hint


def test_runtime_server_service_reads_runtime_server_state():
    gateway = _FakeRuntimeServerGateway()
    service = RuntimeServerService(gateway=gateway)

    server_info = service.server_info(port=5555, timeout_ms=25)
    scan_result = service.scan(ports=(5555, 5555, 7777), timeout_ms=25)
    execution_status = service.execution_status(
        execution_id=_ExecutionTestId.EXECUTE,
        port=5555,
    )

    assert server_info.reachable is True
    assert server_info.server == "OpenHCSExecutionServer"
    assert server_info.running_executions[0].execution_id == _ExecutionTestId.EXECUTE
    assert gateway.server_info_connections[0][1] == 25
    assert scan_result.ports == (5555, 7777)
    assert [server.port for server in scan_result.servers] == [5555, 7777]
    assert execution_status.status == "complete"
    assert gateway.execution_status_requests[0][1] == _ExecutionTestId.EXECUTE
    assert (
        gateway.execution_status_requests[0][2] == OPENHCS_ZMQ_CONFIG.control_timeout_ms
    )


def test_runtime_server_scan_filters_by_declared_server_role():
    service = RuntimeServerService(gateway=_MixedRoleRuntimeServerGateway())

    scan_result = service.scan(ports=(5555,), timeout_ms=25)

    assert [server.port for server in scan_result.servers] == [5555]


def test_runtime_server_service_bounds_failed_execution_status():
    gateway = _FailedRuntimeStatusGateway()
    service = RuntimeServerService(gateway=gateway)

    status = service.execution_status(
        execution_id=_ExecutionTestId.EXECUTE,
        port=5555,
        timeout_ms=17,
    )

    execution_payload = status.response["execution"]

    assert status.status == "failed"
    assert status.errors[0].code == "execution_failed"
    assert status.errors[0].message == "runtime failure"
    assert len(execution_payload["traceback"]) == MAX_EXECUTION_STATUS_TRACEBACK_CHARS
    assert execution_payload["traceback_truncated"] is True
    assert (
        execution_payload["traceback_original_chars"]
        == MAX_EXECUTION_STATUS_TRACEBACK_CHARS + 17
    )
    assert gateway.execution_status_requests[0][2] == 17


def test_runtime_server_info_rejects_viewer_endpoint_with_agent_error():
    service = RuntimeServerService(gateway=_WrongKindRuntimeServerGateway())

    server_info = service.server_info(port=5555, timeout_ms=25)

    assert server_info.reachable is False
    assert server_info.errors[0].code == "runtime_server_wrong_type"
    assert "NapariViewer" in server_info.errors[0].message
    assert "openhcs_scan_runtime_servers" in server_info.errors[0].hint


def test_pipeline_authoring_context_is_task_specific_and_does_not_dump_catalog():
    class _UnexpectedFunctionCatalog:
        def search(self, **_kwargs):
            raise AssertionError("pipeline context must search only after user intent")

    context = AgentAuthoringContextService(
        function_catalog=_UnexpectedFunctionCatalog(),
    ).get_authoring_context("pipeline")

    assert context.kind == "pipeline"
    assert len(context.content) < 16_000
    assert "EXECUTION MENTAL MODEL" in context.content
    assert "PIPELINE AUTHORING WORKFLOW" in context.content
    assert "CONFIG SCHEMA HINTS" in context.content
    assert "openhcs_search_functions" in context.content
    assert "openhcs_describe_function" in context.content
    assert "openhcs_configuration_reference" in context.content
    assert "openhcs_artifact_contract_system" in context.content
    assert "Five independent questions govern each step" in context.content
    assert "not a third InputSource value" in context.content
    assert "does not itself promise a persistent file" in context.content
    assert "compiled artifact and materialization plans" in context.content
    assert "two independent inheritance axes" in context.content
    assert "reduces loading, memory, and processing" in context.content
    assert "viewer-local well filter only suppresses viewer emission" in context.content
    assert "REGISTERED OPENHCS FUNCTIONS" not in context.content
    assert "CORE PIPELINE IMPORTS" not in context.content
    assert "RUNTIME AND UI COORDINATION" not in context.content
    assert "CELLPROFILER MENTAL MODEL BRIDGE" not in context.content
    assert "SOURCE-BINDING WORKFLOW" not in context.content


def test_custom_function_authoring_context_is_not_pipeline_context():
    class _UnexpectedFunctionCatalog:
        def search(self, **_kwargs):
            raise AssertionError("custom function context should not search functions")

    context = AgentAuthoringContextService(
        function_catalog=_UnexpectedFunctionCatalog(),
    ).get_authoring_context("custom_function")

    assert context.kind == "custom_function"
    assert len(context.content) < 16_000
    assert "CORE CUSTOM FUNCTION IMPORTS" not in context.content
    assert "from openhcs.core.memory import numpy" in context.content
    assert "MeasurementsArtifactType" in context.content
    assert "ObjectLabelsArtifactType" in context.content
    assert "CUSTOM FUNCTIONS AND RUNTIME OUTPUTS" in context.content
    assert "FunctionStep.func patterns" in context.content
    assert (
        "processing_config.variable_components are the axes stacked" in context.content
    )
    assert "Dict patterns are routed by processing_config.group_by" in context.content
    assert "@numpy" in context.content
    assert "openhcs_register_custom_function" in context.content
    assert "openhcs_custom_function_management" in context.content
    assert "CustomFunctionManager().register_from_code" not in context.content
    assert "CORE PIPELINE IMPORTS" not in context.content
    assert "CONFIG SCHEMA HINTS" not in context.content
    assert "REGISTERED OPENHCS FUNCTIONS" not in context.content


def test_first_use_authoring_context_is_a_bounded_intent_router():
    class _UnexpectedFunctionCatalog:
        def search(self, **_kwargs):
            raise AssertionError("first-use must not enumerate the function catalog")

    context = AgentAuthoringContextService(
        function_catalog=_UnexpectedFunctionCatalog(),
    ).get_authoring_context("first_use")
    normalized_content = " ".join(context.content.split())

    assert context.kind == "first_use"
    assert len(context.content) < 16_000
    assert context.content.startswith("=== CHOOSE AN OPENHCS WORKFLOW ===")
    assert "=== CORE MODEL IN ONE PASS ===" in context.content
    assert "storage-independent virtual workspace" in normalized_content
    assert "multidimensional grouping and artifact contracts" in normalized_content
    assert "=== CHOOSE ONE TASK ROUTE ===" in context.content
    assert 'kind="ui_visible_workflow"' in context.content
    assert 'kind="folder_onboarding"' in context.content
    assert 'kind="pipeline"' in context.content
    assert 'kind="headless_execution"' in context.content
    assert 'kind="viewer_review"' in context.content
    assert context.content.index('kind="ui_visible_workflow"') < context.content.index(
        'kind="folder_onboarding"'
    )
    assert "UI-owned route always takes precedence" in context.content
    assert "Do not read every knowledge document" in context.content
    assert "openhcs_architecture_quick_start" in context.content
    assert "read this complete first_use curriculum" not in context.content
    assert "=== REGISTERED OPENHCS FUNCTIONS ===" not in context.content
    assert "=== CONFIGURATION SYSTEM ===" not in context.content
    assert "=== SOURCE-BINDING WORKFLOW ===" not in context.content
    assert "=== UI-VISIBLE WORKFLOW ===" not in context.content


def test_default_authoring_context_request_returns_complete_progressive_router():
    request = AuthoringContextRequest()
    context = AgentAuthoringContextService().get_bounded_authoring_context(request)

    assert request.kind == "first_use"
    assert request.max_chars == 16_000
    assert context.kind == "first_use"
    assert "...<truncated" not in context.content
    assert context.content.startswith("=== CHOOSE AN OPENHCS WORKFLOW ===")
    assert "=== CORE MODEL IN ONE PASS ===" in context.content
    assert "=== CHOOSE ONE TASK ROUTE ===" in context.content
    assert 'kind="ui_visible_workflow"' in context.content
    assert 'kind="folder_onboarding"' in context.content
    assert 'kind="pipeline"' in context.content
    assert 'kind="headless_execution"' in context.content
    assert 'kind="viewer_review"' in context.content
    assert "=== SOURCE-BINDING WORKFLOW ===" not in context.content
    assert "=== CONFIGURATION SYSTEM ===" not in context.content


def test_domain_expert_context_routes_by_visible_state_ownership():
    context = AgentAuthoringContextService().get_authoring_context(
        "domain_expert_assisted_setup"
    )

    assert context.kind == "domain_expert_assisted_setup"
    assert len(context.content) < 16_000
    assert 'kind="ui_visible_workflow"' in context.content
    assert 'kind="folder_onboarding"' in context.content
    assert "compiled artifacts, bounded result samples" in context.content
    assert "=== UI-VISIBLE WORKFLOW ===" not in context.content
    assert "=== FOLDER ONBOARDING WORKFLOW ===" not in context.content
