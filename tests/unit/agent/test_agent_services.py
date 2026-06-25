from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import pytest

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
from openhcs.agent.services import viewer_window_service as viewer_window_service_module
from openhcs.agent.services.viewer_window_service import (
    ViewerWindowGatewayABC,
    ViewerWindowService,
    ZMQViewerWindowGateway,
)
from openhcs.agent.dto.viewer import (
    ViewerWindowPayloadRequest,
    ViewerWindowSnapshotRequest,
    ViewerWindowStateRequest,
    ViewerWindowValidationPolicy,
    ViewerWindowValidationRequest,
)
from openhcs.core.config import Backend, PipelineConfig
from openhcs.core.artifacts import ArtifactKind, ArtifactOutputPlan
from openhcs.core.compiled_step_plan import CompiledStepPlan
from openhcs.runtime.window_snapshot import (
    WindowSnapshotCaptureScope,
)
from openhcs.runtime.viewer_protocol import ViewerPayloadControlOptions
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity


def sample_processing_function(image, sigma: float = 1.0):
    """Apply a small sample operation."""
    return image


def _viewer_connection(port: int = 5584) -> ExecutionConnectionSpec:
    return ExecutionConnectionSpec(port=port)


def test_execution_connection_spec_owns_zmq_endpoint_projection():
    connection = ExecutionConnectionSpec(
        host="127.0.0.1",
        port=5555,
        transport_mode="tcp",
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


class _EnvelopeStatusExecutionClient(_FakeExecutionClient):
    def get_status(self, execution_id=None):
        return {
            "status": "ok",
            "execution": {
                "status": "running",
                "execution_id": execution_id,
            },
        }


class _FakeExecutionClientFactory:
    def __init__(self, client: _FakeExecutionClient) -> None:
        self.client = client

    def create_client(self, connection):
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
        )
        step_plan.execution_groups = ["A01"]
        step_plan.artifact_outputs["objects"] = ArtifactOutputPlan(
            name="objects",
            kind=ArtifactKind.OBJECT_LABELS,
            path="/tmp/out/A01/objects.zarr",
            group_keys=("A01",),
            paths_by_group={"A01": "/tmp/out/A01/objects.zarr"},
        )
        context = SimpleNamespace(step_plans={0: step_plan})
        return {
            "execution_bundle": SimpleNamespace(runtime_contexts={"A01": context}),
            "worker_assignments": {"worker-1": ["A01"]},
        }


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
            "snapshot": request.to_wire_payload().as_dict(),
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
                    "title": "IdentifyPrimaryObjects",
                    "mounted": True,
                    "item_count": 2,
                    "data_types": ("image",),
                    "component_values": (
                        {"well": "A14", "site": 1, "channel": 0},
                        {"well": "B13", "site": 1, "channel": 0},
                    ),
                    "payload_summaries": (
                        {
                            "data_type": "image",
                            "path": "/tmp/A14.tif",
                            "components": {"well": "A14", "site": 1, "channel": 0},
                            "payload_type": "ndarray",
                            "shape": (16, 16),
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
                            "dtype": "uint16",
                            "size": 256,
                            "nonzero_count": 96,
                        },
                    ),
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


class _RgbViewerWindowGateway(_FakeViewerWindowGateway):
    def window_state(self, request):
        state = super().window_state(request)
        layer = dict(state["layers"][0])
        layer["data_shape"] = (2, 1, 1, 16, 16, 3)
        layer["payload_summaries"] = tuple(
            {
                **payload_summary,
                "shape": (16, 16, 3),
                "size": 16 * 16 * 3,
            }
            for payload_summary in layer["payload_summaries"]
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
    assert result.viewer.viewer_type == "napari"
    assert result.viewer.title == "OpenHCS Napari Viewer"
    assert result.resource is not None
    assert result.resource.mime_type == "image/png"
    assert result.width == 640
    assert result.height == 480
    assert result.capture_scope is WindowSnapshotCaptureScope.WINDOW
    assert (
        gateway.requests[0].output_dir_path
        == "/tmp/openhcs-mcp-window-snapshots"
    )
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
    assert result.viewer.viewer_type == "napari"
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
    assert layer.visible is True
    assert layer.selected is True
    assert gateway.requests[0].timeout_ms == 5000


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
    assert result.viewer.viewer_type == "napari"
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
    assert result.viewer.viewer_type == "napari"
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
    assert result.payload_count == 2
    assert result.nonzero_payload_count == 2
    assert result.zero_payload_count == 0
    assert result.missing_nonzero_count == 0
    assert result.missing_payload_coordinate_count == 0
    assert result.duplicate_payload_coordinate_count == 0
    assert result.payload_without_coordinate_count == 0
    assert result.spatial_mismatch_count == 0
    assert result.required_axis_labels == ("well", "site", "channel")
    assert result.warnings == ()
    assert result.state is not None
    assert len(result.layer_summaries) == 1
    layer_summary = result.layer_summaries[0]
    assert layer_summary.valid is True
    assert layer_summary.route_key == "IdentifyPrimaryObjects|image"
    assert layer_summary.coordinate_gap_count == 0
    assert layer_summary.expected_coordinate_count == 2
    assert layer_summary.payload_coordinate_count == 2
    assert layer_summary.axis_labels == ("well", "site", "channel", "y", "x")


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


def test_viewer_window_service_validation_accepts_channel_last_rgb_spatial_shape():
    result = ViewerWindowService(gateway=_RgbViewerWindowGateway()).validation_summary(
        ViewerWindowValidationRequest(connection=_viewer_connection())
    )

    assert result.valid is True
    assert result.layer_summaries[0].spatial_mismatch_count == 0


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
    assert (
        fake_client.execution_submissions[0].compile_artifact_id
        == _ExecutionTestId.COMPILE
    )


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


def test_execution_session_service_inspects_pycodified_artifact_plan(
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
        PycodifiedPipelineSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source="pipeline_steps = []\n",
            global_config_id=None,
            pipeline_config_id=None,
            connection=ExecutionConnectionSpec(),
        ),
        axis_filter=("A01",),
    )

    assert compile_gateway.requests[0].plate == tmp_path.resolve()
    assert compile_gateway.requests[0].pipeline_source == "pipeline_steps = []\n"
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
    assert inspection.steps[0].artifact_outputs[0].name == "objects"
    assert inspection.steps[0].artifact_outputs[0].kind == "object_labels"
    assert inspection.steps[0].artifact_outputs[0].paths_by_group == (
        {"group_key": "A01", "path": "/tmp/out/A01/objects.zarr"},
    )


def test_pycodified_pipeline_session_rejects_execution_plate_path(tmp_path: Path):
    with pytest.raises(ValueError, match="execution_plate_id must be None"):
        PycodifiedPipelineSessionRequest(
            identity=ZMQExecutionIdentity(
                plate_id=str(tmp_path),
                execution_plate_id=str(tmp_path),
            ),
            pipeline_source="pipeline_steps = []\n",
            global_config_id=None,
            pipeline_config_id=None,
            connection=ExecutionConnectionSpec(),
        )


def test_execution_session_service_reports_nested_runtime_status(
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
        PycodifiedPipelineSessionRequest(
            identity=ZMQExecutionIdentity(plate_id=str(tmp_path)),
            pipeline_source="pipeline_steps = []\n",
            global_config_id=None,
            pipeline_config_id=None,
            connection=ExecutionConnectionSpec(),
        )
    )
    compile_ref = execution_service.submit_compile(session_ref.session_id)
    status = execution_service.get_job_status(compile_ref.job_id)

    assert status.status == "running"
    assert status.response["status"] == "ok"


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
    context = AgentAuthoringContextService(
        _catalog(monkeypatch)
    ).get_authoring_context()

    assert context.kind == "pipeline"
    assert "CONFIG SCHEMA HINTS" in context.content
    assert "test:sample_processing_function" in context.content
