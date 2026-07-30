from __future__ import annotations

import pickle
import uuid
from multiprocessing import shared_memory

import numpy as np
import pytest
import zmq

from polystore.streaming.identity import StreamProducerIdentity
from zmqruntime.config import TransportMode
from zmqruntime.transport import get_zmq_transport_url, remove_ipc_socket

from openhcs.runtime.viewer_protocol import NapariViewerServerRequest
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG


def test_napari_settlement_surfaces_terminal_transport_failure() -> None:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    server = type(
        "FailedTransportServer",
        (),
        {"transport_failure": RuntimeError("receiver copy failed")},
    )()

    response = napari_viewer_server.NapariSettleControlMessageAction().handle(
        server,
        {},
    )

    assert response["status"] == "error"
    assert response["type"] == "settle_ack"
    assert response["message"] == (
        "Viewer transport failed before settlement: receiver copy failed"
    )


def test_napari_transport_rep_follows_receiver_owned_shared_memory_copy(
    monkeypatch,
) -> None:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    port = 46000 + uuid.uuid4().int % 10000
    remove_ipc_socket(port, OPENHCS_ZMQ_CONFIG)
    server = napari_viewer_server.NapariViewerServer(
        NapariViewerServerRequest(
            port=port,
            viewer_title="transport ownership test",
            replace_layers=False,
            log_file_path=None,
            transport_mode=TransportMode.IPC,
        )
    )
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    shm = shared_memory.SharedMemory(create=True, size=source.nbytes)
    np.ndarray(source.shape, dtype=source.dtype, buffer=shm.buf)[:] = source
    producer = StreamProducerIdentity.pipeline_output(
        output_kind="main",
        output_key="main",
        projection_key="main",
        step_name="TransportTest",
        pipeline_position=0,
    )
    message = {
        "type": "batch",
        "images": [
            {
                "path": "A01.tif",
                "shm_name": shm.name,
                "shape": list(source.shape),
                "dtype": str(source.dtype),
                "metadata": {"well": "A01"},
                "data_type": "image",
                "image_id": "transport-test-image",
                "producer_identity": producer.to_payload(),
            }
        ],
        "display_config": {
            "component_modes": {"well": "stack"},
            "component_order": ["well"],
        },
        "component_value_domain": {"well": ["A01"]},
        "component_names_metadata": {},
    }
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 5000)

    server._running = True
    server.data_transport_pump.start()
    socket.connect(
        get_zmq_transport_url(
            port,
            host="localhost",
            mode=server.transport_mode,
            config=server.config,
        )
    )
    try:
        # Production sender and receiver processes have independent resource
        # trackers. Keep this same-process probe from unregistering the sender.
        from multiprocessing import resource_tracker

        monkeypatch.setattr(resource_tracker, "unregister", lambda *_args: None)
        socket.send_json(message)
        assert socket.recv_json()["status"] == "success"

        # REP certifies that no later Qt work depends on the sender allocation.
        monkeypatch.undo()
        shm.close()
        shm.unlink()
        accepted = server.accepted_stream_batches.get_nowait()
        np.testing.assert_array_equal(accepted.items[0].data, source)
    finally:
        server._running = False
        server.data_transport_pump.stop()
        socket.close(linger=0)
        context.term()
        remove_ipc_socket(port, OPENHCS_ZMQ_CONFIG)
        try:
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def test_napari_control_pump_reports_active_settlement_without_qt_dispatch() -> None:
    napari_viewer_server = pytest.importorskip("openhcs.runtime.napari_viewer_server")
    from polystore.streaming_constants import StreamingDataType

    from openhcs.runtime.napari_streaming_handlers import NapariPendingLayerUpdate
    from openhcs.runtime.viewer_component_system import (
        ViewerComponentAxisSemanticsAuthority,
    )
    from openhcs.runtime.viewer_protocol import (
        ViewerControlResponse,
        ViewerSettlePhase,
        ViewerSettleProgress,
    )

    class FakeTimer:
        def stop(self) -> None:
            pass

    port = 47000 + uuid.uuid4().int % 8000
    control_port = port + OPENHCS_ZMQ_CONFIG.control_port_offset
    remove_ipc_socket(control_port, OPENHCS_ZMQ_CONFIG)
    server = napari_viewer_server.NapariViewerServer(
        NapariViewerServerRequest(
            port=port,
            viewer_title="control ownership test",
            replace_layers=False,
            log_file_path=None,
            transport_mode=TransportMode.IPC,
        )
    )
    server.viewer = object()
    update = NapariPendingLayerUpdate.from_semantics(
        timer=FakeTimer(),
        data_type=StreamingDataType.SHAPES,
        semantics=ViewerComponentAxisSemanticsAuthority.empty(),
    )
    server.layer_route_state.set_pending_update("large-shapes", update)
    settlement = server.layer_route_state.begin_settlement()
    claimed = settlement.begin_next()
    assert claimed is not None
    settlement.begin_active_work_unit(claimed[0])

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, 2000)
    server._running = True
    server.control_transport_pump.start()
    socket.connect(
        get_zmq_transport_url(
            control_port,
            host="localhost",
            mode=server.transport_mode,
            config=server.config,
        )
    )
    try:
        socket.send(pickle.dumps({"type": "settle"}))
        response = pickle.loads(socket.recv())
        progress = ViewerSettleProgress.from_response(ViewerControlResponse(response))
        assert progress.phase is ViewerSettlePhase.RUNNING
        assert progress.completed_update_count == 0
        assert progress.active_route == "large-shapes"
        assert progress.active_route_work_unit_count == 0
        assert progress.active_route_work_unit_active is True
        assert server.accepted_control_requests.empty()
    finally:
        server._running = False
        server.control_transport_pump.stop()
        socket.close(linger=0)
        context.term()
        remove_ipc_socket(control_port, OPENHCS_ZMQ_CONFIG)
