from pyqt_reactive.services.zmq_server_info import (
    BaseServerInfo,
    ExecutionServerInfo,
    GenericServerInfo,
    ViewerServerInfo,
)
from zmqruntime.messages import PongResponse


def test_execution_server_payload_parses_to_typed_info():
    payload = {
        "type": "pong",
        "port": 7777,
        "control_port": 8777,
        "ready": True,
        "server": "OpenHCSExecutionServer",
        "server_type": "execution",
        "server_role": "execution",
        "log_file_path": "/tmp/server.log",
        "workers": [
            {
                "pid": 1234,
                "status": "running",
                "cpu_percent": 12.0,
                "memory_mb": 256.0,
            }
        ],
        "running_executions": [
            {
                "execution_id": "exec-1",
                "plate_id": "/tmp/p1",
                "compile_only": True,
            },
        ],
        "queued_executions": [
            {"execution_id": "exec-2", "plate_id": "/tmp/p2", "queue_position": 1},
        ],
        "compile_status": "compiled success",
        "compile_message": "ok",
    }

    response = PongResponse.from_dict(payload)
    info = BaseServerInfo.from_response(response)

    assert isinstance(info, ExecutionServerInfo)
    assert info.port == 7777
    assert len(info.workers) == 1
    assert info.running_executions == ("exec-1",)
    assert info.queued_executions == ("exec-2",)
    assert info.running_execution_entries[0].plate_id == "/tmp/p1"
    assert info.running_execution_entries[0].compile_only is True
    assert info.queued_execution_entries[0].queue_position == 1
    assert info.response.compile_status == "compiled success"
    assert info.response.compile_message == "ok"


def test_viewer_server_payload_parses_to_typed_info():
    payload = {
        "type": "pong",
        "port": 7780,
        "control_port": 8780,
        "ready": True,
        "server": "NapariViewerServer",
        "server_type": "napari",
        "server_role": "viewer",
        "log_file_path": "/tmp/napari.log",
        "memory_mb": 1024.5,
        "cpu_percent": 8.25,
    }

    info = BaseServerInfo.from_response(PongResponse.from_dict(payload))

    assert isinstance(info, ViewerServerInfo)
    assert info.viewer_name == "napari"
    assert info.memory_mb == 1024.5
    assert info.cpu_percent == 8.25


def test_unknown_server_payload_parses_to_generic_info():
    payload = {
        "type": "pong",
        "port": 9000,
        "control_port": 10000,
        "ready": False,
        "server": "CustomServer",
        "server_role": "generic",
        "log_file_path": None,
    }

    info = BaseServerInfo.from_response(PongResponse.from_dict(payload))

    assert isinstance(info, GenericServerInfo)
    assert info.server_name == "CustomServer"
