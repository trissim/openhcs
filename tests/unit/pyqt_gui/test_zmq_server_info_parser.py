from pyqt_reactive.services.zmq_server_info_parser import (
    DefaultServerInfoParser,
    ExecutionServerInfo,
    ViewerServerInfo,
    GenericServerInfo,
    ServerKind,
)


def test_execution_server_payload_parses_to_typed_info():
    parser = DefaultServerInfoParser()
    payload = {
        "port": 7777,
        "ready": True,
        "server": "OpenHCSExecutionServer",
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

    info = parser.parse(payload)

    assert isinstance(info, ExecutionServerInfo)
    assert info.kind == ServerKind.EXECUTION
    assert info.port == 7777
    assert len(info.workers) == 1
    assert info.running_executions == ("exec-1",)
    assert info.queued_executions == ("exec-2",)
    assert info.running_execution_entries[0].plate_id == "/tmp/p1"
    assert info.running_execution_entries[0].compile_only is True
    assert info.queued_execution_entries[0].queue_position == 1
    assert info.compile_status is not None
    assert info.compile_status.status_text == "compiled success"
    assert info.compile_status.is_success


def test_viewer_server_payload_parses_to_typed_info():
    parser = DefaultServerInfoParser()
    payload = {
        "port": 7780,
        "ready": True,
        "server": "NapariViewerServer",
        "log_file_path": "/tmp/napari.log",
        "memory_mb": 1024.5,
        "cpu_percent": 8.25,
    }

    info = parser.parse(payload)

    assert isinstance(info, ViewerServerInfo)
    assert info.kind == ServerKind.NAPARI
    assert info.memory_mb == 1024.5
    assert info.cpu_percent == 8.25


def test_unknown_server_payload_parses_to_generic_info():
    parser = DefaultServerInfoParser()
    payload = {
        "port": 9000,
        "ready": False,
        "server": "CustomServer",
        "log_file_path": None,
    }

    info = parser.parse(payload)

    assert isinstance(info, GenericServerInfo)
    assert info.kind == ServerKind.GENERIC
    assert info.server_name == "CustomServer"
