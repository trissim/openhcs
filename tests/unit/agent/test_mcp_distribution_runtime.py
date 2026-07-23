from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from openhcs.agent.dto.common import SCHEMA_VERSION
from openhcs.agent.services.ui_bridge_service import (
    UI_BRIDGE_PROTOCOL_VERSION,
    UiBridgeDescriptorDirectoryAuthority,
    UiBridgeDescriptorReader,
)
from openhcs.agent.runtime_platform import (
    AgentRuntimePlatformAuthority,
    AgentRuntimePlatformKey,
    LinuxAgentRuntimePlatformAuthority,
)


def _write_descriptor(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "bridge_protocol_version": UI_BRIDGE_PROTOCOL_VERSION,
                "bridge_instance_id": "windows-test-bridge",
                "pid": os.getpid(),
                "started_at_unix": 1.0,
                "connection": {
                    "host": "127.0.0.1",
                    "port": 7888,
                    "transport_mode": "tcp",
                    "persistent": True,
                },
                "auth_token": "secret",
            }
        ),
        encoding="utf-8",
    )


def test_windows_descriptor_default_uses_local_application_data(
    monkeypatch,
    tmp_path: Path,
):
    local_application_data = tmp_path / "LocalAppData"
    monkeypatch.delenv("OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(local_application_data))
    monkeypatch.setattr(
        AgentRuntimePlatformKey,
        "current",
        classmethod(lambda cls: cls.WINDOWS),
    )

    descriptor_dirs = UiBridgeDescriptorDirectoryAuthority.descriptor_dirs()

    assert descriptor_dirs[0] == (
        local_application_data / "OpenHCS" / "runtime" / "ui-bridge"
    ).resolve(strict=False)


def test_linux_platform_authority_is_registered_under_linux_key():
    authority = AgentRuntimePlatformAuthority.for_enum_member(
        AgentRuntimePlatformKey.LINUX
    )

    assert isinstance(authority, LinuxAgentRuntimePlatformAuthority)


def test_descriptor_reader_uses_cross_platform_process_and_permission_authority(
    monkeypatch,
    tmp_path: Path,
):
    descriptor_path = tmp_path / "ui_bridge_windows.json"
    _write_descriptor(descriptor_path)
    descriptor_path.chmod(0o644)
    windows_authority = AgentRuntimePlatformAuthority.for_enum_member(
        AgentRuntimePlatformKey.WINDOWS
    )
    monkeypatch.setattr(
        AgentRuntimePlatformAuthority,
        "current",
        classmethod(lambda cls: windows_authority),
    )
    monkeypatch.setattr(
        AgentRuntimePlatformAuthority,
        "process_exists",
        staticmethod(lambda pid: pid == os.getpid()),
    )

    result = UiBridgeDescriptorReader.read(descriptor_path)

    assert result.ok is True
    assert result.descriptor is not None
    assert result.descriptor.bridge_instance_id == "windows-test-bridge"


def test_mcp_cold_build_does_not_import_pyqt6():
    script = """
import builtins
import sys

original_import = builtins.__import__

def reject_pyqt(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'PyQt6' or name.startswith('PyQt6.'):
        raise AssertionError(f'headless MCP startup imported {name}')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_pyqt
from openhcs.mcp.server import build_server
build_server()
assert not any(name == 'PyQt6' or name.startswith('PyQt6.') for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_hosted_mcp_cold_build_does_not_import_pyqt6():
    script = """
import builtins
import sys

original_import = builtins.__import__

def reject_pyqt(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'PyQt6' or name.startswith('PyQt6.'):
        raise AssertionError(f'headless hosted MCP startup imported {name}')
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_pyqt
from openhcs.mcp.http import build_http_server
from openhcs.mcp.http_auth import McpHttpResourceServerSettings

settings = McpHttpResourceServerSettings(
    public_url='https://mcp.openhcs.example/mcp',
    issuer_url='https://auth.openhcs.example',
    introspection_url='https://auth.openhcs.example/introspect',
    introspection_client_id='resource-server',
    introspection_client_secret='secret',
    tenant_subject='tenant-user-1',
    required_scopes=('openhcs:use',),
    allowed_hosts=('mcp.openhcs.example',),
)
build_http_server(settings)
assert not any(name == 'PyQt6' or name.startswith('PyQt6.') for name in sys.modules)
assert not any(name == 'openhcs.pyqt_gui' or name.startswith('openhcs.pyqt_gui.') for name in sys.modules)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
