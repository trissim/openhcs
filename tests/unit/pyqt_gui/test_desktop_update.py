from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtNetwork import QNetworkReply

import openhcs.pyqt_gui.main as main_module
from openhcs.pyqt_gui.services.desktop_update import (
    DesktopUpdateError,
    DesktopUpdateService,
    LATEST_RELEASE_API_URL,
    parse_latest_release,
)


def _release_payload(version: str = "0.7.0") -> dict[str, object]:
    tag = f"v{version}"
    base = f"https://github.com/OpenHCSDev/openhcs/releases/download/{tag}"
    return {
        "tag_name": tag,
        "html_url": f"https://github.com/OpenHCSDev/openhcs/releases/tag/{tag}",
        "draft": False,
        "prerelease": False,
        "assets": [
            {
                "name": "OpenHCS-Windows-Installer.exe",
                "browser_download_url": f"{base}/OpenHCS-Windows-Installer.exe",
            },
            {
                "name": "OpenHCS-macOS-Installer.dmg",
                "browser_download_url": f"{base}/OpenHCS-macOS-Installer.dmg",
            },
            {
                "name": "openhcs-0.7.0-py3-none-any.whl",
                "browser_download_url": f"{base}/openhcs-0.7.0-py3-none-any.whl",
            },
        ],
    }


@pytest.mark.parametrize(
    ("system_name", "expected_suffix", "has_native_installer"),
    [
        ("Windows", ".exe", True),
        ("Darwin", ".dmg", True),
        ("Linux", "/tag/v0.7.0", False),
    ],
)
def test_newer_release_routes_to_platform_handoff(
    system_name: str,
    expected_suffix: str,
    has_native_installer: bool,
) -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name=system_name,
    )

    assert update.update_available
    assert update.has_native_installer is has_native_installer
    assert update.handoff_url.endswith(expected_suffix)


def test_current_release_is_not_reported_as_update() -> None:
    update = parse_latest_release(
        _release_payload("0.6.2"),
        installed_version="0.6.2",
        system_name="Windows",
    )

    assert not update.update_available


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(tag_name="not a version"),
        lambda payload: payload.update(
            html_url="https://example.invalid/OpenHCSDev/openhcs/releases/tag/v0.7.0"
        ),
        lambda payload: payload.update(prerelease=True),
        lambda payload: payload.pop("tag_name"),
    ],
)
def test_malformed_or_untrusted_release_is_rejected(mutation) -> None:
    payload = _release_payload()
    mutation(payload)

    with pytest.raises(DesktopUpdateError):
        parse_latest_release(
            payload,
            installed_version="0.6.2",
            system_name="Windows",
        )


def test_untrusted_or_ambiguous_native_asset_falls_back_to_release_page() -> None:
    payload = _release_payload()
    assets = payload["assets"]
    assert isinstance(assets, list)
    windows_asset = assets[0]
    assert isinstance(windows_asset, dict)
    windows_asset["browser_download_url"] = (
        "https://example.invalid/OpenHCS-Windows-Installer.exe"
    )

    update = parse_latest_release(
        payload,
        installed_version="0.6.2",
        system_name="Windows",
    )

    assert not update.has_native_installer
    assert update.handoff_url == update.release_url


class _Signal:
    def __init__(self) -> None:
        self.callback = None

    def connect(self, callback) -> None:
        self.callback = callback

    def emit(self) -> None:
        assert self.callback is not None
        self.callback()


class _UnavailableReply:
    def __init__(self) -> None:
        self.finished = _Signal()
        self.deleted = False

    def error(self):
        return QNetworkReply.NetworkError.HostNotFoundError

    def errorString(self) -> str:
        return "Host not found"

    def readAll(self) -> bytes:
        return b""

    def deleteLater(self) -> None:
        self.deleted = True


class _NetworkManager:
    def __init__(self, reply) -> None:
        self.reply = reply
        self.request = None

    def get(self, request):
        self.request = request
        return self.reply


def test_unavailable_release_service_emits_failure_without_blocking() -> None:
    reply = _UnavailableReply()
    manager = _NetworkManager(reply)
    service = DesktopUpdateService(
        installed_version="0.6.2",
        system_name="Windows",
        network_manager=manager,
    )
    failures = []
    service.check_failed.connect(failures.append)

    assert service.check_for_updates()
    assert manager.request.url().toString() == LATEST_RELEASE_API_URL
    assert failures == []

    reply.finished.emit()

    assert failures == ["Host not found"]
    assert reply.deleted


def test_open_update_refuses_a_mutated_untrusted_destination() -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Windows",
    )
    opened = []
    service = DesktopUpdateService(
        installed_version="0.6.2",
        system_name="Windows",
        network_manager=_NetworkManager(_UnavailableReply()),
        url_opener=lambda url: opened.append(url.toString()) or True,
    )

    assert service.open_update(update)
    assert opened == [update.handoff_url]

    update = replace(update, handoff_url="https://example.invalid/update.exe")
    with pytest.raises(DesktopUpdateError):
        service.open_update(update)


class _Action:
    def __init__(self) -> None:
        self.enabled = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


class _StatusSignal:
    def __init__(self) -> None:
        self.messages = []

    def emit(self, message: str) -> None:
        self.messages.append(message)


class _UpdateService:
    def __init__(self, *, starts: bool = True) -> None:
        self.starts = starts
        self.opened = []

    def check_for_updates(self) -> bool:
        return self.starts

    def open_update(self, update) -> bool:
        self.opened.append(update)
        return True


def test_main_window_starts_check_only_from_explicit_action() -> None:
    action = _Action()
    status = _StatusSignal()
    main_like = SimpleNamespace(
        desktop_update_service=_UpdateService(),
        check_for_updates_action=action,
        status_message=status,
    )

    main_module.OpenHCSMainWindow.check_for_updates(main_like)

    assert not action.enabled
    assert status.messages == ["Checking for OpenHCS updates…"]


def test_main_window_available_update_opens_verified_handoff(monkeypatch) -> None:
    update = parse_latest_release(
        _release_payload(),
        installed_version="0.6.2",
        system_name="Windows",
    )
    action = _Action()
    action.enabled = False
    status = _StatusSignal()
    service = _UpdateService()
    dialog_calls = []

    class _DialogProbe:
        StandardButton = QMessageBox.StandardButton

        @staticmethod
        def question(*args):
            dialog_calls.append(args)
            return QMessageBox.StandardButton.Open

        @staticmethod
        def warning(*_args):
            raise AssertionError("successful handoff must not show a warning")

    monkeypatch.setattr(main_module, "QMessageBox", _DialogProbe)
    main_like = SimpleNamespace(
        desktop_update_service=service,
        check_for_updates_action=action,
        status_message=status,
    )

    main_module.OpenHCSMainWindow._on_update_check_completed(main_like, update)

    assert action.enabled
    assert status.messages == ["OpenHCS 0.7.0 is available"]
    assert service.opened == [update]
    assert len(dialog_calls) == 1
