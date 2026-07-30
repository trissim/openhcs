"""Asynchronous discovery and safe handoff for OpenHCS desktop updates."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import distribution
from importlib.metadata import version as distribution_version
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlparse

from packaging.version import InvalidVersion, Version
from PyQt6.QtCore import QByteArray, QObject, QProcess, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest

LATEST_RELEASE_API_URL = (
    "https://api.github.com/repos/OpenHCSDev/openhcs/releases/latest"
)
_OFFICIAL_RELEASE_HOST = "github.com"
_OFFICIAL_RELEASE_PATH = "/OpenHCSDev/openhcs/releases/"
_SESSION_DOCUMENT_NAME = "session.py"
_HISTORY_DOCUMENT_NAME = "objectstate-history.objectstate"
_WORKER_DOCUMENT_NAME = "desktop-update-worker.py"
_UPDATE_ERROR_NAME = "update-error.txt"
_UV_EXECUTABLE_ENV = "OPENHCS_UV_EXECUTABLE"
UPDATE_SESSION_ARGUMENT = "--restore-update-session"


class DesktopUpdateError(RuntimeError):
    """Raised when official release metadata cannot produce a safe update."""


@dataclass(frozen=True, slots=True)
class DesktopUpdateCommandPlan:
    """One argument-vector update command for the exact running environment."""

    executable: Path
    arguments: tuple[str, ...]

    @classmethod
    def for_environment(
        cls,
        *,
        python_executable: Path,
        latest_version: Version,
        environment: dict[str, str] | None = None,
    ) -> DesktopUpdateCommandPlan:
        environment_values = os.environ if environment is None else environment
        configured_uv = environment_values.get(_UV_EXECUTABLE_ENV)
        uv_executable = Path(configured_uv).expanduser() if configured_uv else None
        if uv_executable is None:
            discovered_uv = shutil.which("uv")
            if discovered_uv is not None:
                uv_executable = Path(discovered_uv)
        requirement = f"openhcs=={latest_version}"
        if uv_executable is not None and uv_executable.is_file():
            return cls(
                executable=uv_executable.resolve(),
                arguments=(
                    "--no-config",
                    "pip",
                    "install",
                    "--python",
                    str(python_executable),
                    "--upgrade",
                    requirement,
                ),
            )
        if find_spec("pip") is None:
            raise DesktopUpdateError(
                "The running OpenHCS environment has neither an available uv "
                "executable nor its own pip module. Use the official installer "
                "or release instructions to update this installation."
            )
        return cls(
            executable=python_executable,
            arguments=(
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--upgrade",
                requirement,
            ),
        )


@dataclass(frozen=True, slots=True)
class DesktopRuntimeEnvironment:
    """Validated installed virtual environment that owns the running GUI."""

    python_executable: Path
    worker_python_executable: Path
    environment_root: Path
    restart_executable: Path
    restart_arguments: tuple[str, ...]

    @classmethod
    def current(cls) -> DesktopRuntimeEnvironment:
        python_executable = Path(sys.executable).absolute()
        worker_python_executable = Path(sys._base_executable).resolve()
        environment_root = Path(sys.prefix).resolve()
        base_prefix = Path(sys.base_prefix).resolve()
        if environment_root == base_prefix:
            raise DesktopUpdateError(
                "Automatic updates require OpenHCS to run from a virtual "
                "environment. Use the official installer or release instructions "
                "for this installation."
            )
        if (
            not python_executable.is_file()
            or not python_executable.is_relative_to(environment_root)
        ):
            raise DesktopUpdateError(
                "The running Python executable does not belong to the OpenHCS "
                f"virtual environment: {python_executable}"
            )
        if (
            not worker_python_executable.is_file()
            or worker_python_executable.is_relative_to(environment_root)
        ):
            raise DesktopUpdateError(
                "Automatic updates require a base Python interpreter outside "
                "the OpenHCS environment. Use the official installer or release "
                "instructions to update this installation."
            )

        installed_distribution = distribution("openhcs")
        distribution_root = Path(installed_distribution.locate_file("")).resolve()
        direct_url_text = installed_distribution.read_text("direct_url.json")
        if direct_url_text is not None:
            try:
                direct_url = json.loads(direct_url_text)
            except json.JSONDecodeError as exc:
                raise DesktopUpdateError(
                    "The installed OpenHCS distribution has invalid origin metadata."
                ) from exc
            if not isinstance(direct_url, dict):
                raise DesktopUpdateError(
                    "The installed OpenHCS distribution has invalid origin metadata."
                )
            directory_info = direct_url.get("dir_info")
            if (
                isinstance(directory_info, dict)
                and directory_info.get("editable") is True
            ):
                raise DesktopUpdateError(
                    "Automatic updates are disabled for editable or source checkouts. "
                    "Use the official release instructions for this installation."
                )
        if not distribution_root.is_relative_to(environment_root):
            raise DesktopUpdateError(
                "Automatic updates are disabled for editable or source checkouts. "
                "Use the official release instructions for this installation."
            )
        if not os.access(environment_root, os.W_OK):
            raise DesktopUpdateError(
                f"The OpenHCS environment is not writable: {environment_root}"
            )

        invoked_path = Path(sys.argv[0]).expanduser()
        restart_arguments = _without_update_session_arguments(sys.argv[1:])
        if invoked_path.is_file() and invoked_path.resolve().is_relative_to(
            environment_root
        ):
            restart_executable = invoked_path.resolve()
        else:
            restart_executable = python_executable
            restart_arguments = ("-m", "openhcs.pyqt_gui", *restart_arguments)
        return cls(
            python_executable=python_executable,
            worker_python_executable=worker_python_executable,
            environment_root=environment_root,
            restart_executable=restart_executable,
            restart_arguments=restart_arguments,
        )

    def update_command(self, latest_version: Version) -> DesktopUpdateCommandPlan:
        return DesktopUpdateCommandPlan.for_environment(
            python_executable=self.python_executable,
            latest_version=latest_version,
        )


def _without_update_session_arguments(arguments: list[str]) -> tuple[str, ...]:
    """Remove updater-owned restore arguments before a successive restart."""

    sanitized: list[str] = []
    position = 0
    assignment_prefix = f"{UPDATE_SESSION_ARGUMENT}="
    while position < len(arguments):
        argument = arguments[position]
        if argument == UPDATE_SESSION_ARGUMENT:
            position += 2
            continue
        if argument.startswith(assignment_prefix):
            position += 1
            continue
        sanitized.append(argument)
        position += 1
    return tuple(sanitized)


@dataclass(frozen=True, slots=True)
class DesktopUpdateSession:
    """Canonical plate-manager source plus ObjectState history for one restart."""

    directory: Path

    @property
    def session_document(self) -> Path:
        return self.directory / _SESSION_DOCUMENT_NAME

    @property
    def history_document(self) -> Path:
        return self.directory / _HISTORY_DOCUMENT_NAME

    @property
    def worker_document(self) -> Path:
        return self.directory / _WORKER_DOCUMENT_NAME

    @property
    def update_error_document(self) -> Path:
        return self.directory / _UPDATE_ERROR_NAME

    @classmethod
    def pending(cls) -> DesktopUpdateSession:
        from openhcs.core.xdg_paths import get_openhcs_cache_dir

        return cls(get_openhcs_cache_dir() / "desktop-updates" / "pending")

    @property
    def is_complete(self) -> bool:
        return self.session_document.is_file() and self.history_document.is_file()

    @classmethod
    def capture(cls, main_window) -> DesktopUpdateSession:
        from objectstate.object_state import ObjectStateRegistry

        from openhcs.pyqt_gui.config import save_ui_config_sync
        from openhcs.pyqt_gui.widgets.plate_manager import (
            PlateManagerCodeSelectionMode,
        )

        plate_manager = main_window.embedded_widgets.require_plate_manager()
        if plate_manager.is_any_plate_running():
            raise DesktopUpdateError(
                "Stop the active plate execution before updating OpenHCS."
            )
        context = plate_manager.orchestrator_code_document_context(
            selection_mode=PlateManagerCodeSelectionMode.ALL,
        )
        session = cls.pending()
        if session.directory.exists():
            raise DesktopUpdateError(
                "A saved update session is already pending. Restart OpenHCS to "
                "recover it before starting another update."
            )
        session.directory.mkdir(parents=True)
        try:
            session.session_document.write_text(context.source, encoding="utf-8")
            ObjectStateRegistry.save_history_to_file(str(session.history_document))
            shutil.copyfile(
                Path(__file__).with_name("desktop_update_worker.py"),
                session.worker_document,
            )
            if not save_ui_config_sync(main_window.runtime_context.ui_config):
                raise DesktopUpdateError(
                    "OpenHCS could not persist the current UI configuration."
                )
        except Exception:
            session.discard()
            raise
        return session

    def restore(self, main_window) -> str | None:
        """Restore through the existing code-document and ObjectState owners."""

        from objectstate.object_state import ObjectStateRegistry

        source = self.session_document.read_text(encoding="utf-8")
        plate_manager = main_window.embedded_widgets.require_plate_manager()
        plate_manager.apply_code_document_source(source)
        ObjectStateRegistry.load_history_from_file(str(self.history_document))
        main_window.time_travel_widget.refresh()
        plate_manager.update_item_list()
        error_message = (
            self.update_error_document.read_text(encoding="utf-8").strip()
            if self.update_error_document.is_file()
            else None
        )
        self.discard()
        return error_message

    def discard(self) -> None:
        shutil.rmtree(self.directory, ignore_errors=True)


@dataclass(frozen=True, slots=True)
class DesktopUpdate:
    """One installed-to-release comparison and its verified handoff URL."""

    installed_version: Version
    latest_version: Version
    release_url: str
    handoff_url: str
    has_native_installer: bool

    @property
    def update_available(self) -> bool:
        return self.latest_version > self.installed_version


def _is_official_release_url(url: str, *, asset: bool = False) -> bool:
    parsed = urlparse(url)
    required_path = (
        f"{_OFFICIAL_RELEASE_PATH}download/" if asset else _OFFICIAL_RELEASE_PATH
    )
    return (
        parsed.scheme == "https"
        and parsed.hostname == _OFFICIAL_RELEASE_HOST
        and parsed.path.startswith(required_path)
        and not parsed.username
        and not parsed.password
    )


def _native_installer_suffix(system_name: str) -> str | None:
    if system_name == "Windows":
        return ".exe"
    if system_name == "Darwin":
        return ".dmg"
    return None


def parse_latest_release(
    payload: object,
    *,
    installed_version: str,
    system_name: str,
) -> DesktopUpdate:
    """Parse the official latest-release projection into a safe handoff."""

    if not isinstance(payload, dict):
        raise DesktopUpdateError("The release service returned an invalid response.")
    if payload.get("draft") is not False or payload.get("prerelease") is not False:
        raise DesktopUpdateError("The release service did not return a stable release.")

    tag_name = payload.get("tag_name")
    release_url = payload.get("html_url")
    if not isinstance(tag_name, str) or not tag_name:
        raise DesktopUpdateError("The latest release has no valid version tag.")
    if not isinstance(release_url, str) or not _is_official_release_url(release_url):
        raise DesktopUpdateError("The latest release has no trusted release page.")

    try:
        current = Version(installed_version)
        latest = Version(tag_name.removeprefix("v"))
    except InvalidVersion as exc:
        raise DesktopUpdateError(
            "The release service returned an invalid version."
        ) from exc

    handoff_url = release_url
    has_native_installer = False
    suffix = _native_installer_suffix(system_name)
    assets = payload.get("assets")
    if suffix is not None and isinstance(assets, list):
        matching_urls = []
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = asset.get("name")
            download_url = asset.get("browser_download_url")
            if (
                isinstance(name, str)
                and name.casefold().endswith(suffix)
                and isinstance(download_url, str)
                and _is_official_release_url(download_url, asset=True)
            ):
                matching_urls.append(download_url)
        if len(matching_urls) == 1:
            handoff_url = matching_urls[0]
            has_native_installer = True

    return DesktopUpdate(
        installed_version=current,
        latest_version=latest,
        release_url=release_url,
        handoff_url=handoff_url,
        has_native_installer=has_native_installer,
    )


class DesktopUpdateService(QObject):
    """Qt-native asynchronous latest-release check and browser handoff."""

    check_completed = pyqtSignal(object)
    check_failed = pyqtSignal(str)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        installed_version: str | None = None,
        system_name: str | None = None,
        network_manager: QNetworkAccessManager | None = None,
        url_opener: Callable[[QUrl], bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self._installed_version = installed_version or distribution_version("openhcs")
        self._system_name = system_name or platform.system()
        self._network_manager = network_manager or QNetworkAccessManager(self)
        self._url_opener = url_opener or QDesktopServices.openUrl
        self._active_reply: QNetworkReply | None = None

    def check_for_updates(self) -> bool:
        """Start a nonblocking check, returning false if one is already active."""

        if self._active_reply is not None:
            return False

        request = QNetworkRequest(QUrl(LATEST_RELEASE_API_URL))
        request.setRawHeader(
            QByteArray(b"Accept"),
            QByteArray(b"application/vnd.github+json"),
        )
        request.setRawHeader(
            QByteArray(b"User-Agent"),
            QByteArray(f"OpenHCS/{self._installed_version}".encode("ascii", "strict")),
        )
        request.setTransferTimeout(15_000)
        reply = self._network_manager.get(request)
        self._active_reply = reply
        reply.finished.connect(lambda: self._finish_check(reply))
        return True

    def _finish_check(self, reply: QNetworkReply) -> None:
        if reply is not self._active_reply:
            return
        self._active_reply = None
        try:
            if reply.error() != QNetworkReply.NetworkError.NoError:
                raise DesktopUpdateError(reply.errorString())
            payload = json.loads(bytes(reply.readAll()))
            result = parse_latest_release(
                payload,
                installed_version=self._installed_version,
                system_name=self._system_name,
            )
        except (DesktopUpdateError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            self.check_failed.emit(str(exc))
        else:
            self.check_completed.emit(result)
        finally:
            reply.deleteLater()

    def open_update(self, update: DesktopUpdate) -> bool:
        """Open a previously verified installer or release-page handoff."""

        is_asset = update.has_native_installer
        if not _is_official_release_url(update.handoff_url, asset=is_asset):
            raise DesktopUpdateError("Refusing an untrusted update destination.")
        return self._url_opener(QUrl(update.handoff_url))

    def start_update(
        self,
        update: DesktopUpdate,
        *,
        runtime: DesktopRuntimeEnvironment,
        session: DesktopUpdateSession,
        parent_pid: int | None = None,
    ) -> bool:
        """Launch the detached updater that waits for this GUI to close."""

        if not update.update_available:
            raise DesktopUpdateError("The selected release is not newer than OpenHCS.")
        if not session.worker_document.is_file():
            raise DesktopUpdateError(
                "The saved update session does not contain its detached update worker."
            )
        command = runtime.update_command(update.latest_version)
        arguments = [
            str(session.worker_document),
            "--parent-pid",
            str(os.getpid() if parent_pid is None else parent_pid),
            "--session-directory",
            str(session.directory),
            "--update-executable",
            str(command.executable),
            "--restart-executable",
            str(runtime.restart_executable),
            "--expected-version",
            str(update.latest_version),
            "--verification-executable",
            str(runtime.python_executable),
            "--error-file",
            str(session.update_error_document),
            f"--restore-option={UPDATE_SESSION_ARGUMENT}",
        ]
        for argument in command.arguments:
            arguments.append(f"--update-argument={argument}")
        for argument in runtime.restart_arguments:
            arguments.append(f"--restart-argument={argument}")
        started, _process_id = QProcess.startDetached(
            str(runtime.worker_python_executable),
            arguments,
        )
        return started
