"""Asynchronous discovery and safe handoff for OpenHCS desktop updates."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from packaging.version import InvalidVersion, Version
from PyQt6.QtCore import QByteArray, QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PyQt6.QtWidgets import QMessageBox
from pyqt_reactive.process_launch import BackgroundProcessLaunchPolicy

from openhcs import __version__ as OPENHCS_VERSION
from openhcs.desktop_deployment import (
    DESKTOP_RESTART_EXECUTABLE_ENVIRONMENT_VARIABLE,
    DesktopDeploymentAuthority,
    DesktopDeploymentContext,
    DesktopDeploymentError,
)
from openhcs.desktop_installation import DESKTOP_INSTALL_PROFILE
from openhcs.mcp.bootstrap import MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE
from openhcs.pyqt_gui.services.desktop_update_worker import DesktopUpdatePlan

if TYPE_CHECKING:
    from openhcs.pyqt_gui.services.service_adapter import PyQtServiceAdapter

LATEST_RELEASE_API_URL = (
    "https://api.github.com/repos/OpenHCSDev/openhcs/releases/latest"
)
_OFFICIAL_RELEASE_HOST = "github.com"
_OFFICIAL_RELEASE_PATH = "/OpenHCSDev/openhcs/releases/"
_SESSION_DOCUMENT_NAME = "session.py"
_HISTORY_DOCUMENT_NAME = "objectstate-history.objectstate"
_WORKER_DOCUMENT_NAME = "desktop-update-worker.py"
_UPDATE_PLAN_DOCUMENT_NAME = "desktop-update-plan.json"
_PROGRESS_THEME_DOCUMENT_NAME = "desktop-update-theme.json"
_PROGRESS_BRAND_DOCUMENT_NAME = "desktop-update-brand.png"
_UPDATE_ERROR_NAME = "update-error.txt"
_SESSION_PURPOSE_NAME = "restart-purpose.txt"
UPDATE_SESSION_ARGUMENT = "--restore-update-session"


class DesktopUpdateError(RuntimeError):
    """Raised when official release metadata cannot produce a safe update."""


class DesktopUpdateCheckOrigin(Enum):
    """User-visible context that requested one release check."""

    EXPLICIT = "explicit"
    STARTUP = "startup"


class DesktopRestartPurpose(Enum):
    """User-facing purpose that owns one captured desktop restart."""

    UPDATE = (
        "update",
        "OpenHCS updated successfully and restored the working session and edit history.",
        True,
    )
    ZMQ_VERSION = (
        "zmq_version",
        "OpenHCS restarted with a matching execution server and restored the "
        "working session and edit history.",
        False,
    )

    def __new__(cls, value: str, success_message: str, requires_update_assets: bool):
        member = object.__new__(cls)
        member._value_ = value
        member.success_message = success_message
        member.requires_update_assets = requires_update_assets
        return member


@dataclass(frozen=True, slots=True)
class DesktopRestartEnvironment:
    """Current interpreter and executable declarations needed for a relaunch."""

    worker_python_executable: Path
    restart_executable: Path
    restart_arguments: tuple[str, ...]

    @classmethod
    def current(cls) -> "DesktopRestartEnvironment":
        python_executable = Path(sys.executable).absolute()
        environment_root = Path(sys.prefix).resolve()
        worker_python_executable = Path(sys._base_executable).resolve()
        if not python_executable.is_file():
            raise DesktopUpdateError(
                f"The running Python executable is unavailable: {python_executable}"
            )
        if not worker_python_executable.is_file():
            worker_python_executable = python_executable

        restart_arguments = _without_update_session_arguments(sys.argv[1:])
        raw_stable_restart = os.environ.get(
            DESKTOP_RESTART_EXECUTABLE_ENVIRONMENT_VARIABLE
        )
        if raw_stable_restart is not None:
            stable_restart = Path(raw_stable_restart).expanduser()
            if not stable_restart.is_absolute():
                raise DesktopUpdateError(
                    "The native installer launcher supplied a relative restart "
                    "executable. Re-run the official installer to repair this "
                    "installation."
                )
            if not stable_restart.is_file():
                raise DesktopUpdateError(
                    "The native installer restart executable is unavailable. "
                    "Re-run the official installer to repair this installation: "
                    f"{stable_restart}"
                )
            restart_executable = stable_restart
        else:
            invoked_path = Path(sys.argv[0]).expanduser()
            if invoked_path.is_file() and invoked_path.resolve().is_relative_to(
                environment_root
            ):
                restart_executable = invoked_path.resolve()
            else:
                restart_executable = python_executable
                restart_arguments = ("-m", "openhcs.pyqt_gui", *restart_arguments)
        return cls(
            worker_python_executable=worker_python_executable,
            restart_executable=restart_executable,
            restart_arguments=restart_arguments,
        )


@dataclass(frozen=True, slots=True)
class DesktopRuntimeEnvironment:
    """Validated installed virtual environment that owns the running GUI."""

    python_executable: Path
    worker_python_executable: Path
    environment_root: Path
    restart_executable: Path
    restart_arguments: tuple[str, ...]
    installation_pointer: Path | None = None

    @staticmethod
    def _environment_entry_path(path: Path) -> Path:
        """Resolve directory indirection while preserving a venv entry symlink."""

        return path.parent.resolve() / path.name

    @classmethod
    def current(cls) -> DesktopRuntimeEnvironment:
        restart = DesktopRestartEnvironment.current()
        python_executable = Path(sys.executable).absolute()
        environment_root = Path(sys.prefix).resolve()
        base_prefix = Path(sys.base_prefix).resolve()
        if environment_root == base_prefix:
            raise DesktopUpdateError(
                "Automatic updates require OpenHCS to run from a virtual "
                "environment. Use the official installer or release instructions "
                "for this installation."
            )
        if not python_executable.is_file() or not cls._environment_entry_path(
            python_executable
        ).is_relative_to(environment_root):
            raise DesktopUpdateError(
                "The running Python executable does not belong to the OpenHCS "
                f"virtual environment: {python_executable}"
            )
        if (
            not restart.worker_python_executable.is_file()
            or restart.worker_python_executable.is_relative_to(environment_root)
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

        raw_installation_pointer = os.environ.get(
            MCP_INSTALLATION_POINTER_ENVIRONMENT_VARIABLE
        )
        installation_pointer = (
            None
            if raw_installation_pointer is None
            else Path(raw_installation_pointer).expanduser()
        )
        if installation_pointer is not None and not installation_pointer.is_absolute():
            raise DesktopUpdateError(
                "The native installer launcher supplied a relative installation "
                "pointer. Re-run the official installer to repair this installation."
            )

        return cls(
            python_executable=python_executable,
            worker_python_executable=restart.worker_python_executable,
            environment_root=environment_root,
            restart_executable=restart.restart_executable,
            restart_arguments=restart.restart_arguments,
            installation_pointer=installation_pointer,
        )

    def update_plan(self, latest_version: Version) -> DesktopUpdatePlan:
        if self.installation_pointer is None:
            raise DesktopUpdateError(
                "Automatic updates require an installation created by the official "
                "Windows or macOS installer. Open the release page to update this "
                "environment manually."
            )
        try:
            context = DesktopDeploymentContext.from_runtime(
                self.installation_pointer,
                environment_root=self.environment_root,
            )
            candidate = DesktopDeploymentAuthority.current().update_candidate(context)
        except DesktopDeploymentError as exc:
            raise DesktopUpdateError(str(exc)) from exc
        try:
            install_profile = DESKTOP_INSTALL_PROFILE.select(
                distribution("openhcs").metadata["Name"],
                latest_version,
            )
        except (PackageNotFoundError, KeyError, TypeError, ValueError) as exc:
            raise DesktopUpdateError(
                "The installed desktop profile is missing or invalid. Re-run the "
                "official installer to repair this installation."
            ) from exc
        return DesktopUpdatePlan(
            update_executable=str(context.uv_executable),
            base_python_executable=str(self.worker_python_executable),
            previous_environment=str(self.environment_root),
            candidate_environment=str(candidate.root),
            candidate_python_executable=str(candidate.python_executable),
            package_requirement=install_profile.package_requirement,
            binary_only_packages=install_profile.binary_only_argument,
            expected_version=str(latest_version),
            installation_pointer=str(self.installation_pointer),
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
class DesktopRestartSession:
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
    def update_plan_document(self) -> Path:
        return self.directory / _UPDATE_PLAN_DOCUMENT_NAME

    @property
    def progress_theme_document(self) -> Path:
        return self.directory / _PROGRESS_THEME_DOCUMENT_NAME

    @property
    def progress_brand_document(self) -> Path:
        return self.directory / _PROGRESS_BRAND_DOCUMENT_NAME

    @property
    def update_error_document(self) -> Path:
        return self.directory / _UPDATE_ERROR_NAME

    @property
    def purpose_document(self) -> Path:
        return self.directory / _SESSION_PURPOSE_NAME

    @property
    def purpose(self) -> DesktopRestartPurpose:
        if not self.purpose_document.is_file():
            return DesktopRestartPurpose.UPDATE
        return DesktopRestartPurpose(
            self.purpose_document.read_text(encoding="utf-8").strip()
        )

    @classmethod
    def pending(cls) -> DesktopRestartSession:
        from openhcs.core.xdg_paths import get_openhcs_cache_dir

        return cls(get_openhcs_cache_dir() / "desktop-updates" / "pending")

    @property
    def is_complete(self) -> bool:
        return self.session_document.is_file() and self.history_document.is_file()

    @classmethod
    def capture(
        cls,
        main_window,
        *,
        purpose: DesktopRestartPurpose = DesktopRestartPurpose.UPDATE,
    ) -> DesktopRestartSession:
        from objectstate.object_state import ObjectStateRegistry

        from openhcs.pyqt_gui.config import save_ui_config_sync
        from openhcs.pyqt_gui.services.desktop_update_worker import (
            DesktopUpdateProgressTheme,
        )
        from openhcs.pyqt_gui.widgets.plate_manager import (
            PlateManagerCodeSelectionMode,
        )
        from openhcs.resources.brand import BrandAsset, brand_asset_path

        plate_manager = main_window.embedded_widgets.require_plate_manager()
        if plate_manager.is_any_plate_running():
            raise DesktopUpdateError(
                "Stop the active plate execution before restarting OpenHCS."
            )
        context = plate_manager.orchestrator_code_document_context(
            selection_mode=PlateManagerCodeSelectionMode.ALL,
        )
        session = cls.pending()
        if session.directory.exists():
            raise DesktopUpdateError(
                "A saved OpenHCS restart session is already pending. Restart "
                "OpenHCS to recover it before starting another restart."
            )
        session.directory.mkdir(parents=True)
        try:
            session.session_document.write_text(context.source, encoding="utf-8")
            ObjectStateRegistry.save_history_to_file(str(session.history_document))
            session.purpose_document.write_text(purpose.value, encoding="utf-8")
            if purpose.requires_update_assets:
                shutil.copyfile(
                    Path(__file__).with_name("desktop_update_worker.py"),
                    session.worker_document,
                )
                color_scheme = main_window.window_services.get_current_color_scheme()
                DesktopUpdateProgressTheme(
                    window_bg=color_scheme.to_hex(color_scheme.window_bg),
                    panel_bg=color_scheme.to_hex(color_scheme.panel_bg),
                    text_primary=color_scheme.to_hex(color_scheme.text_primary),
                    text_secondary=color_scheme.to_hex(color_scheme.text_secondary),
                    text_accent=color_scheme.to_hex(color_scheme.text_accent),
                    border_color=color_scheme.to_hex(color_scheme.border_color),
                    button_bg=color_scheme.to_hex(color_scheme.button_normal_bg),
                    button_text=color_scheme.to_hex(color_scheme.button_text),
                    error_color=color_scheme.to_hex(color_scheme.status_error),
                    progress_color=color_scheme.to_hex(color_scheme.progress_fill),
                ).write(session.progress_theme_document)
                shutil.copyfile(
                    brand_asset_path(BrandAsset.ICON_RASTER),
                    session.progress_brand_document,
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


@dataclass(frozen=True, slots=True)
class DesktopUpdateCheckResult:
    """Successful update check plus the context that requested it."""

    update: DesktopUpdate
    origin: DesktopUpdateCheckOrigin


@dataclass(frozen=True, slots=True)
class DesktopUpdateCheckFailure:
    """Failed update check plus the context that requested it."""

    message: str
    origin: DesktopUpdateCheckOrigin


class DesktopUpdateDialogPresenter:
    """Present update decisions with the application's shared dialog theme."""

    def __init__(
        self,
        dialog_service: PyQtServiceAdapter,
    ) -> None:
        self._dialog_service = dialog_service

    def show_up_to_date(self, update: DesktopUpdate) -> None:
        """Report an explicit check that found no newer stable release."""

        self._dialog_service.create_message_box(
            icon=QMessageBox.Icon.Information,
            title="OpenHCS Updates",
            text=f"OpenHCS {update.installed_version} is the latest stable release.",
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
        ).exec()

    def confirm_update(self, update: DesktopUpdate) -> bool:
        """Ask before saving, closing, and updating the running environment."""

        response = QMessageBox.StandardButton(
            self._dialog_service.create_message_box(
                icon=QMessageBox.Icon.Question,
                title="OpenHCS Update Available",
                text=(
                    f"OpenHCS {update.latest_version} is available "
                    f"(installed: {update.installed_version}).\n\n"
                    "Install the update now? OpenHCS will save the complete working "
                    "session and edit history, close, verify a replacement "
                    "environment, then switch over and restore the session."
                ),
                buttons=(
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                ),
                default_button=QMessageBox.StandardButton.Yes,
            ).exec()
        )
        return response == QMessageBox.StandardButton.Yes

    def show_warning(self, message: str) -> None:
        """Report a user-requested update failure."""

        self._dialog_service.create_message_box(
            icon=QMessageBox.Icon.Warning,
            title="OpenHCS Updates",
            text=message,
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
        ).exec()


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
    check_failed = pyqtSignal(object)

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
        self._installed_version = installed_version or OPENHCS_VERSION
        self._system_name = system_name or platform.system()
        self._network_manager = network_manager or QNetworkAccessManager(self)
        self._url_opener = url_opener or QDesktopServices.openUrl
        self._active_reply: QNetworkReply | None = None
        self._active_origin: DesktopUpdateCheckOrigin | None = None

    def check_for_updates(
        self,
        origin: DesktopUpdateCheckOrigin = DesktopUpdateCheckOrigin.EXPLICIT,
    ) -> bool:
        """Start a nonblocking check, returning false if one is already active."""

        if not isinstance(origin, DesktopUpdateCheckOrigin):
            raise TypeError(
                "DesktopUpdateService.check_for_updates requires "
                f"DesktopUpdateCheckOrigin; got {type(origin).__name__}."
            )
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
        request.setAttribute(
            QNetworkRequest.Attribute.Http2AllowedAttribute,
            False,
        )
        request.setTransferTimeout(15_000)
        reply = self._network_manager.get(request)
        self._active_reply = reply
        self._active_origin = origin
        reply.finished.connect(lambda: self._finish_check(reply))
        return True

    def _finish_check(self, reply: QNetworkReply) -> None:
        if reply is not self._active_reply:
            return
        self._active_reply = None
        origin = self._active_origin
        self._active_origin = None
        if origin is None:
            raise RuntimeError("Active desktop update request has no origin.")
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
            self.check_failed.emit(
                DesktopUpdateCheckFailure(message=str(exc), origin=origin)
            )
        else:
            self.check_completed.emit(
                DesktopUpdateCheckResult(update=result, origin=origin)
            )
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
        session: DesktopRestartSession,
        parent_pid: int | None = None,
    ) -> bool:
        """Launch the detached updater that waits for this GUI to close."""

        if not update.update_available:
            raise DesktopUpdateError("The selected release is not newer than OpenHCS.")
        missing_documents = tuple(
            path.name
            for path in (
                session.worker_document,
                session.progress_theme_document,
                session.progress_brand_document,
            )
            if not path.is_file()
        )
        if missing_documents:
            raise DesktopUpdateError(
                "The saved update session is incomplete: "
                + ", ".join(missing_documents)
            )
        plan = runtime.update_plan(update.latest_version)
        plan.write(session.update_plan_document)
        arguments = [
            str(session.worker_document),
            "--parent-pid",
            str(os.getpid() if parent_pid is None else parent_pid),
            "--session-directory",
            str(session.directory),
            "--update-plan-file",
            str(session.update_plan_document),
            "--restart-executable",
            str(runtime.restart_executable),
            "--progress-theme-file",
            str(session.progress_theme_document),
            "--progress-brand-file",
            str(session.progress_brand_document),
            "--error-file",
            str(session.update_error_document),
            f"--restore-option={UPDATE_SESSION_ARGUMENT}",
        ]
        for argument in runtime.restart_arguments:
            arguments.append(f"--restart-argument={argument}")
        background_spec = BackgroundProcessLaunchPolicy.current().resolve()
        detached_spec = BackgroundProcessLaunchPolicy.current(detached=True).resolve()
        arguments.extend(
            (
                f"--background-creationflags={background_spec.creationflags}",
                f"--detached-creationflags={detached_spec.creationflags}",
            )
        )
        if background_spec.start_new_session:
            arguments.append("--background-start-new-session")
        if detached_spec.start_new_session:
            arguments.append("--detached-start-new-session")
        try:
            worker_executable = BackgroundProcessLaunchPolicy.current(
                detached=True
            ).python_executable(str(runtime.worker_python_executable))
            subprocess.Popen(
                [worker_executable, "-I", *arguments],
                close_fds=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **detached_spec.popen_arguments(),
            )
        except OSError:
            return False
        return True
