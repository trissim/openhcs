"""Asynchronous discovery and safe handoff for OpenHCS desktop updates."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version as distribution_version
import json
import platform
from typing import Callable
from urllib.parse import urlparse

from packaging.version import InvalidVersion, Version
from PyQt6.QtCore import QByteArray, QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest


LATEST_RELEASE_API_URL = (
    "https://api.github.com/repos/OpenHCSDev/openhcs/releases/latest"
)
_OFFICIAL_RELEASE_HOST = "github.com"
_OFFICIAL_RELEASE_PATH = "/OpenHCSDev/openhcs/releases/"


class DesktopUpdateError(RuntimeError):
    """Raised when official release metadata cannot produce a safe update."""


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
        raise DesktopUpdateError("The release service returned an invalid version.") from exc

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
