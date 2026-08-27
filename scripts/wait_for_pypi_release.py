#!/usr/bin/env python3
"""Wait until one exact project version is installable through PyPI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import urlopen

PYPI_JSON_BASE_URL = "https://pypi.org/pypi"
PYPI_SIMPLE_BASE_URL = "https://pypi.org/simple"


class PyPIReleaseMetadataError(ValueError):
    """Raised when exact-version PyPI metadata violates its contract."""


class _SimpleIndexParser(HTMLParser):
    """Collect distribution filenames exposed by a PEP 503 project page."""

    def __init__(self) -> None:
        super().__init__()
        self.filenames: set[str] = set()

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "a":
            return
        href = dict(attrs).get("href")
        if not href:
            return
        path = urlparse(href).path
        filename = unquote(path.rsplit("/", maxsplit=1)[-1])
        if filename:
            self.filenames.add(filename)


@dataclass(frozen=True, slots=True)
class PyPIReleaseProbe:
    """Result of checking one exact package version through PyPI."""

    available: bool
    detail: str
    wheel_url: str | None = None


@dataclass(frozen=True, slots=True)
class PyPIReleaseFile:
    """One immutable file declared by exact-version PyPI metadata."""

    filename: str
    url: str
    sha256: str

    @property
    def hash_pinned_url(self) -> str:
        """Return the installer-safe URL projected from the declared digest."""
        return f"{self.url}#sha256={self.sha256}"


def _release_files_from_payload(
    payload: object,
    version: str,
) -> tuple[PyPIReleaseFile, ...]:
    """Validate exact-version metadata and return its immutable files."""
    if not isinstance(payload, dict):
        raise PyPIReleaseMetadataError("PyPI returned a non-object JSON payload")
    info = payload.get("info")
    published_version = info.get("version") if isinstance(info, dict) else None
    if published_version != version:
        raise PyPIReleaseMetadataError(
            f"PyPI returned version {published_version!r} instead of {version!r}"
        )
    release_files = payload.get("urls")
    downloadable_files = (
        tuple(
            release_file
            for release_file in release_files
            if isinstance(release_file, dict)
            and isinstance(release_file.get("url"), str)
            and release_file["url"]
            and isinstance(release_file.get("filename"), str)
            and release_file["filename"]
        )
        if isinstance(release_files, list)
        else ()
    )
    if not downloadable_files:
        raise PyPIReleaseMetadataError("exact release has no downloadable files yet")

    validated: list[PyPIReleaseFile] = []
    for release_file in downloadable_files:
        filename = release_file["filename"]
        if Path(filename).name != filename or filename in {".", ".."}:
            raise PyPIReleaseMetadataError(
                f"exact release has an unsafe filename: {filename!r}"
            )
        digests = release_file.get("digests")
        sha256 = digests.get("sha256") if isinstance(digests, dict) else None
        if not isinstance(sha256, str) or re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise PyPIReleaseMetadataError(
                f"exact release file {filename!r} has no valid SHA-256 digest"
            )
        file_url = release_file["url"]
        parsed_url = urlparse(file_url)
        if (
            parsed_url.scheme != "https"
            or parsed_url.hostname is None
            or parsed_url.username is not None
            or parsed_url.password is not None
            or parsed_url.fragment
        ):
            raise PyPIReleaseMetadataError(
                f"exact release file {filename!r} does not use a plain HTTPS URL"
            )
        validated.append(PyPIReleaseFile(filename, file_url, sha256))
    if len({release_file.filename for release_file in validated}) != len(validated):
        raise PyPIReleaseMetadataError(
            "exact release metadata contains duplicate filenames"
        )
    return tuple(validated)


def release_json_url(project: str, version: str) -> str:
    """Return the exact-version PyPI JSON endpoint for a project release."""
    return (
        f"{PYPI_JSON_BASE_URL}/{quote(project, safe='')}/{quote(version, safe='')}/json"
    )


def simple_project_url(project: str) -> str:
    """Return the normalized PEP 503 project endpoint used by installers."""
    normalized_project = re.sub(r"[-_.]+", "-", project).casefold()
    return f"{PYPI_SIMPLE_BASE_URL}/{quote(normalized_project, safe='')}/"


def probe_release_wheel(
    project: str,
    version: str,
    *,
    opener: Callable = urlopen,
) -> PyPIReleaseProbe:
    """Return one hash-pinned wheel from exact PyPI release metadata."""
    url = release_json_url(project, version)
    try:
        with opener(url, timeout=30) as response:
            payload = json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return PyPIReleaseProbe(False, "exact release is not visible yet")
        return PyPIReleaseProbe(False, f"PyPI returned HTTP {exc.code}")
    except (OSError, ValueError) as exc:
        return PyPIReleaseProbe(
            False, f"PyPI probe failed: {type(exc).__name__}: {exc}"
        )

    try:
        release_files = _release_files_from_payload(payload, version)
    except PyPIReleaseMetadataError as exc:
        return PyPIReleaseProbe(False, str(exc))
    published_wheels = tuple(
        release_file
        for release_file in release_files
        if release_file.filename.endswith(".whl")
    )
    if not published_wheels:
        return PyPIReleaseProbe(
            False,
            "exact release is visible but has no published wheel",
        )
    selected_wheel = min(
        published_wheels,
        key=lambda release_file: release_file.filename,
    )
    return PyPIReleaseProbe(
        True,
        f"PyPI metadata serves {project}=={version} with a verified wheel",
        selected_wheel.hash_pinned_url,
    )


def materialize_release_files(
    project: str,
    version: str,
    destination: Path,
    *,
    opener: Callable = urlopen,
) -> tuple[Path, ...]:
    """Download and verify the exact files owned by one PyPI release."""
    metadata_url = release_json_url(project, version)
    try:
        with opener(metadata_url, timeout=30) as response:
            release_files = _release_files_from_payload(json.load(response), version)
        verified_payloads = []
        for release_file in release_files:
            with opener(release_file.url, timeout=60) as response:
                payload = response.read()
            observed_sha256 = hashlib.sha256(payload).hexdigest()
            if observed_sha256 != release_file.sha256:
                raise RuntimeError(
                    f"Downloaded {release_file.filename!r} has SHA-256 "
                    f"{observed_sha256}, expected {release_file.sha256}."
                )
            verified_payloads.append((release_file, payload))
    except HTTPError as exc:
        raise RuntimeError(f"PyPI returned HTTP {exc.code}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not materialize {project}=={version}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    destination.mkdir(parents=True, exist_ok=True)
    materialized: list[Path] = []
    for release_file, payload in verified_payloads:
        output_path = destination / release_file.filename
        output_path.write_bytes(payload)
        materialized.append(output_path)
    return tuple(materialized)


def probe_release(
    project: str,
    version: str,
    *,
    opener: Callable = urlopen,
) -> PyPIReleaseProbe:
    """Check whether PyPI's installer index exposes the exact release wheel."""
    wheel_probe = probe_release_wheel(project, version, opener=opener)
    if not wheel_probe.available:
        return wheel_probe
    if wheel_probe.wheel_url is None:
        raise RuntimeError("Available PyPI wheel probe returned no wheel URL.")

    wheel_filename = unquote(urlparse(wheel_probe.wheel_url).path.rsplit("/", 1)[-1])
    simple_url = simple_project_url(project)
    try:
        with opener(simple_url, timeout=30) as response:
            parser = _SimpleIndexParser()
            parser.feed(response.read().decode("utf-8"))
    except HTTPError as exc:
        return PyPIReleaseProbe(
            False,
            f"PyPI installer index returned HTTP {exc.code}",
        )
    except (OSError, UnicodeError, ValueError) as exc:
        return PyPIReleaseProbe(
            False,
            f"PyPI installer-index probe failed: {type(exc).__name__}: {exc}",
        )

    if wheel_filename not in parser.filenames:
        return PyPIReleaseProbe(
            False,
            "exact release metadata is visible but the installer index has not "
            "propagated it yet",
        )
    return PyPIReleaseProbe(
        True,
        f"PyPI metadata and installer index serve {project}=={version} with "
        "an installable wheel",
        wheel_probe.wheel_url,
    )


def wait_for_release(
    project: str,
    version: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    probe: Callable[[str, str], PyPIReleaseProbe] = probe_release,
    monotonic: Callable[[], float] = time.monotonic,
    sleeper: Callable[[float], None] = time.sleep,
) -> PyPIReleaseProbe:
    """Poll the exact release endpoint until it succeeds or the bound expires."""
    deadline = monotonic() + timeout_seconds
    while True:
        result = probe(project, version)
        if result.available:
            return result
        remaining = deadline - monotonic()
        if remaining <= 0:
            return PyPIReleaseProbe(
                False,
                f"timed out waiting for {project}=={version}; last probe: "
                f"{result.detail}",
            )
        sleeper(min(poll_interval_seconds, remaining))


def wait_for_release_wheel(
    project: str,
    version: str,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> PyPIReleaseProbe:
    """Wait for exact release metadata and return its hash-pinned wheel URL."""
    return wait_for_release(
        project,
        version,
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        probe=probe_release_wheel,
    )


def positive_number(value: str) -> float:
    """Parse one positive finite CLI numeric value."""
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project")
    parser.add_argument("version")
    parser.add_argument(
        "--timeout-seconds",
        type=positive_number,
        default=300.0,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=positive_number,
        default=5.0,
    )
    parser.add_argument(
        "--wheel-url-output",
        type=Path,
        help=(
            "Write the verified exact wheel URL and SHA-256 fragment to this "
            "file after the release becomes available."
        ),
    )
    parser.add_argument(
        "--release-directory",
        type=Path,
        help=(
            "Download every exact-version PyPI file into this directory and "
            "verify it against the metadata-declared SHA-256 digest."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = wait_for_release(
        args.project,
        args.version,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(result.detail)
    if result.available and args.wheel_url_output is not None:
        if result.wheel_url is None:
            raise RuntimeError("Available PyPI release probe returned no wheel URL.")
        args.wheel_url_output.write_text(f"{result.wheel_url}\n", encoding="utf-8")
    if result.available and args.release_directory is not None:
        for path in materialize_release_files(
            args.project,
            args.version,
            args.release_directory,
        ):
            print(path)
    return 0 if result.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
