#!/usr/bin/env python3
"""Wait until one exact project version is installable through PyPI."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import quote, unquote, urlparse
from urllib.request import urlopen


PYPI_JSON_BASE_URL = "https://pypi.org/pypi"
PYPI_SIMPLE_BASE_URL = "https://pypi.org/simple"


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


def release_json_url(project: str, version: str) -> str:
    """Return the exact-version PyPI JSON endpoint for a project release."""
    return (
        f"{PYPI_JSON_BASE_URL}/{quote(project, safe='')}/{quote(version, safe='')}/json"
    )


def simple_project_url(project: str) -> str:
    """Return the normalized PEP 503 project endpoint used by installers."""
    normalized_project = re.sub(r"[-_.]+", "-", project).casefold()
    return f"{PYPI_SIMPLE_BASE_URL}/{quote(normalized_project, safe='')}/"


def probe_release(
    project: str,
    version: str,
    *,
    opener: Callable = urlopen,
) -> PyPIReleaseProbe:
    """Check whether PyPI serves metadata for exactly ``project==version``."""
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

    if not isinstance(payload, dict):
        return PyPIReleaseProbe(False, "PyPI returned a non-object JSON payload")
    info = payload.get("info")
    published_version = info.get("version") if isinstance(info, dict) else None
    if published_version != version:
        return PyPIReleaseProbe(
            False,
            f"PyPI returned version {published_version!r} instead of {version!r}",
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
        return PyPIReleaseProbe(False, "exact release has no downloadable files yet")

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

    expected_filenames = {
        release_file["filename"] for release_file in downloadable_files
    }
    visible_filenames = expected_filenames & parser.filenames
    if not visible_filenames:
        return PyPIReleaseProbe(
            False,
            "exact release metadata is visible but the installer index has not "
            "propagated it yet",
        )
    return PyPIReleaseProbe(
        True,
        f"PyPI metadata and installer index serve {project}=={version} with "
        f"{len(visible_filenames)} installable file(s)",
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


def _positive_number(value: str) -> float:
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
        type=_positive_number,
        default=300.0,
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=_positive_number,
        default=5.0,
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
    return 0 if result.available else 1


if __name__ == "__main__":
    raise SystemExit(main())
