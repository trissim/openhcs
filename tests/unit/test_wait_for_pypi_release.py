"""Tests for the exact-release PyPI availability gate."""

import io
import json
from urllib.error import HTTPError

from scripts import wait_for_pypi_release as release_wait


class _JsonResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


def test_probe_release_requires_the_exact_returned_version():
    def opener(url, timeout):
        assert timeout == 30
        if url.endswith("/simple/openhcs/"):
            return _JsonResponse(
                b'<a href="https://files.pythonhosted.org/openhcs-0.5.22-py3-none-any.whl">'
                b"openhcs-0.5.22-py3-none-any.whl</a>"
            )
        assert url.endswith("/openhcs/0.5.22/json")
        return _JsonResponse(
            json.dumps(
                {
                    "info": {"version": "0.5.22"},
                    "urls": [
                        {
                            "filename": "openhcs-0.5.22-py3-none-any.whl",
                            "url": "https://files.pythonhosted.org/openhcs.whl",
                        }
                    ],
                }
            ).encode()
        )

    assert release_wait.probe_release(
        "openhcs",
        "0.5.22",
        opener=opener,
    ) == release_wait.PyPIReleaseProbe(
        True,
        "PyPI metadata and installer index serve openhcs==0.5.22 with "
        "1 installable file(s)",
    )


def test_probe_release_treats_not_found_as_not_visible():
    def opener(url, timeout):
        raise HTTPError(url, 404, "not found", {}, None)

    assert release_wait.probe_release(
        "openhcs",
        "0.5.22",
        opener=opener,
    ) == release_wait.PyPIReleaseProbe(False, "exact release is not visible yet")


def test_probe_release_rejects_malformed_json_shape():
    def opener(url, timeout):
        return _JsonResponse(json.dumps([]).encode())

    assert release_wait.probe_release(
        "openhcs",
        "0.5.22",
        opener=opener,
    ) == release_wait.PyPIReleaseProbe(
        False,
        "PyPI returned a non-object JSON payload",
    )


def test_probe_release_waits_for_downloadable_files():
    def opener(url, timeout):
        return _JsonResponse(
            json.dumps({"info": {"version": "0.5.22"}, "urls": []}).encode()
        )

    assert release_wait.probe_release(
        "openhcs",
        "0.5.22",
        opener=opener,
    ) == release_wait.PyPIReleaseProbe(
        False,
        "exact release has no downloadable files yet",
    )


def test_probe_release_waits_for_installer_index_propagation():
    def opener(url, timeout):
        assert timeout == 30
        if url.endswith("/simple/openhcs/"):
            return _JsonResponse(
                b'<a href="https://files.pythonhosted.org/openhcs-0.5.21.whl">'
                b"openhcs-0.5.21.whl</a>"
            )
        return _JsonResponse(
            json.dumps(
                {
                    "info": {"version": "0.5.22"},
                    "urls": [
                        {
                            "filename": "openhcs-0.5.22-py3-none-any.whl",
                            "url": "https://files.pythonhosted.org/openhcs.whl",
                        }
                    ],
                }
            ).encode()
        )

    assert release_wait.probe_release(
        "OpenHCS",
        "0.5.22",
        opener=opener,
    ) == release_wait.PyPIReleaseProbe(
        False,
        "exact release metadata is visible but the installer index has not "
        "propagated it yet",
    )


def test_wait_for_release_polls_the_signal_within_the_time_bound():
    results = iter(
        (
            release_wait.PyPIReleaseProbe(False, "not yet"),
            release_wait.PyPIReleaseProbe(True, "published"),
        )
    )
    clock = iter((0.0, 0.0))
    sleeps = []

    result = release_wait.wait_for_release(
        "openhcs",
        "0.5.22",
        timeout_seconds=30,
        poll_interval_seconds=5,
        probe=lambda project, version: next(results),
        monotonic=lambda: next(clock),
        sleeper=sleeps.append,
    )

    assert result == release_wait.PyPIReleaseProbe(True, "published")
    assert sleeps == [5]


def test_wait_for_release_reports_last_probe_at_timeout():
    clock = iter((0.0, 10.0))

    result = release_wait.wait_for_release(
        "openhcs",
        "0.5.22",
        timeout_seconds=10,
        poll_interval_seconds=2,
        probe=lambda project, version: release_wait.PyPIReleaseProbe(
            False,
            "registry lag",
        ),
        monotonic=lambda: next(clock),
        sleeper=lambda seconds: None,
    )

    assert result == release_wait.PyPIReleaseProbe(
        False,
        "timed out waiting for openhcs==0.5.22; last probe: registry lag",
    )
