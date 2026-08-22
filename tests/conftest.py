"""Global pytest configuration for OpenHCS integration tests."""

import os

from openhcs._source_dependencies import ensure_source_checkout_external_paths

ensure_source_checkout_external_paths()

import pytest

# Conditionally import pytest-qt only when not in CPU-only mode
CPU_ONLY_MODE = os.getenv("OPENHCS_CPU_ONLY", "false").lower() == "true"
if not CPU_ONLY_MODE:
    pytest_plugins = ["pytestqt"]
else:
    pytest_plugins = []


@pytest.fixture(autouse=True)
def cleanup_test_runtime_resources():
    """Release connection and process resources after every test."""

    yield

    from polystore import cleanup_backend_connections

    cleanup_backend_connections(include_process_resources=True)
