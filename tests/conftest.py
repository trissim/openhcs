"""Global pytest configuration for OpenHCS integration tests."""
import os

from openhcs._source_dependencies import ensure_source_checkout_external_paths

ensure_source_checkout_external_paths()

import pytest

# Conditionally import pytest-qt only when not in CPU-only mode
CPU_ONLY_MODE = os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true'
if not CPU_ONLY_MODE:
    pytest_plugins = ["pytestqt"]
else:
    pytest_plugins = []


@pytest.fixture(autouse=True)
def cleanup_backend_connections():
    """
    Automatically clean up backend connections after each test.

    This preserves the napari window for future tests while cleaning up
    ZeroMQ connections and shared memory to prevent test hanging.
    """
    yield  # Run the test

    # Clean up connections after test completes
    try:
        from polystore import cleanup_backend_connections
        cleanup_backend_connections()
    except Exception as e:
        # Don't fail tests due to cleanup issues
        print(f"Warning: Failed to cleanup backend connections: {e}")
