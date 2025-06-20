"""
Strategic Test Configuration - DNA-Guided Test Setup.

Provides comprehensive test configuration, fixtures, and utilities
for the strategic test suite based on mathematical analysis.

🔬 Configuration Strategy:
- Mathematical test framework integration
- Comprehensive mock factories
- DNA-guided test prioritization
- Production-grade test utilities
"""
import os
import sys
import pytest
import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, AsyncMock

# Add the parent directory to sys.path to allow importing from openhcs
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure test logging only if no handlers exist (respect main app logging)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# Import test framework
try:
    from tests.test_framework.mathematical_test_framework import test_framework
except ImportError:
    test_framework = None

# Define common fixtures that can be used across all tests

@pytest.fixture(scope="session")
def tests_root_dir():
    """Return the root directory of the tests."""
    return Path(__file__).parent

@pytest.fixture(scope="session")
def tests_data_dir():
    """Return the directory for test data."""
    data_dir = Path(__file__).parent / "tests_data"
    data_dir.mkdir(exist_ok=True, parents=True)
    return data_dir

def create_mock_filemanager_with_backend(backend_name="MockBackend"):
    """
    Create a mock FileManager with a properly configured backend attribute.

    Args:
        backend_name: Name to use for the backend's type.__name__

    Returns:
        MagicMock: Configured with FileManager spec and backend attribute
    """
    # Create the mock backend
    mock_backend = MagicMock()
    # Configure the backend's type name
    type(mock_backend).__name__ = backend_name

    # Create the mock file manager
    mock_fm = MagicMock()
    # Set the backend attribute
    mock_fm.backend = mock_backend
    # Set common mock behaviors
    mock_fm.ensure_directory.side_effect = lambda p: Path(p)
    mock_fm.find_image_directory.side_effect = lambda p, extensions=None: Path(p) / "Images"
    mock_fm.list_image_files.return_value = [Path("dummy/plate/Images/Well_A1_Site_1.tif")]
    mock_fm.root_dir = Path("/mock/root")

    return mock_fm


# Strategic Test Configuration
def pytest_configure(config):
    """Configure pytest with strategic test markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "high_priority: High priority tests based on DNA analysis"
    )
    config.addinivalue_line(
        "markers", "medium_priority: Medium priority tests"
    )
    config.addinivalue_line(
        "markers", "low_priority: Low priority tests"
    )
    config.addinivalue_line(
        "markers", "async_test: Asynchronous test requiring event loop"
    )
    config.addinivalue_line(
        "markers", "production: Production-grade tests with strict requirements"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to prioritize based on DNA analysis."""
    # Sort tests by priority markers
    priority_order = {
        'high_priority': 1,
        'medium_priority': 2,
        'low_priority': 3
    }

    def get_priority(item):
        for marker_name, priority in priority_order.items():
            if item.get_closest_marker(marker_name):
                return priority
        return 4  # Default priority for unmarked tests

    items.sort(key=get_priority)


def pytest_ignore_collect(collection_path, config):
    """Ignore collection of specific classes that are not test classes."""
    # This helps avoid warnings about dataclasses in the test framework
    return False


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_tui_state():
    """Create a comprehensive mock TUIState."""
    if test_framework:
        return test_framework.create_mock_state()
    else:
        # Fallback mock
        mock_state = Mock()
        mock_state.selected_plate = None
        mock_state.active_orchestrator = None
        mock_state.is_compiled = False
        mock_state.is_running = False
        mock_state.compiled_contexts = {}
        mock_state.notify = AsyncMock()
        return mock_state


@pytest.fixture
def mock_processing_context():
    """Create a comprehensive mock ProcessingContext."""
    if test_framework:
        return test_framework.create_mock_context()
    else:
        # Fallback mock
        mock_context = Mock()
        mock_context.filemanager = Mock()
        mock_context.common_output_directory = "/tmp/test_output"
        return mock_context


@pytest.fixture
def mock_orchestrator():
    """Create a single mock PipelineOrchestrator."""
    if test_framework:
        return test_framework.create_mock_orchestrator("test_plate")
    else:
        # Fallback mock
        mock_orchestrator = Mock()
        mock_orchestrator.plate_id = "test_plate"
        mock_orchestrator.pipeline_definition = []
        mock_orchestrator.initialize_plate = AsyncMock()
        return mock_orchestrator


@pytest.fixture(autouse=True)
def reset_test_framework():
    """Reset test framework state before each test."""
    if test_framework:
        test_framework.test_results.clear()
        test_framework.coverage_matrix.clear()
    yield
