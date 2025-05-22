"""
Pytest configuration file for EZStitcher tests.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# Add the parent directory to sys.path to allow importing from ezstitcher
sys.path.insert(0, str(Path(__file__).parent.parent))

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
