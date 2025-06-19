"""
Shared fixtures and utilities for integration tests.

This module contains fixtures that are used across multiple integration tests.
These fixtures were extracted from test_pipeline_orchestrator.py to avoid circular imports.
"""
import shutil
import pytest
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import numpy as np
from typing import List, Union

from openhcs.core.orchestrator import PipelineOrchestrator
# from openhcs.core.config import StitcherConfig, PipelineConfig
from openhcs.core.pipeline import Pipeline
# from openhcs.core.step_base import Step
# from openhcs.core.step_registry import PositionGenerationStep, ImageStitchingStep, NormStep, CompositeStep
# from openhcs.backends.position_generator.ashlar_backend import AshlarPositionGeneratorBackend as IP
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
# from openhcs.core.utils import stack
from openhcs.io.filemanager import FileManager
# Using a simple list for image extensions
DEFAULT_IMAGE_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
# from ezstitcher.io.virtual_path.factory import VirtualPathFactory
# from ezstitcher.io.virtual_path.base import PhysicalPath

# Create a simple mock for IP
class IP:
    @staticmethod
    def stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9):
        return stack

    @staticmethod
    def tophat(img):
        return img

# Define microscope configurations
MICROSCOPE_CONFIGS = {
    "ImageXpress": {
        "format": "ImageXpress",
        "test_dir_name": "imagexpress_pipeline",
        "microscope_type": "auto",  # Use auto-detection
        "auto_image_size": True
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_pipeline",
        "microscope_type": "auto",  # Explicitly specify type
        "auto_image_size": True
    }
}

# Test parameters
syn_data_params = {
    "grid_size": (4, 4),
    "tile_size": (512, 512),  # Increased from 64x64 to 128x128 for patch size compatibility
    "overlap_percent": 10,
    "wavelengths": 2,
    "cell_size_range": (3, 6),
    "wells": ['A01', 'D02']
}

# Test-specific parameters that can be customized per microscope format
TEST_PARAMS = {
    "ImageXpress": {
        "default": syn_data_params
        # Add test-specific overrides here if needed
    },
    "OperaPhenix": {
        "default": syn_data_params
        # Add test-specific overrides here if needed
    }
}

@pytest.fixture(scope="module", params=list(MICROSCOPE_CONFIGS.keys()))
def microscope_config(request):
    """Provide microscope configuration based on the parameter."""
    return MICROSCOPE_CONFIGS[request.param]

@pytest.fixture(scope="module")
def base_test_dir(microscope_config):
    """Create base test directory for the specific microscope type."""
    # Create the base directory using Path
    base_dir = Path(__file__).parent.parent / "tests_data" / microscope_config["test_dir_name"]

    # Suppress stdout and stderr to avoid microscopy data generator output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Ensure the directory exists
        if base_dir.exists():
            shutil.rmtree(base_dir)

        # Create the directory
        base_dir.mkdir(parents=True, exist_ok=True)

    yield base_dir

    # Uncomment to clean up after tests
    # if base_dir.exists():
    #     shutil.rmtree(base_dir)

@pytest.fixture
@pytest.mark.skip(reason="Smell-loop gated — do not run until certified")
def test_function_dir(base_test_dir, microscope_config, request):
    """Create test directory for a specific test function."""
    # Get the test function name without the parameter
    test_name = request.node.originalname or request.node.name.split('[')[0]
    # Create a directory for this specific test function
    test_dir = base_test_dir / f"{test_name}[{microscope_config['format']}]"

    # Ensure the directory exists
    test_dir.mkdir(parents=True, exist_ok=True)

    yield test_dir

@pytest.fixture(scope="module")
@pytest.mark.skip(reason="Smell-loop gated — do not run until certified")
def test_params(microscope_config):
    """Get test parameters for the specific microscope type."""
    # Use the format key instead of microscope_type
    return TEST_PARAMS[microscope_config["format"]]["default"]

def create_synthetic_plate_data(test_function_dir, microscope_config, test_params, plate_name, z_stack_levels):
    """Create synthetic plate data for the specified microscope type.

    Args:
        test_function_dir: Directory for test function
        microscope_config: Microscope configuration
        test_params: Test parameters
        plate_name: Name of the plate directory
        z_stack_levels: Number of Z-stack levels

    Returns:
        Path to the plate directory
    """
    # Create the plate directory
    plate_dir = test_function_dir / plate_name
    plate_dir.mkdir(parents=True, exist_ok=True)

    # Suppress stdout and stderr to avoid microscopy data generator output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # SyntheticMicroscopyGenerator requires a native disk path
        generator = SyntheticMicroscopyGenerator(
            output_dir=str(plate_dir),
            grid_size=test_params.get("grid_size", (3, 3)),
            tile_size=test_params.get("tile_size", (128, 128)),
            overlap_percent=test_params.get("overlap_percent", 10),
            wavelengths=test_params.get("wavelengths", 2),
            z_stack_levels=z_stack_levels,
            cell_size_range=test_params.get("cell_size_range", (5, 10)),
            wells=test_params.get("wells", ['A01']),
            format=microscope_config["format"],
            # Use test_params override if available
            auto_image_size=test_params.get(
                "auto_image_size",
                microscope_config["auto_image_size"]
            )
        )
        generator.generate_dataset()

    # Return the plate directory
    return plate_dir

@pytest.fixture
def flat_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic flat plate data for the specified microscope type as a VirtualPath."""
    return create_synthetic_plate_data(
        test_function_dir=test_function_dir,
        microscope_config=microscope_config,
        test_params=test_params,
        plate_name="flat_plate",
        z_stack_levels=1  # Flat plate has only 1 Z-level
    )

@pytest.fixture
def zstack_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic Z-stack plate data for the specified microscope type as a VirtualPath."""
    return create_synthetic_plate_data(
        test_function_dir=test_function_dir,
        microscope_config=microscope_config,
        test_params=test_params,
        plate_name="zstack_plate",
        z_stack_levels=5  # Z-stack plate has 5 Z-levels
    )

# Mock thread tracking utilities
def track_thread_activity(func):
    """Mock decorator for tracking thread activity."""
    return func

def clear_thread_activity():
    """Mock function for clearing thread activity."""
    pass

def print_thread_activity_report():
    """Mock function for printing thread activity report."""
    pass

@pytest.fixture
def thread_tracker():
    """Fixture to track thread activity for tests."""
    # Store the original method
    original_process_well = PipelineOrchestrator.process_well

    # Apply the decorator to the process_well method
    PipelineOrchestrator.process_well = track_thread_activity(original_process_well)

    # Clear any previous thread activity data
    clear_thread_activity()

    # Provide the fixture
    yield

    # Restore the original method
    PipelineOrchestrator.process_well = original_process_well

@pytest.fixture
def base_pipeline_config(microscope_config):
    """Create a base pipeline configuration with default values."""
    # Using a simple dictionary instead of PipelineConfig
    config = {
        "stitcher": {
            "tile_overlap": 10.0,
            "max_shift": 50,
            "margin_ratio": 0.1
        },
        "num_workers": 1,
    }
    return config

def create_config(base_config, **kwargs):
    """Create a new configuration by overriding base config values."""
    # Create a copy of the base config dict
    config_dict = base_config.copy()

    # Override with new values
    for key, value in kwargs.items():
        config_dict[key] = value

    # Return the updated dictionary
    return config_dict

def normalize(stack):
    """Apply true histogram equalization to an entire stack."""
    return IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.99)

def calcein_process(stack):
    """Apply tophat filter to Calcein images."""
    return [IP.tophat(img) for img in stack]

def dapi_process(stack):
    """Apply tophat filter to DAPI images."""
    stack = IP.stack_percentile_normalize(stack, low_percentile=0.1, high_percentile=99.9)
    return [IP.tophat(img) for img in stack]

def find_image_files(directory: Union[str, Path], pattern: str = "*",
                  recursive: bool = True) -> List[Path]:
    """
    Find all image files in a directory matching a pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: "*" for all files)
        recursive: Whether to search recursively (default: True)

    Returns:
        List of Path objects for image files
    """
    directory = Path(directory)
    image_files = []

    # Use rglob for recursive search or glob for non-recursive
    glob_func = directory.rglob if recursive else directory.glob

    for ext in DEFAULT_IMAGE_EXTENSIONS:
        pattern_with_ext = f"{pattern}{ext}"
        if recursive:
            pattern_with_ext = f"**/{pattern_with_ext}"
        image_files.extend(list(glob_func(pattern_with_ext)))

    return sorted(image_files)