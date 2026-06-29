"""
Shared fixtures and utilities for integration tests.

This module contains fixtures that are used across multiple integration tests.
These fixtures were extracted from test_pipeline_orchestrator.py to avoid circular imports.
"""
import shutil
import pytest
import io
import os
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import numpy as np
from typing import List, Union

from openhcs.core.orchestrator import PipelineOrchestrator
from openhcs.core.config import GlobalPipelineConfig, VFSConfig, MaterializationBackend, ZarrConfig, PathPlanningConfig
# from openhcs.core.config import StitcherConfig, PipelineConfig
from openhcs.core.pipeline import Pipeline
# from openhcs.core.step_base import Step
# from openhcs.core.step_registry import PositionGenerationStep, ImageStitchingStep, NormStep, CompositeStep
# from openhcs.backends.position_generator.ashlar_backend import AshlarPositionGeneratorBackend as IP
from openhcs.tests.generators.generate_synthetic_data import SyntheticMicroscopyGenerator
# from openhcs.core.utils import stack
from polystore.filemanager import FileManager
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
    },
    "OperaPhenix": {
        "format": "OperaPhenix",
        "test_dir_name": "opera_phenix_pipeline",
        "microscope_type": "auto",  # Explicitly specify type
    },
    "OpenHCS": {
        "format": "OpenHCS",
        "test_dir_name": "openhcs_pipeline",
        "microscope_type": "auto",  # Use auto-detection (will detect openhcs_metadata.json)
        "base_format": "ImageXpress"  # Generate ImageXpress data then add OpenHCS metadata
    },
    "OMERO": {
        "format": "OMERO",
        "test_dir_name": "omero_test_plate",  # Will be created dynamically
        "microscope_type": "omero",
        "is_virtual": True,  # Flag to indicate virtual backend
        "requires_connection": True  # Flag to indicate OMERO connection needed
    }
}

# Test parameters
syn_data_params = {
    "grid_size": (3, 3),
    "tile_size": (256, 256),  # Increased from 64x64 to 128x128 for patch size compatibility
    "overlap_percent": 10,
    "wavelengths": 2,  # Changed from 3 to 2 channels
    "cell_size_range": (3, 6),
    "wells": ['A01', 'D02', 'B03', 'B06']
}

# Test-specific parameters that can be customized per microscope format
TEST_PARAMS = {
    "ImageXpress": {
        "default": syn_data_params
        # Add test-specific overrides here if needed
    },
    "OperaPhenix": {
        "default": {
            **syn_data_params,
            # Always skip some files to test missing images feature (autofocus failures)
            # Note: Synthetic data generator creates files with ORIGINAL Opera Phenix format:
            # 1-digit site padding (f2) and 2-digit z-index padding (p01)
            # These will be remapped to 3-digit padding during workspace preparation
            "skip_files": [
                "r01c01f2p01-ch1sk1fk1fl1.tiff",  # A01, site 2, z=1, channel 1
                "r02c02f1p01-ch1sk1fk1fl1.tiff",  # B02, site 1, z=1, channel 1
                "r02c02f1p01-ch2sk1fk1fl1.tiff",  # B02, site 1, z=1, channel 2
            ]
        }
    },
    "OpenHCS": {
        "default": syn_data_params
        # OpenHCS uses same params as base formats
    },
    "OMERO": {
        "default": syn_data_params
        # OMERO uses same params as base formats
    }
}

# Backend configurations for parametrized testing
BACKEND_CONFIGS = ["disk", "zarr"]

# Data type configurations for parametrized testing
DATA_TYPE_CONFIGS = {
    "2d": {"z_stack_levels": 1, "name": "flat_plate"},
    "3d": {"z_stack_levels": 3, "name": "zstack_plate"}  # Changed from 5 to 3 z-planes
}

# Execution mode configurations for parametrized testing
EXECUTION_MODE_CONFIGS = ["threading", "multiprocessing"]

# ZMQ execution mode configurations for parametrized testing
# - direct: No ZMQ, use orchestrator directly
# - zmq: ZMQ with IPC transport (default, no firewall prompts)
# - zmq-tcp: ZMQ with TCP transport (opt-in, triggers firewall prompts)
ZMQ_EXECUTION_MODE_CONFIGS = ["direct", "zmq", "zmq-tcp"]

# Sequential processing configurations for parametrized testing
component1 = "TIMEPOINT"
component2 = "TIMEPOINT,CHANNEL"
component_invalid = "CHANNEL"
SEQUENTIAL_CONFIGS = {
    "none": {
        "name": "none",
        "description": "No sequential processing",
        "sequential_components": [],
        "should_fail": False
    },
    "valid_1_component": {
        "name": "valid_1_component",
        "description": "Sequential with 1 component (TIMEPOINT)",
        "sequential_components": [component1],
        "should_fail": False
    },
    "valid_2_components": {
        "name": "valid_2_components",
        "description": "Sequential with 2 components (TIMEPOINT, CHANNEL)",
        "sequential_components": ["TIMEPOINT", "CHANNEL"],
        "should_fail": False,
        "note": "CHANNEL will be filtered out for create_composite step (which uses CHANNEL in variable_components), but applied to other steps"
    },
    "invalid_overlap": {
        "name": "invalid_overlap",
        "description": "CHANNEL conflicts with create_composite's variable_components - will be filtered out",
        "sequential_components": ["CHANNEL"],
        "should_fail": False,
        "note": "CHANNEL will be filtered out for create_composite step, but applied to other steps"
    },
    "invalid_duplicates": {
        "name": "invalid_duplicates",
        "description": "Invalid: Duplicate TIMEPOINT in sequential",
        "sequential_components": [component1, component1],
        "should_fail": True,
        "expected_error": "sequential_components contains duplicates"
    }
}

@pytest.fixture(scope="module")
def microscope_config(request):
    """Provide microscope configuration - parametrized by pytest_generate_tests."""
    # The actual parameter value is passed by pytest via pytest_generate_tests hook
    return request.param

@pytest.fixture(scope="module")
def backend_config(request):
    """Provide backend configuration - parametrized by pytest_generate_tests."""
    # The actual parameter value is passed by pytest via pytest_generate_tests hook
    return request.param

@pytest.fixture(scope="module")
def data_type_config(request):
    """Provide data type configuration - parametrized by pytest_generate_tests."""
    # The actual parameter value is passed by pytest via pytest_generate_tests hook
    return request.param

@pytest.fixture(scope="module")
def execution_mode(request):
    """
    Execution mode fixture with environment variable management.
    Parametrized by pytest_generate_tests hook.

    This allows tests to run with both execution modes to ensure compatibility.
    Threading mode is useful for debugging, multiprocessing for production testing.
    """
    # Store original value if it exists
    original_value = os.environ.get('OPENHCS_USE_THREADING')

    # Set the execution mode based on parameter
    use_threading = request.param == "threading"
    os.environ['OPENHCS_USE_THREADING'] = 'true' if use_threading else 'false'

    yield request.param

    # Restore original value after test
    if original_value is not None:
        os.environ['OPENHCS_USE_THREADING'] = original_value
    else:
        os.environ.pop('OPENHCS_USE_THREADING', None)

@pytest.fixture(scope="module")
def zmq_execution_mode(request):
    """
    ZMQ execution mode fixture with environment variable management.
    Parametrized by pytest_generate_tests hook.

    This allows tests to run with both direct orchestrator and ZMQ execution modes.
    - 'direct': Execute using orchestrator directly (in-process)
    - 'zmq': Execute using ZMQExecutionClient (subprocess/remote)

    Direct mode is useful for debugging and faster tests.
    ZMQ mode tests the full execution stack including client/server communication.
    """
    # Store original value if it exists
    original_value = os.environ.get('OPENHCS_USE_ZMQ_EXECUTION')

    # Set the ZMQ execution mode based on parameter
    use_zmq = request.param == "zmq"
    os.environ['OPENHCS_USE_ZMQ_EXECUTION'] = 'true' if use_zmq else 'false'

    yield request.param

    # Restore original value after test
    if original_value is not None:
        os.environ['OPENHCS_USE_ZMQ_EXECUTION'] = original_value
    else:
        os.environ.pop('OPENHCS_USE_ZMQ_EXECUTION', None)

@pytest.fixture(scope="module")
def sequential_config(request):
    """Provide sequential processing configuration - parametrized by pytest_generate_tests."""
    # The actual parameter value is passed by pytest via pytest_generate_tests hook
    return request.param

@pytest.fixture(scope="module")
def base_test_dir(microscope_config):
    """Create base test directory for the specific microscope type."""
    # Create the base directory using Path
    base_dir = Path(__file__).parent.parent / "tests_data" / microscope_config["test_dir_name"]

    # Suppress stdout and stderr to avoid microscopy data generator output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Ensure the directory exists
        if base_dir.exists():
            # Use ignore_errors=True to handle race conditions with multiprocessing workers
            # that might still have file handles open during cleanup
            shutil.rmtree(base_dir, ignore_errors=True)

        # Create the directory
        base_dir.mkdir(parents=True, exist_ok=True)

    yield base_dir

    # Uncomment to clean up after tests
    # if base_dir.exists():
    #     shutil.rmtree(base_dir)

@pytest.fixture
def test_function_dir(base_test_dir, microscope_config, request):
    """Create test directory for a specific test function."""
    pytest.skip("Smell-loop gated - do not run until certified")

    # Get the test function name without the parameter
    test_name = request.node.originalname or request.node.name.split('[')[0]
    # Create a directory for this specific test function
    test_dir = base_test_dir / f"{test_name}[{microscope_config['format']}]"

    # Ensure the directory exists
    test_dir.mkdir(parents=True, exist_ok=True)

    yield test_dir

@pytest.fixture(scope="module")
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

    # For OpenHCS format, generate base format data first, then add metadata
    generator_format = microscope_config.get("base_format", microscope_config["format"])

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
            format=generator_format,  # Use base format for generation
            # For OpenHCS, include all filename components
            include_all_components=(microscope_config["format"] == "OpenHCS"),
            # Skip files if specified (for testing missing image handling)
            skip_files=test_params.get("skip_files", None)
        )
        generator.generate_dataset()

        # If OpenHCS format, generate metadata file
        if microscope_config["format"] == "OpenHCS":
            # Determine subdirectory based on base format
            if generator_format == "ImageXpress":
                sub_dir = "TimePoint_1"
            else:  # OperaPhenix
                sub_dir = "Images"

            generator.generate_openhcs_metadata(sub_dir=sub_dir, pixel_size=0.65)

    # Return the plate directory
    return plate_dir

@pytest.fixture
def plate_dir(test_function_dir, microscope_config, test_params, data_type_config):
    """
    Create synthetic plate data for the specified microscope type and data type.

    For disk-based microscopes: Returns Path to plate directory
    For OMERO: Returns plate_id (int) after uploading to OMERO
    """
    import shutil

    # Handle OMERO specially
    if microscope_config.get("is_virtual") and microscope_config.get("requires_connection"):
        from openhcs.runtime.omero_instance_manager import OMEROInstanceManager
        import tempfile
        from pathlib import Path

        # Connect to OMERO (automatically starts docker-compose if needed)
        omero_manager = OMEROInstanceManager()
        print("🔄 Connecting to OMERO (will auto-start if needed)...")
        if not omero_manager.connect(timeout=120):  # Increased timeout for docker startup
            pytest.skip("OMERO server not available and could not be started automatically. Check Docker installation.")

        # Generate synthetic data using the SAME generator as disk-based tests
        # This ensures identical test data across all backends
        with tempfile.TemporaryDirectory() as tmpdir:
            # OMERO uses ImageXpress-compatible filenames, so generate ImageXpress format
            generator_format = "ImageXpress" if microscope_config["format"] == "OMERO" else microscope_config["format"]

            # Use the same SyntheticMicroscopyGenerator with same parameters
            generator = SyntheticMicroscopyGenerator(
                output_dir=tmpdir,
                grid_size=test_params.get("grid_size", (3, 3)),
                tile_size=test_params.get("tile_size", (128, 128)),
                overlap_percent=test_params.get("overlap_percent", 10),
                wavelengths=test_params.get("wavelengths", 2),
                z_stack_levels=data_type_config["z_stack_levels"],
                cell_size_range=test_params.get("cell_size_range", (5, 10)),
                wells=test_params.get("wells", ['A01']),
                format=generator_format,  # Use ImageXpress format for OMERO
            )

            # Don't suppress output so we can see what's happening
            generator.generate_dataset()

            # Upload to OMERO with grid dimensions
            from tests.integration.helpers.omero_utils import upload_plate_to_omero
            plate_id = upload_plate_to_omero(
                omero_manager.conn,
                tmpdir,
                plate_name=f"Test_{data_type_config['name']}",
                microscope_format=generator_format,  # Use ImageXpress format
                grid_dimensions=test_params.get("grid_size", (3, 3))  # Pass grid dimensions
            )

        yield plate_id

        # Don't cleanup - leave plates in OMERO for inspection
        omero_manager.close()

    # Standard disk-based microscopes
    else:
        plate_path = create_synthetic_plate_data(
            test_function_dir=test_function_dir,
            microscope_config=microscope_config,
            test_params=test_params,
            plate_name=data_type_config["name"],
            z_stack_levels=data_type_config["z_stack_levels"]
        )

        # Clean up workspace directory if it exists from previous runs
        # This ensures _prepare_workspace() is called, which creates missing images for Opera Phenix
        workspace_path = plate_path / "workspace"
        if workspace_path.exists():
            shutil.rmtree(workspace_path, ignore_errors=True)

        yield plate_path

# Keep legacy fixtures for backward compatibility
@pytest.fixture
def flat_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic flat plate data for the specified microscope type as a VirtualPath."""
    plate_path = create_synthetic_plate_data(
        test_function_dir=test_function_dir,
        microscope_config=microscope_config,
        test_params=test_params,
        plate_name="flat_plate",
        z_stack_levels=1  # Flat plate has only 1 Z-level
    )
    yield plate_path

@pytest.fixture
def zstack_plate_dir(test_function_dir, microscope_config, test_params):
    """Create synthetic Z-stack plate data for the specified microscope type as a VirtualPath."""
    plate_path = create_synthetic_plate_data(
        test_function_dir=test_function_dir,
        microscope_config=microscope_config,
        test_params=test_params,
        plate_name="zstack_plate",
        z_stack_levels=5  # Z-stack plate has 5 Z-levels
    )
    yield plate_path

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

@pytest.fixture
def debug_global_config(execution_mode, backend_config):
    """Create a GlobalPipelineConfig optimized for debugging.

    Following OpenHCS modular design principles:
    - Always create complete configuration with all sections
    - Let backend selection determine what gets used
    - No conditional configuration creation
    """
    use_threading = execution_mode == "threading"

    # Always create complete configuration - let the system use what it needs
    return GlobalPipelineConfig(
        num_workers=2,  # Changed from 1 to 2 workers
        path_planning=PathPlanningConfig(
            sub_dir="images",  # Default subdirectory for processed data
            output_dir_suffix="_outputs"  # Suffix for output directories
        ),
        vfs=VFSConfig(materialization_backend=MaterializationBackend(backend_config)),
        zarr=ZarrConfig(),
        use_threading=use_threading
    )

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
    Find all image files in a directory matching a pattern using breadth-first traversal.

    Args:
        directory: Directory to search
        pattern: Glob pattern to match (default: "*" for all files)
        recursive: Whether to search recursively (default: True)

    Returns:
        List of Path objects for image files sorted by depth (shallower first)
    """
    from collections import deque
    import fnmatch

    directory = Path(directory)
    image_files = []

    if recursive:
        # Use breadth-first traversal for recursive search
        dirs_to_search = deque([(directory, 0)])  # (path, depth)

        while dirs_to_search:
            current_dir, depth = dirs_to_search.popleft()

            try:
                for entry in current_dir.iterdir():
                    if entry.is_file():
                        # Check if file matches pattern and has image extension
                        if fnmatch.fnmatch(entry.name, pattern):
                            if entry.suffix.lower() in [ext.lower() for ext in DEFAULT_IMAGE_EXTENSIONS]:
                                image_files.append((entry, depth))
                    elif entry.is_dir():
                        # Add subdirectory to queue for later processing
                        dirs_to_search.append((entry, depth + 1))
            except (PermissionError, OSError):
                # Skip directories we can't read
                continue

        # Sort by depth first, then by path for consistent ordering
        image_files.sort(key=lambda x: (x[1], str(x[0])))

        # Return just the paths
        return [file_path for file_path, _ in image_files]
    else:
        # Non-recursive search - use simple glob
        for ext in DEFAULT_IMAGE_EXTENSIONS:
            pattern_with_ext = f"{pattern}{ext}"
            image_files.extend(list(directory.glob(pattern_with_ext)))

        return sorted(image_files)
