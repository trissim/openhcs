"""
Integration tests for the pipeline and TUI components.

Refactored using Systematic Code Refactoring Framework:
- Eliminated magic strings and hardcoded values
- Simplified validation logic with fail-loud approach
- Converted to modern Python patterns with dataclasses
- Reduced verbosity and defensive programming patterns
"""
import json
import os
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from openhcs.constants.constants import VariableComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    GlobalPipelineConfig, MaterializationBackend, MaterializationPathConfig,
    PathPlanningConfig, VFSConfig, ZarrConfig
)
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps import FunctionStep as Step

# Processing functions
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import ashlar_compute_tile_positions_cpu
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
from openhcs.processing.backends.processors.numpy_processor import (
    create_composite, create_projection, stack_percentile_normalize
)

# Test utilities and fixtures
from tests.integration.helpers.fixture_utils import (
    backend_config, base_test_dir, data_type_config, execution_mode,
    microscope_config, plate_dir, test_params, print_thread_activity_report
)


@dataclass(frozen=True)
class TestConstants:
    """Centralized constants for test execution and validation."""

    # Test output indicators
    START_INDICATOR: str = "🔥 STARTING TEST"
    SUCCESS_INDICATOR: str = "🔥 TEST COMPLETED SUCCESSFULLY!"
    VALIDATION_INDICATOR: str = "🔍"
    SUCCESS_CHECK: str = "✅"
    FAILURE_INDICATOR: str = "🔥 VALIDATION FAILED"

    # Configuration values
    DEFAULT_WORKERS: int = 1
    DEFAULT_SUB_DIR: str = "images"
    OUTPUT_SUFFIX: str = "_outputs"
    ZARR_STORE_NAME: str = "images.zarr"

    # Metadata validation
    METADATA_FILENAME: str = "openhcs_metadata.json"
    SUBDIRECTORIES_FIELD: str = "subdirectories"
    MIN_METADATA_ENTRIES: int = 2



    # Required metadata fields
    REQUIRED_FIELDS: List[str] = None

    def __post_init__(self):
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(self, 'REQUIRED_FIELDS',
                          ["image_files", "available_backends", "microscope_handler_name"])


@dataclass
class TestConfig:
    """Configuration for test execution."""
    plate_dir: Path
    backend_config: str
    execution_mode: str
    use_threading: bool = False

    def __post_init__(self):
        self.use_threading = self.execution_mode == "threading"


CONSTANTS = TestConstants()


@pytest.fixture
def test_function_dir(base_test_dir, microscope_config, request):
    """Create test directory for a specific test function."""
    test_name = request.node.originalname or request.node.name.split('[')[0]
    test_dir = base_test_dir / f"{test_name}[{microscope_config['format']}]"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir

def create_test_pipeline() -> Pipeline:
    """Create test pipeline with materialization configuration."""
    cpu_only_mode = os.getenv('OPENHCS_CPU_ONLY', 'false').lower() == 'true'
    position_func = ashlar_compute_tile_positions_cpu if cpu_only_mode else ashlar_compute_tile_positions_gpu

    return Pipeline(
        steps=[
            Step(func=create_composite, variable_components=[VariableComponents.CHANNEL]),
            Step(
                name="Z-Stack Flattening",
                func=(create_projection, {'method': 'max_projection'}),
                variable_components=[VariableComponents.Z_INDEX],
                materialization_config=MaterializationPathConfig()
            ),
            Step(
                name="Image Enhancement Processing",
                func=[(stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5})],
                materialization_config=MaterializationPathConfig()
            ),
            Step(name="Position Computation", func=position_func),
            Step(
                name="Secondary Enhancement",
                func=[(stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5})],
                input_source=InputSource.PIPELINE_START,
            ),
            Step(name="CPU Assembly", func=assemble_stack_cpu)
        ],
        name=f"Multi-Subdirectory Test Pipeline{' (CPU-Only)' if cpu_only_mode else ''}",
    )


def _load_metadata(output_dir: Path) -> Dict:
    """Load and validate metadata file existence."""
    metadata_file = output_dir / CONSTANTS.METADATA_FILENAME
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, 'r') as f:
        return json.load(f)


def _validate_metadata_structure(metadata: Dict) -> List[str]:
    """Validate metadata structure and return subdirectory list."""
    if CONSTANTS.SUBDIRECTORIES_FIELD not in metadata:
        raise ValueError(f"Missing '{CONSTANTS.SUBDIRECTORIES_FIELD}' field in metadata")

    subdirs = list(metadata[CONSTANTS.SUBDIRECTORIES_FIELD].keys())

    if len(subdirs) < CONSTANTS.MIN_METADATA_ENTRIES:
        raise ValueError(
            f"Expected at least {CONSTANTS.MIN_METADATA_ENTRIES} metadata entries, "
            f"found {len(subdirs)}: {subdirs}"
        )

    return subdirs


def _get_materialization_subdir() -> str:
    """Get the actual subdirectory name used by MaterializationPathConfig."""
    return MaterializationPathConfig().sub_dir


def _validate_subdirectory_fields(metadata: Dict) -> None:
    """Validate required fields in each subdirectory metadata."""
    materialization_subdir = _get_materialization_subdir()

    for subdir_name, subdir_metadata in metadata[CONSTANTS.SUBDIRECTORIES_FIELD].items():
        missing_fields = [
            field for field in CONSTANTS.REQUIRED_FIELDS
            if field not in subdir_metadata
        ]
        if missing_fields:
            raise ValueError(f"Subdirectory '{subdir_name}' missing fields: {missing_fields}")

        # Validate image_files (allow empty for materialization subdirectory)
        if not subdir_metadata.get("image_files") and subdir_name != materialization_subdir:
            raise ValueError(f"Subdirectory '{subdir_name}' has empty image_files list")


def validate_separate_materialization(plate_dir: Path) -> None:
    """Validate materialization created multiple metadata entries correctly."""
    output_dir = plate_dir.parent / f"{plate_dir.name}{CONSTANTS.OUTPUT_SUFFIX}"

    if not (output_dir.exists() and output_dir.is_dir()):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    print(f"{CONSTANTS.VALIDATION_INDICATOR} Validating materialization in: {output_dir}")

    metadata = _load_metadata(output_dir)
    subdirs = _validate_metadata_structure(metadata)
    _validate_subdirectory_fields(metadata)

    print(f"{CONSTANTS.VALIDATION_INDICATOR} Subdirectories: {sorted(subdirs)}")
    print(f"{CONSTANTS.SUCCESS_CHECK} Materialization validation successful: {len(subdirs)} entries")



def _create_pipeline_config(test_config: TestConfig) -> GlobalPipelineConfig:
    """Create pipeline configuration for test execution."""
    return GlobalPipelineConfig(
        num_workers=CONSTANTS.DEFAULT_WORKERS,
        path_planning=PathPlanningConfig(
            sub_dir=CONSTANTS.DEFAULT_SUB_DIR,
            output_dir_suffix=CONSTANTS.OUTPUT_SUFFIX
        ),
        vfs=VFSConfig(materialization_backend=MaterializationBackend(test_config.backend_config)),
        zarr=ZarrConfig(
            store_name=CONSTANTS.ZARR_STORE_NAME,
            ome_zarr_metadata=True,
            write_plate_metadata=True
        ),
        use_threading=test_config.use_threading
    )


def _initialize_orchestrator(test_config: TestConfig) -> PipelineOrchestrator:
    """Initialize and configure the pipeline orchestrator."""
    from openhcs.io.base import reset_memory_backend
    reset_memory_backend()

    setup_global_gpu_registry()
    config = _create_pipeline_config(test_config)

    orchestrator = PipelineOrchestrator(test_config.plate_dir, global_config=config)
    orchestrator.initialize()
    return orchestrator


def _execute_pipeline_phases(orchestrator: PipelineOrchestrator, pipeline: Pipeline) -> Dict:
    """Execute compilation and execution phases of the pipeline."""
    from openhcs.constants.constants import GroupBy

    wells = orchestrator.get_component_keys(GroupBy.WELL)
    if not wells:
        raise RuntimeError("No wells found for processing")

    # Compilation phase
    compiled_contexts = orchestrator.compile_pipelines(
        pipeline_definition=pipeline.steps,
        well_filter=wells
    )

    if len(compiled_contexts) != len(wells):
        raise RuntimeError(f"Compilation failed: expected {len(wells)} contexts, got {len(compiled_contexts)}")

    # Execution phase
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline.steps,
        compiled_contexts=compiled_contexts
    )

    if len(results) != len(wells):
        raise RuntimeError(f"Execution failed: expected {len(wells)} results, got {len(results)}")

    # Validate all wells succeeded
    failed_wells = [
        well_id for well_id, result in results.items()
        if result.get('status') != 'success'
    ]
    if failed_wells:
        raise RuntimeError(f"Wells failed execution: {failed_wells}")

    return results


def test_main(plate_dir: Union[Path, str], backend_config: str, data_type_config: Dict, execution_mode: str):
    """Unified test for all combinations of microscope types, backends, data types, and execution modes."""
    test_config = TestConfig(Path(plate_dir), backend_config, execution_mode)

    print(f"{CONSTANTS.START_INDICATOR} with plate: {plate_dir}, backend: {backend_config}, mode: {execution_mode}")

    orchestrator = _initialize_orchestrator(test_config)
    pipeline = create_test_pipeline()

    results = _execute_pipeline_phases(orchestrator, pipeline)
    validate_separate_materialization(test_config.plate_dir)

    print_thread_activity_report()
    print(f"{CONSTANTS.SUCCESS_INDICATOR} ({len(results)} wells processed)")



