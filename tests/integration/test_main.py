"""
Integration tests for the pipeline and TUI components.

Refactored using Systematic Code Refactoring Framework:
- Eliminated magic strings and hardcoded values
- Simplified validation logic with fail-loud approach
- Converted to modern Python patterns with dataclasses
- Reduced verbosity and defensive programming patterns
"""

from openhcs.core.pipeline_document import PipelineDocumentAuthority

import json
import os
import pytest
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

from openhcs.constants.constants import (
    GroupBy,
    VariableComponents,
    SequentialComponents,
)
from openhcs.constants.input_source import InputSource
from openhcs.core.config import (
    GlobalPipelineConfig,
    MaterializationBackend,
    PathPlanningConfig,
    StepWellFilterConfig,
    VFSConfig,
    Backend,
    ZarrConfig,
    NapariVariableSizeHandling,
)
from openhcs.runtime.zmq_execution_client import OpenHCSExecutionSubmission
from openhcs.core.config import (
    LazyStepMaterializationConfig,
    LazyNapariStreamingConfig,
    LazyFijiStreamingConfig,
    LazyStepWellFilterConfig,
    LazyPathPlanningConfig,
    LazyDtypeConfig,
    LazyProcessingConfig,
    LazySequentialProcessingConfig,
    LazyVFSConfig,
)
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.function_step_transport import FunctionStepTransportAuthority
from openhcs.core.steps import FunctionStep as Step

# Processing functions
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.pos_gen.ashlar_main_cpu import (
    ashlar_compute_tile_positions_cpu,
)
from openhcs.processing.backends.processors.numpy_processor import (
    NumpyStackProjectionMethod,
    create_composite,
    create_projection,
    stack_percentile_normalize,
)
from openhcs.processing.backends.analysis.cell_counting_cpu import (
    count_cells_single_channel,
    DetectionMethod,
)
from arraybridge.decorators import DtypeConversion

# Test utilities and fixtures
from tests.integration.helpers.fixture_utils import (
    backend_config,
    base_test_dir,
    data_type_config,
    execution_mode,
    microscope_config,
    plate_dir,
    test_params,
    print_thread_activity_report,
    zmq_execution_mode,
)

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig


@dataclass(frozen=True)
class TestConstants:
    """Centralized constants for test execution and validation."""

    # Test output indicators
    START_INDICATOR: str = "[TEST] STARTING TEST"
    SUCCESS_INDICATOR: str = "[TEST] TEST COMPLETED SUCCESSFULLY!"
    VALIDATION_INDICATOR: str = "[DEBUG]"
    SUCCESS_CHECK: str = "[OK]"
    FAILURE_INDICATOR: str = "[TEST] VALIDATION FAILED"

    # Configuration values
    DEFAULT_WORKERS: int = 1
    DEFAULT_SUB_DIR: str = "images"
    OUTPUT_SUFFIX: str = "_outputs"
    STEP_WELL_FILTER_TEST: int = 4
    PIPELINE_STEP_WELL_FILTER_TEST: int = 2
    GLOBAL_STEP_WELL_FILTER_TEST: int = 3

    # Metadata validation
    METADATA_FILENAME: str = "openhcs_metadata.json"
    SUBDIRECTORIES_FIELD: str = "subdirectories"
    MIN_METADATA_ENTRIES: int = 1

    # Required metadata fields
    REQUIRED_FIELDS: List[str] = None

    def __post_init__(self):
        # Use object.__setattr__ for frozen dataclass
        object.__setattr__(
            self,
            "REQUIRED_FIELDS",
            ["image_files", "available_backends", "microscope_handler_name"],
        )


@dataclass
class TestConfig:
    """Configuration for test execution."""

    plate_dir: Union[Path, int]  # Path for disk-based, int (plate_id) for OMERO
    backend_config: str
    execution_mode: str
    microscope_config: Dict
    use_threading: bool = False

    def __post_init__(self):
        self.use_threading = self.execution_mode == "threading"

    @property
    def is_omero(self) -> bool:
        """Check if this is an OMERO test (plate_dir is int)."""
        return isinstance(self.plate_dir, int)

    @property
    def microscope_type(self) -> str:
        """Get microscope type from config."""
        return self.microscope_config.get("microscope_type", "auto")


CONSTANTS = TestConstants()


def _gpu_available() -> bool:
    """Detect if a usable GPU is available without importing heavy libs eagerly.
    Priority:
    - Respect CUDA_VISIBLE_DEVICES empty/-1 as no GPU
    - torch.cuda.is_available()
    - cupy device count
    """
    # Respect visibility first
    cuda_vis = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_vis is not None and (cuda_vis == "" or cuda_vis == "-1"):
        return False
    # Torch
    try:
        import torch  # type: ignore

        if hasattr(torch, "cuda") and torch.cuda.is_available():
            return True
    except Exception:
        pass
    # CuPy
    try:
        import cupy  # type: ignore

        try:
            return cupy.cuda.runtime.getDeviceCount() > 0
        except Exception:
            return False
    except Exception:
        pass
    return False


# Auto-enable CPU-only if no GPU is available (and not explicitly forced on)
if os.getenv("OPENHCS_CPU_ONLY", "false").lower() != "true" and not _gpu_available():
    os.environ["OPENHCS_CPU_ONLY"] = "true"


def _headless_mode() -> bool:
    """Detect headless environment where GUI/visualization should be suppressed.

    CPU-only mode does NOT imply headless - you can run CPU mode with napari.
    Only CI or explicit OPENHCS_HEADLESS flag triggers headless mode.
    """
    try:
        if os.getenv("CI", "").lower() == "true":
            return True
        if os.getenv("OPENHCS_HEADLESS", "").lower() == "true":
            return True
    except Exception:
        pass
    return False


def _napari_enabled() -> bool:
    """Check if Napari streaming should be enabled."""
    if _headless_mode():
        return False
    return os.getenv("OPENHCS_DISABLE_NAPARI", "false").lower() != "true"


def _fiji_enabled() -> bool:
    """Check if Fiji streaming should be enabled."""
    if _headless_mode():
        return False
    return os.getenv("OPENHCS_DISABLE_FIJI", "false").lower() != "true"


@pytest.fixture
def test_function_dir(base_test_dir, microscope_config, request):
    """Create test directory for a specific test function."""
    test_name = request.node.originalname or request.node.name.split("[")[0]
    test_dir = base_test_dir / f"{test_name}[{microscope_config['format']}]"
    test_dir.mkdir(parents=True, exist_ok=True)
    yield test_dir


def create_test_pipeline(
    enable_napari: bool = False,
    enable_fiji: bool = False,
    sequential_config: dict = None,
) -> list[Step]:
    """Create test pipeline with materialization configuration.

    Args:
        enable_napari: Enable Napari streaming
        enable_fiji: Enable Fiji streaming
        sequential_config: Sequential processing configuration dict with keys:
            - sequential_components: List of component names to process sequentially
            - should_fail: Whether this config should fail validation
            - expected_error: Expected error message substring (if should_fail=True)

    Note: sequential_config is NOT used in this function - it should be set in PipelineConfig instead.
    This parameter is kept for backward compatibility but is ignored.
    """
    cpu_only_mode = os.getenv("OPENHCS_CPU_ONLY", "false").lower() == "true"
    if cpu_only_mode:
        position_func = ashlar_compute_tile_positions_cpu
    else:
        try:
            from openhcs.processing.backends.pos_gen.ashlar_main_gpu import (
                ashlar_compute_tile_positions_gpu,
            )

            position_func = ashlar_compute_tile_positions_gpu
        except Exception:
            # Fallback cleanly to CPU if GPU path is unavailable
            os.environ["OPENHCS_CPU_ONLY"] = "true"
            position_func = ashlar_compute_tile_positions_cpu

    return [
        Step(
            name="Image Enhancement Processing",
            func=[
                (
                    stack_percentile_normalize,
                    {"low_percentile": 0.5, "high_percentile": 99.5},
                )
            ],
            step_well_filter_config=LazyStepWellFilterConfig(
                well_filter=CONSTANTS.STEP_WELL_FILTER_TEST
            ),
            step_materialization_config=LazyStepMaterializationConfig(),
            napari_streaming_config=LazyNapariStreamingConfig(
                port=5555, enabled=enable_napari
            ),
            fiji_streaming_config=LazyFijiStreamingConfig(enabled=enable_fiji),
        ),
        Step(
            func=create_composite,
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.CHANNEL],
                group_by=GroupBy.NONE,
            ),
            napari_streaming_config=LazyNapariStreamingConfig(
                port=5557, enabled=enable_napari
            ),
            fiji_streaming_config=LazyFijiStreamingConfig(
                port=5556, enabled=enable_fiji
            ),
        ),
        Step(
            name="Z-Stack Flattening",
            func=(create_projection, {"method": NumpyStackProjectionMethod.MAX}),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.Z_INDEX]
            ),
            step_materialization_config=LazyStepMaterializationConfig(),
        ),
        Step(name="Position Computation", func=position_func),
        Step(
            name="Secondary Enhancement",
            func=[
                (
                    stack_percentile_normalize,
                    {"low_percentile": 0.5, "high_percentile": 99.5},
                )
            ],
            processing_config=LazyProcessingConfig(
                input_source=InputSource.PIPELINE_START
            ),
        ),
        Step(name="CPU Assembly", func=assemble_stack_cpu),
        Step(
            name="Z-Stack Flattening",
            func=(create_projection, {"method": NumpyStackProjectionMethod.MAX}),
            processing_config=LazyProcessingConfig(
                variable_components=[VariableComponents.Z_INDEX]
            ),
        ),
        Step(
            name="Cell Counting",
            func=(
                {
                    "1": (
                        count_cells_single_channel,
                        {
                            "min_cell_area": 40,
                            "max_cell_area": 200,
                            "enable_preprocessing": False,
                            "detection_method": DetectionMethod.WATERSHED,
                            "dtype_config": LazyDtypeConfig(
                                default_dtype_conversion=DtypeConversion.UINT8
                            ),
                        },
                    ),
                    "2": (
                        count_cells_single_channel,
                        {
                            "min_cell_area": 40,
                            "max_cell_area": 200,
                            "enable_preprocessing": False,
                            "detection_method": DetectionMethod.WATERSHED,
                            "dtype_config": LazyDtypeConfig(
                                default_dtype_conversion=DtypeConversion.UINT8
                            ),
                        },
                    ),
                }
            ),
            napari_streaming_config=LazyNapariStreamingConfig(
                port=5559,
                variable_size_handling=NapariVariableSizeHandling.PAD_TO_MAX,
                enabled=enable_napari,
            ),
            fiji_streaming_config=LazyFijiStreamingConfig(enabled=enable_fiji),
        ),
    ]


def _load_metadata(output_dir: Path) -> Dict:
    """Load and validate metadata file existence."""
    metadata_file = output_dir / CONSTANTS.METADATA_FILENAME
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file, "r") as f:
        return json.load(f)


def _validate_metadata_structure(metadata: Dict) -> List[str]:
    """Validate metadata structure and return subdirectory list."""
    if CONSTANTS.SUBDIRECTORIES_FIELD not in metadata:
        raise ValueError(
            f"Missing '{CONSTANTS.SUBDIRECTORIES_FIELD}' field in metadata"
        )

    subdirs = list(metadata[CONSTANTS.SUBDIRECTORIES_FIELD].keys())

    if len(subdirs) < CONSTANTS.MIN_METADATA_ENTRIES:
        raise ValueError(
            f"Expected at least {CONSTANTS.MIN_METADATA_ENTRIES} metadata entries, "
            f"found {len(subdirs)}: {subdirs}"
        )

    return subdirs


def _get_materialization_subdir() -> str:
    """Get the actual subdirectory name used by LazyStepMaterializationConfig."""
    from openhcs.core.config import StepMaterializationConfig

    return StepMaterializationConfig().sub_dir


def _validate_subdirectory_fields(metadata: Dict) -> None:
    """Validate required fields in each subdirectory metadata."""
    materialization_subdir = _get_materialization_subdir()

    for subdir_name, subdir_metadata in metadata[
        CONSTANTS.SUBDIRECTORIES_FIELD
    ].items():
        missing_fields = [
            field for field in CONSTANTS.REQUIRED_FIELDS if field not in subdir_metadata
        ]
        if missing_fields:
            raise ValueError(
                f"Subdirectory '{subdir_name}' missing fields: {missing_fields}"
            )

        # Validate image_files (allow empty for materialization subdirectory)
        if (
            not subdir_metadata.get("image_files")
            and subdir_name != materialization_subdir
        ):
            raise ValueError(f"Subdirectory '{subdir_name}' has empty image_files list")


def validate_separate_materialization(plate_dir: Path) -> None:
    """Validate materialization created multiple metadata entries correctly."""
    output_dir = plate_dir.parent / f"{plate_dir.name}{CONSTANTS.OUTPUT_SUFFIX}"

    if not (output_dir.exists() and output_dir.is_dir()):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    print(
        f"{CONSTANTS.VALIDATION_INDICATOR} Validating materialization in: {output_dir}"
    )

    metadata = _load_metadata(output_dir)
    subdirs = _validate_metadata_structure(metadata)
    _validate_subdirectory_fields(metadata)

    print(f"{CONSTANTS.VALIDATION_INDICATOR} Subdirectories: {sorted(subdirs)}")
    print(
        f"{CONSTANTS.SUCCESS_CHECK} Materialization validation successful: {len(subdirs)} entries"
    )


def _create_pipeline_config(test_config: TestConfig) -> GlobalPipelineConfig:
    """Create pipeline configuration for test execution."""
    from openhcs.constants import Microscope

    # Set microscope type from config
    microscope = (
        Microscope[test_config.microscope_type.upper()]
        if test_config.microscope_type != "auto"
        else Microscope.AUTO
    )

    # For OMERO tests, use omero_local backend for materialization
    # For other tests, use the configured backend
    if test_config.is_omero:
        materialization_backend = MaterializationBackend("omero_local")
    else:
        materialization_backend = MaterializationBackend(test_config.backend_config)

    return GlobalPipelineConfig(
        num_workers=CONSTANTS.DEFAULT_WORKERS,
        microscope=microscope,
        path_planning_config=PathPlanningConfig(
            sub_dir=CONSTANTS.DEFAULT_SUB_DIR, output_dir_suffix=CONSTANTS.OUTPUT_SUFFIX
        ),
        vfs_config=VFSConfig(materialization_backend=materialization_backend),
        zarr_config=ZarrConfig(),
        step_well_filter_config=StepWellFilterConfig(
            well_filter=CONSTANTS.GLOBAL_STEP_WELL_FILTER_TEST
        ),
        use_threading=test_config.use_threading,
    )


def _initialize_orchestrator(
    test_config: TestConfig, sequential_config=None
) -> PipelineOrchestrator:
    """Initialize and configure the pipeline orchestrator."""
    from polystore.base import reset_memory_backend

    reset_memory_backend()
    global_config = _create_pipeline_config(test_config)
    setup_global_gpu_registry(global_config=global_config)

    # Set up global context for orchestrator - legitimate test setup
    ensure_global_config_context(GlobalPipelineConfig, global_config)

    # For OMERO: Register OMERO backend with connection
    omero_manager = None
    if test_config.is_omero:
        # Import OMERO parsers BEFORE creating backend to ensure registration
        # This is required because OMEROLocalBackend accesses FilenameParser.__registry__
        # which is a LazyDiscoveryDict that only populates when first accessed
        from openhcs.runtime.omero_instance_manager import OMEROInstanceManager
        from openhcs.microscopes import omero  # noqa: F401 - Import OMERO parsers to register them
        from polystore.omero_local import OMEROLocalBackend
        from polystore.base import storage_registry

        # Connect to OMERO
        omero_manager = OMEROInstanceManager()
        if not omero_manager.connect(timeout=60):
            pytest.skip("OMERO server not available - skipping OMERO tests")

        # Register OMERO backend with connection in global storage registry
        omero_backend = OMEROLocalBackend(omero_conn=omero_manager.conn)
        storage_registry["omero_local"] = omero_backend

    # Convert sequential component names to SequentialComponents enum
    sequential_components = []
    if sequential_config and sequential_config.get("sequential_components"):
        for comp_name in sequential_config["sequential_components"]:
            sequential_components.append(SequentialComponents[comp_name])

    # Determine materialization backend for OMERO tests
    # For OMERO tests, use omero_local backend for materialization
    # For other tests, use the configured backend
    if test_config.is_omero:
        materialization_backend = MaterializationBackend("omero_local")
    else:
        materialization_backend = MaterializationBackend(test_config.backend_config)

    # Create PipelineConfig with lazy configs for proper hierarchical inheritance
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix=CONSTANTS.OUTPUT_SUFFIX
        ),
        step_well_filter_config=LazyStepWellFilterConfig(
            well_filter=CONSTANTS.PIPELINE_STEP_WELL_FILTER_TEST
        ),
        sequential_processing_config=LazySequentialProcessingConfig(
            sequential_components=sequential_components if sequential_components else []
        ),
        vfs_config=LazyVFSConfig(materialization_backend=materialization_backend),
    )

    # Convert plate_dir to Path - for OMERO, format as /omero/plate_{id}
    if test_config.is_omero:
        plate_path = Path(f"/omero/plate_{test_config.plate_dir}")
    else:
        plate_path = test_config.plate_dir

    orchestrator = PipelineOrchestrator(plate_path, pipeline_config=pipeline_config)
    orchestrator.initialize()
    return orchestrator


def _export_pipeline_to_file(pipeline: list[Step], plate_dir: Path) -> None:
    """Export pipeline to Python file in the plate directory using the same code as the Code button."""
    from datetime import datetime

    # Create output path in the plate directory
    output_path = plate_dir / "test_pipeline.py"

    # Generate code using the same function as the pipeline editor Code button
    # This ensures consistency between UI and test exports
    python_code = FunctionStepTransportAuthority.source_from_pipeline(pipeline)

    # Wrap in a complete script with header and main block
    lines = []
    lines.append("#!/usr/bin/env python3")
    lines.append('"""')
    lines.append("OpenHCS Pipeline Script")
    lines.append(f"Generated: {datetime.now()}")
    lines.append('"""')
    lines.append("")
    lines.append("def create_pipeline():")
    lines.append('    """Create and return pipeline steps."""')
    lines.append("")

    # Indent the generated pipeline steps code
    for line in python_code.split("\n"):
        if line.strip() and not line.startswith("#"):
            lines.append(f"    {line}")
        elif line.strip().startswith("#"):
            lines.append(f"    {line}")

    lines.append("")
    lines.append("    return pipeline_steps")
    lines.append("")
    lines.append("")
    lines.append("pipeline_steps = create_pipeline()")
    lines.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"📝 Pipeline exported to: {output_path}")


def _drain_progress_queue(q):
    """Consumer thread that drains the progress queue to prevent pipe buffer deadlock.

    Without a consumer, worker processes' feeder threads block when the pipe
    buffer fills up. This prevents workers from exiting, which causes
    ProcessPoolExecutor.shutdown(wait=True) to deadlock.
    """
    while True:
        item = q.get()  # blocking get
        if item is None:  # sentinel to stop
            break


def _execute_pipeline_phases(
    orchestrator: PipelineOrchestrator, pipeline: list[Step]
) -> Dict:
    """Execute compilation and execution phases of the pipeline (direct mode)."""
    import multiprocessing
    import threading
    from openhcs.constants import MULTIPROCESSING_AXIS
    from openhcs.core.progress import set_progress_queue
    import logging

    logger = logging.getLogger(__name__)

    wells = orchestrator.get_component_keys(MULTIPROCESSING_AXIS)
    if not wells:
        raise RuntimeError("No wells found for processing")

    mp_ctx = multiprocessing.get_context("spawn")
    progress_queue = mp_ctx.Queue()

    # Start a consumer thread to drain progress messages and prevent pipe
    # buffer deadlock.  This mirrors the pattern used by zmq_execution_server.
    consumer = threading.Thread(
        target=_drain_progress_queue, args=(progress_queue,), daemon=True
    )
    consumer.start()

    try:
        set_progress_queue(progress_queue)
        compilation_result = orchestrator.compile_pipelines(
            pipeline_definition=pipeline, well_filter=wells
        )

        execution_bundle = compilation_result["execution_bundle"]
        compiled_contexts = execution_bundle.runtime_contexts

        if len(compiled_contexts) != len(wells):
            raise RuntimeError(
                f"Compilation failed: expected {len(wells)} contexts, got {len(compiled_contexts)}"
            )

        # DEBUG: Log Napari streaming configs
        logger.info("=" * 80)
        logger.info("DEBUG: Checking Napari streaming configurations")
        for well_id, ctx in compiled_contexts.items():
            logger.info(f"Well {well_id}: {len(ctx.required_visualizers)} visualizers")
            for i, vis_info in enumerate(ctx.required_visualizers):
                config = vis_info.config
                if hasattr(config, "port"):
                    logger.info(f"  Visualizer {i}: Napari port {config.port}")
                else:
                    logger.info(f"  Visualizer {i}: {type(config).__name__}")
        logger.info("=" * 80)

        # Execution phase - pass the progress queue
        import time

        progress_context = {
            "execution_id": f"direct::{int(time.time() * 1_000_000)}",
            "plate_id": str(orchestrator.plate_path),
            "axis_id": "",
        }

        results = orchestrator.execute_compiled_plate(
            execution_bundle=execution_bundle,
            progress_queue=progress_queue,
            progress_context=progress_context,
        )

        if len(results) != len(wells):
            raise RuntimeError(
                f"Execution failed: expected {len(results)} results, got {len(results)}"
            )

        # Validate all wells succeeded
        failed_wells = [
            well_id for well_id, result in results.items() if not result.is_success()
        ]
        if failed_wells:
            raise RuntimeError(f"Wells failed execution: {failed_wells}")

        return results
    finally:
        set_progress_queue(None)
        # Send sentinel to stop the consumer thread, then wait for it
        progress_queue.put(None)
        consumer.join(timeout=10)
        progress_queue.close()
        progress_queue.join_thread()


def _execute_pipeline_zmq(
    test_config: TestConfig,
    pipeline: list[Step],
    global_config: GlobalPipelineConfig,
    pipeline_config: PipelineConfig,
) -> Dict:
    """Execute pipeline using ZMQ execution client."""
    from openhcs.runtime.zmq_execution_client import ZMQExecutionClient
    import logging

    logger = logging.getLogger(__name__)

    logger.info("🔌 Executing pipeline via ZMQ execution client")

    # Create ZMQ client and connect to server
    # The server will create its own orchestrator and get the well list
    client = ZMQExecutionClient(port=7777, persistent=False)

    try:
        # Connect to server (spawns if needed)
        client.connect(timeout=15)
        logger.info("✅ Connected to ZMQ execution server")

        # Execute pipeline - server will create orchestrator and get wells
        response = client.execute_pipeline(
            OpenHCSExecutionSubmission(
                plate_id=str(test_config.plate_dir),
                pipeline_document=PipelineDocumentAuthority.from_values(
                    pipeline_config=pipeline_config, pipeline_steps=pipeline
                ),
                global_config=global_config,
            )
        )

        # Check response
        if response.get("status") != "complete":
            raise RuntimeError(f"ZMQ execution failed: {response!r}")

        logger.info(f"✅ ZMQ execution completed: {response.get('status')}")

        # Convert response to results format expected by tests
        # ZMQ returns {'status': 'complete', 'execution_id': '...', 'result': {...}}
        # We need to return the result dict
        return response.get("result", {})

    finally:
        # Cleanup
        client.disconnect()
        logger.info("🔌 Disconnected from ZMQ execution server")


def _execute_pipeline_with_mode(
    test_config: TestConfig, pipeline: list[Step], zmq_mode: str, sequential_config=None
) -> Dict:
    """
    Execute pipeline using either direct orchestrator or ZMQ client based on mode.

    Args:
        test_config: Test configuration
        pipeline: Mutable FunctionStep list to execute
        zmq_mode: 'direct' for orchestrator, 'zmq' for ZMQ client

    Returns:
        Execution results dict
    """
    import logging

    logger = logging.getLogger(__name__)

    if zmq_mode == "zmq":
        logger.info("📡 Using ZMQ execution mode")

        # Create global config for ZMQ execution
        global_config = _create_pipeline_config(test_config)

        # Convert sequential component names to SequentialComponents enum
        sequential_components = []
        if sequential_config and sequential_config.get("sequential_components"):
            for comp_name in sequential_config["sequential_components"]:
                sequential_components.append(SequentialComponents[comp_name])

        # Determine materialization backend for OMERO tests
        # For OMERO tests, use omero_local backend for materialization
        # For other tests, use the configured backend
        if test_config.is_omero:
            materialization_backend = MaterializationBackend("omero_local")
        else:
            materialization_backend = MaterializationBackend(test_config.backend_config)

        # Create pipeline config with lazy configs
        from openhcs.core.config import (
            LazyPathPlanningConfig,
            LazyStepWellFilterConfig,
            LazySequentialProcessingConfig,
        )

        pipeline_config = PipelineConfig(
            path_planning_config=LazyPathPlanningConfig(
                output_dir_suffix=CONSTANTS.OUTPUT_SUFFIX
            ),
            step_well_filter_config=LazyStepWellFilterConfig(
                well_filter=CONSTANTS.PIPELINE_STEP_WELL_FILTER_TEST
            ),
            sequential_processing_config=LazySequentialProcessingConfig(
                sequential_components=sequential_components
                if sequential_components
                else []
            ),
            vfs_config=LazyVFSConfig(
                materialization_backend=materialization_backend
            ),
        )

        return _execute_pipeline_zmq(
            test_config, pipeline, global_config, pipeline_config
        )
    else:
        logger.info("🔧 Using direct orchestrator mode")
        orchestrator = _initialize_orchestrator(test_config, sequential_config)
        return _execute_pipeline_phases(orchestrator, pipeline)


def test_main(
    plate_dir: Union[Path, str, int],
    backend_config: str,
    data_type_config: Dict,
    execution_mode: str,
    zmq_execution_mode: str,
    microscope_config: Dict,
    enable_napari: bool,
    enable_fiji: bool,
    sequential_config: Dict,
):
    """Unified test for all combinations of microscope types, backends, data types, execution modes, and sequential processing."""
    # Handle both Path and int (OMERO plate_id)
    if isinstance(plate_dir, int):
        test_config = TestConfig(
            plate_dir, backend_config, execution_mode, microscope_config
        )
    else:
        test_config = TestConfig(
            Path(plate_dir), backend_config, execution_mode, microscope_config
        )

    # For OMERO tests, connect to OMERO (automatically starts if needed)
    if test_config.is_omero:
        from openhcs.runtime.omero_instance_manager import OMEROInstanceManager

        print("� Connecting to OMERO (will auto-start if needed)...")
        omero_manager = OMEROInstanceManager()
        if not omero_manager.connect(
            timeout=120
        ):  # Increased timeout for docker startup
            pytest.skip(
                "OMERO server not available and could not be started automatically. Check Docker installation."
            )
        omero_manager.close()
        print("✅ OMERO server is ready")

    print(
        f"{CONSTANTS.START_INDICATOR} with plate: {plate_dir}, backend: {backend_config}, microscope: {microscope_config['format']}, mode: {execution_mode}, zmq: {zmq_execution_mode}, sequential: {sequential_config['name']}"
    )

    # Create pipeline with sequential configuration
    pipeline = create_test_pipeline(
        enable_napari=enable_napari,
        enable_fiji=enable_fiji,
        sequential_config=sequential_config,
    )

    # If this configuration should fail validation, expect ValueError during compilation
    if sequential_config.get("should_fail", False):
        with pytest.raises(ValueError) as exc_info:
            # Export pipeline to Python file (skip for OMERO - no filesystem)
            if not test_config.is_omero:
                _export_pipeline_to_file(pipeline, test_config.plate_dir)

            # Execute using the specified mode (direct or zmq) - should fail during compilation
            _execute_pipeline_with_mode(
                test_config, pipeline, zmq_execution_mode, sequential_config
            )

        # Verify the error message contains the expected substring
        expected_error = sequential_config.get("expected_error", "")
        assert expected_error in str(exc_info.value), (
            f"Expected error message to contain '{expected_error}', but got: {exc_info.value}"
        )
        print(
            f"✅ Validation correctly rejected invalid configuration: {exc_info.value}"
        )
        return  # Test passed - invalid config was rejected as expected

    # Valid configuration - should succeed
    # Export pipeline to Python file (skip for OMERO - no filesystem)
    if not test_config.is_omero:
        _export_pipeline_to_file(pipeline, test_config.plate_dir)

    # Execute using the specified mode (direct or zmq)
    results = _execute_pipeline_with_mode(
        test_config, pipeline, zmq_execution_mode, sequential_config
    )

    # Validate materialization (skip for OMERO - different validation needed)
    if not test_config.is_omero:
        validate_separate_materialization(test_config.plate_dir)
    else:
        # For OMERO tests, open browser to view the plate
        import os
        import webbrowser

        # Get OMERO web URL from config
        omero_web_host = os.getenv("OMERO_WEB_HOST", "localhost")
        omero_web_port = os.getenv("OMERO_WEB_PORT", "4080")
        plate_url = f"http://{omero_web_host}:{omero_web_port}/webclient/?show=plate-{plate_dir}"

        print(f"\n{'=' * 80}")
        print(f"OMERO Plate URL: {plate_url}")
        print(f"{'=' * 80}\n")

        # Open browser
        webbrowser.open(plate_url)

    print_thread_activity_report()
    print(f"{CONSTANTS.SUCCESS_INDICATOR} ({len(results)} wells processed)")


def _test_main_with_code_serialization(
    plate_dir: Union[Path, str, int],
    backend_config: str,
    data_type_config: Dict,
    execution_mode: str,
    zmq_execution_mode: str,
    microscope_config: Dict,
):
    """
    DISABLED: Code serialization test (not run as pytest test).

    This function tests the code serializer for code-based object serialization,
    but is disabled because:
    1. It's redundant with test_main (which tests the actual integration)
    2. Code serialization is already tested in the PyQt UI
    3. It breaks with OMERO (plate_dir is int, not Path)

    The function is kept for reference but prefixed with _ to exclude from pytest.

    Original purpose:
    - Test using the serializer for code-based object serialization
    - Mirror the UI's approach: create objects → convert to code → exec → use
    - Prove code-based serialization works for remote execution
    """
    # Handle both Path and int (OMERO plate_id)
    if isinstance(plate_dir, int):
        test_config = TestConfig(
            plate_dir, backend_config, execution_mode, microscope_config
        )
    else:
        test_config = TestConfig(
            Path(plate_dir), backend_config, execution_mode, microscope_config
        )

    print(
        f"{CONSTANTS.START_INDICATOR} [CODE SERIALIZATION TEST] with plate: {plate_dir}, backend: {backend_config}, mode: {execution_mode}, zmq: {zmq_execution_mode}"
    )

    # Step 1: Create objects normally
    from polystore.base import reset_memory_backend

    reset_memory_backend()

    global_config = _create_pipeline_config(test_config)
    setup_global_gpu_registry(global_config=global_config)

    # Create PipelineConfig with lazy configs for proper hierarchical inheritance
    pipeline_config = PipelineConfig(
        path_planning_config=LazyPathPlanningConfig(
            output_dir_suffix=CONSTANTS.OUTPUT_SUFFIX
        ),
        step_well_filter_config=LazyStepWellFilterConfig(
            well_filter=CONSTANTS.PIPELINE_STEP_WELL_FILTER_TEST
        ),
    )

    pipeline = create_test_pipeline()

    print("📦 Step 1: Created original objects")
    print(f"   - GlobalPipelineConfig: {type(global_config).__name__}")
    print(f"   - PipelineConfig: {type(pipeline_config).__name__}")
    print(f"   - Pipeline: {len(pipeline)} steps")

    # Step 2: Convert to Python code using the serializer
    import openhcs.serialization.pycodify_formatters  # noqa: F401
    from pycodify import Assignment, generate_python_source

    print("\n🔄 Step 2: Converting objects to Python code...")

    # Generate code for GlobalPipelineConfig
    global_config_code = generate_python_source(
        Assignment("config", global_config),
        header="# Configuration Code",
        clean_mode=True,
    )

    # Generate code for PipelineConfig
    pipeline_config_code = generate_python_source(
        Assignment("config", pipeline_config),
        header="# Configuration Code",
        clean_mode=True,
    )

    # Generate code for Pipeline steps through the canonical transport authority.
    pipeline_steps_code = FunctionStepTransportAuthority.source_from_pipeline(pipeline)

    print(f"   - GlobalPipelineConfig code: {len(global_config_code)} chars")
    print(f"   - PipelineConfig code: {len(pipeline_config_code)} chars")
    print(f"   - Pipeline steps code: {len(pipeline_steps_code)} chars")

    # Save the generated code to files for inspection
    code_output_dir = test_config.plate_dir / "generated_code"
    code_output_dir.mkdir(exist_ok=True)

    (code_output_dir / "global_config.py").write_text(global_config_code)
    (code_output_dir / "pipeline_config.py").write_text(pipeline_config_code)
    (code_output_dir / "pipeline_steps.py").write_text(pipeline_steps_code)

    print(f"   - Saved code to: {code_output_dir}")

    # Step 3: Exec the code to recreate objects
    print("\n⚙️  Step 3: Recreating objects from Python code using exec()...")

    # Recreate GlobalPipelineConfig
    global_config_namespace = {}
    exec(global_config_code, global_config_namespace)
    recreated_global_config = global_config_namespace["config"]

    # Recreate PipelineConfig
    pipeline_config_namespace = {}
    exec(pipeline_config_code, pipeline_config_namespace)
    recreated_pipeline_config = pipeline_config_namespace["config"]

    # Recreate Pipeline steps
    pipeline_steps_namespace = {}
    exec(pipeline_steps_code, pipeline_steps_namespace)
    recreated_pipeline_steps = (
        FunctionStepTransportAuthority.pipeline_steps_from_namespace(
            pipeline_steps_namespace
        )
    )

    print(
        f"   - Recreated GlobalPipelineConfig: {type(recreated_global_config).__name__}"
    )
    print(f"   - Recreated PipelineConfig: {type(recreated_pipeline_config).__name__}")
    print(f"   - Recreated Pipeline steps: {len(recreated_pipeline_steps)} steps")

    # Verify the recreated objects match the originals
    print("\n🔍 Step 4: Verifying recreated objects...")

    # Check GlobalPipelineConfig fields
    assert recreated_global_config.num_workers == global_config.num_workers
    assert recreated_global_config.use_threading == global_config.use_threading
    print(f"   ✅ GlobalPipelineConfig fields match")

    # Check PipelineConfig fields
    assert type(recreated_pipeline_config) == type(pipeline_config)
    print(f"   ✅ PipelineConfig type matches")

    # Check Pipeline steps
    assert len(recreated_pipeline_steps) == len(pipeline)
    for i, (orig_step, recreated_step) in enumerate(
        zip(pipeline, recreated_pipeline_steps)
    ):
        assert type(orig_step) == type(recreated_step)
        assert orig_step.name == recreated_step.name
    print(f"   ✅ Pipeline steps match ({len(recreated_pipeline_steps)} steps)")

    # Step 5: Use recreated objects for execution
    print("\n🚀 Step 5: Executing pipeline with recreated objects...")

    # Copy reconstructed steps into the mutable execution list.
    recreated_pipeline = list(recreated_pipeline_steps)

    # Execute using the specified mode (direct or zmq)
    if zmq_execution_mode == "zmq":
        # For ZMQ mode, use the recreated configs directly
        results = _execute_pipeline_zmq(
            test_config,
            recreated_pipeline,
            recreated_global_config,
            recreated_pipeline_config,
        )
    else:
        # For direct mode, set up global context and use orchestrator
        ensure_global_config_context(GlobalPipelineConfig, recreated_global_config)
        orchestrator = PipelineOrchestrator(
            test_config.plate_dir, pipeline_config=recreated_pipeline_config
        )
        orchestrator.initialize()
        results = _execute_pipeline_phases(orchestrator, recreated_pipeline)

    validate_separate_materialization(test_config.plate_dir)

    print_thread_activity_report()
    print(
        f"\n{CONSTANTS.SUCCESS_INDICATOR} [CODE SERIALIZATION TEST] ({len(results)} wells processed)"
    )
    print("✅ Code-based serialization works perfectly!")
    print(
        "   This proves we can use Python code instead of pickling for remote execution."
    )
