"""
Integration tests for the pipeline cccccccccccc and TUI components.
"""
import pytest
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Union, Dict, List, Any, Optional
from pathlib import Path

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps import FunctionStep as Step

# Import processing functions directly
from openhcs.processing.backends.processors.torch_processor import (
    create_projection, sharpen, stack_percentile_normalize,
    stack_equalize_histogram, create_composite
)
from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import gpu_ashlar_align_cupy
from openhcs.processing.backends.pos_gen.mist_processor_cupy import mist_compute_tile_positions
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
from openhcs.processing.backends.enhance.basic_processor_cupy import basic_flatfield_correction_cupy
from openhcs.processing.backends.enhance.n2v2_processor_torch import n2v2_denoise_torch
from openhcs.processing.backends.enhance.self_supervised_3d_deconvolution import self_supervised_3d_deconvolution




# Import fixtures and utilities from fixture_utils.py
from tests.integration.helpers.fixture_utils import (
    microscope_config,
    base_test_dir,
    test_function_dir,
    test_params,
    flat_plate_dir,
    zstack_plate_dir,
    flat_plate_dir,
    thread_tracker,
    base_pipeline_config,
    create_config,
    normalize,
    calcein_process,
    dapi_process,
    find_image_files,
    create_synthetic_plate_data,
    print_thread_activity_report
)

def get_pipeline(input_dir):
    return Pipeline(
        steps=[
            Step(func=create_composite,
                 variable_components=['channel']
            ),
            Step(name="Z-Stack Flattening",
                 func=(create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
            ),
            Step(name="Image Enhancement Processing",
                 func=[
                     (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
                 ],
            ),
            #Step(name="Image Enhancement Processing",
            #     func=[
            #         (sharpen, {'amount': 1.5}),
            #         (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
            #         stack_equalize_histogram  # No parameters needed
            #     ],
            #),
            #Step(func=gpu_ashlar_align_cupy,
            #),
            Step(func=mist_compute_tile_positions,
            ),
            Step(func=n2v2_denoise_torch,
            ),
            Step(func=basic_flatfield_correction_cupy,
            ),
            Step(func=self_supervised_3d_deconvolution,
            ),
            Step(func=(assemble_stack_cupy, {'blend_method': 'rectangular', 'blend_radius': 5.0}),
            )
        ],
        name = "Mega Flex Pipeline",
    )



def test_main_3d(zstack_plate_dir: Union[Path,str]):
    # DO NOT suppress output - we want to see what's happening!
    print(f"🔥 STARTING TEST with plate dir: {zstack_plate_dir}")

    # Initialize GPU registry before creating orchestrator
    print("🔥 Initializing GPU registry...")
    setup_global_gpu_registry()
    print("🔥 GPU registry initialized!")

    # Initialize orchestrator
    print("🔥 Creating orchestrator...")
    orchestrator = PipelineOrchestrator(zstack_plate_dir)
    orchestrator.initialize()
    print("🔥 Orchestrator initialized!")

    # Get pipeline and wells
    wells = orchestrator.get_wells()
    pipeline = get_pipeline(orchestrator.workspace_path)
    print(f"🔥 Found {len(wells)} wells: {wells}")
    print(f"🔥 Pipeline has {len(pipeline.steps)} steps")

    # Phase 1: Compilation - compile pipelines for all wells
    print("🔥 Starting compilation phase...")
    compiled_contexts = orchestrator.compile_pipelines(
        pipeline_definition=pipeline.steps,  # Extract steps from Pipeline object
        well_filter=wells
    )
    print("🔥 Compilation completed!")

    # Verify compilation results with loud failures
    if not compiled_contexts:
        raise RuntimeError("🔥 COMPILATION FAILED: No compiled contexts returned!")
    if len(compiled_contexts) != len(wells):
        raise RuntimeError(f"🔥 COMPILATION FAILED: Expected {len(wells)} contexts, got {len(compiled_contexts)}")
    print(f"🔥 Compilation SUCCESS: {len(compiled_contexts)} contexts compiled")

    # Phase 2: Execution - execute compiled pipelines
    print("🔥 Starting execution phase...")
    results = orchestrator.execute_compiled_plate(
        pipeline_definition=pipeline.steps,  # Use steps, not Pipeline object
        compiled_contexts=compiled_contexts
    )
    print("🔥 Execution completed!")

    # Verify execution results with loud failures
    if not results:
        raise RuntimeError("🔥 EXECUTION FAILED: No results returned!")
    if len(results) != len(wells):
        raise RuntimeError(f"🔥 EXECUTION FAILED: Expected {len(wells)} results, got {len(results)}")

    # Check that all wells executed successfully
    for well_id, result in results.items():
        if result.get('status') != 'success':
            error_msg = result.get('error_message', 'Unknown error')
            raise RuntimeError(f"🔥 EXECUTION FAILED for well {well_id}: {error_msg}")

    print(f"🔥 EXECUTION SUCCESS: {len(results)} wells executed successfully")

    print_thread_activity_report()
    print("🔥 TEST COMPLETED SUCCESSFULLY!")

def test_main_2d(flat_plate_dir: Union[Path,str]):
    # Suppress stdout and stderr to avoid microscopy data generator output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Initialize GPU registry before creating orchestrator
        setup_global_gpu_registry()

        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(flat_plate_dir)
        orchestrator.initialize()

        # Get pipeline and wells
        pipeline = get_pipeline(orchestrator.workspace_path)
        wells = orchestrator.get_wells()

        # Phase 1: Compilation - explicitly compile pipelines
        compiled_contexts = orchestrator.compile_pipelines(
            pipeline_definition=pipeline.steps,  # Extract steps from Pipeline object
            well_filter=wells
        )

        # Verify compilation results
        assert compiled_contexts, "Pipeline compilation failed"
        assert len(compiled_contexts) == len(wells), "Not all wells were compiled"

        # Phase 2: Execution - execute compiled pipelines
        results = orchestrator.execute_compiled_plate(
            pipeline_definition=pipeline.steps,  # Use steps, not Pipeline object
            compiled_contexts=compiled_contexts
        )

        # Verify execution results
        assert results, "Pipeline execution failed"
        assert len(results) == len(wells), "Not all wells were executed"

        print_thread_activity_report()

