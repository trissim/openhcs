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
from openhcs.processing.backends.enhance.self_supervised_2d_deconvolution import self_supervised_2d_deconvolution
from openhcs.processing.backends.enhance.focus_torch import focus_stack_max_sharpness as focus_torch





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
            Step(func=mist_compute_tile_positions,
            ),
            Step(name="Image Enhancement Processing",
                 func=[
                     (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
                 ],
            ),
            Step(func=n2v2_denoise_torch,
            ),
            Step(func=focus_torch,
            ),
            Step(func=basic_flatfield_correction_cupy,
            ),
            Step(func=(assemble_stack_cupy, {'blend_method': 'rectangular', 'blend_radius': 5.0}),
            )
        ],
        name = "Mega Flex Pipeline",
    )



def run(plate_dir: Union[Path,str]):
    # DO NOT suppress output - we want to see what's happening!
    print(f"🔥 STARTING TEST with plate dir: {plate_dir}")

    # Initialize GPU registry before creating orchestrator
    print("🔥 Initializing GPU registry...")
    setup_global_gpu_registry()
    print("🔥 GPU registry initialized!")

    # Initialize orchestrator
    print("🔥 Creating orchestrator...")
    orchestrator = PipelineOrchestrator(plate_dir)
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

    print("🔥 TEST COMPLETED SUCCESSFULLY!")

plates=['/home/ts/nvme_usb/Opera/20250407TS-12w_axoTest/20250407TS-12w_axoTest/20250407TS-12w_axoTest__2025-04-07T14_16_59-Measurement_2']
        #'/home/ts/nvme_usb/Opera/20250407TS-12w_axoTest/20250407TS-12w_axoTest/20250407TS-12w_axoTest__2025-04-07T14_16_59-Measurement_2']

for plate in plates:
    run(plate)
