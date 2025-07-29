"""
Integration tests for the pipeline and TUI components.
"""
import pytest
import sys
import os
import io
import logging
from contextlib import redirect_stdout, redirect_stderr
from typing import Union, Dict, List, Any, Optional
from pathlib import Path

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps import FunctionStep as Step
from openhcs.constants.constants import VariableComponents
from openhcs.core.config import GlobalPipelineConfig, VFSConfig, MaterializationBackend

# Import processing functions directly
from openhcs.processing.backends.processors.numpy_processor import (
    create_projection, sharpen, stack_percentile_normalize,
    stack_equalize_histogram, create_composite
)
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
from openhcs.processing.backends.assemblers.assemble_stack_cpu import assemble_stack_cpu
from openhcs.processing.backends.enhance.basic_processor_jax import basic_flatfield_correction_jax
from openhcs.processing.backends.enhance.basic_processor_numpy import basic_flatfield_correction_numpy
from openhcs.processing.backends.enhance.n2v2_processor_torch import n2v2_denoise_torch
from openhcs.processing.backends.enhance.self_supervised_3d_deconvolution import self_supervised_3d_deconvolution

# Import fixtures and utilities from fixture_utils.py
from tests.integration.helpers.fixture_utils import (
    microscope_config,
    backend_config,
    data_type_config,
    plate_dir,
    base_test_dir,
    test_function_dir,
    test_params,
    flat_plate_dir,
    zstack_plate_dir,
    execution_mode,
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
                 variable_components=[VariableComponents.CHANNEL]
            ),
            Step(name="Z-Stack Flattening",
                 func=(create_projection, {'method': 'max_projection'}),
                 variable_components=[VariableComponents.Z_INDEX],
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
            Step(func=ashlar_compute_tile_positions_gpu,
            ),
            Step(name="Image Enhancement Processing",
                 func=[
                     (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
                 ],
            ),
            #Step(func=n2v2_denoise_torch,
            #),
            #Step(func=basic_flatfield_correction_numpy),
            #),
            #Step(func=self_supervised_3d_deconvolution,
            #),
            #Step(func=(assemble_stack_cupy, {'blend_method': 'rectangular', 'blend_radius': 5.0}),
            #Step(func=(assemble_stack_cupy, {'blend_method': 'rectangular', 'blend_radius': 5.0}),
            Step(func=(assemble_stack_cpu),
                 name="CPU Assembler",
            )
        ],
        name = "Mega Flex Pipeline",
    )



def test_main(plate_dir: Union[Path,str], backend_config: str, data_type_config: Dict[str, Any], execution_mode: str):
    """Unified test for all combinations of microscope types, backends, data types, and execution modes."""

    print(f"🔥 STARTING TEST with plate dir: {plate_dir}, backend: {backend_config}, execution: {execution_mode}")

    def run_test():
        # Initialize GPU registry before creating orchestrator
        print("🔥 Initializing GPU registry...")
        setup_global_gpu_registry()
        print("🔥 GPU registry initialized!")

        # Get threading mode from environment (set by execution_mode fixture)
        use_threading = execution_mode == "threading"
        
        config = GlobalPipelineConfig(
            vfs=VFSConfig(materialization_backend=MaterializationBackend(backend_config)),
            use_threading=use_threading
        )
        
        logger_mode = "THREADING" if use_threading else "MULTIPROCESSING"
        print(f"🔥 EXECUTION MODE: {logger_mode} (use_threading={use_threading})")

        # Initialize orchestrator
        print("🔥 Creating orchestrator...")
        orchestrator = PipelineOrchestrator(plate_dir, global_config=config)
        orchestrator.initialize()
        print("🔥 Orchestrator initialized!")

        # Get pipeline and wells
        from openhcs.constants.constants import GroupBy
        wells = orchestrator.get_component_keys(GroupBy.WELL)
        pipeline = get_pipeline(orchestrator.workspace_path)
        print(f"🔥 Found {len(wells)} wells: {wells}")
        print(f"🔥 Pipeline has {len(pipeline.steps)} steps")

        # Phase 1: Compilation - compile pipelines for all wells
        print("🔥 Starting compilation phase...")

        # DEBUG: Check step IDs before compilation
        step_ids_before = [id(step) for step in pipeline.steps]
        print(f"🔥 Step IDs BEFORE compilation: {step_ids_before}")

        compiled_contexts = orchestrator.compile_pipelines(
            pipeline_definition=pipeline.steps,  # Extract steps from Pipeline object
            well_filter=wells
        )

        # DEBUG: Check step IDs after compilation and in contexts
        step_ids_after = [id(step) for step in pipeline.steps]
        first_well_key = list(compiled_contexts.keys())[0] if compiled_contexts else None
        step_ids_in_contexts = list(compiled_contexts[first_well_key].step_plans.keys()) if first_well_key and hasattr(compiled_contexts[first_well_key], 'step_plans') else []
        print(f"🔥 Step IDs AFTER compilation: {step_ids_after}")
        print(f"🔥 Step IDs in contexts: {step_ids_in_contexts}")

        print("🔥 Compilation completed!")

        # Verify compilation results
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

        # Verify execution results
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
        print(f"🔥 TEST COMPLETED SUCCESSFULLY!")

    # Run the test
    run_test()

