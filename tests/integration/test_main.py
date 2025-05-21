"""
Integration tests for the pipeline orchestrator and TUI components.
"""
import pytest
import sys
import os
import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Union, Dict, List, Any, Optional
from pathlib import Path

from openhcs.core.orchestrator import PipelineOrchestrator
from openhcs.core.pipeline import Pipeline
from openhcs.core.steps import FunctionStep as Step
from openhcs.processing.backends.processors.cupy_processor import CuPyImageProcessor as CPIP
from openhcs.processing.backends.processors.numpy_processor import NumPyImageProcessor as NPIP
from openhcs.processing.backends.processors.torch_processor import TorchImageProcessor as TORCHIP
from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import gpu_ashlar_align_cupy
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

def get_pipeline(input_dir, IP=TORCHIP):
    return Pipeline(
        steps=[
            Step(name="Z-Stack Flattening",
                 func=(IP.create_projection, {'method': 'max_projection'}),
                 variable_components=['z_index'],
            ),
            Step(name="Image Enhancement Processing",
                 func=[
                     ((IP.sharpen), {'amount': 1.5}),
                     (IP.stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
                     IP.stack_equalize_histogram  # No parameters needed
                 ],
            ),
            Step(func=(IP.create_composite),
                 variable_components=['channel']
            ),
            Step(func=(gpu_ashlar_align_cupy),
            ),
            Step(func=(n2v2_denoise_torch),
            ),
            Step(func=(basic_flatfield_correction_cupy),
            ),
            Step(func=(self_supervised_3d_deconvolution),
            ),
            Step(func=(assemble_stack_cupy),
            )
        ],
        name = "Mega Flex Pipeline",
    )



def test_main_3d(zstack_plate_dir: Union[Path,str]):
    # Suppress stdout and stderr to avoid microscopy data generator output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(zstack_plate_dir)
        orchestrator.initialize()

        # Get pipeline and wells
        pipelines = [get_pipeline(orchestrator.workspace_path)]
        wells = orchestrator.get_wells()

        # Phase 1: Compilation - explicitly compile pipelines
        contexts, compiled_pipelines = orchestrator.compile_pipelines(
            pipeline=pipelines[0],
            wells=wells
        )

        # Verify compilation results
        assert contexts, "Pipeline compilation failed"
        assert len(contexts) == len(wells), "Not all wells were compiled"

        # Phase 2: Execution - execute compiled pipelines
        results = orchestrator.execute_pipelines(
            contexts=contexts,
            pipelines=compiled_pipelines,
            wells=wells
        )

        # Verify execution results
        assert results, "Pipeline execution failed"
        assert len(results) == len(wells), "Not all wells were executed"

        print_thread_activity_report()

def test_main_2d(flat_plate_dir: Union[Path,str]):
    # Suppress stdout and stderr to avoid microscopy data generator output
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(flat_plate_dir)
        orchestrator.initialize()

        # Get pipeline and wells
        pipelines = [get_pipeline(orchestrator.workspace_path)]
        wells = orchestrator.get_wells()

        # Phase 1: Compilation - explicitly compile pipelines
        contexts, compiled_pipelines = orchestrator.compile_pipelines(
            pipeline=pipelines[0],
            wells=wells
        )

        # Verify compilation results
        assert contexts, "Pipeline compilation failed"
        assert len(contexts) == len(wells), "Not all wells were compiled"

        # Phase 2: Execution - execute compiled pipelines
        results = orchestrator.execute_pipelines(
            contexts=contexts,
            pipelines=compiled_pipelines,
            wells=wells
        )

        # Verify execution results
        assert results, "Pipeline execution failed"
        assert len(results) == len(wells), "Not all wells were executed"

        print_thread_activity_report()

