#!/usr/bin/env python3
"""
OpenHCS Pipeline Script - Generated from tmponh9r_ig.pkl
Generated: 2025-07-21 15:03:40.743459
"""

import sys
import os
from pathlib import Path

# Add OpenHCS to path
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import (GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig, 
                         MaterializationBackend, ZarrCompressor, ZarrChunkStrategy)
from openhcs.constants.constants import VariableComponents, Backend, Microscope

# Function and Enum imports
from openhcs.constants.constants import VariableComponents
from openhcs.processing.backends.analysis.cell_counting_cpu import DetectionMethod, count_cells_single_channel
from openhcs.processing.backends.analysis.skan_axon_analysis import AnalysisDimension, skan_axon_skeletonize_and_analyze
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
from openhcs.processing.backends.processors.cupy_processor import create_composite, stack_percentile_normalize, tophat
from openhcs.processing.backends.processors.torch_processor import stack_percentile_normalize

def create_pipeline():
    """Create and return the pipeline configuration."""

    # Plate paths
    plate_paths = ['/home/ts/nvme_usb/IMX/20250528-new-f04-analogs-n1-2-Plate-1_Plate_23318']

    # Global configuration
    global_config = GlobalPipelineConfig(
        num_workers=5,
        path_planning=PathPlanningConfig(
            output_dir_suffix="_stitched",
            global_output_folder="/home/ts/nvme_usb/OpenHCS/",
            materialization_results_path="results"
        ),
        vfs=VFSConfig(
            intermediate_backend=Backend.MEMORY,
            materialization_backend=MaterializationBackend.ZARR
        ),
        zarr=ZarrConfig(
            store_name="images.zarr",
            compressor=ZarrCompressor.ZSTD,
            compression_level=1,
            shuffle=True,
            chunk_strategy=ZarrChunkStrategy.SINGLE,
            ome_zarr_metadata=True,
            write_plate_metadata=True
        ),
        microscope=Microscope.AUTO,
        use_threading=None
    )

    # Pipeline steps
    pipeline_data = {}

    # Steps for plate: 20250528-new-f04-analogs-n1-2-Plate-1_Plate_23318
    steps = []

    # Step 1: preprocess1
    step_1 = FunctionStep(
        func=[
            (stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0,
                'target_max': 65535.0
            }),
            (tophat, {
                'selem_radius': 50,
                'downsample_factor': 4
            })
        ],
        name="preprocess1",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_1)

    # Step 2: composite
    step_2 = FunctionStep(
        func=[
            (create_composite, {})
        ],
        name="composite",
        variable_components=[VariableComponents.CHANNEL],
        force_disk_output=False
    )
    steps.append(step_2)

    # Step 3: find_stitch_positions
    step_3 = FunctionStep(
        func=[
            (ashlar_compute_tile_positions_gpu, {
                'overlap_ratio': 0.1,
                'max_shift': 15.0,
                'stitch_alpha': 0.2,
                'upsample_factor': 10,
                'permutation_upsample': 1,
                'permutation_samples': 1000,
                'min_permutation_samples': 10,
                'max_permutation_tries': 100,
                'window_size_factor': 0.1
            })
        ],
        name="find_stitch_positions",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_3)

    # Step 4: preprocess2
    step_4 = FunctionStep(
        func=[
            (stack_percentile_normalize, {
                'low_percentile': 1.0,
                'high_percentile': 99.0,
                'target_max': 65535.0
            }),
            (tophat, {
                'selem_radius': 50,
                'downsample_factor': 4
            })
        ],
        name="preprocess2",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_4)

    # Step 5: assemble
    step_5 = FunctionStep(
        func=[
            (assemble_stack_cupy, {
                'blend_method': "fixed",
                'fixed_margin_ratio': 0.1,
                'overlap_blend_fraction': 1.0
            })
        ],
        name="assemble",
        variable_components=[VariableComponents.SITE],
        force_disk_output=True
    )
    steps.append(step_5)

    # Step 6: skan
    step_6 = FunctionStep(
        func={            '1': [
            (count_cells_single_channel, {
                'min_sigma': 1.0,
                'max_sigma': 10.0,
                'num_sigma': 10,
                'threshold': 0.1,
                'overlap': 0.5,
                'watershed_footprint_size': 3,
                'watershed_min_distance': 5,
                'gaussian_sigma': 1.0,
                'median_disk_size': 1,
                'min_cell_area': 30,
                'max_cell_area': 200,
                'detection_method': DetectionMethod.WATERSHED
            })
        ],
            '2': [
            (skan_axon_skeletonize_and_analyze, {
                'voxel_spacing': (1.0, 1.0, 1.0),
                'min_object_size': 100,
                'min_branch_length': 10.0,
                'analysis_dimension': AnalysisDimension.TWO_D
            })
        ]
        },
        name="skan",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_6)

    pipeline_data["/home/ts/nvme_usb/IMX/20250528-new-f04-analogs-n1-2-Plate-1_Plate_23318"] = steps

    return plate_paths, pipeline_data, global_config

def setup_signal_handlers():
    """Setup signal handlers to kill all child processes and threads on Ctrl+C."""
    import signal
    import os
    import sys

    def cleanup_and_exit(signum, frame):
        print(f"\n🔥 Signal {signum} received! Cleaning up all processes and threads...")

        os._exit(1)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

def run_pipeline():
    os.environ["OPENHCS_SUBPROCESS_MODE"] = "1"
    plate_paths, pipeline_data, global_config = create_pipeline()
    from openhcs.core.orchestrator.gpu_scheduler import setup_global_gpu_registry
    setup_global_gpu_registry(global_config=global_config)
    for plate_path in plate_paths:
        orchestrator = PipelineOrchestrator(plate_path, global_config=global_config)
        orchestrator.initialize()
        compiled_contexts = orchestrator.compile_pipelines(pipeline_data[plate_path])
        orchestrator.execute_compiled_plate(
            pipeline_definition=pipeline_data[plate_path],
            compiled_contexts=compiled_contexts,
            max_workers=global_config.num_workers
        )

if __name__ == "__main__":
    setup_signal_handlers()
    run_pipeline()
