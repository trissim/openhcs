"""
Special I/O Test Scenario

Tests dict patterns with special outputs (cell_counts and axon_analysis).
This scenario was created to debug intermittent special output saving issues.
"""

from pathlib import Path
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig, VFSConfig, ZarrConfig, MaterializationBackend, ZarrCompressor, ZarrChunkStrategy
from openhcs.constants.constants import VariableComponents, Backend, Microscope

# Function imports
from openhcs.processing.backends.analysis.cell_counting_cpu import count_cells_single_channel
from openhcs.processing.backends.analysis.skan_axon_analysis import skan_axon_skeletonize_and_analyze
from openhcs.processing.backends.assemblers.assemble_stack_cupy import assemble_stack_cupy
from openhcs.processing.backends.pos_gen.ashlar_main_gpu import ashlar_compute_tile_positions_gpu
from openhcs.processing.backends.processors.cupy_processor import create_composite, stack_percentile_normalize, tophat

# Enum imports
from openhcs.processing.backends.analysis.cell_counting_pyclesperanto import DetectionMethod
from openhcs.processing.backends.analysis.skan_axon_analysis import AnalysisDimension


def run_special_io_test(plate_path: str = None, output_folder: str = None):
    """
    Run the special I/O test scenario.
    
    Args:
        plate_path: Path to plate folder (default: test plate)
        output_folder: Output folder (default: /home/ts/nvme_usb/OpenHCS/)
    
    Returns:
        PipelineOrchestrator: Configured orchestrator ready to run
    """
    
    # Default paths
    if plate_path is None:
        plate_path = '/home/ts/nvme_usb/IMX/20250528-new-f04-analogs-n1-2-Plate-1_Plate_23318'
    if output_folder is None:
        output_folder = "/home/ts/nvme_usb/OpenHCS/"
    
    plate_paths = [plate_path]
    
    # Global configuration
    global_config = GlobalPipelineConfig(
        num_workers=5,
        path_planning=PathPlanningConfig(
            output_dir_suffix="_stitched",
            global_output_folder=output_folder,
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
    
    # Create pipeline steps
    steps = []
    
    # Step 1: composite
    step_1 = FunctionStep(
        func=create_composite,
        name="composite",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_1)
    
    # Step 2: preprocess1
    step_2 = FunctionStep(
        func=[(stack_percentile_normalize, {'low_percentile': 1, 'high_percentile': 99}), (tophat, {'radius': 30})],
        name="preprocess1", 
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_2)
    
    # Step 3: find_stitch_positions
    step_3 = FunctionStep(
        func=[(ashlar_compute_tile_positions_gpu, {'overlap_ratio': 0.1, 'max_shift': 15.0, 'stitch_alpha': 0.2, 'upsample_factor': 10, 'permutation_upsample': 1, 'permutation_samples': 1000, 'min_permutation_samples': 10, 'max_permutation_tries': 100, 'window_size_factor': 0.1})],
        name="find_stitch_positions",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_3)
    
    # Step 4: preprocess2
    step_4 = FunctionStep(
        func=[(stack_percentile_normalize, {'low_percentile': 1, 'high_percentile': 99})],
        name="preprocess2",
        variable_components=[VariableComponents.SITE],
        force_disk_output=False
    )
    steps.append(step_4)
    
    # Step 5: assemble
    step_5 = FunctionStep(
        func=assemble_stack_cupy,
        name="assemble",
        variable_components=[VariableComponents.SITE],
        force_disk_output=True,
    )
    steps.append(step_5)
    
    # Step 6: skan (dict pattern with special outputs)
    step_6 = FunctionStep(
        func={'1': [count_cells_single_channel], '2': [skan_axon_skeletonize_and_analyze]},
        name="skan",
        variable_components=[VariableComponents.SITE],
        group_by=VariableComponents.CHANNEL,
        force_disk_output=False
    )
    steps.append(step_6)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        plate_paths=plate_paths,
        steps=steps,
        global_config=global_config
    )
    
    return orchestrator


if __name__ == "__main__":
    """Run the test scenario directly."""
    orchestrator = run_special_io_test()
    orchestrator.run()
