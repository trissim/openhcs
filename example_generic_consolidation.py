#!/usr/bin/env python3
"""
Example usage of the generic OpenHCS special outputs consolidation system.

This demonstrates how to use the consolidate_special_outputs function in an OpenHCS pipeline
to automatically consolidate any CSV-based special outputs into summary tables.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add OpenHCS to path
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.core.steps.function_step import FunctionStep
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.config import GlobalPipelineConfig, VFSConfig
from openhcs.constants.constants import VariableComponents, Backend
from openhcs.processing.backends.analysis.consolidate_special_outputs import (
    consolidate_special_outputs, 
    WellPatternType
)


def create_consolidation_pipeline(results_directory: str, output_directory: str):
    """
    Create a pipeline that consolidates special outputs from a results directory.
    
    Args:
        results_directory: Directory containing CSV special outputs
        output_directory: Directory where consolidated results will be saved
        
    Returns:
        List of FunctionStep objects
    """
    
    # Create consolidation step
    consolidation_step = FunctionStep(
        func=(consolidate_special_outputs, {
            'results_directory': results_directory,
            'well_pattern': WellPatternType.STANDARD_96.value,  # A01, B02, etc.
            'file_extensions': [".csv"],
            'include_patterns': None,  # Include all CSV files
            'exclude_patterns': [r".*consolidated.*", r".*summary.*"],  # Exclude existing summaries
        }),
        name="consolidate_special_outputs",
        force_disk_output=True  # Ensure results are saved to disk
    )
    
    return [consolidation_step]


def run_consolidation_example():
    """Run the consolidation example on the axotomy experiment data."""
    
    # Configuration
    results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
    output_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/consolidated_outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create global configuration
    global_config = GlobalPipelineConfig(
        num_workers=1,  # Single worker for this example
        vfs=VFSConfig(
            intermediate_backend=Backend.MEMORY,
            materialization_backend=Backend.DISK
        )
    )
    
    # Create pipeline
    pipeline_steps = create_consolidation_pipeline(results_dir, output_dir)
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        plate_path=output_dir,  # Use output dir as "plate" for this example
        global_config=global_config
    )
    
    # Initialize orchestrator
    orchestrator.initialize()
    
    # Create a dummy input directory with a single dummy image
    dummy_input_dir = Path(output_dir) / "dummy_input"
    dummy_input_dir.mkdir(exist_ok=True)
    
    # Create a minimal dummy image file
    dummy_image_path = dummy_input_dir / "dummy.tif"
    if not dummy_image_path.exists():
        import tifffile
        dummy_image = np.zeros((1, 100, 100), dtype=np.uint16)
        tifffile.imwrite(dummy_image_path, dummy_image)
    
    # Set up pipeline with dummy input
    for step in pipeline_steps:
        step.input_dir = str(dummy_input_dir)
    
    # Compile and execute pipeline
    try:
        print("🔄 Compiling consolidation pipeline...")
        compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)
        
        print("🚀 Executing consolidation pipeline...")
        orchestrator.execute_compiled_plate(
            pipeline_definition=pipeline_steps,
            compiled_contexts=compiled_contexts,
            max_workers=global_config.num_workers
        )
        
        print("✅ Consolidation pipeline completed successfully!")
        print(f"📁 Check output directory: {output_dir}")
        
        # List generated files
        output_path = Path(output_dir)
        generated_files = list(output_path.rglob("*consolidated*")) + list(output_path.rglob("*report*"))
        
        if generated_files:
            print("\n📄 Generated files:")
            for file_path in generated_files:
                print(f"  - {file_path}")
        
    except Exception as e:
        print(f"❌ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()


def test_consolidation_function_directly():
    """Test the consolidation function directly without the full pipeline."""
    
    print("🧪 Testing consolidation function directly...")
    
    # Test parameters
    results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
    
    # Create dummy image stack
    dummy_image = np.zeros((1, 100, 100), dtype=np.uint16)
    
    try:
        # Call the function directly
        result_image, consolidated_summary, detailed_report = consolidate_special_outputs(
            dummy_image,  # image_stack as positional argument
            results_dir,  # results_directory as positional argument
            well_pattern=WellPatternType.STANDARD_96.value,
            file_extensions=[".csv"],
            exclude_patterns=[r".*consolidated.*", r".*summary.*"]
        )
        
        print("✅ Direct function call successful!")
        
        # Print summary information
        if 'metadata' in consolidated_summary:
            metadata = consolidated_summary['metadata']
            print(f"📊 Processed {metadata['total_wells']} wells")
            print(f"📋 Output types: {metadata['output_types']}")
            print(f"📁 Files processed: {metadata['total_files_processed']}")
        
        if 'summary_table' in consolidated_summary:
            summary_table = consolidated_summary['summary_table']
            print(f"📈 Summary table entries: {len(summary_table)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to run the consolidation examples."""
    
    print("="*80)
    print("OPENHCS GENERIC SPECIAL OUTPUTS CONSOLIDATION EXAMPLE")
    print("="*80)
    
    # Test 1: Direct function call
    print("\n1️⃣ Testing direct function call...")
    direct_success = test_consolidation_function_directly()
    
    if direct_success:
        print("\n2️⃣ Running full pipeline example...")
        run_consolidation_example()
    else:
        print("\n❌ Skipping pipeline example due to direct function failure")
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
