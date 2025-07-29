#!/usr/bin/env python3
"""
Example showing how to use the orchestrator with automatic analysis consolidation.

This demonstrates the complete workflow:
1. Configure analysis consolidation in GlobalPipelineConfig
2. Run a pipeline with the orchestrator
3. Automatic consolidation runs after pipeline completion
"""

import sys
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.core.config import (
    GlobalPipelineConfig,
    PathPlanningConfig,
    AnalysisConsolidationConfig,
    PlateMetadataConfig,
    VFSConfig
)
from openhcs.constants.constants import Backend

def create_production_config_with_consolidation():
    """
    Create a production-ready configuration with analysis consolidation enabled.
    
    This shows the recommended pattern for configuring OpenHCS with automatic
    analysis consolidation.
    """
    
    # Configure analysis consolidation
    consolidation_config = AnalysisConsolidationConfig(
        enabled=True,  # Enable automatic consolidation
        metaxpress_style=True,  # Generate MetaXpress-compatible output
        well_pattern=r"([A-Z]\d{2})",  # Standard 96-well pattern
        file_extensions=(".csv",),  # Process CSV files
        exclude_patterns=(  # Avoid processing our own outputs
            r".*consolidated.*",
            r".*metaxpress.*", 
            r".*summary.*"
        ),
        output_filename="experiment_analysis_summary.csv"  # Custom output name
    )
    
    # Configure plate metadata for MetaXpress compatibility
    plate_metadata_config = PlateMetadataConfig(
        barcode="EXP-2025-001",  # Custom experiment barcode
        plate_name="Axotomy FCA DMSO Experiment",  # Descriptive name
        plate_id="13053",  # Plate ID from your experiment
        description="Axotomy experiment with FCA and DMSO treatments. Automated analysis with cell counting, axon morphology, and template matching.",
        acquisition_user="OpenHCS-Pipeline",  # User identifier
        z_step="1"  # Z-step for MetaXpress compatibility
    )
    
    # Configure path planning
    path_planning_config = PathPlanningConfig(
        output_dir_suffix="_openhcs_analysis",  # Custom suffix
        global_output_folder="/home/ts/nvme_usb/OpenHCS/",  # Your output location
        materialization_results_path="results"  # Results subdirectory
    )
    
    # Create the complete configuration
    global_config = GlobalPipelineConfig(
        num_workers=4,  # Adjust based on your system
        path_planning=path_planning_config,
        vfs=VFSConfig(
            intermediate_backend=Backend.MEMORY  # Fast intermediate storage
        ),
        analysis_consolidation=consolidation_config,  # Enable consolidation
        plate_metadata=plate_metadata_config  # Metadata for outputs
    )
    
    return global_config

def example_pipeline_with_consolidation():
    """
    Example of running a pipeline with automatic consolidation.
    
    This shows how the consolidation happens automatically after pipeline completion.
    """
    
    print("🔬 OpenHCS Pipeline with Automatic Analysis Consolidation")
    print("=" * 70)
    
    # Create configuration
    global_config = create_production_config_with_consolidation()
    
    print("📋 Configuration Summary:")
    print(f"  Analysis consolidation: {'✅ Enabled' if global_config.analysis_consolidation.enabled else '❌ Disabled'}")
    print(f"  MetaXpress style: {global_config.analysis_consolidation.metaxpress_style}")
    print(f"  Output filename: {global_config.analysis_consolidation.output_filename}")
    print(f"  Plate metadata: {global_config.plate_metadata.barcode}")
    print(f"  Results path: {global_config.path_planning.materialization_results_path}")
    
    print(f"\n🎯 How it works:")
    print(f"  1. Run your normal OpenHCS pipeline with this configuration")
    print(f"  2. After pipeline completion, consolidation runs automatically")
    print(f"  3. MetaXpress-compatible summary is saved to results directory")
    print(f"  4. Use the summary file with your existing analysis scripts")
    
    # Example of what the orchestrator call would look like:
    print(f"\n💻 Example Usage:")
    print(f"""
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_step import FunctionStep

# Your pipeline steps
pipeline_steps = [
    FunctionStep(func=your_analysis_function, name="analysis_step"),
    # ... more steps
]

# Create orchestrator with consolidation config
orchestrator = PipelineOrchestrator(
    plate_path="/path/to/your/plate",
    global_config=global_config  # This config enables consolidation
)

# Run pipeline - consolidation happens automatically at the end
orchestrator.initialize()
compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)
results = orchestrator.execute_compiled_plate(
    pipeline_definition=pipeline_steps,
    compiled_contexts=compiled_contexts,
    max_workers=global_config.num_workers
)

# After this completes, you'll find:
# {{output_dir}}/results/experiment_analysis_summary.csv
# Ready for your existing MetaXpress processing scripts!
""")

def test_consolidation_on_existing_results():
    """
    Test the consolidation function on existing results to show it works.
    """
    
    print("\n🧪 Testing Consolidation on Existing Results")
    print("=" * 50)
    
    try:
        from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results
        
        # Create config objects
        config = create_production_config_with_consolidation()
        
        # Test on existing results
        results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
        
        print(f"🔄 Running consolidation with new config system...")
        
        summary_df = consolidate_analysis_results(
            results_directory=results_dir,
            consolidation_config=config.analysis_consolidation,
            plate_metadata_config=config.plate_metadata
        )
        
        print(f"✅ Success! Generated summary with:")
        print(f"  📊 {summary_df.shape[0]} wells")
        print(f"  📈 {summary_df.shape[1]} metrics")
        print(f"  📁 Saved as: {config.analysis_consolidation.output_filename}")
        
        # Show sample data
        print(f"\n📋 Sample metrics for well B02:")
        b02_data = summary_df[summary_df['Well'] == 'B02'].iloc[0]
        
        # Show a few key metrics
        for col in summary_df.columns:
            if 'Number of Objects' in col:
                print(f"  {col}: {b02_data[col]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main example function."""
    
    # Show the configuration example
    example_pipeline_with_consolidation()
    
    # Test it on real data
    success = test_consolidation_on_existing_results()
    
    if success:
        print(f"\n🎉 Integration Complete!")
        print(f"✅ Your OpenHCS pipelines will now automatically generate")
        print(f"   MetaXpress-compatible summary tables after completion.")
        print(f"✅ Use your existing analysis scripts without modification!")
    else:
        print(f"\n❌ Integration test failed. Check the error messages above.")

if __name__ == "__main__":
    main()
