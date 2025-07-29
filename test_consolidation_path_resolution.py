#!/usr/bin/env python3
"""
Test the updated consolidation path resolution using compiled contexts.
"""

import sys
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from pathlib import Path
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.config import get_default_global_config
from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results

def test_path_resolution():
    """Test that the orchestrator can resolve paths correctly from compiled contexts."""
    
    print("🔧 Testing Path Resolution from Compiled Contexts")
    print("=" * 60)
    
    # Use an existing plate that has been processed
    plate_path = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched"
    
    if not Path(plate_path).exists():
        print(f"❌ Test plate not found: {plate_path}")
        return False
    
    try:
        # Create orchestrator with default config (consolidation enabled)
        global_config = get_default_global_config()
        print(f"📋 Consolidation enabled: {global_config.analysis_consolidation.enabled}")
        
        orchestrator = PipelineOrchestrator(
            plate_path=plate_path,
            global_config=global_config
        )
        
        # Initialize orchestrator
        print("🔄 Initializing orchestrator...")
        orchestrator.initialize()
        
        # Create a simple pipeline with a function that has special outputs
        # This will help us test the path resolution
        def dummy_analysis_function(image_stack):
            """Dummy function that would produce special outputs."""
            import numpy as np
            return image_stack, {"test_metric": 42}
        
        # Add special outputs decorator to make it produce materialized results
        from openhcs.core.pipeline.function_contracts import special_outputs
        from openhcs.core.memory.decorators import numpy as numpy_func
        
        @numpy_func
        @special_outputs(("test_results", lambda data, path, fm: path.replace('.pkl', '.csv')))
        def test_analysis_step(image_stack):
            return image_stack, {"test_metric": 42}
        
        pipeline_steps = [
            FunctionStep(func=test_analysis_step, name="test_analysis")
        ]
        
        # Compile the pipeline
        print("🔄 Compiling pipeline...")
        compiled_contexts = orchestrator.compile_pipelines(pipeline_steps)
        
        print(f"✅ Compiled {len(compiled_contexts)} contexts")
        
        # Now test the path resolution method
        print("🔍 Testing path resolution...")
        
        # Store the contexts (simulate what execute_compiled_plate does)
        orchestrator._last_compiled_contexts = compiled_contexts
        
        # Test the new path resolution method
        results_dir = orchestrator._get_results_directory_from_contexts()
        
        if results_dir:
            print(f"✅ Resolved results directory: {results_dir}")
            print(f"📁 Directory exists: {results_dir.exists()}")
            
            if results_dir.exists():
                csv_files = list(results_dir.glob("*.csv"))
                print(f"📊 CSV files found: {len(csv_files)}")
                
                if len(csv_files) > 0:
                    print("🎯 Sample CSV files:")
                    for csv_file in csv_files[:3]:
                        print(f"  - {csv_file.name}")
            
            return True
        else:
            print("❌ Failed to resolve results directory")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_actual_consolidation():
    """Test running consolidation on existing results."""
    
    print("\n🧪 Testing Actual Consolidation")
    print("=" * 60)
    
    try:
        # Use existing results directory
        results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
        
        if not Path(results_dir).exists():
            print(f"❌ Results directory not found: {results_dir}")
            return False
        
        # Get default config
        global_config = get_default_global_config()
        
        print(f"🔄 Running consolidation on: {results_dir}")
        
        # Run consolidation directly
        summary_df = consolidate_analysis_results(
            results_directory=results_dir,
            consolidation_config=global_config.analysis_consolidation,
            plate_metadata_config=global_config.plate_metadata
        )
        
        print(f"✅ Consolidation successful!")
        print(f"📊 Shape: {summary_df.shape}")
        print(f"🏥 Wells: {summary_df['Well'].nunique()}")
        
        # Check if output file was created
        output_file = Path(results_dir) / global_config.analysis_consolidation.output_filename
        if output_file.exists():
            print(f"📁 Output file created: {output_file}")
            print(f"📏 File size: {output_file.stat().st_size} bytes")
        else:
            print(f"⚠️ Output file not found: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consolidation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("🚀 OpenHCS Consolidation Path Resolution Tests")
    print("=" * 70)
    
    tests = [
        ("Path Resolution from Contexts", test_path_resolution),
        ("Actual Consolidation", test_actual_consolidation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 70)
    print("TEST RESULTS SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Path resolution is working! The orchestrator should now")
        print("   automatically consolidate analysis results after execution.")

if __name__ == "__main__":
    main()
