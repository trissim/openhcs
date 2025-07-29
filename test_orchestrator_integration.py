#!/usr/bin/env python3
"""
Test the orchestrator integration with automatic analysis consolidation.
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

def test_config_creation():
    """Test creating configuration with analysis consolidation enabled."""
    
    print("🔧 Testing Configuration Creation")
    print("=" * 50)
    
    # Create custom analysis consolidation config
    consolidation_config = AnalysisConsolidationConfig(
        enabled=True,
        metaxpress_style=True,
        well_pattern=r"([A-Z]\d{2})",
        file_extensions=(".csv",),
        exclude_patterns=(".*consolidated.*", ".*metaxpress.*"),
        output_filename="experiment_summary.csv"
    )
    
    # Create custom plate metadata config
    plate_metadata_config = PlateMetadataConfig(
        barcode="TEST-PLATE-001",
        plate_name="Test Axotomy Experiment",
        plate_id="12345",
        description="Test experiment for OpenHCS analysis consolidation",
        acquisition_user="TestUser",
        z_step="1"
    )
    
    # Create path planning config
    path_planning_config = PathPlanningConfig(
        output_dir_suffix="_analysis",
        global_output_folder="/tmp/test_openhcs_output",
        materialization_results_path="results"
    )
    
    # Create global config
    global_config = GlobalPipelineConfig(
        num_workers=2,
        path_planning=path_planning_config,
        vfs=VFSConfig(intermediate_backend=Backend.MEMORY),
        analysis_consolidation=consolidation_config,
        plate_metadata=plate_metadata_config
    )
    
    print("✅ Configuration created successfully!")
    print(f"📊 Analysis consolidation enabled: {global_config.analysis_consolidation.enabled}")
    print(f"🎯 MetaXpress style: {global_config.analysis_consolidation.metaxpress_style}")
    print(f"📁 Output filename: {global_config.analysis_consolidation.output_filename}")
    print(f"🏷️ Plate barcode: {global_config.plate_metadata.barcode}")
    print(f"📂 Results path: {global_config.path_planning.materialization_results_path}")
    
    return global_config

def test_consolidation_function():
    """Test the consolidation function with the new config objects."""
    
    print("\n🧪 Testing Consolidation Function")
    print("=" * 50)
    
    try:
        # Import the updated function
        from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results
        from openhcs.core.config import AnalysisConsolidationConfig, PlateMetadataConfig
        
        # Create config objects
        consolidation_config = AnalysisConsolidationConfig(
            enabled=True,
            metaxpress_style=True,
            exclude_patterns=(".*consolidated.*", ".*metaxpress.*", ".*experiment.*")
        )
        
        plate_metadata_config = PlateMetadataConfig(
            barcode="INTEGRATION-TEST",
            plate_name="Integration Test Plate",
            description="Testing orchestrator integration with analysis consolidation"
        )
        
        # Test on the existing results
        results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
        
        print(f"🔄 Running consolidation on: {results_dir}")
        
        summary_df = consolidate_analysis_results(
            results_directory=results_dir,
            consolidation_config=consolidation_config,
            plate_metadata_config=plate_metadata_config
        )
        
        print(f"✅ Consolidation successful!")
        print(f"📊 Shape: {summary_df.shape}")
        print(f"🏥 Wells: {summary_df['Well'].nunique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Consolidation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_orchestrator_path_resolution():
    """Test that the orchestrator can resolve the results directory correctly."""
    
    print("\n🗂️ Testing Orchestrator Path Resolution")
    print("=" * 50)
    
    try:
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from pathlib import Path
        
        # Create test config
        global_config = test_config_creation()
        
        # Create orchestrator (without initializing)
        test_plate_path = "/tmp/test_plate"
        orchestrator = PipelineOrchestrator(
            plate_path=test_plate_path,
            global_config=global_config
        )
        
        # Test the results directory resolution
        results_dir = orchestrator._get_results_directory()
        
        print(f"📂 Plate path: {test_plate_path}")
        print(f"📁 Resolved results directory: {results_dir}")
        
        # Expected: /tmp/test_openhcs_output/test_plate_analysis/results
        expected_dir = Path("/tmp/test_openhcs_output/test_plate_analysis/results")
        
        if results_dir == expected_dir:
            print("✅ Path resolution correct!")
        else:
            print(f"⚠️ Path resolution mismatch. Expected: {expected_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Path resolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    
    print("🚀 OpenHCS Orchestrator Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Configuration Creation", test_config_creation),
        ("Consolidation Function", test_consolidation_function),
        ("Path Resolution", test_orchestrator_path_resolution)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    if all_passed:
        print("\n🎉 Integration is ready! The orchestrator will now automatically")
        print("   consolidate analysis results after pipeline completion.")

if __name__ == "__main__":
    main()
