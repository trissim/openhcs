#!/usr/bin/env python3
"""
Test the MetaXpress-style consolidation function.
"""

import sys
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results

def main():
    """Test the MetaXpress-style consolidation function."""
    
    results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
    
    print("🔬 Testing MetaXpress-Style Consolidation")
    print("=" * 50)
    
    # Test MetaXpress style
    print("\n1️⃣ MetaXpress Style Output:")
    mx_df = consolidate_analysis_results(
        results_directory=results_dir,
        exclude_patterns=[r".*consolidated.*", r".*metaxpress.*"],
        metaxpress_style=True
    )
    
    print(f"📊 Shape: {mx_df.shape}")
    print(f"📋 Columns ({len(mx_df.columns)}):")
    for i, col in enumerate(mx_df.columns[:15]):  # Show first 15 columns
        print(f"  {i+1:2d}. {col}")
    if len(mx_df.columns) > 15:
        print(f"  ... and {len(mx_df.columns) - 15} more columns")
    
    print(f"\n📈 Sample data for first 3 wells:")
    print(mx_df.head(3).iloc[:, :8].to_string())  # Show first 8 columns
    
    # Compare with original style
    print(f"\n2️⃣ Original Style Output:")
    orig_df = consolidate_analysis_results(
        results_directory=results_dir,
        exclude_patterns=[r".*consolidated.*", r".*metaxpress.*"],
        metaxpress_style=False
    )
    
    print(f"📊 Shape: {orig_df.shape}")
    print(f"📋 First 10 columns:")
    for i, col in enumerate(orig_df.columns[:10]):
        print(f"  {i+1:2d}. {col}")
    
    print(f"\n💾 Files saved:")
    print(f"  - MetaXpress style: {results_dir}/metaxpress_style_summary.csv")
    print(f"  - Original style: {results_dir}/consolidated_analysis_summary.csv")

if __name__ == "__main__":
    main()
