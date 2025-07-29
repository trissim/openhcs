#!/usr/bin/env python3
"""
Test the generic analysis results consolidation function.
"""

import sys
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results

def main():
    """Test the consolidation function on the axotomy experiment data."""
    
    results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
    
    print("🔄 Testing generic analysis consolidation...")
    print(f"📁 Results directory: {results_dir}")
    
    try:
        # Run consolidation
        summary_df = consolidate_analysis_results(
            results_directory=results_dir,
            exclude_patterns=[r".*consolidated.*", r".*summary.*"]  # Exclude our own outputs
        )
        
        print("✅ Consolidation successful!")
        print(f"📊 Summary table shape: {summary_df.shape}")
        print(f"🏥 Wells processed: {summary_df['well_id'].nunique()}")
        
        # Show column structure
        print(f"\n📋 Columns ({len(summary_df.columns)}):")
        for i, col in enumerate(summary_df.columns):
            print(f"  {i+1:2d}. {col}")
        
        # Show first few rows
        print(f"\n📈 First 3 wells:")
        print(summary_df.head(3).to_string())
        
        # Show some key metrics
        print(f"\n🔍 Sample metrics for well B02:")
        b02_data = summary_df[summary_df['well_id'] == 'B02']
        if not b02_data.empty:
            row = b02_data.iloc[0]
            for col in summary_df.columns:
                if 'cell_counts_details' in col and 'count' in col:
                    print(f"  {col}: {row[col]}")
        
        print(f"\n💾 Output saved to: {results_dir}/consolidated_analysis_summary.csv")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
