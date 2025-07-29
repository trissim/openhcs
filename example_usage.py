#!/usr/bin/env python3
"""
Example usage of the generic OpenHCS analysis results consolidation function.

This shows how to use consolidate_analysis_results() to create MetaXpress-style
summary tables from any OpenHCS analysis results.
"""

import sys
sys.path.insert(0, "/home/ts/code/projects/openhcs")

from openhcs.processing.backends.analysis.consolidate_analysis_results import consolidate_analysis_results

def main():
    """Example usage of the consolidation function."""
    
    # Your results directory (change this to your experiment)
    results_dir = "/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results"
    
    print("🔬 Consolidating OpenHCS Analysis Results")
    print("=" * 50)
    
    # Run consolidation
    summary_df = consolidate_analysis_results(
        results_directory=results_dir,
        exclude_patterns=[r".*consolidated.*"]  # Don't process our own outputs
    )
    
    print(f"✅ Created summary table: {summary_df.shape[0]} wells × {summary_df.shape[1]} metrics")
    
    # Example queries - now you can easily ask "what did analysis X give for well Y?"
    print("\n🔍 Example Queries:")
    print("-" * 30)
    
    # Cell count for well B02
    if 'cell_counts_details_total_rows' in summary_df.columns:
        b02_cells = summary_df[summary_df['well_id'] == 'B02']['cell_counts_details_total_rows'].iloc[0]
        print(f"📊 Well B02 total cells: {b02_cells:,}")
    
    # Axon branch count for well B02  
    if 'axon_analysis_branches_branch_distance_count' in summary_df.columns:
        b02_branches = summary_df[summary_df['well_id'] == 'B02']['axon_analysis_branches_branch_distance_count'].iloc[0]
        print(f"🌿 Well B02 total branches: {b02_branches}")
    
    # Mean cell area across all wells
    if 'cell_counts_details_cell_area_mean' in summary_df.columns:
        mean_cell_areas = summary_df['cell_counts_details_cell_area_mean'].mean()
        print(f"📏 Average cell area across all wells: {mean_cell_areas:.1f} pixels²")
    
    # Wells with template matches
    match_cols = [col for col in summary_df.columns if 'match_results' in col and 'missing' in col]
    if match_cols:
        wells_with_matches = summary_df[summary_df[match_cols[0]] != True]['well_id'].tolist()
        print(f"🎯 Wells with template matches: {', '.join(wells_with_matches)}")
    
    print(f"\n💾 Full results saved to: {results_dir}/consolidated_analysis_summary.csv")
    print("\n📈 Ready for downstream stats analysis!")

if __name__ == "__main__":
    main()
