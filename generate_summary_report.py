#!/usr/bin/env python3
"""
Generate a human-readable summary report from consolidated analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_summary_report(csv_file: Path, output_file: Path):
    """Generate a comprehensive summary report from the consolidated CSV."""
    
    # Load the consolidated data
    df = pd.read_csv(csv_file)
    
    # Create the report
    report_lines = []
    report_lines.append("="*100)
    report_lines.append("OPENHCS ANALYSIS RESULTS SUMMARY REPORT")
    report_lines.append("="*100)
    report_lines.append(f"Generated from: {csv_file}")
    report_lines.append(f"Total wells analyzed: {len(df)}")
    report_lines.append("")
    
    # Well coverage summary
    report_lines.append("ANALYSIS COVERAGE BY WELL:")
    report_lines.append("-" * 50)
    for _, row in df.iterrows():
        well_id = row['well_id']
        analyses = []
        if row['has_cell_counts']:
            analyses.append("Cells")
        if row['has_axon_analysis']:
            analyses.append("Axons")
        if row['has_match_results']:
            analyses.append("Matches")
        report_lines.append(f"  {well_id}: {', '.join(analyses)}")
    report_lines.append("")
    
    # Cell analysis summary
    if 'cell_total_cells' in df.columns:
        report_lines.append("CELL ANALYSIS SUMMARY:")
        report_lines.append("-" * 50)
        total_cells = df['cell_total_cells'].sum()
        mean_cells = df['cell_total_cells'].mean()
        std_cells = df['cell_total_cells'].std()
        min_cells = df['cell_total_cells'].min()
        max_cells = df['cell_total_cells'].max()
        
        report_lines.append(f"  Total cells detected across all wells: {total_cells:,}")
        report_lines.append(f"  Mean cells per well: {mean_cells:.1f} ± {std_cells:.1f}")
        report_lines.append(f"  Range: {min_cells} - {max_cells} cells per well")
        
        if 'cell_mean_cell_area' in df.columns:
            mean_area = df['cell_mean_cell_area'].mean()
            report_lines.append(f"  Average cell area: {mean_area:.1f} pixels²")
        
        if 'cell_mean_cell_intensity' in df.columns:
            mean_intensity = df['cell_mean_cell_intensity'].mean()
            report_lines.append(f"  Average cell intensity: {mean_intensity:.0f}")
        
        if 'cell_mean_detection_confidence' in df.columns:
            mean_confidence = df['cell_mean_detection_confidence'].mean()
            report_lines.append(f"  Average detection confidence: {mean_confidence:.3f}")
        
        report_lines.append("")
        
        # Top wells by cell count
        report_lines.append("  Top 5 wells by cell count:")
        top_cells = df.nlargest(5, 'cell_total_cells')[['well_id', 'cell_total_cells']]
        for _, row in top_cells.iterrows():
            report_lines.append(f"    {row['well_id']}: {row['cell_total_cells']:,} cells")
        report_lines.append("")
    
    # Axon analysis summary
    if 'axon_total_branches' in df.columns:
        report_lines.append("AXON ANALYSIS SUMMARY:")
        report_lines.append("-" * 50)
        total_branches = df['axon_total_branches'].sum()
        mean_branches = df['axon_total_branches'].mean()
        std_branches = df['axon_total_branches'].std()
        
        report_lines.append(f"  Total branches detected: {total_branches:,}")
        report_lines.append(f"  Mean branches per well: {mean_branches:.1f} ± {std_branches:.1f}")
        
        if 'axon_total_branch_length' in df.columns:
            total_length = df['axon_total_branch_length'].sum()
            mean_length = df['axon_mean_branch_length'].mean()
            report_lines.append(f"  Total branch length: {total_length:.1f} pixels")
            report_lines.append(f"  Average branch length: {mean_length:.1f} pixels")
        
        if 'axon_unique_skeletons' in df.columns:
            total_skeletons = df['axon_unique_skeletons'].sum()
            report_lines.append(f"  Total unique skeletons: {total_skeletons}")
        
        report_lines.append("")
        
        # Top wells by branch count
        report_lines.append("  Top 5 wells by branch count:")
        top_branches = df.nlargest(5, 'axon_total_branches')[['well_id', 'axon_total_branches']]
        for _, row in top_branches.iterrows():
            report_lines.append(f"    {row['well_id']}: {row['axon_total_branches']} branches")
        report_lines.append("")
    
    # Match results summary
    if 'match_total_matches' in df.columns:
        match_wells = df[df['has_match_results'] == True]
        if len(match_wells) > 0:
            report_lines.append("TEMPLATE MATCHING SUMMARY:")
            report_lines.append("-" * 50)
            total_matches = match_wells['match_total_matches'].sum()
            mean_confidence = match_wells['match_mean_confidence_score'].mean()
            
            report_lines.append(f"  Wells with template matching: {len(match_wells)}")
            report_lines.append(f"  Total matches found: {total_matches}")
            report_lines.append(f"  Average confidence score: {mean_confidence:.3f}")
            
            if 'match_best_matches' in df.columns:
                best_matches = match_wells['match_best_matches'].sum()
                report_lines.append(f"  Best matches: {best_matches}")
            
            if 'match_high_confidence_matches' in df.columns:
                high_conf = match_wells['match_high_confidence_matches'].sum()
                report_lines.append(f"  High confidence matches: {high_conf}")
            
            report_lines.append("")
            
            # Wells with matches
            report_lines.append("  Wells with template matches:")
            for _, row in match_wells.iterrows():
                conf = row['match_mean_confidence_score']
                report_lines.append(f"    {row['well_id']}: {row['match_total_matches']} matches (confidence: {conf:.3f})")
            report_lines.append("")
    
    # Statistical correlations
    report_lines.append("CORRELATIONS:")
    report_lines.append("-" * 50)
    
    if 'cell_total_cells' in df.columns and 'axon_total_branches' in df.columns:
        correlation = df['cell_total_cells'].corr(df['axon_total_branches'])
        report_lines.append(f"  Cell count vs. Branch count correlation: {correlation:.3f}")
    
    if 'cell_mean_cell_area' in df.columns and 'cell_total_cells' in df.columns:
        correlation = df['cell_mean_cell_area'].corr(df['cell_total_cells'])
        report_lines.append(f"  Cell area vs. Cell count correlation: {correlation:.3f}")
    
    report_lines.append("")
    
    # Data quality notes
    report_lines.append("DATA QUALITY NOTES:")
    report_lines.append("-" * 50)
    
    # Check for wells with no data
    if 'cell_total_cells' in df.columns:
        no_cells = df[df['cell_total_cells'] == 0]
        if len(no_cells) > 0:
            report_lines.append(f"  Wells with no cells detected: {', '.join(no_cells['well_id'].tolist())}")
    
    if 'axon_total_branches' in df.columns:
        no_axons = df[df['axon_total_branches'] == 0]
        if len(no_axons) > 0:
            report_lines.append(f"  Wells with no axon branches: {', '.join(no_axons['well_id'].tolist())}")
    
    # Check for outliers (values > 3 standard deviations from mean)
    if 'cell_total_cells' in df.columns:
        mean_cells = df['cell_total_cells'].mean()
        std_cells = df['cell_total_cells'].std()
        outliers = df[abs(df['cell_total_cells'] - mean_cells) > 3 * std_cells]
        if len(outliers) > 0:
            report_lines.append(f"  Cell count outliers (>3σ): {', '.join(outliers['well_id'].tolist())}")
    
    report_lines.append("")
    report_lines.append("="*100)
    report_lines.append("END OF REPORT")
    report_lines.append("="*100)
    
    # Write the report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    # Also print to console
    print('\n'.join(report_lines))
    
    logger.info(f"Summary report saved to: {output_file}")

def main():
    """Main function."""
    
    results_dir = Path("/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results")
    csv_file = results_dir / "consolidated_analysis_summary.csv"
    output_file = results_dir / "analysis_summary_report.txt"
    
    if not csv_file.exists():
        logger.error(f"Consolidated CSV file not found: {csv_file}")
        return
    
    generate_summary_report(csv_file, output_file)

if __name__ == "__main__":
    main()
