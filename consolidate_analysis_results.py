#!/usr/bin/env python3
"""
Consolidate OpenHCS analysis results into a summary table.

This script processes per-well analysis results from:
- cell_counts_details.csv: Cell detection and counting data
- axon_analysis_branches.csv: Axon morphology and branching analysis
- match_results_mtm_matches.csv: Template matching results

Generates a comprehensive summary table with per-well statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_well_id(filename: str) -> str:
    """Extract well ID from filename (e.g., 'B02_cell_counts_details.csv' -> 'B02')."""
    match = re.match(r'^([A-Z]\d{2})_', filename)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract well ID from filename: {filename}")

def process_cell_counts(file_path: Path) -> Dict:
    """Process cell counts details file and return summary statistics."""
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            return {
                'total_cells': 0,
                'mean_cell_area': np.nan,
                'std_cell_area': np.nan,
                'mean_cell_intensity': np.nan,
                'std_cell_intensity': np.nan,
                'mean_detection_confidence': np.nan,
                'detection_methods': '',
                'slices_with_cells': 0
            }
        
        return {
            'total_cells': len(df),
            'mean_cell_area': df['cell_area'].mean(),
            'std_cell_area': df['cell_area'].std(),
            'mean_cell_intensity': df['cell_intensity'].mean(),
            'std_cell_intensity': df['cell_intensity'].std(),
            'mean_detection_confidence': df['detection_confidence'].mean(),
            'detection_methods': ','.join(df['detection_method'].unique()),
            'slices_with_cells': df['slice_index'].nunique()
        }
    except Exception as e:
        logger.error(f"Error processing cell counts file {file_path}: {e}")
        return {}

def process_axon_analysis(file_path: Path) -> Dict:
    """Process axon analysis branches file and return summary statistics."""
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            return {
                'total_branches': 0,
                'total_branch_length': 0.0,
                'mean_branch_length': np.nan,
                'std_branch_length': np.nan,
                'mean_euclidean_distance': np.nan,
                'std_euclidean_distance': np.nan,
                'branch_types': '',
                'unique_skeletons': 0,
                'slices_with_axons': 0
            }
        
        return {
            'total_branches': len(df),
            'total_branch_length': df['branch_distance'].sum(),
            'mean_branch_length': df['branch_distance'].mean(),
            'std_branch_length': df['branch_distance'].std(),
            'mean_euclidean_distance': df['euclidean_distance'].mean(),
            'std_euclidean_distance': df['euclidean_distance'].std(),
            'branch_types': ','.join(map(str, sorted(df['branch_type'].unique()))),
            'unique_skeletons': df['skeleton_id'].nunique(),
            'slices_with_axons': df['z_slice'].nunique()
        }
    except Exception as e:
        logger.error(f"Error processing axon analysis file {file_path}: {e}")
        return {}

def process_match_results(file_path: Path) -> Dict:
    """Process match results file and return summary statistics."""
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            return {
                'total_matches': 0,
                'mean_confidence_score': np.nan,
                'std_confidence_score': np.nan,
                'best_matches': 0,
                'high_confidence_matches': 0,
                'cropped_matches': 0,
                'unique_templates': 0,
                'slices_with_matches': 0
            }
        
        return {
            'total_matches': len(df),
            'mean_confidence_score': df['confidence_score'].mean(),
            'std_confidence_score': df['confidence_score'].std(),
            'best_matches': df['is_best_match'].sum() if 'is_best_match' in df.columns else 0,
            'high_confidence_matches': df['high_confidence'].sum() if 'high_confidence' in df.columns else 0,
            'cropped_matches': df['was_cropped'].sum() if 'was_cropped' in df.columns else 0,
            'unique_templates': df['template_name'].nunique() if 'template_name' in df.columns else 0,
            'slices_with_matches': df['slice_index'].nunique()
        }
    except Exception as e:
        logger.error(f"Error processing match results file {file_path}: {e}")
        return {}

def consolidate_results(results_dir: Path) -> pd.DataFrame:
    """Consolidate all analysis results into a single summary table."""
    
    # Find all CSV files
    csv_files = list(results_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {results_dir}")
    
    # Group files by well ID and type
    wells_data = {}
    
    for csv_file in csv_files:
        try:
            well_id = extract_well_id(csv_file.name)
            
            if well_id not in wells_data:
                wells_data[well_id] = {
                    'well_id': well_id,
                    'has_cell_counts': False,
                    'has_axon_analysis': False,
                    'has_match_results': False
                }
            
            if 'cell_counts_details' in csv_file.name:
                logger.info(f"Processing cell counts for {well_id}")
                cell_data = process_cell_counts(csv_file)
                wells_data[well_id].update({f'cell_{k}': v for k, v in cell_data.items()})
                wells_data[well_id]['has_cell_counts'] = True
                
            elif 'axon_analysis_branches' in csv_file.name:
                logger.info(f"Processing axon analysis for {well_id}")
                axon_data = process_axon_analysis(csv_file)
                wells_data[well_id].update({f'axon_{k}': v for k, v in axon_data.items()})
                wells_data[well_id]['has_axon_analysis'] = True
                
            elif 'match_results_mtm_matches' in csv_file.name:
                logger.info(f"Processing match results for {well_id}")
                match_data = process_match_results(csv_file)
                wells_data[well_id].update({f'match_{k}': v for k, v in match_data.items()})
                wells_data[well_id]['has_match_results'] = True
                
        except Exception as e:
            logger.error(f"Error processing file {csv_file}: {e}")
            continue
    
    # Convert to DataFrame
    summary_df = pd.DataFrame(list(wells_data.values()))
    
    # Sort by well ID
    summary_df = summary_df.sort_values('well_id').reset_index(drop=True)
    
    logger.info(f"Consolidated data for {len(summary_df)} wells")
    return summary_df

def main():
    """Main function to consolidate analysis results."""
    
    results_dir = Path("/home/ts/nvme_usb/OpenHCS/mar-20-axotomy-fca-dmso-Plate-1_Plate_13053_openhcs_stitched_analysis/results")
    
    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return
    
    # Consolidate results
    summary_df = consolidate_results(results_dir)
    
    # Save summary table
    output_file = results_dir / "consolidated_analysis_summary.csv"
    summary_df.to_csv(output_file, index=False)
    logger.info(f"Saved consolidated summary to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ANALYSIS RESULTS CONSOLIDATION SUMMARY")
    print("="*80)
    print(f"Total wells processed: {len(summary_df)}")
    print(f"Wells with cell counts: {summary_df['has_cell_counts'].sum()}")
    print(f"Wells with axon analysis: {summary_df['has_axon_analysis'].sum()}")
    print(f"Wells with match results: {summary_df['has_match_results'].sum()}")
    
    if 'cell_total_cells' in summary_df.columns:
        print(f"\nCell Analysis Summary:")
        print(f"  Total cells detected: {summary_df['cell_total_cells'].sum()}")
        print(f"  Mean cells per well: {summary_df['cell_total_cells'].mean():.1f}")
        print(f"  Wells with cells: {(summary_df['cell_total_cells'] > 0).sum()}")
    
    if 'axon_total_branches' in summary_df.columns:
        print(f"\nAxon Analysis Summary:")
        print(f"  Total branches detected: {summary_df['axon_total_branches'].sum()}")
        print(f"  Mean branches per well: {summary_df['axon_total_branches'].mean():.1f}")
        print(f"  Wells with axons: {(summary_df['axon_total_branches'] > 0).sum()}")
    
    if 'match_total_matches' in summary_df.columns:
        print(f"\nMatch Results Summary:")
        print(f"  Total matches found: {summary_df['match_total_matches'].sum()}")
        print(f"  Mean matches per well: {summary_df['match_total_matches'].mean():.1f}")
        print(f"  Wells with matches: {(summary_df['match_total_matches'] > 0).sum()}")
    
    print(f"\nOutput saved to: {output_file}")
    print("="*80)

if __name__ == "__main__":
    main()
