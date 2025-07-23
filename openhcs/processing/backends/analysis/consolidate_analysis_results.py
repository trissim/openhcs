"""
Consolidate OpenHCS analysis results into summary tables.

This module provides a standalone function for consolidating any CSV-based analysis results
from OpenHCS pipelines into a single summary table. Creates MetaXpress-style output where
each well is a row and analysis metrics are columns.

Usage:
    # Standalone
    df = consolidate_analysis_results("/path/to/results")
    
    # In pipeline
    FunctionStep(func=consolidate_analysis_results_pipeline, ...)
"""

import pandas as pd
import numpy as np
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.constants.constants import Backend

# Import config classes with TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from openhcs.core.config import AnalysisConsolidationConfig, PlateMetadataConfig

logger = logging.getLogger(__name__)


def extract_well_id(filename: str, pattern: str = r"([A-Z]\d{2})") -> Optional[str]:
    """Extract well ID from filename using regex pattern."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def extract_analysis_type(filename: str, well_id: str) -> str:
    """Extract analysis type from filename after removing well ID and extension."""
    # Remove well ID prefix and file extension
    analysis_type = filename.replace(f"{well_id}_", "").replace(Path(filename).suffix, "")
    return analysis_type


def create_metaxpress_header(summary_df: pd.DataFrame, plate_metadata: Optional[Dict[str, str]] = None) -> List[List[str]]:
    """
    Create MetaXpress-style header rows with metadata.

    Returns list of header rows to prepend to the CSV.
    """
    if plate_metadata is None:
        plate_metadata = {}

    # Extract plate info from results directory or use defaults
    barcode = plate_metadata.get('barcode', 'OpenHCS-Plate')
    plate_name = plate_metadata.get('plate_name', 'OpenHCS Analysis Results')
    plate_id = plate_metadata.get('plate_id', '00000')
    description = plate_metadata.get('description', 'Consolidated analysis results from OpenHCS pipeline')
    acquisition_user = plate_metadata.get('acquisition_user', 'OpenHCS')
    z_step = plate_metadata.get('z_step', '1')

    # Create header rows matching MetaXpress format
    header_rows = [
        ['Barcode', barcode],
        ['Plate Name', plate_name],
        ['Plate ID', plate_id],
        ['Description', description],
        ['Acquisition User', acquisition_user],
        ['Z Step', z_step]
    ]

    # Pad header rows to match the number of columns in the data
    num_cols = len(summary_df.columns)
    for row in header_rows:
        while len(row) < num_cols:
            row.append('')

    return header_rows


def save_with_metaxpress_header(summary_df: pd.DataFrame, output_path: str, plate_metadata: Optional[Dict[str, str]] = None):
    """
    Save DataFrame with MetaXpress-style header structure.
    """
    # Create header rows
    header_rows = create_metaxpress_header(summary_df, plate_metadata)

    # Convert DataFrame to list of lists
    data_rows = []

    # Add column headers as a row
    data_rows.append(summary_df.columns.tolist())

    # Add data rows
    for _, row in summary_df.iterrows():
        data_rows.append(row.tolist())

    # Combine header + data
    all_rows = header_rows + data_rows

    # Write to CSV manually to preserve the exact structure
    with open(output_path, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row)


def auto_summarize_column(series: pd.Series, column_name: str, analysis_type: str) -> Dict[str, Any]:
    """
    Automatically summarize a pandas series with MetaXpress-style naming.

    Returns a dictionary of summary statistics with clean, descriptive names.
    """
    summary = {}

    # Handle empty series
    if len(series) == 0:
        return {}

    # Remove NaN values for analysis
    clean_series = series.dropna()

    if len(clean_series) == 0:
        return {}

    # Create clean analysis type name for grouping
    clean_analysis = analysis_type.replace('_', ' ').title()

    # Create meaningful metric names based on column content
    if pd.api.types.is_numeric_dtype(clean_series):
        # Numeric data - focus on key metrics like MetaXpress
        if 'count' in column_name.lower() or 'total' in column_name.lower():
            # Count/total metrics
            summary[f"Total {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.sum()
            summary[f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.mean()

        elif 'area' in column_name.lower():
            # Area metrics
            summary[f"Total {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.sum()
            summary[f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.mean()

        elif 'length' in column_name.lower() or 'distance' in column_name.lower():
            # Length/distance metrics
            summary[f"Total {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.sum()
            summary[f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.mean()

        elif 'intensity' in column_name.lower():
            # Intensity metrics
            summary[f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.mean()

        elif 'confidence' in column_name.lower():
            # Confidence metrics
            summary[f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.mean()

        else:
            # Generic numeric metrics
            summary[f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"] = clean_series.mean()

    elif clean_series.dtype == bool or set(clean_series.unique()).issubset({0, 1, True, False}):
        # Boolean data
        true_count = clean_series.sum()
        total_count = len(clean_series)
        summary[f"Count {column_name.replace('_', ' ').title()} ({clean_analysis})"] = true_count
        summary[f"% {column_name.replace('_', ' ').title()} ({clean_analysis})"] = (true_count / total_count) * 100

    else:
        # Categorical/string data - only include if meaningful
        unique_values = clean_series.unique()
        if len(unique_values) <= 5:  # Only include if not too many categories
            value_counts = clean_series.value_counts()
            most_common = value_counts.index[0] if len(value_counts) > 0 else None
            summary[f"Primary {column_name.replace('_', ' ').title()} ({clean_analysis})"] = most_common

    return summary


def summarize_analysis_file(file_path: str, analysis_type: str) -> Dict[str, Any]:
    """
    Summarize a single analysis CSV file with MetaXpress-style metrics.

    Returns a dictionary of key summary statistics with clean names.
    """
    try:
        df = pd.read_csv(file_path)

        if df.empty:
            logger.warning(f"Empty CSV file: {file_path}")
            return {}

        summary = {}
        clean_analysis = analysis_type.replace('_', ' ').title()

        # Add key file-level metrics first
        summary[f"Number of Objects ({clean_analysis})"] = len(df)

        # Prioritize important columns based on common analysis patterns
        priority_columns = []
        other_columns = []

        for column in df.columns:
            # Skip common index/ID columns
            if column.lower() in ['index', 'unnamed: 0', 'slice_index', 'cell_id', 'match_id', 'skeleton_id']:
                continue

            # Prioritize key metrics
            col_lower = column.lower()
            if any(key in col_lower for key in ['area', 'count', 'length', 'distance', 'intensity', 'confidence', 'branch']):
                priority_columns.append(column)
            else:
                other_columns.append(column)

        # Process priority columns first
        for column in priority_columns:
            col_summary = auto_summarize_column(df[column], column, analysis_type)
            summary.update(col_summary)

        # Process other columns but limit to avoid too many metrics
        for column in other_columns[:5]:  # Limit to 5 additional columns
            col_summary = auto_summarize_column(df[column], column, analysis_type)
            summary.update(col_summary)

        return summary

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return {}


def consolidate_analysis_results(
    results_directory: str,
    well_ids: List[str],
    consolidation_config: 'AnalysisConsolidationConfig',
    plate_metadata_config: 'PlateMetadataConfig',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Consolidate analysis results into a single summary table using configuration objects.

    Args:
        results_directory: Directory containing analysis CSV files
        consolidation_config: Configuration for consolidation behavior
        plate_metadata_config: Configuration for plate metadata
        output_path: Optional path to save consolidated CSV

    Returns:
        DataFrame with wells as rows and analysis metrics as columns
    """
    results_dir = Path(results_directory)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_directory}")

    logger.info(f"Consolidating analysis results from: {results_directory}")

    # Debug config objects
    logger.info(f"DEBUG: consolidation_config type: {type(consolidation_config)}")
    logger.info(f"DEBUG: well_pattern: {repr(consolidation_config.well_pattern)}")
    logger.info(f"DEBUG: file_extensions: {repr(consolidation_config.file_extensions)}")
    logger.info(f"DEBUG: exclude_patterns: {repr(consolidation_config.exclude_patterns)}")

    # Find all relevant files
    all_files = []
    for ext in consolidation_config.file_extensions:
        pattern = f"*{ext}"
        files = list(results_dir.glob(pattern))
        all_files.extend([str(f) for f in files])

    logger.info(f"Found {len(all_files)} files with extensions {consolidation_config.file_extensions}")

    # Apply exclude filters
    if consolidation_config.exclude_patterns:
        # Handle case where exclude_patterns might be a string representation
        exclude_patterns = consolidation_config.exclude_patterns
        if isinstance(exclude_patterns, str):
            # If it's a string representation of a tuple, convert it back
            import ast
            logger.info(f"DEBUG: exclude_patterns is string: {repr(exclude_patterns)}")
            try:
                exclude_patterns = ast.literal_eval(exclude_patterns)
                logger.info(f"DEBUG: Successfully parsed to: {repr(exclude_patterns)}")
            except Exception as e:
                logger.warning(f"Could not parse exclude_patterns string: {exclude_patterns}, error: {e}")
                exclude_patterns = []

        filtered_files = []
        for file_path in all_files:
            filename = Path(file_path).name
            if not any(re.search(pattern, filename) for pattern in exclude_patterns):
                filtered_files.append(file_path)
        all_files = filtered_files
        logger.info(f"After filtering: {len(all_files)} files to process")
    
    # Group files by well ID and analysis type
    wells_data = {}
    analysis_types = set()
    
    for file_path in all_files:
        filename = Path(file_path).name

        # Find well ID by substring matching (much more robust than regex)
        well_id = None
        for candidate_well in well_ids:
            if candidate_well in filename:
                well_id = candidate_well
                break

        if not well_id:
            logger.warning(f"Could not find any well ID from {well_ids} in filename {filename}, skipping")
            continue

        analysis_type = extract_analysis_type(filename, well_id)
        analysis_types.add(analysis_type)

        if well_id not in wells_data:
            wells_data[well_id] = {}

        wells_data[well_id][analysis_type] = file_path
    
    logger.info(f"Processing {len(wells_data)} wells with analysis types: {sorted(analysis_types)}")
    
    # Process each well and create summary
    summary_rows = []

    for well_id in sorted(wells_data.keys()):
        # Always use a consistent well ID column name
        well_summary = {'Well': well_id}

        # Process each analysis type for this well
        for analysis_type in sorted(analysis_types):
            if analysis_type in wells_data[well_id]:
                file_path = wells_data[well_id][analysis_type]
                analysis_summary = summarize_analysis_file(file_path, analysis_type)
                well_summary.update(analysis_summary)

        summary_rows.append(well_summary)

    # Create DataFrame
    summary_df = pd.DataFrame(summary_rows)

    if consolidation_config.metaxpress_style:
        # MetaXpress-style column ordering: Well first, then grouped by analysis type
        # Group columns by analysis type (text in parentheses)
        analysis_groups = {}
        other_cols = []

        for col in summary_df.columns:
            if col == 'Well':
                continue
            if '(' in col and ')' in col:
                analysis_name = col.split('(')[-1].replace(')', '')
                if analysis_name not in analysis_groups:
                    analysis_groups[analysis_name] = []
                analysis_groups[analysis_name].append(col)
            else:
                other_cols.append(col)

        # Reorder columns: Well first, then grouped by analysis type
        ordered_cols = ['Well']
        for analysis_name in sorted(analysis_groups.keys()):
            ordered_cols.extend(sorted(analysis_groups[analysis_name]))
        ordered_cols.extend(sorted(other_cols))

        summary_df = summary_df[ordered_cols]
    else:
        # Original style: sort all columns alphabetically
        if 'Well' in summary_df.columns:
            other_cols = [col for col in summary_df.columns if col != 'Well']
            summary_df = summary_df[['Well'] + sorted(other_cols)]
    
    logger.info(f"Created summary table with {len(summary_df)} wells and {len(summary_df.columns)} metrics")
    
    # Save to CSV if output path specified
    if output_path is None:
        output_path = results_dir / consolidation_config.output_filename

    if consolidation_config.metaxpress_style:
        # Create plate metadata dictionary from config
        plate_metadata = {
            'barcode': plate_metadata_config.barcode or f"OpenHCS-{results_dir.name}",
            'plate_name': plate_metadata_config.plate_name or results_dir.name,
            'plate_id': plate_metadata_config.plate_id or str(hash(str(results_dir)) % 100000),
            'description': plate_metadata_config.description or f"Consolidated analysis results from OpenHCS pipeline: {len(summary_df)} wells analyzed",
            'acquisition_user': plate_metadata_config.acquisition_user,
            'z_step': plate_metadata_config.z_step
        }

        save_with_metaxpress_header(summary_df, output_path, plate_metadata)
        logger.info(f"Saved MetaXpress-style summary with header to: {output_path}")
    else:
        summary_df.to_csv(output_path, index=False)
        logger.info(f"Saved consolidated summary to: {output_path}")

    return summary_df


def materialize_consolidated_results(
    data: pd.DataFrame,
    output_path: str,
    filemanager,
    well_id: str
) -> str:
    """Materialize consolidated results DataFrame to CSV using OpenHCS FileManager."""
    try:
        csv_content = data.to_csv(index=False)
        
        # Remove existing file if present
        if filemanager.exists(output_path, Backend.DISK.value):
            filemanager.delete(output_path, Backend.DISK.value)
        
        filemanager.save(csv_content, output_path, Backend.DISK.value)
        logger.info(f"Materialized consolidated results to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to materialize consolidated results: {e}")
        raise


@numpy_func
@special_outputs(("consolidated_results", materialize_consolidated_results))
def consolidate_analysis_results_pipeline(
    image_stack: np.ndarray,
    results_directory: str,
    consolidation_config: 'AnalysisConsolidationConfig',
    plate_metadata_config: 'PlateMetadataConfig'
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Pipeline-compatible version of consolidate_analysis_results.
    
    This function can be used as a FunctionStep in OpenHCS pipelines.
    """
    # Call the main consolidation function
    summary_df = consolidate_analysis_results(
        results_directory=results_directory,
        consolidation_config=consolidation_config,
        plate_metadata_config=plate_metadata_config,
        output_path=None  # Will be handled by materialization
    )
    
    return image_stack, summary_df
