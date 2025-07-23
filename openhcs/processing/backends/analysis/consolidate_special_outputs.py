"""
Generic consolidation of OpenHCS special outputs into summary tables.

This module provides a generic function for consolidating any CSV-based special outputs
from OpenHCS analysis pipelines into summary tables. It automatically detects well-based
naming patterns and creates comprehensive summary statistics.

Follows OpenHCS architectural principles:
- Uses FileManager for all I/O operations
- Proper memory type decorators
- Special I/O integration
- Fail-loud behavior
- Stateless design
"""

import numpy as np
import pandas as pd
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import Enum

from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.pipeline.function_contracts import special_outputs, special_inputs
from openhcs.constants.constants import Backend

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Aggregation strategies for different data types."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    MIXED = "mixed"


class WellPatternType(Enum):
    """Common well ID patterns for different plate formats."""
    STANDARD_96 = r"([A-H]\d{2})"  # A01, B02, etc.
    STANDARD_384 = r"([A-P]\d{2})"  # A01-P24
    CUSTOM = "custom"


def materialize_consolidated_summary(
    data: Dict[str, Any],
    output_path: str,
    filemanager,
    well_id: str
) -> str:
    """
    Materialize consolidated summary data to CSV file.
    
    Args:
        data: Dictionary containing consolidated summary data
        output_path: Path where CSV should be saved
        filemanager: OpenHCS FileManager instance
        well_id: Well identifier
        
    Returns:
        Path to saved CSV file
    """
    try:
        # Convert to DataFrame
        if 'summary_table' in data:
            df = pd.DataFrame(data['summary_table'])
        else:
            # Fallback: create DataFrame from raw data
            df = pd.DataFrame([data])
        
        # Generate CSV content
        csv_content = df.to_csv(index=False)
        
        # Save using FileManager (remove existing first if present)
        if filemanager.exists(output_path, Backend.DISK.value):
            filemanager.delete(output_path, Backend.DISK.value)
        
        filemanager.save(csv_content, output_path, Backend.DISK.value)
        
        logger.info(f"Materialized consolidated summary to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to materialize consolidated summary: {e}")
        raise


def materialize_detailed_report(
    data: Dict[str, Any],
    output_path: str,
    filemanager,
    well_id: str
) -> str:
    """
    Materialize detailed analysis report to text file.
    
    Args:
        data: Dictionary containing analysis data
        output_path: Path where report should be saved
        filemanager: OpenHCS FileManager instance
        well_id: Well identifier
        
    Returns:
        Path to saved report file
    """
    try:
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("OPENHCS SPECIAL OUTPUTS CONSOLIDATION REPORT")
        report_lines.append("="*80)
        
        if 'metadata' in data:
            metadata = data['metadata']
            report_lines.append(f"Analysis timestamp: {metadata.get('timestamp', 'Unknown')}")
            report_lines.append(f"Total wells processed: {metadata.get('total_wells', 0)}")
            report_lines.append(f"Output types detected: {metadata.get('output_types', [])}")
            report_lines.append("")
        
        if 'summary_stats' in data:
            stats = data['summary_stats']
            report_lines.append("SUMMARY STATISTICS:")
            report_lines.append("-" * 40)
            for output_type, type_stats in stats.items():
                report_lines.append(f"\n{output_type.upper()}:")
                for metric, value in type_stats.items():
                    if isinstance(value, float):
                        report_lines.append(f"  {metric}: {value:.3f}")
                    else:
                        report_lines.append(f"  {metric}: {value}")
        
        report_lines.append("\n" + "="*80)
        
        # Save report
        report_content = "\n".join(report_lines)
        
        if filemanager.exists(output_path, Backend.DISK.value):
            filemanager.delete(output_path, Backend.DISK.value)
        
        filemanager.save(report_content, output_path, Backend.DISK.value)
        
        logger.info(f"Materialized detailed report to {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to materialize detailed report: {e}")
        raise


def extract_well_id(filename: str, pattern: str = WellPatternType.STANDARD_96.value) -> Optional[str]:
    """
    Extract well ID from filename using regex pattern.
    
    Args:
        filename: Name of the file
        pattern: Regex pattern for well ID extraction
        
    Returns:
        Well ID if found, None otherwise
    """
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def detect_aggregation_strategy(series: pd.Series) -> AggregationStrategy:
    """
    Automatically detect the appropriate aggregation strategy for a data series.
    
    Args:
        series: Pandas series to analyze
        
    Returns:
        Appropriate aggregation strategy
    """
    # Check if boolean
    if series.dtype == bool or set(series.dropna().unique()).issubset({0, 1, True, False}):
        return AggregationStrategy.BOOLEAN
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return AggregationStrategy.NUMERIC
    
    # Check if categorical (string/object with limited unique values)
    unique_ratio = len(series.unique()) / len(series)
    if unique_ratio < 0.5:  # Less than 50% unique values suggests categorical
        return AggregationStrategy.CATEGORICAL
    
    return AggregationStrategy.MIXED


def aggregate_series(series: pd.Series, strategy: AggregationStrategy) -> Dict[str, Any]:
    """
    Aggregate a pandas series based on the specified strategy.
    
    Args:
        series: Series to aggregate
        strategy: Aggregation strategy to use
        
    Returns:
        Dictionary of aggregated statistics
    """
    result = {}
    
    if strategy == AggregationStrategy.NUMERIC:
        result.update({
            'count': len(series),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'sum': series.sum(),
            'median': series.median()
        })
    
    elif strategy == AggregationStrategy.BOOLEAN:
        result.update({
            'count': len(series),
            'true_count': series.sum(),
            'false_count': len(series) - series.sum(),
            'true_percentage': (series.sum() / len(series)) * 100
        })
    
    elif strategy == AggregationStrategy.CATEGORICAL:
        value_counts = series.value_counts()
        result.update({
            'count': len(series),
            'unique_values': len(series.unique()),
            'most_common': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_common_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'unique_values_list': ','.join(map(str, series.unique()))
        })
    
    else:  # MIXED
        result.update({
            'count': len(series),
            'unique_values': len(series.unique()),
            'data_type': str(series.dtype)
        })
    
    return result


@numpy_func
@special_outputs(
    ("consolidated_summary", materialize_consolidated_summary),
    ("detailed_report", materialize_detailed_report)
)
def consolidate_special_outputs(
    image_stack: np.ndarray,
    results_directory: str,
    well_pattern: str = WellPatternType.STANDARD_96.value,
    file_extensions: List[str] = [".csv"],
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    custom_aggregations: Optional[Dict[str, Dict[str, str]]] = None
) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    """
    Consolidate special outputs from OpenHCS analysis into summary tables.
    
    This function automatically detects CSV files with well-based naming patterns,
    groups them by output type, and creates comprehensive summary statistics.
    
    Args:
        image_stack: Input image stack (dummy for OpenHCS compatibility)
        results_directory: Directory containing special output files
        well_pattern: Regex pattern for extracting well IDs
        file_extensions: List of file extensions to process
        include_patterns: Optional list of filename patterns to include
        exclude_patterns: Optional list of filename patterns to exclude
        custom_aggregations: Optional custom aggregation rules per output type
        
    Returns:
        Tuple of (image_stack, consolidated_summary, detailed_report)
    """
    from openhcs.io.filemanager import FileManager
    from openhcs.io.base import storage_registry
    from datetime import datetime
    
    # Initialize FileManager
    filemanager = FileManager(storage_registry)
    
    logger.info(f"Consolidating special outputs from: {results_directory}")
    
    # Find all relevant files
    all_files = []
    for ext in file_extensions:
        pattern = f"*{ext}"
        files = filemanager.list_files(results_directory, Backend.DISK.value, pattern=pattern, recursive=False)
        all_files.extend(files)
    
    logger.info(f"Found {len(all_files)} files with extensions {file_extensions}")
    
    # Apply include/exclude filters
    if include_patterns:
        all_files = [f for f in all_files if any(re.search(pattern, Path(f).name) for pattern in include_patterns)]
    
    if exclude_patterns:
        all_files = [f for f in all_files if not any(re.search(pattern, Path(f).name) for pattern in exclude_patterns)]
    
    logger.info(f"After filtering: {len(all_files)} files to process")
    
    # Group files by well ID and output type
    wells_data = {}
    output_types = set()
    
    for file_path in all_files:
        filename = Path(file_path).name
        well_id = extract_well_id(filename, well_pattern)
        
        if not well_id:
            logger.warning(f"Could not extract well ID from {filename}, skipping")
            continue
        
        # Extract output type (everything after well ID and before extension)
        output_type = filename.replace(f"{well_id}_", "").replace(Path(filename).suffix, "")
        output_types.add(output_type)
        
        if well_id not in wells_data:
            wells_data[well_id] = {}
        
        wells_data[well_id][output_type] = file_path
    
    logger.info(f"Processing {len(wells_data)} wells with output types: {sorted(output_types)}")
    
    # Process each output type and create summary statistics
    summary_table = []
    summary_stats = {}
    
    for output_type in sorted(output_types):
        logger.info(f"Processing output type: {output_type}")
        
        # Collect data for this output type across all wells
        type_data = []
        wells_with_type = []
        
        for well_id, well_files in wells_data.items():
            if output_type in well_files:
                try:
                    file_path = well_files[output_type]
                    df = pd.read_csv(file_path)
                    
                    # Create well-level summary
                    well_summary = {'well_id': well_id, 'output_type': output_type}
                    
                    # Aggregate each column
                    for col in df.columns:
                        if col in ['well_id', 'output_type']:
                            continue
                        
                        strategy = detect_aggregation_strategy(df[col])
                        col_stats = aggregate_series(df[col], strategy)
                        
                        # Prefix column stats with column name
                        for stat_name, stat_value in col_stats.items():
                            well_summary[f"{col}_{stat_name}"] = stat_value
                    
                    type_data.append(well_summary)
                    wells_with_type.append(well_id)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
        
        # Add to summary table
        summary_table.extend(type_data)
        
        # Create type-level statistics
        if type_data:
            type_df = pd.DataFrame(type_data)
            type_stats = {
                'wells_count': len(wells_with_type),
                'wells_list': ','.join(sorted(wells_with_type))
            }
            
            # Add aggregate statistics for numeric columns
            numeric_cols = type_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'well_id':
                    type_stats[f"{col}_mean"] = type_df[col].mean()
                    type_stats[f"{col}_std"] = type_df[col].std()
            
            summary_stats[output_type] = type_stats
    
    # Create consolidated summary
    consolidated_summary = {
        'summary_table': summary_table,
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_wells': len(wells_data),
            'output_types': sorted(output_types),
            'total_files_processed': len(all_files),
            'well_pattern': well_pattern
        }
    }
    
    # Create detailed report
    detailed_report = {
        'summary_stats': summary_stats,
        'metadata': consolidated_summary['metadata']
    }
    
    logger.info(f"Consolidation complete: {len(summary_table)} well-output combinations processed")
    
    return image_stack, consolidated_summary, detailed_report
