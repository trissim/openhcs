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

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openhcs.core.memory import numpy as numpy_func
from openhcs.core.config import (
    AnalysisConsolidationConfig,
    PlateMetadataConfig,
)
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.core.vfs_protocol import PlateInputDirectory
from openhcs.processing.materialization import CsvOptions, MaterializationSpec

if TYPE_CHECKING:
    from openhcs.microscopes.microscope_interfaces import FilenameParser

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnalysisTableRecord:
    """One typed analysis table ready for per-well consolidation."""

    well_id: str
    analysis_type: str
    table: pd.DataFrame
    source_name: str

    def __post_init__(self) -> None:
        if not self.well_id:
            raise ValueError("AnalysisTableRecord.well_id cannot be empty.")
        if not self.analysis_type:
            raise ValueError("AnalysisTableRecord.analysis_type cannot be empty.")
        if not self.source_name:
            raise ValueError("AnalysisTableRecord.source_name cannot be empty.")
        if not isinstance(self.table, pd.DataFrame):
            raise TypeError(
                "AnalysisTableRecord.table must be a pandas DataFrame, got "
                f"{type(self.table).__name__}."
            )


class AnalysisTableSource(ABC):
    """Nominal source of analysis tables independent of storage backend."""

    @abstractmethod
    def records(self) -> tuple[AnalysisTableRecord, ...]:
        """Return typed analysis table records."""


class AnalysisWellResolver(ABC):
    """Semantic authority for resolving a well component from an analysis file."""

    @abstractmethod
    def well_id_for(self, filename: str) -> str | None:
        """Return the well component encoded by an analysis filename."""


@dataclass(frozen=True, slots=True)
class FilenameParserWellResolver(AnalysisWellResolver):
    """Resolve wells through an OpenHCS microscope filename parser."""

    filename_parser: "FilenameParser"

    def well_id_for(self, filename: str) -> str | None:
        parsed = self.filename_parser.parse_filename(filename)
        if parsed is None:
            return None
        value = parsed.get("well")
        if value is None:
            return None
        return str(value)


@dataclass(frozen=True, slots=True)
class AutoDetectFilenameParserWellResolver(AnalysisWellResolver):
    """Resolve wells using the registered OpenHCS filename parser family."""

    parser_types: tuple[type["FilenameParser"], ...]

    @classmethod
    def from_registered_parsers(cls) -> "AutoDetectFilenameParserWellResolver":
        from openhcs.microscopes.microscope_interfaces import FilenameParser

        return cls(tuple(FilenameParser.__registry__.values()))

    def well_id_for(self, filename: str) -> str | None:
        for parser_type in self.parser_types:
            if not parser_type.can_parse(filename):
                continue
            return FilenameParserWellResolver(parser_type()).well_id_for(filename)
        return None


@dataclass(frozen=True, slots=True)
class ConfiguredWellListResolver(AnalysisWellResolver):
    """Resolve wells from an explicit caller-provided legacy well set."""

    well_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "well_ids", tuple(self.well_ids))

    def well_id_for(self, filename: str) -> str | None:
        filename_lower = filename.lower()
        for candidate_well in self.well_ids:
            if candidate_well.lower() in filename_lower:
                return candidate_well
        return None


@dataclass(frozen=True, slots=True)
class CsvAnalysisTableSource(AnalysisTableSource):
    """Analysis table source backed by an explicit materialized-file set."""

    file_paths: tuple[Path, ...]
    well_resolver: AnalysisWellResolver
    analysis_consolidation_config: "AnalysisConsolidationConfig"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "file_paths",
            tuple(Path(file_path) for file_path in self.file_paths),
        )
        if not isinstance(self.well_resolver, AnalysisWellResolver):
            raise TypeError(
                "CsvAnalysisTableSource.well_resolver must be "
                "AnalysisWellResolver."
            )

    def records(self) -> tuple[AnalysisTableRecord, ...]:
        return tuple(
            record
            for file_path in self.included_file_paths()
            for record in self._record_for_file(file_path)
        )

    def included_file_paths(self) -> tuple[Path, ...]:
        """Return explicit paths admitted by the consolidation configuration."""

        return tuple(
            file_path for file_path in self.file_paths if self._includes(file_path)
        )

    def _includes(self, file_path: Path) -> bool:
        if file_path.suffix not in self.analysis_consolidation_config.file_extensions:
            return False
        exclude_patterns = _exclude_patterns(self.analysis_consolidation_config)
        return not any(
            re.search(pattern, file_path.name) for pattern in exclude_patterns
        )

    def _record_for_file(
        self,
        file_path: Path,
    ) -> tuple[AnalysisTableRecord, ...]:
        well_id = self.well_resolver.well_id_for(file_path.name)
        if well_id is None:
            logger.warning(
                "Could not resolve well component from filename %s, skipping",
                file_path.name,
            )
            return ()
        return (
            AnalysisTableRecord(
                well_id=well_id,
                analysis_type=extract_analysis_type(file_path.name, well_id),
                table=pd.read_csv(file_path),
                source_name=str(file_path),
            ),
        )


def discover_analysis_file_paths(
    results_directory: Path,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
) -> tuple[Path, ...]:
    """Discover configured files for an explicitly requested directory."""

    results_directory = Path(results_directory)
    if not results_directory.exists():
        raise FileNotFoundError(
            f"Results directory does not exist: {results_directory}"
        )
    return tuple(
        file_path
        for extension in analysis_consolidation_config.file_extensions
        for file_path in results_directory.glob(f"*{extension}")
    )


def _exclude_patterns(
    analysis_consolidation_config: "AnalysisConsolidationConfig",
) -> tuple[str, ...]:
    """Return typed exclude regex patterns from consolidation config."""
    patterns = analysis_consolidation_config.exclude_patterns
    if patterns is None:
        return ()
    if isinstance(patterns, str):
        raise TypeError(
            "AnalysisConsolidationConfig.exclude_patterns must be a sequence "
            "of regex strings, not a string."
        )
    return tuple(str(pattern) for pattern in patterns)


def extract_analysis_type(filename: str, well_id: str) -> str:
    """Extract analysis type from filename after removing well ID and extension.

    Handles both ImageXpress (A01_cell_counts_step2.csv) and Opera Phenix
    (r02c02f001p001-ch1sk1fk1fl1_match_results_step3.csv) formats.
    """
    # Remove file extension first
    name_without_ext = filename.replace(Path(filename).suffix, "")

    # Find the well ID in the filename (case-insensitive)
    well_lower = well_id.lower()
    name_lower = name_without_ext.lower()

    if well_lower in name_lower:
        # Find position after well ID
        well_end_pos = name_lower.index(well_lower) + len(well_lower)
        # Get everything after the well ID
        after_well = name_without_ext[well_end_pos:]

        # Find first underscore (analysis type starts after this)
        if "_" in after_well:
            # Remove everything up to and including first underscore
            analysis_type = after_well[after_well.index("_") + 1 :]
        else:
            # No underscore found, use everything after well ID
            analysis_type = after_well
    else:
        # Fallback: use entire filename without extension
        analysis_type = name_without_ext

    return analysis_type


def create_metaxpress_header(
    summary_df: pd.DataFrame,
    plate_metadata: Mapping[str, str] | None = None,
) -> list[list[str]]:
    """
    Create MetaXpress-style header rows with metadata.

    Returns list of header rows to prepend to the CSV.
    """
    if plate_metadata is None:
        plate_metadata = {}

    # Extract plate info from results directory or use defaults
    barcode = plate_metadata.get("barcode", "OpenHCS-Plate")
    plate_name = plate_metadata.get("plate_name", "OpenHCS Analysis Results")
    plate_id = plate_metadata.get("plate_id", "00000")
    description = plate_metadata.get(
        "description", "Consolidated analysis results from OpenHCS pipeline"
    )
    acquisition_user = plate_metadata.get("acquisition_user", "OpenHCS")
    z_step = plate_metadata.get("z_step", "1")

    # Create header rows matching MetaXpress format
    header_rows = [
        ["Barcode", barcode],
        ["Plate Name", plate_name],
        ["Plate ID", plate_id],
        ["Description", description],
        ["Acquisition User", acquisition_user],
        ["Z Step", z_step],
    ]

    # Pad header rows to match the number of columns in the data
    num_cols = len(summary_df.columns)
    for row in header_rows:
        while len(row) < num_cols:
            row.append("")

    return header_rows


def save_with_metaxpress_header(
    summary_df: pd.DataFrame,
    output_path: str,
    plate_metadata: Mapping[str, str] | None = None,
) -> None:
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
    with open(output_path, "w", newline="") as f:
        import csv

        writer = csv.writer(f)
        for row in all_rows:
            writer.writerow(row)


def auto_summarize_column(
    series: pd.Series,
    column_name: str,
    analysis_type: str,
) -> dict[str, object]:
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
    clean_analysis = analysis_type.replace("_", " ").title()

    # Create meaningful metric names based on column content
    if pd.api.types.is_numeric_dtype(clean_series):
        # Numeric data - focus on key metrics like MetaXpress
        if "count" in column_name.lower() or "total" in column_name.lower():
            # Count/total metrics
            summary[
                f"Total {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.sum()
            summary[
                f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.mean()

        elif "area" in column_name.lower():
            # Area metrics
            summary[
                f"Total {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.sum()
            summary[
                f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.mean()

        elif "length" in column_name.lower() or "distance" in column_name.lower():
            # Length/distance metrics
            summary[
                f"Total {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.sum()
            summary[
                f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.mean()

        elif "intensity" in column_name.lower():
            # Intensity metrics
            summary[
                f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.mean()

        elif "confidence" in column_name.lower():
            # Confidence metrics
            summary[
                f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.mean()

        else:
            # Generic numeric metrics
            summary[
                f"Mean {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = clean_series.mean()

    elif clean_series.dtype == bool or set(clean_series.unique()).issubset(
        {0, 1, True, False}
    ):
        # Boolean data
        true_count = clean_series.sum()
        total_count = len(clean_series)
        summary[f"Count {column_name.replace('_', ' ').title()} ({clean_analysis})"] = (
            true_count
        )
        summary[f"% {column_name.replace('_', ' ').title()} ({clean_analysis})"] = (
            true_count / total_count
        ) * 100

    else:
        # Categorical/string data - only include if meaningful
        unique_values = clean_series.unique()
        if len(unique_values) <= 5:  # Only include if not too many categories
            value_counts = clean_series.value_counts()
            most_common = value_counts.index[0] if len(value_counts) > 0 else None
            summary[
                f"Primary {column_name.replace('_', ' ').title()} ({clean_analysis})"
            ] = most_common

    return summary


def summarize_analysis_table(
    df: pd.DataFrame,
    analysis_type: str,
    *,
    source_name: str,
) -> dict[str, object]:
    """
    Summarize a single analysis table with MetaXpress-style metrics.

    Returns a dictionary of key summary statistics with clean names.
    """
    if df.empty:
        logger.warning("Empty analysis table: %s", source_name)
        return {}

    summary: dict[str, object] = {}
    clean_analysis = analysis_type.replace("_", " ").title()

    summary[f"Number of Objects ({clean_analysis})"] = len(df)

    priority_columns: list[str] = []
    other_columns: list[str] = []

    for column in df.columns:
        if column.lower() in {
            "index",
            "unnamed: 0",
            "slice_index",
            "cell_id",
            "match_id",
            "skeleton_id",
        }:
            continue

        col_lower = column.lower()
        if any(
            key in col_lower
            for key in (
                "area",
                "count",
                "length",
                "distance",
                "intensity",
                "confidence",
                "branch",
            )
        ):
            priority_columns.append(column)
        else:
            other_columns.append(column)

    for column in priority_columns:
        summary.update(auto_summarize_column(df[column], column, analysis_type))

    for column in other_columns[:5]:
        summary.update(auto_summarize_column(df[column], column, analysis_type))

    return summary


def summarize_analysis_file(
    file_path: str,
    analysis_type: str,
) -> dict[str, object]:
    """Summarize one materialized CSV analysis file."""
    return summarize_analysis_table(
        pd.read_csv(file_path),
        analysis_type,
        source_name=file_path,
    )


def consolidate_analysis_table_records(
    records: tuple[AnalysisTableRecord, ...],
    analysis_consolidation_config: "AnalysisConsolidationConfig",
) -> pd.DataFrame:
    """Create the per-well summary table from typed analysis records."""
    records_by_well: dict[str, dict[str, AnalysisTableRecord]] = {}
    analysis_types: set[str] = set()
    for record in records:
        analysis_types.add(record.analysis_type)
        records_by_well.setdefault(record.well_id, {})[record.analysis_type] = record

    logger.info(
        "Processing %d wells with analysis types: %s",
        len(records_by_well),
        sorted(analysis_types),
    )

    summary_rows: list[dict[str, object]] = []
    for well_id in sorted(records_by_well):
        well_summary: dict[str, object] = {"Well": well_id}
        for analysis_type in sorted(analysis_types):
            record = records_by_well[well_id].get(analysis_type)
            if record is None:
                continue
            well_summary.update(
                summarize_analysis_table(
                    record.table,
                    record.analysis_type,
                    source_name=record.source_name,
                )
            )
        summary_rows.append(well_summary)

    return order_consolidated_summary_columns(
        pd.DataFrame(summary_rows),
        metaxpress_style=analysis_consolidation_config.metaxpress_style,
    )


def order_consolidated_summary_columns(
    summary_df: pd.DataFrame,
    *,
    metaxpress_style: bool,
) -> pd.DataFrame:
    """Apply stable OpenHCS/MetaXpress column ordering."""
    if summary_df.empty:
        return summary_df
    if metaxpress_style:
        analysis_groups: dict[str, list[str]] = {}
        other_cols: list[str] = []
        for column in summary_df.columns:
            if column == "Well":
                continue
            if "(" in column and ")" in column:
                analysis_name = column.split("(")[-1].replace(")", "")
                analysis_groups.setdefault(analysis_name, []).append(column)
                continue
            other_cols.append(column)

        ordered_cols = ["Well"]
        for analysis_name in sorted(analysis_groups):
            ordered_cols.extend(sorted(analysis_groups[analysis_name]))
        ordered_cols.extend(sorted(other_cols))
        return summary_df[ordered_cols]

    if "Well" not in summary_df.columns:
        return summary_df.reindex(sorted(summary_df.columns), axis=1)
    other_cols = [column for column in summary_df.columns if column != "Well"]
    return summary_df[["Well", *sorted(other_cols)]]


def consolidated_plate_metadata(
    results_dir: Path,
    summary_df: pd.DataFrame,
    plate_metadata_config: "PlateMetadataConfig",
) -> dict[str, str]:
    """Return MetaXpress-compatible metadata for a consolidated summary."""
    return {
        "barcode": plate_metadata_config.barcode or f"OpenHCS-{results_dir.name}",
        "plate_name": plate_metadata_config.plate_name or results_dir.name,
        "plate_id": plate_metadata_config.plate_id
        or str(hash(str(results_dir)) % 100000),
        "description": plate_metadata_config.description
        or (
            "Consolidated analysis results from OpenHCS pipeline: "
            f"{len(summary_df)} wells analyzed"
        ),
        "acquisition_user": plate_metadata_config.acquisition_user,
        "z_step": plate_metadata_config.z_step,
    }


def write_consolidated_analysis_summary(
    summary_df: pd.DataFrame,
    output_path: str,
    results_dir: Path,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
) -> None:
    """Persist one consolidated analysis summary."""
    if analysis_consolidation_config.metaxpress_style:
        save_with_metaxpress_header(
            summary_df,
            output_path,
            consolidated_plate_metadata(
                results_dir,
                summary_df,
                plate_metadata_config,
            ),
        )
        logger.info("Saved MetaXpress-style summary with header to: %s", output_path)
        return

    summary_df.to_csv(output_path, index=False)
    logger.info("Saved consolidated summary to: %s", output_path)


def analysis_well_resolver(
    *,
    filename_parser: "FilenameParser | None" = None,
    well_ids: tuple[str, ...] = (),
) -> AnalysisWellResolver:
    """Return the explicit well resolver for an analysis consolidation run."""
    if filename_parser is not None:
        return FilenameParserWellResolver(filename_parser)
    if well_ids:
        return ConfiguredWellListResolver(well_ids)
    return AutoDetectFilenameParserWellResolver.from_registered_parsers()


def consolidate_analysis_results(
    results_directory: str,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
    *,
    well_ids: list[str] | None = None,
    output_path: str | None = None,
    filename_parser: "FilenameParser | None" = None,
) -> pd.DataFrame:
    """
    Consolidate analysis results into a single summary table using configuration objects.

    Args:
        results_directory: Directory containing analysis CSV files
        analysis_consolidation_config: Configuration for consolidation behavior
        plate_metadata_config: Configuration for plate metadata
        output_path: Optional path to save consolidated CSV

    Returns:
        DataFrame with wells as rows and analysis metrics as columns
    """
    results_dir = Path(results_directory)
    logger.info("Consolidating analysis results from: %s", results_dir)
    logger.debug(
        "analysis_consolidation_config type: %s",
        type(analysis_consolidation_config),
    )
    logger.debug("well_pattern: %r", analysis_consolidation_config.well_pattern)
    logger.debug("file_extensions: %r", analysis_consolidation_config.file_extensions)
    logger.debug("exclude_patterns: %r", analysis_consolidation_config.exclude_patterns)

    source = CsvAnalysisTableSource(
        file_paths=discover_analysis_file_paths(
            results_dir,
            analysis_consolidation_config,
        ),
        well_resolver=analysis_well_resolver(
            filename_parser=filename_parser,
            well_ids=tuple(well_ids or ()),
        ),
        analysis_consolidation_config=analysis_consolidation_config,
    )
    return consolidate_analysis_table_source(
        source,
        results_directory=results_dir,
        analysis_consolidation_config=analysis_consolidation_config,
        plate_metadata_config=plate_metadata_config,
        output_path=output_path,
    )


def consolidate_materialized_analysis_files(
    file_paths: tuple[Path, ...],
    results_directory: Path,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
    *,
    output_path: str | None = None,
    filename_parser: "FilenameParser",
) -> pd.DataFrame:
    """Consolidate one execution-owned set of materialized analysis files."""

    results_dir = Path(results_directory)
    source = CsvAnalysisTableSource(
        file_paths=file_paths,
        well_resolver=FilenameParserWellResolver(filename_parser),
        analysis_consolidation_config=analysis_consolidation_config,
    )
    return consolidate_analysis_table_source(
        source,
        results_directory=results_dir,
        analysis_consolidation_config=analysis_consolidation_config,
        plate_metadata_config=plate_metadata_config,
        output_path=output_path,
    )


def consolidate_analysis_table_source(
    source: AnalysisTableSource,
    *,
    results_directory: Path,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
    output_path: str | None,
) -> pd.DataFrame:
    """Consolidate a caller-selected typed analysis-table source."""

    results_dir = Path(results_directory)
    records = source.records()
    logger.info(
        "Found %d analysis table records from %s",
        len(records),
        results_dir,
    )
    summary_df = consolidate_analysis_table_records(
        records,
        analysis_consolidation_config,
    )
    logger.info(
        "Created summary table with %d wells and %d metrics",
        len(summary_df),
        len(summary_df.columns),
    )

    resolved_output_path = (
        output_path
        if output_path is not None
        else str(results_dir / analysis_consolidation_config.output_filename)
    )
    write_consolidated_analysis_summary(
        summary_df,
        resolved_output_path,
        results_dir,
        analysis_consolidation_config,
        plate_metadata_config,
    )
    return summary_df


## Greenfield: materialization is writer-driven (no custom materializers).


@numpy_func
@artifact_outputs(
    ("consolidated_results", MaterializationSpec(CsvOptions(filename_suffix=".csv")))
)
def consolidate_analysis_results_pipeline(
    image_stack: np.ndarray,
    results_directory: PlateInputDirectory,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Pipeline-compatible version of consolidate_analysis_results.

    This function can be used as a FunctionStep in OpenHCS pipelines.

    Args:
        results_directory: Plate-relative directory containing the analysis CSV
            files to combine into one well-level summary.
    """
    # Call the main consolidation function
    summary_df = consolidate_analysis_results(
        results_directory=results_directory,
        analysis_consolidation_config=analysis_consolidation_config,
        plate_metadata_config=plate_metadata_config,
        output_path=None,  # Will be handled by materialization
    )

    return image_stack, summary_df


def merge_result_type_summaries(
    summary_paths: list[str],
    output_path: str,
    plate_names: list[str] | None = None,
    plate_folder_name: str | None = None,
    plate_id: str | None = None,
) -> pd.DataFrame:
    """
    Merge multiple MetaXpress-style summaries from different result types within the SAME plate.

    This creates one row per well with all columns combined from different result directories
    (e.g., cellbody_results, images_results, axon_results).

    Args:
        summary_paths: List of paths to MetaXpress summary CSV files from different result types
        output_path: Path where the merged summary should be saved
        plate_names: Optional list of result type names (for logging)
        plate_folder_name: Optional plate folder name for header
        plate_id: Optional plate ID for header

    Returns:
        Merged DataFrame with one row per well
    """
    if not summary_paths:
        logger.warning("No summary paths provided for result type merging")
        return pd.DataFrame()

    logger.info(f"Merging {len(summary_paths)} result type summaries into single table")

    # Read all summaries and merge on Well (one row per well)
    merged_df = None
    for i, summary_path in enumerate(summary_paths):
        if not Path(summary_path).exists():
            logger.warning(f"Summary file not found: {summary_path}, skipping")
            continue

        try:
            # Read MetaXpress CSV, skipping the 6-line header
            df = pd.read_csv(summary_path, skiprows=6)
            result_type = (
                plate_names[i] if plate_names and i < len(plate_names) else f"type_{i}"
            )
            logger.info(f"Loaded {len(df)} rows from {result_type}")

            if merged_df is None:
                merged_df = df
            else:
                # Merge on Well - one row per well with all columns combined
                # Use outer join to keep all wells from all result types
                merged_df = merged_df.merge(
                    df, on="Well", how="outer", suffixes=("", "_dup")
                )

                # Drop duplicate columns (keep first occurrence)
                dup_cols = [col for col in merged_df.columns if col.endswith("_dup")]
                if dup_cols:
                    logger.info(
                        f"Dropping {len(dup_cols)} duplicate columns from merge"
                    )
                    merged_df = merged_df.drop(columns=dup_cols)

        except Exception as e:
            logger.error(f"Failed to read summary from {summary_path}: {e}")
            continue

    if merged_df is None:
        logger.error("No valid summaries could be loaded")
        return pd.DataFrame()

    logger.info(
        f"Merged into {len(merged_df)} unique wells with {len(merged_df.columns)} total columns"
    )

    # Create MetaXpress header for merged summary
    # Use plate folder name and plate ID if provided
    if plate_folder_name and plate_id:
        merged_metadata = {
            "barcode": f"OpenHCS-{plate_folder_name}",
            "plate_name": plate_folder_name,
            "plate_id": plate_id,
            "description": f"Merged analysis from {len(summary_paths)} result types: {', '.join(plate_names[:3]) if plate_names else 'unknown'}",
            "acquisition_user": "OpenHCS",
            "z_step": "1",
        }
    else:
        merged_metadata = {
            "barcode": f"OpenHCS-Merged-{len(summary_paths)}ResultTypes",
            "plate_name": f"Merged Analysis ({len(summary_paths)} result types)",
            "plate_id": str(hash(str(summary_paths)) % 100000),
            "description": f"Merged analysis from {len(summary_paths)} result types: {', '.join(plate_names[:3]) if plate_names else 'unknown'}",
            "acquisition_user": "OpenHCS",
            "z_step": "1",
        }

    # Save with MetaXpress header
    save_with_metaxpress_header(merged_df, output_path, merged_metadata)
    logger.info(f"Saved merged summary to: {output_path}")

    return merged_df


def consolidate_multi_plate_summaries(
    summary_paths: list[str],
    output_path: str,
    plate_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Consolidate multiple MetaXpress-style summaries from DIFFERENT plates into a single table.

    This function reads individual plate summaries and CONCATENATES them (stacks rows),
    keeping each plate's wells as separate rows. Never merges wells from different plates.

    Args:
        summary_paths: List of paths to individual plate MetaXpress summary CSV files
        output_path: Path where the global consolidated summary should be saved
        plate_names: Optional list of plate names (same length as summary_paths).
                    If None, plate names are extracted from the summary file paths.

    Returns:
        Combined DataFrame with all plates' data (rows stacked)

    Example:
        >>> summary_paths = [
        ...     "/data/plate1/global_metaxpress_summary.csv",
        ...     "/data/plate2/global_metaxpress_summary.csv"
        ... ]
        >>> df = consolidate_multi_plate_summaries(
        ...     summary_paths,
        ...     "/data/all_plates_summary.csv"
        ... )
    """
    if not summary_paths:
        logger.warning("No summary paths provided for multi-plate consolidation")
        return pd.DataFrame()

    # Generate plate names if not provided
    if plate_names is None:
        plate_names = []
        for path in summary_paths:
            # Extract plate name from path
            path_obj = Path(path)
            plate_dir = path_obj.parent.name
            plate_names.append(plate_dir)

    if len(plate_names) != len(summary_paths):
        raise ValueError(
            f"plate_names length ({len(plate_names)}) must match summary_paths length ({len(summary_paths)})"
        )

    logger.info(
        f"Concatenating {len(summary_paths)} plate summaries (different plates)"
    )

    # Read all summaries and CONCAT (stack rows - never merge different plates)
    combined_dfs = []
    for plate_name, summary_path in zip(plate_names, summary_paths):
        if not Path(summary_path).exists():
            logger.warning(f"Summary file not found: {summary_path}, skipping")
            continue

        try:
            # Read CSV (skip header if present, otherwise read as-is)
            try:
                # Try reading with MetaXpress header
                df = pd.read_csv(summary_path, skiprows=6)
            except Exception:
                # Fallback: read without skipping
                df = pd.read_csv(summary_path)

            logger.info(f"Loaded {len(df)} rows from {plate_name}")
            combined_dfs.append(df)

        except Exception as e:
            logger.error(f"Failed to read summary from {summary_path}: {e}")
            continue

    if not combined_dfs:
        logger.error("No valid summaries could be loaded")
        return pd.DataFrame()

    # CONCAT all DataFrames (stack rows, keep plates separate)
    result_df = pd.concat(combined_dfs, ignore_index=True)
    logger.info(
        f"Concatenated {len(combined_dfs)} plates into {len(result_df)} total rows"
    )

    # Save as simple CSV
    result_df.to_csv(output_path, index=False)
    logger.info(f"Saved concatenated summary to: {output_path}")

    return result_df


def consolidate_results_directories(
    results_dirs: list[Path],
    plate_path: Path,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
    filename_parser: "FilenameParser",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Discover configured files for an explicitly requested directory operation."""

    analysis_files_by_directory = {
        results_dir: discover_analysis_file_paths(
            results_dir,
            analysis_consolidation_config,
        )
        for results_dir in results_dirs
    }
    return consolidate_analysis_file_groups(
        analysis_files_by_directory=analysis_files_by_directory,
        plate_path=plate_path,
        analysis_consolidation_config=analysis_consolidation_config,
        plate_metadata_config=plate_metadata_config,
        filename_parser=filename_parser,
    )


def consolidate_analysis_file_groups(
    analysis_files_by_directory: Mapping[Path, tuple[Path, ...]],
    plate_path: Path,
    analysis_consolidation_config: "AnalysisConsolidationConfig",
    plate_metadata_config: "PlateMetadataConfig",
    filename_parser: "FilenameParser",
) -> tuple[list[str], list[tuple[str, str]]]:
    """Consolidate caller-owned materialized-file groups without discovery."""

    successful_dirs = []
    failed_dirs = []
    summary_paths = []
    well_resolver = FilenameParserWellResolver(filename_parser)

    for results_dir, file_paths in analysis_files_by_directory.items():
        analysis_file_paths = CsvAnalysisTableSource(
            file_paths=tuple(file_paths),
            well_resolver=well_resolver,
            analysis_consolidation_config=analysis_consolidation_config,
        ).included_file_paths()
        if not analysis_file_paths:
            logger.info(f"Skipping {results_dir} - no CSV files found")
            continue

        logger.info(
            "Consolidating %d CSV files in %s using %s",
            len(analysis_file_paths),
            results_dir,
            type(filename_parser).__name__,
        )

        try:
            consolidate_materialized_analysis_files(
                file_paths=analysis_file_paths,
                results_directory=results_dir,
                analysis_consolidation_config=analysis_consolidation_config,
                plate_metadata_config=plate_metadata_config,
                filename_parser=filename_parser,
            )
            successful_dirs.append(results_dir.name)

            # Track summary path for global consolidation
            summary_filename = analysis_consolidation_config.output_filename
            summary_path = results_dir / summary_filename
            if summary_path.exists():
                summary_paths.append(str(summary_path))

        except Exception as e:
            logger.error(f"Failed to consolidate {results_dir}: {e}", exc_info=True)
            failed_dirs.append((results_dir.name, str(e)))

    # Step 2: Create global summary by merging result types if multiple directories were consolidated
    if len(summary_paths) > 1:
        try:
            logger.info(
                f"Creating global summary from {len(summary_paths)} result type summaries"
            )

            # Use plate_path root for global output
            global_output_dir = plate_path
            global_summary_filename = (
                analysis_consolidation_config.global_summary_filename
            )
            global_summary_path = global_output_dir / global_summary_filename

            # Extract result type names from results directory paths
            result_type_names = [
                results_dir.name
                for results_dir in analysis_files_by_directory
                if (
                    results_dir / analysis_consolidation_config.output_filename
                ).exists()
            ]

            # Get plate folder name and plate ID from first summary
            plate_folder_name = plate_path.name
            plate_id = None
            if summary_paths:
                try:
                    # Read Plate ID from first summary's MetaXpress header (line 3)
                    with open(summary_paths[0], "r") as f:
                        lines = [next(f) for _ in range(3)]
                        plate_id_line = lines[2]  # Line 3: "Plate ID,12345,..."
                        plate_id = plate_id_line.split(",")[1]
                except Exception:
                    pass

            # Merge result types on Well (one row per well with all columns)
            merge_result_type_summaries(
                summary_paths=summary_paths,
                output_path=str(global_summary_path),
                plate_names=result_type_names,
                plate_folder_name=plate_folder_name,
                plate_id=plate_id,
            )
            logger.info(f"✅ Global summary created: {global_summary_path}")

        except Exception as e:
            logger.error(f"Failed to create global summary: {e}", exc_info=True)

    return successful_dirs, failed_dirs
