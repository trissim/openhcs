"""
Converted from CellProfiler: ExportToDatabase
Original: ExportToDatabase module

Note: ExportToDatabase is a data export module that writes measurements to databases.
This is NOT an image processing function - it's a data I/O operation.
In OpenHCS, this functionality is handled by the pipeline's materialization system,
not by individual processing functions.

This stub provides a pass-through function that returns the image unchanged,
as the actual database export functionality should be configured at the pipeline level
using OpenHCS's built-in materialization and export capabilities.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


@dataclass
class ExportMetadata:
    """Metadata about the export operation (placeholder for pipeline-level export)."""
    slice_index: int
    image_number: int
    export_status: str


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("export_metadata", csv_materializer(
    fields=["slice_index", "image_number", "export_status"],
    analysis_type="export_metadata"
)))
def export_to_database(
    image: np.ndarray,
    db_type: str = "sqlite",
    experiment_name: str = "MyExpt",
    table_prefix: str = "",
    wants_agg_mean: bool = True,
    wants_agg_median: bool = False,
    wants_agg_std_dev: bool = False,
) -> Tuple[np.ndarray, ExportMetadata]:
    """
    Placeholder for ExportToDatabase functionality.
    
    In OpenHCS, database export is handled at the pipeline level through
    the materialization system. This function serves as a pass-through
    that preserves the image while recording export metadata.
    
    The actual database export should be configured using:
    - Pipeline-level output configuration
    - csv_materializer for CSV/database outputs
    - Custom materializers for specific database backends
    
    Args:
        image: Input image array with shape (H, W)
        db_type: Database type - "sqlite" or "mysql" (for reference only)
        experiment_name: Name of the experiment
        table_prefix: Prefix for database table names
        wants_agg_mean: Calculate per-image mean values
        wants_agg_median: Calculate per-image median values  
        wants_agg_std_dev: Calculate per-image standard deviation values
    
    Returns:
        Tuple of:
        - Original image unchanged (H, W)
        - ExportMetadata with export status information
    
    Note:
        This is a stub function. In a real OpenHCS pipeline, database export
        is configured through the pipeline's output materialization settings,
        not through individual processing functions. All measurements collected
        during the pipeline run are automatically exported based on the
        pipeline configuration.
    """
    # This function is a pass-through - actual export happens at pipeline level
    # The image is returned unchanged
    
    # Create metadata record indicating this image was processed
    metadata = ExportMetadata(
        slice_index=0,
        image_number=0,  # Will be set by pipeline context
        export_status="pending_pipeline_export"
    )
    
    return image, metadata