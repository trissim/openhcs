"""Converted from CellProfiler: CreateBatchFiles

NOTE: This module is a pipeline management/orchestration module in CellProfiler,
not an image processing function. It handles batch file creation for cluster
computing, path mappings between local and remote systems, and pipeline
serialization.

In OpenHCS, this functionality is handled by the compiler and pipeline
orchestration layer, NOT by individual processing functions. The dimensional
dataflow compiler automatically handles:
- Parallelization across compute nodes
- Path resolution and mapping
- Pipeline serialization and distribution

This conversion provides a pass-through function that preserves the image
unchanged, as the actual batch file creation logic belongs in the OpenHCS
pipeline orchestration layer, not in a processing function.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.materialization import csv_materializer


@dataclass
class BatchFileInfo:
    """Metadata about batch processing configuration.
    
    In OpenHCS, actual batch file creation is handled by the compiler.
    This dataclass captures configuration that would be passed to the
    orchestration layer.
    """
    slice_index: int
    batch_mode: bool
    remote_host_is_windows: bool
    output_directory: str
    local_path_count: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("batch_info", csv_materializer(
    fields=["slice_index", "batch_mode", "remote_host_is_windows", 
            "output_directory", "local_path_count"],
    analysis_type="batch_config"
)))
def create_batch_files(
    image: np.ndarray,
    wants_default_output_directory: bool = True,
    custom_output_directory: str = "",
    remote_host_is_windows: bool = False,
    local_root_path_1: str = "",
    cluster_root_path_1: str = "",
    local_root_path_2: str = "",
    cluster_root_path_2: str = "",
) -> Tuple[np.ndarray, BatchFileInfo]:
    """Pass-through function representing CellProfiler's CreateBatchFiles module.
    
    In CellProfiler, this module creates batch files for cluster processing.
    In OpenHCS, this functionality is handled by the compiler's orchestration
    layer, not by individual processing functions.
    
    This function passes the image through unchanged and records the batch
    configuration metadata for reference.
    
    Args:
        image: Input image array of shape (H, W)
        wants_default_output_directory: If True, use default output directory
        custom_output_directory: Custom path for batch files if not using default
        remote_host_is_windows: True if cluster computers run Windows
        local_root_path_1: Local path prefix for first mapping
        cluster_root_path_1: Cluster path prefix for first mapping
        local_root_path_2: Local path prefix for second mapping
        cluster_root_path_2: Cluster path prefix for second mapping
    
    Returns:
        Tuple of:
        - image: Unchanged input image
        - BatchFileInfo: Configuration metadata
    
    Note:
        In OpenHCS, batch processing is configured at the pipeline level:
        - PipelineConfig handles parallelization strategy
        - Path mappings are handled by the VFS (Virtual File System)
        - The compiler automatically distributes work across compute nodes
        
        This function exists for compatibility but the actual batch creation
        logic should be implemented in the OpenHCS orchestration layer.
    """
    # Count configured path mappings
    path_count = 0
    if local_root_path_1 and cluster_root_path_1:
        path_count += 1
    if local_root_path_2 and cluster_root_path_2:
        path_count += 1
    
    # Determine output directory
    output_dir = "default" if wants_default_output_directory else custom_output_directory
    
    # Create batch info metadata
    batch_info = BatchFileInfo(
        slice_index=0,
        batch_mode=False,  # Not in batch mode when creating files
        remote_host_is_windows=remote_host_is_windows,
        output_directory=output_dir,
        local_path_count=path_count
    )
    
    # Pass image through unchanged - actual batch file creation
    # is handled by OpenHCS compiler/orchestration layer
    return image, batch_info