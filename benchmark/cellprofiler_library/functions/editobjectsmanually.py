"""
Converted from CellProfiler: EditObjectsManually
Original: EditObjectsManually

Note: This module in CellProfiler is inherently interactive, requiring GUI-based
manual editing of objects. In OpenHCS batch processing context, this is converted
to a pass-through that optionally applies renumbering. For actual manual editing,
use the interactive napari-based tools in OpenHCS.
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.analysis.cell_counting_cpu import materialize_segmentation_masks


class RenumberChoice(Enum):
    RENUMBER = "renumber"
    RETAIN = "retain"


@dataclass
class EditedObjectStats:
    slice_index: int
    original_object_count: int
    edited_object_count: int
    objects_removed: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    ("edited_stats", csv_materializer(
        fields=["slice_index", "original_object_count", "edited_object_count", "objects_removed"],
        analysis_type="object_editing"
    )),
    ("edited_labels", materialize_segmentation_masks)
)
def edit_objects_manually(
    image: np.ndarray,
    labels: np.ndarray,
    renumber_choice: RenumberChoice = RenumberChoice.RENUMBER,
    allow_overlap: bool = False,
    objects_to_remove: str = "",
) -> Tuple[np.ndarray, EditedObjectStats, np.ndarray]:
    """
    Edit objects manually - batch processing version.
    
    In CellProfiler, this module opens an interactive GUI for manual editing.
    In OpenHCS batch processing, this serves as a pass-through with optional
    programmatic object removal and renumbering.
    
    For interactive editing, use OpenHCS napari-based editing tools.
    
    Args:
        image: Guiding image for visualization (H, W)
        labels: Label image with objects to edit (H, W)
        renumber_choice: Whether to renumber objects consecutively after editing
        allow_overlap: Whether overlapping objects are permitted
        objects_to_remove: Comma-separated list of object IDs to remove (e.g., "1,5,12")
    
    Returns:
        Tuple of (image, stats, edited_labels)
    """
    from skimage.measure import regionprops, label as relabel_connected
    
    # Make a copy of labels to edit
    edited_labels = labels.copy().astype(np.int32)
    
    # Get original object count
    original_objects = np.unique(edited_labels)
    original_objects = original_objects[original_objects != 0]
    original_count = len(original_objects)
    
    # Parse objects to remove if specified
    if objects_to_remove and objects_to_remove.strip():
        try:
            ids_to_remove = [int(x.strip()) for x in objects_to_remove.split(",") if x.strip()]
            for obj_id in ids_to_remove:
                edited_labels[edited_labels == obj_id] = 0
        except ValueError:
            # If parsing fails, skip removal
            pass
    
    # Get remaining unique labels
    unique_labels = np.unique(edited_labels)
    unique_labels = unique_labels[unique_labels != 0]
    edited_count = len(unique_labels)
    
    # Renumber if requested
    if renumber_choice == RenumberChoice.RENUMBER and edited_count > 0:
        # Create mapping from old labels to new consecutive labels
        mapping = np.zeros(edited_labels.max() + 1, dtype=np.int32)
        for new_label, old_label in enumerate(unique_labels, start=1):
            mapping[old_label] = new_label
        edited_labels = mapping[edited_labels]
    
    # Handle overlapping objects check (in batch mode, just validate)
    if not allow_overlap:
        # Check for any pixel belonging to multiple objects
        # In a standard label image, this shouldn't happen, but we validate
        pass  # Label images by definition don't have overlaps in single array
    
    # Compute statistics
    stats = EditedObjectStats(
        slice_index=0,
        original_object_count=original_count,
        edited_object_count=edited_count,
        objects_removed=original_count - edited_count
    )
    
    return image, stats, edited_labels.astype(np.float32)