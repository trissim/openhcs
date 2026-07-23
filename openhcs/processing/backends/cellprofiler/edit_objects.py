"""
Converted from CellProfiler: EditObjectsManually
Original: EditObjectsManually

Note: This module in CellProfiler is inherently interactive, requiring GUI-based
manual editing of objects. In OpenHCS batch processing context, this is converted
to a pass-through that optionally applies renumbering. For actual manual editing,
use the interactive napari-based tools in OpenHCS.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ObjectLineageTransformContractModule,
)
from openhcs.core.runtime_relationships import (
    DirectedObjectRelationshipPayload,
    object_label_parent_child_payload,
)
from openhcs.core.runtime_object_label_domains import (
    PresentObjectLabelIdsDomainDeclaration,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
    object_label_value_with_dense_labels,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    parse_cellprofiler_bool,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType


class RenumberChoice(Enum):
    RENUMBER = "Renumber"
    RETAIN = "Retain"


@dataclass(frozen=True, slots=True)
class EditedObjectStats:
    slice_index: int
    original_object_count: int
    edited_object_count: int
    objects_removed: int


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def edit_objects_manually(
    image: np.ndarray,
    labels: ObjectLabelValue,
    renumber_choice: RenumberChoice = RenumberChoice.RENUMBER,
    wants_guide_image: bool = False,
    allow_overlap: bool = False,
) -> tuple[
    np.ndarray,
    DataclassMeasurementColumnarRows,
    ObjectLabelValue,
    DirectedObjectRelationshipPayload,
]:
    """
    Edit objects manually - batch processing version.

    In CellProfiler, this module opens an interactive GUI for manual editing.
    In OpenHCS batch processing, this preserves the supplied edit result and
    applies CellProfiler's declared renumbering choice.

    For interactive editing, use OpenHCS napari-based editing tools.

    Args:
        image: Guiding image for visualization (H, W)
        labels: Label image with objects to edit (H, W)
        renumber_choice: Whether to renumber objects consecutively after editing
        allow_overlap: Whether overlapping objects are permitted
        wants_guide_image: Whether the current image is shown while editing

    Returns:
        Tuple of (image, stats, edited labels, parent-child relationship)
    """
    del wants_guide_image
    if allow_overlap:
        raise NotImplementedError(
            "EditObjectsManually overlapping-object edits require a layered label "
            "payload and cannot be represented by one dense ObjectLabelValue."
        )

    # Make a copy of labels to edit
    edited_labels = object_label_dense_array(labels, dtype=np.int32).copy()

    # Get original object count
    original_objects = np.unique(edited_labels)
    original_objects = original_objects[original_objects != 0]
    original_count = len(original_objects)

    # Get remaining unique labels
    unique_labels = np.unique(edited_labels)
    unique_labels = unique_labels[unique_labels != 0]
    edited_count = len(unique_labels)

    # Renumber if requested
    if renumber_choice == RenumberChoice.RENUMBER and edited_count > 0:
        renumbered_labels = np.zeros_like(edited_labels, dtype=np.int32)
        for new_label, old_label in enumerate(unique_labels, start=1):
            renumbered_labels[edited_labels == old_label] = new_label
        edited_labels = renumbered_labels

    # Compute statistics
    stats = EditedObjectStats(
        slice_index=0,
        original_object_count=original_count,
        edited_object_count=edited_count,
        objects_removed=original_count - edited_count,
    )

    edited_value = object_label_value_with_dense_labels(
        labels,
        edited_labels.astype(np.int32, copy=False),
        domain_declaration=PresentObjectLabelIdsDomainDeclaration(),
    )
    return (
        image,
        DataclassMeasurementColumnarRows((stats,), row_type=EditedObjectStats),
        edited_value,
        object_label_parent_child_payload(labels, edited_value),
    )


class EditObjectsManuallyModule(
    LabelsObjectInputPolicy,
    ObjectLineageTransformContractModule,
):
    module_name = "EditObjectsManually"
    function_name = "edit_objects_manually"
    validated = True
    confidence = 1.0
    input_objects_setting = SettingNameFamily("Select the objects to be edited")
    output_objects_setting = SettingNameFamily("Name the edited objects")
    guide_image_setting = SettingNameFamily("Select the guiding image")
    wants_guide_image_setting = SettingNameFamily("Display a guiding image?")
    guide_image_binding = SettingToKeywordBinding.input(
        guide_image_setting,
        ImageArtifactType,
    )
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting,
        ObjectLabelsArtifactType,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_objects_binding,
        output_objects_binding,
        guide_image_binding,
        SettingToKeywordBinding(
            "Numbering of the edited objects",
            "renumber_choice",
            cellprofiler_enum_setting_parser(RenumberChoice),
        ),
        SettingToKeywordBinding(
            wants_guide_image_setting,
            "wants_guide_image",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Allow overlapping objects?",
            "allow_overlap",
            parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        wants_guide_image = optional_setting_value(
            module,
            cls.wants_guide_image_setting,
        )
        if wants_guide_image is None:
            raise ValueError("EditObjectsManually requires its guiding-image choice.")
        return tuple(
            binding
            for binding in bindings
            if parse_cellprofiler_bool(wants_guide_image)
            or binding is not cls.guide_image_binding
        )
