"""Typed settings lowering for CellProfiler UntangleWorms."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any
from xml.dom.minidom import parse

from openhcs.interop.cellprofiler.parser import ModuleBlock
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


class UntangleWormsOverlapStyle(str, Enum):
    WITH_OVERLAP = "with_overlap"
    WITHOUT_OVERLAP = "without_overlap"
    BOTH = "both"


UNTANGLE_WORMS_INPUT_IMAGE_SETTING = SettingNameFamily(
    "Select the input binary image",
    aliases=("Select the input image",),
)
UNTANGLE_WORMS_OVERLAPPING_OBJECTS_SETTING = (
    "Name the output overlapping worm objects"
)
UNTANGLE_WORMS_NONOVERLAPPING_OBJECTS_SETTING = (
    "Name the output non-overlapping worm objects"
)
UNTANGLE_WORMS_TRAINING_FILE_NAME_SETTING = "Training set file name"
_TRAINING_PARAMETER_TAGS: tuple[tuple[str, str, type], ...] = (
    ("min-area", "min_worm_area", float),
    ("max-area", "max_worm_area", float),
    ("cost-threshold", "cost_threshold", float),
    ("num-control-points", "num_control_points", int),
    ("max-radius", "max_radius", float),
    ("max-skel-length", "max_skel_length", float),
    ("min-path-length", "min_path_length", float),
    ("max-path-length", "max_path_length", float),
    ("median-worm-area", "median_worm_area", float),
    ("overlap-weight", "overlap_weight", float),
    ("leftover-weight", "leftover_weight", float),
)
_TRAINING_VECTOR_TAGS: tuple[tuple[str, str], ...] = (
    ("mean-angles", "mean_angles"),
    ("radii-from-training", "radii_from_training"),
)
_TRAINING_MATRIX_TAGS: tuple[tuple[str, str], ...] = (
    ("inv-angles-covariance-matrix", "inv_angles_covariance_matrix"),
)


def untangle_worms_bound_kwargs(module: ModuleBlock) -> dict[str, str | int | float | tuple[Any, ...]]:
    """Bind UntangleWorms settings that affect runtime output semantics."""
    overlap_style = coerce_cellprofiler_enum(
        UntangleWormsOverlapStyle,
        module.get_setting("Overlap style", "Without overlap"),
    )
    kwargs: dict[str, str | int | float | tuple[Any, ...]] = {
        "overlap_style": overlap_style.value
    }
    kwargs["overlapping_object_name"] = required_setting_value(
        module,
        UNTANGLE_WORMS_OVERLAPPING_OBJECTS_SETTING,
    )
    kwargs["nonoverlapping_object_name"] = required_setting_value(
        module,
        UNTANGLE_WORMS_NONOVERLAPPING_OBJECTS_SETTING,
    )
    if (num_control_points := optional_setting_value(
        module,
        "Number of control points",
    )) is not None:
        kwargs["num_control_points"] = int(float(num_control_points))
    kwargs.update(_training_parameter_kwargs(module))
    return kwargs


def _training_parameter_kwargs(module: ModuleBlock) -> dict[str, float | int | tuple[Any, ...]]:
    training_path = _training_file_path(module)
    if training_path is None:
        return {}
    doc = parse(str(training_path))
    kwargs: dict[str, float | int | tuple[Any, ...]] = {}
    for tag_name, parameter_name, coerce in _TRAINING_PARAMETER_TAGS:
        elements = doc.documentElement.getElementsByTagName(tag_name)
        if len(elements) != 1:
            continue
        text = "".join(
            node.data
            for node in elements[0].childNodes
            if node.nodeType == doc.TEXT_NODE
        ).strip()
        if text:
            kwargs[parameter_name] = coerce(float(text)) if coerce is int else coerce(text)
    for tag_name, parameter_name in _TRAINING_VECTOR_TAGS:
        values = _xml_vector_values(doc, tag_name)
        if values:
            kwargs[parameter_name] = values
    for tag_name, parameter_name in _TRAINING_MATRIX_TAGS:
        rows = _xml_matrix_values(doc, tag_name)
        if rows:
            kwargs[parameter_name] = rows
    return kwargs


def _xml_vector_values(doc: Any, tag_name: str) -> tuple[float, ...]:
    elements = doc.documentElement.getElementsByTagName(tag_name)
    if len(elements) != 1:
        return ()
    return tuple(
        _xml_float(value_element, doc)
        for value_element in elements[0].getElementsByTagName("value")
    )


def _xml_matrix_values(doc: Any, tag_name: str) -> tuple[tuple[float, ...], ...]:
    elements = doc.documentElement.getElementsByTagName(tag_name)
    if len(elements) != 1:
        return ()
    return tuple(
        tuple(
            _xml_float(value_element, doc)
            for value_element in values_element.getElementsByTagName("value")
        )
        for values_element in elements[0].getElementsByTagName("values")
    )


def _xml_float(element: Any, doc: Any) -> float:
    text = "".join(
        node.data
        for node in element.childNodes
        if node.nodeType == doc.TEXT_NODE
    ).strip()
    return float(text)


def _training_file_path(module: ModuleBlock) -> Path | None:
    file_name = optional_setting_value(module, UNTANGLE_WORMS_TRAINING_FILE_NAME_SETTING)
    if not file_name or module.cppipe_path is None:
        return None
    for candidate in (
        module.cppipe_path.parent / file_name,
        module.cppipe_path.parent / "images" / file_name,
    ):
        if candidate.is_file():
            return candidate
    return None
