"""Typed lowering for CellProfiler FilterObjects settings."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.runtime_semantics import parent_child_relationship_artifact_name
from openhcs.interop.cellprofiler.setting_names import normalized_symbol_name

from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)

from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    SettingNameFamily,
    block_setting_value,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_values,
)


FILTER_OBJECTS_INPUT_SETTING = SettingNameFamily(
    "Select the object to filter",
    aliases=("Select the objects to filter", "Select the input objects"),
)
FILTER_OBJECTS_OUTPUT_SETTING = "Name the output objects"
FILTER_OBJECTS_MODE_SETTING = SettingNameFamily(
    "Filter using classifier rules or measurements?",
    aliases=("Select the filtering mode",),
)
FILTER_OBJECTS_METHOD_SETTING = "Select the filtering method"
FILTER_OBJECTS_MEASUREMENT_SETTING = "Select the measurement to filter by"
FILTER_OBJECTS_USE_MINIMUM_SETTING = "Filter using a minimum measurement value?"
FILTER_OBJECTS_MINIMUM_SETTING = "Minimum value"
FILTER_OBJECTS_USE_MAXIMUM_SETTING = "Filter using a maximum measurement value?"
FILTER_OBJECTS_MAXIMUM_SETTING = "Maximum value"
FILTER_OBJECTS_MAIN_OUTLINE_SETTING = (
    "Retain the outlines of filtered objects for use later in the pipeline "
    "(for example, in SaveImages)?"
)
FILTER_OBJECTS_OUTLINE_IMAGE_SETTING = "Name the outline image"
FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING = "Select additional object to relabel"
FILTER_OBJECTS_ADDITIONAL_OUTPUT_SETTING = "Name the relabeled objects"
FILTER_OBJECTS_ADDITIONAL_OUTLINE_SETTING = "Save outlines of relabeled objects?"
FILTER_OBJECTS_ENCLOSING_OBJECT_SETTING = (
    "Select the objects that contain the filtered objects"
)
FILTER_OBJECTS_PER_OBJECT_ASSIGNMENT_SETTING = "Assign overlapping child to"


class FilterObjectsOutputRole(str, Enum):
    """Closed runtime output roles emitted by a FilterObjects invocation."""

    MEASUREMENTS = "measurements"
    FILTERED_OBJECTS = "filtered_objects"
    RELATIONSHIPS = "relationships"
    OUTLINE_IMAGE = "outline_image"


@dataclass(frozen=True, slots=True)
class FilterObjectsObjectPair(ABC):
    """Shared input/output object-name pair for FilterObjects rows."""

    input_object_name: str
    output_object_name: str


@dataclass(frozen=True, slots=True)
class FilterObjectsAdditionalObjectRow(FilterObjectsObjectPair):
    """One additional object set relabeled using the primary filter mask."""

    retain_outline: bool = False
    outline_image_name: str | None = None

    @classmethod
    def from_block(
        cls,
        module: ModuleBlock,
        block: Sequence[ModuleSetting],
    ) -> "FilterObjectsAdditionalObjectRow":
        return cls(
            input_object_name=block_setting_value(
                block,
                FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING,
            ),
            output_object_name=block_setting_value(
                block,
                FILTER_OBJECTS_ADDITIONAL_OUTPUT_SETTING,
            ),
            retain_outline=_setting_bool(
                block_setting_value(
                    block,
                    FILTER_OBJECTS_ADDITIONAL_OUTLINE_SETTING,
                    default="No",
                )
            ),
            outline_image_name=normalized_symbol_name(
                block_setting_value(block, FILTER_OBJECTS_OUTLINE_IMAGE_SETTING)
            ),
        ).validated(module)

    def validated(
        self,
        module: ModuleBlock,
    ) -> "FilterObjectsAdditionalObjectRow":
        _require_symbol_value(
            self.input_object_name,
            module,
            FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING,
        )
        _require_symbol_value(
            self.output_object_name,
            module,
            FILTER_OBJECTS_ADDITIONAL_OUTPUT_SETTING,
        )
        if self.retain_outline and self.outline_image_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) retains an "
                "additional FilterObjects outline without an outline image name."
            )
        return self


@dataclass(frozen=True, slots=True)
class FilterObjectsMeasurementRule:
    """One measurement limit rule used by FilterObjects."""

    feature_name: str
    use_minimum: bool
    min_value: float | None
    use_maximum: bool
    max_value: float | None

    @classmethod
    def from_block(
        cls,
        module: ModuleBlock,
        block: Sequence[ModuleSetting],
    ) -> "FilterObjectsMeasurementRule":
        return cls(
            feature_name=block_setting_value(block, FILTER_OBJECTS_MEASUREMENT_SETTING),
            use_minimum=_setting_bool(
                block_setting_value(
                    block,
                    FILTER_OBJECTS_USE_MINIMUM_SETTING,
                    default="No",
                )
            ),
            min_value=_optional_float_literal(
                block_setting_value(block, FILTER_OBJECTS_MINIMUM_SETTING)
            ),
            use_maximum=_setting_bool(
                block_setting_value(
                    block,
                    FILTER_OBJECTS_USE_MAXIMUM_SETTING,
                    default="No",
                )
            ),
            max_value=_optional_float_literal(
                block_setting_value(block, FILTER_OBJECTS_MAXIMUM_SETTING)
            ),
        ).validated(module)

    def validated(self, module: ModuleBlock) -> "FilterObjectsMeasurementRule":
        if self.feature_name.strip():
            return self
        raise ValueError(
            f"Module {module.name}({module.module_num}) has an empty "
            "FilterObjects measurement rule."
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsOutput:
    """One ordered artifact output produced by FilterObjects."""

    role: FilterObjectsOutputRole
    name: str


@dataclass(frozen=True, slots=True)
class FilterObjectsPlan(FilterObjectsObjectPair):
    """Complete typed FilterObjects artifact and runtime plan."""

    retain_outline: bool
    outline_image_name: str | None
    additional_rows: tuple[FilterObjectsAdditionalObjectRow, ...]
    enclosing_object_name: str | None
    per_object_assignment: str

    @property
    def input_object_names(self) -> tuple[str, ...]:
        ordered_names = (
            self.input_object_name,
            *(row.input_object_name for row in self.additional_rows),
            *(
                ()
                if self.enclosing_object_name is None
                else (self.enclosing_object_name,)
            ),
        )
        return tuple(dict.fromkeys(ordered_names))

    @property
    def outputs(self) -> tuple[FilterObjectsOutput, ...]:
        object_pairs = (
            (self.input_object_name, self.output_object_name),
            *(
                (row.input_object_name, row.output_object_name)
                for row in self.additional_rows
            ),
        )
        object_outputs = tuple(
            FilterObjectsOutput(
                FilterObjectsOutputRole.FILTERED_OBJECTS,
                output_object_name,
            )
            for _input_object_name, output_object_name in object_pairs
        )
        outline_outputs = tuple(
            FilterObjectsOutput(FilterObjectsOutputRole.OUTLINE_IMAGE, name)
            for name in self.outline_image_names
        )
        relationship_outputs = tuple(
            FilterObjectsOutput(
                FilterObjectsOutputRole.RELATIONSHIPS,
                parent_child_relationship_artifact_name(
                    input_object_name,
                    output_object_name,
                ),
            )
            for input_object_name, output_object_name in object_pairs
        )
        return (
            FilterObjectsOutput(
                FilterObjectsOutputRole.MEASUREMENTS,
                "",
            ),
            *object_outputs,
            *relationship_outputs,
            *outline_outputs,
        )

    @property
    def outline_image_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.retain_outline:
            if self.outline_image_name is None:
                raise RuntimeError("FilterObjects retained outline has no name.")
            names.append(self.outline_image_name)
        names.extend(
            row.outline_image_name
            for row in self.additional_rows
            if row.retain_outline and row.outline_image_name is not None
        )
        return tuple(names)

    @property
    def outline_object_indices(self) -> tuple[int, ...]:
        indices: list[int] = []
        if self.retain_outline:
            indices.append(0)
        indices.extend(
            index
            for index, row in enumerate(self.additional_rows, start=1)
            if row.retain_outline
        )
        return tuple(indices)


def filter_objects_plan(module: ModuleBlock) -> FilterObjectsPlan:
    """Return the typed FilterObjects compile/runtime plan."""
    plan = FilterObjectsPlan(
        input_object_name=required_setting_value(
            module,
            FILTER_OBJECTS_INPUT_SETTING,
        ),
        output_object_name=required_setting_value(
            module,
            FILTER_OBJECTS_OUTPUT_SETTING,
        ),
        retain_outline=_setting_bool(
            optional_setting_value(module, FILTER_OBJECTS_MAIN_OUTLINE_SETTING)
            or "No"
        ),
        outline_image_name=_main_outline_image_name(module),
        additional_rows=filter_objects_additional_rows(module),
        enclosing_object_name=normalized_symbol_name(
            optional_setting_value(module, FILTER_OBJECTS_ENCLOSING_OBJECT_SETTING)
            or ""
        ),
        per_object_assignment=(
            optional_setting_value(module, FILTER_OBJECTS_PER_OBJECT_ASSIGNMENT_SETTING)
            or "Both parents"
        ),
    )
    _require_symbol_value(
        plan.input_object_name,
        module,
        FILTER_OBJECTS_INPUT_SETTING,
    )
    _require_symbol_value(
        plan.output_object_name,
        module,
        FILTER_OBJECTS_OUTPUT_SETTING,
    )
    if plan.retain_outline and plan.outline_image_name is None:
        raise ValueError(
            f"Module {module.name}({module.module_num}) retains filtered-object "
            "outlines without an outline image name."
        )
    return plan


def filter_objects_additional_rows(
    module: ModuleBlock,
) -> tuple[FilterObjectsAdditionalObjectRow, ...]:
    """Return ordered additional relabel rows from parsed FilterObjects settings."""
    if module.iter_settings():
        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING,
        )
        return tuple(
            FilterObjectsAdditionalObjectRow.from_block(module, block)
            for block in blocks
        )
    return _mapping_additional_rows(module)


def filter_objects_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return absorbed-function kwargs for a typed FilterObjects plan."""
    plan = filter_objects_plan(module)
    measurement_rules = filter_objects_measurement_rules(module)
    return {
        "mode": _filter_mode_value(module),
        "filter_method": optional_setting_value(
            module,
            FILTER_OBJECTS_METHOD_SETTING,
        )
        or "Limits",
        "measurement_features": tuple(rule.feature_name for rule in measurement_rules),
        "measurement_min_values": tuple(rule.min_value for rule in measurement_rules),
        "measurement_max_values": tuple(rule.max_value for rule in measurement_rules),
        "measurement_use_minimum": tuple(
            rule.use_minimum for rule in measurement_rules
        ),
        "measurement_use_maximum": tuple(
            rule.use_maximum for rule in measurement_rules
        ),
        "additional_object_count": len(plan.additional_rows),
        "outline_object_indices": plan.outline_object_indices,
        "enclosing_object_name": plan.enclosing_object_name,
        "per_object_assignment": plan.per_object_assignment,
    }


def filter_objects_measurement_rules(
    module: ModuleBlock,
) -> tuple[FilterObjectsMeasurementRule, ...]:
    """Return ordered measurement limit rules from parsed FilterObjects settings."""
    if module.iter_settings():
        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=FILTER_OBJECTS_MEASUREMENT_SETTING,
        )
        return tuple(
            FilterObjectsMeasurementRule.from_block(module, block)
            for block in blocks
        )
    return _mapping_measurement_rules(module)


def filter_objects_child_count_object_names(module: ModuleBlock) -> tuple[str, ...]:
    """Return child object names needed by Children_<object>_Count rules."""
    child_names = tuple(
        child_name
        for rule in filter_objects_measurement_rules(module)
        for child_name in (child_count_feature_child_name(rule.feature_name),)
        if child_name is not None
    )
    return tuple(dict.fromkeys(child_names))


def _mapping_measurement_rules(
    module: ModuleBlock,
) -> tuple[FilterObjectsMeasurementRule, ...]:
    feature_names = setting_values(module, FILTER_OBJECTS_MEASUREMENT_SETTING)
    use_minimum = setting_values(module, FILTER_OBJECTS_USE_MINIMUM_SETTING)
    min_values = setting_values(module, FILTER_OBJECTS_MINIMUM_SETTING)
    use_maximum = setting_values(module, FILTER_OBJECTS_USE_MAXIMUM_SETTING)
    max_values = setting_values(module, FILTER_OBJECTS_MAXIMUM_SETTING)
    row_count = len(feature_names)
    return tuple(
        FilterObjectsMeasurementRule(
            feature_name=_indexed_value(feature_names, index),
            use_minimum=_setting_bool(
                _indexed_value(use_minimum, index, default="No")
            ),
            min_value=_optional_float_literal(_indexed_value(min_values, index)),
            use_maximum=_setting_bool(
                _indexed_value(use_maximum, index, default="No")
            ),
            max_value=_optional_float_literal(_indexed_value(max_values, index)),
        ).validated(module)
        for index in range(row_count)
    )


def _mapping_additional_rows(
    module: ModuleBlock,
) -> tuple[FilterObjectsAdditionalObjectRow, ...]:
    input_names = setting_values(module, FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING)
    output_names = setting_values(module, FILTER_OBJECTS_ADDITIONAL_OUTPUT_SETTING)
    outline_flags = setting_values(module, FILTER_OBJECTS_ADDITIONAL_OUTLINE_SETTING)
    outline_names = setting_values(module, FILTER_OBJECTS_OUTLINE_IMAGE_SETTING)[1:]
    row_count = max(len(input_names), len(output_names), len(outline_flags))
    return tuple(
        FilterObjectsAdditionalObjectRow(
            input_object_name=_indexed_value(input_names, index),
            output_object_name=_indexed_value(output_names, index),
            retain_outline=_setting_bool(
                _indexed_value(outline_flags, index, default="No")
            ),
            outline_image_name=normalized_symbol_name(
                _indexed_value(outline_names, index)
            ),
        ).validated(module)
        for index in range(row_count)
    )


def _main_outline_image_name(module: ModuleBlock) -> str | None:
    names = setting_values(module, FILTER_OBJECTS_OUTLINE_IMAGE_SETTING)
    if not names:
        return None
    return normalized_symbol_name(names[0])


def _filter_mode_value(module: ModuleBlock) -> str:
    value = optional_setting_value(module, FILTER_OBJECTS_MODE_SETTING)
    if value is None:
        return "Measurements"
    if "border" in value.strip().lower():
        return "Border"
    return value


def _optional_float_literal(value: str | None) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _setting_bool(value: str) -> bool:
    return value.strip().lower() in {"yes", "true", "1"}


def _indexed_value(
    values: tuple[str, ...],
    index: int,
    *,
    default: str = "",
) -> str:
    if not values:
        return default
    if index < len(values):
        return values[index]
    return values[-1]


def _require_symbol_value(
    value: str,
    module: ModuleBlock,
    setting_name: str | SettingNameFamily,
) -> None:
    if normalized_symbol_name(value) is not None:
        return
    raise ValueError(
        f"Module {module.name}({module.module_num}) has an empty "
        f"FilterObjects symbol in setting {setting_name!r}."
    )
