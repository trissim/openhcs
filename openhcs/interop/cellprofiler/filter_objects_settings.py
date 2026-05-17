"""Typed lowering for CellProfiler FilterObjects settings."""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.public_api import declared_public_names
from openhcs.core.runtime_semantics import parent_child_relationship_artifact_name
from .measurement_lookup import CellProfilerMeasurementFeature
from .parser import ModuleBlock, ModuleSetting
from .setting_names import (
    SettingNameFamily,
    RepeatedSettingSequence,
    block_setting_value,
    normalized_symbol_name,
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
class FilterObjectsObjectPair(ABC, metaclass=AutoRegisterMeta):
    """Shared input/output object-name pair for FilterObjects rows."""

    __registry_key__ = "pair_role"
    __skip_if_no_key__ = True

    pair_role: ClassVar[str | None] = None
    input_object_name: str
    output_object_name: str

    @classmethod
    def registered_pair_types(cls) -> tuple[type["FilterObjectsObjectPair"], ...]:
        return tuple(cls.__registry__.values())

    @property
    def filtered_object_output(self) -> "FilterObjectsOutput":
        return FilterObjectsOutput(
            FilterObjectsOutputRole.FILTERED_OBJECTS,
            self.output_object_name,
        )

    @property
    def relationship_output(self) -> "FilterObjectsOutput":
        return FilterObjectsOutput(
            FilterObjectsOutputRole.RELATIONSHIPS,
            parent_child_relationship_artifact_name(
                self.input_object_name,
                self.output_object_name,
            ),
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsSymbolRequirement:
    """Fail-loud FilterObjects symbol-setting validation."""

    value: str
    setting_name: str | SettingNameFamily

    def validate(self, module: ModuleBlock) -> None:
        if normalized_symbol_name(self.value) is not None:
            return
        raise ValueError(
            f"Module {module.name}({module.module_num}) has an empty "
            f"FilterObjects symbol in setting {self.setting_name!r}."
        )


@dataclass(frozen=True, slots=True)
class CellProfilerBooleanLiteral:
    """CellProfiler yes/true/one literal lowered to bool."""

    value: str

    @property
    def boolean(self) -> bool:
        return self.value.strip().lower() in {"yes", "true", "1"}


@dataclass(frozen=True, slots=True)
class FilterObjectsAdditionalObjectRow(FilterObjectsObjectPair):
    """One additional object set relabeled using the primary filter mask."""

    pair_role: ClassVar[str] = "additional"
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
            retain_outline=CellProfilerBooleanLiteral(
                block_setting_value(
                    block,
                    FILTER_OBJECTS_ADDITIONAL_OUTLINE_SETTING,
                    default="No",
                )
            ).boolean,
            outline_image_name=normalized_symbol_name(
                block_setting_value(block, FILTER_OBJECTS_OUTLINE_IMAGE_SETTING)
            ),
        ).validated(module)

    def validated(
        self,
        module: ModuleBlock,
    ) -> "FilterObjectsAdditionalObjectRow":
        FilterObjectsSymbolRequirement(
            self.input_object_name,
            FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING,
        ).validate(module)
        FilterObjectsSymbolRequirement(
            self.output_object_name,
            FILTER_OBJECTS_ADDITIONAL_OUTPUT_SETTING,
        ).validate(module)
        if self.retain_outline and self.outline_image_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) retains an "
                "additional FilterObjects outline without an outline image name."
            )
        return self


@dataclass(frozen=True, slots=True)
class OptionalFloatLiteral:
    """Typed optional float parser for blank CellProfiler numeric literals."""

    raw_value: str | None

    @property
    def value(self) -> float | None:
        if self.raw_value is None:
            return None
        stripped = self.raw_value.strip()
        if not stripped:
            return None
        return float(stripped)


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
            use_minimum=CellProfilerBooleanLiteral(
                block_setting_value(
                    block,
                    FILTER_OBJECTS_USE_MINIMUM_SETTING,
                    default="No",
                )
            ).boolean,
            min_value=OptionalFloatLiteral(
                block_setting_value(block, FILTER_OBJECTS_MINIMUM_SETTING)
            ).value,
            use_maximum=CellProfilerBooleanLiteral(
                block_setting_value(
                    block,
                    FILTER_OBJECTS_USE_MAXIMUM_SETTING,
                    default="No",
                )
            ).boolean,
            max_value=OptionalFloatLiteral(
                block_setting_value(block, FILTER_OBJECTS_MAXIMUM_SETTING)
            ).value,
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
class RepeatedSettingValues(RepeatedSettingSequence):
    """CellProfiler repeated-setting sequence with last-value fallback semantics."""


@dataclass(frozen=True, slots=True)
class FilterObjectsPlan(FilterObjectsObjectPair):
    """Complete typed FilterObjects artifact and runtime plan."""

    pair_role: ClassVar[str] = "primary"
    retain_outline: bool
    outline_image_name: str | None
    additional_rows: tuple[FilterObjectsAdditionalObjectRow, ...]
    enclosing_object_name: str | None
    per_object_assignment: str

    @property
    def input_object_names(self) -> tuple[str, ...]:
        ordered_names = (
            *(pair.input_object_name for pair in self.object_pairs),
            *(
                ()
                if self.enclosing_object_name is None
                else (self.enclosing_object_name,)
            ),
        )
        return tuple(dict.fromkeys(ordered_names))

    @property
    def outputs(self) -> tuple[FilterObjectsOutput, ...]:
        outline_outputs = tuple(
            FilterObjectsOutput(FilterObjectsOutputRole.OUTLINE_IMAGE, name)
            for name in self.outline_image_names
        )
        return (
            FilterObjectsOutput(
                FilterObjectsOutputRole.MEASUREMENTS,
                "",
            ),
            *(pair.filtered_object_output for pair in self.object_pairs),
            *(pair.relationship_output for pair in self.object_pairs),
            *outline_outputs,
        )

    @property
    def object_pairs(self) -> tuple[FilterObjectsObjectPair, ...]:
        return (self, *self.additional_rows)

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


@dataclass(frozen=True, slots=True)
class FilterObjectsMainOutlineImageName:
    """Main retained-outline image name from the ordered outline settings."""

    module: ModuleBlock

    @property
    def value(self) -> str | None:
        names = setting_values(self.module, FILTER_OBJECTS_OUTLINE_IMAGE_SETTING)
        if not names:
            return None
        return normalized_symbol_name(names[0])


@dataclass(frozen=True, slots=True)
class FilterObjectsModeSetting:
    """FilterObjects mode literal normalized for absorbed runtime kwargs."""

    module: ModuleBlock

    @property
    def value(self) -> str:
        value = optional_setting_value(self.module, FILTER_OBJECTS_MODE_SETTING)
        if value is None:
            return "Measurements"
        if "border" in value.strip().lower():
            return "Border"
        return value


@dataclass(frozen=True, slots=True)
class FilterObjectsMappingMeasurementRules:
    """Measurement-rule rows parsed from mapping-style module settings."""

    module: ModuleBlock

    @property
    def rows(self) -> tuple[FilterObjectsMeasurementRule, ...]:
        feature_names = setting_values(self.module, FILTER_OBJECTS_MEASUREMENT_SETTING)
        use_minimum = setting_values(self.module, FILTER_OBJECTS_USE_MINIMUM_SETTING)
        min_values = setting_values(self.module, FILTER_OBJECTS_MINIMUM_SETTING)
        use_maximum = setting_values(self.module, FILTER_OBJECTS_USE_MAXIMUM_SETTING)
        max_values = setting_values(self.module, FILTER_OBJECTS_MAXIMUM_SETTING)
        return tuple(
            FilterObjectsMeasurementRule(
                feature_name=RepeatedSettingValues(feature_names).at(index),
                use_minimum=CellProfilerBooleanLiteral(
                    RepeatedSettingValues(use_minimum, default="No").at(index)
                ).boolean,
                min_value=OptionalFloatLiteral(
                    RepeatedSettingValues(min_values).at(index)
                ).value,
                use_maximum=CellProfilerBooleanLiteral(
                    RepeatedSettingValues(use_maximum, default="No").at(index)
                ).boolean,
                max_value=OptionalFloatLiteral(
                    RepeatedSettingValues(max_values).at(index)
                ).value,
            ).validated(self.module)
            for index in range(len(feature_names))
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsMappingAdditionalRows:
    """Additional-object rows parsed from mapping-style module settings."""

    module: ModuleBlock

    @property
    def rows(self) -> tuple[FilterObjectsAdditionalObjectRow, ...]:
        input_names = setting_values(self.module, FILTER_OBJECTS_ADDITIONAL_INPUT_SETTING)
        output_names = setting_values(self.module, FILTER_OBJECTS_ADDITIONAL_OUTPUT_SETTING)
        outline_flags = setting_values(
            self.module,
            FILTER_OBJECTS_ADDITIONAL_OUTLINE_SETTING,
        )
        outline_names = setting_values(self.module, FILTER_OBJECTS_OUTLINE_IMAGE_SETTING)[1:]
        row_count = max(len(input_names), len(output_names), len(outline_flags))
        return tuple(
            FilterObjectsAdditionalObjectRow(
                input_object_name=RepeatedSettingValues(input_names).at(index),
                output_object_name=RepeatedSettingValues(output_names).at(index),
                retain_outline=CellProfilerBooleanLiteral(
                    RepeatedSettingValues(outline_flags, default="No").at(index)
                ).boolean,
                outline_image_name=normalized_symbol_name(
                    RepeatedSettingValues(outline_names).at(index)
                ),
            ).validated(self.module)
            for index in range(row_count)
        )


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
        retain_outline=CellProfilerBooleanLiteral(
            optional_setting_value(module, FILTER_OBJECTS_MAIN_OUTLINE_SETTING)
            or "No"
        ).boolean,
        outline_image_name=FilterObjectsMainOutlineImageName(module).value,
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
    FilterObjectsSymbolRequirement(
        plan.input_object_name,
        FILTER_OBJECTS_INPUT_SETTING,
    ).validate(module)
    FilterObjectsSymbolRequirement(
        plan.output_object_name,
        FILTER_OBJECTS_OUTPUT_SETTING,
    ).validate(module)
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
    return FilterObjectsMappingAdditionalRows(module).rows


def filter_objects_bound_kwargs(module: ModuleBlock) -> dict[str, Any]:
    """Return absorbed-function kwargs for a typed FilterObjects plan."""
    plan = filter_objects_plan(module)
    measurement_rules = filter_objects_measurement_rules(module)
    return {
        "mode": FilterObjectsModeSetting(module).value,
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
    return FilterObjectsMappingMeasurementRules(module).rows


def filter_objects_child_count_object_names(module: ModuleBlock) -> tuple[str, ...]:
    """Return child object names needed by Children_<object>_Count rules."""
    return CellProfilerMeasurementFeature.child_count_object_names(
        tuple(rule.feature_name for rule in filter_objects_measurement_rules(module))
    )


__all__ = declared_public_names(globals(), constant_prefixes=("FILTER_OBJECTS_",))
