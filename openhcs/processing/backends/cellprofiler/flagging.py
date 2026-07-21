"""FlagImage backend entrypoints for CellProfiler-compatible processing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import astuple, dataclass, field, fields, replace
from enum import Enum
from functools import partial
from types import MappingProxyType
from typing import ClassVar, Self

import numpy as np

from openhcs.core.artifacts import MeasurementsArtifactType
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_tabular_values import ColumnarRows, FieldSpec
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    PriorMeasurementArtifactInputModule,
)
from openhcs.interop.cellprofiler.parser import ModuleSetting
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.setting_names import setting_values
from openhcs.interop.cellprofiler.settings_binder import (
    MeasurementFeatureSettingBinding,
    SettingToKeywordBinding,
    cellprofiler_enum_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)


class CombinationChoice(Enum):
    ANY = "Flag if any fail"
    ALL = "Flag if all fail"


class MeasurementSource(Enum):
    IMAGE = "Whole-image measurement"
    AVERAGE_OBJECT = "Average measurement for all objects in each image"
    ALL_OBJECTS = "Measurements for all objects in each image"


@dataclass
class FlagResult:
    """Result of flag evaluation for an image."""

    slice_index: int
    flag_name: str
    flag_value: int
    measurement_name: str
    measurement_value: float
    min_threshold: float
    max_threshold: float
    pass_fail: str


def flag_image_result(
    *,
    flag_name: str,
    flag_category: str,
    measurement_name: str,
    measurement_value: float,
    check_minimum: bool,
    minimum_value: float,
    check_maximum: bool,
    maximum_value: float,
) -> FlagResult:
    """Return CellProfiler-compatible FlagImage row semantics."""
    fail = False
    if not np.isnan(measurement_value):
        if check_minimum and measurement_value < minimum_value:
            fail = True
        if check_maximum and measurement_value > maximum_value:
            fail = True

    flag_value = 1 if fail else 0
    return FlagResult(
        slice_index=0,
        flag_name=f"{flag_category}_{flag_name}",
        flag_value=flag_value,
        measurement_name=measurement_name,
        measurement_value=float(measurement_value),
        min_threshold=minimum_value if check_minimum else float("nan"),
        max_threshold=maximum_value if check_maximum else float("nan"),
        pass_fail="Fail" if fail else "Pass",
    )


class _FlagImageMeasurementTablesRuntimeParameter(KeywordRuntimeParameter):
    """Measurement artifacts selected from repeated FlagImage criteria."""

    parameter_name = "measurement_tables"
    annotation_type = tuple[MeasurementTable, ...]
    parameter_default = ()


class FlagImageInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind declaration-selected prior measurements to FlagImage."""

    binds_without_declared_inputs = True
    supported_non_object_input_kinds = frozenset({MeasurementsArtifactType})

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        bound = super().bind_runtime_inputs(request)
        tables = request.declared_measurement_tables()
        if not tables:
            raise ValueError("FlagImage requires declared prior measurement artifacts.")
        bound[_FlagImageMeasurementTablesRuntimeParameter.require_parameter_name()] = (
            tables
        )
        return bound


@dataclass(frozen=True, slots=True)
class FlagImageSettingField:
    """One repeated setting column owned by its nominal row field."""

    binding: SettingToKeywordBinding
    coerce: Callable[[object], object]

    @classmethod
    def declare(
        cls,
        binding: SettingToKeywordBinding,
        coerce: Callable[[object], object],
    ):
        """Declare one dataclass field carrying its complete setting semantics."""

        return field(metadata={cls: cls(binding, coerce)})


def _flag_boolean(value: object) -> bool:
    """Coerce only the two public boolean representations accepted by FlagImage."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_cellprofiler_bool(value)
    raise TypeError(
        f"FlagImage boolean columns require bool or str, got {type(value).__name__}."
    )


class FlagImageSettingRow:
    """Nominal owner for one repeated setting-row schema."""

    cardinality_label: ClassVar[str]

    @classmethod
    def setting_fields(cls) -> tuple[FlagImageSettingField, ...]:
        return tuple(
            row_field.metadata[FlagImageSettingField]
            for row_field in fields(cls)
            if FlagImageSettingField in row_field.metadata
        )

    @classmethod
    def setting_bindings(cls) -> tuple[SettingToKeywordBinding, ...]:
        return tuple(row_field.binding for row_field in cls.setting_fields())

    @classmethod
    def from_public_kwargs(cls, kwargs: Mapping[str, object]) -> tuple[Self, ...]:
        columns = tuple(
            kwargs[row_field.binding.require_parameter_name()]
            for row_field in cls.setting_fields()
        )
        return cls.from_columns(columns)

    @classmethod
    def from_module(cls, module) -> tuple[Self, ...]:
        return cls.from_columns(
            tuple(
                setting_values(module, row_field.binding.setting_name)
                for row_field in cls.setting_fields()
            )
        )

    @classmethod
    def from_columns(cls, columns: Sequence[Sequence[object]]) -> tuple[Self, ...]:
        setting_fields = cls.setting_fields()
        if len(columns) != len(setting_fields):
            raise ValueError(
                f"FlagImage {cls.cardinality_label} schema declares "
                f"{len(setting_fields)} columns, got {len(columns)}."
            )
        if any(isinstance(column, (str, bytes)) for column in columns):
            raise TypeError(
                f"FlagImage {cls.cardinality_label} columns must be sequences, "
                "not scalar strings."
            )
        cardinalities = {len(column) for column in columns}
        if cardinalities == {0} or len(cardinalities) != 1:
            raise ValueError(
                f"FlagImage repeated {cls.cardinality_label} columns must have "
                "one equal nonzero cardinality."
            )
        return tuple(
            cls(
                *(
                    row_field.coerce(value)
                    for row_field, value in zip(
                        setting_fields,
                        row_values,
                        strict=True,
                    )
                )
            )
            for row_values in zip(*columns, strict=True)
        )

    @classmethod
    def flatten(cls, rows: Sequence[Self]) -> dict[str, tuple[object, ...]]:
        dataclass_fields = fields(cls)
        setting_indexes = tuple(
            index
            for index, dataclass_field in enumerate(dataclass_fields)
            if FlagImageSettingField in dataclass_field.metadata
        )
        row_values = tuple(astuple(row) for row in rows)
        return {
            setting_field.binding.require_parameter_name(): tuple(
                values[index] for values in row_values
            )
            for index, setting_field in zip(
                setting_indexes,
                cls.setting_fields(),
                strict=True,
            )
        }


@dataclass(frozen=True, slots=True)
class FlagCriterion(FlagImageSettingRow):
    """One exact measurement criterion within a repeated flag group."""

    cardinality_label = "criterion"
    source: MeasurementSource = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Flag is based on",
            "measurement_sources",
            cellprofiler_enum_setting_parser(MeasurementSource),
            repeated=True,
        ),
        partial(coerce_cellprofiler_enum, MeasurementSource),
    )
    object_name: str = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Select the object to be used for flagging",
            "object_names",
            str,
            repeated=True,
        ),
        str,
    )
    feature_name: str = FlagImageSettingField.declare(
        MeasurementFeatureSettingBinding(
            "Which measurement?",
            "measurement_features",
            str,
            repeated=True,
        ),
        str,
    )
    check_minimum: bool = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Flag images based on low values?",
            "check_minimums",
            parse_cellprofiler_bool,
            repeated=True,
        ),
        _flag_boolean,
    )
    minimum_value: float = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Minimum value",
            "minimum_values",
            float,
            repeated=True,
        ),
        float,
    )
    check_maximum: bool = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Flag images based on high values?",
            "check_maximums",
            parse_cellprofiler_bool,
            repeated=True,
        ),
        _flag_boolean,
    )
    maximum_value: float = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Maximum value",
            "maximum_values",
            float,
            repeated=True,
        ),
        float,
    )

    def values(self, measurement_tables: tuple[MeasurementTable, ...]) -> np.ndarray:
        object_name = (
            None if self.source is MeasurementSource.IMAGE else self.object_name
        )
        indexed = MeasurementFeatureQuery(
            self.feature_name,
            object_name=object_name,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        ).optional_value_index(measurement_tables)
        if indexed is None:
            return np.asarray((), dtype=float)
        values_by_label, positional_values = indexed
        return np.asarray(
            (*values_by_label.values(), *positional_values),
            dtype=float,
        )

    def fails(self, measurement_tables: tuple[MeasurementTable, ...]) -> bool:
        values = self.values(measurement_tables)
        if self.source is MeasurementSource.IMAGE:
            if values.size != 1:
                raise ValueError(
                    "FlagImage whole-image criteria require exactly one value; "
                    f"{self.feature_name!r} produced {values.size}."
                )
            minimum = maximum = float(values[0])
        elif values.size == 0:
            return True
        elif self.source is MeasurementSource.AVERAGE_OBJECT:
            minimum = maximum = float(np.mean(values))
        elif self.source is MeasurementSource.ALL_OBJECTS:
            minimum = float(np.min(values))
            maximum = float(np.max(values))
        else:
            raise NotImplementedError(
                f"Unsupported FlagImage measurement source {self.source.value!r}."
            )
        return bool(
            (self.check_minimum and minimum < self.minimum_value)
            or (self.check_maximum and maximum > self.maximum_value)
        )


@dataclass(frozen=True, slots=True)
class FlagDefinition(FlagImageSettingRow):
    """One repeated FlagImage output and its ordered criteria."""

    cardinality_label = "flag"
    category: str = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Name the flag's category",
            "flag_categories",
            str,
            repeated=True,
        ),
        str,
    )
    name: str = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Name the flag",
            "flag_names",
            str,
            repeated=True,
        ),
        str,
    )
    combination_choice: CombinationChoice = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "How should measurements be linked?",
            "combination_choices",
            cellprofiler_enum_setting_parser(CombinationChoice),
            repeated=True,
        ),
        partial(coerce_cellprofiler_enum, CombinationChoice),
    )
    wants_skip: bool = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Skip image set if flagged?",
            "wants_skip",
            parse_cellprofiler_bool,
            repeated=True,
        ),
        _flag_boolean,
    )
    measurement_count: int = FlagImageSettingField.declare(
        SettingToKeywordBinding(
            "Measurement count",
            "measurement_counts",
            int,
            repeated=True,
        ),
        int,
    )
    criteria: tuple[FlagCriterion, ...] = ()

    def __post_init__(self) -> None:
        if not self.category or not self.name:
            raise ValueError("FlagImage category and flag name cannot be empty.")
        if self.measurement_count <= 0:
            raise ValueError(
                f"FlagImage flag {self.feature_name!r} requires at least one criterion."
            )
        if self.criteria and len(self.criteria) != self.measurement_count:
            raise ValueError(
                f"FlagImage flag {self.feature_name!r} declares "
                f"measurement_count={self.measurement_count}, but has "
                f"{len(self.criteria)} criteria."
            )

    @property
    def feature_name(self) -> str:
        return f"{self.category}_{self.name}"

    def with_criteria(self, criteria: tuple[FlagCriterion, ...]) -> Self:
        return replace(self, criteria=criteria)

    def value(self, measurement_tables: tuple[MeasurementTable, ...]) -> int:
        failures = tuple(
            criterion.fails(measurement_tables) for criterion in self.criteria
        )
        failed = (
            any(failures)
            if self.combination_choice is CombinationChoice.ANY
            else all(failures)
        )
        if failed and self.wants_skip:
            raise NotImplementedError(
                "FlagImage skip disposition is unsupported because OpenHCS has no "
                "generic runtime pipeline-disposition authority."
            )
        return int(failed)


@dataclass(frozen=True, slots=True)
class FlagImagePlan:
    """Typed repeated FlagImage topology shared by parsing and execution."""

    ignore_flag_on_last_binding = SettingToKeywordBinding(
        "Ignore flag skips on last cycle?",
        "ignore_flag_on_last",
        parse_cellprofiler_bool,
    )
    flags: tuple[FlagDefinition, ...]
    ignore_flag_on_last: bool = False

    def __post_init__(self) -> None:
        if not self.flags:
            raise ValueError("FlagImage requires at least one flag definition.")
        duplicates = tuple(
            name
            for name in dict.fromkeys(flag.feature_name for flag in self.flags)
            if sum(flag.feature_name == name for flag in self.flags) > 1
        )
        if duplicates:
            raise ValueError(
                f"FlagImage declares duplicate flag features {duplicates!r}."
            )
        if any(flag.wants_skip for flag in self.flags):
            raise NotImplementedError(
                "FlagImage skip disposition is unsupported because OpenHCS has no "
                "generic runtime pipeline-disposition authority."
            )

    @classmethod
    def setting_bindings(cls) -> tuple[SettingToKeywordBinding, ...]:
        return (
            *FlagDefinition.setting_bindings(),
            *FlagCriterion.setting_bindings(),
            cls.ignore_flag_on_last_binding,
        )

    @classmethod
    def from_public_kwargs(cls, kwargs: Mapping[str, object]) -> Self:
        return cls._assemble(
            FlagDefinition.from_public_kwargs(kwargs),
            FlagCriterion.from_public_kwargs(kwargs),
            _flag_boolean(
                kwargs[cls.ignore_flag_on_last_binding.require_parameter_name()]
            ),
        )

    @classmethod
    def from_module(cls, module) -> Self:
        ignore_values = setting_values(
            module,
            cls.ignore_flag_on_last_binding.setting_name,
        )
        if len(ignore_values) != 1:
            raise ValueError(
                "FlagImage requires exactly one ignore-last-cycle setting row."
            )
        return cls._assemble(
            FlagDefinition.from_module(module),
            FlagCriterion.from_module(module),
            _flag_boolean(ignore_values[0]),
        )

    @classmethod
    def _assemble(
        cls,
        flags: tuple[FlagDefinition, ...],
        criteria: tuple[FlagCriterion, ...],
        ignore_flag_on_last: bool,
    ) -> Self:
        expected_criteria = sum(flag.measurement_count for flag in flags)
        if len(criteria) != expected_criteria:
            raise ValueError(
                "FlagImage repeated criterion columns do not match measurement counts: "
                f"expected {expected_criteria}, got {len(criteria)}."
            )
        offset = 0
        completed_flags: list[FlagDefinition] = []
        for flag in flags:
            next_offset = offset + flag.measurement_count
            completed_flags.append(flag.with_criteria(criteria[offset:next_offset]))
            offset = next_offset
        return cls(tuple(completed_flags), ignore_flag_on_last)

    def public_kwargs(self) -> dict[str, object]:
        criteria = tuple(
            criterion for flag in self.flags for criterion in flag.criteria
        )
        return {
            **FlagDefinition.flatten(self.flags),
            **FlagCriterion.flatten(criteria),
            self.ignore_flag_on_last_binding.require_parameter_name(): self.ignore_flag_on_last,
        }

    def rows(self, measurement_tables: tuple[MeasurementTable, ...]) -> ColumnarRows:
        columns: dict[str, tuple[int]] = {
            MeasurementRowAxisField.SLICE_INDEX.value: (0,)
        }
        for flag in self.flags:
            columns[flag.feature_name] = (flag.value(measurement_tables),)
        result_fields = (
            FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
            *(FieldSpec(flag.feature_name, int, required=False) for flag in self.flags),
        )
        return MeasurementProjectedColumnarRows(
            MappingProxyType(columns),
            fields=result_fields,
        )


@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(_FlagImageMeasurementTablesRuntimeParameter)
def flag_image(
    image: np.ndarray,
    flag_categories: tuple[str, ...] = ("Metadata",),
    flag_names: tuple[str, ...] = ("QCFlag",),
    combination_choices: tuple[CombinationChoice, ...] = (CombinationChoice.ANY,),
    wants_skip: tuple[bool, ...] = (False,),
    measurement_counts: tuple[int, ...] = (1,),
    measurement_sources: tuple[MeasurementSource, ...] = (MeasurementSource.IMAGE,),
    object_names: tuple[str, ...] = ("Image",),
    measurement_features: tuple[str, ...] = ("Intensity_Mean",),
    check_minimums: tuple[bool, ...] = (True,),
    minimum_values: tuple[float, ...] = (0.0,),
    check_maximums: tuple[bool, ...] = (True,),
    maximum_values: tuple[float, ...] = (1.0,),
    ignore_flag_on_last: bool = False,
    *,
    measurement_tables: tuple[MeasurementTable, ...] = (),
) -> tuple[np.ndarray, ColumnarRows]:
    """Emit one exact integer image measurement for every repeated flag."""
    plan = FlagImagePlan.from_public_kwargs(locals())
    if not measurement_tables:
        raise ValueError("FlagImage requires declared prior measurement artifacts.")
    return image, plan.rows(measurement_tables)


@numpy(contract=ProcessingContract.PURE_2D)
def flag_image_intensity(
    image: np.ndarray,
    flag_name: str = "IntensityQC",
    flag_category: str = "Metadata",
    check_minimum: bool = True,
    minimum_value: float = 0.0,
    check_maximum: bool = True,
    maximum_value: float = 1.0,
    use_mean: bool = True,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Flag an image based on mean or median intensity."""
    if use_mean:
        measurement_value = float(np.mean(image))
        measurement_name = "intensity_mean"
    else:
        measurement_value = float(np.median(image))
        measurement_name = "intensity_median"
    return image, DataclassMeasurementColumnarRows(
        (
            flag_image_result(
                flag_name=flag_name,
                flag_category=flag_category,
                measurement_name=measurement_name,
                measurement_value=measurement_value,
                check_minimum=check_minimum,
                minimum_value=minimum_value,
                check_maximum=check_maximum,
                maximum_value=maximum_value,
            ),
        ),
        row_type=FlagResult,
    )


class FlagImageModule(
    FlagImageInputPolicy,
    NoObjectNameMeasurementRecordMixin,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "FlagImage"
    function_name = "flag_image"
    validated = True
    confidence = 1.0
    flag_count_setting: ClassVar[str] = "Flag count"
    setting_bindings = FlagImagePlan.setting_bindings()
    ignored_settings = (
        flag_count_setting,
        "Rules file location",
        "Rules file name",
        "Class number",
        "Allow fuzzy feature matching?",
    )

    @classmethod
    def plan(cls, module) -> FlagImagePlan:
        """Return the typed repeated topology encoded by one module block."""
        plan = FlagImagePlan.from_module(module)
        count_values = setting_values(module, cls.flag_count_setting)
        if len(count_values) != 1:
            raise ValueError("FlagImage requires exactly one flag-count setting row.")
        declared_count = int(count_values[0])
        if declared_count != len(plan.flags):
            raise ValueError(
                f"FlagImage declares flag_count={declared_count}, but has "
                f"{len(plan.flags)} repeated flag rows."
            )
        return plan

    @classmethod
    def prior_measurement_feature_names(cls, module) -> tuple[str, ...]:
        """Select exact prior features used by every repeated criterion."""
        return tuple(
            dict.fromkeys(
                criterion.feature_name
                for flag in cls.plan(module).flags
                for criterion in flag.criteria
            )
        )

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind flattened public tuples from the declaration-owned typed plan."""
        bound = cls._bind_declared_settings(module, binder=binder)
        plan = cls.plan(module)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=bound.with_replaced_kwargs(plan.public_kwargs()),
        )

    @classmethod
    def _derived_identity_setting_records(
        cls,
        *,
        invocation,
        block_position,
        existing_records,
        step_context,
    ):
        """Add the hidden repeated-group count for public FunctionSteps."""
        own = (
            ()
            if cls.normalize_setting_name(cls.flag_count_setting)
            in cls._normalized_record_setting_names(existing_records)
            else (
                ModuleSetting(
                    cls.flag_count_setting,
                    str(
                        len(
                            FlagDefinition.from_module(
                                cls._module_block_from_setting_records(existing_records)
                            )
                        )
                    ),
                ),
            )
        )
        return (
            *own,
            *super()._derived_identity_setting_records(
                invocation=invocation,
                block_position=block_position,
                existing_records=(*existing_records, *own),
                step_context=step_context,
            ),
        )


__all__ = public_names_from_objects(
    CombinationChoice,
    FlagCriterion,
    FlagDefinition,
    FlagImagePlan,
    FlagResult,
    MeasurementSource,
    flag_image,
    flag_image_intensity,
    flag_image_result,
)
