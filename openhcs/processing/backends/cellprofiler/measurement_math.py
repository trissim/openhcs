"""CalculateMath measurement semantics for CellProfiler-compatible processing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from openhcs.core.artifacts import ArtifactKind, ArtifactSpecCollection
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_NAME_FIELD,
)
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.runtime_semantics import measurement_row_mapping
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.measurement_lookup import count_feature_object_name
from openhcs.interop.cellprofiler.runtime.binding_authorities import (
    CellProfilerStringKwargAuthority,
)
from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    RuntimeBoundParameterName,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ObjectLabelCountAuthority,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerMeasurementVector,
    CellProfilerObjectMeasurementVectorBatchBinding,
    CellProfilerObjectMeasurementVectorBinding,
    MeasurementImageOperandVectorResolution,
    ObjectInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargDict,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    coerce_cellprofiler_enum,
)
from openhcs.interop.cellprofiler.setting_names import (
    OptionalSettingSymbol,
    SettingNameFamily,
)
from openhcs.core.registry_strategies import enum_member_with_payload

from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    MeasurementDebugViewModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoSourceMeasurementRecordMixin,
    TableMeasurementRecordRowsMixin,
)


class CalculateMathInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind CalculateMath operands from runtime measurement/object state."""

    binds_without_declared_inputs = True
    supported_non_object_input_kinds = frozenset({ArtifactKind.MEASUREMENTS})
    operand1_value_kwarg = RuntimeBoundParameterName("operand1_value")
    operand2_value_kwarg = RuntimeBoundParameterName("operand2_value")

    def bind(
        self,
        request: ObjectInputBindingRequest,
    ) -> CellProfilerKwargDict:
        started_at = time.perf_counter()
        operand_bindings = self.object_operand_bindings(request)
        if operand_bindings is not None:
            vectors = CellProfilerObjectMeasurementVectorBatchBinding(
                operand_bindings
            ).vectors()
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "calculate_math_bind_total",
                time.perf_counter() - started_at,
            )
            return self.operand_value_kwargs(
                vectors[0].calculate_math_operand_value,
                vectors[1].calculate_math_operand_value,
            )

        operand1_started_at = time.perf_counter()
        operand1_value = self.operand_value(
            request,
            feature_kwarg="operand1_feature",
            object_kwarg="operand1_object_name",
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand1_started_at,
            operand="1",
        )
        operand2_started_at = time.perf_counter()
        operand2_value = self.operand_value(
            request,
            feature_kwarg="operand2_feature",
            object_kwarg="operand2_object_name",
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand2_started_at,
            operand="2",
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_bind_total",
            time.perf_counter() - started_at,
        )
        return self.operand_value_kwargs(operand1_value, operand2_value)

    def operand_value_kwargs(
        self,
        operand1_value: CellProfilerRuntimeValue,
        operand2_value: CellProfilerRuntimeValue,
    ) -> CellProfilerKwargDict:
        """Return CalculateMath operand kwargs."""
        return {
            self.operand1_value_kwarg: operand1_value,
            self.operand2_value_kwarg: operand2_value,
        }

    def object_operand_bindings(
        self,
        request: ObjectInputBindingRequest,
    ) -> tuple[CellProfilerObjectMeasurementVectorBinding, ...] | None:
        bindings: list[CellProfilerObjectMeasurementVectorBinding] = []
        for feature_kwarg, object_kwarg in (
            ("operand1_feature", "operand1_object_name"),
            ("operand2_feature", "operand2_object_name"),
        ):
            feature_name = CellProfilerStringKwargAuthority.required(
                request.kwargs,
                feature_kwarg,
                "CalculateMath",
            )
            object_name = CellProfilerStringKwargAuthority.optional(
                request.kwargs,
                object_kwarg,
            )
            if (
                object_name is None
                or count_feature_object_name(feature_name) is not None
            ):
                return None
            object_spec = ArtifactSpecCollection(request.object_inputs).by_name(
                object_name
            )
            if object_spec is None:
                return None
            bindings.append(
                CellProfilerObjectMeasurementVectorBinding.for_object(
                    request,
                    object_ref=object_spec,
                    feature_name=feature_name,
                )
            )
        return tuple(bindings)

    def operand_value(
        self,
        request: ObjectInputBindingRequest,
        *,
        feature_kwarg: str,
        object_kwarg: str,
    ) -> CellProfilerRuntimeValue:
        feature_name = CellProfilerStringKwargAuthority.required(
            request.kwargs,
            feature_kwarg,
            "CalculateMath",
        )
        object_name = CellProfilerStringKwargAuthority.optional(
            request.kwargs,
            object_kwarg,
        )
        count_object_name = count_feature_object_name(feature_name)
        if count_object_name is not None:
            return float(
                ObjectLabelCountAuthority.count_from_adapter(
                    request.adapter,
                    count_object_name,
                )
            )
        if object_name is None:
            return self.image_operand_value(request, feature_name)

        return (
            CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_name,
                feature_name=feature_name,
            )
            .vector()
            .calculate_math_operand_value
        )

    def image_operand_value(
        self,
        request: ObjectInputBindingRequest,
        feature_name: str,
    ) -> CellProfilerRuntimeValue:
        declared_measurement_tables = request.declared_measurement_tables()
        if declared_measurement_tables:
            declared_slice_values = MeasurementImageOperandVectorResolution(
                measurement_tables=declared_measurement_tables,
                feature_name=feature_name,
            ).resolve()
            if declared_slice_values is not None:
                return CellProfilerMeasurementVector(
                    declared_slice_values
                ).slice_aligned_value
            return MeasurementFeatureQuery(
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).scalar_value(declared_measurement_tables)

        tables_started_at = time.perf_counter()
        measurement_resolution = MeasurementImageOperandVectorResolution.from_runtime_feature(
            request.adapter,
            feature_name,
            current_image=request.current_image,
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_measurement_tables",
            time.perf_counter() - tables_started_at,
            feature=feature_name,
            count=len(measurement_resolution.measurement_tables),
        )
        slice_started_at = time.perf_counter()
        slice_values = measurement_resolution.resolve()
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_image_operand_slices",
            time.perf_counter() - slice_started_at,
            feature=feature_name,
            sliced=slice_values is not None,
        )
        if slice_values is None:
            scalar_started_at = time.perf_counter()
            scalar_value = MeasurementFeatureQuery(
                feature_name,
                dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
            ).scalar_value(measurement_resolution.measurement_tables)
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "calculate_math_image_operand_scalar",
                time.perf_counter() - scalar_started_at,
                feature=feature_name,
            )
            return scalar_value
        return CellProfilerMeasurementVector(slice_values).slice_aligned_value


class HomogeneousObjectNameMeasurementRecordMixin:
    """Preserve table-level object ownership for homogeneous object rows."""

    @classmethod
    def measurement_record(
        cls,
        request: "CellProfilerOutputRecordRequest",
    ) -> "CellProfilerMeasurementRecord":
        from openhcs.interop.cellprofiler.runtime.measurement_materialization import (
            CellProfilerMeasurementFieldSchema,
            CellProfilerMeasurementRecord,
        )

        rows = cls.measurement_record_rows(request)
        object_name = cls.homogeneous_row_object_name(rows)
        if object_name is None:
            object_name = cls.measurement_record_object_name(request, rows)
        source_context = cls.measurement_record_source_context(request, rows)
        if (
            object_name is None
            and CellProfilerMeasurementFieldSchema.rows_declare_object_name(rows)
        ):
            source_context = source_context.without_source()
        rows, fields = cls.measurement_record_fields(request, rows)
        return CellProfilerMeasurementRecord(
            rows=rows,
            object_name=object_name,
            source_context=source_context,
            fields=fields,
        )

    @staticmethod
    def homogeneous_row_object_name(rows: list[CellProfilerKwargDict]) -> str | None:
        names = tuple(
            dict.fromkeys(
                str(row_mapping[MEASUREMENT_OBJECT_NAME_FIELD])
                for row in rows
                if (
                    row_mapping := measurement_row_mapping(row)
                ).get(MEASUREMENT_OBJECT_NAME_FIELD)
                not in (None, "")
            )
        )
        if len(names) == 1:
            return names[0]
        return None


class CalculateMathRoundingMethod(Enum):
    """CalculateMath rounding modes and their CellProfiler UI literals."""

    def __new__(
        cls,
        absorbed_value: str,
        *cellprofiler_literals: str,
    ) -> "CalculateMathRoundingMethod":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    NOT_ROUNDED = ("not_rounded", "Not rounded")
    DECIMAL_PLACES = (
        "decimal_places",
        "Rounded to a specified number of decimal places",
    )
    FLOOR = ("floor", "Rounded down to the next-lowest integer")
    CEILING = ("ceiling", "Rounded up to the next-highest integer")

    @classmethod
    def from_cellprofiler_literal(
        cls,
        value: "CalculateMathRoundingMethod | str",
    ) -> "CalculateMathRoundingMethod":
        """Return the rounding mode named by a CellProfiler setting literal."""
        return coerce_cellprofiler_enum(cls, value)


@dataclass(frozen=True, slots=True)
class CalculateMathSettingValue:
    """One CalculateMath setting with default and required-setting semantics."""

    module: "ModuleBlock"
    setting_name: str | SettingNameFamily
    default: str | None = None

    @property
    def value(self) -> str:
        value = optional_setting_value(self.module, self.setting_name)
        if value is not None:
            return value
        if self.default is not None:
            return self.default
        raise ValueError(f"CalculateMath requires setting {self.setting_name!r}.")


@dataclass(frozen=True, slots=True)
class IndexedCalculateMathSettingValue:
    """Repeated CalculateMath setting value selected by operand index."""

    module: "ModuleBlock"
    setting_name: str
    index: int
    default: str

    @property
    def value(self) -> str:
        values = self.module.get_setting_values(self.setting_name)
        if self.index < len(values):
            return values[self.index]
        return self.default


@dataclass(frozen=True, slots=True)
class TypedCalculateMathSettingValue:
    """CalculateMath setting parsed through the shared settings binder."""

    module: "ModuleBlock"
    binder: SettingsBinder
    setting_name: str
    default: str
    index: int = 0

    @property
    def value(self) -> Any:
        return self.binder.parse_value(
            self.setting_name,
            IndexedCalculateMathSettingValue(
                self.module,
                self.setting_name,
                self.index,
                self.default,
            ).value,
        )


@dataclass(frozen=True, slots=True)
class CalculateMathObjectSetting:
    """Optional CalculateMath object selector normalized as an artifact symbol."""

    module: "ModuleBlock"
    setting_name: SettingNameFamily

    @property
    def object_name(self) -> str | None:
        return OptionalSettingSymbol(self.module, self.setting_name).value


@dataclass(frozen=True, slots=True)
class CalculateMathOperandSettings:
    """One CalculateMath operand settings row."""

    module: "ModuleBlock"
    binder: SettingsBinder
    object_setting: SettingNameFamily
    measurement_setting: SettingNameFamily
    operand_index: int

    @property
    def feature_name(self) -> str:
        return CalculateMathSettingValue(
            self.module,
            self.measurement_setting,
        ).value

    @property
    def object_name(self) -> str | None:
        return CalculateMathObjectSetting(
            self.module,
            self.object_setting,
        ).object_name

    @property
    def multiplicand(self) -> Any:
        return TypedCalculateMathSettingValue(
            self.module,
            self.binder,
            "Multiply the above operand by",
            "1.0",
            self.operand_index,
        ).value

    @property
    def exponent(self) -> Any:
        return TypedCalculateMathSettingValue(
            self.module,
            self.binder,
            "Raise the power of above operand by",
            "1.0",
            self.operand_index,
        ).value


@dataclass(frozen=True, slots=True)
class CalculateMathBoundSettings:
    """Runtime kwargs for absorbed CalculateMath execution."""

    module: "ModuleBlock"
    binder: SettingsBinder
    settings: type[Any]

    @property
    def operand1(self) -> CalculateMathOperandSettings:
        return CalculateMathOperandSettings(
            module=self.module,
            binder=self.binder,
            object_setting=self.settings.numerator_objects_setting,
            measurement_setting=self.settings.numerator_measurement_setting,
            operand_index=0,
        )

    @property
    def operand2(self) -> CalculateMathOperandSettings:
        return CalculateMathOperandSettings(
            module=self.module,
            binder=self.binder,
            object_setting=self.settings.denominator_objects_setting,
            measurement_setting=self.settings.denominator_measurement_setting,
            operand_index=1,
        )

    def typed_setting(self, setting_name: str, default: str) -> Any:
        return TypedCalculateMathSettingValue(
            self.module,
            self.binder,
            setting_name,
            default,
        ).value

    @property
    def kwargs(self) -> dict[str, Any]:
        return {
            "output_name": CalculateMathSettingValue(
                self.module,
                self.settings.output_measurement_setting,
                default="Measurement",
            ).value,
            "operation": CalculateMathSettingValue(
                self.module,
                self.settings.operation_setting,
                default="None",
            ).value,
            "operand1_feature": self.operand1.feature_name,
            "operand2_feature": self.operand2.feature_name,
            "operand1_object_name": self.operand1.object_name,
            "operand2_object_name": self.operand2.object_name,
            "operand1_multiplicand": self.operand1.multiplicand,
            "operand1_exponent": self.operand1.exponent,
            "operand2_multiplicand": self.operand2.multiplicand,
            "operand2_exponent": self.operand2.exponent,
            "take_log10": self.typed_setting("Take log10 of result?", "No"),
            "final_multiplicand": self.typed_setting("Multiply the result by", "1.0"),
            "final_exponent": self.typed_setting(
                "Raise the power of result by",
                "1.0",
            ),
            "final_addend": self.typed_setting("Add to the result", "0.0"),
            "rounding": CalculateMathRoundingMethod.from_cellprofiler_literal(
                CalculateMathSettingValue(
                    self.module,
                    "How should the output value be rounded?",
                    default="Not rounded",
                ).value
            ),
            "rounding_digits": self.typed_setting(
                "Enter how many decimal places the value should be rounded to",
                "0",
            ),
            "constrain_lower_bound": self.typed_setting(
                "Constrain the result to a lower bound?",
                "No",
            ),
            "lower_bound": self.typed_setting("Enter the lower bound", "0.0"),
            "constrain_upper_bound": self.typed_setting(
                "Constrain the result to an upper bound?",
                "No",
            ),
            "upper_bound": self.typed_setting("Enter the upper bound", "1.0"),
        }


@dataclass(frozen=True, slots=True)
class CalculateMathObjectDependencies:
    """Object dependencies referenced by CalculateMath measurement operands."""

    module: "ModuleBlock"
    settings: type[Any]

    @property
    def object_names(self) -> tuple[str, ...]:
        names = (
            CalculateMathObjectSetting(
                self.module,
                self.settings.numerator_objects_setting,
            ).object_name,
            CalculateMathObjectSetting(
                self.module,
                self.settings.denominator_objects_setting,
            ).object_name,
            count_feature_object_name(
                optional_setting_value(
                    self.module,
                    self.settings.numerator_measurement_setting,
                )
            ),
            count_feature_object_name(
                optional_setting_value(
                    self.module,
                    self.settings.denominator_measurement_setting,
                )
            ),
        )
        return tuple(dict.fromkeys(name for name in names if name is not None))


def calculate_math_bound_kwargs(
    module: "ModuleBlock",
    binder: SettingsBinder,
) -> dict[str, Any]:
    """Return absorbed-function kwargs for runtime CalculateMath operands."""
    return CalculateMathBoundSettings(
        module=module,
        binder=binder,
        settings=CalculateMathModule,
    ).kwargs


def calculate_math_object_dependencies(module: "ModuleBlock") -> tuple[str, ...]:
    """Return object names referenced by CalculateMath measurement operands."""
    return CalculateMathObjectDependencies(
        module,
        CalculateMathModule,
    ).object_names


class CalculateMathModule(
    HomogeneousObjectNameMeasurementRecordMixin,
    TableMeasurementRecordRowsMixin,
    NoSourceMeasurementRecordMixin,
    MeasurementDebugViewModule,
    CalculateMathInputPolicy,
    BinderSettingsSourceModule,
):
    module_name = 'CalculateMath'
    function_name = 'calculate_math'
    validated = True
    contract = 'unknown'
    confidence = 1.0
    output_measurement_setting = SettingNameFamily("Name the output measurement")
    operation_setting = SettingNameFamily("Operation")
    numerator_objects_setting = SettingNameFamily("Select the numerator objects")
    numerator_measurement_setting = SettingNameFamily("Select the numerator measurement")
    denominator_objects_setting = SettingNameFamily("Select the denominator objects")
    denominator_measurement_setting = SettingNameFamily(
        "Select the denominator measurement"
    )
    settings_source = staticmethod(calculate_math_bound_kwargs)

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        from openhcs.core.artifacts import ArtifactKind, ArtifactSpec

        inputs = [
            builder.require_artifact(ArtifactSpec(name, ArtifactKind.OBJECT_LABELS), module)
            for name in calculate_math_object_dependencies(module)
        ]
        inputs.extend(builder.measurement_outputs())
        outputs = [
            builder.declare_artifact(
                ArtifactSpec(cls.measurement_artifact_name(module), ArtifactKind.MEASUREMENTS),
                module,
            )
        ]
        return assembler.assemble_contract(module, builder, inputs=inputs, outputs=outputs)



from abc import ABC, abstractmethod
from types import MappingProxyType
from dataclasses import dataclass, replace
from typing import Any, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_values import ColumnarRows
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer
RoundingMethod = CalculateMathRoundingMethod
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation as MathOperation,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


@dataclass
class MathResult:
    """Result row emitted by CalculateMath measurement execution."""

    slice_index: int
    output_name: str
    feature_name: str
    result_value: float
    operand1_value: float
    operand2_value: float
    operation: str
    object_label: int | None = None
    object_name: str | None = None

    @classmethod
    def from_mapping(
        cls,
        row: Any,
        *,
        slice_index: int,
    ) -> "MathResult":
        """Project one columnar CalculateMath row into the scalar row record."""
        return cls(
            slice_index=slice_index,
            output_name=str(row["output_name"]),
            feature_name=str(row["feature_name"]),
            result_value=float_or_nan(row["result_value"]),
            operand1_value=float_or_nan(row["operand1_value"]),
            operand2_value=float_or_nan(row["operand2_value"]),
            operation=str(row["operation"]),
            object_label=optional_int(row.get("object_label")),
            object_name=optional_str(row.get("object_name")),
        )


@dataclass(frozen=True, slots=True)
class MathResultColumnarRows(ColumnarRows):
    """Columnar CalculateMath result rows for object-vector outputs."""

    columns: MappingProxyType[str, Any]

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __iter__(self):
        yield from self.row_mappings()


@dataclass(frozen=True)
class MathPowerTransform(ABC):
    """Shared multiplicative/exponential transform."""

    multiplicand: float
    exponent: float


@dataclass(frozen=True)
class MathOperand(MathPowerTransform):
    """One CellProfiler CalculateMath operand and its pre-transform."""

    value: Any

    @property
    def transformed(self) -> Any:
        return np.power(
            np.asarray(self.value, dtype=float) * self.multiplicand,
            self.exponent,
        )


@dataclass(frozen=True)
class MathFinalTransform(MathPowerTransform):
    """Post-operation transform for non-identity math operations."""

    addend: float


@dataclass(frozen=True)
class MathBounds:
    """Optional scalar bounds for CalculateMath output."""

    constrain_lower: bool
    lower: float
    constrain_upper: bool
    upper: float


@dataclass(frozen=True)
class MathCalculationRequest:
    """Typed request for CellProfiler CalculateMath execution."""

    operand1: MathOperand
    operand2: MathOperand
    operation: MathOperation
    take_log10: bool
    final: MathFinalTransform
    rounding: RoundingMethod
    rounding_digits: int
    bounds: MathBounds
    output_name: str
    object_names: tuple[str, ...]

    def for_operand_values(
        self,
        *,
        operand1_value: Any,
        operand2_value: Any,
    ) -> "MathCalculationRequest":
        """Return this request with replacement operand values."""
        return MathCalculationRequest(
            operand1=MathOperand(
                value=operand1_value,
                multiplicand=self.operand1.multiplicand,
                exponent=self.operand1.exponent,
            ),
            operand2=MathOperand(
                value=operand2_value,
                multiplicand=self.operand2.multiplicand,
                exponent=self.operand2.exponent,
            ),
            operation=self.operation,
            take_log10=self.take_log10,
            final=self.final,
            rounding=self.rounding,
            rounding_digits=self.rounding_digits,
            bounds=self.bounds,
            output_name=self.output_name,
            object_names=self.object_names,
        )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "math_results",
        csv_materializer(
            fields=[
                "slice_index",
                "object_name",
                "object_label",
                "output_name",
                "feature_name",
                "result_value",
                "operand1_value",
                "operand2_value",
                "operation",
            ],
            analysis_type="math",
        ),
    )
)
def calculate_math(
    image: np.ndarray,
    operand1_value: Any = 0.0,
    operand2_value: Any = 0.0,
    operand1_feature: str | None = None,
    operand2_feature: str | None = None,
    operand1_object_name: str | None = None,
    operand2_object_name: str | None = None,
    operation: MathOperation = MathOperation.NONE,
    operand1_multiplicand: float = 1.0,
    operand1_exponent: float = 1.0,
    operand2_multiplicand: float = 1.0,
    operand2_exponent: float = 1.0,
    take_log10: bool = False,
    final_multiplicand: float = 1.0,
    final_exponent: float = 1.0,
    final_addend: float = 0.0,
    rounding: RoundingMethod = RoundingMethod.NOT_ROUNDED,
    rounding_digits: int = 0,
    constrain_lower_bound: bool = False,
    lower_bound: float = 0.0,
    constrain_upper_bound: bool = False,
    upper_bound: float = 1.0,
    output_name: str = "Measurement",
) -> tuple[np.ndarray, MathResult | list[MathResult]]:
    """Perform CellProfiler CalculateMath measurement-row execution."""
    del operand1_feature, operand2_feature
    request = MathCalculationRequest(
        operand1=MathOperand(
            value=operand1_value,
            multiplicand=operand1_multiplicand,
            exponent=operand1_exponent,
        ),
        operand2=MathOperand(
            value=operand2_value,
            multiplicand=operand2_multiplicand,
            exponent=operand2_exponent,
        ),
        operation=operation,
        take_log10=take_log10,
        final=MathFinalTransform(
            multiplicand=final_multiplicand,
            exponent=final_exponent,
            addend=final_addend,
        ),
        rounding=rounding,
        rounding_digits=rounding_digits,
        bounds=MathBounds(
            constrain_lower=constrain_lower_bound,
            lower=lower_bound,
            constrain_upper=constrain_upper_bound,
            upper=upper_bound,
        ),
        output_name=output_name,
        object_names=tuple(
            dict.fromkeys(
                name
                for name in (operand1_object_name, operand2_object_name)
                if name is not None
            )
        ),
    )
    return image, CalculateMathExecution(request).result_rows


class MathOperationStrategy(
    EnumKeyedStrategyMixin[MathOperation],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal strategy for the closed CalculateMath operation family."""

    __registry_key__ = "operation_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "operation"
    __enum_label_attr__ = "operation_label"

    operation: ClassVar[MathOperation | None] = None
    operation_label: ClassVar[str | None] = None

    @classmethod
    def for_operation(cls, operation: MathOperation) -> "MathOperationStrategy":
        return cls.for_enum_member(operation)

    @abstractmethod
    def apply(self, request: MathCalculationRequest) -> Any:
        """Return the raw operation result before post-processing."""


class NoneOperationStrategy(MathOperationStrategy):
    operation = MathOperation.NONE

    def apply(self, request: MathCalculationRequest) -> Any:
        return request.operand1.transformed


class AddOperationStrategy(MathOperationStrategy):
    operation = MathOperation.ADD

    def apply(self, request: MathCalculationRequest) -> Any:
        return request.operand1.transformed + request.operand2.transformed


class SubtractOperationStrategy(MathOperationStrategy):
    operation = MathOperation.SUBTRACT

    def apply(self, request: MathCalculationRequest) -> Any:
        return request.operand1.transformed - request.operand2.transformed


class MultiplyOperationStrategy(MathOperationStrategy):
    operation = MathOperation.MULTIPLY

    def apply(self, request: MathCalculationRequest) -> Any:
        return request.operand1.transformed * request.operand2.transformed


class DivideOperationStrategy(MathOperationStrategy):
    operation = MathOperation.DIVIDE

    def apply(self, request: MathCalculationRequest) -> Any:
        denominator = request.operand2.transformed
        with np.errstate(divide="ignore", invalid="ignore"):
            result = request.operand1.transformed / denominator
        if np.isscalar(result) or np.asarray(result).ndim == 0:
            return np.nan if float(denominator) == 0.0 else result
        return np.where(denominator == 0, np.nan, result)


class RoundingStrategy(
    EnumKeyedStrategyMixin[RoundingMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal strategy for the closed CalculateMath rounding family."""

    __registry_key__ = "rounding_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "rounding"
    __enum_label_attr__ = "rounding_label"

    rounding: ClassVar[RoundingMethod | None] = None
    rounding_label: ClassVar[str | None] = None

    @classmethod
    def for_rounding(cls, rounding: RoundingMethod) -> "RoundingStrategy":
        return cls.for_enum_member(rounding)

    @abstractmethod
    def apply(self, value: Any, request: MathCalculationRequest) -> Any:
        """Return rounded value."""


class NotRoundedStrategy(RoundingStrategy):
    rounding = RoundingMethod.NOT_ROUNDED

    def apply(self, value: Any, request: MathCalculationRequest) -> Any:
        del request
        return value


class DecimalPlacesRoundingStrategy(RoundingStrategy):
    rounding = RoundingMethod.DECIMAL_PLACES

    def apply(self, value: Any, request: MathCalculationRequest) -> Any:
        return np.around(value, request.rounding_digits)


class FloorRoundingStrategy(RoundingStrategy):
    rounding = RoundingMethod.FLOOR

    def apply(self, value: Any, request: MathCalculationRequest) -> Any:
        del request
        return np.floor(value)


class CeilingRoundingStrategy(RoundingStrategy):
    rounding = RoundingMethod.CEILING

    def apply(self, value: Any, request: MathCalculationRequest) -> Any:
        del request
        return np.ceil(value)


@dataclass(frozen=True)
class MathOperandSliceAlignment:
    """Slice alignment policy for CalculateMath runtime operands."""

    request: MathCalculationRequest

    @property
    def aligned_operands(self) -> tuple[tuple[int, Any, Any], ...] | None:
        operand1 = self.request.operand1.value
        operand2 = self.request.operand2.value
        aligned_values = tuple(
            value
            for value in (operand1, operand2)
            if isinstance(value, RuntimeSliceAlignedValueSet)
        )
        if not aligned_values:
            return None
        slice_count = max(value.slice_count for value in aligned_values)
        if any(slice_count % value.slice_count != 0 for value in aligned_values):
            raise ValueError(
                "CalculateMath aligned operands must have compatible slice counts."
            )
        return tuple(
            (
                slice_index,
                self.operand_value_for_slice(operand1, slice_index, slice_count),
                self.operand_value_for_slice(operand2, slice_index, slice_count),
            )
            for slice_index in range(slice_count)
        )

    @staticmethod
    def operand_value_for_slice(value: Any, slice_index: int, slice_count: int) -> Any:
        if isinstance(value, RuntimeSliceAlignedValueSet):
            return value.value_for_aligned_slice(slice_index, slice_count)
        return value


@dataclass(frozen=True)
class MathResultRows:
    """Materialize CalculateMath scalar/vector outputs into measurement rows."""

    result: Any
    request: MathCalculationRequest

    @property
    def rows(self) -> MathResult | MathResultColumnarRows:
        result_values = np.asarray(self.result, dtype=float)
        feature_name = f"Math_{self.request.output_name}"
        if result_values.ndim == 0:
            return MathResult(
                slice_index=0,
                output_name=self.request.output_name,
                feature_name=feature_name,
                result_value=float_or_nan(result_values.item()),
                operand1_value=scalar_operand_value(self.request.operand1.value),
                operand2_value=scalar_operand_value(self.request.operand2.value),
                operation=self.request.operation.value,
                object_name=next(iter(self.request.object_names), None),
            )

        flat_results = result_values.reshape(-1)
        object_names = self.request.object_names or (None,)
        operand1_values = broadcast_operand_values(
            self.request.operand1.value,
            len(flat_results),
        )
        operand2_values = broadcast_operand_values(
            self.request.operand2.value,
            len(flat_results),
        )
        object_count = len(flat_results)
        row_count = object_count * len(object_names)
        object_labels = np.tile(np.arange(1, object_count + 1), len(object_names))
        result_column = np.tile(
            np.asarray([float_or_nan(value) for value in flat_results], dtype=float),
            len(object_names),
        )
        operand1_column = np.tile(
            np.asarray([float_or_nan(value) for value in operand1_values], dtype=float),
            len(object_names),
        )
        operand2_column = np.tile(
            np.asarray([float_or_nan(value) for value in operand2_values], dtype=float),
            len(object_names),
        )
        return MathResultColumnarRows(
            MappingProxyType(
                {
                    "slice_index": np.zeros(row_count, dtype=np.int64),
                    "object_name": tuple(
                        object_name
                        for object_name in object_names
                        for _index in range(object_count)
                    ),
                    "object_label": object_labels,
                    "output_name": (self.request.output_name,) * row_count,
                    "feature_name": (feature_name,) * row_count,
                    "result_value": result_column,
                    "operand1_value": operand1_column,
                    "operand2_value": operand2_column,
                    "operation": (self.request.operation.value,) * row_count,
                }
            )
        )


@dataclass(frozen=True)
class CalculateMathExecution:
    """Execute CellProfiler CalculateMath semantics for one runtime request."""

    request: MathCalculationRequest

    @property
    def result_rows(self) -> MathResult | MathResultColumnarRows | list[MathResult]:
        aligned_operands = MathOperandSliceAlignment(self.request).aligned_operands
        if aligned_operands is None:
            return MathResultRows(self.scalar_result(self.request), self.request).rows

        rows: list[MathResult] = []
        for slice_index, operand1_value, operand2_value in aligned_operands:
            slice_request = self.request.for_operand_values(
                operand1_value=operand1_value,
                operand2_value=operand2_value,
            )
            slice_rows = MathResultRows(
                self.scalar_result(slice_request),
                slice_request,
            ).rows
            rows.extend(math_results_with_slice_index(slice_rows, slice_index))
        return rows

    @staticmethod
    def scalar_result(request: MathCalculationRequest) -> Any:
        result = MathOperationStrategy.for_operation(request.operation).apply(request)

        if request.take_log10:
            result = np.where(result > 0, np.log10(result), np.nan)

        if request.operation is not MathOperation.NONE:
            result *= request.final.multiplicand
            result = np.power(result, request.final.exponent)

        result += request.final.addend
        result = RoundingStrategy.for_rounding(request.rounding).apply(result, request)

        if request.bounds.constrain_lower:
            result = np.where(
                np.isnan(result),
                result,
                np.maximum(result, request.bounds.lower),
            )
        if request.bounds.constrain_upper:
            result = np.where(
                np.isnan(result),
                result,
                np.minimum(result, request.bounds.upper),
            )
        return result


def math_results_with_slice_index(
    rows: MathResult | MathResultColumnarRows | list[MathResult],
    slice_index: int,
) -> list[MathResult]:
    """Return scalar result rows with the runtime slice index attached."""
    if isinstance(rows, MathResultColumnarRows):
        return [
            MathResult.from_mapping(row, slice_index=slice_index)
            for row in rows.row_mappings()
        ]
    return [replace(row, slice_index=slice_index) for row in as_result_list(rows)]


def as_result_list(rows: MathResult | list[MathResult]) -> list[MathResult]:
    return rows if isinstance(rows, list) else [rows]


def broadcast_operand_values(value: Any, count: int) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == count:
        return values
    if values.size == 1:
        return np.full(count, float_or_nan(values[0]))
    raise ValueError(
        f"CalculateMath operand produced {values.size} values for {count} results."
    )


def scalar_operand_value(value: Any) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1:
        return np.nan
    return float_or_nan(values[0])


def float_or_nan(value: Any) -> float:
    scalar = float(value)
    return scalar if not np.isnan(scalar) else np.nan


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return int(value)


def optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


__all__ = public_names_from_objects(
    CalculateMathExecution,
    MathBounds,
    MathCalculationRequest,
    MathFinalTransform,
    MathOperand,
    MathOperandSliceAlignment,
    MathOperationStrategy,
    MathPowerTransform,
    MathResult,
    MathResultRows,
    RoundingStrategy,
    as_result_list,
    broadcast_operand_values,
    calculate_math,
    float_or_nan,
    math_results_with_slice_index,
    optional_int,
    optional_str,
    scalar_operand_value,
)
