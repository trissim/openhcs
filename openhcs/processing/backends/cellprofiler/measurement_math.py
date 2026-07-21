"""CalculateMath measurement semantics for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.core.artifacts import ArtifactInputPlan

from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import (
    KeywordRuntimeParameter,
    runtime_image_execution_mode,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_feature_queries import MeasurementFeatureQuery
from openhcs.core.measurement_row_materialization import ConcatenatedColumnarRows
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    enum_member_with_payload,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelPlaneDomainStrategy,
)
from openhcs.core.runtime_slice_alignment import (
    RuntimeSliceAlignedValueSet,
    RuntimeSliceAlignedValues,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_tabular_values import (
    ColumnarRows,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.measurement_lookup import count_feature_object_name
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerMeasurementVector,
    CellProfilerObjectMeasurementVectorBatchBinding,
    CellProfilerObjectMeasurementVectorBinding,
    MeasurementImageOperandVectorResolution,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.interop.cellprofiler.settings_binder import (
    MeasurementFeatureSettingBinding,
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    PriorMeasurementArtifactInputModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    normalized_symbol_name,
    optional_setting_value,
    setting_names,
    setting_values,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation as MathOperation,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )


class _CalculateMathNumeratorObjectRelation(ArtifactSpecRelation):
    """Bind the numerator operand to one exact object-label input."""

    relation_key = "calculate_math_numerator_object"


class _CalculateMathDenominatorObjectRelation(ArtifactSpecRelation):
    """Bind the denominator operand to one exact object-label input."""

    relation_key = "calculate_math_denominator_object"


class _CalculateMathObjectNamesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound object identities represented by CalculateMath rows."""

    parameter_name = "object_names"
    annotation_type = tuple[str, ...]
    parameter_default = ()


class _CalculateMathOperand1ValueRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound numerator resolved from compiled measurement inputs."""

    parameter_name = "operand1_value"
    annotation_type = Any
    parameter_default = 0.0


class _CalculateMathOperand2ValueRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound denominator resolved from compiled measurement inputs."""

    parameter_name = "operand2_value"
    annotation_type = Any
    parameter_default = 0.0


class CalculateMathInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind CalculateMath operands from runtime measurement/object state."""

    binds_without_declared_inputs = True
    supported_non_object_input_kinds = frozenset({MeasurementsArtifactType})

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        started_at = time.perf_counter()
        bound = super().bind_runtime_inputs(request)
        operand_object_specs = cls.operand_object_specs(request)
        operand_bindings = cls.object_operand_bindings(
            request,
            operand_object_specs=operand_object_specs,
        )
        if operand_bindings is not None:
            vectors = CellProfilerObjectMeasurementVectorBatchBinding(
                operand_bindings
            ).vectors()
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "calculate_math_bind_total", time.perf_counter() - started_at
            )
            return {
                **bound,
                **cls.bound_operand_kwargs(
                    operand_object_specs,
                    operand1_value=vectors[0].runtime_value,
                    operand2_value=vectors[1].runtime_value,
                ),
            }
        operand1_started_at = time.perf_counter()
        operand1_value = cls.operand_value(
            request,
            feature_kwarg="operand1_feature",
            object_spec=operand_object_specs[0],
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand1_started_at,
            operand="1",
        )
        operand2_started_at = time.perf_counter()
        operand2_value = cls.operand_value(
            request,
            feature_kwarg="operand2_feature",
            object_spec=operand_object_specs[1],
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_operand_bind",
            time.perf_counter() - operand2_started_at,
            operand="2",
        )
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "calculate_math_bind_total", time.perf_counter() - started_at
        )
        return {
            **bound,
            **cls.bound_operand_kwargs(
                operand_object_specs,
                operand1_value=operand1_value,
                operand2_value=operand2_value,
            ),
        }

    @classmethod
    def bound_operand_kwargs(
        cls,
        operand_object_specs: tuple[ArtifactSpec | None, ArtifactSpec | None],
        *,
        operand1_value: RuntimeCallableArgument,
        operand2_value: RuntimeCallableArgument,
    ) -> dict[str, RuntimeCallableArgument]:
        """Return values and exact object identities supplied by the executor."""
        return {
            _CalculateMathOperand1ValueRuntimeParameter.require_parameter_name(): operand1_value,
            _CalculateMathOperand2ValueRuntimeParameter.require_parameter_name(): operand2_value,
            _CalculateMathObjectNamesRuntimeParameter.require_parameter_name(): tuple(
                dict.fromkeys(
                    spec.name for spec in operand_object_specs if spec is not None
                )
            ),
        }

    @classmethod
    def operand_object_specs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> tuple[ArtifactSpec | None, ArtifactSpec | None]:
        """Resolve operand object identities from exact output-contract relations."""

        return (
            cls._operand_object_spec(request, _CalculateMathNumeratorObjectRelation),
            cls._operand_object_spec(request, _CalculateMathDenominatorObjectRelation),
        )

    @staticmethod
    def _operand_object_spec(
        request: RuntimeInputBindingRequest,
        relation_type: type[ArtifactSpecRelation],
    ) -> ArtifactSpec | None:
        relations = tuple(
            relation
            for output in request.adapter.request.require_callable_contract().artifact_outputs
            if output.artifact_type is MeasurementsArtifactType
            for relation in output.relations
            if isinstance(relation, relation_type)
        )
        if not relations:
            return None
        if len(relations) != 1:
            raise ValueError(
                f"CalculateMath requires at most one {relation_type.__name__}; "
                f"got {len(relations)}."
            )
        object_spec = ArtifactSpecCollection(
            ArtifactSpecCollection(request.object_inputs).unique(
                conflict_context="CalculateMath object input"
            )
        ).by_ref(relations[0].source)
        if object_spec is None:
            raise ValueError(
                "CalculateMath operand relation references an object input absent "
                f"from the active runtime contract: {relations[0].source!r}."
            )
        return object_spec

    @classmethod
    def object_operand_bindings(
        cls,
        request: RuntimeInputBindingRequest,
        *,
        operand_object_specs: tuple[ArtifactSpec | None, ArtifactSpec | None],
    ) -> tuple[CellProfilerObjectMeasurementVectorBinding, ...] | None:
        del cls
        bindings: list[CellProfilerObjectMeasurementVectorBinding] = []
        for feature_kwarg, object_spec in zip(
            ("operand1_feature", "operand2_feature"),
            operand_object_specs,
            strict=True,
        ):
            feature_name = request.require_string_kwarg(feature_kwarg)
            if (
                object_spec is None
                or count_feature_object_name(feature_name) is not None
            ):
                return None
            bindings.append(
                CellProfilerObjectMeasurementVectorBinding.for_object(
                    request, object_ref=object_spec, feature_name=feature_name
                )
            )
        return tuple(bindings)

    @classmethod
    def operand_value(
        cls,
        request: RuntimeInputBindingRequest,
        *,
        feature_kwarg: str,
        object_spec: ArtifactSpec | None,
    ) -> RuntimeCallableArgument:
        feature_name = request.require_string_kwarg(feature_kwarg)
        count_object_name = count_feature_object_name(feature_name)
        if count_object_name is not None:
            labels = request.adapter.get_objects(count_object_name)
            domain = labels.object_label_domain()
            object_id_domains = ObjectLabelPlaneDomainStrategy.for_enum_member(
                domain.scope
            ).plane_domains(
                labels,
                domain=domain,
            )
            counts = tuple(float(len(object_ids)) for object_ids in object_id_domains)
            if labels.declared_plane_projection() is None:
                return counts[0]
            return RuntimeSliceAlignedValues(counts)
        if object_spec is None:
            return cls.image_operand_value(request, feature_name)
        return (
            CellProfilerObjectMeasurementVectorBinding.for_object(
                request, object_ref=object_spec, feature_name=feature_name
            )
            .vector()
            .runtime_value
        )

    @classmethod
    def image_operand_value(
        cls,
        request: RuntimeInputBindingRequest,
        feature_name: str,
    ) -> RuntimeCallableArgument:
        del cls
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
                feature_name, dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
            ).scalar_value(declared_measurement_tables)
        tables_started_at = time.perf_counter()
        measurement_resolution = (
            MeasurementImageOperandVectorResolution.from_runtime_feature(
                request.adapter, feature_name
            )
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
                feature_name, dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
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
    def measurement_record_object_name(
        cls,
        request: "CellProfilerOutputRecordRequest",
        rows: ColumnarRows,
    ) -> str | None:
        """Use row ownership when all rows declare the same object set."""
        object_name = cls.homogeneous_row_object_name(rows)
        if object_name is not None:
            return object_name
        return super().measurement_record_object_name(request, rows)

    @staticmethod
    def homogeneous_row_object_name(rows: ColumnarRows) -> str | None:
        object_name_field = MeasurementRowAxisField.OBJECT_NAME.value
        if object_name_field not in rows.columns:
            return None
        return HomogeneousObjectNameMeasurementRecordMixin._single_nonempty_object_name(
            rows.column_values(object_name_field)
        )

    @staticmethod
    def _single_nonempty_object_name(values: Any) -> str | None:
        candidate = None
        for value in values:
            if value in (None, ""):
                continue
            name = str(value)
            if candidate is None:
                candidate = name
                continue
            if name != candidate:
                return None
        return candidate


class CalculateMathRoundingMethod(Enum):
    """CalculateMath rounding modes and their CellProfiler UI literals."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
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
        cls, value: "CalculateMathRoundingMethod | str"
    ) -> "CalculateMathRoundingMethod":
        """Return the rounding mode named by a CellProfiler setting literal."""
        return coerce_cellprofiler_enum(cls, value)


class CalculateMathModule(
    HomogeneousObjectNameMeasurementRecordMixin,
    CalculateMathInputPolicy,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
):
    module_name = "CalculateMath"
    function_name = "calculate_math"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("math",),)
    calculated_measurement_feature_prefixes = (("math",),)

    output_measurement_setting = SettingNameFamily("Name the output measurement")
    operation_setting = SettingNameFamily("Operation")
    numerator_objects_setting = SettingNameFamily("Select the numerator objects")
    numerator_measurement_setting = SettingNameFamily(
        "Select the numerator measurement"
    )
    denominator_objects_setting = SettingNameFamily("Select the denominator objects")
    denominator_measurement_setting = SettingNameFamily(
        "Select the denominator measurement"
    )
    numerator_measurement_type_setting = SettingNameFamily(
        "Select the numerator measurement type"
    )
    denominator_measurement_type_setting = SettingNameFamily(
        "Select the denominator measurement type"
    )
    operand_multiplicand_setting = SettingNameFamily("Multiply the above operand by")
    operand_exponent_setting = SettingNameFamily("Raise the power of above operand by")
    operand1_multiplicand_binding = SettingToKeywordBinding(
        operand_multiplicand_setting,
        "operand1_multiplicand",
    )
    operand1_exponent_binding = SettingToKeywordBinding(
        operand_exponent_setting,
        "operand1_exponent",
    )
    operand2_multiplicand_binding = SettingToKeywordBinding(
        operand_multiplicand_setting,
        "operand2_multiplicand",
    )
    operand2_exponent_binding = SettingToKeywordBinding(
        operand_exponent_setting,
        "operand2_exponent",
    )
    indexed_operand_setting_bindings = (
        (operand1_multiplicand_binding, 0),
        (operand1_exponent_binding, 0),
        (operand2_multiplicand_binding, 1),
        (operand2_exponent_binding, 1),
    )
    take_log10_setting = SettingNameFamily("Take log10 of result?")
    final_multiplicand_setting = SettingNameFamily("Multiply the result by")
    final_exponent_setting = SettingNameFamily("Raise the power of result by")
    final_addend_setting = SettingNameFamily("Add to the result")
    rounding_setting = SettingNameFamily("How should the output value be rounded?")
    rounding_digits_setting = SettingNameFamily(
        "Enter how many decimal places the value should be rounded to"
    )
    constrain_lower_bound_setting = SettingNameFamily(
        "Constrain the result to a lower bound?"
    )
    lower_bound_setting = SettingNameFamily("Enter the lower bound")
    constrain_upper_bound_setting = SettingNameFamily(
        "Constrain the result to an upper bound?"
    )
    upper_bound_setting = SettingNameFamily("Enter the upper bound")
    numerator_objects_binding = SettingToKeywordBinding.input(
        numerator_objects_setting,
        ObjectLabelsArtifactType,
        parse=normalized_symbol_name,
    )
    denominator_objects_binding = SettingToKeywordBinding.input(
        denominator_objects_setting,
        ObjectLabelsArtifactType,
        parse=normalized_symbol_name,
    )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        """Return only operand rows that select an object measurement domain."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        object_bindings = frozenset(
            cls.declared_artifact_bindings(
                plan_type=ArtifactInputPlan,
                artifact_type=ObjectLabelsArtifactType,
            )
        )
        return tuple(
            binding
            for binding in bindings
            if binding not in object_bindings
            or normalized_symbol_name(
                optional_setting_value(module, binding.setting_name) or ""
            )
            is not None
        )

    numerator_measurement_binding = MeasurementFeatureSettingBinding(
        numerator_measurement_setting,
        "operand1_feature",
        str,
    )
    denominator_measurement_binding = MeasurementFeatureSettingBinding(
        denominator_measurement_setting,
        "operand2_feature",
        str,
    )
    setting_bindings = (
        numerator_objects_binding,
        denominator_objects_binding,
        SettingToKeywordBinding(output_measurement_setting, "output_name", str),
        SettingToKeywordBinding(operation_setting, "operation"),
        numerator_measurement_binding,
        denominator_measurement_binding,
        SettingToKeywordBinding(take_log10_setting, "take_log10"),
        SettingToKeywordBinding(
            final_multiplicand_setting,
            "final_multiplicand",
        ),
        SettingToKeywordBinding(final_exponent_setting, "final_exponent"),
        SettingToKeywordBinding(final_addend_setting, "final_addend"),
        SettingToKeywordBinding(
            rounding_setting,
            "rounding",
            CalculateMathRoundingMethod.from_cellprofiler_literal,
        ),
        SettingToKeywordBinding(rounding_digits_setting, "rounding_digits"),
        SettingToKeywordBinding(
            constrain_lower_bound_setting,
            "constrain_lower_bound",
        ),
        SettingToKeywordBinding(lower_bound_setting, "lower_bound"),
        SettingToKeywordBinding(
            constrain_upper_bound_setting,
            "constrain_upper_bound",
        ),
        SettingToKeywordBinding(upper_bound_setting, "upper_bound"),
    )
    ignored_settings = (
        numerator_measurement_type_setting,
        denominator_measurement_type_setting,
    )

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind public settings and privately parse the two repeated operand rows."""

        bound = cls._bind_declared_settings(module, binder=binder)
        kwargs = dict(bound.kwargs)
        for binding, index in cls.indexed_operand_setting_bindings:
            values = setting_values(module, binding.setting_name)
            raw_value = values[index] if index < len(values) else "1.0"
            kwargs[binding.require_parameter_name()] = binder.parse_value(
                setting_names(binding.setting_name)[0],
                raw_value,
            )
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in (
            cls.operand_multiplicand_setting,
            cls.operand_exponent_setting,
        ):
            for concrete_name in setting_name.names:
                unmapped_kwargs.pop(cls.normalize_setting_name(concrete_name), None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, unmapped_kwargs),
        )

    @classmethod
    def operand_object_names(
        cls,
        module: "ModuleBlock",
    ) -> tuple[str | None, str | None]:
        """Return object identities selected by the two operand rows."""

        return tuple(
            normalized_symbol_name(optional_setting_value(module, object_setting) or "")
            or count_feature_object_name(
                optional_setting_value(module, feature_setting)
            )
            for object_setting, feature_setting in zip(
                (
                    cls.numerator_objects_setting,
                    cls.denominator_objects_setting,
                ),
                (
                    cls.numerator_measurement_setting,
                    cls.denominator_measurement_setting,
                ),
                strict=True,
            )
        )

    @classmethod
    def artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
    ):
        inputs = ArtifactSpecCollection(
            super().artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
            )
        ).unique(conflict_context="CalculateMath artifact input")
        declared_object_names = ArtifactSpecCollection(
            inputs
        ).name_set_of_artifact_type(ObjectLabelsArtifactType)
        operand_object_bindings = (
            cls.numerator_objects_binding,
            cls.denominator_objects_binding,
        )
        operand_object_names = cls.operand_object_names(module)
        operand_object_bindings_by_name = tuple(
            (name, binding)
            for index, (name, binding) in enumerate(
                zip(
                    operand_object_names,
                    operand_object_bindings,
                    strict=True,
                )
            )
            if name is not None and name not in operand_object_names[:index]
        )
        return (
            *inputs,
            *(
                cls.require_available_artifact_input(
                    module,
                    binding=binding,
                    name=name,
                    invocation_key=invocation_key,
                    step_context=step_context,
                )
                for name, binding in operand_object_bindings_by_name
                if name not in declared_object_names
            ),
        )

    @classmethod
    def measurement_output_relations(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        provenance_relations = super().measurement_output_relations(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        operand_names = cls.operand_object_names(module)
        operand_relations = tuple(
            relation_type(
                source=artifact_inputs.require_by_name_and_artifact_type(
                    name,
                    ObjectLabelsArtifactType,
                ).ref()
            )
            for relation_type, name in zip(
                (
                    _CalculateMathNumeratorObjectRelation,
                    _CalculateMathDenominatorObjectRelation,
                ),
                operand_names,
                strict=True,
            )
            if name is not None
        )
        return (*provenance_relations, *operand_relations)


RoundingMethod = CalculateMathRoundingMethod


@dataclass(frozen=True, slots=True)
class MathResultColumnarRows(ColumnarRows):
    """Columnar CalculateMath result rows."""

    columns: MappingProxyType[str, Any]

    fields: ClassVar[tuple[FieldSpec, ...]] = (
        FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
        FieldSpec(
            MeasurementRowAxisField.OBJECT_NAME.value,
            str,
            required=False,
        ),
        FieldSpec(
            MeasurementRowAxisField.OBJECT_LABEL.value,
            int,
            required=False,
        ),
        FieldSpec("output_name", str),
        FieldSpec(MeasurementRowAxisField.FEATURE_NAME.value, str),
        FieldSpec(MeasurementRowValueField.RESULT_VALUE.value, float),
    )

    def __post_init__(self) -> None:
        self.validate_fields()

    def __len__(self) -> int:
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __iter__(self):
        yield from self.row_mappings()


@dataclass(frozen=True)
class MathPowerTransform(ABC):
    """Shared multiplicative/exponential transform."""

    multiplicand: float
    exponent: float

    def transform(self, value: Any) -> Any:
        """Apply this multiplicative/exponential transform."""
        return np.power(
            np.asarray(value, dtype=float) * self.multiplicand, self.exponent
        )


@dataclass(frozen=True)
class MathOperand(MathPowerTransform):
    """One CellProfiler CalculateMath operand and its pre-transform."""

    value: Any

    @property
    def transformed(self) -> Any:
        return self.transform(self.value)


@dataclass(frozen=True)
class MathFinalTransform(MathPowerTransform):
    """Post-operation transform for non-identity math operations."""

    addend: float

    def apply(self, value: Any, *, apply_power: bool) -> Any:
        """Apply the CellProfiler final transform to an operation result."""
        transformed = self.transform(value) if apply_power else value
        return transformed + self.addend


@dataclass(frozen=True)
class MathBounds:
    """Optional scalar bounds for CalculateMath output."""

    constrain_lower: bool
    lower: float
    constrain_upper: bool
    upper: float

    def apply(self, value: Any) -> Any:
        """Apply enabled lower and upper bounds while preserving NaN values."""
        if self.constrain_lower:
            value = np.where(np.isnan(value), value, np.maximum(value, self.lower))
        if self.constrain_upper:
            value = np.where(np.isnan(value), value, np.minimum(value, self.upper))
        return value


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
        self, *, operand1_value: Any, operand2_value: Any
    ) -> "MathCalculationRequest":
        """Return this request with replacement operand values."""
        return replace(
            self,
            operand1=replace(self.operand1, value=operand1_value),
            operand2=replace(self.operand2, value=operand2_value),
        )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(
    _CalculateMathOperand1ValueRuntimeParameter,
    _CalculateMathOperand2ValueRuntimeParameter,
    _CalculateMathObjectNamesRuntimeParameter,
)
def calculate_math(
    image: np.ndarray,
    operand1_value: Any = 0.0,
    operand2_value: Any = 0.0,
    operand1_feature: str | None = None,
    operand2_feature: str | None = None,
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
    *,
    object_names: tuple[str, ...] = (),
) -> tuple[np.ndarray, ColumnarRows]:
    """Perform CellProfiler CalculateMath measurement-row execution.

    Args:
        operand1_multiplicand: Factor applied to the first operand before its
            exponent.
        operand1_exponent: Power applied after scaling the first operand.
        operand2_multiplicand: Factor applied to the second operand before its
            exponent.
        operand2_exponent: Power applied after scaling the second operand.
    """
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
        object_names=tuple(dict.fromkeys(object_names)),
    )
    return (image, CalculateMathExecution(request).result_rows)


class MathOperationStrategy(
    EnumKeyedStrategyMixin[MathOperation], ABC, metaclass=AutoRegisterMeta
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
    EnumKeyedStrategyMixin[RoundingMethod], ABC, metaclass=AutoRegisterMeta
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
            (
                value
                for value in (operand1, operand2)
                if isinstance(value, RuntimeSliceAlignedValueSet)
            )
        )
        if not aligned_values:
            return None
        slice_count = aligned_values[0].slice_count
        if any(value.slice_count != slice_count for value in aligned_values[1:]):
            counts = tuple(value.slice_count for value in aligned_values)
            raise ValueError(
                "CalculateMath aligned operand cardinalities must match exactly; "
                f"got {counts!r}."
            )
        return tuple(
            (
                slice_index,
                RuntimeSliceProjection.value_for_slice(operand1, projection),
                RuntimeSliceProjection.value_for_slice(operand2, projection),
            )
            for slice_index in range(slice_count)
            for projection in (
                RuntimePlaneAxisValueProjection.from_selected_plane(
                    axis=RuntimePlaneAxis.RUNTIME_SLICE,
                    plane_index=slice_index,
                    axis_size=slice_count,
                ),
            )
        )


@dataclass(frozen=True)
class MathResultRows:
    """Project CalculateMath scalar/vector outputs into exact measurement rows."""

    result: Any
    request: MathCalculationRequest
    slice_index: int = 0

    @property
    def rows(self) -> MathResultColumnarRows:
        result_values = np.asarray(self.result, dtype=float)
        feature_name = f"Math_{self.request.output_name}"
        flat_results = result_values.reshape(-1)
        object_names = self.request.object_names if result_values.ndim > 0 else (None,)
        object_count = len(flat_results)
        row_count = object_count * len(object_names)
        object_labels = (
            np.tile(np.arange(1, object_count + 1), len(object_names))
            if result_values.ndim > 0
            else (None,)
        )
        return MathResultColumnarRows(
            MappingProxyType(
                {
                    "slice_index": np.full(
                        row_count,
                        self.slice_index,
                        dtype=np.int64,
                    ),
                    "object_name": tuple(
                        (
                            object_name
                            for object_name in object_names
                            for _index in range(object_count)
                        )
                    ),
                    MeasurementRowAxisField.OBJECT_LABEL.value: object_labels,
                    "output_name": (self.request.output_name,) * row_count,
                    MeasurementRowAxisField.FEATURE_NAME.value: (feature_name,)
                    * row_count,
                    MeasurementRowValueField.RESULT_VALUE.value: np.tile(
                        flat_results,
                        len(object_names),
                    ),
                }
            )
        )


@dataclass(frozen=True)
class CalculateMathExecution:
    """Execute CellProfiler CalculateMath semantics for one runtime request."""

    request: MathCalculationRequest

    @property
    def result_rows(self) -> ColumnarRows:
        aligned_operands = MathOperandSliceAlignment(self.request).aligned_operands
        if aligned_operands is None:
            return MathResultRows(self.scalar_result(self.request), self.request).rows
        row_batches: list[ColumnarRows] = []
        for slice_index, operand1_value, operand2_value in aligned_operands:
            slice_request = self.request.for_operand_values(
                operand1_value=operand1_value, operand2_value=operand2_value
            )
            row_batches.append(
                MathResultRows(
                    self.scalar_result(slice_request),
                    slice_request,
                    slice_index,
                ).rows
            )
        return ConcatenatedColumnarRows(tuple(row_batches))

    @staticmethod
    def scalar_result(request: MathCalculationRequest) -> Any:
        result = MathOperationStrategy.for_operation(request.operation).apply(request)
        if request.take_log10:
            result = np.where(result > 0, np.log10(result), np.nan)
        result = request.final.apply(
            result,
            apply_power=request.operation is not MathOperation.NONE,
        )
        result = RoundingStrategy.for_rounding(request.rounding).apply(result, request)
        return request.bounds.apply(result)


__all__ = public_names_from_objects(
    CalculateMathExecution,
    MathBounds,
    MathCalculationRequest,
    MathFinalTransform,
    MathOperand,
    MathOperandSliceAlignment,
    MathOperationStrategy,
    MathPowerTransform,
    MathResultRows,
    RoundingStrategy,
    calculate_math,
)
