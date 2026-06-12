"""CalculateMath measurement semantics for CellProfiler-compatible processing."""

from __future__ import annotations

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
from openhcs.interop.cellprofiler.calculate_math_settings import (
    CalculateMathRoundingMethod as RoundingMethod,
)
from openhcs.interop.cellprofiler.image_math_settings import (
    ImageMathOperation as MathOperation,
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
