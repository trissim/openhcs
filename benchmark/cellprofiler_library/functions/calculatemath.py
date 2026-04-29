"""
Converted from CellProfiler: CalculateMath
Original: CalculateMath module

Performs arithmetic operations on measurements produced by previous modules.
This is a measurement-only module that operates on pre-computed measurements,
not on image data directly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional, Tuple

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


class MathOperation(Enum):
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    ADD = "add"
    SUBTRACT = "subtract"
    NONE = "none"


class RoundingMethod(Enum):
    NOT_ROUNDED = "not_rounded"
    DECIMAL_PLACES = "decimal_places"
    FLOOR = "floor"
    CEILING = "ceiling"


@dataclass
class MathResult:
    """Result of mathematical calculation on measurements."""
    slice_index: int
    output_name: str
    feature_name: str
    result_value: float
    operand1_value: float
    operand2_value: float
    operation: str
    object_label: Optional[int] = None
    object_name: Optional[str] = None


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
    object_names: Tuple[str, ...]


class MathOperationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy for the closed CalculateMath operation family."""

    __registry_key__ = "operation"
    __skip_if_no_key__ = True
    operation: ClassVar[MathOperation]

    @classmethod
    def for_operation(cls, operation: MathOperation) -> "MathOperationStrategy":
        return cls.__registry__[operation]()

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


class RoundingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy for the closed CalculateMath rounding family."""

    __registry_key__ = "rounding"
    __skip_if_no_key__ = True
    rounding: ClassVar[RoundingMethod]

    @classmethod
    def for_rounding(cls, rounding: RoundingMethod) -> "RoundingStrategy":
        return cls.__registry__[rounding]()

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


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("math_results", csv_materializer(
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
    analysis_type="math"
)))
def calculate_math(
    image: np.ndarray,
    operand1_value: Any = 0.0,
    operand2_value: Any = 0.0,
    operand1_feature: Optional[str] = None,
    operand2_feature: Optional[str] = None,
    operand1_object_name: Optional[str] = None,
    operand2_object_name: Optional[str] = None,
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
) -> Tuple[np.ndarray, MathResult | list[MathResult]]:
    """
    Perform arithmetic operations on measurement values.
    
    This module takes measurement values (typically from previous analysis steps)
    and performs basic arithmetic operations including addition, subtraction,
    multiplication, and division. Results can be log-transformed, raised to a
    power, and constrained to bounds.
    
    Note: This is primarily a measurement calculation module. The image is
    passed through unchanged while the calculation is performed on the
    provided operand values.
    
    Args:
        image: Input image array (H, W), passed through unchanged
        operand1_value: First operand measurement value
        operand2_value: Second operand measurement value (used for binary operations)
        operand1_feature: CellProfiler feature selected for the first runtime operand
        operand2_feature: CellProfiler feature selected for the second runtime operand
        operand1_object_name: Optional object set selected for the first operand
        operand2_object_name: Optional object set selected for the second operand
        operation: Arithmetic operation to perform
        operand1_multiplicand: Multiply first operand by this value before operation
        operand1_exponent: Raise first operand to this power before operation
        operand2_multiplicand: Multiply second operand by this value before operation
        operand2_exponent: Raise second operand to this power before operation
        take_log10: Whether to take log10 of the result
        final_multiplicand: Multiply result by this value
        final_exponent: Raise result to this power
        final_addend: Add this value to the result
        rounding: How to round the output value
        rounding_digits: Number of decimal places for rounding
        constrain_lower_bound: Whether to constrain result to lower bound
        lower_bound: Lower bound value
        constrain_upper_bound: Whether to constrain result to upper bound
        upper_bound: Upper bound value
        output_name: Name for the output measurement
    
    Returns:
        Tuple of (image unchanged, MathResult rows with calculation details)
    """
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
    result = _calculate_scalar_result(request)
    math_result = _math_result_rows(result, request)
    
    return image, math_result


def _calculate_scalar_result(request: MathCalculationRequest) -> Any:
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


def _math_result_rows(
    result: Any,
    request: MathCalculationRequest,
) -> MathResult | list[MathResult]:
    result_values = np.asarray(result, dtype=float)
    feature_name = f"Math_{request.output_name}"
    if result_values.ndim == 0:
        return MathResult(
            slice_index=0,
            output_name=request.output_name,
            feature_name=feature_name,
            result_value=_float_value(result_values.item()),
            operand1_value=_scalar_operand_value(request.operand1.value),
            operand2_value=_scalar_operand_value(request.operand2.value),
            operation=request.operation.value,
            object_name=next(iter(request.object_names), None),
        )

    flat_results = result_values.reshape(-1)
    object_names = request.object_names or (None,)
    operand1_values = _broadcast_operand_values(
        request.operand1.value,
        len(flat_results),
    )
    operand2_values = _broadcast_operand_values(
        request.operand2.value,
        len(flat_results),
    )
    return [
        MathResult(
            slice_index=0,
            object_name=object_name,
            object_label=index + 1,
            output_name=request.output_name,
            feature_name=feature_name,
            result_value=_float_value(result_value),
            operand1_value=_float_value(operand1_values[index]),
            operand2_value=_float_value(operand2_values[index]),
            operation=request.operation.value,
        )
        for object_name in object_names
        for index, result_value in enumerate(flat_results)
    ]


def _broadcast_operand_values(value: Any, count: int) -> np.ndarray:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size == count:
        return values
    if values.size == 1:
        return np.full(count, _float_value(values[0]))
    raise ValueError(
        f"CalculateMath operand produced {values.size} values for {count} results."
    )


def _scalar_operand_value(value: Any) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 1:
        return np.nan
    return _float_value(values[0])


def _float_value(value: Any) -> float:
    scalar = float(value)
    return scalar if not np.isnan(scalar) else np.nan
