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
from typing import ClassVar, Optional, Tuple

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
    result_value: float
    operand1_value: float
    operand2_value: float
    operation: str


@dataclass(frozen=True)
class MathPowerTransform(ABC):
    """Shared multiplicative/exponential transform."""

    multiplicand: float
    exponent: float


@dataclass(frozen=True)
class MathOperand(MathPowerTransform):
    """One CellProfiler CalculateMath operand and its pre-transform."""

    value: float

    @property
    def transformed(self) -> float:
        return float(
            np.power(
                self.value * self.multiplicand,
                self.exponent,
            )
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
    """Typed request for scalar CellProfiler CalculateMath execution."""

    operand1: MathOperand
    operand2: MathOperand
    operation: MathOperation
    take_log10: bool
    final: MathFinalTransform
    rounding: RoundingMethod
    rounding_digits: int
    bounds: MathBounds
    output_name: str


class MathOperationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy for the closed CalculateMath operation family."""

    __registry_key__ = "operation"
    __skip_if_no_key__ = True
    operation: ClassVar[MathOperation]

    @classmethod
    def for_operation(cls, operation: MathOperation) -> "MathOperationStrategy":
        return cls.__registry__[operation]()

    @abstractmethod
    def apply(self, request: MathCalculationRequest) -> float:
        """Return the raw operation result before post-processing."""


class NoneOperationStrategy(MathOperationStrategy):
    operation = MathOperation.NONE

    def apply(self, request: MathCalculationRequest) -> float:
        return request.operand1.transformed


class AddOperationStrategy(MathOperationStrategy):
    operation = MathOperation.ADD

    def apply(self, request: MathCalculationRequest) -> float:
        return request.operand1.transformed + request.operand2.transformed


class SubtractOperationStrategy(MathOperationStrategy):
    operation = MathOperation.SUBTRACT

    def apply(self, request: MathCalculationRequest) -> float:
        return request.operand1.transformed - request.operand2.transformed


class MultiplyOperationStrategy(MathOperationStrategy):
    operation = MathOperation.MULTIPLY

    def apply(self, request: MathCalculationRequest) -> float:
        return request.operand1.transformed * request.operand2.transformed


class DivideOperationStrategy(MathOperationStrategy):
    operation = MathOperation.DIVIDE

    def apply(self, request: MathCalculationRequest) -> float:
        if request.operand2.transformed == 0:
            return np.nan
        return request.operand1.transformed / request.operand2.transformed


class RoundingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal strategy for the closed CalculateMath rounding family."""

    __registry_key__ = "rounding"
    __skip_if_no_key__ = True
    rounding: ClassVar[RoundingMethod]

    @classmethod
    def for_rounding(cls, rounding: RoundingMethod) -> "RoundingStrategy":
        return cls.__registry__[rounding]()

    @abstractmethod
    def apply(self, value: float, request: MathCalculationRequest) -> float:
        """Return rounded value."""


class NotRoundedStrategy(RoundingStrategy):
    rounding = RoundingMethod.NOT_ROUNDED

    def apply(self, value: float, request: MathCalculationRequest) -> float:
        del request
        return value


class DecimalPlacesRoundingStrategy(RoundingStrategy):
    rounding = RoundingMethod.DECIMAL_PLACES

    def apply(self, value: float, request: MathCalculationRequest) -> float:
        return float(np.around(value, request.rounding_digits))


class FloorRoundingStrategy(RoundingStrategy):
    rounding = RoundingMethod.FLOOR

    def apply(self, value: float, request: MathCalculationRequest) -> float:
        del request
        return float(np.floor(value))


class CeilingRoundingStrategy(RoundingStrategy):
    rounding = RoundingMethod.CEILING

    def apply(self, value: float, request: MathCalculationRequest) -> float:
        del request
        return float(np.ceil(value))


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("math_results", csv_materializer(
    fields=["slice_index", "output_name", "result_value", "operand1_value", "operand2_value", "operation"],
    analysis_type="math"
)))
def calculate_math(
    image: np.ndarray,
    operand1_value: float = 0.0,
    operand2_value: float = 0.0,
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
) -> Tuple[np.ndarray, MathResult]:
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
        Tuple of (image unchanged, MathResult with calculation details)
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
    )
    result = _calculate_scalar_result(request)
    math_result = MathResult(
        slice_index=0,
        output_name=output_name,
        result_value=float(result) if not np.isnan(result) else np.nan,
        operand1_value=float(operand1_value),
        operand2_value=float(operand2_value),
        operation=operation.value
    )
    
    return image, math_result


def _calculate_scalar_result(request: MathCalculationRequest) -> float:
    result = MathOperationStrategy.for_operation(request.operation).apply(request)

    if request.take_log10:
        result = np.log10(result) if result > 0 else np.nan

    if request.operation is not MathOperation.NONE:
        result *= request.final.multiplicand
        result = float(np.power(result, request.final.exponent))

    result += request.final.addend
    result = RoundingStrategy.for_rounding(request.rounding).apply(result, request)

    if request.bounds.constrain_lower and not np.isnan(result):
        result = max(result, request.bounds.lower)
    if request.bounds.constrain_upper and not np.isnan(result):
        result = min(result, request.bounds.upper)
    return result
