"""
Converted from CellProfiler: CalculateMath
Original: CalculateMath module

Performs arithmetic operations on measurements produced by previous modules.
This is a measurement-only module that operates on pre-computed measurements,
not on image data directly.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openhcs.core.memory.decorators import numpy
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
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


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("math_results", csv_materializer(
    fields=["slice_index", "output_name", "result_value", "operand1_value", "operand2_value", "operation"],
    analysis_type="math"
)))
def calculate_math(
    image: np.ndarray,
    operand1_value: float = 0.0,
    operand2_value: float = 0.0,
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
    # Pre-process operands
    value1 = operand1_value * operand1_multiplicand
    value1 = np.power(value1, operand1_exponent)
    
    value2 = operand2_value * operand2_multiplicand
    value2 = np.power(value2, operand2_exponent)
    
    # Perform operation
    if operation == MathOperation.NONE:
        result = value1
    elif operation == MathOperation.ADD:
        result = value1 + value2
    elif operation == MathOperation.SUBTRACT:
        result = value1 - value2
    elif operation == MathOperation.MULTIPLY:
        result = value1 * value2
    elif operation == MathOperation.DIVIDE:
        if value2 == 0:
            result = np.nan
        else:
            result = value1 / value2
    else:
        result = value1
    
    # Post-operation transformations
    if take_log10:
        if result > 0:
            result = np.log10(result)
        else:
            result = np.nan
    
    if operation != MathOperation.NONE:
        result = result * final_multiplicand
        result = np.power(result, final_exponent)
    
    result = result + final_addend
    
    # Apply rounding
    if rounding == RoundingMethod.DECIMAL_PLACES:
        result = np.around(result, rounding_digits)
    elif rounding == RoundingMethod.FLOOR:
        result = np.floor(result)
    elif rounding == RoundingMethod.CEILING:
        result = np.ceil(result)
    
    # Apply bounds
    if constrain_lower_bound and not np.isnan(result):
        result = max(result, lower_bound)
    
    if constrain_upper_bound and not np.isnan(result):
        result = min(result, upper_bound)
    
    # Create result dataclass
    math_result = MathResult(
        slice_index=0,
        output_name=output_name,
        result_value=float(result) if not np.isnan(result) else np.nan,
        operand1_value=float(operand1_value),
        operand2_value=float(operand2_value),
        operation=operation.value
    )
    
    return image, math_result


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(("math_array_results", csv_materializer(
    fields=["slice_index", "output_name", "mean_result", "min_result", "max_result", "operation"],
    analysis_type="math_array"
)))
def calculate_math_array(
    image: np.ndarray,
    operand1_values: Optional[np.ndarray] = None,
    operand2_values: Optional[np.ndarray] = None,
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
) -> Tuple[np.ndarray, 'MathArrayResult']:
    """
    Perform arithmetic operations on arrays of measurement values.
    
    This variant handles per-object measurements where operands are arrays
    of values (one per object).
    
    Args:
        image: Input image array (H, W), passed through unchanged
        operand1_values: Array of first operand values (one per object)
        operand2_values: Array of second operand values (one per object)
        operation: Arithmetic operation to perform
        operand1_multiplicand: Multiply first operand by this value
        operand1_exponent: Raise first operand to this power
        operand2_multiplicand: Multiply second operand by this value
        operand2_exponent: Raise second operand to this power
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
        Tuple of (image unchanged, MathArrayResult with calculation summary)
    """
    @dataclass
    class MathArrayResult:
        slice_index: int
        output_name: str
        mean_result: float
        min_result: float
        max_result: float
        operation: str
    
    # Handle None inputs
    if operand1_values is None:
        operand1_values = np.array([0.0])
    if operand2_values is None:
        operand2_values = np.array([0.0])
    
    # Ensure arrays
    values1 = np.atleast_1d(operand1_values).astype(float)
    values2 = np.atleast_1d(operand2_values).astype(float)
    
    # Pre-process operands
    values1 = values1 * operand1_multiplicand
    values1 = np.power(values1, operand1_exponent)
    
    values2 = values2 * operand2_multiplicand
    values2 = np.power(values2, operand2_exponent)
    
    # Handle mismatched array lengths
    if len(values1) != len(values2) and operation != MathOperation.NONE:
        min_len = min(len(values1), len(values2))
        max_len = max(len(values1), len(values2))
        if len(values1) < max_len:
            padded = np.full(max_len, np.nan)
            padded[:len(values1)] = values1
            values1 = padded
        if len(values2) < max_len:
            padded = np.full(max_len, np.nan)
            padded[:len(values2)] = values2
            values2 = padded
    
    # Perform operation
    if operation == MathOperation.NONE:
        result = values1
    elif operation == MathOperation.ADD:
        result = values1 + values2
    elif operation == MathOperation.SUBTRACT:
        result = values1 - values2
    elif operation == MathOperation.MULTIPLY:
        result = values1 * values2
    elif operation == MathOperation.DIVIDE:
        result = values1 / values2
        result[values2 == 0] = np.nan
    else:
        result = values1
    
    # Post-operation transformations
    if take_log10:
        with np.errstate(invalid='ignore', divide='ignore'):
            result = np.log10(result)
    
    if operation != MathOperation.NONE:
        result = result * final_multiplicand
        with np.errstate(invalid='ignore'):
            result = np.power(result, final_exponent)
    
    result = result + final_addend
    
    # Apply rounding
    if rounding == RoundingMethod.DECIMAL_PLACES:
        result = np.around(result, rounding_digits)
    elif rounding == RoundingMethod.FLOOR:
        result = np.floor(result)
    elif rounding == RoundingMethod.CEILING:
        result = np.ceil(result)
    
    # Apply bounds
    if constrain_lower_bound:
        result = np.where(result < lower_bound, lower_bound, result)
    
    if constrain_upper_bound:
        result = np.where(result > upper_bound, upper_bound, result)
    
    # Calculate summary statistics
    valid_results = result[~np.isnan(result)]
    if len(valid_results) > 0:
        mean_val = float(np.mean(valid_results))
        min_val = float(np.min(valid_results))
        max_val = float(np.max(valid_results))
    else:
        mean_val = np.nan
        min_val = np.nan
        max_val = np.nan
    
    math_result = MathArrayResult(
        slice_index=0,
        output_name=output_name,
        mean_result=mean_val,
        min_result=min_val,
        max_result=max_val,
        operation=operation.value
    )
    
    return image, math_result