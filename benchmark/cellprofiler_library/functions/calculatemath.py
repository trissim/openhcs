"""
Converted from CellProfiler: CalculateMath
Original: CalculateMath module

Performs arithmetic operations on measurements produced by previous modules.
This is a measurement-only module that operates on pre-computed measurements,
not on image data directly.
"""

from typing import Any, Optional, Tuple

import numpy as np

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_outputs
from openhcs.interop.cellprofiler.calculate_math_settings import (
    CalculateMathRoundingMethod as RoundingMethod,
)
from openhcs.interop.cellprofiler.image_math_settings import (
    ImageMathOperation as MathOperation,
)
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathExecution,
    MathBounds,
    MathCalculationRequest,
    MathFinalTransform,
    MathOperand,
    MathOperationStrategy,
    MathPowerTransform,
    MathResult,
    RoundingStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
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
    math_result = CalculateMathExecution(request).result_rows
    
    return image, math_result
