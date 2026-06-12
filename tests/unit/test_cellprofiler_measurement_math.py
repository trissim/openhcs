import numpy as np

from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.interop.cellprofiler.calculate_math_settings import (
    CalculateMathRoundingMethod,
)
from openhcs.interop.cellprofiler.image_math_settings import ImageMathOperation
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathExecution,
    MathBounds,
    MathCalculationRequest,
    MathFinalTransform,
    MathOperand,
)


def test_calculate_math_aligned_vector_rows_preserve_slice_index() -> None:
    request = MathCalculationRequest(
        operand1=MathOperand(
            value=RuntimeSliceAlignedValues(
                (
                    np.asarray([10.0, 20.0]),
                    np.asarray([30.0, 40.0]),
                )
            ),
            multiplicand=1.0,
            exponent=1.0,
        ),
        operand2=MathOperand(
            value=RuntimeSliceAlignedValues(
                (
                    np.asarray([2.0, 4.0]),
                    np.asarray([3.0, 8.0]),
                )
            ),
            multiplicand=1.0,
            exponent=1.0,
        ),
        operation=ImageMathOperation.DIVIDE,
        take_log10=False,
        final=MathFinalTransform(
            multiplicand=1.0,
            exponent=1.0,
            addend=0.0,
        ),
        rounding=CalculateMathRoundingMethod.NOT_ROUNDED,
        rounding_digits=0,
        bounds=MathBounds(
            constrain_lower=False,
            lower=0.0,
            constrain_upper=False,
            upper=1.0,
        ),
        output_name="Ratio",
        object_names=("Nuclei",),
    )

    rows = CalculateMathExecution(request).result_rows

    assert [row.slice_index for row in rows] == [0, 0, 1, 1]
    assert [row.object_label for row in rows] == [1, 2, 1, 2]
    np.testing.assert_allclose(
        [row.result_value for row in rows],
        [5.0, 5.0, 10.0, 5.0],
    )
