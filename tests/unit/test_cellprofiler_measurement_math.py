import numpy as np
import pytest

from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathRoundingMethod,
)
from openhcs.processing.backends.cellprofiler.measurement_math import (
    CalculateMathExecution,
    MathBounds,
    MathCalculationRequest,
    MathFinalTransform,
    MathOperand,
    calculate_math,
)
from openhcs.processing.backends.cellprofiler.image_math import ImageMathOperation


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
    row_mappings = tuple(rows.iter_row_mappings())

    assert [row[MeasurementRowAxisField.SLICE_INDEX.value] for row in row_mappings] == [
        0,
        0,
        1,
        1,
    ]
    assert [
        row[MeasurementRowAxisField.OBJECT_LABEL.value] for row in row_mappings
    ] == [1, 2, 1, 2]
    np.testing.assert_allclose(
        [
            row[MeasurementRowValueField.RESULT_VALUE.value]
            for row in row_mappings
        ],
        [5.0, 5.0, 10.0, 5.0],
    )


def test_calculate_math_rejects_inexact_aligned_operand_cardinality() -> None:
    request = MathCalculationRequest(
        operand1=MathOperand(
            value=RuntimeSliceAlignedValues((1.0, 2.0)),
            multiplicand=1.0,
            exponent=1.0,
        ),
        operand2=MathOperand(
            value=RuntimeSliceAlignedValues((1.0, 2.0, 3.0, 4.0)),
            multiplicand=1.0,
            exponent=1.0,
        ),
        operation=ImageMathOperation.ADD,
        take_log10=False,
        final=MathFinalTransform(multiplicand=1.0, exponent=1.0, addend=0.0),
        rounding=CalculateMathRoundingMethod.NOT_ROUNDED,
        rounding_digits=0,
        bounds=MathBounds(
            constrain_lower=False,
            lower=0.0,
            constrain_upper=False,
            upper=1.0,
        ),
        output_name="Result",
        object_names=(),
    )

    with pytest.raises(ValueError, match="cardinalities must match exactly"):
        CalculateMathExecution(request).result_rows


def test_calculate_math_request_replaces_only_operand_values() -> None:
    request = MathCalculationRequest(
        operand1=MathOperand(value=2.0, multiplicand=3.0, exponent=4.0),
        operand2=MathOperand(value=5.0, multiplicand=6.0, exponent=7.0),
        operation=ImageMathOperation.ADD,
        take_log10=False,
        final=MathFinalTransform(multiplicand=8.0, exponent=9.0, addend=10.0),
        rounding=CalculateMathRoundingMethod.FLOOR,
        rounding_digits=2,
        bounds=MathBounds(
            constrain_lower=True,
            lower=11.0,
            constrain_upper=True,
            upper=12.0,
        ),
        output_name="Result",
        object_names=("Nuclei",),
    )

    replaced = request.for_operand_values(operand1_value=13.0, operand2_value=14.0)

    assert replaced.operand1 == MathOperand(
        value=13.0,
        multiplicand=3.0,
        exponent=4.0,
    )
    assert replaced.operand2 == MathOperand(
        value=14.0,
        multiplicand=6.0,
        exponent=7.0,
    )
    assert replaced.final is request.final
    assert replaced.bounds is request.bounds
    assert replaced.operation is request.operation
    assert replaced.output_name == request.output_name
    assert replaced.object_names == request.object_names


def test_calculate_math_scalar_result_does_not_claim_object_row_ownership() -> None:
    request = MathCalculationRequest(
        operand1=MathOperand(value=8.0, multiplicand=1.0, exponent=1.0),
        operand2=MathOperand(value=2.0, multiplicand=1.0, exponent=1.0),
        operation=ImageMathOperation.DIVIDE,
        take_log10=False,
        final=MathFinalTransform(multiplicand=1.0, exponent=1.0, addend=0.0),
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

    row = CalculateMathExecution(request).result_rows.row_mappings()[0]

    assert row[MeasurementRowAxisField.OBJECT_LABEL.value] is None
    assert row[MeasurementRowAxisField.OBJECT_NAME.value] is None


def test_calculate_math_typed_transforms_preserve_cellprofiler_order() -> None:
    image = np.zeros((2, 2), dtype=np.float32)

    _image, identity_rows = calculate_math(
        image,
        operand1_value=2.0,
        operation=ImageMathOperation.NONE,
        final_multiplicand=10.0,
        final_exponent=2.0,
        final_addend=3.0,
    )
    _image, transformed_rows = calculate_math(
        image,
        operand1_value=4.0,
        operand2_value=2.0,
        operation=ImageMathOperation.DIVIDE,
        final_multiplicand=3.0,
        final_exponent=2.0,
        final_addend=1.0,
        constrain_upper_bound=True,
        upper_bound=36.0,
    )

    np.testing.assert_allclose(
        identity_rows.column_values(MeasurementRowValueField.RESULT_VALUE.value),
        (5.0,),
    )
    np.testing.assert_allclose(
        transformed_rows.column_values(MeasurementRowValueField.RESULT_VALUE.value),
        (36.0,),
    )
