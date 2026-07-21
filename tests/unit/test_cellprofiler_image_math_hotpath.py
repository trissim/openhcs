"""Exactness coverage for the allocation-collapsed ImageMath operation leaf."""

from collections.abc import Callable

import numpy as np
import pytest

from openhcs.core.aligned_image_payload import ImagePayloadBundleContext
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
)
from openhcs.processing.backends.cellprofiler.image_math import (
    ImageMathOperation,
    ImageMathOperationStrategy,
    image_math,
)


def _source_binding_operands() -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(17)
    return tuple(
        rng.uniform(0.05, 0.95, size=(5, 17, 19)).astype(np.float32)
        for _ in range(3)
    )


def _source_binding_payload(operands: tuple[np.ndarray, ...]):
    return ImagePayloadBundleContext.from_payloads(
        tuple(
            ImagePayloadMetadata(source_image_names=(f"operand_{index}",)).payload_with(
                operand
            )
            for index, operand in enumerate(operands)
        )
    ).compose()


def _allocating_reference(
    operation: ImageMathOperation,
    operands: tuple[np.ndarray, ...],
) -> np.ndarray:
    pixel_data = [operand.astype(np.float64) for operand in operands]
    output = pixel_data[0].copy()
    operators: dict[ImageMathOperation, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
        ImageMathOperation.ADD: np.add,
        ImageMathOperation.SUBTRACT: np.subtract,
        ImageMathOperation.MULTIPLY: np.multiply,
        ImageMathOperation.DIVIDE: np.divide,
        ImageMathOperation.MINIMUM: np.minimum,
        ImageMathOperation.MAXIMUM: np.maximum,
    }
    if operation is ImageMathOperation.DIFFERENCE:
        for operand in pixel_data[1:]:
            output = np.abs(np.subtract(output, operand))
    elif operation is ImageMathOperation.AVERAGE:
        for operand in pixel_data[1:]:
            output = np.add(output, operand)
        output = output / len(pixel_data)
    else:
        operator = operators[operation]
        for operand in pixel_data[1:]:
            output = operator(output, operand)
    return output.astype(np.float32)


@pytest.mark.parametrize(
    "operation",
    (
        ImageMathOperation.ADD,
        ImageMathOperation.SUBTRACT,
        ImageMathOperation.DIFFERENCE,
        ImageMathOperation.MULTIPLY,
        ImageMathOperation.DIVIDE,
        ImageMathOperation.AVERAGE,
        ImageMathOperation.MINIMUM,
        ImageMathOperation.MAXIMUM,
    ),
)
def test_image_math_in_place_reduction_is_bit_exact_to_allocating_reference(
    operation: ImageMathOperation,
) -> None:
    operands = _source_binding_operands()

    result = image_math(
        _source_binding_payload(operands),
        operation=operation,
        factors=(1.0, 1.0, 1.0),
        truncate_low=False,
        truncate_high=False,
        replace_nan=False,
    )

    expected = _allocating_reference(operation, operands)
    assert np.array_equal(image_payload_data(result), expected)


@pytest.mark.parametrize(
    "operation",
    (
        ImageMathOperation.ADD,
        ImageMathOperation.SUBTRACT,
        ImageMathOperation.DIFFERENCE,
        ImageMathOperation.MULTIPLY,
        ImageMathOperation.DIVIDE,
        ImageMathOperation.AVERAGE,
        ImageMathOperation.MINIMUM,
        ImageMathOperation.MAXIMUM,
    ),
)
def test_image_math_reduction_reuses_the_detached_first_operand(
    operation: ImageMathOperation,
) -> None:
    operands = [operand.astype(np.float64) for operand in _source_binding_operands()]
    strategy = ImageMathOperationStrategy.coerce(operation)
    output = strategy.prepare_initial_output(
        operands[0],
        operands,
        (1.0, 1.0, 1.0),
    )

    result = strategy.apply(output, operands, (1.0, 1.0, 1.0))

    assert output is operands[0]
    assert result is output
