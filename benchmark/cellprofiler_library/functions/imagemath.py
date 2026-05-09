"""
Converted from CellProfiler: ImageMath
Original: ImageMath module

Performs simple mathematical operations on image intensities.
Supports addition, subtraction, multiplication, division, averaging,
min/max, standard deviation, inversion, log transform, and logical operations.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Tuple
from enum import Enum
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory.decorators import numpy
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)

ImageMathBinaryOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]


class MathOperation(Enum):
    ADD = "add"
    SUBTRACT = "subtract"
    DIFFERENCE = "absolute_difference"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    STDEV = "standard_deviation"
    INVERT = "invert"
    COMPLEMENT = "complement"
    LOG_TRANSFORM = "log_transform_base2"
    LOG_TRANSFORM_LEGACY = "log_transform_legacy"
    NONE = "none"
    OR = "or"
    AND = "and"
    NOT = "not"
    EQUALS = "equals"


class ImageMathOperationStrategy(
    EnumKeyedStrategyMixin[MathOperation],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal owner for ImageMath operation arity, literals, and execution."""

    __registry_key__ = "operation_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "operation"
    __enum_label_attr__ = "operation_label"

    operation: ClassVar[MathOperation | None] = None
    operation_label: ClassVar[str | None] = None
    cellprofiler_literals: ClassVar[tuple[str, ...]] = ()
    single_image: ClassVar[bool] = False
    binary_output: ClassVar[bool] = False
    implementation_namespace_key: ClassVar[str | None] = None

    @staticmethod
    def normalized_cellprofiler_literal(value: str) -> str:
        return "_".join(value.strip().lower().replace("-", " ").split())

    @staticmethod
    def operands_are_logical(pixel_data: list[np.ndarray]) -> bool:
        return all(pd.dtype == bool for pd in pixel_data if not np.isscalar(pd))

    @classmethod
    def materialized_namespace(
        cls,
        operation: MathOperation,
        implementation: ImageMathBinaryOperator | None,
    ) -> dict[str, Any]:
        namespace: dict[str, Any] = {
            "__module__": __name__,
            "operation": operation,
        }
        if cls.implementation_namespace_key is not None:
            namespace[cls.implementation_namespace_key] = implementation
        return namespace

    @classmethod
    def coerce(cls, value: MathOperation | str) -> "ImageMathOperationStrategy":
        if isinstance(value, MathOperation):
            return cls.for_enum_member(value)
        literal = cls.normalized_cellprofiler_literal(str(value))
        for strategy_type in cls.registered_strategy_types():
            if literal in strategy_type.normalized_cellprofiler_literals():
                return strategy_type()
        raise ValueError(f"Unsupported ImageMath operation {value!r}.")

    @classmethod
    def normalized_cellprofiler_literals(cls) -> frozenset[str]:
        operation = cls.operation
        if not isinstance(operation, MathOperation):
            raise TypeError(f"{cls.__name__} must declare a MathOperation.")
        literals = (operation.name, operation.value, *cls.cellprofiler_literals)
        return frozenset(cls.normalized_cellprofiler_literal(literal) for literal in literals)

    def prepare_initial_output(
        self,
        image: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        if self.single_image:
            output = image.astype(np.float64, copy=True)
            if not self.binary_output and factors[0] != 1.0:
                output *= factors[0]
            return output
        return pixel_data[0].copy()

    @abstractmethod
    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        """Apply this ImageMath operation to prepared operands."""


class PairwiseNumpyImageMathOperationStrategy(ImageMathOperationStrategy):
    """Template for ImageMath operations that reduce operands with one NumPy op."""

    implementation_namespace_key = "numpy_operator"
    numpy_operator: ClassVar[ImageMathBinaryOperator | None] = None

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        if self.numpy_operator is None:
            raise TypeError(f"{type(self).__name__} must declare a NumPy operator.")
        for pd in pixel_data[1:]:
            output_pixel_data = self.numpy_operator(output_pixel_data, pd)
        return output_pixel_data


class SubtractImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.SUBTRACT

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        if self.operands_are_logical(pixel_data):
            output_pixel_data = pixel_data[0].copy()
            for pd in pixel_data[1:]:
                output_pixel_data[pd.astype(bool)] = False
            return output_pixel_data
        for pd in pixel_data[1:]:
            output_pixel_data = np.subtract(output_pixel_data, pd)
        return output_pixel_data


class DifferenceImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.DIFFERENCE
    cellprofiler_literals = ("difference",)

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        if self.operands_are_logical(pixel_data):
            for pd in pixel_data[1:]:
                output_pixel_data = np.logical_xor(output_pixel_data, pd)
            return output_pixel_data
        for pd in pixel_data[1:]:
            output_pixel_data = np.abs(np.subtract(output_pixel_data, pd))
        return output_pixel_data


class MultiplyImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.MULTIPLY

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        if self.operands_are_logical(pixel_data):
            for pd in pixel_data[1:]:
                output_pixel_data = np.logical_and(output_pixel_data, pd)
            return output_pixel_data
        for pd in pixel_data[1:]:
            output_pixel_data = np.multiply(output_pixel_data, pd)
        return output_pixel_data


class AverageImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.AVERAGE

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        for pd in pixel_data[1:]:
            output_pixel_data = np.add(output_pixel_data, pd)
        if not self.operands_are_logical(pixel_data):
            output_pixel_data = output_pixel_data / sum(factors[: len(pixel_data)])
        return output_pixel_data


class StandardDeviationImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.STDEV
    cellprofiler_literals = ("stdev",)

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del output_pixel_data, factors
        return np.std(np.array(pixel_data), axis=0)


class InvertingImageMathOperationStrategy(ImageMathOperationStrategy):
    """Template for CP ImageMath operations backed by skimage inversion."""

    single_image = True

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del pixel_data, factors
        import skimage.util

        return skimage.util.invert(output_pixel_data)


class LogTransformImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.LOG_TRANSFORM
    cellprofiler_literals = ("log_transform", "log_transform_base_2")
    single_image = True

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del pixel_data, factors
        return np.log2(output_pixel_data + 1)


class LegacyLogTransformImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.LOG_TRANSFORM_LEGACY
    single_image = True

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del pixel_data, factors
        return np.log2(output_pixel_data)


class NoOpImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.NONE
    single_image = True

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del pixel_data, factors
        return output_pixel_data


class LogicalReductionImageMathOperationStrategy(ImageMathOperationStrategy):
    """Template for binary ImageMath operations that reduce logical operands."""

    binary_output = True
    implementation_namespace_key = "logical_operator"
    logical_operator: ClassVar[ImageMathBinaryOperator | None] = None

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        if self.logical_operator is None:
            raise TypeError(f"{type(self).__name__} must declare a logical operator.")
        for pd in pixel_data[1:]:
            output_pixel_data = self.logical_operator(output_pixel_data, pd)
        return output_pixel_data.astype(np.float64)


class NotImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.NOT
    single_image = True
    binary_output = True

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del pixel_data, factors
        return np.logical_not(output_pixel_data).astype(np.float64)


class EqualsImageMathOperationStrategy(ImageMathOperationStrategy):
    operation = MathOperation.EQUALS
    binary_output = True

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del output_pixel_data, factors
        result = np.ones(pixel_data[0].shape, dtype=bool)
        comparator = pixel_data[0]
        for pd in pixel_data[1:]:
            result = result & (comparator == pd)
        return result.astype(np.float64)


@dataclass(frozen=True, slots=True)
class ImageMathOperationClassDeclaration:
    """Typed declaration for metadata-only ImageMath strategy leaves."""

    class_name: str
    base_type: type[ImageMathOperationStrategy]
    operation: MathOperation
    implementation: ImageMathBinaryOperator | None = None

    def materialize_into(
        self,
        namespace: dict[str, Any],
    ) -> type[ImageMathOperationStrategy]:
        class_namespace = self.base_type.materialized_namespace(
            self.operation,
            self.implementation,
        )
        strategy_type = type(self.class_name, (self.base_type,), class_namespace)
        namespace[self.class_name] = strategy_type
        return strategy_type


IMAGE_MATH_OPERATION_CLASS_DECLARATIONS: tuple[
    ImageMathOperationClassDeclaration,
    ...,
] = (
    ImageMathOperationClassDeclaration(
        "AddImageMathOperationStrategy",
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.ADD,
        np.add,
    ),
    ImageMathOperationClassDeclaration(
        "DivideImageMathOperationStrategy",
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.DIVIDE,
        np.divide,
    ),
    ImageMathOperationClassDeclaration(
        "MinimumImageMathOperationStrategy",
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.MINIMUM,
        np.minimum,
    ),
    ImageMathOperationClassDeclaration(
        "MaximumImageMathOperationStrategy",
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.MAXIMUM,
        np.maximum,
    ),
    ImageMathOperationClassDeclaration(
        "InvertImageMathOperationStrategy",
        InvertingImageMathOperationStrategy,
        MathOperation.INVERT,
    ),
    ImageMathOperationClassDeclaration(
        "ComplementImageMathOperationStrategy",
        InvertingImageMathOperationStrategy,
        MathOperation.COMPLEMENT,
    ),
    ImageMathOperationClassDeclaration(
        "OrImageMathOperationStrategy",
        LogicalReductionImageMathOperationStrategy,
        MathOperation.OR,
        np.logical_or,
    ),
    ImageMathOperationClassDeclaration(
        "AndImageMathOperationStrategy",
        LogicalReductionImageMathOperationStrategy,
        MathOperation.AND,
        np.logical_and,
    ),
)


for image_math_operation_class_declaration in IMAGE_MATH_OPERATION_CLASS_DECLARATIONS:
    image_math_operation_class_declaration.materialize_into(globals())


def _apply_image_mask(pixel_data: np.ndarray, mask: Any | None) -> np.ndarray:
    """Apply an image-validity mask using CellProfiler's output pixel semantics."""
    if mask is None:
        return pixel_data

    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape == pixel_data.shape:
        return pixel_data * mask_array

    if pixel_data.ndim == 2:
        if mask_array.ndim == 3 and mask_array.shape[0] == 1:
            mask_array = mask_array[0]
        if mask_array.shape == pixel_data.shape:
            return pixel_data * mask_array

    if pixel_data.ndim == 3:
        if mask_array.ndim == 2 and mask_array.shape == pixel_data.shape[:2]:
            return pixel_data * mask_array[:, :, np.newaxis]
        if mask_array.ndim == 2 and mask_array.shape == pixel_data.shape[1:]:
            return pixel_data * mask_array[np.newaxis, :, :]
        if mask_array.ndim == 3 and mask_array.shape[:2] == pixel_data.shape[:2] and pixel_data.shape[2] in (3, 4):
            return pixel_data * mask_array[:, :, :1]

    if pixel_data.ndim == 4:
        if mask_array.ndim == 2 and mask_array.shape == pixel_data.shape[1:3]:
            return pixel_data * mask_array[np.newaxis, :, :, np.newaxis]
        if mask_array.ndim == 3 and mask_array.shape == pixel_data.shape[:3]:
            return pixel_data * mask_array[:, :, :, np.newaxis]

    return pixel_data * mask_array


def _image_math_operand_masks(source_payload: Any, operand_count: int) -> list[Any | None]:
    """Return one CellProfiler mask domain per ImageMath operand."""
    source_mask = image_payload_mask(source_payload)
    if source_mask is None:
        return [None] * operand_count
    mask_array = np.asarray(source_mask, dtype=bool)
    if mask_array.ndim >= 3 and mask_array.shape[0] >= operand_count:
        return [mask_array[index] for index in range(operand_count)]
    return [mask_array] + [None] * max(operand_count - 1, 0)


def _image_math_output_mask(
    operand_masks: list[Any | None],
    *,
    ignore_masks: bool,
) -> Any | None:
    """Combine operand masks the same way CellProfiler ImageMath does."""
    if ignore_masks:
        return None
    output_mask = operand_masks[0] if operand_masks else None
    for mask in operand_masks[1:]:
        if output_mask is None:
            output_mask = mask
        elif mask is not None:
            output_mask = np.asarray(output_mask, dtype=bool) & np.asarray(mask, dtype=bool)
    return output_mask


@numpy
def image_math(
    image: np.ndarray,
    operation: MathOperation = MathOperation.ADD,
    factors: Tuple[float, ...] = (1.0, 1.0),
    exponent: float = 1.0,
    after_factor: float = 1.0,
    addend: float = 0.0,
    truncate_low: bool = True,
    truncate_high: bool = True,
    replace_nan: bool = True,
    ignore_masks: bool = False,
) -> np.ndarray:
    """
    Perform mathematical operations on image intensities.
    
    Args:
        image: Input array of shape (N, H, W) where N images are stacked along dim 0.
               For single-image operations (INVERT, LOG_TRANSFORM, NOT, NONE),
               only the first slice is used.
               For multi-image operations, all N slices are combined.
        operation: The mathematical operation to perform.
        factors: Tuple of multiplication factors for each input image (applied before operation).
        exponent: Raise the result to this power (after operation).
        after_factor: Multiply the result by this value (after operation).
        addend: Add this value to the result (after operation).
        truncate_low: Set values less than 0 to 0.
        truncate_high: Set values greater than 1 to 1.
        replace_nan: Replace NaN values with 0.
        ignore_masks: Drop any input image mask instead of preserving it.
    
    Returns:
        Processed image of shape (1, H, W).
    """
    operation_strategy = ImageMathOperationStrategy.coerce(operation)
    source_payload = image
    image = image_payload_data(image)
    
    # Handle input dimensions
    if image.ndim == 2:
        image = image[np.newaxis, :, :]
    
    n_images = image.shape[0]
    
    # Extend factors if needed
    if len(factors) < n_images:
        factors = tuple(factors) + (1.0,) * (n_images - len(factors))
    
    # Apply factors to each image (except for binary output operations)
    pixel_data = []
    operand_count = 1 if operation_strategy.single_image else n_images
    operand_masks = _image_math_operand_masks(source_payload, operand_count)
    for i in range(operand_count):
        pd = image[i].astype(np.float64)
        if not operation_strategy.binary_output and factors[i] != 1.0:
            pd = pd * factors[i]
        pixel_data.append(pd)
    output_pixel_data = operation_strategy.apply(
        operation_strategy.prepare_initial_output(image, pixel_data, factors),
        pixel_data,
        factors,
    )
    
    # Post-processing (not for binary output operations)
    if not operation_strategy.binary_output:
        if exponent != 1.0:
            output_pixel_data = output_pixel_data ** exponent
        if after_factor != 1.0:
            output_pixel_data = output_pixel_data * after_factor
        if addend != 0.0:
            output_pixel_data = output_pixel_data + addend
        
        # Truncation
        if truncate_low:
            output_pixel_data[output_pixel_data < 0] = 0
        if truncate_high:
            output_pixel_data[output_pixel_data > 1] = 1
        if replace_nan:
            output_pixel_data[np.isnan(output_pixel_data)] = 0
    
    # Ensure output is (1, H, W)
    if output_pixel_data.ndim == 2:
        output_pixel_data = output_pixel_data[np.newaxis, :, :]

    output_mask = _image_math_output_mask(
        operand_masks,
        ignore_masks=ignore_masks,
    )
    if output_mask is not None:
        output_pixel_data = _apply_image_mask(output_pixel_data, output_mask)

    output = output_pixel_data.astype(np.float32)
    if ignore_masks:
        return output
    return image_payload_with_context(
        output,
        mask=output_mask,
        metadata=image_payload_metadata(source_payload).without_unit_interval_intensity_scale(),
    )
