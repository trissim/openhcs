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
from metaclass_registry import AutoRegisterMeta
from openhcs.core.memory.decorators import numpy
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_values import (
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_with_context,
)
from openhcs.interop.cellprofiler.image_math_settings import (
    ImageMathOperation as MathOperation,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

ImageMathBinaryOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]


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
    single_image: ClassVar[bool] = False
    binary_output: ClassVar[bool] = False

    @staticmethod
    def operands_are_logical(pixel_data: list[np.ndarray]) -> bool:
        return all(pd.dtype == bool for pd in pixel_data if not np.isscalar(pd))

    @classmethod
    def coerce(cls, value: MathOperation | str) -> "ImageMathOperationStrategy":
        return cls.for_enum_member(coerce_cellprofiler_enum(MathOperation, value))

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


@dataclass(frozen=True)
class ImageMathOperationStrategyDeclaration:
    """Typed declaration for metadata-only ImageMath strategy leaves."""

    strategy_base: type[ImageMathOperationStrategy]
    operation: MathOperation
    numpy_operator: ImageMathBinaryOperator | None = None
    logical_operator: ImageMathBinaryOperator | None = None

    @property
    def class_name(self) -> str:
        operation_name = "".join(part.title() for part in self.operation.name.split("_"))
        return f"{operation_name}ImageMathOperationStrategy"

    def materialize(self) -> type[ImageMathOperationStrategy]:
        namespace: dict[str, object] = {
            "__module__": __name__,
            "operation": self.operation,
        }
        if self.numpy_operator is not None:
            namespace["numpy_operator"] = self.numpy_operator
        if self.logical_operator is not None:
            namespace["logical_operator"] = self.logical_operator
        return type(self.class_name, (self.strategy_base,), namespace)


IMAGE_MATH_OPERATION_STRATEGY_DECLARATIONS = (
    ImageMathOperationStrategyDeclaration(
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.ADD,
        numpy_operator=np.add,
    ),
    ImageMathOperationStrategyDeclaration(
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.DIVIDE,
        numpy_operator=np.divide,
    ),
    ImageMathOperationStrategyDeclaration(
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.MINIMUM,
        numpy_operator=np.minimum,
    ),
    ImageMathOperationStrategyDeclaration(
        PairwiseNumpyImageMathOperationStrategy,
        MathOperation.MAXIMUM,
        numpy_operator=np.maximum,
    ),
    ImageMathOperationStrategyDeclaration(
        InvertingImageMathOperationStrategy,
        MathOperation.INVERT,
    ),
    ImageMathOperationStrategyDeclaration(
        InvertingImageMathOperationStrategy,
        MathOperation.COMPLEMENT,
    ),
    ImageMathOperationStrategyDeclaration(
        LogicalReductionImageMathOperationStrategy,
        MathOperation.OR,
        logical_operator=np.logical_or,
    ),
    ImageMathOperationStrategyDeclaration(
        LogicalReductionImageMathOperationStrategy,
        MathOperation.AND,
        logical_operator=np.logical_and,
    ),
)

for image_math_operation_strategy_declaration in IMAGE_MATH_OPERATION_STRATEGY_DECLARATIONS:
    globals()[image_math_operation_strategy_declaration.class_name] = (
        image_math_operation_strategy_declaration.materialize()
    )


@dataclass(frozen=True)
class ImageMathMaskPolicy:
    """CellProfiler ImageMath mask projection and output-composition policy."""

    ignore_masks: bool

    def operand_masks(self, source_payload: Any, operand_count: int) -> tuple[Any | None, ...]:
        source_mask = image_payload_mask(source_payload)
        if source_mask is None:
            return (None,) * operand_count
        mask_array = np.asarray(source_mask, dtype=bool)
        if mask_array.ndim >= 3 and mask_array.shape[0] >= operand_count:
            return tuple(mask_array[index] for index in range(operand_count))
        return (mask_array,) + (None,) * max(operand_count - 1, 0)

    def output_mask(self, operand_masks: tuple[Any | None, ...]) -> Any | None:
        if self.ignore_masks:
            return None
        output_mask = operand_masks[0] if operand_masks else None
        for mask in operand_masks[1:]:
            output_mask = self.combine_output_masks(output_mask, mask)
        return output_mask

    @staticmethod
    def combine_output_masks(current_mask: Any | None, next_mask: Any | None) -> Any | None:
        if current_mask is None:
            return next_mask
        if next_mask is None:
            return current_mask
        return np.asarray(current_mask, dtype=bool) & np.asarray(next_mask, dtype=bool)

    def apply_output_mask(
        self,
        pixel_data: np.ndarray,
        output_mask: Any | None,
    ) -> np.ndarray:
        if output_mask is None:
            return pixel_data
        return pixel_data * self.project_mask_to_pixels(pixel_data, output_mask)

    @staticmethod
    def project_mask_to_pixels(pixel_data: np.ndarray, output_mask: Any) -> np.ndarray:
        mask_array = np.asarray(output_mask, dtype=bool)
        if mask_array.shape == pixel_data.shape:
            return mask_array

        if pixel_data.ndim == 2:
            if mask_array.ndim == 3 and mask_array.shape[0] == 1:
                mask_array = mask_array[0]
            if mask_array.shape == pixel_data.shape:
                return mask_array

        if pixel_data.ndim == 3:
            if mask_array.ndim == 2 and mask_array.shape == pixel_data.shape[:2]:
                return mask_array[:, :, np.newaxis]
            if mask_array.ndim == 2 and mask_array.shape == pixel_data.shape[1:]:
                return mask_array[np.newaxis, :, :]
            if (
                mask_array.ndim == 3
                and mask_array.shape[:2] == pixel_data.shape[:2]
                and pixel_data.shape[2] in (3, 4)
            ):
                return mask_array[:, :, :1]

        if pixel_data.ndim == 4:
            if mask_array.ndim == 2 and mask_array.shape == pixel_data.shape[1:3]:
                return mask_array[np.newaxis, :, :, np.newaxis]
            if mask_array.ndim == 3 and mask_array.shape == pixel_data.shape[:3]:
                return mask_array[:, :, :, np.newaxis]

        return mask_array


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
    mask_policy = ImageMathMaskPolicy(ignore_masks=ignore_masks)
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
    operand_masks = mask_policy.operand_masks(source_payload, operand_count)
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

    output_mask = mask_policy.output_mask(operand_masks)
    output_pixel_data = mask_policy.apply_output_mask(output_pixel_data, output_mask)

    output = output_pixel_data.astype(np.float32)
    if ignore_masks:
        return output
    return image_payload_with_context(
        output,
        mask=output_mask,
        metadata=image_payload_metadata(source_payload).without_unit_interval_intensity_scale(),
    )
