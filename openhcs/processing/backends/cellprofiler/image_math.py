"""ImageMath operation semantics for CellProfiler-compatible processing."""

from __future__ import annotations
from enum import Enum
from openhcs.core.artifacts import (
    ArtifactSpecRef,
    ArtifactSpecRelation,
    GroupLineageSourceRelation,
    ImageArtifactType,
)
from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerKwargDict
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    SpecialInputBindingRequest,
    TrailingImageSpecialInputPolicy,
)
from openhcs.core.registry_strategies import enum_member_with_payload
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactOutputCapability,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import optional_setting_value
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


class ImageMathOperation(Enum):
    """ImageMath operation literals exposed by CellProfiler settings."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "ImageMathOperation":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    ADD = ("add",)
    SUBTRACT = ("subtract",)
    DIFFERENCE = ("absolute_difference", "difference")
    MULTIPLY = ("multiply",)
    DIVIDE = ("divide",)
    AVERAGE = ("average",)
    MINIMUM = ("minimum",)
    MAXIMUM = ("maximum",)
    STDEV = ("standard_deviation", "stdev")
    INVERT = ("invert",)
    COMPLEMENT = ("complement",)
    LOG_TRANSFORM = ("log_transform_base2", "log_transform", "log_transform_base_2")
    LOG_TRANSFORM_LEGACY = ("log_transform_legacy",)
    NONE = ("none",)
    OR = ("or",)
    AND = ("and",)
    NOT = ("not",)
    EQUALS = ("equals",)

    def matches_cellprofiler_literal(self, value: str) -> bool:
        """Return whether a CP setting literal names this operation."""
        normalized = normalize_cellprofiler_setting_name(value)
        return normalized in {
            normalize_cellprofiler_setting_name(literal)
            for literal in (self.name, *self.cellprofiler_literals)
        }

    @classmethod
    def from_cellprofiler_literal(cls, value: str) -> "ImageMathOperation":
        """Return the operation named by a CellProfiler setting literal."""
        matches = tuple(
            (
                operation
                for operation in cls
                if operation.matches_cellprofiler_literal(value)
            )
        )
        if len(matches) == 1:
            return matches[0]
        return coerce_cellprofiler_enum(cls, value)


def parse_image_math_operation(value: str) -> str:
    """Return the absorbed-function operation literal for a CP setting."""
    return ImageMathOperation.from_cellprofiler_literal(value).value


class ImageMathSpecialInputPolicy(TrailingImageSpecialInputPolicy):
    """Bind trailing ImageMath image inputs as ordered operands."""

    def bind(self, request: SpecialInputBindingRequest) -> CellProfilerKwargDict:
        return {
            "image_operands": tuple(
                (request.runtime_value(spec) for spec in request.image_inputs)
            )
        }


class ImageMathModule(
    ImageMathSpecialInputPolicy,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "ImageMath"
    function_name = "image_math"
    validated = True
    contract = ProcessingContract.FLEXIBLE
    confidence = 1.0
    image_operand_settings = (
        "Select the first image",
        "Select the second image",
        "Select the third image",
        "Select the fourth image",
    )
    output_image_setting = "Name the output image"
    image_input_settings = image_operand_settings
    image_output_settings = (output_image_setting,)
    operand_factor_settings = (
        "Multiply the first image by",
        "Multiply the second image by",
        "Multiply the third image by",
        "Multiply the fourth image by",
    )
    operand_choice_setting = "Image or measurement?"
    ignored_settings = (
        *operand_factor_settings,
        *image_operand_settings,
        operand_choice_setting,
        "Measurement",
    )
    setting_bindings = (
        SettingToKeywordBinding("Operation", "operation", parse_image_math_operation),
        SettingToKeywordBinding(
            "Raise the power of the result by", "exponent", parse_cellprofiler_float
        ),
        SettingToKeywordBinding(
            "Multiply the result by", "after_factor", parse_cellprofiler_float
        ),
        SettingToKeywordBinding("Add to result", "addend", parse_cellprofiler_float),
        SettingToKeywordBinding(
            "Set values less than 0 equal to 0?",
            "truncate_low",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Set values greater than 1 equal to 1?",
            "truncate_high",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Replace invalid values with 0?", "replace_nan", parse_cellprofiler_bool
        ),
        SettingToKeywordBinding(
            "Ignore the image masks?", "ignore_masks", parse_cellprofiler_bool
        ),
    )

    @classmethod
    def declared_output_artifact_relations(
        cls,
        builder,
        module,
        *,
        setting,
        capability_type,
        name,
    ) -> tuple[ArtifactSpecRelation, ...]:
        relations = super().declared_output_artifact_relations(
            builder,
            module,
            setting=setting,
            capability_type=capability_type,
            name=name,
        )
        if capability_type is not ImageArtifactOutputCapability:
            return relations
        source_names = cls.active_image_operand_names(module)
        if len(source_names) != 1:
            return relations
        return (
            *relations,
            GroupLineageSourceRelation(
                source=ArtifactSpecRef.input(source_names[0], ImageArtifactType)
            ),
        )

    @classmethod
    def active_image_operand_names(cls, module) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                name
                for setting in cls.image_operand_settings
                for name in cls.artifact_input_names_from_setting(module, setting)
            )
        )

    @classmethod
    def source_binding_participates_in_image_stack(
        cls,
        module: "ModuleBlock",
        symbol: "CellProfilerSymbol",
        input_symbols: tuple["CellProfilerSymbol", ...],
    ) -> bool:
        del module
        if symbol.artifact_spec.artifact_type is not ImageArtifactType:
            return True
        first_external_image = next(
            (
                candidate
                for candidate in input_symbols
                if candidate.is_external_source
                and candidate.artifact_spec.artifact_type is ImageArtifactType
            ),
            None,
        )
        if first_external_image is None:
            return True
        return symbol.key == first_external_image.key

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        factors = tuple(
            (
                parse_cellprofiler_float(value)
                for setting_name in cls.operand_factor_settings
                if (value := optional_setting_value(module, setting_name)) is not None
            )
        )
        if not factors:
            return bound
        return bound.with_kwargs({"factors": factors})


from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.aligned_image_payload import ImagePayloadSourceSpatialDomainAdapter
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.runtime_values import (
    DerivedImagePayloadContext,
    ImagePayloadMetadataCompositionRequest,
    ImagePayloadMetadataInput,
    RuntimeArrayData,
    image_payload_mask,
)
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.runtime_values import (
    RuntimeImagePayloadContext,
    image_payload_data,
    image_payload_metadata,
)

MathOperation = ImageMathOperation
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum

ImageMathBinaryOperator = Callable[[np.ndarray, np.ndarray], np.ndarray]
ImageMathMask = RuntimeArrayData | None
ImageMathOperandPixels = np.ndarray | tuple[np.ndarray, ...]


def meaningful_image_math_mask(mask: ImageMathMask) -> ImageMathMask:
    """Return ``None`` when an ImageMath mask excludes no output pixels."""
    if mask is None:
        return None
    mask_array = np.asarray(mask, dtype=bool)
    if bool(np.all(mask_array)):
        return None
    return mask


class ImageMathOperationStrategy(
    EnumKeyedStrategyMixin[MathOperation], ABC, metaclass=AutoRegisterMeta
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
        return all((pd.dtype == bool for pd in pixel_data if not np.isscalar(pd)))

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

    pairwise_operator: ClassVar[ImageMathBinaryOperator]

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        for pd in pixel_data[1:]:
            output_pixel_data = type(self).pairwise_operator(output_pixel_data, pd)
        return output_pixel_data


class AddImageMathOperationStrategy(PairwiseNumpyImageMathOperationStrategy):
    operation = MathOperation.ADD
    pairwise_operator = np.add


class DivideImageMathOperationStrategy(PairwiseNumpyImageMathOperationStrategy):
    operation = MathOperation.DIVIDE
    pairwise_operator = np.divide


class MinimumImageMathOperationStrategy(PairwiseNumpyImageMathOperationStrategy):
    operation = MathOperation.MINIMUM
    pairwise_operator = np.minimum


class MaximumImageMathOperationStrategy(PairwiseNumpyImageMathOperationStrategy):
    operation = MathOperation.MAXIMUM
    pairwise_operator = np.maximum


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


class InvertImageMathOperationStrategy(InvertingImageMathOperationStrategy):
    operation = MathOperation.INVERT


class ComplementImageMathOperationStrategy(InvertingImageMathOperationStrategy):
    operation = MathOperation.COMPLEMENT


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
    logical_operator: ClassVar[ImageMathBinaryOperator]

    def apply(
        self,
        output_pixel_data: np.ndarray,
        pixel_data: list[np.ndarray],
        factors: tuple[float, ...],
    ) -> np.ndarray:
        del factors
        for pd in pixel_data[1:]:
            output_pixel_data = type(self).logical_operator(output_pixel_data, pd)
        return output_pixel_data.astype(np.float64)


class OrImageMathOperationStrategy(LogicalReductionImageMathOperationStrategy):
    operation = MathOperation.OR
    logical_operator = np.logical_or


class AndImageMathOperationStrategy(LogicalReductionImageMathOperationStrategy):
    operation = MathOperation.AND
    logical_operator = np.logical_and


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
class ImageMathMaskPolicy:
    """CellProfiler ImageMath mask projection and output-composition policy."""

    ignore_masks: bool

    def operand_masks(
        self, operands: tuple[ImagePayloadMetadataInput, ...]
    ) -> tuple[ImageMathMask, ...]:
        return tuple((image_payload_mask(operand) for operand in operands))

    def stacked_operand_masks(
        self, source_payload: ImagePayloadMetadataInput, operand_count: int
    ) -> tuple[ImageMathMask, ...]:
        source_mask = image_payload_mask(source_payload)
        if source_mask is None:
            return (None,) * operand_count
        mask_array = np.asarray(source_mask, dtype=bool)
        if mask_array.ndim >= 3 and mask_array.shape[0] >= operand_count:
            return tuple((mask_array[index] for index in range(operand_count)))
        return (mask_array,) + (None,) * max(operand_count - 1, 0)

    def output_mask(self, operand_masks: tuple[ImageMathMask, ...]) -> ImageMathMask:
        if self.ignore_masks:
            return None
        if not operand_masks:
            return None
        output_mask = operand_masks[0]
        for mask in operand_masks[1:]:
            output_mask = self.combine_output_masks(output_mask, mask)
        return meaningful_image_math_mask(output_mask)

    @staticmethod
    def combine_output_masks(
        current_mask: ImageMathMask, next_mask: ImageMathMask
    ) -> ImageMathMask:
        if current_mask is None:
            return next_mask
        if next_mask is None:
            return current_mask
        return np.asarray(current_mask, dtype=bool) & np.asarray(next_mask, dtype=bool)

    def apply_output_mask(
        self, pixel_data: np.ndarray, output_mask: ImageMathMask
    ) -> np.ndarray:
        if output_mask is None:
            return pixel_data
        return pixel_data * ImageMathMaskProjectionStrategy.project_mask(
            pixel_data, output_mask
        )


class ImageMathMaskProjectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Closed projection family for mapping ImageMath masks to output pixels."""

    __registry_key__ = "pixel_ndim"
    __skip_if_no_key__ = True
    pixel_ndim: ClassVar[int | None] = None

    @classmethod
    def for_pixel_data(
        cls, pixel_data: np.ndarray
    ) -> "ImageMathMaskProjectionStrategy":
        strategy_type = cls.__registry__.get(pixel_data.ndim)
        if strategy_type is None:
            return IdentityMaskProjectionStrategy()
        return strategy_type()

    @classmethod
    def project_mask(
        cls, pixel_data: np.ndarray, output_mask: RuntimeArrayData
    ) -> np.ndarray:
        mask_array = np.asarray(output_mask, dtype=bool)
        if mask_array.shape == pixel_data.shape:
            return mask_array
        return cls.for_pixel_data(pixel_data).project(pixel_data, mask_array)

    @abstractmethod
    def project(self, pixel_data: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        """Return the mask array projected into ``pixel_data`` shape."""


class IdentityMaskProjectionStrategy(ImageMathMaskProjectionStrategy):
    """Leave masks in their native shape when no closed projection case applies."""

    def project(self, pixel_data: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        del pixel_data
        return mask_array


class PlanarMaskProjectionStrategy(ImageMathMaskProjectionStrategy):
    """Project singleton stack masks into 2D ImageMath outputs."""

    pixel_ndim = 2

    def project(self, pixel_data: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        pixel_shape = tuple(pixel_data.shape)
        mask_shape = tuple(mask_array.shape)
        if mask_shape == (1, *pixel_shape):
            return mask_array[0]
        return mask_array


class VolumeMaskProjectionStrategy(ImageMathMaskProjectionStrategy):
    """Project planar masks into 3D ImageMath outputs."""

    pixel_ndim = 3
    color_channel_counts: ClassVar[frozenset[int]] = frozenset((3, 4))

    def project(self, pixel_data: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        pixel_shape = tuple(pixel_data.shape)
        mask_shape = tuple(mask_array.shape)
        if mask_shape == pixel_shape[:2]:
            return mask_array[:, :, np.newaxis]
        if mask_shape == pixel_shape[1:]:
            return mask_array[np.newaxis, :, :]
        if (
            mask_shape[:2] == pixel_shape[:2]
            and pixel_shape[2] in type(self).color_channel_counts
        ):
            return mask_array[:, :, :1]
        return mask_array


class FourDimensionalMaskProjectionStrategy(ImageMathMaskProjectionStrategy):
    """Project planar or volumetric masks into 4D ImageMath outputs."""

    pixel_ndim = 4

    def project(self, pixel_data: np.ndarray, mask_array: np.ndarray) -> np.ndarray:
        pixel_shape = tuple(pixel_data.shape)
        mask_shape = tuple(mask_array.shape)
        if mask_shape == pixel_shape[1:3]:
            return mask_array[np.newaxis, :, :, np.newaxis]
        if mask_shape == pixel_shape[:3]:
            return mask_array[:, :, :, np.newaxis]
        return mask_array


@dataclass(frozen=True, slots=True)
class ImageMathPreparedOperands:
    """Aligned ImageMath operands with masks and normalized factors."""

    source_image: ImagePayloadMetadataInput
    source_payloads: tuple[ImagePayloadMetadataInput, ...]
    operand_pixels: ImageMathOperandPixels
    operand_masks: tuple[ImageMathMask, ...]
    factors: tuple[float, ...]

    @classmethod
    def from_inputs(
        cls,
        *,
        image: RuntimeArrayData,
        image_operands: tuple[ImagePayloadMetadataInput, ...],
        operation_strategy: ImageMathOperationStrategy,
        factors: tuple[float, ...],
        mask_policy: ImageMathMaskPolicy,
    ) -> "ImageMathPreparedOperands":
        source_payloads = cls._source_payloads(image, image_operands)
        operand_pixels = cls._operand_pixels(image, source_payloads, image_operands)
        if operation_strategy.single_image:
            operand_count = 1
        else:
            operand_count = cls._operand_count(operand_pixels)
        return cls(
            source_image=image,
            source_payloads=source_payloads,
            operand_pixels=operand_pixels,
            operand_masks=cls._operand_masks(
                image, image_operands, source_payloads, operand_count, mask_policy
            ),
            factors=cls._factors_for_operands(
                factors, cls._operand_count(operand_pixels)
            ),
        )

    @staticmethod
    def _source_payloads(
        image: RuntimeArrayData, image_operands: tuple[ImagePayloadMetadataInput, ...]
    ) -> tuple[ImagePayloadMetadataInput, ...]:
        if not image_operands:
            return (image,)
        return ImagePayloadSourceSpatialDomainAdapter.payloads_aligned_to_common_source_domain(
            (image, *image_operands)
        )

    @staticmethod
    def _operand_pixels(
        image: RuntimeArrayData,
        source_payloads: tuple[ImagePayloadMetadataInput, ...],
        image_operands: tuple[ImagePayloadMetadataInput, ...],
    ) -> ImageMathOperandPixels:
        if image_operands:
            return tuple(
                (np.asarray(image_payload_data(payload)) for payload in source_payloads)
            )
        operand_pixels = np.asarray(image_payload_data(image))
        if operand_pixels.ndim == 2:
            return operand_pixels[np.newaxis, :, :]
        return operand_pixels

    @staticmethod
    def _operand_count(operand_pixels: ImageMathOperandPixels) -> int:
        if isinstance(operand_pixels, tuple):
            return len(operand_pixels)
        return int(operand_pixels.shape[0])

    def operand_pixel(self, index: int) -> np.ndarray:
        """Return one logical operand without forcing stacked copies."""
        if isinstance(self.operand_pixels, tuple):
            return self.operand_pixels[index]
        return self.operand_pixels[index]

    def initial_output_pixels(self) -> np.ndarray:
        """Return the data passed to the operation's output initializer."""
        if isinstance(self.operand_pixels, tuple):
            return self.operand_pixels[0]
        return self.operand_pixels

    @staticmethod
    def _operand_masks(
        image: RuntimeArrayData,
        image_operands: tuple[ImagePayloadMetadataInput, ...],
        source_payloads: tuple[ImagePayloadMetadataInput, ...],
        operand_count: int,
        mask_policy: ImageMathMaskPolicy,
    ) -> tuple[ImageMathMask, ...]:
        if image_operands:
            return mask_policy.operand_masks(source_payloads[:operand_count])
        return mask_policy.stacked_operand_masks(image, operand_count)

    @staticmethod
    def _factors_for_operands(
        factors: tuple[float, ...], image_count: int
    ) -> tuple[float, ...]:
        if len(factors) >= image_count:
            return factors
        return tuple(factors) + (1.0,) * (image_count - len(factors))

    @property
    def image_count(self) -> int:
        """Return the number of operand planes presented to ImageMath."""
        return self._operand_count(self.operand_pixels)

    def output_value(
        self, output: np.ndarray, output_mask: ImageMathMask
    ) -> RuntimeArrayData:
        """Return a runtime image payload preserving the source stack identity."""
        value_payload = RuntimeImagePayloadContext(
            output,
            mask=output_mask,
            metadata=ImagePayloadMetadataCompositionRequest(
                self.source_payloads[: self.image_count]
            )
            .metadata()
            .without_unit_interval_intensity_scale(),
        ).payload()
        return DerivedImagePayloadContext(
            source_payload=self.source_image, data=value_payload
        ).payload()


@special_inputs("image_operands")
@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def image_math(
    image: RuntimeArrayData,
    image_operands: tuple[ImagePayloadMetadataInput, ...] = (),
    operation: MathOperation = MathOperation.ADD,
    factors: tuple[float, ...] = (1.0, 1.0),
    exponent: float = 1.0,
    after_factor: float = 1.0,
    addend: float = 0.0,
    truncate_low: bool = True,
    truncate_high: bool = True,
    replace_nan: bool = True,
    ignore_masks: bool = False,
) -> np.ndarray:
    """Perform CellProfiler ImageMath through registered operation strategies."""
    operation_strategy = ImageMathOperationStrategy.coerce(operation)
    mask_policy = ImageMathMaskPolicy(ignore_masks=ignore_masks)
    prepared_operands = ImageMathPreparedOperands.from_inputs(
        image=image,
        image_operands=image_operands,
        operation_strategy=operation_strategy,
        factors=factors,
        mask_policy=mask_policy,
    )
    pixel_data = []
    if operation_strategy.single_image:
        operand_count = 1
    else:
        operand_count = prepared_operands.image_count
    for index in range(operand_count):
        pixel = prepared_operands.operand_pixel(index).astype(np.float64)
        factor = prepared_operands.factors[index]
        if not operation_strategy.binary_output and factor != 1.0:
            pixel = pixel * factor
        pixel_data.append(pixel)
    output_pixel_data = operation_strategy.apply(
        operation_strategy.prepare_initial_output(
            prepared_operands.initial_output_pixels(),
            pixel_data,
            prepared_operands.factors,
        ),
        pixel_data,
        prepared_operands.factors,
    )
    if not operation_strategy.binary_output:
        if exponent != 1.0:
            output_pixel_data = output_pixel_data**exponent
        if after_factor != 1.0:
            output_pixel_data = output_pixel_data * after_factor
        if addend != 0.0:
            output_pixel_data = output_pixel_data + addend
        if truncate_low:
            output_pixel_data[output_pixel_data < 0] = 0
        if truncate_high:
            output_pixel_data[output_pixel_data > 1] = 1
        if replace_nan:
            output_pixel_data[np.isnan(output_pixel_data)] = 0
    if output_pixel_data.ndim == 2:
        output_pixel_data = output_pixel_data[np.newaxis, :, :]
    output_mask = mask_policy.output_mask(prepared_operands.operand_masks)
    output_pixel_data = mask_policy.apply_output_mask(output_pixel_data, output_mask)
    output = output_pixel_data.astype(np.float32)
    if ignore_masks:
        return output
    return prepared_operands.output_value(output, output_mask)
