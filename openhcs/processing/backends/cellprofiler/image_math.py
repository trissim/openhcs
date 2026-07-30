"""ImageMath operation semantics for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpecCollection,
    ImageArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.memory.decorators import numpy as numpy_decorator
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
    enum_member_with_payload,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    project_image_mask_to_data_domain,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.module_settings import BoundModuleSettings
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_names,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    ProcessingContract,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.interop.cellprofiler.parser import ModuleBlock


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


class ImageMathModule(
    CellProfilerModule,
):
    module_name = "ImageMath"
    function_name = "image_math"
    validated = True
    confidence = 1.0
    first_image_setting = SettingNameFamily("Select the first image")
    second_image_setting = SettingNameFamily("Select the second image")
    third_image_setting = SettingNameFamily("Select the third image")
    fourth_image_setting = SettingNameFamily("Select the fourth image")
    image_operand_settings = (
        first_image_setting,
        second_image_setting,
        third_image_setting,
        fourth_image_setting,
    )
    output_image_setting = SettingNameFamily("Name the output image")
    operand_factor_settings = (
        "Multiply the first image by",
        "Multiply the second image by",
        "Multiply the third image by",
        "Multiply the fourth image by",
    )
    operand_choice_setting = SettingNameFamily("Image or measurement?")
    measurement_operand_setting = SettingNameFamily("Measurement")
    operation_setting = SettingNameFamily("Operation")
    operation_binding = SettingToKeywordBinding(
        operation_setting,
        "operation",
        parse_image_math_operation,
    )
    setting_bindings = (*tuple(
        SettingToKeywordBinding.input(setting, ImageArtifactType)
        for setting in image_operand_settings
    ), SettingToKeywordBinding.output(output_image_setting, ImageArtifactType),operation_binding,
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
        ),)

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Expose only the image operands active for this module value."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        active_operands = frozenset(cls._active_image_operand_bindings(module))
        declared_operands = frozenset(
            cls.declared_artifact_bindings(
                plan_type=ArtifactInputPlan,
                artifact_type=ImageArtifactType,
            )
        )
        return tuple(
            binding
            for binding in bindings
            if binding not in declared_operands or binding in active_operands
        )

    @classmethod
    def artifact_input_bindings_for_reconstruction(
        cls,
        module,
        *,
        invocation_key,
        step_context,
    ):
        """Select operation-owned operands before root ordered assignment."""

        bindings = super().artifact_input_bindings_for_reconstruction(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
        )
        declared_operands = cls.declared_artifact_bindings(
            plan_type=ArtifactInputPlan,
            artifact_type=ImageArtifactType,
        )
        expected_settings = frozenset(
            cls._image_operand_settings_for_records(
                tuple(module.iter_settings()),
                available_image_count=len(
                    step_context.main_flow_artifacts.of_artifact_type(
                        ImageArtifactType
                    )
                ),
            )
        )
        selected_operands = tuple(
            binding
            for binding, setting in zip(
                declared_operands,
                cls.image_operand_settings,
                strict=True,
            )
            if setting in expected_settings
        )
        return tuple(
            binding
            for binding in dict.fromkeys((*bindings, *selected_operands))
            if binding not in declared_operands or binding in selected_operands
        )

    @classmethod
    def artifact_output_relations(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        binding,
        name,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ):
        """Preserve the first active operand's source-stack scope."""
        del (
            invocation_key,
            step_context,
            binding,
            name,
        )
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if not image_inputs:
            raise ValueError("ImageMath requires at least one active image operand.")
        return (SourceStackLineageSourceRelation(source=image_inputs[0].ref()),)

    @classmethod
    def _active_image_operand_bindings(
        cls,
        module: "ModuleBlock",
    ) -> tuple[SettingToKeywordBinding, ...]:
        operation_value = optional_setting_value(module, cls.operation_setting)
        if operation_value is not None:
            operation_strategy = ImageMathOperationStrategy.for_operation(
                ImageMathOperation.from_cellprofiler_literal(operation_value)
            )
            if operation_strategy.single_image:
                return cls.declared_artifact_bindings(plan_type = ArtifactInputPlan, artifact_type = ImageArtifactType)[:1]
        return tuple(
            binding
            for binding, setting in zip(
                cls.declared_artifact_bindings(plan_type = ArtifactInputPlan, artifact_type = ImageArtifactType),
                cls.image_operand_settings,
                strict=True,
            )
            if any(
                split_symbol_names(value) for value in setting_values(module, setting)
            )
        )

    @classmethod
    def _image_operand_settings_for_records(
        cls,
        existing_records,
        *,
        available_image_count: int,
    ) -> tuple[str, ...]:
        operation_strategy = cls._operation_strategy_from_records(existing_records)
        if operation_strategy is not None and operation_strategy.single_image:
            count = 1
        else:
            explicit_positions = tuple(
                position + 1
                for position, setting in enumerate(cls.image_operand_settings)
                if cls._image_names_for_setting_records(existing_records, setting)
            )
            count = max((1, available_image_count, *explicit_positions))
        if count > len(cls.image_operand_settings):
            raise ValueError(
                "ImageMath supports at most "
                f"{len(cls.image_operand_settings)} image operands, got {count}."
            )
        return cls.image_operand_settings[:count]

    @classmethod
    def _operation_strategy_from_records(cls, existing_records):
        values = cls._setting_record_values(
            existing_records, cls.operation_binding.setting_name
        )
        if len(values) > 1:
            raise ValueError(f"ImageMath declares multiple operation rows: {values!r}.")
        if not values:
            return None
        return ImageMathOperationStrategy.for_operation(
            ImageMathOperation.from_cellprofiler_literal(values[0])
        )

    @staticmethod
    def _setting_record_values(records, setting) -> tuple[str, ...]:
        from openhcs.interop.cellprofiler.setting_names import setting_name_matches

        return tuple(
            str(record.value)
            for record in records
            if setting_name_matches(record.name, setting)
        )

    @classmethod
    def _image_names_for_setting_records(cls, records, setting) -> tuple[str, ...]:
        return tuple(
            image_name
            for value in cls._setting_record_values(records, setting)
            for image_name in split_symbol_names(value)
        )

    @classmethod
    def active_image_operand_names(cls, module) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                name
                for binding in cls._active_image_operand_bindings(module)
                for name in cls.artifact_names_for_binding(module, binding)
            )
        )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        operand_choices = tuple(
            normalize_cellprofiler_setting_name(value)
            for value in setting_values(module, cls.operand_choice_setting)
        )
        unsupported_choices = tuple(
            choice for choice in operand_choices if choice not in {"", "image"}
        )
        if unsupported_choices:
            raise NotImplementedError(
                "ImageMath measurement operands are not supported by the "
                f"absorbed callable: {unsupported_choices!r}."
            )

        factors = tuple(
            (
                parse_cellprofiler_float(value)
                for setting_name in cls.operand_factor_settings
                if (value := optional_setting_value(module, setting_name)) is not None
            )
        )
        private_setting_names = (
            *cls.operand_factor_settings,
            cls.operand_choice_setting,
            cls.measurement_operand_setting,
        )
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting in private_setting_names:
            for setting_name in setting_names(setting):
                unmapped_kwargs.pop(
                    normalize_cellprofiler_setting_name(setting_name),
                    None,
                )
        kwargs = dict(bound.kwargs)
        if factors:
            kwargs["factors"] = factors
        return BoundModuleSettings(
            kwargs,
            unmapped_kwargs,
            bound.setting_coverage,
        )


MathOperation = ImageMathOperation

ImageMathBinaryOperator = np.ufunc
ImageMathMask = RuntimeArrayData | None
ImageMathOperandPixels = tuple[np.ndarray, ...]


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
    def for_operation(cls, value: MathOperation) -> "ImageMathOperationStrategy":
        return cls.for_enum_member(value)

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
        return pixel_data[0]

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
            type(self).pairwise_operator(
                output_pixel_data,
                pd,
                out=output_pixel_data,
            )
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
            for pd in pixel_data[1:]:
                output_pixel_data[pd.astype(bool)] = False
            return output_pixel_data
        for pd in pixel_data[1:]:
            np.subtract(output_pixel_data, pd, out=output_pixel_data)
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
                np.logical_xor(output_pixel_data, pd, out=output_pixel_data)
            return output_pixel_data
        for pd in pixel_data[1:]:
            np.subtract(output_pixel_data, pd, out=output_pixel_data)
            np.abs(output_pixel_data, out=output_pixel_data)
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
                np.logical_and(output_pixel_data, pd, out=output_pixel_data)
            return output_pixel_data
        for pd in pixel_data[1:]:
            np.multiply(output_pixel_data, pd, out=output_pixel_data)
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
            np.add(output_pixel_data, pd, out=output_pixel_data)
        if not self.operands_are_logical(pixel_data):
            np.divide(
                output_pixel_data,
                sum(factors[: len(pixel_data)]),
                out=output_pixel_data,
            )
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
        self, operands: tuple[RuntimeArrayData, ...]
    ) -> tuple[ImageMathMask, ...]:
        return tuple((image_payload_mask(operand) for operand in operands))

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
        self,
        pixel_data: np.ndarray,
        output_mask: ImageMathMask,
        *,
        metadata: ImagePayloadMetadata,
    ) -> np.ndarray:
        if output_mask is None:
            return pixel_data
        projected_mask = project_image_mask_to_data_domain(
            output_mask,
            pixel_data,
            metadata=metadata,
        )
        if projected_mask is None:
            return pixel_data
        return pixel_data * metadata.mask_domain(pixel_data).broadcast_to_data(
            projected_mask
        )


@dataclass(frozen=True, slots=True)
class ImageMathPreparedOperands:
    """Aligned ImageMath operands with masks and normalized factors."""

    source_image: RuntimeArrayData
    source_payloads: tuple[RuntimeArrayData, ...]
    operand_pixels: ImageMathOperandPixels
    operand_masks: tuple[ImageMathMask, ...]
    factors: tuple[float, ...]

    @classmethod
    def from_inputs(
        cls,
        *,
        image: RuntimeArrayData,
        operation_strategy: ImageMathOperationStrategy,
        factors: tuple[float, ...],
        mask_policy: ImageMathMaskPolicy,
    ) -> "ImageMathPreparedOperands":
        source_payloads = cls._source_payloads(image)
        if operation_strategy.single_image and len(source_payloads) != 1:
            raise ValueError(
                f"ImageMath {operation_strategy.operation.value!r} requires exactly "
                f"one declared image operand, got {len(source_payloads)}."
            )
        operand_pixels = tuple(
            np.asarray(image_payload_data(payload)) for payload in source_payloads
        )
        return cls(
            source_image=image,
            source_payloads=source_payloads,
            operand_pixels=operand_pixels,
            operand_masks=mask_policy.operand_masks(source_payloads),
            factors=cls._factors_for_operands(factors, len(operand_pixels)),
        )

    @staticmethod
    def _source_payloads(
        image: RuntimeArrayData,
    ) -> tuple[RuntimeArrayData, ...]:
        metadata = image_payload_metadata(image)
        if metadata.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
            return (image,)
        axis_size = metadata.source_provenance.source_plane_count
        if axis_size <= 0:
            raise ValueError(
                "ImageMath SOURCE_BINDING operand payload has no declared source "
                "plane provenance."
            )
        projection = RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.SOURCE_BINDING,
            axis_size=axis_size,
            source_aliases=metadata.source_image_names,
        )
        projection.validate_shape(
            np.asarray(image_payload_data(image)).shape,
            value_name="ImageMath operand payload",
        )
        return tuple(
            RuntimeSliceProjection.value_for_slice(
                image,
                projection.selected_plane(index),
            )
            for index in range(axis_size)
        )

    def operand_pixel(self, index: int) -> np.ndarray:
        """Return one logical operand without forcing stacked copies."""
        return self.operand_pixels[index]

    def initial_output_pixels(self) -> np.ndarray:
        """Return the data passed to the operation's output initializer."""
        return self.operand_pixels[0]

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
        return len(self.operand_pixels)

    def output_metadata(self) -> ImagePayloadMetadata:
        metadata = image_payload_metadata(self.source_image)
        if metadata.unit_interval_intensity is not None:
            metadata = metadata.without_unit_interval_intensity_scale()
        if metadata.plane_axis is RuntimePlaneAxis.SOURCE_BINDING:
            return metadata.collapse_leading_plane_axis()
        return metadata

    def output_value(
        self, output: np.ndarray, output_mask: ImageMathMask
    ) -> RuntimeArrayData:
        """Return a runtime image payload preserving the source stack identity."""
        value_payload = self.output_metadata().payload_with(output, output_mask)
        return value_payload


@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
def image_math(
    image: RuntimeArrayData,
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
    """Perform CellProfiler ImageMath through registered operation strategies.

    Args:
        factors: Multipliers for the ordered input images; omitted trailing
            factors default to 1 and binary-output operations ignore them.
    """
    operation_strategy = ImageMathOperationStrategy.for_operation(operation)
    mask_policy = ImageMathMaskPolicy(ignore_masks=ignore_masks)
    prepared_operands = ImageMathPreparedOperands.from_inputs(
        image=image,
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
    output_mask = mask_policy.output_mask(prepared_operands.operand_masks)
    output_metadata = prepared_operands.output_metadata()
    output_pixel_data = mask_policy.apply_output_mask(
        output_pixel_data,
        output_mask,
        metadata=output_metadata,
    )
    output = output_pixel_data.astype(np.float32)
    return prepared_operands.output_value(
        output,
        None if ignore_masks else output_mask,
    )
