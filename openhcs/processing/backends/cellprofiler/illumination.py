"""Illumination backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
import logging
import time
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit

from openhcs.constants.constants import MemoryType, VariableComponents
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
)
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    GroupLineageSourceRelation,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    project_image_mask_to_data_domain,
)
from openhcs.core.steps.function_runtime import (
    RuntimeCallableArgument,
    RuntimeCallableKwargs,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.module_settings import BoundModuleSettings
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.cellprofiler._backend import (
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    BackendProviderInput,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.morphology import (
    MorphologyBackendStrategy,
)
from openhcs.processing.backends.cellprofiler.perf_fixtures import (
    capture_array_fixture,
    capture_enabled,
)
from openhcs.processing.backends.cellprofiler.smoothing import (
    MaskedLinearFilterRequest,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey


class IntensityChoice(Enum):
    """CellProfiler CorrectIlluminationCalculate intensity source."""

    REGULAR = "regular"
    BACKGROUND = "background"


class SmoothingMethod(Enum):
    """CellProfiler CorrectIlluminationCalculate smoothing method."""

    NONE = "none"
    CONVEX_HULL = "convex_hull"
    FIT_POLYNOMIAL = "fit_polynomial"
    MEDIAN_FILTER = "median_filter"
    GAUSSIAN_FILTER = "gaussian_filter"
    TO_AVERAGE = "to_average"
    SPLINES = "splines"


SmoothingMethod.NONE.cellprofiler_literals = ("No smoothing",)


class FilterSizeMethod(Enum):
    """CellProfiler CorrectIlluminationCalculate filter-size mode."""

    AUTOMATIC = "automatic"
    OBJECT_SIZE = "object_size"
    MANUALLY = "manually"


class RescaleOption(Enum):
    """CellProfiler CorrectIlluminationCalculate output rescale mode."""

    YES = "yes"
    NO = "no"
    MEDIAN = "median"


class SplineBgMode(Enum):
    """CellProfiler CorrectIlluminationCalculate spline background mode."""

    AUTO = "auto"
    DARK = "dark"
    BRIGHT = "bright"
    GRAY = "gray"


class CalculationScope(Enum):
    """CellProfiler CorrectIlluminationCalculate image aggregation scope."""

    EACH = "each"
    ALL_FIRST_CYCLE = "all_first_cycle"
    ALL_ACROSS_CYCLES = "all_across_cycles"

    @property
    def uses_all_images(self) -> bool:
        return self is not CalculationScope.EACH


class IlluminationCorrectionMethod(Enum):
    DIVIDE = "divide"
    SUBTRACT = "subtract"


class CorrectIlluminationApplyModule(
    CellProfilerModule,
):
    module_name = "CorrectIlluminationApply"
    function_name = "correct_illumination_apply"
    validated = True
    confidence = 1.0

    input_image_setting = "Select the input image"
    output_image_setting = "Name the output image"
    illumination_function_setting = "Select the illumination function"
    method_setting = "Select how the illumination function is applied"
    truncate_low_setting = "Set output image values less than 0 equal to 0?"
    truncate_high_setting = "Set output image values greater than 1 equal to 1?"
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    illumination_function_binding = SettingToKeywordBinding.input(
        illumination_function_setting,
        ImageArtifactType,
        runtime_parameter_name="illumination_function",
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_image_binding,
        illumination_function_binding,
        output_image_binding,
        SettingToKeywordBinding(
            method_setting,
            "method",
            cellprofiler_enum_value_setting_parser(IlluminationCorrectionMethod),
        ),
        SettingToKeywordBinding(
            truncate_low_setting,
            "truncate_low",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            truncate_high_setting,
            "truncate_high",
            parse_cellprofiler_bool,
        ),
    )

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        inputs = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        image_names = setting_values(module, cls.input_image_setting)
        illumination_names = setting_values(
            module,
            cls.illumination_function_setting,
        )
        if not illumination_names:
            return inputs.specs
        if len(image_names) != len(illumination_names):
            raise ValueError(
                f"Module {module.name}({module.module_num}) requires one "
                "illumination function for each input image."
            )
        related_inputs = ArtifactSpecCollection(
            illumination.with_group_scope_relation(
                InputStackBroadcastSourceRelation(source=image.ref()),
            )
            for image_name, illumination_name in zip(
                image_names,
                illumination_names,
                strict=True,
            )
            for image in (
                inputs.require_by_name_and_artifact_type(
                    image_name,
                    ImageArtifactType,
                ),
            )
            for illumination in (
                inputs.require_by_name_and_artifact_type(
                    illumination_name,
                    ImageArtifactType,
                ),
            )
            if illumination.ref() != image.ref()
        ).unique(conflict_context="illumination stack-broadcast input")
        replacements = {spec.ref(): spec for spec in related_inputs}
        return tuple(replacements.get(spec.ref(), spec) for spec in inputs.specs)

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
        del (
            invocation_key,
            step_context,
            binding,
        )
        image_names = setting_values(module, cls.input_image_setting)
        if not image_names or output_position >= len(image_names):
            raise ValueError(
                f"Module {module.name}({module.module_num}) output {name!r} at "
                f"position {output_position} has no corresponding input image."
            )
        source = artifact_inputs.require_by_name_and_artifact_type(
            image_names[output_position],
            ImageArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)

    @classmethod
    def invocation_module_blocks(cls, module: ModuleBlock) -> tuple[ModuleBlock, ...]:
        """Keep each repeated image/function/output tuple as one invocation."""

        image_names = setting_values(module, cls.input_image_setting)
        illumination_names = setting_values(module, cls.illumination_function_setting)
        output_names = setting_values(module, cls.output_image_setting)
        pair_count = len(image_names)
        if (
            pair_count <= 1
            or len(illumination_names) != pair_count
            or len(output_names) != pair_count
        ):
            return super().invocation_module_blocks(module)

        pair_setting_names = {
            cls.input_image_setting,
            cls.illumination_function_setting,
            cls.output_image_setting,
            cls.method_setting,
            cls.truncate_low_setting,
            cls.truncate_high_setting,
        }
        split_blocks: list[ModuleBlock] = []
        for pair_index in range(pair_count):
            occurrences: dict[str, int] = {}
            records: list[ModuleSetting] = []
            for record in module.iter_settings():
                if record.name not in pair_setting_names:
                    records.append(record)
                    continue
                values = module.get_setting_values(record.name)
                if len(values) != pair_count:
                    records.append(record)
                    continue
                occurrence = occurrences.get(record.name, 0)
                occurrences[record.name] = occurrence + 1
                if occurrence == pair_index:
                    records.append(record)
            module_num = module.module_num * 1000 + pair_index + 1
            split_blocks.append(
                replace(
                    module,
                    module_num=module_num,
                    setting_records=records,
                    metadata={**module.metadata, "module_num": str(module_num)},
                )
            )
        return tuple(split_blocks)

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        repeated_kwargs: dict[str, Any] = {}
        method_values = setting_values(module, cls.method_setting)
        if len(method_values) > 1:
            repeated_kwargs["method"] = tuple(
                (
                    coerce_cellprofiler_enum(IlluminationCorrectionMethod, value).value
                    for value in method_values
                )
            )
        repeated_kwargs.update(
            cls._repeated_bool_setting(
                module,
                cls.truncate_low_setting,
                "truncate_low",
            )
        )
        repeated_kwargs.update(
            cls._repeated_bool_setting(
                module,
                cls.truncate_high_setting,
                "truncate_high",
            )
        )
        return bound.with_kwargs(repeated_kwargs)

    @staticmethod
    def _repeated_bool_setting(
        module: "ModuleBlock", setting_name: str, parameter_name: str
    ) -> dict[str, tuple[bool, ...]]:
        values = setting_values(module, setting_name)
        if len(values) <= 1:
            return {}
        return {
            parameter_name: tuple((parse_cellprofiler_bool(value) for value in values))
        }


class IlluminationCalculationScopeExecutionModePolicy:
    """Run all-image illumination calculation once over the full image stack."""

    @classmethod
    def execution_mode(
        cls,
        default: ImagePayloadExecutionMode,
        *,
        image: RuntimeCallableArgument,
        kwargs: RuntimeCallableKwargs,
        variable_components: tuple[VariableComponents, ...],
    ) -> ImagePayloadExecutionMode:
        del cls, image, variable_components
        scope = coerce_cellprofiler_enum(
            CalculationScope,
            kwargs.get("calculation_scope", CalculationScope.EACH),
        )
        if scope.uses_all_images:
            return ImagePayloadExecutionMode.FULL_STACK
        return default


class CorrectIlluminationCalculateModule(
    IlluminationCalculationScopeExecutionModePolicy,
    CellProfilerModule,
):
    module_name = "CorrectIlluminationCalculate"
    function_name = "correct_illumination_calculate"
    validated = True
    confidence = 1.0

    @classmethod
    def main_flow_output_specs(
        cls,
        main_flow_candidates: tuple[ArtifactSpec, ...],
    ) -> tuple[ArtifactSpec, ...]:
        """Record illumination images while preserving the incoming main flow."""

        del cls, main_flow_candidates
        return ()

    input_image_setting = "Select the input image"
    output_image_setting = "Name the output image"
    retain_average_setting = "Retain the averaged image?"
    average_image_setting = "Name the averaged image"
    retain_dilated_setting = "Retain the dilated image?"
    dilated_image_setting = "Name the dilated image"
    calculation_scope_setting = (
        "Calculate function for each image individually, or based on all images?"
    )
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType
    )
    average_image_binding = SettingToKeywordBinding.output(
        average_image_setting,
        ImageArtifactType,
        "average_image_name",
    )
    dilated_image_binding = SettingToKeywordBinding.output(
        dilated_image_setting,
        ImageArtifactType,
        "dilated_image_name",
    )

    @staticmethod
    def calculation_scope_literal(value: str) -> str:
        cleaned = (
            value.replace("\x00", "")
            .replace("\ufeff", "")
            .replace("ÿþ", "")
            .replace("þÿ", "")
        )
        return coerce_cellprofiler_enum(CalculationScope, cleaned).value

    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_image_binding,
        output_image_binding,
        average_image_binding,
        dilated_image_binding,
        SettingToKeywordBinding(
            retain_average_setting,
            "retain_average",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            retain_dilated_setting,
            "retain_dilated",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Select how the illumination function is calculated",
            "intensity_choice",
            cellprofiler_enum_value_setting_parser(IntensityChoice),
        ),
        SettingToKeywordBinding(
            "Dilate objects in the final averaged image?",
            "dilate_objects",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Dilation radius", "object_dilation_radius", parse_cellprofiler_int
        ),
        SettingToKeywordBinding("Block size", "block_size", parse_cellprofiler_int),
        SettingToKeywordBinding(
            "Rescale the illumination function?",
            "rescale_option",
            cellprofiler_enum_value_setting_parser(RescaleOption),
        ),
        SettingToKeywordBinding(
            calculation_scope_setting,
            "calculation_scope",
            calculation_scope_literal,
        ),
        SettingToKeywordBinding(
            "Smoothing method",
            "smoothing_method",
            cellprofiler_enum_value_setting_parser(SmoothingMethod),
        ),
        SettingToKeywordBinding(
            "Method to calculate smoothing filter size",
            "filter_size_method",
            cellprofiler_enum_value_setting_parser(FilterSizeMethod),
        ),
        SettingToKeywordBinding(
            SettingNameFamily(
                "Approximate object diameter", aliases=("Approximate object size",)
            ),
            "object_width",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Smoothing filter size", "manual_filter_size", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            "Automatically calculate spline parameters?",
            "automatic_splines",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Background mode",
            "spline_bg_mode",
            cellprofiler_enum_value_setting_parser(SplineBgMode),
        ),
        SettingToKeywordBinding(
            "Number of spline points", "spline_points", parse_cellprofiler_int
        ),
        SettingToKeywordBinding(
            "Background threshold", "spline_threshold", parse_cellprofiler_float
        ),
        SettingToKeywordBinding(
            "Image resampling factor", "spline_rescale", parse_cellprofiler_float
        ),
        SettingToKeywordBinding(
            "Maximum number of iterations",
            "spline_max_iterations",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Residual value for convergence",
            "spline_convergence",
            parse_cellprofiler_float,
        ),
    )

    @classmethod
    def active_artifact_bindings(
        cls,
        module: ModuleBlock | None = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        retain_average = parse_cellprofiler_bool(
            optional_setting_value(module, cls.retain_average_setting) or "No"
        )
        retain_dilated = parse_cellprofiler_bool(
            optional_setting_value(module, cls.retain_dilated_setting) or "No"
        )
        return tuple(
            binding
            for binding in bindings
            if retain_average or binding is not cls.average_image_binding
            if retain_dilated or binding is not cls.dilated_image_binding
        )

    @classmethod
    def artifact_output_relations(
        cls,
        module: ModuleBlock,
        *,
        invocation_key,
        step_context,
        binding,
        name,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ):
        del invocation_key, binding, name
        del step_context, output_position
        image_inputs = artifact_inputs.for_artifact_type(ImageArtifactType).specs
        if not image_inputs:
            raise ValueError(
                f"Module {module.name}({module.module_num}) requires an image input."
            )
        scope = coerce_cellprofiler_enum(
            CalculationScope,
            required_setting_value(module, cls.calculation_scope_setting),
        )
        relation_type = (
            SourceStackLineageSourceRelation
            if scope is CalculationScope.EACH
            else GroupLineageSourceRelation
        )
        return (relation_type(source=image_inputs[0].ref()),)


NDIMAGE_CONSTANT_MODE = "constant"
ROBUST_FACTOR = 0.02
CORRECT_ILLUMINATION_CALCULATE_NAME = "correct_illumination_calculate"
logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)


def illumination_gaussian_filter(
    pixel_data: np.ndarray, mask: np.ndarray | None, sigma: float
) -> np.ndarray:
    """Apply CellProfiler Gaussian filtering with an implicit all-valid mask."""
    from scipy.ndimage import gaussian_filter

    return MaskedLinearFilterRequest(
        pixels=pixel_data,
        mask=mask,
        operation=lambda image: gaussian_filter(
            image, sigma, mode=NDIMAGE_CONSTANT_MODE, cval=0
        ),
    ).apply()


IlluminationCalculationResult = RuntimeArrayData | AlignedImageStack


class IlluminationCorrectionStrategy(
    EnumKeyedStrategyMixin[IlluminationCorrectionMethod],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal correction implementation for one CellProfiler method."""

    __enum_member_attr__ = "method"
    method: ClassVar[IlluminationCorrectionMethod]

    @abstractmethod
    def apply(
        self, image_pixels: np.ndarray, illumination_function: np.ndarray
    ) -> np.ndarray:
        """Apply the correction method."""


class DivideIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.DIVIDE

    def apply(
        self, image_pixels: np.ndarray, illumination_function: np.ndarray
    ) -> np.ndarray:
        output_dtype = np.result_type(image_pixels, illumination_function, 1e-10)
        output = np.empty(image_pixels.shape, dtype=output_dtype)
        nonzero = illumination_function != 0
        np.divide(image_pixels, illumination_function, out=output, where=nonzero)
        if not np.all(nonzero):
            np.divide(
                image_pixels, output_dtype.type(1e-10), out=output, where=~nonzero
            )
        return output


class SubtractIlluminationCorrectionStrategy(IlluminationCorrectionStrategy):
    method = IlluminationCorrectionMethod.SUBTRACT

    def apply(
        self, image_pixels: np.ndarray, illumination_function: np.ndarray
    ) -> np.ndarray:
        output = np.empty(
            image_pixels.shape,
            dtype=np.result_type(image_pixels, illumination_function, 0.0),
        )
        np.subtract(image_pixels, illumination_function, out=output)
        return output


@dataclass(slots=True)
class IlluminationCalculationRequest:
    """Complete semantic request for CorrectIlluminationCalculate."""

    image_data: np.ndarray
    mask: np.ndarray | None
    intensity_choice: IntensityChoice
    dilate_objects: bool
    object_dilation_radius: int
    block_size: int
    rescale_option: RescaleOption
    smoothing_method: SmoothingMethod
    filter_size_method: FilterSizeMethod
    object_width: int
    manual_filter_size: int
    automatic_splines: bool
    spline_bg_mode: SplineBgMode
    spline_points: int
    spline_threshold: float
    spline_rescale: float
    spline_max_iterations: int
    spline_convergence: float
    calculation_scope: CalculationScope
    morphology: MorphologyBackendStrategy
    convex_hull_backend_provider: CellProfilerBackendProvider | None
    rank_median_backend_provider: CellProfilerBackendProvider | None
    retain_average: bool = False
    retain_dilated: bool = False
    slice_index: int = 0
    image_metadata: ImagePayloadMetadata = field(default_factory=ImagePayloadMetadata)

    def __post_init__(self) -> None:
        if self.mask is None:
            return
        self.mask = project_image_mask_to_data_domain(
            self.mask,
            self.image_data,
            metadata=self.image_metadata,
        )

    @property
    def spatial_image_shape(self) -> tuple[int, ...]:
        if self.calculation_scope.uses_all_images:
            return tuple(self.image_data.shape[1:])
        return tuple(self.image_data.shape)

    def mask_for_stack_slice(self, slice_index: int) -> np.ndarray | None:
        if self.mask is None:
            return None
        if not self.calculation_scope.uses_all_images:
            raise ValueError(
                "Illumination stack-mask projection requires an all-images "
                "calculation scope."
            )
        return np.asarray(self.mask[slice_index], dtype=bool)

    def mask_for_output(self, illumination: np.ndarray) -> np.ndarray | None:
        if self.mask is None:
            return None
        output_mask = (
            np.any(self.mask, axis=0)
            if self.calculation_scope.uses_all_images
            else self.mask
        )
        return project_image_mask_to_data_domain(output_mask, illumination)

    def for_stack_slice(self, slice_index: int) -> "IlluminationCalculationRequest":
        return replace(
            self,
            image_data=np.asarray(self.image_data[slice_index]),
            mask=self.mask_for_stack_slice(slice_index),
            calculation_scope=CalculationScope.EACH,
            slice_index=slice_index,
        )

    def calculate(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_started_at = time.perf_counter()
        filter_size = self.filter_size()
        avg_image = self.average_image()
        dilated_image = self.apply_dilation(avg_image)
        smoothed_image = self.smooth(dilated_image, filter_size)
        output_image = self.apply_scaling(smoothed_image)
        runtime_profiler.log(
            "cic_total",
            time.perf_counter() - total_started_at,
            function=CORRECT_ILLUMINATION_CALCULATE_NAME,
        )
        return (output_image, avg_image, dilated_image)

    def filter_size(self) -> float:
        phase_started_at = time.perf_counter()
        filter_size = SmoothingFilterSizeStrategy.for_enum_member(
            self.filter_size_method
        ).calculate(self)
        runtime_profiler.log(
            "cic_filter_size",
            time.perf_counter() - phase_started_at,
            function=CORRECT_ILLUMINATION_CALCULATE_NAME,
            method=self.filter_size_method.value,
            smoothing=self.smoothing_method.value,
        )
        return filter_size

    def average_image(self) -> np.ndarray:
        phase_started_at = time.perf_counter()
        if not self.calculation_scope.uses_all_images:
            average_image = self.preprocess_for_averaging()
        else:
            if self.image_data.shape[0] == 0:
                raise ValueError(
                    "All-image illumination calculation requires at least one image."
                )
            averaged_inputs = [
                self.for_stack_slice(slice_index).preprocess_for_averaging()
                for slice_index in range(self.image_data.shape[0])
            ]
            average_image = np.mean(np.stack(averaged_inputs, axis=0), axis=0)
        runtime_profiler.log(
            "cic_average_image",
            time.perf_counter() - phase_started_at,
            function=CORRECT_ILLUMINATION_CALCULATE_NAME,
            method=self.intensity_choice.value,
            scope=self.calculation_scope.value,
        )
        return average_image

    def preprocess_for_averaging(self) -> np.ndarray:
        if (
            self.intensity_choice == IntensityChoice.REGULAR
            or self.smoothing_method == SmoothingMethod.SPLINES
        ):
            result = self.image_data.copy()
            if self.mask is not None:
                result[~self.mask] = 0
            return result
        return self.morphology.blockwise_minimum(
            self.image_data, self.mask, self.block_size
        )

    def apply_dilation(self, pixel_data: np.ndarray) -> np.ndarray:
        phase_started_at = time.perf_counter()
        if not self.dilate_objects:
            result = pixel_data
        else:
            result = illumination_gaussian_filter(
                pixel_data, self.mask, self.object_dilation_radius
            )
            if self.mask is not None:
                result[~self.mask] = 0
        runtime_profiler.log(
            "cic_dilation",
            time.perf_counter() - phase_started_at,
            function=CORRECT_ILLUMINATION_CALCULATE_NAME,
            enabled=self.dilate_objects,
        )
        return result

    def smooth(self, pixel_data: np.ndarray, filter_size: float) -> np.ndarray:
        phase_started_at = time.perf_counter()
        smoothed_image = SmoothingPlaneStrategy.for_enum_member(
            self.smoothing_method
        ).smooth(self, pixel_data, filter_size)
        runtime_profiler.log(
            "cic_smoothing",
            time.perf_counter() - phase_started_at,
            function=CORRECT_ILLUMINATION_CALCULATE_NAME,
            method=self.smoothing_method.value,
        )
        return smoothed_image

    def apply_scaling(self, pixel_data: np.ndarray) -> np.ndarray:
        phase_started_at = time.perf_counter()
        if self.rescale_option == RescaleOption.NO:
            result = pixel_data
        else:
            projected_mask = project_image_mask_to_data_domain(self.mask, pixel_data)
            if projected_mask is not None:
                sorted_data = pixel_data[(pixel_data > 0) & projected_mask]
            else:
                sorted_data = pixel_data[pixel_data > 0]
            if sorted_data.size == 0:
                result = pixel_data
            elif self.rescale_option == RescaleOption.YES:
                idx = int(len(sorted_data) * ROBUST_FACTOR)
                robust_minimum = np.partition(sorted_data, idx)[idx]
                result = pixel_data.copy()
                result[result < robust_minimum] = robust_minimum
                if robust_minimum != 0:
                    result = result / robust_minimum
            else:
                idx = len(sorted_data) // 2
                robust_minimum = np.partition(sorted_data, idx)[idx]
                result = pixel_data.copy()
                if robust_minimum != 0:
                    result = result / robust_minimum
        runtime_profiler.log(
            "cic_scaling",
            time.perf_counter() - phase_started_at,
            function=CORRECT_ILLUMINATION_CALCULATE_NAME,
            method=self.rescale_option.value,
        )
        return result

    def calculate_payload(
        self, metadata: ImagePayloadMetadata
    ) -> IlluminationCalculationResult:
        illumination, average, dilated = self.calculate()
        output_metadata = (
            metadata.collapse_leading_plane_axis()
            if self.calculation_scope.uses_all_images
            and metadata.plane_axis is not None
            else metadata
        )
        return self.payload_result(
            illumination,
            average,
            dilated,
            metadata=output_metadata,
        )

    def payload_result(
        self,
        illumination: np.ndarray,
        average: np.ndarray,
        dilated: np.ndarray,
        *,
        metadata: ImagePayloadMetadata,
    ) -> IlluminationCalculationResult:
        image_outputs = [self.image_payload(illumination, metadata)]
        if self.retain_average:
            image_outputs.append(self.image_payload(average, metadata))
        if self.retain_dilated:
            image_outputs.append(self.image_payload(dilated, metadata))
        main_output = (
            image_outputs[0]
            if len(image_outputs) == 1
            else AlignedImageStack(tuple(image_outputs))
        )
        return main_output

    def image_payload(
        self,
        image: np.ndarray,
        metadata: ImagePayloadMetadata,
    ) -> RuntimeArrayData:
        return metadata.payload_with(image, self.mask_for_output(image))


@numpy(contract=ProcessingContract.FLEXIBLE)
def correct_illumination_calculate(
    image: np.ndarray,
    intensity_choice: IntensityChoice | str = IntensityChoice.REGULAR,
    dilate_objects: bool = False,
    object_dilation_radius: int = 1,
    block_size: int = 60,
    rescale_option: RescaleOption | str = RescaleOption.YES,
    smoothing_method: SmoothingMethod | str = SmoothingMethod.FIT_POLYNOMIAL,
    filter_size_method: FilterSizeMethod | str = FilterSizeMethod.AUTOMATIC,
    object_width: int = 10,
    manual_filter_size: int = 10,
    automatic_splines: bool = True,
    spline_bg_mode: SplineBgMode | str = SplineBgMode.AUTO,
    spline_points: int = 5,
    spline_threshold: float = 2.0,
    spline_rescale: float = 2.0,
    spline_max_iterations: int = 40,
    spline_convergence: float = 0.001,
    calculation_scope: CalculationScope | str = CalculationScope.EACH,
    retain_average: bool = False,
    retain_dilated: bool = False,
    convex_hull_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    rank_median_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> IlluminationCalculationResult:
    """Calculate an illumination correction function."""
    intensity_choice = coerce_cellprofiler_enum(IntensityChoice, intensity_choice)
    rescale_option = coerce_cellprofiler_enum(RescaleOption, rescale_option)
    smoothing_method = coerce_cellprofiler_enum(SmoothingMethod, smoothing_method)
    filter_size_method = coerce_cellprofiler_enum(FilterSizeMethod, filter_size_method)
    spline_bg_mode = coerce_cellprofiler_enum(SplineBgMode, spline_bg_mode)
    calculation_scope = coerce_cellprofiler_enum(CalculationScope, calculation_scope)
    morphology = MorphologyBackendStrategy.for_callable(correct_illumination_calculate)
    pixel_data = np.asarray(image_payload_data(image))
    raw_mask = image_payload_mask(image)
    metadata = image_payload_metadata(image).without_unit_interval_intensity_scale()
    request = IlluminationCalculationRequest(
        image_data=pixel_data,
        mask=None if raw_mask is None else np.asarray(raw_mask, dtype=bool),
        intensity_choice=intensity_choice,
        dilate_objects=dilate_objects,
        object_dilation_radius=object_dilation_radius,
        block_size=block_size,
        rescale_option=rescale_option,
        smoothing_method=smoothing_method,
        filter_size_method=filter_size_method,
        object_width=object_width,
        manual_filter_size=manual_filter_size,
        automatic_splines=automatic_splines,
        spline_bg_mode=spline_bg_mode,
        spline_points=spline_points,
        spline_threshold=spline_threshold,
        spline_rescale=spline_rescale,
        spline_max_iterations=spline_max_iterations,
        spline_convergence=spline_convergence,
        calculation_scope=calculation_scope,
        retain_average=retain_average,
        retain_dilated=retain_dilated,
        morphology=morphology,
        convex_hull_backend_provider=convex_hull_backend_provider,
        rank_median_backend_provider=rank_median_backend_provider,
        image_metadata=metadata,
    )
    return request.calculate_payload(metadata)


@processing_prepare(correct_illumination_calculate)
def _prepare_correct_illumination_calculate() -> None:
    """Compile common illumination kernels outside measured step execution."""
    RankMedianSmoothingBackendStrategy.prepare_registered_family()
    ConvexHullSmoothingBackendStrategy.prepare_registered_family()
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    correct_illumination_calculate.__wrapped__(
        image,
        smoothing_method=SmoothingMethod.FIT_POLYNOMIAL,
        filter_size_method=FilterSizeMethod.AUTOMATIC,
        rescale_option=RescaleOption.YES,
    )
    convex_hull_image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(
        (32, 32)
    )
    correct_illumination_calculate.__wrapped__(
        convex_hull_image,
        smoothing_method=SmoothingMethod.CONVEX_HULL,
        filter_size_method=FilterSizeMethod.AUTOMATIC,
        rescale_option=RescaleOption.NO,
    )
    background = np.zeros((64, 64), dtype=np.float32)
    background[::8, ::8] = 1.0
    correct_illumination_calculate.__wrapped__(
        background,
        intensity_choice=IntensityChoice.BACKGROUND,
        block_size=8,
        smoothing_method=SmoothingMethod.MEDIAN_FILTER,
        filter_size_method=FilterSizeMethod.MANUALLY,
        manual_filter_size=32,
        rescale_option=RescaleOption.NO,
    )
    nonconstant_background = np.linspace(0.0, 1.0, 96 * 96, dtype=np.float32).reshape(
        (96, 96)
    )
    correct_illumination_calculate.__wrapped__(
        nonconstant_background,
        intensity_choice=IntensityChoice.REGULAR,
        smoothing_method=SmoothingMethod.MEDIAN_FILTER,
        filter_size_method=FilterSizeMethod.MANUALLY,
        manual_filter_size=96,
        rescale_option=RescaleOption.NO,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("illumination_function")
def correct_illumination_apply(
    image: RuntimeArrayData,
    *,
    illumination_function: RuntimeArrayData,
    method: IlluminationCorrectionMethod | str = IlluminationCorrectionMethod.DIVIDE,
    truncate_low: bool = True,
    truncate_high: bool = True,
) -> RuntimeArrayData:
    """Apply one illumination artifact to one runtime image slice.

    Args:
        illumination_function: Correction-function image with the same shape as
            the input, divided into or subtracted from it according to ``method``.
    """

    image_pixels = np.asarray(image_payload_data(image))
    illumination_pixels = np.asarray(image_payload_data(illumination_function))
    if image_pixels.shape != illumination_pixels.shape:
        raise ValueError(
            f"Input image shape {image_pixels.shape} and illumination function "
            f"shape {illumination_pixels.shape} must be equal."
        )
    resolved_method = coerce_cellprofiler_enum(IlluminationCorrectionMethod, method)
    output_pixels = IlluminationCorrectionStrategy.for_enum_member(
        resolved_method
    ).apply(image_pixels, illumination_pixels)
    if truncate_low:
        np.maximum(output_pixels, 0.0, out=output_pixels)
    if truncate_high:
        np.minimum(output_pixels, 1.0, out=output_pixels)
    mask = image_payload_mask(image)
    return (
        image_payload_metadata(image)
        .without_unit_interval_intensity_scale()
        .payload_with(
            output_pixels, None if mask is None else np.asarray(mask, dtype=bool)
        )
    )


class SmoothingFilterSizeStrategy(
    EnumKeyedStrategyMixin[FilterSizeMethod], ABC, metaclass=AutoRegisterMeta
):
    """Nominal filter-size derivation for one closed CellProfiler mode."""

    __enum_member_attr__ = "method"
    method: ClassVar[FilterSizeMethod | None] = None

    @abstractmethod
    def calculate(self, request: IlluminationCalculationRequest) -> float:
        """Return the smoothing filter size."""


class ManualSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.MANUALLY

    def calculate(self, request: IlluminationCalculationRequest) -> float:
        return float(request.manual_filter_size)


class ObjectWidthSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.OBJECT_SIZE

    def calculate(self, request: IlluminationCalculationRequest) -> float:
        return request.object_width * 2.35 / 3.5


class AutomaticSmoothingFilterSizeStrategy(SmoothingFilterSizeStrategy):
    method = FilterSizeMethod.AUTOMATIC

    def calculate(self, request: IlluminationCalculationRequest) -> float:
        return min(30.0, float(np.max(request.spatial_image_shape)) / 40.0)


class SmoothingPlaneStrategy(
    EnumKeyedStrategyMixin[SmoothingMethod], ABC, metaclass=AutoRegisterMeta
):
    """Nominal smoothing implementation for one closed CellProfiler mode."""

    __enum_member_attr__ = "method"
    method: ClassVar[SmoothingMethod | None] = None

    @abstractmethod
    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        """Return the smoothed illumination plane."""


class NoSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.NONE

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        del request, filter_size
        return pixel_data


class FitPolynomialSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.FIT_POLYNOMIAL

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        del filter_size
        return fit_polynomial_surface(pixel_data, request.mask)


class GaussianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.GAUSSIAN_FILTER

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        return illumination_gaussian_filter(
            pixel_data, request.mask, filter_size / 2.35
        )


class MedianFilterSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.MEDIAN_FILTER

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        filter_sigma = max(1, int(filter_size / 2.35 + 0.5))
        return RankMedianSmoothingBackendStrategy.for_memory_type(
            backend_provider=request.rank_median_backend_provider
        ).smooth_background_plane(
            pixel_data,
            mask=request.mask,
            radius=filter_sigma,
            morphology=request.morphology,
        )


class AverageSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.TO_AVERAGE

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        del filter_size
        if request.mask is not None:
            mean_val = np.mean(pixel_data[request.mask])
        else:
            mean_val = np.mean(pixel_data)
        return np.full(pixel_data.shape, mean_val, dtype=pixel_data.dtype)


class ConvexHullSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.CONVEX_HULL

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        return ConvexHullSmoothingBackendStrategy.for_memory_type(
            backend_provider=request.convex_hull_backend_provider
        ).smooth_background_plane(
            pixel_data,
            mask=request.mask,
            filter_size=filter_size,
            morphology=request.morphology,
        )


class SplinesSmoothingPlaneStrategy(SmoothingPlaneStrategy):
    method = SmoothingMethod.SPLINES

    def smooth(
        self,
        request: IlluminationCalculationRequest,
        pixel_data: np.ndarray,
        filter_size: float,
    ) -> np.ndarray:
        del filter_size
        from scipy.interpolate import RectBivariateSpline

        h, w = pixel_data.shape
        if request.automatic_splines:
            shortest_side = min(h, w)
            scale = max(1, shortest_side // 200)
            n_points = 5
        else:
            scale = int(request.spline_rescale)
            n_points = request.spline_points
        downsampled = pixel_data[::scale, ::scale]
        dh, dw = downsampled.shape
        y_points = np.linspace(0, dh - 1, n_points)
        x_points = np.linspace(0, dw - 1, n_points)
        yi = np.clip(np.round(y_points).astype(int), 0, dh - 1)
        xi = np.clip(np.round(x_points).astype(int), 0, dw - 1)
        spline = RectBivariateSpline(
            y_points, x_points, downsampled[np.ix_(yi, xi)], kx=3, ky=3
        )
        result = spline(np.linspace(0, dh - 1, h), np.linspace(0, dw - 1, w))
        if request.mask is not None:
            result[request.mask] -= np.mean(result[request.mask])
        else:
            result -= np.mean(result)
        return result


def fit_polynomial_surface(
    pixel_data: np.ndarray, mask: np.ndarray | None
) -> np.ndarray:
    """Fit CP's quadratic illumination surface without dense design matrices."""
    image = np.ascontiguousarray(pixel_data, dtype=np.float64)
    if image.ndim != 2:
        raise NotImplementedError(
            f"Fit-polynomial illumination smoothing currently supports 2-D NumPy planes, got shape {image.shape!r}."
        )
    mask_array = (
        np.empty((0, 0), dtype=np.bool_)
        if mask is None
        else np.ascontiguousarray(mask, dtype=np.bool_)
    )
    if mask is not None and mask_array.shape != image.shape:
        raise ValueError(
            f"Fit-polynomial illumination mask must match the image shape; got mask {mask_array.shape!r} for image {image.shape!r}."
        )
    if mask is None:
        gram = fit_polynomial_unmasked_gram(image.shape[0], image.shape[1])
        rhs = _fit_polynomial_unmasked_rhs_numba(image)
    else:
        gram, rhs = _fit_polynomial_normal_equations_numba(image, mask_array, True)
    coeffs = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return _evaluate_polynomial_surface_numba(
        image.shape[0], image.shape[1], np.ascontiguousarray(coeffs, dtype=np.float64)
    )


@lru_cache(maxsize=16)
def fit_polynomial_unmasked_gram(height: int, width: int) -> np.ndarray:
    return _fit_polynomial_unmasked_gram_numba(int(height), int(width))


@njit(cache=True)
def _fit_polynomial_unmasked_gram_numba(height: int, width: int) -> np.ndarray:
    gram = np.zeros((6, 6), dtype=np.float64)
    features = np.empty(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            features[0] = x_value * x_value
            features[1] = y2
            features[2] = x_value * y_value
            features[3] = x_value
            features[4] = y_value
            features[5] = 1.0
            for i in range(6):
                for j in range(6):
                    gram[i, j] += features[i] * features[j]
    return gram


@njit(cache=True)
def _fit_polynomial_unmasked_rhs_numba(pixel_data: np.ndarray) -> np.ndarray:
    height, width = pixel_data.shape
    rhs = np.zeros(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            value = pixel_data[row, col]
            rhs[0] += x_value * x_value * value
            rhs[1] += y2 * value
            rhs[2] += x_value * y_value * value
            rhs[3] += x_value * value
            rhs[4] += y_value * value
            rhs[5] += value
    return rhs


@njit(cache=True)
def _fit_polynomial_normal_equations_numba(
    pixel_data: np.ndarray, mask: np.ndarray, has_mask: bool
) -> tuple[np.ndarray, np.ndarray]:
    height, width = pixel_data.shape
    gram = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    features = np.empty(6, dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            if has_mask and (not mask[row, col]):
                continue
            x_value = col / width - 0.5
            features[0] = x_value * x_value
            features[1] = y2
            features[2] = x_value * y_value
            features[3] = x_value
            features[4] = y_value
            features[5] = 1.0
            value = pixel_data[row, col]
            for i in range(6):
                rhs[i] += features[i] * value
                for j in range(6):
                    gram[i, j] += features[i] * features[j]
    return (gram, rhs)


@njit(cache=True)
def _evaluate_polynomial_surface_numba(
    height: int, width: int, coeffs: np.ndarray
) -> np.ndarray:
    output = np.empty((height, width), dtype=np.float64)
    for row in range(height):
        y_value = row / height - 0.5
        y2 = y_value * y_value
        for col in range(width):
            x_value = col / width - 0.5
            output[row, col] = (
                coeffs[0] * x_value * x_value
                + coeffs[1] * y2
                + coeffs[2] * x_value * y_value
                + coeffs[3] * x_value
                + coeffs[4] * y_value
                + coeffs[5]
            )
    return output


class ConvexHullSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Convex-hull illumination smoothing keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        """Return a smoothed illumination background plane."""


class RankMedianSmoothingBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Rank-median illumination smoothing keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @staticmethod
    def disk_rows(footprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Collapse a dense disk footprint into per-row horizontal radii."""
        center_y = footprint.shape[0] // 2
        center_x = footprint.shape[1] // 2
        rows: list[int] = []
        radii: list[int] = []
        for y in range(footprint.shape[0]):
            xs = np.flatnonzero(footprint[y])
            if xs.size == 0:
                continue
            rows.append(y - center_y)
            radii.append(int(np.max(np.abs(xs - center_x))))
        return (np.asarray(rows, dtype=np.int64), np.asarray(radii, dtype=np.int64))

    @abstractmethod
    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        """Return a rank-median smoothed illumination background plane."""


@dataclass(frozen=True, slots=True)
class RankMedianProfilerPhase:
    """Single authority for rank-median profiler event construction."""

    radius: int
    started_at: float

    @classmethod
    def start(cls, radius: int) -> "RankMedianProfilerPhase":
        return cls(radius=radius, started_at=time.perf_counter())

    def log(self, event_name: str, **fields: object) -> None:
        runtime_profiler.log(
            event_name,
            time.perf_counter() - self.started_at,
            radius=self.radius,
            **fields,
        )


class NumbaNumpyRankMedianSmoothingBackendStrategy(RankMedianSmoothingBackendStrategy):
    """NumPy-memory rank median matching skimage rank median border semantics."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        """Compile rank-median numba kernels during compiler preparation."""
        footprint = np.ones((3, 3), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        scaled = np.arange(16, dtype=np.uint16).reshape((4, 4))
        code_image = np.arange(16, dtype=np.int32).reshape((4, 4))
        mask = np.ones(scaled.shape, dtype=np.bool_)
        _rank_median_global_minimum_is_majority_everywhere_numba(
            scaled, row_offsets_y, row_radii_x, np.uint16(0)
        )
        _rank_median_codes_2d_sliding_histogram_numba(
            code_image, row_offsets_y, row_radii_x, int(code_image.size)
        )
        _rank_median_uint16_2d_sliding_histogram_numba(
            scaled, mask, row_offsets_y, row_radii_x
        )

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                f"Rank-median illumination smoothing currently supports 2-D NumPy planes, got shape {image.shape!r}."
            )
        mask = project_image_mask_to_data_domain(mask, image)
        if mask is not None and np.asarray(mask).shape != image.shape:
            raise ValueError(
                f"Rank-median illumination mask must match the image shape; got mask {np.asarray(mask).shape!r} for image {image.shape!r}."
            )
        footprint = np.asarray(morphology.disk_footprint(radius), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        scaled = (image * 65535.0).astype(np.uint16)
        mask_array = (
            np.ones(image.shape, dtype=np.bool_)
            if mask is None
            else np.asarray(mask, dtype=np.bool_)
        )
        effective_scaled = scaled.copy()
        effective_scaled[~mask_array] = np.uint16(0)
        minimum_value = np.min(effective_scaled)
        phase = RankMedianProfilerPhase.start(radius)
        if np.all(effective_scaled == minimum_value):
            phase.log("rank_median_constant_minimum")
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        phase.log("rank_median_constant_minimum")
        phase = RankMedianProfilerPhase.start(radius)
        if _rank_median_global_minimum_is_majority_everywhere_numba(
            np.ascontiguousarray(effective_scaled),
            row_offsets_y,
            row_radii_x,
            minimum_value,
        ):
            phase.log("rank_median_minimum_majority", result=True)
            return np.full(image.shape, minimum_value, dtype=np.float32) / 65535.0
        phase.log("rank_median_minimum_majority", result=False)
        phase = RankMedianProfilerPhase.start(radius)
        values, inverse = np.unique(effective_scaled, return_inverse=True)
        phase.log("rank_median_unique_codes", value_count=int(values.size))
        phase = RankMedianProfilerPhase.start(radius)
        code_image = inverse.reshape(image.shape).astype(np.int32, copy=False)
        result_codes = _rank_median_codes_2d_sliding_histogram_numba(
            np.ascontiguousarray(code_image),
            row_offsets_y,
            row_radii_x,
            int(values.size),
        )
        phase.log("rank_median_numba_codes", value_count=int(values.size))
        result = values[result_codes]
        return result.astype(np.float32) / 65535.0


class NativeNumpyRankMedianSmoothingBackendStrategy(RankMedianSmoothingBackendStrategy):
    """Compact-domain skimage rank-median backend for NumPy planes."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def prepare_backend(self) -> None:
        """Compile exact compact-domain rank-median kernels during preparation."""
        code_image = np.arange(16, dtype=np.uint8).reshape((4, 4))
        footprint = np.ones((3, 3), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        _rank_median_small_codes_2d_sliding_histogram_numba(
            code_image, row_offsets_y, row_radii_x, int(code_image.size)
        )
        _rank_median_small_codes_minimum_majority_hybrid_numba(
            code_image, row_offsets_y, row_radii_x, int(code_image.size)
        )
        exact_mask = _rank_median_zero_exact_mask_fft(
            code_image, footprint, row_offsets_y, row_radii_x
        )
        _rank_median_small_codes_exact_mask_runs_numba(
            code_image, exact_mask, row_offsets_y, row_radii_x, int(code_image.size)
        )

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        radius: int,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        image = np.asarray(pixel_data, dtype=np.float32)
        projected_mask = project_image_mask_to_data_domain(mask, image)
        mask_array = (
            None
            if projected_mask is None
            else np.asarray(projected_mask, dtype=np.bool_)
        )
        if mask_array is not None and mask_array.shape != image.shape:
            raise ValueError(
                f"Rank-median illumination mask must match the image shape; got mask {mask_array.shape!r} for image {image.shape!r}."
            )
        footprint = np.asarray(morphology.disk_footprint(radius), dtype=np.bool_)
        row_offsets_y, row_radii_x = self.disk_rows(footprint)
        scaled = (image * 65535.0).astype(np.uint16)
        effective_scaled = scaled if mask_array is None else scaled.copy()
        if mask_array is not None:
            effective_scaled[~mask_array] = np.uint16(0)
        return self._smooth_compact_rank_median(effective_scaled, footprint)

    @staticmethod
    def _smooth_compact_rank_median(
        scaled: np.ndarray, footprint: np.ndarray
    ) -> np.ndarray:
        effective_scaled = np.asarray(scaled, dtype=np.uint16)
        if capture_enabled():
            capture_array_fixture(
                "rank_median_compact_domain",
                scaled=effective_scaled,
                footprint=np.asarray(footprint, dtype=np.bool_),
            )
        values, inverse, counts = np.unique(
            effective_scaled, return_inverse=True, return_counts=True
        )
        profile_started_at = time.perf_counter()
        if values.size == 1:
            runtime_profiler.log(
                "rank_median_compact_domain",
                time.perf_counter() - profile_started_at,
                value_count=int(values.size),
                branch="constant",
                minimum_count=int(counts[0]),
                pixel_count=int(effective_scaled.size),
            )
            return (
                np.full(effective_scaled.shape, values[0], dtype=np.float32) / 65535.0
            )
        code_dtype = (
            np.uint8 if values.size <= np.iinfo(np.uint8).max + 1 else np.uint16
        )
        code_image = inverse.reshape(effective_scaled.shape).astype(code_dtype)
        if code_dtype == np.uint8:
            row_offsets_y, row_radii_x = RankMedianSmoothingBackendStrategy.disk_rows(
                footprint
            )
            contiguous_codes = np.ascontiguousarray(code_image)
            if counts[0] > effective_scaled.size // 2:
                use_fft_zero_mask = _rank_median_fft_zero_mask_candidate(
                    contiguous_codes, footprint
                )
                if use_fft_zero_mask:
                    exact_mask = _rank_median_zero_exact_mask_fft(
                        contiguous_codes,
                        np.asarray(footprint, dtype=np.bool_),
                        row_offsets_y,
                        row_radii_x,
                    )
                    result_codes = _rank_median_small_codes_exact_mask_runs_numba(
                        contiguous_codes,
                        exact_mask,
                        row_offsets_y,
                        row_radii_x,
                        int(values.size),
                    )
                else:
                    result_codes = (
                        _rank_median_small_codes_minimum_majority_hybrid_numba(
                            contiguous_codes,
                            row_offsets_y,
                            row_radii_x,
                            int(values.size),
                        )
                    )
                runtime_profiler.log(
                    "rank_median_compact_domain",
                    time.perf_counter() - profile_started_at,
                    value_count=int(values.size),
                    branch=(
                        "minimum_majority_fft_runs"
                        if use_fft_zero_mask
                        else "minimum_majority_hybrid"
                    ),
                    minimum_count=int(counts[0]),
                    pixel_count=int(effective_scaled.size),
                )
            else:
                result_codes = _rank_median_small_codes_2d_sliding_histogram_numba(
                    contiguous_codes,
                    row_offsets_y,
                    row_radii_x,
                    int(values.size),
                )
                runtime_profiler.log(
                    "rank_median_compact_domain",
                    time.perf_counter() - profile_started_at,
                    value_count=int(values.size),
                    branch="small_sliding_histogram",
                    minimum_count=int(counts[0]),
                    pixel_count=int(effective_scaled.size),
                )
            return values[result_codes].astype(np.float32) / 65535.0
        import skimage.filters

        result_codes = skimage.filters.median(code_image, footprint, behavior="rank")
        runtime_profiler.log(
            "rank_median_compact_domain",
            time.perf_counter() - profile_started_at,
            value_count=int(values.size),
            branch="skimage_rank",
            minimum_count=int(counts[0]),
            pixel_count=int(effective_scaled.size),
        )
        return values[result_codes].astype(np.float32) / 65535.0


class CentrosomeNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy
):
    """CellProfiler/centrosome reference convex-hull smoothing for NumPy planes."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.CENTROSOME
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = True

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        del filter_size, morphology
        import centrosome.cpmorphology
        import centrosome.filter

        image = np.asarray(pixel_data)
        mask_array = None if mask is None else np.asarray(mask, dtype=bool)
        eroded = centrosome.cpmorphology.grey_erosion(image, 2, mask_array)
        transformed = centrosome.filter.convex_hull_transform(eroded, mask=mask_array)
        return np.asarray(
            centrosome.cpmorphology.grey_dilation(transformed, 2, mask_array),
            dtype=image.dtype,
        )


class LegacyFastNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy
):
    """Fast CP3-compatible convex-hull smoothing for NumPy planes."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.LEGACY_FAST
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.LEGACY_FAST
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        del morphology
        from scipy.ndimage import grey_dilation, grey_erosion, maximum_filter

        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                f"Legacy-fast convex-hull smoothing currently supports 2-D NumPy planes, got shape {image.shape!r}."
            )
        result = grey_dilation(
            maximum_filter(grey_erosion(image, size=3), size=max(1, int(filter_size))),
            size=3,
        )
        if mask is not None:
            result = np.asarray(result, dtype=np.float32)
            result[~np.asarray(mask, dtype=bool)] = 0
        return result.astype(np.float32, copy=False)


class ExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy
):
    """Numba-accelerated level-set convex-hull reconstruction."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        """Compile exact convex-hull smoothing kernels during compiler preparation."""
        image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
        mask = np.ones(image.shape, dtype=np.bool_)
        morphology = MorphologyBackendStrategy.for_memory_type()
        self.smooth_background_plane(
            image, mask=mask, filter_size=3, morphology=morphology
        )

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        del filter_size
        image = np.asarray(pixel_data, dtype=np.float32)
        if image.ndim != 2:
            raise NotImplementedError(
                f"Exact convex-hull smoothing currently supports 2-D NumPy planes, got shape {image.shape!r}."
            )
        valid_mask = (
            np.ones(image.shape, dtype=bool)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if valid_mask.shape != image.shape:
            raise ValueError(
                f"Convex-hull smoothing requires a mask matching the 2-D image plane, got mask {valid_mask.shape!r} for image {image.shape!r}."
            )
        if not np.any(valid_mask):
            return np.zeros(image.shape, dtype=np.float32)
        grey_morphology = CellProfilerMaskedGreyMorphology.for_convex_hull(morphology)
        eroded = grey_morphology.erode(image, valid_mask)
        valid_values = eroded[valid_mask]
        thresholds = np.linspace(
            float(np.min(valid_values)),
            float(np.max(valid_values)),
            256,
            dtype=np.float32,
        )[1:]
        hull = _exact_level_set_convex_hull_smoothing_numba(
            np.ascontiguousarray(eroded, dtype=np.float32),
            np.ascontiguousarray(valid_mask, dtype=np.bool_),
            np.ascontiguousarray(thresholds, dtype=np.float32),
        )
        return grey_morphology.dilate(hull, valid_mask)


class NativeExactLevelSetNumpyConvexHullSmoothingBackendStrategy(
    ConvexHullSmoothingBackendStrategy
):
    """Reference exact level-set convex-hull reconstruction for NumPy planes."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def smooth_background_plane(
        self,
        pixel_data: np.ndarray,
        *,
        mask: np.ndarray | None,
        filter_size: float,
        morphology: MorphologyBackendStrategy,
    ) -> np.ndarray:
        del filter_size
        return _native_exact_level_set_convex_hull_smoothing(
            np.asarray(pixel_data, dtype=np.float32),
            None if mask is None else np.asarray(mask, dtype=bool),
            morphology,
        )


def _native_exact_level_set_convex_hull_smoothing(
    image: np.ndarray, mask: np.ndarray | None, morphology: MorphologyBackendStrategy
) -> np.ndarray:
    if image.ndim != 2:
        raise NotImplementedError(
            f"Native exact convex-hull smoothing currently supports 2-D NumPy planes, got shape {image.shape!r}."
        )
    valid_mask = (
        np.ones(image.shape, dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if valid_mask.shape != image.shape:
        raise ValueError(
            f"Convex-hull smoothing requires a mask matching the 2-D image plane, got mask {valid_mask.shape!r} for image {image.shape!r}."
        )
    if not np.any(valid_mask):
        return np.zeros(image.shape, dtype=np.float32)
    grey_morphology = CellProfilerMaskedGreyMorphology.for_convex_hull(morphology)
    eroded = grey_morphology.erode(image, valid_mask)
    valid_values = eroded[valid_mask]
    minimum = float(np.min(valid_values))
    maximum = float(np.max(valid_values))
    output = np.full(image.shape, minimum, dtype=np.float32)
    output[~valid_mask] = 0
    if maximum <= minimum:
        return grey_morphology.dilate(output, valid_mask)
    for threshold in np.linspace(minimum, maximum, 256, dtype=np.float32)[1:]:
        level_mask = valid_mask & (eroded >= float(threshold))
        if not np.any(level_mask):
            continue
        output[morphology.convex_hull_image(level_mask) & valid_mask] = threshold
    return grey_morphology.dilate(output, valid_mask)


@dataclass(frozen=True, slots=True)
class CellProfilerMaskedGreyMorphology:
    """CellProfiler masked grey morphology semantics for convex-hull smoothing."""

    footprint: np.ndarray

    @classmethod
    def for_convex_hull(
        cls, morphology: MorphologyBackendStrategy
    ) -> "CellProfilerMaskedGreyMorphology":
        """Build CP's radius-2 disk morphology for convex-hull smoothing."""
        return cls(np.asarray(morphology.disk_footprint(2), dtype=bool))

    def erode(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Match centrosome.cpmorphology.grey_erosion masking semantics."""
        from scipy import ndimage as ndi

        radius = self._padding_radius()
        padded = np.ones(np.asarray(image.shape) + radius * 2, dtype=image.dtype)
        core = self._core_slice(image, radius)
        padded[core] = image
        padded_core = padded[core]
        padded_core[~mask] = 1
        eroded = ndi.grey_erosion(padded, footprint=self.footprint)[core]
        return self._restore_masked_pixels(eroded, image, mask)

    def dilate(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Match centrosome.cpmorphology.grey_dilation masking semantics."""
        from scipy import ndimage as ndi

        radius = self._padding_radius()
        padded = np.zeros(np.asarray(image.shape) + radius * 2, dtype=image.dtype)
        core = self._core_slice(image, radius)
        padded[core] = image
        padded_core = padded[core]
        padded_core[~mask] = 0
        dilated = ndi.grey_dilation(padded, footprint=self.footprint)[core]
        return self._restore_masked_pixels(dilated, image, mask)

    def _padding_radius(self) -> int:
        return max(1, int(np.ceil(np.max(np.asarray(self.footprint.shape)) / 2 - 0.5)))

    @staticmethod
    def _core_slice(image: np.ndarray, radius: int) -> tuple[slice, ...]:
        return tuple((slice(radius, -radius) for _axis in image.shape))

    @staticmethod
    def _restore_masked_pixels(
        morphed: np.ndarray, image: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        result = np.asarray(morphed, dtype=np.float32)
        result[~mask] = image[~mask]
        return result


@njit(cache=True)
def _exact_level_set_convex_hull_smoothing_numba(
    image: np.ndarray, valid_mask: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    minimum = np.float32(0.0)
    maximum = np.float32(0.0)
    found_valid = False
    valid_pixel_count = 0
    for y in range(height):
        for x in range(width):
            if not valid_mask[y, x]:
                continue
            valid_pixel_count += 1
            value = image[y, x]
            if not found_valid:
                minimum = value
                maximum = value
                found_valid = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
    output = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            output[y, x] = minimum if valid_mask[y, x] else np.float32(0.0)
    if not found_valid or maximum <= minimum:
        return output
    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_x = np.empty(point_capacity, dtype=np.int64)
    point_y = np.empty(point_capacity, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    bucket_counts = np.zeros(thresholds.size, dtype=np.int64)
    active_pixel_count = _count_convex_hull_threshold_buckets(
        image, valid_mask, thresholds, bucket_counts
    )
    if active_pixel_count == 0:
        return output
    bucket_offsets = np.empty(thresholds.size + 1, dtype=np.int64)
    offset = 0
    for bucket_index in range(thresholds.size):
        bucket_offsets[bucket_index] = offset
        offset += bucket_counts[bucket_index]
        bucket_counts[bucket_index] = 0
    bucket_offsets[thresholds.size] = offset
    bucket_rows = np.empty(active_pixel_count, dtype=np.int64)
    bucket_cols = np.empty(active_pixel_count, dtype=np.int64)
    _fill_convex_hull_threshold_buckets(
        image,
        valid_mask,
        thresholds,
        bucket_offsets,
        bucket_counts,
        bucket_rows,
        bucket_cols,
    )
    assigned = np.zeros((height, width), dtype=np.bool_)
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807
    assigned_count = 0
    for level_index in range(thresholds.size - 1, -1, -1):
        start = bucket_offsets[level_index]
        end = bucket_offsets[level_index] + bucket_counts[level_index]
        if start == end:
            continue
        changed_extrema = False
        for bucket_position in range(start, end):
            y = bucket_rows[bucket_position]
            x = bucket_cols[bucket_position]
            if _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y - 1, 2 * x):
                changed_extrema = True
            if _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y + 1, 2 * x):
                changed_extrema = True
            if _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x - 1):
                changed_extrema = True
            if _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x + 1):
                changed_extrema = True
        if not changed_extrema:
            continue
        point_count = _emit_diamond_extreme_points(
            min_col_by_row, max_col_by_row, point_x, point_y
        )
        if point_count == 0:
            continue
        hull_count = _monotone_chain_hull(point_x, point_y, point_count, hull_x, hull_y)
        assigned_count += _paint_convex_hull(
            output,
            assigned,
            False,
            valid_mask,
            thresholds[level_index],
            hull_x,
            hull_y,
            hull_count,
        )
        if assigned_count >= valid_pixel_count:
            break
    return output


@njit(cache=True)
def _count_convex_hull_threshold_buckets(
    image: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: np.ndarray,
    bucket_counts: np.ndarray,
) -> int:
    height, width = image.shape
    active_pixel_count = 0
    for y in range(height):
        for x in range(width):
            if not valid_mask[y, x]:
                continue
            bucket_index = _last_threshold_index_not_greater_than(
                thresholds, image[y, x]
            )
            if bucket_index < 0:
                continue
            bucket_counts[bucket_index] += 1
            active_pixel_count += 1
    return active_pixel_count


@njit(cache=True)
def _fill_convex_hull_threshold_buckets(
    image: np.ndarray,
    valid_mask: np.ndarray,
    thresholds: np.ndarray,
    bucket_offsets: np.ndarray,
    bucket_counts: np.ndarray,
    bucket_rows: np.ndarray,
    bucket_cols: np.ndarray,
) -> None:
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            if not valid_mask[y, x]:
                continue
            bucket_index = _last_threshold_index_not_greater_than(
                thresholds, image[y, x]
            )
            if bucket_index < 0:
                continue
            bucket_position = bucket_offsets[bucket_index] + bucket_counts[bucket_index]
            bucket_rows[bucket_position] = y
            bucket_cols[bucket_position] = x
            bucket_counts[bucket_index] += 1


@njit(cache=True)
def _last_threshold_index_not_greater_than(
    thresholds: np.ndarray, value: np.float32
) -> int:
    low = 0
    high = thresholds.size
    while low < high:
        middle = (low + high) // 2
        if thresholds[middle] <= value:
            low = middle + 1
        else:
            high = middle
    return low - 1


@njit(cache=True)
def _emit_diamond_extreme_points(
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
) -> int:
    point_count = 0
    for row_index in range(max_col_by_row.size):
        max_col = max_col_by_row[row_index]
        if max_col < -9223372036854775800:
            continue
        row2 = row_index - 1
        min_col = min_col_by_row[row_index]
        point_x[point_count] = row2
        point_y[point_count] = min_col
        point_count += 1
        if max_col != min_col:
            point_x[point_count] = row2
            point_y[point_count] = max_col
            point_count += 1
    return point_count


@njit(cache=True)
def _collect_diamond_extreme_points(
    image: np.ndarray,
    valid_mask: np.ndarray,
    threshold: np.float32,
    min_col_by_row: np.ndarray,
    max_col_by_row: np.ndarray,
    point_x: np.ndarray,
    point_y: np.ndarray,
) -> int:
    height, width = image.shape
    row_count2 = height * 2 + 1
    for row_index in range(row_count2):
        min_col_by_row[row_index] = 9223372036854775807
        max_col_by_row[row_index] = -9223372036854775807
    for y in range(height):
        for x in range(width):
            if valid_mask[y, x] and image[y, x] >= threshold:
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y - 1, 2 * x)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y + 1, 2 * x)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x - 1)
                _add_diamond_vertex(min_col_by_row, max_col_by_row, 2 * y, 2 * x + 1)
    return _emit_diamond_extreme_points(
        min_col_by_row, max_col_by_row, point_x, point_y
    )


@njit(cache=True)
def _exact_level_set_convex_hull_smoothing_reference_numba(
    image: np.ndarray, valid_mask: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    minimum = np.float32(0.0)
    maximum = np.float32(0.0)
    found_valid = False
    for y in range(height):
        for x in range(width):
            if not valid_mask[y, x]:
                continue
            value = image[y, x]
            if not found_valid:
                minimum = value
                maximum = value
                found_valid = True
            else:
                if value < minimum:
                    minimum = value
                if value > maximum:
                    maximum = value
    output = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            output[y, x] = minimum if valid_mask[y, x] else np.float32(0.0)
    if not found_valid or maximum <= minimum:
        return output
    row_count2 = height * 2 + 1
    min_col_by_row = np.empty(row_count2, dtype=np.int64)
    max_col_by_row = np.empty(row_count2, dtype=np.int64)
    point_capacity = max(2, row_count2 * 2)
    point_x = np.empty(point_capacity, dtype=np.int64)
    point_y = np.empty(point_capacity, dtype=np.int64)
    hull_x = np.empty(point_capacity * 2, dtype=np.int64)
    hull_y = np.empty(point_capacity * 2, dtype=np.int64)
    assigned = np.ones((height, width), dtype=np.bool_)
    for level_index in range(thresholds.size):
        threshold = thresholds[level_index]
        point_count = _collect_diamond_extreme_points(
            image,
            valid_mask,
            threshold,
            min_col_by_row,
            max_col_by_row,
            point_x,
            point_y,
        )
        if point_count == 0:
            continue
        hull_count = _monotone_chain_hull(point_x, point_y, point_count, hull_x, hull_y)
        _paint_convex_hull(
            output, assigned, True, valid_mask, threshold, hull_x, hull_y, hull_count
        )
    return output


@njit(cache=True)
def _add_diamond_vertex(
    min_col_by_row: np.ndarray, max_col_by_row: np.ndarray, row2: int, col2: int
) -> bool:
    row_index = row2 + 1
    changed = False
    if col2 < min_col_by_row[row_index]:
        min_col_by_row[row_index] = col2
        changed = True
    if col2 > max_col_by_row[row_index]:
        max_col_by_row[row_index] = col2
        changed = True
    return changed


@njit(cache=True)
def _cross_points(ax: int, ay: int, bx: int, by: int, cx: int, cy: int) -> int:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _monotone_chain_hull(
    point_x: np.ndarray,
    point_y: np.ndarray,
    point_count: int,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
) -> int:
    if point_count <= 1:
        if point_count == 1:
            hull_x[0] = point_x[0]
            hull_y[0] = point_y[0]
        return point_count
    hull_count = 0
    for index in range(point_count):
        px = point_x[index]
        py = point_y[index]
        while (
            hull_count >= 2
            and _cross_points(
                hull_x[hull_count - 2],
                hull_y[hull_count - 2],
                hull_x[hull_count - 1],
                hull_y[hull_count - 1],
                px,
                py,
            )
            <= 0
        ):
            hull_count -= 1
        hull_x[hull_count] = px
        hull_y[hull_count] = py
        hull_count += 1
    lower_count = hull_count
    for index in range(point_count - 2, -1, -1):
        px = point_x[index]
        py = point_y[index]
        while (
            hull_count > lower_count
            and _cross_points(
                hull_x[hull_count - 2],
                hull_y[hull_count - 2],
                hull_x[hull_count - 1],
                hull_y[hull_count - 1],
                px,
                py,
            )
            <= 0
        ):
            hull_count -= 1
        hull_x[hull_count] = px
        hull_y[hull_count] = py
        hull_count += 1
    if hull_count > 1:
        hull_count -= 1
    return hull_count


@njit(cache=True)
def _paint_convex_hull(
    output: np.ndarray,
    assigned: np.ndarray,
    overwrite_assigned: bool,
    valid_mask: np.ndarray,
    threshold: np.float32,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
    hull_count: int,
) -> int:
    if hull_count <= 0:
        return 0
    if hull_count == 1:
        if hull_x[0] % 2 != 0 or hull_y[0] % 2 != 0:
            return 0
        y = hull_x[0] // 2
        x = hull_y[0] // 2
        if (
            y >= 0
            and y < valid_mask.shape[0]
            and (x >= 0)
            and (x < valid_mask.shape[1])
            and valid_mask[y, x]
            and (overwrite_assigned or not assigned[y, x])
        ):
            output[y, x] = threshold
            was_unassigned = not assigned[y, x]
            assigned[y, x] = True
            return 1 if was_unassigned else 0
        return 0
    min_row2 = hull_x[0]
    max_row2 = hull_x[0]
    min_col2 = hull_y[0]
    max_col2 = hull_y[0]
    for index in range(1, hull_count):
        row2 = hull_x[index]
        col2 = hull_y[index]
        if row2 < min_row2:
            min_row2 = row2
        if row2 > max_row2:
            max_row2 = row2
        if col2 < min_col2:
            min_col2 = col2
        if col2 > max_col2:
            max_col2 = col2
    if hull_count == 2:
        return _paint_line_hull(
            output,
            assigned,
            overwrite_assigned,
            valid_mask,
            threshold,
            hull_x[0],
            hull_y[0],
            hull_x[1],
            hull_y[1],
            min_row2,
            max_row2,
            min_col2,
            max_col2,
        )
    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2(min_row2))
    max_y = min(image_height - 1, _floor_div2(max_row2))
    min_x = max(0, _ceil_div2(min_col2))
    max_x = min(image_width - 1, _floor_div2(max_col2))
    return _paint_polygon_hull_scanlines(
        output,
        assigned,
        overwrite_assigned,
        valid_mask,
        threshold,
        hull_x,
        hull_y,
        hull_count,
        min_y,
        max_y,
        min_x,
        max_x,
    )


@njit(cache=True)
def _paint_polygon_hull_scanlines(
    output: np.ndarray,
    assigned: np.ndarray,
    overwrite_assigned: bool,
    valid_mask: np.ndarray,
    threshold: np.float32,
    hull_x: np.ndarray,
    hull_y: np.ndarray,
    hull_count: int,
    min_y: int,
    max_y: int,
    min_x: int,
    max_x: int,
) -> int:
    assigned_delta = 0
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        left_col2 = 0.0
        right_col2 = 0.0
        found_intersection = False
        for index in range(hull_count):
            next_index = index + 1
            if next_index == hull_count:
                next_index = 0
            row0 = hull_x[index]
            col0 = hull_y[index]
            row1 = hull_x[next_index]
            col1 = hull_y[next_index]
            if row0 == row1:
                if query_row2 != row0:
                    continue
                edge_left = float(min(col0, col1))
                edge_right = float(max(col0, col1))
                if not found_intersection:
                    left_col2 = edge_left
                    right_col2 = edge_right
                    found_intersection = True
                else:
                    if edge_left < left_col2:
                        left_col2 = edge_left
                    if edge_right > right_col2:
                        right_col2 = edge_right
                continue
            row_min = min(row0, row1)
            row_max = max(row0, row1)
            if query_row2 < row_min or query_row2 > row_max:
                continue
            row_fraction = (query_row2 - row0) / (row1 - row0)
            intersection_col2 = col0 + row_fraction * (col1 - col0)
            if not found_intersection:
                left_col2 = intersection_col2
                right_col2 = intersection_col2
                found_intersection = True
            else:
                if intersection_col2 < left_col2:
                    left_col2 = intersection_col2
                if intersection_col2 > right_col2:
                    right_col2 = intersection_col2
        if not found_intersection:
            continue
        scan_min_x = max(min_x, int(np.ceil(left_col2 / 2.0 - 1e-09)))
        scan_max_x = min(max_x, int(np.floor(right_col2 / 2.0 + 1e-09)))
        for x in range(scan_min_x, scan_max_x + 1):
            if not valid_mask[y, x]:
                continue
            if not overwrite_assigned and assigned[y, x]:
                continue
            output[y, x] = threshold
            if not assigned[y, x]:
                assigned_delta += 1
            assigned[y, x] = True
    return assigned_delta


@njit(cache=True)
def _ceil_div2(value: int) -> int:
    if value >= 0:
        return (value + 1) // 2
    return value // 2


@njit(cache=True)
def _floor_div2(value: int) -> int:
    if value >= 0:
        return value // 2
    return -((-value + 1) // 2)


@njit(cache=True)
def _paint_line_hull(
    output: np.ndarray,
    assigned: np.ndarray,
    overwrite_assigned: bool,
    valid_mask: np.ndarray,
    threshold: np.float32,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    min_row2: int,
    max_row2: int,
    min_col2: int,
    max_col2: int,
) -> int:
    dx = x1 - x0
    dy = y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0:
        if valid_mask[y0, x0] and (overwrite_assigned or not assigned[y0, x0]):
            output[y0, x0] = threshold
            was_unassigned = not assigned[y0, x0]
            assigned[y0, x0] = True
            return 1 if was_unassigned else 0
        return 0
    image_height, image_width = output.shape
    min_y = max(0, _ceil_div2(min_row2))
    max_y = min(image_height - 1, _floor_div2(max_row2))
    min_x = max(0, _ceil_div2(min_col2))
    max_x = min(image_width - 1, _floor_div2(max_col2))
    assigned_delta = 0
    for y in range(min_y, max_y + 1):
        query_row2 = y * 2
        for x in range(min_x, max_x + 1):
            if not valid_mask[y, x]:
                continue
            if not overwrite_assigned and assigned[y, x]:
                continue
            query_col2 = x * 2
            dot = (query_row2 - x0) * dx + (query_col2 - y0) * dy
            if dot < 0 or dot > length2:
                continue
            cross = dx * (query_col2 - y0) - dy * (query_row2 - x0)
            if cross == 0:
                output[y, x] = threshold
                if not assigned[y, x]:
                    assigned_delta += 1
                assigned[y, x] = True
    return assigned_delta


@njit(cache=True)
def _rank_median_global_minimum_is_majority_everywhere_numba(
    image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    minimum_value: np.uint16,
) -> bool:
    height, width = image.shape
    for y in range(height):
        total_count = 0
        minimum_count = 0
        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                total_count += 1
                if image[yy, xx] == minimum_value:
                    minimum_count += 1
        if minimum_count <= total_count // 2:
            return False
        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]
                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    total_count -= 1
                    if image[yy, remove_x] == minimum_value:
                        minimum_count -= 1
                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    total_count += 1
                    if image[yy, add_x] == minimum_value:
                        minimum_count += 1
            if minimum_count <= total_count // 2:
                return False
    return True


@njit(cache=True)
def _rank_median_codes_2d_sliding_histogram_numba(
    code_image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.empty((height, width), dtype=np.int32)
    for y in range(height):
        tree = np.zeros(value_count + 1, dtype=np.int64)
        count = 0
        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                _fenwick_add_code(tree, int(code_image[yy, xx]), 1)
                count += 1
        if count == 0:
            output[y, 0] = 0
        else:
            output[y, 0] = _fenwick_select_code(tree, count // 2)
        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]
                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    _fenwick_add_code(tree, int(code_image[yy, remove_x]), -1)
                    count -= 1
                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    _fenwick_add_code(tree, int(code_image[yy, add_x]), 1)
                    count += 1
            if count == 0:
                output[y, x] = 0
            else:
                output[y, x] = _fenwick_select_code(tree, count // 2)
    return output


@njit(cache=True)
def _rank_median_small_codes_2d_sliding_histogram_numba(
    code_image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.empty((height, width), dtype=np.int32)
    for y in range(height):
        histogram = np.zeros(value_count, dtype=np.int32)
        count = 0
        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                histogram[int(code_image[yy, xx])] += 1
                count += 1
        output[y, 0] = _rank_median_select_small_code(histogram, count)
        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]
                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    remove_code = int(code_image[yy, remove_x])
                    histogram[remove_code] -= 1
                    count -= 1
                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    add_code = int(code_image[yy, add_x])
                    histogram[add_code] += 1
                    count += 1
            output[y, x] = _rank_median_select_small_code(histogram, count)
    return output


@njit(cache=True)
def _rank_median_small_codes_minimum_majority_hybrid_numba(
    code_image: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        total_count = 0
        minimum_count = 0
        run_start = -1
        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                total_count += 1
                if code_image[yy, xx] == 0:
                    minimum_count += 1
        if minimum_count <= total_count // 2:
            run_start = 0
        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]
                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    total_count -= 1
                    if code_image[yy, remove_x] == 0:
                        minimum_count -= 1
                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    total_count += 1
                    if code_image[yy, add_x] == 0:
                        minimum_count += 1
            needs_exact = minimum_count <= total_count // 2
            if needs_exact and run_start < 0:
                run_start = x
            elif not needs_exact and run_start >= 0:
                _rank_median_fill_small_code_run(
                    code_image,
                    output,
                    row_offsets_y,
                    row_radii_x,
                    value_count,
                    y,
                    run_start,
                    x - 1,
                )
                run_start = -1
        if run_start >= 0:
            _rank_median_fill_small_code_run(
                code_image,
                output,
                row_offsets_y,
                row_radii_x,
                value_count,
                y,
                run_start,
                width - 1,
            )
    return output


def _rank_median_fft_zero_mask_candidate(
    code_image: np.ndarray,
    footprint: np.ndarray,
) -> bool:
    """Return whether FFT zero-majority counting is likely to beat sliding counts."""
    return bool(code_image.size >= 262_144 and np.asarray(footprint).sum() >= 4096)


def _rank_median_zero_exact_mask_fft(
    code_image: np.ndarray,
    footprint: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
) -> np.ndarray:
    """Return pixels whose local rank median is not provably the minimum code."""
    from scipy import signal

    zero_image = (np.asarray(code_image) == 0).astype(np.float32, copy=False)
    footprint_float = np.asarray(footprint, dtype=np.float32)
    zero_counts = np.rint(
        signal.fftconvolve(zero_image, footprint_float, mode="same")
    ).astype(np.int32, copy=False)
    total_counts = np.rint(
        signal.fftconvolve(
            np.ones(code_image.shape, dtype=np.float32),
            footprint_float,
            mode="same",
        )
    ).astype(np.int32, copy=False)
    exact_mask = zero_counts <= (total_counts // 2)
    uncertain_mask = np.abs(zero_counts - (total_counts // 2)) <= 2
    if np.any(uncertain_mask):
        _rank_median_correct_zero_exact_mask_numba(
            np.ascontiguousarray(code_image),
            exact_mask,
            np.ascontiguousarray(uncertain_mask),
            row_offsets_y,
            row_radii_x,
        )
    return np.ascontiguousarray(exact_mask)


@njit(cache=True)
def _rank_median_correct_zero_exact_mask_numba(
    code_image: np.ndarray,
    exact_mask: np.ndarray,
    uncertain_mask: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
) -> None:
    height, width = code_image.shape
    for y in range(height):
        for x in range(width):
            if not uncertain_mask[y, x]:
                continue
            total_count = 0
            minimum_count = 0
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]
                left = x - radius_x
                if left < 0:
                    left = 0
                right = x + radius_x
                if right >= width:
                    right = width - 1
                for xx in range(left, right + 1):
                    total_count += 1
                    if code_image[yy, xx] == 0:
                        minimum_count += 1
            exact_mask[y, x] = minimum_count <= total_count // 2


@njit(cache=True)
def _rank_median_small_codes_exact_mask_runs_numba(
    code_image: np.ndarray,
    exact_mask: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
) -> np.ndarray:
    height, width = code_image.shape
    output = np.zeros((height, width), dtype=np.int32)
    for y in range(height):
        run_start = -1
        for x in range(width):
            needs_exact = exact_mask[y, x]
            if needs_exact and run_start < 0:
                run_start = x
            elif not needs_exact and run_start >= 0:
                _rank_median_fill_small_code_run(
                    code_image,
                    output,
                    row_offsets_y,
                    row_radii_x,
                    value_count,
                    y,
                    run_start,
                    x - 1,
                )
                run_start = -1
        if run_start >= 0:
            _rank_median_fill_small_code_run(
                code_image,
                output,
                row_offsets_y,
                row_radii_x,
                value_count,
                y,
                run_start,
                width - 1,
            )
    return output


@njit(cache=True)
def _rank_median_fill_small_code_run(
    code_image: np.ndarray,
    output: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
    value_count: int,
    y: int,
    start_x: int,
    end_x: int,
) -> None:
    height, width = code_image.shape
    histogram = np.zeros(value_count, dtype=np.int32)
    count = 0
    for row_index in range(row_offsets_y.shape[0]):
        yy = y + row_offsets_y[row_index]
        if yy < 0 or yy >= height:
            continue
        radius_x = row_radii_x[row_index]
        left = start_x - radius_x
        if left < 0:
            left = 0
        right = start_x + radius_x
        if right >= width:
            right = width - 1
        for xx in range(left, right + 1):
            histogram[int(code_image[yy, xx])] += 1
            count += 1
    output[y, start_x] = _rank_median_select_small_code(histogram, count)
    for x in range(start_x + 1, end_x + 1):
        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            remove_x = x - 1 - radius_x
            if remove_x >= 0 and remove_x < width:
                histogram[int(code_image[yy, remove_x])] -= 1
                count -= 1
            add_x = x + radius_x
            if add_x >= 0 and add_x < width:
                histogram[int(code_image[yy, add_x])] += 1
                count += 1
        output[y, x] = _rank_median_select_small_code(histogram, count)


@njit(cache=True)
def _rank_median_select_small_code(histogram: np.ndarray, count: int) -> int:
    target = count // 2 + 1
    cumulative = 0
    for code in range(histogram.shape[0]):
        cumulative += histogram[code]
        if cumulative >= target:
            return code
    return max(0, histogram.shape[0] - 1)


@njit(cache=True)
def _fenwick_add_code(tree: np.ndarray, code: int, delta: int) -> None:
    index = code + 1
    while index < tree.shape[0]:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_select_code(tree: np.ndarray, kth: int) -> int:
    return _fenwick_select_index(tree, kth, _highest_fenwick_bit(tree))


@njit(cache=True)
def _highest_fenwick_bit(tree: np.ndarray) -> int:
    bit = 1
    while bit < tree.shape[0]:
        bit <<= 1
    return bit >> 1


@njit(cache=True)
def _fenwick_select_index(tree: np.ndarray, kth: int, initial_bit: int) -> int:
    index = 0
    bit = initial_bit
    target = kth + 1
    while bit != 0:
        next_index = index + bit
        if next_index < tree.shape[0] and tree[next_index] < target:
            index = next_index
            target -= tree[next_index]
        bit >>= 1
    return index


@njit(cache=True)
def _rank_median_uint16_2d_sliding_histogram_numba(
    image: np.ndarray,
    mask: np.ndarray,
    row_offsets_y: np.ndarray,
    row_radii_x: np.ndarray,
) -> np.ndarray:
    height, width = image.shape
    output = np.empty((height, width), dtype=np.uint16)
    histogram_size = 65536
    for y in range(height):
        tree = np.zeros(histogram_size + 1, dtype=np.int64)
        count = 0
        for row_index in range(row_offsets_y.shape[0]):
            yy = y + row_offsets_y[row_index]
            if yy < 0 or yy >= height:
                continue
            radius_x = row_radii_x[row_index]
            right = radius_x
            if right >= width:
                right = width - 1
            for xx in range(0, right + 1):
                value = image[yy, xx] if mask[yy, xx] else np.uint16(0)
                _fenwick_add_uint16(tree, value, 1)
                count += 1
        if count == 0:
            output[y, 0] = np.uint16(0)
        else:
            output[y, 0] = _fenwick_select_uint16(tree, count // 2)
        for x in range(1, width):
            for row_index in range(row_offsets_y.shape[0]):
                yy = y + row_offsets_y[row_index]
                if yy < 0 or yy >= height:
                    continue
                radius_x = row_radii_x[row_index]
                remove_x = x - 1 - radius_x
                if remove_x >= 0 and remove_x < width:
                    value = image[yy, remove_x] if mask[yy, remove_x] else np.uint16(0)
                    _fenwick_add_uint16(tree, value, -1)
                    count -= 1
                add_x = x + radius_x
                if add_x >= 0 and add_x < width:
                    value = image[yy, add_x] if mask[yy, add_x] else np.uint16(0)
                    _fenwick_add_uint16(tree, value, 1)
                    count += 1
            if count == 0:
                output[y, x] = np.uint16(0)
            else:
                output[y, x] = _fenwick_select_uint16(tree, count // 2)
    return output


@njit(cache=True)
def _fenwick_add_uint16(tree: np.ndarray, value: np.uint16, delta: int) -> None:
    index = int(value) + 1
    while index < tree.shape[0]:
        tree[index] += delta
        index += index & -index


@njit(cache=True)
def _fenwick_select_uint16(tree: np.ndarray, kth: int) -> np.uint16:
    return np.uint16(_fenwick_select_index(tree, kth, 32768))


@njit(cache=True)
def _rank_median_uint16_2d_numba(
    image: np.ndarray, mask: np.ndarray, offsets_y: np.ndarray, offsets_x: np.ndarray
) -> np.ndarray:
    height, width = image.shape
    output = np.empty((height, width), dtype=np.uint16)
    footprint_size = offsets_y.shape[0]
    for y in range(height):
        values = np.empty(footprint_size, dtype=np.uint16)
        for x in range(width):
            count = 0
            for offset_index in range(footprint_size):
                yy = y + offsets_y[offset_index]
                xx = x + offsets_x[offset_index]
                if 0 <= yy < height and 0 <= xx < width:
                    if mask[yy, xx]:
                        values[count] = image[yy, xx]
                    else:
                        values[count] = np.uint16(0)
                    count += 1
            output[y, x] = _select_uint16(values, count, count // 2)
    return output


@njit(cache=True)
def _select_uint16(values: np.ndarray, count: int, kth: int) -> np.uint16:
    left = 0
    right = count - 1
    while True:
        if left == right:
            return values[left]
        pivot_index = (left + right) // 2
        pivot_index = _partition_uint16(values, left, right, pivot_index)
        if kth == pivot_index:
            return values[kth]
        if kth < pivot_index:
            right = pivot_index - 1
        else:
            left = pivot_index + 1


@njit(cache=True)
def _partition_uint16(
    values: np.ndarray, left: int, right: int, pivot_index: int
) -> int:
    pivot_value = values[pivot_index]
    values[pivot_index] = values[right]
    values[right] = pivot_value
    store_index = left
    for index in range(left, right):
        if values[index] < pivot_value:
            current = values[store_index]
            values[store_index] = values[index]
            values[index] = current
            store_index += 1
    current = values[right]
    values[right] = values[store_index]
    values[store_index] = current
    return store_index


__all__ = public_names_from_objects(
    AutomaticSmoothingFilterSizeStrategy,
    AverageSmoothingPlaneStrategy,
    CalculationScope,
    ConvexHullSmoothingBackendStrategy,
    ConvexHullSmoothingPlaneStrategy,
    DivideIlluminationCorrectionStrategy,
    ExactLevelSetNumpyConvexHullSmoothingBackendStrategy,
    FilterSizeMethod,
    FitPolynomialSmoothingPlaneStrategy,
    GaussianFilterSmoothingPlaneStrategy,
    IlluminationCorrectionMethod,
    IlluminationCorrectionStrategy,
    IntensityChoice,
    LegacyFastNumpyConvexHullSmoothingBackendStrategy,
    ManualSmoothingFilterSizeStrategy,
    MedianFilterSmoothingPlaneStrategy,
    NativeExactLevelSetNumpyConvexHullSmoothingBackendStrategy,
    NativeNumpyRankMedianSmoothingBackendStrategy,
    NoSmoothingPlaneStrategy,
    NumbaNumpyRankMedianSmoothingBackendStrategy,
    ObjectWidthSmoothingFilterSizeStrategy,
    RankMedianSmoothingBackendStrategy,
    RescaleOption,
    SmoothingFilterSizeStrategy,
    SmoothingMethod,
    SmoothingPlaneStrategy,
    SplineBgMode,
    SplinesSmoothingPlaneStrategy,
    SubtractIlluminationCorrectionStrategy,
    correct_illumination_apply,
    correct_illumination_calculate,
    fit_polynomial_surface,
    fit_polynomial_unmasked_gram,
)
