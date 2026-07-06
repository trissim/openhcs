"""Converted from CellProfiler: EnhanceOrSuppressFeatures."""

from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from typing import ClassVar
import numpy as np
import scipy.ndimage
import skimage.exposure
import skimage.filters
import skimage.morphology
import skimage.transform
from numba import njit
from metaclass_registry import AutoRegisterMeta
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactInputModule,
    ImageArtifactOutputModule,
)
from openhcs.core.callable_contract import processing_prepare
from openhcs.core.memory.decorators import numpy
from openhcs.core.runtime_values import (
    RuntimeImagePayloadContext,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
)
from openhcs.core.measurement_image_alignment import (
    ReplicatedChannelMonochromeProjection,
)
class OperationMethod(Enum):
    ENHANCE = "Enhance"
    SUPPRESS = "Suppress"


class EnhanceMethod(Enum):
    SPECKLES = "Speckles"
    NEURITES = "Neurites"
    DARK_HOLES = "Dark holes"
    CIRCLES = "Circles"
    TEXTURE = "Texture"
    DIC = "DIC"


class SpeckleAccuracy(Enum):
    FAST = "Fast"
    SLOW = "Slow"


class NeuriteMethod(Enum):
    GRADIENT = "Line structures"
    TUBENESS = "Tubeness"


STRATEGY_REGISTRY_KEY = "method_label"


@numpy(contract=ProcessingContract.PURE_2D)
def enhance_or_suppress_features(
    image: np.ndarray,
    method: OperationMethod = OperationMethod.ENHANCE,
    enhance_method: EnhanceMethod = EnhanceMethod.SPECKLES,
    radius: float = 10.0,
    speckle_accuracy: SpeckleAccuracy = SpeckleAccuracy.FAST,
    neurite_method: NeuriteMethod = NeuriteMethod.GRADIENT,
    neurite_rescale: bool = False,
    dark_hole_radius_min: int = 1,
    dark_hole_radius_max: int = 10,
    smoothing_value: float = 2.0,
    dic_angle: float = 0.0,
    dic_decay: float = 0.95,
) -> np.ndarray:
    """Enhance or suppress image features using independent CP-compatible semantics."""
    method = coerce_cellprofiler_enum(OperationMethod, method)
    enhance_method = coerce_cellprofiler_enum(EnhanceMethod, enhance_method)
    speckle_accuracy = coerce_cellprofiler_enum(SpeckleAccuracy, speckle_accuracy)
    neurite_method = coerce_cellprofiler_enum(NeuriteMethod, neurite_method)
    image_data = ReplicatedChannelMonochromeProjection().plane(
        image_payload_data(image), name="enhancement image"
    )
    if image_data.dtype != np.float32 and image_data.dtype != np.float64:
        image_data = image_data.astype(np.float32)
    mask_context = FeatureEnhancementMaskContext(
        original=image_data, mask=_enhancement_mask(image, image_data)
    )
    result = FeatureOperationStrategy.for_method(method).apply(
        FeatureEnhancementRequest(
            mask_context=mask_context,
            enhance_method=enhance_method,
            radius=radius,
            speckle_accuracy=speckle_accuracy,
            neurite_method=neurite_method,
            neurite_rescale=neurite_rescale,
            dark_hole_radius_min=dark_hole_radius_min,
            dark_hole_radius_max=dark_hole_radius_max,
            smoothing_value=smoothing_value,
            dic_angle=dic_angle,
            dic_decay=dic_decay,
        )
    )
    return RuntimeImagePayloadContext(
        np.asarray(result, dtype=np.float32),
        mask=image_payload_mask(image),
        metadata=image_payload_metadata(image).without_unit_interval_intensity_scale(),
    ).payload()


class EnhanceOrSuppressFeaturesModule(
    ImageArtifactInputModule, ImageArtifactOutputModule, CellProfilerModule
):
    module_name = "EnhanceOrSuppressFeatures"
    function_name = "enhance_or_suppress_features"
    validated = True
    confidence = 1.0
    image_input_settings = ("Select the input image",)
    image_output_settings = ("Name the output image",)
    ignored_settings = (
        "Select the input image",
        "Name the output image",
        "Rescale result image",
    )
    setting_bindings = (
        SettingToKeywordBinding("Select the operation", "method"),
        SettingToKeywordBinding("Feature type", "enhance_method"),
        SettingToKeywordBinding("Smoothing scale", "smoothing_value"),
        SettingToKeywordBinding("Shear angle", "dic_angle"),
        SettingToKeywordBinding("Decay", "dic_decay"),
        SettingToKeywordBinding("Enhancement method", "neurite_method"),
        SettingToKeywordBinding("Speed and accuracy", "speckle_accuracy"),
        SettingToKeywordBinding("Range of hole sizes", "dark_hole_radius_range"),
        SettingToKeywordBinding("Feature size", "feature_size"),
    )

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: BoundModuleSettings
    ) -> BoundModuleSettings:
        kwargs = dict(bound.kwargs)
        hole_sizes = kwargs.pop("dark_hole_radius_range", None)
        if hole_sizes is not None:
            if not isinstance(hole_sizes, tuple) or len(hole_sizes) != 2:
                raise ValueError(
                    f"{module.name} hole size range must contain two values, got {hole_sizes!r}."
                )
            kwargs["dark_hole_radius_min"], kwargs["dark_hole_radius_max"] = hole_sizes
        feature_size = kwargs.pop("feature_size", None)
        if feature_size is not None:
            kwargs["radius"] = feature_size / 2
        return BoundModuleSettings(
            kwargs,
            bound.unmapped_kwargs,
            bound.invocation_options,
            bound.setting_coverage,
        )


@numpy(contract=ProcessingContract.PURE_3D)
def match_template(
    image: np.ndarray, template: np.ndarray | None = None, pad_input: bool = True
) -> np.ndarray:
    """Match an image template using normalized cross-correlation."""
    from skimage.feature import match_template as skimage_match_template

    if template is None:
        if image.shape[0] < 2:
            raise ValueError(
                "When template is not provided, image must have at least 2 slices in dimension 0: [input_image, template]."
            )
        output = skimage_match_template(
            image=image[0], template=image[1], pad_input=pad_input
        )
        return output[np.newaxis, ...].astype(np.float32)
    template_2d = template[0] if template.ndim == 3 else template
    return np.stack(
        [
            skimage_match_template(
                image=input_slice, template=template_2d, pad_input=pad_input
            )
            for input_slice in image
        ],
        axis=0,
    ).astype(np.float32)


@dataclass(frozen=True, slots=True)
class FeatureEnhancementMaskContext:
    """Image/mask authority for CP feature enhancement background semantics."""

    original: np.ndarray
    mask: np.ndarray

    @property
    def masked_original(self) -> np.ndarray:
        return np.where(self.mask, self.original, 0)

    def restore_background(self, result: np.ndarray) -> np.ndarray:
        output = np.asarray(result, dtype=np.float32).copy()
        output[~self.mask] = self.original[~self.mask]
        return output


@dataclass(frozen=True, slots=True)
class FeatureEnhancementRequest:
    """All CP feature enhancement settings after enum coercion."""

    mask_context: FeatureEnhancementMaskContext
    enhance_method: EnhanceMethod
    radius: float
    speckle_accuracy: SpeckleAccuracy
    neurite_method: NeuriteMethod
    neurite_rescale: bool
    dark_hole_radius_min: int
    dark_hole_radius_max: int
    smoothing_value: float
    dic_angle: float
    dic_decay: float


class FeatureOperationStrategy(
    EnumKeyedStrategyMixin[OperationMethod], ABC, metaclass=AutoRegisterMeta
):
    """Top-level CP enhance/suppress operation semantics."""

    __registry_key__ = STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    __enum_label_attr__ = STRATEGY_REGISTRY_KEY
    method: ClassVar[OperationMethod | None] = None
    method_label: ClassVar[str | None] = None

    @classmethod
    def for_method(cls, method: OperationMethod) -> "FeatureOperationStrategy":
        return cls.for_enum_member(method)

    @abstractmethod
    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        """Apply the top-level operation to a feature-enhancement request."""


class EnhanceFeatureOperationStrategy(FeatureOperationStrategy):
    method = OperationMethod.ENHANCE

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        return FeatureEnhanceMethodStrategy.for_method(request.enhance_method).apply(
            request
        )


class SuppressFeatureOperationStrategy(FeatureOperationStrategy):
    method = OperationMethod.SUPPRESS

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        footprint = _structuring_element(request.radius)
        opened = skimage.morphology.opening(
            request.mask_context.masked_original, footprint=footprint
        )
        return request.mask_context.restore_background(opened)


class FeatureEnhanceMethodStrategy(
    EnumKeyedStrategyMixin[EnhanceMethod], ABC, metaclass=AutoRegisterMeta
):
    """CP feature enhancement method semantics."""

    __registry_key__ = STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    __enum_member_attr__ = "method"
    __enum_label_attr__ = STRATEGY_REGISTRY_KEY
    method: ClassVar[EnhanceMethod | None] = None
    method_label: ClassVar[str | None] = None

    @classmethod
    def for_method(cls, method: EnhanceMethod) -> "FeatureEnhanceMethodStrategy":
        return cls.for_enum_member(method)

    @abstractmethod
    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        """Apply one concrete CP feature enhancement method."""


def _enhancement_mask(image: object, image_data: np.ndarray) -> np.ndarray:
    mask = image_payload_mask(image)
    if mask is None:
        return np.ones(image_data.shape, dtype=bool)
    return np.asarray(mask, dtype=bool)


def _structuring_element(radius: float) -> np.ndarray:
    return skimage.morphology.disk(max(1, int(round(radius))))


class SpecklesFeatureEnhanceMethodStrategy(FeatureEnhanceMethodStrategy):
    method = EnhanceMethod.SPECKLES

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        footprint = _structuring_element(request.radius)
        masked = request.mask_context.masked_original
        if request.speckle_accuracy is SpeckleAccuracy.FAST and request.radius > 3:
            opened = scipy.ndimage.maximum_filter(
                scipy.ndimage.minimum_filter(masked, footprint=footprint),
                footprint=footprint,
            )
            result = masked - opened
        else:
            result = skimage.morphology.white_tophat(masked, footprint=footprint)
        return request.mask_context.restore_background(result)


class NeuritesFeatureEnhanceMethodStrategy(FeatureEnhanceMethodStrategy):
    method = EnhanceMethod.NEURITES

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        masked = request.mask_context.masked_original
        if request.neurite_method is NeuriteMethod.TUBENESS:
            smoothed = scipy.ndimage.gaussian_filter(masked, request.smoothing_value)
            result = _tubeness_response_2d_numba(
                np.ascontiguousarray(smoothed, dtype=np.float64),
                float(request.smoothing_value),
            )
        else:
            footprint = _structuring_element(request.radius)
            result = (
                masked
                + skimage.morphology.white_tophat(masked, footprint=footprint)
                - skimage.morphology.black_tophat(masked, footprint=footprint)
            )
            result = np.clip(result, 0, None)
        if request.neurite_rescale:
            result = skimage.exposure.rescale_intensity(result, out_range=(0.0, 1.0))
        return request.mask_context.restore_background(result)


@njit(cache=True)
def _tubeness_response_2d_numba(
    image: np.ndarray, smoothing_value: float
) -> np.ndarray:
    height, width = image.shape
    result = np.zeros((height, width), dtype=np.float64)
    scale = smoothing_value * smoothing_value
    for y in range(height):
        for x in range(width):
            a = 0.0
            b = 0.0
            c = 0.0
            if 0 < y < height - 1:
                a = image[y - 1, x] - 2.0 * image[y, x] + image[y + 1, x]
            if 0 < x < width - 1:
                c = image[y, x - 1] - 2.0 * image[y, x] + image[y, x + 1]
            if 0 < y < height - 1 and 0 < x < width - 1:
                b = (
                    image[y + 1, x + 1]
                    + image[y - 1, x - 1]
                    - image[y + 1, x - 1]
                    - image[y - 1, x + 1]
                ) / 4.0
            linear = -(a + c)
            constant = a * c - b * b
            discriminant = linear * linear - 4.0 * constant
            if discriminant < 0.0:
                discriminant = 0.0
            sqrt_discriminant = np.sqrt(discriminant)
            root0 = (-linear + sqrt_discriminant) / 2.0
            root1 = (-linear - sqrt_discriminant) / 2.0
            selected = root0
            if abs(root1) > abs(root0):
                selected = root1
            if selected < 0.0:
                result[y, x] = -selected * scale
    return result


class DarkHolesFeatureEnhanceMethodStrategy(FeatureEnhanceMethodStrategy):
    method = EnhanceMethod.DARK_HOLES

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        masked = request.mask_context.masked_original
        radii = range(
            max(1, request.dark_hole_radius_min),
            max(request.dark_hole_radius_min, request.dark_hole_radius_max) + 1,
        )
        responses = [
            skimage.morphology.black_tophat(
                masked, footprint=_structuring_element(radius)
            )
            for radius in radii
        ]
        result = np.maximum.reduce(responses) if responses else np.zeros_like(masked)
        return request.mask_context.restore_background(result)


class CirclesFeatureEnhanceMethodStrategy(FeatureEnhanceMethodStrategy):
    method = EnhanceMethod.CIRCLES

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        masked = request.mask_context.masked_original
        radius_i = max(1, int(round(request.radius)))
        result = skimage.transform.hough_circle(masked, [radius_i])[0]
        return request.mask_context.restore_background(result)


class TextureFeatureEnhanceMethodStrategy(FeatureEnhanceMethodStrategy):
    method = EnhanceMethod.TEXTURE

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        masked = request.mask_context.masked_original.astype(float)
        mean = scipy.ndimage.gaussian_filter(masked, request.smoothing_value)
        mean_squared = scipy.ndimage.gaussian_filter(
            masked * masked, request.smoothing_value
        )
        result = np.maximum(mean_squared - mean * mean, 0)
        return request.mask_context.restore_background(result)


class DicFeatureEnhanceMethodStrategy(FeatureEnhanceMethodStrategy):
    method = EnhanceMethod.DIC

    def apply(self, request: FeatureEnhancementRequest) -> np.ndarray:
        smoothed = scipy.ndimage.gaussian_filter(
            request.mask_context.masked_original, request.smoothing_value
        )
        radians = np.deg2rad(request.dic_angle)
        shift = np.array((np.sin(radians), np.cos(radians))) * max(request.dic_decay, 0)
        coords = np.indices(smoothed.shape, dtype=float)
        forward = scipy.ndimage.map_coordinates(
            smoothed, coords + shift.reshape(2, 1, 1), order=1, mode="nearest"
        )
        backward = scipy.ndimage.map_coordinates(
            smoothed, coords - shift.reshape(2, 1, 1), order=1, mode="nearest"
        )
        result = np.maximum(forward - backward, 0)
        return request.mask_context.restore_background(result)


@processing_prepare(enhance_or_suppress_features)
def _prepare_enhance_or_suppress_features() -> None:
    """Compile accelerated enhancement kernels before timed execution."""
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 16] = 1.0
    enhance_or_suppress_features.__wrapped__(
        image,
        enhance_method=EnhanceMethod.NEURITES,
        neurite_method=NeuriteMethod.TUBENESS,
        smoothing_value=2.0,
    )


class MatchTemplateModule(CellProfilerModule):
    module_name = "MatchTemplate"
    function_name = "match_template"
    validated = True
    contract = ProcessingContract.PURE_3D
    confidence = 1.0
