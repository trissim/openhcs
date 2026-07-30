"""CellProfiler-compatible local maxima detection backend."""

from __future__ import annotations
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, ClassVar, TYPE_CHECKING
from metaclass_registry import AutoRegisterMeta
import numpy as np
import scipy.ndimage
from skimage.feature import peak_local_max
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    ImageMeasurementSubjectRelation,
    ObjectLabelsArtifactType,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.memory.decorators import numpy
from openhcs.core.public_api import public_names_from_objects
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
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

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.interop.cellprofiler.parser import ModuleBlock


class ExcludeMode(Enum):
    THRESHOLD = "threshold"
    MASK = "mask"
    OBJECTS = "objects"


ExcludeMode.THRESHOLD.cellprofiler_literals = ("Threshold",)
ExcludeMode.MASK.cellprofiler_literals = ("Mask",)
ExcludeMode.OBJECTS.cellprofiler_literals = ("Within Objects",)


@dataclass(frozen=True, slots=True)
class MaximaResult(MeasurementFeatureRecord):
    """OpenHCS maxima diagnostics; CellProfiler declares no native measurements."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    maxima_count: int
    min_distance_used: int
    threshold_used: float


@dataclass(frozen=True, slots=True)
class MaximaRequest:
    """Normalized maxima detection request."""

    image: np.ndarray
    min_distance: int
    min_intensity: float
    label_maxima: bool

    @property
    def threshold_abs(self) -> float | None:
        return self.min_intensity if self.min_intensity > 0 else None

    def detect(self) -> tuple[np.ndarray, MaximaResult]:
        maxima_coords = peak_local_max(
            self.image, min_distance=self.min_distance, threshold_abs=self.threshold_abs
        )
        output = np.zeros(self.image.shape, dtype=np.float32)
        if len(maxima_coords) > 0:
            output[tuple(maxima_coords.T)] = 1.0
        if self.label_maxima:
            output = scipy.ndimage.label(output > 0)[0].astype(np.float32)
        return (
            output,
            MaximaResult(
                slice_index=0,
                maxima_count=len(maxima_coords),
                min_distance_used=self.min_distance,
                threshold_used=(
                    self.threshold_abs if self.threshold_abs is not None else 0.0
                ),
            ),
        )


class MaximaInputStrategy(
    EnumKeyedStrategyMixin[ExcludeMode], ABC, metaclass=AutoRegisterMeta
):
    """Build the effective maxima source image for one CP exclusion mode."""

    __registry_key__ = "exclude_mode_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "exclude_mode"
    __enum_label_attr__ = "exclude_mode_label"
    exclude_mode: ClassVar[ExcludeMode | None] = None
    exclude_mode_label: ClassVar[str | None] = None

    @classmethod
    def for_exclude_mode(cls, exclude_mode: ExcludeMode) -> "MaximaInputStrategy":
        return cls.for_enum_member(exclude_mode)

    @abstractmethod
    def image(self, image: np.ndarray) -> np.ndarray:
        """Return the effective image for peak detection."""


class ThresholdMaximaInputStrategy(MaximaInputStrategy):
    exclude_mode = ExcludeMode.THRESHOLD

    def image(self, image: np.ndarray) -> np.ndarray:
        return image.copy()


class MaskMaximaInputStrategy(MaximaInputStrategy):
    exclude_mode = ExcludeMode.MASK

    def image(self, image: np.ndarray) -> np.ndarray:
        intensity_image = image[0].copy()
        intensity_image[~image[1].astype(bool)] = 0
        return intensity_image


class ObjectMaximaInputStrategy(MaskMaximaInputStrategy):
    exclude_mode = ExcludeMode.OBJECTS


@numpy(contract=ProcessingContract.PURE_2D)
def find_maxima(
    image: np.ndarray,
    min_distance: int = 5,
    exclude_mode: ExcludeMode = ExcludeMode.THRESHOLD,
    min_intensity: float = 0.0,
    label_maxima: bool = True,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Find local maxima under the requested CP exclusion policy.

    Args:
        min_distance: Minimum center-to-center separation between retained peaks,
            in pixels.
        exclude_mode: Policy for excluding background, masked, or object pixels
            from peak detection.
        min_intensity: Lowest pixel value eligible to become a maximum.
        label_maxima: Assign a distinct positive label to each peak instead of a
            binary peak mask.
    """
    maxima, result = MaximaRequest(
        image=MaximaInputStrategy.for_exclude_mode(exclude_mode).image(image),
        min_distance=min_distance,
        min_intensity=min_intensity,
        label_maxima=label_maxima,
    ).detect()
    return (
        maxima,
        DataclassMeasurementColumnarRows((result,), row_type=MaximaResult),
    )


@numpy(contract=ProcessingContract.PURE_3D)
def find_maxima_with_mask(
    image: np.ndarray,
    min_distance: int = 5,
    min_intensity: float = 0.0,
    label_maxima: bool = True,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Find local maxima within a stacked mask input."""
    maxima, result = MaximaRequest(
        image=MaximaInputStrategy.for_exclude_mode(ExcludeMode.MASK).image(image),
        min_distance=min_distance,
        min_intensity=min_intensity,
        label_maxima=label_maxima,
    ).detect()
    return (
        maxima[np.newaxis, ...],
        DataclassMeasurementColumnarRows((result,), row_type=MaximaResult),
    )


class FindMaximaModule(
    NoObjectNameMeasurementRecordMixin,
    ProducedImageMeasurementRecordMixin,
    MeasurementArtifactOutputModule,
):
    module_name = "FindMaxima"
    function_name = "find_maxima"
    function_variants = ("find_maxima_with_mask",)
    validated = True
    confidence = 1.0

    input_image_setting = SettingNameFamily("Select the input image")
    output_image_setting = SettingNameFamily("Name the output image")
    label_maxima_setting = SettingNameFamily("Individually label maxima?")
    min_distance_setting = SettingNameFamily("Minimum distance between maxima")
    exclude_mode_setting = SettingNameFamily("Method for excluding background")
    min_intensity_setting = SettingNameFamily(
        "Specify the minimum intensity of a peak"
    )
    mask_image_setting = SettingNameFamily("Select the image to use as a mask")
    mask_objects_setting = SettingNameFamily(
        "Select the objects to search within"
    )

    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting,
        ImageArtifactType,
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting,
        ImageArtifactType,
    )
    mask_image_binding = SettingToKeywordBinding.input(
        mask_image_setting,
        ImageArtifactType,
    )
    mask_objects_binding = SettingToKeywordBinding.input(
        mask_objects_setting,
        ObjectLabelsArtifactType,
    )
    setting_bindings = (
        input_image_binding,
        output_image_binding,
        SettingToKeywordBinding(
            label_maxima_setting,
            "label_maxima",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            min_distance_setting,
            "min_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            exclude_mode_setting,
            "exclude_mode",
            cellprofiler_enum_value_setting_parser(ExcludeMode),
        ),
        SettingToKeywordBinding(
            min_intensity_setting,
            "min_intensity",
            parse_cellprofiler_float,
        ),
        mask_image_binding,
        mask_objects_binding,
    )

    @classmethod
    def _exclude_mode(cls, module: "ModuleBlock") -> ExcludeMode:
        values = setting_values(module, cls.exclude_mode_setting)
        if len(values) > 1:
            raise ValueError(
                f"FindMaxima declares multiple exclusion modes: {values!r}."
            )
        return coerce_cellprofiler_enum(
            ExcludeMode,
            values[0] if values else ExcludeMode.THRESHOLD.value,
        )

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Select the declared threshold, mask-image, or object input domain."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        mode = cls._exclude_mode(module)
        inactive = {
            ExcludeMode.THRESHOLD: (cls.mask_image_binding, cls.mask_objects_binding),
            ExcludeMode.MASK: (cls.mask_objects_binding,),
            ExcludeMode.OBJECTS: (cls.mask_image_binding,),
        }[mode]
        return tuple(binding for binding in bindings if binding not in inactive)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract,
        source_bindings,
    ):
        """Use one composed intensity/mask invocation outside threshold mode."""

        if cls._exclude_mode(module) is not ExcludeMode.THRESHOLD:
            return cls.require_callable("find_maxima_with_mask")
        return super().resolve_function(
            module,
            contract=contract,
            source_bindings=source_bindings,
        )

    @classmethod
    def measurement_output_relations(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        """Declare the maxima image as the diagnostic measurement subject."""

        inherited = super().measurement_output_relations(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        output_name = optional_setting_value(module, cls.output_image_setting)
        if output_name is None:
            return inherited
        return (
            *inherited,
            ImageMeasurementSubjectRelation(
                source=ArtifactSpec.output(output_name, ImageArtifactType).ref()
            ),
        )


__all__ = public_names_from_objects(
    ExcludeMode,
    MaskMaximaInputStrategy,
    MaximaInputStrategy,
    MaximaRequest,
    MaximaResult,
    ObjectMaximaInputStrategy,
    ThresholdMaximaInputStrategy,
    find_maxima,
    find_maxima_with_mask,
)
