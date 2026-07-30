"""Watershed backend strategies for CellProfiler-compatible processing."""

from __future__ import annotations
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable, ClassVar, TYPE_CHECKING
from metaclass_registry import AutoRegisterMeta
from python_introspect import set_signature_analysis_target
from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.registry_strategies import (
    EnumKeyedStrategyMixin,
)
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.core.callable_contract import (
    attach_processing_prepare,
    callable_request,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
)
from openhcs.interop.cellprofiler.module_structuring_element_settings import (
    StructuringElementSetting,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_names,
    setting_values,
    split_symbol_names,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.callable_contract import CallableContract
    from openhcs.interop.cellprofiler.parser import ModuleBlock


def parse_watershed_border_exclusion(value: str) -> bool:
    """Parse legacy and current Watershed border-exclusion settings."""
    normalized = value.strip().lower()
    if normalized in {"yes", "true", "1", "on"}:
        return True
    if normalized in {"no", "false", "0", "off"}:
        return False
    pixel_count = int(value)
    if pixel_count == 0:
        return False
    raise ValueError(
        f"OpenHCS Watershed only supports CellProfiler border exclusion as a boolean edge clear, got pixel border width {pixel_count!r}."
    )


class WatershedMethod(str, Enum):
    """CellProfiler watershed surface source."""

    DISTANCE = "distance"
    INTENSITY = "intensity"
    MARKERS = "markers"


class WatershedModule(
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "Watershed"
    function_name = "watershed_library"
    function_variants = ("watershed_cellprofiler4",)
    validated = True
    respects_masks = True
    confidence = 1.0
    watershed_method_setting = SettingNameFamily(
        "Generate from",
        aliases=("Select watershed method",),
    )
    advanced_setting = "Use advanced settings?"
    declump_method_setting = "Declump method"
    seed_method_setting = "Select seed generation method"
    connectivity_setting = "Connectivity"
    compactness_setting = "Compactness"
    footprint_setting = "Footprint"
    downsample_setting = "Downsample"
    watershed_line_setting = SettingNameFamily(
        "Separate watershed labels",
        aliases=("Watershed line",),
    )
    gaussian_sigma_setting = SettingNameFamily(
        "Segmentation distance transform smoothing factor",
        aliases=("Gaussian sigma",),
    )
    minimum_distance_setting = SettingNameFamily(
        "Minimum distance between seeds",
        aliases=("Minimum distance",),
    )
    minimum_intensity_setting = "Minimum absolute internal distance"
    exclude_border_setting = SettingNameFamily(
        "Pixels from border to exclude",
        aliases=("Exclude objects touching the border?",),
    )
    maximum_seeds_setting = "Maximum number of seeds"
    markers_setting = "Markers"
    mask_setting = "Mask"
    segmentation_image_setting = SettingNameFamily(
        "Select the input image",
        aliases=("InputImage",),
    )
    intensity_image_setting = SettingNameFamily(
        "Reference Image", aliases=("Intensity image",)
    )
    output_object_setting = SettingNameFamily(
        "Name the output object",
        aliases=("OutputObjects",),
    )
    segmentation_image_binding = SettingToKeywordBinding.input(
        segmentation_image_setting, ImageArtifactType
    )
    intensity_image_binding = SettingToKeywordBinding.input(
        intensity_image_setting, ImageArtifactType, runtime_parameter_name="topology_inputs"
    )
    output_object_binding = SettingToKeywordBinding.output(
        output_object_setting, ObjectLabelsArtifactType
    )
    cellprofiler4_max_revision = 3
    ignored_settings = ("Display watershed seeds?",)
    watershed_method_binding = SettingToKeywordBinding(
        watershed_method_setting,
        "watershed_method",
        normalize_cellprofiler_setting_name,
    )
    markers_binding = SettingToKeywordBinding.input(
        markers_setting, ImageArtifactType, runtime_parameter_name="topology_inputs"
    )
    mask_binding = SettingToKeywordBinding.input(
        mask_setting, ImageArtifactType, runtime_parameter_name="topology_inputs"
    )
    declump_method_binding = SettingToKeywordBinding(
        declump_method_setting,
        "declump_method",
        normalize_cellprofiler_setting_name,
    )
    setting_bindings = (output_object_binding,segmentation_image_binding,
        markers_binding,
        intensity_image_binding,
        mask_binding,watershed_method_binding,
        SettingToKeywordBinding(
            advanced_setting,
            "use_advanced_settings",
            parse_cellprofiler_bool,
        ),
        declump_method_binding,
        SettingToKeywordBinding(
            seed_method_setting,
            "seed_method",
            normalize_cellprofiler_setting_name,
        ),
        SettingToKeywordBinding(
            connectivity_setting,
            "connectivity",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            compactness_setting,
            "compactness",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            footprint_setting,
            "footprint",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            downsample_setting,
            "downsample",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            watershed_line_setting,
            "watershed_line",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            gaussian_sigma_setting,
            "gaussian_sigma",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            minimum_distance_setting,
            "min_distance",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            minimum_intensity_setting,
            "min_intensity",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            exclude_border_setting,
            "exclude_border",
            parse_watershed_border_exclusion,
        ),
        SettingToKeywordBinding(
            maximum_seeds_setting,
            "max_seeds",
            parse_cellprofiler_int,
        ),)

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return only image roles active for this Watershed configuration."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        topology = WatershedInputTopology.from_values(
            coerce_cellprofiler_enum(
                WatershedMethod,
                cls._setting_row_value(module, cls.watershed_method_setting)
                or WatershedMethod.DISTANCE.value,
            ),
            coerce_cellprofiler_enum(
                WatershedDeclumpMethod,
                cls._setting_row_value(module, cls.declump_method_setting)
                or WatershedDeclumpMethod.SHAPE.value,
            ),
        )
        return tuple(
            binding
            for binding in bindings
            if topology.requires_markers or binding is not cls.markers_binding
            if topology.requires_intensity_image
            or binding is not cls.intensity_image_binding
            if cls._setting_row_symbol_names(module, cls.mask_setting)
            or binding is not cls.mask_binding
        )

    @classmethod
    def _setting_row_value(
        cls,
        module: "ModuleBlock",
        setting_name: str | SettingNameFamily,
    ) -> str | None:
        values = setting_values(module, setting_name)
        if len(values) > 1:
            raise ValueError(
                f"Watershed declares multiple rows for "
                f"{setting_names(setting_name)[0]!r}: {values!r}."
            )
        return values[0] if values else None

    @classmethod
    def _setting_row_symbol_names(
        cls,
        module: "ModuleBlock",
        setting_name: str | SettingNameFamily,
    ) -> tuple[str, ...]:
        value = cls._setting_row_value(module, setting_name)
        if value is None:
            return ()
        return split_symbol_names(str(value))

    @classmethod
    def artifact_contract_outputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        outputs = super().artifact_contract_outputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            artifact_inputs=artifact_inputs,
        )
        relations = cls._output_object_relations(
            module,
            artifact_inputs=artifact_inputs,
        )
        if not relations:
            return outputs
        return tuple(
            (
                replace(
                    output,
                    relations=(*output.relations, *relations),
                )
                if output.artifact_type is ObjectLabelsArtifactType
                else output
            )
            for output in outputs
        )

    @classmethod
    def _output_object_relations(
        cls,
        module: "ModuleBlock",
        *,
        artifact_inputs: ArtifactSpecCollection,
    ):
        source_name = required_setting_value(module, cls.segmentation_image_setting)
        source = artifact_inputs.require_by_name_and_artifact_type(
            source_name,
            ImageArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        del cls, contract, source_bindings
        module_revision = module.variable_revision_number
        if (
            module_revision is not None
            and module_revision <= WatershedModule.cellprofiler4_max_revision
        ):
            return watershed_cellprofiler4
        return watershed_library

    @classmethod
    def postprocess_bound_settings(
        cls, module: "ModuleBlock", bound: "BoundModuleSettings"
    ) -> "BoundModuleSettings":
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        structuring_element_setting = "Structuring element for seed dilation"
        structuring_element_value = (
            optional_setting_value(module, structuring_element_setting) or "Disk,1"
        )
        kwargs.update(
            StructuringElementSetting.from_cellprofiler_value(
                structuring_element_value
            ).bound_kwargs(
                shape_keyword="structuring_element",
                size_keyword="structuring_element_size",
            )
        )
        unmapped_kwargs.pop(
            normalize_cellprofiler_setting_name(structuring_element_setting), None
        )
        return BoundModuleSettings(kwargs, unmapped_kwargs, bound.setting_coverage)


import logging
import time
from abc import ABC, abstractmethod
import numpy as np
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.image_shapes import trailing_spatial_factors
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_object_label_domains import DenseObjectLabelConsecutiveRelabelingStrategy
from openhcs.core.runtime_image_values import (
    image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.structuring_elements import (
    StructuringElement,
    StructuringElementInput,
    StructuringElementSize,
    adapt_structuring_element_rank,
    build_structuring_element,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

NDIMAGE_CONSTANT_MODE = "constant"
WATERSHED_STRATEGY_REGISTRY_KEY = "strategy_label"
logger = logging.getLogger(__name__)


def watershed_xy_downsample_factors(ndim: int, factor: int) -> tuple[int, ...]:
    """Return rank-matched factors that downsample only the XY image domain."""
    return tuple(
        (int(value) for value in trailing_spatial_factors(ndim, (factor, factor)))
    )


def watershed_resize_labels(
    labels: np.ndarray,
    output_shape: tuple[int, ...],
) -> np.ndarray:
    """Resize labels exactly while avoiding redundant integer conversions."""
    from skimage.transform import resize

    resized = resize(
        labels,
        output_shape,
        mode="edge",
        order=0,
        preserve_range=True,
    )
    if not np.issubdtype(resized.dtype, np.integer):
        np.rint(resized, out=resized)
    return resized.astype(np.uint16, copy=False)


def watershed_connected_components(labels_like: np.ndarray) -> np.ndarray:
    """Label connected components over skimage-supported trailing spatial axes."""
    import skimage.measure

    labels_array = np.asarray(labels_like)
    if not np.any(labels_array):
        return np.zeros(labels_array.shape, dtype=np.int32)
    spatial_rank = min(labels_array.ndim, 3)
    if labels_array.ndim == spatial_rank:
        return skimage.measure.label(labels_array).astype(np.int32, copy=False)
    output = np.zeros(labels_array.shape, dtype=np.int32)
    leading_shape = labels_array.shape[: labels_array.ndim - spatial_rank]
    for leading_index in np.ndindex(leading_shape):
        output[leading_index] = skimage.measure.label(labels_array[leading_index])
    return output


def watershed_regionprops_stats(labels: np.ndarray) -> tuple[int, float]:
    """Return object count and mean area over skimage-supported spatial labels."""
    labels_array = np.asarray(labels)
    spatial_rank = min(labels_array.ndim, 3)
    if labels_array.ndim == spatial_rank:
        object_count = int(np.max(labels_array, initial=0))
        if object_count == 0:
            return (0, 0.0)
        return (
            object_count,
            float(np.count_nonzero(labels_array) / object_count),
        )
    else:
        leading_shape = labels_array.shape[: labels_array.ndim - spatial_rank]
        area_batches: list[np.ndarray] = []
        for leading_index in np.ndindex(leading_shape):
            batch_areas = watershed_label_areas(labels_array[leading_index])
            if batch_areas.size:
                area_batches.append(batch_areas)
        areas = np.concatenate(area_batches) if area_batches else np.asarray(())
    return (int(areas.size), float(np.mean(areas) if areas.size else 0.0))


def watershed_label_areas(labels: np.ndarray) -> np.ndarray:
    """Return positive-label voxel counts for one spatial label domain."""
    positive_labels = np.asarray(labels).reshape(-1)
    positive_labels = positive_labels[positive_labels > 0]
    if positive_labels.size == 0:
        return np.asarray((), dtype=np.int64)
    counts = np.bincount(positive_labels.astype(np.int64, copy=False))
    return counts[counts > 0]


def watershed_profile_enabled() -> bool:
    return RuntimeProfileLogger.enabled()


def log_watershed_profile(label: str, seconds: float, **fields: object) -> None:
    RuntimeProfileLogger.log(logger, label, seconds, **fields)


@dataclass(frozen=True)
class WatershedProfiler:
    """Bound profiler for CellProfiler watershed execution phases."""

    def record(self, label: str, started_at: float, **fields: object) -> None:
        log_watershed_profile(label, time.perf_counter() - started_at, **fields)

    def record_factor(
        self, label: str, started_at: float, factor: int, **fields: object
    ) -> None:
        self.record(label, started_at, factor=factor, **fields)

    def record_method(
        self, label: str, started_at: float, method: WatershedMethod, **fields: object
    ) -> None:
        self.record(label, started_at, method=method.value, **fields)


@dataclass(frozen=True, slots=True)
class WatershedFactorProfiler:
    """Profiler projection for a watershed phase family sharing one factor."""

    profiler: WatershedProfiler
    factor: int
    ndim: int

    def record(self, label: str, started_at: float, **fields: object) -> None:
        self.profiler.record(label, started_at, factor=self.factor, **fields)

    def record_downsample(self, label: str, started_at: float) -> None:
        self.record(label, started_at, ndim=self.ndim)


class WatershedDeclumpMethod(str, Enum):
    """CellProfiler watershed declumping priority family."""

    SHAPE = "shape"
    INTENSITY = "intensity"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class WatershedInputTopology:
    """Nominal owner for method-dependent Watershed input roles."""

    watershed_method: WatershedMethod
    declump_method: WatershedDeclumpMethod

    @classmethod
    def from_values(
        cls,
        watershed_method: WatershedMethod = WatershedMethod.DISTANCE,
        declump_method: WatershedDeclumpMethod = WatershedDeclumpMethod.SHAPE,
    ) -> "WatershedInputTopology":
        return cls(
            watershed_method=watershed_method,
            declump_method=declump_method,
        )

    @property
    def requires_markers(self) -> bool:
        return self.watershed_method is WatershedMethod.MARKERS

    @property
    def requires_intensity_image(self) -> bool:
        return (
            self.watershed_method is WatershedMethod.INTENSITY
            or self.declump_method is WatershedDeclumpMethod.INTENSITY
        )

    def special_inputs(
        self,
        values: tuple[object, ...],
    ) -> "WatershedSpecialInputs":
        """Assign ordered auxiliary images to this topology's exact roles."""

        required_count = int(self.requires_markers) + int(self.requires_intensity_image)
        if len(values) not in (required_count, required_count + 1):
            raise ValueError(
                f"Watershed {self.watershed_method.value}/{self.declump_method.value} "
                f"topology requires {required_count} special image input(s) and "
                f"accepts one trailing mask; got {len(values)}."
            )
        position = 0
        markers = None
        if self.requires_markers:
            markers = values[position]
            position += 1
        intensity_image = None
        if self.requires_intensity_image:
            intensity_image = values[position]
            position += 1
        mask = values[position] if position < len(values) else None
        return WatershedSpecialInputs(
            intensity_image=intensity_image,
            markers=markers,
            mask=mask,
        )


class WatershedSeedMethod(str, Enum):
    """CellProfiler watershed seed detector family."""

    LOCAL = "local"
    REGIONAL = "regional"
    CONNECTED_COMPONENTS = "connected_components"


class WatershedRuntimeFamily(str, Enum):
    """CellProfiler Watershed implementation family selected by module revision."""

    CELLPROFILER4 = "cellprofiler4"
    LIBRARY = "library"


@dataclass(frozen=True, slots=True)
class WatershedSpecialInputs:
    """Method-resolved auxiliary images for one Watershed invocation."""

    intensity_image: object | None
    markers: object | None
    mask: object | None


@dataclass
class WatershedStats:
    """Watershed object-count measurement row."""

    slice_index: int
    object_count: int
    mean_area: float


@dataclass(frozen=True, slots=True)
class WatershedInvocationRequest:
    """Single typed owner for the public Watershed behavior surface."""

    image: np.ndarray
    watershed_method: WatershedMethod = WatershedMethod.DISTANCE
    declump_method: WatershedDeclumpMethod = WatershedDeclumpMethod.SHAPE
    seed_method: WatershedSeedMethod = WatershedSeedMethod.LOCAL
    use_advanced_settings: bool = True
    max_seeds: int = -1
    downsample: int = 1
    min_distance: int = 1
    min_intensity: float = 0.0
    footprint: int = 8
    connectivity: int = 1
    compactness: float = 0.0
    exclude_border: bool = False
    watershed_line: bool = False
    gaussian_sigma: float = 0.0
    structuring_element: StructuringElementInput = StructuringElement.DISK
    structuring_element_size: StructuringElementSize = 1

    @property
    def topology(self) -> WatershedInputTopology:
        """Return the input topology selected by public behavior settings."""

        return WatershedInputTopology.from_values(
            self.watershed_method,
            self.declump_method,
        )

    def execute(
        self,
        runtime_family: WatershedRuntimeFamily,
        special_inputs: tuple[object, ...] = (),
    ) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
        """Apply this request through its declaration-owned runtime family."""

        resolved_inputs = self.topology.special_inputs(special_inputs)
        image_array = watershed_image_array(self.image, parameter_name="image")
        intensity_image_array = (
            None
            if resolved_inputs.intensity_image is None
            else watershed_image_array(
                resolved_inputs.intensity_image,
                parameter_name="intensity_image",
            )
        )
        if (
            intensity_image_array is not None
            and intensity_image_array.shape != image_array.shape
        ):
            raise ValueError(
                "Watershed reference image shape must match the input image shape; "
                f"got intensity_image={intensity_image_array.shape!r}, "
                f"image={image_array.shape!r}."
            )
        mask_array = (
            None
            if resolved_inputs.mask is None
            else watershed_image_array(resolved_inputs.mask, parameter_name="mask")
        )
        markers_array = (
            None
            if resolved_inputs.markers is None
            else watershed_image_array(
                resolved_inputs.markers,
                parameter_name="markers",
            ).astype(np.int32, copy=False)
        )
        if not np.array_equal(image_array, image_array.astype(bool)):
            raise ValueError("Watershed expects a thresholded image as input.")
        parameters = WatershedParameters.from_settings(
            image_ndim=image_array.ndim,
            watershed_method=self.watershed_method,
            declump_method=self.declump_method,
            seed_method=self.seed_method,
            use_advanced_settings=self.use_advanced_settings,
            max_seeds=self.max_seeds,
            downsample=self.downsample,
            min_distance=self.min_distance,
            min_intensity=self.min_intensity,
            footprint=self.footprint,
            connectivity=self.connectivity,
            compactness=self.compactness,
            exclude_border=self.exclude_border,
            watershed_line=self.watershed_line,
            gaussian_sigma=self.gaussian_sigma,
            structuring_element=self.structuring_element,
            structuring_element_size=self.structuring_element_size,
        )
        labels = WatershedRuntimeStrategy.for_enum_member(runtime_family).labels(
            image_array,
            intensity_image_array,
            markers_array,
            mask_array,
            parameters,
        )
        object_count, mean_area = watershed_regionprops_stats(labels)
        stats = WatershedStats(
            slice_index=0,
            object_count=object_count,
            mean_area=mean_area,
        )
        return (
            self.image,
            DataclassMeasurementColumnarRows((stats,), row_type=WatershedStats),
            SourceImageObjectLabelBuildRequest(
                image=self.image,
                labels=labels.astype(np.int32, copy=False),
                declared_object_count=object_count,
                declared_object_ids=tuple(range(1, object_count + 1)),
            ).payload(),
        )


def watershed_image_array(value: object, *, parameter_name: str) -> np.ndarray:
    """Return concrete NumPy image data for Watershed computation."""
    array = np.asarray(image_payload_data(value))
    if array.ndim == 0:
        raise TypeError(
            f"Watershed {parameter_name} requires array-like image data, got {type(value).__name__}."
        )
    return array


@dataclass(frozen=True, slots=True)
class WatershedInputs:
    image: np.ndarray
    intensity_image: np.ndarray | None
    binary: np.ndarray
    mask: np.ndarray
    markers: np.ndarray | None

    def required_intensity_image(self) -> np.ndarray:
        if self.intensity_image is None:
            raise ValueError(
                "Watershed intensity mode requires a declared reference image."
            )
        return self.intensity_image


@dataclass(frozen=True, slots=True)
class WatershedSegmentationSurface:
    watershed_input_image: np.ndarray
    seed_image: np.ndarray | None
    distance_image: np.ndarray | None
    markers: np.ndarray


@dataclass(frozen=True, slots=True)
class WatershedComputationImages:
    input_image: np.ndarray
    mask: np.ndarray | None
    markers: np.ndarray | None
    distance: np.ndarray | None


@dataclass(frozen=True, slots=True)
class WatershedParameters:
    """Normalized Watershed settings consumed by runtime strategies."""

    BASIC_SEED_METHOD: ClassVar[WatershedSeedMethod] = WatershedSeedMethod.LOCAL
    BASIC_MAX_SEEDS: ClassVar[int] = -1
    BASIC_MIN_DISTANCE: ClassVar[int] = 1
    BASIC_MIN_INTENSITY: ClassVar[float] = 0.0
    BASIC_CONNECTIVITY: ClassVar[int] = 1
    BASIC_COMPACTNESS: ClassVar[float] = 0.0
    BASIC_WATERSHED_LINE: ClassVar[bool] = False
    BASIC_GAUSSIAN_SIGMA: ClassVar[float] = 0.0
    method: WatershedMethod
    declump_method: WatershedDeclumpMethod
    seed_method: WatershedSeedMethod
    use_advanced_settings: bool
    max_seeds: int
    downsample: int
    min_distance: int
    min_intensity: float
    footprint: int
    connectivity: int
    compactness: float
    exclude_border: bool
    watershed_line: bool
    gaussian_sigma: float
    structuring_element: np.ndarray

    @classmethod
    def basic_default_values(cls) -> dict[str, object]:
        """Return source-verified CellProfiler basic-mode defaults."""
        return {
            "seed_method": cls.BASIC_SEED_METHOD,
            "max_seeds": cls.BASIC_MAX_SEEDS,
            "min_distance": cls.BASIC_MIN_DISTANCE,
            "min_intensity": cls.BASIC_MIN_INTENSITY,
            "connectivity": cls.BASIC_CONNECTIVITY,
            "compactness": cls.BASIC_COMPACTNESS,
            "watershed_line": cls.BASIC_WATERSHED_LINE,
            "gaussian_sigma": cls.BASIC_GAUSSIAN_SIGMA,
        }

    @classmethod
    def from_settings(
        cls,
        *,
        image_ndim: int,
        watershed_method: WatershedMethod,
        declump_method: WatershedDeclumpMethod,
        seed_method: WatershedSeedMethod,
        use_advanced_settings: bool,
        max_seeds: int,
        downsample: int,
        min_distance: int,
        min_intensity: float,
        footprint: int,
        connectivity: int,
        compactness: float,
        exclude_border: bool,
        watershed_line: bool,
        gaussian_sigma: float,
        structuring_element: StructuringElement,
        structuring_element_size: int,
    ) -> "WatershedParameters":
        """Return the single normalized parameter contract for watershed execution."""
        structuring_element_array = adapt_structuring_element_rank(
            build_structuring_element(
                structuring_element,
                structuring_element_size,
            ),
            image_ndim,
        )
        if structuring_element_array.ndim != image_ndim:
            raise ValueError(
                f"Watershed structuring element dimensionality must match the image; got structuring element ndim={structuring_element_array.ndim} for image ndim={image_ndim}."
            )
        if not use_advanced_settings:
            return cls(
                method=watershed_method,
                declump_method=declump_method,
                seed_method=cls.BASIC_SEED_METHOD,
                use_advanced_settings=False,
                max_seeds=cls.BASIC_MAX_SEEDS,
                downsample=downsample,
                min_distance=cls.BASIC_MIN_DISTANCE,
                min_intensity=cls.BASIC_MIN_INTENSITY,
                footprint=footprint,
                connectivity=cls.BASIC_CONNECTIVITY,
                compactness=cls.BASIC_COMPACTNESS,
                exclude_border=exclude_border,
                watershed_line=cls.BASIC_WATERSHED_LINE,
                gaussian_sigma=cls.BASIC_GAUSSIAN_SIGMA,
                structuring_element=structuring_element_array,
            )
        return cls(
            method=watershed_method,
            declump_method=declump_method,
            seed_method=seed_method,
            use_advanced_settings=True,
            max_seeds=max_seeds,
            downsample=downsample,
            min_distance=min_distance,
            min_intensity=min_intensity,
            footprint=footprint,
            connectivity=connectivity,
            compactness=compactness,
            exclude_border=exclude_border,
            watershed_line=watershed_line,
            gaussian_sigma=gaussian_sigma,
            structuring_element=structuring_element_array,
        )


class WatershedSeedStrategy(
    EnumKeyedStrategyMixin[WatershedSeedMethod], ABC, metaclass=AutoRegisterMeta
):
    """Build seed labels for non-marker Watershed modes."""

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedSeedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        """Return marker labels for the supplied seed image."""


class LocalWatershedSeedStrategy(WatershedSeedStrategy):
    strategy_key = WatershedSeedMethod.LOCAL

    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        from skimage.feature import peak_local_max
        from scipy.ndimage import label as ndi_label
        from skimage.morphology import binary_dilation

        coords = peak_local_max(
            seed_image,
            min_distance=parameters.min_distance,
            footprint=np.ones((parameters.footprint,) * inputs.image.ndim),
            threshold_rel=parameters.min_intensity,
            num_peaks=parameters.max_seeds if parameters.max_seeds != -1 else np.inf,
            exclude_border=False,
        )
        seeds = np.zeros(seed_image.shape, dtype=bool)
        seeds[tuple(coords.T)] = True
        seeds = binary_dilation(seeds, parameters.structuring_element)
        markers, _count = ndi_label(seeds)
        return markers.astype(np.int32, copy=False)


class RegionalWatershedSeedStrategy(WatershedSeedStrategy):
    strategy_key = WatershedSeedMethod.REGIONAL

    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        del inputs
        import mahotas
        from scipy.ndimage import label as ndi_label
        from skimage.morphology import binary_dilation

        maxima_footprint = np.ones((parameters.footprint,) * seed_image.ndim)
        seeds = mahotas.regmax(seed_image, maxima_footprint)
        seeds = binary_dilation(seeds, parameters.structuring_element)
        markers, _count = ndi_label(seeds)
        return markers.astype(np.int32, copy=False)


class ConnectedComponentsWatershedSeedStrategy(WatershedSeedStrategy):
    strategy_key = WatershedSeedMethod.CONNECTED_COMPONENTS

    def markers(
        self,
        seed_image: np.ndarray,
        inputs: WatershedInputs,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        del seed_image, parameters
        from scipy.ndimage import label as ndi_label

        markers, _count = ndi_label(inputs.mask)
        return markers.astype(np.int32, copy=False)


class WatershedDeclumpStrategy(
    EnumKeyedStrategyMixin[WatershedDeclumpMethod], ABC, metaclass=AutoRegisterMeta
):
    """Build the watershed priority surface for one declumping family."""

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedDeclumpMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def priority_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray:
        """Return the skimage watershed input image."""


class ShapeWatershedDeclumpStrategy(WatershedDeclumpStrategy):
    strategy_key = WatershedDeclumpMethod.SHAPE

    def priority_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray:
        del inputs
        if computation.distance is None:
            raise ValueError("Shape declumping requires a distance image.")
        watershed_input = -computation.distance
        return watershed_input - watershed_input.min()


class IntensityWatershedDeclumpStrategy(WatershedDeclumpStrategy):
    strategy_key = WatershedDeclumpMethod.INTENSITY

    def priority_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray:
        del computation
        return 1.0 - inputs.required_intensity_image()


class NoneWatershedDeclumpStrategy(WatershedDeclumpStrategy):
    strategy_key = WatershedDeclumpMethod.NONE

    def priority_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray:
        del inputs, computation
        raise ValueError("No-declump watershed should label the binary input directly.")


class WatershedMethodStrategy(
    EnumKeyedStrategyMixin[WatershedMethod], ABC, metaclass=AutoRegisterMeta
):
    """Build seed labels for one CellProfiler watershed seed source."""

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def seed_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray | None:
        """Return the image used to derive watershed seeds."""

    def markers(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
        seed_strategy: WatershedSeedStrategy,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        seed_image = self.seed_image(inputs, computation)
        if seed_image is None:
            raise ValueError(f"{type(self).__name__} did not provide a seed image.")
        return seed_strategy.markers(seed_image, inputs, parameters)


class DistanceWatershedMethodStrategy(WatershedMethodStrategy):
    strategy_key = WatershedMethod.DISTANCE

    def seed_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray:
        del inputs
        if computation.distance is None:
            raise ValueError(
                "Distance watershed seed method requires a distance image."
            )
        return computation.distance


class IntensityWatershedMethodStrategy(WatershedMethodStrategy):
    strategy_key = WatershedMethod.INTENSITY

    def seed_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray:
        del computation
        return inputs.required_intensity_image()


class MarkerWatershedMethodStrategy(WatershedMethodStrategy):
    strategy_key = WatershedMethod.MARKERS

    def seed_image(
        self, inputs: WatershedInputs, computation: WatershedComputationImages
    ) -> np.ndarray | None:
        del inputs, computation
        return None

    def markers(
        self,
        inputs: WatershedInputs,
        computation: WatershedComputationImages,
        seed_strategy: WatershedSeedStrategy,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        del computation, seed_strategy
        if inputs.markers is None:
            raise ValueError("Watershed marker mode requires marker labels.")
        markers = object_label_dense_array(inputs.markers, dtype=np.int32)
        if markers.shape != inputs.image.shape:
            raise ValueError(
                f"Watershed marker shape must match the input image shape; got markers={markers.shape!r}, image={inputs.image.shape!r}."
            )
        from skimage.morphology import dilation

        return dilation(markers, footprint=parameters.structuring_element).astype(
            np.int32, copy=False
        )


class WatershedRuntimeStrategy(
    EnumKeyedStrategyMixin[WatershedRuntimeFamily], ABC, metaclass=AutoRegisterMeta
):
    """Execute one nominal CellProfiler Watershed implementation family."""

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedRuntimeFamily | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def labels(
        self,
        image: np.ndarray,
        intensity_image: np.ndarray | None,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        """Return object labels using one CellProfiler runtime family."""

    def segmentation_surface(
        self, inputs: WatershedInputs, parameters: WatershedParameters
    ) -> WatershedSegmentationSurface:
        """Build the CellProfiler watershed priority image and marker labels."""
        from scipy.ndimage import distance_transform_edt
        from skimage.filters import gaussian

        needs_distance = (
            parameters.declump_method is WatershedDeclumpMethod.SHAPE
            or parameters.method is WatershedMethod.DISTANCE
        )
        distance = (
            distance_transform_edt(
                gaussian(inputs.image, sigma=parameters.gaussian_sigma)
            )
            if needs_distance
            else None
        )
        computation = WatershedComputationImages(
            input_image=inputs.image,
            mask=inputs.mask,
            markers=inputs.markers,
            distance=distance,
        )
        method_strategy = WatershedMethodStrategy.for_enum_member(parameters.method)
        seed_image = method_strategy.seed_image(inputs, computation)
        markers = method_strategy.markers(
            inputs,
            computation,
            WatershedSeedStrategy.for_enum_member(parameters.seed_method),
            parameters,
        )
        return WatershedSegmentationSurface(
            watershed_input_image=WatershedDeclumpStrategy.for_enum_member(
                parameters.declump_method
            ).priority_image(inputs, computation),
            seed_image=seed_image,
            distance_image=distance,
            markers=markers,
        )


class CellProfiler4InitialWatershedStrategy(
    EnumKeyedStrategyMixin[WatershedMethod], ABC, metaclass=AutoRegisterMeta
):
    """Build the CP4 module's initial watershed labels before advanced refinement."""

    __registry_key__ = WATERSHED_STRATEGY_REGISTRY_KEY
    __skip_if_no_key__ = True
    strategy_key: ClassVar[WatershedMethod | None] = None
    strategy_label: ClassVar[str | None] = None

    @abstractmethod
    def labels(
        self,
        image: np.ndarray,
        intensity_image: np.ndarray | None,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the initial labels and the image-domain mask source."""


class CellProfiler4DistanceInitialWatershedStrategy(
    CellProfiler4InitialWatershedStrategy
):
    strategy_key = WatershedMethod.DISTANCE

    def labels(
        self,
        image: np.ndarray,
        intensity_image: np.ndarray | None,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        del intensity_image, markers, mask
        import mahotas
        import scipy.ndimage
        import skimage.filters
        import skimage.transform

        profiler = WatershedProfiler()
        factor_profiler = WatershedFactorProfiler(
            profiler=profiler, factor=parameters.downsample, ndim=image.ndim
        )
        total_started_at = time.perf_counter()
        input_shape = image.shape
        factor = parameters.downsample
        x_data = image
        if factor > 1:
            phase_started_at = time.perf_counter()
            factors = watershed_xy_downsample_factors(image.ndim, factor)
            x_data = skimage.transform.downscale_local_mean(x_data, factors)
            factor_profiler.record_downsample(
                "watershed_cp4_distance_downsample", phase_started_at
            )
        phase_started_at = time.perf_counter()
        threshold = skimage.filters.threshold_otsu(x_data)
        x_data = x_data > threshold
        factor_profiler.record("watershed_cp4_distance_threshold", phase_started_at)
        phase_started_at = time.perf_counter()
        distance = scipy.ndimage.distance_transform_edt(x_data)
        factor_profiler.record("watershed_cp4_distance_edt", phase_started_at)
        phase_started_at = time.perf_counter()
        distance = mahotas.stretch(distance)
        surface = distance.max() - distance
        factor_profiler.record("watershed_cp4_distance_surface", phase_started_at)
        phase_started_at = time.perf_counter()
        peak_footprint = np.ones((parameters.footprint,) * image.ndim)
        seed_connectivity = np.ones((16,) * image.ndim)
        seed_markers, marker_count, _peaks = (
            CellProfiler4DistanceMarkerBackendStrategy.for_memory_type(
                MemoryType.NUMPY
            ).distance_markers(distance, peak_footprint, seed_connectivity)
        )
        factor_profiler.record(
            "watershed_cp4_distance_markers", phase_started_at, seeds=marker_count
        )
        phase_started_at = time.perf_counter()
        y_data = mahotas.cwatershed(surface, seed_markers)
        y_data *= x_data
        factor_profiler.record("watershed_cp4_distance_cwatershed", phase_started_at)
        if factor > 1:
            phase_started_at = time.perf_counter()
            y_data = watershed_resize_labels(y_data, input_shape)
            x_data = image > threshold
            factor_profiler.record("watershed_cp4_distance_upsample", phase_started_at)
        factor_profiler.record("watershed_cp4_distance_initial_total", total_started_at)
        return (y_data, x_data)


class CellProfiler4MarkerInitialWatershedStrategy(
    CellProfiler4InitialWatershedStrategy
):
    strategy_key = WatershedMethod.MARKERS

    def labels(
        self,
        image: np.ndarray,
        intensity_image: np.ndarray | None,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        del intensity_image
        if markers is None:
            raise ValueError("CellProfiler 4 marker watershed requires markers.")
        if parameters.compactness != 0.0:
            raise NotImplementedError(
                "CellProfiler 4 marker watershed compactness requires legacy compact watershed semantics."
            )
        if parameters.watershed_line:
            raise NotImplementedError(
                "CellProfiler 4 marker watershed lines require legacy watershed-line semantics."
            )
        image_array = np.asarray(image)
        markers_array = np.asarray(markers)
        mask_array = (
            image_array.astype(bool, copy=False)
            if mask is None
            else np.asarray(mask, dtype=bool)
        )
        if (
            np.issubdtype(markers_array.dtype, np.integer)
            and markers_array.shape == image_array.shape == mask_array.shape
            and not np.any(mask_array)
        ):
            return (
                np.zeros(markers_array.shape, dtype=markers_array.dtype),
                image,
            )
        import skimage.segmentation

        y_data = skimage.segmentation.watershed(
            image=image_array,
            markers=markers_array,
            mask=mask_array,
            connectivity=parameters.connectivity,
            compactness=parameters.compactness,
            watershed_line=parameters.watershed_line,
        )
        return (y_data, image)


class CellProfiler4WatershedRuntimeStrategy(WatershedRuntimeStrategy):
    """CellProfiler 4.2 module-level Watershed semantics."""

    strategy_key = WatershedRuntimeFamily.CELLPROFILER4

    def labels(
        self,
        image: np.ndarray,
        intensity_image: np.ndarray | None,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        import scipy.ndimage
        import skimage.feature
        import skimage.filters
        import skimage.morphology
        import skimage.segmentation

        profiler = WatershedProfiler()
        phase_started_at = time.perf_counter()
        y_data, x_data = CellProfiler4InitialWatershedStrategy.for_enum_member(
            parameters.method
        ).labels(image, intensity_image, markers, mask, parameters)
        profiler.record_method(
            "watershed_cp4_initial", phase_started_at, parameters.method
        )
        if parameters.use_advanced_settings:
            if parameters.structuring_element.ndim != image.ndim:
                raise ValueError(
                    f"Watershed structuring element dimensionality must match the image; got structuring element ndim={parameters.structuring_element.ndim} for image ndim={image.ndim}."
                )
            phase_started_at = time.perf_counter()
            peak_image = scipy.ndimage.distance_transform_edt(y_data > 0)
            profiler.record_method(
                "watershed_cp4_peak_distance", phase_started_at, parameters.method
            )
            if parameters.declump_method is WatershedDeclumpMethod.SHAPE:
                watershed_image = -peak_image
                watershed_image -= watershed_image.min()
            else:
                if intensity_image is None:
                    raise ValueError(
                        "CellProfiler 4 intensity declumping requires a declared reference image."
                    )
                watershed_image = 1.0 - intensity_image.astype(float, copy=False)
            phase_started_at = time.perf_counter()
            watershed_image = skimage.filters.gaussian(
                watershed_image, sigma=parameters.gaussian_sigma
            )
            profiler.record_method(
                "watershed_cp4_gaussian", phase_started_at, parameters.method
            )
            phase_started_at = time.perf_counter()
            seed_coords = skimage.feature.peak_local_max(
                peak_image,
                min_distance=parameters.min_distance,
                threshold_rel=parameters.min_intensity,
                exclude_border=parameters.exclude_border,
                num_peaks=(
                    parameters.max_seeds if parameters.max_seeds != -1 else np.inf
                ),
            )
            profiler.record_method(
                "watershed_cp4_peak_local_max",
                phase_started_at,
                parameters.method,
                seeds=len(seed_coords),
            )
            phase_started_at = time.perf_counter()
            seeds = np.zeros_like(peak_image, dtype=bool)
            seeds[tuple(seed_coords.T)] = True
            seeds = skimage.morphology.binary_dilation(
                seeds, parameters.structuring_element
            )
            number_objects = int(np.max(watershed_connected_components(y_data)))
            seeds_dtype = (
                np.uint16 if number_objects < np.iinfo(np.uint16).max else np.uint32
            )
            seeds = scipy.ndimage.label(seeds)[0]
            advanced_markers = np.zeros_like(seeds, dtype=seeds_dtype)
            advanced_markers[seeds > 0] = -seeds[seeds > 0]
            profiler.record_method(
                "watershed_cp4_markers",
                phase_started_at,
                parameters.method,
                objects=number_objects,
            )
            phase_started_at = time.perf_counter()
            watershed_boundaries = skimage.segmentation.watershed(
                image=watershed_image,
                markers=advanced_markers,
                mask=x_data != 0,
                connectivity=parameters.connectivity,
            )
            profiler.record_method(
                "watershed_cp4_segmentation", phase_started_at, parameters.method
            )
            phase_started_at = time.perf_counter()
            y_data = watershed_boundaries.copy()
            zeros = np.where(y_data == 0)
            y_data += np.abs(np.min(y_data)) + 1
            y_data[zeros] = 0
            profiler.record_method(
                "watershed_cp4_relabel_prepare", phase_started_at, parameters.method
            )
        phase_started_at = time.perf_counter()
        labels = watershed_connected_components(y_data)
        profiler.record_method(
            "watershed_cp4_final_label", phase_started_at, parameters.method
        )
        return labels


class LibraryWatershedRuntimeStrategy(WatershedRuntimeStrategy):
    """CellProfiler library-style Watershed semantics."""

    strategy_key = WatershedRuntimeFamily.LIBRARY

    def labels(
        self,
        image: np.ndarray,
        intensity_image: np.ndarray | None,
        markers: np.ndarray | None,
        mask: np.ndarray | None,
        parameters: WatershedParameters,
    ) -> np.ndarray:
        from skimage.segmentation import watershed as skimage_watershed
        from skimage.segmentation import clear_border
        from skimage.transform import downscale_local_mean
        from scipy.ndimage import label as ndi_label

        binary = image.astype(bool, copy=False)
        mask_array = binary.astype(bool) if mask is None else np.asarray(mask) > 0
        if mask_array.shape != image.shape:
            raise ValueError(
                f"Watershed mask shape must match the input image shape; got mask={mask_array.shape!r}, image={image.shape!r}."
            )
        input_shape = binary.shape
        working_image = binary
        working_mask = mask_array
        working_markers = (
            None
            if markers is None
            else object_label_dense_array(markers, dtype=np.int32)
        )
        working_intensity_image = (
            None if intensity_image is None else np.asarray(intensity_image)
        )
        if parameters.downsample > 1:
            factors = watershed_xy_downsample_factors(
                binary.ndim, parameters.downsample
            )
            working_image = downscale_local_mean(binary.astype(np.float32), factors)
            if working_intensity_image is not None:
                working_intensity_image = downscale_local_mean(
                    working_intensity_image.astype(np.float32), factors
                )
            if working_mask is not None:
                working_mask = (
                    downscale_local_mean(working_mask.astype(np.float32), factors) > 0
                )
            if working_markers is not None:
                working_markers = downscale_local_mean(
                    working_markers.astype(np.float32), factors
                )
        working_inputs = WatershedInputs(
            image=working_image,
            intensity_image=working_intensity_image,
            binary=working_image,
            mask=working_mask,
            markers=working_markers,
        )
        if parameters.declump_method is WatershedDeclumpMethod.NONE:
            labels, _count = ndi_label(working_image)
            labels = np.where(working_mask, labels, 0)
        else:
            surface = self.segmentation_surface(working_inputs, parameters)
            labels = skimage_watershed(
                surface.watershed_input_image,
                markers=surface.markers,
                mask=working_mask,
                connectivity=parameters.connectivity,
                compactness=parameters.compactness,
                watershed_line=parameters.watershed_line,
            )
        if parameters.downsample > 1:
            labels = watershed_resize_labels(labels, input_shape)
        if parameters.exclude_border:
            labels = clear_border(labels)
        return DenseObjectLabelConsecutiveRelabelingStrategy.for_labels(labels).relabel(
            labels, dtype=np.int32
        )


class LegacyWatershedBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Legacy watershed operations keyed by OpenHCS memory type."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True
    prefer_fast: ClassVar[bool]

    def validated_request(
        self,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray = 1,
    ) -> "LegacyWatershedRequest":
        """Build a validated legacy watershed request for this backend family."""
        return LegacyWatershedRequest.from_inputs(
            image,
            markers=markers,
            mask=mask,
            connectivity=connectivity,
            prefer_fast=self.prefer_fast,
        )


class NumpyLegacyWatershedBackendStrategy(LegacyWatershedBackendStrategy):
    """NumPy-memory reference legacy watershed backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    is_default_backend = False
    prefer_fast = False


class NumbaNumpyLegacyWatershedBackendStrategy(LegacyWatershedBackendStrategy):
    """NumPy-memory legacy watershed backend with required Numba acceleration."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True
    prefer_fast = True

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        markers = np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0]], dtype=np.int32)
        mask = np.ones(markers.shape, dtype=np.bool_)
        self.validated_request(
            image, markers=markers, mask=mask, connectivity=1
        ).execute()


class CellProfiler4DistanceMarkerBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Build CellProfiler 4 distance-watershed markers through a typed backend."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Return seed markers, marker count, and the regional-maxima mask."""


class MahotasCellProfiler4DistanceMarkerBackendStrategy(
    CellProfiler4DistanceMarkerBackendStrategy
):
    """Reference CellProfiler 4 marker backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        import mahotas

        peaks = mahotas.regmax(distance, peak_footprint)
        seed_markers, marker_count = mahotas.label(peaks, seed_connectivity)
        return (seed_markers, int(marker_count), peaks)


class NumbaCellProfiler4DistanceMarkerBackendStrategy(
    CellProfiler4DistanceMarkerBackendStrategy
):
    """Exact numba backend for CellProfiler 4 regional-maxima markers."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def distance_markers(
        self,
        distance: np.ndarray,
        peak_footprint: np.ndarray,
        seed_connectivity: np.ndarray,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        import scipy.ndimage

        distance_array = np.ascontiguousarray(distance)
        peak_footprint_array = np.asarray(peak_footprint, dtype=bool)
        if distance_array.ndim != 3 or peak_footprint_array.ndim != 3:
            return MahotasCellProfiler4DistanceMarkerBackendStrategy().distance_markers(
                distance_array, peak_footprint_array, seed_connectivity
            )
        local_maxima = (
            distance_array
            == scipy.ndimage.maximum_filter(
                distance_array,
                footprint=peak_footprint_array,
                mode=NDIMAGE_CONSTANT_MODE,
                cval=0,
            )
        ) & (distance_array > 0)
        peaks = _cellprofiler4_regional_maxima_from_candidates_3d_numba(
            distance_array,
            np.ascontiguousarray(local_maxima),
            _footprint_offsets_3d(peak_footprint_array),
        )
        sparse_markers = _sparse_connected_peak_markers_3d(peaks, seed_connectivity)
        if sparse_markers is not None:
            seed_markers, marker_count = sparse_markers
            return (seed_markers, marker_count, peaks)
        import mahotas

        seed_markers, marker_count = mahotas.label(peaks, seed_connectivity)
        return (seed_markers, int(marker_count), peaks)

    def prepare_backend(self) -> None:
        distance = np.zeros((8, 16, 16), dtype=np.uint8)
        distance[2:6, 4:12, 4:12] = 1
        distance[3:5, 6:10, 6:10] = 2
        footprint = np.ones((4, 4, 4), dtype=bool)
        connectivity = np.ones((4, 4, 4), dtype=bool)
        self.distance_markers(distance, footprint, connectivity)


def _sparse_connected_peak_markers_3d(
    peaks: np.ndarray, seed_connectivity: np.ndarray
) -> tuple[np.ndarray, int] | None:
    """Label sparse 3-D peak masks under dense CellProfiler seed connectivity."""
    peaks_array = np.ascontiguousarray(peaks, dtype=bool)
    connectivity_array = np.asarray(seed_connectivity, dtype=bool)
    if peaks_array.ndim != 3 or connectivity_array.ndim != 3:
        return None
    if not bool(np.all(connectivity_array)):
        return None
    coords = np.ascontiguousarray(np.argwhere(peaks_array), dtype=np.int64)
    return _sparse_connected_peak_markers_3d_numba(
        coords,
        int(peaks_array.shape[0]),
        int(peaks_array.shape[1]),
        int(peaks_array.shape[2]),
        int(connectivity_array.shape[0]) // 2,
        int(connectivity_array.shape[1]) // 2,
        int(connectivity_array.shape[2]) // 2,
    )


@njit(cache=True)
def _sparse_connected_peak_markers_3d_numba(
    coords: np.ndarray,
    z_count: int,
    y_count: int,
    x_count: int,
    radius_z: int,
    radius_y: int,
    radius_x: int,
) -> tuple[np.ndarray, int]:
    labels = np.zeros((z_count, y_count, x_count), dtype=np.int32)
    point_count = coords.shape[0]
    if point_count == 0:
        return (labels, 0)
    parents = np.empty(point_count, dtype=np.int64)
    for index in range(point_count):
        parents[index] = index
    for left in range(point_count):
        left_root = _sparse_peak_find_root(parents, left)
        left_z = coords[left, 0]
        left_y = coords[left, 1]
        left_x = coords[left, 2]
        for right in range(left + 1, point_count):
            delta_z = left_z - coords[right, 0]
            if delta_z < 0:
                delta_z = -delta_z
            if delta_z > radius_z:
                continue
            delta_y = left_y - coords[right, 1]
            if delta_y < 0:
                delta_y = -delta_y
            if delta_y > radius_y:
                continue
            delta_x = left_x - coords[right, 2]
            if delta_x < 0:
                delta_x = -delta_x
            if delta_x > radius_x:
                continue
            right_root = _sparse_peak_find_root(parents, right)
            if left_root != right_root:
                parents[right_root] = left_root
    root_labels = np.zeros(point_count, dtype=np.int32)
    label_count = 0
    for index in range(point_count):
        root = _sparse_peak_find_root(parents, index)
        label = root_labels[root]
        if label == 0:
            label_count += 1
            label = label_count
            root_labels[root] = label
        labels[coords[index, 0], coords[index, 1], coords[index, 2]] = label
    return (labels, label_count)


@njit(cache=True)
def _sparse_peak_find_root(parents: np.ndarray, index: int) -> int:
    root = index
    while parents[root] != root:
        root = parents[root]
    while parents[index] != index:
        parent = parents[index]
        parents[index] = root
        index = parent
    return root


def cellprofiler_legacy_watershed(
    image: np.ndarray,
    *,
    markers: np.ndarray,
    mask: np.ndarray,
    connectivity: int | np.ndarray = 1,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> np.ndarray:
    """Run CellProfiler 4.2/skimage 0.18 watershed semantics."""
    return (
        LegacyWatershedBackendStrategy.for_memory_type(
            MemoryType.NUMPY, backend_provider=backend_provider
        )
        .validated_request(image, markers=markers, mask=mask, connectivity=connectivity)
        .execute()
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("topology_inputs")
@callable_request(WatershedInvocationRequest)
def watershed_library(
    request: WatershedInvocationRequest,
    *,
    topology_inputs: tuple[np.ndarray | ObjectLabelValue, ...] = (),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Apply library-style Watershed with topology-resolved auxiliary inputs.

    Args:
        structuring_element: Footprint shape used to define neighboring pixels.
        structuring_element_size: Positive footprint radius or extent in pixels.
        topology_inputs: Ordered marker, intensity, or mask inputs selected by the
            watershed method and supplied by OpenHCS.
    """

    return request.execute(WatershedRuntimeFamily.LIBRARY, topology_inputs)


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("topology_inputs")
@callable_request(WatershedInvocationRequest)
def watershed_cellprofiler4(
    request: WatershedInvocationRequest,
    *,
    topology_inputs: tuple[np.ndarray | ObjectLabelValue, ...] = (),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows, ObjectLabelValue]:
    """Apply CellProfiler 4 Watershed with topology-resolved auxiliary inputs."""

    return request.execute(WatershedRuntimeFamily.CELLPROFILER4, topology_inputs)


set_signature_analysis_target(watershed_cellprofiler4, watershed_library)


def prepare_watershed() -> None:
    """Warm the CellProfiler watershed module paths before timed execution."""
    image = np.zeros((8, 16, 16), dtype=np.float32)
    image[2:6, 4:12, 4:12] = 1.0
    markers = np.zeros(image.shape, dtype=np.int32)
    markers[3, 6, 6] = 1
    markers[4, 10, 10] = 2
    watershed_cellprofiler4(
        image,
        use_advanced_settings=False,
        watershed_method=WatershedMethod.DISTANCE,
        declump_method=WatershedDeclumpMethod.SHAPE,
        footprint=4,
        downsample=2,
    )
    watershed_cellprofiler4(
        image,
        topology_inputs=(markers, image),
        use_advanced_settings=False,
        watershed_method=WatershedMethod.MARKERS,
        declump_method=WatershedDeclumpMethod.SHAPE,
        footprint=4,
        downsample=1,
    )
    watershed_library(
        image,
        use_advanced_settings=False,
        watershed_method=WatershedMethod.DISTANCE,
        declump_method=WatershedDeclumpMethod.SHAPE,
        footprint=4,
        downsample=1,
    )


for _watershed_function_name in WatershedModule.declared_function_names():
    attach_processing_prepare(
        WatershedModule.require_callable(_watershed_function_name),
        prepare_watershed,
    )
del _watershed_function_name


@dataclass(frozen=True, slots=True)
class LegacyWatershedRequest:
    """Validated legacy watershed inputs shared across whole-volume and plane paths."""

    image: np.ndarray
    markers: np.ndarray
    mask: np.ndarray
    connectivity: int | np.ndarray
    prefer_fast: bool

    @classmethod
    def from_inputs(
        cls,
        image: np.ndarray,
        *,
        markers: np.ndarray,
        mask: np.ndarray,
        connectivity: int | np.ndarray,
        prefer_fast: bool,
    ) -> "LegacyWatershedRequest":
        image_array = np.asarray(image, dtype=np.float64)
        mask_array = np.asarray(mask, dtype=bool)
        marker_array = np.asarray(markers) * mask_array
        if marker_array.shape != image_array.shape:
            raise ValueError("markers must have the same shape as image")
        if mask_array.shape != image_array.shape:
            raise ValueError("mask must have the same shape as image")
        return cls(
            image=image_array,
            markers=marker_array,
            mask=mask_array,
            connectivity=connectivity,
            prefer_fast=prefer_fast,
        )

    def plane(self, plane_index: int) -> "LegacyWatershedRequest":
        image_planes = self.image.reshape((-1, *self.image.shape[-2:]))
        marker_planes = self.markers.reshape((-1, *self.markers.shape[-2:]))
        mask_planes = self.mask.reshape((-1, *self.mask.shape[-2:]))
        return type(self)(
            image=image_planes[plane_index],
            markers=marker_planes[plane_index],
            mask=mask_planes[plane_index],
            connectivity=self.connectivity,
            prefer_fast=self.prefer_fast,
        )

    def execute(self) -> np.ndarray:
        """Execute the validated legacy watershed request."""
        from skimage.morphology._util import (
            _offsets_to_raveled_neighbors,
            _validate_connectivity,
        )
        from skimage.util import crop

        if self.is_planewise:
            return self.execute_planewise()
        connectivity_array, offset = _validate_connectivity(
            self.image.ndim, self.connectivity, None
        )
        pad_width = [(int(width), int(width)) for width in offset]
        padded_image = np.pad(self.image, pad_width, mode=NDIMAGE_CONSTANT_MODE)
        padded_mask = np.pad(
            self.mask.astype(np.bool_, copy=False),
            pad_width,
            mode=NDIMAGE_CONSTANT_MODE,
        ).ravel()
        output = np.pad(
            self.markers.astype(np.int32, copy=False),
            pad_width,
            mode=NDIMAGE_CONSTANT_MODE,
        )
        state = LegacyWatershedRaveledState(
            image_flat=padded_image.ravel(),
            mask_flat=padded_mask,
            output_flat=output.ravel(),
            neighbor_offsets=_offsets_to_raveled_neighbors(
                padded_image.shape, connectivity_array, center=offset
            ).astype(np.int64, copy=False),
            marker_locations=np.flatnonzero(output).astype(np.int64, copy=False),
        )
        if self.prefer_fast:
            state.execute_numba()
        else:
            state.execute_python()
        return crop(output, pad_width, copy=True)

    @property
    def is_planewise(self) -> bool:
        if self.image.ndim <= 2:
            return False
        if np.isscalar(self.connectivity):
            return False
        return np.asarray(self.connectivity).ndim == 2

    def execute_planewise(self) -> np.ndarray:
        output = np.empty(self.markers.shape, dtype=np.int32)
        output_planes = output.reshape((-1, *output.shape[-2:]))
        for plane_index in range(output_planes.shape[0]):
            output_planes[plane_index] = self.plane(plane_index).execute()
        return output


@dataclass(frozen=True, slots=True)
class LegacyWatershedRaveledState:
    """Raveled legacy watershed buffers and neighborhood provenance."""

    image_flat: np.ndarray
    mask_flat: np.ndarray
    output_flat: np.ndarray
    neighbor_offsets: np.ndarray
    marker_locations: np.ndarray

    def execute_python(self) -> None:
        heap = LegacyWatershedPythonHeap()
        for marker_location in self.marker_locations:
            location = int(marker_location)
            heap.push(float(self.image_flat[location]), 0, location)
        age = 1
        while heap:
            _value, _entry_age, index = heap.pop()
            label = int(self.output_flat[index])
            if label == 0:
                raise RuntimeError("Legacy watershed heap entry lost its label.")
            for offset_value in self.neighbor_offsets:
                neighbor_index = int(index + offset_value)
                if (
                    not self.mask_flat[neighbor_index]
                    or self.output_flat[neighbor_index] != 0
                ):
                    continue
                self.output_flat[neighbor_index] = label
                age += 1
                heap.push(float(self.image_flat[neighbor_index]), age, neighbor_index)

    def execute_numba(self) -> None:
        _legacy_watershed_raveled_numba(
            self.image_flat,
            self.mask_flat,
            self.output_flat,
            self.neighbor_offsets,
            self.marker_locations,
        )


def _footprint_offsets_3d(footprint: np.ndarray) -> np.ndarray:
    center = np.asarray(footprint.shape, dtype=np.int64) // 2
    offsets = np.argwhere(footprint).astype(np.int64) - center
    return np.ascontiguousarray(offsets[np.any(offsets != 0, axis=1)])


@njit(cache=True)
def _cellprofiler4_regional_maxima_from_candidates_3d_numba(
    image: np.ndarray, candidates: np.ndarray, offsets: np.ndarray
) -> np.ndarray:
    z_size, y_size, x_size = image.shape
    voxel_count = image.size
    visited = np.zeros(voxel_count, np.uint8)
    output = np.zeros(image.shape, np.bool_)
    stack = np.empty(voxel_count, np.int64)
    component = np.empty(voxel_count, np.int64)
    plane_size = y_size * x_size
    for start_index in range(voxel_count):
        z_start = start_index // plane_size
        start_remainder = start_index - z_start * plane_size
        y_start = start_remainder // x_size
        x_start = start_remainder - y_start * x_size
        if not candidates[z_start, y_start, x_start] or visited[start_index]:
            continue
        value = image[z_start, y_start, x_start]
        visited[start_index] = 1
        stack_size = 1
        stack[0] = start_index
        component_size = 0
        has_higher_neighbor = False
        while stack_size > 0:
            stack_size -= 1
            index = stack[stack_size]
            component[component_size] = index
            component_size += 1
            z_index = index // plane_size
            remainder = index - z_index * plane_size
            y_index = remainder // x_size
            x_index = remainder - y_index * x_size
            for offset_index in range(offsets.shape[0]):
                z_neighbor = z_index + offsets[offset_index, 0]
                y_neighbor = y_index + offsets[offset_index, 1]
                x_neighbor = x_index + offsets[offset_index, 2]
                if (
                    z_neighbor < 0
                    or y_neighbor < 0
                    or x_neighbor < 0
                    or (z_neighbor >= z_size)
                    or (y_neighbor >= y_size)
                    or (x_neighbor >= x_size)
                ):
                    continue
                neighbor_value = image[z_neighbor, y_neighbor, x_neighbor]
                if neighbor_value > value:
                    has_higher_neighbor = True
                elif neighbor_value == value:
                    neighbor_index = (
                        z_neighbor * plane_size + y_neighbor * x_size + x_neighbor
                    )
                    if not visited[neighbor_index]:
                        visited[neighbor_index] = 1
                        stack[stack_size] = neighbor_index
                        stack_size += 1
        if not has_higher_neighbor:
            for component_index in range(component_size):
                index = component[component_index]
                z_index = index // plane_size
                remainder = index - z_index * plane_size
                y_index = remainder // x_size
                x_index = remainder - y_index * x_size
                output[z_index, y_index, x_index] = True
    return output


@dataclass(slots=True)
class LegacyWatershedPythonHeap:
    """Priority heap for the Python legacy watershed reference path."""

    values: list[float]
    ages: list[int]
    indexes: list[int]

    def __init__(self) -> None:
        self.values = []
        self.ages = []
        self.indexes = []

    def __bool__(self) -> bool:
        return bool(self.values)

    @staticmethod
    def item_less(
        left_value: float, left_age: int, right_value: float, right_age: int
    ) -> bool:
        if left_value != right_value:
            return left_value < right_value
        return left_age < right_age

    def swap(self, left: int, right: int) -> None:
        self.values[left], self.values[right] = (self.values[right], self.values[left])
        self.ages[left], self.ages[right] = (self.ages[right], self.ages[left])
        self.indexes[left], self.indexes[right] = (
            self.indexes[right],
            self.indexes[left],
        )

    def push(self, value: float, age: int, index: int) -> None:
        self.values.append(value)
        self.ages.append(age)
        self.indexes.append(index)
        position = len(self.values) - 1
        while position > 0:
            parent = (position - 1) // 2
            if not self.item_less(
                self.values[position],
                self.ages[position],
                self.values[parent],
                self.ages[parent],
            ):
                break
            self.swap(position, parent)
            position = parent

    def pop(self) -> tuple[float, int, int]:
        value = self.values[0]
        age = self.ages[0]
        index = self.indexes[0]
        last = len(self.values) - 1
        if last == 0:
            self.values.pop()
            self.ages.pop()
            self.indexes.pop()
            return (value, age, index)
        self.values[0] = self.values.pop()
        self.ages[0] = self.ages.pop()
        self.indexes[0] = self.indexes.pop()
        size = len(self.values)
        position = 0
        while True:
            left = position * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and self.item_less(
                self.values[right], self.ages[right], self.values[left], self.ages[left]
            ):
                smallest = right
            if not self.item_less(
                self.values[smallest],
                self.ages[smallest],
                self.values[position],
                self.ages[position],
            ):
                break
            self.swap(position, smallest)
            position = smallest
        return (value, age, index)


@njit(cache=True)
def _heap_item_less(
    left_value: float, left_age: int, right_value: float, right_age: int
) -> bool:
    if left_value != right_value:
        return left_value < right_value
    return left_age < right_age


@njit(cache=True)
def _heap_swap(
    heap_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], left: int, right: int
) -> None:
    values, ages, indexes = heap_arrays
    value = values[left]
    age = ages[left]
    index = indexes[left]
    values[left] = values[right]
    ages[left] = ages[right]
    indexes[left] = indexes[right]
    values[right] = value
    ages[right] = age
    indexes[right] = index


@njit(cache=True)
def _heap_push(
    heap_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    size: int,
    value: float,
    age: int,
    index: int,
) -> int:
    values, ages, indexes = heap_arrays
    values[size] = value
    ages[size] = age
    indexes[size] = index
    size += 1
    position = size - 1
    while position > 0:
        parent = (position - 1) // 2
        if not _heap_item_less(
            values[position], ages[position], values[parent], ages[parent]
        ):
            break
        _heap_swap(heap_arrays, position, parent)
        position = parent
    return size


@njit(cache=True)
def _heap_pop(
    heap_arrays: tuple[np.ndarray, np.ndarray, np.ndarray], size: int
) -> tuple[int, float, int, int]:
    values, ages, indexes = heap_arrays
    value = values[0]
    age = ages[0]
    index = indexes[0]
    size -= 1
    if size > 0:
        values[0] = values[size]
        ages[0] = ages[size]
        indexes[0] = indexes[size]
        position = 0
        while True:
            left = position * 2 + 1
            right = left + 1
            if left >= size:
                break
            smallest = left
            if right < size and _heap_item_less(
                values[right], ages[right], values[left], ages[left]
            ):
                smallest = right
            if not _heap_item_less(
                values[smallest], ages[smallest], values[position], ages[position]
            ):
                break
            _heap_swap(heap_arrays, position, smallest)
            position = smallest
    return (size, value, age, index)


@njit(cache=True)
def _legacy_watershed_raveled_numba(
    image_flat: np.ndarray,
    mask_flat: np.ndarray,
    output_flat: np.ndarray,
    neighbor_offsets: np.ndarray,
    marker_locations: np.ndarray,
) -> None:
    capacity = output_flat.size
    heap_values = np.empty(capacity, dtype=np.float64)
    heap_ages = np.empty(capacity, dtype=np.int64)
    heap_indexes = np.empty(capacity, dtype=np.int64)
    heap_arrays = (heap_values, heap_ages, heap_indexes)
    heap_size = 0
    for marker_location in marker_locations:
        location = int(marker_location)
        heap_size = _heap_push(
            heap_arrays, heap_size, float(image_flat[location]), 0, location
        )
    age = 1
    while heap_size > 0:
        heap_size, _value, _entry_age, index = _heap_pop(heap_arrays, heap_size)
        label = int(output_flat[index])
        if label == 0:
            raise RuntimeError("Legacy watershed heap entry lost its label.")
        for offset_value in neighbor_offsets:
            neighbor_index = int(index + offset_value)
            if not mask_flat[neighbor_index] or output_flat[neighbor_index] != 0:
                continue
            output_flat[neighbor_index] = label
            age += 1
            heap_size = _heap_push(
                heap_arrays,
                heap_size,
                float(image_flat[neighbor_index]),
                age,
                neighbor_index,
            )


__all__ = public_names_from_objects(
    CellProfiler4DistanceMarkerBackendStrategy,
    LegacyWatershedBackendStrategy,
    MahotasCellProfiler4DistanceMarkerBackendStrategy,
    NumbaCellProfiler4DistanceMarkerBackendStrategy,
    NumbaNumpyLegacyWatershedBackendStrategy,
    NumpyLegacyWatershedBackendStrategy,
    WatershedStats,
    cellprofiler_legacy_watershed,
    *(
        WatershedModule.require_callable(function_name)
        for function_name in WatershedModule.declared_function_names()
    ),
)
