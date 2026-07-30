"""Intensity-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.core.artifacts import ArtifactInputPlan
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, ClassVar

from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    SourceStackLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ImageMeasurementInputModule,
    ObjectArtifactInputModule,
    ObjectMeasurementInputModule,
    SourceQualifiedMeasurementFeatureModule,
    SourceQualifiedWideMeasurementRowsModule,
    PerObjectMeasurementExecutionModule,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    IntensityFeature,
    ObjectLocationFeature,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    DeclaredObjectMeasurementRowPolicy,
    DenseColumnarObjectMeasurementRowsMixin,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    MissingObjectMeasurementValuePolicy,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
)
from openhcs.core.runtime_object_labels import ObjectLabelVariantData

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder


class RescaleIntensityAutomaticHigh(Enum):
    """Automatic upper-bound policies exposed by RescaleIntensity settings."""

    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class RescaleIntensityAutomaticLow(Enum):
    """Automatic lower-bound policies exposed by RescaleIntensity settings."""

    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class RescaleIntensityMethod(Enum):
    """RescaleIntensity method literals exposed by CellProfiler settings."""

    STRETCH = "stretch"
    MANUAL_INPUT_RANGE = "manual_input_range"
    MANUAL_IO_RANGE = "manual_io_range"
    DIVIDE_BY_IMAGE_MINIMUM = "divide_by_image_minimum"
    DIVIDE_BY_IMAGE_MAXIMUM = "divide_by_image_maximum"
    DIVIDE_BY_VALUE = "divide_by_value"


def _parse_image_intensity_percentiles(value: str) -> tuple[int, ...]:
    """Parse CellProfiler's comma-delimited percentile setting."""

    return tuple(
        int(percentile.strip())
        for percentile in value.split(",")
        if percentile.strip()
    )


class MeasureImageIntensityModule(
    LabelsObjectInputPolicy,
    PerObjectMeasurementExecutionModule,
    FieldDerivedMeasurementFeatureModule,
    SourceQualifiedMeasurementFeatureModule,
    ImageMeasurementInputModule,
    ObjectArtifactInputModule,
):
    module_name = "MeasureImageIntensity"
    function_name = "measure_image_intensity"
    function_variants = ("measure_image_intensity_objects",)
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("intensity",),)
    measurement_feature_family = "Intensity"
    object_gate_setting = SettingNameFamily(
        "Select the input objects",
        aliases=("Select input objects", "Select input object sets", "Objects"),
    )
    object_measurement_setting = object_gate_setting
    object_gate_binding = SettingToKeywordBinding.input(
        object_gate_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    calculate_percentiles_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate custom percentiles"
    )
    percentiles_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Specify percentiles to measure"
    )
    setting_bindings = (
        object_gate_binding,
        SettingToKeywordBinding(
            calculate_percentiles_setting,
            "calculate_percentiles",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            percentiles_setting,
            "percentiles",
            _parse_image_intensity_percentiles,
        ),
    )
    ignored_settings = ("Measure the intensity only from areas enclosed by objects?",)

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project image-intensity records into exact CellProfiler features."""

        measurement_name: str

        @classmethod
        def for_request(cls, module_type, request):
            source_image_name = request.source.source_image_name
            if source_image_name is None:
                raise ValueError(
                    "MeasureImageIntensity requires exact source-image ownership."
                )
            object_inputs = ArtifactSpecCollection(
                request.callable_contract.artifact_inputs.specs
            ).of_artifact_type(ObjectLabelsArtifactType)
            if len(object_inputs) > 1:
                raise ValueError(
                    "MeasureImageIntensity accepts at most one object-label input "
                    f"per invocation, got {[spec.name for spec in object_inputs]!r}."
                )
            measurement_name = (
                source_image_name
                if not object_inputs
                else f"{source_image_name}_{object_inputs[0].name}"
            )
            return cls(
                request.output_value,
                module_type=module_type,
                measurement_name=measurement_name,
            )

        def rows(self) -> MeasurementProjectedColumnarRows:
            source_rows = self.source_rows()
            slice_field = self.source_field(MeasurementRowAxisField.SLICE_INDEX)
            percentile_field = self.source_field("percentile_values")
            feature_fields = tuple(
                field_spec
                for field_spec in source_rows.fields
                if field_spec.name not in (slice_field.name, percentile_field.name)
            )
            slice_indices: list[object] = []
            feature_names: list[str] = []
            values: list[object] = []
            for source_record in source_rows.iter_row_mappings():
                slice_index = source_record[slice_field.name]
                for field_spec in feature_fields:
                    slice_indices.append(slice_index)
                    feature_names.append(
                        self.module_type.measurement_feature_name(
                            field_spec.name,
                            self.measurement_name,
                        )
                    )
                    values.append(source_record[field_spec.name])
                percentile_values = source_record[percentile_field.name]
                if not isinstance(percentile_values, Mapping):
                    raise TypeError(
                        "MeasureImageIntensity percentile_values must be a mapping, "
                        f"got {type(percentile_values).__name__}."
                    )
                for percentile, value in percentile_values.items():
                    slice_indices.append(slice_index)
                    feature_names.append(
                        self.module_type.measurement_feature_name(
                            "percentile",
                            percentile,
                            self.measurement_name,
                        )
                    )
                    values.append(value)
            value_field_name = ImageIntensityMeasurement.measurement_value_field.value
            return MeasurementProjectedColumnarRows(
                MappingProxyType(
                    {
                        slice_field.name: tuple(slice_indices),
                        MeasurementRowAxisField.FEATURE_NAME.value: tuple(
                            feature_names
                        ),
                        value_field_name: tuple(values),
                    }
                ),
                fields=(
                    slice_field,
                    FieldSpec(MeasurementRowAxisField.FEATURE_NAME.value, str),
                    FieldSpec(value_field_name),
                ),
            )

    @classmethod
    def ignored_settings_for(
        cls, module: "ModuleBlock"
    ) -> tuple[str | "SettingNameFamily", ...]:
        from openhcs.interop.cellprofiler.setting_names import is_blank_symbol_name

        ignored = tuple(cls.ignored_settings)
        value = cls.setting_value(module, cls.object_gate_setting, include_blank=True)
        if value is not None and (not value.strip() or is_blank_symbol_name(value)):
            return (*ignored, cls.object_gate_setting)
        return ignored

    @classmethod
    def active_artifact_bindings(
        cls,
        module: "ModuleBlock | None" = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Declare masked-image topology only when an object set is selected."""

        from openhcs.interop.cellprofiler.setting_names import is_blank_symbol_name

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        value = cls.setting_value(module, cls.object_gate_setting, include_blank=True)
        active = value is not None and bool(value.strip()) and not is_blank_symbol_name(value)
        return tuple(
            binding
            for binding in bindings
            if active or binding is not cls.object_gate_binding
        )


class RescaleIntensityModule(
    CellProfilerModule
):
    module_name = "RescaleIntensity"
    function_name = "rescale_intensity"
    validated = True
    confidence = 1.0
    setting_bindings = (
        SettingToKeywordBinding.input("Select the input image", ImageArtifactType),
        SettingToKeywordBinding.input(
            "Select image to match in maximum intensity", ImageArtifactType
        ),
        SettingToKeywordBinding.output("Name the output image", ImageArtifactType),
        SettingToKeywordBinding(
            "Rescaling method",
            "rescale_method",
            cellprofiler_enum_value_setting_parser(RescaleIntensityMethod),
        ),
        SettingToKeywordBinding(
            "Method to calculate the minimum intensity",
            "automatic_low",
            cellprofiler_enum_value_setting_parser(RescaleIntensityAutomaticLow),
        ),
        SettingToKeywordBinding(
            "Method to calculate the maximum intensity",
            "automatic_high",
            cellprofiler_enum_value_setting_parser(RescaleIntensityAutomaticHigh),
        ),
        SettingToKeywordBinding(
            "Lower intensity limit for the input image",
            "source_low",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Upper intensity limit for the input image",
            "source_high",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Divisor value", "divisor_value", parse_cellprofiler_float
        ),
    )
    ignored_settings = ("Divisor measurement",)
    range_settings = {
        "Intensity range for the input image": ("source_low", "source_high"),
        "Intensity range for the output image": ("dest_low", "dest_high"),
    }

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
        """Anchor output context to the image being rescaled."""

        del invocation_key, step_context, binding, name, output_position
        source = artifact_inputs.require_by_name_and_artifact_type(
            required_setting_value(
                module,
                cls.declared_artifact_bindings(plan_type = ArtifactInputPlan, artifact_type = ImageArtifactType)[0].setting_name,
            ),
            ImageArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
    ) -> "BoundModuleSettings":
        bound = cls._bind_declared_settings(module, binder=binder)
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name, target_names in cls.range_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is None:
                continue
            parsed_range = binder.parse_value(setting_name, value)
            if not isinstance(parsed_range, tuple) or len(parsed_range) != 2:
                raise ValueError(
                    f"{module.name} {setting_name!r} must contain two values, got {value!r}."
                )
            kwargs[target_names[0]], kwargs[target_names[1]] = parsed_range
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, unmapped_kwargs),
        )


from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from dataclasses import field
from enum import Enum
import logging
import time
from typing import TypeAlias
import numpy as np
from metaclass_registry import AutoRegisterMeta
from openhcs.core.alias_property import AliasProperty
from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import runtime_image_execution_mode
from openhcs.core.memory import numpy as numpy_decorator
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    runtime_bound_parameters,
    special_inputs,
    )
from openhcs.core.runtime_batch_contracts import (
    RuntimePure2DSliceBatchRequest,
    SliceIndexRuntimeParameter,
    measurement_image_batch_executor,
    pure_2d_batch_executor,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_object_label_domains import (
    ConsecutiveObjectLabelIdProjection,
    ObjectLabelDomain,
    dense_object_label_measurement_row_domain,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
    RuntimeMeasurementFeatureRelationDeclaration,
    RuntimeMeasurementFeatureSemanticMarker,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
)
from openhcs.core.runtime_image_values import (
    ImageMetadataPayload,
    MaskedImagePayload,
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
    ObjectMeasurementColumnarRows,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_batch_contracts import RuntimeBatchInvocationRequest
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    PreparedObjectMeasurementInvocation,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendProviderSelection,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.enum_attributes import (
    CellProfilerEnumAttributeMixin,
)
from openhcs.processing.backends.cellprofiler.intensity_object_quantiles_numba import (
    ObjectIntensityArrays,
    ObjectIntensityFeatureValues,
    ObjectIntensityForegroundIndex,
    _object_intensity_quantiles,
    _object_intensity_quantiles_3d_batch_numba,
    _object_intensity_quantiles_3d_sparse_batch_numba,
    _object_intensity_scan_3d_batch_numba,
    _object_intensity_scan_3d_sparse_batch_numba,
    _object_intensity_scan_numba,
)
from openhcs.processing.backends.cellprofiler.shape import (
    ShapeMeasurementBackendStrategy,
    _numpy124_aquicksort_indices,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.equivalence.measurement_features import (
    TieSensitiveLocationValueFeatureRelation,
)

logger = logging.getLogger(__name__)
ImageIntensityOutput: TypeAlias = np.ndarray | ImageMetadataPayload | MaskedImagePayload
ObjectIntensityRuntimeRequest: TypeAlias = (
    RuntimePure2DSliceBatchRequest | RuntimeBatchInvocationRequest
)


class MaxIntensityLocationFeature(ObjectLocationFeature):
    """Location feature tied to the corresponding maximum-intensity value."""


class RescaleMethod(CellProfilerEnumAttributeMixin, Enum):
    """Closed CellProfiler rescale modes with intensity-scale policy metadata."""

    __cellprofiler_attribute_names__ = (
        "_can_preserve_unit_interval_scale",
        "_requires_unit_destination_range",
    )
    STRETCH = ("stretch", True, False)
    MANUAL_INPUT_RANGE = ("manual_input_range", True, False)
    MANUAL_IO_RANGE = ("manual_io_range", True, True)
    DIVIDE_BY_IMAGE_MINIMUM = ("divide_by_image_minimum", False, False)
    DIVIDE_BY_IMAGE_MAXIMUM = ("divide_by_image_maximum", False, False)
    DIVIDE_BY_VALUE = ("divide_by_value", False, False)

    def preserves_unit_interval_intensity_scale(
        self,
        *,
        source_range: tuple[float, float],
        destination_range: tuple[float, float],
    ) -> bool:
        """Return whether this mode preserves a proven unit-interval scale."""
        if not self._can_preserve_unit_interval_scale:
            return False
        source_low, source_high = source_range
        if not (np.isclose(source_low, 0.0) and np.isclose(source_high, 1.0)):
            return False
        if not self._requires_unit_destination_range:
            return True
        destination_low, destination_high = destination_range
        return np.isclose(destination_low, 0.0) and np.isclose(destination_high, 1.0)


class AutomaticLow(Enum):
    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


class AutomaticHigh(Enum):
    CUSTOM = "custom"
    EACH_IMAGE = "each_image"


@dataclass(frozen=True, slots=True)
class ImageIntensityPercentileSpec:
    """Percentile calculation policy for image-intensity rows."""

    enabled: bool = False
    percentiles: tuple[int, ...] = (10, 90)

    def __post_init__(self) -> None:
        if any(
            isinstance(percentile, bool)
            or not isinstance(percentile, int)
            or not 0 <= percentile <= 100
            for percentile in self.percentiles
        ):
            raise ValueError("Image intensity percentiles must be integers from 0 to 100.")

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.percentiles)))

    def measurements_for(self, pixels: np.ndarray) -> dict[int, float]:
        if not self.enabled:
            return {}
        parsed_percentiles = self.values
        if pixels.size == 0:
            return {percentile: 0.0 for percentile in parsed_percentiles}
        if not parsed_percentiles:
            return {}
        percentile_results = np.percentile(pixels, parsed_percentiles)
        return {
            percentile: float(value)
            for percentile, value in zip(parsed_percentiles, percentile_results)
        }


@dataclass(frozen=True, slots=True)
class ImageIntensityMeasurement(MeasurementFeatureRecord):
    """CellProfiler-compatible intensity measurements for one image region."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    total_intensity: float
    mean_intensity: float
    median_intensity: float
    std_intensity: float
    mad_intensity: float
    min_intensity: float
    max_intensity: float
    total_area: int
    percent_maximal: float
    lower_quartile_intensity: float
    upper_quartile_intensity: float
    percentile_values: dict[int, float]

    @classmethod
    def from_pixels(
        cls, pixels: np.ndarray, *, percentile_spec: ImageIntensityPercentileSpec
    ) -> "ImageIntensityMeasurement":
        """Build the authoritative image-intensity measurement row."""
        pixels = pixels[np.isfinite(pixels)]
        pixel_count = pixels.size
        percentile_dict = percentile_spec.measurements_for(pixels)
        if pixel_count == 0:
            pixel_sum = 0.0
            pixel_mean = 0.0
            pixel_std = 0.0
            pixel_mad = 0.0
            pixel_median = 0.0
            pixel_min = 0.0
            pixel_max = 0.0
            pixel_pct_max = 0.0
            pixel_lower_qrt = 0.0
            pixel_upper_qrt = 0.0
        else:
            pixel_sum = float(np.sum(pixels))
            pixel_mean = pixel_sum / float(pixel_count)
            pixel_std = float(np.std(pixels))
            pixel_median = float(np.median(pixels))
            pixel_mad = float(np.median(np.abs(pixels - pixel_median)))
            pixel_min = float(np.min(pixels))
            pixel_max = float(np.max(pixels))
            pixel_pct_max = (
                100.0 * float(np.sum(pixels == pixel_max)) / float(pixel_count)
            )
            quartiles = np.percentile(pixels, [25, 75])
            pixel_lower_qrt = float(quartiles[0])
            pixel_upper_qrt = float(quartiles[1])
        return cls(
            slice_index=0,
            total_intensity=pixel_sum,
            mean_intensity=pixel_mean,
            median_intensity=pixel_median,
            std_intensity=pixel_std,
            mad_intensity=pixel_mad,
            min_intensity=pixel_min,
            max_intensity=pixel_max,
            total_area=int(pixel_count),
            percent_maximal=pixel_pct_max,
            lower_quartile_intensity=pixel_lower_qrt,
            upper_quartile_intensity=pixel_upper_qrt,
            percentile_values=percentile_dict,
        )


@dataclass(frozen=True, slots=True)
class ObjectIntensityMeasurement(ObjectIntensityFeatureValues[float]):
    """Per-object CellProfiler-compatible intensity measurements."""

    slice_index: int
    object_label: int

    @classmethod
    def from_backend_arrays(
        cls, arrays: ObjectIntensityArrays, *, index: int, label: int, slice_index: int
    ) -> "ObjectIntensityMeasurement":
        """Materialize one CellProfiler object-intensity row from backend arrays."""
        return cls(
            slice_index=slice_index,
            object_label=int(label),
            **arrays.scalar_kwargs(index),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectIntensityMeasurementAxisContext:
    """Runtime row-axis context for one object-intensity measurement plane."""

    slice_index: int
    object_domain: tuple[int, ...] | None = None
    object_row_identity: MeasurementObjectRowIdentity | None = None

    def object_labels_for(self, measured_labels: np.ndarray) -> np.ndarray:
        return np.asarray(
            (
                self.object_domain
                if self.object_domain is not None
                else tuple((int(label) for label in measured_labels))
            ),
            dtype=np.int32,
        )

    def axis_columns_for(self, row_count: int) -> dict[str, np.ndarray]:
        columns: dict[str, np.ndarray] = {
            MeasurementRowAxisField.SLICE_INDEX.value: np.full(
                row_count, self.slice_index, dtype=np.int64
            ),
        }
        return columns


@dataclass(frozen=True, slots=True)
class ObjectIntensityMeasurementRows(
    ObjectIntensityMeasurementAxisContext, ObjectMeasurementColumnarRows
):
    """Columnar object-intensity measurements for runtime lookup paths."""

    fields: ClassVar[tuple[FieldSpec, ...]] = (
        FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
        FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int),
        *(
            FieldSpec(
                feature_field.name,
                float,
                required=feature_field.required,
            )
            for feature_field in FieldSpec.from_dataclass_type(
                ObjectIntensityFeatureValues
            )
        ),
    )
    arrays: ObjectIntensityArrays
    _columns: dict[str, np.ndarray] = field(repr=False, compare=False)

    @classmethod
    def from_arrays(
        cls,
        arrays: ObjectIntensityArrays,
        *,
        slice_index: int,
        object_domain: tuple[int, ...] | None = None,
        object_row_identity: MeasurementObjectRowIdentity | None = None,
    ) -> "ObjectIntensityMeasurementRows":
        axis_context = ObjectIntensityMeasurementAxisContext(
            slice_index=slice_index,
            object_domain=object_domain,
            object_row_identity=object_row_identity,
        )
        object_labels = axis_context.object_labels_for(arrays.object_labels)
        row_count = int(object_labels.size)
        measured_index_by_label = {
            int(label): index for index, label in enumerate(arrays.object_labels)
        }
        measured_indexes = np.asarray(
            [
                (
                    measured_index_by_label[label]
                    if label in measured_index_by_label
                    else -1
                )
                for label in object_labels
            ],
            dtype=np.int64,
        )
        measured_mask = measured_indexes >= 0

        def align_column(values: np.ndarray) -> np.ndarray:
            return _align_intensity_column(
                values, object_labels, measured_indexes, measured_mask
            )

        columns: dict[str, np.ndarray] = {
            **axis_context.axis_columns_for(row_count),
            MeasurementRowAxisField.OBJECT_LABEL.value: object_labels,
            **arrays.aligned_feature_columns(align_column),
        }
        return cls(
            arrays=arrays,
            slice_index=slice_index,
            object_domain=object_domain,
            object_row_identity=object_row_identity,
            _columns=columns,
        )

    columns: ClassVar[AliasProperty[dict[str, np.ndarray]]] = AliasProperty("_columns")

    def __post_init__(self) -> None:
        self.validate_fields()

    def __len__(self) -> int:
        return int(self._columns[MeasurementRowAxisField.OBJECT_LABEL.value].size)

    def __iter__(self):
        for row_index in range(len(self)):
            yield self[row_index]

    def __getitem__(self, index: int) -> ObjectIntensityMeasurement:
        columns = self._columns
        return ObjectIntensityMeasurement(
            slice_index=int(columns[MeasurementRowAxisField.SLICE_INDEX.value][index]),
            object_label=int(
                columns[MeasurementRowAxisField.OBJECT_LABEL.value][index]
            ),
            **ObjectIntensityMeasurement.scalar_kwargs_from_columns(columns, index),
        )


ObjectIntensityLabelInput = ObjectLabelValue | np.ndarray
OBJECT_INTENSITY_DEFAULT_SLICE_INDEX = 0
OBJECT_INTENSITY_PREPARED_LABELS_KWARG = "object_intensity_prepared_labels"


def _align_intensity_column(
    measured_values: np.ndarray,
    object_labels: np.ndarray,
    measured_indexes: np.ndarray,
    measured_mask: np.ndarray,
) -> np.ndarray:
    """Align measured object rows to a declared object-label domain."""
    values = np.asarray(measured_values)
    aligned = np.zeros(
        measured_indexes.size, dtype=np.result_type(values.dtype, np.float64)
    )
    if measured_mask.any():
        aligned[measured_mask] = values[measured_indexes[measured_mask]]
        measured_extent = int(np.max(np.asarray(object_labels)[measured_mask]))
    else:
        measured_extent = 0
    aligned[np.asarray(object_labels) > measured_extent] = np.nan
    return aligned


@dataclass(frozen=True, slots=True)
class ObjectIntensityPreparedLabels:
    """Prepared object-label domain reused across same-label intensity images."""

    source: ObjectIntensityLabelInput
    dense_labels: np.ndarray
    object_domain: tuple[int, ...]
    projection: ConsecutiveObjectLabelIdProjection
    relabeled_labels: np.ndarray
    label_to_index: np.ndarray
    foreground_index: ObjectIntensityForegroundIndex | None = None

    @classmethod
    def from_source(
        cls, labels: ObjectIntensityLabelInput, dense_labels: np.ndarray
    ) -> "ObjectIntensityPreparedLabels":
        label_array = np.ascontiguousarray(dense_labels, dtype=np.int32)
        projection = ConsecutiveObjectLabelIdProjection.from_dense_array(label_array)
        relabeled_labels = np.ascontiguousarray(
            projection.relabel_numpy_array(label_array, dtype=np.int32), dtype=np.int32
        )
        label_to_index = cls.label_to_index_for_projection(projection)
        foreground_index = (
            ObjectIntensityForegroundIndex.from_3d_labels(
                relabeled_labels, label_to_index
            )
            if relabeled_labels.ndim == 3
            else None
        )
        return cls(
            source=labels,
            dense_labels=label_array,
            object_domain=dense_object_label_measurement_row_domain(
                labels, projection.positive_label_ids
            ),
            projection=projection,
            relabeled_labels=relabeled_labels,
            label_to_index=label_to_index,
            foreground_index=foreground_index,
        )

    @classmethod
    def from_measurement(
        cls, *, image: object, labels: ObjectIntensityLabelInput, slice_index: int
    ) -> "ObjectIntensityPreparedLabels":
        del image, slice_index
        return cls.from_source(
            labels,
            object_label_dense_array(labels, dtype=np.int32),
        )

    @staticmethod
    def label_to_index_for_projection(
        projection: ConsecutiveObjectLabelIdProjection,
    ) -> np.ndarray:
        label_to_index = np.full(projection.object_count + 1, -1, dtype=np.int64)
        if projection.has_objects:
            label_to_index[1:] = np.arange(projection.object_count, dtype=np.int64)
        return label_to_index

    def with_relabeled_labels(
        self, relabeled_labels: np.ndarray
    ) -> "ObjectIntensityPreparedLabels":
        relabeled = np.ascontiguousarray(relabeled_labels, dtype=np.int32)
        foreground_index = (
            ObjectIntensityForegroundIndex.from_3d_labels(
                relabeled, self.label_to_index
            )
            if relabeled.ndim == 3
            else None
        )
        return type(self)(
            source=self.source,
            dense_labels=self.dense_labels,
            object_domain=self.object_domain,
            projection=self.projection,
            relabeled_labels=relabeled,
            label_to_index=self.label_to_index,
            foreground_index=foreground_index,
        )

    @property
    def object_labels(self) -> np.ndarray:
        return self.projection.positive_label_ids.astype(np.int32, copy=False)

    @property
    def object_count(self) -> int:
        return self.projection.object_count

    @property
    def measurement_row_identity(self) -> MeasurementObjectRowIdentity | None:
        """Return the row identity declared by the prepared label plane domain."""
        if not isinstance(self.source, ObjectLabelValue):
            return None
        return MeasurementObjectRowIdentity.LABEL_ID


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectIntensityMeasurementContext(ObjectIntensityMeasurementAxisContext):
    """Nominal measurement context for object-intensity execution."""

    labels: ObjectIntensityLabelInput
    backend_provider: CellProfilerBackendProviderSelection
    prepared_labels: ObjectIntensityPreparedLabels | None = None

    @classmethod
    def from_function_arguments(
        cls,
        *,
        labels: ObjectIntensityLabelInput,
        backend_provider: BackendProviderInput,
        slice_index: int,
        object_domain: tuple[int, ...] | None = None,
        object_row_identity: MeasurementObjectRowIdentity | None = None,
        prepared_labels: ObjectIntensityPreparedLabels | None = None,
    ) -> "ObjectIntensityMeasurementContext":
        return cls(
            labels=labels,
            backend_provider=CellProfilerBackendAuthority.provider_selection(
                backend_provider
            ),
            slice_index=int(slice_index),
            object_domain=object_domain,
            object_row_identity=object_row_identity,
            prepared_labels=prepared_labels,
        )

    @classmethod
    def from_runtime_request(
        cls, request: ObjectIntensityRuntimeRequest
    ) -> "ObjectIntensityMeasurementContext":
        kwargs = request.kwargs
        if "labels" not in kwargs:
            raise ValueError("Object-intensity runtime kwargs are missing 'labels'.")
        labels = kwargs["labels"]
        if not isinstance(labels, ObjectLabelValue):
            raise TypeError(
                "Object-intensity runtime labels must be an already-projected "
                f"ObjectLabelValue, got {type(labels).__name__}."
            )
        backend_provider = (
            kwargs["object_intensity_backend_provider"]
            if "object_intensity_backend_provider" in kwargs
            else DEFAULT_CELLPROFILER_BACKEND_SELECTION
        )
        prepared_labels = (
            kwargs[OBJECT_INTENSITY_PREPARED_LABELS_KWARG]
            if OBJECT_INTENSITY_PREPARED_LABELS_KWARG in kwargs
            else None
        )
        if prepared_labels is None or isinstance(
            prepared_labels, ObjectIntensityPreparedLabels
        ):
            return cls(
                labels=labels,
                backend_provider=CellProfilerBackendAuthority.provider_selection(
                    backend_provider
                ),
                slice_index=(
                    int(kwargs["slice_index"])
                    if "slice_index" in kwargs
                    else OBJECT_INTENSITY_DEFAULT_SLICE_INDEX
                ),
                prepared_labels=prepared_labels,
            )
        raise TypeError(
            f"object_intensity_prepared_labels must be ObjectIntensityPreparedLabels or None; got {type(prepared_labels).__name__}."
        )

    def batch_key_items(self) -> tuple[tuple[str, Hashable], ...]:
        return (
            (
                "object_intensity_backend_provider",
                self.backend_provider.semantic_identity(),
            ),
            ("slice_index", self.slice_index),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ObjectIntensityMeasurementRequest(ObjectIntensityMeasurementContext):
    """Executable request for one object-intensity image/label plane."""

    image: np.ndarray

    @property
    def measurement_image(self) -> np.ndarray:
        return np.asarray(self.image)

    @property
    def dense_labels(self) -> np.ndarray:
        if self.prepared_labels is not None:
            return self.prepared_labels.dense_labels
        return object_label_dense_array(self.labels, dtype=np.int32)

    @property
    def measurement_object_domain(self) -> tuple[int, ...]:
        """Return the object-intensity row domain without fabricating sparse IDs."""
        if self.object_domain is not None:
            return self.object_domain
        if self.prepared_labels is not None:
            return self.prepared_labels.object_domain
        return dense_object_label_measurement_row_domain(self.labels, self.dense_labels)

    def measurements(self) -> ObjectIntensityMeasurementRows:
        """Measure this image/label plane through the selected backend."""
        prepared_labels = (
            self.prepared_labels
            if self.prepared_labels is not None
            else ObjectIntensityPreparedLabels.from_source(
                self.labels, self.dense_labels
            )
        )
        intensity_arrays = object_intensity_backend(
            backend_provider=self.backend_provider
        ).measure_prepared(self.measurement_image, prepared_labels)
        return ObjectIntensityMeasurementRows.from_arrays(
            intensity_arrays,
            slice_index=self.slice_index,
            object_domain=self.measurement_object_domain,
            object_row_identity=self.object_row_identity,
        )


class ObjectIntensityBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Object-intensity operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def measure(
        self, image: np.ndarray, labels: ObjectIntensityLabelInput
    ) -> ObjectIntensityArrays:
        """Measure object intensity arrays for one image plane."""

    def measure_prepared(
        self, image: np.ndarray, labels: ObjectIntensityPreparedLabels
    ) -> ObjectIntensityArrays:
        """Measure object intensity arrays with a prepared label domain."""
        return self.measure(image, labels.source)

    def measure_prepared_batch(
        self, images: tuple[np.ndarray, ...], labels: ObjectIntensityPreparedLabels
    ) -> tuple[ObjectIntensityArrays, ...]:
        """Measure multiple images that share one prepared label domain."""
        return tuple((self.measure_prepared(image, labels) for image in images))


class NumbaNumpyObjectIntensityBackendStrategy(ObjectIntensityBackendStrategy):
    """Numba-accelerated NumPy object-intensity backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True
    sparse_foreground_max_fraction: ClassVar[float] = 0.6

    def maximum_intensity_positions(
        self,
        image: np.ndarray,
        labels: ObjectIntensityPreparedLabels,
    ) -> tuple[np.ndarray, ...]:
        """Return CellProfiler 4.2 maximum positions in the label domain."""
        label_ids = np.arange(1, labels.object_count + 1, dtype=np.int32)
        shape_backend = ShapeMeasurementBackendStrategy.for_memory_type(
            MemoryType.NUMPY,
            backend_provider=self.backend_provider,
        )
        return shape_backend.maximum_position_of_labels(
            image,
            labels.relabeled_labels,
            label_ids,
            mask=labels.relabeled_labels > 0,
        )

    def maximum_intensity_positions_batch(
        self,
        images: np.ndarray,
        labels: ObjectIntensityPreparedLabels,
    ) -> tuple[tuple[np.ndarray, ...], ...]:
        """Return exact maximum positions without repeating foreground scans."""
        foreground_index = labels.foreground_index
        image_batch = np.asarray(images)
        image_shape = image_batch.shape[1:]
        if foreground_index is None or not foreground_index.matches_shape(image_shape):
            raise ValueError(
                "Batched 3-D maximum positions require a matching prepared "
                "foreground index."
            )

        z_indices = foreground_index.z_indices
        y_indices = foreground_index.y_indices
        x_indices = foreground_index.x_indices
        source_positions = (
            z_indices * (image_shape[1] * image_shape[2])
            + y_indices * image_shape[2]
            + x_indices
        )
        flat_images = image_batch.reshape((image_batch.shape[0], -1))
        positions: list[tuple[np.ndarray, ...]] = []
        for flat_image in flat_images:
            order = _numpy124_aquicksort_indices(flat_image[source_positions])
            maximum_positions = np.zeros(labels.object_count, dtype=np.int64)
            maximum_positions[foreground_index.object_indexes[order]] = (
                source_positions[order]
            )
            positions.append(
                tuple(
                    np.asarray(coordinates, dtype=np.float64)
                    for coordinates in np.unravel_index(
                        maximum_positions,
                        image_shape,
                    )
                )
            )
        return tuple(positions)

    def measure(
        self, image: np.ndarray, labels: ObjectIntensityLabelInput
    ) -> ObjectIntensityArrays:
        image_array = np.ascontiguousarray(image)
        label_array = np.ascontiguousarray(
            object_label_dense_array(labels, dtype=np.int32)
        )
        if image_array.ndim != 2 or label_array.ndim != 2:
            if image_array.ndim == 3 and label_array.ndim == 3:
                return self._measure_3d_prepared(
                    image_array,
                    ObjectIntensityPreparedLabels.from_source(labels, label_array),
                )
            raise NotImplementedError(
                "NumPy object-intensity requires image and labels already projected "
                "into the same 2-D or 3-D domain; got image shape "
                f"{image_array.shape!r} and label shape {label_array.shape!r}."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")
        return self.measure_prepared(
            image_array,
            ObjectIntensityPreparedLabels.from_source(labels, label_array),
        )

    def measure_prepared(
        self, image: np.ndarray, labels: ObjectIntensityPreparedLabels
    ) -> ObjectIntensityArrays:
        image_array = np.ascontiguousarray(image)
        label_array = labels.dense_labels
        if image_array.ndim != 2 or label_array.ndim != 2:
            if image_array.ndim == 3 and label_array.ndim == 3:
                return self._measure_3d_prepared(image_array, labels)
            raise NotImplementedError(
                "NumPy object-intensity requires prepared labels in the exact image "
                f"domain; got image shape {image_array.shape!r} and label shape "
                f"{label_array.shape!r}."
            )
        if image_array.shape != label_array.shape:
            raise ValueError("image and labels must have matching shapes.")
        object_labels = labels.object_labels
        object_count = labels.object_count
        if object_count == 0:
            return ObjectIntensityArrays.empty(object_labels)
        arrays = _object_intensity_scan_numba(
            image_array, labels.relabeled_labels, object_labels, labels.label_to_index
        )
        lower, median, upper, mad = _object_intensity_quantiles(
            image_array,
            labels.relabeled_labels,
            object_labels,
            labels.label_to_index,
            arrays[0].astype(np.int64, copy=False),
        )
        return ObjectIntensityArrays(
            object_labels=object_labels.astype(np.int32, copy=False),
            integrated_intensity=arrays[1],
            mean_intensity=arrays[2],
            std_intensity=arrays[3],
            min_intensity=arrays[4],
            max_intensity=arrays[5],
            integrated_intensity_edge=arrays[6],
            mean_intensity_edge=arrays[7],
            std_intensity_edge=arrays[8],
            min_intensity_edge=arrays[9],
            max_intensity_edge=arrays[10],
            mass_displacement=arrays[11],
            lower_quartile_intensity=lower,
            median_intensity=median,
            mad_intensity=mad,
            upper_quartile_intensity=upper,
            center_mass_intensity_x=arrays[12],
            center_mass_intensity_y=arrays[13],
            center_mass_intensity_z=np.zeros(object_count, dtype=np.float64),
            max_intensity_x=np.zeros(object_count, dtype=np.float64),
            max_intensity_y=np.zeros(object_count, dtype=np.float64),
            max_intensity_z=np.zeros(object_count, dtype=np.float64),
        ).with_max_intensity_positions(
            self.maximum_intensity_positions(image_array, labels)
        )

    def measure_prepared_batch(
        self, images: tuple[np.ndarray, ...], labels: ObjectIntensityPreparedLabels
    ) -> tuple[ObjectIntensityArrays, ...]:
        """Measure a homogeneous 3-D image batch against one label domain."""
        if not images:
            return ()
        image_arrays = tuple((np.ascontiguousarray(image) for image in images))
        if not all((image.ndim == 3 for image in image_arrays)):
            return super().measure_prepared_batch(images, labels)
        image_shape = image_arrays[0].shape
        if any((image.shape != image_shape for image in image_arrays)):
            raise ValueError("Batched object-intensity images must share shape.")
        relabeled_labels = labels.relabeled_labels
        if relabeled_labels.ndim != 3:
            return super().measure_prepared_batch(images, labels)
        if image_shape != relabeled_labels.shape:
            raise ValueError("image and labels must have matching shapes.")
        object_labels = labels.object_labels
        if labels.object_count == 0:
            return tuple(
                (ObjectIntensityArrays.empty(object_labels) for _image in images)
            )
        image_batch = np.ascontiguousarray(np.stack(image_arrays, axis=0))
        label_array = np.ascontiguousarray(relabeled_labels, dtype=np.int32)
        sparse_foreground_index = self._sparse_foreground_index_for_batch(
            labels, image_shape
        )
        if sparse_foreground_index is not None:
            measured_arrays = self._measure_sparse_prepared_batch(
                image_batch, labels, sparse_foreground_index
            )
        else:
            scan_started_at = time.perf_counter()
            scan_result = _object_intensity_scan_3d_batch_numba(
                image_batch, label_array, object_labels, labels.label_to_index
            )
            RuntimeProfileLogger.log(
                logger,
                "object_intensity_scan_3d_batch",
                time.perf_counter() - scan_started_at,
                images=len(images),
                objects=labels.object_count,
                voxels=image_arrays[0].size,
            )
            quantile_started_at = time.perf_counter()
            quantile_result = _object_intensity_quantiles_3d_batch_numba(
                image_batch,
                label_array,
                labels.label_to_index,
                scan_result[0].astype(np.int64, copy=False),
                1.0 / 3.0,
            )
            RuntimeProfileLogger.log(
                logger,
                "object_intensity_quantiles_3d_batch",
                time.perf_counter() - quantile_started_at,
                images=len(images),
                objects=labels.object_count,
                voxels=image_arrays[0].size,
            )
            measured_arrays = tuple(
                ObjectIntensityArrays.from_3d_scan_batch_result(
                    object_labels=object_labels,
                    scan_result=scan_result,
                    quantile_result=quantile_result,
                    image_index=image_index,
                )
                for image_index in range(len(images))
            )
        maximum_positions = self.maximum_intensity_positions_batch(
            image_batch,
            labels,
        )
        return tuple(
            measured.with_max_intensity_positions(positions)
            for measured, positions in zip(
                measured_arrays,
                maximum_positions,
                strict=True,
            )
        )

    def _sparse_foreground_index_for_batch(
        self, labels: ObjectIntensityPreparedLabels, image_shape: tuple[int, ...]
    ) -> ObjectIntensityForegroundIndex | None:
        """Return the prepared sparse index when it is cheaper than dense scan."""
        if len(image_shape) != 3:
            return None
        foreground_index = labels.foreground_index
        if foreground_index is None:
            return None
        shape_3d = (int(image_shape[0]), int(image_shape[1]), int(image_shape[2]))
        if not foreground_index.matches_shape(shape_3d):
            return None
        if foreground_index.foreground_fraction() > self.sparse_foreground_max_fraction:
            return None
        return foreground_index

    def _measure_sparse_prepared_batch(
        self,
        image_batch: np.ndarray,
        labels: ObjectIntensityPreparedLabels,
        foreground_index: ObjectIntensityForegroundIndex,
    ) -> tuple[ObjectIntensityArrays, ...]:
        """Measure sparse 3-D foreground voxels for a prepared label domain."""
        scan_started_at = time.perf_counter()
        scan_result = _object_intensity_scan_3d_sparse_batch_numba(
            image_batch,
            foreground_index.z_indices,
            foreground_index.y_indices,
            foreground_index.x_indices,
            foreground_index.object_indexes,
            foreground_index.edge_flags,
            labels.object_count,
        )
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_scan_3d_sparse_batch",
            time.perf_counter() - scan_started_at,
            images=image_batch.shape[0],
            objects=labels.object_count,
            foreground_voxels=foreground_index.voxel_count,
        )
        quantile_started_at = time.perf_counter()
        quantile_result = _object_intensity_quantiles_3d_sparse_batch_numba(
            image_batch,
            foreground_index.z_indices,
            foreground_index.y_indices,
            foreground_index.x_indices,
            foreground_index.object_indexes,
            scan_result[0].astype(np.int64, copy=False),
            1.0 / 3.0,
        )
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_quantiles_3d_sparse_batch",
            time.perf_counter() - quantile_started_at,
            images=image_batch.shape[0],
            objects=labels.object_count,
            foreground_voxels=foreground_index.voxel_count,
        )
        object_labels = labels.object_labels
        return tuple(
            (
                ObjectIntensityArrays.from_3d_scan_batch_result(
                    object_labels=object_labels,
                    scan_result=scan_result,
                    quantile_result=quantile_result,
                    image_index=image_index,
                )
                for image_index in range(image_batch.shape[0])
            )
        )

    def _measure_3d_prepared(
        self, image_array: np.ndarray, labels: ObjectIntensityPreparedLabels
    ) -> ObjectIntensityArrays:
        if image_array.shape != labels.relabeled_labels.shape:
            raise ValueError("image and labels must have matching shapes.")
        if labels.object_count == 0:
            return ObjectIntensityArrays.empty(labels.object_labels)
        return self.measure_prepared_batch((image_array,), labels)[0]

    def prepare_backend(self) -> None:
        """Compile object-intensity kernels outside measured execution."""
        image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
        labels = np.zeros(image.shape, dtype=np.int32)
        labels[4:16, 4:16] = 1
        labels[16:28, 16:28] = 2
        label_payload = ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
        )
        self.measure(image, label_payload)
        image_3d = np.linspace(0.0, 1.0, 8 * 16 * 16, dtype=np.float32).reshape(
            (8, 16, 16)
        )
        labels_3d = np.zeros(image_3d.shape, dtype=np.int32)
        labels_3d[1:4, 3:9, 3:9] = 1
        labels_3d[4:7, 7:14, 7:14] = 2
        label_payload_3d = ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels_3d),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
        )
        self.measure(image_3d, label_payload_3d)
        prepared_3d = ObjectIntensityPreparedLabels.from_source(
            label_payload_3d,
            labels_3d,
        )
        self.measure_prepared_batch(
            (image_3d, np.ascontiguousarray(1.0 - image_3d)), prepared_3d
        )


def object_intensity_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> ObjectIntensityBackendStrategy:
    """Return the selected object-intensity backend."""
    return ObjectIntensityBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


def measure_object_intensity_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[object]:
    return [
        request.execute_one(slice_index) for slice_index in range(request.slice_count)
    ]


def measure_object_intensity_measurement_image_batch(
    func: Callable[..., object],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[
        [Callable[..., object], RuntimeBatchInvocationRequest], object
    ],
) -> list[object]:
    """Batch intensity measurement images that share one object-label domain."""
    outputs: list[object | None] = [None] * len(requests)
    for group in _object_intensity_batch_groups(requests):
        if len(group) <= 1:
            index, request = group[0]
            outputs[index] = execute_request(func, request)
            continue
        first_request = group[0][1]
        prepared_labels = _object_intensity_prepared_labels_for_batch_group(group)
        if prepared_labels is None:
            for index, request in group:
                outputs[index] = execute_request(func, request)
            continue
        backend = object_intensity_backend(
            backend_provider=ObjectIntensityMeasurementContext.from_runtime_request(
                first_request
            ).backend_provider
        )
        images = tuple(
            (np.asarray(image_payload_data(request.image)) for _index, request in group)
        )
        batch_started_at = time.perf_counter()
        measurement_batches = backend.measure_prepared_batch(images, prepared_labels)
        RuntimeProfileLogger.log(
            logger,
            "object_intensity_prepared_batch",
            time.perf_counter() - batch_started_at,
            images=len(images),
            objects=prepared_labels.object_count,
        )
        for measurement_arrays, (index, request) in zip(
            measurement_batches, group, strict=True
        ):
            rows_started_at = time.perf_counter()
            rows = ObjectIntensityMeasurementRows.from_arrays(
                measurement_arrays,
                slice_index=ObjectIntensityMeasurementContext.from_runtime_request(
                    request
                ).slice_index,
                object_domain=prepared_labels.object_domain,
                object_row_identity=prepared_labels.measurement_row_identity,
            )
            RuntimeProfileLogger.log(
                logger,
                "object_intensity_rows_from_arrays",
                time.perf_counter() - rows_started_at,
                rows=len(rows),
            )
            outputs[index] = (request.image, rows)
    return [output for output in outputs if output is not None]


def _object_intensity_prepared_labels_for_batch_group(
    group: tuple[tuple[int, RuntimeBatchInvocationRequest], ...],
) -> ObjectIntensityPreparedLabels | None:
    first_request = group[0][1]
    if not isinstance(first_request, PreparedObjectMeasurementInvocation):
        return None
    context = ObjectIntensityMeasurementContext.from_runtime_request(first_request)
    labels = first_request.completion_label_payload
    if not isinstance(labels, ObjectLabelValue):
        return None
    if labels.plane_axis is RuntimePlaneAxis.SOURCE_BINDING:
        return None
    return ObjectIntensityPreparedLabels.from_measurement(
        image=image_payload_data(first_request.image),
        labels=labels,
        slice_index=context.slice_index,
    )


def _object_intensity_batch_groups(
    requests: tuple[RuntimeBatchInvocationRequest, ...],
) -> tuple[tuple[tuple[int, RuntimeBatchInvocationRequest], ...], ...]:
    grouped_requests: dict[
        tuple[tuple[str, Hashable], ...],
        list[tuple[int, RuntimeBatchInvocationRequest]],
    ] = {}
    singleton_groups: list[tuple[tuple[int, RuntimeBatchInvocationRequest], ...]] = []
    for index, request in enumerate(requests):
        key = _object_intensity_batch_key(request)
        if key is None:
            singleton_groups.append(((index, request),))
            continue
        if key not in grouped_requests:
            grouped_requests[key] = []
        grouped_requests[key].append((index, request))
    return (*(tuple(group) for group in grouped_requests.values()), *singleton_groups)


def _object_intensity_batch_key(
    request: RuntimeBatchInvocationRequest,
) -> tuple[tuple[str, Hashable], ...] | None:
    if request.execution_mode is not ImagePayloadExecutionMode.FULL_STACK:
        return None
    semantic_group_key = request.semantic_group_key
    if semantic_group_key is None:
        return None
    context = ObjectIntensityMeasurementContext.from_runtime_request(request)
    label_source = context.labels
    if isinstance(request, PreparedObjectMeasurementInvocation):
        label_source = request.completion_label_payload
    label_identity: tuple[tuple[str, Hashable], ...] = ()
    if isinstance(label_source, ObjectLabelValue):
        label_identity = (
            ("object_labels", label_source.object_label_semantic_identity()),
        )
    return (
        ("semantic_group_key", semantic_group_key),
        *label_identity,
        *context.batch_key_items(),
    )


@numpy_decorator(contract=ProcessingContract.FLEXIBLE)
@special_inputs("labels")
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.SLICE_ALIGNED)
@runtime_bound_parameters(SliceIndexRuntimeParameter)
def measure_object_intensity(
    image: np.ndarray,
    labels: ObjectLabelValue,
    object_intensity_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int = OBJECT_INTENSITY_DEFAULT_SLICE_INDEX,
    object_intensity_prepared_labels: ObjectIntensityPreparedLabels | None = None,
) -> tuple[np.ndarray, ObjectIntensityMeasurementRows]:
    """Measure CellProfiler intensity features for identified objects.

    Args:
        labels: Object-label plane defining the regions that receive separate
            intensity measurements.
        object_intensity_prepared_labels: Prepared label-index projection reused
            by batched object measurements; leave unset for ordinary calls.
    """
    context = ObjectIntensityMeasurementContext.from_function_arguments(
        labels=labels,
        backend_provider=object_intensity_backend_provider,
        slice_index=slice_index,
        prepared_labels=object_intensity_prepared_labels,
    )
    if context.prepared_labels is not None:
        return (
            image,
            ObjectIntensityMeasurementRequest(
                image=image,
                labels=context.labels,
                backend_provider=context.backend_provider,
                slice_index=context.slice_index,
                object_domain=context.object_domain,
                object_row_identity=context.object_row_identity,
                prepared_labels=context.prepared_labels,
            ).measurements(),
        )
    measurement_label_array = object_label_dense_array(context.labels, dtype=np.int32)
    return (
        image,
        ObjectIntensityMeasurementRequest(
            image=image,
            labels=context.labels,
            backend_provider=context.backend_provider,
            slice_index=context.slice_index,
            prepared_labels=context.prepared_labels,
            object_domain=dense_object_label_measurement_row_domain(
                context.labels, measurement_label_array
            ),
        ).measurements(),
    )


def prepare_measure_object_intensity() -> None:
    """Prepare object-intensity backend kernels before measured execution."""
    ObjectIntensityBackendStrategy.prepare_registered_family()


@numpy_decorator(contract=ProcessingContract.PURE_2D)
def measure_image_intensity(
    image: np.ndarray,
    calculate_percentiles: bool = False,
    percentiles: tuple[int, ...] = (10, 90),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure intensity across the declared image."""
    image_array = np.asarray(image)
    measurements = ImageIntensityMeasurement.from_pixels(
        image_array.flatten(),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=calculate_percentiles,
            percentiles=percentiles,
        ),
    )
    return (
        image_array,
        DataclassMeasurementColumnarRows(
            (measurements,),
            row_type=ImageIntensityMeasurement,
        ),
    )


@numpy_decorator(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def measure_image_intensity_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    calculate_percentiles: bool = False,
    percentiles: tuple[int, ...] = (10, 90),
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure image intensity within one declared object set.

    Args:
        labels: Object-label plane whose positive pixels select the image pixels
            included in the aggregate intensity measurement.
    """
    image_array = np.asarray(image)
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if image_array.ndim != 2 or label_array.ndim != 2:
        raise ValueError(
            "MeasureImageIntensity object labels must already be projected "
            f"to one 2-D image plane; got image {image_array.shape!r} and "
            f"labels {label_array.shape!r}."
        )
    if image_array.shape != label_array.shape:
        raise ValueError(
            "MeasureImageIntensity image and projected object labels must "
            f"share a shape; got image {image_array.shape!r} and labels "
            f"{label_array.shape!r}."
        )
    measurements = ImageIntensityMeasurement.from_pixels(
        image_array[label_array > 0].flatten(),
        percentile_spec=ImageIntensityPercentileSpec(
            enabled=calculate_percentiles,
            percentiles=percentiles,
        ),
    )
    return (
        image_array,
        DataclassMeasurementColumnarRows(
            (measurements,),
            row_type=ImageIntensityMeasurement,
        ),
    )


@dataclass(frozen=True, slots=True)
class RescaleIntensityContext:
    """Normalized settings and image data for one intensity rescale operation."""

    data: np.ndarray
    automatic_low: AutomaticLow
    automatic_high: AutomaticHigh
    source_low: float
    source_high: float
    dest_low: float
    dest_high: float
    divisor_value: float

    @classmethod
    def from_settings(
        cls,
        image: np.ndarray,
        *,
        automatic_low: AutomaticLow,
        automatic_high: AutomaticHigh,
        source_low: float,
        source_high: float,
        dest_low: float,
        dest_high: float,
        divisor_value: float,
    ) -> "RescaleIntensityContext":
        source_data = np.asarray(image_payload_data(image))
        return cls(
            data=source_data.astype(np.float32, copy=False),
            automatic_low=coerce_cellprofiler_enum(AutomaticLow, automatic_low),
            automatic_high=coerce_cellprofiler_enum(AutomaticHigh, automatic_high),
            source_low=source_low,
            source_high=source_high,
            dest_low=dest_low,
            dest_high=dest_high,
            divisor_value=divisor_value,
        )

    @property
    def source_range(self) -> tuple[float, float]:
        return rescale_source_range(
            self.data,
            self.automatic_low,
            self.automatic_high,
            self.source_low,
            self.source_high,
        )

    def preserves_unit_interval_intensity_scale(
        self, rescale_method: RescaleMethod
    ) -> bool:
        """Return whether declared rescale settings are a unit-interval identity."""
        return rescale_method.preserves_unit_interval_intensity_scale(
            source_range=self.source_range,
            destination_range=(self.dest_low, self.dest_high),
        )

    def linearly_rescaled(
        self, source_range: tuple[float, float], destination_range: tuple[float, float]
    ) -> np.ndarray:
        """Rescale with CellProfiler/skimage tuple-range clipping semantics."""
        source_low, source_high = source_range
        destination_low, destination_high = destination_range
        result = np.empty_like(self.data, dtype=np.float32)
        np.clip(self.data, source_low, source_high, out=result)
        if source_low == source_high:
            np.clip(result, destination_low, destination_high, out=result)
            return result
        result -= source_low
        result /= source_high - source_low
        result *= destination_high - destination_low
        result += destination_low
        return result

    def divided_by(self, divisor: float) -> np.ndarray:
        """Return image data divided by a scalar as a float32 result."""
        result = np.empty_like(self.data, dtype=np.float32)
        np.divide(self.data, divisor, out=result)
        return result


class RescaleMethodRunner(ABC, metaclass=AutoRegisterMeta):
    """Registered implementation for one CellProfiler rescale method."""

    __registry_key__ = "rescale_method"
    __skip_if_no_key__ = True
    rescale_method: ClassVar[RescaleMethod | None] = None

    @classmethod
    def for_method(cls, method: RescaleMethod) -> "RescaleMethodRunner":
        return cls.__registry__[method]()

    @abstractmethod
    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        """Return float32 rescaled image data."""


class StretchRescaleMethodRunner(RescaleMethodRunner):
    """Stretch image intensities to the unit interval."""

    rescale_method = RescaleMethod.STRETCH

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        in_min = np.min(context.data)
        in_max = np.max(context.data)
        if in_min == in_max:
            return np.zeros_like(context.data)
        source_low = float(in_min)
        source_high = float(in_max)
        result = np.empty_like(context.data, dtype=np.float32)
        np.subtract(context.data, source_low, out=result)
        result /= source_high - source_low
        # Retain the legacy destination-offset pass for exact signed-zero semantics.
        result += 0.0
        return result


class ManualInputRangeRescaleMethodRunner(RescaleMethodRunner):
    """Rescale from a declared input range to the unit interval."""

    rescale_method = RescaleMethod.MANUAL_INPUT_RANGE

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        return context.linearly_rescaled(context.source_range, (0.0, 1.0))


class ManualIoRangeRescaleMethodRunner(RescaleMethodRunner):
    """Rescale from a declared input range to a declared output range."""

    rescale_method = RescaleMethod.MANUAL_IO_RANGE

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        return context.linearly_rescaled(
            context.source_range, (context.dest_low, context.dest_high)
        )


class DivideByImageMinimumRescaleMethodRunner(RescaleMethodRunner):
    """Divide image intensities by the image minimum."""

    rescale_method = RescaleMethod.DIVIDE_BY_IMAGE_MINIMUM

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        src_min = np.min(context.data)
        if src_min == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        return context.divided_by(float(src_min))


class DivideByImageMaximumRescaleMethodRunner(RescaleMethodRunner):
    """Divide image intensities by the image maximum."""

    rescale_method = RescaleMethod.DIVIDE_BY_IMAGE_MAXIMUM

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        src_max = np.max(context.data)
        if src_max == 0.0:
            src_max = 1.0
        return context.divided_by(float(src_max))


class DivideByValueRescaleMethodRunner(RescaleMethodRunner):
    """Divide image intensities by a declared scalar."""

    rescale_method = RescaleMethod.DIVIDE_BY_VALUE

    def run(self, context: RescaleIntensityContext) -> np.ndarray:
        if context.divisor_value == 0.0:
            raise ZeroDivisionError("Cannot divide pixel intensity by 0.")
        return context.divided_by(context.divisor_value)


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy_decorator(contract=ProcessingContract.PURE_2D)
def rescale_intensity(
    image: np.ndarray,
    rescale_method: RescaleMethod = RescaleMethod.STRETCH,
    automatic_low: AutomaticLow = AutomaticLow.EACH_IMAGE,
    automatic_high: AutomaticHigh = AutomaticHigh.EACH_IMAGE,
    source_low: float = 0.0,
    source_high: float = 1.0,
    dest_low: float = 0.0,
    dest_high: float = 1.0,
    divisor_value: float = 1.0,
) -> ImageIntensityOutput:
    """Rescale CellProfiler image intensity using its declared range policy.

    Args:
        dest_low: Output intensity assigned to the lower destination endpoint
            when the selected method maps into a declared output range.
        dest_high: Output intensity assigned to the upper destination endpoint
            when the selected method maps into a declared output range.
    """
    context = RescaleIntensityContext.from_settings(
        image,
        automatic_low=automatic_low,
        automatic_high=automatic_high,
        source_low=source_low,
        source_high=source_high,
        dest_low=dest_low,
        dest_high=dest_high,
        divisor_value=divisor_value,
    )
    rescaled = RescaleMethodRunner.for_method(rescale_method).run(context)
    metadata = image_payload_metadata(image)
    if not context.preserves_unit_interval_intensity_scale(rescale_method):
        metadata = metadata.without_unit_interval_intensity_scale()
    return with_image_payload_data(image, rescaled, metadata=metadata)


def rescale_source_range(
    data: np.ndarray,
    automatic_low: AutomaticLow,
    automatic_high: AutomaticHigh,
    source_low: float,
    source_high: float,
) -> tuple[float, float]:
    """Determine the CellProfiler source intensity range from settings."""
    src_min = (
        float(np.min(data)) if automatic_low == AutomaticLow.EACH_IMAGE else source_low
    )
    src_max = (
        float(np.max(data))
        if automatic_high == AutomaticHigh.EACH_IMAGE
        else source_high
    )
    return (src_min, src_max)


@numpy_decorator(contract=ProcessingContract.PURE_3D)
def rescale_intensity_match_maximum(image: np.ndarray) -> np.ndarray:
    """Scale image[0] so its maximum matches image[1]'s maximum."""
    input_data = image[0].astype(np.float64)
    reference_data = image[1].astype(np.float64)
    image_max = np.max(input_data)
    reference_max = np.max(reference_data)
    if image_max == 0:
        result = input_data
    else:
        result = input_data * reference_max / image_max
    return result.astype(np.float32)[np.newaxis, :, :]


measure_object_intensity.__openhcs_prepare__ = prepare_measure_object_intensity
measurement_image_batch_executor(measure_object_intensity_measurement_image_batch)(
    measure_object_intensity
)
pure_2d_batch_executor(measure_object_intensity_batch)(measure_object_intensity)


class MeasureObjectIntensityObjectMeasurementRowPolicy(
    DenseColumnarObjectMeasurementRowsMixin, DeclaredObjectMeasurementRowPolicy
):
    """Object-intensity rows are dense columnar rows over the declared domain."""

    missing_value_policy = (
        MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
    )


class MeasureObjectIntensityModule(
    LabelsObjectInputPolicy,
    MeasureObjectIntensityObjectMeasurementRowPolicy,
    PerObjectMeasurementExecutionModule,
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
    SourceQualifiedWideMeasurementRowsModule,
    IntensityFeature,
):
    module_name = "MeasureObjectIntensity"
    function_name = "measure_object_intensity"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("intensity",), ("location",))

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by MeasureObjectIntensity."""

        INTEGRATED_INTENSITY = ("IntegratedIntensity", (), (IntensityFeature,))
        MEAN_INTENSITY = ("MeanIntensity", (), (IntensityFeature,))
        STD_INTENSITY = ("StdIntensity", (), (IntensityFeature,))
        MIN_INTENSITY = ("MinIntensity", (), (IntensityFeature,))
        MAX_INTENSITY = ("MaxIntensity", (), (IntensityFeature,))
        INTEGRATED_INTENSITY_EDGE = (
            "IntegratedIntensityEdge",
            (),
            (IntensityFeature,),
        )
        MEAN_INTENSITY_EDGE = ("MeanIntensityEdge", (), (IntensityFeature,))
        STD_INTENSITY_EDGE = ("StdIntensityEdge", (), (IntensityFeature,))
        MIN_INTENSITY_EDGE = ("MinIntensityEdge", (), (IntensityFeature,))
        MAX_INTENSITY_EDGE = ("MaxIntensityEdge", (), (IntensityFeature,))
        MASS_DISPLACEMENT = ("MassDisplacement", (), (IntensityFeature,))
        LOWER_QUARTILE_INTENSITY = (
            "LowerQuartileIntensity",
            (),
            (IntensityFeature,),
        )
        MEDIAN_INTENSITY = ("MedianIntensity", (), (IntensityFeature,))
        MAD_INTENSITY = ("MADIntensity", (), (IntensityFeature,))
        UPPER_QUARTILE_INTENSITY = (
            "UpperQuartileIntensity",
            (),
            (IntensityFeature,),
        )
        CENTER_MASS_INTENSITY_X = (
            "CenterMassIntensity_X",
            (),
            (ObjectLocationFeature,),
        )
        CENTER_MASS_INTENSITY_Y = (
            "CenterMassIntensity_Y",
            (),
            (ObjectLocationFeature,),
        )
        CENTER_MASS_INTENSITY_Z = (
            "CenterMassIntensity_Z",
            (),
            (ObjectLocationFeature,),
        )
        MAX_INTENSITY_X = (
            "MaxIntensity_X",
            (),
            (MaxIntensityLocationFeature,),
        )
        MAX_INTENSITY_Y = (
            "MaxIntensity_Y",
            (),
            (MaxIntensityLocationFeature,),
        )
        MAX_INTENSITY_Z = (
            "MaxIntensity_Z",
            (),
            (MaxIntensityLocationFeature,),
        )

        def source_qualified_name(self, source_image_name: str) -> str:
            """Return this member's exact CellProfiler source-qualified name."""
            category_qualifiers = tuple(
                dict.fromkeys(
                    marker_type.family_qualifier
                    for marker_type in self.semantic_markers
                    if marker_type.family_qualifier is not None
                )
            )
            if len(category_qualifiers) != 1:
                raise ValueError(
                    f"{type(self).__name__}.{self.name} requires exactly one "
                    f"measurement category, got {category_qualifiers!r}."
                )
            category = "".join(
                part[:1].upper() + part[1:]
                for part in category_qualifiers[0].split("_")
            )
            return "_".join((category, self.value, source_image_name))

    ignored_settings = (
        "Select images to measure",
        "Select objects to measure",
        "Hidden",
    )

    @classmethod
    def source_qualified_feature_name(
        cls,
        field_name: str,
        source_image_name: str,
    ) -> str:
        """Resolve a raw row field through its nominal feature declaration."""
        matches = tuple(
            feature
            for feature in cls.MeasurementFeature
            if feature.measurement_row_field_name == field_name
        )
        if len(matches) != 1:
            raise ValueError(
                f"{cls.__name__} requires exactly one measurement feature for raw "
                f"field {field_name!r}, got {[feature.value for feature in matches]!r}."
            )
        return matches[0].source_qualified_name(source_image_name)

    @classmethod
    def derived_measurement_feature_relation_declarations(
        cls,
    ) -> tuple[RuntimeMeasurementFeatureRelationDeclaration, ...]:
        """Return marker-derived max-location to max-intensity relations."""
        features = tuple(cls.MeasurementFeature)
        return tuple(
            RuntimeMeasurementFeatureRelationDeclaration(
                source_feature,
                TieSensitiveLocationValueFeatureRelation(
                    target_feature=cls._single_target_feature_for_source(
                        source_feature,
                        features,
                        source_marker=MaxIntensityLocationFeature,
                        target_marker=IntensityFeature,
                    ),
                    source_marker=MaxIntensityLocationFeature,
                    target_marker=IntensityFeature,
                ),
            )
            for source_feature in features
            if MaxIntensityLocationFeature.matches_feature(source_feature)
        )

    @classmethod
    def _single_target_feature_for_source(
        cls,
        source_feature: RuntimeMeasurementFeature,
        features: tuple[RuntimeMeasurementFeature, ...],
        *,
        source_marker: type[RuntimeMeasurementFeatureSemanticMarker],
        target_marker: type[RuntimeMeasurementFeatureSemanticMarker],
    ) -> RuntimeMeasurementFeature:
        """Return the single marker-owned value target for an axis-suffixed source."""
        if not source_marker.matches_feature(source_feature):
            raise ValueError(
                f"{cls.__name__}.{source_feature.name} must carry "
                f"{source_marker.__name__}."
            )
        source_stem, separator, _axis_token = (
            source_feature.feature_family().rpartition("_")
        )
        if not separator:
            raise ValueError(
                f"{cls.__name__}.{source_feature.name} must have an axis suffix."
            )
        matches = tuple(
            target_feature
            for target_feature in features
            if target_marker.matches_feature(target_feature)
            and target_feature.feature_family() == source_stem
        )
        if len(matches) != 1:
            raise ValueError(
                f"{cls.__name__}.{source_feature.name} relation requires exactly "
                f"one {target_marker.__name__} target with family {source_stem!r}, "
                f"got {[feature.name for feature in matches]!r}."
            )
        return matches[0]


__all__ = public_names_from_objects(
    NumbaNumpyObjectIntensityBackendStrategy,
    AutomaticHigh,
    AutomaticLow,
    ImageIntensityMeasurement,
    ImageIntensityPercentileSpec,
    ObjectIntensityMeasurement,
    ObjectIntensityMeasurementRows,
    ObjectIntensityMeasurementRequest,
    ObjectIntensityPreparedLabels,
    RescaleMethod,
    "ObjectIntensityArrays",
    ObjectIntensityBackendStrategy,
    "ObjectIntensityLabelInput",
    measure_image_intensity,
    measure_image_intensity_objects,
    measure_object_intensity,
    measure_object_intensity_batch,
    object_intensity_backend,
    prepare_measure_object_intensity,
    rescale_intensity,
    rescale_intensity_match_maximum,
    rescale_source_range,
)
