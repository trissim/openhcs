"""Image-quality backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
import logging
import time
from types import MappingProxyType
from typing import Annotated, ClassVar, TYPE_CHECKING

from metaclass_registry import AutoRegisterMeta
from numba import njit
import numpy as np

from openhcs.constants.constants import MemoryType
from openhcs.core.artifacts import ArtifactSpec, ImageArtifactType
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementSparseColumnarRows,
    is_structural_missing_measurement_cell,
)
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_payload_intensity_scale,
    image_payload_metadata,
)
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxisValueProjection
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ImageMeasurementInputModule,
    SourceQualifiedMeasurementFeatureModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    SourceQualifiedInputPayloadMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
    measurement_source_image_name_for_slice,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
    threshold_primitives,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract

if TYPE_CHECKING:
    from openhcs.core.function_patterns import FunctionInvocationKey
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext


class ImageQualityThresholdMethod(Enum):
    """Threshold algorithms exposed by MeasureImageQuality settings."""

    def __new__(
        cls,
        value: str,
        feature_field_name: str,
    ):
        member = object.__new__(cls)
        member._value_ = value
        member._feature_field_name = feature_field_name
        return member

    OTSU = ("otsu", "threshold_otsu")
    LI = ("li", "threshold_li")
    TRIANGLE = ("triangle", "threshold_triangle")
    ISODATA = ("isodata", "threshold_isodata")
    MINIMUM = ("minimum", "threshold_minimum")
    MEAN = ("mean", "threshold_mean")
    YEN = ("yen", "threshold_yen")

    @property
    def feature_field_name(self) -> str:
        """Return the field-derived native threshold feature member."""
        return self._feature_field_name

    def descriptor_scale(
        self,
        *,
        otsu_class_count: CellProfilerOtsuMethod,
        otsu_objective: "ImageQualityOtsuObjective",
        assign_middle_to_foreground: CellProfilerThresholdAssignment,
    ) -> str | None:
        """Return the exact native descriptor for this threshold configuration."""
        if self is not ImageQualityThresholdMethod.OTSU:
            return None
        class_token = (
            "2"
            if otsu_class_count is CellProfilerOtsuMethod.TWO_CLASS
            else (
                "3F"
                if assign_middle_to_foreground
                is CellProfilerThresholdAssignment.FOREGROUND
                else "3B"
            )
        )
        return f"{class_token}{otsu_objective.descriptor_token}"


class ImageQualityOtsuObjective(Enum):
    """Objective used to select a CellProfiler image-quality Otsu threshold."""

    WEIGHTED_VARIANCE = ("Weighted variance", "W")
    ENTROPY = ("Entropy", "S")

    def __new__(cls, value: str, descriptor_token: str):
        member = object.__new__(cls)
        member._value_ = value
        member._descriptor_token = descriptor_token
        return member

    @property
    def descriptor_token(self) -> str:
        """Return the token used by CellProfiler threshold feature names."""
        return self._descriptor_token


class ImageQualityThresholdAggregate(Enum):
    """Experiment-level reductions declared by MeasureImageQuality."""

    def __new__(cls, value: str, reducer: Callable[[np.ndarray], np.floating]):
        member = object.__new__(cls)
        member._value_ = value
        member._reducer = reducer
        return member

    MEAN = ("mean", np.mean)
    MEDIAN = ("median", np.median)
    STANDARD_DEVIATION = ("std", np.std)

    def feature_field_name(self, method: ImageQualityThresholdMethod) -> str:
        """Return the field-derived native aggregate feature member."""

        return f"threshold_{self.value}_{method.value}"

    def reduce(self, values: Sequence[float]) -> float:
        """Reduce finite plate values with CellProfiler's population statistic."""

        return float(self._reducer(np.asarray(values, dtype=np.float64)))


class MeasureImageQualityModule(
    SourceQualifiedInputPayloadMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    SourceQualifiedMeasurementFeatureModule,
    ImageMeasurementInputModule,
):
    module_name = "MeasureImageQuality"
    function_name = "measure_image_quality"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (("image", "quality"), ("quality",))
    measurement_feature_family = "ImageQuality"
    scale_qualified_measurement_feature_prefixes = (("local", "focus", "score"),)
    image_selection_setting = "Calculate metrics for which images?"
    all_loaded_images_selection = "All loaded images"
    selected_images_selection = "Select..."
    selected_images_setting: ClassVar[str] = "Select the images to measure"
    image_count_setting = "Image count"
    scale_count_setting = "Scale count"
    threshold_count_setting = "Threshold count"
    include_scaling_setting = "Include the image rescaling value?"
    calculate_blur_setting = "Calculate blur metrics?"
    blur_scale_setting = "Spatial scale for blur measurements"
    calculate_saturation_setting = "Calculate saturation metrics?"
    calculate_intensity_setting = "Calculate intensity metrics?"
    calculate_threshold_setting = "Calculate thresholds?"
    use_all_threshold_methods_setting = "Use all thresholding methods?"
    threshold_method_setting = "Select a thresholding method"
    object_fraction_setting = "Typical fraction of the image covered by objects"
    otsu_class_count_setting = "Two-class or three-class thresholding?"
    otsu_objective_setting = "Minimize the weighted variance or the entropy?"
    otsu_assignment_setting = "Assign pixels in the middle intensity class to the foreground or the background?"
    ignored_settings = (
        image_count_setting,
        image_selection_setting,
        threshold_count_setting,
        use_all_threshold_methods_setting,
        scale_count_setting,
    )
    setting_bindings = (
        SettingToKeywordBinding(
            include_scaling_setting,
            "include_scaling",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            calculate_blur_setting, "calculate_blur", parse_cellprofiler_bool
        ),
        SettingToKeywordBinding(
            calculate_saturation_setting,
            "calculate_saturation",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            calculate_intensity_setting,
            "calculate_intensity",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            calculate_threshold_setting,
            "calculate_threshold",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            object_fraction_setting,
            "object_fraction",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            blur_scale_setting,
            "blur_scales",
            parse_cellprofiler_int,
            repeated=True,
        ),
        SettingToKeywordBinding(
            threshold_method_setting,
            "threshold_method",
            cellprofiler_enum_value_setting_parser(ImageQualityThresholdMethod),
        ),
        SettingToKeywordBinding(
            otsu_class_count_setting,
            "otsu_class_count",
            cellprofiler_enum_value_setting_parser(CellProfilerOtsuMethod),
        ),
        SettingToKeywordBinding(
            otsu_objective_setting,
            "otsu_objective",
            cellprofiler_enum_value_setting_parser(ImageQualityOtsuObjective),
        ),
        SettingToKeywordBinding(
            otsu_assignment_setting,
            "assign_middle_to_foreground",
            cellprofiler_enum_value_setting_parser(CellProfilerThresholdAssignment),
        ),
    )

    measurement_feature_token_aliases = (("mad", "MAD"),)

    @classmethod
    def _threshold_feature_identity(
        cls,
        feature_name: str,
    ) -> tuple[ImageQualityThresholdMethod, str] | None:
        """Parse one exact source-qualified threshold feature emitted by this module."""

        for method in ImageQualityThresholdMethod:
            prefix = f"{cls.measurement_feature_name(method.feature_field_name)}_"
            if feature_name.startswith(prefix) and len(feature_name) > len(prefix):
                return method, feature_name[len(prefix) :]
        return None

    @classmethod
    def experiment_measurement_tables(
        cls,
        tables: Sequence[MeasurementTable],
    ) -> tuple[MeasurementTable, ...]:
        """Derive native threshold summary measurements across the plate."""

        values_by_identity: dict[
            tuple[ImageQualityThresholdMethod, str],
            list[float],
        ] = {}
        for table in tables:
            if table.subject.scope is not MeasurementScope.IMAGE:
                continue
            for field in table.rows.fields:
                identity = cls._threshold_feature_identity(field.name)
                if identity is None:
                    continue
                for value in table.rows.column_values(field.name):
                    if value is None or is_structural_missing_measurement_cell(value):
                        continue
                    numeric_value = float(value)
                    if np.isfinite(numeric_value):
                        values_by_identity.setdefault(identity, []).append(
                            numeric_value
                        )
        if not values_by_identity:
            return ()

        experiment_row: dict[str, float] = {}
        fields: list[FieldSpec] = []
        for (method, suffix), values in values_by_identity.items():
            for aggregate in ImageQualityThresholdAggregate:
                feature_name = cls.measurement_feature_name(
                    aggregate.feature_field_name(method),
                    suffix,
                )
                experiment_row[feature_name] = aggregate.reduce(values)
                fields.append(FieldSpec(feature_name, float, required=False))
        return (
            MeasurementTable(
                name=f"{cls.module_name}_experiment_measurements",
                rows=MeasurementProjectedColumnarRows(
                    MappingProxyType(
                        {field.name: (experiment_row[field.name],) for field in fields}
                    ),
                    fields=tuple(fields),
                ),
                subject=MeasurementSubject(MeasurementScope.EXPERIMENT),
                measurement_feature_owner=cls,
            ),
        )

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project image-quality records into source-qualified CP features."""

        source_metadata: ImagePayloadMetadata
        plane_projection: RuntimePlaneAxisValueProjection | None

        @classmethod
        def for_request(cls, module_type, request):
            return cls(
                request.output_value,
                module_type=module_type,
                source_metadata=image_payload_metadata(request.source.payload),
                plane_projection=request.source.plane_projection,
            )

        def rows(self) -> ColumnarRows:
            source_rows = self.source_rows()
            if not source_rows.fields:
                return MeasurementSparseColumnarRows.from_rows((), fields=())

            fields_by_name = {field.name: field for field in source_rows.fields}
            slice_field_name = MeasurementRowAxisField.SLICE_INDEX.value
            source_field_name = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
            scale_field_name = MeasurementRowAxisField.SCALE.value
            feature_field_name = MeasurementRowAxisField.FEATURE_NAME.value
            value_field_name = MeasurementRowValueField.RESULT_VALUE.value
            projected_fields: dict[str, FieldSpec] = {}
            if slice_field_name in fields_by_name:
                projected_fields[slice_field_name] = fields_by_name[slice_field_name]
            projected_fields[source_field_name] = FieldSpec(source_field_name, str)
            rows_by_slice: dict[int, dict[str, object]] = {}

            for source_row in source_rows.iter_row_mappings():
                slice_index = int(source_row[slice_field_name])
                source_image_name = measurement_source_image_name_for_slice(
                    self.source_metadata,
                    self.plane_projection,
                    slice_index,
                )
                projected_row = rows_by_slice.setdefault(
                    slice_index,
                    {
                        slice_field_name: slice_index,
                        source_field_name: source_image_name,
                    },
                )
                if projected_row[source_field_name] != source_image_name:
                    raise ValueError(
                        "MeasureImageQuality runtime slice has conflicting source "
                        f"identity: {projected_row[source_field_name]!r} != "
                        f"{source_image_name!r}."
                    )
                scale = source_row.get(scale_field_name)
                qualifiers = (source_image_name, scale)
                for field in source_rows.fields:
                    if field.name in MeasurementRowAxisField.field_names():
                        continue
                    if field.name in MeasurementRowValueField.field_names():
                        continue
                    value = source_row.get(field.name)
                    if value is None:
                        continue
                    projected_name = self.module_type.measurement_feature_name(
                        field.name,
                        *qualifiers,
                    )
                    if projected_name in projected_row:
                        raise ValueError(
                            "MeasureImageQuality emitted duplicate feature "
                            f"{projected_name!r} for runtime slice {slice_index}."
                        )
                    projected_row[projected_name] = value
                    projected_fields.setdefault(
                        projected_name,
                        FieldSpec(projected_name, field.dtype, required=False),
                    )

                feature_name = source_row.get(feature_field_name)
                if feature_name is None:
                    continue
                projected_name = self.module_type.measurement_feature_name(
                    str(feature_name),
                    *qualifiers,
                )
                if projected_name in projected_row:
                    raise ValueError(
                        "MeasureImageQuality emitted duplicate threshold feature "
                        f"{projected_name!r} for runtime slice {slice_index}."
                    )
                projected_row[projected_name] = source_row[value_field_name]
                value_field = fields_by_name[value_field_name]
                projected_fields.setdefault(
                    projected_name,
                    FieldSpec(projected_name, value_field.dtype, required=False),
                )

            return MeasurementSparseColumnarRows.from_rows(
                tuple(rows_by_slice.values()),
                fields=tuple(projected_fields.values()),
            )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is not None:
            selection = optional_setting_value(module, cls.image_selection_setting)
            if (
                selection == cls.selected_images_selection
                and not cls._selected_image_names(module)
            ):
                raise ValueError(
                    f"Module {module.name}({module.module_num}) selects explicit "
                    "images but declares no image artifact identities."
                )
            if selection not in (
                None,
                cls.all_loaded_images_selection,
                cls.selected_images_selection,
            ):
                raise ValueError(
                    "Unsupported MeasureImageQuality image-selection mode "
                    f"{selection!r} in module "
                    f"{module.name}({module.module_num})."
                )
        return bindings

    @classmethod
    def artifact_inputs_for_binding(
        cls,
        module: ModuleBlock,
        *,
        binding: SettingToKeywordBinding,
        invocation_key: FunctionInvocationKey,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ArtifactSpec, ...]:
        """Resolve the all-loaded image role from its typed main-flow domain."""

        if (
            binding is not cls.image_measurement_binding
            or optional_setting_value(module, cls.image_selection_setting)
            != cls.all_loaded_images_selection
        ):
            return super().artifact_inputs_for_binding(
                module,
                binding=binding,
                invocation_key=invocation_key,
                step_context=step_context,
            )
        return tuple(
            cls.require_available_artifact_input(
                module,
                binding=binding,
                name=spec.name,
                invocation_key=invocation_key,
                step_context=step_context,
            )
            for spec in step_context.main_flow_artifacts.of_artifact_type(
                ImageArtifactType
            )
        )

    @classmethod
    def invocation_module_blocks(cls, module: ModuleBlock) -> tuple[ModuleBlock, ...]:
        """Expose each serialized image-settings group as one public invocation."""
        image_count_value = optional_setting_value(module, cls.image_count_setting)
        if image_count_value is None:
            return (module,)
        image_count = parse_cellprofiler_int(image_count_value)
        if image_count <= 1:
            return (module,)
        groups = cls._image_setting_group_records(module, image_count=image_count)
        return tuple(cls._image_setting_group_module(module, group) for group in groups)

    @classmethod
    def _image_setting_group_records(
        cls,
        module: ModuleBlock,
        *,
        image_count: int,
    ) -> tuple[tuple[ModuleSetting, ...], ...]:
        records = module.iter_settings()
        header_count = 2 + 2 * image_count
        if len(records) < header_count:
            raise ValueError(
                f"MeasureImageQuality({module.module_num}) declares {image_count} "
                "image setting groups but has an incomplete count header."
            )
        expected_header = (
            cls.image_selection_setting,
            cls.image_count_setting,
            *(
                setting_name
                for _group_index in range(image_count)
                for setting_name in (
                    cls.scale_count_setting,
                    cls.threshold_count_setting,
                )
            ),
        )
        actual_header = tuple(record.name for record in records[:header_count])
        if actual_header != expected_header:
            raise ValueError(
                f"MeasureImageQuality({module.module_num}) setting header does not "
                f"match revision-{module.variable_revision_number} schema: "
                f"{actual_header!r}."
            )

        scale_counts = tuple(
            parse_cellprofiler_int(records[2 + group_index * 2].value)
            for group_index in range(image_count)
        )
        threshold_counts = tuple(
            parse_cellprofiler_int(records[3 + group_index * 2].value)
            for group_index in range(image_count)
        )
        groups: list[tuple[ModuleSetting, ...]] = []
        cursor = header_count
        for group_index, (scale_count, threshold_count) in enumerate(
            zip(scale_counts, threshold_counts, strict=True)
        ):
            group_size = 7 + scale_count + threshold_count * 5
            group = records[cursor : cursor + group_size]
            if len(group) != group_size:
                raise ValueError(
                    f"MeasureImageQuality({module.module_num}) image setting group "
                    f"{group_index + 1} is incomplete."
                )
            expected_names = (
                cls.selected_images_setting,
                cls.include_scaling_setting,
                cls.calculate_blur_setting,
                *((cls.blur_scale_setting,) * scale_count),
                cls.calculate_saturation_setting,
                cls.calculate_intensity_setting,
                cls.calculate_threshold_setting,
                cls.use_all_threshold_methods_setting,
                *(
                    setting_name
                    for _threshold_index in range(threshold_count)
                    for setting_name in (
                        cls.threshold_method_setting,
                        cls.object_fraction_setting,
                        cls.otsu_class_count_setting,
                        cls.otsu_objective_setting,
                        cls.otsu_assignment_setting,
                    )
                ),
            )
            actual_names = tuple(record.name for record in group)
            if actual_names != expected_names:
                raise ValueError(
                    f"MeasureImageQuality({module.module_num}) image setting group "
                    f"{group_index + 1} does not match revision-"
                    f"{module.variable_revision_number} schema: {actual_names!r}."
                )
            groups.append(group)
            cursor += group_size
        if cursor != len(records):
            raise ValueError(
                f"MeasureImageQuality({module.module_num}) has unconsumed setting "
                f"rows after its {image_count} declared image groups."
            )
        return tuple(groups)

    @classmethod
    def _image_setting_group_module(
        cls,
        module: ModuleBlock,
        group: tuple[ModuleSetting, ...],
    ) -> ModuleBlock:
        scale_count = sum(record.name == cls.blur_scale_setting for record in group)
        threshold_count = sum(
            record.name == cls.threshold_method_setting for record in group
        )
        records = (
            ModuleSetting(cls.image_selection_setting, cls.selected_images_selection),
            ModuleSetting(cls.image_count_setting, "1"),
            ModuleSetting(cls.scale_count_setting, str(scale_count)),
            ModuleSetting(cls.threshold_count_setting, str(threshold_count)),
            *group,
        )
        return replace(
            module,
            setting_records=list(records),
        )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: "BoundModuleSettings",
    ) -> "BoundModuleSettings":
        kwargs = dict(bound.kwargs)
        selected_images = cls._selected_image_names(module)
        parameter_name = cls.image_measurement_binding.require_parameter_name()
        if selected_images:
            kwargs[parameter_name] = selected_images
        else:
            kwargs.pop(parameter_name, None)
        blur_scales = tuple(
            parse_cellprofiler_int(value)
            for value in setting_values(module, cls.blur_scale_setting)
        )
        if blur_scales:
            kwargs["blur_scales"] = blur_scales
        return BoundModuleSettings(
            kwargs,
            bound.unmapped_kwargs,
            bound.setting_coverage,
        )

    @classmethod
    def _selected_image_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        return tuple(
            image_name
            for value in setting_values(
                module,
                cls.image_measurement_binding.setting_name,
            )
            for image_name in split_symbol_names(value)
        )


logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)


@dataclass(frozen=True)
class _RadialSpectrumGeometry:
    radii: np.ndarray
    labels: np.ndarray


class ImageQualityMeasurementRecord(MeasurementFeatureRecord):
    """Producer-owned image-quality feature record."""


@dataclass
class ImageQualityIntensityMetrics(ImageQualityMeasurementRecord):
    """Intensity feature family emitted only when intensity metrics are enabled."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX] = 0
    total_area: float = 0.0
    total_intensity: float = 0.0
    mean_intensity: float = 0.0
    median_intensity: float = 0.0
    std_intensity: float = 0.0
    mad_intensity: float = 0.0
    min_intensity: float = 0.0
    max_intensity: float = 0.0


@dataclass
class ImageQualityBlurSummaryMetrics(ImageQualityMeasurementRecord):
    """Blur features without an indexed native descriptor."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX] = 0
    focus_score: float = 0.0
    power_log_log_slope: float = 0.0


@dataclass
class ImageQualitySaturationMetrics(ImageQualityMeasurementRecord):
    """Saturation feature family emitted only when explicitly enabled."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX] = 0
    percent_maximal: float = 0.0
    percent_minimal: float = 0.0


@dataclass
class ImageQualityScalingMetrics(ImageQualityMeasurementRecord):
    """Source intensity scale emitted only when explicitly enabled."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX] = 0
    scaling: float = float("nan")


@dataclass
class ImageQualityBlurMetrics(ImageQualityMeasurementRecord):
    """Blur features qualified by their native spatial scale."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX] = 0
    scale: Annotated[str | None, MeasurementRowAxisField.SCALE] = "20"
    local_focus_score: float = 0.0
    correlation: float = 0.0


@dataclass
class ImageQualityThresholdMetrics(ImageQualityMeasurementRecord):
    """One threshold feature with its method-owned native descriptor."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    feature_name: Annotated[str, MeasurementRowAxisField.FEATURE_NAME]
    scale: Annotated[str | None, MeasurementRowAxisField.SCALE]
    result_value: float = 0.0


_RADIAL_SPECTRUM_GEOMETRY_CACHE: OrderedDict[
    tuple[int, int], _RadialSpectrumGeometry
] = OrderedDict()
_RADIAL_SPECTRUM_GEOMETRY_CACHE_MAX_ENTRIES = 16


class ImageQualityBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Image-quality primitives keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        """Return CP-style Haralick H3 correlation for one image plane."""

    @abstractmethod
    def radial_power_spectrum(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return CP-style radial Fourier spectrum bins."""


@dataclass(frozen=True, slots=True)
class ImageQualityThresholdRequest:
    """Exact MeasureImageQuality threshold settings for one image plane."""

    values: np.ndarray
    object_fraction: float
    otsu_class_count: CellProfilerOtsuMethod
    otsu_objective: ImageQualityOtsuObjective
    assign_middle_to_foreground: CellProfilerThresholdAssignment


class ImageQualityThresholdStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered threshold primitive for MeasureImageQuality threshold metrics."""

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[ImageQualityThresholdMethod | None] = None

    @classmethod
    def for_method(
        cls, method: ImageQualityThresholdMethod
    ) -> "ImageQualityThresholdStrategy":
        strategy_type = cls.__registry__.get(method)
        if strategy_type is None:
            raise NotImplementedError(f"Threshold method {method} not supported.")
        return strategy_type()

    @abstractmethod
    def threshold(self, request: ImageQualityThresholdRequest) -> float:
        """Return the requested threshold for a non-constant image."""


class PrimitiveImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    """MeasureImageQuality threshold backed by one generic threshold primitive."""

    primitive: ClassVar[Callable[[object, np.ndarray], float]]

    def threshold(self, request: ImageQualityThresholdRequest) -> float:
        return float(type(self).primitive(threshold_primitives(), request.values))


class OtsuImageQualityThresholdStrategy(ImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.OTSU

    def threshold(self, request: ImageQualityThresholdRequest) -> float:
        import centrosome.threshold

        _local_threshold, global_threshold = centrosome.threshold.get_threshold(
            centrosome.threshold.TM_OTSU,
            centrosome.threshold.TM_GLOBAL,
            request.values,
            object_fraction=request.object_fraction,
            two_class_otsu=(
                request.otsu_class_count is CellProfilerOtsuMethod.TWO_CLASS
            ),
            use_weighted_variance=(
                request.otsu_objective is ImageQualityOtsuObjective.WEIGHTED_VARIANCE
            ),
            assign_middle_to_foreground=(
                request.assign_middle_to_foreground
                is CellProfilerThresholdAssignment.FOREGROUND
            ),
        )
        return float(global_threshold)


class LiImageQualityThresholdStrategy(PrimitiveImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.LI
    primitive = lambda primitives, values: primitives.li_threshold(values)


class TriangleImageQualityThresholdStrategy(PrimitiveImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.TRIANGLE
    primitive = lambda primitives, values: primitives.triangle_threshold(values)


class IsodataImageQualityThresholdStrategy(PrimitiveImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.ISODATA
    primitive = lambda primitives, values: primitives.isodata_threshold(values)


class MinimumImageQualityThresholdStrategy(PrimitiveImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.MINIMUM
    primitive = lambda primitives, values: primitives.minimum_threshold(values)


class MeanImageQualityThresholdStrategy(PrimitiveImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.MEAN
    primitive = lambda primitives, values: primitives.mean_threshold(values)


class YenImageQualityThresholdStrategy(PrimitiveImageQualityThresholdStrategy):
    method = ImageQualityThresholdMethod.YEN
    primitive = lambda primitives, values: primitives.yen_threshold(values)


class NumpyImageQualityBackendStrategy(ImageQualityBackendStrategy):
    """Independent NumPy implementation of image-quality primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(MemoryType.NUMPY)
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = False

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise NotImplementedError(
                f"Image-quality Haralick correlation currently supports 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        return _haralick_h3_numpy(image_array, int(scale))

    def radial_power_spectrum(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_array = np.asarray(image, dtype=np.float64)
        if image_array.ndim != 2:
            raise NotImplementedError(
                f"Image-quality radial power spectrum currently supports 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        return _radial_power_spectrum_numpy(image_array)


class NumbaNumpyImageQualityBackendStrategy(NumpyImageQualityBackendStrategy):
    """Numba-accelerated NumPy image-quality backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(25, dtype=np.float32).reshape((5, 5))
        self.haralick_h3(image, scale=1)

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        image_array = np.asarray(image, dtype=np.float32)
        if image_array.ndim != 2:
            raise NotImplementedError(
                f"Numba image-quality Haralick correlation currently supports 2-D NumPy planes, got shape {image_array.shape!r}."
            )
        return float(
            _haralick_h3_numba(
                np.ascontiguousarray(image_array, dtype=np.float32), int(scale)
            )
        )


class CentrosomeNumpyImageQualityBackendStrategy(ImageQualityBackendStrategy):
    """Explicit centrosome provider for image-quality primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.CENTROSOME
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.CENTROSOME
    is_default_backend = False

    def haralick_h3(self, image: np.ndarray, *, scale: int) -> float:
        import centrosome.haralick

        image_array = np.asarray(image, dtype=np.float32)
        value = centrosome.haralick.Haralick(
            image_array, np.ones(image_array.shape, dtype=int), 0, int(scale)
        ).H3()
        return _finite_scalar(value)

    def radial_power_spectrum(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import centrosome.radial_power_spectrum

        radii, magnitude, power = centrosome.radial_power_spectrum.rps(
            np.asarray(image)
        )
        return (np.asarray(radii), np.asarray(magnitude), np.asarray(power))


def image_quality_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> ImageQualityBackendStrategy:
    """Return the selected image-quality backend."""
    return ImageQualityBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


def image_quality_focus_score(pixel_data: np.ndarray) -> float:
    """Calculate CP normalized-variance focus score."""
    if pixel_data.size == 0:
        return 0.0
    return float(_focus_score_numba(np.ascontiguousarray(pixel_data, dtype=np.float64)))


def image_quality_local_focus_score(pixel_data: np.ndarray, scale: int) -> float:
    """Calculate CP local focus score using grid-based normalized variance."""
    if pixel_data.size == 0 or scale <= 0:
        return 0.0
    return float(
        _local_focus_score_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float64), int(scale)
        )
    )


def image_quality_haralick_correlation(
    pixel_data: np.ndarray,
    scale: int,
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> float:
    """Calculate CellProfiler's Haralick H3 image-quality correlation."""
    if pixel_data.size == 0:
        return 0.0
    return image_quality_backend(backend_provider=backend_provider).haralick_h3(
        pixel_data, scale=scale
    )


def image_quality_power_spectrum_slope(
    pixel_data: np.ndarray,
    *,
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> float:
    """Calculate CellProfiler's log-log radial power spectrum slope."""
    if pixel_data.size == 0 or not image_quality_has_multiple_unique_values(pixel_data):
        return 0.0
    radii, magnitude, power = image_quality_backend(
        backend_provider=backend_provider
    ).radial_power_spectrum(pixel_data)
    if np.sum(magnitude) <= 0:
        return 0.0
    valid = magnitude > 0
    radii = radii[valid].reshape((-1, 1))
    power = power[valid].reshape((-1, 1))
    if radii.shape[0] <= 1:
        return 0.0
    slope_value = _least_squares_log_log_slope_numba(
        np.ascontiguousarray(radii.ravel(), dtype=np.float64),
        np.ascontiguousarray(power.ravel(), dtype=np.float64),
    )
    return float(slope_value) if np.isfinite(slope_value) else 0.0


def image_quality_saturation(pixel_data: np.ndarray) -> tuple[float, float]:
    """Calculate percent of pixels at max and min values."""
    if pixel_data.size == 0:
        return (0.0, 0.0)
    pixel_count = pixel_data.size
    max_val = np.max(pixel_data)
    min_val = np.min(pixel_data)
    num_maximal = np.sum(pixel_data == max_val)
    num_minimal = np.sum(pixel_data == min_val)
    return (
        100.0 * float(num_maximal) / float(pixel_count),
        100.0 * float(num_minimal) / float(pixel_count),
    )


def image_quality_intensity_metrics(
    pixel_data: np.ndarray,
) -> ImageQualityIntensityMetrics:
    """Calculate intensity-based image quality metrics."""
    if pixel_data.size == 0:
        return ImageQualityIntensityMetrics(
            total_area=0.0,
            total_intensity=0.0,
            mean_intensity=0.0,
            median_intensity=0.0,
            std_intensity=0.0,
            mad_intensity=0.0,
            min_intensity=0.0,
            max_intensity=0.0,
        )
    pixel_median = np.median(pixel_data)
    return ImageQualityIntensityMetrics(
        total_area=float(pixel_data.size),
        total_intensity=float(np.sum(pixel_data)),
        mean_intensity=float(np.mean(pixel_data)),
        median_intensity=float(pixel_median),
        std_intensity=float(np.std(pixel_data)),
        mad_intensity=float(np.median(np.abs(pixel_data - pixel_median))),
        min_intensity=float(np.min(pixel_data)),
        max_intensity=float(np.max(pixel_data)),
    )


def image_quality_threshold(
    pixel_data: np.ndarray,
    method: ImageQualityThresholdMethod,
    *,
    object_fraction: float = 0.1,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    otsu_objective: ImageQualityOtsuObjective = (
        ImageQualityOtsuObjective.WEIGHTED_VARIANCE
    ),
    assign_middle_to_foreground: CellProfilerThresholdAssignment = (
        CellProfilerThresholdAssignment.FOREGROUND
    ),
) -> float:
    """Calculate an automatic threshold using a MeasureImageQuality method."""
    if pixel_data.size == 0 or not image_quality_has_multiple_unique_values(pixel_data):
        return 0.0
    values = pixel_data.astype(np.float32, copy=False)
    return ImageQualityThresholdStrategy.for_method(method).threshold(
        ImageQualityThresholdRequest(
            values=values,
            object_fraction=float(object_fraction),
            otsu_class_count=otsu_class_count,
            otsu_objective=otsu_objective,
            assign_middle_to_foreground=assign_middle_to_foreground,
        )
    )


def image_quality_has_multiple_unique_values(pixel_data: np.ndarray) -> bool:
    """Return whether ``np.unique(pixel_data)`` would contain more than one value."""
    return bool(
        _has_multiple_unique_values_numba(
            np.ascontiguousarray(pixel_data, dtype=np.float32)
        )
    )


def _haralick_h3_numpy(image: np.ndarray, scale: int) -> float:
    if image.size == 0 or scale < 1 or image.shape[1] <= scale:
        return 0.0
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    divisor = maximum - minimum if maximum > minimum else 1.0
    quantized = np.floor((image - minimum) / divisor * 8.0).astype(np.int16)
    quantized = np.clip(quantized, 0, 7)
    level_count = int(np.max(quantized)) + 1
    if level_count <= 0:
        return 0.0
    left = quantized[:, :-scale].ravel()
    right = quantized[:, scale:].ravel()
    pair_count = left.size
    if pair_count == 0:
        return 0.0
    flat_indexes = level_count * left + right
    matrix = (
        np.bincount(flat_indexes, minlength=level_count * level_count)
        .reshape(level_count, level_count)
        .astype(float)
    )
    return _haralick_h3_from_matrix(matrix / float(pair_count))


def _haralick_h3_from_matrix(matrix: np.ndarray) -> float:
    total = float(np.sum(matrix))
    if total <= 0.0:
        return 0.0
    matrix = matrix / total
    px = matrix.sum(axis=1)
    py = matrix.sum(axis=0)
    px_total = float(np.sum(px))
    py_total = float(np.sum(py))
    if px_total <= 0.0 or py_total <= 0.0:
        return 0.0
    px = px / px_total
    py = py / py_total
    levels = np.arange(matrix.shape[0], dtype=float) + 1.0
    mux = float(np.sum(levels * px))
    muy = float(np.sum(levels * py))
    sigmax = float(np.sqrt(np.sum((levels - mux) ** 2 * px)))
    sigmay = float(np.sqrt(np.sum((levels - muy) ** 2 * py)))
    if sigmax <= 0.0 or sigmay <= 0.0:
        return 0.0
    summed = float(np.sum(np.outer(levels, levels) * matrix))
    value = (summed - mux * muy) / (sigmax * sigmay)
    return value if np.isfinite(value) else 0.0


def _radial_power_spectrum_numpy(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.fftpack import fft2

    working = image.astype(np.float64, copy=False)
    if np.ptp(working) > 0.0:
        mean_value = float(np.mean(working))
        mad_value = float(np.median(np.abs(working - mean_value)))
        with np.errstate(divide="ignore", invalid="ignore"):
            working = working / mad_value
    centered = working - np.mean(working)
    magnitude = np.abs(fft2(centered))
    power = magnitude**2
    geometry = _radial_spectrum_geometry(image.shape)
    labels = geometry.labels
    if labels.size == 0:
        return (
            np.array([2], dtype=int),
            np.array([0], dtype=int),
            np.array([0], dtype=int),
        )
    radii_flat = geometry.radii.ravel()
    return (
        labels,
        np.bincount(
            radii_flat, weights=magnitude.ravel(), minlength=int(labels[-1]) + 1
        )[labels],
        np.bincount(radii_flat, weights=power.ravel(), minlength=int(labels[-1]) + 1)[
            labels
        ],
    )


def _radial_spectrum_geometry(shape: tuple[int, int]) -> _RadialSpectrumGeometry:
    key = (int(shape[0]), int(shape[1]))
    geometry = _RADIAL_SPECTRUM_GEOMETRY_CACHE.get(key)
    if geometry is not None:
        _RADIAL_SPECTRUM_GEOMETRY_CACHE.move_to_end(key)
        return geometry
    height, width = key
    row2 = np.arange(height).reshape((height, 1)) ** 2
    col2 = np.arange(width) ** 2
    radii2 = row2 + col2
    radii2 = np.minimum(radii2, np.flipud(radii2))
    radii2 = np.minimum(radii2, np.fliplr(radii2))
    max_width = min(height, width) / 8.0
    geometry = _RadialSpectrumGeometry(
        radii=np.floor(np.sqrt(radii2)).astype(int) + 1,
        labels=np.arange(2, int(np.floor(max_width)), dtype=int),
    )
    _RADIAL_SPECTRUM_GEOMETRY_CACHE[key] = geometry
    _RADIAL_SPECTRUM_GEOMETRY_CACHE.move_to_end(key)
    while (
        len(_RADIAL_SPECTRUM_GEOMETRY_CACHE)
        > _RADIAL_SPECTRUM_GEOMETRY_CACHE_MAX_ENTRIES
    ):
        _RADIAL_SPECTRUM_GEOMETRY_CACHE.popitem(last=False)
    return geometry


def _finite_scalar(value: object) -> float:
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        return 0.0
    scalar = float(array.ravel()[0])
    return scalar if np.isfinite(scalar) else 0.0


@njit(cache=True)
def _focus_score_numba(pixel_data: np.ndarray) -> float:
    flat = pixel_data.ravel()
    count = flat.size
    if count == 0:
        return 0.0
    total = 0.0
    for index in range(count):
        total += flat[index]
    mean_value = total / float(count)
    if mean_value <= 0.0:
        return 0.0
    squared_sum = 0.0
    for index in range(count):
        diff = flat[index] - mean_value
        squared_sum += diff * diff
    return squared_sum / (float(count) * mean_value)


@njit(cache=True)
def _local_focus_score_numba(pixel_data: np.ndarray, scale: int) -> float:
    height, width = pixel_data.shape
    if height == 0 or width == 0 or scale <= 0:
        return 0.0
    grid_rows = (height + scale - 1) // scale
    grid_cols = (width + scale - 1) // scale
    grid_count = grid_rows * grid_cols
    sums = np.zeros(grid_count, dtype=np.float64)
    counts = np.zeros(grid_count, dtype=np.int64)
    for row in range(height):
        grid_row = int(row * float(grid_rows) / float(height))
        if grid_row >= grid_rows:
            grid_row = grid_rows - 1
        for col in range(width):
            grid_col = int(col * float(grid_cols) / float(width))
            if grid_col >= grid_cols:
                grid_col = grid_cols - 1
            grid_index = grid_row * grid_cols + grid_col
            sums[grid_index] += pixel_data[row, col]
            counts[grid_index] += 1
    means = np.zeros(grid_count, dtype=np.float64)
    valid_count = 0
    for grid_index in range(grid_count):
        count = counts[grid_index]
        if count <= 0:
            continue
        mean_value = sums[grid_index] / count
        if mean_value != 0.0 and np.isfinite(mean_value):
            means[grid_index] = mean_value
            valid_count += 1
    if valid_count == 0:
        return 0.0
    squared_sums = np.zeros(grid_count, dtype=np.float64)
    for row in range(height):
        grid_row = int(row * float(grid_rows) / float(height))
        if grid_row >= grid_rows:
            grid_row = grid_rows - 1
        for col in range(width):
            grid_col = int(col * float(grid_cols) / float(width))
            if grid_col >= grid_cols:
                grid_col = grid_cols - 1
            grid_index = grid_row * grid_cols + grid_col
            mean_value = means[grid_index]
            diff = pixel_data[row, col] - mean_value
            squared_sums[grid_index] += diff * diff
    local_norm_var = np.empty(valid_count, dtype=np.float64)
    output_index = 0
    for grid_index in range(grid_count):
        mean_value = means[grid_index]
        if mean_value == 0.0 or not np.isfinite(mean_value):
            continue
        value = squared_sums[grid_index] / (counts[grid_index] * mean_value)
        if np.isfinite(value):
            local_norm_var[output_index] = value
            output_index += 1
    if output_index == 0:
        return 0.0
    values = local_norm_var[:output_index]
    median_value = np.median(values)
    if not np.isfinite(median_value) or median_value <= 0.0:
        return 0.0
    mean_value = 0.0
    for index in range(output_index):
        mean_value += values[index]
    mean_value /= output_index
    variance = 0.0
    for index in range(output_index):
        diff = values[index] - mean_value
        variance += diff * diff
    variance /= output_index
    return variance / median_value


@njit(cache=True)
def _least_squares_log_log_slope_numba(radii: np.ndarray, power: np.ndarray) -> float:
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    sum_xx = 0.0
    sum_xy = 0.0
    for index in range(radii.size):
        radius = radii[index]
        power_value = power[index]
        if radius <= 0.0 or power_value <= 0.0:
            continue
        x_value = np.log(radius)
        y_value = np.log(power_value)
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        count += 1
        sum_x += x_value
        sum_y += y_value
        sum_xx += x_value * x_value
        sum_xy += x_value * y_value
    if count <= 1:
        return 0.0
    denominator = float(count) * sum_xx - sum_x * sum_x
    if denominator == 0.0:
        return 0.0
    return (float(count) * sum_xy - sum_x * sum_y) / denominator


@njit(cache=True)
def _has_multiple_unique_values_numba(pixel_data: np.ndarray) -> bool:
    flat_size = pixel_data.size
    if flat_size <= 1:
        return False
    flat = pixel_data.ravel()
    first = flat[0]
    first_is_nan = np.isnan(first)
    for index in range(1, flat_size):
        value = flat[index]
        value_is_nan = np.isnan(value)
        if first_is_nan:
            if not value_is_nan:
                return True
        elif value_is_nan or value != first:
            return True
    return False


@njit(cache=True)
def _haralick_h3_numba(image: np.ndarray, scale: int) -> float:
    height, width = image.shape
    if height == 0 or width == 0 or scale < 1 or (width <= scale):
        return 0.0
    minimum = image[0, 0]
    maximum = image[0, 0]
    for y in range(height):
        for x in range(width):
            value = image[y, x]
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
    divisor = maximum - minimum
    if divisor <= 0.0:
        divisor = 1.0
    level_count = 1
    for y in range(height):
        for x in range(width):
            level = int((image[y, x] - minimum) / divisor * 8.0)
            if level < 0:
                level = 0
            elif level > 7:
                level = 7
            if level + 1 > level_count:
                level_count = level + 1
    matrix = np.zeros((level_count, level_count), dtype=np.float64)
    pair_count = 0
    for y in range(height):
        for x in range(width - scale):
            left = int((image[y, x] - minimum) / divisor * 8.0)
            if left < 0:
                left = 0
            elif left > 7:
                left = 7
            right = int((image[y, x + scale] - minimum) / divisor * 8.0)
            if right < 0:
                right = 0
            elif right > 7:
                right = 7
            matrix[left, right] += 1.0
            pair_count += 1
    if pair_count == 0:
        return 0.0
    for y in range(level_count):
        for x in range(level_count):
            matrix[y, x] /= pair_count
    px = np.zeros(level_count, dtype=np.float64)
    py = np.zeros(level_count, dtype=np.float64)
    for y in range(level_count):
        for x in range(level_count):
            px[y] += matrix[y, x]
            py[x] += matrix[y, x]
    px_total = 0.0
    py_total = 0.0
    for index in range(level_count):
        px_total += px[index]
        py_total += py[index]
    if px_total <= 0.0 or py_total <= 0.0:
        return 0.0
    for index in range(level_count):
        px[index] /= px_total
        py[index] /= py_total
    mux = 0.0
    muy = 0.0
    for index in range(level_count):
        level_value = index + 1.0
        mux += level_value * px[index]
        muy += level_value * py[index]
    sigmax2 = 0.0
    sigmay2 = 0.0
    for index in range(level_count):
        level_value = index + 1.0
        dx = level_value - mux
        dy = level_value - muy
        sigmax2 += dx * dx * px[index]
        sigmay2 += dy * dy * py[index]
    if sigmax2 <= 0.0 or sigmay2 <= 0.0:
        return 0.0
    sigmax = np.sqrt(sigmax2)
    sigmay = np.sqrt(sigmay2)
    summed = 0.0
    for y in range(level_count):
        for x in range(level_count):
            summed += (y + 1.0) * (x + 1.0) * matrix[y, x]
    value = (summed - mux * muy) / (sigmax * sigmay)
    if np.isfinite(value):
        return value
    return 0.0


@numpy(contract=ProcessingContract.PURE_2D)
def measure_image_quality(
    image: np.ndarray,
    include_scaling: bool = True,
    calculate_blur: bool = True,
    calculate_saturation: bool = True,
    calculate_intensity: bool = True,
    calculate_threshold: bool = True,
    object_fraction: float = 0.1,
    blur_scales: tuple[int, ...] = (20,),
    threshold_method: ImageQualityThresholdMethod = ImageQualityThresholdMethod.OTSU,
    otsu_class_count: CellProfilerOtsuMethod = CellProfilerOtsuMethod.TWO_CLASS,
    otsu_objective: ImageQualityOtsuObjective = (
        ImageQualityOtsuObjective.WEIGHTED_VARIANCE
    ),
    assign_middle_to_foreground: CellProfilerThresholdAssignment = (
        CellProfilerThresholdAssignment.FOREGROUND
    ),
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, ColumnarRows]:
    """Measure CellProfiler-compatible image-quality metrics."""
    total_started_at = time.perf_counter()
    records: list[ImageQualityMeasurementRecord] = []
    phase_started_at = time.perf_counter()
    intensity_scale = image_payload_intensity_scale(image)
    pixel_data = np.asarray(image, dtype=np.float32)
    runtime_profiler.log(
        "miq_prepare_image",
        time.perf_counter() - phase_started_at,
        function="measure_image_quality",
    )
    if include_scaling:
        records.append(
            ImageQualityScalingMetrics(
                slice_index=0,
                scaling=(
                    float(intensity_scale)
                    if intensity_scale is not None
                    else float("nan")
                ),
            )
        )
    if calculate_blur:
        blur_summary = ImageQualityBlurSummaryMetrics(slice_index=0)
        phase_started_at = time.perf_counter()
        blur_summary.focus_score = image_quality_focus_score(pixel_data)
        runtime_profiler.log(
            "miq_focus_score",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        for blur_scale in blur_scales:
            blur_metrics = ImageQualityBlurMetrics(
                slice_index=0,
                scale=str(int(blur_scale)),
            )
            phase_started_at = time.perf_counter()
            blur_metrics.local_focus_score = image_quality_local_focus_score(
                pixel_data, blur_scale
            )
            runtime_profiler.log(
                "miq_local_focus_score",
                time.perf_counter() - phase_started_at,
                function="measure_image_quality",
                scale=int(blur_scale),
            )
            phase_started_at = time.perf_counter()
            blur_metrics.correlation = image_quality_haralick_correlation(
                pixel_data, blur_scale, backend_provider=backend_provider
            )
            runtime_profiler.log(
                "miq_correlation",
                time.perf_counter() - phase_started_at,
                function="measure_image_quality",
                scale=int(blur_scale),
            )
            records.append(blur_metrics)
        phase_started_at = time.perf_counter()
        blur_summary.power_log_log_slope = image_quality_power_spectrum_slope(
            pixel_data, backend_provider=backend_provider
        )
        runtime_profiler.log(
            "miq_power_log_log_slope",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        records.insert(0, blur_summary)
    if calculate_saturation:
        saturation_metrics = ImageQualitySaturationMetrics(slice_index=0)
        phase_started_at = time.perf_counter()
        (
            saturation_metrics.percent_maximal,
            saturation_metrics.percent_minimal,
        ) = image_quality_saturation(pixel_data)
        runtime_profiler.log(
            "miq_saturation",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        records.append(saturation_metrics)
    if calculate_intensity:
        phase_started_at = time.perf_counter()
        intensity_metrics = image_quality_intensity_metrics(pixel_data)
        runtime_profiler.log(
            "miq_intensity",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
        )
        records.append(intensity_metrics)
    if calculate_threshold:
        phase_started_at = time.perf_counter()
        threshold_method = coerce_cellprofiler_enum(
            ImageQualityThresholdMethod, threshold_method
        )
        otsu_class_count = coerce_cellprofiler_enum(
            CellProfilerOtsuMethod, otsu_class_count
        )
        otsu_objective = coerce_cellprofiler_enum(
            ImageQualityOtsuObjective, otsu_objective
        )
        assign_middle_to_foreground = coerce_cellprofiler_enum(
            CellProfilerThresholdAssignment, assign_middle_to_foreground
        )
        threshold_metrics = ImageQualityThresholdMetrics(
            slice_index=0,
            feature_name=threshold_method.feature_field_name,
            scale=threshold_method.descriptor_scale(
                otsu_class_count=otsu_class_count,
                otsu_objective=otsu_objective,
                assign_middle_to_foreground=assign_middle_to_foreground,
            ),
        )
        threshold_metrics.result_value = image_quality_threshold(
            pixel_data,
            threshold_method,
            object_fraction=object_fraction,
            otsu_class_count=otsu_class_count,
            otsu_objective=otsu_objective,
            assign_middle_to_foreground=assign_middle_to_foreground,
        )
        runtime_profiler.log(
            "miq_threshold",
            time.perf_counter() - phase_started_at,
            function="measure_image_quality",
            method=threshold_method.value,
        )
        records.append(threshold_metrics)
    runtime_profiler.log(
        "miq_total",
        time.perf_counter() - total_started_at,
        function="measure_image_quality",
    )
    return (
        image,
        ConcatenatedColumnarRows(
            tuple(
                DataclassMeasurementColumnarRows((record,), row_type=type(record))
                for record in records
            )
        ),
    )


def _prepare_measure_image_quality() -> None:
    sample = (
        (np.arange(64 * 64, dtype=np.uint16) % 256).astype(np.float32).reshape((64, 64))
    )
    measure_image_quality.__wrapped__(sample)


measure_image_quality.__openhcs_prepare__ = _prepare_measure_image_quality
__all__ = [
    "CentrosomeNumpyImageQualityBackendStrategy",
    "ImageQualityBackendStrategy",
    "ImageQualityBlurSummaryMetrics",
    "ImageQualityBlurMetrics",
    "ImageQualityIntensityMetrics",
    "ImageQualityMeasurementRecord",
    "ImageQualityOtsuObjective",
    "ImageQualitySaturationMetrics",
    "ImageQualityScalingMetrics",
    "ImageQualityThresholdMethod",
    "ImageQualityThresholdMetrics",
    "ImageQualityThresholdStrategy",
    "NumbaNumpyImageQualityBackendStrategy",
    "NumpyImageQualityBackendStrategy",
    "image_quality_backend",
    "image_quality_focus_score",
    "image_quality_haralick_correlation",
    "image_quality_has_multiple_unique_values",
    "image_quality_intensity_metrics",
    "image_quality_local_focus_score",
    "image_quality_power_spectrum_slope",
    "image_quality_saturation",
    "image_quality_threshold",
    "measure_image_quality",
]
