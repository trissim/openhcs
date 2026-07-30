"""Intensity-distribution backends for CellProfiler-compatible processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import logging
import time
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta
from numba import njit
import numpy as np
import scipy.sparse

from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.measurement_row_materialization import ConcatenatedColumnarRows
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    runtime_bound_parameters,
    special_inputs,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import enum_member_with_payload
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_image_values import (
    ImagePayloadMetadata,
    image_mask_for_data_domain,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    project_image_mask_to_data_domain,
    with_image_payload_data,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelValue,
    ObjectLabelVariantData,
    object_label_dense_array,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    dense_object_label_id_domain,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimeSliceProjectableValue,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
    setting_name_matches,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
    parse_cellprofiler_bool,
    parse_cellprofiler_int,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
    ObjectMeasurementColumnarRows,
)
from openhcs.processing.backends.cellprofiler.secondary import (
    SecondaryPropagationBackendStrategy,
    secondary_propagation_backend,
)
from openhcs.processing.backends.cellprofiler.shape import (
    ShapeMeasurementBackendStrategy,
    shape_measurement_backend,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
    SourceQualifiedMeasurementFeatureModule,
    PerObjectMeasurementExecutionModule,
)
from openhcs.interop.cellprofiler.module_measurement_features import (
    IntensityFeature,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
    DenseColumnarObjectMeasurementRowsMixin,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    MissingObjectMeasurementValuePolicy,
    MissingObjectMeasurementValueRequest,
    MissingObjectMeasurementValueStrategy,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.processing.backends.cellprofiler.zernike import (
    IntensityZernikeFeatureAuthority,
    IntensityZernikeMeasurementRowsRequest,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)


class IntensityDistributionHeatmapMeasurement(Enum):
    """CellProfiler heatmap value families."""

    FRACTION_AT_DISTANCE = "Fraction at Distance"
    MEAN_FRACTION = "Mean Fraction"
    RADIAL_CV = "Radial CV"


@dataclass(frozen=True, slots=True)
class IntensityDistributionBinGroup:
    """One radial bin configuration selected by heatmap rows."""

    bin_count: int
    wants_scaled: bool
    maximum_radius: int


@dataclass(frozen=True, slots=True)
class IntensityDistributionHeatmapGroup:
    """One serialized heatmap group, including its conditional image output."""

    source_image_name: str
    object_name: str
    bin_count: int
    wants_scaled: bool
    maximum_radius: int
    measurement: IntensityDistributionHeatmapMeasurement
    colormap: str
    save_display: bool
    output_image_name: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("source_image_name", self.source_image_name),
            ("object_name", self.object_name),
            ("output_image_name", self.output_image_name),
        ):
            if not value.strip() or value != value.strip():
                raise ValueError(
                    f"IntensityDistributionHeatmapGroup.{field_name} must be a "
                    "normalized non-empty string."
                )
        if self.bin_count <= 0:
            raise ValueError(
                "Intensity-distribution heatmap bin_count must be positive."
            )


class IntensityDistributionSettingSchema(Enum):
    """Exact repeating-group count layout for one CellProfiler revision."""

    def __new__(
        cls,
        revision: int,
        bin_group_count_index: int,
        heatmap_group_count_index: int,
        records_image_group_count: bool,
    ) -> "IntensityDistributionSettingSchema":
        member = object.__new__(cls)
        member._value_ = revision
        member.bin_group_count_index = bin_group_count_index
        member.heatmap_group_count_index = heatmap_group_count_index
        member.records_image_group_count = records_image_group_count
        return member

    REVISION_5 = (5, 2, 3, True)
    REVISION_6 = (6, 1, 2, False)

    @classmethod
    def imported_module(
        cls, module: ModuleBlock
    ) -> "IntensityDistributionSettingSchema | None":
        """Return the exact serialized schema for an imported module."""
        revision = module.variable_revision_number
        return None if revision is None else cls(revision)

    def declared_bin_group_count(self, hidden_counts: tuple[str, ...]) -> int:
        return int(float(hidden_counts[self.bin_group_count_index]))

    def declared_heatmap_group_count(self, hidden_counts: tuple[str, ...]) -> int:
        return int(float(hidden_counts[self.heatmap_group_count_index]))

    def hidden_count_records(
        self,
        *,
        image_count: int,
        object_count: int,
        bin_count: int,
        heatmap_count: int,
    ) -> tuple[ModuleSetting, ...]:
        counts = (
            (image_count, object_count, bin_count, heatmap_count)
            if self.records_image_group_count
            else (object_count, bin_count, heatmap_count)
        )
        return tuple(ModuleSetting("Hidden", str(count)) for count in counts)


@dataclass(frozen=True)
class IntensityDistributionHeatmapSourceRelation(SourceStackLineageSourceRelation):
    """Declare the exact source and rendering semantics for one heatmap."""

    relation_key: ClassVar[str] = "intensity_distribution_heatmap_source"
    target_artifact_type = ImageArtifactType

    bin_count: int
    wants_scaled: bool
    maximum_radius: int
    measurement: IntensityDistributionHeatmapMeasurement
    colormap: str


class IntensityDistributionHeatmapObjectRelation(ArtifactSpecRelation):
    """Declare the exact object-label domain rendered by one heatmap."""

    relation_key: ClassVar[str] = "intensity_distribution_heatmap_object"
    target_artifact_type = ImageArtifactType

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "Intensity-distribution heatmap object relation requires an "
                f"object source, got {self.source.artifact_type.value}:"
                f"{self.source.name}."
            )


@dataclass(frozen=True, slots=True)
class IntensityDistributionHeatmapRuntimeOutput(RuntimeSliceProjectableValue):
    """Runtime heatmap request in compiled image-output order."""

    group: IntensityDistributionHeatmapGroup
    object_labels: ObjectLabelValue

    def project_runtime_slice(
        self, slice_index: int
    ) -> "IntensityDistributionHeatmapRuntimeOutput":
        """Project the heatmap object-label domain through its declared axis."""
        projection = self.object_labels.declared_plane_projection()
        if projection is None or projection.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return self
        projected = RuntimeSliceProjection.value_for_slice(
            self.object_labels,
            projection.selected_plane(slice_index),
        )
        return replace(self, object_labels=projected)


class _IntensityDistributionHeatmapOutputsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound heatmap requests reconstructed from output relations."""

    parameter_name = "heatmap_outputs"
    annotation_type = tuple[IntensityDistributionHeatmapRuntimeOutput, ...]
    parameter_default = ()


class IntensityDistributionCenterChoice(Enum):
    """Nominal CP center choices for radial intensity distribution."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "IntensityDistributionCenterChoice":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    SELF = ("self", "These objects")
    CENTERS_OF_OTHER = ("centers_of_other", "Centers of other objects")
    EDGES_OF_OTHER = ("edges_of_other", "Edges of other objects")


class IntensityDistributionZernikeMode(Enum):
    """Nominal CP Zernike output modes for intensity distribution."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "IntensityDistributionZernikeMode":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    NONE = ("none",)
    MAGNITUDES = ("magnitudes", "Magnitudes only")
    MAGNITUDES_AND_PHASE = ("magnitudes_and_phase", "Magnitudes and phase")


def parse_intensity_distribution_zernike_mode(value: str) -> str:
    """Return the absorbed-function Zernike mode literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionZernikeMode, value).value


def parse_intensity_distribution_center_choice(value: str) -> str:
    """Return the absorbed-function center-choice literal for a CP setting."""
    return coerce_cellprofiler_enum(IntensityDistributionCenterChoice, value).value


class MeasureObjectIntensityDistributionObjectMeasurementRowPolicy(
    DenseColumnarObjectMeasurementRowsMixin,
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
):
    """Intensity-distribution rows export CP's compact object row identity."""

    missing_value_policy = (
        MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
    )


class MeasureObjectIntensityDistributionModule(
    LabelsObjectInputPolicy,
    MeasureObjectIntensityDistributionObjectMeasurementRowPolicy,
    PerObjectMeasurementExecutionModule,
    ImageMeasurementInputModule,
    ObjectMeasurementInputModule,
    SourceQualifiedMeasurementFeatureModule,
    IntensityZernikeFeatureAuthority,
):
    module_name = "MeasureObjectIntensityDistribution"
    function_name = "measure_object_intensity_distribution"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (
        ("radial", "distribution"),
        ("intensity", "distribution"),
    )
    measurement_record_excluded_fields = frozenset(
        {MeasurementRowAxisField.OBJECT_ROW_IDENTITY.value}
    )

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by MeasureObjectIntensityDistribution."""

        FRACTION_AT_DISTANCE = ("FracAtD", (), (IntensityFeature,))
        MEAN_FRACTION = ("MeanFrac", (), (IntensityFeature,))
        RADIAL_CV = ("RadialCV", (), (IntensityFeature,))

        def source_qualified_name(
            self,
            *,
            source_image_name: str,
        ) -> str:
            return (
                MeasureObjectIntensityDistributionModule.source_qualified_feature_name(
                    self.measurement_row_field_name,
                    source_image_name,
                )
            )

    ignored_settings = ("Hidden",)
    zernike_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate intensity Zernikes?"
    )
    zernike_degree_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Maximum Zernike moment", aliases=("Maximum zernike moment",)
    )
    center_choice_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Object to use as center?"
    )
    wants_scaled_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Scale the bins?"
    )
    bin_count_setting: ClassVar[SettingNameFamily] = SettingNameFamily("Number of bins")
    maximum_radius_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Maximum radius"
    )
    center_objects_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select objects to use as centers"
    )
    heatmap_image_setting = "Image"
    heatmap_object_setting = "Objects to display"
    heatmap_measurement_setting = "Measurement"
    heatmap_colormap_setting = "Color map"
    heatmap_save_setting = "Save display as image?"
    heatmap_output_image_setting = "Output image name"
    heatmap_output_image_binding = SettingToKeywordBinding.output(
        heatmap_output_image_setting,
        ImageArtifactType,
        "heatmap_output_names",
        repeated=True,
    )
    setting_bindings = (
        heatmap_output_image_binding,
        SettingToKeywordBinding(
            zernike_setting,
            "wants_zernikes",
            parse_intensity_distribution_zernike_mode,
        ),
        SettingToKeywordBinding(
            zernike_degree_setting,
            "zernike_degree",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            wants_scaled_setting,
            "wants_scaled",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            bin_count_setting,
            "bin_count",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            maximum_radius_setting,
            "maximum_radius",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            center_choice_setting,
            "center_choice",
            parse_intensity_distribution_center_choice,
        ),
        SettingToKeywordBinding(
            center_objects_setting,
            "select_objects_to_use_as_centers",
        ),
    )

    @classmethod
    def bin_groups(
        cls,
        module: ModuleBlock,
    ) -> tuple[IntensityDistributionBinGroup, ...]:
        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=cls.wants_scaled_setting,
        )
        if blocks:
            hidden_counts = setting_values(module, "Hidden")
            schema = IntensityDistributionSettingSchema.imported_module(module)
            if schema is not None and len(blocks) != (
                declared_count := schema.declared_bin_group_count(hidden_counts)
            ):
                raise ValueError(
                    "MeasureObjectIntensityDistribution bin groups do not match "
                    f"their declared count: {len(blocks)} != {declared_count}."
                )
            return tuple(
                IntensityDistributionBinGroup(
                    bin_count=int(
                        float(block_setting_value(block, cls.bin_count_setting))
                    ),
                    wants_scaled=parse_cellprofiler_bool(
                        block_setting_value(block, cls.wants_scaled_setting)
                    ),
                    maximum_radius=int(
                        float(
                            block_setting_value(
                                block,
                                cls.maximum_radius_setting,
                                default="100",
                            )
                        )
                    ),
                )
                for block in blocks
            )
        bin_values = setting_values(module, cls.bin_count_setting)
        if not bin_values:
            return ()
        scaled_values = setting_values(module, cls.wants_scaled_setting)
        radius_values = setting_values(module, cls.maximum_radius_setting)
        if len(bin_values) != 1 or len(scaled_values) > 1 or len(radius_values) > 1:
            raise ValueError(
                "MeasureObjectIntensityDistribution mapping-only settings cannot "
                "represent repeated bin groups."
            )
        return (
            IntensityDistributionBinGroup(
                bin_count=int(float(bin_values[0])),
                wants_scaled=(
                    parse_cellprofiler_bool(scaled_values[0]) if scaled_values else True
                ),
                maximum_radius=(int(float(radius_values[0])) if radius_values else 100),
            ),
        )

    @classmethod
    def heatmap_groups(
        cls,
        module: ModuleBlock,
    ) -> tuple[IntensityDistributionHeatmapGroup, ...]:
        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=cls.heatmap_image_setting,
        )
        hidden_counts = setting_values(module, "Hidden")
        schema = IntensityDistributionSettingSchema.imported_module(module)
        if schema is not None and len(blocks) != (
            declared_count := schema.declared_heatmap_group_count(hidden_counts)
        ):
            raise ValueError(
                "MeasureObjectIntensityDistribution heatmap groups do not match "
                f"their declared count: {len(blocks)} != {declared_count}."
            )
        bin_groups_by_count: dict[int, IntensityDistributionBinGroup] = {}
        for bin_group in cls.bin_groups(module):
            if bin_group.bin_count in bin_groups_by_count:
                raise ValueError(
                    "MeasureObjectIntensityDistribution heatmap selection is "
                    "ambiguous for duplicate bin count "
                    f"{bin_group.bin_count}."
                )
            bin_groups_by_count[bin_group.bin_count] = bin_group
        groups: list[IntensityDistributionHeatmapGroup] = []
        for group_index, block in enumerate(blocks):
            source_name = normalized_symbol_name(
                block_setting_value(block, cls.heatmap_image_setting)
            )
            object_name = normalized_symbol_name(
                block_setting_value(block, cls.heatmap_object_setting)
            )
            output_name = normalized_symbol_name(
                block_setting_value(block, cls.heatmap_output_image_setting)
            )
            if source_name is None or object_name is None or output_name is None:
                raise ValueError(
                    "MeasureObjectIntensityDistribution heatmap group "
                    f"{group_index + 1} requires exact image, object, and output "
                    "names."
                )
            bin_count = int(float(block_setting_value(block, cls.bin_count_setting)))
            try:
                bin_group = bin_groups_by_count[bin_count]
            except KeyError as exc:
                raise ValueError(
                    "MeasureObjectIntensityDistribution heatmap group "
                    f"{group_index + 1} selects undeclared bin count {bin_count}."
                ) from exc
            measurement = coerce_cellprofiler_enum(
                IntensityDistributionHeatmapMeasurement,
                block_setting_value(block, cls.heatmap_measurement_setting),
            )
            groups.append(
                IntensityDistributionHeatmapGroup(
                    source_image_name=source_name,
                    object_name=object_name,
                    bin_count=bin_group.bin_count,
                    wants_scaled=bin_group.wants_scaled,
                    maximum_radius=bin_group.maximum_radius,
                    measurement=measurement,
                    colormap=block_setting_value(
                        block,
                        cls.heatmap_colormap_setting,
                        default="Default",
                    ),
                    save_display=parse_cellprofiler_bool(
                        block_setting_value(
                            block,
                            cls.heatmap_save_setting,
                            default="No",
                        )
                    ),
                    output_image_name=output_name,
                )
            )
        return tuple(groups)

    @classmethod
    def saved_heatmap_groups(
        cls,
        module: ModuleBlock,
    ) -> tuple[IntensityDistributionHeatmapGroup, ...]:
        return tuple(
            group for group in cls.heatmap_groups(module) if group.save_display
        )

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        return tuple(
            binding
            for binding in bindings
            if cls.saved_heatmap_groups(module)
            or binding is not cls.heatmap_output_image_binding
        )

    @classmethod
    def artifact_names_for_binding(cls, module, binding):
        if binding is cls.heatmap_output_image_binding:
            return tuple(
                group.output_image_name for group in cls.saved_heatmap_groups(module)
            )
        return super().artifact_names_for_binding(module, binding)

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
        del invocation_key, step_context, binding
        groups = cls.saved_heatmap_groups(module)
        if (
            output_position >= len(groups)
            or groups[output_position].output_image_name != name
        ):
            raise ValueError(
                "MeasureObjectIntensityDistribution output order does not match "
                f"its saved heatmaps: position={output_position}, name={name!r}, "
                f"groups={groups!r}."
            )
        group = groups[output_position]
        source = artifact_inputs.require_by_name_and_artifact_type(
            group.source_image_name,
            ImageArtifactType,
        )
        objects = artifact_inputs.require_by_name_and_artifact_type(
            group.object_name,
            ObjectLabelsArtifactType,
        )
        return (
            IntensityDistributionHeatmapSourceRelation(
                source=source.ref(),
                bin_count=group.bin_count,
                wants_scaled=group.wants_scaled,
                maximum_radius=group.maximum_radius,
                measurement=group.measurement,
                colormap=group.colormap,
            ),
            IntensityDistributionHeatmapObjectRelation(source=objects.ref()),
        )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: ModuleBlock,
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        kwargs = dict(bound.kwargs)
        kwargs.pop(cls.heatmap_output_image_binding.require_parameter_name(), None)
        heatmap_groups = cls.heatmap_groups(module)
        if heatmap_groups:
            kwargs["heatmap_groups"] = heatmap_groups
        return bound.with_replaced_kwargs(kwargs).with_consumed_settings(
            cls.heatmap_image_setting,
            cls.heatmap_object_setting,
            cls.heatmap_measurement_setting,
            cls.heatmap_colormap_setting,
            cls.heatmap_save_setting,
            cls.heatmap_output_image_setting,
        )

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation,
        step_context,
    ) -> tuple[ModuleBlock, ...]:
        """Reconstruct all heatmap rows from the public callable declaration."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        raw_groups = invocation.kwargs_dict.get("heatmap_groups", ())
        if not isinstance(raw_groups, (tuple, list)) or any(
            not isinstance(group, IntensityDistributionHeatmapGroup)
            for group in raw_groups
        ):
            raise TypeError(
                "MeasureObjectIntensityDistribution heatmap_groups must contain "
                "only IntensityDistributionHeatmapGroup values."
            )
        groups = tuple(raw_groups)
        return tuple(cls._block_with_heatmap_groups(block, groups) for block in blocks)

    @classmethod
    def _block_with_heatmap_groups(
        cls,
        block: ModuleBlock,
        groups: tuple[IntensityDistributionHeatmapGroup, ...],
    ) -> ModuleBlock:
        records: list[ModuleSetting] = []
        heatmap_rows_started = False
        for record in block.iter_settings():
            if setting_name_matches(record.name, cls.heatmap_image_setting):
                heatmap_rows_started = True
            if heatmap_rows_started or setting_name_matches(record.name, "Hidden"):
                continue
            records.append(record)
        image_count = len(
            cls.artifact_names_for_binding(
                block,
                cls.image_measurement_binding,
            )
        )
        object_count = len(
            cls.artifact_names_for_binding(
                block,
                cls.object_measurement_binding,
            )
        )
        bin_count = len(cls.bin_groups(block))
        if image_count <= 0 or object_count <= 0 or bin_count <= 0:
            raise ValueError(
                "MeasureObjectIntensityDistribution public reconstruction requires "
                "at least one declared image, object, and bin group."
            )
        records[1:1] = (
            IntensityDistributionSettingSchema.REVISION_6.hidden_count_records(
                image_count=image_count,
                object_count=object_count,
                bin_count=bin_count,
                heatmap_count=len(groups),
            )
        )
        for group in groups:
            records.extend(
                (
                    ModuleSetting(cls.heatmap_image_setting, group.source_image_name),
                    ModuleSetting(cls.heatmap_object_setting, group.object_name),
                    ModuleSetting(
                        cls.bin_count_setting.canonical, str(group.bin_count)
                    ),
                    ModuleSetting(
                        cls.heatmap_measurement_setting,
                        group.measurement.value,
                    ),
                    ModuleSetting(cls.heatmap_colormap_setting, group.colormap),
                    ModuleSetting(
                        cls.heatmap_save_setting,
                        "Yes" if group.save_display else "No",
                    ),
                    ModuleSetting(
                        cls.heatmap_output_image_setting,
                        group.output_image_name,
                    ),
                )
            )
        return replace(
            block,
            setting_records=records,
        )

    @classmethod
    def bind_runtime_inputs(cls, request: RuntimeInputBindingRequest):
        return {
            **super().bind_runtime_inputs(request),
            _IntensityDistributionHeatmapOutputsRuntimeParameter.require_parameter_name(): cls._runtime_heatmap_outputs(
                request
            ),
        }

    @classmethod
    def _runtime_heatmap_outputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> tuple[IntensityDistributionHeatmapRuntimeOutput, ...]:
        inputs = request.declared_inputs
        image_outputs = request.adapter.request.require_callable_contract().artifact_outputs.of_artifact_type(
            ImageArtifactType
        )
        outputs: list[IntensityDistributionHeatmapRuntimeOutput] = []
        for output in image_outputs:
            source_relations = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, IntensityDistributionHeatmapSourceRelation)
            )
            object_relations = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, IntensityDistributionHeatmapObjectRelation)
            )
            if len(source_relations) != 1 or len(object_relations) != 1:
                raise ValueError(
                    "MeasureObjectIntensityDistribution heatmap output requires "
                    "one exact source and object relation, got "
                    f"{source_relations!r} and {object_relations!r}."
                )
            source_relation = source_relations[0]
            source_spec = inputs.by_ref(source_relation.source)
            object_spec = inputs.by_ref(object_relations[0].source)
            if source_spec is None or object_spec is None:
                raise ValueError(
                    "MeasureObjectIntensityDistribution heatmap relation refers "
                    "to an input absent from the compiled contract."
                )
            outputs.append(
                IntensityDistributionHeatmapRuntimeOutput(
                    group=IntensityDistributionHeatmapGroup(
                        source_image_name=source_spec.name,
                        object_name=object_spec.name,
                        bin_count=source_relation.bin_count,
                        wants_scaled=source_relation.wants_scaled,
                        maximum_radius=source_relation.maximum_radius,
                        measurement=source_relation.measurement,
                        colormap=source_relation.colormap,
                        save_display=True,
                        output_image_name=output.name,
                    ),
                    object_labels=request.label_payload_for(object_spec),
                )
            )
        return tuple(outputs)


CenterChoice = IntensityDistributionCenterChoice
ZernikeMode = IntensityDistributionZernikeMode

logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)
_RADIAL_LABEL_GEOMETRY_CACHE_LIMIT = 16
_RADIAL_LABEL_GEOMETRY_CACHE: OrderedDict[
    "RadialLabelGeometryCacheKey", "RadialLabelGeometry"
] = OrderedDict()


@dataclass(frozen=True)
class IntensityDistributionProfiler:
    """Bound profiler for object intensity-distribution measurement phases."""

    function_name: str

    def record(self, label: str, started_at: float, **fields: object) -> None:
        runtime_profiler.log(
            label,
            time.perf_counter() - started_at,
            function=self.function_name,
            **fields,
        )

    def record_rows(self, label: str, started_at: float, row_count: int) -> None:
        self.record(label, started_at, rows=row_count)


@dataclass(frozen=True, slots=True)
class RadialDistributionArrays:
    """Dense per-object radial intensity-distribution arrays."""

    fraction_at_distance: np.ndarray
    mean_pixel_fraction: np.ndarray
    radial_cv_by_bin: np.ndarray
    object_has_pixels: np.ndarray
    n_bins: int

    @classmethod
    def empty(cls, *, bin_count: int, wants_scaled: bool) -> "RadialDistributionArrays":
        n_bins = int(bin_count) if wants_scaled else int(bin_count) + 1
        return cls(
            fraction_at_distance=np.zeros((0, int(bin_count) + 1), dtype=float),
            mean_pixel_fraction=np.zeros((0, int(bin_count) + 1), dtype=float),
            radial_cv_by_bin=np.zeros((n_bins, 0), dtype=float),
            object_has_pixels=np.zeros(0, dtype=bool),
            n_bins=n_bins,
        )

    @classmethod
    def from_components(
        cls,
        *,
        fraction_at_distance: np.ndarray,
        mean_pixel_fraction: np.ndarray,
        radial_cv_by_bin: np.ndarray,
        object_has_pixels: np.ndarray,
        n_bins: int,
    ) -> "RadialDistributionArrays":
        return cls(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )


@dataclass(frozen=True, slots=True)
class RadialCenterDistanceFields:
    """CellProfiler center-propagation fields for radial measurements."""

    d_from_center: np.ndarray
    center_labels: np.ndarray
    centers_i: np.ndarray
    centers_j: np.ndarray


@dataclass(frozen=True, slots=True)
class RadialCenterPropagationRequest:
    """Nearest-center propagation for radial intensity-distribution geometry."""

    center_labels: np.ndarray
    colors: np.ndarray
    propagation_backend: SecondaryPropagationBackendStrategy

    def fields(self) -> tuple[np.ndarray, np.ndarray]:
        """Return center distances and propagated center labels by color mask."""
        d_from_center = np.zeros(self.center_labels.shape, dtype=float)
        propagated_center_labels = np.zeros(self.center_labels.shape, dtype=int)
        max_color = int(np.max(self.colors)) if self.colors.size else 0
        seed_labels = np.asarray(self.center_labels, dtype=np.int32)
        for color in range(1, max_color + 1):
            mask = self.colors == color
            seed_mask = mask & (seed_labels > 0)
            if not np.any(seed_mask):
                continue
            propagation = self.propagation_backend.propagate_zero_image_result(
                seed_labels, mask, 1
            )
            propagated_labels = propagation.labels
            distances = propagation.distances
            d_from_center[mask] = distances[mask]
            propagated_center_labels[mask] = propagated_labels[mask]
        return (d_from_center, propagated_center_labels)


@dataclass(frozen=True, slots=True)
class RadialLabelGeometryCacheKey:
    """Content identity for radial geometry derived only from object labels."""

    dtype: str
    shape: tuple[int, ...]
    digest: bytes

    @classmethod
    def from_labels(cls, labels: np.ndarray) -> "RadialLabelGeometryCacheKey":
        label_array = np.ascontiguousarray(labels, dtype=np.int32)
        digest = hashlib.sha1(label_array.view(np.uint8)).digest()
        return cls(
            dtype=str(label_array.dtype),
            shape=tuple((int(value) for value in label_array.shape)),
            digest=digest,
        )


@dataclass(frozen=True, slots=True)
class RadialLabelGeometry:
    """Label-only radial geometry shared by all intensity images for a label plane."""

    d_to_edge: np.ndarray
    center_fields: RadialCenterDistanceFields


@dataclass(frozen=True, slots=True)
class IntensityDistributionSliceInput:
    """One aligned 2D image/label slice."""

    image: np.ndarray
    image_mask: np.ndarray | None
    labels: np.ndarray
    slice_index: int
    object_domain: tuple[int, ...]

    @classmethod
    def from_aligned_arrays(
        cls,
        *,
        image: np.ndarray,
        image_mask: np.ndarray | None = None,
        labels: ObjectLabelValue,
        slice_index: int,
        object_domain: tuple[int, ...] | None = None,
    ) -> "IntensityDistributionSliceInput":
        image_array = np.asarray(image)
        label_array = np.asarray(object_label_dense_array(labels, dtype=np.int32))
        if image_array.ndim != 2 or label_array.ndim != 2:
            raise ValueError(
                "MeasureObjectIntensityDistribution requires image and labels "
                "already projected to one 2-D plane; got image "
                f"{image_array.shape!r} and labels {label_array.shape!r}."
            )
        if image_array.shape != label_array.shape:
            raise ValueError(
                "MeasureObjectIntensityDistribution image and projected labels "
                f"must share a shape; got image {image_array.shape!r} and labels "
                f"{label_array.shape!r}."
            )
        return cls(
            image=image_array,
            image_mask=image_mask,
            labels=label_array,
            slice_index=slice_index,
            object_domain=(
                intensity_distribution_object_domain(labels)
                if object_domain is None
                else object_domain
            ),
        )

    @property
    def radial_labels(self) -> np.ndarray:
        if self.image_mask is None:
            return self.labels
        mask = project_image_mask_to_data_domain(self.image_mask, self.labels)
        if mask is None:
            raise ValueError(
                "MeasureObjectIntensityDistribution image mask cannot be "
                f"projected into label domain; got mask {np.shape(self.image_mask)!r} "
                f"for labels {self.labels.shape!r}."
            )
        return np.where(np.asarray(mask, dtype=bool), self.labels, 0).astype(
            np.int32,
            copy=False,
        )


@dataclass(frozen=True, slots=True)
class RadialDistributionMeasureRequest:
    """Complete per-plane radial-distribution measurement request."""

    image: np.ndarray
    labels: np.ndarray
    d_to_edge: np.ndarray
    d_from_center: np.ndarray
    center_labels: np.ndarray
    centers_i: np.ndarray
    centers_j: np.ndarray
    bin_count: int
    wants_scaled: bool
    maximum_radius: int

    def arrays(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Return validated dense arrays for radial backend execution."""
        image_array = np.ascontiguousarray(self.image)
        labels_array = np.ascontiguousarray(self.labels, dtype=np.int32)
        d_to_edge_array = np.ascontiguousarray(self.d_to_edge, dtype=np.float64)
        d_from_center_array = np.ascontiguousarray(self.d_from_center, dtype=np.float64)
        center_labels_array = np.ascontiguousarray(self.center_labels, dtype=np.int32)
        centers_i_array = np.ascontiguousarray(self.centers_i, dtype=np.float64)
        centers_j_array = np.ascontiguousarray(self.centers_j, dtype=np.float64)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                f"CellProfiler radial intensity distribution currently supports 2-D NumPy planes, got image {image_array.shape!r} and labels {labels_array.shape!r}."
            )
        if labels_array.shape != image_array.shape:
            raise ValueError(
                f"Radial distribution labels must match the image shape; got labels {labels_array.shape!r} for image {image_array.shape!r}."
            )
        if self.bin_count <= 0:
            raise ValueError(f"bin_count must be positive, got {self.bin_count!r}.")
        return (
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
        )


@dataclass(frozen=True, slots=True)
class RadialDistributionGeometryIndex:
    """Per-label radial bin/wedge index reusable across intensity images."""

    pixel_rows: np.ndarray
    pixel_cols: np.ndarray
    object_indices: np.ndarray
    bin_indices: np.ndarray
    radial_indices: np.ndarray
    number_at_distance: np.ndarray
    radial_counts: np.ndarray
    object_count: int
    bin_count: int
    n_bins: int


@dataclass(frozen=True, slots=True)
class IntensityDistributionMeasurementRequest:
    """Executable intensity-distribution request for one aligned slice."""

    slice_input: IntensityDistributionSliceInput
    source_image_name: str
    bin_count: int
    wants_scaled: bool
    maximum_radius: int
    wants_zernikes: ZernikeMode
    zernike_degree: int
    radial_backend: "RadialDistributionBackendStrategy"
    zernike_backend_provider: BackendProviderInput
    profiler: IntensityDistributionProfiler

    def rows(self) -> ColumnarRows:
        labels_2d = self.slice_input.radial_labels
        object_ids = self.slice_input.object_domain
        if not object_ids:
            return ObjectIntensityDistributionMeasurementColumnarRows.empty(
                source_image_name=self.source_image_name,
                slice_index=self.slice_input.slice_index,
            )
        phase_started_at = time.perf_counter()
        radial_arrays = self.radial_backend.measure_self_centered(
            self.slice_input.image,
            labels_2d,
            bin_count=self.bin_count,
            wants_scaled=self.wants_scaled,
            maximum_radius=self.maximum_radius,
        )
        self.profiler.record(
            "idist_radial_backend",
            phase_started_at,
            nobjects=len(object_ids),
            bins=radial_arrays.n_bins,
        )
        phase_started_at = time.perf_counter()
        measurements = ObjectIntensityDistributionMeasurementColumnarRows(
            radial_arrays=radial_arrays,
            object_ids=object_ids,
            source_image_name=self.source_image_name,
            bin_count=self.bin_count,
            slice_index=self.slice_input.slice_index,
        )
        self.profiler.record_rows(
            "idist_radial_rows", phase_started_at, len(measurements)
        )
        if self.wants_zernikes != ZernikeMode.NONE:
            phase_started_at = time.perf_counter()
            zernike_measurements = IntensityZernikeMeasurementRowsRequest(
                image=self.slice_input.image,
                labels=self.slice_input.labels,
                image_mask=self.slice_input.image_mask,
                max_order=self.zernike_degree,
                object_ids=object_ids,
                source_image_name=self.source_image_name,
                include_phase=self.wants_zernikes == ZernikeMode.MAGNITUDES_AND_PHASE,
                slice_index=self.slice_input.slice_index,
                row_identity=MeasurementObjectRowIdentity.LABEL_ID,
                backend_provider=self.zernike_backend_provider,
            ).rows()
            measurements = ConcatenatedColumnarRows(
                (measurements, zernike_measurements)
            )
            self.profiler.record_rows(
                "idist_zernike_rows", phase_started_at, len(measurements)
            )
        return measurements


@dataclass(slots=True)
class ObjectIntensityDistributionMeasurementColumnarRows(ObjectMeasurementColumnarRows):
    """Columnar radial intensity-distribution rows."""

    object_row_identity = MeasurementObjectRowIdentity.LABEL_ID

    radial_arrays: RadialDistributionArrays
    object_ids: tuple[int, ...]
    source_image_name: str
    bin_count: int
    slice_index: int | None = None
    _columns: Mapping[str, np.ndarray] = field(init=False, repr=False, compare=False)
    _fields: tuple[FieldSpec, ...] = field(init=False, repr=False, compare=False)

    @classmethod
    def empty(
        cls,
        *,
        source_image_name: str,
        slice_index: int | None = None,
    ) -> "ObjectIntensityDistributionMeasurementColumnarRows":
        return cls(
            radial_arrays=RadialDistributionArrays.empty(
                bin_count=0, wants_scaled=True
            ),
            object_ids=(),
            source_image_name=source_image_name,
            bin_count=0,
            slice_index=slice_index,
        )

    def __post_init__(self) -> None:
        object_ids = np.asarray(
            tuple((int(object_id) for object_id in self.object_ids))
        )
        row_count = int(object_ids.size) * int(self.radial_arrays.n_bins) * 3
        object_labels = np.empty(row_count, dtype=np.int32)
        feature_names = np.empty(row_count, dtype=object)
        source_image_names = np.full(row_count, self.source_image_name, dtype=object)
        bin_indices = np.empty(row_count, dtype=np.int32)
        bin_counts = np.full(row_count, int(self.bin_count), dtype=np.int32)
        result_values = np.empty(row_count, dtype=np.float64)
        object_has_pixels_by_index = self.radial_arrays.object_has_pixels
        fraction_at_distance = self.radial_arrays.fraction_at_distance
        mean_pixel_fraction = self.radial_arrays.mean_pixel_fraction
        radial_cv_by_bin = self.radial_arrays.radial_cv_by_bin
        radial_cv_missing_values = RadialCVMissingValueAuthority.values(
            object_ids,
            object_has_pixels_by_index,
        )
        row_index = 0
        for bin_idx in range(self.radial_arrays.n_bins):
            bin_index = bin_idx + 1
            fraction_at_distance_feature = MeasureObjectIntensityDistributionModule.MeasurementFeature.FRACTION_AT_DISTANCE.source_qualified_name(
                source_image_name=self.source_image_name,
            )
            mean_fraction_feature = MeasureObjectIntensityDistributionModule.MeasurementFeature.MEAN_FRACTION.source_qualified_name(
                source_image_name=self.source_image_name,
            )
            radial_cv_feature = MeasureObjectIntensityDistributionModule.MeasurementFeature.RADIAL_CV.source_qualified_name(
                source_image_name=self.source_image_name,
            )
            radial_cv = radial_cv_by_bin[bin_idx]
            for object_label in object_ids:
                object_row = DeclaredRadialDistributionObjectRow(
                    int(object_label), object_has_pixels_by_index.size
                )
                obj_idx = object_row.array_index
                object_has_pixels = obj_idx is not None and bool(
                    object_has_pixels_by_index[obj_idx]
                )
                object_labels[row_index : row_index + 3] = int(object_label)
                feature_names[row_index] = fraction_at_distance_feature
                feature_names[row_index + 1] = mean_fraction_feature
                feature_names[row_index + 2] = radial_cv_feature
                bin_indices[row_index : row_index + 3] = bin_index
                result_values[row_index] = (
                    float(fraction_at_distance[obj_idx, bin_idx])
                    if object_has_pixels and obj_idx is not None
                    else np.nan
                )
                result_values[row_index + 1] = (
                    float(mean_pixel_fraction[obj_idx, bin_idx])
                    if object_has_pixels and obj_idx is not None
                    else np.nan
                )
                result_values[row_index + 2] = (
                    RadialCVMissingValueAuthority.export_value(radial_cv[obj_idx])
                    if object_has_pixels and obj_idx is not None
                    else radial_cv_missing_values[int(object_label)]
                )
                row_index += 3
        field_columns: tuple[tuple[FieldSpec, np.ndarray], ...] = (
            (FieldSpec(MeasurementRowAxisField.OBJECT_LABEL.value, int), object_labels),
            (FieldSpec(MeasurementRowAxisField.FEATURE_NAME.value, str), feature_names),
            (
                FieldSpec(MeasurementRowAxisField.SOURCE_IMAGE_NAME.value, str),
                source_image_names,
            ),
            (FieldSpec(MeasurementRowAxisField.BIN_INDEX.value, int), bin_indices),
            (FieldSpec(MeasurementRowAxisField.BIN_COUNT.value, int), bin_counts),
            (
                FieldSpec(MeasurementRowValueField.RESULT_VALUE.value, float),
                result_values,
            ),
        )
        if self.slice_index is not None:
            field_columns = (
                *field_columns,
                (
                    FieldSpec(MeasurementRowAxisField.SLICE_INDEX.value, int),
                    np.full(row_count, int(self.slice_index), dtype=np.int32),
                ),
            )
        self._fields = tuple(field_spec for field_spec, _values in field_columns)
        self._columns = MappingProxyType(
            {field_spec.name: values for field_spec, values in field_columns}
        )
        self.validate_fields()

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        return self._columns

    @property
    def fields(self) -> tuple[FieldSpec, ...]:
        return self._fields


@dataclass(frozen=True, slots=True)
class DeclaredRadialDistributionObjectRow:
    """Array row for a declared object ID, including absent measured objects."""

    object_label: int
    measured_object_count: int

    @property
    def array_index(self) -> int | None:
        index = int(self.object_label) - 1
        if index < 0 or index >= int(self.measured_object_count):
            return None
        return index


class RadialCVMissingValueAuthority:
    """CellProfiler missing-row values for RadialCV over a dense object domain."""

    @staticmethod
    def export_value(value: float) -> float:
        """Normalize undefined radial coefficients to CellProfiler's export value."""
        raw_value = float(value)
        if not np.isfinite(raw_value):
            return 0.0
        return raw_value

    @classmethod
    def values(
        cls,
        object_ids: np.ndarray,
        object_has_pixels_by_index: np.ndarray,
    ) -> Mapping[int, float]:
        measured_positive_extent = cls.measured_positive_extent(
            object_ids,
            object_has_pixels_by_index,
        )
        strategy = MissingObjectMeasurementValueStrategy.for_enum_member(
            MeasureObjectIntensityDistributionModule.missing_value_policy
        )
        return {
            int(object_id): strategy.missing_value(
                MissingObjectMeasurementValueRequest(
                    object_id=int(object_id),
                    label_payload=(),
                    field_name=(
                        MeasureObjectIntensityDistributionModule.MeasurementFeature.RADIAL_CV.value
                    ),
                    positive_label_extent=measured_positive_extent,
                )
            )
            for object_id in object_ids
        }

    @staticmethod
    def measured_positive_extent(
        object_ids: np.ndarray,
        object_has_pixels_by_index: np.ndarray,
    ) -> int:
        measured_ids: list[int] = []
        for object_id in object_ids:
            object_row = DeclaredRadialDistributionObjectRow(
                int(object_id),
                object_has_pixels_by_index.size,
            )
            obj_idx = object_row.array_index
            if obj_idx is not None and bool(object_has_pixels_by_index[obj_idx]):
                measured_ids.append(int(object_id))
        if not measured_ids:
            return 0
        return max(measured_ids)


class RadialDistributionBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Radial-distribution operations keyed by OpenHCS memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True
    center_propagation_backend_provider = CellProfilerBackendProvider.NUMBA
    shape_geometry_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )

    def center_propagation_backend(self) -> SecondaryPropagationBackendStrategy:
        """Return the propagation backend used for center-distance geometry."""
        return secondary_propagation_backend(
            backend_provider=self.center_propagation_backend_provider
        )

    def shape_geometry_backend(self) -> ShapeMeasurementBackendStrategy:
        """Return the shape backend used for label-only radial geometry."""
        return shape_measurement_backend(
            backend_provider=self.shape_geometry_backend_provider
        )

    @abstractmethod
    def measure(
        self, request: RadialDistributionMeasureRequest
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays for a normalized request."""

    def measure_from_centers(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        d_to_edge: np.ndarray,
        centers_i: np.ndarray,
        centers_j: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays while computing center distances."""
        labels_array = np.asarray(labels, dtype=np.int32)
        d_to_edge_array = np.asarray(d_to_edge, dtype=np.float64)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=bin_count, wants_scaled=wants_scaled
            )
        center_fields = self.center_distance_fields(labels_array, centers_i, centers_j)
        return self.measure(
            RadialDistributionMeasureRequest(
                image=image,
                labels=labels_array,
                d_to_edge=d_to_edge_array,
                d_from_center=center_fields.d_from_center,
                center_labels=center_fields.center_labels,
                centers_i=center_fields.centers_i,
                centers_j=center_fields.centers_j,
                bin_count=bin_count,
                wants_scaled=wants_scaled,
                maximum_radius=maximum_radius,
            )
        )

    def measure_self_centered(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays using each object's own center."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=bin_count, wants_scaled=wants_scaled
            )
        geometry = self.label_geometry(labels_array)
        return self.measure_self_centered_with_geometry(
            image,
            labels_array,
            geometry,
            bin_count=bin_count,
            wants_scaled=wants_scaled,
            maximum_radius=maximum_radius,
        )

    def measure_self_centered_with_geometry(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        geometry: RadialLabelGeometry,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> RadialDistributionArrays:
        """Return radial-distribution arrays using precomputed label geometry."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=bin_count, wants_scaled=wants_scaled
            )
        return self.measure(
            RadialDistributionMeasureRequest(
                image=image,
                labels=labels_array,
                d_to_edge=geometry.d_to_edge,
                d_from_center=geometry.center_fields.d_from_center,
                center_labels=geometry.center_fields.center_labels,
                centers_i=geometry.center_fields.centers_i,
                centers_j=geometry.center_fields.centers_j,
                bin_count=bin_count,
                wants_scaled=wants_scaled,
                maximum_radius=maximum_radius,
            )
        )

    def measure_batch_self_centered_with_geometry(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        geometry: RadialLabelGeometry,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> tuple[RadialDistributionArrays, ...]:
        """Return radial arrays for same-label images using shared geometry."""
        return tuple(
            (
                self.measure_self_centered_with_geometry(
                    image,
                    labels,
                    geometry,
                    bin_count=bin_count,
                    wants_scaled=wants_scaled,
                    maximum_radius=maximum_radius,
                )
                for image in images
            )
        )

    def center_distance_fields(
        self, labels: np.ndarray, centers_i: np.ndarray, centers_j: np.ndarray
    ) -> RadialCenterDistanceFields:
        """Return CP-compatible propagated center labels and distances."""
        labels_array = np.asarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        object_labels = np.arange(1, object_count + 1, dtype=np.int32)
        center_labels = np.zeros(labels_array.shape, dtype=int)
        centers_i_int = np.asarray(centers_i, dtype=int)
        centers_j_int = np.asarray(centers_j, dtype=int)
        valid_center_bounds = (
            (centers_i_int >= 0)
            & (centers_i_int < labels_array.shape[0])
            & (centers_j_int >= 0)
            & (centers_j_int < labels_array.shape[1])
        )
        sampled_center_labels = np.zeros(object_count, dtype=np.int32)
        sampled_center_labels[valid_center_bounds] = labels_array[
            centers_i_int[valid_center_bounds], centers_j_int[valid_center_bounds]
        ]
        valid_centers = valid_center_bounds & (sampled_center_labels == object_labels)
        if np.any(valid_centers):
            center_labels[
                centers_i_int[valid_centers], centers_j_int[valid_centers]
            ] = labels_array[centers_i_int[valid_centers], centers_j_int[valid_centers]]
        shape_backend = self.shape_geometry_backend()
        phase_started_at = time.perf_counter()
        colors = shape_backend.color_labels(labels_array)
        runtime_profiler.log(
            "idist_center_color_labels",
            time.perf_counter() - phase_started_at,
            objects=object_count,
            colors=int(np.max(colors)) if colors.size else 0,
        )
        phase_started_at = time.perf_counter()
        d_from_center, propagated_center_labels = RadialCenterPropagationRequest(
            center_labels=center_labels,
            colors=colors,
            propagation_backend=self.center_propagation_backend(),
        ).fields()
        runtime_profiler.log(
            "idist_center_propagate",
            time.perf_counter() - phase_started_at,
            objects=object_count,
            colors=int(np.max(colors)) if colors.size else 0,
        )
        return RadialCenterDistanceFields(
            d_from_center=d_from_center,
            center_labels=propagated_center_labels,
            centers_i=np.asarray(centers_i, dtype=np.float64),
            centers_j=np.asarray(centers_j, dtype=np.float64),
        )

    def label_geometry(self, labels: np.ndarray) -> RadialLabelGeometry:
        """Return CP-compatible radial geometry derived only from object labels."""
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        cache_key = RadialLabelGeometryCacheKey.from_labels(labels_array)
        cached = _RADIAL_LABEL_GEOMETRY_CACHE.get(cache_key)
        if cached is not None:
            _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(cache_key)
            runtime_profiler.log(
                "idist_label_geometry_cache_hit",
                0.0,
                objects=int(labels_array.max()) if labels_array.size else 0,
            )
            return cached
        object_count = int(labels_array.max()) if labels_array.size else 0
        shape_backend = self.shape_geometry_backend()
        phase_started_at = time.perf_counter()
        d_to_edge = shape_backend.distance_to_edge(labels_array)
        runtime_profiler.log(
            "idist_distance_to_edge",
            time.perf_counter() - phase_started_at,
            objects=object_count,
        )
        phase_started_at = time.perf_counter()
        centers_i, centers_j = shape_backend.maximum_position_of_labels(
            d_to_edge, labels_array, np.arange(1, object_count + 1, dtype=np.int32)
        )
        runtime_profiler.log(
            "idist_maximum_position",
            time.perf_counter() - phase_started_at,
            objects=object_count,
        )
        geometry = RadialLabelGeometry(
            d_to_edge=d_to_edge,
            center_fields=self.center_distance_fields(
                labels_array, centers_i, centers_j
            ),
        )
        _RADIAL_LABEL_GEOMETRY_CACHE[cache_key] = geometry
        _RADIAL_LABEL_GEOMETRY_CACHE.move_to_end(cache_key)
        while len(_RADIAL_LABEL_GEOMETRY_CACHE) > _RADIAL_LABEL_GEOMETRY_CACHE_LIMIT:
            _RADIAL_LABEL_GEOMETRY_CACHE.popitem(last=False)
        return geometry


class NativeNumpyRadialDistributionBackendStrategy(RadialDistributionBackendStrategy):
    """CellProfiler-native NumPy radial-distribution backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NATIVE
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE
    is_default_backend = True

    def measure(
        self, request: RadialDistributionMeasureRequest
    ) -> RadialDistributionArrays:
        (
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
        ) = request.arrays()
        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = (
            int(request.bin_count)
            if request.wants_scaled
            else int(request.bin_count) + 1
        )
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=request.bin_count, wants_scaled=request.wants_scaled
            )
        good_mask = center_labels_array > 0
        normalized_distance = np.zeros(labels_array.shape, dtype=float)
        if request.wants_scaled:
            total_distance = d_from_center_array + d_to_edge_array
            normalized_distance[good_mask] = d_from_center_array[good_mask] / (
                total_distance[good_mask] + 0.001
            )
        else:
            normalized_distance[good_mask] = (
                d_from_center_array[good_mask] / request.maximum_radius
            )
        good_labels = labels_array[good_mask]
        bin_indexes = (normalized_distance * int(request.bin_count)).astype(int)
        bin_indexes[bin_indexes > int(request.bin_count)] = int(request.bin_count)
        labels_and_bins = (good_labels - 1, bin_indexes[good_mask])
        histogram = scipy.sparse.coo_matrix(
            (image_array[good_mask], labels_and_bins),
            (object_count, int(request.bin_count) + 1),
        ).toarray()
        sum_by_object = np.sum(histogram, 1)
        fraction_at_distance = (
            histogram / np.dstack([sum_by_object] * (int(request.bin_count) + 1))[0]
        )
        ngood_pixels = int(np.sum(good_mask))
        number_at_distance = scipy.sparse.coo_matrix(
            (np.ones(ngood_pixels), labels_and_bins),
            (object_count, int(request.bin_count) + 1),
        ).toarray()
        sum_by_object = np.sum(number_at_distance, 1)
        fraction_at_bin = (
            number_at_distance
            / np.dstack([sum_by_object] * (int(request.bin_count) + 1))[0]
        )
        mean_pixel_fraction = fraction_at_distance / (
            fraction_at_bin + np.finfo(float).eps
        )
        object_has_pixels = sum_by_object > 0
        radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=float)
        row_index, column_index = np.mgrid[
            0 : labels_array.shape[0], 0 : labels_array.shape[1]
        ]
        i_center = np.zeros(labels_array.shape, dtype=float)
        j_center = np.zeros(labels_array.shape, dtype=float)
        i_center[good_mask] = centers_i_array[center_labels_array[good_mask] - 1]
        j_center[good_mask] = centers_j_array[center_labels_array[good_mask] - 1]
        imask = row_index[good_mask] > i_center[good_mask]
        jmask = column_index[good_mask] > j_center[good_mask]
        absmask = np.abs(row_index[good_mask] - i_center[good_mask]) > np.abs(
            column_index[good_mask] - j_center[good_mask]
        )
        radial_index = (
            imask.astype(int) + jmask.astype(int) * 2 + absmask.astype(int) * 4
        )
        for bin_index in range(n_bins):
            bin_mask = good_mask & (bin_indexes == bin_index)
            bin_pixels = int(np.sum(bin_mask))
            bin_labels = labels_array[bin_mask]
            bin_radial_index = radial_index[bin_indexes[good_mask] == bin_index]
            labels_and_radii = (bin_labels - 1, bin_radial_index)
            radial_values = scipy.sparse.coo_matrix(
                (image_array[bin_mask], labels_and_radii), (object_count, 8)
            ).toarray()
            pixel_count = scipy.sparse.coo_matrix(
                (np.ones(bin_pixels), labels_and_radii), (object_count, 8)
            ).toarray()
            mask = pixel_count == 0
            radial_means = np.ma.masked_array(radial_values / pixel_count, mask)
            radial_cv = np.std(radial_means, 1) / np.mean(radial_means, 1)
            radial_cv[np.sum(~mask, 1) == 0] = 0
            radial_cv_by_bin[bin_index] = np.asarray(radial_cv.filled(0), dtype=float)
        return RadialDistributionArrays.from_components(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )


class NumbaNumpyRadialDistributionBackendStrategy(RadialDistributionBackendStrategy):
    """Numba-accelerated NumPy radial-distribution backend."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = False

    def prepare_backend(self) -> None:
        labels = np.zeros((8, 8), dtype=np.int32)
        labels[2:6, 2:6] = 1
        image = np.zeros((8, 8), dtype=np.float32)
        d_to_edge = np.ones((8, 8), dtype=np.float64)
        centers_i = np.array([3.5], dtype=np.float64)
        centers_j = np.array([3.5], dtype=np.float64)
        self.measure_from_centers(
            image,
            labels,
            d_to_edge,
            centers_i,
            centers_j,
            bin_count=4,
            wants_scaled=True,
            maximum_radius=100,
        )
        geometry = self.label_geometry(labels)
        self.measure_batch_self_centered_with_geometry(
            (image, image),
            labels,
            geometry,
            bin_count=4,
            wants_scaled=True,
            maximum_radius=100,
        )

    def measure(
        self, request: RadialDistributionMeasureRequest
    ) -> RadialDistributionArrays:
        (
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
        ) = request.arrays()
        object_count = int(labels_array.max()) if labels_array.size else 0
        n_bins = (
            int(request.bin_count)
            if request.wants_scaled
            else int(request.bin_count) + 1
        )
        if object_count <= 0:
            return RadialDistributionArrays.empty(
                bin_count=request.bin_count, wants_scaled=request.wants_scaled
            )
        (
            fraction_at_distance,
            mean_pixel_fraction,
            radial_cv_by_bin,
            object_has_pixels,
        ) = _measure_radial_distribution_numba(
            image_array,
            labels_array,
            d_to_edge_array,
            d_from_center_array,
            center_labels_array,
            centers_i_array,
            centers_j_array,
            int(request.bin_count),
            bool(request.wants_scaled),
            int(request.maximum_radius),
            object_count,
        )
        return RadialDistributionArrays.from_components(
            fraction_at_distance=fraction_at_distance,
            mean_pixel_fraction=mean_pixel_fraction,
            radial_cv_by_bin=radial_cv_by_bin,
            object_has_pixels=object_has_pixels,
            n_bins=n_bins,
        )

    def measure_batch_self_centered_with_geometry(
        self,
        images: Sequence[np.ndarray],
        labels: np.ndarray,
        geometry: RadialLabelGeometry,
        *,
        bin_count: int,
        wants_scaled: bool,
        maximum_radius: int,
    ) -> tuple[RadialDistributionArrays, ...]:
        """Reuse radial bin/wedge geometry across same-label image planes."""
        labels_array = np.ascontiguousarray(labels, dtype=np.int32)
        object_count = int(labels_array.max()) if labels_array.size else 0
        if object_count <= 0:
            empty = RadialDistributionArrays.empty(
                bin_count=bin_count, wants_scaled=wants_scaled
            )
            return tuple((empty for _image in images))
        index = RadialDistributionGeometryIndex(
            *_radial_distribution_geometry_index_numba(
                labels_array,
                np.ascontiguousarray(geometry.d_to_edge, dtype=np.float64),
                np.ascontiguousarray(
                    geometry.center_fields.d_from_center, dtype=np.float64
                ),
                np.ascontiguousarray(
                    geometry.center_fields.center_labels, dtype=np.int32
                ),
                np.ascontiguousarray(
                    geometry.center_fields.centers_i, dtype=np.float64
                ),
                np.ascontiguousarray(
                    geometry.center_fields.centers_j, dtype=np.float64
                ),
                int(bin_count),
                bool(wants_scaled),
                int(maximum_radius),
                object_count,
            )
        )
        outputs: list[RadialDistributionArrays] = []
        for image in images:
            (
                fraction_at_distance,
                mean_pixel_fraction,
                radial_cv_by_bin,
                object_has_pixels,
            ) = _measure_radial_distribution_from_geometry_index_numba(
                np.ascontiguousarray(image),
                index.pixel_rows,
                index.pixel_cols,
                index.object_indices,
                index.bin_indices,
                index.radial_indices,
                index.number_at_distance,
                index.radial_counts,
                index.object_count,
                index.bin_count,
                index.n_bins,
            )
            outputs.append(
                RadialDistributionArrays.from_components(
                    fraction_at_distance=fraction_at_distance,
                    mean_pixel_fraction=mean_pixel_fraction,
                    radial_cv_by_bin=radial_cv_by_bin,
                    object_has_pixels=object_has_pixels,
                    n_bins=index.n_bins,
                )
            )
        return tuple(outputs)


def radial_distribution_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> RadialDistributionBackendStrategy:
    """Return the selected radial-distribution backend."""
    return RadialDistributionBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


__all__ = public_names_from_objects(
    NativeNumpyRadialDistributionBackendStrategy,
    NumbaNumpyRadialDistributionBackendStrategy,
    RadialDistributionArrays,
    RadialDistributionBackendStrategy,
    RadialLabelGeometry,
    RadialLabelGeometryCacheKey,
    radial_distribution_backend,
)


@njit(cache=True)
def _numpy_divide_scalar(numerator: float, denominator: float) -> float:
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return np.nan
    if numerator > 0.0:
        return np.inf
    return -np.inf


@njit(cache=True)
def _radial_cv_divide_scalar(numerator: float, denominator: float) -> float:
    if denominator != 0.0:
        return numerator / denominator
    if numerator == 0.0:
        return 0.0
    if numerator > 0.0:
        return np.inf
    return -np.inf


def _radial_distribution_geometry_index_numba(
    labels: np.ndarray,
    d_to_edge: np.ndarray,
    d_from_center: np.ndarray,
    center_labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    bin_count: int,
    wants_scaled: bool,
    maximum_radius: int,
    object_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    height, width = labels.shape
    n_bins = bin_count if wants_scaled else bin_count + 1
    valid_count = 0
    number_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    radial_counts = np.zeros((n_bins, object_count, 8), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius
            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0
            object_index = label_id - 1
            number_at_distance[object_index, bin_index] += 1.0
            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4
                radial_counts[bin_index, object_index, radial_index] += 1.0
            valid_count += 1
    pixel_rows = np.empty(valid_count, dtype=np.int64)
    pixel_cols = np.empty(valid_count, dtype=np.int64)
    object_indices = np.empty(valid_count, dtype=np.int64)
    bin_indices = np.empty(valid_count, dtype=np.int64)
    radial_indices = np.empty(valid_count, dtype=np.int64)
    out_index = 0
    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius
            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0
            radial_index = -1
            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4
            pixel_rows[out_index] = y
            pixel_cols[out_index] = x
            object_indices[out_index] = label_id - 1
            bin_indices[out_index] = bin_index
            radial_indices[out_index] = radial_index
            out_index += 1
    return (
        pixel_rows,
        pixel_cols,
        object_indices,
        bin_indices,
        radial_indices,
        number_at_distance,
        radial_counts,
        object_count,
        bin_count,
        n_bins,
    )


@njit(cache=True)
def _measure_radial_distribution_from_geometry_index_numba(
    image: np.ndarray,
    pixel_rows: np.ndarray,
    pixel_cols: np.ndarray,
    object_indices: np.ndarray,
    bin_indices: np.ndarray,
    radial_indices: np.ndarray,
    number_at_distance: np.ndarray,
    radial_counts: np.ndarray,
    object_count: int,
    bin_count: int,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    histogram = np.zeros((object_count, bin_count + 1), dtype=image.dtype)
    radial_values = np.zeros((n_bins, object_count, 8), dtype=image.dtype)
    for pixel_index in range(pixel_rows.size):
        object_index = object_indices[pixel_index]
        bin_index = bin_indices[pixel_index]
        pixel_value = image[pixel_rows[pixel_index], pixel_cols[pixel_index]]
        histogram[object_index, bin_index] += pixel_value
        radial_index = radial_indices[pixel_index]
        if radial_index >= 0:
            radial_values[bin_index, object_index, radial_index] += pixel_value
    fraction_at_distance = np.zeros((object_count, bin_count + 1), dtype=image.dtype)
    fraction_at_bin = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    object_has_pixels = np.zeros(object_count, dtype=np.bool_)
    eps = np.finfo(np.float64).eps
    for object_index in range(object_count):
        intensity_sum = 0.0
        pixel_count = 0.0
        for bin_index in range(bin_count + 1):
            intensity_sum += histogram[object_index, bin_index]
            pixel_count += number_at_distance[object_index, bin_index]
        if pixel_count > 0.0:
            object_has_pixels[object_index] = True
        for bin_index in range(bin_count + 1):
            fraction_at_distance[object_index, bin_index] = _numpy_divide_scalar(
                histogram[object_index, bin_index], intensity_sum
            )
            fraction_at_bin[object_index, bin_index] = _numpy_divide_scalar(
                number_at_distance[object_index, bin_index], pixel_count
            )
    mean_pixel_fraction = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    for object_index in range(object_count):
        for bin_index in range(bin_count + 1):
            mean_pixel_fraction[object_index, bin_index] = fraction_at_distance[
                object_index, bin_index
            ] / (fraction_at_bin[object_index, bin_index] + eps)
    radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=np.float64)
    for bin_index in range(n_bins):
        for object_index in range(object_count):
            populated_wedges = 0
            wedge_sum = 0.0
            wedge_sum_sq = 0.0
            for radial_index in range(8):
                count = radial_counts[bin_index, object_index, radial_index]
                if count <= 0.0:
                    continue
                radial_mean = (
                    radial_values[bin_index, object_index, radial_index] / count
                )
                populated_wedges += 1
                wedge_sum += radial_mean
                wedge_sum_sq += radial_mean * radial_mean
            if populated_wedges == 0:
                continue
            mean = wedge_sum / populated_wedges
            variance = wedge_sum_sq / populated_wedges - mean * mean
            if variance < 0.0:
                variance = 0.0
            radial_cv_by_bin[bin_index, object_index] = _radial_cv_divide_scalar(
                np.sqrt(variance), mean
            )
    return (
        fraction_at_distance,
        mean_pixel_fraction,
        radial_cv_by_bin,
        object_has_pixels,
    )


@njit(cache=True)
def _measure_radial_distribution_numba(
    image: np.ndarray,
    labels: np.ndarray,
    d_to_edge: np.ndarray,
    d_from_center: np.ndarray,
    center_labels: np.ndarray,
    centers_i: np.ndarray,
    centers_j: np.ndarray,
    bin_count: int,
    wants_scaled: bool,
    maximum_radius: int,
    object_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = labels.shape
    histogram = np.zeros((object_count, bin_count + 1), dtype=image.dtype)
    number_at_distance = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    n_bins = bin_count if wants_scaled else bin_count + 1
    radial_values = np.zeros((n_bins, object_count, 8), dtype=image.dtype)
    radial_counts = np.zeros((n_bins, object_count, 8), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            label_id = labels[y, x]
            if label_id <= 0 or label_id > object_count or center_labels[y, x] <= 0:
                continue
            object_index = label_id - 1
            if wants_scaled:
                denominator = d_from_center[y, x] + d_to_edge[y, x] + 0.001
                normalized_distance = d_from_center[y, x] / denominator
            else:
                normalized_distance = d_from_center[y, x] / maximum_radius
            bin_index = int(normalized_distance * bin_count)
            if bin_index > bin_count:
                bin_index = bin_count
            if bin_index < 0:
                bin_index = 0
            pixel_value = image[y, x]
            histogram[object_index, bin_index] += pixel_value
            number_at_distance[object_index, bin_index] += 1.0
            if bin_index < n_bins:
                center_index = center_labels[y, x] - 1
                center_i = centers_i[center_index]
                center_j = centers_j[center_index]
                imask = 1 if y > center_i else 0
                jmask = 1 if x > center_j else 0
                absmask = 1 if abs(y - center_i) > abs(x - center_j) else 0
                radial_index = imask + jmask * 2 + absmask * 4
                radial_values[bin_index, object_index, radial_index] += pixel_value
                radial_counts[bin_index, object_index, radial_index] += 1.0
    fraction_at_distance = np.zeros((object_count, bin_count + 1), dtype=image.dtype)
    fraction_at_bin = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    object_has_pixels = np.zeros(object_count, dtype=np.bool_)
    eps = np.finfo(np.float64).eps
    for object_index in range(object_count):
        intensity_sum = 0.0
        pixel_count = 0.0
        for bin_index in range(bin_count + 1):
            intensity_sum += histogram[object_index, bin_index]
            pixel_count += number_at_distance[object_index, bin_index]
        if pixel_count > 0.0:
            object_has_pixels[object_index] = True
        for bin_index in range(bin_count + 1):
            fraction_at_distance[object_index, bin_index] = _numpy_divide_scalar(
                histogram[object_index, bin_index], intensity_sum
            )
            fraction_at_bin[object_index, bin_index] = _numpy_divide_scalar(
                number_at_distance[object_index, bin_index], pixel_count
            )
    mean_pixel_fraction = np.zeros((object_count, bin_count + 1), dtype=np.float64)
    for object_index in range(object_count):
        for bin_index in range(bin_count + 1):
            mean_pixel_fraction[object_index, bin_index] = fraction_at_distance[
                object_index, bin_index
            ] / (fraction_at_bin[object_index, bin_index] + eps)
    radial_cv_by_bin = np.zeros((n_bins, object_count), dtype=np.float64)
    for bin_index in range(n_bins):
        for object_index in range(object_count):
            populated_wedges = 0
            wedge_sum = 0.0
            wedge_sum_sq = 0.0
            for radial_index in range(8):
                count = radial_counts[bin_index, object_index, radial_index]
                if count <= 0.0:
                    continue
                radial_mean = (
                    radial_values[bin_index, object_index, radial_index] / count
                )
                populated_wedges += 1
                wedge_sum += radial_mean
                wedge_sum_sq += radial_mean * radial_mean
            if populated_wedges == 0:
                continue
            mean = wedge_sum / populated_wedges
            variance = wedge_sum_sq / populated_wedges - mean * mean
            if variance < 0.0:
                variance = 0.0
            radial_cv_by_bin[bin_index, object_index] = _radial_cv_divide_scalar(
                np.sqrt(variance), mean
            )
    return (
        fraction_at_distance,
        mean_pixel_fraction,
        radial_cv_by_bin,
        object_has_pixels,
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.SLICE_ALIGNED)
@runtime_bound_parameters(
    SliceIndexRuntimeParameter,
    _IntensityDistributionHeatmapOutputsRuntimeParameter,
)
def measure_object_intensity_distribution(
    image: np.ndarray,
    labels: ObjectLabelValue,
    bin_count: int = 4,
    wants_scaled: bool = True,
    maximum_radius: int = 100,
    wants_zernikes: ZernikeMode = ZernikeMode.NONE,
    zernike_degree: int = 9,
    center_choice: CenterChoice = CenterChoice.SELF,
    radial_distribution_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    zernike_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
    heatmap_groups: tuple[IntensityDistributionHeatmapGroup, ...] = (),
    heatmap_outputs: tuple[IntensityDistributionHeatmapRuntimeOutput, ...] = (),
) -> tuple[np.ndarray | AlignedImageStack, ColumnarRows]:
    """Measure CellProfiler-compatible object intensity distribution rows.

    Args:
        labels: Object-label plane defining the regions for radial-distribution
            and Zernike measurements.
        heatmap_groups: Ordered heatmap declarations selecting the source image,
            object set, radial bin or measurement, colormap, and output name for
            each retained heatmap.
    """
    total_started_at = time.perf_counter()
    profiler = IntensityDistributionProfiler(
        function_name="measure_object_intensity_distribution"
    )
    del center_choice
    source_image_names = image_payload_metadata(image).source_image_names
    if len(source_image_names) != 1:
        raise ValueError(
            "MeasureObjectIntensityDistribution requires exactly one declared "
            f"source image name, got {source_image_names!r}."
        )
    source_image_name = source_image_names[0]
    radial_backend = radial_distribution_backend(
        backend_provider=radial_distribution_backend_provider
    )
    image_array = np.asarray(image_payload_data(image))
    image_mask = image_payload_mask(image)
    projected_mask = image_mask_for_data_domain(
        source_payload=image,
        data=image_array,
    )
    if image_mask is not None and projected_mask is None:
        raise ValueError(
            "MeasureObjectIntensityDistribution image mask must already match "
            f"the projected image domain; got mask {np.shape(image_mask)!r} for "
            f"image {image_array.shape!r}."
        )
    slice_input = IntensityDistributionSliceInput.from_aligned_arrays(
        image=image_array,
        image_mask=(
            None if projected_mask is None else np.asarray(projected_mask, dtype=bool)
        ),
        labels=labels,
        slice_index=0 if slice_index is None else int(slice_index),
    )
    measurements = IntensityDistributionMeasurementRequest(
        slice_input=slice_input,
        source_image_name=source_image_name,
        bin_count=bin_count,
        wants_scaled=wants_scaled,
        maximum_radius=maximum_radius,
        wants_zernikes=wants_zernikes,
        zernike_degree=zernike_degree,
        radial_backend=radial_backend,
        zernike_backend_provider=zernike_backend_provider,
        profiler=profiler,
    ).rows()
    profiler.record_rows("idist_total", total_started_at, len(measurements))
    active_heatmap_groups = tuple(
        group for group in heatmap_groups if group.save_display
    )
    runtime_heatmap_groups = tuple(output.group for output in heatmap_outputs)
    if active_heatmap_groups != runtime_heatmap_groups:
        raise ValueError(
            "MeasureObjectIntensityDistribution runtime heatmaps do not match "
            "the public saved-heatmap groups: "
            f"{runtime_heatmap_groups!r} != {active_heatmap_groups!r}."
        )
    output = (
        pack_aligned_image_outputs(
            tuple(
                _intensity_distribution_heatmap(
                    image,
                    request,
                    radial_backend=radial_backend,
                )
                for request in heatmap_outputs
            )
        )
        if heatmap_outputs
        else image
    )
    return (output, measurements)


def _intensity_distribution_heatmap(
    image: np.ndarray,
    request: IntensityDistributionHeatmapRuntimeOutput,
    *,
    radial_backend: "RadialDistributionBackendStrategy",
) -> RuntimeArrayData:
    """Render one CellProfiler radial-distribution heatmap."""

    group = request.group
    source_names = image_payload_metadata(image).source_image_names
    if source_names != (group.source_image_name,):
        raise ValueError(
            "MeasureObjectIntensityDistribution heatmap source does not match "
            f"the active image payload: {group.source_image_name!r} not in "
            f"{source_names!r}."
        )
    image_array = np.asarray(image_payload_data(image))
    image_mask = image_payload_mask(image)
    projected_mask = image_mask_for_data_domain(
        source_payload=image,
        data=image_array,
    )
    if image_mask is not None and projected_mask is None:
        raise ValueError(
            "MeasureObjectIntensityDistribution heatmap image mask must match "
            f"the source image domain: {np.shape(image_mask)!r} for "
            f"{image_array.shape!r}."
        )
    slice_input = IntensityDistributionSliceInput.from_aligned_arrays(
        image=image_array,
        image_mask=(
            None if projected_mask is None else np.asarray(projected_mask, dtype=bool)
        ),
        labels=request.object_labels,
        slice_index=0,
    )
    labels = slice_input.radial_labels
    radial_arrays = radial_backend.measure_self_centered(
        slice_input.image,
        labels,
        bin_count=group.bin_count,
        wants_scaled=group.wants_scaled,
        maximum_radius=group.maximum_radius,
    )
    heatmap = np.zeros(labels.shape, dtype=float)
    if labels.size and int(labels.max()) > 0:
        geometry = radial_backend.label_geometry(labels)
        center_fields = geometry.center_fields
        valid = (labels > 0) & (center_fields.center_labels > 0)
        normalized_distance = np.zeros(labels.shape, dtype=float)
        if group.wants_scaled:
            total_distance = center_fields.d_from_center + geometry.d_to_edge
            normalized_distance[valid] = center_fields.d_from_center[valid] / (
                total_distance[valid] + 0.001
            )
        else:
            normalized_distance[valid] = (
                center_fields.d_from_center[valid] / group.maximum_radius
            )
        bin_indices = (normalized_distance * group.bin_count).astype(int)
        bin_indices[bin_indices > group.bin_count] = group.bin_count
        object_indices = labels - 1
        renderable = (
            valid
            & (object_indices >= 0)
            & (object_indices < radial_arrays.object_has_pixels.size)
            & (bin_indices >= 0)
            & (bin_indices < radial_arrays.n_bins)
        )
        rows, columns = np.nonzero(renderable)
        object_values = object_indices[rows, columns]
        bin_values = bin_indices[rows, columns]
        if (
            group.measurement
            is IntensityDistributionHeatmapMeasurement.FRACTION_AT_DISTANCE
        ):
            values = radial_arrays.fraction_at_distance[object_values, bin_values]
        elif group.measurement is IntensityDistributionHeatmapMeasurement.MEAN_FRACTION:
            values = radial_arrays.mean_pixel_fraction[object_values, bin_values]
        elif group.measurement is IntensityDistributionHeatmapMeasurement.RADIAL_CV:
            values = radial_arrays.radial_cv_by_bin[bin_values, object_values]
        else:
            raise ValueError(
                "Unsupported intensity-distribution heatmap measurement "
                f"{group.measurement!r}."
            )
        heatmap[rows, columns] = values
    colormap_name = group.colormap
    if colormap_name == "gray":
        return with_image_payload_data(
            image,
            heatmap,
            metadata=image_payload_metadata(image).without_source_channel_axis(),
        )
    if colormap_name == "Default":
        colormap_name = "viridis"
    import matplotlib
    from matplotlib.cm import ScalarMappable

    rgb = ScalarMappable(cmap=matplotlib.colormaps[colormap_name]).to_rgba(heatmap)[
        :, :, :3
    ]
    rgb[labels == 0] = 0
    return with_image_payload_data(
        image,
        rgb,
        metadata=image_payload_metadata(image).replace_fields(source_channel_axis=-1),
    )


def intensity_distribution_object_domain(labels: object) -> tuple[int, ...]:
    """Return the object domain for CP intensity-distribution rows."""
    return dense_object_label_id_domain(labels)


def _prepare_measure_object_intensity_distribution() -> None:
    """Compile radial-distribution and intensity-Zernike kernels before execution."""
    image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((64, 64))
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    measure_object_intensity_distribution.__wrapped__(
        ImagePayloadMetadata(source_image_names=("prepare_image",)).attach_to(image),
        ObjectLabelPayload(
            variant_data=ObjectLabelVariantData(labels=labels),
            domain=ObjectLabelDomain(declared_object_ids=(1, 2)),
        ),
        bin_count=4,
        wants_scaled=True,
        maximum_radius=100,
        wants_zernikes=ZernikeMode.MAGNITUDES_AND_PHASE,
        zernike_degree=9,
    )


measure_object_intensity_distribution.__openhcs_prepare__ = (
    _prepare_measure_object_intensity_distribution
)
