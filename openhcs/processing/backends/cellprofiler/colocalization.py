"""Colocalization backends for CellProfiler-compatible measurements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field, fields, make_dataclass, replace
from enum import Enum
import logging
from types import MappingProxyType
import time
from typing import Annotated, ClassVar, Tuple

from metaclass_registry import AutoRegisterMeta
import numpy as np

from openhcs.constants.constants import GroupBy, MemoryType, VariableComponents
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    ImagePayloadSliceProjector,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import (
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import (
    KeywordRuntimeParameter,
    runtime_image_execution_mode,
)
from openhcs.core.memory import numpy
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
)
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    composed_image_payload,
    object_label_input_execution_mode,
    required_variable_components,
    runtime_bound_parameters,
    special_inputs,
    )
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_batch_contracts import (
    RuntimeBatchInvocationRequest,
    measurement_image_batch_executor,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    image_intensity_scale_for_dtype,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    image_payload_slice_context,
    with_image_payload_data,
)
from openhcs.core.runtime_object_label_aggregation import DenseObjectLabelAggregation
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    RuntimeMeasurementFeature,
    RuntimeMeasurementFeatureRelation,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimeSliceInvariantValue,
    RuntimeSliceProjectableValue,
)
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.interop.cellprofiler.measurement_scope import (
    CellProfilerMeasurementTargetScope,
    coerce_cellprofiler_measurement_target_scope,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    PerObjectMeasurementExecutionModule,
    ScopedMeasurementModule,
    SourceQualifiedMeasurementFeatureModule,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerMeasurementImage,
    CellProfilerSourceImagePair,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    PreparedObjectMeasurementInvocation,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    DenseColumnarObjectMeasurementRowsMixin,
    ObjectMeasurementInvocation,
    SourcePairObjectMeasurementInvocation,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.core.steps.function_runtime import RuntimeCallableKwargs
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    optional_setting_value,
    repeating_setting_blocks,
    setting_name_matches,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    SettingToKeywordBinding,
    cellprofiler_enum_value_setting_parser,
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendAuthority,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
)
from openhcs.processing.backends.cellprofiler.colocalization_costes import (
    UnitIntervalDenseRankSemantics,
    _correlation_slopes_numba,
    _costes_manders_numba,
    _linear_costes_numba,
    _linear_costes_sorted_events_numba,
    _pearson_below_threshold_numba,
    _regression_line_numba,
    costes_above_threshold_mask,
    object_colocalization_base_reductions,
    object_colocalization_correlation_reductions,
    object_colocalization_rwc_reductions,
    object_colocalization_threshold_reductions,
    thresholded_colocalization_metrics,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.object_measurement_columnar_rows import (
    ObjectMeasurementColumnarRows,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)


@dataclass(frozen=True, slots=True)
class ColocalizationThresholdMaskGroup:
    """One saved threshold-mask group from the serialized module settings."""

    source_image_name: str
    output_image_name: str
    threshold_percent: float
    object_name: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("source_image_name", self.source_image_name),
            ("output_image_name", self.output_image_name),
        ):
            if not value.strip() or value != value.strip():
                raise ValueError(
                    f"ColocalizationThresholdMaskGroup.{field_name} must be a "
                    "normalized non-empty string."
                )
        if self.object_name is not None and (
            not self.object_name.strip() or self.object_name != self.object_name.strip()
        ):
            raise ValueError(
                "ColocalizationThresholdMaskGroup.object_name must be None or a "
                "normalized non-empty string."
            )


ColocalizationThresholdMaskGroupsInput = Annotated[
    tuple[ColocalizationThresholdMaskGroup, ...],
    (
        "Ordered retained-mask declarations; each names the source image, output "
        "image, threshold percentage, and optional object-label domain."
    ),
]


@dataclass(frozen=True)
class ColocalizationThresholdMaskSourceRelation(SourceStackLineageSourceRelation):
    """Declare the exact source image and threshold for a saved mask."""

    relation_key: ClassVar[str] = "colocalization_threshold_mask_source"
    target_artifact_type = ImageArtifactType

    threshold_percent: float


class ColocalizationThresholdMaskObjectRelation(ArtifactSpecRelation):
    """Declare the optional object-label domain used to threshold one mask."""

    relation_key: ClassVar[str] = "colocalization_threshold_mask_object"
    target_artifact_type = ImageArtifactType

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.source.artifact_type is not ObjectLabelsArtifactType:
            raise ValueError(
                "Colocalization threshold-mask object relation requires an object "
                f"source, got {self.source.artifact_type.value}:{self.source.name}."
            )


@dataclass(frozen=True, slots=True)
class ColocalizationThresholdMaskRuntimeOutput(RuntimeSliceProjectableValue):
    """Runtime mask request in compiled image-output order."""

    group: ColocalizationThresholdMaskGroup
    source_channel_index: int
    object_labels: ObjectLabelValue | None = None

    def project_runtime_slice(
        self, slice_index: int
    ) -> "ColocalizationThresholdMaskRuntimeOutput":
        """Project the optional object-label domain through its declared axis."""
        if self.object_labels is None:
            return self
        projection = self.object_labels.declared_plane_projection()
        if projection is None or projection.axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return self
        projected = RuntimeSliceProjection.value_for_slice(
            self.object_labels,
            projection.selected_plane(slice_index),
        )
        return replace(self, object_labels=projected)


class _ColocalizationThresholdMaskOutputsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound threshold-mask requests from exact output relations."""

    parameter_name = "threshold_mask_outputs"
    annotation_type = tuple[ColocalizationThresholdMaskRuntimeOutput, ...]
    parameter_default = ()


@dataclass(frozen=True, slots=True)
class ColocalizationSourcePairFeatureRelation(RuntimeMeasurementFeatureRelation):
    """Exact CP family and source orientation for one colocalization feature."""

    cellprofiler_family_name: str
    source_endpoint_order: tuple[int, int]
    absorbed_field_name: str | None = None
    directional_alias_index: int | None = None
    measurement_scope: CellProfilerMeasurementTargetScope = (
        CellProfilerMeasurementTargetScope.BOTH
    )

    @property
    def runtime_feature_family(self) -> str:
        """Return the normalized runtime family represented by this relation."""
        return self.cellprofiler_family_name.lower()

    def source_family_names(
        self, source_feature: RuntimeMeasurementFeature
    ) -> tuple[str, ...]:
        """Expose this member's declared source-pair feature family."""
        del source_feature
        return (self.runtime_feature_family,)

    def target_family_name(
        self,
        source_feature: RuntimeMeasurementFeature,
        source_family_name: str,
        feature_type: type[RuntimeMeasurementFeature],
    ) -> str | None:
        """This naming relation does not transform into another feature family."""
        del source_feature, source_family_name, feature_type
        return None

    def feature_name(self, source_pair: CellProfilerSourceImagePair) -> str:
        """Return the exact CellProfiler column name for one source pair."""
        source_names = (
            source_pair.first.display_name,
            source_pair.second.display_name,
        )
        first_name, second_name = (
            source_names[index] for index in self.source_endpoint_order
        )
        return (
            f"{MeasureColocalizationModule.source_qualified_measurement_category()}_"
            f"{self.cellprofiler_family_name}_"
            f"{first_name}_{second_name}"
        )

    def measurement_row_field_name(
        self, source_feature: RuntimeMeasurementFeature
    ) -> str:
        """Return the exact absorbed result field represented by one member."""
        return self.absorbed_field_name or source_feature.value


class MeasureColocalizationObjectMeasurementRowPolicy(
    DenseColumnarObjectMeasurementRowsMixin, CellProfilerObjectMeasurementRowPolicy
):
    """Expand composed source stacks into source-pair object measurements."""

    def invocations(
        self,
        measurement_image: "CellProfilerMeasurementImage",
        kwargs: 'RuntimeCallableKwargs',
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        source_pairs = measurement_image.source_image_pairs()
        if not source_pairs:
            raise ValueError(
                "MeasureColocalization requires one exact declared source-image pair."
            )
        return tuple(
            (
                SourcePairObjectMeasurementInvocation(
                    kwargs={
                        **kwargs,
                        **source_pair.invocation_kwargs(
                            first_channel_kwarg="channel_1",
                            second_channel_kwarg="channel_2",
                        ),
                    },
                    source_pair=source_pair,
                )
                for source_pair in source_pairs
            )
        )

    def project_rows(
        self, rows: ColumnarRows, invocation: ObjectMeasurementInvocation
    ) -> ColumnarRows:
        if invocation.source_pair is None:
            raise ValueError(
                "MeasureColocalization row projection requires an exact source pair."
            )
        return MeasureColocalizationModule.project_source_pair_columnar_rows(
            rows, invocation.source_pair
        )

    def table_source_image_name(
        self,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
        source_image_name: str | None,
    ) -> str | None:
        if not measurement_images:
            return super().table_source_image_name(
                measurement_images,
                source_image_name,
            )
        source_pairs = tuple(
            (
                source_pair
                for measurement_image in measurement_images
                for source_pair in measurement_image.source_image_pairs()
            )
        )
        match source_pairs:
            case (source_pair,):
                return source_pair.source_image_name
            case _:
                return None


def _measure_colocalization_scope(
    value: str,
) -> CellProfilerMeasurementTargetScope:
    """Parse CellProfiler's colocalization target-scope literals."""

    return cellprofiler_enum_from_literal(
        CellProfilerMeasurementTargetScope,
        value,
        aliases={
            "across entire image": CellProfilerMeasurementTargetScope.IMAGE,
            "within objects": CellProfilerMeasurementTargetScope.OBJECT,
            "objects": CellProfilerMeasurementTargetScope.OBJECT,
        },
    )


class MeasureColocalizationModule(
    LabelsObjectInputPolicy,
    NoObjectNameMeasurementRecordMixin,
    MeasureColocalizationObjectMeasurementRowPolicy,
    PerObjectMeasurementExecutionModule,
    SourceQualifiedMeasurementFeatureModule,
    ScopedMeasurementModule,
):
    module_name = "MeasureColocalization"
    function_name = "measure_colocalization"
    validated = True
    aliases = ("MeasureCorrelation",)
    function_variants = ("measure_colocalization_objects",)
    group_by = GroupBy.SITE
    confidence = 1.0
    measurement_category_prefixes = (
        ("correlation",),
        ("colocalization",),
    )
    measurement_scope_binding = SettingToKeywordBinding(
        SettingNameFamily("Select where to measure correlation"),
        "measurement_scope",
        _measure_colocalization_scope,
    )
    measurement_scope_default = CellProfilerMeasurementTargetScope.IMAGE
    object_gate_setting = SettingNameFamily("Select an object to measure")
    ignored_settings = ("Hidden",)

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Exact source-pair feature identities emitted by MeasureColocalization."""

        CORRELATION = (
            "correlation",
            (ColocalizationSourcePairFeatureRelation("Correlation", (0, 1)),),
        )
        REGRESSION_SLOPE = (
            "slope",
            (
                ColocalizationSourcePairFeatureRelation(
                    "Slope",
                    (0, 1),
                    measurement_scope=CellProfilerMeasurementTargetScope.IMAGE,
                ),
            ),
        )
        OVERLAP = (
            "overlap",
            (ColocalizationSourcePairFeatureRelation("Overlap", (0, 1)),),
        )
        OVERLAP_K_FIRST = (
            "k_1",
            (ColocalizationSourcePairFeatureRelation("K", (0, 1), "k1", 1),),
        )
        OVERLAP_K_SECOND = (
            "k_2",
            (ColocalizationSourcePairFeatureRelation("K", (1, 0), "k2", 2),),
        )
        MANDERS_FIRST = (
            "manders_m_1",
            (
                ColocalizationSourcePairFeatureRelation(
                    "Manders", (0, 1), "manders_m1", 1
                ),
            ),
        )
        MANDERS_SECOND = (
            "manders_m_2",
            (
                ColocalizationSourcePairFeatureRelation(
                    "Manders", (1, 0), "manders_m2", 2
                ),
            ),
        )
        RANK_WEIGHTED_FIRST = (
            "rwc_1",
            (ColocalizationSourcePairFeatureRelation("RWC", (0, 1), "rwc1", 1),),
        )
        RANK_WEIGHTED_SECOND = (
            "rwc_2",
            (ColocalizationSourcePairFeatureRelation("RWC", (1, 0), "rwc2", 2),),
        )
        COSTES_MANDERS_FIRST = (
            "costes_m_1",
            (
                ColocalizationSourcePairFeatureRelation(
                    "Costes", (0, 1), "costes_m1", 1
                ),
            ),
        )
        COSTES_MANDERS_SECOND = (
            "costes_m_2",
            (
                ColocalizationSourcePairFeatureRelation(
                    "Costes", (1, 0), "costes_m2", 2
                ),
            ),
        )

        @property
        def source_pair_relation(self) -> ColocalizationSourcePairFeatureRelation:
            """Return this member's single source-pair naming relation."""
            relations = tuple(
                relation
                for relation in self.relations
                if isinstance(relation, ColocalizationSourcePairFeatureRelation)
            )
            if len(relations) != 1:
                raise ValueError(
                    f"{type(self).__name__}.{self.name} must declare exactly one "
                    "ColocalizationSourcePairFeatureRelation."
                )
            return relations[0]

        @property
        def measurement_row_field_name(self) -> str:
            """Return the exact absorbed result field owned by this member."""
            return self.source_pair_relation.measurement_row_field_name(self)

        def feature_family(self) -> str:
            """Return this member's canonical runtime feature family."""
            return self.source_pair_relation.runtime_feature_family

        def emitted_in_scope(self, scope: MeasurementScope) -> bool:
            """Return whether this feature is emitted for the declared scope."""
            scope_selection = (
                self.source_pair_relation.measurement_scope.measurement_scope_selection
            )
            return scope_selection.includes(scope)

        def source_pair_feature_name(
            self, source_pair: CellProfilerSourceImagePair
        ) -> str:
            """Return the exact CellProfiler column name for one source pair."""
            return self.source_pair_relation.feature_name(source_pair)

    directional_pair_feature_aliases = MappingProxyType(
        {
            feature.value: (
                feature.feature_family(),
                feature.source_pair_relation.directional_alias_index,
            )
            for feature in MeasurementFeature
            if feature.source_pair_relation.directional_alias_index is not None
        }
    )
    scale_qualified_measurement_feature_prefixes = (
        (MeasurementFeature.CORRELATION.feature_family(),),
    )
    pair_correlation_feature_name = MeasurementFeature.CORRELATION.feature_family()
    pair_regression_slope_feature_name = (
        MeasurementFeature.REGRESSION_SLOPE.feature_family()
    )
    undirected_pair_feature_names = frozenset(
        {
            MeasurementFeature.CORRELATION.feature_family(),
            MeasurementFeature.OVERLAP.feature_family(),
        }
    )
    threshold_sensitive_pair_feature_names = frozenset(
        {
            MeasurementFeature.COSTES_MANDERS_FIRST.feature_family(),
            MeasurementFeature.MANDERS_FIRST.feature_family(),
            MeasurementFeature.RANK_WEIGHTED_FIRST.feature_family(),
            MeasurementFeature.OVERLAP_K_FIRST.feature_family(),
        }
    )

    @classmethod
    def project_source_pair_columnar_rows(
        cls,
        rows: ColumnarRows,
        source_pair: CellProfilerSourceImagePair,
    ) -> ColumnarRows:
        """Project columnar colocalization fields to exact source-pair names."""
        if isinstance(rows, ConcatenatedColumnarRows):
            return ConcatenatedColumnarRows(
                tuple(
                    cls.project_source_pair_columnar_rows(row_batch, source_pair)
                    for row_batch in rows.row_batches
                )
            )
        object_scope = MeasurementRowAxisField.OBJECT_LABEL.value in rows.columns
        row_type = (
            ObjectColocalizationMeasurements
            if object_scope
            else ColocalizationMeasurements
        )
        expected_fields = FieldSpec.from_dataclass_type(row_type)
        if rows.fields != expected_fields:
            raise ValueError(
                f"{cls.__name__} requires exact raw colocalization fields "
                f"{expected_fields!r}, got {rows.fields!r}."
            )
        measurement_scope = (
            MeasurementScope.OBJECT if object_scope else MeasurementScope.IMAGE
        )
        features_by_field_name = {
            feature.measurement_row_field_name: feature
            for feature in cls.MeasurementFeature
        }
        axis_field_names = MeasurementRowAxisField.field_names()
        projected_field_columns: list[tuple[FieldSpec, object]] = []
        for field_spec in expected_fields:
            field_name = field_spec.name
            if field_name in axis_field_names:
                projected_field_columns.append(
                    (field_spec, rows.column_values(field_name))
                )
                continue
            if field_name not in features_by_field_name:
                continue
            feature = features_by_field_name[field_name]
            if not feature.emitted_in_scope(measurement_scope):
                continue
            projected_field_columns.append(
                (
                    FieldSpec(
                        name=feature.source_pair_feature_name(source_pair),
                        dtype=field_spec.dtype,
                        required=field_spec.required,
                    ),
                    rows.column_values(field_name),
                )
            )
        return MeasurementProjectedColumnarRows(
            MappingProxyType(
                {
                    field_spec.name: values
                    for field_spec, values in projected_field_columns
                }
            ),
            fields=tuple(field_spec for field_spec, _values in projected_field_columns),
            declared_object_measurement_domain_covered=(
                rows.covers_declared_object_measurement_domain
            ),
            object_row_identity=rows.object_row_identity,
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

    threshold_percent_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Set threshold as percentage of maximum intensity for the images"
    )
    correlation_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate correlation and slope metrics?"
    )
    manders_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate the Manders coefficients?"
    )
    rank_weighted_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate the Rank Weighted Colocalization coefficients?",
        aliases=("Calculate the Rank Weighted Coloalization coefficients?",),
    )
    overlap_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate the Overlap coefficients?"
    )
    costes_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Calculate the Manders coefficients using Costes auto threshold?"
    )
    run_all_metrics_setting = "Run all metrics?"
    costes_method_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Method for Costes thresholding"
    )
    channel_thresholds_enabled_setting = "Enable image specific thresholds?"
    channel_threshold_count_setting = "Threshold count"
    channel_threshold_image_setting = "Select the image"
    channel_threshold_percent_setting = (
        "Set threshold as percentage of maximum intensity of selected image"
    )
    save_masks_setting = "Save thresholded mask?"
    save_mask_count_setting = "Save mask count"
    save_mask_source_image_setting = "Which image mask would you like to save"
    save_mask_uses_objects_setting = "Use object for thresholding?"
    save_mask_object_setting = SettingNameFamily(
        "Select an Object for threhsolding",
        aliases=("Select an Object for thresholding",),
    )
    save_mask_output_image_setting = "Name the output image"
    save_mask_output_image_binding = SettingToKeywordBinding.output(
        save_mask_output_image_setting,
        ImageArtifactType,
        "threshold_mask_output_names",
        repeated=True,
    )
    save_mask_object_binding = SettingToKeywordBinding.input(
        save_mask_object_setting,
        ObjectLabelsArtifactType,
        repeated=True,
        )

    class CostesMethod(str, Enum):
        faster = "faster"
        fast = "fast"
        accurate = "accurate"

        @classmethod
        def from_literal(
            cls, value: "MeasureColocalizationModule.CostesMethod | str"
        ) -> "MeasureColocalizationModule.CostesMethod":
            return cellprofiler_enum_from_literal(cls, value)

    threshold_percent_binding: ClassVar[SettingToKeywordBinding] = (
        SettingToKeywordBinding(
            threshold_percent_setting, "threshold_percent", parse_cellprofiler_float
        )
    )
    metric_flag_setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        SettingToKeywordBinding(
            correlation_setting,
            "do_correlation",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            manders_setting,
            "do_manders",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            rank_weighted_setting,
            "do_rwc",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            overlap_setting,
            "do_overlap",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            costes_setting,
            "do_costes",
            parse_cellprofiler_bool,
        ),
    )
    metric_setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        threshold_percent_binding,
        *metric_flag_setting_bindings,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        save_mask_output_image_binding,
        save_mask_object_binding,
        *metric_setting_bindings,
        SettingToKeywordBinding(
            costes_method_setting,
            "costes_method",
            cellprofiler_enum_value_setting_parser(CostesMethod),
        ),
    )
    binds_without_declared_inputs = True

    @classmethod
    def threshold_mask_groups(
        cls,
        module: ModuleBlock,
    ) -> tuple[ColocalizationThresholdMaskGroup, ...]:
        """Parse every active save-mask group from ordered setting records."""

        save_literal = optional_setting_value(module, cls.save_masks_setting)
        if save_literal is None or not parse_cellprofiler_bool(save_literal):
            return ()
        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=cls.save_mask_source_image_setting,
        )
        count_literal = optional_setting_value(module, cls.save_mask_count_setting)
        if count_literal is not None:
            declared_count = int(float(count_literal))
            if len(blocks) != declared_count:
                raise ValueError(
                    "MeasureColocalization save-mask groups do not match their "
                    f"declared count: {len(blocks)} != {declared_count}."
                )
        thresholds = cls.channel_thresholds(module)
        default_threshold = float(
            optional_setting_value(module, cls.threshold_percent_setting) or 15.0
        )
        groups: list[ColocalizationThresholdMaskGroup] = []
        for group_index, block in enumerate(blocks):
            source_name = normalized_symbol_name(
                block_setting_value(block, cls.save_mask_source_image_setting)
            )
            output_name = normalized_symbol_name(
                block_setting_value(block, cls.save_mask_output_image_setting)
            )
            if source_name is None or output_name is None:
                raise ValueError(
                    "MeasureColocalization save-mask group "
                    f"{group_index + 1} requires exact source and output image names."
                )
            uses_objects_literal = block_setting_value(
                block,
                cls.save_mask_uses_objects_setting,
                default="No",
            )
            uses_objects = parse_cellprofiler_bool(uses_objects_literal)
            object_name = (
                normalized_symbol_name(
                    block_setting_value(block, cls.save_mask_object_setting)
                )
                if uses_objects
                else None
            )
            if uses_objects and object_name is None:
                raise ValueError(
                    "MeasureColocalization save-mask group "
                    f"{group_index + 1} enables object thresholding without an "
                    "object name."
                )
            groups.append(
                ColocalizationThresholdMaskGroup(
                    source_image_name=source_name,
                    output_image_name=output_name,
                    threshold_percent=(
                        thresholds[source_name]
                        if source_name in thresholds
                        else default_threshold
                    ),
                    object_name=object_name,
                )
            )
        return tuple(groups)

    @classmethod
    def channel_thresholds(cls, module: ModuleBlock) -> Mapping[str, float]:
        enabled_literal = optional_setting_value(
            module,
            cls.channel_thresholds_enabled_setting,
        )
        if enabled_literal is None or not parse_cellprofiler_bool(enabled_literal):
            return MappingProxyType({})
        blocks = repeating_setting_blocks(
            module.iter_settings(),
            start_name=cls.channel_threshold_image_setting,
        )
        count_literal = optional_setting_value(
            module,
            cls.channel_threshold_count_setting,
        )
        if count_literal is not None and len(blocks) != int(float(count_literal)):
            raise ValueError(
                "MeasureColocalization channel-threshold groups do not match "
                f"their declared count: {len(blocks)} != {count_literal}."
            )
        thresholds: dict[str, float] = {}
        for block in blocks:
            image_name = normalized_symbol_name(
                block_setting_value(block, cls.channel_threshold_image_setting)
            )
            if image_name is None:
                raise ValueError(
                    "MeasureColocalization channel threshold requires an image name."
                )
            if image_name in thresholds:
                raise ValueError(
                    "MeasureColocalization declares duplicate channel thresholds "
                    f"for {image_name!r}."
                )
            thresholds[image_name] = float(
                block_setting_value(block, cls.channel_threshold_percent_setting)
            )
        return MappingProxyType(thresholds)

    @classmethod
    def active_artifact_bindings(cls, module=None, *, invocation_key=None):
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        groups = cls.threshold_mask_groups(module)
        uses_objects = any(group.object_name is not None for group in groups)
        if uses_objects and cls.save_mask_object_binding not in bindings:
            bindings = (*bindings, cls.save_mask_object_binding)
        return tuple(
            binding
            for binding in bindings
            if groups or binding is not cls.save_mask_output_image_binding
            if uses_objects or binding is not cls.save_mask_object_binding
        )

    @classmethod
    def artifact_names_for_binding(cls, module, binding):
        if binding is cls.save_mask_output_image_binding:
            return tuple(
                group.output_image_name for group in cls.threshold_mask_groups(module)
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
        groups = cls.threshold_mask_groups(module)
        if (
            output_position >= len(groups)
            or groups[output_position].output_image_name != name
        ):
            raise ValueError(
                "MeasureColocalization output order does not match its save-mask "
                f"groups: position={output_position}, name={name!r}, groups={groups!r}."
            )
        group = groups[output_position]
        source = artifact_inputs.require_by_name_and_artifact_type(
            group.source_image_name,
            ImageArtifactType,
        )
        object_relation: tuple[ArtifactSpecRelation, ...] = ()
        if group.object_name is not None:
            object_source = artifact_inputs.require_by_name_and_artifact_type(
                group.object_name,
                ObjectLabelsArtifactType,
            )
            object_relation = (
                ColocalizationThresholdMaskObjectRelation(source=object_source.ref()),
            )
        return (
            ColocalizationThresholdMaskSourceRelation(
                source=source.ref(),
                threshold_percent=group.threshold_percent,
            ),
            *object_relation,
        )

    @classmethod
    def bind_runtime_inputs(cls, request: RuntimeInputBindingRequest):
        return {
            **super().bind_runtime_inputs(request),
            _ColocalizationThresholdMaskOutputsRuntimeParameter.require_parameter_name(): cls._runtime_threshold_mask_outputs(
                request,
                tuple(
                    request.adapter.request.require_callable_contract().artifact_outputs.of_artifact_type(
                        ImageArtifactType
                    )
                ),
            ),
        }

    @classmethod
    def _runtime_threshold_mask_outputs(
        cls,
        request: RuntimeInputBindingRequest,
        image_outputs,
    ) -> tuple[ColocalizationThresholdMaskRuntimeOutput, ...]:
        inputs = request.declared_inputs
        image_inputs = inputs.of_artifact_type(ImageArtifactType)
        channel_index_by_ref = {
            spec.ref(): channel_index for channel_index, spec in enumerate(image_inputs)
        }
        outputs: list[ColocalizationThresholdMaskRuntimeOutput] = []
        for output in image_outputs:
            source_relations = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, ColocalizationThresholdMaskSourceRelation)
            )
            object_relations = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, ColocalizationThresholdMaskObjectRelation)
            )
            if len(source_relations) != 1 or len(object_relations) > 1:
                raise ValueError(
                    "MeasureColocalization threshold-mask output requires one "
                    "source relation and at most one object relation, got "
                    f"{source_relations!r} and {object_relations!r}."
                )
            source_relation = source_relations[0]
            source_spec = inputs.by_ref(source_relation.source)
            if source_spec is None:
                raise ValueError(
                    "MeasureColocalization threshold-mask source is absent from "
                    "the declared inputs."
                )
            try:
                source_channel_index = channel_index_by_ref[source_relation.source]
            except KeyError as exc:
                raise ValueError(
                    "MeasureColocalization threshold-mask source is absent from "
                    f"the declared image inputs: {source_relation.source!r}."
                ) from exc
            object_labels = None
            object_name = None
            if object_relations:
                object_spec = inputs.by_ref(object_relations[0].source)
                if object_spec is None:
                    raise ValueError(
                        "MeasureColocalization threshold-mask object source is "
                        "absent from the declared inputs."
                    )
                object_name = object_spec.name
                object_labels = request.label_payload_for(object_spec)
            outputs.append(
                ColocalizationThresholdMaskRuntimeOutput(
                    group=ColocalizationThresholdMaskGroup(
                        source_image_name=source_spec.name,
                        output_image_name=output.name,
                        threshold_percent=source_relation.threshold_percent,
                        object_name=object_name,
                    ),
                    source_channel_index=source_channel_index,
                    object_labels=object_labels,
                )
            )
        return tuple(outputs)

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
        run_all_value = optional_setting_value(module, cls.run_all_metrics_setting)
        if run_all_value is not None:
            if cls.run_all_metrics_enabled(run_all_value, binder):
                kwargs.update(
                    {
                        binding.require_parameter_name(): True
                        for binding in cls.metric_flag_setting_bindings
                    }
                )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.run_all_metrics_setting), None
            )
        kwargs.pop(
            cls.save_mask_output_image_binding.require_parameter_name(),
            None,
        )
        threshold_mask_groups = cls.threshold_mask_groups(module)
        if threshold_mask_groups:
            kwargs["threshold_mask_groups"] = threshold_mask_groups
        bound = BoundModuleSettings(kwargs, unmapped_kwargs).with_consumed_settings(
            cls.channel_thresholds_enabled_setting,
            cls.channel_threshold_count_setting,
            cls.channel_threshold_image_setting,
            cls.channel_threshold_percent_setting,
            cls.save_masks_setting,
            cls.save_mask_count_setting,
            cls.save_mask_source_image_setting,
            cls.save_mask_uses_objects_setting,
            cls.save_mask_object_setting,
            cls.save_mask_output_image_setting,
        )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(module, bound),
        )

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation,
        step_context,
    ) -> tuple[ModuleBlock, ...]:
        """Reconstruct saved-mask rows from the public callable declaration."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        raw_groups = invocation.kwargs_dict.get("threshold_mask_groups", ())
        if not isinstance(raw_groups, (tuple, list)) or any(
            not isinstance(group, ColocalizationThresholdMaskGroup)
            for group in raw_groups
        ):
            raise TypeError(
                "MeasureColocalization threshold_mask_groups must contain only "
                "ColocalizationThresholdMaskGroup values."
            )
        groups = tuple(raw_groups)
        reconstructed = tuple(
            cls._block_with_threshold_mask_groups(block, groups) for block in blocks
        )
        return reconstructed

    @classmethod
    def _block_with_threshold_mask_groups(
        cls,
        block: ModuleBlock,
        groups: tuple[ColocalizationThresholdMaskGroup, ...],
    ) -> ModuleBlock:
        owned_settings = (
            cls.channel_thresholds_enabled_setting,
            cls.channel_threshold_count_setting,
            cls.channel_threshold_image_setting,
            cls.channel_threshold_percent_setting,
            cls.save_masks_setting,
            cls.save_mask_count_setting,
            cls.save_mask_source_image_setting,
            cls.save_mask_uses_objects_setting,
            cls.save_mask_object_setting,
            cls.save_mask_output_image_setting,
        )
        records = [
            record
            for record in block.iter_settings()
            if not any(
                setting_name_matches(record.name, setting) for setting in owned_settings
            )
        ]
        unique_thresholds: dict[str, float] = {}
        for group in groups:
            previous = unique_thresholds.get(group.source_image_name)
            if previous is not None and previous != group.threshold_percent:
                raise ValueError(
                    "MeasureColocalization public mask groups declare conflicting "
                    f"thresholds for {group.source_image_name!r}."
                )
            unique_thresholds[group.source_image_name] = group.threshold_percent
        records.extend(
            (
                ModuleSetting(
                    cls.channel_thresholds_enabled_setting,
                    "Yes" if unique_thresholds else "No",
                ),
                ModuleSetting(
                    cls.channel_threshold_count_setting,
                    str(len(unique_thresholds)),
                ),
            )
        )
        for source_name, threshold_percent in unique_thresholds.items():
            records.extend(
                (
                    ModuleSetting(cls.channel_threshold_image_setting, source_name),
                    ModuleSetting(
                        cls.channel_threshold_percent_setting,
                        str(threshold_percent),
                    ),
                )
            )
        records.extend(
            (
                ModuleSetting(cls.save_masks_setting, "Yes" if groups else "No"),
                ModuleSetting(cls.save_mask_count_setting, str(len(groups))),
            )
        )
        for group in groups:
            records.extend(
                (
                    ModuleSetting(
                        cls.save_mask_source_image_setting,
                        group.source_image_name,
                    ),
                    ModuleSetting(
                        cls.save_mask_uses_objects_setting,
                        "Yes" if group.object_name is not None else "No",
                    ),
                    *(
                        ()
                        if group.object_name is None
                        else (
                            ModuleSetting(
                                cls.save_mask_object_setting.canonical,
                                group.object_name,
                            ),
                        )
                    ),
                    ModuleSetting(
                        cls.save_mask_output_image_setting,
                        group.output_image_name,
                    ),
                )
            )
        return replace(
            block,
            setting_records=records,
        )

    @staticmethod
    def run_all_metrics_enabled(value: str, binder: "SettingsBinder") -> bool:
        normalized = value.strip().lower()
        if normalized in binder.BOOL_TRUE:
            return True
        if normalized in binder.BOOL_FALSE:
            return False
        return bool(value.strip())


logger = logging.getLogger(__name__)
runtime_profiler = CellProfilerRuntimeProfiler(logger)
_COLOCALIZATION_MEASUREMENT_FUNCTION = "_colocalization_measurement"
ColocalizationDenseLabelProjectionIdentity = tuple[tuple[str, Hashable], ...]


def _log_colocalization_measurement_phase(
    phase_name: str, started_at: float, **fields: object
) -> None:
    runtime_profiler.log(
        phase_name,
        time.perf_counter() - started_at,
        function=_COLOCALIZATION_MEASUREMENT_FUNCTION,
        **fields,
    )


class ColocalizationCostesBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Costes thresholding primitives keyed by OpenHCS memory/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def linear_costes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
        scale_max: int,
        fast_mode: bool,
    ) -> tuple[float, float]:
        """Return CellProfiler linear Costes thresholds."""

    @abstractmethod
    def scaled_second_channel_costes(
        self, first_pixels: np.ndarray, second_pixels: np.ndarray, scale_max: int
    ) -> tuple[float, float]:
        """Return CellProfiler scaled-bin second-channel Costes thresholds."""

    @abstractmethod
    def correlation_slopes(
        self, first_pixels: np.ndarray, second_pixels: np.ndarray
    ) -> tuple[float, float, float]:
        """Return Pearson correlation plus forward/reverse regression slopes."""


class NumbaNumpyColocalizationCostesBackendStrategy(
    ColocalizationCostesBackendStrategy
):
    """Numba implementation of Costes threshold searches."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def linear_costes(
        self,
        first_pixels: np.ndarray,
        second_pixels: np.ndarray,
        scale_max: int,
        fast_mode: bool,
    ) -> tuple[float, float]:
        first = np.ascontiguousarray(first_pixels, dtype=np.float64)
        second = np.ascontiguousarray(second_pixels, dtype=np.float64)
        valid, slope, intercept = _regression_line_numba(first, second)
        if not valid:
            return (0.0, 0.0)
        if slope > 0.0:
            event_threshold = np.minimum(first, (second - intercept) / slope)
            order = np.argsort(event_threshold)
            sorted_first = np.ascontiguousarray(first[order])
            sorted_second = np.ascontiguousarray(second[order])
            return _linear_costes_sorted_events_numba(
                (
                    np.ascontiguousarray(event_threshold[order]),
                    np.ascontiguousarray(np.cumsum(sorted_first)),
                    np.ascontiguousarray(np.cumsum(sorted_second)),
                    np.ascontiguousarray(np.cumsum(sorted_first * sorted_first)),
                    np.ascontiguousarray(np.cumsum(sorted_second * sorted_second)),
                    np.ascontiguousarray(np.cumsum(sorted_first * sorted_second)),
                    int(scale_max),
                    slope,
                    intercept,
                ),
                bool(fast_mode),
            )
        return _linear_costes_numba(first, second, int(scale_max), bool(fast_mode))

    def scaled_second_channel_costes(
        self, first_pixels: np.ndarray, second_pixels: np.ndarray, scale_max: int
    ) -> tuple[float, float]:
        first = np.ascontiguousarray(first_pixels, dtype=np.float64)
        second = np.ascontiguousarray(second_pixels, dtype=np.float64)
        non_zero = (first > 0.0) | (second > 0.0)
        first_non_zero = first[non_zero]
        second_non_zero = second[non_zero]
        first_variance = np.var(first_non_zero, axis=0, ddof=1)
        second_variance = np.var(second_non_zero, axis=0, ddof=1)
        first_mean = np.mean(first_non_zero, axis=0)
        second_mean = np.mean(second_non_zero, axis=0)
        summed_variance = np.var(
            first_non_zero + second_non_zero,
            axis=0,
            ddof=1,
        )
        covariance = 0.5 * (summed_variance - (first_variance + second_variance))
        variance_delta = second_variance - first_variance
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = (
                variance_delta
                + np.sqrt(
                    variance_delta * variance_delta + 4.0 * covariance * covariance
                )
            ) / (2.0 * covariance)
        intercept = second_mean - slope * first_mean

        scale = int(scale_max)
        left = 1
        right = scale
        mid = ((right - left) // (6 / 5)) + left
        last_mid = 0
        valid_mid = 1
        while last_mid != mid:
            first_threshold = mid / scale
            second_threshold = slope * first_threshold + intercept
            count, correlation = _pearson_below_threshold_numba(
                first,
                second,
                first_threshold,
                second_threshold,
            )
            if count <= 2:
                left = mid - 1
            elif correlation < 0.0:
                left = mid - 1
            elif correlation >= 0.0:
                right = mid + 1
                valid_mid = mid
            last_mid = mid
            if right - left > 6:
                mid = ((right - left) // (6 / 5)) + left
            else:
                mid = ((right - left) // 2) + left

        first_threshold = (valid_mid - 1) / scale
        second_threshold = slope * first_threshold + intercept
        return float(first_threshold), float(second_threshold)

    def correlation_slopes(
        self, first_pixels: np.ndarray, second_pixels: np.ndarray
    ) -> tuple[float, float, float]:
        correlation, slope, reverse_slope = _correlation_slopes_numba(
            np.ascontiguousarray(first_pixels, dtype=np.float64),
            np.ascontiguousarray(second_pixels, dtype=np.float64),
        )
        return float(correlation), float(slope), float(reverse_slope)

    def prepare_backend(self) -> None:
        """Compile numba Costes kernels outside measured execution."""
        first = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32)
        second = np.flipud(first.reshape((64, 64))).ravel().copy()
        _correlation_slopes_numba(
            np.ascontiguousarray(first, dtype=np.float64),
            np.ascontiguousarray(second, dtype=np.float64),
        )
        thresholded_colocalization_metrics(first, second, 15.0, True, True, True)
        self.linear_costes(first, second, 255, False)
        quantized_codes = np.arange(64 * 64, dtype=np.uint16) % 512 + 1024
        quantized = quantized_codes.astype(np.float32) / np.float32(65535)
        self.scaled_second_channel_costes(quantized, quantized.copy(), 255)


def costes_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> ColocalizationCostesBackendStrategy:
    """Resolve the explicit/default Costes backend for NumPy data."""
    return ColocalizationCostesBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


class CostesMethod(Enum):
    FASTER = "faster"
    FAST = "fast"
    ACCURATE = "accurate"


@dataclass(slots=True)
class ColocalizationMeasurements:
    """Colocalization measurements between two channels."""

    slice_index: int
    correlation: float
    slope: float
    overlap: float
    k1: float
    k2: float
    manders_m1: float
    manders_m2: float
    rwc1: float
    rwc2: float
    costes_m1: float
    costes_m2: float
    costes_threshold_1: float
    costes_threshold_2: float


def _colocalization_image_measurement_rows(
    measurement: ColocalizationMeasurements,
) -> DataclassMeasurementColumnarRows:
    """Return one exact image-scoped row from its nominal dataclass schema."""
    return DataclassMeasurementColumnarRows(
        (measurement,),
        row_type=ColocalizationMeasurements,
    )


class ColocalizationMeasurementSchema:
    """Derive object rows from the image schema and nominal feature scopes."""

    @staticmethod
    def object_measurement_type(
        measurement_type: type[ColocalizationMeasurements],
    ) -> type:
        measurement_fields = tuple(fields(measurement_type))
        excluded_fields = frozenset(
            feature.measurement_row_field_name
            for feature in MeasureColocalizationModule.MeasurementFeature
            if not feature.emitted_in_scope(MeasurementScope.OBJECT)
        )
        (slice_index_field,) = tuple(
            field
            for field in measurement_fields
            if field.name == MeasurementRowAxisField.SLICE_INDEX.value
        )
        row_fields = (
            (slice_index_field.name, slice_index_field.type),
            (MeasurementRowAxisField.OBJECT_LABEL.value, int),
            *(
                (field.name, field.type)
                for field in measurement_fields
                if field is not slice_index_field
                if field.name not in excluded_fields
            ),
        )
        return make_dataclass(
            "ObjectColocalizationMeasurements",
            row_fields,
            slots=True,
            namespace={
                "__module__": __name__,
                "__doc__": "Colocalization measurements scoped to one labeled object.",
            },
        )


ObjectColocalizationMeasurements = (
    ColocalizationMeasurementSchema.object_measurement_type(ColocalizationMeasurements)
)


@dataclass(frozen=True)
class ColocalizationMeasurementOptions:
    """Metric switches shared by image- and object-scoped colocalization."""

    threshold_percent: float
    do_correlation: bool
    do_manders: bool
    do_rwc: bool
    do_overlap: bool
    do_costes: bool
    costes_method: CostesMethod
    scale_max: int
    unit_interval_intensity_scale: int | None = None
    costes_backend_provider: BackendProviderInput = (
        DEFAULT_CELLPROFILER_BACKEND_SELECTION
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "costes_method", CostesMethod(self.costes_method))


@dataclass(frozen=True, slots=True)
class ColocalizationCostesThresholds:
    """Precomputed Costes thresholds for one resolved image source pair."""

    first: float
    second: float

    @classmethod
    def from_thresholds(
        cls, first: float, second: float
    ) -> "ColocalizationCostesThresholds":
        return cls(first=float(first), second=float(second))


@dataclass(frozen=True, slots=True)
class ColocalizationImagePairCacheKey:
    """Batch-local identity for one resolved colocalization image pair."""

    image_payload_id: int
    image_data_id: int
    channel_1: int
    channel_2: int


@dataclass(frozen=True, slots=True)
class ColocalizationObjectLabelCacheKey:
    """Batch-local identity for labels projected into one image-pair mask."""

    label_identity: ColocalizationDenseLabelProjectionIdentity | None
    label_data_id: int
    label_shape: tuple[int, ...]
    label_dtype: str
    pair_valid_mask_id: int
    measurement_shape: tuple[int, ...] | None
    slice_index: int

    @classmethod
    def from_dense_label_payload(
        cls,
        label_payload: object,
        label_array: np.ndarray,
        pair_valid_mask: np.ndarray | None,
        *,
        measurement_shape: tuple[int, ...] | None,
        slice_index: int = 0,
    ) -> "ColocalizationObjectLabelCacheKey":
        label_identity = (
            label_payload.object_label_dense_projection_identity()
            if isinstance(label_payload, ObjectLabelValue)
            else None
        )
        return cls(
            label_identity=label_identity,
            label_data_id=0 if label_identity is not None else id(label_array),
            label_shape=tuple(label_array.shape),
            label_dtype=np.dtype(label_array.dtype).str,
            pair_valid_mask_id=id(pair_valid_mask),
            measurement_shape=measurement_shape,
            slice_index=int(slice_index),
        )


@dataclass(frozen=True, slots=True)
class ColocalizationCostesThresholdCacheKey:
    """Batch-local identity for Costes thresholds over one image pair."""

    image_payload_id: int
    image_data_id: int
    channel_1: int
    channel_2: int
    method: CostesMethod
    scale_max: int
    backend_provider: object


@dataclass(frozen=True, slots=True)
class ColocalizationImagePairContext:
    """Resolved image-pair pixels shared by batched object colocalization calls."""

    image_data: np.ndarray
    image_float: np.ndarray
    first_image: np.ndarray
    second_image: np.ndarray
    pair_valid_mask: np.ndarray | None
    full_first_pixels: np.ndarray
    full_second_pixels: np.ndarray

    @staticmethod
    def valid_mask(
        image: object, image_data: np.ndarray, channel_1: int, channel_2: int
    ) -> np.ndarray | None:
        """Return CellProfiler-style valid pixels for a two-image measurement."""
        first_pixels = image_data[channel_1]
        second_pixels = image_data[channel_2]
        mask = image_payload_mask(image)
        if mask is None:
            if bool(np.all(np.isfinite(first_pixels))) and bool(
                np.all(np.isfinite(second_pixels))
            ):
                return None
            return np.isfinite(first_pixels) & np.isfinite(second_pixels)
        valid = np.isfinite(first_pixels) & np.isfinite(second_pixels)
        metadata = image_payload_metadata(image)
        if metadata.plane_axis is None:
            raise ValueError(
                "MeasureColocalization masked image pairs require a declared "
                "image plane axis."
            )
        projector = ImagePayloadSliceProjector(
            mask=mask,
            metadata=metadata,
        )
        first_mask = projector.mask_for_slice(first_pixels, channel_1)
        second_mask = projector.mask_for_slice(second_pixels, channel_2)
        if first_mask is None or second_mask is None:
            raise AssertionError(
                "Masked image plane projection must preserve its mask."
            )
        resolved_valid = (
            valid
            & np.asarray(first_mask, dtype=bool)
            & np.asarray(second_mask, dtype=bool)
        )
        if bool(np.all(resolved_valid)):
            return None
        return resolved_valid

    @classmethod
    def from_request(
        cls, image: object, *, channel_1: int, channel_2: int
    ) -> "ColocalizationImagePairContext":
        image_data = cls.measurement_pixels(image)
        image_float = np.asarray(image_data, dtype=np.float32)
        first_image = image_float[channel_1]
        second_image = image_float[channel_2]
        pair_valid_mask = cls.valid_mask(image, image_float, channel_1, channel_2)
        if pair_valid_mask is None:
            full_first_pixels = first_image.ravel()
            full_second_pixels = second_image.ravel()
        else:
            full_first_pixels = first_image[pair_valid_mask]
            full_second_pixels = second_image[pair_valid_mask]
        return cls(
            image_data=image_data,
            image_float=image_float,
            first_image=first_image,
            second_image=second_image,
            pair_valid_mask=pair_valid_mask,
            full_first_pixels=full_first_pixels,
            full_second_pixels=full_second_pixels,
        )

    @staticmethod
    def measurement_pixels(image: object) -> np.ndarray:
        """Return stacked image pixels for colocalization measurement."""
        image_data = image_payload_data(image)
        if isinstance(image_data, AlignedImageStack):
            return np.stack(
                tuple(
                    (
                        np.asarray(image_payload_data(slice_payload))
                        for slice_payload in image_data.slices
                    )
                ),
                axis=0,
            )
        return np.asarray(image_data)

    @classmethod
    def cellprofiler_float_pixels(cls, image: object) -> np.ndarray:
        """Return image pixels in CellProfiler's native float image domain."""
        return np.asarray(cls.measurement_pixels(image), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ColocalizationObjectLabelContext:
    """Resolved object-label reductions shared by batched image-pair calls."""

    labels: np.ndarray
    max_label: int
    label_range: np.ndarray
    object_mask: np.ndarray
    object_labels: np.ndarray
    object_counts: np.ndarray
    slice_index: int = 0

    @classmethod
    def from_labels(
        cls,
        labels: ObjectLabelValue,
        *,
        pair_valid_mask: np.ndarray | None,
        measurement_shape: tuple[int, ...] | None = None,
        slice_index: int = 0,
    ) -> "ColocalizationObjectLabelContext":
        return cls.from_dense_labels(
            object_label_dense_array(labels, dtype=np.int32),
            pair_valid_mask=pair_valid_mask,
            measurement_shape=measurement_shape,
            slice_index=slice_index,
        )

    @classmethod
    def from_dense_labels(
        cls,
        label_array: np.ndarray,
        *,
        pair_valid_mask: np.ndarray | None,
        measurement_shape: tuple[int, ...] | None = None,
        slice_index: int = 0,
    ) -> "ColocalizationObjectLabelContext":
        """Build reductions from an already-resolved dense label array."""
        label_array = np.asarray(label_array, dtype=np.int32)
        if label_array.ndim != 2:
            raise ValueError(
                "MeasureColocalization object labels must be projected to one "
                f"2-D plane, got shape {label_array.shape!r}."
            )
        if measurement_shape is not None and tuple(label_array.shape) != tuple(
            measurement_shape
        ):
            raise ValueError(
                "MeasureColocalization projected labels must match the image-pair "
                f"spatial domain; got labels {label_array.shape!r} and image "
                f"{measurement_shape!r}."
            )
        if pair_valid_mask is not None and pair_valid_mask.shape != label_array.shape:
            raise ValueError(
                "MeasureColocalization valid mask and projected labels must share "
                f"a shape; got mask {pair_valid_mask.shape!r} and labels "
                f"{label_array.shape!r}."
            )
        max_label = int(np.max(label_array)) if label_array.size else 0
        label_range = np.arange(1, max_label + 1, dtype=np.int32)
        object_mask = label_array > 0
        if pair_valid_mask is not None:
            object_mask = object_mask & pair_valid_mask
        object_labels = label_array[object_mask].astype(np.int32, copy=False)
        aggregation = DenseObjectLabelAggregation(
            labels=object_labels, object_count=max_label
        )
        return cls(
            labels=label_array,
            max_label=max_label,
            label_range=label_range,
            object_mask=object_mask,
            object_labels=object_labels,
            object_counts=aggregation.counts(),
            slice_index=int(slice_index),
        )


@dataclass(frozen=True, slots=True)
class ObjectColocalizationRequestContext:
    """Resolved object-colocalization request state shared by all metric stages."""

    image: object
    image_data: np.ndarray
    channel_1: int
    channel_2: int
    image_pair: ColocalizationImagePairContext
    labels: ColocalizationObjectLabelContext
    options: ColocalizationMeasurementOptions

    @property
    def has_labels(self) -> bool:
        return self.labels.max_label > 0

    @property
    def has_object_pixels(self) -> bool:
        return bool(self.labels.object_labels.size)


@dataclass(frozen=True, slots=True)
class ObjectColocalizationBaseStage:
    """Per-object base reductions used by all downstream object metrics."""

    first_pixels: np.ndarray
    second_pixels: np.ndarray
    object_labels: np.ndarray
    full_first_pixels: np.ndarray
    full_second_pixels: np.ndarray
    object_counts: np.ndarray
    sum1: np.ndarray
    sum2: np.ndarray
    sum1_sq: np.ndarray
    sum2_sq: np.ndarray
    product_sum: np.ndarray
    max1: np.ndarray
    max2: np.ndarray

    @classmethod
    def from_context(
        cls, context: ObjectColocalizationRequestContext
    ) -> "ObjectColocalizationBaseStage":
        labels = context.labels
        first_pixels = context.image_pair.first_image[labels.object_mask]
        second_pixels = context.image_pair.second_image[labels.object_mask]
        object_counts, sum1, sum2, sum1_sq, sum2_sq, product_sum, max1, max2 = (
            object_colocalization_base_reductions(
                first_pixels, second_pixels, labels.object_labels, labels.max_label
            )
        )
        return cls(
            first_pixels=first_pixels,
            second_pixels=second_pixels,
            object_labels=labels.object_labels,
            full_first_pixels=context.image_pair.full_first_pixels,
            full_second_pixels=context.image_pair.full_second_pixels,
            object_counts=object_counts,
            sum1=sum1,
            sum2=sum2,
            sum1_sq=sum1_sq,
            sum2_sq=sum2_sq,
            product_sum=product_sum,
            max1=max1,
            max2=max2,
        )


@dataclass(frozen=True, slots=True)
class ObjectColocalizationRankCacheKey:
    """Batch-local identity for one object-label/channel rank vector."""

    labels_id: int
    image_data_id: int
    channel_index: int
    scale_max: int
    unit_interval_intensity_scale: int | None


class ObjectColocalizationRankProvider(RuntimeSliceInvariantValue, ABC):
    """Resolve RWC dense ranks for one object-colocalization channel."""

    @abstractmethod
    def ranks(
        self,
        context: ObjectColocalizationRequestContext,
        pixels: np.ndarray,
        *,
        channel_index: int,
    ) -> np.ndarray:
        """Return dense CellProfiler RWC ranks for the supplied object pixels."""


class DirectObjectColocalizationRankProvider(ObjectColocalizationRankProvider):
    """Compute RWC ranks directly for one scalar colocalization call."""

    def ranks(
        self,
        context: ObjectColocalizationRequestContext,
        pixels: np.ndarray,
        *,
        channel_index: int,
    ) -> np.ndarray:
        del channel_index
        return UnitIntervalDenseRankSemantics.ranks(
            pixels,
            preferred_scale=context.options.scale_max,
            proven_unit_interval_scale=context.options.unit_interval_intensity_scale,
        )


_DIRECT_OBJECT_COLOCALIZATION_RANK_PROVIDER = DirectObjectColocalizationRankProvider()


class _ObjectColocalizationRankProviderRuntimeParameter(KeywordRuntimeParameter):
    """Batch-supplied rank provider for object colocalization."""

    parameter_name = "rank_provider"
    annotation_type = ObjectColocalizationRankProvider
    parameter_default = _DIRECT_OBJECT_COLOCALIZATION_RANK_PROVIDER


@dataclass(slots=True)
class CachedObjectColocalizationRankProvider(ObjectColocalizationRankProvider):
    """Share RWC ranks across source-pair calls in one measurement-image batch."""

    ranks_by_key: dict[ObjectColocalizationRankCacheKey, np.ndarray] = field(
        default_factory=dict
    )

    def ranks(
        self,
        context: ObjectColocalizationRequestContext,
        pixels: np.ndarray,
        *,
        channel_index: int,
    ) -> np.ndarray:
        key = ObjectColocalizationRankCacheKey(
            labels_id=id(context.labels),
            image_data_id=id(context.image_data),
            channel_index=int(channel_index),
            scale_max=int(context.options.scale_max),
            unit_interval_intensity_scale=context.options.unit_interval_intensity_scale,
        )
        ranks = self.ranks_by_key.get(key)
        if ranks is None:
            ranks = UnitIntervalDenseRankSemantics.ranks(
                pixels,
                preferred_scale=context.options.scale_max,
                proven_unit_interval_scale=context.options.unit_interval_intensity_scale,
            )
            self.ranks_by_key[key] = ranks
        return ranks


@dataclass
class ObjectColocalizationMetricArrays:
    """Mutable metric arrays populated by object-colocalization stages."""

    correlation: np.ndarray
    overlap: np.ndarray
    k1: np.ndarray
    k2: np.ndarray
    manders_m1: np.ndarray
    manders_m2: np.ndarray
    rwc1: np.ndarray
    rwc2: np.ndarray
    costes_m1: np.ndarray
    costes_m2: np.ndarray
    costes_threshold_1: np.ndarray
    costes_threshold_2: np.ndarray

    @classmethod
    def empty(cls, max_label: int) -> "ObjectColocalizationMetricArrays":
        axis_field_names = MeasurementRowAxisField.field_names()
        return cls(
            **{
                field_spec.name: np.zeros(max_label, dtype=float)
                for field_spec in FieldSpec.from_dataclass_type(
                    ObjectColocalizationMeasurements
                )
                if field_spec.name not in axis_field_names
            }
        )

    def rows_for(
        self, label_range: np.ndarray, *, slice_index: int = 0
    ) -> "ObjectColocalizationColumnarMeasurements":
        return ObjectColocalizationColumnarMeasurements(
            object_labels=np.asarray(label_range, dtype=np.int32),
            metrics=self,
            slice_index=int(slice_index),
        )

    def columns_for(
        self, object_labels: np.ndarray, *, slice_index: int = 0
    ) -> Mapping[str, np.ndarray]:
        metric_columns = {
            field_name: np.asarray(values, dtype=float)
            for field_name, values in vars(self).items()
        }
        declared_columns = {
            MeasurementRowAxisField.SLICE_INDEX.value: np.full(
                len(object_labels), int(slice_index), dtype=np.int32
            ),
            MeasurementRowAxisField.OBJECT_LABEL.value: object_labels,
            **metric_columns,
        }
        return MappingProxyType(
            {
                field_spec.name: declared_columns[field_spec.name]
                for field_spec in FieldSpec.from_dataclass_type(
                    ObjectColocalizationMeasurements
                )
            }
        )


@dataclass(frozen=True, slots=True)
class ObjectColocalizationColumnarMeasurements(ObjectMeasurementColumnarRows):
    """Columnar object-colocalization rows preserving direct row iteration."""

    fields: ClassVar[tuple[FieldSpec, ...]] = FieldSpec.from_dataclass_type(
        ObjectColocalizationMeasurements
    )
    object_labels: np.ndarray
    metrics: ObjectColocalizationMetricArrays
    slice_index: int = 0
    _columns: Mapping[str, np.ndarray] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        columns = self.metrics.columns_for(
            self.object_labels,
            slice_index=self.slice_index,
        )
        object.__setattr__(
            self,
            "_columns",
            columns,
        )
        self.validate_fields()

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        return self._columns

    def __len__(self) -> int:
        return len(self.object_labels)

    def __iter__(self):
        for row_index in range(len(self)):
            yield ObjectColocalizationMeasurements(
                **{
                    field_spec.name: self.columns[field_spec.name][row_index]
                    for field_spec in self.fields
                }
            )


@dataclass(frozen=True, slots=True)
class ObjectColocalizationThresholdStage:
    """Threshold masks and reductions for object Manders/RWC/overlap metrics."""

    threshold_1: np.ndarray
    threshold_2: np.ndarray
    threshold_counts: np.ndarray
    combined_threshold_has_values: bool
    total_first_threshold: np.ndarray
    total_second_threshold: np.ndarray
    threshold_sum1: np.ndarray
    threshold_sum2: np.ndarray
    threshold_sum1_sq: np.ndarray
    threshold_sum2_sq: np.ndarray
    threshold_product_sum: np.ndarray
    total_first_costes: np.ndarray
    total_second_costes: np.ndarray
    costes_sum1: np.ndarray
    costes_sum2: np.ndarray

    @classmethod
    def from_base(
        cls,
        context: ObjectColocalizationRequestContext,
        base: ObjectColocalizationBaseStage,
        costes_thresholds: ColocalizationCostesThresholds | None,
    ) -> "ObjectColocalizationThresholdStage":
        max_label = context.labels.max_label
        options = context.options
        threshold_metrics_requested = any(
            (options.do_manders, options.do_rwc, options.do_overlap)
        )
        if threshold_metrics_requested:
            first_threshold_scale = np.asarray(
                options.threshold_percent / 100,
                dtype=base.first_pixels.dtype,
            )
            second_threshold_scale = np.asarray(
                options.threshold_percent / 100,
                dtype=base.second_pixels.dtype,
            )
            threshold_1 = first_threshold_scale * np.asarray(
                base.max1,
                dtype=base.first_pixels.dtype,
            )
            threshold_2 = second_threshold_scale * np.asarray(
                base.max2,
                dtype=base.second_pixels.dtype,
            )
        else:
            threshold_1 = np.zeros(max_label, dtype=float)
            threshold_2 = np.zeros(max_label, dtype=float)
        threshold_reductions_requested = threshold_metrics_requested or (
            options.do_costes and base.full_first_pixels.size
        )
        if threshold_reductions_requested:
            first_costes_threshold = (
                _pixel_dtype_threshold(base.first_pixels, costes_thresholds.first)
                if costes_thresholds is not None
                else 0.0
            )
            second_costes_threshold = (
                _pixel_dtype_threshold(base.second_pixels, costes_thresholds.second)
                if costes_thresholds is not None
                else 0.0
            )
            (
                total_first_threshold,
                total_second_threshold,
                threshold_sum1,
                threshold_sum2,
                threshold_sum1_sq,
                threshold_sum2_sq,
                threshold_product_sum,
                threshold_counts,
                total_first_costes,
                total_second_costes,
                costes_sum1,
                costes_sum2,
            ) = object_colocalization_threshold_reductions(
                base.first_pixels,
                base.second_pixels,
                base.object_labels,
                threshold_1,
                threshold_2,
                first_costes_threshold,
                second_costes_threshold,
                max_label,
            )
        else:
            empty = np.zeros(max_label, dtype=float)
            total_first_threshold = empty
            total_second_threshold = empty.copy()
            threshold_sum1 = empty.copy()
            threshold_sum2 = empty.copy()
            threshold_sum1_sq = empty.copy()
            threshold_sum2_sq = empty.copy()
            threshold_product_sum = empty.copy()
            threshold_counts = empty.copy()
            total_first_costes = empty.copy()
            total_second_costes = empty.copy()
            costes_sum1 = empty.copy()
            costes_sum2 = empty.copy()
        return cls(
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            threshold_counts=threshold_counts,
            combined_threshold_has_values=bool(np.any(threshold_counts > 0.0)),
            total_first_threshold=total_first_threshold,
            total_second_threshold=total_second_threshold,
            threshold_sum1=threshold_sum1,
            threshold_sum2=threshold_sum2,
            threshold_sum1_sq=threshold_sum1_sq,
            threshold_sum2_sq=threshold_sum2_sq,
            threshold_product_sum=threshold_product_sum,
            total_first_costes=total_first_costes,
            total_second_costes=total_second_costes,
            costes_sum1=costes_sum1,
            costes_sum2=costes_sum2,
        )


def _prepare_object_colocalization_context(
    image: object,
    labels: ObjectLabelValue,
    *,
    channel_1: int,
    channel_2: int,
    threshold_percent: float,
    do_correlation: bool,
    do_manders: bool,
    do_rwc: bool,
    do_overlap: bool,
    do_costes: bool,
    costes_method: CostesMethod,
    scale_max: int | None,
    costes_backend_provider: BackendProviderInput,
    image_pair_context: ColocalizationImagePairContext | None,
    object_label_context: ColocalizationObjectLabelContext | None,
) -> ObjectColocalizationRequestContext:
    if image_pair_context is None:
        image_pair_context = ColocalizationImagePairContext.from_request(
            image, channel_1=channel_1, channel_2=channel_2
        )
    image_data = image_pair_context.image_data
    if object_label_context is None:
        object_label_context = ColocalizationObjectLabelContext.from_labels(
            labels,
            pair_valid_mask=image_pair_context.pair_valid_mask,
            measurement_shape=tuple(image_pair_context.first_image.shape),
        )
    options = ColocalizationMeasurementOptions(
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=ColocalizationCostesThresholdRequest.scale_max_for_image_pair(
            image, image_data, channel_1, channel_2, scale_max
        ),
        costes_backend_provider=costes_backend_provider,
    )
    return ObjectColocalizationRequestContext(
        image=image,
        image_data=image_data,
        channel_1=channel_1,
        channel_2=channel_2,
        image_pair=image_pair_context,
        labels=object_label_context,
        options=options,
    )


def _empty_object_colocalization_rows(
    label_range: np.ndarray, *, slice_index: int = 0
) -> ObjectColocalizationColumnarMeasurements:
    return ObjectColocalizationMetricArrays.empty(len(label_range)).rows_for(
        label_range, slice_index=slice_index
    )


def _resolve_object_costes_thresholds(
    context: ObjectColocalizationRequestContext,
    base: ObjectColocalizationBaseStage,
    provided: ColocalizationCostesThresholds | None,
    metrics: ObjectColocalizationMetricArrays,
) -> ColocalizationCostesThresholds | None:
    options = context.options
    if not (options.do_costes and base.full_first_pixels.size):
        return None
    if provided is not None:
        resolved = provided
    elif options.costes_method == CostesMethod.FASTER:
        threshold_c1, threshold_c2 = costes_backend(
            backend_provider=options.costes_backend_provider
        ).scaled_second_channel_costes(
            base.full_first_pixels, base.full_second_pixels, options.scale_max
        )
        resolved = ColocalizationCostesThresholds.from_thresholds(
            threshold_c1, threshold_c2
        )
    else:
        threshold_c1, threshold_c2 = costes_backend(
            backend_provider=options.costes_backend_provider
        ).linear_costes(
            base.full_first_pixels,
            base.full_second_pixels,
            options.scale_max,
            options.costes_method == CostesMethod.FAST,
        )
        resolved = ColocalizationCostesThresholds.from_thresholds(
            threshold_c1, threshold_c2
        )
    metrics.costes_threshold_1.fill(resolved.first)
    metrics.costes_threshold_2.fill(resolved.second)
    return resolved


def _populate_object_correlation_metrics(
    options: ColocalizationMeasurementOptions,
    base: ObjectColocalizationBaseStage,
    metrics: ObjectColocalizationMetricArrays,
) -> None:
    if not options.do_correlation:
        return
    metrics.correlation = object_colocalization_correlation_reductions(
        base.first_pixels,
        base.second_pixels,
        base.object_labels,
        base.object_counts,
        base.sum1,
        base.sum2,
        len(base.object_counts),
    )
    metrics.correlation[~np.isfinite(metrics.correlation)] = 0.0


def _populate_object_threshold_metrics(
    context: ObjectColocalizationRequestContext,
    base: ObjectColocalizationBaseStage,
    threshold: ObjectColocalizationThresholdStage,
    metrics: ObjectColocalizationMetricArrays,
    rank_provider: ObjectColocalizationRankProvider,
) -> None:
    options = context.options
    if options.do_manders and threshold.combined_threshold_has_values:
        metrics.manders_m1 = _divide_measurements(
            threshold.threshold_sum1, threshold.total_first_threshold
        )
        metrics.manders_m2 = _divide_measurements(
            threshold.threshold_sum2, threshold.total_second_threshold
        )
    if options.do_rwc:
        rank_image_1 = rank_provider.ranks(
            context, base.first_pixels, channel_index=context.channel_1
        )
        rank_image_2 = rank_provider.ranks(
            context, base.second_pixels, channel_index=context.channel_2
        )
        max_rank = max(rank_image_1.max(), rank_image_2.max()) + 1
        if threshold.combined_threshold_has_values:
            weighted_first, weighted_second = object_colocalization_rwc_reductions(
                base.first_pixels,
                base.second_pixels,
                base.object_labels,
                threshold.threshold_1,
                threshold.threshold_2,
                rank_image_1,
                rank_image_2,
                int(max_rank),
                len(threshold.threshold_1),
            )
            metrics.rwc1 = _divide_measurements(
                weighted_first, threshold.total_first_threshold
            )
            metrics.rwc2 = _divide_measurements(
                weighted_second, threshold.total_second_threshold
            )
    if options.do_overlap and threshold.combined_threshold_has_values:
        metrics.overlap = _divide_measurements(
            threshold.threshold_product_sum,
            np.sqrt(threshold.threshold_sum1_sq * threshold.threshold_sum2_sq),
        )
        metrics.k1 = _divide_measurements(
            threshold.threshold_product_sum, threshold.threshold_sum1_sq
        )
        metrics.k2 = _divide_measurements(
            threshold.threshold_product_sum, threshold.threshold_sum2_sq
        )


def _populate_object_costes_metrics(
    options: ColocalizationMeasurementOptions,
    base: ObjectColocalizationBaseStage,
    threshold: ObjectColocalizationThresholdStage,
    metrics: ObjectColocalizationMetricArrays,
) -> None:
    if not (options.do_costes and base.full_first_pixels.size):
        return
    metrics.costes_m1 = _divide_costes_measurements(
        threshold.costes_sum1, threshold.total_first_costes
    )
    metrics.costes_m2 = _divide_costes_measurements(
        threshold.costes_sum2, threshold.total_second_costes
    )


def _colocalization_measurement(
    first_pixels: np.ndarray,
    second_pixels: np.ndarray,
    *,
    options: ColocalizationMeasurementOptions,
    valid_mask: np.ndarray | None = None,
) -> ColocalizationMeasurements:
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    corr = np.nan
    slope = np.nan
    overlap = np.nan
    k1 = np.nan
    k2 = np.nan
    m1 = np.nan
    m2 = np.nan
    rwc1 = np.nan
    rwc2 = np.nan
    c1 = np.nan
    c2 = np.nan
    thr_fi_c = np.nan
    thr_si_c = np.nan
    if valid_mask is None:
        first_array = np.asarray(first_pixels)
        second_array = np.asarray(second_pixels)
        finite_mask = np.isfinite(first_array) & np.isfinite(second_array)
        if np.any(finite_mask):
            if bool(np.all(finite_mask)):
                fi = np.ravel(first_array)
                si = np.ravel(second_array)
            else:
                fi = first_array[finite_mask]
                si = second_array[finite_mask]
        else:
            fi = np.empty(0, dtype=np.asarray(first_pixels).dtype)
            si = np.empty(0, dtype=np.asarray(second_pixels).dtype)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if np.any(mask):
            fi = first_pixels[mask]
            si = second_pixels[mask]
        else:
            fi = np.empty(0, dtype=np.asarray(first_pixels).dtype)
            si = np.empty(0, dtype=np.asarray(second_pixels).dtype)
    _log_colocalization_measurement_phase(
        "coloc_prepare_pixels", phase_started_at, pixels=fi.size
    )
    if fi.size:
        if options.do_correlation:
            phase_started_at = time.perf_counter()
            corr, slope, _ = ColocalizationCostesBackendStrategy.for_memory_type(
                backend_provider=options.costes_backend_provider
            ).correlation_slopes(fi, si)
            _log_colocalization_measurement_phase("coloc_correlation", phase_started_at)
        if any((options.do_manders, options.do_rwc, options.do_overlap)):
            phase_started_at = time.perf_counter()
            m1, m2, rwc1, rwc2, overlap, k1, k2 = thresholded_colocalization_metrics(
                np.ascontiguousarray(fi),
                np.ascontiguousarray(si),
                float(options.threshold_percent),
                bool(options.do_manders),
                bool(options.do_rwc),
                bool(options.do_overlap),
                int(options.scale_max),
                options.unit_interval_intensity_scale,
            )
            _log_colocalization_measurement_phase(
                "coloc_thresholded_metrics", phase_started_at
            )
        if options.do_costes:
            phase_started_at = time.perf_counter()
            if options.costes_method == CostesMethod.FASTER:
                thr_fi_c, thr_si_c = costes_backend(
                    backend_provider=options.costes_backend_provider
                ).scaled_second_channel_costes(fi, si, options.scale_max)
            else:
                fast_mode = options.costes_method == CostesMethod.FAST
                thr_fi_c, thr_si_c = costes_backend(
                    backend_provider=options.costes_backend_provider
                ).linear_costes(fi, si, options.scale_max, fast_mode)
            _log_colocalization_measurement_phase(
                "coloc_costes_thresholds",
                phase_started_at,
                method=options.costes_method.value,
            )
            phase_started_at = time.perf_counter()
            c1, c2 = _costes_manders_numba(
                np.ascontiguousarray(fi),
                np.ascontiguousarray(si),
                _pixel_dtype_threshold(fi, thr_fi_c),
                _pixel_dtype_threshold(si, thr_si_c),
            )
            _log_colocalization_measurement_phase(
                "coloc_costes_manders", phase_started_at
            )
    result = ColocalizationMeasurements(
        slice_index=0,
        correlation=float(corr) if not np.isnan(corr) else 0.0,
        slope=float(slope) if not np.isnan(slope) else 0.0,
        overlap=float(overlap) if not np.isnan(overlap) else 0.0,
        k1=float(k1) if not np.isnan(k1) else 0.0,
        k2=float(k2) if not np.isnan(k2) else 0.0,
        manders_m1=float(m1) if not np.isnan(m1) else 0.0,
        manders_m2=float(m2) if not np.isnan(m2) else 0.0,
        rwc1=float(rwc1) if not np.isnan(rwc1) else 0.0,
        rwc2=float(rwc2) if not np.isnan(rwc2) else 0.0,
        costes_m1=float(c1) if not np.isnan(c1) else 0.0,
        costes_m2=float(c2) if not np.isnan(c2) else 0.0,
        costes_threshold_1=float(thr_fi_c) if not np.isnan(thr_fi_c) else 0.0,
        costes_threshold_2=float(thr_si_c) if not np.isnan(thr_si_c) else 0.0,
    )
    _log_colocalization_measurement_phase("coloc_total", total_started_at)
    return result


def _pixel_dtype_threshold(pixels: np.ndarray, threshold: float) -> float:
    """Round scalar thresholds into the pixel dtype before bin comparisons."""
    return float(np.asarray(threshold, dtype=np.asarray(pixels).dtype).item())


def _cellprofiler_float_pixels(image: np.ndarray) -> np.ndarray:
    """Return image pixels in CellProfiler's native float image domain."""
    return ColocalizationImagePairContext.cellprofiler_float_pixels(image)


def _colocalization_unit_interval_scale(
    image: object, channel_1: int, channel_2: int
) -> int | None:
    """Return a shared proof scale when both channels are exact unit interval."""
    metadata = image_payload_metadata(image)
    first_scale = metadata.unit_interval_intensity_scale_for_source_plane(channel_1)
    second_scale = metadata.unit_interval_intensity_scale_for_source_plane(channel_2)
    if first_scale is None or second_scale is None:
        return None
    if int(first_scale) != int(second_scale):
        return None
    return int(first_scale)


@required_variable_components(VariableComponents.CHANNEL)
@composed_image_payload
@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.FLEXIBLE)
@runtime_bound_parameters(_ColocalizationThresholdMaskOutputsRuntimeParameter)
def measure_colocalization(
    image: np.ndarray,
    channel_1: int = 0,
    channel_2: int = 1,
    threshold_percent: float = 15.0,
    do_correlation: bool = True,
    do_manders: bool = True,
    do_rwc: bool = True,
    do_overlap: bool = True,
    do_costes: bool = True,
    costes_method: CostesMethod = CostesMethod.FASTER,
    scale_max: int | None = None,
    threshold_mask_groups: ColocalizationThresholdMaskGroupsInput = (),
    threshold_mask_outputs: tuple[ColocalizationThresholdMaskRuntimeOutput, ...] = (),
    costes_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[RuntimeArrayData | AlignedImageStack, MeasurementProjectedColumnarRows]:
    """
    Measure colocalization between two channels from an N-channel image.

    Args:
        image: Shape (N, H, W) - N channel images stacked along dim 0
        channel_1: Index of first channel to compare (default 0)
        channel_2: Index of second channel to compare (default 1)
        threshold_percent: Threshold as percentage of max intensity (0-99)
        do_correlation: Calculate Pearson correlation and slope
        do_manders: Calculate Manders coefficients
        do_rwc: Calculate Rank Weighted Colocalization coefficients
        do_overlap: Calculate Overlap coefficients
        do_costes: Calculate Manders coefficients using Costes auto threshold
        costes_method: Method for Costes thresholding (faster, fast, accurate)
        scale_max: Optional explicit maximum scale for Costes calculation. When
            omitted, OpenHCS resolves it from generic source image metadata.
        costes_backend_provider: Optional explicit Costes backend provider.

    Returns:
        Tuple of (first channel image, exact image-scoped columnar rows)

    CellProfiler Parameter Mapping:
    (CellProfiler setting -> Python parameter)
        'Select images to measure' -> (pipeline-handled)
        'Set threshold as percentage of maximum intensity for the images' -> threshold_percent
        'Run all metrics?' -> (pipeline-handled)
        'Calculate correlation and slope metrics?' -> do_correlation
        'Calculate the Manders coefficients?' -> do_manders
        'Calculate the Rank Weighted Colocalization coefficients?' -> do_rwc
        'Calculate the Overlap coefficients?' -> do_overlap
        'Calculate the Manders coefficients using Costes auto threshold?' -> do_costes
        'Method for Costes thresholding' -> costes_method"""
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    image_data = image_payload_data(image)
    if channel_1 >= image_data.shape[0] or channel_2 >= image_data.shape[0]:
        raise ValueError(
            f"Channel indices ({channel_1}, {channel_2}) out of range for image with {image_data.shape[0]} channels"
        )
    runtime_profiler.log(
        "measure_coloc_input",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )
    phase_started_at = time.perf_counter()
    options = ColocalizationMeasurementOptions(
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=ColocalizationCostesThresholdRequest.scale_max_for_image_pair(
            image, image_data, channel_1, channel_2, scale_max
        ),
        unit_interval_intensity_scale=_colocalization_unit_interval_scale(
            image, channel_1, channel_2
        ),
        costes_backend_provider=costes_backend_provider,
    )
    runtime_profiler.log(
        "measure_coloc_options",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
        scale_max=options.scale_max,
        source_plane_intensity_scales=(
            image_payload_metadata(image).source_plane_intensity_scales
        ),
        intensity_scale=image_payload_metadata(image).intensity_scale,
        payload_type=type(image).__name__,
    )
    phase_started_at = time.perf_counter()
    image_float = _cellprofiler_float_pixels(image_data)
    valid_mask = ColocalizationImagePairContext.valid_mask(
        image, image_float, channel_1, channel_2
    )
    runtime_profiler.log(
        "measure_coloc_prepare_arrays",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
        full_valid=valid_mask is None,
    )
    phase_started_at = time.perf_counter()
    measurements = _colocalization_measurement(
        image_float[channel_1],
        image_float[channel_2],
        options=options,
        valid_mask=valid_mask,
    )
    runtime_profiler.log(
        "measure_coloc_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )
    phase_started_at = time.perf_counter()
    output = _colocalization_threshold_mask_canonical_output(
        image,
        threshold_mask_groups=threshold_mask_groups,
        threshold_mask_outputs=threshold_mask_outputs,
        fallback_channel_index=channel_1,
    )
    runtime_profiler.log(
        "measure_coloc_output_payload",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization",
    )
    runtime_profiler.log(
        "measure_coloc_total",
        time.perf_counter() - total_started_at,
        function="measure_colocalization",
    )
    return (output, _colocalization_image_measurement_rows(measurements))


def _measure_colocalization_objects_core(
    context: ObjectColocalizationRequestContext,
    *,
    costes_thresholds: ColocalizationCostesThresholds | None = None,
    rank_provider: ObjectColocalizationRankProvider = DirectObjectColocalizationRankProvider(),
) -> Tuple[RuntimeArrayData, ObjectColocalizationColumnarMeasurements]:
    """Measure colocalization between two channels within labeled objects."""
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
    runtime_profiler.log(
        "coloc_object_prepare_context",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
        objects=context.labels.max_label,
        object_pixels=int(context.labels.object_labels.size),
    )
    if not context.has_labels:
        return (
            image_payload_metadata(context.image).project_channel_payload(
                context.image, context.image_data, context.channel_1
            ),
            _empty_object_colocalization_rows(
                context.labels.label_range,
                slice_index=context.labels.slice_index,
            ),
        )
    if not context.has_object_pixels:
        return (
            image_payload_metadata(context.image).project_channel_payload(
                context.image, context.image_data, context.channel_1
            ),
            _empty_object_colocalization_rows(
                context.labels.label_range, slice_index=context.labels.slice_index
            ),
        )
    phase_started_at = time.perf_counter()
    base = ObjectColocalizationBaseStage.from_context(context)
    runtime_profiler.log(
        "coloc_object_base_reductions",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
        pixels=int(base.object_labels.size),
    )
    phase_started_at = time.perf_counter()
    metrics = ObjectColocalizationMetricArrays.empty(context.labels.max_label)
    _populate_object_correlation_metrics(context.options, base, metrics)
    runtime_profiler.log(
        "coloc_object_correlation_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
    )
    phase_started_at = time.perf_counter()
    resolved_costes_thresholds = _resolve_object_costes_thresholds(
        context, base, costes_thresholds, metrics
    )
    runtime_profiler.log(
        "coloc_object_costes_thresholds",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
        provided=costes_thresholds is not None,
    )
    phase_started_at = time.perf_counter()
    threshold = ObjectColocalizationThresholdStage.from_base(
        context, base, resolved_costes_thresholds
    )
    runtime_profiler.log(
        "coloc_object_threshold_stage",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
        thresholds=threshold.combined_threshold_has_values,
    )
    phase_started_at = time.perf_counter()
    _populate_object_threshold_metrics(context, base, threshold, metrics, rank_provider)
    runtime_profiler.log(
        "coloc_object_threshold_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
    )
    phase_started_at = time.perf_counter()
    _populate_object_costes_metrics(context.options, base, threshold, metrics)
    runtime_profiler.log(
        "coloc_object_costes_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
    )
    phase_started_at = time.perf_counter()
    rows = metrics.rows_for(
        context.labels.label_range, slice_index=context.labels.slice_index
    )
    runtime_profiler.log(
        "coloc_object_rows",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
        rows=len(rows),
    )
    runtime_profiler.log(
        "coloc_object_total",
        time.perf_counter() - total_started_at,
        function="measure_colocalization_objects",
        channel_1=context.channel_1,
        channel_2=context.channel_2,
    )
    return (
        image_payload_metadata(context.image).project_channel_payload(
            context.image, context.image_data, context.channel_1
        ),
        rows,
    )


@required_variable_components(VariableComponents.CHANNEL)
@composed_image_payload
@numpy(contract=ProcessingContract.FLEXIBLE)
@special_inputs("labels")
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.SLICE_ALIGNED)
@runtime_bound_parameters(
    _ObjectColocalizationRankProviderRuntimeParameter,
    _ColocalizationThresholdMaskOutputsRuntimeParameter,
)
def measure_colocalization_objects(
    image: np.ndarray,
    labels: ObjectLabelValue,
    measurement_scope: CellProfilerMeasurementTargetScope = CellProfilerMeasurementTargetScope.OBJECT,
    channel_1: int = 0,
    channel_2: int = 1,
    threshold_percent: float = 15.0,
    do_correlation: bool = True,
    do_manders: bool = True,
    do_rwc: bool = True,
    do_overlap: bool = True,
    do_costes: bool = True,
    costes_method: CostesMethod = CostesMethod.FASTER,
    scale_max: int | None = None,
    costes_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    costes_thresholds: ColocalizationCostesThresholds | None = None,
    image_pair_context: ColocalizationImagePairContext | None = None,
    object_label_context: ColocalizationObjectLabelContext | None = None,
    threshold_mask_groups: ColocalizationThresholdMaskGroupsInput = (),
    threshold_mask_outputs: tuple[ColocalizationThresholdMaskRuntimeOutput, ...] = (),
    *,
    rank_provider: ObjectColocalizationRankProvider = _DIRECT_OBJECT_COLOCALIZATION_RANK_PROVIDER,
) -> Tuple[RuntimeArrayData | AlignedImageStack, ColumnarRows]:
    """Measure image and/or object colocalization through one declared callable.

    Args:
        labels: Object-label plane defining the regions that receive separate
            colocalization measurements.
        channel_1: Zero-based index of the first image channel to compare.
        channel_2: Zero-based index of the second image channel to compare.
        scale_max: Maximum integer intensity scale used by the Costes threshold
            search; omit it to infer the scale from source-image metadata.
        costes_thresholds: Precomputed Costes thresholds for this image pair;
            leave unset to calculate them from the selected channels.
        image_pair_context: Prepared selected-channel pixels and validity mask for
            repeated object measurements; leave unset for ordinary calls.
        object_label_context: Prepared label reductions aligned to the selected
            image pair; leave unset for ordinary calls.
    """
    target_scope = coerce_cellprofiler_measurement_target_scope(
        measurement_scope,
        CellProfilerMeasurementTargetScope.OBJECT,
    ).measurement_scope_selection
    context = _prepare_object_colocalization_context(
        image,
        labels,
        channel_1=channel_1,
        channel_2=channel_2,
        threshold_percent=threshold_percent,
        do_correlation=do_correlation,
        do_manders=do_manders,
        do_rwc=do_rwc,
        do_overlap=do_overlap,
        do_costes=do_costes,
        costes_method=costes_method,
        scale_max=scale_max,
        costes_backend_provider=costes_backend_provider,
        image_pair_context=image_pair_context,
        object_label_context=object_label_context,
    )
    output, object_rows = _measure_colocalization_objects_core(
        context,
        costes_thresholds=costes_thresholds,
        rank_provider=rank_provider,
    )
    output = _colocalization_threshold_mask_canonical_output(
        image,
        threshold_mask_groups=threshold_mask_groups,
        threshold_mask_outputs=threshold_mask_outputs,
        fallback_channel_index=channel_1,
    )
    if not target_scope.includes(MeasurementScope.IMAGE):
        return (output, object_rows)
    image_row = _colocalization_measurement(
        context.image_pair.first_image,
        context.image_pair.second_image,
        options=context.options,
        valid_mask=context.image_pair.pair_valid_mask,
    )
    image_rows = _colocalization_image_measurement_rows(image_row)
    if not target_scope.includes(MeasurementScope.OBJECT):
        return (output, image_rows)
    return (output, ConcatenatedColumnarRows((image_rows, object_rows)))


def _colocalization_threshold_mask_canonical_output(
    image: np.ndarray,
    *,
    threshold_mask_groups: tuple[ColocalizationThresholdMaskGroup, ...],
    threshold_mask_outputs: tuple[ColocalizationThresholdMaskRuntimeOutput, ...],
    fallback_channel_index: int,
) -> RuntimeArrayData | AlignedImageStack:
    """Return saved masks in contract order or the ordinary fallback image."""

    runtime_groups = tuple(request.group for request in threshold_mask_outputs)
    if threshold_mask_groups != runtime_groups:
        raise ValueError(
            "MeasureColocalization runtime mask outputs do not match the public "
            f"saved-mask groups: {runtime_groups!r} != {threshold_mask_groups!r}."
        )
    image_data = np.asarray(image_payload_data(image))
    if not threshold_mask_outputs:
        return image_payload_metadata(image).project_channel_payload(
            image,
            image_data,
            fallback_channel_index,
        )
    return pack_aligned_image_outputs(
        tuple(
            _colocalization_threshold_mask(image, request)
            for request in threshold_mask_outputs
        )
    )


def _colocalization_threshold_mask(
    image: np.ndarray,
    request: ColocalizationThresholdMaskRuntimeOutput,
) -> RuntimeArrayData:
    """Apply CellProfiler's whole-image or per-object percentage threshold."""

    image_data = np.asarray(image_payload_data(image))
    if image_data.ndim < 3:
        raise ValueError(
            "MeasureColocalization saved masks require a composed channel image, "
            f"got shape {image_data.shape!r}."
        )
    if not 0 <= request.source_channel_index < image_data.shape[0]:
        raise ValueError(
            "MeasureColocalization saved-mask source channel is outside the "
            f"composed image: {request.source_channel_index} for "
            f"{image_data.shape[0]} channels."
        )
    pixels = np.asarray(image_data[request.source_channel_index])
    valid_mask = np.isfinite(pixels)
    payload_mask = image_payload_mask(image)
    if payload_mask is not None:
        mask_array = np.asarray(payload_mask, dtype=bool)
        if mask_array.shape == image_data.shape:
            valid_mask &= mask_array[request.source_channel_index]
        elif mask_array.shape == pixels.shape:
            valid_mask &= mask_array
        else:
            raise ValueError(
                "MeasureColocalization saved-mask image mask does not match its "
                f"source plane: {mask_array.shape!r} for {pixels.shape!r}."
            )
    output = np.zeros(pixels.shape, dtype=bool)
    if np.any(valid_mask):
        threshold_fraction = float(request.group.threshold_percent) / 100.0
        if request.object_labels is None:
            threshold = threshold_fraction * float(np.max(pixels[valid_mask]))
            output[valid_mask] = pixels[valid_mask] > threshold
        else:
            labels = object_label_dense_array(request.object_labels, dtype=np.int32)
            if labels.shape != pixels.shape:
                raise ValueError(
                    "MeasureColocalization saved-mask object labels must match the "
                    f"source image plane: {labels.shape!r} != {pixels.shape!r}."
                )
            for object_id in np.unique(labels[valid_mask & (labels > 0)]):
                object_mask = valid_mask & (labels == object_id)
                threshold = threshold_fraction * float(np.max(pixels[object_mask]))
                output[object_mask] = pixels[object_mask] >= threshold
    return _colocalization_source_plane_output(
        image,
        image_data,
        request.source_channel_index,
        output,
    )


def _colocalization_source_plane_output(
    image: RuntimeArrayData,
    image_data: np.ndarray,
    source_channel_index: int,
    output_data: np.ndarray,
) -> RuntimeArrayData:
    """Attach one declared composed-image source plane to a derived output."""

    metadata = image_payload_metadata(image)
    if metadata.plane_axis is None:
        return metadata.project_channel_payload(
            image,
            image_data,
            source_channel_index,
            channel_data=output_data,
        )
    if metadata.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
        raise ValueError(
            "MeasureColocalization source channels require the declared "
            f"source-binding plane axis, got {metadata.plane_axis.value!r}."
        )
    source_plane = image_payload_slice_context(
        image,
        image_data[source_channel_index],
        source_channel_index,
    )
    return with_image_payload_data(source_plane, output_data)


@dataclass(frozen=True)
class ColocalizationCostesThresholdRequest:
    """Resolved inputs needed to compute one image-pair Costes threshold."""

    image: object
    image_data: np.ndarray
    channel_1: int
    channel_2: int
    method: CostesMethod
    scale_max: int
    backend_provider: BackendProviderInput | None
    image_pair_context: ColocalizationImagePairContext | None = None

    @property
    def cache_key(self) -> ColocalizationCostesThresholdCacheKey:
        """Return the batch-local identity for this resolved source pair."""
        return ColocalizationCostesThresholdCacheKey(
            id(self.image),
            id(self.image_data),
            self.channel_1,
            self.channel_2,
            self.method,
            self.scale_max,
            self.backend_provider,
        )

    @staticmethod
    def scale_max_for_image_pair(
        image: object,
        image_data: np.ndarray,
        channel_1: int,
        channel_2: int,
        explicit_scale_max: int | None,
    ) -> int:
        """Resolve Costes scale from image metadata, with dtype fallback."""
        if explicit_scale_max is not None:
            return int(explicit_scale_max)
        metadata = image_payload_metadata(image)
        metadata_scales = tuple(
            (
                scale
                for scale in (
                    metadata.intensity_scale_for_source_plane(channel_1),
                    metadata.intensity_scale_for_source_plane(channel_2),
                )
                if scale is not None and scale > 0
            )
        )
        if metadata_scales:
            return int(round(max(metadata_scales)))
        dtype_scale = image_intensity_scale_for_dtype(np.asarray(image_data).dtype)
        if dtype_scale is not None and dtype_scale > 0:
            return int(round(dtype_scale))
        return 255

    @classmethod
    def from_batch_request(
        cls,
        request: RuntimeBatchInvocationRequest,
        image_pair_context: ColocalizationImagePairContext | None = None,
    ) -> "ColocalizationCostesThresholdRequest | None":
        """Build a Costes request from runtime invocation metadata."""
        kwargs = request.kwargs
        if not bool(kwargs.get("do_costes", True)):
            return None
        image_data = (
            image_pair_context.image_data
            if image_pair_context is not None
            else image_payload_data(request.image)
        )
        channel_1 = int(kwargs.get("channel_1", 0))
        channel_2 = int(kwargs.get("channel_2", 1))
        return cls(
            image=request.image,
            image_data=image_data,
            channel_1=channel_1,
            channel_2=channel_2,
            method=CostesMethod(kwargs.get("costes_method", CostesMethod.FASTER)),
            scale_max=cls.scale_max_for_image_pair(
                request.image, image_data, channel_1, channel_2, kwargs.get("scale_max")
            ),
            backend_provider=kwargs.get("costes_backend_provider"),
            image_pair_context=image_pair_context,
        )

    def thresholds(self) -> ColocalizationCostesThresholds:
        """Compute Costes thresholds for this resolved image source pair."""
        if self.image_pair_context is None:
            image_pair_context = ColocalizationImagePairContext.from_request(
                self.image, channel_1=self.channel_1, channel_2=self.channel_2
            )
        else:
            image_pair_context = self.image_pair_context
        first_pixels = image_pair_context.full_first_pixels
        second_pixels = image_pair_context.full_second_pixels
        if not first_pixels.size:
            return ColocalizationCostesThresholds.from_thresholds(0.0, 0.0)
        if self.method is CostesMethod.FASTER:
            first, second = costes_backend(
                backend_provider=self.backend_provider
            ).scaled_second_channel_costes(first_pixels, second_pixels, self.scale_max)
        else:
            first, second = costes_backend(
                backend_provider=self.backend_provider
            ).linear_costes(
                first_pixels,
                second_pixels,
                self.scale_max,
                self.method is CostesMethod.FAST,
            )
        return ColocalizationCostesThresholds.from_thresholds(first, second)


class ColocalizationCostesThresholdBatch:
    """Batch-local Costes threshold cache keyed by resolved image-pair identity."""

    def __init__(self) -> None:
        self._thresholds: dict[
            ColocalizationCostesThresholdCacheKey, ColocalizationCostesThresholds
        ] = {}
        self._image_pairs: dict[
            ColocalizationImagePairCacheKey, ColocalizationImagePairContext
        ] = {}
        self._label_contexts: dict[
            ColocalizationObjectLabelCacheKey, ColocalizationObjectLabelContext
        ] = {}

    def image_pair_context(
        self, request: RuntimeBatchInvocationRequest
    ) -> ColocalizationImagePairContext:
        """Return the batch-local resolved image-pair context."""
        kwargs = request.kwargs
        image_data = image_payload_data(request.image)
        channel_1 = int(kwargs.get("channel_1", 0))
        channel_2 = int(kwargs.get("channel_2", 1))
        key = ColocalizationImagePairCacheKey(
            id(request.image), id(image_data), channel_1, channel_2
        )
        context = self._image_pairs.get(key)
        if context is None:
            context = ColocalizationImagePairContext.from_request(
                request.image, channel_1=channel_1, channel_2=channel_2
            )
            self._image_pairs[key] = context
        return context

    def object_label_context(
        self,
        request: RuntimeBatchInvocationRequest,
        image_pair_context: ColocalizationImagePairContext,
        *,
        slice_index: int = 0,
    ) -> ColocalizationObjectLabelContext:
        """Return the batch-local resolved object-label context."""
        label_payload = request.kwargs["labels"]
        if not isinstance(label_payload, ObjectLabelValue):
            raise TypeError(
                "MeasureColocalization batch labels must be an already-projected "
                f"ObjectLabelValue, got {type(label_payload).__name__}."
            )
        dense_label_array = object_label_dense_array(label_payload, dtype=np.int32)
        label_identity_payload = (
            request.completion_label_payload
            if isinstance(request, PreparedObjectMeasurementInvocation)
            else label_payload
        )
        measurement_shape = tuple(image_pair_context.first_image.shape)
        key = ColocalizationObjectLabelCacheKey.from_dense_label_payload(
            label_identity_payload,
            dense_label_array,
            image_pair_context.pair_valid_mask,
            measurement_shape=measurement_shape,
            slice_index=slice_index,
        )
        context = self._label_contexts.get(key)
        if context is None:
            context = ColocalizationObjectLabelContext.from_dense_labels(
                dense_label_array,
                pair_valid_mask=image_pair_context.pair_valid_mask,
                measurement_shape=measurement_shape,
                slice_index=slice_index,
            )
            self._label_contexts[key] = context
        return context

    def request_kwargs(
        self, request: RuntimeBatchInvocationRequest
    ) -> dict[str, object]:
        """Return request kwargs with source-pair thresholds materialized once."""
        image_pair_context = self.image_pair_context(request)
        object_label_context = self.object_label_context(
            request,
            image_pair_context,
        )
        threshold_request = ColocalizationCostesThresholdRequest.from_batch_request(
            request, image_pair_context
        )
        thresholds = None
        if threshold_request is not None:
            key = threshold_request.cache_key
            thresholds = self._thresholds.get(key)
            if thresholds is None:
                thresholds = threshold_request.thresholds()
                self._thresholds[key] = thresholds
        kwargs = {
            **request.kwargs,
            "image_pair_context": image_pair_context,
            "object_label_context": object_label_context,
        }
        if thresholds is not None:
            kwargs["costes_thresholds"] = thresholds
        return kwargs


def measure_colocalization_objects_batch(
    func: Callable[..., object],
    requests: tuple[RuntimeBatchInvocationRequest, ...],
    execute_request: Callable[
        [Callable[..., object], RuntimeBatchInvocationRequest], object
    ],
) -> list[object]:
    """Batch object colocalization over shared image-pair thresholds and ranks."""
    threshold_batch = ColocalizationCostesThresholdBatch()
    rank_provider = CachedObjectColocalizationRankProvider()
    outputs: list[object] = []
    profile_enabled = runtime_profiler.enabled()
    for request in requests:
        request_started_at = time.perf_counter() if profile_enabled else 0.0
        phase_started_at = time.perf_counter() if profile_enabled else 0.0
        prepared_kwargs = {
            **threshold_batch.request_kwargs(request),
            "rank_provider": rank_provider,
        }
        if profile_enabled:
            runtime_profiler.log(
                "coloc_object_batch_prepare",
                time.perf_counter() - phase_started_at,
                function="measure_colocalization_objects",
            )
        phase_started_at = time.perf_counter() if profile_enabled else 0.0
        outputs.append(execute_request(func, replace(request, kwargs=prepared_kwargs)))
        if profile_enabled:
            runtime_profiler.log(
                "coloc_object_batch_fallback",
                time.perf_counter() - phase_started_at,
                function="measure_colocalization_objects",
            )
            runtime_profiler.log(
                "coloc_object_batch_request",
                time.perf_counter() - request_started_at,
                function="measure_colocalization_objects",
            )
    return outputs


measurement_image_batch_executor(measure_colocalization_objects_batch)(
    measure_colocalization_objects
)


def _prepare_measure_colocalization_objects() -> None:
    """Compile object-colocalization reduction kernels before measured execution."""
    _prepare_measure_colocalization()
    first_pixels = np.linspace(0.0, 1.0, 16, dtype=np.float32)
    second_pixels = np.linspace(1.0, 0.0, 16, dtype=np.float32)
    object_labels = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    object_count = 4
    reductions = object_colocalization_base_reductions(
        first_pixels, second_pixels, object_labels, object_count
    )
    threshold_1 = 0.15 * reductions[6]
    threshold_2 = 0.15 * reductions[7]
    object_colocalization_threshold_reductions(
        first_pixels,
        second_pixels,
        object_labels,
        threshold_1,
        threshold_2,
        0.1,
        0.1,
        object_count,
    )
    ranks = np.arange(first_pixels.size, dtype=np.int64)
    object_colocalization_rwc_reductions(
        first_pixels,
        second_pixels,
        object_labels,
        threshold_1,
        threshold_2,
        ranks,
        ranks,
        first_pixels.size,
        object_count,
    )


def _prepare_measure_colocalization() -> None:
    """Compile image-colocalization kernels before measured execution."""
    first_pixels = np.linspace(0.0, 1.0, 64, dtype=np.float64)
    second_pixels = np.linspace(1.0, 0.0, 64, dtype=np.float64)
    costes_backend().prepare_backend()
    _costes_manders_numba(first_pixels, second_pixels, 0.25, 0.25)


measure_colocalization.__openhcs_prepare__ = _prepare_measure_colocalization
measure_colocalization_objects.__openhcs_prepare__ = (
    _prepare_measure_colocalization_objects
)


def _divide_measurements(numerator: object, denominator: object) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator_array / denominator_array
    result[~np.isfinite(result)] = 0
    return result


def _divide_costes_measurements(numerator: object, denominator: object) -> np.ndarray:
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return numerator_array / denominator_array


__all__ = public_names_from_objects(
    ColocalizationCostesBackendStrategy,
    ColocalizationCostesThresholdBatch,
    ColocalizationCostesThresholdRequest,
    ColocalizationCostesThresholds,
    ColocalizationImagePairContext,
    ColocalizationMeasurementOptions,
    ColocalizationMeasurementSchema,
    ColocalizationMeasurements,
    ColocalizationObjectLabelContext,
    CostesMethod,
    NumbaNumpyColocalizationCostesBackendStrategy,
    ObjectColocalizationMeasurements,
    ObjectColocalizationColumnarMeasurements,
    UnitIntervalDenseRankSemantics,
    costes_above_threshold_mask,
    costes_backend,
    measure_colocalization,
    measure_colocalization_objects,
    measure_colocalization_objects_batch,
    object_colocalization_base_reductions,
    object_colocalization_threshold_reductions,
    thresholded_colocalization_metrics,
)
