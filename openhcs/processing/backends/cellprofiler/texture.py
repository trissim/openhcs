"""Texture-measurement backends for CellProfiler-compatible processing."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from inspect import unwrap
from typing import ClassVar

from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    setting_names,
)
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
    parse_cellprofiler_int,
)

from openhcs.processing.backends.cellprofiler.module_classes import (
    ArtifactContractModule,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
    ObjectMeasurementRowsModule,
    PerObjectMeasurementExecutionModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.dual_scope_measurement_policies import (
    DeclaredDualScopeMeasurementPolicy,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import cellprofiler_enum_from_literal


from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_ID_FIELDS,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD,
    MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
    measurement_object_label,
)
from openhcs.core.runtime_semantics import (
    MeasurementObjectRowIdentity,
    MeasurementRowAxisField,
    MeasurementScalarLiteral,
    ObjectLabelDomainScope,
    RuntimePlaneAxis,
    measurement_row_mapping,
)
from openhcs.core.runtime_values import ColumnarRows
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup
from openhcs.interop.cellprofiler.runtime.measurement_rows import LABEL_PAYLOAD_FINAL
from openhcs.interop.cellprofiler.runtime.object_measurement_row_completion import (
    MissingObjectMeasurementValuePolicy,
    MissingObjectMeasurementValueRequest,
    MissingObjectMeasurementValueStrategy,
    ObjectMeasurementIdsByAxisView,
    ObjectMeasurementProjectedRowKeys,
    ObjectMeasurementRowCompletionSchema,
    ObjectMeasurementRowIdentityProjectionRequest,
    ObjectMeasurementRowIdentityProjectionResult,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargDict,
    CellProfilerRuntimeValue,
    CellProfilerRuntimeValues,
    CellProfilerRuntimeValueSequence,
    MeasurementRowMapping,
    MeasurementRowsInput,
)
from openhcs.interop.cellprofiler.runtime.processing_contracts import (
    RuntimeShapeInspection,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

class MeasureTextureObjectMeasurementRowPolicy(
    DeclaredDomainCompactMeasuredObjectMeasurementRowPolicy
):
    """Use direct texture rows when CP already emitted the declared dense domain."""

    row_identity = MeasurementObjectRowIdentity.ROW_SEQUENCE
    row_sequence_axis_fields: ClassVar[frozenset[str]] = frozenset(
        (
            MeasurementRowAxisField.SCALE.value,
            MeasurementRowAxisField.DIRECTION.value,
            MeasurementRowAxisField.GRAY_LEVELS.value,
        )
    )

    def row_identity_axis_fields(
        self,
        axis_fields: Sequence[str],
        *,
        label_payload: CellProfilerRuntimeValue | None = None,
    ) -> tuple[str, ...]:
        """Texture compact rows are sequenced by feature axis, not source/slice axes."""
        if not MeasureTextureMissingValueDomain.from_payload(
            label_payload
        ).is_multi_source_plane_domain():
            return tuple(axis_fields)
        return tuple(
            field_name
            for field_name in axis_fields
            if field_name in type(self).row_sequence_axis_fields
        )

    def object_identity_for_label_payload(
        self,
        label_payload: CellProfilerRuntimeValue,
    ) -> MeasurementObjectRowIdentity:
        """Use row-sequence identity only for multi-source plane-domain texture rows."""
        if MeasureTextureMissingValueDomain.from_payload(
            label_payload
        ).is_multi_source_plane_domain():
            return MeasurementObjectRowIdentity.ROW_SEQUENCE
        return MeasurementObjectRowIdentity.ROW_ORDINAL

    def missing_measurement_value(
        self,
        *,
        object_id: int,
        label_payload: CellProfilerRuntimeValue,
        field_name: str,
        positive_label_extent: int | None = None,
    ) -> float:
        missing_domain = MeasureTextureMissingValueDomain.from_payload(label_payload)
        value_policy = missing_domain.missing_value_policy(type(self).missing_value_policy)
        if positive_label_extent is None:
            positive_label_extent = missing_domain.compact_row_ordinal_positive_extent()
        strategy = MissingObjectMeasurementValueStrategy.for_enum_member(value_policy)
        return strategy.missing_value(
            MissingObjectMeasurementValueRequest(
                object_id=object_id,
                label_payload=label_payload,
                field_name=field_name,
                positive_label_extent=positive_label_extent,
            )
        )

    def complete_rows(
        self,
        rows: MeasurementRowsInput,
        *,
        label_payload: CellProfilerRuntimeValue,
        func: CellProfilerFunction,
    ) -> MeasurementRowsInput:
        """Avoid padding work only when emitted rows already match the domain."""
        if isinstance(rows, ColumnarRows):
            return rows
        missing_domain = MeasureTextureMissingValueDomain.from_payload(label_payload)
        schema = ObjectMeasurementRowCompletionSchema.from_rows(rows, func)
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "measure_texture_complete_rows",
            0.0,
            rows=len(rows),
            axis_fields=schema.axis_fields,
            object_id_field=schema.object_id_field,
            multi_source_domain=missing_domain.is_multi_source_plane_domain(),
            label_payload_type=type(label_payload).__name__,
            label_shape=RuntimeShapeInspection(
                np.asarray(LABEL_PAYLOAD_FINAL.value(label_payload))
            ).shape_tuple(),
        )
        from openhcs.processing.backends.cellprofiler.texture import (
            measure_texture_objects,
        )

        if (
            rows
            and unwrap(func) is unwrap(measure_texture_objects)
            and not missing_domain.is_multi_source_plane_domain()
        ):
            return list(rows)
        if not schema.axis_fields and rows:
            required_object_ids = schema.object_ids_for_axis(
                label_payload=label_payload,
                object_identity=self.object_identity(),
                axis_key=(),
            )
            emitted_object_ids = tuple(
                int(measurement_row_mapping(row)[schema.object_id_field])
                for row in rows
                if self.row_is_object_scoped(row)
            )
            if emitted_object_ids == required_object_ids:
                return missing_domain.normalize_existing_rows(
                    rows,
                    field_names=schema.field_names,
                    object_id_field=schema.object_id_field,
                    axis_fields=schema.axis_fields,
                )
        if schema.axis_fields and rows:
            complete_axis_rows = TextureAxisMeasurementRows.from_rows(
                rows,
                schema=schema,
                label_payload=label_payload,
                row_policy=self,
            )
            if complete_axis_rows.already_complete:
                return missing_domain.normalize_existing_rows(
                    rows,
                    field_names=schema.field_names,
                    object_id_field=schema.object_id_field,
                    axis_fields=schema.axis_fields,
                )
        completed_rows = super().complete_rows(
            rows,
            label_payload=label_payload,
            func=func,
        )
        if isinstance(completed_rows, ColumnarRows):
            return completed_rows
        return missing_domain.normalize_existing_rows(
            completed_rows,
            field_names=schema.field_names,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
        )


@dataclass(frozen=True, slots=True)
class TextureAxisMeasurementRows:
    """Observed MeasureTexture row coverage by measurement axis."""

    emitted_object_ids_by_axis: ObjectMeasurementIdsByAxisView
    required_object_ids_by_axis: ObjectMeasurementIdsByAxisView

    @classmethod
    def from_rows(
        cls,
        rows: CellProfilerRuntimeValueSequence,
        *,
        schema: ObjectMeasurementRowCompletionSchema,
        label_payload: CellProfilerRuntimeValue,
        row_policy: MeasureTextureObjectMeasurementRowPolicy,
    ) -> "TextureAxisMeasurementRows":
        emitted_ids: dict[CellProfilerRuntimeValues, list[int]] = {}
        projection_request = ObjectMeasurementRowIdentityProjectionRequest(
            rows=rows,
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            row_policy=row_policy,
        )
        for row in rows:
            row_mapping = measurement_row_mapping(row)
            object_id = measurement_object_label(
                row_mapping,
                object_id_field=schema.object_id_field,
            )
            if object_id is None:
                continue
            axis_key = projection_request.axis_key_from_mapping(row_mapping)
            if axis_key not in emitted_ids:
                emitted_ids[axis_key] = []
            emitted_ids[axis_key].append(int(object_id))
        axis_keys = tuple(emitted_ids)
        required_ids_by_axis = row_policy.required_object_ids_by_axis(
            label_payload=label_payload,
            projection=ObjectMeasurementRowIdentityProjectionResult(
                rows=tuple(rows),
                row_keys=ObjectMeasurementProjectedRowKeys(
                    tuple(
                        (object_id, axis_key)
                        for axis_key, object_ids in emitted_ids.items()
                        for object_id in object_ids
                    )
                ),
                measured_row_keys=ObjectMeasurementProjectedRowKeys(
                    tuple(
                        (object_id, axis_key)
                        for axis_key, object_ids in emitted_ids.items()
                        for object_id in object_ids
                    )
                ),
                axis_keys=axis_keys,
            ),
            object_identity=row_policy.object_identity_for_label_payload(label_payload),
            object_id_field=schema.object_id_field,
            axis_fields=schema.axis_fields,
            axis_keys=axis_keys,
        )
        return cls(
            emitted_object_ids_by_axis={
                axis_key: tuple(object_ids)
                for axis_key, object_ids in emitted_ids.items()
            },
            required_object_ids_by_axis=required_ids_by_axis,
        )

    @property
    def already_complete(self) -> bool:
        if not self.emitted_object_ids_by_axis:
            return False
        if self.emitted_object_ids_by_axis.keys() != self.required_object_ids_by_axis.keys():
            return False
        for axis_key, object_ids in self.emitted_object_ids_by_axis.items():
            required_object_ids = self.required_object_ids_by_axis[axis_key]
            if len(object_ids) != len(required_object_ids):
                return False
            if frozenset(object_ids) != frozenset(required_object_ids):
                return False
        return True


@dataclass(frozen=True, slots=True)
class MeasureTextureMissingValueDomain:
    """Resolve texture padding semantics from the object-label measurement domain."""

    label_payload: ObjectLabelValue | None

    @classmethod
    def from_payload(
        cls,
        label_payload: CellProfilerRuntimeValue | None,
    ) -> "MeasureTextureMissingValueDomain":
        """Return texture domain semantics only for object-label payloads."""
        if isinstance(label_payload, ObjectLabelValue):
            return cls(label_payload)
        return cls(None)

    def missing_value_policy(
        self,
        default_policy: MissingObjectMeasurementValuePolicy,
    ) -> MissingObjectMeasurementValuePolicy:
        if self.is_multi_source_plane_domain():
            return MissingObjectMeasurementValuePolicy.ZERO_WITHIN_POSITIVE_EXTENT
        return default_policy

    def is_multi_source_plane_domain(self) -> bool:
        payload = self.label_payload
        if payload is None:
            return False
        if payload.domain.scope is not ObjectLabelDomainScope.PLANE:
            return False
        if payload.plane_axis is not RuntimePlaneAxis.RUNTIME_SLICE:
            return False
        if len(payload.source_image_names) <= 1:
            return False
        labels = np.asarray(LABEL_PAYLOAD_FINAL.value(payload))
        return labels.ndim >= 4

    def normalize_existing_rows(
        self,
        rows: CellProfilerRuntimeValueSequence,
        *,
        field_names: Sequence[str],
        object_id_field: str,
        axis_fields: Sequence[str],
    ) -> list[CellProfilerRuntimeValue]:
        if not self.is_multi_source_plane_domain():
            return list(rows)
        extent = self.compact_row_ordinal_positive_extent()
        if extent is None:
            return list(rows)
        extent = max(extent, self.compact_row_ordinal_extent_from_rows(rows))
        identity_fields = {
            object_id_field,
            *MEASUREMENT_OBJECT_ID_FIELDS,
            *axis_fields,
            MEASUREMENT_OBJECT_NAME_FIELD,
            MEASUREMENT_SOURCE_IMAGE_NAME_FIELD,
        }
        measurement_fields = tuple(
            field_name for field_name in field_names if field_name not in identity_fields
        )
        return MeasureTextureExistingRowsNormalizer(
            rows=rows,
            extent=extent,
            field_names=tuple(field_names),
            identity_fields=frozenset(identity_fields),
            first_measurement_field=FirstMeasurementField(
                measurement_fields
            ).value_or_none(),
        ).normalized_rows()

    def compact_row_ordinal_positive_extent(self) -> int | None:
        """Return the declared compact-row extent for multi-source texture labels."""
        if not self.is_multi_source_plane_domain():
            return None
        if not isinstance(self.label_payload, ObjectLabelValue):
            raise TypeError(
                "MeasureTexture row-ordinal extent requires ObjectLabelValue, got "
                f"{type(self.label_payload).__name__}."
            )
        domain = self.label_payload.object_label_domain()
        if domain.declared_object_id_domains:
            return max(len(object_ids) for object_ids in domain.declared_object_id_domains)
        explicit_domain = domain.explicit_id_domain()
        if explicit_domain is not None:
            return len(explicit_domain)
        return None

    @staticmethod
    def compact_row_ordinal_extent_from_rows(rows: CellProfilerRuntimeValueSequence) -> int:
        """Return the largest compact row ordinal already emitted by texture rows."""
        object_ids = tuple(
            object_id
            for row in rows
            for object_id in (measurement_object_label(measurement_row_mapping(row)),)
            if object_id is not None
        )
        if not object_ids:
            return 0
        return max(object_ids)


@dataclass(frozen=True, slots=True)
class FirstMeasurementField:
    """First non-identity measurement field available for row diagnostics."""

    measurement_fields: tuple[str, ...]

    def value_or_none(self) -> str | None:
        match self.measurement_fields:
            case (field_name, *_):
                return field_name
            case _:
                return None


@dataclass(slots=True)
class MeasureTextureExistingRowsNormalizer:
    """Normalize existing MeasureTexture rows for compact multi-source domains."""

    rows: CellProfilerRuntimeValueSequence
    extent: int
    field_names: tuple[str, ...]
    identity_fields: frozenset[str]
    first_measurement_field: str | None
    first_field_value_types: Counter[str] = field(default_factory=Counter)
    first_field_sample_values: list[str] = field(default_factory=list)
    nan_replacements: int = 0
    none_replacements: int = 0
    absent_replacements: int = 0

    def normalized_rows(self) -> list[CellProfilerRuntimeValue]:
        normalized_rows = [self.normalized_row(row) for row in self.rows]
        self.log_profile(len(normalized_rows))
        return normalized_rows

    def normalized_row(self, row: CellProfilerRuntimeValue) -> CellProfilerRuntimeValue:
        row_mapping = measurement_row_mapping(row)
        object_id = measurement_object_label(row_mapping)
        if object_id is None or object_id > self.extent:
            return row
        normalized_row = dict(row_mapping)
        normalized_row[MEASUREMENT_OBJECT_ROW_IDENTITY_FIELD] = (
            MeasurementObjectRowIdentity.ROW_SEQUENCE.value
        )
        self.record_first_field_sample(row_mapping)
        self.replace_existing_measurements(row_mapping, normalized_row)
        self.add_absent_measurements(normalized_row)
        return normalized_row

    def record_first_field_sample(self, row_mapping: MeasurementRowMapping) -> None:
        field_name = self.first_measurement_field
        if field_name is None:
            return
        first_value = MappingValueLookup(row_mapping, field_name).value_or("<ABSENT>")
        self.first_field_value_types[type(first_value).__name__] += 1
        if len(self.first_field_sample_values) < 6:
            self.first_field_sample_values.append(repr(first_value))

    def replace_existing_measurements(
        self,
        row_mapping: MeasurementRowMapping,
        normalized_row: CellProfilerKwargDict,
    ) -> None:
        for field_name, value in row_mapping.items():
            if field_name in self.identity_fields:
                continue
            self.replace_existing_measurement(field_name, value, normalized_row)

    def replace_existing_measurement(
        self,
        field_name: str,
        value: CellProfilerRuntimeValue,
        normalized_row: CellProfilerKwargDict,
    ) -> None:
        if value is None:
            normalized_row[field_name] = 0.0
            self.none_replacements += 1
            return
        if MeasurementScalarLiteral(value).is_padding_measurement_value:
            normalized_row[field_name] = 0.0
            self.nan_replacements += 1

    def add_absent_measurements(self, normalized_row: CellProfilerKwargDict) -> None:
        for field_name in self.field_names:
            if field_name in self.identity_fields or field_name in normalized_row:
                continue
            normalized_row[field_name] = 0.0
            self.absent_replacements += 1

    def log_profile(self, row_count: int) -> None:
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "measure_texture_normalize_rows",
            0.0,
            rows=row_count,
            extent=self.extent,
            field_count=len(self.field_names),
            first_measurement_field=self.first_measurement_field,
            first_field_value_types=dict(self.first_field_value_types),
            first_field_sample_values=tuple(self.first_field_sample_values),
            nan_replacements=self.nan_replacements,
            none_replacements=self.none_replacements,
            absent_replacements=self.absent_replacements,
        )

class MeasureTextureModule(
    LabelsObjectInputPolicy,
    PerObjectMeasurementExecutionModule,
    ObjectMeasurementRowsModule,
    MeasureTextureObjectMeasurementRowPolicy,
    DeclaredDualScopeMeasurementPolicy,
    ScopedMeasurementModule,
):
    module_name = 'MeasureTexture'
    function_name = 'measure_texture'
    validated = True
    function_variants = ('measure_texture_objects',)
    image_function_name = 'measure_texture'
    confidence = 1.0
    measurement_scope_setting = SettingNameFamily(
        "Measure images or objects?",
        aliases=("Measure whole images or objects?",),
    )
    object_measurement_setting = SettingNameFamily(
        "Select objects to measure",
        aliases=("Select an object to measure",),
    )
    ignored_settings = (
        "Hidden",
        "Angles to measure",
        "Measure Gabor features?",
        "Number of angles to compute for Gabor",
    )

    class MeasurementScope(str, Enum):
        image = "Images"
        objects = "Objects"
        both = "Both"

        @classmethod
        def from_literal(
            cls,
            value: "MeasureTextureModule.MeasurementScope | str",
        ) -> "MeasureTextureModule.MeasurementScope":
            return cellprofiler_enum_from_literal(cls, value)

    measurement_scope_default = MeasurementScope.image

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        default_function_name: str | None = None,
    ) -> "ResolvedModuleFunction":
        scope = cls.MeasurementScope.from_literal(
            cls.setting_value(module, cls.measurement_scope_setting)
            or cls.measurement_scope_default.value
        )
        object_values = setting_values(module, cls.object_measurement_setting)
        has_objects = any(split_symbol_names(value) for value in object_values)
        if (
            scope in (cls.MeasurementScope.objects, cls.MeasurementScope.both)
            and has_objects
        ):
            return super().resolve_function(
                module,
                default_function_name=cls.function_variants[0],
            )
        return super().resolve_function(
            module,
            default_function_name=default_function_name,
        )

    @classmethod
    def measurement_target_scope(
        cls,
        module: "ModuleBlock",
    ) -> "MeasureTextureModule.MeasurementScope":
        return cls.MeasurementScope.from_literal(
            cls.setting_value(module, cls.measurement_scope_setting)
            or cls.measurement_scope_default.value
        )

    texture_scale_setting = "Texture scale to measure"
    gray_levels_setting = "Enter how many gray levels to measure the texture at"

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: "SettingsBinder",
        param_mapping: Mapping[str, Any],
        ignored_unmapped_settings: frozenset[str] = frozenset(),
    ) -> "BoundModuleSettings":
        bound = cls._bind_generic_settings(
            module,
            binder=binder,
            param_mapping=param_mapping,
        )
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in setting_names(cls.measurement_scope_setting):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)

        texture_scales = setting_values(module, cls.texture_scale_setting)
        if texture_scales:
            parsed_scales = tuple(parse_cellprofiler_int(value) for value in texture_scales)
            kwargs["scale"] = (
                parsed_scales[0] if len(parsed_scales) == 1 else parsed_scales
            )
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.texture_scale_setting),
                None,
            )

        gray_levels = optional_setting_value(module, cls.gray_levels_setting)
        if gray_levels is not None:
            kwargs["gray_levels"] = parse_cellprofiler_int(gray_levels)
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.gray_levels_setting),
                None,
            )

        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(
                module,
                BoundModuleSettings(kwargs, unmapped_kwargs),
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        return cls.measurement_artifact_contract_from_declared_settings(
            assembler,
            builder,
            module,
        )



from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from metaclass_registry import AutoRegisterMeta
from openhcs.core.alias_property import AliasProperty
from numba import njit

from openhcs.constants.constants import MemoryType
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import (
    pure_2d_batch_executor,
    special_inputs,
    special_outputs,
)
from openhcs.core.runtime_batch_contracts import RuntimePure2DSliceBatchRequest
from openhcs.core.runtime_semantics import dense_object_label_id_domain
from openhcs.core.runtime_values import (
    DenseObjectLabelPlaneDomainStackRequest,
    ObjectLabelMeasurementPayloadStrategy,
    ObjectLabelSourcePlaneProjectionRequest,
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.materialization import csv_materializer
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


F_HARALICK = [
    "AngularSecondMoment",
    "Contrast",
    "Correlation",
    "Variance",
    "InverseDifferenceMoment",
    "SumAverage",
    "SumVariance",
    "SumEntropy",
    "Entropy",
    "DifferenceVariance",
    "DifferenceEntropy",
    "InfoMeas1",
    "InfoMeas2",
]

N_DIRECTIONS_2D = 4
ObjectIntensityCrops = tuple[np.ndarray, tuple[np.ndarray, ...]]
TextureLabelSource: TypeAlias = ObjectLabelValue | np.ndarray | None
ObjectTextureResult: TypeAlias = tuple[np.ndarray, list["ObjectTextureMeasurement"]]


@dataclass(frozen=True, slots=True)
class HaralickFeatureColumn:
    """Descriptor exposing one Haralick vector coordinate as a row column."""

    feature_index: int

    def __get__(
        self,
        instance: "HaralickFeatureColumns | None",
        owner: type["HaralickFeatureColumns"] | None = None,
    ) -> float | "HaralickFeatureColumn":
        del owner
        if instance is None:
            return self
        return float(instance.features.values[self.feature_index])


class TextureAxisColumns:
    """Output-column aliases owned by the texture axis carrier."""

    slice_index: ClassVar[AliasProperty[int]] = AliasProperty("axis.slice_index")
    scale: ClassVar[AliasProperty[int]] = AliasProperty("axis.scale")
    direction: ClassVar[AliasProperty[int]] = AliasProperty("axis.direction")
    gray_levels: ClassVar[AliasProperty[int]] = AliasProperty("axis.gray_levels")


class HaralickFeatureColumns:
    """Output-column aliases owned by the Haralick feature vector."""

    angular_second_moment: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(0)
    contrast: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(1)
    correlation: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(2)
    variance: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(3)
    inverse_difference_moment: ClassVar[HaralickFeatureColumn] = (
        HaralickFeatureColumn(4)
    )
    sum_average: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(5)
    sum_variance: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(6)
    sum_entropy: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(7)
    entropy: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(8)
    difference_variance: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(9)
    difference_entropy: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(10)
    info_meas1: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(11)
    info_meas2: ClassVar[HaralickFeatureColumn] = HaralickFeatureColumn(12)


@dataclass
class TextureMeasurement(TextureAxisColumns, HaralickFeatureColumns):
    """Texture measurement results for a single slice/image."""

    axis: "TextureMeasurementAxis"
    features: "HaralickFeatureVector"
    source_image_name: str | None = None


@dataclass
class ObjectTextureMeasurement(TextureAxisColumns, HaralickFeatureColumns):
    """Texture measurement results per object."""

    object_label: int
    axis: "TextureMeasurementAxis"
    features: "HaralickFeatureVector"
    source_image_name: str | None = None


class ObjectTextureCropBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Extract masked object intensity crops for texture measurement."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def object_intensity_crops(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityCrops:
        """Return positive object labels and CP-style masked intensity crops."""


class HaralickTextureBackendStrategy(
    CellProfilerBackendStrategyMixin,
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Compute CP-compatible 2-D Haralick feature matrices."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        """Return one Haralick feature row per 2-D direction."""


class NumbaNumpyObjectTextureCropBackendStrategy(ObjectTextureCropBackendStrategy):
    """Numba-accelerated NumPy backend for object texture crop extraction."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(9, dtype=np.float64).reshape((3, 3))
        labels = np.array([[0, 1, 1], [0, 1, 0], [2, 2, 0]], dtype=np.int64)
        self.object_intensity_crops(image, labels)

    def object_intensity_crops(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> ObjectIntensityCrops:
        image_array = np.asarray(image)
        labels_array = np.asarray(labels)
        if image_array.ndim != 2 or labels_array.ndim != 2:
            raise NotImplementedError(
                "Numba texture crop backend currently supports 2-D NumPy planes."
            )
        if image_array.shape != labels_array.shape:
            raise ValueError(
                "Texture image and labels must have identical shapes; got "
                f"{image_array.shape!r} and {labels_array.shape!r}."
            )
        object_labels, boxes = _object_bounding_boxes_numba(
            np.ascontiguousarray(labels_array, dtype=np.int64)
        )
        crops: list[np.ndarray] = []
        for index, object_label in enumerate(object_labels):
            y0, y1, x0, x1 = boxes[index]
            label_crop = labels_array[y0:y1, x0:x1]
            intensity_crop = np.asarray(image_array[y0:y1, x0:x1]).copy()
            intensity_crop[label_crop != object_label] = 0
            crops.append(intensity_crop)
        return object_labels.astype(np.int64, copy=False), tuple(crops)


class NumbaNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Numba implementation of mahotas' default 2-D Haralick semantics."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NUMBA,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        image = np.arange(25, dtype=np.int64).reshape((5, 5))
        self.haralick_features(image, scale=1, ignore_zeros=False)

    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        pixel_array = np.ascontiguousarray(pixel_data)
        if pixel_array.ndim != 2:
            raise ValueError("Haralick texture backend expects a 2-D image plane.")
        if scale < 1:
            raise ValueError(f"Haralick texture scale must be positive, got {scale}.")
        if pixel_array.shape[0] <= scale or pixel_array.shape[1] <= scale:
            return np.zeros((4, 13), dtype=np.float64)
        return _haralick_2d_features_numba(
            pixel_array.astype(np.int64, copy=False),
            int(scale),
            bool(ignore_zeros),
        )


class NativeNumpyHaralickTextureBackendStrategy(HaralickTextureBackendStrategy):
    """Explicit mahotas backend used as the native reference implementation."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY,
        CellProfilerBackendProvider.NATIVE,
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NATIVE

    def haralick_features(
        self,
        pixel_data: np.ndarray,
        *,
        scale: int,
        ignore_zeros: bool,
    ) -> np.ndarray:
        import mahotas.features as mahotas_features

        return np.asarray(
            mahotas_features.haralick(
                np.asarray(pixel_data),
                distance=scale,
                ignore_zeros=ignore_zeros,
            ),
            dtype=np.float64,
        )


def _normalize_gray_levels(gray_levels: int) -> int:
    return max(2, min(256, int(gray_levels)))


def _texture_scales(scale: int | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(scale, (tuple, list)):
        return tuple(int(value) for value in scale)
    return (int(scale),)


@dataclass(frozen=True, slots=True)
class CellProfilerTexturePixelDataRequest:
    """Quantize image data the same way CellProfiler MeasureTexture does."""

    image: np.ndarray
    gray_levels: int

    def pixel_data(self) -> np.ndarray:
        from skimage.exposure import rescale_intensity
        from skimage.util import img_as_ubyte

        pixel_data = (
            self.image.copy()
            if self.image.dtype == np.uint8
            else img_as_ubyte(self.image)
        )
        if self.gray_levels != 256:
            pixel_data = rescale_intensity(
                pixel_data,
                in_range=(0, 255),
                out_range=(0, self.gray_levels - 1),
            ).astype(np.uint8)
        return pixel_data


def _zero_feature_matrix() -> np.ndarray:
    return np.zeros((N_DIRECTIONS_2D, len(F_HARALICK)), dtype=float)


def _clean_feature_vector(features: np.ndarray) -> np.ndarray:
    clean = np.asarray(features, dtype=float).copy()
    clean[~np.isfinite(clean)] = 0
    return clean


@dataclass(frozen=True, slots=True)
class HaralickFeatureMatrixRequest:
    """Request for CP-compatible Haralick rows using an explicit backend."""

    pixel_data: np.ndarray
    scale: int
    ignore_zeros: bool
    backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION

    def feature_matrix(self) -> np.ndarray:
        pixel_data = np.asarray(self.pixel_data)
        if not _haralick_has_valid_domain(
            pixel_data,
            scale=self.scale,
            ignore_zeros=self.ignore_zeros,
        ):
            return _zero_feature_matrix()

        backend = HaralickTextureBackendStrategy.for_memory_type(
            backend_provider=self.backend_provider,
        )
        return np.asarray(
            backend.haralick_features(
                pixel_data,
                scale=self.scale,
                ignore_zeros=self.ignore_zeros,
            ),
            dtype=float,
        )


def _haralick_has_valid_domain(
    pixel_data: np.ndarray,
    *,
    scale: int,
    ignore_zeros: bool,
) -> bool:
    if pixel_data.ndim != 2:
        raise ValueError(
            "MeasureTexture expects a 2D image plane. Stack dispatch must be "
            "handled by the OpenHCS processing contract."
        )
    if scale < 1:
        raise ValueError(f"MeasureTexture scale must be positive, got {scale}.")
    if pixel_data.shape[0] <= scale or pixel_data.shape[1] <= scale:
        return False
    if not ignore_zeros:
        return True
    nonzero = pixel_data != 0
    return _has_nonzero_haralick_pairs(nonzero, scale)


def _has_nonzero_haralick_pairs(nonzero: np.ndarray, scale: int) -> bool:
    return (
        np.any(nonzero[:, :-scale] & nonzero[:, scale:])
        and np.any(nonzero[:-scale, :-scale] & nonzero[scale:, scale:])
        and np.any(nonzero[:-scale, :] & nonzero[scale:, :])
        and np.any(nonzero[:-scale, scale:] & nonzero[scale:, :-scale])
    )


@dataclass(frozen=True, slots=True)
class TextureMeasurementAxis:
    """Scale, direction, and gray-level coordinates for one Haralick row."""

    slice_index: int
    scale: int
    direction: int
    gray_levels: int

    def object_key(self, object_label: int) -> tuple[int, int, int, int, int]:
        return (
            object_label,
            self.slice_index,
            self.scale,
            self.direction,
            self.gray_levels,
        )


@dataclass(frozen=True, slots=True)
class HaralickFeatureVector:
    """Cleaned Haralick feature row with constructors for output records."""

    values: np.ndarray

    @classmethod
    def from_matrix(
        cls,
        feature_matrix: np.ndarray,
        direction: int,
    ) -> "HaralickFeatureVector":
        if direction >= feature_matrix.shape[0]:
            return cls.zeros()
        return cls(_clean_feature_vector(feature_matrix[direction, :]))

    @classmethod
    def zeros(cls) -> "HaralickFeatureVector":
        return cls(np.zeros((len(F_HARALICK),), dtype=float))

    def image_measurement(self, axis: TextureMeasurementAxis) -> TextureMeasurement:
        return TextureMeasurement(
            axis=axis,
            features=self,
        )

    def object_measurement(
        self,
        axis: TextureMeasurementAxis,
        *,
        object_label: int,
    ) -> ObjectTextureMeasurement:
        return ObjectTextureMeasurement(
            object_label=object_label,
            axis=axis,
            features=self,
        )


@numpy(contract=ProcessingContract.PURE_2D)
@special_outputs(
    (
        "texture_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "scale",
                "direction",
                "gray_levels",
                "angular_second_moment",
                "contrast",
                "correlation",
                "variance",
                "inverse_difference_moment",
                "sum_average",
                "sum_variance",
                "sum_entropy",
                "entropy",
                "difference_variance",
                "difference_entropy",
                "info_meas1",
                "info_meas2",
                "source_image_name",
            ],
            analysis_type="texture",
        ),
    )
)
def measure_texture(
    image: np.ndarray,
    scale: int | tuple[int, ...] | list[int] = 3,
    gray_levels: int = 256,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, list[TextureMeasurement]]:
    """Measure Haralick texture features on a grayscale image."""
    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = CellProfilerTexturePixelDataRequest(
        image=image,
        gray_levels=gray_levels,
    ).pixel_data()

    measurements = []
    for texture_scale in _texture_scales(scale):
        feature_matrix = HaralickFeatureMatrixRequest(
            pixel_data=pixel_data,
            scale=texture_scale,
            ignore_zeros=False,
            backend_provider=haralick_backend_provider,
        ).feature_matrix()

        for direction in range(N_DIRECTIONS_2D):
            axis = TextureMeasurementAxis(
                slice_index=0,
                scale=texture_scale,
                direction=direction,
                gray_levels=gray_levels,
            )
            measurements.append(
                HaralickFeatureVector.from_matrix(
                    feature_matrix,
                    direction,
                ).image_measurement(
                    axis,
                )
            )

    return image, measurements


@dataclass(frozen=True, slots=True)
class ObjectTextureMeasurementCompletionRequest:
    """Fill missing per-object texture rows for the declared label domain."""

    measurements: tuple[ObjectTextureMeasurement, ...]
    labels: TextureLabelSource
    scale: int | tuple[int, ...] | list[int]
    gray_levels: int

    def complete(self) -> list[ObjectTextureMeasurement]:
        object_domain = dense_object_label_id_domain(self.labels)
        if not object_domain:
            return list(self.measurements)

        by_key = {
            measurement.axis.object_key(measurement.object_label): measurement
            for measurement in self.measurements
        }
        complete: list[ObjectTextureMeasurement] = []
        zero_features = HaralickFeatureVector.zeros()
        axes = self.axes
        for object_label in object_domain:
            for axis in axes:
                key = axis.object_key(object_label)
                if key in by_key:
                    complete.append(by_key[key])
                    continue
                complete.append(
                    zero_features.object_measurement(
                        axis,
                        object_label=object_label,
                    )
                )
        return complete

    @property
    def axes(self) -> tuple[TextureMeasurementAxis, ...]:
        axes = tuple(
            dict.fromkeys(
                measurement.axis
                for measurement in self.measurements
            )
        )
        if axes:
            return axes
        return tuple(
            TextureMeasurementAxis(
                slice_index=0,
                scale=texture_scale,
                direction=direction,
                gray_levels=self.gray_levels,
            )
            for texture_scale in _texture_scales(self.scale)
            for direction in range(N_DIRECTIONS_2D)
        )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "object_texture_measurements",
        csv_materializer(
            fields=[
                "slice_index",
                "object_label",
                "scale",
                "direction",
                "gray_levels",
                "angular_second_moment",
                "contrast",
                "correlation",
                "variance",
                "inverse_difference_moment",
                "sum_average",
                "sum_variance",
                "sum_entropy",
                "entropy",
                "difference_variance",
                "difference_entropy",
                "info_meas1",
                "info_meas2",
                "source_image_name",
            ],
            analysis_type="object_texture",
        ),
    )
)
def measure_texture_objects(
    image: np.ndarray,
    labels: np.ndarray,
    scale: int | tuple[int, ...] | list[int] = 3,
    gray_levels: int = 256,
    texture_crop_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    haralick_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int = 0,
) -> tuple[np.ndarray, list[ObjectTextureMeasurement]]:
    """Measure Haralick texture features for each labeled object."""
    image_array = np.asarray(image)
    original_labels = labels
    if image_array.ndim == 2 and slice_index == 0:
        plane_domain_stack = DenseObjectLabelPlaneDomainStackRequest(
            labels,
            dtype=np.int32,
        ).stack()
        if plane_domain_stack is not None:
            measurements: list[ObjectTextureMeasurement] = []
            for plane_index in range(plane_domain_stack.plane_count):
                _image, plane_measurements = measure_texture_objects.__wrapped__(
                    image,
                    plane_domain_stack.plane(plane_index),
                    scale=scale,
                    gray_levels=gray_levels,
                    texture_crop_backend_provider=texture_crop_backend_provider,
                    haralick_backend_provider=haralick_backend_provider,
                    slice_index=plane_index,
                )
                measurements.extend(plane_measurements)
            return image, measurements

    gray_levels = _normalize_gray_levels(gray_levels)
    pixel_data = CellProfilerTexturePixelDataRequest(
        image=image,
        gray_levels=gray_levels,
    ).pixel_data()
    crop_backend = ObjectTextureCropBackendStrategy.for_callable(
        measure_texture_objects,
        backend_provider=texture_crop_backend_provider,
    )

    measurements = []
    label_projection = TextureLabelSliceProjection.from_source(
        labels,
        np.asarray(image),
        slice_index,
    )
    labels_2d = label_projection.labels_2d()
    if labels_2d is not None:
        labels = labels_2d
    object_labels, intensity_crops = crop_backend.object_intensity_crops(
        pixel_data,
        labels,
    )
    if object_labels.size == 0:
        return image, ObjectTextureMeasurementCompletionRequest(
            measurements=tuple(measurements),
            labels=original_labels,
            scale=scale,
            gray_levels=gray_levels,
        ).complete()

    for object_label, label_data in zip(object_labels, intensity_crops, strict=True):
        for texture_scale in _texture_scales(scale):
            feature_matrix = HaralickFeatureMatrixRequest(
                pixel_data=label_data,
                scale=texture_scale,
                ignore_zeros=True,
                backend_provider=haralick_backend_provider,
            ).feature_matrix()

            for direction in range(N_DIRECTIONS_2D):
                axis = TextureMeasurementAxis(
                    slice_index=slice_index,
                    scale=texture_scale,
                    direction=direction,
                    gray_levels=gray_levels,
                )
                measurements.append(
                    HaralickFeatureVector.from_matrix(
                        feature_matrix,
                        direction,
                    ).object_measurement(
                        axis,
                        object_label=int(object_label),
                    )
                )

    return image, ObjectTextureMeasurementCompletionRequest(
        measurements=tuple(measurements),
        labels=original_labels,
        scale=scale,
        gray_levels=gray_levels,
    ).complete()


def measure_texture_objects_batch(
    request: RuntimePure2DSliceBatchRequest,
) -> list[ObjectTextureResult]:
    """Measure per-slice object texture with labels projected to each image plane."""
    kwargs = request.kwargs
    if "labels" in kwargs:
        labels = kwargs["labels"]
    else:
        labels = None
    label_array = TextureDenseLabelArray.from_value(labels)
    results: list[ObjectTextureResult] = []
    for slice_index, slice_2d in enumerate(request.slices_2d):
        slice_kwargs = kwargs
        label_projection = TextureLabelSliceProjection(
            source=labels,
            dense_labels=label_array,
            slice_array=np.asarray(slice_2d),
            slice_index=slice_index,
        )
        labels_2d = label_projection.labels_2d()
        if labels_2d is not None:
            slice_kwargs = dict(kwargs)
            slice_kwargs["labels"] = label_projection.projected_payload(labels_2d)
        results.append(request.execute_one_with_kwargs(slice_index, slice_kwargs))
    return results


class TextureDenseLabelArray:
    """Dense label coercion for MeasureTexture object inputs."""

    @classmethod
    def from_value(cls, labels: TextureLabelSource) -> np.ndarray | None:
        if labels is None:
            return None
        if isinstance(labels, ObjectLabelValue):
            return object_label_dense_array(labels, dtype=np.int32)
        return np.asarray(labels)


@dataclass(frozen=True, slots=True)
class TextureLabelSliceProjection:
    """Project object labels onto the image plane being texture-measured."""

    source: TextureLabelSource
    dense_labels: np.ndarray | None
    slice_array: np.ndarray
    slice_index: int

    @classmethod
    def from_source(
        cls,
        source: TextureLabelSource,
        slice_array: np.ndarray,
        slice_index: int,
    ) -> "TextureLabelSliceProjection":
        return cls(
            source=source,
            dense_labels=TextureDenseLabelArray.from_value(source),
            slice_array=slice_array,
            slice_index=slice_index,
        )

    def labels_2d(self) -> np.ndarray | None:
        if self.dense_labels is None or self.slice_array.ndim != 2:
            return None
        selected = self.dense_labels
        while (
            selected.ndim > 2
            and selected.shape[-2:] == self.slice_array.shape
            and selected.shape[0] > 0
        ):
            selected = selected[min(self.slice_index, selected.shape[0] - 1)]
        if selected.ndim == 2 and selected.shape == self.slice_array.shape:
            return np.asarray(selected, dtype=np.int32)
        return None

    def projected_payload(self, labels_2d: np.ndarray) -> TextureLabelSource:
        return ObjectLabelMeasurementPayloadStrategy.for_source(
            self.source
        ).materialize(
            self.source,
            ObjectLabelSourcePlaneProjectionRequest(labels_2d, self.slice_index),
        )


pure_2d_batch_executor(measure_texture_objects_batch)(measure_texture_objects)


def _prepare_measure_texture() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    measure_texture.__wrapped__(image)


def _prepare_measure_texture_objects() -> None:
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape((32, 32))
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[8:24, 8:24] = 1
    measure_texture_objects.__wrapped__(image, labels)


measure_texture.__openhcs_prepare__ = _prepare_measure_texture
measure_texture_objects.__openhcs_prepare__ = _prepare_measure_texture_objects


@njit(cache=True)
def _max_value_2d_numba(values: np.ndarray) -> int:
    height, width = values.shape
    max_value = 0
    for y in range(height):
        for x in range(width):
            value = values[y, x]
            if value > max_value:
                max_value = value
    return max_value


@njit(cache=True)
def _object_bounding_boxes_numba(
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = labels.shape
    max_label = _max_value_2d_numba(labels)

    min_y = np.full(max_label + 1, height, dtype=np.int64)
    min_x = np.full(max_label + 1, width, dtype=np.int64)
    max_y = np.full(max_label + 1, -1, dtype=np.int64)
    max_x = np.full(max_label + 1, -1, dtype=np.int64)
    for y in range(height):
        for x in range(width):
            label = labels[y, x]
            if label <= 0:
                continue
            if y < min_y[label]:
                min_y[label] = y
            if x < min_x[label]:
                min_x[label] = x
            if y > max_y[label]:
                max_y[label] = y
            if x > max_x[label]:
                max_x[label] = x

    object_count = 0
    for label in range(1, max_label + 1):
        if max_y[label] >= 0:
            object_count += 1

    object_labels = np.empty(object_count, dtype=np.int64)
    boxes = np.empty((object_count, 4), dtype=np.int64)
    index = 0
    for label in range(1, max_label + 1):
        if max_y[label] < 0:
            continue
        object_labels[index] = label
        boxes[index, 0] = min_y[label]
        boxes[index, 1] = max_y[label] + 1
        boxes[index, 2] = min_x[label]
        boxes[index, 3] = max_x[label] + 1
        index += 1
    return object_labels, boxes


@njit(cache=True)
def _haralick_2d_features_numba(
    image: np.ndarray,
    distance: int,
    ignore_zeros: bool,
) -> np.ndarray:
    height, width = image.shape
    max_value = _max_value_2d_numba(image)

    gray_count = max_value + 1
    features = np.zeros((4, 13), dtype=np.float64)
    deltas_y = np.array((0, 1, 1, 1), dtype=np.int64)
    deltas_x = np.array((1, 1, 0, -1), dtype=np.int64)

    for direction in range(4):
        cmat = np.zeros((gray_count, gray_count), dtype=np.float64)
        dy = deltas_y[direction] * distance
        dx = deltas_x[direction] * distance
        for y in range(height):
            yy = y + dy
            if yy < 0 or yy >= height:
                continue
            for x in range(width):
                xx = x + dx
                if xx < 0 or xx >= width:
                    continue
                a = image[y, x]
                b = image[yy, xx]
                if ignore_zeros and (a == 0 or b == 0):
                    continue
                cmat[a, b] += 1.0
                cmat[b, a] += 1.0

        total = cmat.sum()
        if total == 0.0:
            continue
        features[direction, :] = _haralick_features_from_cmat_numba(cmat, total)
    return features


@njit(cache=True)
def _haralick_features_from_cmat_numba(
    cmat: np.ndarray,
    total: float,
) -> np.ndarray:
    gray_count = cmat.shape[0]
    feats = np.zeros(13, dtype=np.float64)
    px = np.zeros(gray_count, dtype=np.float64)
    py = np.zeros(gray_count, dtype=np.float64)
    px_plus_y = np.zeros(gray_count * 2, dtype=np.float64)
    px_minus_y = np.zeros(gray_count, dtype=np.float64)

    for i in range(gray_count):
        for j in range(gray_count):
            p = cmat[i, j] / total
            px[j] += p
            py[i] += p
            px_plus_y[i + j] += p
            diff = i - j
            if diff < 0:
                diff = -diff
            px_minus_y[diff] += p
            feats[0] += p * p
            feats[1] += diff * diff * p
            feats[4] += p / (1.0 + diff * diff)

    ux = 0.0
    uy = 0.0
    for k in range(gray_count):
        ux += px[k] * k
        uy += py[k] * k

    vx = 0.0
    vy = 0.0
    for k in range(gray_count):
        vx += px[k] * k * k
        vy += py[k] * k * k
    vx -= ux * ux
    vy -= uy * uy

    sx = np.sqrt(vx)
    sy = np.sqrt(vy)
    if sx == 0.0 or sy == 0.0:
        feats[2] = 1.0
    else:
        ijp = 0.0
        for i in range(gray_count):
            for j in range(gray_count):
                ijp += i * j * (cmat[i, j] / total)
        feats[2] = (ijp - ux * uy) / (sx * sy)

    feats[3] = vx
    sum_average = 0.0
    sum_second = 0.0
    for k in range(gray_count * 2):
        sum_average += k * px_plus_y[k]
        sum_second += k * k * px_plus_y[k]
    feats[5] = sum_average
    feats[7] = _entropy_numba(px_plus_y)
    feats[6] = sum_second - sum_average * sum_average
    feats[8] = _entropy_matrix_numba(cmat, total)

    mean_minus = 0.0
    for k in range(gray_count):
        mean_minus += px_minus_y[k]
    mean_minus /= gray_count
    variance_minus = 0.0
    for k in range(gray_count):
        delta = px_minus_y[k] - mean_minus
        variance_minus += delta * delta
    feats[9] = variance_minus / gray_count
    feats[10] = _entropy_numba(px_minus_y)

    hx = _entropy_numba(px)
    hy = _entropy_numba(py)
    hxy1 = 0.0
    hxy2 = 0.0
    for i in range(gray_count):
        for j in range(gray_count):
            p = cmat[i, j] / total
            cross = py[i] * px[j]
            if cross > 0.0 and p > 0.0:
                hxy1 -= p * np.log2(cross)
            if cross > 0.0:
                hxy2 -= cross * np.log2(cross)

    if hx >= hy:
        max_h = hx
    else:
        max_h = hy
    if max_h == 0.0:
        feats[11] = feats[8] - hxy1
    else:
        feats[11] = (feats[8] - hxy1) / max_h
    info2 = 1.0 - np.exp(-2.0 * (hxy2 - feats[8]))
    if info2 < 0.0:
        info2 = 0.0
    feats[12] = np.sqrt(info2)
    return feats


@njit(cache=True)
def _entropy_numba(values: np.ndarray) -> float:
    result = 0.0
    for value in values:
        if value > 0.0:
            result -= value * np.log2(value)
    return result


@njit(cache=True)
def _entropy_matrix_numba(cmat: np.ndarray, total: float) -> float:
    result = 0.0
    height, width = cmat.shape
    for y in range(height):
        for x in range(width):
            p = cmat[y, x] / total
            if p > 0.0:
                result -= p * np.log2(p)
    return result


__all__ = public_names_from_objects(
    CellProfilerTexturePixelDataRequest,
    HaralickFeatureMatrixRequest,
    HaralickTextureBackendStrategy,
    NativeNumpyHaralickTextureBackendStrategy,
    NumbaNumpyHaralickTextureBackendStrategy,
    NumbaNumpyObjectTextureCropBackendStrategy,
    ObjectTextureCropBackendStrategy,
    ObjectTextureMeasurement,
    TextureMeasurement,
    measure_texture,
    measure_texture_objects,
    extra_names=(
        "F_HARALICK",
        "N_DIRECTIONS_2D",
        "ObjectIntensityCrops",
    ),
)
