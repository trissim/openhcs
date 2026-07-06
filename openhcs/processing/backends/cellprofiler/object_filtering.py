"""Object-filtering semantics for CellProfiler-compatible processing."""

from __future__ import annotations
from openhcs.interop.cellprofiler.measurement_lookup import (
    CellProfilerMeasurementFeature,
)
from openhcs.interop.cellprofiler.setting_names import (
    RepeatedSettingSequence,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
)
from openhcs.interop.cellprofiler.settings_binder import parse_cellprofiler_bool
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    measurement_values_for_feature,
    normalize_measurement_token,
    ordered_measurement_feature_candidates,
)
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRef,
    GroupLineageSourceRelation,
    ImageArtifactType,
    ObjectLabelsArtifactType,
    MeasurementsArtifactType,
    RelationshipsArtifactType,
)
from openhcs.core.runtime_semantics import (
    DenseObjectLabelExtentDomainDeclaration,
    DenseObjectLabelPairAligner,
    DenseObjectLabelStack,
    ObjectLabelIdDomainStrategy,
    ObjectLabelMeasurementValues,
    ParentChildRelationshipPayload,
    parent_child_relationship_artifact_name,
)
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.backends.analysis.region_properties import (
    LabelRegionPropertiesBackendStrategy,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.cellprofiler.shape import (
    MeasureObjectSizeShapeModule,
    form_factor_values,
)
from openhcs.processing.backends.cellprofiler.relationships import (
    ObjectRelationshipBackendStrategy,
)
from openhcs.processing.materialization import csv_materializer, segmentation_mask_rois
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    ImageArtifactOutputCapability,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputCapability,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    ObjectLabelArtifactInputCapability,
    ObjectLabelArtifactOutputCapability,
    ObjectArtifactOutputModule,
    PlaneRuntimeArtifactModule,
    PriorMeasurementArtifactInputModule,
    RelationshipArtifactInputCapability,
    RelationshipArtifactInputModule,
    RelationshipArtifactOutputCapability,
    RelationshipArtifactOutputModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectLabelPayload,
    ObjectLabelValue,
    ObjectRelationship,
    object_label_dense_array,
    object_label_value_with_dense_labels,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    ObjectRowsWithMeasurementsInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    RuntimeBoundParameterName,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectMeasurementVectorBinding,
    ObjectInputBindingRequest,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_tables import (
    object_measurement_tables_for_object,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargDict,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)

_FILTER_OBJECTS_ADDITIONAL_OBJECT_COUNT_KWARG = "additional_object_count"
_FILTER_OBJECTS_ENCLOSING_OBJECT_NAME_KWARG = "enclosing_object_name"
_FILTER_OBJECTS_MEASUREMENT_FEATURES_KWARG = "measurement_features"


@dataclass(frozen=True, slots=True)
class FilterObjectsKwargSettings:
    """Typed FilterObjects settings projected from CellProfiler kwargs."""

    additional_object_count: int
    enclosing_object_name: str | None
    measurement_features: tuple[str, ...]

    @classmethod
    def from_kwargs(cls, kwargs: CellProfilerKwargs) -> "FilterObjectsKwargSettings":
        raw_additional_count = kwargs.get(_FILTER_OBJECTS_ADDITIONAL_OBJECT_COUNT_KWARG)
        if raw_additional_count is None:
            additional_object_count = 0
        else:
            additional_object_count = int(raw_additional_count)
        raw_enclosing_name = kwargs.get(_FILTER_OBJECTS_ENCLOSING_OBJECT_NAME_KWARG)
        if raw_enclosing_name is None:
            enclosing_object_name = None
        else:
            enclosing_object_name = str(raw_enclosing_name)
        raw_measurement_features = kwargs.get(
            _FILTER_OBJECTS_MEASUREMENT_FEATURES_KWARG
        )
        if raw_measurement_features is None:
            measurement_features = ()
        else:
            measurement_features = tuple(
                (str(value) for value in raw_measurement_features)
            )
        return cls(
            additional_object_count=additional_object_count,
            enclosing_object_name=enclosing_object_name,
            measurement_features=measurement_features,
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsRuntimeInputPlan:
    """Runtime object-label partition for one FilterObjects invocation."""

    measurement_tables_kwarg: ClassVar[RuntimeBoundParameterName] = (
        ObjectRowsWithMeasurementsInputPolicy.measurement_tables_kwarg
    )
    measurement_values_kwarg: ClassVar[RuntimeBoundParameterName] = (
        RuntimeBoundParameterName("measurement_values")
    )
    object_specs: tuple[ArtifactSpec, ...]
    enclosing_spec: ArtifactSpec | None
    settings: FilterObjectsKwargSettings
    relationship_spec: ArtifactSpec | None = None
    measurement_relationship_specs: tuple[ArtifactSpec, ...] = ()

    @classmethod
    def from_inputs(
        cls, runtime_inputs: tuple[ArtifactSpec, ...], kwargs: CellProfilerKwargs
    ) -> "FilterObjectsRuntimeInputPlan":
        object_inputs = ArtifactSpecCollection(runtime_inputs).of_artifact_type(
            ObjectLabelsArtifactType
        )
        settings = FilterObjectsKwargSettings.from_kwargs(kwargs)
        object_count = settings.additional_object_count + 1
        enclosing_name = settings.enclosing_object_name
        object_specs = object_inputs[:object_count]
        enclosing_spec = None
        relationship_spec = None
        measurement_relationship_specs: list[ArtifactSpec] = []
        if enclosing_name is not None:
            enclosing_spec = ArtifactSpecCollection(object_inputs).by_name(
                enclosing_name
            )
            if enclosing_spec is None:
                raise RuntimeError(
                    f"FilterObjects enclosing object input {enclosing_name!r} was not declared in the runtime contract."
                )
            if object_specs:
                relationship_spec = ArtifactSpecCollection(
                    runtime_inputs
                ).by_name_and_artifact_type(
                    parent_child_relationship_artifact_name(
                        enclosing_name, object_specs[0].name
                    ),
                    RelationshipsArtifactType,
                )
        if object_specs:
            for (
                child_object_name
            ) in CellProfilerMeasurementFeature.child_count_object_names(
                settings.measurement_features
            ):
                relationship = ArtifactSpecCollection(
                    runtime_inputs
                ).by_name_and_artifact_type(
                    parent_child_relationship_artifact_name(
                        object_specs[0].name, child_object_name
                    ),
                    RelationshipsArtifactType,
                )
                if relationship is not None:
                    measurement_relationship_specs.append(relationship)
        return cls(
            object_specs=object_specs,
            enclosing_spec=enclosing_spec,
            settings=settings,
            relationship_spec=relationship_spec,
            measurement_relationship_specs=ArtifactSpecCollection(
                measurement_relationship_specs
            ).unique(conflict_context="CellProfiler input spec"),
        )

    @property
    def primary_object_spec(self) -> ArtifactSpec | None:
        if not self.object_specs:
            return None
        return self.object_specs[0]

    def bind_measurement_inputs(
        self, request: ObjectInputBindingRequest
    ) -> CellProfilerKwargDict:
        """Return FilterObjects measurement bindings owned by this runtime plan."""
        scoped_request = request.with_object_inputs(self.object_specs)
        measurement_values = self.measurement_vector(scoped_request)
        if measurement_values is not None:
            return {self.measurement_values_kwarg: measurement_values}
        measurement_tables = self.measurement_tables(scoped_request)
        if measurement_tables is None:
            return {}
        return {self.measurement_tables_kwarg: measurement_tables}

    def measurement_vector(
        self, request: ObjectInputBindingRequest
    ) -> CellProfilerRuntimeValue | None:
        object_spec = self.primary_object_spec
        if object_spec is None:
            return None
        feature_names = self.settings.measurement_features
        if len(feature_names) != 1:
            return None
        feature_name = str(feature_names[0])
        labels = request.labels_for(object_spec)
        return (
            CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=feature_name,
                labels=labels,
            )
            .vector()
            .slice_aligned_value
        )

    def measurement_tables(
        self, request: ObjectInputBindingRequest
    ) -> tuple[MeasurementTable, ...] | None:
        object_spec = self.primary_object_spec
        if object_spec is None:
            return None
        feature_names = self.settings.measurement_features
        if not feature_names:
            return None
        labels = request.labels_for(object_spec)
        tables_by_identity: dict[int, MeasurementTable] = {}
        for feature_name in feature_names:
            binding = CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=str(feature_name),
                labels=labels,
            )
            tables = binding.measurement_tables(request.adapter, match_group=False)
            for table in tables:
                table_identity = id(table)
                if table_identity not in tables_by_identity:
                    tables_by_identity[table_identity] = table
        if not tables_by_identity:
            return object_measurement_tables_for_object(
                request.adapter, object_spec.name
            )
        return tuple(tables_by_identity.values())


@dataclass(frozen=True, slots=True)
class FilterObjectsBoundMeasurementInputs:
    """Measurement binding profile for FilterObjects logging."""

    bound: CellProfilerKwargs

    @property
    def measurement_tables(self) -> tuple[MeasurementTable, ...]:
        value = self.bound.get(FilterObjectsRuntimeInputPlan.measurement_tables_kwarg)
        if value is None:
            return ()
        return tuple(value)

    @property
    def measurement_values(self) -> CellProfilerRuntimeValue | None:
        return self.bound.get(FilterObjectsRuntimeInputPlan.measurement_values_kwarg)

    @property
    def measurement_values_type(self) -> str:
        measurement_values = self.measurement_values
        if measurement_values is None:
            return "none"
        return type(measurement_values).__name__


class FilterObjectsInputPolicy(ObjectRowsWithMeasurementsInputPolicy):
    """Bind ordered primary/additional object rows for FilterObjects."""

    supported_non_object_input_kinds = frozenset(
        {MeasurementsArtifactType, RelationshipsArtifactType}
    )

    def bind(self, request: ObjectInputBindingRequest) -> CellProfilerKwargDict:
        runtime_inputs = request.runtime_inputs
        if not runtime_inputs:
            runtime_inputs = request.object_inputs
        plan = FilterObjectsRuntimeInputPlan.from_inputs(runtime_inputs, request.kwargs)
        bound = super().bind(request.with_object_inputs(plan.object_specs))
        bound.update(plan.bind_measurement_inputs(request))
        bound_measurements = FilterObjectsBoundMeasurementInputs(bound)
        measurement_values = bound_measurements.measurement_values
        CellProfilerRuntimeProfileLogger.log_module_profile(
            "filterobjects_bound_measurements",
            0.0,
            module=request.module_name,
            table_count=len(bound_measurements.measurement_tables),
            has_measurement_values=measurement_values is not None,
            measurement_values_type=bound_measurements.measurement_values_type,
            measurement_features=len(plan.settings.measurement_features),
        )
        if plan.enclosing_spec is not None:
            bound["enclosing_object_labels"] = request.labels_for(plan.enclosing_spec)
        if plan.relationship_spec is not None:
            bound["parent_child_relationship"] = request.current_plane_relationship_for(
                plan.relationship_spec
            )
        if plan.measurement_relationship_specs:
            bound["parent_child_relationships"] = tuple(
                (
                    request.current_plane_relationship_for(relationship_spec)
                    for relationship_spec in plan.measurement_relationship_specs
                )
            )
        return bound


class FilterObjectsModule(
    PlaneRuntimeArtifactModule,
    FilterObjectsInputPolicy,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    RelationshipArtifactInputModule,
    RelationshipArtifactOutputModule,
    ImageArtifactOutputModule,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
):
    module_name = "FilterObjects"
    function_name = "filter_objects"
    validated = True
    confidence = 1.0
    input_setting = SettingNameFamily(
        "Select the object to filter",
        aliases=("Select the objects to filter", "Select the input objects"),
    )
    output_setting = "Name the output objects"
    mode_setting = SettingNameFamily(
        "Filter using classifier rules or measurements?",
        aliases=("Select the filtering mode",),
    )
    method_setting = "Select the filtering method"
    measurement_setting = "Select the measurement to filter by"
    use_minimum_setting = "Filter using a minimum measurement value?"
    minimum_setting = "Minimum value"
    use_maximum_setting = "Filter using a maximum measurement value?"
    maximum_setting = "Maximum value"
    main_outline_setting = "Retain the outlines of filtered objects for use later in the pipeline (for example, in SaveImages)?"
    outline_image_setting = "Name the outline image"
    additional_input_setting = "Select additional object to relabel"
    additional_output_setting = "Name the relabeled objects"
    additional_outline_setting = "Save outlines of relabeled objects?"
    enclosing_object_setting = "Select the objects that contain the filtered objects"
    per_object_assignment_setting = "Assign overlapping child to"

    @dataclass(frozen=True, slots=True)
    class SymbolRequirement:
        """Fail-loud FilterObjects symbol-setting validation."""

        value: str
        setting_name: str | SettingNameFamily

        def validate(self, module: "ModuleBlock") -> None:
            if normalized_symbol_name(self.value) is not None:
                return
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an empty FilterObjects symbol in setting {self.setting_name!r}."
            )

    @dataclass(frozen=True, slots=True)
    class ObjectPair:
        """Shared input/output object-name pair for FilterObjects rows."""

        input_object_name: str
        output_object_name: str

    @dataclass(frozen=True, slots=True)
    class AdditionalObjectRow(ObjectPair):
        """One additional object set relabeled using the primary filter mask."""

        retain_outline: bool = False
        outline_image_name: str | None = None

        @classmethod
        def from_block(
            cls, module: "ModuleBlock", block: Sequence["ModuleSetting"]
        ) -> "FilterObjectsModule.AdditionalObjectRow":
            return cls(
                input_object_name=block_setting_value(
                    block, FilterObjectsModule.additional_input_setting
                ),
                output_object_name=block_setting_value(
                    block, FilterObjectsModule.additional_output_setting
                ),
                retain_outline=parse_cellprofiler_bool(
                    block_setting_value(
                        block,
                        FilterObjectsModule.additional_outline_setting,
                        default="No",
                    )
                ),
                outline_image_name=normalized_symbol_name(
                    block_setting_value(
                        block, FilterObjectsModule.outline_image_setting
                    )
                ),
            ).validated(module)

        def validated(
            self, module: "ModuleBlock"
        ) -> "FilterObjectsModule.AdditionalObjectRow":
            FilterObjectsModule.SymbolRequirement(
                self.input_object_name, FilterObjectsModule.additional_input_setting
            ).validate(module)
            FilterObjectsModule.SymbolRequirement(
                self.output_object_name, FilterObjectsModule.additional_output_setting
            ).validate(module)
            if self.retain_outline and self.outline_image_name is None:
                raise ValueError(
                    f"Module {module.name}({module.module_num}) retains an additional FilterObjects outline without an outline image name."
                )
            return self

    @dataclass(frozen=True, slots=True)
    class MeasurementRule:
        """One measurement limit rule used by FilterObjects."""

        feature_name: str
        use_minimum: bool
        min_value: float | None
        use_maximum: bool
        max_value: float | None

        @classmethod
        def from_block(
            cls, module: "ModuleBlock", block: Sequence["ModuleSetting"]
        ) -> "FilterObjectsModule.MeasurementRule":
            return cls(
                feature_name=block_setting_value(
                    block, FilterObjectsModule.measurement_setting
                ),
                use_minimum=parse_cellprofiler_bool(
                    block_setting_value(
                        block, FilterObjectsModule.use_minimum_setting, default="No"
                    )
                ),
                min_value=FilterObjectsModule.optional_float(
                    block_setting_value(block, FilterObjectsModule.minimum_setting)
                ),
                use_maximum=parse_cellprofiler_bool(
                    block_setting_value(
                        block, FilterObjectsModule.use_maximum_setting, default="No"
                    )
                ),
                max_value=FilterObjectsModule.optional_float(
                    block_setting_value(block, FilterObjectsModule.maximum_setting)
                ),
            ).validated(module)

        def validated(
            self, module: "ModuleBlock"
        ) -> "FilterObjectsModule.MeasurementRule":
            if self.feature_name.strip():
                return self
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an empty FilterObjects measurement rule."
            )

    @dataclass(frozen=True, slots=True)
    class Plan(ObjectPair):
        """Complete typed FilterObjects artifact and runtime plan."""

        retain_outline: bool
        outline_image_name: str | None
        additional_rows: tuple["FilterObjectsModule.AdditionalObjectRow", ...]
        enclosing_object_name: str | None
        per_object_assignment: str

        @property
        def input_object_names(self) -> tuple[str, ...]:
            ordered_names = (
                *(pair.input_object_name for pair in self.object_pairs),
                *(
                    ()
                    if self.enclosing_object_name is None
                    else (self.enclosing_object_name,)
                ),
            )
            return tuple(dict.fromkeys(ordered_names))

        @property
        def object_pairs(self) -> tuple["FilterObjectsModule.ObjectPair", ...]:
            return (self, *self.additional_rows)

        def output_artifact_specs(
            self,
            measurement_artifact_name: str,
        ) -> tuple[ArtifactSpec, ...]:
            object_sources = tuple(
                (
                    (
                        pair,
                        ArtifactSpecRef.input(
                            pair.input_object_name, ObjectLabelsArtifactType
                        ),
                    )
                    for pair in self.object_pairs
                )
            )
            return (
                MeasurementArtifactOutputCapability.spec(
                    measurement_artifact_name,
                    relations=(
                        GroupLineageSourceRelation(source=object_sources[0][1]),
                    ),
                ),
                *(
                    ObjectLabelArtifactOutputCapability.spec(
                        pair.output_object_name,
                        relations=(GroupLineageSourceRelation(source=source),),
                    )
                    for pair, source in object_sources
                ),
                *(
                    RelationshipArtifactOutputCapability.spec(
                        parent_child_relationship_artifact_name(
                            pair.input_object_name, pair.output_object_name
                        ),
                        relations=(GroupLineageSourceRelation(source=source),),
                    )
                    for pair, source in object_sources
                ),
                *(
                    ImageArtifactOutputCapability.spec(
                        outline_name,
                        relations=(
                            GroupLineageSourceRelation(
                                source=ArtifactSpecRef.input(
                                    input_name,
                                    ObjectLabelsArtifactType,
                                )
                            ),
                        ),
                    )
                    for outline_name, input_name in self.outline_source_pairs()
                ),
            )

        @property
        def outline_image_names(self) -> tuple[str, ...]:
            names: list[str] = []
            if self.retain_outline:
                if self.outline_image_name is None:
                    raise RuntimeError("FilterObjects retained outline has no name.")
                names.append(self.outline_image_name)
            names.extend(
                (
                    row.outline_image_name
                    for row in self.additional_rows
                    if row.retain_outline and row.outline_image_name is not None
                )
            )
            return tuple(names)

        def outline_source_pairs(self) -> tuple[tuple[str, str], ...]:
            pairs: list[tuple[str, str]] = []
            if self.retain_outline:
                if self.outline_image_name is None:
                    raise RuntimeError("FilterObjects retained outline has no name.")
                pairs.append((self.outline_image_name, self.input_object_name))
            pairs.extend(
                (
                    (row.outline_image_name, row.input_object_name)
                    for row in self.additional_rows
                    if row.retain_outline and row.outline_image_name is not None
                )
            )
            return tuple(pairs)

        @property
        def outline_object_indices(self) -> tuple[int, ...]:
            indices: list[int] = []
            if self.retain_outline:
                indices.append(0)
            indices.extend(
                (
                    index
                    for index, row in enumerate(self.additional_rows, start=1)
                    if row.retain_outline
                )
            )
            return tuple(indices)

    @classmethod
    def settings_source(cls, module: "ModuleBlock") -> "CellProfilerKwargs":
        """Return absorbed-function kwargs for a typed FilterObjects plan."""
        plan = cls.plan(module)
        measurement_rules = cls.measurement_rules(module)
        return {
            "mode": cls.mode_value(module),
            "filter_method": optional_setting_value(module, cls.method_setting)
            or "Limits",
            "measurement_features": tuple(
                (rule.feature_name for rule in measurement_rules)
            ),
            "measurement_min_values": tuple(
                (rule.min_value for rule in measurement_rules)
            ),
            "measurement_max_values": tuple(
                (rule.max_value for rule in measurement_rules)
            ),
            "measurement_use_minimum": tuple(
                (rule.use_minimum for rule in measurement_rules)
            ),
            "measurement_use_maximum": tuple(
                (rule.use_maximum for rule in measurement_rules)
            ),
            "additional_object_count": len(plan.additional_rows),
            "outline_object_indices": plan.outline_object_indices,
            "enclosing_object_name": plan.enclosing_object_name,
            "per_object_assignment": plan.per_object_assignment,
        }

    @classmethod
    def plan(cls, module: "ModuleBlock") -> "FilterObjectsModule.Plan":
        """Return the typed FilterObjects compile/runtime plan."""
        plan = cls.Plan(
            input_object_name=required_setting_value(module, cls.input_setting),
            output_object_name=required_setting_value(module, cls.output_setting),
            retain_outline=parse_cellprofiler_bool(
                optional_setting_value(module, cls.main_outline_setting) or "No"
            ),
            outline_image_name=cls.main_outline_image_name(module),
            additional_rows=cls.additional_rows(module),
            enclosing_object_name=normalized_symbol_name(
                optional_setting_value(module, cls.enclosing_object_setting) or ""
            ),
            per_object_assignment=optional_setting_value(
                module, cls.per_object_assignment_setting
            )
            or "Both parents",
        )
        cls.SymbolRequirement(plan.input_object_name, cls.input_setting).validate(
            module
        )
        cls.SymbolRequirement(plan.output_object_name, cls.output_setting).validate(
            module
        )
        if plan.retain_outline and plan.outline_image_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) retains filtered-object outlines without an outline image name."
            )
        return plan

    @classmethod
    def additional_rows(
        cls, module: "ModuleBlock"
    ) -> tuple["FilterObjectsModule.AdditionalObjectRow", ...]:
        """Return ordered additional relabel rows from parsed settings."""
        if module.iter_settings():
            blocks = repeating_setting_blocks(
                module.iter_settings(), start_name=cls.additional_input_setting
            )
            return tuple(
                (cls.AdditionalObjectRow.from_block(module, block) for block in blocks)
            )
        input_names = setting_values(module, cls.additional_input_setting)
        output_names = setting_values(module, cls.additional_output_setting)
        outline_flags = setting_values(module, cls.additional_outline_setting)
        outline_names = setting_values(module, cls.outline_image_setting)[1:]
        row_count = max(len(input_names), len(output_names), len(outline_flags))
        return tuple(
            (
                cls.AdditionalObjectRow(
                    input_object_name=RepeatedSettingSequence(input_names).at(index),
                    output_object_name=RepeatedSettingSequence(output_names).at(index),
                    retain_outline=parse_cellprofiler_bool(
                        RepeatedSettingSequence(outline_flags, default="No").at(index)
                    ),
                    outline_image_name=normalized_symbol_name(
                        RepeatedSettingSequence(outline_names).at(index)
                    ),
                ).validated(module)
                for index in range(row_count)
            )
        )

    @classmethod
    def measurement_rules(
        cls, module: "ModuleBlock"
    ) -> tuple["FilterObjectsModule.MeasurementRule", ...]:
        """Return ordered measurement limit rules from parsed settings."""
        if module.iter_settings():
            blocks = repeating_setting_blocks(
                module.iter_settings(), start_name=cls.measurement_setting
            )
            return tuple(
                (cls.MeasurementRule.from_block(module, block) for block in blocks)
            )
        feature_names = setting_values(module, cls.measurement_setting)
        use_minimum = setting_values(module, cls.use_minimum_setting)
        min_values = setting_values(module, cls.minimum_setting)
        use_maximum = setting_values(module, cls.use_maximum_setting)
        max_values = setting_values(module, cls.maximum_setting)
        return tuple(
            (
                cls.MeasurementRule(
                    feature_name=RepeatedSettingSequence(feature_names).at(index),
                    use_minimum=parse_cellprofiler_bool(
                        RepeatedSettingSequence(use_minimum, default="No").at(index)
                    ),
                    min_value=cls.optional_float(
                        RepeatedSettingSequence(min_values).at(index)
                    ),
                    use_maximum=parse_cellprofiler_bool(
                        RepeatedSettingSequence(use_maximum, default="No").at(index)
                    ),
                    max_value=cls.optional_float(
                        RepeatedSettingSequence(max_values).at(index)
                    ),
                ).validated(module)
                for index in range(len(feature_names))
            )
        )

    @classmethod
    def child_count_object_names(cls, module: "ModuleBlock") -> tuple[str, ...]:
        """Return child object names needed by Children_<object>_Count rules."""
        return CellProfilerMeasurementFeature.child_count_object_names(
            tuple((rule.feature_name for rule in cls.measurement_rules(module)))
        )

    @classmethod
    def main_outline_image_name(cls, module: "ModuleBlock") -> str | None:
        names = setting_values(module, cls.outline_image_setting)
        if not names:
            return None
        return normalized_symbol_name(names[0])

    @classmethod
    def mode_value(cls, module: "ModuleBlock") -> str:
        value = optional_setting_value(module, cls.mode_setting)
        if value is None:
            return "Measurements"
        if "border" in value.strip().lower():
            return "Border"
        return value

    @staticmethod
    def optional_float(raw_value: str | None) -> float | None:
        if raw_value is None:
            return None
        stripped = raw_value.strip()
        if not stripped:
            return None
        return float(stripped)

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        plan = cls.plan(module)
        inputs = [
            *(
                ObjectLabelArtifactInputCapability.bind_artifact(cls, builder, module, ObjectLabelArtifactInputCapability.spec(name))
                for name in plan.input_object_names
            ),
            *cls.prior_measurement_artifact_inputs(builder),
        ]
        if plan.enclosing_object_name is not None:
            relationship = builder.optional_artifact(
                RelationshipArtifactInputCapability.spec(parent_child_relationship_artifact_name(
                        plan.enclosing_object_name, plan.input_object_name
                    ))
            )
            if relationship is not None:
                inputs.append(relationship)
        for child_object_name in cls.child_count_object_names(module):
            relationship = builder.optional_artifact(
                RelationshipArtifactInputCapability.spec(parent_child_relationship_artifact_name(
                        plan.input_object_name, child_object_name
                    ))
            )
            if relationship is not None:
                inputs.append(relationship)
        outputs = [
            builder.declare_artifact(spec, module)
            for spec in plan.output_artifact_specs(cls.measurement_artifact_name(module))
        ]
        return assembler.assemble_contract(
            module, builder, inputs=inputs, outputs=outputs
        )


class FilterMethod(Enum):
    """CellProfiler FilterObjects measurement selection modes."""

    MINIMAL = "minimal"
    MAXIMAL = "maximal"
    MINIMAL_PER_OBJECT = "minimal_per_object"
    MAXIMAL_PER_OBJECT = "maximal_per_object"
    LIMITS = "limits"


class FilterMode(Enum):
    """CellProfiler FilterObjects top-level filter modes."""

    MEASUREMENTS = "measurements"
    BORDER = "border"


class PerObjectAssignment(Enum):
    """How per-object FilterObjects assigns child objects to parents."""

    BOTH_PARENTS = "both_parents"
    PARENT_WITH_MOST_OVERLAP = "parent_with_most_overlap"


FilterObjectsParentChildRelationship = (
    ObjectRelationship | ParentChildRelationshipPayload
)
FilterObjectsParentChildRelationships = tuple[FilterObjectsParentChildRelationship, ...]


@dataclass(frozen=True, slots=True)
class FilterObjectsRelationshipEndpointIds:
    """Dense integer ids carried by a FilterObjects relationship endpoint."""

    values: object

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple((int(value) for value in np.asarray(self.values).reshape(-1)))


@dataclass(frozen=True, slots=True)
class FilterObjectsLabelPlane:
    """Dense label plane with FilterObjects projection/alignment semantics."""

    labels: np.ndarray

    @property
    def projected(self) -> np.ndarray:
        labels = object_label_dense_array(self.labels, dtype=np.int32)
        return DenseObjectLabelStack.from_labels(
            labels
        ).project_xy_plane_without_relabeling()

    def aligned_to(self, reference_labels: np.ndarray) -> np.ndarray:
        _aligned_reference, aligned_labels = DenseObjectLabelPairAligner(
            reference_labels, object_label_dense_array(self.labels, dtype=np.int32)
        ).aligned()
        return aligned_labels.astype(np.int32, copy=False)

    @classmethod
    def optional_aligned_to(
        cls, reference_labels: np.ndarray, labels: np.ndarray | None
    ) -> np.ndarray | None:
        if labels is None:
            return None
        return cls(labels).aligned_to(reference_labels)


@dataclass(frozen=True, slots=True)
class FilterObjectsMeasurementLimitWindow:
    """Object-id retention policy for FilterObjects measurement bounds."""

    values: ObjectLabelMeasurementValues
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool

    @classmethod
    def from_label_indexed_values(
        cls,
        values: np.ndarray,
        *,
        min_value: float | None,
        max_value: float | None,
        use_minimum: bool,
        use_maximum: bool,
    ) -> "FilterObjectsMeasurementLimitWindow":
        object_ids = tuple(range(1, len(values) + 1))
        return cls(
            ObjectLabelMeasurementValues.from_label_indexed_values(object_ids, values),
            min_value=min_value,
            max_value=max_value,
            use_minimum=use_minimum,
            use_maximum=use_maximum,
        )

    @property
    def retained_ids(self) -> list[int]:
        return list(
            self.values.ids_within_limits(
                min_value=self.min_value,
                max_value=self.max_value,
                use_minimum=self.use_minimum,
                use_maximum=self.use_maximum,
            )
        )


@dataclass
class FilterObjectsStats:
    """FilterObjects object-count output row."""

    slice_index: int
    objects_pre_filter: int
    objects_post_filter: int
    objects_removed: int

    @classmethod
    def from_counts(
        cls, *, objects_pre_filter: int, objects_post_filter: int, slice_index: int = 0
    ) -> "FilterObjectsStats":
        return cls(
            slice_index=slice_index,
            objects_pre_filter=objects_pre_filter,
            objects_post_filter=objects_post_filter,
            objects_removed=objects_pre_filter - objects_post_filter,
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsRelabeledPlane:
    """Relabeled object plane plus the direct input-to-output relationship."""

    labels: np.ndarray
    relationship: ParentChildRelationshipPayload

    @classmethod
    def from_ordered_parent_ids(
        cls, labels: np.ndarray, *, parent_ids_by_child: Sequence[int]
    ) -> "FilterObjectsRelabeledPlane":
        """Relabel ``labels`` in child order and retain the exact parent mapping."""
        label_array = np.asarray(labels, dtype=np.int32)
        max_label = int(label_array.max()) if label_array.size else 0
        label_mapping = np.zeros(max_label + 1, dtype=np.int32)
        parent_ids: list[int] = []
        child_ids: list[int] = []
        for child_id, parent_id in enumerate(parent_ids_by_child, start=1):
            if 0 < parent_id <= max_label:
                label_mapping[int(parent_id)] = child_id
                parent_ids.append(int(parent_id))
                child_ids.append(child_id)
        return cls(
            labels=label_mapping[label_array],
            relationship=ParentChildRelationshipPayload(
                parent_ids=tuple(parent_ids), child_ids=tuple(child_ids)
            ),
        )

    @classmethod
    def from_retained_mask(
        cls, labels: np.ndarray, retained_mask: np.ndarray
    ) -> "FilterObjectsRelabeledPlane":
        """Relabel source IDs overlapping retained primary objects."""
        label_array = np.asarray(labels, dtype=np.int32)
        source_ids = np.unique(label_array[np.asarray(retained_mask, dtype=bool)])
        source_ids = source_ids[source_ids > 0]
        return cls.from_ordered_parent_ids(
            label_array,
            parent_ids_by_child=tuple((int(source_id) for source_id in source_ids)),
        )


@dataclass(frozen=True, slots=True)
class FilterObjectsSelectionRequest:
    """Inputs needed to choose retained primary object labels."""

    labels: np.ndarray
    object_ids: tuple[int, ...]
    filter_method: FilterMethod
    measurement_values: ObjectLabelMeasurementValues | None
    measurement_features: tuple[str, ...]
    measurement_min_values: tuple[float | None, ...]
    measurement_max_values: tuple[float | None, ...]
    measurement_use_minimum: tuple[bool, ...]
    measurement_use_maximum: tuple[bool, ...]
    measurement_tables: tuple[MeasurementTable, ...]
    enclosing_labels: np.ndarray | None
    parent_child_relationship: FilterObjectsParentChildRelationship | None
    parent_child_relationships: FilterObjectsParentChildRelationships
    per_object_assignment: PerObjectAssignment
    min_value: float | None
    max_value: float | None
    use_minimum: bool
    use_maximum: bool

    @property
    def num_objects_pre(self) -> int:
        return len(self.object_ids)

    def measurement_values_for_feature(
        self, feature_name: str
    ) -> ObjectLabelMeasurementValues:
        """Resolve one FilterObjects measurement rule through the nominal source chain."""
        if (
            self.measurement_values is not None
            and self.measurement_features
            and (feature_name == self.measurement_features[0])
        ):
            return self.measurement_values
        return FilterObjectsMeasurementValuesSource.resolve_feature(self, feature_name)

    def first_measurement_values(self) -> ObjectLabelMeasurementValues:
        if self.measurement_values is not None:
            return self.measurement_values
        if self.measurement_features:
            return self.measurement_values_for_feature(self.measurement_features[0])
        return self.area_measurement_values()

    def area_measurement_values(self) -> ObjectLabelMeasurementValues:
        return ObjectLabelMeasurementValues.from_label_indexed_values(
            self.object_ids,
            DerivedMeasurementValuesStrategy.for_enum_member(
                MeasureObjectSizeShapeModule.MeasurementFeature.AREA
            ).values(self.labels),
        )

    def matching_measurement_rule_ids(self) -> list[int]:
        self.validate_measurement_rule_lengths()
        retained_ids = set(self.object_ids)
        for index, feature_name in enumerate(self.measurement_features):
            keep_ids = FilterObjectsMeasurementLimitWindow(
                values=self.measurement_values_for_feature(feature_name),
                min_value=self.measurement_min_values[index],
                max_value=self.measurement_max_values[index],
                use_minimum=self.measurement_use_minimum[index],
                use_maximum=self.measurement_use_maximum[index],
            )
            retained_ids.intersection_update(keep_ids.retained_ids)
        return sorted(retained_ids)

    def validate_measurement_rule_lengths(self) -> None:
        expected = len(self.measurement_features)
        lengths = {
            len(self.measurement_min_values),
            len(self.measurement_max_values),
            len(self.measurement_use_minimum),
            len(self.measurement_use_maximum),
        }
        if lengths == {expected}:
            return
        raise ValueError("FilterObjects measurement rule kwargs must align by row.")


@dataclass(frozen=True, slots=True)
class FilterSelectionKey:
    """Nominal retained-object selection identity."""

    mode: FilterMode
    method: FilterMethod | None = None

    @property
    def label(self) -> str:
        if self.method is None:
            return self.mode.value
        return f"{self.mode.value}:{self.method.value}"

    def lookup_candidates(self) -> tuple["FilterSelectionKey", ...]:
        return (self, FilterSelectionKey(self.mode))


class DerivedMeasurementValuesStrategy(
    EnumKeyedStrategyMixin[MeasureObjectSizeShapeModule.MeasurementFeature],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Derived object-measurement values available from dense labels."""

    __registry_key__ = "strategy_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "feature"
    feature: ClassVar[MeasureObjectSizeShapeModule.MeasurementFeature]
    strategy_label: ClassVar[str | None] = None

    @classmethod
    def for_feature_name(
        cls, feature_name: str
    ) -> "DerivedMeasurementValuesStrategy | None":
        candidates = ordered_measurement_feature_candidates(
            feature_name, dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
        )
        strategies_by_feature_name = {
            normalize_measurement_token(strategy_type.feature.value): strategy_type
            for strategy_type in cls.registered_strategy_types()
        }
        for candidate in candidates:
            strategy_type = strategies_by_feature_name.get(candidate)
            if strategy_type is not None:
                return strategy_type()
        return None

    @abstractmethod
    def values(self, labels: np.ndarray) -> np.ndarray:
        """Return one derived value per object label."""


class AreaDerivedMeasurementValuesStrategy(DerivedMeasurementValuesStrategy):
    """Area is directly measurable from the label geometry."""

    feature = MeasureObjectSizeShapeModule.MeasurementFeature.AREA

    def values(self, labels: np.ndarray) -> np.ndarray:
        return (
            LabelRegionPropertiesBackendStrategy.for_memory_type()
            .measure_2d(labels.astype(np.int32, copy=False))
            .area
        )


class FormFactorDerivedMeasurementValuesStrategy(DerivedMeasurementValuesStrategy):
    """FormFactor can be derived from area/perimeter label geometry."""

    feature = MeasureObjectSizeShapeModule.MeasurementFeature.FORM_FACTOR
    minimum_value: ClassVar[float] = 0.0
    maximum_value: ClassVar[float] = 1.0

    def values(self, labels: np.ndarray) -> np.ndarray:
        label_ids = np.arange(1, int(labels.max()) + 1, dtype=np.int32)
        if label_ids.size == 0:
            return np.array([], dtype=float)
        values = form_factor_values(labels.astype(np.int32, copy=False), label_ids)
        return np.clip(values, type(self).minimum_value, type(self).maximum_value)


class FilterObjectsMeasurementValuesSource(ABC, metaclass=AutoRegisterMeta):
    """MRO-ordered FilterObjects measurement-value source chain."""

    __registry_key__ = "source_label"
    __skip_if_no_key__ = True
    source_label: ClassVar[str | None] = None

    @classmethod
    def active_source_type(cls) -> type["FilterObjectsMeasurementValuesSource"]:
        """Return the most-derived registered source; MRO defines precedence."""
        return max(
            cls.__registry__.values(), key=lambda source_type: len(source_type.__mro__)
        )

    @classmethod
    def resolve_feature(
        cls, request: FilterObjectsSelectionRequest, feature_name: str
    ) -> ObjectLabelMeasurementValues:
        if cls is FilterObjectsMeasurementValuesSource:
            return cls.active_source_type().resolve_feature(request, feature_name)
        values = measurement_values_for_feature(
            request.measurement_tables,
            feature_name,
            object_count=request.num_objects_pre,
            object_ids=request.object_ids,
            dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
        )
        return ObjectLabelMeasurementValues(request.object_ids, values)


class DerivedFilterObjectsMeasurementValuesSource(FilterObjectsMeasurementValuesSource):
    """Resolve label-intrinsic measurements before falling back to tables."""

    source_label = "derived"

    @classmethod
    def resolve_feature(
        cls, request: FilterObjectsSelectionRequest, feature_name: str
    ) -> ObjectLabelMeasurementValues:
        strategy = DerivedMeasurementValuesStrategy.for_feature_name(feature_name)
        if strategy is not None:
            return ObjectLabelMeasurementValues.from_label_indexed_values(
                request.object_ids, strategy.values(request.labels)
            )
        return super().resolve_feature(request, feature_name)


class TableFilterObjectsMeasurementValuesSource(
    DerivedFilterObjectsMeasurementValuesSource
):
    """Resolve explicit measurement tables before label-derived fallbacks."""

    source_label = "measurement_table"

    @classmethod
    def resolve_feature(
        cls, request: FilterObjectsSelectionRequest, feature_name: str
    ) -> ObjectLabelMeasurementValues:
        if request.measurement_tables:
            value_index = MeasurementFeatureQuery(
                feature_name, dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT
            ).optional_value_index(request.measurement_tables)
            if value_index is not None:
                values_by_label, positional_values = value_index
                if values_by_label:
                    return ObjectLabelMeasurementValues.from_value_mapping(
                        request.object_ids, values_by_label
                    )
                if positional_values:
                    return ObjectLabelMeasurementValues.from_positional_values(
                        request.object_ids, positional_values
                    )
        return super().resolve_feature(request, feature_name)


class RelationshipChildCountFilterObjectsMeasurementValuesSource(
    TableFilterObjectsMeasurementValuesSource
):
    """Resolve Children_* measurement rules from parent-child relationships."""

    source_label = "relationship_child_count"

    @classmethod
    def resolve_feature(
        cls, request: FilterObjectsSelectionRequest, feature_name: str
    ) -> ObjectLabelMeasurementValues:
        child_name = child_count_feature_child_name(feature_name)
        if child_name is not None:
            for relationship in request.parent_child_relationships:
                parent_ids = cls.parent_ids_for_child(relationship, child_name)
                if parent_ids is None:
                    continue
                counts_by_parent_id: dict[int, float] = {
                    object_id: 0.0 for object_id in request.object_ids
                }
                for parent_id in parent_ids:
                    if parent_id in counts_by_parent_id:
                        counts_by_parent_id[parent_id] += 1.0
                return ObjectLabelMeasurementValues.from_value_mapping(
                    request.object_ids, counts_by_parent_id
                )
        return super().resolve_feature(request, feature_name)

    @classmethod
    def parent_ids_for_child(
        cls, relationship: FilterObjectsParentChildRelationship, child_name: str
    ) -> tuple[int, ...] | None:
        if isinstance(relationship, ObjectRelationship):
            if relationship.target.name != child_name:
                return None
            return FilterObjectsRelationshipEndpointIds(relationship.source_ids).ids
        return FilterObjectsRelationshipEndpointIds(relationship.parent_ids).ids


class FilterSelectionStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal retained-object selection for each FilterObjects behavior."""

    __registry_key__ = "selection_label"
    __skip_if_no_key__ = True
    selection_label: ClassVar[str | None] = None
    selection_key: ClassVar[FilterSelectionKey | None] = None

    @classmethod
    def for_mode_and_method(
        cls, mode: FilterMode, method: FilterMethod
    ) -> "FilterSelectionStrategy":
        requested_key = FilterSelectionKey(mode, method)
        for key in requested_key.lookup_candidates():
            strategy_type = cls.__registry__.get(key.label)
            if strategy_type is not None:
                return strategy_type()
        raise ValueError(
            f"Unsupported FilterObjects selection {requested_key.label!r}."
        )

    @abstractmethod
    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        """Return one-indexed primary object labels to retain."""


class BorderFilterSelectionStrategy(FilterSelectionStrategy):
    """Remove primary objects touching the image border."""

    selection_key = FilterSelectionKey(FilterMode.BORDER)
    selection_label = selection_key.label

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        return self.discard_border_objects(request.labels)

    @staticmethod
    def discard_border_objects(labels: np.ndarray) -> list[int]:
        from scipy import ndimage as ndi

        interior_pixels = ndi.binary_erosion(np.ones_like(labels, dtype=bool))
        border_labels = set(labels[~interior_pixels])
        keep_labels = list(set(labels.ravel()).difference(border_labels))
        if 0 in keep_labels:
            keep_labels.remove(0)
        keep_labels.sort()
        return keep_labels


class LimitsFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep objects whose measurement falls within configured limits."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.LIMITS)
    selection_label = selection_key.label

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        if request.measurement_features:
            return request.matching_measurement_rule_ids()
        values = request.measurement_values
        if values is None:
            values = request.area_measurement_values()
        return FilterObjectsMeasurementLimitWindow(
            values=values,
            min_value=request.min_value,
            max_value=request.max_value,
            use_minimum=request.use_minimum,
            use_maximum=request.use_maximum,
        ).retained_ids


class ExtremumFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep one object selected by a measurement extremum."""

    keep_max: ClassVar[bool | None] = None

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        keep_max = type(self).keep_max
        if keep_max is None:
            raise TypeError("ExtremumFilterSelectionStrategy must define keep_max.")
        values = request.measurement_values
        if values is None:
            values = request.first_measurement_values()
        return keep_one_object(values, keep_max=keep_max)


class MinimalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the minimum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MINIMAL)
    selection_label = selection_key.label
    keep_max = False


class MaximalFilterSelectionStrategy(ExtremumFilterSelectionStrategy):
    """Keep the object with the maximum measurement value."""

    selection_key = FilterSelectionKey(FilterMode.MEASUREMENTS, FilterMethod.MAXIMAL)
    selection_label = selection_key.label
    keep_max = True


class PerObjectFilterSelectionStrategy(FilterSelectionStrategy):
    """Keep one child object per enclosing parent object."""

    selection_key: ClassVar[FilterSelectionKey | None] = None

    def indexes_to_keep(self, request: FilterObjectsSelectionRequest) -> list[int]:
        selection_key = type(self).selection_key
        if selection_key is None or selection_key.method is None:
            raise TypeError("PerObjectFilterSelectionStrategy must define method.")
        values = request.first_measurement_values().dense_label_indexed(
            max_label=int(request.labels.max()) if request.labels.size else 0
        )
        return PerObjectAssignmentStrategy.for_assignment(
            request.per_object_assignment
        ).indexes_to_keep(
            PerObjectAssignmentRequest(
                child_labels=request.labels,
                enclosing_labels=require_enclosing_labels(request),
                measurement_values=values,
                child_count=request.num_objects_pre,
                keep_max=selection_key.method is FilterMethod.MAXIMAL_PER_OBJECT,
                parent_child_relationship=request.parent_child_relationship,
            )
        )


class MinimalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for minimal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS, FilterMethod.MINIMAL_PER_OBJECT
    )
    selection_label = selection_key.label


class MaximalPerObjectFilterSelectionStrategy(PerObjectFilterSelectionStrategy):
    """Fail loudly for maximal-per-parent filtering until relationships exist."""

    selection_key = FilterSelectionKey(
        FilterMode.MEASUREMENTS, FilterMethod.MAXIMAL_PER_OBJECT
    )
    selection_label = selection_key.label


@dataclass(frozen=True, slots=True)
class PerObjectAssignmentRequest:
    """Inputs for assigning candidate child objects to enclosing parents."""

    child_labels: np.ndarray
    enclosing_labels: np.ndarray
    measurement_values: np.ndarray
    child_count: int
    keep_max: bool
    parent_child_relationship: FilterObjectsParentChildRelationship | None = None

    def __post_init__(self) -> None:
        if self.child_labels.shape != self.enclosing_labels.shape:
            raise ValueError(
                f"FilterObjects per-object child and enclosing labels must have matching shape, got {self.child_labels.shape} and {self.enclosing_labels.shape}."
            )


class PerObjectAssignmentStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal parent-assignment strategy for per-object filtering."""

    __registry_key__ = "assignment_label"
    __skip_if_no_key__ = True
    assignment_label: ClassVar[str | None] = None
    assignment: ClassVar[PerObjectAssignment | None] = None

    @classmethod
    def for_assignment(
        cls, assignment: PerObjectAssignment
    ) -> "PerObjectAssignmentStrategy":
        strategy_type = cls.__registry__.get(assignment.value)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported FilterObjects per-object assignment {assignment.value!r}."
            )
        return strategy_type()

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        parent_children = self.parent_children_from_relationship(
            request
        ) or self.parent_children(request)
        return self.best_child_indexes_by_parent(parent_children, request)

    def parent_children_from_relationship(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        relationship = request.parent_child_relationship
        if relationship is None:
            return {}
        if isinstance(relationship, ObjectRelationship):
            parent_ids = relationship.source_ids
            child_ids = relationship.target_ids
        elif isinstance(relationship, ParentChildRelationshipPayload):
            parent_ids = relationship.parent_ids
            child_ids = relationship.child_ids
        else:
            raise TypeError(
                f"FilterObjects parent_child_relationship must be ObjectRelationship or ParentChildRelationshipPayload, got {type(relationship).__name__}."
            )
        parent_children: dict[int, set[int]] = {}
        for parent_id, child_id in zip(
            FilterObjectsRelationshipEndpointIds(parent_ids).ids,
            FilterObjectsRelationshipEndpointIds(child_ids).ids,
            strict=True,
        ):
            if parent_id > 0 and child_id > 0:
                parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children

    def best_child_indexes_by_parent(
        self, parent_children: dict[int, set[int]], request: PerObjectAssignmentRequest
    ) -> list[int]:
        selected: set[int] = set()
        for child_ids in parent_children.values():
            child_values = tuple(
                (
                    (
                        child_id,
                        self.measurement_value_for_child(
                            request.measurement_values, child_id
                        ),
                    )
                    for child_id in child_ids
                )
            )
            finite_child_values = tuple(
                (
                    (child_id, value)
                    for child_id, value in child_values
                    if np.isfinite(value)
                )
            )
            if not finite_child_values:
                continue
            selected.add(
                min(
                    finite_child_values,
                    key=(
                        (lambda item: (-item[1], item[0]))
                        if request.keep_max
                        else lambda item: (item[1], item[0])
                    ),
                )[0]
            )
        return sorted(selected)

    @staticmethod
    def measurement_value_for_child(
        measurement_values: np.ndarray, child_id: int
    ) -> float:
        value_index = child_id - 1
        if value_index < 0 or value_index >= len(measurement_values):
            return float("nan")
        return float(measurement_values[value_index])

    @staticmethod
    def overlap_label_pairs(
        request: PerObjectAssignmentRequest,
    ) -> tuple[tuple[int, int], ...]:
        overlap_mask = (request.child_labels > 0) & (request.enclosing_labels > 0)
        child_ids = request.child_labels[overlap_mask].astype(np.int64, copy=False)
        parent_ids = request.enclosing_labels[overlap_mask].astype(np.int64, copy=False)
        return tuple(
            (
                (int(child_id), int(parent_id))
                for child_id, parent_id in zip(child_ids, parent_ids, strict=True)
            )
        )

    @abstractmethod
    def parent_children(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        """Return child labels eligible for each enclosing parent label."""


class BothParentsAssignmentStrategy(PerObjectAssignmentStrategy):
    """Assign an overlapping child as a candidate for every touched parent."""

    assignment = PerObjectAssignment.BOTH_PARENTS
    assignment_label = assignment.value

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        return best_child_indexes_both_parents(
            request.child_labels,
            request.enclosing_labels,
            request.measurement_values,
            request.keep_max,
        )

    def parent_children(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        parent_children: dict[int, set[int]] = {}
        for child_id, parent_id in self.overlap_label_pairs(request):
            parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children


class ParentWithMostOverlapAssignmentStrategy(PerObjectAssignmentStrategy):
    """Assign each child only to its most-overlapped enclosing parent."""

    assignment = PerObjectAssignment.PARENT_WITH_MOST_OVERLAP
    assignment_label = assignment.value

    def indexes_to_keep(self, request: PerObjectAssignmentRequest) -> list[int]:
        parent_children = self.parent_children_from_relationship(request)
        if parent_children:
            return self.best_child_indexes_by_parent(parent_children, request)
        return best_child_indexes_parent_with_most_overlap(
            request.child_labels,
            request.enclosing_labels,
            request.measurement_values,
            request.keep_max,
        )

    def parent_children(
        self, request: PerObjectAssignmentRequest
    ) -> dict[int, set[int]]:
        counts_by_child: dict[int, dict[int, int]] = {}
        for child_id, parent_id in self.overlap_label_pairs(request):
            parent_counts = counts_by_child.setdefault(child_id, {})
            parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1
        parent_children: dict[int, set[int]] = {}
        for child_id, parent_counts in counts_by_child.items():
            parent_id = min(
                parent_counts,
                key=lambda candidate: (-parent_counts[candidate], candidate),
            )
            parent_children.setdefault(parent_id, set()).add(child_id)
        return parent_children


def keep_one_object(
    values: ObjectLabelMeasurementValues, keep_max: bool = True
) -> list[int]:
    """Keep only the object with the maximum or minimum finite measurement."""
    selected_id = values.extremum_id(keep_max=keep_max)
    return [] if selected_id is None else [selected_id]


def require_enclosing_labels(request: FilterObjectsSelectionRequest) -> np.ndarray:
    if request.enclosing_labels is not None:
        return request.enclosing_labels
    raise ValueError(
        "FilterObjects per-object filtering requires enclosing object labels."
    )


def best_child_indexes_both_parents(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    import scipy.ndimage

    values = np.asarray(measurement_values, dtype=np.float64)
    if values.size == 0:
        return []
    child_array = np.asarray(child_labels, dtype=np.int32)
    parent_array = np.asarray(enclosing_labels, dtype=np.int32)
    max_parent = int(parent_array.max()) if parent_array.size else 0
    if max_parent <= 0:
        return []
    pixel_values = np.empty(values.size + 1, dtype=np.float64)
    pixel_values[1:] = values
    pixel_values[0] = -np.inf if keep_max else np.inf
    source_values = pixel_values[child_array]
    parent_range = np.arange(1, max_parent + 1)
    position_fn = (
        scipy.ndimage.maximum_position if keep_max else scipy.ndimage.minimum_position
    )
    positions = position_fn(source_values, parent_array, parent_range)
    positions = np.asarray(
        (positions,) if isinstance(positions, tuple) else positions, dtype=np.uint32
    )
    if positions.size == 0:
        return []
    indexes = tuple(map(tuple, positions.transpose()))
    selected = sorted(set((int(label) for label in child_array[indexes])))
    if selected and selected[0] == 0:
        selected = selected[1:]
    return selected


def best_child_indexes_parent_with_most_overlap(
    child_labels: np.ndarray,
    enclosing_labels: np.ndarray,
    measurement_values: np.ndarray,
    keep_max: bool,
) -> list[int]:
    max_child = int(np.max(child_labels))
    if max_child <= 0:
        return []
    child_ids, parent_ids, overlap_counts = unique_overlap_counts(
        child_labels, enclosing_labels
    )
    if child_ids.size == 0:
        return []
    return selected_labels_to_list(
        _best_child_selected_mask_parent_with_most_overlap_numba(
            np.ascontiguousarray(child_ids, dtype=np.int32),
            np.ascontiguousarray(parent_ids, dtype=np.int32),
            np.ascontiguousarray(overlap_counts, dtype=np.int64),
            np.ascontiguousarray(measurement_values, dtype=np.float64),
            max_child,
            bool(keep_max),
        )
    )


def unique_overlap_counts(
    child_labels: np.ndarray, enclosing_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    child_array = np.asarray(child_labels, dtype=np.int64)
    parent_array = np.asarray(enclosing_labels, dtype=np.int64)
    max_parent = int(parent_array.max())
    overlap_mask = (child_array > 0) & (parent_array > 0)
    if not np.any(overlap_mask):
        empty = np.array([], dtype=np.int64)
        return (empty, empty, empty)
    encoded_pairs = (
        child_array[overlap_mask] * (max_parent + 1) + parent_array[overlap_mask]
    )
    unique_pairs, overlap_counts = np.unique(encoded_pairs, return_counts=True)
    return (
        unique_pairs // (max_parent + 1),
        unique_pairs % (max_parent + 1),
        overlap_counts,
    )


@njit(cache=True)
def _best_child_selected_mask_parent_with_most_overlap_numba(
    child_ids: np.ndarray,
    parent_ids: np.ndarray,
    overlap_counts: np.ndarray,
    measurement_values: np.ndarray,
    max_child: int,
    keep_max: bool,
) -> np.ndarray:
    selected = np.zeros(max_child + 1, dtype=np.bool_)
    if child_ids.size == 0:
        return selected
    max_parent = int(np.max(parent_ids))
    best_parent_by_child = np.zeros(max_child + 1, dtype=np.int32)
    best_overlap_by_child = np.zeros(max_child + 1, dtype=np.int64)
    for index in range(child_ids.size):
        child_id = int(child_ids[index])
        parent_id = int(parent_ids[index])
        overlap_count = int(overlap_counts[index])
        best_parent = int(best_parent_by_child[child_id])
        best_overlap = int(best_overlap_by_child[child_id])
        if (
            best_parent == 0
            or overlap_count > best_overlap
            or (overlap_count == best_overlap and parent_id < best_parent)
        ):
            best_parent_by_child[child_id] = parent_id
            best_overlap_by_child[child_id] = overlap_count
    best_child_by_parent = np.zeros(max_parent + 1, dtype=np.int32)
    best_value_by_parent = np.empty(max_parent + 1, dtype=np.float64)
    for child_id in range(1, max_child + 1):
        parent_id = int(best_parent_by_child[child_id])
        if parent_id <= 0:
            continue
        value_index = child_id - 1
        if value_index < 0 or value_index >= measurement_values.size:
            continue
        value = float(measurement_values[value_index])
        if not np.isfinite(value):
            continue
        best_child = int(best_child_by_parent[parent_id])
        if best_child == 0:
            best_child_by_parent[parent_id] = child_id
            best_value_by_parent[parent_id] = value
        else:
            best_value = float(best_value_by_parent[parent_id])
            if keep_max:
                if value > best_value or (
                    value == best_value and child_id < best_child
                ):
                    best_child_by_parent[parent_id] = child_id
                    best_value_by_parent[parent_id] = value
            elif value < best_value or (value == best_value and child_id < best_child):
                best_child_by_parent[parent_id] = child_id
                best_value_by_parent[parent_id] = value
    for parent_id in range(1, max_parent + 1):
        child_id = int(best_child_by_parent[parent_id])
        if child_id > 0:
            selected[child_id] = True
    return selected


def selected_labels_to_list(selected: np.ndarray) -> list[int]:
    return np.flatnonzero(np.asarray(selected, dtype=bool)).astype(int).tolist()


def filtered_object_payloads(
    inputs: Sequence[ObjectLabelValue], outputs: Sequence[np.ndarray]
) -> tuple[ObjectLabelValue, ...]:
    """Return dense object payloads preserving the original object domain."""
    return tuple(
        (
            filtered_object_payload(input_value, output_labels)
            for input_value, output_labels in zip(inputs, outputs, strict=True)
        )
    )


def filtered_object_payload(
    input_value: ObjectLabelValue, output_labels: np.ndarray
) -> ObjectLabelValue:
    """Wrap filtered labels with the input object's dense extent domain."""
    return object_label_value_with_dense_labels(
        input_value,
        output_labels,
        domain_declaration=DenseObjectLabelExtentDomainDeclaration(),
    )


def filter_objects_relationship_tuple(
    relationship: FilterObjectsParentChildRelationship | None,
    relationships: Sequence[FilterObjectsParentChildRelationship],
) -> FilterObjectsParentChildRelationships:
    """Return stable relationship inputs without duplicate object-pair entries."""
    ordered = (
        *(() if relationship is None else (relationship,)),
        *tuple(relationships),
    )
    unique: list[FilterObjectsParentChildRelationship] = []
    seen: set[tuple[str, str] | int] = set()
    for value in ordered:
        key: tuple[str, str] | int
        if isinstance(value, ObjectRelationship):
            key = (value.source.name, value.target.name)
        else:
            key = id(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return tuple(unique)


def relabel_overlapping_objects(
    labels: np.ndarray, filtered_primary_labels: np.ndarray
) -> np.ndarray:
    """Relabel additional objects by overlap with retained primary objects."""
    retained_mask = filtered_primary_labels > 0
    labels = labels.astype(np.int32)
    if labels.shape != retained_mask.shape:
        raise ValueError(
            "FilterObjects additional object labels must match primary labels."
        )
    return FilterObjectsRelabeledPlane.from_retained_mask(labels, retained_mask).labels


def object_transform_relationships(
    input_label_planes: tuple[np.ndarray, ...],
    relabeled_objects: tuple[np.ndarray, ...],
) -> tuple[ParentChildRelationshipPayload, ...]:
    """Derive parent-child payloads between input and filtered object labels."""
    if len(input_label_planes) != len(relabeled_objects):
        raise ValueError(
            "Object transform relationship derivation requires aligned input and output label planes."
        )
    relationship_backend = ObjectRelationshipBackendStrategy.for_memory_type()
    return tuple(
        (
            relationship_backend.parent_child_payload_from_labels(
                np.asarray(input_labels), np.asarray(output_labels)
            )
            for input_labels, output_labels in zip(
                input_label_planes, relabeled_objects, strict=True
            )
        )
    )


def filter_objects_outline_images(
    relabeled_objects: tuple[np.ndarray, ...], outline_object_indices: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    """Return requested FilterObjects outline sidecar images."""
    return tuple(
        (
            filter_objects_outline_image(relabeled_objects[index])
            for index in outline_object_indices
        )
    )


def filter_objects_outline_image(labels: np.ndarray) -> np.ndarray:
    """Return a binary outline image for dense 2-D object labels."""
    labels = np.asarray(labels).astype(np.int32)
    if labels.ndim != 2:
        raise ValueError("FilterObjects outline images require 2D labels.")
    boundary = np.zeros(labels.shape, dtype=bool)
    boundary[:-1, :] |= labels[:-1, :] != labels[1:, :]
    boundary[1:, :] |= labels[:-1, :] != labels[1:, :]
    boundary[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    boundary[:, 1:] |= labels[:, :-1] != labels[:, 1:]
    boundary &= labels > 0
    return boundary.astype(np.uint8)


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(
    (
        "filter_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "objects_pre_filter",
                "objects_post_filter",
                "objects_removed",
            ],
            analysis_type="filter_objects",
        ),
    ),
    ("filtered_labels", segmentation_mask_rois()),
)
def filter_objects(
    image: np.ndarray,
    mode: FilterMode = FilterMode.MEASUREMENTS,
    filter_method: FilterMethod = FilterMethod.LIMITS,
    object_labels: tuple[np.ndarray, ...] = (),
    measurement_values: np.ndarray | None = None,
    measurement_features: tuple[str, ...] = (),
    measurement_min_values: tuple[float | None, ...] = (),
    measurement_max_values: tuple[float | None, ...] = (),
    measurement_use_minimum: tuple[bool, ...] = (),
    measurement_use_maximum: tuple[bool, ...] = (),
    measurement_tables: tuple[MeasurementTable, ...] = (),
    enclosing_object_labels: np.ndarray | None = None,
    parent_child_relationship: FilterObjectsParentChildRelationship | None = None,
    parent_child_relationships: FilterObjectsParentChildRelationships = (),
    per_object_assignment: PerObjectAssignment = PerObjectAssignment.BOTH_PARENTS,
    min_value: float | None = None,
    max_value: float | None = None,
    use_minimum: bool = True,
    use_maximum: bool = True,
    additional_object_count: int = 0,
    outline_object_indices: tuple[int, ...] = (),
    slice_by_slice: bool = True,
) -> tuple[
    np.ndarray, FilterObjectsStats, np.ndarray | ParentChildRelationshipPayload, ...
]:
    """Filter dense object labels using CellProfiler-compatible selection policy."""
    if object_labels is None:
        object_labels = ()
    elif isinstance(object_labels, np.ndarray):
        object_labels = (object_labels,)
    if len(object_labels) == 0:
        raise ValueError("FilterObjects requires at least one object label input.")
    mode = coerce_cellprofiler_enum(FilterMode, mode)
    filter_method = coerce_cellprofiler_enum(FilterMethod, filter_method)
    per_object_assignment = coerce_cellprofiler_enum(
        PerObjectAssignment, per_object_assignment
    )
    if additional_object_count != len(object_labels) - 1:
        raise ValueError(
            "FilterObjects additional_object_count must match additional object label inputs."
        )
    labels = FilterObjectsLabelPlane(object_labels[0]).projected.astype(np.int32)
    additional_label_planes = tuple(
        (
            FilterObjectsLabelPlane(value).aligned_to(labels)
            for value in object_labels[1:]
        )
    )
    input_label_planes = (labels, *additional_label_planes)
    max_label = labels.max()
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0, objects_post_filter=0
        )
        relabeled_planes = tuple(
            (
                FilterObjectsRelabeledPlane.from_ordered_parent_ids(
                    label_plane, parent_ids_by_child=()
                )
                for label_plane in (labels, *additional_label_planes)
            )
        )
        relabeled_objects = filtered_object_payloads(
            object_labels, tuple((plane.labels for plane in relabeled_planes))
        )
        return (
            image,
            stats,
            *relabeled_objects,
            *(plane.relationship for plane in relabeled_planes),
            *filter_objects_outline_images(relabeled_objects, outline_object_indices),
        )
    object_ids = ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels)
    selection_measurement_values = (
        None
        if measurement_values is None
        else ObjectLabelMeasurementValues.from_positional_values(
            object_ids, measurement_values
        )
    )
    indexes_to_keep = FilterSelectionStrategy.for_mode_and_method(
        mode, filter_method
    ).indexes_to_keep(
        FilterObjectsSelectionRequest(
            labels=labels,
            object_ids=object_ids,
            filter_method=filter_method,
            measurement_values=selection_measurement_values,
            measurement_features=measurement_features,
            measurement_min_values=measurement_min_values,
            measurement_max_values=measurement_max_values,
            measurement_use_minimum=measurement_use_minimum,
            measurement_use_maximum=measurement_use_maximum,
            measurement_tables=measurement_tables,
            enclosing_labels=FilterObjectsLabelPlane.optional_aligned_to(
                labels, enclosing_object_labels
            ),
            parent_child_relationship=parent_child_relationship,
            parent_child_relationships=filter_objects_relationship_tuple(
                parent_child_relationship, parent_child_relationships
            ),
            per_object_assignment=per_object_assignment,
            min_value=min_value,
            max_value=max_value,
            use_minimum=use_minimum,
            use_maximum=use_maximum,
        )
    )
    primary_plane = FilterObjectsRelabeledPlane.from_ordered_parent_ids(
        labels, parent_ids_by_child=tuple(indexes_to_keep)
    )
    filtered_labels = primary_plane.labels
    retained_mask = filtered_labels > 0
    additional_relabeled_planes = tuple(
        (
            FilterObjectsRelabeledPlane.from_retained_mask(additional, retained_mask)
            for additional in additional_label_planes
        )
    )
    relabeled_planes = (primary_plane, *additional_relabeled_planes)
    relabeled_objects = filtered_object_payloads(
        object_labels, tuple((plane.labels for plane in relabeled_planes))
    )
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=len(object_ids), objects_post_filter=len(indexes_to_keep)
    )
    return (
        image,
        stats,
        *relabeled_objects,
        *(plane.relationship for plane in relabeled_planes),
        *filter_objects_outline_images(relabeled_objects, outline_object_indices),
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "filter_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "objects_pre_filter",
                "objects_post_filter",
                "objects_removed",
            ],
            analysis_type="filter_objects",
        ),
    ),
    ("filtered_labels", segmentation_mask_rois()),
)
def filter_objects_by_size(
    image: np.ndarray,
    labels: np.ndarray,
    min_area: float = 0.0,
    max_area: float = float("inf"),
    use_minimum: bool = True,
    use_maximum: bool = True,
) -> tuple[np.ndarray, FilterObjectsStats, np.ndarray]:
    """Filter objects based on area measurements."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    max_label = labels.max()
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0, objects_post_filter=0
        )
        return (image, stats, labels)
    region_props = LabelRegionPropertiesBackendStrategy.for_memory_type().measure_2d(
        labels
    )
    indexes_to_keep = FilterObjectsMeasurementLimitWindow.from_label_indexed_values(
        region_props.area,
        min_value=min_area,
        max_value=max_area,
        use_minimum=use_minimum,
        use_maximum=use_maximum,
    ).retained_ids
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=len(region_props.label),
        objects_post_filter=len(indexes_to_keep),
    )
    return (image, stats, label_mapping[labels])


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "filter_stats",
        csv_materializer(
            fields=[
                "slice_index",
                "objects_pre_filter",
                "objects_post_filter",
                "objects_removed",
            ],
            analysis_type="filter_objects",
        ),
    ),
    ("filtered_labels", segmentation_mask_rois()),
)
def filter_border_objects(
    image: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, FilterObjectsStats, np.ndarray]:
    """Remove objects touching the image border."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    max_label = labels.max()
    if max_label == 0:
        stats = FilterObjectsStats.from_counts(
            objects_pre_filter=0, objects_post_filter=0
        )
        return (image, stats, labels)
    object_ids = ObjectLabelIdDomainStrategy.for_value(labels).present_ids(labels)
    indexes_to_keep = BorderFilterSelectionStrategy.discard_border_objects(labels)
    label_mapping = np.zeros(max_label + 1, dtype=np.int32)
    for new_idx, old_idx in enumerate(indexes_to_keep, start=1):
        if old_idx <= max_label:
            label_mapping[old_idx] = new_idx
    stats = FilterObjectsStats.from_counts(
        objects_pre_filter=len(object_ids), objects_post_filter=len(indexes_to_keep)
    )
    return (image, stats, label_mapping[labels])


__all__ = public_names_from_objects(
    FilterMethod,
    FilterMode,
    FilterObjectsMeasurementValuesSource,
    FilterObjectsLabelPlane,
    FilterObjectsMeasurementLimitWindow,
    "FilterObjectsParentChildRelationship",
    "FilterObjectsParentChildRelationships",
    FilterObjectsRelationshipEndpointIds,
    FilterObjectsSelectionRequest,
    FilterObjectsStats,
    FilterSelectionStrategy,
    "FilterSelectionKey",
    PerObjectAssignment,
    PerObjectAssignmentRequest,
    PerObjectAssignmentStrategy,
    best_child_indexes_both_parents,
    best_child_indexes_parent_with_most_overlap,
    filter_objects_outline_image,
    filter_objects_outline_images,
    filter_objects_relationship_tuple,
    filter_border_objects,
    filter_objects,
    filter_objects_by_size,
    filtered_object_payload,
    filtered_object_payloads,
    keep_one_object,
    object_transform_relationships,
    relabel_overlapping_objects,
    require_enclosing_labels,
)
