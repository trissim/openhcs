"""Classification backends for CellProfiler-compatible object measurements."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar
from openhcs.core.runtime_semantics import MeasurementRowAxisField
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeArtifactTypeStrategy,
)
from openhcs.interop.cellprofiler.runtime.binding_authorities import (
    CellProfilerStringKwargAuthority,
)
from openhcs.interop.cellprofiler.runtime.bound_parameters import (
    RuntimeBoundParameterName,
    RuntimeBoundParameterNames,
    declared_runtime_bound_parameter_names,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectInputCountAuthority,
    CellProfilerObjectMeasurementVectorBinding,
    CellProfilerObjectMeasurementVectorSource,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerKwargDict,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.special_input_policies import (
    NoSpecialImageInputsMixin,
    SpecialInputBindingRequest,
)
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ImageArtifactOutputCapability,
    ImageArtifactOutputModule,
    MeasurementArtifactOutputModule,
    ModuleSettingsSourceModule,
    ObjectArtifactInputModule,
    PriorMeasurementArtifactInputModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    ColumnarFieldsMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
    NoSourceMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    CellProfilerMeasurementStatField,
    FormattingMeasurementFeatureTemplate,
    ModuleOwnedResultMeasurementRows,
    _measurement_object_name,
)
from openhcs.interop.cellprofiler.runtime.mapping_lookup import MappingValueLookup


class ClassifyObjectsMeasurementInputPolicy(NoSpecialImageInputsMixin):
    """Resolve ClassifyObjects label and measurement-vector inputs."""

    measurement_value_parameters: ClassVar[RuntimeBoundParameterNames] = (
        RuntimeBoundParameterNames(
            "measurement_values", "measurement1_values", "measurement2_values"
        )
    )
    measurement_feature_kwargs: ClassVar[tuple[str, ...]] = (
        "measurement_feature",
        "measurement1_feature",
        "measurement2_feature",
    )
    measurement_kwarg_by_parameter: ClassVar[Mapping[str, str]] = dict(
        zip(measurement_value_parameters, measurement_feature_kwargs, strict=True)
    )
    labels_kwarg: ClassVar[RuntimeBoundParameterName] = RuntimeBoundParameterName(
        "labels"
    )
    measurement_values_by_rule_kwarg: ClassVar[RuntimeBoundParameterName] = (
        RuntimeBoundParameterName("measurement_values_by_rule")
    )

    def extra_bound_parameter_names(
        self, plan: CellProfilerModuleRuntimePlan
    ) -> tuple[str, ...]:
        """Return runtime-derived classification measurement vectors."""
        del plan
        declared_names = declared_runtime_bound_parameter_names(type(self))
        return tuple((name for name in declared_names if name != self.labels_kwarg))

    def bind(self, request: SpecialInputBindingRequest) -> CellProfilerKwargDict:
        object_inputs = request.object_inputs
        CellProfilerObjectInputCountAuthority.require_exact(
            request.module_name, object_inputs, 1
        )
        object_spec = object_inputs[0]
        labels = request.labels_for(object_spec)
        measurement_labels = request.label_payload_for(object_spec)
        image_number = RuntimeArtifactTypeStrategy.for_artifact_type(
            object_spec.artifact_type
        ).cellprofiler_image_number(request.artifact_input_request(object_spec))
        if "classification_rules" in request.kwargs:
            rules = request.kwargs["classification_rules"]
            if not isinstance(rules, (tuple, list)):
                raise ValueError(
                    f"{request.module_name} classification_rules must be an ordered tuple or list."
                )
            return {
                self.labels_kwarg: labels,
                self.measurement_values_by_rule_kwarg: tuple(
                    (
                        CellProfilerObjectMeasurementVectorBinding.for_object(
                            request,
                            object_ref=object_spec,
                            feature_name=_classification_rule_measurement_feature(
                                rule, request.module_name
                            ),
                            labels=measurement_labels,
                            image_number=image_number,
                        )
                        .vector()
                        .slice_aligned_value
                        for rule in rules
                    )
                ),
            }
        bound_values = {
            parameter_name: CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=CellProfilerStringKwargAuthority.required(
                    request.kwargs, kwarg_name, request.module_name
                ),
                labels=measurement_labels,
                image_number=image_number,
                source=CellProfilerObjectMeasurementVectorSource.CURRENT_OBJECT_SHAPE_FEATURE,
            )
            .vector()
            .slice_aligned_value
            for parameter_name, kwarg_name in type(
                self
            ).measurement_kwarg_by_parameter.items()
            if kwarg_name in request.kwargs
        }
        return {self.labels_kwarg: labels, **bound_values}


def _classification_rule_measurement_feature(
    rule: CellProfilerRuntimeValue, module_name: str
) -> str:
    if not isinstance(rule, Mapping):
        raise ValueError(f"{module_name} classification rule must be a mapping.")
    value = rule.get("measurement_feature")
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{module_name} classification rule requires non-empty 'measurement_feature'."
        )
    return value


class ClassifyObjectsSingleMeasurementModule(
    NoObjectNameMeasurementRecordMixin,
    NoSourceMeasurementRecordMixin,
    ColumnarFieldsMeasurementRecordMixin,
    ClassifyObjectsMeasurementInputPolicy,
    ObjectArtifactInputModule,
    ImageArtifactOutputModule,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
    BinderSettingsSourceModule,
):
    module_name = "ClassifyObjectsSingleMeasurement"
    function_name = "classify_objects_single_measurement"
    function_variants = ("classify_objects_two_measurements",)
    contract = ProcessingContract.FLEXIBLE
    validated = True
    aliases = ("ClassifyObjects", "ClassifyObjectsTwoMeasurements")
    confidence = 0.0
    measurement_category_prefixes = (("classify",),)
    classification_decision_count_setting = SettingNameFamily(
        "Make each classification decision on how many measurements?"
    )
    input_objects_setting = SettingNameFamily("Select the object to be classified")
    single_measurement_feature_setting = SettingNameFamily(
        "Select the measurement to classify by"
    )
    first_measurement_feature_setting = SettingNameFamily(
        "Select the first measurement"
    )
    second_measurement_feature_setting = SettingNameFamily(
        "Select the second measurement"
    )
    bin_spacing_setting = SettingNameFamily("Select bin spacing")
    bin_count_setting = SettingNameFamily("Number of bins")
    low_threshold_setting = SettingNameFamily("Lower threshold")
    high_threshold_setting = SettingNameFamily("Upper threshold")
    wants_low_bin_setting = SettingNameFamily(
        "Use a bin for objects below the threshold?"
    )
    wants_high_bin_setting = SettingNameFamily(
        "Use a bin for objects above the threshold?"
    )
    custom_thresholds_setting = SettingNameFamily(
        "Enter the custom thresholds separating the values between bins"
    )
    bin_names_setting = SettingNameFamily("Enter the bin names separated by commas")
    threshold_method_setting = SettingNameFamily("Method to select the cutoff")
    threshold_value_setting = SettingNameFamily("Enter the cutoff value")
    low_low_bin_name_setting = SettingNameFamily("Enter the low-low bin name")
    low_high_bin_name_setting = SettingNameFamily("Enter the low-high bin name")
    high_low_bin_name_setting = SettingNameFamily("Enter the high-low bin name")
    high_high_bin_name_setting = SettingNameFamily("Enter the high-high bin name")
    retain_image_setting = SettingNameFamily(
        "Retain an image of the classified objects?"
    )
    output_image_setting = SettingNameFamily("Name the output image")
    classification_decision_default = "Single measurement"
    measurement_feature_default = ""
    bin_spacing_default = "Evenly spaced bins"
    bin_count_default = "3"
    low_threshold_default = "0.0"
    high_threshold_default = "1.0"
    wants_low_bin_default = "No"
    wants_high_bin_default = "No"
    custom_thresholds_default = "0,1"
    bin_names_default = ""
    threshold_method_default = "Mean"
    threshold_value_default = "0.5"
    low_low_bin_name_default = "low_low"
    low_high_bin_name_default = "low_high"
    high_low_bin_name_default = "high_low"
    high_high_bin_name_default = "high_high"

    class MeasurementStatField(CellProfilerMeasurementStatField):
        """Absorbed ClassifyObjects result fields."""

        BIN_COUNTS = "bin_counts"
        BIN_PERCENTAGES = "bin_percentages"
        OBJECT_CLASSES = "object_classes"
        TOTAL_OBJECTS = "total_objects"
        SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value

    class MeasurementFeatureTemplate(FormattingMeasurementFeatureTemplate):
        """Templated CellProfiler ClassifyObjects measurement features."""

        OBJECTS_PER_BIN = "Classify_{bin_name}_NumObjectsPerBin"
        PERCENT_PER_BIN = "Classify_{bin_name}_PctObjectsPerBin"
        OBJECT_CLASS = "Classify_{bin_name}"

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project absorbed ClassifyObjects results into CP measurement rows."""

        registry_key = "classify_objects"
        object_name: str | None

        @classmethod
        def for_request(cls, module_type, request):
            return cls(
                request.output_value,
                module_type=module_type,
                object_name=_measurement_object_name(request.declared_input_specs),
            )

        def rows(self) -> list[CellProfilerKwargDict]:
            rows: list[CellProfilerKwargDict] = []
            stat_field = self.stat_field_type
            feature_template = self.feature_template_type
            for result in self.source_rows():
                bin_counts = self.json_object_mapping(
                    self.row_value(result, stat_field.BIN_COUNTS, {})
                )
                bin_percentages = self.json_object_mapping(
                    self.row_value(result, stat_field.BIN_PERCENTAGES, {})
                )
                object_classes = self.json_object_mapping(
                    self.row_value(result, stat_field.OBJECT_CLASSES, {})
                )
                slice_index = int(self.row_value(result, stat_field.SLICE_INDEX, 0))
                bin_names = tuple(str(name) for name in bin_counts)
                for bin_name, count in bin_counts.items():
                    rows.append(
                        self.measurement_row(
                            axis_values={
                                MeasurementRowAxisField.SLICE_INDEX.value: slice_index
                            },
                            feature_name=feature_template.OBJECTS_PER_BIN.feature_name(
                                bin_name=str(bin_name)
                            ),
                            value=count,
                        )
                    )
                    rows.append(
                        self.measurement_row(
                            axis_values={
                                MeasurementRowAxisField.SLICE_INDEX.value: slice_index
                            },
                            feature_name=feature_template.PERCENT_PER_BIN.feature_name(
                                bin_name=str(bin_name)
                            ),
                            value=MappingValueLookup(
                                bin_percentages,
                                bin_name,
                            ).value_or(0.0),
                        )
                    )
                rows.extend(
                    self.object_class_rows(
                        object_classes=object_classes,
                        bin_names=bin_names,
                        result=result,
                        slice_index=slice_index,
                    )
                )
            return rows

        def object_class_rows(
            self,
            *,
            object_classes: CellProfilerKwargs,
            bin_names: tuple[str, ...],
            result: CellProfilerRuntimeValue,
            slice_index: int,
        ) -> list[CellProfilerKwargDict]:
            if self.object_name is None:
                return []
            stat_field = self.stat_field_type
            feature_template = self.feature_template_type
            total_objects = int(
                self.row_value(result, stat_field.TOTAL_OBJECTS, 0)
            )
            class_labels = tuple(sorted(int(label) for label in object_classes))
            dense_labels = tuple(range(1, total_objects + 1))
            object_labels = tuple(dict.fromkeys((*dense_labels, *class_labels)))
            return [
                self.object_measurement_row(
                    object_name=self.object_name,
                    object_label=object_label,
                    axis_values={
                        MeasurementRowAxisField.SLICE_INDEX.value: slice_index
                    },
                    feature_name=feature_template.OBJECT_CLASS.feature_name(
                        bin_name=bin_name
                    ),
                    value=int(
                        object_classes.get(str(object_label)) == bin_name
                    ),
                )
                for object_label in object_labels
                for bin_name in bin_names
            ]

    @classmethod
    def settings_source(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "CellProfilerKwargs":
        if cls.uses_two_measurements(module):
            return cls._two_measurement_kwargs(module, binder)
        return cls._single_measurement_kwargs(module, binder)

    @classmethod
    def indexed_setting_value(
        cls,
        module: "ModuleBlock",
        setting_name: str | SettingNameFamily,
        *,
        default: str,
        value_index: int = 0,
    ) -> str:
        values = setting_values(module, setting_name)
        if not values:
            return default
        if value_index < len(values):
            return values[value_index]
        return values[-1]

    @classmethod
    def uses_two_measurements(cls, module: "ModuleBlock") -> bool:
        value = cls.indexed_setting_value(
            module,
            cls.classification_decision_count_setting,
            default=cls.classification_decision_default,
        ).lower()
        if "two" in value:
            return True
        if "single" in value:
            return False
        raise ValueError(
            f"Unsupported ClassifyObjects measurement count setting: {value!r}."
        )

    @classmethod
    def function_name_for_module(cls, module: "ModuleBlock") -> str:
        if cls.uses_two_measurements(module):
            return cls.function_variants[0]
        return str(cls.function_name)

    @classmethod
    def resolve_function(
        cls, module: "ModuleBlock", *, default_function_name: str | None = None
    ) -> "ResolvedModuleFunction":
        del default_function_name
        return super().resolve_function(
            module, default_function_name=cls.function_name_for_module(module)
        )

    @staticmethod
    def _canonical_setting_name(setting_name: str | SettingNameFamily) -> str:
        if isinstance(setting_name, SettingNameFamily):
            return setting_name.canonical
        return setting_name

    @classmethod
    def _typed_setting_value(
        cls,
        module: "ModuleBlock",
        binder: "SettingsBinder",
        setting_name: str | SettingNameFamily,
        *,
        default: str,
        value_index: int = 0,
    ) -> Any:
        return binder.parse_value(
            cls._canonical_setting_name(setting_name),
            cls.indexed_setting_value(
                module, setting_name, default=default, value_index=value_index
            ),
        )

    @classmethod
    def _required_indexed_setting_value(
        cls,
        module: "ModuleBlock",
        setting_name: str | SettingNameFamily,
        *,
        default: str,
        value_index: int = 0,
    ) -> str:
        value = cls.indexed_setting_value(
            module, setting_name, default=default, value_index=value_index
        ).strip()
        if not value:
            raise ValueError(f"ClassifyObjects requires setting {setting_name!r}.")
        return value

    @staticmethod
    def _bin_choice(value: str) -> str:
        normalized = value.strip().lower()
        if "custom" in normalized:
            return "custom"
        if "even" in normalized:
            return "even"
        raise ValueError(f"Unsupported ClassifyObjects bin spacing: {value!r}.")

    @staticmethod
    def _threshold_method(value: str) -> str:
        normalized = value.strip().lower()
        if "median" in normalized:
            return "median"
        if "mean" in normalized:
            return "mean"
        if "custom" in normalized:
            return "custom"
        raise ValueError(f"Unsupported ClassifyObjects threshold method: {value!r}.")

    @classmethod
    def _single_measurement_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "CellProfilerKwargs":
        measurement_features = setting_values(
            module, cls.single_measurement_feature_setting
        )
        if len(measurement_features) > 1:
            return {
                "classification_rules": tuple(
                    (
                        cls._single_measurement_rule_kwargs(module, binder, index)
                        for index in range(len(measurement_features))
                    )
                )
            }
        return cls._single_measurement_rule_kwargs(module, binder, 0)

    @classmethod
    def _single_measurement_rule_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder", value_index: int
    ) -> "CellProfilerKwargs":
        bin_names = cls.indexed_setting_value(
            module,
            cls.bin_names_setting,
            default=cls.bin_names_default,
            value_index=value_index,
        )
        return {
            "measurement_feature": cls._required_indexed_setting_value(
                module,
                cls.single_measurement_feature_setting,
                default=cls.measurement_feature_default,
                value_index=value_index,
            ),
            "bin_choice": cls._bin_choice(
                cls.indexed_setting_value(
                    module,
                    cls.bin_spacing_setting,
                    default=cls.bin_spacing_default,
                    value_index=value_index,
                )
            ),
            "bin_count": cls._typed_setting_value(
                module,
                binder,
                cls.bin_count_setting,
                default=cls.bin_count_default,
                value_index=value_index,
            ),
            "low_threshold": cls._typed_setting_value(
                module,
                binder,
                cls.low_threshold_setting,
                default=cls.low_threshold_default,
                value_index=value_index,
            ),
            "high_threshold": cls._typed_setting_value(
                module,
                binder,
                cls.high_threshold_setting,
                default=cls.high_threshold_default,
                value_index=value_index,
            ),
            "wants_low_bin": cls._typed_setting_value(
                module,
                binder,
                cls.wants_low_bin_setting,
                default=cls.wants_low_bin_default,
                value_index=value_index,
            ),
            "wants_high_bin": cls._typed_setting_value(
                module,
                binder,
                cls.wants_high_bin_setting,
                default=cls.wants_high_bin_default,
                value_index=value_index,
            ),
            "custom_thresholds": cls.indexed_setting_value(
                module,
                cls.custom_thresholds_setting,
                default=cls.custom_thresholds_default,
                value_index=value_index,
            ),
            "bin_names": bin_names or None,
        }

    @classmethod
    def _two_measurement_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "CellProfilerKwargs":
        return {
            "measurement1_feature": cls._required_indexed_setting_value(
                module,
                cls.first_measurement_feature_setting,
                default=cls.measurement_feature_default,
            ),
            "measurement2_feature": cls._required_indexed_setting_value(
                module,
                cls.second_measurement_feature_setting,
                default=cls.measurement_feature_default,
            ),
            "threshold1_method": cls._threshold_method(
                cls.indexed_setting_value(
                    module,
                    cls.threshold_method_setting,
                    default=cls.threshold_method_default,
                )
            ),
            "threshold1_value": cls._typed_setting_value(
                module,
                binder,
                cls.threshold_value_setting,
                default=cls.threshold_value_default,
            ),
            "threshold2_method": cls._threshold_method(
                cls.indexed_setting_value(
                    module,
                    cls.threshold_method_setting,
                    default=cls.threshold_method_default,
                    value_index=-1,
                )
            ),
            "threshold2_value": cls._typed_setting_value(
                module,
                binder,
                cls.threshold_value_setting,
                default=cls.threshold_value_default,
                value_index=-1,
            ),
            "low_low_name": cls.indexed_setting_value(
                module,
                cls.low_low_bin_name_setting,
                default=cls.low_low_bin_name_default,
            ),
            "low_high_name": cls.indexed_setting_value(
                module,
                cls.low_high_bin_name_setting,
                default=cls.low_high_bin_name_default,
            ),
            "high_low_name": cls.indexed_setting_value(
                module,
                cls.high_low_bin_name_setting,
                default=cls.high_low_bin_name_default,
            ),
            "high_high_name": cls.indexed_setting_value(
                module,
                cls.high_high_bin_name_setting,
                default=cls.high_high_bin_name_default,
            ),
        }

    @classmethod
    def object_input_setting_names(cls):
        return (cls.input_objects_setting,)

    @classmethod
    def artifact_contract_outputs(cls, builder, module):
        outputs = []
        if optional_setting_value(module, cls.retain_image_setting) in {
            "Yes",
            "yes",
            "True",
            "true",
        }:
            output_name = optional_setting_value(module, cls.output_image_setting)
            if output_name is not None:
                outputs.append(
                    cls.image_output_artifact(
                        builder,
                        module,
                        output_name,
                        setting=cls.output_image_setting,
                    )
                )
        return (*outputs, *super().artifact_contract_outputs(builder, module))


from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
import json
from typing import ClassVar
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_semantics import ObjectLabelMeasurementValues
from openhcs.core.runtime_values import object_label_dense_array
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.processing.materialization import (
    csv_dataclass_materializer,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)

class ClassificationMethod(Enum):
    """CellProfiler ClassifyObjects measurement-count mode."""

    SINGLE_MEASUREMENT = "single_measurement"
    TWO_MEASUREMENTS = "two_measurements"


class ClassificationThresholdMethod(Enum):
    """CellProfiler ClassifyObjects threshold selection mode."""

    MEAN = "mean"
    MEDIAN = "median"
    CUSTOM = "custom"


class ClassificationBinChoice(Enum):
    """CellProfiler ClassifyObjects bin spacing mode."""

    EVEN = "even"
    CUSTOM = "custom"


class ClassificationThresholdStrategy(ABC, metaclass=AutoRegisterMeta):
    """Threshold calculation strategy for ClassifyObjects measurement vectors."""

    __registry_key__ = "method"
    __skip_if_no_key__ = True
    method: ClassVar[ClassificationThresholdMethod | None] = None

    @classmethod
    def for_method(
        cls, method: ClassificationThresholdMethod
    ) -> "ClassificationThresholdStrategy":
        return cls.__registry__[method]()

    def threshold(self, values: np.ndarray, custom_value: float) -> float:
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return custom_value
        return self._threshold(valid_values, custom_value)

    @abstractmethod
    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        """Return a threshold for finite measurement values."""


class MeanClassificationThresholdStrategy(ClassificationThresholdStrategy):
    """Mean-based ClassifyObjects threshold."""

    method = ClassificationThresholdMethod.MEAN

    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        del custom_value
        return float(np.mean(valid_values))


class MedianClassificationThresholdStrategy(ClassificationThresholdStrategy):
    """Median-based ClassifyObjects threshold."""

    method = ClassificationThresholdMethod.MEDIAN

    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        del custom_value
        return float(np.median(valid_values))


class CustomClassificationThresholdStrategy(ClassificationThresholdStrategy):
    """User-specified ClassifyObjects threshold."""

    method = ClassificationThresholdMethod.CUSTOM

    def _threshold(self, valid_values: np.ndarray, custom_value: float) -> float:
        del valid_values
        return custom_value


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Results from object classification."""

    slice_index: int
    total_objects: int
    bin_counts: str
    bin_percentages: str
    object_classes: str = "{}"

    @classmethod
    def empty(cls, *, total_objects: int = 0) -> "ClassificationResult":
        """Return an empty classification result row."""
        return cls(
            slice_index=0,
            total_objects=total_objects,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({}),
        )


@dataclass(frozen=True, slots=True)
class ClassificationMeasurementVector:
    """Measurement vector normalized to the current object-label domain."""

    values: np.ndarray

    @classmethod
    def from_value(cls, values: np.ndarray) -> "ClassificationMeasurementVector":
        return cls(np.asarray(values, dtype=np.float64).reshape(-1))

    def aligned_to_labels(self, label_ids: np.ndarray) -> np.ndarray:
        """Return values ordered like the materially present object labels."""
        if label_ids.size == 0:
            return np.zeros(0, dtype=np.float64)
        if self.values.size == label_ids.size:
            return self.values.copy()
        max_label = int(label_ids[-1])
        if self.values.size >= max_label and max_label > label_ids.size:
            return ObjectLabelMeasurementValues.from_label_indexed_values(
                tuple((int(label_id) for label_id in label_ids)), self.values
            ).values
        aligned = np.full(label_ids.size, np.nan, dtype=np.float64)
        copied = min(self.values.size, aligned.size)
        if copied:
            aligned[:copied] = self.values[:copied]
        return aligned


@dataclass(frozen=True, slots=True)
class SingleMeasurementClassificationRequest:
    """Semantic request for single-measurement object classification."""

    measurement_values: np.ndarray | None = None
    bin_choice: ClassificationBinChoice | str = ClassificationBinChoice.EVEN
    bin_count: int = 3
    low_threshold: float = 0.0
    high_threshold: float = 1.0
    wants_low_bin: bool = False
    wants_high_bin: bool = False
    custom_thresholds: str = "0,1"
    bin_names: str | None = None

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        bin_choice = coerce_cellprofiler_enum(ClassificationBinChoice, self.bin_choice)
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if num_objects == 0:
            return (labels, ClassificationResult.empty())
        if self.measurement_values is None:
            values = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values = ClassificationMeasurementVector.from_value(
                self.measurement_values
            ).aligned_to_labels(unique_labels)
        if bin_choice == ClassificationBinChoice.EVEN:
            low_threshold = self.low_threshold
            high_threshold = self.high_threshold
            if low_threshold >= high_threshold:
                low_threshold, high_threshold = (high_threshold, low_threshold)
            thresholds = np.linspace(low_threshold, high_threshold, self.bin_count + 1)
        else:
            thresholds = np.array(
                [float(x.strip()) for x in self.custom_thresholds.split(",")]
            )
        threshold_list = []
        if self.wants_low_bin:
            threshold_list.append(-np.inf)
        threshold_list.extend(thresholds.tolist())
        if self.wants_high_bin:
            threshold_list.append(np.inf)
        thresholds = np.array(threshold_list)
        num_bins = len(thresholds) - 1
        if self.bin_names is not None:
            names = [name.strip() for name in self.bin_names.split(",")]
        else:
            names = [f"Bin_{index + 1}" for index in range(num_bins)]
        while len(names) < num_bins:
            names.append(f"Bin_{len(names) + 1}")
        object_bins = np.zeros(num_objects, dtype=np.int32)
        for index, value in enumerate(values):
            if np.isnan(value):
                object_bins[index] = 0
            else:
                for bin_index in range(num_bins):
                    if thresholds[bin_index] < value <= thresholds[bin_index + 1]:
                        object_bins[index] = bin_index + 1
                        break
        return (
            labels,
            classification_result_from_bins(unique_labels, object_bins, names),
        )


@dataclass(frozen=True, slots=True)
class TwoMeasurementClassificationRequest:
    """Semantic request for two-measurement object classification."""

    measurement1_values: np.ndarray | None = None
    measurement2_values: np.ndarray | None = None
    threshold1_method: ClassificationThresholdMethod | str = (
        ClassificationThresholdMethod.MEAN
    )
    threshold1_value: float = 0.5
    threshold2_method: ClassificationThresholdMethod | str = (
        ClassificationThresholdMethod.MEAN
    )
    threshold2_value: float = 0.5
    low_low_name: str = "low_low"
    low_high_name: str = "low_high"
    high_low_name: str = "high_low"
    high_high_name: str = "high_high"

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        threshold1_method = coerce_cellprofiler_enum(
            ClassificationThresholdMethod, self.threshold1_method
        )
        threshold2_method = coerce_cellprofiler_enum(
            ClassificationThresholdMethod, self.threshold2_method
        )
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if num_objects == 0:
            return (labels, ClassificationResult.empty())
        if self.measurement1_values is None:
            values1 = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values1 = ClassificationMeasurementVector.from_value(
                self.measurement1_values
            ).aligned_to_labels(unique_labels)
        if self.measurement2_values is None:
            values2 = np.bincount(
                labels.astype(np.intp, copy=False).ravel(),
                minlength=int(unique_labels[-1]) + 1 if num_objects else 1,
            )[unique_labels].astype(float)
        else:
            values2 = ClassificationMeasurementVector.from_value(
                self.measurement2_values
            ).aligned_to_labels(unique_labels)
        t1 = classification_threshold(values1, threshold1_method, self.threshold1_value)
        t2 = classification_threshold(values2, threshold2_method, self.threshold2_value)
        high1 = values1 >= t1
        high2 = values2 >= t2
        has_nan = np.isnan(values1) | np.isnan(values2)
        object_class = np.zeros(num_objects, dtype=np.int32)
        object_class[~high1 & ~high2 & ~has_nan] = 1
        object_class[high1 & ~high2 & ~has_nan] = 2
        object_class[~high1 & high2 & ~has_nan] = 3
        object_class[high1 & high2 & ~has_nan] = 4
        names = [
            self.low_low_name,
            self.high_low_name,
            self.low_high_name,
            self.high_high_name,
        ]
        return (
            labels,
            classification_result_from_bins(unique_labels, object_class, names),
        )


@dataclass(frozen=True, slots=True)
class IntensityBinsClassificationRequest:
    """Semantic request for intensity-bin object classification."""

    num_bins: int = 3
    use_percentiles: bool = True

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if num_objects == 0:
            return (labels, ClassificationResult.empty())
        values = backend.mean_intensity_values(labels, image, unique_labels)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if len(valid_values) == 0:
            return (labels, ClassificationResult.empty(total_objects=num_objects))
        if self.use_percentiles:
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            thresholds = np.percentile(valid_values, percentiles)
        else:
            thresholds = np.linspace(
                np.min(valid_values), np.max(valid_values), self.num_bins + 1
            )
        object_bins = np.zeros(num_objects, dtype=np.int32)
        for index, value in enumerate(values):
            if np.isnan(value):
                continue
            for bin_index in range(self.num_bins):
                if bin_index == self.num_bins - 1:
                    if thresholds[bin_index] <= value <= thresholds[bin_index + 1]:
                        object_bins[index] = bin_index + 1
                elif thresholds[bin_index] <= value < thresholds[bin_index + 1]:
                    object_bins[index] = bin_index + 1
                    break
        bin_names = [f"Intensity_Bin_{index + 1}" for index in range(self.num_bins)]
        return (
            labels,
            classification_result_from_bins(unique_labels, object_bins, bin_names),
        )


def classification_threshold(
    values: np.ndarray, method: ClassificationThresholdMethod, custom_value: float
) -> float:
    """Return the threshold for one ClassifyObjects measurement vector."""
    method = coerce_cellprofiler_enum(ClassificationThresholdMethod, method)
    return ClassificationThresholdStrategy.for_method(method).threshold(
        values, custom_value
    )


def classification_result_from_bins(
    unique_labels: np.ndarray, object_bins: np.ndarray, names: list[str]
) -> ClassificationResult:
    """Return serialized ClassifyObjects measurement rows from bin ids."""
    num_objects = len(unique_labels)
    bin_counts: dict[str, int] = {}
    bin_percentages: dict[str, float] = {}
    for bin_index, name in enumerate(names):
        count = np.sum(object_bins == bin_index + 1)
        bin_counts[name] = int(count)
        bin_percentages[name] = (
            float(count / num_objects * 100) if num_objects > 0 else 0.0
        )
    object_classes: dict[int, str] = {}
    for index, label_value in enumerate(unique_labels):
        if object_bins[index] > 0:
            object_classes[int(label_value)] = names[object_bins[index] - 1]
    return ClassificationResult(
        slice_index=0,
        total_objects=num_objects,
        bin_counts=json.dumps(bin_counts),
        bin_percentages=json.dumps(bin_percentages),
        object_classes=json.dumps(object_classes),
    )


class ObjectClassificationBackendStrategy(
    CellProfilerBackendStrategyMixin, ABC, metaclass=AutoRegisterMeta
):
    """Object classification primitives keyed by memory type/provider."""

    __registry_key__ = "backend_key"
    __skip_if_no_key__ = True

    @abstractmethod
    def positive_label_ids(self, labels: np.ndarray) -> np.ndarray:
        """Return positive label ids present in ``labels``."""

    @abstractmethod
    def mean_intensity_values(
        self, labels: np.ndarray, image: np.ndarray, label_ids: np.ndarray
    ) -> np.ndarray:
        """Return mean intensity for ``label_ids``."""

    @abstractmethod
    def apply_object_bins(
        self, labels: np.ndarray, label_ids: np.ndarray, object_bins: np.ndarray
    ) -> np.ndarray:
        """Map source labels to classification bin ids in one image pass."""


class NumbaNumpyObjectClassificationBackendStrategy(
    ObjectClassificationBackendStrategy
):
    """Numba-backed NumPy object classification primitives."""

    backend_key = CellProfilerBackendAuthority.backend_key(
        MemoryType.NUMPY, CellProfilerBackendProvider.NUMBA
    )
    memory_type = MemoryType.NUMPY
    backend_provider = CellProfilerBackendProvider.NUMBA
    is_default_backend = True

    def prepare_backend(self) -> None:
        labels = np.array([[0, 1], [2, 2]], dtype=np.int32)
        image = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
        label_ids = np.array([1, 2], dtype=np.int32)
        object_bins = np.array([1, 2], dtype=np.int32)
        self.mean_intensity_values(labels, image, label_ids)
        self.apply_object_bins(labels, label_ids, object_bins)

    def positive_label_ids(self, labels: np.ndarray) -> np.ndarray:
        labels_array = np.asarray(labels, dtype=np.int32)
        if labels_array.size == 0:
            return np.zeros(0, dtype=np.int32)
        max_label = int(labels_array.max())
        if max_label <= 0:
            return np.zeros(0, dtype=np.int32)
        present = np.bincount(labels_array.ravel(), minlength=max_label + 1) > 0
        return np.flatnonzero(present[1:]).astype(np.int32) + 1

    def mean_intensity_values(
        self, labels: np.ndarray, image: np.ndarray, label_ids: np.ndarray
    ) -> np.ndarray:
        return _mean_intensity_values_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(image, dtype=np.float64),
            np.asarray(label_ids, dtype=np.int32),
        )

    def apply_object_bins(
        self, labels: np.ndarray, label_ids: np.ndarray, object_bins: np.ndarray
    ) -> np.ndarray:
        return _apply_object_bins_numba(
            np.asarray(labels, dtype=np.int32),
            np.asarray(label_ids, dtype=np.int32),
            np.asarray(object_bins, dtype=np.int32),
        )


def object_classification_backend(
    *, backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION
) -> ObjectClassificationBackendStrategy:
    """Return the selected object-classification backend."""
    return ObjectClassificationBackendStrategy.for_memory_type(
        MemoryType.NUMPY, backend_provider=backend_provider
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "classification_results",
        csv_dataclass_materializer(
            ClassificationResult,
            analysis_type="classification",
        ),
    )
)
def classify_objects_single_measurement(
    image: np.ndarray,
    labels: np.ndarray,
    measurement_values: np.ndarray | None = None,
    measurement_values_by_rule: tuple[np.ndarray, ...] = (),
    classification_rules: tuple[dict[str, object], ...] = (),
    bin_choice: ClassificationBinChoice = ClassificationBinChoice.EVEN,
    bin_count: int = 3,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
    wants_low_bin: bool = False,
    wants_high_bin: bool = False,
    custom_thresholds: str = "0,1",
    bin_names: str | None = None,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    slice_index: int | None = None,
) -> tuple[np.ndarray, ClassificationResult | tuple[ClassificationResult, ...]]:
    """Classify objects based on one measurement or declared rule rows."""
    label_array = object_label_dense_array(labels, dtype=np.int32)
    if (
        slice_index is None
        and np.asarray(image).ndim == 2
        and (label_array.ndim == 3)
        and (label_array.shape[-2:] == np.asarray(image).shape)
    ):
        results: list[ClassificationResult] = []
        value_offset = 0
        measurement_vector = (
            None
            if measurement_values is None
            else np.asarray(measurement_values, dtype=np.float64).reshape(-1)
        )
        for plane_index, label_plane in enumerate(label_array):
            object_count = int(np.unique(label_plane[label_plane > 0]).size)
            plane_values = None
            if measurement_vector is not None:
                plane_values = measurement_vector[
                    value_offset : value_offset + object_count
                ]
            value_offset += object_count
            _image, result = classify_objects_single_measurement(
                image,
                label_plane,
                measurement_values=plane_values,
                measurement_values_by_rule=measurement_values_by_rule,
                classification_rules=classification_rules,
                bin_choice=bin_choice,
                bin_count=bin_count,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
                wants_low_bin=wants_low_bin,
                wants_high_bin=wants_high_bin,
                custom_thresholds=custom_thresholds,
                bin_names=bin_names,
                classification_backend_provider=classification_backend_provider,
                slice_index=plane_index,
            )
            if isinstance(result, tuple):
                results.extend(
                    (replace(item, slice_index=plane_index) for item in result)
                )
            else:
                results.append(replace(result, slice_index=plane_index))
        return (image, tuple(results))
    slice_index = 0 if slice_index is None else int(slice_index)
    labels = _labels_for_image_slice(label_array, image, slice_index)
    backend = object_classification_backend(
        backend_provider=classification_backend_provider
    )
    if classification_rules:
        results: list[ClassificationResult] = []
        classified_labels = labels
        for rule_index, rule in enumerate(classification_rules):
            rule_values = (
                measurement_values_by_rule[rule_index]
                if rule_index < len(measurement_values_by_rule)
                else None
            )
            classified_labels, result = SingleMeasurementClassificationRequest(
                measurement_values=rule_values,
                bin_choice=rule.get("bin_choice", ClassificationBinChoice.EVEN),
                bin_count=int(rule.get("bin_count", 3)),
                low_threshold=float(rule.get("low_threshold", 0.0)),
                high_threshold=float(rule.get("high_threshold", 1.0)),
                wants_low_bin=bool(rule.get("wants_low_bin", False)),
                wants_high_bin=bool(rule.get("wants_high_bin", False)),
                custom_thresholds=str(rule.get("custom_thresholds", "0,1")),
                bin_names=rule.get("bin_names"),
            ).classify(image, labels, backend)
            results.append(result)
        return (classified_labels, tuple(results))
    return SingleMeasurementClassificationRequest(
        measurement_values=measurement_values,
        bin_choice=bin_choice,
        bin_count=bin_count,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        wants_low_bin=wants_low_bin,
        wants_high_bin=wants_high_bin,
        custom_thresholds=custom_thresholds,
        bin_names=bin_names,
    ).classify(image, labels, backend)


def _labels_for_image_slice(
    labels: np.ndarray, image: np.ndarray, slice_index: int
) -> np.ndarray:
    image_array = np.asarray(image)
    if (
        image_array.ndim == 2
        and labels.ndim > 2
        and (labels.shape[-2:] == image_array.shape)
    ):
        if 0 <= slice_index < labels.shape[0]:
            labels = labels[slice_index]
        if labels.ndim > 2:
            labels = np.max(labels, axis=tuple(range(labels.ndim - 2)))
    return np.asarray(labels, dtype=np.int32)


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "classification_results",
        csv_dataclass_materializer(
            ClassificationResult,
            analysis_type="classification",
        ),
    )
)
def classify_objects_two_measurements(
    image: np.ndarray,
    labels: np.ndarray,
    measurement1_values: np.ndarray | None = None,
    measurement2_values: np.ndarray | None = None,
    threshold1_method: ClassificationThresholdMethod = ClassificationThresholdMethod.MEAN,
    threshold1_value: float = 0.5,
    threshold2_method: ClassificationThresholdMethod = ClassificationThresholdMethod.MEAN,
    threshold2_value: float = 0.5,
    low_low_name: str = "low_low",
    low_high_name: str = "low_high",
    high_low_name: str = "high_low",
    high_high_name: str = "high_high",
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, ClassificationResult]:
    """Classify objects from two measurements into four quadrants."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    return TwoMeasurementClassificationRequest(
        measurement1_values=measurement1_values,
        measurement2_values=measurement2_values,
        threshold1_method=threshold1_method,
        threshold1_value=threshold1_value,
        threshold2_method=threshold2_method,
        threshold2_value=threshold2_value,
        low_low_name=low_low_name,
        low_high_name=low_high_name,
        high_low_name=high_low_name,
        high_high_name=high_high_name,
    ).classify(
        image,
        labels,
        object_classification_backend(backend_provider=classification_backend_provider),
    )


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@special_outputs(
    (
        "classification_results",
        csv_dataclass_materializer(
            ClassificationResult,
            analysis_type="classification",
        ),
    )
)
def classify_objects_by_intensity_bins(
    image: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 3,
    use_percentiles: bool = True,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, ClassificationResult]:
    """Classify objects by mean intensity into evenly distributed bins."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    return IntensityBinsClassificationRequest(
        num_bins=num_bins, use_percentiles=use_percentiles
    ).classify(
        image,
        labels,
        object_classification_backend(backend_provider=classification_backend_provider),
    )


@njit(cache=True)
def _mean_intensity_values_numba(
    labels: np.ndarray, image: np.ndarray, label_ids: np.ndarray
) -> np.ndarray:
    max_label = 0
    for i in range(label_ids.size):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    sums = np.zeros(max_label + 1, dtype=np.float64)
    counts = np.zeros(max_label + 1, dtype=np.int64)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                sums[label] += image[row, col]
                counts[label] += 1
    values = np.empty(label_ids.size, dtype=np.float64)
    for i in range(label_ids.size):
        label = int(label_ids[i])
        if label <= 0 or label > max_label or counts[label] == 0:
            values[i] = np.nan
        else:
            values[i] = sums[label] / counts[label]
    return values


@njit(cache=True)
def _apply_object_bins_numba(
    labels: np.ndarray, label_ids: np.ndarray, object_bins: np.ndarray
) -> np.ndarray:
    max_label = 0
    for i in range(label_ids.size):
        label_id = int(label_ids[i])
        if label_id > max_label:
            max_label = label_id
    bin_by_label = np.zeros(max_label + 1, dtype=np.int32)
    count = label_ids.size
    if object_bins.size < count:
        count = object_bins.size
    for i in range(count):
        label = int(label_ids[i])
        if label > 0 and label <= max_label:
            bin_by_label[label] = int(object_bins[i])
    output = np.zeros(labels.shape, dtype=np.int32)
    rows, cols = labels.shape
    for row in range(rows):
        for col in range(cols):
            label = int(labels[row, col])
            if label > 0 and label <= max_label:
                output[row, col] = bin_by_label[label]
    return output


__all__ = public_names_from_objects(
    ClassificationBinChoice,
    ClassificationMethod,
    ClassificationResult,
    ClassificationThresholdMethod,
    IntensityBinsClassificationRequest,
    NumbaNumpyObjectClassificationBackendStrategy,
    ObjectClassificationBackendStrategy,
    SingleMeasurementClassificationRequest,
    TwoMeasurementClassificationRequest,
    classify_objects_by_intensity_bins,
    classify_objects_single_measurement,
    classify_objects_two_measurements,
    classification_result_from_bins,
    classification_threshold,
    object_classification_backend,
)
