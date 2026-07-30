"""Classification backends for CellProfiler-compatible object measurements."""

from __future__ import annotations
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar
import numpy as np

from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import (
    ObjectLabelsArtifactType, ArtifactSpecCollection,
    ImageArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import CallableContract, KeywordRuntimeParameter
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_plane_projection import RuntimeSliceInvariantValue
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import (
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_bindings import StepSourceBindingsConfig
from openhcs.interop.cellprofiler.runtime.object_measurement_vectors import (
    CellProfilerObjectMeasurementVectorBinding,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument, RuntimeCallableKwargs
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
    setting_name_matches,
)
from openhcs.interop.cellprofiler.parser import ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import (
    MeasurementFeatureSettingBinding,
    SettingToKeywordBinding,
    cellprofiler_setting_literal,
    parse_cellprofiler_bool,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    PriorMeasurementArtifactInputModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    setting_values,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    NoObjectNameMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    FormattingMeasurementFeatureTemplate,
    ModuleOwnedResultMeasurementRows,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import RuntimeInputBindingRequest

if TYPE_CHECKING:
    from openhcs.interop.cellprofiler.parser import ModuleBlock
    from openhcs.interop.cellprofiler.settings_binder import SettingsBinder


@dataclass(frozen=True, slots=True)
class ClassifiedImageOutput:
    """One active classified-image output owned by a single rule group."""

    rule_index: int
    name: str


@dataclass(frozen=True)
class ClassifiedImageSourceRelation(SourceStackLineageSourceRelation):
    """Tie one classified image to its exact repeated classification rule."""

    relation_key: ClassVar[str] = "classify_objects_classified_image_source"
    target_artifact_type = ImageArtifactType

    rule_index: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.rule_index < 0:
            raise ValueError("Classified image rule_index must be non-negative.")


class _ClassifiedImageRuleIndicesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound classification rules ordered by declared image outputs."""

    parameter_name = "classified_image_rule_indices"
    annotation_type = tuple[int, ...]
    parameter_default = ()


class _ClassificationMeasurementValuesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound vector for single-measurement classification."""

    parameter_name = "measurement_values"
    annotation_type = np.ndarray | None
    parameter_default = None


class _ClassificationMeasurement1ValuesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound first vector for two-measurement classification."""

    parameter_name = "measurement1_values"
    annotation_type = np.ndarray | None
    parameter_default = None


class _ClassificationMeasurement2ValuesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound second vector for two-measurement classification."""

    parameter_name = "measurement2_values"
    annotation_type = np.ndarray | None
    parameter_default = None


class _ClassificationRuleValuesRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound vectors for imported classification rules."""

    parameter_name = "measurement_values_by_rule"
    annotation_type = tuple[np.ndarray, ...]
    parameter_default = ()


class ClassifyObjectsMeasurementInputPolicy:
    """Resolve ClassifyObjects label and measurement-vector inputs."""
    measurement_value_parameters: ClassVar[
        tuple[type[KeywordRuntimeParameter], ...]
    ] = (
        _ClassificationMeasurementValuesRuntimeParameter,
        _ClassificationMeasurement1ValuesRuntimeParameter,
        _ClassificationMeasurement2ValuesRuntimeParameter,
    )
    measurement_feature_kwargs: ClassVar[tuple[str, ...]] = (
        "measurement_feature",
        "measurement1_feature",
        "measurement2_feature",
    )
    measurement_kwarg_by_parameter: ClassVar[Mapping[str, str]] = dict(
        zip(
            (
                parameter_type.require_parameter_name()
                for parameter_type in measurement_value_parameters
            ),
            measurement_feature_kwargs,
            strict=True,
        )
    )

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        object_inputs = request.object_inputs
        object_spec = object_inputs[0]
        special_inputs = super().bind_runtime_inputs(request)
        if len(special_inputs) != 1:
            raise ValueError(
                f"{request.adapter.request.require_callable_contract().module_name} requires one special object-label "
                f"parameter, got {tuple(special_inputs)!r}."
            )
        labels_parameter, labels = next(iter(special_inputs.items()))
        measurement_labels = request.label_payload_for(object_spec)
        classified_image_rule_indices = cls.classified_image_rule_indices(request)
        if "classification_rules" in request.kwargs:
            rules = request.kwargs["classification_rules"]
            if not isinstance(rules, (tuple, list)):
                raise ValueError(
                    f"{request.adapter.request.require_callable_contract().module_name} classification_rules must be an ordered tuple or list."
                )
            return {
                labels_parameter: labels,
                _ClassificationRuleValuesRuntimeParameter.require_parameter_name(): tuple(
                    (
                        CellProfilerObjectMeasurementVectorBinding.for_object(
                            request,
                            object_ref=object_spec,
                            feature_name=_classification_rule_measurement_feature(
                                rule, request.adapter.request.require_callable_contract().module_name
                            ),
                            labels=measurement_labels,
                        )
                        .vector()
                        .runtime_value
                        for rule in rules
                    )
                ),
                _ClassifiedImageRuleIndicesRuntimeParameter.require_parameter_name(): (
                    classified_image_rule_indices
                ),
            }
        bound_values = {
            parameter_name: CellProfilerObjectMeasurementVectorBinding.for_object(
                request,
                object_ref=object_spec,
                feature_name=request.require_string_kwarg(kwarg_name),
                labels=measurement_labels,
            )
            .vector()
            .runtime_value
            for parameter_name, kwarg_name in cls.measurement_kwarg_by_parameter.items()
            if kwarg_name in request.kwargs
        }
        return {
            labels_parameter: labels,
            **bound_values,
            _ClassifiedImageRuleIndicesRuntimeParameter.require_parameter_name(): (
                classified_image_rule_indices
            ),
        }

    @staticmethod
    def classified_image_rule_indices(
        request: RuntimeInputBindingRequest,
    ) -> tuple[int, ...]:
        """Return rule indices in the compiled image-output declaration order."""

        image_outputs = request.adapter.request.require_callable_contract().artifact_outputs.of_artifact_type(
            ImageArtifactType
        )
        indices: list[int] = []
        for output in image_outputs:
            relations = tuple(
                relation
                for relation in output.relations
                if isinstance(relation, ClassifiedImageSourceRelation)
            )
            if len(relations) != 1:
                raise ValueError(
                    "ClassifyObjects classified image outputs require one exact "
                    f"ClassifiedImageSourceRelation, got {relations!r} for "
                    f"{output.name!r}."
                )
            indices.append(relations[0].rule_index)
        return tuple(indices)


def _classification_rule_measurement_feature(
    rule: RuntimeCallableArgument, module_name: str
) -> str:
    if not isinstance(rule, SingleMeasurementClassificationRule):
        raise ValueError(
            f"{module_name} classification rule must be a "
            "SingleMeasurementClassificationRule."
        )
    if not rule.measurement_feature.strip():
        raise ValueError(
            f"{module_name} classification rule requires a measurement feature."
        )
    return rule.measurement_feature


class ClassifyObjectsSingleMeasurementModule(
    NoObjectNameMeasurementRecordMixin,
    ClassifyObjectsMeasurementInputPolicy,
    ObjectArtifactInputModule,
    PriorMeasurementArtifactInputModule,
    MeasurementArtifactOutputModule,
):
    module_name = "ClassifyObjectsSingleMeasurement"
    function_name = "classify_objects_single_measurement"
    function_variants = ("classify_objects_two_measurements",)
    validated = True
    aliases = ("ClassifyObjects", "ClassifyObjectsTwoMeasurements")
    confidence = 0.0
    measurement_category_prefixes = (("classify",),)
    ignored_settings = ("Hidden",)
    classification_decision_count_setting = SettingNameFamily(
        "Make each classification decision on how many measurements?"
    )
    input_objects_setting: ClassVar[SettingNameFamily] = SettingNameFamily(
        "Select the object to be classified",
        aliases=("Select the object name",),
    )
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting, ObjectLabelsArtifactType, runtime_parameter_name="labels"
    )
    single_measurement_feature_setting = SettingNameFamily(
        "Select the measurement to classify by"
    )
    single_measurement_feature_binding = MeasurementFeatureSettingBinding(
        single_measurement_feature_setting,
        "measurement_feature",
    )
    first_measurement_feature_setting = SettingNameFamily(
        "Select the first measurement"
    )
    first_measurement_feature_binding = MeasurementFeatureSettingBinding(
        first_measurement_feature_setting,
        "measurement1_feature",
    )
    second_measurement_feature_setting = SettingNameFamily(
        "Select the second measurement"
    )
    second_measurement_feature_binding = MeasurementFeatureSettingBinding(
        second_measurement_feature_setting,
        "measurement2_feature",
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
    give_bin_names_setting = SettingNameFamily(
        "Give each bin a name?",
        aliases=("Use custom names for the bins?",),
    )
    threshold_method_setting = SettingNameFamily("Method to select the cutoff")
    threshold_value_setting = SettingNameFamily("Enter the cutoff value")
    low_low_bin_name_setting = SettingNameFamily("Enter the low-low bin name")
    low_high_bin_name_setting = SettingNameFamily("Enter the low-high bin name")
    high_low_bin_name_setting = SettingNameFamily("Enter the high-low bin name")
    high_high_bin_name_setting = SettingNameFamily("Enter the high-high bin name")
    retain_image_setting = SettingNameFamily(
        "Retain an image of the classified objects?"
    )
    output_image_setting = SettingNameFamily(
        "Name the output image",
        aliases=("Enter the image name",),
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting,
        ImageArtifactType,
        "retained_image_name",
        repeated=True,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        input_objects_binding,
        single_measurement_feature_binding,
        first_measurement_feature_binding,
        second_measurement_feature_binding,
        output_image_binding,
    )
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

    class MeasurementFeatureTemplate(FormattingMeasurementFeatureTemplate):
        """Templated CellProfiler ClassifyObjects measurement features."""

        OBJECTS_PER_BIN = ("Classify_{bin_name}_NumObjectsPerBin", int)
        PERCENT_PER_BIN = ("Classify_{bin_name}_PctObjectsPerBin", float)
        OBJECT_CLASS = ("Classify_{bin_name}", int)

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project absorbed ClassifyObjects results into CP measurement rows."""

        object_name: str | None

        @classmethod
        def for_request(cls, module_type, request):
            return cls(
                request.output_value,
                module_type=module_type,
                object_name=module_type.runtime_object_measurement_row_policy().table_object_owner(
                    request.callable_contract.artifact_inputs.specs
                ),
            )

        def rows(self) -> MeasurementSparseColumnarRows:
            rows: list[dict[str, object]] = []
            slice_field = self.source_field_annotated_by(
                ClassificationResult,
                MeasurementRowAxisField.SLICE_INDEX,
            )
            total_objects_field = self.source_field_annotated_by(
                ClassificationResult,
                _ClassificationResultFieldRole.OBJECT_DOMAIN_SIZE,
            )
            feature_fields = {
                template: field_spec
                for field_spec, template in self.source_fields_annotated_with(
                    ClassificationResult,
                    FormattingMeasurementFeatureTemplate,
                )
            }
            feature_template = self.feature_template_type
            declared_templates = frozenset(feature_template)
            if frozenset(feature_fields) != declared_templates:
                raise TypeError(
                    "ClassificationResult feature annotations must exactly match "
                    f"{feature_template.__name__}: got {tuple(feature_fields)!r}."
                )
            bin_counts_field = feature_fields[feature_template.OBJECTS_PER_BIN]
            bin_percentages_field = feature_fields[feature_template.PERCENT_PER_BIN]
            object_classes_field = feature_fields[feature_template.OBJECT_CLASS]
            fields: list[FieldSpec] = [
                slice_field,
                FieldSpec(
                    MeasurementRowAxisField.OBJECT_NAME.value,
                    str,
                    required=False,
                ),
                FieldSpec(
                    MeasurementRowAxisField.OBJECT_LABEL.value,
                    int,
                    required=False,
                ),
            ]
            for result in self.source_rows().iter_row_mappings():
                bin_counts = self.json_object_mapping(result[bin_counts_field.name])
                bin_percentages = self.json_object_mapping(
                    result[bin_percentages_field.name]
                )
                object_classes = self.json_object_mapping(
                    result[object_classes_field.name]
                )
                slice_index = int(result[slice_field.name])
                total_objects = int(result[total_objects_field.name])
                bin_names = tuple(str(name) for name in bin_counts)
                summary_row: dict[str, object] = {
                    slice_field.name: slice_index,
                }
                for bin_name, count in bin_counts.items():
                    count_feature = feature_template.OBJECTS_PER_BIN.feature_name(
                        bin_name=str(bin_name)
                    )
                    percent_feature = feature_template.PERCENT_PER_BIN.feature_name(
                        bin_name=str(bin_name)
                    )
                    summary_row[count_feature] = int(count)
                    summary_row[percent_feature] = float(bin_percentages[bin_name])
                    fields.extend(
                        (
                            feature_template.OBJECTS_PER_BIN.field_spec(count_feature),
                            feature_template.PERCENT_PER_BIN.field_spec(
                                percent_feature
                            ),
                        )
                    )
                rows.append(summary_row)
                rows.extend(
                    self.object_class_rows(
                        object_classes=object_classes,
                        bin_names=bin_names,
                        slice_index=slice_index,
                        total_objects=total_objects,
                    )
                )
                fields.extend(
                    feature_template.OBJECT_CLASS.field_spec(
                        feature_template.OBJECT_CLASS.feature_name(bin_name=bin_name)
                    )
                    for bin_name in bin_names
                )
            return MeasurementSparseColumnarRows.from_rows(
                rows,
                fields=FieldSpec.merge_exact(
                    (fields,),
                    context="ClassifyObjects fields",
                ),
            )

        def object_class_rows(
            self,
            *,
            object_classes: RuntimeCallableKwargs,
            bin_names: tuple[str, ...],
            slice_index: int,
            total_objects: int,
        ) -> list[dict[str, object]]:
            if self.object_name is None:
                return []
            feature_template = self.feature_template_type
            class_labels = tuple(sorted(int(label) for label in object_classes))
            dense_labels = tuple(range(1, total_objects + 1))
            object_labels = tuple(dict.fromkeys((*dense_labels, *class_labels)))
            return [
                {
                    MeasurementRowAxisField.OBJECT_NAME.value: self.object_name,
                    MeasurementRowAxisField.OBJECT_LABEL.value: object_label,
                    MeasurementRowAxisField.SLICE_INDEX.value: slice_index,
                    **{
                        feature_template.OBJECT_CLASS.feature_name(
                            bin_name=bin_name
                        ): int(
                            str(object_label) in object_classes
                            and object_classes[str(object_label)] == bin_name
                        )
                        for bin_name in bin_names
                    },
                }
                for object_label in object_labels
            ]

    @classmethod
    def _classification_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "RuntimeCallableKwargs":
        if cls.uses_two_measurements(module):
            return cls._two_measurement_kwargs(module, binder)
        return cls._single_measurement_kwargs(module, binder)

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind scalar identities and the module's conditional classification rows."""

        bound = cls._bind_declared_settings(module, binder=binder)
        settings_kwargs = cls._classification_kwargs(module, binder)
        bound_kwargs = dict(bound.kwargs)
        if "classification_rules" in settings_kwargs:
            bound_kwargs.pop(
                cls.single_measurement_feature_binding.require_parameter_name(),
                None,
            )
        bound = bound.with_replaced_kwargs({**bound_kwargs, **settings_kwargs})
        bound = bound.with_consumed_settings(
            cls.classification_decision_count_setting,
            cls.single_measurement_feature_setting,
            cls.first_measurement_feature_setting,
            cls.second_measurement_feature_setting,
            cls.bin_spacing_setting,
            cls.bin_count_setting,
            cls.low_threshold_setting,
            cls.high_threshold_setting,
            cls.wants_low_bin_setting,
            cls.wants_high_bin_setting,
            cls.custom_thresholds_setting,
            cls.bin_names_setting,
            cls.give_bin_names_setting,
            cls.threshold_method_setting,
            cls.threshold_value_setting,
            cls.low_low_bin_name_setting,
            cls.low_high_bin_name_setting,
            cls.high_low_bin_name_setting,
            cls.high_high_bin_name_setting,
            cls.retain_image_setting,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

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
        method = coerce_cellprofiler_enum(
            ClassificationMethod,
            cls.indexed_setting_value(
                module,
                cls.classification_decision_count_setting,
                default=cls.classification_decision_default,
            ),
        )
        return method is ClassificationMethod.TWO_MEASUREMENTS

    @classmethod
    def function_name_for_module(cls, module: "ModuleBlock") -> str:
        if cls.uses_two_measurements(module):
            return cls.function_variants[0]
        return str(cls.function_name)

    @classmethod
    def resolve_function(
        cls,
        module: "ModuleBlock",
        *,
        contract: "CallableContract",
        source_bindings: "StepSourceBindingsConfig",
    ) -> Callable[..., object]:
        del contract, source_bindings
        return cls.require_callable(cls.function_name_for_module(module))

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation,
        step_context,
    ) -> tuple[ModuleBlock, ...]:
        """Restore complete repeated classification groups from public behavior."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        rules = invocation.kwargs_dict.get("classification_rules")
        if rules is None:
            retained_image_name = normalized_symbol_name(
                str(invocation.kwargs_dict.get("retained_image_name") or "")
            )
            return tuple(
                    cls._block_with_scalar_classification_output(
                        block,
                        retained_image_name=retained_image_name,
                    )
                    for block in blocks
                )
        if not isinstance(rules, tuple) or any(
            not isinstance(rule, SingleMeasurementClassificationRule)
            for rule in rules
        ):
            raise TypeError(
                "ClassifyObjects classification_rules must be a tuple of "
                "SingleMeasurementClassificationRule values."
            )
        reconstructed = []
        for block in blocks:
            records = (
                *(
                    record
                    for record in block.iter_settings()
                    if not cls._single_group_setting_name(record.name)
                    and not setting_name_matches(record.name, "Hidden")
                ),
                ModuleSetting("Hidden", str(len(rules))),
                *(
                    record
                    for rule in rules
                    for record in cls._single_rule_setting_records(rule)
                ),
            )
            reconstructed.append(
                replace(
                    block,
                    setting_records=list(records),
                )
            )
        return tuple(reconstructed)

    @classmethod
    def _block_with_scalar_classification_output(
        cls,
        block,
        *,
        retained_image_name: str | None,
    ):
        records = tuple(
            record
            for record in block.iter_settings()
            if not setting_name_matches(record.name, "Hidden")
            and not setting_name_matches(record.name, cls.retain_image_setting)
            and not setting_name_matches(record.name, cls.output_image_setting)
        )
        records = (
            *records,
            ModuleSetting("Hidden", "1"),
            ModuleSetting(
                cls.retain_image_setting.canonical,
                "Yes" if retained_image_name is not None else "No",
            ),
            *(
                ()
                if retained_image_name is None
                else (
                    ModuleSetting(
                        cls.output_image_setting.canonical,
                        retained_image_name,
                    ),
                )
            ),
        )
        return replace(
            block,
            setting_records=list(records),
        )

    @classmethod
    def _single_group_setting_name(cls, name: str) -> bool:
        return any(
            setting_name_matches(name, setting)
            for setting in (
                cls.single_measurement_feature_setting,
                cls.bin_spacing_setting,
                cls.bin_count_setting,
                cls.low_threshold_setting,
                cls.high_threshold_setting,
                cls.wants_low_bin_setting,
                cls.wants_high_bin_setting,
                cls.custom_thresholds_setting,
                cls.give_bin_names_setting,
                cls.bin_names_setting,
                cls.retain_image_setting,
                cls.output_image_setting,
            )
        )

    @classmethod
    def _single_rule_setting_records(
        cls,
        rule: "SingleMeasurementClassificationRule",
    ) -> tuple[ModuleSetting, ...]:
        feature_name = rule.measurement_feature
        bin_choice = rule.bin_choice
        bin_names = rule.bin_names
        retained_image_name = rule.retained_image_name
        return (
            ModuleSetting(
                cls.single_measurement_feature_setting.canonical, feature_name
            ),
            ModuleSetting(
                cls.bin_spacing_setting.canonical,
                (
                    "Custom-defined bins"
                    if bin_choice is ClassificationBinChoice.CUSTOM
                    else "Evenly spaced bins"
                ),
            ),
            ModuleSetting(
                cls.bin_count_setting.canonical,
                cellprofiler_setting_literal(rule.bin_count),
            ),
            ModuleSetting(
                cls.low_threshold_setting.canonical,
                cellprofiler_setting_literal(rule.low_threshold),
            ),
            ModuleSetting(
                cls.wants_low_bin_setting.canonical,
                cellprofiler_setting_literal(rule.wants_low_bin),
            ),
            ModuleSetting(
                cls.high_threshold_setting.canonical,
                cellprofiler_setting_literal(rule.high_threshold),
            ),
            ModuleSetting(
                cls.wants_high_bin_setting.canonical,
                cellprofiler_setting_literal(rule.wants_high_bin),
            ),
            ModuleSetting(
                cls.custom_thresholds_setting.canonical,
                ",".join(
                    cellprofiler_setting_literal(value)
                    for value in rule.custom_thresholds
                ),
            ),
            ModuleSetting(
                cls.give_bin_names_setting.canonical,
                "Yes" if bin_names else "No",
            ),
            ModuleSetting(
                cls.bin_names_setting.canonical,
                "" if bin_names is None else ",".join(bin_names),
            ),
            ModuleSetting(
                cls.retain_image_setting.canonical,
                "Yes" if retained_image_name is not None else "No",
            ),
            *(
                ()
                if retained_image_name is None
                else (
                    ModuleSetting(
                        cls.output_image_setting.canonical,
                        retained_image_name,
                    ),
                )
            ),
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
    def _bin_choice(value: str) -> "ClassificationBinChoice":
        return coerce_cellprofiler_enum(ClassificationBinChoice, value)

    @staticmethod
    def _threshold_method(value: str) -> "ClassificationThresholdMethod":
        return coerce_cellprofiler_enum(ClassificationThresholdMethod, value)

    @classmethod
    def _single_measurement_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "RuntimeCallableKwargs":
        measurement_features = setting_values(
            module, cls.single_measurement_feature_setting
        )
        if len(measurement_features) > 1:
            return {
                "classification_rules": tuple(
                    (
                        cls._single_measurement_rule(module, binder, index)
                        for index in range(len(measurement_features))
                    )
                )
            }
        return asdict(cls._single_measurement_rule(module, binder, 0))

    @classmethod
    def _single_measurement_rule(
        cls, module: "ModuleBlock", binder: "SettingsBinder", value_index: int
    ) -> "SingleMeasurementClassificationRule":
        bin_names = cls.indexed_setting_value(
            module,
            cls.bin_names_setting,
            default=cls.bin_names_default,
            value_index=value_index,
        )
        give_bin_names = bool(
            cls._typed_setting_value(
                module,
                binder,
                cls.give_bin_names_setting,
                default="Yes" if bin_names else "No",
                value_index=value_index,
            )
        )
        custom_thresholds = tuple(
            float(value.strip())
            for value in cls.indexed_setting_value(
                module,
                cls.custom_thresholds_setting,
                default=cls.custom_thresholds_default,
                value_index=value_index,
            ).split(",")
            if value.strip()
        )
        parsed_bin_names = (
            tuple(name.strip() for name in bin_names.split(",") if name.strip())
            if give_bin_names and bin_names
            else None
        )
        return SingleMeasurementClassificationRule(
            measurement_feature=cls._required_indexed_setting_value(
                module,
                cls.single_measurement_feature_setting,
                default=cls.measurement_feature_default,
                value_index=value_index,
            ),
            bin_choice=cls._bin_choice(
                cls.indexed_setting_value(
                    module,
                    cls.bin_spacing_setting,
                    default=cls.bin_spacing_default,
                    value_index=value_index,
                )
            ),
            bin_count=cls._typed_setting_value(
                module,
                binder,
                cls.bin_count_setting,
                default=cls.bin_count_default,
                value_index=value_index,
            ),
            low_threshold=cls._typed_setting_value(
                module,
                binder,
                cls.low_threshold_setting,
                default=cls.low_threshold_default,
                value_index=value_index,
            ),
            high_threshold=cls._typed_setting_value(
                module,
                binder,
                cls.high_threshold_setting,
                default=cls.high_threshold_default,
                value_index=value_index,
            ),
            wants_low_bin=cls._typed_setting_value(
                module,
                binder,
                cls.wants_low_bin_setting,
                default=cls.wants_low_bin_default,
                value_index=value_index,
            ),
            wants_high_bin=cls._typed_setting_value(
                module,
                binder,
                cls.wants_high_bin_setting,
                default=cls.wants_high_bin_default,
                value_index=value_index,
            ),
            custom_thresholds=custom_thresholds,
            bin_names=parsed_bin_names,
            retained_image_name=cls._single_measurement_retained_image_name(
                module,
                value_index,
            ),
        )

    @classmethod
    def _single_measurement_retained_image_name(
        cls,
        module: "ModuleBlock",
        value_index: int,
    ) -> str | None:
        retain_values = setting_values(module, cls.retain_image_setting)
        output_names = setting_values(module, cls.output_image_setting)
        if value_index >= len(retain_values):
            return None
        if not parse_cellprofiler_bool(retain_values[value_index]):
            return None
        if value_index >= len(output_names):
            raise ValueError(
                "ClassifyObjects retained classification group "
                f"{value_index + 1} has no output image name."
            )
        output_name = normalized_symbol_name(output_names[value_index])
        if output_name is None:
            raise ValueError(
                "ClassifyObjects retained classification group "
                f"{value_index + 1} has a blank output image name."
            )
        return output_name

    @classmethod
    def _two_measurement_kwargs(
        cls, module: "ModuleBlock", binder: "SettingsBinder"
    ) -> "RuntimeCallableKwargs":
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
            if cls.classified_image_outputs(module)
            or binding is not cls.output_image_binding
        )

    @classmethod
    def classified_image_outputs(
        cls,
        module: "ModuleBlock",
    ) -> tuple[ClassifiedImageOutput, ...]:
        """Return every active repeated classified-image row in rule order."""

        if cls.uses_two_measurements(module):
            return cls._two_measurement_classified_image_outputs(module)
        records = module.iter_settings()
        if records:
            blocks = repeating_setting_blocks(
                records,
                start_name=cls.single_measurement_feature_setting,
            )
            hidden_counts = setting_values(module, "Hidden")
            if hidden_counts:
                declared_count = int(float(hidden_counts[0]))
                if len(blocks) != declared_count:
                    raise ValueError(
                        "ClassifyObjects repeated classification settings do not "
                        f"match their declared count: {len(blocks)} != "
                        f"{declared_count}."
                    )
            return tuple(
                output
                for rule_index, block in enumerate(blocks)
                if (
                    output := cls._classified_image_output_from_block(
                        block,
                        rule_index=rule_index,
                    )
                )
                is not None
            )
        measurement_features = setting_values(
            module,
            cls.single_measurement_feature_setting,
        )
        return tuple(
            ClassifiedImageOutput(rule_index, output_name)
            for rule_index in range(len(measurement_features))
            if (
                output_name := cls._single_measurement_retained_image_name(
                    module,
                    rule_index,
                )
            )
            is not None
        )

    @classmethod
    def _classified_image_output_from_block(
        cls,
        block,
        *,
        rule_index: int,
    ) -> ClassifiedImageOutput | None:
        retain_literal = block_setting_value(block, cls.retain_image_setting)
        if not retain_literal or not parse_cellprofiler_bool(retain_literal):
            return None
        output_name = normalized_symbol_name(
            block_setting_value(block, cls.output_image_setting)
        )
        if output_name is None:
            raise ValueError(
                "ClassifyObjects retained classification group "
                f"{rule_index + 1} has no output image name."
            )
        return ClassifiedImageOutput(rule_index, output_name)

    @classmethod
    def _two_measurement_classified_image_outputs(
        cls,
        module: "ModuleBlock",
    ) -> tuple[ClassifiedImageOutput, ...]:
        records = module.iter_settings()
        if records:
            blocks = repeating_setting_blocks(
                records,
                start_name="Select the object name",
            )
            if not blocks:
                return ()
            output = cls._classified_image_output_from_block(
                blocks[0],
                rule_index=0,
            )
            return () if output is None else (output,)
        retain_values = setting_values(module, cls.retain_image_setting)
        output_values = setting_values(module, cls.output_image_setting)
        if not retain_values or not parse_cellprofiler_bool(retain_values[-1]):
            return ()
        output_name = normalized_symbol_name(output_values[-1] if output_values else "")
        if output_name is None:
            raise ValueError(
                "ClassifyObjects two-measurement retained image has no output name."
            )
        return (ClassifiedImageOutput(0, output_name),)

    @classmethod
    def artifact_names_for_binding(cls, module, binding):
        if binding is cls.output_image_binding:
            return tuple(output.name for output in cls.classified_image_outputs(module))
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
        inherited = super().artifact_output_relations(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
            binding=binding,
            name=name,
            artifact_inputs=artifact_inputs,
            output_position=output_position,
        )
        source_relations = tuple(
            relation
            for relation in inherited
            if isinstance(relation, SourceStackLineageSourceRelation)
        )
        if len(source_relations) != 1:
            raise ValueError(
                "ClassifyObjects classified images require one exact source "
                f"lineage relation, got {source_relations!r}."
            )
        outputs = cls.classified_image_outputs(module)
        if output_position >= len(outputs) or outputs[output_position].name != name:
            raise ValueError(
                "ClassifyObjects image output position does not match its repeated "
                f"rule declaration: position={output_position}, name={name!r}, "
                f"outputs={outputs!r}."
            )
        source_relation = source_relations[0]
        return (
            *(relation for relation in inherited if relation is not source_relation),
            ClassifiedImageSourceRelation(
                source=source_relation.source,
                rule_index=outputs[output_position].rule_index,
            ),
        )


from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import json
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.memory.decorators import numpy
from openhcs.core.registry_strategies import enum_member_with_payload
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_measurements import ObjectLabelMeasurementValues
from openhcs.core.runtime_object_labels import (
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
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum


class ClassificationMethod(Enum):
    """CellProfiler ClassifyObjects measurement-count mode."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "ClassificationMethod":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    SINGLE_MEASUREMENT = ("single_measurement", "Single measurement")
    TWO_MEASUREMENTS = (
        "two_measurements",
        "Pair of measurements",
        "Two measurements",
    )


class ClassificationThresholdMethod(Enum):
    """CellProfiler ClassifyObjects threshold selection mode."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "ClassificationThresholdMethod":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    MEAN = ("mean", "Mean")
    MEDIAN = ("median", "Median")
    CUSTOM = ("custom", "Custom")


class ClassificationBinChoice(Enum):
    """CellProfiler ClassifyObjects bin spacing mode."""

    def __new__(
        cls, absorbed_value: str, *cellprofiler_literals: str
    ) -> "ClassificationBinChoice":
        return enum_member_with_payload(
            cls,
            absorbed_value,
            payload_attribute="cellprofiler_literals",
            payload=(absorbed_value, *cellprofiler_literals),
        )

    EVEN = ("even", "Evenly spaced bins")
    CUSTOM = ("custom", "Custom-defined bins")


@dataclass(frozen=True, slots=True)
class SingleMeasurementClassificationRule(RuntimeSliceInvariantValue):
    """One typed ClassifyObjects single-measurement policy."""

    measurement_feature: str
    bin_choice: ClassificationBinChoice = ClassificationBinChoice.EVEN
    bin_count: int = 3
    low_threshold: float = 0.0
    high_threshold: float = 1.0
    wants_low_bin: bool = False
    wants_high_bin: bool = False
    custom_thresholds: tuple[float, ...] = (0.0, 1.0)
    bin_names: tuple[str, ...] | None = None
    retained_image_name: str | None = None

    def __post_init__(self) -> None:
        if not self.custom_thresholds:
            raise ValueError(
                "SingleMeasurementClassificationRule requires at least one "
                "custom threshold."
            )
        if self.bin_names is not None and any(
            not name.strip() for name in self.bin_names
        ):
            raise ValueError(
                "SingleMeasurementClassificationRule bin names must not be blank."
            )


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


class _ClassificationResultFieldRole(str, Enum):
    """Routing roles carried by the classification producer row schema."""

    OBJECT_DOMAIN_SIZE = "object_domain_size"


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Results from object classification."""

    total_objects: Annotated[
        int,
        _ClassificationResultFieldRole.OBJECT_DOMAIN_SIZE,
    ]
    bin_counts: Annotated[
        str,
        ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECTS_PER_BIN,
    ]
    bin_percentages: Annotated[
        str,
        ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.PERCENT_PER_BIN,
    ]
    object_classes: Annotated[
        str,
        ClassifyObjectsSingleMeasurementModule.MeasurementFeatureTemplate.OBJECT_CLASS,
    ] = "{}"
    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX] = 0

    @classmethod
    def empty(cls, *, total_objects: int = 0) -> "ClassificationResult":
        """Return an empty classification result row."""
        return cls(
            total_objects=total_objects,
            bin_counts=json.dumps({}),
            bin_percentages=json.dumps({}),
        )

    @classmethod
    def columnar(
        cls,
        results: "ClassificationResult | tuple[ClassificationResult, ...]",
    ) -> DataclassMeasurementColumnarRows:
        """Return exact raw classification fields for runtime aggregation."""

        rows = results if isinstance(results, tuple) else (results,)
        return DataclassMeasurementColumnarRows(rows, row_type=cls)


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

    rule: SingleMeasurementClassificationRule
    measurement_values: np.ndarray | None = None

    def classify(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        backend: ObjectClassificationBackendStrategy,
    ) -> tuple[np.ndarray, ClassificationResult]:
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        if self.rule.bin_choice is ClassificationBinChoice.EVEN:
            low_threshold = self.rule.low_threshold
            high_threshold = self.rule.high_threshold
            if low_threshold >= high_threshold:
                low_threshold, high_threshold = (high_threshold, low_threshold)
            thresholds = np.linspace(
                low_threshold,
                high_threshold,
                self.rule.bin_count + 1,
            )
        else:
            thresholds = np.asarray(self.rule.custom_thresholds, dtype=float)
        threshold_list = []
        if self.rule.wants_low_bin:
            threshold_list.append(-np.inf)
        threshold_list.extend(thresholds.tolist())
        if self.rule.wants_high_bin:
            threshold_list.append(np.inf)
        thresholds = np.array(threshold_list)
        num_bins = len(thresholds) - 1
        if self.rule.bin_names is not None:
            names = list(self.rule.bin_names)
        else:
            names = [f"Bin_{index + 1}" for index in range(num_bins)]
        while len(names) < num_bins:
            names.append(f"Bin_{len(names) + 1}")
        if num_objects == 0:
            return (
                np.zeros_like(labels, dtype=np.int32),
                classification_result_from_bins(
                    unique_labels,
                    np.zeros(0, dtype=np.int32),
                    names,
                ),
            )
        if self.measurement_values is None:
            values = backend.mean_intensity_values(labels, image, unique_labels)
        else:
            values = ClassificationMeasurementVector.from_value(
                self.measurement_values
            ).aligned_to_labels(unique_labels)
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
            backend.apply_object_bins(labels, unique_labels, object_bins),
            classification_result_from_bins(unique_labels, object_bins, names),
        )


@dataclass(frozen=True, slots=True)
class TwoMeasurementClassificationRequest:
    """Semantic request for two-measurement object classification."""

    measurement1_values: np.ndarray | None = None
    measurement2_values: np.ndarray | None = None
    threshold1_method: ClassificationThresholdMethod = (
        ClassificationThresholdMethod.MEAN
    )
    threshold1_value: float = 0.5
    threshold2_method: ClassificationThresholdMethod = (
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
        unique_labels = backend.positive_label_ids(labels)
        num_objects = len(unique_labels)
        names = [
            self.low_low_name,
            self.high_low_name,
            self.low_high_name,
            self.high_high_name,
        ]
        if num_objects == 0:
            return (
                labels,
                classification_result_from_bins(
                    unique_labels,
                    np.zeros(0, dtype=np.int32),
                    names,
                ),
            )
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
        t1 = classification_threshold(
            values1,
            self.threshold1_method,
            self.threshold1_value,
        )
        t2 = classification_threshold(
            values2,
            self.threshold2_method,
            self.threshold2_value,
        )
        high1 = values1 >= t1
        high2 = values2 >= t2
        has_nan = np.isnan(values1) | np.isnan(values2)
        object_class = np.zeros(num_objects, dtype=np.int32)
        object_class[~high1 & ~high2 & ~has_nan] = 1
        object_class[high1 & ~high2 & ~has_nan] = 2
        object_class[~high1 & high2 & ~has_nan] = 3
        object_class[high1 & high2 & ~has_nan] = 4
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
        bin_names = [f"Intensity_Bin_{index + 1}" for index in range(self.num_bins)]
        if num_objects == 0:
            return (
                labels,
                classification_result_from_bins(
                    unique_labels,
                    np.zeros(0, dtype=np.int32),
                    bin_names,
                ),
            )
        values = backend.mean_intensity_values(labels, image, unique_labels)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if len(valid_values) == 0:
            return (
                labels,
                classification_result_from_bins(
                    unique_labels,
                    np.zeros(num_objects, dtype=np.int32),
                    bin_names,
                ),
            )
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
        return (
            labels,
            classification_result_from_bins(unique_labels, object_bins, bin_names),
        )


def classification_threshold(
    values: np.ndarray, method: ClassificationThresholdMethod, custom_value: float
) -> float:
    """Return the threshold for one ClassifyObjects measurement vector."""
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
@runtime_bound_parameters(
    _ClassificationMeasurementValuesRuntimeParameter,
    _ClassificationRuleValuesRuntimeParameter,
    _ClassifiedImageRuleIndicesRuntimeParameter,
    SliceIndexRuntimeParameter,
)
def classify_objects_single_measurement(
    image: np.ndarray,
    labels: ObjectLabelValue,
    measurement_feature: str = "",
    measurement_values: np.ndarray | None = None,
    measurement_values_by_rule: tuple[np.ndarray, ...] = (),
    classification_rules: tuple[SingleMeasurementClassificationRule, ...] = (),
    bin_choice: ClassificationBinChoice = ClassificationBinChoice.EVEN,
    bin_count: int = 3,
    low_threshold: float = 0.0,
    high_threshold: float = 1.0,
    wants_low_bin: bool = False,
    wants_high_bin: bool = False,
    custom_thresholds: tuple[float, ...] = (0.0, 1.0),
    bin_names: tuple[str, ...] | None = None,
    retained_image_name: str | None = None,
    classified_image_rule_indices: tuple[int, ...] = (),
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[RuntimeArrayData | AlignedImageStack, DataclassMeasurementColumnarRows]:
    """Classify objects based on one measurement or declared rule rows.

    Args:
        labels: Object regions assigned to measurement bins or rule classes.
        classification_rules: Ordered typed rules; each rule may override the
            single-measurement bin controls for one classified output.
        bin_choice: Use evenly spaced boundaries or explicit custom limits.
        bin_count: Number of equal-width bins between the low and high thresholds.
        low_threshold: Lower edge of the evenly spaced classification range.
        high_threshold: Upper edge of the evenly spaced classification range.
        wants_low_bin: Add a bin for values at or below the configured range.
        wants_high_bin: Add a bin for values above the configured range.
        custom_thresholds: Ascending bin boundaries used by the custom bin policy.
        bin_names: Optional display names in resulting bin order.
    """
    labels = object_label_dense_array(labels, dtype=np.int32)
    image_array = np.asarray(image)
    if labels.ndim != 2:
        raise ValueError(
            "ClassifyObjects requires labels already projected to one 2-D plane, "
            f"got {labels.shape!r}."
        )
    requires_intensity_image = (
        any(
            rule_index >= len(measurement_values_by_rule)
            or measurement_values_by_rule[rule_index] is None
            for rule_index in range(len(classification_rules))
        )
        if classification_rules
        else measurement_values is None
    )
    if requires_intensity_image and (
        image_array.ndim != 2 or labels.shape != image_array.shape
    ):
        raise ValueError(
            "ClassifyObjects intensity image and projected labels must be 2-D and "
            "share a shape; got "
            f"image {image_array.shape!r} and labels {labels.shape!r}."
        )
    backend = object_classification_backend(
        backend_provider=classification_backend_provider
    )
    classified_image_metadata = image_payload_metadata(image).replace_fields(
        source_channel_axis=-1
    )
    if classification_rules:
        results: list[ClassificationResult] = []
        classified_images: list[RuntimeArrayData] = []
        for rule_index, rule in enumerate(classification_rules):
            rule_values = (
                measurement_values_by_rule[rule_index]
                if rule_index < len(measurement_values_by_rule)
                else None
            )
            classified_labels, result = SingleMeasurementClassificationRequest(
                rule=rule,
                measurement_values=rule_values,
            ).classify(image, labels, backend)
            classified_images.append(
                with_image_payload_data(
                    image,
                    classification_rgb_image(classified_labels),
                    metadata=classified_image_metadata,
                )
            )
            results.append(result)
        configured_output_indices = tuple(
            rule_index
            for rule_index, rule in enumerate(classification_rules)
            if rule.retained_image_name is not None
        )
        if classified_image_rule_indices != configured_output_indices:
            raise ValueError(
                "ClassifyObjects declared image outputs do not match active "
                "classification rules: "
                f"{classified_image_rule_indices!r} != {configured_output_indices!r}."
            )
        output = (
            pack_aligned_image_outputs(
                tuple(
                    classified_images[index] for index in classified_image_rule_indices
                )
            )
            if classified_image_rule_indices
            else labels
        )
        return (output, ClassificationResult.columnar(tuple(results)))
    classified_labels, result = SingleMeasurementClassificationRequest(
        rule=SingleMeasurementClassificationRule(
            measurement_feature=measurement_feature,
            bin_choice=bin_choice,
            bin_count=bin_count,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            wants_low_bin=wants_low_bin,
            wants_high_bin=wants_high_bin,
            custom_thresholds=custom_thresholds,
            bin_names=bin_names,
            retained_image_name=retained_image_name,
        ),
        measurement_values=measurement_values,
    ).classify(image, labels, backend)
    configured_output_indices = () if retained_image_name is None else (0,)
    if classified_image_rule_indices != configured_output_indices:
        raise ValueError(
            "ClassifyObjects declared image outputs do not match the active "
            f"classification rule: {classified_image_rule_indices!r} != "
            f"{configured_output_indices!r}."
        )
    output = (
        with_image_payload_data(
            image,
            classification_rgb_image(classified_labels),
            metadata=classified_image_metadata,
        )
        if classified_image_rule_indices
        else labels
    )
    return (output, ClassificationResult.columnar(result))


def classification_rgb_image(classified_labels: np.ndarray) -> np.ndarray:
    """Render CP's zero-background classified-label colormap image."""

    import matplotlib
    from matplotlib.cm import ScalarMappable

    labels = np.asarray(classified_labels, dtype=np.int32)
    bin_count = int(labels.max()) if labels.size else 0
    if bin_count <= 0:
        return np.zeros((*labels.shape, 3), dtype=float)
    scalar_mappable = ScalarMappable(cmap=matplotlib.colormaps["viridis"])
    colors = scalar_mappable.to_rgba(np.arange(bin_count) + 1)[:, :3]
    color_table = np.vstack((np.zeros((1, 3), dtype=colors.dtype), colors))
    return color_table[labels]


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
@runtime_bound_parameters(
    _ClassificationMeasurement1ValuesRuntimeParameter,
    _ClassificationMeasurement2ValuesRuntimeParameter,
)
def classify_objects_two_measurements(
    image: np.ndarray,
    labels: ObjectLabelValue,
    measurement1_feature: str = "",
    measurement2_feature: str = "",
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
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Classify objects from two measurements into four quadrants.

    Args:
        labels: Object regions assigned to one of four measurement quadrants.
        threshold1_method: Statistic or fixed-value policy for the first
            measurement split.
        threshold1_value: Explicit first split value when its policy uses a
            user-provided threshold.
        threshold2_method: Statistic or fixed-value policy for the second
            measurement split.
        threshold2_value: Explicit second split value when its policy uses a
            user-provided threshold.
        low_low_name: Name for objects below both measurement thresholds.
        low_high_name: Name for objects below the first and at or above the second
            threshold.
        high_low_name: Name for objects at or above the first and below the second
            threshold.
        high_high_name: Name for objects at or above both measurement thresholds.
    """
    labels = object_label_dense_array(labels, dtype=np.int32)
    image_array = np.asarray(image)
    if labels.ndim != 2:
        raise ValueError(
            "ClassifyObjects requires labels already projected to one 2-D plane, "
            f"got {labels.shape!r}."
        )
    if measurement1_values is None and (
        image_array.ndim != 2 or labels.shape != image_array.shape
    ):
        raise ValueError(
            "ClassifyObjects intensity image and projected labels must be 2-D and "
            "share a shape; got "
            f"image {image_array.shape!r} and labels {labels.shape!r}."
        )
    classified_labels, result = TwoMeasurementClassificationRequest(
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
        image_array,
        labels,
        object_classification_backend(backend_provider=classification_backend_provider),
    )
    return (classified_labels, ClassificationResult.columnar(result))


@numpy(contract=ProcessingContract.PURE_2D)
@special_inputs("labels")
def classify_objects_by_intensity_bins(
    image: np.ndarray,
    labels: ObjectLabelValue,
    num_bins: int = 3,
    use_percentiles: bool = True,
    classification_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Classify objects by mean intensity into evenly distributed bins."""
    labels = object_label_dense_array(labels, dtype=np.int32)
    image_array = np.asarray(image)
    if labels.ndim != 2 or image_array.ndim != 2:
        raise ValueError(
            "ClassifyObjects requires image and labels already projected to one "
            f"2-D plane, got image {image_array.shape!r} and labels {labels.shape!r}."
        )
    if labels.shape != image_array.shape:
        raise ValueError(
            "ClassifyObjects image and projected labels must share a shape; got "
            f"image {image_array.shape!r} and labels {labels.shape!r}."
        )
    classified_labels, result = IntensityBinsClassificationRequest(
        num_bins=num_bins, use_percentiles=use_percentiles
    ).classify(
        image_array,
        labels,
        object_classification_backend(backend_provider=classification_backend_provider),
    )
    return (classified_labels, ClassificationResult.columnar(result))


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
    SingleMeasurementClassificationRule,
    SingleMeasurementClassificationRequest,
    TwoMeasurementClassificationRequest,
    classify_objects_by_intensity_bins,
    classify_objects_single_measurement,
    classify_objects_two_measurements,
    classification_result_from_bins,
    classification_threshold,
    object_classification_backend,
)
