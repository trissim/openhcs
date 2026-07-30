"""
Converted from CellProfiler: UntangleWorms
Original: UntangleWorms module for untangling overlapping worms

This module untangles overlapping worms using a trained worm model.
It takes a binary image and labels the worms, untangling them and
associating all of a worm's pieces together.
"""

from __future__ import annotations

import inspect

from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    optional_setting_value,
    repeating_setting_blocks,
    required_setting_value,
    setting_name_matches,
    setting_names,
    setting_values,
)
from openhcs.interop.cellprofiler.settings_binder import (
    SettingsBinder,
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
    parse_cellprofiler_float,
    parse_cellprofiler_int,
)
import numpy as np
import re
import scipy.ndimage
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Callable, ClassVar, TypeVar
from xml.dom.minidom import parse
from metaclass_registry import AutoRegisterMeta
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    binary_opening,
    find_objects,
    label,
)
from python_introspect import set_signature_analysis_target
from openhcs.constants.constants import GroupBy, VariableComponents
from openhcs.core.memory.decorators import numpy
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    pack_aligned_image_outputs,
)
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ArtifactSpecRelation,
    ImageArtifactType,
    MeasurementsArtifactType,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import KeywordRuntimeParameter
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementRowQualifier,
    MeasurementSparseColumnarRows,
    QualifiedMeasurementColumnarRows,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.worm_measurements import (
    WormControlPointMeasurementSchema,
)
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
    MeasurementObjectRowIdentity,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelRepresentation,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.runtime_object_labels import (
    ObjectLabelPayload,
    ObjectLabelValue,
    object_label_dense_array,
    object_label_sparse_ijv_rows,
    object_label_value_with_dense_labels,
)
from openhcs.core.runtime_image_values import (
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.core.runtime_object_label_building import (
    SourceImageObjectLabelBuildRequest,
)
from openhcs.core.runtime_sparse_labels import SparseIJVLabelRows
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    object_label_input_execution_mode,
    required_variable_components,
    runtime_bound_parameters,
    special_inputs,
)
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.interop.cellprofiler.settings_binder import coerce_cellprofiler_enum
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    CurrentPayloadMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
    measurement_table_rows,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.distance_propagation_numba import (
    _propagate_labels_and_distances_zero_image_numba,
)
from openhcs.processing.backends.cellprofiler.object_images import (
    object_label_colormap,
)
from openhcs.processing.backends.cellprofiler.worm_geometry import (
    branchpoints,
    calculate_cumulative_lengths,
    control_points_for_label_image,
    endpoints,
    eight_connectivity,
    rebuild_worm_from_control_points_approx,
    sample_control_points,
    skeletonize_worm_mask,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.runtime.output_record_request import (
        CellProfilerOutputRecordRequest,
    )

MAX_CLUSTER_PATHS = 400
MAX_CLUSTER_PATH_SETS_CONSIDERED = 50_000


class OverlapStyle(str, Enum):
    WITH_OVERLAP = "with_overlap"
    WITHOUT_OVERLAP = "without_overlap"
    BOTH = "both"


WormOutput = TypeVar("WormOutput")


class WormLabelOutputStrategy(
    EnumKeyedStrategyMixin[OverlapStyle], ABC, metaclass=AutoRegisterMeta
):
    """Own the exact UntangleWorms output topology for one overlap style."""

    __registry_key__ = "overlap_style_label"
    __skip_if_no_key__ = True
    __enum_member_attr__ = "overlap_style"
    __enum_label_attr__ = "overlap_style_label"
    overlap_style_label: ClassVar[str | None] = None
    overlap_style: ClassVar[OverlapStyle | None] = None
    callable_name: ClassVar[str]

    @classmethod
    def for_overlap_style(
        cls, overlap_style: OverlapStyle
    ) -> "WormLabelOutputStrategy":
        return cls.for_enum_member(overlap_style)

    @abstractmethod
    def select(
        self,
        *,
        overlapping: WormOutput,
        nonoverlapping: WormOutput,
    ) -> tuple[WormOutput, ...]:
        """Select ordered values from the two declared object roles."""


class WithOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.WITH_OVERLAP
    callable_name = "untangle_worms_with_overlap"

    def select(
        self,
        *,
        overlapping: WormOutput,
        nonoverlapping: WormOutput,
    ) -> tuple[WormOutput, ...]:
        del nonoverlapping
        return (overlapping,)


class WithoutOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.WITHOUT_OVERLAP
    callable_name = "untangle_worms"

    def select(
        self,
        *,
        overlapping: WormOutput,
        nonoverlapping: WormOutput,
    ) -> tuple[WormOutput, ...]:
        del overlapping
        return (nonoverlapping,)


class BothOverlapWormLabelOutputStrategy(WormLabelOutputStrategy):
    overlap_style = OverlapStyle.BOTH
    callable_name = "untangle_worms_both"

    def select(
        self,
        *,
        overlapping: WormOutput,
        nonoverlapping: WormOutput,
    ) -> tuple[WormOutput, ...]:
        return (overlapping, nonoverlapping)


class UntangleWormsModule(
    CurrentPayloadMeasurementRecordMixin,
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
):
    module_name = "UntangleWorms"
    function_name = "untangle_worms"
    function_variants = tuple(
        strategy.callable_name
        for strategy in WormLabelOutputStrategy.__registry__.values()
        if strategy.overlap_style is not OverlapStyle.WITHOUT_OVERLAP
    )
    validated = True
    confidence = 1.0

    calculated_measurement_feature_prefixes = (
        ("worm",),
        ("fat", "regions"),
        ("mean", "fat", "regions"),
    )
    input_image_setting = SettingNameFamily(
        "Select the input binary image", aliases=("Select the input image",)
    )
    overlapping_objects_setting = "Name the output overlapping worm objects"
    nonoverlapping_objects_setting = "Name the output non-overlapping worm objects"
    overlap_style_setting = "Overlap style"
    num_control_points_setting = "Number of control points"
    overlapping_object_output_binding = SettingToKeywordBinding.output(
        overlapping_objects_setting, ObjectLabelsArtifactType
    )
    nonoverlapping_object_output_binding = SettingToKeywordBinding.output(
        nonoverlapping_objects_setting, ObjectLabelsArtifactType
    )
    retain_overlapping_outline_setting = "Retain outlines of the overlapping objects?"
    overlapping_outline_colormap_setting = "Outline colormap?"
    overlapping_outline_name_setting = "Name the overlapped outline image"
    retain_nonoverlapping_outline_setting = (
        "Retain outlines of the non-overlapping worms?"
    )
    nonoverlapping_outline_name_setting = "Name the non-overlapped outlines image"
    overlapping_outline_output_binding = SettingToKeywordBinding.output(
        overlapping_outline_name_setting,
        ImageArtifactType,
        "overlapping_outline_name",
    )
    nonoverlapping_outline_output_binding = SettingToKeywordBinding.output(
        nonoverlapping_outline_name_setting,
        ImageArtifactType,
        "nonoverlapping_outline_name",
    )
    training_file_name_setting = "Training set file name"
    training_file_location_setting = "Training set file location"
    use_training_weights_setting = "Use training set weights?"
    execution_mode_setting = "Train or untangle worms?"
    overlap_weight_setting = "Overlap weight"
    leftover_weight_setting = "Leftover weight"
    minimum_area_percentile_setting = "Minimum area percentile"
    minimum_area_factor_setting = "Minimum area factor"
    maximum_area_percentile_setting = "Maximum area percentile"
    maximum_area_factor_setting = "Maximum area factor"
    minimum_length_percentile_setting = "Minimum length percentile"
    minimum_length_factor_setting = "Minimum length factor"
    maximum_length_percentile_setting = "Maximum length percentile"
    maximum_length_factor_setting = "Maximum length factor"
    maximum_cost_percentile_setting = "Maximum cost percentile"
    maximum_cost_factor_setting = "Maximum cost factor"
    maximum_radius_percentile_setting = "Maximum radius percentile"
    maximum_radius_factor_setting = "Maximum radius factor"
    maximum_complexity_setting = "Maximum complexity"
    custom_complexity_setting = "Custom complexity"
    training_adjustment_settings = (
        minimum_area_percentile_setting,
        minimum_area_factor_setting,
        maximum_area_percentile_setting,
        maximum_area_factor_setting,
        minimum_length_percentile_setting,
        minimum_length_factor_setting,
        maximum_length_percentile_setting,
        maximum_length_factor_setting,
        maximum_cost_percentile_setting,
        maximum_cost_factor_setting,
        maximum_radius_percentile_setting,
        maximum_radius_factor_setting,
    )
    overlap_style_binding = SettingToKeywordBinding(
        overlap_style_setting,
        "overlap_style",
        lambda value: coerce_overlap_style(value),
    )
    num_control_points_binding = SettingToKeywordBinding(
        num_control_points_setting,
        "num_control_points",
        parse_cellprofiler_int,
    )
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        overlapping_object_output_binding,
        nonoverlapping_object_output_binding,
        overlapping_outline_output_binding,
        nonoverlapping_outline_output_binding,
        overlap_style_binding,
        num_control_points_binding,
        SettingToKeywordBinding(
            overlap_weight_setting,
            "overlap_weight",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            leftover_weight_setting,
            "leftover_weight",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            retain_overlapping_outline_setting,
            "retain_overlapping_outline",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            overlapping_outline_colormap_setting,
            "overlapping_outline_colormap",
        ),
        SettingToKeywordBinding(
            retain_nonoverlapping_outline_setting,
            "retain_nonoverlapping_outline",
            parse_cellprofiler_bool,
        ),
    )
    training_parameter_tags: ClassVar[tuple[tuple[str, str, type], ...]] = (
        ("min-area", "min_worm_area", float),
        ("max-area", "max_worm_area", float),
        ("cost-threshold", "cost_threshold", float),
        ("num-control-points", "num_control_points", int),
        ("max-radius", "max_radius", float),
        ("max-skel-length", "max_skel_length", float),
        ("min-path-length", "min_path_length", float),
        ("max-path-length", "max_path_length", float),
        ("median-worm-area", "median_worm_area", float),
        ("overlap-weight", "overlap_weight", float),
        ("leftover-weight", "leftover_weight", float),
    )
    training_vector_tags: ClassVar[tuple[tuple[str, str], ...]] = (
        ("mean-angles", "mean_angles"),
        ("radii-from-training", "radii_from_training"),
    )
    training_matrix_tags: ClassVar[tuple[tuple[str, str], ...]] = (
        ("inv-angles-covariance-matrix", "inv_angles_covariance_matrix"),
    )

    @classmethod
    def overlap_output_strategy(
        cls,
        module: ModuleBlock,
    ) -> WormLabelOutputStrategy:
        """Return the nominal output topology selected by the module settings."""

        return WormLabelOutputStrategy.for_overlap_style(
            coerce_overlap_style(
                required_setting_value(module, cls.overlap_style_setting)
            )
        )

    @classmethod
    def active_artifact_bindings(
        cls,
        module: ModuleBlock | None = None,
        *,
        invocation_key: "FunctionInvocationKey | None" = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        """Return object and outline outputs active for the overlap style."""

        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        object_outputs = cls.overlap_output_strategy(module).select(
            overlapping=cls.overlapping_object_output_binding,
            nonoverlapping=cls.nonoverlapping_object_output_binding,
        )
        outline_outputs = cls.overlap_output_strategy(module).select(
            overlapping=(
                cls.overlapping_outline_output_binding
                if cls._retains_outline(
                    module,
                    cls.retain_overlapping_outline_setting,
                )
                else None
            ),
            nonoverlapping=(
                cls.nonoverlapping_outline_output_binding
                if cls._retains_outline(
                    module,
                    cls.retain_nonoverlapping_outline_setting,
                )
                else None
            ),
        )
        selected = frozenset(
            (*object_outputs, *(item for item in outline_outputs if item is not None))
        )
        declared_outputs = frozenset(
            cls.declared_artifact_bindings(plan_type=ArtifactOutputPlan)
        )
        return tuple(
            binding
            for binding in bindings
            if binding not in declared_outputs or binding in selected
        )

    @classmethod
    def _retains_outline(
        cls,
        module: ModuleBlock,
        setting_name: str,
    ) -> bool:
        value = optional_setting_value(module, setting_name)
        return value is not None and parse_cellprofiler_bool(value)

    @classmethod
    def resolve_function(
        cls,
        module: ModuleBlock,
        *,
        contract,
        source_bindings,
    ) -> Callable[..., object]:
        """Select the callable whose static label ABI matches the overlap style."""

        del contract, source_bindings
        return cls.require_callable(cls.overlap_output_strategy(module).callable_name)

    @classmethod
    def bind_settings(
        cls,
        module: "ModuleBlock",
        *,
        binder: SettingsBinder,
    ) -> BoundModuleSettings:
        """Expand one training model into ordinary typed callable kwargs."""

        execution_mode = optional_setting_value(module, cls.execution_mode_setting)
        if execution_mode is not None and execution_mode.casefold() != "untangle":
            raise NotImplementedError("UntangleWorms training mode is interactive.")
        use_training_weights = optional_setting_value(
            module,
            cls.use_training_weights_setting,
        )
        if use_training_weights is not None and not parse_cellprofiler_bool(
            use_training_weights
        ):
            raise NotImplementedError(
                "UntangleWorms custom percentile/factor weights are not supported."
            )
        maximum_complexity = optional_setting_value(
            module,
            cls.maximum_complexity_setting,
        )
        if (
            maximum_complexity is not None
            and maximum_complexity.casefold() != "process all clusters"
        ):
            raise NotImplementedError(
                "UntangleWorms custom cluster complexity is not supported."
            )

        training_kwargs = cls.training_parameter_kwargs(module, binder=binder)
        if use_training_weights is not None and not training_kwargs:
            raise FileNotFoundError(
                f"UntangleWorms({module.module_num}) could not load training model "
                f"{optional_setting_value(module, cls.training_file_name_setting)!r}."
            )
        bound = cls._bind_declared_settings(module, binder=binder)
        bound = bound.with_kwargs(training_kwargs).with_consumed_settings(
            cls.execution_mode_setting,
            cls.training_file_location_setting,
            cls.training_file_name_setting,
            cls.use_training_weights_setting,
            cls.maximum_complexity_setting,
            cls.custom_complexity_setting,
            *cls.training_adjustment_settings,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

    @classmethod
    def measurement_record_rows(
        cls, request: "CellProfilerOutputRecordRequest"
    ) -> ColumnarRows:
        rows = measurement_table_rows(request.output_value)
        object_name_field = MeasurementRowAxisField.OBJECT_NAME.value
        object_number_field = MeasurementRowAxisField.OBJECT_NUMBER.value
        field_names = frozenset(field_spec.name for field_spec in rows.fields)
        row_batches: list[ColumnarRows] = []
        if rows.row_count():
            if (
                object_name_field in field_names
                or object_number_field not in field_names
            ):
                row_batches.append(rows)
            else:
                row_batches.extend(
                    QualifiedMeasurementColumnarRows(
                        rows,
                        (
                            MeasurementRowQualifier(
                                field_name=object_name_field,
                                value=object_name,
                            ),
                        ),
                    )
                    for object_name in (
                        object_spec.name
                        for object_spec in cls.measurement_object_output_specs_for_request(
                            request
                        )
                    )
                )
        row_batches.append(super().measurement_record_rows(request))
        return ConcatenatedColumnarRows(tuple(row_batches))

    @classmethod
    def training_parameter_kwargs(
        cls,
        module: "ModuleBlock",
        *,
        binder: SettingsBinder,
    ) -> dict[str, float | int | tuple[Any, ...]]:
        file_name = optional_setting_value(module, cls.training_file_name_setting)
        if not file_name:
            return {}
        training_path = binder.resolve_source_file(file_name)
        doc = parse(str(training_path))
        kwargs: dict[str, float | int | tuple[Any, ...]] = {}
        for tag_name, parameter_name, coerce in cls.training_parameter_tags:
            elements = doc.documentElement.getElementsByTagName(tag_name)
            if len(elements) != 1:
                continue
            text = "".join(
                (
                    node.data
                    for node in elements[0].childNodes
                    if node.nodeType == doc.TEXT_NODE
                )
            ).strip()
            if text:
                kwargs[parameter_name] = (
                    coerce(float(text)) if coerce is int else coerce(text)
                )
        for tag_name, parameter_name in cls.training_vector_tags:
            values = cls.xml_vector_values(doc, tag_name)
            if values:
                kwargs[parameter_name] = values
        for tag_name, parameter_name in cls.training_matrix_tags:
            rows = cls.xml_matrix_values(doc, tag_name)
            if rows:
                kwargs[parameter_name] = rows
        return kwargs

    @classmethod
    def xml_vector_values(cls, doc: Any, tag_name: str) -> tuple[float, ...]:
        elements = doc.documentElement.getElementsByTagName(tag_name)
        if len(elements) != 1:
            return ()
        return tuple(
            (
                cls.xml_float(value_element, doc)
                for value_element in elements[0].getElementsByTagName("value")
            )
        )

    @classmethod
    def xml_matrix_values(
        cls, doc: Any, tag_name: str
    ) -> tuple[tuple[float, ...], ...]:
        elements = doc.documentElement.getElementsByTagName(tag_name)
        if len(elements) != 1:
            return ()
        return tuple(
            (
                tuple(
                    (
                        cls.xml_float(value_element, doc)
                        for value_element in values_element.getElementsByTagName(
                            "value"
                        )
                    )
                )
                for values_element in elements[0].getElementsByTagName("values")
            )
        )

    @staticmethod
    def xml_float(element: Any, doc: Any) -> float:
        text = "".join(
            (node.data for node in element.childNodes if node.nodeType == doc.TEXT_NODE)
        ).strip()
        return float(text)


class _StraightenWormControlPointsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound control points reconstructed from producer measurements."""

    parameter_name = "control_points"
    annotation_type = np.ndarray | None
    parameter_default = None


class StraightenWormsSpecialInputPolicy:
    """Resolve worm labels plus producer-derived control points."""

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        object_inputs = request.object_inputs
        measurement_inputs = request.declared_measurement_specs
        bound = super().bind_runtime_inputs(request)
        if not measurement_inputs:
            return bound
        if "num_control_points" in request.kwargs:
            num_control_points = int(request.kwargs["num_control_points"])
        else:
            parameter = inspect.signature(request.func).parameters["num_control_points"]
            if parameter.default is inspect.Parameter.empty:
                raise ValueError(
                    f"{cls.module_name} requires num_control_points to reconstruct "
                    "producer control-point measurements."
                )
            num_control_points = int(parameter.default)
        control_points = WormControlPointMeasurementSchema(
            num_control_points=num_control_points
        ).control_points_from_rows(
            request.runtime_value_for_spec(measurement_inputs[0]),
            object_name=object_inputs[0].name,
        )
        if control_points is not None:
            bound[
                _StraightenWormControlPointsRuntimeParameter.require_parameter_name()
            ] = control_points
        return bound


class FlipMode(Enum):
    """CellProfiler head/tail alignment policy for straightened worms."""

    NONE = "do_not_align"
    TOP = "top_brightest"
    BOTTOM = "bottom_brightest"
    MANUAL = "flip_manually"


class StraightenWormsModule(
    StraightenWormsSpecialInputPolicy,
    ObjectArtifactInputModule,
    ObjectArtifactOutputModule,
    MeasurementArtifactOutputModule,
):
    module_name = "StraightenWorms"
    function_name = "straighten_worms"
    validated = True
    group_by = GroupBy.SITE
    confidence = 1.0

    input_objects_setting = "Select the input untangled worm objects"
    output_objects_setting = "Name the output straightened worm objects"
    input_image_setting = "Select an input image to straighten"
    output_image_setting = "Name the output straightened image"
    worm_width_setting = "Worm width"
    measure_intensity_setting = "Measure intensity distribution?"
    transverse_segments_setting = "Number of transverse segments"
    longitudinal_stripes_setting = "Number of longitudinal stripes"
    alignment_setting = "Align worms?"
    alignment_image_setting = "Alignment image"
    image_count_setting = "Image count"
    training_file_location_setting = "Training set file location"
    training_file_name_setting = "Training set file name"
    input_objects_binding = SettingToKeywordBinding.input(
        input_objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="worm_labels",
    )
    output_objects_binding = SettingToKeywordBinding.output(
        output_objects_setting, ObjectLabelsArtifactType
    )
    input_image_binding = SettingToKeywordBinding.input(
        input_image_setting, ImageArtifactType, repeated=True
    )
    output_image_binding = SettingToKeywordBinding.output(
        output_image_setting, ImageArtifactType, repeated=True
    )
    worm_width_binding = SettingToKeywordBinding(
        worm_width_setting,
        "worm_width",
        parse_cellprofiler_int,
    )
    measure_intensity_binding = SettingToKeywordBinding(
        measure_intensity_setting,
        "measure_intensity",
        parse_cellprofiler_bool,
    )
    transverse_segments_binding = SettingToKeywordBinding(
        transverse_segments_setting,
        "number_of_segments",
        parse_cellprofiler_int,
    )
    longitudinal_stripes_binding = SettingToKeywordBinding(
        longitudinal_stripes_setting,
        "number_of_stripes",
        parse_cellprofiler_int,
    )
    alignment_binding = SettingToKeywordBinding(
        alignment_setting,
        "flip_mode",
        lambda value: coerce_cellprofiler_enum(FlipMode, value),
    )
    setting_bindings = (
        worm_width_binding,
        measure_intensity_binding,
        transverse_segments_binding,
        longitudinal_stripes_binding,
        alignment_binding,
        input_objects_binding,
        output_objects_binding,
        input_image_binding,
        output_image_binding,
    )

    @classmethod
    def bind_settings(cls, module: ModuleBlock, *, binder) -> BoundModuleSettings:
        """Bind ordered image rows and their shared alignment semantics."""

        image_bindings = cls.image_bindings(module)
        image_count = optional_setting_value(module, cls.image_count_setting)
        if image_count is not None and int(float(image_count)) != len(image_bindings):
            raise ValueError(
                f"StraightenWorms({module.module_num}) declares image count "
                f"{image_count!r} but contains {len(image_bindings)} image rows."
            )

        output_names = tuple(binding.output_image_name for binding in image_bindings)
        kwargs: dict[str, Any] = {
            cls.output_image_binding.require_parameter_name(): (
                output_names[0] if len(output_names) == 1 else output_names
            )
        }
        alignment_image = optional_setting_value(module, cls.alignment_image_setting)
        if alignment_image is not None:
            matching_indexes = tuple(
                index
                for index, binding in enumerate(image_bindings)
                if binding.input_image_name == alignment_image
            )
            if len(matching_indexes) != 1:
                raise ValueError(
                    f"StraightenWorms({module.module_num}) alignment image "
                    f"{alignment_image!r} must select exactly one input image."
                )
            kwargs["alignment_image_index"] = matching_indexes[0]

        training_name = optional_setting_value(
            module,
            cls.training_file_name_setting,
        )
        if training_name is not None:
            training_kwargs = UntangleWormsModule.training_parameter_kwargs(
                module,
                binder=binder,
            )
            if not training_kwargs:
                raise FileNotFoundError(
                    f"StraightenWorms({module.module_num}) could not load training "
                    f"model {training_name!r}."
                )
            if "num_control_points" in training_kwargs:
                kwargs["num_control_points"] = training_kwargs["num_control_points"]

        bound = cls._bind_declared_settings(module, binder=binder)
        bound = bound.with_kwargs(kwargs).with_consumed_settings(
            cls.alignment_image_setting,
            cls.image_count_setting,
            cls.training_file_location_setting,
            cls.training_file_name_setting,
        )
        return cls._finalize_bound_settings(module, binder=binder, bound=bound)

    @classmethod
    def artifact_output_relations(
        cls,
        module: ModuleBlock,
        *,
        invocation_key,
        step_context,
        binding,
        name,
        artifact_inputs: ArtifactSpecCollection,
        output_position: int,
    ):
        """Anchor each straightened artifact to its exact declared input."""

        if binding is cls.output_objects_binding:
            del name, invocation_key, step_context, output_position
            source = artifact_inputs.require_by_name_and_artifact_type(
                cls.input_objects_name(module),
                ObjectLabelsArtifactType,
            )
            return (SourceStackLineageSourceRelation(source=source.ref()),)
        del invocation_key, step_context, name
        image_bindings = cls.image_bindings(module)
        if output_position >= len(image_bindings):
            raise ValueError(
                f"StraightenWorms({module.module_num}) output position "
                f"{output_position} has no declared image row."
            )
        source = artifact_inputs.require_by_name_and_artifact_type(
            image_bindings[output_position].input_image_name,
            ImageArtifactType,
        )
        return (SourceStackLineageSourceRelation(source=source.ref()),)

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ModuleBlock, ...]:
        """Reconstruct all ordered input/output image pairs exactly once."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        reconstructed: list[ModuleBlock] = []
        output_offset = 0
        for block in blocks:
            rebuilt = cls._block_with_image_bindings(
                block,
                output_offset=output_offset,
                step_context=step_context,
            )
            reconstructed.append(rebuilt)
            output_offset += len(cls.image_bindings(rebuilt))
        return tuple(reconstructed)

    @classmethod
    def derives_missing_output_identity(
        cls,
        binding: SettingToKeywordBinding,
    ) -> bool:
        """Leave repeated image-output cardinality with its row owner."""

        return (
            False
            if binding is cls.output_image_binding
            else super().derives_missing_output_identity(binding)
        )

    @classmethod
    def _block_with_image_bindings(
        cls,
        block: ModuleBlock,
        *,
        output_offset: int,
        step_context: ArtifactDeclarationStepContext,
    ) -> ModuleBlock:
        input_names = cls._required_image_names(
            block,
            setting=cls.input_image_setting,
        )
        output_names = cls._image_names(
            block,
            setting=cls.output_image_setting,
        )
        if not output_names:
            output_names = tuple(
                cls.canonical_output_artifact_name(
                    artifact_type=ImageArtifactType,
                    output_position=output_position,
                    block_position=output_offset,
                    step_context=step_context,
                )
                for output_position in range(len(input_names))
            )
        if len(output_names) != len(input_names):
            raise ValueError(
                "StraightenWorms public image identity kwargs must contain "
                "the same number of input and output image names."
            )

        records = [
            record
            for record in block.iter_settings()
            if not (
                setting_name_matches(
                    record.name,
                    cls.input_image_binding.setting_name,
                )
                or setting_name_matches(
                    record.name,
                    cls.output_image_binding.setting_name,
                )
            )
        ]
        for input_name, output_name in zip(input_names, output_names, strict=True):
            records.extend(
                (
                    ModuleSetting(cls.input_image_setting, input_name),
                    ModuleSetting(cls.output_image_setting, output_name),
                )
            )
        return replace(
            block,
            setting_records=records,
        )

    @classmethod
    def _required_image_names(
        cls,
        block: ModuleBlock,
        *,
        setting: str | SettingNameFamily,
    ) -> tuple[str, ...]:
        names = cls._image_names(block, setting=setting)
        if not names:
            raise ValueError(
                "StraightenWorms row reconstruction requires at least one "
                f"value for {setting_names(setting)[0]!r}."
            )
        return names

    @classmethod
    def _image_names(
        cls,
        block: ModuleBlock,
        *,
        setting: str | SettingNameFamily,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for value in setting_values(block, setting)
            if (name := normalized_symbol_name(value)) is not None
        )

    @dataclass(frozen=True, slots=True)
    class ImageBinding:
        input_image_name: str
        output_image_name: str

    @classmethod
    def input_objects_name(cls, module: "ModuleBlock") -> str:
        return required_setting_value(module, cls.input_objects_binding.setting_name)

    @classmethod
    def image_bindings(
        cls, module: "ModuleBlock"
    ) -> tuple["StraightenWormsModule.ImageBinding", ...]:
        blocks = repeating_setting_blocks(
            module.iter_settings(), start_name=cls.input_image_binding.setting_name
        )
        bindings = tuple(cls._image_binding(module, block) for block in blocks)
        if not bindings:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no "
                "StraightenWorms image rows."
            )
        return bindings

    @classmethod
    def _image_binding(
        cls,
        module: ModuleBlock,
        block: Sequence[ModuleSetting],
    ) -> "StraightenWormsModule.ImageBinding":
        input_name = normalized_symbol_name(
            block_setting_value(block, cls.input_image_binding.setting_name)
        )
        output_name = normalized_symbol_name(
            block_setting_value(block, cls.output_image_binding.setting_name)
        )
        if input_name is None or output_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) requires an input and "
                "output image name for every StraightenWorms row."
            )
        return cls.ImageBinding(input_name, output_name)

    @classmethod
    def producer_measurement_input(
        cls,
        object_input: ArtifactSpec,
        *,
        step_context: "ArtifactDeclarationStepContext",
    ) -> ArtifactSpec | None:
        """Select the measurement artifact related to one exact object output."""

        if object_input.artifact_type is not ObjectLabelsArtifactType:
            raise TypeError(
                f"{cls.__name__} requires an object-label input, got "
                f"{object_input.artifact_type.value}."
            )
        object_output_ref = object_input.ref().for_plan_type(ArtifactOutputPlan)
        producer_measurements = ArtifactSpecCollection(
            spec
            for spec, relation in step_context.available_artifacts.relation_refs(
                ArtifactSpecRelation
            )
            if spec.artifact_type is MeasurementsArtifactType
            and relation.source == object_output_ref
        ).unique(conflict_context=f"{cls.__name__} producer measurement")
        if not producer_measurements:
            return None
        if len(producer_measurements) != 1:
            raise ValueError(
                f"{cls.__name__} found multiple measurement artifacts related to "
                f"{object_output_ref!r}: "
                f"{tuple(spec.ref() for spec in producer_measurements)!r}."
            )
        return producer_measurements[0].for_plan_type(ArtifactInputPlan)

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: FunctionInvocationKey,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ArtifactSpec, ...]:
        input_objects_name = cls.input_objects_name(module)
        inputs = super().artifact_contract_inputs(
            module,
            invocation_key=invocation_key,
            step_context=step_context,
        )
        input_objects = ArtifactSpecCollection(
            inputs
        ).require_by_name_and_artifact_type(
            input_objects_name,
            ObjectLabelsArtifactType,
        )
        producer_measurements = cls.producer_measurement_input(
            input_objects,
            step_context=step_context,
        )
        if producer_measurements is None:
            return inputs
        return (*inputs, producer_measurements)


@dataclass(frozen=True, slots=True)
class WormMeasurement:
    """StraightenWorms per-object intensity row."""

    slice_index: int
    object_number: int
    center_x: float
    center_y: float
    mean_intensity: float
    std_intensity: float


@dataclass(frozen=True, slots=True)
class DeadWormStats(MeasurementFeatureRecord):
    """IdentifyDeadWorms summary carrier retained for runtime diagnostics."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_count: int
    mean_center_x: float
    mean_center_y: float
    mean_angle: float


@dataclass(frozen=True, slots=True)
class DeadWormAngleMeasurement(MeasurementFeatureRecord):
    """IdentifyDeadWorms per-object angle measurement."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    object_label: Annotated[int, MeasurementRowAxisField.OBJECT_LABEL]
    angle: float


@dataclass(frozen=True, slots=True)
class StraightenedWormPlacement:
    """Source-to-output mapping for one straightened worm block."""

    object_number: int
    output_y: slice
    output_x: slice
    source_y: np.ndarray
    source_x: np.ndarray


@dataclass(frozen=True, slots=True)
class StraightenWormControlPoints:
    """Control-point normalization policy for StraightenWorms."""

    points: np.ndarray | None
    labels: np.ndarray
    num_control_points: int

    @property
    def normalized(self) -> np.ndarray:
        if self.points is None:
            return control_points_for_label_image(self.labels, self.num_control_points)
        points = np.asarray(self.points, dtype=float)
        if points.ndim != 3:
            raise ValueError(
                "StraightenWorms control_points must have shape (objects, 2, control_points) or (2, control_points, objects)."
            )
        if points.shape[1] == 2:
            normalized = points
        elif points.shape[0] == 2:
            normalized = points.transpose(2, 0, 1)
        else:
            raise ValueError(
                "StraightenWorms control_points must include one coordinate axis of length 2."
            )
        if normalized.shape[2] != self.num_control_points:
            raise ValueError(
                f"StraightenWorms expected {self.num_control_points} control points; got {normalized.shape[2]}."
            )
        return normalized


@dataclass(frozen=True, slots=True)
class StraightenWormsSliceRequest:
    """Executable StraightenWorms request for one 2-D runtime slice."""

    image: np.ndarray
    labels: np.ndarray
    control_points: np.ndarray
    worm_width: int
    num_control_points: int
    flip_mode: FlipMode
    measure_intensity: bool
    slice_index: int

    @property
    def half_width(self) -> int:
        return self.worm_width // 2

    @property
    def output_width(self) -> int:
        return 2 * self.half_width + 1

    @property
    def positive_labels(self) -> np.ndarray:
        labels = np.unique(self.labels)
        return labels[labels > 0]

    def planned_placements(self) -> list[StraightenedWormPlacement]:
        """Return object placements using this slice's alignment intensities."""

        unique_labels = self.positive_labels
        if len(unique_labels) == 0:
            return []
        lengths = self.worm_lengths(len(unique_labels))
        if not lengths:
            return []
        return self.placements(unique_labels, lengths)

    def execute(
        self,
        placements: list[StraightenedWormPlacement] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, DataclassMeasurementColumnarRows]:
        image = self.image
        resolved_placements = (
            self.planned_placements() if placements is None else placements
        )
        if not resolved_placements:
            shape = (self.output_width, self.output_width)
            return (
                np.zeros(shape, dtype=image.dtype),
                np.zeros(shape, dtype=np.int32),
                DataclassMeasurementColumnarRows((), row_type=WormMeasurement),
            )
        shape = (
            max(placement.output_y.stop for placement in resolved_placements),
            max(placement.output_x.stop for placement in resolved_placements),
        )
        straightened_image = np.zeros(shape, dtype=image.dtype)
        straightened_labels = np.zeros(shape, dtype=np.int32)
        self.apply_placements(
            straightened_image,
            straightened_labels,
            resolved_placements,
        )
        return (
            straightened_image,
            straightened_labels,
            self.measurements(
                straightened_image,
                straightened_labels,
                resolved_placements,
            ),
        )

    def worm_lengths(self, worm_count: int) -> list[int]:
        lengths: list[int] = []
        for index in range(min(worm_count, self.control_points.shape[0])):
            control_point = self.control_points[index]
            lengths.append(
                int(np.ceil(calculate_cumulative_lengths(control_point.T)[-1]))
            )
        return lengths

    def placements(
        self, unique_labels: np.ndarray, lengths: list[int]
    ) -> list[StraightenedWormPlacement]:
        placements: list[StraightenedWormPlacement] = []
        for index, object_number in enumerate(unique_labels):
            if index >= len(lengths) or lengths[index] == 0:
                continue
            if index >= self.control_points.shape[0]:
                continue
            placements.append(
                self.placement_for_object(
                    object_number=int(object_number),
                    object_index=index,
                    length=lengths[index],
                )
            )
        return placements

    def placement_for_object(
        self, *, object_number: int, object_index: int, length: int
    ) -> StraightenedWormPlacement:
        control_point = self.control_points[object_index]
        ii = control_point[0]
        jj = control_point[1]
        t_orig = np.linspace(0, length, self.num_control_points)
        t_new = np.arange(0, length + 1)
        ci = np.interp(t_new, t_orig, ii)
        cj = np.interp(t_new, t_orig, jj)
        di = np.diff(ci, prepend=ci[0])
        dj = np.diff(cj, prepend=cj[0])
        di[0] = di[1] if len(di) > 1 else 0
        dj[0] = dj[1] if len(dj) > 1 else 0
        norm = np.sqrt(di**2 + dj**2)
        norm[norm == 0] = 1
        ni = -dj / norm
        nj = di / norm
        half_width = self.half_width
        ci_ext = np.concatenate(
            [
                np.arange(-half_width, 0) * nj[0] + ci[0],
                ci,
                np.arange(1, half_width + 1) * nj[-1] + ci[-1],
            ]
        )
        cj_ext = np.concatenate(
            [
                np.arange(-half_width, 0) * -ni[0] + cj[0],
                cj,
                np.arange(1, half_width + 1) * -ni[-1] + cj[-1],
            ]
        )
        ni_ext = np.concatenate([[ni[0]] * half_width, ni, [ni[-1]] * half_width])
        nj_ext = np.concatenate([[nj[0]] * half_width, nj, [nj[-1]] * half_width])
        iii, jjj = np.mgrid[0 : len(ci_ext), -half_width : half_width + 1]
        source_y = ci_ext[iii] + ni_ext[iii] * jjj
        source_x = cj_ext[iii] + nj_ext[iii] * jjj
        if self.should_flip(object_number, ci_ext, cj_ext, ni_ext, nj_ext, iii, jjj):
            iii_flip = len(ci_ext) - iii - 1
            jjj_flip = -jjj
            source_y = ci_ext[iii_flip] + ni_ext[iii_flip] * jjj_flip
            source_x = cj_ext[iii_flip] + nj_ext[iii_flip] * jjj_flip
        return StraightenedWormPlacement(
            object_number=object_number,
            output_y=slice(0, len(ci_ext)),
            output_x=slice(
                self.output_width * object_index, self.output_width * (object_index + 1)
            ),
            source_y=np.ascontiguousarray(source_y, dtype=float),
            source_x=np.ascontiguousarray(source_x, dtype=float),
        )

    def should_flip(
        self,
        object_number: int,
        ci_ext: np.ndarray,
        cj_ext: np.ndarray,
        ni_ext: np.ndarray,
        nj_ext: np.ndarray,
        iii: np.ndarray,
        jjj: np.ndarray,
    ) -> bool:
        if self.flip_mode is FlipMode.NONE:
            return False
        source_y = ci_ext[iii] + ni_ext[iii] * jjj
        source_x = cj_ext[iii] + nj_ext[iii] * jjj
        sampled_image = scipy.ndimage.map_coordinates(
            self.image, [source_y, source_x], order=1, mode="constant"
        )
        sampled_mask = scipy.ndimage.map_coordinates(
            (self.labels == object_number).astype(np.float32),
            [source_y, source_x],
            order=0,
        )
        sampled_image = sampled_image * sampled_mask
        halfway = len(ci_ext) // 2
        area_top = np.sum(sampled_mask[:halfway, :])
        area_bottom = np.sum(sampled_mask[halfway:, :])
        if area_top <= 0 or area_bottom <= 0:
            return False
        top_intensity = np.sum(sampled_image[:halfway, :]) / area_top
        bottom_intensity = np.sum(sampled_image[halfway:, :]) / area_bottom
        return (
            self.flip_mode is FlipMode.TOP
            and top_intensity < bottom_intensity
            or (self.flip_mode is FlipMode.BOTTOM and bottom_intensity < top_intensity)
        )

    def apply_placements(
        self,
        straightened_image: np.ndarray,
        straightened_labels: np.ndarray,
        placements: list[StraightenedWormPlacement],
    ) -> None:
        if not placements:
            return
        flat_source_y = np.concatenate(
            [placement.source_y.ravel() for placement in placements]
        )
        flat_source_x = np.concatenate(
            [placement.source_x.ravel() for placement in placements]
        )
        flat_image = scipy.ndimage.map_coordinates(
            self.image, [flat_source_y, flat_source_x], order=1, mode="constant"
        )
        flat_labels = scipy.ndimage.map_coordinates(
            self.labels,
            [flat_source_y, flat_source_x],
            order=0,
            mode="constant",
            cval=0,
        )
        offset = 0
        for placement in placements:
            block_shape = placement.source_y.shape
            block_size = placement.source_y.size
            next_offset = offset + block_size
            image_block = flat_image[offset:next_offset].reshape(block_shape)
            label_block = flat_labels[offset:next_offset].reshape(block_shape)
            straightened_image[placement.output_y, placement.output_x] = image_block
            output_label_block = straightened_labels[
                placement.output_y, placement.output_x
            ]
            output_label_block[label_block == placement.object_number] = (
                placement.object_number
            )
            offset = next_offset

    def measurements(
        self,
        straightened_image: np.ndarray,
        straightened_labels: np.ndarray,
        placements: list[StraightenedWormPlacement],
    ) -> DataclassMeasurementColumnarRows:
        if not self.measure_intensity:
            return DataclassMeasurementColumnarRows((), row_type=WormMeasurement)
        measurements: list[WormMeasurement] = []
        for placement in placements:
            mask = (
                straightened_labels[placement.output_y, placement.output_x]
                == placement.object_number
            )
            if np.sum(mask) == 0:
                continue
            image_block = straightened_image[placement.output_y, placement.output_x]
            values = image_block[mask]
            center_y, center_x = scipy.ndimage.center_of_mass(mask.astype(float))
            measurements.append(
                WormMeasurement(
                    slice_index=self.slice_index,
                    object_number=placement.object_number,
                    center_x=(
                        float(center_x) + float(placement.output_x.start)
                        if not np.isnan(center_x)
                        else 0.0
                    ),
                    center_y=(
                        float(center_y) + float(placement.output_y.start)
                        if not np.isnan(center_y)
                        else 0.0
                    ),
                    mean_intensity=float(np.mean(values)),
                    std_intensity=float(np.std(values)),
                )
            )
        return DataclassMeasurementColumnarRows(
            tuple(measurements),
            row_type=WormMeasurement,
        )


@dataclass(frozen=True, slots=True)
class DeadWormDiamondTemplate:
    """Diamond-shaped dead-worm structuring element at one angle."""

    worm_width: int
    worm_length: int
    angle: float

    def footprint(self) -> np.ndarray:
        from scipy.ndimage import binary_fill_holes

        x0 = int(np.sin(self.angle) * self.worm_length / 2)
        x1 = int(np.cos(self.angle) * self.worm_width / 2)
        x2 = -x0
        x3 = -x1
        y2 = int(np.cos(self.angle) * self.worm_length / 2)
        y1 = int(np.sin(self.angle) * self.worm_width / 2)
        y0 = -y2
        y3 = -y1
        xmax = np.max(np.abs([x0, x1, x2, x3]))
        ymax = np.max(np.abs([y0, y1, y2, y3]))
        footprint = np.zeros((ymax * 2 + 1, xmax * 2 + 1), bool)
        pts_y0 = np.array([y0, y1, y2, y3]) + ymax
        pts_x0 = np.array([x0, x1, x2, x3]) + xmax
        pts_y1 = np.array([y1, y2, y3, y0]) + ymax
        pts_x1 = np.array([x1, x2, x3, x0]) + xmax
        i_pts, j_pts = LineSegments.from_endpoints(
            pts_y0, pts_x0, pts_y1, pts_x1
        ).points()
        valid = (
            (i_pts >= 0)
            & (i_pts < footprint.shape[0])
            & (j_pts >= 0)
            & (j_pts < footprint.shape[1])
        )
        footprint[i_pts[valid], j_pts[valid]] = True
        return binary_fill_holes(footprint)


@dataclass(frozen=True, slots=True)
class LineSegments:
    """Integer points along one or more line segments."""

    y0: np.ndarray
    x0: np.ndarray
    y1: np.ndarray
    x1: np.ndarray

    @classmethod
    def from_endpoints(
        cls, y0: np.ndarray, x0: np.ndarray, y1: np.ndarray, x1: np.ndarray
    ) -> "LineSegments":
        return cls(y0=y0, x0=x0, y1=y1, x1=x1)

    def points(self) -> tuple[np.ndarray, np.ndarray]:
        all_i: list[int] = []
        all_j: list[int] = []
        for index in range(len(self.y0)):
            dy = abs(self.y1[index] - self.y0[index])
            dx = abs(self.x1[index] - self.x0[index])
            sy = 1 if self.y0[index] < self.y1[index] else -1
            sx = 1 if self.x0[index] < self.x1[index] else -1
            err = dx - dy
            cy = self.y0[index]
            cx = self.x0[index]
            while True:
                all_i.append(cy)
                all_j.append(cx)
                if cy == self.y1[index] and cx == self.x1[index]:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    cx += sx
                if e2 < dx:
                    err += dx
                    cy += sy
        return (np.array(all_i), np.array(all_j))


@dataclass(frozen=True, slots=True)
class ConnectedComponentEdges:
    """Union-find connected components for an integer edge list."""

    first: np.ndarray
    second: np.ndarray

    def labels(self) -> np.ndarray:
        if len(self.first) == 0:
            return np.zeros(0, dtype=int)
        vertex_count = max(np.max(self.first), np.max(self.second)) + 1
        labels = np.arange(vertex_count)

        def find(vertex: int) -> int:
            root = vertex
            while labels[root] != root:
                root = labels[root]
            while labels[vertex] != root:
                next_vertex = labels[vertex]
                labels[vertex] = root
                vertex = next_vertex
            return int(root)

        def union(first: int, second: int) -> None:
            first_root = find(first)
            second_root = find(second)
            if first_root != second_root:
                labels[first_root] = second_root

        for first, second in zip(self.first, self.second):
            union(int(first), int(second))
        for index in range(vertex_count):
            labels[index] = find(index)
        unique_labels = np.unique(labels)
        label_map = {old: new for new, old in enumerate(unique_labels)}
        return np.array([label_map[label] for label in labels])


@dataclass(frozen=True, slots=True)
class DeadWormAdjacencyPolicy:
    """CP dead-worm hit grouping policy in spatial/angle space."""

    i: np.ndarray
    j: np.ndarray
    angle: np.ndarray
    space_dist: float
    angle_dist: float

    def edges(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.i) < 2:
            return (np.zeros(0, dtype=int), np.zeros(0, dtype=int))
        order = np.lexsort((self.angle, self.j, self.i))
        i_sorted = self.i[order]
        j_sorted = self.j[order]
        angle_sorted = self.angle[order]
        first: list[int] = []
        second: list[int] = []
        for idx1 in range(len(self.i)):
            for idx2 in range(idx1 + 1, len(self.i)):
                spatial_dist_sq = (i_sorted[idx1] - i_sorted[idx2]) ** 2 + (
                    j_sorted[idx1] - j_sorted[idx2]
                ) ** 2
                if spatial_dist_sq > self.space_dist**2:
                    continue
                angle_diff = abs(angle_sorted[idx1] - angle_sorted[idx2])
                if (
                    angle_diff <= self.angle_dist
                    or np.pi - angle_diff <= self.angle_dist
                ):
                    first.append(order[idx1])
                    second.append(order[idx2])
        return (np.array(first, dtype=int), np.array(second, dtype=int))


def coerce_overlap_style(value: str | OverlapStyle) -> OverlapStyle:
    """Normalize CellProfiler overlap-style literals into the typed enum."""
    if isinstance(value, OverlapStyle):
        return value
    normalized = re.sub("[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    for style in OverlapStyle:
        literals = (style.name.lower(), style.value, style.value.replace("_", ""))
        if normalized in literals:
            return style
    raise ValueError(
        f"overlap_style must be one of {', '.join((style.value for style in OverlapStyle))}; got {value!r}."
    )


@dataclass(frozen=True, slots=True)
class WormControlPointGeometry:
    """CP-compatible geometry derived from sampled worm control points."""

    control_coords: np.ndarray
    path_length: float

    @classmethod
    def from_path_coords(
        cls,
        path_coords: np.ndarray,
        cumulative_lengths: np.ndarray,
        *,
        num_control_points: int,
    ) -> "WormControlPointGeometry":
        """Sample one path once for all downstream worm geometry consumers."""

        if len(path_coords) < 2:
            return cls(
                np.zeros((num_control_points, 2), dtype=float),
                0.0,
            )
        return cls(
            sample_control_points(
                path_coords,
                cumulative_lengths,
                num_control_points,
            ),
            float(cumulative_lengths[-1]),
        )

    @property
    def angles(self) -> np.ndarray:
        """Extract angles at each interior control point."""
        if len(self.control_coords) < 3:
            return np.array([])
        segments_delta = self.control_coords[1:] - self.control_coords[:-1]
        segment_bearings = np.arctan2(segments_delta[:, 0], segments_delta[:, 1])
        angles = segment_bearings[1:] - segment_bearings[:-1]
        angles[angles > np.pi] -= 2 * np.pi
        angles[angles < -np.pi] += 2 * np.pi
        return angles


@dataclass(frozen=True, slots=True)
class WormLabelCandidates:
    """The two semantic object-label roles produced by worm untangling."""

    overlapping: ObjectLabelValue
    nonoverlapping: ObjectLabelPayload


@dataclass(frozen=True, slots=True)
class UntangleWormsExecution:
    """Shared algorithm result lowered by the three public output ABIs."""

    image: RuntimeArrayData
    measurements: MeasurementSparseColumnarRows
    labels: WormLabelCandidates

    def canonical_image_output(
        self,
        *,
        overlap_style: OverlapStyle,
        retain_overlapping_outline: bool,
        retain_nonoverlapping_outline: bool,
        overlapping_outline_colormap: str,
    ) -> RuntimeArrayData | AlignedImageStack:
        strategy = WormLabelOutputStrategy.for_overlap_style(overlap_style)
        selected = strategy.select(
            overlapping=(
                _overlapping_worm_outline(
                    self.image,
                    self.labels.overlapping,
                    overlapping_outline_colormap,
                )
                if retain_overlapping_outline
                else None
            ),
            nonoverlapping=(
                _nonoverlapping_worm_outline(
                    self.image,
                    self.labels.nonoverlapping,
                )
                if retain_nonoverlapping_outline
                else None
            ),
        )
        retained = tuple(output for output in selected if output is not None)
        return pack_aligned_image_outputs(retained) if retained else self.image


def _execute_untangle_worms(
    image: np.ndarray,
    min_worm_area: float = 100.0,
    max_worm_area: float = 5000.0,
    num_control_points: int = 21,
    cost_threshold: float = 100.0,
    min_path_length: float = 50.0,
    max_path_length: float = 500.0,
    overlap_weight: float = 5.0,
    leftover_weight: float = 10.0,
    median_worm_area: float | None = None,
    max_radius: float | None = None,
    max_skel_length: float | None = None,
    mean_angles: tuple[float, ...] | None = None,
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None = None,
    radii_from_training: tuple[float, ...] | None = None,
) -> UntangleWormsExecution:
    """
    Untangle overlapping worms in a binary image.

    This function takes a binary image where foreground indicates worm shapes
    and attempts to identify and separate individual worms, even when they
    overlap or cross each other.

    Args:
        min_worm_area: Minimum accepted worm area in square pixels.
        max_worm_area: Maximum accepted worm area in square pixels.
        num_control_points: Number of points sampled along each candidate worm.
        cost_threshold: Maximum model-assignment cost accepted for a candidate worm.
        min_path_length: Minimum skeleton-path length in pixels.
        max_path_length: Maximum skeleton-path length in pixels.
        overlap_weight: Cost penalty for pixels assigned to overlapping worms.
        leftover_weight: Cost penalty for foreground pixels left unassigned.
        median_worm_area: Training-set median worm area in square pixels.
        max_radius: Maximum training-model radius in pixels.
        max_skel_length: Maximum accepted skeleton length in pixels.
        mean_angles: Training-set mean angle at each interior control point.
        inv_angles_covariance_matrix: Inverse covariance matrix for the training angles.
        radii_from_training: Training-set worm radii sampled at the control points.

    Returns the complete semantic result before output-topology projection.
    """
    mean_angles_array = _coerce_mean_angles(mean_angles, num_control_points)
    inv_angles_covariance_array = _coerce_inverse_covariance(
        inv_angles_covariance_matrix, num_control_points
    )
    radii_array = _coerce_worm_radii(radii_from_training, num_control_points)
    binary = image > 0
    labels, count = label(binary, structure=eight_connectivity())
    if count == 0:
        return UntangleWormsExecution(
            image=image,
            measurements=_worm_descriptor_rows(
                [],
                num_control_points=num_control_points,
            ),
            labels=_worm_label_outputs(
                [],
                source_image=image,
                image_shape=image.shape,
                radii_from_training=radii_array,
            ),
        )
    skeleton = skeletonize_worm_mask(binary)
    eroded = binary_erosion(binary, structure=eight_connectivity())
    skeleton = skeletonize_worm_mask(skeleton & eroded)
    areas = np.bincount(labels.ravel())
    component_slices = find_objects(labels)
    all_path_coords: list[np.ndarray] = []
    for i, object_slice in enumerate(component_slices, start=1):
        if object_slice is None:
            continue
        component_area = areas[i]
        if component_area < min_worm_area:
            continue
        row_slice, column_slice = object_slice
        local_labels = labels[object_slice]
        mask = local_labels == i
        component_skeleton = skeleton[object_slice] & mask
        if not np.any(component_skeleton):
            continue
        if component_area <= max_worm_area:
            path_coords = _longest_worm_graph_path_coords(
                mask, component_skeleton, max_length=max_path_length
            )
            if len(path_coords) < 2:
                continue
            cumul_lengths = calculate_cumulative_lengths(path_coords)
            if not WormShapeCostRequest(
                geometry=WormControlPointGeometry.from_path_coords(
                    path_coords,
                    cumul_lengths,
                    num_control_points=num_control_points,
                ),
                mean_angles=mean_angles_array,
                inv_angles_covariance_matrix=inv_angles_covariance_array,
            ).passes(cost_threshold):
                continue
            all_path_coords.append(
                _offset_path_coords(
                    path_coords,
                    row_offset=row_slice.start,
                    column_offset=column_slice.start,
                )
            )
        else:
            graph = WormGraphFromBinaryRequest(
                binary_image=mask,
                skeleton=component_skeleton,
                max_radius=max_radius,
                max_skel_length=max_skel_length,
            ).build()
            paths = graph.paths_between_lengths(
                min_length=min_path_length, max_length=max_path_length
            )
            all_path_coords.extend(
                (
                    _offset_path_coords(
                        path_coords,
                        row_offset=row_slice.start,
                        column_offset=column_slice.start,
                    )
                    for path_coords in WormClusterPathSelectionPolicy(
                        median_worm_area=median_worm_area,
                        component_area=int(component_area),
                        num_control_points=num_control_points,
                        mean_angles=mean_angles_array,
                        inv_angles_covariance_matrix=inv_angles_covariance_array,
                        cost_threshold=cost_threshold,
                        overlap_weight=overlap_weight,
                        leftover_weight=leftover_weight,
                        min_path_length=min_path_length,
                        max_path_length=max_path_length,
                    ).select(graph, paths)
                )
            )
    worm_geometries = tuple(
        WormControlPointGeometry.from_path_coords(
            path_coords,
            calculate_cumulative_lengths(path_coords),
            num_control_points=num_control_points,
        )
        for path_coords in all_path_coords
    )
    return UntangleWormsExecution(
        image=image,
        measurements=_worm_descriptor_rows(
            worm_geometries,
            num_control_points=num_control_points,
        ),
        labels=_worm_label_outputs(
            worm_geometries,
            source_image=image,
            image_shape=image.shape,
            radii_from_training=radii_array,
        ),
    )


def _untangle_worms_output(
    image: np.ndarray,
    *,
    expected_style: OverlapStyle,
    overlap_style: OverlapStyle,
    min_worm_area: float,
    max_worm_area: float,
    num_control_points: int,
    cost_threshold: float,
    min_path_length: float,
    max_path_length: float,
    overlap_weight: float,
    leftover_weight: float,
    median_worm_area: float | None,
    max_radius: float | None,
    max_skel_length: float | None,
    mean_angles: tuple[float, ...] | None,
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None,
    radii_from_training: tuple[float, ...] | None,
    retain_overlapping_outline: bool,
    retain_nonoverlapping_outline: bool,
    overlapping_outline_colormap: str,
) -> tuple[
    RuntimeArrayData
    | AlignedImageStack
    | MeasurementSparseColumnarRows
    | ObjectLabelValue,
    ...,
]:
    """Execute once and lower through the statically selected public ABI."""

    resolved_style = coerce_overlap_style(overlap_style)
    if resolved_style is not expected_style:
        raise ValueError(
            f"{WormLabelOutputStrategy.for_overlap_style(expected_style).callable_name} "
            f"requires overlap_style={expected_style.value!r}, got "
            f"{resolved_style.value!r}."
        )
    execution = _execute_untangle_worms(
        image,
        min_worm_area=min_worm_area,
        max_worm_area=max_worm_area,
        num_control_points=num_control_points,
        cost_threshold=cost_threshold,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        overlap_weight=overlap_weight,
        leftover_weight=leftover_weight,
        median_worm_area=median_worm_area,
        max_radius=max_radius,
        max_skel_length=max_skel_length,
        mean_angles=mean_angles,
        inv_angles_covariance_matrix=inv_angles_covariance_matrix,
        radii_from_training=radii_from_training,
    )
    labels = WormLabelOutputStrategy.for_overlap_style(resolved_style).select(
        overlapping=execution.labels.overlapping,
        nonoverlapping=execution.labels.nonoverlapping,
    )
    return (
        execution.canonical_image_output(
            overlap_style=resolved_style,
            retain_overlapping_outline=retain_overlapping_outline,
            retain_nonoverlapping_outline=retain_nonoverlapping_outline,
            overlapping_outline_colormap=overlapping_outline_colormap,
        ),
        execution.measurements,
        *labels,
    )


@numpy(contract=ProcessingContract.PURE_2D)
def untangle_worms(
    image: np.ndarray,
    overlap_style: OverlapStyle = OverlapStyle.WITHOUT_OVERLAP,
    min_worm_area: float = 100.0,
    max_worm_area: float = 5000.0,
    num_control_points: int = 21,
    cost_threshold: float = 100.0,
    min_path_length: float = 50.0,
    max_path_length: float = 500.0,
    overlap_weight: float = 5.0,
    leftover_weight: float = 10.0,
    median_worm_area: float | None = None,
    max_radius: float | None = None,
    max_skel_length: float | None = None,
    mean_angles: tuple[float, ...] | None = None,
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None = None,
    radii_from_training: tuple[float, ...] | None = None,
    retain_overlapping_outline: bool = False,
    retain_nonoverlapping_outline: bool = False,
    overlapping_outline_colormap: str = "viridis",
) -> tuple[
    RuntimeArrayData | AlignedImageStack,
    MeasurementSparseColumnarRows,
    ObjectLabelPayload,
]:
    """Untangle worms while excluding pixels shared by multiple worms."""

    output = _untangle_worms_output(
        image,
        expected_style=OverlapStyle.WITHOUT_OVERLAP,
        overlap_style=overlap_style,
        min_worm_area=min_worm_area,
        max_worm_area=max_worm_area,
        num_control_points=num_control_points,
        cost_threshold=cost_threshold,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        overlap_weight=overlap_weight,
        leftover_weight=leftover_weight,
        median_worm_area=median_worm_area,
        max_radius=max_radius,
        max_skel_length=max_skel_length,
        mean_angles=mean_angles,
        inv_angles_covariance_matrix=inv_angles_covariance_matrix,
        radii_from_training=radii_from_training,
        retain_overlapping_outline=retain_overlapping_outline,
        retain_nonoverlapping_outline=retain_nonoverlapping_outline,
        overlapping_outline_colormap=overlapping_outline_colormap,
    )
    return (output[0], output[1], output[2])


@numpy(contract=ProcessingContract.PURE_2D)
def untangle_worms_with_overlap(
    image: np.ndarray,
    overlap_style: OverlapStyle = OverlapStyle.WITH_OVERLAP,
    min_worm_area: float = 100.0,
    max_worm_area: float = 5000.0,
    num_control_points: int = 21,
    cost_threshold: float = 100.0,
    min_path_length: float = 50.0,
    max_path_length: float = 500.0,
    overlap_weight: float = 5.0,
    leftover_weight: float = 10.0,
    median_worm_area: float | None = None,
    max_radius: float | None = None,
    max_skel_length: float | None = None,
    mean_angles: tuple[float, ...] | None = None,
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None = None,
    radii_from_training: tuple[float, ...] | None = None,
    retain_overlapping_outline: bool = False,
    retain_nonoverlapping_outline: bool = False,
    overlapping_outline_colormap: str = "viridis",
) -> tuple[
    RuntimeArrayData | AlignedImageStack,
    MeasurementSparseColumnarRows,
    ObjectLabelValue,
]:
    """Untangle worms while retaining pixels shared by multiple worms."""

    output = _untangle_worms_output(
        image,
        expected_style=OverlapStyle.WITH_OVERLAP,
        overlap_style=overlap_style,
        min_worm_area=min_worm_area,
        max_worm_area=max_worm_area,
        num_control_points=num_control_points,
        cost_threshold=cost_threshold,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        overlap_weight=overlap_weight,
        leftover_weight=leftover_weight,
        median_worm_area=median_worm_area,
        max_radius=max_radius,
        max_skel_length=max_skel_length,
        mean_angles=mean_angles,
        inv_angles_covariance_matrix=inv_angles_covariance_matrix,
        radii_from_training=radii_from_training,
        retain_overlapping_outline=retain_overlapping_outline,
        retain_nonoverlapping_outline=retain_nonoverlapping_outline,
        overlapping_outline_colormap=overlapping_outline_colormap,
    )
    return (output[0], output[1], output[2])


@numpy(contract=ProcessingContract.PURE_2D)
def untangle_worms_both(
    image: np.ndarray,
    overlap_style: OverlapStyle = OverlapStyle.BOTH,
    min_worm_area: float = 100.0,
    max_worm_area: float = 5000.0,
    num_control_points: int = 21,
    cost_threshold: float = 100.0,
    min_path_length: float = 50.0,
    max_path_length: float = 500.0,
    overlap_weight: float = 5.0,
    leftover_weight: float = 10.0,
    median_worm_area: float | None = None,
    max_radius: float | None = None,
    max_skel_length: float | None = None,
    mean_angles: tuple[float, ...] | None = None,
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None = None,
    radii_from_training: tuple[float, ...] | None = None,
    retain_overlapping_outline: bool = False,
    retain_nonoverlapping_outline: bool = False,
    overlapping_outline_colormap: str = "viridis",
) -> tuple[
    RuntimeArrayData | AlignedImageStack,
    MeasurementSparseColumnarRows,
    ObjectLabelValue,
    ObjectLabelPayload,
]:
    """Untangle worms and return both overlapping and exclusive object sets."""

    output = _untangle_worms_output(
        image,
        expected_style=OverlapStyle.BOTH,
        overlap_style=overlap_style,
        min_worm_area=min_worm_area,
        max_worm_area=max_worm_area,
        num_control_points=num_control_points,
        cost_threshold=cost_threshold,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        overlap_weight=overlap_weight,
        leftover_weight=leftover_weight,
        median_worm_area=median_worm_area,
        max_radius=max_radius,
        max_skel_length=max_skel_length,
        mean_angles=mean_angles,
        inv_angles_covariance_matrix=inv_angles_covariance_matrix,
        radii_from_training=radii_from_training,
        retain_overlapping_outline=retain_overlapping_outline,
        retain_nonoverlapping_outline=retain_nonoverlapping_outline,
        overlapping_outline_colormap=overlapping_outline_colormap,
    )
    return (output[0], output[1], output[2], output[3])


for _function_name in UntangleWormsModule.declared_function_names():
    set_signature_analysis_target(
        UntangleWormsModule.require_callable(_function_name),
        _execute_untangle_worms,
    )
del _function_name


@required_variable_components(VariableComponents.CHANNEL)
@numpy(contract=ProcessingContract.FLEXIBLE)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.SLICE_ALIGNED)
@special_inputs("worm_labels")
@runtime_bound_parameters(_StraightenWormControlPointsRuntimeParameter)
def straighten_worms(
    image: np.ndarray,
    worm_labels: ObjectLabelValue,
    control_points: np.ndarray | None = None,
    worm_width: int = 20,
    num_control_points: int = 21,
    flip_mode: FlipMode = FlipMode.NONE,
    alignment_image_index: int = 0,
    number_of_segments: int = 4,
    number_of_stripes: int = 3,
    measure_intensity: bool = True,
) -> tuple[
    RuntimeArrayData | AlignedImageStack,
    ColumnarRows,
    ObjectLabelValue,
]:
    """Straighten labeled worms using sampled or provided control points.

    Args:
        worm_labels: Segmented worm regions that define each body to straighten.
        num_control_points: Number of centerline samples used to warp each worm;
            must be positive.
        alignment_image_index: Zero-based channel used to plan a common worm
            orientation for every input channel.
    """
    del number_of_segments, number_of_stripes
    if flip_mode is FlipMode.MANUAL:
        raise NotImplementedError("StraightenWorms manual flipping is interactive.")
    image_data = np.asarray(image_payload_data(image))
    image_stack = image_data[np.newaxis, :, :] if image_data.ndim == 2 else image_data
    source_metadata = image_payload_metadata(image)
    if image_stack.shape[0] > 1 and source_metadata.plane_axis is None:
        raise ValueError(
            "StraightenWorms requires a declared leading plane axis for "
            "multiple input images."
        )
    if not isinstance(worm_labels, ObjectLabelValue):
        raise TypeError(
            "StraightenWorms requires a runtime-projected ObjectLabelValue."
        )
    label_plane = object_label_dense_array(worm_labels, dtype=np.int32)
    if label_plane.ndim != 2:
        raise ValueError(
            "StraightenWorms requires one runtime-projected 2-D object-label plane; "
            f"got shape {label_plane.shape!r}, plane axis {worm_labels.plane_axis!r}, "
            f"and domain scope {worm_labels.domain.scope!r}."
        )
    if not 0 <= alignment_image_index < image_stack.shape[0]:
        raise ValueError(
            "StraightenWorms alignment_image_index must address the input image "
            f"stack, got {alignment_image_index} for {image_stack.shape[0]} images."
        )
    normalized_control_points = StraightenWormControlPoints(
        points=control_points,
        labels=label_plane,
        num_control_points=num_control_points,
    ).normalized
    placements = StraightenWormsSliceRequest(
        image=image_stack[alignment_image_index],
        labels=label_plane,
        control_points=normalized_control_points,
        worm_width=worm_width,
        num_control_points=num_control_points,
        flip_mode=flip_mode,
        measure_intensity=measure_intensity,
        slice_index=alignment_image_index,
    ).planned_placements()
    straightened_images: list[RuntimeArrayData] = []
    straightened_labels: np.ndarray | None = None
    measurement_batches: list[ColumnarRows] = []
    for slice_index in range(image_stack.shape[0]):
        slice_image, slice_labels, measurements = StraightenWormsSliceRequest(
            image=image_stack[slice_index],
            labels=label_plane,
            control_points=normalized_control_points,
            worm_width=worm_width,
            num_control_points=num_control_points,
            flip_mode=FlipMode.NONE,
            measure_intensity=measure_intensity,
            slice_index=slice_index,
        ).execute(placements)
        output_metadata = (
            source_metadata.for_leading_source_plane(slice_index)
            if source_metadata.plane_axis is not None
            else source_metadata
        )
        output_spatial_shape = output_metadata.spatial_shape_yx(slice_image)
        if output_spatial_shape is None:
            raise ValueError(
                "StraightenWorms output does not declare two spatial axes."
            )
        output_metadata = output_metadata.with_spatial_resize(output_spatial_shape)
        straightened_images.append(output_metadata.payload_with(slice_image))
        if slice_index == alignment_image_index:
            straightened_labels = slice_labels
        measurement_batches.append(measurements)
    if straightened_labels is None:
        raise RuntimeError("StraightenWorms did not execute its alignment image.")
    return (
        pack_aligned_image_outputs(straightened_images),
        ConcatenatedColumnarRows(tuple(measurement_batches)),
        object_label_value_with_dense_labels(
            worm_labels,
            straightened_labels,
            source_spatial_domain=(
                worm_labels.object_label_source_spatial_domain().with_spatial_resize(
                    straightened_labels.shape[-2:]
                )
            ),
        ),
    )


@numpy(contract=ProcessingContract.PURE_2D)
def identify_dead_worms(
    image: np.ndarray,
    worm_width: int = 10,
    worm_length: int = 100,
    angle_count: int = 32,
    auto_distance: bool = True,
    space_distance: float = 5.0,
    angular_distance: float = 30.0,
) -> tuple[np.ndarray, ConcatenatedColumnarRows, ObjectLabelValue]:
    """Identify straight dead worms by diamond-template matches across angles.

    Args:
        worm_width: Expected worm width in pixels for the diamond template.
        worm_length: Expected worm length in pixels for the diamond template.
        angle_count: Number of evenly spaced orientations tested over 180 degrees.
        auto_distance: Derive spatial and angular clustering distances from worm
            dimensions when enabled.
        space_distance: Maximum center separation in pixels when automatic spacing
            is disabled.
        angular_distance: Maximum orientation difference in degrees when automatic
            spacing is disabled.
    """
    from scipy.ndimage import binary_erosion

    mask = image > 0
    i_coords: list[np.ndarray] = []
    j_coords: list[np.ndarray] = []
    a_coords: list[np.ndarray] = []
    ig, jg = np.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
    for angle_index in range(angle_count):
        angle = float(angle_index) * np.pi / float(angle_count)
        footprint = DeadWormDiamondTemplate(
            worm_width=worm_width, worm_length=worm_length, angle=angle
        ).footprint()
        erosion = binary_erosion(mask, footprint)
        point_count = np.sum(erosion)
        if point_count <= 0:
            continue
        i_coords.append(ig[erosion])
        j_coords.append(jg[erosion])
        a_coords.append(np.ones(point_count) * angle)
    if not i_coords:
        labels = np.zeros(mask.shape, dtype=np.int32)
        return (
            image,
            ConcatenatedColumnarRows(
                (
                    DataclassMeasurementColumnarRows(
                        (DeadWormStats(0, 0, 0.0, 0.0, 0.0),),
                        row_type=DeadWormStats,
                    ),
                    DataclassMeasurementColumnarRows(
                        (),
                        row_type=DeadWormAngleMeasurement,
                    ),
                )
            ),
            SourceImageObjectLabelBuildRequest(
                image=image,
                labels=labels,
                declared_object_count=0,
            ).payload(),
        )
    i = np.concatenate(i_coords)
    j = np.concatenate(j_coords)
    a = np.concatenate(a_coords)
    if auto_distance:
        space_dist = float(worm_width)
        angle_dist = np.arctan2(worm_width, worm_length) + np.pi / angle_count
    else:
        space_dist = space_distance
        angle_dist = angular_distance * np.pi / 180.0
    first, second = DeadWormAdjacencyPolicy(
        i=i, j=j, angle=a, space_dist=space_dist, angle_dist=angle_dist
    ).edges()
    if len(first) > 0:
        ij_labels = ConnectedComponentEdges(first, second).labels() + 1
        label_count = int(np.max(ij_labels))
        label_indexes = np.arange(1, label_count + 1)
        center_x = np.array([np.mean(j[ij_labels == label]) for label in label_indexes])
        center_y = np.array([np.mean(i[ij_labels == label]) for label in label_indexes])
        angles = np.array([np.mean(a[ij_labels == label]) for label in label_indexes])
        labels = np.zeros(mask.shape, dtype=np.int32)
        labels[i, j] = ij_labels
    else:
        label_count = len(i)
        labels = np.zeros(mask.shape, dtype=np.int32)
        if label_count > 0:
            labels[i, j] = np.arange(1, label_count + 1)
            center_x = j.astype(float)
            center_y = i.astype(float)
            angles = a
        else:
            center_x = np.array([])
            center_y = np.array([])
            angles = np.array([])
    stats = DeadWormStats(
        slice_index=0,
        object_count=int(label_count),
        mean_center_x=float(np.mean(center_x)) if len(center_x) > 0 else 0.0,
        mean_center_y=float(np.mean(center_y)) if len(center_y) > 0 else 0.0,
        mean_angle=float(np.mean(angles) * 180 / np.pi) if len(angles) > 0 else 0.0,
    )
    angle_measurements = tuple(
        DeadWormAngleMeasurement(
            slice_index=0,
            object_label=object_label,
            angle=float(angle * 180 / np.pi),
        )
        for object_label, angle in enumerate(angles, start=1)
    )
    return (
        image,
        ConcatenatedColumnarRows(
            (
                DataclassMeasurementColumnarRows(
                    (stats,),
                    row_type=DeadWormStats,
                ),
                DataclassMeasurementColumnarRows(
                    angle_measurements,
                    row_type=DeadWormAngleMeasurement,
                ),
            )
        ),
        SourceImageObjectLabelBuildRequest(
            image=image,
            labels=labels,
            declared_object_count=int(label_count),
            declared_object_ids=tuple(range(1, int(label_count) + 1)),
        ).payload(),
    )


@dataclass(frozen=True, slots=True)
class WormGraphPath:
    """Ordered path through a CP worm graph."""

    segments: tuple[int, ...]
    branch_areas: tuple[int, ...]

    def to_pixel_coords(self, graph: "WormGraph") -> np.ndarray:
        if len(self.segments) == 1:
            return graph.segments[self.segments[0]][0]
        direction = graph.incidence_directions[self.branch_areas[0], self.segments[0]]
        result = [graph.segments[self.segments[0]][int(direction)]]
        for branch_area, segment in zip(
            self.branch_areas, self.segments[1:], strict=True
        ):
            direction = not graph.incidence_directions[branch_area, segment]
            result.append(graph.segments[segment][int(direction)])
        return np.vstack(result)


@dataclass(frozen=True, slots=True)
class WormGraph:
    """CP worm branch-area graph with path enumeration semantics."""

    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    segment_lengths: np.ndarray
    incidence_matrix: np.ndarray
    incidence_directions: np.ndarray
    incident_branch_areas: tuple[np.ndarray, ...]
    incident_segments: tuple[np.ndarray, ...]

    def paths_between_lengths(
        self, *, min_length: float, max_length: float
    ) -> list[WormGraphPath]:
        paths: list[WormGraphPath] = []
        for segment_index, current_length in enumerate(self.segment_lengths):
            if current_length >= min_length:
                paths.append(WormGraphPath((segment_index,), ()))
            unfinished_branches = tuple(
                (
                    (int(branch_index),)
                    for branch_index in self.incident_branch_areas[segment_index]
                )
            )
            paths.extend(
                self._paths_from(
                    unfinished_segments=(segment_index,),
                    unfinished_branch_areas=unfinished_branches,
                    current_length=float(current_length),
                    min_length=min_length,
                    max_length=max_length,
                )
            )
        return paths

    def _paths_from(
        self,
        *,
        unfinished_segments: tuple[int, ...],
        unfinished_branch_areas: tuple[tuple[int, ...], ...],
        current_length: float,
        min_length: float,
        max_length: float,
    ) -> list[WormGraphPath]:
        if not unfinished_segments:
            return []
        paths: list[WormGraphPath] = []
        last_segment = unfinished_segments[-1]
        for unfinished_branch in unfinished_branch_areas:
            end_branch = unfinished_branch[-1]
            direction = self.incidence_directions[end_branch, last_segment]
            last_coord = self.segments[last_segment][int(direction)][-1]
            for segment_index in self.incident_segments[end_branch]:
                segment_index = int(segment_index)
                if segment_index in unfinished_segments:
                    continue
                direction = not self.incidence_directions[end_branch, segment_index]
                first_coord = self.segments[segment_index][int(direction)][0]
                gap_length = float(np.sqrt(np.sum((last_coord - first_coord) ** 2)))
                next_length = (
                    current_length + gap_length + self.segment_lengths[segment_index]
                )
                if next_length > max_length:
                    continue
                next_segments = (*unfinished_segments, segment_index)
                if segment_index > unfinished_segments[0] and next_length >= min_length:
                    paths.append(WormGraphPath(next_segments, unfinished_branch))
                next_branches = tuple(
                    (
                        (*unfinished_branch, int(branch_index))
                        for branch_index in self.incident_branch_areas[segment_index]
                        if int(branch_index) != end_branch
                        and int(branch_index) not in unfinished_branch
                    )
                )
                paths.extend(
                    self._paths_from(
                        unfinished_segments=next_segments,
                        unfinished_branch_areas=next_branches,
                        current_length=float(next_length),
                        min_length=min_length,
                        max_length=max_length,
                    )
                )
        return paths


@dataclass(frozen=True, slots=True)
class WormGraphFromBinaryRequest:
    """Inputs for CP-style worm branch-area/segment graph construction."""

    binary_image: np.ndarray
    skeleton: np.ndarray
    max_radius: float | None
    max_skel_length: float | None

    def build(self) -> WormGraph:
        branch_areas = branchpoints(self.skeleton)
        if self.max_radius is not None:
            far = binary_erosion(
                self.binary_image, structure=_cellprofiler_strel_disk(self.max_radius)
            )
            far = binary_opening(far, structure=eight_connectivity())
            far_labels, _count = label(far)
            if far_labels.size:
                far_counts = np.bincount(
                    far_labels.ravel(), weights=branch_areas.ravel().astype(float)
                )
                far[far_counts[far_labels] < 2] = False
                branch_areas |= far
        branch_areas = binary_dilation(branch_areas, structure=eight_connectivity())
        segments = self.skeleton & ~branch_areas
        if self.max_skel_length is not None and np.any(segments):
            segments, branch_areas = _insert_long_segment_breakpoints(
                segments,
                branch_areas,
                max_skel_length=max(int(self.max_skel_length), 2),
            )
        return _worm_graph_from_branching_areas(branch_areas, segments)


@dataclass(frozen=True, slots=True)
class WormSegmentTrace:
    """Ordered pixels, labels, and distances for traced worm graph segments."""

    rows: np.ndarray
    columns: np.ndarray
    labels: np.ndarray
    order: np.ndarray
    distance: np.ndarray
    segment_count: int

    @classmethod
    def from_segments(cls, segments: np.ndarray) -> "WormSegmentTrace":
        segment_labels, segment_count = label(segments, structure=eight_connectivity())
        if segment_count == 0:
            empty_i = np.zeros(0, dtype=int)
            empty_distance = np.zeros(0, dtype=float)
            return cls(empty_i, empty_i, empty_i, empty_i, empty_distance, 0)

        endpoint_mask = endpoints(segments)
        order_image = np.arange(np.prod(segments.shape))
        order_image.shape = segments.shape
        order_image[~endpoint_mask] += np.prod(segments.shape)
        label_range = np.arange(segment_count + 1).astype(int)
        endpoint_loc = np.array(
            scipy.ndimage.minimum_position(order_image, segment_labels, label_range),
            dtype=int,
        )
        endpoint_labels = np.zeros(segment_labels.shape, dtype=np.int16)
        endpoint_labels[endpoint_loc[:, 0], endpoint_loc[:, 1]] = segment_labels[
            endpoint_loc[:, 0], endpoint_loc[:, 1]
        ]

        loops = ~endpoint_mask[endpoint_loc[1:, 0], endpoint_loc[1:, 1]]
        if np.any(loops):
            dilated_endpoint_labels = scipy.ndimage.grey_dilation(
                endpoint_labels,
                footprint=np.ones((3, 3), bool),
            )
            dilated_endpoint_labels[dilated_endpoint_labels != segment_labels] = 0
            loop_endpoints = np.array(
                scipy.ndimage.maximum_position(
                    order_image,
                    dilated_endpoint_labels.astype(int),
                    label_range[1:][loops],
                ),
                dtype=int,
            )
            traced_segments = segments.copy()
            traced_segments[loop_endpoints[:, 0], loop_endpoints[:, 1]] = False
        else:
            traced_segments = segments

        _propagated, distances = _propagate_labels_and_distances_zero_image_numba(
            np.ascontiguousarray(endpoint_labels, dtype=np.int32),
            np.ascontiguousarray(traced_segments, dtype=np.bool_),
            1.0,
            -1.0,
        )
        if np.any(loops):
            distances[loop_endpoints[:, 0], loop_endpoints[:, 1]] = np.inf

        rows, columns = np.mgrid[0 : segments.shape[0], 0 : segments.shape[1]]
        rows = rows[segments]
        columns = columns[segments]
        labels = segment_labels[segments]
        distance_values = distances[segments]
        sort_order = np.lexsort((distance_values, labels))
        rows = rows[sort_order]
        columns = columns[sort_order]
        labels = labels[sort_order]
        distance_values = distance_values[sort_order]
        segment_order = np.arange(len(rows), dtype=int)
        areas = np.bincount(labels)
        indexes = np.cumsum(areas) - areas
        segment_order -= indexes[labels]
        return cls(
            rows.astype(int),
            columns.astype(int),
            labels.astype(int),
            segment_order,
            distance_values,
            segment_count,
        )


def _insert_long_segment_breakpoints(
    segments: np.ndarray, branch_areas: np.ndarray, *, max_skel_length: int
) -> tuple[np.ndarray, np.ndarray]:
    trace = WormSegmentTrace.from_segments(segments)
    if trace.segment_count == 0:
        return (segments, branch_areas)
    max_order = np.zeros(trace.segment_count + 1, dtype=int)
    for label_id in range(1, trace.segment_count + 1):
        label_orders = trace.order[trace.labels == label_id]
        if len(label_orders):
            max_order[label_id] = int(np.max(label_orders))
    big_segment = max_order >= max_skel_length
    segment_count_per_label = np.maximum(
        ((max_order + max_skel_length - 1) / max_skel_length).astype(int), 1
    )
    segment_length = np.maximum(
        ((max_order + 1) / segment_count_per_label).astype(int), 1
    )
    new_breakpoints = (
        (trace.order % segment_length[trace.labels] == segment_length[trace.labels] - 1)
        & (trace.order != max_order[trace.labels])
        & big_segment[trace.labels]
    )
    if not np.any(new_breakpoints):
        return (segments, branch_areas)
    new_branch_areas = np.zeros(segments.shape, dtype=bool)
    new_branch_areas[trace.rows[new_breakpoints], trace.columns[new_breakpoints]] = True
    new_branch_areas = binary_dilation(new_branch_areas, structure=eight_connectivity())
    return (segments & ~new_branch_areas, branch_areas | new_branch_areas)


def _worm_graph_from_branching_areas(
    branch_areas: np.ndarray, segments: np.ndarray
) -> WormGraph:
    branch_labels, branch_count = label(branch_areas, structure=eight_connectivity())
    trace = WormSegmentTrace.from_segments(segments)
    if trace.segment_count == 0:
        empty_incidence = np.zeros((branch_count, 0), dtype=bool)
        return WormGraph(
            segments=(),
            segment_lengths=np.zeros(0, dtype=float),
            incidence_matrix=empty_incidence,
            incidence_directions=empty_incidence.copy(),
            incident_branch_areas=(),
            incident_segments=tuple(
                (np.zeros(0, dtype=int) for _ in range(branch_count))
            ),
        )
    sort_order = np.lexsort((trace.order, trace.labels))
    i = trace.rows[sort_order]
    j = trace.columns[sort_order]
    labels = trace.labels[sort_order]
    order = trace.order[sort_order]
    segment_count = trace.segment_count
    counts = np.bincount(labels)[1:]
    indexes = np.cumsum(counts) - counts
    coords = np.column_stack((i, j))
    graph_segments = tuple(
        (
            (
                coords[indexes[index] : indexes[index] + counts[index]],
                coords[indexes[index] : indexes[index] + counts[index]][::-1],
            )
            for index in range(len(counts))
        )
    )
    start_labels = np.zeros(segments.shape, dtype=int)
    starts = order == 0
    start_labels[i[starts], j[starts]] = labels[starts]
    ends = np.cumsum(counts) - 1
    end_labels = np.zeros(segments.shape, dtype=int)
    end_labels[i[ends], j[ends]] = labels[ends]
    incidence_directions = _incidence_matrix(
        branch_labels, branch_count, start_labels, segment_count
    )
    incidence_matrix = _incidence_matrix(
        branch_labels, branch_count, end_labels, segment_count
    )
    incidence_matrix |= incidence_directions
    segment_lengths = np.array(
        [calculate_cumulative_lengths(segment[0])[-1] for segment in graph_segments],
        dtype=float,
    )
    incident_segments = tuple(
        (
            np.flatnonzero(incidence_matrix[branch_index, :])
            for branch_index in range(branch_count)
        )
    )
    incident_branch_areas = tuple(
        (
            np.flatnonzero(incidence_matrix[:, segment_index])
            for segment_index in range(segment_count)
        )
    )
    return WormGraph(
        segments=graph_segments,
        segment_lengths=segment_lengths,
        incidence_matrix=incidence_matrix,
        incidence_directions=incidence_directions,
        incident_branch_areas=incident_branch_areas,
        incident_segments=incident_segments,
    )


def _cellprofiler_strel_disk(radius: float) -> np.ndarray:
    """Return CellProfiler/centrosome's disk footprint semantics."""
    integer_radius = int(radius)
    rows, columns = np.mgrid[
        -integer_radius : integer_radius + 1, -integer_radius : integer_radius + 1
    ]
    return rows * rows + columns * columns <= radius * radius


def _offset_path_coords(
    coords: np.ndarray, *, row_offset: int, column_offset: int
) -> np.ndarray:
    if len(coords) == 0:
        return coords
    offset = np.array((row_offset, column_offset), dtype=coords.dtype)
    return coords + offset


def _longest_worm_graph_path_coords(
    binary_image: np.ndarray, skeleton: np.ndarray, *, max_length: float
) -> np.ndarray:
    graph = WormGraphFromBinaryRequest(
        binary_image=binary_image,
        skeleton=skeleton,
        max_radius=None,
        max_skel_length=None,
    ).build()
    longest_coords = np.zeros((0, 2), dtype=int)
    longest_length = 0.0
    for path in graph.paths_between_lengths(min_length=0.0, max_length=max_length):
        coords = path.to_pixel_coords(graph)
        path_length = float(calculate_cumulative_lengths(coords)[-1])
        if path_length >= longest_length:
            longest_coords = coords
            longest_length = path_length
    return longest_coords


def _incidence_matrix(
    branch_labels: np.ndarray,
    branch_count: int,
    endpoint_labels: np.ndarray,
    segment_count: int,
) -> np.ndarray:
    incidence = np.zeros((branch_count, segment_count), dtype=bool)
    if branch_count == 0 or segment_count == 0:
        return incidence
    rows, columns = np.nonzero(branch_labels)
    height, width = branch_labels.shape
    for row, column in zip(rows, columns, strict=True):
        branch_id = int(branch_labels[row, column])
        for row_delta in (-1, 0, 1):
            neighbor_row = row + row_delta
            if neighbor_row < 0 or neighbor_row >= height:
                continue
            for column_delta in (-1, 0, 1):
                neighbor_column = column + column_delta
                if neighbor_column < 0 or neighbor_column >= width:
                    continue
                segment_id = int(endpoint_labels[neighbor_row, neighbor_column])
                if segment_id > 0:
                    incidence[branch_id - 1, segment_id - 1] = True
    return incidence


@dataclass(frozen=True, slots=True)
class WormClusterPathSelectionPolicy:
    """Shape and coverage policy for selecting candidate paths in a worm cluster."""

    median_worm_area: float | None
    component_area: int
    num_control_points: int
    mean_angles: np.ndarray
    inv_angles_covariance_matrix: np.ndarray
    cost_threshold: float
    overlap_weight: float
    leftover_weight: float
    min_path_length: float
    max_path_length: float

    def select(self, graph: WormGraph, paths: list[WormGraphPath]) -> list[np.ndarray]:
        paths_costs_and_coords: list[tuple[WormGraphPath, float, np.ndarray]] = []
        for path in paths:
            coords = path.to_pixel_coords(graph)
            cumulative_lengths = calculate_cumulative_lengths(coords)
            total_length = float(cumulative_lengths[-1])
            if (
                total_length > self.max_path_length
                or total_length < self.min_path_length
            ):
                continue
            cost = WormShapeCostRequest(
                geometry=WormControlPointGeometry.from_path_coords(
                    coords,
                    cumulative_lengths,
                    num_control_points=self.num_control_points,
                ),
                mean_angles=self.mean_angles,
                inv_angles_covariance_matrix=self.inv_angles_covariance_matrix,
            ).cost
            if cost < self.cost_threshold:
                paths_costs_and_coords.append((path, cost, coords))
        if not paths_costs_and_coords:
            return []
        costs = np.asarray(
            [cost for _path, cost, _coords in paths_costs_and_coords],
            dtype=float,
        )
        order = np.lexsort([costs])
        if len(order) > MAX_CLUSTER_PATHS:
            order = order[:MAX_CLUSTER_PATHS]
        costs = costs[order]
        path_segment_matrix = np.zeros((len(graph.segments), len(order)), dtype=bool)
        for column, ordered_index in enumerate(order):
            path = paths_costs_and_coords[int(ordered_index)][0]
            path_segment_matrix[list(path.segments), column] = True
        selected_indexes = WormPathSubsetSelectionContext(
            costs=costs,
            path_segment_matrix=path_segment_matrix,
            segment_lengths=graph.segment_lengths,
            overlap_weight=self.overlap_weight,
            leftover_weight=self.leftover_weight,
            max_worms=_cluster_max_worms(
                self.component_area, median_worm_area=self.median_worm_area
            ),
        ).select()
        return [
            paths_costs_and_coords[int(order[selected_index])][2]
            for selected_index in selected_indexes
        ]


def _cluster_max_worms(component_area: int, *, median_worm_area: float | None) -> int:
    if median_worm_area is None or median_worm_area <= 0:
        return 1
    return max(1, int(np.ceil(component_area / median_worm_area)))


@dataclass(frozen=True, slots=True)
class WormPathSelectionState:
    """Mutable-search state returned immutably between path selection levels."""

    best_subset: list[int]
    best_cost: float
    path_segment_matrix: np.ndarray
    path_choices: np.ndarray


@dataclass(frozen=True, slots=True)
class WormPathSubsetSelectionContext:
    """CP path coverage objective for selecting non-overlapping worm paths."""

    costs: np.ndarray
    path_segment_matrix: np.ndarray
    segment_lengths: np.ndarray
    overlap_weight: float
    leftover_weight: float
    max_worms: int

    def select(self) -> list[int]:
        state = WormPathSelectionState(
            best_subset=[],
            best_cost=float(np.sum(self.segment_lengths) * self.leftover_weight),
            path_segment_matrix=self.path_segment_matrix.astype(int),
            path_choices=np.eye(len(self.costs), dtype=bool),
        )
        for _level in range(min(self.max_worms, len(self.costs))):
            state = self._select_one_level(state)
            if np.prod(state.path_choices.shape) == 0:
                break
        return state.best_subset

    def _select_one_level(
        self, state: WormPathSelectionState
    ) -> WormPathSelectionState:
        partial_costs = (
            np.sum(self.costs[:, np.newaxis] * state.path_choices, axis=0)
            + np.sum(
                np.maximum(state.path_segment_matrix - 1, 0)
                * self.segment_lengths[:, np.newaxis],
                axis=0,
            )
            * self.overlap_weight
        )
        total_costs = (
            partial_costs
            + np.sum(
                (state.path_segment_matrix == 0) * self.segment_lengths[:, np.newaxis],
                axis=0,
            )
            * self.leftover_weight
        )
        order = np.lexsort([total_costs])
        best_subset = state.best_subset
        best_cost = state.best_cost
        if len(order) and total_costs[order[0]] < best_cost:
            best_subset = np.flatnonzero(state.path_choices[:, order[0]]).tolist()
            best_cost = float(total_costs[order[0]])
        mask = partial_costs < best_cost
        if not np.any(mask):
            return self._empty_state(best_subset, best_cost)
        order = order[mask[order]]
        if len(order) * len(self.costs) > MAX_CLUSTER_PATH_SETS_CONSIDERED:
            order = order[: 1 + MAX_CLUSTER_PATH_SETS_CONSIDERED // len(self.costs)]
        path_segment_matrix = state.path_segment_matrix[:, order]
        path_choices = state.path_choices[:, order]
        i, j = np.mgrid[0 : len(self.costs), 0 : len(self.costs)]
        disallow = i >= j
        allowed = np.dot(disallow, path_choices) == 0
        if not np.any(allowed):
            return self._empty_state(best_subset, best_cost)
        i, j = np.argwhere(allowed).transpose()
        return WormPathSelectionState(
            best_subset=best_subset,
            best_cost=best_cost,
            path_segment_matrix=self.path_segment_matrix[:, i]
            + path_segment_matrix[:, j],
            path_choices=np.eye(len(self.costs), dtype=bool)[:, i] | path_choices[:, j],
        )

    def _empty_state(
        self, best_subset: list[int], best_cost: float
    ) -> WormPathSelectionState:
        return WormPathSelectionState(
            best_subset=best_subset,
            best_cost=best_cost,
            path_segment_matrix=np.zeros((len(self.costs), 0), dtype=int),
            path_choices=np.zeros((len(self.costs), 0), dtype=bool),
        )


def _worm_label_outputs(
    worm_geometries: Sequence[WormControlPointGeometry],
    *,
    source_image: object,
    image_shape: tuple[int, int],
    radii_from_training: np.ndarray,
) -> WormLabelCandidates:
    ijv_parts: list[np.ndarray] = []
    overlap_hits = np.zeros(image_shape, dtype=np.int16)
    overlapping = np.zeros(image_shape, dtype=np.int32)
    for object_number, geometry in enumerate(worm_geometries, start=1):
        rows, cols = _reconstructed_worm_pixels(
            geometry,
            image_shape=image_shape,
            radii_from_training=radii_from_training,
        )
        if len(rows) == 0:
            continue
        ijv_parts.append(
            np.column_stack(
                (
                    rows.astype(np.int32, copy=False),
                    cols.astype(np.int32, copy=False),
                    np.full(len(rows), object_number, dtype=np.int32),
                )
            )
        )
        overlap_hits[rows, cols] += 1
        overlapping[rows, cols] = object_number
    nonoverlapping = overlapping.copy()
    nonoverlapping[overlap_hits != 1] = 0
    ijv = (
        np.vstack(ijv_parts).astype(np.int32, copy=False)
        if ijv_parts
        else np.zeros((0, 3), dtype=np.int32)
    )
    sparse_overlapping = SourceImageObjectLabelBuildRequest(
        image=source_image, labels=SparseIJVLabelRows(ijv)
    ).payload(
        representation=ObjectLabelRepresentation.SPARSE_IJV,
    )
    nonoverlapping_payload = SourceImageObjectLabelBuildRequest(
        image=source_image,
        labels=nonoverlapping,
    ).payload()
    return WormLabelCandidates(
        overlapping=sparse_overlapping,
        nonoverlapping=nonoverlapping_payload,
    )


def _overlapping_worm_outline(
    source_image: RuntimeArrayData,
    labels: ObjectLabelValue,
    colormap_name: str,
) -> RuntimeArrayData:
    """Render per-worm inner outlines without erasing sparse overlaps."""

    from skimage.segmentation import find_boundaries

    rows = object_label_sparse_ijv_rows(labels)
    row_array = rows.as_array()
    image_shape = tuple(np.asarray(source_image).shape[-2:])
    max_label = int(np.max(row_array[:, rows.label_column])) if row_array.size else 0
    resolved_colormap = (
        "viridis" if colormap_name.strip().casefold() == "default" else colormap_name
    )
    colors = object_label_colormap(resolved_colormap, max_label)
    output = np.zeros((*image_shape, 3), dtype=np.float32)
    label_ids = row_array[:, rows.label_column].astype(int, copy=False)
    for label_id in np.unique(label_ids[label_ids > 0]):
        label_rows = row_array[label_ids == label_id]
        label_y = label_rows[:, rows.y_column].astype(int, copy=False)
        label_x = label_rows[:, rows.x_column].astype(int, copy=False)
        y_start = max(int(np.min(label_y)) - 1, 0)
        y_stop = min(int(np.max(label_y)) + 2, image_shape[0])
        x_start = max(int(np.min(label_x)) - 1, 0)
        x_stop = min(int(np.max(label_x)) + 2, image_shape[1])
        local_mask = np.zeros((y_stop - y_start, x_stop - x_start), dtype=bool)
        local_mask[label_y - y_start, label_x - x_start] = True
        local_outline = find_boundaries(local_mask, mode="inner")
        output[y_start:y_stop, x_start:x_stop][local_outline] = colors[label_id]
    return with_image_payload_data(
        source_image,
        output,
        metadata=image_payload_metadata(source_image).replace_fields(
            source_channel_axis=-1
        ),
    )


def _nonoverlapping_worm_outline(
    source_image: RuntimeArrayData,
    labels: ObjectLabelValue,
) -> RuntimeArrayData:
    """Render the binary inner outline of exclusive worm labels."""

    from skimage.segmentation import find_boundaries

    outline = find_boundaries(
        object_label_dense_array(labels, dtype=np.int32),
        mode="inner",
    )
    return with_image_payload_data(
        source_image,
        outline,
        metadata=image_payload_metadata(source_image).without_source_channel_axis(),
    )


def _reconstructed_worm_pixels(
    geometry: WormControlPointGeometry,
    *,
    image_shape: tuple[int, int],
    radii_from_training: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(geometry.control_coords) < 2:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int))
    return rebuild_worm_from_control_points_approx(
        geometry.control_coords,
        radii_from_training,
        image_shape,
    )


def _coerce_mean_angles(
    mean_angles: tuple[float, ...] | None, num_control_points: int
) -> np.ndarray:
    if mean_angles is None:
        return np.zeros(max(num_control_points - 1, 0), dtype=float)
    return np.asarray(mean_angles, dtype=float)


def _coerce_inverse_covariance(
    inv_angles_covariance_matrix: tuple[tuple[float, ...], ...] | None,
    num_control_points: int,
) -> np.ndarray:
    if inv_angles_covariance_matrix is None:
        return np.eye(max(num_control_points - 1, 0), dtype=float)
    return np.asarray(inv_angles_covariance_matrix, dtype=float)


def _coerce_worm_radii(
    radii_from_training: tuple[float, ...] | None, num_control_points: int
) -> np.ndarray:
    if radii_from_training is None:
        return np.ones(num_control_points, dtype=float)
    radii = np.asarray(radii_from_training, dtype=float)
    if len(radii) == num_control_points:
        return radii
    if len(radii) == 0:
        return np.ones(num_control_points, dtype=float)
    if len(radii) < num_control_points:
        return np.pad(radii, (0, num_control_points - len(radii)), mode="edge")
    return radii[:num_control_points]


@dataclass(frozen=True, slots=True)
class WormShapeCostRequest:
    """Mahalanobis-style CP worm shape cost for one candidate path."""

    geometry: WormControlPointGeometry
    mean_angles: np.ndarray
    inv_angles_covariance_matrix: np.ndarray

    @property
    def cost(self) -> float:
        num_control_points = len(self.geometry.control_coords)
        if len(self.mean_angles) != num_control_points - 1:
            return 0.0
        expected_shape = (num_control_points - 1, num_control_points - 1)
        if self.inv_angles_covariance_matrix.shape != expected_shape:
            return 0.0
        feature_vector = (
            np.hstack((self.geometry.angles, [self.geometry.path_length]))
            - self.mean_angles
        )
        return float(
            feature_vector @ self.inv_angles_covariance_matrix @ feature_vector
        )

    def passes(self, cost_threshold: float) -> bool:
        return self.cost < cost_threshold


def _worm_descriptor_rows(
    worm_geometries: Sequence[WormControlPointGeometry],
    *,
    num_control_points: int,
) -> MeasurementSparseColumnarRows:
    """Return CellProfiler-compatible per-object worm descriptor rows."""
    schema = WormControlPointMeasurementSchema(num_control_points=num_control_points)
    control_point_fields = tuple(
        schema.row_fields(np.zeros((num_control_points, 2), dtype=float))
    )
    return MeasurementSparseColumnarRows.from_rows(
        tuple(
            _worm_descriptor_row(
                geometry,
                object_number=object_number,
            )
            for object_number, geometry in enumerate(worm_geometries, start=1)
        ),
        fields=(
            FieldSpec(MeasurementRowAxisField.OBJECT_NUMBER.value, int),
            FieldSpec("worm_length", float),
            *(
                FieldSpec(f"worm_angle_{index}", float)
                for index in range(1, max(num_control_points - 2, 0) + 1)
            ),
            *(FieldSpec(field_name, float) for field_name in control_point_fields),
        ),
    )


def _worm_descriptor_row(
    geometry: WormControlPointGeometry,
    *,
    object_number: int,
) -> dict[str, float | int]:
    row: dict[str, float | int] = {
        "object_number": object_number,
        "worm_length": geometry.path_length,
    }
    for index, angle in enumerate(geometry.angles, start=1):
        row[f"worm_angle_{index}"] = float(angle)
    row.update(
        WormControlPointMeasurementSchema(
            num_control_points=len(geometry.control_coords)
        ).row_fields(geometry.control_coords)
    )
    return row


class IdentifyDeadWormsModule(
    FieldDerivedMeasurementFeatureModule,
    MeasurementArtifactOutputModule,
    ObjectArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "IdentifyDeadWorms"
    function_name = "identify_dead_worms"
    validated = True
    confidence = 1.0
    measurement_feature_family = "Worm"
    measurement_category_prefixes = (("worm",),)

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Exact IdentifyDeadWorms feature vocabulary beyond core object rows."""

        ANGLE = "Angle"

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project owned angle records to exact CellProfiler worm features."""

        object_name: str

        @classmethod
        def for_request(cls, module_type, request):
            object_outputs = request.callable_contract.artifact_outputs.of_artifact_type(
                ObjectLabelsArtifactType
            )
            if len(object_outputs) != 1:
                raise ValueError(
                    "IdentifyDeadWorms measurement projection requires exactly one "
                    f"object output, got {[spec.name for spec in object_outputs]!r}."
                )
            return cls(
                request.output_value,
                module_type=module_type,
                object_name=object_outputs[0].name,
            )

        def rows(self) -> MeasurementSparseColumnarRows:
            source_rows = self.source_rows()
            if not isinstance(source_rows, ConcatenatedColumnarRows):
                raise TypeError(
                    "IdentifyDeadWorms measurement projection requires "
                    "ConcatenatedColumnarRows, got "
                    f"{type(source_rows).__name__}."
                )
            angle_batches = tuple(
                batch
                for batch in source_rows.row_batches
                if isinstance(batch, DataclassMeasurementColumnarRows)
                and batch.row_type is DeadWormAngleMeasurement
            )
            if len(angle_batches) != 1:
                raise TypeError(
                    "IdentifyDeadWorms measurement projection requires exactly one "
                    f"{DeadWormAngleMeasurement.__name__} batch, got "
                    f"{angle_batches!r}."
                )
            projected_rows = self.module_type.measurement_feature_rows_from_records(
                tuple(angle_batches[0].rows)
            )
            object_name_field = MeasurementRowAxisField.OBJECT_NAME.value
            for row in projected_rows:
                row[object_name_field] = self.object_name
            return MeasurementSparseColumnarRows.from_rows(
                projected_rows,
                fields=(
                    FieldSpec(object_name_field, str),
                    *self.module_type.measurement_feature_row_fields(
                        DeadWormAngleMeasurement
                    ),
                ),
                object_row_identity=MeasurementObjectRowIdentity.LABEL_ID,
            )

    input_image_setting = SettingNameFamily("Select the input image")
    output_objects_setting = SettingNameFamily(
        "Name the dead worm objects to be identified"
    )
    setting_bindings = (
        SettingToKeywordBinding.input(input_image_setting, ImageArtifactType),
        SettingToKeywordBinding.output(
            output_objects_setting,
            ObjectLabelsArtifactType,
        ),
        SettingToKeywordBinding(
            "Worm width",
            "worm_width",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Worm length",
            "worm_length",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Number of angles",
            "angle_count",
            parse_cellprofiler_int,
        ),
        SettingToKeywordBinding(
            "Automatically calculate distance parameters?",
            "auto_distance",
            parse_cellprofiler_bool,
        ),
        SettingToKeywordBinding(
            "Spatial distance",
            "space_distance",
            parse_cellprofiler_float,
        ),
        SettingToKeywordBinding(
            "Angular distance",
            "angular_distance",
            parse_cellprofiler_float,
        ),
    )
