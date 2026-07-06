"""Colocalization backends for CellProfiler-compatible measurements."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar
from python_introspect import set_parameter_exclusions
from scipy.linalg import lstsq
from openhcs.constants.constants import VariableComponents
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    measurement_row_mapping,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_values import ColumnarRows
from openhcs.interop.cellprofiler.setting_names import SettingNameFamily, setting_names
from openhcs.interop.cellprofiler.settings_binder import (
    normalize_cellprofiler_setting_name,
)
from openhcs.interop.cellprofiler.module_declarations import (
    ProcessingContract,
    ArtifactContractModule,
    BinderSettingsSourceModule,
    BoundModuleSettings,
    CellProfilerModule,
    ComposedImageObjectMeasurementExecutionModule,
    ModuleSettingsSourceModule,
    ScopedMeasurementModule,
    StructuringElementSettingsModule,
    ObjectMeasurementRowsModule,
)
from openhcs.interop.cellprofiler.runtime.object_measurement_row_policies import (
    CellProfilerObjectMeasurementRowPolicy,
    DenseColumnarObjectMeasurementRowsMixin,
    ObjectMeasurementInvocation,
    SourcePairObjectMeasurementInvocation,
)
from openhcs.interop.cellprofiler.runtime.invocation import (
    CellProfilerSourceImagePair,
    CellProfilerSourcePairFeature,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    LabelsObjectInputPolicy,
)
from openhcs.interop.cellprofiler.runtime.dual_scope_measurement_policies import (
    DeclaredDualScopeMeasurementPolicy,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    CurrentSourceMeasurementRecordMixin,
    NoObjectNameMeasurementRecordMixin,
    TableMeasurementRecordRowsMixin,
)
from openhcs.interop.cellprofiler.setting_names import (
    optional_setting_value,
    required_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.cellprofiler_literals import (
    cellprofiler_enum_from_literal,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    ThresholdSettingsModule,
)


class CellProfilerMeasurementRowIdentityField(str, Enum):
    """CellProfiler row fields that carry ownership/axis identity, not values."""

    SLICE_INDEX = MeasurementRowAxisField.SLICE_INDEX.value
    OBJECT_LABEL = MeasurementRowAxisField.OBJECT_LABEL.value
    OBJECT_NAME = MeasurementRowAxisField.OBJECT_NAME.value
    SOURCE_IMAGE_NAME = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value


@dataclass(frozen=True, slots=True)
class SourceImagePairCollection:
    """Cardinality authority for source-image pairs."""

    source_pairs: tuple[CellProfilerSourceImagePair, ...]

    def single_source_image_name(self) -> str | None:
        match self.source_pairs:
            case [source_pair]:
                return source_pair.source_image_name
            case _:
                return None


class MeasureColocalizationObjectMeasurementRowPolicy(
    DenseColumnarObjectMeasurementRowsMixin, CellProfilerObjectMeasurementRowPolicy
):
    """Expand composed source stacks into source-pair object measurements."""

    identity_fields: ClassVar[frozenset[str]] = frozenset(
        (field.value for field in CellProfilerMeasurementRowIdentityField)
    )

    def invocations(
        self,
        measurement_image: "CellProfilerMeasurementImage",
        kwargs: "CellProfilerKwargs",
    ) -> tuple[ObjectMeasurementInvocation, ...]:
        source_pairs = measurement_image.source_image_pairs()
        if not source_pairs:
            return super().invocations(measurement_image, kwargs)
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
        self, rows: "MeasurementRowsInput", invocation: ObjectMeasurementInvocation
    ) -> "MeasurementRowsInput":
        if invocation.source_pair is None:
            return rows if isinstance(rows, ColumnarRows) else list(rows)
        if isinstance(rows, ColumnarRows):
            return CellProfilerSourcePairFeature.project_columnar_rows_for_pair(
                rows,
                invocation.source_pair,
                retain_field=type(self).identity_fields.__contains__,
            )
        return [self.project_row(row, invocation.source_pair) for row in rows]

    def project_row(
        self, row: "CellProfilerRuntimeValue", source_pair: CellProfilerSourceImagePair
    ) -> "CellProfilerKwargDict":
        """Return one row with CellProfiler source-pair feature names."""
        row_mapping = measurement_row_mapping(row)
        identity_fields = type(self).identity_fields
        return CellProfilerSourcePairFeature.project_row_for_pair(
            row_mapping, source_pair, retain_field=identity_fields.__contains__
        )

    def table_source_image_name(
        self,
        measurement_images: tuple["CellProfilerMeasurementImage", ...],
        source_image_name: str | None,
    ) -> str | None:
        del source_image_name
        source_pairs = tuple(
            (
                source_pair
                for measurement_image in measurement_images
                for source_pair in measurement_image.source_image_pairs()
            )
        )
        return SourceImagePairCollection(source_pairs).single_source_image_name()


class MeasureColocalizationModule(
    LabelsObjectInputPolicy,
    TableMeasurementRecordRowsMixin,
    NoObjectNameMeasurementRecordMixin,
    CurrentSourceMeasurementRecordMixin,
    ComposedImageObjectMeasurementExecutionModule,
    ObjectMeasurementRowsModule,
    MeasureColocalizationObjectMeasurementRowPolicy,
    DeclaredDualScopeMeasurementPolicy,
    ScopedMeasurementModule,
):
    module_name = "MeasureColocalization"
    function_name = "measure_colocalization"
    validated = True
    aliases = ("MeasureCorrelation",)
    function_variants = ("measure_colocalization_objects",)
    contract = ProcessingContract.FLEXIBLE
    image_function_name = "measure_colocalization"
    default_variable_components = (VariableComponents.CHANNEL,)
    confidence = 1.0
    measurement_category_prefixes = (
        ("correlation",),
        ("colocalization",),
    )
    measurement_scope_setting = SettingNameFamily("Select where to measure correlation")
    object_measurement_setting = SettingNameFamily(
        "Select objects to measure", aliases=("Select an object to measure",)
    )
    object_gate_setting = SettingNameFamily("Select an object to measure")
    ignored_settings = ("Select objects to measure", "Hidden")

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by MeasureColocalization."""

        CORRELATION = "correlation"
        REGRESSION_SLOPE = "slope"
        OVERLAP = "overlap"
        COSTES_MANDERS = "costes"
        MANDERS = "manders"
        RANK_WEIGHTED_COLOCALIZATION = "rwc"
        OVERLAP_K = "k"

    directional_pair_feature_aliases = MappingProxyType(
        {
            "costes_m_1": (MeasurementFeature.COSTES_MANDERS.value, 1),
            "costes_m_2": (MeasurementFeature.COSTES_MANDERS.value, 2),
            "k_1": (MeasurementFeature.OVERLAP_K.value, 1),
            "k_2": (MeasurementFeature.OVERLAP_K.value, 2),
            "manders_m_1": (MeasurementFeature.MANDERS.value, 1),
            "manders_m_2": (MeasurementFeature.MANDERS.value, 2),
            "rwc_1": (MeasurementFeature.RANK_WEIGHTED_COLOCALIZATION.value, 1),
            "rwc_2": (MeasurementFeature.RANK_WEIGHTED_COLOCALIZATION.value, 2),
        }
    )
    scale_qualified_measurement_feature_prefixes = (
        (MeasurementFeature.CORRELATION.value,),
    )
    pair_correlation_feature_name = MeasurementFeature.CORRELATION.value
    pair_regression_slope_feature_name = MeasurementFeature.REGRESSION_SLOPE.value
    undirected_pair_feature_names = frozenset(
        {
            MeasurementFeature.CORRELATION.value,
            MeasurementFeature.OVERLAP.value,
        }
    )
    threshold_sensitive_pair_feature_names = frozenset(
        {
            MeasurementFeature.COSTES_MANDERS.value,
            MeasurementFeature.MANDERS.value,
            MeasurementFeature.RANK_WEIGHTED_COLOCALIZATION.value,
            MeasurementFeature.OVERLAP_K.value,
        }
    )
    measurement_value_fields = (
        "correlation",
        "slope",
        "overlap",
        "k1",
        "k2",
        "manders_m1",
        "manders_m2",
        "rwc1",
        "rwc2",
        "costes_m1",
        "costes_m2",
        "costes_threshold_1",
        "costes_threshold_2",
    )

    @classmethod
    def materialized_measurement_fields(cls, *, object_scope: bool = False) -> tuple[str, ...]:
        axis_fields = ("slice_index", "object_label") if object_scope else ("slice_index",)
        return (*axis_fields, *cls.measurement_value_fields)

    class MeasurementScope(str, Enum):
        image = "Across entire image"
        objects = "Objects"
        both = "Both"

        @classmethod
        def from_literal(
            cls, value: "MeasureColocalizationModule.MeasurementScope | str"
        ) -> "MeasureColocalizationModule.MeasurementScope":
            return cellprofiler_enum_from_literal(
                cls,
                value,
                aliases={
                    "across entire image": cls.image,
                    "object": cls.objects,
                    "objects": cls.objects,
                    "within objects": cls.objects,
                    "both": cls.both,
                },
            )

    measurement_scope_default = MeasurementScope.image

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
    def resolve_function(
        cls, module: "ModuleBlock", *, default_function_name: str | None = None
    ) -> "ResolvedModuleFunction":
        scope = cls.MeasurementScope.from_literal(
            cls.setting_value(module, cls.measurement_scope_setting)
            or cls.measurement_scope_default.value
        )
        object_values = setting_values(module, cls.object_measurement_setting)
        has_objects = any((split_symbol_names(value) for value in object_values))
        if (
            scope in (cls.MeasurementScope.objects, cls.MeasurementScope.both)
            and has_objects
        ):
            return super().resolve_function(
                module, default_function_name=cls.function_variants[0]
            )
        return super().resolve_function(
            module, default_function_name=default_function_name
        )

    @classmethod
    def measurement_target_scope(
        cls, module: "ModuleBlock"
    ) -> "MeasureColocalizationModule.MeasurementScope":
        return cls.MeasurementScope.from_literal(
            cls.setting_value(module, cls.measurement_scope_setting)
            or cls.measurement_scope_default.value
        )

    metric_settings: ClassVar[Mapping[str | SettingNameFamily, str]] = {
        "Set threshold as percentage of maximum intensity for the images": "threshold_percent",
        "Calculate correlation and slope metrics?": "do_correlation",
        "Calculate the Manders coefficients?": "do_manders",
        SettingNameFamily(
            "Calculate the Rank Weighted Colocalization coefficients?",
            aliases=("Calculate the Rank Weighted Coloalization coefficients?",),
        ): "do_rwc",
        "Calculate the Overlap coefficients?": "do_overlap",
        "Calculate the Manders coefficients using Costes auto threshold?": "do_costes",
    }
    metric_flags: ClassVar[tuple[str, ...]] = (
        "do_correlation",
        "do_manders",
        "do_rwc",
        "do_overlap",
        "do_costes",
    )
    run_all_metrics_setting = "Run all metrics?"
    costes_method_setting = "Method for Costes thresholding"

    class CostesMethod(str, Enum):
        faster = "faster"
        fast = "fast"
        accurate = "accurate"

        @classmethod
        def from_literal(
            cls, value: "MeasureColocalizationModule.CostesMethod | str"
        ) -> "MeasureColocalizationModule.CostesMethod":
            return cellprofiler_enum_from_literal(cls, value)

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
            module, binder=binder, param_mapping=param_mapping
        )
        kwargs = dict(bound.kwargs)
        unmapped_kwargs = dict(bound.unmapped_kwargs)
        for setting_name in setting_names(cls.measurement_scope_setting):
            unmapped_kwargs.pop(normalize_cellprofiler_setting_name(setting_name), None)
        for setting_name, parameter_name in cls.metric_settings.items():
            value = optional_setting_value(module, setting_name)
            if value is not None:
                kwargs[parameter_name] = binder.parse_value(setting_name, value)
            for concrete_name in setting_names(setting_name):
                unmapped_kwargs.pop(
                    normalize_cellprofiler_setting_name(concrete_name), None
                )
        run_all_value = optional_setting_value(module, cls.run_all_metrics_setting)
        if run_all_value is not None:
            if cls.run_all_metrics_enabled(run_all_value, binder):
                kwargs.update({flag: True for flag in cls.metric_flags})
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.run_all_metrics_setting), None
            )
        costes_method = optional_setting_value(module, cls.costes_method_setting)
        if costes_method is not None:
            kwargs["costes_method"] = cls.CostesMethod.from_literal(costes_method).value
            unmapped_kwargs.pop(
                normalize_cellprofiler_setting_name(cls.costes_method_setting), None
            )
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=cls.postprocess_bound_settings(
                module, BoundModuleSettings(kwargs, unmapped_kwargs)
            ),
            ignored_unmapped_settings=ignored_unmapped_settings,
        )

    @staticmethod
    def run_all_metrics_enabled(value: str, binder: "SettingsBinder") -> bool:
        normalized = value.strip().lower()
        if normalized in binder.BOOL_TRUE:
            return True
        if normalized in binder.BOOL_FALSE:
            return False
        return bool(value.strip())

    @classmethod
    def artifact_contract(cls, assembler, builder, module):
        return cls.measurement_artifact_contract_from_declared_settings(
            assembler, builder, module
        )


import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import asdict, dataclass, field, fields, make_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import ClassVar, Tuple
import numpy as np
from metaclass_registry import AutoRegisterMeta
from numba import njit
from openhcs.constants.constants import MemoryType
from openhcs.core.aligned_image_payload import (
    AlignedImageStack,
    ImagePayloadExecutionMode,
    aligned_image_stack_kwargs,
)
from openhcs.core.memory import detect_memory_type, numpy
from openhcs.core.pipeline.function_contracts import (
    composed_image_payload,
    measurement_image_batch_executor,
    special_inputs,
    special_outputs,
)
from openhcs.core.public_api import public_names_from_objects
from openhcs.core.runtime_invocation import RuntimeBatchInvocationRequest
from openhcs.core.runtime_semantics import (
    DenseObjectLabelStack,
    RuntimePlaneAxis,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValues
from openhcs.core.runtime_values import (
    ColumnarRows,
    DenseObjectLabelAggregation,
    ImagePayloadChannelProjection,
    ObjectLabelDenseDataStrategy,
    ObjectLabelValue,
    image_intensity_scale_for_dtype,
    image_payload_data,
    image_payload_mask,
    image_payload_metadata,
    object_label_dense_array,
)
from openhcs.processing.backends.cellprofiler._backend import (
    BackendProviderInput,
    DEFAULT_CELLPROFILER_BACKEND_SELECTION,
    CellProfilerBackendProvider,
    CellProfilerBackendStrategyMixin,
    CellProfilerBackendAuthority,
)
from openhcs.processing.backends.cellprofiler.granularity import (
    CellProfilerRuntimeProfiler,
)
from openhcs.processing.backends.cellprofiler.colocalization_costes import (
    UnitIntervalDenseRankSemantics,
    _correlation_slopes_numba,
    _costes_manders_numba,
    _integer_unit_interval_codes_for_scale_numba,
    _linear_costes_numba,
    _linear_costes_sorted_events_numba,
    _regression_line_numba,
    _scaled_second_channel_costes_grouped_events_numba,
    _scaled_second_channel_costes_numba,
    _scaled_second_channel_costes_sorted_events_numba,
    _thresholded_colocalization_metrics_with_ranks_numba,
    costes_above_threshold_mask,
    object_colocalization_base_reductions,
    object_colocalization_correlation_reductions,
    object_colocalization_rwc_reductions,
    object_colocalization_threshold_reductions,
    quantized_unit_interval_event_summaries,
    thresholded_colocalization_metrics,
)
from openhcs.interop.cellprofiler.runtime.image_payload_collapse import (
    SINGLETON_STACK_OUTPUT_COLLAPSE,
)
from openhcs.interop.cellprofiler.runtime.measurement_execution_support import (
    PreparedObjectMeasurementInvocation,
)
from openhcs.interop.cellprofiler.runtime.pure2d_output_aggregation import (
    CellProfilerPure2DOutputAggregator,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    Pure2DSliceResultBatch,
)
from openhcs.processing.materialization import csv_materializer

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
        valid, slope, intercept = _regression_line_numba(first, second)
        if not valid:
            return (0.0, 0.0)
        if slope > 0.0:
            event_summaries = quantized_unit_interval_event_summaries(
                first, second, slope, intercept, int(scale_max)
            )
            if event_summaries is None:
                event_threshold = np.minimum(second, slope * first + intercept)
                unique_events, inverse = np.unique(event_threshold, return_inverse=True)
                counts = np.bincount(inverse)
                first_sum = np.bincount(inverse, weights=first)
                second_sum = np.bincount(inverse, weights=second)
                first_square_sum = np.bincount(inverse, weights=first * first)
                second_square_sum = np.bincount(inverse, weights=second * second)
                product_sum = np.bincount(inverse, weights=first * second)
            else:
                (
                    unique_events,
                    counts,
                    first_sum,
                    second_sum,
                    first_square_sum,
                    second_square_sum,
                    product_sum,
                ) = event_summaries
            return _scaled_second_channel_costes_grouped_events_numba(
                (
                    np.ascontiguousarray(unique_events, dtype=np.float64),
                    np.ascontiguousarray(np.cumsum(first_sum), dtype=np.float64),
                    np.ascontiguousarray(np.cumsum(second_sum), dtype=np.float64),
                    np.ascontiguousarray(np.cumsum(first_square_sum), dtype=np.float64),
                    np.ascontiguousarray(
                        np.cumsum(second_square_sum), dtype=np.float64
                    ),
                    np.ascontiguousarray(np.cumsum(product_sum), dtype=np.float64),
                    int(scale_max),
                    slope,
                    intercept,
                ),
                np.ascontiguousarray(np.cumsum(counts), dtype=np.int64),
            )
        return _scaled_second_channel_costes_numba(first, second, int(scale_max))

    def correlation_slopes(
        self, first_pixels: np.ndarray, second_pixels: np.ndarray
    ) -> tuple[float, float, float]:
        correlation, _, reverse_slope = _correlation_slopes_numba(
            np.ascontiguousarray(first_pixels, dtype=np.float64),
            np.ascontiguousarray(second_pixels, dtype=np.float64),
        )
        slope = lstsq(
            np.array((first_pixels, np.ones_like(first_pixels))).transpose(),
            second_pixels,
        )[0][0]
        return float(correlation), float(slope), float(reverse_slope)

    def prepare_backend(self) -> None:
        """Compile numba Costes kernels outside measured execution."""
        first = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32)
        second = np.flipud(first.reshape((64, 64))).ravel().copy()
        self.correlation_slopes(first, second)
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


class ColocalizationMeasurementSchema:
    """Authoritative row schema for image- and object-scoped colocalization."""

    object_label_field = ("object_label", int)

    @classmethod
    def object_measurement_type(
        cls, measurement_type: type[ColocalizationMeasurements]
    ) -> type:
        measurement_fields = tuple(fields(measurement_type))
        row_fields = (
            (measurement_fields[0].name, measurement_fields[0].type),
            cls.object_label_field,
            *((field.name, field.type) for field in measurement_fields[1:]),
        )
        return make_dataclass(
            "ObjectColocalizationMeasurements",
            row_fields,
            slots=True,
            namespace={
                "__module__": __name__,
                "__doc__": "Colocalization measurements scoped to one labeled object.",
                "from_measurement": classmethod(cls.from_measurement),
                "from_values": classmethod(cls.from_values),
            },
        )

    @staticmethod
    def finite_or_zero(value: float) -> float:
        return float(value) if np.isfinite(value) else 0.0

    @staticmethod
    def from_measurement(
        row_type: type, *, object_label: int, measurement: ColocalizationMeasurements
    ) -> object:
        measurement_values = asdict(measurement)
        return row_type(object_label=object_label, **measurement_values)

    @classmethod
    def from_values(
        cls,
        row_type: type,
        object_label: int,
        *,
        slice_index: int = 0,
        correlation: float = 0.0,
        slope: float = 0.0,
        overlap: float = 0.0,
        k1: float = 0.0,
        k2: float = 0.0,
        manders_m1: float = 0.0,
        manders_m2: float = 0.0,
        rwc1: float = 0.0,
        rwc2: float = 0.0,
        costes_m1: float = 0.0,
        costes_m2: float = 0.0,
        costes_threshold_1: float = 0.0,
        costes_threshold_2: float = 0.0,
    ) -> object:
        """Build one object-row record using CellProfiler finite-value semantics."""
        return row_type(
            slice_index=int(slice_index),
            object_label=object_label,
            correlation=cls.finite_or_zero(correlation),
            slope=cls.finite_or_zero(slope),
            overlap=cls.finite_or_zero(overlap),
            k1=cls.finite_or_zero(k1),
            k2=cls.finite_or_zero(k2),
            manders_m1=cls.finite_or_zero(manders_m1),
            manders_m2=cls.finite_or_zero(manders_m2),
            rwc1=cls.finite_or_zero(rwc1),
            rwc2=cls.finite_or_zero(rwc2),
            costes_m1=float(costes_m1),
            costes_m2=float(costes_m2),
            costes_threshold_1=cls.finite_or_zero(costes_threshold_1),
            costes_threshold_2=cls.finite_or_zero(costes_threshold_2),
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
    first_denominator: float
    second_denominator: float

    @classmethod
    def from_thresholds(
        cls, first: float, second: float, *, scale_max: int
    ) -> "ColocalizationCostesThresholds":
        first_denominator = float(first)
        second_denominator = float(second)
        first_threshold = float(
            np.nextafter(np.float32(first_denominator), np.float32(np.inf))
        )
        second_threshold = float(
            np.nextafter(np.float32(second_denominator), np.float32(np.inf))
        )
        scaled_first = first_denominator * scale_max
        nearest_first_bin = round(scaled_first)
        first_denominator = (
            nearest_first_bin / scale_max
            if scale_max > 0
            and np.isfinite(first)
            and np.isclose(scaled_first, nearest_first_bin, rtol=0.0, atol=0.001)
            else first_denominator
        )
        return cls(
            first=first_threshold,
            second=second_threshold,
            first_denominator=float(first_denominator),
            second_denominator=second_denominator,
        )


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
class ColocalizationMaskRequest:
    """Resolved mask-shape matching request for an image-pair measurement."""

    valid: np.ndarray
    image_data: np.ndarray
    mask_array: np.ndarray
    channel_1: int
    channel_2: int


class ColocalizationMaskStrategy(ABC, metaclass=AutoRegisterMeta):
    """Nominal matcher for one supported MeasureColocalization mask layout."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True
    registry_key: ClassVar[str | None] = None

    @classmethod
    def strategies(cls) -> tuple["ColocalizationMaskStrategy", ...]:
        return tuple((strategy_type() for strategy_type in cls.__registry__.values()))

    @abstractmethod
    def matches(self, request: ColocalizationMaskRequest) -> bool:
        """Return whether this strategy owns the request's mask layout."""

    @abstractmethod
    def apply(self, request: ColocalizationMaskRequest) -> np.ndarray:
        """Return valid pixels constrained by this mask layout."""


class SpatialColocalizationMaskStrategy(ColocalizationMaskStrategy):
    """Mask already matches the shared 2D spatial domain."""

    registry_key = "spatial"

    def matches(self, request: ColocalizationMaskRequest) -> bool:
        return request.mask_array.shape == request.valid.shape

    def apply(self, request: ColocalizationMaskRequest) -> np.ndarray:
        return request.valid & request.mask_array


class ImageStackColocalizationMaskStrategy(ColocalizationMaskStrategy):
    """Mask matches the whole channel-first image stack."""

    registry_key = "image_stack"

    def matches(self, request: ColocalizationMaskRequest) -> bool:
        return request.mask_array.shape == request.image_data.shape

    def apply(self, request: ColocalizationMaskRequest) -> np.ndarray:
        return (
            request.valid
            & request.mask_array[request.channel_1]
            & request.mask_array[request.channel_2]
        )


class ChannelLeadingColocalizationMaskStrategy(ColocalizationMaskStrategy):
    """Mask has channel-leading planes with the same shared spatial domain."""

    registry_key = "channel_leading"

    def matches(self, request: ColocalizationMaskRequest) -> bool:
        return (
            request.mask_array.ndim >= 3
            and request.mask_array.shape[0] == request.image_data.shape[0]
            and (request.mask_array.shape[1:] == request.valid.shape)
        )

    def apply(self, request: ColocalizationMaskRequest) -> np.ndarray:
        return (
            request.valid
            & request.mask_array[request.channel_1]
            & request.mask_array[request.channel_2]
        )


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
        mask_array = np.asarray(mask, dtype=bool)
        request = ColocalizationMaskRequest(
            valid=valid,
            image_data=image_data,
            mask_array=mask_array,
            channel_1=channel_1,
            channel_2=channel_2,
        )
        for strategy in ColocalizationMaskStrategy.strategies():
            if strategy.matches(request):
                resolved_valid = strategy.apply(request)
                if bool(np.all(resolved_valid)):
                    return None
                return resolved_valid
        raise ValueError(
            f"MeasureColocalization image mask must match the shared spatial domain or channel stack; got mask {mask_array.shape!r} for image {image_data.shape!r}."
        )

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
    def requires_slice_local_context(image: object) -> bool:
        """Return whether context resolution must happen after slice projection."""
        return isinstance(image_payload_data(image), AlignedImageStack)

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
        labels: object,
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
        label_array = cls._labels_aligned_to_mask(
            label_array, pair_valid_mask, measurement_shape=measurement_shape
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

    @staticmethod
    def _labels_aligned_to_mask(
        label_array: np.ndarray,
        pair_valid_mask: np.ndarray | None,
        *,
        measurement_shape: tuple[int, ...] | None,
    ) -> np.ndarray:
        """Return labels in the same runtime-stack domain as the image mask."""
        if (
            measurement_shape is not None
            and label_array.ndim == len(measurement_shape) + 1
            and (tuple(label_array.shape[1:]) == tuple(measurement_shape))
        ):
            try:
                return DenseObjectLabelStack.from_labels(
                    label_array
                ).project_xy_plane_without_relabeling()
            except ValueError:
                return np.max(label_array, axis=0).astype(np.int32, copy=False)
        if pair_valid_mask is None:
            return label_array
        if tuple(label_array.shape) == tuple(pair_valid_mask.shape):
            return label_array
        if pair_valid_mask.ndim == label_array.ndim + 1 and tuple(
            pair_valid_mask.shape[1:]
        ) == tuple(label_array.shape):
            return np.broadcast_to(label_array, pair_valid_mask.shape)
        return label_array


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


class ObjectColocalizationRankProvider(ABC):
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


@dataclass(slots=True)
class ObjectColocalizationMetricArrays:
    """Mutable metric arrays populated by object-colocalization stages."""

    corr: np.ndarray
    slope: np.ndarray
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

        def values() -> np.ndarray:
            return np.zeros(max_label, dtype=float)

        return cls(
            corr=values(),
            slope=values(),
            overlap=values(),
            k1=values(),
            k2=values(),
            manders_m1=values(),
            manders_m2=values(),
            rwc1=values(),
            rwc2=values(),
            costes_m1=values(),
            costes_m2=values(),
            costes_threshold_1=values(),
            costes_threshold_2=values(),
        )

    def rows_for(
        self, label_range: np.ndarray, *, slice_index: int = 0
    ) -> "ObjectColocalizationColumnarMeasurements":
        return ObjectColocalizationColumnarMeasurements(
            object_labels=np.asarray(label_range, dtype=np.int32),
            metrics=self,
            slice_index=int(slice_index),
        )

    @staticmethod
    def finite_or_zero_column(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return np.where(np.isfinite(array), array, 0.0)

    def columns_for(
        self, object_labels: np.ndarray, *, slice_index: int = 0
    ) -> Mapping[str, np.ndarray]:
        return MappingProxyType(
            {
                "slice_index": np.full(
                    len(object_labels), int(slice_index), dtype=np.int32
                ),
                "object_label": object_labels,
                "correlation": self.finite_or_zero_column(self.corr),
                "slope": self.finite_or_zero_column(self.slope),
                "overlap": self.finite_or_zero_column(self.overlap),
                "k1": self.finite_or_zero_column(self.k1),
                "k2": self.finite_or_zero_column(self.k2),
                "manders_m1": self.finite_or_zero_column(self.manders_m1),
                "manders_m2": self.finite_or_zero_column(self.manders_m2),
                "rwc1": self.finite_or_zero_column(self.rwc1),
                "rwc2": self.finite_or_zero_column(self.rwc2),
                "costes_m1": np.asarray(self.costes_m1, dtype=float),
                "costes_m2": np.asarray(self.costes_m2, dtype=float),
                "costes_threshold_1": self.finite_or_zero_column(
                    self.costes_threshold_1
                ),
                "costes_threshold_2": self.finite_or_zero_column(
                    self.costes_threshold_2
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class ObjectColocalizationColumnarMeasurements(ColumnarRows):
    """Columnar object-colocalization rows preserving direct row iteration."""

    object_labels: np.ndarray
    metrics: ObjectColocalizationMetricArrays
    slice_index: int = 0
    _columns: Mapping[str, np.ndarray] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_columns",
            self.metrics.columns_for(self.object_labels, slice_index=self.slice_index),
        )

    @property
    def columns(self) -> Mapping[str, np.ndarray]:
        return self._columns

    def __len__(self) -> int:
        return len(self.object_labels)

    def __iter__(self):
        for index, object_label in enumerate(self.object_labels):
            yield ObjectColocalizationMeasurements.from_values(
                int(object_label),
                slice_index=self.slice_index,
                correlation=self.metrics.corr[index],
                slope=self.metrics.slope[index],
                overlap=self.metrics.overlap[index],
                k1=self.metrics.k1[index],
                k2=self.metrics.k2[index],
                manders_m1=self.metrics.manders_m1[index],
                manders_m2=self.metrics.manders_m2[index],
                rwc1=self.metrics.rwc1[index],
                rwc2=self.metrics.rwc2[index],
                costes_m1=self.metrics.costes_m1[index],
                costes_m2=self.metrics.costes_m2[index],
                costes_threshold_1=self.metrics.costes_threshold_1[index],
                costes_threshold_2=self.metrics.costes_threshold_2[index],
            )

    @property
    def covers_declared_object_measurement_domain(self) -> bool:
        """Object-colocalization rows are emitted over the resolved label range."""
        return True


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
                costes_thresholds.first if costes_thresholds is not None else 0.0,
                costes_thresholds.second if costes_thresholds is not None else 0.0,
                (
                    costes_thresholds.first_denominator
                    if costes_thresholds is not None
                    else 0.0
                ),
                (
                    costes_thresholds.second_denominator
                    if costes_thresholds is not None
                    else 0.0
                ),
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
    labels: object,
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
            threshold_c1, threshold_c2, scale_max=options.scale_max
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
            threshold_c1, threshold_c2, scale_max=options.scale_max
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
    metrics.corr = object_colocalization_correlation_reductions(
        base.first_pixels,
        base.second_pixels,
        base.object_labels,
        base.object_counts,
        base.sum1,
        base.sum2,
        len(base.object_counts),
    )
    metrics.corr[~np.isfinite(metrics.corr)] = np.nan


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
            corr, slope, _ = (
                ColocalizationCostesBackendStrategy.for_memory_type(
                    backend_provider=options.costes_backend_provider
                ).correlation_slopes(fi, si)
            )
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


@composed_image_payload
@numpy(contract=ProcessingContract.FLEXIBLE)
@special_outputs(
    (
        "colocalization_measurements",
        csv_materializer(
            fields=MeasureColocalizationModule.materialized_measurement_fields(),
            analysis_type="colocalization",
        ),
    )
)
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
    costes_backend_provider: BackendProviderInput = DEFAULT_CELLPROFILER_BACKEND_SELECTION,
) -> Tuple[np.ndarray, ColocalizationMeasurements]:
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
        Tuple of (first channel image, ColocalizationMeasurements)

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
        'Method for Costes thresholding' -> costes_method
    """
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
    output = ImagePayloadChannelProjection.from_channel(
        image, image_data, channel_1
    ).payload()
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
    return (output, measurements)


def _measure_colocalization_objects_core(
    image: np.ndarray,
    labels: np.ndarray,
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
    rank_provider: ObjectColocalizationRankProvider = DirectObjectColocalizationRankProvider(),
) -> Tuple[np.ndarray, ObjectColocalizationColumnarMeasurements]:
    """Measure colocalization between two channels within labeled objects."""
    total_started_at = time.perf_counter()
    phase_started_at = time.perf_counter()
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
    runtime_profiler.log(
        "coloc_object_prepare_context",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
        objects=context.labels.max_label,
        object_pixels=int(context.labels.object_labels.size),
    )
    if not context.has_labels:
        return (
            ImagePayloadChannelProjection.from_channel(
                context.image, context.image_data, context.channel_1
            ).payload(),
            [],
        )
    if not context.has_object_pixels:
        return (
            ImagePayloadChannelProjection.from_channel(
                context.image, context.image_data, context.channel_1
            ).payload(),
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
        channel_1=channel_1,
        channel_2=channel_2,
        pixels=int(base.object_labels.size),
    )
    phase_started_at = time.perf_counter()
    metrics = ObjectColocalizationMetricArrays.empty(context.labels.max_label)
    _populate_object_correlation_metrics(context.options, base, metrics)
    runtime_profiler.log(
        "coloc_object_correlation_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
    )
    phase_started_at = time.perf_counter()
    resolved_costes_thresholds = _resolve_object_costes_thresholds(
        context, base, costes_thresholds, metrics
    )
    runtime_profiler.log(
        "coloc_object_costes_thresholds",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
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
        channel_1=channel_1,
        channel_2=channel_2,
        thresholds=threshold.combined_threshold_has_values,
    )
    phase_started_at = time.perf_counter()
    _populate_object_threshold_metrics(context, base, threshold, metrics, rank_provider)
    runtime_profiler.log(
        "coloc_object_threshold_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
    )
    phase_started_at = time.perf_counter()
    _populate_object_costes_metrics(context.options, base, threshold, metrics)
    runtime_profiler.log(
        "coloc_object_costes_metrics",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
    )
    phase_started_at = time.perf_counter()
    rows = metrics.rows_for(
        context.labels.label_range, slice_index=context.labels.slice_index
    )
    runtime_profiler.log(
        "coloc_object_rows",
        time.perf_counter() - phase_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
        rows=len(rows),
    )
    runtime_profiler.log(
        "coloc_object_total",
        time.perf_counter() - total_started_at,
        function="measure_colocalization_objects",
        channel_1=channel_1,
        channel_2=channel_2,
    )
    return (
        ImagePayloadChannelProjection.from_channel(
            context.image, context.image_data, context.channel_1
        ).payload(),
        rows,
    )


@numpy(contract=ProcessingContract.FLEXIBLE)
@special_inputs("labels")
@special_outputs(
    (
        "object_colocalization_measurements",
        csv_materializer(
            fields=MeasureColocalizationModule.materialized_measurement_fields(
                object_scope=True
            ),
            analysis_type="object_colocalization",
        ),
    )
)
def measure_colocalization_objects(
    image: np.ndarray,
    labels: np.ndarray,
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
    rank_provider: ObjectColocalizationRankProvider = DirectObjectColocalizationRankProvider(),
) -> Tuple[np.ndarray, ObjectColocalizationColumnarMeasurements]:
    """Measure colocalization between two channels within labeled objects."""
    return _measure_colocalization_objects_core(
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
        costes_thresholds=costes_thresholds,
        image_pair_context=image_pair_context,
        object_label_context=object_label_context,
        rank_provider=rank_provider,
    )


set_parameter_exclusions(measure_colocalization_objects, ("rank_provider",))


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
            return ColocalizationCostesThresholds.from_thresholds(
                0.0, 0.0, scale_max=self.scale_max
            )
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
        return ColocalizationCostesThresholds.from_thresholds(
            first, second, scale_max=self.scale_max
        )


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
        dense_label_data = ObjectLabelDenseDataStrategy.for_payload(label_payload).data(
            label_payload
        )
        dense_label_array = np.asarray(dense_label_data)
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
            label_array = np.asarray(dense_label_data, dtype=np.int32)
            context = ColocalizationObjectLabelContext.from_dense_labels(
                label_array,
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
        if ColocalizationImagePairContext.requires_slice_local_context(request.image):
            return self.slice_aligned_request_kwargs(request)
        image_pair_context = self.image_pair_context(request)
        object_label_context = self.object_label_context(request, image_pair_context)
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

    def slice_aligned_request_kwargs(
        self, request: RuntimeBatchInvocationRequest
    ) -> dict[str, object]:
        """Return kwargs carrying one cached context per aligned image slice."""
        image_data = image_payload_data(request.image)
        if not isinstance(image_data, AlignedImageStack):
            return dict(request.kwargs)
        image_pair_contexts: list[ColocalizationImagePairContext] = []
        object_label_contexts: list[ColocalizationObjectLabelContext] = []
        costes_thresholds: list[ColocalizationCostesThresholds | None] = []
        has_thresholds = False
        for slice_index, slice_payload in enumerate(image_data.slices):
            slice_kwargs = {
                **request.kwargs,
                **aligned_image_stack_kwargs(
                    {"labels": request.kwargs["labels"]},
                    slice_index,
                    len(image_data.slices),
                    reference_payload=slice_payload,
                ),
            }
            slice_request = replace(request, image=slice_payload, kwargs=slice_kwargs)
            image_pair_context = self.image_pair_context(slice_request)
            object_label_context = self.object_label_context(
                slice_request, image_pair_context, slice_index=slice_index
            )
            threshold_request = ColocalizationCostesThresholdRequest.from_batch_request(
                slice_request, image_pair_context
            )
            thresholds = None
            if threshold_request is not None:
                key = threshold_request.cache_key
                thresholds = self._thresholds.get(key)
                if thresholds is None:
                    thresholds = threshold_request.thresholds()
                    self._thresholds[key] = thresholds
                has_thresholds = True
            image_pair_contexts.append(image_pair_context)
            object_label_contexts.append(object_label_context)
            costes_thresholds.append(thresholds)
        kwargs: dict[str, object] = {
            **request.kwargs,
            "image_pair_context": RuntimeSliceAlignedValues(tuple(image_pair_contexts)),
            "object_label_context": RuntimeSliceAlignedValues(
                tuple(object_label_contexts)
            ),
        }
        if has_thresholds:
            kwargs["costes_thresholds"] = RuntimeSliceAlignedValues(
                tuple(costes_thresholds)
            )
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
        direct_output = _direct_colocalization_batch_request(
            request, prepared_kwargs, rank_provider
        )
        if profile_enabled:
            runtime_profiler.log(
                "coloc_object_batch_direct",
                time.perf_counter() - phase_started_at,
                function="measure_colocalization_objects",
            )
        if direct_output is not None:
            outputs.append(direct_output)
            if profile_enabled:
                runtime_profiler.log(
                    "coloc_object_batch_request",
                    time.perf_counter() - request_started_at,
                    function="measure_colocalization_objects",
                )
            continue
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


def _direct_colocalization_batch_request(
    request: RuntimeBatchInvocationRequest,
    prepared_kwargs: Mapping[str, object],
    rank_provider: ObjectColocalizationRankProvider,
) -> object | None:
    """Return direct backend output for batch modes owned by colocalization."""
    image_data = image_payload_data(request.image)
    if (
        request.execution_mode
        is not ImagePayloadExecutionMode.ALIGNED_MULTI_IMAGE_STACK
        or not isinstance(image_data, AlignedImageStack)
    ):
        return None
    slice_results: list[object] = []
    slice_count = len(image_data.slices)
    profile_enabled = runtime_profiler.enabled()
    for slice_index, slice_payload in enumerate(image_data.slices):
        slice_started_at = time.perf_counter() if profile_enabled else 0.0
        slice_kwargs = aligned_image_stack_kwargs(
            prepared_kwargs, slice_index, slice_count, reference_payload=slice_payload
        )
        slice_results.append(
            SINGLETON_STACK_OUTPUT_COLLAPSE.collapse(
                _measure_colocalization_objects_from_runtime_kwargs(
                    slice_payload, slice_kwargs, rank_provider
                )
            )
        )
        if profile_enabled:
            runtime_profiler.log(
                "coloc_object_direct_slice",
                time.perf_counter() - slice_started_at,
                function="measure_colocalization_objects",
            )
    aggregate_started_at = time.perf_counter() if profile_enabled else 0.0
    output = _aggregate_direct_colocalization_slice_results(slice_results)
    if profile_enabled:
        runtime_profiler.log(
            "coloc_object_direct_aggregate",
            time.perf_counter() - aggregate_started_at,
            function="measure_colocalization_objects",
        )
    return output


def _measure_colocalization_objects_from_runtime_kwargs(
    image: object,
    kwargs: Mapping[str, object],
    rank_provider: ObjectColocalizationRankProvider,
) -> object:
    """Execute the object-colocalization core from one prepared runtime request."""
    scale_max = kwargs["scale_max"] if "scale_max" in kwargs else None
    costes_thresholds = (
        kwargs["costes_thresholds"] if "costes_thresholds" in kwargs else None
    )
    return _measure_colocalization_objects_core(
        image,
        kwargs["labels"],
        channel_1=int(kwargs["channel_1"]),
        channel_2=int(kwargs["channel_2"]),
        threshold_percent=float(kwargs["threshold_percent"]),
        do_correlation=bool(kwargs["do_correlation"]),
        do_manders=bool(kwargs["do_manders"]),
        do_rwc=bool(kwargs["do_rwc"]),
        do_overlap=bool(kwargs["do_overlap"]),
        do_costes=bool(kwargs["do_costes"]),
        costes_method=CostesMethod(kwargs["costes_method"]),
        scale_max=None if scale_max is None else int(scale_max),
        costes_backend_provider=kwargs["costes_backend_provider"],
        costes_thresholds=costes_thresholds,
        image_pair_context=kwargs["image_pair_context"],
        object_label_context=kwargs["object_label_context"],
        rank_provider=rank_provider,
    )


def _aggregate_direct_colocalization_slice_results(
    slice_results: list[object],
) -> object:
    """Aggregate direct aligned-stack colocalization results like the executor."""
    result_batch = Pure2DSliceResultBatch.from_results(slice_results)
    memory_type = detect_memory_type(image_payload_data(result_batch.main_outputs[0]))
    stacked_main_output = CellProfilerPure2DOutputAggregator.aggregate(
        result_batch.main_outputs,
        memory_type,
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    )
    if not result_batch.auxiliary_groups:
        return stacked_main_output
    return (
        stacked_main_output,
        *(
            CellProfilerPure2DOutputAggregator.aggregate(
                auxiliary_group, memory_type, plane_axis=RuntimePlaneAxis.RUNTIME_SLICE
            )
            for auxiliary_group in result_batch.auxiliary_groups
        ),
    )


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
