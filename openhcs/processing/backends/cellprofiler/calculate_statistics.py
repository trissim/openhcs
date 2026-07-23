"""Experiment-level CalculateStatistics measurement semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Tuple

import numpy as np
import scipy.optimize

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactOutputPlan,
    ArtifactSpecCollection,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import (
    KeywordRuntimeParameter,
    runtime_image_execution_mode,
)
from openhcs.core.equivalence.policy import normalize_runtime_identifier
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_feature_queries import (
    MeasurementFeatureQuery,
    MeasurementTableObjectFeatureSemantics,
)
from openhcs.core.measurement_row_materialization import (
    MeasurementProjectedColumnarRows,
    MeasurementSparseColumnarRows,
    measurement_table_axis_values,
)
from openhcs.core.pipeline.function_contracts import (
    runtime_bound_parameters,
    )
from openhcs.core.runtime_artifact_queries import MeasurementTableAxisProjection
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementRowValueField,
    MeasurementScope,
    MeasurementSubject,
    ObjectLocationFeatureMarker,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.interop.cellprofiler.measurement_dialect import (
    CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
)
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_declarations import (
    CellProfilerModule,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.core.steps.function_runtime import RuntimeCallableArgument
from openhcs.interop.cellprofiler.setting_names import setting_values
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    parse_cellprofiler_bool,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeInputBindingRequest,
)


class _CalculateStatisticsMeasurementTablesRuntimeParameter(KeywordRuntimeParameter):
    """Measurement artifacts selected by the compiled module contract."""

    parameter_name = "measurement_tables"
    annotation_type = tuple[MeasurementTable, ...]
    parameter_default = ()


class CalculateStatisticsInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind only declaration-selected measurement artifacts at runtime."""

    binds_without_declared_inputs = True
    supported_non_object_input_kinds = frozenset({MeasurementsArtifactType})

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: RuntimeInputBindingRequest,
    ) -> dict[str, RuntimeCallableArgument]:
        bound = super().bind_runtime_inputs(request)
        tables = request.declared_measurement_tables()
        if not tables:
            raise ValueError(
                "CalculateStatistics requires declared prior measurement artifacts."
            )
        bound[
            _CalculateStatisticsMeasurementTablesRuntimeParameter.require_parameter_name()
        ] = tables
        return bound


@dataclass(frozen=True, slots=True)
class StatisticsFeatureSeries:
    """One eligible prior feature reduced to one value per image slice."""

    object_name: str
    feature_name: str
    values_by_slice: Mapping[int, float]

    def aligned_values(self, slice_indices: Sequence[int]) -> np.ndarray:
        return np.asarray(
            [self.values_by_slice.get(index, np.nan) for index in slice_indices],
            dtype=float,
        )


@dataclass(frozen=True, slots=True)
class LocShrinkMeanStd:
    """Grouped mean/std summary for CellProfiler statistics calculations."""

    labels: np.ndarray
    values: np.ndarray

    def compute(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ncols = self.values.shape[1]
        labels, labnum, xs = _loc_vector_labels(self.labels)
        avers = np.zeros((labnum, ncols))
        stds = avers.copy()

        for ilab in range(labnum):
            labinds = labels == ilab
            labmatr = self.values[labinds, :]
            if labmatr.shape[0] == 1:
                avers[ilab, :] = labmatr[0, :]
            else:
                avers[ilab, :] = np.mean(labmatr, 0)
                stds[ilab, :] = np.std(labmatr, 0)
        return xs, avers, stds


def _loc_vector_labels(x: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray]:
    """Identify unique labels from the vector of image labels.

    Args:
        x: A vector of one label or dose per image

    Returns:
        labels: Ordinal per image indexing into unique labels
        labnum: Number of unique labels
        uniqsortvals: Vector of unique labels
    """
    order = np.lexsort((x,))
    reverse_order = np.lexsort((order,))
    sorted_x = x[order]

    first_occurrence = np.ones(len(x), bool)
    first_occurrence[1:] = sorted_x[:-1] != sorted_x[1:]
    sorted_labels = np.cumsum(first_occurrence) - 1
    labels = sorted_labels[reverse_order]
    uniqsortvals = sorted_x[first_occurrence]
    return labels, len(uniqsortvals), uniqsortvals


def _z_factors(
    xcol: np.ndarray, ymatr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Z' factors for assay quality.

    Args:
        xcol: Grouping values (positive/negative control designations)
        ymatr: Matrix of measurements (observations x measures)

    Returns:
        z: Z' factors
        z_one_tailed: One-tailed Z' factors
        xs: Ordered unique doses
        avers: Ordered average values
    """
    xs, avers, stds = LocShrinkMeanStd(xcol, ymatr).compute()

    # Z' factor from positive and negative controls (extremes by dose)
    zrange = np.abs(avers[0, :] - avers[-1, :])
    zstd = stds[0, :] + stds[-1, :]
    zstd[zrange == 0] = 1
    zrange[zrange == 0] = 0.000001
    z = 1 - 3 * (zstd / zrange)

    # One-tailed Z' factor using only samples between means
    zrange = np.abs(avers[0, :] - avers[-1, :])
    exp1_vals = ymatr[xcol == xs[0], :]
    exp2_vals = ymatr[xcol == xs[-1], :]
    sort_avers = np.sort(np.array((avers[0, :], avers[-1, :])), 0)

    for i in range(sort_avers.shape[1]):
        exp1_cvals = exp1_vals[:, i]
        exp2_cvals = exp2_vals[:, i]
        vals1 = exp1_cvals[
            (exp1_cvals >= sort_avers[0, i]) & (exp1_cvals <= sort_avers[1, i])
        ]
        vals2 = exp2_cvals[
            (exp2_cvals >= sort_avers[0, i]) & (exp2_cvals <= sort_avers[1, i])
        ]
        if len(vals1) > 0:
            stds[0, i] = np.sqrt(np.sum((vals1 - sort_avers[0, i]) ** 2) / len(vals1))
        if len(vals2) > 0:
            stds[1, i] = np.sqrt(np.sum((vals2 - sort_avers[1, i]) ** 2) / len(vals2))

    zstd = stds[0, :] + stds[1, :]
    z_one_tailed = 1 - 3 * (zstd / zrange)
    z_one_tailed[(~np.isfinite(zstd)) | (zrange == 0)] = -1e5

    return z, z_one_tailed, xs, avers


def _v_factors(xcol: np.ndarray, ymatr: np.ndarray) -> np.ndarray:
    """Calculate V factors for assay quality.

    V factor = 1 - 6 * mean(std) / range

    Args:
        xcol: Grouping values (doses)
        ymatr: Matrix of measurements

    Returns:
        v: V factors for each measurement
    """
    xs, avers, stds = LocShrinkMeanStd(xcol, ymatr).compute()
    vrange = np.max(avers, 0) - np.min(avers, 0)

    vstd = np.zeros(len(vrange))
    vstd[vrange == 0] = 1
    vstd[vrange != 0] = np.mean(stds[:, vrange != 0], 0)
    vrange[vrange == 0] = 0.000001
    v = 1 - 6 * (vstd / vrange)
    return v


def _sigmoid(v: np.ndarray, x: np.ndarray) -> np.ndarray:
    """EC50 sigmoid function.

    Args:
        v: Parameters [min, max, ec50, hill_coefficient]
        x: Input values

    Returns:
        Sigmoid response values
    """
    p_min, p_max, ec50, hill = v
    return p_min + ((p_max - p_min) / (1 + (x / ec50) ** hill))


def _calc_init_params(
    x: np.ndarray, y: np.ndarray
) -> Tuple[float, float, float, float]:
    """Calculate initial parameters for sigmoid fitting.

    Args:
        x: Dose values
        y: Response values

    Returns:
        Initial parameters (min, max, ec50, hill)
    """
    min_0 = float(np.min(y))
    max_0 = float(np.max(y))

    y_mid = (min_0 + max_0) / 2
    dist = np.abs(y - y_mid)
    loc = np.argmin(dist)
    x_mid = x[loc]

    if x_mid == np.min(x) or x_mid == np.max(x):
        ec50 = float((np.min(x) + np.max(x)) / 2)
    else:
        ec50 = float(x_mid)

    min_idx = np.argmin(x)
    max_idx = np.argmax(x)
    y0 = y[min_idx]
    y1 = y[max_idx]

    if y1 > y0:
        hillc = -1.0
    else:
        hillc = 1.0

    return min_0, max_0, ec50, hillc


def _calculate_ec50(
    conc: np.ndarray, responses: np.ndarray, log_transform: bool = False
) -> np.ndarray:
    """Calculate EC50 values by fitting dose-response curves.

    Args:
        conc: Concentration/dose values
        responses: Response matrix (observations x measurements)
        log_transform: Whether to log-transform concentrations

    Returns:
        EC50 coefficients matrix (measurements x 4 parameters)
    """
    if log_transform:
        conc = np.log(conc + 1e-10)  # Avoid log(0)

    n = responses.shape[1]
    results = np.zeros((n, 4))

    def error_fn(v, x, y):
        return np.sum((_sigmoid(v, x) - y) ** 2)

    for i in range(n):
        response = responses[:, i]
        try:
            v0 = _calc_init_params(conc, response)
            v = scipy.optimize.fmin(
                error_fn,
                v0,
                args=(conc, response),
                maxiter=1000,
                maxfun=1000,
                disp=False,
            )
            results[i, :] = v
        except (ValueError, RuntimeError):
            results[i, :] = [np.nan, np.nan, np.nan, np.nan]

    return results


def _feature_has_location_semantics(
    table: MeasurementTable,
    feature_name: str,
) -> bool:
    """Query the producer's nominal feature members for location semantics."""
    owner = table.measurement_feature_owner
    if owner is None or not issubclass(owner, CellProfilerModule):
        return False
    parts = tuple(
        part for part in normalize_runtime_identifier(feature_name).split("_") if part
    )
    suffixes = tuple(
        parts[len(prefix) :]
        for prefix in owner.measurement_category_prefixes
        if len(parts) > len(prefix) and parts[: len(prefix)] == prefix
    )
    for feature_type in owner.measurement_feature_types():
        for feature in feature_type:
            family = tuple(part for part in feature.feature_family().split("_") if part)
            if any(suffix[: len(family)] == family for suffix in suffixes) and any(
                issubclass(marker, ObjectLocationFeatureMarker)
                for marker in feature.semantic_markers
            ):
                return True
    return False


def _feature_values_by_slice(
    table: MeasurementTable,
    *,
    feature_name: str,
    object_name: str | None,
) -> Mapping[int, float] | None:
    """Return one numeric image value or per-object mean for each slice."""
    slice_axis = MeasurementRowAxisField.SLICE_INDEX
    slice_indices = tuple(sorted(measurement_table_axis_values(table, slice_axis)))
    if not slice_indices:
        slice_indices = (0,)
    query = MeasurementFeatureQuery(
        feature_name,
        object_name=object_name,
        dialect=CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT,
    )
    values_by_slice: dict[int, float] = {}
    for slice_index in slice_indices:
        projected = MeasurementTableAxisProjection(
            axis=slice_axis,
            value=slice_index,
            table=table,
        ).apply()
        try:
            indexed = query.optional_value_index((projected,))
        except (TypeError, ValueError):
            return None
        if indexed is None:
            continue
        values_by_label, positional_values = indexed
        values = tuple(values_by_label.values()) + tuple(positional_values)
        if table.subject.scope is MeasurementScope.OBJECT:
            finite_values = tuple(value for value in values if np.isfinite(value))
            values_by_slice[slice_index] = (
                float(np.mean(finite_values)) if finite_values else float("nan")
            )
            continue
        if len(values) != 1:
            raise ValueError(
                "CalculateStatistics requires one image value per feature and "
                f"slice; {table.name!r}/{feature_name!r}/slice={slice_index} "
                f"produced {len(values)} values."
            )
        values_by_slice[slice_index] = float(values[0])
    return MappingProxyType(values_by_slice)


def _statistics_feature_series(
    measurement_tables: Sequence[MeasurementTable],
) -> tuple[StatisticsFeatureSeries, ...]:
    """Enumerate numeric image and averaged-object features from typed tables."""
    series_by_identity: dict[tuple[str, str], dict[int, float]] = {}
    for table in measurement_tables:
        if table.subject.scope not in {MeasurementScope.IMAGE, MeasurementScope.OBJECT}:
            continue
        semantics = MeasurementTableObjectFeatureSemantics.from_table(table)
        object_names = (
            (MeasurementScope.IMAGE.value.title(),)
            if table.subject.scope is MeasurementScope.IMAGE
            else semantics.object_names
        )
        for object_name in object_names:
            query_object_name = (
                None if table.subject.scope is MeasurementScope.IMAGE else object_name
            )
            for feature_name in sorted(semantics.feature_names):
                if (
                    feature_name in MeasurementRowAxisField.field_names()
                    or feature_name in MeasurementRowValueField.field_names()
                ):
                    continue
                if _feature_has_location_semantics(table, feature_name):
                    continue
                values = _feature_values_by_slice(
                    table,
                    feature_name=feature_name,
                    object_name=query_object_name,
                )
                if values is None or not values:
                    continue
                target = series_by_identity.setdefault((object_name, feature_name), {})
                for slice_index, value in values.items():
                    existing = target.get(slice_index)
                    if existing is not None and not np.isclose(
                        existing,
                        value,
                        equal_nan=True,
                    ):
                        raise ValueError(
                            "CalculateStatistics received conflicting values for "
                            f"{object_name!r}/{feature_name!r}/slice={slice_index}."
                        )
                    target[slice_index] = value
    return tuple(
        StatisticsFeatureSeries(
            object_name,
            feature_name,
            MappingProxyType(values_by_slice),
        )
        for (object_name, feature_name), values_by_slice in series_by_identity.items()
    )


def _require_image_series(
    series: Sequence[StatisticsFeatureSeries],
    feature_name: str,
) -> StatisticsFeatureSeries:
    matching = tuple(
        item
        for item in series
        if item.object_name == "Image" and item.feature_name == feature_name
    )
    if len(matching) != 1:
        raise ValueError(
            "CalculateStatistics requires exactly one declared image measurement "
            f"series for {feature_name!r}, got {matching!r}."
        )
    return matching[0]


def _statistics_rows(
    measurement_tables: Sequence[MeasurementTable],
    *,
    grouping_feature: str,
    dose_features: tuple[str, ...],
    log_transform_doses: tuple[bool, ...],
) -> ColumnarRows:
    if not measurement_tables:
        raise ValueError(
            "CalculateStatistics requires declared prior measurement artifacts."
        )
    if not dose_features:
        raise ValueError("CalculateStatistics requires at least one dose feature.")
    if len(dose_features) != len(log_transform_doses):
        raise ValueError(
            "CalculateStatistics dose features and log-transform flags must have "
            "equal cardinality."
        )

    all_series = _statistics_feature_series(measurement_tables)
    grouping = _require_image_series(all_series, grouping_feature)
    doses = tuple(
        _require_image_series(all_series, dose_feature)
        for dose_feature in dose_features
    )
    slice_indices = tuple(sorted(grouping.values_by_slice))
    if not slice_indices:
        raise ValueError("CalculateStatistics grouping feature has no image values.")
    grouping_values = grouping.aligned_values(slice_indices)
    dose_values = tuple(dose.aligned_values(slice_indices) for dose in doses)
    if any(
        not np.all(np.isfinite(values)) for values in (grouping_values, *dose_values)
    ):
        raise ValueError(
            "CalculateStatistics grouping and dose features require finite values "
            "for every image slice."
        )

    control_features = frozenset((grouping_feature, *dose_features))
    eligible = tuple(
        item
        for item in all_series
        if not (item.object_name == "Image" and item.feature_name in control_features)
    )
    if not eligible:
        return MeasurementSparseColumnarRows.from_rows((), fields=())
    data = np.column_stack(
        tuple(item.aligned_values(slice_indices) for item in eligible)
    )
    z_factors, one_tailed_z_factors, _, _ = _z_factors(grouping_values, data)
    v_factors = _v_factors(dose_values[0], data)
    ec50_values = tuple(
        _calculate_ec50(dose, data, log_transform)[:, 2]
        for dose, log_transform in zip(
            dose_values,
            log_transform_doses,
            strict=True,
        )
    )

    columns: dict[str, tuple[float]] = {}
    for feature_index, item in enumerate(eligible):
        suffix = f"{item.object_name}_{item.feature_name}"
        columns[f"Zfactor_{suffix}"] = (float(z_factors[feature_index]),)
        columns[f"Vfactor_{suffix}"] = (float(v_factors[feature_index]),)
        columns[f"OneTailedZfactor_{suffix}"] = (
            float(one_tailed_z_factors[feature_index]),
        )
        for dose_index, (dose_feature, values) in enumerate(
            zip(dose_features, ec50_values, strict=True)
        ):
            ec50_name = "EC50" if len(dose_features) == 1 else f"EC50_{dose_feature}"
            columns[f"{ec50_name}_{suffix}"] = (float(values[feature_index]),)
    fields = tuple(FieldSpec(name, float, required=False) for name in columns)
    return MeasurementProjectedColumnarRows(
        MappingProxyType(columns),
        fields=fields,
    )


@runtime_image_execution_mode(ImagePayloadExecutionMode.FULL_STACK)
@numpy(contract=ProcessingContract.PURE_2D)
@runtime_bound_parameters(_CalculateStatisticsMeasurementTablesRuntimeParameter)
def calculate_statistics(
    image: np.ndarray,
    grouping_feature: str = "Metadata_Control",
    dose_features: tuple[str, ...] = ("Metadata_Dose",),
    log_transform_doses: tuple[bool, ...] = (False,),
    *,
    measurement_tables: tuple[MeasurementTable, ...] = (),
) -> tuple[np.ndarray, ColumnarRows]:
    """Calculate experiment statistics from contract-selected measurements."""
    return image, _statistics_rows(
        measurement_tables,
        grouping_feature=grouping_feature,
        dose_features=tuple(dose_features),
        log_transform_doses=tuple(bool(value) for value in log_transform_doses),
    )


class CalculateStatisticsModule(
    CalculateStatisticsInputPolicy,
    MeasurementArtifactOutputModule,
    CellProfilerModule,
):
    module_name = "CalculateStatistics"
    function_name = "calculate_statistics"
    validated = True
    confidence = 1.0
    measurement_category_prefixes = (
        ("zfactor",),
        ("vfactor",),
        ("one", "tailed", "zfactor"),
        ("ec50",),
    )
    grouping_setting: ClassVar[str] = (
        "Select the image measurement describing the positive and negative control status"
    )
    dose_setting: ClassVar[str] = (
        "Select the image measurement describing the treatment dose"
    )
    log_transform_setting: ClassVar[str] = "Log-transform the dose values?"
    wants_figure_setting: ClassVar[str] = "Create dose-response plots?"
    figure_name_setting: ClassVar[str] = "Figure prefix"
    output_path_setting: ClassVar[str] = "Output file location"
    setting_bindings = (
        SettingToKeywordBinding(grouping_setting, "grouping_feature", str),
        SettingToKeywordBinding(
            dose_setting,
            "dose_features",
            str,
            repeated=True,
        ),
        SettingToKeywordBinding(
            log_transform_setting,
            "log_transform_doses",
            parse_cellprofiler_bool,
            repeated=True,
        ),
    )
    ignored_settings = (
        wants_figure_setting,
        figure_name_setting,
        output_path_setting,
    )

    @classmethod
    def bind_settings(cls, module, *, binder):
        """Bind repeated dose rows with exact cardinality and booleans."""
        bound = cls._bind_declared_settings(module, binder=binder)
        dose_features = tuple(setting_values(module, cls.dose_setting))
        log_transform_values = tuple(
            parse_cellprofiler_bool(value)
            for value in setting_values(module, cls.log_transform_setting)
        )
        if len(dose_features) != len(log_transform_values):
            raise ValueError(
                "CalculateStatistics parsed dose rows have mismatched feature and "
                "log-transform cardinality."
            )
        kwargs = dict(bound.kwargs)
        kwargs["dose_features"] = dose_features
        kwargs["log_transform_doses"] = log_transform_values
        return cls._finalize_bound_settings(
            module,
            binder=binder,
            bound=BoundModuleSettings(kwargs, bound.unmapped_kwargs),
        )

    @classmethod
    def finalize_artifact_contract_inputs(
        cls,
        module,
        *,
        invocation_key,
        step_context,
        artifact_inputs: ArtifactSpecCollection,
    ):
        """Consume every prior declared measurement output exactly once."""
        direct = ArtifactSpecCollection(
            super().finalize_artifact_contract_inputs(
                module,
                invocation_key=invocation_key,
                step_context=step_context,
                artifact_inputs=artifact_inputs,
            )
        )
        prior_measurements = tuple(
            dict.fromkeys(
                producer.spec.for_plan_type(ArtifactInputPlan)
                for producer in step_context.available_artifact_producers
                if producer.spec.plan_type is ArtifactOutputPlan
                and producer.spec.artifact_type is MeasurementsArtifactType
            )
        )
        return ArtifactSpecCollection((*direct.specs, *prior_measurements)).unique(
            conflict_context="CalculateStatistics prior measurement input"
        )

    @classmethod
    def build_measurement_table(
        cls,
        *,
        name: str,
        rows: ColumnarRows,
        object_name: str | None,
        source_image_name: str | None,
        source_metadata,
    ) -> MeasurementTable:
        """Own the callable's rows as one experiment-scoped measurement table."""
        del object_name, source_image_name, source_metadata
        return MeasurementTable(
            name=name,
            rows=rows,
            subject=MeasurementSubject(MeasurementScope.EXPERIMENT),
            measurement_feature_owner=cls,
        )
