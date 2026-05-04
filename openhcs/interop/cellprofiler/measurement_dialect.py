"""CellProfiler measurement-name dialect for semantic output equivalence."""

from __future__ import annotations

from types import MappingProxyType

from openhcs.core.equivalence import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementFeatureNumericTolerance,
    RuntimeMeasurementDialect,
)
from openhcs.core.runtime_semantics import PairMeasurementFeature
from openhcs.core.runtime_semantics import MeasurementScope


BENCHMARK_CACHE_DOMAINS = frozenset({"parity"})
CELLPROFILER_MEASUREMENT_DIALECT = RuntimeMeasurementDialect(
    category_prefixes=(
        ("area", "occupied"),
        ("image", "quality"),
        ("area", "shape"),
        ("intensity",),
        ("texture",),
        ("location",),
        ("children",),
        ("parent",),
        ("neighbors",),
        ("math",),
        ("classify",),
        ("correlation",),
        ("colocalization",),
        ("quality",),
        ("radial", "distribution"),
        ("threshold",),
    ),
    feature_part_aliases=MappingProxyType(
        {
            ("area", "retained"): ("crop", "area", "retained", "after", "cropping"),
            ("number", "object", "number"): ("object", "number"),
            ("original", "area"): ("crop", "original", "image", "area"),
            ("otsu",): ("threshold", "otsu"),
        }
    ),
    source_feature_prefixes=(
        ("crop", "area", "retained", "after", "cropping"),
        ("crop", "original", "image", "area"),
    ),
    directional_pair_feature_aliases=MappingProxyType(
        {
            "costes_m_1": (PairMeasurementFeature.COSTES_MANDERS.value, 1),
            "costes_m_2": (PairMeasurementFeature.COSTES_MANDERS.value, 2),
            "k_1": (PairMeasurementFeature.OVERLAP_K.value, 1),
            "k_2": (PairMeasurementFeature.OVERLAP_K.value, 2),
            "manders_m_1": (PairMeasurementFeature.MANDERS.value, 1),
            "manders_m_2": (PairMeasurementFeature.MANDERS.value, 2),
            "slope_reverse": (PairMeasurementFeature.REGRESSION_SLOPE.value, 2),
            "rwc_1": (
                PairMeasurementFeature.RANK_WEIGHTED_COLOCALIZATION.value,
                1,
            ),
            "rwc_2": (
                PairMeasurementFeature.RANK_WEIGHTED_COLOCALIZATION.value,
                2,
            ),
        }
    ),
    scale_qualified_feature_prefixes=(
        (PairMeasurementFeature.CORRELATION.value,),
        ("local", "focus", "score"),
    ),
    threshold_qualifier_tokens=frozenset(
        {
            "w",
            "weighted",
            "variance",
            "entropy",
            "foreground",
            "background",
            "classes",
            "class",
        }
    ),
    source_qualifier_prefix_tokens=frozenset({"crop", "orig", "raw", "image"}),
    source_qualifier_suffix_tokens=frozenset(
        {"red", "green", "blue", "gray", "grey", "dna", "gfp", "rfp"}
    ),
    numbered_feature_prefix_aliases=MappingProxyType(
        {"gs": ("granularity",)}
    ),
)
CELLPROFILER_FEATURE_NUMERIC_TOLERANCES = (
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=(
            "frac_at_d_",
            "mean_frac_",
            "radial_cv_",
        ),
        subject_scope=MeasurementScope.OBJECT,
        statistic="value",
        numeric_abs_tolerance=0.25,
        numeric_rel_tolerance=0.05,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=("granularity_",),
        subject_scope=MeasurementScope.OBJECT,
        statistic="value",
        numeric_abs_tolerance=0.5,
        numeric_rel_tolerance=0.5,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=("weighted_variance",),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=6e-3,
        numeric_rel_tolerance=1e-5,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=(
            "angular_second_moment_",
            "contrast_",
            "correlation_",
            "difference_entropy_",
            "entropy_",
            "info_meas_1_",
            "info_meas_2_",
            "inverse_difference_moment_",
            "sum_average_",
            "sum_entropy_",
            "sum_variance_",
            "variance_",
        ),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=1.5e-3,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=("final_threshold", "orig_threshold"),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=5e-4,
        numeric_rel_tolerance=5e-4,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"area_occupied"}),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=1000.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"perimeter"}),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=100.0,
        numeric_rel_tolerance=1e-2,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"colocalized"}),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=5e-4,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"area"}),
        subject_scope=MeasurementScope.OBJECT,
        numeric_abs_tolerance=1.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"bounding_box_area"}),
        subject_scope=MeasurementScope.OBJECT,
        numeric_abs_tolerance=32.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"bounding_box_minimum_x", "bounding_box_minimum_y"}),
        subject_scope=MeasurementScope.OBJECT,
        numeric_abs_tolerance=1.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"convex_area"}),
        subject_scope=MeasurementScope.OBJECT,
        numeric_abs_tolerance=3.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"perimeter"}),
        subject_scope=MeasurementScope.OBJECT,
        numeric_abs_tolerance=1.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset({"center_x", "center_y"}),
        subject_scope=MeasurementScope.OBJECT,
        statistic="mean",
        numeric_abs_tolerance=1.0,
        numeric_rel_tolerance=1e-3,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=("align_xshift", "align_yshift"),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=1.0,
        numeric_rel_tolerance=0.0,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset(
            {
                "defined_grid_grid_x_spacing",
                "defined_grid_grid_y_spacing",
            }
        ),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=5e-2,
        numeric_rel_tolerance=5e-4,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_names=frozenset(
            {
                "defined_grid_grid_x_location_of_lowest_x_spot",
                "defined_grid_grid_y_location_of_lowest_y_spot",
            }
        ),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=1.0,
        numeric_rel_tolerance=0.0,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=(
            "large_num_objects_per_bin",
            "small_num_objects_per_bin",
            "tiny_num_objects_per_bin",
            "red_num_objects_per_bin",
            "white_num_objects_per_bin",
        ),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=25.0,
        numeric_rel_tolerance=0.0,
    ),
    RuntimeMeasurementFeatureNumericTolerance(
        feature_name_prefixes=(
            "large_pct_objects_per_bin",
            "small_pct_objects_per_bin",
            "tiny_pct_objects_per_bin",
            "red_pct_objects_per_bin",
            "white_pct_objects_per_bin",
        ),
        subject_scope=MeasurementScope.IMAGE,
        statistic="value",
        numeric_abs_tolerance=1.1,
        numeric_rel_tolerance=0.0,
    ),
)


def cellprofiler_runtime_equivalence_policy(
    **overrides: object,
) -> RuntimeEquivalencePolicy:
    """Build a runtime-equivalence policy with CellProfiler measurement dialect."""
    overrides.setdefault("measurement_dialect", CELLPROFILER_MEASUREMENT_DIALECT)
    overrides.setdefault(
        "feature_numeric_tolerances",
        CELLPROFILER_FEATURE_NUMERIC_TOLERANCES,
    )
    overrides.setdefault("numeric_abs_tolerance", 1e-6)
    overrides.setdefault("numeric_rel_tolerance", 1e-6)
    overrides.setdefault("threshold_entropy_abs_tolerance", 4e-2)
    overrides.setdefault("allow_tie_sensitive_location_mismatches", True)
    overrides.setdefault("allow_sparse_object_boundary_jitter", True)
    overrides.setdefault("object_boundary_jitter_abs_tolerance", 5.0)
    overrides.setdefault("object_boundary_jitter_max_unstable_values", 50)
    overrides.setdefault("object_boundary_jitter_max_unstable_fraction", 0.02)
    overrides.setdefault("object_boundary_jitter_aggregate_abs_tolerance", 1.5)
    overrides.setdefault("image_abs_tolerance", 1e-6)
    overrides.setdefault("image_rel_tolerance", 1e-6)
    return RuntimeEquivalencePolicy(**overrides)
