"""CellProfiler measurement-name dialect for semantic output equivalence."""

from __future__ import annotations
from types import MappingProxyType
from openhcs.core.equivalence import (
    RuntimeEquivalencePolicy,
    RuntimeMeasurementDialect,
    RuntimeMeasurementSourceNameEncoding,
)
from openhcs.core.measurement_lookup_dialect import (
    RuntimeMeasurementFeatureLookup,
    RuntimeMeasurementLookupDialect,
    RuntimeMeasurementObjectDomainPolicy,
)
from openhcs.core.runtime_semantics import (
    MeasurementScope,
    ObjectCoreMeasurementFeature,
    RuntimeMeasurementIndexedDescriptorDeclaration,
)
from openhcs.interop.cellprofiler.measurement_lookup import (
    child_count_feature_child_name,
)
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler import (
    measurement_semantic_profiles as _measurement_semantic_profiles,
)

BENCHMARK_CACHE_DOMAINS = frozenset({"parity"})
CELLPROFILER_CORE_MEASUREMENT_FEATURE_PART_ALIASES = MappingProxyType(
    {
        ("number", "object", "number"): ("object", "number"),
        **{
            tuple(feature.value.split("_")): tuple(feature.value.split("_"))
            for feature in (
                ObjectCoreMeasurementFeature.CENTER_X,
                ObjectCoreMeasurementFeature.CENTER_Y,
                ObjectCoreMeasurementFeature.CENTER_Z,
            )
        },
    }
)
class CellProfilerMeasurementObjectDomainPolicy(RuntimeMeasurementObjectDomainPolicy):
    """Object-domain semantics for CellProfiler measurement rows."""

    def query_object_name(
        self, lookup: RuntimeMeasurementFeatureLookup, object_name: str | None
    ) -> str | None:
        """Return the CellProfiler row object constraint for a feature lookup."""
        if child_count_feature_child_name(lookup.feature_name) is not None:
            return None
        return object_name


CELLPROFILER_MEASUREMENT_LOOKUP_DIALECT = RuntimeMeasurementLookupDialect(
    category_prefixes_provider=CellProfilerModule.measurement_category_prefix_declarations,
    feature_part_aliases=CELLPROFILER_CORE_MEASUREMENT_FEATURE_PART_ALIASES,
    feature_part_aliases_provider=CellProfilerModule.measurement_feature_part_rewrite_declarations,
    alternative_feature_part_aliases_provider=(
        CellProfilerModule.alternative_measurement_feature_part_aliases
    ),
    source_qualified_feature_families_provider=CellProfilerModule.source_qualified_measurement_feature_family_parts,
    object_domain_policy=CellProfilerMeasurementObjectDomainPolicy(),
)
CELLPROFILER_MEASUREMENT_DIALECT = RuntimeMeasurementDialect(
    category_prefixes_provider=CellProfilerModule.measurement_category_prefix_declarations,
    feature_part_aliases=CELLPROFILER_CORE_MEASUREMENT_FEATURE_PART_ALIASES,
    feature_part_aliases_provider=CellProfilerModule.measurement_feature_part_rewrite_declarations,
    source_feature_prefixes_provider=CellProfilerModule.measurement_source_feature_prefix_declarations,
    calculated_feature_prefixes_provider=CellProfilerModule.calculated_measurement_feature_prefix_declarations,
    directional_pair_feature_aliases_provider=CellProfilerModule.directional_pair_feature_alias_declarations,
    scale_qualified_feature_prefixes_provider=(
        CellProfilerModule.scale_qualified_measurement_feature_prefix_declarations
    ),
    pair_correlation_feature_name_provider=CellProfilerModule.pair_correlation_feature_name_declaration,
    pair_regression_slope_feature_name_provider=(
        CellProfilerModule.pair_regression_slope_feature_name_declaration
    ),
    undirected_pair_feature_names_provider=CellProfilerModule.undirected_pair_feature_name_declarations,
    threshold_sensitive_pair_feature_names_provider=(
        CellProfilerModule.threshold_sensitive_pair_feature_name_declarations
    ),
    measurement_feature_marker_provider=(
        CellProfilerModule.measurement_feature_marker_types_for_key
    ),
    indexed_descriptor_suffix_width_provider=(
        RuntimeMeasurementIndexedDescriptorDeclaration.indexed_suffix_token_width_for
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
    numbered_feature_prefix_aliases_provider=(
        CellProfilerModule.numbered_measurement_feature_prefix_alias_declarations
    ),
    source_name_encoding_by_scope=MappingProxyType(
        {
            MeasurementScope.IMAGE: RuntimeMeasurementSourceNameEncoding.FEATURE_SUFFIX,
            MeasurementScope.OBJECT: RuntimeMeasurementSourceNameEncoding.FEATURE_SUFFIX,
        }
    ),
    measurement_feature_relation_provider=CellProfilerModule.measurement_feature_relation_declarations,
)
def cellprofiler_runtime_equivalence_policy(
    **overrides: object,
) -> RuntimeEquivalencePolicy:
    """Build a runtime-equivalence policy with CellProfiler measurement dialect."""
    overrides.setdefault("measurement_dialect", CELLPROFILER_MEASUREMENT_DIALECT)
    overrides.setdefault("numeric_abs_tolerance", 1e-06)
    overrides.setdefault("numeric_rel_tolerance", 1e-06)
    overrides.setdefault("threshold_entropy_abs_tolerance", 0.04)
    overrides.setdefault("allow_tie_sensitive_location_mismatches", True)
    overrides.setdefault("allow_sparse_object_boundary_jitter", True)
    overrides.setdefault("allow_unstable_shape_descriptors", True)
    overrides.setdefault("allow_unstable_zernike_descriptors", True)
    overrides.setdefault("shape_descriptor_abs_tolerance", 1e-06)
    overrides.setdefault("zernike_descriptor_magnitude_abs_tolerance", 1e-06)
    overrides.setdefault("object_boundary_jitter_abs_tolerance", 5.0)
    overrides.setdefault("object_boundary_jitter_max_unstable_values", 50)
    overrides.setdefault("object_boundary_jitter_max_unstable_fraction", 0.02)
    overrides.setdefault("object_boundary_jitter_aggregate_abs_tolerance", 1.5)
    overrides.setdefault("image_abs_tolerance", 1e-06)
    overrides.setdefault("image_rel_tolerance", 1e-06)
    return RuntimeEquivalencePolicy(**overrides)
