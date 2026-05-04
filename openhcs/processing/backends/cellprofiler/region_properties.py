"""CellProfiler compatibility import path for shared region-property backends."""

from openhcs.processing.backends.analysis.region_properties import (
    AnalysisBackendProvider,
    DenseLabelRegionProperties,
    LabelRegionPropertiesBackendStrategy,
    NumbaNumpyLabelRegionPropertiesBackendStrategy,
    analysis_backend_key,
    label_region_properties_backend,
)

__all__ = [
    "AnalysisBackendProvider",
    "DenseLabelRegionProperties",
    "LabelRegionPropertiesBackendStrategy",
    "NumbaNumpyLabelRegionPropertiesBackendStrategy",
    "analysis_backend_key",
    "label_region_properties_backend",
]
