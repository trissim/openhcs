"""CellProfiler compatibility import path for shared region-property backends."""

from openhcs.core.public_api import public_names_from_objects
from openhcs.processing.backends.analysis.region_properties import (
    AnalysisBackendProvider,
    DenseLabelRegionProperties,
    LabelRegionPropertiesBackendStrategy,
    NumbaNumpyLabelRegionPropertiesBackendStrategy,
    analysis_backend_key,
    label_region_properties_backend,
)

__all__ = public_names_from_objects(
    AnalysisBackendProvider,
    DenseLabelRegionProperties,
    LabelRegionPropertiesBackendStrategy,
    NumbaNumpyLabelRegionPropertiesBackendStrategy,
    analysis_backend_key,
    label_region_properties_backend,
)
