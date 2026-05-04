"""
Image analysis backend implementations.

This package contains implementations of various image analysis algorithms
for different memory backends (NumPy, CuPy, PyTorch, TensorFlow, JAX).

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
- Clause 273 — Memory Backend Restrictions: GPU-only implementations are marked
"""

from openhcs.processing.backends.analysis.region_properties import (
    AnalysisBackendProvider,
    DenseLabelRegionProperties,
    LabelRegionPropertiesBackendStrategy,
    NumbaNumpyLabelRegionPropertiesBackendStrategy,
    label_region_properties_backend,
)

__all__ = [
    "AnalysisBackendProvider",
    "DenseLabelRegionProperties",
    "LabelRegionPropertiesBackendStrategy",
    "NumbaNumpyLabelRegionPropertiesBackendStrategy",
    "label_region_properties_backend",
]
