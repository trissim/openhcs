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

# Import DXF mask pipeline
try:
    from openhcs.processing.backends.analysis.dxf_mask_pipeline import \
        dxf_mask_pipeline
except ImportError:
    pass

# Import focus analyzer
try:
    from openhcs.processing.backends.analysis.focus_analyzer import \
        FocusAnalyzer
except ImportError:
    pass

# Import simple cell counting
try:
    from openhcs.processing.backends.analysis.count_cells_simple import \
        count_cells_simple, count_cells_simple_dual_channel, \
        ThresholdMethod, Foreground, SimpleCellSegmentationConfig, \
        MetaXpressWavelengthSettings, MetaXpressW2Settings, StainedArea
except ImportError:
    pass

# Import skeleton analysis
try:
    from openhcs.processing.backends.analysis.skeletonize_and_save import \
        SkeletonizationResult, skeletonize_and_save
except ImportError:
    pass

# Import MetaXpress-style neurite outgrowth analysis
try:
    from openhcs.processing.backends.analysis.neurite_outgrowth import (
        MetaXpressCellBodySettings,
        MetaXpressNuclearSettings,
        MetaXpressOutgrowthSettings,
        NeuriteIllumination,
        NeuriteOutgrowthCellResult,
        NeuriteOutgrowthSummary,
        neurite_outgrowth_metaxpress,
    )
except ImportError:
    pass

__all__ = [
    # DXF mask pipeline
    "dxf_mask_pipeline",

    # Focus analyzer
    "FocusAnalyzer",

    # Simple cell counting
    "count_cells_simple",
    "count_cells_simple_dual_channel",
    "ThresholdMethod",
    "Foreground",
    "SimpleCellSegmentationConfig",
    "MetaXpressWavelengthSettings",
    "MetaXpressW2Settings",
    "StainedArea",

    # Skeleton analysis
    "SkeletonizationResult",
    "skeletonize_and_save",

    # MetaXpress-style neurite outgrowth
    "neurite_outgrowth_metaxpress",
    "NeuriteIllumination",
    "MetaXpressCellBodySettings",
    "MetaXpressOutgrowthSettings",
    "MetaXpressNuclearSettings",
    "NeuriteOutgrowthSummary",
    "NeuriteOutgrowthCellResult",
]
