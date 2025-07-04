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

# Auto-registration is now handled centrally in func_registry._scan_and_register_functions()
# No need to trigger it here anymore

__all__ = [
    # DXF mask pipeline
    "dxf_mask_pipeline",

    # Focus analyzer
    "FocusAnalyzer",
]
