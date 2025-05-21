"""
Image enhancement backend implementations.

This package contains implementations of various image enhancement algorithms
for different memory backends (NumPy, CuPy, PyTorch, TensorFlow, JAX).

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
- Clause 273 — Memory Backend Restrictions: GPU-only implementations are marked
"""

# Import BaSiC processor implementations
try:
    from openhcs.processing.backends.enhance.basic_processor_numpy import (
        basic_flatfield_correction_batch_numpy,
        basic_flatfield_correction_numpy)
except ImportError:
    pass

try:
    from openhcs.processing.backends.enhance.basic_processor_cupy import (
        basic_flatfield_correction_batch_cupy, basic_flatfield_correction_cupy)
except ImportError:
    pass

# Import N2V2 processor implementation
try:
    from openhcs.processing.backends.enhance.n2v2_processor_torch import \
        n2v2_denoise_torch
except ImportError:
    pass

__all__ = [
    # BaSiC processor implementations
    "basic_flatfield_correction_numpy",
    "basic_flatfield_correction_batch_numpy",
    "basic_flatfield_correction_cupy",
    "basic_flatfield_correction_batch_cupy",
    
    # N2V2 processor implementation
    "n2v2_denoise_torch",
]
