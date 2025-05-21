"""
Position generation backend implementations.

This package contains implementations of various position generation algorithms
for microscopy image stitching, including MIST and Ashlar.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 88 — No Inferred Capabilities: Explicit backend requirements
- Clause 106-A — Declared Memory Types: All methods specify memory types
- Clause 273 — Memory Backend Restrictions: GPU-only implementations are marked
"""

# Import MIST processor implementation
try:
    from openhcs.processing.backends.pos_gen.mist_processor_cupy import \
        mist_compute_tile_positions
except ImportError:
    pass

# Import Ashlar processor implementation
try:
    from openhcs.processing.backends.pos_gen.ashlar_processor_cupy import \
        gpu_ashlar_align_cupy
except ImportError:
    pass

__all__ = [
    # MIST processor implementation
    "mist_compute_tile_positions",
    
    # Ashlar processor implementation
    "gpu_ashlar_align_cupy",
]
