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

try:
    from openhcs.processing.backends.enhance.basic_processor_jax import (
        basic_flatfield_correction_batch_jax, basic_flatfield_correction_jax)
except ImportError:
    pass

# Import N2V2 processor implementation
try:
    from openhcs.processing.backends.enhance.n2v2_processor_torch import \
        n2v2_denoise_torch
except ImportError:
    pass

# Import self-supervised deconvolution implementations
try:
    from openhcs.processing.backends.enhance.self_supervised_2d_deconvolution import \
        self_supervised_2d_deconvolution
except ImportError:
    pass

try:
    from openhcs.processing.backends.enhance.self_supervised_3d_deconvolution import \
        self_supervised_3d_deconvolution
except ImportError:
    pass

# Import torch_nlm processor implementation
try:
    from openhcs.processing.backends.enhance.torch_nlm_processor import (
        non_local_means_denoise_torch,
        torch_nlm_denoise
    )
except ImportError:
    pass

# Import JAX NLM processor implementation
try:
    from openhcs.processing.backends.enhance.jax_nlm_processor import (
        non_local_means_denoise_jax,
        jax_nlm_denoise
    )
except ImportError:
    pass

__all__ = [
    # BaSiC processor implementations
    "basic_flatfield_correction_numpy",
    "basic_flatfield_correction_batch_numpy",
    "basic_flatfield_correction_cupy",
    "basic_flatfield_correction_batch_cupy",
    "basic_flatfield_correction_jax",
    "basic_flatfield_correction_batch_jax",

    # N2V2 processor implementation
    "n2v2_denoise_torch",

    # Self-supervised deconvolution implementations
    "self_supervised_2d_deconvolution",
    "self_supervised_3d_deconvolution",

    # torch_nlm processor implementations
    "non_local_means_denoise_torch",
    "torch_nlm_denoise",

    # JAX NLM processor implementations
    "non_local_means_denoise_jax",
    "jax_nlm_denoise",
]
