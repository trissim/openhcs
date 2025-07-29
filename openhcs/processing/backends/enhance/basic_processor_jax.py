"""
BaSiC (Background and Shading Correction) Implementation using JAX via BaSiCPy

This module provides OpenHCS-compatible wrapper functions for BaSiCPy's
JAX-based BaSiC implementation, integrating with OpenHCS memory decorators
and pipeline system.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit JAX dependency via BaSiCPy
- Clause 273 — Memory Backend Restrictions: JAX-only implementation
"""
from __future__ import annotations 

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

# Import decorator directly from core.memory to avoid circular imports
from openhcs.core.memory.decorators import jax as jax_func
from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import jax.numpy as jnp

# Import jax.numpy for runtime type hint evaluation
try:
    import jax.numpy as jnp
except ImportError:
    jnp = None

# Import BaSiCPy as an optional dependency
basicpy = optional_import("basicpy")
if basicpy is not None:
    BaSiC = basicpy.BaSiC
else:
    BaSiC = None

logger = logging.getLogger(__name__)


def _validate_jax_array(array: Any, name: str = "input") -> None:
    """
    Validate that BaSiCPy is available and input is compatible.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        ImportError: If BaSiCPy is not available
        ValueError: If the array is not compatible
    """
    if basicpy is None or BaSiC is None:
        raise ImportError(
            "BaSiCPy is not available. Please install BaSiCPy for BaSiC correction. "
            "Install with: pip install basicpy"
        )

    if not hasattr(array, 'shape') or not hasattr(array, 'dtype'):
        raise ValueError(
            f"{name} must be an array-like object with shape and dtype attributes, "
            f"got {type(array)}."
        )


@jax_func
def basic_flatfield_correction_jax(
    image: "jnp.ndarray",
    max_iters: int = 50,
    lambda_sparse: float = 0.01,
    lambda_lowrank: float = 0.1,
    epsilon: float = 0.1,
    smoothness_flatfield: float = 1.0,
    smoothness_darkfield: float = 1.0,
    sparse_cost_darkfield: float = 0.01,
    get_darkfield: bool = False,
    fitting_mode: str = "ladmap",
    working_size: Optional[Union[int, list]] = 128,
    verbose: bool = False,
    **kwargs
) -> "jnp.ndarray":
    """
    Perform BaSiC-style illumination correction on a 3D image stack using JAX via BaSiCPy.

    This function provides OpenHCS integration for BaSiCPy's sophisticated BaSiC
    algorithm implementation, supporting both LADMAP and approximate fitting modes
    with automatic parameter tuning capabilities.

    Args:
        image: 3D JAX array of shape (Z, Y, X)
        max_iters: Maximum number of iterations for optimization
        lambda_sparse: Regularization parameter for sparse component (mapped to epsilon)
        lambda_lowrank: Regularization parameter for low-rank component (mapped to smoothness)
        epsilon: Weight regularization term
        smoothness_flatfield: Weight of flatfield term in Lagrangian
        smoothness_darkfield: Weight of darkfield term in Lagrangian
        sparse_cost_darkfield: Weight of darkfield sparse term in Lagrangian
        get_darkfield: Whether to estimate darkfield component
        fitting_mode: Fitting mode ('ladmap' or 'approximate')
        working_size: Size for running computations (None means no rescaling)
        verbose: Whether to print progress information
        **kwargs: Additional parameters (ignored for compatibility)

    Returns:
        Corrected 3D JAX array of shape (Z, Y, X)

    Raises:
        ImportError: If BaSiCPy is not available
        ValueError: If input is not a 3D array
        RuntimeError: If BaSiC fitting fails
    """
    # Validate input and dependencies
    _validate_jax_array(image)

    if image.ndim != 3:
        raise ValueError(f"Input must be a 3D array, got {image.ndim}D")

    logger.debug(f"BaSiC correction: {image.shape} image, mode={fitting_mode}")

    try:
        # Convert JAX array to numpy for BaSiCPy (it handles JAX internally)
        import numpy as np
        image_np = np.asarray(image)

        # Create BaSiC instance with parameters
        basic = BaSiC(
            # Core algorithm parameters
            max_iterations=max_iters,
            epsilon=epsilon,
            smoothness_flatfield=smoothness_flatfield,
            smoothness_darkfield=smoothness_darkfield,
            sparse_cost_darkfield=sparse_cost_darkfield,
            get_darkfield=get_darkfield,
            fitting_mode=fitting_mode,
            working_size=working_size,
            
            # Optimization parameters
            optimization_tol=1e-3,
            optimization_tol_diff=1e-2,
            reweighting_tol=1e-2,
            max_reweight_iterations=10,
            
            # Memory and performance
            resize_mode="jax",
            sort_intensity=False,
        )

        # Fit and transform the image
        logger.debug("Starting BaSiC fit and transform")
        corrected_np = basic.fit_transform(image_np, timelapse=False)

        # Convert back to JAX array
        import jax.numpy as jnp
        corrected = jnp.asarray(corrected_np)

        logger.debug(f"BaSiC correction completed: {corrected.shape}")
        return corrected.astype(image.dtype)

    except Exception as e:
        logger.error(f"BaSiC correction failed: {e}")
        raise RuntimeError(f"BaSiC flat field correction failed: {e}") from e


@jax_func
def basic_flatfield_correction_batch_jax(
    image_batch: "jnp.ndarray",
    *,
    batch_dim: int = 0,
    **kwargs
) -> "jnp.ndarray":
    """
    Apply BaSiC flatfield correction to a batch of 3D image stacks.

    Args:
        image_batch: 4D JAX array of shape (B, Z, Y, X) or (Z, B, Y, X)
        batch_dim: Dimension along which the batch is organized (0 or 1)
        **kwargs: Additional parameters passed to basic_flatfield_correction_jax

    Returns:
        Corrected 4D JAX array of the same shape as input

    Raises:
        ImportError: If BaSiCPy is not available
        ValueError: If input is not a 4D array or batch_dim is invalid
    """
    # Validate input
    _validate_jax_array(image_batch)

    if image_batch.ndim != 4:
        raise ValueError(f"Input must be a 4D array, got {image_batch.ndim}D")

    if batch_dim not in [0, 1]:
        raise ValueError(f"batch_dim must be 0 or 1, got {batch_dim}")

    logger.debug(f"BaSiC batch correction: {image_batch.shape}, batch_dim={batch_dim}")

    # Process each 3D stack in the batch
    result_list = []

    if batch_dim == 0:
        # Batch is organized as (B, Z, Y, X)
        for b in range(image_batch.shape[0]):
            corrected = basic_flatfield_correction_jax(image_batch[b], **kwargs)
            result_list.append(corrected)

        # Stack along batch dimension
        import jax.numpy as jnp
        return jnp.stack(result_list, axis=0)

    # Batch is organized as (Z, B, Y, X)
    for b in range(image_batch.shape[1]):
        corrected = basic_flatfield_correction_jax(image_batch[:, b], **kwargs)
        result_list.append(corrected)

    # Stack along batch dimension
    import jax.numpy as jnp
    return jnp.stack(result_list, axis=1)
