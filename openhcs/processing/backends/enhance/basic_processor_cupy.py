"""
BaSiC (Background and Shading Correction) Implementation using CuPy

This module implements the BaSiC algorithm for illumination correction
using CuPy for GPU acceleration. The implementation is based on the paper:
Peng et al., "A BaSiC tool for background and shading correction of optical
microscopy images", Nature Communications, 2017.

The algorithm performs low-rank + sparse matrix decomposition to separate
uneven illumination artifacts from structural features in microscopy images.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy: All functions are pure and stateless
- Clause 65 — Fail Loudly: No silent fallbacks or inferred capabilities
- Clause 88 — No Inferred Capabilities: Explicit CuPy dependency
- Clause 273 — Memory Backend Restrictions: GPU-only implementation
"""

import logging
from typing import TYPE_CHECKING, Any

# Import decorator directly from core.memory to avoid circular imports
from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.utils import optional_import

# For type checking only
if TYPE_CHECKING:
    import cupy as cp
    from cupyx.scipy import linalg

# Import CuPy as an optional dependency
cp = optional_import("cupy")
linalg = None
if cp is not None:
    cupyx_scipy = optional_import("cupyx.scipy")
    if cupyx_scipy is not None:
        linalg = cupyx_scipy.linalg

logger = logging.getLogger(__name__)


def _validate_cupy_array(array: Any, name: str = "input") -> None:
    """
    Validate that the input is a CuPy array.

    Args:
        array: Array to validate
        name: Name of the array for error messages

    Raises:
        ImportError: If CuPy is not available
        TypeError: If the array is not a CuPy array
        ValueError: If the array doesn't support DLPack
    """
    # The compiler will ensure this function is only called when CuPy is available
    # No need to check for CuPy availability here

    if not isinstance(array, cp.ndarray):
        raise TypeError(
            f"{name} must be a CuPy array, got {type(array)}. "
            f"No automatic conversion is performed to maintain explicit contracts. "
            f"Use DLPack for zero-copy GPU-to-GPU transfers."
        )

    # Ensure the array supports DLPack
    if not hasattr(array, "__dlpack__") and not hasattr(array, "toDlpack"):
        raise ValueError(
            f"{name} does not support DLPack protocol. "
            f"DLPack is required for GPU memory conversions."
        )


def _low_rank_approximation(matrix: "cp.ndarray", rank: int = 3) -> "cp.ndarray":
    """
    Compute a low-rank approximation of a matrix using truncated SVD.

    Args:
        matrix: Input matrix to approximate
        rank: Target rank for the approximation

    Returns:
        Low-rank approximation of the input matrix
    """
    # Perform SVD
    U, s, Vh = linalg.svd(matrix, full_matrices=False)

    # Truncate to the specified rank
    s[rank:] = 0

    # Reconstruct the low-rank matrix
    low_rank = (U * s) @ Vh

    return low_rank


def _soft_threshold(matrix: "cp.ndarray", threshold: float) -> "cp.ndarray":
    """
    Apply soft thresholding (shrinkage operator) to a matrix.

    Args:
        matrix: Input matrix
        threshold: Threshold value for soft thresholding

    Returns:
        Soft-thresholded matrix
    """
    return cp.sign(matrix) * cp.maximum(cp.abs(matrix) - threshold, 0)


@cupy_func
def basic_flatfield_correction_cupy(
    image: "cp.ndarray",
    *,
    max_iters: int = 50,
    lambda_sparse: float = 0.01,
    lambda_lowrank: float = 0.1,
    rank: int = 3,
    tol: float = 1e-4,
    correction_mode: str = "divide",
    normalize_output: bool = True,
    verbose: bool = False,
    **kwargs
) -> "cp.ndarray":
    """
    Perform BaSiC-style illumination correction on a 3D image stack using CuPy.

    This function implements the BaSiC algorithm for illumination correction
    using low-rank + sparse matrix decomposition. It models the background
    (shading field) as a low-rank matrix across slices and the residuals
    (e.g., nuclei, structures) as sparse features.

    Args:
        image: 3D CuPy array of shape (Z, Y, X)
        max_iters: Maximum number of iterations for the alternating minimization
        lambda_sparse: Regularization parameter for the sparse component
        lambda_lowrank: Regularization parameter for the low-rank component
        rank: Target rank for the low-rank approximation
        tol: Tolerance for convergence
        correction_mode: Mode for applying the correction ('divide' or 'subtract')
        normalize_output: Whether to normalize the output to preserve dynamic range
        verbose: Whether to print progress information
        **kwargs: Additional parameters (ignored)

    Returns:
        Corrected 3D CuPy array of shape (Z, Y, X)

    Raises:
        ImportError: If CuPy is not available
        TypeError: If the input is not a CuPy array
        ValueError: If the input is not a 3D array or if correction_mode is invalid
                   or if the input array doesn't support DLPack
    """
    # Validate input
    _validate_cupy_array(image)

    if image.ndim != 3:
        raise ValueError(f"Input must be a 3D array, got {image.ndim}D")

    if correction_mode not in ["divide", "subtract"]:
        raise ValueError(f"Invalid correction mode: {correction_mode}. "
                        f"Must be 'divide' or 'subtract'")

    # Store original shape and dtype
    z, y, x = image.shape
    orig_dtype = image.dtype

    # Convert to float for processing
    image_float = image.astype(cp.float32)

    # Flatten each Z-slice into a row vector
    # D has shape (Z, Y*X)
    D = image_float.reshape(z, y * x)

    # Initialize variables for alternating minimization
    L = cp.zeros_like(D)  # Low-rank component (background/illumination)
    S = cp.zeros_like(D)  # Sparse component (foreground/structures)

    # Compute initial norm for convergence check
    norm_D = cp.linalg.norm(D, 'fro')

    # Alternating minimization loop
    for iteration in range(max_iters):
        # Update low-rank component (L)
        L = _low_rank_approximation(D - S, rank=rank)

        # Apply regularization to L if needed
        if lambda_lowrank > 0:
            L = L * (1 - lambda_lowrank)

        # Update sparse component (S)
        S = _soft_threshold(D - L, lambda_sparse)

        # Check convergence
        residual = cp.linalg.norm(D - L - S, 'fro') / norm_D
        if verbose and (iteration % 10 == 0 or iteration == max_iters - 1):
            logger.info(f"Iteration {iteration+1}/{max_iters}, residual: {residual:.6f}")

        if residual < tol:
            if verbose:
                logger.info(f"Converged after {iteration+1} iterations")
            break

    # Reshape the low-rank component back to 3D
    L_stack = L.reshape(z, y, x)

    # Apply correction
    if correction_mode == "divide":
        # Add small epsilon to avoid division by zero
        eps = 1e-6
        corrected = image_float / (L_stack + eps)

        # Normalize to preserve dynamic range
        if normalize_output:
            corrected *= cp.mean(L_stack)
    else:  # subtract
        corrected = image_float - L_stack

        # Normalize to preserve dynamic range
        if normalize_output:
            corrected += cp.mean(L_stack)

    # Clip to valid range and convert back to original dtype
    if cp.issubdtype(orig_dtype, cp.integer):
        max_val = cp.iinfo(orig_dtype).max
        corrected = cp.clip(corrected, 0, max_val).astype(orig_dtype)
    else:
        corrected = cp.clip(corrected, 0, None).astype(orig_dtype)

    return corrected


def basic_flatfield_correction_batch_cupy(
    image_batch: "cp.ndarray",
    *,
    batch_dim: int = 0,
    **kwargs
) -> "cp.ndarray":
    """
    Apply BaSiC flatfield correction to a batch of 3D image stacks.

    This function applies the BaSiC algorithm to each 3D stack in a batch.

    Args:
        image_batch: 4D CuPy array of shape (B, Z, Y, X) or (Z, B, Y, X)
        batch_dim: Dimension along which the batch is organized (0 or 1)
        **kwargs: Additional parameters passed to basic_flatfield_correction_cupy

    Returns:
        Corrected 4D CuPy array of the same shape as input

    Raises:
        ImportError: If CuPy is not available
        TypeError: If the input is not a CuPy array
        ValueError: If the input is not a 4D array or if batch_dim is invalid
                   or if the input array doesn't support DLPack
    """
    # Validate input
    _validate_cupy_array(image_batch)

    if image_batch.ndim != 4:
        raise ValueError(f"Input must be a 4D array, got {image_batch.ndim}D")

    if batch_dim not in [0, 1]:
        raise ValueError(f"batch_dim must be 0 or 1, got {batch_dim}")

    # Process each 3D stack in the batch
    result_list = []

    if batch_dim == 0:
        # Batch is organized as (B, Z, Y, X)
        for b in range(image_batch.shape[0]):
            corrected = basic_flatfield_correction_cupy(image_batch[b], **kwargs)
            result_list.append(corrected)

        # Stack along batch dimension
        return cp.stack(result_list, axis=0)

    # Batch is organized as (Z, B, Y, X)
    for b in range(image_batch.shape[1]):
        corrected = basic_flatfield_correction_cupy(image_batch[:, b], **kwargs)
        result_list.append(corrected)

    # Stack along batch dimension
    return cp.stack(result_list, axis=1)
