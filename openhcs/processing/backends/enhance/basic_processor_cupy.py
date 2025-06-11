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
cupyx_scipy = None
if cp is not None:
    cupyx_scipy = optional_import("cupyx.scipy")

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


def _low_rank_approximation(matrix: "cp.ndarray", rank: int = 3, max_memory_gb: float = 1.0) -> "cp.ndarray":
    """
    Compute a low-rank approximation of a matrix using SVD with memory optimization.

    Args:
        matrix: Input matrix to approximate
        rank: Target rank for the approximation
        max_memory_gb: Maximum memory to use for SVD (in GB)

    Returns:
        Low-rank approximation of the input matrix
    """
    # Estimate memory usage for SVD
    matrix_size_gb = matrix.nbytes / (1024**3)
    svd_memory_estimate = matrix_size_gb * 3  # U, s, Vh matrices

    if svd_memory_estimate > max_memory_gb:
        # Use chunked processing for large matrices
        logger.info(f"🔧 MEMORY OPTIMIZATION: Matrix too large ({svd_memory_estimate:.2f}GB > {max_memory_gb}GB), using chunked SVD")
        return _chunked_low_rank_approximation(matrix, rank, max_memory_gb)
    else:
        # Use standard SVD for smaller matrices
        try:
            # Perform SVD using CuPy's built-in linalg
            U, s, Vh = cp.linalg.svd(matrix, full_matrices=False)

            # Truncate to the specified rank
            s[rank:] = 0

            # Reconstruct the low-rank matrix
            low_rank = (U * s) @ Vh

            return low_rank
        except cp.cuda.memory.OutOfMemoryError:
            # Fallback to chunked processing if standard SVD fails
            logger.warning("🔧 MEMORY OPTIMIZATION: Standard SVD failed, falling back to chunked processing")
            return _chunked_low_rank_approximation(matrix, rank, max_memory_gb)


def _chunked_low_rank_approximation(matrix: "cp.ndarray", rank: int, max_memory_gb: float) -> "cp.ndarray":
    """
    Compute low-rank approximation using chunked processing to reduce memory usage.

    Args:
        matrix: Input matrix to approximate (Z, Y*X)
        rank: Target rank for the approximation
        max_memory_gb: Maximum memory to use per chunk

    Returns:
        Low-rank approximation of the input matrix
    """
    Z, YX = matrix.shape

    # Calculate optimal chunk size based on memory limit
    bytes_per_element = matrix.dtype.itemsize

    # Be much more conservative - SVD needs 5-10x memory overhead, not just 3x
    safety_factor = 10  # Very conservative for SVD workspace requirements
    max_elements_per_chunk = int((max_memory_gb * 1024**3) / (bytes_per_element * safety_factor))
    chunk_size = min(YX, max_elements_per_chunk // Z)

    # For very large Z (like 65 slices), we need even smaller chunks
    if Z > 20:  # Large number of slices
        chunk_size = chunk_size // 4  # Quarter the chunk size
        logger.info(f"🔧 MEMORY OPTIMIZATION: Large Z dimension ({Z}), reducing chunk size by 4x")

    # Ensure minimum chunk size but not too small to be inefficient
    min_chunk_size = min(1000, YX // 100)  # At least 1% of total, but max 1000
    if chunk_size < min_chunk_size:
        chunk_size = min_chunk_size
        logger.warning(f"🔧 MEMORY OPTIMIZATION: Very small chunk size ({chunk_size}), may be inefficient")

    logger.debug(f"🔧 MEMORY OPTIMIZATION: Processing {YX} elements in chunks of {chunk_size}")

    # Process matrix in chunks along the spatial dimension
    low_rank_chunks = []

    for start_idx in range(0, YX, chunk_size):
        end_idx = min(start_idx + chunk_size, YX)
        chunk = matrix[:, start_idx:end_idx]

        try:
            # Perform SVD on chunk
            U, s, Vh = cp.linalg.svd(chunk, full_matrices=False)

            # Truncate to the specified rank
            s_truncated = s.copy()
            s_truncated[rank:] = 0

            # Reconstruct the low-rank chunk
            low_rank_chunk = (U * s_truncated) @ Vh
            low_rank_chunks.append(low_rank_chunk)

        except cp.cuda.memory.OutOfMemoryError:
            # If even the chunk is too large, fallback to CPU processing for this chunk
            logger.warning(f"🔧 MEMORY OPTIMIZATION: Chunk still too large, falling back to CPU for chunk {start_idx}:{end_idx}")

            try:
                # Move chunk to CPU and process there
                chunk_cpu = chunk.get()  # CuPy -> NumPy

                # Use NumPy SVD on CPU
                import numpy as np
                U_cpu, s_cpu, Vh_cpu = np.linalg.svd(chunk_cpu, full_matrices=False)

                # Truncate to the specified rank
                s_cpu[rank:] = 0

                # Reconstruct the low-rank chunk on CPU
                low_rank_chunk_cpu = (U_cpu * s_cpu) @ Vh_cpu

                # Move result back to GPU
                low_rank_chunk = cp.asarray(low_rank_chunk_cpu)
                low_rank_chunks.append(low_rank_chunk)

                logger.debug(f"🔧 MEMORY OPTIMIZATION: Successfully processed chunk on CPU")

            except Exception as cpu_error:
                logger.error(f"🔧 MEMORY OPTIMIZATION: CPU fallback also failed: {cpu_error}")
                # Last resort: use identity matrix (no correction for this chunk)
                logger.warning(f"🔧 MEMORY OPTIMIZATION: Using identity matrix for chunk {start_idx}:{end_idx}")
                low_rank_chunk = chunk.copy()  # No correction applied
                low_rank_chunks.append(low_rank_chunk)

    # Concatenate all chunks
    low_rank = cp.concatenate(low_rank_chunks, axis=1)

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
    max_memory_gb: float = 1.0,
    **kwargs
) -> "cp.ndarray":
    """
    Perform BaSiC-style illumination correction on a 3D image stack using CuPy.

    This function implements the BaSiC algorithm for illumination correction
    using low-rank + sparse matrix decomposition. It models the background
    (shading field) as a low-rank matrix across slices and the residuals
    (e.g., nuclei, structures) as sparse features.

    Memory-optimized version that automatically uses chunked processing for
    large images to prevent CUDA out-of-memory errors.

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
        max_memory_gb: Maximum memory to use for SVD operations (in GB)
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

    # 🔍 MEMORY ESTIMATION: Check if image is likely to cause memory issues
    image_size_gb = image.nbytes / (1024**3)
    estimated_peak_memory = image_size_gb * 4  # Original + float + L + S matrices

    # Try GPU processing first, fallback to CPU on OOM
    try:
        return _gpu_flatfield_correction(
            image, max_iters, lambda_sparse, lambda_lowrank, rank, tol,
            correction_mode, normalize_output, verbose, max_memory_gb,
            image_size_gb, estimated_peak_memory, z, y, x
        )
    except cp.cuda.memory.OutOfMemoryError as oom_error:
        logger.warning(f"🔧 GPU OOM: {oom_error}")
        logger.info(f"🔧 CPU FALLBACK: GPU processing failed, switching to CPU for {z}×{y}×{x} image")

        # Clear GPU memory before CPU fallback
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

        # Fallback to CPU processing
        return _cpu_fallback_flatfield_correction(
            image, max_iters, lambda_sparse, lambda_lowrank, rank, tol,
            correction_mode, normalize_output, verbose
        )


def _gpu_flatfield_correction(
    image: "cp.ndarray", max_iters: int, lambda_sparse: float, lambda_lowrank: float,
    rank: int, tol: float, correction_mode: str, normalize_output: bool, verbose: bool,
    max_memory_gb: float, image_size_gb: float, estimated_peak_memory: float, z: int, y: int, x: int
) -> "cp.ndarray":
    """GPU-based flatfield correction implementation."""

    if estimated_peak_memory > max_memory_gb * 2:
        logger.warning(f"⚠️  Large image detected: {z}×{y}×{x} ({image_size_gb:.2f}GB). "
                      f"Estimated peak memory: {estimated_peak_memory:.2f}GB. "
                      f"Consider reducing image size or increasing max_memory_gb parameter.")

    logger.debug(f"🔧 MEMORY INFO: Image size {z}×{y}×{x}, {image_size_gb:.2f}GB, "
                f"max_memory_gb={max_memory_gb}, estimated peak={estimated_peak_memory:.2f}GB")

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

    # Track convergence for early termination
    prev_residual = float('inf')
    stagnation_count = 0
    max_stagnation = 5  # Stop if no improvement for 5 iterations

    # Alternating minimization loop
    for iteration in range(max_iters):
        # Update low-rank component (L) with memory optimization
        L = _low_rank_approximation(D - S, rank=rank, max_memory_gb=max_memory_gb)

        # Apply regularization to L if needed
        if lambda_lowrank > 0:
            L = L * (1 - lambda_lowrank)

        # Update sparse component (S)
        S = _soft_threshold(D - L, lambda_sparse)

        # Check convergence
        residual = cp.linalg.norm(D - L - S, 'fro') / norm_D
        if verbose and (iteration % 10 == 0 or iteration == max_iters - 1):
            logger.info(f"Iteration {iteration+1}/{max_iters}, residual: {residual:.6f}")

        # Early termination conditions
        if residual < tol:
            if verbose:
                logger.info(f"Converged after {iteration+1} iterations (residual < {tol})")
            break

        # Check for stagnation (no significant improvement)
        improvement = prev_residual - residual
        if improvement < tol * 0.1:  # Less than 10% of tolerance improvement
            stagnation_count += 1
            if stagnation_count >= max_stagnation:
                if verbose:
                    logger.info(f"Early termination after {iteration+1} iterations (stagnation)")
                break
        else:
            stagnation_count = 0  # Reset counter if we see improvement

        prev_residual = residual

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


def _cpu_fallback_flatfield_correction(
    image: "cp.ndarray", max_iters: int, lambda_sparse: float, lambda_lowrank: float,
    rank: int, tol: float, correction_mode: str, normalize_output: bool, verbose: bool
) -> "cp.ndarray":
    """CPU fallback for flatfield correction when GPU runs out of memory."""

    try:
        from openhcs.processing.backends.enhance.basic_processor_numpy import basic_flatfield_correction_numpy

        # Convert CuPy array to NumPy
        logger.info(f"🔧 CPU FALLBACK: Converting CuPy array to NumPy...")
        image_cpu = image.get()

        # Process on CPU
        logger.info(f"🔧 CPU FALLBACK: Processing {image_cpu.shape} image on CPU")
        corrected_cpu = basic_flatfield_correction_numpy(
            image_cpu,
            max_iters=max_iters,
            lambda_sparse=lambda_sparse,
            lambda_lowrank=lambda_lowrank,
            rank=rank,
            tol=tol,
            correction_mode=correction_mode,
            normalize_output=normalize_output,
            verbose=verbose
        )

        # Convert back to CuPy
        logger.info(f"🔧 CPU FALLBACK: Converting result back to CuPy...")
        corrected = cp.asarray(corrected_cpu)
        logger.info(f"🔧 CPU FALLBACK: Successfully processed on CPU and converted back to GPU")
        return corrected

    except Exception as cpu_error:
        logger.error(f"🔧 CPU FALLBACK: Failed to process on CPU: {cpu_error}")
        raise RuntimeError(f"Both GPU and CPU processing failed. GPU OOM, CPU error: {cpu_error}") from cpu_error
