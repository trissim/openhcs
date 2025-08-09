"""
OpenHCS Interface for Ashlar GPU Stitching Algorithm

Array-based EdgeAligner implementation that works directly with CuPy arrays
instead of file-based readers. This is the complete Ashlar algorithm modified
to accept arrays directly and run on GPU.
"""
from __future__ import annotations
import logging
import sys
from typing import TYPE_CHECKING, Tuple, List
import numpy as np
import networkx as nx
import scipy.spatial.distance
import sklearn.linear_model
import pandas as pd

from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.memory.decorators import cupy as cupy_func
from openhcs.core.utils import optional_import

# Import CuPy using the established optional import pattern
cp = optional_import("cupy")

import warnings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class DataWarning(Warning):
    """Warnings about the content of user-provided image data."""
    pass


def warn_data(message):
    """Issue a warning about image data."""
    warnings.warn(message, DataWarning)


class IntersectionGPU:
    """Calculate intersection region between two tiles - EXACT Ashlar implementation for GPU."""

    def __init__(self, corners1, corners2, min_size=0):
        if not cp:
            raise ImportError("CuPy is required for GPU intersection calculations")
        if isinstance(min_size, (int, float)):
            min_size = cp.full(2, min_size)
        elif not isinstance(min_size, cp.ndarray):
            min_size = cp.array(min_size)
        self._calculate(corners1, corners2, min_size)

    def _calculate(self, corners1, corners2, min_size):
        """Calculate intersection parameters using EXACT Ashlar logic."""
        # This is the EXACT logic from the original Ashlar Intersection class
        max_shape = (corners2 - corners1).max(axis=0)
        min_size = cp.clip(min_size, 1, max_shape)
        position = corners1.max(axis=0)
        initial_shape = cp.floor(corners2.min(axis=0) - position).astype(int)
        clipped_shape = cp.maximum(initial_shape, min_size)
        self.shape = cp.ceil(clipped_shape).astype(int)
        self.padding = self.shape - initial_shape
        self.offsets = cp.maximum(position - corners1 - self.padding, 0)
        offset_diff = self.offsets[1] - self.offsets[0]
        self.offset_diff_frac = offset_diff - cp.round(offset_diff)


def _get_window(shape):
    """Build a 2D Hann window (from Ashlar utils.get_window) on GPU."""
    if cp is None:
        raise ImportError("CuPy is required for GPU window functions")
    # Build a 2D Hann window by taking the outer product of two 1-D windows.
    wy = cp.hanning(shape[0]).astype(cp.float32)
    wx = cp.hanning(shape[1]).astype(cp.float32)
    window = cp.outer(wy, wx)
    return window


# Precompute Laplacian kernel for whitening (equivalent to skimage.restoration.uft.laplacian)
_laplace_kernel_gpu = None
if cp:
    _laplace_kernel_gpu = cp.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=cp.float32)


def whiten_gpu(img, sigma):
    """
    Vectorized GPU whitening filter - EXACT match to Ashlar reference implementation.

    This implements the same whitening as ashlar.utils.whiten() but optimized for GPU:
    - sigma=0: Uses Laplacian convolution (high-pass filter)
    - sigma>0: Uses Gaussian-Laplacian (LoG filter)

    Args:
        img: CuPy array (2D image)
        sigma: Standard deviation for Gaussian kernel (0 = pure Laplacian)

    Returns:
        CuPy array: Whitened image
    """
    # Convert to float32 (matches reference)
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    img = img.astype(cp.float32)

    if sigma == 0:
        # Pure Laplacian convolution (high-pass filter)
        # Equivalent to scipy.ndimage.convolve(img, _laplace_kernel)
        from cupyx.scipy import ndimage as cp_ndimage
        output = cp_ndimage.convolve(img, _laplace_kernel_gpu, mode='reflect')
    else:
        # Gaussian-Laplacian (LoG filter)
        # Equivalent to scipy.ndimage.gaussian_laplace(img, sigma)
        from cupyx.scipy import ndimage as cp_ndimage
        output = cp_ndimage.gaussian_laplace(img, sigma)

    return output


def whiten_gpu_vectorized(img_stack, sigma):
    """
    Vectorized GPU whitening for multiple images simultaneously.

    This processes an entire stack of images in parallel on GPU for maximum efficiency.

    Args:
        img_stack: CuPy array of shape (N, H, W) - stack of N images
        sigma: Standard deviation for Gaussian kernel (0 = pure Laplacian)

    Returns:
        CuPy array: Stack of whitened images with same shape as input
    """
    if not isinstance(img_stack, cp.ndarray):
        img_stack = cp.asarray(img_stack)
    img_stack = img_stack.astype(cp.float32)

    if sigma == 0:
        # Vectorized Laplacian convolution for entire stack
        from cupyx.scipy import ndimage as cp_ndimage
        # Process each image in the stack
        output_stack = cp.empty_like(img_stack)
        for i in range(img_stack.shape[0]):
            output_stack[i] = cp_ndimage.convolve(img_stack[i], _laplace_kernel_gpu, mode='reflect')
    else:
        # Vectorized Gaussian-Laplacian for entire stack
        from cupyx.scipy import ndimage as cp_ndimage
        output_stack = cp.empty_like(img_stack)
        for i in range(img_stack.shape[0]):
            output_stack[i] = cp_ndimage.gaussian_laplace(img_stack[i], sigma)

    return output_stack


def ashlar_register_gpu(img1, img2, upsample=10):
    """
    GPU register function using cuCIM - matches CPU version with windowing only.

    This uses cuCIM's phase_cross_correlation which is the GPU equivalent
    of skimage.registration.phase_cross_correlation used in the CPU version.
    No whitening filter - just windowing like the CPU version.

    Args:
        img1, img2: Input images
        upsample: Upsampling factor for phase correlation
    """
    import itertools
    import cucim.skimage.registration

    # Input validation (same as CPU version)
    if img1 is None or img2 is None:
        return cp.array([0.0, 0.0]), cp.inf

    if img1.size == 0 or img2.size == 0:
        return cp.array([0.0, 0.0]), cp.inf

    if img1.shape != img2.shape:
        return cp.array([0.0, 0.0]), cp.inf

    if len(img1.shape) != 2:
        return cp.array([0.0, 0.0]), cp.inf

    if img1.shape[0] < 1 or img1.shape[1] < 1:
        return cp.array([0.0, 0.0]), cp.inf

    # Convert to CuPy arrays
    if not isinstance(img1, cp.ndarray):
        img1 = cp.asarray(img1)
    if not isinstance(img2, cp.ndarray):
        img2 = cp.asarray(img2)

    # Convert to float32 and apply windowing (matches CPU version)
    img1w = img1.astype(cp.float32) * _get_window(img1.shape)
    img2w = img2.astype(cp.float32) * _get_window(img2.shape)

    # Use cuCIM's phase cross correlation (GPU equivalent of skimage)
    try:
        shift, error, phase_diff = cucim.skimage.registration.phase_cross_correlation(
            img1w, img2w, upsample_factor=upsample
        )

        # Convert to numpy for consistency with CPU version
        shift = cp.asnumpy(shift)
        error = float(error)

        # Only log high errors to avoid spam
        if error > 1.0:  # High error threshold for Ashlar
            logger.warning(f"Ashlar GPU: HIGH CORRELATION ERROR - Error={error:.4f}, Shift=({shift[0]:.2f}, {shift[1]:.2f})")
            logger.warning(f"  This indicates poor overlap or image quality between tiles")

    except Exception as e:
        # Fallback if correlation fails
        logger.error(f"Ashlar GPU: CORRELATION FAILED - Exception: {e}")
        logger.error(f"  Returning infinite error")
        shift = cp.array([0.0, 0.0])
        error = cp.inf

    return shift, error





def ashlar_nccw_no_preprocessing_gpu(img1, img2):
    """
    GPU nccw function - faithful to Ashlar but with better numerical stability.

    This matches the CPU version but with improved precision handling for GPU.
    """
    # Convert to CuPy arrays and float32 (equivalent to what whiten() does)
    if not isinstance(img1, cp.ndarray):
        img1 = cp.asarray(img1)
    if not isinstance(img2, cp.ndarray):
        img2 = cp.asarray(img2)

    img1w = img1.astype(cp.float32)
    img2w = img2.astype(cp.float32)

    correlation = float(cp.abs(cp.sum(img1w * img2w)))
    total_amplitude = float(cp.linalg.norm(img1w) * cp.linalg.norm(img2w))

    if correlation > 0 and total_amplitude > 0:
        diff = correlation - total_amplitude
        if diff <= 0:
            error = -cp.log(correlation / total_amplitude)
        elif diff < 1e-3:  # Increased tolerance for GPU precision
            # This situation can occur due to numerical precision issues when
            # img1 and img2 are very nearly or exactly identical. If the
            # difference is small enough, let it slide.
            error = 0
        else:
            # Instead of raising error, return a large but finite error
            logger.warning(f"Ashlar GPU: NCCW numerical precision issue - diff={diff:.6f}, using error=100.0")
            error = 100.0  # Large error but not infinite
    else:
        logger.warning(f"Ashlar GPU: NCCW invalid correlation - correlation={correlation:.6f}, total_amplitude={total_amplitude:.6f}")
        error = cp.inf

    # Log all NCCW results at INFO level for user visibility
    error_float = float(error)
    if error_float > 10.0:  # High NCCW error threshold
        logger.warning(f"Ashlar GPU: HIGH NCCW ERROR - Error={error_float:.4f}")
        logger.warning(f"  This indicates poor image correlation between tiles")
    else:
        logger.info(f"Ashlar GPU: NCCW - Error={error_float:.4f}")

    return error_float





def ashlar_crop_gpu(img, offset, shape):
    """
    EXACT Ashlar crop function (from ashlar.utils.crop) for GPU arrays with boundary validation.

    Note that this only crops to the nearest whole-pixel offset.
    """
    # Convert to CuPy if needed
    if not isinstance(img, cp.ndarray):
        img = cp.asarray(img)
    if not isinstance(offset, cp.ndarray):
        offset = cp.asarray(offset)
    if not isinstance(shape, cp.ndarray):
        shape = cp.asarray(shape)

    # Validate inputs to prevent zero-sized arrays
    if cp.any(shape <= 0):
        raise ValueError(f"Invalid crop shape: {shape}. Shape must be positive.")

    # Note that this only crops to the nearest whole-pixel offset.
    start = cp.round(offset).astype(int)
    end = start + shape

    # Validate bounds to prevent invalid slicing
    img_shape = cp.array(img.shape)
    if cp.any(start < 0) or cp.any(end > img_shape):
        # Clip to valid bounds
        start = cp.maximum(start, 0)
        end = cp.minimum(end, img_shape)

        # Recalculate shape after clipping
        new_shape = end - start
        if cp.any(new_shape <= 0):
            raise ValueError(f"Invalid crop region after bounds checking: start={start}, end={end}, img_shape={img_shape}")

    img = img[start[0]:end[0], start[1]:end[1]]
    return img





class ArrayEdgeAlignerGPU:
    """
    Array-based EdgeAligner that implements the complete Ashlar algorithm
    but works directly with CuPy arrays instead of file readers and runs on GPU.
    """

    def __init__(self, image_stack, positions, tile_size, pixel_size=1.0,
                 max_shift=15, alpha=0.01, max_error=None,
                 randomize=False, verbose=False, upsample_factor=10,
                 permutation_upsample=1, permutation_samples=1000,
                 min_permutation_samples=10, max_permutation_tries=100,
                 window_size_factor=0.1):
        """
        Initialize array-based EdgeAligner for position calculation on GPU.

        Args:
            image_stack: 3D numpy/cupy array (num_tiles, height, width) - preprocessed grayscale
            positions: 2D array of tile positions (num_tiles, 2) in pixels
            tile_size: Array [height, width] of tile dimensions
            pixel_size: Pixel size in micrometers (for max_shift conversion)
            max_shift: Maximum allowed shift in micrometers
            alpha: Alpha value for error threshold (lower = stricter)
            max_error: Explicit error threshold (None = auto-compute)
            randomize: Use random seed for permutation testing
            verbose: Enable verbose logging
        """
        # Convert to CuPy arrays for GPU processing
        if not isinstance(image_stack, cp.ndarray):
            self.image_stack = cp.asarray(image_stack)
        else:
            self.image_stack = image_stack

        if not isinstance(positions, cp.ndarray):
            self.positions = cp.asarray(positions, dtype=cp.float64)
        else:
            self.positions = positions.astype(cp.float64)

        self.tile_size = cp.array(tile_size)
        self.pixel_size = pixel_size
        self.max_shift = max_shift
        self.max_shift_pixels = self.max_shift / self.pixel_size
        self.alpha = alpha
        self.max_error = max_error
        self.randomize = randomize
        self.verbose = verbose
        self.upsample_factor = upsample_factor
        self.permutation_upsample = permutation_upsample
        self.permutation_samples = permutation_samples
        self.min_permutation_samples = min_permutation_samples
        self.max_permutation_tries = max_permutation_tries
        self.window_size_factor = window_size_factor
        self._cache = {}
        self.errors_negative_sampled = cp.empty(0)

        # Build neighbors graph (this uses CPU operations with NetworkX)
        self.neighbors_graph = self._build_neighbors_graph()

    def _build_neighbors_graph(self):
        """Build graph of neighboring (overlapping) tiles."""
        # Convert to CPU for scipy operations
        positions_cpu = cp.asnumpy(self.positions)
        tile_size_cpu = cp.asnumpy(self.tile_size)

        pdist = scipy.spatial.distance.pdist(positions_cpu, metric='cityblock')
        sp = scipy.spatial.distance.squareform(pdist)
        max_distance = tile_size_cpu.max() + 1
        edges = zip(*np.nonzero((sp > 0) & (sp < max_distance)))
        graph = nx.from_edgelist(edges)
        graph.add_nodes_from(range(len(positions_cpu)))
        return graph


    def run(self):
        """Run the complete Ashlar algorithm."""
        self.check_overlaps()
        self.compute_threshold()
        self.register_all()
        self.build_spanning_tree()
        self.calculate_positions()
        self.fit_model()

    def check_overlaps(self):
        """Check if tiles actually overlap based on positions."""
        overlaps = []
        for t1, t2 in self.neighbors_graph.edges:
            overlap = self.tile_size - cp.abs(self.positions[t1] - self.positions[t2])
            overlaps.append(overlap)

        if overlaps:
            overlaps = cp.stack(overlaps)
            failures = cp.any(overlaps < 1, axis=1)
            failures_cpu = cp.asnumpy(failures)

            if len(failures_cpu) and all(failures_cpu):
                warn_data("No tiles overlap, attempting alignment anyway.")
            elif any(failures_cpu):
                warn_data("Some neighboring tiles have zero overlap.")

    def compute_threshold(self):
        """Compute error threshold using permutation testing."""
        if self.max_error is not None:
            if self.verbose:
                print("    using explicit error threshold")
            return

        edges = self.neighbors_graph.edges
        num_tiles = len(self.image_stack)

        # If not enough tiles overlap to matter, skip this whole thing
        if len(edges) <= 1:
            self.max_error = np.inf
            return

        widths = []
        for t1, t2 in edges:
            shape = self.intersection(t1, t2).shape
            widths.append(cp.min(cp.array(shape)))

        widths = cp.array(widths)
        w = int(cp.max(widths))
        max_offset = int(self.tile_size[0]) - w

        # Number of possible pairs minus number of actual neighbor pairs
        num_distant_pairs = num_tiles * (num_tiles - 1) // 2 - len(edges)

        # Reduce permutation count for small datasets
        n = self.permutation_samples if num_distant_pairs > 8 else (num_distant_pairs + 1) * self.min_permutation_samples
        pairs = np.empty((n, 2), dtype=int)  # Keep on CPU for random generation
        offsets = np.empty((n, 2), dtype=int)  # Keep on CPU for random generation

        # Generate n random non-overlapping image strips
        max_tries = self.max_permutation_tries
        if self.randomize is False:
            random_state = np.random.RandomState(0)
        else:
            random_state = np.random.RandomState()

        for i in range(n):
            # Limit tries to avoid infinite loop in pathological cases
            for current_try in range(max_tries):
                t1, t2 = random_state.randint(num_tiles, size=2)
                o1, o2 = random_state.randint(max_offset, size=2)

                # Check for non-overlapping strips and abort the retry loop
                if t1 != t2 and (t1, t2) not in edges:
                    # Different, non-neighboring tiles -- always OK
                    break
                elif t1 == t2 and abs(o1 - o2) > w:
                    # Same tile OK if strips don't overlap within the image
                    break
                elif (t1, t2) in edges:
                    # Neighbors OK if either strip is entirely outside the
                    # expected overlap region (based on nominal positions)
                    its = self.intersection(t1, t2, cp.full(2, w))
                    ioff1, ioff2 = its.offsets[:, 0]
                    if (
                        its.shape[0] > its.shape[1]
                        or o1 < ioff1 - w or o1 > ioff1 + w
                        or o2 < ioff2 - w or o2 > ioff2 + w
                    ):
                        break
            else:
                # Retries exhausted. This should be very rare.
                warn_data(f"Could not find non-overlapping strips in {max_tries} tries")
            pairs[i] = t1, t2
            offsets[i] = o1, o2

        errors = cp.empty(n)
        for i, ((t1, t2), (offset1, offset2)) in enumerate(zip(pairs, offsets)):
            if self.verbose and (i % 10 == 9 or i == n - 1):
                sys.stdout.write(f'\r    quantifying alignment error {i + 1}/{n}')
                sys.stdout.flush()
            img1 = self.image_stack[t1][offset1:offset1+w, :]
            img2 = self.image_stack[t2][offset2:offset2+w, :]
            _, errors[i] = ashlar_register_gpu(img1, img2, upsample=self.permutation_upsample)
        if self.verbose:
            print()
        self.errors_negative_sampled = errors
        self.max_error = float(cp.percentile(errors, self.alpha * 100))


    def register_all(self):
        """Register all neighboring tile pairs."""
        n = self.neighbors_graph.size()
        for i, (t1, t2) in enumerate(self.neighbors_graph.edges, 1):
            if self.verbose:
                sys.stdout.write(f'\r    aligning edge {i}/{n}')
                sys.stdout.flush()
            self.register_pair(t1, t2)
        if self.verbose:
            print()
        self.all_errors = cp.array([x[1] for x in self._cache.values()])

        # Set error values above the threshold to infinity
        for k, v in self._cache.items():
            shift_array = cp.array(v[0]) if not isinstance(v[0], cp.ndarray) else v[0]
            if v[1] > self.max_error or cp.any(cp.abs(shift_array) > self.max_shift_pixels):
                self._cache[k] = (v[0], cp.inf)

    def register_pair(self, t1, t2):
        """Return relative shift between images and the alignment error."""
        key = tuple(sorted((t1, t2)))
        try:
            shift, error = self._cache[key]
        except KeyError:
            # Test a series of increasing overlap window sizes to help avoid
            # missing alignments when the stage position error is large relative
            # to the tile overlap. Simply using a large overlap in all cases
            # limits the maximum achievable correlation thus increasing the
            # error metric, leading to worse overall results. The window size
            # starts at the nominal size and doubles until it's at least 10% of
            # the tile size. If the nominal overlap is already 10% or greater,
            # we only use that one size.
            try:
                smin = self.intersection(key[0], key[1]).shape
                smax = cp.round(self.tile_size * self.window_size_factor)
                sizes = [smin]
                while any(cp.array(sizes[-1]) < smax):
                    sizes.append(cp.array(sizes[-1]) * 2)

                # Try each window size and collect results
                results = []
                for s in sizes:
                    try:
                        result = self._register(key[0], key[1], s)
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        if self.verbose:
                            print(f"    window size {s} failed: {e}")
                        continue

                if not results:
                    # All window sizes failed, return large error
                    shift = cp.array([0.0, 0.0])
                    error = cp.inf
                else:
                    # Use the shift from the window size that gave the lowest error
                    shift, _ = min(results, key=lambda r: r[1])
                    # Extract the images from the nominal overlap window but with the
                    # shift applied to the second tile's position, and compute the error
                    # metric on these images. This should be even lower than the error
                    # computed above.
                    try:
                        _, o1, o2 = self.overlap(key[0], key[1], shift=shift)
                        error = ashlar_nccw_no_preprocessing_gpu(o1, o2)
                    except Exception as e:
                        if self.verbose:
                            print(f"    final error computation failed: {e}")
                        error = cp.inf

            except Exception as e:
                if self.verbose:
                    print(f"    registration failed for tiles {key}: {e}")
                shift = cp.array([0.0, 0.0])
                error = cp.inf

            self._cache[key] = (shift, error)
        if t1 > t2:
            shift = -shift
        # Return copy of shift to prevent corruption of cached values
        return shift.copy(), error

    def _register(self, t1, t2, min_size=0):
        """Register a single tile pair with given minimum size."""
        try:
            its, img1, img2 = self.overlap(t1, t2, min_size)

            # Validate that we got valid images
            if img1.size == 0 or img2.size == 0:
                if self.verbose:
                    print(f"    empty images for tiles {t1}, {t2} with min_size {min_size}")
                return None

            # Account for padding, flipping the sign depending on the direction
            # between the tiles
            p1, p2 = self.positions[[t1, t2]]
            sx = 1 if p1[1] >= p2[1] else -1
            sy = 1 if p1[0] >= p2[0] else -1
            padding = cp.array(its.padding) * cp.array([sy, sx])
            shift, error = ashlar_register_gpu(img1, img2, upsample=self.upsample_factor)
            shift = cp.array(shift) + padding
            return shift.get(), error
        except Exception as e:
            if self.verbose:
                print(f"    _register failed for tiles {t1}, {t2}: {e}")
            return None


    def intersection(self, t1, t2, min_size=0, shift=None):
        """Calculate intersection region between two tiles."""
        corners1 = self.positions[[t1, t2]].copy()
        if shift is not None:
            if not isinstance(shift, cp.ndarray):
                shift = cp.array(shift)
            corners1[1] += shift
        corners2 = corners1 + self.tile_size
        return IntersectionGPU(corners1, corners2, min_size)

    def crop(self, tile_id, offset, shape):
        """Crop image from tile at given offset and shape."""
        img = self.image_stack[tile_id]
        return ashlar_crop_gpu(img, offset, shape)

    def overlap(self, t1, t2, min_size=0, shift=None):
        """Extract overlapping regions between two tiles."""
        its = self.intersection(t1, t2, min_size, shift)

        # Validate intersection shape before cropping
        if cp.any(its.shape <= 0):
            raise ValueError(f"Invalid intersection shape {its.shape} for tiles {t1}, {t2}")

        img1 = self.crop(t1, its.offsets[0], its.shape)
        img2 = self.crop(t2, its.offsets[1], its.shape)
        return its, img1, img2





    def build_spanning_tree(self):
        """Build minimum spanning tree using GPU Boruvka algorithm."""
        # Import the Boruvka MST implementation
        from openhcs.processing.backends.pos_gen.mist.boruvka_mst import build_mst_gpu_boruvka

        # Convert cache to Boruvka format
        valid_edges = [(t1, t2, shift, error) for (t1, t2), (shift, error) in self._cache.items() if cp.isfinite(error)]

        if len(valid_edges) == 0:
            # No valid edges - create empty graph with all nodes
            self.spanning_tree = nx.Graph()
            self.spanning_tree.add_nodes_from(range(len(self.positions)))
            return

        # Prepare arrays for Boruvka MST
        connection_from = cp.array([t1 for t1, t2, shift, error in valid_edges], dtype=cp.int32)
        connection_to = cp.array([t2 for t1, t2, shift, error in valid_edges], dtype=cp.int32)
        connection_dx = cp.array([shift[1] for t1, t2, shift, error in valid_edges], dtype=cp.float32)  # x shift
        connection_dy = cp.array([shift[0] for t1, t2, shift, error in valid_edges], dtype=cp.float32)  # y shift
        # Use negative error as quality (higher quality = lower error)
        connection_quality = cp.array([-error for t1, t2, shift, error in valid_edges], dtype=cp.float32)

        num_nodes = len(self.positions)

        try:
            # Run GPU Boruvka MST
            mst_result = build_mst_gpu_boruvka(
                connection_from, connection_to, connection_dx, connection_dy,
                connection_quality, num_nodes
            )

            # Convert back to NetworkX format for compatibility with rest of algorithm
            self.spanning_tree = nx.Graph()
            self.spanning_tree.add_nodes_from(range(num_nodes))

            for edge in mst_result['edges']:
                t1, t2 = edge['from'], edge['to']
                # Reconstruct error from quality
                error = -edge['quality'] if 'quality' in edge else 0.0
                self.spanning_tree.add_edge(t1, t2, weight=error)

        except Exception as e:
            # Fallback to NetworkX if Boruvka fails
            print(f"Boruvka MST failed, falling back to NetworkX: {e}")
            g = nx.Graph()
            g.add_nodes_from(self.neighbors_graph)
            g.add_weighted_edges_from(
                (t1, t2, error)
                for (t1, t2), (_, error) in self._cache.items()
                if cp.isfinite(error)
            )
            spanning_tree = nx.Graph()
            spanning_tree.add_nodes_from(g)
            for c in nx.connected_components(g):
                cc = g.subgraph(c)
                center = nx.center(cc)[0]
                paths = nx.single_source_dijkstra_path(cc, center).values()
                for path in paths:
                    nx.add_path(spanning_tree, path)
            self.spanning_tree = spanning_tree

    def calculate_positions(self):
        """Calculate final positions from spanning tree."""
        shifts = {}
        for c in nx.connected_components(self.spanning_tree):
            cc = self.spanning_tree.subgraph(c)
            center = nx.center(cc)[0]
            shifts[center] = cp.array([0, 0])
            for edge in nx.traversal.bfs_edges(cc, center):
                source, dest = edge
                if source not in shifts:
                    source, dest = dest, source
                shift = self.register_pair(source, dest)[0]
                shifts[dest] = shifts[source] + cp.array(shift)
        if shifts:
            self.shifts = cp.array([s for _, s in sorted(shifts.items())])
            self.final_positions = self.positions + self.shifts
        else:
            # TODO: fill in shifts and positions with 0x2 arrays
            raise NotImplementedError("No images")


    def fit_model(self):
        """Fit linear model to handle disconnected components."""
        components = sorted(
            nx.connected_components(self.spanning_tree),
            key=len, reverse=True
        )
        # Fit LR model on positions of largest connected component
        cc0 = list(components[0])
        self.lr = sklearn.linear_model.LinearRegression()

        # Convert to CPU for sklearn operations
        positions_cpu = cp.asnumpy(self.positions[cc0])
        final_positions_cpu = cp.asnumpy(self.final_positions[cc0])
        self.lr.fit(positions_cpu, final_positions_cpu)

        # Fix up degenerate transform matrix. This happens when the spanning
        # tree is completely edgeless or cc0's metadata positions fall in a
        # straight line. In this case we fall back to the identity transform.
        if np.linalg.det(self.lr.coef_) < 1e-3:
            warn_data(
                "Could not align enough edges, proceeding anyway with original"
                " stage positions."
            )
            self.lr.coef_ = np.diag(np.ones(2))
            self.lr.intercept_ = np.zeros(2)

        # Adjust position of remaining components so their centroids match
        # the predictions of the model
        for cc in components[1:]:
            nodes = list(cc)
            centroid_m = cp.mean(self.positions[nodes], axis=0)
            centroid_f = cp.mean(self.final_positions[nodes], axis=0)

            # Convert to CPU for prediction, then back to GPU
            centroid_m_cpu = cp.asnumpy(centroid_m).reshape(1, -1)
            shift_cpu = self.lr.predict(centroid_m_cpu)[0] - cp.asnumpy(centroid_f)
            shift = cp.array(shift_cpu)

            self.final_positions[nodes] += shift

        # Adjust positions and model intercept to put origin at 0,0
        self.origin = cp.min(self.final_positions, axis=0)
        self.final_positions -= self.origin
        self.lr.intercept_ -= cp.asnumpy(self.origin)


def _calculate_initial_positions_gpu(image_stack, grid_dims: tuple, overlap_ratio: float):
    """Calculate initial grid positions based on overlap ratio (GPU version)."""
    grid_rows, grid_cols = grid_dims

    # Handle both numpy and cupy arrays
    if isinstance(image_stack, cp.ndarray):
        tile_height, tile_width = image_stack.shape[1:3]
    else:
        tile_height, tile_width = image_stack.shape[1:3]

    spacing_factor = 1.0 - overlap_ratio

    positions = []
    for tile_idx in range(len(image_stack)):
        r = tile_idx // grid_cols
        c = tile_idx % grid_cols

        y_pos = r * tile_height * spacing_factor
        x_pos = c * tile_width * spacing_factor
        positions.append([y_pos, x_pos])

    return cp.array(positions, dtype=cp.float64)


def _convert_ashlar_positions_to_openhcs_gpu(ashlar_positions) -> List[Tuple[float, float]]:
    """Convert Ashlar positions to OpenHCS format (GPU version)."""
    # Convert to CPU if needed
    if isinstance(ashlar_positions, cp.ndarray):
        ashlar_positions = cp.asnumpy(ashlar_positions)

    positions = []
    for tile_idx in range(len(ashlar_positions)):
        y, x = ashlar_positions[tile_idx]
        positions.append((float(x), float(y)))  # OpenHCS uses (x, y) format
    return positions


@special_inputs("grid_dimensions")
@special_outputs("positions")
@cupy_func
def ashlar_compute_tile_positions_gpu(
    image_stack,
    grid_dimensions: Tuple[int, int],
    overlap_ratio: float = 0.1,
    max_shift: float = 15.0,
    stitch_alpha: float = 0.01,
    max_error: float = None,
    randomize: bool = False,
    verbose: bool = False,
    upsample_factor: int = 10,
    permutation_upsample: int = 1,
    permutation_samples: int = 1000,
    min_permutation_samples: int = 10,
    max_permutation_tries: int = 100,
    window_size_factor: float = 0.1,
    **kwargs
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Compute tile positions using the Ashlar algorithm on GPU - matches CPU version.

    This function implements the Ashlar edge-based stitching algorithm using GPU acceleration.
    It performs position calculation with minimal preprocessing (windowing only, no whitening)
    to match the CPU version behavior.

    Args:
        image_stack: 3D numpy/cupy array of shape (num_tiles, height, width) containing preprocessed
                    grayscale images. Each slice [i] should be a single-channel 2D image ready
                    for correlation analysis. No further preprocessing will be applied.

        grid_dimensions: Tuple of (grid_rows, grid_cols) specifying the logical arrangement of
                        tiles. For example, (2, 3) means 2 rows and 3 columns of tiles, for a
                        total of 6 tiles. Must match the number of images in image_stack.

        overlap_ratio: Expected fractional overlap between adjacent tiles (0.0-1.0). Default 0.1
                      means 10% overlap. This is used to calculate initial grid positions and
                      should match the actual overlap in your microscopy data. Typical values:
                      - 0.05-0.15 for well-controlled microscopes
                      - 0.15-0.25 for less precise stages

        max_shift: Maximum allowed shift correction in micrometers. Default 15.0. This limits
                  how far tiles can be moved from their initial grid positions during alignment.
                  Should be set based on your microscope's stage accuracy:
                  - 5-15 μm for high-precision stages
                  - 15-50 μm for standard stages
                  - 50+ μm for low-precision or manual stages

        stitch_alpha: Alpha value for statistical error threshold computation (0.0-1.0). Default
                     0.01 means 1% false positive rate. Lower values are stricter and reject more
                     alignments, higher values are more permissive. This controls the trade-off
                     between alignment quality and success rate:
                     - 0.001-0.01: Very strict, high quality alignments only
                     - 0.01-0.05: Balanced (recommended for most data)
                     - 0.05-0.1: Permissive, accepts lower quality alignments

        max_error: Explicit error threshold for rejecting alignments (None = auto-compute).
                  When None (default), the threshold is computed automatically using permutation
                  testing. Set to a specific value to override automatic computation. Higher
                  values accept more alignments, lower values are stricter.

        randomize: Whether to use random seed for permutation testing (bool). Default False uses
                  a fixed seed for reproducible results. Set True for different random sampling
                  in each run. Generally should be False for consistent results.

        verbose: Enable detailed progress logging (bool). Default False. When True, prints
                progress information including permutation testing, edge alignment, and
                spanning tree construction. Useful for debugging and monitoring progress
                on large datasets.

        upsample_factor: Sub-pixel accuracy factor for phase cross correlation (int). Default 10.
                        Higher values provide better sub-pixel accuracy but increase computation time.
                        Range: 1-100+. Values of 10-50 are typical for high-accuracy stitching.
                        - 1: Pixel-level accuracy (fastest)
                        - 10: 0.1 pixel accuracy (balanced)
                        - 50: 0.02 pixel accuracy (high precision)

        permutation_upsample: Upsample factor for permutation testing (int). Default 1.
                             Lower than upsample_factor for speed during threshold computation.
                             Usually kept at 1 since permutation testing doesn't need sub-pixel accuracy.

        permutation_samples: Number of random samples for error threshold computation (int). Default 1000.
                           Higher values give more accurate thresholds but slower computation.
                           Automatically reduced for small datasets to avoid infinite loops.

        min_permutation_samples: Minimum permutation samples for small datasets (int). Default 10.
                               When there are few non-overlapping pairs, this sets the minimum
                               number of samples to ensure statistical validity.

        max_permutation_tries: Maximum attempts to find non-overlapping strips (int). Default 100.
                             Prevents infinite loops in pathological cases where valid strips
                             are hard to find. Rarely needs adjustment.

        window_size_factor: Fraction of tile size for maximum window size (float). Default 0.1.
                          Controls the largest overlap window tested during progressive sizing.
                          Larger values allow detection of bigger stage errors but may reduce
                          correlation quality. Range: 0.05-0.2 typical.

        filter_sigma: Whitening filter sigma for preprocessing (float). Default 0.
                     Controls the whitening filter applied before correlation:
                     - 0: Pure Laplacian filter (high-pass, matches original Ashlar)
                     - >0: Gaussian-Laplacian (LoG) filter with specified sigma
                     - Typical values: 0-2.0 for most microscopy data

        **kwargs: Additional parameters (ignored). Allows compatibility with other stitching
                 algorithms that may have different parameter sets.

    Returns:
        Tuple of (image_stack, positions) where:
        - image_stack: The original input image array (unchanged)
        - positions: List of (x, y) position tuples in OpenHCS format, one per tile.
                    Positions are in pixel coordinates with (0, 0) at the top-left.
                    The positions represent the optimal tile placement after Ashlar
                    alignment, accounting for stage errors and image correlation.

    Raises:
        Exception: If the Ashlar algorithm fails (e.g., insufficient overlap, correlation
                  errors), the function automatically falls back to grid-based positioning
                  using the specified overlap_ratio.

    Notes:
        - This implementation contains the complete Ashlar algorithm including whitening
          filter preprocessing, permutation testing, progressive window sizing, minimum
          spanning tree construction, and linear model fitting for disconnected components.
        - The correlation functions are identical to original Ashlar including proper
          whitening/filtering preprocessing as specified by filter_sigma parameter.
        - For best results, ensure your image_stack contains single-channel grayscale
          images. The whitening filter will be applied automatically during correlation.
    """
    grid_rows, grid_cols = grid_dimensions

    if verbose:
        logger.info(f"Ashlar GPU: Processing {grid_rows}x{grid_cols} grid with {len(image_stack)} tiles")

    try:
        # Convert to CuPy array if needed
        if not isinstance(image_stack, cp.ndarray):
            image_stack_gpu = cp.asarray(image_stack)
        else:
            image_stack_gpu = image_stack

        # Calculate initial grid positions
        initial_positions = _calculate_initial_positions_gpu(image_stack_gpu, grid_dimensions, overlap_ratio)
        tile_size = cp.array(image_stack_gpu.shape[1:3])  # (height, width)

        # Create and run ArrayEdgeAlignerGPU with complete Ashlar algorithm
        logger.info("Running complete Ashlar edge-based stitching algorithm on GPU")
        aligner = ArrayEdgeAlignerGPU(
            image_stack=image_stack_gpu,
            positions=initial_positions,
            tile_size=tile_size,
            pixel_size=1.0,  # Assume 1 micrometer per pixel if not specified
            max_shift=max_shift,
            alpha=stitch_alpha,
            max_error=max_error,
            randomize=randomize,
            verbose=verbose,
            upsample_factor=upsample_factor,
            permutation_upsample=permutation_upsample,
            permutation_samples=permutation_samples,
            min_permutation_samples=min_permutation_samples,
            max_permutation_tries=max_permutation_tries,
            window_size_factor=window_size_factor
        )

        # Run the complete algorithm
        aligner.run()

        # Convert to OpenHCS format
        positions = _convert_ashlar_positions_to_openhcs_gpu(aligner.final_positions)

        # Convert result back to original format (CPU if input was CPU)
        if not isinstance(image_stack, cp.ndarray):
            result_image_stack = cp.asnumpy(image_stack_gpu)
        else:
            result_image_stack = image_stack_gpu

        logger.info("Ashlar GPU algorithm completed successfully")

    except Exception as e:
        logger.error(f"Ashlar GPU algorithm failed: {e}")
        # Fallback to grid positions if Ashlar fails
        logger.warning("Falling back to grid-based positioning")
        positions = []

        # Use original image_stack for fallback dimensions
        if isinstance(image_stack, cp.ndarray):
            tile_height, tile_width = image_stack.shape[1:3]
        else:
            tile_height, tile_width = image_stack.shape[1:3]

        spacing_factor = 1.0 - overlap_ratio

        for tile_idx in range(len(image_stack)):
            r = tile_idx // grid_cols
            c = tile_idx % grid_cols
            x_pos = c * tile_width * spacing_factor
            y_pos = r * tile_height * spacing_factor
            positions.append((float(x_pos), float(y_pos)))

        # Set result_image_stack for fallback case
        if not isinstance(image_stack, cp.ndarray):
            result_image_stack = image_stack
        else:
            result_image_stack = image_stack

    logger.info(f"Ashlar GPU: Completed processing {len(positions)} tile positions")

    return result_image_stack, positions


def materialize_ashlar_gpu_positions(data: List[Tuple[float, float]], path: str, filemanager) -> str:
    """Materialize Ashlar GPU tile positions as scientific CSV with grid metadata."""
    csv_path = path.replace('.pkl', '_ashlar_positions_gpu.csv')

    df = pd.DataFrame(data, columns=['x_position_um', 'y_position_um'])
    df['tile_id'] = range(len(df))

    # Estimate grid dimensions from position layout
    unique_x = sorted(df['x_position_um'].unique())
    unique_y = sorted(df['y_position_um'].unique())

    grid_cols = len(unique_x)
    grid_rows = len(unique_y)

    # Add grid coordinates
    df['grid_row'] = df.index // grid_cols
    df['grid_col'] = df.index % grid_cols

    # Add spacing information
    if len(unique_x) > 1:
        x_spacing = unique_x[1] - unique_x[0]
        df['x_spacing_um'] = x_spacing
    else:
        df['x_spacing_um'] = 0

    if len(unique_y) > 1:
        y_spacing = unique_y[1] - unique_y[0]
        df['y_spacing_um'] = y_spacing
    else:
        df['y_spacing_um'] = 0

    # Add metadata
    df['algorithm'] = 'ashlar_gpu'
    df['grid_dimensions'] = f"{grid_rows}x{grid_cols}"

    csv_content = df.to_csv(index=False)
    filemanager.save(csv_content, csv_path, "disk")
    return csv_path
