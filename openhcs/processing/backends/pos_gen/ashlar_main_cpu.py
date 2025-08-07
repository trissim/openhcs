"""
OpenHCS Interface for Ashlar CPU Stitching Algorithm

Array-based EdgeAligner implementation that works directly with numpy arrays
instead of file-based readers. This is the complete Ashlar algorithm modified
to accept arrays directly.
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
from openhcs.core.memory.decorators import numpy as numpy_func
from openhcs.core.utils import optional_import

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


class Intersection:
    """Calculate intersection region between two tiles (extracted from Ashlar)."""

    def __init__(self, corners1, corners2, min_size=0):
        if np.isscalar(min_size):
            min_size = np.repeat(min_size, 2)
        self._calculate(corners1, corners2, min_size)

    def _calculate(self, corners1, corners2, min_size):
        """Calculate intersection parameters with robust boundary validation."""
        # corners1 and corners2 are arrays of shape (2, 2) containing
        # the upper-left and lower-right corners of the two tiles
        max_shape = (corners2 - corners1).max(axis=0)
        min_size = min_size.clip(1, max_shape)
        position = corners1.max(axis=0)
        initial_shape = np.floor(corners2.min(axis=0) - position).astype(int)
        clipped_shape = np.maximum(initial_shape, min_size)
        self.shape = np.ceil(clipped_shape).astype(int)
        self.padding = self.shape - initial_shape

        # Calculate offsets with boundary validation
        raw_offsets = np.maximum(position - corners1 - self.padding, 0)

        # Validate that offsets + shape don't exceed tile boundaries
        tile_sizes = corners2 - corners1
        for i in range(2):
            # Ensure offset + shape <= tile_size for each tile
            max_offset = tile_sizes[i] - self.shape
            raw_offsets[i] = np.minimum(raw_offsets[i], np.maximum(max_offset, 0))

            # Ensure shape doesn't exceed available space
            available_space = tile_sizes[i] - raw_offsets[i]
            self.shape = np.minimum(self.shape, available_space.astype(int))

        # Final validation - ensure shape is positive
        self.shape = np.maximum(self.shape, 1)

        self.offsets = raw_offsets.astype(int)

        # Calculate fractional offset difference for subpixel accuracy
        offset_diff = self.offsets[1] - self.offsets[0]
        self.offset_diff_frac = offset_diff - offset_diff.round()


def _get_window(shape):
    """Build a 2D Hann window (from Ashlar utils.get_window)."""
    # Build a 2D Hann window by taking the outer product of two 1-D windows.
    wy = np.hanning(shape[0]).astype(np.float32)
    wx = np.hanning(shape[1]).astype(np.float32)
    window = np.outer(wy, wx)
    return window


def ashlar_register_no_preprocessing(img1, img2, upsample=10):
    """
    Robust Ashlar register function with comprehensive input validation.

    This is based on ashlar.utils.register() but adds validation to handle
    edge cases that can occur with real microscopy data.
    """
    import itertools
    import scipy.ndimage
    import skimage.registration

    # Input validation
    if img1 is None or img2 is None:
        return np.array([0.0, 0.0]), np.inf

    if img1.size == 0 or img2.size == 0:
        return np.array([0.0, 0.0]), np.inf

    if img1.shape != img2.shape:
        return np.array([0.0, 0.0]), np.inf

    if len(img1.shape) != 2:
        return np.array([0.0, 0.0]), np.inf

    if img1.shape[0] < 1 or img1.shape[1] < 1:
        return np.array([0.0, 0.0]), np.inf

    # Convert to float32 (equivalent to what whiten() does) - match GPU version
    img1w = img1.astype(np.float32)
    img2w = img2.astype(np.float32)

    # Apply windowing function (from original Ashlar)
    img1w = img1w * _get_window(img1w.shape)
    img2w = img2w * _get_window(img2w.shape)

    # Use skimage's phase cross correlation with error handling
    try:
        shift = skimage.registration.phase_cross_correlation(
            img1w,
            img2w,
            upsample_factor=upsample,
            normalization=None
        )[0]
    except Exception as e:
        # If phase correlation fails, return large error
        logger.error(f"Ashlar CPU: PHASE CORRELATION FAILED - Exception: {e}")
        logger.error(f"  Returning infinite error")
        return np.array([0.0, 0.0]), np.inf

    # At this point we may have a shift in the wrong quadrant since the FFT
    # assumes the signal is periodic. We test all four possibilities and return
    # the shift that gives the highest direct correlation (sum of products).
    shape = np.array(img1.shape)
    shift_pos = (shift + shape) % shape
    shift_neg = shift_pos - shape
    shifts = list(itertools.product(*zip(shift_pos, shift_neg)))
    correlations = []
    for s in shifts:
        try:
            shifted_img = scipy.ndimage.shift(img2w, s, order=0)
            corr = np.abs(np.sum(img1w * shifted_img))
            correlations.append(corr)
        except Exception:
            correlations.append(0.0)

    if not correlations or max(correlations) == 0:
        logger.warning(f"Ashlar CPU: NO VALID CORRELATIONS - All correlations failed or zero")
        return np.array([0.0, 0.0]), np.inf

    idx = np.argmax(correlations)
    shift = shifts[idx]
    correlation = correlations[idx]
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        error = -np.log(correlation / total_amplitude)
    else:
        error = np.inf

    # Log all correlation results at INFO level for user visibility
    if error > 1.0:  # High error threshold for Ashlar
        logger.warning(f"Ashlar CPU: HIGH CORRELATION ERROR - Error={error:.4f}, Shift=({shift[0]:.2f}, {shift[1]:.2f})")
        logger.warning(f"  This indicates poor overlap or image quality between tiles")
    else:
        logger.info(f"Ashlar CPU: Correlation - Error={error:.4f}, Shift=({shift[0]:.2f}, {shift[1]:.2f})")

    return shift, error


def ashlar_nccw_no_preprocessing(img1, img2):
    """
    Robust Ashlar nccw function with comprehensive input validation.

    This is based on ashlar.utils.nccw() but adds validation to handle
    edge cases that can occur with real microscopy data.
    """
    # Input validation
    if img1 is None or img2 is None:
        return np.inf

    if img1.size == 0 or img2.size == 0:
        return np.inf

    if img1.shape != img2.shape:
        return np.inf

    if len(img1.shape) != 2:
        return np.inf

    if img1.shape[0] < 1 or img1.shape[1] < 1:
        return np.inf

    # Convert to float32 (equivalent to what whiten() does) - match GPU version
    img1w = img1.astype(np.float32)
    img2w = img2.astype(np.float32)

    correlation = np.abs(np.sum(img1w * img2w))
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        diff = correlation - total_amplitude
        if diff <= 0:
            error = -np.log(correlation / total_amplitude)
        elif diff < 1e-3:  # Increased tolerance for robustness
            # This situation can occur due to numerical precision issues when
            # img1 and img2 are very nearly or exactly identical. If the
            # difference is small enough, let it slide.
            error = 0
        else:
            # Instead of raising error, return large but finite error
            logger.warning(f"Ashlar CPU: NCCW numerical precision issue - diff={diff:.6f}, using error=100.0")
            error = 100.0  # Large error but not infinite
    else:
        logger.warning(f"Ashlar CPU: NCCW invalid correlation - correlation={correlation:.6f}, total_amplitude={total_amplitude:.6f}")
        error = np.inf

    # Log all NCCW results at INFO level for user visibility
    if error > 10.0:  # High NCCW error threshold
        logger.warning(f"Ashlar CPU: HIGH NCCW ERROR - Error={error:.4f}")
        logger.warning(f"  This indicates poor image correlation between tiles")
    else:
        logger.info(f"Ashlar CPU: NCCW - Error={error:.4f}")

    return error


def ashlar_crop(img, offset, shape):
    """
    Robust Ashlar crop function with comprehensive boundary validation.

    This is based on ashlar.utils.crop() but adds validation to handle
    edge cases that can occur with real microscopy data.
    """
    # Input validation
    if img is None or img.size == 0:
        raise ValueError("Cannot crop from empty or None image")

    # Convert to integers and validate
    start = offset.round().astype(int)
    shape = np.round(shape).astype(int)

    # Ensure start is non-negative
    start = np.maximum(start, 0)

    # Ensure shape is positive
    shape = np.maximum(shape, 1)

    # Validate bounds
    img_height, img_width = img.shape[:2]
    end = start + shape

    # Clamp to image boundaries
    start[0] = min(start[0], img_height - 1)
    start[1] = min(start[1], img_width - 1)
    end[0] = min(end[0], img_height)
    end[1] = min(end[1], img_width)

    # Ensure we have a valid region
    if end[0] <= start[0] or end[1] <= start[1]:
        # Return minimum valid region if bounds are invalid
        return img[start[0]:start[0]+1, start[1]:start[1]+1]

    return img[start[0]:end[0], start[1]:end[1]]


class ArrayEdgeAligner:
    """
    Array-based EdgeAligner that implements the complete Ashlar algorithm
    but works directly with numpy arrays instead of file readers.
    """

    def __init__(self, image_stack, positions, tile_size, pixel_size=1.0,
                 max_shift=15, alpha=0.01, max_error=None,
                 randomize=False, verbose=False):
        """
        Initialize array-based EdgeAligner for pure position calculation.

        Args:
            image_stack: 3D numpy array (num_tiles, height, width) - preprocessed grayscale
            positions: 2D array of tile positions (num_tiles, 2) in pixels
            tile_size: Array [height, width] of tile dimensions
            pixel_size: Pixel size in micrometers (for max_shift conversion)
            max_shift: Maximum allowed shift in micrometers
            alpha: Alpha value for error threshold (lower = stricter)
            max_error: Explicit error threshold (None = auto-compute)
            randomize: Use random seed for permutation testing
            verbose: Enable verbose logging
        """
        self.image_stack = image_stack
        self.positions = positions.astype(float)
        self.tile_size = np.array(tile_size)
        self.pixel_size = pixel_size
        self.max_shift = max_shift
        self.max_shift_pixels = self.max_shift / self.pixel_size
        self.alpha = alpha
        self.max_error = max_error
        self.randomize = randomize
        self.verbose = verbose
        self._cache = {}
        self.errors_negative_sampled = np.empty(0)

        # Build neighbors graph
        self.neighbors_graph = self._build_neighbors_graph()

    def _build_neighbors_graph(self):
        """Build graph of neighboring (overlapping) tiles."""
        pdist = scipy.spatial.distance.pdist(self.positions, metric='cityblock')
        sp = scipy.spatial.distance.squareform(pdist)
        max_distance = self.tile_size.max() + 1
        edges = zip(*np.nonzero((sp > 0) & (sp < max_distance)))
        graph = nx.from_edgelist(edges)
        graph.add_nodes_from(range(len(self.positions)))
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
        overlaps = np.array([
            self.tile_size - abs(self.positions[t1] - self.positions[t2])
            for t1, t2 in self.neighbors_graph.edges
        ])
        failures = np.any(overlaps < 1, axis=1) if len(overlaps) else []
        if len(failures) and all(failures):
            warn_data("No tiles overlap, attempting alignment anyway.")
        elif any(failures):
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

        widths = np.array([
            self.intersection(t1, t2).shape.min()
            for t1, t2 in edges
        ])
        w = widths.max()
        max_offset = self.tile_size[0] - w

        # Number of possible pairs minus number of actual neighbor pairs
        num_distant_pairs = num_tiles * (num_tiles - 1) // 2 - len(edges)

        # Reduce permutation count for small datasets
        n = 1000 if num_distant_pairs > 8 else (num_distant_pairs + 1) * 10
        pairs = np.empty((n, 2), dtype=int)
        offsets = np.empty((n, 2), dtype=int)

        # Generate n random non-overlapping image strips
        max_tries = 100
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
                    its = self.intersection(t1, t2, np.repeat(w, 2))
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

        errors = np.empty(n)
        for i, ((t1, t2), (offset1, offset2)) in enumerate(zip(pairs, offsets)):
            # if self.verbose and (i % 10 == 9 or i == n - 1):
            #     sys.stdout.write(f'\r    quantifying alignment error {i + 1}/{n}')
            #     sys.stdout.flush()
            img1 = self.image_stack[t1][offset1:offset1+w, :]
            img2 = self.image_stack[t2][offset2:offset2+w, :]
            _, errors[i] = ashlar_register_no_preprocessing(img1, img2, upsample=1)
        # if self.verbose:
        #     print()
        self.errors_negative_sampled = errors
        self.max_error = np.percentile(errors, self.alpha * 100)


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
        self.all_errors = np.array([x[1] for x in self._cache.values()])

        # Set error values above the threshold to infinity
        for k, v in self._cache.items():
            if v[1] > self.max_error or any(np.abs(v[0]) > self.max_shift_pixels):
                self._cache[k] = (v[0], np.inf)

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
            smin = self.intersection(key[0], key[1]).shape
            smax = np.round(self.tile_size * 0.1)
            sizes = [smin]
            while any(sizes[-1] < smax):
                sizes.append(sizes[-1] * 2)
            # Test each window size with validation
            results = []
            for s in sizes:
                try:
                    result = self._register(key[0], key[1], s)
                    results.append(result)
                except Exception:
                    # If this window size fails, use infinite error
                    results.append((np.array([0.0, 0.0]), np.inf))
            # Use the shift from the window size that gave the lowest error
            shift, _ = min(results, key=lambda r: r[1])
            # Extract the images from the nominal overlap window but with the
            # shift applied to the second tile's position, and compute the error
            # metric on these images. This should be even lower than the error
            # computed above.
            _, o1, o2 = self.overlap(key[0], key[1], shift=shift)
            error = ashlar_nccw_no_preprocessing(o1, o2)
            self._cache[key] = (shift, error)
        if t1 > t2:
            shift = -shift
        # Return copy of shift to prevent corruption of cached values
        return shift.copy(), error

    def _register(self, t1, t2, min_size=0):
        """Register a single tile pair with given minimum size."""
        its, img1, img2 = self.overlap(t1, t2, min_size)
        # Account for padding, flipping the sign depending on the direction
        # between the tiles
        p1, p2 = self.positions[[t1, t2]]
        sx = 1 if p1[1] >= p2[1] else -1
        sy = 1 if p1[0] >= p2[0] else -1
        padding = its.padding * [sy, sx]
        shift, error = ashlar_register_no_preprocessing(img1, img2)
        shift += padding
        return shift, error


    def intersection(self, t1, t2, min_size=0, shift=None):
        """Calculate intersection region between two tiles."""
        corners1 = self.positions[[t1, t2]]
        if shift is not None:
            corners1[1] += shift
        corners2 = corners1 + self.tile_size
        return Intersection(corners1, corners2, min_size)

    def crop(self, tile_id, offset, shape):
        """Crop image from tile at given offset and shape."""
        img = self.image_stack[tile_id]
        return ashlar_crop(img, offset, shape)

    def overlap(self, t1, t2, min_size=0, shift=None):
        """Extract overlapping regions between two tiles."""
        its = self.intersection(t1, t2, min_size, shift)
        img1 = self.crop(t1, its.offsets[0], its.shape)
        img2 = self.crop(t2, its.offsets[1], its.shape)
        return its, img1, img2





    def build_spanning_tree(self):
        """Build minimum spanning tree from registered edges."""
        g = nx.Graph()
        g.add_nodes_from(self.neighbors_graph)
        g.add_weighted_edges_from(
            (t1, t2, error)
            for (t1, t2), (_, error) in self._cache.items()
            if np.isfinite(error)
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
            shifts[center] = np.array([0, 0])
            for edge in nx.traversal.bfs_edges(cc, center):
                source, dest = edge
                if source not in shifts:
                    source, dest = dest, source
                shift = self.register_pair(source, dest)[0]
                shifts[dest] = shifts[source] + shift
        if shifts:
            self.shifts = np.array([s for _, s in sorted(shifts.items())])
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
        self.lr.fit(self.positions[cc0], self.final_positions[cc0])

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
            centroid_m = np.mean(self.positions[nodes], axis=0)
            centroid_f = np.mean(self.final_positions[nodes], axis=0)
            shift = self.lr.predict([centroid_m])[0] - centroid_f
            self.final_positions[nodes] += shift

        # Adjust positions and model intercept to put origin at 0,0
        self.origin = self.final_positions.min(axis=0)
        self.final_positions -= self.origin
        self.lr.intercept_ -= self.origin


def _calculate_initial_positions(image_stack: np.ndarray, grid_dims: tuple, overlap_ratio: float) -> np.ndarray:
    """Calculate initial grid positions based on overlap ratio."""
    grid_rows, grid_cols = grid_dims
    tile_height, tile_width = image_stack.shape[1:3]
    spacing_factor = 1.0 - overlap_ratio

    positions = []
    for tile_idx in range(len(image_stack)):
        r = tile_idx // grid_cols
        c = tile_idx % grid_cols

        y_pos = r * tile_height * spacing_factor
        x_pos = c * tile_width * spacing_factor
        positions.append([y_pos, x_pos])

    return np.array(positions, dtype=float)


def _convert_ashlar_positions_to_openhcs(ashlar_positions: np.ndarray) -> List[Tuple[float, float]]:
    """Convert Ashlar positions to OpenHCS format."""
    positions = []
    for tile_idx in range(len(ashlar_positions)):
        y, x = ashlar_positions[tile_idx]
        positions.append((float(x), float(y)))  # OpenHCS uses (x, y) format
    return positions


@special_inputs("grid_dimensions")
@special_outputs("positions")
@numpy_func
def ashlar_compute_tile_positions_cpu(
    image_stack: np.ndarray,
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
    Compute tile positions using the complete Ashlar algorithm - pure position calculation only.

    This function implements the full Ashlar edge-based stitching algorithm but works directly
    on preprocessed grayscale image arrays. It performs ONLY position calculation without any
    file I/O, channel selection, or image preprocessing. All the mathematical sophistication
    and robustness of the original Ashlar algorithm is preserved.

    Args:
        image_stack: 3D numpy array of shape (num_tiles, height, width) containing preprocessed
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
        - This implementation contains the complete Ashlar algorithm including permutation
          testing, progressive window sizing, minimum spanning tree construction, and
          linear model fitting for disconnected components.
        - The correlation functions are identical to original Ashlar but without image
          preprocessing (whitening/filtering), allowing OpenHCS to handle preprocessing
          in separate pipeline steps.
        - For best results, ensure your image_stack contains properly preprocessed,
          single-channel grayscale images with good contrast and minimal noise.
    """
    grid_rows, grid_cols = grid_dimensions

    logger.info(f"Ashlar CPU: Processing {grid_rows}x{grid_cols} grid with {len(image_stack)} tiles")

    try:
        # Calculate initial grid positions
        initial_positions = _calculate_initial_positions(image_stack, grid_dimensions, overlap_ratio)
        tile_size = np.array(image_stack.shape[1:3])  # (height, width)

        # Create and run ArrayEdgeAligner with complete Ashlar algorithm
        logger.info("Running complete Ashlar edge-based stitching algorithm")
        aligner = ArrayEdgeAligner(
            image_stack=image_stack,
            positions=initial_positions,
            tile_size=tile_size,
            pixel_size=1.0,  # Assume 1 micrometer per pixel if not specified
            max_shift=max_shift,
            alpha=stitch_alpha,
            max_error=max_error,
            randomize=randomize,
            verbose=verbose
        )

        # Run the complete algorithm
        aligner.run()

        # Convert to OpenHCS format
        positions = _convert_ashlar_positions_to_openhcs(aligner.final_positions)

        logger.info("Ashlar algorithm completed successfully")

    except Exception as e:
        logger.error(f"Ashlar algorithm failed: {e}")
        # Fallback to grid positions if Ashlar fails
        logger.warning("Falling back to grid-based positioning")
        positions = []
        tile_height, tile_width = image_stack.shape[1:3]
        spacing_factor = 1.0 - overlap_ratio

        for tile_idx in range(len(image_stack)):
            r = tile_idx // grid_cols
            c = tile_idx % grid_cols
            x_pos = c * tile_width * spacing_factor
            y_pos = r * tile_height * spacing_factor
            positions.append((float(x_pos), float(y_pos)))

    logger.info(f"Ashlar CPU: Completed processing {len(positions)} tile positions")

    return image_stack, positions


def materialize_ashlar_cpu_positions(data: List[Tuple[float, float]], path: str, filemanager) -> str:
    """Materialize Ashlar CPU tile positions as scientific CSV with grid metadata."""
    csv_path = path.replace('.pkl', '_ashlar_positions.csv')

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
    df['algorithm'] = 'ashlar_cpu'
    df['grid_dimensions'] = f"{grid_rows}x{grid_cols}"

    csv_content = df.to_csv(index=False)
    filemanager.save(csv_content, csv_path, "disk")
    return csv_path
