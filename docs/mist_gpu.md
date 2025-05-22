# GPU-Accelerated MIST Implementation

This document describes the GPU-accelerated implementation of MIST (Microscopy Image Stitching Tool) in ezstitcher.

**Note**: MIST is considered image processing functionality and is therefore located in the `ezstitcher.core.processing.backends` module, with a FunctionStep wrapper in `ezstitcher.core.steps`.

## Overview

The MIST algorithm is a method for computing the positions of microscopy image tiles in a grid. This implementation uses CuPy to accelerate the computation on GPUs, providing significant performance improvements over CPU-based implementations.

The implementation follows the OpenHCS doctrinal principles:
- **Clause 3 — Declarative Primacy**: All functions are pure and stateless
- **Clause 65 — Fail Loudly**: No silent fallbacks or inferred capabilities
- **Clause 88 — No Inferred Capabilities**: Explicit CuPy dependency
- **Clause 101 — Memory Declaration**: Memory-resident output, no side effects
- **Clause 273 — Memory Backend Restrictions**: GPU-only implementation

## Algorithm

The MIST algorithm computes tile positions in a grid by:

1. **Phase Correlation**: Computing the relative shift between adjacent tiles using phase correlation
2. **Iterative Refinement**: Refining the positions by considering multiple neighbors
3. **Global Optimization**: (Optional) Minimizing the overall alignment error

### Phase Correlation

Phase correlation is a method for computing the relative shift between two images. It works by:

1. Computing the Fourier transform of both images
2. Computing the cross-power spectrum
3. Normalizing the cross-power spectrum
4. Computing the inverse Fourier transform
5. Finding the peak in the resulting correlation matrix

The position of the peak corresponds to the relative shift between the images.

### Iterative Refinement

The positions are refined iteratively by:

1. Computing the relative shift between each tile and its neighbors
2. Updating the position of each tile based on the computed shifts
3. Repeating for a specified number of iterations

## Usage

### Direct Usage

```python
import cupy as cp
from ezstitcher.core.positioning.mist_gpu import mist_compute_tile_positions

# Create a stack of tiles
# tile_stack shape: (num_tiles, tile_height, tile_width)
tile_stack = cp.array(...)

# Compute tile positions
positions = mist_compute_tile_positions(
    tile_stack,
    num_rows=3,
    num_cols=3,
    patch_size=128,
    search_radius=20,
    overlap_ratio=0.1,
    refinement_iterations=2,
    verbose=True
)

# positions shape: (num_tiles, 2)
# Each row is [y, x] position
```

### Using the PositionGenerationStep

```python
from ezstitcher.core.positioning.mist_step import MistPositionGenerationStep
from ezstitcher.core.context import ProcessingContext

# Create the step
step = MistPositionGenerationStep(
    num_rows=3,
    num_cols=3,
    patch_size=128,
    search_radius=20,
    overlap_ratio=0.1,
    refinement_iterations=2
)

# Create a context with input data
context = ProcessingContext()
context.set_input_data(tile_stack)

# Process the data
result = step.process(context)

# Get the positions from the result
positions = result.data["positions"]
```

## Parameters

The `mist_compute_tile_positions` function accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_stack` | cp.ndarray | (required) | 3D tensor of shape (Z, Y, X) where each Z slice is a 2D tile |
| `num_rows` | int | (required) | Number of rows in the grid |
| `num_cols` | int | (required) | Number of columns in the grid |
| `patch_size` | int | 128 | Size of the patches used for correlation |
| `search_radius` | int | 20 | Maximum search radius for correlation |
| `stride` | int | 64 | Stride for patch extraction |
| `method` | str | "phase_correlation" | Method for computing correlation |
| `normalize` | bool | True | Whether to normalize the images before correlation |
| `fft_backend` | str | "cupy" | FFT backend to use (must be "cupy") |
| `verbose` | bool | False | Whether to print progress information |
| `overlap_ratio` | float | 0.1 | Expected overlap ratio between adjacent tiles |
| `subpixel` | bool | True | Whether to compute subpixel precision |
| `refinement_iterations` | int | 1 | Number of refinement iterations |
| `global_optimization` | bool | False | Whether to perform global optimization |

## Performance Considerations

- The GPU-accelerated implementation is significantly faster than CPU-based implementations, especially for large grids of tiles.
- The performance depends on the size of the tiles, the number of tiles, and the GPU capabilities.
- The implementation uses CuPy's FFT capabilities, which are highly optimized for GPUs.
- The implementation supports processing tiles in batches to avoid memory issues.

## Example

See the example script at `examples/mist_gpu_example.py` for a complete demonstration of the MIST GPU implementation, including:

- Creating a synthetic grid of tiles with known positions
- Computing tile positions using MIST
- Evaluating the accuracy of the computed positions
- Visualizing the results

## References

1. Chalfoun, J., Majurski, M., Blattner, T., Bhadriraju, K., Keyrouz, W., Bajcsy, P., & Brady, M. (2017). MIST: Accurate and Scalable Microscopy Image Stitching Tool with Stage Modeling and Error Minimization. Scientific Reports, 7(1), 4988.
2. Chalfoun, J., Majurski, M., Dima, A., Halter, M., Bhadriraju, K., & Brady, M. (2015). Microscopy Image Stitching Tool (MIST). https://isg.nist.gov/deepzoomweb/resources/csmet/pages/mist/mist.html
