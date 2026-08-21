"""
Template system for custom function creation.

Provides starter code templates for different memory types (numpy, cupy, torch, etc.)
with proper imports, decorators, and docstring examples.

All functions follow OpenHCS contracts:
    - First parameter must be named 'image' (3D array: C, Y, X)
    - Must be decorated with memory type decorator (@numpy, @cupy, etc.)
    - Must return processed image (optionally with metadata dict)

For analysis functions that produce structured outputs (cell counts, measurements, etc.):
    - Use @artifact_outputs decorator to declare outputs
    - Use materialization functions from openhcs.processing.materialization
    - Return tuple: (processed_image, analysis_result_1, analysis_result_2, ...)
"""

from arraybridge import MemoryType

# Public UI projection of ArrayBridge's framework declarations.
AVAILABLE_MEMORY_TYPES = tuple(memory_type.value for memory_type in MemoryType)

# Available template categories
AVAILABLE_TEMPLATE_CATEGORIES = ["processing", "analysis"]


def get_default_template() -> str:
    """
    Get the default custom function template (numpy backend).

    Returns:
        Template string with numpy decorator and example function
    """
    return NUMPY_TEMPLATE


def get_template_for_memory_type(memory_type: str) -> str:
    """
    Get template for a specific memory type.

    Args:
        memory_type: One of 'numpy', 'cupy', 'torch', 'tensorflow', 'jax', 'pyclesperanto'

    Returns:
        Template string with appropriate decorator and imports

    Raises:
        ValueError: If memory_type is not recognized
    """
    template_map: dict[str, str] = {
        "numpy": NUMPY_TEMPLATE,
        "cupy": CUPY_TEMPLATE,
        "torch": TORCH_TEMPLATE,
        "tensorflow": TENSORFLOW_TEMPLATE,
        "jax": JAX_TEMPLATE,
        "pyclesperanto": PYCLESPERANTO_TEMPLATE,
    }

    if memory_type not in template_map:
        raise ValueError(
            f"Unknown memory type: {memory_type}. "
            f"Must be one of: {', '.join(AVAILABLE_MEMORY_TYPES)}"
        )

    return template_map[memory_type]


# Template constants with proper typing and documentation

NUMPY_TEMPLATE = """from openhcs.core.memory import numpy
import numpy as np

# =============================================================================
# MEMORY DECORATORS - Choose ONE based on your compute backend:
#
#   @numpy         - CPU processing (default, always available)
#   @cupy          - NVIDIA GPU via CuPy (requires CUDA)
#   @torch         - PyTorch tensors (CPU or GPU)
#   @tensorflow    - TensorFlow tensors
#   @jax           - JAX arrays (CPU/GPU/TPU)
#   @pyclesperanto - GPU via OpenCL (AMD/NVIDIA/Intel)
#
# Import: from openhcs.core.memory import numpy, cupy, torch, ...
# =============================================================================

@numpy
def my_custom_function(image, scale: float = 1.0, offset: float = 0.0):
    \"\"\"
    Custom image processing function.

    Args:
        image: Input image as 3D array (C, Y, X)
               C = slices/channels, Y = height, X = width
        scale: Scaling factor to multiply image values
        offset: Offset to add after scaling

    Returns:
        Processed image as 3D array (C, Y, X)
    \"\"\"
    processed = image * scale + offset
    return processed


# =============================================================================
# ANALYSIS OUTPUTS - To produce structured data (CSVs, measurements, etc.):
#
# 1. Import the typed artifact declaration and row container:
#
#    from dataclasses import dataclass
#    from openhcs.core.artifacts import (
#        ArtifactMeasurementSubjectRelation,
#        ArtifactSpec,
#        MeasurementsArtifactType,
#    )
#    from openhcs.core.measurement_row_materialization import DataclassMeasurementColumnarRows
#    from openhcs.core.pipeline.function_contracts import artifact_outputs
#    from openhcs.processing.materialization import csv_only
#
# 2. Declare the semantic output and derive its columns from the row dataclass:
#
#    @dataclass(frozen=True)
#    class IntensityMeasurement:
#        slice_index: int
#        mean: float
#        std: float
#
#    @numpy
#    @artifact_outputs(
#        ArtifactSpec(
#            "measurements",
#            MeasurementsArtifactType,
#            materialization=csv_only(),
#            relations=(ArtifactMeasurementSubjectRelation(),),
#        )
#    )
#    def analyze_intensity(image, threshold: float = 0.5):
#        results = []
#        for i, slice_data in enumerate(image):
#            results.append(IntensityMeasurement(
#                slice_index=i,
#                mean=float(np.mean(slice_data)),
#                std=float(np.std(slice_data)),
#            ))
#        return image, DataclassMeasurementColumnarRows(
#            results,
#            row_type=IntensityMeasurement,
#        )
#
# =============================================================================
"""


CUPY_TEMPLATE = """from openhcs.core.memory import cupy
import cupy as cp

@cupy
def my_custom_function(image, sigma: float = 1.0):
    \"\"\"
    Custom image processing function using CuPy (GPU-accelerated).

    Args:
        image: Input image as 3D cupy array (C, Y, X)
               C = channels, Y = height, X = width
        sigma: Gaussian blur sigma parameter

    Returns:
        Processed image as 3D cupy array (C, Y, X)

    Notes:
        - Requires CUDA-compatible GPU
        - First parameter MUST be named 'image'
        - Must be decorated with @cupy
        - Input and output must be 3D arrays (C, Y, X)
    \"\"\"
    # Your GPU processing code here
    # Example: simple thresholding
    threshold = cp.mean(image) + sigma * cp.std(image)
    processed = cp.where(image > threshold, image, 0)

    return processed
"""


TORCH_TEMPLATE = """from openhcs.core.memory import torch
import torch

@torch
def my_custom_function(image, kernel_size: int = 3):
    \"\"\"
    Custom image processing function using PyTorch.

    Args:
        image: Input image as 3D torch tensor (C, Y, X)
               C = channels, Y = height, X = width
        kernel_size: Size of processing kernel

    Returns:
        Processed image as 3D torch tensor (C, Y, X)

    Notes:
        - Can run on CPU or GPU (automatic device handling)
        - First parameter MUST be named 'image'
        - Must be decorated with @torch
        - Input and output must be 3D tensors (C, Y, X)
    \"\"\"
    # Your PyTorch processing code here
    # Example: max pooling followed by upsampling
    import torch.nn.functional as F

    # Add batch dimension for pooling
    x = image.unsqueeze(0)  # (1, C, Y, X)

    # Apply max pooling
    pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    # Remove batch dimension
    processed = pooled.squeeze(0)  # (C, Y, X)

    return processed
"""


TENSORFLOW_TEMPLATE = """from openhcs.core.memory import tensorflow
import tensorflow as tf

@tensorflow
def my_custom_function(image, strength: float = 0.5):
    \"\"\"
    Custom image processing function using TensorFlow.

    Args:
        image: Input image as 3D tensorflow tensor (C, Y, X)
               C = channels, Y = height, X = width
        strength: Processing strength parameter

    Returns:
        Processed image as 3D tensorflow tensor (C, Y, X)

    Notes:
        - First parameter MUST be named 'image'
        - Must be decorated with @tensorflow
        - Input and output must be 3D tensors (C, Y, X)
    \"\"\"
    # Your TensorFlow processing code here
    # Example: contrast adjustment
    mean = tf.reduce_mean(image)
    processed = (image - mean) * (1.0 + strength) + mean

    return processed
"""


JAX_TEMPLATE = """from openhcs.core.memory import jax
import jax.numpy as jnp

@jax
def my_custom_function(image, power: float = 2.0):
    \"\"\"
    Custom image processing function using JAX.

    Args:
        image: Input image as 3D jax array (C, Y, X)
               C = channels, Y = height, X = width
        power: Power transformation parameter

    Returns:
        Processed image as 3D jax array (C, Y, X)

    Notes:
        - JAX provides automatic differentiation and JIT compilation
        - First parameter MUST be named 'image'
        - Must be decorated with @jax
        - Input and output must be 3D arrays (C, Y, X)
    \"\"\"
    # Your JAX processing code here
    # Example: power transformation
    # Normalize to [0, 1] range
    img_min = jnp.min(image)
    img_max = jnp.max(image)
    normalized = (image - img_min) / (img_max - img_min + 1e-7)

    # Apply power transformation
    transformed = jnp.power(normalized, power)

    # Scale back to original range
    processed = transformed * (img_max - img_min) + img_min

    return processed
"""


PYCLESPERANTO_TEMPLATE = """from openhcs.core.memory import pyclesperanto
import pyclesperanto_prototype as cle

@pyclesperanto
def my_custom_function(image, radius: float = 2.0):
    \"\"\"
    Custom image processing function using pyclesperanto (GPU-accelerated).

    Args:
        image: Input image as 3D array compatible with pyclesperanto (C, Y, X)
               C = channels, Y = height, X = width
        radius: Radius for morphological operations

    Returns:
        Processed image as 3D array (C, Y, X)

    Notes:
        - Optimized for GPU execution via OpenCL
        - First parameter MUST be named 'image'
        - Must be decorated with @pyclesperanto
        - Input and output must be 3D arrays (C, Y, X)
    \"\"\"
    # Your pyclesperanto processing code here
    # Example: Gaussian blur using cle
    processed = cle.gaussian_blur(image, sigma_x=radius, sigma_y=radius, sigma_z=0)

    return processed
"""


# =============================================================================
# ANALYSIS TEMPLATES - For functions that produce structured analysis outputs
# =============================================================================

NUMPY_ANALYSIS_TEMPLATE = """from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import CsvOptions, MaterializationSpec
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class AnalysisResult:
    \"\"\"Result dataclass - fields become CSV columns.\"\"\"
    slice_index: int
    measurement: float
    count: int
    # Add your analysis fields here


@numpy
@artifact_outputs(("analysis_results", MaterializationSpec(CsvOptions(
    fields=["slice_index", "measurement", "count"],
    filename_suffix=".csv"
))))
def my_analysis_function(image, threshold: float = 0.5) -> Tuple[np.ndarray, List[AnalysisResult]]:
    \"\"\"
    Analysis function that produces both processed image AND structured results.

    Args:
        image: Input image as 3D numpy array (C, Y, X)
        threshold: Analysis threshold parameter

    Returns:
        Tuple of:
        - Processed image as 3D numpy array (C, Y, X)
        - List of AnalysisResult dataclasses (auto-serialized to CSV)

    Notes:
        - @artifact_outputs declares that this function produces analysis data
        - CsvOptions auto-converts AnalysisResult fields to CSV columns
        - Return is ALWAYS a tuple: (image, special_output_1, special_output_2, ...)
    \"\"\"
    results = []
    
    # Process each slice and collect measurements
    for i, slice_data in enumerate(image):
        # Your analysis logic here
        measurement = float(np.mean(slice_data[slice_data > threshold]))
        count = int(np.sum(slice_data > threshold))
        
        results.append(AnalysisResult(
            slice_index=i,
            measurement=measurement,
            count=count
        ))
    
    # Processing (can be identity if you only want analysis)
    processed = image.copy()
    
    # Return tuple: (processed_image, analysis_results)
    return processed, results
"""


NUMPY_DUAL_OUTPUT_TEMPLATE = """from openhcs.core.memory import numpy
from openhcs.core.pipeline.function_contracts import artifact_outputs
from openhcs.processing.materialization import CsvOptions, JsonOptions, MaterializationSpec
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np

@dataclass
class CellMeasurement:
    \"\"\"Per-cell measurement data.\"\"\"
    slice_index: int
    cell_id: int
    x_position: float
    y_position: float
    area: float
    intensity: float


@dataclass
class SliceSummary:
    \"\"\"Per-slice summary statistics.\"\"\"
    slice_index: int
    cell_count: int
    total_area: float
    mean_intensity: float


@numpy
@artifact_outputs(
    ("cell_measurements", MaterializationSpec(CsvOptions(filename_suffix="_cells.csv"))),
    ("slice_summaries", MaterializationSpec(
        JsonOptions(filename_suffix=".json", wrap_list=True),
        CsvOptions(filename_suffix="_details.csv"),
        primary=0,
    ))
)
def analyze_cells(
    image,
    intensity_threshold: float = 100.0,
    min_area: int = 10
) -> Tuple[np.ndarray, List[CellMeasurement], List[SliceSummary]]:
    \"\"\"
    Multi-output analysis: cell details + slice summaries.

    This demonstrates:
    - Multiple @artifact_outputs with different materializers
    - CSV for detailed per-cell data
    - Dual (JSON+CSV) for summary statistics

    Args:
        image: Input image (C, Y, X)
        intensity_threshold: Threshold for cell detection
        min_area: Minimum cell area in pixels

    Returns:
        Tuple of (processed_image, cell_measurements, slice_summaries)
    \"\"\"
    from scipy import ndimage
    
    cell_measurements = []
    slice_summaries = []
    
    for slice_idx, slice_data in enumerate(image):
        # Simple threshold + connected components
        binary = slice_data > intensity_threshold
        labeled, num_features = ndimage.label(binary)
        
        slice_cells = []
        for cell_id in range(1, num_features + 1):
            mask = labeled == cell_id
            area = np.sum(mask)
            
            if area >= min_area:
                # Get centroid
                y_coords, x_coords = np.where(mask)
                x_pos = float(np.mean(x_coords))
                y_pos = float(np.mean(y_coords))
                intensity = float(np.mean(slice_data[mask]))
                
                cell_measurements.append(CellMeasurement(
                    slice_index=slice_idx,
                    cell_id=len(slice_cells),
                    x_position=x_pos,
                    y_position=y_pos,
                    area=float(area),
                    intensity=intensity
                ))
                slice_cells.append(area)
        
        # Slice summary
        slice_summaries.append(SliceSummary(
            slice_index=slice_idx,
            cell_count=len(slice_cells),
            total_area=float(sum(slice_cells)),
            mean_intensity=float(np.mean(slice_data)) if slice_data.size > 0 else 0.0
        ))
    
    return image.copy(), cell_measurements, slice_summaries
"""


def get_analysis_template(memory_type: str = "numpy") -> str:
    """
    Get an analysis template for the specified memory type.

    Args:
        memory_type: Currently only 'numpy' supported for analysis templates

    Returns:
        Template string with analysis pattern
    """
    if memory_type == "numpy":
        return NUMPY_ANALYSIS_TEMPLATE
    else:
        # For other memory types, return the basic analysis template
        # Users can adapt the numpy pattern
        return NUMPY_ANALYSIS_TEMPLATE


def get_multi_output_template() -> str:
    """Get template demonstrating multiple special outputs."""
    return NUMPY_DUAL_OUTPUT_TEMPLATE
