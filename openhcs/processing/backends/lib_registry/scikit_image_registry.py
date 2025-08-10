"""
Clean Scikit-Image Registry Implementation

Implements clean abstraction with internal library-specific logic.
All scikit-image-specific details (array compliance, channel_axis, etc.)
are handled internally without leaking into the ABC.
"""
from __future__ import annotations

import inspect
import numpy as np
from functools import wraps
from typing import Tuple, Callable, List, Any, Dict

from openhcs.constants import MemoryType
from openhcs.core.utils import optional_import
from .unified_registry import LibraryRegistryBase, ProcessingContract, FunctionMetadata

skimage = optional_import("skimage")


class SkimageRegistry(LibraryRegistryBase):
    """Clean scikit-image registry with internal array compliance logic."""

    # Library-specific exclusions (uses common ones)
    EXCLUSIONS = LibraryRegistryBase.COMMON_EXCLUSIONS

    # Modules to scan for functions
    MODULES_TO_SCAN = ['filters', 'morphology', 'segmentation', 'feature',
                       'measure', 'transform', 'restoration', 'exposure']

    # Memory type for this registry
    MEMORY_TYPE = MemoryType.NUMPY.value

    # Float dtype for this registry
    FLOAT_DTYPE = np.float32

    def __init__(self):
        super().__init__("skimage")

    # ===== ESSENTIAL ABC METHODS =====
    def get_library_version(self) -> str:
        return skimage.__version__

    def is_library_available(self) -> bool:
        return skimage is not None

    def get_library_object(self):
        return skimage

    # ===== HOOK IMPLEMENTATIONS =====
    def _create_array(self, shape: Tuple[int, ...], dtype):
        return np.random.rand(*shape).astype(dtype)

    def _check_first_parameter(self, first_param, func_name: str) -> bool:
        return (first_param.annotation in [np.ndarray, "np.ndarray", "ndarray"] or
               first_param.name.lower() in {'image', 'input', 'array', 'img'})

    def _preprocess_input(self, image, func_name: str):
        return image  # No preprocessing needed for scikit-image

    def _postprocess_output(self, result, original_image, func_name: str):
        # ProcessingContract system handles dimensional behavior - no categorization needed
        return result

    # ===== LIBRARY-SPECIFIC IMPLEMENTATIONS =====
    def _generate_function_name(self, name: str, module_name: str) -> str:
        """Generate function name with module prefix."""
        return f"{module_name}.{name}"

    def _stack_2d_results(self, func, test_3d):
        """Stack 2D results using NumPy."""
        return np.stack([func(test_3d[z]) for z in range(test_3d.shape[0])])

    def _arrays_close(self, arr1, arr2):
        """Compare arrays using NumPy."""
        return np.allclose(arr1, arr2, rtol=1e-5, atol=1e-8)

    def _expand_2d_to_3d(self, array_2d):
        """Expand 2D array to 3D using NumPy expansion."""
        return np.expand_dims(array_2d, 0)
