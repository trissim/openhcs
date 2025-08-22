"""
Clean CuPy Registry Implementation

Implements clean abstraction with internal library-specific logic.
All CuPy-specific details (GPU handling, CuCIM integration, etc.)
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

cp = optional_import("cupy")
cucim = optional_import("cucim")
cucim_skimage = optional_import("cucim.skimage")


class CupyRegistry(LibraryRegistryBase):
    """Clean CuPy registry with internal GPU handling logic."""

    # Library-specific exclusions (uses common ones)
    EXCLUSIONS = LibraryRegistryBase.COMMON_EXCLUSIONS

    # Modules to scan for functions
    MODULES_TO_SCAN = ['filters', 'morphology', 'measure', 'segmentation',
                       'feature', 'restoration', 'transform', 'exposure',
                       'color', 'util']

    # Memory type for this registry
    MEMORY_TYPE = MemoryType.CUPY.value

    # Float dtype for this registry
    FLOAT_DTYPE = cp.float32

    def __init__(self):
        super().__init__("cupy")

    # ===== ESSENTIAL ABC METHODS =====
    def get_library_version(self) -> str:
        return cucim.__version__

    def is_library_available(self) -> bool:
        return cp is not None and cucim_skimage is not None

    def get_library_object(self):
        return cucim_skimage

    # ===== HOOK IMPLEMENTATIONS =====
    def _create_array(self, shape: Tuple[int, ...], dtype):
        try:
            return cp.random.rand(*shape).astype(dtype)
        except Exception as e:
            # If CUDA initialization fails, raise a more descriptive error
            raise RuntimeError(f"CUDA initialization failed during CuPy array creation: {e}") from e

    def _check_first_parameter(self, first_param, func_name: str) -> bool:
        return first_param.name.lower() in {'image', 'input', 'array', 'img'}

    def _preprocess_input(self, image, func_name: str):
        return image  # No preprocessing needed for CuPy

    def _postprocess_output(self, result, original_image, func_name: str):
        # ProcessingContract system handles dimensional behavior - no categorization needed
        return result

    # ===== LIBRARY-SPECIFIC IMPLEMENTATIONS =====
    def _generate_function_name(self, name: str, module_name: str) -> str:
        """Generate function name - original for filters, prefixed for others."""
        return name if module_name == 'filters' else f"{module_name}_{name}"

    def _generate_tags(self, func_name: str) -> List[str]:
        """Generate tags with GPU tag."""
        tags = func_name.lower().replace("_", " ").split()
        tags.append("gpu")
        return tags

    def _stack_2d_results(self, func, test_3d):
        """Stack 2D results using CuPy."""
        results = [func(test_3d[z]) for z in range(test_3d.shape[0])]
        return cp.stack(results)

    def _arrays_close(self, arr1, arr2):
        """Compare arrays using CuPy."""
        return np.allclose(arr1.get(), arr2.get(), rtol=1e-5, atol=1e-8)

    def _expand_2d_to_3d(self, array_2d):
        """Expand 2D array to 3D using CuPy expansion."""
        return array_2d[None, ...]
