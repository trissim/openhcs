"""
Clean CuPy Registry Implementation

Implements clean abstraction with internal library-specific logic.
All CuPy-specific details (GPU handling, CuCIM integration, etc.)
are handled internally without leaking into the ABC.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from openhcs.constants import MemoryType
from openhcs.utils.import_utils import optional_import_placeholder

from .unified_registry import LibraryRegistryBase, RuntimeTestingRegistryBase


class CupyRegistry(RuntimeTestingRegistryBase):
    """Clean CuPy registry with internal GPU handling logic."""

    # Registry name for auto-registration
    _registry_name = "cupy"

    # Library-specific exclusions (uses common ones)
    EXCLUSIONS = LibraryRegistryBase.COMMON_EXCLUSIONS

    # Modules to scan for functions
    MODULES_TO_SCAN = [
        "filters",
        "morphology",
        "measure",
        "segmentation",
        "feature",
        "restoration",
        "transform",
        "exposure",
        "color",
        "util",
    ]

    # Memory type for this registry
    MEMORY_TYPE = MemoryType.CUPY.value

    # Float dtype for this registry
    FLOAT_DTYPE = np.float32

    def __init__(self):
        super().__init__("cupy")
        self._cupy = optional_import_placeholder("cupy")
        self._cucim = optional_import_placeholder("cucim")
        self._cucim_skimage = optional_import_placeholder("cucim.skimage")

    # ===== ESSENTIAL ABC METHODS =====
    def get_library_version(self) -> str:
        return self._cucim.__version__

    def is_library_available(self) -> bool:
        return bool(self._cupy) and bool(self._cucim_skimage)

    def get_library_object(self):
        return self._cucim_skimage

    def get_module_patterns(self) -> List[str]:
        """Get module patterns for CuPy (includes cucim patterns)."""
        return ["cupy", "cucim"]

    def get_display_name(self) -> str:
        """Get proper display name for CuPy."""
        return "CuPy"

    def _warmup_library(self) -> None:
        """
        Ensure CuPy can create basic arrays before registry discovery.

        This mirrors GUI behavior (PyQtGraph imports CuPy early) so detached
        interpreters fail fast if CUDA libraries are missing.
        """
        if not self._cupy or not self._cucim_skimage:
            raise RuntimeError("CuPy or CuCIM not available for warm-up")

        try:
            _ = self._cupy.zeros((1,), dtype=self.FLOAT_DTYPE)
            self._cupy.cuda.runtime.deviceSynchronize()
        except Exception as exc:
            raise RuntimeError(f"CuPy warm-up failed: {exc}") from exc

    # ===== HOOK IMPLEMENTATIONS =====
    def _create_array(self, shape: Tuple[int, ...], dtype):
        try:
            return self._cupy.random.rand(*shape).astype(dtype)
        except Exception as e:
            # If CUDA initialization fails, raise a more descriptive error
            raise RuntimeError(
                f"CUDA initialization failed during CuPy array creation: {e}"
            ) from e

    def _check_first_parameter(self, first_param, func_name: str) -> bool:
        return first_param.name.lower() in {"image", "input", "array", "img"}

    def _preprocess_input(self, image, func_name: str):
        return image  # No preprocessing needed for CuPy

    def _postprocess_output(self, result, original_image, func_name: str):
        # ProcessingContract system handles dimensional behavior - no categorization needed
        return result

    # ===== LIBRARY-SPECIFIC IMPLEMENTATIONS =====
    def _generate_function_name(self, name: str, module_name: str) -> str:
        """Generate function name - original for filters, prefixed for others."""
        return name if module_name == "filters" else f"{module_name}_{name}"

    def _generate_tags(self, func_name: str) -> List[str]:
        """Generate tags with GPU tag."""
        tags = func_name.lower().replace("_", " ").split()
        tags.append("gpu")
        return tags

    def _stack_2d_results(self, func, test_3d):
        """Stack 2D results using CuPy."""
        results = [func(test_3d[z]) for z in range(test_3d.shape[0])]
        return self._cupy.stack(results)

    def _arrays_close(self, arr1, arr2):
        """Compare arrays using CuPy."""
        return np.allclose(arr1.get(), arr2.get(), rtol=1e-5, atol=1e-8)
