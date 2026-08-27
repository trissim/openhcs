"""
Clean Pyclesperanto Registry Implementation

Implements clean abstraction with internal library-specific logic.
All pyclesperanto-specific details (dtype conversions, Z-parameters, etc.)
are handled internally without leaking into the ABC.
"""

from __future__ import annotations

import inspect
from enum import Enum
from typing import List, Tuple

import numpy as np

from openhcs.constants import MemoryType
from openhcs.utils.import_utils import optional_import_placeholder

from .unified_registry import LibraryRegistryBase, RuntimeTestingRegistryBase


class PyclesperantoRuntimeParameter(Enum):
    """pyclesperanto parameters whose defaults belong to runtime execution."""

    OUTPUT_BUFFER = "output_image"
    OUTPUT_MATRIX_BUFFER = "output_image_matrix"
    DEVICE = "device"

    def declared_name(self, signature: inspect.Signature) -> str | None:
        """Return this parameter name when the callable declares it."""

        return self.value if self.value in signature.parameters else None


class PyclesperantoRegistry(RuntimeTestingRegistryBase):
    """Clean pyclesperanto registry with internal library-specific logic."""

    # Registry name for auto-registration
    _registry_name = "pyclesperanto"

    # Library-specific exclusions extending common ones
    EXCLUSIONS = LibraryRegistryBase.COMMON_EXCLUSIONS | {
        "push_zyx",
        "pull_zyx",
        "create_zyx",
        "set_wait_for_kernel_finish",
        "get_device",
        "select_device",
        "list_available_devices",
    }

    # Modules to scan for functions
    MODULES_TO_SCAN = [""]  # Pyclesperanto functions are in main namespace

    # Memory type for this registry
    MEMORY_TYPE = MemoryType.PYCLESPERANTO.value

    # Float dtype for this registry
    FLOAT_DTYPE = np.float32

    def __init__(self):
        super().__init__("pyclesperanto")
        self._pyclesperanto = optional_import_placeholder("pyclesperanto")
        # Internal constants for dtype handling
        self._BINARY_FUNCTIONS = {"binary_infsup", "binary_supinf"}
        self._UINT8_FUNCTIONS = {"mode", "mode_box", "mode_sphere"}
        self._IMAGE_PARAM_NAMES = {
            "src",
            "source",
            "image",
            "input",
            "src1",
            "input_image",
            "input_image0",
        }

    # ===== ESSENTIAL ABC METHODS =====
    def get_library_version(self) -> str:
        return self._pyclesperanto.__version__

    def is_library_available(self) -> bool:
        return bool(self._pyclesperanto)

    def get_library_object(self):
        return self._pyclesperanto

    def get_module_patterns(self) -> List[str]:
        """Get module patterns for pyclesperanto (includes 'cle' alternative)."""
        return ["pyclesperanto", "cle"]

    # ===== HOOK IMPLEMENTATIONS =====
    def _create_array(self, shape: Tuple[int, ...], dtype):
        return np.random.rand(*shape).astype(dtype)

    def _check_first_parameter(self, first_param, func_name: str) -> bool:
        return (
            first_param.name.lower() in self._IMAGE_PARAM_NAMES
            and first_param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )

    def runtime_owned_parameter_names(
        self,
        signature: inspect.Signature,
    ) -> tuple[str, ...]:
        """Keep output allocation and device selection at library defaults."""

        return tuple(
            name
            for parameter in PyclesperantoRuntimeParameter
            for name in (parameter.declared_name(signature),)
            if name is not None
        )

    def _preprocess_input(self, image, func_name: str):
        return self._convert_input_dtype(image, func_name)

    def _postprocess_output(self, result, original_image, func_name: str):
        return self._convert_output_dtype(result, original_image.dtype, func_name)

    # ===== LIBRARY-SPECIFIC HELPER METHODS =====
    def _convert_input_dtype(self, image, func_name):
        """Internal dtype conversion logic."""
        if func_name in self._BINARY_FUNCTIONS:
            return ((image > 0.5) * 255).astype(np.uint8)
        elif func_name in self._UINT8_FUNCTIONS:
            return (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return image

    def _convert_output_dtype(self, result, original_dtype, func_name):
        """Internal output dtype conversion."""
        if func_name in self._BINARY_FUNCTIONS or func_name in self._UINT8_FUNCTIONS:
            if result.dtype != original_dtype:
                if result.dtype == np.uint8 and original_dtype == np.float32:
                    return result.astype(np.float32) / 255.0
                elif result.dtype == np.bool_ and original_dtype == np.float32:
                    return result.astype(np.float32)
        return result

    # ===== LIBRARY-SPECIFIC IMPLEMENTATIONS =====
    def _stack_2d_results(self, func, test_3d):
        """Stack 2D results using CLE."""
        results = [func(test_3d[z]) for z in range(test_3d.shape[0])]
        return self._pyclesperanto.concatenate_along_z(*results)

    def _arrays_close(self, arr1, arr2):
        """Compare arrays using CLE."""
        return np.allclose(arr1.get(), arr2.get(), rtol=1e-5, atol=1e-8)
