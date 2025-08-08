"""openhcs.plugin.pycle_registry
--------------------------------
Static discovery & registration of *pyclesperanto* kernels for the
**OpenHCS** declarative execution engine.

This helper is imported at *compile time* (not at run‑time) so that every
GPU operation is validated *before* the execution context is frozen.

Highlights
~~~~~~~~~~
*   Zero GPU work – purely uses ``inspect`` & naming heuristics.
*   Classifies each function into one of three contracts that matter to
    OpenHCS:

    * ``SLICE_SAFE``        – never couples neighbouring Z‑slices (or can
      be forced not to by setting a ``*_z`` parameter to 0).
    * ``CROSS_Z``           – fuses slices whenever the input stack has
      ``shape[0] > 1`` (e.g. label/ watershed / voronoi). Works fine on
      single‑slice input.
    * ``DIM_CHANGE``        – changes the dimensionality or returns a
      scalar/table (e.g. projections, statistics). Rejected for the
      *img→img* contract.

*   Provides a single public function ``register_pycle_ops()`` that either
    (a) returns an organised registry **or** (b) auto‑calls the OpenHCS
    registrar if available.

Usage
~~~~~
>>> from openhcs.plugin.pycle_registry import register_pycle_ops
>>> registry = register_pycle_ops()           # during compile phase

----------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import textwrap
import numpy as np
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional

try:
    import pyclesperanto as cle
except ModuleNotFoundError as e:  # pragma: no cover – won't happen in CI
    raise ImportError("pyclesperanto must be installed before importing "
                      "openhcs.plugin.pycle_registry") from e

# ---------------------------------------------------------------------------
# 1. Contracts & metadata
# ---------------------------------------------------------------------------

class Contract(Enum):
    """Semantic contracts recognised by OpenHCS."""

    SLICE_SAFE = auto()   # keeps each Z‑slice independent (sigma_z or radius_z can be 0)
    CROSS_Z = auto()      # 3‑D connectivity – fuses neighbouring slices
    DIM_CHANGE = auto()   # output is scalar, table or different shape ➔ excluded


# Import DtypeConversion from centralized location
try:
    from openhcs.core.memory.decorators import DtypeConversion, _scale_and_convert_pyclesperanto
except ImportError:
    # Fallback for standalone usage
    from enum import Enum
    import numpy as np

    class DtypeConversion(Enum):
        """Data type conversion modes for pyclesperanto functions."""
        PRESERVE_INPUT = "preserve"
        NATIVE_OUTPUT = "native"
        UINT8 = "uint8"
        UINT16 = "uint16"
        INT16 = "int16"
        INT32 = "int32"
        FLOAT32 = "float32"
        FLOAT64 = "float64"

        @property
        def numpy_dtype(self):
            dtype_map = {
                self.UINT8: np.uint8,
                self.UINT16: np.uint16,
                self.INT16: np.int16,
                self.INT32: np.int32,
                self.FLOAT32: np.float32,
                self.FLOAT64: np.float64,
            }
            return dtype_map.get(self, None)

    def _scale_and_convert_pyclesperanto(result, target_dtype):
        """Fallback scaling function."""
        return result.astype(target_dtype)


@dataclass(frozen=True)
class OpMeta:
    """Compile‑time metadata for a pyclesperanto kernel."""

    name: str
    func: Callable
    contract: Contract
    has_z_kw: bool
    doc: str = ""
    tags: List[str] = field(default_factory=list)

    # Convenience helpers ----------------------------------------------------
    @property
    def slice_safe(self) -> bool:  # noqa: D401 – property is short
        return self.contract is Contract.SLICE_SAFE

    @property
    def cross_z(self) -> bool:
        return self.contract is Contract.CROSS_Z

    @property
    def dim_change(self) -> bool:
        return self.contract is Contract.DIM_CHANGE


# ---------------------------------------------------------------------------
# 2. Heuristic patterns (kept in one place for easy tweaking)
# ---------------------------------------------------------------------------

# Operators that *change* dimensionality or produce non‑image output
_DIM_DROP_PAT = (
    "projection",                 # z‑ or dimension projections
    "_of_all_pixels",             # statistics reductions
    "statistics_of_",            # DataFrame/table
    "distance_matrix_to_",       # mesh utilities
)

# Operators whose neighbourhood must reach across Z – no sigma_z/radius_z
_CROSS_Z_PAT = (
    "connected", "watershed", "distance", "label",
    "voronoi", "skeleton", "mesh", "morphological_geodesic",  # etc.
)

_IMAGE_PARAM_NAMES = {"src", "source", "image", "input", "src1", "input_image", "input_image0"}


# ---------------------------------------------------------------------------
# 3. Static discovery
# ---------------------------------------------------------------------------

def _is_image_param(p: inspect.Parameter) -> bool:
    return p.name.lower() in _IMAGE_PARAM_NAMES and p.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def _discover() -> Dict[str, OpMeta]:
    """Inspect the *cle* namespace once and classify every callable."""

    registry: Dict[str, OpMeta] = {}

    for name in dir(cle):
        if name.startswith("_"):
            continue  # private
        fn = getattr(cle, name)
        if not callable(fn):
            continue

        # Skip classes and types
        if inspect.isclass(fn) or isinstance(fn, type):
            continue

        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            continue  # Skip functions without inspectable signatures
        params = list(sig.parameters.values())
        if not params or not _is_image_param(params[0]):
            continue  # first positional arg is not an image

        lname = name.lower()
        has_z_kw = any("z" in p.name and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                        for p in params)

        if any(pat in lname for pat in _DIM_DROP_PAT):
            contract = Contract.DIM_CHANGE
        elif any(pat in lname for pat in _CROSS_Z_PAT):
            contract = Contract.CROSS_Z
        elif has_z_kw:
            contract = Contract.SLICE_SAFE
        else:
            contract = Contract.SLICE_SAFE  # default safe (per‑pixel)

        doc_lines = textwrap.dedent(fn.__doc__ or "").splitlines()
        first_line_doc = doc_lines[0] if doc_lines else ""
        registry[name] = OpMeta(
            name=name,
            func=fn,
            contract=contract,
            has_z_kw=has_z_kw,
            doc=first_line_doc,
            tags=[t for t in lname.replace("_", " ").split() if t],
        )

    return registry


_REGISTRY: Optional[Dict[str, OpMeta]] = None  # module‑level cache


# ---------------------------------------------------------------------------
# 4. Persistent Cache System (following scikit-image pattern)
# ---------------------------------------------------------------------------

def _get_pyclesperanto_version() -> str:
    """Get pyclesperanto version for cache validation."""
    try:
        import pyclesperanto as cle
        return cle.__version__
    except Exception:
        return "unknown"


def _extract_pyclesperanto_cache_data(meta: OpMeta) -> Dict[str, str]:
    """Extract cacheable data from OpMeta object."""
    return {
        'name': meta.name,
        'module': meta.func.__module__,
        'contract': meta.contract.value
    }


def _save_pyclesperanto_metadata(registry: Dict[str, OpMeta]) -> None:
    """Save pyclesperanto function metadata to cache."""
    from openhcs.processing.backends.analysis.cache_utils import save_library_metadata

    # Only cache functions that will be decorated (skip UNKNOWN/DIM_CHANGE)
    filtered_registry = {
        name: meta for name, meta in registry.items()
        if meta.contract in [Contract.SLICE_SAFE, Contract.CROSS_Z]
    }

    save_library_metadata(
        library_name="pyclesperanto",
        registry=filtered_registry,
        get_version_func=_get_pyclesperanto_version,
        extract_cache_data_func=_extract_pyclesperanto_cache_data
    )


def _load_pyclesperanto_metadata() -> Optional[Dict[str, Dict[str, str]]]:
    """Load pyclesperanto function metadata from cache with validation."""
    from openhcs.processing.backends.analysis.cache_utils import load_library_metadata
    return load_library_metadata("pyclesperanto", _get_pyclesperanto_version)


def clear_pyclesperanto_cache() -> None:
    """Clear the pyclesperanto metadata cache to force rebuild on next startup."""
    from openhcs.processing.backends.analysis.cache_utils import clear_library_cache
    clear_library_cache("pyclesperanto")


def _get_pyclesperanto_function(module_path: str, func_name: str):
    """Get pyclesperanto function object from module path and name."""
    try:
        import pyclesperanto as cle

        # Handle pyclesperanto module structure
        if module_path.startswith('pyclesperanto'):
            # Get the function from pyclesperanto
            if hasattr(cle, func_name):
                return getattr(cle, func_name)

        return None
    except Exception:
        return None


def _register_pyclesperanto_from_cache() -> bool:
    """Register pyclesperanto functions using cached metadata. Returns True if used."""
    from openhcs.processing.backends.analysis.cache_utils import register_functions_from_cache
    from openhcs.processing.func_registry import _register_function
    from openhcs.constants import MemoryType

    # Load cached metadata
    cached_metadata = _load_pyclesperanto_metadata()
    if not cached_metadata:
        return False

    def register_pyclesperanto_function(original_func, func_name: str, memory_type: str):
        """Register a pyclesperanto function with unified decoration."""
        from openhcs.processing.func_registry import _apply_unified_decoration

        wrapper_func = _apply_unified_decoration(
            original_func=original_func,
            func_name=func_name,
            memory_type=MemoryType.PYCLESPERANTO,
            create_wrapper=True  # pyclesperanto needs dtype preservation
        )

        _register_function(wrapper_func, MemoryType.PYCLESPERANTO.value)

    # Register functions from cache
    register_functions_from_cache(
        library_name="pyclesperanto",
        cached_metadata=cached_metadata,
        get_function_func=_get_pyclesperanto_function,
        register_function_func=register_pyclesperanto_function,
        memory_type=MemoryType.PYCLESPERANTO.value
    )
    return True


# ---------------------------------------------------------------------------
# 5. Public API for OpenHCS – called in compile phase
# ---------------------------------------------------------------------------

def build_pycle_registry(refresh: bool = False) -> Dict[str, OpMeta]:
    """Return a *cached* registry mapping kernel‑name → :class:`OpMeta`."""
    global _REGISTRY  # noqa: PLW0603 – intentional module cache
    if refresh or _REGISTRY is None:
        _REGISTRY = _discover()
    return _REGISTRY


def register_pycle_ops(*, auto_register: bool = True):
    """Build the registry *and* (optionally) register array-in/array-out functions
    with the OpenHCS function catalogue using @pyclesperanto decorator.

    Parameters
    ----------
    auto_register:
        If ``True`` (default) try to import OpenHCS memory decorators and register
        all array-in/array-out functions with @pyclesperanto decorator.
        If the import fails nothing is registered – you still get the
        registry back so the caller may handle registration.
    """
    import inspect

    registry = build_pycle_registry()

    if not auto_register:
        return registry

    try:
        from openhcs.core.memory.decorators import pyclesperanto
        from openhcs.processing.func_registry import register_function
    except ImportError:
        return registry  # silently skip – OpenHCS not on PYTHONPATH

    registered_count = 0
    skipped_count = 0

    for meta in registry.values():
        # Check if function returns array (array-in/array-out pattern)
        try:
            sig = inspect.signature(meta.func)
            return_annotation = str(sig.return_annotation)

            # Skip functions that don't return arrays
            if not ('Array' in return_annotation or 'ndarray' in return_annotation):
                skipped_count += 1
                continue

        except Exception:
            skipped_count += 1
            continue

        # Skip dimension-changing functions (they break array chains)
        # OpenHCS requires array-in/array-out with consistent 3D array returns
        if meta.dim_change:
            skipped_count += 1
            continue

        try:
            # Apply @pyclesperanto decorator
            decorated_func = pyclesperanto(meta.func)

            # Register with OpenHCS function registry
            register_function(
                decorated_func,
                backend="pyclesperanto",
                contract=meta.contract.name.lower(),  # e.g. "slice_safe"
                tags=meta.tags,
                doc=meta.doc,
            )
            registered_count += 1

        except Exception as e:
            print(f"Warning: Failed to register {meta.name}: {e}")
            skipped_count += 1

    print(f"✅ Registered {registered_count} pyclesperanto functions with @pyclesperanto")
    print(f"⚠️  Skipped {skipped_count} functions (non-array returns or dim_change - OpenHCS requires array-in/array-out)")

    return registry


# ---------------------------------------------------------------------------
# 5. If executed as a script, emit a summary for debugging
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover – manual use
    from pprint import pprint

    r = register_pycle_ops(auto_register=False)
    summary = {
        "slice_safe": [n for n, m in r.items() if m.slice_safe],
        "cross_z":    [n for n, m in r.items() if m.cross_z],
        "dim_change": [n for n, m in r.items() if m.dim_change],
    }
    pprint({k: len(v) for k, v in summary.items()})


# _scale_and_convert_pyclesperanto is now imported from decorators module


def _create_pyclesperanto_array_compliant_wrapper(original_func, func_name):
    """
    Create a wrapper that ensures array-in/array-out compliance and dtype preservation for pyclesperanto functions.

    All OpenHCS functions must:
    1. Take 3D pyclesperanto array as first argument
    2. Return 3D pyclesperanto array as first output
    3. Additional outputs (values, coordinates) as 2nd, 3rd, etc. returns
    4. Preserve input dtype when appropriate
    """
    from functools import wraps

    @wraps(original_func)
    def pyclesperanto_dtype_and_slice_preserving_wrapper(image_3d, *args, slice_by_slice: bool = False, dtype_conversion: DtypeConversion = DtypeConversion.PRESERVE_INPUT, **kwargs):
        try:
            # Store original dtype for preservation
            original_dtype = image_3d.dtype

            # Handle slice_by_slice processing for 3D arrays using OpenHCS stack utilities
            if slice_by_slice and hasattr(image_3d, 'ndim') and image_3d.ndim == 3:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices

                # Process each slice individually
                slices = unstack_slices(image_3d)
                processed_slices = []

                for slice_2d in slices:
                    # Apply function to 2D slice
                    result_slice = original_func(slice_2d, *args, **kwargs)
                    processed_slices.append(result_slice)

                # Stack results back to 3D
                result = stack_slices(processed_slices)
            else:
                # Normal 3D processing
                result = original_func(image_3d, *args, **kwargs)

            # Check if result is 2D and needs expansion to 3D (like CuPy wrapper does)
            if hasattr(result, 'ndim') and result.ndim == 2:
                # Expand 2D result to 3D single slice
                # Use pyclesperanto's concatenate_along_z and slice to create single 3D slice
                try:
                    # Concatenate with itself to create 3D, then take first slice
                    temp_3d = cle.concatenate_along_z(result, result)  # Creates (2, Y, X)
                    result = temp_3d[0:1, :, :]  # Take first slice to get (1, Y, X)
                except Exception:
                    # If expansion fails, return original 2D result
                    # This maintains backward compatibility
                    pass

            # Apply dtype conversion based on enum value
            if hasattr(result, 'dtype') and hasattr(result, 'shape'):
                if dtype_conversion == DtypeConversion.PRESERVE_INPUT:
                    # Preserve input dtype
                    if result.dtype != original_dtype:
                        return _scale_and_convert_pyclesperanto(result, original_dtype)
                    return result

                elif dtype_conversion == DtypeConversion.NATIVE_OUTPUT:
                    # Return pyclesperanto's native output dtype
                    return result

                else:
                    # Force specific dtype
                    target_dtype = dtype_conversion.numpy_dtype
                    if target_dtype is not None and result.dtype != target_dtype:
                        return _scale_and_convert_pyclesperanto(result, target_dtype)
                    return result
            else:
                # Non-array result, return as-is
                return result

        except Exception as e:
            # If anything goes wrong, fall back to original function
            return original_func(image_3d, *args, **kwargs)

    # Update function signature to include new parameters
    try:
        import inspect
        original_sig = inspect.signature(original_func)
        new_params = list(original_sig.parameters.values())

        # Add slice_by_slice parameter
        slice_param = inspect.Parameter(
            'slice_by_slice',
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool
        )
        new_params.append(slice_param)

        # Add dtype_conversion parameter
        dtype_param = inspect.Parameter(
            'dtype_conversion',
            inspect.Parameter.KEYWORD_ONLY,
            default=DtypeConversion.PRESERVE_INPUT,
            annotation=DtypeConversion
        )
        new_params.append(dtype_param)

        new_sig = original_sig.replace(parameters=new_params)
        pyclesperanto_dtype_and_slice_preserving_wrapper.__signature__ = new_sig

        # Set type annotations manually for get_type_hints() compatibility
        pyclesperanto_dtype_and_slice_preserving_wrapper.__annotations__ = getattr(original_func, '__annotations__', {}).copy()
        pyclesperanto_dtype_and_slice_preserving_wrapper.__annotations__['slice_by_slice'] = bool
        pyclesperanto_dtype_and_slice_preserving_wrapper.__annotations__['dtype_conversion'] = DtypeConversion

    except Exception:
        # If signature modification fails, continue without it
        pass

    # Update docstring to mention additional parameters - following scikit-image/CuPy pattern
    original_doc = pyclesperanto_dtype_and_slice_preserving_wrapper.__doc__ or ""
    additional_doc = """

    Additional OpenHCS Parameters
    -----------------------------
    slice_by_slice : bool, optional (default: False)
        If True, process 3D arrays slice-by-slice to avoid cross-slice contamination.
        If False, use original 3D behavior. Recommended for edge detection functions
        on stitched microscopy data to prevent artifacts at field boundaries.

    dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
        Controls output data type conversion:

        - PRESERVE_INPUT: Keep input dtype (uint16 → uint16)
        - NATIVE_OUTPUT: Use pyclesperanto's native output (often float32)
        - UINT8: Force 8-bit unsigned integer (0-255 range)
        - UINT16: Force 16-bit unsigned integer (microscopy standard)
        - INT16: Force 16-bit signed integer
        - INT32: Force 32-bit signed integer
        - FLOAT32: Force 32-bit float (GPU performance)
        - FLOAT64: Force 64-bit float (maximum precision)

        Examples:
            # Preserve input type
            result = func(uint16_image, dtype_conversion=DtypeConversion.PRESERVE_INPUT)

            # Force float32 for performance
            result = func(float64_image, dtype_conversion=DtypeConversion.FLOAT32)

            # Force uint16 for microscopy pipeline
            result = func(float32_image, dtype_conversion=DtypeConversion.UINT16)
    """
    pyclesperanto_dtype_and_slice_preserving_wrapper.__doc__ = original_doc + additional_doc

    return pyclesperanto_dtype_and_slice_preserving_wrapper


def _get_z_parameter_guidance(func: Callable) -> str:
    """
    Get specific guidance for Z parameters based on their actual types.

    Different pyclesperanto functions use different parameter types:
    - sigma_z: Set to 0.0 for slice-by-slice processing
    - radius_z: Set to 0 for slice-by-slice processing

    Args:
        func: The pyclesperanto function to analyze

    Returns:
        Specific guidance string for the function's Z parameters
    """
    try:
        import inspect

        sig = inspect.signature(func)
        z_params = []
        sigma_params = []
        radius_params = []

        # Categorize Z parameters by type
        for param_name, param in sig.parameters.items():
            if 'z' in param_name.lower():
                z_params.append(param_name)
                if 'sigma' in param_name.lower():
                    sigma_params.append(param_name)
                elif 'radius' in param_name.lower():
                    radius_params.append(param_name)

        if not z_params:
            return ""

        # Build specific guidance based on parameter types
        guidance_parts = []

        if sigma_params:
            sigma_list = ", ".join(sigma_params)
            guidance_parts.append(f"{sigma_list}=0.0 for slice-by-slice")

        if radius_params:
            radius_list = ", ".join(radius_params)
            guidance_parts.append(f"{radius_list}=0 for slice-by-slice")

        # Handle any other Z parameters generically
        other_params = [p for p in z_params if p not in sigma_params and p not in radius_params]
        if other_params:
            other_list = ", ".join(other_params)
            guidance_parts.append(f"{other_list}=0 for slice-by-slice")

        if guidance_parts:
            guidance = " Has Z-axis parameters: " + ", ".join(guidance_parts) + "."
        else:
            # Fallback to generic guidance
            param_list = ", ".join(z_params)
            guidance = f" Has Z-axis parameters ({param_list}) - consult function docs for slice-by-slice values."

        return guidance

    except Exception:
        # Fallback to generic guidance if analysis fails
        return " Has Z-axis parameters - consult function documentation for slice-by-slice values."


def _enhance_docstring_with_contract_info(func: Callable, meta: OpMeta) -> None:
    """
    Enhance function docstring with contract information following the OpenHCS pattern.

    This follows the same pattern as CuPy and scikit-image backends for consistency.

    Args:
        func: The function to enhance
        meta: OpMeta containing contract information
    """
    # Get original docstring
    original_doc = func.__doc__ or ""

    # Create contract description
    contract_descriptions = {
        Contract.SLICE_SAFE: "Processes each Z-slice independently by default. Safe for stitched microscopy data.",
        Contract.CROSS_Z: "Uses 3D connectivity across Z-slices. May cause artifacts at field boundaries in stitched data.",
        Contract.DIM_CHANGE: "Changes output dimensionality (e.g., projections, statistics)."
    }

    contract_desc = contract_descriptions.get(meta.contract, f"Processing behavior: {meta.contract.name.lower()}")

    # Add Z-parameter information if available
    z_param_info = ""
    if meta.has_z_kw:
        z_param_info = _get_z_parameter_guidance(meta.func)

    # Create OpenHCS information section following existing pattern
    openhcs_doc = f"""

    OpenHCS Processing Information
    -----------------------------
    Processing Behavior: {meta.contract.name.lower()}
        {contract_desc}{z_param_info}

    Backend: pyclesperanto (GPU-accelerated via OpenCL)
    """

    # Append to original docstring
    func.__doc__ = original_doc + openhcs_doc


def _register_pycle_ops_direct() -> None:
    """
    Direct decoration of pyclesperanto functions with caching support.

    Checks cache first for fast registration, falls back to full discovery if needed.
    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    from openhcs.processing.backends.analysis.cache_utils import run_cached_registration

    # Attempt cache‑based registration first
    if run_cached_registration("pyclesperanto", _register_pyclesperanto_from_cache):
        return

    # Fall back to full discovery
    from openhcs.processing.func_registry import _register_function
    from openhcs.core.memory.decorators import pyclesperanto
    import inspect

    print("🔧 Direct decoration of pyclesperanto functions - FULL DISCOVERY...")

    decorated_count = 0
    skipped_count = 0

    # Get functions using build_pycle_registry (same as register_pycle_ops)
    registry = build_pycle_registry()

    for meta in registry.values():
        # Check if function returns array (array-in/array-out pattern)
        try:
            sig = inspect.signature(meta.func)
            return_annotation = str(sig.return_annotation)

            # Skip functions that don't return arrays
            if not ('Array' in return_annotation or 'ndarray' in return_annotation):
                skipped_count += 1
                continue

        except Exception:
            skipped_count += 1
            continue

        # Skip dimension-changing functions (they break array chains)
        if meta.dim_change:
            skipped_count += 1
            continue

        try:
            # Apply unified decoration pattern
            from openhcs.constants import MemoryType
            from openhcs.processing.func_registry import _apply_unified_decoration

            wrapper_func = _apply_unified_decoration(
                original_func=meta.func,
                func_name=meta.name,
                memory_type=MemoryType.PYCLESPERANTO,
                create_wrapper=True  # pyclesperanto needs dtype preservation (many functions convert to float32)
            )

            # Enhance docstring with contract information (following CuPy/scikit-image pattern)
            _enhance_docstring_with_contract_info(wrapper_func, meta)

            # Register the function
            _register_function(wrapper_func, MemoryType.PYCLESPERANTO.value)
            decorated_count += 1

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to decorate {meta.name}: {e}")
            skipped_count += 1

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Decorated {decorated_count} pyclesperanto functions as OpenHCS functions")
    logger.info(f"Skipped {skipped_count} functions (dim_change or errors)")

    # Save metadata to cache for future fast startup
    _save_pyclesperanto_metadata(registry)
