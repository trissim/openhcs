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

# Functions requiring specific dtype conversions
BINARY_FUNCTIONS = {'binary_infsup', 'binary_supinf'}
UINT8_FUNCTIONS = {'mode', 'mode_box', 'mode_sphere'}  # Functions requiring uint8 input

# ---- Module-level adapter: pyclesperanto → OpenHCS policy ----
def _pycle_adapt_function(original_func):
    from functools import wraps
    import numpy as np
    DIM_ERR_TOKENS = ("dimension", "dimensional", "3d", "ndim", "axis", "rank", "shapes not aligned")

    func_name = getattr(original_func, '__name__', 'unknown')

    @wraps(original_func)
    def adapted(image, *args, slice_by_slice: bool = False, **kwargs):
        try:
            import pyclesperanto as cle
        except Exception:
            return original_func(image, *args, **kwargs)

        original_dtype = image.dtype
        converted_image = image

        # Apply dtype conversion for functions with specific requirements
        if func_name in BINARY_FUNCTIONS:
            # Convert to binary: float32 [0,1] -> uint8 {0,255} with threshold at 0.5
            if image.dtype == np.float32:
                converted_image = ((image > 0.5) * 255).astype(np.uint8)
        elif func_name in UINT8_FUNCTIONS:
            # Convert to uint8: float32 [0,1] -> uint8 [0,255]
            if image.dtype == np.float32:
                converted_image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        if hasattr(converted_image, 'ndim') and converted_image.ndim == 3:
            if slice_by_slice:
                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type
                mem = _detect_memory_type(converted_image)
                slices = unstack_slices(converted_image, mem, 0)
                results = [original_func(sl, *args, **kwargs) for sl in slices]
                result = stack_slices(results, mem, 0)
            else:
                try:
                    result = original_func(converted_image, *args, **kwargs)
                except Exception as e:
                    msg = str(e).lower()
                    if any(tok in msg for tok in DIM_ERR_TOKENS):
                        from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type
                        mem = _detect_memory_type(converted_image)
                        slices = unstack_slices(converted_image, mem, 0)
                        results = [original_func(sl, *args, **kwargs) for sl in slices]
                        result = stack_slices(results, mem, 0)
                    else:
                        raise
            # Promote 2D to singleton-Z
            if hasattr(result, 'ndim') and result.ndim == 2:
                try:
                    temp = cle.concatenate_along_z(result, result)
                    result = temp[0:1, :, :]
                except Exception:
                    pass
            elif isinstance(result, tuple) and hasattr(result[0], 'ndim') and result[0].ndim == 2:
                try:
                    temp = cle.concatenate_along_z(result[0], result[0])
                    main = temp[0:1, :, :]
                    result = (main, *result[1:])
                except Exception:
                    pass
        else:
            result = original_func(converted_image, *args, **kwargs)

        # Convert result back to original dtype if needed
        if func_name in BINARY_FUNCTIONS or func_name in UINT8_FUNCTIONS:
            if hasattr(result, 'dtype') and result.dtype != original_dtype:
                if result.dtype == np.uint8 and original_dtype == np.float32:
                    # Convert back: uint8 [0,255] -> float32 [0,1]
                    result = result.astype(np.float32) / 255.0
                elif result.dtype == np.bool_ and original_dtype == np.float32:
                    # Convert back: bool -> float32
                    result = result.astype(np.float32)
            elif isinstance(result, tuple):
                # Handle tuple results (array, value) - convert array part
                if hasattr(result[0], 'dtype') and result[0].dtype != original_dtype:
                    if result[0].dtype == np.uint8 and original_dtype == np.float32:
                        converted_array = result[0].astype(np.float32) / 255.0
                        result = (converted_array, *result[1:])
                    elif result[0].dtype == np.bool_ and original_dtype == np.float32:
                        converted_array = result[0].astype(np.float32)
                        result = (converted_array, *result[1:])

        return result
    return adapted

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

# Import blacklist system
from .function_classifier import is_blacklisted, log_blacklist_stats

# ---------------------------------------------------------------------------
# 1. Contracts & metadata
# ---------------------------------------------------------------------------

class Contract(Enum):
    """Semantic contracts recognised by OpenHCS."""

    SLICE_SAFE = auto()   # keeps each Z‑slice independent (sigma_z or radius_z can be 0)
    CROSS_Z = auto()      # 3‑D connectivity – fuses neighbouring slices
    DIM_CHANGE = auto()   # output is scalar, table or different shape ➔ excluded





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
# 2. Result handling (same approach as CuPy/scikit-image)
# ---------------------------------------------------------------------------

def _handle_pycle_result(result, original_image):
    """
    Handle pyclesperanto function results to maintain array-in/array-out contract.
    Same logic as CuPy and scikit-image registries.
    """
    import numpy as np
    import pyclesperanto as cle

    # If result is already a 3D array, return as-is
    if hasattr(result, 'ndim') and result.ndim == 3:
        return result

    # If result is scalar, return (original_image, result) - same as CuPy/scikit-image
    elif np.isscalar(result):
        return original_image, result

    # If result is 2D, promote to 3D (singleton Z)
    elif hasattr(result, 'ndim') and result.ndim == 2:
        try:
            temp = cle.concatenate_along_z(result, result)
            return temp[0:1, :, :]
        except Exception:
            # Fallback: use numpy
            return np.expand_dims(result, 0)

    # If result is tuple with 2D first element, promote first element
    elif isinstance(result, tuple) and hasattr(result[0], 'ndim') and result[0].ndim == 2:
        try:
            temp = cle.concatenate_along_z(result[0], result[0])
            main = temp[0:1, :, :]
            return (main, *result[1:])
        except Exception:
            # Fallback: use numpy
            main = np.expand_dims(result[0], 0)
            return (main, *result[1:])

    # If result is array with ndim < 3, return (original_image, result)
    elif hasattr(result, 'ndim') and result.ndim < 3:
        return original_image, result

    # Fallback: return original image
    else:
        return original_image


# ---------------------------------------------------------------------------
# 3. Runtime classification (same approach as CuPy/scikit-image)
# ---------------------------------------------------------------------------

_IMAGE_PARAM_NAMES = {"src", "source", "image", "input", "src1", "input_image", "input_image0"}


# ---------------------------------------------------------------------------
# 3. Static discovery
# ---------------------------------------------------------------------------

def _is_image_param(p: inspect.Parameter) -> bool:
    return p.name.lower() in _IMAGE_PARAM_NAMES and p.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )


def _classify_pycle_function(func) -> Tuple[Contract, bool]:
    """
    Test how a pyclesperanto function behaves with 3D input to determine its processing contract.
    Same logic as CuPy and scikit-image registries.
    """
    try:
        import numpy as np
        import pyclesperanto as cle

        # Get full import path for debugging
        module_name = getattr(func, '__module__', 'unknown')
        func_name = getattr(func, '__name__', 'unknown')
        full_import_path = f"{module_name}.{func_name}"

        # Create test data
        test_3d = np.random.rand(3, 20, 20).astype(np.float32)
        test_2d = test_3d[0]

        # Print function being tested for warning attribution
        print(f"    🧪 Testing pyclesperanto function: {full_import_path}")

        # Test if function accepts 3D input
        result_3d = func(test_3d)
        result_2d = func(test_2d)

        # Check if shapes are preserved
        if hasattr(result_3d, 'shape'):
            if result_3d.shape == ():
                # Scalar result - this is a value-returning function (same as CuPy/scikit-image)
                return Contract.SLICE_SAFE, True
            elif result_3d.shape != test_3d.shape:
                # Output shape changed - likely dimension-changing function
                return Contract.DIM_CHANGE, True
        else:
            # No shape attribute - this is a value-returning function
            return Contract.SLICE_SAFE, True

        # Test if processing is slice-by-slice
        manual_3d = np.stack([func(test_3d[z]) for z in range(test_3d.shape[0])])

        if np.allclose(result_3d, manual_3d, rtol=1e-5, atol=1e-8):
            # Results match slice-by-slice processing
            return Contract.SLICE_SAFE, True
        else:
            # Different results - likely volumetric processing
            return Contract.CROSS_Z, True

    except Exception as e:
        # Function failed on 3D input
        error_msg = str(e).lower()

        # Check if it's a dimension error
        if any(keyword in error_msg for keyword in ['dimension', 'shape', '3d', 'axis']):
            return Contract.SLICE_SAFE, False  # 2D only, but slice-safe
        else:
            return Contract.DIM_CHANGE, False  # Unknown behavior, mark as dimension-changing





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

        # Skip known problematic functions that open windows or have side effects
        problematic_names = {
            'imshow', 'show', 'plot', 'display', 'visualize', 'viewer',
            'save', 'write', 'export', 'print_info', 'info'
        }
        if name.lower() in problematic_names:
            continue

        # Skip blacklisted functions
        if is_blacklisted(fn, name):
            continue

        # Use runtime testing to classify function behavior (same as CuPy/scikit-image)
        contract, is_valid = _classify_pycle_function(fn)

        # Skip functions that failed classification
        if not is_valid:
            continue

        # Check for Z-axis parameters (still needed for metadata)
        has_z_kw = any("z" in p.name and p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                       for p in params)

        # Get lowercase name for tags
        lname = name.lower()

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

    # Register functions from cache (following scikit-image pattern)
    decorated_count = 0
    skipped_count = 0

    for full_name, func_data in cached_metadata.items():
        try:
            func_name = func_data['name']
            contract = func_data['contract']

            # Skip functions with unknown or dimension-changing contracts
            if contract in ['unknown', 'dim_change']:
                skipped_count += 1
                continue

            # Get the actual function object
            original_func = _get_pyclesperanto_function("pyclesperanto", func_name)
            if original_func is None:
                logger.warning(f"Could not find function {func_name} in pyclesperanto")
                skipped_count += 1
                continue

            # Adapt then apply unified decoration (following scikit-image pattern)
            from openhcs.processing.func_registry import _apply_unified_decoration, _register_function
            adapted = _pycle_adapt_function(original_func)
            wrapper_func = _apply_unified_decoration(
                original_func=adapted,
                func_name=func_name,
                memory_type=MemoryType.PYCLESPERANTO,
                create_wrapper=True
            )

            _register_function(wrapper_func, MemoryType.PYCLESPERANTO.value)
            decorated_count += 1

        except Exception as e:
            logger.warning(f"Failed to register {func_name}: {e}")
            skipped_count += 1

    logger.info(f"Registered {decorated_count} pyclesperanto functions from cache")
    logger.info(f"Skipped {skipped_count} functions (unknown/dim_change contracts or errors)")
    return True


# ---------------------------------------------------------------------------
# 5. Public API for OpenHCS – called in compile phase
# ---------------------------------------------------------------------------

def build_pycle_registry(refresh: bool = False) -> Dict[str, OpMeta]:
    """Return a *cached* registry mapping kernel‑name → :class:`OpMeta`."""
    global _REGISTRY  # noqa: PLW0603 – intentional module cache
    if refresh or _REGISTRY is None:
        # Log blacklist information
        log_blacklist_stats()
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
            # Use the new unified registration system (same as cache path)
            from openhcs.constants import MemoryType
            from openhcs.processing.func_registry import _apply_unified_decoration

            adapted = _pycle_adapt_function(meta.func)
            wrapper_func = _apply_unified_decoration(
                original_func=adapted,
                func_name=meta.name,
                memory_type=MemoryType.PYCLESPERANTO,
                create_wrapper=True
            )

            _register_function(wrapper_func, MemoryType.PYCLESPERANTO.value)
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
        # Register all functions with valid contracts (same as CuPy/scikit-image)
        # pyclesperanto doesn't have UNKNOWN contract, so register all

        # Register the function with OpenHCS
        try:
            from openhcs.constants import MemoryType
            from openhcs.processing.func_registry import _apply_unified_decoration
            from functools import wraps

            # Define adapter function locally (same as cache registration)
            DIM_ERR_TOKENS = ("dimension", "dimensional", "3d", "ndim", "axis", "rank", "shapes not aligned")

            @wraps(meta.func)
            def adapted(image, *args, slice_by_slice: bool = False, **kwargs):
                try:
                    import pyclesperanto as cle
                except Exception:
                    return meta.func(image, *args, **kwargs)
                if hasattr(image, 'ndim') and image.ndim == 3:
                    if slice_by_slice:
                        from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type
                        mem = _detect_memory_type(image)
                        slices = unstack_slices(image, mem, 0)
                        results = [meta.func(sl, *args, **kwargs) for sl in slices]
                        result = stack_slices(results, mem, 0)
                    else:
                        try:
                            result = meta.func(image, *args, **kwargs)
                        except Exception as e:
                            msg = str(e).lower()
                            if any(tok in msg for tok in DIM_ERR_TOKENS):
                                from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type
                                mem = _detect_memory_type(image)
                                slices = unstack_slices(image, mem, 0)
                                results = [meta.func(sl, *args, **kwargs) for sl in slices]
                                result = stack_slices(results, mem, 0)
                            else:
                                raise
                    # Handle different result types (same as CuPy/scikit-image)
                    result = _handle_pycle_result(result, image)
                else:
                    try:
                        result = meta.func(image, *args, **kwargs)
                    except Exception as e:
                        msg = str(e).lower()
                        if any(tok in msg for tok in DIM_ERR_TOKENS):
                            from openhcs.core.memory.stack_utils import unstack_slices, stack_slices, _detect_memory_type
                            mem = _detect_memory_type(image)
                            slices = unstack_slices(image, mem, 0)
                            results = [meta.func(sl, *args, **kwargs) for sl in slices]
                            result = stack_slices(results, mem, 0)
                        else:
                            raise
                # Handle different result types (same as CuPy/scikit-image)
                return _handle_pycle_result(result, image)

            wrapper_func = _apply_unified_decoration(
                original_func=adapted,
                func_name=meta.name,
                memory_type=MemoryType.PYCLESPERANTO,
                create_wrapper=True
            )

            _register_function(wrapper_func, MemoryType.PYCLESPERANTO.value)
            decorated_count += 1
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to register {meta.name}: {e}")
            skipped_count += 1

    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Decorated {decorated_count} pyclesperanto functions as OpenHCS functions")
    logger.info(f"Skipped {skipped_count} functions (dim_change or errors)")

    # Save metadata to cache for future fast startup
    _save_pyclesperanto_metadata(registry)
