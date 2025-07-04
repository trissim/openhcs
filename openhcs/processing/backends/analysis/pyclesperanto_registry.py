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
# 4. Public API for OpenHCS – called in compile phase
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


def _register_pycle_ops_direct() -> None:
    """
    Direct registration of pyclesperanto functions without triggering registry initialization.

    This is called during Phase 2 of registry initialization to avoid circular dependencies.
    """
    from openhcs.processing.func_registry import _register_function
    from openhcs.core.memory.decorators import pyclesperanto
    import inspect

    print("🔧 Direct registration of pyclesperanto functions...")

    registered_count = 0
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
            # Instead of creating a wrapper, add memory type attributes directly to the original function
            # This makes the function pickleable since it's the same object as the module function
            original_func = meta.func

            # Add memory type attributes directly to the original function
            original_func.input_memory_type = "pyclesperanto"
            original_func.output_memory_type = "pyclesperanto"

            # Direct registration without triggering initialization
            _register_function(original_func, "pyclesperanto")
            registered_count += 1

        except Exception as e:
            print(f"Warning: Failed to register {meta.name}: {e}")
            skipped_count += 1

    print(f"✅ Direct registered {registered_count} pyclesperanto functions")
    print(f"⚠️  Skipped {skipped_count} functions (dim_change or errors)")
