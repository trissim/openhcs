"""Shared helpers for materialization handlers.

Keep this module free of backend/registry concerns.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def extract_fields(item: Any, field_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract fields from dataclass, dict, or pandas objects.

    Supports:
    - dataclass instances (uses dataclass reflection)
    - dicts (uses dict keys)
    - pandas DataFrames (uses column names)
    - pandas Series (uses index)
    """

    if isinstance(item, dict):
        if field_names:
            return {f: item.get(f) for f in field_names if f in item}
        return item

    if isinstance(item, pd.DataFrame):
        if field_names:
            return {f: item[f].tolist() for f in field_names if f in item.columns}
        return {col: item[col].tolist() for col in item.columns}

    if isinstance(item, pd.Series):
        if field_names:
            return {f: item[f] for f in field_names if f in item.index}
        return item.to_dict()

    if is_dataclass(item):
        if field_names:
            return {f: getattr(item, f, None) for f in field_names if hasattr(item, f)}
        return {name: getattr(item, name) for name in _dataclass_field_names(type(item))}

    return {"value": item}


@lru_cache(maxsize=256)
def _dataclass_field_names(item_type: type[object]) -> Tuple[str, ...]:
    """Return dataclass field names without per-row reflection overhead."""
    return tuple(field.name for field in fields(item_type))


def coerce_jsonable(value: Any) -> Any:
    """Convert numpy scalars/arrays to JSON-serializable Python types."""
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    return value


def normalize_slices(obj: Any, *, name: str):
    """Normalize input to list of 2D numpy arrays."""
    import numpy as np

    if obj is None:
        return []
    if isinstance(obj, list):
        return [np.asarray(x) for x in obj]
    arr = np.asarray(obj)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    raise ValueError(f"{name} must be a 2D/3D array or list of 2D arrays, got shape {arr.shape}")


def discover_array_fields(item: Any) -> List[str]:
    """Discover array/tuple fields in a dataclass instance."""
    if not hasattr(item, "__dataclass_fields__"):
        return []

    from typing import get_origin

    array_fields: List[str] = []
    for f in fields(item):
        value = getattr(item, f.name, None)
        origin = hasattr(f.type, "__origin__") and get_origin(f.type)
        if origin in (list, List, tuple, Tuple):
            array_fields.append(f.name)
        elif isinstance(value, list) and value and isinstance(value[0], (tuple, list)):
            array_fields.append(f.name)
    return array_fields


def expand_array_field(
    array_data: List[Any],
    base_row: Dict[str, Any],
    row_columns: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Expand an array field into multiple rows."""
    if not array_data:
        return [base_row]

    rows: List[Dict[str, Any]] = []
    for elem in array_data:
        if isinstance(elem, (list, tuple)):
            cols = {
                col: elem[int(idx)]
                for idx, col in row_columns.items()
                if str(idx).isdigit() and int(idx) < len(elem)
            }
        else:
            cols = {}
        rows.append({**base_row, **cols})
    return rows
