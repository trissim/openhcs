"""
CellProfiler Library - Absorbed into OpenHCS

Maps CellProfiler module names to OpenHCS functions.
Functions are loaded dynamically from contracts.json to avoid import errors.
"""

from typing import Dict, Callable, Optional
from pathlib import Path
import json
import importlib

# Load contracts registry
_CONTRACTS_PATH = Path(__file__).parent / "contracts.json"
_contracts: Dict = {}

if _CONTRACTS_PATH.exists():
    _contracts = json.loads(_CONTRACTS_PATH.read_text())

# Dynamic function loader
_function_cache: Dict[str, Callable] = {}


def get_function(module_name: str) -> Optional[Callable]:
    """Get an OpenHCS function by its CellProfiler module name.

    Dynamically loads the function from the appropriate module file.
    Returns None if the module is not found.
    """
    if module_name in _function_cache:
        return _function_cache[module_name]

    if module_name not in _contracts:
        return None

    meta = _contracts[module_name]
    func_name = meta["function_name"]

    # Try multiple file name patterns
    # 1. Remove underscores from function name
    # 2. Use module name (lowercase, no underscores)
    # 3. Try partial matches
    file_stems_to_try = [
        func_name.replace('_', ''),  # gray_to_color_rgb -> graytocolorrgb
        module_name.lower().replace('_', ''),  # GrayToColorRgb -> graytocolorrgb
    ]

    # Also try common prefixes (e.g., graytocolor for gray_to_color_rgb)
    parts = func_name.split('_')
    if len(parts) > 2:
        file_stems_to_try.append(''.join(parts[:-1]))  # gray_to_color_rgb -> graytocolor

    for file_stem in file_stems_to_try:
        try:
            module = importlib.import_module(
                f".functions.{file_stem}",
                package=__package__,
            )
            func = getattr(module, func_name, None)
            if func is not None:
                _function_cache[module_name] = func
                return func
        except (ImportError, AttributeError):
            continue

    return None


def get_contract(module_name: str) -> Optional[Dict]:
    """Get the contract metadata for a CellProfiler module."""
    return _contracts.get(module_name)


def list_modules() -> list:
    """List all available CellProfiler module names."""
    return list(_contracts.keys())


__all__ = [
    "get_function",
    "get_contract",
    "list_modules",
]
