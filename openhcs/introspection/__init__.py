"""
OpenHCS Introspection Package

Re-exports from python_introspect with OpenHCS-specific type resolution registered.

This package provides:
- SignatureAnalyzer: Extract parameter info from functions/dataclasses
- UnifiedParameterAnalyzer: Unified interface for all parameter sources
- Lazy dataclass utilities: Discovery and patching for lazy dataclasses

The core introspection logic lives in python_introspect (external dependency).
This module registers the OpenHCS namespace provider used for forward references.
"""

# Re-export from python_introspect
from python_introspect import (
    SignatureAnalyzer,
    ParameterInfo,
    DocstringInfo,
    DocstringExtractor,
    UnifiedParameterAnalyzer,
    UnifiedParameterInfo,
    register_namespace_provider,
)

# =============================================================================
# REGISTER OPENHCS-SPECIFIC FORWARD-REFERENCE NAMESPACE
# =============================================================================

def _openhcs_namespace_provider():
    """Provide OpenHCS types for forward reference resolution."""
    try:
        import objectstate.lazy_factory as lazy_module
        import openhcs.core.config as config_module
        return {**vars(lazy_module), **vars(config_module)}
    except ImportError:
        return {}


# Register on module load
register_namespace_provider(_openhcs_namespace_provider)


__all__ = [
    # Signature analysis (from python_introspect)
    'SignatureAnalyzer',
    'ParameterInfo',
    'DocstringInfo',
    'DocstringExtractor',
    # Unified analysis (from python_introspect)
    'UnifiedParameterAnalyzer',
    'UnifiedParameterInfo',
    # Plugin registration (from python_introspect)
    'register_namespace_provider',
]
