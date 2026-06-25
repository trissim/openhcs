
"""Public step package exports without eager concrete-step import cycles."""

from __future__ import annotations

from typing import TYPE_CHECKING

from openhcs.core.steps.abstract import AbstractStep

if TYPE_CHECKING:
    from openhcs.core.steps.function_step import FunctionStep
# Specialized step implementations
#from openhcs.core.steps.specialized import (CompositeStep, FocusStep,
                                               #NormStep, ZFlatStep)
# Removed StepContext, StepResult, StepState, and StepStatus imports as part of context standardization

# Define public exports
__all__ = [
    # New API - Core interfaces
    'AbstractStep',

    # New API - Canonical step types
    'FunctionStep',

    # Specialized step implementations
   # 'ZFlatStep',
   # 'FocusStep',
   # 'CompositeStep',
   # 'NormStep',
    ]


def __getattr__(name: str) -> object:
    if name == "FunctionStep":
        from openhcs.core.steps.function_step import FunctionStep

        return FunctionStep
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# PERFORMANCE OPTIMIZATION: Pre-warm step editor cache at import time
try:
    from openhcs.config_framework import prewarm_callable_analysis_cache
    prewarm_callable_analysis_cache(AbstractStep.__init__)
except ImportError:
    # Circular import during subprocess initialization - cache warming not needed
    # for non-UI execution contexts (ZMQ server, workers, etc.)
    pass
