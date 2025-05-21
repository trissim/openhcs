"""
Specialized step implementations for the OpenHCS pipeline architecture.

This module provides specialized step implementations that build on the
three canonical step types (FunctionStep, PositionGenerationStep, ImageAssemblyStep).

These specialized steps provide convenient interfaces for common operations
while maintaining doctrinal purity by delegating to the appropriate backends.

Doctrinal Clauses:
- Clause 3 — Declarative Primacy
- Clause 12 — Absolute Clean Execution
- Clause 21 — Context Immunity
- Clause 52 — Declarative Repetition Doctrine
- Clause 65 — No Fallback Logic
- Clause 88 — No Inferred Capabilities
- Clause 92 — Structural Validation First
- Clause 244 — Rot Intolerance
- Clause 245 — Declarative Enforcement
- Clause 246 — Statelessness Mandate
"""

from openhcs.core.steps.specialized.composite_step import CompositeStep
from openhcs.core.steps.specialized.focus_step import FocusStep
from openhcs.core.steps.specialized.norm_step import NormStep
from openhcs.core.steps.specialized.zflat_step import ZFlatStep

__all__ = [
    'ZFlatStep',
    'FocusStep',
    'CompositeStep',
    'NormStep'
]
