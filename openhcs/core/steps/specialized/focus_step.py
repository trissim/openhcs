"""
Focus-based Z-stack processing step for the OpenHCS pipeline architecture.

This module provides a specialized step for focus-based Z-stack processing,
which finds the best focus plane in a Z-stack using focus metrics.

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

import logging

from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.enhance.dl_edof_unsupervised import \
    dl_edof_unsupervised as deep_focus

logger = logging.getLogger(__name__)


class FocusStep(FunctionStep):
    """
    Specialized step for focus-based Z-stack processing.

    This step finds the best focus plane in a Z-stack using focus metrics.
    It is a specialized FunctionStep that processes Z-stacks to find the best focus.

    Note:
        This step is built on the FunctionStep canonical type in OpenHCS.
        It is a pure functional wrapper with hardcoded values for focus parameters.
    """

    def __init__(self):
        """
        Initialize a focus-based Z-stack processing step with memory type declarations.

        Raises:
            ValueError: If memory types are invalid
        """

        # Initialize the FunctionStep with the focus function
        super().__init__(
            func=deep_focus,
            name="Focus Selection"
        )

        # Set variable components
        self.variable_components = ['z_index']
        self.group_by = None
