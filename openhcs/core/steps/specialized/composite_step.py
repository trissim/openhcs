"""
Channel compositing step for the OpenHCS pipeline architecture.

This module provides a specialized step for creating composite images from
multiple channels with specified weights.

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
from typing import List, Optional

from openhcs.core.image_processor import ImageProcessor
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


class CompositeStep(FunctionStep):
    """
    Specialized step for creating composite images from multiple channels.

    This step creates composite images from multiple channels with specified weights.
    It is a specialized FunctionStep that processes multiple channel images.

    Note:
        This step is built on the FunctionStep canonical type in OpenHCS.
        It is a pure functional wrapper with hardcoded values for compositing parameters.
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        weights: Optional[List[float]] = None,
    ):
        """
        Initialize a channel compositing step with memory type declarations.

        Args:
            weights: Channel weights (default: equal weights for all channels)

        Raises:
            ValueError: If memory types are invalid
        """
        # Initialize the FunctionStep with the compositing function
        super().__init__(
            func=(ImageProcessor.create_composite, {'weights': weights}),
            name="Channel Compositing"
        )

        # Set variable components
        self.variable_components = [VariableComponents.CHANNEL]
        self.group_by = None
