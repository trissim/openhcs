"""
Image normalization step for the OpenHCS pipeline architecture.

This module provides a specialized step for image normalization, which
performs percentile-based normalization on images.

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

from openhcs.core.image_processor import ImageProcessor
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


class NormStep(FunctionStep):
    """
    Specialized step for image normalization.

    This step performs percentile-based normalization on images.
    It is a specialized FunctionStep that normalizes images.

    Note:
        This step is built on the FunctionStep canonical type in OpenHCS.
        It is a pure functional wrapper with hardcoded values for normalization parameters.
    """

    def __init__(
        self,
        *,  # Force keyword-only arguments
        low_percentile: float = 0.1,
        high_percentile: float = 99.9,
    ):
        """
        Initialize an image normalization step with memory type declarations.

        Args:
            low_percentile: Low percentile for normalization (0-100)
            high_percentile: High percentile for normalization (0-100)

        Raises:
            ValueError: If percentiles are invalid or memory types are invalid
        """
        # Validate percentiles
        if not 0 <= low_percentile <= 100:
            raise ValueError(f"Low percentile must be between 0 and 100, got {low_percentile}")
        if not 0 <= high_percentile <= 100:
            raise ValueError(f"High percentile must be between 0 and 100, got {high_percentile}")
        if low_percentile >= high_percentile:
            msg = (f"Low percentile ({low_percentile}) must be less than "
                   f"high percentile ({high_percentile})")
            raise ValueError(msg)

        # Initialize the FunctionStep with the normalization function
        super().__init__(
            func=(ImageProcessor.stack_percentile_normalize, {
                'low_percentile': low_percentile,
                'high_percentile': high_percentile
            }),
            name="Image Normalization"
        )

        # Set variable components
        self.variable_components = ['site']
        self.group_by = None
