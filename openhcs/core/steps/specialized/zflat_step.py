"""Z-stack flattening step for creating 2D projections from Z-stacks."""

import logging

from openhcs.core.steps.function_step import FunctionStep
from openhcs.processing.backends.processors.numpy_processor import \
    NumPyImageProcessor as ImageProcessor

logger = logging.getLogger(__name__)


class ZFlatStep(FunctionStep):
    """Specialized step for Z-stack flattening using various projection methods."""

    PROJECTION_METHODS = {
        "max": "max_projection",
        "mean": "mean_projection",
        "median": "median_projection",
        "min": "min_projection",
        "std": "std_projection",
        "sum": "sum_projection"
    }

    def __init__(self, *, method: str = "max"):
        """Initialize Z-stack flattening step.

        Args:
            method: Projection method ("max", "mean", "median", "min", "std", "sum")

        Raises:
            ValueError: If method is invalid
        """
        if method not in self.PROJECTION_METHODS and method not in self.PROJECTION_METHODS.values():
            valid_methods = list(self.PROJECTION_METHODS.keys())
            raise ValueError(f"Invalid projection method: {method}. Valid methods: {valid_methods}")

        projection_method = self.PROJECTION_METHODS.get(method, method)

        super().__init__(
            func=(ImageProcessor.create_projection, {'method': projection_method}),
            name="Z-Stack Flattening"
        )

        self.variable_components = ['z_index']
        self.group_by = None
