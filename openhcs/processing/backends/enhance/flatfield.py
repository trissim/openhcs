"""Nominal contracts shared by flat-field correction implementations."""

from enum import Enum


class FlatfieldCorrectionMode(Enum):
    """How an estimated illumination field is removed from an image."""

    DIVIDE = "divide"
    SUBTRACT = "subtract"


class BasicFittingMode(Enum):
    """Optimization algorithm exposed by BaSiCPy."""

    LADMAP = "ladmap"
    APPROXIMATE = "approximate"
