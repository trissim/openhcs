# __doctrinal_owner__: OpenHCS/core-architecture
# __doctrinal_clauses__: Clause 65
"""
Custom exceptions for the OpenHCS core system.
Ensures that errors are specific and fail loudly as per Clause 65.
"""

class OpenHCSError(Exception):
    """Base class for all OpenHCS custom exceptions."""
    pass

class ImmutabilityError(OpenHCSError, AttributeError):
    """Raised when an attempt is made to modify an immutable object after initialization."""
    pass

class BackendNotFoundError(OpenHCSError, KeyError):
    """Raised when a requested backend is not found in a registry."""
    pass

class SchemaMismatchError(OpenHCSError, ValueError):
    """Raised when a schema hash mismatch occurs between components."""
    pass

class DoctrineViolationError(OpenHCSError, RuntimeError):
    """Raised when a doctrinal principle is violated at runtime in a critical way."""
    pass

class DoctrineWarning(UserWarning):
    """Warning issued for non-critical but doctrinally questionable practices."""
    pass

# Specific error for registry immutability from the plan, though ImmutabilityError covers it.
# class RegistryMutabilityError(ImmutabilityError):
#     """Raised specifically when a registry's immutability is breached."""
#     pass

# VFSExclusivityError was mentioned in the MyPy plugin plan, might be defined there or here.
# For now, keeping it focused on registry-related exceptions from this plan.