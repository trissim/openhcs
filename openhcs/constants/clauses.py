"""
OpenHCS Doctrinal Clauses.

This module defines the doctrinal clauses that govern the OpenHCS codebase.
These clauses are referenced by constants to indicate which clause governs
their usage and enforcement.
"""

from enum import Enum, auto


class Clause(Enum):
    """
    Enum for OpenHCS doctrinal clauses.
    
    These values are used to identify which clause governs a particular
    constant or behavior in the codebase.
    """
    # Core architectural principles
    DECLARATIVE_PRIMACY = auto()  # Clause 3
    FAIL_LOUDLY = auto()  # Clause 65
    NO_INFERRED_CAPABILITIES = auto()  # Clause 88
    MEMORY_DECLARATION = auto()  # Clause 101
    DECLARED_MEMORY_TYPES = auto()  # Clause 106-A
    MEMORY_BACKEND_RESTRICTIONS = auto()  # Clause 273
    
    def __str__(self) -> str:
        """Return a string representation of the clause."""
        clause_numbers = {
            Clause.DECLARATIVE_PRIMACY: "3",
            Clause.FAIL_LOUDLY: "65",
            Clause.NO_INFERRED_CAPABILITIES: "88",
            Clause.MEMORY_DECLARATION: "101",
            Clause.DECLARED_MEMORY_TYPES: "106-A",
            Clause.MEMORY_BACKEND_RESTRICTIONS: "273",
        }
        return f"Clause {clause_numbers[self]}"
