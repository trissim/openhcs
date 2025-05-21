"""
Models for the Structural Intent Analysis system.
"""

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, TypeIntent, StructuralIntent, IntentType, IntentSource,
    CodeLocation, IntentHierarchy
)

__all__ = [
    "Intent",
    "NameIntent",
    "TypeIntent",
    "StructuralIntent",
    "IntentType",
    "IntentSource",
    "CodeLocation",
    "IntentHierarchy"
]
