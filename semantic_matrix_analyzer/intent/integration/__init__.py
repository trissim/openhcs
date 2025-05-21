"""
Integration components for the Structural Intent Analysis system.
"""

from semantic_matrix_analyzer.intent.integration.combiner import ConfigurableIntentCombiner
from semantic_matrix_analyzer.intent.integration.hierarchy import ConfigurableHierarchyBuilder
from semantic_matrix_analyzer.intent.integration.reporter import (
    IntentRelationshipAnalyzer, ConfigurableIntentReporter
)

__all__ = [
    "ConfigurableIntentCombiner",
    "ConfigurableHierarchyBuilder",
    "IntentRelationshipAnalyzer",
    "ConfigurableIntentReporter"
]
