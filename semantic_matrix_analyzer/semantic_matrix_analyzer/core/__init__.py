"""
Core module for Semantic Matrix Analyzer.

This module provides the core functionality of the analyzer.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from semantic_matrix_analyzer.language import LanguageParser, language_registry
from semantic_matrix_analyzer.patterns import Intent, Pattern, PatternMatch


@dataclass
class IntentRegistry:
    """Registry for intents.
    
    This class maintains a registry of intents and provides methods to access them.
    """
    
    intents: Dict[str, Intent] = field(default_factory=dict)
    
    def register_intent(self, intent: Intent) -> None:
        """Register an intent.
        
        Args:
            intent: The intent to register.
        """
        self.intents[intent.name] = intent
    
    def get_intent(self, name: str) -> Optional[Intent]:
        """Get an intent by name.
        
        Args:
            name: The name of the intent.
            
        Returns:
            The intent, or None if not found.
        """
        return self.intents.get(name)
    
    def get_all_intents(self) -> List[Intent]:
        """Get all registered intents.
        
        Returns:
            A list of all registered intents.
        """
        return list(self.intents.values())
    
    def get_intent_names(self) -> List[str]:
        """Get all registered intent names.
        
        Returns:
            A list of all registered intent names.
        """
        return list(self.intents.keys())


@dataclass
class ComponentAnalysis:
    """Analysis results for a component."""
    
    name: str
    file_path: Path
    ast_node: Optional[Any] = None
    source_code: str = ""
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    intent_alignments: Dict[str, float] = field(default_factory=dict)  # intent_name -> alignment_score (0.0 to 1.0)
    pattern_matches: List[PatternMatch] = field(default_factory=list)


@dataclass
class SemanticMatrix:
    """A semantic matrix correlating components with intents."""
    
    components: List[str]
    intents: List[str]
    matrix: np.ndarray  # shape: (len(components), len(intents))
    component_analyses: Dict[str, ComponentAnalysis]


class SemanticMatrixBuilder:
    """Builder for semantic matrices."""
    
    def __init__(
        self, 
        components: List[str], 
        intents: List[str], 
        project_dir: str = ".", 
        components_config: Dict[str, Optional[str]] = None,
        intent_registry: IntentRegistry = None
    ):
        """Initialize the builder.
        
        Args:
            components: List of component names to analyze
            intents: List of intents to analyze
            project_dir: Root directory of the project
            components_config: Dictionary mapping component names to file paths (relative to project_dir)
                              If a value is None, the builder will try to infer the file path
            intent_registry: Registry of intents to detect in code
        """
        self.components = components
        self.intents = intents
        self.project_dir = Path(project_dir)
        self.components_config = components_config or {}
        self.intent_registry = intent_registry or intent_registry_global
        self.matrix = np.zeros((len(components), len(intents)))
        self.component_analyses = {}
    
    def get_file_path_for_component(self, component: str) -> Optional[Path]:
        """Get the file path for a component.
        
        First checks if the component has a path specified in components_config.
        If not, tries to infer the path based on common patterns.
        
        Args:
            component: The name of the component.
            
        Returns:
            The file path for the component, or None if not found.
        """
        # Check if the component has a path specified in components_config
        if component in self.components_config and self.components_config[component]:
            return self.project_dir / self.components_config[component]
        
        # Try to find the file by searching for the component name
        for ext in [".py", ".pyx", ".pyi"]:
            # Try snake_case version of the component name
            snake_case = ''.join(['_' + c.lower() if c.isupper() else c for c in component]).lstrip('_')
            potential_paths = list(self.project_dir.glob(f"**/{snake_case}{ext}"))
            if potential_paths:
                return potential_paths[0]
            
            # Try lowercase version of the component name
            potential_paths = list(self.project_dir.glob(f"**/{component.lower()}{ext}"))
            if potential_paths:
                return potential_paths[0]
            
            # Try exact case of the component name
            potential_paths = list(self.project_dir.glob(f"**/{component}{ext}"))
            if potential_paths:
                return potential_paths[0]
        
        # If all else fails, return None
        return None
    
    def analyze_component(self, component: str, file_path: Path) -> ComponentAnalysis:
        """Analyze a component.
        
        Args:
            component: The name of the component.
            file_path: The file path for the component.
            
        Returns:
            The analysis results for the component.
        """
        try:
            # Get the appropriate language parser
            language_parser = language_registry.get_parser_for_file(file_path)
            if not language_parser:
                logging.warning(f"No language parser found for {file_path}")
                return ComponentAnalysis(name=component, file_path=file_path)
            
            # Parse the file
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()
            
            ast_node = language_parser.parse_file(file_path)
            
            # Create component analysis
            analysis = ComponentAnalysis(
                name=component,
                file_path=file_path,
                ast_node=ast_node,
                source_code=source_code
            )
            
            # TODO: Extract dependencies
            
            # TODO: Analyze component
            
            # TODO: Detect patterns
            
            # TODO: Calculate intent alignments
            
            return analysis
        
        except Exception as e:
            logging.error(f"Error analyzing component {component}: {e}")
            return ComponentAnalysis(name=component, file_path=file_path)
    
    def build_matrix(self) -> SemanticMatrix:
        """Build the semantic matrix.
        
        Returns:
            The semantic matrix.
        """
        for i, component in enumerate(self.components):
            # Determine file path based on component name
            file_path = self.get_file_path_for_component(component)
            
            if file_path and file_path.exists():
                analysis = self.analyze_component(component, file_path)
                self.component_analyses[component] = analysis
                
                # Update matrix based on intent alignments
                for j, intent in enumerate(self.intents):
                    self.matrix[i, j] = analysis.intent_alignments.get(intent, 0.0)
            else:
                logging.warning(f"Could not find file for component {component}")
                # Create a placeholder analysis
                self.component_analyses[component] = ComponentAnalysis(
                    name=component,
                    file_path=Path(f"unknown_path_for_{component}.py")
                )
        
        return SemanticMatrix(
            components=self.components,
            intents=self.intents,
            matrix=self.matrix,
            component_analyses=self.component_analyses
        )


# Global intent registry
intent_registry_global = IntentRegistry()
