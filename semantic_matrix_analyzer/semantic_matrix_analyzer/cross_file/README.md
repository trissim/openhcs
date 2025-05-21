# Cross-File Analysis

The Cross-File Analysis system is a component of the Semantic Matrix Analyzer that provides functionality for analyzing relationships and dependencies between files, enabling the detection of architectural patterns, inconsistencies, and potential refactoring opportunities across the codebase.

## Overview

The Cross-File Analysis system consists of several key components:

1. **Dependency Graph**: Represents relationships between files, classes, functions, etc.
2. **Dependency Extraction**: Extracts dependencies from code
3. **Pattern Detection**: Detects architectural patterns in the codebase
4. **Inconsistency Detection**: Identifies inconsistencies in the codebase
5. **Refactoring Opportunity Detection**: Discovers opportunities for refactoring

## Usage

### Basic Usage

```python
from semantic_matrix_analyzer.cross_file import (
    DependencyGraph, DependencyExtractor, ArchitecturalPatternDetector,
    InconsistencyDetector, RefactoringOpportunityDetector
)

# Create a dependency graph
dependency_graph = DependencyGraph()

# Create a dependency extractor
dependency_extractor = DependencyExtractor()

# Extract dependencies from a file
nodes, edges = dependency_extractor.extract_dependencies(Path("example.py"))

# Add nodes and edges to the graph
for node in nodes:
    dependency_graph.add_node(node)

for edge in edges:
    dependency_graph.add_edge(edge)

# Detect architectural patterns
pattern_detector = ArchitecturalPatternDetector(dependency_graph)
patterns = pattern_detector.detect_patterns()

# Detect inconsistencies
inconsistency_detector = InconsistencyDetector(dependency_graph)
inconsistencies = inconsistency_detector.detect_inconsistencies()

# Detect refactoring opportunities
opportunity_detector = RefactoringOpportunityDetector(dependency_graph)
opportunities = opportunity_detector.detect_opportunities()
```

### Saving and Loading the Dependency Graph

```python
# Save the dependency graph
dependency_graph.save()

# Load the dependency graph
dependency_graph = DependencyGraph("dependency_graph.json")
```

## Components

### Dependency Graph

Represents relationships between files, classes, functions, etc.:

- `Node`: A node in the dependency graph (file, class, function, etc.)
- `Edge`: An edge in the dependency graph (imports, calls, inherits, etc.)
- `DependencyGraph`: A graph of dependencies between files, classes, functions, etc.

### Dependency Extraction

Extracts dependencies from code:

- `DependencyExtractor`: Extracts dependencies from code
- `_extract_imports`: Extracts import dependencies
- `_extract_classes`: Extracts class dependencies
- `_extract_functions`: Extracts function dependencies
- `_extract_variables`: Extracts variable dependencies

### Pattern Detection

Detects architectural patterns in the codebase:

- `ArchitecturalPattern`: An architectural pattern detected in the codebase
- `ArchitecturalPatternDetector`: Detects architectural patterns in the dependency graph
- `_detect_layered_architecture`: Detects layered architecture pattern
- `_detect_microservices_architecture`: Detects microservices architecture pattern
- `_detect_event_driven_architecture`: Detects event-driven architecture pattern
- `_detect_mvc_architecture`: Detects model-view-controller architecture pattern
- `_detect_repository_pattern`: Detects repository pattern
- `_detect_factory_pattern`: Detects factory pattern
- `_detect_singleton_pattern`: Detects singleton pattern

### Inconsistency Detection

Identifies inconsistencies in the codebase:

- `Inconsistency`: An inconsistency detected in the codebase
- `InconsistencyDetector`: Detects inconsistencies in the dependency graph
- `_detect_naming_inconsistencies`: Detects naming inconsistencies
- `_detect_interface_inconsistencies`: Detects interface inconsistencies
- `_detect_implementation_inconsistencies`: Detects implementation inconsistencies
- `_detect_documentation_inconsistencies`: Detects documentation inconsistencies

### Refactoring Opportunity Detection

Discovers opportunities for refactoring:

- `RefactoringOpportunity`: A refactoring opportunity detected in the codebase
- `RefactoringOpportunityDetector`: Detects refactoring opportunities in the dependency graph
- `_detect_extract_class_opportunities`: Detects extract class opportunities
- `_detect_move_method_opportunities`: Detects move method opportunities
- `_detect_rename_opportunities`: Detects rename opportunities
- `_detect_extract_interface_opportunities`: Detects extract interface opportunities

## Benefits

The Cross-File Analysis system provides several benefits:

1. **Architectural Understanding**: Gain insights into the architecture of the codebase
2. **Inconsistency Detection**: Identify inconsistencies in naming, interfaces, implementations, etc.
3. **Refactoring Guidance**: Discover opportunities for improving the codebase
4. **Pattern Recognition**: Detect common architectural patterns
5. **Dependency Management**: Understand relationships between components
