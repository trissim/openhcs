# Plan 03: Cross-File Analysis

## Objective

Develop a comprehensive system for analyzing relationships and dependencies between files, enabling the detection of architectural patterns, inconsistencies, and potential refactoring opportunities across the codebase.

## Rationale

Code analysis that focuses on individual files in isolation misses important architectural patterns and dependencies. By implementing cross-file analysis:

1. **Architectural Understanding**: Identify patterns and structures across the codebase
2. **Dependency Tracking**: Understand how changes in one file affect others
3. **Inconsistency Detection**: Find inconsistencies in implementation across related files
4. **Refactoring Opportunities**: Identify opportunities for architectural improvements
5. **Impact Analysis**: Assess the impact of changes before implementation

## Implementation Details

### 1. Dependency Graph Construction

Create a system for constructing a dependency graph of the codebase:

```python
@dataclass
class Node:
    """A node in the dependency graph."""
    id: str
    type: str  # "file", "class", "function", "variable", etc.
    name: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Edge:
    """An edge in the dependency graph."""
    source_id: str
    target_id: str
    type: str  # "imports", "calls", "inherits", "references", etc.
    attributes: Dict[str, Any] = field(default_factory=dict)

class DependencyGraph:
    """A graph of dependencies between files, classes, functions, etc."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.node_index: Dict[str, List[str]] = {}  # type -> [node_id]
        self.edge_index: Dict[str, List[Edge]] = {}  # type -> [edge]
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        
        # Update the node index
        if node.type not in self.node_index:
            self.node_index[node.type] = []
        self.node_index[node.type].append(node.id)
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph."""
        self.edges.append(edge)
        
        # Update the edge index
        if edge.type not in self.edge_index:
            self.edge_index[edge.type] = []
        self.edge_index[edge.type].append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_nodes_by_type(self, node_type: str) -> List[Node]:
        """Get all nodes of a specific type."""
        node_ids = self.node_index.get(node_type, [])
        return [self.nodes[node_id] for node_id in node_ids]
    
    def get_edges_by_type(self, edge_type: str) -> List[Edge]:
        """Get all edges of a specific type."""
        return self.edge_index.get(edge_type, [])
    
    def get_outgoing_edges(self, node_id: str) -> List[Edge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges if edge.source_id == node_id]
    
    def get_incoming_edges(self, node_id: str) -> List[Edge]:
        """Get all edges targeting a node."""
        return [edge for edge in self.edges if edge.target_id == node_id]
    
    def get_neighbors(self, node_id: str, direction: str = "outgoing") -> List[Node]:
        """Get all neighbors of a node."""
        if direction == "outgoing":
            edges = self.get_outgoing_edges(node_id)
            return [self.nodes[edge.target_id] for edge in edges if edge.target_id in self.nodes]
        elif direction == "incoming":
            edges = self.get_incoming_edges(node_id)
            return [self.nodes[edge.source_id] for edge in edges if edge.source_id in self.nodes]
        else:
            raise ValueError(f"Invalid direction: {direction}")
```

### 2. Dependency Extraction

Create a system for extracting dependencies from code:

```python
class DependencyExtractor:
    """Extracts dependencies from code."""
    
    def __init__(self, language_parser):
        self.language_parser = language_parser
    
    def extract_dependencies(self, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract dependencies from a file."""
        # Parse the file
        ast_node = self.language_parser.parse_file(file_path)
        
        # Extract nodes and edges
        nodes = []
        edges = []
        
        # Add a node for the file
        file_node = Node(
            id=f"file:{file_path}",
            type="file",
            name=file_path.name,
            file_path=file_path
        )
        nodes.append(file_node)
        
        # Extract imports
        import_nodes, import_edges = self._extract_imports(ast_node, file_node.id, file_path)
        nodes.extend(import_nodes)
        edges.extend(import_edges)
        
        # Extract classes
        class_nodes, class_edges = self._extract_classes(ast_node, file_node.id, file_path)
        nodes.extend(class_nodes)
        edges.extend(class_edges)
        
        # Extract functions
        function_nodes, function_edges = self._extract_functions(ast_node, file_node.id, file_path)
        nodes.extend(function_nodes)
        edges.extend(function_edges)
        
        # Extract variables
        variable_nodes, variable_edges = self._extract_variables(ast_node, file_node.id, file_path)
        nodes.extend(variable_nodes)
        edges.extend(variable_edges)
        
        return nodes, edges
    
    def _extract_imports(self, ast_node, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract import dependencies from an AST node."""
        # This would implement extraction of import dependencies
        return [], []
    
    def _extract_classes(self, ast_node, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract class dependencies from an AST node."""
        # This would implement extraction of class dependencies
        return [], []
    
    def _extract_functions(self, ast_node, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract function dependencies from an AST node."""
        # This would implement extraction of function dependencies
        return [], []
    
    def _extract_variables(self, ast_node, file_id: str, file_path: Path) -> Tuple[List[Node], List[Edge]]:
        """Extract variable dependencies from an AST node."""
        # This would implement extraction of variable dependencies
        return [], []
```

### 3. Architectural Pattern Detection

Create a system for detecting architectural patterns in the dependency graph:

```python
@dataclass
class ArchitecturalPattern:
    """An architectural pattern detected in the codebase."""
    name: str
    description: str
    nodes: List[Node]
    edges: List[Edge]
    confidence: float

class ArchitecturalPatternDetector:
    """Detects architectural patterns in the dependency graph."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        self.dependency_graph = dependency_graph
        self.patterns = []
    
    def detect_patterns(self) -> List[ArchitecturalPattern]:
        """Detect architectural patterns in the dependency graph."""
        self.patterns = []
        
        # Detect layered architecture
        self._detect_layered_architecture()
        
        # Detect microservices architecture
        self._detect_microservices_architecture()
        
        # Detect event-driven architecture
        self._detect_event_driven_architecture()
        
        # Detect model-view-controller architecture
        self._detect_mvc_architecture()
        
        # Detect repository pattern
        self._detect_repository_pattern()
        
        # Detect factory pattern
        self._detect_factory_pattern()
        
        # Detect singleton pattern
        self._detect_singleton_pattern()
        
        return self.patterns
    
    def _detect_layered_architecture(self) -> None:
        """Detect layered architecture pattern."""
        # This would implement detection of layered architecture
        pass
    
    def _detect_microservices_architecture(self) -> None:
        """Detect microservices architecture pattern."""
        # This would implement detection of microservices architecture
        pass
    
    def _detect_event_driven_architecture(self) -> None:
        """Detect event-driven architecture pattern."""
        # This would implement detection of event-driven architecture
        pass
    
    def _detect_mvc_architecture(self) -> None:
        """Detect model-view-controller architecture pattern."""
        # This would implement detection of model-view-controller architecture
        pass
    
    def _detect_repository_pattern(self) -> None:
        """Detect repository pattern."""
        # This would implement detection of repository pattern
        pass
    
    def _detect_factory_pattern(self) -> None:
        """Detect factory pattern."""
        # This would implement detection of factory pattern
        pass
    
    def _detect_singleton_pattern(self) -> None:
        """Detect singleton pattern."""
        # This would implement detection of singleton pattern
        pass
```

### 4. Inconsistency Detection

Create a system for detecting inconsistencies in the codebase:

```python
@dataclass
class Inconsistency:
    """An inconsistency detected in the codebase."""
    type: str  # "naming", "interface", "implementation", etc.
    description: str
    nodes: List[Node]
    severity: str  # "info", "warning", "error"
    suggestion: Optional[str] = None

class InconsistencyDetector:
    """Detects inconsistencies in the codebase."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        self.dependency_graph = dependency_graph
    
    def detect_inconsistencies(self) -> List[Inconsistency]:
        """Detect inconsistencies in the codebase."""
        inconsistencies = []
        
        # Detect naming inconsistencies
        naming_inconsistencies = self._detect_naming_inconsistencies()
        inconsistencies.extend(naming_inconsistencies)
        
        # Detect interface inconsistencies
        interface_inconsistencies = self._detect_interface_inconsistencies()
        inconsistencies.extend(interface_inconsistencies)
        
        # Detect implementation inconsistencies
        implementation_inconsistencies = self._detect_implementation_inconsistencies()
        inconsistencies.extend(implementation_inconsistencies)
        
        # Detect documentation inconsistencies
        documentation_inconsistencies = self._detect_documentation_inconsistencies()
        inconsistencies.extend(documentation_inconsistencies)
        
        return inconsistencies
    
    def _detect_naming_inconsistencies(self) -> List[Inconsistency]:
        """Detect naming inconsistencies."""
        # This would implement detection of naming inconsistencies
        return []
    
    def _detect_interface_inconsistencies(self) -> List[Inconsistency]:
        """Detect interface inconsistencies."""
        # This would implement detection of interface inconsistencies
        return []
    
    def _detect_implementation_inconsistencies(self) -> List[Inconsistency]:
        """Detect implementation inconsistencies."""
        # This would implement detection of implementation inconsistencies
        return []
    
    def _detect_documentation_inconsistencies(self) -> List[Inconsistency]:
        """Detect documentation inconsistencies."""
        # This would implement detection of documentation inconsistencies
        return []
```

### 5. Refactoring Opportunity Detection

Create a system for detecting refactoring opportunities:

```python
@dataclass
class RefactoringOpportunity:
    """A refactoring opportunity detected in the codebase."""
    type: str  # "extract_class", "move_method", "rename", etc.
    description: str
    nodes: List[Node]
    edges: List[Edge]
    benefit: str
    effort: str  # "low", "medium", "high"
    suggestion: Optional[str] = None

class RefactoringOpportunityDetector:
    """Detects refactoring opportunities in the codebase."""
    
    def __init__(self, dependency_graph: DependencyGraph):
        self.dependency_graph = dependency_graph
    
    def detect_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect refactoring opportunities in the codebase."""
        opportunities = []
        
        # Detect extract class opportunities
        extract_class_opportunities = self._detect_extract_class_opportunities()
        opportunities.extend(extract_class_opportunities)
        
        # Detect move method opportunities
        move_method_opportunities = self._detect_move_method_opportunities()
        opportunities.extend(move_method_opportunities)
        
        # Detect rename opportunities
        rename_opportunities = self._detect_rename_opportunities()
        opportunities.extend(rename_opportunities)
        
        # Detect extract interface opportunities
        extract_interface_opportunities = self._detect_extract_interface_opportunities()
        opportunities.extend(extract_interface_opportunities)
        
        return opportunities
    
    def _detect_extract_class_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect extract class opportunities."""
        # This would implement detection of extract class opportunities
        return []
    
    def _detect_move_method_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect move method opportunities."""
        # This would implement detection of move method opportunities
        return []
    
    def _detect_rename_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect rename opportunities."""
        # This would implement detection of rename opportunities
        return []
    
    def _detect_extract_interface_opportunities(self) -> List[RefactoringOpportunity]:
        """Detect extract interface opportunities."""
        # This would implement detection of extract interface opportunities
        return []
```

## Success Criteria

1. Construction of a comprehensive dependency graph
2. Detection of architectural patterns across the codebase
3. Identification of inconsistencies in implementation
4. Discovery of refactoring opportunities
5. Impact analysis for proposed changes

## Dependencies

- Existing AST parsing system
- Existing language parsers

## Timeline

- Research and design: 1 week
- Dependency graph construction: 2 weeks
- Dependency extraction: 2 weeks
- Architectural pattern detection: 2 weeks
- Inconsistency detection: 1 week
- Refactoring opportunity detection: 2 weeks
- Testing and documentation: 1 week

Total: 11 weeks
