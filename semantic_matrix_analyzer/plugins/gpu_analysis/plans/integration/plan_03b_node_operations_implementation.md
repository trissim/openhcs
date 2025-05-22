# Plan 03b: Node Operations Implementation

## Objective

Implement all node-related methods required by SMA's `LanguageParser` interface in the GPU Language Parser, ensuring proper handling of both standard and GPU-accelerated AST formats.

## Background

SMA's `LanguageParser` interface requires several methods for working with AST nodes, such as getting node types, names, children, and source ranges. These methods are essential for SMA's code analysis capabilities and must be properly implemented in the GPU Language Parser to enable GPU-accelerated analysis of unimplemented SMA methods.

## Current State

The current GPU Language Parser doesn't implement the node-related methods required by SMA's `LanguageParser` interface. These methods are essential for traversing and analyzing AST nodes, which is a core part of SMA's functionality.

SMA's `LanguageParser` interface in `semantic_matrix_analyzer/semantic_matrix_analyzer/language/__init__.py` requires the following node-related methods:

```python
@abstractmethod
def get_node_type(self, node: Any) -> str:
    """Get the type of an AST node.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A string representing the type of the node.
    """
    pass

@abstractmethod
def get_node_name(self, node: Any) -> Optional[str]:
    """Get the name of an AST node, if applicable.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        The name of the node, or None if the node does not have a name.
    """
    pass

@abstractmethod
def get_node_children(self, node: Any) -> List[Any]:
    """Get the children of an AST node.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A list of child nodes.
    """
    pass

@abstractmethod
def get_node_source_range(self, node: Any) -> Optional[Tuple[int, int]]:
    """Get the source range of an AST node.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A tuple of (start_line, end_line), or None if not available.
        Line numbers are 1-based.
    """
    pass

@abstractmethod
def get_node_source(self, node: Any, file_content: str) -> Optional[str]:
    """Get the source code for an AST node.

    Args:
        node: An AST node returned by parse_file.
        file_content: The content of the file.

    Returns:
        The source code for the node, or None if not available.
    """
    pass
```

## Implementation Plan

### 1. Implement Node Type Method

Implement the `get_node_type` method to handle both standard and GPU-accelerated AST formats:

```python
def get_node_type(self, node: Any) -> str:
    """
    Get the type of an AST node.

    This method handles both standard AST nodes and GPU-accelerated AST nodes,
    delegating to the base parser for standard nodes and extracting type information
    from GPU-accelerated nodes.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A string representing the type of the node.
    """
    try:
        # If node is a dict with both ast and gpu_ast, use the ast
        if isinstance(node, dict) and "ast" in node:
            return self.base_parser.get_node_type(node["ast"])

        # If node is a dict with gpu_ast only, extract type from tensor
        elif isinstance(node, dict) and "gpu_ast" in node:
            # Extract type from GPU tensor
            gpu_ast = node["gpu_ast"]
            if "node_types" in gpu_ast and "node_indices" in gpu_ast:
                # Get the type index for the root node (index 0)
                type_idx = gpu_ast["node_types"][0].item()
                # Map type index to type name using tensorizer's type mapping
                return self.tensorizer.get_type_name(type_idx)
            else:
                raise ValueError("Invalid GPU AST format: missing node_types or node_indices")

        # Otherwise, delegate to base parser
        return self.base_parser.get_node_type(node)
    except Exception as e:
        logger.error(f"Error getting node type: {e}")
        # Fall back to a generic type
        return "unknown"
```

### 2. Implement Node Name Method

Implement the `get_node_name` method to handle both standard and GPU-accelerated AST formats:

```python
def get_node_name(self, node: Any) -> Optional[str]:
    """
    Get the name of an AST node, if applicable.

    This method handles both standard AST nodes and GPU-accelerated AST nodes,
    delegating to the base parser for standard nodes and extracting name information
    from GPU-accelerated nodes.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        The name of the node, or None if the node does not have a name.
    """
    try:
        # If node is a dict with both ast and gpu_ast, use the ast
        if isinstance(node, dict) and "ast" in node:
            return self.base_parser.get_node_name(node["ast"])

        # If node is a dict with gpu_ast only, extract name from tensor
        elif isinstance(node, dict) and "gpu_ast" in node:
            # Extract name from GPU tensor
            gpu_ast = node["gpu_ast"]
            if "node_names" in gpu_ast and len(gpu_ast["node_names"]) > 0:
                # Get the name for the root node (index 0)
                name_idx = gpu_ast["node_names"][0].item()
                # Map name index to name using tensorizer's name mapping
                return self.tensorizer.get_name(name_idx) if name_idx >= 0 else None
            else:
                # Node might not have a name
                return None

        # Otherwise, delegate to base parser
        return self.base_parser.get_node_name(node)
    except Exception as e:
        logger.error(f"Error getting node name: {e}")
        return None
```

### 3. Implement Node Children Method

Implement the `get_node_children` method to handle both standard and GPU-accelerated AST formats:

```python
def get_node_children(self, node: Any) -> List[Any]:
    """
    Get the children of an AST node.

    This method handles both standard AST nodes and GPU-accelerated AST nodes,
    delegating to the base parser for standard nodes and extracting children
    from GPU-accelerated nodes.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A list of child nodes.
    """
    try:
        # If node is a dict with both ast and gpu_ast, use the ast
        if isinstance(node, dict) and "ast" in node:
            # Get children from standard AST
            ast_children = self.base_parser.get_node_children(node["ast"])

            # If gpu_ast is available, create combined nodes for children
            if "gpu_ast" in node:
                gpu_ast = node["gpu_ast"]
                if "node_children" in gpu_ast and "node_indices" in gpu_ast:
                    # Get children indices for the root node (index 0)
                    children_indices = self.tensorizer.get_children_indices(gpu_ast, 0)

                    # Create combined nodes for each child
                    combined_children = []
                    for i, ast_child in enumerate(ast_children):
                        if i < len(children_indices):
                            child_idx = children_indices[i]
                            gpu_child = self.tensorizer.extract_subtree(gpu_ast, child_idx)
                            combined_children.append({
                                "ast": ast_child,
                                "gpu_ast": gpu_child,
                                "file_path": node.get("file_path")
                            })
                        else:
                            # Fall back to standard AST if indices don't match
                            combined_children.append({
                                "ast": ast_child,
                                "file_path": node.get("file_path")
                            })

                    return combined_children

            # If no gpu_ast or error, return standard AST children
            return ast_children

        # If node is a dict with gpu_ast only, extract children from tensor
        elif isinstance(node, dict) and "gpu_ast" in node:
            gpu_ast = node["gpu_ast"]
            if "node_children" in gpu_ast and "node_indices" in gpu_ast:
                # Get children indices for the root node (index 0)
                children_indices = self.tensorizer.get_children_indices(gpu_ast, 0)

                # Create GPU-only nodes for each child
                gpu_children = []
                for child_idx in children_indices:
                    gpu_child = self.tensorizer.extract_subtree(gpu_ast, child_idx)
                    gpu_children.append({
                        "gpu_ast": gpu_child,
                        "file_path": node.get("file_path")
                    })

                return gpu_children
            else:
                # No children information available
                return []

        # Otherwise, delegate to base parser
        return self.base_parser.get_node_children(node)
    except Exception as e:
        logger.error(f"Error getting node children: {e}")
        return []
```

### 4. Implement Node Source Range Method

Implement the `get_node_source_range` method to handle both standard and GPU-accelerated AST formats:

```python
def get_node_source_range(self, node: Any) -> Optional[Tuple[int, int]]:
    """
    Get the source range of an AST node.

    This method handles both standard AST nodes and GPU-accelerated AST nodes,
    delegating to the base parser for standard nodes and extracting source range
    from GPU-accelerated nodes.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A tuple of (start_line, end_line), or None if not available.
        Line numbers are 1-based.
    """
    try:
        # If node is a dict with both ast and gpu_ast, use the ast
        if isinstance(node, dict) and "ast" in node:
            return self.base_parser.get_node_source_range(node["ast"])

        # If node is a dict with gpu_ast only, extract source range from tensor
        elif isinstance(node, dict) and "gpu_ast" in node:
            gpu_ast = node["gpu_ast"]
            if "node_line_ranges" in gpu_ast:
                # Get the source range for the root node (index 0)
                line_ranges = gpu_ast["node_line_ranges"]
                if line_ranges.shape[0] > 0:
                    start_line = line_ranges[0, 0].item()
                    end_line = line_ranges[0, 1].item()
                    # Convert to 1-based line numbers if they're 0-based
                    if start_line == 0:
                        start_line = 1
                    return (start_line, end_line)

            # No source range information available
            return None

        # Otherwise, delegate to base parser
        return self.base_parser.get_node_source_range(node)
    except Exception as e:
        logger.error(f"Error getting node source range: {e}")
        return None
```

### 5. Implement Node Source Method

Implement the `get_node_source` method to handle both standard and GPU-accelerated AST formats:

```python
def get_node_source(self, node: Any, file_content: str) -> Optional[str]:
    """
    Get the source code for an AST node.

    This method handles both standard AST nodes and GPU-accelerated AST nodes,
    delegating to the base parser for standard nodes and extracting source code
    from GPU-accelerated nodes using source ranges.

    Args:
        node: An AST node returned by parse_file.
        file_content: The content of the file.

    Returns:
        The source code for the node, or None if not available.
    """
    try:
        # If node is a dict with both ast and gpu_ast, use the ast
        if isinstance(node, dict) and "ast" in node:
            return self.base_parser.get_node_source(node["ast"], file_content)

        # If node is a dict with gpu_ast only, extract source using source range
        elif isinstance(node, dict) and "gpu_ast" in node:
            # Get source range
            source_range = self.get_node_source_range(node)
            if source_range is None:
                return None

            # Extract source code using source range
            start_line, end_line = source_range
            lines = file_content.splitlines()

            # Check if line numbers are valid
            if start_line < 1 or start_line > len(lines) or end_line < 1 or end_line > len(lines):
                return None

            # Extract source code (convert to 0-based indices)
            source_lines = lines[start_line - 1:end_line]
            return "\n".join(source_lines)

        # Otherwise, delegate to base parser
        return self.base_parser.get_node_source(node, file_content)
    except Exception as e:
        logger.error(f"Error getting node source: {e}")
        return None
```

### 6. Add Helper Methods for Node Operations

Add helper methods to support the node operations:

```python
def get_node_attributes(self, node: Any) -> Dict[str, Any]:
    """
    Get the attributes of an AST node.

    This method extracts attributes from both standard and GPU-accelerated AST nodes,
    providing a unified interface for accessing node attributes.

    Args:
        node: An AST node returned by parse_file.

    Returns:
        A dictionary of node attributes.
    """
    try:
        # If node is a dict with both ast and gpu_ast, use the ast
        if isinstance(node, dict) and "ast" in node:
            # Get attributes from standard AST
            if hasattr(self.base_parser, 'get_node_attributes'):
                return self.base_parser.get_node_attributes(node["ast"])
            else:
                # Fall back to ast module
                import ast
                if isinstance(node["ast"], ast.AST):
                    return {name: getattr(node["ast"], name) for name in node["ast"]._fields}

        # If node is a dict with gpu_ast only, extract attributes from tensor
        elif isinstance(node, dict) and "gpu_ast" in node:
            gpu_ast = node["gpu_ast"]
            if "node_attributes" in gpu_ast:
                # Get attributes for the root node (index 0)
                return self.tensorizer.get_node_attributes(gpu_ast, 0)

        # No attributes available
        return {}
    except Exception as e:
        logger.error(f"Error getting node attributes: {e}")
        return {}

def is_node_of_type(self, node: Any, node_type: str) -> bool:
    """
    Check if a node is of a specific type.

    This method checks if a node is of a specific type, handling both
    standard and GPU-accelerated AST nodes.

    Args:
        node: An AST node returned by parse_file.
        node_type: The type to check for.

    Returns:
        True if the node is of the specified type, False otherwise.
    """
    try:
        # Get node type
        actual_type = self.get_node_type(node)

        # Check if types match
        return actual_type == node_type
    except Exception as e:
        logger.error(f"Error checking node type: {e}")
        return False
```

## Implementation Focus

The implementation should focus on:

1. **Node Method Implementation**: Implementing all node-related methods required by SMA's `LanguageParser` interface.

2. **Format Handling**: Ensuring methods handle both standard AST nodes and GPU-accelerated AST nodes correctly.

3. **Behavioral Consistency**: Ensuring methods provide consistent behavior regardless of the AST format.

4. **Basic Error Handling**: Implementing essential error handling for architectural correctness.

5. **GPU Acceleration**: Enabling GPU-accelerated analysis of unimplemented SMA methods.

## Success Criteria

1. All node-related methods required by SMA's `LanguageParser` interface are correctly implemented in the GPU Language Parser.

2. The methods handle both standard AST nodes and GPU-accelerated AST nodes correctly.

3. The methods provide consistent behavior regardless of the AST format.

4. The methods enable GPU-accelerated analysis of unimplemented SMA methods.

Note: Testing will be deferred until the complete architecture is implemented and stable. The focus is on velocity and architectural completion rather than incremental validation.

## References

1. SMA Language Parser Interface: `semantic_matrix_analyzer/semantic_matrix_analyzer/language/__init__.py`

2. SMA Python Parser: `semantic_matrix_analyzer/semantic_matrix_analyzer/language/python_parser.py`

3. GPU AST Adapter: `brain/gpu_analysis/ast_adapter.py`

4. GPU AST Tensorizer: `brain/gpu_analysis/ast_tensor.py`
