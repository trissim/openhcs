#!/usr/bin/env python3
"""
Call Graph Analyzer - A tool to analyze function call relationships in Python code.

This tool:
1. Builds a call graph showing which functions call which other functions
2. Identifies entry points and leaf functions
3. Detects potential circular dependencies
4. Visualizes the call hierarchy
"""

import os
import sys
import ast
import json
import argparse
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from collections import defaultdict

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

class FunctionInfo:
    """Information about a function or method."""
    def __init__(self, name: str, module: str, class_name: Optional[str] = None):
        self.name = name
        self.module = module
        self.class_name = class_name
        self.calls = set()  # Set of FunctionInfo objects this function calls
        self.called_by = set()  # Set of FunctionInfo objects that call this function
        self.is_async = False
        self.parameters = []
        self.return_annotation = None
        self.docstring = None
        self.lineno = 0
        self.end_lineno = 0

    @property
    def full_name(self) -> str:
        """Get the fully qualified name of the function."""
        if self.class_name:
            return f"{self.module}.{self.class_name}.{self.name}"
        return f"{self.module}.{self.name}"

    def __repr__(self) -> str:
        return self.full_name

    def __hash__(self) -> int:
        return hash(self.full_name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionInfo):
            return False
        return self.full_name == other.full_name

class CallGraphVisitor(ast.NodeVisitor):
    """AST visitor to build a call graph."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.functions = {}  # Maps full_name to FunctionInfo
        self.current_function = None
        self.current_class = None

    def visit_ClassDef(self, node):
        """Process class definitions."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Process function definitions."""
        self._process_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node):
        """Process async function definitions."""
        self._process_function(node, is_async=True)

    def _process_function(self, node, is_async: bool):
        """Common processing for function and async function definitions."""
        old_function = self.current_function

        # Create function info
        function_name = node.name
        function_info = FunctionInfo(
            name=function_name,
            module=self.module_name,
            class_name=self.current_class
        )
        function_info.is_async = is_async
        function_info.lineno = node.lineno
        function_info.end_lineno = getattr(node, 'end_lineno', node.lineno)

        # Extract parameters
        function_info.parameters = [arg.arg for arg in node.args.args]

        # Extract return annotation if present
        if node.returns:
            if isinstance(node.returns, ast.Name):
                function_info.return_annotation = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                function_info.return_annotation = f"{node.returns.value.id}.{node.returns.attr}"
            elif isinstance(node.returns, ast.Subscript):
                # This is a simplification; handling complex type annotations would require more work
                function_info.return_annotation = "complex_type"

        # Extract docstring if present
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            function_info.docstring = node.body[0].value.value

        # Store function info
        self.functions[function_info.full_name] = function_info
        self.current_function = function_info

        # Visit function body to find calls
        self.generic_visit(node)

        # Restore previous function context
        self.current_function = old_function

    def visit_Call(self, node):
        """Process function calls."""
        if self.current_function is None:
            # Call outside of a function (module level)
            self.generic_visit(node)
            return

        # Try to determine the called function
        called_name = None
        if isinstance(node.func, ast.Name):
            # Direct function call: func()
            called_name = node.func.id
            # Look for a function with this name in the current module
            for func_info in self.functions.values():
                if func_info.name == called_name and func_info.class_name is None:
                    self.current_function.calls.add(func_info)
                    func_info.called_by.add(self.current_function)
        elif isinstance(node.func, ast.Attribute):
            # Method call: obj.method()
            if isinstance(node.func.value, ast.Name):
                # Simple attribute access: obj.method()
                obj_name = node.func.value.id
                method_name = node.func.attr

                # Special case for 'self' in methods
                if obj_name == 'self' and self.current_class:
                    # Look for a method with this name in the current class
                    for func_info in self.functions.values():
                        if (func_info.name == method_name and
                                func_info.class_name == self.current_class):
                            self.current_function.calls.add(func_info)
                            func_info.called_by.add(self.current_function)

        self.generic_visit(node)

def analyze_file(file_path: str) -> Dict[str, FunctionInfo]:
    """
    Analyze a file to build a call graph.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Dictionary mapping function names to FunctionInfo objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine module name from file path
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        module_name = os.path.splitext(rel_path.replace(os.path.sep, '.'))[0]

        # Parse the file
        tree = ast.parse(content, filename=file_path)

        # Build the call graph
        visitor = CallGraphVisitor(module_name)
        visitor.visit(tree)

        return visitor.functions

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {}

def analyze_directory(directory: str, exclude_patterns: List[str] = None) -> Dict[str, FunctionInfo]:
    """
    Analyze all Python files in a directory to build a call graph.

    Args:
        directory: Directory to analyze
        exclude_patterns: List of patterns to exclude

    Returns:
        Dictionary mapping function names to FunctionInfo objects
    """
    if exclude_patterns is None:
        exclude_patterns = ["*__pycache__*", "*.git*", "*venv*", "*env*"]

    all_functions = {}

    for root, _, files in os.walk(directory):
        # Check if this directory should be excluded
        if any(pattern in root for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if this file should be excluded
                if not any(pattern in file_path for pattern in exclude_patterns):
                    functions = analyze_file(file_path)
                    all_functions.update(functions)

    return all_functions

def build_call_graph(functions: Dict[str, FunctionInfo]) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from function call relationships.

    Args:
        functions: Dictionary mapping function names to FunctionInfo objects

    Returns:
        NetworkX DiGraph representing the call graph
    """
    G = nx.DiGraph()

    # Add nodes
    for func_name, func_info in functions.items():
        G.add_node(func_name,
                   module=func_info.module,
                   class_name=func_info.class_name,
                   name=func_info.name,
                   is_async=func_info.is_async,
                   parameters=func_info.parameters,
                   return_annotation=func_info.return_annotation,
                   docstring=func_info.docstring,
                   lineno=func_info.lineno)

    # Add edges
    for func_name, func_info in functions.items():
        for called_func in func_info.calls:
            if called_func.full_name in functions:
                G.add_edge(func_name, called_func.full_name)

    return G

def find_entry_points(G: nx.DiGraph) -> List[str]:
    """
    Find entry points in the call graph (functions that are not called by others).

    Args:
        G: NetworkX DiGraph representing the call graph

    Returns:
        List of function names that are entry points
    """
    entry_points = []
    for node in G.nodes():
        if G.in_degree(node) == 0:
            entry_points.append(node)
    return entry_points

def find_leaf_functions(G: nx.DiGraph) -> List[str]:
    """
    Find leaf functions in the call graph (functions that don't call others).

    Args:
        G: NetworkX DiGraph representing the call graph

    Returns:
        List of function names that are leaf functions
    """
    leaf_functions = []
    for node in G.nodes():
        if G.out_degree(node) == 0:
            leaf_functions.append(node)
    return leaf_functions

def find_cycles(G: nx.DiGraph) -> List[List[str]]:
    """
    Find cycles in the call graph (potential circular dependencies).

    Args:
        G: NetworkX DiGraph representing the call graph

    Returns:
        List of cycles, where each cycle is a list of function names
    """
    try:
        return list(nx.simple_cycles(G))
    except nx.NetworkXNoCycle:
        return []

def visualize_call_graph(G: nx.DiGraph, output_file: str = None):
    """
    Visualize the call graph.

    Args:
        G: NetworkX DiGraph representing the call graph
        output_file: Path to save the visualization (if None, display interactively)
    """
    plt.figure(figsize=(12, 8))

    # Use a hierarchical layout
    pos = nx.spring_layout(G, seed=42)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True)

    # Draw labels
    labels = {node: G.nodes[node]['name'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    plt.title("Function Call Graph")
    plt.axis('off')

    if output_file:
        plt.savefig(output_file)
    else:
        plt.show()

def format_results_markdown(G: nx.DiGraph, project_root: str) -> str:
    """
    Format call graph analysis results as Markdown.

    Args:
        G: NetworkX DiGraph representing the call graph
        project_root: Project root directory for relative paths

    Returns:
        Markdown string
    """
    lines = []
    lines.append("# Call Graph Analysis")
    lines.append(f"\nAnalyzed {len(G.nodes())} functions with {len(G.edges())} call relationships.\n")

    # Entry points
    entry_points = find_entry_points(G)
    lines.append("## Entry Points")
    lines.append("Functions that are not called by any other function:")
    lines.append("```")
    for func in sorted(entry_points):
        module = G.nodes[func]['module']
        name = G.nodes[func]['name']
        class_name = G.nodes[func]['class_name']
        if class_name:
            lines.append(f"- {module}.{class_name}.{name}")
        else:
            lines.append(f"- {module}.{name}")
    lines.append("```")

    # Leaf functions
    leaf_functions = find_leaf_functions(G)
    lines.append("\n## Leaf Functions")
    lines.append("Functions that don't call any other function:")
    lines.append("```")
    for func in sorted(leaf_functions):
        module = G.nodes[func]['module']
        name = G.nodes[func]['name']
        class_name = G.nodes[func]['class_name']
        if class_name:
            lines.append(f"- {module}.{class_name}.{name}")
        else:
            lines.append(f"- {module}.{name}")
    lines.append("```")

    # Cycles
    cycles = find_cycles(G)
    if cycles:
        lines.append("\n## Circular Dependencies")
        lines.append("Potential circular dependencies in the call graph:")
        lines.append("```")
        for i, cycle in enumerate(cycles, 1):
            lines.append(f"Cycle {i}:")
            for func in cycle:
                module = G.nodes[func]['module']
                name = G.nodes[func]['name']
                class_name = G.nodes[func]['class_name']
                if class_name:
                    lines.append(f"  - {module}.{class_name}.{name}")
                else:
                    lines.append(f"  - {module}.{name}")
            lines.append("")
        lines.append("```")

    # Most called functions
    in_degrees = sorted([(func, G.in_degree(func)) for func in G.nodes()],
                        key=lambda x: x[1], reverse=True)
    lines.append("\n## Most Called Functions")
    lines.append("Functions called by the most other functions:")
    lines.append("```")
    for func, degree in in_degrees[:10]:  # Top 10
        if degree == 0:
            continue
        module = G.nodes[func]['module']
        name = G.nodes[func]['name']
        class_name = G.nodes[func]['class_name']
        if class_name:
            lines.append(f"- {module}.{class_name}.{name}: {degree} callers")
        else:
            lines.append(f"- {module}.{name}: {degree} callers")
    lines.append("```")

    # Functions that call the most other functions
    out_degrees = sorted([(func, G.out_degree(func)) for func in G.nodes()],
                         key=lambda x: x[1], reverse=True)
    lines.append("\n## Functions with Most Calls")
    lines.append("Functions that call the most other functions:")
    lines.append("```")
    for func, degree in out_degrees[:10]:  # Top 10
        if degree == 0:
            continue
        module = G.nodes[func]['module']
        name = G.nodes[func]['name']
        class_name = G.nodes[func]['class_name']
        if class_name:
            lines.append(f"- {module}.{class_name}.{name}: {degree} callees")
        else:
            lines.append(f"- {module}.{name}: {degree} callees")
    lines.append("```")

    return "\n".join(lines)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Call Graph Analyzer - Analyze function call relationships in Python code",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "path",
        help="File or directory to analyze"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output file for the report (default: stdout)"
    )

    parser.add_argument(
        "--visualize",
        help="Generate a visualization of the call graph and save to the specified file"
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching pattern (can be used multiple times)"
    )

    args = parser.parse_args()

    # Analyze the code
    if os.path.isfile(args.path):
        functions = analyze_file(args.path)
    elif os.path.isdir(args.path):
        functions = analyze_directory(args.path, args.exclude)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return 1

    # Build the call graph
    G = build_call_graph(functions)

    # Generate the report
    report = format_results_markdown(G, PROJECT_ROOT)

    # Output the report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Generate visualization if requested
    if args.visualize:
        try:
            visualize_call_graph(G, args.visualize)
            print(f"Visualization saved to {args.visualize}")
        except Exception as e:
            print(f"Error generating visualization: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
