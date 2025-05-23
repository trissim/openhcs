#!/usr/bin/env python3
"""
Interface Classifier - A tool to analyze and classify interfaces in Python code.

This tool:
1. Identifies abstract base classes and interfaces
2. Detects duck typing patterns
3. Analyzes interface coverage and implementation
4. Identifies protocol adherence
"""

import os
import sys
import ast
import re
import json
import argparse
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from collections import defaultdict

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

class InterfaceType(Enum):
    """Types of interfaces."""
    ABSTRACT_BASE_CLASS = auto()  # Explicit ABC
    IMPLICIT_INTERFACE = auto()   # Class with abstract methods but not ABC
    PROTOCOL = auto()             # PEP 544 Protocol
    DUCK_TYPE = auto()            # Duck typing pattern
    CONCRETE_CLASS = auto()       # Regular concrete class
    UNKNOWN = auto()              # Unknown type

class MethodType(Enum):
    """Types of methods."""
    ABSTRACT = auto()             # Abstract method
    CONCRETE = auto()             # Concrete method
    PROPERTY = auto()             # Property
    CLASSMETHOD = auto()          # Class method
    STATICMETHOD = auto()         # Static method
    UNKNOWN = auto()              # Unknown type

class ClassInfo:
    """Information about a class."""
    def __init__(self, name: str, module: str):
        self.name = name
        self.module = module
        self.interface_type = InterfaceType.UNKNOWN
        self.base_classes = []
        self.methods = {}  # Maps method name to MethodInfo
        self.properties = {}  # Maps property name to PropertyInfo
        self.docstring = None
        self.lineno = 0
        self.end_lineno = 0
        self.implementations = []  # Classes that implement this interface
        self.implements = []  # Interfaces implemented by this class

    @property
    def full_name(self) -> str:
        """Get the fully qualified name of the class."""
        return f"{self.module}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.full_name} ({self.interface_type.name})"

class MethodInfo:
    """Information about a method."""
    def __init__(self, name: str, class_info: ClassInfo):
        self.name = name
        self.class_info = class_info
        self.method_type = MethodType.UNKNOWN
        self.parameters = []
        self.return_annotation = None
        self.docstring = None
        self.lineno = 0
        self.end_lineno = 0
        self.is_async = False

    @property
    def full_name(self) -> str:
        """Get the fully qualified name of the method."""
        return f"{self.class_info.full_name}.{self.name}"

    def __repr__(self) -> str:
        return f"{self.full_name} ({self.method_type.name})"

class PropertyInfo:
    """Information about a property."""
    def __init__(self, name: str, class_info: ClassInfo):
        self.name = name
        self.class_info = class_info
        self.has_getter = False
        self.has_setter = False
        self.has_deleter = False
        self.return_annotation = None
        self.docstring = None
        self.lineno = 0

    @property
    def full_name(self) -> str:
        """Get the fully qualified name of the property."""
        return f"{self.class_info.full_name}.{self.name}"

    def __repr__(self) -> str:
        accessors = []
        if self.has_getter:
            accessors.append("getter")
        if self.has_setter:
            accessors.append("setter")
        if self.has_deleter:
            accessors.append("deleter")
        return f"{self.full_name} (property: {', '.join(accessors)})"

class InterfaceVisitor(ast.NodeVisitor):
    """AST visitor to analyze interfaces."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.classes = {}  # Maps full_name to ClassInfo
        self.current_class = None

    def visit_ClassDef(self, node):
        """Process class definitions."""
        # Create class info
        class_name = node.name
        class_info = ClassInfo(name=class_name, module=self.module_name)
        class_info.lineno = node.lineno
        class_info.end_lineno = getattr(node, 'end_lineno', node.lineno)

        # Extract base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                class_info.base_classes.append(base.id)
            elif isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name):
                    class_info.base_classes.append(f"{base.value.id}.{base.attr}")

        # Extract docstring if present
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            class_info.docstring = node.body[0].value.value

        # Store class info
        self.classes[class_info.full_name] = class_info
        old_class = self.current_class
        self.current_class = class_info

        # Visit class body
        self.generic_visit(node)

        # Determine interface type
        self._determine_interface_type(class_info)

        # Restore previous class context
        self.current_class = old_class

    def visit_FunctionDef(self, node):
        """Process function definitions."""
        self._process_method(node, is_async=False)

    def visit_AsyncFunctionDef(self, node):
        """Process async function definitions."""
        self._process_method(node, is_async=True)

    def _process_method(self, node, is_async: bool):
        """Common processing for method and async method definitions."""
        if self.current_class is None:
            # Not a method (function at module level)
            return

        # Create method info
        method_name = node.name
        method_info = MethodInfo(name=method_name, class_info=self.current_class)
        method_info.is_async = is_async
        method_info.lineno = node.lineno
        method_info.end_lineno = getattr(node, 'end_lineno', node.lineno)

        # Extract parameters
        method_info.parameters = [arg.arg for arg in node.args.args]

        # Extract return annotation if present
        if node.returns:
            if isinstance(node.returns, ast.Name):
                method_info.return_annotation = node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                method_info.return_annotation = f"{node.returns.value.id}.{node.returns.attr}"
            elif isinstance(node.returns, ast.Subscript):
                # This is a simplification; handling complex type annotations would require more work
                method_info.return_annotation = "complex_type"

        # Extract docstring if present
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            method_info.docstring = node.body[0].value.value

        # Check for property decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'property':
                # This is a property getter
                property_name = method_name
                if property_name not in self.current_class.properties:
                    self.current_class.properties[property_name] = PropertyInfo(
                        name=property_name, class_info=self.current_class)
                self.current_class.properties[property_name].has_getter = True
                self.current_class.properties[property_name].return_annotation = method_info.return_annotation
                self.current_class.properties[property_name].docstring = method_info.docstring
                self.current_class.properties[property_name].lineno = method_info.lineno
                return
            elif isinstance(decorator, ast.Attribute) and isinstance(decorator.value, ast.Name):
                if decorator.attr == 'setter' and decorator.value.id in self.current_class.properties:
                    # This is a property setter
                    self.current_class.properties[decorator.value.id].has_setter = True
                    return
                elif decorator.attr == 'deleter' and decorator.value.id in self.current_class.properties:
                    # This is a property deleter
                    self.current_class.properties[decorator.value.id].has_deleter = True
                    return
            elif isinstance(decorator, ast.Name) and decorator.id == 'abstractmethod':
                # This is an abstract method
                method_info.method_type = MethodType.ABSTRACT
            elif isinstance(decorator, ast.Name) and decorator.id == 'classmethod':
                # This is a class method
                method_info.method_type = MethodType.CLASSMETHOD
            elif isinstance(decorator, ast.Name) and decorator.id == 'staticmethod':
                # This is a static method
                method_info.method_type = MethodType.STATICMETHOD

        # Check for abstract method pattern (raising NotImplementedError)
        if method_info.method_type == MethodType.UNKNOWN:
            for stmt in node.body:
                if isinstance(stmt, ast.Raise):
                    if isinstance(stmt.exc, ast.Name) and stmt.exc.id == 'NotImplementedError':
                        method_info.method_type = MethodType.ABSTRACT
                        break
                    elif isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == 'NotImplementedError':
                        method_info.method_type = MethodType.ABSTRACT
                        break

        # If still unknown, it's a concrete method
        if method_info.method_type == MethodType.UNKNOWN:
            method_info.method_type = MethodType.CONCRETE

        # Store method info
        self.current_class.methods[method_name] = method_info

    def _determine_interface_type(self, class_info: ClassInfo):
        """Determine the interface type of a class."""
        # Check for ABC
        if 'ABC' in class_info.base_classes or 'abc.ABC' in class_info.base_classes:
            class_info.interface_type = InterfaceType.ABSTRACT_BASE_CLASS
            return

        # Check for Protocol
        if 'Protocol' in class_info.base_classes or 'typing.Protocol' in class_info.base_classes:
            class_info.interface_type = InterfaceType.PROTOCOL
            return

        # Check for abstract methods
        has_abstract_methods = any(method.method_type == MethodType.ABSTRACT
                                  for method in class_info.methods.values())
        if has_abstract_methods:
            class_info.interface_type = InterfaceType.IMPLICIT_INTERFACE
            return

        # Default to concrete class
        class_info.interface_type = InterfaceType.CONCRETE_CLASS

def analyze_file(file_path: str) -> Dict[str, ClassInfo]:
    """
    Analyze a file to identify interfaces.

    Args:
        file_path: Path to the file to analyze

    Returns:
        Dictionary mapping class names to ClassInfo objects
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine module name from file path
        rel_path = os.path.relpath(file_path, PROJECT_ROOT)
        module_name = os.path.splitext(rel_path.replace(os.path.sep, '.'))[0]

        # Parse the file
        tree = ast.parse(content, filename=file_path)

        # Analyze interfaces
        visitor = InterfaceVisitor(module_name)
        visitor.visit(tree)

        return visitor.classes

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {}

def analyze_directory(directory: str, exclude_patterns: List[str] = None) -> Dict[str, ClassInfo]:
    """
    Analyze all Python files in a directory to identify interfaces.

    Args:
        directory: Directory to analyze
        exclude_patterns: List of patterns to exclude

    Returns:
        Dictionary mapping class names to ClassInfo objects
    """
    if exclude_patterns is None:
        exclude_patterns = ["*__pycache__*", "*.git*", "*venv*", "*env*"]

    all_classes = {}

    for root, _, files in os.walk(directory):
        # Check if this directory should be excluded
        if any(pattern in root for pattern in exclude_patterns):
            continue

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if this file should be excluded
                if not any(pattern in file_path for pattern in exclude_patterns):
                    classes = analyze_file(file_path)
                    all_classes.update(classes)

    # Identify implementations
    _identify_implementations(all_classes)

    return all_classes

def _identify_implementations(classes: Dict[str, ClassInfo]):
    """
    Identify implementations of interfaces.

    Args:
        classes: Dictionary mapping class names to ClassInfo objects
    """
    # First pass: identify direct implementations
    for class_name, class_info in classes.items():
        for base_class in class_info.base_classes:
            # Check if the base class is in our analyzed classes
            if base_class in classes:
                base_class_info = classes[base_class]
                class_info.implements.append(base_class)
                base_class_info.implementations.append(class_name)
            else:
                # Check for fully qualified base class names
                for potential_base in classes.values():
                    if potential_base.name == base_class:
                        class_info.implements.append(potential_base.full_name)
                        potential_base.implementations.append(class_name)

    # Second pass: identify duck typing implementations
    for class_name, class_info in classes.items():
        if class_info.interface_type in [InterfaceType.CONCRETE_CLASS, InterfaceType.UNKNOWN]:
            # Check if this class implements any interfaces through duck typing
            for potential_interface in classes.values():
                if potential_interface.interface_type in [InterfaceType.ABSTRACT_BASE_CLASS,
                                                         InterfaceType.IMPLICIT_INTERFACE,
                                                         InterfaceType.PROTOCOL]:
                    # Check if all abstract methods in the interface are implemented in this class
                    implements_all = True
                    for method_name, method_info in potential_interface.methods.items():
                        if method_info.method_type == MethodType.ABSTRACT:
                            if method_name not in class_info.methods:
                                implements_all = False
                                break

                    if implements_all and potential_interface.methods:
                        # This class implements the interface through duck typing
                        if potential_interface.full_name not in class_info.implements:
                            class_info.implements.append(potential_interface.full_name)
                        if class_name not in potential_interface.implementations:
                            potential_interface.implementations.append(class_name)
                        class_info.interface_type = InterfaceType.DUCK_TYPE

def format_results_markdown(classes: Dict[str, ClassInfo]) -> str:
    """
    Format interface analysis results as Markdown.

    Args:
        classes: Dictionary mapping class names to ClassInfo objects

    Returns:
        Markdown string
    """
    lines = []
    lines.append("# Interface Analysis")
    lines.append(f"\nAnalyzed {len(classes)} classes.\n")

    # Group classes by interface type
    classes_by_type = defaultdict(list)
    for class_info in classes.values():
        classes_by_type[class_info.interface_type].append(class_info)

    # Report interfaces
    interfaces = []
    for interface_type in [InterfaceType.ABSTRACT_BASE_CLASS, InterfaceType.IMPLICIT_INTERFACE, InterfaceType.PROTOCOL]:
        interfaces.extend(classes_by_type[interface_type])

    if interfaces:
        lines.append("## Interfaces")
        lines.append(f"Found {len(interfaces)} interfaces:")
        lines.append("```")
        for interface in sorted(interfaces, key=lambda c: c.full_name):
            abstract_methods = [m.name for m in interface.methods.values()
                               if m.method_type == MethodType.ABSTRACT]
            lines.append(f"- {interface.full_name} ({interface.interface_type.name})")
            if abstract_methods:
                lines.append(f"  Abstract methods: {', '.join(abstract_methods)}")
            if interface.implementations:
                lines.append(f"  Implementations: {len(interface.implementations)}")
        lines.append("```")

    # Report implementations
    implementations = classes_by_type[InterfaceType.CONCRETE_CLASS] + classes_by_type[InterfaceType.DUCK_TYPE]
    if implementations:
        lines.append("\n## Implementations")
        lines.append(f"Found {len(implementations)} implementations:")
        lines.append("```")
        for impl in sorted(implementations, key=lambda c: c.full_name):
            if impl.implements:
                lines.append(f"- {impl.full_name} implements: {', '.join(impl.implements)}")
        lines.append("```")

    # Report interface coverage
    if interfaces:
        lines.append("\n## Interface Coverage")
        lines.append("Coverage of interfaces by implementations:")
        lines.append("```")
        for interface in sorted(interfaces, key=lambda c: c.full_name):
            if interface.implementations:
                lines.append(f"- {interface.full_name}: {len(interface.implementations)} implementations")
                for impl_name in sorted(interface.implementations):
                    if impl_name in classes:
                        impl = classes[impl_name]
                        lines.append(f"  - {impl.full_name} ({impl.interface_type.name})")
            else:
                lines.append(f"- {interface.full_name}: No implementations")
        lines.append("```")

    return "\n".join(lines)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Interface Classifier - Analyze and classify interfaces in Python code",
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
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching pattern (can be used multiple times)"
    )

    args = parser.parse_args()

    # Analyze the code
    if os.path.isfile(args.path):
        classes = analyze_file(args.path)
    elif os.path.isdir(args.path):
        classes = analyze_directory(args.path, args.exclude)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return 1

    # Generate the report
    report = format_results_markdown(classes)

    # Output the report
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    return 0

if __name__ == "__main__":
    sys.exit(main())
