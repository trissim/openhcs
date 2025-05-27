#!/usr/bin/env python3
"""
CodeArchitect - A tool for extracting architectural insights from Python code.

This tool analyzes Python files to extract information about their structure,
including classes, functions, methods, and type annotations. It's designed to
help agents and humans understand code architecture without reading all files in full.
"""

import ast
import sys
import os
import json
import argparse
import glob
import re
from collections import Counter, defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple, Union
import textwrap
from enum import Enum
import importlib.util

# Check for optional dependencies
try:
    import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    # Define dummy color constants
    class DummyFore:
        RED = YELLOW = GREEN = BLUE = CYAN = MAGENTA = WHITE = RESET = ""
    class DummyStyle:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""
    Fore = DummyFore()
    Style = DummyStyle()

class OutputFormat(Enum):
    """Output format options."""
    JSON = "json"
    TABLE = "table"
    TEXT = "text"
    MARKDOWN = "markdown"
    CSV = "csv"

class AnalysisLevel(Enum):
    """Analysis detail level options."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

class DefinitionExtractor(ast.NodeVisitor):
    """
    AST visitor that extracts definitions and their relationships from Python code.

    This class walks the AST of a Python file and collects information about:
    - Classes and their inheritance hierarchy
    - Functions and methods
    - Type annotations for parameters and return values
    - Imports and dependencies
    - Decorators and their usage
    - Class attributes and properties
    """

    def __init__(self, filepath: str, analysis_level: AnalysisLevel = AnalysisLevel.DETAILED):
        self.filepath = filepath
        self.analysis_level = analysis_level

        # Basic information
        self.classes: List[str] = []
        self.functions: List[str] = []
        self.methods: List[str] = []
        self.param_types: List[str] = []
        self.return_types: List[str] = []

        # Detailed information
        self.imports: Dict[str, List[str]] = {}  # module -> [imported names]
        self.class_bases: Dict[str, List[str]] = {}  # class -> [base classes]
        self.class_methods: Dict[str, List[str]] = {}  # class -> [methods]
        self.function_params: Dict[str, List[str]] = {}  # function -> [parameters]
        self.method_params: Dict[str, List[str]] = {}  # class.method -> [parameters]
        self.decorators: Dict[str, List[str]] = {}  # function/method -> [decorators]
        self.class_attributes: Dict[str, List[str]] = {}  # class -> [attributes]
        self.function_calls: Dict[str, List[str]] = {}  # function/method -> [called functions]

        # State tracking
        self.current_class_name: Optional[str] = None
        self.current_function_name: Optional[str] = None
        self.in_function_def: bool = False

    def visit_Import(self, node: ast.Import) -> None:
        """Extract information from import statements."""
        for name in node.names:
            alias = name.asname or name.name
            self.imports[name.name] = self.imports.get(name.name, []) + [alias]
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Extract information from from-import statements."""
        if node.module is not None:
            module = node.module
            for name in node.names:
                alias = name.asname or name.name
                self.imports[module] = self.imports.get(module, []) + [f"{name.name} as {alias}" if name.asname else name.name]
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract information from class definitions."""
        self.classes.append(node.name)

        # Track base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(f"{self._get_attribute_name(base)}")
        self.class_bases[node.name] = bases

        # Track decorators
        if node.decorator_list:
            self.decorators[node.name] = [self._get_decorator_name(d) for d in node.decorator_list]

        # Initialize class methods list
        self.class_methods[node.name] = []
        self.class_attributes[node.name] = []

        # Visit class body with class context
        old_class_name = self.current_class_name
        self.current_class_name = node.name
        self.generic_visit(node)
        self.current_class_name = old_class_name

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract information from function definitions."""
        self._extract_function_info(node)

        # Track function context for nested analysis
        old_function_name = self.current_function_name
        self.current_function_name = node.name
        old_in_function = self.in_function_def
        self.in_function_def = True

        self.generic_visit(node)

        # Restore context
        self.current_function_name = old_function_name
        self.in_function_def = old_in_function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Extract information from async function definitions."""
        self._extract_function_info(node, is_async=True)

        # Track function context for nested analysis
        old_function_name = self.current_function_name
        self.current_function_name = node.name
        old_in_function = self.in_function_def
        self.in_function_def = True

        self.generic_visit(node)

        # Restore context
        self.current_function_name = old_function_name
        self.in_function_def = old_in_function

    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract class attributes from assignment statements."""
        if self.current_class_name and not self.in_function_def:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.class_attributes[self.current_class_name].append(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Extract class attributes from annotated assignment statements."""
        if self.current_class_name and not self.in_function_def:
            if isinstance(node.target, ast.Name):
                attr_name = node.target.id
                annotation = self._get_annotation_value(node.annotation)
                self.class_attributes[self.current_class_name].append(f"{attr_name}: {annotation}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Extract function calls."""
        if self.analysis_level == AnalysisLevel.COMPREHENSIVE and self.current_function_name:
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = self._get_attribute_name(node.func)

            if func_name:
                caller = f"{self.current_class_name}.{self.current_function_name}" if self.current_class_name else self.current_function_name
                if caller not in self.function_calls:
                    self.function_calls[caller] = []
                self.function_calls[caller].append(func_name)

        self.generic_visit(node)

    def _extract_function_info(self, node: ast.FunctionDef, is_async: bool = False) -> None:
        """Extract detailed information from a function definition."""
        name = node.name
        prefix = "async " if is_async else ""

        # Track function/method
        if self.current_class_name:
            self.methods.append(name)
            self.class_methods[self.current_class_name].append(name)

            # Track method parameters
            method_key = f"{self.current_class_name}.{name}"
            self.method_params[method_key] = []
            for arg in node.args.args:
                if arg.arg != 'self' and arg.arg != 'cls':
                    param_name = arg.arg
                    if arg.annotation:
                        param_type = self._get_annotation_value(arg.annotation)
                        self.param_types.append(param_type)
                        param_str = f"{param_name}: {param_type}"
                    else:
                        param_str = param_name
                    self.method_params[method_key].append(param_str)
        else:
            self.functions.append(name)

            # Track function parameters
            self.function_params[name] = []
            for arg in node.args.args:
                param_name = arg.arg
                if arg.annotation:
                    param_type = self._get_annotation_value(arg.annotation)
                    self.param_types.append(param_type)
                    param_str = f"{param_name}: {param_type}"
                else:
                    param_str = param_name
                self.function_params[name].append(param_str)

        # Track return type
        if node.returns:
            return_type = self._get_annotation_value(node.returns)
            self.return_types.append(return_type)

        # Track decorators
        if node.decorator_list:
            func_key = f"{self.current_class_name}.{name}" if self.current_class_name else name
            self.decorators[func_key] = [self._get_decorator_name(d) for d in node.decorator_list]

    def _get_annotation_value(self, node: ast.AST) -> str:
        """Extract a string representation of a type annotation."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_attribute_name(node)}"
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_value(node.value)
            slice_value = self._get_annotation_value(node.slice)
            return f"{value}[{slice_value}]"
        elif isinstance(node, ast.Tuple):
            return ", ".join([self._get_annotation_value(el) for el in node.elts])
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.List):
            return f"[{', '.join(self._get_annotation_value(el) for el in node.elts)}]"
        elif isinstance(node, ast.Dict):
            if not node.keys:  # Empty dict
                return "{}"
            key_values = []
            for i in range(len(node.keys)):
                key = self._get_annotation_value(node.keys[i])
                value = self._get_annotation_value(node.values[i])
                key_values.append(f"{key}: {value}")
            return f"{{{', '.join(key_values)}}}"
        else:
            return "<complex_annotation>"

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full dotted name of an attribute."""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        else:
            return f"<...>.{node.attr}"

    def _get_decorator_name(self, node: ast.AST) -> str:
        """Get the name of a decorator."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_attribute_name(node)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                return f"{node.func.id}(...)"
            elif isinstance(node.func, ast.Attribute):
                return f"{self._get_attribute_name(node.func)}(...)"
        return "<complex_decorator>"

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the extracted information.

        Returns:
            A dictionary containing summary information about the analyzed code.
        """
        all_names = self.functions + self.methods
        top_names = [name for name, _ in Counter(all_names).most_common(5)]
        top_param_types = [t for t, _ in Counter(self.param_types).most_common(5)]
        top_return_types = [t for t, _ in Counter(self.return_types).most_common(5)]

        summary = {
            "filepath": self.filepath,
            "classes_count": len(self.classes),
            "functions_count": len(self.functions),
            "methods_count": len(self.methods),
            "total_definitions": len(self.classes) + len(self.functions) + len(self.methods),
            "top_names": top_names,
            "top_param_types": top_param_types,
            "top_return_types": top_return_types
        }

        # Add detailed information based on analysis level
        if self.analysis_level in [AnalysisLevel.DETAILED, AnalysisLevel.COMPREHENSIVE]:
            summary.update({
                "classes": self.classes,
                "functions": self.functions,
                "methods": self.methods,
                "imports": self.imports,
                "class_bases": self.class_bases,
                "class_methods": self.class_methods,
                "class_attributes": self.class_attributes,
                "decorators": self.decorators
            })

            if self.analysis_level == AnalysisLevel.COMPREHENSIVE:
                summary.update({
                    "function_params": self.function_params,
                    "method_params": self.method_params,
                    "function_calls": self.function_calls
                })

        return summary

def extract_summary_from_file(filepath: str, analysis_level: AnalysisLevel = AnalysisLevel.DETAILED) -> Dict[str, Any]:
    """
    Extract a summary of definitions from a Python file.

    Args:
        filepath: Path to the Python file to analyze
        analysis_level: Level of detail for the analysis

    Returns:
        A dictionary containing summary information about the analyzed code
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()

        tree = ast.parse(file_content, filename=filepath)

        extractor = DefinitionExtractor(filepath, analysis_level)
        extractor.visit(tree)
        return extractor.get_summary()
    except SyntaxError as e:
        # Return a default summary for files with syntax errors
        return {
            "filepath": filepath,
            "classes_count": 0,
            "functions_count": 0,
            "methods_count": 0,
            "total_definitions": 0,
            "top_names": ["SyntaxError"],
            "top_param_types": [],
            "top_return_types": [],
            "error": f"SyntaxError: {str(e)}"
        }
    except Exception as e:
        # Catch any other exceptions during parsing or extraction
        return {
            "filepath": filepath,
            "classes_count": 0,
            "functions_count": 0,
            "methods_count": 0,
            "total_definitions": 0,
            "top_names": [f"Error: {type(e).__name__}"],
            "top_param_types": [],
            "top_return_types": [],
            "error": f"{type(e).__name__}: {str(e)}"
        }

def find_python_files(path: str, recursive: bool = True, exclude_patterns: List[str] = None) -> List[str]:
    """
    Find Python files in the given path.

    Args:
        path: Path to search for Python files
        recursive: Whether to search recursively
        exclude_patterns: List of glob patterns to exclude

    Returns:
        List of paths to Python files
    """
    if exclude_patterns is None:
        exclude_patterns = ["*__pycache__*", "*.git*", "*venv*", "*env*", "*.egg-info*"]

    if os.path.isfile(path) and path.endswith(".py"):
        return [path]

    python_files = []

    if os.path.isdir(path):
        pattern = os.path.join(path, "**", "*.py") if recursive else os.path.join(path, "*.py")
        all_files = glob.glob(pattern, recursive=recursive)

        # Apply exclusion patterns
        for exclude in exclude_patterns:
            all_files = [f for f in all_files if not glob.fnmatch.fnmatch(f, exclude)]

        python_files.extend(all_files)

    return sorted(python_files)

def analyze_files(files: List[str], analysis_level: AnalysisLevel = AnalysisLevel.DETAILED) -> List[Dict[str, Any]]:
    """
    Analyze a list of Python files.

    Args:
        files: List of file paths to analyze
        analysis_level: Level of detail for the analysis

    Returns:
        List of summary dictionaries, one for each file
    """
    results = []

    for filepath in files:
        summary = extract_summary_from_file(filepath, analysis_level)
        results.append(summary)

    return results

def filter_results(results: List[Dict[str, Any]],
                  min_definitions: int = 0,
                  has_classes: bool = False,
                  has_functions: bool = False,
                  pattern: str = None) -> List[Dict[str, Any]]:
    """
    Filter analysis results based on criteria.

    Args:
        results: List of summary dictionaries
        min_definitions: Minimum number of total definitions
        has_classes: Whether the file must have classes
        has_functions: Whether the file must have functions
        pattern: Regex pattern to match against filepath

    Returns:
        Filtered list of summary dictionaries
    """
    filtered = results.copy()

    if min_definitions > 0:
        filtered = [r for r in filtered if r.get("total_definitions", 0) >= min_definitions]

    if has_classes:
        filtered = [r for r in filtered if r.get("classes_count", 0) > 0]

    if has_functions:
        filtered = [r for r in filtered if r.get("functions_count", 0) > 0]

    if pattern:
        regex = re.compile(pattern)
        filtered = [r for r in filtered if regex.search(r.get("filepath", ""))]

    return filtered

def format_output(results: List[Dict[str, Any]],
                 output_format: OutputFormat = OutputFormat.TEXT,
                 sort_by: str = "total_definitions",
                 reverse: bool = True) -> str:
    """
    Format analysis results for output.

    Args:
        results: List of summary dictionaries
        output_format: Output format
        sort_by: Field to sort results by
        reverse: Whether to sort in reverse order

    Returns:
        Formatted output string
    """
    # Sort results
    if sort_by in results[0] if results else False:
        results = sorted(results, key=lambda x: x.get(sort_by, 0), reverse=reverse)

    if output_format == OutputFormat.JSON:
        return json.dumps(results, indent=2)

    elif output_format == OutputFormat.CSV:
        if not results:
            return ""

        headers = ["filepath", "classes_count", "functions_count", "methods_count", "total_definitions"]
        rows = []

        for result in results:
            row = [
                result.get("filepath", ""),
                result.get("classes_count", 0),
                result.get("functions_count", 0),
                result.get("methods_count", 0),
                result.get("total_definitions", 0)
            ]
            rows.append(row)

        csv_output = []
        csv_output.append(",".join(headers))
        for row in rows:
            csv_output.append(",".join([str(cell) for cell in row]))

        return "\n".join(csv_output)

    elif output_format == OutputFormat.MARKDOWN:
        if not results:
            return "No results found."

        headers = ["File Path", "Classes", "Functions", "Methods", "Total"]
        rows = []

        for result in results:
            row = [
                result.get("filepath", ""),
                result.get("classes_count", 0),
                result.get("functions_count", 0),
                result.get("methods_count", 0),
                result.get("total_definitions", 0)
            ]
            rows.append(row)

        md_output = []
        md_output.append("| " + " | ".join(headers) + " |")
        md_output.append("| " + " | ".join(["---"] * len(headers)) + " |")

        for row in rows:
            md_output.append("| " + " | ".join([str(cell) for cell in row]) + " |")

        return "\n".join(md_output)

    elif output_format == OutputFormat.TABLE:
        if not HAS_TABULATE:
            return "Table output requires the 'tabulate' package. Please install it with 'pip install tabulate'."

        if not results:
            return "No results found."

        headers = ["File Path", "Classes", "Functions", "Methods", "Total"]
        rows = []

        for result in results:
            row = [
                result.get("filepath", ""),
                result.get("classes_count", 0),
                result.get("functions_count", 0),
                result.get("methods_count", 0),
                result.get("total_definitions", 0)
            ]
            rows.append(row)

        return tabulate.tabulate(rows, headers=headers, tablefmt="grid")

    else:  # OutputFormat.TEXT
        if not results:
            return "No results found."

        text_output = []

        for result in results:
            filepath = result.get("filepath", "Unknown")
            classes = result.get("classes_count", 0)
            functions = result.get("functions_count", 0)
            methods = result.get("methods_count", 0)
            total = result.get("total_definitions", 0)

            if HAS_COLORAMA:
                filepath = f"{Fore.CYAN}{filepath}{Fore.RESET}"
                classes_str = f"{Fore.GREEN}{classes}{Fore.RESET}"
                functions_str = f"{Fore.YELLOW}{functions}{Fore.RESET}"
                methods_str = f"{Fore.MAGENTA}{methods}{Fore.RESET}"
                total_str = f"{Style.BRIGHT}{total}{Style.RESET_ALL}"
            else:
                classes_str = str(classes)
                functions_str = str(functions)
                methods_str = str(methods)
                total_str = str(total)

            text_output.append(f"{filepath}")
            text_output.append(f"  Classes: {classes_str}, Functions: {functions_str}, Methods: {methods_str}, Total: {total_str}")

            # Add top names if available
            top_names = result.get("top_names", [])
            if top_names:
                text_output.append(f"  Top Names: {', '.join(top_names)}")

            # Add error if present
            if "error" in result:
                if HAS_COLORAMA:
                    error = f"{Fore.RED}{result['error']}{Fore.RESET}"
                else:
                    error = result["error"]
                text_output.append(f"  Error: {error}")

            text_output.append("")

        return "\n".join(text_output)

def generate_detailed_report(result: Dict[str, Any]) -> str:
    """
    Generate a detailed report for a single file.

    Args:
        result: Summary dictionary for a file

    Returns:
        Detailed report string
    """
    if not result:
        return "No data available."

    filepath = result.get("filepath", "Unknown")

    report = []
    report.append(f"{'=' * 80}")

    if HAS_COLORAMA:
        report.append(f"{Style.BRIGHT}{Fore.CYAN}DETAILED REPORT: {filepath}{Style.RESET_ALL}")
    else:
        report.append(f"DETAILED REPORT: {filepath}")

    report.append(f"{'=' * 80}")
    report.append("")

    # Basic statistics
    report.append("SUMMARY:")
    report.append(f"  Classes: {result.get('classes_count', 0)}")
    report.append(f"  Functions: {result.get('functions_count', 0)}")
    report.append(f"  Methods: {result.get('methods_count', 0)}")
    report.append(f"  Total Definitions: {result.get('total_definitions', 0)}")
    report.append("")

    # Error information if present
    if "error" in result:
        if HAS_COLORAMA:
            report.append(f"{Fore.RED}ERROR: {result['error']}{Fore.RESET}")
        else:
            report.append(f"ERROR: {result['error']}")
        report.append("")
        return "\n".join(report)  # Return early if there's an error

    # Classes and inheritance
    if result.get("classes", []):
        report.append("CLASSES:")
        for cls in result.get("classes", []):
            bases = result.get("class_bases", {}).get(cls, [])
            bases_str = f" ({', '.join(bases)})" if bases else ""

            if HAS_COLORAMA:
                report.append(f"  {Fore.GREEN}{cls}{Fore.RESET}{bases_str}")
            else:
                report.append(f"  {cls}{bases_str}")

            # Class methods
            methods = result.get("class_methods", {}).get(cls, [])
            if methods:
                for method in methods:
                    method_key = f"{cls}.{method}"
                    decorators = result.get("decorators", {}).get(method_key, [])
                    decorators_str = " ".join([f"@{d}" for d in decorators])

                    params = result.get("method_params", {}).get(method_key, [])
                    params_str = ", ".join(params)

                    if decorators:
                        if HAS_COLORAMA:
                            report.append(f"    {Fore.BLUE}{decorators_str}{Fore.RESET}")
                        else:
                            report.append(f"    {decorators_str}")

                    if HAS_COLORAMA:
                        report.append(f"    {Fore.YELLOW}def {method}({params_str}){Fore.RESET}")
                    else:
                        report.append(f"    def {method}({params_str})")

            # Class attributes
            attributes = result.get("class_attributes", {}).get(cls, [])
            if attributes:
                report.append("    Attributes:")
                for attr in attributes:
                    report.append(f"      {attr}")

            report.append("")

    # Functions
    if result.get("functions", []):
        report.append("FUNCTIONS:")
        for func in result.get("functions", []):
            decorators = result.get("decorators", {}).get(func, [])
            decorators_str = " ".join([f"@{d}" for d in decorators])

            params = result.get("function_params", {}).get(func, [])
            params_str = ", ".join(params)

            if decorators:
                if HAS_COLORAMA:
                    report.append(f"  {Fore.BLUE}{decorators_str}{Fore.RESET}")
                else:
                    report.append(f"  {decorators_str}")

            if HAS_COLORAMA:
                report.append(f"  {Fore.YELLOW}def {func}({params_str}){Fore.RESET}")
            else:
                report.append(f"  def {func}({params_str})")

            # Function calls (if available)
            calls = result.get("function_calls", {}).get(func, [])
            if calls:
                unique_calls = sorted(set(calls))
                report.append(f"    Calls: {', '.join(unique_calls)}")

            report.append("")

    # Imports
    imports = result.get("imports", {})
    if imports:
        report.append("IMPORTS:")
        for module, names in imports.items():
            if HAS_COLORAMA:
                report.append(f"  {Fore.MAGENTA}from {module} import {', '.join(names)}{Fore.RESET}")
            else:
                report.append(f"  from {module} import {', '.join(names)}")
        report.append("")

    return "\n".join(report)

def extract_paths_from_file(file_path: str) -> List[str]:
    """
    Extract file paths from a text file (like a plan or documentation).

    Args:
        file_path: Path to the file to extract paths from

    Returns:
        List of extracted file paths
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist.")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract paths that look like Python files
        # This regex looks for patterns like:
        # - openhcs/path/to/file.py
        # - /path/to/file.py
        # - ./path/to/file.py
        # - path/to/file.py
        # - "path/to/file.py"
        # - 'path/to/file.py'
        path_patterns = [
            r'(?:^|\s|[\'"`(])(?:\.{0,2}/|/)?(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.py(?:[\'"`)\s]|$)',
            r'(?:^|\s)(?:openhcs|semantic_matrix_analyzer)/(?:[a-zA-Z0-9_-]+/)*[a-zA-Z0-9_-]+\.py(?:\s|$)'
        ]

        all_paths = []
        for pattern in path_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                # Clean up the path
                path = match.strip()
                path = re.sub(r'^[\'"`(]|[\'"`)\s]$', '', path)

                # Skip paths that are likely not actual file paths
                if 'import ' in path or 'from ' in path:
                    continue

                all_paths.append(path)

        # Remove duplicates and sort
        unique_paths = sorted(set(all_paths))

        # Resolve relative paths
        resolved_paths = []
        for path in unique_paths:
            # If path starts with ./ or ../, resolve it relative to the plan file
            if path.startswith('./') or path.startswith('../'):
                base_dir = os.path.dirname(os.path.abspath(file_path))
                resolved_path = os.path.normpath(os.path.join(base_dir, path))
                resolved_paths.append(resolved_path)
            else:
                # For absolute paths or paths without ./ or ../, use as is
                resolved_paths.append(path)

        return resolved_paths

    except Exception as e:
        print(f"Error extracting paths from {file_path}: {str(e)}")
        return []

def analyze_plan(plan_file: str, output_dir: str = None, analysis_level: AnalysisLevel = AnalysisLevel.COMPREHENSIVE) -> None:
    """
    Analyze a plan file and extract architectural insights from all files mentioned in it.

    Args:
        plan_file: Path to the plan file
        output_dir: Directory to save analysis results (if None, print to stdout)
        analysis_level: Level of detail for the analysis
    """
    print(f"Analyzing plan: {plan_file}")

    # Extract file paths from the plan
    file_paths = extract_paths_from_file(plan_file)

    if not file_paths:
        print("No file paths found in the plan.")
        return

    print(f"Found {len(file_paths)} files to analyze:")
    for path in file_paths:
        print(f"  - {path}")

    # Filter to only existing files
    existing_files = [path for path in file_paths if os.path.exists(path)]
    if len(existing_files) < len(file_paths):
        print(f"Warning: {len(file_paths) - len(existing_files)} files not found.")

    if not existing_files:
        print("No existing files to analyze.")
        return

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Results will be saved to: {output_dir}")

    # Analyze each file
    results = []
    for file_path in existing_files:
        print(f"Analyzing {file_path}...")

        summary = extract_summary_from_file(file_path, analysis_level)
        results.append(summary)

        if output_dir:
            # Save detailed report for each file
            base_name = os.path.basename(file_path)
            output_file = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_analysis.txt")

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(generate_detailed_report(summary))

            print(f"  Detailed report saved to {output_file}")
        else:
            # Print detailed report to stdout
            print(generate_detailed_report(summary))

    # Generate summary report
    if output_dir:
        # Save summary report
        summary_file = os.path.join(output_dir, "summary_report.md")
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(format_output(results, OutputFormat.MARKDOWN))

        print(f"Summary report saved to {summary_file}")

    # Analyze cross-cutting concerns if we have multiple files
    if len(existing_files) > 1 and output_dir:
        print("Analyzing cross-cutting concerns...")

        # Find common directories
        common_dirs = set()
        for file_path in existing_files:
            dir_path = os.path.dirname(file_path)
            if dir_path:
                common_dirs.add(dir_path)

        for dir_path in common_dirs:
            # Analyze async methods
            async_file = os.path.join(output_dir, f"{os.path.basename(dir_path)}_async_methods.txt")
            async_results = analyze_files(
                find_python_files(dir_path, recursive=True, exclude_patterns=["*test*"]),
                analysis_level
            )
            async_filtered = filter_results(async_results, pattern="async")

            with open(async_file, 'w', encoding='utf-8') as f:
                f.write(format_output(async_filtered, OutputFormat.TEXT))

            # Analyze notify calls
            notify_file = os.path.join(output_dir, f"{os.path.basename(dir_path)}_notify_calls.txt")
            notify_results = analyze_files(
                find_python_files(dir_path, recursive=True, exclude_patterns=["*test*"]),
                analysis_level
            )
            notify_filtered = filter_results(notify_results, pattern="notify")

            with open(notify_file, 'w', encoding='utf-8') as f:
                f.write(format_output(notify_filtered, OutputFormat.TEXT))

        print(f"Cross-cutting concerns analysis saved to {output_dir}")

    print("Analysis complete!")

def print_help() -> None:
    """Print detailed help information."""
    help_text = """
CodeArchitect - A tool for extracting architectural insights from Python code.

USAGE:
    python extract_definitions.py [OPTIONS] PATH [PATH...]

DESCRIPTION:
    This tool analyzes Python files to extract information about their structure,
    including classes, functions, methods, and type annotations. It's designed to
    help agents and humans understand code architecture without reading all files in full.

ARGUMENTS:
    PATH                    Path to a Python file or directory to analyze

OPTIONS:
    -h, --help              Show this help message and exit
    -o, --output FORMAT     Output format: json, table, text, markdown, csv (default: text)
    -l, --level LEVEL       Analysis level: basic, detailed, comprehensive (default: detailed)
    -r, --recursive         Recursively search directories for Python files
    -s, --sort FIELD        Sort results by field (default: total_definitions)
    --no-reverse            Sort in ascending order (default is descending)
    -m, --min-definitions N Minimum number of total definitions to include in results
    -c, --has-classes       Only include files with classes
    -f, --has-functions     Only include files with functions
    -p, --pattern PATTERN   Only include files matching regex pattern
    -d, --detailed-report   Generate a detailed report for each file
    --exclude PATTERN       Exclude files matching glob pattern (can be used multiple times)
    --plan PLAN_FILE        Analyze a plan file and extract insights from all files mentioned in it
    --output-dir DIR        Directory to save analysis results when using --plan

EXAMPLES:
    # Analyze a single file
    python extract_definitions.py myfile.py

    # Analyze all Python files in a directory
    python extract_definitions.py myproject/

    # Analyze files with at least 5 definitions and output as JSON
    python extract_definitions.py -m 5 -o json myproject/

    # Find files with classes that match a pattern
    python extract_definitions.py -c -p "model|view|controller" myproject/

    # Generate a detailed report for a file
    python extract_definitions.py -d myfile.py

    # Exclude test files and output as a table
    python extract_definitions.py -o table --exclude "*test*" myproject/

    # Analyze all files mentioned in a plan
    python extract_definitions.py --plan plans/my_plan.md --output-dir analysis_results
    """
    print(help_text)

def main() -> None:
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="CodeArchitect - Extract architectural insights from Python code",
        add_help=False  # We'll handle help manually for more detailed output
    )

    parser.add_argument(
        "paths",
        nargs="*",
        help="Path to a Python file or directory to analyze"
    )

    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show this help message and exit"
    )

    parser.add_argument(
        "-o", "--output",
        choices=[f.value for f in OutputFormat],
        default="text",
        help="Output format: json, table, text, markdown, csv (default: text)"
    )

    parser.add_argument(
        "-l", "--level",
        choices=[l.value for l in AnalysisLevel],
        default="detailed",
        help="Analysis level: basic, detailed, comprehensive (default: detailed)"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursively search directories for Python files"
    )

    parser.add_argument(
        "-s", "--sort",
        default="total_definitions",
        help="Sort results by field (default: total_definitions)"
    )

    parser.add_argument(
        "--no-reverse",
        action="store_true",
        help="Sort in ascending order (default is descending)"
    )

    parser.add_argument(
        "-m", "--min-definitions",
        type=int,
        default=0,
        help="Minimum number of total definitions to include in results"
    )

    parser.add_argument(
        "-c", "--has-classes",
        action="store_true",
        help="Only include files with classes"
    )

    parser.add_argument(
        "-f", "--has-functions",
        action="store_true",
        help="Only include files with functions"
    )

    parser.add_argument(
        "-p", "--pattern",
        help="Only include files matching regex pattern"
    )

    parser.add_argument(
        "-d", "--detailed-report",
        action="store_true",
        help="Generate a detailed report for each file"
    )

    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching glob pattern (can be used multiple times)"
    )

    # New arguments for plan analysis
    parser.add_argument(
        "--plan",
        help="Analyze a plan file and extract insights from all files mentioned in it"
    )

    parser.add_argument(
        "--output-dir",
        help="Directory to save analysis results when using --plan"
    )

    args = parser.parse_args()

    # Show help if requested or if no paths provided and no plan specified
    if args.help or (not args.paths and not args.plan):
        print_help()
        return

    # Convert string arguments to enums
    output_format = OutputFormat(args.output)
    analysis_level = AnalysisLevel(args.level)

    # Plan analysis mode
    if args.plan:
        analyze_plan(args.plan, args.output_dir, analysis_level)
        return

    # Standard file/directory analysis mode
    # Find Python files
    all_files = []
    for path in args.paths:
        files = find_python_files(path, args.recursive, args.exclude)
        all_files.extend(files)

    if not all_files:
        print("No Python files found.")
        return

    # Analyze files
    results = analyze_files(all_files, analysis_level)

    # Filter results
    filtered_results = filter_results(
        results,
        min_definitions=args.min_definitions,
        has_classes=args.has_classes,
        has_functions=args.has_functions,
        pattern=args.pattern
    )

    # Generate detailed reports if requested
    if args.detailed_report:
        for result in filtered_results:
            report = generate_detailed_report(result)
            print(report)
    else:
        # Format and print output
        output = format_output(
            filtered_results,
            output_format=output_format,
            sort_by=args.sort,
            reverse=not args.no_reverse
        )
        print(output)

if __name__ == "__main__":
    main()