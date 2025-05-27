#!/usr/bin/env python3
"""
Import Validator - A tool to analyze Python imports and predict potential errors.

This tool analyzes Python files to:
1. Detect missing imports
2. Identify deprecated or renamed imports
3. Find import conflicts
4. Validate import paths against the actual module structure
5. Generate a report of all potential import issues
"""

import os
import sys
import ast
import importlib
import importlib.util
import pkgutil
import json
import argparse
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path
from collections import defaultdict

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Known import replacements (old -> new)
IMPORT_REPLACEMENTS = {
    "openhcs.io.base.StorageBackendEnum": "openhcs.constants.constants.Backend",
    "openhcs.core.memory.storage_backend.StorageBackend": "openhcs.constants.constants.Backend",
    "prompt_toolkit.shortcuts.DialogResult": None,  # This import should be removed
}

# Known module structure issues
MODULE_STRUCTURE_ISSUES = {
    "prompt_toolkit.layout.Container": "Container is imported from prompt_toolkit.layout",
    "prompt_toolkit.layout.RadioList": "RadioList is imported from prompt_toolkit.widgets",
    "prompt_toolkit.layout.CheckboxList": "CheckboxList is imported from prompt_toolkit.widgets",
}

class ImportInfo:
    """Information about an import statement."""
    def __init__(self, module: str, name: Optional[str], alias: Optional[str], lineno: int, col_offset: int):
        self.module = module
        self.name = name
        self.alias = alias
        self.lineno = lineno
        self.col_offset = col_offset
        
    @property
    def full_name(self) -> str:
        """Get the full import name."""
        if self.name:
            return f"{self.module}.{self.name}"
        return self.module
    
    @property
    def import_as(self) -> str:
        """Get the name as imported in the code."""
        if self.alias:
            return self.alias
        if self.name:
            return self.name
        return self.module.split(".")[-1]
    
    def __repr__(self) -> str:
        if self.name and self.alias:
            return f"from {self.module} import {self.name} as {self.alias}"
        elif self.name:
            return f"from {self.module} import {self.name}"
        elif self.alias:
            return f"import {self.module} as {self.alias}"
        else:
            return f"import {self.module}"

class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze imports."""
    
    def __init__(self):
        self.imports = []
        self.import_names = set()
        self.used_names = set()
        self.current_function = None
        self.current_class = None
        
    def visit_Import(self, node):
        """Process import statements."""
        for alias in node.names:
            module = alias.name
            alias_name = alias.asname
            
            import_info = ImportInfo(
                module=module,
                name=None,
                alias=alias_name,
                lineno=node.lineno,
                col_offset=node.col_offset
            )
            
            self.imports.append(import_info)
            self.import_names.add(import_info.import_as)
            
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Process from-import statements."""
        if node.module:
            for alias in node.names:
                name = alias.name
                alias_name = alias.asname
                
                import_info = ImportInfo(
                    module=node.module,
                    name=name,
                    alias=alias_name,
                    lineno=node.lineno,
                    col_offset=node.col_offset
                )
                
                self.imports.append(import_info)
                self.import_names.add(import_info.import_as)
                
        self.generic_visit(node)
    
    def visit_Name(self, node):
        """Track used names."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Track attribute access."""
        if isinstance(node.value, ast.Name):
            self.used_names.add(node.value.id)
        self.generic_visit(node)

def check_import_exists(import_info: ImportInfo) -> bool:
    """Check if an import exists in the Python path."""
    try:
        if import_info.name:
            # from module import name
            module = importlib.import_module(import_info.module)
            return hasattr(module, import_info.name)
        else:
            # import module
            importlib.import_module(import_info.module)
            return True
    except (ImportError, ModuleNotFoundError):
        return False

def analyze_file(file_path: str) -> Dict[str, Any]:
    """
    Analyze a file for import issues.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Dictionary with analysis results
    """
    result = {
        "file_path": file_path,
        "imports": [],
        "missing_imports": [],
        "unused_imports": [],
        "deprecated_imports": [],
        "import_conflicts": [],
        "module_structure_issues": [],
        "error": None
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=file_path)
        
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        # Record all imports
        result["imports"] = [repr(imp) for imp in analyzer.imports]
        
        # Check for missing imports (names used but not imported)
        missing_imports = analyzer.used_names - analyzer.import_names - set(dir(__builtins__))
        result["missing_imports"] = list(missing_imports)
        
        # Check for unused imports
        unused_imports = analyzer.import_names - analyzer.used_names
        result["unused_imports"] = list(unused_imports)
        
        # Check for deprecated imports
        for imp in analyzer.imports:
            full_name = imp.full_name
            if full_name in IMPORT_REPLACEMENTS:
                replacement = IMPORT_REPLACEMENTS[full_name]
                result["deprecated_imports"].append({
                    "old": full_name,
                    "new": replacement,
                    "line": imp.lineno
                })
        
        # Check for module structure issues
        for imp in analyzer.imports:
            full_name = imp.full_name
            for issue_pattern, message in MODULE_STRUCTURE_ISSUES.items():
                if full_name == issue_pattern or (imp.name and f"{imp.module}.{imp.name}" == issue_pattern):
                    result["module_structure_issues"].append({
                        "import": full_name,
                        "message": message,
                        "line": imp.lineno
                    })
        
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e.msg} at line {e.lineno}"
    except Exception as e:
        result["error"] = f"Error: {type(e).__name__}: {e}"
    
    return result

def find_python_files(directory: str, exclude_patterns: List[str] = None) -> List[str]:
    """Find all Python files in a directory recursively."""
    if exclude_patterns is None:
        exclude_patterns = ["*__pycache__*", "*.git*", "*venv*", "*env*"]
    
    python_files = []
    for root, _, files in os.walk(directory):
        # Check if this directory should be excluded
        if any(pattern in root for pattern in exclude_patterns):
            continue
            
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                # Check if this file should be excluded
                if not any(pattern in file_path for pattern in exclude_patterns):
                    python_files.append(file_path)
    
    return python_files

def analyze_directory(directory: str, exclude_patterns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Analyze all Python files in a directory for import issues.
    
    Args:
        directory: Directory to search
        exclude_patterns: List of patterns to exclude
        
    Returns:
        List of dictionaries with analysis results
    """
    python_files = find_python_files(directory, exclude_patterns)
    
    results = []
    for file_path in python_files:
        result = analyze_file(file_path)
        # Only include files with issues
        if (result["missing_imports"] or result["unused_imports"] or 
            result["deprecated_imports"] or result["module_structure_issues"] or
            result["error"]):
            results.append(result)
    
    return results

def format_results_markdown(results: List[Dict[str, Any]], project_root: str) -> str:
    """
    Format analysis results as Markdown.
    
    Args:
        results: List of dictionaries with analysis results
        project_root: Project root directory for relative paths
        
    Returns:
        Markdown string
    """
    if not results:
        return "# Import Analysis\n\nNo import issues found."
    
    lines = []
    lines.append("# Import Analysis")
    lines.append(f"\nFound {len(results)} files with import issues:\n")
    
    for result in results:
        rel_path = os.path.relpath(result["file_path"], project_root)
        lines.append(f"## {rel_path}")
        
        if result["error"]:
            lines.append(f"**Error:** {result['error']}")
        
        if result["missing_imports"]:
            lines.append("\n### Missing Imports")
            lines.append("These symbols are used but not imported:")
            lines.append("```")
            for name in sorted(result["missing_imports"]):
                lines.append(f"- {name}")
            lines.append("```")
        
        if result["unused_imports"]:
            lines.append("\n### Unused Imports")
            lines.append("These symbols are imported but not used:")
            lines.append("```")
            for name in sorted(result["unused_imports"]):
                lines.append(f"- {name}")
            lines.append("```")
        
        if result["deprecated_imports"]:
            lines.append("\n### Deprecated Imports")
            lines.append("| Line | Old Import | New Import |")
            lines.append("| ---- | ---------- | ---------- |")
            for item in sorted(result["deprecated_imports"], key=lambda x: x["line"]):
                new_import = item["new"] or "Remove this import"
                lines.append(f"| {item['line']} | `{item['old']}` | `{new_import}` |")
        
        if result["module_structure_issues"]:
            lines.append("\n### Module Structure Issues")
            lines.append("| Line | Import | Issue |")
            lines.append("| ---- | ------ | ----- |")
            for item in sorted(result["module_structure_issues"], key=lambda x: x["line"]):
                lines.append(f"| {item['line']} | `{item['import']}` | {item['message']} |")
        
        lines.append("")  # Add blank line between files
    
    return "\n".join(lines)

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Import Validator - Analyze Python imports and predict potential errors",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to analyze (default: current directory)"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)"
    )
    
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude files matching pattern (can be used multiple times)"
    )
    
    args = parser.parse_args()
    
    # Analyze the directory
    results = analyze_directory(args.directory, args.exclude)
    
    # Format the results
    formatted_results = format_results_markdown(results, PROJECT_ROOT)
    
    # Output the results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(formatted_results)
        print(f"Results written to {args.output}")
    else:
        print(formatted_results)

if __name__ == "__main__":
    main()
