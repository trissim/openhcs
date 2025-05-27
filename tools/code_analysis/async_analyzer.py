#!/usr/bin/env python3
"""
Async Pattern Analyzer

This module analyzes async/await patterns in Python code, identifying:
- Async functions without proper return type annotations
- Coroutines that are not properly awaited
- Improper use of async/await
"""

import ast
import argparse
import logging
import os
import sys
from typing import List, Dict, Set, Optional, Tuple, Any, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
REPORTS_DIR = os.path.join("reports", "code_analysis")

class AsyncPatternVisitor(ast.NodeVisitor):
    """
    AST visitor for analyzing async/await patterns.
    
    This visitor identifies:
    - Async functions without proper return type annotations
    - Coroutines that are not properly awaited
    - Improper use of async/await
    """
    
    def __init__(self):
        self.async_functions = []
        self.async_functions_without_return_type = []
        self.unawaited_coroutines = []
        self.awaited_coroutines = []
        self.current_function = None
        self.current_class = None
        self.coroutine_functions = set()  # Track functions known to return coroutines
        
    def visit_AsyncFunctionDef(self, node):
        """Visit async function definitions."""
        old_function = self.current_function
        self.current_function = node
        
        # Add to set of coroutine functions
        func_name = node.name
        if self.current_class:
            func_name = f"{self.current_class.name}.{func_name}"
        self.coroutine_functions.add(func_name)
        
        # Check if the function has a return type annotation
        has_return_type = False
        if node.returns:
            has_return_type = True
            # TODO: Check if return type is Awaitable, Coroutine, etc.
        
        # Add to the list of async functions
        self.async_functions.append({
            'name': node.name,
            'lineno': node.lineno,
            'class': self.current_class.name if self.current_class else None,
            'has_return_type': has_return_type
        })
        
        # Add to the list of async functions without return type annotations
        if not has_return_type:
            self.async_functions_without_return_type.append({
                'name': node.name,
                'lineno': node.lineno,
                'class': self.current_class.name if self.current_class else None
            })
        
        # Visit the function body
        self.generic_visit(node)
        
        self.current_function = old_function
        
    def visit_ClassDef(self, node):
        """Visit class definitions."""
        old_class = self.current_class
        self.current_class = node
        
        # Visit the class body
        self.generic_visit(node)
        
        self.current_class = old_class
        
    def visit_Call(self, node):
        """Visit function calls."""
        # Check if the call is to a coroutine
        is_coroutine = False
        func_name = None
        
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            # Check if this is a known coroutine function
            if func_name in self.coroutine_functions:
                is_coroutine = True
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            # Check if this is a known coroutine method
            if hasattr(node.func.value, 'id'):
                full_name = f"{node.func.value.id}.{func_name}"
                if full_name in self.coroutine_functions:
                    is_coroutine = True
            
            # Common coroutine methods in prompt_toolkit
            if func_name in ['run_async', 'show', 'execute']:
                is_coroutine = True
                
        # Check if the call is awaited
        is_awaited = False
        parent = getattr(node, 'parent', None)
        if parent and isinstance(parent, ast.Await):
            is_awaited = True
            
        # Add to the appropriate list
        if is_coroutine:
            if is_awaited:
                self.awaited_coroutines.append({
                    'name': func_name,
                    'lineno': node.lineno,
                    'function': self.current_function.name if self.current_function else None,
                    'class': self.current_class.name if self.current_class else None
                })
            else:
                self.unawaited_coroutines.append({
                    'name': func_name,
                    'lineno': node.lineno,
                    'function': self.current_function.name if self.current_function else None,
                    'class': self.current_class.name if self.current_class else None
                })
                
        # Visit the call arguments
        self.generic_visit(node)
        
    def visit_Await(self, node):
        """Visit await expressions."""
        # Set the parent of the value node
        node.value.parent = node
        
        # Visit the value
        self.visit(node.value)

def analyze_async_patterns(filepath: str) -> Dict[str, Any]:
    """
    Analyze async/await patterns in a Python file.
    
    Args:
        filepath: Path to the Python file to analyze
        
    Returns:
        Dictionary with analysis results
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()
        
    try:
        tree = ast.parse(source)
        visitor = AsyncPatternVisitor()
        visitor.visit(tree)
        
        return {
            'async_functions': visitor.async_functions,
            'async_functions_without_return_type': visitor.async_functions_without_return_type,
            'unawaited_coroutines': visitor.unawaited_coroutines,
            'awaited_coroutines': visitor.awaited_coroutines
        }
    except SyntaxError as e:
        logger.error(f"Syntax error in {filepath}: {e}")
        return {
            'error': str(e),
            'async_functions': [],
            'async_functions_without_return_type': [],
            'unawaited_coroutines': [],
            'awaited_coroutines': []
        }

def format_async_analysis_as_markdown(filepath: str, results: Dict[str, Any]) -> str:
    """
    Format async analysis results as Markdown.
    
    Args:
        filepath: Path to the analyzed file
        results: Analysis results
        
    Returns:
        Markdown string
    """
    markdown = f"# Async Pattern Analysis for {os.path.basename(filepath)}\n\n"
    
    # Add error if present
    if 'error' in results:
        markdown += f"## Error\n\n{results['error']}\n\n"
        
    # Add async functions
    markdown += "## Async Functions\n\n"
    if results['async_functions']:
        markdown += "| Name | Line | Class | Has Return Type |\n"
        markdown += "|------|------|-------|----------------|\n"
        for func in results['async_functions']:
            markdown += f"| {func['name']} | {func['lineno']} | {func['class'] or 'None'} | {func['has_return_type']} |\n"
    else:
        markdown += "No async functions found.\n"
        
    # Add async functions without return type annotations
    markdown += "\n## Async Functions Without Return Type Annotations\n\n"
    if results['async_functions_without_return_type']:
        markdown += "| Name | Line | Class |\n"
        markdown += "|------|------|-------|\n"
        for func in results['async_functions_without_return_type']:
            markdown += f"| {func['name']} | {func['lineno']} | {func['class'] or 'None'} |\n"
    else:
        markdown += "No async functions without return type annotations found.\n"
        
    # Add unawaited coroutines
    markdown += "\n## Unawaited Coroutines\n\n"
    if results['unawaited_coroutines']:
        markdown += "| Name | Line | Function | Class |\n"
        markdown += "|------|------|----------|-------|\n"
        for coro in results['unawaited_coroutines']:
            markdown += f"| {coro['name']} | {coro['lineno']} | {coro['function'] or 'None'} | {coro['class'] or 'None'} |\n"
    else:
        markdown += "No unawaited coroutines found.\n"
        
    # Add awaited coroutines
    markdown += "\n## Awaited Coroutines\n\n"
    if results['awaited_coroutines']:
        markdown += "| Name | Line | Function | Class |\n"
        markdown += "|------|------|----------|-------|\n"
        for coro in results['awaited_coroutines']:
            markdown += f"| {coro['name']} | {coro['lineno']} | {coro['function'] or 'None'} | {coro['class'] or 'None'} |\n"
    else:
        markdown += "No awaited coroutines found.\n"
        
    return markdown

def handle_async_patterns(args):
    """
    Handle the async-patterns command.
    
    Args:
        args: Command-line arguments
    """
    print(f"Analyzing async patterns in {args.path}...")
    
    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"Error: Path not found at {path}", file=sys.stderr)
        return
        
    # Determine output directory
    output_dir = args.output_dir if hasattr(args, 'output_dir') and args.output_dir else REPORTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isfile(path):
        # Analyze a single file
        results = analyze_async_patterns(path)
        markdown = format_async_analysis_as_markdown(path, results)
        
        # Determine output filename
        if args.output:
            output_filename = args.output
        else:
            file_basename = os.path.basename(path)
            file_name_without_ext = os.path.splitext(file_basename)[0]
            output_filename = os.path.join(output_dir, f"async_patterns_{file_name_without_ext}.md")
            
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Async pattern analysis generated: {output_filename}")
    else:
        # Analyze a directory
        python_files = []
        for root, _, files in os.walk(path):
            for file_name in files:
                if file_name.endswith(".py"):
                    python_files.append(os.path.join(root, file_name))
                    
        # Determine output filename
        if args.output:
            output_filename = args.output
        else:
            dir_name = os.path.basename(path.rstrip(os.sep)) or "root"
            output_filename = os.path.join(output_dir, f"async_patterns_{dir_name}.md")
            
        # Analyze each file and combine results
        all_results = {}
        for filepath in python_files:
            results = analyze_async_patterns(filepath)
            all_results[filepath] = results
            
        # Generate combined markdown
        markdown = f"# Async Pattern Analysis for {path}\n\n"
        
        # Add summary
        markdown += "## Summary\n\n"
        markdown += "| File | Async Functions | Without Return Type | Unawaited Coroutines | Awaited Coroutines |\n"
        markdown += "|------|----------------|---------------------|---------------------|-------------------|\n"
        for filepath, results in all_results.items():
            rel_path = os.path.relpath(filepath, path)
            markdown += f"| {rel_path} | {len(results['async_functions'])} | {len(results['async_functions_without_return_type'])} | {len(results['unawaited_coroutines'])} | {len(results['awaited_coroutines'])} |\n"
            
        # Add details for each file
        for filepath, results in all_results.items():
            rel_path = os.path.relpath(filepath, path)
            markdown += f"\n## {rel_path}\n\n"
            
            # Add async functions without return type annotations
            markdown += "### Async Functions Without Return Type Annotations\n\n"
            if results['async_functions_without_return_type']:
                markdown += "| Name | Line | Class |\n"
                markdown += "|------|------|-------|\n"
                for func in results['async_functions_without_return_type']:
                    markdown += f"| {func['name']} | {func['lineno']} | {func['class'] or 'None'} |\n"
            else:
                markdown += "No async functions without return type annotations found.\n"
                
            # Add unawaited coroutines
            markdown += "\n### Unawaited Coroutines\n\n"
            if results['unawaited_coroutines']:
                markdown += "| Name | Line | Function | Class |\n"
                markdown += "|------|------|----------|-------|\n"
                for coro in results['unawaited_coroutines']:
                    markdown += f"| {coro['name']} | {coro['lineno']} | {coro['function'] or 'None'} | {coro['class'] or 'None'} |\n"
            else:
                markdown += "No unawaited coroutines found.\n"
                
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"Async pattern analysis generated: {output_filename}")

def main():
    """Main entry point for the async analyzer."""
    parser = argparse.ArgumentParser(description="Async Pattern Analyzer")
    parser.add_argument("path", help="Path to the file or directory to analyze.")
    parser.add_argument("-o", "--output", help="Output Markdown file path. Defaults to reports/code_analysis/async_patterns_<name>.md")
    parser.add_argument("--output-dir", help="Output directory for the report. Defaults to reports/code_analysis/")
    
    args = parser.parse_args()
    
    # Handle the command
    handle_async_patterns(args)
    
if __name__ == "__main__":
    main()
