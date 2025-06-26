#!/usr/bin/env python
"""
Command-line tool for running AST-based validation on openhcs codebase.

This tool validates Python files for compliance with openhcs's architectural
principles using static AST-based analysis.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

from openhcs.validation.ast_validator import ValidationViolation, validate_file


def find_python_files(directory: Path, exclude_dirs: Optional[Set[str]] = None) -> List[Path]:
    """
    Find all Python files in a directory recursively using breadth-first traversal.

    Args:
        directory: Directory to search.
        exclude_dirs: Set of directory names to exclude.

    Returns:
        List of Python file paths sorted by depth (shallower first).
    """
    from collections import deque

    if exclude_dirs is None:
        exclude_dirs = set()

    python_files = []
    # Use deque for breadth-first traversal
    dirs_to_search = deque([(directory, 0)])  # (path, depth)

    while dirs_to_search:
        current_dir, depth = dirs_to_search.popleft()

        try:
            for entry in current_dir.iterdir():
                if entry.is_file() and entry.suffix == '.py':
                    python_files.append((entry, depth))
                elif entry.is_dir() and entry.name not in exclude_dirs:
                    # Add subdirectory to queue for later processing
                    dirs_to_search.append((entry, depth + 1))
        except (PermissionError, OSError):
            # Skip directories we can't read
            continue

    # Sort by depth first, then by path for consistent ordering
    python_files.sort(key=lambda x: (x[1], str(x[0])))

    # Return just the paths
    return [file_path for file_path, _ in python_files]


def validate_directory(directory: Path, exclude_dirs: Optional[Set[str]] = None) -> List[ValidationViolation]:
    """
    Validate all Python files in a directory.
    
    Args:
        directory: Directory to validate.
        exclude_dirs: Set of directory names to exclude.
    
    Returns:
        List of validation violations.
    """
    python_files = find_python_files(directory, exclude_dirs)
    violations = []
    
    for file_path in python_files:
        file_violations = validate_file(str(file_path))
        violations.extend(file_violations)
    
    return violations


def main():
    """Main entry point for the validation tool."""
    parser = argparse.ArgumentParser(
        description='AST-based validation for openhcs codebase'
    )
    parser.add_argument(
        'paths',
        nargs='+',
        type=str,
        help='Files or directories to validate'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=['__pycache__', '.git', 'venv', 'env', '.venv', '.env'],
        help='Directories to exclude from validation'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for validation results (default: stdout)'
    )
    parser.add_argument(
        '--fail-on-error',
        action='store_true',
        help='Exit with non-zero status if violations are found'
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    paths = [Path(p) for p in args.paths]
    exclude_dirs = set(args.exclude)
    
    # Collect all violations
    all_violations = []
    
    for path in paths:
        if path.is_file() and path.suffix == '.py':
            file_violations = validate_file(str(path))
            all_violations.extend(file_violations)
        elif path.is_dir():
            dir_violations = validate_directory(path, exclude_dirs)
            all_violations.extend(dir_violations)
        else:
            print(f"Warning: {path} is not a Python file or directory", file=sys.stderr)
    
    # Group violations by file
    violations_by_file = {}
    for violation in all_violations:
        if violation.file_path not in violations_by_file:
            violations_by_file[violation.file_path] = []
        violations_by_file[violation.file_path].append(violation)
    
    # Sort violations by file and line number
    for file_path in violations_by_file:
        violations_by_file[file_path].sort(key=lambda v: v.line_number)
    
    # Output violations
    output_file = open(args.output, 'w') if args.output else sys.stdout
    
    try:
        if all_violations:
            print(f"Found {len(all_violations)} validation violations:", file=output_file)
            
            for file_path, violations in sorted(violations_by_file.items()):
                print(f"\n{file_path}:", file=output_file)
                
                for violation in violations:
                    print(f"  Line {violation.line_number}: {violation.violation_type} - {violation.message}", 
                          file=output_file)
            
            if args.fail_on_error:
                sys.exit(1)
        else:
            print("No validation violations found.", file=output_file)
    finally:
        if args.output:
            output_file.close()


if __name__ == '__main__':
    main()
