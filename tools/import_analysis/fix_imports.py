#!/usr/bin/env python3
"""
Import Fixer - A tool to automatically fix common import issues.

This tool:
1. Fixes deprecated imports (StorageBackendEnum -> Backend)
2. Adds missing Container imports
3. Fixes RadioList and CheckboxList imports
4. Removes DialogResult imports
"""

import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional

# Add the project root to the path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

# Import replacements (old -> new)
IMPORT_REPLACEMENTS = {
    "from openhcs.io.base import StorageBackendEnum": "from openhcs.constants.constants import Backend",
    "from openhcs.core.memory.storage_backend import StorageBackend": "from openhcs.constants.constants import Backend",
    "from prompt_toolkit.shortcuts import message_dialog, DialogResult": "from prompt_toolkit.shortcuts import message_dialog",
}

# Missing imports to add
MISSING_IMPORTS = {
    "Container": "from prompt_toolkit.layout import Container",
    "RadioList": "from prompt_toolkit.widgets import RadioList",
    "CheckboxList": "from prompt_toolkit.widgets import CheckboxList",
    "Optional": "from typing import Optional",
    "List": "from typing import List",
    "Dict": "from typing import Dict",
    "Union": "from typing import Union",
    "Tuple": "from typing import Tuple",
    "Any": "from typing import Any",
    "Callable": "from typing import Callable",
}

def fix_file_imports(file_path: str, dry_run: bool = False) -> Tuple[bool, List[str]]:
    """
    Fix import issues in a file.

    Args:
        file_path: Path to the file to fix
        dry_run: If True, don't actually modify the file

    Returns:
        Tuple of (success, list of changes made)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes = []

        # Fix deprecated imports
        for old_import, new_import in IMPORT_REPLACEMENTS.items():
            if old_import in content:
                content = content.replace(old_import, new_import)
                changes.append(f"Replaced '{old_import}' with '{new_import}'")

        # Fix Container import
        if "Container" in content and "from prompt_toolkit.layout import Container" not in content:
            # Check if there's already a prompt_toolkit.layout import we can add to
            layout_import_match = re.search(r'from prompt_toolkit.layout import \((.*?)\)', content, re.DOTALL)
            if layout_import_match:
                # Add Container to existing import if not already there
                import_content = layout_import_match.group(1)
                if "Container," not in import_content and "Container\n" not in import_content:
                    # Find the last item in the import list
                    last_item_match = re.search(r'([^\s,]+)(\s*#[^\n]*)?\s*\)', import_content)
                    if last_item_match:
                        last_item = last_item_match.group(1)
                        # Replace the last item with last_item, Container
                        new_import_content = import_content.replace(
                            last_item,
                            f"{last_item},\n    Container"
                        )
                        content = content.replace(
                            f"from prompt_toolkit.layout import ({import_content})",
                            f"from prompt_toolkit.layout import ({new_import_content})"
                        )
                        changes.append("Added Container to existing prompt_toolkit.layout import")
            else:
                # Add a new import
                content = f"from prompt_toolkit.layout import Container\n{content}"
                changes.append("Added new import: from prompt_toolkit.layout import Container")

        # Fix DialogResult references
        if "DialogResult" in content:
            # Remove DialogResult type annotations
            content = re.sub(r'\s*\|\s*DialogResult', '', content)
            changes.append("Removed DialogResult from type annotations")

            # Remove isinstance checks for DialogResult
            content = re.sub(r'not isinstance\([^,]+, DialogResult\) and ', '', content)
            changes.append("Removed isinstance checks for DialogResult")

        # Fix RadioList and CheckboxList imports
        if ("RadioList" in content or "CheckboxList" in content) and "from prompt_toolkit.widgets import" not in content:
            widgets_to_add = []
            if "RadioList" in content:
                widgets_to_add.append("RadioList")
            if "CheckboxList" in content:
                widgets_to_add.append("CheckboxList")

            if widgets_to_add:
                content = f"from prompt_toolkit.widgets import {', '.join(widgets_to_add)}\n{content}"
                changes.append(f"Added new import: from prompt_toolkit.widgets import {', '.join(widgets_to_add)}")

        # Fix typing imports (Optional, List, Dict, etc.)
        for type_name in ["Optional", "List", "Dict", "Union", "Tuple", "Any", "Callable"]:
            if type_name in content and "from typing import" in content and type_name not in content.split("from typing import")[1].split("\n")[0]:
                # Add the type to existing typing import
                typing_import_match = re.search(r'from typing import ([^\n]+)', content)
                if typing_import_match:
                    imports = typing_import_match.group(1)
                    if type_name not in imports:
                        new_imports = f"{imports}, {type_name}"
                        content = content.replace(f"from typing import {imports}", f"from typing import {new_imports}")
                        changes.append(f"Added {type_name} to existing typing import")
            elif type_name in content and "from typing import" not in content:
                # Add a new typing import
                content = f"from typing import {type_name}\n{content}"
                changes.append(f"Added new import: from typing import {type_name}")

        # Fix StorageBackend.DISK references
        if "StorageBackend.DISK" in content:
            content = content.replace("StorageBackend.DISK", "Backend.DISK")
            changes.append("Replaced StorageBackend.DISK with Backend.DISK")

        # Fix StorageBackendEnum.LOCAL references
        if "StorageBackendEnum.LOCAL" in content:
            content = content.replace("StorageBackendEnum.LOCAL", "Backend.DISK")
            changes.append("Replaced StorageBackendEnum.LOCAL with Backend.DISK")

        # Only write the file if changes were made and not in dry run mode
        if content != original_content and not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes

        return content != original_content, changes

    except Exception as e:
        return False, [f"Error: {str(e)}"]

def fix_directory_imports(directory: str, dry_run: bool = False) -> Dict[str, List[str]]:
    """
    Fix import issues in all Python files in a directory.

    Args:
        directory: Directory to process
        dry_run: If True, don't actually modify files

    Returns:
        Dictionary mapping file paths to lists of changes made
    """
    results = {}

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                success, changes = fix_file_imports(file_path, dry_run)
                if changes:
                    rel_path = os.path.relpath(file_path, PROJECT_ROOT)
                    results[rel_path] = changes

    return results

def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Import Fixer - Automatically fix common import issues",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "path",
        help="File or directory to process"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually modify files, just show what would be changed"
    )

    args = parser.parse_args()

    path = args.path
    dry_run = args.dry_run

    if os.path.isfile(path):
        success, changes = fix_file_imports(path, dry_run)
        if changes:
            print(f"Changes for {path}:")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"No changes needed for {path}")
    elif os.path.isdir(path):
        results = fix_directory_imports(path, dry_run)
        if results:
            print(f"Changes for {len(results)} files:")
            for file_path, changes in results.items():
                print(f"\n{file_path}:")
                for change in changes:
                    print(f"  - {change}")
        else:
            print("No changes needed")
    else:
        print(f"Error: {path} is not a valid file or directory")
        return 1

    if dry_run:
        print("\nThis was a dry run. No files were modified.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
