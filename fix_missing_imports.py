#!/usr/bin/env python3
"""
Fix all missing get_task_manager imports systematically.
"""
import re
import os

def fix_file_imports(filepath):
    """Fix missing get_task_manager imports in a file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find get_task_manager() calls without local import
    lines = content.split('\n')
    modified = False
    
    for i, line in enumerate(lines):
        if 'get_task_manager()' in line:
            # Check if this line is inside a function
            indent_level = len(line) - len(line.lstrip())
            
            # Look backwards to find the function definition
            func_start = None
            for j in range(i - 1, -1, -1):
                if lines[j].strip().startswith('def ') and len(lines[j]) - len(lines[j].lstrip()) < indent_level:
                    func_start = j
                    break
            
            if func_start is not None:
                # Check if there's already an import in this function
                has_import = False
                for k in range(func_start + 1, i):
                    if 'from openhcs.tui.utils.unified_task_manager import get_task_manager' in lines[k]:
                        has_import = True
                        break
                
                if not has_import:
                    # Find the right place to insert the import (after function def and docstring)
                    insert_pos = func_start + 1
                    
                    # Skip docstring if present
                    if insert_pos < len(lines) and '"""' in lines[insert_pos]:
                        # Multi-line docstring
                        while insert_pos < len(lines) and not (lines[insert_pos].count('"""') >= 2 or (insert_pos > func_start + 1 and '"""' in lines[insert_pos])):
                            insert_pos += 1
                        insert_pos += 1
                    elif insert_pos < len(lines) and lines[insert_pos].strip().startswith('"""'):
                        # Single line docstring
                        insert_pos += 1
                    
                    # Get the indentation of the function body
                    func_indent = ''
                    for k in range(insert_pos, len(lines)):
                        if lines[k].strip():
                            func_indent = lines[k][:len(lines[k]) - len(lines[k].lstrip())]
                            break
                    
                    # Insert the import
                    import_line = func_indent + 'from openhcs.tui.utils.unified_task_manager import get_task_manager'
                    lines.insert(insert_pos, import_line)
                    modified = True
                    print(f"Added import to {filepath} at line {insert_pos + 1}")
    
    if modified:
        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))
        print(f"✅ Fixed imports in {filepath}")
    
    return modified

def main():
    """Fix all missing imports in TUI files."""
    files_to_fix = [
        'openhcs/tui/editors/function_pattern_editor.py',
        'openhcs/tui/panes/plate_manager.py', 
        'openhcs/tui/components/list_manager.py',
        'openhcs/tui/components/parameter_editor.py',
        'openhcs/tui/layout/simple_launcher.py',
        'openhcs/tui/layout/status_bar.py'
    ]
    
    total_fixed = 0
    for filepath in files_to_fix:
        if os.path.exists(filepath):
            if fix_file_imports(filepath):
                total_fixed += 1
        else:
            print(f"❌ File not found: {filepath}")
    
    print(f"\n🎉 Fixed imports in {total_fixed} files")

if __name__ == "__main__":
    main()
