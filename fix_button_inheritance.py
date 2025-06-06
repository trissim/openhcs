#!/usr/bin/env python3
"""
Script to fix circular Button inheritance in OpenHCS TUI files.
Replaces 'class Button(Button):' with 'class SafeButton(Button):' across all files.
"""

import os
import re
from pathlib import Path

def fix_button_inheritance(file_path):
    """Fix circular Button inheritance in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has the circular inheritance issue
        if 'class Button(Button):' not in content:
            return False
        
        print(f"Fixing {file_path}")
        
        # Replace circular inheritance
        content = content.replace('class Button(Button):', 'class SafeButton(Button):')
        
        # Also need to update any references to the old Button class within the same file
        # This is more complex and might need manual review, but for now we'll do basic replacement
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all files."""
    # Files with circular Button inheritance
    files_to_fix = [
        'openhcs/tui/dialogs/global_settings_editor.py',
        'openhcs/tui/dialogs/plate_config_editor.py', 
        'openhcs/tui/dialogs/plate_dialog_manager.py',
        'openhcs/tui/dialogs/help_dialog.py',
        'openhcs/tui/file_browser.py',
        'openhcs/tui/components/interactive_list_item.py',
        'openhcs/tui/components/parameter_editor.py',
        'openhcs/tui/dual_step_func_editor.py',
        'openhcs/tui/function_pattern_editor.py',
        'openhcs/tui/services/external_editor_service.py',
        'openhcs/tui/utils/dialog_helpers.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_button_inheritance(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
