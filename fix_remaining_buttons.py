#!/usr/bin/env python3
"""
Script to fix all remaining Button usage in OpenHCS TUI files.
Replaces 'Button(' with 'SafeButton(' and adds necessary imports.
"""

import os
import re
from pathlib import Path

def fix_button_usage_in_file(file_path):
    """Fix Button usage in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if file has Button usage (excluding SafeButton and FramedButton)
        if not re.search(r'\bButton\(', content) or 'SafeButton(' in content or 'FramedButton(' in content:
            # Skip if no Button usage or already has SafeButton/FramedButton
            if not re.search(r'(?<!Safe)(?<!Framed)\bButton\(', content):
                return False
        
        print(f"Fixing {file_path}")
        
        # Add SafeButton import if not present
        if 'SafeButton' not in content:
            # Find existing imports section
            import_pattern = r'(from prompt_toolkit\.widgets import[^)]*)\)'
            if re.search(import_pattern, content):
                # Add SafeButton to existing prompt_toolkit.widgets import
                content = re.sub(
                    import_pattern,
                    r'\1, SafeButton)',
                    content
                )
                # Add SafeButton class definition after imports
                safebutton_def = '''
# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""
    
    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)
    
    def _get_text_fragments(self):
        """Safe version that handles formatting errors gracefully."""
        try:
            return super()._get_text_fragments()
        except (ValueError, TypeError, AttributeError):
            # Fallback to simple text formatting without centering
            text = str(self.text) if self.text is not None else ""
            safe_text = text.replace('{', '{{').replace('}', '}}')
            return [("class:button", f" {safe_text} ")]

'''
                # Insert after the last import line
                lines = content.split('\n')
                last_import_line = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith(('import ', 'from ')) and not line.strip().startswith('#'):
                        last_import_line = i
                
                if last_import_line >= 0:
                    lines.insert(last_import_line + 1, safebutton_def)
                    content = '\n'.join(lines)
            else:
                # Add import and class definition at the top
                safebutton_import_and_def = '''from prompt_toolkit.widgets import Button

# Define SafeButton locally to avoid circular imports
class SafeButton(Button):
    """Safe wrapper around Button that handles formatting errors."""
    
    def __init__(self, text="", handler=None, width=None, **kwargs):
        # Sanitize text before passing to parent
        if text is not None:
            text = str(text).replace('{', '{{').replace('}', '}}').replace(':', ' ')
        super().__init__(text=text, handler=handler, width=width, **kwargs)
    
    def _get_text_fragments(self):
        """FAIL FAST: No fallback text formatting - if formatting fails, crash immediately."""
        return super()._get_text_fragments()

'''
                content = safebutton_import_and_def + content
        
        # Replace Button( with SafeButton( (but not SafeButton( or FramedButton()
        content = re.sub(r'(?<!Safe)(?<!Framed)\bButton\(', 'SafeButton(', content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """Main function to fix all files."""
    # Get all Python files in TUI directory
    tui_dir = Path('openhcs/tui')
    python_files = list(tui_dir.rglob('*.py'))
    
    fixed_count = 0
    for file_path in python_files:
        if fix_button_usage_in_file(file_path):
            fixed_count += 1
    
    print(f"Fixed {fixed_count} files")

if __name__ == '__main__':
    main()
