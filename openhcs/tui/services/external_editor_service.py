import ast
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union

from prompt_toolkit.application import get_app
from prompt_toolkit.widgets import Dialog, Label, Button
from prompt_toolkit.layout import HSplit

from openhcs.core.pipeline.funcstep_contract_validator import FuncStepContractValidator

# SafeButton eliminated - use Button directly

class ExternalEditorService:
    """
    Service for handling external text editor interactions,
    specifically for editing Python literal patterns.
    """
    def __init__(self, state: Any):
        self.state = state # TUIState instance

    async def edit_pattern_in_external_editor(self, initial_content: str) -> Tuple[bool, Optional[Union[List, Dict]], Optional[str]]:
        """
        Launches an external editor (e.g., Vim) with the given content,
        waits for the user to edit, and then validates the content.

        Args:
            initial_content: The initial string content to put into the editor.

        Returns:
            A tuple: (success: bool, pattern: Optional[Union[List, Dict]], error_message: Optional[str])
        """
        editor = os.environ.get('EDITOR', 'vim')

        # Create a temporary file with the initial content
        with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.py', encoding='utf-8') as tmp_file:
            tmp_file.write(initial_content)
            tmp_file_path = tmp_file.name

        try:
            # Launch the external editor
            # Use get_app().run_system_command for integration with prompt_toolkit's event loop
            await get_app().run_system_command(f"{editor} {tmp_file_path}", wait_for_enter=True)

            # Read the modified content
            with open(tmp_file_path, "r", encoding="utf-8") as f:
                modified_content = f.read()

            # Validate the modified content
            is_valid, pattern, error_message = self._validate_pattern_file(modified_content)

            if not is_valid:
                await self._show_error_dialog(f"Validation Error:\n{error_message}")
                return False, None, error_message

            return True, pattern, None

        except FileNotFoundError:
            await self._show_error_dialog(f"Editor '{editor}' not found. Please ensure it's installed and in your PATH.")
            return False, None, f"Editor '{editor}' not found."
        except Exception as e:
            await self._show_error_dialog(f"An error occurred while editing: {e}")
            return False, None, str(e)
        finally:
            # Clean up the temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def _validate_pattern_file(self, content: str) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Validate that a file contains only a valid pattern assignment.

        Args:
            content: The content of the file to validate

        Returns:
            Tuple of (is_valid, pattern, error_message)
        """
        try:
            # Parse the file
            tree = ast.parse(content)

            # Check that there's only one statement
            if len(tree.body) != 1:
                return False, None, "File must contain exactly one statement (pattern assignment)"

            # Check that the statement is an assignment
            stmt = tree.body[0]
            if not isinstance(stmt, ast.Assign):
                return False, None, "File must contain a pattern assignment"

            # Check that the assignment target is 'pattern'
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name) or stmt.targets[0].id != 'pattern':
                return False, None, "Assignment target must be 'pattern'"

            # Use ast.literal_eval to safely evaluate the pattern
            pattern_str = ast.unparse(stmt.value) if hasattr(ast, 'unparse') else content.strip().split('=', 1)[1].strip()
            try:
                pattern = ast.literal_eval(pattern_str)

                # Validate pattern structure using FuncStepContractValidator
                # This will raise ValueError if the pattern is invalid
                try:
                    # Extract functions to validate pattern structure
                    # We don't care about the actual functions, just that the structure is valid
                    FuncStepContractValidator._extract_functions_from_pattern(
                        pattern, "External Editor Service"
                    )
                    return True, pattern, None
                except ValueError as e:
                    return False, None, f"Invalid pattern structure: {str(e)}"

            except (ValueError, SyntaxError) as e:
                # If literal_eval fails, the pattern contains non-literal expressions
                return False, None, f"Pattern must contain only literals: {str(e)}"

        except SyntaxError as e:
            return False, None, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, None, f"Validation error: {str(e)}"

    async def _show_error_dialog(self, message: str):
        """Helper to show an error dialog."""
        dialog = Dialog(
            title="Error",
            body=HSplit([Label(message)]),
            buttons=[Button("OK", width=len("OK") + 2)]
        )
        await self.state.show_dialog(dialog) # Assuming TUIState has a show_dialog method