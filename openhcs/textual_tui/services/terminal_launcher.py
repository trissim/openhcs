"""
Terminal launcher service for running commands in TUI terminal windows.

Provides a clean interface for launching terminal applications within the existing
TUI terminal infrastructure instead of external processes.
"""

import tempfile
import os
from pathlib import Path
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class TerminalLauncher:
    """Service for launching terminal applications within TUI terminal windows."""
    
    def __init__(self, app):
        """
        Initialize terminal launcher.
        
        Args:
            app: The TUI application instance
        """
        self.app = app
    
    async def launch_editor_for_file(self, file_content: str, file_extension: str = '.py', 
                                   on_save_callback: Optional[Callable[[str], None]] = None) -> None:
        """
        Launch an editor in a terminal window for editing file content.
        
        Args:
            file_content: Initial content to edit
            file_extension: File extension (e.g., '.py', '.txt')
            on_save_callback: Callback function called with edited content when saved
        """
        try:
            # Create temporary file with content
            with tempfile.NamedTemporaryFile(mode='w', suffix=file_extension, delete=False) as f:
                f.write(file_content)
                temp_path = f.name
            
            # Get editor from environment
            editor = os.environ.get('EDITOR', 'nano')
            
            # Create command that will edit the file and then read it back
            command = self._create_editor_command(editor, temp_path, on_save_callback)
            
            # Launch terminal window with the command
            await self._launch_terminal_with_command(command, f"Edit File ({editor})")
            
        except Exception as e:
            logger.error(f"Failed to launch editor: {e}")
            self.app.show_error("Editor Error", f"Failed to launch editor: {str(e)}")
    
    def _create_editor_command(self, editor: str, file_path: str, 
                             on_save_callback: Optional[Callable[[str], None]]) -> str:
        """
        Create a shell command that runs the editor and handles the callback.
        
        Args:
            editor: Editor command (e.g., 'vim', 'nano')
            file_path: Path to temporary file
            on_save_callback: Callback for when file is saved
            
        Returns:
            Shell command string
        """
        # Create a wrapper script that:
        # 1. Runs the editor
        # 2. Reads the file content after editing
        # 3. Calls the callback with the content
        # 4. Cleans up the temp file
        
        callback_script = self._create_callback_script(file_path, on_save_callback)
        
        # Command that runs editor in a proper login shell environment
        command = f"""
# Source user's shell configuration
if [ -f ~/.bashrc ]; then source ~/.bashrc; fi
if [ -f ~/.zshrc ]; then source ~/.zshrc; fi
if [ -f ~/.profile ]; then source ~/.profile; fi

echo "Opening {editor} for editing..."
echo "Save and exit to apply changes, or exit without saving to cancel."
echo ""
{editor} "{file_path}"
echo ""
echo "Editor closed. Processing changes..."
python3 "{callback_script}"
echo "Terminal will close automatically."
exit 0
"""
        return command.strip()
    
    def _create_callback_script(self, file_path: str,
                              on_save_callback: Optional[Callable[[str], None]]) -> str:
        """
        Create a simple script that signals completion without importing OpenHCS.

        Args:
            file_path: Path to the edited file
            on_save_callback: Callback function

        Returns:
            Path to callback script
        """
        if on_save_callback:
            # Store callback and create signal file approach
            callback_id = id(on_save_callback)
            self.app._terminal_callbacks = getattr(self.app, '_terminal_callbacks', {})
            self.app._terminal_callbacks[callback_id] = on_save_callback

            # Create signal file path
            signal_file = f"{file_path}.done"

            script_content = f"""
# Simple completion signal - no OpenHCS imports needed
import os

try:
    print("Editor session completed.")

    # Create signal file to notify main process
    with open("{signal_file}", 'w') as f:
        f.write("{callback_id}")

    print("Changes will be processed by main application.")

except Exception as e:
    print(f"Error creating signal file: {{e}}")
"""

            # Start polling for the signal file in main process
            # We'll pass the terminal window reference when we create it
            self._pending_callback = (file_path, signal_file, callback_id)

        else:
            # No callback, just clean up
            script_content = f"""
import os
try:
    os.unlink("{file_path}")
    print("Temporary file cleaned up.")
except:
    pass
"""

        # Write callback script to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(script_content)
            return f.name

    def _start_polling(self, file_path: str, signal_file: str, callback_id: int, terminal_window=None) -> None:
        """Start polling for completion signal file."""
        import asyncio

        async def poll_for_completion():
            """Poll for signal file and handle callback."""
            while True:
                try:
                    if os.path.exists(signal_file):
                        logger.info(f"Signal file detected: {signal_file}")
                        # Signal file exists, read the edited content
                        with open(file_path, 'r') as f:
                            content = f.read()

                        # Get and call the callback
                        callbacks = getattr(self.app, '_terminal_callbacks', {})
                        callback = callbacks.get(callback_id)
                        if callback:
                            logger.info("Calling editor callback")
                            callback(content)
                        else:
                            logger.warning(f"No callback found for ID: {callback_id}")

                        # Close the terminal window if provided
                        if terminal_window:
                            try:
                                logger.info(f"Closing terminal window: {terminal_window}")
                                terminal_window.close_window()
                                logger.info("Terminal window closed successfully")
                            except Exception as e:
                                logger.error(f"Error closing terminal window: {e}")
                        else:
                            logger.warning("No terminal window reference found for cleanup")

                        # Clean up
                        try:
                            os.unlink(file_path)
                            os.unlink(signal_file)
                            callbacks.pop(callback_id, None)
                        except:
                            pass

                        break

                    # Wait before checking again
                    await asyncio.sleep(0.5)

                except Exception as e:
                    logger.error(f"Error in polling: {e}")
                    break

        # Start the polling task
        asyncio.create_task(poll_for_completion())

    def _create_login_shell_wrapper(self, command: str) -> str:
        """Create a wrapper script that runs command in a login shell environment."""
        import os
        import tempfile
        import stat

        # Get user's shell (same logic as TerminalWindow)
        user_shell = self._get_user_shell()

        # Create script that sources user configs and runs command
        script_content = f'''#!/bin/bash
# Run in login shell to load user environment
exec {user_shell} -l -c "{command.replace('"', '\\"')}"
'''

        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(script_content)
            script_path = f.name

        # Make executable
        os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

        return script_path

    def _get_user_shell(self) -> str:
        """Get user's preferred shell (same logic as TerminalWindow)."""
        import os

        # Method 1: Check SHELL environment variable
        if 'SHELL' in os.environ and os.path.exists(os.environ['SHELL']):
            return os.environ['SHELL']

        # Method 2: Check /etc/passwd
        try:
            import pwd
            user_shell = pwd.getpwuid(os.getuid()).pw_shell
            if user_shell and os.path.exists(user_shell):
                return user_shell
        except (ImportError, KeyError, OSError):
            pass

        # Method 3: Try common shells
        common_shells = ['/bin/zsh', '/usr/bin/zsh', '/bin/bash', '/usr/bin/bash']
        for shell_path in common_shells:
            if os.path.exists(shell_path):
                return shell_path

        # Fallback
        return '/bin/bash'

    async def _launch_terminal_with_command(self, command: str, title: str = "Terminal") -> None:
        """
        Launch a terminal window with a specific command.
        
        Args:
            command: Shell command to run
            title: Window title
        """
        from openhcs.textual_tui.windows.terminal_window import TerminalWindow
        from textual.css.query import NoMatches
        
        # Create wrapper command that runs our command in a login shell (like regular terminal)
        shell_command = self._create_login_shell_wrapper(command)
        
        try:
            # Try to find existing terminal window
            window = self.app.query_one(TerminalWindow)
            # If terminal exists, we could either reuse it or create a new one
            # For now, let's create a new one for the editor
            window = TerminalWindow(shell_command=shell_command)
            await self.app.mount(window)
            window.open_state = True

        except NoMatches:
            # No existing terminal, create new one
            window = TerminalWindow(shell_command=shell_command)
            await self.app.mount(window)
            window.open_state = True

        # Start polling with the terminal window reference if we have a pending callback
        if hasattr(self, '_pending_callback'):
            file_path, signal_file, callback_id = self._pending_callback
            logger.info(f"Starting polling with terminal window reference: {window}")
            self._start_polling(file_path, signal_file, callback_id, window)
            delattr(self, '_pending_callback')
