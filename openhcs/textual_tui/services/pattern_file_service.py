"""
Pattern File Service - Async-safe file I/O operations for function patterns.

This service handles loading/saving .func files and external editor integration
with proper FileManager abstraction and async safety.
"""

import asyncio
import dill as pickle
import logging
from pathlib import Path
from typing import Union, List, Dict, Optional, Any

from openhcs.textual_tui.services.external_editor_service import ExternalEditorService

logger = logging.getLogger(__name__)


class PatternFileService:
    """
    Async-safe file I/O operations for function patterns.
    
    Handles .func file loading/saving with proper FileManager abstraction
    and external editor integration.
    """
    
    def __init__(self, state: Any):
        """
        Initialize the pattern file service.
        
        Args:
            state: TUIState instance for external editor integration
        """
        self.state = state
        self.external_editor_service = ExternalEditorService(state)
    
    async def load_pattern_from_file(self, file_path: Path) -> Union[List, Dict]:
        """
        Load and validate .func files with async safety.
        
        Uses run_in_executor to prevent event loop deadlocks.
        
        Args:
            file_path: Path to .func file
            
        Returns:
            Loaded pattern (List or Dict)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file content is invalid
            Exception: For other loading errors
        """
        def _sync_load_pattern(path: Path) -> Union[List, Dict]:
            """Synchronous pattern loading for executor."""
            if not path.exists():
                raise FileNotFoundError(f"Pattern file not found: {path}")
            
            if not path.is_file():
                raise ValueError(f"Path is not a file: {path}")
            
            try:
                with open(path, "rb") as f:
                    pattern = pickle.load(f)
                
                # Basic validation
                if not isinstance(pattern, (list, dict)):
                    raise ValueError(f"Invalid pattern type: {type(pattern)}. Expected list or dict.")
                
                return pattern
                
            except pickle.PickleError as e:
                raise ValueError(f"Failed to unpickle pattern file: {e}")
            except Exception as e:
                raise Exception(f"Failed to load pattern file: {e}")
        
        # Use asyncio.get_running_loop() instead of deprecated get_event_loop()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_load_pattern, file_path)
    
    async def save_pattern_to_file(self, pattern: Union[List, Dict], file_path: Path) -> None:
        """
        Save patterns with pickle using async safety.
        
        Uses run_in_executor to prevent event loop deadlocks.
        
        Args:
            pattern: Pattern to save (List or Dict)
            file_path: Path to save to
            
        Raises:
            ValueError: If pattern is invalid
            Exception: For saving errors
        """
        def _sync_save_pattern(pattern_data: Union[List, Dict], path: Path) -> None:
            """Synchronous pattern saving for executor."""
            # Basic validation
            if not isinstance(pattern_data, (list, dict)):
                raise ValueError(f"Invalid pattern type: {type(pattern_data)}. Expected list or dict.")
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                with open(path, "wb") as f:
                    pickle.dump(pattern_data, f)
                    
            except Exception as e:
                raise Exception(f"Failed to save pattern file: {e}")
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _sync_save_pattern, pattern, file_path)
    
    async def edit_pattern_externally(self, pattern: Union[List, Dict]) -> tuple[bool, Union[List, Dict], Optional[str]]:
        """
        Edit pattern in external editor (Vim) via ExternalEditorService.
        
        Args:
            pattern: Pattern to edit
            
        Returns:
            Tuple of (success, new_pattern, error_message)
        """
        try:
            # Format pattern for external editing
            initial_content = f"pattern = {repr(pattern)}"
            
            # Use existing ExternalEditorService
            success, new_pattern, error_message = await self.external_editor_service.edit_pattern_in_external_editor(initial_content)
            
            return success, new_pattern, error_message
            
        except Exception as e:
            logger.error(f"External editor integration failed: {e}")
            return False, pattern, f"External editor failed: {e}"
    
    async def validate_pattern_file(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """
        Validate .func file without loading it completely.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        def _sync_validate_file(path: Path) -> tuple[bool, Optional[str]]:
            """Synchronous file validation for executor."""
            if not path.exists():
                return False, f"File does not exist: {path}"
            
            if not path.is_file():
                return False, f"Path is not a file: {path}"
            
            if not path.suffix == '.func':
                return False, f"File does not have .func extension: {path}"
            
            try:
                # Try to load just the header to check if it's a valid pickle
                with open(path, "rb") as f:
                    # Read first few bytes to check pickle format
                    header = f.read(10)
                    if not header.startswith(b'\x80'):  # Pickle protocol marker
                        return False, "File is not a valid pickle file"
                
                return True, None
                
            except Exception as e:
                return False, f"File validation failed: {e}"
        
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_validate_file, file_path)
    
    def get_default_save_path(self, base_name: str = "pattern") -> str:
        """
        Get default save path for .func files.
        
        Args:
            base_name: Base filename without extension
            
        Returns:
            Default save path string
        """
        return f"{base_name}.func"
    
    def ensure_func_extension(self, file_path: str) -> str:
        """
        Ensure file path has .func extension.
        
        Args:
            file_path: Original file path
            
        Returns:
            File path with .func extension
        """
        path = Path(file_path)
        if path.suffix != '.func':
            return str(path.with_suffix('.func'))
        return file_path
    
    async def backup_pattern_file(self, file_path: Path) -> Optional[Path]:
        """
        Create backup of existing pattern file before overwriting.
        
        Args:
            file_path: Original file path
            
        Returns:
            Backup file path if created, None if no backup needed
        """
        if not file_path.exists():
            return None
        
        def _sync_backup_file(original_path: Path) -> Path:
            """Synchronous file backup for executor."""
            backup_path = original_path.with_suffix(f"{original_path.suffix}.backup")
            
            # If backup already exists, add timestamp
            if backup_path.exists():
                import time
                timestamp = int(time.time())
                backup_path = original_path.with_suffix(f"{original_path.suffix}.backup.{timestamp}")
            
            # Copy file
            import shutil
            shutil.copy2(original_path, backup_path)
            return backup_path
        
        try:
            loop = asyncio.get_running_loop()
            backup_path = await loop.run_in_executor(None, _sync_backup_file, file_path)
            return backup_path
        except Exception as e:
            logger.warning(f"Failed to create backup for {file_path}: {e}")
            return None
