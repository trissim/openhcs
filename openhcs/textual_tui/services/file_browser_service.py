"""File browser service for FileManager integration."""

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from enum import Enum

from openhcs.constants.constants import Backend
from openhcs.io.filemanager import FileManager


class SelectionMode(Enum):
    """File selection mode."""
    FILES_ONLY = "files"
    DIRECTORIES_ONLY = "directories"
    BOTH = "both"


@dataclass
class FileItem:
    """File or directory item."""
    name: str
    path: Path
    is_dir: bool
    size: Optional[int] = None
    mtime: Optional[float] = None
    
    @property
    def display_size(self) -> str:
        """Get formatted size string."""
        if self.size is None or self.is_dir:
            return ""
        
        size = self.size
        if size < 1024:
            return f"{size} B"
        
        for unit in ['KB', 'MB', 'GB', 'TB']:
            size /= 1024.0
            if size < 1024.0:
                return f"{size:.1f} {unit}"
        return f"{size:.1f} PB"
    
    @property
    def display_mtime(self) -> str:
        """Get formatted modification time."""
        if self.mtime is None:
            return ""
        try:
            return datetime.datetime.fromtimestamp(self.mtime).strftime('%Y-%m-%d %H:%M')
        except (ValueError, OSError):
            return ""


class FileBrowserService:
    """Service for file browser operations using FileManager."""
    
    def __init__(self, file_manager: FileManager):
        self.file_manager = file_manager
    
    def load_directory(self, path: Path, backend: Backend) -> List[FileItem]:
        """Load directory contents using FileManager."""
        # Convert Backend enum to string for FileManager (evidence: file_browser.py line 110)
        backend_str = backend.value

        try:
            entries = self.file_manager.list_dir(path, backend_str)
            items = []

            for name in entries:
                item_path = path / name
                try:
                    is_dir = self.file_manager.is_dir(item_path, backend_str)
                except (NotADirectoryError, FileNotFoundError):
                    # Path is a file, not a directory, or doesn't exist (broken symlink)
                    is_dir = False
                items.append(FileItem(name=name, path=item_path, is_dir=is_dir))

            # Sort: directories first, then files, alphabetically
            return sorted(items, key=lambda x: (not x.is_dir, x.name.lower()))

        except Exception as e:
            # Log error but return empty list for robustness
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to load directory {path}: {e}")
            return []
    
    def can_select_item(self, item: FileItem, selection_mode: SelectionMode) -> bool:
        """Check if item can be selected based on mode."""
        if selection_mode == SelectionMode.FILES_ONLY:
            return not item.is_dir
        elif selection_mode == SelectionMode.DIRECTORIES_ONLY:
            return item.is_dir
        else:  # SelectionMode.BOTH
            return True
    
    def filter_items(self, items: List[FileItem], 
                    show_hidden: bool = False,
                    extensions: Optional[List[str]] = None) -> List[FileItem]:
        """Filter items based on criteria."""
        filtered = []
        
        for item in items:
            # Hidden files filter
            if not show_hidden and item.name.startswith('.'):
                continue
            
            # Extension filter (only for files)
            if extensions and not item.is_dir:
                if not any(item.name.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            filtered.append(item)
        
        return filtered
