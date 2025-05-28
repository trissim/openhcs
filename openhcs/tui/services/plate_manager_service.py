"""
Plate Manager Service - Business Logic Layer.

Separates business logic from UI concerns for better architecture.
Handles plate data management, validation, and orchestrator lifecycle.

🔒 Clause 295: Component Boundaries
Clear separation between service layer and UI layer.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.constants.constants import Backend

logger = logging.getLogger(__name__)


class PlateManagerService:
    """
    Service layer for plate management operations.
    
    Handles:
    - Plate data lifecycle
    - Orchestrator management
    - File system operations
    - Validation coordination
    """
    
    def __init__(self, context: ProcessingContext, storage_registry: Any):
        self.context = context
        self.registry = storage_registry
        self.filemanager = context.filemanager if hasattr(context, 'filemanager') else None
        
        # Thread-safe plate storage
        self.plates: List[Dict[str, Any]] = []
        self.plates_lock = asyncio.Lock()
        
        # I/O executor for file operations
        self.io_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="plate-io-")
        
    async def add_plate(self, path_str: str, global_config: Any) -> Dict[str, Any]:
        """
        Add a new plate to the manager.
        
        Args:
            path_str: Path to the plate directory
            global_config: Global configuration object
            
        Returns:
            Dictionary containing plate details
            
        Raises:
            ValueError: If plate already exists or invalid path
        """
        try:
            orchestrator = PipelineOrchestrator(
                plate_path=path_str, 
                config=global_config, 
                storage_registry=self.registry
            )
            
            plate_tui_id = str(Path(path_str).name) + f"_{orchestrator.config.vfs.default_storage_backend}"
            new_plate_entry = {
                'id': plate_tui_id, 
                'name': Path(path_str).name, 
                'path': path_str,
                'status': 'not_initialized', 
                'orchestrator': orchestrator,
                'backend': orchestrator.config.vfs.default_storage_backend
            }
            
            async with self.plates_lock:
                if any(p['id'] == new_plate_entry['id'] for p in self.plates):
                    raise ValueError(f"Plate with TUI ID '{new_plate_entry['id']}' already exists")
                self.plates.append(new_plate_entry)
                
            logger.info(f"Added new plate '{new_plate_entry['name']}' (ID: {new_plate_entry['id']})")
            return new_plate_entry
            
        except Exception as e:
            logger.error(f"Error creating orchestrator for path '{path_str}': {e}", exc_info=True)
            raise
    
    async def remove_plates(self, plate_ids: List[str]) -> int:
        """
        Remove plates by their IDs.
        
        Args:
            plate_ids: List of plate IDs to remove
            
        Returns:
            Number of plates actually removed
        """
        async with self.plates_lock:
            original_length = len(self.plates)
            self.plates = [p for p in self.plates if p.get('id') not in plate_ids]
            num_removed = original_length - len(self.plates)
            
        logger.info(f"Removed {num_removed} plates with IDs: {plate_ids}")
        return num_removed
    
    async def update_plate_status(self, plate_id: str, new_status: str, message: Optional[str] = None) -> bool:
        """
        Update the status of a specific plate.
        
        Args:
            plate_id: ID of the plate to update
            new_status: New status value
            message: Optional status message
            
        Returns:
            True if plate was found and updated, False otherwise
        """
        async with self.plates_lock:
            for plate in self.plates:
                if plate.get('id') == plate_id:
                    plate['status'] = new_status
                    if message and new_status == 'error':
                        plate['error_message'] = message
                    else:
                        plate.pop('error_message', None)
                    return True
        return False
    
    async def get_plates(self) -> List[Dict[str, Any]]:
        """Get a copy of all plates."""
        async with self.plates_lock:
            return list(self.plates)
    
    async def get_plate_by_id(self, plate_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific plate by ID."""
        async with self.plates_lock:
            for plate in self.plates:
                if plate.get('id') == plate_id:
                    return dict(plate)  # Return copy
        return None
    
    async def refresh_plates_from_directory(self, common_output_directory: str) -> List[Dict[str, Any]]:
        """
        Refresh plates by scanning a directory.
        
        Args:
            common_output_directory: Directory to scan for plates
            
        Returns:
            List of discovered plate entries
        """
        if not self.filemanager:
            raise RuntimeError("FileManager not available for directory scanning")
            
        # Ensure directory exists
        if not self.filemanager.exists(common_output_directory, backend='disk'):
            logger.info(f"Creating common output directory: {common_output_directory}")
            self.filemanager.make_dir(common_output_directory, backend='disk')
        
        # Discover plate paths
        paths = self.filemanager.list_dir(common_output_directory, backend='disk')
        plate_paths = [path for path in paths if self.filemanager.is_dir(path, backend='disk')]
        
        # Create plate entries
        new_plates = []
        for path in plate_paths:
            plate_detail = self._create_plate_detail_from_path(path)
            new_plates.append(plate_detail)
        
        # Update internal storage
        async with self.plates_lock:
            self.plates = new_plates
            
        logger.info(f"Refreshed {len(new_plates)} plates from directory")
        return new_plates
    
    def _create_plate_detail_from_path(self, path: str) -> Dict[str, Any]:
        """Create a plate detail entry from a path."""
        plate_id = Path(path).name
        return {
            'id': plate_id,
            'path': str(path),
            'status': 'not_initialized',
            'name': plate_id,
            'orchestrator': None
        }
    
    async def shutdown(self):
        """Clean up resources."""
        logger.info("PlateManagerService: Shutting down...")
        if hasattr(self, 'io_executor') and self.io_executor is not None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.io_executor.shutdown, True)
            self.io_executor = None
        logger.info("PlateManagerService: Shutdown complete.")
