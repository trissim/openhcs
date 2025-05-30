"""
Orchestrator Manager for OpenHCS TUI.

Manages PipelineOrchestrator instances for plates, providing a clean interface
for the TUI to create, access, and manage orchestrators without directly
handling the orchestrator lifecycle.
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import RLock

from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.config import GlobalPipelineConfig

logger = logging.getLogger(__name__)


class OrchestratorManager:
    """
    Manages PipelineOrchestrator instances for the TUI.
    
    Provides thread-safe access to orchestrators and handles their lifecycle.
    Based on the orchestrator management from tui_launcher.py.
    """
    
    def __init__(self, global_config: GlobalPipelineConfig, storage_registry, common_output_root: Path):
        """
        Initialize the orchestrator manager.
        
        Args:
            global_config: Global pipeline configuration
            storage_registry: Shared storage registry
            common_output_root: Root directory for orchestrator workspaces
        """
        self.global_config = global_config
        self.storage_registry = storage_registry
        self.common_output_root = common_output_root
        
        # Thread-safe orchestrator storage
        self.orchestrators: Dict[str, PipelineOrchestrator] = {}
        self.orchestrators_lock = RLock()
        
        logger.info("OrchestratorManager initialized")
    
    async def add_plate(self, plate_id: str, plate_path: str) -> bool:
        """
        Add a plate and create its orchestrator.
        
        Args:
            plate_id: Unique identifier for the plate
            plate_path: Path to the plate directory
            
        Returns:
            True if successful, False otherwise
        """
        async with asyncio.Lock():  # Use async lock for async method
            if plate_id in self.orchestrators:
                logger.warning(f"Plate '{plate_id}' already has an orchestrator. Ignoring add request.")
                return False
            
            try:
                # Construct plate-specific workspace path
                safe_plate_id = plate_id.replace(':', '_').replace('/', '_').replace('\\', '_')
                workspace_path = self.common_output_root / f"plate_{safe_plate_id}"
                
                logger.debug(f"Creating PipelineOrchestrator for plate '{plate_id}'.")
                orchestrator = PipelineOrchestrator(
                    plate_path=plate_path,
                    workspace_path=workspace_path,
                    global_config=self.global_config,
                    storage_registry=self.storage_registry
                )
                
                # Store the orchestrator (not initialized yet)
                with self.orchestrators_lock:
                    self.orchestrators[plate_id] = orchestrator
                
                logger.info(f"PipelineOrchestrator created for plate '{plate_id}'. Initialization pending.")
                return True
                
            except Exception as e:
                logger.error(f"Failed to create orchestrator for plate '{plate_id}': {e}", exc_info=True)
                return False
    
    async def register_existing_orchestrator(self, plate_id: str, orchestrator: PipelineOrchestrator) -> bool:
        """
        Register an existing orchestrator created by PlateManagerService.

        This method enables coordination between PlateManagerService and OrchestratorManager
        without creating duplicate orchestrator instances.

        Args:
            plate_id: Unique identifier for the plate
            orchestrator: Existing PipelineOrchestrator instance

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Registering existing orchestrator for plate: id='{plate_id}'")

        with self.orchestrators_lock:
            if plate_id in self.orchestrators:
                logger.warning(f"Plate '{plate_id}' already has a registered orchestrator. Replacing with new one.")

            self.orchestrators[plate_id] = orchestrator
            logger.info(f"Orchestrator for plate '{plate_id}' registered successfully.")
            return True

    async def unregister_orchestrator(self, plate_id: str) -> bool:
        """
        Unregister an orchestrator without shutting it down.

        This is used for coordination when PlateManagerService removes a plate
        but manages the orchestrator lifecycle itself.

        Args:
            plate_id: Unique identifier for the plate

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Unregistering orchestrator for plate: id='{plate_id}'")

        with self.orchestrators_lock:
            if plate_id in self.orchestrators:
                self.orchestrators.pop(plate_id)
                logger.info(f"Orchestrator for plate '{plate_id}' unregistered.")
                return True
            else:
                logger.warning(f"Attempted to unregister plate '{plate_id}', but no orchestrator found.")
                return False

    async def remove_plate(self, plate_id: str) -> bool:
        """
        Remove a plate and its orchestrator.

        Args:
            plate_id: Unique identifier for the plate

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Attempting to remove plate: id='{plate_id}'")

        with self.orchestrators_lock:
            if plate_id in self.orchestrators:
                removed_orchestrator = self.orchestrators.pop(plate_id)
                logger.info(f"Orchestrator for plate '{plate_id}' removed.")
                # TODO: Add proper orchestrator cleanup if needed
                return True
            else:
                logger.warning(f"Attempted to remove plate '{plate_id}', but no orchestrator found.")
                return False
    
    def get_orchestrator(self, plate_id: str) -> Optional[PipelineOrchestrator]:
        """
        Get the orchestrator for a specific plate.
        
        Args:
            plate_id: Unique identifier for the plate
            
        Returns:
            PipelineOrchestrator instance or None if not found
        """
        with self.orchestrators_lock:
            return self.orchestrators.get(plate_id)
    
    def get_all_orchestrators(self) -> Dict[str, PipelineOrchestrator]:
        """
        Get all orchestrators.
        
        Returns:
            Dictionary mapping plate_id to PipelineOrchestrator
        """
        with self.orchestrators_lock:
            return self.orchestrators.copy()
    
    def get_selected_orchestrators(self, state) -> List[PipelineOrchestrator]:
        """
        Get orchestrators for selected plates from TUI state.
        
        Args:
            state: TUI state object
            
        Returns:
            List of PipelineOrchestrator instances for selected plates
        """
        orchestrators = []
        
        # Handle single selected plate
        if hasattr(state, 'selected_plate') and state.selected_plate:
            plate_id = state.selected_plate.get('id')
            if plate_id:
                orchestrator = self.get_orchestrator(plate_id)
                if orchestrator:
                    orchestrators.append(orchestrator)
        
        # TODO: Handle multiple selected plates if supported
        # if hasattr(state, 'selected_plates') and state.selected_plates:
        #     for plate in state.selected_plates:
        #         plate_id = plate.get('id')
        #         if plate_id:
        #             orchestrator = self.get_orchestrator(plate_id)
        #             if orchestrator:
        #                 orchestrators.append(orchestrator)
        
        return orchestrators
    
    def get_orchestrator_count(self) -> int:
        """Get the total number of orchestrators."""
        with self.orchestrators_lock:
            return len(self.orchestrators)
    
    def get_orchestrator_status(self, plate_id: str) -> str:
        """
        Get the status of an orchestrator.
        
        Args:
            plate_id: Unique identifier for the plate
            
        Returns:
            Status string: 'not_found', 'created', 'initialized', 'compiled', 'running'
        """
        orchestrator = self.get_orchestrator(plate_id)
        if not orchestrator:
            return 'not_found'
        
        # Check orchestrator state
        if hasattr(orchestrator, 'is_initialized') and orchestrator.is_initialized():
            return 'initialized'
        else:
            return 'created'
        
        # TODO: Add compiled and running status checks
        # This would require tracking compilation and execution state
    
    async def shutdown_all(self):
        """Shutdown all orchestrators."""
        logger.info("Shutting down all orchestrators...")
        
        with self.orchestrators_lock:
            for plate_id, orchestrator in self.orchestrators.items():
                try:
                    # TODO: Add proper orchestrator shutdown if needed
                    logger.debug(f"Shutting down orchestrator for plate '{plate_id}'")
                except Exception as e:
                    logger.error(f"Error shutting down orchestrator for plate '{plate_id}': {e}")
            
            self.orchestrators.clear()
        
        logger.info("All orchestrators shut down.")
