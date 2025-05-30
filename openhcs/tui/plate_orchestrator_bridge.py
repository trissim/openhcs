"""
Plate-Orchestrator Coordination Bridge for OpenHCS TUI.

Coordinates between PlateManagerService orchestrators and OrchestratorManager
to ensure single source of truth while maintaining compatibility with both systems.

This bridge implements the coordination pattern identified in Plan 06, where
PlateManagerService creates orchestrators and OrchestratorManager coordinates them.
"""
import asyncio
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class PlateOrchestratorCoordinationBridge:
    """
    Coordinates between PlateManagerService orchestrators and OrchestratorManager.

    Ensures single source of truth while maintaining compatibility with both systems.
    Implements the event-based coordination pattern from Plan 06.
    """

    def __init__(self, plate_manager_pane, orchestrator_manager, tui_state):
        """
        Initialize the coordination bridge.
        
        Args:
            plate_manager_pane: PlateManagerPane instance
            orchestrator_manager: OrchestratorManager instance  
            tui_state: TUI state object with observer pattern
        """
        self.plate_manager = plate_manager_pane
        self.orchestrator_manager = orchestrator_manager
        self.tui_state = tui_state
        
        # Track initialization state
        self._initialized = False
        
        logger.info("PlateOrchestratorCoordinationBridge created")

    async def initialize(self):
        """
        Initialize the bridge by registering event observers.
        
        This method must be called after instantiation to activate coordination.
        """
        if self._initialized:
            logger.warning("Bridge already initialized, skipping")
            return
            
        try:
            # Register as observer of plate manager events
            self.tui_state.add_observer('plate_added', self.on_plate_added)
            self.tui_state.add_observer('plates_removed', self.on_plates_removed)
            self.tui_state.add_observer('plate_selected', self.on_plate_selected)
            
            # Register for orchestrator status events if they exist
            self.tui_state.add_observer('orchestrator_status_changed', self.on_orchestrator_status_changed)
            
            self._initialized = True
            logger.info("PlateOrchestratorCoordinationBridge initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize coordination bridge: {e}", exc_info=True)
            raise

    async def on_plate_added(self, event_data: Dict[str, Any]):
        """
        Handle plate addition by registering existing orchestrator.
        
        Args:
            event_data: Event data containing plate information
        """
        try:
            plate = event_data.get('plate')
            if not plate:
                logger.warning("plate_added event received without plate data")
                return

            plate_id = plate.get('id')
            plate_path = plate.get('path')
            orchestrator = plate.get('orchestrator')

            if not plate_id:
                logger.warning("plate_added event received without plate_id")
                return

            if orchestrator:
                # Register the EXISTING orchestrator with OrchestratorManager
                # Don't create a new one - just coordinate the existing one
                success = await self.orchestrator_manager.register_existing_orchestrator(
                    plate_id, orchestrator
                )
                
                if success:
                    # Sync status between systems
                    await self._sync_orchestrator_status(plate_id, orchestrator)
                    
                    # Update TUI state with active orchestrator
                    await self._update_active_orchestrator(plate_id, orchestrator)
                    
                    logger.info(f"Successfully coordinated orchestrator for plate {plate_id}")
                else:
                    logger.error(f"Failed to register orchestrator for plate {plate_id}")
            else:
                logger.warning(f"Plate {plate_id} added without orchestrator - coordination skipped")

        except Exception as e:
            logger.error(f"Error coordinating plate addition: {e}", exc_info=True)

    async def on_plates_removed(self, event_data: Dict[str, Any]):
        """
        Handle plate removal by unregistering orchestrators.
        
        Args:
            event_data: Event data containing removed plate IDs
        """
        try:
            plate_ids = event_data.get('plate_ids', [])
            
            if not plate_ids:
                logger.warning("plates_removed event received without plate_ids")
                return

            for plate_id in plate_ids:
                # Remove from OrchestratorManager coordination
                success = await self.orchestrator_manager.unregister_orchestrator(plate_id)
                
                if success:
                    logger.info(f"Successfully unregistered orchestrator for plate {plate_id}")
                else:
                    logger.warning(f"Failed to unregister orchestrator for plate {plate_id}")

        except Exception as e:
            logger.error(f"Error coordinating plate removal: {e}", exc_info=True)

    async def on_plate_selected(self, event_data: Dict[str, Any]):
        """
        Handle plate selection by updating active orchestrator.
        
        Args:
            event_data: Event data containing selected plate information
        """
        try:
            plate = event_data.get('plate')
            if not plate:
                logger.debug("plate_selected event received without plate data")
                return

            plate_id = plate.get('id')
            orchestrator = plate.get('orchestrator')

            if plate_id and orchestrator:
                # Update active orchestrator in TUI state
                await self._update_active_orchestrator(plate_id, orchestrator)
                
                logger.debug(f"Updated active orchestrator for selected plate {plate_id}")

        except Exception as e:
            logger.error(f"Error coordinating plate selection: {e}", exc_info=True)

    async def on_orchestrator_status_changed(self, event_data: Dict[str, Any]):
        """
        Handle orchestrator status changes by updating plate status.
        
        Args:
            event_data: Event data containing orchestrator status information
        """
        try:
            plate_id = event_data.get('plate_id')
            status = event_data.get('status')
            
            if not plate_id or not status:
                logger.warning("orchestrator_status_changed event missing plate_id or status")
                return
                
            # Update PlateManagerService status if it has the method
            if (hasattr(self.plate_manager, 'service') and 
                hasattr(self.plate_manager.service, 'update_plate_status')):
                await self.plate_manager.service.update_plate_status(plate_id, status)
                logger.debug(f"Updated plate status for {plate_id}: {status}")

        except Exception as e:
            logger.error(f"Error handling orchestrator status change: {e}", exc_info=True)

    async def _sync_orchestrator_status(self, plate_id: str, orchestrator):
        """
        Sync orchestrator status between systems.
        
        Args:
            plate_id: Unique identifier for the plate
            orchestrator: PipelineOrchestrator instance
        """
        try:
            # Determine orchestrator status
            if hasattr(orchestrator, 'is_initialized') and callable(orchestrator.is_initialized):
                is_initialized = orchestrator.is_initialized()
            elif hasattr(orchestrator, 'is_initialized'):
                is_initialized = orchestrator.is_initialized
            else:
                is_initialized = False
                
            status = 'initialized' if is_initialized else 'not_initialized'

            # Update PlateManagerService status if possible
            if (hasattr(self.plate_manager, 'service') and 
                hasattr(self.plate_manager.service, 'update_plate_status')):
                await self.plate_manager.service.update_plate_status(plate_id, status)

            # Notify TUI state of status change
            await self.tui_state.notify('orchestrator_status_changed', {
                'plate_id': plate_id,
                'status': status,
                'source': 'PlateOrchestratorCoordinationBridge'
            })
            
            logger.debug(f"Synced orchestrator status for {plate_id}: {status}")

        except Exception as e:
            logger.error(f"Error syncing orchestrator status for {plate_id}: {e}", exc_info=True)

    async def _update_active_orchestrator(self, plate_id: str, orchestrator):
        """
        Update the active orchestrator in TUI state.
        
        Args:
            plate_id: Unique identifier for the plate
            orchestrator: PipelineOrchestrator instance
        """
        try:
            # Set active orchestrator in state
            if hasattr(self.tui_state, 'active_orchestrator'):
                self.tui_state.active_orchestrator = orchestrator
                
            # Notify observers of active orchestrator change
            await self.tui_state.notify('active_orchestrator_changed', {
                'plate_id': plate_id,
                'orchestrator': orchestrator,
                'source': 'PlateOrchestratorCoordinationBridge'
            })
            
            logger.debug(f"Updated active orchestrator for plate {plate_id}")

        except Exception as e:
            logger.error(f"Error updating active orchestrator for {plate_id}: {e}", exc_info=True)

    async def shutdown(self):
        """
        Shutdown the coordination bridge by unregistering observers.
        """
        if not self._initialized:
            return
            
        try:
            # Unregister observers
            if hasattr(self.tui_state, 'remove_observer'):
                self.tui_state.remove_observer('plate_added', self.on_plate_added)
                self.tui_state.remove_observer('plates_removed', self.on_plates_removed)
                self.tui_state.remove_observer('plate_selected', self.on_plate_selected)
                self.tui_state.remove_observer('orchestrator_status_changed', self.on_orchestrator_status_changed)
            
            self._initialized = False
            logger.info("PlateOrchestratorCoordinationBridge shut down successfully")
            
        except Exception as e:
            logger.error(f"Error shutting down coordination bridge: {e}", exc_info=True)
