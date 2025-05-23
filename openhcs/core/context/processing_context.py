"""
Processing Context for OpenHCS.

This module defines the ProcessingContext class, which maintains state during pipeline execution.
"""

from typing import Any, Dict, Optional, Union
from pathlib import Path

from openhcs.core.config import GlobalPipelineConfig, VFSConfig, PathPlanningConfig


class ProcessingContext:
    """
    Maintains state during pipeline execution.

    The ProcessingContext is the canonical owner of all state during pipeline execution.
    After compilation and freezing, it should be treated as immutable by processing steps.

    OWNERSHIP: This class may ONLY be instantiated by PipelineOrchestrator.
    All other components must receive a context instance, never create one.

    Attributes:
        step_plans: Dictionary mapping step IDs to execution plans.
        outputs: Dictionary for step outputs (usage may change with VFS-centric model).
        intermediates: Dictionary for intermediate results (usage may change).
        current_step: Current executing step ID (usage may change).
        well_id: Identifier of the well being processed.
        filemanager: Instance of FileManager for VFS operations.
        global_config: GlobalPipelineConfig holding system-wide configurations.
        _is_frozen: Internal flag indicating if the context is immutable.
    """

    def __init__(
        self,
        global_config: GlobalPipelineConfig, # Made a required argument
        step_plans: Optional[Dict[str, Dict[str, Any]]] = None,
        well_id: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the processing context.

        Args:
            global_config: The global pipeline configuration object.
            step_plans: Dictionary mapping step IDs to execution plans.
            well_id: Identifier of the well being processed.
            **kwargs: Additional context attributes (e.g., filemanager, microscope_handler).
        """
        # Initialize _is_frozen first to allow other attributes to be set by __setattr__
        # This direct assignment bypasses the custom __setattr__ during initialization.
        object.__setattr__(self, '_is_frozen', False)

        self.step_plans = step_plans or {}
        self.outputs = {}  # Future use TBD, primary data flow via VFS
        self.intermediates = {} # Future use TBD, primary data flow via VFS
        self.current_step = None # Future use TBD
        self.well_id = well_id
        self.global_config = global_config # Store the global config
        self.filemanager = None # Expected to be set by Orchestrator via kwargs or direct assignment

        # Add any additional attributes from kwargs
        # Note: 'filemanager' is often passed via kwargs by PipelineOrchestrator.create_context
        for key, value in kwargs.items():
            setattr(self, key, value) # This will now go through our __setattr__

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an attribute, preventing modification if the context is frozen.
        """
        if getattr(self, '_is_frozen', False) and name != '_is_frozen':
            raise AttributeError(f"Cannot modify attribute '{name}' of a frozen ProcessingContext.")
        super().__setattr__(name, value)

    def inject_plan(self, step_id: str, plan: Dict[str, Any]) -> None:
        """
        Inject a step plan into the context.

        This method is the canonical way to add step plans to the context during compilation.
        All step configuration must be injected into the context using this method.

        Args:
            step_id: The unique identifier of the step
            plan: The step execution plan

        Raises:
            AttributeError: If the context is frozen.
        """
        if self._is_frozen:
            raise AttributeError("Cannot inject plan into a frozen ProcessingContext.")
        self.step_plans[step_id] = plan

    def freeze(self) -> None:
        """
        Freezes the context, making its attributes immutable.

        This should be called after all compilation and plan injection is complete.
        Essential attributes like step_plans, filemanager, and well_id must be set.

        Raises:
            RuntimeError: If essential attributes are not set before freezing.
        """
        if not self.well_id:
            raise RuntimeError("Cannot freeze ProcessingContext: 'well_id' is not set.")
        if not hasattr(self, 'filemanager') or self.filemanager is None:
            raise RuntimeError("Cannot freeze ProcessingContext: 'filemanager' is not set.")
        # step_plans can be empty if the pipeline is empty, but it must exist.
        if not hasattr(self, 'step_plans'):
             raise RuntimeError("Cannot freeze ProcessingContext: 'step_plans' attribute does not exist.")

        self._is_frozen = True # This assignment is allowed by __setattr__

    def is_frozen(self) -> bool:
        """
        Check if the context is frozen.

        Returns:
            True if the context is frozen, False otherwise.
        """
        return self._is_frozen

    def get_step_plan(self, step_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the execution plan for a specific step.

        Args:
            step_id: The unique identifier of the step.

        Returns:
            The step plan dictionary if found, else None.
        """
        return self.step_plans.get(step_id)

    # update_from_step_result method is removed as per plan.

    # --- Config Getters ---

    def get_vfs_config(self) -> VFSConfig:
        """Returns the VFSConfig part of the global configuration."""
        if not hasattr(self, 'global_config') or self.global_config is None:
            # This case should ideally not happen if Orchestrator always sets it.
            raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
        return self.global_config.vfs

    def get_path_planning_config(self) -> PathPlanningConfig:
        """Returns the PathPlanningConfig part of the global configuration."""
        if not hasattr(self, 'global_config') or self.global_config is None:
            raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
        return self.global_config.path_planning

    def get_num_workers(self) -> int:
        """Returns the number of workers from the global configuration."""
        if not hasattr(self, 'global_config') or self.global_config is None:
            raise RuntimeError("GlobalPipelineConfig not set on ProcessingContext.")
        return self.global_config.num_workers
