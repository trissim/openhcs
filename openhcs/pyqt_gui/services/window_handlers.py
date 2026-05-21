"""OpenHCS window factory registration.

Registers all OpenHCS-specific scope patterns with the generic WindowFactory.
Called during application initialization.
"""

import logging
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QWidget
from pyqt_reactive.services import ScopeWindowRegistry

if TYPE_CHECKING:
    from openhcs.config_framework.object_state import ObjectState

# Import FunctionStep for type checking in tab selection
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


def _create_global_config_window(scope_id: str, object_state=None) -> Optional[QWidget]:
    """Create GlobalPipelineConfig editor window."""
    from openhcs.pyqt_gui.windows.config_window import ConfigWindow
    from openhcs.core.config import GlobalPipelineConfig
    from openhcs.config_framework.global_config import (
        get_current_global_config,
        set_global_config_for_editing,
    )

    current_config = (
        get_current_global_config(GlobalPipelineConfig) or GlobalPipelineConfig()
    )

    def handle_save(new_config):
        set_global_config_for_editing(GlobalPipelineConfig, new_config)
        logger.info("Global config saved via window")

    window = ConfigWindow(
        config_class=GlobalPipelineConfig,
        current_config=current_config,
        on_save_callback=handle_save,
        scope_id=scope_id,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def _create_plates_root_window(scope_id: str, object_state=None) -> Optional[QWidget]:
    """Root plate list state - no window to create."""
    logger.debug(f"[WINDOW_FACTORY] Skipping window creation for __plates__ scope")
    return None


def _create_plate_config_window(scope_id: str, object_state=None) -> Optional[QWidget]:
    """Create PipelineConfig editor window for a plate."""
    from openhcs.pyqt_gui.windows.config_window import ConfigWindow
    from openhcs.core.config import PipelineConfig
    from pyqt_reactive.services import ServiceRegistry
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
    from objectstate import ObjectStateRegistry

    # Get plate manager from ServiceRegistry
    plate_manager = ServiceRegistry.get(PlateManagerWidget)
    if not plate_manager:
        logger.warning("Could not find PlateManager for plate config window")
        return None

    orchestrator = ObjectStateRegistry.get_object(scope_id)
    if not orchestrator:
        logger.warning(f"No orchestrator found for scope: {scope_id}")
        return None

    window = ConfigWindow(
        config_class=PipelineConfig,
        current_config=orchestrator.pipeline_config,
        on_save_callback=None,  # ObjectState handles save
        scope_id=scope_id,
    )
    window.show()
    window.raise_()
    window.activateWindow()
    return window


def _create_step_editor_window(
    scope_id: str, object_state: Optional["ObjectState"] = None
) -> Optional[QWidget]:
    """Create DualEditorWindow for step or function scope."""
    from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
    from pyqt_reactive.services import ServiceRegistry
    from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
    from objectstate import ObjectStateRegistry

    parts = scope_id.split("::")
    if len(parts) < 2:
        logger.warning(f"Invalid step scope_id format: {scope_id}")
        return None

    plate_path = parts[0]
    step_token = parts[1]
    is_function_scope = len(parts) >= 3
    step_scope_id = f"{parts[0]}::{parts[1]}"

    # Get plate manager from ServiceRegistry
    plate_manager = ServiceRegistry.get(PlateManagerWidget)
    if not plate_manager:
        logger.warning("Could not find PlateManager for step editor")
        return None

    orchestrator = ObjectStateRegistry.get_object(plate_path)
    if not orchestrator:
        logger.warning(f"No orchestrator found for plate: {plate_path}")
        return None

    # Get step: from ObjectState if provided, else find by token
    step = None
    if object_state:
        if is_function_scope:
            step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
            step = step_state.object_instance if step_state else None
        else:
            step = object_state.object_instance

    if not step:
        step = _find_step_by_token(plate_manager, plate_path, step_token)

    if not step:
        logger.warning(f"Could not find step for scope: {scope_id}")
        return None

    window = DualEditorWindow(
        step_data=step,
        is_new=False,
        on_save_callback=None,  # ObjectState handles save
        orchestrator=orchestrator,
        service_adapter=plate_manager.service_adapter,
        parent=None,
    )
    if window.tab_widget:
        # Determine which tab to show based on the type of object being edited
        # Look up ObjectState by scope_id to get the actual object instance type
        state_for_tab_selection = ObjectStateRegistry.get_by_scope(scope_id)
        if state_for_tab_selection:
            obj_instance = state_for_tab_selection.object_instance
            is_function_step = isinstance(obj_instance, FunctionStep)
            logger.debug(
                f"[TAB_SELECT] scope_id={scope_id}, object_type={type(obj_instance).__name__}, "
                f"is_function_step={is_function_step}"
            )
            if is_function_step:
                # Editing a FunctionStep - show Step Settings tab
                logger.debug(
                    f"[TAB_SELECT] Setting Step Settings tab (index 0) - FunctionStep instance"
                )
                window.tab_widget.setCurrentIndex(0)
            else:
                # Editing something else (e.g., function pattern) - show Function Pattern tab
                logger.debug(
                    f"[TAB_SELECT] Setting Function Pattern tab (index 1) - not a FunctionStep"
                )
                window.tab_widget.setCurrentIndex(1)
        else:
            # Fallback: use old logic if we can't look up the state
            logger.debug(
                f"[TAB_SELECT] Fallback: scope_id={scope_id}, is_function_scope={is_function_scope}, "
                f"dirty_fields={getattr(object_state, 'dirty_fields', None)}"
            )
            if is_function_scope:
                logger.debug(
                    f"[TAB_SELECT] Setting Function Pattern tab (index 1) for function scope (fallback)"
                )
                window.tab_widget.setCurrentIndex(1)
            elif object_state and "func" in object_state.dirty_fields:
                logger.debug(
                    f"[TAB_SELECT] Setting Function Pattern tab (index 1) - func in dirty_fields (fallback)"
                )
                window.tab_widget.setCurrentIndex(1)
            else:
                logger.debug(
                    f"[TAB_SELECT] Setting Step Settings tab (index 0) - default (fallback)"
                )
                window.tab_widget.setCurrentIndex(0)

    window.show()
    window.raise_()
    window.activateWindow()
    return window


def _find_step_by_token(plate_manager, plate_path: str, step_token: str):
    """Find a step in the pipeline by its scope token."""
    from objectstate import ObjectStateRegistry

    pipeline_scope = f"{plate_path}::pipeline"
    pipeline_state = ObjectStateRegistry.get_by_scope(pipeline_scope)

    if not pipeline_state:
        logger.debug(f"No pipeline state for {plate_path}")
        return None

    step_scope_ids = pipeline_state.parameters.get("step_scope_ids") or []

    for scope_id in step_scope_ids:
        step_state = ObjectStateRegistry.get_by_scope(scope_id)
        if step_state:
            step = step_state.object_instance
            token = getattr(step, "_scope_token", None)
            if token == step_token:
                return step
            token2 = getattr(step, "_pipeline_scope_token", None)
            if token2 == step_token:
                return step

    logger.debug(f"Step token '{step_token}' not found in {len(step_scope_ids)} steps")
    return None


def register_openhcs_window_handlers():
    """Register all OpenHCS window handlers with the generic factory.

    Call this during application startup.
    """
    # Order matters - more specific patterns should come first

    # Step/function editors (::functionstep_N or ::functionstep_N::func_M)
    # Note: uses "functionstep" prefix derived from FunctionStep class name
    ScopeWindowRegistry.register_handler(
        pattern=r"^.*::functionstep_\d+(::func_\d+)?$",
        handler=_create_step_editor_window,
    )

    # Plate configs (/path/to/plate - no :: in scope_id)
    ScopeWindowRegistry.register_handler(
        pattern=r"^/[^:]*$", handler=_create_plate_config_window
    )

    # Plates root list
    ScopeWindowRegistry.register_handler(
        pattern=r"^__plates__$", handler=_create_plates_root_window
    )

    # Global config (empty string)
    ScopeWindowRegistry.register_handler(
        pattern=r"^$", handler=_create_global_config_window
    )

    logger.info("[WINDOW_FACTORY] Registered OpenHCS window handlers")
