"""OpenHCS window factory registration.

Registers all OpenHCS-specific scope patterns with the generic WindowFactory.
Called during application initialization.
"""

import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QWidget
from pyqt_reactive.services import ScopeWindowRegistry

if TYPE_CHECKING:
    from openhcs.config_framework.object_state import ObjectState
    from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

# Import FunctionStep for type checking in tab selection
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StepEditorScope:
    """Parsed identity for a step/function editor scope."""

    scope_id: str
    plate_path: str
    step_scope_id: str
    is_function_scope: bool

    @classmethod
    def parse(cls, scope_id: str) -> "StepEditorScope":
        parts = scope_id.split("::")
        if len(parts) < 2:
            raise ValueError(f"Invalid step scope_id format: {scope_id}")
        return cls(
            scope_id=scope_id,
            plate_path=parts[0],
            step_scope_id=f"{parts[0]}::{parts[1]}",
            is_function_scope=len(parts) >= 3,
        )


class OpenHCSWindowCreationAuthority:
    """Nominal owner for OpenHCS scope-to-window construction."""

    def create_global_config_window(
        self, scope_id: str, object_state=None
    ) -> Optional[QWidget]:
        """Create GlobalPipelineConfig editor window."""
        del object_state
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
        self._show_window(window)
        return window

    def create_plates_root_window(
        self, scope_id: str, object_state=None
    ) -> Optional[QWidget]:
        """Root plate list state - no window to create."""
        del scope_id, object_state
        logger.debug("[WINDOW_FACTORY] Skipping window creation for __plates__ scope")
        return None

    def create_plate_config_window(
        self, scope_id: str, object_state=None
    ) -> Optional[QWidget]:
        """Create PipelineConfig editor window for a plate."""
        del object_state
        from openhcs.pyqt_gui.windows.config_window import ConfigWindow
        from openhcs.core.config import PipelineConfig
        from objectstate import ObjectStateRegistry

        orchestrator = ObjectStateRegistry.get_object(scope_id)
        if not orchestrator:
            logger.warning("No orchestrator found for scope: %s", scope_id)
            return None

        window = ConfigWindow(
            config_class=PipelineConfig,
            current_config=orchestrator.pipeline_config,
            on_save_callback=None,  # ObjectState handles save
            scope_id=scope_id,
        )
        self._show_window(window)
        return window

    def create_step_editor_window(
        self, scope_id: str, object_state: Optional["ObjectState"] = None
    ) -> Optional[QWidget]:
        """Create DualEditorWindow for step or function scope."""
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
        from objectstate import ObjectStateRegistry

        editor_scope = StepEditorScope.parse(scope_id)

        plate_manager = self._plate_manager()
        if plate_manager is None:
            logger.warning("Could not find PlateManager for step editor")
            return None

        orchestrator = ObjectStateRegistry.get_object(editor_scope.plate_path)
        if not orchestrator:
            logger.warning("No orchestrator found for plate: %s", editor_scope.plate_path)
            return None

        step = self._resolve_step(editor_scope, object_state)
        if step is None:
            logger.warning("Could not find step for scope: %s", scope_id)
            return None

        window = DualEditorWindow(
            step_data=step,
            is_new=False,
            on_save_callback=None,  # ObjectState handles save
            orchestrator=orchestrator,
            service_adapter=plate_manager.service_adapter,
            parent=None,
        )
        self._select_step_editor_tab(window, editor_scope)
        self._show_window(window)
        return window

    def _resolve_step(
        self,
        editor_scope: StepEditorScope,
        object_state: Optional["ObjectState"],
    ) -> Optional[FunctionStep]:
        """Resolve the owning FunctionStep from registered ObjectState scopes."""
        from objectstate import ObjectStateRegistry

        if object_state is not None and not editor_scope.is_function_scope:
            if isinstance(object_state.object_instance, FunctionStep):
                return object_state.object_instance
            return None

        step_state = ObjectStateRegistry.get_by_scope(editor_scope.step_scope_id)
        if step_state is not None and isinstance(step_state.object_instance, FunctionStep):
            return step_state.object_instance

        return self._find_step_by_scope_id(editor_scope.plate_path, editor_scope.step_scope_id)

    def _find_step_by_scope_id(
        self,
        plate_path: str,
        step_scope_id: str,
    ) -> Optional[FunctionStep]:
        """Find a step through the pipeline state's declared step scopes."""
        from objectstate import ObjectStateRegistry

        pipeline_state = ObjectStateRegistry.get_by_scope(f"{plate_path}::pipeline")
        if pipeline_state is None:
            logger.debug("No pipeline state for %s", plate_path)
            return None

        step_scope_ids = pipeline_state.parameters.get("step_scope_ids") or []
        if step_scope_id not in step_scope_ids:
            logger.debug(
                "Step scope '%s' not found in %d registered steps",
                step_scope_id,
                len(step_scope_ids),
            )
            return None

        step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
        if step_state is None or not isinstance(step_state.object_instance, FunctionStep):
            return None
        return step_state.object_instance

    def _select_step_editor_tab(
        self,
        window: "DualEditorWindow",
        editor_scope: StepEditorScope,
    ) -> None:
        """Select the tab implied by the scope's nominal editor target."""
        from objectstate import ObjectStateRegistry

        state_for_tab_selection = ObjectStateRegistry.get_by_scope(editor_scope.scope_id)
        if state_for_tab_selection is not None:
            obj_instance = state_for_tab_selection.object_instance
            select_function_tab = not isinstance(obj_instance, FunctionStep)
            logger.debug(
                "[TAB_SELECT] scope_id=%s, object_type=%s, select_function_tab=%s",
                editor_scope.scope_id,
                type(obj_instance).__name__,
                select_function_tab,
            )
        else:
            select_function_tab = editor_scope.is_function_scope
            logger.debug(
                "[TAB_SELECT] scope_id=%s, select_function_tab=%s from scope identity",
                editor_scope.scope_id,
                select_function_tab,
            )

        window.tab_widget.setCurrentIndex(1 if select_function_tab else 0)

    def _plate_manager(self):
        from pyqt_reactive.services import ServiceRegistry
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        return ServiceRegistry.get(PlateManagerWidget)

    def _show_window(self, window: QWidget) -> None:
        window.show()
        window.raise_()
        window.activateWindow()


def register_openhcs_window_handlers():
    """Register all OpenHCS window handlers with the generic factory.

    Call this during application startup.
    """
    window_authority = OpenHCSWindowCreationAuthority()

    # Order matters - more specific patterns should come first

    # Step/function editors (::functionstep_N or ::functionstep_N::func_M)
    # Note: uses "functionstep" prefix derived from FunctionStep class name
    ScopeWindowRegistry.register_handler(
        pattern=r"^.*::functionstep_\d+(::func_\d+)?$",
        handler=window_authority.create_step_editor_window,
    )

    # Plate configs (/path/to/plate - no :: in scope_id)
    ScopeWindowRegistry.register_handler(
        pattern=r"^/[^:]*$", handler=window_authority.create_plate_config_window
    )

    # Plates root list
    ScopeWindowRegistry.register_handler(
        pattern=r"^__plates__$", handler=window_authority.create_plates_root_window
    )

    # Global config (empty string)
    ScopeWindowRegistry.register_handler(
        pattern=r"^$", handler=window_authority.create_global_config_window
    )

    logger.info("[WINDOW_FACTORY] Registered OpenHCS window handlers")
