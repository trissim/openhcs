"""OpenHCS window factory registration.

Registers all OpenHCS-specific scope patterns with the generic WindowFactory.
Called during application initialization.
"""

import logging
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QWidget
from pyqt_reactive.services.scope_window_factory import (
    ScopeWindowCreationRequest,
    ScopeWindowRegistry,
)
from openhcs.ui.shared.plate_scope_identity import PipelineScopeIdentity
from openhcs.pyqt_gui.services.step_scope_identity import StepEditorScope
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId

if TYPE_CHECKING:
    from openhcs.config_framework.object_state import ObjectState
    from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow

# Import FunctionStep for type checking in tab selection
from openhcs.core.steps.function_step import FunctionStep

logger = logging.getLogger(__name__)


ORCHESTRATOR_SCOPE_PATTERN = r"^/(?!.*::).*$"


def pipeline_scope_window_id(scope_id: str) -> str:
    """Resolve a Pipeline ObjectState scope to the embedded pipeline editor."""
    PipelineScopeIdentity.from_scope_id(scope_id)
    return OpenHCSUiWindowId.pipeline_editor


def global_config_window_scope_id(scope_id: str) -> str:
    """Resolve legacy global config scope requests to the stable window id."""
    return OpenHCSUiWindowId.agent_window_id_for_manager_scope(scope_id)


class OpenHCSWindowCreationAuthority:
    """Nominal owner for OpenHCS scope-to-window construction."""

    def create_global_config_window(
        self,
        request: ScopeWindowCreationRequest,
    ) -> Optional[QWidget]:
        """Create GlobalPipelineConfig editor window."""
        from openhcs.pyqt_gui.windows.config_window import (
            ConfigWindow,
            ConfigWindowTabSpec,
        )
        from openhcs.pyqt_gui.config import UIConfig
        from openhcs.core.config import GlobalPipelineConfig
        from openhcs.core.config_cache import save_global_config_sync
        from openhcs.config_framework.global_config import (
            set_global_config_for_editing,
        )
        from objectstate import ObjectStateRegistry

        plate_manager = self._plate_manager()
        if plate_manager is None:
            raise RuntimeError(
                "GlobalPipelineConfig window creation requires the running PlateManager."
            )
        global_state = ObjectStateRegistry.get_by_scope("")
        if global_state is None:
            raise RuntimeError(
                "Application GlobalPipelineConfig ObjectState is not registered."
            )
        if type(global_state.saved_object) is not GlobalPipelineConfig:
            raise TypeError(
                "Application global ObjectState must contain GlobalPipelineConfig; "
                f"got {type(global_state.saved_object).__name__}."
            )

        ui_scope_id = UIConfig.object_state_scope_id()
        ui_state = ObjectStateRegistry.get_by_scope(ui_scope_id)
        if ui_state is None:
            raise RuntimeError("Application UIConfig ObjectState is not registered.")
        if type(ui_state.saved_object) is not UIConfig:
            raise TypeError(
                "Application UI ObjectState must contain UIConfig; "
                f"got {type(ui_state.saved_object).__name__}."
            )

        def handle_save(new_config: GlobalPipelineConfig) -> None:
            if not self._emit_main_window_global_config_changed(new_config):
                set_global_config_for_editing(GlobalPipelineConfig, new_config)
            if not save_global_config_sync(new_config):
                logger.error("Failed to save global config to cache via window")
            logger.info("Global config saved via window")

        def handle_ui_save(new_config: UIConfig) -> None:
            main_window = self._main_window()
            if main_window is None:
                raise RuntimeError("UIConfig save requires the running main window.")
            main_window.set_ui_config(new_config)
            logger.info("UI config saved via window")

        window = ConfigWindow(
            tabs=(
                ConfigWindowTabSpec(
                    state=global_state,
                    on_save=handle_save,
                    before_mutation=(
                        plate_manager.require_pipeline_definition_mutation_allowed
                    ),
                ),
                ConfigWindowTabSpec(
                    state=ui_state,
                    on_save=handle_ui_save,
                ),
            ),
            scope_id=OpenHCSUiWindowId.canonical_manager_scope_for_agent_window_id(
                request.scope_id
            ),
            title_text="Configure OpenHCS",
        )
        self._show_window(window)
        return window

    def create_plates_root_window(
        self,
        request: ScopeWindowCreationRequest,
    ) -> Optional[QWidget]:
        """Root plate list state - no window to create."""
        del request
        logger.debug("[WINDOW_FACTORY] Skipping window creation for __plates__ scope")
        return None

    def create_plate_config_window(
        self,
        request: ScopeWindowCreationRequest,
    ) -> Optional[QWidget]:
        """Create PipelineConfig editor window for a plate."""
        from openhcs.pyqt_gui.windows.config_window import (
            ConfigWindow,
            ConfigWindowTabSpec,
        )
        from openhcs.core.config import PipelineConfig
        from objectstate import ObjectStateRegistry

        plate_manager = self._plate_manager()
        if plate_manager is None:
            raise RuntimeError(
                "PipelineConfig window creation requires the running PlateManager."
            )
        scope_id = request.scope_id
        state = ObjectStateRegistry.get_by_scope(scope_id)
        if state is None:
            logger.warning("No orchestrator found for scope: %s", scope_id)
            return None
        if not state.has_delegate:
            logger.warning(
                "Scope %s is not an orchestrator config scope; got %s",
                scope_id,
                type(state.object_instance).__name__,
            )
            return None

        pipeline_config = state.saved_object
        if type(pipeline_config) is not PipelineConfig:
            logger.warning(
                "Scope %s delegate is not PipelineConfig; got %s",
                scope_id,
                type(pipeline_config).__name__,
            )
            return None

        window = ConfigWindow(
            tabs=(
                ConfigWindowTabSpec(
                    state=state,
                    before_mutation=(
                        plate_manager.require_pipeline_definition_mutation_allowed
                    ),
                ),
            ),
            scope_id=scope_id,
        )
        self._show_window(window)
        return window

    def create_step_editor_window(
        self,
        request: ScopeWindowCreationRequest,
    ) -> Optional[QWidget]:
        """Create DualEditorWindow for step or function scope."""
        from openhcs.pyqt_gui.windows.dual_editor_window import DualEditorWindow
        from objectstate import ObjectStateRegistry

        scope_id = request.scope_id
        editor_scope = StepEditorScope.parse(scope_id)

        plate_manager = self._plate_manager()
        if plate_manager is None:
            logger.warning("Could not find PlateManager for step editor")
            return None

        orchestrator = ObjectStateRegistry.get_object(editor_scope.plate_scope)
        if not orchestrator:
            logger.warning(
                "No orchestrator found for plate scope: %s",
                editor_scope.plate_scope,
            )
            return None

        step = self._resolve_step(editor_scope, request.object_state)
        if step is None:
            logger.warning("Could not find step for scope: %s", scope_id)
            return None

        window = DualEditorWindow(
            step_data=step,
            is_new=False,
            on_save_callback=None,  # ObjectState handles save
            orchestrator=orchestrator,
            service_adapter=plate_manager.service_adapter,
            step_index=editor_scope.step_token.index,
            plate_scope=editor_scope.plate_scope,
            compiled_artifact_inspection_provider=(
                plate_manager.compiled_artifact_inspection_for_plate
            ),
            before_mutation=(
                plate_manager.require_pipeline_definition_mutation_allowed
            ),
            parent=None,
        )
        window.connect_artifact_signals(
            compiled_artifact_signal=(
                plate_manager.compiled_artifact_inspection_changed
            ),
            runtime_artifact_signal=plate_manager.runtime_artifact_available,
            debug_snapshot_signal=plate_manager.debug_snapshot_available,
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
        if step_state is not None and isinstance(
            step_state.object_instance, FunctionStep
        ):
            return step_state.object_instance

        return self._find_step_by_scope_id(
            editor_scope.plate_scope,
            editor_scope.step_scope_id,
        )

    def _find_step_by_scope_id(
        self,
        plate_scope: str,
        step_scope_id: str,
    ) -> Optional[FunctionStep]:
        """Find a step through the pipeline state's declared step scopes."""
        from objectstate import ObjectStateRegistry

        pipeline_state = ObjectStateRegistry.get_by_scope(
            PipelineScopeIdentity.from_plate_scope(plate_scope).scope_id
        )
        if pipeline_state is None:
            logger.debug("No pipeline state for %s", plate_scope)
            return None

        if "step_scope_ids" not in pipeline_state.parameters:
            step_scope_ids = []
        else:
            step_scope_ids = pipeline_state.parameters["step_scope_ids"]
        if step_scope_id not in step_scope_ids:
            logger.debug(
                "Step scope '%s' not found in %d registered steps",
                step_scope_id,
                len(step_scope_ids),
            )
            return None

        step_state = ObjectStateRegistry.get_by_scope(step_scope_id)
        if step_state is None or not isinstance(
            step_state.object_instance, FunctionStep
        ):
            return None
        return step_state.object_instance

    def _select_step_editor_tab(
        self,
        window: "DualEditorWindow",
        editor_scope: StepEditorScope,
    ) -> None:
        """Select the tab implied by the scope's nominal editor target."""
        from objectstate import ObjectStateRegistry

        state_for_tab_selection = ObjectStateRegistry.get_by_scope(
            editor_scope.scope_id
        )
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

        tab_index = 0
        if select_function_tab:
            tab_index = 1
        window.tab_widget.setCurrentIndex(tab_index)

    def _plate_manager(self):
        from pyqt_reactive.services.service_registry import ServiceRegistry
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        return ServiceRegistry.get(PlateManagerWidget)

    def _main_window(self):
        plate_manager = self._plate_manager()
        if plate_manager is None:
            return None
        return plate_manager.service_adapter.main_window

    def _emit_main_window_global_config_changed(self, new_config) -> bool:
        plate_manager = self._plate_manager()
        if plate_manager is None:
            return False

        plate_manager.service_adapter.main_window.config_changed.emit(new_config)
        return True

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

    # Step/function editors owned by the typed step scope contract.
    ScopeWindowRegistry.register_handler(
        pattern=StepEditorScope.handler_pattern(),
        handler=window_authority.create_step_editor_window,
        window_scope_resolver=StepEditorScope.window_scope_id_for_scope,
        field_path_resolver=StepEditorScope.window_field_path_for_scope,
    )

    ScopeWindowRegistry.register_handler(
        pattern=PipelineScopeIdentity.handler_pattern(),
        window_scope_resolver=pipeline_scope_window_id,
    )

    # Plate configs. Step/function scopes are registered first; every remaining
    # absolute scope must prove it is a delegated PipelineConfig ObjectState in
    # create_plate_config_window().
    ScopeWindowRegistry.register_handler(
        pattern=ORCHESTRATOR_SCOPE_PATTERN,
        handler=window_authority.create_plate_config_window,
    )

    # Plates root list
    ScopeWindowRegistry.register_handler(
        pattern=r"^__plates__$", handler=window_authority.create_plates_root_window
    )

    # Global config (empty string)
    ScopeWindowRegistry.register_handler(
        pattern=r"^$",
        handler=window_authority.create_global_config_window,
        window_scope_resolver=global_config_window_scope_id,
    )
    ScopeWindowRegistry.register_handler(
        pattern=rf"^{OpenHCSUiWindowId.global_config}$",
        handler=window_authority.create_global_config_window,
        window_scope_resolver=global_config_window_scope_id,
    )

    logger.info("[WINDOW_FACTORY] Registered OpenHCS window handlers")
