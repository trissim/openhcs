"""PyQt debug/test-mode toolbar built on declared debug actions."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from openhcs.agent.ui_bridge_identities import (
    PipelineDebugSessionStateSurfaceIdentityDeclaration,
    PipelineDebugToolbarWidgetIdentity,
)
from openhcs.core.debug import DebugCommand, DebugCommandType
from openhcs.core.execution_state import ManagerExecutionState
from openhcs.pyqt_gui.services.ui_bridge_contracts import (
    UiOwnedStateSurfaceDeclaration,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_session_projection import (
    DebugActionRenderModel,
    DebugSessionPanelText,
    DebugToolbarActionProjector,
    PipelineDebugSessionContext,
)
from openhcs.pyqt_gui.widgets.shared.services.pipeline_debug_actions import (
    DebugActionPlacement,
    DebugToolbarAuxiliaryAction,
    PipelineDebugActionDeclarationBase,
)
from pyqt_reactive.widgets.shared.button_panel import ButtonPanel


class DebugToolbarWidget(QWidget):
    """Compact command surface for bounded debug/test-mode runs."""

    UI_STATE_SURFACE_DECLARATIONS = (
        UiOwnedStateSurfaceDeclaration(
            identity=PipelineDebugSessionStateSurfaceIdentityDeclaration,
            title="Pipeline debug session state",
            payload_schema="openhcs.ui.pipeline_debug_session_state.v1",
            related_action_ids=tuple(
                declaration.action_id()
                for declaration in DebugToolbarActionProjector.declarations()
            ),
        ),
    )
    UI_BRIDGE_WIDGET_IDENTITY = PipelineDebugToolbarWidgetIdentity

    command_requested = pyqtSignal(object)
    runtime_inspection_requested = pyqtSignal()

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        style_generator=None,
    ) -> None:
        super().__init__(parent)
        self.button_panel: ButtonPanel | None = None
        self.primary_panel: ButtonPanel | None = None
        self.session_panel: ButtonPanel | None = None
        self.inspector_panel: ButtonPanel | None = None
        self.phase_label = QLabel("No Plate")
        self.cursor_label = QLabel("Select a plate to debug.")
        self.buttons: dict[DebugCommandType, QPushButton] = {}
        self.auxiliary_buttons: dict[DebugToolbarAuxiliaryAction, QPushButton] = {}
        self.menu_actions: dict[DebugCommandType, QAction] = {}
        self.auxiliary_actions: dict[DebugToolbarAuxiliaryAction, QAction] = {}
        self.runtime_inspection_button: QPushButton | None = None
        self.runtime_inspection_action: QAction | None = None
        self._buttons_by_id: dict[str, QPushButton] = {}
        self._action_models: dict[str, DebugActionRenderModel] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        initial_context = PipelineDebugSessionContext(
            target=None,
            session=None,
            manager_execution_state=ManagerExecutionState.IDLE,
        )
        initial_models = DebugToolbarActionProjector.render_models(initial_context)
        layout.addWidget(self.phase_label)
        layout.addWidget(self.cursor_label)
        self.primary_panel = self._build_panel(
            initial_models,
            DebugActionPlacement.PRIMARY,
            style_generator,
        )
        self.button_panel = self.primary_panel
        self.session_panel = self._build_panel(
            initial_models,
            DebugActionPlacement.SESSION,
            style_generator,
        )
        self.inspector_panel = self._build_panel(
            initial_models,
            DebugActionPlacement.INSPECTOR,
            style_generator,
        )
        layout.addWidget(self.primary_panel)
        layout.addWidget(self.session_panel)
        layout.addWidget(self.inspector_panel)
        self.set_action_models(initial_models)

    def _build_panel(
        self,
        initial_models: tuple[DebugActionRenderModel, ...],
        placement: DebugActionPlacement,
        style_generator,
    ) -> ButtonPanel:
        models = tuple(
            model
            for model in initial_models
            if model.placement is placement
        )
        panel = ButtonPanel(
            button_configs=[
                (model.label, model.action_id, model.tooltip)
                for model in models
            ],
            on_action=self.emit_debug_command,
            style_generator=style_generator,
            parent=self,
        )
        for model in models:
            button = panel.buttons[model.action_id]
            model.declaration.register_widget_button(self, button)
            self._buttons_by_id[model.action_id] = button
        return panel

    def emit_debug_command(self, action_id: str) -> None:
        declaration = PipelineDebugActionDeclarationBase.for_action_id(action_id)
        identity = declaration.require_identity()
        if isinstance(identity, DebugCommandType):
            self.command_requested.emit(DebugCommand(identity))
            return
        if isinstance(identity, DebugToolbarAuxiliaryAction):
            self.runtime_inspection_requested.emit()
            return
        raise TypeError(f"Unsupported debug toolbar action identity: {identity!r}")

    def set_debug_session_context(self, context: PipelineDebugSessionContext) -> None:
        """Project toolbar controls from the current PipelineEditor debug context."""

        panel_text = DebugSessionPanelText.from_context(context)
        self.phase_label.setText(panel_text.title)
        self.cursor_label.setText(panel_text.detail)
        self.set_action_models(DebugToolbarActionProjector.render_models(context))

    def set_action_models(
        self,
        models: tuple[DebugActionRenderModel, ...],
    ) -> None:
        """Apply projected debug action models to the rendered toolbar."""

        self._action_models = {model.action_id: model for model in models}
        for model in models:
            tooltip = self._tooltip_for_model(model)
            button = self._buttons_by_id[model.action_id]
            button.setText(model.label)
            button.setToolTip(tooltip)
            button.setEnabled(model.enabled)

    def command_enabled(self, command_type: DebugCommandType) -> bool:
        model = self._action_models.get(command_type.value)
        if model is None:
            raise ValueError(f"Debug command is not exposed by the toolbar: {command_type}")
        return model.enabled

    def auxiliary_action_enabled(
        self,
        action_type: DebugToolbarAuxiliaryAction,
    ) -> bool:
        model = self._action_models.get(action_type.value)
        if model is None:
            raise ValueError(
                f"Debug auxiliary action is not exposed by the toolbar: {action_type}"
            )
        return model.enabled

    @staticmethod
    def _tooltip_for_model(model: DebugActionRenderModel) -> str:
        if model.disabled_reason is None:
            return model.tooltip
        return f"{model.tooltip}\n\nDisabled: {model.disabled_reason.message}"


__all__ = ("DebugToolbarWidget",)
