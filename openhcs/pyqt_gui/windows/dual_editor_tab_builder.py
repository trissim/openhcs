"""Tab construction for DualEditorWindow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

from objectstate.context_manager import config_context
from PyQt6.QtCore import Qt
from pyqt_reactive.widgets.function_list_editor import FunctionListEditorWidget
from pyqt_reactive.widgets.shared import (
    ActionTabbedWindowBody,
    ActionTabSpec,
    TabLabelDeclarationMixin,
)

from openhcs.pyqt_gui.widgets.artifact_plan_view import (
    ArtifactPlanViewWidget,
)
from openhcs.pyqt_gui.widgets.step_parameter_editor import StepParameterEditorWidget
from openhcs.pyqt_gui.windows.dual_editor_session import (
    DualEditorFunctionPatternController,
    DualEditorSession,
)


class DualEditorTab(TabLabelDeclarationMixin, str, Enum):
    """Dual-editor tabs with declaration-owned public labels."""

    def __new__(cls, value: str, label: str) -> Self:
        member = str.__new__(cls, value)
        member._value_ = value
        member.label = label
        return member

    STEP_SETTINGS = ("step_settings", "Step Settings")
    FUNCTION_PATTERN = ("function_pattern", "Function Pattern")
    ARTIFACTS = ("artifacts", "Artifacts")

    def select(self, body: ActionTabbedWindowBody) -> None:
        """Select this declaration in one live dual-editor body."""

        labels = tuple(
            body.tab_bar.tabText(index) for index in range(body.tab_bar.count())
        )
        body.setCurrentIndex(self.index_in(labels))


@dataclass(frozen=True, slots=True)
class _DualEditorTabBuildContext:
    """All non-layout authorities needed to construct DualEditor tabs."""

    editing_step: Any
    orchestrator: Any
    color_scheme: Any
    scope_id: str
    step_index: int | None
    scope_accent_color: Any
    source_bindings: Any
    source_binding_context: Any
    invocation_badge_provider: Callable[[str, int, Callable], str | None] | None
    main_window: Any
    session: DualEditorSession
    on_form_parameter_changed: Callable[[str, Any], None]
    update_window_title: Callable[[], None]
    detect_changes: Callable[[], None]
    sync_function_editor_from_step: Callable[[], None]
    invalidate_artifact_plan: Callable[[], None]
    compiled_artifact_inspection: Any
    before_mutation: Callable[[], None] | None


@dataclass(frozen=True, slots=True)
class _DualEditorTabs:
    """Widgets and controllers built for a dual editor."""

    step_editor: Any
    func_editor: Any
    artifact_plan_view: ArtifactPlanViewWidget
    function_pattern_controller: DualEditorFunctionPatternController


class _DualEditorTabBuilder:
    """Build and wire the Step, Function Pattern, and Artifacts tabs."""

    def __init__(self, context: _DualEditorTabBuildContext) -> None:
        self.context = context

    def build_into(self, tab_body: ActionTabbedWindowBody) -> _DualEditorTabs:
        step_editor = self._build_step_editor()
        self.context.session.step_editor = step_editor
        self._wire_step_editor(step_editor)
        tab_body.add_tab(
            ActionTabSpec(
                label=DualEditorTab.STEP_SETTINGS.label,
                content=step_editor,
                actions=step_editor.get_action_buttons(),
            )
        )

        func_editor = self._build_function_editor()
        self.context.session.func_editor = func_editor
        function_pattern_controller = self._wire_function_editor(func_editor)
        tab_body.add_tab(
            ActionTabSpec(
                label=DualEditorTab.FUNCTION_PATTERN.label,
                content=func_editor,
                actions=func_editor.get_action_buttons(),
            )
        )

        artifact_plan_view = ArtifactPlanViewWidget(
            inspection=self.context.compiled_artifact_inspection,
            step_index=self.context.step_index,
        )
        tab_body.add_tab(
            ActionTabSpec(
                label=DualEditorTab.ARTIFACTS.label,
                content=artifact_plan_view,
            )
        )

        return _DualEditorTabs(
            step_editor=step_editor,
            func_editor=func_editor,
            artifact_plan_view=artifact_plan_view,
            function_pattern_controller=function_pattern_controller,
        )

    def _build_step_editor(self) -> StepParameterEditorWidget:
        with (
            config_context(self.context.orchestrator.pipeline_config),
            config_context(self.context.editing_step),
        ):
            return StepParameterEditorWidget(
                self.context.editing_step,
                service_adapter=None,
                color_scheme=self.context.color_scheme,
                pipeline_config=self.context.orchestrator.pipeline_config,
                scope_id=self.context.scope_id,
                step_index=self.context.step_index,
                scope_accent_color=self.context.scope_accent_color,
                render_header=False,
                button_style="compact",
                source_bindings=self.context.source_bindings,
                source_binding_context=self.context.source_binding_context,
                before_mutation=self.context.before_mutation,
            )

    def _wire_step_editor(self, step_editor: StepParameterEditorWidget) -> None:
        step_editor.form_manager.parameter_changed.connect(
            self.context.on_form_parameter_changed
        )

        def update_title_on_state_changed(_: set[str]) -> None:
            self.context.update_window_title()
            self.context.detect_changes()

        step_editor.state.on_state_changed(update_title_on_state_changed)

        def update_title_on_resolved_changed(_: set) -> None:
            self.context.update_window_title()
            self.context.detect_changes()

        step_editor.state.on_resolved_changed(update_title_on_resolved_changed)
        self.context.update_window_title()
        self.context.detect_changes()

    def _build_function_editor(self) -> FunctionListEditorWidget:
        return FunctionListEditorWidget(
            initial_functions=self.context.editing_step.func or [],
            context_identifier=self.context.editing_step.name,
            service_adapter=None,
            scope_id=self.context.scope_id,
            render_header=False,
            button_style="compact",
            scope_index=self.context.step_index,
            invocation_badge_provider=self.context.invocation_badge_provider,
            before_mutation=self.context.before_mutation,
        )

    def _wire_function_editor(
        self,
        func_editor: FunctionListEditorWidget,
    ) -> DualEditorFunctionPatternController:
        if self.context.main_window:
            func_editor.main_window = self.context.main_window

        self.context.sync_function_editor_from_step()
        func_editor.apply_selected_pattern_key_from_state()
        controller = DualEditorFunctionPatternController(
            session=self.context.session,
            detect_changes=self.context.detect_changes,
            invalidate_artifact_plan=self.context.invalidate_artifact_plan,
        )
        func_editor.function_pattern_changed.connect(
            controller.handle_change,
            type=Qt.ConnectionType.DirectConnection,
        )
        return controller
