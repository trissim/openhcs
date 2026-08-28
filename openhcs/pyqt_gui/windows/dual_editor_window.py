"""
Dual Editor Window for PyQt6

Step and function editing dialog with tabbed interface.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from collections.abc import Callable
from functools import cache

from objectstate import is_global_config_instance
from objectstate.global_config import get_current_global_config
from objectstate.object_state import ObjectStateRegistry
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtGui import QShowEvent
from PyQt6.QtWidgets import (
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.services.function_navigation import is_function_field_path
from pyqt_reactive.services.scope_token_service import ScopeTokenService
from pyqt_reactive.services.window_code_document import WindowCodeDocumentDriver
from pyqt_reactive.services.window_navigation import (
    FieldWindowNavigationDriver,
    NavigationWaitReason,
    RegisteredWindowNavigationReadiness,
    RegisteredWindowNavigationRequest,
    WindowNavigationDriver,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared import (
    ActionTabbedWindowBody,
    BaseFormDialog,
    DirtyWindowPresentation,
    FormWindowActionHeader,
    HeaderAction,
    HeaderActionGroup,
    ManagedWindowActionCapabilities,
)

from openhcs.core.artifact_inspection import CompiledArtifactInspection
from openhcs.core.config import PipelineConfig
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_bindings import SourceBindingsConfig
from openhcs.core.steps.abstract import AbstractStep
from openhcs.core.steps.function_step import FunctionSpec, FunctionStep
from openhcs.introspection import SignatureAnalyzer
from openhcs.pyqt_gui.services.step_scope_identity import (
    StepEditorScope,
    build_step_scope_id,
)
from openhcs.pyqt_gui.widgets.shared.services.compile_workflow_service import (
    PlateCompiledState,
)
from openhcs.pyqt_gui.widgets.shared.services.debug_progress_service import (
    DebugSnapshotAvailableNotification,
)
from openhcs.pyqt_gui.widgets.shared.services.runtime_artifact_progress_service import (
    RuntimeArtifactAvailableNotification,
)
from openhcs.pyqt_gui.windows.dual_editor_session import (
    DualEditorSession,
)
from openhcs.pyqt_gui.windows.dual_editor_tab_builder import (
    DualEditorTab,
    _DualEditorTabBuildContext,
    _DualEditorTabBuilder,
)
from openhcs.ui.shared.pattern_data_manager import PatternDataManager

logger = logging.getLogger(__name__)


def _nearest_parent_service_adapter(parent_widget):
    """Resolve the nearest manager-owned service adapter."""
    from pyqt_reactive.widgets.shared.abstract_manager_widget import (
        AbstractManagerWidget,
    )

    current = parent_widget
    while current is not None:
        if isinstance(current, AbstractManagerWidget):
            return current.service_adapter
        current = current.parent()
    return None


def _resolve_service_adapter(service_adapter, parent_widget):
    """Resolve the service adapter available to this editor window."""
    return (
        service_adapter
        if service_adapter is not None
        else _nearest_parent_service_adapter(parent_widget)
    )


@cache
def _function_step_form_field_names() -> frozenset[str]:
    """Return editable top-level FunctionStep fields from constructor signatures."""
    return frozenset(
        SignatureAnalyzer.analyze(AbstractStep.__init__).keys()
    ) | frozenset(SignatureAnalyzer.analyze(FunctionStep.__init__).keys())


class DualEditorWindowNavigationDriver(WindowNavigationDriver):
    """Navigate through the child editor that owns the requested field target."""

    def __init__(self, owner: "DualEditorWindow") -> None:
        self.owner = owner
        self.step_navigation_driver: WindowNavigationDriver | None = None
        self.function_navigation_driver: WindowNavigationDriver | None = None

    def readiness(
        self,
        request: RegisteredWindowNavigationRequest,
    ) -> RegisteredWindowNavigationReadiness:
        driver = self.driver_for_field_path(request.field_path)
        if driver is None:
            return RegisteredWindowNavigationReadiness(
                wait_reason=NavigationWaitReason.ROOT_WIDGETS,
            )
        return driver.readiness(request)

    def accepts_field_path(
        self,
        request: RegisteredWindowNavigationRequest,
    ) -> bool:
        driver = self.driver_for_field_path(request.field_path)
        return driver is not None and driver.accepts_field_path(request)

    def prepare(self, request: RegisteredWindowNavigationRequest) -> None:
        self.select_tab_for_field_path(request.field_path)

    def register_readiness_callback(
        self,
        request: RegisteredWindowNavigationRequest,
        callback: Callable[[], None],
    ) -> bool:
        driver = self.driver_for_field_path(request.field_path)
        if driver is None:
            return False
        return driver.register_readiness_callback(request, callback)

    def execute(self, request: RegisteredWindowNavigationRequest) -> None:
        driver = self.driver_for_field_path(request.field_path)
        if driver is None:
            return
        driver.execute(request)

    def driver_for_field_path(
        self,
        field_path: str | None,
    ) -> WindowNavigationDriver | None:
        if field_path is None:
            return None
        if is_function_field_path(field_path):
            func_editor = self.owner.func_editor
            if func_editor is None:
                return None
            if self.function_navigation_driver is None:
                self.function_navigation_driver = FieldWindowNavigationDriver(
                    func_editor.select_and_scroll_to_field
                )
            return self.function_navigation_driver
        step_editor = self.owner.step_editor
        if step_editor is None:
            return None
        if self.step_navigation_driver is None:
            self.step_navigation_driver = step_editor.window_navigation_driver()
        return self.step_navigation_driver

    def select_tab_for_field_path(self, field_path: str | None) -> None:
        tab_widget = self.owner.tab_widget
        if tab_widget is None or field_path is None:
            return
        if is_function_field_path(field_path):
            DualEditorTab.FUNCTION_PATTERN.select(tab_widget)
        else:
            DualEditorTab.STEP_SETTINGS.select(tab_widget)


class DualEditorWindow(BaseFormDialog):
    """
    PyQt6 Multi-Tab Parameter Editor Window.

    Generic parameter editing dialog with inheritance hierarchy-based tabbed interface.
    Creates one tab per class in the inheritance hierarchy, showing parameters specific
    to each class level. Preserves all business logic from Textual version with clean PyQt6 UI.

    Inherits from BaseFormDialog to automatically handle unregistration from
    cross-window placeholder updates when the dialog closes.
    """

    # Signals
    step_saved = pyqtSignal(object)  # FunctionStep
    step_cancelled = pyqtSignal()

    def __init__(
        self,
        step_data: FunctionStep | None = None,
        is_new: bool = False,
        on_save_callback: Callable | None = None,
        color_scheme: ColorScheme | None = None,
        orchestrator=None,
        parent=None,
        service_adapter=None,
        step_index: int | None = None,
        *,
        plate_scope: str,
        source_bindings: SourceBindingsConfig | None = None,
        source_binding_context: SourceBindingContext | None = None,
        function_invocation_badge_provider: (
            Callable[[str, int, Callable], str | None] | None
        ) = None,
        compiled_artifact_inspection_provider: (
            Callable[[str], CompiledArtifactInspection | None] | None
        ) = None,
        before_mutation: Callable[[], None] | None = None,
    ):
        """
        Initialize the dual editor window.

        Args:
            step_data: FunctionStep to edit (None for new step)
            is_new: Whether this is a new step
            on_save_callback: Function to call when step is saved
            color_scheme: Color scheme for UI components
            orchestrator: Orchestrator instance for context management
            parent: Parent widget
            service_adapter: PyQt service adapter that owns main window services
            step_index: Position in pipeline (for border pattern matching list item)
            plate_scope: Logical plate ObjectState scope used for step child scopes
        """
        super().__init__(parent)

        # Store step_index for border pattern (used by ScopedBorderMixin.init_scope_border)
        self._step_index = step_index
        self._function_invocation_badge_provider = function_invocation_badge_provider
        self._compiled_artifact_inspection_provider = (
            compiled_artifact_inspection_provider
        )
        self._before_mutation = before_mutation
        self.service_adapter = _resolve_service_adapter(service_adapter, parent)

        # Make window non-modal (like plate manager and pipeline editor)
        self.setModal(False)

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or ColorScheme()
        self.source_bindings = source_bindings
        self.source_binding_context = source_binding_context

        # Business logic state (extracted from Textual version)
        self.is_new = is_new
        self.on_save_callback = on_save_callback
        self.orchestrator = orchestrator  # Store orchestrator for context management
        self.plate_scope = str(plate_scope)

        # Pattern management (extracted from Textual version)
        self.pattern_manager = PatternDataManager()

        # Store original step reference (never modified)
        # CRITICAL: For new steps, this must be None until first save
        self.original_step_reference = None if is_new else step_data

        if step_data:
            # CRITICAL FIX: Work on a copy to prevent immediate modification of original
            self.editing_step = self._clone_step(step_data)
            self.original_step = self._clone_step(step_data)
        else:
            self.editing_step = self._create_new_step()
            self.original_step = None
        self._session = DualEditorSession(editing_step=self.editing_step)

        self.current_tab = "step"

        # UI components
        self.tab_widget: ActionTabbedWindowBody | None = None
        self.header_label: QLabel | None = None
        self.parameter_editors: dict[
            str, QWidget
        ] = {}  # Map tab titles to editor widgets
        self.class_hierarchy: list = []  # Store inheritance hierarchy info

        # Editors are created during setup_ui(); initialize here so scope styling
        # hooks can run during init_scope_border() without attribute errors.
        self.step_editor = None
        self.func_editor = None
        self.artifact_plan_view = None

        self._flash_overlay = None  # Window flash overlay for visual feedback
        self._flash_overlay_cleaned = False  # Track if overlay was cleaned up
        self._default_size_applied = False
        self._save_button_base_style = ""
        self._function_pattern_controller = None
        self._time_travel_title_refresh_callback = None
        self._event_bus = None
        self._orchestrator_config_signal = None
        self._compiled_artifact_signal = None
        self._runtime_artifact_signal = None
        self._debug_snapshot_signal = None

        # Setup UI
        self.setup_ui()
        self.setup_connections()

        # Ensure the initial save button state reflects current ObjectState.
        # setup_ui() may call detect_changes() before changes_detected is connected.
        self.detect_changes()

        # Connect automatic change detection (BaseManagedWindow feature)
        # This automatically calls detect_changes() when any parameter changes
        self.connect_change_detection()
        self._connect_time_travel_title_refresh()

        logger.debug(f"Dual editor window initialized (new={is_new})")

    def _connect_time_travel_title_refresh(self) -> None:
        """Refresh local dirty/title state after registry time-travel completes."""
        if not self.scope_id:
            return

        def refresh_after_time_travel(_dirty_states, _triggering_scope) -> None:
            QTimer.singleShot(0, self._refresh_after_time_travel)

        ObjectStateRegistry.add_time_travel_complete_callback(refresh_after_time_travel)
        self._time_travel_title_refresh_callback = refresh_after_time_travel

        def cleanup_refresh_callback(*_args) -> None:
            callback = self._time_travel_title_refresh_callback
            if callback is not None:
                ObjectStateRegistry.remove_time_travel_complete_callback(callback)
                self._time_travel_title_refresh_callback = None

        self.destroyed.connect(cleanup_refresh_callback)

    def _refresh_dirty_title_state(self) -> None:
        """Synchronize save-button and title markers from current ObjectState."""
        self.detect_changes()
        self._update_window_title()

    def _refresh_after_time_travel(self) -> None:
        """Synchronize all step-derived tabs from restored ObjectState state."""

        self._sync_function_editor_from_step()
        self._refresh_dirty_title_state()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not self._default_size_applied:
            self.resize(550, 600)
            self._default_size_applied = True
        self._log_window_size("shown")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._log_window_size("resized")

    def _log_window_size(self, context: str) -> None:
        size = self.size()
        logger.debug(
            "Dual editor window %s size=%dx%d pos=%d,%d",
            context,
            size.width(),
            size.height(),
            self.x(),
            self.y(),
        )

    @property
    def state(self):
        """Expose step_editor's ObjectState for BaseManagedWindow compatibility.

        This allows BaseManagedWindow.reject() to find and restore the state
        when the window is cancelled or closed without saving.

        Returns None if step_editor hasn't been created yet.
        """
        if self.step_editor:
            return self.step_editor.state
        return None

    def form_managers(self):
        """Return root form managers for BaseFormDialog change detection."""
        if self.step_editor is None:
            return ()
        return (self.step_editor.form_manager,)

    def window_code_document_driver(self) -> WindowCodeDocumentDriver | None:
        """Expose the active step editor's pycodified code-mode document."""
        step_editor = self.step_editor
        if step_editor is None:
            return None
        return step_editor.code_document_driver()

    def managed_window_action_capabilities(
        self,
    ) -> ManagedWindowActionCapabilities:
        """Expose dual-editor save/cancel semantics to agent actions."""
        return ManagedWindowActionCapabilities(
            save_and_close=True,
            save_without_close=True,
            discard_and_close=True,
        )

    def agent_save_managed_window(self, *, close_window: bool) -> None:
        """Save through the same workflow as the visible Save button."""
        self.save_edit(close_window=close_window)

    def set_original_step_for_change_detection(self):
        """Set the original step for change detection. Must be called within proper context."""
        # Original step is already set in __init__ when working on a copy
        # This method is kept for compatibility but no longer needed

    def setup_ui(self):
        """Setup the user interface."""
        # Note: _update_window_title() is called at the end after header_label is created
        if self.size().isEmpty():
            self.resize(550, 600)

        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Get centralized button styles
        button_styles = self.color_scheme.styles.generate_config_button_styles()

        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        cancel_button.setMinimumWidth(70)
        cancel_button.clicked.connect(self.cancel_edit)
        cancel_button.setStyleSheet(button_styles["compact"])

        self.save_button = QPushButton()
        self._update_save_button_text()
        self.save_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        self.save_button.setMinimumWidth(70)
        self.save_button.setEnabled(False)
        self.setup_save_button(self.save_button, self.save_edit)
        self._save_button_base_style = button_styles["compact"]
        self.save_button.setStyleSheet(self._save_button_base_style)

        self._title_header = FormWindowActionHeader(
            title_text="",
            title_color=self.color_scheme.to_hex(self.color_scheme.text_accent),
            action_groups=[
                HeaderActionGroup(
                    "group_save",
                    [
                        HeaderAction("cancel", cancel_button),
                        HeaderAction("save", self.save_button),
                    ],
                )
            ],
            stay_priority=["group_save"],
            right_aligned_group_ids=["group_save"],
            parent=self,
        )
        self._title_header.setStyleSheet(f"""
            QWidget {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border-radius: 3px;
            }}
        """)
        self.header_label = self._title_header.header_label

        layout.addWidget(self._title_header)

        self._tab_body = ActionTabbedWindowBody(
            color_scheme=self.color_scheme,
            parent=self,
        )
        self.tab_widget = self._tab_body
        self.tab_bar = self._tab_body.tab_bar
        layout.addWidget(self._tab_body, 1)

        # Scope ID for singleton behavior and border styling.
        # Must be initialized BEFORE creating editors so scope accent color is available.
        if self.orchestrator is None:
            raise RuntimeError(
                "DualEditorWindow requires orchestrator to build scope styling"
            )
        if self.editing_step is None:
            raise RuntimeError(
                "DualEditorWindow requires editing_step to build scope styling"
            )
        self.scope_id = self._build_step_scope_id()
        logger.debug(
            "[DUAL_EDITOR] Set scope_id to: %s, calling init_scope_border()",
            self.scope_id,
        )
        self.init_scope_border()

        tabs = _DualEditorTabBuilder(
            _DualEditorTabBuildContext(
                editing_step=self.editing_step,
                orchestrator=self.orchestrator,
                color_scheme=self.color_scheme,
                scope_id=self.scope_id,
                step_index=self._step_index,
                scope_accent_color=self._scope_accent_color,
                source_bindings=self.source_bindings,
                source_binding_context=self.source_binding_context,
                invocation_badge_provider=self._function_invocation_badge_provider,
                main_window=self._find_main_window(),
                session=self._session,
                on_form_parameter_changed=self.on_form_parameter_changed,
                update_window_title=self._update_window_title,
                detect_changes=self.detect_changes,
                sync_function_editor_from_step=self._sync_function_editor_from_step,
                invalidate_artifact_plan=self._invalidate_artifact_plan,
                compiled_artifact_inspection=(
                    self._current_compiled_artifact_inspection()
                ),
                before_mutation=self._before_mutation,
            )
        ).build_into(self._tab_body)
        self.step_editor = tabs.step_editor
        self.func_editor = tabs.func_editor
        self.artifact_plan_view = tabs.artifact_plan_view
        self._function_pattern_controller = tabs.function_pattern_controller

        # Editors now exist; apply scope styling to their widget trees.
        self.apply_scope_accent_styling()

        # Debounce timer for function editor synchronization (batches rapid updates)
        self._function_sync_timer = QTimer(self)
        self._function_sync_timer.setSingleShot(True)
        self._function_sync_timer.timeout.connect(self._flush_function_editor_sync)
        self._pending_function_editor_sync = False

        # Update title now that header_label exists
        self._update_window_title()

    def _numbered_step_title_name(self, step_name: str) -> str:
        """Return the UI title name with pipeline position when available."""
        if self._step_index is None:
            return step_name
        return f"{self._step_index + 1}. {step_name}"

    def _title_step_name(self) -> str:
        """Resolve the required step name used by the editor title."""
        step_editor = self.step_editor
        if step_editor is not None and step_editor.state is not None:
            current_values = step_editor.state.get_current_values()
            if "name" not in current_values:
                raise RuntimeError(
                    "DualEditorWindow step state is missing required 'name'"
                )
            return str(current_values["name"])

        if self.editing_step is None:
            raise RuntimeError(
                "DualEditorWindow cannot build a title without editing_step"
            )

        return str(self.editing_step.name)

    def dirty_window_widgets(self) -> tuple[QLabel, QPushButton] | None:
        if self.header_label is None:
            return None
        return self.header_label, self.save_button

    def dirty_window_presentation(self) -> DirtyWindowPresentation:
        """Build dirty/signature-diff presentation from the current step state."""
        if self.is_new:
            mode = "New"
        else:
            mode = "Edit"

        step_name = self._title_step_name()
        display_name = self._numbered_step_title_name(step_name)
        base_title = f"{mode} Step: {display_name}"
        self._base_window_title = base_title
        return DirtyWindowPresentation(
            window_title=base_title,
            header_text=base_title,
            save_label="Create" if self.is_new else "Save",
            is_dirty=self.dirty_state.is_dirty,
            has_signature_diff=self.dirty_state.has_signature_diff,
            mark_save_label_dirty=True,
        )

    def _apply_dirty_window_presentation(self) -> None:
        self.apply_dirty_window_presentation()

    def _update_window_title(self):
        """Update window title/header/save markers from ObjectState dirtiness."""
        self._apply_dirty_window_presentation()

    def _update_save_button_text(self):
        self._apply_dirty_window_presentation()
        if self._save_button_base_style:
            self.save_button.setStyleSheet(self._save_button_base_style)

    def apply_scope_accent_styling(self) -> None:
        """Apply scope accent color to dual editor window elements.

        Overrides the empty implementation in ScopedBorderMixin to style:
        - Save button
        - Tab bar tabs
        - Window flash overlay
        """
        accent_color = self._required_scope_accent_color()
        self._scope_accent_color = accent_color
        hex_color = accent_color.name()

        self._style_save_button_for_scope(accent_color)
        self._style_header_for_scope(hex_color)
        self._style_tabs_for_scope(hex_color, accent_color)
        self._style_step_editor_for_scope(hex_color)
        self._style_function_editor_for_scope(hex_color)
        self._ensure_window_flash_overlay()

        super().apply_scope_accent_styling()

    def _required_scope_accent_color(self):
        accent_color = self.get_scope_accent_color()
        if accent_color is None:
            raise RuntimeError(
                "Scope accent color is missing; call init_scope_border() after setting scope_id"
            )
        return accent_color

    def _style_save_button_for_scope(self, accent_color) -> None:
        self._save_button_base_style = (
            self.color_scheme.styles.generate_scope_accent_button_style(accent_color)
        )
        self.save_button.setStyleSheet(self._save_button_base_style)

    def _style_header_for_scope(self, hex_color: str) -> None:
        self.header_label.setStyleSheet(f"color: {hex_color};")

    def _style_tabs_for_scope(self, hex_color: str, accent_color) -> None:
        if not self.tab_bar:
            return

        self.tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                padding: 0px 16px;
                margin-right: 2px;
                border: none;
                border-radius: 4px 4px 0 0;
                height: {CURRENT_LAYOUT.button_height}px;
            }}
            QTabBar::tab:selected {{
                background-color: {hex_color};
                color: white;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {accent_color.lighter(115).name()};
                color: white;
            }}
        """)

    def _style_step_editor_for_scope(self, hex_color: str) -> None:
        if not self.step_editor:
            return

        tree_style = self.get_scope_tree_selection_stylesheet()
        if tree_style and self.step_editor.hierarchy_tree:
            current_style = self.step_editor.hierarchy_tree.styleSheet()
            self.step_editor.hierarchy_tree.setStyleSheet(
                f"{current_style}\n{tree_style}"
            )

        if self.step_editor.header_label is not None:
            self.step_editor.header_label.setStyleSheet(
                f"color: {hex_color}; font-weight: bold; font-size: 14px;"
            )

        if self._scope_color_scheme:
            self.step_editor.apply_scope_color_scheme(self._scope_color_scheme)

    def _style_function_editor_for_scope(self, hex_color: str) -> None:
        if not self.func_editor:
            return

        if self.func_editor.header_label is not None:
            self.func_editor.header_label.setStyleSheet(
                f"color: {hex_color}; font-weight: bold; font-size: 14px;"
            )

        if self._scope_color_scheme:
            self.func_editor.set_scope_color_scheme(self._scope_color_scheme)

    def _ensure_window_flash_overlay(self) -> None:
        if self._flash_overlay is None:
            from pyqt_reactive.animation import WindowFlashOverlay

            self._flash_overlay = WindowFlashOverlay.get_for_window(self)
            self._flash_overlay_cleaned = False

    def _build_step_scope_id(self) -> str:
        return build_step_scope_id(self.plate_scope, self.editing_step)

    def _on_pipeline_changed(self, new_pipeline_steps: list):
        """Handle pipeline_changed signal from global event bus.

        CRITICAL: This is connected to the global event bus in setup_connections().
        It receives updates from ANY window that modifies the pipeline:
        - Pipeline editor code button
        - Plate manager code button
        - Pipeline editor UI
        - Any future pipeline editing source

        This is the OpenHCS "set and forget" pattern - one handler receives ALL updates.

        Args:
            new_pipeline_steps: Updated list of FunctionStep objects from the pipeline
        """
        # Find our step in the new pipeline by matching scope token.
        window_scope_id = self.scope_id
        if not window_scope_id:
            return

        try:
            window_step_token = StepEditorScope.parse(window_scope_id).step_token.raw
        except ValueError:
            logger.debug(
                "Ignoring non-step editor scope during pipeline refresh: %s",
                window_scope_id,
            )
            return

        # Find matching step by scope token
        updated_step = None
        new_index = None
        for i, step in enumerate(new_pipeline_steps):
            step_token = ScopeTokenService.object_token(step)
            if step_token == window_step_token:
                updated_step = step
                new_index = i
                break

        # Check if step position changed - refresh scope border styling only
        # (no need to repopulate widgets, just update colors)
        if new_index is not None and new_index != self._step_index:
            logger.debug(
                f"Step moved from index {self._step_index} to {new_index} - refreshing scope border"
            )
            self._step_index = new_index
            if self.step_editor:
                self.step_editor.step_index = new_index
            if self.func_editor:
                self.func_editor.set_scope_index(new_index)
            if self.artifact_plan_view:
                self.artifact_plan_view.set_step_index(new_index)
            self._refresh_scope_border()
            self._update_window_title()

        # Only refresh data if the step OBJECT was replaced in the pipeline
        # (e.g., from code editor saving a new step instance).
        # For simple reorders, updated_step IS original_step_reference, so we skip.
        # NOTE: We never replace editing_step with the pipeline step - editing_step
        # is a clone that preserves isolation for unsaved changes.
        if updated_step and updated_step is not self.original_step_reference:
            logger.debug(
                f"Pipeline step object was replaced - syncing data for: {updated_step.name}"
            )

            # Update reference to the new pipeline step
            self.original_step_reference = updated_step

            # Update function list editor with new func (this recreates panes)
            if self.func_editor and updated_step.func:
                self.func_editor._initialize_pattern_data(updated_step.func)
                self.func_editor._populate_function_list()
            self._invalidate_artifact_plan()

            # Detect changes (might have unsaved changes now)
            self.detect_changes()

    def _on_config_changed(self, config):
        """Handle config_changed signal from global event bus.

        CRITICAL: This is connected to the global event bus in setup_connections().
        It receives updates from ANY window that modifies configs:
        - PlateManager code button (GlobalPipelineConfig, PipelineConfig)
        - ConfigWindow code button (GlobalPipelineConfig, PipelineConfig, StepConfig)
        - Any future config editing source

        This is the OpenHCS "set and forget" pattern - one handler receives ALL updates.

        Args:
            config: Updated config object (GlobalPipelineConfig, PipelineConfig, or StepConfig)
        """
        # Only care about global configs and PipelineConfig changes
        # (StepConfig changes are handled by the step editor's own form manager)
        is_global = is_global_config_instance(config)
        is_pipeline = isinstance(config, PipelineConfig)
        if not (is_global or is_pipeline):
            return

        # Only refresh if this is for our orchestrator
        if not self.orchestrator:
            return

        # Check if this config belongs to our orchestrator
        if is_pipeline:
            # Check if this is our orchestrator's pipeline config
            if config is not self.orchestrator.pipeline_config:
                return
        elif is_global:
            # Check if this is the current global config
            current_global = get_current_global_config(type(config))
            if config is not current_global:
                return

        logger.debug(f"Step editor received config change: {type(config).__name__}")

        # Trigger cross-window refresh for all form managers
        # This will update placeholders in the step editor to show new inherited values
        ObjectStateRegistry.increment_token()
        logger.debug("Triggered global cross-window refresh after config change")

    def setup_connections(self):
        """Setup signal/slot connections."""
        # Tab change tracking
        self._tab_body.current_changed.connect(self.on_tab_changed)

        # CRITICAL: Connect to global event bus for cross-window updates
        # This is the OpenHCS "set and forget" pattern - one connection handles ALL sources
        event_bus = self._get_event_bus()
        if event_bus:
            event_bus.pipeline_changed.connect(self._on_pipeline_changed)
            event_bus.config_changed.connect(self._on_config_changed)
            event_bus.register_window(self)
            self._event_bus = event_bus
            logger.debug("Connected to global event bus for cross-window updates")

    def connect_orchestrator_config_signal(self, signal) -> None:
        """Subscribe this editor to orchestrator config updates."""
        signal.connect(self.on_orchestrator_config_changed)
        self._orchestrator_config_signal = signal

    def connect_artifact_signals(
        self,
        *,
        compiled_artifact_signal,
        runtime_artifact_signal,
        debug_snapshot_signal,
    ) -> None:
        """Subscribe the Artifact tab to typed compile/runtime notifications."""

        compiled_artifact_signal.connect(self._on_compiled_artifact_state_changed)
        runtime_artifact_signal.connect(self._on_runtime_artifact_available)
        debug_snapshot_signal.connect(self._on_debug_snapshot_available)
        self._compiled_artifact_signal = compiled_artifact_signal
        self._runtime_artifact_signal = runtime_artifact_signal
        self._debug_snapshot_signal = debug_snapshot_signal

    def _cleanup_managed_resources(self) -> None:
        """Release editor-owned subscriptions and generic form resources."""
        event_bus = self._event_bus
        if event_bus is not None:
            try:
                event_bus.pipeline_changed.disconnect(self._on_pipeline_changed)
            except TypeError:
                pass
            try:
                event_bus.config_changed.disconnect(self._on_config_changed)
            except TypeError:
                pass
            event_bus.unregister_window(self)
            self._event_bus = None

        orchestrator_signal = self._orchestrator_config_signal
        if orchestrator_signal is not None:
            try:
                orchestrator_signal.disconnect(self.on_orchestrator_config_changed)
            except TypeError:
                pass
            self._orchestrator_config_signal = None

        compiled_artifact_signal = self._compiled_artifact_signal
        if compiled_artifact_signal is not None:
            compiled_artifact_signal.disconnect(
                self._on_compiled_artifact_state_changed
            )
            self._compiled_artifact_signal = None

        runtime_artifact_signal = self._runtime_artifact_signal
        if runtime_artifact_signal is not None:
            runtime_artifact_signal.disconnect(self._on_runtime_artifact_available)
            self._runtime_artifact_signal = None

        debug_snapshot_signal = self._debug_snapshot_signal
        if debug_snapshot_signal is not None:
            debug_snapshot_signal.disconnect(self._on_debug_snapshot_available)
            self._debug_snapshot_signal = None

        super()._cleanup_managed_resources()

    def _schedule_function_editor_sync(self):
        """Schedule a batched sync of the function editor."""
        self._pending_function_editor_sync = True
        if not self._function_sync_timer.isActive():
            self._function_sync_timer.start(0)

    def _flush_function_editor_sync(self):
        """Run any pending function editor sync."""
        if not self._pending_function_editor_sync:
            return
        self._pending_function_editor_sync = False
        self._sync_function_editor_from_step()
        self.detect_changes()

    def _sync_function_editor_from_step(self):
        """
        SINGLE SOURCE OF TRUTH: Sync function editor state from step editor's CURRENT form values.

        CRITICAL: This reads from the form manager's current values (live context), NOT from self.editing_step.
        The form manager's values are the live working copy that updates as the user types.
        self.editing_step only gets updated when the user saves.

        This method extracts all step configuration that affects the function editor
        and updates it. Call this whenever ANY step parameter changes to ensure
        the function editor stays in sync.

        If the step structure changes in the future, only this method needs updating.
        """
        logger.debug("🔄 _sync_function_editor_from_step called")

        if not self._session.sync_function_editor_from_step():
            return

        self._invalidate_artifact_plan()

        logger.debug("✅ Triggered function editor refresh from context")

    def _current_function_spec(self) -> FunctionSpec:
        """Compatibility wrapper for tests; authority lives in DualEditorSession."""
        return self._session.current_function_spec()

    def _invalidate_artifact_plan(self) -> None:
        if self.artifact_plan_view is None:
            return
        self.artifact_plan_view.set_inspection(None)

    def _current_compiled_artifact_inspection(
        self,
    ) -> CompiledArtifactInspection | None:
        provider = self._compiled_artifact_inspection_provider
        if provider is None:
            return None
        inspection = provider(self.plate_scope)
        if inspection is not None and not isinstance(
            inspection,
            CompiledArtifactInspection,
        ):
            raise TypeError(
                "Compiled artifact inspection provider returned "
                f"{type(inspection).__name__}."
            )
        return inspection

    def _on_compiled_artifact_state_changed(
        self,
        plate_path: str,
        state: PlateCompiledState | None,
    ) -> None:
        if plate_path != self.plate_scope:
            return
        if state is not None and not isinstance(state, PlateCompiledState):
            raise TypeError(
                "Compiled artifact state signal requires PlateCompiledState or None, got "
                f"{type(state).__name__}."
            )
        if self.artifact_plan_view is not None:
            self.artifact_plan_view.set_inspection(
                None if state is None else state.inspection
            )

    def _on_runtime_artifact_available(
        self,
        notification: RuntimeArtifactAvailableNotification,
    ) -> None:
        if notification.event.plate_id != self.plate_scope:
            return
        if self.artifact_plan_view is not None:
            self.artifact_plan_view.apply_runtime_notification(notification)

    def _on_debug_snapshot_available(
        self,
        notification: DebugSnapshotAvailableNotification,
    ) -> None:
        if notification.progress_event.plate_id != self.plate_scope:
            return
        if self.artifact_plan_view is not None and notification.snapshot is not None:
            self.artifact_plan_view.apply_debug_snapshot(notification.snapshot)

    def _find_main_window(self):
        """Return the main window from the editor service adapter."""
        if self.service_adapter is None:
            logger.warning("Could not find service adapter for main window")
            return None
        return self.service_adapter.main_window

    def _get_event_bus(self):
        """Return the event bus from the editor service adapter."""
        if self.service_adapter is None:
            logger.warning("Could not find service adapter for event bus")
            return None
        return self.service_adapter.get_event_bus()

    # Old function pane methods removed - now using dedicated FunctionListEditorWidget

    def get_function_info(self) -> str:
        """
        Get function information for display.

        Returns:
            Function information string
        """
        if not self.editing_step.func:
            return "No function assigned"

        func = DualEditorSession.callable_from_function_spec(self.editing_step.func)
        if func is None:
            return "No callable function assigned"

        func_name = func.__name__
        func_module = func.__module__

        info = f"Function: {func_name}\n"
        info += f"Module: {func_module}\n"

        # Add parameter info if available
        if self.editing_step.parameters:
            params = self.editing_step.parameters
            if params:
                info += f"\nParameters ({len(params)}):\n"
                for param_name, param_value in params.items():
                    info += f"  {param_name}: {param_value}\n"

        return info

    def on_orchestrator_config_changed(self, plate_scope: str, effective_config):
        """Handle orchestrator configuration changes for placeholder refresh.

        This is called when the pipeline config is saved and the orchestrator's
        effective config changes. We need to update our stored pipeline_config
        reference and refresh the step editor's placeholders.

        Args:
            plate_scope: Logical plate scope whose orchestrator config changed
            effective_config: The orchestrator's new effective configuration
        """
        # Only refresh if this is for our orchestrator
        if self.orchestrator and self.plate_scope == plate_scope:
            logger.debug(
                f"Step editor received orchestrator config change for {plate_scope}"
            )

            # Update our stored pipeline_config reference to the orchestrator's current config
            self.pipeline_config = self.orchestrator.pipeline_config

            # Update the step editor's pipeline_config reference
            self.step_editor.pipeline_config = self.orchestrator.pipeline_config

            # Update the form manager's context_obj to use the new pipeline config
            if self.step_editor.form_manager:
                # CRITICAL: Update context_obj for root form manager AND all nested managers
                # Nested managers (e.g., processing_config) also have context_obj references that need updating
                self._update_context_obj_recursively(
                    self.step_editor.form_manager, self.orchestrator.pipeline_config
                )

                # Refresh placeholders to show new inherited values
                # Use the same pattern as on_config_changed (line 466)
                ObjectStateRegistry.increment_token()
                logger.debug(
                    "Triggered global cross-window refresh after pipeline config change"
                )

    def _update_context_obj_recursively(self, form_manager, new_context_obj):
        """Recursively update context_obj for a form manager and all its nested managers.

        This is critical for proper placeholder resolution after pipeline config changes.
        When the pipeline config is saved, we get a new PipelineConfig object from the
        orchestrator. We need to update not just the root form manager's context_obj,
        but also all nested managers (processing_config, zarr_config, etc.) so they
        resolve placeholders against the new config.

        Args:
            form_manager: The ParameterFormManager to update
            new_context_obj: The new context object (pipeline_config)
        """
        # Update this manager's context_obj
        form_manager.context_obj = new_context_obj

        # Recursively update all nested managers
        for nested_manager in form_manager.nested_managers.values():
            self._update_context_obj_recursively(nested_manager, new_context_obj)

    def _normalized_form_change_path(self, param_name: str) -> tuple[str, ...]:
        """Return the step-relative form path for a parameter-change signal."""
        path_parts = tuple(part for part in param_name.split(".") if part)
        if len(path_parts) <= 1:
            return path_parts

        if path_parts[0] not in _function_step_form_field_names():
            return path_parts[1:]
        return path_parts

    def on_form_parameter_changed(self, param_name: str, value):
        """Handle form parameter changes directly from form manager.

        SINGLE SOURCE OF TRUTH: Always sync function editor on any parameter change.
        This ensures the function editor stays in sync regardless of which parameter
        changed or how the step structure evolves in the future.

        Handles both top-level parameters (e.g., 'name', 'processing_config') and
        nested parameters from nested forms (e.g., 'group_by' from processing_config form).
        """
        logger.debug("🔔 DUAL_EDITOR: on_form_parameter_changed called")
        logger.debug(f"  param_name={param_name}")
        logger.debug(f"  value type={type(value).__name__}")
        logger.debug(f"  value={repr(value)[:100]}")

        # Handle reset_all completion signal
        if param_name == "__reset_all_complete__":
            logger.debug(
                "🔄 Received reset_all_complete signal, syncing function editor"
            )
            self._schedule_function_editor_sync()
            return

        # param_name is now a full path like "processing_config.group_by" or just "name"
        # Parse the path to determine if it's a nested field
        path_parts = self._normalized_form_change_path(param_name)
        logger.debug(f"  path_parts={path_parts}")

        if not path_parts:
            logger.warning("Received empty form parameter path; syncing editor only")
        elif path_parts == ("func",) and callable(value):
            self._session.apply_function_spec_to_state(value)
        else:
            logger.debug("Form state updated for %s", ".".join(path_parts))

        # SINGLE SOURCE OF TRUTH: Always sync function editor from step (batched)
        logger.debug(f"  🔄 Scheduling function editor sync after {param_name} change")
        self._schedule_function_editor_sync()

    def on_tab_changed(self, index: int):
        """Handle tab changes."""
        tab_names = ["step", "function"]
        if 0 <= index < len(tab_names):
            self.current_tab = tab_names[index]
            logger.debug(f"Tab changed to: {self.current_tab}")

    def save_edit(self, *, close_window=True):
        """Save the edited step. If close_window is True, close after saving; else, keep open."""
        try:
            self.require_managed_state_mutation_allowed()
            # CRITICAL FIX: Sync function pattern from function editor BEFORE collecting form values
            # The function editor doesn't use a form manager, so we need to explicitly sync it
            current_pattern = self._session.apply_function_spec_to_state(
                self._session.current_function_pattern()
            )
            logger.debug("Synced function pattern before save: %r", current_pattern)

            step_to_save = self._session.object_session().to_object(
                update_delegate=False
            )

            # Validate step
            step_name = step_to_save.name
            if not step_name or not step_name.strip():
                QMessageBox.warning(
                    self, "Validation Error", "Step name cannot be empty."
                )
                return

            logger.debug(
                "Save: is_new=%s, original_step_reference=%s",
                self.is_new,
                self.original_step_reference is not None,
            )

            if self.original_step_reference is None:
                # For new steps, after first save, switch to edit mode
                logger.debug("Creating new step, switching to edit mode")
                self.is_new = False
                logger.debug("Set is_new=False")
                self._update_window_title()
                self._update_save_button_text()
            else:
                logger.debug("Saving edited step: %s", step_to_save.name)

            self.original_step_reference = step_to_save
            self.editing_step = step_to_save
            self._session.editing_step = step_to_save

            # Emit signals and call callback
            logger.debug("Emitting step_saved signal for: %s", step_to_save.name)
            self.step_saved.emit(step_to_save)

            if self.on_save_callback:
                logger.debug("Calling on_save_callback")
                self.on_save_callback(step_to_save)

            # After a successful save, update original_step and detect changes
            # ObjectState.mark_saved() is called by accept() or mark_saved_and_refresh_all()
            self.original_step = self._clone_step(step_to_save)

            self.finish_managed_save(close_window=close_window)

        except Exception as error:
            logger.exception("Failed to save step")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save step:\n{error}",
            )

    def select_and_scroll_to_field(self, field_path: str) -> None:
        logger.debug(f"[SCROLL] select_and_scroll_to_field called with: {field_path!r}")
        if not field_path:
            logger.debug("[SCROLL] field_path is falsy, returning early")
            return

        from objectstate import ObjectStateRegistry

        if not self.scope_id:
            return

        state = ObjectStateRegistry.get_by_scope(self.scope_id)
        if not state or state.object_instance is None:
            return

        is_step = isinstance(state.object_instance, FunctionStep)

        # If the navigation target is the function pattern, use the function tab.
        # This avoids fighting tab selection done by time-travel navigation.
        if is_function_field_path(field_path):
            if self.func_editor is None:
                return
            if self.tab_widget:
                DualEditorTab.FUNCTION_PATTERN.select(self.tab_widget)
            self.func_editor.select_and_scroll_to_field(field_path)
            return

        if is_step and self.step_editor:
            if self.tab_widget:
                DualEditorTab.STEP_SETTINGS.select(self.tab_widget)
            self.step_editor.select_and_scroll_to_field(field_path)
            return

        if self.func_editor:
            if self.tab_widget:
                DualEditorTab.FUNCTION_PATTERN.select(self.tab_widget)
            self.func_editor.select_and_scroll_to_field(field_path)

    def window_navigation_driver(self) -> WindowNavigationDriver:
        """Return explicit field navigation behavior for WindowManager."""
        return DualEditorWindowNavigationDriver(self)

    def _clone_step(self, step):
        """Clone a step object using deep copy."""
        import copy

        return copy.deepcopy(step)

    # NOTE: Snapshot-based change detection removed - now using ObjectState.dirty_fields
    # This is simpler, more reliable, and automatically handles nested fields

    def _create_new_step(self):
        """Create a new empty step."""
        return FunctionStep(
            func=[],  # Start with empty function list
            name="New_Step",
        )

    def cancel_edit(self):
        """Cancel editing and close dialog."""
        # Just call reject() - it handles everything including the confirmation dialog
        self.reject()

    def require_managed_state_mutation_allowed(self) -> None:
        if self._before_mutation is not None:
            self._before_mutation()

    def before_managed_reject(self) -> None:
        self.step_cancelled.emit()
        logger.debug("DualEditorWindow: About to call super().reject()")

    def after_managed_reject(self) -> None:
        logger.debug("DualEditorWindow: About to trigger global refresh")
        ObjectStateRegistry.increment_token()
        logger.debug("DualEditorWindow: Triggered global refresh after cancel")

    def before_managed_close(self) -> None:
        # Cleanup tree helper subscriptions to prevent memory leaks
        if self.step_editor is not None:
            self.step_editor.tree_helper.cleanup_subscriptions()

    # No need to override _get_form_managers() - BaseFormDialog automatically
    # discovers all ParameterFormManager instances recursively in the widget tree
