"""
Dual Editor Window for PyQt6

Step and function editing dialog with tabbed interface.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Optional, Callable, Dict, List, Any

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTabWidget,
    QWidget,
    QStackedWidget,
    QSizePolicy,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QFont, QShowEvent

from openhcs.core.steps.function_step import FunctionStep
from openhcs.constants.constants import GroupBy
from openhcs.core.config import PipelineConfig
from openhcs.config_framework import is_global_config_instance
from openhcs.config_framework.context_manager import config_context
from openhcs.config_framework.global_config import get_current_global_config
from openhcs.config_framework.lazy_factory import get_base_type_for_lazy
from openhcs.ui.shared.pattern_data_manager import PatternDataManager

from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.services.scope_token_service import ScopeTokenService
from pyqt_reactive.services.function_navigation import is_function_field_path
from pyqt_reactive.widgets.function_list_editor import FunctionListEditorWidget
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.widgets.shared import BaseFormDialog, ResponsiveTwoRowWidget
from openhcs.config_framework.object_state import ObjectStateRegistry
from openhcs.introspection import UnifiedParameterAnalyzer
from openhcs.pyqt_gui.widgets.step_parameter_editor import StepParameterEditorWidget

logger = logging.getLogger(__name__)


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
    changes_detected = pyqtSignal(bool)  # has_changes

    def __init__(
        self,
        step_data: Optional[FunctionStep] = None,
        is_new: bool = False,
        on_save_callback: Optional[Callable] = None,
        color_scheme: Optional[ColorScheme] = None,
        orchestrator=None,
        gui_config=None,
        parent=None,
        step_index: Optional[int] = None,
    ):
        """
        Initialize the dual editor window.

        Args:
            step_data: FunctionStep to edit (None for new step)
            is_new: Whether this is a new step
            on_save_callback: Function to call when step is saved
            color_scheme: Color scheme for UI components
            orchestrator: Orchestrator instance for context management
            gui_config: Optional GUI configuration passed from PipelineEditor
            parent: Parent widget
            step_index: Position in pipeline (for border pattern matching list item)
        """
        super().__init__(parent)

        # Store step_index for border pattern (used by ScopedBorderMixin._init_scope_border)
        self._step_index = step_index

        # Make window non-modal (like plate manager and pipeline editor)
        self.setModal(False)

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        self.gui_config = gui_config

        # Business logic state (extracted from Textual version)
        self.is_new = is_new
        self.on_save_callback = on_save_callback
        self.orchestrator = orchestrator  # Store orchestrator for context management

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

        # Change tracking (now uses ObjectState.dirty_fields instead of snapshots)
        self.has_changes = False
        self.current_tab = "step"

        # Change detection callback (ObjectState.on_state_changed expects a no-arg callable)
        self._dirty_detect_callback = lambda: self.detect_changes()

        # UI components
        self.tab_widget: Optional[QTabWidget] = None
        self.parameter_editors: Dict[
            str, QWidget
        ] = {}  # Map tab titles to editor widgets
        self.class_hierarchy: List = []  # Store inheritance hierarchy info

        # Editors are created during setup_ui(); initialize here so scope styling
        # hooks can run during _init_scope_border() without attribute errors.
        self.step_editor = None
        self.func_editor = None

        self._flash_overlay = None  # Window flash overlay for visual feedback
        self._flash_overlay_cleaned = False  # Track if overlay was cleaned up
        self._step_buttons_widget: Optional[QWidget] = (
            None  # Button widget from step editor
        )
        self._func_buttons_widget: Optional[QWidget] = (
            None  # Button widget from func editor
        )

        # Setup UI
        self.setup_ui()
        self.setup_connections()

        # Ensure the initial save button state reflects current ObjectState.
        # setup_ui() may call detect_changes() before changes_detected is connected.
        self.detect_changes()

        # Connect automatic change detection (BaseManagedWindow feature)
        # This automatically calls detect_changes() when any parameter changes
        self._connect_change_detection()

        logger.debug(f"Dual editor window initialized (new={is_new})")

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if not getattr(self, "_default_size_applied", False):
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

    def set_original_step_for_change_detection(self):
        """Set the original step for change detection. Must be called within proper context."""
        # Original step is already set in __init__ when working on a copy
        # This method is kept for compatibility but no longer needed
        pass

    def setup_ui(self):
        """Setup the user interface."""
        # Note: _update_window_title() is called at the end after header_label is created
        if self.size().isEmpty():
            self.resize(550, 600)

        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)

        # Get centralized button styles
        button_styles = self.style_generator.generate_config_button_styles()

        # Title row: title on left, buttons on right (responsive - wraps to row 2 when narrow)
        self._title_header = ResponsiveTwoRowWidget(width_threshold=400, parent=self)
        self._title_header.setStyleSheet(f"""
            QWidget {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border-radius: 3px;
            }}
        """)

        # Title label (left side - stays in row 1)
        self.header_label = QLabel()
        self.header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.header_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)}; background-color: transparent;"
        )
        self._title_header.add_left_widget(self.header_label)

        # Cancel button (right side - wraps to row 2 when narrow)
        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        cancel_button.setMinimumWidth(70)
        cancel_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        cancel_button.clicked.connect(self.cancel_edit)
        cancel_button.setStyleSheet(button_styles["compact"])
        self._title_header.add_right_widget(cancel_button)

        # Save/Create button (right side - wraps to row 2 when narrow)
        self.save_button = QPushButton()
        self._update_save_button_text()
        self.save_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        self.save_button.setMinimumWidth(70)
        self.save_button.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )
        self.save_button.setEnabled(False)
        self._setup_save_button(self.save_button, self.save_edit)
        self._save_button_base_style = (
            button_styles["compact"]
            + f"""
            QPushButton:disabled {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border: none;
            }}
        """
        )
        self.save_button.setStyleSheet(self._save_button_base_style)
        self._title_header.add_right_widget(self.save_button)

        # Connect change detection BEFORE tabs are created.
        # create_step_tab() can call detect_changes() during setup.
        self.changes_detected.connect(self.on_changes_detected)

        layout.addWidget(self._title_header)

        # Tab row: tabs on left, action buttons on right (responsive - buttons wrap when narrow)
        self._tab_row = ResponsiveTwoRowWidget(width_threshold=600, parent=self)

        # Tab widget (left side - stays in row 1)
        self.tab_widget = QTabWidget()
        self.tab_bar = self.tab_widget.tabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setUsesScrollButtons(False)
        self.tab_bar.setFixedHeight(CURRENT_LAYOUT.button_height)
        self._tab_row.add_left_widget(self.tab_bar)

        # Style the tab bar
        self.tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                padding: 0px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: none;
                height: {CURRENT_LAYOUT.button_height}px;
            }}
            QTabBar::tab:selected {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
            }}
            QTabBar::tab:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }}
        """)

        # Container to hold the action buttons from all tabs
        self._active_buttons_container = QWidget()
        self._active_buttons_layout = QHBoxLayout(self._active_buttons_container)
        self._active_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self._active_buttons_layout.setSpacing(0)
        self._tab_row.add_right_widget(self._active_buttons_container)

        layout.addWidget(self._tab_row)

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
            "[DUAL_EDITOR] Set scope_id to: %s, calling _init_scope_border()",
            self.scope_id,
        )
        self._init_scope_border()

        # Create tabs (this adds content to the tab widget)
        self.create_step_tab()
        self.create_function_tab()

        # Editors now exist; apply scope styling to their widget trees.
        self._apply_scope_accent_styling()

        # Add the tab widget's content area (stacked widget) below the tab row
        # The tab bar is already in tab_row, so we only add the content pane here
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Get the stacked widget from the tab widget and add it
        stacked_widget = self.tab_widget.findChild(QStackedWidget)
        if stacked_widget:
            content_layout.addWidget(stacked_widget)

        layout.addWidget(content_container)

        # Connect tab change to swap buttons
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Setup tab button containers (extract buttons from editors once)
        QTimer.singleShot(0, self._setup_tab_button_containers)

        # Debounce timer for function editor synchronization (batches rapid updates)
        self._function_sync_timer = QTimer(self)
        self._function_sync_timer.setSingleShot(True)
        self._function_sync_timer.timeout.connect(self._flush_function_editor_sync)
        self._pending_function_editor_sync = False

        # Update title now that header_label exists
        self._update_window_title()

    def _setup_tab_button_containers(self) -> None:
        """Extract buttons from editors and add to active buttons container.

        This is called ONCE after tabs are created to avoid adding widgets
        to the layout multiple times.
        """
        # Extract step editor buttons widget and add to layout
        self._step_buttons_widget = self.step_editor.get_action_buttons()
        if self._step_buttons_widget:
            self._step_buttons_widget.setVisible(False)
            self._active_buttons_layout.addWidget(self._step_buttons_widget)

        # Extract function editor buttons widget and add to layout
        self._func_buttons_widget = self.func_editor.get_action_buttons()
        if self._func_buttons_widget:
            self._func_buttons_widget.setVisible(False)
            self._active_buttons_layout.addWidget(self._func_buttons_widget)

        # Show buttons for the initially selected tab
        self._show_tab_buttons()

    def _show_tab_buttons(self) -> None:
        """Show only buttons for the currently active tab.

        Toggles visibility of pre-embedded button widgets based on
        which tab is currently selected.
        """
        current_index = self.tab_widget.currentIndex()

        # Hide all button widgets first
        if self._step_buttons_widget:
            self._step_buttons_widget.setVisible(False)
        if self._func_buttons_widget:
            self._func_buttons_widget.setVisible(False)

        # Show only the active tab's buttons
        if current_index == 0 and self._step_buttons_widget:
            self._step_buttons_widget.setVisible(True)
        elif current_index == 1 and self._func_buttons_widget:
            self._func_buttons_widget.setVisible(True)

    def _on_tab_changed(self, index: int) -> None:
        """Handle tab change - show appropriate action buttons."""
        self._show_tab_buttons()

    def _update_window_title(self):
        """Update window title with dirty marker and signature diff underline.

        Two orthogonal visual semantics:
        - Asterisk (*): dirty (resolved_live != resolved_saved)
        - Underline: signature diff (raw != signature default)

        Reads step name from ObjectState for live updates when user edits the name field.
        """
        # Get step name from ObjectState if available, otherwise fall back to editing_step
        step_editor = getattr(self, "step_editor", None)
        if step_editor and step_editor.state:
            # Read name from ObjectState (updates live as user types)
            current_values = step_editor.state.get_current_values()
            step_name = current_values.get("name", "Unnamed")
        else:
            # Fallback to editing_step.name during initial setup
            step_name = (
                getattr(self.editing_step, "name", "Unnamed")
                if self.editing_step
                else "Unnamed"
            )

        base_title = f"{'New' if self.is_new else 'Edit'} Step: {step_name}"
        self._base_window_title = base_title

        # Check dirty state from ObjectState
        is_dirty = bool(
            step_editor and step_editor.state and step_editor.state.dirty_fields
        )
        has_sig_diff = bool(
            step_editor
            and step_editor.state
            and step_editor.state.signature_diff_fields
        )

        title = f"* {base_title}" if is_dirty else base_title
        self.setWindowTitle(title)
        if getattr(self, "header_label", None):
            self.header_label.setText(title)
            # Apply underline for signature diff (independent of dirty)
            font = self.header_label.font()
            font.setUnderline(has_sig_diff)
            self.header_label.setFont(font)

    def _update_save_button_text(self):
        base_text = "Create" if self.is_new else "Save"
        # Add asterisk if there are unsaved changes
        has_changes = self.has_changes
        new_text = f"* {base_text}" if has_changes else base_text
        logger.info(
            f"🔘 Updating save button text: is_new={self.is_new}, has_changes={has_changes} → '{new_text}'"
        )
        self.save_button.setText(new_text)
        if getattr(self, "_save_button_base_style", ""):
            self.save_button.setStyleSheet(self._save_button_base_style)

    def _apply_scope_accent_styling(self) -> None:
        """Apply scope accent color to dual editor window elements.

        Overrides the empty implementation in ScopedBorderMixin to style:
        - Save button
        - Tab bar tabs
        - Window flash overlay
        """
        accent_color = self.get_scope_accent_color()
        if accent_color is None:
            raise RuntimeError(
                "Scope accent color is missing; call _init_scope_border() after setting scope_id"
            )

        # Store for child widgets that need the computed accent.
        self._scope_accent_color = accent_color

        hex_color = accent_color.name()

        # Style Save button with scope accent color
        self._save_button_base_style = f"""
            QPushButton {{
                background-color: {hex_color};
                color: white;
                border: none;
                border-radius: 3px;
                padding: 8px;
            }}
            QPushButton:hover {{
                background-color: {accent_color.lighter(115).name()};
            }}
            QPushButton:disabled {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.border_light)};
                border: none;
            }}
        """
        self.save_button.setStyleSheet(self._save_button_base_style)

        # Style header label with scope accent color
        self.header_label.setStyleSheet(f"color: {hex_color};")

        # Style tab bar with scope accent color
        if self.tab_bar:
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

        # Style step_editor elements
        if self.step_editor:
            # Tree selection
            tree_style = self.get_scope_tree_selection_stylesheet()
            if tree_style and self.step_editor.hierarchy_tree:
                current_style = self.step_editor.hierarchy_tree.styleSheet() or ""
                self.step_editor.hierarchy_tree.setStyleSheet(
                    f"{current_style}\n{tree_style}"
                )

            # "Step Parameters" header label (may be None if render_header=False)
            if self.step_editor.header_label is not None:
                self.step_editor.header_label.setStyleSheet(
                    f"color: {hex_color}; font-weight: bold; font-size: 14px;"
                )

            if self._scope_color_scheme:
                self.step_editor.apply_scope_color_scheme(self._scope_color_scheme)

        # Style func_editor elements (Function Pattern tab)
        if self.func_editor:
            # "Functions" header label (may be None if render_header=False)
            if self.func_editor.header_label is not None:
                self.func_editor.header_label.setStyleSheet(
                    f"color: {hex_color}; font-weight: bold; font-size: 14px;"
                )

            # Apply scope color scheme to function panes (for enableable styling and colors)
            # Use inheritance - _scope_color_scheme is set by ScopedBorderMixin._init_scope_border()
            if self._scope_color_scheme:
                self.func_editor.set_scope_color_scheme(self._scope_color_scheme)

        # Show window flash when dual editor opens
        if self._flash_overlay is None:
            from pyqt_reactive.animation import WindowFlashOverlay

            self._flash_overlay = WindowFlashOverlay(self)
            self._flash_overlay_cleaned = False

        super()._apply_scope_accent_styling()

    def _build_step_scope_id(self) -> str:
        return ScopeTokenService.build_scope_id(
            self.orchestrator.plate_path, self.editing_step
        )

    def create_step_tab(self):
        """Create the step settings tab (using dedicated widget)."""
        # Create step parameter editor widget with proper nested context
        # Step must be nested: GlobalPipelineConfig -> PipelineConfig -> Step
        # CRITICAL: Use hierarchical scope_id to isolate this step editor + its function panes
        scope_id = self._build_step_scope_id()

        with config_context(self.orchestrator.pipeline_config):  # Pipeline level
            with config_context(self.editing_step):  # Step level
                self.step_editor = StepParameterEditorWidget(
                    self.editing_step,
                    service_adapter=None,
                    color_scheme=self.color_scheme,
                    pipeline_config=self.orchestrator.pipeline_config,
                    scope_id=scope_id,  # Same hierarchical scope_id as step editor
                    step_index=self._step_index,  # Align scope styling with pipeline order
                    scope_accent_color=self._scope_accent_color,
                    render_header=False,  # Don't render header - buttons will be managed externally
                    button_style="compact",  # Borderless compact style to match function editor
                )

        # NOTE: parameter_changed connection is now handled automatically by BaseManagedWindow._connect_change_detection()
        # which is called at the end of __init__. This automatically calls detect_changes() when any parameter changes.
        # We still need on_form_parameter_changed() for function editor sync, so connect it here.
        self.step_editor.form_manager.parameter_changed.connect(
            self.on_form_parameter_changed
        )

        # NOTE: context_changed subscription REMOVED - FunctionListEditorWidget now subscribes
        # directly to ObjectState.on_resolved_changed, which is the proper mechanism for
        # reacting to resolved value changes from ANY ancestor (PipelineConfig, GlobalPipelineConfig)

        # Subscribe to state changes for window title updates
        self._dirty_title_callback = self._update_window_title
        self.step_editor.state.on_state_changed(self._dirty_title_callback)
        self.step_editor.state.on_state_changed(self._dirty_detect_callback)

        def _update_title_on_resolved_changed(_: set) -> None:
            self._update_window_title()
            self.detect_changes()

        self.step_editor.state.on_resolved_changed(_update_title_on_resolved_changed)

        # CRITICAL: Set initial title now that ObjectState is available
        # This ensures title shows immediately instead of waiting for first state change
        self._update_window_title()
        self.detect_changes()

        self.tab_widget.addTab(self.step_editor, "Step Settings")

    def create_function_tab(self):
        """Create the function pattern tab (using dedicated widget)."""
        # Convert step func to function list format
        initial_functions = self._convert_step_func_to_list()

        # Create function list editor widget (mirrors Textual TUI)
        # CRITICAL: Pass editing_step for context hierarchy (Function → Step → Pipeline → Global)
        # CRITICAL: Use same hierarchical scope_id as step editor to isolate this step editor + its function panes
        scope_id = self._build_step_scope_id()
        step_name = getattr(self.editing_step, "name", "unknown_step")

        self.func_editor = FunctionListEditorWidget(
            initial_functions=initial_functions,
            context_identifier=step_name,
            service_adapter=None,
            scope_id=scope_id,  # Same hierarchical scope_id as step editor
            render_header=False,
            button_style="compact",  # Borderless compact style for tab row
            scope_index=self._step_index,  # Align scope styling with pipeline order
        )

        # Store main window reference for orchestrator access (find it through parent chain)
        main_window = self._find_main_window()
        if main_window:
            self.func_editor.main_window = main_window

        # SINGLE SOURCE OF TRUTH: Initialize function editor state from step
        self._sync_function_editor_from_step()

        # Restore last selected dict-pattern key (persisted in ObjectState.metadata)
        self.func_editor.apply_selected_pattern_key_from_state()

        # Connect function pattern changes
        # Use DirectConnection to keep execution synchronous within atomic context
        self.func_editor.function_pattern_changed.connect(
            self._on_function_pattern_changed, type=Qt.ConnectionType.DirectConnection
        )

        self.tab_widget.addTab(self.func_editor, "Function Pattern")

    def _on_function_pattern_changed(self):
        """Handle function pattern changes from function editor."""
        # Update step func from function editor - use current_pattern to get full pattern data
        current_pattern = self.func_editor.current_pattern
        logger.debug(
            "[FUNC_PATTERN] current_pattern type=%s value=%r",
            type(current_pattern).__name__,
            current_pattern,
        )

        # CRITICAL FIX: Use fresh imports to avoid unpicklable registry wrappers
        if callable(current_pattern):
            try:
                import importlib

                module = importlib.import_module(current_pattern.__module__)
                current_pattern = getattr(module, current_pattern.__name__)
            except Exception:
                pass  # Use original if refresh fails

        # ATOMIC: Coalesce function parameter change + step func update into single undo
        # Without this, editing a function parameter creates two undo entries:
        # one for the function parameter and one for the step's func pattern
        state = self.step_editor.state if self.step_editor else None
        state_func = (
            state.parameters.get("func")
            if state is not None and "func" in state.parameters
            else None
        )
        step_func = getattr(self.editing_step, "func", None)
        if state_func == current_pattern and step_func == current_pattern:
            logger.debug("[FUNC_PATTERN] Ignoring no-op function pattern update")
            return

        with ObjectStateRegistry.atomic("edit func"):
            if step_func != current_pattern:
                self.editing_step.func = current_pattern

            # CRITICAL: Also update ObjectState so list item preview updates in real-time
            # The step_editor's state tracks 'func' parameter - update it with the new pattern
            if state is not None:
                if "func" in state.parameters and state_func != current_pattern:
                    state.update_parameter("func", current_pattern)
                    logger.debug(
                        f"Updated ObjectState 'func' parameter for real-time preview"
                    )
                    logger.debug(
                        "[FUNC_PATTERN] ObjectState dirty_fields after update: %s",
                        state.dirty_fields,
                    )

        self.detect_changes()
        logger.debug(f"Function pattern changed: {current_pattern}")

    def _get_event_bus(self):
        """Get the global event bus from the service adapter.

        Returns:
            GlobalEventBus instance or None if not found
        """
        try:
            # Navigate up to find main window with service adapter
            current = self.parent()
            while current:
                if current.service_adapter:
                    return current.service_adapter.get_event_bus()
                current = current.parent()

            logger.warning("Could not find service adapter for event bus")
            return None
        except Exception as e:
            logger.error(f"Error getting event bus: {e}")
            return None

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
        # Find our step in the new pipeline by matching scope_id
        # CRITICAL: Use scope_id matching (more robust than object identity)
        # The window's scope_id is "plate_path::functionstep_N", extract the token
        window_scope_id = self.scope_id
        if not window_scope_id:
            return

        # Extract step token from scope_id (e.g., "plate_path::functionstep_3" -> "functionstep_3")
        window_step_token = (
            window_scope_id.split("::")[-1] if "::" in window_scope_id else None
        )
        if not window_step_token:
            return

        # Find matching step by scope token
        updated_step = None
        new_index = None
        for i, step in enumerate(new_pipeline_steps):
            step_token = getattr(step, "_scope_token", None)
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
            self._refresh_scope_border()

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
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

        # CRITICAL: Connect to global event bus for cross-window updates
        # This is the OpenHCS "set and forget" pattern - one connection handles ALL sources
        event_bus = self._get_event_bus()
        if event_bus:
            event_bus.pipeline_changed.connect(self._on_pipeline_changed)
            event_bus.config_changed.connect(self._on_config_changed)
            event_bus.register_window(self)
            logger.debug("Connected to global event bus for cross-window updates")

    def _convert_step_func_to_list(self):
        """Convert step func to initial pattern format for function list editor."""
        if not self.editing_step.func:
            return []

        # Return the step func directly - the function list editor will handle the conversion
        result = self.editing_step.func
        logger.debug(f"🔍 DUAL EDITOR _convert_step_func_to_list: returning {result}")
        return result

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

        # Guard: Only sync if function editor exists
        if self.func_editor is None:
            logger.debug("⏭️  Function editor doesn't exist yet, skipping sync")
            return

        # Trigger a refresh from the editor's authoritative ObjectState context.
        self.func_editor.refresh_from_context()

        logger.debug("✅ Triggered function editor refresh from context")

    def _find_main_window(self):
        """Find the main window through the parent chain."""
        try:
            # Navigate up the parent chain to find OpenHCSMainWindow
            current = self.parent()
            while current:
                # Check if this is a main window (has floating_windows attribute)
                if current.floating_windows and current.service_adapter:
                    logger.debug(f"Found main window: {type(current).__name__}")
                    return current
                current = current.parent()

            logger.warning("Could not find main window in parent chain")
            return None

        except Exception as e:
            logger.error(f"Error finding main window: {e}")
            return None

    def _get_current_plate_from_pipeline_editor(self):
        """Get current plate from pipeline editor (mirrors Textual TUI pattern)."""
        try:
            # Navigate up to find pipeline editor widget
            current = self.parent()
            while current:
                # Check if this is a pipeline editor widget
                try:
                    current.current_plate
                    current.pipeline_steps
                    current_plate = current.current_plate
                    if current_plate:
                        logger.debug(
                            f"Found current plate from pipeline editor: {current_plate}"
                        )
                        return current_plate
                except AttributeError:
                    logger.debug(
                        f"Widget doesn't have current_plate/pipeline_steps attributes"
                    )

                # Check children for pipeline editor widget
                for child in current.findChildren(QWidget):
                    try:
                        child.current_plate
                        child.pipeline_steps
                        current_plate = child.current_plate
                        if current_plate:
                            logger.debug(
                                f"Found current plate from pipeline editor child: {current_plate}"
                            )
                            return current_plate
                    except AttributeError:
                        logger.debug(
                            f"Child widget doesn't have current_plate/pipeline_steps attributes"
                        )

                current = current.parent()

            logger.warning("Could not find current plate from pipeline editor")
            return None

        except Exception as e:
            logger.error(f"Error getting current plate from pipeline editor: {e}")
            return None

    # Old function pane methods removed - now using dedicated FunctionListEditorWidget

    def get_function_info(self) -> str:
        """
        Get function information for display.

        Returns:
            Function information string
        """
        if not self.editing_step.func:
            return "No function assigned"

        func = self.editing_step.func
        func_name = getattr(func, "__name__", "Unknown Function")
        func_module = getattr(func, "__module__", "Unknown Module")

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

    def on_orchestrator_config_changed(self, plate_path: str, effective_config):
        """Handle orchestrator configuration changes for placeholder refresh.

        This is called when the pipeline config is saved and the orchestrator's
        effective config changes. We need to update our stored pipeline_config
        reference and refresh the step editor's placeholders.

        Args:
            plate_path: Path of the plate whose orchestrator config changed
            effective_config: The orchestrator's new effective configuration
        """
        # Only refresh if this is for our orchestrator
        if self.orchestrator and str(self.orchestrator.plate_path) == plate_path:
            logger.debug(
                f"Step editor received orchestrator config change for {plate_path}"
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
        for nested_name, nested_manager in form_manager.nested_managers.items():
            self._update_context_obj_recursively(nested_manager, new_context_obj)

    def on_form_parameter_changed(self, param_name: str, value):
        """Handle form parameter changes directly from form manager.

        SINGLE SOURCE OF TRUTH: Always sync function editor on any parameter change.
        This ensures the function editor stays in sync regardless of which parameter
        changed or how the step structure evolves in the future.

        Handles both top-level parameters (e.g., 'name', 'processing_config') and
        nested parameters from nested forms (e.g., 'group_by' from processing_config form).
        """
        logger.debug(f"🔔 DUAL_EDITOR: on_form_parameter_changed called")
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
        path_parts = param_name.split(".")
        logger.debug(f"  path_parts={path_parts}")

        # Skip the first part if it's the form manager's field_id (type name like "FunctionStep")
        # The path format is: "TypeName.field" or "TypeName.nested.field"
        if len(path_parts) > 1:
            # Remove the type name prefix (e.g., "FunctionStep")
            path_parts = path_parts[1:]
            logger.debug(f"  path_parts after removing type prefix={path_parts}")

        if len(path_parts) == 1:
            # Top-level field (e.g., "name", "func", "processing_config")
            field_name = path_parts[0]

            # CRITICAL FIX: For function parameters, use fresh imports to avoid unpicklable registry wrappers
            if field_name == "func" and callable(value):
                try:
                    import importlib

                    module = importlib.import_module(value.__module__)
                    value = module.__dict__.get(value.__name__)
                except Exception:
                    pass  # Use original if refresh fails

            # CRITICAL FIX: For nested dataclass parameters (like processing_config),
            # don't replace the entire lazy dataclass - instead update individual fields
            # This preserves lazy resolution for fields that weren't changed
            if is_dataclass(value) and not isinstance(value, type):
                logger.debug(
                    f"📦 {field_name} is a nested dataclass, updating fields individually"
                )
                existing_config = self.editing_step
                try:
                    existing_config._resolve_field_value
                    logger.debug(f"✅ {field_name} is lazy, preserving lazy resolution")
                except AttributeError:
                    logger.debug(
                        f"Config doesn't have _resolve_field_value, updating directly"
                    )
                    for field in fields(value):
                        raw_value = object.__getattribute__(value, field.name)
                        object.__setattr__(existing_config, field.name, raw_value)
                        logger.debug(f"    ✏️  Updated {field.name} to {raw_value}")
                else:
                    setattr(self.editing_step, field_name, value)
            else:
                setattr(self.editing_step, field_name, value)
        else:
            # Nested field (e.g., ["processing_config", "group_by"])
            # The nested form manager already updated self.editing_step via _mark_parents_modified
            # We just need to sync the function editor
            logger.debug(f"  🔄 Nested field change: {'.'.join(path_parts)}")
            logger.debug(f"  Nested field already updated by _mark_parents_modified")

        # SINGLE SOURCE OF TRUTH: Always sync function editor from step (batched)
        logger.debug(f"  🔄 Scheduling function editor sync after {param_name} change")
        self._schedule_function_editor_sync()

    def on_tab_changed(self, index: int):
        """Handle tab changes."""
        tab_names = ["step", "function"]
        if 0 <= index < len(tab_names):
            self.current_tab = tab_names[index]
            logger.debug(f"Tab changed to: {self.current_tab}")

    def detect_changes(self):
        """Detect if changes have been made using ObjectState's dirty tracking.

        This replaces the old snapshot-based approach with ObjectState's built-in
        dirty tracking, which automatically detects changes to any parameter
        (including nested fields) by comparing current values to saved baseline.
        """
        # Use ObjectState's dirty tracking instead of custom snapshot comparison
        has_changes = bool(self.state.is_raw_dirty) if self.state else False

        logger.debug(f"🔍 DETECT_CHANGES:")
        logger.debug(
            f"  dirty_fields={self.state.dirty_fields if self.state else 'no state'}"
        )
        logger.debug(f"  has_changes={has_changes}")

        if has_changes != self.has_changes:
            self.has_changes = has_changes
            logger.debug(f"  ✅ Emitting changes_detected({has_changes})")
            self.changes_detected.emit(has_changes)
        else:
            logger.debug(f"  ⏭️  No change in has_changes state")

    def on_changes_detected(self, has_changes: bool):
        """Handle changes detection."""
        # Enable/disable save button based on changes
        self.save_button.setEnabled(has_changes)
        # Update save button text to show asterisk when there are unsaved changes
        self._update_save_button_text()

    def save_edit(self, *, close_window=True):
        """Save the edited step. If close_window is True, close after saving; else, keep open."""
        try:
            # CRITICAL FIX: Sync function pattern from function editor BEFORE collecting form values
            # The function editor doesn't use a form manager, so we need to explicitly sync it
            if self.func_editor:
                current_pattern = self.func_editor.current_pattern

            # CRITICAL FIX: Use fresh imports to avoid unpicklable registry wrappers
            if callable(current_pattern):
                try:
                    import importlib

                    module = importlib.import_module(current_pattern.__module__)
                    current_pattern = module.__dict__.get(current_pattern.__name__)
                except Exception:
                    pass  # Use original if refresh fails

                self.editing_step.func = current_pattern
                logger.debug(f"Synced function pattern before save: {current_pattern}")

            # CRITICAL FIX: Collect current values from all tab states before saving
            # This ensures nested dataclass field values are properly saved to the step object
            for tab_index in range(self.tab_widget.count()):
                tab_widget = self.tab_widget.widget(tab_index)
                if tab_widget and hasattr(tab_widget, "state") and tab_widget.state:
                    # Get current values from this tab's state
                    current_values = tab_widget.state.get_current_values()

                    # Apply values to editing step
                    for param_name, value in current_values.items():
                        setattr(self.editing_step, param_name, value)
                        logger.debug(f"Applied {param_name}={value} to editing step")

            # Validate step
            step_name = self.editing_step.name
            if not step_name or not step_name.strip():
                QMessageBox.warning(
                    self, "Validation Error", "Step name cannot be empty."
                )
                return

            # CRITICAL FIX: For existing steps, apply changes to original step object
            # This ensures the pipeline gets the updated step with the same object identity
            logger.info(
                f"💾 Save: is_new={self.is_new}, original_step_reference={self.original_step_reference is not None}"
            )

            if self.original_step_reference is not None:
                # Editing existing step
                logger.info(
                    f"💾 Editing existing step: {self.original_step_reference.name}"
                )
                self._apply_changes_to_original()
                step_to_save = self.original_step_reference
            else:
                # For new steps, after first save, switch to edit mode
                logger.info(f"💾 Creating new step, switching to edit mode")
                step_to_save = self.editing_step
                self.original_step_reference = self.editing_step
                self.is_new = False
                logger.info(f"💾 Set is_new=False, original_step_reference set")
                self._update_window_title()
                self._update_save_button_text()

            # Emit signals and call callback
            logger.info(
                f"💾 Emitting step_saved signal for: {getattr(step_to_save, 'name', 'Unknown')}"
            )
            self.step_saved.emit(step_to_save)

            if self.on_save_callback:
                logger.info(f"💾 Calling on_save_callback")
                self.on_save_callback(step_to_save)

            # After a successful save, update original_step and detect changes
            # ObjectState.mark_saved() is called by accept() or _mark_saved_and_refresh_all()
            self.original_step = self._clone_step(self.editing_step)

            # UNIFIED: Both paths share same logic, differ only in whether to close window
            if close_window:
                self.accept()  # Marks saved + unregisters + cleans up + closes
            else:
                self._mark_saved_and_refresh_all()  # Marks saved + refreshes, but stays open

            # Detect changes after marking saved (should show no changes now)
            self.detect_changes()

        except Exception as e:
            logger.error(f"Failed to save step: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save step:\n{e}")

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
                self.tab_widget.setCurrentIndex(1)
            self.func_editor.select_and_scroll_to_field(field_path)
            return

        if is_step and self.step_editor:
            if self.tab_widget:
                self.tab_widget.setCurrentIndex(0)
            self.step_editor.select_and_scroll_to_field(field_path)
            return

        if self.func_editor:
            if self.tab_widget:
                self.tab_widget.setCurrentIndex(1)
            self.func_editor.select_and_scroll_to_field(field_path)

    def _apply_changes_to_original(self):
        """Apply all changes from editing_step to original_step_reference."""
        if self.original_step_reference is None:
            return

        # Copy all attributes from editing_step to original_step_reference
        if is_dataclass(self.editing_step):
            # For dataclass steps, copy all field values
            for field in fields(self.editing_step):
                value = getattr(self.editing_step, field.name)
                setattr(self.original_step_reference, field.name, value)
        else:
            # CRITICAL FIX: Use reflection to copy ALL attributes, not just hardcoded list
            # This ensures optional dataclass attributes like step_materialization_config are copied
            for attr_name in dir(self.editing_step):
                # Skip private/magic attributes and methods
                if not attr_name.startswith("_") and not callable(
                    getattr(self.editing_step, attr_name, None)
                ):
                    try:
                        value = getattr(self.editing_step, attr_name)
                        setattr(self.original_step_reference, attr_name, value)
                        logger.debug(f"Copied attribute {attr_name}: {value}")
                    except AttributeError:
                        pass

        logger.debug("Applied changes to original step object")

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

    def reject(self):
        """Handle dialog rejection (Cancel button or Escape key).

        Restores ObjectState to last saved state, undoing all unsaved changes.
        """
        # No confirmation needed - time travel allows recovery of any state

        self.step_cancelled.emit()

        logger.info("🔍 DualEditorWindow: About to call super().reject()")

        # CRITICAL: super().reject() calls state.restore_saved() to undo ALL unsaved changes
        # This restores all parameters (including func) to last saved state automatically
        super().reject()  # BaseFormDialog handles state restoration + unregistration

        # CRITICAL: Trigger global refresh AFTER unregistration so other windows
        # re-collect live context without this cancelled window's values
        logger.info("🔍 DualEditorWindow: About to trigger global refresh")
        ObjectStateRegistry.increment_token()
        logger.info("🔍 DualEditorWindow: Triggered global refresh after cancel")

    def closeEvent(self, event):
        """Handle dialog close event."""
        # No confirmation needed - time travel allows recovery of any state

        # Cleanup tree helper subscriptions to prevent memory leaks
        if self.step_editor is not None:
            self.step_editor.tree_helper.cleanup_subscriptions()
            self.step_editor.state.off_state_changed(self._dirty_title_callback)
            self.step_editor.state.off_state_changed(self._dirty_detect_callback)

        super().closeEvent(event)  # BaseFormDialog handles unregistration

    # No need to override _get_form_managers() - BaseFormDialog automatically
    # discovers all ParameterFormManager instances recursively in the widget tree
