"""
Configuration Window for PyQt6

Configuration editing dialog with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import dataclasses
import os
from typing import Type, Any, Callable, Optional, Dict

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QLabel,
    QScrollArea,
    QWidget,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QShowEvent

# Infrastructure classes removed - functionality migrated to ParameterFormManager service layer
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager, FormManagerConfig
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.widgets.shared.config_hierarchy_tree import (
    ConfigHierarchyTreeHelper,
    ConfigTreeItemKind,
    ConfigTreeItemPayload,
)
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin
from pyqt_reactive.core.collapsible_splitter_helper import CollapsibleSplitterHelper
from pyqt_reactive.widgets.shared.clickable_help_components import HelpButton, HelpContext
from pyqt_reactive.services.parameter_ops_service import ParameterOpsService
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared import (
    BaseFormDialog,
    DirtyWindowPresentation,
    DirtyWindowPresenter,
    FormWindowActionHeader,
    HeaderAction,
    HeaderActionGroup,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.pyqt_gui.windows.config_edit_session import ConfigEditSession
import openhcs.serialization.pycodify_formatters  # noqa: F401
from pycodify import Assignment, generate_python_source
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry

# ❌ REMOVED: require_config_context decorator - enhanced decorator events system handles context automatically
from openhcs.core.lazy_placeholder import (
    LazyDefaultPlaceholderService as FullLazyDefaultPlaceholderService,
)
from openhcs.core.lazy_placeholder_simplified import (
    LazyDefaultPlaceholderService as SimplifiedLazyDefaultPlaceholderService,
)


logger = logging.getLogger(__name__)


class ConfigWindowStateResolver:
    """Resolve the ObjectState a config window is allowed to edit."""

    def __init__(
        self,
        config_class: Type,
        current_config: Any,
        scope_id: Optional[str],
    ) -> None:
        self.config_class = config_class
        self.current_config = current_config
        self.scope_id = scope_id

    def resolve(self) -> ObjectState:
        state = ObjectStateRegistry.get_by_scope(self.scope_id)
        if self.config_class is PipelineConfig:
            return self._required_pipeline_config_state(state)

        if state is not None:
            return state
        return ObjectState(
            object_instance=self.current_config,
            scope_id=self.scope_id,
        )

    def _required_pipeline_config_state(
        self,
        state: ObjectState | None,
    ) -> ObjectState:
        if self.scope_id in (None, ""):
            raise RuntimeError(
                "PipelineConfig editor requires a non-empty orchestrator scope."
            )
        if state is None:
            raise RuntimeError(
                "PipelineConfig editor requires an existing orchestrator ObjectState "
                f"for scope {self.scope_id!r}; refusing to create a standalone "
                "PipelineConfig state."
            )
        if not state.has_delegate:
            raise RuntimeError(
                "PipelineConfig editor scope must resolve to an orchestrator "
                f"ObjectState delegated to pipeline_config; got "
                f"{type(state.object_instance).__name__} at scope {self.scope_id!r}."
            )
        if not isinstance(state.saved_object, PipelineConfig):
            raise RuntimeError(
                "PipelineConfig editor delegate must be a PipelineConfig; got "
                f"{type(state.saved_object).__name__} at scope {self.scope_id!r}."
            )
        return state


class ConfigWidgetValueProjection:
    """Project nullable config values into concrete Qt widget values."""

    EMPTY_SPIN_VALUE = 0
    EMPTY_DOUBLE_SPIN_VALUE = 0.0
    EMPTY_LINE_EDIT_TEXT = ""

    @classmethod
    def checkbox_value(cls, value: Any) -> bool:
        return bool(value)

    @classmethod
    def spin_box_value(cls, value: Any) -> int:
        if value is None:
            return cls.EMPTY_SPIN_VALUE
        return int(value)

    @classmethod
    def double_spin_box_value(cls, value: Any) -> float:
        if value is None:
            return cls.EMPTY_DOUBLE_SPIN_VALUE
        return float(value)

    @classmethod
    def line_edit_text(cls, value: Any) -> str:
        if value is None:
            return cls.EMPTY_LINE_EDIT_TEXT
        return str(value)


def _handle_dataclass_tree_item(
    window: "ConfigWindow",
    payload: ConfigTreeItemPayload,
) -> None:
    if payload.navigation_path:
        window._scroll_to_section(payload.navigation_path)
        logger.debug("Navigating to section: %s", payload.navigation_path)
        return

    class_name = payload.class_obj.__name__ if payload.class_obj else "Unknown"
    logger.debug("Double-clicked on root dataclass: %s", class_name)


def _handle_inheritance_link_tree_item(
    window: "ConfigWindow",
    payload: ConfigTreeItemPayload,
) -> None:
    target_class = payload.target_class
    if target_class is None:
        return

    field_name = window._find_field_for_class(target_class)
    if field_name:
        window._scroll_to_section(field_name)
        logger.debug(
            "Navigating to inherited section: %s (class: %s)",
            field_name,
            target_class.__name__,
        )
        return

    logger.warning("Could not find field for class %s", target_class.__name__)


CONFIG_TREE_ITEM_HANDLERS: dict[
    ConfigTreeItemKind,
    Callable[["ConfigWindow", ConfigTreeItemPayload], None],
] = {
    ConfigTreeItemKind.DATACLASS: _handle_dataclass_tree_item,
    ConfigTreeItemKind.INHERITANCE_LINK: _handle_inheritance_link_tree_item,
}


# Infrastructure classes removed - functionality migrated to ParameterFormManager service layer


class ConfigWindow(ScrollableFormMixin, BaseFormDialog):
    """
    PyQt6 Configuration Window.

    Configuration editing dialog with parameter forms and validation.
    Preserves all business logic from Textual version with clean PyQt6 UI.

    Inherits from BaseFormDialog to automatically handle unregistration from
    cross-window placeholder updates when the dialog closes.

    Inherits from ScrollableFormMixin to provide scroll-to-section functionality.

    Tree items flash via form_manager's FlashMixin - ONE source of truth.
    """

    # Signals
    config_saved = pyqtSignal(object)  # saved config
    config_cancelled = pyqtSignal()
    changes_detected = pyqtSignal(bool)  # has_changes

    def __init__(
        self,
        config_class: Type,
        current_config: Any,
        on_save_callback: Optional[Callable] = None,
        color_scheme: Optional[ColorScheme] = None,
        parent=None,
        scope_id: Optional[str] = None,
    ):
        """
        Initialize the configuration window.

        Args:
            config_class: Configuration class type
            current_config: Current configuration instance
            on_save_callback: Function to call when config is saved
            color_scheme: Color scheme for styling (optional, uses default if None)
            parent: Parent widget
            scope_id: Optional scope identifier (e.g., plate_path) to limit cross-window updates to same orchestrator
        """
        super().__init__(parent)

        # Business logic state (extracted from Textual version)
        self.config_class = config_class
        self.current_config = current_config
        self.on_save_callback = on_save_callback
        self.scope_id = scope_id  # Store scope_id for passing to form_manager

        # Flag to prevent refresh during save operation
        self.restore_descendants_on_close = True

        # Change tracking
        self.has_changes = False

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)
        self._dirty_presenter = DirtyWindowPresenter()
        self.tree_helper = ConfigHierarchyTreeHelper()

        # NOTE: init_scope_border() will be called AFTER setup_ui() creates the widgets
        # This ensures widgets exist when apply_scope_accent_styling() tries to style them
        self._scope_accent_color = None
        self._header_label: QLabel | None = None
        self._save_button: QPushButton | None = None
        self._help_btn: HelpButton | None = None
        self.tree_widget: QTreeWidget | None = None
        self.form_manager: ParameterFormManager | None = None
        self._default_size_applied = False

        # SIMPLIFIED: Use dual-axis resolution
        # Determine placeholder prefix based on actual instance type (not class type)
        is_lazy_dataclass = FullLazyDefaultPlaceholderService.has_lazy_resolution(
            type(current_config)
        )
        placeholder_prefix = "Pipeline default" if is_lazy_dataclass else "Default"

        # SIMPLIFIED: Use ParameterFormManager with dual-axis resolution
        root_field_id = type(
            current_config
        ).__name__  # e.g., "GlobalPipelineConfig" or "PipelineConfig"
        global_config_type = GlobalPipelineConfig  # Always use GlobalPipelineConfig for dual-axis resolution

        # CRITICAL FIX: Pipeline Config Editor should NOT use itself as parent context
        # context_obj=None means inherit from thread-local GlobalPipelineConfig only
        # The overlay (current form state) will be built by ParameterFormManager
        # This fixes the circular context bug where reset showed old values instead of global defaults

        self.state = ConfigWindowStateResolver(
            config_class=self.config_class,
            current_config=current_config,
            scope_id=self.scope_id,
        ).resolve()

        # When editing per-orchestrator PipelineConfig we typically reuse the orchestrator's
        # ObjectState (delegated to pipeline_config) under the plate scope_id.
        # On Cancel/close we want to restore the PipelineConfig fields, but NOT restore
        # descendant step ObjectStates (which can clear the visible pipeline).
        if (
            self.config_class is PipelineConfig
            and self.scope_id not in (None, "")
            and self.state.has_delegate
        ):
            self.restore_descendants_on_close = False

        self._config_session = ConfigEditSession(
            config_class=self.config_class,
            state=self.state,
            original_config=current_config,
        )

        # CRITICAL: Config window manages its own scroll area, so tell form_manager NOT to create one
        config = FormManagerConfig(
            parent=None,
            scope_id=self.scope_id,
            color_scheme=self.color_scheme,
            scope_accent_color=self._scope_accent_color,
        )
        # Provide canonical dotted `field_id` for this root form
        # Root forms use an empty `field_id` (top-level) so no traversal is attempted
        config.field_id = ""
        self.form_manager = ParameterFormManager(state=self.state, config=config)

        if self._config_session.is_global_config:
            self.form_manager.parameter_changed.connect(
                self._on_global_config_field_changed
            )

        # No config_editor needed - everything goes through form_manager
        self.config_editor = None

        # Subscribe to dirty state changes for window title updates
        self._base_window_title = f"Configuration - {self.config_class.__name__}"
        self._dirty_title_callback = self._update_window_title_dirty_marker
        self.state.on_state_changed(self._dirty_title_callback)

        # Change detection
        self.changes_detected.connect(self.on_changes_detected)

        # Setup UI
        self.setup_ui()

        # Connect automatic change detection (BaseManagedWindow feature)
        # This automatically calls detect_changes() when any parameter changes
        self.connect_change_detection()

        # Initialize save button state
        self.detect_changes()

        logger.debug(f"Config window initialized for {config_class.__name__}")

    def form_managers(self) -> tuple[ParameterFormManager, ...]:
        """Return root form managers for BaseFormDialog change detection."""
        return (self.form_manager,) if self.form_manager is not None else ()

    def _update_window_title_dirty_marker(self) -> None:
        """Update window title with dirty marker and signature diff underline.

        Two orthogonal visual semantics:
        - Asterisk (*): unsaved raw edits (parameters != saved_parameters)
        - Underline: signature diff (raw != signature default)
        """
        is_dirty = bool(self.state.is_raw_dirty)
        has_sig_diff = bool(self.state.signature_diff_fields)
        if self._header_label is None or self._save_button is None:
            return

        self._dirty_presenter.apply(
            window=self,
            header_label=self._header_label,
            save_button=self._save_button,
            presentation=DirtyWindowPresentation(
                window_title=self._base_window_title,
                header_text=f"Configure {self.config_class.__name__}",
                save_label="Save",
                is_dirty=is_dirty,
                has_signature_diff=has_sig_diff,
                mark_save_label_dirty=False,
            ),
        )

    def detect_changes(self):
        """Detect if changes have been made using ObjectState's dirty tracking.

        This replaces old snapshot-based approach with ObjectState's built-in
        dirty tracking, which automatically detects changes to any parameter
        (including nested fields) by comparing current values to saved baseline.
        """
        # Use ObjectState's dirty tracking instead of custom snapshot comparison
        has_changes = bool(self.state.is_raw_dirty) if self.state else False

        if has_changes != self.has_changes:
            self.has_changes = has_changes
            self.changes_detected.emit(has_changes)

    def on_changes_detected(self, has_changes: bool):
        """Handle changes detection."""
        del has_changes
        self._update_window_title_dirty_marker()

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(self._base_window_title)
        self.setModal(False)  # Non-modal like plate manager and pipeline editor
        if self.size().isEmpty():
            self.resize(550, 600)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        button_styles = self.style_generator.generate_config_button_styles()
        title_text = f"Configure {self.config_class.__name__}"

        reset_button = QPushButton("Reset to Defaults")
        reset_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        reset_button.setMinimumWidth(100)
        reset_button.clicked.connect(self.reset_to_defaults)
        reset_button.setStyleSheet(button_styles["compact"])

        view_code_button = QPushButton("View Code")
        view_code_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        view_code_button.setMinimumWidth(80)
        view_code_button.clicked.connect(self._view_code)
        view_code_button.setStyleSheet(button_styles["compact"])

        help_actions = []
        if dataclasses.is_dataclass(self.config_class):
            self._help_btn = HelpButton(
                help_context=HelpContext(
                    help_target=self.config_class,
                    color_scheme=self.color_scheme,
                    scope_accent_color=self._scope_accent_color,
                ),
                text="Help",
            )
            self._help_btn.setMaximumWidth(80)
            self._help_btn.setFixedHeight(CURRENT_LAYOUT.button_height)
            help_actions.append(HeaderAction("help", self._help_btn))

        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        cancel_button.setMinimumWidth(70)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(button_styles["compact"])

        self._save_button = QPushButton("Save")
        self._save_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        self._save_button.setMinimumWidth(70)
        self.setup_save_button(self._save_button, self.save_config)
        self._save_button.setStyleSheet(button_styles["compact"])

        action_groups = []
        if help_actions:
            action_groups.append(HeaderActionGroup("group_help", help_actions))
        action_groups.extend(
            [
                HeaderActionGroup(
                    "group_reset",
                    [
                        HeaderAction("reset", reset_button),
                        HeaderAction("view_code", view_code_button),
                    ],
                ),
                HeaderActionGroup(
                    "group_save",
                    [
                        HeaderAction("cancel", cancel_button),
                        HeaderAction("save", self._save_button),
                    ],
                ),
            ],
        )

        header_widget = FormWindowActionHeader(
            title_text=title_text,
            title_color=self.color_scheme.to_hex(self.color_scheme.text_accent),
            action_groups=action_groups,
            stay_priority=["group_save", "group_help", "group_reset"],
            right_aligned_group_ids=["group_save"],
            parent=self,
        )
        self._header_label = header_widget.header_label
        self._layout.addWidget(header_widget)

        # Create splitter with tree view on left and form on right
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(True)  # Allow collapsing to 0
        self.splitter.setHandleWidth(5)  # Make handle more visible

        # Left panel - Inheritance hierarchy tree
        self.tree_widget = self._create_inheritance_tree()
        self.splitter.addWidget(self.tree_widget)

        # Right panel - Parameter form with scroll area
        # Always use scroll area for consistent navigation behavior
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setWidget(self.form_manager)
        self.splitter.addWidget(self.scroll_area)

        # Set splitter proportions (30% tree, 70% form)
        self.splitter.setSizes([300, 700])

        # Install collapsible splitter helper for double-click toggle
        self.splitter_helper = CollapsibleSplitterHelper(
            self.splitter, left_panel_index=0
        )
        self.splitter_helper.set_initial_size(300)

        # Add splitter with stretch factor so it expands to fill available space
        self._layout.addWidget(self.splitter, 1)  # stretch factor = 1

        # Apply centralized styling (config window style includes tree styling now)
        self.setStyleSheet(
            self.style_generator.generate_config_window_style()
            + "\n"
            + self.style_generator.generate_tree_widget_style()
        )

        # CRITICAL: Initialize scope-based border styling AFTER widgets are created
        # This ensures widgets exist when apply_scope_accent_styling() tries to style them
        # (mirrors DualEditorWindow pattern which calls init_scope_border in setup_connections)
        if self.scope_id:
            self.init_scope_border()

    def showEvent(self, a0) -> None:
        super().showEvent(a0)
        if not self._default_size_applied:
            self.resize(550, 600)
            QTimer.singleShot(0, lambda: self.resize(550, 600))
            self._default_size_applied = True
            self.setProperty("_fixed_default_size", True)
        self._log_window_size("shown")

    def resizeEvent(self, a0) -> None:
        super().resizeEvent(a0)
        self._log_window_size("resized")

    def _log_window_size(self, context: str) -> None:
        size = self.size()
        logger.debug(
            "Config window %s size=%dx%d pos=%d,%d",
            context,
            size.width(),
            size.height(),
            self.x(),
            self.y(),
        )

    def apply_scope_accent_styling(self) -> None:
        """Apply scope accent color to ConfigWindow-specific elements.

        Extends base class to add: Save button, header label, tree selection.
        """
        # Call base class for common elements (input focus, HelpButtons)
        super().apply_scope_accent_styling()

        accent_color = self.get_scope_accent_color()
        if not accent_color:
            return

        hex_color = accent_color.name()

        # Style Save button with hover effect
        save_button_style = f"""
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
        """
        if self._save_button is not None:
            self._save_button.setStyleSheet(save_button_style)

        # Style header label with scope accent color
        if self._header_label is not None:
            self._header_label.setStyleSheet(f"color: {hex_color};")

        # Style tree selection with scope accent
        tree_style = self.get_scope_tree_selection_stylesheet()
        if tree_style and self.tree_widget is not None:
            current_style = self.tree_widget.styleSheet() or ""
            self.tree_widget.setStyleSheet(f"{current_style}\n{tree_style}")

        # Style help button with scope accent color
        if self._help_btn is not None:
            self._help_btn.set_scope_accent_color(accent_color)

    def _create_inheritance_tree(self) -> QTreeWidget:
        """Create tree widget showing inheritance hierarchy for navigation."""
        # Pass form_manager as flash_manager - tree reads from SAME _flash_colors dict as groupboxes
        # ONE source of truth: form_manager already subscribes to ObjectState.on_resolved_changed
        # Pass state for automatic dirty tracking subscription (handled by helper)
        tree = self.tree_helper.create_tree_widget(
            flash_manager=self.form_manager,
            state=self.state,
            strip_config_suffix=True,
        )
        self.tree_helper.populate_from_root_dataclass(tree, self.config_class)

        # Initialize dirty styling AFTER population (when _field_to_item is filled)
        self.tree_helper.initialize_dirty_styling()

        # Register tree repaint callback so flash animation triggers tree repaint
        self.form_manager.register_repaint_callback(lambda: tree.viewport().update())

        # Connect double-click to navigation
        tree.itemDoubleClicked.connect(self._on_tree_item_double_clicked)

        return tree

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item double-clicks for navigation."""
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        if not isinstance(data, ConfigTreeItemPayload):
            raise TypeError(
                "Config tree item data must be ConfigTreeItemPayload; got "
                f"{type(data).__name__}."
            )
        payload = data

        # Check if this item is ui_hidden - if so, ignore the double-click
        if payload.ui_hidden:
            logger.debug("Ignoring double-click on ui_hidden item")
            return

        CONFIG_TREE_ITEM_HANDLERS[payload.item_type](self, payload)

    def _find_field_for_class(self, target_class) -> str:
        """Find the field name that has the given class type (or its lazy version)."""

        # Get the root dataclass type
        root_config = self.form_manager.object_instance
        if not dataclasses.is_dataclass(root_config):
            return None

        root_type = type(root_config)

        # Search through all fields to find one with matching type
        for field in dataclasses.fields(root_type):
            field_type = field.type

            # Check if field type matches target class directly
            if field_type == target_class:
                return field.name

            # Check if field type is a lazy version of target class
            if SimplifiedLazyDefaultPlaceholderService.has_lazy_resolution(field_type):
                # Get the base class of the lazy type
                for base in field_type.__bases__:
                    if base == target_class:
                        return field.name

        return None

    # _scroll_to_section is provided by ScrollableFormMixin

    def update_widget_value(self, widget: QWidget, value: Any):
        """
        Update widget value without triggering signals.

        Args:
            widget: Widget to update
            value: New value
        """
        # Temporarily block signals to avoid recursion
        widget.blockSignals(True)

        try:
            if isinstance(widget, QCheckBox):
                widget.setChecked(ConfigWidgetValueProjection.checkbox_value(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(ConfigWidgetValueProjection.spin_box_value(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(
                    ConfigWidgetValueProjection.double_spin_box_value(value)
                )
            elif isinstance(widget, QComboBox):
                for i in range(widget.count()):
                    if widget.itemData(i) == value:
                        widget.setCurrentIndex(i)
                        break
            elif isinstance(widget, QLineEdit):
                widget.setText(ConfigWidgetValueProjection.line_edit_text(value))
        finally:
            widget.blockSignals(False)

    def reset_to_defaults(self):
        """Reset all parameters using centralized service with full sophistication."""
        # Service layer now contains ALL the sophisticated logic previously in infrastructure classes
        # This includes nested dataclass reset, lazy awareness, and recursive traversal
        # NOTE: reset_all_parameters already handles placeholder refresh internally via
        # refresh_with_live_context, so no additional call needed
        self.form_manager.reset_all_parameters()

        logger.debug("Reset all parameters using enhanced ParameterFormManager service")

    def save_config(self, *, close_window=True):
        """Save the configuration preserving lazy behavior for unset fields. If close_window is True, close after saving; else, keep open."""
        try:
            # CRITICAL: Use to_object() to reconstruct nested dataclass structure from flat storage
            # get_current_values() returns flat dict with dotted paths like 'well_filter_config.well_filter'
            # which cannot be passed directly to the dataclass constructor
            new_config = self._config_session.to_object()

            # CRITICAL: Set flag to prevent refresh_config from recreating the form
            # The window already has the correct data - it just saved it!
            self._config_session.begin_save_callback(id(self))
            try:
                # Emit signal and call callback
                self.config_saved.emit(new_config)

                if self.on_save_callback:
                    logger.info(
                        f"🔍 SAVE_CONFIG: Calling on_save_callback (id={id(self)})"
                    )
                    self.on_save_callback(new_config)
                    logger.info(
                        f"🔍 SAVE_CONFIG: Returned from on_save_callback (id={id(self)})"
                    )
            finally:
                self._config_session.end_save_callback(id(self))

            self._config_session.publish_saved_global_config(new_config)

            # UNIFIED: Both paths share same logic, differ only in whether to close window
            if close_window:
                self.accept()  # Marks saved + unregisters + cleans up + closes
            else:
                self.mark_saved_and_refresh_all()  # Marks saved + refreshes, but stays open

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Failed to save configuration:\n{e}"
            )

    def _view_code(self):
        """Open code editor to view/edit the configuration as Python code."""
        try:
            # CRITICAL: Refresh with live context BEFORE getting current values
            # This ensures code editor shows unsaved changes from other open windows
            # Example: GlobalPipelineConfig editor open with unsaved zarr_config changes
            #          → PipelineConfig code editor should show those live zarr_config values
            ParameterOpsService().refresh_with_live_context(self.form_manager)

            # Get current config using to_object() to reconstruct nested structure from flat storage
            current_config = self._config_session.to_object()

            # Generate code using existing function
            python_code = generate_python_source(
                Assignment("config", current_config),
                header="# Configuration Code",
                clean_mode=True,
            )

            # Launch editor
            editor_service = SimpleCodeEditorService(self)
            use_external = os.environ.get(
                "OPENHCS_USE_EXTERNAL_EDITOR", ""
            ).lower() in ("1", "true", "yes")

            editor_service.edit_code(
                initial_content=python_code,
                title=f"View/Edit {self.config_class.__name__}",
                callback=self._handle_edited_config_code,
                use_external=use_external,
                code_type="config",
                code_data={"config_class": self.config_class, "clean_mode": True},
            )

        except Exception as e:
            logger.error(f"Failed to view config code: {e}")
            QMessageBox.critical(self, "View Code Error", f"Failed to view code:\n{e}")

    def _handle_edited_config_code(self, edited_code: str):
        """Handle edited configuration code from the code editor."""
        try:
            # SIMPLIFIED: Just exec with patched constructors
            # The patched constructors preserve None vs concrete distinction in raw field values
            # No need to parse code - just inspect raw values after exec
            namespace = {}

            with CodeEditorFormUpdater.patch_lazy_constructors():
                exec(edited_code, namespace)

            new_config = namespace.get("config")
            if not new_config:
                raise ValueError("No 'config' variable found in edited code")

            if not isinstance(new_config, self.config_class):
                raise ValueError(
                    f"Expected {self.config_class.__name__}, got {type(new_config).__name__}"
                )

            # Update current config
            self.current_config = new_config

            # For PipelineConfig: no context update is needed here. The
            # orchestrator applies pipeline config in the save callback.
            self._config_session.apply_code_edit_context(new_config)
            self._update_form_from_config(new_config)

            logger.info("Updated config from edited code")

        except Exception as e:
            logger.error(f"Failed to apply edited config code: {e}")
            QMessageBox.critical(
                self, "Code Edit Error", f"Failed to apply edited code:\n{e}"
            )

    def _on_global_config_field_changed(self, param_name: str, value: Any):
        """Track that global config has unsaved changes.

        NOTE: LIVE thread-local is now auto-updated by ObjectState.update_parameter()
        This callback just tracks dirty state for UI purposes.
        """
        del param_name, value
        self._config_session.mark_global_field_changed()

    def _update_form_from_config(self, new_config):
        """Update form values from new config using the shared updater."""
        # NOTE:
        # Do NOT set _block_cross_window_updates here.
        # We want code-mode edits to behave like a series of normal user edits,
        # so FieldChangeDispatcher will emit parameter_changed and
        # context_value_changed just like manual widget changes.
        CodeEditorFormUpdater.update_form_from_instance(
            self.form_manager,
            new_config,
        )

    def reject(self):
        """Handle dialog rejection (Cancel button).

        Restores global config context and ObjectState to last saved state.
        """
        if self._config_session.restore_global_context_if_dirty():
            logger.debug(f"Restored {self.config_class.__name__} context after cancel")

        self.config_cancelled.emit()

        # CRITICAL: super().reject() calls state.restore_saved() to undo ALL unsaved changes
        # This restores all parameters (not just global context) to last saved state
        super().reject()  # BaseFormDialog handles state restoration + unregistration

        # CRITICAL: Trigger global refresh AFTER unregistration so other windows
        # re-collect live context without this cancelled window's values
        # This ensures group_by selector and other placeholders sync correctly
        ObjectStateRegistry.increment_token()
        logger.debug(
            f"Triggered global refresh after cancelling {self.config_class.__name__} editor"
        )

    def closeEvent(self, a0):
        """Override to cleanup dirty subscriptions before closing."""
        self.state.off_state_changed(self._dirty_title_callback)
        self.tree_helper.cleanup_subscriptions()
        super().closeEvent(a0)
