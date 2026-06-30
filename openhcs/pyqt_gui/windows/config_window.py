"""
Configuration Window for PyQt6

Configuration editing dialog with full feature parity to Textual TUI version.
Uses hybrid approach: extracted business logic + clean PyQt6 UI.
"""

import logging
import dataclasses
from functools import partial
from typing import Callable, Optional

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QMessageBox,
)
from PyQt6.QtCore import pyqtSignal, QTimer

# Infrastructure classes removed - functionality migrated to ParameterFormManager service layer
from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager, FormManagerConfig
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.widgets.shared.config_hierarchy_tree import (
    ConfigHierarchyTreeHelper,
)
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin
from pyqt_reactive.widgets.shared.clickable_help_components import HelpButton, HelpContext
from pyqt_reactive.services.parameter_ops_service import ParameterOpsService
from pyqt_reactive.services.window_code_document import WindowCodeDocumentDriver
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.forms.parameter_value_contracts import ParameterValue
from pyqt_reactive.forms.widget_strategies import PyQt6WidgetEnhancer
from pyqt_reactive.theming import ColorScheme, WidgetTheme
from pyqt_reactive.widgets.shared import (
    BaseFormDialog,
    DirtyWindowPresentation,
    FormWindowActionHeader,
    HeaderAction,
    HeaderActionGroup,
    ManagedWindowActionCapabilities,
    ManagedStateRestorePolicy,
    create_scrollable_form_body,
)
from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
from openhcs.pyqt_gui.windows.config_edit_session import ConfigEditSession
from openhcs.pyqt_gui.services.pycodified_window_code_document import (
    ExternalCodeEditorPreference,
    PycodifiedObjectCodeDocumentDriver,
    PycodifiedObjectDocumentSpec,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry

# ❌ REMOVED: require_config_context decorator - enhanced decorator events system handles context automatically
from openhcs.core.lazy_placeholder import (
    LazyDefaultPlaceholderService as FullLazyDefaultPlaceholderService,
)


logger = logging.getLogger(__name__)


ConfigObject = GlobalPipelineConfig | PipelineConfig


class ConfigWindowStateResolver:
    """Resolve the ObjectState a config window is allowed to edit."""

    def __init__(
        self,
        config_class: type[ConfigObject],
        current_config: ConfigObject,
        scope_id: Optional[str],
    ) -> None:
        self.config_class = config_class
        self.current_config = current_config
        self.scope_id = scope_id

    def resolve(self) -> ObjectState:
        state = self._registered_state()
        if self.config_class is PipelineConfig:
            return self._required_pipeline_config_state(state)

        if state is not None:
            return state
        return ObjectState(
            object_instance=self.current_config,
            scope_id=self._canonical_scope_id(),
        )

    def _registered_state(self) -> ObjectState | None:
        for scope_id in self._candidate_scope_ids():
            state = ObjectStateRegistry.get_by_scope(scope_id)
            if state is not None:
                return state
        return None

    def _candidate_scope_ids(self) -> tuple[str | None, ...]:
        if self.scope_id is None:
            return (None,)
        return OpenHCSUiWindowId.manager_scopes_for_agent_window_id(self.scope_id)

    def _canonical_scope_id(self) -> str | None:
        if self.scope_id is None:
            return None
        return OpenHCSUiWindowId.canonical_manager_scope_for_agent_window_id(
            self.scope_id
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

    def __init__(
        self,
        config_class: type[ConfigObject],
        current_config: ConfigObject,
        on_save_callback: Callable[[ConfigObject], None] | None = None,
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

        self.state_restore_policy = ManagedStateRestorePolicy()

        # Initialize theme surface
        self.theme = WidgetTheme.from_optional(color_scheme)
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
        placeholder_prefix = (
            "Pipeline default"
            if is_lazy_dataclass
            else "Default"
        )

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
            self.state_restore_policy = ManagedStateRestorePolicy(
                propagate_descendants=False
            )

        self._config_session = ConfigEditSession(
            config_class=self.config_class,
            state=self.state,
            original_config=current_config,
        )
        self._code_document_driver = PycodifiedObjectCodeDocumentDriver(
            spec=PycodifiedObjectDocumentSpec(
                assignment_name="config",
                title=f"View/Edit {self.config_class.__name__}",
                header="# Configuration Code",
                expected_type=self.config_class,
            ),
            current_object=self._current_config_for_code_document,
            apply_object=self._apply_config_from_code_document,
            before_read=self._refresh_code_document_context,
        )

        # CRITICAL: Config window manages its own scroll area, so tell form_manager NOT to create one
        config = FormManagerConfig(
            parent=None,
            scope_id=self.scope_id,
            color_scheme=self.theme.scheme,
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
        if self.form_manager is None:
            raise RuntimeError("ConfigWindow form manager is not initialized.")
        return (self.form_manager,)

    def window_code_document_driver(self) -> WindowCodeDocumentDriver | None:
        """Expose this config window's pycodified code-mode document."""
        return self._code_document_driver

    def window_manager_scope_id(self) -> str | None:
        """Expose the stable UI/window id for WindowManager registration."""
        if self.scope_id is None:
            return None
        return OpenHCSUiWindowId.agent_window_id_for_manager_scope(self.scope_id)

    def managed_window_action_capabilities(
        self,
    ) -> ManagedWindowActionCapabilities:
        """Expose config-window save/cancel semantics to agent actions."""
        return ManagedWindowActionCapabilities(
            save_and_close=True,
            save_without_close=True,
            discard_and_close=True,
        )

    def agent_save_managed_window(self, *, close_window: bool) -> None:
        """Save through the same workflow as the visible Save button."""
        self.save_config(close_window=close_window)

    def dirty_window_widgets(self) -> tuple[QLabel, QPushButton] | None:
        if self._header_label is None or self._save_button is None:
            return None
        return self._header_label, self._save_button

    def dirty_window_presentation(self) -> DirtyWindowPresentation:
        """Build config-window dirty/signature presentation."""
        return DirtyWindowPresentation(
            window_title=self._base_window_title,
            header_text=f"Configure {self.config_class.__name__}",
            save_label="Save",
            is_dirty=self.dirty_state.is_dirty,
            has_signature_diff=self.dirty_state.has_signature_diff,
            mark_save_label_dirty=False,
        )

    def _update_window_title_dirty_marker(self) -> None:
        """Update config dirty/signature markers."""
        self.apply_dirty_window_presentation()

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(self._base_window_title)
        self.setModal(False)  # Non-modal like plate manager and pipeline editor
        if self.size().isEmpty():
            self.resize(550, 600)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(4)

        button_styles = self.theme.styles.generate_config_button_styles()
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
                    color_scheme=self.theme.scheme,
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
            title_color=self.theme.scheme.to_hex(self.theme.scheme.text_accent),
            action_groups=action_groups,
            stay_priority=["group_save", "group_help", "group_reset"],
            right_aligned_group_ids=["group_save"],
            parent=self,
        )
        self._header_label = header_widget.header_label
        self._layout.addWidget(header_widget)

        self.tree_widget = self.tree_helper.create_tree_from_root_dataclass(
            root_dataclass=self.config_class,
            form_manager=self.form_manager,
            state=self.state,
            strip_config_suffix=True,
            on_item_double_clicked=self._on_tree_item_double_clicked,
        )
        body_parts = create_scrollable_form_body(
            form_widget=self.form_manager,
            tree_widget=self.tree_widget,
            tree_initial_size=300,
            form_initial_size=700,
            parent=self,
        )
        self.scroll_area = body_parts.scroll_area
        self.splitter = body_parts.splitter
        self.splitter_helper = body_parts.splitter_helper

        # Add splitter with stretch factor so it expands to fill available space
        self._layout.addWidget(body_parts.body_widget, 1)

        # Apply centralized styling (config window style includes tree styling now)
        self.setStyleSheet(
            self.theme.styles.generate_config_window_style()
            + "\n"
            + self.theme.styles.generate_tree_widget_style()
        )

        # CRITICAL: Initialize scope-based border styling AFTER widgets are created
        # This ensures widgets exist when apply_scope_accent_styling() tries to style them
        # (mirrors DualEditorWindow pattern which calls init_scope_border in setup_connections)
        if self.scope_id is not None:
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

        # Style Save button with hover effect
        if self._save_button is not None:
            self._save_button.setStyleSheet(self.get_scope_accent_stylesheet())

        # Style header label with scope accent color
        if self._header_label is not None:
            self._header_label.setStyleSheet(f"color: {accent_color.name()};")

        # Style tree selection with scope accent
        tree_style = self.get_scope_tree_selection_stylesheet()
        if tree_style and self.tree_widget is not None:
            current_style = self.tree_widget.styleSheet()
            self.tree_widget.setStyleSheet(f"{current_style}\n{tree_style}")

        # Style help button with scope accent color
        if self._help_btn is not None:
            self._help_btn.set_scope_accent_color(accent_color)

    def _on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle tree item double-clicks for navigation."""
        self.tree_helper.activate_item(
            item,
            scroll_to_section=self._scroll_to_section,
            field_for_class=partial(
                self.tree_helper.field_for_class_in_dataclass_instance,
                self.form_manager.object_instance,
            ),
        )

    # _scroll_to_section is provided by ScrollableFormMixin

    def update_widget_value(self, widget: QWidget, value: ParameterValue | None) -> None:
        """
        Update widget value without triggering signals.

        Args:
            widget: Widget to update
            value: New value
        """
        PyQt6WidgetEnhancer.set_widget_value(widget, value)

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

            self.finish_managed_save(close_window=close_window)

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Failed to save configuration:\n{e}"
            )

    def _view_code(self):
        """Open code editor to view/edit the configuration as Python code."""
        try:
            document = self._code_document_driver.read_document()
            editor_service = SimpleCodeEditorService(self)

            editor_service.edit_code(
                initial_content=document.source,
                title=document.title,
                callback=self._handle_edited_config_code,
                use_external=ExternalCodeEditorPreference.use_external_editor(),
                code_type="config",
                code_data={"config_class": self.config_class, "clean_mode": True},
            )

        except Exception as e:
            logger.error(f"Failed to view config code: {e}")
            QMessageBox.critical(self, "View Code Error", f"Failed to view code:\n{e}")

    def _handle_edited_config_code(self, edited_code: str):
        """Handle edited configuration code from the code editor."""
        try:
            self._code_document_driver.apply_source(edited_code)
            logger.info("Updated config from edited code")

        except Exception as e:
            logger.error(f"Failed to apply edited config code: {e}")
            QMessageBox.critical(
                self, "Code Edit Error", f"Failed to apply edited code:\n{e}"
            )

    def _refresh_code_document_context(self) -> None:
        """Refresh live context before rendering this config as source."""
        ParameterOpsService().refresh_with_live_context(self.form_manager)

    def _current_config_for_code_document(self) -> ConfigObject:
        """Return the current config object reconstructed from ObjectState."""
        return self._config_session.to_object()

    def _apply_config_from_code_document(self, new_config: ConfigObject) -> None:
        """Apply a parsed code-mode config through the normal form path."""
        self.current_config = new_config
        self._config_session.apply_code_edit_context(new_config)
        self._update_form_from_config(new_config)

    def _on_global_config_field_changed(
        self,
        param_name: str,
        value: ParameterValue | None,
    ) -> None:
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
