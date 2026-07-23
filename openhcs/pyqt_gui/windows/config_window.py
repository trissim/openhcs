"""ObjectState-backed configuration editor windows."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable

from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from objectstate import ObjectState, ObjectStateRegistry
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.forms.parameter_form_manager import (
    FormManagerConfig,
    ParameterFormManager,
)
from pyqt_reactive.services.parameter_ops_service import ParameterOpsService
from pyqt_reactive.services.window_code_document import (
    WindowCodeDocument,
    WindowCodeDocumentDriver,
)
from pyqt_reactive.theming import ColorScheme, WidgetTheme
from pyqt_reactive.widgets.editors.simple_code_editor import SimpleCodeEditorService
from pyqt_reactive.widgets.shared import (
    ActionTabSpec,
    ActionTabbedWindowBody,
    BaseFormDialog,
    DirtyWindowPresentation,
    FormWindowActionHeader,
    HeaderAction,
    HeaderActionGroup,
    ManagedStateRestorePolicy,
    ManagedWindowActionCapabilities,
    create_scrollable_form_body,
)
from pyqt_reactive.widgets.shared.clickable_help_components import (
    HelpButton,
    HelpContext,
)
from pyqt_reactive.widgets.shared.config_hierarchy_tree import (
    ConfigHierarchyTreeHelper,
)
from pyqt_reactive.widgets.shared.form_window_action_header import (
    HeaderActionGroupRole,
)
from pyqt_reactive.widgets.shared.scrollable_form_mixin import ScrollableFormMixin

from openhcs.pyqt_gui.services.pycodified_window_code_document import (
    ExternalCodeEditorPreference,
    PycodifiedConfigDocumentSpec,
    PycodifiedObjectCodeDocumentDriver,
)
from openhcs.pyqt_gui.services.ui_window_ids import OpenHCSUiWindowId
from openhcs.pyqt_gui.windows.config_edit_session import ConfigEditSession
from openhcs.ui.shared.code_editor_form_updater import CodeEditorFormUpdater


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ConfigWindowTabSpec:
    """One caller-owned config state and its persistence boundary."""

    state: ObjectState
    on_save: Callable[[object], None] | None = None

    @property
    def label(self) -> str:
        return type(self.state.saved_object).__name__


@dataclass(slots=True)
class _ConfigEditorTab:
    spec: ConfigWindowTabSpec
    state: ObjectState
    session: ConfigEditSession
    form_manager: ParameterFormManager
    tree_helper: ConfigHierarchyTreeHelper
    tree_widget: QTreeWidget
    content: QWidget
    actions: QWidget
    scroll_area: QWidget
    code_document_driver: PycodifiedObjectCodeDocumentDriver
    restore_policy: ManagedStateRestorePolicy
    state_changed_callback: Callable[[set[str]], None]
    help_button: HelpButton | None


class _ActiveConfigCodeDocumentDriver(WindowCodeDocumentDriver):
    """Expose the selected config tab through the managed-window code API."""

    def __init__(self, active_driver: Callable[[], WindowCodeDocumentDriver]) -> None:
        self._active_driver = active_driver

    def read_document(self, clean: bool = True) -> WindowCodeDocument:
        return self._active_driver().read_document(clean=clean)

    def validate_source(self, source: str) -> None:
        self._active_driver().validate_source(source)

    def apply_source(self, source: str) -> None:
        self._active_driver().apply_source(source)


class ConfigWindow(ScrollableFormMixin, BaseFormDialog):
    """Edit one or more authoritative configuration objects in tabs."""

    config_saved = pyqtSignal(object)
    config_cancelled = pyqtSignal()

    def __init__(
        self,
        tabs: tuple[ConfigWindowTabSpec, ...],
        color_scheme: ColorScheme | None = None,
        parent=None,
        scope_id: str | None = None,
        title_text: str | None = None,
    ) -> None:
        if not tabs:
            raise ValueError("ConfigWindow requires at least one config tab.")
        super().__init__(parent)

        self.scope_id = scope_id
        self.theme = WidgetTheme.from_optional(color_scheme)
        self._scope_accent_color = None
        self._header_label: QLabel | None = None
        self._save_button: QPushButton | None = None
        self._default_size_applied = False
        self._base_window_title = title_text or f"Config {tabs[0].label}"
        self._action_header: FormWindowActionHeader | None = None
        self._tabs: list[_ConfigEditorTab] = []
        self._tab_body = ActionTabbedWindowBody(
            color_scheme=self.theme.scheme,
            parent=self,
        )

        for spec in tabs:
            tab = self._build_tab(spec)
            self._tabs.append(tab)
            self._tab_body.add_tab(
                ActionTabSpec(
                    label=spec.label,
                    content=tab.content,
                )
            )

        self.state = self.active_tab.state
        self.state_restore_policy = self.active_tab.restore_policy
        self._sync_tab_action_visibility()
        self._code_document_driver = _ActiveConfigCodeDocumentDriver(
            lambda: self.active_tab.code_document_driver
        )

        self.setup_ui()
        self._tab_body.current_changed.connect(self._on_tab_changed)
        self.connect_change_detection()
        self.detect_changes()
        logger.debug("Config window initialized with %d tab(s)", len(self._tabs))

    @property
    def active_tab(self) -> _ConfigEditorTab:
        return self._tabs[max(self._tab_body.current_index(), 0)]

    @property
    def form_manager(self) -> ParameterFormManager:
        return self.active_tab.form_manager

    @property
    def tree_widget(self) -> QTreeWidget:
        return self.active_tab.tree_widget

    @property
    def scroll_area(self):
        return self.active_tab.scroll_area

    def _build_tab(self, spec: ConfigWindowTabSpec) -> _ConfigEditorTab:
        state = spec.state
        config = state.saved_object
        config_type = type(config)
        session = ConfigEditSession(
            state=state,
            original_config=config,
        )
        form_config = FormManagerConfig(
            parent=None,
            scope_id=self.scope_id,
            color_scheme=self.theme.scheme,
            scope_accent_color=self._scope_accent_color,
        )
        form_config.field_id = ""
        form_manager = ParameterFormManager(state=state, config=form_config)
        if session.is_global_config:
            form_manager.parameter_changed.connect(
                lambda _name, _value, current_session=session: (
                    current_session.mark_global_field_changed()
                )
            )

        tree_helper = ConfigHierarchyTreeHelper()
        tree_widget = tree_helper.create_tree_from_root_dataclass(
            root_dataclass=config_type,
            form_manager=form_manager,
            state=state,
            strip_config_suffix=True,
            on_item_double_clicked=partial(
                self._on_tree_item_double_clicked,
                tree_helper,
                form_manager,
            ),
        )
        body_parts = create_scrollable_form_body(
            form_widget=form_manager,
            tree_widget=tree_widget,
            tree_initial_size=300,
            form_initial_size=700,
            parent=self,
        )

        code_document_driver = PycodifiedObjectCodeDocumentDriver(
            spec=PycodifiedConfigDocumentSpec(
                title=f"View/Edit {config_type.__name__}",
                expected_type=config_type,
            ),
            current_object=session.to_code_document_object,
            apply_object=partial(
                self._apply_config_from_code_document,
                session,
                form_manager,
            ),
            before_read=partial(
                ParameterOpsService().refresh_with_live_context,
                form_manager,
            ),
        )
        actions, help_button = self._build_tab_actions(
            config_type,
            form_manager,
            code_document_driver,
        )
        restore_policy = ManagedStateRestorePolicy(
            propagate_descendants=not state.has_delegate
        )

        def state_changed(_fields: set[str]) -> None:
            self.detect_changes()

        state.on_state_changed(state_changed)
        return _ConfigEditorTab(
            spec=spec,
            state=state,
            session=session,
            form_manager=form_manager,
            tree_helper=tree_helper,
            tree_widget=tree_widget,
            content=body_parts.body_widget,
            actions=actions,
            scroll_area=body_parts.scroll_area,
            code_document_driver=code_document_driver,
            restore_policy=restore_policy,
            state_changed_callback=state_changed,
            help_button=help_button,
        )

    def _build_tab_actions(
        self,
        config_type: type[object],
        form_manager: ParameterFormManager,
        code_document_driver: PycodifiedObjectCodeDocumentDriver,
    ) -> tuple[QWidget, HelpButton | None]:
        actions = QWidget(self)
        layout = QHBoxLayout(actions)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        button_styles = self.theme.styles.generate_config_button_styles()

        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(form_manager.reset_all_parameters)
        reset_button.setStyleSheet(button_styles["compact"])
        layout.addWidget(reset_button)

        code_button = QPushButton("View Code")
        code_button.clicked.connect(partial(self._view_code, code_document_driver))
        code_button.setStyleSheet(button_styles["compact"])
        layout.addWidget(code_button)

        help_button = None
        if dataclasses.is_dataclass(config_type):
            help_button = HelpButton(
                help_context=HelpContext(
                    help_target=config_type,
                    color_scheme=self.theme.scheme,
                    scope_accent_color=self._scope_accent_color,
                ),
                text="Help",
            )
            help_button.setStyleSheet(button_styles["compact"])
            help_button.setFixedHeight(CURRENT_LAYOUT.button_height)

        for index in range(layout.count()):
            widget = layout.itemAt(index).widget()
            if widget is not None:
                widget.setFixedHeight(CURRENT_LAYOUT.button_height)
        return actions, help_button

    def form_managers(self) -> tuple[ParameterFormManager, ...]:
        return tuple(tab.form_manager for tab in self._tabs)

    def window_code_document_driver(self) -> WindowCodeDocumentDriver:
        return self._code_document_driver

    def window_manager_scope_id(self) -> str | None:
        if self.scope_id is None:
            return None
        return OpenHCSUiWindowId.agent_window_id_for_manager_scope(self.scope_id)

    def managed_window_action_capabilities(
        self,
    ) -> ManagedWindowActionCapabilities:
        return ManagedWindowActionCapabilities(
            save_and_close=True,
            save_without_close=True,
            discard_and_close=True,
        )

    def agent_save_managed_window(self, *, close_window: bool) -> None:
        self.save_config(close_window=close_window)

    def dirty_window_widgets(self) -> tuple[QLabel, QPushButton] | None:
        if self._header_label is None or self._save_button is None:
            return None
        return self._header_label, self._save_button

    def dirty_window_presentation(self) -> DirtyWindowPresentation:
        is_dirty = any(tab.state.is_raw_dirty for tab in self._tabs)
        has_signature_diff = any(
            bool(tab.state.signature_diff_fields) for tab in self._tabs
        )
        return DirtyWindowPresentation(
            window_title=self._base_window_title,
            header_text=self._base_window_title,
            save_label="Save",
            is_dirty=is_dirty,
            has_signature_diff=has_signature_diff,
            mark_save_label_dirty=False,
        )

    def detect_changes(self) -> None:
        has_changes = any(tab.state.is_raw_dirty for tab in self._tabs)
        if has_changes != self.dirty_state.has_changes:
            self.dirty_state.has_changes = has_changes
            self.changes_detected.emit(has_changes)
        else:
            self.apply_dirty_window_presentation()

    def setup_ui(self) -> None:
        self.setWindowTitle(self._base_window_title)
        self.setModal(False)
        if self.size().isEmpty():
            self.resize(650, 650)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        button_styles = self.theme.styles.generate_config_button_styles()

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(button_styles["compact"])

        self._save_button = QPushButton("Save")
        self._save_button.setEnabled(False)
        self.setup_save_button(self._save_button, self.save_config)
        self._save_button.setStyleSheet(button_styles["compact"])

        for button in (cancel_button, self._save_button):
            button.setFixedHeight(CURRENT_LAYOUT.button_height)

        header = FormWindowActionHeader(
            title_text=self._base_window_title,
            title_color=self.theme.scheme.to_hex(self.theme.scheme.text_accent),
            action_groups=(
                HeaderActionGroup(
                    "group_title_companion",
                    tuple(
                        HeaderAction(f"tab_{index}_help", tab.help_button)
                        for index, tab in enumerate(self._tabs)
                        if tab.help_button is not None
                    ),
                    role=HeaderActionGroupRole.TITLE_COMPANION,
                ),
                HeaderActionGroup(
                    "group_auxiliary",
                    tuple(
                        HeaderAction(f"tab_{index}_actions", tab.actions)
                        for index, tab in enumerate(self._tabs)
                    ),
                    role=HeaderActionGroupRole.AUXILIARY,
                ),
                HeaderActionGroup(
                    "group_commit",
                    (
                        HeaderAction("cancel", cancel_button),
                        HeaderAction("save", self._save_button),
                    ),
                    role=HeaderActionGroupRole.COMMIT,
                ),
            ),
            parent=self,
        )
        self._action_header = header
        self._header_label = header.header_label
        self._sync_tab_action_visibility()
        layout.addWidget(header)
        layout.addWidget(self._tab_body, 1)

        self.setStyleSheet(
            self.theme.styles.generate_config_window_style()
            + "\n"
            + self.theme.styles.generate_tree_widget_style()
        )
        if self.scope_id is not None:
            self.init_scope_border()

    def _on_tab_changed(self, _index: int) -> None:
        self._sync_tab_action_visibility()
        self.state = self.active_tab.state
        self.state_restore_policy = self.active_tab.restore_policy
        self.apply_scope_accent_styling()
        self.detect_changes()

    def _sync_tab_action_visibility(self) -> None:
        active_index = max(self._tab_body.current_index(), 0)
        for index, tab in enumerate(self._tabs):
            tab.actions.setVisible(index == active_index)
            if tab.help_button is not None:
                tab.help_button.setVisible(index == active_index)
        if self._action_header is not None:
            self._action_header.refresh_layout()

    def _on_tree_item_double_clicked(
        self,
        tree_helper: ConfigHierarchyTreeHelper,
        form_manager: ParameterFormManager,
        item: QTreeWidgetItem,
        column: int,
    ) -> None:
        del column
        tree_helper.activate_item(
            item,
            scroll_to_section=self._scroll_to_section,
            field_for_class=partial(
                tree_helper.field_for_class_in_dataclass_instance,
                form_manager.object_instance,
            ),
        )

    def reset_to_defaults(self) -> None:
        self.active_tab.form_manager.reset_all_parameters()

    def save_config(self, *, close_window: bool = True) -> None:
        try:
            # Validate every page before mutating any authoritative state.
            for tab in self._tabs:
                tab.session.to_object()

            for tab in self._tabs:
                tab.state.mark_saved()

            committed = tuple(
                (tab, tab.state.saved_object) for tab in self._tabs
            )
            for tab, config in committed:
                tab.session.publish_saved_global_config(config)

            for tab, config in committed:
                tab.session.begin_save_callback(id(self))
                try:
                    self.config_saved.emit(config)
                    if tab.spec.on_save is not None:
                        tab.spec.on_save(config)
                finally:
                    tab.session.end_save_callback(id(self))

            ObjectStateRegistry.increment_token(notify=True)
            if close_window:
                self.accept_committed_state()
            else:
                self.detect_changes()
        except Exception as error:
            logger.error("Failed to save configuration: %s", error, exc_info=True)
            QMessageBox.critical(
                self,
                "Save Error",
                f"Failed to save configuration:\n{error}",
            )

    def _view_code(
        self,
        code_document_driver: PycodifiedObjectCodeDocumentDriver | None = None,
    ) -> None:
        driver = code_document_driver or self.active_tab.code_document_driver
        try:
            document = driver.read_document()
            SimpleCodeEditorService(self).edit_code(
                initial_content=document.source,
                title=document.title,
                callback=driver.apply_source,
                use_external=ExternalCodeEditorPreference.use_external_editor(),
                code_type="config",
                code_data={"clean_mode": True},
            )
        except Exception as error:
            logger.error("Failed to view config code: %s", error, exc_info=True)
            QMessageBox.critical(
                self,
                "View Code Error",
                f"Failed to view code:\n{error}",
            )

    @staticmethod
    def _apply_config_from_code_document(
        session: ConfigEditSession,
        form_manager: ParameterFormManager,
        new_config: object,
    ) -> None:
        session.apply_code_edit_context(new_config)
        CodeEditorFormUpdater.update_form_from_instance(form_manager, new_config)

    def reject(self) -> None:
        for tab in self._tabs:
            tab.session.restore_global_context_if_dirty()
            tab.restore_policy.restore(tab.state)
        self.config_cancelled.emit()
        super().reject()
        ObjectStateRegistry.increment_token()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._default_size_applied:
            self.resize(650, 650)
            QTimer.singleShot(0, lambda: self.resize(650, 650))
            self._default_size_applied = True
            self.setProperty("_fixed_default_size", True)

    def apply_scope_accent_styling(self) -> None:
        super().apply_scope_accent_styling()
        accent_color = self.get_scope_accent_color()
        if not accent_color:
            return
        if self._save_button is not None:
            self._save_button.setStyleSheet(
                self.theme.styles.generate_scope_accent_button_style(accent_color)
            )
        if self._header_label is not None:
            self._header_label.setStyleSheet(f"color: {accent_color.name()};")
        tree_style = self.get_scope_tree_selection_stylesheet()
        for tab in self._tabs:
            if tree_style:
                tab.tree_widget.setStyleSheet(
                    f"{tab.tree_widget.styleSheet()}\n{tree_style}"
                )
            if tab.help_button is not None:
                tab.help_button.set_scope_accent_color(accent_color)

    def closeEvent(self, event) -> None:
        for tab in self._tabs:
            tab.state.off_state_changed(tab.state_changed_callback)
            tab.tree_helper.cleanup_subscriptions()
            tab.restore_policy.restore(tab.state)
        super().closeEvent(event)
