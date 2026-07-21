"""
Image Browser Widget for PyQt6 GUI.

Displays a table of all image files from plate metadata and allows users to
view them in Napari with configurable display settings.
"""

import logging
import re
import subprocess
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

from polystore.base import storage_registry
from polystore.filemanager import FileManager
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from pyqt_reactive.forms.parameter_form_manager import (
    FormManagerConfig,
    ParameterFormManager,
)
from pyqt_reactive.theming import ColorScheme, StyleSheetGenerator
from pyqt_reactive.widgets.shared import TabbedFormConfig, TabbedFormWidget, TabConfig
from pyqt_reactive.widgets.shared.image_table_browser import (
    ImageTableBrowser,
    ImageTableValue,
)

from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.constants.constants import AllComponents, Backend, FileFormat
from openhcs.core.config import StreamingConfig
from openhcs.core.plate_image_inventory import (
    PlateFileInventory,
    PlateFileKind,
    PlateFileRecord,
    PlateResultFileInventory,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig

logger = logging.getLogger(__name__)


ALL_COMPONENT_VALUES = frozenset(component.value for component in AllComponents)


def _streaming_config_field_names() -> tuple[str, ...]:
    """Registered streaming-config field names used by image-browser controls."""
    return StreamingConfig.supported_config_keys()


@dataclass(frozen=True)
class ResultFileAction:
    """Double-click behavior for one result-file type."""
    file_type: FileFormat
    display_name: str
    handle: Callable[["ImageBrowserWidget", Path], None]

    def run(self, browser: "ImageBrowserWidget", file_path: Path) -> None:
        self.handle(browser, file_path)


@dataclass(slots=True)
class ImageBrowserItem(Mapping[str, ImageTableValue]):
    """One image/result row plus optional result-file action metadata."""

    key: str
    metadata: dict[str, ImageTableValue]
    result_file_type: FileFormat | None = None
    full_path: Path | None = None

    @property
    def is_result(self) -> bool:
        return self.result_file_type is not None

    @property
    def filename(self) -> str:
        return str(self.metadata["filename"])

    def result_path(self) -> Path:
        if self.full_path is None:
            raise RuntimeError(f"Image browser item {self.key!r} is not a result file.")
        return self.full_path

    def __getitem__(self, key: str) -> ImageTableValue:
        return self.metadata[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.metadata)

    def __len__(self) -> int:
        return len(self.metadata)


def _stream_roi_result(browser: "ImageBrowserWidget", file_path: Path) -> None:
    browser._stream_roi_file(file_path)


def _open_result_in_default_app(browser: "ImageBrowserWidget", file_path: Path) -> None:
    subprocess.run(["xdg-open", str(file_path)])


RESULT_FILE_ACTIONS = {
    FileFormat.ROI: ResultFileAction(
        file_type=FileFormat.ROI,
        display_name="ROI",
        handle=_stream_roi_result,
    ),
    FileFormat.CSV: ResultFileAction(
        file_type=FileFormat.CSV,
        display_name="CSV",
        handle=_open_result_in_default_app,
    ),
    FileFormat.JSON: ResultFileAction(
        file_type=FileFormat.JSON,
        display_name="JSON",
        handle=_open_result_in_default_app,
    ),
    FileFormat.TEXT: ResultFileAction(
        file_type=FileFormat.TEXT,
        display_name="TEXT",
        handle=_open_result_in_default_app,
    ),
}


@dataclass(frozen=True)
class StreamingViewerField:
    """Display metadata for one registered streaming viewer field."""

    field_name: str

    @property
    def display_name(self) -> str:
        return StreamingConfig.display_name_for_config_key(self.field_name)


def streaming_viewer_fields() -> tuple[StreamingViewerField, ...]:
    """Registered streaming-viewer fields with display metadata."""
    return tuple(
        StreamingViewerField(field_name)
        for field_name in _streaming_config_field_names()
    )


ImageBrowserConfig = make_dataclass(
    "ImageBrowserConfig",
    tuple(
        (
            field_name,
            StreamingConfig.config_type_for_key(field_name),
            field(default_factory=StreamingConfig.config_type_for_key(field_name)),
        )
        for field_name in _streaming_config_field_names()
    ),
    frozen=False,
    slots=True,
)


class ImageBrowserViewerControls:
    """Own viewer button construction and enabled-state projection."""

    def __init__(
        self,
        state: ObjectState,
        style_gen: StyleSheetGenerator,
        view_requested: Callable[[str], None],
    ):
        self.state = state
        self.style_gen = style_gen
        self.view_requested = view_requested
        self.buttons: Dict[str, QPushButton] = {}

    def create_header_buttons(self) -> list[QPushButton]:
        self.buttons.clear()
        buttons = []
        for field in streaming_viewer_fields():
            button = QPushButton(f"View in {field.display_name}")
            button.clicked.connect(
                lambda checked, fn=field.field_name: self.view_requested(fn)
            )
            button.setStyleSheet(self.style_gen.generate_button_style())
            button.setEnabled(False)
            self.buttons[field.field_name] = button
            buttons.append(button)
        return buttons

    def is_enabled(self, config_key: str) -> bool:
        enabled_path = f"{config_key}.enabled"
        return self.state.get_resolved_value(enabled_path) is True

    def enabled_viewers(self) -> list[str]:
        return [
            field.field_name
            for field in streaming_viewer_fields()
            if self.is_enabled(field.field_name)
        ]

    def update_button_state(self, config_key: str, has_selection: bool) -> None:
        button = self.buttons.get(config_key)
        if button is None:
            logger.warning("Streaming config key %s not in view buttons", config_key)
            return
        button.setEnabled(has_selection and self.is_enabled(config_key))

    def update_all_button_states(self, selected_keys: list) -> None:
        has_selection = len(selected_keys) > 0
        for config_key in self.buttons:
            self.update_button_state(config_key, has_selection)


class ImageBrowserMetadataDisplayResolver:
    """Resolve and cache domain display values for image metadata cells."""

    def __init__(self, orchestrator_getter: Callable[[], object | None]):
        self.orchestrator_getter = orchestrator_getter
        self._cache: dict[tuple[str, str], str] = {}
        self._raw_by_display: dict[tuple[str, str], str] = {}

    def clear(self) -> None:
        self._cache.clear()
        self._raw_by_display.clear()

    def display_value(self, metadata_key: str, raw_value: ImageTableValue) -> str:
        if raw_value is None:
            return "N/A"

        value_str = str(raw_value)
        cache_key = (metadata_key, value_str)
        cached_value = self._cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        display_value = self._resolve_display_value(metadata_key, value_str)
        self._cache[cache_key] = display_value
        self._raw_by_display[(metadata_key, display_value)] = value_str
        return display_value

    def display_values(
        self,
        metadata_key: str,
        raw_values: Set[str],
    ) -> tuple[str, ...]:
        """Format semantic values through the same table-cell projection."""
        return tuple(
            self.display_value(metadata_key, raw_value)
            for raw_value in raw_values
        )

    def raw_values(
        self,
        metadata_key: str,
        display_values: Set[str],
    ) -> Set[str]:
        """Recover semantic values already published by the cell projection."""
        return {
            self._raw_by_display[(metadata_key, display_value)]
            for display_value in display_values
            if (metadata_key, display_value) in self._raw_by_display
        }

    def _resolve_display_value(self, metadata_key: str, value_str: str) -> str:
        orchestrator = self.orchestrator_getter()
        if orchestrator is None:
            return value_str

        try:
            if metadata_key not in ALL_COMPONENT_VALUES:
                return value_str
            component = AllComponents(metadata_key)
            metadata_name = (
                orchestrator._metadata_cache_service.get_component_metadata(
                    component,
                    value_str,
                )
            )
            if metadata_name and metadata_name != "None":
                return f"{value_str} | {metadata_name}"
            logger.debug("No metadata name found for %s %s", metadata_key, value_str)
            return value_str
        except Exception as exc:
            logger.warning(
                "Could not get metadata for %s %s: %s",
                metadata_key,
                value_str,
                exc,
                exc_info=True,
            )
            return value_str


class ImageBrowserFilterController:
    """Own search, folder, and plate-well filtering for ImageBrowserWidget."""

    def __init__(self, browser: "ImageBrowserWidget"):
        self.browser = browser

    def apply_combined_filters(self) -> None:
        """Apply search, folder, and plate-well filters in one pass."""
        browser = self.browser
        selected_items = browser.folder_tree.selectedItems()
        folder_path = None
        results_folder_path = None
        if selected_items:
            folder_path = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
            if folder_path:
                results_folder_path = f"{folder_path}_results"

        if not browser.file_items:
            browser._set_visible_files({}, rebuild_index=False)
            return

        search_items = browser.image_table_browser.search_items(browser.search_input.text())
        result = {}
        for filename, item in search_items.items():
            include = True
            metadata = item.metadata

            if folder_path and include:
                include = (
                    str(Path(filename).parent) == folder_path
                    or filename.startswith(folder_path + "/")
                    or str(Path(filename).parent) == results_folder_path
                    or filename.startswith(results_folder_path + "/")
                )

            if browser.selected_wells and include:
                include = self._matches_wells(filename, metadata)

            if include:
                result[filename] = item

        browser._set_visible_files(result, rebuild_index=False)
        logger.debug("Combined filters: %s images shown", len(result))

    def filter_images(self, _search_term: str) -> None:
        """Compose text search with folder and plate-well filters."""
        self.apply_combined_filters()

    def _matches_wells(self, filename: str, metadata: dict) -> bool:
        try:
            well_id = self.browser._extract_well_id(metadata)
            matches = well_id in self.browser.selected_wells
            if not matches:
                logger.debug("[MATCH] Well %s not in selected_wells", well_id)
            return matches
        except (KeyError, ValueError) as exc:
            logger.debug("[MATCH] No well metadata for %s: %s", filename, exc)
            return False

class ImageBrowserFileFocusController:
    """Own semantic file focusing for ImageBrowserWidget."""

    def __init__(self, browser: "ImageBrowserWidget"):
        self.browser = browser

    def focus_path(self, file_path: str | Path) -> bool:
        browser = self.browser
        if not browser.file_items and browser.orchestrator:
            browser.load_images()

        for key in self._candidate_keys(file_path):
            if key in browser.file_items:
                return self._focus_key(key)
        unique_basename_key = self._unique_basename_key(file_path)
        if unique_basename_key is not None:
            return self._focus_key(unique_basename_key)
        return False

    def _candidate_keys(self, file_path: str | Path) -> tuple[str, ...]:
        browser = self.browser
        path = Path(file_path)
        candidates = [str(file_path)]
        if browser.orchestrator and path.is_absolute():
            try:
                candidates.append(str(path.relative_to(browser.orchestrator.plate_path)))
            except ValueError:
                pass
        candidates.append(path.name)
        return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _focus_key(self, key: str) -> bool:
        browser = self.browser
        if key not in browser.image_table_browser.filtered_items:
            browser._set_visible_files({key: browser.file_items[key]}, rebuild_index=False)
        return browser.image_table_browser.select_key(key)

    def _unique_basename_key(self, file_path: str | Path) -> str | None:
        basename = Path(file_path).name
        if not basename:
            return None
        matches = [
            key for key in self.browser.file_items
            if Path(key).name == basename
        ]
        if len(matches) == 1:
            return matches[0]
        return None


class ImageBrowserWidget(QWidget):
    """
    Image browser widget that displays all image files from plate metadata.

    Users can click on files to view them in Napari with configurable settings
    from the current PipelineConfig.
    """

    # Signals
    image_selected = pyqtSignal(str)  # Emitted when an image is selected
    _status_update_signal = pyqtSignal(
        str
    )  # Internal signal for thread-safe status updates

    def __init__(
        self,
        orchestrator=None,
        color_scheme: Optional[ColorScheme] = None,
        zmq_config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
        parent=None,
    ):
        super().__init__(parent)

        self.orchestrator = orchestrator
        self._zmq_config = zmq_config
        self.color_scheme = color_scheme or ColorScheme()
        self.style_gen = StyleSheetGenerator(self.color_scheme)
        # Fallback for standalone browsing; orchestrator-owned runs derive their
        # FileManager from the orchestrator property below.
        self._fallback_filemanager = FileManager(storage_registry)

        # Create root ObjectState from dynamically generated config container
        # This gives us a single registered state with nested configs via dotted paths
        self.config = ImageBrowserConfig()
        self.scope_id: Optional[str] = None
        self.state = self._create_state_for_orchestrator(orchestrator)

        # TabbedFormWidget will be created lazily in _create_right_panel
        # to avoid heavy initialization during widget construction.
        self.tabbed_form = None

        self.viewer_controls = ImageBrowserViewerControls(
            self.state,
            self.style_gen,
            self._view_selected_in_viewer,
        )
        # View buttons - dictionary keyed by viewer_type for dynamic handling
        self.view_buttons: Dict[str, QPushButton] = self.viewer_controls.buttons

        # File data tracking (images + results)
        self.file_items: dict[str, ImageBrowserItem] = {}
        self.selected_wells = set()  # Selected wells for filtering
        self.metadata_keys = []  # Column names from parser metadata (union of all keys)
        self.metadata_display_resolver = ImageBrowserMetadataDisplayResolver(
            lambda: self.orchestrator
        )
        self._syncing_plate_filter_selection = False
        self.filter_controller = ImageBrowserFilterController(self)
        self.file_focus_controller = ImageBrowserFileFocusController(self)

        # Plate view widget (will be created in init_ui)
        self.plate_view_widget = None
        self.plate_view_detached_window = None
        self.middle_splitter = None  # Reference to splitter for reattaching

        # ZMQ manager widget (may be created in init_ui)
        self.zmq_manager = None
        self.main_splitter = None
        self.right_panel = None

        # Streaming service for unified Napari/Fiji streaming.
        self._streaming_service_cache = None
        self._streaming_service_orchestrator = None

        self.init_ui()

        # Connect internal signal for thread-safe status updates
        self._status_update_signal.connect(self._update_status_label)

        # Load images if orchestrator is provided
        if self.orchestrator:
            QTimer.singleShot(0, self.load_images)

    @property
    def filemanager(self):
        """Current FileManager derived from orchestrator when available."""
        if self.orchestrator:
            return self.orchestrator.filemanager
        return self._fallback_filemanager

    @property
    def streaming_service(self):
        """Streaming service derived from the current orchestrator."""
        if not self.orchestrator:
            return None
        if self._streaming_service_orchestrator is not self.orchestrator:
            from openhcs.core.viewer_streaming_service import StreamingService

            self._streaming_service_cache = StreamingService(
                filemanager=self.filemanager,
                microscope_handler=self.orchestrator.microscope_handler,
                plate_path=self.orchestrator.plate_path,
                transport_config=self.orchestrator.transport_config,
            )
            self._streaming_service_orchestrator = self.orchestrator
        return self._streaming_service_cache

    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        layout.setSpacing(5)  # Reduced spacing between rows

        # Search input row with buttons on the right
        search_layout = QHBoxLayout()
        search_layout.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search images by filename or metadata...")
        self.search_input.textChanged.connect(self.filter_controller.filter_images)
        # Apply same styling as function selector
        self.search_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: {self.color_scheme.to_hex(self.color_scheme.input_text)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_border)};
                border-radius: 3px;
                padding: 5px;
            }}
            QLineEdit:focus {{
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.input_focus_border)};
            }}
        """)
        search_layout.addWidget(self.search_input, 1)  # Stretch factor 1 - can compress

        # Plate view toggle button (moved from bottom)
        self.plate_view_toggle_btn = QPushButton("Show Plate View")
        self.plate_view_toggle_btn.setCheckable(True)
        self.plate_view_toggle_btn.clicked.connect(self._toggle_plate_view)
        self.plate_view_toggle_btn.setStyleSheet(self.style_gen.generate_button_style())
        search_layout.addWidget(self.plate_view_toggle_btn, 0)  # No stretch

        # Refresh button (moved from bottom)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_images)
        self.refresh_btn.setStyleSheet(self.style_gen.generate_button_style())
        search_layout.addWidget(self.refresh_btn, 0)  # No stretch

        # Info label (moved from bottom)
        self.info_label = QLabel("No images loaded")
        self.info_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_disabled)};"
        )
        search_layout.addWidget(self.info_label, 0)  # No stretch

        layout.addLayout(search_layout)

        # Create main splitter ((tree above filters) | table | config)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = main_splitter

        # The generic table owner keeps its filter panel and table side by side;
        # contribute the folder navigator as spatial context above that exact
        # filter-panel instance.
        tree_widget = self._create_folder_tree()

        # Middle: Vertical splitter for plate view and tabs
        self.middle_splitter = QSplitter(Qt.Orientation.Vertical)

        # Plate view (initially hidden)
        from openhcs.pyqt_gui.widgets.shared.plate_view_widget import PlateViewWidget

        self.plate_view_widget = PlateViewWidget(
            color_scheme=self.color_scheme, parent=self
        )
        self.plate_view_widget.wells_selected.connect(self._on_wells_selected)
        self.plate_view_widget.detach_requested.connect(self._detach_plate_view)
        self.plate_view_widget.setVisible(False)
        self.middle_splitter.addWidget(self.plate_view_widget)

        # Single table for both images and results (no tabs needed)
        image_table_widget = self._create_table_widget()
        image_table_widget.set_column_filter_context_widget(tree_widget)
        self.middle_splitter.addWidget(image_table_widget)

        # Set initial sizes (30% plate view, 70% table when visible)
        self.middle_splitter.setSizes([150, 350])

        main_splitter.addWidget(self.middle_splitter)

        # Right: Napari config panel + instance manager
        right_panel = self._create_right_panel()
        self.right_panel = right_panel
        main_splitter.addWidget(right_panel)

        # The browser consumes the flexible space; viewer controls remain right.
        main_splitter.setSizes([2000, 400])

        # Add splitter with stretch factor to fill vertical space
        layout.addWidget(main_splitter, 1)

        # Note: Selection and double-click signals are connected in _create_table_widget()

    def _create_folder_tree(self):
        """Create folder tree widget for filtering images by directory."""
        tree = QTreeWidget()
        tree.setHeaderLabel("Folders")
        tree.setMinimumWidth(150)

        # Apply styling
        tree.setStyleSheet(self.style_gen.generate_tree_widget_style())

        # Connect selection to filter table
        tree.itemSelectionChanged.connect(self.on_folder_selection_changed)

        # Store reference
        self.folder_tree = tree

        return tree

    def _create_table_widget(self):
        """Create and configure the unified file table widget (images + results)."""
        # Use ImageTableBrowser for unified table (multi-select, dynamic columns)
        self.image_table_browser = ImageTableBrowser(
            color_scheme=self.color_scheme,
            metadata_value_formatter=self.metadata_display_resolver.display_value,
            parent=self,
        )
        self.image_table_browser.search_input.setVisible(False)
        self.image_table_browser.column_filter_selection_changed.connect(
            self._on_table_filter_selection_changed
        )

        # Connect signals
        self.image_table_browser.item_double_clicked.connect(
            self._on_file_double_clicked
        )
        self.image_table_browser.items_selected.connect(self._on_files_selected)

        return self.image_table_browser

    # Removed _create_results_widget - now using unified file table

    def _create_right_panel(self):
        """Create the right panel with streaming config forms and instance manager.

        Uses TabbedFormWidget to show each streaming config in its own tab.
        """
        container = QWidget()
        container.setMinimumWidth(
            360
        )  # Wider minimum for better config visibility (80% increase from 200)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Vertical splitter: tabbed config form on top, instance manager below
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(True)  # Allow collapsing to 0

        # Top panel: Tabbed streaming config forms
        # Create view buttons for each streaming config (will be added to tab bar row)
        header_widgets = self.viewer_controls.create_header_buttons()

        # Create a tab for each streaming config type
        tabs = []
        for field in streaming_viewer_fields():
            tabs.append(TabConfig(name=field.display_name, field_ids=[field.field_name]))

        tabbed_config = TabbedFormConfig(
            tabs=tabs,
            color_scheme=self.color_scheme,
            use_scroll_area=True,  # Each tab gets its own scroll area
            header_widgets=header_widgets,  # View buttons on same row as tabs
        )

        self.tabbed_form = TabbedFormWidget(state=self.state, config=tabbed_config)
        splitter.addWidget(self.tabbed_form)

        # Connect to parameter changes to update view button states
        self.tabbed_form.parameter_changed.connect(self._on_parameter_changed)

        # Bottom panel: Instance manager (ZMQ server browser)
        instance_panel = self._create_instance_manager_panel()
        splitter.addWidget(instance_panel)

        # Set initial splitter sizes (70% config, 30% instance manager)
        splitter.setSizes([350, 150])

        layout.addWidget(splitter, 1)  # stretch factor = 1

        return container

    def _create_instance_manager_panel(self):
        """Create the viewer instance manager panel using ZMQServerManagerWidget."""
        from openhcs.core.config import get_all_streaming_ports
        from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import (
            ZMQServerManagerWidget,
        )

        ports_to_scan = get_all_streaming_ports(
            config=self.orchestrator.pipeline_config if self.orchestrator else None,
            num_ports_per_type=self._zmq_config.ports_per_server_type,
        )

        # Create ZMQ server manager widget
        zmq_manager = ZMQServerManagerWidget(
            ports_to_scan=ports_to_scan,
            title="Viewer Instances",
            style_generator=self.style_gen,
            config=self._zmq_config,
            parent=self,
        )
        self.zmq_manager = zmq_manager
        return zmq_manager

    def set_zmq_config(self, config: OpenHCSZMQConfig) -> None:
        """Use the resolved process transport config for viewer discovery."""

        self._zmq_config = config
        if self.zmq_manager is None:
            return
        from openhcs.core.config import get_all_streaming_ports

        self.zmq_manager.set_zmq_config(
            config,
            get_all_streaming_ports(
                config=(
                    self.orchestrator.pipeline_config if self.orchestrator else None
                ),
                num_ports_per_type=config.ports_per_server_type,
            ),
        )

    def _create_state_for_orchestrator(self, orchestrator):
        """Create browser config state under the selected plate hierarchy."""
        self.scope_id = (
            f"{orchestrator.plate_path}::image_browser" if orchestrator else None
        )
        parent_state = (
            ObjectStateRegistry.get_by_scope(str(orchestrator.plate_path))
            if orchestrator
            else None
        )
        state = ObjectState(
            object_instance=self.config,
            scope_id=self.scope_id,
            parent_state=parent_state,
        )
        if self.scope_id:
            ObjectStateRegistry.register(state, _skip_snapshot=True)
        return state

    def _replace_state_for_orchestrator(self, orchestrator) -> None:
        if self.scope_id:
            ObjectStateRegistry.unregister(self.state, _skip_snapshot=True)
        self.config = ImageBrowserConfig()
        self.state = self._create_state_for_orchestrator(orchestrator)
        self.viewer_controls.state = self.state

    def _rebuild_right_panel(self) -> None:
        if self.main_splitter is None or self.right_panel is None:
            return
        old_panel = self.right_panel
        panel_index = self.main_splitter.indexOf(old_panel)
        if panel_index < 0:
            return

        self.tabbed_form = None
        self.zmq_manager = None
        new_panel = self._create_right_panel()
        self.right_panel = new_panel
        replaced_panel = self.main_splitter.replaceWidget(panel_index, new_panel)
        if replaced_panel is not None:
            replaced_panel.deleteLater()

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator and load images."""
        self.orchestrator = orchestrator
        self._replace_state_for_orchestrator(orchestrator)
        self._rebuild_right_panel()
        self.load_images()

    def focus_file_by_path(self, file_path: str | Path) -> bool:
        """Focus a loaded image/result file by semantic artifact path."""
        return self.file_focus_controller.focus_path(file_path)

    def _restore_folder_selection(self, folder_path: str, folder_items: Dict):
        """Restore folder selection after tree rebuild."""
        if folder_path in folder_items:
            item = folder_items[folder_path]
            item.setSelected(True)
            # Expand parents to make selection visible
            parent = item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()

    def on_folder_selection_changed(self):
        """Handle folder tree selection changes to filter table."""
        # Apply folder filter on top of search filter
        self.filter_controller.apply_combined_filters()

        # Update plate view for new folder
        if self.plate_view_widget and self.plate_view_widget.isVisible():
            self._update_plate_view()

    def load_images(self):
        """Load image files from the orchestrator's metadata."""
        if not self.orchestrator:
            self.info_label.setText("No plate loaded")
            return

        image_items: dict[str, ImageBrowserItem] = {}
        result_items: dict[str, ImageBrowserItem] = {}
        try:
            self.metadata_display_resolver.clear()
            logger.info("IMAGE BROWSER: Starting load_images()")
            inventory = PlateFileInventory.from_orchestrator(
                self.orchestrator,
                all_subdirs=True,
            )
            logger.info(
                "IMAGE BROWSER: plate file inventory returned %s images and %s results",
                len(inventory.image_records),
                len(inventory.result_records),
            )

            image_items, result_items = self._items_from_file_records(
                inventory.file_records()
            )

            logger.info(
                "IMAGE BROWSER: Built file item set with %s images and %s results",
                len(image_items),
                len(result_items),
            )

        except Exception as e:
            logger.error(f"Failed to load plate files: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load plate files: {e}")
            self.info_label.setText("Error loading plate files")
            image_items.clear()
            result_items.clear()

        self.file_items = {**image_items, **result_items}

        all_keys = set()
        for item in self.file_items.values():
            all_keys.update(item.metadata.keys())

        all_keys.discard("filename")
        self.metadata_keys = sorted(all_keys, key=lambda k: (k != "extension", k))

        self.image_table_browser.set_metadata_keys(self.metadata_keys)

        folder_start = time.perf_counter()
        self._build_folder_tree()
        logger.info(
            "IMAGE BROWSER: Built folder tree in %.3fs",
            time.perf_counter() - folder_start,
        )

        populate_start = time.perf_counter()
        self._set_visible_files(self.file_items, rebuild_index=True)
        logger.info(
            "IMAGE BROWSER: Populated table in %.3fs",
            time.perf_counter() - populate_start,
        )

        total_files = len(self.file_items)
        num_images = sum(1 for item in self.file_items.values() if not item.is_result)
        num_results = sum(1 for item in self.file_items.values() if item.is_result)
        self.info_label.setText(
            f"{total_files} files loaded ({num_images} images, {num_results} results)"
        )

        # Update plate view if visible
        if self.plate_view_widget and self.plate_view_widget.isVisible():
            self._update_plate_view()

    def load_results(self) -> dict[str, ImageBrowserItem]:
        """Load result files (ROI JSON, CSV) from the results directory."""
        if not self.orchestrator:
            logger.warning("IMAGE BROWSER RESULTS: No orchestrator available")
            return {}

        try:
            inventory = PlateResultFileInventory.from_orchestrator(self.orchestrator)
            return self._result_items_from_inventory(inventory)

        except Exception as e:
            logger.error(
                f"IMAGE BROWSER RESULTS: Failed to load results: {e}", exc_info=True
            )
        return {}

    @staticmethod
    def _items_from_file_records(
        file_records: tuple[PlateFileRecord, ...],
    ) -> tuple[dict[str, ImageBrowserItem], dict[str, ImageBrowserItem]]:
        """Project shared file inventory records into browser rows."""
        image_items: dict[str, ImageBrowserItem] = {}
        result_items: dict[str, ImageBrowserItem] = {}
        for record in file_records:
            if record.kind is PlateFileKind.IMAGE:
                image_items[record.key] = ImageBrowserItem(
                    key=record.key,
                    metadata=dict(record.metadata),
                )
            elif (
                record.kind is PlateFileKind.RESULT
                and record.file_format is not None
                and record.full_path is not None
            ):
                action = RESULT_FILE_ACTIONS[record.file_format]
                logger.info(
                    "IMAGE BROWSER RESULTS: matched as %s: %s",
                    action.display_name,
                    record.key,
                )
                result_items[record.key] = ImageBrowserItem(
                    key=record.key,
                    metadata=dict(record.metadata),
                    result_file_type=record.file_format,
                    full_path=Path(record.full_path),
                )
        return image_items, result_items

    @staticmethod
    def _result_items_from_inventory(
        inventory: PlateFileInventory | PlateResultFileInventory,
    ) -> dict[str, ImageBrowserItem]:
        """Project shared result-file inventory records into browser rows."""
        result_records = (
            inventory.result_records
            if isinstance(inventory, PlateFileInventory)
            else inventory.records
        )
        scanned_file_count = (
            inventory.scanned_result_file_count
            if isinstance(inventory, PlateFileInventory)
            else inventory.scanned_file_count
        )
        if not result_records:
            logger.warning("IMAGE BROWSER RESULTS: No declared analysis result files")
            return {}

        result_items: dict[str, ImageBrowserItem] = {}
        for record in result_records:
            action = RESULT_FILE_ACTIONS[record.file_format]
            logger.info(
                "IMAGE BROWSER RESULTS: matched as %s: %s",
                action.display_name,
                record.relative_path,
            )
            result_items[record.relative_path] = ImageBrowserItem(
                key=record.relative_path,
                metadata=dict(record.metadata),
                result_file_type=record.file_format,
                full_path=record.full_path_obj,
            )

        logger.info(
            "IMAGE BROWSER RESULTS: Scanned %s total files, matched %s result files",
            scanned_file_count,
            len(result_items),
        )
        return result_items

    # Removed _populate_results_table - now using unified file table
    # Removed on_result_double_clicked - now using unified on_file_double_clicked

    def _stream_roi_file(self, roi_zip_path: Path):
        """Stream ROI .roi.zip file to enabled viewer(s) asynchronously.

        This method now only performs lightweight UI-thread work:
        - Checks which viewers are enabled.
        - Resolves streaming configs and viewers.
        - Spawns background workers that do all heavy ROI loading + streaming.
        """
        try:
            # Check which viewers are enabled by querying ObjectState
            enabled_viewers = self.viewer_controls.enabled_viewers()

            if not enabled_viewers:
                QMessageBox.information(
                    self,
                    "No Viewer Enabled",
                    "Please enable at least one viewer streaming to view ROIs.",
                )
                return

            if not self.orchestrator:
                raise RuntimeError("No orchestrator set")

            from objectstate import spawn_thread_with_context

            for viewer_field_name in enabled_viewers:
                def _stream_to_viewer(field_name=viewer_field_name):
                    try:
                        self._stream_rois_to_viewer([str(roi_zip_path)], field_name)
                    except Exception as e:
                        logger.error(
                            f"Failed to start ROI streaming to {field_name}: {e}",
                            exc_info=True,
                        )
                        QTimer.singleShot(
                            0,
                            lambda field_name=field_name, e=e: QMessageBox.warning(
                                self,
                                "Error",
                                f"Failed to stream ROI to {field_name}: {e}",
                            ),
                        )

                spawn_thread_with_context(
                    _stream_to_viewer,
                    name=f"stream_roi_{viewer_field_name}",
                )

            logger.info(f"Started async streaming of ROI file {roi_zip_path.name}")

        except Exception as e:
            logger.error(f"Failed to start ROI streaming: {e}")
            QMessageBox.warning(self, "Error", f"Failed to stream ROI file: {e}")

    def _set_visible_files(
        self,
        files_dict: dict[str, ImageBrowserItem],
        *,
        rebuild_index: bool,
    ):
        """Project visible file rows into ImageTableBrowser and status text."""
        if rebuild_index:
            self.image_table_browser.set_items(files_dict)
        else:
            self.image_table_browser.set_filtered_items(files_dict)

        total = len(self.file_items)
        filtered = len(self.image_table_browser.filtered_items)
        self.image_table_browser.status_label.setText(f"Files: {filtered}/{total}")

    def _build_folder_tree(self):
        """Build folder tree from file paths (images + results)."""
        # Save current selection before rebuilding
        selected_folder = None
        selected_items = self.folder_tree.selectedItems()
        if selected_items:
            selected_folder = selected_items[0].data(0, Qt.ItemDataRole.UserRole)

        self.folder_tree.clear()

        # Extract unique folder paths (exclude *_results folders since they're auto-included)
        folders: Set[str] = set()
        for filename in self.file_items:
            path = Path(filename)
            # Add all parent directories
            for parent in path.parents:
                parent_str = str(parent)
                if parent_str != "." and not parent_str.endswith("_results"):
                    folders.add(parent_str)

        # Build tree structure
        root_item = QTreeWidgetItem(["All Files"])
        root_item.setData(0, Qt.ItemDataRole.UserRole, None)
        self.folder_tree.addTopLevelItem(root_item)

        # Sort folders for consistent display
        sorted_folders = sorted(folders)

        # Create tree items for each folder
        folder_items = {}
        for folder in sorted_folders:
            parts = Path(folder).parts
            if len(parts) == 1:
                # Top-level folder
                item = QTreeWidgetItem([folder])
                item.setData(0, Qt.ItemDataRole.UserRole, folder)
                root_item.addChild(item)
                folder_items[folder] = item
            else:
                # Nested folder - find parent
                parent_path = str(Path(folder).parent)
                if parent_path in folder_items:
                    item = QTreeWidgetItem([Path(folder).name])
                    item.setData(0, Qt.ItemDataRole.UserRole, folder)
                    folder_items[parent_path].addChild(item)
                    folder_items[folder] = item

        # Start with everything collapsed (user can expand to explore)
        root_item.setExpanded(False)

        # Restore previous selection if it still exists
        if selected_folder is not None:
            self._restore_folder_selection(selected_folder, folder_items)

    def _on_parameter_changed(self, param_name: str, value: object):
        """Handle parameter changes from the tabbed form.

        Updates view button states when streaming config 'enabled' fields change.
        """
        logger.info(
            f"🔔 ImageBrowser._on_parameter_changed: param_name={param_name}, value={value}"
        )

        # Strip leading dot if present (root PFM with field_id='' emits paths like ".napari_streaming_config.enabled")
        normalized_param = param_name.lstrip(".")

        # Streaming controls are a live surface rather than a save/cancel editor.
        # Advance the ObjectState baseline with every edit so current and saved
        # resolution stay identical and no false unsaved marker is presented.
        self.state.mark_saved()

        # Check if this is an 'enabled' field for any streaming config
        for viewer_type in _streaming_config_field_names():
            enabled_path = f"{viewer_type}.enabled"
            logger.debug(f"  Checking if {normalized_param} == {enabled_path}")
            if normalized_param == enabled_path:
                logger.info(f"  ✅ Match! Updating button state for {viewer_type}")
                self.viewer_controls.update_button_state(
                    viewer_type,
                    len(self.image_table_browser.get_selected_keys()) > 0,
                )
                break

    def _on_files_selected(self, keys: list):
        """Handle selection change from ImageTableBrowser."""
        self.viewer_controls.update_all_button_states(keys)

    def _on_file_double_clicked(self, key: str, item: ImageBrowserItem):
        """Handle double-click from ImageTableBrowser."""
        if item.is_result:
            self._handle_result_double_click(item)
        else:
            self._handle_image_double_click()

    def _handle_image_double_click(self):
        """Handle double-click on an image - stream to enabled viewer(s)."""
        # Find all enabled viewers by querying ObjectState
        enabled_viewers = self.viewer_controls.enabled_viewers()

        # Stream to whichever viewer(s) are enabled
        if enabled_viewers:
            for config_key in enabled_viewers:
                self._view_selected_in_viewer(config_key)
        else:
            # No viewers enabled - show message
            QMessageBox.information(
                self,
                "No Viewer Enabled",
                "Please enable at least one viewer streaming to view images.",
            )

    def _handle_result_double_click(self, item: ImageBrowserItem):
        """Handle double-click on a result file - stream ROIs or display CSV."""
        if item.result_file_type is None:
            raise RuntimeError(f"Image browser item {item.key!r} has no result type.")
        RESULT_FILE_ACTIONS[item.result_file_type].run(self, item.result_path())

    def _view_selected_in_viewer(self, config_key: str):
        """View all selected images in the specified viewer as a batch (builds hyperstack)."""
        selected_keys = self.image_table_browser.get_selected_keys()
        if not selected_keys:
            return

        selected_items = tuple(self.file_items[key] for key in selected_keys)
        image_filenames = [
            item.key for item in selected_items
            if not item.is_result
        ]
        roi_filenames = [
            item.key for item in selected_items
            if item.result_file_type is FileFormat.ROI
        ]

        logger.info(
            f"🎯 IMAGE BROWSER: User selected {len(image_filenames)} images and {len(roi_filenames)} ROI files to view in {config_key}"
        )
        if image_filenames:
            logger.info(
                f"🎯 IMAGE BROWSER: Image filenames: {image_filenames[:5]}{'...' if len(image_filenames) > 5 else ''}"
            )
        if roi_filenames:
            logger.info(f"🎯 IMAGE BROWSER: ROI filenames: {roi_filenames}")

        from objectstate import spawn_thread_with_context

        def _view_async():
            # Stream ROI files in a batch (get viewer once, stream all ROIs)
            if roi_filenames:
                self._stream_rois_to_viewer(roi_filenames, config_key)

            # Stream image files as a batch
            if image_filenames:
                self._stream_images_to_viewer(image_filenames, config_key)

        spawn_thread_with_context(_view_async, name=f"view_{config_key}")

    def _prepare_streaming(self, config_key: str) -> tuple:
        """Prepare for streaming: resolve config, get viewer, get read backend.

        Returns: (viewer, read_backend, config)
        """
        if not self.orchestrator:
            raise RuntimeError("No orchestrator set")

        plate_path = Path(self.orchestrator.plate_path)

        # Resolve backend
        read_backend = self.orchestrator.microscope_handler.get_primary_backend(
            plate_path, self.orchestrator.filemanager
        )

        # Get fully resolved streaming config from ObjectState (includes inheritance)
        # get_resolved_value now returns reconstructed dataclass with all sub-fields populated
        config = self.state.get_resolved_value(config_key)

        viewer = self.orchestrator.get_or_create_visualizer(config)
        return viewer, read_backend, config

    def _stream_images_to_viewer(self, filenames: list, config_key: str):
        """Load and stream images to specified viewer type."""
        viewer, read_backend, config = self._prepare_streaming(config_key)
        from openhcs.core.viewer_streaming_service import (
            ImageStreamingRequest,
        )

        streaming_service = self.streaming_service
        if streaming_service is None:
            raise RuntimeError("No orchestrator set")

        streaming_service.stream_images_async(
            ImageStreamingRequest(
                viewer=viewer,
                config=config,
                status_callback=self._status_update_signal.emit,
                error_callback=lambda e: self._show_streaming_error(
                    config.display_name,
                    e,
                ),
                filenames=tuple(filenames),
                read_backend=read_backend,
            )
        )
        logger.info(f"Streaming {len(filenames)} images to {config.display_name}...")

    def _show_streaming_error(self, viewer_name: str, error_msg: str):
        """Show streaming error in UI thread."""
        QMessageBox.warning(
            self,
            "Streaming Error",
            f"Failed to stream images to {viewer_name}: {error_msg}",
        )

    def _stream_rois_to_viewer(self, roi_filenames: list, config_key: str):
        """Stream ROI files to specified viewer type."""
        viewer, _, config = self._prepare_streaming(config_key)
        from openhcs.core.viewer_streaming_service import (
            RoiStreamingRequest,
        )

        streaming_service = self.streaming_service
        if streaming_service is None:
            raise RuntimeError("No orchestrator set")

        streaming_service.stream_rois_async(
            RoiStreamingRequest(
                viewer=viewer,
                config=config,
                status_callback=self._status_update_signal.emit,
                error_callback=lambda e: self._show_streaming_error(
                    config.display_name,
                    e,
                ),
                roi_filenames=tuple(roi_filenames),
            )
        )
        logger.info(f"Streaming {len(roi_filenames)} ROI files to {config.display_name}...")

    def cleanup(self):
        """Clean up resources before widget destruction."""
        # Cleanup ZMQ server manager widget (always initialized to None in __init__)
        if self.zmq_manager is not None:
            self.zmq_manager.cleanup()
        if self.scope_id:
            ObjectStateRegistry.unregister(self.state, _skip_snapshot=True)
            self.scope_id = None

    # ========== Plate View Methods ==========

    def _toggle_plate_view(self, checked: bool):
        """Toggle plate view visibility."""
        # If detached, just show/hide the window
        if self.plate_view_detached_window:
            self.plate_view_detached_window.setVisible(checked)
            if checked:
                self.plate_view_toggle_btn.setText("Hide Plate View")
            else:
                self.plate_view_toggle_btn.setText("Show Plate View")
            return

        # Otherwise toggle in main layout
        self.plate_view_widget.setVisible(checked)

        if checked:
            self.plate_view_toggle_btn.setText("Hide Plate View")
            # Update plate view with current images
            self._update_plate_view()
        else:
            self.plate_view_toggle_btn.setText("Show Plate View")

    def _detach_plate_view(self):
        """Detach plate view to external window."""
        if self.plate_view_detached_window:
            # Already detached, just show it
            self.plate_view_detached_window.show()
            self.plate_view_detached_window.raise_()
            return

        from PyQt6.QtWidgets import QDialog

        # Create detached window
        self.plate_view_detached_window = QDialog(self)
        self.plate_view_detached_window.setWindowTitle("Plate View")
        self.plate_view_detached_window.setWindowFlags(Qt.WindowType.Dialog)
        self.plate_view_detached_window.setMinimumSize(600, 400)
        self.plate_view_detached_window.resize(800, 600)

        # Create layout for window
        window_layout = QVBoxLayout(self.plate_view_detached_window)
        window_layout.setContentsMargins(10, 10, 10, 10)

        # Add reattach button
        reattach_btn = QPushButton("⬅ Reattach to Main Window")
        reattach_btn.setStyleSheet(self.style_gen.generate_button_style())
        reattach_btn.clicked.connect(self._reattach_plate_view)
        window_layout.addWidget(reattach_btn)

        # Move plate view widget to window
        self.plate_view_widget.setParent(self.plate_view_detached_window)
        self.plate_view_widget.setVisible(True)
        window_layout.addWidget(self.plate_view_widget)

        # Connect close event to reattach
        self.plate_view_detached_window.closeEvent = (
            lambda event: self._on_detached_window_closed(event)
        )

        # Show window
        self.plate_view_detached_window.show()

        logger.info("Plate view detached to external window")

    def _reattach_plate_view(self):
        """Reattach plate view to main layout."""
        if not self.plate_view_detached_window:
            return

        # Store reference before clearing
        window = self.plate_view_detached_window
        self.plate_view_detached_window = None

        # Move plate view widget back to splitter
        self.plate_view_widget.setParent(self)
        self.middle_splitter.insertWidget(0, self.plate_view_widget)
        self.plate_view_widget.setVisible(self.plate_view_toggle_btn.isChecked())

        # Close and cleanup detached window
        window.close()
        window.deleteLater()

        logger.info("Plate view reattached to main window")

    def _on_detached_window_closed(self, event):
        """Handle detached window close event - reattach automatically."""
        # Only reattach if window still exists (not already reattached)
        if self.plate_view_detached_window:
            # Clear reference first to prevent double-close
            window = self.plate_view_detached_window
            self.plate_view_detached_window = None

            # Move plate view widget back to splitter
            self.plate_view_widget.setParent(self)
            self.middle_splitter.insertWidget(0, self.plate_view_widget)
            self.plate_view_widget.setVisible(self.plate_view_toggle_btn.isChecked())

            logger.info("Plate view reattached to main window (window closed)")

        event.accept()

    def _on_wells_selected(self, well_ids: Set[str]):
        """Handle well selection from plate view."""
        logger.info(f"[WELLS_SELECTED] Received {len(well_ids)} wells: {well_ids}")
        well_key = AllComponents.WELL.value
        self._syncing_plate_filter_selection = True
        try:
            synced = self.image_table_browser.set_column_filter_selection(
                well_key,
                (
                    self.metadata_display_resolver.display_values(well_key, well_ids)
                    if well_ids
                    else None
                ),
            )
        finally:
            self._syncing_plate_filter_selection = False
        self.selected_wells = set() if synced else well_ids
        self.filter_controller.apply_combined_filters()

    def _on_table_filter_selection_changed(
        self,
        column_key: str,
        selected_values: frozenset[str],
    ) -> None:
        """Compose the generic Well filter selection into the plate view."""
        well_key = AllComponents.WELL.value
        if (
            self._syncing_plate_filter_selection
            or column_key != well_key
            or self.plate_view_widget is None
        ):
            return
        self.selected_wells = set()
        self.plate_view_widget.select_wells(
            self.metadata_display_resolver.raw_values(
                well_key,
                set(selected_values),
            ),
            emit_signal=False,
        )
        self.filter_controller.apply_combined_filters()

    def _update_plate_view(self):
        """Update plate view with current file data (images + results)."""
        # Extract all well IDs from current files (filter out failures)
        well_ids = set()
        for filename, item in self.file_items.items():
            try:
                well_id = self._extract_well_id(item.metadata)
                well_ids.add(well_id)
            except (KeyError, ValueError):
                # Skip files without well metadata (e.g., plate-level files)
                pass

        # Detect plate dimensions and build coordinate mapping
        plate_dimensions = self._detect_plate_dimensions(well_ids) if well_ids else None

        # Build mapping from (row_index, col_index) to actual well_id
        # This handles different well ID formats (A01 vs R01C01)
        coord_to_well = {}
        parser = self.orchestrator.microscope_handler.parser
        for well_id in well_ids:
            row, col = parser.extract_component_coordinates(well_id)
            # Convert row letter to index (A=1, B=2, etc.)
            row_idx = sum(
                (ord(c.upper()) - ord("A") + 1) * (26**i)
                for i, c in enumerate(reversed(row))
            )
            coord_to_well[(row_idx, int(col))] = well_id

        # Update plate view with well IDs, dimensions, and coordinate mapping
        self.plate_view_widget.set_available_wells(
            well_ids, plate_dimensions, coord_to_well
        )

        # Handle subdirectory selection
        current_folder = self._get_current_folder()
        subdirs = self._detect_plate_subdirs(current_folder)
        self.plate_view_widget.set_subdirectories(subdirs)

    def _get_current_folder(self) -> Optional[str]:
        """Get currently selected folder path from tree."""
        selected_items = self.folder_tree.selectedItems()
        if selected_items:
            folder_path = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
            return folder_path
        return None

    def _detect_plate_subdirs(self, current_folder: Optional[str]) -> List[str]:
        """
        Detect plate output subdirectories.

        Logic:
        - If at plate root (no folder selected or root selected), look for subdirs with well images
        - If in a subdir, return empty list (already in a plate output)

        Returns list of subdirectory names (not full paths).
        """
        if not self.orchestrator:
            return []

        plate_path = self.orchestrator.plate_path

        # If no folder selected or root selected, we're at plate root
        if current_folder is None:
            base_path = plate_path
        else:
            # Check if current folder is plate root
            if str(Path(current_folder)) == str(plate_path):
                base_path = plate_path
            else:
                # Already in a subdirectory, no subdirs to show
                return []

        # Find immediate subdirectories that contain well files
        subdirs_with_wells = set()

        for filename, item in self.file_items.items():
            file_path = Path(filename)

            # Check if file is in a subdirectory of base_path
            try:
                relative = file_path.relative_to(base_path)
                parts = relative.parts

                # If file is in a subdirectory (not directly in base_path)
                if len(parts) > 1:
                    subdir_name = parts[0]

                    # Check if this file has well metadata
                    try:
                        self._extract_well_id(item.metadata)
                        # Has well metadata, add subdir
                        subdirs_with_wells.add(subdir_name)
                    except (KeyError, ValueError):
                        # No well metadata, skip
                        pass
            except ValueError:
                # File not relative to base_path, skip
                pass

        return sorted(list(subdirs_with_wells))

    # ========== Plate View Helper Methods ==========

    def _extract_well_id(self, metadata: dict) -> str:
        """
        Extract well ID from metadata.

        Returns well ID like 'A01', 'B03', 'R01C03', etc.
        Raises KeyError if metadata missing 'well' component.
        """
        # Well ID is a single component in metadata
        return str(metadata["well"])

    def _detect_plate_dimensions(self, well_ids: Set[str]) -> tuple[int, int]:
        """
        Auto-detect plate dimensions from well IDs.

        Uses existing infrastructure:
        - FilenameParser.extract_component_coordinates() to parse each well ID
        - Determines max row/col from parsed coordinates

        Returns (rows, cols) tuple.
        Raises ValueError if well IDs are invalid format.
        """
        parser = self.orchestrator.microscope_handler.parser

        rows = set()
        cols = set()

        for well_id in well_ids:
            # REUSE: Parser's extract_component_coordinates (fail loud if invalid)
            row, col = parser.extract_component_coordinates(well_id)
            rows.add(row)
            cols.add(int(col))

        # Convert row letters to indices (A=1, B=2, AA=27, etc.)
        row_indices = [
            sum(
                (ord(c.upper()) - ord("A") + 1) * (26**i)
                for i, c in enumerate(reversed(row))
            )
            for row in rows
        ]

        return (max(row_indices), max(cols))

    def _update_status_threadsafe(self, message: str):
        """Update status label from any thread (thread-safe).

        Args:
            message: Status message to display
        """
        self._status_update_signal.emit(message)

    @pyqtSlot(str)
    def _update_status_label(self, message: str):
        """Update status label (called on main thread via signal)."""
        self.info_label.setText(message)
