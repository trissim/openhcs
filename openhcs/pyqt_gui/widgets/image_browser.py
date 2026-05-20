"""
Image Browser Widget for PyQt6 GUI.

Displays a table of all image files from plate metadata and allows users to
view them in Napari with configurable display settings.
"""

import logging
import time
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, List, Dict, Set, Any

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidgetItem,
    QPushButton,
    QLabel,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QSplitter,
    QGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
    QScrollArea,
    QLineEdit,
    QTabWidget,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer

from openhcs.constants.constants import Backend, FileFormat
from polystore.filemanager import FileManager
from polystore.base import storage_registry
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.widgets.shared.column_filter_widget import MultiColumnFilterPanel
from pyqt_reactive.widgets.shared.image_table_browser import ImageTableBrowser
from pyqt_reactive.widgets.shared import TabbedFormWidget, TabConfig, TabbedFormConfig
from openhcs.config_framework.object_state import ObjectState, ObjectStateRegistry
from openhcs.core.config import StreamingConfig
from objectstate.lazy_factory import get_base_type_for_lazy
from pyqt_reactive.forms import ParameterFormManager, FormManagerConfig

logger = logging.getLogger(__name__)


STREAMING_CONFIG_FIELD_NAMES = tuple(StreamingConfig.__registry__.keys())


def _streaming_config_field_names() -> tuple[str, ...]:
    """Registered streaming-config field names used by image-browser controls."""
    return STREAMING_CONFIG_FIELD_NAMES


class ResultFileType(str, Enum):
    """Result-file types shown by the image browser."""
    ROI = "ROI"
    CSV = "CSV"
    JSON = "JSON"

    @classmethod
    def from_path(cls, file_path: Path) -> Optional["ResultFileType"]:
        suffix = file_path.suffix.lower()
        if file_path.name.endswith(".roi.zip"):
            return cls.ROI
        if suffix in FileFormat.CSV.value:
            return cls.CSV
        if suffix in FileFormat.JSON.value:
            return cls.JSON
        return None


@dataclass(frozen=True)
class ResultFileAction:
    """Double-click behavior for one result-file type."""
    file_type: ResultFileType
    display_name: str
    handle: Callable[["ImageBrowserWidget", Path], None]

    def run(self, browser: "ImageBrowserWidget", file_path: Path) -> None:
        self.handle(browser, file_path)


def _stream_roi_result(browser: "ImageBrowserWidget", file_path: Path) -> None:
    browser._stream_roi_file(file_path)


def _open_result_in_default_app(browser: "ImageBrowserWidget", file_path: Path) -> None:
    subprocess.run(["xdg-open", str(file_path)])


RESULT_FILE_ACTIONS = {
    ResultFileType.ROI: ResultFileAction(
        file_type=ResultFileType.ROI,
        display_name="ROI",
        handle=_stream_roi_result,
    ),
    ResultFileType.CSV: ResultFileAction(
        file_type=ResultFileType.CSV,
        display_name="CSV",
        handle=_open_result_in_default_app,
    ),
    ResultFileType.JSON: ResultFileAction(
        file_type=ResultFileType.JSON,
        display_name="JSON",
        handle=_open_result_in_default_app,
    ),
}


@dataclass(frozen=True)
class StreamingViewerField:
    """Display metadata for one registered streaming viewer field."""

    field_name: str

    @property
    def display_name(self) -> str:
        from openhcs.ui.shared.streaming_service import StreamingService

        return StreamingService.display_name_for_viewer_type(self.field_name)


STREAMING_VIEWER_FIELDS = tuple(
    StreamingViewerField(field_name)
    for field_name in _streaming_config_field_names()
)


def streaming_viewer_fields() -> tuple[StreamingViewerField, ...]:
    """Registered streaming-viewer fields with display metadata."""
    return STREAMING_VIEWER_FIELDS


def _create_image_browser_config():
    """Create ImageBrowser config container with LAZY streaming configs.

    Uses SimpleNamespace with dynamically injected LAZY streaming configs.
    Lazy configs resolve values from parent_state (plate) through the
    ObjectState hierarchy, enabling live context updates.
    """
    from types import SimpleNamespace

    config = SimpleNamespace()

    # Auto-discover streaming configs from registry
    # Registry keys are now snake_case field names (e.g., 'napari_streaming_config')
    for field_name in _streaming_config_field_names():
        lazy_class = StreamingConfig.__registry__[field_name]
        instance = lazy_class()  # Lazy config resolves from plate via parent_state
        setattr(config, field_name, instance)

    return config


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

    def is_enabled(self, viewer_type: str) -> bool:
        enabled_path = f"{viewer_type}.enabled"
        return self.state.get_resolved_value(enabled_path) is True

    def enabled_viewers(self) -> list[str]:
        return [
            field.field_name
            for field in streaming_viewer_fields()
            if self.is_enabled(field.field_name)
        ]

    def update_button_state(self, viewer_type: str, has_selection: bool) -> None:
        button = self.buttons.get(viewer_type)
        if button is None:
            logger.warning("viewer_type %s not in view buttons", viewer_type)
            return
        button.setEnabled(has_selection and self.is_enabled(viewer_type))

    def update_all_button_states(self, selected_keys: list) -> None:
        has_selection = len(selected_keys) > 0
        for viewer_type in self.buttons:
            self.update_button_state(viewer_type, has_selection)


class ImageBrowserMetadataDisplayResolver:
    """Resolve and cache display values for metadata table/filter cells."""

    COMPONENT_KEY_MAP = {
        "channel": "CHANNEL",
        "site": "SITE",
        "z_index": "Z_INDEX",
        "timepoint": "TIMEPOINT",
        "well": "WELL",
    }

    def __init__(self, orchestrator_getter: Callable[[], object | None]):
        self.orchestrator_getter = orchestrator_getter
        self._cache: dict[tuple[str, str], str] = {}

    def clear(self) -> None:
        self._cache.clear()

    def display_value(self, metadata_key: str, raw_value: Any) -> str:
        if raw_value is None:
            return "N/A"

        value_str = str(raw_value)
        cache_key = (metadata_key, value_str)
        cached_value = self._cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        display_value = self._resolve_display_value(metadata_key, value_str)
        self._cache[cache_key] = display_value
        return display_value

    def _resolve_display_value(self, metadata_key: str, value_str: str) -> str:
        orchestrator = self.orchestrator_getter()
        if orchestrator is None:
            return value_str

        try:
            from openhcs.constants import AllComponents

            component_name = self.COMPONENT_KEY_MAP.get(metadata_key)
            component = getattr(AllComponents, component_name) if component_name else None
            if component is None:
                return value_str

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


class ImageBrowserResultCatalog:
    """Own result-file metadata and full-path lookup as one catalog."""

    def __init__(self):
        self.metadata_by_key: dict[str, dict] = {}
        self.path_by_key: dict[str, Path] = {}

    def clear(self) -> None:
        self.metadata_by_key.clear()
        self.path_by_key.clear()

    def add(self, key: str, metadata: dict, full_path: Path) -> None:
        self.metadata_by_key[key] = metadata
        self.path_by_key[key] = full_path

    def __len__(self) -> int:
        return len(self.metadata_by_key)

    def is_result(self, key: str) -> bool:
        return key in self.path_by_key

    def path_for(self, key: str) -> Path:
        return self.path_by_key[key]


class ImageBrowserImageCatalog:
    """Own loaded image metadata before image/result merge."""

    def __init__(self):
        self.metadata_by_key: dict[str, dict] = {}

    def clear(self) -> None:
        self.metadata_by_key.clear()

    def add(self, key: str, metadata: dict) -> None:
        self.metadata_by_key[key] = metadata

    def __len__(self) -> int:
        return len(self.metadata_by_key)


class ImageBrowserFilterController:
    """Own search, folder, well, and column filtering for ImageBrowserWidget."""

    def __init__(self, browser: "ImageBrowserWidget"):
        self.browser = browser
        self._search_service = None

    def apply_combined_filters(self) -> None:
        """Apply search, folder, well, and column filters in one pass."""
        browser = self.browser
        selected_items = browser.folder_tree.selectedItems()
        folder_path = None
        results_folder_path = None
        if selected_items:
            folder_path = selected_items[0].data(0, Qt.ItemDataRole.UserRole)
            if folder_path:
                results_folder_path = f"{folder_path}_results"

        active_filters = self._active_column_filters()

        result = {}
        for filename, metadata in browser.filtered_files.items():
            include = True

            if folder_path and include:
                include = (
                    str(Path(filename).parent) == folder_path
                    or filename.startswith(folder_path + "/")
                    or str(Path(filename).parent) == results_folder_path
                    or filename.startswith(results_folder_path + "/")
                )

            if browser.selected_wells and include:
                include = self._matches_wells(filename, metadata)

            if active_filters and include:
                include = self._matches_column_filters(metadata, active_filters)

            if include:
                result[filename] = metadata

        browser._update_table_with_filtered_items(result)
        logger.debug("Combined filters: %s images shown", len(result))

    def precompute_display_values(self) -> None:
        """Pre-compute display values for all metadata keys in all files."""
        browser = self.browser
        for metadata in browser.all_files.values():
            for metadata_key in browser.metadata_keys:
                raw_value = metadata.get(metadata_key)
                if raw_value is not None:
                    display_value = (
                        browser.metadata_display_resolver.display_value(
                            metadata_key, raw_value
                        )
                    )
                    metadata[f"_display_{metadata_key}"] = display_value

    def build_column_filters(self) -> None:
        """Build column filter widgets from loaded file metadata."""
        browser = self.browser
        if not browser.all_files or not browser.metadata_keys:
            return

        browser.column_filter_panel.clear_all_filters()

        for metadata_key in browser.metadata_keys:
            unique_values = {
                browser.metadata_display_resolver.display_value(metadata_key, value)
                for metadata in browser.all_files.values()
                if (value := metadata.get(metadata_key)) is not None
            }

            if unique_values:
                column_display_name = metadata_key.replace("_", " ").title()
                browser.column_filter_panel.add_column_filter(
                    column_display_name, sorted(unique_values)
                )

        if browser.column_filter_panel.column_filters:
            browser.column_filter_panel.setVisible(True)

        if (
            "Well" in browser.column_filter_panel.column_filters
            and browser.plate_view_widget
        ):
            well_filter = browser.column_filter_panel.column_filters["Well"]
            browser.plate_view_widget.well_filter_widget = well_filter
            well_filter.filter_changed.connect(self.on_well_filter_changed)

        logger.debug(
            "Built %s column filters",
            len(browser.column_filter_panel.column_filters),
        )

    def filter_images(self, search_term: str) -> None:
        """Filter files using shared search service."""
        from openhcs.ui.shared.search_service import SearchService

        browser = self.browser
        if self._search_service is None:
            self._search_service = SearchService(
                all_items=browser.all_files,
                searchable_text_extractor=self._searchable_text,
            )
        else:
            self._search_service.update_items(browser.all_files)

        browser.filtered_files = self._search_service.filter(search_term)
        self.apply_combined_filters()

    def on_well_filter_changed(self) -> None:
        """Sync well checkbox changes back to plate view and table filters."""
        if self.browser.plate_view_widget:
            self.browser.plate_view_widget.sync_from_well_filter()
        self.apply_combined_filters()

    def _active_column_filters(self) -> dict[str, set] | None:
        browser = self.browser
        if not browser.column_filter_panel:
            return None

        active_filters = browser.column_filter_panel.get_active_filters()
        if active_filters and browser.selected_wells and "Well" in active_filters:
            active_filters = {
                key: value
                for key, value in active_filters.items()
                if key != "Well"
            }
            logger.debug(
                "[FILTER] Skipping Well column filter (plate view is filtering)"
            )
        return active_filters

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

    @staticmethod
    def _matches_column_filters(metadata: dict, active_filters: dict[str, set]) -> bool:
        for column_name, selected_values in active_filters.items():
            metadata_key = column_name.lower().replace(" ", "_")
            item_value = metadata.get(f"_display_{metadata_key}", "")
            if item_value not in selected_values:
                return False
        return True

    @staticmethod
    def _searchable_text(metadata: dict) -> str:
        searchable_fields = [metadata["filename"]]
        searchable_fields.extend(
            str(value)
            for key, value in metadata.items()
            if key != "filename" and value is not None
        )
        return " ".join(str(field) for field in searchable_fields)


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
        self, orchestrator=None, color_scheme: Optional[ColorScheme] = None, parent=None
    ):
        super().__init__(parent)

        self.orchestrator = orchestrator
        self.color_scheme = color_scheme or ColorScheme()
        self.style_gen = StyleSheetGenerator(self.color_scheme)
        # Fallback for standalone browsing; orchestrator-owned runs derive their
        # FileManager from the orchestrator property below.
        self._fallback_filemanager = FileManager(storage_registry)

        # Scope ID for cross-window live context filtering (make distinct from PipelineConfig window)
        # Append a suffix so image browser uses a separate scope per plate
        self.scope_id: Optional[str] = (
            f"{orchestrator.plate_path}::image_browser" if orchestrator else None
        )

        # Create root ObjectState from dynamically generated config container
        # This gives us a single registered state with nested configs via dotted paths
        self.config = _create_image_browser_config()
        parent_state = (
            ObjectStateRegistry.get_by_scope(self.scope_id) if self.scope_id else None
        )
        self.state = ObjectState(
            object_instance=self.config,
            scope_id=self.scope_id,
            parent_state=parent_state,
        )
        # Register in ObjectStateRegistry for cross-window inheritance
        # Use _skip_snapshot=True since this is hidden machinery, not user-facing state
        if self.scope_id:
            ObjectStateRegistry.register(self.state, _skip_snapshot=True)

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
        self.all_files = {}  # filename -> metadata dict (merged images + results)
        self.image_catalog = ImageBrowserImageCatalog()
        self.result_catalog = ImageBrowserResultCatalog()
        self.filtered_files = {}  # filename -> metadata dict (after search/filter)
        self.selected_wells = set()  # Selected wells for filtering
        self.metadata_keys = []  # Column names from parser metadata (union of all keys)
        self.metadata_display_resolver = ImageBrowserMetadataDisplayResolver(
            lambda: self.orchestrator
        )
        self.filter_controller = ImageBrowserFilterController(self)

        # Plate view widget (will be created in init_ui)
        self.plate_view_widget = None
        self.plate_view_detached_window = None
        self.middle_splitter = None  # Reference to splitter for reattaching

        # Column filter panel
        self.column_filter_panel = None

        # ZMQ manager widget (may be created in init_ui)
        self.zmq_manager = None

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
            from openhcs.ui.shared.streaming_service import StreamingService

            self._streaming_service_cache = StreamingService(
                filemanager=self.filemanager,
                microscope_handler=self.orchestrator.microscope_handler,
                plate_path=self.orchestrator.plate_path,
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

        # Create main splitter (tree+filters | table | config)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Vertical splitter for Folder tree + Column filters
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Folder tree
        tree_widget = self._create_folder_tree()
        left_splitter.addWidget(tree_widget)

        # Column filter panel (initially empty, populated when images load)
        # DO NOT wrap in scroll area - breaks splitter resizing!
        # Each filter has its own scroll area for checkboxes
        self.column_filter_panel = MultiColumnFilterPanel(
            color_scheme=self.color_scheme
        )
        self.column_filter_panel.filters_changed.connect(
            self.filter_controller.apply_combined_filters
        )
        self.column_filter_panel.setVisible(False)  # Hidden until images load
        left_splitter.addWidget(self.column_filter_panel)

        # Set initial sizes: filters get more space (20% tree, 80% filters)
        left_splitter.setSizes([100, 400])

        main_splitter.addWidget(left_splitter)

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
        self.middle_splitter.addWidget(image_table_widget)

        # Set initial sizes (30% plate view, 70% table when visible)
        self.middle_splitter.setSizes([150, 350])

        main_splitter.addWidget(self.middle_splitter)

        # Right: Napari config panel + instance manager
        right_panel = self._create_right_panel()
        main_splitter.addWidget(right_panel)

        # Set initial splitter sizes (100px left, flexible middle, 400px right)
        # Middle uses large value so it takes remaining space proportionally
        main_splitter.setSizes([100, 2000, 400])

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
            color_scheme=self.color_scheme, parent=self
        )

        # Connect signals
        self.image_table_browser.item_double_clicked.connect(
            self._on_file_double_clicked
        )
        self.image_table_browser.items_selected.connect(self._on_files_selected)

        # Alias for backward compatibility during transition
        self.file_table = self.image_table_browser.table_widget

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
        from openhcs.pyqt_gui.widgets.shared.zmq_server_manager import (
            ZMQServerManagerWidget,
        )
        from openhcs.core.config import get_all_streaming_ports

        # Scan all streaming ports using orchestrator's pipeline config
        # This ensures we find viewers launched with custom ports
        # Exclude execution server port (only want viewer ports)
        from openhcs.constants.constants import DEFAULT_EXECUTION_SERVER_PORT

        all_ports = get_all_streaming_ports(
            config=self.orchestrator.pipeline_config if self.orchestrator else None,
            num_ports_per_type=10,
        )
        ports_to_scan = [p for p in all_ports if p != DEFAULT_EXECUTION_SERVER_PORT]

        # Create ZMQ server manager widget
        zmq_manager = ZMQServerManagerWidget(
            ports_to_scan=ports_to_scan,
            title="Viewer Instances",
            style_generator=self.style_gen,
            parent=self,
        )

        return zmq_manager

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator and load images."""
        self.orchestrator = orchestrator
        # CRITICAL: Preserve ::image_browser suffix to avoid scope conflicts with ConfigWindow
        self.scope_id = (
            f"{orchestrator.plate_path}::image_browser" if orchestrator else None
        )

        # Update state context and scope_id to use new pipeline_config
        if self.state and orchestrator:
            self.state.context_obj = orchestrator.pipeline_config
            self.state.scope_id = self.scope_id
            # Refresh form placeholders for all PFMs in tabs
            if self.tabbed_form:
                for form in self.tabbed_form.get_all_forms():
                    form._refresh_all_placeholders()

        self.load_images()

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
            # Still try to load results even if no orchestrator
            self.load_results()
            return

        try:
            self.metadata_display_resolver.clear()
            logger.info("IMAGE BROWSER: Starting load_images()")
            # Get metadata handler from orchestrator
            handler = self.orchestrator.microscope_handler
            metadata_handler = handler.metadata_handler
            logger.info(
                f"IMAGE BROWSER: Got metadata handler: {type(metadata_handler).__name__}"
            )

            # Get image files from metadata (all subdirectories for browsing)
            plate_path = self.orchestrator.plate_path
            logger.info(
                f"IMAGE BROWSER: Calling get_image_files for plate: {plate_path}"
            )
            image_files = metadata_handler.get_image_files(plate_path, all_subdirs=True)
            logger.info(
                f"IMAGE BROWSER: get_image_files returned {len(image_files) if image_files else 0} files"
            )

            if not image_files:
                self.info_label.setText("No images found")
                # Still load results even if no images
                self.load_results()
                return

            self.image_catalog.clear()
            for filename in image_files:
                parsed = handler.parser.parse_filename(filename)

                # Get file size
                file_path = plate_path / filename
                if file_path.exists():
                    size_bytes = file_path.stat().st_size
                    if size_bytes < 1024:
                        size_str = f"{size_bytes} B"
                    elif size_bytes < 1024 * 1024:
                        size_str = f"{size_bytes / 1024:.1f} KB"
                    else:
                        size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = "N/A"

                metadata = {"filename": filename, "type": "Image", "size": size_str}
                if parsed:
                    metadata.update(parsed)
                self.image_catalog.add(filename, metadata)

            logger.info(
                f"IMAGE BROWSER: Built image catalog with {len(self.image_catalog)} entries"
            )

        except Exception as e:
            logger.error(f"Failed to load images: {e}", exc_info=True)
            QMessageBox.warning(self, "Error", f"Failed to load images: {e}")
            self.info_label.setText("Error loading images")
            self.image_catalog.clear()

        # Load results and merge with images
        self.load_results()

        # Merge images and results into unified all_files dictionary
        self.all_files = {
            **self.image_catalog.metadata_by_key,
            **self.result_catalog.metadata_by_key,
        }

        # Determine metadata keys from all files (union of all keys)
        all_keys = set()
        for file_metadata in self.all_files.values():
            all_keys.update(file_metadata.keys())

        # Remove 'filename' from keys (it's always the first column)
        all_keys.discard("filename")

        # Sort keys for consistent column order (extension first, then alphabetical)
        self.metadata_keys = sorted(all_keys, key=lambda k: (k != "extension", k))

        # Configure ImageTableBrowser with dynamic columns
        self.image_table_browser.set_metadata_keys(self.metadata_keys)

        # Pre-compute display values for fast filtering
        self.filter_controller.precompute_display_values()

        # Initialize filtered files to all files
        self.filtered_files = self.all_files.copy()

        # Build folder tree from file paths
        folder_start = time.perf_counter()
        self._build_folder_tree()
        logger.info(
            "IMAGE BROWSER: Built folder tree in %.3fs",
            time.perf_counter() - folder_start,
        )

        # Populate table first to keep UI responsive
        populate_start = time.perf_counter()
        self._populate_table(self.filtered_files)
        logger.info(
            "IMAGE BROWSER: Populated table in %.3fs",
            time.perf_counter() - populate_start,
        )

        # Build column filters after initial render
        def _build_filters_later():
            filters_start = time.perf_counter()
            self.filter_controller.build_column_filters()
            logger.info(
                "IMAGE BROWSER: Built column filters in %.3fs",
                time.perf_counter() - filters_start,
            )

        QTimer.singleShot(0, _build_filters_later)

        # Update info label
        total_files = len(self.all_files)
        num_images = len(self.image_catalog)
        num_results = len(self.result_catalog)
        self.info_label.setText(
            f"{total_files} files loaded ({num_images} images, {num_results} results)"
        )

        # Update plate view if visible
        if self.plate_view_widget and self.plate_view_widget.isVisible():
            self._update_plate_view()

    def load_results(self):
        """Load result files (ROI JSON, CSV) from the results directory."""
        self.result_catalog.clear()

        if not self.orchestrator:
            logger.warning("IMAGE BROWSER RESULTS: No orchestrator available")
            return

        try:
            # Get results directory from metadata (single source of truth)
            # The metadata contains the results_dir field that was calculated during compilation
            handler = self.orchestrator.microscope_handler
            plate_path = self.orchestrator.plate_path

            # Load metadata JSON directly
            from openhcs.microscopes.openhcs import get_metadata_path
            import json

            metadata_path = get_metadata_path(plate_path)
            if not metadata_path.exists():
                logger.warning(
                    f"IMAGE BROWSER RESULTS: Metadata file not found: {metadata_path}"
                )
                self.result_catalog.clear()
                return

            with open(metadata_path) as f:
                metadata = json.load(f)

            # Collect ALL results directories from ALL subdirectories
            results_dirs = []
            if metadata and "subdirectories" in metadata:
                for subdir_name, subdir_metadata in metadata["subdirectories"].items():
                    if (
                        "results_dir" in subdir_metadata
                        and subdir_metadata["results_dir"]
                    ):
                        results_dir_path = plate_path / subdir_metadata["results_dir"]
                        results_dirs.append((subdir_name, results_dir_path))
                        logger.info(
                            f"IMAGE BROWSER RESULTS: Found results_dir for subdirectory '{subdir_name}': {subdir_metadata['results_dir']}"
                        )

            if not results_dirs:
                logger.warning(
                    "IMAGE BROWSER RESULTS: No results_dir found in any subdirectory"
                )
                return

            logger.info(
                f"IMAGE BROWSER RESULTS: Scanning {len(results_dirs)} results directories"
            )

            # Get parser from orchestrator for filename parsing
            handler = self.orchestrator.microscope_handler

            # Scan all results directories
            file_count = 0
            for subdir_name, results_dir in results_dirs:
                if not results_dir.exists():
                    logger.warning(
                        f"IMAGE BROWSER RESULTS: Results directory does not exist: {results_dir}"
                    )
                    continue

                logger.info(
                    f"IMAGE BROWSER RESULTS: Scanning results directory for '{subdir_name}': {results_dir}"
                )

                # Scan for ROI JSON files and CSV files
                for file_path in results_dir.rglob("*"):
                    if file_path.is_file():
                        file_count += 1
                        suffix = file_path.suffix.lower()
                        logger.debug(
                            f"IMAGE BROWSER RESULTS: Found file: {file_path.name} (suffix={suffix})"
                        )

                        result_file_type = ResultFileType.from_path(file_path)
                        if result_file_type is not None:
                            action = RESULT_FILE_ACTIONS[result_file_type]
                            logger.info(
                                f"IMAGE BROWSER RESULTS: ✓ Matched as {action.display_name}: {file_path.name}"
                            )
                        if result_file_type is None:
                            logger.debug(
                                f"IMAGE BROWSER RESULTS: ✗ Filtered out: {file_path.name} (suffix={suffix})"
                            )

                        if result_file_type:
                            # Get relative path from plate_path (not results_dir) to include subdirectory
                            rel_path = file_path.relative_to(plate_path)

                            # Get file size
                            size_bytes = file_path.stat().st_size
                            if size_bytes < 1024:
                                size_str = f"{size_bytes} B"
                            elif size_bytes < 1024 * 1024:
                                size_str = f"{size_bytes / 1024:.1f} KB"
                            else:
                                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

                            # Parse ONLY the filename (not the full path) to extract metadata
                            parsed = handler.parser.parse_filename(file_path.name)

                            # Build file info with parsed metadata (no full_path in metadata dict)
                            file_info = {
                                "filename": str(rel_path),
                                "type": result_file_type.value,
                                "size": size_str,
                            }

                            # Add parsed metadata components if parsing succeeded
                            if parsed:
                                file_info.update(parsed)
                                logger.info(
                                    f"IMAGE BROWSER RESULTS: ✓ Parsed result: {file_path.name} -> {parsed}"
                                )
                                logger.info(
                                    f"IMAGE BROWSER RESULTS:   Full file_info: {file_info}"
                                )
                            else:
                                logger.warning(
                                    f"IMAGE BROWSER RESULTS: ✗ Could not parse filename: {file_path.name}"
                                )

                            # Store file info and full path separately
                            self.result_catalog.add(str(rel_path), file_info, file_path)

            logger.info(
                f"IMAGE BROWSER RESULTS: Scanned {file_count} total files, matched {len(self.result_catalog)} result files"
            )

        except Exception as e:
            logger.error(
                f"IMAGE BROWSER RESULTS: Failed to load results: {e}", exc_info=True
            )

    # Removed _populate_results_table - now using unified _populate_table
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

            from openhcs.pyqt_gui.utils.threading_utils import spawn_thread_with_context

            # For each enabled viewer, resolve config + viewer on UI thread, then spawn worker
            for viewer_field_name in enabled_viewers:
                # Get fully resolved streaming config from ObjectState (includes inheritance)
                config = self.state.get_resolved_value(viewer_field_name)
                backend_enum = config.backend

                # Create closure to capture viewer_config
                def _make_acquire_and_stream(cfg, field_name, backend):
                    def _acquire_and_stream():
                        try:
                            viewer = self.orchestrator.get_or_create_visualizer(cfg)
                            # _stream_single_roi_async itself starts a worker thread,
                            # so we call it here to kick off the streaming flow.
                            self._stream_single_roi_async(
                                viewer, roi_zip_path, cfg, backend, cfg.viewer_type
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to acquire {field_name} viewer or start streaming: {e}"
                            )
                            from PyQt6.QtCore import QTimer

                            QTimer.singleShot(
                                0,
                                lambda field_name=field_name, e=e: QMessageBox.warning(
                                    self,
                                    "Error",
                                    f"Failed to stream ROI to {field_name}: {e}",
                                ),
                            )

                    return _acquire_and_stream

                # Spawn the thread with captured config
                spawn_thread_with_context(
                    _make_acquire_and_stream(config, viewer_field_name, backend_enum),
                    name=f"acquire_{viewer_field_name}",
                )

            logger.info(f"Started async streaming of ROI file {roi_zip_path.name}")

        except Exception as e:
            logger.error(f"Failed to start ROI streaming: {e}")
            QMessageBox.warning(self, "Error", f"Failed to stream ROI file: {e}")

    def _stream_single_roi_async(
        self, viewer, roi_zip_path: Path, config, backend_enum, viewer_type: str
    ):
        """Worker: load a single ROI file and stream to a viewer in a background thread.

        Heavy operations only:
        - load_rois_from_zip
        - viewer.wait_for_ready (long timeout)
        - filemanager.save
        """
        from objectstate import spawn_thread_with_context

        def _worker():
            try:
                from pathlib import Path as _Path
                from PyQt6.QtCore import QTimer
                from polystore.roi import load_rois_from_zip

                # Load ROIs from disk
                self._status_update_signal.emit(
                    f"Loading ROIs from {roi_zip_path.name}..."
                )
                rois = load_rois_from_zip(roi_zip_path)

                if not rois:
                    msg = f"No ROIs found in {roi_zip_path.name}"
                    self._status_update_signal.emit(msg)

                    # Show info dialog on UI thread
                    QTimer.singleShot(
                        0, lambda: QMessageBox.information(self, "No ROIs", msg)
                    )
                    return

                # Wait for viewer to be ready (never on UI thread)
                is_already_running = viewer.wait_for_ready(timeout=0.1)
                if not is_already_running:
                    logger.info(
                        f"Waiting for {viewer_type.capitalize()} viewer on port {viewer.port} to be ready..."
                    )
                    if not viewer.wait_for_ready(timeout=15.0):
                        raise RuntimeError(
                            f"{viewer_type.capitalize()} viewer on port {viewer.port} failed to become ready"
                        )

                # Prepare metadata for streaming
                source = _Path(roi_zip_path).parent.name
                metadata = {
                    "port": viewer.port,
                    "display_config": config,
                    "microscope_handler": self.orchestrator.microscope_handler,
                    "plate_path": self.orchestrator.plate_path,
                    "source": source,
                }

                # Stream ROIs to viewer backend
                self._status_update_signal.emit(
                    f"Streaming ROIs from {roi_zip_path.name} to {viewer_type.capitalize()}..."
                )
                self.filemanager.save(
                    rois,
                    roi_zip_path,
                    backend_enum.value,
                    **metadata,
                )

                msg = (
                    f"Streamed {len(rois)} ROIs from {roi_zip_path.name} "
                    f"to {viewer_type.capitalize()} on port {viewer.port}"
                )
                logger.info(msg)
                self._status_update_signal.emit(msg)

            except Exception as e:  # noqa: BLE001
                logger.error(
                    f"Failed to load/stream ROI file {roi_zip_path.name} "
                    f"to {viewer_type.capitalize()}: {e}"
                )
                self._status_update_signal.emit(f"Error: {e}")

                from PyQt6.QtCore import QTimer
                from openhcs.ui.shared.streaming_service import StreamingService

                # Route to appropriate error dialog on UI thread
                if StreamingService.is_napari_viewer_type(viewer_type):
                    QTimer.singleShot(0, lambda: self._show_streaming_error(str(e)))
                else:
                    QTimer.singleShot(
                        0, lambda: self._show_fiji_streaming_error(str(e))
                    )

        spawn_thread_with_context(_worker, name="stream_roi")

    def _populate_table(self, files_dict: Dict[str, Dict]):
        """Populate table with files (images + results) using ImageTableBrowser."""
        self.image_table_browser.set_items(files_dict)

        # Update status label with file counts
        total = len(self.all_files)
        filtered = len(files_dict)
        self.image_table_browser.status_label.setText(f"Files: {filtered}/{total}")

    def _update_table_with_filtered_items(self, files_dict: Dict[str, Dict]):
        """Update table with filtered items without rebuilding SearchService.

        Use this when all_files has not changed, only filter criteria changed.
        Much faster than _populate_table() for checkbox filter updates.
        """
        self.image_table_browser.set_filtered_items(files_dict)

        # Update status label with file counts
        total = len(self.all_files)
        filtered = len(files_dict)
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
        for filename in self.all_files.keys():
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

    # Backward compatibility alias
    def on_selection_changed(self):
        """Handle selection change in the table (legacy)."""
        selected_keys = self.image_table_browser.get_selected_keys()
        self._on_files_selected(selected_keys)

    def _on_file_double_clicked(self, key: str, item: dict):
        """Handle double-click from ImageTableBrowser."""
        file_info = self.all_files[key]

        # Check if this is a result file (ROI, CSV, JSON) or an image
        # Result files are stored in result_catalog, images are not
        filename = file_info["filename"]
        if self.result_catalog.is_result(filename):
            # This is a result file (ROI, CSV, JSON)
            self._handle_result_double_click(file_info)
        else:
            # This is an image file
            self._handle_image_double_click()

    # Backward compatibility alias
    def on_file_double_clicked(self, row: int, column: int):
        """Handle double-click on a file row (legacy)."""
        filename_item = self.file_table.item(row, 0)
        filename = filename_item.data(Qt.ItemDataRole.UserRole)
        file_info = self.all_files[filename]
        self._on_file_double_clicked(filename, file_info)

    def _handle_image_double_click(self):
        """Handle double-click on an image - stream to enabled viewer(s)."""
        # Find all enabled viewers by querying ObjectState
        enabled_viewers = self.viewer_controls.enabled_viewers()

        # Stream to whichever viewer(s) are enabled
        if enabled_viewers:
            for viewer_type in enabled_viewers:
                self._view_selected_in_viewer(viewer_type)
        else:
            # No viewers enabled - show message
            QMessageBox.information(
                self,
                "No Viewer Enabled",
                "Please enable at least one viewer streaming to view images.",
            )

    def _handle_result_double_click(self, file_info: dict):
        """Handle double-click on a result file - stream ROIs or display CSV."""
        filename = file_info["filename"]
        file_path = self.result_catalog.path_for(filename)
        result_file_type = ResultFileType(file_info["type"])
        RESULT_FILE_ACTIONS[result_file_type].run(self, file_path)

    def _view_selected_in_viewer(self, viewer_type: str):
        """View all selected images in the specified viewer as a batch (builds hyperstack)."""
        selected_keys = self.image_table_browser.get_selected_keys()
        if not selected_keys:
            return

        # Separate ROI files from images
        image_filenames = [k for k in selected_keys if not k.endswith(".roi.zip")]
        roi_filenames = [k for k in selected_keys if k.endswith(".roi.zip")]

        logger.info(
            f"🎯 IMAGE BROWSER: User selected {len(image_filenames)} images and {len(roi_filenames)} ROI files to view in {viewer_type}"
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
                plate_path = Path(self.orchestrator.plate_path)
                self._stream_rois_to_viewer(roi_filenames, plate_path, viewer_type)

            # Stream image files as a batch
            if image_filenames:
                self._stream_images_to_viewer(image_filenames, viewer_type)

        spawn_thread_with_context(_view_async, name=f"view_{viewer_type}")

    def _prepare_streaming(self, viewer_type: str) -> tuple:
        """Prepare for streaming: resolve config, get viewer, get read backend.

        Returns: (viewer, plate_path, read_backend, config)
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
        config = self.state.get_resolved_value(viewer_type)

        viewer = self.orchestrator.get_or_create_visualizer(config)
        return viewer, plate_path, read_backend, config

    def _stream_images_to_viewer(self, filenames: list, viewer_type: str):
        """Load and stream images to specified viewer type."""
        viewer, plate_path, read_backend, config = self._prepare_streaming(viewer_type)
        from openhcs.ui.shared.streaming_service import (
            ImageStreamingRequest,
            StreamingService,
            ViewerStreamingContext,
        )

        streaming_service = self.streaming_service
        if streaming_service is None:
            raise RuntimeError("No orchestrator set")

        streaming_service.stream_images_async(
            ImageStreamingRequest(
                context=ViewerStreamingContext(
                    viewer=viewer,
                    plate_path=plate_path,
                    config=config,
                    viewer_type=config.viewer_type,
                    status_callback=self._status_update_signal.emit,
                    error_callback=lambda e: self._show_streaming_error(e)
                    if StreamingService.is_napari_viewer_type(config.viewer_type)
                    else self._show_fiji_streaming_error(e),
                ),
                filenames=tuple(filenames),
                read_backend=read_backend,
            )
        )
        logger.info(f"Streaming {len(filenames)} images to {viewer_type}...")

    @pyqtSlot(str)
    def _show_streaming_error(self, error_msg: str):
        """Show streaming error in UI thread (called via QMetaObject.invokeMethod)."""
        QMessageBox.warning(
            self, "Streaming Error", f"Failed to stream images to Napari: {error_msg}"
        )

    @pyqtSlot(str)
    def _show_fiji_streaming_error(self, error_msg: str):
        """Show Fiji streaming error in UI thread."""
        QMessageBox.warning(
            self, "Streaming Error", f"Failed to stream images to Fiji: {error_msg}"
        )

    def _stream_rois_to_viewer(
        self, roi_filenames: list, plate_path: Path, viewer_type: str
    ):
        """Stream ROI files to specified viewer type."""
        viewer, _, _, config = self._prepare_streaming(viewer_type)
        from openhcs.ui.shared.streaming_service import (
            RoiStreamingRequest,
            StreamingService,
            ViewerStreamingContext,
        )

        streaming_service = self.streaming_service
        if streaming_service is None:
            raise RuntimeError("No orchestrator set")

        streaming_service.stream_rois_async(
            RoiStreamingRequest(
                context=ViewerStreamingContext(
                    viewer=viewer,
                    plate_path=plate_path,
                    config=config,
                    viewer_type=config.viewer_type,
                    status_callback=self._status_update_signal.emit,
                    error_callback=lambda e: self._show_streaming_error(e)
                    if StreamingService.is_napari_viewer_type(config.viewer_type)
                    else self._show_fiji_streaming_error(e),
                ),
                roi_filenames=tuple(roi_filenames),
            )
        )
        logger.info(f"Streaming {len(roi_filenames)} ROI files to {viewer_type}...")

    def cleanup(self):
        """Clean up resources before widget destruction."""
        # Cleanup ZMQ server manager widget (always initialized to None in __init__)
        if self.zmq_manager is not None:
            self.zmq_manager.cleanup()

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
        self.selected_wells = well_ids
        self.filter_controller.apply_combined_filters()

    def _update_plate_view(self):
        """Update plate view with current file data (images + results)."""
        # Extract all well IDs from current files (filter out failures)
        well_ids = set()
        for filename, metadata in self.all_files.items():
            try:
                well_id = self._extract_well_id(metadata)
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

        for filename in self.all_files.keys():
            file_path = Path(filename)

            # Check if file is in a subdirectory of base_path
            try:
                relative = file_path.relative_to(base_path)
                parts = relative.parts

                # If file is in a subdirectory (not directly in base_path)
                if len(parts) > 1:
                    subdir_name = parts[0]

                    # Check if this file has well metadata
                    metadata = self.all_files[filename]
                    try:
                        self._extract_well_id(metadata)
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
