"""
Plate Viewer Window - Tabbed interface for Image Browser and Metadata Viewer.

Combines image browsing and metadata viewing in a single window with tabs.
"""

import logging

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QWidget,
    QLabel,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from pyqt_reactive.forms.object_form_document_renderer import (
    ObjectFormDocumentRenderer,
    ObjectFormRenderContext,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared import BaseFormDialog
from openhcs.pyqt_gui.config import ProgressUIConfig
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG, OpenHCSZMQConfig

logger = logging.getLogger(__name__)


class PlateViewerWindow(BaseFormDialog):
    """
    Tabbed window for viewing plate images and metadata.

    Combines:
    - Image Browser (tab 1): Browse and view images in Napari
    - Metadata Viewer (tab 2): View plate metadata

    Inherits singleton-per-scope behavior from BaseFormDialog.
    Only ONE PlateViewerWindow per plate can be open at a time.
    """

    def __init__(
        self,
        orchestrator,
        zmq_config: OpenHCSZMQConfig = OPENHCS_ZMQ_CONFIG,
        progress_config: ProgressUIConfig | None = None,
        parent=None,
    ):
        """
        Initialize plate viewer window.

        Args:
            orchestrator: PipelineOrchestrator instance
            parent: Parent widget
        """
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.zmq_config = zmq_config
        self.progress_config = (
            ProgressUIConfig() if progress_config is None else progress_config
        )

        # scope_id for singleton behavior - one viewer per plate
        # Use ::plate_viewer suffix to avoid conflicts with ConfigWindow (which uses just plate_path)
        self.scope_id = (
            f"{orchestrator.plate_path}::plate_viewer" if orchestrator else None
        )
        # Store plate path for styling (without suffix) so border matches plate's ConfigWindow
        self._style_scope_id = str(orchestrator.plate_path) if orchestrator else None

        # CRITICAL: Initialize scope-based styling BEFORE creating child widgets
        # This sets self._scope_accent_color for use in this class
        if self._style_scope_id:
            self.init_scope_border()

        # Get scope accent color and create color scheme from it
        from pyqt_reactive.services.scope_color_service import ScopeColorService

        accent_color = ScopeColorService.instance().get_accent_color(
            self._style_scope_id
        )

        # Create color scheme with accent color as text_accent (convert QColor to RGB tuple)
        if accent_color:
            self.color_scheme = ColorScheme()
            self.color_scheme.text_accent = (
                accent_color.red(),
                accent_color.green(),
                accent_color.blue(),
            )
        else:
            self.color_scheme = ColorScheme()

        self.image_browser: QWidget | None = None
        self.image_browser_tab: QWidget | None = None
        self.metadata_viewer_tab: QWidget | None = None
        self._metadata_viewer_loaded = False
        self._metadata_tab_index = -1

        plate_name = orchestrator.plate_path.name if orchestrator else "Unknown"
        self.setWindowTitle(f"Plate Viewer - {plate_name}")
        self.setMinimumSize(800, 600)
        self.resize(1200, 800)

        # Make floating window with Dialog hint so tiling WMs don't fullscreen it
        # Qt.WindowType.Window alone strips the Dialog flag, causing tiling WMs to tile/fullscreen
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.Dialog)

        self._setup_ui()

    def init_scope_border(self) -> None:
        """Override to use plate-level styling (not step-level).

        PlateViewerWindow uses scope_id with ::plate_viewer suffix for WindowManager,
        but should use the plate path (without suffix) for border styling to match
        the plate's ConfigWindow.
        """
        # Temporarily swap scope_id to use plate-level styling
        original_scope_id = self.scope_id
        self.scope_id = self._style_scope_id
        try:
            super().init_scope_border()
        finally:
            self.scope_id = original_scope_id

    def _setup_ui(self):
        """Setup the window UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        layout.setSpacing(5)  # Reduced spacing

        # Single row: tabs + title + button
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(0, 0, 0, 0)  # No margins - let tabs breathe
        tab_row.setSpacing(10)

        # Tab widget (tabs on the left)
        self.tab_widget = QTabWidget()
        # Get the tab bar and add it to our horizontal layout
        self.tab_bar = self.tab_widget.tabBar()
        # Prevent tab scrolling by setting expanding to false and using minimum size hint
        self.tab_bar.setExpanding(False)
        self.tab_bar.setUsesScrollButtons(False)
        tab_row.addWidget(self.tab_bar, 0)  # 0 stretch - don't expand

        # Show plate name with full path in parentheses, with elision (title on right of tabs)
        if self.orchestrator:
            plate_name = self.orchestrator.plate_path.name
            full_path = str(self.orchestrator.plate_path)
            title_text = f"Plate: {plate_name} ({full_path})"
        else:
            title_text = "Plate: Unknown"

        title_label = QLabel(title_text)
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        title_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};"
        )
        title_label.setWordWrap(False)  # Single line
        title_label.setTextFormat(Qt.TextFormat.PlainText)
        title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )  # Allow copying
        # Enable elision (text will be cut with ... when too long)
        from PyQt6.QtWidgets import QSizePolicy

        title_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        tab_row.addWidget(title_label, 1)  # Stretch to fill available space

        tab_row.addStretch()

        # Consolidate Results button
        consolidate_btn = QPushButton("Consolidate Results")
        consolidate_btn.clicked.connect(self._consolidate_results)
        consolidate_btn.setToolTip(
            "Generate MetaXpress-style summary CSV from analysis results"
        )
        consolidate_btn.setStyleSheet(self.color_scheme.styles.generate_button_style())
        tab_row.addWidget(consolidate_btn)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_btn.setStyleSheet(self.color_scheme.styles.generate_button_style())
        tab_row.addWidget(close_btn)

        layout.addLayout(tab_row)

        # Style tab bar
        self.tab_bar.setStyleSheet(f"""
            QTabBar::tab {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.input_bg)};
                color: white;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                border: none;
            }}
            QTabBar::tab:selected {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};
            }}
            QTabBar::tab:hover {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.button_hover_bg)};
            }}
        """)

        # Tab 1: Image Browser
        self.image_browser_tab = self._create_image_browser_tab()
        self.tab_widget.addTab(self.image_browser_tab, "Image Browser")

        # Tab 2: Metadata Viewer (lazy-loaded to avoid slow startup)
        self.metadata_viewer_tab = self._create_metadata_placeholder_tab()
        self._metadata_viewer_loaded = False
        self._metadata_tab_index = self.tab_widget.addTab(
            self.metadata_viewer_tab, "Metadata"
        )
        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        # Add the tab widget's content area (stacked widget) below the tab row
        # The tab bar is already in tab_row, so we only add the content pane here
        from PyQt6.QtWidgets import QStackedWidget

        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Get the stacked widget from the tab widget and add it
        stacked_widget = self.tab_widget.findChild(QStackedWidget)
        if stacked_widget:
            content_layout.addWidget(stacked_widget)

        layout.addWidget(content_container)

    def _create_image_browser_tab(self) -> QWidget:
        """Create the image browser tab."""
        from openhcs.pyqt_gui.widgets.image_browser import ImageBrowserWidget

        # Create image browser widget
        browser = ImageBrowserWidget(
            orchestrator=self.orchestrator,
            color_scheme=self.color_scheme,
            zmq_config=self.zmq_config,
            progress_config=self.progress_config,
            parent=self,
        )

        # Store reference
        self.image_browser = browser

        return browser

    def _create_metadata_placeholder_tab(self) -> QWidget:
        """Create a lightweight placeholder for lazy metadata loading."""
        placeholder = QWidget()
        layout = QVBoxLayout(placeholder)
        layout.setContentsMargins(10, 10, 10, 10)
        label = QLabel("Open this tab to load metadata...")
        layout.addWidget(label)
        layout.addStretch()
        return placeholder

    def _on_tab_changed(self, index: int) -> None:
        """Lazy-load metadata viewer when the Metadata tab is opened."""
        if self._metadata_viewer_loaded:
            return
        if index != self._metadata_tab_index:
            return

        from PyQt6.QtCore import QSignalBlocker

        self._metadata_viewer_loaded = True
        metadata_viewer = self._create_metadata_viewer_tab()
        with QSignalBlocker(self.tab_widget):
            self.tab_widget.removeTab(index)
            self.tab_widget.insertTab(index, metadata_viewer, "Metadata")
            self.tab_widget.setCurrentIndex(index)
        self.metadata_viewer_tab = metadata_viewer

    def _create_metadata_viewer_tab(self) -> QWidget:
        """Create the metadata viewer tab."""
        # Create scroll area for metadata content
        from PyQt6.QtWidgets import QScrollArea

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)

        # Container for metadata forms
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(5, 5, 5, 5)

        try:
            metadata_handler = self.orchestrator.microscope_handler.metadata_handler
            plate_path = self.orchestrator.plate_path
            document = metadata_handler.build_metadata_view_document(
                plate_path,
                self.orchestrator.microscope_handler,
            )
            ObjectFormDocumentRenderer(
                ObjectFormRenderContext(
                    parent=container,
                    scope_id=self._style_scope_id,
                    color_scheme=self.color_scheme,
                    exclude_params=("image_files", "workspace_mapping"),
                )
            ).render(layout, document)

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}", exc_info=True)
            error_label = QLabel(f"<b>Error loading metadata:</b><br>{str(e)}")
            error_label.setWordWrap(True)
            error_label.setStyleSheet("color: red; padding: 10px;")
            layout.addWidget(error_label)

        layout.addStretch()

        # Set container as scroll area widget
        scroll_area.setWidget(container)
        return scroll_area

    def _consolidate_results(self):
        """Manually trigger analysis results consolidation."""
        from PyQt6.QtWidgets import QMessageBox

        try:
            # Find results directories from the metadata handler's format contract.
            plate_path = self.orchestrator.plate_path
            metadata_handler = self.orchestrator.microscope_handler.metadata_handler
            results_dirs = [
                result_directory.path
                for result_directory in metadata_handler.analysis_result_directories(
                    plate_path
                )
            ]

            if not results_dirs:
                QMessageBox.warning(
                    self,
                    "No Results Found",
                    f"No analysis results directories are declared for {plate_path}.",
                )
                return

            if not self.orchestrator.pipeline_config:
                QMessageBox.warning(
                    self,
                    "No Pipeline Config",
                    "No pipeline configuration found. Please ensure the orchestrator is properly initialized.",
                )
                return

            effective_config = self.orchestrator.get_effective_config()
            analysis_consolidation_config = (
                effective_config.analysis_consolidation_config
            )
            plate_metadata_config = effective_config.plate_metadata_config

            # Use consolidated function that handles both per-directory and global consolidation
            from openhcs.processing.backends.analysis.consolidate_analysis_results import (
                consolidate_results_directories,
            )

            successful_dirs, failed_dirs = consolidate_results_directories(
                results_dirs=results_dirs,
                plate_path=plate_path,
                analysis_consolidation_config=analysis_consolidation_config,
                plate_metadata_config=plate_metadata_config,
                filename_parser=self.orchestrator.microscope_handler.parser,
            )

            # Show results
            if not successful_dirs and not failed_dirs:
                QMessageBox.warning(
                    self,
                    "No CSV Files",
                    "No CSV files found in any results directories. Nothing to consolidate.",
                )
            elif successful_dirs and not failed_dirs:
                msg = (
                    f"Successfully consolidated {len(successful_dirs)} results directories:\n"
                    + "\n".join(f"  ✓ {d}" for d in successful_dirs)
                )
                if len(successful_dirs) > 1:
                    msg += f"\n\n✅ Global summary created: {analysis_consolidation_config.global_summary_filename}"
                QMessageBox.information(self, "Consolidation Complete", msg)
            elif successful_dirs and failed_dirs:
                QMessageBox.warning(
                    self,
                    "Partial Success",
                    f"Consolidated {len(successful_dirs)} of {len(results_dirs)} directories.\n\n"
                    f"Successful:\n"
                    + "\n".join(f"  ✓ {d}" for d in successful_dirs)
                    + "\n\n"
                    "Failed:\n" + "\n".join(f"  ✗ {d}: {e}" for d, e in failed_dirs),
                )
            else:
                QMessageBox.critical(
                    self,
                    "Consolidation Failed",
                    f"All {len(failed_dirs)} directories failed to consolidate:\n\n"
                    + "\n".join(f"  ✗ {d}: {e}" for d, e in failed_dirs),
                )

        except Exception as e:
            logger.error(f"Failed to consolidate results: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Consolidation Failed",
                f"Failed to consolidate results:\n\n{str(e)}",
            )

    def cleanup(self):
        """Clean up resources."""
        if self.image_browser is not None:
            self.image_browser.cleanup()
