"""
Plate Metadata Viewer Dialog

Read-only viewer for plate metadata using generic reflection.
Displays SubdirectoryKeyedMetadata or OpenHCSMetadata directly.
"""

import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QWidget,
    QLabel,
)
from PyQt6.QtCore import Qt

from pyqt_reactive.forms.object_form_document_renderer import (
    ObjectFormDocumentRenderer,
    ObjectFormRenderContext,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared import BaseFormDialog

logger = logging.getLogger(__name__)


class MetadataViewerDialog(BaseFormDialog):
    """
    Read-only metadata viewer dialog.

    Uses ParameterFormManager with generic reflection to display
    SubdirectoryKeyedMetadata or OpenHCSMetadata instances.

    Inherits singleton-per-scope behavior from BaseFormDialog.
    Only ONE MetadataViewerDialog per plate can be open at a time.
    """

    def __init__(
        self, orchestrator, color_scheme: Optional[ColorScheme] = None, parent=None
    ):
        """
        Initialize metadata viewer dialog.

        Args:
            orchestrator: PipelineOrchestrator instance
            color_scheme: Color scheme for styling
            parent: Parent widget
        """
        super().__init__(parent)
        self.orchestrator = orchestrator
        self.color_scheme = color_scheme or ColorScheme()

        # scope_id for singleton behavior - one viewer per plate
        self.scope_id = str(orchestrator.plate_path) if orchestrator else None

        self.setWindowTitle(f"Plate Metadata - {orchestrator.plate_path.name}")
        self.setMinimumSize(800, 600)
        self.resize(1000, 700)

        # Make floating like other OpenHCS windows
        self.setWindowFlags(Qt.WindowType.Dialog)

        self._setup_ui()
        self._load_metadata()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)  # Reduced margins
        layout.setSpacing(5)  # Reduced spacing

        # Title label
        title_label = QLabel(f"<b>Plate:</b> {self.orchestrator.plate_path}")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        # Scroll area for metadata form
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container for the form
        self.form_container = QWidget()
        scroll_area.setWidget(self.form_container)
        layout.addWidget(scroll_area)

        # Button row
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setMinimumWidth(100)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def _load_metadata(self):
        """Load and display metadata through the handler projection contract."""
        try:
            metadata_handler = self.orchestrator.microscope_handler.metadata_handler
            plate_path = self.orchestrator.plate_path

            document = metadata_handler.build_metadata_view_document(
                plate_path,
                self.orchestrator.microscope_handler,
            )
            form_layout = QVBoxLayout(self.form_container)
            form_layout.setContentsMargins(5, 5, 5, 5)
            ObjectFormDocumentRenderer(
                ObjectFormRenderContext(
                    parent=self.form_container,
                    color_scheme=self.color_scheme,
                    exclude_params=("image_files", "workspace_mapping"),
                )
            ).render(form_layout, document)

            # Update window title
            self.setWindowTitle(f"{document.title} - {self.orchestrator.plate_path.name}")

            logger.info(f"Loaded metadata for {self.orchestrator.plate_path}")

        except Exception as e:
            logger.error(f"Failed to load metadata: {e}", exc_info=True)

            # Show error in form container
            error_layout = QVBoxLayout(self.form_container)
            error_label = QLabel(f"<b>Error loading metadata:</b><br>{str(e)}")
            error_label.setWordWrap(True)
            error_label.setStyleSheet("color: red; padding: 10px;")
            error_layout.addWidget(error_label)
            error_layout.addStretch()
