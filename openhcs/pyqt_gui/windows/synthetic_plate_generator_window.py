"""
Synthetic Plate Generator Window for PyQt6

Provides a user-friendly interface for generating synthetic microscopy plates
for testing OpenHCS without requiring real microscopy data.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QWidget,
    QMessageBox,
    QFileDialog,
)
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont

from pyqt_reactive.forms.parameter_form_manager import ParameterFormManager
from pyqt_reactive.forms.layout_constants import CURRENT_LAYOUT
from pyqt_reactive.theming import StyleSheetGenerator
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.shared.reflowing_vertical_scroll_area import (
    ReflowingVerticalScrollArea,
)
from openhcs.tests.generators.generate_synthetic_data import (
    SyntheticMicroscopyGenerator,
)

logger = logging.getLogger(__name__)


class SyntheticPlateGeneratorWindow(QDialog):
    """
    Dialog window for generating synthetic microscopy plates.

    Provides a parameter form for SyntheticMicroscopyGenerator with sensible
    defaults from the test suite, allowing users to easily generate test data.
    """

    # Signals
    plate_generated = pyqtSignal(str, str)  # output_dir path, pipeline_path

    def __init__(self, color_scheme: Optional[ColorScheme] = None, parent=None):
        """
        Initialize the synthetic plate generator window.

        Args:
            color_scheme: Color scheme for styling (optional)
            parent: Parent widget
        """
        super().__init__(parent)

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)

        # Output directory (will be set by user or use temp)
        self.output_dir: Optional[str] = None
        self.form_manager: Optional[ParameterFormManager] = None

        # Setup UI
        self.setup_ui()

        logger.debug("Synthetic plate generator window initialized")

    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle("Generate Synthetic Plate")
        self.setModal(True)
        self.setMinimumSize(700, 600)
        self.resize(800, 700)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header with title and description
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 10, 10, 10)

        header_label = QLabel("Generate Synthetic Microscopy Plate")
        header_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};"
        )
        header_layout.addWidget(header_label)

        description_label = QLabel(
            "Generate synthetic microscopy data for testing OpenHCS without real data. "
            "The defaults match the test suite configuration and will create a small 4-well plate "
            "with 3x3 grid, 2 channels, and 3 z-levels."
        )
        description_label.setWordWrap(True)
        description_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; padding: 5px;"
        )
        header_layout.addWidget(description_label)

        layout.addWidget(header_widget)

        # Output directory selection
        output_dir_widget = self._create_output_dir_selector()
        layout.addWidget(output_dir_widget)

        # Parameter form with scroll area
        scroll_area = ReflowingVerticalScrollArea()

        # Create form manager from SyntheticMicroscopyGenerator class
        # This automatically builds the UI from the __init__ signature (same pattern as function_pane.py)
        # CRITICAL: Pass color_scheme as parameter to ensure consistent theming with other parameter forms
        from pyqt_reactive.forms.parameter_form_manager import FormManagerConfig
        from objectstate.object_state import ObjectState

        # Standalone tool - create local ObjectState (not registered in registry)
        self.state = ObjectState(
            object_instance=SyntheticMicroscopyGenerator,  # Pass the class itself, not __init__
            scope_id=None,
            exclude_params=[
                "output_dir",
                "skip_files",
                "include_all_components",
                "random_seed",
            ],
        )

        self.form_manager = ParameterFormManager(
            state=self.state,
            config=FormManagerConfig(
                parent=self,
                color_scheme=self.color_scheme,  # Pass color_scheme as instance parameter
            ),
        )

        scroll_area.setWidget(self.form_manager)
        layout.addWidget(scroll_area)

        # Button row
        button_row = QHBoxLayout()
        button_row.setContentsMargins(10, 5, 10, 10)
        button_row.setSpacing(10)

        button_row.addStretch()

        # Get centralized button styles
        button_styles = self.style_generator.generate_config_button_styles()

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        cancel_button.setMinimumWidth(100)
        cancel_button.clicked.connect(self.reject)
        cancel_button.setStyleSheet(button_styles["cancel"])
        button_row.addWidget(cancel_button)

        # Generate button
        generate_button = QPushButton("Generate Plate")
        generate_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        generate_button.setMinimumWidth(100)
        generate_button.clicked.connect(self.generate_plate)
        generate_button.setStyleSheet(button_styles["save"])
        button_row.addWidget(generate_button)

        layout.addLayout(button_row)

        # Apply window styling
        self.setStyleSheet(self.style_generator.generate_dialog_style())

    def _create_output_dir_selector(self) -> QWidget:
        """Create the output directory selector widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)

        label = QLabel("Output Directory:")
        label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};"
        )
        layout.addWidget(label)

        self.output_dir_label = QLabel("<temp directory>")
        self.output_dir_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)}; "
            f"font-style: italic; padding: 5px;"
        )
        layout.addWidget(self.output_dir_label, 1)

        browse_button = QPushButton("Browse...")
        browse_button.setFixedHeight(CURRENT_LAYOUT.button_height)
        browse_button.setMinimumWidth(80)
        browse_button.clicked.connect(self.browse_output_dir)
        browse_button.setStyleSheet(self.style_generator.generate_button_style())
        layout.addWidget(browse_button)

        return widget

    def browse_output_dir(self):
        """Open file dialog to select output directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory for Synthetic Plate",
            str(Path.home()),
            QFileDialog.Option.ShowDirsOnly,
        )

        if dir_path:
            self.output_dir = dir_path
            self.output_dir_label.setText(dir_path)
            self.output_dir_label.setStyleSheet(
                f"color: {self.color_scheme.to_hex(self.color_scheme.text_primary)}; padding: 5px;"
            )

    def generate_plate(self):
        """Generate the synthetic plate with current parameters."""
        try:
            # Get parameters from state
            params = self.state.get_current_values()

            # Determine output directory
            if self.output_dir is None:
                # Use temp directory
                temp_dir = tempfile.mkdtemp(prefix="openhcs_synthetic_plate_")
                output_dir = temp_dir
                logger.info(f"Using temporary directory: {output_dir}")
            else:
                output_dir = self.output_dir

            # Add output_dir to params
            params["output_dir"] = output_dir

            # Create generator and generate dataset
            logger.info(f"Generating synthetic plate with params: {params}")
            generator = SyntheticMicroscopyGenerator(**params)
            generator.generate_dataset()

            logger.info(f"Successfully generated synthetic plate at: {output_dir}")

            # Get path to test_pipeline.py
            from openhcs.tests import test_pipeline

            pipeline_path = Path(test_pipeline.__file__)

            # Emit signal with output directory and pipeline path
            self.plate_generated.emit(output_dir, str(pipeline_path))

            # Close dialog
            self.accept()

        except Exception as e:
            logger.error(f"Failed to generate synthetic plate: {e}", exc_info=True)

    def reject(self):
        """Handle dialog rejection (Cancel button)."""
        # Cleanup before closing
        self._cleanup()
        super().reject()

    def accept(self):
        """Handle dialog acceptance (Generate button)."""
        # Cleanup before closing
        self._cleanup()
        super().accept()

    def _cleanup(self):
        """Cleanup resources before window closes."""
        # Unregister from cross-window updates
        if self.form_manager is not None:
            try:
                self.form_manager.unregister_from_cross_window_updates()
            except RuntimeError:
                # Widget already deleted, ignore
                pass

        # Disconnect all signals to prevent accessing deleted Qt objects
        try:
            self.plate_generated.disconnect()
        except (RuntimeError, TypeError):
            # Already disconnected or no connections, ignore
            pass
