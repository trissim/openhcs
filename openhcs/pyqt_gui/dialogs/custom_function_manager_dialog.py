"""
Custom Function Manager Dialog.

Provides GUI interface for managing custom functions (list, create, edit, delete).
Automatically refreshes when custom functions change via signals.
"""

from types import FunctionType

from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QGroupBox,
    QLabel,
    QMessageBox,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets.editors.simple_code_editor import QScintillaCodeEditorDialog
from pyqt_reactive.widgets.shared import BaseFormDialog
from openhcs.processing.custom_functions import CustomFunctionManager
from openhcs.processing.custom_functions.signals import custom_function_signals
from openhcs.processing.custom_functions.templates import get_default_template
from openhcs.processing.custom_functions.validation import ValidationError


class CustomFunctionManagerDialog(BaseFormDialog):
    """
    Dialog for managing custom functions.

    Features:
        - List all custom functions
        - Create new functions
        - Edit existing functions
        - Delete functions
        - Automatic refresh via signals
    """

    def __init__(self, parent=None):
        """
        Initialize custom function manager dialog.

        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Custom Function Manager")
        self.resize(700, 500)

        # Initialize manager and color scheme
        self.manager = CustomFunctionManager()
        self.color_scheme = ColorScheme()

        # Setup UI first (widgets must exist before signal connection)
        self._setup_ui()

        # Connect to global signals for automatic refresh AFTER UI setup
        # Use singleton signal instance, not manager instance signals
        custom_function_signals.functions_changed.connect(self._refresh_function_list)

        # Initial population
        self._populate_function_list()

    def _setup_ui(self):
        """Setup UI components."""
        layout = QVBoxLayout(self)

        # Function list
        list_group = QGroupBox("Custom Functions")
        list_layout = QVBoxLayout(list_group)

        self.function_list = QListWidget()
        self.function_list.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.function_list)

        layout.addWidget(list_group)

        # Function details
        details_group = QGroupBox("Function Details")
        details_layout = QVBoxLayout(details_group)

        self.name_label = QLabel("Name: ")
        self.backend_label = QLabel("Backend: ")
        self.file_label = QLabel("File: ")

        details_layout.addWidget(self.name_label)
        details_layout.addWidget(self.backend_label)
        details_layout.addWidget(self.file_label)

        layout.addWidget(details_group)

        # Action buttons
        button_layout = QHBoxLayout()

        self.create_btn = QPushButton("Create New")
        self.create_btn.clicked.connect(self._on_create_clicked)
        button_layout.addWidget(self.create_btn)

        self.edit_btn = QPushButton("Edit")
        self.edit_btn.clicked.connect(self._on_edit_clicked)
        self.edit_btn.setEnabled(False)
        button_layout.addWidget(self.edit_btn)

        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        self.delete_btn.setEnabled(False)
        button_layout.addWidget(self.delete_btn)

        button_layout.addStretch()

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.close_btn)

        layout.addLayout(button_layout)

    def _populate_function_list(self):
        """Populate function list from manager."""
        self.function_list.clear()

        functions = self.manager.list_custom_functions()
        for func_info in functions:
            self.function_list.addItem(func_info.name)

        if not functions:
            self.function_list.addItem("(No custom functions)")

    def _refresh_function_list(self):
        """Refresh function list when functions change."""
        self._populate_function_list()
        self._clear_details()

    def _on_selection_changed(self):
        """Handle function selection change."""
        selected_items = self.function_list.selectedItems()

        if not selected_items or selected_items[0].text() == "(No custom functions)":
            self.edit_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            self._clear_details()
            return

        self.edit_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)

        # Show function details
        func_name = selected_items[0].text()
        functions = self.manager.list_custom_functions()

        for func_info in functions:
            if func_info.name == func_name:
                self.name_label.setText(f"Name: {func_info.name}")
                self.backend_label.setText(f"Backend: {func_info.memory_type}")
                self.file_label.setText(f"File: {func_info.file_path.name}")
                break

    def _clear_details(self):
        """Clear function details display."""
        self.name_label.setText("Name: ")
        self.backend_label.setText("Backend: ")
        self.file_label.setText("File: ")

    def _on_create_clicked(self):
        """Handle create new function button click."""
        # Get default template (numpy backend)
        # NOTE: Default template works for all backends - users can change
        # the decorator (@numpy, @cupy, etc.) in the editor
        # Future enhancement: Add template selection dropdown
        template = get_default_template()

        # Open code editor (LLM assist always available via button)
        editor = QScintillaCodeEditorDialog(
            parent=self,
            initial_content=template,
            title="Create Custom Function",
            declaration_type=FunctionType,
        )

        if editor.exec():
            # User clicked Save
            code = editor.get_content()

            try:
                functions = self.manager.register_from_code(code)
                func_names = ", ".join(f.__name__ for f in functions)
                QMessageBox.information(
                    self,
                    "Success",
                    f"Function(s) '{func_names}' registered successfully!"
                )
                # List will auto-refresh via signal
            except ValidationError as e:
                # Validation failed - show specific error
                QMessageBox.critical(
                    self,
                    "Validation Failed",
                    f"Function code validation failed:\n\n{str(e)}"
                )
            # Let other exceptions propagate (fail-loud)

    def _on_edit_clicked(self):
        """Handle edit function button click."""
        selected_items = self.function_list.selectedItems()
        if not selected_items:
            return

        func_name = selected_items[0].text()

        try:
            # Get current code
            code = self.manager.get_function_code(func_name)

            # Open code editor with existing code (LLM assist always available via button)
            editor = QScintillaCodeEditorDialog(
                parent=self,
                initial_content=code,
                title=f"Edit Custom Function: {func_name}",
                declaration_type=FunctionType,
            )

            if editor.exec():
                # User clicked Save
                new_code = editor.get_content()

                try:
                    new_name = self.manager.update_custom_function(func_name, new_code)
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Function updated successfully!\n"
                        f"{'Function renamed to: ' + new_name if new_name != func_name else ''}"
                    )
                    # List will auto-refresh via signal
                except ValidationError as e:
                    # Validation failed - show specific error with preservation message
                    QMessageBox.critical(
                        self,
                        "Validation Failed",
                        f"Function code validation failed:\n\n{str(e)}\n\n"
                        f"Original function '{func_name}' was not modified."
                    )
                except ValueError as e:
                    # Function not found
                    QMessageBox.warning(self, "Function Not Found", str(e))
                # Let other exceptions propagate (fail-loud)

        except ValueError as e:
            # Function not found
            QMessageBox.warning(self, "Function Not Found", str(e))

    def _on_delete_clicked(self):
        """Handle delete function button click."""
        selected_items = self.function_list.selectedItems()
        if not selected_items:
            return

        func_name = selected_items[0].text()

        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete function '{func_name}'?\n\n"
            "This will remove the source file from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            success = self.manager.delete_custom_function(func_name)
            if success:
                QMessageBox.information(
                    self,
                    "Deleted",
                    f"Function '{func_name}' deleted successfully."
                )
                # List will auto-refresh via signal
            else:
                QMessageBox.warning(
                    self,
                    "Not Found",
                    f"Function '{func_name}' file not found."
                )
