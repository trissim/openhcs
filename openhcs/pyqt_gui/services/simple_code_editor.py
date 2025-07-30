"""
Simple Code Editor Service for PyQt GUI.

Provides modular text editing with QScintilla (default) or external program launch.
No threading complications - keeps it simple and direct.
"""

import logging
import tempfile
import os
import subprocess
from pathlib import Path
from typing import Optional, Callable

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout,
                             QMessageBox, QMenuBar, QMenu, QFileDialog)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QAction, QKeySequence

logger = logging.getLogger(__name__)

# Try to import QScintilla, fall back to QTextEdit if not available
try:
    from PyQt6.Qsci import QsciScintilla, QsciLexerPython
    QSCINTILLA_AVAILABLE = True
    logger.info("QScintilla successfully imported")
except ImportError as e:
    logger.warning(f"QScintilla not available: {e}")
    logger.info("Install with: pip install PyQt6-QScintilla")
    QSCINTILLA_AVAILABLE = False


class SimpleCodeEditorService:
    """
    Simple, modular code editor service.

    Uses QScintilla for professional Python editing (default) or external programs.
    Falls back to QTextEdit if QScintilla is not available.
    No threading - keeps it simple and reliable.
    """
    
    def __init__(self, parent_widget):
        """
        Initialize the code editor service.
        
        Args:
            parent_widget: Parent widget for dialogs
        """
        self.parent = parent_widget
    
    def edit_code(self, initial_content: str, title: str = "Edit Code", 
                  callback: Optional[Callable[[str], None]] = None,
                  use_external: bool = False) -> None:
        """
        Edit code using either Qt native editor or external program.
        
        Args:
            initial_content: Initial code content
            title: Editor window title
            callback: Callback function called with edited content
            use_external: If True, use external editor; if False, use Qt native
        """
        if use_external:
            self._edit_with_external_program(initial_content, callback)
        else:
            self._edit_with_qt_native(initial_content, title, callback)
    
    def _edit_with_qt_native(self, initial_content: str, title: str,
                           callback: Optional[Callable[[str], None]]) -> None:
        """Edit code using Qt native text editor dialog (QScintilla preferred)."""
        try:
            if QSCINTILLA_AVAILABLE:
                logger.debug("Using QScintilla editor for code editing")
                dialog = QScintillaCodeEditorDialog(self.parent, initial_content, title)
            else:
                logger.debug("QScintilla not available, using QTextEdit fallback")
                dialog = CodeEditorDialog(self.parent, initial_content, title)

            if dialog.exec() == QDialog.DialogCode.Accepted:
                edited_content = dialog.get_content()
                if callback:
                    callback(edited_content)

        except Exception as e:
            logger.error(f"Qt native editor failed: {e}")
            self._show_error(f"Editor failed: {str(e)}")
    
    def _edit_with_external_program(self, initial_content: str,
                                  callback: Optional[Callable[[str], None]]) -> None:
        """Edit code using external program (vim, nano, vscode, etc.)."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(initial_content)
                temp_path = Path(f.name)
            
            # Get editor from environment or use default
            editor = os.environ.get('EDITOR', 'nano')
            
            # Launch editor and wait for completion
            result = subprocess.run([editor, str(temp_path)], 
                                  capture_output=False, 
                                  text=True)
            
            if result.returncode == 0:
                # Read edited content
                with open(temp_path, 'r') as f:
                    edited_content = f.read()
                
                if callback:
                    callback(edited_content)
            else:
                self._show_error(f"Editor exited with code {result.returncode}")
                
        except FileNotFoundError:
            self._show_error(f"Editor '{editor}' not found. Set EDITOR environment variable or install nano/vim.")
        except Exception as e:
            logger.error(f"External editor failed: {e}")
            self._show_error(f"External editor failed: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except:
                pass
    
    def _show_error(self, message: str) -> None:
        """Show error message to user."""
        QMessageBox.critical(self.parent, "Editor Error", message)


class QScintillaCodeEditorDialog(QDialog):
    """
    Professional code editor dialog using QScintilla.

    Provides Python syntax highlighting, code folding, line numbers, and more.
    Integrates with PyQt6ColorScheme for consistent theming.
    """

    def __init__(self, parent, initial_content: str, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(900, 700)

        # Get color scheme from parent
        from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
        self.color_scheme = PyQt6ColorScheme()

        # Setup UI
        self._setup_ui(initial_content)

        # Apply theming
        self._apply_theme()

        # Focus on editor
        self.editor.setFocus()



    def _setup_ui(self, initial_content: str):
        """Setup the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)

        # Menu bar
        self._setup_menu_bar()
        main_layout.addWidget(self.menu_bar)

        # QScintilla editor
        self.editor = QsciScintilla()
        self.editor.setText(initial_content)

        # Set Python lexer for syntax highlighting
        self.lexer = QsciLexerPython()
        self.editor.setLexer(self.lexer)

        # Configure editor features
        self._configure_editor()

        main_layout.addWidget(self.editor)

        # Buttons
        button_layout = QHBoxLayout()

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.accept)
        button_layout.addWidget(self.save_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        button_layout.addStretch()
        main_layout.addLayout(button_layout)

    def _setup_menu_bar(self):
        """Setup menu bar with File, Edit, View menus."""
        from PyQt6.QtWidgets import QMenuBar

        self.menu_bar = QMenuBar(self)

        # File menu
        file_menu = self.menu_bar.addMenu("&File")

        # New action
        new_action = QAction("&New", self)
        new_action.setShortcut(QKeySequence.StandardKey.New)
        new_action.triggered.connect(self._new_file)
        file_menu.addAction(new_action)

        # Open action
        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_file)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        # Save action
        save_action = QAction("&Save", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self.accept)
        file_menu.addAction(save_action)

        # Save As action
        save_as_action = QAction("Save &As...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # Close action
        close_action = QAction("&Close", self)
        close_action.setShortcut(QKeySequence.StandardKey.Close)
        close_action.triggered.connect(self.reject)
        file_menu.addAction(close_action)

        # Edit menu
        edit_menu = self.menu_bar.addMenu("&Edit")

        # Undo action
        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(lambda: self.editor.undo())
        edit_menu.addAction(undo_action)

        # Redo action
        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(lambda: self.editor.redo())
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        # Cut, Copy, Paste
        cut_action = QAction("Cu&t", self)
        cut_action.setShortcut(QKeySequence.StandardKey.Cut)
        cut_action.triggered.connect(lambda: self.editor.cut())
        edit_menu.addAction(cut_action)

        copy_action = QAction("&Copy", self)
        copy_action.setShortcut(QKeySequence.StandardKey.Copy)
        copy_action.triggered.connect(lambda: self.editor.copy())
        edit_menu.addAction(copy_action)

        paste_action = QAction("&Paste", self)
        paste_action.setShortcut(QKeySequence.StandardKey.Paste)
        paste_action.triggered.connect(lambda: self.editor.paste())
        edit_menu.addAction(paste_action)

        edit_menu.addSeparator()

        # Select All
        select_all_action = QAction("Select &All", self)
        select_all_action.setShortcut(QKeySequence.StandardKey.SelectAll)
        select_all_action.triggered.connect(lambda: self.editor.selectAll())
        edit_menu.addAction(select_all_action)

        # View menu
        view_menu = self.menu_bar.addMenu("&View")

        # Toggle line numbers
        toggle_line_numbers = QAction("Toggle &Line Numbers", self)
        toggle_line_numbers.setCheckable(True)
        toggle_line_numbers.setChecked(True)
        toggle_line_numbers.triggered.connect(self._toggle_line_numbers)
        view_menu.addAction(toggle_line_numbers)

        # Toggle code folding
        toggle_folding = QAction("Toggle Code &Folding", self)
        toggle_folding.setCheckable(True)
        toggle_folding.setChecked(True)
        toggle_folding.triggered.connect(self._toggle_code_folding)
        view_menu.addAction(toggle_folding)

    def _configure_editor(self):
        """Configure QScintilla editor with professional features."""
        # Line numbers
        self.editor.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
        self.editor.setMarginWidth(0, "0000")
        self.editor.setMarginLineNumbers(0, True)
        self.editor.setMarginsBackgroundColor(Qt.GlobalColor.lightGray)

        # Current line highlighting
        self.editor.setCaretLineVisible(True)
        self.editor.setCaretLineBackgroundColor(Qt.GlobalColor.lightGray)

        # Set font
        font = QFont("Consolas", 10)
        if not font.exactMatch():
            font = QFont("Courier New", 10)
        self.editor.setFont(font)

        # Indentation
        self.editor.setIndentationsUseTabs(False)
        self.editor.setIndentationWidth(4)
        self.editor.setTabWidth(4)
        self.editor.setAutoIndent(True)

        # Code folding
        self.editor.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)

        # Brace matching
        self.editor.setBraceMatching(QsciScintilla.BraceMatch.SloppyBraceMatch)

        # Selection
        self.editor.setSelectionBackgroundColor(Qt.GlobalColor.blue)

        # Enable UTF-8
        self.editor.setUtf8(True)

    def _apply_theme(self):
        """Apply PyQt6ColorScheme theming to QScintilla editor."""
        cs = self.color_scheme

        # Apply dialog styling
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {cs.to_hex(cs.window_bg)};
                color: {cs.to_hex(cs.text_primary)};
            }}
            QPushButton {{
                background-color: {cs.to_hex(cs.button_normal_bg)};
                color: {cs.to_hex(cs.button_text)};
                border: 1px solid {cs.to_hex(cs.border_light)};
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }}
            QPushButton:hover {{
                background-color: {cs.to_hex(cs.button_hover_bg)};
            }}
            QPushButton:pressed {{
                background-color: {cs.to_hex(cs.button_pressed_bg)};
            }}
            QMenuBar {{
                background-color: {cs.to_hex(cs.panel_bg)};
                color: {cs.to_hex(cs.text_primary)};
                border-bottom: 1px solid {cs.to_hex(cs.border_color)};
            }}
            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 8px;
            }}
            QMenuBar::item:selected {{
                background-color: {cs.to_hex(cs.button_hover_bg)};
            }}
            QMenu {{
                background-color: {cs.to_hex(cs.panel_bg)};
                color: {cs.to_hex(cs.text_primary)};
                border: 1px solid {cs.to_hex(cs.border_color)};
            }}
            QMenu::item {{
                padding: 4px 20px;
            }}
            QMenu::item:selected {{
                background-color: {cs.to_hex(cs.button_hover_bg)};
            }}
        """)

        # Apply QScintilla-specific theming
        self._apply_qscintilla_theme()

    def _apply_qscintilla_theme(self):
        """Apply color scheme to QScintilla editor and lexer."""
        cs = self.color_scheme

        # Set editor background and text colors
        self.editor.setColor(cs.to_qcolor(cs.text_primary))
        self.editor.setPaper(cs.to_qcolor(cs.panel_bg))

        # Set margin colors
        self.editor.setMarginsBackgroundColor(cs.to_qcolor(cs.frame_bg))
        self.editor.setMarginsForegroundColor(cs.to_qcolor(cs.text_secondary))

        # Set caret line color
        self.editor.setCaretLineBackgroundColor(cs.to_qcolor(cs.selection_bg))

        # Set selection colors
        self.editor.setSelectionBackgroundColor(cs.to_qcolor(cs.selection_bg))
        self.editor.setSelectionForegroundColor(cs.to_qcolor(cs.selection_text))

        # Configure Python lexer colors
        if self.lexer:
            # Keywords (def, class, if, etc.)
            self.lexer.setColor(cs.to_qcolor(cs.python_keyword_color), QsciLexerPython.Keyword)

            # Strings
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.SingleQuotedString)
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.DoubleQuotedString)
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.TripleSingleQuotedString)
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.TripleDoubleQuotedString)

            # F-strings
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.SingleQuotedFString)
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.DoubleQuotedFString)
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.TripleSingleQuotedFString)
            self.lexer.setColor(cs.to_qcolor(cs.python_string_color), QsciLexerPython.TripleDoubleQuotedFString)

            # Comments
            self.lexer.setColor(cs.to_qcolor(cs.python_comment_color), QsciLexerPython.Comment)
            self.lexer.setColor(cs.to_qcolor(cs.python_comment_color), QsciLexerPython.CommentBlock)

            # Numbers
            self.lexer.setColor(cs.to_qcolor(cs.python_number_color), QsciLexerPython.Number)

            # Functions and classes
            self.lexer.setColor(cs.to_qcolor(cs.python_function_color), QsciLexerPython.FunctionMethodName)
            self.lexer.setColor(cs.to_qcolor(cs.python_class_color), QsciLexerPython.ClassName)

            # Operators
            self.lexer.setColor(cs.to_qcolor(cs.python_operator_color), QsciLexerPython.Operator)

            # Identifiers
            self.lexer.setColor(cs.to_qcolor(cs.python_name_color), QsciLexerPython.Identifier)
            self.lexer.setColor(cs.to_qcolor(cs.python_name_color), QsciLexerPython.HighlightedIdentifier)

            # Decorators
            self.lexer.setColor(cs.to_qcolor(cs.python_function_color), QsciLexerPython.Decorator)

            # Set default background for lexer
            self.lexer.setPaper(cs.to_qcolor(cs.panel_bg))

    # Menu action handlers
    def _new_file(self):
        """Clear editor content."""
        self.editor.clear()

    def _open_file(self):
        """Open file dialog and load content."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Python File", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.editor.setText(content)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open file: {str(e)}")

    def _save_as(self):
        """Save content to file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Python File", "", "Python Files (*.py);;All Files (*)"
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.editor.text())
                QMessageBox.information(self, "Success", f"File saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {str(e)}")

    def _toggle_line_numbers(self, checked):
        """Toggle line number display."""
        if checked:
            self.editor.setMarginType(0, QsciScintilla.MarginType.NumberMargin)
            self.editor.setMarginWidth(0, "0000")
            self.editor.setMarginLineNumbers(0, True)
        else:
            self.editor.setMarginWidth(0, 0)

    def _toggle_code_folding(self, checked):
        """Toggle code folding."""
        if checked:
            self.editor.setFolding(QsciScintilla.FoldStyle.BoxedTreeFoldStyle)
        else:
            self.editor.setFolding(QsciScintilla.FoldStyle.NoFoldStyle)

    def get_content(self) -> str:
        """Get the edited content."""
        return self.editor.text()


class CodeEditorDialog(QDialog):
    """
    Fallback Qt native code editor dialog using QTextEdit.

    Used when QScintilla is not available.
    """

    def __init__(self, parent, initial_content: str, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(800, 600)

        # Setup UI
        layout = QVBoxLayout(self)

        # Text editor
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(initial_content)

        # Use monospace font for code
        font = QFont("Consolas", 10)
        if not font.exactMatch():
            font = QFont("Courier New", 10)
        self.text_edit.setFont(font)

        layout.addWidget(self.text_edit)

        # Buttons
        button_layout = QHBoxLayout()

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Focus on text editor
        self.text_edit.setFocus()

    def get_content(self) -> str:
        """Get the edited content."""
        return self.text_edit.toPlainText()
