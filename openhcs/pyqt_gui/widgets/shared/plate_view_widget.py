"""
Plate View Widget - Visual grid representation of plate wells.

Displays a clickable grid of wells (e.g., A01-H12 for 96-well plate) with visual
states for empty/has-images/selected. Supports multi-select and subdirectory selection.
"""

import logging
from typing import Set, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QPushButton,
    QLabel,
    QFrame,
    QButtonGroup,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator

logger = logging.getLogger(__name__)


class SquareButton(QPushButton):
    """QPushButton that fills its grid cell."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use Expanding policy so buttons fill available space uniformly
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class AspectRatioContainer(QWidget):
    """Container that maintains aspect ratio for its child widget.

    This widget acts as a wrapper that sizes its child to maintain a specific
    aspect ratio while centering it within the available space.
    """

    # Minimum cell size in pixels (wells won't shrink smaller than this)
    MIN_CELL_SIZE = 8

    def __init__(self, child_widget: QWidget, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aspect_ratio = 1.0  # width / height ratio
        self.num_cols = 1
        self.num_rows = 1
        self.child_widget = child_widget
        self.child_widget.setParent(self)
        # Use Expanding to fill available space
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_aspect_ratio(self, num_cols: int, num_rows: int):
        """Set the aspect ratio based on grid dimensions."""
        self.num_cols = num_cols
        self.num_rows = num_rows
        if num_rows > 0:
            self.aspect_ratio = num_cols / num_rows
        # Set minimum size on container so layout respects it
        min_width = num_cols * self.MIN_CELL_SIZE
        min_height = num_rows * self.MIN_CELL_SIZE
        self.setMinimumSize(min_width, min_height)
        self._update_child_geometry()

    def resizeEvent(self, event):
        """Resize child to maintain aspect ratio, centered in available space."""
        super().resizeEvent(event)
        self._update_child_geometry()

    def _update_child_geometry(self):
        """Calculate and set child geometry to maintain aspect ratio."""
        if self.aspect_ratio <= 0:
            return

        available_w = self.width()
        available_h = self.height()

        if available_w <= 0 or available_h <= 0:
            return

        # Calculate minimum size based on cell count
        min_width = self.num_cols * self.MIN_CELL_SIZE
        min_height = self.num_rows * self.MIN_CELL_SIZE

        # Calculate the largest size that fits while maintaining aspect ratio
        height_for_width = int(available_w / self.aspect_ratio)
        width_for_height = int(available_h * self.aspect_ratio)

        if height_for_width <= available_h:
            # Width-constrained: use full width
            child_w = available_w
            child_h = height_for_width
        else:
            # Height-constrained: use full height
            child_w = width_for_height
            child_h = available_h

        # Enforce minimum size (cells won't shrink below MIN_CELL_SIZE)
        child_w = max(child_w, min_width)
        child_h = max(child_h, min_height)

        # Center the child widget (may overflow if below minimum)
        x = (available_w - child_w) // 2
        y = (available_h - child_h) // 2

        self.child_widget.setGeometry(x, y, child_w, child_h)


class PlateViewWidget(QWidget):
    """
    Visual plate grid widget with clickable wells.

    Features:
    - Auto-detects plate dimensions from well IDs
    - Clickable well buttons with visual states (empty/has-images/selected)
    - Multi-select support (Ctrl+Click, Shift+Click)
    - Subdirectory selector for multiple plate outputs
    - Clear selection button
    - Detachable to external window

    Signals:
        wells_selected: Emitted when well selection changes (set of well IDs)
        detach_requested: Emitted when user clicks detach button
    """

    wells_selected = pyqtSignal(set)
    detach_requested = pyqtSignal()

    def __init__(self, color_scheme: Optional[ColorScheme] = None, parent=None):
        super().__init__(parent)

        self.color_scheme = color_scheme or ColorScheme()
        self.style_gen = StyleSheetGenerator(self.color_scheme)

        # State
        self.well_buttons = {}  # well_id -> QPushButton
        self.wells_with_images = set()  # Set of well IDs that have images
        self.selected_wells = set()  # Currently selected wells
        self.plate_dimensions = (8, 12)  # rows, cols (default 96-well)
        self.row_offset = 0  # Offset for tight bounding box (first row index - 1)
        self.col_offset = 0  # Offset for tight bounding box (first col index - 1)
        self.subdirs = []  # List of subdirectory names
        self.active_subdir = None  # Currently selected subdirectory
        self.coord_to_well = {}  # (row_index, col_index) -> well_id mapping
        self.well_to_coord = {}  # well_id -> (row_index, col_index) reverse mapping

        # Drag selection state
        self.is_dragging = False
        self.drag_start_well = None
        self.drag_current_well = None
        self.drag_selection_mode = None  # 'select' or 'deselect'
        self.drag_affected_wells = set()  # Wells affected by current drag operation
        self.pre_drag_selection = set()  # Selection state before drag started
        self.drag_moved = False  # Track if mouse actually moved during drag

        # Rectangle selection state (for dragging in empty space)
        self.is_rect_selecting = False
        self.rect_start_pos = None
        self.rect_current_pos = None
        self.selection_rect_widget = None  # Visual rectangle overlay

        # Column filter sync
        self.well_filter_widget = (
            None  # Reference to ColumnFilterWidget for 'well' column
        )

        # UI components
        self.subdir_buttons = {}  # subdir_name -> QPushButton
        self.subdir_button_group = None
        self.well_grid_layout = None
        self.status_label = None

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Header with title, detach button, and clear button
        header_layout = QHBoxLayout()
        title_label = QLabel("Plate View")
        title_label.setStyleSheet(
            f"font-weight: bold; color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};"
        )
        header_layout.addWidget(title_label)

        header_layout.addStretch()

        detach_btn = QPushButton("↗")
        detach_btn.setToolTip("Detach to separate window")
        detach_btn.setFixedWidth(30)
        detach_btn.setStyleSheet(self.style_gen.generate_button_style())
        detach_btn.clicked.connect(lambda: self.detach_requested.emit())
        header_layout.addWidget(detach_btn)

        clear_btn = QPushButton("Clear Selection")
        clear_btn.setStyleSheet(self.style_gen.generate_button_style())
        # Use lambda to avoid clicked signal's bool arg being passed to clear_selection
        clear_btn.clicked.connect(lambda: self.clear_selection())
        header_layout.addWidget(clear_btn)

        layout.addLayout(header_layout)

        # Subdirectory selector (initially hidden)
        self.subdir_frame = QFrame()
        self.subdir_layout = QHBoxLayout(self.subdir_frame)
        self.subdir_layout.setContentsMargins(0, 0, 0, 0)
        self.subdir_layout.setSpacing(5)

        subdir_label = QLabel("Plate Output:")
        self.subdir_layout.addWidget(subdir_label)

        self.subdir_button_group = QButtonGroup(self)
        self.subdir_button_group.setExclusive(True)

        self.subdir_layout.addStretch()
        self.subdir_frame.setVisible(False)
        layout.addWidget(self.subdir_frame)

        # Well grid container (background color shows through as grid lines)
        grid_container = QFrame()
        # Use panel_bg color for grid lines (shows between 1px spacing)
        grid_container.setStyleSheet(
            f"background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)}; border-radius: 3px;"
        )
        grid_layout_wrapper = QVBoxLayout(grid_container)
        grid_layout_wrapper.setContentsMargins(10, 10, 10, 10)

        # Inner grid widget that holds the actual grid layout
        inner_grid_widget = QWidget()
        inner_grid_widget.setMouseTracking(True)
        self.well_grid_layout = QGridLayout(inner_grid_widget)
        self.well_grid_layout.setSpacing(2)  # Thin grid lines (2px for visibility)
        self.well_grid_layout.setContentsMargins(0, 0, 0, 0)

        # Wrap in AspectRatioContainer to maintain square cells
        aspect_container = AspectRatioContainer(inner_grid_widget)
        self.grid_widget = inner_grid_widget  # Store reference to inner widget
        self.aspect_container = aspect_container  # Store reference to container

        # Install event filter on inner grid widget for rectangle selection
        inner_grid_widget.installEventFilter(self)

        # Create selection rectangle overlay (initially hidden)
        self.selection_rect_widget = QLabel(inner_grid_widget)
        self.selection_rect_widget.setStyleSheet(f"""
            background-color: {self.color_scheme.to_hex(self.color_scheme.selection_bg)}40;
            border: 2px solid {self.color_scheme.to_hex(self.color_scheme.selection_bg)};
        """)
        self.selection_rect_widget.hide()
        self.selection_rect_widget.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.selection_rect_widget.raise_()  # Ensure it's on top

        # Add aspect container to wrapper (it will expand and center the grid)
        grid_layout_wrapper.addWidget(aspect_container, 1)  # stretch factor 1 to expand

        layout.addWidget(grid_container, 1)  # Stretch to fill

        # Status label
        self.status_label = QLabel("No wells")
        self.status_label.setStyleSheet(
            f"color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};"
        )
        layout.addWidget(self.status_label)

    def set_subdirectories(self, subdirs: List[str]):
        """
        Set available subdirectories for plate outputs.

        Args:
            subdirs: List of subdirectory names
        """
        self.subdirs = subdirs

        # Clear existing buttons
        for btn in self.subdir_buttons.values():
            self.subdir_button_group.removeButton(btn)
            btn.deleteLater()
        self.subdir_buttons.clear()

        if len(subdirs) == 0:
            # No subdirs, hide selector
            self.subdir_frame.setVisible(False)
            self.active_subdir = None
        elif len(subdirs) == 1:
            # Single subdir, auto-select and hide selector
            self.subdir_frame.setVisible(False)
            self.active_subdir = subdirs[0]
        else:
            # Multiple subdirs, show selector
            self.subdir_frame.setVisible(True)

            # Create button for each subdir
            for subdir in subdirs:
                btn = QPushButton(subdir)
                btn.setCheckable(True)
                btn.setStyleSheet(self.style_gen.generate_button_style())
                btn.clicked.connect(
                    lambda checked, s=subdir: self._on_subdir_selected(s)
                )

                self.subdir_button_group.addButton(btn)
                self.subdir_layout.insertWidget(
                    self.subdir_layout.count() - 1, btn
                )  # Before stretch
                self.subdir_buttons[subdir] = btn

            # Auto-select first subdir
            if subdirs:
                first_btn = self.subdir_buttons[subdirs[0]]
                first_btn.setChecked(True)
                self.active_subdir = subdirs[0]

    def _on_subdir_selected(self, subdir: str):
        """Handle subdirectory selection."""
        self.active_subdir = subdir
        # Could emit signal here if needed for filtering by subdir

    def set_available_wells(
        self,
        well_ids: Set[str],
        plate_dimensions: Optional[Tuple[int, int]] = None,
        coord_to_well: Optional[dict] = None,
    ):
        """
        Update which wells have images and rebuild grid.

        Args:
            well_ids: Set of well IDs that have images
            plate_dimensions: Optional (rows, cols) tuple. If None, auto-detects from well IDs.
            coord_to_well: Optional mapping from (row_index, col_index) to well_id.
                          Required for non-standard well ID formats (e.g., Opera Phenix R01C01).
        """
        self.wells_with_images = well_ids

        # If coord_to_well not provided, build it from well_ids (for standard formats)
        # IMPORTANT: Do this BEFORE calling _detect_dimensions
        if coord_to_well is None:
            self.coord_to_well = {}
            for well_id in well_ids:
                # Parse standard well ID format (A01, B12, etc.)
                row_part = "".join(c for c in well_id if c.isalpha())
                col_part = "".join(c for c in well_id if c.isdigit())

                if row_part and col_part:
                    # Convert row letter to index (A=1, B=2, etc.)
                    row_idx = sum(
                        (ord(c.upper()) - ord("A") + 1) * (26**i)
                        for i, c in enumerate(reversed(row_part))
                    )
                    col_idx = int(col_part)
                    self.coord_to_well[(row_idx, col_idx)] = well_id
        else:
            self.coord_to_well = coord_to_well

        # Build reverse mapping (well_id -> coord)
        self.well_to_coord = {
            well_id: coord for coord, well_id in self.coord_to_well.items()
        }

        if not well_ids:
            self._clear_grid()
            self.status_label.setText("No wells")
            return

        # Use provided dimensions or auto-detect (NOW coord_to_well is populated)
        if plate_dimensions is not None:
            self.plate_dimensions = plate_dimensions
        else:
            self.plate_dimensions = self._detect_dimensions(well_ids)

        # Rebuild grid
        self._build_grid()

        # Update status
        self._update_status()

    def _detect_dimensions(self, well_ids: Set[str]) -> Tuple[int, int]:
        """
        Auto-detect plate dimensions from well IDs (tight bounding box).

        Only includes rows/columns that contain wells with images.

        Args:
            well_ids: Set of well IDs

        Returns:
            (rows, cols) tuple
        """
        if not well_ids:
            return (8, 12)  # Default

        min_row = float("inf")
        max_row = 0
        min_col = float("inf")
        max_col = 0

        # If we have coord_to_well mapping, use that directly
        if self.coord_to_well:
            for (row_idx, col_idx), well_id in self.coord_to_well.items():
                if well_id in well_ids:
                    min_row = min(min_row, row_idx)
                    max_row = max(max_row, row_idx)
                    min_col = min(min_col, col_idx)
                    max_col = max(max_col, col_idx)
        else:
            # Parse well IDs to extract coordinates
            for well_id in well_ids:
                # Extract row letter(s) and column number(s)
                row_part = "".join(c for c in well_id if c.isalpha())
                col_part = "".join(c for c in well_id if c.isdigit())

                if row_part:
                    # Convert row letter to index (A=1, B=2, AA=27, etc.)
                    row_idx = sum(
                        (ord(c.upper()) - ord("A") + 1) * (26**i)
                        for i, c in enumerate(reversed(row_part))
                    )
                    min_row = min(min_row, row_idx)
                    max_row = max(max_row, row_idx)

                if col_part:
                    col_idx = int(col_part)
                    min_col = min(min_col, col_idx)
                    max_col = max(max_col, col_idx)

        # Store offset for coordinate mapping
        self.row_offset = min_row - 1 if min_row != float("inf") else 0
        self.col_offset = min_col - 1 if min_col != float("inf") else 0

        rows = max_row - min_row + 1 if max_row > 0 else 8
        cols = max_col - min_col + 1 if max_col > 0 else 12

        return (rows, cols)

    def _clear_grid(self):
        """Clear the well grid."""
        for btn in self.well_buttons.values():
            btn.deleteLater()
        self.well_buttons.clear()

        # Clear layout
        while self.well_grid_layout.count():
            item = self.well_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_grid(self):
        """Build the well grid based on current dimensions."""
        self._clear_grid()

        # Get unique rows and columns that actually have wells
        actual_rows = sorted(set(coord[0] for coord in self.coord_to_well.keys()))
        actual_cols = sorted(set(coord[1] for coord in self.coord_to_well.keys()))

        if not actual_rows or not actual_cols:
            return

        # Get bounding rectangle (min to max, inclusive)
        min_row, max_row = min(actual_rows), max(actual_rows)
        min_col, max_col = min(actual_cols), max(actual_cols)

        # Create full range including empty positions
        all_rows = list(range(min_row, max_row + 1))
        all_cols = list(range(min_col, max_col + 1))

        # Calculate minimum size based on label width
        # Find longest column number for width calculation
        max_col_label = str(max_col)
        from PyQt6.QtGui import QFontMetrics
        from PyQt6.QtGui import QFont

        font = QFont()
        font.setPointSize(10)
        fm = QFontMetrics(font)
        min_col_width = max(
            fm.horizontalAdvance(max_col_label) + 8, 15
        )  # +8 for padding, min 15px
        min_row_height = max(fm.height() + 4, 15)  # +4 for padding, min 15px

        # Header minimum sizes (can be smaller than well size)
        min_header_width = max(min_col_width, 8)
        min_header_height = max(min_row_height, 8)

        # Well buttons use MIN_CELL_SIZE directly - allow shrinking independently of headers
        min_well_size = AspectRatioContainer.MIN_CELL_SIZE

        # Top-left corner: Invert selection button
        invert_btn = QPushButton("⇄")
        invert_btn.setFlat(True)
        invert_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        invert_btn.setMinimumSize(18, 18)
        invert_btn.setToolTip("Invert Selection")
        invert_btn.setStyleSheet(f"""
            QPushButton {{
                color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};
                font-size: 12px;
                border: none;
                background: transparent;
            }}
            QPushButton:hover {{
                color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
            }}
        """)
        invert_btn.clicked.connect(lambda: self._invert_selection())
        self.well_grid_layout.addWidget(invert_btn, 0, 0)

        # Add column headers - for all columns in bounding rectangle
        for grid_col, actual_col in enumerate(all_cols, start=1):
            header = QPushButton(str(actual_col))
            header.setFlat(True)
            header.setCursor(Qt.CursorShape.PointingHandCursor)
            header.setMinimumSize(min_header_width, 18)
            header.setMaximumHeight(18)
            header.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            header.setStyleSheet(f"""
                QPushButton {{
                    color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};
                    font-size: 10px;
                    border: none;
                    background: transparent;
                }}
                QPushButton:hover {{
                    color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                    background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                }}
            """)
            header.clicked.connect(
                lambda checked, c=actual_col: self._toggle_column_selection(c)
            )
            self.well_grid_layout.addWidget(header, 0, grid_col)

        # Add row headers and well buttons - for all rows in bounding rectangle
        for grid_row, actual_row in enumerate(all_rows, start=1):
            # Row header (A, B, C, ...)
            row_letter = self._index_to_row_letter(actual_row)
            header = QPushButton(row_letter)
            header.setFlat(True)
            header.setCursor(Qt.CursorShape.PointingHandCursor)
            header.setMinimumSize(18, min_header_height)
            header.setMaximumWidth(18)
            header.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Ignored)
            header.setStyleSheet(f"""
                QPushButton {{
                    color: {self.color_scheme.to_hex(self.color_scheme.text_secondary)};
                    font-size: 10px;
                    border: none;
                    background: transparent;
                }}
                QPushButton:hover {{
                    color: {self.color_scheme.to_hex(self.color_scheme.text_primary)};
                    background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                }}
            """)
            header.clicked.connect(
                lambda checked, r=actual_row: self._toggle_row_selection(r)
            )
            self.well_grid_layout.addWidget(header, grid_row, 0)

            # Well buttons - for all columns in bounding rectangle
            for grid_col, actual_col in enumerate(all_cols, start=1):
                # Check if this coordinate has a well
                well_id = self.coord_to_well.get((actual_row, actual_col))

                btn = SquareButton()  # Use SquareButton to maintain 1:1 aspect ratio
                btn.setMinimumSize(min_well_size, min_well_size)
                btn.setCheckable(True)

                if well_id and well_id in self.wells_with_images:
                    # Well exists and has images
                    btn.setEnabled(True)
                    btn.setStyleSheet(self._get_well_button_style("has_images"))
                    btn.clicked.connect(
                        lambda checked, wid=well_id: self._on_well_clicked(wid, checked)
                    )
                    btn.setProperty("well_id", well_id)
                    btn.installEventFilter(self)
                    self.well_buttons[well_id] = btn
                else:
                    # Empty position in bounding rectangle
                    btn.setEnabled(False)
                    btn.setStyleSheet(self._get_well_button_style("empty"))
                    btn.setProperty("well_id", None)
                    # Install event filter for rectangle selection
                    btn.installEventFilter(self)

                self.well_grid_layout.addWidget(btn, grid_row, grid_col)

        # Set uniform column and row stretches so all cells get equal space
        # This ensures wells expand uniformly and stay aligned
        for grid_col in range(1, len(all_cols) + 1):
            self.well_grid_layout.setColumnStretch(grid_col, 1)
        for grid_row in range(1, len(all_rows) + 1):
            self.well_grid_layout.setRowStretch(grid_row, 1)

        # Set aspect ratio on container to maintain square wells
        # Add 1 to account for header row/column
        self.aspect_container.set_aspect_ratio(len(all_cols) + 1, len(all_rows) + 1)

    def _index_to_row_letter(self, index: int) -> str:
        """Convert row index to letter(s) (1=A, 2=B, 27=AA, etc.)."""
        result = ""
        while index > 0:
            index -= 1
            result = chr(ord("A") + (index % 26)) + result
            index //= 26
        return result

    def _get_well_button_style(self, state: str) -> str:
        """Generate style for well button based on state."""
        cs = self.color_scheme

        if state == "empty":
            return f"""
                QPushButton {{
                    background-color: {cs.to_hex(cs.button_disabled_bg)};
                    color: {cs.to_hex(cs.button_disabled_text)};
                    border: none;
                    border-radius: 3px;
                }}
            """
        elif state == "selected":
            return f"""
                QPushButton {{
                    background-color: {cs.to_hex(cs.selection_bg)};
                    color: {cs.to_hex(cs.selection_text)};
                    border: 2px solid {cs.to_hex(cs.border_color)};
                    border-radius: 3px;
                }}
            """
        else:  # has_images
            return f"""
                QPushButton {{
                    background-color: {cs.to_hex(cs.button_normal_bg)};
                    color: {cs.to_hex(cs.button_text)};
                    border: none;
                    border-radius: 3px;
                }}
                QPushButton:hover {{
                    background-color: {cs.to_hex(cs.button_hover_bg)};
                }}
            """

    def _on_well_clicked(self, well_id: str, checked: bool):
        """Handle well button click (only for non-drag clicks)."""
        # Skip if this was part of a drag operation
        if self.is_dragging:
            return

        if checked:
            self.selected_wells.add(well_id)
            self.well_buttons[well_id].setStyleSheet(
                self._get_well_button_style("selected")
            )
        else:
            self.selected_wells.discard(well_id)
            self.well_buttons[well_id].setStyleSheet(
                self._get_well_button_style("has_images")
            )

        self._update_status()
        self.sync_to_well_filter()
        self.wells_selected.emit(self.selected_wells.copy())

    def clear_selection(self, emit_signal: bool = True, sync_to_filter: bool = True):
        """
        Clear all selected wells.

        Args:
            emit_signal: Whether to emit wells_selected signal (default True)
            sync_to_filter: Whether to sync to well filter (default True)
        """
        logger.info(
            f"[CLEAR] clear_selection called, had {len(self.selected_wells)} wells, emit_signal={emit_signal}, sync_to_filter={sync_to_filter}"
        )
        for well_id in list(self.selected_wells):
            if well_id in self.well_buttons:
                btn = self.well_buttons[well_id]
                btn.setChecked(False)
                btn.setStyleSheet(self._get_well_button_style("has_images"))

        self.selected_wells.clear()
        self._update_status()
        logger.info("[CLEAR] After clear, about to sync")

        # Sync to well filter BEFORE emitting signal
        # This ensures the Well column filter has all values selected
        # before _apply_combined_filters() runs
        if sync_to_filter:
            logger.info("[CLEAR] Syncing to well filter")
            self.sync_to_well_filter()
            logger.info("[CLEAR] Done syncing to well filter")

        if emit_signal:
            logger.info("[CLEAR] Emitting wells_selected with empty set")
            self.wells_selected.emit(set())
            logger.info("[CLEAR] Done emitting")

    def select_wells(self, well_ids: Set[str], emit_signal: bool = True):
        """
        Programmatically select wells.

        Args:
            well_ids: Set of well IDs to select
            emit_signal: Whether to emit wells_selected signal (default True)
        """
        # Clear without syncing to filter (we'll sync after setting new selection)
        self.clear_selection(emit_signal=False, sync_to_filter=False)

        for well_id in well_ids:
            if well_id in self.well_buttons and well_id in self.wells_with_images:
                self.selected_wells.add(well_id)
                btn = self.well_buttons[well_id]
                btn.setChecked(True)
                btn.setStyleSheet(self._get_well_button_style("selected"))

        self._update_status()
        if emit_signal:
            self.sync_to_well_filter()
            self.wells_selected.emit(self.selected_wells.copy())

    def _update_status(self):
        """Update status label."""
        total_wells = len(self.wells_with_images)
        selected_count = len(self.selected_wells)

        if selected_count > 0:
            self.status_label.setText(
                f"{total_wells} wells have images | {selected_count} selected"
            )
        else:
            self.status_label.setText(f"{total_wells} wells have images")

    def eventFilter(self, obj, event):
        """Handle mouse events on buttons for drag selection and rectangle selection in empty space."""
        from PyQt6.QtCore import QEvent, QRect

        # Handle events from well buttons (both active and empty)
        if isinstance(obj, QPushButton):
            well_id = obj.property("well_id")

            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Always start rectangle selection for visual feedback
                    self.is_rect_selecting = True
                    self.rect_start_pos = obj.mapTo(self.grid_widget, event.pos())
                    self.rect_current_pos = self.rect_start_pos
                    self.pre_drag_selection = self.selected_wells.copy()

                    # Grab mouse on grid widget so we get events even outside buttons
                    self.grid_widget.grabMouse()

                    # Show rectangle at starting position (even if small)
                    rect = QRect(
                        self.rect_start_pos, self.rect_current_pos
                    ).normalized()
                    self.selection_rect_widget.setGeometry(rect)
                    self.selection_rect_widget.raise_()  # Bring to front
                    self.selection_rect_widget.show()

                    if well_id and well_id in self.wells_with_images:
                        # Also track drag selection on active well for immediate toggle
                        self.is_dragging = True
                        self.drag_start_well = well_id
                        self.drag_current_well = well_id
                        self.drag_affected_wells = set()
                        self.drag_moved = False  # Track if mouse actually moved

                        # Determine selection mode
                        self.drag_selection_mode = (
                            "deselect" if well_id in self.selected_wells else "select"
                        )

                        # Apply to starting well
                        self._toggle_well_selection(
                            well_id, self.drag_selection_mode == "select"
                        )
                        self.drag_affected_wells.add(well_id)

                        # Emit signal immediately for drag start
                        self._update_status()
                        self.sync_to_well_filter()
                        self.wells_selected.emit(self.selected_wells.copy())

                    # Accept event to prevent button's clicked signal
                    event.accept()
                    return True

            elif event.type() == QEvent.Type.MouseMove:
                if (
                    self.is_rect_selecting
                    and event.buttons() & Qt.MouseButton.LeftButton
                ):
                    # Update rectangle from button (always update rectangle during any drag)
                    self.rect_current_pos = obj.mapTo(self.grid_widget, event.pos())
                    rect = QRect(
                        self.rect_start_pos, self.rect_current_pos
                    ).normalized()
                    self.selection_rect_widget.setGeometry(rect)
                    self.selection_rect_widget.raise_()  # Keep on top

                    # Find wells inside rectangle and select them
                    self._update_rectangle_selection(rect)

                    if self.is_dragging:
                        self.drag_moved = True  # Mark that we actually dragged

                    # Accept event to prevent default handling
                    event.accept()
                    return True

            elif event.type() == QEvent.Type.MouseButtonRelease:
                if event.button() == Qt.MouseButton.LeftButton:
                    if self.is_rect_selecting or self.is_dragging:
                        # Release mouse grab
                        self.grid_widget.releaseMouse()

                        # End both drag and rectangle selection
                        if self.is_dragging:
                            self.is_dragging = False
                            self.drag_start_well = None
                            self.drag_current_well = None
                            self.drag_selection_mode = None
                            self.drag_affected_wells.clear()
                            self.drag_moved = False

                        if self.is_rect_selecting:
                            self.is_rect_selecting = False
                            self.rect_start_pos = None
                            self.rect_current_pos = None
                            self.selection_rect_widget.hide()

                        # Accept event to prevent button's clicked signal
                        event.accept()
                        return True

        # Handle events from grid widget (empty space or grabbed mouse) for rectangle selection
        elif obj == self.grid_widget:
            if event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    # Check if we clicked on empty space (not a button)
                    target = self.grid_widget.childAt(event.pos())
                    if not isinstance(target, QPushButton):
                        # Start rectangle selection
                        self.is_rect_selecting = True
                        self.rect_start_pos = event.pos()
                        self.rect_current_pos = event.pos()
                        self.pre_drag_selection = self.selected_wells.copy()

                        # Grab mouse so we get events even outside the widget
                        self.grid_widget.grabMouse()

                        # Show rectangle at starting position
                        rect = QRect(
                            self.rect_start_pos, self.rect_current_pos
                        ).normalized()
                        self.selection_rect_widget.setGeometry(rect)
                        self.selection_rect_widget.raise_()  # Bring to front
                        self.selection_rect_widget.show()

                        event.accept()
                        return True

            elif event.type() == QEvent.Type.MouseMove:
                if (
                    self.is_rect_selecting
                    and event.buttons() & Qt.MouseButton.LeftButton
                ):
                    # Update rectangle
                    self.rect_current_pos = event.pos()
                    rect = QRect(
                        self.rect_start_pos, self.rect_current_pos
                    ).normalized()
                    self.selection_rect_widget.setGeometry(rect)
                    self.selection_rect_widget.raise_()  # Keep on top

                    # Find wells inside rectangle and select them
                    self._update_rectangle_selection(rect)

                    event.accept()
                    return True

            elif event.type() == QEvent.Type.MouseButtonRelease:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and self.is_rect_selecting
                ):
                    # Release mouse grab
                    self.grid_widget.releaseMouse()

                    # End rectangle selection
                    self.is_rect_selecting = False
                    self.rect_start_pos = None
                    self.rect_current_pos = None
                    self.selection_rect_widget.hide()

                    event.accept()
                    return True

        return super().eventFilter(obj, event)

    def _toggle_well_selection(self, well_id: str, select: bool):
        """Toggle selection state of a single well."""
        if well_id not in self.well_buttons or well_id not in self.wells_with_images:
            return

        btn = self.well_buttons[well_id]

        if select and well_id not in self.selected_wells:
            self.selected_wells.add(well_id)
            btn.setChecked(True)
            btn.setStyleSheet(self._get_well_button_style("selected"))
        elif not select and well_id in self.selected_wells:
            self.selected_wells.discard(well_id)
            btn.setChecked(False)
            btn.setStyleSheet(self._get_well_button_style("has_images"))

    def _update_drag_selection(self):
        """Update selection for all wells in the drag rectangle."""
        if not self.drag_start_well or not self.drag_current_well:
            return

        # Get coordinates
        start_coord = self.well_to_coord.get(self.drag_start_well)
        current_coord = self.well_to_coord.get(self.drag_current_well)

        if not start_coord or not current_coord:
            return

        # Calculate rectangle bounds
        min_row = min(start_coord[0], current_coord[0])
        max_row = max(start_coord[0], current_coord[0])
        min_col = min(start_coord[1], current_coord[1])
        max_col = max(start_coord[1], current_coord[1])

        # Find all wells in current rectangle
        wells_in_rectangle = set()
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                well_id = self.coord_to_well.get((row, col))
                if well_id and well_id in self.wells_with_images:
                    wells_in_rectangle.add(well_id)

        # Revert wells that were affected by previous drag but are no longer in rectangle
        # Restore them to their pre-drag state
        wells_to_revert = self.drag_affected_wells - wells_in_rectangle
        for well_id in wells_to_revert:
            was_selected_before_drag = well_id in self.pre_drag_selection
            self._toggle_well_selection(well_id, was_selected_before_drag)

        # Apply selection to all wells in current rectangle
        for well_id in wells_in_rectangle:
            self._toggle_well_selection(well_id, self.drag_selection_mode == "select")

        # Update affected wells to current rectangle
        self.drag_affected_wells = wells_in_rectangle.copy()

        # Sync to well filter BEFORE emitting signal
        self._update_status()
        self.sync_to_well_filter()
        self.wells_selected.emit(self.selected_wells.copy())

    def _update_rectangle_selection(self, rect):
        """Update selection based on rectangle drawn in empty space."""
        from PyQt6.QtCore import QRect

        # Find all wells whose buttons intersect with the rectangle
        wells_in_rect = set()
        for well_id, btn in self.well_buttons.items():
            if well_id not in self.wells_with_images:
                continue

            # Get button geometry in grid widget coordinates
            btn_rect = QRect(btn.pos(), btn.size())

            # Check if button intersects with selection rectangle
            if rect.intersects(btn_rect):
                wells_in_rect.add(well_id)

        # Restore pre-drag selection for wells outside the rectangle
        for well_id in self.wells_with_images:
            if well_id not in wells_in_rect:
                # Restore to pre-drag state
                should_be_selected = well_id in self.pre_drag_selection
                self._toggle_well_selection(well_id, should_be_selected)
            else:
                # Select wells in rectangle
                self._toggle_well_selection(well_id, True)

        # Sync to well filter BEFORE emitting signal
        self._update_status()
        self.sync_to_well_filter()
        self.wells_selected.emit(self.selected_wells.copy())

    def _toggle_row_selection(self, row_index: int):
        """Toggle selection for all wells in a row."""
        # Find all wells in this row
        wells_in_row = []
        for well_id, coord in self.well_to_coord.items():
            if coord[0] == row_index and well_id in self.wells_with_images:
                wells_in_row.append(well_id)

        if not wells_in_row:
            return

        # Check if all wells in row are selected
        all_selected = all(well_id in self.selected_wells for well_id in wells_in_row)

        # If all selected, deselect all; otherwise select all
        select = not all_selected

        for well_id in wells_in_row:
            self._toggle_well_selection(well_id, select)

        # Sync to well filter BEFORE emitting signal
        self._update_status()
        self.sync_to_well_filter()
        self.wells_selected.emit(self.selected_wells.copy())

    def _toggle_column_selection(self, col_index: int):
        """Toggle selection for all wells in a column."""
        # Find all wells in this column
        wells_in_col = []
        for well_id, coord in self.well_to_coord.items():
            if coord[1] == col_index and well_id in self.wells_with_images:
                wells_in_col.append(well_id)

        if not wells_in_col:
            return

        # Check if all wells in column are selected
        all_selected = all(well_id in self.selected_wells for well_id in wells_in_col)

        # If all selected, deselect all; otherwise select all
        select = not all_selected

        for well_id in wells_in_col:
            self._toggle_well_selection(well_id, select)

        # Sync to well filter BEFORE emitting signal
        self._update_status()
        self.sync_to_well_filter()
        self.wells_selected.emit(self.selected_wells.copy())

    def _invert_selection(self):
        """Invert the current selection (select unselected, deselect selected)."""
        # Get all wells with images
        all_wells = self.wells_with_images.copy()

        logger.info(
            f"[INVERT] Before: selected_wells={len(self.selected_wells)}, all_wells={len(all_wells)}"
        )
        logger.info(f"[INVERT] selected_wells={self.selected_wells}")

        # Calculate inverted selection
        new_selection = all_wells - self.selected_wells
        logger.info(f"[INVERT] new_selection={len(new_selection)} wells")

        # Clear current selection
        for well_id in self.selected_wells.copy():
            self._toggle_well_selection(well_id, False)

        # Apply new selection
        for well_id in new_selection:
            self._toggle_well_selection(well_id, True)

        logger.info(f"[INVERT] After: selected_wells={len(self.selected_wells)}")
        logger.info(f"[INVERT] selected_wells={self.selected_wells}")

        # Sync to well filter BEFORE emitting signal
        # This ensures the Well column filter is updated before _apply_combined_filters() runs
        self._update_status()
        self.sync_to_well_filter()
        self.wells_selected.emit(self.selected_wells.copy())

    def set_well_filter_widget(self, well_filter_widget):
        """
        Set reference to the well column filter widget for bidirectional sync.

        Args:
            well_filter_widget: ColumnFilterWidget instance for the 'well' column
        """
        self.well_filter_widget = well_filter_widget

    def sync_to_well_filter(self):
        """Sync current plate view selection to well filter checkboxes."""
        if not self.well_filter_widget:
            return

        # Block signals for performance - table updates via wells_selected signal
        if self.selected_wells:
            self.well_filter_widget.set_selected_values(
                self.selected_wells, block_signals=True
            )
        else:
            self.well_filter_widget.select_all(block_signals=True)

    def sync_from_well_filter(self):
        """Sync well filter checkbox selection to plate view."""
        if not self.well_filter_widget:
            return

        # Get selected wells from filter
        selected_in_filter = self.well_filter_widget.get_selected_values()

        # Update plate view to match (without emitting signal to avoid loop)
        self.select_wells(selected_in_filter, emit_signal=False)
