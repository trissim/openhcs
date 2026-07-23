"""
Plate View Widget - Visual grid representation of plate wells.

Displays a clickable grid of wells (e.g., A01-H12 for 96-well plate) with visual
states for empty/has-images/selected. Supports multi-select and subdirectory selection.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
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
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QEvent, QRect
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.theming import StyleSheetGenerator

logger = logging.getLogger(__name__)


class PlateSelectionEventTarget(Enum):
    """Event source handled by plate selection routing."""

    BUTTON = "button"
    GRID = "grid"


class PlateSubdirectoryMode(Enum):
    """Display mode for the plate-output subdirectory selector."""

    NONE = "none"
    SINGLE = "single"
    MULTIPLE = "multiple"

    @classmethod
    def from_count(cls, count: int) -> "PlateSubdirectoryMode":
        if count == 0:
            return cls.NONE
        if count == 1:
            return cls.SINGLE
        return cls.MULTIPLE


class WellButtonState(Enum):
    """Visual state for a well button."""

    EMPTY = "empty"
    HAS_IMAGES = "has_images"
    SELECTED = "selected"


@dataclass(frozen=True, slots=True)
class WellButtonStyleColors:
    """Color attribute names used to render a well button state."""

    background: str
    text: str
    border: str | None = None
    hover_background: str | None = None


WELL_BUTTON_STYLE_COLORS: dict[WellButtonState, WellButtonStyleColors] = {
    WellButtonState.EMPTY: WellButtonStyleColors(
        background="button_disabled_bg",
        text="button_disabled_text",
    ),
    WellButtonState.HAS_IMAGES: WellButtonStyleColors(
        background="button_normal_bg",
        text="button_text",
        hover_background="button_hover_bg",
    ),
    WellButtonState.SELECTED: WellButtonStyleColors(
        background="selection_bg",
        text="selection_text",
        border="border_color",
    ),
}


@dataclass(frozen=True, slots=True)
class PlateGridBounds:
    """Tight bounding rectangle for wells in plate coordinates."""

    min_row: int
    max_row: int
    min_col: int
    max_col: int

    @property
    def rows(self) -> list[int]:
        return list(range(self.min_row, self.max_row + 1))

    @property
    def cols(self) -> list[int]:
        return list(range(self.min_col, self.max_col + 1))

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self.max_row - self.min_row + 1, self.max_col - self.min_col + 1)


@dataclass(frozen=True, slots=True)
class PlateGridModel:
    """Pure plate-coordinate model backing the Qt grid facade."""

    wells_with_images: frozenset[str]
    coord_to_well: dict[tuple[int, int], str]
    well_to_coord: dict[str, tuple[int, int]]
    plate_dimensions: tuple[int, int]
    row_offset: int
    col_offset: int
    bounds: PlateGridBounds | None

    @classmethod
    def empty(cls) -> "PlateGridModel":
        return cls(
            wells_with_images=frozenset(),
            coord_to_well={},
            well_to_coord={},
            plate_dimensions=(8, 12),
            row_offset=0,
            col_offset=0,
            bounds=None,
        )

    @classmethod
    def from_wells(
        cls,
        well_ids: Set[str],
        plate_dimensions: Optional[Tuple[int, int]] = None,
        coord_to_well: Optional[dict] = None,
    ) -> "PlateGridModel":
        coordinates = (
            dict(coord_to_well)
            if coord_to_well is not None
            else cls._coordinates_from_standard_well_ids(well_ids)
        )
        reverse_coordinates = {
            well_id: coord for coord, well_id in coordinates.items()
        }

        if not well_ids:
            return cls.empty()

        bounds = cls._detect_bounds(well_ids, coordinates)
        dimensions = (
            plate_dimensions
            if plate_dimensions is not None
            else (bounds.dimensions if bounds is not None else (8, 12))
        )

        return cls(
            wells_with_images=frozenset(well_ids),
            coord_to_well=coordinates,
            well_to_coord=reverse_coordinates,
            plate_dimensions=dimensions,
            row_offset=(bounds.min_row - 1) if bounds is not None else 0,
            col_offset=(bounds.min_col - 1) if bounds is not None else 0,
            bounds=bounds,
        )

    @classmethod
    def _coordinates_from_standard_well_ids(
        cls, well_ids: Set[str]
    ) -> dict[tuple[int, int], str]:
        coordinates: dict[tuple[int, int], str] = {}
        for well_id in well_ids:
            coord = cls._parse_standard_well_id(well_id)
            if coord is not None:
                coordinates[coord] = well_id
        return coordinates

    @staticmethod
    def _parse_standard_well_id(well_id: str) -> tuple[int, int] | None:
        row_part = "".join(c for c in well_id if c.isalpha())
        col_part = "".join(c for c in well_id if c.isdigit())
        if not row_part or not col_part:
            return None

        row_idx = sum(
            (ord(c.upper()) - ord("A") + 1) * (26**i)
            for i, c in enumerate(reversed(row_part))
        )
        return (row_idx, int(col_part))

    @classmethod
    def _detect_bounds(
        cls, well_ids: Set[str], coord_to_well: dict[tuple[int, int], str]
    ) -> PlateGridBounds | None:
        occupied = [
            coord for coord, well_id in coord_to_well.items() if well_id in well_ids
        ]
        if not occupied:
            return None

        rows = [row for row, _ in occupied]
        cols = [col for _, col in occupied]
        return PlateGridBounds(
            min_row=min(rows),
            max_row=max(rows),
            min_col=min(cols),
            max_col=max(cols),
        )

    @property
    def is_empty(self) -> bool:
        return not self.wells_with_images

    @property
    def actual_rows(self) -> list[int]:
        return self.bounds.rows if self.bounds is not None else []

    @property
    def actual_cols(self) -> list[int]:
        return self.bounds.cols if self.bounds is not None else []

    def well_at(self, row: int, col: int) -> str | None:
        return self.coord_to_well.get((row, col))

    def wells_on_axis(self, *, axis_index: int, axis_value: int) -> list[str]:
        return sorted(
            well_id
            for well_id, coord in self.well_to_coord.items()
            if coord[axis_index] == axis_value and well_id in self.wells_with_images
        )


class PlateSubdirectoryButtonRegistry:
    """Own the Qt button registry for the plate-output selector."""

    def __init__(self, button_group: QButtonGroup, layout: QHBoxLayout, style_gen):
        self.button_group = button_group
        self.layout = layout
        self.style_gen = style_gen
        self.buttons: dict[str, QPushButton] = {}

    def clear(self) -> None:
        for button in self.buttons.values():
            self.button_group.removeButton(button)
            button.deleteLater()
        self.buttons.clear()

    def populate(
        self,
        subdirs: List[str],
        on_selected: Callable[[str], None],
    ) -> str:
        self.clear()
        for subdir in subdirs:
            button = QPushButton(subdir)
            button.setCheckable(True)
            button.setStyleSheet(self.style_gen.generate_button_style())
            button.clicked.connect(lambda checked, s=subdir: on_selected(s))

            self.button_group.addButton(button)
            self.layout.insertWidget(self.layout.count() - 1, button)
            self.buttons[subdir] = button

        first_subdir = subdirs[0]
        self.buttons[first_subdir].setChecked(True)
        return first_subdir


class PlateSubdirectoryController:
    """Own plate-output subdirectory selector state and visibility."""

    MODE_HANDLERS: dict[
        PlateSubdirectoryMode,
        Callable[["PlateSubdirectoryController", List[str]], None],
    ] = {
        PlateSubdirectoryMode.NONE: lambda controller, subdirs: (
            controller._set_no_subdirectories()
        ),
        PlateSubdirectoryMode.SINGLE: lambda controller, subdirs: (
            controller._set_single_subdirectory(subdirs[0])
        ),
        PlateSubdirectoryMode.MULTIPLE: lambda controller, subdirs: (
            controller._set_multiple_subdirectories(subdirs)
        ),
    }

    def __init__(self, view: "PlateViewWidget"):
        self.view = view

    def set_subdirectories(self, subdirs: List[str]) -> None:
        self.view.subdirs = subdirs
        self.view.subdir_button_registry.clear()

        mode = PlateSubdirectoryMode.from_count(len(subdirs))
        self.MODE_HANDLERS[mode](self, subdirs)

    def _set_no_subdirectories(self) -> None:
        self.view.subdir_frame.setVisible(False)
        self.view.active_subdir = None

    def _set_single_subdirectory(self, subdir: str) -> None:
        self.view.subdir_frame.setVisible(False)
        self.view.active_subdir = subdir

    def _set_multiple_subdirectories(self, subdirs: List[str]) -> None:
        self.view.subdir_frame.setVisible(True)

        def select_subdirectory(subdir: str) -> None:
            self.view.active_subdir = subdir

        self.view.active_subdir = self.view.subdir_button_registry.populate(
            subdirs,
            select_subdirectory,
        )


class PlateWellButtonRegistry:
    """Own well-id to Qt button lookup for the plate grid."""

    def __init__(self):
        self._buttons: dict[str, QPushButton] = {}

    def clear(self) -> None:
        for button in self._buttons.values():
            button.deleteLater()
        self._buttons.clear()

    def register(self, well_id: str, button: QPushButton) -> None:
        self._buttons[well_id] = button

    def contains(self, well_id: str) -> bool:
        return well_id in self._buttons

    def button(self, well_id: str) -> QPushButton:
        return self._buttons[well_id]

    def items(self):
        return self._buttons.items()


class PlateSelectionController:
    """Own well selection mutation, status projection, and filter sync."""

    def __init__(self, view: "PlateViewWidget"):
        self.view = view

    def clear_selection(
        self,
        emit_signal: bool = True,
        sync_to_filter: bool = True,
    ) -> None:
        logger.info(
            "[CLEAR] clear_selection called, had %s wells, emit_signal=%s, sync_to_filter=%s",
            len(self.view.selected_wells),
            emit_signal,
            sync_to_filter,
        )
        for well_id in list(self.view.selected_wells):
            if self.view.well_button_registry.contains(well_id):
                btn = self.view.well_button_registry.button(well_id)
                btn.setChecked(False)
                btn.setStyleSheet(
                    self.view._get_well_button_style(WellButtonState.HAS_IMAGES)
                )

        self.view.selected_wells.clear()
        self.update_status()

        if sync_to_filter:
            self.sync_to_well_filter()

        if emit_signal:
            self.view.wells_selected.emit(set())

    def select_wells(self, well_ids: Set[str], emit_signal: bool = True) -> None:
        self.clear_selection(emit_signal=False, sync_to_filter=False)

        for well_id in well_ids:
            if (
                self.view.well_button_registry.contains(well_id)
                and well_id in self.view.wells_with_images
            ):
                self.toggle_well_selection(well_id, True)

        self.update_status()
        if emit_signal:
            self.sync_to_well_filter()
            self.view.wells_selected.emit(self.view.selected_wells.copy())

    def publish_selection_change(self) -> None:
        self.update_status()
        self.sync_to_well_filter()
        self.view.wells_selected.emit(self.view.selected_wells.copy())

    def toggle_well_selection(self, well_id: str, select: bool) -> None:
        if (
            not self.view.well_button_registry.contains(well_id)
            or well_id not in self.view.wells_with_images
        ):
            return

        btn = self.view.well_button_registry.button(well_id)

        if select and well_id not in self.view.selected_wells:
            self.view.selected_wells.add(well_id)
            btn.setChecked(True)
            btn.setStyleSheet(
                self.view._get_well_button_style(WellButtonState.SELECTED)
            )
        elif not select and well_id in self.view.selected_wells:
            self.view.selected_wells.discard(well_id)
            btn.setChecked(False)
            btn.setStyleSheet(
                self.view._get_well_button_style(WellButtonState.HAS_IMAGES)
            )

    def update_rectangle_selection(self, rect: QRect) -> None:
        wells_in_rect = set()
        for well_id, btn in self.view.well_button_registry.items():
            if well_id not in self.view.wells_with_images:
                continue
            btn_rect = QRect(btn.pos(), btn.size())
            if rect.intersects(btn_rect):
                wells_in_rect.add(well_id)

        for well_id in self.view.wells_with_images:
            if well_id not in wells_in_rect:
                should_be_selected = well_id in self.view.pre_drag_selection
                self.toggle_well_selection(well_id, should_be_selected)
            else:
                self.toggle_well_selection(well_id, True)

        self.publish_selection_change()

    def toggle_row_selection(self, row_index: int) -> None:
        self.toggle_axis_selection(axis_index=0, axis_value=row_index)

    def toggle_column_selection(self, col_index: int) -> None:
        self.toggle_axis_selection(axis_index=1, axis_value=col_index)

    def toggle_axis_selection(self, *, axis_index: int, axis_value: int) -> None:
        wells_in_axis = self.view.grid_model.wells_on_axis(
            axis_index=axis_index,
            axis_value=axis_value,
        )

        if not wells_in_axis:
            return

        all_selected = all(
            well_id in self.view.selected_wells for well_id in wells_in_axis
        )
        select = not all_selected

        for well_id in wells_in_axis:
            self.toggle_well_selection(well_id, select)

        self.publish_selection_change()

    def invert_selection(self) -> None:
        all_wells = self.view.wells_with_images.copy()
        new_selection = all_wells - self.view.selected_wells

        for well_id in self.view.selected_wells.copy():
            self.toggle_well_selection(well_id, False)

        for well_id in new_selection:
            self.toggle_well_selection(well_id, True)

        self.publish_selection_change()

    def sync_to_well_filter(self) -> None:
        if not self.view.well_filter_widget:
            return

        if self.view.selected_wells:
            self.view.well_filter_widget.set_selected_values(
                self.view.selected_wells,
                block_signals=True,
            )
        else:
            self.view.well_filter_widget.select_all(block_signals=True)

    def sync_from_well_filter(self) -> None:
        if not self.view.well_filter_widget:
            return

        selected_in_filter = self.view.well_filter_widget.get_selected_values()
        self.select_wells(selected_in_filter, emit_signal=False)

    def update_status(self) -> None:
        total_wells = len(self.view.wells_with_images)
        selected_count = len(self.view.selected_wells)

        if selected_count > 0:
            self.view.status_label.setText(
                f"{total_wells} wells have images | {selected_count} selected"
            )
        else:
            self.view.status_label.setText(f"{total_wells} wells have images")


@dataclass(frozen=True, slots=True)
class PlateSelectionEventRoute:
    """One event route into plate selection handling."""

    target: PlateSelectionEventTarget
    event_type: QEvent.Type
    handler: Callable[["PlateSelectionEventController", object, object], bool | None]


class SquareButton(QPushButton):
    """QPushButton that fills its grid cell."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Use Expanding policy so buttons fill available space uniformly
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)


class PlateSelectionInteractionLifecycle:
    """Own begin/update/finish transitions for plate selection gestures."""

    def __init__(self, view: "PlateViewWidget"):
        self.view = view

    def is_left_rectangle_drag(self, event) -> bool:
        return (
            self.view.is_rect_selecting
            and event.buttons() & Qt.MouseButton.LeftButton
        )

    def begin_rectangle_selection(self, start_pos) -> None:
        self.view.is_rect_selecting = True
        self.view.rect_start_pos = start_pos
        self.view.rect_current_pos = start_pos
        self.view.pre_drag_selection = self.view.selected_wells.copy()
        self.view.grid_widget.grabMouse()
        self.show_rectangle(QRect(start_pos, start_pos).normalized())

    def begin_well_drag(self, well_id: str) -> None:
        self.view.is_dragging = True
        self.view.drag_start_well = well_id
        self.view.drag_current_well = well_id
        self.view.drag_affected_wells = set()
        self.view.drag_moved = False
        self.view.drag_selection_mode = (
            "deselect" if well_id in self.view.selected_wells else "select"
        )
        self.view.selection_controller.toggle_well_selection(
            well_id, self.view.drag_selection_mode == "select"
        )
        self.view.drag_affected_wells.add(well_id)
        self.view.selection_controller.publish_selection_change()

    def update_rectangle(self, current_pos) -> None:
        self.view.rect_current_pos = current_pos
        rect = QRect(self.view.rect_start_pos, current_pos).normalized()
        self.show_rectangle(rect)
        self.view.selection_controller.update_rectangle_selection(rect)

    def show_rectangle(self, rect: QRect) -> None:
        self.view.selection_rect_widget.setGeometry(rect)
        self.view.selection_rect_widget.raise_()
        self.view.selection_rect_widget.show()

    def finish_interaction(self) -> None:
        self.view.grid_widget.releaseMouse()
        if self.view.is_dragging:
            self.finish_well_drag()
        if self.view.is_rect_selecting:
            self.finish_rectangle_selection()

    def finish_well_drag(self) -> None:
        self.view.is_dragging = False
        self.view.drag_start_well = None
        self.view.drag_current_well = None
        self.view.drag_selection_mode = None
        self.view.drag_affected_wells.clear()
        self.view.drag_moved = False

    def finish_rectangle_selection(self) -> None:
        self.view.grid_widget.releaseMouse()
        self.view.is_rect_selecting = False
        self.view.rect_start_pos = None
        self.view.rect_current_pos = None
        self.view.selection_rect_widget.hide()


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


class PlateSelectionEventController:
    """Own mouse-event semantics for plate drag and rectangle selection."""

    EVENT_ROUTES: tuple[PlateSelectionEventRoute, ...] = (
        PlateSelectionEventRoute(
            PlateSelectionEventTarget.BUTTON,
            QEvent.Type.MouseButtonPress,
            lambda controller, obj, event: controller._handle_button_press(obj, event),
        ),
        PlateSelectionEventRoute(
            PlateSelectionEventTarget.BUTTON,
            QEvent.Type.MouseMove,
            lambda controller, obj, event: controller._handle_button_move(obj, event),
        ),
        PlateSelectionEventRoute(
            PlateSelectionEventTarget.BUTTON,
            QEvent.Type.MouseButtonRelease,
            lambda controller, obj, event: controller._handle_button_release(event),
        ),
        PlateSelectionEventRoute(
            PlateSelectionEventTarget.GRID,
            QEvent.Type.MouseButtonPress,
            lambda controller, obj, event: controller._handle_grid_press(event),
        ),
        PlateSelectionEventRoute(
            PlateSelectionEventTarget.GRID,
            QEvent.Type.MouseMove,
            lambda controller, obj, event: controller._handle_grid_move(event),
        ),
        PlateSelectionEventRoute(
            PlateSelectionEventTarget.GRID,
            QEvent.Type.MouseButtonRelease,
            lambda controller, obj, event: controller._handle_grid_release(event),
        ),
    )
    EVENT_ROUTE_BY_KEY = {
        (route.target, route.event_type): route for route in EVENT_ROUTES
    }

    def __init__(self, view: "PlateViewWidget"):
        self.view = view
        self.lifecycle = PlateSelectionInteractionLifecycle(view)

    def handle(self, obj, event) -> bool | None:
        """Return handled state, or None when the widget should use Qt default."""
        if isinstance(obj, QPushButton):
            target = PlateSelectionEventTarget.BUTTON
        elif obj == self.view.grid_widget:
            target = PlateSelectionEventTarget.GRID
        else:
            return None

        route = self.EVENT_ROUTE_BY_KEY.get((target, event.type()))
        if route is None:
            return None
        return route.handler(self, obj, event)

    def _handle_button_press(self, button: QPushButton, event) -> bool | None:
        if event.button() != Qt.MouseButton.LeftButton:
            return None

        well_id = button.property("well_id")
        self.lifecycle.begin_rectangle_selection(
            button.mapTo(self.view.grid_widget, event.pos())
        )

        if well_id and well_id in self.view.wells_with_images:
            self.lifecycle.begin_well_drag(well_id)

        event.accept()
        return True

    def _handle_button_move(self, button: QPushButton, event) -> bool | None:
        if not self.lifecycle.is_left_rectangle_drag(event):
            return None

        self.lifecycle.update_rectangle(
            button.mapTo(self.view.grid_widget, event.pos())
        )
        if self.view.is_dragging:
            self.view.drag_moved = True

        event.accept()
        return True

    def _handle_button_release(self, event) -> bool | None:
        if event.button() != Qt.MouseButton.LeftButton:
            return None
        if not (self.view.is_rect_selecting or self.view.is_dragging):
            return None

        self.lifecycle.finish_interaction()
        event.accept()
        return True

    def _handle_grid_press(self, event) -> bool | None:
        if event.button() != Qt.MouseButton.LeftButton:
            return None
        if isinstance(self.view.grid_widget.childAt(event.pos()), QPushButton):
            return None

        self.lifecycle.begin_rectangle_selection(event.pos())
        event.accept()
        return True

    def _handle_grid_move(self, event) -> bool | None:
        if not self.lifecycle.is_left_rectangle_drag(event):
            return None

        self.lifecycle.update_rectangle(event.pos())
        event.accept()
        return True

    def _handle_grid_release(self, event) -> bool | None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or not self.view.is_rect_selecting
        ):
            return None

        self.lifecycle.finish_rectangle_selection()
        event.accept()
        return True


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
        self.well_button_registry = PlateWellButtonRegistry()
        self.wells_with_images = set()  # Set of well IDs that have images
        self.selected_wells = set()  # Currently selected wells
        self.grid_model = PlateGridModel.empty()
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
        self.selection_controller = PlateSelectionController(self)
        self.subdirectory_controller = PlateSubdirectoryController(self)

        # UI components
        self.subdir_buttons = {}  # subdir_name -> QPushButton
        self.subdir_button_group = None
        self.subdir_button_registry = None
        self.well_grid_layout = None
        self.status_label = None
        self.selection_event_controller = PlateSelectionEventController(self)

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
        self.subdir_button_registry = PlateSubdirectoryButtonRegistry(
            self.subdir_button_group,
            self.subdir_layout,
            self.style_gen,
        )
        self.subdir_buttons = self.subdir_button_registry.buttons

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
        self.subdirectory_controller.set_subdirectories(subdirs)

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
        self.grid_model = PlateGridModel.from_wells(
            well_ids,
            plate_dimensions=plate_dimensions,
            coord_to_well=coord_to_well,
        )
        self.wells_with_images = set(self.grid_model.wells_with_images)
        self.coord_to_well = self.grid_model.coord_to_well
        self.well_to_coord = self.grid_model.well_to_coord
        self.plate_dimensions = self.grid_model.plate_dimensions
        self.row_offset = self.grid_model.row_offset
        self.col_offset = self.grid_model.col_offset

        if self.grid_model.is_empty:
            self._clear_grid()
            self.status_label.setText("No wells")
            return

        # Rebuild grid
        self._build_grid()

        # Update status
        self.selection_controller.update_status()

    def _clear_grid(self):
        """Clear the well grid."""
        self.well_button_registry.clear()

        # Clear layout
        while self.well_grid_layout.count():
            item = self.well_grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_grid(self):
        """Build the well grid based on current dimensions."""
        self._clear_grid()

        actual_rows = self.grid_model.actual_rows
        actual_cols = self.grid_model.actual_cols

        if not actual_rows or not actual_cols:
            return

        all_rows = actual_rows
        all_cols = actual_cols

        # Calculate minimum size based on label width
        # Find longest column number for width calculation
        max_col_label = str(all_cols[-1])
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
        invert_btn.clicked.connect(
            lambda: self.selection_controller.invert_selection()
        )
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
                lambda checked, c=actual_col: (
                    self.selection_controller.toggle_column_selection(c)
                )
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
                lambda checked, r=actual_row: (
                    self.selection_controller.toggle_row_selection(r)
                )
            )
            self.well_grid_layout.addWidget(header, grid_row, 0)

            # Well buttons - for all columns in bounding rectangle
            for grid_col, actual_col in enumerate(all_cols, start=1):
                well_id = self.grid_model.well_at(actual_row, actual_col)

                btn = SquareButton()  # Use SquareButton to maintain 1:1 aspect ratio
                btn.setMinimumSize(min_well_size, min_well_size)
                btn.setCheckable(True)

                if well_id and well_id in self.wells_with_images:
                    # Well exists and has images
                    btn.setEnabled(True)
                    btn.setStyleSheet(
                        self._get_well_button_style(WellButtonState.HAS_IMAGES)
                    )
                    btn.clicked.connect(
                        lambda checked, wid=well_id: self._on_well_clicked(wid, checked)
                    )
                    btn.setProperty("well_id", well_id)
                    btn.installEventFilter(self)
                    self.well_button_registry.register(well_id, btn)
                else:
                    # Empty position in bounding rectangle
                    btn.setEnabled(False)
                    btn.setStyleSheet(
                        self._get_well_button_style(WellButtonState.EMPTY)
                    )
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

    def _get_well_button_style(self, state: WellButtonState) -> str:
        """Generate style for well button based on state."""
        cs = self.color_scheme
        colors = WELL_BUTTON_STYLE_COLORS[state]
        border = (
            f"2px solid {cs.to_hex(getattr(cs, colors.border))}"
            if colors.border
            else "none"
        )
        hover = (
            f"""
                QPushButton:hover {{
                    background-color: {cs.to_hex(getattr(cs, colors.hover_background))};
                }}
            """
            if colors.hover_background
            else ""
        )

        return f"""
            QPushButton {{
                background-color: {cs.to_hex(getattr(cs, colors.background))};
                color: {cs.to_hex(getattr(cs, colors.text))};
                border: {border};
                border-radius: 3px;
            }}
            {hover}
            """

    def _on_well_clicked(self, well_id: str, checked: bool):
        """Handle well button click (only for non-drag clicks)."""
        # Skip if this was part of a drag operation
        if self.is_dragging:
            return

        if checked:
            self.selection_controller.toggle_well_selection(well_id, True)
        else:
            self.selection_controller.toggle_well_selection(well_id, False)

        self.selection_controller.publish_selection_change()

    def clear_selection(self, emit_signal: bool = True, sync_to_filter: bool = True):
        """
        Clear all selected wells.

        Args:
            emit_signal: Whether to emit wells_selected signal (default True)
            sync_to_filter: Whether to sync to well filter (default True)
        """
        self.selection_controller.clear_selection(
            emit_signal=emit_signal,
            sync_to_filter=sync_to_filter,
        )

    def select_wells(self, well_ids: Set[str], emit_signal: bool = True):
        """
        Programmatically select wells.

        Args:
            well_ids: Set of well IDs to select
            emit_signal: Whether to emit wells_selected signal (default True)
        """
        self.selection_controller.select_wells(well_ids, emit_signal=emit_signal)

    def eventFilter(self, obj, event):
        """Handle mouse events on buttons for drag selection and rectangle selection in empty space."""
        handled = self.selection_event_controller.handle(obj, event)
        if handled is not None:
            return handled

        return super().eventFilter(obj, event)

    def sync_to_well_filter(self):
        """Sync current plate view selection to well filter checkboxes."""
        self.selection_controller.sync_to_well_filter()

    def sync_from_well_filter(self):
        """Sync well filter checkbox selection to plate view."""
        self.selection_controller.sync_from_well_filter()
