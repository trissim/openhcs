"""
Time-Travel Debugging Widget for ObjectStateRegistry.

Provides a compact slider UI for scrubbing through the ENTIRE system history.
All ObjectStates across all scopes are captured together for coherent time-travel.
"""

import logging
from typing import TYPE_CHECKING, Any

from objectstate.object_state import ObjectStateRegistry
from PyQt6.QtCore import QPoint, Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolTip,
    QWidget,
)
from pyqt_reactive.theming import ColorScheme
from pyqt_reactive.widgets import ThemedCheckBox

if TYPE_CHECKING:
    from openhcs.pyqt_gui.services.main_window_workflows import (
        MainWindowTimeTravelWorkflow,
    )

logger = logging.getLogger(__name__)


class TimeTravelWidget(QWidget):
    """Compact time-travel debugging widget for ObjectStateRegistry.

    Tracks the ENTIRE system state (all ObjectStates) at each point in time.
    When you scrub, ALL config windows update simultaneously.

    Features:
    - Slider to scrub through global history
    - Back/Forward buttons
    - Label showing current snapshot info
    - Tooltip showing num states captured
    """

    # OpenHCS-specific filter: hide system-only scopes from history
    _HIDDEN_SCOPES = frozenset({"", "__plates__"})

    # Emitted when time-travel position changes
    position_changed = pyqtSignal(int)  # (index)

    def __init__(
        self,
        color_scheme: ColorScheme | None = None,
        show_browse_button: bool = True,
        *,
        time_travel_workflow: "MainWindowTimeTravelWorkflow",
        parent=None,
    ):
        super().__init__(parent)
        self.color_scheme = color_scheme or ColorScheme()
        self._show_browse_button = show_browse_button
        self.time_travel_workflow = time_travel_workflow
        self._setup_ui()

        # Subscribe to history changes (event-based, no polling)
        self._history_subscription = ObjectStateRegistry.add_history_changed_callback(
            self._update_ui
        )
        self.destroyed.connect(self._history_subscription.release)

    def _create_icon_button(self, text: str, tooltip: str) -> QPushButton:
        """Create a compact icon button with consistent styling."""
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Let the button size to its content
        btn.setMinimumWidth(0)
        btn.adjustSize()
        return btn

    def _setup_ui(self):
        """Build the compact UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)

        # Time-travel icon/label
        self.icon_label = QLabel("⏱️")
        self.icon_label.setToolTip("Time-Travel Debugging (All States)")
        layout.addWidget(self.icon_label)

        # Back button
        self.back_btn = self._create_icon_button("◀", "Step back in history")
        self.back_btn.clicked.connect(self._on_back)
        layout.addWidget(self.back_btn)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.setFixedWidth(150)
        self.slider.setToolTip("Scrub through system state history")
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderMoved.connect(self._show_tooltip_at_position)
        layout.addWidget(self.slider)

        # Forward button
        self.forward_btn = self._create_icon_button("▶", "Step forward in history")
        self.forward_btn.clicked.connect(self._on_forward)
        layout.addWidget(self.forward_btn)

        # Head button (return to present)
        self.head_btn = self._create_icon_button(
            "⏭", "Return to present (latest state)"
        )
        self.head_btn.clicked.connect(self._on_head)
        layout.addWidget(self.head_btn)

        # Branch dropdown (auto-hides when only one branch)
        self.branch_combo = QComboBox()
        self.branch_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.branch_combo.setToolTip("Switch branch")
        self.branch_combo.currentTextChanged.connect(self._on_branch_changed)
        layout.addWidget(self.branch_combo)

        # Skip unsaved checkbox
        self.skip_unsaved_cb = ThemedCheckBox("Saves only")
        self.skip_unsaved_cb.setToolTip(
            "Skip unsaved changes, jump only between save points"
        )
        self.skip_unsaved_cb.setChecked(False)
        layout.addWidget(self.skip_unsaved_cb)

        # Browse button - opens full snapshot browser window (optional)
        if self._show_browse_button:
            self.browse_btn = self._create_icon_button("⏳", "Open Snapshot Browser")
            self.browse_btn.clicked.connect(self._on_browse)
            layout.addWidget(self.browse_btn)

        # Status label: right-most widget, text left-aligned
        self.status_label = QLabel("No history")
        self.status_label.setMinimumWidth(180)
        self.status_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        self.status_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        font = QFont()
        font.setPointSize(9)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)

        # Apply the canonical control styles to both the timeline buttons and
        # its branch selector. The application stylesheet deliberately avoids
        # form geometry, so local composite controls opt into their owned input
        # style explicitly.
        self.setStyleSheet(
            self.color_scheme.styles.generate_button_style()
            + self.color_scheme.styles.generate_combobox_style()
        )

        self._update_ui()

    def _update_ui(self):
        """Update slider and labels from registry history.

        History: [oldest, ..., newest] - index 0 = oldest, len-1 = head.
        Slider: 0 = oldest, max = head. Direct mapping, no inversion needed.
        """
        history = ObjectStateRegistry.get_history_info(
            filter_fn=lambda scope_id: scope_id not in self._HIDDEN_SCOPES
        )

        if not history:
            self.slider.setMaximum(0)
            self.slider.setEnabled(False)
            self.back_btn.setEnabled(False)
            self.forward_btn.setEnabled(False)
            self.head_btn.setEnabled(False)
            self.status_label.setText("No history yet")
            return

        self.slider.setEnabled(True)
        self.slider.setMaximum(len(history) - 1)

        # Find current position within the filtered list.
        # NOTE: history entries include a global "index" which is not guaranteed to be
        # contiguous after filtering; using it as a list index will crash.
        current_pos = next(
            (i for i, h in enumerate(history) if h["is_current"]), len(history) - 1
        )
        self.slider.blockSignals(True)
        self.slider.setValue(current_pos)
        self.slider.blockSignals(False)

        # Update status
        current = history[current_pos]
        is_traveling = ObjectStateRegistry.is_time_traveling()

        status = f"{current['timestamp']} | {current['label'][:30]}"
        if is_traveling:
            status += f" ({current_pos + 1}/{len(history)})"
        else:
            status += " (HEAD)"

        self.status_label.setText(status)

        # Enable/disable buttons (back = toward 0, forward = toward len-1)
        self.back_btn.setEnabled(current_pos > 0)
        self.forward_btn.setEnabled(current_pos < len(history) - 1)
        self.head_btn.setEnabled(is_traveling)

        # Visual indicator when time-traveling
        if is_traveling:
            self.icon_label.setText("⏪")
            self.icon_label.setStyleSheet("color: orange;")
        else:
            self.icon_label.setText("⏱️")
            self.icon_label.setStyleSheet("")

        # Update branch dropdown
        branches = ObjectStateRegistry.list_branches()
        self.branch_combo.setVisible(len(branches) >= 1)
        if branches:
            self.branch_combo.blockSignals(True)
            self.branch_combo.clear()
            for b in branches:
                self.branch_combo.addItem(b["name"])
                if b["is_current"]:
                    self.branch_combo.setCurrentText(b["name"])
            self.branch_combo.blockSignals(False)

    def _on_branch_changed(self, name: str):
        """Handle branch dropdown selection change."""
        if (
            name
            and name != ObjectStateRegistry.get_current_branch()
            and not self.time_travel_workflow.switch_branch(name)
        ):
            self._update_ui()

    def _on_slider_changed(self, value: int):
        """Handle slider value change. Direct mapping: slider value = history index."""
        history = ObjectStateRegistry.get_history_info(
            filter_fn=lambda scope_id: scope_id not in self._HIDDEN_SCOPES
        )
        if value < 0 or value >= len(history):
            return

        if not self.time_travel_workflow.to_index(history[value]["index"]):
            self._update_ui()
            return
        self._update_ui()
        self.position_changed.emit(value)

    def _is_save_snapshot(self, label: str) -> bool:
        """Check if a snapshot label represents a save operation."""
        return label.startswith(("save", "init"))

    def _find_next_save_index(
        self, history: list[dict[str, Any]], current_pos: int, direction: int
    ) -> int:
        """Find next save snapshot in given direction.

        Args:
            history: Filtered history list.
            current_pos: Current position in the filtered list (0 = oldest, len-1 = head)
            direction: -1 for back (toward older), +1 for forward (toward newer)

        Returns:
            Index of next save snapshot, or current if none found
        """
        if not history:
            return current_pos

        search_pos = current_pos + direction
        while 0 <= search_pos < len(history):
            if self._is_save_snapshot(history[search_pos]["label"]):
                return search_pos
            search_pos += direction

        # No save found - stay at current or go to boundary
        if direction > 0:
            return len(history) - 1  # Go to head
        return current_pos

    def _on_back(self):
        """Step back in history (toward older = lower index)."""
        if self.skip_unsaved_cb.isChecked():
            history = ObjectStateRegistry.get_history_info(
                filter_fn=lambda scope_id: scope_id not in self._HIDDEN_SCOPES
            )
            if history:
                current_pos = next(
                    (i for i, h in enumerate(history) if h["is_current"]),
                    len(history) - 1,
                )
                target_pos = self._find_next_save_index(history, current_pos, -1)
                if target_pos != current_pos:
                    self.time_travel_workflow.to_index(history[target_pos]["index"])
        else:
            self.time_travel_workflow.back()
        self._update_ui()

    def _on_forward(self):
        """Step forward in history (toward newer = higher index)."""
        if self.skip_unsaved_cb.isChecked():
            history = ObjectStateRegistry.get_history_info(
                filter_fn=lambda scope_id: scope_id not in self._HIDDEN_SCOPES
            )
            if history:
                current_pos = next(
                    (i for i, h in enumerate(history) if h["is_current"]),
                    len(history) - 1,
                )
                target_pos = self._find_next_save_index(history, current_pos, +1)
                if target_pos != current_pos:
                    self.time_travel_workflow.to_index(history[target_pos]["index"])
        else:
            self.time_travel_workflow.forward()
        self._update_ui()

    def _on_head(self):
        """Return to present."""
        self.time_travel_workflow.to_head()
        self._update_ui()

    def _on_browse(self):
        """Open the full Snapshot Browser window."""
        from openhcs.pyqt_gui.windows.snapshot_browser_window import (
            SnapshotBrowserWindow,
        )

        color_scheme = None
        parent = self.window()
        if parent and hasattr(parent, "color_scheme"):
            color_scheme = parent.color_scheme

        self._snapshot_browser = SnapshotBrowserWindow(
            color_scheme=color_scheme,
            time_travel_workflow=self.time_travel_workflow,
            parent=self,
        )
        self._snapshot_browser.show()

    def _show_tooltip_at_position(self, value: int):
        """Show tooltip with snapshot details. Slider value = history index."""
        history = ObjectStateRegistry.get_history_info(
            filter_fn=lambda scope_id: scope_id not in self._HIDDEN_SCOPES
        )
        if value < 0 or value >= len(history):
            return

        h = history[value]
        tooltip = f"<b>{h['timestamp']}</b><br>{h['label']}<br>States captured: {h['num_states']}"
        pos = self.slider.mapToGlobal(QPoint(self.slider.width() // 2, -30))
        QToolTip.showText(pos, tooltip)

    def refresh(self):
        """Refresh UI from registry (call after external changes)."""
        self._update_ui()

    def closeEvent(self, event):
        """Unsubscribe from callbacks on close."""
        self._history_subscription.release()
        super().closeEvent(event)
