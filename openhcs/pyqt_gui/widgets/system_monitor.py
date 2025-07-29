"""
System Monitor Widget for PyQt6

Real-time system monitoring with CPU, RAM, GPU, and VRAM usage graphs.
Migrated from Textual TUI with full feature parity.
"""

import logging
from typing import Optional
from datetime import datetime
from collections import deque

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Import the SystemMonitor service
from openhcs.textual_tui.services.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)


class SystemMetricsWorker(QThread):
    """Background worker thread for collecting system metrics."""

    metrics_ready = pyqtSignal(dict)  # Emitted when metrics are collected

    def __init__(self, monitor, update_interval=1.0):
        super().__init__()
        self.monitor = monitor
        self.update_interval = update_interval
        self.running = False

    def run(self):
        """Run the metrics collection loop in background thread."""
        self.running = True
        import time

        while self.running:
            try:
                # Collect metrics in background thread (blocking operations)
                self.monitor.update_metrics()

                # Get cached metrics (non-blocking)
                metrics = self.monitor.get_metrics_dict()

                # Emit signal to update UI on main thread
                self.metrics_ready.emit(metrics)

                # Sleep for update interval
                time.sleep(self.update_interval)

            except Exception as e:
                logger.warning(f"Error in metrics worker: {e}")
                time.sleep(2.0)  # Longer sleep on error

    def stop(self):
        """Stop the worker thread."""
        self.running = False
        self.wait()  # Wait for thread to finish


class SystemMonitorWidget(QWidget):
    """
    PyQt6 System Monitor Widget.
    
    Displays real-time system metrics with graphs for CPU, RAM, GPU, and VRAM usage.
    Provides the same functionality as the Textual SystemMonitorTextual widget.
    """
    
    # Signals
    metrics_updated = pyqtSignal(dict)  # Emitted when metrics are updated
    
    def __init__(self, parent=None):
        """
        Initialize the system monitor widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # Core monitoring
        self.monitor = SystemMonitor()
        self.metrics_worker = SystemMetricsWorker(self.monitor, update_interval=1.0)
        self.update_interval = 1000  # 1 second in milliseconds (for UI updates)

        # Non-blocking update management
        self.pending_metrics = None
        self.plot_update_timer = QTimer()
        self.plot_update_timer.setSingleShot(True)
        self.plot_update_timer.timeout.connect(self._deferred_plot_update)
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        # Start monitoring
        self.start_monitoring()
        
        logger.debug("System monitor widget initialized")

    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_monitoring()
        event.accept()

    def __del__(self):
        """Cleanup when widget is destroyed."""
        if hasattr(self, 'metrics_worker'):
            self.stop_monitoring()
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section
        header_layout = self.create_header_section()
        layout.addLayout(header_layout)
        
        # Monitoring section
        if MATPLOTLIB_AVAILABLE:
            monitoring_widget = self.create_matplotlib_section()
        else:
            monitoring_widget = self.create_fallback_section()
        
        layout.addWidget(monitoring_widget)
        
        # Set background styling
        self.setStyleSheet("""
            SystemMonitorWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                border: 1px solid #555555;
                border-radius: 5px;
            }
            QLabel {
                color: #ffffff;
            }
            #header_label {
                color: #00aaff;
                font-weight: bold;
                font-size: 14px;
            }
            #info_label {
                color: #cccccc;
                font-size: 10px;
            }
        """)
    
    def create_header_section(self) -> QHBoxLayout:
        """
        Create the header section with title and system info.
        
        Returns:
            Header layout
        """
        header_layout = QHBoxLayout()
        
        # ASCII header (left side)
        self.header_label = QLabel(self.get_ascii_header())
        self.header_label.setObjectName("header_label")
        font = QFont("Courier", 10)
        font.setBold(True)
        self.header_label.setFont(font)
        header_layout.addWidget(self.header_label)
        
        # System info (right side)
        self.info_label = QLabel("")
        self.info_label.setObjectName("info_label")
        self.info_label.setFont(QFont("Courier", 8))
        self.info_label.setWordWrap(True)
        header_layout.addWidget(self.info_label)
        
        return header_layout
    
    def create_matplotlib_section(self) -> QWidget:
        """
        Create matplotlib-based monitoring section.
        
        Returns:
            Widget containing matplotlib plots
        """
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #2b2b2b;")
        
        # Create subplots
        self.cpu_ax = self.figure.add_subplot(2, 2, 1)
        self.ram_ax = self.figure.add_subplot(2, 2, 2)
        self.gpu_ax = self.figure.add_subplot(2, 2, 3)
        self.vram_ax = self.figure.add_subplot(2, 2, 4)
        
        # Style subplots
        for ax in [self.cpu_ax, self.ram_ax, self.gpu_ax, self.vram_ax]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(colors='white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['left'].set_color('white')
            ax.set_ylim(0, 100)
        
        self.figure.tight_layout()
        
        layout.addWidget(self.canvas, 0, 0)
        return widget
    
    def create_fallback_section(self) -> QWidget:
        """
        Create fallback text-based monitoring section.
        
        Returns:
            Widget containing text-based display
        """
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 10px;
            }
        """)
        
        layout = QVBoxLayout(widget)
        
        self.fallback_label = QLabel("")
        self.fallback_label.setFont(QFont("Courier", 10))
        self.fallback_label.setStyleSheet("color: #00aaff;")
        layout.addWidget(self.fallback_label)
        
        return widget
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        # Connect worker signals to UI updates
        self.metrics_worker.metrics_ready.connect(self.update_display)
    
    def start_monitoring(self):
        """Start the background metrics worker."""
        self.metrics_worker.start()
        logger.debug("System monitoring started")

    def stop_monitoring(self):
        """Stop the background metrics worker."""
        self.metrics_worker.stop()
        logger.debug("System monitoring stopped")
    
    def update_display(self, metrics: dict):
        """
        Update the display with new metrics (called from worker thread signal).
        Non-blocking approach using deferred updates.

        Args:
            metrics: Dictionary of system metrics from background worker
        """
        try:
            # Store metrics for deferred update
            self.pending_metrics = metrics

            # Update system info immediately (lightweight)
            self.update_system_info(metrics)

            # Schedule deferred plot update (non-blocking)
            if not self.plot_update_timer.isActive():
                self.plot_update_timer.start(10)  # 10ms delay for non-blocking update

        except Exception as e:
            logger.warning(f"Failed to update display: {e}")

    def _deferred_plot_update(self):
        """Deferred plot update to avoid blocking UI thread."""
        if self.pending_metrics is None:
            return

        try:
            # Update plots or fallback display
            if MATPLOTLIB_AVAILABLE:
                self._update_matplotlib_plots_nonblocking()
            else:
                self.update_fallback_display(self.pending_metrics)

        except Exception as e:
            logger.warning(f"Failed to update plots: {e}")
        finally:
            self.pending_metrics = None
    
    def _update_matplotlib_plots_nonblocking(self):
        """Update matplotlib plots with non-blocking approach using incremental updates."""
        try:
            # Get data for plotting
            x_range = list(range(len(self.monitor.cpu_history)))

            # Use QTimer.singleShot to break up the plotting work
            self._update_single_plot(0, x_range)

        except Exception as e:
            logger.warning(f"Failed to update matplotlib plots: {e}")

    def _update_single_plot(self, plot_index: int, x_range: list):
        """Update a single plot to avoid blocking UI thread."""
        try:
            if plot_index == 0:
                # Update CPU plot
                self.cpu_ax.clear()
                self.cpu_ax.plot(x_range, list(self.monitor.cpu_history), 'cyan', linewidth=1)
                self.cpu_ax.set_title(f'CPU: {self.monitor.cpu_history[-1]:.1f}%', color='white', fontsize=10)
                self.cpu_ax.set_ylim(0, 100)
                self.cpu_ax.set_facecolor('#1e1e1e')
                self._style_axis(self.cpu_ax)

                # Schedule next plot update
                QTimer.singleShot(5, lambda: self._update_single_plot(1, x_range))

            elif plot_index == 1:
                # Update RAM plot
                self.ram_ax.clear()
                self.ram_ax.plot(x_range, list(self.monitor.ram_history), 'lime', linewidth=1)
                self.ram_ax.set_title(f'RAM: {self.monitor.ram_history[-1]:.1f}%', color='white', fontsize=10)
                self.ram_ax.set_ylim(0, 100)
                self.ram_ax.set_facecolor('#1e1e1e')
                self._style_axis(self.ram_ax)

                # Schedule next plot update
                QTimer.singleShot(5, lambda: self._update_single_plot(2, x_range))

            elif plot_index == 2:
                # Update GPU plot
                self.gpu_ax.clear()
                if any(self.monitor.gpu_history):
                    self.gpu_ax.plot(x_range, list(self.monitor.gpu_history), 'orange', linewidth=1)
                    self.gpu_ax.set_title(f'GPU: {self.monitor.gpu_history[-1]:.1f}%', color='white', fontsize=10)
                else:
                    self.gpu_ax.set_title('GPU: N/A', color='white', fontsize=10)
                self.gpu_ax.set_ylim(0, 100)
                self.gpu_ax.set_facecolor('#1e1e1e')
                self._style_axis(self.gpu_ax)

                # Schedule next plot update
                QTimer.singleShot(5, lambda: self._update_single_plot(3, x_range))

            elif plot_index == 3:
                # Update VRAM plot
                self.vram_ax.clear()
                if any(self.monitor.vram_history):
                    self.vram_ax.plot(x_range, list(self.monitor.vram_history), 'magenta', linewidth=1)
                    self.vram_ax.set_title(f'VRAM: {self.monitor.vram_history[-1]:.1f}%', color='white', fontsize=10)
                else:
                    self.vram_ax.set_title('VRAM: N/A', color='white', fontsize=10)
                self.vram_ax.set_ylim(0, 100)
                self.vram_ax.set_facecolor('#1e1e1e')
                self._style_axis(self.vram_ax)

                # Final canvas draw - schedule this too to avoid blocking
                QTimer.singleShot(5, self._final_canvas_draw)

        except Exception as e:
            logger.warning(f"Failed to update plot {plot_index}: {e}")

    def _style_axis(self, ax):
        """Apply styling to a single axis (lightweight operation)."""
        ax.tick_params(colors='white', labelsize=6)
        for spine in ax.spines.values():
            spine.set_color('white')

    def _final_canvas_draw(self):
        """Final canvas draw operation, deferred to avoid blocking."""
        try:
            self.canvas.draw_idle()  # Use draw_idle instead of draw for better performance
        except Exception as e:
            logger.warning(f"Failed to draw canvas: {e}")
    
    def update_fallback_display(self, metrics: dict):
        """
        Update fallback text display.
        
        Args:
            metrics: Dictionary of system metrics
        """
        try:
            display_text = f"""
┌─────────────────────────────────────────────────────────────────┐
│ CPU:  {self.create_text_bar(metrics.get('cpu_percent', 0))} {metrics.get('cpu_percent', 0):5.1f}%
│ RAM:  {self.create_text_bar(metrics.get('ram_percent', 0))} {metrics.get('ram_percent', 0):5.1f}% ({metrics.get('ram_used_gb', 0):.1f}/{metrics.get('ram_total_gb', 0):.1f}GB)
│ GPU:  {self.create_text_bar(metrics.get('gpu_percent', 0))} {metrics.get('gpu_percent', 0):5.1f}%
│ VRAM: {self.create_text_bar(metrics.get('vram_percent', 0))} {metrics.get('vram_percent', 0):5.1f}%
└─────────────────────────────────────────────────────────────────┘
"""
            self.fallback_label.setText(display_text)
            
        except Exception as e:
            logger.warning(f"Failed to update fallback display: {e}")
    
    def update_system_info(self, metrics: dict):
        """
        Update system information display.
        
        Args:
            metrics: Dictionary of system metrics
        """
        try:
            info_text = f"""
═══════════════════════════════════════════════════════════════════════
System Information | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═══════════════════════════════════════════════════════════════════════
CPU Cores: {metrics.get('cpu_cores', 'N/A')} | CPU Frequency: {metrics.get('cpu_freq_mhz', 0):.0f} MHz
Total RAM: {metrics.get('ram_total_gb', 0):.1f} GB | Used RAM: {metrics.get('ram_used_gb', 0):.1f} GB"""
            
            # Add GPU info if available
            if 'gpu_name' in metrics:
                info_text += f"\nGPU: {metrics.get('gpu_name', 'N/A')} | Temperature: {metrics.get('gpu_temp', 'N/A')}°C"
                info_text += f"\nVRAM: {metrics.get('vram_used_mb', 0):.0f}/{metrics.get('vram_total_mb', 0):.0f} MB"
            
            info_text += "\n═══════════════════════════════════════════════════════════════════════"
            
            self.info_label.setText(info_text)
            
        except Exception as e:
            logger.warning(f"Failed to update system info: {e}")
    
    def create_text_bar(self, percent: float) -> str:
        """
        Create a text-based progress bar.
        
        Args:
            percent: Percentage value (0-100)
            
        Returns:
            Text progress bar
        """
        bar_length = 20
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        return f"[{bar}]"
    
    def get_ascii_header(self) -> str:
        """
        Get ASCII art header.
        
        Returns:
            ASCII art header string
        """
        return """
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗  ██╗ ██████╗███████╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║  ██║██╔════╝██╔════╝
██║   ██║██████╔╝█████╗  ██╔██╗ ██║███████║██║     ███████╗
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██╔══██║██║     ╚════██║
╚██████╔╝██║     ███████╗██║ ╚████║██║  ██║╚██████╗███████║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝╚══════╝
        """
    
    def set_update_interval(self, interval_seconds: float):
        """
        Set the update interval for monitoring.

        Args:
            interval_seconds: Update interval in seconds
        """
        # Stop current worker
        if self.metrics_worker.isRunning():
            self.stop_monitoring()

        # Create new worker with updated interval
        self.metrics_worker = SystemMetricsWorker(self.monitor, update_interval=interval_seconds)
        self.setup_connections()

        # Restart monitoring
        self.start_monitoring()
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_monitoring()
        event.accept()
