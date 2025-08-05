"""
System Monitor Widget for PyQt6

Real-time system monitoring with CPU, RAM, GPU, and VRAM usage graphs.
Migrated from Textual TUI with full feature parity.
"""

import logging
import time
from typing import Optional
from datetime import datetime
from collections import deque

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont

# Import PyQtGraph for high-performance plotting
try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Import the SystemMonitor service
from openhcs.textual_tui.services.system_monitor import SystemMonitor
from openhcs.pyqt_gui.services.persistent_system_monitor import PersistentSystemMonitor
from openhcs.pyqt_gui.shared.style_generator import StyleSheetGenerator
from openhcs.pyqt_gui.shared.color_scheme import PyQt6ColorScheme
from openhcs.pyqt_gui.config import PyQtGUIConfig, PerformanceMonitorConfig, get_default_pyqt_gui_config

logger = logging.getLogger(__name__)


class SystemMonitorWidget(QWidget):
    """
    PyQt6 System Monitor Widget.
    
    Displays real-time system metrics with graphs for CPU, RAM, GPU, and VRAM usage.
    Provides the same functionality as the Textual SystemMonitorTextual widget.
    """
    
    # Signals
    metrics_updated = pyqtSignal(dict)  # Emitted when metrics are updated
    
    def __init__(self,
                 color_scheme: Optional[PyQt6ColorScheme] = None,
                 config: Optional[PyQtGUIConfig] = None,
                 parent=None):
        """
        Initialize the system monitor widget.

        Args:
            color_scheme: Color scheme for styling (optional, uses default if None)
            config: GUI configuration (optional, uses default if None)
            parent: Parent widget
        """
        super().__init__(parent)

        # Initialize configuration
        self.config = config or get_default_pyqt_gui_config()
        self.monitor_config = self.config.performance_monitor

        # Initialize color scheme and style generator
        self.color_scheme = color_scheme or PyQt6ColorScheme()
        self.style_generator = StyleSheetGenerator(self.color_scheme)

        # Calculate monitoring parameters from configuration
        update_interval = self.monitor_config.update_interval_seconds
        history_length = self.monitor_config.calculated_max_data_points

        # Core monitoring - use persistent thread for non-blocking metrics collection
        self.monitor = SystemMonitor(history_length=history_length)  # Match the dynamic history length

        self.persistent_monitor = PersistentSystemMonitor(
            update_interval=update_interval,
            history_length=history_length
        )
        # No timer needed - the persistent thread handles timing
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()

        # Delay monitoring start until widget is shown (fixes WSL2 hanging)
        self._monitoring_started = False

        logger.debug("System monitor widget initialized")

    def showEvent(self, event):
        """Handle widget show event - start monitoring when widget becomes visible."""
        super().showEvent(event)
        if not self._monitoring_started:
            # Start monitoring only when widget is actually shown
            # This prevents WSL2 hanging issues during initialization
            self.start_monitoring()
            self._monitoring_started = True
            logger.debug("System monitoring started on widget show")

    def closeEvent(self, event):
        """Handle widget close event - cleanup resources."""
        self.cleanup()
        super().closeEvent(event)

    def __del__(self):
        """Destructor - ensure cleanup happens."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction
    
    def setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Header section
        header_layout = self.create_header_section()
        layout.addLayout(header_layout)
        
        # Monitoring section
        if PYQTGRAPH_AVAILABLE:
            monitoring_widget = self.create_pyqtgraph_section()
        else:
            monitoring_widget = self.create_fallback_section()
        
        layout.addWidget(monitoring_widget)
        
        # Apply centralized styling
        self.setStyleSheet(self.style_generator.generate_system_monitor_style())
    
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
    
    def create_pyqtgraph_section(self) -> QWidget:
        """
        Create PyQtGraph-based monitoring section with consolidated graphs.

        Returns:
            Widget containing consolidated PyQtGraph plots
        """
        widget = QWidget()
        layout = QGridLayout(widget)

        # Configure PyQtGraph based on config settings
        pg.setConfigOption('background', self.color_scheme.to_hex(self.color_scheme.window_bg))
        pg.setConfigOption('foreground', 'white')
        pg.setConfigOption('antialias', self.monitor_config.antialiasing)

        # Create consolidated PyQtGraph plots in 1x2 grid
        self.cpu_gpu_plot = pg.PlotWidget(title="CPU/GPU Usage")
        self.ram_vram_plot = pg.PlotWidget(title="RAM/VRAM Usage")

        # Store plot data items for efficient updates using configured colors and line width
        colors = self.monitor_config.chart_colors
        line_width = self.monitor_config.line_width

        # CPU/GPU plot curves
        self.cpu_curve = self.cpu_gpu_plot.plot(pen=pg.mkPen(colors['cpu'], width=line_width), name='CPU')
        self.gpu_curve = self.cpu_gpu_plot.plot(pen=pg.mkPen(colors['gpu'], width=line_width), name='GPU')

        # RAM/VRAM plot curves
        self.ram_curve = self.ram_vram_plot.plot(pen=pg.mkPen(colors['ram'], width=line_width), name='RAM')
        self.vram_curve = self.ram_vram_plot.plot(pen=pg.mkPen(colors['vram'], width=line_width), name='VRAM')

        # Style CPU/GPU plot
        self.cpu_gpu_plot.setBackground(self.color_scheme.to_hex(self.color_scheme.panel_bg))
        self.cpu_gpu_plot.setYRange(0, 100)
        self.cpu_gpu_plot.setXRange(0, self.monitor_config.history_duration_seconds)  # Show time range in seconds
        self.cpu_gpu_plot.setLabel('left', 'Usage (%)')
        self.cpu_gpu_plot.setLabel('bottom', 'Time (seconds)')
        self.cpu_gpu_plot.showGrid(x=self.monitor_config.show_grid, y=self.monitor_config.show_grid, alpha=0.3)
        self.cpu_gpu_plot.getAxis('left').setTextPen('white')
        self.cpu_gpu_plot.getAxis('bottom').setTextPen('white')
        self.cpu_gpu_plot.addLegend()

        # Style RAM/VRAM plot
        self.ram_vram_plot.setBackground(self.color_scheme.to_hex(self.color_scheme.panel_bg))
        self.ram_vram_plot.setYRange(0, 100)
        self.ram_vram_plot.setXRange(0, self.monitor_config.history_duration_seconds)  # Show time range in seconds
        self.ram_vram_plot.setLabel('left', 'Usage (%)')
        self.ram_vram_plot.setLabel('bottom', 'Time (seconds)')
        self.ram_vram_plot.showGrid(x=self.monitor_config.show_grid, y=self.monitor_config.show_grid, alpha=0.3)
        self.ram_vram_plot.getAxis('left').setTextPen('white')
        self.ram_vram_plot.getAxis('bottom').setTextPen('white')
        self.ram_vram_plot.addLegend()

        # Add plots to grid layout (1x2 instead of 2x2)
        layout.addWidget(self.cpu_gpu_plot, 0, 0)
        layout.addWidget(self.ram_vram_plot, 0, 1)

        return widget
    
    def create_fallback_section(self) -> QWidget:
        """
        Create fallback text-based monitoring section.
        
        Returns:
            Widget containing text-based display
        """
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setStyleSheet(f"""
            QFrame {{
                background-color: {self.color_scheme.to_hex(self.color_scheme.panel_bg)};
                border: 1px solid {self.color_scheme.to_hex(self.color_scheme.border_color)};
                border-radius: 3px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout(widget)
        
        self.fallback_label = QLabel("")
        self.fallback_label.setFont(QFont("Courier", 10))
        self.fallback_label.setStyleSheet(f"color: {self.color_scheme.to_hex(self.color_scheme.text_accent)};")
        layout.addWidget(self.fallback_label)
        
        return widget
    
    def setup_connections(self):
        """Setup signal/slot connections."""
        self.metrics_updated.connect(self.update_display)

        # Connect persistent monitor signals
        self.persistent_monitor.connect_signals(
            metrics_callback=self.on_metrics_updated,
            error_callback=self.on_metrics_error
        )
    
    def start_monitoring(self):
        """Start the persistent monitoring thread."""
        self.persistent_monitor.start_monitoring()
        logger.debug("System monitoring started")

    def stop_monitoring(self):
        """Stop the persistent monitoring thread."""
        self.persistent_monitor.stop_monitoring()
        logger.debug("System monitoring stopped")

    def cleanup(self):
        """Clean up widget resources."""
        try:
            logger.debug("Cleaning up SystemMonitorWidget...")

            # Stop monitoring first
            self.stop_monitoring()

            # Clean up pyqtgraph plots
            if PYQTGRAPH_AVAILABLE and hasattr(self, 'cpu_plot'):
                try:
                    self.cpu_plot.clear()
                    self.ram_plot.clear()
                    self.gpu_plot.clear()
                    self.vram_plot.clear()

                    # Clear plot widgets
                    if hasattr(self, 'cpu_plot_widget'):
                        self.cpu_plot_widget.close()
                    if hasattr(self, 'ram_plot_widget'):
                        self.ram_plot_widget.close()
                    if hasattr(self, 'gpu_plot_widget'):
                        self.gpu_plot_widget.close()
                    if hasattr(self, 'vram_plot_widget'):
                        self.vram_plot_widget.close()

                except Exception as e:
                    logger.warning(f"Error cleaning up pyqtgraph plots: {e}")

            # Clear data
            if hasattr(self, 'monitor'):
                self.monitor.cpu_history.clear()
                self.monitor.ram_history.clear()
                self.monitor.gpu_history.clear()
                self.monitor.vram_history.clear()
                self.monitor.time_stamps.clear()

            logger.debug("SystemMonitorWidget cleanup completed")

        except Exception as e:
            logger.warning(f"Error during SystemMonitorWidget cleanup: {e}")
    
    def on_metrics_updated(self, metrics: dict):
        """Handle metrics update from persistent monitor thread."""
        try:
            # Update the sync monitor's history for compatibility with existing plotting code
            if metrics:
                self.monitor.cpu_history.append(metrics.get('cpu_percent', 0))
                self.monitor.ram_history.append(metrics.get('ram_percent', 0))
                self.monitor.gpu_history.append(metrics.get('gpu_percent', 0))
                self.monitor.vram_history.append(metrics.get('vram_percent', 0))
                self.monitor.time_stamps.append(time.time())

                # Update cached metrics
                self.monitor._current_metrics = metrics.copy()

            # Use QTimer.singleShot to ensure UI update happens on main thread
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, lambda: self.metrics_updated.emit(metrics))

        except Exception as e:
            logger.warning(f"Failed to process metrics update: {e}")

    def on_metrics_error(self, error_message: str):
        """Handle metrics collection error."""
        logger.warning(f"Metrics collection failed: {error_message}")
        # Continue with cached/default metrics to keep UI responsive

    def update_display(self, metrics: dict):
        """
        Update the display with new metrics.

        Args:
            metrics: Dictionary of system metrics
        """
        try:
            # Update system info
            self.update_system_info(metrics)

            # Update plots or fallback display
            if PYQTGRAPH_AVAILABLE:
                self.update_pyqtgraph_plots()
            else:
                self.update_fallback_display(metrics)

        except Exception as e:
            logger.warning(f"Failed to update display: {e}")
    
    def update_pyqtgraph_plots(self):
        """Update consolidated PyQtGraph plots with current data - non-blocking and fast."""
        try:
            # Convert data point indices to time values in seconds
            data_length = len(self.monitor.cpu_history)
            if data_length == 0:
                return

            # Create time axis: each data point represents update_interval_seconds
            update_interval = self.monitor_config.update_interval_seconds
            x_time = [i * update_interval for i in range(data_length)]

            # Get current data
            cpu_data = list(self.monitor.cpu_history)
            ram_data = list(self.monitor.ram_history)
            gpu_data = list(self.monitor.gpu_history)
            vram_data = list(self.monitor.vram_history)

            # Update CPU/GPU consolidated plot
            self.cpu_curve.setData(x_time, cpu_data)

            # Handle GPU data (may not be available)
            if any(gpu_data):
                self.gpu_curve.setData(x_time, gpu_data)
                gpu_status = f'{gpu_data[-1]:.1f}%' if gpu_data else 'N/A'
            else:
                self.gpu_curve.setData([], [])  # Clear data
                gpu_status = 'Not Available'

            # Update CPU/GPU plot title with current values
            cpu_status = f'{cpu_data[-1]:.1f}%' if cpu_data else 'N/A'
            self.cpu_gpu_plot.setTitle(f'CPU/GPU Usage - CPU: {cpu_status}, GPU: {gpu_status}')

            # Update RAM/VRAM consolidated plot
            self.ram_curve.setData(x_time, ram_data)

            # Handle VRAM data (may not be available)
            if any(vram_data):
                self.vram_curve.setData(x_time, vram_data)
                vram_status = f'{vram_data[-1]:.1f}%' if vram_data else 'N/A'
            else:
                self.vram_curve.setData([], [])  # Clear data
                vram_status = 'Not Available'

            # Update RAM/VRAM plot title with current values
            ram_status = f'{ram_data[-1]:.1f}%' if ram_data else 'N/A'
            self.ram_vram_plot.setTitle(f'RAM/VRAM Usage - RAM: {ram_status}, VRAM: {vram_status}')

        except Exception as e:
            logger.warning(f"Failed to update PyQtGraph plots: {e}")
    
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
    
    def set_update_interval(self, interval_ms: int):
        """
        Set the update interval for monitoring.

        Args:
            interval_ms: Update interval in milliseconds
        """
        interval_seconds = interval_ms / 1000.0
        self.persistent_monitor.set_update_interval(interval_seconds)

    def update_config(self, new_config: PyQtGUIConfig):
        """
        Update the widget configuration and apply changes.

        Args:
            new_config: New configuration to apply
        """
        old_config = self.config
        self.config = new_config
        self.monitor_config = new_config.performance_monitor

        # Check if we need to restart monitoring with new parameters
        if (old_config.performance_monitor.update_fps != new_config.performance_monitor.update_fps or
            old_config.performance_monitor.history_duration_seconds != new_config.performance_monitor.history_duration_seconds):

            logger.info(f"Updating performance monitor: {new_config.performance_monitor.update_fps} FPS, "
                       f"{new_config.performance_monitor.history_duration_seconds}s history")

            # Stop current monitoring
            self.stop_monitoring()

            # Recalculate parameters
            update_interval = self.monitor_config.update_interval_seconds
            history_length = self.monitor_config.calculated_max_data_points

            # Create new monitors with updated config
            self.monitor = SystemMonitor(history_length=history_length)
            self.persistent_monitor = PersistentSystemMonitor(
                update_interval=update_interval,
                history_length=history_length
            )

            # Reconnect signals
            self.persistent_monitor.connect_signals(
                metrics_callback=self.on_metrics_updated,
                error_callback=self.on_metrics_error
            )

            # Restart monitoring
            self.start_monitoring()

        # Update plot appearance if needed
        if (old_config.performance_monitor.chart_colors != new_config.performance_monitor.chart_colors or
            old_config.performance_monitor.line_width != new_config.performance_monitor.line_width):
            self._update_plot_appearance()

        logger.debug("Performance monitor configuration updated")

    def _update_plot_appearance(self):
        """Update plot appearance based on current configuration."""
        colors = self.monitor_config.chart_colors
        line_width = self.monitor_config.line_width

        # Update curve pens
        self.cpu_curve.setPen(pg.mkPen(colors['cpu'], width=line_width))
        self.ram_curve.setPen(pg.mkPen(colors['ram'], width=line_width))
        self.gpu_curve.setPen(pg.mkPen(colors['gpu'], width=line_width))
        self.vram_curve.setPen(pg.mkPen(colors['vram'], width=line_width))

        # Update plot grid for consolidated plots (don't change X range here - let update_pyqtgraph_plots handle it)
        plots = [self.cpu_gpu_plot, self.ram_vram_plot]
        for plot in plots:
            plot.showGrid(x=self.monitor_config.show_grid, y=self.monitor_config.show_grid, alpha=0.3)
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_monitoring()
        event.accept()
