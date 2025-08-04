"""
PyQt GUI configuration dataclasses for OpenHCS.

This module defines configuration objects for the PyQt GUI components,
following the same patterns as the core configuration system.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PlotTheme(Enum):
    """Available plot themes for PyQtGraph components."""
    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"  # Follow system theme


class UpdateStrategy(Enum):
    """Update strategies for real-time monitoring components."""
    FIXED_RATE = "fixed_rate"  # Fixed FPS regardless of data availability
    ADAPTIVE = "adaptive"      # Adapt rate based on data changes
    ON_DEMAND = "on_demand"    # Update only when explicitly requested


@dataclass(frozen=True)
class PerformanceMonitorConfig:
    """Configuration for the system performance monitor widget."""
    
    # Update frequency settings
    update_fps: float = 5.0
    """Update frequency in frames per second (FPS). Default: 5 FPS for good performance."""
    
    history_duration_seconds: float = 60.0
    """Duration of historical data to display in seconds. Default: 60 seconds."""
    
    # Display settings
    plot_theme: PlotTheme = PlotTheme.DARK
    """Theme for plots and charts."""
    
    show_grid: bool = True
    """Whether to show grid lines on plots."""
    
    antialiasing: bool = True
    """Enable antialiasing for smoother plot rendering."""
    
    # Performance settings
    update_strategy: UpdateStrategy = UpdateStrategy.FIXED_RATE
    """Strategy for updating the display."""
    
    max_data_points: Optional[int] = None
    """Maximum number of data points to keep. If None, calculated from update_fps and history_duration."""
    
    # GPU monitoring settings
    enable_gpu_monitoring: bool = True
    """Enable GPU usage monitoring if available."""
    
    gpu_temperature_monitoring: bool = True
    """Enable GPU temperature monitoring if available."""
    
    # CPU monitoring settings
    cpu_frequency_monitoring: bool = True
    """Enable CPU frequency monitoring."""
    
    per_core_cpu_monitoring: bool = False
    """Monitor individual CPU cores (more detailed but higher overhead)."""
    
    # Memory monitoring settings
    detailed_memory_info: bool = True
    """Include detailed memory information (available, cached, etc.)."""
    
    # Chart appearance
    line_width: float = 2.0
    """Width of plot lines in pixels."""
    
    chart_colors: Dict[str, str] = field(default_factory=lambda: {
        'cpu': 'cyan',
        'ram': 'lime', 
        'gpu': 'orange',
        'vram': 'magenta'
    })
    """Color scheme for different metrics."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.update_fps <= 0:
            raise ValueError("update_fps must be positive")
        if self.history_duration_seconds <= 0:
            raise ValueError("history_duration_seconds must be positive")
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
    
    @property
    def update_interval_seconds(self) -> float:
        """Calculate update interval in seconds from FPS."""
        return 1.0 / self.update_fps
    
    @property
    def calculated_max_data_points(self) -> int:
        """Calculate maximum data points based on FPS and history duration."""
        if self.max_data_points is not None:
            return self.max_data_points
        return int(self.history_duration_seconds / self.update_interval_seconds)


@dataclass(frozen=True)
class WindowConfig:
    """Configuration for main window behavior."""
    
    # Window properties
    default_width: int = 1200
    """Default window width in pixels."""
    
    default_height: int = 800
    """Default window height in pixels."""
    
    remember_window_state: bool = True
    """Remember window size and position between sessions."""
    
    floating_by_default: bool = True
    """Whether main window should be floating (non-tiled) by default."""
    
    # Behavior settings
    confirm_close: bool = True
    """Show confirmation dialog when closing the application."""
    
    minimize_to_tray: bool = False
    """Minimize to system tray instead of taskbar."""
    
    auto_save_interval_minutes: Optional[int] = 5
    """Auto-save interval in minutes. None to disable auto-save."""


@dataclass(frozen=True)
class StyleConfig:
    """Configuration for GUI styling and appearance."""
    
    # Theme settings
    theme: PlotTheme = PlotTheme.DARK
    """Overall application theme."""
    
    # Font settings
    default_font_family: str = "Arial"
    """Default font family for the application."""
    
    default_font_size: int = 10
    """Default font size in points."""
    
    monospace_font_family: str = "Consolas"
    """Font family for monospace text (logs, code, etc.)."""
    
    # Color customization
    custom_colors: Dict[str, str] = field(default_factory=dict)
    """Custom color overrides for theme colors."""
    
    # Animation settings
    enable_animations: bool = True
    """Enable UI animations and transitions."""
    
    animation_duration_ms: int = 200
    """Duration of animations in milliseconds."""


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for GUI logging and debugging."""
    
    # Log display settings
    max_log_entries: int = 1000
    """Maximum number of log entries to keep in memory."""
    
    auto_scroll_logs: bool = True
    """Automatically scroll to newest log entries."""
    
    log_level_filter: str = "INFO"
    """Minimum log level to display in GUI."""
    
    # Log file settings
    enable_file_logging: bool = True
    """Enable logging to file."""
    
    log_file_max_size_mb: int = 10
    """Maximum log file size in MB before rotation."""
    
    log_file_backup_count: int = 5
    """Number of backup log files to keep."""


@dataclass(frozen=True)
class PyQtGUIConfig:
    """
    Root configuration object for the PyQt GUI application.
    
    This follows the same pattern as GlobalPipelineConfig, providing
    a centralized, immutable configuration for all GUI components.
    """
    
    # Component configurations
    performance_monitor: PerformanceMonitorConfig = field(default_factory=PerformanceMonitorConfig)
    """Configuration for the system performance monitor."""
    
    window: WindowConfig = field(default_factory=WindowConfig)
    """Configuration for main window behavior."""
    
    style: StyleConfig = field(default_factory=StyleConfig)
    """Configuration for GUI styling and appearance."""
    
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Configuration for GUI logging."""
    
    # Global GUI settings
    enable_debug_mode: bool = False
    """Enable debug mode with additional logging and diagnostics."""
    
    check_for_updates: bool = True
    """Check for application updates on startup."""
    
    # Future extension points
    plugin_settings: Dict[str, Any] = field(default_factory=dict)
    """Settings for GUI plugins and extensions."""


# --- Default Configuration Providers ---

_DEFAULT_PERFORMANCE_MONITOR_CONFIG = PerformanceMonitorConfig(
    update_fps=5.0,  # 5 FPS for good performance balance
    history_duration_seconds=60.0,
    plot_theme=PlotTheme.DARK,
    enable_gpu_monitoring=True
)

_DEFAULT_WINDOW_CONFIG = WindowConfig(
    default_width=1200,
    default_height=800,
    floating_by_default=True,  # User preference for tiling window manager
    remember_window_state=True
)

_DEFAULT_STYLE_CONFIG = StyleConfig(
    theme=PlotTheme.DARK,
    enable_animations=True
)

_DEFAULT_LOGGING_CONFIG = LoggingConfig(
    max_log_entries=1000,
    auto_scroll_logs=True,
    log_level_filter="INFO"
)


def get_default_pyqt_gui_config() -> PyQtGUIConfig:
    """
    Provides a default instance of PyQtGUIConfig.
    
    This function provides sensible defaults for the PyQt GUI application,
    following the same pattern as get_default_global_config().
    
    Returns:
        PyQtGUIConfig: Default configuration instance
    """
    logger.debug("Initializing with default PyQtGUIConfig.")
    return PyQtGUIConfig(
        performance_monitor=_DEFAULT_PERFORMANCE_MONITOR_CONFIG,
        window=_DEFAULT_WINDOW_CONFIG,
        style=_DEFAULT_STYLE_CONFIG,
        logging=_DEFAULT_LOGGING_CONFIG,
        enable_debug_mode=False,
        check_for_updates=True
    )


def create_high_performance_config() -> PyQtGUIConfig:
    """
    Create a high-performance configuration preset.
    
    Returns:
        PyQtGUIConfig: High-performance configuration
    """
    return PyQtGUIConfig(
        performance_monitor=PerformanceMonitorConfig(
            update_fps=30.0,  # High refresh rate
            history_duration_seconds=30.0,  # Shorter history for performance
            antialiasing=False,  # Disable for performance
            per_core_cpu_monitoring=True,  # More detailed monitoring
            detailed_memory_info=True
        ),
        style=StyleConfig(
            enable_animations=False  # Disable animations for performance
        )
    )


def create_low_resource_config() -> PyQtGUIConfig:
    """
    Create a low-resource configuration preset.
    
    Returns:
        PyQtGUIConfig: Low-resource configuration
    """
    return PyQtGUIConfig(
        performance_monitor=PerformanceMonitorConfig(
            update_fps=1.0,  # Very low refresh rate
            history_duration_seconds=120.0,  # Longer history with fewer points
            antialiasing=False,
            enable_gpu_monitoring=False,  # Disable GPU monitoring
            gpu_temperature_monitoring=False,
            cpu_frequency_monitoring=False,
            detailed_memory_info=False
        ),
        logging=LoggingConfig(
            max_log_entries=100,  # Fewer log entries
            enable_file_logging=False
        ),
        style=StyleConfig(
            enable_animations=False
        )
    )
