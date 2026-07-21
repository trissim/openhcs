"""
PyQt GUI configuration dataclasses for OpenHCS.

This module defines configuration objects for the PyQt GUI components,
following the same patterns as the core configuration system.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
import os
import secrets
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Callable
from enum import Enum

from zmqruntime.transport import get_default_transport_mode

from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.services.ui_bridge_service import (
    UiBridgeDescriptorDirectoryAuthority,
)
from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority
from openhcs.core.config import GlobalPipelineConfig
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

logger = logging.getLogger(__name__)

DEFAULT_AGENT_UI_BRIDGE_TRANSPORT = get_default_transport_mode().value


GUIPluginSettingValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple["GUIPluginSettingValue", ...]
    | dict[str, "GUIPluginSettingValue"]
)


# ============================================================================
# Declarative Keyboard Shortcuts System
# ============================================================================


@dataclass(frozen=True)
class Shortcut:
    """Single keyboard shortcut binding."""

    key: str
    """Qt key sequence that triggers the shortcut, such as Ctrl+Z or F1."""

    action: str
    """OpenHCS action route invoked when the configured key sequence is pressed."""

    description: str
    """User-facing description of the command performed by this shortcut."""


ShortcutItem = tuple[str, Shortcut]


@dataclass(frozen=True)
class ShortcutConfig:
    """
    Declarative keyboard shortcuts for PyQt GUI.

    All shortcuts are defined here - no hardcoding in widgets or main.py.
    To change a shortcut, modify this dataclass.
    """

    # Time travel (global - works everywhere)
    time_travel_back: Shortcut = Shortcut(
        "Ctrl+Z", "time_travel_back", "Step back in history"
    )
    time_travel_forward: Shortcut = Shortcut(
        "Ctrl+Y", "time_travel_forward", "Step forward in history"
    )
    time_travel_to_head: Shortcut = Shortcut(
        "Ctrl+Shift+Y", "time_travel_to_head", "Return to present"
    )

    # Window management
    show_plate_manager: Shortcut = Shortcut(
        "Ctrl+P", "show_plate_manager", "Show Plate Manager"
    )
    show_pipeline_editor: Shortcut = Shortcut(
        "Ctrl+E", "show_pipeline_editor", "Show Pipeline Editor"
    )
    show_image_browser: Shortcut = Shortcut(
        "Ctrl+I", "show_image_browser", "Show Image Browser"
    )
    show_log_viewer: Shortcut = Shortcut("Ctrl+L", "show_log_viewer", "Show Log Viewer")
    show_zmq_server_manager: Shortcut = Shortcut(
        "Ctrl+M", "show_zmq_server_manager", "Show ZMQ Server Manager"
    )
    show_configuration: Shortcut = Shortcut(
        "Ctrl+G", "show_configuration", "Show Global Configuration"
    )
    show_synthetic_plate_generator: Shortcut = Shortcut(
        "Ctrl+Shift+G", "show_synthetic_plate_generator", "Generate Synthetic Plate"
    )

    # Help
    show_help: Shortcut = Shortcut("F1", "show_help", "Show Documentation")

    # Application
    quit_app: Shortcut = Shortcut("Ctrl+Q", "close", "Quit Application")

    def shortcut_items(self) -> tuple[ShortcutItem, ...]:
        """Return shortcuts in the declaration order used by menus and help."""
        return (
            ("time_travel_back", self.time_travel_back),
            ("time_travel_forward", self.time_travel_forward),
            ("time_travel_to_head", self.time_travel_to_head),
            ("show_plate_manager", self.show_plate_manager),
            ("show_pipeline_editor", self.show_pipeline_editor),
            ("show_image_browser", self.show_image_browser),
            ("show_log_viewer", self.show_log_viewer),
            ("show_zmq_server_manager", self.show_zmq_server_manager),
            ("show_configuration", self.show_configuration),
            ("show_synthetic_plate_generator", self.show_synthetic_plate_generator),
            ("show_help", self.show_help),
            ("quit_app", self.quit_app),
        )


class PlotTheme(Enum):
    """Available plot themes for PyQtGraph components."""

    DARK = "dark"
    LIGHT = "light"
    AUTO = "auto"  # Follow system theme


class UpdateStrategy(Enum):
    """Update strategies for real-time monitoring components."""

    FIXED_RATE = "fixed_rate"  # Fixed FPS regardless of data availability
    ADAPTIVE = "adaptive"  # Adapt rate based on data changes
    ON_DEMAND = "on_demand"  # Update only when explicitly requested


@dataclass(frozen=True)
class PerformanceMonitorConfig:
    """Configuration for the system performance monitor widget."""

    # Update frequency settings
    update_fps: float = 10.0
    """Update frequency in frames per second (FPS). Default: 10 FPS for responsive plots."""

    render_fps: float = 30.0
    """Legacy render-timer cap; fixed-axis plots refresh when samples arrive."""

    history_duration_seconds: float = 60.0
    """Duration of historical data to display in seconds. Default: 60 seconds."""

    # Display settings
    plot_theme: PlotTheme = PlotTheme.DARK
    """Theme for plots and charts."""

    show_grid: bool = True
    """Whether to show grid lines on plots."""

    antialiasing: bool = True
    """Enable antialiasing for smoother plot rendering."""

    use_opengl: bool = True
    """Use OpenGL-backed pyqtgraph curve rendering when available."""

    # Performance settings
    update_strategy: UpdateStrategy = UpdateStrategy.FIXED_RATE
    """Strategy for updating the display."""

    max_data_points: int | None = None
    """Maximum number of data points to keep. If None, calculated from update_fps and history_duration."""

    # GPU monitoring settings
    enable_gpu_monitoring: bool = True
    """Enable GPU usage monitoring if available."""

    gpu_refresh_seconds: float = 1.0
    """Refresh interval for slow GPU provider polling; per-tick samples reuse cached values."""

    gpu_temperature_monitoring: bool = True
    """Enable GPU temperature monitoring if available."""

    # CPU monitoring settings
    cpu_frequency_monitoring: bool = True
    """Enable CPU frequency monitoring."""

    cpu_frequency_refresh_seconds: float = 5.0
    """Refresh interval for slow CPU frequency polling; per-tick samples reuse cached values."""

    per_core_cpu_monitoring: bool = False
    """Monitor individual CPU cores (more detailed but higher overhead)."""

    # Memory monitoring settings
    detailed_memory_info: bool = True
    """Include detailed memory information (available, cached, etc.)."""

    # Chart appearance
    line_width: float = 2.0
    """Width of plot lines in pixels."""

    chart_colors: Dict[str, str] = field(
        default_factory=lambda: {
            "cpu": "cyan",
            "ram": "lime",
            "gpu": "orange",
            "vram": "magenta",
        }
    )
    """Color scheme for different metrics."""

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.update_fps <= 0:
            raise ValueError("update_fps must be positive")
        if self.render_fps <= 0:
            raise ValueError("render_fps must be positive")
        if self.history_duration_seconds <= 0:
            raise ValueError("history_duration_seconds must be positive")
        if self.line_width <= 0:
            raise ValueError("line_width must be positive")
        if self.gpu_refresh_seconds <= 0:
            raise ValueError("gpu_refresh_seconds must be positive")
        if self.cpu_frequency_refresh_seconds <= 0:
            raise ValueError("cpu_frequency_refresh_seconds must be positive")

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

    auto_save_interval_minutes: int | None = 5
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
class ProgressUIConfig:
    """Configuration for progress UI update coalescing."""

    update_fps: float = 30.0
    """Maximum progress UI update rate in frames per second.

    Background threads set a dirty flag on each progress message;
    a QTimer fires at this rate and performs the actual UI update.
    Higher values give smoother progress display but use more CPU.
    """

    @property
    def update_interval_ms(self) -> int:
        """Timer interval in milliseconds derived from update_fps."""
        return max(1, int(1000.0 / self.update_fps))


@dataclass(frozen=True, slots=True)
class AgentUiBridgeConfig(ExecutionConnectionSpec):
    """Configuration for the local agent/MCP bridge into the running PyQt UI."""

    host: str = "127.0.0.1"
    """Network interface or host used by local MCP clients. Keep 127.0.0.1 for local-only access."""
    port: int | None = 7888
    """UI bridge data port. Set an explicit free port when the bridge is enabled."""
    transport_mode: str | None = DEFAULT_AGENT_UI_BRIDGE_TRANSPORT
    """ZMQ transport mode for bridge traffic, typically tcp or ipc."""
    persistent: bool = True
    """Keep the bridge connection available across multiple client requests."""
    enabled: bool = True
    """Start the UI bridge with the PyQt application."""
    timeout_ms: int = 5000
    """Default client request timeout in milliseconds."""
    descriptor_directory_path: str | Path | None = None
    """Directory for generated bridge descriptor files. None uses the runtime default directory."""
    descriptor_file_path: str | Path | None = None
    """Exact descriptor file path. When set, this overrides descriptor_directory_path."""
    bridge_instance_id: str | None = None
    """Stable bridge instance identifier. None generates a fresh ui-prefixed identifier at startup."""
    auth_token: str | None = None
    """Authentication token required by bridge clients. None generates a random token at startup."""
    poll_timeout_ms: int = 100
    """Bridge server socket polling interval in milliseconds. Must be positive."""
    shutdown_timeout_seconds: float = 2.0
    """Maximum time to wait for the bridge server thread to stop."""
    max_code_document_bytes: int = 2_000_000
    """Reserved maximum code-document payload size in bytes; the current bridge does not yet enforce this limit."""
    max_request_bytes: int = 4_000_000
    """Reserved maximum request size in bytes; the current bridge transport does not yet enforce this limit."""
    max_response_bytes: int = 4_000_000
    """Reserved maximum response size in bytes; the current bridge transport does not yet enforce this limit."""
    confirmation_timeout_ms: int = 30_000
    """Reserved mutation-confirmation timeout in milliseconds; interactive confirmation is not yet enforced."""
    require_confirmation_for_mutations: bool = True
    """Reserved policy requiring confirmation before mutations; the current bridge does not yet enforce it."""
    allow_unsafe_code_documents: bool = False
    """Reserved opt-in for unsafe code-document operations; the current bridge does not expose an unsafe bypass."""

    @classmethod
    def from_environment(
        cls,
        base: "AgentUiBridgeConfig | None" = None,
    ) -> "AgentUiBridgeConfig":
        base = base or cls()
        return cls(
            host=EnvironmentValueAuthority.text(
                "OPENHCS_UI_BRIDGE_HOST",
                base.host,
            ),
            port=EnvironmentValueAuthority.integer(
                "OPENHCS_UI_BRIDGE_PORT",
                base.port,
            ),
            transport_mode=EnvironmentValueAuthority.text(
                "OPENHCS_UI_BRIDGE_TRANSPORT_MODE",
                base.transport_mode,
            ),
            persistent=base.persistent,
            enabled=EnvironmentValueAuthority.boolean(
                "OPENHCS_ENABLE_UI_BRIDGE",
                base.enabled,
            ),
            timeout_ms=EnvironmentValueAuthority.integer(
                "OPENHCS_UI_BRIDGE_TIMEOUT_MS",
                base.timeout_ms,
            ),
            descriptor_directory_path=EnvironmentValueAuthority.optional_text(
                "OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR",
                base.descriptor_directory_path,
            ),
            descriptor_file_path=EnvironmentValueAuthority.optional_text(
                "OPENHCS_UI_BRIDGE_DESCRIPTOR",
                base.descriptor_file_path,
            ),
            bridge_instance_id=base.bridge_instance_id,
            auth_token=base.auth_token,
            poll_timeout_ms=base.poll_timeout_ms,
            shutdown_timeout_seconds=base.shutdown_timeout_seconds,
            max_code_document_bytes=base.max_code_document_bytes,
            max_request_bytes=base.max_request_bytes,
            max_response_bytes=base.max_response_bytes,
            confirmation_timeout_ms=base.confirmation_timeout_ms,
            require_confirmation_for_mutations=base.require_confirmation_for_mutations,
            allow_unsafe_code_documents=base.allow_unsafe_code_documents,
        )

    def resolve_bridge_instance_id(self) -> str:
        if self.bridge_instance_id is not None:
            return self.bridge_instance_id
        return f"ui-{uuid.uuid4()}"

    def resolve_auth_token(self) -> str:
        if self.auth_token is not None:
            return self.auth_token
        return secrets.token_urlsafe(32)

    def descriptor_path_for(self, bridge_instance_id: str) -> Path:
        if self.descriptor_file_path is not None:
            return AgentRuntimePlatformAuthority.resolved_path(
                self.descriptor_file_path
            )
        return (
            self.descriptor_directory_or_default()
            / f"ui_bridge_{bridge_instance_id}.json"
        ).resolve(strict=False)

    def descriptor_directory_or_default(self) -> Path:
        if self.descriptor_directory_path is not None:
            return AgentRuntimePlatformAuthority.resolved_path(
                self.descriptor_directory_path
            )
        return UiBridgeDescriptorDirectoryAuthority.default_descriptor_dir()


@dataclass(frozen=True)
class UIConfig:
    """
    Root configuration object for the PyQt GUI application.

    This follows the same pattern as GlobalPipelineConfig, providing
    a centralized, immutable configuration for all GUI components.
    """

    # Component configurations
    performance_monitor: PerformanceMonitorConfig = field(
        default_factory=PerformanceMonitorConfig
    )
    """Configuration for the system performance monitor."""

    progress: ProgressUIConfig = field(default_factory=ProgressUIConfig)
    """Configuration for progress UI update coalescing."""

    shortcuts: ShortcutConfig = field(default_factory=ShortcutConfig)
    """Application-wide keyboard shortcut bindings."""

    zmq: OpenHCSZMQConfig = field(default_factory=OpenHCSZMQConfig)
    """Process-level execution transport settings."""

    agent_bridge: AgentUiBridgeConfig = field(default_factory=AgentUiBridgeConfig)
    """Configuration for the local agent/MCP bridge into the running UI."""

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
    plugin_settings: Dict[str, GUIPluginSettingValue] = field(default_factory=dict)
    """Settings for GUI plugins and extensions."""

    @classmethod
    def object_state_scope_id(cls) -> str:
        """Return the ObjectState scope owned by this process-global config type."""

        return f"{cls.__module__}.{cls.__qualname__}"


@dataclass(frozen=True)
class PyQtGuiRuntimeContext:
    """Startup-resolved GUI runtime context shared by app and main window."""

    ui_config: UIConfig
    pipeline_runtime: GlobalPipelineConfig = field(default_factory=GlobalPipelineConfig)

    def with_pipeline_runtime(
        self,
        pipeline_runtime: GlobalPipelineConfig,
    ) -> "PyQtGuiRuntimeContext":
        return replace(self, pipeline_runtime=pipeline_runtime)

    def with_ui_config(self, ui_config: UIConfig) -> "PyQtGuiRuntimeContext":
        return replace(self, ui_config=ui_config)


# --- Default Configuration Providers ---

_DEFAULT_PERFORMANCE_MONITOR_CONFIG = PerformanceMonitorConfig(
    update_fps=10.0,  # Responsive default; fixed-axis plots keep this inexpensive
    render_fps=30.0,
    history_duration_seconds=60.0,
    plot_theme=PlotTheme.DARK,
    antialiasing=True,
    use_opengl=True,
    enable_gpu_monitoring=True,
)

_DEFAULT_WINDOW_CONFIG = WindowConfig(
    default_width=1200,
    default_height=800,
    floating_by_default=True,  # User preference for tiling window manager
    remember_window_state=True,
)

_DEFAULT_STYLE_CONFIG = StyleConfig(theme=PlotTheme.DARK, enable_animations=True)

_DEFAULT_LOGGING_CONFIG = LoggingConfig(
    max_log_entries=1000, auto_scroll_logs=True, log_level_filter="INFO"
)


def get_default_ui_config() -> UIConfig:
    """
    Provides a default instance of UIConfig.

    This function provides sensible defaults for the PyQt GUI application,
    following the same pattern as GlobalPipelineConfig().

    Returns:
        UIConfig: Default configuration instance
    """
    logger.debug("Initializing with default UIConfig.")
    return UIConfig(
        performance_monitor=_DEFAULT_PERFORMANCE_MONITOR_CONFIG,
        agent_bridge=AgentUiBridgeConfig.from_environment(),
        window=_DEFAULT_WINDOW_CONFIG,
        style=_DEFAULT_STYLE_CONFIG,
        logging=_DEFAULT_LOGGING_CONFIG,
        enable_debug_mode=False,
        check_for_updates=True,
    )


class EnvironmentValueAuthority:
    """Typed access rules for PyQt GUI environment overrides."""

    @staticmethod
    def text(name: str, default: str) -> str:
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        return value

    @staticmethod
    def optional_text(name: str, default: str | None) -> str | None:
        value = os.environ.get(name)
        if value is None:
            return default
        if value == "":
            return None
        return value

    @staticmethod
    def integer(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer.") from exc

    @staticmethod
    def boolean(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None or value == "":
            return default
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{name} must be a boolean value.")


def create_high_performance_config() -> UIConfig:
    """
    Create a high-performance configuration preset.

    Returns:
        UIConfig: High-performance configuration
    """
    return UIConfig(
        performance_monitor=PerformanceMonitorConfig(
            update_fps=30.0,  # High refresh rate
            render_fps=30.0,
            history_duration_seconds=30.0,  # Shorter history for performance
            antialiasing=False,  # Disable for performance
            per_core_cpu_monitoring=True,  # More detailed monitoring
            detailed_memory_info=True,
        ),
        style=StyleConfig(
            enable_animations=False  # Disable animations for performance
        ),
    )


def create_low_resource_config() -> UIConfig:
    """
    Create a low-resource configuration preset.

    Returns:
        UIConfig: Low-resource configuration
    """
    return UIConfig(
        performance_monitor=PerformanceMonitorConfig(
            update_fps=1.0,  # Very low refresh rate
            render_fps=1.0,
            history_duration_seconds=120.0,  # Longer history with fewer points
            antialiasing=False,
            enable_gpu_monitoring=False,  # Disable GPU monitoring
            gpu_temperature_monitoring=False,
            cpu_frequency_monitoring=False,
            detailed_memory_info=False,
        ),
        progress=ProgressUIConfig(
            update_fps=10.0,  # Lower progress update rate to save CPU
        ),
        logging=LoggingConfig(
            max_log_entries=100,  # Fewer log entries
            enable_file_logging=False,
        ),
        style=StyleConfig(enable_animations=False),
    )
