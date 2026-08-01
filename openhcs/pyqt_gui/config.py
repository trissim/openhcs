"""
PyQt GUI configuration dataclasses for OpenHCS.

This module defines configuration objects for the PyQt GUI components,
following the same patterns as the core configuration system.
Configuration is intended to be immutable and provided as Python objects.
"""

import logging
import secrets
import uuid
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Annotated

from python_introspect import (
    EnvironmentVariable,
    overlay_dataclass_from_environment,
    validate_annotated_dataclass,
)
from zmqruntime.config import (
    NonBlankString,
    PositiveFloat,
    PositiveInteger,
    SocketPort,
    TransportMode,
)
from zmqruntime.transport import get_default_transport_mode
from pyqt_reactive.qt_types import QtKeySequenceText
from pyqt_reactive.services.system_monitor_config import PerformanceMonitorConfig

from openhcs.agent.dto.execution import ExecutionConnectionSpec
from openhcs.agent.services.ui_bridge_service import (
    UiBridgeDescriptorDirectoryAuthority,
)
from openhcs.agent.runtime_platform import AgentRuntimePlatformAuthority
from openhcs.core.config import GlobalPipelineConfig
from openhcs.core.config_cache import ConfigCacheSpec
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

logger = logging.getLogger(__name__)


class GuiLogLevel(str, Enum):
    """Closed logging-level axis exposed by the desktop configuration UI."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SILENT = "SILENT"

    @property
    def logging_level(self) -> int:
        if self is GuiLogLevel.SILENT:
            return logging.CRITICAL + 1
        return getattr(logging, self.value)

    @classmethod
    def choices(cls) -> tuple[str, ...]:
        return tuple(level.value for level in cls)

    @classmethod
    def from_text(cls, value: str) -> "GuiLogLevel":
        try:
            return cls(value.upper())
        except ValueError as error:
            raise ValueError(f"Unsupported GUI log level: {value}") from error


@dataclass(frozen=True)
class LoggingConfig:
    """Process logging declaration consumed by the GUI logging lifecycle."""

    level: GuiLogLevel = GuiLogLevel.INFO
    """Minimum severity emitted by OpenHCS and its root logging handlers."""

    log_directory: Path | None = None
    """Directory for timestamped GUI logs. None uses the OpenHCS data directory."""

    enable_console_logging: bool = True
    """Emit records to the terminal that launched the GUI."""

    enable_file_logging: bool = True
    """Emit records to a rotating timestamped log file."""

    max_file_size_mb: PositiveInteger = 10
    """Maximum size of one log file before it is rotated."""

    backup_count: PositiveInteger = 5
    """Number of rotated log files retained for the current GUI session."""

    def resolved_log_directory(self) -> Path:
        """Resolve the declared log directory without creating it."""

        if self.log_directory is not None:
            return self.log_directory.expanduser().resolve(strict=False)
        from openhcs.core.xdg_paths import get_openhcs_data_dir

        return get_openhcs_data_dir() / "logs"

    def __post_init__(self) -> None:
        validate_annotated_dataclass(self)


# ============================================================================
# Declarative Keyboard Shortcuts System
# ============================================================================


@dataclass(frozen=True)
class ShortcutConfig:
    """Application-wide key sequences.

    Concrete actions and their descriptions are owned by the main-window
    ``QAction`` declarations. This object configures keys only.
    """

    time_travel_back: QtKeySequenceText = "Ctrl+Z"
    """Step back in ObjectState history."""

    time_travel_forward: QtKeySequenceText = "Ctrl+Y"
    """Step forward in ObjectState history."""

    time_travel_to_head: QtKeySequenceText = "Ctrl+Shift+Y"
    """Return to the current ObjectState history head."""

    show_plate_manager: QtKeySequenceText = "Ctrl+P"
    """Show the Plate Manager."""

    show_pipeline_editor: QtKeySequenceText = "Ctrl+E"
    """Show the Pipeline Editor."""

    show_image_browser: QtKeySequenceText = "Ctrl+I"
    """Show the Image Browser."""

    show_log_viewer: QtKeySequenceText = "Ctrl+L"
    """Show the Log Viewer."""

    show_zmq_server_manager: QtKeySequenceText = "Ctrl+M"
    """Show the ZMQ Server Manager."""

    show_configuration: QtKeySequenceText = "Ctrl+G"
    """Show the global configuration window."""

    show_synthetic_plate_generator: QtKeySequenceText = "Ctrl+Shift+G"
    """Show the synthetic plate generator."""

    show_help: QtKeySequenceText = "F1"
    """Show the OpenHCS knowledge base."""

    quit_app: QtKeySequenceText = "Ctrl+Q"
    """Quit OpenHCS."""

    def __post_init__(self) -> None:
        validate_annotated_dataclass(self)


@dataclass(frozen=True)
class ProgressUIConfig:
    """Configuration for progress UI update coalescing."""

    update_fps: PositiveFloat = 30.0
    """Maximum progress UI update rate in frames per second.

    Background threads set a dirty flag on each progress message;
    a QTimer fires at this rate and performs the actual UI update.
    Higher values give smoother progress display but use more CPU.
    """

    @property
    def update_interval_ms(self) -> int:
        """Timer interval in milliseconds derived from update_fps."""
        return max(1, int(1000.0 / self.update_fps))

    def __post_init__(self) -> None:
        validate_annotated_dataclass(self)


@dataclass(frozen=True, slots=True)
class AgentUiBridgeConfig(ExecutionConnectionSpec):
    """Configuration for the local agent/MCP bridge into the running PyQt UI."""

    host: Annotated[
        NonBlankString,
        EnvironmentVariable("OPENHCS_UI_BRIDGE_HOST"),
    ] = "127.0.0.1"
    """Network interface or host used by local MCP clients. Keep 127.0.0.1 for local-only access."""
    port: Annotated[
        SocketPort | None,
        EnvironmentVariable("OPENHCS_UI_BRIDGE_PORT"),
    ] = 7888
    """UI bridge data port. Set an explicit free port when the bridge is enabled."""
    transport_mode: Annotated[
        TransportMode | None,
        EnvironmentVariable("OPENHCS_UI_BRIDGE_TRANSPORT_MODE"),
    ] = field(default_factory=get_default_transport_mode)
    """ZMQ transport mode for bridge traffic."""
    persistent: bool = True
    """Keep the bridge connection available across multiple client requests."""
    enabled: Annotated[
        bool,
        EnvironmentVariable("OPENHCS_ENABLE_UI_BRIDGE"),
    ] = True
    """Start the UI bridge with the PyQt application."""
    descriptor_directory_path: Annotated[
        Path | None,
        EnvironmentVariable(
            "OPENHCS_UI_BRIDGE_DESCRIPTOR_DIR",
            clear_on_empty=True,
        ),
    ] = None
    """Directory for generated bridge descriptor files. None uses the runtime default directory."""
    descriptor_file_path: Annotated[
        Path | None,
        EnvironmentVariable(
            "OPENHCS_UI_BRIDGE_DESCRIPTOR",
            clear_on_empty=True,
        ),
    ] = None
    """Exact descriptor file path. When set, this overrides descriptor_directory_path."""
    bridge_instance_id: NonBlankString | None = field(
        default=None,
        metadata={"ui_hidden": True},
    )
    """Stable bridge instance identifier. None generates a fresh ui-prefixed identifier at startup."""
    auth_token: NonBlankString | None = field(
        default=None,
        metadata={"ui_hidden": True},
    )
    """Authentication token required by bridge clients. None generates a random token at startup."""
    poll_timeout_ms: PositiveInteger = 100
    """Bridge server socket polling interval in milliseconds. Must be positive."""
    shutdown_timeout_seconds: PositiveFloat = 2.0
    """Maximum time to wait for the bridge server thread to stop."""

    @classmethod
    def from_environment(
        cls,
        base: "AgentUiBridgeConfig | None" = None,
    ) -> "AgentUiBridgeConfig":
        if base is None:
            base = cls()
        return overlay_dataclass_from_environment(base)

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

    check_for_updates_on_startup: bool = True
    """Check the official release service after the desktop UI becomes ready.

    The check is asynchronous and only offers an update when a newer stable
    release exists. OpenHCS never downloads or installs an update without the
    user's confirmation.
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

    logging: LoggingConfig = field(default_factory=LoggingConfig)
    """Process logging level, destinations, and rotation policy."""

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


def get_default_ui_config() -> UIConfig:
    """
    Provides a default instance of UIConfig.

    This function provides sensible defaults for the PyQt GUI application,
    following the same pattern as GlobalPipelineConfig().

    Returns:
        UIConfig: Default configuration instance
    """
    logger.debug("Initializing with default UIConfig.")
    return UIConfig(agent_bridge=AgentUiBridgeConfig.from_environment())


def ui_config_cache_spec() -> ConfigCacheSpec[UIConfig]:
    """Return the typed persistence identity for the process UI config."""

    from openhcs.core.xdg_paths import get_config_file_path

    return ConfigCacheSpec(
        config_type=UIConfig,
        cache_file=get_config_file_path("ui_config.config"),
    )


def load_cached_ui_config_sync() -> UIConfig:
    """Load persisted UI settings and apply authoritative environment overrides."""

    from openhcs.core.config_cache import load_config_sync

    cached = load_config_sync(ui_config_cache_spec())
    base = cached if cached is not None else get_default_ui_config()
    return replace(
        base,
        agent_bridge=AgentUiBridgeConfig.from_environment(base.agent_bridge),
    )


def save_ui_config_sync(config: UIConfig) -> bool:
    """Persist the exact process UI configuration."""

    from openhcs.core.config_cache import save_config_sync

    return save_config_sync(config, ui_config_cache_spec())
