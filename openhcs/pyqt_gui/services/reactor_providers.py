"""OpenHCS adapters for pyqt-reactor provider protocols.

This module registers OpenHCS-specific providers with pyqt-reactor:
- LLM service for pipeline generation
- Codegen provider for Python code generation
- Endpoint function catalog for discoverable functions
- Log discovery provider
- Server scan provider
- Window factory for launching PyQt windows
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, TypeVar

from pyqt_reactive.protocols import (
    FormGenConfig,
    set_form_config,
    register_llm_service,
    register_codegen_provider,
    register_preview_formatter,
    register_log_discovery_provider,
    register_server_scan_provider,
    register_component_selection_provider,
    register_function_selection_provider,
)
import openhcs.serialization.pycodify_formatters  # noqa: F401
from openhcs.pyqt_gui.config import UIConfig
from openhcs.pyqt_gui.services.function_catalog_projection import (
    ZMQFunctionCatalogProjectionService,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

DeclarationT = TypeVar("DeclarationT")


@dataclass
class OpenHCSFormGenConfig(FormGenConfig):
    """OpenHCS-specific overrides for pyqt-reactor config."""

    log_dir: Optional[str] = None
    log_prefixes: List[str] = field(default_factory=lambda: ["openhcs_"])
    log_root_logger_name: Optional[str] = "openhcs"
    performance_logger_name: str = "openhcs.performance"
    performance_log_filename: str = "performance.log"
    path_cache_file: Optional[str] = None


class OpenHCSCodegenProvider:
    """Codegen provider backed by pycodify with OpenHCS formatters."""

    def render_assignment(
        self,
        value: object,
        *,
        assignment_name: str,
        header: str,
        clean_mode: bool,
    ) -> str:
        from pycodify import Assignment
        from openhcs.serialization.source_path_factoring import (
            OpenHCSPythonSourceDocument,
        )

        return OpenHCSPythonSourceDocument(
            Assignment(assignment_name, value),
            header=header,
            clean_mode=clean_mode,
        ).render()

    def normalize_source(
        self,
        source: str,
        *,
        declaration_type: type[DeclarationT],
        clean_mode: bool,
    ) -> str:
        from openhcs.pyqt_gui.services.llm_pipeline_service import (
            CodeDeclarationStrategy,
        )

        return CodeDeclarationStrategy.for_declaration_type(
            declaration_type
        ).normalize_source(
            source,
            declaration_type=declaration_type,
            clean_mode=clean_mode,
        )


class OpenHCSLogDiscoveryProvider:
    """Adapter for OpenHCS log discovery utilities."""

    def __init__(self, config_provider: Callable[[], UIConfig]) -> None:
        self._config_provider = config_provider

    def get_current_log_path(self) -> Path:
        from openhcs.core.log_utils import get_current_log_file_path

        return Path(get_current_log_file_path())

    def discover_logs(
        self,
        base_log_path: Optional[str] = None,
        include_main_log: bool = True,
        log_directory: Optional[Path] = None,
    ):
        from openhcs.core.log_utils import discover_logs

        if log_directory is None:
            log_directory = self._config_provider().logging.resolved_log_directory()
        logs = discover_logs(
            base_log_path=base_log_path,
            include_main_log=include_main_log,
            log_directory=log_directory,
        )
        return logs


class OpenHCSServerScanProvider:
    """Scan OpenHCS ZMQ servers for log paths."""

    def __init__(
        self,
        config_provider: Callable[[], OpenHCSZMQConfig],
    ) -> None:
        self._config_provider = config_provider

    def scan_for_server_logs(self):
        from pathlib import Path

        from openhcs.core.config import get_all_streaming_ports
        from openhcs.core.log_utils import classify_log_file
        from pyqt_reactive.services.zmq_server_scan_service import (
            ZMQServerScanService,
        )
        from zmqruntime.execution.logs import ExecutionWorkerLogObservation

        config = self._config_provider()
        discovered = []
        ports_to_scan = (
            config.default_port,
            *get_all_streaming_ports(num_ports_per_type=config.ports_per_server_type),
        )

        scan_service = ZMQServerScanService(
            config=config,
            host=config.client_host,
            transport_mode=config.transport_mode,
            timeout_ms=config.server_scan_timeout_ms,
        )
        for pong in scan_service.scan_ports(ports_to_scan):
            if pong.log_file_path:
                log_path = Path(pong.log_file_path)
                if log_path.exists():
                    log_info = classify_log_file(log_path, None, False)
                    discovered.append(
                        replace(
                            log_info,
                            process_identity=pong.process_identity,
                        )
                    )
                for worker_log in ExecutionWorkerLogObservation.discover(
                    log_path.parent,
                    running_executions=pong.running_executions,
                    workers=pong.workers,
                ):
                    worker_log_info = classify_log_file(
                        worker_log.path,
                        None,
                        False,
                    )
                    discovered.append(
                        replace(
                            worker_log_info,
                            process_identity=worker_log.process_identity,
                        )
                    )
        return discovered


class OpenHCSComponentSelectionProvider:
    """Component selection provider backed by OpenHCS orchestrator metadata."""

    def __init__(self) -> None:
        self._last_debug_info = "No debug info available"

    def get_groupby_enum(self) -> Any:
        from openhcs.constants.constants import GroupBy

        return GroupBy

    def _get_plate_manager(self):
        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget
        from pyqt_reactive.services.service_registry import ServiceRegistry

        return ServiceRegistry.get(PlateManagerWidget)

    def _get_current_orchestrator(self):
        plate_manager = self._get_plate_manager()
        debug_info = f"[_get_current_orchestrator] plate_manager={plate_manager}\n"
        if not plate_manager:
            debug_info += "[_get_current_orchestrator] No plate manager found!\n"
            # Store debug info for exception
            self._last_debug_info = debug_info
            return None

        current_plate = plate_manager.selected_plate_path
        debug_info += f"[_get_current_orchestrator] current_plate={current_plate!r}\n"

        from objectstate import ObjectStateRegistry
        from openhcs.core.orchestrator.orchestrator import OrchestratorState

        orchestrator = ObjectStateRegistry.get_object(current_plate)
        debug_info += f"[_get_current_orchestrator] orchestrator={orchestrator}\n"

        if orchestrator:
            debug_info += (
                f"[_get_current_orchestrator] orchestrator.state={orchestrator.state}\n"
            )
            debug_info += (
                "[_get_current_orchestrator] OrchestratorState.CREATED="
                f"{OrchestratorState.CREATED}\n"
            )
            debug_info += f"[_get_current_orchestrator] state != CREATED: {orchestrator.state != OrchestratorState.CREATED}\n"

        # Use same check as plate manager - check state != CREATED
        if orchestrator and orchestrator.state != OrchestratorState.CREATED:
            debug_info += "[_get_current_orchestrator] Returning orchestrator\n"
            self._last_debug_info = debug_info
            return orchestrator

        debug_info += (
            f"[_get_current_orchestrator] Returning None! orchestrator={orchestrator}\n"
        )
        self._last_debug_info = debug_info
        return None

    def has_components_available(self, group_by: Any) -> bool:
        """Check if components are available without fetching them all."""
        orchestrator = self._get_current_orchestrator()
        if not orchestrator:
            return False
        # Try to get component keys - if empty, no components available
        try:
            components = orchestrator.get_component_keys(group_by)
            return len(components) > 0
        except Exception:
            return False

    def get_component_keys(self, group_by: Any) -> List[str]:
        orchestrator = self._get_current_orchestrator()
        if not orchestrator:
            # Return debug info as the error so it shows in the UI
            raise RuntimeError(
                f"Cannot get component keys - no initialized orchestrator found!\n\n"
                f"DEBUG INFO:\n{self._last_debug_info}\n"
                f"GROUP_BY: {group_by}"
            )
        return orchestrator.get_component_keys(group_by)

    def get_component_display_name(
        self, group_by: Any, component_key: str
    ) -> Optional[str]:
        orchestrator = self._get_current_orchestrator()
        if not orchestrator:
            return None
        return orchestrator.metadata_cache.get_component_metadata(
            group_by, component_key
        )

    def select_components(
        self,
        available_components: Iterable[str],
        selected_components: Iterable[str],
        group_by: Any,
        parent: Optional[Any] = None,
        **context: Any,
    ) -> Optional[List[str]]:
        from pyqt_reactive.dialogs.group_by_selector_dialog import GroupBySelectorDialog

        return GroupBySelectorDialog.select_components(
            available_components=list(available_components),
            selected_components=list(selected_components),
            group_by=group_by,
            metadata_lookup=self.get_component_display_name,
            parent=parent,
        )


class OpenHCSFunctionSelectionProvider:
    """Function selection provider backed by OpenHCS dialogs."""

    def __init__(
        self,
        function_catalog: ZMQFunctionCatalogProjectionService,
    ) -> None:
        self.function_catalog = function_catalog

    def select_function(
        self, parent: Optional[Any] = None, **context: Any
    ) -> Optional[Callable]:
        from openhcs.pyqt_gui.dialogs.function_selector_dialog import (
            FunctionSelectorDialog,
        )

        return FunctionSelectorDialog.select_function(
            self.function_catalog,
            parent=parent,
        )


def register_openhcs_window_handlers():
    """Register OpenHCS window handlers with the generic factory.

    Note: This is a convenience re-export. The actual registration
    happens in window_handlers module.
    """
    from openhcs.pyqt_gui.services.window_handlers import (
        register_openhcs_window_handlers as _register,
    )

    _register()


def register_reactor_providers(
    ui_config_provider: Callable[[], UIConfig],
    *,
    function_catalog_projection: ZMQFunctionCatalogProjectionService,
) -> None:
    """Register all OpenHCS providers with pyqt-reactor."""
    # FormGenConfig with OpenHCS paths
    config = OpenHCSFormGenConfig()
    try:
        from openhcs.core.xdg_paths import get_data_file_path

        config.path_cache_file = str(get_data_file_path("path_cache.json"))
        config.log_dir = str(ui_config_provider().logging.resolved_log_directory())
    except Exception:
        config.path_cache_file = None
        config.log_dir = None

    # Jedi project paths (openhcs package + repo root if available)
    pkg_root = Path(__file__).resolve().parents[2]
    repo_root = pkg_root.parent if pkg_root.name == "openhcs" else pkg_root
    config.jedi_project_paths = [str(pkg_root), str(repo_root)]

    set_form_config(config)

    # Providers
    from openhcs.pyqt_gui.services.llm_pipeline_service import LLMPipelineService

    register_llm_service(LLMPipelineService(function_catalog_projection))
    register_codegen_provider(OpenHCSCodegenProvider())
    register_log_discovery_provider(OpenHCSLogDiscoveryProvider(ui_config_provider))
    register_server_scan_provider(
        OpenHCSServerScanProvider(lambda: ui_config_provider().zmq)
    )
    register_component_selection_provider(OpenHCSComponentSelectionProvider())
    register_function_selection_provider(
        OpenHCSFunctionSelectionProvider(function_catalog_projection)
    )
    # Window handlers are registered in main.py after widgets are created

    # Preview formatters (OpenHCS-specific)
    try:
        from openhcs.core.config import WellFilterConfig
        from openhcs.pyqt_gui.widgets.config_preview_formatters import (
            format_config_indicator,
        )

        def _format_config(config_obj, field_name: str) -> Optional[str]:
            return format_config_indicator(field_name, config_obj)

        register_preview_formatter(WellFilterConfig, _format_config)
    except Exception:
        pass
