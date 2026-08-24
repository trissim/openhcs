"""OpenHCS adapters for pyqt-reactive host contracts.

This module registers OpenHCS-specific providers with pyqt-reactive:
- Codegen provider for Python code generation
- Endpoint function catalog for discoverable functions
- Log discovery provider
- Server scan provider
- Window factory for launching PyQt windows
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pyqt_reactive.core.log_utils import LogFileInfo
from pyqt_reactive.protocols import (
    CodegenProviderABC,
    ComponentSelectionProviderABC,
    FormGenConfig,
    FunctionSelectionProviderABC,
    LogDiscoveryProviderABC,
    ServerScanProviderABC,
    register_codegen_provider,
    register_component_selection_provider,
    register_function_selection_provider,
    register_log_discovery_provider,
    register_preview_formatter,
    register_server_scan_provider,
    set_form_config,
)

import openhcs.serialization.pycodify_formatters  # noqa: F401
from openhcs.pyqt_gui.config import UIConfig
from openhcs.pyqt_gui.services.function_catalog_projection import (
    ZMQFunctionCatalogProjectionService,
)
from openhcs.runtime.zmq_config import OpenHCSZMQConfig

if TYPE_CHECKING:
    from openhcs.core.orchestrator.orchestrator import Orchestrator

DeclarationT = TypeVar("DeclarationT")


@dataclass
class OpenHCSFormGenConfig(FormGenConfig):
    """OpenHCS-specific overrides for pyqt-reactive config."""

    log_dir: str | None = None
    log_prefixes: list[str] = field(default_factory=lambda: ["openhcs_"])
    log_root_logger_name: str | None = "openhcs"
    performance_logger_name: str = "openhcs.performance"
    performance_log_filename: str = "performance.log"
    path_cache_file: str | None = None


class OpenHCSCodegenProvider(CodegenProviderABC):
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
        from openhcs.pyqt_gui.services.code_document_normalization import (
            CodeDocumentNormalizationStrategy,
        )

        return CodeDocumentNormalizationStrategy.for_declaration_type(
            declaration_type
        ).normalize_source(
            source,
            declaration_type=declaration_type,
            clean_mode=clean_mode,
        )


class OpenHCSLogDiscoveryProvider(LogDiscoveryProviderABC):
    """Adapter for OpenHCS log discovery utilities."""

    def __init__(self, config_provider: Callable[[], UIConfig]) -> None:
        self._config_provider = config_provider

    def get_current_log_path(self) -> Path:
        from openhcs.core.log_utils import get_current_log_file_path

        return Path(get_current_log_file_path())

    def discover_logs(
        self,
        base_log_path: str | None = None,
        include_main_log: bool = True,
        log_directory: Path | None = None,
    ) -> list[LogFileInfo]:
        from openhcs.core.log_utils import discover_logs

        if log_directory is None:
            log_directory = self._config_provider().logging.resolved_log_directory()
        logs = discover_logs(
            base_log_path=base_log_path,
            include_main_log=include_main_log,
            log_directory=log_directory,
        )
        return logs


class OpenHCSServerScanProvider(ServerScanProviderABC):
    """Scan OpenHCS ZMQ servers for log paths."""

    def __init__(
        self,
        config_provider: Callable[[], OpenHCSZMQConfig],
    ) -> None:
        self._config_provider = config_provider

    def scan_for_server_logs(self) -> list[LogFileInfo]:
        from pathlib import Path

        from pyqt_reactive.services.zmq_server_scan_service import (
            ZMQServerScanService,
        )
        from zmqruntime.execution.logs import ExecutionWorkerLogObservation

        from openhcs.core.config import get_all_streaming_ports
        from openhcs.core.log_utils import classify_log_file

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
        for pong in scan_service.scan_ports(ports_to_scan).responses:
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


class OpenHCSComponentSelectionProvider(ComponentSelectionProviderABC):
    """Component selection provider backed by OpenHCS orchestrator metadata."""

    def get_groupby_enum(self) -> Any:
        from openhcs.constants.constants import GroupBy

        return GroupBy

    def _get_plate_manager(self):
        from pyqt_reactive.services.service_registry import ServiceRegistry

        from openhcs.pyqt_gui.widgets.plate_manager import PlateManagerWidget

        return ServiceRegistry.get(PlateManagerWidget)

    def _get_current_orchestrator(self) -> Orchestrator | None:
        plate_manager = self._get_plate_manager()
        if plate_manager is None or not plate_manager.selected_plate_path:
            return None

        from objectstate import ObjectStateRegistry

        from openhcs.core.orchestrator.orchestrator import (
            Orchestrator,
            OrchestratorState,
        )

        orchestrator = ObjectStateRegistry.get_object(plate_manager.selected_plate_path)
        if not isinstance(orchestrator, Orchestrator):
            return None
        if orchestrator.state is OrchestratorState.CREATED:
            return None
        return orchestrator

    def has_components_available(self, group_by: Any) -> bool:
        """Check if components are available without fetching them all."""
        orchestrator = self._get_current_orchestrator()
        if orchestrator is None:
            return False
        return bool(orchestrator.get_component_keys(group_by))

    def get_component_keys(self, group_by: Any) -> list[str]:
        orchestrator = self._get_current_orchestrator()
        if orchestrator is None:
            raise RuntimeError(
                "Component selection requires an initialized plate orchestrator."
            )
        return orchestrator.get_component_keys(group_by)

    def get_component_display_name(
        self, group_by: Any, component_key: str
    ) -> str | None:
        orchestrator = self._get_current_orchestrator()
        if orchestrator is None:
            return None
        return orchestrator.metadata_cache.get_component_metadata(
            group_by, component_key
        )

    def select_components(
        self,
        available_components: Iterable[str],
        selected_components: Iterable[str],
        group_by: Any,
        parent: Any | None = None,
        **context: Any,
    ) -> list[str] | None:
        from pyqt_reactive.dialogs.group_by_selector_dialog import GroupBySelectorDialog

        return GroupBySelectorDialog.select_components(
            available_components=list(available_components),
            selected_components=list(selected_components),
            group_by=group_by,
            metadata_lookup=self.get_component_display_name,
            parent=parent,
        )


class OpenHCSFunctionSelectionProvider(FunctionSelectionProviderABC):
    """Function selection provider backed by OpenHCS dialogs."""

    def __init__(
        self,
        function_catalog: ZMQFunctionCatalogProjectionService,
    ) -> None:
        self.function_catalog = function_catalog

    def select_function(
        self, parent: Any | None = None, **context: Any
    ) -> Callable | None:
        from openhcs.pyqt_gui.dialogs.function_selector_dialog import (
            FunctionSelectorDialog,
        )

        return FunctionSelectorDialog.select_function(
            self.function_catalog,
            parent=parent,
        )


def register_reactor_providers(
    ui_config_provider: Callable[[], UIConfig],
    *,
    function_catalog_projection: ZMQFunctionCatalogProjectionService,
) -> None:
    """Register all OpenHCS providers with pyqt-reactive."""
    # FormGenConfig with OpenHCS paths
    from openhcs.core.xdg_paths import get_data_file_path

    config = OpenHCSFormGenConfig(
        path_cache_file=str(get_data_file_path("path_cache.json")),
        log_dir=str(ui_config_provider().logging.resolved_log_directory()),
    )

    # Jedi project paths (openhcs package + repo root if available)
    pkg_root = Path(__file__).resolve().parents[2]
    repo_root = pkg_root.parent if pkg_root.name == "openhcs" else pkg_root
    config.jedi_project_paths = [str(pkg_root), str(repo_root)]

    set_form_config(config)

    # Providers
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
    from openhcs.core.config import WellFilterConfig
    from openhcs.pyqt_gui.widgets.config_preview_formatters import (
        format_well_filter_config,
    )

    register_preview_formatter(WellFilterConfig, format_well_filter_config)
