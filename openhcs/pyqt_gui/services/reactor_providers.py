"""OpenHCS adapters for pyqt-reactor provider protocols.

This module registers OpenHCS-specific providers with pyqt-reactor:
- LLM service for pipeline generation
- Codegen provider for Python code generation
- Function registry for discoverable functions
- Log discovery provider
- Server scan provider
- Window factory for launching PyQt windows
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable, Iterable, List, Dict

from PyQt6.QtWidgets import QWidget

from pyqt_reactive.protocols import (
    FormGenConfig,
    set_form_config,
    register_llm_service,
    register_codegen_provider,
    register_function_registry,
    register_preview_formatter,
    register_log_discovery_provider,
    register_server_scan_provider,
    register_window_factory,
    register_component_selection_provider,
    register_function_selection_provider,
)
import openhcs.serialization.pycodify_formatters  # noqa: F401


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

    def generate_complete_orchestrator_code(
        self,
        plate_paths,
        pipeline_data,
        global_config=None,
        per_plate_configs=None,
        pipeline_config=None,
        clean_mode=True,
    ) -> str:
        from openhcs.core.config import PipelineConfig
        from pycodify import Assignment, BlankLine, CodeBlock, generate_python_source

        code_items = [
            Assignment("plate_paths", plate_paths),
            BlankLine(),
            Assignment("global_config", global_config),
            BlankLine(),
        ]

        if per_plate_configs:
            code_items.append(Assignment("per_plate_configs", per_plate_configs))
            code_items.append(BlankLine())
        elif pipeline_config is not None:
            code_items.append(Assignment("pipeline_config", pipeline_config))
            code_items.append(BlankLine())
        else:
            code_items.append(Assignment("pipeline_config", PipelineConfig()))
            code_items.append(BlankLine())

        code_items.append(Assignment("pipeline_data", pipeline_data))

        return generate_python_source(
            CodeBlock.from_items(code_items),
            header="# Edit this orchestrator configuration and save to apply changes",
            clean_mode=clean_mode,
        )

    def generate_complete_pipeline_steps_code(
        self, pipeline_steps, clean_mode=True
    ) -> str:
        from pycodify import Assignment, generate_python_source

        return generate_python_source(
            Assignment("pipeline_steps", pipeline_steps),
            header="# Edit this pipeline and save to apply changes",
            clean_mode=clean_mode,
        )

    def generate_complete_function_pattern_code(
        self, func_obj, clean_mode=False
    ) -> str:
        from pycodify import Assignment, generate_python_source

        return generate_python_source(
            Assignment("pattern", func_obj),
            header="# Edit this function pattern and save to apply changes",
            clean_mode=clean_mode,
        )

    def generate_step_code(self, step_obj, clean_mode=True) -> str:
        from pycodify import Assignment, generate_python_source

        return generate_python_source(
            Assignment("step", step_obj),
            header="# Function Step",
            clean_mode=clean_mode,
        )

    def generate_config_code(
        self, config_obj, clean_mode=True, config_class: Optional[type] = None
    ) -> str:
        from pycodify import Assignment, generate_python_source

        return generate_python_source(
            Assignment("config", config_obj),
            header="# Configuration Code",
            clean_mode=clean_mode,
        )


class OpenHCSFunctionRegistry:
    """Adapter for OpenHCS RegistryService."""

    def _get_metadata(self) -> Dict[str, Any]:
        from openhcs.processing.backends.lib_registry.registry_service import (
            RegistryService,
        )

        return RegistryService.get_all_functions_with_metadata()

    def get_function_by_name(self, name: str) -> Optional[Callable]:
        metadata = self._get_metadata()
        if name in metadata:
            return metadata[name].func
        # Fallback: match by bare function name
        for meta in metadata.values():
            if meta.name == name or meta.original_name == name:
                return meta.func
        return None

    def get_all_functions(self) -> Dict[str, Callable]:
        return {key: meta.func for key, meta in self._get_metadata().items()}

    def get_function_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        metadata = self._get_metadata()
        meta = metadata.get(name)
        if meta is None:
            # Try bare name match
            for m in metadata.values():
                if m.name == name or m.original_name == name:
                    meta = m
                    break
        if meta is None:
            return None
        return {
            "name": meta.name,
            "module": meta.module,
            "doc": meta.doc,
            "tags": list(meta.tags) if meta.tags else [],
            "registry_name": meta.get_registry_name(),
            "memory_type": meta.get_memory_type(),
        }


class PyQtLogInfoProjectionMixin:
    """Project OpenHCS log classification into pyqt-reactive's log row."""

    def pyqt_log_file_info(self, log_info: Any):
        from pyqt_reactive.core.log_utils import LogFileInfo

        return LogFileInfo(
            path=log_info.path,
            log_type=log_info.log_type,
            worker_id=log_info.worker_id,
            display_name=log_info.display_name,
        )


class OpenHCSLogDiscoveryProvider(PyQtLogInfoProjectionMixin):
    """Adapter for OpenHCS log discovery utilities."""

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

        logs = discover_logs(
            base_log_path=base_log_path,
            include_main_log=include_main_log,
            log_directory=log_directory,
        )
        # Convert to pyqt_reactive LogFileInfo for consistency
        converted = [self.pyqt_log_file_info(log_info) for log_info in logs]
        return converted


class OpenHCSServerScanProvider(PyQtLogInfoProjectionMixin):
    """Scan OpenHCS ZMQ servers for log paths."""

    def scan_for_server_logs(self):
        from pathlib import Path
        import zmq
        import pickle

        from openhcs.core.config import get_all_streaming_ports
        from openhcs.constants.constants import CONTROL_PORT_OFFSET
        from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG
        from zmqruntime.transport import (
            get_zmq_transport_url,
            get_default_transport_mode,
        )
        from openhcs.core.log_utils import classify_log_file

        discovered = []
        ports_to_scan = get_all_streaming_ports(num_ports_per_type=10)

        def ping_server(port: int) -> dict:
            control_port = port + CONTROL_PORT_OFFSET
            try:
                context = zmq.Context()
                socket = context.socket(zmq.REQ)
                socket.setsockopt(zmq.LINGER, 0)
                socket.setsockopt(zmq.RCVTIMEO, 1000)

                transport_mode = get_default_transport_mode()
                control_url = get_zmq_transport_url(
                    control_port,
                    host="localhost",
                    mode=transport_mode,
                    config=OPENHCS_ZMQ_CONFIG,
                )
                socket.connect(control_url)

                socket.send(pickle.dumps({"type": "ping"}))
                response = socket.recv()
                pong = pickle.loads(response)

                socket.close()
                context.term()
                return pong
            except Exception:
                return None

        for port in ports_to_scan:
            pong = ping_server(port)
            if pong and pong.get("log_file_path"):
                log_path = Path(pong["log_file_path"])
                if log_path.exists():
                    log_info = classify_log_file(log_path, None, False)
                    discovered.append(self.pyqt_log_file_info(log_info))
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
        import logging

        logger = logging.getLogger(__name__)

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
            debug_info += f"[_get_current_orchestrator] OrchestratorState.CREATED={OrchestratorState.CREATED}\n"
            debug_info += f"[_get_current_orchestrator] state != CREATED: {orchestrator.state != OrchestratorState.CREATED}\n"

        # Use same check as plate manager - check state != CREATED
        if orchestrator and orchestrator.state != OrchestratorState.CREATED:
            debug_info += f"[_get_current_orchestrator] Returning orchestrator\n"
            self._last_debug_info = debug_info
            return orchestrator

        debug_info += (
            f"[_get_current_orchestrator] Returning None! orchestrator={orchestrator}\n"
        )
        self._last_debug_info = debug_info
        return None

        current_plate = plate_manager.selected_plate_path
        logger.info(f"[_get_current_orchestrator] current_plate={current_plate!r}")

        from objectstate import ObjectStateRegistry
        from openhcs.core.orchestrator.orchestrator import OrchestratorState

        orchestrator = ObjectStateRegistry.get_object(current_plate)
        logger.info(f"[_get_current_orchestrator] orchestrator={orchestrator}")

        if orchestrator:
            logger.info(
                f"[_get_current_orchestrator] orchestrator.state={orchestrator.state}"
            )
            logger.info(
                f"[_get_current_orchestrator] OrchestratorState.CREATED={OrchestratorState.CREATED}"
            )
            logger.info(
                f"[_get_current_orchestrator] state != CREATED: {orchestrator.state != OrchestratorState.CREATED}"
            )

        # Use same check as plate manager - check state != CREATED
        if orchestrator and orchestrator.state != OrchestratorState.CREATED:
            logger.info(f"[_get_current_orchestrator] Returning orchestrator")
            return orchestrator

        logger.error(
            f"[_get_current_orchestrator] Returning None! orchestrator={orchestrator}"
        )
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

    def select_function(
        self, parent: Optional[Any] = None, **context: Any
    ) -> Optional[Callable]:
        from openhcs.pyqt_gui.dialogs.function_selector_dialog import (
            FunctionSelectorDialog,
        )

        return FunctionSelectorDialog.select_function(parent=parent)


def register_openhcs_window_handlers():
    """Register OpenHCS window handlers with the generic factory.

    Note: This is a convenience re-export. The actual registration
    happens in window_handlers module.
    """
    from openhcs.pyqt_gui.services.window_handlers import (
        register_openhcs_window_handlers as _register,
    )

    _register()


def register_reactor_providers() -> None:
    """Register all OpenHCS providers with pyqt-reactor."""
    # FormGenConfig with OpenHCS paths
    config = OpenHCSFormGenConfig()
    try:
        from openhcs.core.xdg_paths import get_data_file_path, get_openhcs_data_dir

        config.path_cache_file = str(get_data_file_path("path_cache.json"))
        config.log_dir = str(get_openhcs_data_dir() / "logs")
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

    register_llm_service(LLMPipelineService())
    register_codegen_provider(OpenHCSCodegenProvider())
    register_function_registry(OpenHCSFunctionRegistry())
    register_log_discovery_provider(OpenHCSLogDiscoveryProvider())
    register_server_scan_provider(OpenHCSServerScanProvider())
    register_component_selection_provider(OpenHCSComponentSelectionProvider())
    register_function_selection_provider(OpenHCSFunctionSelectionProvider())
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
