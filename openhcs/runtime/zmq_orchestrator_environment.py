"""Environment preparation for ZMQ orchestrator execution."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

from openhcs.runtime.zmq_execution_signature import ZMQExecutionIdentity

if TYPE_CHECKING:
    from openhcs.core.debug import DebugExecutionConfig, DebugExecutionPolicy


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ZMQOrchestratorEnvironment:
    """Prepared execution environment for one ZMQ orchestrator run."""

    debug_execution_policy: DebugExecutionPolicy
    debug_execution_config: DebugExecutionConfig | None
    plate_path_str: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ZMQOrchestratorEnvironmentRequest(ZMQExecutionIdentity):
    """Inputs needed to prepare the worker execution environment."""

    execution_id: str
    debug_execution_config: DebugExecutionConfig | None

    def prepare(self) -> ZMQOrchestratorEnvironment:
        from openhcs.core.debug import DebugExecutionPolicy
        from polystore.base import reset_memory_backend, storage_registry

        reset_memory_backend()
        self.cleanup_gpu_frameworks()

        debug_execution_policy = DebugExecutionPolicy.from_config(
            self.debug_execution_config
        )

        return ZMQOrchestratorEnvironment(
            debug_execution_policy=debug_execution_policy,
            debug_execution_config=self.debug_execution_config,
            plate_path_str=self.prepared_plate_path(storage_registry),
        )

    def cleanup_gpu_frameworks(self) -> None:
        try:
            from openhcs.core.memory import cleanup_all_gpu_frameworks

            cleanup_all_gpu_frameworks()
        except Exception as cleanup_error:
            logger.warning(
                "[%s] Failed to trigger GPU cleanup: %s",
                self.execution_id,
                cleanup_error,
            )

    def prepared_plate_path(self, storage_registry) -> str:
        if self.selected_pipeline_path is not None and self.execution_plate_id is not None:
            plate_path_str = str(self.execution_plate_id)
        else:
            plate_path_str = str(self.plate_id)
        is_omero_plate_id = False
        try:
            int(plate_path_str)
            is_omero_plate_id = True
        except ValueError:
            is_omero_plate_id = plate_path_str.startswith("/omero/")

        if not is_omero_plate_id:
            return plate_path_str

        from openhcs.runtime.omero_instance_manager import OMEROInstanceManager
        from openhcs.microscopes import omero  # noqa: F401
        from polystore.omero_local import OMEROLocalBackend

        omero_manager = OMEROInstanceManager()
        if not omero_manager.connect(timeout=60):
            raise RuntimeError("OMERO server not available")
        storage_registry["omero_local"] = OMEROLocalBackend(
            omero_conn=omero_manager.conn,
            namespace_prefix="openhcs",
            lock_dir_name=".openhcs",
        )
        if plate_path_str.startswith("/omero/"):
            return plate_path_str
        return f"/omero/plate_{plate_path_str}"
