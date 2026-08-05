"""Registered OpenHCS control request routing for the execution server."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from zmqruntime.messages import MessageFields, ResponseType

if TYPE_CHECKING:
    from openhcs.agent.services.function_catalog_service import FunctionCatalogService
    from openhcs.runtime.function_catalog_preparation import FunctionCatalogPreparation
    from openhcs.runtime.zmq_compilation import ZMQCompileArtifactRecord


@dataclass(frozen=True, slots=True)
class ZMQControlRequestContext:
    """Server-owned resources available to registered control strategies."""

    compiled_artifacts: Mapping[str, "ZMQCompileArtifactRecord"]
    function_catalog: "FunctionCatalogService | None" = None
    function_catalog_preparation: "FunctionCatalogPreparation | None" = None

    @classmethod
    def empty(cls) -> "ZMQControlRequestContext":
        return cls(compiled_artifacts={})

    def require_function_catalog(self) -> "FunctionCatalogService":
        """Return the catalog owned by the serving execution endpoint."""

        if self.function_catalog is None:
            raise RuntimeError(
                "The execution endpoint did not provide its function catalog service."
            )
        return self.function_catalog

    def require_function_catalog_preparation(self) -> "FunctionCatalogPreparation":
        """Return the preparation lifecycle owned by this execution endpoint."""

        if self.function_catalog_preparation is None:
            raise RuntimeError(
                "The execution endpoint did not provide function catalog "
                "preparation."
            )
        return self.function_catalog_preparation


class ZMQControlMessageStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered handler for one OpenHCS control message type."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    registry_key: ClassVar[str | None] = None

    @abstractmethod
    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        """Handle one typed control message."""

    @staticmethod
    def error_response(error: Exception) -> dict:
        return {
            MessageFields.STATUS: ResponseType.ERROR.value,
            MessageFields.ERROR: str(error),
        }


class DebugControlMessageSupportMixin:
    """Storage construction shared only by debug control strategy leaves."""

    @staticmethod
    def snapshot_store_for_request(request):
        from openhcs.core.debug import (
            FileManagerDebugSnapshotStore,
            LocalDebugSnapshotStore,
        )

        if request.snapshot_store_backend is None:
            return LocalDebugSnapshotStore(
                root_path=request.snapshot_store_ref,
                debug_session_id=request.debug_session_id,
            )

        from polystore.base import ensure_storage_registry, storage_registry
        from polystore.filemanager import FileManager

        ensure_storage_registry()
        return FileManagerDebugSnapshotStore(
            filemanager=FileManager(storage_registry),
            backend=request.snapshot_store_backend,
            root_path=request.snapshot_store_ref,
            debug_session_id=request.debug_session_id,
        )

    @staticmethod
    def artifact_filemanager(request):
        if request.artifact_ref.storage_backend is None:
            return None
        from polystore.base import ensure_storage_registry, storage_registry
        from polystore.filemanager import FileManager

        ensure_storage_registry()
        return FileManager(storage_registry)

class ZMQControlMessageRouter:
    """Nominal facade for registered OpenHCS control messages."""

    @classmethod
    def handles(cls, message: dict) -> bool:
        return message.get(MessageFields.TYPE) in ZMQControlMessageStrategy.__registry__

    @classmethod
    def handle(
        cls,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        message_type = message.get(MessageFields.TYPE)
        strategy_type = ZMQControlMessageStrategy.__registry__.get(message_type)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported OpenHCS control message type {message_type!r}."
            )
        return strategy_type().handle(message, context)


class CompiledArtifactInspectionMessageStrategy(ZMQControlMessageStrategy):
    """Read the exact artifact plans retained by one compile artifact."""

    from openhcs.core.artifact_inspection import ArtifactInspectionControlMessageType

    registry_key = ArtifactInspectionControlMessageType.READ_COMPILED.value

    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        from openhcs.core.artifact_inspection import (
            CompiledArtifactInspection,
            CompiledArtifactInspectionControlPayload,
            CompiledArtifactInspectionResponse,
        )

        try:
            request = CompiledArtifactInspectionControlPayload.from_dict(
                message
            ).to_request()
            record = context.compiled_artifacts.get(request.compile_artifact_id)
            if record is None:
                raise KeyError(
                    f"Compile artifact {request.compile_artifact_id!r} is unavailable."
                )
            inspection = CompiledArtifactInspection.from_execution_bundle(
                compile_artifact_id=record.execution_id,
                plate_id=record.plate_id,
                bundle=record.compilation.execution_bundle,
            )
            return CompiledArtifactInspectionResponse(
                inspection=inspection
            ).to_control_response()
        except Exception as error:
            return self.error_response(error)


class FunctionCatalogMessageStrategy(ZMQControlMessageStrategy):
    """Template method for catalog requests gated by live preparation."""

    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        from openhcs.agent.dto.functions import (
            FunctionCatalogPreparationControlResponse,
        )

        try:
            future = context.require_function_catalog_preparation().ensure_started()
            if not future.done():
                return FunctionCatalogPreparationControlResponse().to_control_response()
            future.result()
            return self.handle_ready(
                message,
                context.require_function_catalog(),
            )
        except Exception as error:
            return self.error_response(error)

    @abstractmethod
    def handle_ready(
        self,
        message: dict,
        function_catalog: "FunctionCatalogService",
    ) -> dict:
        """Handle one request after the endpoint catalog is ready."""


class FunctionCatalogReadMessageStrategy(FunctionCatalogMessageStrategy):
    """Project the execution endpoint's authoritative callable catalog."""

    from openhcs.agent.dto.functions import FunctionCatalogControlMessageType

    registry_key = FunctionCatalogControlMessageType.READ_CATALOG.value

    def handle_ready(
        self,
        message: dict,
        function_catalog: "FunctionCatalogService",
    ) -> dict:
        from openhcs.agent.dto.functions import (
            FunctionCatalogControlPayload,
            FunctionCatalogControlResponse,
        )

        request = FunctionCatalogControlPayload.from_dict(message).request
        catalog = function_catalog.catalog(
            compact_signatures=request.compact_signatures,
        )
        return FunctionCatalogControlResponse(
            catalog=catalog,
        ).to_control_response()


class FunctionCatalogSearchMessageStrategy(FunctionCatalogMessageStrategy):
    """Run the catalog owner's search policy at the execution endpoint."""

    from openhcs.agent.dto.functions import FunctionCatalogControlMessageType

    registry_key = FunctionCatalogControlMessageType.SEARCH_CATALOG.value

    def handle_ready(
        self,
        message: dict,
        function_catalog: "FunctionCatalogService",
    ) -> dict:
        from openhcs.agent.dto.functions import (
            FunctionCatalogControlResponse,
            FunctionSearchControlPayload,
        )

        request = FunctionSearchControlPayload.from_dict(message).request
        catalog = function_catalog.search(
            query=request.query,
            library=request.library,
            limit=request.limit,
            compact_signatures=request.compact_signatures,
        )
        return FunctionCatalogControlResponse(
            catalog=catalog,
        ).to_control_response()


class FunctionDetailReadMessageStrategy(FunctionCatalogMessageStrategy):
    """Project one callable detail from an exact endpoint catalog revision."""

    from openhcs.agent.dto.functions import FunctionCatalogControlMessageType

    registry_key = FunctionCatalogControlMessageType.READ_DETAIL.value

    def handle_ready(
        self,
        message: dict,
        function_catalog: "FunctionCatalogService",
    ) -> dict:
        from openhcs.agent.dto.functions import (
            FunctionDetailControlPayload,
            FunctionDetailControlResponse,
        )

        request = FunctionDetailControlPayload.from_dict(message).request
        current_catalog = function_catalog.catalog(compact_signatures=True)
        if request.catalog_revision != current_catalog.revision:
            raise ValueError(
                "The execution server function catalog changed after it was read. "
                "Refresh the catalog before selecting a function."
            )
        detail = function_catalog.get(
            request.function_id,
            max_doc_chars=request.max_doc_chars,
            compact_signature=request.compact_signature,
        )
        return FunctionDetailControlResponse(
            detail=detail,
        ).to_control_response()


class DebugSnapshotReadMessageStrategy(
    DebugControlMessageSupportMixin,
    ZMQControlMessageStrategy,
):
    """Handle debug snapshot read control messages."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.READ_SNAPSHOT.value

    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        del context
        from openhcs.core.debug import (
            DebugSnapshotReadControlPayload,
            DebugSnapshotReadResponse,
        )

        try:
            request = DebugSnapshotReadControlPayload.from_dict(message).to_request()
            snapshot = self.snapshot_store_for_request(request).read_snapshot(
                request.snapshot_id
            )
            return DebugSnapshotReadResponse(snapshot=snapshot).to_control_response()
        except Exception as error:
            return self.error_response(error)


class DebugWorkerCommandMessageStrategy(ZMQControlMessageStrategy):
    """Handle persistent paused-worker control messages."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.WORKER_COMMAND.value

    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        del context
        from openhcs.core.debug import (
            DebugPausedWorkerRegistry,
            DebugWorkerCommandControlPayload,
            DebugWorkerCommandResponse,
        )

        try:
            request = DebugWorkerCommandControlPayload.from_dict(message).to_request()
            status = DebugPausedWorkerRegistry.controller_for(
                request.debug_session_id
            ).apply_command(request.command_type)
            return DebugWorkerCommandResponse(status=status).to_control_response()
        except Exception as error:
            return self.error_response(error)


class DebugArtifactExportMessageStrategy(
    DebugControlMessageSupportMixin,
    ZMQControlMessageStrategy,
):
    """Handle debug artifact export control messages."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.EXPORT_ARTIFACT.value

    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        del context
        from openhcs.core.debug import (
            DebugArtifactExportControlPayload,
            DebugArtifactExportPlan,
            DebugArtifactExportResponse,
        )

        try:
            request = DebugArtifactExportControlPayload.from_dict(message).to_request()
            exported_path = DebugArtifactExportPlan(
                artifact_ref=request.artifact_ref,
                export_root=Path(request.export_root),
                filemanager=self.artifact_filemanager(request),
            ).export()
            return DebugArtifactExportResponse(
                exported_ref=str(exported_path)
            ).to_control_response()
        except Exception as error:
            return self.error_response(error)


class DebugRuntimeInspectionMessageStrategy(ZMQControlMessageStrategy):
    """Handle paused-worker live runtime inspection requests."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.INSPECT_RUNTIME.value

    def handle(
        self,
        message: dict,
        context: ZMQControlRequestContext,
    ) -> dict:
        del context
        from openhcs.core.debug import (
            DebugPausedWorkerRegistry,
            DebugRuntimeInspectionControlPayload,
            DebugRuntimeInspectionResponse,
        )

        try:
            request = DebugRuntimeInspectionControlPayload.from_dict(message).to_request()
            view_model = DebugPausedWorkerRegistry.controller_for(
                request.debug_session_id
            ).runtime_inspection_view()
            return DebugRuntimeInspectionResponse(
                view_model=view_model
            ).to_control_response()
        except Exception as error:
            return self.error_response(error)
