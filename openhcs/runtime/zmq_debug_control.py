"""Debug control request routing for the ZMQ execution server."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta, RegistryFamily, RegistryKeyAttribute
from zmqruntime.messages import MessageFields, ResponseType


class DebugControlMessageStrategy(metaclass=AutoRegisterMeta):
    """Registered handler for one debug control message type."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)

    registry_key: ClassVar[str | None] = None

    @classmethod
    def for_message(cls, message: dict) -> "DebugControlMessageStrategy":
        message_type = message.get(MessageFields.TYPE)
        strategy_type = cls.__registry__.get(message_type)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported debug control message type {message_type!r}."
            )
        return strategy_type()

    def handle(self, message: dict) -> dict:
        raise NotImplementedError

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

    @staticmethod
    def error_response(error: Exception) -> dict:
        return {
            MessageFields.STATUS: ResponseType.ERROR.value,
            MessageFields.ERROR: str(error),
        }


class DebugControlMessageRouter:
    """Nominal facade for ZMQ debug-control message routing."""

    @classmethod
    def handles(cls, message: dict) -> bool:
        return message.get(MessageFields.TYPE) in DebugControlMessageStrategy.__registry__

    @classmethod
    def handle(cls, message: dict) -> dict:
        message_type = message.get(MessageFields.TYPE)
        strategy_type = DebugControlMessageStrategy.__registry__.get(message_type)
        if strategy_type is None:
            raise ValueError(
                f"Unsupported debug control message type {message_type!r}."
            )
        return strategy_type().handle(message)


class DebugSnapshotReadMessageStrategy(DebugControlMessageStrategy):
    """Handle debug snapshot read control messages."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.READ_SNAPSHOT.value

    def handle(self, message: dict) -> dict:
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


class DebugWorkerCommandMessageStrategy(DebugControlMessageStrategy):
    """Handle persistent paused-worker control messages."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.WORKER_COMMAND.value

    def handle(self, message: dict) -> dict:
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


class DebugArtifactExportMessageStrategy(DebugControlMessageStrategy):
    """Handle debug artifact export control messages."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.EXPORT_ARTIFACT.value

    def handle(self, message: dict) -> dict:
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


class DebugRuntimeInspectionMessageStrategy(DebugControlMessageStrategy):
    """Handle paused-worker live runtime inspection requests."""

    from openhcs.core.debug import DebugControlMessageType

    registry_key = DebugControlMessageType.INSPECT_RUNTIME.value

    def handle(self, message: dict) -> dict:
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
