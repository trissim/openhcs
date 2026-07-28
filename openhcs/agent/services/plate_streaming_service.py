"""Agent-facing plate file streaming service."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from openhcs.agent.dto.common import AgentError, AgentWarning, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.plate import (
    PlateFileStreamRequest,
    PlateFileStreamResult,
    PlatePathInspectionRequest,
)
from openhcs.agent.services.plate_inspection_service import (
    PlateInspectionFileQueryProjection,
    PlateInspectionService,
)
from openhcs.agent.services.stdio import AgentStdoutRedirect
from openhcs.constants.constants import FileFormat
from openhcs.core.config import StreamingConfig, TransportMode
from openhcs.core.plate_image_inventory import (
    PlateFileInventoryQuery,
    PlateFileKind,
    PlateFileRecord,
)
from openhcs.core.viewer_streaming_service import (
    ImageStreamingRequest,
    RoiStreamingRequest,
    StreamingService,
    StreamingViewerLifecycle,
)
from openhcs.runtime.viewer_protocol import DetachedViewerLaunchFailure


class PlateStreamingService:
    """Stream plate inventory records to managed viewers through public core APIs."""

    def __init__(self, plate_inspection_service: PlateInspectionService) -> None:
        self._plate_inspection_service = plate_inspection_service

    def stream_files(
        self,
        request: PlateFileStreamRequest,
        *,
        launch_environment: Mapping[str, str] | None = None,
    ) -> PlateFileStreamResult:
        with AgentStdoutRedirect.to_stderr():
            return self._stream_files(
                request,
                launch_environment=launch_environment,
            )

    def _stream_files(
        self,
        request: PlateFileStreamRequest,
        *,
        launch_environment: Mapping[str, str] | None,
    ) -> PlateFileStreamResult:
        context_plate_path = request.context_plate_path or request.plate_path
        context, errors, warnings = self._plate_inspection_service.open_context(
            PlatePathInspectionRequest(
                plate_path=context_plate_path,
                microscope_type=request.microscope_type,
                pattern_format=request.pattern_format,
            )
        )
        if errors:
            return PlateFileStreamResult(
                schema_version=SCHEMA_VERSION,
                plate_path=request.plate_path,
                requested_microscope_type=request.microscope_type,
                viewer_config_key=request.viewer_config_key,
                errors=errors,
                warnings=warnings,
            )
        if context is None:
            raise RuntimeError("Plate context resolution returned no context and no error.")

        inventory_plate_path = context.plate_path
        if request.context_plate_path is not None:
            resolved_plate_path, path_errors = (
                self._plate_inspection_service.resolve_plate_path(request.plate_path)
            )
            if path_errors:
                return PlateFileStreamResult(
                    schema_version=SCHEMA_VERSION,
                    plate_path=request.plate_path,
                    requested_microscope_type=request.microscope_type,
                    viewer_config_key=request.viewer_config_key,
                    errors=path_errors,
                    warnings=warnings,
                )
            if resolved_plate_path is None:
                raise RuntimeError(
                    "Plate path resolution returned no path and no error."
                )
            inventory_plate_path = resolved_plate_path

        config = None
        connection = request.connection
        try:
            stream_context = replace(context, plate_path=inventory_plate_path)
            config = self._streaming_config(request)
            connection = replace(
                request.connection,
                host=config.host,
                port=config.port,
                transport_mode=config.transport_mode.value,
                persistent=config.persistent,
            )
            inventory, inventory_warnings = self._plate_inspection_service.file_inventory(
                stream_context,
                kind=None if request.file_paths else request.kind,
            )
            resolved_records = self._resolve_records(request, inventory)
            (
                image_paths,
                roi_paths,
                roi_component_metadata_by_path,
                skipped_records,
            ) = self._streamable_paths(
                resolved_records
            )
            all_warnings = inventory_warnings
            if skipped_records:
                all_warnings = (
                    *all_warnings,
                    AgentWarning(
                        code="plate_file_stream_skipped_records",
                        message=(
                            "Only image records and ROI result artifacts can be "
                            "streamed to viewers."
                        ),
                    ),
                )
            if not image_paths and not roi_paths:
                return PlateFileStreamResult(
                    schema_version=SCHEMA_VERSION,
                    plate_path=str(stream_context.plate_path),
                    requested_microscope_type=request.microscope_type,
                    detected_microscope_type=context.microscope_type,
                    handler_class=type(context.handler).__name__,
                    parser_class=(
                        None if context.parser is None else type(context.parser).__name__
                    ),
                    viewer_config_key=request.viewer_config_key,
                    viewer_type=config.viewer_type,
                    connection=connection,
                    requested_paths=request.file_paths,
                    resolved_records=self._record_summaries(resolved_records),
                    skipped_records=self._record_summaries(skipped_records),
                    errors=(
                        AgentError(
                            code="plate_file_stream_no_streamable_records",
                            message=(
                                "The resolved plate file records did not include any "
                                "streamable images or ROI artifacts."
                            ),
                        ),
                    ),
                    warnings=all_warnings,
                )

            viewer = StreamingViewerLifecycle.get_or_create_visualizer(
                filemanager=stream_context.filemanager,
                config=config,
                fresh=request.fresh_viewer,
                ready_timeout=30.0,
                launch_environment=launch_environment,
            )
            streaming_service = StreamingService(
                filemanager=stream_context.filemanager,
                microscope_handler=stream_context.handler,
                plate_path=stream_context.plate_path,
            )
            status_messages: list[str] = []
            read_backend = stream_context.handler.get_primary_backend(
                stream_context.plate_path,
                stream_context.filemanager,
            )
            if image_paths:
                streaming_service.stream_images(
                    ImageStreamingRequest(
                        viewer=viewer,
                        config=config,
                        status_callback=status_messages.append,
                        error_callback=status_messages.append,
                        filenames=image_paths,
                        read_backend=read_backend,
                    )
                )
            if roi_paths:
                streaming_service.stream_rois(
                    RoiStreamingRequest(
                        viewer=viewer,
                        config=config,
                        status_callback=status_messages.append,
                        error_callback=status_messages.append,
                        roi_filenames=roi_paths,
                        component_metadata_by_path=roi_component_metadata_by_path,
                    )
                )
        except Exception as exc:
            return PlateFileStreamResult(
                schema_version=SCHEMA_VERSION,
                plate_path=str(inventory_plate_path),
                requested_microscope_type=request.microscope_type,
                detected_microscope_type=context.microscope_type,
                handler_class=type(context.handler).__name__,
                parser_class=None if context.parser is None else type(context.parser).__name__,
                viewer_config_key=request.viewer_config_key,
                viewer_type=None if config is None else config.viewer_type,
                connection=connection,
                requested_paths=request.file_paths,
                errors=(
                    self._stream_error(exc, plate_path=request.plate_path),
                ),
                warnings=warnings,
            )

        return PlateFileStreamResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(inventory_plate_path),
            requested_microscope_type=request.microscope_type,
            detected_microscope_type=context.microscope_type,
            handler_class=type(context.handler).__name__,
            parser_class=None if context.parser is None else type(context.parser).__name__,
            viewer_config_key=request.viewer_config_key,
            viewer_type=config.viewer_type,
            connection=connection,
            requested_paths=request.file_paths,
            resolved_records=self._record_summaries(resolved_records),
            streamed_image_paths=image_paths,
            streamed_roi_paths=roi_paths,
            skipped_records=self._record_summaries(skipped_records),
            status_messages=tuple(status_messages),
            warnings=all_warnings,
        )

    @staticmethod
    def _stream_error(
        exception: Exception,
        *,
        plate_path: str,
    ) -> AgentError:
        if isinstance(exception, DetachedViewerLaunchFailure):
            return AgentError.from_exception(
                "plate_file_stream_failed",
                exception,
                hint=(
                    "The bounded tail of the detached viewer launch log is included "
                    "in this error. Inspect the structured path for the complete log."
                ),
                path=str(exception.log_file),
            )
        return AgentError.from_exception(
            "plate_file_stream_failed",
            exception,
            path=plate_path,
        )

    @staticmethod
    def _streaming_config(request: PlateFileStreamRequest) -> StreamingConfig:
        config_type = StreamingConfig.config_type_for_key(request.viewer_config_key)
        values = {
            "enabled": True,
            "host": request.connection.host,
            "persistent": request.connection.persistent,
        }
        if request.connection.port is not None:
            values["port"] = request.connection.port
        if request.connection.transport_mode is not None:
            values["transport_mode"] = TransportMode(request.connection.transport_mode)
        return config_type(**values)

    @classmethod
    def _resolve_records(
        cls,
        request: PlateFileStreamRequest,
        inventory,
    ) -> tuple[PlateFileRecord, ...]:
        kinds = PlateFileInventoryQuery.kinds_for(request.kind)
        if request.file_paths:
            return tuple(
                inventory.require_file_record(file_path, kinds=kinds)
                for file_path in request.file_paths
            )
        requested_limit = max(0, int(request.limit))
        if requested_limit == 0:
            return ()

        inventory_query = PlateFileInventoryQuery(
            kinds=kinds,
            path_contains=request.path_contains,
            well=request.well,
            offset=0,
            limit=requested_limit,
        )
        query = inventory.query_files(inventory_query)
        if (
            cls._streamable_record_count(query.records) < requested_limit
            and query.truncated_count
        ):
            query = inventory.query_files(
                replace(inventory_query, limit=query.total_count)
            )
        return cls._query_streamable_records(query.records, requested_limit)

    @classmethod
    def _query_streamable_records(
        cls,
        records: tuple[PlateFileRecord, ...],
        limit: int,
    ) -> tuple[PlateFileRecord, ...]:
        streamable_records = tuple(
            record for record in records if cls._is_streamable_record(record)
        )
        if streamable_records:
            return streamable_records[:limit]
        return records[:limit]

    @classmethod
    def _streamable_record_count(
        cls,
        records: tuple[PlateFileRecord, ...],
    ) -> int:
        return sum(1 for record in records if cls._is_streamable_record(record))

    @staticmethod
    def _is_streamable_record(record: PlateFileRecord) -> bool:
        if record.kind is PlateFileKind.IMAGE:
            return True
        return (
            record.kind is PlateFileKind.RESULT
            and record.file_format is FileFormat.ROI
        )

    @classmethod
    def _streamable_paths(
        cls,
        records: tuple[PlateFileRecord, ...],
    ) -> tuple[
        tuple[str, ...],
        tuple[str, ...],
        dict[str, JsonObject],
        tuple[PlateFileRecord, ...],
    ]:
        image_paths: list[str] = []
        roi_paths: list[str] = []
        roi_component_metadata_by_path: dict[str, JsonObject] = {}
        skipped_records: list[PlateFileRecord] = []
        for record in records:
            if not cls._is_streamable_record(record):
                skipped_records.append(record)
                continue
            if record.kind is PlateFileKind.IMAGE:
                image_paths.append(record.key)
            elif record.kind is PlateFileKind.RESULT:
                roi_path = record.full_path or record.key
                roi_paths.append(roi_path)
                if record.metadata:
                    roi_component_metadata_by_path[roi_path] = dict(record.metadata)
        return (
            tuple(image_paths),
            tuple(roi_paths),
            roi_component_metadata_by_path,
            tuple(skipped_records),
        )

    @staticmethod
    def _record_summaries(records: tuple[PlateFileRecord, ...]):
        return tuple(
            PlateInspectionFileQueryProjection.record(
                record,
                include_preview=False,
                max_preview_lines=0,
                max_preview_bytes=0,
            )
            for record in records
        )
