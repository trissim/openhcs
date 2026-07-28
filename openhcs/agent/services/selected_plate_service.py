"""Agent service for operations on the UI-selected PlateManager row."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from openhcs.agent.dto.common import AgentError, AgentWarning, JsonValue, SCHEMA_VERSION
from openhcs.agent.dto.plate import (
    PlateInspectionDefaults,
    PlatePathInspectionRequest,
    SelectedPlateFileQueryRequest,
    SelectedPlateFileQueryResult,
    SelectedPlateFileQueryTarget,
    SelectedPlateFileStreamRequest,
    SelectedPlateFileStreamResult,
    SelectedPlateImageInspectionRequest,
    SelectedPlateImageInspectionResult,
    SelectedPlateImageSampleRequest,
    SelectedPlateImageSampleResult,
)
from openhcs.agent.dto.ui_bridge import (
    UiBridgeConnectionSpec,
    UiStateSurfaceDocument,
    UiStateSurfaceRequest,
)
from openhcs.agent.ui_bridge_identities import PlateManagerStateSurfaceIdentityDeclaration
from openhcs.agent.services.plate_inspection_service import PlateInspectionService
from openhcs.agent.services.plate_streaming_service import PlateStreamingService
from openhcs.agent.services.ui_bridge_service import UiBridgeService
from openhcs.core.plate_image_inventory import PlateFileKind
from openhcs.core.selection import SelectedAllSelectionMode


@dataclass(frozen=True, slots=True)
class SelectedPlateStateResolution:
    selected_plate: Mapping[str, JsonValue] | None
    warnings: tuple[AgentWarning, ...] = ()
    errors: tuple[AgentError, ...] = ()

    @property
    def plate_root(self) -> str | None:
        if self.selected_plate is None:
            return None
        plate_root = self.selected_plate.get("plate_root")
        if not isinstance(plate_root, str):
            return None
        return plate_root


@dataclass(frozen=True, slots=True)
class SelectedPlateTargetRoot:
    plate_root: str
    microscope_type: str


@dataclass(frozen=True, slots=True)
class SelectedPlateService:
    """Service-owned selected-plate operations for MCP/UI agent tools."""

    ui_bridge_service: UiBridgeService
    plate_inspection_service: PlateInspectionService
    plate_streaming_service: PlateStreamingService

    def inspect_images(
        self,
        request: SelectedPlateImageInspectionRequest,
        connection: UiBridgeConnectionSpec,
    ) -> SelectedPlateImageInspectionResult:
        selected_plate = self.selected_plate_state(connection)
        if selected_plate.errors:
            return SelectedPlateImageInspectionResult(
                schema_version=SCHEMA_VERSION,
                errors=selected_plate.errors,
                warnings=selected_plate.warnings,
            )
        target_root, target_error = self.selected_plate_target_root(
            selected_plate,
            target=request.target,
            microscope_type=request.microscope_type,
        )
        if target_error is not None:
            return SelectedPlateImageInspectionResult(
                schema_version=SCHEMA_VERSION,
                selected_plate=dict(selected_plate.selected_plate or {}),
                target=request.target,
                errors=(target_error,),
                warnings=selected_plate.warnings,
            )
        if target_root is None:
            raise RuntimeError(
                "Selected plate target resolution returned no root and no error."
            )

        inspection = self.plate_inspection_service.inspect(
            request.to_plate_path_inspection_request(
                plate_path=target_root.plate_root,
                microscope_type=target_root.microscope_type,
            )
        )
        return SelectedPlateImageInspectionResult(
            schema_version=SCHEMA_VERSION,
            selected_plate=dict(selected_plate.selected_plate or {}),
            target=request.target,
            inspection=inspection,
            warnings=selected_plate.warnings,
        )

    def query_files(
        self,
        request: SelectedPlateFileQueryRequest,
        connection: UiBridgeConnectionSpec,
    ) -> SelectedPlateFileQueryResult:
        selected_plate = self.selected_plate_state(connection)
        if selected_plate.errors:
            return SelectedPlateFileQueryResult(
                schema_version=SCHEMA_VERSION,
                errors=selected_plate.errors,
                warnings=selected_plate.warnings,
            )
        target_root, target_error = self.selected_plate_target_root(
            selected_plate,
            target=request.target,
            microscope_type=request.microscope_type,
        )
        if target_error is not None:
            return SelectedPlateFileQueryResult(
                schema_version=SCHEMA_VERSION,
                selected_plate=dict(selected_plate.selected_plate or {}),
                target=request.target,
                errors=(target_error,),
                warnings=selected_plate.warnings,
            )
        if target_root is None:
            raise RuntimeError(
                "Selected plate target resolution returned no root and no error."
            )

        query = self.plate_inspection_service.query_files(
            request.to_plate_file_query_request(
                plate_path=target_root.plate_root,
                microscope_type=target_root.microscope_type,
            )
        )
        return SelectedPlateFileQueryResult(
            schema_version=SCHEMA_VERSION,
            selected_plate=dict(selected_plate.selected_plate or {}),
            target=request.target,
            query=query,
            errors=query.errors,
            warnings=(*selected_plate.warnings, *query.warnings),
        )

    def sample_image(
        self,
        request: SelectedPlateImageSampleRequest,
        connection: UiBridgeConnectionSpec,
    ) -> SelectedPlateImageSampleResult:
        selected_plate = self.selected_plate_state(connection)
        if selected_plate.errors:
            return SelectedPlateImageSampleResult(
                schema_version=SCHEMA_VERSION,
                errors=selected_plate.errors,
                warnings=selected_plate.warnings,
            )
        target_root, target_error = self.selected_plate_target_root(
            selected_plate,
            target=request.target,
            microscope_type=request.microscope_type,
        )
        if target_error is not None:
            return SelectedPlateImageSampleResult(
                schema_version=SCHEMA_VERSION,
                selected_plate=dict(selected_plate.selected_plate or {}),
                target=request.target,
                errors=(target_error,),
                warnings=selected_plate.warnings,
            )
        if target_root is None:
            raise RuntimeError(
                "Selected plate target resolution returned no root and no error."
            )

        image_path = request.image_path
        auto_selected_image_path = image_path is None
        warnings = selected_plate.warnings
        if image_path is None:
            image_path, auto_warnings, auto_errors = (
                self.first_selected_plate_image_path(
                    plate_root=target_root.plate_root,
                    microscope_type=target_root.microscope_type,
                    pattern_format=request.pattern_format,
                )
            )
            warnings = (*warnings, *auto_warnings)
            if auto_errors:
                return SelectedPlateImageSampleResult(
                    schema_version=SCHEMA_VERSION,
                    selected_plate=dict(selected_plate.selected_plate or {}),
                    target=request.target,
                    auto_selected_image_path=auto_selected_image_path,
                    errors=auto_errors,
                    warnings=warnings,
                )
        if image_path is None:
            raise RuntimeError(
                "Auto-selected image path returned no image and no error."
            )

        sample = self.plate_inspection_service.sample_image(
            request.to_plate_image_sample_request(
                plate_path=target_root.plate_root,
                image_path=image_path,
                microscope_type=target_root.microscope_type,
            )
        )
        return SelectedPlateImageSampleResult(
            schema_version=SCHEMA_VERSION,
            selected_plate=dict(selected_plate.selected_plate or {}),
            target=request.target,
            image_path=image_path,
            auto_selected_image_path=auto_selected_image_path,
            sample=sample,
            errors=sample.errors,
            warnings=(*warnings, *sample.warnings),
        )

    def stream_files(
        self,
        request: SelectedPlateFileStreamRequest,
        connection: UiBridgeConnectionSpec,
    ) -> SelectedPlateFileStreamResult:
        selected_plate = self.selected_plate_state(connection)
        if selected_plate.errors:
            return SelectedPlateFileStreamResult(
                schema_version=SCHEMA_VERSION,
                errors=selected_plate.errors,
                warnings=selected_plate.warnings,
            )
        target_root, target_error = self.selected_plate_target_root(
            selected_plate,
            target=request.target,
            microscope_type=request.microscope_type,
        )
        if target_error is not None:
            return SelectedPlateFileStreamResult(
                schema_version=SCHEMA_VERSION,
                selected_plate=dict(selected_plate.selected_plate or {}),
                target=request.target,
                errors=(target_error,),
                warnings=selected_plate.warnings,
            )
        if target_root is None:
            raise RuntimeError(
                "Selected plate target resolution returned no root and no error."
            )

        stream = self.plate_streaming_service.stream_files(
            request.to_plate_file_stream_request(
                plate_path=target_root.plate_root,
                context_plate_path=self.stream_context_plate_root(
                    selected_plate,
                    target=request.target,
                    kind=request.kind,
                ),
                microscope_type=target_root.microscope_type,
            ),
            launch_environment=self.ui_bridge_service.graphical_child_environment(
                connection
            ),
        )
        return SelectedPlateFileStreamResult(
            schema_version=SCHEMA_VERSION,
            selected_plate=dict(selected_plate.selected_plate or {}),
            target=request.target,
            stream=stream,
            errors=stream.errors,
            warnings=(*selected_plate.warnings, *stream.warnings),
        )

    def selected_plate_state(
        self,
        connection: UiBridgeConnectionSpec,
    ) -> SelectedPlateStateResolution:
        state_document = self.ui_bridge_service.get_state_surface(
            UiStateSurfaceRequest(
                surface_id=PlateManagerStateSurfaceIdentityDeclaration.require_value(),
                selection_mode=SelectedAllSelectionMode.SELECTED.value,
            ),
            connection,
        )
        if state_document.errors:
            return SelectedPlateStateResolution(
                selected_plate=None,
                errors=state_document.errors,
                warnings=state_document.warnings,
            )

        selected_row, selection_error = self.selected_plate_row_from_state_surface(
            state_document
        )
        if selection_error is not None:
            return SelectedPlateStateResolution(
                selected_plate=None,
                errors=(selection_error,),
                warnings=state_document.warnings,
            )
        if selected_row is None:
            raise RuntimeError("Selected row resolution returned no row and no error.")
        return SelectedPlateStateResolution(
            selected_plate=selected_row,
            warnings=state_document.warnings,
        )

    def selected_plate_target_root(
        self,
        selected_plate: SelectedPlateStateResolution,
        *,
        target: SelectedPlateFileQueryTarget,
        microscope_type: str,
    ) -> tuple[SelectedPlateTargetRoot | None, AgentError | None]:
        state_surface_id = PlateManagerStateSurfaceIdentityDeclaration.require_value()
        plate_root = selected_plate.plate_root
        if plate_root is None:
            raise RuntimeError("Selected row plate_root was validated as a string.")
        selected_row = selected_plate.selected_plate or {}
        if target is SelectedPlateFileQueryTarget.OUTPUT:
            output_root = selected_row.get("output_plate_root")
            if not isinstance(output_root, str) or not output_root:
                return None, AgentError(
                    code="ui_selected_plate_output_root_unavailable",
                    message=(
                        "The selected PlateManager row does not expose an "
                        "output_plate_root."
                    ),
                    hint=(
                        "Call openhcs_ui_get_state_surface(surface_id="
                        f"{state_surface_id!r}) to inspect source/output plate "
                        "relationships."
                    ),
                )
            plate_root = output_root
        elif target is SelectedPlateFileQueryTarget.SOURCE:
            source_root = selected_row.get("source_plate_root")
            if isinstance(source_root, str) and source_root:
                plate_root = source_root
        return SelectedPlateTargetRoot(
            plate_root=plate_root,
            microscope_type=microscope_type,
        ), None

    @staticmethod
    def selected_plate_row_from_state_surface(
        state_document: UiStateSurfaceDocument,
    ) -> tuple[Mapping[str, JsonValue] | None, AgentError | None]:
        state_surface_id = PlateManagerStateSurfaceIdentityDeclaration.require_value()
        selected_scope_ids = state_document.selected_scope_ids
        if len(selected_scope_ids) != 1:
            return None, AgentError(
                code="ui_selected_plate_count_not_one",
                message=(
                    "Expected exactly one selected PlateManager row; "
                    f"found {len(selected_scope_ids)}."
                ),
                hint=(
                    "Select one plate in the UI, or call "
                    "openhcs_ui_get_state_surface and then "
                    "openhcs_inspect_plate_path for an explicit plate_root."
                ),
            )

        selected_scope_id = selected_scope_ids[0]
        rows_value = state_document.payload.get("rows")
        if not isinstance(rows_value, list | tuple):
            return None, AgentError(
                code="ui_plate_manager_state_rows_unavailable",
                message=f"{state_surface_id} did not include a rows collection.",
                hint=(
                    "Call openhcs_ui_get_state_surface(surface_id="
                    f"{state_surface_id!r}) to inspect the raw UI state."
                ),
            )

        matched_rows = tuple(
            row
            for row in rows_value
            if isinstance(row, Mapping)
            and row.get("plate_scope_id") == selected_scope_id
        )
        if len(matched_rows) != 1:
            return None, AgentError(
                code="ui_selected_plate_row_not_found",
                message=(
                    f"{state_surface_id} selected_scope_ids did not match "
                    f"exactly one row for scope {selected_scope_id!r}."
                ),
                hint=(
                    f"Refresh {state_surface_id} and verify the UI bridge is "
                    "connected to the intended window."
                ),
            )

        selected_row = matched_rows[0]
        if selected_row.get("selected") is not True:
            return None, AgentError(
                code="ui_selected_plate_row_inconsistent",
                message=(
                    f"{state_surface_id} returned a matching selected_scope_id "
                    "row whose selected flag was not true."
                ),
                hint=(
                    f"Refresh {state_surface_id} before inspecting selected "
                    "plate images."
                ),
            )
        if (
            not isinstance(selected_row.get("plate_root"), str)
            or not selected_row.get("plate_root")
        ):
            return None, AgentError(
                code="ui_selected_plate_root_unavailable",
                message=(
                    "The selected PlateManager row did not include a usable "
                    "plate_root."
                ),
                hint=(
                    "Use openhcs_ui_get_state_surface to inspect the selected "
                    "row payload."
                ),
            )
        return selected_row, None

    def first_selected_plate_image_path(
        self,
        *,
        plate_root: str,
        microscope_type: str,
        pattern_format: str | None,
    ) -> tuple[str | None, tuple[AgentWarning, ...], tuple[AgentError, ...]]:
        inspection = self.plate_inspection_service.inspect(
            PlatePathInspectionRequest.from_fields(
                plate_path=plate_root,
                microscope_type=microscope_type,
                pattern_format=pattern_format,
                max_sample_files=1,
                max_component_values=0,
                max_parse_failure_samples=0,
                max_files_to_parse=(
                    PlateInspectionDefaults.DEFAULT_MAX_FILES_TO_PARSE
                ),
            )
        )
        if inspection.errors:
            return None, inspection.warnings, inspection.errors
        if inspection.image_files.sampled_records:
            return (
                inspection.image_files.sampled_records[0].virtual_path,
                inspection.warnings,
                (),
            )
        if inspection.image_files.sampled_files:
            return inspection.image_files.sampled_files[0], inspection.warnings, ()
        return (
            None,
            inspection.warnings,
            (
                AgentError(
                    code="ui_selected_plate_no_sample_image",
                    message=(
                        "The selected plate did not report any image files to sample."
                    ),
                    hint=(
                        "Call openhcs_ui_inspect_selected_plate_images to inspect "
                        "the selected plate inventory."
                    ),
                    path=plate_root,
                ),
            ),
        )

    @staticmethod
    def stream_context_plate_root(
        selected_plate: SelectedPlateStateResolution,
        *,
        target: SelectedPlateFileQueryTarget,
        kind: PlateFileKind | None,
    ) -> str | None:
        if (
            target is not SelectedPlateFileQueryTarget.OUTPUT
            or kind is not PlateFileKind.RESULT
        ):
            return None
        selected_row = selected_plate.selected_plate or {}
        source_root = selected_row.get("source_plate_root")
        if isinstance(source_root, str) and source_root:
            return source_root
        return selected_plate.plate_root
