"""Synthetic plate generation service for agent-facing workflows."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from openhcs.agent.dto.common import AgentError, SCHEMA_VERSION
from openhcs.agent.dto.plate import (
    PlateInspectionDefaults,
    PlatePathInspectionRequest,
    PlatePathInspectionResult,
    SyntheticPlateGenerationRequest,
    SyntheticPlateGenerationResult,
)
from openhcs.agent.path_policy import AgentPathPolicy, AgentPathPolicyError
from openhcs.agent.services.stdio import AgentStdoutRedirect
from openhcs.agent.services.plate_inspection_service import PlateInspectionService
from openhcs.core.synthetic_plate_generation import (
    SYNTHETIC_PLATE_GENERATION_PROFILE,
    SyntheticPlateFormat,
    SyntheticPlateGenerationParameters,
    SyntheticPlateGenerationProfile,
)


class SyntheticPlateGenerationIssueCode(str, Enum):
    """Structured issue codes returned by synthetic plate generation."""

    PATH_POLICY_REJECTED = "synthetic_plate_path_policy_rejected"
    OUTPUT_PATH_NOT_DIRECTORY = "synthetic_plate_output_path_not_directory"
    INVALID_ARGUMENT = "synthetic_plate_invalid_argument"
    GENERATION_FAILED = "synthetic_plate_generation_failed"


class SyntheticPlateGenerationText:
    """Agent-facing hints for synthetic plate generation failures."""

    WRITE_ROOT_HINT = "Pass output_dir under OPENHCS_AGENT_WRITE_ROOTS."
    DIRECTORY_HINT = "The synthetic plate output path must be a directory or creatable."
    ARGUMENT_HINT = "Use bounded positive dimensions and at least one well."
    GENERATION_HINT = "Retry with a smaller plate or inspect the generator arguments."


class SyntheticPlateGenerationService:
    """Generate bounded synthetic microscopy plates through the shared generator."""

    def __init__(
        self,
        path_policy: AgentPathPolicy | None = None,
        plate_inspection_service: PlateInspectionService | None = None,
        profile: SyntheticPlateGenerationProfile = SYNTHETIC_PLATE_GENERATION_PROFILE,
    ) -> None:
        self._path_policy = path_policy or AgentPathPolicy.from_environment()
        self._plate_inspection_service = plate_inspection_service or PlateInspectionService(
            self._path_policy
        )
        self._profile = profile

    def generate(
        self,
        request: SyntheticPlateGenerationRequest,
    ) -> SyntheticPlateGenerationResult:
        with AgentStdoutRedirect.to_stderr():
            return self._generate(request)

    def _generate(
        self,
        request: SyntheticPlateGenerationRequest,
    ) -> SyntheticPlateGenerationResult:
        try:
            output_dir = self._path_policy.assert_writable(request.output_dir)
        except AgentPathPolicyError as exc:
            return self._error_result(
                request,
                AgentError.from_exception(
                    SyntheticPlateGenerationIssueCode.PATH_POLICY_REJECTED.value,
                    exc,
                    hint=SyntheticPlateGenerationText.WRITE_ROOT_HINT,
                    path=request.output_dir,
                ),
            )
        try:
            self._validate_request(request)
            self._prepare_output_dir(output_dir)
            self._generate_dataset(request, output_dir)
        except ValueError as exc:
            return self._error_result(
                request,
                AgentError.from_exception(
                    SyntheticPlateGenerationIssueCode.INVALID_ARGUMENT.value,
                    exc,
                    hint=SyntheticPlateGenerationText.ARGUMENT_HINT,
                    path=str(output_dir),
                ),
                output_dir=output_dir,
            )
        except NotADirectoryError as exc:
            return self._error_result(
                request,
                AgentError.from_exception(
                    SyntheticPlateGenerationIssueCode.OUTPUT_PATH_NOT_DIRECTORY.value,
                    exc,
                    hint=SyntheticPlateGenerationText.DIRECTORY_HINT,
                    path=str(output_dir),
                ),
                output_dir=output_dir,
            )
        except Exception as exc:
            return self._error_result(
                request,
                AgentError.from_exception(
                    SyntheticPlateGenerationIssueCode.GENERATION_FAILED.value,
                    exc,
                    hint=SyntheticPlateGenerationText.GENERATION_HINT,
                    path=str(output_dir),
                ),
                output_dir=output_dir,
            )

        inspection = self._plate_inspection_service.inspect(
            PlatePathInspectionRequest.from_fields(
                plate_path=str(output_dir),
                microscope_type=PlateInspectionDefaults.MICROSCOPE_AUTO,
                max_sample_files=request.sample_file_limit,
            )
        )
        sampled_files = self._sampled_image_paths_from_inspection(inspection)
        return SyntheticPlateGenerationResult(
            schema_version=SCHEMA_VERSION,
            output_dir=str(output_dir),
            requested_format=self._result_format(request),
            grid_size=(request.grid_rows, request.grid_cols),
            tile_size=(request.tile_width, request.tile_height),
            overlap_percent=request.overlap_percent,
            stage_error_px=request.stage_error_px,
            wells=request.wells,
            wavelengths=request.wavelengths,
            z_stack_levels=request.z_stack_levels,
            num_cells=request.num_cells,
            shared_cell_fraction=request.shared_cell_fraction,
            image_count=inspection.image_files.count,
            sampled_image_files=sampled_files,
            truncated_image_count=inspection.image_files.truncated_file_count,
            metadata_file_path=inspection.metadata_file_path,
            detected_microscope_type=inspection.detected_microscope_type,
            handler_class=inspection.handler_class,
            include_all_components=request.include_all_components,
            errors=inspection.errors,
            warnings=inspection.warnings,
        )

    def _generate_dataset(
        self,
        request: SyntheticPlateGenerationRequest,
        output_dir: Path,
    ) -> None:
        from openhcs.demo.synthetic_data import (
            SyntheticMicroscopyGenerator,
        )

        generator = SyntheticMicroscopyGenerator(
            output_dir=str(output_dir),
            grid_size=(request.grid_rows, request.grid_cols),
            tile_size=(request.tile_width, request.tile_height),
            overlap_percent=request.overlap_percent,
            stage_error_px=request.stage_error_px,
            wavelengths=request.wavelengths,
            z_stack_levels=request.z_stack_levels,
            num_cells=request.num_cells,
            shared_cell_fraction=request.shared_cell_fraction,
            wells=list(request.wells),
            format=self._profile.format_from_value(request.format).value,
            openhcs_format=request.openhcs_format,
            include_all_components=request.include_all_components,
            random_seed=request.random_seed,
        )
        generator.generate_dataset()

    @staticmethod
    def _prepare_output_dir(output_dir: Path) -> None:
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(str(output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_request(self, request: SyntheticPlateGenerationRequest) -> None:
        self._profile.validate(self._parameters_from_request(request))

    @staticmethod
    def _parameters_from_request(
        request: SyntheticPlateGenerationRequest,
    ) -> SyntheticPlateGenerationParameters:
        return SyntheticPlateGenerationParameters(
            grid_rows=request.grid_rows,
            grid_cols=request.grid_cols,
            tile_width=request.tile_width,
            tile_height=request.tile_height,
            overlap_percent=request.overlap_percent,
            stage_error_px=request.stage_error_px,
            wavelengths=request.wavelengths,
            z_stack_levels=request.z_stack_levels,
            num_cells=request.num_cells,
            shared_cell_fraction=request.shared_cell_fraction,
            wells=request.wells,
            format=request.format,
            openhcs_format=request.openhcs_format,
            include_all_components=request.include_all_components,
            random_seed=request.random_seed,
            sample_file_limit=request.sample_file_limit,
        )

    @staticmethod
    def _sampled_image_paths_from_inspection(
        inspection: PlatePathInspectionResult,
    ) -> tuple[str, ...]:
        sampled_records = inspection.image_files.sampled_records
        if sampled_records:
            return tuple(record.virtual_path for record in sampled_records)
        return inspection.image_files.sampled_files

    def _error_result(
        self,
        request: SyntheticPlateGenerationRequest,
        error: AgentError,
        *,
        output_dir: Path | None = None,
    ) -> SyntheticPlateGenerationResult:
        return SyntheticPlateGenerationResult(
            schema_version=SCHEMA_VERSION,
            output_dir=str(output_dir) if output_dir is not None else request.output_dir,
            requested_format=self._result_format(request),
            grid_size=(request.grid_rows, request.grid_cols),
            tile_size=(request.tile_width, request.tile_height),
            overlap_percent=request.overlap_percent,
            stage_error_px=request.stage_error_px,
            wells=request.wells,
            wavelengths=request.wavelengths,
            z_stack_levels=request.z_stack_levels,
            num_cells=request.num_cells,
            shared_cell_fraction=request.shared_cell_fraction,
            include_all_components=request.include_all_components,
            errors=(error,),
        )

    def _result_format(
        self,
        request: SyntheticPlateGenerationRequest,
    ) -> SyntheticPlateFormat:
        try:
            return self._profile.format_from_value(request.format)
        except ValueError:
            return self._profile.default_request.format
