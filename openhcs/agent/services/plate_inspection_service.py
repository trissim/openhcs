"""Read-only plate folder inspection service for OpenHCS agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from polystore.base import ImageSamplingRequest

from openhcs.agent.dto.common import AgentError, AgentWarning, JsonObject, SCHEMA_VERSION
from openhcs.agent.dto.plate import (
    PlateFileQueryRecordSummary,
    PlateFileQueryRequest,
    PlateFileQueryResult,
    PlateInspectionBounds,
    PlateInspectionComponentSummary,
    PlateInspectionComponentValue,
    PlateInspectionConfidence,
    PlateInspectionDefaults,
    PlateInspectionHandlerCandidate,
    PlateInspectionImageFileSummary,
    PlateInspectionImageRecordSummary,
    PlateInspectionIngestionRoute,
    PlateInspectionIssueCode,
    PlateInspectionParseFailure,
    PlateInspectionParseSummary,
    PlateInspectionResultFileRecordSummary,
    PlateInspectionResultFilePreview,
    PlateInspectionResultFileSummary,
    PlateInspectionStatus,
    PlateInspectionSourceBindingRole,
    PlateInspectionValueSource,
    PlateInspectionWorkflowAdvice,
    PlateInspectionWorkflowScope,
    PlateInspectionWorkspacePreparation,
    PlateImageSampleRequest,
    PlateImageSampleResult,
    PlatePathInspectionRequest,
    PlatePathInspectionResult,
    PlateWorkspacePreparationOperation,
)
from openhcs.agent.path_policy import AgentPathPolicy, AgentPathPolicyError
from openhcs.agent.services.stdio import AgentStdoutRedirect
from openhcs.agent.ui_bridge_actions import PlateOperation
from openhcs.agent.ui_bridge_identities import (
    PlateManagerOrchestratorCodeDocumentIdentity,
)
from openhcs.constants.constants import AllComponents, Backend, Microscope
from openhcs.core.config import GlobalPipelineConfig, PathPlanningConfig
from openhcs.core.plate_image_inventory import (
    PlateFileKind,
    PlateFileInventory,
    PlateFileInventoryQuery,
    PlateFileRecord,
    PlateImageInventory,
    PlateImageSampler,
    PlateResultFileRecord,
    PlateResultFilePreviewReader,
    PlateResultFileInventory,
)
from openhcs.core.source_workspace_projection import (
    VirtualWorkspaceSourceProjectionAuthority,
)

if TYPE_CHECKING:
    from openhcs.core.components.parser_metaprogramming import (
        FilenameParseResult,
        FilenameParseValue,
    )
    from openhcs.microscopes.microscope_base import MicroscopeHandler
    from openhcs.microscopes.microscope_interfaces import (
        FilenameParser,
        MetadataComponentValueSet,
        MetadataHandler,
    )
    from polystore.filemanager import FileManager


class PlateInspectionText:
    """Static user-facing text for plate inspection issues and advice."""

    LOCAL_PATH_HINT = "Pass a plate folder under OPENHCS_AGENT_READ_ROOTS."
    DIRECTORY_HINT = "The inspection target must be a local directory."
    HANDLER_HINT = "Check microscope_type or make sure microscope metadata is present."
    LISTING_HINT = "The selected handler could not list image files read-only."
    NO_IMAGES_HINT = "Check that image files exist under the plate root or selected source layout."
    RESULT_FILES_AVAILABLE_HINT = (
        "This root has analysis result artifacts but no inspectable image "
        "inventory. Query result files, or inspect viewer payloads if the run "
        "streamed images."
    )
    PARSER_HINT = "The selected handler has no filename parser available for read-only parsing."
    PARSE_LIMIT_HINT = "Increase max_files_to_parse if full filename coverage is needed."
    LOW_PARSE_COVERAGE_HINT = (
        "This may be the wrong folder, microscope_type, or pattern_format. "
        "Inspect a more specific image folder or pass an explicit microscope_type."
    )
    RAW_WORKSPACE_PREPARATION_REASON = (
        "Raw microscope layouts usually need initialize_workspace before execution; "
        "this inspection did not mutate the plate."
    )
    PREPARED_WORKSPACE_REASON = (
        "OpenHCS metadata is already present; read-only inspection did not need "
        "workspace preparation."
    )


class PlateInspectionFileManagerFactory:
    """Create the FileManager expected by microscope handlers."""

    def create(self) -> "FileManager":
        from polystore.base import ensure_storage_registry, storage_registry
        from polystore.filemanager import FileManager

        ensure_storage_registry()
        return FileManager(storage_registry)


@dataclass(frozen=True, slots=True)
class PlateInspectionContext:
    """Resolved plate IO context shared by inspection, sampling, and streaming."""

    plate_path: Path
    filemanager: "FileManager"
    handler: "MicroscopeHandler"
    parser: "FilenameParser | None"
    warnings: tuple[AgentWarning, ...] = ()

    @property
    def microscope_type(self) -> str:
        return self.handler.microscope_type


class PlateInspectionBackendProjection:
    """Project backend declarations to stable string names."""

    @staticmethod
    def names(backends: Sequence["Backend"]) -> tuple[str, ...]:
        return tuple(backend.value for backend in backends)


class PlateInspectionHandlerCandidateProjection:
    """Recover format-specific candidates from registered parser/detector owners."""

    @classmethod
    def candidates(
        cls,
        *,
        requested_microscope_type: str,
        selected_handler: "MicroscopeHandler",
        plate_path: Path,
        filemanager: "FileManager",
        max_files_to_parse: int,
    ) -> tuple[PlateInspectionHandlerCandidate, ...]:
        from openhcs.microscopes.microscope_base import (
            MicroscopeHandler,
            MicroscopeSourceSelectionRole,
        )

        if requested_microscope_type != PlateInspectionDefaults.MICROSCOPE_AUTO:
            return ()
        if (
            type(selected_handler).source_selection_role()
            is not MicroscopeSourceSelectionRole.BROAD_STRUCTURED_STORE
        ):
            return ()

        try:
            source_paths = tuple(
                str(path)
                for path in selected_handler.metadata_handler.physical_source_paths(
                    plate_path
                )[:max_files_to_parse]
            )
        except Exception:
            return ()
        if not source_paths:
            return ()

        candidates: list[PlateInspectionHandlerCandidate] = []
        for handler_type in MicroscopeHandler.__registry__.values():
            if (
                handler_type.source_selection_role()
                is not MicroscopeSourceSelectionRole.FORMAT_SPECIFIC
            ):
                continue
            candidate = cls._candidate(
                handler_type=handler_type,
                source_paths=source_paths,
                plate_path=plate_path,
                filemanager=filemanager,
            )
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    @staticmethod
    def supports_explicit_incomplete_export(
        candidate: PlateInspectionHandlerCandidate,
    ) -> bool:
        """Project the candidate handler's declaration-owned subset policy."""

        from openhcs.microscopes.microscope_base import MicroscopeHandler

        return MicroscopeHandler.__registry__[
            candidate.microscope_type
        ].supports_explicit_incomplete_export()

    @staticmethod
    def _candidate(
        *,
        handler_type: type["MicroscopeHandler"],
        source_paths: tuple[str, ...],
        plate_path: Path,
        filemanager: "FileManager",
    ) -> PlateInspectionHandlerCandidate | None:
        try:
            handler = handler_type.create(filemanager=filemanager)
            handler.plate_folder = plate_path
            parser = handler.parser
            if parser is None:
                return None
            recognized_file_count = sum(
                bool(parser.can_parse(Path(source_path).name))
                for source_path in source_paths
            )
            root_dir = handler.root_dir
        except Exception:
            return None

        tested_file_count = len(source_paths)
        recognizes_all = recognized_file_count == tested_file_count
        expected_root = (plate_path / root_dir).resolve()
        files_under_expected_root = all(
            Path(source_path).resolve().is_relative_to(expected_root)
            for source_path in source_paths
        )
        if not recognizes_all or not files_under_expected_root:
            return None

        metadata_detected = False
        metadata_file_path: str | None = None
        metadata_diagnostic: str | None = None
        try:
            metadata_detected = bool(handler_type.detect(plate_path, filemanager))
        except Exception as exc:
            metadata_diagnostic = str(exc)
        try:
            metadata_file_path = str(
                handler.metadata_handler.find_metadata_file(plate_path)
            )
        except Exception as exc:
            if metadata_diagnostic is None:
                metadata_diagnostic = str(exc)

        return PlateInspectionHandlerCandidate(
            microscope_type=handler.microscope_type,
            handler_class=type(handler).__name__,
            parser_class=type(parser).__name__,
            root_dir=root_dir,
            tested_file_count=tested_file_count,
            recognized_file_count=recognized_file_count,
            recognizes_all_tested_files=recognizes_all,
            files_under_expected_root=files_under_expected_root,
            metadata_detected=metadata_detected,
            metadata_file_path=metadata_file_path,
            metadata_diagnostic=metadata_diagnostic,
        )


class PlateInspectionFileQueryProjection:
    """Project shared plate file inventory queries into agent DTOs."""

    @staticmethod
    def record(
        record: PlateFileRecord,
        *,
        include_preview: bool,
        max_preview_lines: int,
        max_preview_bytes: int,
    ) -> PlateFileQueryRecordSummary:
        return PlateFileQueryRecordSummary(
            kind=record.kind,
            key=record.key,
            metadata=dict(record.metadata),
            virtual_path=record.virtual_path,
            full_virtual_path=record.full_virtual_path,
            source_path=record.source_path,
            relative_path=record.relative_path,
            full_path=record.full_path,
            file_format=None if record.file_format is None else record.file_format.name,
            preview=PlateInspectionFileQueryProjection.preview(
                record,
                include_preview=include_preview,
                max_preview_lines=max_preview_lines,
                max_preview_bytes=max_preview_bytes,
            ),
        )

    @staticmethod
    def preview(
        record: PlateFileRecord,
        *,
        include_preview: bool,
        max_preview_lines: int,
        max_preview_bytes: int,
    ) -> PlateInspectionResultFilePreview | None:
        if (
            not include_preview
            or record.kind is not PlateFileKind.RESULT
            or record.relative_path is None
            or record.full_path is None
            or record.file_format is None
        ):
            return None
        preview = PlateResultFilePreviewReader.preview(
            PlateResultFileRecord(
                relative_path=record.relative_path,
                full_path=record.full_path,
                file_format=record.file_format,
                metadata=record.metadata,
            ),
            max_lines=max_preview_lines,
            max_bytes=max_preview_bytes,
        )
        if preview is None:
            return None
        return PlateInspectionResultFilePreview(
            text_lines=preview.text_lines,
            csv_columns=preview.csv_columns,
            csv_rows=tuple(dict(row) for row in preview.csv_rows),
            roi_count=preview.roi_count,
            roi_member_count=preview.roi_member_count,
            roi_duplicate_member_count=preview.roi_duplicate_member_count,
            roi_area_min=preview.roi_area_min,
            roi_area_max=preview.roi_area_max,
            roi_area_mean=preview.roi_area_mean,
            roi_examples=tuple(dict(row) for row in preview.roi_examples),
            truncated=preview.truncated,
            omitted_reason=preview.omitted_reason,
        )


@dataclass(frozen=True, slots=True)
class PlateInspectionComponentEntrySet:
    """Nominal value set for one HCS component axis."""

    component: AllComponents
    values: tuple[PlateInspectionComponentValue, ...] = ()

    @classmethod
    def empty(
        cls,
        component: AllComponents,
    ) -> "PlateInspectionComponentEntrySet":
        return cls(component=component)

    @classmethod
    def parsed(
        cls,
        component: AllComponents,
        keys: tuple[str, ...],
    ) -> "PlateInspectionComponentEntrySet":
        return cls(
            component=component,
            values=tuple(
                PlateInspectionComponentValue(
                    key=key,
                    declared_in_metadata=False,
                    observed_in_filenames=True,
                )
                for key in keys
            ),
        )

    @classmethod
    def metadata(
        cls,
        component: AllComponents,
        values: Mapping[str, str | None] | None,
    ) -> "PlateInspectionComponentEntrySet":
        if values is None:
            return cls.empty(component)
        return cls(
            component=component,
            values=tuple(
                PlateInspectionComponentValue(
                    key=str(key),
                    label=None if value is None else str(value),
                    declared_in_metadata=True,
                    observed_in_filenames=False,
                )
                for key, value in values.items()
            ),
        )

    def keys(self) -> tuple[str, ...]:
        return tuple(value.key for value in self.values)

    def contains(self, key: str) -> bool:
        return any(value.key == key for value in self.values)

    def label_for(self, key: str) -> str | None:
        for value in self.values:
            if value.key == key:
                return value.label
        return None


@dataclass(frozen=True, slots=True)
class PlateInspectionComponentCollection:
    """Nominal collection of all inspected HCS component value sets."""

    entries: tuple[PlateInspectionComponentEntrySet, ...]

    @classmethod
    def empty(cls) -> "PlateInspectionComponentCollection":
        return cls(
            tuple(
                PlateInspectionComponentEntrySet.empty(component)
                for component in AllComponents
            )
        )

    def for_component(
        self,
        component: AllComponents,
    ) -> PlateInspectionComponentEntrySet:
        for entry in self.entries:
            if entry.component is component:
                return entry
        return PlateInspectionComponentEntrySet.empty(component)


class PlateInspectionComponentMetadataAccess:
    """Read component maps through the microscope metadata handler contract."""

    def collect(
        self,
        metadata_handler: "MetadataHandler",
        plate_path: Path,
    ) -> PlateInspectionComponentCollection:
        return PlateInspectionMetadataComponentProjection.from_value_set(
            metadata_handler.component_value_set(plate_path)
        )


class PlateInspectionMetadataComponentProjection:
    """Project the microscope metadata value set into inspection component entries."""

    @staticmethod
    def from_value_set(
        component_values: "MetadataComponentValueSet",
    ) -> PlateInspectionComponentCollection:
        return PlateInspectionComponentCollection(
            tuple(
                PlateInspectionComponentEntrySet.metadata(
                    component,
                    component_values.values_for(component),
                )
                for component in AllComponents
            )
        )


@dataclass(slots=True)
class PlateInspectionMutableParsedComponents:
    """Mutable accumulator before parsed component values are frozen."""

    values: dict[AllComponents, set[str]] = field(
        default_factory=lambda: {component: set() for component in AllComponents}
    )

    def add_parsed_filename_values(self, parsed: "FilenameParseResult") -> None:
        for component in AllComponents:
            self._add_optional(
                self.values[component],
                parsed.get(component.value),
            )

    @staticmethod
    def _add_optional(target: set[str], value: "FilenameParseValue") -> None:
        if value is not None:
            target.add(str(value))

    def freeze(self) -> PlateInspectionComponentCollection:
        return PlateInspectionComponentCollection(
            tuple(
                PlateInspectionComponentEntrySet.parsed(
                    component,
                    tuple(
                        sorted(
                            self.values[component],
                            key=PlateInspectionComponentSort.key,
                        )
                    ),
                )
                for component in AllComponents
            )
        )


@dataclass(frozen=True, slots=True)
class PlateInspectionParsedFileSet:
    """Filename parse coverage and parsed component values."""

    summary: PlateInspectionParseSummary
    components: PlateInspectionComponentCollection


class PlateInspectionFilenameParser:
    """Parse bounded image filenames through the selected microscope parser."""

    def parse(
        self,
        *,
        parser: "FilenameParser | None",
        image_files: tuple[str, ...],
        bounds: PlateInspectionBounds,
    ) -> PlateInspectionParsedFileSet:
        parsed_components = PlateInspectionMutableParsedComponents()
        if parser is None:
            return PlateInspectionParsedFileSet(
                summary=PlateInspectionParseSummary(
                    skipped_file_count=len(image_files),
                ),
                components=parsed_components.freeze(),
            )

        parse_limit = min(len(image_files), bounds.max_files_to_parse)
        failures: list[PlateInspectionParseFailure] = []
        parsed_count = 0
        failed_count = 0

        for filename in image_files[:parse_limit]:
            try:
                parsed = parser.parse_filename(Path(filename).name)
            except Exception as exc:
                failed_count += 1
                self._append_failure(
                    failures,
                    bounds,
                    filename,
                    f"{type(exc).__name__}: {exc}",
                )
                continue
            if parsed is None:
                failed_count += 1
                self._append_failure(failures, bounds, filename, "not parsed")
                continue
            parsed_count += 1
            parsed_components.add_parsed_filename_values(parsed)

        return PlateInspectionParsedFileSet(
            summary=PlateInspectionParseSummary(
                attempted_file_count=parse_limit,
                skipped_file_count=len(image_files) - parse_limit,
                parsed_file_count=parsed_count,
                failed_file_count=failed_count,
                failure_samples=tuple(failures),
                truncated_failure_count=max(0, failed_count - len(failures)),
            ),
            components=parsed_components.freeze(),
        )

    @staticmethod
    def _append_failure(
        failures: list[PlateInspectionParseFailure],
        bounds: PlateInspectionBounds,
        filename: str,
        reason: str,
    ) -> None:
        if len(failures) < bounds.max_parse_failure_samples:
            failures.append(
                PlateInspectionParseFailure(
                    filename=filename,
                    reason=reason,
                )
            )


class PlateInspectionComponentSummaryBuilder:
    """Merge metadata component maps with values observed in filenames."""

    def build(
        self,
        *,
        metadata_values: PlateInspectionComponentCollection,
        parsed_values: PlateInspectionComponentCollection,
        bounds: PlateInspectionBounds,
    ) -> tuple[PlateInspectionComponentSummary, ...]:
        summaries: list[PlateInspectionComponentSummary] = []
        for component in AllComponents:
            metadata_entries = metadata_values.for_component(component)
            parsed_entries = parsed_values.for_component(component)
            keys = tuple(
                sorted(
                    set(metadata_entries.keys()) | set(parsed_entries.keys()),
                    key=PlateInspectionComponentSort.key,
                )
            )
            values = tuple(
                PlateInspectionComponentValue(
                    key=key,
                    label=metadata_entries.label_for(key),
                    declared_in_metadata=metadata_entries.contains(key),
                    observed_in_filenames=parsed_entries.contains(key),
                )
                for key in keys[: bounds.max_component_values]
            )
            summaries.append(
                PlateInspectionComponentSummary(
                    component=component,
                    source=self._source(metadata_entries, parsed_entries),
                    count=len(keys),
                    values=values,
                    truncated_value_count=max(0, len(keys) - len(values)),
                )
            )
        return tuple(summaries)

    @staticmethod
    def _source(
        metadata_entries: PlateInspectionComponentEntrySet,
        parsed_entries: PlateInspectionComponentEntrySet,
    ) -> PlateInspectionValueSource:
        if metadata_entries.values and parsed_entries.values:
            return PlateInspectionValueSource.METADATA_AND_PARSED_FILENAMES
        if metadata_entries.values:
            return PlateInspectionValueSource.METADATA
        if parsed_entries.values:
            return PlateInspectionValueSource.PARSED_FILENAMES
        return PlateInspectionValueSource.UNAVAILABLE


class PlateInspectionComponentSort:
    """Stable sort key for component values."""

    @staticmethod
    def key(value: str) -> tuple[int, int | str]:
        if value.isdigit():
            return (0, int(value))
        return (1, value)


@dataclass(frozen=True, slots=True)
class PlateInspectionParseCoverage:
    """Parsed filename coverage for confidence and agent-facing warnings."""

    attempted_file_count: int
    parsed_file_count: int

    LOW_COVERAGE_MIN_ATTEMPTED = 10
    LOW_COVERAGE_RATIO = 0.5

    @classmethod
    def from_summary(
        cls,
        summary: PlateInspectionParseSummary,
    ) -> "PlateInspectionParseCoverage":
        return cls(
            attempted_file_count=summary.attempted_file_count,
            parsed_file_count=summary.parsed_file_count,
        )

    @property
    def ratio(self) -> float:
        if self.attempted_file_count <= 0:
            return 0.0
        return self.parsed_file_count / self.attempted_file_count

    @property
    def is_low(self) -> bool:
        return (
            self.attempted_file_count >= self.LOW_COVERAGE_MIN_ATTEMPTED
            and self.ratio < self.LOW_COVERAGE_RATIO
        )

    @property
    def percent(self) -> float:
        return self.ratio * 100.0


class PlateInspectionStatusPolicy:
    """Derive status and confidence from inspection evidence."""

    def status(
        self,
        *,
        errors: tuple[AgentError, ...],
        warnings: tuple[AgentWarning, ...],
    ) -> PlateInspectionStatus:
        if errors:
            return PlateInspectionStatus.ERROR
        if warnings:
            return PlateInspectionStatus.PARTIAL
        return PlateInspectionStatus.OK

    def confidence(
        self,
        *,
        errors: tuple[AgentError, ...],
        image_file_count: int,
        parse_coverage: PlateInspectionParseCoverage,
        detected_microscope_type: str | None,
    ) -> PlateInspectionConfidence:
        if errors or detected_microscope_type is None:
            return PlateInspectionConfidence.NONE
        if not image_file_count:
            return PlateInspectionConfidence.LOW
        if parse_coverage.parsed_file_count <= 0:
            return PlateInspectionConfidence.LOW
        if parse_coverage.is_low:
            return PlateInspectionConfidence.LOW
        if parse_coverage.ratio < 1.0:
            return PlateInspectionConfidence.MEDIUM
        return PlateInspectionConfidence.HIGH


class PlateInspectionWorkspacePreparationPolicy:
    """Describe mutating workspace preparation without performing it."""

    PREPARED_MICROSCOPE_TYPE = Microscope.OPENHCS.value

    def for_microscope(
        self,
        microscope_type: str | None,
    ) -> PlateInspectionWorkspacePreparation:
        if microscope_type == self.PREPARED_MICROSCOPE_TYPE:
            return PlateInspectionWorkspacePreparation(
                read_only_inspection=True,
                required_before_execution=False,
                operation=PlateWorkspacePreparationOperation.NONE,
                reason=PlateInspectionText.PREPARED_WORKSPACE_REASON,
            )
        return PlateInspectionWorkspacePreparation(
            read_only_inspection=True,
            required_before_execution=microscope_type is not None,
            operation=(
                PlateWorkspacePreparationOperation.INITIALIZE_WORKSPACE
                if microscope_type is not None
                else PlateWorkspacePreparationOperation.NONE
            ),
            reason=(
                PlateInspectionText.RAW_WORKSPACE_PREPARATION_REASON
                if microscope_type is not None
                else None
            ),
        )


class PlateInspectionWorkflowAdvicePolicy:
    """Route inspection evidence without replacing the selected source owner."""

    KNOWLEDGE_QUERY = "source model CZI Bio-Formats source bindings"

    @classmethod
    def for_handler(
        cls,
        handler: "MicroscopeHandler",
        *,
        format_specific_candidates: tuple[
            PlateInspectionHandlerCandidate, ...
        ] = (),
        requested_microscope_type: str = PlateInspectionDefaults.MICROSCOPE_AUTO,
    ) -> PlateInspectionWorkflowAdvice:
        from openhcs.microscopes.microscope_base import (
            MicroscopeSourceSelectionRole,
        )

        ingestion_owner = handler.microscope_type
        projects_bindings = type(handler).projects_declared_source_bindings()
        selection_role = type(handler).source_selection_role()
        supported_partial_candidates = tuple(
            candidate
            for candidate in format_specific_candidates
            if PlateInspectionHandlerCandidateProjection.supports_explicit_incomplete_export(
                candidate
            )
        )
        unsupported_partial_candidates = tuple(
            candidate
            for candidate in format_specific_candidates
            if candidate not in supported_partial_candidates
        )
        probable_native_owners = tuple(
            candidate.microscope_type for candidate in supported_partial_candidates
        )
        if unsupported_partial_candidates and not supported_partial_candidates:
            ingestion_route = PlateInspectionIngestionRoute.SOURCE_BINDINGS_HANDLER
            ingestion_owner = Microscope.SOURCE_BINDINGS.value
            source_binding_role = PlateInspectionSourceBindingRole.INGESTION_OWNER
            evidence = "; ".join(
                cls._candidate_evidence(candidate)
                for candidate in unsupported_partial_candidates
            )
            message = (
                "A registered format-specific parser recognizes these filenames, "
                f"but its owner requires the complete native detection contract: {evidence}. "
                "Do not explicitly select that native microscope type while its "
                "required metadata is absent. Obtain the complete vendor export for "
                "native plate semantics. If these are intentionally loose, ordinary "
                "image files, declare their file selection and Well/Site/Channel/Z/Time "
                "identity in SourceBindingsConfig so SourceBindingsHandler owns "
                "ingestion instead of accepting the broad decoder's inferred sample "
                "layout."
            )
        elif supported_partial_candidates:
            ingestion_route = PlateInspectionIngestionRoute.DETECTED_HANDLER
            source_binding_role = (
                PlateInspectionSourceBindingRole.SEMANTIC_SELECTION
                if projects_bindings
                else PlateInspectionSourceBindingRole.NOT_PROJECTED_BY_HANDLER
            )
            evidence = "; ".join(
                cls._candidate_evidence(candidate)
                for candidate in supported_partial_candidates
            )
            unsupported_note = (
                " Other recognized filename candidates require their complete "
                "detection contract and are not eligible for explicit partial "
                "selection: "
                + "; ".join(
                    cls._candidate_evidence(candidate)
                    for candidate in unsupported_partial_candidates
                )
                + "."
                if unsupported_partial_candidates
                else ""
            )
            message = (
                f"Auto-detection selected the broad {ingestion_owner} handler, but "
                f"registered format-specific candidate evidence is stronger: {evidence}. "
                "This is a probable incomplete vendor export, not proof that the "
                "broad decoder is the better semantic owner. Prefer obtaining the "
                "complete vendor export so native auto-detection and full plate "
                "metadata work. For a knowingly partial analysis, keep files under "
                "the candidate's declared root_dir and explicitly select that "
                "microscope_type; expect missing metadata-derived fields. Source "
                "bindings may name/select planes only when the selected handler "
                "projects them, and are not a replacement vendor decoder."
                f"{unsupported_note}"
            )
        elif (
            selection_role
            is MicroscopeSourceSelectionRole.DECLARED_FILE_FALLBACK
        ):
            ingestion_route = PlateInspectionIngestionRoute.SOURCE_BINDINGS_HANDLER
            source_binding_role = PlateInspectionSourceBindingRole.INGESTION_OWNER
            message = (
                "The SourceBindingsHandler owns ingestion for this arbitrary image "
                "folder. Keep file selection, metadata extraction, and semantic "
                "aliases in SourceBindingsConfig."
            )
        else:
            ingestion_route = PlateInspectionIngestionRoute.DETECTED_HANDLER
            source_binding_role = (
                PlateInspectionSourceBindingRole.SEMANTIC_SELECTION
                if projects_bindings
                else PlateInspectionSourceBindingRole.NOT_PROJECTED_BY_HANDLER
            )
            binding_message = (
                "SourceBindingsConfig may name or select the planes emitted by this "
                "handler; it does not open or replace the detected store."
                if projects_bindings
                else "This handler does not project declared source bindings."
            )
            selection_message = type(handler).source_selection_guidance()
            initialization_message = (
                f"initialize with explicit microscope_type={ingestion_owner!r}"
                if requested_microscope_type
                != PlateInspectionDefaults.MICROSCOPE_AUTO
                else "initialize with auto-detection"
            )
            message = (
                f"Keep the detected {ingestion_owner} handler as the ingestion "
                "owner. For a visible workflow, add the plate directory through "
                f"the PlateManager code document and {initialization_message}. "
                f"{selection_message} {binding_message}"
            )
        return cls._advice(
            ingestion_route=ingestion_route,
            ingestion_owner=ingestion_owner,
            source_binding_role=source_binding_role,
            probable_native_ingestion_owners=probable_native_owners,
            message=message,
        )

    @staticmethod
    def _candidate_evidence(candidate: PlateInspectionHandlerCandidate) -> str:
        detection = (
            f"metadata detected at {candidate.metadata_file_path}"
            if candidate.metadata_detected
            else "metadata detection unsatisfied"
        )
        diagnostic = (
            ""
            if candidate.metadata_diagnostic is None
            else f" ({candidate.metadata_diagnostic})"
        )
        return (
            f"{candidate.microscope_type} parser {candidate.parser_class} recognized "
            f"{candidate.recognized_file_count}/{candidate.tested_file_count} source "
            f"filenames under root_dir={candidate.root_dir!r}; {detection}{diagnostic}"
        )

    @classmethod
    def unresolved(cls) -> PlateInspectionWorkflowAdvice:
        return cls._advice(
            ingestion_route=PlateInspectionIngestionRoute.UNRESOLVED,
            ingestion_owner=None,
            source_binding_role=PlateInspectionSourceBindingRole.UNRESOLVED,
            message=(
                "No ingestion owner was selected. For an arbitrary TIFF, PNG, or "
                "similar folder, declare a non-empty SourceBindingsConfig so the "
                "SourceBindingsHandler can own ingestion. For CZI, OME, or another "
                "structured microscopy store, repair or enable its decoder instead "
                "of routing it through the arbitrary-file fallback."
            ),
        )

    @classmethod
    def _advice(
        cls,
        *,
        ingestion_route: PlateInspectionIngestionRoute,
        ingestion_owner: str | None,
        source_binding_role: PlateInspectionSourceBindingRole,
        probable_native_ingestion_owners: tuple[str, ...] = (),
        message: str,
    ) -> PlateInspectionWorkflowAdvice:
        return PlateInspectionWorkflowAdvice(
            workflow_scope=PlateInspectionWorkflowScope.DIAGNOSTIC,
            ingestion_route=ingestion_route,
            ingestion_owner=ingestion_owner,
            source_binding_role=source_binding_role,
            ui_code_document_id=(
                PlateManagerOrchestratorCodeDocumentIdentity.require_value()
            ),
            ui_operation=PlateOperation.INIT.value,
            knowledge_query=cls.KNOWLEDGE_QUERY,
            probable_native_ingestion_owners=probable_native_ingestion_owners,
            message=message,
        )


class PlateInspectionPathPlanningConfigProvider:
    """Resolve the saved global path-planning config used for result discovery."""

    def path_planning_config(self) -> PathPlanningConfig:
        from objectstate import get_current_global_config

        global_config = get_current_global_config(
            GlobalPipelineConfig,
            use_live=False,
        )
        if global_config is None:
            return PathPlanningConfig()
        return global_config.path_planning_config


class PlateInspectionService:
    """Read local plate folders through OpenHCS microscope handlers."""

    def __init__(
        self,
        path_policy: AgentPathPolicy | None = None,
        filemanager_factory: PlateInspectionFileManagerFactory | None = None,
        path_config_provider: PlateInspectionPathPlanningConfigProvider | None = None,
    ) -> None:
        self._path_policy = path_policy or AgentPathPolicy.from_environment()
        self._filemanager_factory = filemanager_factory or PlateInspectionFileManagerFactory()
        self._path_config_provider = (
            path_config_provider or PlateInspectionPathPlanningConfigProvider()
        )
        self._metadata_access = PlateInspectionComponentMetadataAccess()
        self._filename_parser = PlateInspectionFilenameParser()
        self._component_builder = PlateInspectionComponentSummaryBuilder()
        self._status_policy = PlateInspectionStatusPolicy()
        self._preparation_policy = PlateInspectionWorkspacePreparationPolicy()
        self._workflow_advice_policy = PlateInspectionWorkflowAdvicePolicy()

    def inspect(
        self,
        request: PlatePathInspectionRequest,
    ) -> PlatePathInspectionResult:
        with AgentStdoutRedirect.to_stderr():
            return self._inspect(request)

    def query_files(
        self,
        request: PlateFileQueryRequest,
    ) -> PlateFileQueryResult:
        with AgentStdoutRedirect.to_stderr():
            return self._query_files(request)

    def sample_image(
        self,
        request: PlateImageSampleRequest,
    ) -> PlateImageSampleResult:
        with AgentStdoutRedirect.to_stderr():
            return self._sample_image(request)

    def open_context(
        self,
        request: PlatePathInspectionRequest,
    ) -> tuple[PlateInspectionContext | None, tuple[AgentError, ...], tuple[AgentWarning, ...]]:
        """Resolve the plate handler/filemanager context without querying files."""
        with AgentStdoutRedirect.to_stderr():
            return self._open_context(request)

    def resolve_plate_path(self, plate_path: str) -> tuple[Path | None, tuple[AgentError, ...]]:
        """Resolve and validate a local plate path without creating a handler."""
        with AgentStdoutRedirect.to_stderr():
            return self._resolve_plate_path(plate_path)

    def _open_context(
        self,
        request: PlatePathInspectionRequest,
    ) -> tuple[PlateInspectionContext | None, tuple[AgentError, ...], tuple[AgentWarning, ...]]:
        plate_path, path_errors = self._resolve_plate_path(request.plate_path)
        if path_errors:
            return None, path_errors, ()
        if plate_path is None:
            raise RuntimeError("Plate path resolution returned no path and no error.")

        filemanager = self._filemanager_factory.create()
        try:
            handler = self._create_handler(request, plate_path, filemanager)
        except Exception as exc:
            return None, (
                AgentError.from_exception(
                    PlateInspectionIssueCode.HANDLER_DETECTION_FAILED.value,
                    exc,
                    hint=PlateInspectionText.HANDLER_HINT,
                    path=str(plate_path),
                ),
            ), ()

        warnings: list[AgentWarning] = []
        parser = self._parser(handler, warnings, warn=True)
        return PlateInspectionContext(
            plate_path=plate_path,
            filemanager=filemanager,
            handler=handler,
            parser=parser,
            warnings=tuple(warnings),
        ), (), tuple(warnings)

    def _resolve_plate_path(self, plate_path: str) -> tuple[Path | None, tuple[AgentError, ...]]:
        try:
            resolved_path = self._path_policy.assert_readable(plate_path)
            if not resolved_path.is_dir():
                return None, (
                    AgentError(
                        code=PlateInspectionIssueCode.PATH_NOT_DIRECTORY.value,
                        message=f"Plate path is not a directory: {resolved_path}",
                        hint=PlateInspectionText.DIRECTORY_HINT,
                        path=str(resolved_path),
                    ),
                )
        except AgentPathPolicyError as exc:
            return None, (
                AgentError.from_exception(
                    PlateInspectionIssueCode.PATH_POLICY_REJECTED.value,
                    exc,
                    hint=PlateInspectionText.LOCAL_PATH_HINT,
                    path=plate_path,
                ),
            )
        return resolved_path, ()

    def file_inventory(
        self,
        context: PlateInspectionContext,
        *,
        kind: PlateFileKind | None,
    ) -> tuple[PlateFileInventory, tuple[AgentWarning, ...]]:
        """Build a file inventory from an already resolved plate context."""
        warnings = list(context.warnings)
        inventory = self._plate_file_inventory_for_query(
            context.handler,
            context.plate_path,
            context.parser,
            context.filemanager,
            PlateFileInventoryQuery.kind_from_value(kind),
            warnings,
        )
        return inventory, tuple(warnings)

    def _query_files(
        self,
        request: PlateFileQueryRequest,
    ) -> PlateFileQueryResult:
        try:
            plate_path = self._path_policy.assert_readable(request.plate_path)
            if not plate_path.is_dir():
                return self._query_files_error(
                    request,
                    AgentError(
                        code=PlateInspectionIssueCode.PATH_NOT_DIRECTORY.value,
                        message=f"Plate path is not a directory: {plate_path}",
                        hint=PlateInspectionText.DIRECTORY_HINT,
                        path=str(plate_path),
                    ),
                    plate_path=plate_path,
                )
        except AgentPathPolicyError as exc:
            return self._query_files_error(
                request,
                AgentError.from_exception(
                    PlateInspectionIssueCode.PATH_POLICY_REJECTED.value,
                    exc,
                    hint=PlateInspectionText.LOCAL_PATH_HINT,
                    path=request.plate_path,
                ),
            )

        query_kind = PlateFileInventoryQuery.kind_from_value(request.kind)
        if (
            query_kind is PlateFileKind.RESULT
            and request.microscope_type == PlateInspectionDefaults.MICROSCOPE_AUTO
            and request.pattern_format is None
            and request.well is None
        ):
            result_only = self._result_only_query_files(
                request=request,
                plate_path=plate_path,
                handler_error=None,
            )
            if result_only is not None and result_only.total_count:
                return result_only

        filemanager = self._filemanager_factory.create()
        if (
            query_kind is PlateFileKind.IMAGE
            and request.microscope_type == PlateInspectionDefaults.MICROSCOPE_AUTO
            and request.pattern_format is None
            and request.well is None
        ):
            result_only = self._result_only_query_files(
                request=request,
                plate_path=plate_path,
                handler_error=None,
            )
            if (
                result_only is not None
                and self._has_disk_image_files(filemanager, plate_path) is False
            ):
                return result_only

        try:
            handler = self._create_handler(
                PlatePathInspectionRequest(
                    plate_path=str(plate_path),
                    microscope_type=request.microscope_type,
                    pattern_format=request.pattern_format,
                ),
                plate_path,
                filemanager,
            )
        except Exception as exc:
            if query_kind in (PlateFileKind.RESULT, None):
                result_only = self._result_only_query_files(
                    request=request,
                    plate_path=plate_path,
                    handler_error=exc,
                )
                if result_only is not None:
                    return result_only
            return self._query_files_error(
                request,
                AgentError.from_exception(
                    PlateInspectionIssueCode.HANDLER_DETECTION_FAILED.value,
                    exc,
                    hint=PlateInspectionText.HANDLER_HINT,
                    path=str(plate_path),
                ),
                plate_path=plate_path,
            )

        warnings: list[AgentWarning] = []
        parser = self._parser(
            handler,
            warnings,
            warn=query_kind is not PlateFileKind.RESULT or request.well is not None,
        )
        file_inventory = self._plate_file_inventory_for_query(
            handler,
            plate_path,
            parser,
            filemanager,
            query_kind,
            warnings,
        )
        return self._query_files_from_inventory(
            request=request,
            plate_path=plate_path,
            file_inventory=file_inventory,
            detected_microscope_type=handler.microscope_type,
            handler_class=type(handler).__name__,
            parser_class=None if parser is None else type(parser).__name__,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _query_files_from_inventory(
        *,
        request: PlateFileQueryRequest,
        plate_path: Path,
        file_inventory: PlateFileInventory,
        detected_microscope_type: str | None,
        handler_class: str | None,
        parser_class: str | None,
        warnings: tuple[AgentWarning, ...],
    ) -> PlateFileQueryResult:
        query_result = file_inventory.query_files(
            PlateFileInventoryQuery(
                kinds=PlateFileInventoryQuery.kinds_for(request.kind),
                path_contains=request.path_contains,
                well=request.well,
                offset=request.offset,
                limit=request.limit,
            )
        )
        return PlateFileQueryResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path),
            requested_microscope_type=request.microscope_type,
            detected_microscope_type=detected_microscope_type,
            handler_class=handler_class,
            parser_class=parser_class,
            total_count=query_result.total_count,
            returned_count=len(query_result.records),
            offset=query_result.offset,
            limit=query_result.limit,
            truncated_count=query_result.truncated_count,
            records=tuple(
                PlateInspectionFileQueryProjection.record(
                    record,
                    include_preview=request.include_previews,
                    max_preview_lines=request.max_preview_lines,
                    max_preview_bytes=request.max_preview_bytes,
                )
                for record in query_result.records
            ),
            warnings=warnings,
        )

    def _sample_image(
        self,
        request: PlateImageSampleRequest,
    ) -> PlateImageSampleResult:
        try:
            plate_path = self._path_policy.assert_readable(request.plate_path)
            if not plate_path.is_dir():
                return self._sample_image_error(
                    request,
                    AgentError(
                        code=PlateInspectionIssueCode.PATH_NOT_DIRECTORY.value,
                        message=f"Plate path is not a directory: {plate_path}",
                        hint=PlateInspectionText.DIRECTORY_HINT,
                        path=str(plate_path),
                    ),
                    plate_path=plate_path,
                )
        except AgentPathPolicyError as exc:
            return self._sample_image_error(
                request,
                AgentError.from_exception(
                    PlateInspectionIssueCode.PATH_POLICY_REJECTED.value,
                    exc,
                    hint=PlateInspectionText.LOCAL_PATH_HINT,
                    path=request.plate_path,
                ),
            )

        filemanager = self._filemanager_factory.create()
        try:
            handler = self._create_handler(
                PlatePathInspectionRequest(
                    plate_path=str(plate_path),
                    microscope_type=request.microscope_type,
                    pattern_format=request.pattern_format,
                ),
                plate_path,
                filemanager,
            )
            warnings: list[AgentWarning] = []
            parser = self._parser(handler, warnings, warn=False)
            inventory = self._plate_file_inventory_for_query(
                handler,
                plate_path,
                parser,
                filemanager,
                PlateFileKind.IMAGE,
                warnings,
            )
            record = inventory.require_image_record(request.image_path)
            source_path = self._path_policy.assert_readable(record.source_path)
            sample = PlateImageSampler.from_storage_sample(
                record,
                filemanager.sample(
                    record.full_virtual_path,
                    record.backend,
                    ImageSamplingRequest(
                        origin_yx=(request.y, request.x),
                        shape_yx=(request.height, request.width),
                        resolution_index=request.resolution_index,
                        max_auto_resolution_size=request.max_auto_resolution_size,
                    ),
                ),
                include_array_values=request.include_array_values,
                max_array_elements=request.max_array_elements,
            )
        except Exception as exc:
            return self._sample_image_error(
                request,
                AgentError.from_exception(
                    "plate_image_sample_failed",
                    exc,
                    hint=(
                        "Use openhcs_inspect_plate_path to list virtual image "
                        "names, then pass one to openhcs_sample_plate_image."
                    ),
                    path=request.plate_path,
                ),
                plate_path=plate_path,
            )

        return PlateImageSampleResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path),
            requested_image_path=request.image_path,
            virtual_path=record.virtual_path,
            full_virtual_path=record.full_virtual_path,
            source_path=str(source_path),
            source_metadata=dict(record.metadata),
            shape=sample.shape,
            resolution_shape=sample.resolution_shape,
            dtype=sample.dtype,
            minimum=sample.minimum,
            maximum=sample.maximum,
            mean=sample.mean,
            requested_resolution_index=request.resolution_index,
            selected_resolution_index=sample.selected_resolution_index,
            resolution_count=sample.resolution_count,
            downsample_yx=sample.downsample_yx,
            statistics_scope=sample.statistics_scope,
            sample_origin_yx=sample.sample_origin_yx,
            sample_shape=sample.sample_shape,
            sample_included=sample.sample_included,
            sample_values=sample.sample_values,
            sample_omitted_reason=sample.sample_omitted_reason,
            warnings=tuple(warnings),
        )

    def _path_planning_config(self) -> PathPlanningConfig:
        return self._path_config_provider.path_planning_config()

    def _result_only_query_files(
        self,
        *,
        request: PlateFileQueryRequest,
        plate_path: Path,
        handler_error: Exception | None,
    ) -> PlateFileQueryResult | None:
        result_inventory = PlateResultFileInventory.from_configured_output_root(
            plate_path=plate_path,
            path_config=self._path_planning_config(),
        )
        if not result_inventory.records:
            return None

        warnings: list[AgentWarning] = []
        if handler_error is not None:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.HANDLER_DETECTION_FAILED.value,
                    message=(
                        "Microscope handler detection failed, but OpenHCS analysis "
                        "result artifacts were found."
                    ),
                    hint=str(handler_error),
                ),
            )
        if PlateFileInventoryQuery.kind_from_value(request.kind) is PlateFileKind.IMAGE:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value,
                    message=(
                        "No image records were discoverable, but "
                        f"{len(result_inventory.records)} analysis result "
                        "artifact(s) were found."
                    ),
                    hint=PlateInspectionText.RESULT_FILES_AVAILABLE_HINT,
                )
            )
        file_inventory = PlateFileInventory.from_inventories(
            PlateImageInventory(plate_path=plate_path, records=()),
            result_inventory,
        )
        return PlateInspectionService._query_files_from_inventory(
            request=request,
            plate_path=plate_path,
            file_inventory=file_inventory,
            detected_microscope_type=None,
            handler_class=None,
            parser_class=None,
            warnings=tuple(warnings),
        )

    @staticmethod
    def _has_disk_image_files(
        filemanager: "FileManager",
        plate_path: Path,
    ) -> bool | None:
        try:
            return bool(
                filemanager.list_image_files(
                    plate_path,
                    Backend.DISK.value,
                    recursive=True,
                )
            )
        except Exception:
            return None

    @staticmethod
    def _query_files_error(
        request: PlateFileQueryRequest,
        error: AgentError,
        *,
        plate_path: Path | None = None,
    ) -> PlateFileQueryResult:
        return PlateFileQueryResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path) if plate_path is not None else request.plate_path,
            requested_microscope_type=request.microscope_type,
            errors=(error,),
        )

    def _inspect(
        self,
        request: PlatePathInspectionRequest,
    ) -> PlatePathInspectionResult:
        bounds = request.bounds.normalized()
        available_types = self._available_handler_types()
        try:
            plate_path = self._path_policy.assert_readable(request.plate_path)
            if not plate_path.is_dir():
                return self._error_result(
                    request=request,
                    available_types=available_types,
                    error=AgentError(
                        code=PlateInspectionIssueCode.PATH_NOT_DIRECTORY.value,
                        message=f"Plate path is not a directory: {plate_path}",
                        hint=PlateInspectionText.DIRECTORY_HINT,
                        path=str(plate_path),
                    ),
                )
        except AgentPathPolicyError as exc:
            return self._error_result(
                request=request,
                available_types=available_types,
                error=AgentError.from_exception(
                    PlateInspectionIssueCode.PATH_POLICY_REJECTED.value,
                    exc,
                    hint=PlateInspectionText.LOCAL_PATH_HINT,
                    path=request.plate_path,
                ),
            )

        filemanager = self._filemanager_factory.create()
        if (
            request.microscope_type == PlateInspectionDefaults.MICROSCOPE_AUTO
            and request.pattern_format is None
        ):
            result_only = self._result_only_inspection(
                request=request,
                available_types=available_types,
                plate_path=plate_path,
                handler_error=None,
                bounds=bounds,
            )
            if (
                result_only is not None
                and self._has_disk_image_files(filemanager, plate_path) is False
            ):
                return result_only

        try:
            handler = self._create_handler(request, plate_path, filemanager)
        except Exception as exc:
            result_only = self._result_only_inspection(
                request=request,
                available_types=available_types,
                plate_path=plate_path,
                handler_error=exc,
                bounds=bounds,
            )
            if result_only is not None:
                return result_only
            return self._error_result(
                request=request,
                available_types=available_types,
                error=AgentError.from_exception(
                    PlateInspectionIssueCode.HANDLER_DETECTION_FAILED.value,
                    exc,
                    hint=PlateInspectionText.HANDLER_HINT,
                    path=str(plate_path),
                ),
                plate_path=plate_path,
            )

        warnings: list[AgentWarning] = []
        metadata_file_path = self._metadata_file_path(handler, plate_path, warnings)
        grid_dimensions = self._grid_dimensions(handler, plate_path, warnings)
        pixel_size = self._pixel_size(handler, plate_path, warnings)
        available_backends = self._available_backends(handler, plate_path, warnings)
        parser = self._parser(handler, warnings)
        file_inventory = self._plate_file_inventory(
            handler,
            plate_path,
            parser,
            filemanager,
            warnings,
        )
        image_files = tuple(
            record.virtual_path for record in file_inventory.image_records
        )
        parsed = self._filename_parser.parse(
            parser=parser,
            image_files=image_files,
            bounds=bounds,
        )
        if parser is None:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.PARSER_UNAVAILABLE.value,
                    message=PlateInspectionText.PARSER_HINT,
                    hint=PlateInspectionText.PARSER_HINT,
                )
            )
        if parsed.summary.skipped_file_count:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.PARSE_LIMIT_REACHED.value,
                    message=(
                        f"Parsed {parsed.summary.attempted_file_count} of "
                        f"{len(image_files)} image files."
                    ),
                    hint=PlateInspectionText.PARSE_LIMIT_HINT,
                )
            )
        if parsed.summary.failed_file_count:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.PARSE_FAILURES.value,
                    message=(
                        f"{parsed.summary.failed_file_count} image filenames "
                        "could not be parsed by the selected handler."
                    ),
                )
            )
        parse_coverage = PlateInspectionParseCoverage.from_summary(parsed.summary)
        if parse_coverage.is_low:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.LOW_PARSE_COVERAGE.value,
                    message=(
                        f"Only {parse_coverage.parsed_file_count} of "
                        f"{parse_coverage.attempted_file_count} image filenames "
                        f"parsed ({parse_coverage.percent:.1f}%)."
                    ),
                    hint=PlateInspectionText.LOW_PARSE_COVERAGE_HINT,
                )
            )
        if not image_files:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.NO_IMAGE_FILES.value,
                    message="No image files were reported by the selected handler.",
                    hint=PlateInspectionText.NO_IMAGES_HINT,
                )
            )

        metadata_values = self._metadata_values(handler, plate_path, warnings)
        source_diagnostics = self._source_diagnostics(
            handler,
            plate_path,
            warnings,
        )
        components = self._component_builder.build(
            metadata_values=metadata_values,
            parsed_values=parsed.components,
            bounds=bounds,
        )
        image_summary = PlateInspectionImageFileSummary(
            count=len(image_files),
            sampled_files=bounds.sample_strings(image_files, bounds.max_sample_files),
            sampled_records=self._sampled_image_records(
                file_inventory.image_inventory,
                bounds.max_sample_files,
            ),
            truncated_file_count=max(0, len(image_files) - bounds.max_sample_files),
        )
        result_summary = self._result_file_summary(
            file_inventory.result_inventory,
            bounds.max_sample_files,
        )
        format_specific_candidates = (
            PlateInspectionHandlerCandidateProjection.candidates(
                requested_microscope_type=request.microscope_type,
                selected_handler=handler,
                plate_path=plate_path,
                filemanager=filemanager,
                max_files_to_parse=bounds.max_files_to_parse,
            )
        )
        if format_specific_candidates:
            candidate_names = ", ".join(
                candidate.microscope_type
                for candidate in format_specific_candidates
            )
            supports_partial = all(
                PlateInspectionHandlerCandidateProjection.supports_explicit_incomplete_export(
                    candidate
                )
                for candidate in format_specific_candidates
            )
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.PROBABLE_NATIVE_HANDLER.value,
                    message=(
                        "A registered format-specific parser recognized every "
                        "tested source filename under its expected layout: "
                        f"{candidate_names}."
                    ),
                    hint=(
                        (
                            "Inspect format_specific_handler_candidates and choose "
                            "between a complete vendor export or explicit partial "
                            "analysis with the native microscope_type."
                        )
                        if supports_partial
                        else (
                            "The candidate owner requires its complete metadata "
                            "detection contract. Obtain the complete export, or use "
                            "SourceBindingsConfig to declare intentionally loose "
                            "ordinary image files without selecting the native handler."
                        )
                    ),
                )
            )
        errors: tuple[AgentError, ...] = ()
        warnings_tuple = tuple(warnings)
        detected_type = handler.microscope_type
        status = self._status_policy.status(
            errors=errors,
            warnings=warnings_tuple,
        )
        confidence = self._status_policy.confidence(
            errors=errors,
            image_file_count=len(image_files),
            parse_coverage=parse_coverage,
            detected_microscope_type=detected_type,
        )

        return PlatePathInspectionResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path),
            requested_microscope_type=request.microscope_type,
            status=status,
            confidence=confidence,
            available_microscope_types=available_types,
            detected_microscope_type=detected_type,
            handler_class=type(handler).__name__,
            parser_class=None if parser is None else type(parser).__name__,
            metadata_handler_class=type(handler.metadata_handler).__name__,
            root_dir=handler.root_dir,
            compatible_backends=PlateInspectionBackendProjection.names(
                handler.compatible_backends,
            ),
            available_backends=available_backends,
            metadata_file_path=metadata_file_path,
            grid_dimensions=grid_dimensions,
            pixel_size=pixel_size,
            image_files=image_summary,
            result_files=result_summary,
            parse_summary=parsed.summary,
            components=components,
            source_diagnostics=source_diagnostics,
            format_specific_handler_candidates=format_specific_candidates,
            workspace_preparation=self._preparation_policy.for_microscope(
                detected_type
            ),
            workflow_advice=self._workflow_advice_policy.for_handler(
                handler,
                format_specific_candidates=format_specific_candidates,
                requested_microscope_type=request.microscope_type,
            ),
            errors=errors,
            warnings=warnings_tuple,
        )

    def _result_only_inspection(
        self,
        *,
        request: PlatePathInspectionRequest,
        available_types: tuple[str, ...],
        plate_path: Path,
        handler_error: Exception | None,
        bounds: PlateInspectionBounds,
    ) -> PlatePathInspectionResult | None:
        result_inventory = PlateResultFileInventory.from_configured_output_root(
            plate_path=plate_path,
            path_config=self._path_planning_config(),
        )
        if not result_inventory.records:
            return None

        warnings: list[AgentWarning] = [
            AgentWarning(
                code=PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value,
                message=(
                    "This root has analysis result artifacts but no inspectable "
                    "image inventory."
                ),
                hint=PlateInspectionText.RESULT_FILES_AVAILABLE_HINT,
            )
        ]
        if handler_error is not None:
            warnings.insert(
                0,
                AgentWarning(
                    code=PlateInspectionIssueCode.HANDLER_DETECTION_FAILED.value,
                    message=(
                        "Microscope handler detection failed, but OpenHCS analysis "
                        "result artifacts were found."
                    ),
                    hint=str(handler_error),
                ),
            )
        return PlatePathInspectionResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path),
            requested_microscope_type=request.microscope_type,
            status=PlateInspectionStatus.PARTIAL,
            confidence=PlateInspectionConfidence.NONE,
            available_microscope_types=available_types,
            result_files=PlateInspectionService._result_file_summary(
                result_inventory,
                bounds.max_sample_files,
            ),
            workflow_advice=self._workflow_advice_policy.unresolved(),
            warnings=tuple(warnings),
        )

    @staticmethod
    def _available_handler_types() -> tuple[str, ...]:
        from openhcs.microscopes import get_all_handler_types

        return tuple(sorted(get_all_handler_types()))

    @staticmethod
    def _create_handler(
        request: PlatePathInspectionRequest,
        plate_path: Path,
        filemanager: "FileManager",
    ) -> "MicroscopeHandler":
        from openhcs.microscopes import create_microscope_handler

        return create_microscope_handler(
            microscope_type=request.microscope_type,
            plate_folder=plate_path,
            filemanager=filemanager,
            pattern_format=request.pattern_format,
        )

    @staticmethod
    def _metadata_file_path(
        handler: "MicroscopeHandler",
        plate_path: Path,
        warnings: list[AgentWarning],
    ) -> str | None:
        try:
            metadata_path = handler.metadata_handler.find_metadata_file(plate_path)
            return str(metadata_path)
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.METADATA_FILE_UNAVAILABLE.value,
                    message=str(exc),
                )
            )
            return None

    @staticmethod
    def _grid_dimensions(
        handler: "MicroscopeHandler",
        plate_path: Path,
        warnings: list[AgentWarning],
    ) -> tuple[int, int] | None:
        try:
            dims = handler.metadata_handler.get_grid_dimensions(plate_path)
            return (int(dims[0]), int(dims[1]))
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.GRID_DIMENSIONS_UNAVAILABLE.value,
                    message=str(exc),
                )
            )
            return None

    @staticmethod
    def _pixel_size(
        handler: "MicroscopeHandler",
        plate_path: Path,
        warnings: list[AgentWarning],
    ) -> float | None:
        try:
            return float(handler.metadata_handler.get_pixel_size(plate_path))
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.PIXEL_SIZE_UNAVAILABLE.value,
                    message=str(exc),
                )
            )
            return None

    @staticmethod
    def _available_backends(
        handler: "MicroscopeHandler",
        plate_path: Path,
        warnings: list[AgentWarning],
    ) -> tuple[str, ...]:
        try:
            return PlateInspectionBackendProjection.names(
                handler.get_available_backends(plate_path)
            )
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.AVAILABLE_BACKENDS_UNAVAILABLE.value,
                    message=str(exc),
                )
            )
            return ()

    def _plate_file_inventory_for_query(
        self,
        handler: "MicroscopeHandler",
        plate_path: Path,
        parser: "FilenameParser | None",
        filemanager: "FileManager",
        query_kind: PlateFileKind | None,
        warnings: list[AgentWarning],
    ) -> PlateFileInventory:
        if query_kind is PlateFileKind.IMAGE:
            image_inventory = PlateInspectionService._image_inventory(
                handler,
                plate_path,
                parser,
                filemanager,
                warnings,
            )
            if not image_inventory.records:
                result_inventory = self._result_file_inventory(
                    handler,
                    plate_path,
                    parser,
                    warnings,
                    warn_on_recovered_listing_failure=False,
                )
                if result_inventory.records:
                    warnings.append(
                        AgentWarning(
                            code=PlateInspectionIssueCode.RESULT_FILES_AVAILABLE.value,
                            message=(
                                "No image records were discoverable, but "
                                f"{len(result_inventory.records)} analysis result "
                                "artifact(s) were found."
                            ),
                            hint=PlateInspectionText.RESULT_FILES_AVAILABLE_HINT,
                        )
                    )
            return PlateFileInventory.from_inventories(
                image_inventory,
                PlateResultFileInventory(plate_path=plate_path, records=()),
            )
        if query_kind is PlateFileKind.RESULT:
            return PlateFileInventory.from_inventories(
                PlateImageInventory(plate_path=plate_path, records=()),
                self._result_file_inventory(
                    handler,
                    plate_path,
                    parser,
                    warnings,
                    warn_on_recovered_listing_failure=False,
                ),
            )
        return PlateFileInventory.from_inventories(
            PlateInspectionService._image_inventory(
                handler,
                plate_path,
                parser,
                filemanager,
                warnings,
            ),
            self._result_file_inventory(
                handler,
                plate_path,
                parser,
                warnings,
                warn_on_recovered_listing_failure=False,
            ),
        )

    def _plate_file_inventory(
        self,
        handler: "MicroscopeHandler",
        plate_path: Path,
        parser: "FilenameParser | None",
        filemanager: "FileManager",
        warnings: list[AgentWarning],
    ) -> PlateFileInventory:
        image_inventory = PlateInspectionService._image_inventory(
            handler,
            plate_path,
            parser,
            filemanager,
            warnings,
        )
        result_inventory = self._result_file_inventory(
            handler,
            plate_path,
            parser,
            warnings,
        )
        return PlateFileInventory.from_inventories(image_inventory, result_inventory)

    @staticmethod
    def _image_inventory(
        handler: "MicroscopeHandler",
        plate_path: Path,
        parser: "FilenameParser | None",
        filemanager: "FileManager",
        warnings: list[AgentWarning],
    ) -> PlateImageInventory:
        try:
            source_projection = (
                VirtualWorkspaceSourceProjectionAuthority.from_plate_metadata(
                    plate_path=plate_path,
                    metadata_handler=handler.metadata_handler,
                    filemanager=filemanager,
                ).projection_if_available()
            )
            if source_projection is not None:
                handler.register_workspace_backends(plate_path, filemanager)
            return PlateImageInventory.from_handler(
                plate_path=plate_path,
                metadata_handler=handler.metadata_handler,
                parser=parser,
                filemanager=filemanager,
                backend=handler.get_primary_backend(plate_path, filemanager),
                all_subdirs=True,
            )
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.IMAGE_FILE_LISTING_FAILED.value,
                    message=str(exc),
                    hint=PlateInspectionText.LISTING_HINT,
                )
            )
            return PlateImageInventory(plate_path=plate_path, records=())

    def _result_file_inventory(
        self,
        handler: "MicroscopeHandler",
        plate_path: Path,
        parser: "FilenameParser | None",
        warnings: list[AgentWarning],
        *,
        warn_on_recovered_listing_failure: bool = True,
    ) -> PlateResultFileInventory:
        records_by_path: dict[str, PlateResultFileRecord] = {}
        scanned_file_count = 0
        try:
            handler_inventory = (
                PlateResultFileInventory.from_handler_and_configured_output_root(
                    plate_path=plate_path,
                    metadata_handler=handler.metadata_handler,
                    parser=parser,
                    path_config=self._path_planning_config(),
                )
            )
            scanned_file_count += handler_inventory.scanned_file_count
            records_by_path.update(
                (record.full_path, record) for record in handler_inventory.records
            )
        except Exception as exc:
            output_inventory = PlateResultFileInventory.from_configured_output_root(
                plate_path=plate_path,
                path_config=self._path_planning_config(),
                parser=parser,
            )
            if warn_on_recovered_listing_failure or not output_inventory.records:
                warnings.append(
                    AgentWarning(
                        code=PlateInspectionIssueCode.RESULT_FILE_LISTING_FAILED.value,
                        message=str(exc),
                        hint=(
                            "The selected handler declared result directories that "
                            "could not be scanned read-only."
                        ),
                    )
                )
            scanned_file_count += output_inventory.scanned_file_count
            records_by_path.update(
                (record.full_path, record) for record in output_inventory.records
            )
        return PlateResultFileInventory(
            plate_path=plate_path,
            records=tuple(
                sorted(records_by_path.values(), key=lambda record: record.relative_path)
            ),
            scanned_file_count=scanned_file_count,
        )

    @staticmethod
    def _sampled_image_records(
        inventory: PlateImageInventory,
        max_sample_files: int,
    ) -> tuple[PlateInspectionImageRecordSummary, ...]:
        bounded_limit = max(int(max_sample_files), PlateInspectionDefaults.MIN_BOUND)
        return tuple(
            PlateInspectionImageRecordSummary(
                virtual_path=record.virtual_path,
                full_virtual_path=record.full_virtual_path,
                source_path=record.source_path,
                metadata=dict(record.metadata),
            )
            for record in inventory.records[:bounded_limit]
        )

    @staticmethod
    def _result_file_summary(
        inventory: PlateResultFileInventory,
        max_sample_files: int,
    ) -> PlateInspectionResultFileSummary:
        bounded_limit = max(int(max_sample_files), PlateInspectionDefaults.MIN_BOUND)
        sampled_records = tuple(
            PlateInspectionResultFileRecordSummary(
                relative_path=record.relative_path,
                full_path=record.full_path,
                file_format=record.file_format.name,
                metadata=dict(record.metadata),
                preview=PlateInspectionService._result_file_preview(record),
            )
            for record in inventory.records[:bounded_limit]
        )
        sampled_files = tuple(record.relative_path for record in sampled_records)
        return PlateInspectionResultFileSummary(
            count=len(inventory.records),
            scanned_file_count=inventory.scanned_file_count,
            sampled_files=sampled_files,
            sampled_records=sampled_records,
            truncated_file_count=max(0, len(inventory.records) - bounded_limit),
        )

    @staticmethod
    def _result_file_preview(
        record: PlateResultFileRecord,
    ) -> PlateInspectionResultFilePreview | None:
        preview = PlateResultFilePreviewReader.preview(record)
        if preview is None:
            return None
        return PlateInspectionResultFilePreview(
            text_lines=preview.text_lines,
            csv_columns=preview.csv_columns,
            csv_rows=tuple(dict(row) for row in preview.csv_rows),
            roi_count=preview.roi_count,
            roi_member_count=preview.roi_member_count,
            roi_duplicate_member_count=preview.roi_duplicate_member_count,
            roi_area_min=preview.roi_area_min,
            roi_area_max=preview.roi_area_max,
            roi_area_mean=preview.roi_area_mean,
            roi_examples=tuple(dict(row) for row in preview.roi_examples),
            truncated=preview.truncated,
            omitted_reason=preview.omitted_reason,
        )

    @staticmethod
    def _parser(
        handler: "MicroscopeHandler",
        warnings: list[AgentWarning],
        *,
        warn: bool = True,
    ) -> "FilenameParser | None":
        try:
            return handler.parser
        except Exception as exc:
            if warn:
                warnings.append(
                    AgentWarning(
                        code=PlateInspectionIssueCode.PARSER_UNAVAILABLE.value,
                        message=str(exc),
                        hint=PlateInspectionText.PARSER_HINT,
                    )
                )
            return None

    def _metadata_values(
        self,
        handler: "MicroscopeHandler",
        plate_path: Path,
        warnings: list[AgentWarning],
    ) -> PlateInspectionComponentCollection:
        try:
            return self._metadata_access.collect(handler.metadata_handler, plate_path)
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=PlateInspectionIssueCode.METADATA_FILE_UNAVAILABLE.value,
                    message=str(exc),
                )
            )
            return PlateInspectionComponentCollection.empty()

    @staticmethod
    def _source_diagnostics(
        handler: "MicroscopeHandler",
        plate_path: Path,
        warnings: list[AgentWarning],
    ) -> tuple[JsonObject, ...]:
        """Read structured diagnostics through the metadata-handler owner."""

        try:
            return tuple(
                dict(diagnostic)
                for diagnostic in handler.metadata_handler.source_diagnostics(
                    plate_path
                )
            )
        except Exception as exc:
            warnings.append(
                AgentWarning(
                    code=(
                        PlateInspectionIssueCode.SOURCE_DIAGNOSTICS_UNAVAILABLE.value
                    ),
                    message=str(exc),
                )
            )
            return ()

    def _error_result(
        self,
        *,
        request: PlatePathInspectionRequest,
        available_types: tuple[str, ...],
        error: AgentError,
        plate_path: Path | None = None,
    ) -> PlatePathInspectionResult:
        return PlatePathInspectionResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path) if plate_path is not None else request.plate_path,
            requested_microscope_type=request.microscope_type,
            status=PlateInspectionStatus.ERROR,
            confidence=PlateInspectionConfidence.NONE,
            available_microscope_types=available_types,
            workflow_advice=self._workflow_advice_policy.unresolved(),
            errors=(error,),
        )

    @staticmethod
    def _sample_image_error(
        request: PlateImageSampleRequest,
        error: AgentError,
        *,
        plate_path: Path | None = None,
    ) -> PlateImageSampleResult:
        return PlateImageSampleResult(
            schema_version=SCHEMA_VERSION,
            plate_path=str(plate_path) if plate_path is not None else request.plate_path,
            requested_image_path=request.image_path,
            errors=(error,),
        )
