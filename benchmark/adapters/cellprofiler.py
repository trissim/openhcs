"""Native CellProfiler tool adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from contextlib import ExitStack
from dataclasses import dataclass
from enum import StrEnum
from os import environ
from pathlib import Path
from typing import Any, ClassVar, Mapping
from urllib.parse import quote

from metaclass_registry import AutoRegisterMeta
from benchmark.adapters.cellprofiler_installation import (
    CELLPROFILER_EXECUTABLE_ENV,
    CellProfilerExecutableResolver,
)
from benchmark.adapters.cppipe_source import (
    CPPipeSourceRequest,
    CPPipeSourceResolution,
    resolve_cppipe_source,
)
from benchmark.adapters.openhcs import OpenHCSAxisSelection
from benchmark.contracts.metric import MetricCollector
from benchmark.contracts.tool_adapter import (
    BenchmarkResult,
    ToolAdapter,
    ToolExecutionError,
    ToolNotInstalledError,
)
from benchmark.timing import BenchmarkPhase, PhaseTimingTrace
from openhcs.core.pipeline_image_schema import ImportedMetadataTable
from openhcs.core.source_schema_workspace import (
    ImportedMetadataPathResolver,
    SourceSchemaWorkspaceMaterialization,
    materialize_source_schema_workspace,
)
from openhcs.core.runtime_equivalence import RuntimeOutputSnapshot
from openhcs.interop.cellprofiler.parser import CPPipeParser
from openhcs.interop.cellprofiler.source_schema import (
    compile_image_schema,
    is_imported_metadata_method,
)


BENCHMARK_CACHE_DOMAINS = frozenset({"native_reference"})
PYTHONHASHSEED_ENV = "PYTHONHASHSEED"
DETERMINISTIC_PYTHONHASHSEED = "0"
NATIVE_CELLPROFILER_SUCCESS_MARKER = ".cellprofiler_benchmark_reference.json"
CELLPROFILER_FIRST_IMAGE_SET_PARAM = "cellprofiler_first_image_set"
CELLPROFILER_LAST_IMAGE_SET_PARAM = "cellprofiler_last_image_set"
CELLPROFILER_IMAGE_SET_BOUND_PARAMS = frozenset(
    {
        CELLPROFILER_FIRST_IMAGE_SET_PARAM,
        CELLPROFILER_LAST_IMAGE_SET_PARAM,
    }
)


class NativeCellProfilerInputDomainStrategyKey(StrEnum):
    """Registered native CellProfiler input-domain identities."""

    SELECTED_SOURCE_SCHEMA_WELLS = "selected_source_schema_wells"
    EMBEDDED_IMAGE_PLANES = "embedded_image_planes"
    DATASET_FOLDER = "dataset_folder"


class NativeCellProfilerSelectedSourceMode(StrEnum):
    """Native CP source delivery mode for selected source-schema inputs."""

    EMBEDDED_IMAGE_PLANES = "embedded_image_planes"
    FILE_LIST = "file_list"


class HeadlessCellProfilerPipelinePatch(StrEnum):
    """Headless native CellProfiler pipeline patches with explicit semantics."""

    ALLOW_SAVE_OVERWRITE = "allow_save_overwrite"
    TRUST_SELECTED_SOURCE_UNIVERSE = "trust_selected_source_universe"

    def apply(self, source_text: str) -> str:
        """Apply this headless-execution patch to pipeline source text."""
        if self is HeadlessCellProfilerPipelinePatch.ALLOW_SAVE_OVERWRITE:
            return source_text.replace(
                "Overwrite existing files without warning?:No",
                "Overwrite existing files without warning?:Yes",
            )
        if self is HeadlessCellProfilerPipelinePatch.TRUST_SELECTED_SOURCE_UNIVERSE:
            return source_text.replace(
                "Filter images?:Images only",
                "Filter images?:No filtering",
            )
        raise AssertionError(f"Unhandled headless CellProfiler patch: {self!r}")


class NativeCellProfilerProvenanceField(StrEnum):
    """Native CellProfiler provenance fields with cross-run semantics."""

    INPUT_DOMAIN_STRATEGY = "native_input_domain_strategy"
    SOURCE_WORKSPACE = "native_source_workspace"
    SOURCE_PLANE_COUNT = "native_source_plane_count"
    SELECTED_WELLS = "native_selected_wells"
    SELECTED_SOURCE_FILE_COUNT = "native_selected_source_file_count"
    SELECTED_SOURCE_MODE = "native_selected_source_mode"
    SELECTED_SOURCE_FLATTENED = "native_selected_source_flattened"
    SELECTED_SOURCE_STAGING_ROOT = "native_selected_source_staging_root"
    FILE_LIST_PATH = "native_file_list_path"


class HeadlessCellProfilerPipelinePolicy:
    """Prepare a CellProfiler pipeline for non-interactive native execution."""

    @staticmethod
    def execution_path(
        cppipe_path: Path,
        output_dir: Path,
        patches: tuple[HeadlessCellProfilerPipelinePatch, ...] = (
            HeadlessCellProfilerPipelinePatch.ALLOW_SAVE_OVERWRITE,
        ),
    ) -> Path:
        source_text = Path(cppipe_path).read_text(encoding="utf-8")
        patched_text = source_text
        for patch in patches:
            patched_text = patch.apply(patched_text)
        if patched_text == source_text:
            return cppipe_path
        patched_path = output_dir / "native_cellprofiler_headless" / cppipe_path.name
        patched_path.parent.mkdir(parents=True, exist_ok=True)
        patched_path.write_text(patched_text, encoding="utf-8")
        return patched_path


@dataclass(frozen=True, slots=True)
class NativeCellProfilerImportedMetadataPipelinePatch:
    """Point native CellProfiler metadata-import blocks at staged input files."""

    imported_metadata_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "imported_metadata_paths",
            tuple(Path(path) for path in self.imported_metadata_paths),
        )

    def execution_path(self, cppipe_path: Path, output_dir: Path) -> Path:
        """Return a patched pipeline path when imported metadata paths are staged."""
        if not self.imported_metadata_paths:
            return cppipe_path
        source_text = Path(cppipe_path).read_text(encoding="utf-8")
        patched_text = self.patch_text(source_text)
        if patched_text == source_text:
            return cppipe_path
        patched_path = output_dir / "native_cellprofiler_headless" / cppipe_path.name
        patched_path.parent.mkdir(parents=True, exist_ok=True)
        patched_path.write_text(patched_text, encoding="utf-8")
        return patched_path

    def patch_text(self, source_text: str) -> str:
        """Rewrite imported-metadata settings in CellProfiler pipeline text."""
        lines = source_text.splitlines(keepends=True)
        patched_lines: list[str] = []
        in_metadata_module = False
        in_imported_metadata_block = False
        import_index = 0
        for line in lines:
            module_match = CPPipeParser.MODULE_HEADER_PATTERN.match(line.rstrip("\n"))
            if module_match is not None:
                in_metadata_module = module_match.group(1) == "Metadata"
                in_imported_metadata_block = False
                patched_lines.append(line)
                continue
            if not in_metadata_module:
                patched_lines.append(line)
                continue
            setting = _cppipe_setting_line(line)
            if setting is None:
                patched_lines.append(line)
                continue
            setting_name, setting_value, newline = setting
            if setting_name == "Metadata extraction method":
                in_imported_metadata_block = is_imported_metadata_method(setting_value)
                patched_lines.append(line)
                continue
            if not in_imported_metadata_block:
                patched_lines.append(line)
                continue
            if setting_name == "Metadata file location":
                patched_lines.append(
                    "    Metadata file location:Default Input Folder|" + newline
                )
                continue
            if setting_name == "Metadata file name":
                if import_index >= len(self.imported_metadata_paths):
                    raise ToolExecutionError(
                        "Native CellProfiler pipeline declares more imported metadata "
                        "blocks than the source schema resolved."
                    )
                patched_lines.append(
                    "    Metadata file name:"
                    f"{self.imported_metadata_paths[import_index].name}"
                    f"{newline}"
                )
                import_index += 1
                continue
            patched_lines.append(line)
        if import_index != len(self.imported_metadata_paths):
            raise ToolExecutionError(
                "Native CellProfiler source schema resolved imported metadata tables "
                "that were not found in the pipeline Metadata module."
            )
        return "".join(patched_lines)


def _cppipe_setting_line(line: str) -> tuple[str, str, str] | None:
    """Return CellProfiler setting name, value, and newline suffix for one line."""
    newline = "\n" if line.endswith("\n") else ""
    content = line.removesuffix("\n")
    setting_match = CPPipeParser.SETTING_PATTERN.match(content)
    if setting_match is None:
        return None
    return setting_match.group(1).strip(), setting_match.group(2).strip(), newline


@dataclass(frozen=True, slots=True)
class CellProfilerRunRequest:
    """Authoritative native CellProfiler run request."""

    dataset_path: Path
    pipeline_name: str
    pipeline_params: dict[str, Any]
    metrics: tuple[MetricCollector, ...]
    output_dir: Path

    @property
    def dataset_id(self) -> str:
        return str(self.pipeline_params.get("dataset_id", self.dataset_path.name))

    @property
    def timeout_seconds(self) -> float | None:
        value = self.pipeline_params.get("cellprofiler_timeout_seconds")
        if value is None:
            return None
        return float(value)

    @property
    def first_image_set(self) -> int | None:
        return _optional_positive_int(
            self.pipeline_params.get(CELLPROFILER_FIRST_IMAGE_SET_PARAM),
            CELLPROFILER_FIRST_IMAGE_SET_PARAM,
        )

    @property
    def last_image_set(self) -> int | None:
        return _optional_positive_int(
            self.pipeline_params.get(CELLPROFILER_LAST_IMAGE_SET_PARAM),
            CELLPROFILER_LAST_IMAGE_SET_PARAM,
        )

    @property
    def openhcs_axis_selection(self) -> OpenHCSAxisSelection:
        return OpenHCSAxisSelection.from_pipeline_params(self.pipeline_params)

    @property
    def cppipe_source(self) -> CPPipeSourceRequest:
        return CPPipeSourceRequest.from_pipeline_params(
            dataset_id=self.dataset_id,
            output_dir=self.output_dir,
            pipeline_params=self.pipeline_params,
        )


@dataclass(frozen=True, slots=True)
class NativeCellProfilerInputDomain:
    """Concrete native-CellProfiler input domain for one run."""

    cppipe_path: Path
    input_dir: Path
    provenance: dict[str, Any]
    file_list_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))
        object.__setattr__(self, "input_dir", Path(self.input_dir))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if self.file_list_path is not None:
            object.__setattr__(self, "file_list_path", Path(self.file_list_path))


@dataclass(frozen=True, slots=True)
class NativeCellProfilerSourcePlacement:
    """One selected native CellProfiler source and its input-dir relative path."""

    source_path: Path
    relative_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(
            self,
            "relative_path",
            Path(str(self.relative_path).lstrip("/")),
        )


@dataclass(frozen=True, slots=True)
class NativeCellProfilerSelectedSourceUniverse:
    """Selected source-schema files projected into a native CP input directory."""

    source_paths: tuple[Path, ...]
    placements: tuple[NativeCellProfilerSourcePlacement, ...] = ()
    imported_metadata_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        source_paths = tuple(Path(path) for path in self.source_paths)
        object.__setattr__(
            self,
            "source_paths",
            source_paths,
        )
        placements = tuple(self.placements) or tuple(
            NativeCellProfilerSourcePlacement(path, Path(path.name))
            for path in source_paths
        )
        object.__setattr__(
            self,
            "placements",
            placements,
        )
        object.__setattr__(
            self,
            "imported_metadata_paths",
            tuple(Path(path) for path in self.imported_metadata_paths),
        )

    @classmethod
    def from_workspace_wells(
        cls,
        workspace: SourceSchemaWorkspaceMaterialization,
        well_ids: tuple[str, ...],
        *,
        imported_metadata_tables: tuple[ImportedMetadataTable, ...] = (),
    ) -> "NativeCellProfilerSelectedSourceUniverse":
        source_paths = workspace.source_paths_for_primary_wells(
            well_ids,
            imported_metadata_tables=imported_metadata_tables,
        )
        placement_plan = NativeCellProfilerImportedMetadataPlacementPlan(
            workspace.source_root,
            imported_metadata_tables,
            source_paths,
        )
        placements = placement_plan.placements()
        imported_metadata_paths = tuple(
            placement_plan.imported_metadata_path(table)
            for table in imported_metadata_tables
        )
        return cls(source_paths, placements, imported_metadata_paths)

    def materialize_flat_input_dir(self, input_dir: Path) -> tuple[Path, ...]:
        input_dir = Path(input_dir)
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        projected_paths: list[Path] = []
        seen_targets: dict[Path, Path] = {}
        for placement in self.placements:
            target_path = input_dir / placement.relative_path
            existing = seen_targets.get(target_path)
            if existing is not None and existing != placement.source_path:
                raise ToolExecutionError(
                    "Native CellProfiler selected source universe has ambiguous "
                    f"target path {target_path!s}: {existing} and {placement.source_path}."
                )
            seen_targets[target_path] = placement.source_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if placement.source_path in self.imported_metadata_paths:
                self._materialize_selected_imported_metadata_table(
                    placement.source_path,
                    target_path,
                )
            else:
                try:
                    target_path.symlink_to(placement.source_path)
                except OSError:
                    shutil.copy2(placement.source_path, target_path)
            projected_paths.append(target_path)
        return tuple(projected_paths)

    def _materialize_selected_imported_metadata_table(
        self,
        source_path: Path,
        target_path: Path,
    ) -> None:
        source_names = {path.name for path in self.source_paths}
        with source_path.open(newline="", encoding="utf-8") as source_handle:
            reader = csv.DictReader(source_handle)
            if reader.fieldnames is None:
                raise ToolExecutionError(
                    f"Imported metadata table {source_path} has no header row."
                )
            rows = [
                row
                for row in reader
                if self._metadata_row_references_selected_source(row, source_names)
            ]
        with target_path.open("w", newline="", encoding="utf-8") as target_handle:
            writer = csv.DictWriter(target_handle, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _metadata_row_references_selected_source(
        row: Mapping[str, str],
        source_names: set[str],
    ) -> bool:
        filenames = tuple(
            str(value).strip()
            for key, value in row.items()
            if key.startswith("Image_FileName_") and value
        )
        if not filenames:
            return True
        return any(filename in source_names for filename in filenames)


@dataclass(frozen=True, slots=True)
class NativeCellProfilerScannerSafeInputPath:
    """Stage native inputs away from accidental folder-metadata tokens."""

    request: CellProfilerRunRequest
    source: CPPipeSourceResolution

    def input_dir(self) -> Path:
        return self.staging_root() / "input"

    def staging_root(self) -> Path:
        digest = hashlib.sha256(self.identity().encode("utf-8")).hexdigest()
        return Path("/tmp") / "openhcsnativecellprofiler" / _alpha_slug(digest[:16])

    def identity(self) -> str:
        selection = self.request.openhcs_axis_selection
        return json.dumps(
            {
                "axis_filter": selection.axis_filter,
                "dataset": str(self.request.dataset_path),
                "max_axis_count": selection.max_axis_count,
                "pipeline": str(self.source.path),
            },
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class NativeCellProfilerImportedMetadataPlacementPlan:
    """Project CP imported metadata path columns into native input paths."""

    source_root: Path
    imported_metadata_tables: tuple[ImportedMetadataTable, ...]
    source_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(
            self,
            "imported_metadata_tables",
            tuple(self.imported_metadata_tables),
        )
        object.__setattr__(
            self,
            "source_paths",
            tuple(Path(path) for path in self.source_paths),
        )

    def placements(self) -> tuple[NativeCellProfilerSourcePlacement, ...]:
        relative_paths_by_name = self.relative_paths_by_name()
        placements: list[NativeCellProfilerSourcePlacement] = []
        for source_path in self.source_paths:
            relative_path = relative_paths_by_name.get(
                source_path.name,
                Path(source_path.name),
            )
            placements.append(
                NativeCellProfilerSourcePlacement(source_path, relative_path)
            )
        return tuple(placements)

    def relative_paths_by_name(self) -> Mapping[str, Path]:
        relative_paths: dict[str, Path] = {}
        for table in self.imported_metadata_tables:
            table_path = self.imported_metadata_path(table)
            with table_path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    for filename_key, filename in row.items():
                        if not filename_key.startswith("Image_FileName_") or not filename:
                            continue
                        alias = filename_key.removeprefix("Image_FileName_")
                        path_name = row.get(f"Image_PathName_{alias}", "")
                        relative_path = Path(str(path_name).strip()) / str(filename).strip()
                        if relative_path.is_absolute():
                            relative_path = Path(relative_path.name)
                        existing = relative_paths.get(str(filename).strip())
                        if existing is not None and existing != relative_path:
                            raise ToolExecutionError(
                                "Native CellProfiler imported metadata gives "
                                f"ambiguous paths for {filename!r}: {existing} and "
                                f"{relative_path}."
                            )
                        relative_paths[str(filename).strip()] = relative_path
        return relative_paths

    def imported_metadata_path(self, table: ImportedMetadataTable) -> Path:
        resolver = ImportedMetadataPathResolver(self.source_root)
        for candidate in resolver.path_candidates(table):
            if candidate.is_file():
                return candidate
        raise ToolExecutionError(
            "Imported metadata table does not exist for native CellProfiler: "
            f"{table.location!r}. Searched: "
            f"{tuple(str(candidate) for candidate in resolver.path_candidates(table))!r}."
        )


class NativeCellProfilerInputDomainStrategy(ABC, metaclass=AutoRegisterMeta):
    """Prepare the native CellProfiler pipeline/input domain from typed source semantics."""

    __registry_key__ = "strategy_key"
    __skip_if_no_key__ = True
    strategy_key: ClassVar[NativeCellProfilerInputDomainStrategyKey | None] = None

    @classmethod
    def select_for(
        cls,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> "NativeCellProfilerInputDomainStrategy":
        candidate_strategies = tuple(
            strategy_type() for strategy_type in cls.__registry__.values()
        )
        matching_strategies = tuple(
            strategy
            for strategy in candidate_strategies
            if strategy.accepts(request, source)
        )
        if len(matching_strategies) > 1:
            names = tuple(strategy.strategy_key for strategy in matching_strategies)
            raise ToolExecutionError(
                "Native CellProfiler input domain is ambiguous for "
                f"{source.path}: {names!r}."
            )
        if matching_strategies:
            return matching_strategies[0]
        return DefaultNativeCellProfilerInputDomainStrategy()

    @abstractmethod
    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        """Return whether this strategy owns the source semantics."""

    @abstractmethod
    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        """Return the concrete native CellProfiler input domain."""

    def accepts_success_marker_source_schema(
        self,
        source_schema: Any,
        provenance: Mapping[str, Any],
    ) -> bool:
        """Return whether a success marker from this domain is reusable."""
        return True

    def reference_scope_slugs(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> tuple[str, ...]:
        """Return native-reference scope suffixes owned by this input domain."""
        slug = native_cellprofiler_image_set_scope_slug(request.pipeline_params)
        return (slug,) if slug is not None else ()


class SelectedWellSourceSchemaNativeCellProfilerInputDomainStrategy(
    NativeCellProfilerInputDomainStrategy
):
    """Run native CellProfiler on the same source-schema wells selected for OpenHCS."""

    strategy_key = NativeCellProfilerInputDomainStrategyKey.SELECTED_SOURCE_SCHEMA_WELLS

    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        modules = CPPipeParser().parse(source.path)
        schema = compile_image_schema(modules)
        selection = request.openhcs_axis_selection
        return (
            not schema.is_empty
            and (bool(selection.axis_filter) or selection.max_axis_count is not None)
        )

    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        modules = CPPipeParser().parse(source.path)
        schema = compile_image_schema(modules)
        workspace = materialize_source_schema_workspace(
            request.dataset_path,
            request.output_dir / "native_cellprofiler_selected_source_workspace",
            schema,
            image_set_selection=request.openhcs_axis_selection.source_schema_selection(),
        )
        selected_wells = workspace.primary_wells()
        selected_source_universe = NativeCellProfilerSelectedSourceUniverse.from_workspace_wells(
            workspace,
            selected_wells,
            imported_metadata_tables=schema.imported_metadata_tables,
        )
        selected_paths = selected_source_universe.source_paths
        if not selected_paths:
            raise ToolExecutionError(
                f"Native CellProfiler selected no source files for wells {selected_wells!r}."
            )
        selected_cppipe_path = HeadlessCellProfilerPipelinePolicy.execution_path(
            execution_cppipe_path,
            request.output_dir,
            patches=(
                HeadlessCellProfilerPipelinePatch.TRUST_SELECTED_SOURCE_UNIVERSE,
            ),
        )
        selected_cppipe_path = NativeCellProfilerImportedMetadataPipelinePatch(
            selected_source_universe.imported_metadata_paths
        ).execution_path(
            selected_cppipe_path,
            request.output_dir,
        )
        if schema.image_plane_sources:
            embedded_strategy = EmbeddedImagePlaneNativeCellProfilerInputDomainStrategy()
            patched_cppipe_path = embedded_strategy.rewrite_embedded_image_plane_sources(
                selected_cppipe_path,
                request.output_dir / "native_cellprofiler_headless",
                tuple(embedded_strategy.file_uri(path) for path in selected_paths),
            )
            input_dir = request.output_dir / "native_cellprofiler_empty_input"
            if input_dir.exists():
                shutil.rmtree(input_dir)
            input_dir.mkdir(parents=True, exist_ok=True)
            return NativeCellProfilerInputDomain(
                cppipe_path=patched_cppipe_path,
                input_dir=input_dir,
                provenance={
                    NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                        self.strategy_key
                    ),
                    NativeCellProfilerProvenanceField.SOURCE_WORKSPACE: str(
                        workspace.workspace_root
                    ),
                    NativeCellProfilerProvenanceField.SELECTED_WELLS: selected_wells,
                    NativeCellProfilerProvenanceField.SELECTED_SOURCE_FILE_COUNT: len(
                        selected_paths
                    ),
                    NativeCellProfilerProvenanceField.SELECTED_SOURCE_MODE: (
                        NativeCellProfilerSelectedSourceMode.EMBEDDED_IMAGE_PLANES
                    ),
                },
            )
        requires_flat_input_dir = bool(
            workspace.auxiliary_mappings or schema.imported_metadata_tables
        )
        selected_input_path = NativeCellProfilerScannerSafeInputPath(request, source)
        selected_input_dir = selected_input_path.input_dir()
        file_list_path = None
        if requires_flat_input_dir:
            selected_source_universe.materialize_flat_input_dir(selected_input_dir)
        else:
            file_list_path = self._write_file_list(
                request.output_dir / "native_cellprofiler_file_list.txt",
                selected_paths,
            )
        return NativeCellProfilerInputDomain(
            cppipe_path=selected_cppipe_path,
            input_dir=(
                selected_input_dir
                if requires_flat_input_dir
                else request.dataset_path
            ),
            file_list_path=file_list_path,
            provenance={
                NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                    self.strategy_key
                ),
                NativeCellProfilerProvenanceField.SOURCE_WORKSPACE: str(
                    workspace.workspace_root
                ),
                NativeCellProfilerProvenanceField.SELECTED_WELLS: selected_wells,
                NativeCellProfilerProvenanceField.SELECTED_SOURCE_FILE_COUNT: len(
                    selected_paths
                ),
                NativeCellProfilerProvenanceField.SELECTED_SOURCE_MODE: (
                    NativeCellProfilerSelectedSourceMode.FILE_LIST
                ),
                NativeCellProfilerProvenanceField.SELECTED_SOURCE_FLATTENED: bool(
                    requires_flat_input_dir
                ),
                NativeCellProfilerProvenanceField.SELECTED_SOURCE_STAGING_ROOT: (
                    str(selected_input_path.staging_root())
                    if requires_flat_input_dir
                    else None
                ),
            },
        )

    def _write_file_list(self, path: Path, selected_paths: tuple[Path, ...]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        uris = tuple(
            EmbeddedImagePlaneNativeCellProfilerInputDomainStrategy.file_uri(
                source_path
            )
            for source_path in selected_paths
        )
        path.write_text("\n".join(uris) + "\n", encoding="utf-8")
        return path

    def reference_scope_slugs(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> tuple[str, ...]:
        sample_slug = native_cellprofiler_sample_scope_slug(request.pipeline_params)
        return tuple(
            slug
            for slug in (
                sample_slug,
                *super().reference_scope_slugs(request, source),
            )
            if slug is not None
        )


class EmbeddedImagePlaneNativeCellProfilerInputDomainStrategy(
    NativeCellProfilerInputDomainStrategy
):
    """Run embedded image-plane pipelines against a closed local source universe."""

    strategy_key = NativeCellProfilerInputDomainStrategyKey.EMBEDDED_IMAGE_PLANES

    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        if _request_has_openhcs_axis_selection(request):
            return False
        modules = CPPipeParser().parse(source.path)
        return bool(compile_image_schema(modules).image_plane_sources)

    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        modules = CPPipeParser().parse(source.path)
        schema = compile_image_schema(modules)
        workspace = materialize_source_schema_workspace(
            request.dataset_path,
            request.output_dir / "native_cellprofiler_source_workspace",
            schema,
        )
        patched_cppipe_path = self.rewrite_embedded_image_plane_sources(
            execution_cppipe_path,
            request.output_dir / "native_cellprofiler_headless",
            tuple(
                self.file_uri((workspace.workspace_root / real_path).resolve())
                for real_path in workspace.primary_mappings.values()
            ),
        )
        input_dir = request.output_dir / "native_cellprofiler_empty_input"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True, exist_ok=True)
        return NativeCellProfilerInputDomain(
            cppipe_path=patched_cppipe_path,
            input_dir=input_dir,
            provenance={
                NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                    self.strategy_key
                ),
                NativeCellProfilerProvenanceField.SOURCE_WORKSPACE: str(
                    workspace.workspace_root
                ),
                NativeCellProfilerProvenanceField.SOURCE_PLANE_COUNT: len(
                    workspace.primary_mappings
                ),
            },
        )

    def rewrite_embedded_image_plane_sources(
        self,
        cppipe_path: Path,
        target_dir: Path,
        source_uris: tuple[str, ...],
    ) -> Path:
        source_text = cppipe_path.read_text(encoding="utf-8")
        lines = source_text.splitlines()
        if not source_uris:
            raise ToolExecutionError(
                "Embedded image-plane native input strategy requires at least one "
                "materialized source mapping."
            )
        patched_lines = self._replace_image_plane_rows(lines, source_uris)
        target_dir.mkdir(parents=True, exist_ok=True)
        patched_path = target_dir / cppipe_path.name
        patched_path.write_text("\n".join(patched_lines) + "\n", encoding="utf-8")
        return patched_path

    def _replace_image_plane_rows(
        self,
        lines: list[str],
        source_uris: tuple[str, ...],
    ) -> list[str]:
        parser = CPPipeParser()
        for index, line in enumerate(lines):
            version_match = parser.IMAGE_PLANE_DETAILS_PATTERN.match(line.strip())
            if version_match is None:
                continue
            count_line = line.replace(
                f'"PlaneCount":"{version_match.group("count")}"',
                f'"PlaneCount":"{len(source_uris)}"',
            )
            header_index = index + 1
            row_start = header_index + 1
            row_stop = row_start + int(version_match.group("count"))
            if header_index >= len(lines) or row_stop > len(lines):
                raise ToolExecutionError("Malformed embedded image-plane table.")
            header = self._csv_image_plane_row(lines[header_index])
            if header[:4] != ["URL", "Series", "Index", "Channel"]:
                raise ToolExecutionError(
                    "Embedded image-plane table has unsupported header "
                    f"{header!r}."
                )
            replacement_rows = [
                f'"{source_uri}",,,'
                for source_uri in source_uris
            ]
            return [
                *lines[:index],
                count_line,
                lines[header_index],
                *replacement_rows,
                *lines[row_stop:],
            ]
        raise ToolExecutionError("Pipeline has no embedded image-plane table to rewrite.")

    @staticmethod
    def file_uri(path: Path) -> str:
        return "file://" + quote(str(path))

    def _csv_image_plane_row(self, line: str) -> list[str]:
        return next(csv.reader([line]))


class DefaultNativeCellProfilerInputDomainStrategy(
    NativeCellProfilerInputDomainStrategy
):
    """Run native CellProfiler against the visible dataset directory."""

    strategy_key = None

    def accepts(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
    ) -> bool:
        del request, source
        return False

    def prepare(
        self,
        request: CellProfilerRunRequest,
        source: CPPipeSourceResolution,
        execution_cppipe_path: Path,
    ) -> NativeCellProfilerInputDomain:
        del source
        return NativeCellProfilerInputDomain(
            cppipe_path=execution_cppipe_path,
            input_dir=request.dataset_path,
            provenance={
                NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY: (
                    NativeCellProfilerInputDomainStrategyKey.DATASET_FOLDER
                )
            },
        )


class CellProfilerAdapter(ToolAdapter):
    """Run a native CellProfiler `.cppipe` as the semantic reference tool."""

    name = "CellProfiler"

    def __init__(self, executable: str | Path | None = None) -> None:
        self._executable_resolver = CellProfilerExecutableResolver(
            Path(executable) if executable is not None else None
        )
        self.version = "unknown"

    def validate_installation(self) -> None:
        """Check that the CellProfiler command-line runner is available."""
        executable = self._cellprofiler_executable()
        try:
            result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ToolNotInstalledError(
                f"CellProfiler executable not found: {executable}"
            ) from exc
        if result.returncode != 0:
            raise ToolExecutionError(
                "Failed to query CellProfiler version:\n"
                + _subprocess_output(result)
            )
        self.version = (result.stdout or result.stderr).strip() or "unknown"

    def run(
        self,
        dataset_path: Path,
        pipeline_name: str,
        pipeline_params: dict[str, Any],
        metrics: list[Any],
        output_dir: Path,
    ) -> BenchmarkResult:
        """Execute a native CellProfiler pipeline headlessly."""
        request = CellProfilerRunRequest(
            dataset_path=Path(dataset_path).resolve(),
            pipeline_name=pipeline_name,
            pipeline_params=dict(pipeline_params),
            metrics=self._validated_metric_collectors(metrics),
            output_dir=Path(output_dir).resolve(),
        )
        request.output_dir.mkdir(parents=True, exist_ok=True)
        phase_timing = PhaseTimingTrace(
            run_id=f"{request.dataset_id}:{request.pipeline_name}:native_cellprofiler",
            pipeline_name=request.pipeline_name,
            tool=self.name,
        )
        with phase_timing.phase(BenchmarkPhase.RESOLVE_SOURCE):
            source = resolve_cppipe_source(request.cppipe_source)
            execution_cppipe_path = HeadlessCellProfilerPipelinePolicy.execution_path(
                source.path,
                request.output_dir,
            )
            native_input_strategy = NativeCellProfilerInputDomainStrategy.select_for(
                request,
                source,
            )
            native_input_domain = native_input_strategy.prepare(
                request,
                source,
                execution_cppipe_path,
            )
        native_output_root = native_cellprofiler_output_root(request)
        if native_output_root.exists():
            shutil.rmtree(native_output_root)
        native_output_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(self._cellprofiler_executable()),
            "-c",
            "-r",
            "-p",
            str(native_input_domain.cppipe_path),
            "-i",
            str(native_input_domain.input_dir),
            "-o",
            str(native_output_root),
        ]
        if request.first_image_set is not None:
            command.extend(("--first-image-set", str(request.first_image_set)))
        if request.last_image_set is not None:
            command.extend(("--last-image-set", str(request.last_image_set)))
        if native_input_domain.file_list_path is not None:
            command.extend(("--file-list", str(native_input_domain.file_list_path)))
        subprocess_env = {
            **environ,
            PYTHONHASHSEED_ENV: environ.get(
                PYTHONHASHSEED_ENV,
                DETERMINISTIC_PYTHONHASHSEED,
            ),
        }

        with ExitStack() as stack:
            for metric in request.metrics:
                stack.enter_context(metric)
            try:
                with phase_timing.phase(BenchmarkPhase.EXECUTE_NATIVE_CP):
                    result = subprocess.run(
                        command,
                        cwd=native_output_root,
                        env=subprocess_env,
                        capture_output=True,
                        text=True,
                        timeout=request.timeout_seconds,
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                raise ToolExecutionError(
                    "Native CellProfiler execution timed out "
                    f"after {request.timeout_seconds}s:\n"
                    + " ".join(command)
                ) from exc
            except FileNotFoundError as exc:
                raise ToolNotInstalledError(
                    f"CellProfiler executable not found: {command[0]}"
                ) from exc
        if result.returncode != 0:
            raise ToolExecutionError(
                "Native CellProfiler execution failed:\n"
                + _subprocess_output(result)
            )

        with phase_timing.phase(BenchmarkPhase.SNAPSHOT_OUTPUTS):
            snapshot = RuntimeOutputSnapshot.from_output_root(native_output_root)
        provenance: dict[str, Any] = {
            "cellprofiler_version": self.version,
            "pipeline_source": "native_cppipe",
            "cppipe_path": str(source.path),
            "execution_cppipe_path": str(native_input_domain.cppipe_path),
            "native_input_dir": str(native_input_domain.input_dir),
            "csv_output_count": len(snapshot.tables),
            "image_output_count": len(snapshot.images),
            "pythonhashseed": subprocess_env[PYTHONHASHSEED_ENV],
            "phase_timing_records": phase_timing.payloads(),
            **native_input_domain.provenance,
        }
        if request.first_image_set is not None:
            provenance[CELLPROFILER_FIRST_IMAGE_SET_PARAM] = request.first_image_set
        if request.last_image_set is not None:
            provenance[CELLPROFILER_LAST_IMAGE_SET_PARAM] = request.last_image_set
        if native_input_domain.file_list_path is not None:
            provenance[NativeCellProfilerProvenanceField.FILE_LIST_PATH] = str(
                native_input_domain.file_list_path
            )
        if source.reference_url is not None:
            provenance["cppipe_reference_url"] = source.reference_url
        _write_native_reference_success_marker(native_output_root, provenance)
        return BenchmarkResult(
            tool_name=self.name,
            dataset_id=request.dataset_id,
            pipeline_name=request.pipeline_name,
            metrics={
                metric.name: metric.get_result()
                for metric in request.metrics
            },
            output_path=native_output_root,
            success=True,
            error_message=None,
            provenance=provenance,
        )

    def _cellprofiler_executable(self) -> Path:
        return self._executable_resolver.resolve()

    def _validated_metric_collectors(
        self,
        metrics: list[Any],
    ) -> tuple[MetricCollector, ...]:
        validated_metrics: list[MetricCollector] = []
        for metric in metrics:
            if not isinstance(metric, MetricCollector):
                raise ToolExecutionError(
                    f"Metric {metric} does not extend MetricCollector"
                )
            validated_metrics.append(metric)
        return tuple(validated_metrics)


def _subprocess_output(result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    return "\n".join(part for part in (stdout, stderr) if part)


def native_cellprofiler_image_set_scope_slug(
    pipeline_params: Mapping[str, Any],
) -> str | None:
    """Return a stable native-reference scope suffix for bounded image-set runs."""
    first_image_set = _optional_positive_int(
        pipeline_params.get(CELLPROFILER_FIRST_IMAGE_SET_PARAM),
        CELLPROFILER_FIRST_IMAGE_SET_PARAM,
    )
    last_image_set = _optional_positive_int(
        pipeline_params.get(CELLPROFILER_LAST_IMAGE_SET_PARAM),
        CELLPROFILER_LAST_IMAGE_SET_PARAM,
    )
    if first_image_set is None and last_image_set is None:
        return None
    parts = []
    if first_image_set is not None:
        parts.append(f"first{first_image_set}")
    if last_image_set is not None:
        parts.append(f"last{last_image_set}")
    return "image_sets_" + "_".join(parts)


def native_cellprofiler_sample_scope_slug(
    pipeline_params: Mapping[str, Any],
) -> str | None:
    """Return a stable native-reference scope suffix for OpenHCS well selection."""
    selection = OpenHCSAxisSelection.from_pipeline_params(pipeline_params)
    parts: list[str] = []
    if selection.axis_filter:
        parts.append("wells_" + "_".join(selection.axis_filter))
    if selection.max_axis_count is not None:
        parts.append(f"first{selection.max_axis_count}wells")
    if not parts:
        return None
    return "samples_" + "_".join(parts)


def native_cellprofiler_reference_scope_slugs(
    *,
    dataset_path: Path,
    pipeline_name: str,
    pipeline_params: Mapping[str, Any],
    output_dir: Path,
) -> tuple[str, ...]:
    """Return native-reference scope suffixes from the selected input domain."""
    request = CellProfilerRunRequest(
        dataset_path=Path(dataset_path),
        pipeline_name=pipeline_name,
        pipeline_params=dict(pipeline_params),
        metrics=(),
        output_dir=Path(output_dir),
    )
    source = resolve_cppipe_source(request.cppipe_source)
    strategy = NativeCellProfilerInputDomainStrategy.select_for(request, source)
    return strategy.reference_scope_slugs(request, source)


def native_cellprofiler_output_root(request: CellProfilerRunRequest) -> Path:
    """Return the output directory owned by one native CellProfiler run."""
    return (
        request.output_dir
        / f"{request.dataset_path.name}_{request.pipeline_name}_native_cellprofiler"
    )


def _optional_positive_int(value: Any, parameter_name: str) -> int | None:
    if value is None:
        return None
    resolved_value = int(value)
    if resolved_value <= 0:
        raise ValueError(f"{parameter_name} must be positive.")
    return resolved_value


def _alpha_slug(value: str) -> str:
    return "".join(chr(ord("a") + int(char, 16)) for char in value.lower())


def _request_has_openhcs_axis_selection(request: CellProfilerRunRequest) -> bool:
    selection = request.openhcs_axis_selection
    return bool(selection.axis_filter) or selection.max_axis_count is not None


def native_cellprofiler_reference_is_complete(reference_output_dir: Path) -> bool:
    """Return whether a native reference has a registered completeness proof."""
    reference = Path(reference_output_dir)
    if (reference / NATIVE_CELLPROFILER_SUCCESS_MARKER).is_file():
        return NativeCellProfilerSuccessMarkerReferenceCompletenessStrategy().is_complete(
            reference
        )
    return any(
        strategy_type().is_complete(reference)
        for strategy_type in NativeCellProfilerReferenceCompletenessStrategy.__registry__.values()
    )


class NativeCellProfilerReferenceCompletenessStrategy(
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal proof that a native CellProfiler reference can be reused."""

    __registry_key__ = "proof_name"
    __skip_if_no_key__ = True
    proof_name: ClassVar[str | None] = None

    @abstractmethod
    def is_complete(self, reference_output_dir: Path) -> bool:
        """Return whether this proof accepts the reference directory."""


class NativeCellProfilerSuccessMarkerReferenceCompletenessStrategy(
    NativeCellProfilerReferenceCompletenessStrategy
):
    """Accept references explicitly marked by the native adapter."""

    proof_name = "success_marker"

    def is_complete(self, reference_output_dir: Path) -> bool:
        marker = reference_output_dir / NATIVE_CELLPROFILER_SUCCESS_MARKER
        if not marker.is_file():
            return False
        provenance = native_cellprofiler_reference_provenance(reference_output_dir)
        cppipe_path = provenance.get("cppipe_path")
        if not isinstance(cppipe_path, str):
            return True
        source_path = Path(cppipe_path)
        if not source_path.exists():
            return True
        modules = CPPipeParser().parse(source_path)
        source_schema = compile_image_schema(modules)
        if not source_schema.image_plane_sources:
            return True
        strategy_key = provenance.get(
            NativeCellProfilerProvenanceField.INPUT_DOMAIN_STRATEGY
        )
        strategy_type = NativeCellProfilerInputDomainStrategy.__registry__.get(
            strategy_key
        )
        if strategy_type is None:
            return False
        return strategy_type().accepts_success_marker_source_schema(
            source_schema,
            provenance,
        )


class NativeCellProfilerSemanticSnapshotReferenceCompletenessStrategy(
    NativeCellProfilerReferenceCompletenessStrategy
):
    """Accept references with loadable semantic output artifacts."""

    proof_name = "semantic_snapshot"

    def is_complete(self, reference_output_dir: Path) -> bool:
        if not reference_output_dir.exists():
            return False
        try:
            snapshot = RuntimeOutputSnapshot.from_output_root(reference_output_dir)
        except (OSError, ValueError):
            return False
        return bool(snapshot.tables or snapshot.images)


def native_cellprofiler_reference_provenance(
    reference_output_dir: Path,
) -> dict[str, Any]:
    """Load successful native-reference provenance, if present."""
    marker = Path(reference_output_dir) / NATIVE_CELLPROFILER_SUCCESS_MARKER
    if not marker.is_file():
        return {}
    payload = json.loads(marker.read_text(encoding="utf-8"))
    provenance = payload.get("provenance")
    return dict(provenance) if isinstance(provenance, dict) else {}


def _write_native_reference_success_marker(
    reference_output_dir: Path,
    provenance: dict[str, Any],
) -> None:
    marker = reference_output_dir / NATIVE_CELLPROFILER_SUCCESS_MARKER
    marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": provenance,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
