"""CellProfiler source-schema workspace ingestion for OpenHCS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.constants.constants import Microscope
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_matching import is_image_path, source_filters_match
from openhcs.core.source_schema_workspace import (
    SourceSchemaCandidateDiscoveryMode,
    SourceSchemaImageSetProbe,
    SourceSchemaImageSetSelection,
    SourceSchemaWorkspaceMaterialization,
    materialize_source_schema_workspace,
)
from openhcs.core.vfs_protocol import FileManagerLike
from openhcs.interop.cellprofiler.parser import CPPipeParser
from openhcs.interop.cellprofiler.runtime_pipeline import (
    PreparedGeneratedPipeline,
    prepare_generated_pipeline,
)
from openhcs.interop.cellprofiler.source_schema import compile_image_schema


CellProfilerSourcePathExclusionTypes: TypeAlias = tuple[
    type["CellProfilerSourcePathExclusion"], ...
]
CellProfilerSourceSchemaMaterializerCallable: TypeAlias = Callable[
    [Path, Path, PipelineImageSchema],
    SourceSchemaWorkspaceMaterialization,
]


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaWorkspaceRequest:
    """Request to prepare a `.cppipe` plus image source as an OpenHCS workspace."""

    source_root: Path
    cppipe_path: Path
    workspace_root: Path
    generated_pipeline_path: Path
    filemanager: FileManagerLike | None = None
    cppipe_filemanager: FileManagerLike | None = None
    generated_pipeline_filemanager: FileManagerLike | None = None
    source_backend: Backend = Backend.DISK
    workspace_backend: Backend = Backend.DISK
    cppipe_backend: Backend = Backend.DISK
    generated_pipeline_backend: Backend = Backend.DISK
    image_set_selection: SourceSchemaImageSetSelection | None = None
    prune_dead_unmaterialized_artifact_steps: bool = False
    materialize_skipped_save_images: bool = True
    materialize_terminal_images: bool = True
    force_materialization: bool = False

    @classmethod
    def from_paths(
        cls,
        *,
        source_root: Path | str,
        cppipe_path: Path | str,
        workspace_root: Path | str,
        generated_pipeline_path: Path | str,
        filemanager: FileManagerLike | None = None,
        cppipe_filemanager: FileManagerLike | None = None,
        generated_pipeline_filemanager: FileManagerLike | None = None,
        source_backend: Backend = Backend.DISK,
        workspace_backend: Backend = Backend.DISK,
        cppipe_backend: Backend = Backend.DISK,
        generated_pipeline_backend: Backend = Backend.DISK,
        image_set_selection: SourceSchemaImageSetSelection | None = None,
        prune_dead_unmaterialized_artifact_steps: bool = False,
        materialize_skipped_save_images: bool = True,
        materialize_terminal_images: bool = True,
        force_materialization: bool = False,
    ) -> "CellProfilerSourceSchemaWorkspaceRequest":
        normalized_cppipe_path = Path(cppipe_path)
        normalized_generated_pipeline_path = Path(generated_pipeline_path)
        cls._validate_pipeline_paths(
            normalized_cppipe_path,
            normalized_generated_pipeline_path,
        )
        return cls(
            source_root=Path(source_root),
            cppipe_path=normalized_cppipe_path,
            workspace_root=Path(workspace_root),
            generated_pipeline_path=normalized_generated_pipeline_path,
            filemanager=filemanager,
            cppipe_filemanager=cppipe_filemanager,
            generated_pipeline_filemanager=generated_pipeline_filemanager,
            source_backend=source_backend,
            workspace_backend=workspace_backend,
            cppipe_backend=cppipe_backend,
            generated_pipeline_backend=generated_pipeline_backend,
            image_set_selection=image_set_selection,
            prune_dead_unmaterialized_artifact_steps=(
                prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=materialize_skipped_save_images,
            materialize_terminal_images=materialize_terminal_images,
            force_materialization=force_materialization,
        )

    def __post_init__(self) -> None:
        self._validate_pipeline_paths(
            Path(self.cppipe_path),
            Path(self.generated_pipeline_path),
        )

    @staticmethod
    def _validate_pipeline_paths(
        cppipe_path: Path,
        generated_pipeline_path: Path,
    ) -> None:
        if cppipe_path.suffix != ".cppipe":
            raise ValueError(
                "CellProfiler source-schema ingestion requires a .cppipe file, "
                f"got {cppipe_path}."
            )
        if generated_pipeline_path.suffix != ".py":
            raise ValueError(
                "generated_pipeline_path must point to a Python module, "
                f"got {generated_pipeline_path}."
            )


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaProjection:
    """Shared execution projection for CellProfiler source-schema workspaces."""

    materialization: SourceSchemaWorkspaceMaterialization | None
    source_root: Path

    @property
    def execution_plate_path(self) -> Path:
        """Return the plate path that should be passed to `PipelineOrchestrator`."""
        if self.materialization is None:
            return self.source_root
        return self.materialization.workspace_root

    @property
    def source_workspace_path(self) -> Path | None:
        """Return materialized workspace root when source-schema projection was needed."""
        if self.materialization is None:
            return None
        return self.materialization.workspace_root


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaWorkspace(CellProfilerSourceSchemaProjection):
    """Prepared CellProfiler pipeline plus optional OpenHCS source workspace."""

    prepared_pipeline: PreparedGeneratedPipeline

    @property
    def runtime_pipeline(self) -> Pipeline:
        """Return the executable OpenHCS pipeline for this prepared workspace."""
        return self.prepared_pipeline.runtime_pipeline

    @property
    def runtime_pipeline_steps(self) -> Sequence[Any]:
        """Return executable OpenHCS steps without exposing generation internals."""
        return self.prepared_pipeline.runtime_pipeline_steps


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaPreparation(CellProfilerSourceSchemaProjection):
    """Source-schema-only workspace prepared from CellProfiler setup modules."""

    source_schema: PipelineImageSchema
    cppipe_path: Path


class CellProfilerSourceSchemaIngestionError(RuntimeError):
    """Base error for CellProfiler source-schema ingestion failures."""


class CellProfilerPipelinePreparationError(CellProfilerSourceSchemaIngestionError):
    """Raised when `.cppipe` conversion fails before source workspace creation."""


class CellProfilerSourceWorkspaceMaterializationError(
    CellProfilerSourceSchemaIngestionError
):
    """Raised when source-schema workspace materialization fails."""


def prepare_cellprofiler_source_schema_workspace(
    request: CellProfilerSourceSchemaWorkspaceRequest,
) -> CellProfilerSourceSchemaWorkspace:
    """Prepare a CellProfiler pipeline and source-schema OpenHCS workspace."""
    try:
        prepared = prepare_generated_pipeline(
            request.cppipe_path,
            output_path=request.generated_pipeline_path,
            prune_dead_unmaterialized_artifact_steps=(
                request.prune_dead_unmaterialized_artifact_steps
            ),
            materialize_skipped_save_images=request.materialize_skipped_save_images,
            materialize_terminal_images=request.materialize_terminal_images,
            cppipe_filemanager=request.cppipe_filemanager,
            generated_pipeline_filemanager=request.generated_pipeline_filemanager,
            cppipe_backend=request.cppipe_backend,
            generated_pipeline_backend=request.generated_pipeline_backend,
        )
    except ValueError as exc:
        raise CellProfilerPipelinePreparationError(
            f"Failed to prepare converted .cppipe pipeline "
            f"{request.cppipe_path.name}: {exc}"
        ) from exc

    if prepared.source_schema.is_empty:
        return CellProfilerSourceSchemaWorkspace(
            materialization=None,
            source_root=request.source_root,
            prepared_pipeline=prepared,
        )
    pipeline_config = prepared.generated_pipeline.pipeline_config
    if (
        not request.force_materialization
        and
        pipeline_config is not None
        and pipeline_config.microscope is Microscope.SOURCE_BINDINGS
    ):
        return CellProfilerSourceSchemaWorkspace(
            materialization=None,
            source_root=request.source_root,
            prepared_pipeline=prepared,
        )
    source_root, materialization = CellProfilerSourceSchemaMaterializer(
        request=request,
        source_schema=prepared.source_schema,
    ).materialize()

    return CellProfilerSourceSchemaWorkspace(
        materialization=materialization,
        source_root=source_root,
        prepared_pipeline=prepared,
    )


def prepare_cellprofiler_source_schema_only_workspace(
    request: CellProfilerSourceSchemaWorkspaceRequest,
) -> CellProfilerSourceSchemaPreparation:
    """Prepare only source workspace semantics from CellProfiler setup modules."""
    source_schema = compile_cellprofiler_source_schema(request)

    if source_schema.is_empty:
        return CellProfilerSourceSchemaPreparation(
            materialization=None,
            source_root=request.source_root,
            source_schema=source_schema,
            cppipe_path=request.cppipe_path,
        )

    source_root, materialization = CellProfilerSourceSchemaMaterializer(
        request=request,
        source_schema=source_schema,
    ).materialize()

    return CellProfilerSourceSchemaPreparation(
        materialization=materialization,
        source_root=source_root,
        source_schema=source_schema,
        cppipe_path=request.cppipe_path,
    )


def compile_cellprofiler_source_schema(
    request: CellProfilerSourceSchemaWorkspaceRequest,
) -> PipelineImageSchema:
    """Compile CellProfiler setup modules into source schema without materializing."""

    try:
        modules = CPPipeParser().parse(
            request.cppipe_path,
            filemanager=request.cppipe_filemanager,
            backend=request.cppipe_backend,
        )
    except ValueError as exc:
        raise CellProfilerPipelinePreparationError(
            f"Failed to parse CellProfiler source schema "
            f"{request.cppipe_path.name}: {exc}"
        ) from exc
    return compile_image_schema(modules)


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaMaterializer:
    """Authority for materializing a CellProfiler source schema."""

    request: CellProfilerSourceSchemaWorkspaceRequest
    source_schema: PipelineImageSchema

    def materialize(self) -> tuple[Path, SourceSchemaWorkspaceMaterialization]:
        """Materialize source images and return their effective source root."""

        resolved_source = CellProfilerSourceRootResolver(
            self.request.source_root,
            self.source_schema,
        ).source_root()

        try:
            materialization = (
                CellProfilerSourceSchemaMaterializationScope.from_request(
                    self.request
                ).materialize(
                    source_root=resolved_source.root,
                    workspace_root=self.request.workspace_root,
                    source_schema=self.source_schema,
                    source_files=resolved_source.source_files,
                )
            )
        except Exception as exc:
            raise CellProfilerSourceWorkspaceMaterializationError(
                f"Failed to materialize CellProfiler source schema for "
                f"{self.request.cppipe_path.name}: {exc}"
            ) from exc
        return resolved_source.root, materialization


@dataclass(frozen=True, slots=True)
class CellProfilerSourceSchemaMaterializationScope:
    """Nominal source-scope carrier for source-schema workspace materialization."""

    filemanager: FileManagerLike | None
    source_backend: Backend
    workspace_backend: Backend
    image_set_selection: SourceSchemaImageSetSelection | None

    @classmethod
    def from_request(
        cls,
        request: CellProfilerSourceSchemaWorkspaceRequest,
    ) -> "CellProfilerSourceSchemaMaterializationScope":
        return cls(
            filemanager=request.filemanager,
            source_backend=request.source_backend,
            workspace_backend=request.workspace_backend,
            image_set_selection=request.image_set_selection,
        )

    def materialize(
        self,
        *,
        source_root: Path,
        workspace_root: Path,
        source_schema: PipelineImageSchema,
        source_files: Sequence[Path] | None = None,
    ) -> SourceSchemaWorkspaceMaterialization:
        return self.bound_materializer()(
            source_root,
            workspace_root,
            source_schema,
            source_files=source_files,
        )

    def bound_materializer(self) -> CellProfilerSourceSchemaMaterializerCallable:
        return partial(
            materialize_source_schema_workspace,
            filemanager=self.filemanager,
            source_backend=self.source_backend,
            workspace_backend=self.workspace_backend,
            image_set_selection=self.image_set_selection,
            candidate_discovery_mode=SourceSchemaCandidateDiscoveryMode.LOCAL_FILES,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerSourceRootResolver:
    """Resolve the effective image root for a selected CellProfiler folder."""

    selected_root: Path
    schema: PipelineImageSchema

    def source_root(self) -> "CellProfilerResolvedSourceRoot":
        """Return a child image folder only when the source universe is unambiguous."""
        selected_root = Path(self.selected_root)
        if not selected_root.exists():
            return CellProfilerResolvedSourceRoot(selected_root, ())
        fallback_candidates = self._source_root_candidates(selected_root)
        for candidate_root in self._candidate_search_roots(selected_root):
            usable_candidates = tuple(
                candidate
                for candidate in self._source_root_candidates(candidate_root)
                if candidate.usable_for(self.schema)
            )
            if len(usable_candidates) == 1:
                return usable_candidates[0].resolved_root()
        if len(fallback_candidates) == 1:
            return fallback_candidates[0].resolved_root()
        return CellProfilerResolvedSourceRoot(
            selected_root,
            tuple(
                path
                for candidate in fallback_candidates
                for path in candidate.source_files
            ),
        )

    def _candidate_search_roots(self, selected_root: Path) -> tuple[Path, ...]:
        roots = [selected_root]
        if not any(selected_root.glob("*.cppipe")):
            return tuple(roots)
        for parent in selected_root.parents:
            if any(parent.glob("*.cppipe")):
                roots.append(parent)
                break
        return tuple(roots)

    def _source_root_candidates(
        self,
        selected_root: Path,
    ) -> tuple["CellProfilerSourceRootCandidate", ...]:
        source_files_by_root: dict[Path, list[Path]] = {}
        admission = CellProfilerSourcePathAdmission(
            selected_root=selected_root,
            schema=self.schema,
        )
        for path in sorted(selected_root.rglob("*")):
            if not path.is_file():
                continue
            bucket_root = admission.bucket_for(path)
            if bucket_root is None:
                continue
            source_files = source_files_by_root.get(bucket_root)
            if source_files is None:
                source_files_by_root[bucket_root] = [path]
                continue
            source_files.append(path)
        return tuple(
            CellProfilerSourceRootCandidate(
                root=root,
                source_files=tuple(paths),
            )
            for root, paths in sorted(source_files_by_root.items())
        )


@dataclass(frozen=True, slots=True)
class CellProfilerSourceRootCandidate:
    """One candidate source root and the files admitted under that root."""

    root: Path
    source_files: tuple[Path, ...]

    def usable_for(self, schema: PipelineImageSchema) -> bool:
        return SourceSchemaImageSetProbe(
            source_root=self.root,
            source_files=self.source_files,
            schema=schema,
            discovery_mode=SourceSchemaCandidateDiscoveryMode.LOCAL_FILES,
        ).result().usable

    def resolved_root(self) -> "CellProfilerResolvedSourceRoot":
        return CellProfilerResolvedSourceRoot(
            root=self.root,
            source_files=self.source_files,
        )


@dataclass(frozen=True, slots=True)
class CellProfilerResolvedSourceRoot:
    """Resolved CellProfiler image root plus the admitted source file universe."""

    root: Path
    source_files: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class CellProfilerSourcePathAdmission:
    """Single authority for admitting files into a CellProfiler source root bucket."""

    selected_root: Path
    schema: PipelineImageSchema
    exclusion_policies: Iterable[type["CellProfilerSourcePathExclusion"]] = ()

    def bucket_for(self, path: Path) -> Path | None:
        context = CellProfilerSourcePathContext(
            selected_root=self.selected_root,
            path=path,
        )
        if any(policy().excludes(context) for policy in self.policies()):
            return None
        if not self._schema_admits_path(context.relative_to_selected.as_posix()):
            return None
        if not self._schema_admits_path(context.relative_to_bucket.as_posix()):
            return None
        return context.bucket_root

    def policies(self) -> CellProfilerSourcePathExclusionTypes:
        if self.exclusion_policies:
            return tuple(self.exclusion_policies)
        return CellProfilerSourcePathExclusion.ordered()

    def _schema_admits_path(self, relative_path: str) -> bool:
        if self._schema_admits_source_artifact(relative_path):
            return True
        if not is_image_path(relative_path):
            return False
        images_rule = self.schema.images_rule
        if images_rule is None:
            return True
        return source_filters_match(relative_path, images_rule.filters)

    def _schema_admits_source_artifact(self, relative_path: str) -> bool:
        return any(
            source_filters_match(relative_path, assignment.selector.filters)
            for assignment in self.schema.source_artifacts_by_alias.values()
        )


@dataclass(frozen=True, slots=True)
class CellProfilerSourcePathContext:
    """Path coordinates used by CellProfiler source-root admission policies."""

    selected_root: Path
    path: Path

    @property
    def relative_to_selected(self) -> Path:
        return self.path.relative_to(self.selected_root)

    @property
    def bucket_root(self) -> Path:
        relative_path = self.relative_to_selected
        if len(relative_path.parts) <= 1:
            return self.selected_root
        return self.selected_root / relative_path.parts[0]

    @property
    def relative_to_bucket(self) -> Path:
        return self.path.relative_to(self.bucket_root)


class CellProfilerSourcePathExclusion(ABC, metaclass=AutoRegisterMeta):
    """Nominal family for paths excluded from CellProfiler source-root inference."""

    __registry_key__ = "policy_key"
    __skip_if_no_key__ = True
    policy_key: ClassVar[str | None] = None

    @classmethod
    def ordered(cls) -> CellProfilerSourcePathExclusionTypes:
        registered = set(cls.__registry__.values())
        ordered: list[type[CellProfilerSourcePathExclusion]] = []
        seen: set[type[CellProfilerSourcePathExclusion]] = set()

        def visit(owner: type[CellProfilerSourcePathExclusion]) -> None:
            for child in owner.__subclasses__():
                visit(child)
            if owner in registered and owner not in seen:
                ordered.append(owner)
                seen.add(owner)

        visit(cls)
        return tuple(ordered)

    @abstractmethod
    def excludes(self, context: CellProfilerSourcePathContext) -> bool:
        """Return whether the path is outside the inferred source image universe."""


class ControlDirectorySourcePathExclusion(CellProfilerSourcePathExclusion):
    """Exclude hidden/control folders that are not CellProfiler image inputs."""

    policy_key = "control_directory"

    def excludes(self, context: CellProfilerSourcePathContext) -> bool:
        return any(
            part.startswith(".") or part == "__MACOSX"
            for part in context.relative_to_selected.parts
        )


class NestedPipelineRootSourcePathExclusion(CellProfilerSourcePathExclusion):
    """Exclude nested folders that declare their own CellProfiler pipeline root."""

    policy_key = "nested_pipeline_root"

    def excludes(self, context: CellProfilerSourcePathContext) -> bool:
        return (
            context.bucket_root != context.selected_root
            and any(context.bucket_root.glob("*.cppipe"))
        )
