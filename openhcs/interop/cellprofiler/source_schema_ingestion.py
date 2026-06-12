"""CellProfiler source-schema workspace ingestion for OpenHCS."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta
from openhcs.constants import Backend
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.core.source_bindings import MetadataSource
from openhcs.core.source_matching import is_image_path, source_filters_match
from openhcs.core.source_schema_workspace import (
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_root", Path(self.source_root))
        object.__setattr__(self, "cppipe_path", Path(self.cppipe_path))
        object.__setattr__(self, "workspace_root", Path(self.workspace_root))
        object.__setattr__(
            self,
            "generated_pipeline_path",
            Path(self.generated_pipeline_path),
        )
        if self.cppipe_path.suffix != ".cppipe":
            raise ValueError(
                "CellProfiler source-schema ingestion requires a .cppipe file, "
                f"got {self.cppipe_path}."
            )
        if self.generated_pipeline_path.suffix != ".py":
            raise ValueError(
                "generated_pipeline_path must point to a Python module, "
                f"got {self.generated_pipeline_path}."
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

        source_root = CellProfilerSourceRootResolver(
            self.request.source_root,
            self.source_schema,
        ).source_root()

        try:
            materialization = materialize_source_schema_workspace(
                source_root,
                self.request.workspace_root,
                self.source_schema,
                filemanager=self.request.filemanager,
                source_backend=self.request.source_backend,
                workspace_backend=self.request.workspace_backend,
                image_set_selection=self.request.image_set_selection,
            )
        except Exception as exc:
            raise CellProfilerSourceWorkspaceMaterializationError(
                f"Failed to materialize CellProfiler source schema for "
                f"{self.request.cppipe_path.name}: {exc}"
            ) from exc
        return source_root, materialization


@dataclass(frozen=True, slots=True)
class CellProfilerSourceRootResolver:
    """Resolve the effective image root for a selected CellProfiler folder."""

    selected_root: Path
    schema: PipelineImageSchema

    def source_root(self) -> Path:
        """Return a child image folder only when the source universe is unambiguous."""
        selected_root = Path(self.selected_root)
        if self._uses_folder_metadata() or not selected_root.exists():
            return selected_root
        buckets = self._admitted_image_buckets(selected_root)
        if selected_root in buckets:
            return selected_root
        if len(buckets) == 1:
            return next(iter(buckets))
        return selected_root

    def _uses_folder_metadata(self) -> bool:
        return any(
            rule.source is MetadataSource.FOLDER_NAME
            for rule in self.schema.metadata_rules
        )

    def _admitted_image_buckets(self, selected_root: Path) -> dict[Path, None]:
        buckets: dict[Path, None] = {}
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
            buckets[bucket_root] = None
        return buckets


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
        if not is_image_path(relative_path):
            return False
        images_rule = self.schema.images_rule
        if images_rule is None:
            return True
        return source_filters_match(relative_path, images_rule.filters)


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
