"""Read-only architecture projection backed by real OpenHCS symbols."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeAlias

from metaclass_registry import AutoRegisterMeta

from openhcs.agent.dto.architecture import (
    ArchitectureTopic,
    ArchitectureTopicPage,
    ArchitectureTopicSummary,
    InternalApiSymbol,
)
from openhcs.agent.dto.common import SCHEMA_VERSION


InspectableSymbol: TypeAlias = Callable | type


@dataclass(frozen=True, slots=True)
class InternalApiSymbolSpec:
    symbol_id: str
    title: str
    role: str
    symbol: InspectableSymbol
    source_symbol: InspectableSymbol | None = None

    def project(self) -> InternalApiSymbol:
        source_location = _source_location(self.symbol_source_authority())
        return InternalApiSymbol(
            symbol_id=self.symbol_id,
            title=self.title,
            role=self.role,
            import_path=_import_path(self.symbol),
            symbol_kind=_symbol_kind(self.symbol),
            signature=_signature(self.symbol),
            doc_summary=_doc_summary(self.symbol),
            source_path=source_location.source_path,
            line_number=source_location.line_number,
        )

    def symbol_source_authority(self) -> InspectableSymbol:
        return self.source_symbol or self.symbol


class ArchitectureTopicProjection(ABC, metaclass=AutoRegisterMeta):
    """Nominal owner for one source-backed architecture topic."""

    __registry_key__ = "topic_id"
    __skip_if_no_key__ = True

    topic_id: ClassVar[str | None] = None
    title: ClassVar[str]
    summary: ClassVar[str]
    concepts_text: ClassVar[tuple[str, ...]] = ()
    cellprofiler_translation_notes_text: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def projection_types(cls) -> tuple[type["ArchitectureTopicProjection"], ...]:
        return tuple(cls.__registry__.values())

    @classmethod
    def projection_instances(cls) -> tuple["ArchitectureTopicProjection", ...]:
        return tuple(
            projection_type()
            for projection_type in cls.projection_types()
        )

    @classmethod
    def for_topic_id(cls, topic_id: str) -> "ArchitectureTopicProjection":
        normalized = topic_id.casefold()
        for projection in cls.projection_instances():
            if normalized == projection.required_topic_id().casefold():
                return projection
        known = ", ".join(
            projection.required_topic_id()
            for projection in cls.projection_instances()
        )
        raise ValueError(f"Unknown architecture topic {topic_id!r}. Known topics: {known}")

    def summary_dto(self) -> ArchitectureTopicSummary:
        return ArchitectureTopicSummary(
            topic_id=self.required_topic_id(),
            title=self.title,
            summary=self.summary,
        )

    def topic(self) -> ArchitectureTopic:
        return ArchitectureTopic(
            schema_version=SCHEMA_VERSION,
            topic_id=self.required_topic_id(),
            title=self.title,
            summary=self.summary,
            concepts=self.concepts_text,
            cellprofiler_translation_notes=self.cellprofiler_translation_notes_text,
            internal_symbols=tuple(spec.project() for spec in self.symbol_specs()),
        )

    def required_topic_id(self) -> str:
        if self.topic_id is None:
            raise ValueError(f"{type(self).__name__} is missing topic_id")
        return self.topic_id

    @abstractmethod
    def symbol_specs(self) -> tuple[InternalApiSymbolSpec, ...]:
        raise NotImplementedError


class PipelineModelArchitectureTopic(ArchitectureTopicProjection):
    topic_id = "pipeline_model"
    title = "OpenHCS pipeline model"
    summary = (
        "How OpenHCS represents user work as FunctionStep declarations, lazy "
        "configuration, and rendered reviewable source."
    )
    concepts_text = (
        "FunctionStep is the declaration object agents should reason about; MCP drafts FunctionStepSpec DTOs and only later resolves them into FunctionStep objects.",
        "GlobalPipelineConfig owns session defaults; PipelineConfig represents per-pipeline lazy overrides without materializing inherited defaults.",
        "FunctionStepTransportAuthority normalizes callable identity before serialization, compile, or process-boundary transport.",
        "Rendered Python is a review/export artifact, not the canonical MCP state.",
    )

    def symbol_specs(self) -> tuple[InternalApiSymbolSpec, ...]:
        from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
        from openhcs.core.function_step_transport import FunctionStepTransportAuthority
        from openhcs.core.steps.function_step import FunctionStep
        from openhcs.serialization.pycodify_formatters import FunctionStepFormatter

        return (
            InternalApiSymbolSpec(
                "core.FunctionStep",
                "FunctionStep",
                "Pipeline declaration authority for one processing step.",
                FunctionStep,
            ),
            InternalApiSymbolSpec(
                "core.GlobalPipelineConfig",
                "GlobalPipelineConfig",
                "Session-wide default configuration authority.",
                GlobalPipelineConfig,
            ),
            InternalApiSymbolSpec(
                "core.PipelineConfig",
                "PipelineConfig",
                "Per-pipeline lazy override configuration authority.",
                PipelineConfig,
                GlobalPipelineConfig,
            ),
            InternalApiSymbolSpec(
                "core.FunctionStepTransportAuthority",
                "FunctionStepTransportAuthority",
                "Canonical callable normalization before source/process transport.",
                FunctionStepTransportAuthority,
            ),
            InternalApiSymbolSpec(
                "serialization.FunctionStepFormatter",
                "FunctionStepFormatter",
                "Python source rendering authority for FunctionStep declarations.",
                FunctionStepFormatter,
            ),
        )


class CellProfilerTranslationArchitectureTopic(ArchitectureTopicProjection):
    topic_id = "cellprofiler_translation"
    title = "CellProfiler to OpenHCS translation"
    summary = (
        "How .cppipe modules, images, objects, and measurements are parsed and "
        "translated into ordinary public OpenHCS pipeline declarations."
    )
    concepts_text = (
        "CellProfiler .cppipe text is parsed into ordered ModuleBlock records.",
        "CellProfilerModule is the nominal registry whose declarations own module lookup, settings lowering, artifact contracts, callable selection, and processing configuration.",
        "import_cellprofiler_pipeline translates parsed modules into the same FunctionStep and PipelineConfig declarations authored by native OpenHCS pipelines.",
        "The importer derives stack axes, post-stack grouping, previous-step versus pipeline-start main flow, and exact named artifact contracts from source, module, callable, and producer declarations.",
        "The returned public declarations enter the ordinary OpenHCS compile and execution path without a generated-pipeline or retained .cppipe runtime carrier.",
    )
    cellprofiler_translation_notes_text = (
        "A CellProfiler Image name becomes an OpenHCS semantic source binding or runtime image input.",
        "A CellProfiler Object name becomes an OpenHCS object-label runtime value or artifact contract.",
        "A CellProfiler Measure module becomes a backend function whose typed measurement or relationship observations are recorded in RuntimeValueStore independently of file materialization.",
        "CellProfiler SaveImages, ExportToSpreadsheet, and ExportToDatabase are explicit executable FunctionStep declarations; plate-wide table/database exporters use terminal PLATE execution scope.",
        "A .cppipe does not choose native OpenHCS viewer, checkpoint, VFS, or persistence intent that it never declared; inspect the compiled materialization plans separately.",
    )

    def symbol_specs(self) -> tuple[InternalApiSymbolSpec, ...]:
        from openhcs.core.config import GlobalPipelineConfig, PipelineConfig
        from openhcs.core.steps.function_step import FunctionStep
        from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
        from openhcs.interop.cellprofiler.parser import CPPipeParser, ModuleBlock
        from openhcs.interop.cellprofiler.pipeline_import import (
            import_cellprofiler_pipeline,
        )

        return (
            InternalApiSymbolSpec(
                "cellprofiler.CPPipeParser",
                "CPPipeParser",
                "Parses CellProfiler .cppipe text into module records.",
                CPPipeParser,
            ),
            InternalApiSymbolSpec(
                "cellprofiler.ModuleBlock",
                "ModuleBlock",
                "Ordered CellProfiler module and setting record consumed by translation.",
                ModuleBlock,
            ),
            InternalApiSymbolSpec(
                "cellprofiler.CellProfilerModule",
                "CellProfilerModule",
                "Nominal declaration registry for CellProfiler module semantics.",
                CellProfilerModule,
            ),
            InternalApiSymbolSpec(
                "cellprofiler.import_cellprofiler_pipeline",
                "import_cellprofiler_pipeline",
                "Pure translation boundary returning public steps and configuration.",
                import_cellprofiler_pipeline,
            ),
            InternalApiSymbolSpec(
                "core.FunctionStep",
                "FunctionStep",
                "Ordinary public processing-step declaration returned by translation.",
                FunctionStep,
            ),
            InternalApiSymbolSpec(
                "core.PipelineConfig",
                "PipelineConfig",
                "Ordinary public pipeline configuration returned by translation.",
                PipelineConfig,
                GlobalPipelineConfig,
            ),
        )


class SourceSemanticsArchitectureTopic(ArchitectureTopicProjection):
    topic_id = "source_semantics"
    title = "Source bindings and semantic image names"
    summary = (
        "How filenames, metadata, source bindings, and virtual workspace names "
        "preserve image/object semantics."
    )
    concepts_text = (
        "CellProfiler setup modules lower directly into the same SourceBindingsConfig used by native OpenHCS pipelines.",
        "Filename-derived axes and semantic aliases must enter through MetadataExtractionRule and StepSourceBindingsConfig rather than ad hoc parsing at call sites.",
        "Prepared CellProfiler workspaces are logical OpenHCS input workspaces, not native microscope folders.",
    )

    def symbol_specs(self) -> tuple[InternalApiSymbolSpec, ...]:
        from openhcs.core.source_bindings import (
            MetadataExtractionRule,
            SourceBindingsConfig,
            StepSourceBindingsConfig,
        )
        from openhcs.core.source_binding_workspace import (
            SourceBindingWorkspaceProjector,
        )
        from openhcs.interop.cellprofiler.plate_workspace import (
            CellProfilerPlateWorkspacePreparer,
            prepare_cellprofiler_input_workspace,
        )
        return (
            InternalApiSymbolSpec(
                "source.MetadataExtractionRule",
                "MetadataExtractionRule",
                "Declarative filename/metadata extraction rule.",
                MetadataExtractionRule,
            ),
            InternalApiSymbolSpec(
                "source.SourceBindingsConfig",
                "SourceBindingsConfig",
                "Pipeline-level source-universe and semantic binding declaration.",
                SourceBindingsConfig,
            ),
            InternalApiSymbolSpec(
                "source.StepSourceBindingsConfig",
                "StepSourceBindingsConfig",
                "First-class FunctionStep field for semantic input bindings.",
                StepSourceBindingsConfig,
            ),
            InternalApiSymbolSpec(
                "source.SourceBindingWorkspaceProjector",
                "SourceBindingWorkspaceProjector",
                "Projects resolved source bindings into an OpenHCS workspace.",
                SourceBindingWorkspaceProjector,
            ),
            InternalApiSymbolSpec(
                "cellprofiler.CellProfilerPlateWorkspacePreparer",
                "CellProfilerPlateWorkspacePreparer",
                "CellProfiler folder-to-workspace preparation authority.",
                CellProfilerPlateWorkspacePreparer,
            ),
            InternalApiSymbolSpec(
                "cellprofiler.prepare_cellprofiler_input_workspace",
                "prepare_cellprofiler_input_workspace",
                "High-level CellProfiler input workspace preparation function.",
                prepare_cellprofiler_input_workspace,
            ),
        )


class ExecutionRuntimeArchitectureTopic(ArchitectureTopicProjection):
    topic_id = "execution_runtime"
    title = "Compile and execution runtime"
    summary = (
        "How authored pipelines become normalized execution submissions without "
        "exposing live orchestrator objects."
    )
    concepts_text = (
        "MCP should create draft refs and submissions, not expose live PipelineOrchestrator methods.",
        "ZMQ execution payloads are the process boundary authority for compile and run requests.",
        "OpenHCSExecutionSubmission is the client-side nominal payload; ZMQExecutionRequestPayload is the server-side normalized request.",
    )

    def symbol_specs(self) -> tuple[InternalApiSymbolSpec, ...]:
        from openhcs.core.function_step_transport import FunctionStepTransportAuthority
        from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
        from openhcs.runtime.zmq_execution_client import (
            OpenHCSExecutionSubmission,
            ZMQExecutionClient,
        )
        from openhcs.runtime.zmq_execution_signature import ZMQExecutionRequestPayload

        return (
            InternalApiSymbolSpec(
                "core.PipelineOrchestrator",
                "PipelineOrchestrator",
                "Core compile/execution coordinator; exposed to agents only through handles.",
                PipelineOrchestrator,
            ),
            InternalApiSymbolSpec(
                "core.FunctionStepTransportAuthority",
                "FunctionStepTransportAuthority",
                "Normalizes pipeline definitions before execution transport.",
                FunctionStepTransportAuthority,
            ),
            InternalApiSymbolSpec(
                "runtime.OpenHCSExecutionSubmission",
                "OpenHCSExecutionSubmission",
                "Client-side nominal ZMQ execution payload.",
                OpenHCSExecutionSubmission,
            ),
            InternalApiSymbolSpec(
                "runtime.ZMQExecutionRequestPayload",
                "ZMQExecutionRequestPayload",
                "Server-side normalized execution request payload.",
                ZMQExecutionRequestPayload,
            ),
            InternalApiSymbolSpec(
                "runtime.ZMQExecutionClient",
                "ZMQExecutionClient",
                "ZMQ client for compile/run submissions.",
                ZMQExecutionClient,
            ),
        )


class ArchitectureProjectionService:
    """Expose architecture facts without exposing live internal method calls."""

    def list_topics(self) -> ArchitectureTopicPage:
        return ArchitectureTopicPage(
            schema_version=SCHEMA_VERSION,
            topics=tuple(
                projection.summary_dto()
                for projection in ArchitectureTopicProjection.projection_instances()
            ),
        )

    def explain_topic(self, topic_id: str) -> ArchitectureTopic:
        projection = ArchitectureTopicProjection.for_topic_id(topic_id)
        return projection.topic()

    def describe_internal_symbol(self, symbol_id: str) -> InternalApiSymbol:
        for projection in ArchitectureTopicProjection.projection_instances():
            for spec in projection.symbol_specs():
                if spec.symbol_id == symbol_id:
                    return spec.project()
        raise KeyError(f"Unknown OpenHCS architecture symbol_id: {symbol_id}")


@dataclass(frozen=True, slots=True)
class SourceLocation:
    source_path: str
    line_number: int


def _source_location(symbol: InspectableSymbol) -> SourceLocation:
    path = inspect.getsourcefile(symbol)
    if path is None:
        raise ValueError(f"Architecture symbol has no source file: {_import_path(symbol)}")
    _lines, line_number = inspect.getsourcelines(symbol)
    return SourceLocation(
        source_path=_repo_relative_path(Path(path)),
        line_number=line_number,
    )


def _repo_relative_path(path: Path) -> str:
    repo_root = Path(__file__).resolve().parents[3]
    return str(path.resolve().relative_to(repo_root))


def _import_path(symbol: InspectableSymbol) -> str:
    return f"{symbol.__module__}.{symbol.__qualname__}"


def _symbol_kind(symbol: InspectableSymbol) -> str:
    if inspect.isclass(symbol):
        return "class"
    if inspect.isfunction(symbol):
        return "function"
    return "callable"


def _signature(symbol: InspectableSymbol) -> str:
    return str(inspect.signature(symbol))


def _doc_summary(symbol: InspectableSymbol) -> str | None:
    doc = inspect.getdoc(symbol)
    if doc is None:
        return None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None
