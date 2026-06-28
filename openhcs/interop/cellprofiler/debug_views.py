"""CellProfiler-specific debug view models over generic OpenHCS snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.debug import DebugSnapshot
from openhcs.core.debug_views import (
    DebugViewModel,
    DebugViewSection,
    DebugViewTable,
    DebugViewTableProjection,
)


@dataclass(frozen=True, slots=True)
class CellProfilerDebugSectionSpec:
    """Declarative section builder for CellProfiler debug views."""

    title: str
    table_factory: Callable[[DebugSnapshot], DebugViewTable | None] | None = None
    text_factory: Callable[[DebugSnapshot], str | None] | None = None

    def section_for(self, snapshot: DebugSnapshot) -> DebugViewSection:
        return DebugViewSection(
            title=self.title,
            table=None if self.table_factory is None else self.table_factory(snapshot),
            text=None if self.text_factory is None else self.text_factory(snapshot),
        )


def summary_table(snapshot: DebugSnapshot) -> DebugViewTable:
    return DebugViewTable(
        columns=("Field", "Value"),
        rows=(
            ("step", snapshot.step_name),
            ("callable", snapshot.callable_name or ""),
            ("axis", snapshot.axis_id or ""),
            ("cursor", snapshot.cursor.invocation_key or ""),
            (
                "timing_seconds",
                "" if snapshot.timing_seconds is None else f"{snapshot.timing_seconds:.6f}",
            ),
        ),
    )


def source_table(snapshot: DebugSnapshot) -> DebugViewTable:
    return DebugViewTable(
        columns=("Source path",),
        rows=tuple((path,) for path in snapshot.source_paths),
    )


def object_output_table(snapshot: DebugSnapshot) -> DebugViewTable:
    return DebugViewTable(
        columns=("Artifact", "Storage ref", "Shape", "DType"),
        rows=tuple(
            row[:1] + row[2:]
            for row in DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
                snapshot.output_artifact_refs
            ).rows
        ),
    )


def artifact_overview_table(snapshot: DebugSnapshot) -> DebugViewTable:
    """Compact artifact-count summary for thumbnail/table-heavy modules."""

    return DebugViewTable(
        columns=("Artifact family", "Count"),
        rows=(
            ("inputs", str(len(snapshot.input_artifact_refs))),
            ("outputs", str(len(snapshot.output_artifact_refs))),
            ("previews", str(len(snapshot.preview_refs))),
            ("measurements", str(len(snapshot.measurement_refs))),
            ("relationships", str(len(snapshot.relationship_refs))),
        ),
    )


def timing_text(snapshot: DebugSnapshot) -> str | None:
    if snapshot.timing_seconds is None:
        return None
    return f"{snapshot.timing_seconds:.6f}s"


DEFAULT_DEBUG_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Summary", table_factory=summary_table),
    CellProfilerDebugSectionSpec("Sources", table_factory=source_table),
    CellProfilerDebugSectionSpec(
        "Input Artifacts",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.input_artifact_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Output Artifacts",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.output_artifact_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Preview Artifacts",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.preview_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Invocation Parameters",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.INVOCATION_PARAMETERS, 
            snapshot.invocation_parameters
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Measurements",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.measurement_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Relationships",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.relationship_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Exception",
        text_factory=lambda snapshot: snapshot.exception,
    ),
)


IDENTIFY_PRIMARY_OBJECTS_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Object Outputs", table_factory=object_output_table),
    CellProfilerDebugSectionSpec(
        "Measurements",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.measurement_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Relationships",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.relationship_refs
        ),
    ),
    CellProfilerDebugSectionSpec("Timing", text_factory=timing_text),
)


MEASUREMENT_MODULE_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Summary", table_factory=summary_table),
    CellProfilerDebugSectionSpec(
        "Measurement Outputs",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.measurement_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Inputs",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.input_artifact_refs
        ),
    ),
    CellProfilerDebugSectionSpec("Timing", text_factory=timing_text),
)


RELATIONSHIP_MODULE_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Summary", table_factory=summary_table),
    CellProfilerDebugSectionSpec(
        "Relationship Outputs",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.relationship_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Measurement Outputs",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.measurement_refs
        ),
    ),
    CellProfilerDebugSectionSpec("Timing", text_factory=timing_text),
)


IMAGE_PROCESSING_MODULE_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Summary", table_factory=summary_table),
    CellProfilerDebugSectionSpec("Artifact Overview", table_factory=artifact_overview_table),
    CellProfilerDebugSectionSpec(
        "Input Images",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.input_artifact_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Output Images",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.output_artifact_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Previews",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.preview_refs
        ),
    ),
    CellProfilerDebugSectionSpec("Timing", text_factory=timing_text),
)


OBJECT_MODULE_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Summary", table_factory=summary_table),
    CellProfilerDebugSectionSpec("Artifact Overview", table_factory=artifact_overview_table),
    CellProfilerDebugSectionSpec("Object Outputs", table_factory=object_output_table),
    CellProfilerDebugSectionSpec(
        "Input Artifacts",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.input_artifact_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Measurement Outputs",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.measurement_refs
        ),
    ),
    CellProfilerDebugSectionSpec("Timing", text_factory=timing_text),
)


DISPLAY_EXPORT_MODULE_SECTION_SPECS = (
    CellProfilerDebugSectionSpec("Summary", table_factory=summary_table),
    CellProfilerDebugSectionSpec("Artifact Overview", table_factory=artifact_overview_table),
    CellProfilerDebugSectionSpec(
        "Available Artifacts",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.input_artifact_refs
            + snapshot.output_artifact_refs
            + snapshot.preview_refs
        ),
    ),
    CellProfilerDebugSectionSpec(
        "Measurement Tables",
        table_factory=lambda snapshot: DebugViewTable.from_projection(DebugViewTableProjection.ARTIFACT_REFS, 
            snapshot.measurement_refs
        ),
    ),
    CellProfilerDebugSectionSpec("Timing", text_factory=timing_text),
)


def build_sections(
    snapshot: DebugSnapshot,
    specs: tuple[CellProfilerDebugSectionSpec, ...],
) -> tuple[DebugViewSection, ...]:
    return tuple(spec.section_for(snapshot) for spec in specs)


def declared_section_specs(
    module_name: str,
) -> tuple[str, tuple[CellProfilerDebugSectionSpec, ...]] | None:
    """Resolve debug sections from the backend module declaration type."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
        CellProfilerDebugViewModule,
        DisplayExportDebugViewModule,
        ImageProcessingDebugViewModule,
        IdentifyPrimaryObjectsDebugViewModule,
        MeasurementDebugViewModule,
        ObjectDebugViewModule,
        RelationshipDebugViewModule,
    )

    module_type = CellProfilerModule.for_module(module_name)
    if module_type is None or not issubclass(module_type, CellProfilerDebugViewModule):
        return None
    if issubclass(module_type, IdentifyPrimaryObjectsDebugViewModule):
        return str(module_type.module_name), IDENTIFY_PRIMARY_OBJECTS_SECTION_SPECS
    if issubclass(module_type, RelationshipDebugViewModule):
        return str(module_type.module_name), RELATIONSHIP_MODULE_SECTION_SPECS
    if issubclass(module_type, MeasurementDebugViewModule):
        return str(module_type.module_name), MEASUREMENT_MODULE_SECTION_SPECS
    if issubclass(module_type, ObjectDebugViewModule):
        return str(module_type.module_name), OBJECT_MODULE_SECTION_SPECS
    if issubclass(module_type, DisplayExportDebugViewModule):
        return str(module_type.module_name), DISPLAY_EXPORT_MODULE_SECTION_SPECS
    if issubclass(module_type, ImageProcessingDebugViewModule):
        return str(module_type.module_name), IMAGE_PROCESSING_MODULE_SECTION_SPECS
    return None


def declared_debug_view_modules() -> tuple[type[object], ...]:
    """Return registered CellProfiler module declarations with debug sections."""
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerDebugViewModule,
        CellProfilerModule,
    )

    return tuple(
        module_type
        for module_type in CellProfilerModule.__registry__.values()
        if module_type.module_name is not None
        and issubclass(module_type, CellProfilerDebugViewModule)
    )


class CellProfilerDebugView(ABC, metaclass=AutoRegisterMeta):
    """Registered CellProfiler renderer for generic debug snapshots."""

    __registry_key__ = "module_name"
    __skip_if_no_key__ = True
    module_name: ClassVar[str | None] = None

    @classmethod
    def for_module(cls, module_name: str | None) -> "CellProfilerDebugView":
        if module_name is not None:
            renderer_type = cls.__registry__.get(module_name)
            if renderer_type is not None:
                return renderer_type()
            declaration = declared_section_specs(module_name)
            if declaration is not None:
                declared_module_name, section_specs = declaration
                return DeclaredCellProfilerDebugView(
                    module_name=declared_module_name,
                    section_specs=section_specs,
                )
        return DefaultCellProfilerDebugView()

    @abstractmethod
    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        """Build a renderer-independent view model for one snapshot."""


class DefaultCellProfilerDebugView(CellProfilerDebugView):
    """Fallback renderer for CellProfiler modules without specialized displays."""

    module_name = "default"

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        return DebugViewModel(
            title=snapshot.callable_name or snapshot.step_name,
            sections=build_sections(snapshot, DEFAULT_DEBUG_SECTION_SPECS),
        )


class SectionSpecCellProfilerDebugView(CellProfilerDebugView):
    """Renderer driven by declarative CellProfiler section specs."""

    __skip__ = True
    section_specs: ClassVar[tuple[CellProfilerDebugSectionSpec, ...]]

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        if self.module_name is None:
            raise RuntimeError("SectionSpecCellProfilerDebugView requires module_name.")
        return DebugViewModel(
            title=self.module_name,
            sections=build_sections(snapshot, self.section_specs),
        )


class DeclaredCellProfilerDebugView(CellProfilerDebugView):
    """Renderer selected from the CellProfiler module declaration type."""

    __skip__ = True

    def __init__(
        self,
        *,
        module_name: str,
        section_specs: tuple[CellProfilerDebugSectionSpec, ...],
    ) -> None:
        self.module_name = module_name
        self.section_specs = section_specs

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        return DebugViewModel(
            title=self.module_name,
            sections=build_sections(snapshot, self.section_specs),
        )


def is_cellprofiler_debug_view_export(name: str, value: object) -> bool:
    return (
        isinstance(value, type)
        and value.__module__ == __name__
        and not name.startswith("_")
    )


__all__ = tuple(
    name
    for name, value in globals().items()
    if is_cellprofiler_debug_view_export(name, value)
)
