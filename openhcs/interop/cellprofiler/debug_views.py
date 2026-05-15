"""CellProfiler-specific debug view models over generic OpenHCS snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, ClassVar, Mapping

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


@dataclass(frozen=True, slots=True)
class CellProfilerModuleDebugRendererSpec:
    """Declarative renderer assignment for one CellProfiler module family."""

    module_names: tuple[str, ...]
    section_specs: tuple[CellProfilerDebugSectionSpec, ...]


CELLPROFILER_MODULE_DEBUG_RENDERER_SPECS = (
    CellProfilerModuleDebugRendererSpec(
        module_names=(
            "IdentifySecondaryObjects",
            "IdentifyTertiaryObjects",
            "MaskObjects",
            "FilterObjects",
            "ExpandOrShrinkObjects",
            "ConvertObjectsToImage",
            "ConvertImageToObjects",
        ),
        section_specs=OBJECT_MODULE_SECTION_SPECS,
    ),
    CellProfilerModuleDebugRendererSpec(
        module_names=(
            "MeasureObjectIntensity",
            "MeasureObjectIntensityDistribution",
            "MeasureColocalization",
            "MeasureGranularity",
            "MeasureImageAreaOccupied",
            "MeasureImageQuality",
            "MeasureObjectNeighbors",
            "MeasureTexture",
        ),
        section_specs=MEASUREMENT_MODULE_SECTION_SPECS,
    ),
    CellProfilerModuleDebugRendererSpec(
        module_names=(
            "CalculateMath",
            "ClassifyObjects",
            "MeasureCorrelation",
            "MeasureImageOverlap",
            "ScoreAll",
        ),
        section_specs=MEASUREMENT_MODULE_SECTION_SPECS,
    ),
    CellProfilerModuleDebugRendererSpec(
        module_names=(
            "Resize",
            "RescaleIntensity",
            "Smooth",
            "GaussianFilter",
            "MedianFilter",
            "ReduceNoise",
            "Threshold",
            "MaskImage",
            "Crop",
            "ColorToGray",
            "GrayToColor",
            "CorrectIlluminationCalculate",
            "CorrectIlluminationApply",
            "EnhanceEdges",
            "EnhanceOrSuppressFeatures",
            "InvertForPrinting",
            "Morph",
            "OverlayOutlines",
            "SaveCroppedObjects",
            "UnmixColors",
        ),
        section_specs=IMAGE_PROCESSING_MODULE_SECTION_SPECS,
    ),
    CellProfilerModuleDebugRendererSpec(
        module_names=(
            "DisplayDataOnImage",
            "DisplayHistogram",
            "DisplayPlatemap",
            "DisplayScatterPlot",
            "DisplayDensityPlot",
            "ExportToDatabase",
            "ExportToSpreadsheet",
            "SaveImages",
        ),
        section_specs=DISPLAY_EXPORT_MODULE_SECTION_SPECS,
    ),
)


def build_table_driven_renderer_specs() -> Mapping[
    str,
    tuple[CellProfilerDebugSectionSpec, ...],
]:
    return {
        module_name: spec.section_specs
        for spec in CELLPROFILER_MODULE_DEBUG_RENDERER_SPECS
        for module_name in spec.module_names
    }


TABLE_DRIVEN_RENDERER_SPECS = build_table_driven_renderer_specs()


def build_sections(
    snapshot: DebugSnapshot,
    specs: tuple[CellProfilerDebugSectionSpec, ...],
) -> tuple[DebugViewSection, ...]:
    return tuple(spec.section_for(snapshot) for spec in specs)


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
            section_specs = TABLE_DRIVEN_RENDERER_SPECS.get(module_name)
            if section_specs is not None:
                return TableDrivenCellProfilerDebugView(
                    module_name=module_name,
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


class IdentifyPrimaryObjectsDebugView(CellProfilerDebugView):
    """Specialized summary for IdentifyPrimaryObjects snapshots."""

    module_name = "IdentifyPrimaryObjects"

    def build_view_model(self, snapshot: DebugSnapshot) -> DebugViewModel:
        return DebugViewModel(
            title="IdentifyPrimaryObjects",
            sections=build_sections(snapshot, IDENTIFY_PRIMARY_OBJECTS_SECTION_SPECS),
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


class TableDrivenCellProfilerDebugView(CellProfilerDebugView):
    """Renderer selected from the CellProfiler module-category spec table."""

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


@dataclass(frozen=True, slots=True)
class CellProfilerSectionRendererDeclaration:
    """Named specialized renderer declaration for one CellProfiler module."""

    class_name: str
    module_name: str
    section_specs: tuple[CellProfilerDebugSectionSpec, ...]


def declare_section_spec_debug_view(
    declaration: CellProfilerSectionRendererDeclaration,
) -> type[SectionSpecCellProfilerDebugView]:
    return type(
        declaration.class_name,
        (SectionSpecCellProfilerDebugView,),
        {
            "__module__": __name__,
            "__doc__": f"Specialized summary for {declaration.module_name} snapshots.",
            "module_name": declaration.module_name,
            "section_specs": declaration.section_specs,
        },
    )


for _renderer_declaration in (
    CellProfilerSectionRendererDeclaration(
        "MeasureImageIntensityDebugView",
        "MeasureImageIntensity",
        MEASUREMENT_MODULE_SECTION_SPECS,
    ),
    CellProfilerSectionRendererDeclaration(
        "MeasureObjectSizeShapeDebugView",
        "MeasureObjectSizeShape",
        MEASUREMENT_MODULE_SECTION_SPECS,
    ),
    CellProfilerSectionRendererDeclaration(
        "RelateObjectsDebugView",
        "RelateObjects",
        RELATIONSHIP_MODULE_SECTION_SPECS,
    ),
    CellProfilerSectionRendererDeclaration(
        "TrackObjectsDebugView",
        "TrackObjects",
        RELATIONSHIP_MODULE_SECTION_SPECS,
    ),
    CellProfilerSectionRendererDeclaration(
        "ImageMathDebugView",
        "ImageMath",
        IMAGE_PROCESSING_MODULE_SECTION_SPECS,
    ),
    CellProfilerSectionRendererDeclaration(
        "CorrectIlluminationApplyDebugView",
        "CorrectIlluminationApply",
        IMAGE_PROCESSING_MODULE_SECTION_SPECS,
    ),
):
    globals()[_renderer_declaration.class_name] = declare_section_spec_debug_view(
        _renderer_declaration
    )
del _renderer_declaration


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
