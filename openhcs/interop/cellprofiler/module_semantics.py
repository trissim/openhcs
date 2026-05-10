"""Nominal CellProfiler module semantics from the CellProfiler 4.2 manual."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class CellProfilerModuleCategory(Enum):
    """Manual-facing CellProfiler module category."""

    INPUT = ("Input", True)
    FILE_PROCESSING = ("File Processing", True)
    IMAGE_PROCESSING = ("Image Processing", False)
    OBJECT_PROCESSING = ("Object Processing", False)
    MEASUREMENT = ("Measurement", False)
    ADVANCED = ("Advanced", False)
    WORM_TOOLBOX = ("Worm Toolbox", False)
    OTHER = ("Other", False)
    DATA_TOOLS = ("Data Tools", False)

    def __new__(cls, label: str, is_infrastructure: bool):
        obj = object.__new__(cls)
        obj._value_ = label
        obj._is_infrastructure = is_infrastructure
        return obj

    @property
    def is_infrastructure(self) -> bool:
        """Return whether OpenHCS handles this category as infrastructure."""
        return self._is_infrastructure


class CellProfilerModuleDimensionality(Enum):
    """Manual-facing dimensional support mapped onto OpenHCS execution contracts."""

    PLANAR = ProcessingContract.PURE_2D
    VOLUMETRIC = ProcessingContract.PURE_3D
    PLANAR_AND_VOLUMETRIC = ProcessingContract.FLEXIBLE

    @property
    def processing_contract(self) -> ProcessingContract:
        """Return the closest OpenHCS processing contract for this support mode."""
        return self.value

    @property
    def supports_2d(self) -> bool:
        return self is not CellProfilerModuleDimensionality.VOLUMETRIC

    @property
    def supports_3d(self) -> bool:
        return self is not CellProfilerModuleDimensionality.PLANAR


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSemanticTraits(metaclass=AutoRegisterMeta):
    """Shared semantic traits for CellProfiler module declarations."""

    __registry_key__ = "registry_key"
    __skip_if_no_key__ = True

    registry_key: ClassVar[str | None] = None
    category: CellProfilerModuleCategory
    dimensionality: CellProfilerModuleDimensionality
    respects_masks: bool

    @property
    def supports_2d(self) -> bool:
        """Return CellProfiler manual 2D support derived from dimensionality."""
        return self.dimensionality.supports_2d

    @property
    def supports_3d(self) -> bool:
        """Return CellProfiler manual 3D support derived from dimensionality."""
        return self.dimensionality.supports_3d

    @property
    def is_infrastructure(self) -> bool:
        """Return whether OpenHCS handles this as runtime/file infrastructure."""
        return self.category.is_infrastructure


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSemantics(CellProfilerModuleSemanticTraits):
    """Typed manual semantics for one CellProfiler module."""

    registry_key: ClassVar[str] = "module"
    module_name: str


def cellprofiler_module_semantics(
    module_name: str,
) -> CellProfilerModuleSemantics | None:
    """Return manual semantics for a CellProfiler module name, if known."""
    normalized_name = module_name.strip()
    if not normalized_name:
        raise ValueError("CellProfiler module name cannot be empty.")
    return CELLPROFILER_MODULE_SEMANTICS_BY_KEY.get(normalized_name.casefold())


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSemanticsFamilySpec(CellProfilerModuleSemanticTraits):
    """Authoritative declaration for a family of equivalent module semantics."""

    registry_key: ClassVar[str] = "family"
    family_name: str
    module_names: tuple[str, ...]

    def __init__(
        self,
        family_name: str,
        category: CellProfilerModuleCategory,
        dimensionality: CellProfilerModuleDimensionality,
        respects_masks: bool,
        module_names: Iterable[str],
    ) -> None:
        object.__setattr__(self, "family_name", family_name)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "dimensionality", dimensionality)
        object.__setattr__(self, "respects_masks", respects_masks)
        object.__setattr__(self, "module_names", tuple(module_names))

    def declared_semantics(self) -> tuple[CellProfilerModuleSemantics, ...]:
        """Materialize all module declarations in this family."""
        return tuple(
            CellProfilerModuleSemantics(
                module_name=module_name,
                category=self.category,
                dimensionality=self.dimensionality,
                respects_masks=self.respects_masks,
            )
            for module_name in self.module_names
        )


@dataclass(frozen=True, slots=True)
class CellProfilerModuleAliasFamilySpec:
    """Authoritative declaration for CellProfiler legacy module-name aliases."""

    family_name: str
    aliases: tuple[tuple[str, str], ...]


def _registered_semantics() -> tuple[CellProfilerModuleSemantics, ...]:
    return tuple(
        semantics
        for family in CELLPROFILER_MODULE_SEMANTICS_FAMILY_SPECS
        for semantics in family.declared_semantics()
    )


def _registered_aliases() -> tuple[tuple[str, str], ...]:
    return tuple(
        alias
        for family in CELLPROFILER_MODULE_ALIAS_FAMILY_SPECS
        for alias in family.aliases
    )


_C = CellProfilerModuleCategory
_D = CellProfilerModuleDimensionality
CELLPROFILER_MODULE_SEMANTICS_FAMILY_SPECS = (
    CellProfilerModuleSemanticsFamilySpec(
        "InputUnmasked",
        _C.INPUT,
        _D.PLANAR_AND_VOLUMETRIC,
        False,
        (
            "Images",
            "LoadImages",
            "Metadata",
            "Groups",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "InputMasked", _C.INPUT, _D.PLANAR_AND_VOLUMETRIC, True, ("NamesAndTypes",)
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "FileProcessingUnmasked3D",
        _C.FILE_PROCESSING,
        _D.PLANAR_AND_VOLUMETRIC,
        False,
        ("CreateBatchFiles",),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "FileProcessingMasked3D",
        _C.FILE_PROCESSING,
        _D.PLANAR_AND_VOLUMETRIC,
        True,
        ("ExportToDatabase", "ExportToSpreadsheet", "SaveCroppedObjects", "SaveImages"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "FileProcessingUnmasked2D",
        _C.FILE_PROCESSING,
        _D.PLANAR,
        False,
        ("LabelImages",),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "FileProcessingMasked2D",
        _C.FILE_PROCESSING,
        _D.PLANAR,
        True,
        ("LoadData",),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ImageProcessingUnmasked2D",
        _C.IMAGE_PROCESSING,
        _D.PLANAR,
        False,
        (
            "ColorToGray",
            "FlipAndRotate",
            "GrayToColor",
            "InvertForPrinting",
            "Tile",
            "UnmixColors",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ImageProcessingMasked2D",
        _C.IMAGE_PROCESSING,
        _D.PLANAR,
        True,
        (
            "Align",
            "CorrectIlluminationCalculate",
            "Crop",
            "EnhanceEdges",
            "MakeProjection",
            "Morph",
            "OverlayObjects",
            "Smooth",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ImageProcessingUnmasked2DApply",
        _C.IMAGE_PROCESSING,
        _D.PLANAR,
        False,
        ("CorrectIlluminationApply",),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ImageProcessingMasked3D",
        _C.IMAGE_PROCESSING,
        _D.PLANAR_AND_VOLUMETRIC,
        True,
        (
            "EnhanceOrSuppressFeatures",
            "ImageMath",
            "MaskImage",
            "RescaleIntensity",
            "Resize",
            "Threshold",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ImageProcessingUnmasked3D",
        _C.IMAGE_PROCESSING,
        _D.PLANAR_AND_VOLUMETRIC,
        False,
        ("OverlayOutlines",),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ObjectProcessingUnmasked2D",
        _C.OBJECT_PROCESSING,
        _D.PLANAR,
        False,
        ("ClassifyObjects", "IdentifyObjectsManually"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ObjectProcessingMasked2D",
        _C.OBJECT_PROCESSING,
        _D.PLANAR,
        True,
        (
            "EditObjectsManually",
            "ExpandOrShrinkObjects",
            "IdentifyObjectsInGrid",
            "IdentifyPrimaryObjects",
            "IdentifySecondaryObjects",
            "IdentifyTertiaryObjects",
            "MaskObjects",
            "SplitOrMergeObjects",
            "TrackObjects",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ObjectProcessingUnmasked3D",
        _C.OBJECT_PROCESSING,
        _D.PLANAR_AND_VOLUMETRIC,
        False,
        ("CombineObjects", "ConvertImageToObjects", "ResizeObjects"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "ObjectProcessingMasked3D",
        _C.OBJECT_PROCESSING,
        _D.PLANAR_AND_VOLUMETRIC,
        True,
        ("ConvertObjectsToImage", "FilterObjects", "RelateObjects"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "MeasurementMasked3D",
        _C.MEASUREMENT,
        _D.PLANAR_AND_VOLUMETRIC,
        True,
        (
            "MeasureColocalization",
            "MeasureGranularity",
            "MeasureImageAreaOccupied",
            "MeasureImageIntensity",
            "MeasureImageOverlap",
            "MeasureImageQuality",
            "MeasureImageSkeleton",
            "MeasureObjectIntensity",
            "MeasureTexture",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "MeasurementMasked2D",
        _C.MEASUREMENT,
        _D.PLANAR,
        True,
        ("MeasureObjectIntensityDistribution",),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "MeasurementUnmasked3D",
        _C.MEASUREMENT,
        _D.PLANAR_AND_VOLUMETRIC,
        False,
        ("MeasureObjectNeighbors", "MeasureObjectSizeShape"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "MeasurementUnmasked2D",
        _C.MEASUREMENT,
        _D.PLANAR,
        False,
        ("MeasureObjectOverlap", "MeasureObjectSkeleton"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "AdvancedUnmasked3D",
        _C.ADVANCED,
        _D.PLANAR_AND_VOLUMETRIC,
        False,
        (
            "Closing",
            "DilateImage",
            "DilateObjects",
            "ErodeImage",
            "ErodeObjects",
            "FillObjects",
            "GaussianFilter",
            "MedialAxis",
            "MedianFilter",
            "MorphologicalSkeleton",
            "Opening",
            "ReduceNoise",
            "RemoveHoles",
            "ShrinkToObjectCenters",
        ),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "AdvancedUnmasked2D",
        _C.ADVANCED,
        _D.PLANAR,
        False,
        ("MatchTemplate", "RunImageJMacro"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "AdvancedMasked3D", _C.ADVANCED, _D.PLANAR_AND_VOLUMETRIC, True, ("Watershed",)
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "WormToolboxMasked2D",
        _C.WORM_TOOLBOX,
        _D.PLANAR,
        True,
        ("IdentifyDeadWorms", "StraightenWorms", "UntangleWorms"),
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "OtherUnmasked2D", _C.OTHER, _D.PLANAR, False, ("DefineGrid",)
    ),
    CellProfilerModuleSemanticsFamilySpec(
        "DataToolsUnmasked2D",
        _C.DATA_TOOLS,
        _D.PLANAR,
        False,
        (
            "CalculateMath",
            "CalculateStatistics",
            "DisplayDataOnImage",
            "DisplayDensityPlot",
            "DisplayHistogram",
            "DisplayPlatemap",
            "DisplayScatterPlot",
            "FindMaxima",
            "FlagImage",
        ),
    ),
)

CELLPROFILER_MODULE_ALIAS_FAMILY_SPECS = (
    CellProfilerModuleAliasFamilySpec(
        "LegacyModuleName",
        (
            ("ClassifyObjectsSingleMeasurement", "ClassifyObjects"),
            ("Combineobjects", "CombineObjects"),
            ("DefineGridManual", "DefineGrid"),
            ("MeasureImageAreaOccupiedBinary", "MeasureImageAreaOccupied"),
            ("Measureimageoverlap", "MeasureImageOverlap"),
            ("Medialaxis", "MedialAxis"),
            ("Medianfilter", "MedianFilter"),
            ("Morphologicalskeleton", "MorphologicalSkeleton"),
            ("Reducenoise", "ReduceNoise"),
        ),
    ),
)


_REGISTERED_CELLPROFILER_MODULE_SEMANTICS = _registered_semantics()


CELLPROFILER_MODULE_SEMANTICS: Mapping[str, CellProfilerModuleSemantics] = (
    MappingProxyType(
        {
            semantics.module_name: semantics
            for semantics in _REGISTERED_CELLPROFILER_MODULE_SEMANTICS
        }
    )
)

CELLPROFILER_MODULE_SEMANTICS_BY_KEY: Mapping[
    str,
    CellProfilerModuleSemantics,
] = MappingProxyType(
    {
        **{
            semantics.module_name.casefold(): semantics
            for semantics in _REGISTERED_CELLPROFILER_MODULE_SEMANTICS
        },
        **{
            alias.casefold(): CELLPROFILER_MODULE_SEMANTICS[canonical]
            for alias, canonical in _registered_aliases()
        },
    }
)
