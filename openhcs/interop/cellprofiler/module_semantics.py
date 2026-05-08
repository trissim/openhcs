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

    INPUT = "Input"
    FILE_PROCESSING = "File Processing"
    IMAGE_PROCESSING = "Image Processing"
    OBJECT_PROCESSING = "Object Processing"
    MEASUREMENT = "Measurement"
    ADVANCED = "Advanced"
    WORM_TOOLBOX = "Worm Toolbox"
    OTHER = "Other"
    DATA_TOOLS = "Data Tools"


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
class CellProfilerModuleSemantics:
    """Typed manual semantics for one CellProfiler module."""

    module_name: str
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
        return self.category in {
            CellProfilerModuleCategory.INPUT,
            CellProfilerModuleCategory.FILE_PROCESSING,
        }


def cellprofiler_module_semantics(
    module_name: str,
) -> CellProfilerModuleSemantics | None:
    """Return manual semantics for a CellProfiler module name, if known."""
    normalized_name = module_name.strip()
    if not normalized_name:
        raise ValueError("CellProfiler module name cannot be empty.")
    return CELLPROFILER_MODULE_SEMANTICS_BY_KEY.get(normalized_name.casefold())


class CellProfilerModuleSemanticsFamily(metaclass=AutoRegisterMeta):
    """Auto-registered declaration for a family of equivalent module semantics."""

    __registry_key__ = "family_name"
    __skip_if_no_key__ = True

    family_name: ClassVar[str | None] = None
    category: ClassVar[CellProfilerModuleCategory]
    dimensionality: ClassVar[CellProfilerModuleDimensionality]
    respects_masks: ClassVar[bool]
    module_names: ClassVar[tuple[str, ...]]

    @classmethod
    def declared_semantics(cls) -> tuple[CellProfilerModuleSemantics, ...]:
        """Materialize all module declarations in this family."""
        return tuple(
            CellProfilerModuleSemantics(
                module_name=module_name,
                category=cls.category,
                dimensionality=cls.dimensionality,
                respects_masks=cls.respects_masks,
            )
            for module_name in cls.module_names
        )


class CellProfilerModuleAliasFamily(metaclass=AutoRegisterMeta):
    """Auto-registered declaration for legacy module names."""

    __registry_key__ = "family_name"
    __skip_if_no_key__ = True

    family_name: ClassVar[str | None] = None
    aliases: ClassVar[tuple[tuple[str, str], ...]]


def _register_semantics_family(
    family_name: str,
    category: CellProfilerModuleCategory,
    dimensionality: CellProfilerModuleDimensionality,
    respects_masks: bool,
    module_names: Iterable[str],
) -> None:
    class_name = f"CellProfiler{family_name}Semantics"
    globals()[class_name] = type(
        class_name,
        (CellProfilerModuleSemanticsFamily,),
        {
            "__module__": __name__,
            "family_name": family_name,
            "category": category,
            "dimensionality": dimensionality,
            "respects_masks": respects_masks,
            "module_names": tuple(module_names),
        },
    )


def _register_alias_family(
    family_name: str,
    aliases: Iterable[tuple[str, str]],
) -> None:
    class_name = f"CellProfiler{family_name}Aliases"
    globals()[class_name] = type(
        class_name,
        (CellProfilerModuleAliasFamily,),
        {
            "__module__": __name__,
            "family_name": family_name,
            "aliases": tuple(aliases),
        },
    )


def _registered_semantics() -> tuple[CellProfilerModuleSemantics, ...]:
    return tuple(
        semantics
        for family_type in CellProfilerModuleSemanticsFamily.__registry__.values()
        for semantics in family_type.declared_semantics()
    )


def _registered_aliases() -> tuple[tuple[str, str], ...]:
    return tuple(
        alias
        for family_type in CellProfilerModuleAliasFamily.__registry__.values()
        for alias in family_type.aliases
    )


_C = CellProfilerModuleCategory
_D = CellProfilerModuleDimensionality
_register_semantics_family("InputUnmasked", _C.INPUT, _D.PLANAR_AND_VOLUMETRIC, False, (
    "Images",
    "LoadImages",
    "Metadata",
    "Groups",
))
_register_semantics_family("InputMasked", _C.INPUT, _D.PLANAR_AND_VOLUMETRIC, True, ("NamesAndTypes",))
_register_semantics_family(
    "FileProcessingUnmasked3D",
    _C.FILE_PROCESSING,
    _D.PLANAR_AND_VOLUMETRIC,
    False,
    ("CreateBatchFiles",),
)
_register_semantics_family(
    "FileProcessingMasked3D",
    _C.FILE_PROCESSING,
    _D.PLANAR_AND_VOLUMETRIC,
    True,
    ("ExportToDatabase", "ExportToSpreadsheet", "SaveCroppedObjects", "SaveImages"),
)
_register_semantics_family(
    "FileProcessingUnmasked2D",
    _C.FILE_PROCESSING,
    _D.PLANAR,
    False,
    ("LabelImages",),
)
_register_semantics_family(
    "FileProcessingMasked2D",
    _C.FILE_PROCESSING,
    _D.PLANAR,
    True,
    ("LoadData",),
)
_register_semantics_family(
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
)
_register_semantics_family(
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
)
_register_semantics_family(
    "ImageProcessingUnmasked2DApply",
    _C.IMAGE_PROCESSING,
    _D.PLANAR,
    False,
    ("CorrectIlluminationApply",),
)
_register_semantics_family(
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
)
_register_semantics_family(
    "ImageProcessingUnmasked3D",
    _C.IMAGE_PROCESSING,
    _D.PLANAR_AND_VOLUMETRIC,
    False,
    ("OverlayOutlines",),
)
_register_semantics_family(
    "ObjectProcessingUnmasked2D",
    _C.OBJECT_PROCESSING,
    _D.PLANAR,
    False,
    ("ClassifyObjects", "IdentifyObjectsManually"),
)
_register_semantics_family(
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
)
_register_semantics_family(
    "ObjectProcessingUnmasked3D",
    _C.OBJECT_PROCESSING,
    _D.PLANAR_AND_VOLUMETRIC,
    False,
    ("CombineObjects", "ConvertImageToObjects", "ResizeObjects"),
)
_register_semantics_family(
    "ObjectProcessingMasked3D",
    _C.OBJECT_PROCESSING,
    _D.PLANAR_AND_VOLUMETRIC,
    True,
    ("ConvertObjectsToImage", "FilterObjects", "RelateObjects"),
)
_register_semantics_family(
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
)
_register_semantics_family(
    "MeasurementMasked2D",
    _C.MEASUREMENT,
    _D.PLANAR,
    True,
    ("MeasureObjectIntensityDistribution",),
)
_register_semantics_family(
    "MeasurementUnmasked3D",
    _C.MEASUREMENT,
    _D.PLANAR_AND_VOLUMETRIC,
    False,
    ("MeasureObjectNeighbors", "MeasureObjectSizeShape"),
)
_register_semantics_family(
    "MeasurementUnmasked2D",
    _C.MEASUREMENT,
    _D.PLANAR,
    False,
    ("MeasureObjectOverlap", "MeasureObjectSkeleton"),
)
_register_semantics_family(
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
)
_register_semantics_family(
    "AdvancedUnmasked2D",
    _C.ADVANCED,
    _D.PLANAR,
    False,
    ("MatchTemplate", "RunImageJMacro"),
)
_register_semantics_family("AdvancedMasked3D", _C.ADVANCED, _D.PLANAR_AND_VOLUMETRIC, True, ("Watershed",))
_register_semantics_family(
    "WormToolboxMasked2D",
    _C.WORM_TOOLBOX,
    _D.PLANAR,
    True,
    ("IdentifyDeadWorms", "StraightenWorms", "UntangleWorms"),
)
_register_semantics_family("OtherUnmasked2D", _C.OTHER, _D.PLANAR, False, ("DefineGrid",))
_register_semantics_family(
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
)
_register_alias_family(
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
)


_REGISTERED_CELLPROFILER_MODULE_SEMANTICS = _registered_semantics()


CELLPROFILER_MODULE_SEMANTICS: Mapping[str, CellProfilerModuleSemantics] = MappingProxyType(
    {
        semantics.module_name: semantics
        for semantics in _REGISTERED_CELLPROFILER_MODULE_SEMANTICS
    }
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
