"""Regression tests for JSON-safe CellProfiler compatibility registries."""

from benchmark.cellprofiler_library.functions.align import AlignCropModeStrategy
from benchmark.cellprofiler_library.functions.calculatemath import (
    MathOperationStrategy,
    RoundingStrategy,
)
from benchmark.cellprofiler_library.functions.convertobjectstoimage import (
    ImageModeRenderer,
)
from benchmark.cellprofiler_library.functions.correctilluminationapply import (
    IlluminationCorrectionStrategy,
)
from benchmark.cellprofiler_library.functions.correctilluminationcalculate import (
    SmoothingFilterSizeStrategy,
    SmoothingPlaneStrategy,
)
from benchmark.cellprofiler_library.functions.crop import CropShapeMaskStrategy
from benchmark.cellprofiler_library.functions.enhanceedges import EdgeEnhancementStrategy
from benchmark.cellprofiler_library.functions.filterobjects import (
    FilterSelectionStrategy,
    PerObjectAssignmentStrategy,
)
from benchmark.cellprofiler_library.functions.graytocolor import GrayToColorSchemeRunner
from benchmark.cellprofiler_library.functions.identifyprimaryobjects import (
    WatershedImageBuilder,
)
from benchmark.cellprofiler_library.functions.identifysecondaryobjects import (
    SecondarySegmentationStrategy,
    ThresholdCalculator,
)
from benchmark.cellprofiler_library.functions.measureobjectneighbors import (
    NeighborDistancePlanner,
)
from benchmark.cellprofiler_library.functions.smooth import SmoothingStrategy
from benchmark.cellprofiler_library.functions.structuring_elements import (
    StructuringElementFactory,
)
from benchmark.cellprofiler_library.functions.untangleworms import (
    WormLabelOutputStrategy,
)


JSON_SAFE_REGISTRY_KEY_TYPES = (str, int, float, bool, type(None))


def test_cellprofiler_strategy_registry_keys_are_json_safe():
    registry_classes = (
        AlignCropModeStrategy,
        MathOperationStrategy,
        RoundingStrategy,
        ImageModeRenderer,
        IlluminationCorrectionStrategy,
        SmoothingFilterSizeStrategy,
        SmoothingPlaneStrategy,
        CropShapeMaskStrategy,
        EdgeEnhancementStrategy,
        FilterSelectionStrategy,
        PerObjectAssignmentStrategy,
        GrayToColorSchemeRunner,
        WatershedImageBuilder,
        SecondarySegmentationStrategy,
        ThresholdCalculator,
        NeighborDistancePlanner,
        SmoothingStrategy,
        StructuringElementFactory,
        WormLabelOutputStrategy,
    )

    for registry_class in registry_classes:
        assert registry_class.__registry__
        assert all(
            isinstance(key, JSON_SAFE_REGISTRY_KEY_TYPES)
            for key in registry_class.__registry__
        ), registry_class.__name__
