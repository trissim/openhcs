"""Decorator-backed CellProfiler cppipe benchmark declarations."""

from __future__ import annotations

from collections.abc import Callable

from benchmark.contracts.dataset import BenchmarkCategory

DEFAULT_BENCHMARK_CATEGORY = BenchmarkCategory(
    assay="Uncategorized assay",
    module="Mixed modules",
)
AGGREGATE_BENCHMARK_CATEGORY = BenchmarkCategory(
    assay="Aggregate",
    module="Aggregate",
)

_OFFICIAL_CP3_CASE_DECLARATIONS: dict[str, BenchmarkCategory] = {}


def official_cp3_case(
    *,
    assay: str,
    module: str,
) -> Callable[[Callable[[], None]], Callable[[], None]]:
    """Attach benchmark category metadata to an official CP3 case declaration."""

    def decorate(declaration: Callable[[], None]) -> Callable[[], None]:
        _OFFICIAL_CP3_CASE_DECLARATIONS[declaration.__name__] = BenchmarkCategory(
            assay=assay,
            module=module,
        )
        return declaration

    return decorate


def official_cp3_case_category(case_name: str) -> BenchmarkCategory:
    """Return declared category metadata for an official CellProfiler example."""
    return _OFFICIAL_CP3_CASE_DECLARATIONS.get(
        case_name,
        DEFAULT_BENCHMARK_CATEGORY,
    )


@official_cp3_case(
    assay="Colocalization microscopy",
    module="Colocalization + object measurement",
)
def ExampleColocalization() -> None:
    """Official CP3 colocalization example."""


@official_cp3_case(assay="DNA damage assay", module="Segmentation + shape measurement")
def ExampleCometAssay() -> None:
    """Official CP3 comet assay example."""


@official_cp3_case(
    assay="Tissue/object morphology",
    module="Segmentation + object measurement",
)
def ExampleFly() -> None:
    """Official CP3 fly morphology example."""


@official_cp3_case(
    assay="Tissue/object morphology",
    module="Segmentation + object measurement",
)
def ExampleFlyURL() -> None:
    """Official CP3 fly morphology URL example."""


@official_cp3_case(
    assay="Cell morphology",
    module="Segmentation + intensity measurement",
)
def ExampleHuman() -> None:
    """Official CP3 human cell example."""


@official_cp3_case(assay="Illumination correction", module="Image correction")
def ExampleIlluminationCorrection_Example1_AllMethod() -> None:
    """Official CP3 illumination correction example."""


@official_cp3_case(assay="Illumination correction", module="Image correction")
def ExampleIlluminationCorrection_Example1_EachMethod() -> None:
    """Official CP3 illumination correction example."""


@official_cp3_case(assay="Illumination correction", module="Image correction")
def ExampleIlluminationCorrection_Example2() -> None:
    """Official CP3 illumination correction example."""


@official_cp3_case(assay="Illumination correction", module="Image correction")
def ExampleIlluminationCorrection_Example3() -> None:
    """Official CP3 illumination correction example."""


@official_cp3_case(
    assay="Imaging flow cytometry",
    module="Grid layout + object measurement",
)
def ExampleImagingFlowCytometryObjectsInGrid() -> None:
    """Official CP3 imaging-flow cytometry example."""


@official_cp3_case(
    assay="Spatial organization",
    module="Object relationship measurement",
)
def ExampleNeighbors() -> None:
    """Official CP3 object-neighbor example."""


@official_cp3_case(
    assay="Positive-cell classification",
    module="Thresholding + classification",
)
def ExamplePercentPositive() -> None:
    """Official CP3 percent-positive example."""


@official_cp3_case(assay="Spot detection", module="Small-object detection")
def ExampleSpeckles() -> None:
    """Official CP3 speckle detection example."""


@official_cp3_case(
    assay="Time-lapse tracking",
    module="Tracking + object measurement",
)
def ExampleTrackObjects() -> None:
    """Official CP3 tracking example."""


@official_cp3_case(
    assay="Tumor morphology",
    module="Segmentation + intensity measurement",
)
def ExampleTumor() -> None:
    """Official CP3 tumor morphology example."""


@official_cp3_case(assay="Worm phenotyping", module="Worm model + straightening")
def ExampleUntangleAndStraightenWorms() -> None:
    """Official CP3 untangle-and-straighten worms example."""


@official_cp3_case(assay="Worm phenotyping", module="Worm model + shape measurement")
def ExampleUntangleWorms() -> None:
    """Official CP3 untangle worms example."""


@official_cp3_case(
    assay="Worm phenotyping",
    module="Worm model + brightfield segmentation",
)
def ExampleUntangleWormsBrightField() -> None:
    """Official CP3 brightfield worm example."""


@official_cp3_case(assay="Cell morphology", module="Segmentation + shape measurement")
def ExampleVitra() -> None:
    """Official CP3 Vitra morphology example."""


@official_cp3_case(
    assay="Migration/scratch assay",
    module="Segmentation + distance measurement",
)
def ExampleWoundHealing() -> None:
    """Official CP3 wound-healing example."""


@official_cp3_case(
    assay="Yeast colony screening",
    module="Colony segmentation + measurement",
)
def ExampleYeastColonies() -> None:
    """Official CP3 yeast-colonies example."""


@official_cp3_case(
    assay="Yeast colony screening",
    module="Texture/patch measurement",
)
def ExampleYeastPatches() -> None:
    """Official CP3 yeast-patches example."""
