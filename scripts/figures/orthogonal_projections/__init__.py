"""
Headless Orthogonal Projection Generator for OpenHCS.

This package provides memory-efficient generation of XY/XZ/YZ projections
for invasion assay plates, with multi-channel color overlay and mosaic generation.

Usage:
    python -m scripts.figures.orthogonal_projections.cli \\
        --plate-dir /path/to/plate \\
        --output-dir /path/to/output

Modules:
    - constants: Default configurations for colors, layouts, labels
    - labeling: Modular figure labeling system
    - composer: Multi-channel composite figure generation
    - mosaic: Plate mosaic generation
    - discovery: Well/channel discovery via OpenHCS API
    - io_handler: File loading and saving
    - cli: Command-line interface
"""

from .constants import (
    ChannelColorMapping,
    CompositeLayout,
    MosaicLayout,
    LabelConfig,
    DEFAULT_CHANNEL_COLORS,
    STANDARD_LABEL_CONFIG,
    PUBLICATION_LABEL_CONFIG,
    MINIMAL_LABEL_CONFIG,
)

from .discovery import (
    WellChannelKey,
    discover_well_channels,
    group_z_slices_by_well_channel,
)

from .io_handler import (
    ProjectionOutput,
    load_z_stack,
    save_projection,
    save_all_projections,
)

from .composer import (
    create_projection_composite,
    create_multi_channel_composite,
    save_composite_figure,
)

from .mosaic import (
    ArbitraryMosaicSpec,
    create_plate_mosaic,
    create_arbitrary_mosaic,
    save_mosaic,
)

from .labeling import (
    FigureLabeler,
    get_labeler,
)

__all__ = [
    "ChannelColorMapping",
    "CompositeLayout",
    "MosaicLayout",
    "LabelConfig",
    "DEFAULT_CHANNEL_COLORS",
    "STANDARD_LABEL_CONFIG",
    "PUBLICATION_LABEL_CONFIG",
    "MINIMAL_LABEL_CONFIG",
    "WellChannelKey",
    "discover_well_channels",
    "group_z_slices_by_well_channel",
    "ProjectionOutput",
    "load_z_stack",
    "save_projection",
    "save_all_projections",
    "create_projection_composite",
    "create_multi_channel_composite",
    "save_composite_figure",
    "ArbitraryMosaicSpec",
    "create_plate_mosaic",
    "create_arbitrary_mosaic",
    "save_mosaic",
    "FigureLabeler",
    "get_labeler",
]
