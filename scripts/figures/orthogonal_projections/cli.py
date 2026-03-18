"""
Command-line interface for orthogonal projection generation.

This module provides the main entry point for:
- Processing plates one well at a time (or in parallel)
- Generating projections, composites, mosaics, and Z-stack slice movies
- Tracking progress and handling errors

Features:
- Well-by-well memory-efficient processing
- Optional multiprocessing via -j/--jobs flag
- Composite figures with XY/XZ/YZ projections
- Z-stack slice movies (XY through Z, XZ through Y, YZ through X)
- Z-gap stretching for XZ/YZ (consistent between figures and movies)
- Plate mosaics and arbitrary group mosaics

Usage:
    # Single process - composites only
    python -m scripts.figures.orthogonal_projections.cli \\
        --plate-dir /path/to/plate_stitched/ \\
        --output-dir /output/path/ \\
        --z-gap 4.25

    # Parallel processing with movies
    python -m scripts.figures.orthogonal_projections.cli \\
        --plate-dir /path/to/plate_stitched/ \\
        --output-dir /output/path/ \\
        --z-gap 4.25 \\
        --create-movies \\
        --movie-types xy xz yz \\
        --movie-fps 10 \\
        -j 4

Invariants:
- Sequential or parallel well processing
- Progress tracking via logging
- Failures are logged, not silent
- All outputs tracked
"""

import argparse
import gc
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from .constants import (
    ChannelColorMapping,
    CompositeLayout,
    DEFAULT_CHANNEL_COLORS,
    LabelConfig,
    MosaicLayout,
)
from .discovery import (
    WellChannelKey,
    discover_well_channels,
    group_z_slices_by_well_channel,
    get_microscope_handler,
)
from .io_handler import (
    MovieOutput,
    ProjectionOutput,
    load_z_stack,
    save_all_projections,
    save_slice_movies_for_well,
)
from .composer import (
    create_multi_channel_composite,
    save_composite_figure,
)
from .mosaic import (
    ArbitraryMosaicSpec,
    create_plate_mosaic,
    create_arbitrary_mosaic,
    save_mosaic,
)
from .labeling import FigureLabeler, get_labeler
from openhcs.processing.backends.processors.numpy_processor import (
    create_orthogonal_projections,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for processing run."""

    plate_path: Path
    output_dir: Path
    projections: Tuple[str, ...] = ("xy", "xz", "yz")
    output_format: str = "tiff"
    channel_colors: Tuple[ChannelColorMapping, ...] = DEFAULT_CHANNEL_COLORS
    excluded_channels: FrozenSet[str] = frozenset()
    included_channels: FrozenSet[str] = frozenset()
    included_wells: FrozenSet[str] = frozenset()
    include_mode: bool = False
    save_individual_projections: bool = True
    create_composites: bool = True
    create_movies: bool = False
    movie_types: Tuple[str, ...] = ("xy", "xz", "yz")
    movie_fps: int = 10
    movie_bit_depth: int = 8
    create_plate_mosaic: bool = False
    mosaic_layout: MosaicLayout = None
    arbitrary_mosaics: Tuple[ArbitraryMosaicSpec, ...] = ()
    labeler_config: LabelConfig = None
    dpi: int = 150
    z_gap: float = 1.0
    z_aspect: float = 0.1
    num_workers: int = 1


@dataclass
class WellResult:
    """Result from processing a single well."""

    well_id: str
    success: bool
    composite_path: Optional[Path] = None
    movie_outputs: List[MovieOutput] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class ProcessingResult:
    """Mutable accumulator for processing results."""

    total_wells: int = 0
    processed_wells: int = 0
    failed_wells: List[str] = field(default_factory=list)
    outputs: List[ProjectionOutput] = field(default_factory=list)
    composite_outputs: List[Path] = field(default_factory=list)
    movie_outputs: List[MovieOutput] = field(default_factory=list)
    mosaic_outputs: List[Path] = field(default_factory=list)


def _get_default_dtype_config():
    """Get default dtype config for standalone use."""
    from openhcs.core.config import DtypeConfig
    from openhcs.constants.constants import DtypeConversion

    return DtypeConfig(default_dtype_conversion=DtypeConversion.PRESERVE_INPUT)


def process_single_channel(
    well_key: WellChannelKey, z_paths: Tuple[Path, ...], config: ProcessingConfig
) -> Dict[str, np.ndarray]:
    """
    Process one well/channel combination.

    Returns projections dict for composite generation.
    """
    logger.info(f"  Processing channel {well_key.channel_id} ({well_key.channel_name})")

    z_stack = load_z_stack(z_paths)

    dtype_config = _get_default_dtype_config()
    projections = create_orthogonal_projections(
        z_stack, config.projections, dtype_config=dtype_config
    )

    if config.save_individual_projections:
        proj_dir = config.output_dir / "projections"
        outputs = save_all_projections(
            projections, well_key, proj_dir, config.output_format
        )
        config_ref = config

    del z_stack
    gc.collect()

    return projections


def process_well_all_channels(
    well_id: str,
    channel_keys: Tuple[WellChannelKey, ...],
    z_paths_by_channel: Dict[WellChannelKey, Tuple[Path, ...]],
    config: ProcessingConfig,
    labeler: FigureLabeler,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[Path], List[MovieOutput]]:
    """
    Process all channels for a single well.

    Returns:
        - Dict of projections by channel (empty after cleanup)
        - Path to composite figure (if created)
        - List of movie outputs (if created)
    """
    all_projections = {}
    all_z_stacks = {}
    movie_outputs = []

    for channel_key in channel_keys:
        if channel_key.well_id != well_id:
            continue

        if config.include_mode:
            if channel_key.channel_id not in config.included_channels:
                continue
        else:
            if channel_key.channel_id in config.excluded_channels:
                continue

        z_paths = z_paths_by_channel.get(channel_key)
        if not z_paths:
            logger.warning(
                f"  No Z-slices found for {well_id} channel {channel_key.channel_id}"
            )
            continue

        z_stack = load_z_stack(z_paths)
        all_z_stacks[channel_key.channel_id] = z_stack

        projections = process_single_channel(channel_key, z_paths, config)
        all_projections[channel_key.channel_id] = projections

    composite_path = None
    if config.create_composites and all_projections:
        channel_names = tuple(
            ck.channel_name
            for ck in channel_keys
            if ck.well_id == well_id and ck.channel_id in all_projections
        )

        title = labeler.format_title(well_id, channel_names)

        layout = CompositeLayout(z_gap=config.z_gap, z_aspect=config.z_aspect)
        fig = create_multi_channel_composite(
            all_projections,
            title,
            layout=layout,
            channel_colors=config.channel_colors,
            labeler=labeler,
        )

        composite_dir = config.output_dir / "composites"
        composite_dir.mkdir(parents=True, exist_ok=True)
        composite_path = composite_dir / f"{well_id}_composite.png"
        save_composite_figure(fig, composite_path, dpi=config.dpi)

        logger.info(f"  Saved composite: {composite_path}")

    if config.create_movies and all_z_stacks:
        movie_dir = config.output_dir / "movies"
        movie_outputs = save_slice_movies_for_well(
            all_z_stacks,
            well_id,
            movie_dir,
            slice_types=config.movie_types,
            fps=config.movie_fps,
            z_gap=config.z_gap,
            bit_depth=config.movie_bit_depth,
            channel_colors=config.channel_colors,
        )
        for mo in movie_outputs:
            logger.info(f"  Saved movie: {mo.output_path}")

    for proj_dict in all_projections.values():
        for arr in proj_dict.values():
            del arr
    all_projections.clear()

    for stack in all_z_stacks.values():
        del stack
    all_z_stacks.clear()
    gc.collect()

    return all_projections, composite_path, movie_outputs


def _process_well_worker(
    well_id: str,
    plate_path_str: str,
    z_paths_dict: Dict,
    config_dict: Dict,
) -> WellResult:
    """
    Worker function for parallel well processing.

    Args:
        well_id: Well ID to process
        plate_path_str: String path to plate directory
        z_paths_dict: Serialized z_paths_by_channel dict
        config_dict: Serialized ProcessingConfig dict

    Returns:
        WellResult with outputs
    """
    from pathlib import Path as PP

    from openhcs.core.config import DtypeConfig
    from openhcs.constants.constants import DtypeConversion
    from openhcs.processing.backends.processors.numpy_processor import (
        create_orthogonal_projections,
    )

    from .composer import create_multi_channel_composite, save_composite_figure
    from .constants import ChannelColorMapping, CompositeLayout, DEFAULT_CHANNEL_COLORS
    from .discovery import WellChannelKey
    from .io_handler import MovieOutput, load_z_stack, save_slice_movies_for_well
    from .labeling import get_labeler

    try:
        plate_path = PP(plate_path_str)
        dtype_config = DtypeConfig(
            default_dtype_conversion=DtypeConversion.PRESERVE_INPUT
        )

        channel_colors = tuple(
            ChannelColorMapping(
                cc["channel_id"], cc["channel_name"], cc["color"], cc["visible"]
            )
            for cc in config_dict.get("channel_colors", [])
        )

        output_dir = PP(config_dict["output_dir"])
        create_composites = config_dict.get("create_composites", True)
        create_movies = config_dict.get("create_movies", False)
        movie_types = tuple(config_dict.get("movie_types", ["xy", "xz", "yz"]))
        movie_fps = config_dict.get("movie_fps", 10)
        movie_bit_depth = config_dict.get("movie_bit_depth", 8)
        z_gap = config_dict.get("z_gap", 1.0)
        z_aspect = config_dict.get("z_aspect", 0.1)
        dpi = config_dict.get("dpi", 150)

        z_paths_by_channel = {
            WellChannelKey(
                well_id=wk[0],
                channel_id=wk[1],
                channel_name=wk[2],
            ): tuple(PP(p) for p in paths)
            for wk, paths in z_paths_dict.items()
        }

        well_channel_keys = [
            wk for wk in z_paths_by_channel.keys() if wk.well_id == well_id
        ]

        included_channels = frozenset(config_dict.get("included_channels", []))
        excluded_channels = frozenset(config_dict.get("excluded_channels", []))
        include_mode = config_dict.get("include_mode", False)

        all_z_stacks = {}
        all_projections = {}

        for wk in well_channel_keys:
            if include_mode:
                if wk.channel_id not in included_channels:
                    continue
            else:
                if wk.channel_id in excluded_channels:
                    continue

            z_paths = z_paths_by_channel.get(wk)
            if not z_paths:
                continue

            z_stack = load_z_stack(z_paths)
            all_z_stacks[wk.channel_id] = z_stack

            projections = create_orthogonal_projections(
                z_stack,
                dtype_config=dtype_config,
            )
            all_projections[wk.channel_id] = projections

        composite_path = None
        if create_composites and all_projections:
            layout = CompositeLayout(z_gap=z_gap, z_aspect=z_aspect)
            channel_names = tuple(
                wk.channel_name
                for wk in well_channel_keys
                if wk.channel_id in all_projections
            )
            title = f"Well {well_id}"

            fig = create_multi_channel_composite(
                all_projections,
                title,
                layout=layout,
                channel_colors=channel_colors or DEFAULT_CHANNEL_COLORS,
                labeler=get_labeler("standard"),
            )

            composite_dir = output_dir / "composites"
            composite_dir.mkdir(parents=True, exist_ok=True)
            composite_path = composite_dir / f"{well_id}_composite.png"
            save_composite_figure(fig, composite_path, dpi=dpi)

        movie_outputs = []
        if create_movies and all_z_stacks:
            movie_dir = output_dir / "movies"
            movie_outputs = save_slice_movies_for_well(
                all_z_stacks,
                well_id,
                movie_dir,
                slice_types=movie_types,
                fps=movie_fps,
                z_gap=z_gap,
                bit_depth=movie_bit_depth,
                channel_colors=channel_colors or DEFAULT_CHANNEL_COLORS,
            )

        return WellResult(
            well_id=well_id,
            success=True,
            composite_path=composite_path,
            movie_outputs=movie_outputs,
        )

    except Exception as e:
        import traceback

        return WellResult(
            well_id=well_id,
            success=False,
            error=f"{type(e).__name__}: {e}",
        )


def process_plate(config: ProcessingConfig) -> ProcessingResult:
    """
    Process entire plate, with optional parallel processing.
    """
    result = ProcessingResult()

    logger.info(f"Discovering wells in: {config.plate_path}")
    microscope_handler = get_microscope_handler(config.plate_path)

    channel_filter = (
        config.included_channels if config.include_mode else config.excluded_channels
    )

    well_keys = discover_well_channels(
        config.plate_path,
        microscope_handler,
        channel_filter=channel_filter if channel_filter else None,
        include_mode=config.include_mode,
    )

    if not well_keys:
        logger.error("No wells found!")
        return result

    logger.info(f"Found {len(well_keys)} well/channel combinations")

    z_paths_by_channel = group_z_slices_by_well_channel(
        config.plate_path, well_keys, microscope_handler
    )

    wells = sorted(set(wk.well_id for wk in well_keys))

    if config.included_wells:
        wells = [w for w in wells if w in config.included_wells]

    result.total_wells = len(wells)

    labeler = get_labeler("standard")
    if config.labeler_config:
        labeler = FigureLabeler(config.labeler_config)

    composite_paths = {}

    if config.num_workers > 1:
        logger.info(f"Processing {len(wells)} wells with {config.num_workers} workers")

        config_dict = {
            "output_dir": str(config.output_dir),
            "create_composites": config.create_composites,
            "create_movies": config.create_movies,
            "movie_types": config.movie_types,
            "movie_fps": config.movie_fps,
            "movie_bit_depth": config.movie_bit_depth,
            "z_gap": config.z_gap,
            "z_aspect": config.z_aspect,
            "dpi": config.dpi,
            "included_channels": list(config.included_channels),
            "excluded_channels": list(config.excluded_channels),
            "include_mode": config.include_mode,
            "channel_colors": [
                {
                    "channel_id": cc.channel_id,
                    "channel_name": cc.channel_name,
                    "color": cc.color,
                    "visible": cc.visible,
                }
                for cc in config.channel_colors
            ],
        }

        z_paths_dict = {
            (wk.well_id, wk.channel_id, wk.channel_name): [str(p) for p in paths]
            for wk, paths in z_paths_by_channel.items()
        }

        with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(
                    _process_well_worker,
                    well_id,
                    str(config.plate_path),
                    z_paths_dict,
                    config_dict,
                ): well_id
                for well_id in wells
            }

            for future in as_completed(futures):
                well_id = futures[future]
                try:
                    well_result = future.result()
                    if well_result.success:
                        if well_result.composite_path:
                            composite_paths[well_id] = well_result.composite_path
                            result.composite_outputs.append(well_result.composite_path)
                        result.movie_outputs.extend(well_result.movie_outputs)
                        result.processed_wells += 1
                        logger.info(f"  Completed {well_id}")
                    else:
                        result.failed_wells.append(well_id)
                        logger.error(f"  Failed {well_id}: {well_result.error}")
                except Exception as e:
                    result.failed_wells.append(well_id)
                    logger.error(f"  Failed {well_id}: {e}")
    else:
        for i, well_id in enumerate(wells, 1):
            logger.info(f"Processing well {well_id} ({i}/{len(wells)})")

            try:
                _, composite_path, movie_outputs = process_well_all_channels(
                    well_id, well_keys, z_paths_by_channel, config, labeler
                )

                if composite_path:
                    composite_paths[well_id] = composite_path
                    result.composite_outputs.append(composite_path)

                result.movie_outputs.extend(movie_outputs)

                result.processed_wells += 1

            except Exception as e:
                import traceback

                logger.error(f"Failed to process well {well_id}: {e}")
                logger.debug(traceback.format_exc())
                result.failed_wells.append(well_id)

            gc.collect()

    if config.create_plate_mosaic and composite_paths:
        logger.info("Creating plate mosaic...")
        try:
            mosaic_fig = create_plate_mosaic(
                composite_paths,
                layout=config.mosaic_layout or MosaicLayout(),
                labeler=labeler,
            )

            mosaic_dir = config.output_dir / "mosaics"
            mosaic_dir.mkdir(parents=True, exist_ok=True)
            mosaic_path = mosaic_dir / "plate_mosaic_XY.png"
            save_mosaic(mosaic_fig, mosaic_path, dpi=config.dpi)
            result.mosaic_outputs.append(mosaic_path)
            logger.info(f"Saved plate mosaic: {mosaic_path}")
        except Exception as e:
            logger.error(f"Failed to create plate mosaic: {e}")

    for spec in config.arbitrary_mosaics:
        logger.info(f"Creating arbitrary mosaic: {spec.name}")
        try:
            mosaic_fig = create_arbitrary_mosaic(composite_paths, spec, labeler=labeler)

            mosaic_dir = config.output_dir / "mosaics"
            mosaic_path = mosaic_dir / f"{spec.name}_mosaic.png"
            save_mosaic(mosaic_fig, mosaic_path, dpi=config.dpi)
            result.mosaic_outputs.append(mosaic_path)
            logger.info(f"Saved mosaic: {mosaic_path}")
        except Exception as e:
            logger.error(f"Failed to create mosaic '{spec.name}': {e}")

    return result


def parse_color_mapping(color_str: str) -> ChannelColorMapping:
    """Parse 'channel_id:color' string into ChannelColorMapping."""
    parts = color_str.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid color mapping: {color_str}. Expected format: 'channel_id:color'"
        )

    channel_id, color = parts
    return ChannelColorMapping(
        channel_id.strip(), f"Channel {channel_id.strip()}", color.strip()
    )


def parse_mosaic_group(group_str: str) -> ArbitraryMosaicSpec:
    """Parse 'name:well1,well2,well3' string into ArbitraryMosaicSpec."""
    parts = group_str.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid mosaic group: {group_str}. Expected format: 'name:well1,well2,well3'"
        )

    name = parts[0].strip()
    wells = tuple(w.strip() for w in parts[1].split(","))
    return ArbitraryMosaicSpec(name=name, well_ids=wells)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate orthogonal projections for invasion assay plates"
    )

    parser.add_argument(
        "--plate-dir",
        "-i",
        required=True,
        help="Path to stitched plate directory (with openhcs_metadata.json)",
    )

    parser.add_argument(
        "--output-dir", "-o", required=True, help="Path to output directory"
    )

    parser.add_argument(
        "--projections",
        nargs="+",
        default=["xy", "xz", "yz"],
        choices=["xy", "xz", "yz"],
        help="Projection types to generate",
    )

    parser.add_argument(
        "--channels", nargs="+", help="Include ONLY these channels (channel IDs)"
    )

    parser.add_argument(
        "--wells", nargs="+", help="Include ONLY these wells (well IDs, e.g. R01C06)"
    )

    parser.add_argument(
        "--exclude-channels", nargs="+", help="Exclude these channels (channel IDs)"
    )

    parser.add_argument(
        "--channel-colors",
        nargs="+",
        help="Channel color mappings (format: '1:cyan' '2:green')",
    )

    parser.add_argument(
        "--create-plate-mosaic", action="store_true", help="Create full-plate mosaic"
    )

    parser.add_argument(
        "--mosaic-group",
        nargs="+",
        dest="mosaic_groups",
        help="Create arbitrary mosaic (format: 'name:well1,well2,well3')",
    )

    parser.add_argument(
        "--label-style",
        choices=["standard", "publication", "minimal"],
        default="standard",
        help="Labeling style for figures",
    )

    parser.add_argument(
        "--no-individual-projections",
        action="store_true",
        help="Skip saving individual projection files",
    )

    parser.add_argument(
        "--no-composites", action="store_true", help="Skip creating composite figures"
    )

    parser.add_argument(
        "--create-movies",
        action="store_true",
        help="Create Z-stack slice movies (XY, XZ, YZ animations)",
    )

    parser.add_argument(
        "--movie-types",
        nargs="+",
        default=["xy", "xz", "yz"],
        choices=["xy", "xz", "yz"],
        help="Which slice movie types to create",
    )

    parser.add_argument(
        "--movie-fps",
        type=int,
        default=10,
        help="Frames per second for movies",
    )

    parser.add_argument(
        "--movie-bit-depth",
        type=int,
        default=8,
        choices=[8, 16],
        help="Bit depth for movie output (8 or 16)",
    )

    parser.add_argument(
        "--format",
        default="tiff",
        choices=["tiff", "png"],
        help="Output format for individual projections",
    )

    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures")

    parser.add_argument(
        "--z-gap",
        type=float,
        default=1.0,
        help="Z-slice gap multiplier for XZ/YZ projections",
    )

    parser.add_argument(
        "--z-aspect", type=float, default=0.1, help="Aspect ratio for XZ/YZ projections"
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )

    args = parser.parse_args()

    channel_colors = DEFAULT_CHANNEL_COLORS
    if args.channel_colors:
        channel_colors = tuple(parse_color_mapping(c) for c in args.channel_colors)

    included_channels = frozenset(args.channels) if args.channels else frozenset()
    excluded_channels = (
        frozenset(args.exclude_channels) if args.exclude_channels else frozenset()
    )
    include_mode = bool(args.channels)

    arbitrary_mosaics = ()
    if args.mosaic_groups:
        arbitrary_mosaics = tuple(parse_mosaic_group(g) for g in args.mosaic_groups)

    config = ProcessingConfig(
        plate_path=Path(args.plate_dir),
        output_dir=Path(args.output_dir),
        projections=tuple(args.projections),
        output_format=args.format,
        channel_colors=channel_colors,
        excluded_channels=excluded_channels,
        included_channels=included_channels,
        included_wells=frozenset(args.wells) if args.wells else frozenset(),
        include_mode=include_mode,
        save_individual_projections=not args.no_individual_projections,
        create_composites=not args.no_composites,
        create_movies=args.create_movies,
        movie_types=tuple(args.movie_types),
        movie_fps=args.movie_fps,
        movie_bit_depth=args.movie_bit_depth,
        create_plate_mosaic=args.create_plate_mosaic,
        arbitrary_mosaics=arbitrary_mosaics,
        dpi=args.dpi,
        z_gap=args.z_gap,
        z_aspect=args.z_aspect,
        num_workers=args.jobs,
    )

    logger.info(f"Processing plate: {config.plate_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Projections: {config.projections}")
    logger.info(f"Workers: {config.num_workers}")

    result = process_plate(config)

    logger.info("=" * 50)
    logger.info("Processing complete!")
    logger.info(f"  Total wells: {result.total_wells}")
    logger.info(f"  Processed: {result.processed_wells}")
    logger.info(f"  Failed: {len(result.failed_wells)}")
    if result.failed_wells:
        logger.info(f"  Failed wells: {', '.join(result.failed_wells)}")
    logger.info(f"  Composites: {len(result.composite_outputs)}")
    logger.info(f"  Movies: {len(result.movie_outputs)}")
    logger.info(f"  Mosaics: {len(result.mosaic_outputs)}")


if __name__ == "__main__":
    main()
