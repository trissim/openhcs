"""Asset collection helpers for slide export."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

from ..io_handler import AnimatedGifOutput, MovieOutput
from .models import FigureAsset, FigureAssetCollection


def _build_composite_assets(
    composite_outputs: Iterable[Path],
) -> Tuple[FigureAsset, ...]:
    assets = []
    for path in sorted(composite_outputs):
        well_id = path.stem.replace("_composite", "")
        assets.append(
            FigureAsset(
                well_id=well_id,
                asset_type="composite_png",
                path=path,
                title=f"Well {well_id}",
                caption="Composite orthogonal projection",
            )
        )
    return tuple(assets)


def _build_sync_gif_assets(
    sync_gif_outputs: Iterable[AnimatedGifOutput],
) -> Tuple[FigureAsset, ...]:
    assets = []
    for output in sorted(sync_gif_outputs, key=lambda item: item.well_id):
        assets.append(
            FigureAsset(
                well_id=output.well_id,
                asset_type="sync_gif",
                path=output.output_path,
                title=f"Well {output.well_id}",
                caption="Synchronized orthogonal GIF",
            )
        )
    return tuple(assets)


def _build_movie_assets(
    movie_outputs: Iterable[MovieOutput],
) -> Tuple[FigureAsset, ...]:
    assets = []
    for output in sorted(
        movie_outputs, key=lambda item: (item.well_id, item.movie_type)
    ):
        if output.channel_id != "composite":
            continue
        assets.append(
            FigureAsset(
                well_id=output.well_id,
                asset_type="movie_mp4",
                path=output.output_path,
                title=f"Well {output.well_id}",
                caption=f"{output.movie_type.upper()} movie",
            )
        )
    return tuple(assets)


def collect_figure_assets(
    composite_outputs: Iterable[Path],
    sync_gif_outputs: Iterable[AnimatedGifOutput],
    movie_outputs: Iterable[MovieOutput],
) -> FigureAssetCollection:
    """Normalize generated figure outputs for slide export."""
    return FigureAssetCollection(
        composites=_build_composite_assets(composite_outputs),
        sync_gifs=_build_sync_gif_assets(sync_gif_outputs),
        movies=_build_movie_assets(movie_outputs),
    )


def select_assets_for_export(
    collection: FigureAssetCollection,
    include_composites: bool,
    include_sync_gifs: bool,
    include_movies: bool,
) -> Tuple[FigureAsset, ...]:
    """Select assets according to export options."""
    selected = []
    if include_composites:
        selected.extend(collection.composites)
    if include_sync_gifs:
        selected.extend(collection.sync_gifs)
    if include_movies:
        selected.extend(collection.movies)
    return tuple(selected)


def index_assets_by_well(
    assets: Iterable[FigureAsset],
) -> Dict[str, Tuple[FigureAsset, ...]]:
    """Group normalized assets by well id."""
    grouped: Dict[str, list[FigureAsset]] = {}
    for asset in assets:
        grouped.setdefault(asset.well_id, []).append(asset)
    return {well_id: tuple(items) for well_id, items in sorted(grouped.items())}
