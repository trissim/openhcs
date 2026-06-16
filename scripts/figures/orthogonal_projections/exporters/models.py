"""Dataclasses for slide deck export."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

AssetType = Literal["composite_png", "sync_gif", "movie_mp4"]
DeckBackend = Literal["pptx"]
GroupMode = Literal["well"]
LayoutMode = Literal["two_panel"]
MediaFitMode = Literal["contain"]


@dataclass(frozen=True)
class FigureAsset:
    """Normalized figure asset for deck export."""

    well_id: str
    asset_type: AssetType
    path: Path
    title: str
    caption: Optional[str] = None
    channel_summary: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FigureAssetCollection:
    """Structured figure assets grouped by output kind."""

    composites: Tuple[FigureAsset, ...] = ()
    sync_gifs: Tuple[FigureAsset, ...] = ()
    movies: Tuple[FigureAsset, ...] = ()

    def all_assets(self) -> Tuple[FigureAsset, ...]:
        return self.composites + self.sync_gifs + self.movies


@dataclass(frozen=True)
class SlideContentSpec:
    """Backend-agnostic content definition for one slide."""

    slide_id: str
    title: str
    subtitle: Optional[str]
    assets: Tuple[FigureAsset, ...]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class SlideLayoutOptions:
    """Presentation layout settings."""

    mode: LayoutMode = "two_panel"
    media_fit_mode: MediaFitMode = "contain"
    margin_inches: float = 0.4
    title_height_inches: float = 0.6
    subtitle_height_inches: float = 0.3
    footer_height_inches: float = 0.25
    media_gap_inches: float = 0.2


@dataclass(frozen=True)
class DeckExportOptions:
    """Top-level export settings for a slide deck."""

    output_path: Path
    backend: DeckBackend = "pptx"
    group_by: GroupMode = "well"
    include_composites: bool = True
    include_sync_gifs: bool = True
    include_movies: bool = False
    prefer_video_over_gif: bool = True
    slide_layout: SlideLayoutOptions = field(default_factory=SlideLayoutOptions)
    deck_title: str = "OpenHCS Figure Export"
    plate_name: Optional[str] = None


@dataclass(frozen=True)
class DeckExportResult:
    """Result of a completed deck export."""

    output_path: Path
    slide_count: int
