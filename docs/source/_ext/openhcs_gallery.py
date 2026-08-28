"""Sphinx projection for declaration-owned OpenHCS gallery scenarios."""

from __future__ import annotations

import os
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from scripts.gallery_catalog import (
    GalleryScenarioABC,
    documentation_gallery_scenario_for_id,
    gallery_asset_root,
    ui_window_reference_gallery_scenarios,
)

GALLERY_DIRECTIVE = "openhcs-gallery"
UI_WINDOW_REFERENCE_DIRECTIVE = "openhcs-ui-window-reference"


def _scenario_figure(
    directive: SphinxDirective,
    scenario: GalleryScenarioABC,
) -> nodes.figure:
    """Project one declared scenario into a documentation figure."""

    asset_root = gallery_asset_root()
    asset_path = asset_root / scenario.representative_image_path()
    dimensions = scenario.representative_image_dimensions(asset_root)
    source_path = Path(directive.state.document.current_source).resolve()
    relative_uri = os.path.relpath(asset_path, source_path.parent)

    image = nodes.image(
        "",
        uri=Path(relative_uri).as_posix(),
        alt=scenario.alt_text,
        width=f"{dimensions.width}px",
    )
    figure = nodes.figure("", image, classes=["openhcs-gallery-figure"])
    figure += nodes.caption("", scenario.heading)
    figure += nodes.legend("", nodes.paragraph("", scenario.description))
    directive.set_source_info(figure)
    return figure


class OpenHCSGalleryDirective(SphinxDirective):
    """Project one gallery scenario as an accessible documentation figure."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False

    def run(self) -> list[nodes.Node]:
        scenario = documentation_gallery_scenario_for_id(self.arguments[0].strip())
        return [_scenario_figure(self, scenario)]


class OpenHCSUiWindowReferenceDirective(SphinxDirective):
    """Project every registered stable UI window without a copied inventory."""

    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False

    def run(self) -> list[nodes.Node]:
        return [
            _scenario_figure(self, scenario)
            for scenario in ui_window_reference_gallery_scenarios()
        ]


def setup(app):
    app.add_directive(GALLERY_DIRECTIVE, OpenHCSGalleryDirective)
    app.add_directive(
        UI_WINDOW_REFERENCE_DIRECTIVE,
        OpenHCSUiWindowReferenceDirective,
    )
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
