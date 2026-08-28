"""Sphinx projection for declaration-owned OpenHCS gallery scenarios."""

from __future__ import annotations

import os
from pathlib import Path

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

from scripts.gallery_catalog import (
    OpenHCSGalleryScenarioCatalog,
    gallery_asset_root,
)

GALLERY_DIRECTIVE = "openhcs-gallery"


class OpenHCSGalleryDirective(SphinxDirective):
    """Project one gallery scenario as an accessible documentation figure."""

    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    has_content = False

    def run(self) -> list[nodes.Node]:
        scenario = OpenHCSGalleryScenarioCatalog.for_id(self.arguments[0].strip())
        asset_path = gallery_asset_root() / scenario.representative_image_path()
        source_path = Path(self.state.document.current_source).resolve()
        relative_uri = os.path.relpath(asset_path, source_path.parent)

        image = nodes.image(
            "",
            uri=Path(relative_uri).as_posix(),
            alt=scenario.alt_text,
            width=f"{scenario.width}px",
        )
        figure = nodes.figure("", image, classes=["openhcs-gallery-figure"])
        figure += nodes.caption("", scenario.heading)
        figure += nodes.legend("", nodes.paragraph("", scenario.description))
        self.set_source_info(figure)
        return [figure]


def setup(app):
    app.add_directive(GALLERY_DIRECTIVE, OpenHCSGalleryDirective)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
