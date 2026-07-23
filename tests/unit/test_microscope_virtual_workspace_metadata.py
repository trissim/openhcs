from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants.constants import AllComponents
from openhcs.core.virtual_workspace_metadata import FIELDS
from openhcs.microscopes.opera_phenix import OperaPhenixHandler
from openhcs.core.source_workspace_projection import (
    VirtualWorkspaceSourceProjection,
)


def test_virtual_workspace_metadata_records_parser_owned_axis_values(
    tmp_path: Path,
    monkeypatch,
) -> None:
    handler = OperaPhenixHandler(SimpleNamespace())
    monkeypatch.setattr(
        handler.metadata_handler,
        "get_grid_dimensions",
        lambda plate_path: (3, 3),
    )
    monkeypatch.setattr(
        handler.metadata_handler,
        "get_pixel_size",
        lambda plate_path: 1.0,
    )
    virtual_a = "Images/r01c01f001p001-ch1sk1fk1fl1.tiff"
    virtual_b = "Images/r02c03f001p001-ch1sk1fk1fl1.tiff"

    handler.save_virtual_workspace_metadata(
        tmp_path,
        {
            virtual_a: SourcePixelRef("disk", "Images/source-a.tiff"),
            virtual_b: SourcePixelRef("disk", "Images/source-b.tiff"),
        },
    )

    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )
    source_metadata = metadata[FIELDS.SUBDIRECTORIES]["Images"][
        FIELDS.SOURCE_METADATA
    ]
    assert source_metadata[virtual_a][AllComponents.WELL.value] == "R01C01"
    assert source_metadata[virtual_b][AllComponents.WELL.value] == "R02C03"

    projection = VirtualWorkspaceSourceProjection.from_openhcs_metadata(
        tmp_path,
        metadata,
    )
    assert projection.pipeline_start_files(axis_id="R01C01") == (
        str(tmp_path / virtual_a),
    )
    assert projection.pipeline_start_files(axis_id="R02C03") == (
        str(tmp_path / virtual_b),
    )
