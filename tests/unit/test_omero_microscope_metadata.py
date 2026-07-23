from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from openhcs.microscopes.omero import OMEROMetadataHandler


def test_grid_dimensions_follow_canonical_key_across_annotation_namespaces(
    monkeypatch,
) -> None:
    map_annotation_type = object()
    omero_module = ModuleType("omero")
    omero_model_module = ModuleType("omero.model")
    omero_model_module.MapAnnotationI = map_annotation_type
    omero_module.model = omero_model_module
    monkeypatch.setitem(sys.modules, "omero", omero_module)
    monkeypatch.setitem(sys.modules, "omero.model", omero_model_module)

    annotation = SimpleNamespace(
        OMERO_TYPE=map_annotation_type,
        getNs=lambda: "polystore.metadata",
        getMapValue=lambda: (
            SimpleNamespace(name="polystore.parser", value="OMEROFilenameParser"),
            SimpleNamespace(
                name=OMEROMetadataHandler.GRID_DIMENSIONS_METADATA_KEY,
                value="3,4",
            ),
        ),
    )
    plate = SimpleNamespace(listAnnotations=lambda: (annotation,))
    connection = SimpleNamespace(getObject=lambda kind, plate_id: plate)
    backend = SimpleNamespace(_get_connection=lambda: connection)
    filemanager = SimpleNamespace(registry={"omero_local": backend})

    handler = OMEROMetadataHandler(filemanager)

    assert handler.get_grid_dimensions(17) == (3, 4)
