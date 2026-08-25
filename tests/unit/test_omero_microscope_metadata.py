from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from openhcs.formats.pattern.pattern_discovery import PatternDiscoveryEngine
from openhcs.microscopes.omero import OMEROFilenameParser, OMEROMetadataHandler


def test_omero_filename_parser_projects_polystore_address_declaration() -> None:
    parser = OMEROFilenameParser()

    filename = parser.construct_filename(
        parser.bind_component_values(
            {
                "well": "AA01",
                "site": 9,
                "channel": 2,
                "z_index": 3,
                "timepoint": 4,
            },
            extension=".ome.tif",
        )
    )

    assert filename == "AA01_s009_w2_z003_t004.ome.tif"
    parsed = parser.parse_filename(filename)
    assert parsed is not None
    assert dict(parsed.wire_mapping()) == {
        "well": "AA01",
        "site": 9,
        "channel": 2,
        "z_index": 3,
        "timepoint": 4,
        "extension": ".ome.tif",
    }
    assert parser.extract_component_coordinates("AA01") == ("AA", "01")


def test_omero_pattern_discovery_round_trips_symbolic_site() -> None:
    parser = OMEROFilenameParser()
    patterns = PatternDiscoveryEngine(
        parser,
        SimpleNamespace(),
    ).auto_detect_patterns_from_axis_files(
        [
            "/omero/plate_1/A01_s001_w2_z003_t001.tif",
            "/omero/plate_1/A01_s002_w2_z003_t001.tif",
        ],
        axis_id="A01",
        variable_components=["site"],
    )

    assert patterns == {"A01": ["A01_s{iii}_w2_z003_t001.tif"]}
    pattern = patterns["A01"][0]
    parsed = parser.parse_filename(pattern)
    assert parsed is not None
    assert parsed.value_for(parser.component_for_name("site")) == "{iii}"


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
