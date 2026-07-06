from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from openhcs.constants.constants import AllComponents, Microscope
from openhcs.core.artifacts import ImageArtifactType
from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector
from openhcs.core.source_bindings import (
    ComponentSelector,
    MetadataExtractionRule,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.openhcs import (
    AtomicMetadataWriter,
    FIELDS,
    OpenHCSMetadataHandler,
    get_metadata_path,
)
from openhcs.microscopes.source_bindings_handler import SourceBindingsHandler
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager


def _write_tiff_stack(path: Path, values: tuple[int, ...]) -> None:
    stack = np.stack(
        [np.full((4, 4), value, dtype=np.uint16) for value in values],
        axis=0,
    )
    tifffile.imwrite(path, stack, metadata={"axes": "ZYX"})


def test_source_binding_workspace_projector_assigns_selector_channels(tmp_path):
    nuclei = tmp_path / "raw" / "nuclei_A01_s1.png"
    membrane = tmp_path / "raw" / "membrane_A01_s1.png"
    nuclei.parent.mkdir()
    nuclei.touch()
    membrane.touch()

    projection_set = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r".*_(?P<well>[A-Z]\d+)_s(?P<site>\d+)\.png",
                ),
            ),
            bindings=(
                NamedSourceBinding(
                    alias="nuclei",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="nuclei_",
                            ),
                        ),
                        components=(
                            ComponentSelector(AllComponents.CHANNEL, "1"),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="membrane",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="membrane_",
                            ),
                        ),
                        components=(
                            ComponentSelector(AllComponents.CHANNEL, "2"),
                        ),
                    ),
                ),
            ),
        )
    ).projection_set(tmp_path, (nuclei, membrane))

    by_alias = {
        projection.source_alias: projection
        for projection in projection_set.projections
    }

    assert by_alias["nuclei"].address.well == "A01"
    assert by_alias["nuclei"].address.site == "1"
    assert by_alias["nuclei"].address.channel == "1"
    assert by_alias["nuclei"].ref.source_path == "raw/nuclei_A01_s1.png"
    assert by_alias["membrane"].address.well == "A01"
    assert by_alias["membrane"].address.site == "1"
    assert by_alias["membrane"].address.channel == "2"
    assert by_alias["membrane"].ref.source_path == "raw/membrane_A01_s1.png"


def test_source_binding_workspace_projector_order_matches_aliases(tmp_path):
    dapi = tmp_path / "DAPI_001.png"
    actin = tmp_path / "Actin_001.png"
    dapi.touch()
    actin.touch()

    projection_set = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="DAPI",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="DAPI_",
                            ),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="Actin",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="Actin_",
                            ),
                        ),
                    ),
                ),
            ),
            match_plan=SourceBindingMatchPlan(
                method=SourceBindingMatchMethod.ORDER,
            ),
        )
    ).projection_set(tmp_path, (dapi, actin))

    by_alias = {
        projection.source_alias: projection
        for projection in projection_set.projections
    }

    assert by_alias["DAPI"].address.well == "A01"
    assert by_alias["DAPI"].address.site == "1"
    assert by_alias["DAPI"].address.channel == "1"
    assert by_alias["Actin"].address.well == "A01"
    assert by_alias["Actin"].address.site == "1"
    assert by_alias["Actin"].address.channel == "2"


def test_source_binding_workspace_projector_expands_tiff_stack_planes(tmp_path):
    stack = tmp_path / "nuclei_stack.tif"
    _write_tiff_stack(stack, (10, 20, 30))

    projection_set = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="Nuclei",
                    selector=SourceSelector(
                        components=(
                            ComponentSelector(AllComponents.CHANNEL, "1"),
                        ),
                    ),
                ),
            ),
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        )
    ).projection_set(tmp_path, (stack,))

    assert tuple(
        projection.address.z_index
        for projection in projection_set.projections
    ) == ("1", "2", "3")
    assert tuple(
        projection.ref.plane_index
        for projection in projection_set.projections
    ) == (0, 1, 2)
    assert tuple(
        projection.ref.source_z_index
        for projection in projection_set.projections
    ) == (1, 2, 3)


def test_source_binding_workspace_projector_rejects_incomplete_order_matches(
    tmp_path,
):
    dapi_1 = tmp_path / "DAPI_001.png"
    dapi_2 = tmp_path / "DAPI_002.png"
    actin_1 = tmp_path / "Actin_001.png"
    dapi_1.touch()
    dapi_2.touch()
    actin_1.touch()

    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="DAPI",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="DAPI_",
                            ),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="Actin",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="Actin_",
                            ),
                        ),
                    ),
                ),
            ),
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        )
    )

    with pytest.raises(ValueError, match="incomplete"):
        projector.projection_set(tmp_path, (dapi_1, dapi_2, actin_1))


def test_replace_subdirectory_metadata_preserves_unmanaged_subdirectories(tmp_path):
    metadata_path = get_metadata_path(tmp_path)
    metadata_path.write_text(
        json.dumps(
            {
                "subdirectories": {
                    FIELDS.DEFAULT_SUBDIRECTORY: {
                        "image_files": ["old.tif"],
                        "available_backends": {"disk": True, "zarr": True},
                    },
                    "analysis": {
                        "image_files": ["measurements.csv"],
                        "available_backends": {"disk": True},
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    AtomicMetadataWriter().replace_subdirectory_metadata(
        metadata_path,
        FIELDS.DEFAULT_SUBDIRECTORY,
        {
            "image_files": ["new.tif"],
            "available_backends": {"disk": True, "virtual_workspace": True},
        },
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["subdirectories"][FIELDS.DEFAULT_SUBDIRECTORY] == {
        "image_files": ["new.tif"],
        "available_backends": {"disk": True, "virtual_workspace": True},
    }
    assert metadata["subdirectories"]["analysis"] == {
        "image_files": ["measurements.csv"],
        "available_backends": {"disk": True},
    }


def test_openhcs_metadata_handler_resolves_subdirectory_inputs(tmp_path):
    plate = tmp_path / "plate"
    image_dir = plate / "TimePoint_1"
    image_dir.mkdir(parents=True)
    metadata_path = get_metadata_path(plate)
    metadata_path.write_text(
        json.dumps(
            {
                "subdirectories": {
                    "TimePoint_1": {
                        "microscope_handler_name": "imagexpress",
                        "source_filename_parser_name": "ImageXpressFilenameParser",
                        "grid_dimensions": [2, 2],
                        "pixel_size": 0.65,
                        "image_files": ["TimePoint_1/A01_s001_w1_z001_t001.tif"],
                        "channels": {"1": "DAPI"},
                        "wells": {"A01": "A01"},
                        "sites": {"1": "1"},
                        "z_indexes": {"1": "1"},
                        "timepoints": {"1": "1"},
                        "available_backends": {"disk": True},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    ensure_storage_registry()
    handler = OpenHCSMetadataHandler(FileManager(dict(storage_registry)))

    assert handler.find_metadata_file(image_dir) == metadata_path
    assert handler.get_grid_dimensions(image_dir) == (2, 2)
    assert handler.get_pixel_size(image_dir) == 0.65
    assert handler.get_source_filename_parser_name(image_dir) == "ImageXpressFilenameParser"


def test_source_bindings_handler_is_registry_constructed():
    handler = create_microscope_handler(
        microscope_type=Microscope.SOURCE_BINDINGS.value,
        plate_folder=Path("."),
        filemanager=object(),
        source_bindings_config=SourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),),
        ),
    )

    assert isinstance(handler, SourceBindingsHandler)


def test_source_bindings_handler_materializes_non_stack_source_artifacts(tmp_path):
    source_image = tmp_path / "Channel 1-A01.png"
    source_image.touch()
    illumination = tmp_path / "illumination.npy"
    np.save(illumination, np.ones((4, 4), dtype=np.float32))
    ensure_storage_registry()
    filemanager = FileManager(dict(storage_registry))
    handler = SourceBindingsHandler(
        filemanager,
        source_bindings_config=SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="Orig",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.CONTAINS,
                                value="Channel 1",
                            ),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="Illum",
                    artifact_kind=ImageArtifactType,
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.EQUALS,
                                value="illumination.npy",
                            ),
                        ),
                    ),
                    participates_in_image_stack=False,
                ),
            ),
        ),
    )

    handler.initialize_workspace(tmp_path, filemanager)

    metadata = json.loads((tmp_path / "openhcs_metadata.json").read_text())
    default_metadata = metadata["subdirectories"][FIELDS.DEFAULT_SUBDIRECTORY]
    source_metadata = metadata["subdirectories"]["_source"]
    assert default_metadata["image_files"] == ["A01_s001_w1_z001_t001.png"]
    assert source_metadata["image_files"] == [
        "_source/Illum/001_illumination.npy"
    ]
    assert (
        source_metadata["workspace_mapping"]["_source/Illum/001_illumination.npy"]
        == "_source/Illum/illumination.npy"
    )
