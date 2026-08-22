from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from openhcs.constants.constants import AllComponents, Backend, Microscope
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.core.source_metadata import SourceMetadataRoleView
from openhcs.core.source_bindings import (
    ComponentSelector,
    ImagePlaneSource,
    ImportedMetadataJoin,
    ImportedMetadataTable,
    MetadataExtractionRule,
    MetadataSource,
    NamedSourceBinding,
    SourceBindingMatchDimension,
    SourceBindingMatchField,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSetRole,
    SourceSelector,
)
from openhcs.microscopes import create_microscope_handler
from openhcs.microscopes.bioformats_adapter import SourcePlaneStoreAdapter
from openhcs.microscopes.openhcs import (
    AtomicMetadataWriter,
    FIELDS,
    OpenHCSMetadataHandler,
    get_metadata_path,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser
from openhcs.microscopes.source_bindings_handler import SourceBindingsHandler
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager


def _write_tiff_stack(path: Path, values: tuple[int, ...]) -> None:
    stack = np.stack(
        [np.full((4, 4), value, dtype=np.uint16) for value in values],
        axis=0,
    )
    tifffile.imwrite(path, stack, metadata={"axes": "ZYX"})


def _filemanager() -> FileManager:
    ensure_storage_registry()
    return FileManager(dict(storage_registry))


def test_source_workspace_excludes_non_pixel_sidecars_before_projection(tmp_path):
    source_root = tmp_path / "source"
    source_root.mkdir()
    first = source_root / "A01_s001_w1.tif"
    second = source_root / "A01_s001_w2.tif"
    sidecar = source_root / "plate.HTD"
    for path in (first, second, sidecar):
        path.touch()

    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            source_filters=(
                SourceFilterClause(
                    subject=SourceFilterSubject.EXTENSION,
                    match_type=SourceFilterMatchType.IS_IMAGE,
                ),
            ),
            bindings=(
                NamedSourceBinding(
                    alias="Blue",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                NamedSourceBinding(
                    alias="Green",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                ),
            ),
        ),
        parser=SourceSchemaFilenameParser(),
    )
    candidates = projector.source_candidates(
        source_root,
        (first, second, sidecar),
        source_backend=Backend.DISK,
    )

    assert tuple(candidate.relative_path for candidate in candidates) == (
        first.name,
        second.name,
    )

    with pytest.raises(ValueError, match="must distinguish aliases"):
        projector.projection_set(
            source_root,
            (first, second, sidecar),
            filemanager=_filemanager(),
        )


def test_source_binding_workspace_projector_assigns_selector_channels(tmp_path):
    nuclei = tmp_path / "raw" / "nuclei_A01_s1.png"
    membrane = tmp_path / "raw" / "membrane_A01_s1.png"
    nuclei.parent.mkdir()
    nuclei.touch()
    membrane.touch()

    projector = SourceBindingWorkspaceProjector(
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
                        components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
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
                        components=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                    ),
                ),
            ),
        ),
        parser=SourceSchemaFilenameParser(),
    )
    projection_set = projector.projection_set(
        tmp_path,
        (nuclei, membrane),
        filemanager=_filemanager(),
    )

    by_alias = {
        projection.source_alias: projection for projection in projection_set.projections
    }

    assert by_alias["nuclei"].address.well == "A01"
    assert by_alias["nuclei"].address.site == "1"
    assert by_alias["nuclei"].address.channel == "1"
    assert by_alias["nuclei"].ref.backend_address == "raw/nuclei_A01_s1.png"
    assert by_alias["membrane"].address.well == "A01"
    assert by_alias["membrane"].address.site == "1"
    assert by_alias["membrane"].address.channel == "2"
    assert by_alias["membrane"].ref.backend_address == "raw/membrane_A01_s1.png"

    metadata = projection_set.metadata_dict(
        parser=projector.parser,
        microscope_handler_name=Microscope.SOURCE_BINDINGS.value,
        source_filename_parser_name=type(projector.parser).__name__,
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )
    serialized_sources = metadata[FIELDS.SOURCE_METADATA].values()
    assert {source["source_alias"] for source in serialized_sources} == {
        "nuclei",
        "membrane",
    }
    assert all("image_type" not in source for source in serialized_sources)
    filter_paths_by_alias = {
        str(source["source_alias"]): SourceMetadataRoleView(
            source
        ).source_filter_paths()
        for source in serialized_sources
    }
    assert filter_paths_by_alias == {
        "nuclei": (
            "raw/nuclei_A01_s1.png",
            str(nuclei).replace("\\", "/"),
        ),
        "membrane": (
            "raw/membrane_A01_s1.png",
            str(membrane).replace("\\", "/"),
        ),
    }


def test_source_binding_workspace_projects_semantic_identity_after_raw_selection(
    tmp_path,
):
    source = tmp_path / "A01_s1_w1.tif"
    tifffile.imwrite(source, np.zeros((4, 4), dtype=np.uint16))
    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=(
                        r"^(?P<Well>[A-Z][0-9]+)_s(?P<Site>[0-9]+)_w"
                        r"(?P<Channel>[0-9]+)[.]tif$"
                    ),
                ),
            ),
            bindings=(
                NamedSourceBinding(
                    alias="DNA",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                    ),
                    component_identity=(
                        ComponentSelector(AllComponents.CHANNEL, "MCP_DNA"),
                    ),
                ),
            ),
        ),
        parser=SourceSchemaFilenameParser(),
    )

    (projection,) = projector.projection_set(
        tmp_path,
        (source,),
        filemanager=_filemanager(),
    ).projections

    assert projection.address.channel == "MCP_DNA"
    original_metadata = dict(
        SourceMetadataRoleView(projection.source_metadata).original_items()
    )
    assert original_metadata[AllComponents.CHANNEL.value] == "1"


def test_source_binding_workspace_remaps_store_addresses_and_labels(tmp_path):
    for alias in ("DNA", "RNA"):
        tifffile.imwrite(
            tmp_path / f"B02_s3_{alias}.tif",
            np.zeros((4, 4), dtype=np.uint16),
        )

    def binding(alias: str, channel: str) -> NamedSourceBinding:
        return NamedSourceBinding(
            alias=alias,
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.ENDS_WITH,
                        f"_{alias}.tif",
                    ),
                ),
            ),
            component_identity=(ComponentSelector(AllComponents.CHANNEL, channel),),
        )

    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    MetadataSource.FILE_NAME,
                    r"^(?P<Well>[A-H][0-9]{2})_s"
                    r"(?P<Site>[0-9]+)_(?:DNA|RNA)[.]tif$",
                ),
            ),
            bindings=(binding("DNA", "1"), binding("RNA", "2")),
            match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        ),
        parser=SourceSchemaFilenameParser(),
    )
    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    projection_set = projector.projection_set_for_candidates(
        tmp_path,
        dataset.candidates,
        filemanager=_filemanager(),
    )
    metadata = projection_set.metadata_dict(
        parser=projector.parser,
        microscope_handler_name=Microscope.BIOFORMATS.value,
        source_filename_parser_name=type(projector.parser).__name__,
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )

    assert {
        (projection.address.well, projection.address.site)
        for projection in projection_set.projections
    } == {("B02", "3")}
    assert metadata[FIELDS.WELLS] == {"B02": None}
    assert metadata[FIELDS.CHANNELS] == {"1": "DNA", "2": "RNA"}


def test_source_binding_workspace_projects_declared_groups_to_wells(tmp_path):
    source_paths = []
    for run in ("Sequence1", "Sequence2"):
        run_dir = tmp_path / run
        run_dir.mkdir()
        for frame in ("0000", "0001"):
            path = run_dir / f"Embryo_GFP_{frame}.tif"
            path.touch()
            source_paths.append(path)

    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^.*_(?P<FrameNumber>[0-9]+)\.tif$",
                ),
                MetadataExtractionRule(
                    source=MetadataSource.FOLDER_NAME,
                    pattern=r".*[\\/](?P<Run>[^\\/]+)$",
                ),
            ),
            bindings=(
                NamedSourceBinding(
                    alias="OrigColor",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.CONTAINS,
                                value="GFP",
                            ),
                        ),
                    ),
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
            ),
            match_plan=SourceBindingMatchPlan(
                method=SourceBindingMatchMethod.ORDER,
            ),
            grouping_metadata_fields=("Run",),
        ),
        parser=SourceSchemaFilenameParser(),
    )

    projection_set = projector.projection_set(
        tmp_path,
        source_paths,
        filemanager=_filemanager(),
    )

    assert {
        (
            projection.address.well,
            projection.address.site,
            projection.address.channel,
            projection.address.z_index,
            projection.address.timepoint,
        )
        for projection in projection_set.projections
    } == {
        ("Sequence1", "1", "1", "1", "0"),
        ("Sequence1", "1", "1", "1", "1"),
        ("Sequence2", "1", "1", "1", "0"),
        ("Sequence2", "1", "1", "1", "1"),
    }


def test_source_binding_workspace_projects_registered_well_parts(tmp_path):
    source_paths = tuple(
        tmp_path / name
        for name in (
            "image-A-01.tif",
            "image-B-02.tif",
        )
    )
    for source_path in source_paths:
        source_path.touch()

    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=(r"^image-(?P<WellRow>[A-Z])-(?P<WellCol>[0-9]{2})\.tif$"),
                ),
            ),
            bindings=(NamedSourceBinding(alias="DNA"),),
            match_plan=SourceBindingMatchPlan(
                method=SourceBindingMatchMethod.ORDER,
            ),
            grouping_metadata_fields=("WellRow", "WellCol"),
        )
    )

    projection_set = projector.projection_set(
        tmp_path,
        source_paths,
        filemanager=_filemanager(),
    )

    assert {projection.address.well for projection in projection_set.projections} == {
        "A01",
        "B02",
    }


def test_group_address_preserves_distinct_source_well_as_literal_metadata(tmp_path):
    source_path = tmp_path / "Sequence1" / "A01_GFP_0000.tif"
    source_path.parent.mkdir()
    source_path.touch()

    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    source=MetadataSource.FILE_NAME,
                    pattern=r"^(?P<Well>[A-Z][0-9]+)_.*_(?P<FrameNumber>[0-9]+)\.tif$",
                ),
                MetadataExtractionRule(
                    source=MetadataSource.FOLDER_NAME,
                    pattern=r".*[\\/](?P<Run>[^\\/]+)$",
                ),
            ),
            bindings=(
                NamedSourceBinding(
                    alias="OrigColor",
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
            ),
            match_plan=SourceBindingMatchPlan(
                method=SourceBindingMatchMethod.ORDER,
            ),
            grouping_metadata_fields=("Run",),
        ),
        parser=SourceSchemaFilenameParser(),
    )

    projection_set = projector.projection_set(
        tmp_path,
        (source_path,),
        filemanager=_filemanager(),
    )
    metadata = projection_set.metadata_dict(
        parser=projector.parser,
        microscope_handler_name=Microscope.SOURCE_BINDINGS.value,
        source_filename_parser_name=type(projector.parser).__name__,
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )
    source_metadata = next(iter(metadata[FIELDS.SOURCE_METADATA].values()))

    assert source_metadata["well"] == "Sequence1"
    assert dict(SourceMetadataRoleView(source_metadata).original_items()) == {
        "Well": "A01",
        "FrameNumber": "0000",
        "Run": "Sequence1",
    }


def test_nonempty_source_bindings_select_their_declared_microscope_handler(
    tmp_path,
):
    handler = create_microscope_handler(
        microscope_type=Microscope.AUTO.value,
        plate_folder=tmp_path,
        filemanager=_filemanager(),
        source_bindings_config=SourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),)
        ),
    )

    assert isinstance(handler, SourceBindingsHandler)


def test_nonempty_source_bindings_override_format_specific_microscope(
    tmp_path,
):
    handler = create_microscope_handler(
        microscope_type=Microscope.IMAGEXPRESS.value,
        plate_folder=tmp_path,
        filemanager=_filemanager(),
        source_bindings_config=SourceBindingsConfig(
            bindings=(NamedSourceBinding(alias="DNA"),)
        ),
    )

    assert isinstance(handler, SourceBindingsHandler)


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
    ).projection_set(tmp_path, (dapi, actin), filemanager=_filemanager())

    by_alias = {
        projection.source_alias: projection for projection in projection_set.projections
    }

    assert by_alias["DAPI"].address.well == "A01"
    assert by_alias["DAPI"].address.site == "1"
    assert by_alias["DAPI"].address.channel == "1"
    assert by_alias["Actin"].address.well == "A01"
    assert by_alias["Actin"].address.site == "1"
    assert by_alias["Actin"].address.channel == "2"


def test_order_source_sets_join_imported_metadata_across_aliases(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    metadata_table = metadata_dir / "plate.csv"
    metadata_table.write_text(
        "WellID,SiteID,Compound,Dose\nA01,1,DMSO,0\n",
        encoding="utf-8",
    )
    dna = tmp_path / "DNA_A01.tif"
    actin = tmp_path / "Actin_1.tif"
    dna.touch()
    actin.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^DNA_(?P<Well>[A-Z][0-9]+)\.tif$",
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.STARTS_WITH,
                        "DNA_",
                    ),
                ),
            ),
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^Actin_(?P<Site>[0-9]+)\.tif$",
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.STARTS_WITH,
                        "Actin_",
                    ),
                ),
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.STARTS_WITH,
                            "DNA_",
                        ),
                    ),
                ),
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="Actin",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.STARTS_WITH,
                            "Actin_",
                        ),
                    ),
                ),
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="metadata/plate.csv",
                joins=(
                    ImportedMetadataJoin("Well", "WellID"),
                    ImportedMetadataJoin("Site", "SiteID"),
                ),
            ),
        ),
    )

    projections = (
        SourceBindingWorkspaceProjector(config)
        .projection_set(
            tmp_path,
            (dna, actin),
            filemanager=_filemanager(),
        )
        .projections
    )

    assert {projection.source_alias for projection in projections} == {"DNA", "Actin"}
    for projection in projections:
        original = dict(
            SourceMetadataRoleView(projection.source_metadata).original_items()
        )
        assert original["Compound"] == "DMSO"
        assert original["Dose"] == "0"


def test_order_source_sets_preserve_shared_virtual_stack_coordinates(tmp_path):
    dna = tmp_path / "DNA.tif"
    membrane = tmp_path / "Membrane.tif"
    _write_tiff_stack(dna, (1, 2, 3))
    _write_tiff_stack(membrane, (4, 5, 6))

    projection_set = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="DNA",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.STARTS_WITH,
                                "DNA",
                            ),
                        ),
                    ),
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                ),
                NamedSourceBinding(
                    alias="Membrane",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.STARTS_WITH,
                                "Membrane",
                            ),
                        ),
                    ),
                    component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
                ),
            ),
            match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
            source_stack_components=(AllComponents.Z_INDEX,),
        )
    ).projection_set(
        tmp_path,
        (dna, membrane),
        filemanager=_filemanager(),
    )

    assert tuple(
        (projection.source_alias, projection.address.z_index)
        for projection in projection_set.projections
    ) == (
        ("DNA", "1"),
        ("Membrane", "1"),
        ("DNA", "2"),
        ("Membrane", "2"),
        ("DNA", "3"),
        ("Membrane", "3"),
    )


def test_metadata_source_sets_propagate_imported_metadata(tmp_path):
    table = tmp_path / "plate.csv"
    table.write_text("WellID,Compound\nA01,DrugA\n", encoding="utf-8")
    dna = tmp_path / "A01_DNA.tif"
    actin = tmp_path / "A01_Actin.tif"
    dna.touch()
    actin.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Well>[A-Z][0-9]+)_(?:DNA|Actin)\.tif$",
            ),
        ),
        bindings=tuple(
            NamedSourceBinding(
                alias=alias,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.ENDS_WITH,
                            f"_{alias}.tif",
                        ),
                    ),
                ),
            )
            for alias in ("DNA", "Actin")
        ),
        match_plan=SourceBindingMatchPlan(
            SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=tuple(
                        SourceBindingMatchField(alias, "Well")
                        for alias in ("DNA", "Actin")
                    ),
                ),
            ),
        ),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Well", "WellID"),),
            ),
        ),
    )

    projections = (
        SourceBindingWorkspaceProjector(config)
        .projection_set(
            tmp_path,
            (dna, actin),
            filemanager=_filemanager(),
        )
        .projections
    )

    assert len(projections) == 2
    assert all(
        dict(SourceMetadataRoleView(item.source_metadata).original_items())["Compound"]
        == "DrugA"
        for item in projections
    )


def test_imported_metadata_duplicate_join_uses_first_matching_row(tmp_path):
    table = tmp_path / "plate.csv"
    table.write_text(
        "PlateID,WellID,Site,Compound\n" "20585,A01,1,First\n" "20585,A01,2,Second\n",
        encoding="utf-8",
    )
    image = tmp_path / "20585_A01_DNA.tif"
    image.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Plate>[0-9]+)_(?P<Well>[A-Z][0-9]+)_DNA\.tif$",
            ),
        ),
        bindings=(NamedSourceBinding(alias="DNA"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(
                    ImportedMetadataJoin("Plate", "PlateID"),
                    ImportedMetadataJoin("Well", "WellID"),
                ),
            ),
        ),
    )

    metadata = (
        SourceBindingWorkspaceProjector(config)
        .projection_set(
            tmp_path,
            (image,),
            filemanager=_filemanager(),
        )
        .projections[0]
        .source_metadata
    )

    assert metadata["Site"] == "1"
    assert metadata["Compound"] == "First"
    original = dict(SourceMetadataRoleView(metadata).original_items())
    assert original["Site"] == "1"
    assert original["Plate"] == "20585"
    assert original["Well"] == "A01"
    assert metadata["PlateID"] == "20585"
    assert metadata["WellID"] == "A01"
    assert original["PlateID"] == "20585"
    assert original["WellID"] == "A01"


def test_imported_metadata_later_stage_overrides_extracted_field(tmp_path):
    table = tmp_path / "plate.csv"
    table.write_text(
        "Well,Plate,Dose\nA01,plate_1,0\n",
        encoding="utf-8",
    )
    image = tmp_path / "BBBC013_A01_s1_w2.tif"
    image.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Plate>[^_]+)_(?P<Well>[A-Z][0-9]+)_s(?P<Site>[0-9]+)_w(?P<ChannelNumber>[0-9]+)\.tif$",
            ),
        ),
        bindings=(NamedSourceBinding(alias="DNA"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Well", "Well"),),
            ),
        ),
    )

    projection = (
        SourceBindingWorkspaceProjector(config)
        .projection_set(
            tmp_path,
            (image,),
            filemanager=_filemanager(),
        )
        .projections[0]
    )
    metadata = projection.source_metadata

    assert metadata["Plate"] == "plate_1"
    assert metadata["Dose"] == "0"
    assert projection.address.well == "A01"
    original = dict(SourceMetadataRoleView(metadata).original_items())
    assert original["Plate"] == "plate_1"
    assert original["Well"] == "A01"
    assert original["ChannelNumber"] == "2"


def test_imported_metadata_coerces_join_and_payload_values_through_declared_types(
    tmp_path,
):
    table = tmp_path / "plate.csv"
    table.write_text("Site,Dose,Frame\n05,0.25,7\n", encoding="utf-8")
    image = tmp_path / "5_DNA.tif"
    image.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Site>[0-9]+)_DNA\.tif$",
            ),
        ),
        bindings=(NamedSourceBinding(alias="DNA"),),
        match_plan=SourceBindingMatchPlan(
            SourceBindingMatchMethod.METADATA,
            dimensions=(
                SourceBindingMatchDimension(
                    fields=(SourceBindingMatchField("DNA", "Site"),),
                ),
            ),
        ),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Site", "Site"),),
            ),
        ),
        metadata_fields=(
            FieldSpec("Site", int, required=False),
            FieldSpec("Dose", float, required=False),
            FieldSpec("Frame", int, required=False),
        ),
    )

    projector = SourceBindingWorkspaceProjector(
        config,
        parser=SourceSchemaFilenameParser(),
    )
    projection_set = projector.projection_set(
        tmp_path,
        (image,),
        filemanager=_filemanager(),
    )
    projection = projection_set.projections[0]
    original = dict(SourceMetadataRoleView(projection.source_metadata).original_items())

    assert original["Site"] == 5
    assert original["Dose"] == 0.25
    assert original["Frame"] == 7

    metadata = projection_set.metadata_dict(
        parser=projector.parser,
        microscope_handler_name=Microscope.SOURCE_BINDINGS.value,
        source_filename_parser_name=type(projector.parser).__name__,
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )
    serialized = next(iter(metadata[FIELDS.SOURCE_METADATA].values()))
    serialized_original = dict(SourceMetadataRoleView(serialized).original_items())
    assert serialized_original["Dose"] == 0.25
    assert serialized_original["Frame"] == 7


@pytest.mark.parametrize(
    ("image_names", "table_text", "error"),
    (
        (("A01_DNA.tif", "B01_Actin.tif"), "WellID,Drug\nA01,A\n", "conflicting"),
        (("B01_DNA.tif",), "WellID,Drug\nA01,A\n", "no matching row"),
    ),
)
def test_imported_metadata_rejects_conflicting_or_unmatched_complete_joins(
    tmp_path,
    image_names,
    table_text,
    error,
):
    table = tmp_path / "plate.csv"
    table.write_text(table_text, encoding="utf-8")
    images = tuple(tmp_path / name for name in image_names)
    for image in images:
        image.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Well>[A-Z][0-9]+)_(?P<Stain>DNA|Actin)\.tif$",
            ),
        ),
        bindings=tuple(
            NamedSourceBinding(
                alias=stain,
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            SourceFilterSubject.FILE,
                            SourceFilterMatchType.ENDS_WITH,
                            f"_{stain}.tif",
                        ),
                    ),
                ),
            )
            for stain in ("DNA", "Actin")
            if any(name.endswith(f"_{stain}.tif") for name in image_names)
        ),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Well", "WellID"),),
            ),
        ),
    )

    with pytest.raises(ValueError, match=error):
        SourceBindingWorkspaceProjector(config).projection_set(
            tmp_path,
            images,
            filemanager=_filemanager(),
        )


def test_imported_metadata_skips_source_sets_with_partial_join_identity(tmp_path):
    table = tmp_path / "plate.csv"
    table.write_text("WellID,Compound\nA01,DrugA\n", encoding="utf-8")
    image = tmp_path / "DNA.tif"
    image.touch()
    config = SourceBindingsConfig(
        bindings=(NamedSourceBinding(alias="DNA"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Well", "WellID"),),
            ),
        ),
    )

    projection = (
        SourceBindingWorkspaceProjector(config)
        .projection_set(
            tmp_path,
            (image,),
            filemanager=_filemanager(),
        )
        .projections[0]
    )

    assert "Compound" not in dict(
        SourceMetadataRoleView(projection.source_metadata).original_items()
    )


def test_imported_metadata_location_is_resolved_exactly(tmp_path):
    nested = tmp_path / "metadata"
    nested.mkdir()
    (nested / "plate.csv").write_text(
        "WellID,Compound\nA01,DrugA\n",
        encoding="utf-8",
    )
    image = tmp_path / "A01_DNA.tif"
    image.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Well>[A-Z][0-9]+)_DNA\.tif$",
            ),
        ),
        bindings=(NamedSourceBinding(alias="DNA"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Well", "WellID"),),
            ),
        ),
    )

    with pytest.raises(FileNotFoundError, match="Declared source file does not exist"):
        SourceBindingWorkspaceProjector(config).projection_set(
            tmp_path,
            (image,),
            filemanager=_filemanager(),
        )


@pytest.mark.parametrize(
    ("location", "joins", "table_text", "error"),
    (
        (None, (ImportedMetadataJoin("Well", "WellID"),), None, "requires a declared"),
        ("plate.csv", (), "WellID,Drug\nA01,A\n", "requires at least one join"),
        (
            "plate.csv",
            (ImportedMetadataJoin("Well", "Missing"),),
            "WellID,Drug\nA01,A\n",
            "lacks declared join fields",
        ),
        (
            "plate.csv",
            (ImportedMetadataJoin("Well", "WellID"),),
            "WellID,Drug\n",
            "contains no data rows",
        ),
    ),
)
def test_imported_metadata_rejects_invalid_table_contracts(
    tmp_path,
    location,
    joins,
    table_text,
    error,
):
    if location is not None and table_text is not None:
        (tmp_path / location).write_text(table_text, encoding="utf-8")
    image = tmp_path / "A01_DNA.tif"
    image.touch()
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Well>[A-Z][0-9]+)_DNA\.tif$",
            ),
        ),
        bindings=(NamedSourceBinding(alias="DNA"),),
        imported_metadata_tables=(
            ImportedMetadataTable(location=location, joins=joins),
        ),
    )

    with pytest.raises((TypeError, ValueError), match=error):
        SourceBindingWorkspaceProjector(config).projection_set(
            tmp_path,
            (image,),
            filemanager=_filemanager(),
        )


def test_source_bindings_handler_does_not_reinterpret_prepared_workspace(tmp_path):
    image = tmp_path / "A01_DNA.tif"
    image.touch()
    table = tmp_path / "plate.csv"
    table.write_text("WellID,Compound\nA01,First\n", encoding="utf-8")
    config = SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                MetadataSource.FILE_NAME,
                r"^(?P<Well>[A-Z][0-9]+)_DNA\.tif$",
            ),
        ),
        bindings=(NamedSourceBinding(alias="DNA"),),
        imported_metadata_tables=(
            ImportedMetadataTable(
                location="plate.csv",
                joins=(ImportedMetadataJoin("Well", "WellID"),),
            ),
        ),
    )
    filemanager = _filemanager()
    handler = SourceBindingsHandler.create(
        filemanager=filemanager,
        source_bindings_config=config,
    )

    handler.initialize_workspace(tmp_path, filemanager)
    first_document = json.loads((tmp_path / "openhcs_metadata.json").read_text())
    first_metadata = next(
        iter(
            first_document["subdirectories"][FIELDS.DEFAULT_SUBDIRECTORY][
                FIELDS.SOURCE_METADATA
            ].values()
        )
    )
    assert (
        dict(SourceMetadataRoleView(first_metadata).original_items())["Compound"]
        == "First"
    )

    metadata_path = tmp_path / "openhcs_metadata.json"
    first_payload = metadata_path.read_bytes()
    table.write_text("WellID,Compound\nA01,Second\n", encoding="utf-8")

    handler.initialize_workspace(tmp_path, filemanager)
    second_document = json.loads(metadata_path.read_text())
    default_metadata = second_document["subdirectories"][FIELDS.DEFAULT_SUBDIRECTORY]
    second_metadata = next(iter(default_metadata[FIELDS.SOURCE_METADATA].values()))

    assert metadata_path.read_bytes() == first_payload
    assert (
        dict(SourceMetadataRoleView(second_metadata).original_items())["Compound"]
        == "First"
    )


def test_source_binding_workspace_projector_expands_declared_source_stack(tmp_path):
    stack = tmp_path / "nuclei_stack.tif"
    _write_tiff_stack(stack, (10, 20, 30))

    projection_set = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="Nuclei",
                    selector=SourceSelector(
                        components=(ComponentSelector(AllComponents.CHANNEL, "1"),),
                    ),
                ),
            ),
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
            source_stack_components=(AllComponents.Z_INDEX,),
        )
    ).projection_set(tmp_path, (stack,), filemanager=_filemanager())

    assert tuple(
        projection.address.z_index for projection in projection_set.projections
    ) == ("1", "2", "3")
    assert tuple(
        projection.ref.source_axis_indices for projection in projection_set.projections
    ) == ((0,), (1,), (2,))


def test_order_source_sets_pair_expanded_stack_planes_across_aliases(tmp_path):
    dna_stack = tmp_path / "DNA_stack.tif"
    membrane_stack = tmp_path / "Membrane_stack.tif"
    _write_tiff_stack(dna_stack, (10, 20, 30))
    _write_tiff_stack(membrane_stack, (40, 50, 60))

    projection_set = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="DNA",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="DNA_",
                            ),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="Membrane",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                subject=SourceFilterSubject.FILE,
                                match_type=SourceFilterMatchType.STARTS_WITH,
                                value="Membrane_",
                            ),
                        ),
                    ),
                ),
            ),
            match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
            source_stack_components=(AllComponents.Z_INDEX,),
        )
    ).projection_set(
        tmp_path,
        (dna_stack, membrane_stack),
        filemanager=_filemanager(),
    )

    assert tuple(
        (
            projection.source_alias,
            projection.address.z_index,
            projection.ref.source_axis_indices,
        )
        for projection in projection_set.projections
    ) == (
        ("DNA", "1", (0,)),
        ("Membrane", "1", (0,)),
        ("DNA", "2", (1,)),
        ("Membrane", "2", (1,)),
        ("DNA", "3", (2,)),
        ("Membrane", "3", (2,)),
    )


def test_source_binding_workspace_projector_order_uses_shortest_member_count(
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

    projection_set = projector.projection_set(
        tmp_path,
        (dapi_1, dapi_2, actin_1),
        filemanager=_filemanager(),
    )

    assert tuple(
        (projection.source_alias, projection.ref.backend_address)
        for projection in projection_set.projections
    ) == (("DAPI", "DAPI_001.png"), ("Actin", "Actin_001.png"))


def test_source_binding_workspace_broadcasts_explicit_single_members(tmp_path):
    dapi_1 = tmp_path / "DAPI_001.png"
    dapi_2 = tmp_path / "DAPI_002.png"
    flatfield = tmp_path / "flatfield.npy"
    dapi_1.touch()
    dapi_2.touch()
    np.save(flatfield, np.ones((4, 4), dtype=np.float32))
    explicit_source = ImagePlaneSource(
        uri=str(flatfield),
        series="2",
        index="3",
        channel="4",
    )
    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=(
                NamedSourceBinding(
                    alias="DAPI",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.STARTS_WITH,
                                "DAPI_",
                            ),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="Flatfield",
                    artifact_kind=ImageArtifactType,
                    source_set_role=SourceSetRole.BROADCAST,
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    explicit_source=explicit_source,
                ),
            ),
            image_plane_sources=(explicit_source,),
            match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        )
    )

    projection_set = projector.projection_set(
        tmp_path,
        (dapi_1, dapi_2),
        filemanager=_filemanager(),
    )

    flatfield_projections = tuple(
        projection
        for projection in projection_set.artifact_projections
        if projection.source_alias == "Flatfield"
    )
    assert len(flatfield_projections) == 2
    assert {projection.address.site for projection in flatfield_projections} == {
        "1",
        "2",
    }
    assert {projection.ref.backend_address for projection in flatfield_projections} == {
        "flatfield.npy"
    }


def test_metadata_source_sets_reuse_declared_partial_match_members(tmp_path):
    dapi_1 = tmp_path / "A01_s1_DAPI.tif"
    dapi_2 = tmp_path / "A01_s2_DAPI.tif"
    illumination = tmp_path / "plate_illum.tif"
    for path in (dapi_1, dapi_2, illumination):
        path.touch()
    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            metadata_rules=(
                MetadataExtractionRule(
                    MetadataSource.FILE_NAME,
                    r"^(?P<Well>[A-Z][0-9]+)_s(?P<Site>[0-9]+)_DAPI\.tif$",
                ),
            ),
            bindings=(
                NamedSourceBinding(
                    alias="DAPI",
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.ENDS_WITH,
                                "DAPI.tif",
                            ),
                        ),
                    ),
                ),
                NamedSourceBinding(
                    alias="Illumination",
                    artifact_kind=ImageArtifactType,
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.EQUALS,
                                illumination.name,
                            ),
                        ),
                    ),
                ),
            ),
            match_plan=SourceBindingMatchPlan(
                SourceBindingMatchMethod.METADATA,
                dimensions=(
                    SourceBindingMatchDimension(
                        fields=(SourceBindingMatchField("DAPI", "Well"),),
                    ),
                    SourceBindingMatchDimension(
                        fields=(SourceBindingMatchField("DAPI", "Site"),),
                    ),
                ),
            ),
        )
    )

    projection_set = projector.projection_set(
        tmp_path,
        (dapi_1, dapi_2, illumination),
        filemanager=_filemanager(),
    )

    illumination_projections = tuple(
        projection
        for projection in projection_set.artifact_projections
        if projection.source_alias == "Illumination"
    )
    assert len(illumination_projections) == 2
    assert {
        projection.ref.backend_address for projection in illumination_projections
    } == {"plate_illum.tif"}
    assert {projection.address.site for projection in illumination_projections} == {
        "1",
        "2",
    }


def test_source_binding_workspace_materializes_artifact_only_matched_groups(
    tmp_path,
):
    source_root = tmp_path / "source"
    workspace_root = tmp_path / "workspace"
    source_root.mkdir()
    first = source_root / "FirstObjects.tif"
    second = source_root / "SecondObjects.tif"
    first.touch()
    second.touch()
    projector = SourceBindingWorkspaceProjector(
        SourceBindingsConfig(
            bindings=tuple(
                NamedSourceBinding(
                    alias=alias,
                    artifact_kind=ObjectLabelsArtifactType,
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                    selector=SourceSelector(
                        filters=(
                            SourceFilterClause(
                                SourceFilterSubject.FILE,
                                SourceFilterMatchType.EQUALS,
                                path.name,
                            ),
                        ),
                    ),
                )
                for alias, path in (
                    ("FirstObjects", first),
                    ("SecondObjects", second),
                )
            ),
            match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
        ),
        parser=SourceSchemaFilenameParser(),
    )

    materialization = projector.materialize(
        source_root,
        workspace_root,
        filemanager=_filemanager(),
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=(first, second),
    )

    metadata = json.loads(materialization.metadata_path.read_text())
    default_metadata = metadata["subdirectories"][FIELDS.DEFAULT_SUBDIRECTORY]
    assert materialization.plane_mappings == {}
    assert len(materialization.artifact_mappings) == 2
    assert default_metadata["image_files"] == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]
    assert tuple(materialization.artifact_mappings) == tuple(
        default_metadata["image_files"]
    )


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
    assert (
        handler.get_source_filename_parser_name(image_dir)
        == "ImageXpressFilenameParser"
    )


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
                    projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
                ),
            ),
        ),
    )

    handler.initialize_workspace(tmp_path, filemanager)

    metadata = json.loads((tmp_path / "openhcs_metadata.json").read_text())
    default_metadata = metadata["subdirectories"][FIELDS.DEFAULT_SUBDIRECTORY]
    assert default_metadata["image_files"] == ["A01_s001_w1_z001_t001.tif"]
    assert default_metadata["workspace_mapping"][
        "_source/Illum/A01_s001_w1_z001_t001.tif"
    ] == {
        "backend": "disk",
        "backend_address": str(illumination),
        "source_axis_indices": [],
    }
