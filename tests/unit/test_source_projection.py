from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
from polystore.bioformats_storage import BioFormatsPlaneRef

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.source_bindings import SourceProjectionRole
from openhcs.core.source_metadata import (
    ORIGINAL_SOURCE_METADATA_FIELD,
    SourceMetadataRoleView,
)
from openhcs.core.source_projection import (
    OpenHCSPlaneAddress,
    SourceArtifactProjection,
    SourcePixelRef,
    SourcePlaneProjection,
    SourceProjectionSet,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


_OWNED_PROJECTION_MODULES = (
    Path("openhcs/core/source_projection.py"),
    Path("openhcs/core/source_binding_workspace.py"),
    Path("openhcs/core/source_workspace_projection.py"),
)


def test_projection_nominal_owners_declare_projection_semantics() -> None:
    address = OpenHCSPlaneAddress(
        well="A01",
        site="1",
        channel="1",
        z_index="1",
        timepoint="1",
    )
    plane = SourcePlaneProjection(
        address=address,
        ref=SourcePixelRef("disk", "plane.tif"),
        source_alias="DNA",
    )
    artifact = SourceArtifactProjection(
        address=address,
        ref=SourcePixelRef("disk", "labels.tif"),
        source_alias="Nuclei",
        artifact_kind=ObjectLabelsArtifactType,
    )

    assert plane.projection_role is SourceProjectionRole.PRIMARY_PLANE
    assert plane.source_alias == "DNA"
    assert plane.artifact_kind is ImageArtifactType
    assert plane.identity_key == (SourceProjectionRole.PRIMARY_PLANE, address)
    assert plane.payload_composition_alias == "DNA"
    assert plane.virtual_workspace_path("plane.tif", execution_anchor=False) == (
        "plane.tif"
    )

    assert artifact.projection_role is SourceProjectionRole.SOURCE_ARTIFACT
    assert artifact.source_alias == "Nuclei"
    assert artifact.artifact_kind is ObjectLabelsArtifactType
    assert artifact.identity_key == (
        SourceProjectionRole.SOURCE_ARTIFACT,
        address,
        "Nuclei",
    )
    assert artifact.payload_composition_alias is None
    assert artifact.virtual_workspace_path(
        "labels.tif",
        execution_anchor=False,
    ) == "_source/Nuclei/labels.tif"

    plane_metadata: dict[str, object] = {}
    plane_payload: dict[str, object] = {}
    plane.extend_source_metadata(plane_metadata)
    plane.extend_serialized_payload(plane_payload)
    assert plane_metadata == {}
    assert plane_payload == {}

    artifact_metadata: dict[str, object] = {}
    artifact_payload: dict[str, object] = {}
    artifact.extend_source_metadata(artifact_metadata)
    artifact.extend_serialized_payload(artifact_payload)
    assert artifact_metadata == {"source_artifact_type": "object_labels"}
    assert artifact_payload == {"artifact_kind": "object_labels"}


def test_owned_projection_modules_do_not_dispatch_on_concrete_projection_types() -> None:
    forbidden_types = {"SourcePlaneProjection", "SourceArtifactProjection"}
    for relative_path in _OWNED_PROJECTION_MODULES:
        source_path = Path(__file__).parents[2] / relative_path
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "isinstance"
                or len(node.args) < 2
            ):
                continue
            dispatched_types = {
                child.id
                for child in ast.walk(node.args[1])
                if isinstance(child, ast.Name)
            }
            assert forbidden_types.isdisjoint(dispatched_types), relative_path


def test_source_projection_serializes_canonical_virtual_filename() -> None:
    projection_set = SourceProjectionSet(
        (
            SourcePlaneProjection(
                address=OpenHCSPlaneAddress(
                    well="A01",
                    site="1",
                    channel="2",
                    z_index="3",
                    timepoint="4",
                ),
                ref=SourcePixelRef(
                    backend="bioformats",
                    backend_address=BioFormatsPlaneRef(
                        source_path="stack.ome.tif",
                        series_index=5,
                        plane_index=6,
                    ).to_backend_address(),
                ),
                component_labels={
                    "channel": "DAPI",
                    "well": "A01",
                    "site": "Site 1",
                    "z_index": "Z3",
                    "timepoint": "T4",
                },
            ),
        )
    )

    metadata = projection_set.metadata_dict(
        parser=SourceSchemaFilenameParser(),
        microscope_handler_name="bioformats",
        source_filename_parser_name="SourceSchemaFilenameParser",
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )

    assert metadata["image_files"] == ["A01_s001_w2_z003_t004.tif"]
    assert metadata["channels"] == {"2": "DAPI"}
    assert metadata["z_indexes"] == {"3": "Z3"}
    assert metadata["workspace_mapping"]["A01_s001_w2_z003_t004.tif"] == {
        "backend": "bioformats",
        "backend_address": (
            '{"plane_index":6,"series_index":5,'
            '"source_path":"stack.ome.tif"}'
        ),
        "source_axis_indices": [],
    }
    assert metadata["source_projection"][0]["address"] == {
        "well": "A01",
        "site": "1",
        "channel": "2",
        "z_index": "3",
        "timepoint": "4",
    }


def test_source_plane_address_canonicalizes_numeric_axis_padding() -> None:
    address = OpenHCSPlaneAddress(
        well="01",
        site="001",
        channel="02",
        z_index="003",
        timepoint="0004",
    )

    assert address == OpenHCSPlaneAddress(
        well="01",
        site="1",
        channel="2",
        z_index="3",
        timepoint="4",
    )


def test_source_projection_rejects_metadata_component_conflict() -> None:
    projection_set = SourceProjectionSet(
        (
            SourcePlaneProjection(
                address=OpenHCSPlaneAddress(
                    well="A01",
                    site="1",
                    channel="2",
                    z_index="3",
                    timepoint="4",
                ),
                ref=SourcePixelRef(
                    backend="disk",
                    backend_address="image.tif",
                ),
                source_metadata={"z_index": "99"},
            ),
        )
    )

    with pytest.raises(ValueError, match="conflicts with canonical z_index"):
        projection_set.metadata_dict(
            parser=SourceSchemaFilenameParser(),
            microscope_handler_name="openhcs",
            source_filename_parser_name="SourceSchemaFilenameParser",
            grid_dimensions=[1, 1],
            pixel_size=1.0,
        )


@pytest.mark.parametrize(
    ("component", "source_value", "address_value"),
    (
        (AllComponents.WELL, "A01", "fields"),
        (AllComponents.SITE, "7", "1"),
        (AllComponents.CHANNEL, "9", "2"),
        (AllComponents.Z_INDEX, "8", "3"),
        (AllComponents.TIMEPOINT, "6", "4"),
    ),
)
def test_source_projection_preserves_provenance_owned_component_remaps(
    component: AllComponents,
    source_value: str,
    address_value: str,
) -> None:
    address = replace(
        OpenHCSPlaneAddress(
            well="A01",
            site="1",
            channel="2",
            z_index="3",
            timepoint="4",
        ),
        **{component.value: address_value},
    )
    projection_set = SourceProjectionSet(
        (
            SourcePlaneProjection(
                address=address,
                ref=SourcePixelRef(
                    backend="disk",
                    backend_address="image.tif",
                ),
                source_metadata={
                    component.value: source_value,
                    ORIGINAL_SOURCE_METADATA_FIELD: {
                        component.value: source_value,
                    },
                },
            ),
        )
    )

    metadata = projection_set.metadata_dict(
        parser=SourceSchemaFilenameParser(),
        microscope_handler_name="openhcs",
        source_filename_parser_name="SourceSchemaFilenameParser",
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )
    source_metadata = next(iter(metadata["source_metadata"].values()))

    assert source_metadata[component.value] == address_value
    assert dict(SourceMetadataRoleView(source_metadata).original_items())[
        component.value
    ] == source_value


def test_source_projection_rejects_duplicate_addresses() -> None:
    address = OpenHCSPlaneAddress(
        well="A01",
        site="1",
        channel="1",
        z_index="1",
        timepoint="1",
    )

    with pytest.raises(ValueError, match="Duplicate source projection address"):
        SourceProjectionSet(
            (
                SourcePlaneProjection(
                    address=address,
                    ref=SourcePixelRef("disk", "a.tif"),
                ),
                SourcePlaneProjection(
                    address=address,
                    ref=SourcePixelRef("disk", "b.tif"),
                ),
            )
        )


def test_artifact_only_projection_set_serializes_typed_execution_anchors() -> None:
    projections = tuple(
        SourceArtifactProjection(
            address=OpenHCSPlaneAddress(
                well="A01",
                site="1",
                channel=str(channel),
                z_index="1",
                timepoint="1",
            ),
            ref=SourcePixelRef("disk", source_path),
            source_alias=alias,
            artifact_kind=ObjectLabelsArtifactType,
        )
        for channel, alias, source_path in (
            (1, "FirstObjects", "first.tif"),
            (2, "SecondObjects", "second.tif"),
        )
    )
    projection_set = SourceProjectionSet(projections)

    metadata = projection_set.metadata_dict(
        parser=SourceSchemaFilenameParser(),
        microscope_handler_name="source_bindings",
        source_filename_parser_name="SourceSchemaFilenameParser",
        grid_dimensions=[1, 1],
        pixel_size=1.0,
    )

    assert metadata["image_files"] == [
        "A01_s001_w1_z001_t001.tif",
        "A01_s001_w2_z001_t001.tif",
    ]
    assert tuple(metadata["workspace_mapping"]) == tuple(metadata["image_files"])
    assert {
        item["source_alias"]: (item["projection_role"], item["artifact_kind"])
        for item in metadata["source_projection"]
    } == {
        "FirstObjects": ("source_artifact", "object_labels"),
        "SecondObjects": ("source_artifact", "object_labels"),
    }
