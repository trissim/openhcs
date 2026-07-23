"""Exact source-binding workspace ingestion boundaries."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
from polystore.disk import DiskBackend
from polystore.filemanager import FileManager
from polystore.virtual_workspace import SourcePixelRef

from openhcs.constants.constants import AllComponents, Backend
from openhcs.core.source_binding_context import SourceBindingContext
from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector
from openhcs.core.source_projection import SourceCandidate
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


def _field_names(record_type: type[object]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(record_type))


def _two_channel_config(
    *,
    source_stack_components: tuple[AllComponents, ...] = (),
) -> SourceBindingsConfig:
    return SourceBindingsConfig(
        metadata_rules=(
            MetadataExtractionRule(
                source=MetadataSource.FILE_NAME,
                pattern=(
                    r"^(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9]+)_" r"(?P<Stain>[^.]+)"
                ),
            ),
        ),
        match_plan=SourceBindingMatchPlan(method=SourceBindingMatchMethod.ORDER),
        source_filters=(
            SourceFilterClause(
                subject=SourceFilterSubject.EXTENSION,
                match_type=SourceFilterMatchType.IS_IMAGE,
            ),
        ),
        bindings=(
            NamedSourceBinding(
                alias="DNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="DNA",
                        ),
                    ),
                ),
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "1"),),
            ),
            NamedSourceBinding(
                alias="RNA",
                selector=SourceSelector(
                    filters=(
                        SourceFilterClause(
                            subject=SourceFilterSubject.FILE,
                            match_type=SourceFilterMatchType.CONTAINS,
                            value="RNA",
                        ),
                    ),
                ),
                component_identity=(ComponentSelector(AllComponents.CHANNEL, "2"),),
            ),
        ),
        source_stack_components=source_stack_components,
    )


def _write_image(path: Path, value: int) -> None:
    np.save(path, np.full((4, 4), value, dtype=np.uint16))


def _filemanager() -> FileManager:
    return FileManager({Backend.DISK.value: DiskBackend()})


def test_source_binding_context_has_exact_public_workspace_inputs() -> None:
    assert _field_names(SourceBindingContext) == (
        "logical_plate_id",
        "display_plate_root",
        "execution_plate_path",
        "source_bindings",
        "filemanager",
        "source_backend",
    )


def test_source_candidate_owns_one_exact_source_ref() -> None:
    source_ref = SourcePixelRef("disk", "/source/A01_w1.tif")
    candidate = SourceCandidate(
        source_ref=source_ref,
        relative_path="A01_w1.tif",
        metadata={},
    )

    assert candidate.source_ref is source_ref


def test_component_projection_registry_exactly_covers_all_components() -> None:
    from openhcs.core.source_metadata import SourceComponentProjectionStrategy

    assert {
        registered_type.strategy_key
        for registered_type in SourceComponentProjectionStrategy.registered_strategy_types()
    } == set(AllComponents)


def test_workspace_projection_uses_exact_submitted_root_and_file_universe(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "parent"
    source_root = parent / "source"
    nested = source_root / "nested"
    source_root.mkdir(parents=True)
    nested.mkdir()
    _write_image(parent / "A01_s1_DNA.npy", 99)
    _write_image(nested / "A01_s1_DNA.npy", 88)
    dna = source_root / "A01_s1_DNA.npy"
    rna = source_root / "A01_s1_RNA.npy"
    _write_image(dna, 1)
    _write_image(rna, 2)

    projection_set = SourceBindingWorkspaceProjector(
        _two_channel_config()
    ).projection_set(source_root, (dna, rna), filemanager=_filemanager())

    assert len(projection_set.projections) == 2
    assert tuple(
        projection.ref.backend for projection in projection_set.projections
    ) == (Backend.DISK.value, Backend.DISK.value)
    assert tuple(
        projection.ref.backend_address for projection in projection_set.projections
    ) == ("A01_s1_DNA.npy", "A01_s1_RNA.npy")
    assert all(
        "nested" not in projection.ref.backend_address
        for projection in projection_set.projections
    )


def test_workspace_projection_applies_config_filters_to_submitted_files(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    dna = source_root / "A01_s1_DNA.npy"
    rna = source_root / "A01_s1_RNA.npy"
    ignored = source_root / "A01_s1_DNA.txt"
    _write_image(dna, 1)
    _write_image(rna, 2)
    ignored.write_text("not an image", encoding="utf-8")

    projection_set = SourceBindingWorkspaceProjector(
        _two_channel_config()
    ).projection_set(
        source_root,
        (dna, rna, ignored),
        filemanager=_filemanager(),
    )

    assert tuple(
        projection.ref.backend_address for projection in projection_set.projections
    ) == ("A01_s1_DNA.npy", "A01_s1_RNA.npy")


def test_workspace_projection_materializes_complete_declared_universe(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    files = tuple(
        source_root / name
        for name in (
            "A01_s1_DNA.npy",
            "A01_s1_RNA.npy",
            "A01_s2_DNA.npy",
            "A01_s2_RNA.npy",
        )
    )
    for index, path in enumerate(files):
        _write_image(path, index)

    projection_set = SourceBindingWorkspaceProjector(
        _two_channel_config()
    ).projection_set(source_root, files, filemanager=_filemanager())

    assert len(projection_set.projections) == 4
    assert {
        (projection.address.site, projection.address.channel)
        for projection in projection_set.projections
    } == {("1", "1"), ("1", "2"), ("2", "1"), ("2", "2")}


def test_workspace_projection_fails_when_required_alias_has_no_source(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    dna = source_root / "A01_s1_DNA.npy"
    _write_image(dna, 1)

    with pytest.raises(ValueError, match="RNA|matched no"):
        SourceBindingWorkspaceProjector(_two_channel_config()).projection_set(
            source_root,
            (dna,),
            filemanager=_filemanager(),
        )


def test_source_stack_expansion_is_declared_by_config(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    dna = source_root / "A01_s1_DNA.npy"
    rna = source_root / "A01_s1_RNA.npy"
    np.save(dna, np.stack((np.full((4, 4), 1), np.full((4, 4), 2))))
    np.save(rna, np.stack((np.full((4, 4), 3), np.full((4, 4), 4))))

    projection_set = SourceBindingWorkspaceProjector(
        _two_channel_config(source_stack_components=(AllComponents.TIMEPOINT,))
    ).projection_set(source_root, (dna, rna), filemanager=_filemanager())

    assert len(projection_set.projections) == 4
    assert {
        projection.address.timepoint for projection in projection_set.projections
    } == {"1", "2"}
    assert {
        projection.ref.source_axis_indices for projection in projection_set.projections
    } == {(0,), (1,)}


def test_undeclared_source_stack_remains_one_source_per_binding(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    dna = source_root / "A01_s1_DNA.npy"
    rna = source_root / "A01_s1_RNA.npy"
    np.save(dna, np.stack((np.full((4, 4), 1), np.full((4, 4), 2))))
    np.save(rna, np.stack((np.full((4, 4), 3), np.full((4, 4), 4))))

    projection_set = SourceBindingWorkspaceProjector(
        _two_channel_config()
    ).projection_set(source_root, (dna, rna), filemanager=_filemanager())

    assert len(projection_set.projections) == 2
    assert {
        projection.ref.source_axis_indices for projection in projection_set.projections
    } == {()}


def test_workspace_projection_refs_use_exact_polystore_source_pixel_ref_type(
    tmp_path: Path,
) -> None:
    import polystore

    source_ref_type = getattr(polystore, "SourcePixelRef", None)
    assert source_ref_type is not None
    assert tuple(field.name for field in fields(source_ref_type)) == (
        "backend",
        "backend_address",
        "source_axis_indices",
    )

    source_root = tmp_path / "source"
    source_root.mkdir()
    dna = source_root / "A01_s1_DNA.npy"
    rna = source_root / "A01_s1_RNA.npy"
    _write_image(dna, 1)
    _write_image(rna, 2)
    projection_set = SourceBindingWorkspaceProjector(
        _two_channel_config()
    ).projection_set(source_root, (dna, rna), filemanager=_filemanager())

    assert all(
        isinstance(projection.ref, source_ref_type)
        for projection in projection_set.projections
    )
    assert all(
        source_ref_type.from_workspace_mapping(projection.ref.to_workspace_mapping())
        == projection.ref
        for projection in projection_set.projections
    )
