"""Generic image-file and file-bundle materialization contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from polystore.disk import DiskStorageBackend
from polystore.filemanager import FileManager
from polystore.memory import MemoryStorageBackend

from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.processing.materialization import options as materialization_options
from openhcs.processing.materialization.constants import WriteMode
from openhcs.processing.materialization.core import (
    MaterializationSpec,
    materialization_outputs,
    materialize,
)
from openhcs.processing.materialization.options import (
    FileOutputOptions,
    MaterializedFilenameIdentity,
    SourceOptions,
)


def _option_types():
    image_options = getattr(materialization_options, "ImageFileOptions", None)
    bundle_options = getattr(materialization_options, "FileBundleOptions", None)
    assert image_options is not None
    assert bundle_options is not None
    return image_options, bundle_options


def test_image_and_bundle_options_extend_existing_writer_options() -> None:
    image_options, bundle_options = _option_types()

    assert issubclass(image_options, FileOutputOptions)
    assert issubclass(image_options, SourceOptions)
    assert issubclass(bundle_options, FileOutputOptions)
    assert bundle_options().filename_identity is (
        MaterializedFilenameIdentity.ARTIFACT_NAME
    )


@pytest.mark.parametrize("suffix", (".png", ".tif", ".npy"))
def test_image_file_options_dispatch_registered_image_suffixes(suffix: str) -> None:
    image_options, _bundle_options = _option_types()
    spec = MaterializationSpec(image_options(relative_path_template=f"preview{suffix}"))

    assert spec.candidate_paths("/analysis/output.pkl") == (
        f"/analysis/preview{suffix}",
    )


def test_source_identity_image_file_projects_addressable_stack_planes() -> None:
    image_options, _bundle_options = _option_types()
    planes = tuple(np.full((4, 5, 3), value, dtype=np.uint8) for value in (1, 2, 3))
    payload = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/input/site{index}.tif" for index in range(1, 4)),
            component_metadata=tuple({"site": str(index)} for index in range(1, 4)),
        ),
    ).payload_with(np.stack(planes), None)
    spec = MaterializationSpec(
        image_options(
            filename_suffix="_saved.npy",
            filename_identity=MaterializedFilenameIdentity.SOURCE_IDENTITY,
        )
    )
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    primary_path = materialize(
        spec,
        data=payload,
        path="/analysis/aggregate.pkl",
        filemanager=filemanager,
        backends=("memory",),
    )

    assert primary_path == "/analysis/site1_saved.npy"
    assert spec.emits_variable_component_planes(payload)
    assert tuple(
        identity.path for identity in spec.emitted_source_identities(payload)
    ) == tuple(f"/input/site{index}.tif" for index in range(1, 4))
    for index, plane in enumerate(planes, start=1):
        np.testing.assert_array_equal(
            filemanager.load(f"/analysis/site{index}_saved.npy", "memory"),
            plane,
        )


def test_file_bundle_preserves_bytes_and_utf8_encodes_text() -> None:
    _image_options, bundle_options = _option_types()
    filemanager = FileManager({"memory": MemoryStorageBackend()})
    spec = MaterializationSpec(bundle_options())

    primary_path = materialize(
        spec,
        data={
            "tables/Image.csv": "ImageNumber,Count\n1,2\n",
            "analysis.sqlite": b"SQLite format 3\x00\x01",
        },
        path="/analysis/ExportBundle.pkl",
        filemanager=filemanager,
        backends=("memory",),
    )

    assert primary_path == "/analysis/tables/Image.csv"
    assert filemanager.load(primary_path, "memory") == (b"ImageNumber,Count\n1,2\n")
    assert filemanager.load("/analysis/analysis.sqlite", "memory") == (
        b"SQLite format 3\x00\x01"
    )


def test_file_bundle_outputs_retain_declared_text_semantics() -> None:
    _image_options, bundle_options = _option_types()
    outputs = materialization_outputs(
        MaterializationSpec(bundle_options()),
        data={
            "tables/Image.csv": "ImageNumber,Count\n1,2\n",
            "analysis.sqlite": b"SQLite format 3\x00\x01",
        },
        path="/analysis/ExportBundle.pkl",
        filemanager=FileManager({"memory": MemoryStorageBackend()}),
    )

    assert outputs[0].require_text_content() == "ImageNumber,Count\n1,2\n"
    with pytest.raises(TypeError, match="not declared as text"):
        outputs[1].require_text_content()


def test_file_bundle_persists_bytes_and_utf8_text_to_disk(tmp_path) -> None:
    _image_options, bundle_options = _option_types()
    filemanager = FileManager({"disk": DiskStorageBackend()})
    spec = MaterializationSpec(bundle_options())

    primary_path = materialize(
        spec,
        data={
            "tables/Image.csv": "ImageNumber,Count\n1,2\n",
            "analysis.sqlite": b"SQLite format 3\x00\x01",
        },
        path=str(tmp_path / "ExportBundle.pkl"),
        filemanager=filemanager,
        backends=("disk",),
    )

    assert primary_path == str(tmp_path / "tables" / "Image.csv")
    assert (tmp_path / "tables" / "Image.csv").read_text(encoding="utf-8") == (
        "ImageNumber,Count\n1,2\n"
    )
    assert (tmp_path / "analysis.sqlite").read_bytes() == (b"SQLite format 3\x00\x01")


def test_materialization_spec_error_write_mode_refuses_existing_path(tmp_path) -> None:
    _image_options, bundle_options = _option_types()
    output_path = tmp_path / "report.txt"
    output_path.write_bytes(b"existing")
    filemanager = FileManager({"disk": DiskStorageBackend()})

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        materialize(
            MaterializationSpec(
                bundle_options(),
                write_mode=WriteMode.ERROR,
            ),
            data={"report.txt": b"replacement"},
            path=str(tmp_path / "ExportBundle.pkl"),
            filemanager=filemanager,
            backends=("disk",),
        )

    assert output_path.read_bytes() == b"existing"


@pytest.mark.parametrize(
    "bundle",
    (
        {"/absolute.txt": b"bad"},
        {"../escape.txt": b"bad"},
        {"nested/../../escape.txt": b"bad"},
        {"same//path.txt": b"first", "same/path.txt": b"second"},
        {1: b"bad key"},
        {"unsupported.bin": 42},
    ),
)
def test_file_bundle_rejects_unsafe_or_unsupported_entries(bundle) -> None:
    _image_options, bundle_options = _option_types()
    filemanager = FileManager({"memory": MemoryStorageBackend()})

    with pytest.raises((TypeError, ValueError), match="path|relative|bundle|str|bytes"):
        materialize(
            MaterializationSpec(bundle_options()),
            data=bundle,
            path=str(Path("/analysis/ExportBundle.pkl")),
            filemanager=filemanager,
            backends=("memory",),
        )
