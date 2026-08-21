import json
from pathlib import Path

import numpy as np
import zarr

from objectstate.lazy_factory import ensure_global_config_context
from openhcs.constants.constants import AllComponents, Backend, OrchestratorState
from openhcs.core.config import (
    GlobalPipelineConfig,
    LazySourceBindingsConfig,
    PipelineConfig,
)
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.core.orchestrator.orchestrator import PipelineOrchestrator
from openhcs.core.steps.function_io import (
    save_materialized_data,
    update_metadata_for_zarr_conversion,
)
from openhcs.core.source_bindings import (
    NamedSourceBinding,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceSelector,
)
from openhcs.microscopes.bioformats import BioFormatsHandler
from openhcs.microscopes.bioformats_adapter import SourcePlaneStoreAdapter
from openhcs.microscopes.microscope_base import create_microscope_handler
from openhcs.microscopes.openhcs import OpenHCSMicroscopeHandler
from polystore.base import ensure_storage_registry, storage_registry
from polystore.bioformats_java import BioFormatsJavaContext
from polystore.filemanager import FileManager
from polystore.zarr import ZarrStorageBackend
from polystore.zarr_batch import (
    ZarrBatchAxis,
    ZarrBatchAxisRole,
    ZarrBatchLayout,
)


class _NoJavaStores:
    def declares_path(self, source_path: Path) -> bool:
        del source_path
        return False


def _filemanager() -> FileManager:
    ensure_storage_registry()
    return FileManager(dict(storage_registry))


def _binding(alias: str, file_name: str) -> NamedSourceBinding:
    return NamedSourceBinding(
        alias=alias,
        selector=SourceSelector(
            filters=(
                SourceFilterClause(
                    SourceFilterSubject.FILE,
                    SourceFilterMatchType.EQUALS,
                    file_name,
                ),
            ),
        ),
    )


def _write_ngff_plate(path: Path, pixels: np.ndarray) -> None:
    root = zarr.open_group(str(path), mode="w")
    root.attrs["plate"] = {
        "columns": [{"name": "01"}],
        "name": "Plate:mixed",
        "rows": [{"name": "A"}],
        "version": "0.4",
        "wells": [{"columnIndex": 0, "path": "A/01", "rowIndex": 0}],
    }
    well = root.require_group("A/01")
    well.attrs["well"] = {"images": [{"path": "0"}], "version": "0.4"}
    image = well.require_group("0")
    image.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "field", "type": "field"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {
                    "coordinateTransformations": [
                        {"scale": [1.0] * 5, "type": "scale"}
                    ],
                    "path": "0",
                }
            ],
            "name": "Image:ngff",
            "version": "0.4",
        }
    ]
    image.attrs["omero"] = {"channels": [{"label": "NGFF"}]}
    image.create_dataset("0", data=pixels[None, None, None])


def _write_mixed_stores(
    root: Path,
) -> dict[str, tuple[Path, np.ndarray]]:
    stores = {
        "NGFF": (root / "plate.zarr", np.full((3, 4), 7, dtype=np.uint16)),
        "TIFF": (root / "plain.tif", np.full((3, 4), 11, dtype=np.uint16)),
        "PNG": (root / "mask.png", np.full((3, 4), 13, dtype=np.uint16)),
    }
    _write_ngff_plate(*stores["NGFF"])
    for alias in ("TIFF", "PNG"):
        path, pixels = stores[alias]
        ImageFileFormat.require_path(path).write(path, pixels)
    return stores


def test_polystore_zarr_semantic_coordinates_round_trip_through_store_discovery(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _NoJavaStores()),
    )
    store_root = tmp_path / "images"
    output_paths = [
        store_root / "A01_s003_w2_z001_t002.tif",
        store_root / "A01_s003_w1_z001_t001.tif",
        store_root / "A01_s003_w2_z001_t001.tif",
        store_root / "A01_s003_w1_z001_t002.tif",
    ]
    layout = ZarrBatchLayout(
        axes=(
            ZarrBatchAxis("t", "time", ("2", "1")),
            ZarrBatchAxis(
                "field",
                "field",
                ("3",),
                ZarrBatchAxisRole.HCS_IMAGE,
            ),
            ZarrBatchAxis("c", "channel", ("2", "1")),
            ZarrBatchAxis("z", "space", ("1",)),
        ),
        item_coordinates=(
            (0, 0, 0, 0),
            (1, 0, 1, 0),
            (1, 0, 0, 0),
            (0, 0, 1, 0),
        ),
    )
    pixels = [np.full((3, 4), index, dtype=np.uint16) for index in range(4)]
    ZarrStorageBackend().save_batch(
        pixels,
        output_paths,
        chunk_name="A01",
        batch_layout=layout,
        row="A",
        col="01",
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(store_root)

    assert {
        (
            candidate.declared_address.site,
            candidate.declared_address.channel,
            candidate.declared_address.z_index,
            candidate.declared_address.timepoint,
        )
        for candidate in dataset.candidates
    } == {
        ("3", "1", "1", "1"),
        ("3", "1", "1", "2"),
        ("3", "2", "1", "1"),
        ("3", "2", "1", "2"),
    }


def test_mixed_plane_stores_bind_and_load_through_virtual_workspace(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stores = _write_mixed_stores(tmp_path)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _NoJavaStores()),
    )

    dataset = SourcePlaneStoreAdapter.discover_dataset(tmp_path)

    assert dataset.identity.value == "Plate:mixed"
    assert {candidate.source_ref.backend for candidate in dataset.candidates} == {
        Backend.DISK.value,
        Backend.OME_ZARR.value,
    }
    assert {candidate.declared_address.well for candidate in dataset.candidates} == {
        "A01",
        "mask.png",
        "plain.tif",
    }

    source_bindings = SourceBindingsConfig(
        bindings=tuple(
            _binding(alias, path.name) for alias, (path, _pixels) in stores.items()
        )
    )
    filemanager = _filemanager()
    handler = create_microscope_handler(
        "auto",
        plate_folder=tmp_path,
        filemanager=filemanager,
        source_bindings_config=source_bindings,
    )
    assert isinstance(handler, BioFormatsHandler)
    assert handler.parser.extract_component_coordinates("plain.tif") == (
        "S",
        "112108097105110046116105102",
    )
    handler.initialize_workspace(tmp_path, filemanager)
    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )["subdirectories"]["."]
    paths_by_alias = {
        source_metadata["source_alias"]: virtual_path
        for virtual_path, source_metadata in metadata["source_metadata"].items()
    }

    assert set(paths_by_alias) == {"NGFF", "TIFF", "PNG"}
    for alias, (_path, pixels) in stores.items():
        loaded = filemanager.load(
            tmp_path / paths_by_alias[alias],
            backend=Backend.VIRTUAL_WORKSPACE.value,
        )
        np.testing.assert_array_equal(loaded, pixels)


def test_saved_source_bindings_rebuild_canonical_store_projection(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stores = _write_mixed_stores(tmp_path)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _NoJavaStores()),
    )
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    initial_bindings = SourceBindingsConfig(
        bindings=tuple(
            _binding(alias, path.name) for alias, (path, _pixels) in stores.items()
        )
    )
    orchestrator = PipelineOrchestrator(
        plate_path=tmp_path,
        pipeline_config=PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig.from_config(
                initial_bindings
            ),
        ),
    ).initialize()
    initial_handler = orchestrator.microscope_handler
    assert isinstance(initial_handler, BioFormatsHandler)
    initial_projection = orchestrator.source_workspace_projection()
    assert {
        projection.source_alias
        for path in initial_projection.relative_virtual_paths()
        if (
            projection := initial_projection.source_projections_by_virtual_path[path]
        )
    } == set(stores)

    edited_aliases = {"NGFF": "RawNGFF", "TIFF": "RawTIFF", "PNG": "Mask"}
    edited_bindings = SourceBindingsConfig(
        bindings=tuple(
            _binding(edited_aliases[alias], path.name)
            for alias, (path, _pixels) in stores.items()
        )
    )
    orchestrator.apply_pipeline_config(
        PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig.from_config(
                edited_bindings
            )
        )
    )

    assert orchestrator.state is OrchestratorState.CREATED
    assert not orchestrator.is_initialized()
    assert orchestrator.microscope_handler is None
    assert (
        orchestrator.get_effective_config().source_bindings_config
        == edited_bindings
    )

    orchestrator.initialize()

    assert isinstance(orchestrator.microscope_handler, BioFormatsHandler)
    assert orchestrator.microscope_handler is not initial_handler
    projection = orchestrator.source_workspace_projection()
    records = tuple(
        projection.source_projections_by_virtual_path[path]
        for path in projection.relative_virtual_paths()
    )
    assert {record.source_alias for record in records} == set(edited_aliases.values())
    assert {record.address.well for record in records} == {
        "A01",
        "mask.png",
        "plain.tif",
    }
    assert {
        (
            record.address.site,
            record.address.channel,
            record.address.z_index,
            record.address.timepoint,
        )
        for record in records
    } == {("1", "1", "1", "1")}
    assert {record.ref.backend for record in records} == {
        Backend.DISK.value,
        Backend.OME_ZARR.value,
    }
    assert len({record.ref.backend_address for record in records}) == len(records)
    expected_components = {
        AllComponents.WELL: {"A01", "mask.png", "plain.tif"},
        AllComponents.SITE: {"1"},
        AllComponents.CHANNEL: {"1"},
        AllComponents.Z_INDEX: {"1"},
        AllComponents.TIMEPOINT: {"1"},
    }
    assert {
        component: set(orchestrator.get_component_keys(component))
        for component in AllComponents
    } == expected_components


def test_mixed_plane_stores_materialize_and_reopen_with_source_identity(
    monkeypatch,
    tmp_path: Path,
) -> None:
    stores = _write_mixed_stores(tmp_path)
    monkeypatch.setattr(
        BioFormatsJavaContext,
        "instance",
        classmethod(lambda cls: _NoJavaStores()),
    )
    ensure_global_config_context(GlobalPipelineConfig, GlobalPipelineConfig())
    source_bindings = SourceBindingsConfig(
        bindings=tuple(
            _binding(alias, path.name) for alias, (path, _pixels) in stores.items()
        )
    )
    orchestrator = PipelineOrchestrator(
        plate_path=tmp_path,
        pipeline_config=PipelineConfig(
            source_bindings_config=LazySourceBindingsConfig.from_config(
                source_bindings
            )
        ),
    ).initialize()
    context = orchestrator.create_context(axis_id="A01")
    projection = orchestrator.source_workspace_projection()
    source_records = {
        projection.source_projections_by_virtual_path[path].source_alias: (path, projection.source_projections_by_virtual_path[path])
        for path in projection.relative_virtual_paths()
    }

    for alias, (virtual_path, record) in source_records.items():
        payload = orchestrator.filemanager.load(
            tmp_path / virtual_path,
            Backend.VIRTUAL_WORKSPACE.value,
        )
        save_materialized_data(
            orchestrator.filemanager,
            [payload],
            [str(tmp_path / "zarr" / virtual_path)],
            Backend.ZARR.value,
            orchestrator.get_effective_config().zarr_config,
            context,
            record.address.well,
        )
        np.testing.assert_array_equal(payload, stores[alias][1])

    update_metadata_for_zarr_conversion(tmp_path, ".", "zarr", context)

    metadata = json.loads(
        (tmp_path / "openhcs_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["subdirectories"]["."]["main"] is False
    zarr_metadata = metadata["subdirectories"]["zarr"]
    assert zarr_metadata["main"] is True
    assert zarr_metadata["available_backends"] == {Backend.ZARR.value: True}
    assert {
        source_metadata["source_alias"]
        for source_metadata in zarr_metadata["source_metadata"].values()
    } == set(stores)
    assert {
        payload["backend"]
        for payload in zarr_metadata["workspace_mapping"].values()
    } == {Backend.ZARR.value}

    reopened = PipelineOrchestrator(plate_path=tmp_path).initialize()
    assert isinstance(reopened.microscope_handler, OpenHCSMicroscopeHandler)
    assert reopened.input_dir == tmp_path / "zarr"
    assert (
        reopened.microscope_handler.get_primary_backend(
            reopened.input_dir,
            reopened.filemanager,
        )
        == Backend.ZARR.value
    )
    reopened_projection = reopened.source_workspace_projection()
    reopened_records = {
        reopened_projection.source_projections_by_virtual_path[path].source_alias: (
            path,
            reopened_projection.source_projections_by_virtual_path[path],
        )
        for path in reopened_projection.relative_virtual_paths()
    }
    assert set(reopened_records) == set(stores)
    for alias, (virtual_path, record) in reopened_records.items():
        assert record.ref.backend == Backend.ZARR.value
        assert record.ref.backend_address == virtual_path
        np.testing.assert_array_equal(
            reopened.filemanager.load(tmp_path / virtual_path, Backend.ZARR.value),
            stores[alias][1],
        )
