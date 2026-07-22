"""Ownership and runtime handoff tests for Zarr configuration."""

from dataclasses import fields

from polystore.config import (
    ZarrChunkStrategy as PolyStoreZarrChunkStrategy,
    ZarrCompressor as PolyStoreZarrCompressor,
    ZarrConfig as PolyStoreZarrConfig,
)
from polystore.zarr import ZarrStorageBackend
from pycodify import Assignment, generate_python_source
from objectstate.lazy_factory import FIELD_ABBREVIATIONS_REGISTRY

from openhcs.core.config import (
    GlobalPipelineConfig,
    LazyZarrConfig,
    ZarrChunkStrategy,
    ZarrCompressor,
    ZarrConfig,
)


def test_openhcs_reexports_polystore_zarr_nominal_owners() -> None:
    assert ZarrChunkStrategy is PolyStoreZarrChunkStrategy
    assert ZarrCompressor is PolyStoreZarrCompressor
    assert issubclass(ZarrConfig, PolyStoreZarrConfig)
    assert ZarrConfig.__dict__.get("__annotations__", {}) == {}
    assert [field.name for field in fields(ZarrConfig)] == [
        "compressor",
        "compression_level",
        "chunk_strategy",
    ]


def test_openhcs_zarr_defaults_are_polystore_owned_identities() -> None:
    config = ZarrConfig()

    assert config.compressor is PolyStoreZarrCompressor.ZLIB
    assert config.compression_level == 3
    assert config.chunk_strategy is PolyStoreZarrChunkStrategy.WELL
    assert GlobalPipelineConfig().zarr_config == config


def test_openhcs_zarr_registration_preserves_lazy_and_presentation_metadata() -> None:
    lazy = LazyZarrConfig()

    assert object.__getattribute__(lazy, "compressor") is None
    assert object.__getattribute__(lazy, "compression_level") is None
    assert object.__getattribute__(lazy, "chunk_strategy") is None
    assert lazy.compressor is PolyStoreZarrCompressor.ZLIB
    assert lazy.compression_level == 3
    assert lazy.chunk_strategy is PolyStoreZarrChunkStrategy.WELL
    assert FIELD_ABBREVIATIONS_REGISTRY[ZarrConfig] == {
        "compressor": "compressor",
        "compression_level": "level",
        "chunk_strategy": "chunks",
    }


def test_openhcs_config_handoff_controls_polystore_chunking() -> None:
    well_config = ZarrConfig(chunk_strategy=ZarrChunkStrategy.WELL)
    file_config = ZarrConfig(chunk_strategy=ZarrChunkStrategy.FILE)

    assert ZarrStorageBackend(well_config)._calculate_chunks((2, 3, 4, 10, 20)) == (
        2,
        3,
        4,
        10,
        20,
    )
    assert ZarrStorageBackend(file_config)._calculate_chunks((2, 3, 4, 10, 20)) == (
        1,
        1,
        1,
        10,
        20,
    )


def test_openhcs_zarr_config_code_roundtrips_through_public_imports() -> None:
    config = ZarrConfig(
        compressor=ZarrCompressor.ZSTD,
        compression_level=7,
        chunk_strategy=ZarrChunkStrategy.FILE,
    )
    source = generate_python_source(Assignment("config", config))
    namespace: dict[str, object] = {}

    exec(source, namespace)

    assert namespace["config"] == config
    assert isinstance(namespace["config"], ZarrConfig)
