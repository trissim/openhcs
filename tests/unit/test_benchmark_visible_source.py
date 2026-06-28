from pathlib import Path

from polystore.filemanager import FileManager

from benchmark.datasets.visible_source import resolve_visible_source_path
from openhcs.constants.constants import Backend


def test_visible_source_alias_is_writable_polystore_workspace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from polystore.base import ensure_storage_registry, storage_registry

    backend = Backend.DISK.value
    ensure_storage_registry()
    filemanager = FileManager(storage_registry)
    hidden_source = tmp_path / ".cache" / "images"
    visible_root = tmp_path / "visible_sources"
    filemanager.ensure_directory(hidden_source, backend)
    filemanager.save("payload", hidden_source / "image.txt", backend)
    filemanager.save({"source": True}, hidden_source / "openhcs_metadata.json", backend)
    monkeypatch.setenv("OPENHCS_BENCHMARK_VISIBLE_SOURCE_ROOT", str(visible_root))

    alias = resolve_visible_source_path(hidden_source)

    assert alias != hidden_source
    assert not any(part.startswith(".") for part in alias.parts if part not in {alias.anchor, ""})
    assert filemanager.is_symlink(alias / "image.txt", backend)
    assert not filemanager.is_symlink(alias / "openhcs_metadata.json", backend)

    filemanager.save({"ok": True}, alias / "openhcs_metadata.json", backend)

    assert filemanager.exists(alias / "openhcs_metadata.json", backend)
    assert filemanager.load(hidden_source / "openhcs_metadata.json", backend) == {
        "source": True,
    }
    assert resolve_visible_source_path(hidden_source) == alias


def test_visible_source_alias_preserves_pipeline_parent_siblings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from polystore.base import ensure_storage_registry, storage_registry

    backend = Backend.DISK.value
    ensure_storage_registry()
    filemanager = FileManager(storage_registry)
    pipeline_root = tmp_path / ".cache" / "AdvancedSegmentation"
    image_root = pipeline_root / "BBBC022_20585_AE"
    visible_root = tmp_path / "visible_sources"
    filemanager.ensure_directory(image_root, backend)
    (pipeline_root / "BBBC022_Analysis_Final.cppipe").write_text(
        "pipeline",
        encoding="utf-8",
    )
    (pipeline_root / "20585_AE.csv").write_text("metadata", encoding="utf-8")
    (image_root / "IXMtest_A01_s1_w1.tif").write_text("image", encoding="utf-8")
    monkeypatch.setenv("OPENHCS_BENCHMARK_VISIBLE_SOURCE_ROOT", str(visible_root))

    alias = resolve_visible_source_path(image_root)

    assert alias.name == image_root.name
    assert alias.parent.name.startswith(f"{pipeline_root.name}_")
    assert filemanager.is_symlink(alias / "IXMtest_A01_s1_w1.tif", backend)
    assert filemanager.is_symlink(alias.parent / "20585_AE.csv", backend)
    assert filemanager.is_symlink(alias.parent / "BBBC022_Analysis_Final.cppipe", backend)
    assert resolve_visible_source_path(image_root) == alias
