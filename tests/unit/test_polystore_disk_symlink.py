from pathlib import Path

from polystore.constants import Backend
from polystore.disk import DiskStorageBackend
from polystore.filemanager import FileManager


def test_filemanager_delete_unlinks_directory_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    backend = Backend.DISK.value
    filemanager = FileManager({backend: DiskStorageBackend()})
    target_dir = tmp_path / "target"
    link_dir = tmp_path / "link"
    filemanager.ensure_directory(target_dir, backend)
    filemanager.save("payload", target_dir / "payload.txt", backend)
    filemanager.create_symlink(target_dir, link_dir, backend)

    filemanager.delete(link_dir, backend)

    assert not filemanager.is_symlink(link_dir, backend)
    assert filemanager.exists(target_dir / "payload.txt", backend)


def test_filemanager_delete_all_unlinks_directory_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    backend = Backend.DISK.value
    filemanager = FileManager({backend: DiskStorageBackend()})
    target_dir = tmp_path / "target"
    link_dir = tmp_path / "link"
    filemanager.ensure_directory(target_dir, backend)
    filemanager.save("payload", target_dir / "payload.txt", backend)
    filemanager.create_symlink(target_dir, link_dir, backend)

    filemanager.delete_all(link_dir, backend)

    assert not filemanager.is_symlink(link_dir, backend)
    assert filemanager.exists(target_dir / "payload.txt", backend)
