"""Dataset acquisition utilities."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from benchmark.contracts.dataset import (
    AcquiredDataset,
    ArchiveFormat,
    DatasetSpec,
    DatasetValidationRule,
)

IMAGE_EXTENSIONS = {".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class DatasetAcquisitionError(Exception):
    """Raised when dataset download, extraction, or validation fails."""


def _download_file(url: str, destination: Path) -> None:
    """Stream a URL to disk with progress display."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        try:
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network failure path
            raise DatasetAcquisitionError(f"Failed to download {url}: {exc}") from exc

        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
            leave=False,
        )
        with tmp_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file_obj.write(chunk)
                    progress.update(len(chunk))
        progress.close()

    tmp_path.rename(destination)


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip archive into target_dir."""
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise DatasetAcquisitionError(f"Corrupted zip archive: {zip_path}") from exc


def _count_images(root: Path) -> int:
    """Count image files under root recursively."""
    return sum(1 for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _validate_count(root: Path, expected: int) -> int:
    """Validate image count within ±5% tolerance."""
    if expected is None:
        raise DatasetAcquisitionError("expected_count must be provided for count validation")

    found = _count_images(root)
    lower = int(expected * 0.95)
    upper = int(expected * 1.05)
    if not (lower <= found <= upper):
        raise DatasetAcquisitionError(
            f"Validation failed: found {found} images, expected {expected} (tolerance ±5%)"
        )
    return found


def _validate_manifest(root: Path, manifest: Path) -> int:
    """Validate files listed in manifest exist under root."""
    if not manifest.exists():
        raise DatasetAcquisitionError(f"Manifest file missing: {manifest}")

    missing: list[str] = []
    count = 0
    for line in manifest.read_text().splitlines():
        relative = line.strip()
        if not relative:
            continue
        count += 1
        if not (root / relative).exists():
            missing.append(relative)
    if missing:
        raise DatasetAcquisitionError(f"{len(missing)} files listed in manifest are missing")
    return count


def _validate_non_empty(root: Path) -> int:
    """Validate that extraction produced files; return discovered image count."""
    if not any(root.rglob("*")):
        raise DatasetAcquisitionError(f"Extracted dataset is empty: {root}")
    return _count_images(root)


def _validate_dataset(spec: DatasetSpec, dataset_dir: Path) -> int:
    """Run validation rules and return image count."""
    if spec.validation_rule is DatasetValidationRule.IMAGE_COUNT:
        return _validate_count(dataset_dir, spec.expected_count)
    if spec.validation_rule is DatasetValidationRule.MANIFEST:
        if spec.manifest_path is None:
            raise DatasetAcquisitionError("manifest_path must be provided for manifest validation")
        return _validate_manifest(dataset_dir, spec.manifest_path)
    if spec.validation_rule is DatasetValidationRule.NON_EMPTY:
        return _validate_non_empty(dataset_dir)
    raise DatasetAcquisitionError(f"Unknown validation rule '{spec.validation_rule.name}'")


def acquire_dataset(spec: DatasetSpec) -> AcquiredDataset:
    """
    Acquire dataset (download, extract, validate, cache).

    Download to: ~/.cache/openhcs/benchmark_datasets/{spec.id}/

    Returns:
        AcquiredDataset with path, image_count, metadata

    Raises:
        DatasetAcquisitionError: If download/extraction/validation fails
    """
    cache_root = Path.home() / ".cache" / "openhcs" / "benchmark_datasets" / spec.id
    archive_dir = cache_root / "archives"
    extract_dir = cache_root / "data"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Fast path: existing extraction that still validates
    if extract_dir.exists():
        try:
            image_count = _validate_dataset(spec, extract_dir)
            return AcquiredDataset(
                id=spec.id,
                path=extract_dir,
                microscope_type=spec.microscope_type,
                image_count=image_count,
                metadata={"cached": True},
            )
        except DatasetAcquisitionError:
            # Re-download and extract
            shutil.rmtree(extract_dir, ignore_errors=True)

    # Download missing archives
    for url in spec.urls:
        archive_path = archive_dir / Path(url).name
        if not archive_path.exists():
            _download_file(url, archive_path)

    # Extract all archives into temporary dir then atomically move
    tmp_extract = cache_root / ".extract_tmp"
    if tmp_extract.exists():
        shutil.rmtree(tmp_extract)
    tmp_extract.mkdir(parents=True, exist_ok=True)

    for url in spec.urls:
        archive_path = archive_dir / Path(url).name
        if spec.archive_format is ArchiveFormat.ZIP:
            _extract_zip(archive_path, tmp_extract)
        else:
            raise DatasetAcquisitionError(f"Unsupported archive format: {spec.archive_format.name}")

    # Replace existing extraction atomically
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    tmp_extract.rename(extract_dir)

    image_count = _validate_dataset(spec, extract_dir)

    metadata = {
        "source_urls": spec.urls,
        "cached": False,
        "size_bytes": spec.size_bytes,
    }
    return AcquiredDataset(
        id=spec.id,
        path=extract_dir,
        microscope_type=spec.microscope_type,
        image_count=image_count,
        metadata=metadata,
    )
