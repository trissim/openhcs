"""Dataset acquisition utilities."""

from __future__ import annotations

import subprocess
import shutil
import zipfile
from abc import ABC, abstractmethod
from pathlib import Path

from metaclass_registry import AutoRegisterMeta
from nominal_refactor_advisor.record_algebra import product_record
import requests
from tqdm import tqdm

from benchmark.contracts.dataset import (
    AcquiredDataset,
    ArchiveFormat,
    DatasetSpec,
    DatasetSourceKind,
    DatasetSourceSpec,
    DatasetValidationRule,
)

IMAGE_EXTENSIONS = {".bmp", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class DatasetAcquisitionError(Exception):
    """Raised when dataset download, extraction, or validation fails."""


DatasetAcquisitionContext = product_record(
    "DatasetAcquisitionContext",
    "spec: DatasetSpec; cache_root: Path; archive_dir: Path; data_dir: Path",
    doc="Resolved cache coordinates for one dataset acquisition.",
    module_name=__name__,
)
DatasetValidationContext = product_record(
    "DatasetValidationContext",
    "spec: DatasetSpec; data_dir: Path",
    doc="Validation inputs for an acquired dataset.",
    module_name=__name__,
)


class DatasetValidationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered validator for one DatasetValidationRule."""

    __registry_key__ = "validation_rule"
    validation_rule: str | None = None

    @classmethod
    def for_rule(cls, rule: DatasetValidationRule) -> "DatasetValidationStrategy":
        strategy_type = cls.__registry__.get(rule.value)
        if strategy_type is None:
            raise DatasetAcquisitionError(f"Unknown validation rule '{rule.name}'")
        return strategy_type()

    @abstractmethod
    def validate(self, context: DatasetValidationContext) -> int:
        """Validate a data directory and return image count."""


class ImageCountValidationStrategy(DatasetValidationStrategy):
    """Validate by image-count tolerance."""

    validation_rule = DatasetValidationRule.IMAGE_COUNT.value

    def validate(self, context: DatasetValidationContext) -> int:
        return _validate_count(context.data_dir, context.spec.expected_count)


class ManifestValidationStrategy(DatasetValidationStrategy):
    """Validate by dataset manifest."""

    validation_rule = DatasetValidationRule.MANIFEST.value

    def validate(self, context: DatasetValidationContext) -> int:
        if context.spec.manifest_path is None:
            raise DatasetAcquisitionError(
                "manifest_path must be provided for manifest validation"
            )
        manifest_path = context.spec.manifest_path
        if not manifest_path.is_absolute():
            manifest_path = context.data_dir / manifest_path
        return _validate_manifest(context.data_dir, manifest_path)


class NonEmptyValidationStrategy(DatasetValidationStrategy):
    """Validate by checking the acquired tree is non-empty."""

    validation_rule = DatasetValidationRule.NON_EMPTY.value

    def validate(self, context: DatasetValidationContext) -> int:
        return _validate_non_empty(context.data_dir)


class DatasetSourceHandler(ABC, metaclass=AutoRegisterMeta):
    """Registered acquisition implementation for one source family."""

    __registry_key__ = "source_kind"
    source_kind: str | None = None

    @classmethod
    def for_source(cls, source: DatasetSourceSpec) -> "DatasetSourceHandler":
        handler_type = cls.__registry__.get(source.kind.value)
        if handler_type is None:
            raise DatasetAcquisitionError(f"Unsupported dataset source: {source.kind.name}")
        return handler_type()

    @abstractmethod
    def acquire(self, context: DatasetAcquisitionContext, source: DatasetSourceSpec) -> bool:
        """Acquire into context.data_dir and return whether cached data was reused."""


class ArchiveUrlSourceHandler(DatasetSourceHandler):
    """Acquire one or more URL archives into the dataset cache."""

    source_kind = DatasetSourceKind.ARCHIVE_URLS.value

    def acquire(self, context: DatasetAcquisitionContext, source: DatasetSourceSpec) -> bool:
        context.archive_dir.mkdir(parents=True, exist_ok=True)
        if context.data_dir.exists():
            return True

        if not source.urls:
            raise DatasetAcquisitionError(
                f"Dataset {context.spec.id!r} has no archive URLs to acquire."
            )

        for url in source.urls:
            archive_path = context.archive_dir / Path(url).name
            if not archive_path.exists():
                _download_file(url, archive_path)

        tmp_extract = context.cache_root / ".extract_tmp"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)

        for url in source.urls:
            archive_path = context.archive_dir / Path(url).name
            if context.spec.archive_format is ArchiveFormat.ZIP:
                _extract_zip(archive_path, tmp_extract)
            else:
                raise DatasetAcquisitionError(
                    f"Unsupported archive format: {context.spec.archive_format.name}"
                )

        if context.data_dir.exists():
            shutil.rmtree(context.data_dir)
        tmp_extract.rename(context.data_dir)
        return False


class GitSparseSourceHandler(DatasetSourceHandler):
    """Acquire selected paths from a git repository."""

    source_kind = DatasetSourceKind.GIT_SPARSE.value

    def acquire(self, context: DatasetAcquisitionContext, source: DatasetSourceSpec) -> bool:
        if not source.git_url:
            raise DatasetAcquisitionError(
                f"Dataset {context.spec.id!r} uses git_sparse without git_url."
            )

        if (context.data_dir / ".git").exists():
            cached = True
            if source.git_ref != "HEAD":
                self._run_git(
                    ["fetch", "--depth", "1", "origin", source.git_ref],
                    context.data_dir,
                )
        else:
            cached = False
            if context.data_dir.exists():
                shutil.rmtree(context.data_dir)
            context.data_dir.parent.mkdir(parents=True, exist_ok=True)
            clone_command = [
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
            ]
            if source.git_ref != "HEAD":
                clone_command.extend(["--branch", source.git_ref])
            clone_command.extend([source.git_url, str(context.data_dir)])
            self._run_git(clone_command, None)

        if source.sparse_paths:
            self._run_git(["sparse-checkout", "set", *source.sparse_paths], context.data_dir)
        if source.git_ref == "HEAD":
            self._run_git(["pull", "--ff-only"], context.data_dir)
        else:
            self._run_git(["checkout", "FETCH_HEAD"], context.data_dir)
        _materialize_nested_archives(context.data_dir)
        return cached

    @staticmethod
    def _run_git(args: list[str], cwd: Path | None) -> None:
        try:
            subprocess.run(
                ["git", *args],
                cwd=cwd,
                check=True,
                text=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or exc.stdout.strip() or str(exc)
            raise DatasetAcquisitionError(f"git {' '.join(args)} failed: {detail}") from exc


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


def _materialize_nested_archives(root: Path) -> None:
    """Expose payload files that upstream repos store inside nested archives."""
    for archive_path in tuple(sorted(root.rglob("*.zip"))):
        if not archive_path.is_file():
            continue
        _extract_zip_missing_members(archive_path, archive_path.parent)


def _extract_zip_missing_members(zip_path: Path, target_dir: Path) -> None:
    """Extract missing zip members beside an archive without clobbering checkout files."""
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                target_path = _safe_archive_member_path(target_dir, member.filename)
                if target_path.exists():
                    continue
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target_path.open("wb") as target:
                    shutil.copyfileobj(source, target)
    except zipfile.BadZipFile as exc:
        raise DatasetAcquisitionError(f"Corrupted zip archive: {zip_path}") from exc


def _safe_archive_member_path(target_dir: Path, member_name: str) -> Path:
    """Resolve an archive member path and reject traversal outside target_dir."""
    target_root = target_dir.resolve()
    target_path = (target_dir / member_name).resolve()
    if target_path == target_root or target_root not in target_path.parents:
        raise DatasetAcquisitionError(
            f"Archive member {member_name!r} escapes target directory {target_dir}."
        )
    return target_path


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
    strategy = DatasetValidationStrategy.for_rule(spec.validation_rule)
    return strategy.validate(DatasetValidationContext(spec=spec, data_dir=dataset_dir))


def acquire_dataset(
    spec: DatasetSpec,
    *,
    cache_base: Path | None = None,
) -> AcquiredDataset:
    """
    Acquire dataset (download, extract, validate, cache).

    Download to: {cache_base or ~/.cache/openhcs/benchmark_datasets}/{spec.id}/

    Returns:
        AcquiredDataset with path, image_count, metadata

    Raises:
        DatasetAcquisitionError: If download/extraction/validation fails
    """
    base_dir = cache_base or Path.home() / ".cache" / "openhcs" / "benchmark_datasets"
    cache_root = base_dir / spec.id
    archive_dir = cache_root / "archives"
    extract_dir = cache_root / "data"
    context = DatasetAcquisitionContext(
        spec=spec,
        cache_root=cache_root,
        archive_dir=archive_dir,
        data_dir=extract_dir,
    )
    source = spec.acquisition_source()

    # Fast path: existing extraction that still validates
    if extract_dir.exists():
        try:
            _materialize_nested_archives(extract_dir)
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

    cached = DatasetSourceHandler.for_source(source).acquire(context, source)

    image_count = _validate_dataset(spec, extract_dir)

    metadata = {
        "source_kind": source.kind.value,
        "source_urls": tuple(source.urls),
        "git_url": source.git_url,
        "git_ref": source.git_ref,
        "cached": cached,
        "size_bytes": spec.size_bytes,
    }
    return AcquiredDataset(
        id=spec.id,
        path=extract_dir,
        microscope_type=spec.microscope_type,
        image_count=image_count,
        metadata=metadata,
    )
