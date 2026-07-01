"""Dataset acquisition utilities."""

from __future__ import annotations

import subprocess
import shutil
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from metaclass_registry import AutoRegisterMeta
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
from benchmark.datasets.cache import default_benchmark_dataset_cache_root

IMAGE_EXTENSIONS = {".bmp", ".dib", ".flex", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


class DatasetAcquisitionError(Exception):
    """Raised when dataset download, extraction, or validation fails."""


@dataclass(frozen=True)
class DatasetAcquisitionContext:
    """Resolved cache coordinates for one dataset acquisition."""

    spec: DatasetSpec
    cache_root: Path
    archive_dir: Path
    data_dir: Path


@dataclass(frozen=True)
class DatasetValidationContext:
    """Validation inputs for an acquired dataset."""

    spec: DatasetSpec
    data_dir: Path


class DatasetFileDownloader:
    """Download policy shared by dataset source handlers."""

    def download(self, url: str, destination: Path, *, tls_verify: bool = True) -> None:
        """Stream a URL to disk with progress display."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = destination.with_suffix(destination.suffix + ".part")

        with requests.get(url, stream=True, timeout=60, verify=tls_verify) as response:
            try:
                response.raise_for_status()
            except Exception as exc:  # pragma: no cover - network failure path
                raise DatasetAcquisitionError(
                    f"Failed to download {url}: {exc}"
                ) from exc

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

    def destination_name(self, url: str) -> str:
        """Resolve the local filename for one plain-file URL."""
        name = Path(unquote(urlparse(url).path)).name
        if not name:
            raise DatasetAcquisitionError(f"URL has no file name: {url}")
        return name


DEFAULT_DATASET_FILE_DOWNLOADER = DatasetFileDownloader()


class DatasetArchiveMaterializer:
    """Archive extraction policy shared by acquisition handlers."""

    def materialize_nested_archives(self, root: Path) -> None:
        """Expose payload files that upstream repos store inside nested archives."""
        for archive_path in tuple(sorted(root.rglob("*.zip"))):
            if not archive_path.is_file():
                continue
            self.extract_missing_members(archive_path, archive_path.parent)

    def extract_missing_members(self, zip_path: Path, target_dir: Path) -> None:
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
                    with (
                        archive.open(member, "r") as source,
                        target_path.open("wb") as target,
                    ):
                        shutil.copyfileobj(source, target)
        except zipfile.BadZipFile as exc:
            raise DatasetAcquisitionError(f"Corrupted zip archive: {zip_path}") from exc


DEFAULT_DATASET_ARCHIVE_MATERIALIZER = DatasetArchiveMaterializer()


class DatasetValidationStrategy(ABC, metaclass=AutoRegisterMeta):
    """Registered validator for one DatasetValidationRule."""

    __registry_key__ = "validation_rule"
    validation_rule: DatasetValidationRule | None = None

    @classmethod
    def for_rule(cls, rule: DatasetValidationRule) -> "DatasetValidationStrategy":
        try:
            strategy_type = cls.__registry__[rule]
        except KeyError as exc:
            raise DatasetAcquisitionError(
                f"Unknown validation rule '{rule.name}'"
            ) from exc
        return strategy_type()

    @abstractmethod
    def validate(self, context: DatasetValidationContext) -> int:
        """Validate a data directory and return image count."""


class ImageCountValidationStrategy(DatasetValidationStrategy):
    """Validate by image-count tolerance."""

    validation_rule = DatasetValidationRule.IMAGE_COUNT

    def validate(self, context: DatasetValidationContext) -> int:
        return _validate_count(context.data_dir, context.spec.expected_count)


class ManifestValidationStrategy(DatasetValidationStrategy):
    """Validate by dataset manifest."""

    validation_rule = DatasetValidationRule.MANIFEST

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

    validation_rule = DatasetValidationRule.NON_EMPTY

    def validate(self, context: DatasetValidationContext) -> int:
        return _validate_non_empty(context.data_dir)


class DatasetSourceHandler(ABC, metaclass=AutoRegisterMeta):
    """Registered acquisition implementation for one source family."""

    __registry_key__ = "source_kind"
    source_kind: DatasetSourceKind | None = None
    downloader: DatasetFileDownloader = DEFAULT_DATASET_FILE_DOWNLOADER
    archive_materializer: DatasetArchiveMaterializer = (
        DEFAULT_DATASET_ARCHIVE_MATERIALIZER
    )

    @classmethod
    def for_source(cls, source: DatasetSourceSpec) -> "DatasetSourceHandler":
        try:
            handler_type = cls.__registry__[source.kind]
        except KeyError as exc:
            raise DatasetAcquisitionError(
                f"Unsupported dataset source: {source.kind.name}"
            ) from exc
        return handler_type()

    @abstractmethod
    def acquire(
        self, context: DatasetAcquisitionContext, source: DatasetSourceSpec
    ) -> bool:
        """Acquire into context.data_dir and return whether cached data was reused."""

    def download_archives(
        self,
        context: DatasetAcquisitionContext,
        source: DatasetSourceSpec,
    ) -> tuple[tuple[Path, ...], bool]:
        """Download all archive URLs for a source and report whether all were cached."""
        if not source.urls:
            raise DatasetAcquisitionError(
                f"Dataset {context.spec.id!r} has no archive URLs to acquire."
            )

        context.archive_dir.mkdir(parents=True, exist_ok=True)
        archive_paths = tuple(
            context.archive_dir / Path(url).name for url in source.urls
        )
        cached = all(path.exists() for path in archive_paths)
        for url, archive_path in zip(source.urls, archive_paths, strict=True):
            if not archive_path.exists():
                self.downloader.download(
                    url,
                    archive_path,
                    tls_verify=source.tls_verify,
                )
        return archive_paths, cached


class ArchiveUrlSourceHandler(DatasetSourceHandler):
    """Acquire one or more URL archives into the dataset cache."""

    source_kind = DatasetSourceKind.ARCHIVE_URLS

    def acquire(
        self, context: DatasetAcquisitionContext, source: DatasetSourceSpec
    ) -> bool:
        context.archive_dir.mkdir(parents=True, exist_ok=True)
        if context.data_dir.exists():
            return True

        archive_paths, _ = self.download_archives(context, source)

        tmp_extract = context.cache_root / ".extract_tmp"
        if tmp_extract.exists():
            shutil.rmtree(tmp_extract)
        tmp_extract.mkdir(parents=True, exist_ok=True)

        for archive_path in archive_paths:
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


class UrlFilesSourceHandler(DatasetSourceHandler):
    """Acquire one or more plain files into the dataset cache."""

    source_kind = DatasetSourceKind.URL_FILES

    def acquire(
        self, context: DatasetAcquisitionContext, source: DatasetSourceSpec
    ) -> bool:
        if not source.urls:
            raise DatasetAcquisitionError(
                f"Dataset {context.spec.id!r} has no file URLs to acquire."
            )
        if context.data_dir.exists():
            return True
        context.data_dir.mkdir(parents=True, exist_ok=True)
        for url in source.urls:
            self.downloader.download(
                url,
                context.data_dir / self.downloader.destination_name(url),
                tls_verify=source.tls_verify,
            )
        return False


class GitSparseWithArchiveUrlsSourceHandler(DatasetSourceHandler):
    """Acquire sparse repository files plus companion dataset archives."""

    source_kind = DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES

    def acquire(
        self, context: DatasetAcquisitionContext, source: DatasetSourceSpec
    ) -> bool:
        cached_repo = GitSparseSourceHandler().acquire(context, source)
        archive_paths, cached_archives = self.download_archives(context, source)

        for archive_path in archive_paths:
            if context.spec.archive_format is ArchiveFormat.ZIP:
                self.archive_materializer.extract_missing_members(
                    archive_path,
                    context.data_dir,
                )
            else:
                raise DatasetAcquisitionError(
                    f"Unsupported archive format: {context.spec.archive_format.name}"
                )

        _materialize_nested_archives(context.data_dir)
        return cached_repo and cached_archives


class GitSparseSourceHandler(DatasetSourceHandler):
    """Acquire selected paths from a git repository."""

    source_kind = DatasetSourceKind.GIT_SPARSE

    def acquire(
        self, context: DatasetAcquisitionContext, source: DatasetSourceSpec
    ) -> bool:
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
            self._run_git(
                ["sparse-checkout", "set", *source.sparse_paths], context.data_dir
            )
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
            raise DatasetAcquisitionError(
                f"git {' '.join(args)} failed: {detail}"
            ) from exc


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a zip archive into target_dir."""
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise DatasetAcquisitionError(f"Corrupted zip archive: {zip_path}") from exc


def _materialize_nested_archives(root: Path) -> None:
    DEFAULT_DATASET_ARCHIVE_MATERIALIZER.materialize_nested_archives(root)


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
        raise DatasetAcquisitionError(
            "expected_count must be provided for count validation"
        )

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
        raise DatasetAcquisitionError(
            f"{len(missing)} files listed in manifest are missing"
        )
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

    Download to: {cache_base or default benchmark dataset cache}/{spec.id}/

    Returns:
        AcquiredDataset with path, image_count, metadata

    Raises:
        DatasetAcquisitionError: If download/extraction/validation fails
    """
    base_dir = cache_base or default_benchmark_dataset_cache_root()
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

    # Composite sources must still visit the source handler, because one side of
    # the source can be present while the other still needs materialization.
    if (
        extract_dir.exists()
        and source.kind is not DatasetSourceKind.GIT_SPARSE_WITH_ARCHIVES
    ):
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
        "tls_verify": source.tls_verify,
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
