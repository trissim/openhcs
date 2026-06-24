"""Image I/O helpers used by FunctionStep orchestration."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, TypeAlias

from openhcs.constants.constants import Backend, LOADABLE_IMAGE_EXTENSIONS
from openhcs.core.image_file_serialization import prepare_disk_image_payloads
from openhcs.core.runtime_values import (
    ImagePayloadMetadataInput,
    ImagePayloadSourceMetadataContext,
    RuntimeImagePayloadContext,
)
from openhcs.core.source_image_provenance import SourceImageIdentity

if TYPE_CHECKING:
    from openhcs.core.config import ZarrConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.microscopes.microscope_base import MicroscopeHandler
    from polystore.filemanager import FileManager

logger = logging.getLogger(__name__)

BackendOptionValue: TypeAlias = "str | int | float | bool | ZarrConfig | None"
ZarrBackendConfig: TypeAlias = "Mapping[str, BackendOptionValue]"


@dataclass(frozen=True, slots=True)
class StepPreloadFileSet:
    """Execution-scoped source files that must be available in memory backend."""

    paths: tuple[str, ...]

    @classmethod
    def from_paths(cls, paths: Sequence[str | Path]) -> "StepPreloadFileSet":
        """Return a deterministic file set while preserving first-seen order."""
        return cls(tuple(dict.fromkeys(str(path) for path in paths)))

    def missing_memory_paths(
        self,
        filemanager: FileManager,
    ) -> tuple[str, ...]:
        """Return source paths not already copied into the execution memory backend."""
        return tuple(
            path
            for path in self.paths
            if not filemanager.exists(path, Backend.MEMORY.value)
        )

    def load_missing_payloads(
        self,
        *,
        filemanager: FileManager,
        read_backend: str,
        zarr_config: ZarrBackendConfig | None,
    ) -> tuple[tuple[str, ...], list[ImagePayloadMetadataInput]]:
        """Load missing source payloads and wrap them with source metadata."""
        missing_paths = self.missing_memory_paths(filemanager)
        if not missing_paths:
            return (), []
        if read_backend == Backend.ZARR.value:
            raw_images = filemanager.load_batch(
                list(missing_paths),
                read_backend,
                zarr_config=zarr_config,
            )
        else:
            raw_images = filemanager.load_batch(list(missing_paths), read_backend)
        return missing_paths, [
            RuntimeImagePayloadContext(
                image,
                mask=None,
                metadata=ImagePayloadSourceMetadataContext(
                    SourceImageIdentity(file_path),
                    read_backend=read_backend,
                    filemanager=filemanager,
                ).metadata_request(image).metadata(),
            ).payload()
            for image, file_path in zip(raw_images, missing_paths, strict=True)
        ]

def generate_materialized_paths(
    memory_paths: Sequence[str],
    step_output_dir: Path,
    materialized_output_dir: Path,
) -> list[str]:
    """Generate materialized paths by replacing the step output directory prefix."""
    return [
        str(materialized_output_dir / Path(memory_path).relative_to(step_output_dir))
        for memory_path in memory_paths
    ]

def calculate_zarr_dimensions(
    file_paths: Sequence[str | Path],
    microscope_handler: MicroscopeHandler,
) -> tuple[int, int, int]:
    """Calculate Zarr channel/z/site dimensions from parsed filenames."""
    parsed_files = [
        microscope_handler.parser.parse_filename(Path(file_path).name)
        for file_path in file_paths
    ]

    n_channels = len(
        {
            parsed.get("channel")
            for parsed in parsed_files
            if parsed and parsed.get("channel") is not None
        }
    )
    n_z = len(
        {
            parsed.get("z_index")
            for parsed in parsed_files
            if parsed and parsed.get("z_index") is not None
        }
    )
    n_fields = len(
        {
            parsed.get("site")
            for parsed in parsed_files
            if parsed and parsed.get("site") is not None
        }
    )

    return max(1, n_channels), max(1, n_z), max(1, n_fields)

def save_materialized_data(
    filemanager: FileManager,
    memory_data: Sequence[ImagePayloadMetadataInput],
    materialized_paths: Sequence[str],
    materialized_backend: str,
    zarr_config: ZarrBackendConfig | None,
    context: ProcessingContext,
    axis_id: str,
) -> None:
    """Save data to a materialized backend with microscope/Zarr metadata."""
    save_kwargs: dict[str, BackendOptionValue] = {
        "parser_name": context.microscope_handler.parser.__class__.__name__,
        "microscope_type": context.microscope_handler.microscope_type,
    }

    if materialized_backend == Backend.ZARR.value:
        n_channels, n_z, n_fields = calculate_zarr_dimensions(
            materialized_paths, context.microscope_handler
        )
        row, col = context.microscope_handler.parser.extract_component_coordinates(
            axis_id
        )
        save_kwargs.update(
            {
                "chunk_name": axis_id,
                "zarr_config": zarr_config,
                "n_channels": n_channels,
                "n_z": n_z,
                "n_fields": n_fields,
                "row": row,
                "col": col,
            }
        )

    payloads = (
        prepare_disk_image_payloads(memory_data, materialized_paths)
        if materialized_backend == Backend.DISK.value
        else memory_data
    )
    filemanager.save_batch(
        payloads, list(materialized_paths), materialized_backend, **save_kwargs
    )

def get_all_image_paths(
    input_dir: str | Path,
    backend: str,
    axis_id: str,
    filemanager: FileManager,
    microscope_handler: MicroscopeHandler,
) -> list[str]:
    """Get all image file paths for one multiprocessing axis value."""
    from openhcs.constants import MULTIPROCESSING_AXIS

    all_image_files = filemanager.list_image_files(
        str(input_dir),
        backend,
        extensions=LOADABLE_IMAGE_EXTENSIONS,
    )
    axis_key = MULTIPROCESSING_AXIS.value
    parser = microscope_handler.parser

    axis_files = []
    for file_path in all_image_files:
        filename = os.path.basename(str(file_path))
        metadata = parser.parse_filename(filename)
        if metadata and metadata.get(axis_key) == axis_id:
            axis_files.append(str(file_path))

    input_dir_path = Path(input_dir)
    full_file_paths = [
        str(input_dir_path / Path(file_path).name)
        for file_path in sorted(set(axis_files))
    ]

    logger.debug(
        "Found %s total files, %s for axis %s",
        len(all_image_files),
        len(full_file_paths),
        axis_id,
    )
    return full_file_paths

def create_image_path_getter(
    axis_id: str,
    filemanager: FileManager,
    microscope_handler: MicroscopeHandler,
) -> Callable[[str | Path, str], list[str]]:
    """Create a path getter bound to one multiprocessing axis value."""

    def get_paths_for_axis(input_dir: str | Path, backend: str) -> list[str]:
        return get_all_image_paths(
            input_dir=input_dir,
            axis_id=axis_id,
            backend=backend,
            filemanager=filemanager,
            microscope_handler=microscope_handler,
        )

    return get_paths_for_axis

def bulk_preload_step_images(
    step_input_dir: Path,
    axis_id: str,
    read_backend: str,
    filemanager: FileManager,
    microscope_handler: MicroscopeHandler,
    zarr_config: ZarrBackendConfig | None = None,
    patterns_to_preload: Sequence[str] | None = None,
    variable_components: Sequence[str] | None = None,
) -> None:
    """Preload this step's images from the source backend into the memory backend."""
    if patterns_to_preload is not None:
        all_files = (
            file_path
            for pattern in patterns_to_preload
            for file_path in microscope_handler.path_list_from_pattern(
                str(step_input_dir),
                pattern,
                filemanager,
                read_backend,
                variable_components,
            )
        )
        full_file_paths = (
            str(step_input_dir / file_path)
            if not Path(file_path).is_absolute()
            else str(file_path)
            for file_path in all_files
        )
    else:
        get_paths_for_axis = create_image_path_getter(
            axis_id, filemanager, microscope_handler
        )
        full_file_paths = get_paths_for_axis(step_input_dir, read_backend)

    preload_file_set = StepPreloadFileSet.from_paths(full_file_paths)

    if not preload_file_set.paths:
        raise RuntimeError(
            f"Bulk preload found no files for axis {axis_id} in {step_input_dir} "
            f"with backend {read_backend}."
        )

    filemanager.ensure_directory(str(step_input_dir), Backend.MEMORY.value)
    missing_paths, raw_images = preload_file_set.load_missing_payloads(
        filemanager=filemanager,
        read_backend=read_backend,
        zarr_config=zarr_config,
    )
    if not missing_paths:
        logger.debug(
            "Bulk preload reused %s memory-backed files for axis %s",
            len(preload_file_set.paths),
            axis_id,
        )
        return
    logger.debug(
        "Bulk preload loading %s/%s files for axis %s",
        len(missing_paths),
        len(preload_file_set.paths),
        axis_id,
    )
    filemanager.save_batch(list(raw_images), list(missing_paths), Backend.MEMORY.value)

def update_metadata_for_zarr_conversion(
    plate_root: Path,
    original_subdir: str,
    zarr_subdir: str | None,
    context: ProcessingContext,
) -> None:
    """Update OpenHCS metadata after a Zarr input conversion."""
    from openhcs.microscopes.openhcs import (
        AtomicMetadataWriter,
        OpenHCSMetadataGenerator,
        get_metadata_path,
    )

    metadata_path = get_metadata_path(plate_root)
    writer = AtomicMetadataWriter()

    if zarr_subdir:
        zarr_dir = plate_root / zarr_subdir
        metadata_generator = OpenHCSMetadataGenerator(context.filemanager)
        metadata_generator.create_metadata(
            context,
            str(zarr_dir),
            Backend.ZARR.value,
            is_main=True,
            plate_root=str(plate_root),
            sub_dir=zarr_subdir,
            skip_if_complete=True,
        )
        writer.merge_subdirectory_metadata(
            metadata_path, {original_subdir: {"main": False}}
        )
        logger.info(
            "Ensured complete metadata for %s, set %s main=false",
            zarr_subdir,
            original_subdir,
        )
        return

    writer.merge_subdirectory_metadata(
        metadata_path,
        {original_subdir: {"available_backends": {Backend.ZARR.value: True}}},
    )
    logger.info("Updated metadata: %s now has zarr backend", original_subdir)
