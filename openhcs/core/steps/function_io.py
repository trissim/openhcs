"""Image I/O helpers used by FunctionStep orchestration."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from openhcs.constants.constants import Backend, LOADABLE_IMAGE_EXTENSIONS

if TYPE_CHECKING:
    from openhcs.core.context.processing_context import ProcessingContext


logger = logging.getLogger(__name__)


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
    microscope_handler: Any,
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
    filemanager: Any,
    memory_data: Sequence[Any],
    materialized_paths: Sequence[str],
    materialized_backend: str,
    zarr_config: Mapping[str, Any] | None,
    context: ProcessingContext,
    axis_id: str,
) -> None:
    """Save data to a materialized backend with microscope/Zarr metadata."""
    save_kwargs: dict[str, Any] = {
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

    filemanager.save_batch(
        memory_data, list(materialized_paths), materialized_backend, **save_kwargs
    )


def get_all_image_paths(
    input_dir: str | Path,
    backend: str,
    axis_id: str,
    filemanager: Any,
    microscope_handler: Any,
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
    filemanager: Any,
    microscope_handler: Any,
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
    filemanager: Any,
    microscope_handler: Any,
    zarr_config: Mapping[str, Any] | None = None,
    patterns_to_preload: Sequence[str] | None = None,
    variable_components: Sequence[str] | None = None,
) -> None:
    """Preload this step's images from the source backend into the memory backend."""
    if patterns_to_preload is not None:
        all_files = [
            file_path
            for pattern in patterns_to_preload
            for file_path in microscope_handler.path_list_from_pattern(
                str(step_input_dir),
                pattern,
                filemanager,
                read_backend,
                variable_components,
            )
        ]
        full_file_paths = [
            str(step_input_dir / file_path)
            if not Path(file_path).is_absolute()
            else str(file_path)
            for file_path in set(all_files)
        ]
    else:
        get_paths_for_axis = create_image_path_getter(
            axis_id, filemanager, microscope_handler
        )
        full_file_paths = get_paths_for_axis(step_input_dir, read_backend)

    if not full_file_paths:
        raise RuntimeError(
            f"Bulk preload found no files for axis {axis_id} in {step_input_dir} "
            f"with backend {read_backend}."
        )

    if read_backend == Backend.ZARR.value:
        raw_images = filemanager.load_batch(
            full_file_paths, read_backend, zarr_config=zarr_config
        )
    else:
        raw_images = filemanager.load_batch(full_file_paths, read_backend)

    filemanager.ensure_directory(str(step_input_dir), Backend.MEMORY.value)
    for file_path in full_file_paths:
        if filemanager.exists(file_path, Backend.MEMORY.value):
            filemanager.delete(file_path, Backend.MEMORY.value)

    filemanager.save_batch(raw_images, full_file_paths, Backend.MEMORY.value)


def update_metadata_for_zarr_conversion(
    plate_root: Path,
    original_subdir: str,
    zarr_subdir: str | None,
    context: ProcessingContext,
) -> None:
    """Update OpenHCS metadata after a Zarr input conversion."""
    from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator
    from polystore.metadata_writer import AtomicMetadataWriter, get_metadata_path

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
