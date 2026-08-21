"""Image I/O helpers used by FunctionStep orchestration."""

from __future__ import annotations

from abc import ABC
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable, ClassVar, Mapping, Sequence, TypeAlias

from metaclass_registry import AutoRegisterMeta
from polystore.zarr_batch import ZarrBatchAxis, ZarrBatchAxisRole, ZarrBatchLayout

from openhcs.constants.constants import AllComponents, Backend, LOADABLE_IMAGE_EXTENSIONS
from openhcs.core.image_file_serialization import prepare_disk_image_payloads
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import image_payload_data, image_payload_mask, image_payload_metadata
from openhcs.core.source_image_provenance import SourceImageIdentity
from openhcs.core.steps.function_output_identity import FunctionOutputIdentity

if TYPE_CHECKING:
    from openhcs.core.config import ZarrConfig
    from openhcs.core.context.processing_context import ProcessingContext
    from openhcs.microscopes.microscope_base import MicroscopeHandler
    from polystore.filemanager import FileManager

logger = logging.getLogger(__name__)

BackendOptionValue: TypeAlias = (
    "str | int | float | bool | ZarrConfig | ZarrBatchLayout | None"
)
ZarrBackendConfig: TypeAlias = "Mapping[str, BackendOptionValue]"


def prepare_storage_image_payloads(
    payloads: Sequence[RuntimeArrayData],
    paths: Sequence[str | Path],
    backend: str,
) -> list[RuntimeArrayData]:
    """Project OpenHCS image values onto one storage-backend payload boundary."""

    if len(payloads) != len(paths):
        raise ValueError(
            "Storage image payload/path length mismatch: "
            f"{len(payloads)} payloads for {len(paths)} paths."
        )
    if backend == Backend.DISK.value:
        return prepare_disk_image_payloads(payloads, paths)
    return [image_payload_data(payload) for payload in payloads]


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
    ) -> tuple[tuple[str, ...], list[RuntimeArrayData]]:
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
            _preloaded_image_payload(
                image,
                source_path=file_path,
                read_backend=read_backend,
                filemanager=filemanager,
            )
            for image, file_path in zip(raw_images, missing_paths, strict=True)
        ]


def _preloaded_image_payload(
    image: RuntimeArrayData,
    *,
    source_path: str,
    read_backend: str,
    filemanager: FileManager,
) -> RuntimeArrayData:
    """Preserve loader-owned metadata or derive it through the source backend."""
    metadata = image_payload_metadata(image)
    if not metadata.has_values:
        metadata = ImagePayloadSourceMetadataContext(
            SourceImageIdentity(source_path),
            read_backend=read_backend,
            filemanager=filemanager,
        ).metadata(image)
    return metadata.payload_with(
        image_payload_data(image),
        image_payload_mask(image),
    )

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

@dataclass(frozen=True, slots=True)
class ZarrBatchItemIdentity:
    """Application-owned semantic identity for one stored image plane."""

    component_values: Mapping[str, str | int]
    filename_qualifier: str | None = None

    @classmethod
    def from_output(
        cls,
        output_identity: FunctionOutputIdentity,
    ) -> "ZarrBatchItemIdentity":
        return cls(
            component_values=output_identity.component_values,
            filename_qualifier=output_identity.filename_qualifier,
        )


class ZarrComponentAxisProjection(
    EnumKeyedStrategyMixin[AllComponents],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Project one OpenHCS component into its declaration-owned NGFF axis."""

    strategy_key: ClassVar[AllComponents | None] = None
    axis_order: ClassVar[int]
    axis_name: ClassVar[str]
    axis_type: ClassVar[str]
    axis_role: ClassVar[ZarrBatchAxisRole] = ZarrBatchAxisRole.ARRAY

    @classmethod
    def ordered_types(cls) -> tuple[type["ZarrComponentAxisProjection"], ...]:
        """Return declared storage axes in NGFF-valid order."""

        return tuple(
            sorted(cls.registered_strategy_types(), key=lambda item: item.axis_order)
        )

    @classmethod
    def batch_layout(
        cls,
        item_identities: Sequence[ZarrBatchItemIdentity],
    ) -> ZarrBatchLayout:
        """Project declared output identities into exact dense coordinates."""

        axis_types = cls.ordered_types()
        item_values = tuple(
            tuple(axis_type.item_value(identity) for axis_type in axis_types)
            for identity in item_identities
        )
        axes = tuple(
            ZarrBatchAxis(
                name=axis_type.axis_name,
                axis_type=axis_type.axis_type,
                values=tuple(
                    dict.fromkeys(values[axis_index] for values in item_values)
                ),
                role=axis_type.axis_role,
            )
            for axis_index, axis_type in enumerate(axis_types)
        )
        value_coordinates = tuple(
            {value: index for index, value in enumerate(axis.values)}
            for axis in axes
        )
        return ZarrBatchLayout(
            axes=axes,
            item_coordinates=tuple(
                tuple(
                    value_coordinates[axis_index][value]
                    for axis_index, value in enumerate(values)
                )
                for values in item_values
            ),
        )

    @classmethod
    def item_value(cls, identity: ZarrBatchItemIdentity) -> str:
        component = cls.strategy_key
        if component is None:
            raise RuntimeError("Zarr axis projection is missing its component owner")
        value = identity.component_values.get(component.value)
        if value is None:
            raise ValueError(
                f"Parsed output identity is missing component {component.value!r}"
            )
        return str(value)


class TimepointZarrAxisProjection(ZarrComponentAxisProjection):
    strategy_key = AllComponents.TIMEPOINT
    axis_order = 0
    axis_name = "t"
    axis_type = "time"


class SiteZarrAxisProjection(ZarrComponentAxisProjection):
    strategy_key = AllComponents.SITE
    axis_order = 1
    axis_name = "field"
    axis_type = "field"
    axis_role = ZarrBatchAxisRole.HCS_IMAGE


class ChannelZarrAxisProjection(ZarrComponentAxisProjection):
    strategy_key = AllComponents.CHANNEL
    axis_order = 2
    axis_name = "c"
    axis_type = "channel"

    @classmethod
    def item_value(cls, identity: ZarrBatchItemIdentity) -> str:
        channel = super().item_value(identity)
        qualifier = identity.filename_qualifier
        return channel if qualifier is None else f"{channel}:{qualifier}"


class ZIndexZarrAxisProjection(ZarrComponentAxisProjection):
    strategy_key = AllComponents.Z_INDEX
    axis_order = 3
    axis_name = "z"
    axis_type = "space"


def zarr_batch_layout(
    file_paths: Sequence[str | Path],
    microscope_handler: MicroscopeHandler,
) -> ZarrBatchLayout:
    """Return the declaration-driven Zarr layout for output image planes."""

    identities: list[ZarrBatchItemIdentity] = []
    unparsed: list[str] = []
    for file_path in file_paths:
        parsed = microscope_handler.parser.parse_filename(Path(file_path).name)
        if parsed is None:
            unparsed.append(str(file_path))
            continue
        identities.append(ZarrBatchItemIdentity(component_values=parsed))
    if unparsed:
        raise ValueError(
            "Cannot derive Zarr batch coordinates from paths "
            f"{tuple(unparsed)!r}"
        )
    return ZarrComponentAxisProjection.batch_layout(identities)


def zarr_output_batch_layout(
    output_identities: Sequence[FunctionOutputIdentity],
) -> ZarrBatchLayout:
    """Return a Zarr layout from full declared output identities."""

    return ZarrComponentAxisProjection.batch_layout(
        tuple(ZarrBatchItemIdentity.from_output(item) for item in output_identities)
    )

def save_materialized_data(
    filemanager: FileManager,
    memory_data: Sequence[RuntimeArrayData],
    materialized_paths: Sequence[str],
    materialized_backend: str,
    zarr_config: ZarrBackendConfig | None,
    context: ProcessingContext,
    axis_id: str,
    *,
    output_identities: Sequence[FunctionOutputIdentity] = (),
) -> None:
    """Save data to a materialized backend with microscope/Zarr metadata."""
    save_kwargs: dict[str, BackendOptionValue] = {
        "parser_name": context.microscope_handler.parser.__class__.__name__,
        "microscope_type": context.microscope_handler.microscope_type,
    }

    if materialized_backend == Backend.ZARR.value:
        row, col = context.microscope_handler.parser.extract_component_coordinates(
            axis_id
        )
        save_kwargs.update(
            {
                "chunk_name": axis_id,
                "zarr_config": zarr_config,
                "batch_layout": (
                    zarr_output_batch_layout(output_identities)
                    if output_identities
                    else zarr_batch_layout(
                        materialized_paths,
                        context.microscope_handler,
                    )
                ),
                "row": row,
                "col": col,
            }
        )

    payloads = prepare_storage_image_payloads(
        memory_data,
        materialized_paths,
        materialized_backend,
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
        recursive=True,
    )
    axis_key = MULTIPROCESSING_AXIS.value
    parser = microscope_handler.parser

    axis_files = []
    for file_path in all_image_files:
        filename = os.path.basename(str(file_path))
        metadata = parser.parse_filename(filename)
        if metadata and metadata.component_matches(axis_key, axis_id):
            axis_files.append(str(file_path))

    full_file_paths = sorted(
        {
            str(
                filemanager.resolve_listed_address(
                    path,
                    backend,
                    directory=input_dir,
                )
            )
            for path in axis_files
        }
    )

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
    for parent in dict.fromkeys(
        str(Path(path).parent) for path in preload_file_set.paths
    ):
        filemanager.ensure_directory(parent, Backend.MEMORY.value)
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
    from openhcs.core.virtual_workspace_metadata import (
        AtomicMetadataWriter,
        OpenHCSMetadataSubdirectories,
        VirtualWorkspaceSourceProjectionEntries,
        get_metadata_path,
    )
    from openhcs.microscopes.openhcs import (
        OpenHCSMetadataGenerator,
        OpenHCSMetadataHandler,
    )

    metadata_path = get_metadata_path(plate_root)
    writer = AtomicMetadataWriter()

    if zarr_subdir:
        zarr_dir = plate_root / zarr_subdir
        metadata_handler = OpenHCSMetadataHandler(context.filemanager)
        metadata_document = metadata_handler.load_metadata_document(plate_root)
        grid_dimensions = metadata_handler.get_grid_dimensions(plate_root)
        pixel_size = metadata_handler.get_pixel_size(plate_root)
        subdirectories = dict(
            OpenHCSMetadataSubdirectories(metadata_document).items()
        )
        if original_subdir not in subdirectories:
            raise ValueError(
                "Zarr conversion metadata is missing original subdirectory "
                f"{original_subdir!r}."
            )
        source_projections = VirtualWorkspaceSourceProjectionEntries.from_subdirectory(
            subdirectories[original_subdir]
        )
        if source_projections.entries:
            from openhcs.core.source_projection import (
                SourceProjectionMetadataSerializer,
                SourceProjectionSet,
            )
            from polystore.virtual_workspace import SourcePixelRef

            materialized_projections = []
            for output_path in context.filemanager.list_image_files(
                str(zarr_dir), Backend.ZARR.value
            ):
                try:
                    source_virtual_path = Path(output_path).relative_to(zarr_dir)
                except ValueError as exc:
                    raise ValueError(
                        "Zarr conversion output lies outside its declared store: "
                        f"{output_path!r}."
                    ) from exc
                source_virtual_text = source_virtual_path.as_posix()
                if source_virtual_text not in source_projections.entries:
                    raise ValueError(
                        "Zarr conversion output has no declared source projection: "
                        f"{source_virtual_text!r}."
                    )
                materialized_path = str(
                    PurePosixPath(zarr_subdir) / source_virtual_path
                )
                materialized_projections.append(
                    replace(
                        source_projections.entries[source_virtual_text],
                        ref=SourcePixelRef(
                            backend=Backend.ZARR.value,
                            backend_address=materialized_path,
                        ),
                    )
                )
            if not materialized_projections:
                raise ValueError(
                    f"Zarr conversion produced no image planes in {zarr_dir}."
                )
            zarr_metadata = SourceProjectionMetadataSerializer(
                parser=context.microscope_handler.parser,
                path_prefix=zarr_subdir,
            ).metadata_dict(
                SourceProjectionSet(tuple(materialized_projections)),
                microscope_handler_name=context.microscope_handler.microscope_type,
                source_filename_parser_name=type(
                    context.microscope_handler.parser
                ).__name__,
                grid_dimensions=list(grid_dimensions),
                pixel_size=pixel_size,
                available_backends={Backend.ZARR.value: True},
                main=True,
            )
            writer.merge_subdirectory_metadata(
                metadata_path,
                {
                    zarr_subdir: zarr_metadata,
                    original_subdir: {"main": False},
                },
            )
        else:
            OpenHCSMetadataGenerator(context.filemanager).create_metadata(
                context,
                str(zarr_dir),
                Backend.ZARR.value,
                is_main=True,
                plate_root=str(plate_root),
                sub_dir=zarr_subdir,
                grid_dimensions=grid_dimensions,
                pixel_size=pixel_size,
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
