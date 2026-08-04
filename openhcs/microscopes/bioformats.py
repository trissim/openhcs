"""Microscope handler for stores that emit exact source-plane declarations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Type, Union

from polystore.bioformats_storage import BioFormatsStorageBackend
from polystore.filemanager import FileManager
from polystore.ome_zarr_storage import OmeZarrStorageBackend

from openhcs.constants.constants import AllComponents, Backend, Microscope
from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector
from openhcs.core.source_bindings import (
    SourceBindingsConfig,
    source_bindings_defaults_to_base,
)
from openhcs.core.source_projection import SourcePlaneDataset
from openhcs.core.virtual_workspace_metadata import (
    AtomicMetadataWriter,
    FIELDS,
    get_metadata_path,
)
from openhcs.microscopes.bioformats_adapter import (
    BioFormatsAdapterUnavailableError,
    SourcePlaneStoreAdapter,
)
from openhcs.microscopes.microscope_base import (
    BroadMicroscopeDetector,
    MicroscopeHandler,
    register_metadata_handler,
)
from openhcs.microscopes.microscope_interfaces import MetadataHandler
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


class BioFormatsFilenameParser(SourceSchemaFilenameParser):
    """Parser for canonical virtual paths backed by addressable stores."""

    def extract_component_coordinates(self, component_value: str) -> tuple[str, str]:
        """Project exact sample identity into an injective NGFF well coordinate."""

        return "S", "".join(
            f"{byte:03d}" for byte in str(component_value).encode("utf-8")
        )


class BioFormatsMetadataHandler(MetadataHandler):
    """Metadata view over exact store-emitted source planes."""

    def __init__(self, filemanager: FileManager | None = None):
        super().__init__()
        self.filemanager = filemanager

    def source_dataset(self, plate_path: Union[str, Path]) -> SourcePlaneDataset:
        return SourcePlaneStoreAdapter.discover_dataset(plate_path)

    def source_diagnostics(
        self,
        plate_path: Union[str, Path],
    ) -> tuple[Mapping[str, object], ...]:
        """Project decoder-owned dataset diagnostics for metadata consumers."""

        return tuple(
            dict(diagnostic.metadata_payload())
            for diagnostic in self.source_dataset(plate_path).diagnostics
        )

    def physical_source_paths(
        self,
        plate_path: Union[str, Path],
    ) -> tuple[Path, ...]:
        """Return store-declared physical containers, not projected plane names."""

        dataset = self.source_dataset(plate_path)
        return tuple(
            dict.fromkeys(
                dataset.root / candidate.relative_path
                for candidate in dataset.candidates
            )
        )

    def _component_values(
        self,
        plate_path: Union[str, Path],
        component: AllComponents,
    ) -> Optional[Dict[str, Optional[str]]]:
        values: dict[str, str | None] = {}
        for candidate in self.source_dataset(plate_path).candidates:
            address = candidate.declared_address
            if address is None:
                raise ValueError("Store candidate lacks an exact plane address.")
            coordinate = address.component_values()[component]
            label = candidate.component_labels.get(component.value)
            previous = values.get(coordinate)
            if previous is not None and label is not None and previous != label:
                raise ValueError(
                    f"Conflicting {component.value} label for {coordinate!r}."
                )
            values[coordinate] = label if label is not None else previous
        return dict(sorted(values.items())) or None

    def find_metadata_file(self, plate_path: Union[str, Path]) -> Path:
        try:
            self.source_dataset(plate_path)
        except BioFormatsAdapterUnavailableError as exc:
            raise FileNotFoundError(
                f"No addressable Bio-Formats source dataset found for {plate_path}."
            ) from exc
        return Path(plate_path)

    def get_grid_dimensions(self, plate_path: Union[str, Path]) -> tuple[int, int]:
        del plate_path
        return (1, 1)

    def get_pixel_size(self, plate_path: Union[str, Path]) -> float:
        return self.source_dataset(plate_path).pixel_size

    def get_channel_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        return self._component_values(plate_path, AllComponents.CHANNEL)

    def get_well_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        return self._component_values(plate_path, AllComponents.WELL)

    def get_site_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        return self._component_values(plate_path, AllComponents.SITE)

    def get_z_index_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        return self._component_values(plate_path, AllComponents.Z_INDEX)

    def get_timepoint_values(
        self,
        plate_path: Union[str, Path],
    ) -> Optional[Dict[str, Optional[str]]]:
        return self._component_values(plate_path, AllComponents.TIMEPOINT)

    def get_image_files(
        self,
        plate_path: Union[str, Path],
        all_subdirs: bool = False,
    ) -> list[str]:
        del all_subdirs
        parser = BioFormatsFilenameParser()
        return [
            parser.construct_filename(
                **candidate.declared_address.as_component_metadata(),
            )
            for candidate in self.source_dataset(plate_path).candidates
            if candidate.declared_address is not None
        ]


class BioFormatsHandler(BroadMicroscopeDetector, MicroscopeHandler):
    """Project addressable store planes into one generic virtual workspace."""

    _microscope_type = Microscope.BIOFORMATS.value
    _metadata_handler_class = BioFormatsMetadataHandler

    @classmethod
    def create(
        cls,
        *,
        filemanager: FileManager,
        pattern_format: str | None = None,
        source_bindings_config: SourceBindingsConfig | None = None,
    ) -> "BioFormatsHandler":
        return cls(
            filemanager,
            pattern_format=pattern_format,
            source_bindings_config=source_bindings_config,
        )

    @classmethod
    def projects_declared_source_bindings(cls) -> bool:
        return True

    def __init__(
        self,
        filemanager: FileManager,
        pattern_format: str | None = None,
        source_bindings_config: SourceBindingsConfig | None = None,
    ):
        self.parser = BioFormatsFilenameParser(filemanager, pattern_format)
        self.metadata_handler = BioFormatsMetadataHandler(filemanager)
        self.source_bindings = source_bindings_defaults_to_base(
            source_bindings_config or SourceBindingsConfig()
        )
        super().__init__(parser=self.parser, metadata_handler=self.metadata_handler)

    @property
    def root_dir(self) -> str:
        return FIELDS.DEFAULT_SUBDIRECTORY

    @property
    def microscope_type(self) -> str:
        return self._microscope_type

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        return self._metadata_handler_class

    @property
    def compatible_backends(self) -> List[Backend]:
        return [Backend.BIOFORMATS]

    def initialize_workspace(
        self,
        plate_path: Path,
        filemanager: FileManager,
    ) -> Path:
        plate_root = Path(plate_path)
        self._write_dataset(
            plate_root,
            self.metadata_handler.source_dataset(plate_root),
            filemanager,
        )
        self.register_workspace_backends(plate_root, filemanager)
        self.plate_folder = plate_root
        return plate_root

    def get_available_backends(
        self,
        plate_path: Union[str, Path],
    ) -> List[Backend]:
        del plate_path
        return [Backend.BIOFORMATS]

    def post_workspace(
        self,
        plate_path: Union[str, Path],
        filemanager: FileManager,
        skip_preparation: bool = False,
    ) -> Path:
        plate_root = Path(plate_path)
        if not skip_preparation:
            self._write_dataset(
                plate_root,
                self.metadata_handler.source_dataset(plate_root),
                filemanager,
            )
        self.register_workspace_backends(plate_root, filemanager)
        return plate_root

    def _write_dataset(
        self,
        plate_root: Path,
        dataset: SourcePlaneDataset,
        filemanager: FileManager,
    ) -> None:
        projection_set = SourceBindingWorkspaceProjector(
            source_bindings=self.source_bindings,
            parser=self.parser,
        ).projection_set_for_candidates(
            plate_root,
            dataset.candidates,
            filemanager=filemanager,
            diagnostics=dataset.diagnostics,
        )
        metadata = projection_set.metadata_dict(
            parser=self.parser,
            microscope_handler_name=self._microscope_type,
            source_filename_parser_name=type(self.parser).__name__,
            grid_dimensions=[1, 1],
            pixel_size=dataset.pixel_size,
            main=True,
        )
        AtomicMetadataWriter().merge_subdirectory_metadata(
            get_metadata_path(plate_root),
            {FIELDS.DEFAULT_SUBDIRECTORY: metadata},
        )

    def register_workspace_backends(
        self,
        plate_root: Path,
        filemanager: FileManager,
    ) -> None:
        self.register_source_backends(filemanager)
        self._register_virtual_workspace_backend(plate_root, filemanager)

    def register_source_backends(self, filemanager: FileManager) -> None:
        """Register the decoder backends owned by exact store source references."""

        filemanager.register_backend(
            Backend.BIOFORMATS.value,
            BioFormatsStorageBackend(),
        )
        filemanager.register_backend(
            Backend.OME_ZARR,
            OmeZarrStorageBackend(),
        )

register_metadata_handler(BioFormatsHandler, BioFormatsMetadataHandler)
