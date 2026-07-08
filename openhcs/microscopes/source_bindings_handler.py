"""Microscope handler for source-binding projected workspaces."""

from __future__ import annotations

from pathlib import Path
from typing import Type, Union

from polystore.filemanager import FileManager

from openhcs.constants.constants import Backend, Microscope
from openhcs.core.source_bindings import (
    SourceBindingsConfig,
    source_bindings_defaults_to_base,
)
from openhcs.microscopes.microscope_base import MicroscopeHandler
from openhcs.microscopes.microscope_interfaces import MetadataHandler
from openhcs.microscopes.openhcs import (
    FIELDS,
    OpenHCSMetadataHandler,
)
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


class SourceBindingsHandler(MicroscopeHandler):
    """Handler for arbitrary image folders using source-binding declarations."""

    _microscope_type = Microscope.SOURCE_BINDINGS.value
    _metadata_handler_class = OpenHCSMetadataHandler

    @classmethod
    def create(
        cls,
        *,
        filemanager: FileManager,
        pattern_format: str | None = None,
        source_bindings_config: SourceBindingsConfig | None = None,
    ) -> "SourceBindingsHandler":
        if source_bindings_config is None:
            raise ValueError(
                "SourceBindingsHandler requires SourceBindingsConfig declarations."
            )
        return cls(
            filemanager,
            source_bindings_config=source_bindings_config,
            pattern_format=pattern_format,
        )

    def __init__(
        self,
        filemanager: FileManager,
        source_bindings_config: SourceBindingsConfig,
        pattern_format: str | None = None,
    ):
        source_bindings_config = source_bindings_defaults_to_base(
            source_bindings_config
        )
        if source_bindings_config.is_empty:
            raise ValueError(
                "SourceBindingsHandler requires non-empty SourceBindingsConfig "
                "declarations."
            )
        parser = SourceSchemaFilenameParser(filemanager, pattern_format)
        super().__init__(
            parser=parser,
            metadata_handler=OpenHCSMetadataHandler(filemanager),
        )
        from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector

        self._source_bindings_config = source_bindings_config
        self._projector = SourceBindingWorkspaceProjector(
            source_bindings=source_bindings_config,
            parser=parser,
        )

    @property
    def root_dir(self) -> str:
        return FIELDS.DEFAULT_SUBDIRECTORY

    @property
    def microscope_type(self) -> str:
        return self._microscope_type

    @property
    def metadata_handler_class(self) -> Type[MetadataHandler]:
        return OpenHCSMetadataHandler

    @property
    def compatible_backends(self) -> list[Backend]:
        return [Backend.DISK]

    @classmethod
    def detect(cls, plate_folder: Path, filemanager: FileManager) -> bool:
        del plate_folder, filemanager
        return False

    def initialize_workspace(
        self,
        plate_path: Union[str, Path],
        filemanager: FileManager,
    ) -> Path:
        from openhcs.core.source_schema_workspace import (
            materialize_source_schema_workspace,
        )

        plate_root = Path(plate_path)
        self.plate_folder = plate_root
        metadata_path = plate_root / "openhcs_metadata.json"
        if metadata_path.exists():
            from openhcs.core.source_workspace_projection import (
                VirtualWorkspaceSourceProjection,
            )

            metadata = self.metadata_handler._load_metadata_dict(plate_root)
            if VirtualWorkspaceSourceProjection.openhcs_metadata_has_workspace_mapping(
                metadata
            ):
                self._register_virtual_workspace_backend(plate_root, filemanager)
                return plate_root
        materialize_source_schema_workspace(
            plate_root,
            plate_root,
            self._projector.source_schema(),
            filemanager=filemanager,
            source_backend=Backend.DISK,
            workspace_backend=Backend.DISK,
            source_files=self._list_source_files(plate_root, filemanager),
        )
        self._register_virtual_workspace_backend(plate_root, filemanager)
        return plate_root

    def _list_source_files(
        self,
        plate_path: Path,
        filemanager: FileManager,
    ) -> tuple[Path, ...]:
        return tuple(
            Path(path)
            for path in filemanager.list_files(
                plate_path,
                Backend.DISK.value,
                recursive=True,
            )
        )
