"""Mixins for common microscope detection patterns."""

from pathlib import Path
from typing import Type

from openhcs.microscopes.microscope_interfaces import MetadataHandler
from polystore.exceptions import MetadataNotFoundError
from polystore.filemanager import FileManager


class MetadataDetectMixin:
    """
    Provides a detect() implementation that delegates to a metadata handler.

    Handlers declare `_metadata_handler_class` (already used by the registry);
    no duplicate class attributes are required.
    """

    @classmethod
    def detect(cls, plate_folder: Path, filemanager: FileManager) -> bool:
        handler_cls: Type[MetadataHandler] = getattr(cls, "_metadata_handler_class", None)
        if handler_cls is None:
            raise RuntimeError(f"{cls.__name__} missing _metadata_handler_class for detection")

        handler = handler_cls(filemanager)
        try:
            handler.find_metadata_file(plate_folder)
            return True
        except (MetadataNotFoundError, FileNotFoundError, TypeError):
            return False
