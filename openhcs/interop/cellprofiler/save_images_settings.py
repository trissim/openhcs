"""Typed CellProfiler SaveImages setting semantics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.runtime_exports import RuntimeImageExportBitDepth

from .parser import ModuleBlock
from .setting_names import SettingNameFamily, optional_setting_value


SAVE_IMAGES_SOURCE_IMAGE_SETTING = SettingNameFamily("Select the image to save")
SAVE_IMAGES_BIT_DEPTH_SETTING = SettingNameFamily("Image bit depth")
SAVE_IMAGES_FILE_FORMAT_SETTING = SettingNameFamily("Saved file format")


class SaveImagesBitDepthLiteral(ABC, metaclass=AutoRegisterMeta):
    """Nominal parser for CellProfiler SaveImages bit-depth UI literals."""

    __registry_key__ = "literal"
    __skip_if_no_key__ = True
    literal: ClassVar[str | None] = None

    @classmethod
    def parse(cls, value: str | None) -> RuntimeImageExportBitDepth:
        if value is None:
            return RuntimeImageExportBitDepth.NATIVE
        literal = value.strip().lower()
        parser_type = cls.__registry__.get(literal)
        if parser_type is None:
            raise ValueError(f"Unsupported SaveImages bit depth {value!r}.")
        return parser_type().bit_depth()

    @abstractmethod
    def bit_depth(self) -> RuntimeImageExportBitDepth:
        """Return the OpenHCS runtime export bit depth."""


class SaveImagesUint8BitDepth(SaveImagesBitDepthLiteral):
    literal = "8-bit integer"

    def bit_depth(self) -> RuntimeImageExportBitDepth:
        return RuntimeImageExportBitDepth.UINT8


class SaveImagesUint16BitDepth(SaveImagesBitDepthLiteral):
    literal = "16-bit integer"

    def bit_depth(self) -> RuntimeImageExportBitDepth:
        return RuntimeImageExportBitDepth.UINT16


class SaveImagesFloat32BitDepth(SaveImagesBitDepthLiteral):
    literal = "32-bit floating point"

    def bit_depth(self) -> RuntimeImageExportBitDepth:
        return RuntimeImageExportBitDepth.FLOAT32


class SaveImagesNativeBitDepth(SaveImagesBitDepthLiteral):
    literal = "raw"

    def bit_depth(self) -> RuntimeImageExportBitDepth:
        return RuntimeImageExportBitDepth.NATIVE


def save_images_bit_depth(module: ModuleBlock) -> RuntimeImageExportBitDepth:
    """Return the runtime export bit depth declared by a SaveImages module."""
    try:
        return SaveImagesBitDepthLiteral.parse(
            optional_setting_value(module, SAVE_IMAGES_BIT_DEPTH_SETTING)
        )
    except ValueError as exc:
        raise ValueError(
            f"Unsupported SaveImages bit depth in module {module.module_num}."
        ) from exc
