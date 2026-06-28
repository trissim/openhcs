"""CellProfiler image-save infrastructure module declarations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.artifacts import ArtifactKind
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    optional_setting_value,
    setting_values,
    split_symbol_names,
)
from openhcs.processing.backends.cellprofiler.module_classes import (
    InfrastructureCellProfilerModule,
)

from openhcs.core.runtime_exports import RuntimeImageExportBitDepth


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

class SaveCroppedObjectsModule(InfrastructureCellProfilerModule):
    module_name = 'SaveCroppedObjects'
    function_name = 'save_cropped_objects'
    validated = True
    contract = 'unknown'
    confidence = 1.0


class SaveImagesModule(InfrastructureCellProfilerModule):
    module_name = 'SaveImages'
    function_name = 'save_images'
    validated = True
    confidence = 1.0
    infrastructure_import_note = "SaveImages -> handled by runtime image materialization"
    infrastructure_exports_images = True
    source_image_setting = SettingNameFamily("Select the image to save")
    bit_depth_setting = SettingNameFamily("Image bit depth")
    file_format_setting = SettingNameFamily("Saved file format")

    @classmethod
    def infrastructure_retained_artifacts(
        cls,
        module: "ModuleBlock",
        *,
        contracts_by_module_num: Mapping[int, "ModuleArtifactContracts"],
    ) -> frozenset["ArtifactSpecKey"]:
        del contracts_by_module_num
        from openhcs.interop.cellprofiler.module_roles import ArtifactSpecKey

        return frozenset(
            ArtifactSpecKey(ArtifactKind.IMAGE, image_name)
            for value in setting_values(module, cls.source_image_setting)
            for image_name in split_symbol_names(value)
        )

    @classmethod
    def image_export_specs(
        cls,
        module: "ModuleBlock",
    ) -> tuple["RuntimeImageExportSpec", ...]:
        """Return runtime image-export expectations declared by SaveImages."""
        from openhcs.core.runtime_exports import RuntimeImageExportSpec

        try:
            bit_depth = SaveImagesBitDepthLiteral.parse(
                optional_setting_value(module, cls.bit_depth_setting)
            )
        except ValueError as exc:
            raise ValueError(
                f"Unsupported SaveImages bit depth in module {module.module_num}."
            ) from exc
        return tuple(
            RuntimeImageExportSpec(
                artifact_name=image_name,
                bit_depth=bit_depth,
                file_format=optional_setting_value(module, cls.file_format_setting),
            )
            for value in setting_values(module, cls.source_image_setting)
            for image_name in split_symbol_names(value)
        )
