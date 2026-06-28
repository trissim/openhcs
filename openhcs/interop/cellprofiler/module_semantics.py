"""Compatibility CellProfiler module semantics derived from declarations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nominal_refactor_advisor.descriptor_algebra import AliasProperty

from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


class CellProfilerModuleCategory(Enum):
    """Manual-facing CellProfiler module category."""

    INPUT = ("Input", True)
    FILE_PROCESSING = ("File Processing", True)
    IMAGE_PROCESSING = ("Image Processing", False)
    OBJECT_PROCESSING = ("Object Processing", False)
    MEASUREMENT = ("Measurement", False)
    ADVANCED = ("Advanced", False)
    WORM_TOOLBOX = ("Worm Toolbox", False)
    OTHER = ("Other", False)
    DATA_TOOLS = ("Data Tools", False)

    def __new__(cls, label: str, is_infrastructure: bool):
        obj = object.__new__(cls)
        obj._value_ = label
        obj._is_infrastructure = is_infrastructure
        return obj

    is_infrastructure = AliasProperty[bool]("_is_infrastructure")


class CellProfilerModuleDimensionality(Enum):
    """Manual-facing dimensional support mapped onto OpenHCS execution contracts."""

    PLANAR = ProcessingContract.PURE_2D
    VOLUMETRIC = ProcessingContract.PURE_3D
    PLANAR_AND_VOLUMETRIC = ProcessingContract.FLEXIBLE

    processing_contract = AliasProperty[ProcessingContract]("value")

    @property
    def supports_2d(self) -> bool:
        return self is not CellProfilerModuleDimensionality.VOLUMETRIC

    @property
    def supports_3d(self) -> bool:
        return self is not CellProfilerModuleDimensionality.PLANAR


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSemanticTraits:
    """Shared semantic traits projected from CellProfiler module declarations."""

    category: CellProfilerModuleCategory
    dimensionality: CellProfilerModuleDimensionality
    respects_masks: bool

    @property
    def supports_2d(self) -> bool:
        """Return CellProfiler manual 2D support derived from dimensionality."""
        return self.dimensionality.supports_2d

    @property
    def supports_3d(self) -> bool:
        """Return CellProfiler manual 3D support derived from dimensionality."""
        return self.dimensionality.supports_3d

    @property
    def is_infrastructure(self) -> bool:
        """Return whether OpenHCS handles this as runtime/file infrastructure."""
        return self.category.is_infrastructure


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSemantics(CellProfilerModuleSemanticTraits):
    """Typed compatibility semantics for one CellProfiler module."""

    module_name: str


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSemanticFamily(CellProfilerModuleSemanticTraits):
    """Declaration-derived family for modules with the same semantic traits."""

    family_name: str
    module_names: tuple[str, ...]


def cellprofiler_module_semantics(
    module_name: str,
) -> CellProfilerModuleSemantics | None:
    """Return declaration-derived semantics for a CellProfiler module name, if known."""
    normalized_name = module_name.strip()
    if not normalized_name:
        raise ValueError("CellProfiler module name cannot be empty.")
    lookup_key = normalized_name.casefold()
    semantics_by_name = {
        semantics.module_name.casefold(): semantics
        for semantics in _declared_semantics()
    }
    if lookup_key in semantics_by_name:
        return semantics_by_name[lookup_key]
    for alias, canonical in _declared_aliases():
        if alias.casefold() == lookup_key:
            return semantics_by_name.get(canonical.casefold())
    return None


def cellprofiler_module_semantics_family(
    module_name: str,
) -> CellProfilerModuleSemanticFamily | None:
    """Return the declaration-derived semantic family for one module."""
    semantics = cellprofiler_module_semantics(module_name)
    if semantics is None:
        return None
    family_members = tuple(
        candidate.module_name
        for candidate in _declared_semantics()
        if candidate.category is semantics.category
        and candidate.dimensionality is semantics.dimensionality
        and candidate.respects_masks is semantics.respects_masks
    )
    return CellProfilerModuleSemanticFamily(
        family_name=(
            f"{semantics.category.value} / {semantics.dimensionality.processing_contract.value}"
        ),
        category=semantics.category,
        dimensionality=semantics.dimensionality,
        respects_masks=semantics.respects_masks,
        module_names=family_members,
    )


def _declared_dimensionality(contract_name: str) -> CellProfilerModuleDimensionality:
    contract = ProcessingContract.from_declared_name(contract_name)
    if contract is ProcessingContract.PURE_2D:
        return CellProfilerModuleDimensionality.PLANAR
    if contract is ProcessingContract.PURE_3D:
        return CellProfilerModuleDimensionality.VOLUMETRIC
    return CellProfilerModuleDimensionality.PLANAR_AND_VOLUMETRIC


def _declared_category(module_type: type[object]) -> CellProfilerModuleCategory:
    from openhcs.processing.backends.cellprofiler.module_classes import (
        InfrastructureCellProfilerModule,
        ObjectMeasurementRowsModule,
        ScopedMeasurementModule,
    )

    if issubclass(module_type, InfrastructureCellProfilerModule):
        return CellProfilerModuleCategory.FILE_PROCESSING
    if issubclass(module_type, (ObjectMeasurementRowsModule, ScopedMeasurementModule)):
        return CellProfilerModuleCategory.MEASUREMENT
    declared_category = module_type.category
    if declared_category in {"image_operation", "channel_operation", "z_projection"}:
        return CellProfilerModuleCategory.IMAGE_PROCESSING
    return CellProfilerModuleCategory.OTHER


def _declared_semantics() -> tuple[CellProfilerModuleSemantics, ...]:
    from openhcs.interop.cellprofiler.source_schema import SetupModuleCompiler
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    backend_semantics = tuple(
        CellProfilerModuleSemantics(
            module_name=str(module_type.module_name),
            category=_declared_category(module_type),
            dimensionality=_declared_dimensionality(module_type.contract),
            respects_masks=False,
        )
        for module_type in CellProfilerModule.__registry__.values()
    )
    setup_semantics = tuple(
        CellProfilerModuleSemantics(
            module_name=str(module_name),
            category=CellProfilerModuleCategory.INPUT,
            dimensionality=CellProfilerModuleDimensionality.PLANAR_AND_VOLUMETRIC,
            respects_masks=False,
        )
        for module_name in SetupModuleCompiler.__registry__
    )
    return (*backend_semantics, *setup_semantics)


def _declared_aliases() -> tuple[tuple[str, str], ...]:
    from openhcs.processing.backends.cellprofiler.module_classes import (
        CellProfilerModule,
    )

    return tuple(
        (alias, str(module_type.module_name))
        for module_type in CellProfilerModule.__registry__.values()
        for alias in module_type.aliases
    )
