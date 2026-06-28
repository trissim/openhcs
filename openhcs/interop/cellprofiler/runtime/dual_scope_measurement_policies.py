"""CellProfiler dual-scope measurement policy contracts."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, TypeVar

from metaclass_registry import RegistryFamily, RegistryKeyAttribute

from openhcs.interop.cellprofiler.runtime.payload_types import CellProfilerFunction
from openhcs.interop.cellprofiler.runtime.policy_registry import (
    CellProfilerModulePolicyAutoRegisterMeta,
    CellProfilerModulePolicyLookupMixin,
)
from openhcs.processing.backends.cellprofiler.library import require_function

RequiredAttrT = TypeVar("RequiredAttrT")


class CellProfilerDualScopeMeasurementPolicyMixin:
    """Declaration-owned policy for modules with image and object measurement scope."""

    image_function_name: ClassVar[str | None] = None

    def image_function(self, object_func: CellProfilerFunction) -> CellProfilerFunction:
        del object_func
        return require_function(
            _required_class_attr(type(self).module_name, "module_name"),
            function_name=_required_class_attr(
                type(self).image_function_name,
                "image_function_name",
            ),
        )


class CellProfilerDualScopeMeasurementPolicy(
    CellProfilerDualScopeMeasurementPolicyMixin,
    CellProfilerModulePolicyLookupMixin,
    ABC,
    metaclass=CellProfilerModulePolicyAutoRegisterMeta,
):
    """Registered lookup root for dual-scope measurement policies."""

    __registry_family__ = RegistryFamily(RegistryKeyAttribute.REGISTRY_KEY)
    declaration_policy_bases = (CellProfilerDualScopeMeasurementPolicyMixin,)
    fallback_registry_key = None


class DeclaredDualScopeMeasurementPolicy(CellProfilerDualScopeMeasurementPolicyMixin):
    """Shared base for modules with image+object measurement scope."""


def _required_class_attr(value: RequiredAttrT | None, name: str) -> RequiredAttrT:
    if value is None:
        raise TypeError(f"CellProfiler policy must define {name}.")
    return value
