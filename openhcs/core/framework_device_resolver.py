"""Compile-time resolution of framework-local GPU device assignments."""

from __future__ import annotations

from dataclasses import dataclass

from arraybridge import MemoryType

from openhcs.core.compiled_step_plan import FrameworkDeviceAssignment
from openhcs.utils.environment import OpenHCSProcessEnvironment


@dataclass(frozen=True, slots=True)
class FrameworkDeviceResolver:
    """Resolve declared GPU frameworks to available local device identifiers."""

    preferred_device_id: int = 0

    def resolve(
        self,
        memory_types: frozenset[MemoryType],
    ) -> FrameworkDeviceAssignment:
        """Resolve every requested declaration without conflating namespaces."""

        if not memory_types:
            return FrameworkDeviceAssignment()
        if OpenHCSProcessEnvironment.gpu_imports_disabled():
            raise ValueError(
                "GPU execution was requested while GPU imports are disabled for "
                "this process."
            )

        assignments: dict[MemoryType, int] = {}
        missing_packages: list[str] = []
        unavailable_devices: list[str] = []
        for memory_type in sorted(memory_types, key=lambda item: item.value):
            framework = memory_type.import_if_installed()
            if framework is None:
                missing_packages.append(memory_type.import_name)
                continue
            available = memory_type.available_device_ids(framework)
            if not available:
                unavailable_devices.append(memory_type.display_name)
                continue
            assignments[memory_type] = (
                self.preferred_device_id
                if self.preferred_device_id in available
                else available[0]
            )

        if missing_packages:
            raise ValueError(
                "Required GPU framework packages are not installed: "
                f"{', '.join(missing_packages)}."
            )
        if unavailable_devices:
            raise ValueError(
                "No GPU device is available through: "
                f"{', '.join(unavailable_devices)}."
            )
        return FrameworkDeviceAssignment.from_mapping(assignments)


def resolve_framework_devices(
    memory_types: frozenset[MemoryType],
) -> FrameworkDeviceAssignment:
    """Resolve one pipeline's declaration-derived GPU footprint."""

    return FrameworkDeviceResolver().resolve(memory_types)


__all__ = ["FrameworkDeviceResolver", "resolve_framework_devices"]
