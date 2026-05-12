"""Compiler-bound preparation for AutoRegisterMeta registries."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from functools import lru_cache
from types import ModuleType
from typing import Iterable

from metaclass_registry import AutoRegisterMeta


@dataclass(slots=True)
class AutoRegisterRegistryPreparationReport:
    """Summary of AutoRegister registries prepared before worker execution."""

    registry_count: int = 0
    class_count: int = 0
    failures: dict[str, str] = field(default_factory=dict)


class AutoRegisterRegistryPreparation:
    """Prepare loaded AutoRegisterMeta registries at compiler/runtime boundaries."""

    @classmethod
    def prepare_loaded_registries(cls) -> AutoRegisterRegistryPreparationReport:
        """Force discovery for every loaded AutoRegisterMeta registry family."""
        return cls.prepare_module_registries(tuple(sys.modules.values()))

    @classmethod
    def prepare_module_registries(
        cls,
        modules: Iterable[ModuleType | None],
    ) -> AutoRegisterRegistryPreparationReport:
        """Force discovery for AutoRegisterMeta registries defined in modules."""
        report = AutoRegisterRegistryPreparationReport()
        prepared_registry_ids: set[int] = set()
        for module in modules:
            if not isinstance(module, ModuleType):
                continue
            for candidate in cls.module_registry_families(module):
                registry = candidate.__registry__
                registry_id = id(registry)
                if registry_id in prepared_registry_ids:
                    continue
                try:
                    report.class_count += len(tuple(registry.values()))
                    report.registry_count += 1
                    prepared_registry_ids.add(registry_id)
                except Exception as exc:
                    report.failures[
                        f"{candidate.__module__}.{candidate.__qualname__}"
                    ] = str(exc)
        return report

    @staticmethod
    def module_registry_families(
        module: ModuleType,
    ) -> tuple[type[object], ...]:
        """Return AutoRegisterMeta classes in a module that own registries."""
        return AutoRegisterRegistryPreparation.cached_module_registry_families(
            module.__name__
        )

    @staticmethod
    @lru_cache(maxsize=None)
    def cached_module_registry_families(
        module_name: str,
    ) -> tuple[type[object], ...]:
        """Return cached AutoRegisterMeta registry families for one loaded module."""
        module = sys.modules.get(module_name)
        if module is None:
            return ()
        families: list[type[object]] = []
        for candidate in vars(module).values():
            if not isinstance(candidate, type):
                continue
            if not isinstance(candidate, AutoRegisterMeta):
                continue
            try:
                candidate.__registry__
            except AttributeError:
                continue
            families.append(candidate)
        return tuple(families)
