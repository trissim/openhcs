"""Compile-time CellProfiler module settings carried by generated steps."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import ClassVar

from openhcs.core.function_patterns import CompileTimeFunctionKwarg
from openhcs.core.python_source_literal import PythonSourceLiteral
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting


@dataclass(frozen=True, slots=True)
class CellProfilerModuleSettingsPayload(PythonSourceLiteral):
    """Serializable subset of a CellProfiler module declaration."""

    module_name: str
    module_num: int
    setting_records: tuple[tuple[str, str], ...]
    enabled: bool = True

    @classmethod
    def from_module(cls, module: ModuleBlock) -> "CellProfilerModuleSettingsPayload":
        """Capture ordered settings needed to reconstruct artifact contracts."""
        records = tuple((record.name, record.value) for record in module.setting_records)
        if not records:
            records = tuple(module.settings.items())
        return cls(
            module_name=module.name,
            module_num=module.module_num,
            setting_records=records,
            enabled=module.enabled,
        )

    def module_block(self) -> ModuleBlock:
        """Reconstruct the parser module block used by the CP symbol table."""
        records = [
            ModuleSetting(name=name, value=value)
            for name, value in self.setting_records
        ]
        return ModuleBlock(
            name=self.module_name,
            module_num=self.module_num,
            enabled=self.enabled,
            settings={record.name: record.value for record in records},
            setting_records=records,
        )

    def source_literal(self) -> str:
        """Return importable Python source for generated pipeline files."""
        constructor_args = ", ".join(repr(value) for value in astuple(self))
        return f"{type(self).__name__}({constructor_args})"

    def source_literal_imports(self) -> frozenset[tuple[str, str]]:
        """Return imports needed by generated source."""
        return frozenset({(type(self).__module__, type(self).__name__)})


class CellProfilerModuleSettingsKwarg(
    CompileTimeFunctionKwarg[CellProfilerModuleSettingsPayload]
):
    """Registered compile-only kwarg for CellProfiler module settings."""

    payload_type: ClassVar[type[object] | None] = CellProfilerModuleSettingsPayload

    @classmethod
    def invocation_contract_provider_for_session(
        cls,
        session: object,
    ):
        """Return CP runtime contracts derived from generated module settings."""
        from openhcs.interop.cellprofiler.compile_time_contracts import (
            cellprofiler_module_settings_invocation_contract_provider_for_session,
        )

        return cellprofiler_module_settings_invocation_contract_provider_for_session(
            session
        )
