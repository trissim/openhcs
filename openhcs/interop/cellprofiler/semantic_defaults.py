"""Source-derived CellProfiler semantic-default contracts.

These contracts guard defaults that CellProfiler applies in module glue code
rather than in the underlying library function signature.
"""

from __future__ import annotations

import ast
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.callable_contract import CallableContract


SOURCE_MODULE_ROOT = (
    Path(__file__).resolve().parents[3]
    / "benchmark"
    / "cellprofiler_source"
    / "modules"
)


@dataclass(frozen=True, slots=True)
class SourceDictField:
    """One expected source dict field and its absorbed semantic value."""

    source_key: str
    absorbed_value: object


@dataclass(frozen=True, slots=True)
class SourceCallKeyword:
    """One source call keyword that must match an absorbed callable default."""

    callable_name: str
    keyword_name: str
    absorbed_callable: Callable[..., Any]


class SourceModuleSemantics:
    """Typed AST view over one vendored CellProfiler module source file."""

    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self.syntax_tree = ast.parse(source_path.read_text())
        self.constant_values = self.collect_module_constants()

    def collect_module_constants(self) -> Mapping[str, object]:
        values: dict[str, object] = {}
        for statement in self.syntax_tree.body:
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and isinstance(statement.value, ast.Constant):
                    values[target.id] = statement.value.value
        return values

    def literal_dict(self, variable_name: str) -> Mapping[str, object]:
        for statement in self.syntax_tree.body:
            if not isinstance(statement, ast.Assign):
                continue
            if len(statement.targets) != 1:
                continue
            target = statement.targets[0]
            if not isinstance(target, ast.Name) or target.id != variable_name:
                continue
            if not isinstance(statement.value, ast.Dict):
                raise TypeError(f"{variable_name} is not a literal dict in {self.source_path}")
            return {
                self.literal_value(key): self.literal_value(value)
                for key, value in zip(statement.value.keys, statement.value.values, strict=True)
            }
        raise KeyError(f"{variable_name} not found in {self.source_path}")

    def call_keyword_value(self, callable_name: str, keyword_name: str) -> object:
        for node in ast.walk(self.syntax_tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != callable_name:
                continue
            for keyword in node.keywords:
                if keyword.arg == keyword_name:
                    return self.literal_value(keyword.value)
        raise KeyError(
            f"{callable_name}(..., {keyword_name}=...) not found in {self.source_path}"
        )

    def supports_volumetric_images(self) -> bool:
        docstring = ast.get_docstring(self.syntax_tree) or ""
        return "Supports 3D?" in docstring and "YES          YES" in docstring

    def call_receives_pixel_data(self, callable_name: str) -> bool:
        pixel_data_names = self.pixel_data_names()
        for node in ast.walk(self.syntax_tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != callable_name:
                continue
            for argument in node.args:
                if isinstance(argument, ast.Name) and argument.id in pixel_data_names:
                    return True
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Name) and keyword.value.id in pixel_data_names:
                    return True
        return False

    def pixel_data_names(self) -> frozenset[str]:
        names: set[str] = set()
        for node in ast.walk(self.syntax_tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if self.is_pixel_data_expression(node.value):
                names.add(target.id)
        return frozenset(names)

    def is_pixel_data_expression(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "pixel_data"
        )

    def literal_value(self, node: ast.AST | None) -> object:
        if isinstance(node, ast.Constant):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            return -node.operand.value
        if isinstance(node, ast.Name):
            return self.constant_values[node.id]
        raise TypeError(
            f"Unsupported semantic default expression in {self.source_path}: "
            f"{ast.dump(node)}"
        )


class CellProfilerSemanticDefaultContract(ABC, metaclass=AutoRegisterMeta):
    """Auto-registered source-vs-absorbed semantic-default contract."""

    __registry_key__ = "contract_key"
    __skip_if_no_key__ = True

    contract_key: ClassVar[str | None] = None
    module_name: ClassVar[str | None] = None
    source_filename: ClassVar[str | None] = None

    @classmethod
    def registered_contracts(cls) -> tuple["CellProfilerSemanticDefaultContract", ...]:
        return tuple(contract_type() for contract_type in cls.__registry__.values())

    def validate(self) -> None:
        semantics = SourceModuleSemantics(self.source_path)
        self.validate_against_source(semantics)

    @property
    def source_path(self) -> Path:
        if self.source_filename is None:
            raise ValueError(f"{type(self).__name__} must declare source_filename.")
        return SOURCE_MODULE_ROOT / self.source_filename

    @abstractmethod
    def validate_against_source(self, semantics: SourceModuleSemantics) -> None:
        """Raise if absorbed semantics diverge from vendored CellProfiler source."""

    def require_equal(self, label: str, source_value: object, absorbed_value: object) -> None:
        if source_value != absorbed_value:
            raise AssertionError(
                f"{type(self).__name__} mismatch for {label}: "
                f"source={source_value!r}, absorbed={absorbed_value!r}"
            )


class SourceDictSemanticDefaultContract(CellProfilerSemanticDefaultContract):
    """Contract for module-level source dict defaults mirrored by absorbed semantics."""

    source_dict_name: ClassVar[str | None] = None

    @abstractmethod
    def source_dict_fields(self) -> tuple[SourceDictField, ...]:
        """Return source dict fields and their absorbed values."""

    def validate_against_source(self, semantics: SourceModuleSemantics) -> None:
        if self.source_dict_name is None:
            raise ValueError(f"{type(self).__name__} must declare source_dict_name.")
        source_defaults = semantics.literal_dict(self.source_dict_name)
        for field in self.source_dict_fields():
            self.require_equal(
                field.source_key,
                self.normalize_source_value(source_defaults[field.source_key]),
                self.normalize_absorbed_value(field.absorbed_value),
            )

    def normalize_source_value(self, value: object) -> object:
        return value

    def normalize_absorbed_value(self, value: object) -> object:
        return value


class SourceCallKeywordDefaultContract(CellProfilerSemanticDefaultContract):
    """Contract for source call keywords that must be absorbed callable defaults."""

    @abstractmethod
    def source_call_keywords(self) -> tuple[SourceCallKeyword, ...]:
        """Return source call keywords and absorbed callables that mirror them."""

    def validate_against_source(self, semantics: SourceModuleSemantics) -> None:
        for keyword in self.source_call_keywords():
            source_value = semantics.call_keyword_value(
                keyword.callable_name,
                keyword.keyword_name,
            )
            absorbed_default = self.callable_default(
                keyword.absorbed_callable,
                keyword.keyword_name,
            )
            self.require_equal(keyword.keyword_name, source_value, absorbed_default)

    def callable_default(
        self,
        absorbed_callable: Callable[..., Any],
        keyword_name: str,
    ) -> object:
        signature = inspect.signature(inspect.unwrap(absorbed_callable))
        parameter = signature.parameters[keyword_name]
        if parameter.default is inspect.Parameter.empty:
            raise AssertionError(
                f"{absorbed_callable.__name__}.{keyword_name} has no absorbed default."
            )
        return parameter.default


class SourceVolumetricPixelDataExecutionContract(CellProfilerSemanticDefaultContract):
    """Contract for CP modules that pass volumetric pixel_data through directly."""

    callable_name: ClassVar[str | None] = None
    absorbed_callable: ClassVar[Callable[..., Any] | None] = None
    required_execution_mode: ClassVar[ImagePayloadExecutionMode] = (
        ImagePayloadExecutionMode.FULL_STACK
    )

    def validate_against_source(self, semantics: SourceModuleSemantics) -> None:
        if self.callable_name is None:
            raise ValueError(f"{type(self).__name__} must declare callable_name.")
        if self.absorbed_callable is None:
            raise ValueError(f"{type(self).__name__} must declare absorbed_callable.")
        if not semantics.supports_volumetric_images():
            return
        if not semantics.call_receives_pixel_data(self.callable_name):
            return
        contract = CallableContract.from_callable(self.absorbed_callable)
        self.require_equal(
            "runtime_image_execution_mode",
            self.required_execution_mode,
            contract.runtime_image_execution_mode,
        )


class MedianFilterSemanticDefaultContract(SourceCallKeywordDefaultContract):
    contract_key = "MedianFilter.semantic_defaults"
    module_name = "MedianFilter"
    source_filename = "medianfilter.py"

    def source_call_keywords(self) -> tuple[SourceCallKeyword, ...]:
        from openhcs.processing.backends.cellprofiler.median_filter import medianfilter

        return (
            SourceCallKeyword(
                callable_name="medianfilter",
                keyword_name="mode",
                absorbed_callable=medianfilter,
            ),
        )


class MedianFilterExecutionDomainContract(SourceVolumetricPixelDataExecutionContract):
    contract_key = "MedianFilter.execution_domain"
    module_name = "MedianFilter"
    source_filename = "medianfilter.py"
    callable_name = "medianfilter"

    @property
    def absorbed_callable(self) -> Callable[..., Any]:
        from openhcs.processing.backends.cellprofiler.median_filter import medianfilter

        return medianfilter


class WatershedBasicSemanticDefaultContract(SourceDictSemanticDefaultContract):
    contract_key = "Watershed.basic_defaults"
    module_name = "Watershed"
    source_filename = "watershed.py"
    source_dict_name = "basic_mode_defaults"

    def source_dict_fields(self) -> tuple[SourceDictField, ...]:
        from openhcs.processing.backends.cellprofiler.watershed import (
            CELLPROFILER_WATERSHED_BASIC_DEFAULTS,
        )

        defaults = CELLPROFILER_WATERSHED_BASIC_DEFAULTS
        return (
            SourceDictField("seed_method", defaults.seed_method),
            SourceDictField("max_seeds", defaults.max_seeds),
            SourceDictField("min_distance", defaults.min_distance),
            SourceDictField("min_intensity", defaults.min_intensity),
            SourceDictField("connectivity", defaults.connectivity),
            SourceDictField("compactness", defaults.compactness),
            SourceDictField("watershed_line", defaults.watershed_line),
            SourceDictField("gaussian_sigma", defaults.gaussian_sigma),
        )

    def normalize_source_value(self, value: object) -> object:
        return value.casefold() if isinstance(value, str) else value

    def normalize_absorbed_value(self, value: object) -> object:
        return value.value if isinstance(value, Enum) else value


class WatershedExecutionDomainContract(SourceVolumetricPixelDataExecutionContract):
    contract_key = "Watershed.execution_domain"
    module_name = "Watershed"
    source_filename = "watershed.py"
    callable_name = "watershed"

    @property
    def absorbed_callable(self) -> Callable[..., Any]:
        from openhcs.processing.backends.cellprofiler.watershed import watershed

        return watershed
