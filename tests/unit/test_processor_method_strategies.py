"""Processor method strategy registration tests."""

from __future__ import annotations

import ast
from enum import Enum
import importlib
from pathlib import Path

import numpy as np

from openhcs.core.callable_contract import CallableContract
from openhcs.processing.backends.lib_registry.openhcs_registry import OpenHCSRegistry
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.processing.backends.processors import method_axes, numpy_processor
from openhcs.processing.backends.processors.numpy_processor import (
    NumpyStackProjectionMethod,
)
from openhcs.processing.backends.processors.method_axes import (
    SpatialBinMethod,
)


def _enum_members_for_symbol(
    tree: ast.Module,
    symbol_name: str,
) -> set[str]:
    """Resolve one local or imported enum owner without a name-specific mirror."""

    local_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == symbol_name
        ),
        None,
    )
    if local_class is not None:
        return {
            target.id
            for statement in local_class.body
            if isinstance(statement, ast.Assign)
            for target in statement.targets
            if isinstance(target, ast.Name)
        }

    imported_symbol = next(
        (
            (statement.module, imported.name)
            for statement in tree.body
            if isinstance(statement, ast.ImportFrom)
            for imported in statement.names
            if (imported.asname or imported.name) == symbol_name
        ),
        None,
    )
    assert imported_symbol is not None, symbol_name
    module_name, attribute_name = imported_symbol
    assert module_name is not None
    enum_type = getattr(importlib.import_module(module_name), attribute_name)
    assert isinstance(enum_type, type) and issubclass(enum_type, Enum)
    return {member.name for member in enum_type}


def test_numpy_processor_method_strategies_use_distinct_inherited_registries() -> None:
    assert set(numpy_processor.NumpySpatialBinStrategy.__registry__) == {
        method.value for method in SpatialBinMethod
    }
    assert set(numpy_processor.NumpyStackProjectionStrategy.__registry__) == {
        method.value for method in NumpyStackProjectionMethod
    }
    assert (
        numpy_processor.NumpySpatialBinStrategy.__registry__
        is not numpy_processor.NumpyStackProjectionStrategy.__registry__
    )


def test_numpy_processor_method_strategies_dispatch_behavior() -> None:
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)

    assert numpy_processor.spatial_bin_2d(
        stack, 2, SpatialBinMethod.MEAN
    ).shape == (2, 2, 2)
    assert numpy_processor.spatial_bin_3d(
        stack, 2, SpatialBinMethod.MAX
    ).shape == (1, 2, 2)
    assert numpy_processor.create_projection(
        stack, NumpyStackProjectionMethod.MEAN
    ).shape == (4, 4)
    np.testing.assert_array_equal(
        numpy_processor.create_projection(stack, NumpyStackProjectionMethod.MIN),
        np.min(stack, axis=0),
    )
    assert (
        CallableContract.from_callable(
            numpy_processor.create_projection
        ).require_processing_contract()
        is ProcessingContract.VOLUMETRIC_TO_SLICE
    )


def test_projection_method_types_exactly_match_registered_dispatch() -> None:
    processor_root = (
        Path(__file__).parents[2]
        / "openhcs/processing/backends/processors"
    )
    discovered: dict[str, set[str]] = {}

    for path in processor_root.glob("*_processor.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        create_projection = functions.get("create_projection")
        if create_projection is None:
            continue
        method_parameter = next(
            argument
            for argument in (
                *create_projection.args.args,
                *create_projection.args.kwonlyargs,
            )
            if argument.arg == "method"
        )
        method_type_name = ast.unparse(method_parameter.annotation)
        accepted_members = _enum_members_for_symbol(tree, method_type_name)

        strategy_roots = {
            node.name
            for node in tree.body
            if isinstance(node, ast.ClassDef)
                and any(
                    isinstance(base, ast.Subscript)
                    and ast.unparse(base.value) == "EnumKeyedStrategyMixin"
                    and ast.unparse(base.slice) == method_type_name
                    for base in node.bases
                )
                and any(
                    isinstance(statement, ast.Assign)
                    and any(
                        isinstance(target, ast.Name)
                        and target.id == "__enum_member_attr__"
                        for target in statement.targets
                    )
                    and isinstance(statement.value, ast.Constant)
                    and statement.value.value == "method"
                    for statement in node.body
                )
        }
        registered_members = {
            statement.value.attr
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(
                isinstance(base, ast.Name) and base.id in strategy_roots
                for base in node.bases
            )
            for statement in node.body
            if isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "method"
                for target in statement.targets
            )
            and isinstance(statement.value, ast.Attribute)
        }
        discovered[path.name] = registered_members
        assert registered_members == accepted_members, path

    assert discovered


def test_processor_method_roots_use_shared_enum_strategy_directly() -> None:
    method_axes_source = (
        Path(method_axes.__file__).read_text(encoding="utf-8")
    )

    assert "RegisteredProcessorMethodStrategy" not in method_axes_source


def test_openhcs_registry_discovery_does_not_replace_public_processor_callables() -> (
    None
):
    declared_projection = numpy_processor.create_projection
    registry = OpenHCSRegistry()
    registry.MODULES_TO_SCAN = [numpy_processor.__name__]

    functions = registry.discover_functions()

    assert numpy_processor.create_projection is declared_projection
    assert (
        functions["processors_numpy_processor_create_projection"].func
        is not declared_projection
    )
    stack = np.arange(2 * 4 * 4, dtype=np.uint16).reshape(2, 4, 4)
    assert numpy_processor.create_projection(
        stack, NumpyStackProjectionMethod.MEAN
    ).shape == (4, 4)
