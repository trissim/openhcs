"""Focused tests for runtime callable resolution ownership."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

from openhcs.core.callable_contract import CallableContract, CallableMetadata
from openhcs.core.function_reference import FunctionReference
from openhcs.core.runtime_adapters import RuntimeAdapterSpec

PROJECT_ROOT = Path(__file__).parents[2]
CALLABLE_CONTRACT_PATH = PROJECT_ROOT / "openhcs" / "core" / "callable_contract.py"


def _adapter_spec(
    runtime_callable_factory: Callable | None = None,
) -> RuntimeAdapterSpec:
    return RuntimeAdapterSpec(
        parameter_name="runtime_adapter",
        factory=lambda _request: object(),
        runtime_callable_factory=runtime_callable_factory,
    )


def _contract(
    func: Callable[..., object] | FunctionReference,
    runtime_adapter: RuntimeAdapterSpec | None,
) -> CallableContract:
    return CallableContract(
        func=func,
        function_name="raw_callable",
        module_name=__name__,
        metadata=CallableMetadata(runtime_adapter=runtime_adapter),
    )


def test_runtime_adapter_without_runtime_factory_returns_resolved_callable() -> None:
    def raw_callable(value: object) -> object:
        return value

    spec = _adapter_spec()
    contract = _contract(raw_callable, spec)

    assert spec.executable_callable(raw_callable, contract) is raw_callable
    assert contract.resolve_runtime_callable() is raw_callable


def test_function_reference_without_runtime_adapter_returns_resolved_callable(
    monkeypatch,
) -> None:
    def raw_callable(value: object) -> object:
        return value

    reference = FunctionReference(
        function_name="raw_callable",
        registry_name="python",
        memory_type="python",
        composite_key="python:tests:raw_callable",
        original_module=__name__,
    )
    contract = _contract(reference, None)
    resolve_calls: list[FunctionReference] = []

    def resolve(current: FunctionReference) -> Callable[..., object]:
        resolve_calls.append(current)
        return raw_callable

    monkeypatch.setattr(FunctionReference, "resolve", resolve)

    assert contract.resolve_runtime_callable() is raw_callable
    assert resolve_calls == [reference]


def test_runtime_adapter_runtime_factory_is_called_exactly_once() -> None:
    def raw_callable(value: object) -> object:
        return value

    def executable_callable(value: object) -> object:
        return value

    calls: list[tuple[Callable[..., object], CallableContract]] = []

    def runtime_callable_factory(
        resolved_callable: Callable[..., object],
        contract: CallableContract,
    ) -> Callable[..., object]:
        calls.append((resolved_callable, contract))
        return executable_callable

    spec = _adapter_spec(runtime_callable_factory)
    contract = _contract(raw_callable, spec)

    assert contract.resolve_runtime_callable() is executable_callable
    assert calls == [(raw_callable, contract)]


def test_function_reference_resolves_once_before_runtime_factory(
    monkeypatch,
) -> None:
    def raw_callable(value: object) -> object:
        return value

    def executable_callable(value: object) -> object:
        return value

    factory_calls: list[tuple[Callable[..., object], CallableContract]] = []

    def runtime_callable_factory(
        resolved_callable: Callable[..., object],
        contract: CallableContract,
    ) -> Callable[..., object]:
        factory_calls.append((resolved_callable, contract))
        return executable_callable

    spec = _adapter_spec(runtime_callable_factory)
    reference = FunctionReference(
        function_name="raw_callable",
        registry_name="python",
        memory_type="python",
        composite_key="python:tests:raw_callable",
        original_module=__name__,
        metadata=CallableMetadata(runtime_adapter=spec),
    )
    contract = _contract(reference, spec)
    resolve_calls: list[FunctionReference] = []

    def resolve(current: FunctionReference) -> Callable[..., object]:
        resolve_calls.append(current)
        return raw_callable

    monkeypatch.setattr(FunctionReference, "resolve", resolve)

    assert contract.resolve_runtime_callable() is executable_callable
    assert resolve_calls == [reference]
    assert factory_calls == [(raw_callable, contract)]


def _method_node(
    path: Path,
    class_name: str,
    method_name: str,
) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_runtime_callable_resolution_has_no_scan_fallback_or_cache_mirror() -> None:
    tree = ast.parse(CALLABLE_CONTRACT_PATH.read_text())
    contract_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CallableContract"
    )
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "contract_cache_identity"
        for node in contract_class.body
    )

    method = _method_node(
        CALLABLE_CONTRACT_PATH,
        "CallableContract",
        "resolve_runtime_callable",
    )
    attribute_calls = tuple(
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_declared_callable"
    )
    resolver_attribute_calls = tuple(
        node.func.attr
        for node in ast.walk(resolver)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    )

    assert attribute_calls.count("resolve") == 0
    assert resolver_attribute_calls.count("resolve") == 1
    assert attribute_calls.count("executable_callable") == 1
    assert not set(attribute_calls).intersection(
        {"rehydrate_reference", "supports", "runtime_callable_factory"}
    )
    assert not any(
        isinstance(
            node,
            (
                ast.For,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )
        for node in ast.walk(method)
    )
    assert not any(
        isinstance(node, ast.Name) and "cache" in node.id.lower()
        for node in ast.walk(method)
    )
