from dataclasses import dataclass

import pytest

from openhcs.core.runtime_invocation import RuntimeInvocationOptions
from openhcs.ui.shared.pattern_data_manager import PatternDataManager


def _identity(image):
    return image


@dataclass(frozen=True, slots=True, kw_only=True)
class ExampleInvocationOptions(RuntimeInvocationOptions):
    mode: str


def test_extract_func_and_kwargs_ignores_runtime_invocation_options():
    kwargs = {"sigma": 2}
    options = ExampleInvocationOptions(mode="once")

    func, extracted_kwargs = PatternDataManager.extract_func_and_kwargs(
        (_identity, kwargs, options)
    )

    assert func is _identity
    assert extracted_kwargs is kwargs
    assert "runtime_invocation_options" not in extracted_kwargs


def test_extract_func_and_kwargs_rejects_unknown_tuple_metadata():
    with pytest.raises(TypeError, match="RuntimeInvocationOptions"):
        PatternDataManager.extract_func_and_kwargs((_identity, {}, object()))
