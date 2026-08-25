from __future__ import annotations

import json

import pytest

from openhcs.utils import import_utils
from openhcs.utils.import_utils import (
    create_placeholder_class,
    optional_import_or_none,
    optional_import_placeholder,
)


def test_optional_import_absence_semantics_are_explicit() -> None:
    missing_name = "openhcs_test_dependency_that_does_not_exist"

    assert optional_import_or_none(missing_name) is None

    placeholder = optional_import_placeholder(missing_name)
    assert not placeholder
    assert not placeholder.array.dtype
    with pytest.raises(ImportError, match=missing_name):
        placeholder()


def test_optional_imports_return_installed_module_identity() -> None:
    assert optional_import_or_none("json") is json
    assert optional_import_placeholder("json") is json


def test_optional_import_propagates_broken_installed_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_import(module_name: str):
        raise AttributeError(f"{module_name} imported but is incompatible")

    monkeypatch.setattr(
        import_utils,
        "import_module_preserving_root_logging",
        broken_import,
    )

    with pytest.raises(AttributeError, match="incompatible"):
        optional_import_or_none("broken_dependency")


def test_placeholder_class_preserves_available_base_or_fails_on_use() -> None:
    class AvailableBase:
        pass

    assert create_placeholder_class("Available", AvailableBase) is AvailableBase

    unavailable = create_placeholder_class("Unavailable", required_library="MissingLib")
    instance = unavailable()
    with pytest.raises(ImportError, match="MissingLib"):
        _ = instance.operation
