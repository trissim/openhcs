"""Transaction and concurrency boundaries for persisted custom functions."""

from __future__ import annotations

import concurrent.futures
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from arraybridge import MemoryType

import openhcs.processing.custom_functions as custom_functions
import openhcs.processing.custom_functions.manager as manager_module
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from openhcs.core.function_reference import FunctionReferenceTransportAuthority
from openhcs.processing.custom_functions.manager import CustomFunctionManager
from openhcs.processing.custom_functions.runtime_registry import (
    CustomFunctionRuntimeRegistry,
)
from openhcs.processing.custom_functions.templates import AVAILABLE_MEMORY_TYPES
from openhcs.processing.custom_functions.validation import ValidationError


def _source(name: str, expression: str = "image") -> str:
    return f"@numpy\ndef {name}(image):\n    return {expression}\n"


@pytest.fixture
def isolated_custom_runtime(monkeypatch, tmp_path):
    storage_dir = tmp_path / "custom_functions"
    storage_dir.mkdir()
    monkeypatch.setattr(manager_module, "get_data_file_path", lambda _name: storage_dir)
    monkeypatch.setattr(CustomFunctionRuntimeRegistry, "_metadata_by_name", {})
    monkeypatch.setattr(CustomFunctionRuntimeRegistry, "_published_exports", {})
    monkeypatch.setattr(CustomFunctionRuntimeRegistry, "_preparation_outcomes", {})
    monkeypatch.setattr(CustomFunctionRuntimeRegistry, "_preparation_threads", {})
    monkeypatch.setattr(CustomFunctionRuntimeRegistry, "_source_revision", None)
    yield storage_dir
    CustomFunctionRuntimeRegistry.clear()


def test_register_rejects_multi_declaration_source_without_partial_publication(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()

    with pytest.raises(ValidationError, match="exactly one"):
        manager.register_from_code(
            _source("first_probe") + "\n" + _source("second_probe"),
        )

    assert CustomFunctionRuntimeRegistry.metadata_by_name() == {}
    assert not tuple(isolated_custom_runtime.glob("*.py"))
    assert "first_probe" not in vars(custom_functions)


def test_invalid_update_preserves_file_and_runtime_identity(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()
    [original] = manager.register_from_code(_source("stable_probe"))
    source_path = isolated_custom_runtime / "stable_probe.py"
    original_source = source_path.read_text(encoding="utf-8")

    with pytest.raises(ValidationError):
        manager.update_custom_function(
            "stable_probe",
            "@numpy\ndef broken_probe(value):\n    return value\n",
        )

    assert source_path.read_text(encoding="utf-8") == original_source
    assert vars(custom_functions)["stable_probe"] is original
    assert (
        CustomFunctionRuntimeRegistry.metadata_by_name()["stable_probe"].func
        is original
    )


def test_rename_reconciles_file_runtime_and_public_export(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()
    manager.register_from_code(_source("old_probe"))

    new_name = manager.update_custom_function(
        "old_probe", _source("new_probe", "image + 1")
    )

    assert new_name == "new_probe"
    assert not (isolated_custom_runtime / "old_probe.py").exists()
    assert (isolated_custom_runtime / "new_probe.py").exists()
    assert "old_probe" not in vars(custom_functions)
    assert "old_probe" not in CustomFunctionRuntimeRegistry.metadata_by_name()
    assert np.array_equal(vars(custom_functions)["new_probe"](np.asarray([[1]])), [[2]])


def test_rename_rejects_existing_runtime_target_without_mutation(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()
    [old_callable] = manager.register_from_code(_source("old_probe"))
    [target_callable] = manager.register_from_code(
        _source("target_probe"), persist=False
    )
    old_source = (isolated_custom_runtime / "old_probe.py").read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        manager.update_custom_function("old_probe", _source("target_probe"))

    assert (isolated_custom_runtime / "old_probe.py").read_text(
        encoding="utf-8"
    ) == old_source
    assert vars(custom_functions)["old_probe"] is old_callable
    assert vars(custom_functions)["target_probe"] is target_callable


def test_failed_bulk_reconciliation_preserves_last_proven_projection(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()
    (isolated_custom_runtime / "stable_probe.py").write_text(
        _source("stable_probe"), encoding="utf-8"
    )
    assert manager.load_all_custom_functions() == 1
    prior_revision = CustomFunctionRuntimeRegistry.source_revision()
    prior_callable = vars(custom_functions)["stable_probe"]
    (isolated_custom_runtime / "broken_probe.py").write_text(
        "@numpy\ndef broken_probe(value):\n    return value\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        manager.load_all_custom_functions()

    assert CustomFunctionRuntimeRegistry.source_revision() == prior_revision
    assert vars(custom_functions)["stable_probe"] is prior_callable
    assert "broken_probe" not in CustomFunctionRuntimeRegistry.metadata_by_name()

    (isolated_custom_runtime / "broken_probe.py").write_text(
        _source("broken_probe"), encoding="utf-8"
    )
    assert manager.load_all_custom_functions() == 2
    assert CustomFunctionRuntimeRegistry.source_revision() == manager.source_revision()


def test_concurrent_lazy_imports_publish_one_callable_identity(
    isolated_custom_runtime,
) -> None:
    function_name = "concurrent_probe"
    (isolated_custom_runtime / f"{function_name}.py").write_text(
        _source(function_name), encoding="utf-8"
    )
    vars(custom_functions).pop(function_name, None)
    start = threading.Barrier(12)

    def import_one():
        start.wait(timeout=5)
        return getattr(custom_functions, function_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        callables = tuple(executor.map(lambda _index: import_one(), range(12)))

    assert len({id(func) for func in callables}) == 1
    assert vars(custom_functions)[function_name] is callables[0]


def test_concurrent_failed_loads_share_one_exact_source_outcome(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    function_name = "concurrent_failure_probe"
    (isolated_custom_runtime / f"{function_name}.py").write_text(
        _source(function_name), encoding="utf-8"
    )
    prepare_calls = 0
    prepare_lock = threading.Lock()

    def failing_prepare(self, code):
        del self, code
        nonlocal prepare_calls
        with prepare_lock:
            prepare_calls += 1
        raise ValidationError("shared source failure")

    monkeypatch.setattr(CustomFunctionManager, "_prepare_source", failing_prepare)
    start = threading.Barrier(8)

    def load_one():
        start.wait(timeout=5)
        return CustomFunctionManager().load_custom_function(function_name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = tuple(executor.submit(load_one) for _index in range(8))
        for future in futures:
            with pytest.raises(ValidationError, match="shared source failure"):
                future.result(timeout=5)

    assert prepare_calls == 1
    assert function_name not in CustomFunctionRuntimeRegistry.metadata_by_name()


def test_delete_linearizes_after_inflight_lazy_load(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    function_name = "delete_race_probe"
    (isolated_custom_runtime / f"{function_name}.py").write_text(
        _source(function_name), encoding="utf-8"
    )
    vars(custom_functions).pop(function_name, None)
    entered = threading.Event()
    release = threading.Event()
    original_prepare = CustomFunctionManager._prepare_source

    def blocking_prepare(self, code):
        entered.set()
        assert release.wait(timeout=5)
        return original_prepare(self, code)

    monkeypatch.setattr(CustomFunctionManager, "_prepare_source", blocking_prepare)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        load_future = executor.submit(getattr, custom_functions, function_name)
        assert entered.wait(timeout=5)
        delete_future = executor.submit(
            CustomFunctionManager().delete_custom_function, function_name
        )
        assert delete_future.result(timeout=5)
        release.set()
        with pytest.raises(ValidationError, match="changed during preparation"):
            load_future.result(timeout=5)

    assert not (isolated_custom_runtime / f"{function_name}.py").exists()
    assert function_name not in vars(custom_functions)
    assert function_name not in CustomFunctionRuntimeRegistry.metadata_by_name()


def test_bulk_reconciliation_rejects_revision_drift_before_publication(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    function_name = "revision_race_probe"
    source_path = isolated_custom_runtime / f"{function_name}.py"
    source_path.write_text(
        _source(function_name, "image + 1"),
        encoding="utf-8",
    )
    entered = threading.Event()
    release = threading.Event()
    original_prepare = CustomFunctionManager._prepare_source

    def blocking_prepare(self, code):
        entered.set()
        assert release.wait(timeout=5)
        return original_prepare(self, code)

    monkeypatch.setattr(CustomFunctionManager, "_prepare_source", blocking_prepare)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        load_future = executor.submit(CustomFunctionManager().load_all_custom_functions)
        assert entered.wait(timeout=5)
        source_path.write_text(
            _source(function_name, "image + 2"),
            encoding="utf-8",
        )
        release.set()
        with pytest.raises(ValidationError, match="changed during preparation"):
            load_future.result(timeout=5)

    assert CustomFunctionRuntimeRegistry.metadata_by_name() == {}
    assert CustomFunctionRuntimeRegistry.source_revision() is None
    monkeypatch.setattr(CustomFunctionManager, "_prepare_source", original_prepare)
    assert CustomFunctionManager().load_all_custom_functions() == 1
    assert np.array_equal(
        vars(custom_functions)[function_name](np.asarray([[1]])),
        [[3]],
    )


def test_source_preparation_does_not_hold_lifecycle_lock(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    function_name = "reentrant_prepare_probe"
    source_path = isolated_custom_runtime / f"{function_name}.py"
    source_path.write_text(_source(function_name), encoding="utf-8")
    original_prepare = CustomFunctionManager._prepare_source
    worker_threads = []
    delete_completed_during_prepare = []

    def reentrant_prepare(self, code):
        worker = threading.Thread(
            target=CustomFunctionManager().delete_custom_function,
            args=(function_name,),
        )
        worker_threads.append(worker)
        worker.start()
        worker.join(timeout=1)
        delete_completed_during_prepare.append(not worker.is_alive())
        return original_prepare(self, code)

    monkeypatch.setattr(CustomFunctionManager, "_prepare_source", reentrant_prepare)
    with pytest.raises(ValidationError, match="changed during preparation"):
        CustomFunctionManager().load_custom_function(function_name)
    for worker in worker_threads:
        worker.join(timeout=5)

    assert delete_completed_during_prepare == [True]
    assert not source_path.exists()
    assert function_name not in CustomFunctionRuntimeRegistry.metadata_by_name()


def test_public_package_api_name_collision_preserves_original_owner(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()
    original_manager_class = vars(custom_functions)["CustomFunctionManager"]

    with pytest.raises(ValueError, match="public package export"):
        manager.register_from_code(
            _source("CustomFunctionManager"),
        )

    assert vars(custom_functions)["CustomFunctionManager"] is original_manager_class
    assert (
        "CustomFunctionManager" not in CustomFunctionRuntimeRegistry.metadata_by_name()
    )
    assert not (isolated_custom_runtime / "CustomFunctionManager.py").exists()


def test_removal_preserves_export_that_displaced_published_callable(
    isolated_custom_runtime,
) -> None:
    function_name = "displaced_export_probe"
    manager = CustomFunctionManager()
    manager.register_from_code(_source(function_name))
    replacement_owner = object()
    setattr(custom_functions, function_name, replacement_owner)

    assert manager.delete_custom_function(function_name)

    assert vars(custom_functions)[function_name] is replacement_owner
    assert function_name not in CustomFunctionRuntimeRegistry.metadata_by_name()
    vars(custom_functions).pop(function_name)


def test_cold_registration_never_prepares_global_catalog(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    monkeypatch.setattr(RegistryService, "_metadata_cache", None)
    monkeypatch.setattr(
        RegistryService,
        "get_all_functions_with_metadata",
        classmethod(
            lambda cls, **kwargs: pytest.fail("cold registration prepared catalog")
        ),
    )

    [registered] = CustomFunctionManager().register_from_code(
        _source("cold_probe"), persist=False
    )

    assert registered is vars(custom_functions)["cold_probe"]


def test_custom_function_framework_surfaces_derive_from_memory_type_owner(
    isolated_custom_runtime,
) -> None:
    manager = CustomFunctionManager()
    declared_names = tuple(memory_type.value for memory_type in MemoryType)

    assert AVAILABLE_MEMORY_TYPES == declared_names
    assert set(manager._create_execution_namespace()) == {
        "__name__",
        *declared_names,
    }


def test_source_helper_is_not_misidentified_as_processing_declaration(
    isolated_custom_runtime,
) -> None:
    source = (
        "def helper(image):\n"
        "    return image + 1\n\n"
        "@numpy\n"
        "def processing_probe(image):\n"
        "    return helper(image)\n"
    )
    source_path = isolated_custom_runtime / "processing_probe.py"
    source_path.write_text(source, encoding="utf-8")

    info = CustomFunctionManager().list_custom_functions()

    assert [(item.name, item.memory_type) for item in info] == [
        ("processing_probe", MemoryType.NUMPY.value)
    ]
    assert CustomFunctionRuntimeRegistry.metadata_by_name() == {}


def test_registration_rejects_name_claim_proven_by_cached_canonical_catalog(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        RegistryService,
        "_metadata_cache",
        {
            "openhcs:cellprofiler_crop": SimpleNamespace(
                tags=("openhcs", "cellprofiler")
            )
        },
    )
    monkeypatch.setattr(
        RegistryService,
        "get_all_functions_with_metadata",
        classmethod(
            lambda cls, **kwargs: pytest.fail("collision check prepared catalog")
        ),
    )

    with pytest.raises(ValueError, match="canonical OpenHCS function"):
        CustomFunctionManager().register_from_code(_source("cellprofiler_crop"))

    assert not (isolated_custom_runtime / "cellprofiler_crop.py").exists()
    assert "cellprofiler_crop" not in CustomFunctionRuntimeRegistry.metadata_by_name()


def test_compiled_custom_reference_rejects_changed_source_revision(
    isolated_custom_runtime,
    monkeypatch,
) -> None:
    manager = CustomFunctionManager()
    [original] = manager.register_from_code(_source("revision_contract_probe"))
    reference = FunctionReferenceTransportAuthority.function_reference(original)

    manager.update_custom_function(
        "revision_contract_probe",
        "@cupy\ndef revision_contract_probe(image):\n    return image\n",
    )

    with pytest.raises(RuntimeError, match="changed after this reference was compiled"):
        reference.resolve()

    monkeypatch.setattr(
        RegistryService,
        "get_all_functions_with_metadata",
        classmethod(
            lambda cls, **kwargs: pytest.fail(
                "current custom declaration fell back to global catalog discovery"
            )
        ),
    )
    current = vars(custom_functions)["revision_contract_probe"]
    assert (
        FunctionReferenceTransportAuthority.function_reference(current).resolve()
        is current
    )
