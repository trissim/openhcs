"""Public API tests for declaration-owned CellProfiler callables."""

from __future__ import annotations

import importlib
import inspect
import pickle
import pytest

from openhcs.core.callable_contract import CallableContract
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule


def _synthetic_unowned_exclusion(image, hidden=None):
    del hidden
    return image


def _synthetic_owned_exclusion(image, labels=None):
    del labels
    return image


def test_backend_package_returns_underlying_declared_callable() -> None:
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    module_type = CellProfilerModule.require_module("Crop")
    declared = module_type.require_callable("crop")
    implementation_module = importlib.import_module(module_type.__module__)

    assert cellprofiler_backend.crop is declared
    assert declared is vars(implementation_module)["crop"]
    assert declared.__module__ == module_type.__module__


def test_submodule_import_does_not_replace_backend_callable() -> None:
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    crop = cellprofiler_backend.crop
    importlib.import_module("openhcs.processing.backends.cellprofiler.crop")

    assert cellprofiler_backend.crop is crop
    assert CellProfilerModule.for_function_name("crop").require_callable("crop") is crop


def test_declared_callable_survives_generic_pickle_by_identity() -> None:
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    crop = cellprofiler_backend.crop

    assert pickle.loads(pickle.dumps(crop)) is crop


def test_callable_lookup_and_contract_read_leave_function_namespace_unchanged() -> None:
    module_type = CellProfilerModule.require_module("Threshold")
    func = module_type.require_callable("threshold")
    before = dict(vars(func))

    assert CellProfilerModule.for_function_name("threshold") is module_type
    CallableContract.from_callable(func)
    module_type.require_callable("threshold")

    assert vars(func) == before


def test_backend_directory_is_derived_from_nominal_module_registry() -> None:
    import openhcs.processing.backends.cellprofiler as cellprofiler_backend

    declared_names = {
        function_name
        for module_type in CellProfilerModule.__registry__.values()
        for function_name in module_type.declared_function_names()
    }

    assert declared_names <= set(dir(cellprofiler_backend))
    assert all(callable(getattr(cellprofiler_backend, name)) for name in declared_names)


def test_every_declared_callable_has_one_exact_nominal_owner() -> None:
    owners: dict[str, type[CellProfilerModule]] = {}
    for module_type in CellProfilerModule.__registry__.values():
        for function_name in module_type.declared_function_names():
            assert function_name not in owners
            owners[function_name] = module_type
            func = module_type.require_callable(function_name)
            assert callable(func)
            assert inspect.getmodule(func).__name__ == module_type.__module__
            assert CellProfilerModule.for_function_name(function_name) is module_type


def test_duplicate_function_ownership_fails_during_module_declaration() -> None:
    existing_owner = CellProfilerModule.require_module("Crop")

    with pytest.raises(
        ValueError,
        match=r"duplicates CellProfiler function names.*CropModule.*crop",
    ):

        class DuplicateCropFunctionModule(CellProfilerModule):
            module_name = "SyntheticDuplicateCropFunction"
            function_name = existing_owner.function_name


def test_require_callable_rejects_excluded_parameter_without_nominal_owner() -> None:
    from python_introspect import set_parameter_exclusions

    set_parameter_exclusions(_synthetic_unowned_exclusion, ("hidden",))

    class SyntheticUnownedExclusionModule(CellProfilerModule):
        module_name = None

    SyntheticUnownedExclusionModule.module_name = "SyntheticUnownedExclusion"
    SyntheticUnownedExclusionModule.function_name = (
        _synthetic_unowned_exclusion.__name__
    )

    with pytest.raises(
        ValueError,
        match=r"SyntheticUnownedExclusion.*synthetic_unowned_exclusion.*hidden",
    ):
        SyntheticUnownedExclusionModule.require_callable()


def test_require_callable_accepts_exclusion_owned_by_generic_special_input() -> None:
    from python_introspect import set_parameter_exclusions

    from openhcs.core.pipeline.function_contracts import special_inputs

    special_inputs("labels")(_synthetic_owned_exclusion)
    set_parameter_exclusions(_synthetic_owned_exclusion, ("labels",))

    class SyntheticOwnedExclusionModule(CellProfilerModule):
        module_name = None

    SyntheticOwnedExclusionModule.module_name = "SyntheticOwnedExclusion"
    SyntheticOwnedExclusionModule.function_name = _synthetic_owned_exclusion.__name__

    assert (
        SyntheticOwnedExclusionModule.require_callable()
        is _synthetic_owned_exclusion
    )


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    (
        ("DisplayDataOnImage", "display_data_on_image"),
        ("DisplayDensityPlot", "display_density_plot"),
        ("DisplayHistogram", "display_histogram"),
        ("DisplayPlatemap", "display_platemap"),
        ("DisplayScatterPlot", "display_scatter_plot"),
    ),
)
def test_display_module_declaration_and_callable_are_colocated(
    module_name: str,
    function_name: str,
) -> None:
    module_type = CellProfilerModule.require_module(module_name)
    implementation = module_type.require_callable(function_name)

    assert module_type.__module__ == (
        "openhcs.processing.backends.cellprofiler.display_modules"
    )
    assert implementation.__module__ == module_type.__module__


@pytest.mark.parametrize(
    "module_name",
    ("LoadData", "LabelImages", "CreateBatchFiles", "SaveCroppedObjects"),
)
def test_unsupported_pass_through_modules_have_no_declaration(
    module_name: str,
) -> None:
    assert CellProfilerModule.for_module(module_name) is None
    with pytest.raises(KeyError, match="No CellProfiler module declaration"):
        CellProfilerModule.require_module(module_name)
