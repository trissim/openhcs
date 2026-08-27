"""Every locally available catalog declaration must be editable in the real UI."""

from __future__ import annotations

import time

from PyQt6.QtTest import QTest

from objectstate import ObjectStateRegistry
from openhcs.processing.backends.lib_registry.registry_service import RegistryService
from pyqt_reactive.widgets.function_pane import FunctionPaneWidget


def _wait_for_form_build(qapp, pane: FunctionPaneWidget, *, timeout: float) -> None:
    manager = pane.form_manager
    if manager is None:
        raise AssertionError("Function pane did not create a parameter form manager")

    deadline = time.monotonic() + timeout
    while not (
        manager.form_build_complete
        or manager.form_build_cancelled
        or manager.form_build_failure is not None
    ):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Form did not build within {timeout:g} seconds")
        qapp.processEvents()
        QTest.qWait(1)

    if manager.form_build_failure is not None:
        raise manager.form_build_failure
    if manager.form_build_cancelled:
        raise AssertionError("Form construction was cancelled")


def test_every_catalog_function_materializes_in_function_pane(qapp) -> None:
    """Exercise the same full-catalog form surface used by Function Selector."""

    metadata_by_key = RegistryService.get_all_functions_with_metadata()
    assert metadata_by_key

    failures: list[str] = []
    ObjectStateRegistry.clear()

    for composite_key, metadata in sorted(metadata_by_key.items()):
        pane = None
        try:
            pane = FunctionPaneWidget(
                (metadata.func, {}),
                0,
                None,
                scope_id=f"catalog-form::{composite_key}",
                func_scope_prefix=f"catalog-form::{composite_key}",
                func_scope_token="function-0",
            )
            pane.show()
            _wait_for_form_build(qapp, pane, timeout=5.0)
        except Exception as exc:
            failures.append(
                f"{composite_key} ({metadata.func.__module__}."
                f"{metadata.func.__qualname__}): {type(exc).__name__}: {exc}"
            )
        finally:
            if pane is not None:
                if pane.form_manager is not None:
                    pane.form_manager.dispose()
                pane.close()
                pane.deleteLater()
            qapp.processEvents()
            ObjectStateRegistry.clear()

    assert failures == [], "Catalog functions with unrenderable forms:\n" + "\n".join(
        failures
    )
