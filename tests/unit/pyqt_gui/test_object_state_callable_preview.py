from dataclasses import dataclass

import pytest

from openhcs.agent.dto.ui_bridge import UiObjectStateScopeListRequest
from objectstate.object_state import ObjectState, ObjectStateRegistry
from openhcs.pyqt_gui.services.ui_bridge_object_state import (
    ObjectStateScopeProjectionService,
)


def preview_test_callable(image, sigma: float = 1.0):
    return image


@dataclass
class CallablePreviewHolder:
    bare_func: object = preview_test_callable
    tuple_func: object = (preview_test_callable, {"sigma": 2.0})


@pytest.fixture(autouse=True)
def clear_object_state_registry():
    ObjectStateRegistry.clear()
    yield
    ObjectStateRegistry.clear()


def test_object_state_callable_previews_use_import_paths():
    state = ObjectState(CallablePreviewHolder(), scope_id="callables")
    ObjectStateRegistry.register(state, _skip_snapshot=True)

    catalog = ObjectStateScopeProjectionService().catalog(
        UiObjectStateScopeListRequest(
            include_fields=True,
            include_field_values=True,
            field_paths=("bare_func", "tuple_func"),
        )
    )

    scope = catalog.scopes[0]
    fields = {field.address.field_path: field for field in scope.fields}
    expected_path = f"{__name__}.preview_test_callable"

    assert fields["bare_func"].raw_value_preview is not None
    assert fields["bare_func"].raw_value_preview.text == expected_path
    assert fields["tuple_func"].raw_value_preview is not None
    assert expected_path in fields["tuple_func"].raw_value_preview.text
    assert "<function preview_test_callable" not in fields["tuple_func"].raw_value_preview.text
