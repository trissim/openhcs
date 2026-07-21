from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from openhcs.serialization.json import to_jsonable


class ThresholdMode(Enum):
    OTSU = "otsu"


@dataclass(frozen=True)
class ImmutableMappingPayload:
    values: Mapping[str, object]


def sample_agent_callable(image, threshold: float = 0.5):
    return image


def test_to_jsonable_projects_callable_identity():
    payload = to_jsonable(sample_agent_callable)

    assert payload == {
        "kind": "callable",
        "name": "sample_agent_callable",
        "module": __name__,
        "qualname": "sample_agent_callable",
        "import_path": f"{__name__}.sample_agent_callable",
    }


def test_to_jsonable_projects_functionstep_callable_tuple():
    payload = to_jsonable(
        (
            sample_agent_callable,
            {
                "threshold": 0.25,
                "mode": ThresholdMode.OTSU,
            },
        )
    )

    assert payload == [
        {
            "kind": "callable",
            "name": "sample_agent_callable",
            "module": __name__,
            "qualname": "sample_agent_callable",
            "import_path": f"{__name__}.sample_agent_callable",
        },
        {
            "threshold": 0.25,
            "mode": "otsu",
        },
    ]


def test_to_jsonable_projects_mapping_proxy_nested_in_dataclass():
    payload = to_jsonable(
        ImmutableMappingPayload(values=MappingProxyType({"threshold": 0.25}))
    )

    assert payload == {"values": {"threshold": 0.25}}
