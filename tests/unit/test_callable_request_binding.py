"""Tests for callable request-record public signature binding."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

from openhcs.core.callable_contract import (
    CallableContract,
    CallableRequestBinding,
    callable_request,
)


@dataclass(frozen=True, slots=True)
class ExampleRequest:
    image: object
    scale: int
    label: str = "default"


def test_callable_request_expands_public_signature_and_invokes_request() -> None:
    @callable_request(
        ExampleRequest,
        public_defaults={"scale": 2},
    )
    def process(request: ExampleRequest, *, enabled: bool = True) -> tuple[object, ...]:
        return request.image, request.scale, request.label, enabled

    signature = inspect.signature(process)

    assert tuple(signature.parameters) == ("image", "scale", "label", "enabled")
    assert signature.parameters["scale"].default == 2
    assert signature.parameters["label"].default == "default"
    assert process("img", enabled=False) == ("img", 2, "default", False)


def test_callable_request_metadata_is_visible_to_callable_contract() -> None:
    @callable_request(ExampleRequest, public_defaults={"scale": 3})
    def process(request: ExampleRequest) -> object:
        return request

    contract = CallableContract.from_callable(process)

    assert isinstance(contract.request_binding, CallableRequestBinding)
    assert contract.request_binding.request_type is ExampleRequest
    assert contract.request_binding.public_fields == ("image", "scale", "label")


def test_callable_request_rejects_missing_request_parameter() -> None:
    with pytest.raises(ValueError, match="must declare request parameter"):

        @callable_request(ExampleRequest)
        def process(image: object) -> object:
            return image


def test_callable_request_rejects_unknown_public_field() -> None:
    with pytest.raises(ValueError, match="has no request field"):
        callable_request(ExampleRequest, public_fields=("image", "missing"))
