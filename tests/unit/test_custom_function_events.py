"""Custom-function notifications do not acquire subscriber lifetimes."""

from __future__ import annotations

import gc
import weakref

from openhcs.processing.custom_functions.events import (
    CustomFunctionChangedEvent,
    custom_function_changed,
)
from openhcs.processing.custom_functions.signals import CustomFunctionSignals


class _DomainSubscriber:
    def __init__(self, observations: list[str]) -> None:
        self._observations = observations

    def observe(self) -> None:
        self._observations.append("changed")


class _EquivalentSubscriber:
    def __init__(self, observation: str, observations: list[str]) -> None:
        self._observation = observation
        self._observations = observations

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _EquivalentSubscriber)

    def __call__(self) -> None:
        self._observations.append(self._observation)


def test_domain_event_observes_without_owning_subscriber_lifetime() -> None:
    event = CustomFunctionChangedEvent()
    observations: list[str] = []
    subscriber = _DomainSubscriber(observations)
    subscriber_reference = weakref.ref(subscriber)
    event.subscribe(subscriber.observe)

    event.emit()
    assert observations == ["changed"]

    del subscriber
    gc.collect()

    assert subscriber_reference() is None
    event.emit()
    assert observations == ["changed"]


def test_domain_event_subscribes_one_bound_callback_once() -> None:
    event = CustomFunctionChangedEvent()
    observations: list[str] = []
    subscriber = _DomainSubscriber(observations)

    event.subscribe(subscriber.observe)
    event.subscribe(subscriber.observe)
    event.emit()

    assert observations == ["changed"]


def test_domain_event_preserves_distinct_equal_callable_identities() -> None:
    event = CustomFunctionChangedEvent()
    observations: list[str] = []
    first = _EquivalentSubscriber("first", observations)
    second = _EquivalentSubscriber("second", observations)

    event.subscribe(first)
    event.subscribe(second)
    event.emit()

    assert observations == ["first", "second"]


def test_qt_adapter_is_not_retained_by_the_domain_event(qapp) -> None:
    del qapp
    adapter = CustomFunctionSignals()
    adapter_reference = weakref.ref(adapter)

    del adapter
    gc.collect()

    assert adapter_reference() is None
    custom_function_changed.emit()
