"""Typed rehydration hooks for compiler function references."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from metaclass_registry import AutoRegisterMeta

from openhcs.core.callable_contract import CallableContract


@dataclass(frozen=True, slots=True)
class FunctionReferenceRehydrationRequest:
    """Runtime data needed to rebuild a callable from preserved compile metadata."""

    reference: Any
    contract: CallableContract
    resolved_callable: Callable[..., Any]


class FunctionReferenceRehydrator(ABC, metaclass=AutoRegisterMeta):
    """Domain extension point for references whose runtime callable is contextual."""

    __registry_key__ = "rehydrator_key"
    __skip_if_no_key__ = True

    rehydrator_key: ClassVar[str | None] = None

    @abstractmethod
    def supports(self, request: FunctionReferenceRehydrationRequest) -> bool:
        """Return whether this rehydrator owns the reference contract."""

    @abstractmethod
    def rehydrate(
        self,
        request: FunctionReferenceRehydrationRequest,
    ) -> Callable[..., Any]:
        """Return the callable that should execute this invocation."""

    @classmethod
    def rehydrate_reference(
        cls,
        request: FunctionReferenceRehydrationRequest,
    ) -> Callable[..., Any]:
        """Apply the first registered contextual rehydrator, or return fallback."""

        for rehydrator_type in cls.__registry__.values():
            rehydrator = rehydrator_type()
            if rehydrator.supports(request):
                return rehydrator.rehydrate(request)
        return request.resolved_callable
