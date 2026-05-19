"""Shared substrate for processor method strategy families."""

from __future__ import annotations

from abc import ABC
from typing import ClassVar, Self


PROCESSOR_METHOD_REGISTRY_KEY = "method"


class RegisteredProcessorMethodStrategy(ABC):
    """Mixin for AutoRegisterMeta roots keyed by public processor method names."""

    __registry_key__ = PROCESSOR_METHOD_REGISTRY_KEY
    __skip_if_no_key__ = True
    method: ClassVar[str | None] = None

    @classmethod
    def for_method(cls, method: str) -> Self:
        try:
            return cls.__registry__[method]()
        except KeyError as exc:
            choices = ", ".join(str(key) for key in cls.__registry__)
            raise ValueError(f"{cls.__name__} must be one of: {choices}") from exc
