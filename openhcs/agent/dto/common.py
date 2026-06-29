"""Shared DTOs for the headless OpenHCS agent API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Self, TypeAlias


SCHEMA_VERSION = "openhcs.agent.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | tuple["JsonValue", ...] | list["JsonValue"]
JsonObject: TypeAlias = Mapping[str, JsonValue]


@dataclass(frozen=True, slots=True)
class AgentCliArgumentSpec:
    """CLI argument shape declared by agent request DTOs."""

    field_name: str
    flags: tuple[str, ...] = ()
    positional: bool = False
    nargs: str | int | None = None
    action: str | None = None
    help: str | None = None


class AgentCliRequest(ABC):
    """Nominal request DTO that declares generated CLI argument projection."""

    @classmethod
    @abstractmethod
    def from_fields(cls, **kwargs) -> Self:
        raise NotImplementedError

    @classmethod
    def agent_cli_factory(cls):
        return cls.from_fields

    @classmethod
    def agent_cli_argument_specs(cls) -> tuple[AgentCliArgumentSpec, ...]:
        return ()

    @abstractmethod
    def as_tool_arguments(self) -> JsonObject:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class AgentWarning:
    code: str
    message: str
    hint: str | None = None


@dataclass(frozen=True, slots=True)
class AgentError:
    code: str
    message: str
    hint: str | None = None
    exception_type: str | None = None
    path: str | None = None

    @classmethod
    def from_exception(
        cls,
        code: str,
        exception: Exception,
        *,
        hint: str | None = None,
        path: str | None = None,
    ) -> "AgentError":
        return cls(
            code=code,
            message=str(exception),
            hint=hint,
            exception_type=type(exception).__name__,
            path=path,
        )


@dataclass(frozen=True, kw_only=True)
class AgentResultEnvelope:
    schema_version: str
    errors: tuple[AgentError, ...] = ()
    warnings: tuple[AgentWarning, ...] = ()


@dataclass(frozen=True, kw_only=True)
class AgentTimedStatusEnvelope(AgentResultEnvelope):
    status: str
    started_at_unix: float


@dataclass(frozen=True, slots=True)
class AgentResourceRef:
    uri: str
    title: str
    mime_type: str = "application/json"
    path: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None


@dataclass(frozen=True, slots=True)
class RenderedSource:
    schema_version: str
    title: str
    source: str
    mime_type: str = "text/x-python"
