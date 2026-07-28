"""Shared DTOs for the headless OpenHCS agent API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cache
from typing import Self, get_type_hints

from openhcs.serialization.json import (
    JsonObject,
    JsonScalar as JsonScalar,
    JsonValue as JsonValue,
)

SCHEMA_VERSION = "openhcs.agent.v1"
AGENT_PARAMETER_DESCRIPTION_METADATA_KEY = "agent_parameter_description"
AGENT_PARAMETER_PRODUCER_OUTPUT_CONTRACT_METADATA_KEY = (
    "agent_parameter_producer_output_contract"
)


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
    def code_from_serialized_mapping(cls, value: Mapping[str, object]) -> str | None:
        """Project the error code through this DTO's declared field descriptor."""

        code = value.get(cls.code.__name__)
        return code if isinstance(code, str) else None

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

    @classmethod
    @cache
    def serialized_errors_field_name(cls) -> str:
        """Derive the error-list wire field from the owning dataclass declaration."""

        type_hints = get_type_hints(cls)
        matching_fields = tuple(
            field.name
            for field in fields(cls)
            if type_hints[field.name] == tuple[AgentError, ...]
        )
        if len(matching_fields) != 1:
            raise TypeError(
                f"{cls.__name__} must declare exactly one AgentError tuple field."
            )
        return matching_fields[0]

    @classmethod
    def error_items_from_serialized_mapping(
        cls,
        value: Mapping[str, object],
    ) -> list[object] | None:
        """Return the serialized error list through the declared envelope field."""

        errors = value.get(cls.serialized_errors_field_name())
        return errors if isinstance(errors, list) else None


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
