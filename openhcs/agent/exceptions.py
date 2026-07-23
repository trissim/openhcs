"""Typed exceptions for agent-facing OpenHCS service failures."""

from __future__ import annotations

from typing import ClassVar

from openhcs.agent.dto.common import AgentError


class AgentFacingErrorMixin:
    """Mixin for exceptions that can project themselves to stable agent errors."""

    agent_error_code: ClassVar[str]
    agent_error_hint: ClassVar[str | None] = None

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path

    @property
    def message(self) -> str:
        if not self.args:
            return ""
        return str(self.args[0])

    def __str__(self) -> str:
        return self.message

    def to_agent_error(self) -> AgentError:
        return AgentError(
            code=self.agent_error_code,
            message=self.message,
            hint=self.agent_error_hint,
            exception_type=type(self).__name__,
            path=self.path,
        )
