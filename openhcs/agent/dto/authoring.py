"""Authoring context DTOs for agent integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AuthoringContext:
    schema_version: str
    kind: str
    content: str
