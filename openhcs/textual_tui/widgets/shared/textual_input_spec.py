"""Authoritative Textual Input construction spec."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from textual.widgets import Input


InputKind = Literal["text", "integer", "number"]


@dataclass(frozen=True)
class TextualInputSpec:
    """Single construction authority for Textual Input widgets."""

    value: Any
    input_type: InputKind
    widget_id: str | None

    @classmethod
    def for_kind(
        cls,
        input_type: InputKind,
        value: Any,
        widget_id: str | None,
    ) -> "TextualInputSpec":
        return cls(value=value, input_type=input_type, widget_id=widget_id)

    def build(self) -> Input:
        return Input(
            value="" if self.value is None else str(self.value),
            type=self.input_type,
            id=self.widget_id,
        )
