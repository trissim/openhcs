"""Shared measurement schema helpers."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, make_dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DataclassFieldInsertion:
    """Field inserted into a dataclass companion schema."""

    name: str
    annotation: type[Any] | object
    after_field: str | None = None


@dataclass(frozen=True, slots=True)
class DataclassCompanionSchema:
    """Derive a companion dataclass from an authoritative dataclass schema."""

    source_type: type[object]
    companion_name: str
    insertions: tuple[DataclassFieldInsertion, ...]
    module_name: str
    doc: str | None = None

    def materialize(self) -> type[object]:
        row_fields: list[Any] = []
        pending_insertions = list(self.insertions)
        for source_field in fields(self.source_type):
            row_fields.append(self._field_spec(source_field))
            row_fields.extend(
                self._insertion_spec(insertion)
                for insertion in tuple(pending_insertions)
                if insertion.after_field == source_field.name
            )
            pending_insertions = [
                insertion
                for insertion in pending_insertions
                if insertion.after_field != source_field.name
            ]
        row_fields.extend(
            self._insertion_spec(insertion)
            for insertion in pending_insertions
            if insertion.after_field is None
        )
        return make_dataclass(
            self.companion_name,
            row_fields,
            namespace={
                "__module__": self.module_name,
                "__doc__": self.doc,
            },
        )

    @staticmethod
    def _field_spec(source_field: Any) -> Any:
        if source_field.default is not MISSING:
            return (
                source_field.name,
                source_field.type,
                field(default=source_field.default),
            )
        if source_field.default_factory is not MISSING:
            return (
                source_field.name,
                source_field.type,
                field(default_factory=source_field.default_factory),
            )
        return (source_field.name, source_field.type)

    @staticmethod
    def _insertion_spec(insertion: DataclassFieldInsertion) -> tuple[str, object]:
        return (insertion.name, insertion.annotation)
