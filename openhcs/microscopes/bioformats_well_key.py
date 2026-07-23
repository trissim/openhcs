"""Nominal Bio-Formats well-key projection authority."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BioFormatsWellKeyAuthority:
    """Convert vendor and OME row/column coordinates to OpenHCS well keys."""

    def key_from_ome(self, row: int | str, column: int | str) -> str:
        row_index = self.ome_row_index(row)
        column_index = self.ome_column_index(column)
        return f"{self.label_from_one_based(row_index + 1)}{column_index + 1:02d}"

    def key_from_one_based(self, row: int, column: int) -> str:
        return f"{self.label_from_one_based(row)}{column:02d}"

    def label_from_one_based(self, row: int) -> str:
        if row < 1:
            raise ValueError(f"Well row must be 1-based, got {row}.")
        letters = []
        current = row
        while current:
            current, remainder = divmod(current - 1, 26)
            letters.append(chr(ord("A") + remainder))
        return "".join(reversed(letters))

    def ome_row_index(self, row: int | str) -> int:
        if isinstance(row, int):
            if row < 0:
                raise ValueError(f"OME well row must be non-negative: {row}")
            return row
        row_text = str(row).strip()
        if row_text.isdecimal():
            return self.ome_row_index(int(row_text))
        if row_text.isalpha():
            index = 0
            for char in row_text.upper():
                index = index * 26 + (ord(char) - ord("A") + 1)
            return index - 1
        raise ValueError(f"Unsupported OME well row value: {row!r}")

    def ome_column_index(self, column: int | str) -> int:
        if isinstance(column, int):
            if column < 0:
                raise ValueError(f"OME well column must be non-negative: {column}")
            return column
        column_text = str(column).strip()
        if not column_text.isdecimal():
            raise ValueError(f"Unsupported OME well column value: {column!r}")
        return self.ome_column_index(int(column_text))


BIOFORMATS_WELL_KEYS = BioFormatsWellKeyAuthority()
