"""Nominal row-name classification for experimental-analysis plate layouts."""

from __future__ import annotations


class ExperimentalLayoutRowRole:
    """Classify layout row labels used by Metaxpress/CX5 analysis workbooks."""

    def __init__(self, row_name: object) -> None:
        self.normalized = str(row_name).lower()

    @property
    def is_replicate_count(self) -> bool:
        return self.normalized in {"n", "ns", "replicate", "replicates"}

    @property
    def is_well_all_replicates(self) -> bool:
        return self.normalized in {"well", "wells"}

    @property
    def is_well_specific_replicate(self) -> bool:
        return "well" in self.normalized and self.normalized[-1:].isdigit()
