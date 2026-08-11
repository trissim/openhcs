"""Nominal helpers for experimental-analysis workbook formats."""

from __future__ import annotations

from enum import Enum


class ExperimentalLayoutRowRole:
    """Classify layout row labels used by Metaxpress/CX5 analysis workbooks."""

    def __init__(self, row_name: object) -> None:
        self.normalized = str(row_name).lower().replace("_", "").replace(" ", "")

    @property
    def is_replicate_count(self) -> bool:
        return self.normalized in {"n", "ns", "replicate", "replicates"}

    @property
    def is_well_all_replicates(self) -> bool:
        return self.normalized in {"well", "wells"}

    @property
    def is_well_specific_replicate(self) -> bool:
        return self.specific_replicate is not None

    @property
    def specific_replicate(self) -> int | None:
        for prefix in ("wells", "well"):
            if not self.normalized.startswith(prefix):
                continue
            suffix = self.normalized[len(prefix) :]
            if suffix.isdigit():
                return int(suffix)
        return None


class ExperimentalAnalysisScope(Enum):
    """Stable workbook declarations for supported experimental result formats."""

    CX5 = "EDDU_CX5"
    METAXPRESS = "EDDU_metaxpress"

    @classmethod
    def coerce(cls, scope: object) -> "ExperimentalAnalysisScope":
        for member in cls:
            if scope == member or scope == member.value:
                return member
        raise ValueError(f"microscope {scope} not known")
