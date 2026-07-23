"""Nominal helpers for experimental-analysis workbook formats."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import pandas as pd


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


FeatureReader = Callable[[pd.DataFrame], object]
PlateDictBuilder = Callable[[pd.DataFrame], object]
PlateDictFiller = Callable[[pd.DataFrame, object, object], object]


@dataclass(frozen=True, slots=True)
class ExperimentalAnalysisFeatureReaders:
    """Feature extractors keyed by the stable experimental-analysis scope order."""

    cx5: FeatureReader
    metaxpress: FeatureReader

    def for_scope(self, scope: "ExperimentalAnalysisScope") -> FeatureReader:
        return (self.cx5, self.metaxpress)[scope.dispatch_index]


@dataclass(frozen=True, slots=True)
class ExperimentalAnalysisPlateHandlers:
    """Plate dictionary operations keyed by the stable scope order."""

    cx5_builder: PlateDictBuilder
    metaxpress_builder: PlateDictBuilder
    cx5_filler: PlateDictFiller
    metaxpress_filler: PlateDictFiller

    def builder_for(self, scope: "ExperimentalAnalysisScope") -> PlateDictBuilder:
        return (self.cx5_builder, self.metaxpress_builder)[scope.dispatch_index]

    def filler_for(self, scope: "ExperimentalAnalysisScope") -> PlateDictFiller:
        return (self.cx5_filler, self.metaxpress_filler)[scope.dispatch_index]


class ExperimentalAnalysisScope(Enum):
    """Supported EDDU experimental-analysis result scopes."""

    CX5 = ("EDDU_CX5", "Rawdata", 0)
    METAXPRESS = ("EDDU_metaxpress", None, 1)

    def __new__(
        cls,
        scope_value: str,
        result_sheet_name: str | None,
        dispatch_index: int,
    ) -> "ExperimentalAnalysisScope":
        member = object.__new__(cls)
        member._value_ = scope_value
        member.result_sheet_name = result_sheet_name
        member.dispatch_index = dispatch_index
        return member

    @classmethod
    def coerce(cls, scope: object) -> "ExperimentalAnalysisScope":
        for member in cls:
            if scope == member.value:
                return member
        raise ValueError(f"microscope {scope} not known")

    def sheet_name_for(self, xls: pd.ExcelFile) -> str:
        if self.result_sheet_name is not None:
            return self.result_sheet_name
        return xls.sheet_names[0]

    def read_results(self, xls: pd.ExcelFile) -> pd.DataFrame:
        return pd.read_excel(xls, self.sheet_name_for(xls))

    def features(
        self,
        raw_df: pd.DataFrame,
        readers: ExperimentalAnalysisFeatureReaders,
    ) -> object:
        return readers.for_scope(self)(raw_df)

    def create_plates_dict(
        self,
        raw_df: pd.DataFrame,
        handlers: ExperimentalAnalysisPlateHandlers,
    ) -> object:
        return handlers.builder_for(self)(raw_df)

    def fill_plates_dict(
        self,
        raw_df: pd.DataFrame,
        plates_dict: object,
        features: object,
        handlers: ExperimentalAnalysisPlateHandlers,
    ) -> object:
        return handlers.filler_for(self)(raw_df, plates_dict, features)
