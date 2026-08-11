"""Nominal result-format strategies for standalone experimental analysis."""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
import string
from typing import Any

from metaclass_registry import AutoRegisterMeta
import pandas as pd

from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.formats.experimental_layout_rows import ExperimentalAnalysisScope


PlateDictionary = dict[str, dict[str, dict[str, Any]]]


def plate_well_ids() -> tuple[str, ...]:
    """Return canonical 96-well identifiers."""
    return tuple(
        f"{row_name}{column:02d}"
        for row_name in string.ascii_uppercase[:8]
        for column in range(1, 13)
    )


class ExperimentalResultFormatStrategy(
    EnumKeyedStrategyMixin[ExperimentalAnalysisScope],
    metaclass=AutoRegisterMeta,
):
    """Own all result-file behavior for one declared workbook scope."""

    __enum_member_attr__ = "scope"

    @abstractmethod
    def read_results(self, results_path: str | Path) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def features(self, raw_df: pd.DataFrame) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def create_plates_dict(self, raw_df: pd.DataFrame) -> PlateDictionary:
        raise NotImplementedError

    @abstractmethod
    def fill_plates_dict(
        self,
        raw_df: pd.DataFrame,
        plates_dict: PlateDictionary,
        features: list[str],
    ) -> PlateDictionary:
        raise NotImplementedError

    def process(self, results_path: str | Path) -> dict[str, Any]:
        """Read one result file through this strategy's complete contract."""
        raw_df = self.read_results(results_path)
        features = self.features(raw_df)
        plates_dict = self.fill_plates_dict(
            raw_df,
            self.create_plates_dict(raw_df),
            features,
        )
        return {
            "raw_df": raw_df,
            "features": features,
            "plate_names": list(plates_dict),
            "plates_dict": plates_dict,
            "format_name": self.scope.value,
        }

    def empty_wells(self, raw_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        feature_names = self.features(raw_df)
        return {
            well: {feature: None for feature in feature_names}
            for well in plate_well_ids()
        }

    @staticmethod
    def measurement_value(value: Any) -> Any:
        """Project numeric spreadsheet cells into a stable floating domain."""
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


class CX5ExperimentalResultFormat(ExperimentalResultFormatStrategy):
    """ThermoFisher CX5 result-file behavior."""

    scope = ExperimentalAnalysisScope.CX5

    def read_results(self, results_path: str | Path) -> pd.DataFrame:
        return pd.read_excel(results_path, sheet_name="Rawdata")

    def features(self, raw_df: pd.DataFrame) -> list[str]:
        replicate_index = raw_df.columns.str.find("Replicate").argmax()
        return list(raw_df.iloc[:, replicate_index + 1 : -1].columns)

    def create_plates_dict(self, raw_df: pd.DataFrame) -> PlateDictionary:
        return {
            str(plate_id): self.empty_wells(raw_df)
            for plate_id in raw_df["UniquePlateId"].tolist()
        }

    def fill_plates_dict(
        self,
        raw_df: pd.DataFrame,
        plates_dict: PlateDictionary,
        features: list[str],
    ) -> PlateDictionary:
        for _index, row in raw_df.iterrows():
            plate_id = str(row.iloc[1])
            well = f"{chr(int(row.iloc[2]) + 64)}{int(row.iloc[3]):02d}"
            if plate_id not in plates_dict or well not in plates_dict[plate_id]:
                continue
            for feature in features:
                plates_dict[plate_id][well][feature] = self.measurement_value(
                    row[feature]
                )
        return plates_dict


class MetaXpressExperimentalResultFormat(ExperimentalResultFormatStrategy):
    """MetaXpress consolidated CSV and workbook result behavior."""

    scope = ExperimentalAnalysisScope.METAXPRESS

    def read_results(self, results_path: str | Path) -> pd.DataFrame:
        path = Path(results_path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=None)
        workbook = pd.ExcelFile(path)
        return pd.read_excel(workbook, workbook.sheet_names[0])

    @staticmethod
    def well_header_row(raw_df: pd.DataFrame) -> int | None:
        for row_index in range(min(10, len(raw_df))):
            if str(raw_df.iloc[row_index, 0]).strip().lower() == "well":
                return row_index
        return None

    @staticmethod
    def feature_cells(row: pd.Series) -> list[str]:
        return [
            str(cell).strip()
            for cell in row.iloc[1:]
            if pd.notna(cell) and str(cell).strip()
        ]

    def features(self, raw_df: pd.DataFrame) -> list[str]:
        well_header_row = self.well_header_row(raw_df)
        if well_header_row is not None:
            return self.feature_cells(raw_df.iloc[well_header_row])
        try:
            feature_cells = raw_df[pd.isnull(raw_df.iloc[:, 0])].iloc[0].tolist()[2:]
        except (IndexError, KeyError):
            return [
                str(column)
                for column in raw_df.columns[1:]
                if column and str(column) != "nan"
            ]
        return [str(cell) for cell in feature_cells if pd.notna(cell)]

    def create_plates_dict(self, raw_df: pd.DataFrame) -> PlateDictionary:
        plate_id_rows = raw_df[(raw_df == "Plate ID").any(axis=1)]
        plate_name_rows = raw_df[(raw_df == "Plate Name").any(axis=1)]
        if not plate_id_rows.empty:
            plate_names = plate_id_rows.iloc[:, 1].tolist()
        elif not plate_name_rows.empty:
            plate_names = plate_name_rows.iloc[:, 1].tolist()
        else:
            plate_names = ["default_plate"]
        return {str(plate_name): self.empty_wells(raw_df) for plate_name in plate_names}

    def fill_plates_dict(
        self,
        raw_df: pd.DataFrame,
        plates_dict: PlateDictionary,
        features: list[str],
    ) -> PlateDictionary:
        well_header_row = self.well_header_row(raw_df)
        if well_header_row is None:
            return self._fill_excel(raw_df, plates_dict, features)

        row_index = 0
        while row_index < len(raw_df):
            if str(raw_df.iloc[row_index, 0]).strip() != "Barcode":
                row_index += 1
                continue
            plate_id, header_index = self._section_header(raw_df, row_index)
            if plate_id is None or header_index is None:
                row_index += 1
                continue
            section_end = self._next_section(raw_df, header_index + 1)
            section_features = self.feature_cells(raw_df.iloc[header_index])
            for data_index in range(header_index + 1, section_end):
                row = raw_df.iloc[data_index]
                well = str(row.iloc[0]).strip()
                if plate_id not in plates_dict or well not in plates_dict[plate_id]:
                    continue
                for feature_index, feature in enumerate(section_features, start=1):
                    if feature in plates_dict[plate_id][well] and feature_index < len(
                        row
                    ):
                        plates_dict[plate_id][well][feature] = self.measurement_value(
                            row.iloc[feature_index]
                        )
            row_index = section_end
        return plates_dict

    @staticmethod
    def _section_header(
        raw_df: pd.DataFrame,
        start_index: int,
    ) -> tuple[str | None, int | None]:
        plate_id = None
        well_header_index = None
        for row_index in range(start_index, min(start_index + 10, len(raw_df))):
            label = str(raw_df.iloc[row_index, 0]).strip()
            if label == "Plate ID":
                plate_id = str(raw_df.iloc[row_index, 1]).strip()
            if label.lower() == "well":
                well_header_index = row_index
                break
        return plate_id, well_header_index

    @staticmethod
    def _next_section(raw_df: pd.DataFrame, start_index: int) -> int:
        for row_index in range(start_index, len(raw_df)):
            if str(raw_df.iloc[row_index, 0]).strip() == "Barcode":
                return row_index
        return len(raw_df)

    def _fill_excel(
        self,
        raw_df: pd.DataFrame,
        plates_dict: PlateDictionary,
        features: list[str],
    ) -> PlateDictionary:
        named_rows = raw_df.set_axis(["Well", "Laser Focus", *features], axis=1)
        plate_name = None
        collecting = False
        for _index, row in named_rows.iterrows():
            if row.iloc[0] == "Barcode":
                collecting = False
            elif collecting:
                well = row.iloc[0]
                if plate_name in plates_dict and well in plates_dict[plate_name]:
                    for feature in features:
                        plates_dict[plate_name][well][feature] = self.measurement_value(
                            row[feature]
                        )
            if row.iloc[0] == "Plate Name":
                plate_name = str(row.iloc[1])
            elif pd.isnull(row.iloc[0]):
                collecting = True
        return plates_dict
