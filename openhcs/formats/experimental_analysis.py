"""
Experimental analysis for CX5 and MetaXpress microscopy data.

This module provides comprehensive analysis capabilities for high-content screening data
from ThermoFisher CX5 and MetaXpress systems. It handles experimental design configuration,
data parsing, replicate averaging, normalization, and result export.

Supports:
- CX5 format (ThermoFisher)
- MetaXpress format (Molecular Devices)
- Complex experimental designs with multiple conditions, doses, and replicates
- Control-based normalization
- Excel-based configuration and output
"""

import copy
import string
from statistics import fmean, pstdev
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from openhcs.core.config import NormalizationMethod
from openhcs.formats.experimental_layout_rows import (
    ExperimentalAnalysisScope,
    ExperimentalLayoutRowRole,
)
from openhcs.formats.experimental_result_formats import ExperimentalResultFormatStrategy


@dataclass(frozen=True, slots=True)
class ExcludedWellSet:
    """Replicate-local well exclusion matcher for nested analysis structures."""

    wells: frozenset[str]

    @classmethod
    def from_wells(cls, wells) -> "ExcludedWellSet":
        if wells is None:
            return cls(frozenset())
        return cls(frozenset(str(well).upper() for well in wells))

    @classmethod
    def from_positions(cls, excluded_positions, replicate) -> "ExcludedWellSet":
        if not excluded_positions or replicate not in excluded_positions:
            return cls(frozenset())
        return cls.from_wells(
            well for well, _plate_group in excluded_positions[replicate]
        )

    @property
    def empty(self) -> bool:
        return not self.wells

    def allows_well_id(self, well_id) -> bool:
        return str(well_id).upper() not in self.wells

    def allows_well_tuple(self, well_tuple) -> bool:
        return self.allows_well_id(well_tuple[0])

    def filter_nested(self, data):
        if self.empty:
            return data
        if not isinstance(data, dict):
            return data

        filtered = {}
        for key, value in data.items():
            if isinstance(value, dict):
                if is_well_key(key):
                    if self.allows_well_id(key):
                        filtered[key] = self.filter_nested(value)
                else:
                    filtered[key] = self.filter_nested(value)
            elif isinstance(value, list):
                filtered[key] = [
                    item
                    for item in value
                    if not is_well_tuple(item) or self.allows_well_tuple(item)
                ]
            else:
                filtered[key] = value
        return filtered


def is_well_key(key) -> bool:
    return (
        isinstance(key, str)
        and len(key) == 3
        and key[0].isalpha()
        and key[1:].isdigit()
    )


def is_well_tuple(item) -> bool:
    return isinstance(item, tuple) and len(item) >= 1


def result_format_strategy(scope: object) -> ExperimentalResultFormatStrategy:
    """Resolve one declared workbook scope through the nominal strategy root."""
    return ExperimentalResultFormatStrategy.for_enum_member(
        ExperimentalAnalysisScope.coerce(scope)
    )


def read_results(results_path: str, scope: Optional[str] = None) -> pd.DataFrame:
    """Read results through the strategy owned by the workbook scope."""
    return result_format_strategy(scope).read_results(results_path)


def get_features(raw_df, scope=None):
    """Return features through the strategy owned by the workbook scope."""
    return result_format_strategy(scope).features(raw_df)


def is_N_row(row_name):
    return ExperimentalLayoutRowRole(row_name).is_replicate_count


def is_well_all_replicates_row(row_name):
    return ExperimentalLayoutRowRole(row_name).is_well_all_replicates


def is_well_specific_replicate_row(row_name):
    return ExperimentalLayoutRowRole(row_name).is_well_specific_replicate


def plate_well_ids(row_indices=range(8), columns=range(1, 13)) -> list[str]:
    return [
        f"{string.ascii_uppercase[row_index]}{column:02d}"
        for row_index in row_indices
        for column in columns
    ]


def sanitize_compare(string1, string2):
    string1 = str(string1).lower()
    string2 = str(string2).lower()
    string1 = string1.replace("_", "")
    string1 = string1.replace(" ", "")
    string2 = string2.replace("_", "")
    string2 = string2.replace(" ", "")
    if len(string1) > 0 and not string1[-1] == "s":
        string1 += "s"
    if len(string2) > 0 and not string2[-1] == "s":
        string2 += "s"
    return string1 == string2


@dataclass(slots=True)
class PlateLayoutRoleState:
    wells: object = None
    wells_aligned: object = None
    groups: object = None
    positions_replicates: object = None
    positions: object = None

    def append_wells(self, values) -> None:
        if self.wells is None:
            self.wells = []
        self.wells += values

    def append_groups(self, values) -> None:
        if self.groups is None:
            self.groups = []
        self.groups += values

    def align_current_wells(self, values) -> None:
        if self.positions_replicates is None:
            self.positions_replicates = []
        if self.wells_aligned is None:
            self.wells_aligned = []
        self.positions_replicates += values
        self.wells_aligned += self.wells
        self.wells = None


@dataclass(slots=True)
class PlateLayoutState:
    layout: dict
    condition: object = None
    doses: object = None
    wells: object = None
    plate_groups: object = None
    N: Optional[int] = None
    specific_N: Optional[int] = None
    scope: object = None
    per_well_datapoints: bool = False
    conditions: list = None
    control: PlateLayoutRoleState = None
    excluded: PlateLayoutRoleState = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.control is None:
            self.control = PlateLayoutRoleState()
        if self.excluded is None:
            self.excluded = PlateLayoutRoleState()

    def result(self):
        return (
            self.scope,
            self.layout,
            self.conditions,
            self.control.positions,
            self.excluded.positions,
            self.per_well_datapoints,
        )


@dataclass(frozen=True, slots=True)
class PlateLayoutRow:
    row: object
    row_content: str
    row_name: str

    @classmethod
    def from_pandas_row(cls, row) -> "PlateLayoutRow":
        return cls(
            row=row,
            row_content=str(row.iloc[0]) if pd.notna(row.iloc[0]) else "",
            row_name=str(row.name) if pd.notna(row.name) else "",
        )

    @property
    def values(self):
        return self.row.dropna().tolist()

    @property
    def has_values(self) -> bool:
        return bool(self.values)

    def name_is(self, label: str) -> bool:
        return sanitize_compare(self.row_name, label)

    def content_is(self, label: str) -> bool:
        return sanitize_compare(self.row_content, label)

    def marker_is(self, *labels: str) -> bool:
        return any(self.content_is(label) or self.name_is(label) for label in labels)

    def parameter_value(self):
        if any(self.name_is(label) for label in ("scope", "microscope")):
            return self.row.iloc[0]
        return self.row.iloc[1]


class PlateLayoutParameterReader:
    def __init__(self, state: PlateLayoutState):
        self.state = state

    def read_replicate_count(self, row: PlateLayoutRow) -> bool:
        if not (is_N_row(row.row_content) or is_N_row(row.row_name)):
            return False
        self.state.N = int(
            row.row.iloc[0] if is_N_row(row.row_name) else row.row.iloc[1]
        )
        for replicate_index in range(self.state.N):
            self.state.layout["N" + str(replicate_index + 1)] = {}
        return True

    def read_scope(self, row: PlateLayoutRow) -> None:
        if row.marker_is("scope", "microscope"):
            self.state.scope = row.parameter_value()

    def read_per_well_datapoints(self, row: PlateLayoutRow) -> None:
        labels = (
            "per well datapoints",
            "per well datapoint",
            "individual wells",
            "individual well",
        )
        if not row.marker_is(*labels):
            return
        value = (
            row.row.iloc[0]
            if any(row.name_is(label) for label in labels)
            else row.row.iloc[1]
        )
        self.state.per_well_datapoints = str(value).lower().strip() in [
            "true",
            "1",
            "yes",
            "on",
            "enabled",
        ]

    def read_doses(self, row: PlateLayoutRow) -> None:
        if row.name_is("dose"):
            self.state.doses = row.values

    def read_wells(self, row: PlateLayoutRow) -> None:
        if is_well_all_replicates_row(row.row_name):
            self.state.wells = row.values
            self.state.specific_N = None
        row_role = ExperimentalLayoutRowRole(row.row_name)
        specific_replicate = row_role.specific_replicate
        if specific_replicate is not None:
            if self.state.N is None or not 1 <= specific_replicate <= self.state.N:
                raise ValueError(
                    f"well row {row.row_name!r} selects replicate "
                    f"{specific_replicate} outside 1..{self.state.N}"
                )
            self.state.specific_N = specific_replicate
            self.state.wells = row.values


class PlateLayoutRoleReader:
    def __init__(self, state: PlateLayoutState):
        self.state = state

    def read_plate_group(
        self,
        row: PlateLayoutRow,
        role: PlateLayoutRoleState,
        content_match: bool = True,
    ) -> bool:
        is_plate_group = (
            row.marker_is("plate group")
            if content_match
            else row.name_is("plate group")
        )
        if not (is_plate_group and role.wells is not None):
            return False
        role.append_groups(row.values)
        return True

    def read_wells(
        self,
        row: PlateLayoutRow,
        role: PlateLayoutRoleState,
        labels: tuple[str, ...],
        name_match_only: bool = False,
    ) -> bool:
        is_well_row = (
            any(row.name_is(label) for label in labels)
            if name_match_only
            else row.marker_is(*labels)
        )
        if not (is_well_row and row.has_values):
            return False
        role.append_wells(row.values)
        return True

    def read_replicates(
        self,
        row: PlateLayoutRow,
        role: PlateLayoutRoleState,
        content_match: bool = True,
    ) -> bool:
        is_group_row = (
            row.marker_is("group n") if content_match else row.name_is("group n")
        )
        if not (is_group_row and role.wells is not None):
            return False
        role.align_current_wells(row.values)
        return True

    def finalize_control_positions(self) -> None:
        control = self.state.control
        if control.wells is not None and control.wells_aligned is None:
            control.wells_aligned = control.wells
        control.wells = None
        if control.wells_aligned is None:
            control.positions = None
            return

        control.positions = {"N" + str(i + 1): [] for i in range(self.state.N)}
        if control.positions_replicates is not None:
            for index in range(len(control.wells_aligned)):
                replicate = "N" + str(control.positions_replicates[index])
                control.positions[replicate].append(
                    (control.wells_aligned[index], control.groups[index])
                )
            return

        for replicate in control.positions.keys():
            for index in range(len(control.wells_aligned)):
                control.positions[replicate].append(
                    (control.wells_aligned[index], control.groups[index])
                )

    def finalize_excluded_positions(self) -> None:
        excluded = self.state.excluded
        has_exclusions = (
            excluded.wells_aligned is not None
            and excluded.groups is not None
            and excluded.positions_replicates is not None
        )
        if not has_exclusions:
            excluded.positions = None
            return

        excluded.positions = {"N" + str(i + 1): [] for i in range(self.state.N)}
        filtered_wells = [
            well for well in excluded.wells_aligned if well != "Exclude Wells"
        ]
        filtered_groups = [group for group in excluded.groups if group != "Plate Group"]
        filtered_replicates = [
            replicate
            for replicate in excluded.positions_replicates
            if replicate != "Group N" and isinstance(replicate, (int, float))
        ]

        for index in range(
            min(len(filtered_wells), len(filtered_groups), len(filtered_replicates))
        ):
            replicate_key = "N" + str(int(filtered_replicates[index]))
            if replicate_key in excluded.positions:
                excluded.positions[replicate_key].append(
                    (filtered_wells[index], filtered_groups[index])
                )
        excluded.wells = None


class PlateLayoutConditionReader:
    def __init__(self, state: PlateLayoutState, roles: PlateLayoutRoleReader):
        self.state = state
        self.roles = roles

    def read_condition(self, row: PlateLayoutRow) -> None:
        if not row.marker_is("condition"):
            return
        self.roles.finalize_control_positions()
        self.roles.finalize_excluded_positions()
        condition = row.row.iloc[0]
        for replicate_index in range(self.state.N):
            replicate_key = "N" + str(replicate_index + 1)
            if condition not in self.state.layout[replicate_key]:
                self.state.layout[replicate_key][condition] = {}
        self.state.condition = condition
        self.state.conditions.append(condition)


class PlateLayoutAssignmentReader:
    def __init__(self, state: PlateLayoutState):
        self.state = state

    def read_plate_group_assignments(self, row: PlateLayoutRow) -> None:
        if not row.name_is("plate group"):
            return
        self.state.plate_groups = row.values
        if self.state.specific_N is None:
            self.assign_plate_groups_to_all_replicates()
            return
        self.assign_plate_groups_to_specific_replicate(self.state.specific_N)

    def assign_plate_groups_to_all_replicates(self) -> None:
        for replicate_index in range(self.state.N):
            self.assign_plate_groups_to_specific_replicate(replicate_index + 1)

    def assign_plate_groups_to_specific_replicate(self, specific_N: int) -> None:
        replicate_key = "N" + str(specific_N)
        assignment_columns = (
            self.state.doses,
            self.state.wells,
            self.state.plate_groups,
        )
        if any(values is None for values in assignment_columns):
            raise ValueError(
                "plate layout assignments require dose, well, and plate group rows"
            )
        cardinalities = tuple(len(values) for values in assignment_columns)
        if len(set(cardinalities)) != 1:
            raise ValueError(
                "plate layout dose, well, and plate group rows require equal "
                f"column counts; received {cardinalities}"
            )
        condition_values = self.state.layout[replicate_key][self.state.condition]
        for dose, well, plate_group in zip(*assignment_columns, strict=True):
            if dose not in condition_values:
                condition_values[dose] = []
            condition_values[dose].append((well, plate_group))


class PlateLayoutBuilder:
    def __init__(self):
        self.state = PlateLayoutState(layout={})
        self.parameters = PlateLayoutParameterReader(self.state)
        self.roles = PlateLayoutRoleReader(self.state)
        self.conditions = PlateLayoutConditionReader(self.state, self.roles)
        self.assignments = PlateLayoutAssignmentReader(self.state)

    def parse(self, df):
        for _index, row in df.iterrows():
            self.process_row(PlateLayoutRow.from_pandas_row(row))
        return self.state.result()

    def process_row(self, row: PlateLayoutRow) -> None:
        if self.parameters.read_replicate_count(row):
            return
        self.parameters.read_scope(row)
        self.parameters.read_per_well_datapoints(row)
        if self.roles.read_plate_group(row, self.state.control, content_match=False):
            return
        if self.roles.read_wells(
            row, self.state.control, ("control", "control well"), name_match_only=True
        ):
            return
        if self.roles.read_wells(
            row, self.state.excluded, ("exclude wells", "excluded wells", "exclude")
        ):
            return
        if self.roles.read_plate_group(row, self.state.excluded):
            return
        if self.roles.read_replicates(row, self.state.excluded):
            return
        if self.roles.read_replicates(row, self.state.control, content_match=False):
            return
        self.conditions.read_condition(row)
        self.parameters.read_doses(row)
        self.parameters.read_wells(row)
        self.assignments.read_plate_group_assignments(row)


def read_plate_layout(config_path, sheet_name: str = "drug_curve_map"):
    xls = pd.ExcelFile(config_path)
    df = pd.read_excel(xls, sheet_name, index_col=0, header=None)
    return PlateLayoutBuilder().parse(df.dropna(how="all"))


def create_well_dict(raw_df, wells=None, scope=None):
    if wells is None:
        wells = plate_well_ids()
    features = get_features(raw_df, scope=scope)
    return {well: {feature: None for feature in features} for well in wells}


def add_well_to_well_dict(wells, well_dict, raw_df):
    features = get_features(raw_df).columns
    for well in wells:
        well_dict[well] = {feature: None for feature in features}
    return well_dict


def create_plates_dict(raw_df, scope=None):
    return result_format_strategy(scope).create_plates_dict(raw_df)


def fill_plates_dict(raw_df, plates_dict, scope=None):
    features = get_features(raw_df, scope=scope)
    return result_format_strategy(scope).fill_plates_dict(
        raw_df,
        plates_dict,
        list(features),
    )


def average_plates_all_replicates(plate_groups, plates_dict, raw_df):
    averaged_plates_dict = {replicate: None for replicate in plate_groups.keys()}
    for replicate in plate_groups.keys():
        one_replicate = average_plates_one_replicate(
            plate_groups[replicate], plates_dict, raw_df
        )
        averaged_plates_dict[replicate] = one_replicate
    return averaged_plates_dict


def average_plates_duplicate_rows(
    plate_groups, plates_dict, raw_df, wells_to_average=None, scope=None
):
    features = get_features(raw_df, scope=scope)
    averaged_plates_dict = {}
    for plate_name, plate in plates_dict.items():
        average_plate = create_well_dict(raw_df, scope=scope, wells=wells_to_average)
        for well in wells_to_average:
            average_plate = average_rows(plate, average_plate, well, features)
        averaged_plates_dict[plate_name] = average_plate
    return plates_dict


def average_rows(plate_dict, average_plate, well, features, num_rows_average=2):
    original_well = well
    wells_to_average = []
    wells_to_average.append(well)
    for i in range(num_rows_average - 1):
        well_next_row = get_well_next_row(well)
        wells_to_average.append(well_next_row)
        well_next_row = well
    for feature in features:
        average_value = 0
        for well in wells_to_average:
            average_value += plate_dict[well][feature]
        average_value = average_value / num_rows_average
        average_plate[original_well][feature] = average_value
    return average_plate


def get_well_next_row(well):
    return chr(ord(well[0]) + 1) + well[1:]


def average_plates(plates, raw_df, scope=None):
    average_plate = create_well_dict(raw_df, scope=scope)
    features = get_features(raw_df)
    for well in average_plate.keys():
        for feature in features:
            average_value = 0
            for plate in plates:
                average_value += plate[well][feature]
            average_value = average_value / len(plates)
            average_plate[well][feature] = average_value
    return average_plate


def average_plates_one_replicate(averaged_plates_names_dict, plates_dict, raw_df):
    averaged_plates_dict = {
        plate_average_name: None
        for plate_average_name in averaged_plates_names_dict.keys()
    }
    for plate_average_name in averaged_plates_dict.keys():
        plates_to_average = averaged_plates_names_dict[plate_average_name]
        plates_to_average = [
            plates_dict[plate_name] for plate_name in plates_to_average
        ]
        averaged_plates_dict[plate_average_name] = average_plates(
            plates_to_average, raw_df
        )
    return averaged_plates_dict


def apply_excluded_positions_to_experiment_locations(
    experiment_dict_locations,
    excluded_positions,
) -> None:
    """Remove replicate-scoped excluded wells from experiment locations in-place."""
    if not excluded_positions:
        return

    for condition_locations in experiment_dict_locations.values():
        for replicate, dose_locations in condition_locations.items():
            excluded = ExcludedWellSet.from_positions(
                excluded_positions,
                replicate,
            )
            if excluded.empty:
                continue
            for dose, locations in dose_locations.items():
                dose_locations[dose] = [
                    location
                    for location in locations
                    if excluded.allows_well_tuple(location)
                ]


def apply_excluded_positions_to_control_positions(
    ctrl_positions,
    excluded_positions,
) -> None:
    """Remove replicate-scoped excluded wells from control positions in-place."""
    if not ctrl_positions or not excluded_positions:
        return

    for replicate, ctrl_wells in ctrl_positions.items():
        excluded = ExcludedWellSet.from_positions(excluded_positions, replicate)
        if not excluded.empty:
            ctrl_positions[replicate] = [
                well_tuple
                for well_tuple in ctrl_wells
                if excluded.allows_well_tuple(well_tuple)
            ]


def load_plate_groups(config_path, sheet_name: str = "plate_groups"):
    xls = pd.ExcelFile(config_path)
    df = pd.read_excel(xls, sheet_name, index_col=0, header=None)
    replicates = df.index.tolist()[1:]
    groups = [str(group) for group in df.columns.tolist()]
    plate_groups = {
        replicate: {group: None for group in groups} for replicate in replicates
    }
    for group in groups:
        for replicate in replicates:
            # well_replicates = df.filter(like=group).loc[replicate].tolist()[0]
            plate_groups[replicate][group] = df.loc[replicate][int(group)]
    return plate_groups


def normalize_plate(plate, reference_wells, raw_df, ctrl_avg_name):
    features = get_features(raw_df)
    normalized_plate = create_well_dict(raw_df)
    normalized_plate = add_well_to_well_dict([ctrl_avg_name], normalized_plate, raw_df)
    for feature in features:
        control_values = [plate[well][feature] for well in reference_wells]
        control_avg = np.mean(np.array(control_values))
        normalized_plate[ctrl_avg_name][feature] = control_avg
        for well in normalized_plate.keys():
            if well not in ctrl_avg_name:
                try:
                    normalized_plate[well][feature] = plate[well][feature] / control_avg
                except (KeyError, TypeError, ZeroDivisionError):
                    normalized_plate[well][feature] = plate[well][feature]
    return normalized_plate


def normalize_all_plates(plates_dict, reference_wells, raw_df, ctrl_avg_name):
    normalized_plates = {replicate: {} for replicate in plates_dict.keys()}
    for replicate, condition_plates in plates_dict.items():
        for condition, plate in condition_plates.items():
            normalized_plates[replicate][condition] = normalize_plate(
                plate, reference_wells, raw_df, ctrl_avg_name
            )
    return normalized_plates


def create_table_for_feature(feature, experiment_dict_values):
    conditions = list(experiment_dict_values.keys())

    # Create hierarchical column structure: (condition, replicate)
    col_tuples = []
    values = []

    for condition in conditions:
        # Get replicates for this specific condition (they may differ in per-well mode)
        condition_replicates = list(experiment_dict_values[condition].keys())

        for replicate in condition_replicates:
            # Get the value from any available dose for this condition-replicate
            for dose in experiment_dict_values[condition][replicate].keys():
                try:
                    feature_data = experiment_dict_values[condition][replicate][dose][
                        feature
                    ]
                    if isinstance(feature_data, dict):
                        # Handle both averaged and per-well dictionary formats
                        if "averaged" in feature_data:
                            # Averaged mode - single value
                            col_tuples.append((condition, replicate))
                            values.append(feature_data["averaged"])
                        else:
                            # Per-well mode - multiple values (each well becomes a separate column)
                            for well_id, value in feature_data.items():
                                col_tuples.append((condition, replicate))
                                values.append(value)
                    else:
                        # Fallback for old format (shouldn't happen now)
                        col_tuples.append((condition, replicate))
                        values.append(feature_data)
                    break  # Use the first available dose
                except (KeyError, TypeError):
                    continue

    # Create DataFrame in GraphPad Prism format: N as y-axis (rows), conditions as x-axis (columns)
    # Group values by condition
    condition_data = {}
    for i, (condition, replicate) in enumerate(col_tuples):
        if condition not in condition_data:
            condition_data[condition] = []
        condition_data[condition].append(values[i])

    # Create DataFrame with conditions as columns and N as rows
    # Handle case where all conditions are empty (no data available)
    if not condition_data or all(len(vals) == 0 for vals in condition_data.values()):
        # Return empty DataFrame with condition columns
        return pd.DataFrame(
            columns=sorted(condition_data.keys()) if condition_data else []
        )

    max_n = max(len(vals) for vals in condition_data.values())
    data_matrix = []
    for n in range(max_n):
        row = []
        for condition in sorted(condition_data.keys()):
            if n < len(condition_data[condition]):
                row.append(condition_data[condition][n])
            else:
                row.append(None)  # Fill missing values with None
        data_matrix.append(row)

    feature_table = pd.DataFrame(
        data_matrix,
        columns=sorted(condition_data.keys()),
        index=[f"N{i + 1}" for i in range(max_n)],
    )

    return feature_table


def create_table_for_feature_per_well(feature, experiment_dict_values):
    """Create feature table with individual wells as columns."""
    conditions = list(experiment_dict_values.keys())
    replicates = list(list(experiment_dict_values.values())[0].keys())

    col_names = []
    values = []

    for condition in conditions:
        for replicate in replicates:
            # Get the value from any available dose for this condition-replicate
            for dose in experiment_dict_values[condition][replicate].keys():
                feature_data = experiment_dict_values[condition][replicate][dose][
                    feature
                ]
                if isinstance(feature_data, dict):  # Per-well mode
                    for well_id, value in feature_data.items():
                        col_names.append(f"{condition}_{replicate}_{well_id}")
                        values.append(value)
                else:  # Regular averaged mode (fallback)
                    col_names.append(f"{condition}_{replicate}")
                    values.append(feature_data)
                break  # Use first available dose

    return pd.DataFrame([values], columns=col_names)


def create_table_for_feature_per_well_mode(feature, experiment_dict_values):
    """
    Create feature table for per-well mode with:
    - Rows: Doses
    - Columns: Hierarchical (Condition > Replicate/Well)
    """
    conditions = sorted(experiment_dict_values.keys())

    # Collect all doses across all conditions and replicates
    all_doses = set()
    for condition in conditions:
        for replicate in experiment_dict_values[condition].keys():
            all_doses.update(experiment_dict_values[condition][replicate].keys())
    doses = sorted(all_doses)

    # Build hierarchical column structure: (condition, replicate_well)
    col_tuples = []
    data_matrix = []

    for dose in doses:
        row_data = []
        for condition in conditions:
            for replicate in sorted(experiment_dict_values[condition].keys()):
                if dose in experiment_dict_values[condition][replicate]:
                    feature_data = experiment_dict_values[condition][replicate][dose][
                        feature
                    ]
                    if isinstance(feature_data, dict):
                        # Per-well mode: add each well as a separate column
                        for well_id in sorted(feature_data.keys()):
                            if dose == doses[0]:  # Only add column headers once
                                col_tuples.append((condition, f"{replicate}_{well_id}"))
                            row_data.append(feature_data[well_id])
                    else:
                        # Averaged mode fallback
                        if dose == doses[0]:
                            col_tuples.append((condition, replicate))
                        row_data.append(feature_data)
                else:
                    # Dose not available for this condition/replicate
                    if dose == doses[0]:
                        col_tuples.append((condition, replicate))
                    row_data.append(None)
        data_matrix.append(row_data)

    # Create DataFrame with hierarchical columns
    if col_tuples:
        multi_index = pd.MultiIndex.from_tuples(col_tuples)
        df = pd.DataFrame(data_matrix, columns=multi_index, index=doses)
    else:
        df = pd.DataFrame()

    return df


def create_all_feature_tables(
    experiment_dict_values, features, per_well_datapoints=False
):
    """Create feature tables. Both modes now use the same function since data is in dict format."""
    feature_tables = {feature: None for feature in features}
    for feature in features:
        if per_well_datapoints:
            feature_tables[feature] = create_table_for_feature_per_well_mode(
                feature, experiment_dict_values
            )
        else:
            feature_tables[feature] = create_table_for_feature(
                feature, experiment_dict_values
            )
    return feature_tables


def filter_feature_tables_by_plot_config(feature_tables, config_file):
    """
    Filter feature tables to only include metrics listed in plot_config sheet.

    Args:
        feature_tables: Dict of feature name -> DataFrame
        config_file: Path to config.xlsx with plot_config sheet

    Returns:
        Filtered dict with only metrics from plot_config
    """
    try:
        # Read plot_config sheet
        plot_config_df = pd.read_excel(
            config_file, sheet_name="plot_config", header=None
        )

        # Get metric names from first column (skip header if present)
        metrics_to_plot = set()
        for value in plot_config_df[0].dropna():
            # Skip header row if it says "Metric" or similar
            if isinstance(value, str) and value.lower() not in [
                "metric",
                "metrics",
                "name",
            ]:
                metrics_to_plot.add(value)

        if not metrics_to_plot:
            # No plot_config or empty, return all features
            return feature_tables

        # Filter feature tables
        filtered_tables = {
            k: v for k, v in feature_tables.items() if k in metrics_to_plot
        }

        print(
            f"Filtered to {len(filtered_tables)} metrics from plot_config (out of {len(feature_tables)} total)"
        )
        return filtered_tables

    except Exception as e:
        # If plot_config doesn't exist or error reading it, return all features
        print(f"Could not read plot_config sheet: {e}")
        print("Including all metrics in compiled results")
        return feature_tables


def feature_tables_to_excel(feature_tables, outpath):
    def remove_inval_chars(name):
        inval_chars = ["[", "]", ":", "*", "?", "/", "\\"]
        for char in inval_chars:
            name = name.replace(char, "")
        # Modern Excel supports up to 255 characters for sheet names
        return name[:255]

    with pd.ExcelWriter(outpath, engine="openpyxl") as writer:
        for feature in feature_tables.keys():
            table = feature_tables[feature]
            if table is not None:
                # Write with merge_cells=False to avoid Excel merge conflicts
                table.to_excel(
                    writer, sheet_name=remove_inval_chars(feature), merge_cells=False
                )


def create_duplicate_wells():
    return plate_well_ids(row_indices=range(0, 8, 2))


def make_experiment_dict_locations(plate_groups, plate_layout, conditions):
    experiment_dict = {condition: {} for condition in conditions}
    # experiment_dict={replicate:{} for replicate in plate_layout.keys()}
    for replicate, conditions in plate_layout.items():
        for condition, doses in conditions.items():
            experiment_dict[condition][replicate] = {
                dose: locations for dose, locations in doses.items()
            }
    return experiment_dict


def make_experiment_dict_values(
    plates, experiment_dict_locations, features, plate_groups, per_well_datapoints=False
):
    if per_well_datapoints:
        # In per-well mode, restructure data so each well becomes a separate "replicate"
        experiment_dict_values = {}

        for condition, replicates in experiment_dict_locations.items():
            experiment_dict_values[condition] = {}
            well_counter = 1  # Counter for creating unique replicate names

            for replicate, doses in replicates.items():
                for dose, locations in doses.items():
                    # Each well becomes a separate "replicate"
                    for location in locations:
                        well, plate_group = location
                        plate_name = str(plate_groups[replicate][str(plate_group)])

                        # Create unique replicate name for this well
                        well_replicate_name = f"N{well_counter}"
                        well_counter += 1

                        # Get individual well values
                        feature_value_dict = {}
                        for feature in features:
                            value = plates[plate_name][well][feature]
                            try:
                                well_value = float(value) if value is not None else 0.0
                            except (ValueError, TypeError):
                                well_value = 0.0
                            # Store as single-well dictionary to maintain format consistency
                            well_id = f"{well}_P{plate_group}"
                            feature_value_dict[feature] = {well_id: well_value}

                        # Store this well as its own replicate
                        experiment_dict_values[condition][well_replicate_name] = {
                            dose: feature_value_dict
                        }
    else:
        # Original averaging mode
        experiment_dict_values = copy.deepcopy(experiment_dict_locations)
        for condition, replicates in experiment_dict_locations.items():
            for replicate, doses in replicates.items():
                for dose, locations in doses.items():
                    feature_value_dict = {
                        feature: average_wells(
                            locations,
                            replicate,
                            feature,
                            plates,
                            plate_groups,
                        )
                        for feature in features
                    }
                    experiment_dict_values[condition][replicate][dose] = (
                        feature_value_dict
                    )

    return experiment_dict_values


def average_wells(locations, replicate, feature, plates, plate_groups):
    """Return dict with averaged value to match per-well format.

    Gracefully handles missing wells - only averages wells that exist in the data.
    Returns None if no valid wells are found.
    """
    values = location_values(locations, replicate, feature, plates, plate_groups)
    if not values:
        return {"averaged": None}  # No valid wells found

    averaged_value = sum(values) / float(len(values))
    # Return as dictionary to match per-well format
    return {"averaged": averaged_value}


def individual_wells(locations, replicate, feature, plates, plate_groups):
    """Return dict of individual well values instead of averaging.

    Gracefully handles missing wells - only includes wells that exist in the data.
    """
    well_values = {}
    for location in locations:
        value = location_to_value(location, replicate, feature, plates, plate_groups)
        if value is not None:  # Skip missing wells
            well, plate_group = location
            # Create unique well identifier including plate group
            well_id = f"{well}_P{plate_group}"
            well_values[well_id] = value
    return well_values


def location_values(locations, replicate, feature, plates, plate_groups):
    values = []
    for location in locations:
        value = location_to_value(location, replicate, feature, plates, plate_groups)
        if value is not None:
            values.append(value)
    return values


def location_to_value(location, replicate, feature, plates, plate_groups):
    well, plate_group = location
    plate_name = str(
        plate_groups[replicate][str(plate_group)]
    )  # Ensure string conversion

    # Check if plate exists
    if plate_name not in plates or plate_name == "nan":
        return None  # Plate doesn't exist

    # Check if well exists in plate
    if well not in plates[plate_name]:
        return None  # Well doesn't exist in this plate

    value = plates[plate_name][well][feature]
    # Convert to float for numerical operations
    try:
        return float(value)
    except (ValueError, TypeError):
        return None  # Non-numeric or missing values


@dataclass(frozen=True, slots=True)
class ControlNormalizationReference:
    """Replicate-local control distribution for one measurement feature."""

    mean: float
    std: float

    @classmethod
    def from_values(cls, values) -> "ControlNormalizationReference | None":
        numeric_values = tuple(float(value) for value in values if value is not None)
        if not numeric_values:
            return None
        return cls(mean=fmean(numeric_values), std=pstdev(numeric_values))

    def normalize(self, value, method: NormalizationMethod):
        if value is None:
            return None
        return method.normalize(
            float(value),
            control_mean=self.mean,
            control_std=self.std,
        )


def _normalize_feature_value(value, reference, method: NormalizationMethod):
    if reference is None:
        return None if not isinstance(value, dict) else {key: None for key in value}
    if isinstance(value, dict):
        return {key: reference.normalize(item, method) for key, item in value.items()}
    return reference.normalize(value, method)


def normalize_experiment(
    experiment_dict_values,
    ctrl_positions,
    features,
    plates,
    plate_groups,
    method: NormalizationMethod = NormalizationMethod.FOLD_CHANGE,
):
    """Normalize every feature to its replicate-local declared control wells."""
    normalized = copy.deepcopy(experiment_dict_values)
    references = {
        feature: {
            replicate: ControlNormalizationReference.from_values(
                location_values(
                    control_wells,
                    replicate,
                    feature,
                    plates,
                    plate_groups,
                )
            )
            for replicate, control_wells in ctrl_positions.items()
        }
        for feature in features
    }

    for condition, replicates in experiment_dict_values.items():
        for replicate, doses in replicates.items():
            for dose, feature_values in doses.items():
                for feature in features:
                    normalized[condition][replicate][dose][feature] = (
                        _normalize_feature_value(
                            feature_values[feature],
                            references.get(feature, {}).get(replicate),
                            method,
                        )
                    )
    return normalized


def project_plates_without_excluded_positions(
    plates_dict,
    excluded_positions,
    plate_groups,
):
    """Return a plate projection with replicate-scoped exclusions removed."""
    projected_plates = copy.deepcopy(plates_dict)
    if not excluded_positions:
        return projected_plates

    for replicate, excluded_locations in excluded_positions.items():
        replicate_plate_groups = plate_groups.get(replicate, {})
        for well, plate_group in excluded_locations:
            plate_identifier = replicate_plate_groups.get(str(plate_group))
            if plate_identifier is None or pd.isna(plate_identifier):
                continue
            if isinstance(plate_identifier, (int, float)):
                plate_identifier = str(int(plate_identifier))
            else:
                plate_identifier = str(plate_identifier)
            projected_plates.get(plate_identifier, {}).pop(well, None)
    return projected_plates


def write_values_heat_map(plates_dict, features, outpath):
    """Write one conditionally formatted 8x12 plate grid per feature and plate."""
    with pd.ExcelWriter(outpath, engine="xlsxwriter") as writer:
        workbook = writer.book
        for feature in features:
            sheet_name = remove_inval_chars(str(feature)[:31])
            worksheet = workbook.add_worksheet(sheet_name)
            writer.sheets[sheet_name] = worksheet
            start_row = 0
            for plate, wells in plates_dict.items():
                worksheet.write(start_row, 0, str(plate))
                plate_values = []
                for row_name in string.ascii_uppercase[:8]:
                    row_values = []
                    for column in range(1, 13):
                        value = wells.get(f"{row_name}{column:02d}", {}).get(feature)
                        try:
                            value = float(value) if value is not None else None
                        except (TypeError, ValueError):
                            pass
                        row_values.append(value)
                    plate_values.append(row_values)
                pd.DataFrame(plate_values).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=start_row + 1,
                    header=False,
                    index=False,
                )
                worksheet.conditional_format(
                    start_row + 1,
                    0,
                    start_row + 8,
                    11,
                    {"type": "3_color_scale"},
                )
                start_row += 10


def create_reference_wells():
    rows = [string.ascii_uppercase[i] for i in range(8)]
    cols = [i + 1 for i in range(6, 12)]
    wells = []
    for row in rows:
        for col in cols:
            wells.append((str(row) + str(col).zfill(2), 2))
    return wells


def remove_inval_chars(name):
    inval_chars = ["[", "]", ":", "*", "?", "/", "\\"]
    for char in inval_chars:
        name = name.replace(char, "")
    return name


def run_experimental_analysis(
    results_path: str = "mx_results.xlsx",
    config_file: str = "./config.xlsx",
    compiled_results_path: str = "./compiled_results_normalized.xlsx",
    heatmap_path: str = "./heatmaps.xlsx",
):
    """Run the compatibility entry point through the current engine authority."""
    import warnings

    from openhcs.core.config import ExperimentalAnalysisConfig
    from openhcs.processing.backends.experimental_analysis import (
        ExperimentalAnalysisEngine,
    )

    warnings.warn(
        "run_experimental_analysis is deprecated. Use ExperimentalAnalysisEngine instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = ExperimentalAnalysisEngine(ExperimentalAnalysisConfig()).run_analysis(
        results_path=results_path,
        config_file=config_file,
        compiled_results_path=compiled_results_path,
        heatmap_path=heatmap_path,
    )
    return result["experiment_values"], result["feature_tables"]
