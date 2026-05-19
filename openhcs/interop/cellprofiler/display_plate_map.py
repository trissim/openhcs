"""
Converted from CellProfiler: DisplayPlatemap
Original: DisplayPlatemap

Note: DisplayPlatemap is a visualization/data tool module that displays
measurements in a plate map view. In OpenHCS, this is converted to a
measurement aggregation function that produces plate map data for
visualization by the frontend.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Dict, List, Optional, Tuple

import numpy as np
from metaclass_registry import AutoRegisterMeta

from openhcs.core.memory.decorators import numpy
from openhcs.core.pipeline.function_contracts import special_inputs, special_outputs
from openhcs.core.registry_strategies import EnumKeyedStrategyMixin
from openhcs.processing.materialization import csv_materializer


class AggregationMethod(Enum):
    AVG = ("avg", np.mean)
    MEDIAN = ("median", np.median)
    STDEV = ("stdev", np.std)
    CV = ("cv%", None)

    def __init__(self, label: str, helper: object | None) -> None:
        self._value_ = label
        self._helper = helper

    def aggregate(self, values: np.ndarray) -> float:
        if self is AggregationMethod.CV:
            mean_value = np.mean(values)
            if mean_value == 0:
                return np.nan
            return float(np.std(values) / mean_value)
        if self._helper is None:
            raise NotImplementedError(
                f"{type(self).__name__}.{self.name} does not declare an aggregation helper."
            )
        return float(self._helper(values))


class PlateType(Enum):
    PLATE_96 = "96"
    PLATE_384 = "384"


class WellFormat(Enum):
    NAME = "well_name"
    ROWCOL = "row_column"


class ObjectOrImage(Enum):
    OBJECTS = "Object"
    IMAGE = "Image"


@dataclass
class PlatemapData:
    """Aggregated measurement data for plate map visualization."""
    plate: str
    well: str
    row: str
    column: str
    value: float
    measurement_name: str
    aggregation_method: str
    object_name: str


@dataclass
class PlatemapSummary:
    """Summary statistics for the entire plate map."""
    plate: str
    measurement_name: str
    aggregation_method: str
    min_value: float
    max_value: float
    mean_value: float
    well_count: int


class PlateDimensionStrategy(
    EnumKeyedStrategyMixin[PlateType],
    ABC,
    metaclass=AutoRegisterMeta,
):
    """Nominal dimensions for one CellProfiler plate type."""

    __registry_key__ = "plate_type_label"
    __skip_if_no_key__ = True

    plate_type: ClassVar[PlateType | None] = None
    plate_type_label: ClassVar[str | None] = None
    __enum_member_attr__ = "plate_type"
    __enum_label_attr__ = "plate_type_label"

    @classmethod
    def for_plate_type(cls, plate_type: PlateType) -> "PlateDimensionStrategy":
        return cls.for_enum_member(plate_type)

    @abstractmethod
    def dimensions(self) -> Tuple[int, int]:
        """Return row and column count for this plate type."""


class Plate96DimensionStrategy(PlateDimensionStrategy):
    """96-well plate dimensions."""

    plate_type = PlateType.PLATE_96

    def dimensions(self) -> Tuple[int, int]:
        return 8, 12


class Plate384DimensionStrategy(PlateDimensionStrategy):
    """384-well plate dimensions."""

    plate_type = PlateType.PLATE_384

    def dimensions(self) -> Tuple[int, int]:
        return 16, 24


def _parse_well_name(well: str) -> Tuple[str, str]:
    """Parse well name like 'A01' into row 'A' and column '01'."""
    if len(well) >= 2:
        row = well[0].upper()
        col = well[1:]
        return row, col
    return "", ""


def _aggregate_values(values: np.ndarray, method: AggregationMethod) -> float:
    """Aggregate array of values using specified method."""
    if len(values) == 0:
        return np.nan
    return method.aggregate(values)


@numpy
@special_outputs(
    ("platemap_data", csv_materializer(
        fields=["plate", "well", "row", "column", "value", 
                "measurement_name", "aggregation_method", "object_name"],
        analysis_type="platemap"
    )),
    ("platemap_summary", csv_materializer(
        fields=["plate", "measurement_name", "aggregation_method",
                "min_value", "max_value", "mean_value", "well_count"],
        analysis_type="platemap_summary"
    ))
)
def display_platemap(
    image: np.ndarray,
    measurement_values: Optional[np.ndarray] = None,
    plate_metadata: Optional[List[str]] = None,
    well_metadata: Optional[List[str]] = None,
    well_row_metadata: Optional[List[str]] = None,
    well_col_metadata: Optional[List[str]] = None,
    objects_or_image: ObjectOrImage = ObjectOrImage.IMAGE,
    object_name: str = "Image",
    measurement_name: str = "Measurement",
    plate_type: PlateType = PlateType.PLATE_96,
    well_format: WellFormat = WellFormat.NAME,
    agg_method: AggregationMethod = AggregationMethod.AVG,
    title: str = "",
) -> Tuple[np.ndarray, List[PlatemapData], List[PlatemapSummary]]:
    """
    Aggregate measurements by well for plate map visualization.
    
    This function aggregates per-image or per-object measurements into
    per-well values suitable for plate map display. The actual visualization
    is handled by the OpenHCS frontend.
    
    Args:
        image: Input image array (D, H, W) - passed through unchanged
        measurement_values: Array of measurement values to aggregate
        plate_metadata: List of plate identifiers per image
        well_metadata: List of well names (e.g., 'A01') per image
        well_row_metadata: List of well rows (e.g., 'A') per image
        well_col_metadata: List of well columns (e.g., '01') per image
        objects_or_image: Whether measurements are from objects or images
        object_name: Name of object type being measured
        measurement_name: Name of the measurement being displayed
        plate_type: Format of multiwell plate (96 or 384)
        well_format: How well location is specified (name or row/column)
        agg_method: How to aggregate multiple values per well
        title: Optional title for the plot
    
    Returns:
        Tuple of (image, platemap_data, platemap_summary)
    """
    platemap_entries = []
    platemap_summaries = []
    
    # If no measurement data provided, return empty results
    if measurement_values is None or plate_metadata is None:
        return image, platemap_entries, platemap_summaries
    
    # Construct well identifiers
    if well_format == WellFormat.NAME and well_metadata is not None:
        wells = well_metadata
    elif well_format == WellFormat.ROWCOL and well_row_metadata is not None and well_col_metadata is not None:
        wells = [f"{r}{c}" for r, c in zip(well_row_metadata, well_col_metadata)]
    else:
        return image, platemap_entries, platemap_summaries
    
    # Build dictionary mapping plate -> well -> list of values
    pm_dict: Dict[str, Dict[str, List[float]]] = {}
    
    for plate, well, data in zip(plate_metadata, wells, measurement_values):
        if data is None:
            continue
        
        # Handle both scalar and array measurements
        if isinstance(data, np.ndarray):
            values = data.flatten().tolist()
        else:
            values = [float(data)]
        
        if plate not in pm_dict:
            pm_dict[plate] = {}
        
        if well not in pm_dict[plate]:
            pm_dict[plate][well] = []
        
        pm_dict[plate][well].extend(values)
    
    # Aggregate values and create output entries
    for plate, well_dict in pm_dict.items():
        all_aggregated = []
        
        for well, values in well_dict.items():
            values_arr = np.array(values)
            aggregated = _aggregate_values(values_arr, agg_method)
            all_aggregated.append(aggregated)
            
            row, col = _parse_well_name(well)
            
            platemap_entries.append(PlatemapData(
                plate=plate,
                well=well,
                row=row,
                column=col,
                value=aggregated,
                measurement_name=measurement_name,
                aggregation_method=agg_method.value,
                object_name=object_name if objects_or_image == ObjectOrImage.OBJECTS else "Image"
            ))
        
        # Create summary for this plate
        if all_aggregated:
            valid_values = [v for v in all_aggregated if not np.isnan(v)]
            if valid_values:
                platemap_summaries.append(PlatemapSummary(
                    plate=plate,
                    measurement_name=measurement_name,
                    aggregation_method=agg_method.value,
                    min_value=float(np.min(valid_values)),
                    max_value=float(np.max(valid_values)),
                    mean_value=float(np.mean(valid_values)),
                    well_count=len(valid_values)
                ))
    
    return image, platemap_entries, platemap_summaries
