"""
Position format module for handling position data in various formats.

This module provides a canonical format for position data and utilities for
parsing and serializing position data in different formats.
"""

import enum
import re
from dataclasses import dataclass
from io import StringIO
from typing import Any, Callable, Dict, List, Optional, Type

import pandas as pd


# Canonical pattern format schema
@dataclass(frozen=True)
class PatternFormatSchema:
    """
    Schema for pattern formats.
    
    This class defines the canonical schema for pattern formats, ensuring
    consistent validation across all pattern-related operations.
    
    Attributes:
        name: The name of the format
        description: A description of the format
        required_fields: List of field names that must be present
        optional_fields: List of field names that may be present
        field_types: Dictionary mapping field names to their expected types
    """
    name: str
    description: str
    required_fields: List[str]
    optional_fields: List[str]
    field_types: Dict[str, Type]
    
    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Validate data against the schema.
        
        Args:
            data: The data to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If required fields are missing or field types are incorrect
        """
        # Check required fields
        for field in self.required_fields:
            if field not in data:
                raise ValueError(f"Required field '{field}' missing from data")
        
        # Check field types
        for field, value in data.items():
            if field in self.field_types:
                expected_type = self.field_types[field]
                if not isinstance(value, expected_type):
                    raise ValueError(f"Field '{field}' has incorrect type. Expected {expected_type.__name__}, got {type(value).__name__}")
        
        return True

# Minimal PositionRecord structure for parsers/serializers
@dataclass
class PositionRecordData:
    """
    Data structure for position records.
    
    This class represents a position record with standardized fields.
    """
    filename: str
    grid_x: int
    grid_y: int
    pos_x: float
    pos_y: float

    # Schema for position record data
    SCHEMA = PatternFormatSchema(
        name="PositionRecordData",
        description="Schema for position record data",
        required_fields=["filename", "pos_x", "pos_y"],
        optional_fields=["grid_x", "grid_y"],
        field_types={
            "filename": str,
            "grid_x": int,
            "grid_y": int,
            "pos_x": float,
            "pos_y": float
        }
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionRecordData':
        """
        Create a PositionRecordData from a dictionary.
        
        Args:
            data: Dictionary containing position record data
            
        Returns:
            PositionRecordData instance
            
        Raises:
            ValueError: If data does not conform to schema
        """
        # Validate against schema
        cls.SCHEMA.validate(data)
        
        return cls(
            filename=str(data["filename"]),
            grid_x=int(data.get("grid_x", 0)),
            grid_y=int(data.get("grid_y", 0)),
            pos_x=float(data["pos_x"]),
            pos_y=float(data["pos_y"])
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary.
        
        Returns:
            Dictionary representation of the position record
        """
        data = {
            "filename": self.filename,
            "grid_x": self.grid_x,
            "grid_y": self.grid_y,
            "pos_x": self.pos_x,
            "pos_y": self.pos_y,
        }
        
        # Validate against schema
        self.SCHEMA.validate(data)
        
        return data

class CSVPositionFormat(enum.Enum):
    """Enumeration of supported CSV position file formats."""
    STANDARD = "standard"  # filename,grid_x,grid_y,pos_x,pos_y with header
    KV_SEMICOLON = "kv_semicolon"  # file: ...; position: (...); (optional grid: (...))

# Type for parser and serializer functions
ParserFunction = Callable[[str], List[PositionRecordData]]
SerializerFunction = Callable[[List[PositionRecordData]], str]

@dataclass
class PositionCSVFormatSpec:
    """
    Specification for a CSV position file format.
    
    This class defines the specification for a CSV position file format,
    including parsing and serialization functions.
    """
    format_name: str
    format_description: str
    delimiter: Optional[str]
    has_header: bool
    parser: ParserFunction
    serializer: SerializerFunction
    # Optional: regex pattern if needed by parser/serializer, or they handle it internally
    pattern: Optional[str] = None
    
    # Schema for CSV format specification
    SCHEMA = PatternFormatSchema(
        name="PositionCSVFormatSpec",
        description="Schema for CSV format specification",
        required_fields=["format_name", "format_description", "has_header", "parser", "serializer"],
        optional_fields=["delimiter", "pattern"],
        field_types={
            "format_name": str,
            "format_description": str,
            "delimiter": (str, type(None)),
            "has_header": bool,
            "parser": Callable,
            "serializer": Callable,
            "pattern": (str, type(None))
        }
    )

# --- Parser Implementations ---

def _parse_standard_csv(content: str) -> List[PositionRecordData]:
    """
    Parse standard CSV format.
    
    Format: filename,grid_x,grid_y,pos_x,pos_y (with header)
    
    Args:
        content: CSV content as string
        
    Returns:
        List of PositionRecordData objects
        
    Raises:
        ValueError: If required columns are missing or data is invalid
    """
    df = pd.read_csv(StringIO(content))
    required_cols = {"filename", "pos_x", "pos_y"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Standard CSV missing required columns: {missing}")
    
    records = []
    for _, row in df.iterrows():
        # Create data dictionary for validation
        data = {
            "filename": str(row["filename"]),
            "grid_x": int(row.get("grid_x", 0)),
            "grid_y": int(row.get("grid_y", 0)),
            "pos_x": float(row["pos_x"]),
            "pos_y": float(row["pos_y"])
        }
        
        # Create record with validation
        records.append(PositionRecordData.from_dict(data))
    return records

def _parse_kv_semicolon_csv(content: str) -> List[PositionRecordData]:
    """
    Parse key-value semicolon CSV format.
    
    Format: file: ...; position: (...); [grid: (...)]
    
    Args:
        content: CSV content as string
        
    Returns:
        List of PositionRecordData objects
        
    Raises:
        ValueError: If data format is invalid
    """
    # Pattern: file: <filename>; position: (<x>, <y>)[; grid: (<gx>, <gy>)]
    line_pattern = re.compile(
        r"file:\s*(?P<filename>[^;]+);\s*"
        r"position:\s*\((?P<pos_x>[^,]+),\s*(?P<pos_y>[^)]+)\)"
        r"(?:;\s*grid:\s*\((?P<grid_x>[^,]+),\s*(?P<grid_y>[^)]+)\))?" # Optional grid
    )
    records = []
    for line_num, line_text in enumerate(content.strip().split('\n')):
        line_text = line_text.strip()
        if not line_text:
            continue
        match = line_pattern.match(line_text)
        if not match:
            raise ValueError(f"Malformed line {line_num + 1} in KV_SEMICOLON CSV: '{line_text}'")
        
        # Create data dictionary for validation
        data = match.groupdict()
        data_dict = {
            "filename": data["filename"].strip(),
            "grid_x": int(data.get("grid_x", 0)) if data.get("grid_x") else 0,
            "grid_y": int(data.get("grid_y", 0)) if data.get("grid_y") else 0,
            "pos_x": float(data["pos_x"].strip()),
            "pos_y": float(data["pos_y"].strip())
        }
        
        # Create record with validation
        records.append(PositionRecordData.from_dict(data_dict))
    return records

# --- Serializer Implementations ---

def _serialize_standard_csv(records: List[PositionRecordData]) -> str:
    """
    Serialize to standard CSV format.
    
    Format: filename,grid_x,grid_y,pos_x,pos_y (with header)
    
    Args:
        records: List of PositionRecordData objects
        
    Returns:
        CSV content as string
    """
    if not records:
        return "filename,grid_x,grid_y,pos_x,pos_y\n" # Header for empty file
    
    # Validate each record before serializing
    dict_records = []
    for record in records:
        # This will validate the record
        dict_records.append(record.to_dict())
    
    df = pd.DataFrame(dict_records)
    # Ensure correct column order
    df = df[["filename", "grid_x", "grid_y", "pos_x", "pos_y"]]
    return df.to_csv(index=False)

def _serialize_kv_semicolon_csv(records: List[PositionRecordData]) -> str:
    """
    Serialize to key-value semicolon CSV format.
    
    Format: file: ...; position: (...); grid: (...)
    
    Args:
        records: List of PositionRecordData objects
        
    Returns:
        CSV content as string
    """
    lines = []
    for record in records:
        # Validate record before serializing
        record.to_dict()
        
        # Always include grid info if available, even if it was 0,0 from parsing
        lines.append(
            f"file: {record.filename}; position: ({record.pos_x}, {record.pos_y}); grid: ({record.grid_x}, {record.grid_y})"
        )
    return "\n".join(lines)


# Central registry for CSV formats
POSITION_CSV_FORMATS: Dict[CSVPositionFormat, PositionCSVFormatSpec] = {
    CSVPositionFormat.STANDARD: PositionCSVFormatSpec(
        format_name="Standard CSV",
        format_description="Standard CSV format with header: filename,grid_x,grid_y,pos_x,pos_y",
        delimiter=",",
        has_header=True,
        parser=_parse_standard_csv,
        serializer=_serialize_standard_csv
    ),
    CSVPositionFormat.KV_SEMICOLON: PositionCSVFormatSpec(
        format_name="Key-Value Semicolon",
        format_description="Key-value format with semicolons: file: ...; position: (...); grid: (...)",
        delimiter=";", # Not strictly a CSV delimiter, parsing is regex based
        has_header=False,
        parser=_parse_kv_semicolon_csv,
        serializer=_serialize_kv_semicolon_csv,
        pattern=r"file:\s*(?P<filename>[^;]+);\s*position:\s*\((?P<pos_x>[^,]+),\s*(?P<pos_y>[^)]+)\)(?:;\s*grid:\s*\((?P<grid_x>[^,]+),\s*(?P<grid_y>[^)]+)\))?"
    )
}

def get_format_spec(format_enum: CSVPositionFormat) -> PositionCSVFormatSpec:
    """
    Retrieve the format specification for a given CSVPositionFormat enum.
    
    Args:
        format_enum: CSVPositionFormat enum value
        
    Returns:
        PositionCSVFormatSpec for the given format
        
    Raises:
        ValueError: If format_enum is not a valid CSVPositionFormat
    """
    # Validate format_enum is a valid CSVPositionFormat
    if not isinstance(format_enum, CSVPositionFormat):
        raise ValueError(f"Invalid format enum: {format_enum}. Must be a CSVPositionFormat enum value.")
    
    # No .get() with default, direct access ensures key exists or raises KeyError
    # This is doctrinally pure as the enum guarantees valid keys if used correctly.
    try:
        format_spec = POSITION_CSV_FORMATS[format_enum]
        
        # Validate format spec against schema
        PositionCSVFormatSpec.SCHEMA.validate({
            "format_name": format_spec.format_name,
            "format_description": format_spec.format_description,
            "delimiter": format_spec.delimiter,
            "has_header": format_spec.has_header,
            "parser": format_spec.parser,
            "serializer": format_spec.serializer,
            "pattern": format_spec.pattern
        })
        
        return format_spec
    except KeyError:
        # This should ideally not happen if CSVPositionFormat enum is exhaustive
        # and used correctly as type hint.
        raise ValueError(f"Unsupported CSVPositionFormat: {format_enum}")