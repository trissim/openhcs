"""
CPPipeParser - Parse CellProfiler .cppipe pipeline files.

Parses the custom .cppipe format (not XML, but a custom text format) into
structured ModuleBlock dataclasses for conversion to OpenHCS.

Format example:
    ModuleName:[module_num:5|svn_version:'Unknown'|...]
        Setting Name:Value
        Another Setting:Another Value
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from openhcs.constants import Backend
from openhcs.core.vfs_protocol import FileManagerLike

from .cellprofiler_literals import decode_cellprofiler_setting_literal

logger = logging.getLogger(__name__)

CellProfilerMetadataValue = str | tuple[dict[str, str | None], ...]


@dataclass(frozen=True, slots=True)
class ModuleSetting:
    """One ordered CellProfiler module setting."""

    name: str
    value: str

    def __post_init__(self) -> None:
        normalized_name = decode_cellprofiler_setting_literal(self.name).strip()
        if not normalized_name:
            raise ValueError("ModuleSetting.name cannot be empty.")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "value",
            decode_cellprofiler_setting_literal(self.value).strip(),
        )


@dataclass
class ModuleBlock:
    """Represents a single CellProfiler module from a .cppipe file."""

    name: str  # e.g., "IdentifyPrimaryObjects"
    module_num: int  # Position in pipeline
    enabled: bool = True
    settings: dict[str, str] = field(default_factory=dict)
    setting_records: list[ModuleSetting] = field(default_factory=list)
    metadata: dict[str, CellProfilerMetadataValue] = field(default_factory=dict)
    
    @property
    def library_module_name(self) -> str:
        """Convert module name to library module filename (lowercase with underscore prefix)."""
        # IdentifyPrimaryObjects -> _identifyprimaryobjects
        return f"_{self.name.lower()}"
    
    def get_setting(self, key: str, default: str = "") -> str:
        """Get a setting value by key."""
        return self.settings.get(key, default)

    @property
    def variable_revision_number(self) -> int | None:
        """Return the CellProfiler module schema revision when declared."""
        value = self.metadata.get("variable_revision_number")
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def get_setting_values(self, key: str) -> tuple[str, ...]:
        """Get all values for a setting key in .cppipe order."""
        normalized_key = key.strip()
        return tuple(
            setting.value
            for setting in self.setting_records
            if setting.name == normalized_key
        )

    def iter_settings(self, key: str | None = None) -> tuple[ModuleSetting, ...]:
        """Iterate ordered typed settings, optionally filtered by key."""
        if key is None:
            return tuple(self.setting_records)
        normalized_key = key.strip()
        return tuple(
            setting
            for setting in self.setting_records
            if setting.name == normalized_key
        )


class CPPipeParser:
    """
    Parser for CellProfiler .cppipe pipeline files.
    
    The .cppipe format is a custom text format (not XML) with:
    - Header lines (CellProfiler Pipeline, Version, etc.)
    - Module blocks starting with ModuleName:[metadata]
    - Indented setting lines: "    Setting Name:Value"
    """
    
    # Pattern for module header line: ModuleName:[metadata]
    # The metadata can contain nested brackets like array([], dtype=uint8)
    MODULE_HEADER_PATTERN = re.compile(
        r'^\s*(\w+):\[(.+)\]$'
    )
    
    # Pattern for module metadata parsing
    METADATA_PATTERN = re.compile(
        r'(\w+):([^|]+)'
    )
    
    # Pattern for setting line (4 spaces + Setting Name:Value)
    SETTING_PATTERN = re.compile(
        r'^    ([^:]+):(.*)$'
    )

    IMAGE_PLANE_DETAILS_PATTERN = re.compile(
        r'^"Version":"(?P<version>[^"]+)","PlaneCount":"(?P<count>\d+)"$'
    )

    # Older .pipeline resources store module settings without indentation.
    UNINDENTED_SETTING_PATTERN = re.compile(
        r'^([^:]+):(.*)$'
    )
    
    def __init__(self, cppipe_path: Path | None = None):
        """
        Initialize parser.
        
        Args:
            cppipe_path: Path to .cppipe file (can also pass to parse())
        """
        self.cppipe_path = Path(cppipe_path) if cppipe_path else None
        self.modules: list[ModuleBlock] = []
        self.header: dict[str, str] = {}
        self.image_plane_sources: tuple[dict[str, str | None], ...] = ()
    
    def parse(
        self,
        cppipe_path: Path | None = None,
        *,
        filemanager: FileManagerLike | None = None,
        backend: Backend = Backend.DISK,
    ) -> list[ModuleBlock]:
        """
        Parse a .cppipe file and return list of ModuleBlock.
        
        Args:
            cppipe_path: Path to .cppipe file (uses self.cppipe_path if None)
            
        Returns:
            List of ModuleBlock dataclasses
        """
        path = Path(cppipe_path) if cppipe_path else self.cppipe_path
        if not path:
            raise ValueError("No .cppipe path provided")
        
        if not isinstance(backend, Backend):
            raise TypeError(
                "CPPipeParser.parse backend must be an openhcs.constants.Backend, "
                f"got {type(backend).__name__}."
            )

        if filemanager is None and not path.exists():
            raise FileNotFoundError(f".cppipe file not found: {path}")
        
        logger.info(f"Parsing .cppipe file: {path}")
        
        content = self._read_cppipe_text(path, filemanager=filemanager, backend=backend)
        lines = content.split('\n')
        
        self.modules = []
        self.header = {}
        self.image_plane_sources = ()
        current_module: ModuleBlock | None = None
        
        for line in lines:
            # Check for module header
            header_match = self.MODULE_HEADER_PATTERN.match(line)
            if header_match:
                # Save previous module
                if current_module:
                    self.modules.append(current_module)
                
                # Parse new module
                module_name = header_match.group(1)
                metadata_str = header_match.group(2)
                metadata = self._parse_metadata(metadata_str)
                
                current_module = ModuleBlock(
                    name=module_name,
                    module_num=int(metadata.get('module_num', 0)),
                    enabled=metadata.get('enabled', 'True') == 'True',
                    metadata=metadata
                )
                continue

            # Indented module settings are authoritative even when the setting
            # label itself starts with "#", e.g. CellProfiler's "# of deviations".
            setting_match = self.SETTING_PATTERN.match(line)
            if setting_match and current_module:
                if self._has_setting_name(setting_match):
                    setting = ModuleSetting(
                        name=setting_match.group(1),
                        value=setting_match.group(2),
                    )
                    current_module.setting_records.append(setting)
                    current_module.settings[setting.name] = setting.value
                continue

            # Skip comments
            if line.strip().startswith('#'):
                continue

            # Skip empty lines
            if not line.strip():
                continue
            
            # Check for setting line. Real CellProfiler corpora include both
            # indented .cppipe settings and unindented legacy .pipeline settings.
            setting_match = self.UNINDENTED_SETTING_PATTERN.match(line)
            if self._has_setting_name(setting_match) and current_module:
                setting = ModuleSetting(
                    name=setting_match.group(1),
                    value=setting_match.group(2),
                )
                current_module.setting_records.append(setting)
                current_module.settings[setting.name] = setting.value
                continue
            
            # Header line (key:value without module bracket)
            if ':' in line and not line.startswith(' '):
                parts = line.split(':', 1)
                if len(parts) == 2:
                    self.header[parts[0].strip()] = parts[1].strip()
        
        # Don't forget the last module
        if current_module:
            self.modules.append(current_module)

        self.image_plane_sources = self._parse_image_plane_sources(lines)
        if self.image_plane_sources:
            for module in self.modules:
                module.metadata["image_plane_sources"] = self.image_plane_sources
        
        logger.info(f"Parsed {len(self.modules)} modules from {path.name}")
        return self.modules

    @staticmethod
    def _read_cppipe_text(
        path: Path,
        *,
        filemanager: FileManagerLike | None,
        backend: Backend,
    ) -> str:
        if filemanager is None:
            return path.read_text()
        content = filemanager.load(str(path), backend.value)
        if isinstance(content, bytes):
            return content.decode()
        if not isinstance(content, str):
            raise TypeError(
                "CPPipeParser expected FileManager.load to return str or bytes "
                f"for {path}, got {type(content).__name__}."
            )
        return content

    def _setting_match(
        self,
        line: str,
        current_module: ModuleBlock | None,
    ) -> re.Match[str] | None:
        if current_module is None:
            return None
        setting_match = self.SETTING_PATTERN.match(line)
        if self._has_setting_name(setting_match):
            return setting_match

        setting_match = self.UNINDENTED_SETTING_PATTERN.match(line)
        if self._has_setting_name(setting_match):
            return setting_match
        return None

    def _has_setting_name(
        self,
        setting_match: re.Match[str] | None,
    ) -> bool:
        return setting_match is not None and bool(setting_match.group(1).strip())
    
    def _parse_metadata(self, metadata_str: str) -> dict[str, str]:
        """Parse module metadata from bracket content."""
        metadata = {}
        for match in self.METADATA_PATTERN.finditer(metadata_str):
            key = match.group(1)
            value = match.group(2).strip().strip("'")
            metadata[key] = value
        return metadata

    def _parse_image_plane_sources(
        self,
        lines: list[str],
    ) -> tuple[dict[str, str | None], ...]:
        """Parse CellProfiler's optional embedded image-plane details table."""

        for index, line in enumerate(lines):
            version_match = self.IMAGE_PLANE_DETAILS_PATTERN.match(line.strip())
            if version_match is None:
                continue
            header_index = index + 1
            if header_index >= len(lines):
                return ()
            header = self._csv_image_plane_row(lines[header_index])
            if not header or header[0] != "URL":
                return ()
            plane_sources: list[dict[str, str | None]] = []
            expected_count = int(version_match.group("count"))
            for row_line in lines[header_index + 1 :]:
                if not row_line.strip():
                    continue
                row = self._csv_image_plane_row(row_line)
                if not row:
                    continue
                values = {
                    name: (value.strip() or None)
                    for name, value in zip(header, row, strict=False)
                }
                uri = values.get("URL")
                if not uri:
                    continue
                plane_sources.append(
                    {
                        "uri": uri,
                        "series": values.get("Series"),
                        "index": values.get("Index"),
                        "channel": values.get("Channel"),
                    }
                )
                if len(plane_sources) == expected_count:
                    break
            return tuple(plane_sources)
        return ()

    def _csv_image_plane_row(self, line: str) -> list[str]:
        import csv

        return next(csv.reader([line]))
    
    def get_module_by_name(self, name: str) -> ModuleBlock | None:
        """Get a module by name (case-insensitive)."""
        name_lower = name.lower()
        for module in self.modules:
            if module.name.lower() == name_lower:
                return module
        return None
    
    def get_enabled_modules(self) -> list[ModuleBlock]:
        """Get only enabled modules."""
        return [m for m in self.modules if m.enabled]
