"""Filename parsing bound to the canonical OpenHCS component declaration."""

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, TypeAlias

from polystore.streaming.viewer_transport import ViewerFilenameParseResultABC

from openhcs.constants.constants import AllComponents
from openhcs.core.components.component_values import OpenHCSComponentValues

logger = logging.getLogger(__name__)

FilenameParseValue: TypeAlias = str | int | float | bool | None


@dataclass(frozen=True, slots=True, init=False)
class FilenameParseResult(ViewerFilenameParseResultABC):
    """Immutable parser result keyed by nominal component declarations."""

    components: OpenHCSComponentValues[FilenameParseValue]
    extension: str

    def __init__(
        self,
        component_values: Iterable[tuple[AllComponents, FilenameParseValue]],
        *,
        extension: str,
    ) -> None:
        normalized_extension = str(extension).strip()
        if not normalized_extension:
            raise ValueError("Filename extension cannot be empty")
        if not normalized_extension.startswith("."):
            normalized_extension = f".{normalized_extension}"

        object.__setattr__(self, "components", OpenHCSComponentValues(component_values))
        object.__setattr__(self, "extension", normalized_extension)

    @classmethod
    def from_projection(
        cls,
        source_values: Iterable[tuple[Enum, FilenameParseValue]],
        *,
        extension: str,
    ) -> "FilenameParseResult":
        """Project one nominal component enum onto another by declared member name."""

        return cls.from_components(
            OpenHCSComponentValues.from_member_projection(source_values),
            extension=extension,
        )

    @classmethod
    def from_components(
        cls,
        components: OpenHCSComponentValues[FilenameParseValue],
        *,
        extension: str,
    ) -> "FilenameParseResult":
        """Bind canonical component values to one filename extension."""

        return cls(components.declared_values(), extension=extension)

    @classmethod
    def from_wire_mapping(
        cls,
        component_values: Mapping[str, FilenameParseValue],
        *,
        extension: str,
    ) -> "FilenameParseResult":
        """Create a nominal result at an explicit keyword or wire boundary."""

        return cls(
            (
                (component, component_values.get(component.value))
                for component in AllComponents
            ),
            extension=extension,
        )

    def declared_values(
        self,
    ) -> tuple[tuple[AllComponents, FilenameParseValue], ...]:
        """Return parsed values in declaration order."""

        return self.components.declared_values()

    def value_for(self, component: AllComponents) -> FilenameParseValue:
        """Return the value owned by one exact component declaration."""

        return self.components.value_for(component)

    def with_value(
        self,
        component: AllComponents,
        value: FilenameParseValue,
    ) -> "FilenameParseResult":
        """Return a result with one nominal component value replaced."""

        return type(self).from_components(
            self.components.with_value(component, value),
            extension=self.extension,
        )

    def with_values(
        self,
        replacements: Iterable[tuple[AllComponents, FilenameParseValue]],
    ) -> "FilenameParseResult":
        """Return a result with nominally declared component replacements."""

        components = self.components
        seen: set[AllComponents] = set()
        for component, value in replacements:
            if component in seen:
                raise ValueError(
                    f"Filename component {component.value!r} was replaced more than once"
                )
            seen.add(component)
            components = components.with_value(component, value)
        return type(self).from_components(components, extension=self.extension)

    def component_matches(
        self,
        component: AllComponents,
        expected_value: object,
    ) -> bool:
        """Compare one parsed component through its nominal declaration."""

        parsed_value = self.value_for(component)
        if parsed_value is None or expected_value is None:
            return parsed_value is expected_value
        return str(parsed_value) == str(expected_value)

    def required_value(self, component: AllComponents) -> str | int | float | bool:
        """Return one required component or fail with its declared identity."""

        value = self.value_for(component)
        if value is None or value == "":
            raise MissingFilenameComponentError(component.value, empty=True)
        return value

    def wire_mapping(self) -> Mapping[str, FilenameParseValue]:
        """Project this nominal result at an explicit string-keyed wire boundary."""

        return {**self.components.wire_mapping(), "extension": self.extension}

    def component_wire_mapping(self) -> Mapping[str, FilenameParseValue]:
        """Project only component values for metadata and viewer boundaries."""

        return self.components.wire_mapping()

    def __hash__(self) -> int:
        return hash((self.components.declared_values(), self.extension))


class MissingFilenameComponentError(ValueError):
    """A filename parser cannot construct a name without one component."""

    def __init__(self, component_name: str, *, empty: bool = False):
        self.component_name = component_name
        detail = "cannot be empty" if empty else "is required"
        super().__init__(f"Filename component {component_name!r} {detail}.")


def format_filename_component(value: object, padding: int = 0) -> str:
    """Format a declared filename component without inventing missing values."""
    if isinstance(value, str):
        return value
    return f"{int(value):0{padding}d}" if padding else str(int(value))


class GenericFilenameParser(ABC):
    """Filename parser bound to the canonical OpenHCS component declaration."""

    DEFAULT_EXTENSION = ".tif"

    def __init__(self) -> None:
        self.FILENAME_COMPONENTS = tuple(AllComponents)
        self.PLACEHOLDER_PATTERN = "{iii}"

    @classmethod
    @abstractmethod
    def can_parse(cls, filename: str) -> bool:
        """Check if this parser can parse the given filename."""
        pass

    @abstractmethod
    def extract_component_coordinates(self, component_value: str) -> Tuple[str, str]:
        """Extract coordinates from component identifier (typically well)."""
        pass

    @abstractmethod
    def parse_filename(self, filename: str) -> Optional[FilenameParseResult]:
        """Parse a filename to extract all components."""
        pass

    @abstractmethod
    def construct_filename(self, components: FilenameParseResult) -> str:
        """Construct a filename from one nominal component result."""
        pass

    def get_component_names(self) -> tuple[str, ...]:
        """Get all component names for this parser."""
        return AllComponents.ordered_names()

    def component_for_name(self, component_name: str) -> AllComponents:
        """Resolve one external component name through the declared enum."""

        return AllComponents(component_name)

    def bind_component_values(
        self,
        component_values: Mapping[str, FilenameParseValue],
        *,
        extension: str | None = None,
    ) -> FilenameParseResult:
        """Bind an external component mapping before filename construction."""

        return FilenameParseResult.from_wire_mapping(
            component_values,
            extension=extension or self.DEFAULT_EXTENSION,
        )

    def bind_declared_values(
        self,
        component_values: Iterable[tuple[AllComponents, FilenameParseValue]],
        *,
        extension: str | None = None,
    ) -> FilenameParseResult:
        """Bind canonical component values without crossing a string boundary."""

        return FilenameParseResult(
            component_values,
            extension=extension or self.DEFAULT_EXTENSION,
        )

    @staticmethod
    def validate_component_value(value: FilenameParseValue) -> bool:
        """Validate one generic parsed value without a mirrored component table."""

        if value is None:
            return True
        if isinstance(value, str):
            return bool(value) or "{" in value
        if isinstance(value, int):
            return value >= 0
        return isinstance(value, (float, bool))

    def extract_component(
        self,
        filename: str,
        component: AllComponents,
    ) -> FilenameParseValue:
        """Extract one exact nominal component from a parsed filename."""

        parsed = self.parse_filename(filename)
        return None if parsed is None else parsed.value_for(component)

    def validate_parse_result(self, result: FilenameParseResult) -> bool:
        """Validate a complete result against this parser's declaration."""

        declared_components = tuple(AllComponents)
        if (
            tuple(component for component, _ in result.declared_values())
            != declared_components
        ):
            logger.warning("Parsed filename components do not match parser declaration")
            return False
        for component, value in result.declared_values():
            if not self.validate_component_value(value):
                logger.warning("Invalid value for %s: %r", component.value, value)
                return False
        return True
