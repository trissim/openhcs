"""Runtime equivalence report records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RuntimeEquivalenceDifferenceKind(str, Enum):
    """Closed families of semantic runtime-output differences."""

    RUNTIME_ARTIFACT_COUNTS = "runtime_artifact_counts"
    MEASUREMENT_FEATURE = "measurement_feature"
    MEASUREMENT_CONTENT = "measurement_content"
    TABLE_SCHEMA = "table_schema"
    TABLE_COUNT = "table_count"
    TABLE_CONTENT = "table_content"
    IMAGE_COUNT = "image_count"
    IMAGE_CONTENT = "image_content"


@dataclass(frozen=True, slots=True)
class RuntimeEquivalenceDifference:
    """One semantic difference between two runtime outputs."""

    kind: RuntimeEquivalenceDifferenceKind
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            (
                self.kind
                if isinstance(self.kind, RuntimeEquivalenceDifferenceKind)
                else RuntimeEquivalenceDifferenceKind(self.kind)
            ),
        )


@dataclass(frozen=True, slots=True)
class RuntimeEquivalenceReport:
    """Semantic equivalence result for two runtime outputs."""

    differences: tuple[RuntimeEquivalenceDifference, ...]

    @property
    def is_equivalent(self) -> bool:
        """Return whether the compared outputs are semantically equivalent."""
        return not self.differences

    def failure_messages(self) -> tuple[str, ...]:
        """Return stable human-readable failure messages."""
        return tuple(difference.message for difference in self.differences)
