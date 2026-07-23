"""Generic path-template matching for OpenHCS virtual filenames."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PathPatternTemplateMatcher:
    """Matcher for path patterns containing OpenHCS placeholder fields."""

    pattern: str
    regex: re.Pattern[str]

    @classmethod
    def from_pattern(
        cls,
        pattern: str,
    ) -> "PathPatternTemplateMatcher | None":
        if "{" not in pattern or "}" not in pattern:
            return None

        regex_parts: list[str] = []
        cursor = 0
        for match in re.finditer(r"\{[^{}]+\}", pattern):
            regex_parts.append(re.escape(pattern[cursor : match.start()]))
            regex_parts.append(r"[^/]*")
            cursor = match.end()
        regex_parts.append(re.escape(pattern[cursor:]))
        return cls(
            pattern=pattern,
            regex=re.compile(rf"^{''.join(regex_parts)}$"),
        )

    def matches(self, path: str) -> bool:
        candidate = Path(path)
        return bool(
            self.regex.match(path)
            or self.regex.match(candidate.as_posix())
            or self.regex.match(candidate.name)
        )
