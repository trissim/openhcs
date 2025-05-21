"""
Data models for agent-driven analysis.

This module provides data models for the agent-driven analyzer.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union


class FileRelevance(Enum):
    """Relevance levels for files."""
    
    CRITICAL = 5.0  # Files that are central to the user's concerns
    HIGH = 3.0      # Files that are highly relevant
    MEDIUM = 1.0    # Files that are moderately relevant
    LOW = 0.5       # Files that are somewhat relevant
    IRRELEVANT = 0.0  # Files that are not relevant


class FileComplexity(Enum):
    """Complexity levels for files."""
    
    VERY_HIGH = 5.0  # Extremely complex files
    HIGH = 3.0       # Highly complex files
    MEDIUM = 1.0     # Moderately complex files
    LOW = 0.5        # Simple files
    TRIVIAL = 0.1    # Trivial files


@dataclass
class FileAnalysisHistory:
    """History of file analysis."""
    
    file_path: Path
    times_selected: int = 0
    times_useful: int = 0
    last_relevance_score: float = 0.0
    last_information_value: float = 0.0
    last_effort_score: float = 1.0
    
    @property
    def usefulness_ratio(self) -> float:
        """Calculate the ratio of times the file was useful."""
        if self.times_selected == 0:
            return 0.0
        return self.times_useful / self.times_selected


@dataclass
class FileAnalysisMetrics:
    """Metrics for file analysis."""
    
    file_path: Path
    size_bytes: int
    line_count: int
    complexity: FileComplexity = FileComplexity.MEDIUM
    dependencies: List[Path] = field(default_factory=list)
    dependents: List[Path] = field(default_factory=list)
    change_frequency: float = 0.0  # Changes per month
    
    @property
    def is_large(self) -> bool:
        """Check if the file is large."""
        return self.line_count > 500
    
    @property
    def is_complex(self) -> bool:
        """Check if the file is complex."""
        return self.complexity in (FileComplexity.HIGH, FileComplexity.VERY_HIGH)
    
    @property
    def is_central(self) -> bool:
        """Check if the file is central to the codebase."""
        return len(self.dependents) > 5


@dataclass
class UserIntent:
    """User's intent for analysis."""
    
    primary_concerns: List[str]
    file_mentions: List[Path] = field(default_factory=list)
    component_mentions: List[str] = field(default_factory=list)
    pattern_mentions: List[str] = field(default_factory=list)
    excluded_files: List[Path] = field(default_factory=list)
    excluded_patterns: List[str] = field(default_factory=list)
    
    def is_file_relevant(self, file_path: Path) -> bool:
        """Check if a file is relevant to the user's intent."""
        # Explicitly excluded files are not relevant
        if file_path in self.excluded_files:
            return False
        
        # Explicitly mentioned files are relevant
        if file_path in self.file_mentions:
            return True
        
        # Files that match mentioned components are relevant
        for component in self.component_mentions:
            if component.lower() in file_path.name.lower():
                return True
        
        # Default to True - the agent will calculate the actual relevance score
        return True


@dataclass
class FileSelectionResult:
    """Result of file selection."""
    
    file_path: Path
    score: float
    relevance: float
    information_value: float
    effort: float
    
    def to_dict(self) -> Dict[str, Union[str, float]]:
        """Convert to dictionary."""
        return {
            "file_path": str(self.file_path),
            "score": self.score,
            "relevance": self.relevance,
            "information_value": self.information_value,
            "effort": self.effort
        }
