"""
Recommendation module for semantic grounding.

This module provides functionality for grounding AI recommendations in evidence from the codebase.
"""

import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.grounding.evidence import Evidence, EvidenceCollector

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """An AI recommendation grounded in evidence from the codebase."""
    
    id: str
    type: str  # "refactoring", "bug_fix", "optimization", etc.
    title: str
    description: str
    evidence_ids: List[str]
    confidence: float
    file_paths: List[Path]
    line_ranges: List[Tuple[int, int]]
    suggested_code: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
            "file_paths": [str(path) for path in self.file_paths],
            "line_ranges": self.line_ranges,
            "metadata": self.metadata
        }
        
        if self.suggested_code:
            result["suggested_code"] = self.suggested_code
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Recommendation':
        """Create from dictionary after deserialization."""
        return cls(
            id=data["id"],
            type=data["type"],
            title=data["title"],
            description=data["description"],
            evidence_ids=data["evidence_ids"],
            confidence=data["confidence"],
            file_paths=[Path(path) for path in data["file_paths"]],
            line_ranges=data["line_ranges"],
            suggested_code=data.get("suggested_code"),
            metadata=data.get("metadata", {})
        )
    
    def get_location_str(self) -> str:
        """Get a string representation of the recommendation location."""
        if not self.file_paths:
            return "Unknown location"
        
        if len(self.file_paths) == 1 and len(self.line_ranges) == 1:
            return f"{self.file_paths[0]}:{self.line_ranges[0][0]}-{self.line_ranges[0][1]}"
        
        return f"Multiple locations ({len(self.file_paths)} files)"
    
    def get_summary(self) -> str:
        """Get a summary of the recommendation."""
        return f"{self.type} recommendation: {self.title} ({self.confidence:.2f})"


class RecommendationGrounder:
    """Grounds AI recommendations in evidence from the codebase."""
    
    def __init__(self, evidence_collector: EvidenceCollector):
        """Initialize the recommendation grounder.
        
        Args:
            evidence_collector: The evidence collector.
        """
        self.evidence_collector = evidence_collector
        self.recommendations: Dict[str, Recommendation] = {}
    
    def ground_recommendation(
        self,
        type: str,
        title: str,
        description: str,
        file_paths: List[Path],
        line_ranges: List[Tuple[int, int]],
        suggested_code: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Recommendation:
        """Ground a recommendation in evidence from the codebase.
        
        Args:
            type: The recommendation type.
            title: The recommendation title.
            description: The recommendation description.
            file_paths: The file paths affected by the recommendation.
            line_ranges: The line ranges affected by the recommendation.
            suggested_code: The suggested code (optional).
            metadata: Additional metadata (optional).
            
        Returns:
            The grounded recommendation.
        """
        # Collect evidence for the files
        all_evidence = []
        for file_path in file_paths:
            evidence = self.evidence_collector.collect_evidence_for_file(file_path)
            all_evidence.extend(evidence)
        
        # Filter evidence to the line ranges
        filtered_evidence = []
        for evidence in all_evidence:
            for file_path, (line_start, line_end) in zip(file_paths, line_ranges):
                if evidence.file_path == file_path and evidence.line_start >= line_start and evidence.line_end <= line_end:
                    filtered_evidence.append(evidence)
        
        # Calculate confidence based on evidence
        confidence = self._calculate_confidence(filtered_evidence)
        
        # Create the recommendation
        recommendation = Recommendation(
            id=str(uuid.uuid4()),
            type=type,
            title=title,
            description=description,
            evidence_ids=[evidence.id for evidence in filtered_evidence],
            confidence=confidence,
            file_paths=file_paths,
            line_ranges=line_ranges,
            suggested_code=suggested_code,
            metadata=metadata or {}
        )
        
        # Store the recommendation
        self.recommendations[recommendation.id] = recommendation
        
        return recommendation
    
    def _calculate_confidence(self, evidence: List[Evidence]) -> float:
        """Calculate confidence based on evidence.
        
        Args:
            evidence: The evidence.
            
        Returns:
            The confidence score (0.0 to 1.0).
        """
        if not evidence:
            return 0.0
        
        # Calculate the average confidence of the evidence
        evidence_confidence = sum(e.confidence for e in evidence) / len(evidence)
        
        # Adjust based on the amount of evidence
        evidence_count_factor = min(1.0, len(evidence) / 5.0)
        
        # Combine the factors
        confidence = evidence_confidence * evidence_count_factor
        
        return confidence
    
    def get_recommendation(self, recommendation_id: str) -> Optional[Recommendation]:
        """Get a recommendation by ID.
        
        Args:
            recommendation_id: The recommendation ID.
            
        Returns:
            The recommendation, or None if not found.
        """
        return self.recommendations.get(recommendation_id)
    
    def get_recommendations_by_type(self, type: str) -> List[Recommendation]:
        """Get recommendations by type.
        
        Args:
            type: The recommendation type.
            
        Returns:
            A list of recommendations of the specified type.
        """
        return [r for r in self.recommendations.values() if r.type == type]
    
    def get_recommendations_for_file(self, file_path: Path) -> List[Recommendation]:
        """Get recommendations for a file.
        
        Args:
            file_path: The file path.
            
        Returns:
            A list of recommendations for the specified file.
        """
        return [r for r in self.recommendations.values() if file_path in r.file_paths]
    
    def get_evidence_for_recommendation(self, recommendation_id: str) -> List[Evidence]:
        """Get evidence for a recommendation.
        
        Args:
            recommendation_id: The recommendation ID.
            
        Returns:
            A list of evidence items for the recommendation.
        """
        recommendation = self.get_recommendation(recommendation_id)
        if not recommendation:
            return []
        
        evidence = []
        for evidence_id in recommendation.evidence_ids:
            evidence_item = self.evidence_collector.get_evidence(evidence_id)
            if evidence_item:
                evidence.append(evidence_item)
        
        return evidence
