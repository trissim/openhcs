"""
Verification module for semantic grounding.

This module provides functionality for verifying that AI recommendations and findings
are grounded in evidence from the codebase.
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.grounding.evidence import Evidence, EvidenceCollector
from semantic_matrix_analyzer.grounding.recommendation import Recommendation, RecommendationGrounder

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """The result of verifying a recommendation."""
    
    recommendation_id: str
    is_grounded: bool
    evidence_coverage: float
    confidence: float
    issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommendation_id": self.recommendation_id,
            "is_grounded": self.is_grounded,
            "evidence_coverage": self.evidence_coverage,
            "confidence": self.confidence,
            "issues": self.issues
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VerificationResult':
        """Create from dictionary after deserialization."""
        return cls(
            recommendation_id=data["recommendation_id"],
            is_grounded=data["is_grounded"],
            evidence_coverage=data["evidence_coverage"],
            confidence=data["confidence"],
            issues=data["issues"]
        )


class GroundingVerifier:
    """Verifies that recommendations are grounded in evidence."""
    
    def __init__(self, evidence_collector: EvidenceCollector, recommendation_grounder: RecommendationGrounder):
        """Initialize the grounding verifier.
        
        Args:
            evidence_collector: The evidence collector.
            recommendation_grounder: The recommendation grounder.
        """
        self.evidence_collector = evidence_collector
        self.recommendation_grounder = recommendation_grounder
        self.verification_results: Dict[str, VerificationResult] = {}
    
    def verify_recommendation(self, recommendation: Recommendation) -> VerificationResult:
        """Verify that a recommendation is grounded in evidence.
        
        Args:
            recommendation: The recommendation to verify.
            
        Returns:
            The verification result.
        """
        # Get the evidence for the recommendation
        evidence = self.recommendation_grounder.get_evidence_for_recommendation(recommendation.id)
        
        # Check if there is any evidence
        if not evidence:
            return VerificationResult(
                recommendation_id=recommendation.id,
                is_grounded=False,
                evidence_coverage=0.0,
                confidence=0.0,
                issues=["No evidence found for recommendation"]
            )
        
        # Calculate evidence coverage
        evidence_coverage = self._calculate_evidence_coverage(recommendation, evidence)
        
        # Check for issues
        issues = self._check_for_issues(recommendation, evidence)
        
        # Calculate confidence
        confidence = self._calculate_confidence(recommendation, evidence, evidence_coverage, issues)
        
        # Determine if the recommendation is grounded
        is_grounded = confidence >= 0.5 and evidence_coverage >= 0.5 and not issues
        
        # Create the verification result
        result = VerificationResult(
            recommendation_id=recommendation.id,
            is_grounded=is_grounded,
            evidence_coverage=evidence_coverage,
            confidence=confidence,
            issues=issues
        )
        
        # Store the result
        self.verification_results[recommendation.id] = result
        
        return result
    
    def _calculate_evidence_coverage(self, recommendation: Recommendation, evidence: List[Evidence]) -> float:
        """Calculate the evidence coverage for a recommendation.
        
        Args:
            recommendation: The recommendation.
            evidence: The evidence for the recommendation.
            
        Returns:
            The evidence coverage (0.0 to 1.0).
        """
        # Calculate the total number of lines in the recommendation
        total_lines = sum(end - start + 1 for start, end in recommendation.line_ranges)
        
        # Calculate the number of lines covered by evidence
        covered_lines = set()
        for e in evidence:
            for line in range(e.line_start, e.line_end + 1):
                covered_lines.add((str(e.file_path), line))
        
        # Calculate the coverage
        if total_lines == 0:
            return 0.0
        
        return len(covered_lines) / total_lines
    
    def _check_for_issues(self, recommendation: Recommendation, evidence: List[Evidence]) -> List[str]:
        """Check for issues with a recommendation.
        
        Args:
            recommendation: The recommendation.
            evidence: The evidence for the recommendation.
            
        Returns:
            A list of issues.
        """
        issues = []
        
        # Check if there is enough evidence
        if len(evidence) < 3:
            issues.append(f"Insufficient evidence: only {len(evidence)} evidence items found")
        
        # Check if the evidence is relevant
        for e in evidence:
            if not self._is_evidence_relevant(recommendation, e):
                issues.append(f"Irrelevant evidence: {e.get_location_str()}")
        
        # Check if the recommendation is consistent with the evidence
        if not self._is_recommendation_consistent(recommendation, evidence):
            issues.append("Recommendation is inconsistent with evidence")
        
        return issues
    
    def _is_evidence_relevant(self, recommendation: Recommendation, evidence: Evidence) -> bool:
        """Check if evidence is relevant to a recommendation.
        
        Args:
            recommendation: The recommendation.
            evidence: The evidence.
            
        Returns:
            True if the evidence is relevant, False otherwise.
        """
        # Check if the evidence is in one of the recommendation's files
        if evidence.file_path not in recommendation.file_paths:
            return False
        
        # Check if the evidence is in one of the recommendation's line ranges
        for file_path, (start, end) in zip(recommendation.file_paths, recommendation.line_ranges):
            if evidence.file_path == file_path and evidence.line_start >= start and evidence.line_end <= end:
                return True
        
        return False
    
    def _is_recommendation_consistent(self, recommendation: Recommendation, evidence: List[Evidence]) -> bool:
        """Check if a recommendation is consistent with evidence.
        
        Args:
            recommendation: The recommendation.
            evidence: The evidence.
            
        Returns:
            True if the recommendation is consistent with the evidence, False otherwise.
        """
        # This is a placeholder implementation
        # In a real implementation, we would perform more sophisticated consistency checks
        
        return True
    
    def _calculate_confidence(
        self,
        recommendation: Recommendation,
        evidence: List[Evidence],
        evidence_coverage: float,
        issues: List[str]
    ) -> float:
        """Calculate the confidence for a verification result.
        
        Args:
            recommendation: The recommendation.
            evidence: The evidence for the recommendation.
            evidence_coverage: The evidence coverage.
            issues: The issues with the recommendation.
            
        Returns:
            The confidence score (0.0 to 1.0).
        """
        # Start with the recommendation's confidence
        confidence = recommendation.confidence
        
        # Adjust based on evidence coverage
        confidence *= evidence_coverage
        
        # Adjust based on issues
        confidence *= max(0.1, 1.0 - (len(issues) * 0.2))
        
        return confidence
    
    def get_verification_result(self, recommendation_id: str) -> Optional[VerificationResult]:
        """Get a verification result by recommendation ID.
        
        Args:
            recommendation_id: The recommendation ID.
            
        Returns:
            The verification result, or None if not found.
        """
        return self.verification_results.get(recommendation_id)
    
    def get_grounded_recommendations(self) -> List[Recommendation]:
        """Get all grounded recommendations.
        
        Returns:
            A list of grounded recommendations.
        """
        grounded_ids = [
            result.recommendation_id
            for result in self.verification_results.values()
            if result.is_grounded
        ]
        
        return [
            self.recommendation_grounder.get_recommendation(recommendation_id)
            for recommendation_id in grounded_ids
            if self.recommendation_grounder.get_recommendation(recommendation_id)
        ]
    
    def get_ungrounded_recommendations(self) -> List[Recommendation]:
        """Get all ungrounded recommendations.
        
        Returns:
            A list of ungrounded recommendations.
        """
        ungrounded_ids = [
            result.recommendation_id
            for result in self.verification_results.values()
            if not result.is_grounded
        ]
        
        return [
            self.recommendation_grounder.get_recommendation(recommendation_id)
            for recommendation_id in ungrounded_ids
            if self.recommendation_grounder.get_recommendation(recommendation_id)
        ]
