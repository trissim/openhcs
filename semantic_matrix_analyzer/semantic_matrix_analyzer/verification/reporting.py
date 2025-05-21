"""
Verification reporting module for AST verification.

This module provides functionality for reporting verification results.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.verification.suggestion import CodeSuggestion, VerificationResult

logger = logging.getLogger(__name__)


class VerificationReporter:
    """Reports verification results."""
    
    def generate_report(self, suggestion: CodeSuggestion, verification_result: VerificationResult) -> Dict[str, Any]:
        """Generate a report of verification results.
        
        Args:
            suggestion: The code suggestion.
            verification_result: The verification result.
            
        Returns:
            A dictionary with the report.
        """
        return {
            "suggestion": {
                "file_path": str(suggestion.file_path),
                "start_line": suggestion.start_line,
                "end_line": suggestion.end_line,
                "original_code": suggestion.original_code,
                "suggested_code": suggestion.suggested_code,
                "description": suggestion.description,
                "confidence": suggestion.confidence
            },
            "verification": {
                "is_valid": verification_result.is_valid,
                "syntax_valid": verification_result.syntax_valid,
                "semantic_valid": verification_result.semantic_valid,
                "side_effects": verification_result.side_effects,
                "confidence": verification_result.confidence,
                "error_message": verification_result.error_message
            }
        }
    
    def format_report(self, report: Dict[str, Any], format: str = "text") -> str:
        """Format a report in the specified format.
        
        Args:
            report: The report to format.
            format: The format to use ("text", "markdown", or "json").
            
        Returns:
            The formatted report.
        """
        if format == "text":
            return self._format_text_report(report)
        elif format == "markdown":
            return self._format_markdown_report(report)
        elif format == "json":
            return json.dumps(report, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format a report as plain text.
        
        Args:
            report: The report to format.
            
        Returns:
            The formatted report.
        """
        suggestion = report["suggestion"]
        verification = report["verification"]
        
        lines = []
        
        # Suggestion details
        lines.append("Suggestion:")
        lines.append(f"  File: {suggestion['file_path']}")
        lines.append(f"  Lines: {suggestion['start_line']}-{suggestion['end_line']}")
        lines.append(f"  Description: {suggestion['description']}")
        lines.append(f"  Confidence: {suggestion['confidence']:.2f}")
        
        # Original code
        lines.append("\nOriginal code:")
        for i, line in enumerate(suggestion["original_code"].splitlines()):
            lines.append(f"  {suggestion['start_line'] + i}: {line}")
        
        # Suggested code
        lines.append("\nSuggested code:")
        for i, line in enumerate(suggestion["suggested_code"].splitlines()):
            lines.append(f"  {suggestion['start_line'] + i}: {line}")
        
        # Verification results
        lines.append("\nVerification results:")
        lines.append(f"  Valid: {verification['is_valid']}")
        lines.append(f"  Syntax valid: {verification['syntax_valid']}")
        lines.append(f"  Semantic valid: {verification['semantic_valid']}")
        lines.append(f"  Confidence: {verification['confidence']:.2f}")
        
        if verification["error_message"]:
            lines.append(f"  Error: {verification['error_message']}")
        
        # Side effects
        if verification["side_effects"]:
            lines.append("\nPotential side effects:")
            for effect in verification["side_effects"]:
                lines.append(f"  - {effect}")
        
        return "\n".join(lines)
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format a report as Markdown.
        
        Args:
            report: The report to format.
            
        Returns:
            The formatted report.
        """
        suggestion = report["suggestion"]
        verification = report["verification"]
        
        lines = []
        
        # Suggestion details
        lines.append("# Code Suggestion Verification Report")
        lines.append("")
        lines.append("## Suggestion Details")
        lines.append("")
        lines.append(f"- **File**: `{suggestion['file_path']}`")
        lines.append(f"- **Lines**: {suggestion['start_line']}-{suggestion['end_line']}")
        lines.append(f"- **Description**: {suggestion['description']}")
        lines.append(f"- **Confidence**: {suggestion['confidence']:.2f}")
        
        # Original code
        lines.append("")
        lines.append("## Original Code")
        lines.append("")
        lines.append("```python")
        lines.append(suggestion["original_code"])
        lines.append("```")
        
        # Suggested code
        lines.append("")
        lines.append("## Suggested Code")
        lines.append("")
        lines.append("```python")
        lines.append(suggestion["suggested_code"])
        lines.append("```")
        
        # Verification results
        lines.append("")
        lines.append("## Verification Results")
        lines.append("")
        lines.append(f"- **Valid**: {'✅' if verification['is_valid'] else '❌'}")
        lines.append(f"- **Syntax valid**: {'✅' if verification['syntax_valid'] else '❌'}")
        lines.append(f"- **Semantic valid**: {'✅' if verification['semantic_valid'] else '❌'}")
        lines.append(f"- **Confidence**: {verification['confidence']:.2f}")
        
        if verification["error_message"]:
            lines.append("")
            lines.append("### Error")
            lines.append("")
            lines.append(f"```")
            lines.append(verification["error_message"])
            lines.append(f"```")
        
        # Side effects
        if verification["side_effects"]:
            lines.append("")
            lines.append("## Potential Side Effects")
            lines.append("")
            for effect in verification["side_effects"]:
                lines.append(f"- {effect}")
        
        return "\n".join(lines)
    
    def summarize_verification_result(self, verification_result: VerificationResult) -> str:
        """Summarize a verification result.
        
        Args:
            verification_result: The verification result to summarize.
            
        Returns:
            A summary of the verification result.
        """
        if not verification_result.is_valid:
            if not verification_result.syntax_valid:
                return "❌ Invalid: Syntax error"
            elif not verification_result.semantic_valid:
                return "❌ Invalid: Semantic error"
            else:
                return f"❌ Invalid: {verification_result.error_message}"
        
        if verification_result.side_effects:
            return f"⚠️ Valid with {len(verification_result.side_effects)} potential side effects"
        
        return "✅ Valid"
