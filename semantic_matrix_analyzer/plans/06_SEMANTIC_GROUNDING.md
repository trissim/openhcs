# Plan 06: Semantic Grounding

## Objective

Develop a system for grounding all AI recommendations and findings in actual code patterns found in the AST, preventing hallucination and ensuring that all insights are directly tied to evidence from the codebase.

## Rationale

AI agents can sometimes generate plausible-sounding but incorrect analyses or recommendations. By implementing semantic grounding:

1. **Prevent Hallucination**: Ensure all findings are based on actual code patterns
2. **Increase Trustworthiness**: Users can verify the evidence behind each finding
3. **Improve Accuracy**: Reduce false positives and negatives in code analysis
4. **Enable Traceability**: Link findings directly to specific files, functions, and lines
5. **Support Verification**: Allow other tools to verify the findings independently

## Implementation Details

### 1. Evidence Collection

Create a system for collecting evidence from the codebase:

```python
@dataclass
class Evidence:
    """Evidence from the codebase supporting a finding."""
    file_path: Path
    node: UASTNode
    source_range: Tuple[int, int]  # (start, end) line numbers
    source_code: str
    pattern_name: str
    confidence: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the evidence to a dictionary."""
        return {
            "file_path": str(self.file_path),
            "source_range": self.source_range,
            "source_code": self.source_code,
            "pattern_name": self.pattern_name,
            "confidence": self.confidence
        }

class EvidenceCollector:
    """Collects evidence from the codebase."""
    
    def __init__(self, pattern_matcher: UASTPatternMatcher):
        self.pattern_matcher = pattern_matcher
    
    def collect_evidence(self, pattern: Dict[str, Any], file_path: Path, root_node: UASTNode) -> List[Evidence]:
        """Collect evidence for a pattern in a file."""
        evidence_list = []
        
        def visit(node):
            if self.pattern_matcher.match(pattern, node):
                # Calculate confidence based on pattern specificity
                confidence = self._calculate_confidence(pattern, node)
                
                # Extract source code
                source_code = self._extract_source_code(file_path, node.source_range)
                
                # Create evidence
                evidence = Evidence(
                    file_path=file_path,
                    node=node,
                    source_range=node.source_range,
                    source_code=source_code,
                    pattern_name=pattern.get("name", "unnamed_pattern"),
                    confidence=confidence
                )
                
                evidence_list.append(evidence)
            
            for child in node.children:
                visit(child)
        
        visit(root_node)
        return evidence_list
    
    def _calculate_confidence(self, pattern: Dict[str, Any], node: UASTNode) -> float:
        """Calculate confidence for a pattern match."""
        # This would implement a heuristic for confidence calculation
        # For example, more specific patterns would have higher confidence
        return 0.8  # Default confidence
    
    def _extract_source_code(self, file_path: Path, source_range: Tuple[int, int]) -> str:
        """Extract source code for a node."""
        if not source_range:
            return ""
        
        start_line, end_line = source_range
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
                # Adjust for 0-based indexing
                start_idx = max(0, start_line - 1)
                end_idx = min(len(lines), end_line)
                
                return "".join(lines[start_idx:end_idx])
        except Exception:
            return ""
```

### 2. Finding Generation

Create a system for generating findings based on evidence:

```python
@dataclass
class Finding:
    """A finding about the codebase."""
    id: str
    title: str
    description: str
    severity: str  # "info", "warning", "error"
    category: str
    evidence: List[Evidence]
    recommendation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the finding to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "category": self.category,
            "evidence": [e.to_dict() for e in self.evidence],
            "recommendation": self.recommendation
        }
    
    @property
    def confidence(self) -> float:
        """Calculate the overall confidence of the finding."""
        if not self.evidence:
            return 0.0
        
        return sum(e.confidence for e in self.evidence) / len(self.evidence)

class FindingGenerator:
    """Generates findings based on evidence."""
    
    def __init__(self, evidence_collector: EvidenceCollector):
        self.evidence_collector = evidence_collector
        self.finding_templates = {}
    
    def register_finding_template(self, pattern_name: str, template: Dict[str, Any]) -> None:
        """Register a finding template for a pattern."""
        self.finding_templates[pattern_name] = template
    
    def generate_findings(self, patterns: Dict[str, Dict[str, Any]], files: Dict[Path, UASTNode]) -> List[Finding]:
        """Generate findings for patterns across files."""
        findings = []
        
        for pattern_name, pattern in patterns.items():
            # Collect evidence for the pattern
            pattern_evidence = []
            for file_path, root_node in files.items():
                evidence = self.evidence_collector.collect_evidence(pattern, file_path, root_node)
                pattern_evidence.extend(evidence)
            
            if pattern_evidence:
                # Generate a finding based on the evidence
                finding = self._generate_finding(pattern_name, pattern_evidence)
                findings.append(finding)
        
        return findings
    
    def _generate_finding(self, pattern_name: str, evidence: List[Evidence]) -> Finding:
        """Generate a finding based on evidence."""
        template = self.finding_templates.get(pattern_name, {})
        
        # Generate a unique ID
        finding_id = f"{pattern_name}_{uuid.uuid4().hex[:8]}"
        
        # Use template values or defaults
        title = template.get("title", f"Found pattern: {pattern_name}")
        description = template.get("description", f"The pattern '{pattern_name}' was found in the codebase.")
        severity = template.get("severity", "info")
        category = template.get("category", "general")
        
        # Generate a recommendation if available
        recommendation = template.get("recommendation")
        if recommendation and callable(recommendation):
            recommendation = recommendation(evidence)
        
        return Finding(
            id=finding_id,
            title=title,
            description=description,
            severity=severity,
            category=category,
            evidence=evidence,
            recommendation=recommendation
        )
```

### 3. Evidence Verification

Create a system for verifying evidence:

```python
class EvidenceVerifier:
    """Verifies evidence for findings."""
    
    def verify_evidence(self, evidence: Evidence) -> bool:
        """Verify that evidence is valid."""
        # Check if the file exists
        if not evidence.file_path.exists():
            return False
        
        # Check if the source code matches
        actual_source = self._extract_source_code(evidence.file_path, evidence.source_range)
        if actual_source != evidence.source_code:
            return False
        
        return True
    
    def verify_finding(self, finding: Finding) -> bool:
        """Verify that a finding has valid evidence."""
        if not finding.evidence:
            return False
        
        # Verify each piece of evidence
        valid_evidence = [e for e in finding.evidence if self.verify_evidence(e)]
        
        # Update the finding with only valid evidence
        finding.evidence = valid_evidence
        
        # A finding is valid if it has at least one piece of valid evidence
        return len(valid_evidence) > 0
    
    def _extract_source_code(self, file_path: Path, source_range: Tuple[int, int]) -> str:
        """Extract source code for verification."""
        # Similar to the implementation in EvidenceCollector
        pass
```

### 4. Semantic Linking

Create a system for linking findings to semantic concepts:

```python
class SemanticLinker:
    """Links findings to semantic concepts."""
    
    def __init__(self):
        self.semantic_concepts = {}
    
    def register_semantic_concept(self, name: str, description: str, patterns: List[str]) -> None:
        """Register a semantic concept."""
        self.semantic_concepts[name] = {
            "description": description,
            "patterns": patterns
        }
    
    def link_finding_to_concepts(self, finding: Finding) -> List[str]:
        """Link a finding to semantic concepts."""
        linked_concepts = []
        
        for concept_name, concept in self.semantic_concepts.items():
            # Check if any of the finding's evidence matches the concept's patterns
            for evidence in finding.evidence:
                if evidence.pattern_name in concept["patterns"]:
                    linked_concepts.append(concept_name)
                    break
        
        return linked_concepts
    
    def get_concept_description(self, concept_name: str) -> Optional[str]:
        """Get the description of a semantic concept."""
        concept = self.semantic_concepts.get(concept_name)
        if concept:
            return concept["description"]
        return None
```

### 5. Explanation Generation

Create a system for generating explanations based on evidence:

```python
class ExplanationGenerator:
    """Generates explanations for findings based on evidence."""
    
    def __init__(self, semantic_linker: SemanticLinker):
        self.semantic_linker = semantic_linker
        self.explanation_templates = {}
    
    def register_explanation_template(self, concept_name: str, template: str) -> None:
        """Register an explanation template for a semantic concept."""
        self.explanation_templates[concept_name] = template
    
    def generate_explanation(self, finding: Finding) -> str:
        """Generate an explanation for a finding."""
        # Link the finding to semantic concepts
        concepts = self.semantic_linker.link_finding_to_concepts(finding)
        
        if not concepts:
            # Default explanation if no concepts are linked
            return self._generate_default_explanation(finding)
        
        # Generate explanations for each concept
        concept_explanations = []
        for concept in concepts:
            template = self.explanation_templates.get(concept)
            if template:
                explanation = self._fill_template(template, finding, concept)
                concept_explanations.append(explanation)
            else:
                # Use the concept description as a fallback
                description = self.semantic_linker.get_concept_description(concept)
                if description:
                    concept_explanations.append(f"{concept}: {description}")
        
        return "\n\n".join(concept_explanations)
    
    def _generate_default_explanation(self, finding: Finding) -> str:
        """Generate a default explanation for a finding."""
        return f"Found {len(finding.evidence)} instances of {finding.title}."
    
    def _fill_template(self, template: str, finding: Finding, concept: str) -> str:
        """Fill an explanation template with finding details."""
        # Replace placeholders in the template
        result = template
        result = result.replace("{title}", finding.title)
        result = result.replace("{description}", finding.description)
        result = result.replace("{evidence_count}", str(len(finding.evidence)))
        result = result.replace("{concept}", concept)
        
        # Replace {evidence_list} with a formatted list of evidence
        if "{evidence_list}" in result:
            evidence_list = "\n".join([
                f"- {e.file_path}:{e.source_range[0]}-{e.source_range[1]}" 
                for e in finding.evidence
            ])
            result = result.replace("{evidence_list}", evidence_list)
        
        return result
```

## Success Criteria

1. Evidence collection system with >95% accuracy
2. Finding generation based on evidence
3. Evidence verification system
4. Semantic linking to concepts
5. Explanation generation with templates
6. Documentation and examples for creating custom findings

## Dependencies

- Plan 04: Pattern Detection (for pattern matching)
- Plan 05: Language Abstraction (for UAST)

## Timeline

- Research and design: 1 week
- Evidence collection implementation: 2 weeks
- Finding generation implementation: 2 weeks
- Evidence verification implementation: 1 week
- Semantic linking implementation: 2 weeks
- Explanation generation implementation: 2 weeks
- Testing and documentation: 2 weeks

Total: 12 weeks
