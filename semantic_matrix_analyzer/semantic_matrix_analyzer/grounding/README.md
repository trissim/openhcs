# Semantic Grounding

The Semantic Grounding system is a component of the Semantic Matrix Analyzer that provides functionality for grounding AI recommendations and findings in actual code patterns found in the AST, preventing hallucination and ensuring that all insights are directly tied to evidence from the codebase.

## Overview

The Semantic Grounding system consists of several key components:

1. **Evidence Collection**: Collects evidence from the codebase
2. **Pattern Matching**: Defines and matches code patterns
3. **Recommendation Grounding**: Grounds recommendations in evidence
4. **Grounding Verification**: Verifies that recommendations are grounded in evidence

## Usage

### Basic Usage

```python
from semantic_matrix_analyzer.grounding import (
    Evidence, EvidenceCollector, Recommendation, RecommendationGrounder,
    CodePattern, PatternMatcher, GroundingVerifier
)

# Create an evidence collector
evidence_collector = EvidenceCollector()

# Collect evidence from a file
evidence = evidence_collector.collect_evidence_for_file(Path("example.py"))

# Create a pattern matcher
pattern_matcher = PatternMatcher()

# Create and add patterns
pattern = CodePattern(
    id="function_definition",
    name="Function Definition",
    description="A function definition",
    ast_pattern={"type": "FunctionDef"}
)
pattern_matcher.add_pattern(pattern)

# Match patterns in code
with open("example.py", "r") as f:
    code = f.read()
pattern_matches = pattern_matcher.match_patterns(code, Path("example.py"))

# Create a recommendation grounder
recommendation_grounder = RecommendationGrounder(evidence_collector)

# Ground a recommendation
recommendation = recommendation_grounder.ground_recommendation(
    type="optimization",
    title="Optimize function",
    description="This function can be optimized.",
    file_paths=[Path("example.py")],
    line_ranges=[(1, 10)],
    suggested_code="def optimized_function(): pass"
)

# Create a grounding verifier
grounding_verifier = GroundingVerifier(evidence_collector, recommendation_grounder)

# Verify a recommendation
verification_result = grounding_verifier.verify_recommendation(recommendation)

# Get grounded recommendations
grounded_recommendations = grounding_verifier.get_grounded_recommendations()
```

## Components

### Evidence Collection

Collects evidence from the codebase:

- `Evidence`: Evidence from the codebase to support a recommendation or finding
- `EvidenceCollector`: Collects evidence from the codebase
- `collect_evidence_for_file`: Collects evidence from a file
- `_collect_ast_node_evidence`: Collects evidence from AST nodes
- `_collect_code_pattern_evidence`: Collects evidence from code patterns

### Pattern Matching

Defines and matches code patterns:

- `CodePattern`: A pattern of code to match in the codebase
- `PatternMatcher`: Matches code patterns in the codebase
- `match_patterns`: Matches patterns in code
- `_match_ast_pattern`: Matches an AST pattern in code
- `_match_regex_pattern`: Matches a regex pattern in code
- `_match_custom_pattern`: Matches a custom pattern in code

### Recommendation Grounding

Grounds recommendations in evidence:

- `Recommendation`: An AI recommendation grounded in evidence from the codebase
- `RecommendationGrounder`: Grounds AI recommendations in evidence from the codebase
- `ground_recommendation`: Grounds a recommendation in evidence from the codebase
- `get_evidence_for_recommendation`: Gets evidence for a recommendation

### Grounding Verification

Verifies that recommendations are grounded in evidence:

- `VerificationResult`: The result of verifying a recommendation
- `GroundingVerifier`: Verifies that recommendations are grounded in evidence
- `verify_recommendation`: Verifies that a recommendation is grounded in evidence
- `get_grounded_recommendations`: Gets all grounded recommendations
- `get_ungrounded_recommendations`: Gets all ungrounded recommendations

## Benefits

The Semantic Grounding system provides several benefits:

1. **Prevents Hallucination**: Ensures that AI recommendations are based on actual code patterns
2. **Increases Confidence**: Provides evidence to support recommendations
3. **Improves Accuracy**: Verifies that recommendations are consistent with the codebase
4. **Enhances Transparency**: Makes it clear why recommendations are being made
5. **Reduces False Positives**: Filters out recommendations that are not well-grounded
