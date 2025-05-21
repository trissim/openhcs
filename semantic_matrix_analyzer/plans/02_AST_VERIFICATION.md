# Plan 02: Enhanced AST Verification

## Objective

Develop a robust system for verifying all suggestions and code changes against the AST before implementation, ensuring syntactic and semantic correctness, detecting potential side effects, and providing confidence scores for each suggestion.

## Rationale

AI agents can sometimes generate plausible-sounding but incorrect code changes. By implementing enhanced AST verification:

1. **Improved Correctness**: All suggestions are verified against the AST before being proposed
2. **Reduced Cognitive Load**: Users don't need to mentally verify the correctness of suggestions
3. **Side Effect Detection**: Potential unintended consequences of changes are identified
4. **Confidence Scoring**: Users can prioritize high-confidence suggestions
5. **Feedback Loop**: The system learns from verification results to improve future suggestions

## Implementation Details

### 1. AST-Based Suggestion Verification

Create a system for verifying suggestions against the AST:

```python
@dataclass
class CodeSuggestion:
    """A suggestion for changing code."""
    file_path: Path
    start_line: int
    end_line: int
    original_code: str
    suggested_code: str
    description: str
    confidence: float = 0.0
    verification_result: Optional['VerificationResult'] = None
    
@dataclass
class VerificationResult:
    """The result of verifying a suggestion."""
    is_valid: bool
    syntax_valid: bool
    semantic_valid: bool
    side_effects: List[str]
    confidence: float
    error_message: Optional[str] = None
    
class SuggestionVerifier:
    """Verifies code suggestions against the AST."""
    
    def verify_suggestion(self, suggestion: CodeSuggestion, file_path: Path) -> VerificationResult:
        """Verify a code suggestion against the AST."""
        # Check syntax validity
        syntax_valid = self._check_syntax(suggestion.suggested_code)
        
        if not syntax_valid:
            return VerificationResult(
                is_valid=False,
                syntax_valid=False,
                semantic_valid=False,
                side_effects=[],
                confidence=0.0,
                error_message="Syntax error in suggested code"
            )
        
        # Check semantic validity
        semantic_valid, semantic_errors = self._check_semantics(suggestion, file_path)
        
        if not semantic_valid:
            return VerificationResult(
                is_valid=False,
                syntax_valid=True,
                semantic_valid=False,
                side_effects=[],
                confidence=0.0,
                error_message=f"Semantic error in suggested code: {', '.join(semantic_errors)}"
            )
        
        # Check for side effects
        side_effects = self._check_side_effects(suggestion, file_path)
        
        # Calculate confidence
        confidence = self._calculate_confidence(suggestion, syntax_valid, semantic_valid, side_effects)
        
        return VerificationResult(
            is_valid=True,
            syntax_valid=True,
            semantic_valid=True,
            side_effects=side_effects,
            confidence=confidence
        )
    
    def _check_syntax(self, code: str) -> bool:
        """Check if the code is syntactically valid."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _check_semantics(self, suggestion: CodeSuggestion, file_path: Path) -> Tuple[bool, List[str]]:
        """Check if the code is semantically valid."""
        # This would implement semantic checks such as:
        # - Variable references
        # - Type compatibility
        # - Import resolution
        # - etc.
        return True, []
    
    def _check_side_effects(self, suggestion: CodeSuggestion, file_path: Path) -> List[str]:
        """Check for potential side effects of the suggestion."""
        # This would implement side effect detection such as:
        # - Changes to global state
        # - Changes to function signatures
        # - Changes to class interfaces
        # - etc.
        return []
    
    def _calculate_confidence(self, suggestion: CodeSuggestion, syntax_valid: bool, semantic_valid: bool, side_effects: List[str]) -> float:
        """Calculate the confidence score for the suggestion."""
        # Start with the suggestion's initial confidence
        confidence = suggestion.confidence
        
        # Adjust based on verification results
        if not syntax_valid:
            confidence *= 0.1
        if not semantic_valid:
            confidence *= 0.2
        
        # Reduce confidence based on side effects
        confidence *= max(0.1, 1.0 - (len(side_effects) * 0.1))
        
        return confidence
```

### 2. Code Change Simulation

Create a system for simulating code changes before applying them:

```python
class CodeChangeSimulator:
    """Simulates code changes to detect potential issues."""
    
    def simulate_change(self, suggestion: CodeSuggestion, file_path: Path) -> Dict[str, Any]:
        """Simulate a code change and return the results."""
        # Create a temporary copy of the file
        temp_file = self._create_temp_file(file_path)
        
        # Apply the suggestion to the temporary file
        self._apply_suggestion(suggestion, temp_file)
        
        # Parse the modified file
        try:
            with open(temp_file, "r", encoding="utf-8") as f:
                modified_code = f.read()
            
            modified_ast = ast.parse(modified_code)
            
            # Analyze the modified AST
            analysis_results = self._analyze_ast(modified_ast)
            
            # Clean up
            os.remove(temp_file)
            
            return {
                "success": True,
                "analysis": analysis_results
            }
        except Exception as e:
            # Clean up
            os.remove(temp_file)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_temp_file(self, file_path: Path) -> Path:
        """Create a temporary copy of the file."""
        temp_file = file_path.with_suffix(".temp" + file_path.suffix)
        shutil.copy(file_path, temp_file)
        return temp_file
    
    def _apply_suggestion(self, suggestion: CodeSuggestion, temp_file: Path) -> None:
        """Apply the suggestion to the temporary file."""
        with open(temp_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Replace the lines
        new_lines = suggestion.suggested_code.splitlines(True)
        lines[suggestion.start_line - 1:suggestion.end_line] = new_lines
        
        with open(temp_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze the AST for potential issues."""
        # This would implement AST analysis such as:
        # - Detecting unused variables
        # - Detecting unreachable code
        # - Detecting potential bugs
        # - etc.
        return {}
```

### 3. Side Effect Detector

Create a system for detecting potential side effects of code changes:

```python
class SideEffectDetector:
    """Detects potential side effects of code changes."""
    
    def detect_side_effects(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect potential side effects of a code change."""
        side_effects = []
        
        # Check for changes to function signatures
        function_changes = self._detect_function_signature_changes(original_ast, modified_ast)
        side_effects.extend(function_changes)
        
        # Check for changes to class interfaces
        class_changes = self._detect_class_interface_changes(original_ast, modified_ast)
        side_effects.extend(class_changes)
        
        # Check for changes to global variables
        global_changes = self._detect_global_variable_changes(original_ast, modified_ast)
        side_effects.extend(global_changes)
        
        # Check for changes to imports
        import_changes = self._detect_import_changes(original_ast, modified_ast)
        side_effects.extend(import_changes)
        
        return side_effects
    
    def _detect_function_signature_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to function signatures."""
        # This would implement detection of changes to function signatures
        return []
    
    def _detect_class_interface_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to class interfaces."""
        # This would implement detection of changes to class interfaces
        return []
    
    def _detect_global_variable_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to global variables."""
        # This would implement detection of changes to global variables
        return []
    
    def _detect_import_changes(self, original_ast: ast.AST, modified_ast: ast.AST) -> List[str]:
        """Detect changes to imports."""
        # This would implement detection of changes to imports
        return []
```

### 4. Verification Reporting

Create a system for reporting verification results:

```python
class VerificationReporter:
    """Reports verification results."""
    
    def generate_report(self, suggestion: CodeSuggestion, verification_result: VerificationResult) -> Dict[str, Any]:
        """Generate a report of verification results."""
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
        """Format a report in the specified format."""
        if format == "text":
            return self._format_text_report(report)
        elif format == "markdown":
            return self._format_markdown_report(report)
        elif format == "json":
            return json.dumps(report, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format a report as plain text."""
        # This would implement text formatting of the report
        return ""
    
    def _format_markdown_report(self, report: Dict[str, Any]) -> str:
        """Format a report as Markdown."""
        # This would implement Markdown formatting of the report
        return ""
```

## Success Criteria

1. Verification of all code suggestions against the AST
2. Detection of potential side effects of code changes
3. Confidence scoring for suggestions
4. Simulation of code changes before application
5. Comprehensive reporting of verification results

## Dependencies

- Existing AST parsing system
- Existing code suggestion system

## Timeline

- Research and design: 1 week
- AST-based suggestion verification: 2 weeks
- Code change simulation: 1 week
- Side effect detector: 2 weeks
- Verification reporting: 1 week
- Testing and documentation: 1 week

Total: 8 weeks
