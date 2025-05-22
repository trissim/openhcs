# plan_02_name_analysis.md
## Component: Name Analysis

### Objective
Create components for analyzing class, method, and variable names to extract intent. These components will tokenize compound names, extract semantic meaning, and classify intent based on naming patterns.

### Plan
1. Create a `NameTokenizer` class for breaking down compound names
2. Create a `SemanticExtractor` class for extracting semantic meaning from tokens
3. Create a `NameAnalyzer` class for analyzing class, method, and variable names
4. Implement common naming pattern recognition (e.g., verb-noun pairs, prefixes/suffixes)
5. Create a database of common programming terms and their semantic meanings

### Findings
Name analysis is a critical part of intent extraction, as names often directly convey the purpose of code elements. Different naming conventions (camelCase, snake_case, PascalCase) need to be handled, and compound names need to be broken down into meaningful tokens.

Key patterns to recognize:
- Action verbs in method names (get, set, validate, process, etc.)
- Domain objects in class names (User, Order, Transaction, etc.)
- State indicators in variable names (is_valid, has_permission, etc.)
- Design patterns in class names (Factory, Builder, Strategy, etc.)

### Implementation Draft

```python
"""
Name analysis components for extracting intent from names.
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, IntentType, CodeLocation
)

logger = logging.getLogger(__name__)


class NameTokenizer:
    """Tokenizes compound names into meaningful parts."""
    
    def __init__(self):
        """Initialize the name tokenizer."""
        # Common separators in different naming conventions
        self.separators = [
            "_",  # snake_case
            "-",  # kebab-case
            " ",  # space separated
        ]
        
        # Regex for splitting camelCase and PascalCase
        self.camel_case_pattern = re.compile(r'(?<=[a-z])(?=[A-Z])')
    
    def tokenize(self, name: str) -> List[str]:
        """Tokenize a name into parts.
        
        Args:
            name: The name to tokenize.
            
        Returns:
            A list of tokens.
        """
        # Handle empty names
        if not name:
            return []
        
        # First, try to split by separators
        for separator in self.separators:
            if separator in name:
                return [part for part in name.split(separator) if part]
        
        # If no separators, try to split camelCase or PascalCase
        if re.search(r'[a-z][A-Z]', name):
            return [part for part in self.camel_case_pattern.split(name) if part]
        
        # If all else fails, return the name as a single token
        return [name]
    
    def normalize_token(self, token: str) -> str:
        """Normalize a token.
        
        Args:
            token: The token to normalize.
            
        Returns:
            The normalized token.
        """
        # Convert to lowercase
        token = token.lower()
        
        # Remove common prefixes and suffixes
        prefixes = ["get", "set", "is", "has", "can", "should", "will", "do"]
        for prefix in prefixes:
            if token.startswith(prefix) and len(token) > len(prefix):
                token = token[len(prefix):]
        
        suffixes = ["er", "or", "able", "ible", "ize", "ise", "ify", "fy", "ing"]
        for suffix in suffixes:
            if token.endswith(suffix) and len(token) > len(suffix):
                token = token[:-len(suffix)]
        
        return token


class SemanticExtractor:
    """Extracts semantic meaning from tokens."""
    
    def __init__(self):
        """Initialize the semantic extractor."""
        # Common action verbs and their meanings
        self.action_verbs = {
            "get": "Retrieve or access",
            "set": "Modify or update",
            "create": "Create or instantiate",
            "delete": "Remove or destroy",
            "update": "Modify or change",
            "validate": "Check or verify",
            "process": "Handle or transform",
            "calculate": "Compute or determine",
            "find": "Search or locate",
            "check": "Verify or test",
            "is": "Test condition",
            "has": "Test possession",
            "can": "Test capability",
            "should": "Test recommendation",
            "will": "Indicate future action",
            "do": "Perform action",
            # Add more as needed
        }
        
        # Common design patterns and their meanings
        self.design_patterns = {
            "factory": "Create objects",
            "builder": "Construct complex objects",
            "singleton": "Ensure single instance",
            "adapter": "Convert interface",
            "decorator": "Add responsibilities",
            "observer": "Notify of changes",
            "strategy": "Define algorithm family",
            "command": "Encapsulate request",
            "iterator": "Access elements",
            "composite": "Treat objects uniformly",
            "proxy": "Control access",
            "facade": "Simplify interface",
            # Add more as needed
        }
        
        # Common domain objects and their meanings
        self.domain_objects = {
            "user": "User or account",
            "customer": "Client or buyer",
            "order": "Purchase or request",
            "product": "Item or good",
            "service": "Functionality or offering",
            "transaction": "Exchange or operation",
            "payment": "Financial transaction",
            "account": "User profile or financial account",
            "message": "Communication or notification",
            "event": "Occurrence or happening",
            "request": "Ask or demand",
            "response": "Answer or reply",
            "data": "Information or content",
            "config": "Configuration or settings",
            "manager": "Controller or supervisor",
            "handler": "Processor or responder",
            "provider": "Supplier or source",
            "consumer": "User or recipient",
            # Add more as needed
        }
    
    def extract_intent_from_tokens(self, tokens: List[str]) -> Tuple[str, str, IntentType]:
        """Extract intent from tokens.
        
        Args:
            tokens: The tokens to extract intent from.
            
        Returns:
            A tuple of (name, description, intent_type).
        """
        if not tokens:
            return "", "", IntentType.OTHER
        
        # Check for action verbs
        if tokens[0].lower() in self.action_verbs:
            verb = tokens[0].lower()
            verb_meaning = self.action_verbs[verb]
            
            # Check for verb-noun pairs
            if len(tokens) > 1:
                noun = tokens[1].lower()
                noun_meaning = self.domain_objects.get(noun, noun.capitalize())
                
                name = f"{verb.capitalize()} {noun_meaning}"
                description = f"{verb_meaning} {noun_meaning}"
                return name, description, IntentType.ACTION
            
            name = verb.capitalize()
            description = verb_meaning
            return name, description, IntentType.ACTION
        
        # Check for design patterns
        for token in tokens:
            if token.lower() in self.design_patterns:
                pattern = token.lower()
                pattern_meaning = self.design_patterns[pattern]
                
                name = f"{pattern.capitalize()} Pattern"
                description = pattern_meaning
                return name, description, IntentType.PATTERN
        
        # Check for domain objects
        for token in tokens:
            if token.lower() in self.domain_objects:
                object_name = token.lower()
                object_meaning = self.domain_objects[object_name]
                
                name = object_meaning
                description = f"Represents {object_meaning}"
                return name, description, IntentType.ENTITY
        
        # Check for state indicators
        state_prefixes = ["is", "has", "can", "should", "will"]
        if tokens[0].lower() in state_prefixes:
            prefix = tokens[0].lower()
            
            if len(tokens) > 1:
                state = " ".join(tokens[1:]).lower()
                
                name = f"{prefix.capitalize()} {state}"
                description = f"Indicates whether {state}"
                return name, description, IntentType.STATE
        
        # Default: use the tokens as is
        name = " ".join(token.capitalize() for token in tokens)
        description = f"Represents {name}"
        return name, description, IntentType.OTHER


class NameAnalyzer:
    """Analyzes class, method, and variable names to extract intent."""
    
    def __init__(self):
        """Initialize the name analyzer."""
        self.tokenizer = NameTokenizer()
        self.semantic_extractor = SemanticExtractor()
    
    def analyze_name(
        self,
        name: str,
        name_type: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a name to extract intent.
        
        Args:
            name: The name to analyze.
            name_type: The type of name ("class", "method", "variable", etc.).
            file_path: The path to the file containing the name (optional).
            line_number: The line number of the name (optional).
            
        Returns:
            A NameIntent object.
        """
        # Tokenize the name
        tokens = self.tokenizer.tokenize(name)
        
        # Extract intent from tokens
        intent_name, description, intent_type = self.semantic_extractor.extract_intent_from_tokens(tokens)
        
        # Create a location if file_path and line_number are provided
        location = None
        if file_path and line_number:
            location = CodeLocation(
                file_path=file_path,
                start_line=line_number,
                end_line=line_number
            )
        
        # Calculate confidence based on token quality
        confidence = self._calculate_confidence(tokens, name_type)
        
        # Create a NameIntent
        intent = NameIntent(
            name=intent_name,
            description=description,
            type=intent_type,
            confidence=confidence,
            location=location,
            original_name=name,
            tokens=tokens,
            name_type=name_type
        )
        
        return intent
    
    def _calculate_confidence(self, tokens: List[str], name_type: str) -> float:
        """Calculate confidence based on token quality.
        
        Args:
            tokens: The tokens to calculate confidence for.
            name_type: The type of name.
            
        Returns:
            The confidence score (0.0 to 1.0).
        """
        if not tokens:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Adjust based on token count
        if len(tokens) > 1:
            confidence += 0.1  # Compound names are more informative
        
        # Adjust based on token quality
        meaningful_tokens = 0
        for token in tokens:
            token_lower = token.lower()
            if (token_lower in self.semantic_extractor.action_verbs or
                token_lower in self.semantic_extractor.design_patterns or
                token_lower in self.semantic_extractor.domain_objects):
                meaningful_tokens += 1
        
        if tokens:
            confidence += 0.2 * (meaningful_tokens / len(tokens))
        
        # Adjust based on name type
        if name_type == "class":
            confidence += 0.1  # Class names are usually more descriptive
        elif name_type == "method":
            confidence += 0.1  # Method names often contain verbs
        
        # Cap confidence at 1.0
        return min(1.0, confidence)
    
    def analyze_class_name(
        self,
        name: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a class name to extract intent.
        
        Args:
            name: The class name to analyze.
            file_path: The path to the file containing the class (optional).
            line_number: The line number of the class (optional).
            
        Returns:
            A NameIntent object.
        """
        return self.analyze_name(name, "class", file_path, line_number)
    
    def analyze_method_name(
        self,
        name: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a method name to extract intent.
        
        Args:
            name: The method name to analyze.
            file_path: The path to the file containing the method (optional).
            line_number: The line number of the method (optional).
            
        Returns:
            A NameIntent object.
        """
        return self.analyze_name(name, "method", file_path, line_number)
    
    def analyze_variable_name(
        self,
        name: str,
        file_path: Optional[Path] = None,
        line_number: Optional[int] = None
    ) -> NameIntent:
        """Analyze a variable name to extract intent.
        
        Args:
            name: The variable name to analyze.
            file_path: The path to the file containing the variable (optional).
            line_number: The line number of the variable (optional).
            
        Returns:
            A NameIntent object.
        """
        return self.analyze_name(name, "variable", file_path, line_number)
```
