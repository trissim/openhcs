# plan_02_pattern_intent_extraction.md
## Component: Pattern and Intent Extraction System

### Objective
Create a system for extracting patterns and intents from code, user input, and feedback. This system will enable the brain agent to recognize recurring patterns, understand user intentions, and adapt its behavior accordingly.

### Plan
1. Implement a PatternExtractor class for identifying patterns in code and text
2. Create an IntentExtractor class for determining user intentions
3. Develop pattern matching algorithms with configurable weights
4. Implement similarity scoring between patterns
5. Create pattern evolution tracking
6. Integrate with the DynamicConfigManager for adaptive thresholds

### Findings
The current codebase has some pattern matching capabilities in the SMA system, but lacks a comprehensive pattern and intent extraction system with configurable weights and adaptive thresholds. The brain agent requires these components to implement pattern matching with weights, intent extraction aligned with patterns, and configurable analyzers with adaptive thresholds.

### Implementation Draft

#### 1. PatternExtractor Class

```python
"""
Pattern Extraction System for brain agent.

This module provides functionality for identifying and extracting patterns
from code, text, and user feedback with configurable weights and thresholds.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import re
import hashlib
from dataclasses import dataclass
import numpy as np


@dataclass
class Pattern:
    """Represents a detected pattern."""
    
    id: str  # Unique identifier
    name: str  # Human-readable name
    pattern_type: str  # Type of pattern (e.g., "code", "text", "behavior")
    content: str  # The actual pattern content
    weight: float  # Importance weight (0.0 to 1.0)
    confidence: float  # Confidence in the pattern (0.0 to 1.0)
    metadata: Dict[str, Any]  # Additional metadata
    
    @staticmethod
    def create_id(content: str, pattern_type: str) -> str:
        """
        Create a unique ID for a pattern.
        
        Args:
            content: Pattern content
            pattern_type: Type of pattern
            
        Returns:
            Unique ID string
        """
        hash_input = f"{pattern_type}:{content}"
        return hashlib.md5(hash_input.encode()).hexdigest()


@dataclass
class PatternMatch:
    """Represents a match between a pattern and content."""
    
    pattern: Pattern  # The matched pattern
    content: str  # The content that matched
    score: float  # Match score (0.0 to 1.0)
    locations: List[Tuple[int, int]]  # Start and end positions of matches
    context: Dict[str, Any]  # Additional context


class PatternExtractor:
    """
    Extracts and matches patterns in code and text.
    
    This class provides:
    1. Pattern extraction from various sources
    2. Pattern matching with configurable thresholds
    3. Pattern similarity scoring
    4. Pattern evolution tracking
    """
    
    def __init__(self, config_manager):
        """
        Initialize the pattern extractor.
        
        Args:
            config_manager: Dynamic configuration manager
        """
        self.config_manager = config_manager
        self.patterns: Dict[str, Pattern] = {}
        
        # Initialize default pattern configuration if not present
        config = config_manager.get_config()
        if "patterns" not in config:
            pattern_config = {
                "patterns": {
                    "min_match_threshold": 0.7,
                    "max_patterns_per_type": 100,
                    "similarity_threshold": 0.8,
                    "pattern_types": {
                        "code": {
                            "enabled": True,
                            "weight": 1.0
                        },
                        "text": {
                            "enabled": True,
                            "weight": 0.8
                        },
                        "behavior": {
                            "enabled": True,
                            "weight": 0.9
                        }
                    }
                }
            }
            config_manager.update_config(pattern_config, "system")
        
    def extract_patterns(self, content: str, pattern_type: str, 
                         context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from content.
        
        Args:
            content: Content to extract patterns from
            pattern_type: Type of pattern to extract
            context: Optional context information
            
        Returns:
            List of extracted patterns
        """
        config = self.config_manager.get_config()
        patterns = []
        
        # Check if pattern type is enabled
        if not config["patterns"]["pattern_types"].get(pattern_type, {}).get("enabled", False):
            return patterns
            
        # Extract patterns based on type
        if pattern_type == "code":
            patterns = self._extract_code_patterns(content, context)
        elif pattern_type == "text":
            patterns = self._extract_text_patterns(content, context)
        elif pattern_type == "behavior":
            patterns = self._extract_behavior_patterns(content, context)
            
        # Store new patterns
        for pattern in patterns:
            if pattern.id not in self.patterns:
                self.patterns[pattern.id] = pattern
            else:
                # Update existing pattern
                existing = self.patterns[pattern.id]
                existing.weight = (existing.weight + pattern.weight) / 2
                existing.confidence = (existing.confidence + pattern.confidence) / 2
                
        return patterns
        
    def _extract_code_patterns(self, code: str, 
                              context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from code.
        
        Args:
            code: Code to extract patterns from
            context: Optional context information
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        # Example pattern extraction logic for code
        # 1. Function definitions
        func_pattern = r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1)
            pattern_content = f"function:{func_name}"
            pattern_id = Pattern.create_id(pattern_content, "code")
            
            patterns.append(Pattern(
                id=pattern_id,
                name=f"Function: {func_name}",
                pattern_type="code",
                content=pattern_content,
                weight=0.8,
                confidence=0.9,
                metadata={
                    "function_name": func_name,
                    "position": match.start()
                }
            ))
            
        # 2. Class definitions
        class_pattern = r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*"
        for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            pattern_content = f"class:{class_name}"
            pattern_id = Pattern.create_id(pattern_content, "code")
            
            patterns.append(Pattern(
                id=pattern_id,
                name=f"Class: {class_name}",
                pattern_type="code",
                content=pattern_content,
                weight=0.9,
                confidence=0.9,
                metadata={
                    "class_name": class_name,
                    "position": match.start()
                }
            ))
            
        # Additional pattern extraction logic would go here
        
        return patterns
        
    def _extract_text_patterns(self, text: str, 
                              context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from text.
        
        Args:
            text: Text to extract patterns from
            context: Optional context information
            
        Returns:
            List of extracted patterns
        """
        patterns = []
        
        # Example pattern extraction logic for text
        # 1. Keywords and phrases
        keywords = ["improve", "fix", "update", "add", "remove", "change", "optimize"]
        for keyword in keywords:
            pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                pattern_content = f"keyword:{keyword.lower()}"
                pattern_id = Pattern.create_id(pattern_content, "text")
                
                patterns.append(Pattern(
                    id=pattern_id,
                    name=f"Keyword: {keyword}",
                    pattern_type="text",
                    content=pattern_content,
                    weight=0.7,
                    confidence=0.8,
                    metadata={
                        "keyword": keyword.lower(),
                        "position": match.start()
                    }
                ))
                
        # Additional pattern extraction logic would go here
        
        return patterns
        
    def _extract_behavior_patterns(self, behavior: str, 
                                  context: Optional[Dict[str, Any]] = None) -> List[Pattern]:
        """
        Extract patterns from behavior descriptions.
        
        Args:
            behavior: Behavior description to extract patterns from
            context: Optional context information
            
        Returns:
            List of extracted patterns
        """
        # Implementation would depend on how behavior is represented
        # This is a placeholder
        return []
        
    def match_patterns(self, content: str, pattern_type: str, 
                       threshold: Optional[float] = None) -> List[PatternMatch]:
        """
        Match content against known patterns.
        
        Args:
            content: Content to match
            pattern_type: Type of patterns to match against
            threshold: Minimum match score (0.0 to 1.0)
            
        Returns:
            List of pattern matches
        """
        config = self.config_manager.get_config()
        if threshold is None:
            threshold = config["patterns"]["min_match_threshold"]
            
        matches = []
        
        # Get patterns of the specified type
        relevant_patterns = [p for p in self.patterns.values() if p.pattern_type == pattern_type]
        
        for pattern in relevant_patterns:
            # Match based on pattern type
            if pattern_type == "code":
                match_result = self._match_code_pattern(pattern, content)
            elif pattern_type == "text":
                match_result = self._match_text_pattern(pattern, content)
            elif pattern_type == "behavior":
                match_result = self._match_behavior_pattern(pattern, content)
            else:
                continue
                
            # Add match if score exceeds threshold
            if match_result and match_result.score >= threshold:
                matches.append(match_result)
                
        # Sort by score (highest first)
        matches.sort(key=lambda m: m.score, reverse=True)
        
        return matches
        
    def _match_code_pattern(self, pattern: Pattern, code: str) -> Optional[PatternMatch]:
        """
        Match a code pattern against code.
        
        Args:
            pattern: Pattern to match
            code: Code to match against
            
        Returns:
            PatternMatch if matched, None otherwise
        """
        # Extract pattern details
        if pattern.content.startswith("function:"):
            func_name = pattern.content.split(":", 1)[1]
            func_pattern = re.compile(r"def\s+" + re.escape(func_name) + r"\s*\(")
            
            # Find all matches
            locations = []
            for match in func_pattern.finditer(code):
                locations.append((match.start(), match.end()))
                
            if locations:
                return PatternMatch(
                    pattern=pattern,
                    content=code,
                    score=1.0,  # Exact match
                    locations=locations,
                    context={}
                )
                
        elif pattern.content.startswith("class:"):
            class_name = pattern.content.split(":", 1)[1]
            class_pattern = re.compile(r"class\s+" + re.escape(class_name) + r"\s*")
            
            # Find all matches
            locations = []
            for match in class_pattern.finditer(code):
                locations.append((match.start(), match.end()))
                
            if locations:
                return PatternMatch(
                    pattern=pattern,
                    content=code,
                    score=1.0,  # Exact match
                    locations=locations,
                    context={}
                )
                
        return None
        
    def _match_text_pattern(self, pattern: Pattern, text: str) -> Optional[PatternMatch]:
        """
        Match a text pattern against text.
        
        Args:
            pattern: Pattern to match
            text: Text to match against
            
        Returns:
            PatternMatch if matched, None otherwise
        """
        # Extract pattern details
        if pattern.content.startswith("keyword:"):
            keyword = pattern.content.split(":", 1)[1]
            keyword_pattern = re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
            
            # Find all matches
            locations = []
            for match in keyword_pattern.finditer(text):
                locations.append((match.start(), match.end()))
                
            if locations:
                return PatternMatch(
                    pattern=pattern,
                    content=text,
                    score=1.0,  # Exact match
                    locations=locations,
                    context={}
                )
                
        return None
        
    def _match_behavior_pattern(self, pattern: Pattern, behavior: str) -> Optional[PatternMatch]:
        """
        Match a behavior pattern against behavior.
        
        Args:
            pattern: Pattern to match
            behavior: Behavior to match against
            
        Returns:
            PatternMatch if matched, None otherwise
        """
        # Implementation would depend on how behavior is represented
        # This is a placeholder
        return None
```

#### 2. IntentExtractor Class

```python
"""
Intent Extraction System for brain agent.

This module provides functionality for determining user intentions
from code, text, and feedback using pattern matching and context.
"""

from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass


@dataclass
class Intent:
    """Represents a detected intent."""
    
    id: str  # Unique identifier
    name: str  # Human-readable name
    confidence: float  # Confidence in the intent (0.0 to 1.0)
    patterns: List[PatternMatch]  # Patterns that contributed to this intent
    context: Dict[str, Any]  # Additional context
    metadata: Dict[str, Any]  # Additional metadata


class IntentExtractor:
    """
    Extracts user intents from content using pattern matching.
    
    This class provides:
    1. Intent extraction from various sources
    2. Intent confidence scoring
    3. Intent-to-pattern mapping
    """
    
    def __init__(self, config_manager, pattern_extractor):
        """
        Initialize the intent extractor.
        
        Args:
            config_manager: Dynamic configuration manager
            pattern_extractor: Pattern extractor
        """
        self.config_manager = config_manager
        self.pattern_extractor = pattern_extractor
        
        # Initialize default intent configuration if not present
        config = config_manager.get_config()
        if "intents" not in config:
            intent_config = {
                "intents": {
                    "min_confidence_threshold": 0.6,
                    "max_intents_per_content": 5,
                    "intent_types": {
                        "code_improvement": {
                            "enabled": True,
                            "weight": 1.0,
                            "patterns": ["function", "class", "optimize"]
                        },
                        "bug_fix": {
                            "enabled": True,
                            "weight": 1.0,
                            "patterns": ["fix", "bug", "issue", "error"]
                        },
                        "feature_request": {
                            "enabled": True,
                            "weight": 0.9,
                            "patterns": ["add", "feature", "implement", "new"]
                        }
                    }
                }
            }
            config_manager.update_config(intent_config, "system")
        
    def extract_intent(self, content: str, content_type: str, 
                      context: Optional[Dict[str, Any]] = None) -> List[Intent]:
        """
        Extract intents from content.
        
        Args:
            content: Content to extract intents from
            content_type: Type of content ("code", "text", "behavior")
            context: Optional context information
            
        Returns:
            List of extracted intents
        """
        config = self.config_manager.get_config()
        intents = []
        
        # Extract patterns from content
        patterns = self.pattern_extractor.extract_patterns(content, content_type, context)
        
        # Match against known patterns
        matches = self.pattern_extractor.match_patterns(content, content_type)
        
        # Combine all pattern matches
        all_matches = matches + [
            PatternMatch(
                pattern=p,
                content=content,
                score=1.0,  # New patterns are exact matches
                locations=[],  # No specific locations for new patterns
                context=context or {}
            ) for p in patterns
        ]
        
        # Map patterns to intents
        intent_matches: Dict[str, List[PatternMatch]] = {}
        
        for match in all_matches:
            pattern_content = match.pattern.content
            
            # Check which intents this pattern contributes to
            for intent_type, intent_config in config["intents"]["intent_types"].items():
                if not intent_config.get("enabled", True):
                    continue
                    
                # Check if pattern matches any of the intent's patterns
                for intent_pattern in intent_config.get("patterns", []):
                    if intent_pattern.lower() in pattern_content.lower():
                        if intent_type not in intent_matches:
                            intent_matches[intent_type] = []
                        intent_matches[intent_type].append(match)
                        break
        
        # Create intent objects
        for intent_type, matches in intent_matches.items():
            # Calculate confidence based on pattern matches
            total_score = sum(match.score * match.pattern.weight for match in matches)
            avg_score = total_score / len(matches) if matches else 0
            
            # Apply intent type weight
            intent_weight = config["intents"]["intent_types"][intent_type].get("weight", 1.0)
            confidence = avg_score * intent_weight
            
            # Only include intents above threshold
            if confidence >= config["intents"]["min_confidence_threshold"]:
                intents.append(Intent(
                    id=f"{intent_type}_{hash(content) % 10000}",
                    name=intent_type.replace("_", " ").title(),
                    confidence=confidence,
                    patterns=matches,
                    context=context or {},
                    metadata={
                        "intent_type": intent_type,
                        "pattern_count": len(matches)
                    }
                ))
        
        # Sort by confidence (highest first)
        intents.sort(key=lambda i: i.confidence, reverse=True)
        
        # Limit number of intents
        max_intents = config["intents"]["max_intents_per_content"]
        return intents[:max_intents]
```

The FeedbackProcessor component will be implemented in the next plan file to complete the brain agent interface.
