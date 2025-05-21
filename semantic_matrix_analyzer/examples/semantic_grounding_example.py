#!/usr/bin/env python3
"""
Example of using the semantic grounding system.

This script demonstrates how to use the semantic grounding system to ground AI recommendations
and findings in actual code patterns found in the AST, preventing hallucination and ensuring
that all insights are directly tied to evidence from the codebase.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.grounding import (
    Evidence, EvidenceCollector, Recommendation, RecommendationGrounder,
    CodePattern, PatternMatcher, GroundingVerifier
)


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_sample_file():
    """Create a sample file for testing.
    
    Returns:
        The path to the sample file.
    """
    # Create a temporary file
    fd, temp_path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    
    # Write sample code to the file
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write("""
def calculate_average(values):
    \"\"\"Calculate the average of a list of values.
    
    Args:
        values: A list of numeric values.
        
    Returns:
        The average of the values.
    \"\"\"
    total = 0
    count = 0
    
    for value in values:
        total += value
        count += 1
    
    if count == 0:
        return 0
    
    return total / count

def process_data(data):
    \"\"\"Process a list of data items.
    
    Args:
        data: A list of data items.
        
    Returns:
        A list of processed items.
    \"\"\"
    result = []
    
    for item in data:
        if item is None:
            continue
        
        try:
            processed = item * 2
            result.append(processed)
        except:
            print("Error processing item")
    
    return result

class DataProcessor:
    \"\"\"A class for processing data.\"\"\"
    
    def __init__(self, multiplier=2):
        \"\"\"Initialize the data processor.
        
        Args:
            multiplier: The multiplier to use.
        \"\"\"
        self.multiplier = multiplier
    
    def process_item(self, item):
        \"\"\"Process an item.
        
        Args:
            item: The item to process.
            
        Returns:
            The processed item.
        \"\"\"
        return item * self.multiplier
    
    def process_items(self, items):
        \"\"\"Process a list of items.
        
        Args:
            items: The items to process.
            
        Returns:
            The processed items.
        \"\"\"
        return [self.process_item(item) for item in items if item is not None]
""")
    
    return temp_path


def collect_evidence(file_path):
    """Collect evidence from a file.
    
    Args:
        file_path: The path to the file.
        
    Returns:
        A tuple of (evidence_collector, evidence).
    """
    # Create an evidence collector
    evidence_collector = EvidenceCollector()
    
    # Collect evidence from the file
    evidence = evidence_collector.collect_evidence_for_file(Path(file_path))
    
    return evidence_collector, evidence


def create_patterns():
    """Create code patterns.
    
    Returns:
        A pattern matcher with patterns.
    """
    # Create a pattern matcher
    pattern_matcher = PatternMatcher()
    
    # Create patterns
    
    # 1. Function definition pattern
    function_pattern = CodePattern(
        id="function_definition",
        name="Function Definition",
        description="A function definition",
        ast_pattern={"type": "FunctionDef"}
    )
    pattern_matcher.add_pattern(function_pattern)
    
    # 2. Class definition pattern
    class_pattern = CodePattern(
        id="class_definition",
        name="Class Definition",
        description="A class definition",
        ast_pattern={"type": "ClassDef"}
    )
    pattern_matcher.add_pattern(class_pattern)
    
    # 3. For loop pattern
    for_loop_pattern = CodePattern(
        id="for_loop",
        name="For Loop",
        description="A for loop",
        ast_pattern={"type": "For"}
    )
    pattern_matcher.add_pattern(for_loop_pattern)
    
    # 4. Try-except pattern
    try_except_pattern = CodePattern(
        id="try_except",
        name="Try-Except Block",
        description="A try-except block",
        ast_pattern={"type": "Try"}
    )
    pattern_matcher.add_pattern(try_except_pattern)
    
    # 5. Bare except pattern
    bare_except_pattern = CodePattern(
        id="bare_except",
        name="Bare Except",
        description="A bare except clause",
        regex_pattern=r"except\s*:"
    )
    pattern_matcher.add_pattern(bare_except_pattern)
    
    return pattern_matcher


def create_recommendations(file_path, evidence_collector):
    """Create recommendations.
    
    Args:
        file_path: The path to the file.
        evidence_collector: The evidence collector.
        
    Returns:
        A tuple of (recommendation_grounder, recommendations).
    """
    # Create a recommendation grounder
    recommendation_grounder = RecommendationGrounder(evidence_collector)
    
    # Create recommendations
    recommendations = []
    
    # 1. Optimize calculate_average function
    recommendations.append(recommendation_grounder.ground_recommendation(
        type="optimization",
        title="Optimize calculate_average function",
        description="The calculate_average function can be optimized by using the built-in sum and len functions.",
        file_paths=[Path(file_path)],
        line_ranges=[(2, 19)],
        suggested_code="""
def calculate_average(values):
    \"\"\"Calculate the average of a list of values.
    
    Args:
        values: A list of numeric values.
        
    Returns:
        The average of the values.
    \"\"\"
    if not values:
        return 0
    
    return sum(values) / len(values)
"""
    ))
    
    # 2. Fix bare except in process_data function
    recommendations.append(recommendation_grounder.ground_recommendation(
        type="bug_fix",
        title="Fix bare except in process_data function",
        description="The process_data function uses a bare except clause, which can hide unexpected errors.",
        file_paths=[Path(file_path)],
        line_ranges=[(21, 38)],
        suggested_code="""
def process_data(data):
    \"\"\"Process a list of data items.
    
    Args:
        data: A list of data items.
        
    Returns:
        A list of processed items.
    \"\"\"
    result = []
    
    for item in data:
        if item is None:
            continue
        
        try:
            processed = item * 2
            result.append(processed)
        except Exception as e:
            print(f"Error processing item: {e}")
    
    return result
"""
    ))
    
    # 3. Use list comprehension in DataProcessor.process_items
    recommendations.append(recommendation_grounder.ground_recommendation(
        type="refactoring",
        title="Use list comprehension in DataProcessor.process_items",
        description="The process_items method can be simplified by using a list comprehension.",
        file_paths=[Path(file_path)],
        line_ranges=[(58, 68)],
        suggested_code="""
    def process_items(self, items):
        \"\"\"Process a list of items.
        
        Args:
            items: The items to process.
            
        Returns:
            The processed items.
        \"\"\"
        return [self.process_item(item) for item in items if item is not None]
"""
    ))
    
    return recommendation_grounder, recommendations


def verify_recommendations(evidence_collector, recommendation_grounder, recommendations):
    """Verify recommendations.
    
    Args:
        evidence_collector: The evidence collector.
        recommendation_grounder: The recommendation grounder.
        recommendations: The recommendations to verify.
        
    Returns:
        A tuple of (grounding_verifier, verification_results).
    """
    # Create a grounding verifier
    grounding_verifier = GroundingVerifier(evidence_collector, recommendation_grounder)
    
    # Verify recommendations
    verification_results = []
    for recommendation in recommendations:
        result = grounding_verifier.verify_recommendation(recommendation)
        verification_results.append(result)
    
    return grounding_verifier, verification_results


def main():
    """Main entry point for the script."""
    setup_logging()
    
    # Create a sample file
    file_path = create_sample_file()
    print(f"Created sample file: {file_path}")
    
    try:
        # Collect evidence
        print("\nCollecting evidence...")
        evidence_collector, evidence = collect_evidence(file_path)
        print(f"Collected {len(evidence)} evidence items")
        
        # Create patterns
        print("\nCreating patterns...")
        pattern_matcher = create_patterns()
        print(f"Created {len(pattern_matcher.patterns)} patterns")
        
        # Match patterns
        print("\nMatching patterns...")
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        pattern_matches = pattern_matcher.match_patterns(code, Path(file_path))
        print(f"Found {len(pattern_matches)} pattern matches:")
        for pattern, matches in pattern_matches:
            print(f"  {pattern.name}: {len(matches)} matches")
        
        # Create recommendations
        print("\nCreating recommendations...")
        recommendation_grounder, recommendations = create_recommendations(file_path, evidence_collector)
        print(f"Created {len(recommendations)} recommendations:")
        for recommendation in recommendations:
            print(f"  {recommendation.get_summary()}")
            print(f"    Location: {recommendation.get_location_str()}")
            print(f"    Evidence: {len(recommendation.evidence_ids)} items")
        
        # Verify recommendations
        print("\nVerifying recommendations...")
        grounding_verifier, verification_results = verify_recommendations(evidence_collector, recommendation_grounder, recommendations)
        print(f"Verified {len(verification_results)} recommendations:")
        for result in verification_results:
            recommendation = recommendation_grounder.get_recommendation(result.recommendation_id)
            print(f"  {recommendation.get_summary()}")
            print(f"    Grounded: {result.is_grounded}")
            print(f"    Evidence coverage: {result.evidence_coverage:.2f}")
            print(f"    Confidence: {result.confidence:.2f}")
            if result.issues:
                print(f"    Issues: {', '.join(result.issues)}")
        
        # Get grounded and ungrounded recommendations
        grounded = grounding_verifier.get_grounded_recommendations()
        ungrounded = grounding_verifier.get_ungrounded_recommendations()
        
        print(f"\nGrounded recommendations: {len(grounded)}")
        print(f"Ungrounded recommendations: {len(ungrounded)}")
    
    finally:
        # Clean up
        os.remove(file_path)
        print(f"\nRemoved sample file: {file_path}")


if __name__ == "__main__":
    main()
