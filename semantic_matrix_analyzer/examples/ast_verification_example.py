#!/usr/bin/env python3
"""
Example of using the AST verification system.

This script demonstrates how to use the AST verification system to verify code suggestions
before applying them.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.verification import (
    CodeSuggestion, VerificationResult, SuggestionVerifier,
    CodeChangeSimulator, SideEffectDetector, VerificationReporter
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
    total = sum(values)
    count = len(values)
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
            processed = item.process()
            result.append(processed)
        except:
            print("Error processing item")
    return result
""")
    
    return temp_path


def create_sample_suggestions(file_path):
    """Create sample code suggestions for testing.
    
    Args:
        file_path: The path to the sample file.
        
    Returns:
        A list of code suggestions.
    """
    # Read the sample file
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    
    # Create a list of suggestions
    suggestions = []
    
    # Suggestion 1: Add a check for empty values list
    suggestions.append(CodeSuggestion(
        file_path=Path(file_path),
        start_line=2,
        end_line=12,
        original_code="""def calculate_average(values):
    \"\"\"Calculate the average of a list of values.
    
    Args:
        values: A list of numeric values.
        
    Returns:
        The average of the values.
    \"\"\"
    total = sum(values)
    count = len(values)
    return total / count""",
        suggested_code="""def calculate_average(values):
    \"\"\"Calculate the average of a list of values.
    
    Args:
        values: A list of numeric values.
        
    Returns:
        The average of the values, or None if the list is empty.
    \"\"\"
    if not values:
        return None
    total = sum(values)
    count = len(values)
    return total / count""",
        description="Add a check for empty values list",
        confidence=0.9
    ))
    
    # Suggestion 2: Improve error handling in process_data
    suggestions.append(CodeSuggestion(
        file_path=Path(file_path),
        start_line=14,
        end_line=29,
        original_code="""def process_data(data):
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
            processed = item.process()
            result.append(processed)
        except:
            print("Error processing item")
    return result""",
        suggested_code="""def process_data(data):
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
            processed = item.process()
            result.append(processed)
        except Exception as e:
            print(f"Error processing item: {e}")
    return result""",
        description="Improve error handling in process_data",
        confidence=0.8
    ))
    
    # Suggestion 3: Change function signature (should be flagged as a side effect)
    suggestions.append(CodeSuggestion(
        file_path=Path(file_path),
        start_line=14,
        end_line=29,
        original_code="""def process_data(data):
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
            processed = item.process()
            result.append(processed)
        except:
            print("Error processing item")
    return result""",
        suggested_code="""def process_data(data, ignore_errors=False):
    \"\"\"Process a list of data items.
    
    Args:
        data: A list of data items.
        ignore_errors: Whether to ignore processing errors.
        
    Returns:
        A list of processed items.
    \"\"\"
    result = []
    for item in data:
        if item is None:
            continue
        try:
            processed = item.process()
            result.append(processed)
        except Exception as e:
            if not ignore_errors:
                print(f"Error processing item: {e}")
    return result""",
        description="Add ignore_errors parameter to process_data",
        confidence=0.7
    ))
    
    # Suggestion 4: Syntax error (should be flagged as invalid)
    suggestions.append(CodeSuggestion(
        file_path=Path(file_path),
        start_line=2,
        end_line=12,
        original_code="""def calculate_average(values):
    \"\"\"Calculate the average of a list of values.
    
    Args:
        values: A list of numeric values.
        
    Returns:
        The average of the values.
    \"\"\"
    total = sum(values)
    count = len(values)
    return total / count""",
        suggested_code="""def calculate_average(values):
    \"\"\"Calculate the average of a list of values.
    
    Args:
        values: A list of numeric values.
        
    Returns:
        The average of the values, or None if the list is empty.
    \"\"\"
    if not values:
        return None
    total = sum(values)
    count = len(values)
    return total / count""",
        description="Add a check for empty values list (with syntax error)",
        confidence=0.9
    ))
    
    return suggestions


def main():
    """Main entry point for the script."""
    setup_logging()
    
    # Create a sample file
    file_path = create_sample_file()
    print(f"Created sample file: {file_path}")
    
    # Create sample suggestions
    suggestions = create_sample_suggestions(file_path)
    print(f"Created {len(suggestions)} sample suggestions")
    
    # Create verifiers
    side_effect_detector = SideEffectDetector()
    suggestion_verifier = SuggestionVerifier(side_effect_detector)
    code_change_simulator = CodeChangeSimulator()
    verification_reporter = VerificationReporter()
    
    # Verify each suggestion
    for i, suggestion in enumerate(suggestions):
        print(f"\n=== Suggestion {i+1}: {suggestion.description} ===")
        
        # Verify the suggestion
        verification_result = suggestion_verifier.verify_suggestion(suggestion)
        
        # Store the verification result with the suggestion
        suggestion.verification_result = verification_result
        
        # Print a summary of the verification result
        summary = verification_reporter.summarize_verification_result(verification_result)
        print(f"Verification result: {summary}")
        
        # If the suggestion is valid, simulate the change
        if verification_result.is_valid:
            print("\nSimulating change...")
            simulation_result = code_change_simulator.simulate_change(suggestion)
            
            if simulation_result["success"]:
                print("Simulation successful")
                
                # Print analysis results
                if "analysis" in simulation_result and simulation_result["analysis"]:
                    print("\nAnalysis results:")
                    for category, results in simulation_result["analysis"].items():
                        print(f"  {category}:")
                        if isinstance(results, list):
                            for result in results:
                                if isinstance(result, dict):
                                    print(f"    - {result.get('type', 'Unknown')}: {result.get('message', '')}")
                                else:
                                    print(f"    - {result}")
                        else:
                            print(f"    {results}")
            else:
                print(f"Simulation failed: {simulation_result.get('error', 'Unknown error')}")
        
        # Generate a detailed report
        report = verification_reporter.generate_report(suggestion, verification_result)
        
        # Format the report as text
        text_report = verification_reporter.format_report(report, format="text")
        
        # Print the report
        print("\nDetailed report:")
        print(text_report)
    
    # Clean up
    os.remove(file_path)
    print(f"\nRemoved sample file: {file_path}")


if __name__ == "__main__":
    main()
