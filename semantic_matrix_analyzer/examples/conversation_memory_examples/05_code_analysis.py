#!/usr/bin/env python3
"""
Example 5: Code Analysis

This example demonstrates how conversation memory improves code analysis by remembering
user concerns and preferences.

Key concepts demonstrated:
1. Intent-driven code analysis
2. Personalized recommendations
3. Context-aware analysis

Cognitive load reduction:
- The system focuses analysis on what matters most to the user
- Users don't need to specify analysis criteria for each file
- The system remembers user preferences and applies them automatically
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory, MemoryAugmentedAgent


class MemoryAugmentedAnalyzer:
    """A code analyzer that uses conversation memory to improve analysis."""
    
    def __init__(self, storage_dir: str):
        """Initialize the analyzer.
        
        Args:
            storage_dir: Directory to store conversations and knowledge graph.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_graph_path = self.storage_dir / "knowledge_graph.json"
        self.agent = AgentFactory.create_agent(self.storage_dir, self.knowledge_graph_path)
        self.conversation_id = None
    
    def start_conversation(self, title: str = "Code Analysis") -> str:
        """Start a new conversation.
        
        Args:
            title: The title of the conversation.
            
        Returns:
            The ID of the conversation.
        """
        self.conversation_id = self.agent.start_conversation()
        return self.conversation_id
    
    def process_message(self, message: str) -> str:
        """Process a message from the user.
        
        Args:
            message: The message from the user.
            
        Returns:
            The response from the agent.
        """
        return self.agent.process_message(message)
    
    def get_active_intents(self) -> set:
        """Get the active intents for the current conversation.
        
        Returns:
            A set of active intent names.
        """
        return self.agent.get_active_intents()
    
    def analyze_code(self, code: str, file_path: str) -> dict:
        """Analyze code based on active intents.
        
        Args:
            code: The code to analyze.
            file_path: The path to the file containing the code.
            
        Returns:
            A dictionary with analysis results.
        """
        # Get active intents
        active_intents = self.get_active_intents()
        
        # Analyze code based on active intents
        results = {}
        
        if "Clean Code" in active_intents:
            results["clean_code"] = self._analyze_clean_code(code)
        
        if "Error Handling" in active_intents:
            results["error_handling"] = self._analyze_error_handling(code)
        
        if "Performance" in active_intents:
            results["performance"] = self._analyze_performance(code)
        
        return results
    
    def _analyze_clean_code(self, code: str) -> dict:
        """Analyze code for clean code principles.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        lines = code.splitlines()
        line_count = len(lines)
        
        # Check for long lines
        long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 80]
        
        # Check for long functions
        # This is a very simplistic approach
        function_lines = {}
        current_function = None
        function_start = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith("def "):
                if current_function:
                    function_lines[current_function] = i - function_start
                
                current_function = line.strip().split("def ")[1].split("(")[0].strip()
                function_start = i
            
            if line.strip().startswith("class "):
                current_function = None
        
        if current_function:
            function_lines[current_function] = len(lines) - function_start
        
        long_functions = {func: lines for func, lines in function_lines.items() if lines > 20}
        
        return {
            "line_count": line_count,
            "long_lines": long_lines,
            "long_functions": long_functions
        }
    
    def _analyze_error_handling(self, code: str) -> dict:
        """Analyze code for error handling.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        # Check for try-except blocks
        try_except_count = code.count("try:") if "try:" in code else 0
        
        # Check for bare excepts
        bare_except_count = code.count("except:") if "except:" in code else 0
        
        # Check for specific exception types
        specific_except_count = code.count("except ") if "except " in code else 0
        
        return {
            "try_except_count": try_except_count,
            "bare_except_count": bare_except_count,
            "specific_except_count": specific_except_count
        }
    
    def _analyze_performance(self, code: str) -> dict:
        """Analyze code for performance issues.
        
        Args:
            code: The code to analyze.
            
        Returns:
            A dictionary with analysis results.
        """
        # Check for list comprehensions vs. loops
        list_comprehension_count = code.count("[") and code.count("for") and code.count("]")
        
        # Check for generator expressions
        generator_expression_count = code.count("(") and code.count("for") and code.count(")")
        
        return {
            "list_comprehension_count": list_comprehension_count,
            "generator_expression_count": generator_expression_count
        }
    
    def close_conversation(self) -> None:
        """Close the current conversation."""
        self.agent.close_conversation()
        self.conversation_id = None


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/code_analysis")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the analyzer
    analyzer = MemoryAugmentedAnalyzer(storage_dir)
    
    # Sample code to analyze
    code_samples = {
        "sample1.py": """
def process_data(data):
    # This is a long function that does many things
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
""",
        "sample2.py": """
def calculate_statistics(values):
    if not values:
        return None
    
    total = sum(values)
    count = len(values)
    average = total / count
    
    squared_diff = [(x - average) ** 2 for x in values]
    variance = sum(squared_diff) / count
    std_dev = variance ** 0.5
    
    return {
        "count": count,
        "total": total,
        "average": average,
        "variance": variance,
        "std_dev": std_dev
    }
"""
    }
    
    # Start a conversation
    conversation_id = analyzer.start_conversation("Code Analysis")
    print(f"Started conversation: {conversation_id}")
    
    # First session: Focus on clean code
    print("\n=== Session 1: Clean Code Focus ===")
    response = analyzer.process_message("I want to improve the clean code in my project.")
    print(f"User: I want to improve the clean code in my project.")
    print(f"AI: {response}")
    
    # Get active intents
    active_intents = analyzer.get_active_intents()
    print(f"Active intents: {', '.join(active_intents)}")
    
    # Analyze code samples
    print("\nAnalyzing code samples with clean code focus:")
    for file_name, code in code_samples.items():
        print(f"\nAnalyzing {file_name}:")
        results = analyzer.analyze_code(code, file_name)
        for category, category_results in results.items():
            print(f"  {category}:")
            for key, value in category_results.items():
                print(f"    {key}: {value}")
    
    # Close the conversation
    analyzer.close_conversation()
    
    # Start a new conversation
    conversation_id = analyzer.start_conversation("Code Analysis 2")
    print(f"\n=== Session 2: Error Handling Focus ===")
    
    # Second session: Focus on error handling
    response = analyzer.process_message("I want to make sure my error handling is robust.")
    print(f"User: I want to make sure my error handling is robust.")
    print(f"AI: {response}")
    
    # Get active intents
    active_intents = analyzer.get_active_intents()
    print(f"Active intents: {', '.join(active_intents)}")
    
    # Analyze the same code samples
    print("\nAnalyzing code samples with error handling focus:")
    for file_name, code in code_samples.items():
        print(f"\nAnalyzing {file_name}:")
        results = analyzer.analyze_code(code, file_name)
        for category, category_results in results.items():
            print(f"  {category}:")
            for key, value in category_results.items():
                print(f"    {key}: {value}")
    
    # Close the conversation
    analyzer.close_conversation()
    
    print("\nThis example demonstrates how the system adapts its analysis based on the user's concerns.")
    print("In the first session, it focused on clean code issues.")
    print("In the second session, it focused on error handling issues.")
    print("The system automatically remembers these concerns and applies them to the analysis.")


if __name__ == "__main__":
    main()
