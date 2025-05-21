#!/usr/bin/env python3
"""
Example 7: Multi-Session Project

This example demonstrates a multi-session project where the system maintains a persistent
understanding of the project across multiple sessions, reducing cognitive load for the user.

Key concepts demonstrated:
1. Project-level memory
2. Cross-session understanding
3. Progressive refinement of understanding
4. Implicit concern recognition

Cognitive load reduction:
- The system builds a comprehensive understanding of the project over time
- Users don't need to re-explain project context in each session
- The system can recognize implicit concerns based on project history
- The system can make connections between different parts of the project
"""

import os
import sys
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semantic_matrix_analyzer.conversation.memory import Conversation, ConversationStore
from semantic_matrix_analyzer.conversation.memory.agent import AgentFactory, MemoryAugmentedAgent
from semantic_matrix_analyzer.conversation.memory.knowledge_graph import KnowledgeGraph, KnowledgeGraphBuilder


class ProjectMemoryAssistant:
    """An assistant that maintains memory across multiple project sessions."""
    
    def __init__(self, project_name: str, storage_dir: str):
        """Initialize the project memory assistant.
        
        Args:
            project_name: The name of the project.
            storage_dir: Directory to store conversations and knowledge graph.
        """
        self.project_name = project_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.knowledge_graph_path = self.storage_dir / "knowledge_graph.json"
        self.agent = AgentFactory.create_agent(self.storage_dir, self.knowledge_graph_path)
        self.conversation_id = None
        
        # Project components
        self.components = {}
        
        # Project concerns
        self.concerns = set()
    
    def start_session(self, session_name: str) -> str:
        """Start a new session.
        
        Args:
            session_name: The name of the session.
            
        Returns:
            The ID of the session.
        """
        self.conversation_id = self.agent.start_conversation()
        
        # Add project context to the conversation
        self.agent.process_message(f"I'm working on the {self.project_name} project.")
        
        # Add component context if available
        if self.components:
            components_str = ", ".join(f"{name} ({desc})" for name, desc in self.components.items())
            self.agent.process_message(f"The project has the following components: {components_str}")
        
        # Add concern context if available
        if self.concerns:
            concerns_str = ", ".join(self.concerns)
            self.agent.process_message(f"I'm concerned about: {concerns_str}")
        
        return self.conversation_id
    
    def process_message(self, message: str) -> str:
        """Process a message from the user.
        
        Args:
            message: The message from the user.
            
        Returns:
            The response from the agent.
        """
        # Update project context based on the message
        self._update_project_context(message)
        
        return self.agent.process_message(message)
    
    def _update_project_context(self, message: str) -> None:
        """Update project context based on the message.
        
        Args:
            message: The message from the user.
        """
        # Check for component definitions
        if "component" in message.lower() and ":" in message:
            parts = message.split(":", 1)
            component_part = parts[0].lower()
            if "component" in component_part:
                component_name = component_part.split("component", 1)[1].strip()
                component_desc = parts[1].strip()
                self.components[component_name] = component_desc
        
        # Check for concerns
        for concern in ["performance", "security", "maintainability", "scalability", "reliability"]:
            if concern in message.lower():
                self.concerns.add(concern)
    
    def get_active_intents(self) -> set:
        """Get the active intents for the current session.
        
        Returns:
            A set of active intent names.
        """
        return self.agent.get_active_intents()
    
    def get_project_summary(self) -> dict:
        """Get a summary of the project.
        
        Returns:
            A dictionary with project summary information.
        """
        # Get all conversations
        conversation_store = ConversationStore(self.storage_dir)
        conversations = conversation_store.get_all_conversations()
        
        # Get all intents
        all_intents = set()
        for conversation in conversations:
            all_intents.update(conversation.extracted_intents.keys())
        
        # Load the knowledge graph
        knowledge_graph = KnowledgeGraph(self.knowledge_graph_path)
        
        return {
            "project_name": self.project_name,
            "components": self.components,
            "concerns": self.concerns,
            "sessions": len(conversations),
            "intents": all_intents,
            "entities": len(knowledge_graph.entities),
            "relationships": len(knowledge_graph.relationships)
        }
    
    def close_session(self) -> None:
        """Close the current session."""
        self.agent.close_conversation()
        self.conversation_id = None


def simulate_session(assistant, session_name, messages):
    """Simulate a project session.
    
    Args:
        assistant: The project memory assistant.
        session_name: The name of the session.
        messages: A list of user messages.
        
    Returns:
        The session ID.
    """
    print(f"\n=== {session_name} ===")
    
    # Start a session
    session_id = assistant.start_session(session_name)
    print(f"Started session: {session_id}")
    
    # Process messages
    for message in messages:
        print(f"\nUser: {message}")
        response = assistant.process_message(message)
        print(f"AI: {response}")
    
    # Close the session
    assistant.close_session()
    print(f"Ended session: {session_id}")
    
    return session_id


def main():
    """Main entry point for the example."""
    # Create a directory for storing conversations
    storage_dir = Path("conversation_memory_examples/multi_session_project")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the project memory assistant
    assistant = ProjectMemoryAssistant("E-commerce Platform", storage_dir)
    
    # Simulate the first session: Project setup
    print("\nIn this first session, the user defines the project components and initial concerns.")
    session_1_messages = [
        "I'm setting up a new e-commerce platform project.",
        "Component Frontend: The user interface built with React.",
        "Component Backend: The API server built with Django.",
        "Component Database: PostgreSQL database for storing product and user data.",
        "I'm concerned about the performance of the product search functionality."
    ]
    session_1_id = simulate_session(assistant, "Session 1: Project Setup", session_1_messages)
    
    # Simulate a delay between sessions
    print("\nSimulating a delay between sessions (3 seconds)...")
    time.sleep(3)
    
    # Simulate the second session: Frontend development
    print("\nIn this second session, the user focuses on frontend development.")
    print("Note that they don't need to re-explain the project components.")
    session_2_messages = [
        "I'm working on the frontend component today.",
        "I need to implement the product search interface.",
        "I want to make sure it's responsive and user-friendly."
    ]
    session_2_id = simulate_session(assistant, "Session 2: Frontend Development", session_2_messages)
    
    # Simulate a delay between sessions
    print("\nSimulating a delay between sessions (3 seconds)...")
    time.sleep(3)
    
    # Simulate the third session: Backend development
    print("\nIn this third session, the user focuses on backend development.")
    print("The system remembers the performance concern from session 1.")
    session_3_messages = [
        "Now I'm working on the backend component.",
        "I need to implement the product search API.",
        "It should support filtering by category, price range, and ratings."
    ]
    session_3_id = simulate_session(assistant, "Session 3: Backend Development", session_3_messages)
    
    # Simulate a delay between sessions
    print("\nSimulating a delay between sessions (3 seconds)...")
    time.sleep(3)
    
    # Simulate the fourth session: Database optimization
    print("\nIn this fourth session, the user focuses on database optimization.")
    print("The system connects this to the performance concern from session 1.")
    session_4_messages = [
        "I need to optimize the database queries for product search.",
        "The current implementation is too slow for large product catalogs.",
        "I'm thinking about adding indexes to improve query performance."
    ]
    session_4_id = simulate_session(assistant, "Session 4: Database Optimization", session_4_messages)
    
    # Print a project summary
    print("\n=== Project Summary ===")
    summary = assistant.get_project_summary()
    
    print(f"Project: {summary['project_name']}")
    
    print("\nComponents:")
    for name, desc in summary['components'].items():
        print(f"- {name}: {desc}")
    
    print("\nConcerns:")
    for concern in summary['concerns']:
        print(f"- {concern}")
    
    print(f"\nSessions: {summary['sessions']}")
    print(f"Intents: {', '.join(summary['intents'])}")
    print(f"Knowledge graph: {summary['entities']} entities, {summary['relationships']} relationships")
    
    print("\nThis example demonstrates how the system maintains a persistent understanding")
    print("of a project across multiple sessions. The user doesn't need to re-explain")
    print("project components or concerns in each session, and the system can make")
    print("connections between different parts of the project based on the accumulated knowledge.")


if __name__ == "__main__":
    main()
