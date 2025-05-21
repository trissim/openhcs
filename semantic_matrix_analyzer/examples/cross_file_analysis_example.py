#!/usr/bin/env python3
"""
Example of using the cross-file analysis system.

This script demonstrates how to use the cross-file analysis system to analyze relationships
and dependencies between files, detect architectural patterns, inconsistencies, and
potential refactoring opportunities.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.cross_file import (
    DependencyGraph, DependencyExtractor, ArchitecturalPatternDetector,
    InconsistencyDetector, RefactoringOpportunityDetector
)


def setup_logging():
    """Set up logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_sample_files():
    """Create sample files for testing.
    
    Returns:
        A tuple of (temp_dir, file_paths).
    """
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Create sample files
    file_paths = []
    
    # 1. Create a model file
    model_path = os.path.join(temp_dir, "user_model.py")
    with open(model_path, "w", encoding="utf-8") as f:
        f.write("""
class User:
    \"\"\"A user in the system.\"\"\"
    
    def __init__(self, id, name, email):
        \"\"\"Initialize the user.
        
        Args:
            id: The user ID.
            name: The user name.
            email: The user email.
        \"\"\"
        self.id = id
        self.name = name
        self.email = email
    
    def get_id(self):
        \"\"\"Get the user ID.
        
        Returns:
            The user ID.
        \"\"\"
        return self.id
    
    def get_name(self):
        \"\"\"Get the user name.
        
        Returns:
            The user name.
        \"\"\"
        return self.name
    
    def get_email(self):
        \"\"\"Get the user email.
        
        Returns:
            The user email.
        \"\"\"
        return self.email
    
    def set_name(self, name):
        \"\"\"Set the user name.
        
        Args:
            name: The new name.
        \"\"\"
        self.name = name
    
    def set_email(self, email):
        \"\"\"Set the user email.
        
        Args:
            email: The new email.
        \"\"\"
        self.email = email
""")
    file_paths.append(model_path)
    
    # 2. Create a repository file
    repo_path = os.path.join(temp_dir, "user_repository.py")
    with open(repo_path, "w", encoding="utf-8") as f:
        f.write("""
from user_model import User

class UserRepository:
    \"\"\"A repository for users.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the repository.\"\"\"
        self.users = {}
    
    def get_user(self, id):
        \"\"\"Get a user by ID.
        
        Args:
            id: The user ID.
            
        Returns:
            The user, or None if not found.
        \"\"\"
        return self.users.get(id)
    
    def save_user(self, user):
        \"\"\"Save a user.
        
        Args:
            user: The user to save.
        \"\"\"
        self.users[user.get_id()] = user
    
    def delete_user(self, id):
        \"\"\"Delete a user.
        
        Args:
            id: The ID of the user to delete.
            
        Returns:
            True if the user was deleted, False otherwise.
        \"\"\"
        if id in self.users:
            del self.users[id]
            return True
        return False
    
    def get_all_users(self):
        \"\"\"Get all users.
        
        Returns:
            A list of all users.
        \"\"\"
        return list(self.users.values())
""")
    file_paths.append(repo_path)
    
    # 3. Create a service file
    service_path = os.path.join(temp_dir, "user_service.py")
    with open(service_path, "w", encoding="utf-8") as f:
        f.write("""
from user_model import User
from user_repository import UserRepository

class UserService:
    \"\"\"A service for user operations.\"\"\"
    
    def __init__(self, user_repository):
        \"\"\"Initialize the service.
        
        Args:
            user_repository: The user repository.
        \"\"\"
        self.user_repository = user_repository
    
    def get_user(self, id):
        \"\"\"Get a user by ID.
        
        Args:
            id: The user ID.
            
        Returns:
            The user, or None if not found.
        \"\"\"
        return self.user_repository.get_user(id)
    
    def create_user(self, id, name, email):
        \"\"\"Create a new user.
        
        Args:
            id: The user ID.
            name: The user name.
            email: The user email.
            
        Returns:
            The created user.
        \"\"\"
        user = User(id, name, email)
        self.user_repository.save_user(user)
        return user
    
    def update_user(self, id, name=None, email=None):
        \"\"\"Update a user.
        
        Args:
            id: The user ID.
            name: The new name (optional).
            email: The new email (optional).
            
        Returns:
            The updated user, or None if not found.
        \"\"\"
        user = self.user_repository.get_user(id)
        if user:
            if name:
                user.set_name(name)
            if email:
                user.set_email(email)
            self.user_repository.save_user(user)
        return user
    
    def delete_user(self, id):
        \"\"\"Delete a user.
        
        Args:
            id: The ID of the user to delete.
            
        Returns:
            True if the user was deleted, False otherwise.
        \"\"\"
        return self.user_repository.delete_user(id)
    
    def get_all_users(self):
        \"\"\"Get all users.
        
        Returns:
            A list of all users.
        \"\"\"
        return self.user_repository.get_all_users()
""")
    file_paths.append(service_path)
    
    # 4. Create a controller file
    controller_path = os.path.join(temp_dir, "user_controller.py")
    with open(controller_path, "w", encoding="utf-8") as f:
        f.write("""
from user_service import UserService
from user_repository import UserRepository

class UserController:
    \"\"\"A controller for user operations.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the controller.\"\"\"
        self.user_repository = UserRepository()
        self.user_service = UserService(self.user_repository)
    
    def handle_get_user(self, id):
        \"\"\"Handle a request to get a user.
        
        Args:
            id: The user ID.
            
        Returns:
            A dictionary with the user data, or an error message.
        \"\"\"
        user = self.user_service.get_user(id)
        if user:
            return {
                "id": user.get_id(),
                "name": user.get_name(),
                "email": user.get_email()
            }
        return {"error": "User not found"}
    
    def handle_create_user(self, id, name, email):
        \"\"\"Handle a request to create a user.
        
        Args:
            id: The user ID.
            name: The user name.
            email: The user email.
            
        Returns:
            A dictionary with the user data.
        \"\"\"
        user = self.user_service.create_user(id, name, email)
        return {
            "id": user.get_id(),
            "name": user.get_name(),
            "email": user.get_email()
        }
    
    def handle_update_user(self, id, name=None, email=None):
        \"\"\"Handle a request to update a user.
        
        Args:
            id: The user ID.
            name: The new name (optional).
            email: The new email (optional).
            
        Returns:
            A dictionary with the user data, or an error message.
        \"\"\"
        user = self.user_service.update_user(id, name, email)
        if user:
            return {
                "id": user.get_id(),
                "name": user.get_name(),
                "email": user.get_email()
            }
        return {"error": "User not found"}
    
    def handle_delete_user(self, id):
        \"\"\"Handle a request to delete a user.
        
        Args:
            id: The ID of the user to delete.
            
        Returns:
            A dictionary with the result.
        \"\"\"
        result = self.user_service.delete_user(id)
        if result:
            return {"success": True}
        return {"error": "User not found"}
    
    def handle_get_all_users(self):
        \"\"\"Handle a request to get all users.
        
        Returns:
            A list of dictionaries with user data.
        \"\"\"
        users = self.user_service.get_all_users()
        return [
            {
                "id": user.get_id(),
                "name": user.get_name(),
                "email": user.get_email()
            }
            for user in users
        ]
""")
    file_paths.append(controller_path)
    
    # 5. Create a main file
    main_path = os.path.join(temp_dir, "main.py")
    with open(main_path, "w", encoding="utf-8") as f:
        f.write("""
from user_controller import UserController

def main():
    \"\"\"Main entry point for the application.\"\"\"
    controller = UserController()
    
    # Create some users
    controller.handle_create_user(1, "Alice", "alice@example.com")
    controller.handle_create_user(2, "Bob", "bob@example.com")
    
    # Get all users
    users = controller.handle_get_all_users()
    print("All users:")
    for user in users:
        print(f"  {user['name']} ({user['email']})")
    
    # Update a user
    controller.handle_update_user(1, email="alice.new@example.com")
    
    # Get a user
    user = controller.handle_get_user(1)
    print(f"Updated user: {user['name']} ({user['email']})")
    
    # Delete a user
    controller.handle_delete_user(2)
    
    # Get all users again
    users = controller.handle_get_all_users()
    print("All users after deletion:")
    for user in users:
        print(f"  {user['name']} ({user['email']})")

if __name__ == "__main__":
    main()
""")
    file_paths.append(main_path)
    
    return temp_dir, file_paths


def analyze_dependencies(file_paths):
    """Analyze dependencies between files.
    
    Args:
        file_paths: A list of file paths.
        
    Returns:
        A dependency graph.
    """
    # Create a dependency graph
    dependency_graph = DependencyGraph()
    
    # Create a dependency extractor
    dependency_extractor = DependencyExtractor()
    
    # Extract dependencies from each file
    for file_path in file_paths:
        print(f"Extracting dependencies from {os.path.basename(file_path)}...")
        nodes, edges = dependency_extractor.extract_dependencies(Path(file_path))
        
        # Add nodes and edges to the graph
        for node in nodes:
            dependency_graph.add_node(node)
        
        for edge in edges:
            dependency_graph.add_edge(edge)
    
    return dependency_graph


def detect_patterns(dependency_graph):
    """Detect architectural patterns in the dependency graph.
    
    Args:
        dependency_graph: The dependency graph.
        
    Returns:
        A list of detected patterns.
    """
    # Create a pattern detector
    pattern_detector = ArchitecturalPatternDetector(dependency_graph)
    
    # Detect patterns
    patterns = pattern_detector.detect_patterns()
    
    return patterns


def detect_inconsistencies(dependency_graph):
    """Detect inconsistencies in the dependency graph.
    
    Args:
        dependency_graph: The dependency graph.
        
    Returns:
        A list of detected inconsistencies.
    """
    # Create an inconsistency detector
    inconsistency_detector = InconsistencyDetector(dependency_graph)
    
    # Detect inconsistencies
    inconsistencies = inconsistency_detector.detect_inconsistencies()
    
    return inconsistencies


def detect_refactoring_opportunities(dependency_graph):
    """Detect refactoring opportunities in the dependency graph.
    
    Args:
        dependency_graph: The dependency graph.
        
    Returns:
        A list of detected refactoring opportunities.
    """
    # Create a refactoring opportunity detector
    opportunity_detector = RefactoringOpportunityDetector(dependency_graph)
    
    # Detect opportunities
    opportunities = opportunity_detector.detect_opportunities()
    
    return opportunities


def main():
    """Main entry point for the script."""
    setup_logging()
    
    # Create sample files
    temp_dir, file_paths = create_sample_files()
    print(f"Created sample files in {temp_dir}")
    
    try:
        # Analyze dependencies
        print("\nAnalyzing dependencies...")
        dependency_graph = analyze_dependencies(file_paths)
        
        # Print statistics
        print(f"\nDependency graph statistics:")
        print(f"  Nodes: {len(dependency_graph.nodes)}")
        print(f"  Edges: {len(dependency_graph.edges)}")
        print(f"  Node types: {', '.join(dependency_graph.node_index.keys())}")
        print(f"  Edge types: {', '.join(dependency_graph.edge_index.keys())}")
        
        # Detect patterns
        print("\nDetecting architectural patterns...")
        patterns = detect_patterns(dependency_graph)
        
        print(f"Detected {len(patterns)} architectural patterns:")
        for i, pattern in enumerate(patterns):
            print(f"  {i+1}. {pattern.name}: {pattern.description}")
            print(f"     Confidence: {pattern.confidence}")
            print(f"     Nodes: {len(pattern.nodes)}")
            print(f"     Edges: {len(pattern.edges)}")
        
        # Detect inconsistencies
        print("\nDetecting inconsistencies...")
        inconsistencies = detect_inconsistencies(dependency_graph)
        
        print(f"Detected {len(inconsistencies)} inconsistencies:")
        for i, inconsistency in enumerate(inconsistencies):
            print(f"  {i+1}. {inconsistency.type}: {inconsistency.description}")
            print(f"     Severity: {inconsistency.severity}")
            print(f"     Nodes: {len(inconsistency.nodes)}")
            if inconsistency.suggestion:
                print(f"     Suggestion: {inconsistency.suggestion}")
        
        # Detect refactoring opportunities
        print("\nDetecting refactoring opportunities...")
        opportunities = detect_refactoring_opportunities(dependency_graph)
        
        print(f"Detected {len(opportunities)} refactoring opportunities:")
        for i, opportunity in enumerate(opportunities):
            print(f"  {i+1}. {opportunity.type}: {opportunity.description}")
            print(f"     Benefit: {opportunity.benefit}")
            print(f"     Effort: {opportunity.effort}")
            print(f"     Nodes: {len(opportunity.nodes)}")
            if opportunity.suggestion:
                print(f"     Suggestion: {opportunity.suggestion}")
    
    finally:
        # Clean up
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass
        
        try:
            os.rmdir(temp_dir)
        except:
            pass


if __name__ == "__main__":
    main()
