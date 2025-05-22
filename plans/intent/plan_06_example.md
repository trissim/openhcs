# plan_06_example.md
## Component: Example Application

### Objective
Create an example application that demonstrates the Structural Intent Analysis system. This example will show how to extract intent from method/class/variable names, type hints, and the dependency graph, and how to combine these intents into a hierarchical model.

### Plan
1. Create a sample codebase with various naming patterns, type hints, and structural patterns
2. Implement an example script that extracts intent from the sample codebase
3. Show how to combine intents from different sources
4. Generate a report of the extracted intents
5. Visualize the intent hierarchy

### Findings
An example application will help users understand how to use the Structural Intent Analysis system and what kind of insights it can provide. The example should cover a variety of naming patterns, type hints, and structural patterns to demonstrate the full capabilities of the system.

### Implementation Draft

```python
#!/usr/bin/env python3
"""
Example of using the Structural Intent Analysis system.

This script demonstrates how to extract intent from method/class/variable names,
type hints, and the dependency graph, and how to combine these intents into a
hierarchical model.
"""

import ast
import logging
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.intent.models.intent import (
    Intent, NameIntent, TypeIntent, StructuralIntent, IntentType,
    IntentHierarchy, CodeLocation
)
from semantic_matrix_analyzer.intent.analyzers.name_analyzer import NameAnalyzer
from semantic_matrix_analyzer.intent.analyzers.type_analyzer import (
    TypeHintAnalyzer, TypeHintExtractor
)
from semantic_matrix_analyzer.cross_file.dependency_graph import (
    DependencyGraph, DependencyExtractor
)
from semantic_matrix_analyzer.intent.analyzers.structural_analyzer import (
    StructuralAnalyzer, ArchitecturalIntentExtractor
)
from semantic_matrix_analyzer.intent.integration.combiner import IntentCombiner
from semantic_matrix_analyzer.intent.integration.hierarchy import HierarchyBuilder
from semantic_matrix_analyzer.intent.integration.reporter import IntentReporter


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
    
    # 1. Create a user model file
    user_model_path = os.path.join(temp_dir, "user_model.py")
    with open(user_model_path, "w", encoding="utf-8") as f:
        f.write("""
class User:
    \"\"\"A user in the system.\"\"\"
    
    def __init__(self, id: int, name: str, email: str):
        \"\"\"Initialize the user.
        
        Args:
            id: The user ID.
            name: The user name.
            email: The user email.
        \"\"\"
        self.id = id
        self.name = name
        self.email = email
    
    def get_id(self) -> int:
        \"\"\"Get the user ID.
        
        Returns:
            The user ID.
        \"\"\"
        return self.id
    
    def get_name(self) -> str:
        \"\"\"Get the user name.
        
        Returns:
            The user name.
        \"\"\"
        return self.name
    
    def get_email(self) -> str:
        \"\"\"Get the user email.
        
        Returns:
            The user email.
        \"\"\"
        return self.email
    
    def set_name(self, name: str) -> None:
        \"\"\"Set the user name.
        
        Args:
            name: The new name.
        \"\"\"
        self.name = name
    
    def set_email(self, email: str) -> None:
        \"\"\"Set the user email.
        
        Args:
            email: The new email.
        \"\"\"
        self.email = email
""")
    file_paths.append(user_model_path)
    
    # 2. Create a user repository file
    user_repository_path = os.path.join(temp_dir, "user_repository.py")
    with open(user_repository_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import Dict, List, Optional

from user_model import User

class UserRepository:
    \"\"\"A repository for users.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the repository.\"\"\"
        self.users: Dict[int, User] = {}
    
    def get_user(self, id: int) -> Optional[User]:
        \"\"\"Get a user by ID.
        
        Args:
            id: The user ID.
            
        Returns:
            The user, or None if not found.
        \"\"\"
        return self.users.get(id)
    
    def save_user(self, user: User) -> None:
        \"\"\"Save a user.
        
        Args:
            user: The user to save.
        \"\"\"
        self.users[user.get_id()] = user
    
    def delete_user(self, id: int) -> bool:
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
    
    def get_all_users(self) -> List[User]:
        \"\"\"Get all users.
        
        Returns:
            A list of all users.
        \"\"\"
        return list(self.users.values())
""")
    file_paths.append(user_repository_path)
    
    # 3. Create a user service file
    user_service_path = os.path.join(temp_dir, "user_service.py")
    with open(user_service_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import List, Optional

from user_model import User
from user_repository import UserRepository

class UserService:
    \"\"\"A service for user operations.\"\"\"
    
    def __init__(self, user_repository: UserRepository):
        \"\"\"Initialize the service.
        
        Args:
            user_repository: The user repository.
        \"\"\"
        self.user_repository = user_repository
    
    def get_user(self, id: int) -> Optional[User]:
        \"\"\"Get a user by ID.
        
        Args:
            id: The user ID.
            
        Returns:
            The user, or None if not found.
        \"\"\"
        return self.user_repository.get_user(id)
    
    def create_user(self, id: int, name: str, email: str) -> User:
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
    
    def update_user(self, id: int, name: Optional[str] = None, email: Optional[str] = None) -> Optional[User]:
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
    
    def delete_user(self, id: int) -> bool:
        \"\"\"Delete a user.
        
        Args:
            id: The ID of the user to delete.
            
        Returns:
            True if the user was deleted, False otherwise.
        \"\"\"
        return self.user_repository.delete_user(id)
    
    def get_all_users(self) -> List[User]:
        \"\"\"Get all users.
        
        Returns:
            A list of all users.
        \"\"\"
        return self.user_repository.get_all_users()
""")
    file_paths.append(user_service_path)
    
    # 4. Create a user controller file
    user_controller_path = os.path.join(temp_dir, "user_controller.py")
    with open(user_controller_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import Dict, List, Optional, Union

from user_service import UserService
from user_repository import UserRepository

class UserController:
    \"\"\"A controller for user operations.\"\"\"
    
    def __init__(self):
        \"\"\"Initialize the controller.\"\"\"
        self.user_repository = UserRepository()
        self.user_service = UserService(self.user_repository)
    
    def handle_get_user(self, id: int) -> Dict[str, Union[int, str, Dict[str, str]]]:
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
    
    def handle_create_user(self, id: int, name: str, email: str) -> Dict[str, Union[int, str]]:
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
    
    def handle_update_user(self, id: int, name: Optional[str] = None, email: Optional[str] = None) -> Dict[str, Union[int, str, Dict[str, str]]]:
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
    
    def handle_delete_user(self, id: int) -> Dict[str, Union[bool, str]]:
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
    
    def handle_get_all_users(self) -> List[Dict[str, Union[int, str]]]:
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
    file_paths.append(user_controller_path)
    
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


def analyze_names(file_paths):
    """Analyze names in the codebase.
    
    Args:
        file_paths: A list of file paths.
        
    Returns:
        A list of NameIntent objects.
    """
    name_analyzer = NameAnalyzer()
    name_intents = []
    
    for file_path in file_paths:
        # Parse the file
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        tree = ast.parse(code, filename=file_path)
        
        # Analyze class names
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                intent = name_analyzer.analyze_class_name(
                    node.name,
                    Path(file_path),
                    node.lineno
                )
                name_intents.append(intent)
            
            elif isinstance(node, ast.FunctionDef):
                # Check if this is a method
                is_method = False
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef) and node in parent.body:
                        is_method = True
                        break
                
                if is_method:
                    intent = name_analyzer.analyze_method_name(
                        node.name,
                        Path(file_path),
                        node.lineno
                    )
                else:
                    intent = name_analyzer.analyze_name(
                        node.name,
                        "function",
                        Path(file_path),
                        node.lineno
                    )
                name_intents.append(intent)
            
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        intent = name_analyzer.analyze_variable_name(
                            target.id,
                            Path(file_path),
                            node.lineno
                        )
                        name_intents.append(intent)
    
    return name_intents


def analyze_type_hints(file_paths):
    """Analyze type hints in the codebase.
    
    Args:
        file_paths: A list of file paths.
        
    Returns:
        A list of TypeIntent objects.
    """
    type_hint_extractor = TypeHintExtractor()
    type_hint_analyzer = TypeHintAnalyzer()
    type_intents = []
    
    for file_path in file_paths:
        # Parse the file
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
        
        tree = ast.parse(code, filename=file_path)
        
        # Extract type hints from functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                type_hints = type_hint_extractor.extract_type_hints_from_ast(node)
                
                # Analyze parameter type hints
                for param_name, type_hint in type_hints.items():
                    if param_name != "return":
                        intent = type_hint_analyzer.analyze_parameter_type(
                            param_name,
                            type_hint,
                            Path(file_path),
                            node.lineno
                        )
                        type_intents.append(intent)
                
                # Analyze return type hint
                if "return" in type_hints:
                    intent = type_hint_analyzer.analyze_return_type(
                        type_hints["return"],
                        Path(file_path),
                        node.lineno
                    )
                    type_intents.append(intent)
    
    return type_intents


def analyze_structure(file_paths):
    """Analyze the structure of the codebase.
    
    Args:
        file_paths: A list of file paths.
        
    Returns:
        A list of StructuralIntent objects.
    """
    # Create a dependency graph
    dependency_graph = DependencyGraph()
    dependency_extractor = DependencyExtractor()
    
    # Extract dependencies from each file
    for file_path in file_paths:
        nodes, edges = dependency_extractor.extract_dependencies(Path(file_path))
        
        # Add nodes and edges to the graph
        for node in nodes:
            dependency_graph.add_node(node)
        
        for edge in edges:
            dependency_graph.add_edge(edge)
    
    # Create an architectural intent extractor
    intent_extractor = ArchitecturalIntentExtractor(dependency_graph)
    
    # Extract architectural intents
    structural_intents = intent_extractor.extract_architectural_intents()
    
    return structural_intents


def main():
    """Main entry point for the script."""
    setup_logging()
    
    # Create sample files
    temp_dir, file_paths = create_sample_files()
    print(f"Created sample files in {temp_dir}")
    
    try:
        # Analyze names
        print("\nAnalyzing names...")
        name_intents = analyze_names(file_paths)
        print(f"Found {len(name_intents)} name intents")
        
        # Analyze type hints
        print("\nAnalyzing type hints...")
        type_intents = analyze_type_hints(file_paths)
        print(f"Found {len(type_intents)} type intents")
        
        # Analyze structure
        print("\nAnalyzing structure...")
        structural_intents = analyze_structure(file_paths)
        print(f"Found {len(structural_intents)} structural intents")
        
        # Combine intents
        print("\nCombining intents...")
        intent_combiner = IntentCombiner()
        combined_intents = intent_combiner.combine_intents([name_intents, type_intents, structural_intents])
        print(f"Combined into {len(combined_intents)} intents")
        
        # Build hierarchy
        print("\nBuilding intent hierarchy...")
        hierarchy_builder = HierarchyBuilder()
        intent_hierarchy = hierarchy_builder.build_hierarchy(combined_intents)
        print(f"Built hierarchy with {len(intent_hierarchy.intents)} intents")
        
        # Generate report
        print("\nGenerating report...")
        intent_reporter = IntentReporter()
        report = intent_reporter.generate_report(intent_hierarchy)
        
        # Format report
        formatted_report = intent_reporter.format_report(report, "text")
        print("\nIntent Analysis Report:")
        print(formatted_report)
    
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
```
