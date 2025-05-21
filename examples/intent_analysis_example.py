#!/usr/bin/env python3
"""
Example of using the Structural Intent Analysis system to extract intent from code structure.
"""

import ast
import logging
import sys
import tempfile
import os
from pathlib import Path

# Add the semantic_matrix_analyzer directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "semantic_matrix_analyzer"))

from intent.config.configuration import Configuration
from intent.analyzers.intent_analyzer import ConfigurableIntentAnalyzer


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
    file_paths.append(Path(user_model_path))
    
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
    file_paths.append(Path(user_repository_path))
    
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
    file_paths.append(Path(user_service_path))
    
    return temp_dir, file_paths


def analyze_codebase(file_paths):
    """Analyze a codebase to extract intent.
    
    Args:
        file_paths: A list of file paths to analyze.
        
    Returns:
        A report of the analysis.
    """
    # Create a configuration
    config = Configuration()
    
    # Create an intent analyzer
    analyzer = ConfigurableIntentAnalyzer(config)
    
    # Analyze the codebase
    report = analyzer.analyze_codebase(file_paths)
    
    # Format the report
    formatted_report = analyzer.format_report(report, "markdown")
    
    return formatted_report


def main():
    """Main entry point for the example."""
    setup_logging()
    
    # Create sample files
    temp_dir, file_paths = create_sample_files()
    print(f"Created sample files in {temp_dir}")
    
    try:
        # Analyze the codebase
        report = analyze_codebase(file_paths)
        
        # Print the report
        print("\nIntent Analysis Report:")
        print("=" * 80)
        print(report)
    
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
