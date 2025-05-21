#!/usr/bin/env python3
"""
Example of using the Structural Analyzer to extract intent from code structure.
"""

import logging
import sys
import tempfile
import os
from pathlib import Path

# Add the parent directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantic_matrix_analyzer.intent.config.configuration import Configuration
from semantic_matrix_analyzer.cross_file.dependency_graph import DependencyGraph, DependencyExtractor
from semantic_matrix_analyzer.intent.analyzers.structural_analyzer import ConfigurableStructuralAnalyzer


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

    # Create a layered architecture

    # 1. Create a model layer
    os.makedirs(os.path.join(temp_dir, "model"), exist_ok=True)

    # 1.1. Create a user model file
    user_model_path = os.path.join(temp_dir, "model", "user.py")
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
""")
    file_paths.append(Path(user_model_path))

    # 1.2. Create an order model file
    order_model_path = os.path.join(temp_dir, "model", "order.py")
    with open(order_model_path, "w", encoding="utf-8") as f:
        f.write("""
from datetime import datetime
from typing import List

from model.user import User

class Order:
    \"\"\"An order in the system.\"\"\"

    def __init__(self, id: int, user: User, items: List[str], total: float):
        \"\"\"Initialize the order.

        Args:
            id: The order ID.
            user: The user who placed the order.
            items: The items in the order.
            total: The total amount of the order.
        \"\"\"
        self.id = id
        self.user = user
        self.items = items
        self.total = total
        self.date = datetime.now()

    def get_id(self) -> int:
        \"\"\"Get the order ID.

        Returns:
            The order ID.
        \"\"\"
        return self.id

    def get_user(self) -> User:
        \"\"\"Get the user who placed the order.

        Returns:
            The user.
        \"\"\"
        return self.user

    def get_items(self) -> List[str]:
        \"\"\"Get the items in the order.

        Returns:
            The items.
        \"\"\"
        return self.items

    def get_total(self) -> float:
        \"\"\"Get the total amount of the order.

        Returns:
            The total amount.
        \"\"\"
        return self.total

    def get_date(self) -> datetime:
        \"\"\"Get the date of the order.

        Returns:
            The date.
        \"\"\"
        return self.date
""")
    file_paths.append(Path(order_model_path))

    # 2. Create a repository layer
    os.makedirs(os.path.join(temp_dir, "repository"), exist_ok=True)

    # 2.1. Create a user repository file
    user_repository_path = os.path.join(temp_dir, "repository", "user_repository.py")
    with open(user_repository_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import Dict, List, Optional

from model.user import User

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

    # 2.2. Create an order repository file
    order_repository_path = os.path.join(temp_dir, "repository", "order_repository.py")
    with open(order_repository_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import Dict, List, Optional

from model.order import Order
from model.user import User

class OrderRepository:
    \"\"\"A repository for orders.\"\"\"

    def __init__(self):
        \"\"\"Initialize the repository.\"\"\"
        self.orders: Dict[int, Order] = {}

    def get_order(self, id: int) -> Optional[Order]:
        \"\"\"Get an order by ID.

        Args:
            id: The order ID.

        Returns:
            The order, or None if not found.
        \"\"\"
        return self.orders.get(id)

    def save_order(self, order: Order) -> None:
        \"\"\"Save an order.

        Args:
            order: The order to save.
        \"\"\"
        self.orders[order.get_id()] = order

    def delete_order(self, id: int) -> bool:
        \"\"\"Delete an order.

        Args:
            id: The ID of the order to delete.

        Returns:
            True if the order was deleted, False otherwise.
        \"\"\"
        if id in self.orders:
            del self.orders[id]
            return True
        return False

    def get_all_orders(self) -> List[Order]:
        \"\"\"Get all orders.

        Returns:
            A list of all orders.
        \"\"\"
        return list(self.orders.values())

    def get_orders_by_user(self, user: User) -> List[Order]:
        \"\"\"Get all orders by a user.

        Args:
            user: The user.

        Returns:
            A list of orders.
        \"\"\"
        return [order for order in self.orders.values() if order.get_user().get_id() == user.get_id()]
""")
    file_paths.append(Path(order_repository_path))

    # 3. Create a service layer
    os.makedirs(os.path.join(temp_dir, "service"), exist_ok=True)

    # 3.1. Create a user service file
    user_service_path = os.path.join(temp_dir, "service", "user_service.py")
    with open(user_service_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import List, Optional

from model.user import User
from repository.user_repository import UserRepository

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

    # 3.2. Create an order service file
    order_service_path = os.path.join(temp_dir, "service", "order_service.py")
    with open(order_service_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import List, Optional

from model.order import Order
from model.user import User
from repository.order_repository import OrderRepository
from service.user_service import UserService

class OrderService:
    \"\"\"A service for order operations.\"\"\"

    def __init__(self, order_repository: OrderRepository, user_service: UserService):
        \"\"\"Initialize the service.

        Args:
            order_repository: The order repository.
            user_service: The user service.
        \"\"\"
        self.order_repository = order_repository
        self.user_service = user_service

    def get_order(self, id: int) -> Optional[Order]:
        \"\"\"Get an order by ID.

        Args:
            id: The order ID.

        Returns:
            The order, or None if not found.
        \"\"\"
        return self.order_repository.get_order(id)

    def create_order(self, id: int, user_id: int, items: List[str], total: float) -> Optional[Order]:
        \"\"\"Create a new order.

        Args:
            id: The order ID.
            user_id: The ID of the user who placed the order.
            items: The items in the order.
            total: The total amount of the order.

        Returns:
            The created order, or None if the user was not found.
        \"\"\"
        user = self.user_service.get_user(user_id)
        if not user:
            return None

        order = Order(id, user, items, total)
        self.order_repository.save_order(order)
        return order

    def delete_order(self, id: int) -> bool:
        \"\"\"Delete an order.

        Args:
            id: The ID of the order to delete.

        Returns:
            True if the order was deleted, False otherwise.
        \"\"\"
        return self.order_repository.delete_order(id)

    def get_all_orders(self) -> List[Order]:
        \"\"\"Get all orders.

        Returns:
            A list of all orders.
        \"\"\"
        return self.order_repository.get_all_orders()

    def get_orders_by_user(self, user_id: int) -> List[Order]:
        \"\"\"Get all orders by a user.

        Args:
            user_id: The user ID.

        Returns:
            A list of orders.
        \"\"\"
        user = self.user_service.get_user(user_id)
        if not user:
            return []

        return self.order_repository.get_orders_by_user(user)
""")
    file_paths.append(Path(order_service_path))

    # 4. Create a controller layer
    os.makedirs(os.path.join(temp_dir, "controller"), exist_ok=True)

    # 4.1. Create a user controller file
    user_controller_path = os.path.join(temp_dir, "controller", "user_controller.py")
    with open(user_controller_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import Dict, List, Union

from service.user_service import UserService

class UserController:
    \"\"\"A controller for user operations.\"\"\"

    def __init__(self, user_service: UserService):
        \"\"\"Initialize the controller.

        Args:
            user_service: The user service.
        \"\"\"
        self.user_service = user_service

    def get_user(self, id: int) -> Dict[str, Union[int, str, Dict[str, str]]]:
        \"\"\"Get a user by ID.

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

    def create_user(self, id: int, name: str, email: str) -> Dict[str, Union[int, str]]:
        \"\"\"Create a new user.

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

    def delete_user(self, id: int) -> Dict[str, Union[bool, str]]:
        \"\"\"Delete a user.

        Args:
            id: The ID of the user to delete.

        Returns:
            A dictionary with the result.
        \"\"\"
        result = self.user_service.delete_user(id)
        if result:
            return {"success": True}
        return {"error": "User not found"}

    def get_all_users(self) -> List[Dict[str, Union[int, str]]]:
        \"\"\"Get all users.

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
    file_paths.append(Path(user_controller_path))

    # 4.2. Create an order controller file
    order_controller_path = os.path.join(temp_dir, "controller", "order_controller.py")
    with open(order_controller_path, "w", encoding="utf-8") as f:
        f.write("""
from typing import Dict, List, Union

from service.order_service import OrderService

class OrderController:
    \"\"\"A controller for order operations.\"\"\"

    def __init__(self, order_service: OrderService):
        \"\"\"Initialize the controller.

        Args:
            order_service: The order service.
        \"\"\"
        self.order_service = order_service

    def get_order(self, id: int) -> Dict[str, Union[int, str, List[str], float, Dict[str, str]]]:
        \"\"\"Get an order by ID.

        Args:
            id: The order ID.

        Returns:
            A dictionary with the order data, or an error message.
        \"\"\"
        order = self.order_service.get_order(id)
        if order:
            return {
                "id": order.get_id(),
                "user": {
                    "id": order.get_user().get_id(),
                    "name": order.get_user().get_name(),
                    "email": order.get_user().get_email()
                },
                "items": order.get_items(),
                "total": order.get_total(),
                "date": order.get_date().isoformat()
            }
        return {"error": "Order not found"}

    def create_order(self, id: int, user_id: int, items: List[str], total: float) -> Dict[str, Union[int, str, List[str], float, Dict[str, str]]]:
        \"\"\"Create a new order.

        Args:
            id: The order ID.
            user_id: The ID of the user who placed the order.
            items: The items in the order.
            total: The total amount of the order.

        Returns:
            A dictionary with the order data, or an error message.
        \"\"\"
        order = self.order_service.create_order(id, user_id, items, total)
        if order:
            return {
                "id": order.get_id(),
                "user": {
                    "id": order.get_user().get_id(),
                    "name": order.get_user().get_name(),
                    "email": order.get_user().get_email()
                },
                "items": order.get_items(),
                "total": order.get_total(),
                "date": order.get_date().isoformat()
            }
        return {"error": "User not found"}

    def delete_order(self, id: int) -> Dict[str, Union[bool, str]]:
        \"\"\"Delete an order.

        Args:
            id: The ID of the order to delete.

        Returns:
            A dictionary with the result.
        \"\"\"
        result = self.order_service.delete_order(id)
        if result:
            return {"success": True}
        return {"error": "Order not found"}

    def get_all_orders(self) -> List[Dict[str, Union[int, str, List[str], float, Dict[str, str]]]]:
        \"\"\"Get all orders.

        Returns:
            A list of dictionaries with order data.
        \"\"\"
        orders = self.order_service.get_all_orders()
        return [
            {
                "id": order.get_id(),
                "user": {
                    "id": order.get_user().get_id(),
                    "name": order.get_user().get_name(),
                    "email": order.get_user().get_email()
                },
                "items": order.get_items(),
                "total": order.get_total(),
                "date": order.get_date().isoformat()
            }
            for order in orders
        ]

    def get_orders_by_user(self, user_id: int) -> List[Dict[str, Union[int, str, List[str], float, Dict[str, str]]]]:
        \"\"\"Get all orders by a user.

        Args:
            user_id: The user ID.

        Returns:
            A list of dictionaries with order data.
        \"\"\"
        orders = self.order_service.get_orders_by_user(user_id)
        return [
            {
                "id": order.get_id(),
                "user": {
                    "id": order.get_user().get_id(),
                    "name": order.get_user().get_name(),
                    "email": order.get_user().get_email()
                },
                "items": order.get_items(),
                "total": order.get_total(),
                "date": order.get_date().isoformat()
            }
            for order in orders
        ]
""")
    file_paths.append(Path(order_controller_path))

    return temp_dir, file_paths


def analyze_structure(file_paths):
    """Analyze the structure of a codebase.

    Args:
        file_paths: A list of file paths to analyze.

    Returns:
        A list of structural intents.
    """
    # Create a dependency graph
    dependency_graph = DependencyGraph()
    dependency_extractor = DependencyExtractor()

    # Extract dependencies from each file
    for file_path in file_paths:
        nodes, edges = dependency_extractor.extract_dependencies(file_path)

        # Add nodes and edges to the graph
        for node in nodes:
            dependency_graph.add_node(node)

        for edge in edges:
            dependency_graph.add_edge(edge)

    # Create a configuration
    config = Configuration()

    # Create a structural analyzer
    analyzer = ConfigurableStructuralAnalyzer(dependency_graph, config)

    # Analyze the dependency graph
    intents = analyzer.analyze_dependency_graph()

    return intents


def main():
    """Main entry point for the example."""
    setup_logging()

    # Create sample files
    temp_dir, file_paths = create_sample_files()
    print(f"Created sample files in {temp_dir}")

    try:
        # Analyze the structure
        intents = analyze_structure(file_paths)

        # Print the intents
        print("\nStructural Intents:")
        print("=" * 80)
        for intent in intents:
            print(f"Pattern: {intent.pattern_name}")
            print(f"Description: {intent.description}")
            print(f"Confidence: {intent.confidence:.2f}")
            print(f"Components: {len(intent.components)}")
            print(f"Relationships: {len(intent.relationships)}")
            print("-" * 80)

    finally:
        # Clean up
        for file_path in file_paths:
            try:
                os.remove(file_path)
            except:
                pass

        # Remove directories
        for dir_name in ["controller", "service", "repository", "model"]:
            try:
                os.rmdir(os.path.join(temp_dir, dir_name))
            except:
                pass

        try:
            os.rmdir(temp_dir)
        except:
            pass


if __name__ == "__main__":
    main()
