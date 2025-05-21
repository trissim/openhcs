#!/usr/bin/env python3
"""
Example of good code that follows clean code principles.
"""

class UserRepository:
    """
    Repository for user data operations.
    
    This class handles all database operations related to users,
    following the repository pattern for clean separation of concerns.
    """
    
    def __init__(self, database_connection):
        """
        Initialize the repository with a database connection.
        
        Args:
            database_connection: Connection to the database
        """
        self.database = database_connection
    
    def find_by_id(self, user_id):
        """
        Find a user by their ID.
        
        Args:
            user_id: The unique identifier of the user
            
        Returns:
            User object if found, None otherwise
            
        Raises:
            DatabaseConnectionError: If there's an issue with the database connection
        """
        try:
            query = "SELECT * FROM users WHERE id = %s"
            result = self.database.execute(query, (user_id,))
            return result.first()
        except DatabaseConnectionError as error:
            logger.error(f"Database connection error: {error}")
            raise
    
    def save(self, user):
        """
        Save a user to the database.
        
        Args:
            user: User object to save
            
        Returns:
            User object with updated ID
            
        Raises:
            ValidationError: If the user data is invalid
            DatabaseConnectionError: If there's an issue with the database connection
        """
        try:
            if not user.is_valid():
                raise ValidationError("Invalid user data")
                
            if user.id is None:
                # Insert new user
                query = "INSERT INTO users (name, email) VALUES (%s, %s) RETURNING id"
                result = self.database.execute(query, (user.name, user.email))
                user.id = result.first()[0]
            else:
                # Update existing user
                query = "UPDATE users SET name = %s, email = %s WHERE id = %s"
                self.database.execute(query, (user.name, user.email, user.id))
                
            return user
        except DatabaseConnectionError as error:
            logger.error(f"Database connection error: {error}")
            raise
        except ValidationError as error:
            logger.warning(f"Validation error: {error}")
            raise
