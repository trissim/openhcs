"""
Memory conversion exceptions for OpenHCS.

This module defines exceptions for memory conversion operations,
enforcing Clause 65 (Fail Loudly) and Clause 88 (No Inferred Capabilities).
"""



class MemoryConversionError(Exception):
    """
    Exception raised when memory conversion fails.
    
    This exception is raised when a memory conversion operation fails and
    CPU fallback is not explicitly authorized.
    
    Attributes:
        source_type: The source memory type
        target_type: The target memory type
        method: The conversion method that was attempted
        reason: The reason for the failure
    """
    
    def __init__(self, source_type: str, target_type: str, method: str, reason: str):
        """
        Initialize a MemoryConversionError.
        
        Args:
            source_type: The source memory type
            target_type: The target memory type
            method: The conversion method that was attempted
            reason: The reason for the failure
        """
        self.source_type = source_type
        self.target_type = target_type
        self.method = method
        self.reason = reason
        
        message = (
            f"Cannot convert from {source_type} to {target_type} using {method}. "
            f"Reason: {reason}. "
            f"No CPU fallback permitted unless explicitly authorized via allow_cpu_roundtrip=True."
        )
        
        super().__init__(message)
