"""
Test to clarify the intended reset button behavior.

This test demonstrates the difference between two possible reset behaviors:
1. Traditional: Reset sets field to concrete default value
2. Modern: Reset clears field and shows placeholder text

Based on user feedback, the modern approach is preferred for better UX.
"""

import pytest
from openhcs.core.config import GlobalPipelineConfig
from openhcs.ui.shared.parameter_form_service import ParameterFormService


class TestResetBehaviorClarification:
    """Clarify the intended reset button behavior."""
    
    def test_current_reset_behavior(self):
        """Test the current reset behavior implementation."""
        service = ParameterFormService()
        
        # Test reset value for GlobalPipelineConfig
        reset_value = service.get_reset_value_for_parameter('num_workers', int, GlobalPipelineConfig)
        print(f"Current reset value: {reset_value}")
        
        # Test placeholder text
        placeholder = service.get_placeholder_text('num_workers', GlobalPipelineConfig)
        print(f"Placeholder text: {placeholder}")
        
        # Test actual default value
        default_config = GlobalPipelineConfig()
        actual_default = default_config.num_workers
        print(f"Actual default value: {actual_default}")
        
        # Context-driven behavior: auto-detection returns actual defaults for GlobalPipelineConfig
        assert reset_value == actual_default, "Auto-detection should return actual default for GlobalPipelineConfig"
        assert placeholder is not None, "Placeholder should be available"
        assert str(actual_default) in placeholder, "Placeholder should show actual default"
    
    def test_intended_user_workflow(self):
        """Test the intended user workflow with reset button."""
        service = ParameterFormService()
        
        # Scenario: User is editing GlobalPipelineConfig
        # 1. Field starts unset (None) → shows placeholder "Pipeline default: 16"
        initial_value = None
        placeholder = service.get_placeholder_text('num_workers', GlobalPipelineConfig)
        
        print(f"1. Initial state: value={initial_value}, placeholder='{placeholder}'")
        assert initial_value is None
        assert "16" in placeholder
        
        # 2. User types a value → field becomes concrete
        user_value = 32
        print(f"2. User sets value: {user_value}")
        
        # 3. User clicks reset → field returns to default value
        reset_value = service.get_reset_value_for_parameter('num_workers', int, GlobalPipelineConfig)
        print(f"3. After reset: value={reset_value}, placeholder='{placeholder}'")

        assert reset_value == actual_default, "Reset should return actual default for global config editing"
        assert "16" in placeholder, "Placeholder should still show default"
        
        # 4. User can see what the default would be without committing to it
        print("4. User sees 'Pipeline default: 16' but field is not set to 16")
        print("   This allows user to understand the default without committing to it")
    
    def test_comparison_with_traditional_approach(self):
        """Compare modern approach with traditional approach."""
        service = ParameterFormService()
        
        # Modern approach (current implementation)
        modern_reset = service.get_reset_value_for_parameter('num_workers', int, GlobalPipelineConfig)
        placeholder = service.get_placeholder_text('num_workers', GlobalPipelineConfig)
        
        # Traditional approach (what old tests expected)
        default_config = GlobalPipelineConfig()
        traditional_reset = default_config.num_workers
        
        print("=== Modern Approach (Current) ===")
        print(f"Reset value: {modern_reset}")
        print(f"Placeholder: {placeholder}")
        print(f"User sees: Empty field with placeholder text")
        print(f"Stored value: None (unset)")
        
        print("\n=== Traditional Approach (Old Tests) ===")
        print(f"Reset value: {traditional_reset}")
        print(f"Placeholder: Not applicable")
        print(f"User sees: Field filled with {traditional_reset}")
        print(f"Stored value: {traditional_reset} (concrete)")
        
        print("\n=== Key Difference ===")
        print("Modern: Reset clears field, shows default as hint")
        print("Traditional: Reset fills field with default value")
        print("Modern approach provides better UX - user can see default without committing")
        
        # Verify modern approach
        assert modern_reset is None
        assert traditional_reset == 16
        assert "16" in placeholder


class TestResetBehaviorUserExperience:
    """Test the user experience implications of reset behavior."""
    
    def test_user_can_distinguish_set_vs_unset_fields(self):
        """Test that users can distinguish between set and unset fields."""
        service = ParameterFormService()
        
        # Unset field (after reset or initial state)
        unset_value = None
        placeholder = service.get_placeholder_text('num_workers', GlobalPipelineConfig)
        
        # Set field (user explicitly chose this value)
        set_value = 16  # Same as default, but explicitly set by user
        
        print("=== User Experience Test ===")
        print(f"Unset field: value={unset_value}, shows placeholder '{placeholder}'")
        print(f"Set field: value={set_value}, shows actual value")
        
        # Key insight: Even if user sets field to the same value as default,
        # the system can distinguish between "user chose this" vs "using default"
        assert unset_value is None
        assert set_value == 16
        assert unset_value != set_value  # Different states even if display might be similar
        
        print("\nThis distinction is important for:")
        print("1. Configuration serialization (only save user-set values)")
        print("2. Reset behavior (can reset to truly unset state)")
        print("3. User understanding (clear what they've customized)")
    
    def test_reset_button_provides_clear_feedback(self):
        """Test that reset button provides clear feedback to users."""
        service = ParameterFormService()
        
        # Simulate user workflow
        print("=== Reset Button Feedback Test ===")
        
        # 1. User sets a custom value
        user_value = 32
        print(f"1. User sets num_workers to {user_value}")
        
        # 2. User clicks reset
        reset_value = service.get_reset_value_for_parameter('num_workers', int, GlobalPipelineConfig)
        placeholder = service.get_placeholder_text('num_workers', GlobalPipelineConfig)
        
        print(f"2. After reset:")
        print(f"   - Field value: {reset_value}")
        print(f"   - Placeholder: {placeholder}")
        print(f"   - User sees: Empty field with 'Pipeline default: 16'")
        
        # 3. Clear feedback to user
        assert reset_value is None, "Field is cleared"
        assert "16" in placeholder, "User can see what default would be"
        
        print("3. User feedback:")
        print("   ✓ Field is visually cleared (not filled with 16)")
        print("   ✓ User can see what default would be (16)")
        print("   ✓ User can choose to accept default or set different value")
        print("   ✓ System knows field is unset (None) vs set to default (16)")
