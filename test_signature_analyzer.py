#!/usr/bin/env python3
"""
Test script to debug SignatureAnalyzer behavior with decorator-added parameters.
"""

import sys
import os
sys.path.insert(0, '/home/ts/code/projects/textual-window')

from openhcs.textual_tui.widgets.shared.signature_analyzer import SignatureAnalyzer, DocstringExtractor

def test_decorator_parameters():
    """Test what SignatureAnalyzer detects for functions with decorator-added parameters."""

    # Test with a real OpenHCS function that should have these parameters
    try:
        from openhcs.processing.func_registry import FUNCTION_REGISTRY

        # Find a function that should have dtype_conversion and slice_by_slice
        test_functions = []
        for func_name, func in FUNCTION_REGISTRY.items():
            if hasattr(func, '__signature__'):
                sig = func.__signature__
                if 'dtype_conversion' in sig.parameters and 'slice_by_slice' in sig.parameters:
                    test_functions.append((func_name, func))
                    print(f"Found real OpenHCS function with parameters: {func_name}")
                    break

        if not test_functions:
            print("No OpenHCS functions found with dtype_conversion/slice_by_slice, using mock")
            raise ImportError("No suitable functions found")

    except (ImportError, AttributeError):
        print("Testing with a mock decorated function")
        # Create a simple test function with the decorator pattern
        from openhcs.core.memory.decorators import DtypeConversion

        def mock_func(image_3d, sigma=1.0):
            """Test function.

            Parameters
            ----------
            image_3d : array
                Input image
            sigma : float
                Blur sigma
            slice_by_slice : bool, optional (default: False)
                If True, process 3D arrays slice by slice to avoid cross-slice contamination.
            dtype_conversion : DtypeConversion, optional (default: PRESERVE_INPUT)
                Controls output data type conversion.
            """
            return image_3d

        # Manually add the parameters to simulate decorator behavior
        import inspect
        original_sig = inspect.signature(mock_func)
        new_params = list(original_sig.parameters.values())

        dtype_param = inspect.Parameter(
            'dtype_conversion',
            inspect.Parameter.KEYWORD_ONLY,
            default=DtypeConversion.PRESERVE_INPUT,
            annotation=DtypeConversion
        )
        slice_param = inspect.Parameter(
            'slice_by_slice',
            inspect.Parameter.KEYWORD_ONLY,
            default=False,
            annotation=bool
        )
        new_params.extend([dtype_param, slice_param])

        new_sig = original_sig.replace(parameters=new_params)
        mock_func.__signature__ = new_sig
        mock_func.__annotations__ = {'dtype_conversion': DtypeConversion, 'slice_by_slice': bool}

        test_functions = [("mock_func", mock_func)]
    
    for func_name, func in test_functions:
        print(f"\n{'='*60}")
        print(f"Testing function: {func_name}")
        print(f"Module: {getattr(func, '__module__', 'unknown')}")
        
        # Test signature inspection
        import inspect
        try:
            sig = inspect.signature(func)
            print(f"Function signature: {sig}")
            print(f"Parameters in signature: {list(sig.parameters.keys())}")
        except Exception as e:
            print(f"Error getting signature: {e}")
        
        # Test docstring extraction
        try:
            print(f"Function docstring:")
            print(f"  {func.__doc__}")
            print()

            docstring_info = DocstringExtractor.extract(func)
            print(f"Docstring parameters: {list(docstring_info.parameters.keys())}")
            for param_name, param_desc in docstring_info.parameters.items():
                if param_name in ['dtype_conversion', 'slice_by_slice']:
                    print(f"  {param_name}: {param_desc[:100]}...")
        except Exception as e:
            print(f"Error extracting docstring: {e}")
        
        # Test SignatureAnalyzer
        try:
            param_info = SignatureAnalyzer.analyze(func)
            print(f"SignatureAnalyzer parameters: {list(param_info.keys())}")
            
            for param_name in ['dtype_conversion', 'slice_by_slice']:
                if param_name in param_info:
                    info = param_info[param_name]
                    print(f"  {param_name}:")
                    print(f"    type: {info.param_type}")
                    print(f"    default: {info.default_value}")
                    print(f"    description: {info.description}")
                else:
                    print(f"  {param_name}: NOT FOUND")
        except Exception as e:
            print(f"Error with SignatureAnalyzer: {e}")

if __name__ == "__main__":
    test_decorator_parameters()
