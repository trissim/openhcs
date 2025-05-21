def prepare_patterns_and_functions(patterns, processing_funcs, component='default'):
    """
    Prepare patterns, processing functions, and processing args for processing.

    This function handles three main tasks:
    1. Ensuring patterns are in a component-keyed dictionary format
    2. Determining which processing functions to use for each component
    3. Determining which processing args to use for each component

    Args:
        patterns (list or dict): Patterns to process, either as a flat list or grouped by component
        processing_funcs (callable, list, dict, tuple, optional): Processing functions to apply.
            Can be a single callable, a tuple of (callable, kwargs), a list of either,
            or a dictionary mapping component values to any of these.
        component (str): Component name for grouping (only used for clarity in the result)

    Returns:
        tuple: (grouped_patterns, component_to_funcs, component_to_args)
            - grouped_patterns: Dictionary mapping component values to patterns
            - component_to_funcs: Dictionary mapping component values to processing functions
            - component_to_args: Dictionary mapping component values to processing args
    """
    # Ensure patterns are in a dictionary format
    # If already a dict, use as is; otherwise wrap the list in a dictionary
    grouped_patterns = patterns if isinstance(patterns, dict) else {component: patterns}

    # Initialize dictionaries for functions and args
    component_to_funcs = {}
    component_to_args = {}

    # Helper function to extract function and args from a function item
    def extract_func_and_args(func_item):
        if isinstance(func_item, tuple) and len(func_item) == 2 and callable(func_item[0]):
            # It's a (function, kwargs) tuple
            return func_item[0], func_item[1]
        if callable(func_item):
            # It's just a function, use default args
            return func_item, {}
        # Fail loudly and early if the function item is invalid
        raise ValueError(f"Invalid function item for pattern processing: {func_item}")

    for comp_value in grouped_patterns.keys():
        # Get functions and args for this component
        # No special handling for 'channel' component (Clause 77: Rot Intolerance)
        if isinstance(processing_funcs, dict) and comp_value in processing_funcs:
            # Direct mapping for this component
            func_item = processing_funcs[comp_value]
        else:
            # Use the same function for all components
            func_item = processing_funcs

        # Extract function and args
        if isinstance(func_item, list):
            # List of functions or function tuples
            component_to_funcs[comp_value] = func_item
            # For lists, we'll extract args during processing
            component_to_args[comp_value] = {}
        else:
            # Single function or function tuple
            func, args = extract_func_and_args(func_item)
            component_to_funcs[comp_value] = func
            component_to_args[comp_value] = args

    return grouped_patterns, component_to_funcs, component_to_args


