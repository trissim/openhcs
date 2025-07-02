from typing import List, Dict
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
    import logging
    logger = logging.getLogger(__name__)

    # Debug: Log what we received
    logger.debug(f"🔍 PATTERN DEBUG: prepare_patterns_and_functions called")
    logger.debug(f"🔍 PATTERN DEBUG: patterns type: {type(patterns)}")
    logger.debug(f"🔍 PATTERN DEBUG: patterns keys/content: {list(patterns.keys()) if isinstance(patterns, dict) else f'List with {len(patterns)} items'}")
    logger.debug(f"🔍 PATTERN DEBUG: processing_funcs type: {type(processing_funcs)}")
    logger.debug(f"🔍 PATTERN DEBUG: processing_funcs keys: {list(processing_funcs.keys()) if isinstance(processing_funcs, dict) else 'Not a dict'}")
    logger.debug(f"🔍 PATTERN DEBUG: component: {component}")

    # Ensure patterns are in a dictionary format
    # If already a dict, use as is; otherwise wrap the list in a dictionary
    grouped_patterns = patterns if isinstance(patterns, dict) else {component: patterns}

    logger.debug(f"🔍 PATTERN DEBUG: grouped_patterns keys: {list(grouped_patterns.keys())}")

    # SMART FILTERING: If processing_funcs is a dict, only process components that have function definitions
    if isinstance(processing_funcs, dict) and isinstance(grouped_patterns, dict):
        original_components = set(grouped_patterns.keys())
        function_components = set(processing_funcs.keys())

        # Handle type mismatches (string vs int keys)
        available_function_keys = set()
        for key in function_components:
            available_function_keys.add(key)
            available_function_keys.add(str(key))  # Add string version
            if isinstance(key, str) and key.isdigit():
                available_function_keys.add(int(key))  # Add int version if string is numeric

        # Filter to only components that have function definitions
        filtered_grouped_patterns = {
            comp_value: patterns
            for comp_value, patterns in grouped_patterns.items()
            if comp_value in available_function_keys
        }

        # Log what was filtered
        filtered_out = original_components - set(filtered_grouped_patterns.keys())
        if filtered_out:
            logger.debug(f"🔍 PATTERN DEBUG: Filtered out components without function definitions: {filtered_out}")

        logger.debug(f"🔍 PATTERN DEBUG: Processing components: {list(filtered_grouped_patterns.keys())}")
        grouped_patterns = filtered_grouped_patterns

        # Validate that we have at least one component to process
        if not grouped_patterns:
            available_keys = list(processing_funcs.keys())
            discovered_keys = list(original_components)
            raise ValueError(
                f"No components match between discovered data and function pattern. "
                f"Discovered components: {discovered_keys}. "
                f"Function pattern keys: {available_keys}. "
                f"Function pattern keys must match discovered component values."
            )

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
        if isinstance(func_item, dict):
            # It's a dictionary pattern - this should be handled at a higher level
            # This indicates a logic error where the entire dict was passed instead of individual components
            raise ValueError(
                f"Dictionary pattern passed to extract_func_and_args: {func_item}. "
                f"This indicates a component lookup failure in prepare_patterns_and_functions. "
                f"Dictionary patterns should be resolved to individual function lists before reaching this point."
            )
        # Fail loudly and early if the function item is invalid
        raise ValueError(f"Invalid function item for pattern processing: {func_item}")

    for comp_value in grouped_patterns.keys():
        # Get functions and args for this component
        # No special handling for 'channel' component (Clause 77: Rot Intolerance)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Processing component value: '{comp_value}' (type: {type(comp_value)})")
        logger.debug(f"Function pattern keys: {list(processing_funcs.keys()) if isinstance(processing_funcs, dict) else 'Not a dict'}")

        if isinstance(processing_funcs, dict):
            # Direct lookup with type conversion fallback
            # Compile-time validation guarantees dict keys are valid
            if comp_value in processing_funcs:
                func_item = processing_funcs[comp_value]
                logger.debug(f"Found direct match for '{comp_value}': {type(func_item)}")
            else:
                # Handle type mismatch: pattern detection returns strings, but function pattern might use integers
                logger.debug(f"No direct match for '{comp_value}', trying integer conversion")
                try:
                    comp_value_int = int(comp_value)
                    if comp_value_int in processing_funcs:
                        func_item = processing_funcs[comp_value_int]
                    else:
                        # Try converting keys to int for comparison
                        found = False
                        for key in processing_funcs.keys():
                            try:
                                if int(key) == comp_value_int:
                                    func_item = processing_funcs[key]
                                    found = True
                                    break
                            except (ValueError, TypeError):
                                continue
                        if not found:
                            # This should not happen due to compile-time validation
                            func_item = processing_funcs[comp_value]
                except (ValueError, TypeError):
                    # This should not happen due to compile-time validation
                    func_item = processing_funcs[comp_value]
        else:
            # Use the same function for all components
            func_item = processing_funcs

        # Extract function and args
        logger.debug(f"Processing func_item for '{comp_value}': {type(func_item)}")
        if isinstance(func_item, list):
            # List of functions or function tuples
            logger.debug(f"func_item is a list with {len(func_item)} items")
            component_to_funcs[comp_value] = func_item
            # For lists, we'll extract args during processing
            component_to_args[comp_value] = {}
        else:
            # Single function or function tuple
            logger.debug(f"Calling extract_func_and_args with: {type(func_item)}")
            func, args = extract_func_and_args(func_item)
            component_to_funcs[comp_value] = func
            component_to_args[comp_value] = args

    return grouped_patterns, component_to_funcs, component_to_args


