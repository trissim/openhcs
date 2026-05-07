from typing import Any, Callable, Iterator, Optional, Tuple


def iter_pattern_items(pattern: Any) -> Iterator[Tuple[Any, str, int]]:
    """Iterate all items in a pattern regardless of structure (dict/list/single).
    
    Yields (func_item, key, position) tuples where:
    - func_item: raw function item (callable, tuple, or nested pattern)
    - key: dict key or "default" for list/single patterns
    - position: index within the list at that key
    """
    if isinstance(pattern, dict):
        for key, value in pattern.items():
            items = value if isinstance(value, list) else [value]
            for pos, func in enumerate(items):
                yield (func, key, pos)
    elif isinstance(pattern, list):
        for pos, func in enumerate(pattern):
            yield (func, "default", pos)
    else:
        yield (pattern, "default", 0)


def get_core_callable(func_pattern: Any) -> Any:  # Returns Callable or FunctionReference
    """Extract the first effective Python callable from a func_pattern.

    Handles: direct callable, (callable, kwargs) tuple, list (chain), dict pattern.

    NOTE: FunctionReference objects are returned as-is (not resolved) because they
    expose all needed attributes via __getattr__. Resolution happens in worker process.

    Returns either a Callable or FunctionReference (which acts like a callable via __getattr__).
    """
    # Check for FunctionReference first - return as-is, don't resolve!
    try:
        from openhcs.core.pipeline.compiler import FunctionReference
        if isinstance(func_pattern, FunctionReference):
            return func_pattern  # Return FunctionReference, don't resolve
    except ImportError:
        pass

    if callable(func_pattern) and not isinstance(func_pattern, type):
        return func_pattern
    elif isinstance(func_pattern, tuple) and func_pattern:
        first_element = func_pattern[0]
        try:
            from openhcs.core.pipeline.compiler import FunctionReference
            if isinstance(first_element, FunctionReference):
                return first_element  # Return FunctionReference, don't resolve
        except ImportError:
            pass
        if callable(first_element) and not isinstance(first_element, type):
            return first_element
    elif isinstance(func_pattern, list) and func_pattern:
        return get_core_callable(func_pattern[0])
    elif isinstance(func_pattern, dict) and func_pattern:
        for key, value in func_pattern.items():
            if core_callable := get_core_callable(value):
                return core_callable
    return None


def _resolve_function_references(func_value):
    """
    Recursively resolve FunctionReference objects to actual functions.

    This handles all function pattern formats and resolves any FunctionReference
    objects back to their actual decorated functions from the registry.
    """
    # Import here to avoid circular imports
    try:
        from openhcs.core.pipeline.compiler import FunctionReference
    except ImportError:
        # If FunctionReference doesn't exist, just return the original value
        return func_value

    if isinstance(func_value, FunctionReference):
        # Resolve FunctionReference to actual function
        return func_value.resolve()
    elif isinstance(func_value, tuple) and len(func_value) in {2, 3}:
        # Tuple: (function_or_ref, kwargs[, invocation_options])
        func_or_ref, kwargs, *rest = func_value
        resolved_func = _resolve_function_references(func_or_ref)
        return (resolved_func, kwargs, *rest)
    elif isinstance(func_value, list):
        # List of functions/tuples → List of resolved functions/tuples
        return [_resolve_function_references(item) for item in func_value]
    elif isinstance(func_value, dict):
        # Dict of functions/tuples → Dict of resolved functions/tuples
        return {key: _resolve_function_references(value) for key, value in func_value.items()}
    else:
        # Not a function pattern or already a callable, return as-is
        return func_value


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
    logger.debug("🔍 PATTERN DEBUG: prepare_patterns_and_functions called")
    logger.debug(f"🔍 PATTERN DEBUG: patterns type: {type(patterns)}")
    logger.debug(f"🔍 PATTERN DEBUG: patterns keys/content: {list(patterns.keys()) if isinstance(patterns, dict) else f'List with {len(patterns)} items'}")
    logger.debug(f"🔍 PATTERN DEBUG: processing_funcs type: {type(processing_funcs)}")
    logger.debug(f"🔍 PATTERN DEBUG: processing_funcs keys: {list(processing_funcs.keys()) if isinstance(processing_funcs, dict) else 'Not a dict'}")
    logger.debug(f"🔍 PATTERN DEBUG: component: {component}")

    # DO NOT resolve FunctionReference objects here!
    # They must remain as FunctionReference for picklability.
    # Resolution happens in the worker process during execution.

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
        # Check for FunctionReference
        try:
            from openhcs.core.pipeline.compiler import FunctionReference
            is_func_ref = isinstance(func_item, FunctionReference)
        except ImportError:
            is_func_ref = False

        if isinstance(func_item, tuple) and len(func_item) in {2, 3}:
            first_elem = func_item[0]
            # Check if first element is FunctionReference or callable
            try:
                from openhcs.core.pipeline.compiler import FunctionReference
                is_first_func_ref = isinstance(first_elem, FunctionReference)
            except ImportError:
                is_first_func_ref = False

            if is_first_func_ref or callable(first_elem):
                # It's a (function/FunctionReference, kwargs[, invocation_options]) tuple
                return first_elem, func_item[1]

        if is_func_ref or callable(func_item):
            # It's just a function/FunctionReference, use default args
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
        if not isinstance(func_item, list):
            # Normalize single function to list so execution always uses chain logic
            logger.debug(f"Normalizing single function for '{comp_value}' into list")
            func, args = extract_func_and_args(func_item)
            func_item = [(func, args)]

        # List of functions or function tuples (already normalized)
        logger.debug(f"func_item is a list with {len(func_item)} items")
        component_to_funcs[comp_value] = func_item
        component_to_args[comp_value] = {}

    return grouped_patterns, component_to_funcs, component_to_args
