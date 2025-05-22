# Plan 03: GPU-Accelerated Pattern Matching

## Objective

Implement GPU-accelerated pattern matching based on the parallel algorithms described in the Voetter paper, ensuring compatibility with SMA's existing pattern matching system while keeping all data in GPU memory.

## Background

The research paper demonstrates efficient parallel algorithms for pattern matching tasks, including:
- Parallel bracket matching
- "Previous smaller or equal value" algorithm for pattern matching
- Parallel boolean expression evaluation

SMA has a pattern matching system with different pattern types:
- String patterns
- Regex patterns
- AST patterns
- Semantic patterns

## Current State

SMA's pattern matching system is defined in `semantic_matrix_analyzer/patterns/__init__.py`:
- `PatternMatcher` abstract base class
- Concrete implementations for different pattern types
- Pattern matching is primarily sequential and CPU-based

## Implementation Plan

### 1. Create GPU Pattern Matcher Base Class

Create a base class for GPU-accelerated pattern matchers that keeps all data in GPU memory:

```python
class GPUPatternMatcher(PatternMatcher):
    """Base class for GPU-accelerated pattern matchers."""
    
    def __init__(self, device="cuda", memory_manager=None):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.memory_manager = memory_manager or GPUMemoryManager(device=self.device)
        
    def match_pattern(self, pattern, file_path, file_content, ast_node):
        """Match a pattern against a file using GPU acceleration."""
        # Generate a unique key for this pattern matching operation
        key = f"pattern_{hash(pattern)}_{hash(file_path)}"
        
        # Check if result is already in GPU memory
        cached_result = self.memory_manager.get(key)
        if cached_result is not None:
            return cached_result
        
        # Convert inputs to GPU tensors
        gpu_inputs = self._prepare_inputs(pattern, file_content, ast_node)
        
        # Dispatch to appropriate matcher based on pattern type
        if pattern.pattern_type == PatternType.STRING:
            matches = self._match_string_pattern(pattern, file_path, gpu_inputs)
        elif pattern.pattern_type == PatternType.REGEX:
            matches = self._match_regex_pattern(pattern, file_path, gpu_inputs)
        elif pattern.pattern_type == PatternType.AST:
            matches = self._match_ast_pattern(pattern, file_path, gpu_inputs)
        elif pattern.pattern_type == PatternType.SEMANTIC:
            matches = self._match_semantic_pattern(pattern, file_path, gpu_inputs)
        else:
            matches = []
        
        # Cache result in GPU memory
        self.memory_manager.allocate(key, matches)
        
        return matches
    
    def _prepare_inputs(self, pattern, file_content, ast_node):
        """Prepare inputs for GPU processing."""
        # Convert file content to tensor
        if isinstance(file_content, str):
            content_tensor = torch.tensor([ord(c) for c in file_content], 
                                         dtype=torch.int32, 
                                         device=self.device)
        else:
            content_tensor = file_content
            
        # Tensorize AST if not already tensorized
        if ast_node is not None and not isinstance(ast_node, dict):
            from gpu_analysis.ast_tensor import ASTTensorizer
            tensorizer = ASTTensorizer(device=self.device)
            ast_tensors = tensorizer.tensorize(ast_node)
        else:
            ast_tensors = ast_node
            
        return {
            "content_tensor": content_tensor,
            "ast_tensors": ast_tensors
        }
```

### 2. Implement String Pattern Matcher

Implement GPU-accelerated string pattern matching using parallel algorithms:

```python
class GPUStringPatternMatcher(GPUPatternMatcher):
    """GPU-accelerated matcher for string patterns."""
    
    def _match_string_pattern(self, pattern, file_path, gpu_inputs):
        """Match a string pattern using GPU acceleration."""
        content_tensor = gpu_inputs["content_tensor"]
        string_pattern = pattern.pattern
        
        # Convert pattern to tensor
        pattern_tensor = torch.tensor([ord(c) for c in string_pattern], 
                                     dtype=torch.int32, 
                                     device=self.device)
        
        # Find all occurrences using parallel algorithm
        matches = self._find_string_matches(content_tensor, pattern_tensor)
        
        # Convert matches to PatternMatch objects
        return self._convert_to_pattern_matches(pattern, file_path, matches)
    
    def _find_string_matches(self, content_tensor, pattern_tensor):
        """Find all occurrences of pattern in content using parallel algorithm."""
        content_len = content_tensor.size(0)
        pattern_len = pattern_tensor.size(0)
        
        if pattern_len > content_len:
            return torch.tensor([], dtype=torch.int64, device=self.device)
        
        # Create a tensor of all possible starting positions
        positions = torch.arange(content_len - pattern_len + 1, device=self.device)
        
        # For each position, check if the pattern matches
        matches = []
        for pos in positions:
            # Extract substring
            substring = content_tensor[pos:pos+pattern_len]
            
            # Check if substring matches pattern
            if torch.all(substring == pattern_tensor):
                matches.append(pos.item())
        
        return torch.tensor(matches, dtype=torch.int64, device=self.device)
    
    def _convert_to_pattern_matches(self, pattern, file_path, matches):
        """Convert tensor matches to PatternMatch objects."""
        # Implementation details...
```

### 3. Implement Parallel String Matching Algorithm

Implement a more efficient parallel string matching algorithm based on the paper:

```python
def parallel_string_match(content_tensor, pattern_tensor):
    """
    Find all occurrences of pattern in content using a parallel algorithm.
    
    This implementation is based on the parallel string matching algorithm
    described in the Voetter paper.
    """
    content_len = content_tensor.size(0)
    pattern_len = pattern_tensor.size(0)
    
    if pattern_len > content_len:
        return torch.tensor([], dtype=torch.int64, device=content_tensor.device)
    
    # Step 1: Compute a boolean mask for each character in the pattern
    masks = []
    for i in range(pattern_len):
        mask = (content_tensor == pattern_tensor[i])
        masks.append(mask)
    
    # Step 2: Compute a sliding window product of the masks
    result = torch.ones(content_len - pattern_len + 1, dtype=torch.bool, 
                       device=content_tensor.device)
    
    for i in range(pattern_len):
        result = result & masks[i][i:i+content_len-pattern_len+1]
    
    # Step 3: Find the indices where the result is True
    matches = torch.nonzero(result).squeeze(-1)
    
    return matches
```

### 4. Implement AST Pattern Matcher

Implement GPU-accelerated AST pattern matching:

```python
class GPUASTPatternMatcher(GPUPatternMatcher):
    """GPU-accelerated matcher for AST patterns."""
    
    def _match_ast_pattern(self, pattern, file_path, gpu_inputs):
        """Match an AST pattern using GPU acceleration."""
        ast_tensors = gpu_inputs["ast_tensors"]
        node_type, condition = pattern.pattern
        
        # Find nodes of the specified type
        matching_nodes = self._find_nodes_by_type(ast_tensors, node_type)
        
        # Apply condition to matching nodes
        if condition:
            matching_nodes = self._apply_condition(ast_tensors, matching_nodes, condition)
        
        # Convert matches to PatternMatch objects
        return self._convert_to_pattern_matches(pattern, file_path, ast_tensors, matching_nodes)
    
    def _find_nodes_by_type(self, ast_tensors, node_type):
        """Find nodes of the specified type in parallel."""
        nodes = ast_tensors["nodes"]
        node_types = ast_tensors["node_types"]
        
        # Create a mapping from node type string to integer
        if not hasattr(self, "_node_type_map"):
            self._node_type_map = {}
        
        if node_type not in self._node_type_map:
            self._node_type_map[node_type] = len(self._node_type_map)
        
        type_id = self._node_type_map[node_type]
        
        # Find nodes with matching type
        matches = torch.nonzero(node_types == type_id).squeeze(-1)
        
        return matches
    
    def _apply_condition(self, ast_tensors, nodes, condition):
        """Apply condition to nodes in parallel."""
        # Implementation using parallel boolean expression evaluation
        # This is a simplified version; the actual implementation would be more complex
        
        # For each condition key-value pair, filter nodes
        for key, value in condition.items():
            # Get the attribute values for all nodes
            if key not in ast_tensors:
                continue
            
            attr_values = ast_tensors[key]
            
            # Convert value to tensor if it's not already
            if not isinstance(value, torch.Tensor):
                value_tensor = torch.tensor(value, device=self.device)
            else:
                value_tensor = value
            
            # Filter nodes based on attribute value
            valid_nodes = []
            for node_idx in nodes:
                if attr_values[node_idx] == value_tensor:
                    valid_nodes.append(node_idx)
            
            nodes = torch.tensor(valid_nodes, device=self.device)
            
            if nodes.size(0) == 0:
                break
        
        return nodes
```

### 5. Implement Parallel Boolean Expression Evaluation

Implement parallel boolean expression evaluation for pattern conditions:

```python
class ParallelBooleanEvaluator:
    """Evaluates boolean expressions in parallel on GPU."""
    
    def __init__(self, device="cuda"):
        self.device = device
        
    def evaluate(self, expression, variables):
        """Evaluate a boolean expression in parallel."""
        # Convert expression to postfix notation
        postfix = self._to_postfix(expression)
        
        # Evaluate postfix expression in parallel
        return self._evaluate_postfix(postfix, variables)
    
    def _to_postfix(self, expression):
        """Convert infix expression to postfix notation."""
        # Implementation details...
        
    def _evaluate_postfix(self, postfix, variables):
        """Evaluate postfix expression in parallel."""
        # Implementation using parallel reduction
```

## Integration with SMA

### 1. Create GPU Pattern Matcher Registry

Create a registry for GPU-accelerated pattern matchers:

```python
class GPUPatternMatcherRegistry:
    """Registry for GPU-accelerated pattern matchers."""
    
    def __init__(self, device="cuda"):
        self.device = device
        self._matchers = {}
        self.memory_manager = GPUMemoryManager(device=device)
        
    def register_matcher(self, pattern_type, matcher):
        """Register a matcher for a pattern type."""
        self._matchers[pattern_type] = matcher
        
    def get_matcher(self, pattern_type):
        """Get the matcher for a pattern type."""
        matcher = self._matchers.get(pattern_type)
        if matcher is None:
            return None
        
        # Initialize matcher with memory manager
        return matcher(device=self.device, memory_manager=self.memory_manager)
    
    def match_pattern(self, pattern, file_path, file_content, ast_node):
        """Match a pattern using the appropriate matcher."""
        matcher = self.get_matcher(pattern.pattern_type)
        if matcher is None:
            return []
        
        return matcher.match_pattern(pattern, file_path, file_content, ast_node)
```

### 2. Extend SMA's Pattern Matching System

Extend SMA's pattern matching system to support GPU acceleration:

```python
def extend_pattern_matching_system():
    """Extend SMA's pattern matching system to support GPU acceleration."""
    # Create GPU pattern matcher registry
    gpu_registry = GPUPatternMatcherRegistry()
    
    # Register GPU pattern matchers
    gpu_registry.register_matcher(PatternType.STRING, GPUStringPatternMatcher)
    gpu_registry.register_matcher(PatternType.REGEX, GPURegexPatternMatcher)
    gpu_registry.register_matcher(PatternType.AST, GPUASTPatternMatcher)
    gpu_registry.register_matcher(PatternType.SEMANTIC, GPUSemanticPatternMatcher)
    
    # Add GPU registry to SMA's pattern matching system
    # Implementation details...
```

## Performance Considerations

1. **Keep Everything in GPU Memory**: Ensure all data remains in GPU memory throughout the pattern matching process:
   - Convert inputs to GPU tensors once and reuse them
   - Keep intermediate results in GPU memory
   - Only transfer final results back to CPU when necessary

2. **Batch Processing**: Process multiple patterns simultaneously for better GPU utilization:
   - Group patterns by type
   - Apply patterns in batches
   - Combine results at the end

3. **Memory Management**: Use the GPU memory manager to handle memory allocation and deallocation:
   - Cache frequently used patterns and ASTs
   - Release memory when no longer needed
   - Handle out-of-memory situations gracefully

4. **Algorithm Optimization**: Implement efficient parallel algorithms for pattern matching:
   - Use parallel scan for string matching
   - Use parallel reduction for boolean expression evaluation
   - Use parallel tree operations for AST pattern matching

## Testing Strategy

1. **Correctness Tests**: Verify that GPU-accelerated pattern matching produces the same results as the original implementation.

2. **Performance Tests**: Measure the performance improvement for different pattern types and input sizes.

3. **Memory Tests**: Monitor GPU memory usage to ensure it stays within reasonable bounds.

## Success Criteria

1. GPU-accelerated pattern matching is correctly implemented and produces the same results as the original implementation.

2. Pattern matching is faster on GPU than on CPU for large inputs.

3. All data remains in GPU memory throughout the pattern matching process, minimizing CPU-GPU transfers.

4. The implementation integrates seamlessly with SMA's existing pattern matching system.

## References

1. Voetter, R. F. (2020-2021). "Parallel Lexing, Parsing and Semantic Analysis on the GPU."

2. SMA codebase: `semantic_matrix_analyzer/patterns/__init__.py`

3. SMA codebase: `semantic_matrix_analyzer/patterns/ast_matcher.py`
