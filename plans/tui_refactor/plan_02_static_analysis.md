# plan_02_static_analysis.md
## Component: Static Analysis Engine

### Objective
Build the core static analysis engine that generates UI forms directly from Python signatures, eliminating schema dependencies and ensuring 1:1 API reflection.

### Plan
1. Create signature analyzer for AbstractStep and FunctionStep constructors
2. Build enum analyzer for VariableComponents and GroupBy
3. Implement function registry analyzer for dropdown population
4. Create type-to-widget mapping system
5. Build form generator that produces prompt_toolkit widgets
6. Add validation system based on actual type annotations

### Findings
Key signatures to analyze:
- AbstractStep.__init__: 6 optional keyword-only parameters
- FunctionStep.__init__: func required, 4 optional parameters
- GlobalPipelineConfig: 4 fields with defaults via dataclass
- VariableComponents enum: 4 values (SITE, CHANNEL, Z_INDEX, WELL)
- GroupBy enum: 4 values mapping to VariableComponents
- FUNC_REGISTRY: Dict[str, List[Callable]] grouped by backend
- get_function_info(): Returns name, backend, memory types, doc, module

Type mapping strategy:
- Optional[str] → TextArea with default
- Optional[bool] → Checkbox with default
- Optional[List[str]] → Multi-select or text area
- Enum types → Dropdown with enum values
- Union[str, Path] → TextArea with path validation

### Implementation Draft
(Only after smell loop passes)
