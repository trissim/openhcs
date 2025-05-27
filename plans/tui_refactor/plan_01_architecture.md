# plan_01_architecture.md
## Component: TUI Architecture Foundation

### Objective
Establish the core architectural foundation for the new TUI that perfectly reflects the static API through 1:1 abstraction. Replace TUI2's schema-dependent approach with pure static analysis.

### Plan
1. Create main application controller that orchestrates the 5-bar layout
2. Implement static analysis engine for form generation
3. Build generic ListManagerComponent base class (shared by plate manager and pipeline editor)
4. Create component configuration system for ListManager instances
5. Establish command dispatch system
6. Create layout manager for the hierarchical structure

### Findings
From DNA analysis:
- TUI1: 38 files, 12,505 lines, monolithic (menu_bar.py: 1,235 lines)
- TUI2: 12 files, 2,993 lines, cleaner but schema-dependent
- Current error density: TUI2 has 2x higher error density than TUI1
- Schema dependencies in dual_editor_controller.py and step_settings_editor.py
- Function registry provides perfect static discovery mechanism
- AbstractStep.__init__ has 6 optional parameters - ideal for form generation
- GlobalPipelineConfig is frozen dataclass - perfect for settings editor

Key architectural insight:
- PlateManagerPane and PipelineEditorPane follow identical structure
- Only differences: title text, button configs, item rendering, action handlers
- Both use Frame(HSplit([toolbar, separator, list])) pattern
- This suggests a single generic ListManagerComponent with configuration objects
- Eliminates code duplication while maintaining type safety and clear separation

### Implementation Draft
(Only after smell loop passes)
