# plan_05_list_manager_instances.md
## Component: List Manager Component Instances

### Objective
Create the two specific instances of the generic ListManagerComponent: one configured for plate management and one for pipeline editing. This demonstrates the reusable base component pattern.

### Plan
1. Create PlateManagerConfig with plate-specific buttons, renderers, and actions
2. Create PipelineEditorConfig with step-specific buttons, renderers, and actions
3. Implement plate item renderer with status symbols (o, !, ?, ✓)
4. Implement step item renderer with reordering arrows (^/v)
5. Configure action dispatchers for plate operations vs pipeline operations
6. Wire both instances to the same generic ListManagerComponent

### Findings
Configuration differences between the two instances:

**Plate Manager Configuration:**
- Title: "Plate Manager"
- Buttons: ┌─────┬─────┬──────┬──────┬─────────┬─────┐ (6 buttons, shared walls)
           │ add │ del │ edit │ init │ compile │ run │
           └─────┴─────┴──────┴──────┴─────────┴─────┘
- Item renderer: status + arrows + name + path (o/!/✓/? + ^/v + name + | path)
- Actions: file dialogs, orchestrator operations, status updates
- Status indicators: o (not init), ! (error), ? (unknown), ✓ (success)

**Plate Item Format (from screenshot):**
```
|o| ^/v 1: axotomy_n1_03-02-24 | (...)/nvme_usb/
|o| ^/v 2: axotomy_n2_03-03-24 | (...)/nvme_usb/
|?| ^/v 3: axotomy_n3_03-04-24 | (...)/nvme_usb/
```
- Shows BOTH status symbol AND arrows
- Includes full path after pipe separator
- Format: `|status| ^/v number: name | path`

**Pipeline Editor Configuration:**
- Title: "Pipeline Editor"
- Buttons: ┌─────┬─────┬──────┬──────┬──────┐ (5 buttons, shared walls)
           │ add │ del │ edit │ load │ save │
           └─────┴─────┴──────┴──────┴──────┘
- Item renderer: status + arrows + name (o + ^/v + step name)
- Actions: step creation, .pipeline file I/O, step/func editor trigger
- Status indicators: o (status) + ^/v (reorder arrows)

**Pipeline Item Format (from screenshot):**
```
|o| ^/v 1: pos_gen_pattern
|o| ^/v 2: enhance + assemble
|o| ^/v 3: trace_analysis
```
- Shows BOTH status symbol AND arrows (like plates)
- Format: `|status| ^/v number: step_name`
- Steps also have status indicators, not just reorder arrows

**File Operations Detail:**
- **[load]**: File dialog → select .pipeline file → unpickle step list → populate pipeline editor
- **[save]**: Collect current step list → pickle → save to .pipeline file
- **.pipeline files**: Pickled lists of step instances (AbstractStep/FunctionStep objects)

**Shared Functionality (in base component):**
- Selection and multi-selection logic
- Keyboard navigation (up/down/enter/space)
- Button toolbar layout and focus management
- Frame container with title and separator
- Async action dispatch system
- Dynamic container updates

**Instance Creation:**
```python
# Same component, different configurations
plate_manager = ListManagerComponent(PlateManagerConfig(
    title="Plate Manager",
    buttons=[add_plate_btn, del_plate_btn, edit_plate_btn, init_btn, compile_btn, run_btn],
    item_renderer=PlateItemRenderer(),  # Renders: "|o| ^/v 1: name | path"
    action_dispatcher=PlateActionDispatcher()
))

pipeline_editor = ListManagerComponent(PipelineEditorConfig(
    title="Pipeline Editor",
    buttons=[add_step_btn, del_step_btn, edit_step_btn, load_btn, save_btn],
    item_renderer=StepItemRenderer(),   # Renders: "|o| ^/v 1: step_name"
    action_dispatcher=StepActionDispatcher()
))
```

**Key Architectural Insight:**
- ONE component implementation
- TWO different configurations
- ZERO code duplication
- Consistent behavior patterns
- Icons always on LEFT side
- Scrollbars automatically appear when needed

### Implementation Draft
(Only after smell loop passes)
