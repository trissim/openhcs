# plan_06_step_func_editor.md
## Component: Dual Step/Func Editor

### Objective
Build the dual mode editor that replaces the plate manager container. Uses toggle buttons to show ONE form at a time (Step OR Func, never both). Each mode uses pure static analysis for form generation.

### Plan
1. Create dual editor container with mode toggle system
2. Build step settings form using AbstractStep static analysis
3. Build complex function pattern editor with dict key management
4. Implement function list management with add/delete/reorder capabilities
5. Create per-function parameter forms using signature analysis
6. Add file operations ([load]/[save_as]) for .step and .func pickled objects
7. Integrate external editor ([edit_in_vim]) functionality
8. Implement DynamicContainer for mode switching (one view at a time)
9. Add save/close functionality with change detection

### Findings
**Crystal Clear Architecture:**

**Toggle Mode System (One View at a Time):**
```
Step Mode Active:                    Func Mode Active:
┌─────────────────────────┐         ┌─────────────────────────┐
│ ┌─────────┐┌────────┐   │         │ ┌────────┐┌─────────┐   │ ← Squared toggle buttons
│ │X Step X││  Func  │   │         │ │  Step  ││X Func X│   │
│ └─────────┘└────────┘   │         │ └────────┘└─────────┘   │
├─────────────────────────┤         ├─────────────────────────┤
│ ┌──────┐┌───────┐       │         │ ┌──────┐┌───────┐       │ ← Squared action buttons
│ │ save ││ close │       │         │ │ save ││ close │       │
│ └──────┘└───────┘       │         │ └──────┘└───────┘       │
├─────────────────────────┤         ├─────────────────────────┤
│ Step Settings Editor    │         │ Func Pattern Editor     │ ← Different content
│ ┌──────┐┌─────────┐     │         │ ┌─────┐┌──────┐┌────────┐│
│ │ load ││ save_as │     │         │ │ add ││ load ││save_as ││
│ └──────┘└─────────┘     │         │ └─────┘└──────┘└────────┘│
└─────────────────────────┘         └─────────────────────────┘
```

**Container Architecture:**
```python
step_func_editor = HSplit([
    # Toggle + action buttons
    HSplit([
        VSplit([
            toggle_button_step,    # [X Step X] or [  Step  ]
            toggle_button_func,    # [X Func X] or [  Func  ]
            save_button,           # [save] (disabled until changes)
            close_button           # [close]
        ])
    ]),
    # Dynamic content (switches based on mode)
    DynamicContainer(get_current_form)
])

def get_current_form():
    if mode == "step":
        return step_settings_form
    elif mode == "func":
        return func_pattern_form
    else:
        return step_settings_form  # default
```

**Static Analysis Form Generation:**

**Step Mode** (AbstractStep.__init__ analysis):
```
|_X_Step_X_|___Func___|_[save]__[close]_________| ← Toggle + action buttons
|^|_Step_settings_editor__[load]_[save_as]______| ← Step editor with file ops + scroll indicator
|X| [reset] Name: [...]                         | ← Each field has reset button
|X| [reset] input_dir: [...]                    |
|X| [reset] output_dir: [...]                   |
|X| [reset] force_disk_output: [ ]              | ← Checkbox
|X| [reset] variable_components: |site|V|       | ← Dropdown from enum
|X| [reset] group_by: |channel|V|               | ← Dropdown from enum
```

**Step Mode Form Fields:**
- name: Optional[str] → TextArea with [reset] button
- input_dir: Optional[Union[str,Path]] → TextArea + file dialog
- output_dir: Optional[Union[str,Path]] → TextArea + file dialog
- force_disk_output: Optional[bool] → Checkbox
- variable_components: Optional[List[str]] → Dropdown (VariableComponents enum)
- group_by: Optional[str] → Dropdown (GroupBy enum)

**Func Mode** (Complex Function Pattern Editor):
```
┌───Step───┬─X─Func─X─┬─save──┬─close─────────┐ ← Toggle + action buttons (shared walls)
├──────────┴──────────┴───────┴───────────────┤
│ Func Pattern Editor  ┌─────┐┌──────┐┌────────┐│ ← Label + action buttons (shared walls)
│                      │ add ││ load ││save_as ││
├──────────────────────┴─────┴──────┴────────┤
│ dict_keys: ┌──────┐┌─┐┌─┐  ┌─────────────┐ │ ← Dict key management bar
│            │ None ▼││+││-│  │ edit_in_vim │ │ ← Dropdown + add/remove + vim editor
├────────────┴──────┴─┴─┴─┴──┴─────────────┴─┤
│ Func 1: ┌─────────────────────────────────┐ │ ← Function dropdown from registry
│         │ percentile stack normalize    ▼ │ │
├─────────┴─────────────────────────────────┴─┤
│  move  ┌───────┐ percentile_low:  0.1 ...   │ ← Auto-generated parameter forms
│   /\   │ reset │ percentile_high: 99.9 ...  │ ← Move up/down arrows + reset
│   \/   └───────┘                            │
│        ┌─────┐                              │ ← Add/delete function
│        │ add │                              │
│        └─────┘                              │
│        ┌────────┐                           │
│        │ delete │                           │
├────────┴────────┴───────────────────────────┤
│ Func 2: ┌─────────────┐                     │ ← Multiple functions in list
│         │ n2v2      ▼ │                     │
├─────────┴─────────────┴─────────────────────┤
│         ┌───────┐ random_seed: 42 ...       │ ← Each function has own params
│   move  │ reset │ device: cuda ...          │
│    /\   └───────┘ blindspot_prob: 0.05 ...  │
│    \/             max_epochs: 10 ...        │
│                   batch_size: 4 ...         │
│                   learning_rate: 1e-4 ...   │
│                   save_model_path: ...      │
│         ┌───────────┐                       │ ← Reset all params for this func
│         │ reset all │                       │
│         └───────────┘                       │
│         ┌─────┐                             │ ← Add/delete function
│         │ add │                             │
│         └─────┘                             │
│         ┌────────┐                          │
│         │ delete │                          │
├─────────┴────────┴──────────────────────────┤
│ Func 3: ┌─────────────┐                     │ ← Additional functions
│         │ 3d_deconv ▼ │                     │
│         └─────────────┘                     │
│         ┌───────┐ random_seed:  42 ...      │
│    move │ reset │ device: cuda  ...         │
│     /\  └───────┘ blindspot_prob: 0.05 ...  │
│     \/  vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv │ ← Scrollbar
└─────────────────────────────────────────────┘
```

**Complex Architecture Requirements:**

**Three-Bar Structure:**
1. **Toggle + Action Bar**: Step/Func toggles + save/close buttons (shared walls)
2. **Pattern Editor Bar**: Label + add/load/save_as buttons (shared walls)
3. **Dict Key Management Bar**: Label + dropdown + +/- buttons + edit_in_vim (shared walls)
4. **Function List Area**: Scrollable list of function panes

**Dict Key Management Logic:**
- **Default**: Dropdown starts with "None" selected
- **Add Key**: [+] button opens text dialog → adds key to dropdown → selects new key
- **Remove Key**: [-] button removes selected key from dropdown
- **Key Selection**: When key selected → loads associated value → introspects functions/fields
- **None Selection**: When "None" selected → shows default function list

**Function Addition Logic:**
- **Global Add**: [add] button in pattern editor bar adds function when none exist
- **Per-Function Add**: [add] button in each function pane adds function after that one
- **No Gap Problem**: Global add solves the "can't add first function" issue

**Shared Wall Architecture:**
- **No Double Walls**: Buttons share walls with surrounding containers
- **Continuous Borders**: All separators connect seamlessly
- **No Gaps**: Buttons integrate directly into container walls
- **Box Drawing**: Use proper ┌┬┐├┼┤└┴┘ characters for connections

**Dynamic Content Loading:**
- **Key-Based Introspection**: Selected dict key determines function list content
- **Signature Analysis**: Each function dropdown populated from registry
- **Parameter Forms**: Auto-generated from function signatures
- **Reset Functionality**: Individual [reset] + [reset all] per function

**Static Analysis Requirements:**

**Step Mode Form Generation:**
```python
# Analyze AbstractStep.__init__ signature
sig = inspect.signature(AbstractStep.__init__)
for param_name, param in sig.parameters.items():
    if param_name == 'self':
        continue
    # Generate form field based on type annotation
    # name: Optional[str] → TextArea with [reset] button
    # force_disk_output: Optional[bool] → Checkbox with [reset]
    # variable_components: Optional[List[str]] → Dropdown from VariableComponents enum
    # etc.
```

**Func Mode Form Generation:**
```python
# 1. Dict key management
dict_keys_dropdown = generate_dropdown_from_registry_keys()
# 2. Function discovery from registry
available_functions = FUNC_REGISTRY  # {backend: [functions]}
# 3. Per-function parameter analysis
for func in selected_functions:
    sig = inspect.signature(func)
    param_forms = generate_parameter_forms(sig)
    # Each param gets [reset] button to restore signature default
```

**Function Pattern Object Construction:**
- **Callable**: Single function
- **(Callable, Dict)**: Function with kwargs override
- **List[Callable]**: Multiple functions in sequence
- **Dict[str, List[Callable]]**: Multiple functions grouped by keys

**File Operations:**
- **.step files**: Pickled AbstractStep instances with all configured parameters
- **.func files**: Pickled func pattern objects (callable/tuple/list/dict structures)
- **[load]**: File dialog → unpickle → populate forms
- **[save_as]**: Collect form data → construct object → pickle → file dialog

**External Editor Integration:**
- **[edit_in_vim]**: Export current func pattern to temp file → launch vim → import changes

### Implementation Draft
(Only after smell loop passes)
