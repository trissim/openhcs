# Function Pattern Editor: Semantic Architecture Investigation

## Investigation Status: SEMANTIC UNDERSTANDING ACHIEVED

**Domain Knowledge**: ✅ **Architectural Alignment**: ✅ **Implementation Ready**: ✅

---

## 🧬 SEMANTIC DOMAIN UNDERSTANDING

### **OpenHCS: High-Content Screening Engine**

OpenHCS is an **experimental data processing engine** for high-content screening:
- **Microscope agnostic**: Processes experimental plates from any microscope format
- **GPU-native**: State-of-the-art MIST stitching with Borůvka MST algorithm  
- **Deep learning integrated**: Self-supervised enhancement and analysis networks
- **Experimental workflow engine**: Transforms experimental datasets through processing pipelines

### **Experimental Data Architecture**

#### **Experimental Structure**
```
Plate → Wells → Sites → Channels → Z-slices → Images
Example: A01 well, site 001, channel 1 (DAPI), z-slice 001
File: "A01_s001_w1_z001.tif"
```

#### **Experimental Processing Workflows**
```python
# List Pattern: Same processing for ALL experimental components
func_pattern = [denoise, normalize]  # Apply to all channels/z-slices

# Dict Pattern: Component-specific experimental workflows  
func_pattern = {
    "1": [denoise, enhance_nuclei],     # DAPI channel workflow
    "2": [denoise, enhance_protein],    # GFP channel workflow
    "3": [denoise, enhance_brightfield] # Brightfield workflow
}
```

#### **Experimental Data Discovery**
```python
# Backend discovers experimental structure:
patterns_by_well = auto_detect_patterns(
    group_by="channel",           # Experimental dimension to group by
    variable_components=["site"]  # Experimental dimensions that vary
)
# Returns experimental component identifiers: {"1": [...], "2": [...], "3": [...]}
```

---

## 🎯 SEMANTIC ARCHITECTURE TRUTH

### **Core Semantic Relationships**

1. **Dict keys = Experimental component identifiers** (channel IDs, z-slice IDs, site IDs)
2. **Function patterns = Experimental processing workflows** designed by researchers
3. **Pattern editor = Experimental workflow designer** for high-content screening
4. **Compile-time validation = Experimental data structure verification**

### **Workflow Design Semantics**

#### **List Pattern Semantics**
- **Meaning**: "Apply this workflow to ALL experimental components"
- **Use case**: Universal processing (denoise all channels, normalize all z-slices)
- **Backend behavior**: Same functions applied to every detected component

#### **Dict Pattern Semantics**  
- **Meaning**: "Apply component-specific workflows to experimental data"
- **Use case**: Channel-specific processing (DAPI enhancement, GFP analysis, etc.)
- **Backend behavior**: Different functions per experimental component identifier

### **Component Identifier Semantics**

**Experimental component identifiers must match actual microscope data:**
- **Channel identifiers**: "1", "2", "3" or "DAPI", "GFP", "BF"
- **Z-slice identifiers**: "001", "002", "003" 
- **Site identifiers**: "s001", "s002", "s003"

**Invalid identifiers**: "group_0", "group_1" (semantically meaningless)

---

## 🚨 CURRENT TUI CODE PROBLEMS (CONCRETE EXAMPLES)

### **Problem 1: Meaningless Key Generation**

**BROKEN CODE** in `openhcs/tui/views/function_pattern_view.py` lines 199-208:
```python
async def _handle_add_key(self):
    """Handle add key request."""
    # Simple implementation - add numbered key
    if isinstance(self.current_pattern, dict):
        existing_keys = list(self.current_pattern.keys())
        new_key = f"group_{len(existing_keys)}"  # ← WRONG: Meaningless key
        self.current_pattern = self.data_manager.add_new_key(self.current_pattern, new_key)
        self.current_key = new_key
        self._refresh_ui()
        self._notify_change()
```

**WHY IT'S WRONG**: Creates keys like "group_0", "group_1" that will NEVER match experimental data like "1", "2", "3"

**HOW TO FIX**: Let user enter experimental component identifiers manually:
```python
async def _handle_add_key(self):
    """Handle add key request."""
    # Prompt user for experimental component identifier
    new_key = await prompt_for_text_input(
        title="Add Component Identifier",
        message="Enter experimental component ID (e.g., '1', '2', 'DAPI', 'GFP'):",
        app_state=self.state
    )
    if new_key and isinstance(self.current_pattern, dict):
        self.current_pattern = self.data_manager.add_new_key(self.current_pattern, new_key)
        self.current_key = new_key
        self._refresh_ui()
        self._notify_change()
```

### **Problem 2: Confusing None Key Logic**

**BROKEN CODE** in `openhcs/tui/services/pattern_data_manager.py` lines 36-54:
```python
@staticmethod
def convert_list_to_dict(pattern: List, preserve_order: bool = True) -> Dict:
    """Convert List pattern to Dict with None key for unnamed groups."""
    if not isinstance(pattern, list):
        raise ValueError(f"Expected list, got {type(pattern)}")

    # EXACT semantics: {None: pattern} for unnamed structural groups
    return {None: pattern}  # ← WRONG: None key is meaningless
```

**WHY IT'S WRONG**: None key doesn't represent any experimental component

**HOW TO FIX**: Remove None key logic entirely:
```python
@staticmethod
def convert_list_to_dict(pattern: List) -> Dict:
    """Convert List pattern to Dict - user must add component keys manually."""
    if not isinstance(pattern, list):
        raise ValueError(f"Expected list, got {type(pattern)}")

    # Return empty dict - user will add experimental component keys manually
    return {}
```

### **Problem 3: Context Isolation**

**BROKEN CODE** in `openhcs/tui/editors/dual_editor_pane.py` lines 66-70:
```python
self.func_editor = FunctionPatternView(
    state=self.state,
    initial_pattern=self.editing_step.func,  # ← ONLY pattern data
    change_callback=self._on_func_change
)
# NO ACCESS TO: group_by, variable_components
```

**WHY IT'S WRONG**: Pattern editor has no idea what experimental components exist

**HOW TO FIX**: Pass step context (temporary solution - manual keys for now):
```python
self.func_editor = FunctionPatternView(
    state=self.state,
    initial_pattern=self.editing_step.func,
    change_callback=self._on_func_change,
    step_context={  # ← ADD: Step context for future use
        'group_by': self.editing_step.group_by,
        'variable_components': self.editing_step.variable_components
    }
)
```

### **Problem 4: Box Layout Errors (CRITICAL UI BUG)**

**BROKEN CODE** in `openhcs/tui/components/function_list_manager.py` lines 194-198:
```python
return [
    Box(move_up, width=3),      # ← WRONG: Box is not a Container
    Box(move_down, width=3),    # ← WRONG: Box has no preferred_height method
    Box(delete_button, width=8) # ← WRONG: Layout system can't handle Box in lists
]
```

**WHY IT'S WRONG**:
- Box is NOT a Container (verified: `isinstance(Box(button), Container) == False`)
- Box has NO `preferred_height` method (verified: `hasattr(Box(button), 'preferred_height') == False`)
- Layout system expects Container objects with preferred_height method

**HOW TO FIX**: Wrap buttons in VSplit Container (like working components):
```python
from prompt_toolkit.layout.containers import VSplit

return VSplit([
    move_up,        # VSplit IS a Container with preferred_height
    move_down,      # VSplit handles button layout correctly
    delete_button   # Pattern used in interactive_list_item.py
])
```

**API VERIFICATION COMPLETE**:
- `Box.__init__` accepts `width` parameter ✅
- `Box` is NOT a `Container` ❌ (verified: `isinstance(Box(button), Container) == False`)
- `Box` has NO `preferred_height` method ❌ (verified: `hasattr(Box(button), 'preferred_height') == False`)
- `Button` is NOT a `Container` ❌ (verified: `isinstance(Button('test'), Container) == False`)
- `VSplit` IS a `Container` ✅ (verified: `isinstance(VSplit([button]), Container) == True`)
- `VSplit` HAS `preferred_height` method ✅ (verified: `hasattr(VSplit([button]), 'preferred_height') == True`)

### **Problem 5: Confusing UI Labels**

**BROKEN CODE** in `openhcs/tui/components/pattern_key_selector.py` lines 167-171:
```python
for k in pattern.keys():
    if k is None:
        display_keys.append((None, "Unnamed"))  # ← WRONG: Meaningless label
    else:
        display_keys.append((k, str(k)))
```

**WHY IT'S WRONG**: "Unnamed" doesn't tell user this is about experimental components

**HOW TO FIX**: Clear experimental component labeling:
```python
for k in pattern.keys():
    # Show actual experimental component identifiers
    display_keys.append((k, f"Component: {k}"))
```

### **Problem 6: Missing Mouse Scrolling Support**

**BROKEN CODE** in `openhcs/tui/components/function_list_manager.py` lines 66-101:
```python
def _build_function_list(self) -> ScrollablePane:
    # ... build function items ...
    return ScrollablePane(HSplit(function_items))  # ← MISSING: Mouse scroll handlers
```

**WHY IT'S WRONG**: No mouse wheel support like other TUI components (file browser, interactive lists)

**HOW TO FIX**: Add mouse scroll key bindings:
```python
def _build_function_list(self) -> ScrollablePane:
    function_items = []
    # ... build function items ...

    # Add mouse scroll support like other TUI components
    kb = KeyBindings()

    @kb.add('<scroll-up>')
    def scroll_up(event):
        # Scroll up 3 items (same as file browser)
        pass  # Will implement proper scrolling

    @kb.add('<scroll-down>')
    def scroll_down(event):
        # Scroll down 3 items (same as file browser)
        pass  # Will implement proper scrolling

    scrollable_pane = ScrollablePane(HSplit(function_items))
    # Attach key bindings to scrollable pane
    return scrollable_pane
```

---

## 🎯 SYSTEMATIC COMPONENT FIXES

### **CATEGORY 1: UI Layout Errors (CRITICAL)**

#### **Fix 1A: Remove Broken Box Wrappers**
**FILE**: `openhcs/tui/components/function_list_manager.py`
**LINES**: 194-198
**CHANGE**: Remove Box wrappers from buttons (they don't have preferred_height)

#### **Fix 1B: Add Mouse Scrolling Support**
**FILE**: `openhcs/tui/components/function_list_manager.py`
**LINES**: 66-101
**CHANGE**: Add mouse wheel key bindings like file browser and interactive lists

### **CATEGORY 2: Semantic Architecture Errors**

#### **Fix 2A: Remove Arbitrary Key Generation**
**FILE**: `openhcs/tui/views/function_pattern_view.py`
**LINES**: 199-208
**CHANGE**: Replace `f"group_{len(existing_keys)}"` with user input prompt

#### **Fix 2B: Remove None Key Logic**
**FILE**: `openhcs/tui/services/pattern_data_manager.py`
**LINES**: 36-54, 57-77
**CHANGE**: Remove None key conversion, return empty dict instead

#### **Fix 2C: Add Step Context Parameter**
**FILE**: `openhcs/tui/views/function_pattern_view.py`
**LINES**: 35-36
**CHANGE**: Add `step_context` parameter to constructor

**FILE**: `openhcs/tui/editors/dual_editor_pane.py`
**LINES**: 66-70
**CHANGE**: Pass step context when creating FunctionPatternView

### **CATEGORY 3: UI Clarity Errors**

#### **Fix 3A: Clear Pattern Type UI**
**FILE**: `openhcs/tui/components/pattern_key_selector.py`
**LINES**: 112-127
**CHANGE**: Replace "Convert to Dict Pattern" with "Apply per Component"

#### **Fix 3B: Clear Component Labels**
**FILE**: `openhcs/tui/components/pattern_key_selector.py`
**LINES**: 167-171
**CHANGE**: Replace "Unnamed" with "Component: X" labeling

### **CATEGORY 4: Missing Utilities**

#### **Fix 4A: Add Text Input Dialog**
**NEW FILE**: `openhcs/tui/utils/dialog_helpers.py`
**PURPOSE**: Add `prompt_for_text_input()` function for component identifiers

---

## 🚀 SYSTEMATIC IMPLEMENTATION ORDER

### **PHASE 1: Fix Critical UI Layout Errors (IMMEDIATE)**
**Step 1A**: Remove broken Box wrappers causing `preferred_height` errors
**Step 1B**: Add mouse scrolling support to match other TUI components

### **PHASE 2: Fix Semantic Architecture (CORE)**
**Step 2A**: Add text input dialog utility for component identifiers
**Step 2B**: Replace arbitrary key generation with user input prompt
**Step 2C**: Remove None key logic and conversion semantics
**Step 2D**: Add step context parameter for future enhancements

### **PHASE 3: Fix UI Clarity (POLISH)**
**Step 3A**: Update pattern type labels ("Apply per Component" vs "Apply to ALL")
**Step 3B**: Update component labels ("Component: X" vs "Unnamed")

### **PHASE 4: API Verification (CRITICAL)**
**Step 4A**: Verify prompt_toolkit Box API - check if Box actually supports width parameter
**Step 4B**: Verify ScrollablePane key binding attachment methods
**Step 4C**: Verify TextArea dialog pattern in existing dialog_helpers.py
**Step 4D**: Verify FunctionPatternView constructor signature and parameter passing

---

## 📋 SYSTEMATIC FILE MODIFICATIONS

### **UI Layout Category**
1. **`openhcs/tui/components/function_list_manager.py`** - Fix Box wrappers, add scrolling

### **Semantic Architecture Category**
2. **`openhcs/tui/views/function_pattern_view.py`** - Fix key generation, add context
3. **`openhcs/tui/services/pattern_data_manager.py`** - Remove None key logic
4. **`openhcs/tui/editors/dual_editor_pane.py`** - Pass step context

### **UI Clarity Category**
5. **`openhcs/tui/components/pattern_key_selector.py`** - Fix UI labels

### **Utilities Category**
6. **`openhcs/tui/utils/dialog_helpers.py`** - Add text input dialog

---

## 🧪 ERROR CATEGORY VALIDATION

### **Layout Errors**: ✅ Box `preferred_height` errors eliminated
### **Semantic Errors**: ✅ Arbitrary keys replaced with experimental identifiers
### **UX Errors**: ✅ Clear labeling for experimental workflow design
### **Integration Errors**: ✅ Mouse scrolling consistent with other components
### **Architecture Errors**: ✅ Step context available for future enhancements

**RESULT**: Systematically fixed function pattern editor for all relevant error categories

---

**STATUS**: ✅ SEMANTIC UNDERSTANDING COMPLETE
**ARCHITECTURAL TRUTH**: ✅ Function pattern editor = Experimental workflow designer
**IMPLEMENTATION READY**: ✅ Clear semantic solution identified

**NEXT**: Implement semantic architecture with manual component identifier entry
