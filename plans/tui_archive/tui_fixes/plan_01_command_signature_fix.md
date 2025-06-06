# plan_01_command_signature_fix.md
## Component: Command Signature Standardization

### Objective
Fix the command signature mismatch between PlateManagerPane button handlers and Command.execute() methods, identified by DNA analysis as the highest priority issue (Priority: 10.5, Complexity: 147, 36 issues).

### DNA Mathematical Evidence
```
Ψ[2P2SPUAJ:1:147]](C³₂){σ}⟨λG⟩  # Line 147: High complexity in plate_manager_core.py
Ψ[2P2SP2MB:1:82]](C³₂){σ}⟨λG⟩   # Line 82: High complexity in commands.py
FN[PUAJ:getplatedisplay:11]       # Function complexity hotspot
FN[P2MB:execute:10]               # Command execute complexity hotspot
```

**Ultra-Dense Notation:**
- `Φ{0P2SPGDI→plate_manager_core.py,0P00PPO6→commands.py}`
- **Coupling Issue**: `CP[PUAJ→P2MB:2]` (PlateManagerPane → Commands)

### Plan

#### **Phase 1: Analyze Current Signature Patterns**

**1.1 Document Current Command Signatures**
- **File**: `openhcs/tui/commands.py`
- **Lines**: 105, 164, 204, 230, 253, 269, 321, 386, 456, 517, 571
- **Current Pattern**: `async def execute(self, state: "TUIState" = None, context: "ProcessingContext" = None, **kwargs: Any) -> None:`

**1.2 Document Current Caller Patterns**
- **File**: `openhcs/tui/plate_manager_core.py`
- **Lines**: 200-217
- **Current Pattern**: `Command().execute(self.state, self.context, param=value)`

**1.3 Identify Interface Mismatch**
- **Problem**: Commands expect optional parameters, callers pass positional parameters
- **Root Cause**: No standardized command invocation pattern
- **Impact**: 11 command classes affected, 6 button handlers affected

#### **Phase 2: Design Standardized Command Interface**

**2.1 Define Standard Command Signature**
```python
async def execute(self, state: "TUIState", context: "ProcessingContext", **kwargs: Any) -> None:
    """
    Standard command execution interface.
    
    Args:
        state: TUIState instance (required)
        context: ProcessingContext instance (required) 
        **kwargs: Command-specific parameters
    """
```

**2.2 Define Standard Command Invocation Pattern**
```python
# Pattern for button handlers:
command = CommandClass()
await command.execute(self.state, self.context, param1=value1, param2=value2)

# Pattern for async background tasks:
get_app().create_background_task(
    CommandClass().execute(self.state, self.context, param1=value1)
)
```

**2.3 Define ProcessingContext Interface Requirements**
- **Required Properties**: `filemanager`, `storage_registry`
- **Required Methods**: None currently identified
- **Validation**: Commands must check `context.filemanager is not None`

#### **Phase 3: Update Command Signatures (11 Commands)**

**3.1 Update ShowGlobalSettingsDialogCommand**
- **File**: `openhcs/tui/commands.py`
- **Lines**: 105-108
- **Change**: Remove `= None` defaults from state and context parameters
- **Validation**: Add `assert state is not None` and `assert context is not None`

**3.2 Update ShowHelpCommand**
- **File**: `openhcs/tui/commands.py`
- **Lines**: 164-167
- **Change**: Same signature standardization
- **Validation**: Same assertions

**3.3 Update PlateManagerPane Commands (6 commands)**
- **ShowAddPlateDialogCommand** (Line 204)
- **DeleteSelectedPlatesCommand** (Line 230)
- **ShowEditPlateConfigDialogCommand** (Line 253)
- **InitializePlatesCommand** (Line 269)
- **CompilePlatesCommand** (Line 321)
- **RunPlatesCommand** (Line 386)

**3.4 Update PipelineEditorPane Commands (3 commands)**
- **AddStepCommand** (Line 456)
- **DeleteSelectedStepsCommand** (Line 517)
- **ShowEditStepDialogCommand** (Line 571)

#### **Phase 4: Update Command Invocations (6 Button Handlers)**

**4.1 Update PlateManagerPane Button Handlers**
- **File**: `openhcs/tui/plate_manager_core.py`
- **Lines**: 200-217

**Current Code:**
```python
self.add_button = FramedButton("Add", handler=lambda: get_app().create_background_task(
    ShowAddPlateDialogCommand().execute(self.state, self.context, plate_dialog_manager=self.dialog_manager)
), width=6)
```

**New Code:**
```python
self.add_button = FramedButton("Add", handler=lambda: get_app().create_background_task(
    ShowAddPlateDialogCommand().execute(self.state, self.context, plate_dialog_manager=self.dialog_manager)
), width=6)
```
*Note: No change needed in invocation, only in command signature*

**4.2 Verify ProcessingContext Interface**
- **Check**: `self.context.filemanager` exists and is not None
- **Check**: `self.context.storage_registry` exists if needed
- **Location**: PlateManagerPane.__init__ and _on_filemanager_available

#### **Phase 5: Add Command Validation**

**5.1 Add Base Command Validation**
- **File**: `openhcs/tui/commands.py`
- **Location**: After line 50 (base Command class)
- **Add Method**:
```python
def _validate_execution_context(self, state: "TUIState", context: "ProcessingContext") -> None:
    """Validate that state and context are properly provided."""
    if state is None:
        raise ValueError(f"{self.__class__.__name__}: state parameter is required")
    if context is None:
        raise ValueError(f"{self.__class__.__name__}: context parameter is required")
    if not hasattr(context, 'filemanager'):
        raise ValueError(f"{self.__class__.__name__}: context.filemanager is required")
```

**5.2 Add Validation Calls**
- **Location**: First line of each command's execute() method
- **Code**: `self._validate_execution_context(state, context)`

### Findings

#### **DNA Analysis Results**
- **Total Issues**: 74 (36 in plate_manager_core.py + 38 in commands.py)
- **Complexity Hotspots**: Lines 147, 258 (plate_manager_core.py), Line 82 (commands.py)
- **Coupling Issues**: PlateManagerPane → Commands (strength: 2)
- **Entropy Levels**: 111.5 (plate_manager_core.py), 114.4 (commands.py)

#### **Command Interface Analysis**
- **11 Command Classes** with inconsistent signatures
- **6 Button Handlers** in PlateManagerPane calling commands
- **ProcessingContext Usage**: Commands expect `context.filemanager`
- **TUIState Usage**: Commands expect `state.active_orchestrator`, `state.global_config`

#### **Risk Assessment**
- **High Risk**: Commands failing silently due to None parameters
- **Medium Risk**: ProcessingContext interface changes breaking commands
- **Low Risk**: TUIState interface changes (more stable)

#### **Dependencies**
- **No external dependencies** - this is a pure interface fix
- **No database changes** required
- **No configuration changes** required
- **Testing**: Manual verification of button functionality

#### **Success Criteria**
1. All 11 commands have standardized signatures
2. All 6 button handlers work without errors
3. No None parameter errors in command execution
4. ProcessingContext validation passes for all commands
5. DNA analysis shows reduced complexity and coupling

### Implementation Draft
*To be added after smell loop approval*
