# plan_08_state_synchronization.md
## Component: Multi-Component State Synchronization System

### Objective
Create a unified state synchronization system that keeps all TUI components (PlateManagerPane, PipelineEditorPane, OrchestratorManager, SimpleTUIState) coordinated and consistent. Ensure that changes in one component properly propagate to all other components.

### Plan
1. **Audit Existing State Objects**
   - Map all state objects and their responsibilities
   - Identify overlapping data and potential conflicts
   - Document current observer patterns and event systems
   - Assess compatibility between different state management approaches

2. **Design State Synchronization Architecture**
   - Choose between centralized vs distributed state management
   - Design event system for cross-component communication
   - Define state ownership and update responsibilities
   - Create conflict resolution strategy for competing updates

3. **Implement State Event System**
   - Create unified event bus or extend existing observer patterns
   - Define standard event types for plate, pipeline, and orchestrator changes
   - Implement event listeners for each component
   - Add event validation and error handling

4. **Add State Consistency Validation**
   - Create state consistency checker
   - Implement automatic state repair for common inconsistencies
   - Add logging for state synchronization issues
   - Design recovery strategies for state corruption

5. **Test State Synchronization**
   - Test cross-component state updates
   - Test error handling and state recovery
   - Test edge cases (rapid updates, component failures)
   - Verify performance with multiple plates and complex pipelines

### Findings

#### **Comprehensive State Object Inventory**
**Certainty Level**: **Medium (50%)**

**INVESTIGATION STRATEGY FOR STATE ANALYSIS:**
1. **State Object Discovery**: Audit all components for state management patterns
2. **Data Flow Mapping**: Trace how data moves between components
3. **Event System Analysis**: Identify existing observer patterns and event systems
4. **Conflict Detection**: Find overlapping responsibilities and potential conflicts
5. **Synchronization Requirements**: Define what needs to stay in sync
6. **Performance Analysis**: Assess impact of synchronization on UI responsiveness

**IDENTIFIED STATE OBJECTS:**

1. **SimpleTUIState** (canonical_layout.py)
   - **Purpose**: Main TUI state coordination and global state management
   - **Data Structure**:
     ```python
     {
         'plates': [],                    # List of plate objects
         'selected_plate': None,          # Currently selected plate
         'current_pipeline': [],          # Pipeline for selected plate
         'orchestrator_status': {},       # plate_id -> status symbol
         'ui_state': {}                   # UI-specific state
     }
     ```
   - **Observer Pattern**: Basic `notify(event_type, data)` system
   - **Scope**: Global TUI state coordination
   - **Update Frequency**: High (UI interactions)

2. **PlateManagerPane Internal State**
   - **Purpose**: Plate management, validation, and display
   - **Data Structure**: Unknown - needs investigation
     ```python
     # Assumed structure:
     {
         'plate_list': [],               # Internal plate representation
         'selected_indices': [],         # Multi-selection state
         'validation_status': {},        # plate_id -> validation result
         'storage_state': {}             # Persistence state
     }
     ```
   - **Observer Pattern**: Unknown - needs investigation
   - **Scope**: Plate CRUD operations and validation
   - **Update Frequency**: Medium (user plate operations)

3. **OrchestratorManager State**
   - **Purpose**: Orchestrator lifecycle and status management
   - **Data Structure**:
     ```python
     {
         'orchestrators': {},            # plate_id -> PipelineOrchestrator
         'orchestrator_status': {},      # plate_id -> internal status
         'workspace_paths': {},          # plate_id -> workspace_path
         'creation_errors': {}           # plate_id -> error_info
     }
     ```
   - **Observer Pattern**: None currently implemented
   - **Scope**: Orchestrator instance management
   - **Update Frequency**: Low (orchestrator lifecycle events)

4. **Individual PipelineOrchestrator State**
   - **Purpose**: Per-plate execution state and compiled contexts
   - **Data Structure**:
     ```python
     {
         'is_initialized': bool,         # Initialization status
         'compiled_contexts': {},        # well_id -> ProcessingContext
         'execution_state': {},          # Current execution status
         'workspace_path': Path,         # Orchestrator workspace
         'error_state': {}               # Error information
     }
     ```
   - **Observer Pattern**: None
   - **Scope**: Single plate execution lifecycle
   - **Update Frequency**: Variable (depends on operations)

5. **PipelineEditorPane Internal State**
   - **Purpose**: Pipeline step management and editing state
   - **Data Structure**: Unknown - needs investigation
     ```python
     # Assumed structure:
     {
         'current_pipeline': [],         # List of step definitions
         'selected_step': None,          # Currently selected step
         'editor_state': {},             # Step editor state
         'dirty_flag': bool,             # Unsaved changes
         'undo_stack': []                # Undo/redo history
     }
     ```
   - **Observer Pattern**: Unknown - needs investigation
   - **Scope**: Pipeline editing and step management
   - **Update Frequency**: High (pipeline editing operations)

6. **DualStepFuncEditorPane State** (when active)
   - **Purpose**: Step and function pattern editing
   - **Data Structure**: Unknown - needs investigation
   - **Observer Pattern**: Unknown
   - **Scope**: Individual step editing
   - **Update Frequency**: High (during step editing)

**ADDITIONAL STATE OBJECTS TO INVESTIGATE:**
- **MenuBar State**: Menu selection and keyboard navigation state
- **StatusBar State**: Status messages, log drawer state, event history
- **File Browser State**: Directory navigation, selection state
- **Dialog State**: Modal dialog stack, dialog-specific state

**STATE OWNERSHIP ANALYSIS:**

| Data Type | Primary Owner | Secondary Owners | Conflicts |
|-----------|---------------|------------------|-----------|
| Plate List | PlateManagerPane | SimpleTUIState | Duplicate tracking |
| Selected Plate | SimpleTUIState | PlateManagerPane | Selection sync |
| Pipeline Definition | PipelineEditorPane | SimpleTUIState | Pipeline sync |
| Orchestrator Status | OrchestratorManager | SimpleTUIState | Status sync |
| UI Selection State | SimpleTUIState | All UI Components | Multi-component selection |

**CRITICAL STATE SYNCHRONIZATION REQUIREMENTS:**

1. **Plate-Orchestrator Consistency**
   - Every plate must have corresponding orchestrator (or null)
   - Orchestrator status must reflect actual orchestrator state
   - Plate removal must trigger orchestrator cleanup

2. **Pipeline-Plate Association**
   - Pipeline definition must match selected plate
   - Pipeline changes must invalidate compiled state
   - Multi-plate pipeline sharing coordination

3. **Selection State Coordination**
   - Selected plate must be consistent across all components
   - Selection changes must propagate to all interested components
   - Multi-selection state must be coordinated

4. **Status Symbol Accuracy**
   - Status symbols must reflect actual orchestrator lifecycle state
   - Status updates must be atomic and consistent
   - Error states must be properly represented and recoverable

#### **State Synchronization Challenges**

**Data Overlap Issues:**
- **Plate List**: Both SimpleTUIState and PlateManagerPane track plates
- **Pipeline Definition**: Both SimpleTUIState and PipelineEditorPane manage pipelines
- **Orchestrator Status**: Both SimpleTUIState and OrchestratorManager track status
- **Selection State**: Multiple components track what's currently selected

**Event System Conflicts:**
- **SimpleTUIState**: Basic observer pattern with notify() method
- **Production Components**: May have their own event systems
- **Orchestrator Events**: No event system currently
- **Cross-Component Events**: No coordination mechanism

**Update Ordering Issues:**
- **Plate Addition**: PlateManagerPane → OrchestratorManager → SimpleTUIState
- **Pipeline Changes**: PipelineEditorPane → SimpleTUIState → OrchestratorManager
- **Status Updates**: OrchestratorManager → SimpleTUIState → UI Components
- **Selection Changes**: UI → SimpleTUIState → All Components

#### **State Synchronization Architecture Options**

**Option 1: Centralized State (Redux-like)**
- **Pros**: Single source of truth, predictable updates, easy debugging
- **Cons**: Major refactoring required, may conflict with existing patterns
- **Effort**: High (complete rewrite of state management)

**Option 2: Event Bus System**
- **Pros**: Minimal changes to existing components, flexible, extensible
- **Cons**: Potential for event loops, harder to debug, loose coupling
- **Effort**: Medium (add event system, modify components)

**Option 3: State Coordinator**
- **Pros**: Centralized coordination, keeps existing patterns, clear ownership
- **Cons**: Single point of failure, potential bottleneck
- **Effort**: Medium (create coordinator, add integration points)

**Option 4: Enhanced Observer Pattern**
- **Pros**: Extends existing SimpleTUIState pattern, familiar, incremental
- **Cons**: May not scale well, potential for complex dependencies
- **Effort**: Low (extend existing observer system)

#### **Enhanced State Synchronization Architecture**

**COMPREHENSIVE ARCHITECTURE ANALYSIS:**

**Option 1: Centralized State Manager (Redux-like)**
```python
class CentralizedTUIState:
    def __init__(self):
        self.state = {
            'plates': {},                    # plate_id -> plate_data
            'orchestrators': {},             # plate_id -> orchestrator_info
            'pipelines': {},                 # plate_id -> pipeline_definition
            'selection': {                   # Selection state
                'current_plate': None,
                'selected_plates': [],
                'selected_step': None
            },
            'ui': {                          # UI-specific state
                'active_editor': None,
                'dialog_stack': [],
                'status_message': ''
            }
        }
        self.reducers = {}
        self.middleware = []

    def dispatch(self, action):
        """Redux-style action dispatch with middleware."""
        # Apply middleware
        for middleware in self.middleware:
            action = middleware(action, self.state)

        # Apply reducer
        reducer = self.reducers.get(action['type'])
        if reducer:
            new_state = reducer(self.state, action)
            self.state = new_state

        # Notify subscribers
        self.notify_subscribers(action['type'], action)
```

**Option 2: Event-Driven State Coordination**
```python
class EventDrivenStateCoordinator:
    def __init__(self):
        self.components = {}             # component_id -> component_instance
        self.event_bus = EventBus()
        self.state_validators = []
        self.conflict_resolvers = {}

    def register_component(self, component_id, component, state_interests):
        """Register component with its state interests."""
        self.components[component_id] = component

        # Subscribe to relevant events
        for event_type in state_interests:
            self.event_bus.subscribe(event_type,
                                   lambda data, comp=component: comp.handle_state_change(event_type, data))

    async def coordinate_state_change(self, source_component, event_type, data):
        """Coordinate state change across all interested components."""
        # Validate state change
        if not self.validate_state_change(event_type, data):
            raise ValueError(f"Invalid state change: {event_type}")

        # Check for conflicts
        conflicts = self.detect_conflicts(event_type, data)
        if conflicts:
            resolved_data = self.resolve_conflicts(conflicts, data)
        else:
            resolved_data = data

        # Emit event to all interested components
        await self.event_bus.emit(event_type, resolved_data)

        # Validate final state consistency
        await self.validate_global_consistency()
```

**Option 3: Hierarchical State Management**
```python
class HierarchicalStateManager:
    def __init__(self):
        self.global_state = GlobalTUIState()
        self.component_states = {}       # component_id -> local_state
        self.state_bridges = {}          # component_id -> state_bridge

    def create_component_bridge(self, component_id, component):
        """Create bidirectional state bridge for component."""
        bridge = ComponentStateBridge(
            component=component,
            global_state=self.global_state,
            local_state=self.component_states.get(component_id, {})
        )
        self.state_bridges[component_id] = bridge
        return bridge

class ComponentStateBridge:
    def __init__(self, component, global_state, local_state):
        self.component = component
        self.global_state = global_state
        self.local_state = local_state

    async def sync_to_global(self, local_changes):
        """Sync local component changes to global state."""
        # Validate changes
        # Apply to global state
        # Notify other components

    async def sync_from_global(self, global_changes):
        """Sync global state changes to local component."""
        # Filter relevant changes
        # Apply to local state
        # Update component UI
```

**Option 4: Enhanced Observer Pattern (Recommended)**
```python
class EnhancedTUIStateManager:
    def __init__(self):
        self.observers = defaultdict(list)   # event_type -> [observers]
        self.state_data = {
            'plates': {},                    # plate_id -> plate_info
            'orchestrators': {},             # plate_id -> orchestrator_status
            'pipelines': {},                 # plate_id -> pipeline_definition
            'selection': {                   # Selection coordination
                'current_plate': None,
                'selected_plates': [],
                'selected_step': None
            },
            'ui_state': {}                   # UI-specific state
        }
        self.state_validators = []
        self.state_history = []              # For debugging and undo
        self.consistency_checkers = []

    async def notify(self, event_type, data, source_component=None):
        """Enhanced notification with validation and consistency checking."""
        # Log state change
        self.state_history.append({
            'timestamp': time.time(),
            'event_type': event_type,
            'data': data,
            'source': source_component,
            'state_before': copy.deepcopy(self.state_data)
        })

        # Validate state change
        for validator in self.state_validators:
            if not validator.validate(event_type, data, self.state_data):
                raise ValueError(f"State validation failed for {event_type}")

        # Apply state change atomically
        async with self.state_lock:
            old_state = copy.deepcopy(self.state_data)
            try:
                self.apply_state_change(event_type, data)

                # Check consistency
                for checker in self.consistency_checkers:
                    if not checker.check_consistency(self.state_data):
                        # Rollback on consistency failure
                        self.state_data = old_state
                        raise ValueError(f"Consistency check failed after {event_type}")

                # Notify observers
                for observer in self.observers[event_type]:
                    try:
                        await observer(data, self.state_data)
                    except Exception as e:
                        logger.error(f"Observer error for {event_type}: {e}")

            except Exception as e:
                # Rollback on any error
                self.state_data = old_state
                raise

    def add_consistency_checker(self, checker):
        """Add consistency checker for state validation."""
        self.consistency_checkers.append(checker)

    def add_state_validator(self, validator):
        """Add validator for state changes."""
        self.state_validators.append(validator)
```

**ARCHITECTURE DECISION MATRIX:**

| Approach | Complexity | Performance | Maintainability | Integration Effort | Risk |
|----------|------------|-------------|-----------------|-------------------|------|
| Centralized (Redux) | High | Good | Excellent | High | Medium |
| Event-Driven | Medium | Good | Good | Medium | Medium |
| Hierarchical | High | Excellent | Good | High | Low |
| Enhanced Observer | Low | Excellent | Good | Low | Low |

**RECOMMENDED APPROACH: Enhanced Observer Pattern**

**Rationale**:
- Builds on existing SimpleTUIState observer pattern
- Minimal disruption to production components
- Incremental implementation possible
- Familiar pattern for maintenance
- Low risk and effort
- Excellent performance characteristics

**Key Enhancements:**
- **State Validation**: Validate changes before applying
- **Consistency Checking**: Ensure state remains consistent
- **Atomic Updates**: All-or-nothing state changes
- **Error Recovery**: Rollback on failures
- **State History**: Debugging and undo capabilities
- **Performance Optimization**: Efficient observer notification

#### **Event Types to Implement**

**Plate Events:**
- `plate_added`: New plate created
- `plate_removed`: Plate deleted
- `plate_selected`: Selection changed
- `plate_status_changed`: Status symbol updated

**Pipeline Events:**
- `pipeline_modified`: Steps added/removed/edited
- `pipeline_compiled`: Compilation successful
- `pipeline_compilation_failed`: Compilation error

**Orchestrator Events:**
- `orchestrator_created`: New orchestrator instance
- `orchestrator_initialized`: Initialization complete
- `orchestrator_error`: Operation failed

**UI Events:**
- `selection_changed`: User changed selection
- `editor_opened`: Step editor opened
- `dialog_shown`: Modal dialog displayed

#### **State Consistency Rules**

**Invariants to Maintain:**
1. Every plate must have corresponding orchestrator (or null)
2. Every orchestrator must have corresponding plate
3. Pipeline definition must match selected plate
4. Status symbols must reflect actual orchestrator state
5. Selection state must be consistent across components

**Consistency Checks:**
- Orphaned orchestrators (orchestrator without plate)
- Missing orchestrators (plate without orchestrator)
- Status mismatches (status doesn't match orchestrator state)
- Selection conflicts (multiple components think different plates selected)

#### **Critical Integration Points**
1. **PlateManagerPane Integration**: Hook into plate add/remove operations
2. **PipelineEditorPane Integration**: Hook into pipeline modification events
3. **OrchestratorManager Integration**: Hook into orchestrator lifecycle events
4. **UI Component Integration**: Hook into selection and interaction events

#### **Risk Assessment (Updated)**
**~~High Risk~~**: **REDUCED** - Production components confirmed to support event integration
**Medium Risk**: Event system may introduce performance overhead
**Low Risk**: Observer pattern is well-understood and debuggable

#### **Cross-Validation Impact**
**Plan 08 Status**: **VALIDATED AND ENHANCED** - Cross-validation confirmed component architecture

**Key Confirmations from Cross-Validation:**
- **SimpleTUIState** has robust observer pattern: `add_observer()` and `notify()` (simple_launcher.py:26-55)
- **PlateManagerController** uses 8 event types: `refresh_plates`, `plate_status_changed`, `ui_request_show_add_plate_dialog`, `add_predefined_plate`, `plates_removed`, `plate_manager_ui_update`, `error`, `plate_selected`
- **PipelineEditorPane** uses 6 event types: `plate_selected`, `steps_updated`, `step_pattern_saved`, `edit_step_dialog_requested`, `operation_status_changed`, `info`
- **Production event handlers confirmed**: 15+ actual event types found in active production code
- **function_pattern_editor.py** and **dual_step_func_editor.py** are confirmed essential components
- **No risk of component removal** - all components are production-ready

**Enhanced Integration Strategy:**
- **Leverage existing event system** in PlateManagerPane (confirmed robust)
- **Coordinate with confirmed essential components** (function_pattern_editor.py, dual_step_func_editor.py)
- **Build on proven observer pattern** already implemented in production
- **No architectural conflicts** with cleanup plan (Plan 09)

**Revised Risk Assessment:**
- **Risk Level**: **Reduced from High to Low** (event integration confirmed supported)
- **Complexity**: **Reduced** (can build on existing event system)
- **Dependencies**: All components confirmed stable and essential

**Updated Critical Integration Points:**
1. **PlateManagerPane Integration**: ✅ **CONFIRMED** - 8 event types in production use
2. **PipelineEditorPane Integration**: ✅ **CONFIRMED** - 6 event types in production use
3. **OrchestratorManager Integration**: Straightforward (canonical_layout.py)
4. **UI Component Integration**: ✅ **CONFIRMED** - Observer pattern extensively used

#### **Detailed Cross-Validation Evidence**

**SimpleTUIState Observer Pattern (simple_launcher.py:26-55):**
```python
def add_observer(self, event_type: str, callback):
    if event_type not in self.observers:
        self.observers[event_type] = []
    self.observers[event_type].append(callback)

async def notify(self, event_type: str, data=None):
    if event_type in self.observers:
        for callback in self.observers[event_type]:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)
```

**PlateManagerController Event Registration (lines 49-55):**
```python
self.state.add_observer('refresh_plates', self._handle_refresh_request)
self.state.add_observer('plate_status_changed', self._handle_plate_status_changed)
self.state.add_observer('ui_request_show_add_plate_dialog', self._handle_show_add_dialog_request)
self.state.add_observer('add_predefined_plate', self._handle_add_predefined_plate)
```

**PipelineEditorPane Event Registration (lines 178-186):**
```python
self.state.add_observer('plate_selected', lambda plate: get_app().create_background_task(self._on_plate_selected(plate)))
self.state.add_observer('steps_updated', lambda _: get_app().create_background_task(self._refresh_steps()))
self.state.add_observer('step_pattern_saved', lambda data: get_app().create_background_task(self._handle_step_pattern_saved(data)))
self.state.add_observer('edit_step_dialog_requested', lambda data: get_app().create_background_task(self._handle_edit_step_request(data)))
```

**Event Emission Examples:**
```python
# PlateManagerController line 187:
await self.state.notify('plate_selected', plate_data)

# PipelineEditorPane line 629:
await self.state.notify('operation_status_changed', {
    'message': f"Pipeline saved to {file_path}",
    'status': 'success',
    'source': self.__class__.__name__
})
```

**Cross-Validation Confidence**: **High (95%)** - Direct code observation of production event system

### Implementation Draft
*Implementation will be added after investigation and smell loop approval*

**Implementation Priority**: **High** (enables all other integrations)
**Complexity**: **Low** (reduced from Medium - can leverage existing systems)
**Risk**: **Low** (reduced from Medium - components confirmed stable)
**Dependencies**: Plans 06, 07 (coordination bridges), Plan 09 (cleanup - no conflicts)
