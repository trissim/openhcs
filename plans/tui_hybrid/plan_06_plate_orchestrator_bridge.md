# plan_06_plate_orchestrator_bridge.md
## Component: Plate-to-Orchestrator Integration Bridge

### Objective
Create the missing integration bridge between PlateManagerPane's plate management and OrchestratorManager's orchestrator lifecycle. When users add/remove plates in the UI, corresponding orchestrators must be created/destroyed automatically.

### Plan
1. **Investigate PlateManagerPane Event System**
   - Audit PlateManagerPane service layer for event emission
   - Identify add/remove/edit plate operations
   - Determine if observer pattern exists or needs to be added
   - Map plate data structure to orchestrator requirements

2. **Create Plate-Orchestrator Event Bridge**
   - Design event listener that connects PlateManagerPane to OrchestratorManager
   - Handle plate addition: folder selection → orchestrator creation
   - Handle plate removal: orchestrator cleanup and shutdown
   - Handle plate editing: orchestrator reconfiguration if needed

3. **Implement Orchestrator Status Integration**
   - Connect orchestrator state to TUI status symbols
   - Map orchestrator lifecycle to status progression: `?` → `-` → `o` → `!`
   - Update PlateManagerPane display when orchestrator status changes
   - Handle error states and status rollback

4. **Add Multi-Folder Selection Support**
   - Extend file browser to support multiple folder selection
   - Create multiple orchestrators from single add operation
   - Batch orchestrator creation with proper error handling
   - Update UI to show all created plates with initial `?` status

5. **Test Integration Flow**
   - Test add plate → orchestrator creation → status display
   - Test remove plate → orchestrator cleanup
   - Test error handling during orchestrator creation
   - Verify state consistency between components

### Findings

#### **PlateManagerPane Complete Architecture Analysis**
**Certainty Level**: **High (95%)**

**INVESTIGATION COMPLETED - MAJOR DISCOVERY:**

**🎯 CRITICAL FINDING: PlateManagerPane ALREADY CREATES ORCHESTRATORS**

The investigation reveals that PlateManagerPane is **already orchestrator-aware** and creates PipelineOrchestrator instances for each plate. This fundamentally changes our integration strategy.

**1. CLEAN MVC ARCHITECTURE CONFIRMED**
```
PlateManagerPane (Facade)
├── PlateManagerService (Business Logic)
│   ├── plates: List[Dict] (Thread-safe with asyncio.Lock)
│   ├── add_plate() -> Creates PipelineOrchestrator ← KEY DISCOVERY
│   ├── update_plate_status() -> Updates plate status
│   └── get_plates() -> Returns plate data
├── PlateManagerController (Event Coordination)
│   ├── Event Handlers: refresh_plates, plate_status_changed, etc.
│   ├── Dialog Management: PlateDialogManager integration
│   └── State Notifications: plate_selected, plates_removed, etc.
└── PlateManagerView (UI Rendering)
    ├── Interactive List Items with status indicators
    ├── Status Symbols: ✅ ⚪ 🔄 ❌ (already implemented!)
    └── Button Handlers: Add, Remove, Refresh
```

**2. ORCHESTRATOR INTEGRATION ALREADY EXISTS**
```python
# From PlateManagerService.add_plate():
orchestrator = PipelineOrchestrator(
    plate_path=path_str,
    config=global_config,
    storage_registry=self.registry
)

new_plate_entry = {
    'id': plate_tui_id,
    'name': Path(path_str).name,
    'path': path_str,
    'status': 'not_initialized',
    'orchestrator': orchestrator,  # ← ORCHESTRATOR STORED HERE
    'backend': orchestrator.config.vfs.default_storage_backend
}
```

**3. ROBUST EVENT SYSTEM DISCOVERED**
```python
# Controller registers these event handlers:
self.state.add_observer('refresh_plates', self._handle_refresh_request)
self.state.add_observer('plate_status_changed', self._handle_plate_status_changed)
self.state.add_observer('ui_request_show_add_plate_dialog', self._handle_show_add_dialog_request)
self.state.add_observer('add_predefined_plate', self._handle_add_predefined_plate)

# Controller emits these events:
await self.state.notify('plate_selected', {'plate': selected_plate, 'index': index})
await self.state.notify('plates_removed', {'plate_ids': plate_ids, 'count': num_removed})
await self.state.notify('plate_manager_ui_update', {...})
```

**4. STATUS MANAGEMENT SYSTEM COMPLETE**
```python
# Service layer handles status updates:
async def update_plate_status(self, plate_id: str, new_status: str, message: Optional[str] = None):
    # Updates plate['status'] and plate['error_message']

# View layer displays status with indicators:
status_indicators = {
    'ready': '✅',
    'not_initialized': '⚪',
    'initializing': '🔄',
    'error': '❌',
    'running': '🔄',
    'completed': '✅'
}
```

**5. PLATE DATA STRUCTURE IDENTIFIED**
```python
plate_entry = {
    'id': f"{plate_name}_{backend}",           # Unique identifier
    'name': plate_name,                        # Display name
    'path': plate_path,                        # File system path
    'status': 'not_initialized',               # Status for UI display
    'orchestrator': PipelineOrchestrator(...), # ORCHESTRATOR INSTANCE
    'backend': backend_type,                   # Storage backend
    'error_message': optional_error            # Error details if status='error'
}
```

**ARCHITECTURAL ASSESSMENT:**
✅ **PlateManagerPane is ALREADY orchestrator-aware**
✅ **Clean event system for status synchronization**
✅ **Thread-safe data management with asyncio.Lock**
✅ **Proper MVC separation with clear integration points**
✅ **Status indicators already implemented**
✅ **Error handling and validation systems in place**

**INTEGRATION STRATEGY COMPLETELY REVISED:**
This is NOT a "bridge" problem - it's a **coordination problem**.

The issue is that we have TWO orchestrator management systems:
1. **PlateManagerService** (creates and stores orchestrators per plate)
2. **OrchestratorManager** (canonical_layout.py - separate orchestrator storage)

We need **orchestrator coordination**, not orchestrator creation.

#### **Revised Integration Requirements**
**Based on architectural discovery:**

**PROBLEM IDENTIFIED: Dual Orchestrator Management**
- **PlateManagerService**: Creates orchestrators per plate (production system)
- **OrchestratorManager**: Separate orchestrator storage (canonical_layout.py)
- **Result**: Duplicate orchestrator instances, inconsistent state

**SOLUTION: Orchestrator Coordination Bridge**

**Add Plate Flow (Revised):**
```
1. User clicks [add] button in PlateManagerPane
2. File browser opens with multi-folder selection
3. User selects one or more folders
4. For each folder:
   - PlateManagerPane.add_plate() creates PipelineOrchestrator (existing)
   - Coordination Bridge detects plate_added event
   - Bridge registers orchestrator with OrchestratorManager
   - Status synchronization ensures consistent display
```

**Remove Plate Flow (Revised):**
```
1. User selects plate(s) and clicks [del] button
2. PlateManagerPane removes plate from its service (existing)
3. Coordination Bridge detects plates_removed event
4. Bridge removes orchestrator from OrchestratorManager
5. UI updates reflect consistent state
```

**Key Insight**: We don't need to CREATE orchestrators - we need to COORDINATE existing ones.

#### **Status Symbol Integration**
**Orchestrator State → TUI Status Mapping:**
- **Created**: `orchestrator = PipelineOrchestrator(...)` → `?` (gray)
- **Initialized**: `orchestrator.initialize()` success → `-` (yellow)
- **Compiled**: `orchestrator.compile_pipelines()` success → `o` (green)
- **Running**: `orchestrator.execute_compiled_plate()` → `!` (red)
- **Error**: Any operation failure → `!` (red) + error message

#### **Orchestrator Coordination Bridge Design**
**Revised Integration Pattern:**

**COORDINATION BRIDGE: Event-Based Orchestrator Sync**
```python
class PlateOrchestratorCoordinationBridge:
    """
    Coordinates between PlateManagerService orchestrators and OrchestratorManager.

    Ensures single source of truth while maintaining compatibility with both systems.
    """

    def __init__(self, plate_manager_pane, orchestrator_manager, tui_state):
        self.plate_manager = plate_manager_pane
        self.orchestrator_manager = orchestrator_manager
        self.tui_state = tui_state

        # Register as observer of plate manager events
        self.tui_state.add_observer('plate_added', self.on_plate_added)
        self.tui_state.add_observer('plates_removed', self.on_plates_removed)
        self.tui_state.add_observer('plate_selected', self.on_plate_selected)

    async def on_plate_added(self, event_data):
        """Handle plate addition by registering existing orchestrator."""
        try:
            plate = event_data.get('plate')
            if not plate:
                return

            plate_id = plate['id']
            plate_path = plate['path']
            orchestrator = plate.get('orchestrator')

            if orchestrator:
                # Register the EXISTING orchestrator with OrchestratorManager
                # Don't create a new one - just coordinate the existing one
                await self.orchestrator_manager.register_existing_orchestrator(
                    plate_id, orchestrator
                )

                # Sync status between systems
                await self._sync_orchestrator_status(plate_id, orchestrator)

                logger.info(f"Coordinated orchestrator for plate {plate_id}")
            else:
                logger.warning(f"Plate {plate_id} added without orchestrator")

        except Exception as e:
            logger.error(f"Error coordinating plate addition: {e}")

    async def on_plates_removed(self, event_data):
        """Handle plate removal by unregistering orchestrators."""
        try:
            plate_ids = event_data.get('plate_ids', [])

            for plate_id in plate_ids:
                # Remove from OrchestratorManager coordination
                await self.orchestrator_manager.unregister_orchestrator(plate_id)

                logger.info(f"Unregistered orchestrator for plate {plate_id}")

        except Exception as e:
            logger.error(f"Error coordinating plate removal: {e}")

    async def on_plate_selected(self, event_data):
        """Handle plate selection by updating active orchestrator."""
        try:
            plate = event_data.get('plate')
            if plate:
                plate_id = plate['id']
                orchestrator = plate.get('orchestrator')

                # Update active orchestrator in TUI state
                await self.tui_state.notify('active_orchestrator_changed', {
                    'plate_id': plate_id,
                    'orchestrator': orchestrator
                })

        except Exception as e:
            logger.error(f"Error coordinating plate selection: {e}")

    async def _sync_orchestrator_status(self, plate_id, orchestrator):
        """Sync orchestrator status between systems."""
        # Determine orchestrator status
        if hasattr(orchestrator, 'is_initialized') and orchestrator.is_initialized:
            status = 'initialized'
        else:
            status = 'not_initialized'

        # Update PlateManagerService status
        await self.plate_manager.service.update_plate_status(plate_id, status)

        # Update TUI state
        await self.tui_state.notify('orchestrator_status_changed', {
            'plate_id': plate_id,
            'status': status
        })
```

**PATTERN 2: Service Layer Interception**
```python
class EnhancedPlateManagerService:
    def __init__(self, original_service, orchestrator_bridge):
        self.original_service = original_service
        self.bridge = orchestrator_bridge

    async def add_plate(self, plate_data):
        """Enhanced add_plate with orchestrator integration."""
        # Call original service method
        result = await self.original_service.add_plate(plate_data)

        if result.success:
            # Trigger orchestrator creation
            await self.bridge.create_orchestrator_for_plate(plate_data)

        return result
```

**PATTERN 3: Event Bus Integration**
```python
class TUIEventBus:
    def __init__(self):
        self.listeners = defaultdict(list)

    def subscribe(self, event_type, handler):
        self.listeners[event_type].append(handler)

    async def emit(self, event_type, data):
        for handler in self.listeners[event_type]:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

# Usage:
event_bus.subscribe('plate_added', orchestrator_bridge.create_orchestrator)
event_bus.subscribe('orchestrator_status_changed', ui_updater.update_status)
```

**INTEGRATION DECISION MATRIX:**

| Pattern | Pros | Cons | Effort | Risk |
|---------|------|------|--------|------|
| Observer Extension | Clean, standard pattern | Requires existing observer support | Medium | Medium |
| Service Interception | Works with any service | Requires service replacement | High | Low |
| Event Bus | Very flexible, decoupled | New infrastructure needed | High | Medium |

**RECOMMENDED APPROACH**: Start with Observer Extension, fallback to Service Interception

**Required Integration Points:**

1. **PlateManagerPane → OrchestratorManager**
   - **Event**: `plate_added` → Create orchestrator
   - **Event**: `plate_removed` → Cleanup orchestrator
   - **Event**: `plate_validated` → Update orchestrator config
   - **Error Handling**: Orchestrator creation failures

2. **OrchestratorManager → PlateManagerPane**
   - **Event**: `orchestrator_status_changed` → Update UI status symbol
   - **Event**: `orchestrator_error` → Display error in plate list
   - **Event**: `orchestrator_initialized` → Enable compile button

3. **Bidirectional State Synchronization**
   - **Consistency Check**: Ensure every plate has orchestrator
   - **Orphan Cleanup**: Remove orchestrators without plates
   - **Status Reconciliation**: Sync status symbols with orchestrator state
   - **Selection Coordination**: Keep selected plate in sync across components

#### **Multi-Folder Selection Requirements**
**From tui_final.md**: "multiple folders may be selected at once"

**Implementation Needs:**
- File browser component with multi-selection capability
- Batch orchestrator creation from folder list
- Error handling for partial failures (some folders valid, some invalid)
- UI feedback during batch creation process

#### **Integration Gaps RESOLVED**
✅ **Event System**: Robust observer pattern with state.add_observer() and state.notify()
✅ **Data Structure**: Complete plate data format identified with orchestrator field
✅ **Error Handling**: Comprehensive error handling with status updates and error messages
✅ **State Updates**: Event-driven UI updates via plate_manager_ui_update events

#### **Risk Assessment UPDATED**
**~~High Risk~~**: ✅ **RESOLVED** - PlateManagerPane has excellent integration hooks
**~~Medium Risk~~**: ✅ **RESOLVED** - Event system is robust and well-designed
**Low Risk**: ✅ **CONFIRMED** - Orchestrator coordination is straightforward

#### **Key Architectural Insights**
1. **No Bridge Needed**: PlateManagerPane already creates orchestrators
2. **Coordination Required**: Need to sync between PlateManagerService and OrchestratorManager
3. **Event System Ready**: Existing observer pattern can handle coordination
4. **Status System Complete**: UI status indicators already implemented
5. **Clean Architecture**: MVC separation provides clear integration points

#### **Next Steps for Plan 06**
1. **Implement Coordination Bridge**: Create PlateOrchestratorCoordinationBridge
2. **Extend OrchestratorManager**: Add register_existing_orchestrator() method
3. **Test Event Flow**: Verify plate_added/plates_removed events work correctly
4. **Status Synchronization**: Ensure status updates flow between systems
5. **Integration Testing**: Test add/remove plate flows with coordination

#### **Cross-Validation Impact**
**Plan 06 Status**: **VALIDATED** - No changes needed based on cross-validation
- PlateManagerPane architecture analysis remains accurate
- Integration strategy (coordination vs creation) confirmed correct
- No dependency on components identified as legacy in Plan 09

### Implementation Draft
⚠️ **PARTIALLY IMPLEMENTED (70% Complete)**

**Implementation Status**: **FUNCTIONAL BUT NEEDS REFINEMENT**

#### **✅ What's Properly Implemented (70%)**
- ✅ PlateOrchestratorCoordinationBridge core architecture and event handling
- ✅ OrchestratorManager extended with register_existing_orchestrator() and unregister_orchestrator()
- ✅ Event flow tested: plate_added → orchestrator registration → plates_removed → orchestrator unregistration
- ✅ Bridge lifecycle (initialize/shutdown) working
- ✅ Core coordination pattern is architecturally sound

#### **⚠️ Duck Tape Fixes That Need Proper Implementation (30%)**

**1. PlateManagerController Validation Integration**
- **Duck Tape**: Added `_setup_validation_callbacks()` and `_handle_validation_result()` without verifying PlateValidationService constructor
- **Assumption**: PlateValidationService takes `(state, context)` parameters
- **Risk**: Might not match actual validation service initialization
- **Needs**: Verify actual PlateValidationService integration pattern

**2. Global Config Access Pattern**
- **Duck Tape**: Controller gets global_config from `self.state.global_config`
- **Assumption**: State always has global_config available
- **Risk**: Might fail if state doesn't have global_config in some usage patterns
- **Needs**: Verify global_config availability across all initialization paths

**3. SimpleTUIState Enhancement**
- **Duck Tape**: Added `remove_observer()` to "simple" state class
- **Assumption**: This doesn't break the minimal design intent
- **Risk**: Might conflict with other state implementations
- **Needs**: Verify this is the correct state class to extend

**4. Event Timing and Consistency**
- **Not Addressed**: Event ordering, race conditions, state consistency
- **Risk**: Events might arrive out of order or during invalid states
- **Needs**: Proper event sequencing and state validation

#### **Files Created/Modified:**
1. **NEW**: `openhcs/tui/plate_orchestrator_bridge.py` - Core bridge (solid)
2. **MODIFIED**: `openhcs/tui/orchestrator_manager.py` - Extensions (solid)
3. **MODIFIED**: `openhcs/tui/simple_launcher.py` - remove_observer() (duck tape)
4. **MODIFIED**: `openhcs/tui/controllers/plate_manager_controller.py` - Validation integration (duck tape)
5. **MODIFIED**: `openhcs/tui/canonical_layout.py` - Bridge initialization (duck tape assumptions)

#### **Test Results**: ✅ Basic coordination working, ⚠️ Integration assumptions untested
- Event registration/unregistration: ✅ Working in isolation
- Orchestrator coordination: ✅ Working with mock data
- Bridge lifecycle: ✅ Working
- **Real integration**: ⚠️ Untested with actual PlateManagerPane initialization
- **Edge cases**: ⚠️ Not tested (missing global_config, validation failures, etc.)

#### **🔧 Remaining Work to Complete Plan 06 (30%)**

**Priority**: **Medium** (should be done during Plans 07-08 integration)

**Tasks to Complete:**
1. **Verify PlateValidationService Integration**
   - Check actual constructor signature and initialization pattern
   - Verify callback setup matches existing architecture
   - Test with real validation service, not mocks

2. **Validate Global Config Access**
   - Trace all initialization paths to confirm state.global_config availability
   - Add fallback mechanisms for missing global_config
   - Test edge cases where global_config might be None

3. **Proper State Class Integration**
   - Verify SimpleTUIState is the correct class to extend
   - Check if other state implementations need remove_observer()
   - Ensure consistency across all state management

4. **Event Sequencing and Error Handling**
   - Add event ordering validation
   - Handle race conditions between plate operations
   - Add comprehensive error recovery for failed coordination

5. **Integration Testing**
   - Test with actual PlateManagerPane initialization flow
   - Test error scenarios (validation failures, missing orchestrators)
   - Test concurrent plate operations

**Completion Strategy**: Address these during Plans 07-08 implementation when we have better understanding of the full integration patterns.

**Implementation Priority**: **Medium** (coordination vs creation)
**Complexity**: **Low** (event-based coordination) → **Medium** (when including proper integration)
**Risk**: **Low** (well-defined integration points) → **Medium** (due to integration assumptions)
**Dependencies**: None (independent of cleanup plan)
