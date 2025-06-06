# TUI Final Implementation Plans

This directory contains the comprehensive implementation plans for completing the OpenHCS TUI according to the specification in `/plans/snapshot_fix/tui_final.md`.

## Plan Sequence

**🎉 SIMPLIFIED PLANS BASED ON REALITY (86% COMPLETE!)**

### Phase 1: Copy and Adapt Required Components (6.5 hours)
**plan_SIMPLE_01_copy_missing_components.md** - Copy FramedButton and StatusBar from archive (MenuBar not needed for canonical layout)

### Phase 2: Create Canonical Layout Structure (12 hours)
**plan_SIMPLE_02_create_layout_components.md** - Implement exact layout from tui_final.md with dual editor system

### Phase 3: Canonical Status Symbol System (8 hours)
**plan_SIMPLE_03_fix_status_symbols.md** - Implement `?`/`-`/`o`/`!` progression with orchestrator lifecycle tracking

### Phase 4: Orchestrator Integration & File Browser (11 hours)
**plan_SIMPLE_04_fix_file_browser.md** - Implement canonical button functionality with orchestrator methods and multi-folder selection

### Phase 5: Integration & Manual Verification (14 hours)
**plan_SIMPLE_05_integration_testing.md** - Wire everything together and verify against canonical specification

**TOTAL ESTIMATED TIME: ~51.5 hours (refined after canonical specification alignment)**

## Key Objectives

**🎯 CANONICAL SPECIFICATION ALIGNMENT:**

After reviewing tui_final.md, the plans have been rewritten to implement the exact canonical specification rather than generic TUI assumptions.

**CANONICAL REQUIREMENTS FROM tui_final.md:**
- **Exact layout structure**: 3-row header + dual-pane main + status bar
- **Orchestrator integration**: Buttons call `PipelineOrchestrator` methods (`initialize()`, `compile_pipelines()`, `execute_compiled_plate()`)
- **Status symbol progression**: `?` (created) → `-` (initialized) → `o` (compiled) → `!` (running/error)
- **Dual editor system**: Step/Func toggle using existing `FunctionPatternEditor` that replaces plate manager pane
- **Multi-folder selection**: "multiple folders may be selected at once" for add plate
- **Clean architecture**: TUI manages orchestrator instances externally, never accesses `ProcessingContext` or internal state

**CRITICAL IMPLEMENTATION PATTERNS DOCUMENTED:**
- **FunctionStep Construction**: All step editor parameters + pattern editor output → `FunctionStep(func=pattern, ...)`
- **File Operations**: Pickle load/save for `.pipeline`, `.step`, `.func` files → assign to appropriate variables
- **Error Recovery**: DON'T update status flags on failure - keep at previous state, show error in status bar + modal dialog
- **Multi-Folder Creation**: File dialog → multiple `PipelineOrchestrator` instances → track in TUI state

**IMPLEMENTATION TASKS:**
1. **Copy Required Components**: FramedButton + StatusBar (MenuBar not needed) (6.5 hours)
2. **Create Canonical Layout**: Exact structure from specification with dual editor (12 hours)
3. **Implement Status Symbols**: `?`/`-`/`o`/`!` progression with orchestrator lifecycle tracking (8 hours)
4. **Orchestrator Integration**: Button functionality + multi-folder selection (11 hours)
5. **Manual Verification**: Against canonical specification (14 hours)

## DNA Analysis Integration

The plans specifically address the DNA analysis findings:
- **PJLJ:handleok:16** - File browser OK button handler (complexity 16)
- **PJLJ:handleitemactiv:11** - File browser item activation handler (complexity 11)

These high-complexity functions are the root cause of non-functional buttons and dialogs.

## Success Criteria

**✅ ALREADY WORKING:**
- Interactive list components with selection and navigation
- Dual step/func editor with toggle functionality
- File browser with directory selection
- Action toolbars with command integration
- Parameter editors with static reflection

**🔧 CRITICAL FIXES DISCOVERED:**
- **Command signature mismatch** (toolbar ↔ commands incompatible)
- **Interface incompatibility** (archive components need major adaptation)
- **Status symbol inconsistencies** (multiple conflicting mappings across components)
- **File browser complexity** (DNA issues causing button failures)
- **End-to-end workflow breakage** (multiple failure points)

**🎯 FINAL GOAL:**
1. **All 21 required components present and working**
2. **Command system integration fixed** (signature compatibility)
3. **Archive components properly adapted** (interface compatibility)
4. **Centralized status symbol system** (consistent across all components)
5. **TUI matches the hierarchical design spec exactly**
6. **All buttons clickable and functional**
7. **File browser dialogs work for plate selection**
8. **DNA complexity issues resolved** (complexity <5)
9. **End-to-end workflows verified** (button click → completion)
10. **Integration tests pass for all workflows**

## Implementation Approach

**INTELLECTUAL HONESTY FRAMEWORK SUCCESS:**

**Before Deep Dive Investigation:**
- Assumed simple integration work (25 hours)
- Surface-level analysis of component compatibility
- Missed critical architectural incompatibilities
- Would have caused massive implementation failures

**After Deep Dive Investigation:**
- Discovered critical architectural issues (47 hours)
- Found command signature mismatches
- Revealed interface incompatibilities
- Identified status symbol inconsistencies across components

**KEY LESSONS:**
1. **Deep dive investigation prevents implementation disasters**
2. **Surface-level analysis is dangerously misleading**
3. **Intellectual honesty requires thorough verification**
4. **Proper investigation doubles time estimates but prevents failures**

**INTELLECTUAL HONESTY TRIUMPH:**
- Initial surface analysis: 25 hours of generic TUI work
- Deep dive investigation: 47 hours of architectural fixes
- Canonical specification review: 51.5 hours of exact implementation
- **Result**: Plans now implement the actual specification rather than assumptions!

This represents the **canonical implementation** of the OpenHCS TUI system as specified in tui_final.md.
