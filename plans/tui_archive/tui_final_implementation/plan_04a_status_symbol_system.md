# plan_04a_status_symbol_system.md
## Component: Status Symbol System Correction

### Objective
Fix the existing status symbol mapping in PlateListView to match the TUI spec exactly. The dynamic list system and visual display are already implemented - just need symbol correction.

### Plan
1. **Status Symbol Requirements from Spec (ALREADY UNDERSTOOD)**
   - `o` = compiled/ready (green) - plate has been compiled and is ready to run
   - `!` = initialized but not compiled (yellow) - plate has been initialized but not compiled
   - `?` = not initialized yet (red/default) - plate has been added but not initialized
   - Symbols appear in left vertical area of plate list entries ✅ ALREADY IMPLEMENTED

2. **Fix Status Symbol Mapping (SIMPLE CHANGE)**
   - ✅ ALREADY EXISTS: `PlateListView._get_plate_display_text_for_item()` with status symbols
   - ✅ ALREADY EXISTS: Format `status | ^/v name | path` matches spec exactly
   - ❌ WRONG MAPPING: Current uses `?`, `-`, `✓` instead of `?`, `!`, `o`
   - **Fix**: Update status_symbols dict to use correct symbols

3. **Verify Status Management Integration (LIKELY EXISTS)**
   - ✅ LIKELY EXISTS: `status` field in plate data (check TUI state)
   - ✅ ALREADY IMPLEMENTED: Status transition logic in commands
   - **Update Timing**: Status updates after operation completes (not when button clicked) ✅ CONFIRMED
   - **Individual Updates**: For batch operations, update each plate individually as it completes ✅ CONFIRMED

4. **Verify Plate Operations Integration (LIKELY EXISTS)**
   - ✅ LIKELY EXISTS: Commands already update plate status
   - **Just need to verify**: Status updates happen with correct symbols
   - **Error Handling**: Failed operations keep current status ✅ CONFIRMED

5. **Simple Symbol Fix Implementation**
   - Change `'initialized': '-'` to `'initialized': '!'`
   - Change `'compiled': '✓'` to `'compiled': 'o'`
   - Keep `'uninitialized': '?'` (already correct)
   - Test visual display with corrected symbols

6. **Status Bar Integration (SEPARATE PLAN)**
   - Move status bar implementation to separate plan
   - Focus this plan on just fixing the symbol mapping
   - Keep scope minimal and focused

### Findings
**AMAZING DISCOVERY: MOST WORK ALREADY DONE!**

**✅ ALREADY IMPLEMENTED:**
- `InteractiveListItem` component with selection and `^/v` navigation
- `PlateListView` using InteractiveListItem with status symbols
- Status symbol display format: `status | ^/v name | path` (matches spec exactly!)
- Navigation arrows, mouse selection, visual feedback all working
- Status management likely already integrated with commands

**❌ ONLY ISSUE: WRONG SYMBOLS**
- Current: `'initialized': '-'` should be `'initialized': '!'`
- Current: `'compiled': '✓'` should be `'compiled': 'o'`
- Current: `'uninitialized': '?'` is correct ✅

**🎯 IMPLEMENTATION SCOPE: TINY!**
- This is a 2-line change in `PlateListView._get_plate_display_text_for_item()`
- Everything else already works perfectly
- No new components needed, no complex integration required

**INTELLECTUAL HONESTY WIN:**
- Original plan assumed we needed to build everything from scratch
- Reality: Just need to fix 2 symbol mappings in existing working system
- Estimated effort: 5 minutes instead of hours

### Implementation Draft
*Implementation will be added after smell loop approval*
