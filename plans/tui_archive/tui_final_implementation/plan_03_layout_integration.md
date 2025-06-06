# plan_03a_three_bar_layout.md
## Component: 3-Bar Layout Structure Implementation

### Objective
Implement the exact 3-bar horizontal layout structure specified in the TUI spec, with proper visual hierarchy and component integration.

### Plan
1. **Create 3-Bar Layout Architecture**
   - **Bar 1 (Top)**: Global menu `[Global Settings] [Help] OpenHCS V1.0`
   - **Bar 2 (Titles)**: Section titles `"1 plate manager"` | `"2 Pipeline editor"`
   - **Bar 3 (Actions)**: Button toolbars for each section
   - **Main Content**: Two-pane area below bars
   - **Bottom**: Status bar with message display

2. **Implement MenuBar Component (Bar 1)**
   - Copy and adapt `menu_bar.py` from archive
   - Create `[Global Settings]` button → opens GlobalSettingsDialog
   - Create `[Help]` button → opens HelpDialog
   - Add centered "OpenHCS V1.0" title text
   - Ensure proper button spacing and alignment

3. **Create Section Title Bar (Bar 2)**
   - Design `SectionTitleBar` component
   - Left side: "1 plate manager" title
   - Right side: "2 Pipeline editor" title
   - Implement proper alignment and visual separation
   - Add visual indicators for active/inactive sections

4. **Create Action Button Bar (Bar 3)**
   - Left side: `PlateActionsToolbar` with `[add] [del] [edit] [init] [compile] [run]`
   - Right side: `PipelineActionsToolbar` with `[add] [del] [edit] [load] [save]`
   - Ensure proper alignment with section titles above
   - Implement visual separation between left and right toolbars

5. **Implement Main Layout Container**
   - Create `ThreeBarLayout` class to coordinate all bars
   - Use HSplit for vertical stacking of bars and content
   - Ensure fixed heights for bars, flexible height for content
   - **Focus Hierarchy**: Top bar is top level → two containers (each with title/action/list) → bottom status bar
   - **Navigation**: Arrow keys navigate list items and between UI elements
   - **Dual Editor Focus**: When dual editor opens, focus goes there; when closes, back to pipeline editor

### Findings
**TUI Specification Requirements:**
- 3-bar horizontal layout with specific content in each bar
- 2-pane main content area with plate manager (left) and pipeline editor (right)
- Dynamic pane replacement for step editing
- Bottom status bar with log drawer functionality
- Proper visual separation and styling

**Existing Components Available:**
- `PlateActionsToolbar` and `PipelineActionsToolbar` (from archive)
- `PlateListView` and `StepListView` (existing)
- `MenuBar` and `StatusBar` (from archive)
- `DualStepFuncEditor` (existing)

**Integration Challenges:**
- Coordinating multiple layout components
- Managing pane replacement during editing
- Ensuring proper focus management
- Maintaining responsive layout behavior

### Implementation Draft
*Implementation will be added after smell loop approval*
