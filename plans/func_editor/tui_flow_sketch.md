# Function Pattern Editor TUI Flow

## Frame 1: Initial State (List Pattern - "Apply to ALL")

```
┌─ Function Pattern Editor ─────────────────────────────────────┐
│                                                               │
│ Pattern Type: List (Apply to ALL experimental components)     │
│                                                               │
│ [ Apply per Component ]                                       │
│                                                               │
│ Functions (applied to ALL channels/z-slices/sites):           │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ [↑] [↓] [×]  denoise                    (no args)         │ │
│ │ [↑] [↓] [×]  stack_percentile_normalize  low_percentile=0.5│ │
│ │                                         high_percentile=99.5│ │
│ │ [↑] [↓] [×]  n2v2_denoise_torch         (no args)         │ │
│ │                                                           │ │
│ │ [ Add Function ]                                          │ │
│ └───────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

Semantic Meaning: 
- Same processing workflow for ALL experimental components
- If data has channels 1, 2, 3 → all get [denoise, normalize, n2v2]
```

## Frame 2: User Clicks "Apply per Component"

```
┌─ Function Pattern Editor ─────────────────────────────────────┐
│                                                               │
│ Pattern Type: Dict (Apply per experimental component)         │
│                                                               │
│ Component: [ No components yet - click + to add ]  [ + ]      │
│                                                               │
│ Functions for selected component:                             │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │                                                           │ │
│ │           No component selected                           │ │
│ │                                                           │ │
│ │                                                           │ │
│ └───────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

Semantic Meaning:
- Ready for component-specific experimental workflows
- Empty dict - user needs to add experimental component identifiers
```

## Frame 3: User Clicks "+" to Add Component

```
┌─ Add Component Identifier ────────────────────────────────────┐
│                                                               │
│ Enter experimental component ID (e.g., '1', '2', 'DAPI', 'GFP'): │
│                                                               │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 1                                                         │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                               │
│                           [ OK ]  [ Cancel ]                 │
└───────────────────────────────────────────────────────────────┘

Semantic Meaning:
- User enters actual experimental component identifier
- "1" = Channel 1 from microscope data (e.g., DAPI channel)
- Must match actual component values in experimental data
```

## Frame 4: Component "1" Added - Building Channel 1 Workflow

```
┌─ Function Pattern Editor ─────────────────────────────────────┐
│                                                               │
│ Pattern Type: Dict (Apply per experimental component)         │
│                                                               │
│ Component: [ Component: 1 ▼ ]                        [ + ]   │
│                                                               │
│ Functions for Component: 1 (Channel 1 - DAPI):               │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ [↑] [↓] [×]  denoise                    (no args)         │ │
│ │ [↑] [↓] [×]  enhance_nuclei             gain=2.0          │ │
│ │                                                           │ │
│ │ [ Add Function ]                                          │ │
│ └───────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

Semantic Meaning:
- Building DAPI channel-specific processing workflow
- Channel 1 gets: denoise → enhance_nuclei (nuclei enhancement)
- Different from other channels (GFP, brightfield, etc.)
```

## Frame 5: User Adds Component "2" for GFP Channel

```
┌─ Function Pattern Editor ─────────────────────────────────────┐
│                                                               │
│ Pattern Type: Dict (Apply per experimental component)         │
│                                                               │
│ Component: [ Component: 2 ▼ ]                        [ + ]   │
│            [ Component: 1   ]                                │
│            [ Component: 2   ] ← Selected                     │
│                                                               │
│ Functions for Component: 2 (Channel 2 - GFP):                │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ [↑] [↓] [×]  denoise                    (no args)         │ │
│ │ [↑] [↓] [×]  enhance_protein            contrast=1.5       │ │
│ │ [↑] [↓] [×]  measure_intensity          (no args)         │ │
│ │                                                           │ │
│ │ [ Add Function ]                                          │ │
│ └───────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘

Semantic Meaning:
- Building GFP channel-specific processing workflow  
- Channel 2 gets: denoise → enhance_protein → measure_intensity
- Completely different workflow from Channel 1 (DAPI)
```

## Frame 6: Final Experimental Workflow (Backend Result)

```
Generated Function Pattern (Dict):
{
    "1": [denoise, (enhance_nuclei, {"gain": 2.0})],
    "2": [denoise, (enhance_protein, {"contrast": 1.5}), measure_intensity]
}

Backend Execution:
- Channel 1 files (A01_s001_w1_z001.tif, etc.) → denoise → enhance_nuclei
- Channel 2 files (A01_s001_w2_z001.tif, etc.) → denoise → enhance_protein → measure_intensity

Experimental Workflow:
- DAPI channel: Optimized for nuclei detection and enhancement
- GFP channel: Optimized for protein localization and quantification
- Each channel gets component-specific processing pipeline
```

## Key Semantic Differences

### **List Pattern ("Apply to ALL")**
```python
func_pattern = [denoise, normalize]
# Same workflow for ALL experimental components
# All channels get identical processing
```

### **Dict Pattern ("Apply per Component")**  
```python
func_pattern = {
    "1": [denoise, enhance_nuclei],      # DAPI-specific workflow
    "2": [denoise, enhance_protein],     # GFP-specific workflow  
    "3": [denoise, enhance_brightfield]  # Brightfield-specific workflow
}
# Different workflow per experimental component
# Each channel gets optimized processing
```

## User Journey Summary

1. **Start**: List pattern (same processing for all)
2. **Click "Apply per Component"**: Convert to dict pattern
3. **Add component identifiers**: "1", "2", "3" (actual channel IDs)
4. **Design workflows**: Different functions per component
5. **Result**: Component-specific experimental processing workflows

**The "Apply per Component" button transforms the editor from "one-size-fits-all" to "component-specific experimental workflow design."**
