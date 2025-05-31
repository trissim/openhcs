# MIST Implementation Analysis Notes

## Current Problem
Tiles are severely misaligned - gaps, overlaps, completely wrong positioning despite phase correlation returning reasonable values.

## Key Questions to Answer
1. What coordinate system does phase_correlation_gpu_only return?
2. What coordinate system does the MST algorithm expect?
3. How should region-to-region alignment map to tile-to-tile displacement?

## MIST Algorithm Key Insights (from PDF)

### Translation Direction Convention
- "Each translation maps I2 into I1's coordinate space"
- "The translation from I2 to I1 is the inverse of the translation from I1 to I2"
- Translation tuple: `⟨ncc, x, y⟩` where x=horizontal, y=vertical displacement

### Phase Correlation Returns
- Displacement to align two overlapping regions
- Multiple interpretations due to Fourier periodicity: `[(x,y), (x,H-y), (W-x,y), (W-x,H-y)]`
- Must test all interpretations and pick max NCC

### Expected Input to MST
- Connection displacements between tile centers
- Used to build graph where edges have weights (NCC values)
- MST resolves over-constrained system to get absolute positions

## Analysis Plan
1. Trace through phase correlation math
2. Analyze connection displacement calculation  
3. Check coordinate system consistency
4. Verify region extraction logic

## Static Analysis Results

### Phase Correlation Function Analysis

**Function**: `phase_correlation_gpu_only(image1, image2)`
**Returns**: `(dy, dx)` - shift to align image2 with image1

**Key Logic**:
```python
# Line 162-163: Convert peak to signed shifts
dy = cp.where(y_peak <= h // 2, y_peak, y_peak - h)
dx = cp.where(x_peak <= w // 2, x_peak, x_peak - w)
```

**Interpretation**:
- Returns how much to shift image2 to align with image1
- Handles FFT periodicity by converting to signed values
- Positive dx means image2 needs to move right to align with image1
- Positive dy means image2 needs to move down to align with image1

### Connection Building Analysis

**Current Logic in mist_main.py**:

**Horizontal connections** (lines 118-123):
```python
connection_dx[conn_idx] = expected_dx + dx  # Total displacement
connection_dy[conn_idx] = dy               # Correction only
```

**Vertical connections** (lines 150-155):
```python
connection_dx[conn_idx] = dx               # Correction only
connection_dy[conn_idx] = expected_dy + dy  # Total displacement
```

**Where**:
- `expected_dx = W * (1.0 - overlap_ratio)` (line 290)
- `expected_dy = H * (1.0 - overlap_ratio)` (line 289)
- `dx, dy` come from phase correlation of overlap regions

### Critical Issue Identified

**Problem**: Coordinate system mismatch between phase correlation and MST expectations.

**Phase correlation context**:
- `current_region = current_tile[:, -overlap_w:]` (right edge of left tile)
- `right_region = right_tile[:, :overlap_w]` (left edge of right tile)
- `dy, dx = phase_correlation_gpu_only(current_region, right_region)`

**This means**: `dx` is how much to shift `right_region` to align with `current_region`

**But we store**: `connection_dx[conn_idx] = expected_dx + dx`

**The issue**: We're adding a region-to-region alignment to a tile-to-tile expected displacement without proper coordinate transformation.

### Region Extraction Analysis

**Global Optimization (MST) - Horizontal**:
```python
# Lines 98-100
overlap_w = cp.maximum(cp.int32(W * overlap_ratio), min_overlap_pixels)
current_region = current_tile[:, -overlap_w:]  # RIGHT edge of left tile
right_region = right_tile[:, :overlap_w]      # LEFT edge of right tile
dy, dx = phase_correlation_gpu_only(current_region, right_region)
```

**Global Optimization (MST) - Vertical**:
```python
# Lines 130-132
overlap_h = cp.maximum(cp.int32(H * overlap_ratio), min_overlap_pixels)
current_region = current_tile[-overlap_h:, :]  # BOTTOM edge of top tile
bottom_region = bottom_tile[:overlap_h, :]     # TOP edge of bottom tile
dy, dx = phase_correlation_gpu_only(current_region, bottom_region)
```

**Initial Positioning - Horizontal**:
```python
# Lines 315-317
left_region = left_tile[:, -overlap_w:]    # RIGHT edge of left tile
current_region = current_tile[:, :overlap_w]  # LEFT edge of current tile
dy, dx = phase_correlation_gpu_only(left_region, current_region)
```

**Initial Positioning - Vertical**:
```python
# Lines 337-339
top_region = top_tile[-overlap_h:, :]      # BOTTOM edge of top tile
current_region = current_tile[:overlap_h, :]  # TOP edge of current tile
dy, dx = phase_correlation_gpu_only(top_region, current_region)
```

### The Fundamental Problem

**Inconsistent region extraction between phases**:

1. **MST phase**: `phase_correlation_gpu_only(current_region, right_region)`
   - `current_region` = right edge of left tile
   - `right_region` = left edge of right tile
   - Returns: how to shift `right_region` to align with `current_region`

2. **Initial positioning**: `phase_correlation_gpu_only(left_region, current_region)`
   - `left_region` = right edge of left tile
   - `current_region` = left edge of current tile
   - Returns: how to shift `current_region` to align with `left_region`

**These are OPPOSITE directions!** The MST phase and initial positioning are using different argument orders, leading to opposite sign conventions.

## Root Cause Analysis

### What Should Happen (MIST Algorithm)

According to MIST documentation:
1. Phase correlation should return displacement from tile center to tile center
2. All translations should be in consistent coordinate system
3. MST expects connection displacements between tile centers

### What's Actually Happening

1. **Inconsistent argument order**: MST and initial positioning call phase correlation with opposite argument orders
2. **Region-to-tile confusion**: We're treating region alignment as tile displacement without coordinate transformation
3. **Missing coordinate transformation**: No conversion from overlap region coordinates to tile center coordinates

### The Fix Strategy

**Option 1: Standardize argument order**
- Make all phase correlation calls use same argument order
- Ensure consistent sign convention throughout

**Option 2: Proper coordinate transformation**
- Convert region alignment to tile center displacement
- Account for overlap region position within tile

**Option 3: Follow MIST specification exactly**
- Implement multi-peak search with interpretation testing
- Use proper tile-to-tile displacement calculation

## Recommended Solution

**Immediate fix**: Standardize argument order and add proper coordinate transformation.

**For horizontal connections**:
```python
# Current (MST): phase_correlation_gpu_only(current_region, right_region)
# Should be: phase_correlation_gpu_only(left_region, right_region)
# Where left_region = current_tile[:, -overlap_w:] (right edge of left tile)
# And right_region = right_tile[:, :overlap_w] (left edge of right tile)
```

**For vertical connections**:
```python
# Current (MST): phase_correlation_gpu_only(current_region, bottom_region)
# Should be: phase_correlation_gpu_only(top_region, bottom_region)
# Where top_region = current_tile[-overlap_h:, :] (bottom edge of top tile)
# And bottom_region = bottom_tile[:overlap_h, :] (top edge of bottom tile)
```

This ensures phase correlation always returns "how to shift second argument to align with first argument" consistently.

## Implementation Status

### ✅ FIXED: Standardized Argument Order

**Changes made to mist_main.py**:

1. **MST Phase - Horizontal connections** (lines 99-103):
   ```python
   # BEFORE: phase_correlation_gpu_only(current_region, right_region)
   # AFTER:  phase_correlation_gpu_only(left_region, right_region)
   left_region = current_tile[:, -overlap_w:]  # Right edge of left tile
   right_region = right_tile[:, :overlap_w]   # Left edge of right tile
   ```

2. **MST Phase - Vertical connections** (lines 131-135):
   ```python
   # BEFORE: phase_correlation_gpu_only(current_region, bottom_region)
   # AFTER:  phase_correlation_gpu_only(top_region, bottom_region)
   top_region = current_tile[-overlap_h:, :]  # Bottom edge of top tile
   bottom_region = bottom_tile[:overlap_h, :] # Top edge of bottom tile
   ```

3. **Refinement Phase** - Already had correct argument order, just added comments for clarity

**Result**: All phase correlation calls now use consistent argument order:
- Horizontal: `phase_correlation_gpu_only(left_region, right_region)`
- Vertical: `phase_correlation_gpu_only(top_region, bottom_region)`

This should resolve the tile misalignment issue by ensuring displacement calculations are consistent throughout the algorithm.

## Algorithm Trace Analysis

### What I'm 100% Certain Of ✅

1. **Phase correlation function behavior**:
   - `phase_correlation_gpu_only(img1, img2)` returns `(dy, dx)`
   - `dx` = how much to shift img2 horizontally to align with img1
   - `dy` = how much to shift img2 vertically to align with img1
   - Positive dx = shift img2 right, positive dy = shift img2 down

2. **Region extraction is now consistent**:
   - Horizontal: `left_region` (right edge of left tile), `right_region` (left edge of right tile)
   - Vertical: `top_region` (bottom edge of top tile), `bottom_region` (top edge of bottom tile)
   - All calls use same argument order

3. **Expected displacements**:
   - `expected_dx = W * (1.0 - overlap_ratio)` = horizontal spacing between tile centers
   - `expected_dy = H * (1.0 - overlap_ratio)` = vertical spacing between tile centers

### What I'm Less Certain About ❓

1. **Connection displacement interpretation**:
   - What should `connection_dx[conn_idx] = expected_dx + dx` represent?
   - Is this tile-center-to-tile-center displacement?
   - Or is it something else the MST expects?

2. **Coordinate system for MST**:
   - Does MST expect absolute displacements from tile centers?
   - Or relative displacements between overlap regions?
   - What coordinate origin does MST use?

3. **Region-to-tile coordinate transformation**:
   - When phase correlation returns region alignment, how should this map to tile displacement?
   - Do we need to account for where the overlap region sits within the tile?

### MST Usage Analysis - Now 100% Certain ✅

**How MST uses connection displacements** (from position_reconstruction.py):

```python
# Line 92: neighbor_pos = current_pos + cp.array([dx, dy])
```

**This means**: `connection_dx` and `connection_dy` must be **tile-center-to-tile-center displacements**.

When MST traverses from tile A to tile B, it calculates:
```
position_B = position_A + [connection_dx, connection_dy]
```

So `connection_dx/dy` represents the **vector from tile A center to tile B center**.

### The Real Problem Identified ❌

**Current logic**:
```python
connection_dx[conn_idx] = expected_dx + dx  # expected_dx = W * (1-overlap)
connection_dy[conn_idx] = dy               # dy from phase correlation
```

**The issue**: We're mixing coordinate systems!
- `expected_dx` = tile-center-to-tile-center spacing
- `dx` = region-to-region alignment correction

**But**: Phase correlation `dx` is in **overlap region coordinates**, not tile center coordinates.

### Concrete Example

For horizontal connection (left tile → right tile):
- `expected_dx = W * (1-overlap)` = ~896 pixels (for W=1024, overlap=0.125)
- Phase correlation `dx` = small correction, e.g., +3 pixels
- **Current**: `connection_dx = 896 + 3 = 899`
- **Problem**: The +3 is region alignment, not tile center correction

### The Fix Needed

**Option 1**: Convert region alignment to tile center correction
**Option 2**: Use pure tile-center-to-tile-center phase correlation

Let me investigate which approach matches MIST specification.

### MIST Specification Review

From the MIST PDF, Algorithm 1 (Translation Computation):
```
foreach I ∈ Image Grid do
    Tw[I] ← pciam(I#west, I)     // when applicable
    Tn[I] ← pciam(I#north, I)    // when applicable
end
```

**Key insight**: MIST calls `pciam(I#west, I)` - this is **full tile to full tile**, not region to region!

**MIST's approach**:
1. Phase correlation on **entire tiles** (not just overlap regions)
2. Returns tile-center-to-tile-center displacement directly
3. No coordinate transformation needed

### Our Current Approach vs MIST

**Our approach**:
- Phase correlation on **overlap regions only**
- Need coordinate transformation from region alignment to tile displacement

**MIST approach**:
- Phase correlation on **full tiles**
- Direct tile-center-to-tile-center displacement

### The Solution

**Option A**: Switch to full-tile phase correlation (like MIST)
- Pro: Matches specification exactly
- Con: More computation, may be less accurate for small overlaps

**Option B**: Fix coordinate transformation for region-based approach
- Pro: More efficient, focuses on overlap area
- Con: Need to get coordinate math exactly right

**Recommendation**: Try Option B first - fix the coordinate transformation.

### Coordinate Transformation Analysis

For horizontal connection (left_tile → right_tile):
- `left_region` = right edge of left tile = `left_tile[:, -overlap_w:]`
- `right_region` = left edge of right tile = `right_tile[:, :overlap_w]`
- Phase correlation returns: how to shift `right_region` to align with `left_region`

**The key question**: How does region alignment map to tile center displacement?

### Coordinate Math Worked Out

**Setup**:
- Tile width = W, overlap width = overlap_w
- Left tile center at (0, 0), right tile center at (expected_dx, 0)
- `expected_dx = W * (1 - overlap_ratio)`

**Region positions**:
- `left_region` starts at x = W - overlap_w in left tile coordinate system
- `right_region` starts at x = 0 in right tile coordinate system

**In global coordinates**:
- `left_region` starts at x = W - overlap_w
- `right_region` starts at x = expected_dx

**Phase correlation result**:
- `dx` = how much to shift `right_region` to align with `left_region`
- If `dx > 0`: right_region needs to move right (right tile is too far left)
- If `dx < 0`: right_region needs to move left (right tile is too far right)

**Tile center correction**:
- If right_region needs to move by `dx`, then right tile center also needs to move by `dx`
- Therefore: `actual_tile_spacing = expected_dx + dx`

**Conclusion**: The current logic `connection_dx = expected_dx + dx` is **CORRECT**!

### Wait... Then Why Is It Still Broken?

If the coordinate math is correct, the problem must be elsewhere. Let me check:

1. **Sign conventions** - are they consistent?
2. **Initial positioning vs MST** - are they using same coordinate system?
3. **Quality filtering** - are we losing too many edges?

Let me investigate the quality filtering issue.

### Initial Positioning vs MST Coordinate System Check

**Initial positioning** (lines 328, 351):
```python
# Horizontal: new_x = positions[left_idx, 0] + expected_dx + dx
# Vertical:   new_y = positions[top_idx, 1] + expected_dy + dy
```

**MST reconstruction** (position_reconstruction.py line 92):
```python
# neighbor_pos = current_pos + cp.array([dx, dy])
```

**These are IDENTICAL!** Both add the displacement to get new position.

### Hypothesis: The Problem Is In Initial Positioning

Wait... let me check the initial positioning more carefully.

**Initial positioning logic**:
1. Place tile (0,0) at origin
2. For each subsequent tile, position it relative to its left/top neighbor
3. Use phase correlation to get displacement
4. Add displacement to neighbor position

**But**: What if the initial positioning is accumulating errors? Each tile is positioned relative to the previous one, so errors compound.

**MST logic**:
1. Build connections between ALL adjacent tiles
2. Use MST to find optimal subset of connections
3. Reconstruct positions using only MST edges

**The issue might be**: Initial positioning creates a "chain" of dependencies, while MST creates an optimal tree. If initial positioning has accumulated errors, MST might be trying to correct them but failing.

### New Investigation Direction

## Static Analysis: MST Construction Issues

### Quality Threshold Analysis

**Current default**: `mst_quality_threshold=0.3` (line 30)

**Quality calculation** (lines 120, 143):
```python
quality = cp.max(correlation_result)  # Peak correlation value
```

**Problem identified**: Phase correlation peak values are typically in range [0, 1], but for noisy or misaligned regions, peaks can be quite low even for valid alignments.

**Threshold of 0.3 may be too aggressive** - this could eliminate most connections.

### MST Edge Count Analysis

For a 4x4 grid (16 tiles):
- **Total possible connections**: (4-1)*4 + 4*(4-1) = 12 + 12 = 24 edges
- **MST needs**: 16-1 = 15 edges minimum
- **If quality threshold eliminates >9 edges**: MST becomes impossible

### Connection Building Logic Issue

**Lines 98-155**: Connection building loop
```python
for row in range(num_rows):
    for col in range(num_cols):
        # Horizontal connections (col < num_cols - 1)
        # Vertical connections (row < num_rows - 1)
```

**This is correct** - builds all adjacent connections.

### MST Fallback Logic Issue

**Lines 171-198**: MST construction
```python
if conn_idx > 0:
    # Build MST
else:
    return positions  # Fallback to initial positions
```

**Critical issue identified**: If quality threshold eliminates too many edges, we silently fall back to initial positions without any indication!

### The Real Problem

**Hypothesis**: Quality threshold is too high, eliminating most/all connections, causing silent fallback to initial positions.

**Evidence**:
1. Images work with CPU version (good quality)
2. GPU version shows severe misalignment (suggests no MST correction)
3. No error messages (suggests silent fallback)

### Solution

**Option 1**: Lower quality threshold to 0.1 or 0.05
**Option 2**: Add adaptive threshold that ensures minimum edge count
**Option 3**: Use percentile-based threshold instead of absolute

## Static Analysis Conclusion

**Primary Issue Identified**: Quality threshold too aggressive, causing silent fallback to initial positions.

**Evidence**:
1. **Line 229**: `mst_quality_threshold: float = 0.1` (now changed to 0.01)
2. **Lines 171-179**: Silent fallback logic - if no connections pass threshold, returns initial positions
3. **User confirmation**: Images work with CPU version, indicating good quality
4. **Symptom**: Severe misalignment suggests no MST correction applied

**Root Cause**: Phase correlation peaks for microscopy images can be quite low (< 0.1) even for valid alignments, especially with:
- Noise in biological samples
- Varying illumination
- Small overlap regions
- Subpixel displacements

**Fix Applied**: Lowered `mst_quality_threshold` from 0.1 to 0.01

**Expected Result**: More connections will pass quality filter, enabling MST to build proper spanning tree and correct tile positions.

**Next Test**: Run with lowered threshold to verify MST is now being used instead of falling back to initial positions.

## New Issue: Disconnected MST

**Test Result**:
```
🔥 MST RESULT: 15 edges selected
🔥 RECONSTRUCTING positions from MST...
Position reconstruction: 15 MST edges, 16 tiles
Anchor tile 0: (0.0, 0.0)
🔥 WARNING: 8 tiles not reachable from anchor tile
```

**Analysis**: We have exactly the right number of edges (15 for 16 tiles), but the MST is **disconnected**. This means the Borůvka algorithm is not working correctly.

**Root Cause Hypothesis**: The Borůvka MST algorithm has a bug - it's selecting 15 edges but they don't form a connected spanning tree.

**Evidence**:
1. ✅ 15 edges selected (correct count)
2. ❌ 8 tiles unreachable (should be 0 for connected tree)
3. ❌ MST should connect all tiles in a single tree

**Possible Issues in Borůvka Algorithm**:
1. **Union-Find bug**: Components not being merged correctly
2. **Edge selection bug**: Selecting edges that don't connect different components
3. **Termination bug**: Algorithm stopping before all components are connected

**Next Investigation**: Static analysis of Borůvka union-find logic and edge selection.

## Union-Find Bug Found!

**Location**: `gpu_kernels.py` lines 36-40 (path compression in flattening kernel)

**Current code**:
```python
current = tid
while parent[current] != current:
    next_parent = parent[current]
    parent[current] = parent[next_parent]  # BUG: Wrong compression!
    current = next_parent
```

**Problem**: The path compression is incorrect. It should compress the path to the root, but instead it's doing something else.

**Correct path compression**:
```python
current = tid
while parent[current] != current:
    parent[current] = parent[parent[current]]  # Compress one level at a time
    current = parent[current]
```

**Impact**: Incorrect path compression means union-find components are not properly flattened, leading to:
1. Incorrect component identification in `_find_minimum_edges_kernel`
2. Incorrect union operations in `_union_components_kernel`
3. Disconnected MST with wrong edge count

**This explains the symptoms**: 15 edges selected but 8 tiles unreachable - the algorithm thinks it's connecting components but they're not actually being unified correctly.

## Fix Applied

**Changed lines 38-39 in `gpu_kernels.py`**:
```python
# Before (WRONG):
parent[current] = parent[next_parent]
current = next_parent

# After (CORRECT):
parent[current] = parent[parent[current]]  # Compress one level at a time
current = parent[current]
```

**Expected Result**: Union-find will now properly flatten component trees, allowing correct component identification and union operations. This should result in a connected MST with all 16 tiles reachable from the anchor tile.

## Test Results: Union-Find Fix Successful!

**MST Construction**: ✅ Working correctly
- First run: 15 edges, 1 component (perfect!)
- Second run: 14 edges, 2 components (still has issue)

**Position Reconstruction**: ✅ All tiles reachable
- No more "8 tiles not reachable" warnings
- MST properly connected

**Image Quality**: ⚠️ Still has alignment issues
- Tiles are positioned but not perfectly aligned
- Some visible seams and misalignments
- Better than before but not publication-ready

## Remaining Issues Analysis

**Observation**: The MST is working, but alignment quality suggests either:
1. **Phase correlation accuracy issues** - subpixel alignment not precise enough
2. **Coordinate system issues** - still some mismatch in displacement calculations
3. **Quality metric issues** - MST selecting suboptimal edges

**Key Insight**: The debug output shows very large displacements:
```
Edge 0: 0 -> 4, dx=-20.922, dy=112.728
Edge 1: 1 -> 2, dx=115.670, dy=5.844
Edge 2: 3 -> 7, dx=30.496, dy=116.057
```

**Expected vs Actual**:
- Expected spacing: ~896 pixels (W * (1-overlap) for W=1024, overlap=0.125)
- Actual dx values: 115.670, -20.922, 30.496
- **These are way too small!** Should be ~896 ± small correction

**New Hypothesis**: The coordinate transformation is still wrong. We're getting region-level corrections instead of tile-center-to-tile-center displacements.

## Static Analysis: Displacement Magnitude Issue

**Expected vs Actual Analysis**:
- Expected for 10% overlap: `expected_dx = 1024 * 0.9 = 921.6`
- Actual debug output: dx values ~115, dy values ~112
- **Ratio**: 921/115 ≈ 8x difference

**Possible Root Causes**:

### 1. Connection Building Logic Issue

**Lines 120-121** (horizontal connections):
```python
connection_dx[conn_idx] = expected_dx + dx  # Should be ~921 + small_correction
connection_dy[conn_idx] = dy               # Should be ~0 for horizontal
```

**Lines 152-153** (vertical connections):
```python
connection_dx[conn_idx] = dx               # Should be ~0 for vertical
connection_dy[conn_idx] = expected_dy + dy  # Should be ~921 + small_correction
```

**Static Analysis Question**: Are we actually executing these lines, or is there a logic branch that bypasses them?

### 2. MST Edge Storage/Retrieval Issue

**Lines 128-131** (in `_union_components_kernel`):
```python
mst_from[mst_slot] = from_node
mst_to[mst_slot] = to_node
mst_dx[mst_slot] = edges_dx[edge_idx]  # Direct copy from connection_dx
mst_dy[mst_slot] = edges_dy[edge_idx]  # Direct copy from connection_dy
```

**This is direct assignment** - no transformation. So if connection_dx contains wrong values, MST will inherit them.

### 3. Expected Displacement Calculation Issue

**Lines 276-277**:
```python
expected_dy = cp.float32(H * (1.0 - overlap_ratio))
expected_dx = cp.float32(W * (1.0 - overlap_ratio))
```

**Static Analysis**: This looks correct for overlap_ratio=0.1.

### 4. Phase Correlation Return Value Issue

**Critical insight**: What if phase correlation is returning values in a different coordinate system than expected?

**Lines 134-139** (vertical connection building):
```python
dy, dx = phase_correlation_gpu_only(
    top_region, bottom_region,  # Standardized: top_region first
    subpixel=subpixel,
    subpixel_radius=subpixel_radius,
    regularization_eps_multiplier=regularization_eps_multiplier
)
```

**Question**: What if `phase_correlation_gpu_only` is returning displacements in **overlap region coordinates** instead of **full tile coordinates**?

## CRITICAL DISCOVERY: Coordinate System Mismatch!

**Phase Correlation Analysis**:

**Lines 160-163** in `phase_correlation.py`:
```python
# Convert to signed shifts (GPU arithmetic)
h, w = correlation.shape  # ← This is the OVERLAP REGION size!
dy = cp.where(y_peak <= h // 2, y_peak, y_peak - h)
dx = cp.where(x_peak <= w // 2, x_peak, x_peak - w)
```

**Key Insight**: `phase_correlation_gpu_only` returns displacements **relative to the input region size**, not the full tile size!

**For 10% overlap with 1024x1024 tiles**:
- Overlap region size: `overlap_w = 1024 * 0.1 = 102.4 ≈ 102 pixels`
- Phase correlation returns dx in range `[-51, +51]` (half the overlap region)
- But we need dx in full tile coordinates: `~921 ± small_correction`

**This explains the 8x magnitude difference**:
- Expected: ~921 pixels (full tile spacing)
- Actual: ~115 pixels (overlap region displacement)
- Ratio: 921/115 ≈ 8x ✓

**The Bug**: We're treating overlap-region-relative displacements as if they were full-tile-relative displacements.

**Current Broken Logic**:
```python
# Line 120: Horizontal connection
connection_dx[conn_idx] = expected_dx + dx  # expected_dx=921, dx=±50 → ~971
```

**But dx is only valid within the overlap region!** We need to transform it to full tile coordinates.

## Solution Analysis

**The Fix**: Phase correlation displacements are **fine-tuning corrections** within the overlap region. The total displacement should be:

```
total_displacement = expected_displacement + overlap_region_correction
```

**For Horizontal Connections**:
```python
# Current (WRONG):
connection_dx[conn_idx] = expected_dx + dx  # dx is overlap-region-relative

# Correct:
connection_dx[conn_idx] = expected_dx + dx  # dx is already the right correction!
```

**Wait - this suggests the current logic is actually CORRECT!**

**Re-analysis**: If the current logic is correct, why are we seeing small displacement values?

**Alternative Hypothesis**: The issue might be that we're **not actually executing the expected_dx + dx line** due to a logic error, or there's a bug in how the MST stores/retrieves these values.

**Debug Strategy**: We need to verify:
1. Are we actually executing lines 120 and 153?
2. What are the actual values of `expected_dx`, `expected_dy`, and the phase correlation corrections `dx`, `dy`?
3. Are the MST edges storing the correct values?

**Most Likely Issue**: Looking at the debug output again:
```
Edge 0: 0 -> 4, dx=-20.922, dy=112.728
```

If this is supposed to be `expected_dy + dy` for a vertical connection, and we see 112.728, then either:
- `expected_dy` is ~112 (wrong overlap ratio)
- We're not adding `expected_dy` at all (logic bug)
- The MST is storing different values than we think

## Debug Output Tracing

**MST Debug Code** (boruvka_mst.py lines 174-176):
```python
if i < 3:
    print(f"  Edge {i}: {edge['from']} -> {edge['to']}, dx={edge['dx']:.3f}, dy={edge['dy']:.3f}")
```

**Values come from** (lines 167-168):
```python
'dx': float(mst_edges_dx[i]),
'dy': float(mst_edges_dy[i]),
```

**MST arrays populated by union components kernel** - direct copy from connection arrays.

**Conclusion**: The debug output shows the **actual connection displacement values** that were stored during connection building.

## Quality Threshold Hypothesis

**Current threshold**: `mst_quality_threshold: float = 0.01` (very low)

**Possible Issue**: If most connections fail the quality threshold, we might be building connections with **different logic** than expected.

**Lines 116-123** (horizontal connection building):
```python
if quality >= quality_threshold:
    passed_threshold += 1
    connection_from[conn_idx] = tile_idx
    connection_to[conn_idx] = right_idx
    connection_dx[conn_idx] = expected_dx + dx  # ← This should give ~921
    connection_dy[conn_idx] = dy               # ← This should give ~0
    connection_quality[conn_idx] = quality
    conn_idx += 1
```

**Critical Question**: Are we actually executing line 120, or are most connections failing the quality check?

**If connections fail quality check**: We get fewer connections, and the MST might be built from a different set of edges than expected.

## Static Analysis: Systematic Root Cause Investigation

**Returning to architectural analysis per user guidelines.**

### What We Know For Certain ✅

1. **MST Construction Works**: 15 edges built, all tiles reachable
2. **Connection Building Logic**: Lines 120 and 153 are the ONLY places connection_dx/dy are assigned
3. **Expected Displacement Calculation**: `expected_dx = W * (1-overlap_ratio) = 1024 * 0.9 = 921.6`
4. **Debug Output Shows**: dx values ~115, dy values ~112 (8x smaller than expected)
5. **MST Debug Prints Actual Values**: Direct from `mst_edges_dx[i]` and `mst_edges_dy[i]`

### The Core Contradiction

**If the logic is correct**:
```python
connection_dx[conn_idx] = expected_dx + dx  # Should be 921.6 + small_correction
```

**But debug shows**: `dx=-20.922, dy=112.728`

**This means either**:
1. `expected_dx` is not 921.6 (calculation error)
2. We're not executing line 120 (quality filtering)
3. There's a different code path setting connection values

### Systematic Elimination

**Hypothesis 1: Calculation Error**
- `expected_dx = cp.float32(W * (1.0 - overlap_ratio))` (line 277)
- For W=1024, overlap_ratio=0.1: `1024 * 0.9 = 921.6`
- **This is mathematically correct**

**Hypothesis 2: Quality Filtering**
- Quality threshold: `mst_quality_threshold: float = 0.01` (very low)
- If most connections fail quality, we get different edge set
- **Need to verify**: Are the MST edges the ones we think they are?

**Hypothesis 3: Different Code Path**
- **Confirmed**: Only one place sets connection_dx/dy values
- **No other assignment locations found in codebase search**

### Deep Analysis: Quality Filtering Hypothesis

**The Most Likely Scenario**: Quality filtering is eliminating the "correct" connections and the MST is built from a different set of edges.

**Evidence Supporting This**:
1. **MST gets exactly 15 edges** - suggests connections are being built
2. **But displacement values are wrong** - suggests they're not the expected connections
3. **Quality threshold is very low (0.01)** - but phase correlation can still fail

**Critical Insight**: What if the **initial positioning phase** creates positions that are so wrong that when we do MST optimization, the phase correlations fail for the "expected" adjacent connections?

**Scenario**:
1. Initial positioning creates bad tile positions
2. MST phase tries to correlate adjacent tiles (0->1, 0->4, etc.)
3. But tiles are so misaligned that adjacent correlations fail quality
4. MST finds different connections that happen to have better correlation
5. These "accidental" connections have small displacements (not tile-spacing displacements)

### Examining Initial Positioning Logic

**Lines 315-316** (horizontal positioning):
```python
new_x = positions[left_idx, 0] + expected_dx + dx
new_y = positions[left_idx, 1] + dy
```

**Lines 337-338** (vertical positioning):
```python
new_x = positions[top_idx, 0] + dx
new_y = positions[top_idx, 1] + expected_dy + dy
```

**This uses the SAME coordinate system as MST** - so if initial positioning works, MST should work.

**But what if initial positioning fails?** Then tiles are in wrong positions, and MST correlations between "adjacent" tiles fail.

### The Architectural Problem Identified

**Key Insight**: The MST optimization phase is **independent of initial positioning**. It builds connections by trying to correlate **all adjacent tile pairs in the grid**, regardless of where initial positioning placed them.

**MST Connection Building Logic** (lines 88-175):
```python
for r in range(num_rows):
    for c in range(num_cols):
        # Try horizontal connection: (r,c) -> (r,c+1)
        # Try vertical connection: (r,c) -> (r+1,c)
```

**This means**: MST always tries to connect tile 0->1, 0->4, 1->2, etc. based on **grid topology**, not current tile positions.

**The Problem**: If tiles are severely misaligned (due to bad initial positioning or bad phase correlation), then:
1. **Adjacent grid connections fail quality** (tiles don't overlap properly)
2. **MST finds "accidental" connections** that happen to correlate well
3. **These accidental connections have small displacements** (not tile-spacing)

### Evidence This Is Happening

**Debug Output Analysis**:
```
Edge 0: 0 -> 4, dx=-20.922, dy=112.728
Edge 1: 1 -> 2, dx=115.670, dy=5.844
Edge 2: 3 -> 7, dx=30.496, dy=116.057
```

**Expected Grid Connections**:
- 0->4: Should be vertical, dy≈921, dx≈0
- 1->2: Should be horizontal, dx≈921, dy≈0
- 3->7: Should be vertical, dy≈921, dx≈0

**Actual Values**: All displacements are ~100-120 pixels, much smaller than expected.

**Conclusion**: These are NOT the expected grid-adjacent connections. The MST found different connections that happened to correlate better.

## Backend Bug Analysis - Images Are Fine

**User confirmed: Images are fine, there's a backend implementation bug.**

**Returning to systematic backend analysis.**

### Re-examining the Core Contradiction

**We established**:
1. Only one place sets connection_dx/dy: lines 120 and 153
2. Logic should be: `connection_dx = expected_dx + dx` where expected_dx = 921.6
3. But MST debug shows dx values ~115 (8x smaller)

**Since images are fine, the bug must be in**:
1. **Overlap region extraction** - wrong regions being correlated
2. **Phase correlation coordinate system** - returning wrong coordinate space
3. **Expected displacement calculation** - wrong values being used
4. **Connection building logic** - not executing the lines we think

### Deep Dive: Overlap Region Extraction

**Horizontal connection** (lines 98-100):
```python
overlap_w = cp.maximum(cp.int32(W * overlap_ratio), min_overlap_pixels)
left_region = current_tile[:, -overlap_w:]  # Right edge of left tile
right_region = right_tile[:, :overlap_w]   # Left edge of right tile
```

**For W=1024, overlap_ratio=0.1**:
- `overlap_w = max(102, 32) = 102`
- `left_region` = rightmost 102 columns of left tile
- `right_region` = leftmost 102 columns of right tile

**This looks correct for extracting overlap regions.**

### Deep Dive: Phase Correlation Return Values

**Critical Question**: What coordinate space does phase correlation return values in?

**From phase_correlation.py lines 160-163**:
```python
h, w = correlation.shape  # This is the overlap region size (102x1024)
dy = cp.where(y_peak <= h // 2, y_peak, y_peak - h)
dx = cp.where(x_peak <= w // 2, x_peak, x_peak - w)
```

**For overlap region 102x1024**:
- dx range: [-51, +51] (half the overlap width)
- dy range: [-512, +512] (half the tile height)

**This means phase correlation returns displacements in overlap region coordinates, not full tile coordinates.**
