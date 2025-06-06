# Technical State Snapshot - OpenHCS Debug Session

## Current Test Case
**Test**: `test_main_3d[ImageXpress]` in `tests/integration/test_main.py`
**Data**: ImageXpress microscope data with wells A01, D02
**Channels**: w1, w2 (should be composited together)
**Z-stacks**: z001, z002 (should be flattened)

## Pipeline Definition (from test_main.py)
```python
Step(func=[
    create_projection,
    (sharpen, {'amount': 1.5}),
    (stack_percentile_normalize, {'low_percentile': 0.5, 'high_percentile': 99.5}),
    stack_equalize_histogram
]),
Step(func=create_composite, variable_components=['channel']),  # ← ISSUE HERE
Step(func=gpu_ashlar_align_cupy),  # @chain_breaker
Step(func=n2v2_denoise_torch),
Step(func=basic_flatfield_correction_cupy),
Step(func=self_supervised_3d_deconvolution),
Step(func=assemble_stack_cupy)
```

## Error Patterns Observed

### Pattern Matching Working
✅ **Pattern discovery is working**: Function receives all 16 sites correctly
✅ **Context injection working**: Grid dimensions from HTD file: 4x4  
✅ **GPU libraries working**: All GPU libraries loaded and executing

### The Core Problem
❌ **Both w1 AND w2 reaching Ashlar**: Should only be w1 after channel compositing
❌ **Directory collision**: `workspace_outputs_outputs_outputs_outputs` (FIXED)
❌ **Memory collision**: Multiple wells saving to same filename (FIXED)

## Log Evidence
```
DEBUG Processing pattern group A01_s{iii}_w1_z001.tif for well A01, component 1
INFO Using grid dimensions from HTD file: 4x4
DEBUG Saving special output 'positions' to VFS path '...positions.pkl' (memory backend)
ERROR Parent path does not exist: ...positions.pkl (FIXED)
```

## Key Functions and Decorators

### gpu_ashlar_align_cupy
```python
@chain_breaker
@special_outputs("positions")
@cupy_func
def gpu_ashlar_align_cupy(tiles, **kwargs) -> Tuple[cp.ndarray, cp.ndarray]:
```

### create_composite (needs investigation)
```python
# Should flatten channels w1+w2 → w1 only
# variable_components=['channel'] should group w{iii} patterns
```

## Memory Architecture Details

### Thread Safety
- 1 thread = 1 well (A01, D02, etc.)
- Shared memory backend across threads
- Special outputs need well_id prefix: `A01_positions.pkl`, `D02_positions.pkl`

### Backend Usage
- **FileManager operations**: Always use `Backend.MEMORY.value` as last positional arg
- **Stack utils**: Use `MemoryType` for numpy/cupy/torch conversion
- **Compiler**: Use `MemoryType` for memory contract validation

## Files and Line Numbers

### function_step.py Key Areas
- Line 30-38: `_execute_function_core` signature (MODIFIED)
- Line 95-105: Special output saving with well_id prefix (MODIFIED)  
- Line 224-250: Image saving logic with flattening support (MODIFIED)

### path_planner.py Key Areas
- Line 154-173: Output directory logic (MODIFIED - no more nested _outputs)
- Line 241-257: Chain breaker handling (WORKING)

## Pattern Discovery Investigation Needed

### Questions to Answer
1. How does `variable_components=['channel']` create patterns?
2. Does `w{iii}` actually match `w1, w2`?
3. Are channels being grouped or processed separately?
4. What does the EZStitcher documentation say about this?

### Files to Investigate
- `openhcs/formats/pattern/pattern_discovery.py`
- `openhcs/formats/func_arg_prep.py`
- `openhcs/microscopes/imagexpress.py` (pattern generation)

## User Preferences (Important!)
- **Architecture over patches**: Fix root causes, not symptoms
- **Intellectual honesty**: Thorough investigation before implementation  
- **Clean code**: Low entropy, functional patterns, minimal bloat
- **Methodical approach**: "When you slow down, you go faster"
- **Collaborative debugging**: User helps reset mindset when rushing

## EZStitcher Heritage
- OpenHCS evolved from EZStitcher ~1.5 months ago
- EZStitcher docs contain group-by functionality explanation
- This bug likely touches core architectural concepts from that evolution
- Reference: https://ezstitcher.readthedocs.io/en/latest/concepts/step.html#group-by
