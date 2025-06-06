# Investigation Roadmap - Channel Grouping Issue

## Current Status
**CRITICAL BUG**: `create_composite` with `variable_components=['channel']` not grouping w1 and w2 channels together.

**Expected**: Pattern `A01_s001_w{iii}_z001.tif` should match both w1 and w2, composite them, output only w1
**Actual**: Both w1 and w2 patterns processed separately, both reach Ashlar

## Investigation Strategy

### Phase 1: Pattern Discovery Analysis
**Goal**: Understand how `variable_components=['channel']` should work

#### Key Questions
1. How does pattern discovery generate `w{iii}` patterns?
2. Does `w{iii}` actually match `w1, w2` files?
3. Are patterns being grouped correctly by channel?
4. What does the EZStitcher documentation specify?

#### Files to Investigate
- `openhcs/formats/pattern/pattern_discovery.py`
- `openhcs/microscopes/imagexpress.py` (ImageXpress-specific patterns)
- `openhcs/formats/func_arg_prep.py` (pattern grouping)

#### Investigation Commands
```python
# Check pattern discovery output
patterns_by_well = context.microscope_handler.auto_detect_patterns(
    str(step_input_dir),
    context.filemanager,
    read_backend,
    well_filter=[well_id],
    extensions=DEFAULT_IMAGE_EXTENSIONS,
    group_by=group_by,  # Should be 'channel'
    variable_components=variable_components  # Should be ['channel']
)

# Check what patterns are actually generated
logger.debug(f"Patterns for {well_id}: {patterns_by_well[well_id]}")
```

### Phase 2: EZStitcher Documentation Review
**Reference**: https://ezstitcher.readthedocs.io/en/latest/concepts/step.html#group-by

#### Key Documentation Areas
- Group-by functionality explanation
- Variable components behavior
- Channel handling specifics
- Pattern matching rules

#### Questions for Documentation
1. How should `group_by='channel'` with `variable_components=['channel']` behave?
2. What's the difference between `group_by` and `variable_components`?
3. Are there examples of channel compositing workflows?

### Phase 3: Grouping Logic Analysis
**Goal**: Understand how patterns are grouped and functions are applied

#### Key Areas
- `prepare_patterns_and_functions` in `func_arg_prep.py`
- Pattern grouping by component value
- Function application to grouped patterns

#### Investigation Points
```python
# In prepare_patterns_and_functions
grouped_patterns, comp_to_funcs, comp_to_base_args = prepare_patterns_and_functions(
    patterns_by_well[well_id], 
    func_from_plan, 
    component=group_by  # 'channel'
)

# Check grouping results
logger.debug(f"Grouped patterns: {grouped_patterns}")
logger.debug(f"Component to functions: {comp_to_funcs}")
```

### Phase 4: Test Case Analysis
**Goal**: Create minimal test case to isolate the issue

#### Minimal Test Setup
1. Two files: `A01_s001_w1_z001.tif`, `A01_s001_w2_z001.tif`
2. Single step: `create_composite` with `variable_components=['channel']`
3. Expected: One output file (composited channels)
4. Debug pattern discovery and grouping at each step

#### Debug Logging Strategy
```python
# Add debug logging to key functions
logger.debug(f"🔍 Pattern discovery input: group_by={group_by}, variable_components={variable_components}")
logger.debug(f"🔍 Generated patterns: {patterns}")
logger.debug(f"🔍 Grouped patterns: {grouped_patterns}")
logger.debug(f"🔍 Files matched per pattern: {matching_files}")
```

## Hypothesis Testing

### Hypothesis 1: Pattern Discovery Issue
**Theory**: `w{iii}` pattern not matching `w1, w2` files correctly
**Test**: Check pattern generation logic in ImageXpress handler
**Evidence Needed**: Pattern strings generated vs files matched

### Hypothesis 2: Grouping Logic Issue  
**Theory**: Patterns are discovered correctly but not grouped by channel
**Test**: Check `prepare_patterns_and_functions` grouping behavior
**Evidence Needed**: Grouped patterns structure and function application

### Hypothesis 3: Variable Components Misunderstanding
**Theory**: `variable_components=['channel']` doesn't mean what we think
**Test**: Review EZStitcher docs and compare with implementation
**Evidence Needed**: Documentation vs actual behavior

### Hypothesis 4: Group-by vs Variable Components Confusion
**Theory**: `group_by='channel'` and `variable_components=['channel']` conflict
**Test**: Try different combinations and see behavior
**Evidence Needed**: Different parameter combinations and their effects

## Success Criteria

### Immediate Goal
Understand why `create_composite` processes w1 and w2 separately instead of together

### Long-term Goal  
Fix channel grouping so that:
1. `create_composite` receives both w1 and w2 for each site
2. Composites them into single channel output
3. Only w1 images reach `gpu_ashlar_align_cupy`
4. Pipeline flows correctly with chain breaker logic

## Investigation Tools

### Code Analysis
- Use `codebase-retrieval` to understand pattern discovery
- Add debug logging to trace pattern grouping
- Create minimal test cases

### Documentation Review
- Read EZStitcher docs thoroughly
- Compare documented behavior with implementation
- Look for examples and edge cases

### Empirical Testing
- Run pipeline with extensive debug logging
- Test different parameter combinations
- Isolate the issue with minimal examples

## Next Actions for New Thread

1. **Start with EZStitcher docs**: Understand intended group-by behavior
2. **Investigate pattern discovery**: How are `w{iii}` patterns generated?
3. **Trace grouping logic**: Follow patterns through grouping and function application
4. **Create minimal test**: Isolate channel grouping issue
5. **Fix root cause**: Implement proper channel grouping behavior

**Remember**: This is about understanding the architectural intent from EZStitcher heritage, not just patching symptoms.
