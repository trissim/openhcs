# Registry Key Declaration Refactor Plan

Date: 2026-05-16

## Problem

The current advisor baseline still reports repeated hardcoded registry-key strings in registered policy families. These are real smell candidates, but they are also load-bearing.

`AutoRegisterMeta` consumes root declarations such as:

```python
__registry_key__ = "strategy_label"
strategy_label = None
```

Prior naive cleanup attempted to replace literals with constants. That was unsafe: it disrupted CellProfiler registry behavior and created a much larger advisor regression. The next pass must treat registry declarations as architecture, not string cleanup.

## Target

Introduce a nominal registry declaration substrate that keeps runtime behavior explicit while preventing duplicated low-level declaration conventions from spreading.

Acceptable end states:

- A small registered-family declaration model used by root classes and generated leaves.
- A metaclass-compatible class declaration helper that preserves the attributes `AutoRegisterMeta` needs.
- Advisor integration updated only if the runtime declaration model is actually cleaner.

Unacceptable end states:

- Constants that hide registry literals but do not change the abstraction.
- Per-file local helper functions.
- Compatibility wrappers that must remain forever.
- Stringly typed dict dispatch replacing nominal families.

## Current Hotspots

Targeted advisor findings currently include registry-key string repetition in:

- `CellProfilerModulePolicyRegistryDefaults`
- `RuntimePlaneAxisProjectionStrategy`
- `CellProfilerSpecialInputValueStrategy`
- `SourceSpatialDomainAdapter`
- `CellProfilerPayloadSpatialRankStrategy`
- `ObjectLabelDenseDataStrategy`

These families are not all equivalent. Some are CellProfiler module-policy registries, while others are runtime value/semantic registries. The solution should support both without flattening domain meaning.

## Investigation Steps

1. Inspect `metaclass_registry.AutoRegisterMeta` usage and local registered family conventions.
2. Identify whether root class declarations can share a typed declaration object while still exposing literal class attributes before metaclass construction.
3. Check whether existing generated leaf substrates already provide enough structure:
   - CellProfiler module policy leaf specs.
   - Runtime semantic/value generated families.
   - Measurement feature enum generation.
4. Add focused tests for the selected declaration route before broad migration.

## Implementation Stages

### Stage 1: Characterization

- Add or extend tests proving root registry keys, leaf registry keys, and registry lookup behavior for one CellProfiler family and one runtime family.
- Run advisor before and after to ensure no new metaclass-family findings are introduced.

### Stage 2: Shared Declaration Substrate

- Add one nominal declaration type only if it can be consumed by both runtime code and advisor expectations.
- Keep declaration data close to the class family root.
- Do not use private helper functions as the long-term API.

### Stage 3: Migrate One Family

- Start with the smallest runtime family with focused tests.
- Verify behavior and advisor output.
- Commit and push if count improves or code architecture is materially clearer without count regression.

### Stage 4: Expand Only On Evidence

- Migrate additional families only if Stage 3 succeeds.
- Stop migrating if the abstraction becomes more complex than the repeated root declarations it replaces.

## Verification

- Focused registry tests.
- Focused CellProfiler strategy tests.
- Full unit suite.
- Targeted advisor scan.

## Completion Criteria

- Registry declaration behavior is represented by a nominal shared abstraction or explicitly documented as intentionally literal for metaclass compatibility.
- No CellProfiler registry lookup regressions.
- No increase in advisor findings.
- Any remaining repeated registry-key findings are either removed or classified as analyzer noise with a concrete reason.

