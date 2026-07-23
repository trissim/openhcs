# Remaining Cleanup And Advisor Noise Calibration - 2026-05-18

## Evidence

The refreshed full scan still contains large cleanup/noise buckets:

- 41 nonsemantic blank source region findings.
- 40 overcompressed source layout findings.
- Deprecated Textual TUI findings dominate early dispatch/reflective-access
  sections but are out of production refactor scope.
- Some metaclass/registry-key literal findings may be advisor-package issues:
  replacing `__registry_key__ = "backend_key"` with a local constant currently
  hides the stable key-axis signal from the advisor.
- Explicit nominal strategy subclasses sometimes look like metadata-only class
  families but are intentionally consumed through class identity and registry
  membership.

## Purpose

This plan prevents cleanup and noisy advisor findings from derailing priority
production work. It is not the primary execution lane. The user priority order
is CellProfiler, microscope/format, then active PyQt/runtime GUI.

## Target Shape

- Maintain `advisor_known_noise.md` for confirmed false positives or intended
  architecture.
- Fix advisor detectors when findings are systematically wrong or incentivize
  bad refactors.
- Batch formatting-only cleanup separately from architecture campaigns.
- Keep deprecated TUI excluded unless a deletion/deprecation plan owns it.

## Phases

1. Add known-noise entries only with evidence:
   - explicit nominal class family consumed by registry/class identity,
   - compatibility wrapper that is public API,
   - Numba kernel signature where request objects would hurt compilation.
2. File advisor fixes when detector behavior is wrong:
   - annotated assignments without values,
   - registry-key constants hiding key-axis proof,
   - recommendations that favor dynamic class materialization over explicit
     subclasses.
3. Batch pure source-layout cleanup after priority refactors, not before.
4. Re-run full scan after priority campaigns and compare active counts.

## Verification Gates

```bash
timeout 240 .venv/bin/python -m nominal_refactor_advisor openhcs > /tmp/advisor_openhcs_remaining_after_priority_pass.txt

.venv/bin/python -m pytest tests/unit -q
```

For advisor-package fixes, run that repo's test suite before committing.

## Non-Goals

- Do not spend priority-campaign time polishing deprecated Textual TUI.
- Do not classify risky production findings as noise just because they are
  large.
- Do not replace explicit nominal subclasses with dynamic `type(...)`
  materialization.
