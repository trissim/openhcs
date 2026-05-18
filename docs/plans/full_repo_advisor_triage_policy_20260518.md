# Full Repo Advisor Triage Policy - 2026-05-18

## Purpose

The full OpenHCS advisor scan is now fast enough to run as a regular campaign
planning tool. It also emits many findings that are not all equal. This policy
defines how to sort scan output into architecture campaigns, cleanup batches,
and accepted noise.

## Scan Baseline

Command:

```bash
python -m nominal_refactor_advisor openhcs
```

Current successful scan:

- 1,140 findings
- 60 unique finding titles
- 61.032s total

Top high-volume categories:

- reused private helpers;
- readability compression;
- blank regions;
- string/numeric closed dispatch;
- attribute probing;
- manual public API surfaces;
- enum strategy ladders;
- repeated parameter families.

## Campaign-Grade Findings

Treat a finding as campaign-grade when it meets at least two criteria:

- crosses module boundaries;
- affects runtime/compiler/debug/GUI behavior;
- points to a named domain concept already present in OpenHCS;
- has multiple consumers or variants;
- replacing it would reduce future feature cost;
- can be verified with existing tests or small new characterization tests.

Examples from the current scan:

- `PipelineOrchestrator.execute_compiled_plate`
- runtime artifact query axis duplication;
- Napari streaming handler table duplication;
- AST validator family;
- preset pipeline spec duplication;
- Ashlar parameter record duplication;
- `PlateViewWidget.eventFilter`

## Cleanup-Grade Findings

Treat as cleanup-grade when the change is local and mechanically verifiable:

- overlong lines;
- blank-line runs;
- one-off string dispatch in terminal/UI compatibility code;
- small unreferenced private helpers;
- trivial forwarding wrappers where the public API is not involved.

Cleanup-grade work should be batched separately and should not interrupt a
campaign unless it blocks tests or readability in touched files.

## Known Noise / Accepted Findings

Move findings to a known-noise ledger only after inspecting the code seam.

Required ledger fields:

- stable id;
- evidence;
- reason not refactored;
- date;
- command used.

Existing CP ledger:

- `docs/plans/advisor_known_noise.md`

Do not silence advisor findings by fake inheritance, meaningless marker bases,
or generic predicate shells.

## Execution Rules

1. Start each campaign with characterization tests.
2. Refactor one semantic seam at a time.
3. Run focused advisor on touched files before full advisor.
4. Commit after a green focused test/advisor slice.
5. Run full unit tests before pushing.
6. Run full advisor after push-ready state to update campaign queue.

## Advisor Output Handling

Store full scan output outside the repo unless it is deliberately curated:

```bash
python -m nominal_refactor_advisor openhcs > /tmp/advisor_openhcs_YYYYMMDD.txt
```

Commit only:

- plan files;
- known-noise ledger entries;
- small summarized metrics;
- stable IDs needed for future tracking.

Do not commit multi-megabyte raw advisor output.

## Completion Criteria

- Each campaign has a plan file with evidence, target shape, phases, risks, and
  verification gates.
- Full advisor scan remains under 90 seconds on the current machine.
- Known-noise entries are explicit and reviewed.
- Cleanup-grade findings are not confused with architecture blockers.
