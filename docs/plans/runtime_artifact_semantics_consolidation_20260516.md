# Runtime Artifact Semantics Consolidation Plan

Date: 2026-05-16

## Problem

The advisor still reports under-amortized infrastructure around runtime semantic/value families:

- `ObjectLabelPlaneDomainStrategy`
- `RuntimePlaneAxisProjectionStrategy`
- `SourceSpatialDomainAdapter`
- `ObjectLabelMeasurementPayloadStrategy`

These findings overlap with real runtime architecture, but they may also be scan-scope artifacts. Several of these families have consumers outside the targeted four-file advisor scan.

## Target

Runtime artifact semantics should remain nominal and domain-specific while sharing infrastructure only where the domain operation is genuinely the same.

The target is not fewer classes at all costs. The target is:

- one authority for source spatial domain adaptation,
- one authority for runtime plane-axis projection semantics,
- one authority for object-label payload shape/identity semantics,
- explicit fanout into aligned image payload and measurement alignment code.

## Investigation Steps

1. Trace full fanout of each reported family across:
   - `openhcs/core/runtime_semantics.py`
   - `openhcs/core/runtime_values.py`
   - `openhcs/core/aligned_image_payload.py`
   - `openhcs/core/measurement_image_alignment.py`
   - `openhcs/core/runtime_equivalence.py`
2. Distinguish true duplicate behavior from parallel domain witnesses.
3. Identify whether consolidation belongs in core runtime semantics, aligned payload, or equivalence policy.

## Refactor Rules

- Keep object labels, images, tables, and measurement facts semantically distinct even when backed by arrays.
- Prefer registered strategy families over dict dispatch.
- Prefer value objects over loose tuples or dict bags.
- Do not add local private helpers as an abstraction substitute.
- Do not collapse leaf classes that exist to make domain-specific registry membership explicit.

## Staged Work

### Stage 1: Fanout Map

- Build a small documented fanout map from `rg` and code inspection.
- Decide which findings are true consolidation opportunities.

### Stage 2: Shared Primitive Extraction

- Extract only primitives with identical semantics across families.
- Candidate primitives:
  - axis selection records,
  - spatial rank projection records,
  - object-label identity payload records.

### Stage 3: Domain Rebinding

- Rebind each registered family to the shared primitive while preserving the domain-specific class root.
- Keep family-specific names visible at call sites.

### Stage 4: Validation

- Run runtime semantic/value tests and CP module tests.
- Run full unit suite.
- Re-run advisor across both targeted files and cross-fanout files.

## Completion Criteria

- Under-amortized findings are removed or clearly documented as false positives due to scan scope.
- Runtime semantic call sites are more explicit, not more generic.
- No object-label/image/table parity regressions.

