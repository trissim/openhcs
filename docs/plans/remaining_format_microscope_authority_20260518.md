# Remaining Format And Microscope Authority Refactor - 2026-05-18

## Evidence

Refreshed advisor scan reports active findings in:

- `openhcs/formats/experimental_analysis.py`: 27 findings.
- `openhcs/formats/metaxpress.py`: closed-family dispatch and bare function
  family findings.
- `openhcs/processing/backends/analysis/cx5_format.py`: format-reader dispatch
  and bare function family findings.
- `openhcs/microscopes/microscope_interfaces.py`: string dispatch over
  microscope/interface choices.
- `openhcs/microscopes/bbbc.py`: ABC hierarchy candidate for BBBC filename
  parser classes.
- `openhcs/microscopes/openhcs.py`: remaining parser/interface authority debt.

## Problem

Reader/parser code still exposes file-format families as bare functions and
literal dispatch blocks. These are domain models: plate format, feature table
kind, microscope/parser family, and filename convention. Keeping them as loose
function clusters makes new formats harder to add and makes validation rules
harder to test.

## Target Shape

- Nominal reader/parser interfaces for format families:
  `PlateResultReader`, `FeatureTableReader`, `MicroscopeFilenameParser`, or
  equivalent existing names.
- Closed dispatch through typed format keys, not local string branches.
- Shared parser skeletons moved into ABC/template-method bases when BBBC/OpenHCS
  parsers repeat the same method roles.
- Preserve public reader function wrappers only as compatibility facades around
  nominal readers.

## Phases

1. Add characterization tests for `experimental_analysis`, `metaxpress`, and
   `cx5_format` public reader outputs if not already present.
2. Introduce nominal reader classes behind existing public functions.
3. Move `read_results`, `get_features`, `create_plates_dict`, and
   `fill_plates_dict` dispatch into reader methods or operation specs.
4. Extract BBBC parser shared method skeletons into an ABC if tests confirm
   identical behavior roles.
5. Convert microscope interface string dispatch to a typed key/factory
   authority.

## Verification Gates

```bash
.venv/bin/python -m py_compile \
  openhcs/formats/experimental_analysis.py \
  openhcs/formats/metaxpress.py \
  openhcs/processing/backends/analysis/cx5_format.py \
  openhcs/microscopes/bbbc.py \
  openhcs/microscopes/openhcs.py \
  openhcs/microscopes/microscope_interfaces.py

timeout 120 .venv/bin/python -m nominal_refactor_advisor \
  openhcs/formats/experimental_analysis.py \
  openhcs/formats/metaxpress.py \
  openhcs/processing/backends/analysis/cx5_format.py \
  openhcs/microscopes/bbbc.py \
  openhcs/microscopes/openhcs.py \
  openhcs/microscopes/microscope_interfaces.py
```

Add or run any reader/parser tests discovered during phase 1.

## Risks

- Some file-format behavior may rely on real fixture data. If fixtures are
  missing, add small synthetic characterization fixtures before refactoring.
- Public bare functions may be imported by users; keep wrappers unless the
  public API plan owns a migration.
