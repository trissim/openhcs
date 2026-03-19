# PDB Redocking Benchmark

- Phase: `complete`
- Charge method: `simple`
- Requested complexes: `1`
- Completed DQ-Dock complexes: `1`
- Excluded complexes: `0`
- Poses: `500`
- Optimization steps: `3`
- Pocket-guided sampling: `True`
- Competitors: `Vina`

## Summary

- DQ-Dock avg RMSD: `0.4031113088130951`
- DQ-Dock total time (s): `13.025433778762817`
- Total benchmark time (s): `138.4484405517578`

## DQ-Dock

| PDB | Target | RMSD | Time (s) | Energy | Gap Proof | Native Rank | Energy Gap |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1hvr | HIV-1 Protease | 0.403 | 13.025 | -10.668 | proved | 1 | 10.18148124217987 |

## Vina

- Score: `affinity`
- Prep: `direct_pdb`
- Completed complexes: `1`
- Successful complexes: `1`
- Avg top RMSD: `6.4361395835876465`
- Avg best-returned RMSD: `6.198639869689941`
- Total time (s): `122.19366264343262`

| PDB | Target | Status | Top RMSD | Best Returned RMSD | Time (s) | Score | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1hvr | HIV-1 Protease | success | 6.4361395835876465 | 6.198639869689941 | 122.194 | -15.1 | None |
