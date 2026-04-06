# PDB Redocking Benchmark

- Phase: `dq_dock`
- Charge method: `simple`
- Requested complexes: `10`
- Completed DQ-Dock complexes: `5`
- Excluded complexes: `0`
- Poses: `50`
- Optimization steps: `50`
- Pocket-guided sampling: `True`
- Competitors: `Vina`

## Summary

- DQ-Dock avg RMSD: `2.8772232592105866`
- DQ-Dock total time (s): `90.51376461982727`
- Total benchmark time (s): `108.5041275024414`

## DQ-Dock

| PDB | Target | RMSD | Time (s) | Energy | Gap Proof | Native Rank | Energy Gap |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1hvr | HIV-1 Protease | 0.325 | 6.944 | -10.932 | proved | 1 | 41.31803512573242 |
| 1fkf | FKBP12-FK506 | 7.205 | 35.756 | -3.639 | proved | 1 | 4.311635494232178 |
| 2rh1 | beta2-Adrenergic Receptor | 0.619 | 3.485 | -6.895 | proved | 1 | 5.854984402656555 |
| 2xp9 | PIN1 | 5.114 | 16.532 | -3.689 | proved | 1 | 3.4123940467834473 |
| 3p5o | BRD4 Bromodomain 1 | 1.124 | 27.797 | -5.471 | proved | 1 | 4.031202793121338 |

## Vina

- Score: `affinity`
- Prep: `direct_pdb`
- Completed complexes: `0`
- Successful complexes: `0`
- Avg top RMSD: `None`
- Avg best-returned RMSD: `None`
- Total time (s): `0.0`

| PDB | Target | Status | Top RMSD | Best Returned RMSD | Time (s) | Score | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
