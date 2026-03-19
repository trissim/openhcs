# PDB Redocking Benchmark

- Phase: `complete`
- Charge method: `simple`
- Requested complexes: `10`
- Completed DQ-Dock complexes: `10`
- Completed Vina complexes: `10`
- Excluded complexes: `0`
- Poses: `50`
- Optimization steps: `50`
- Pocket-guided sampling: `True`
- Vina preparation: `meeko_pdbqt_with_direct_fallback`
- Competitors: `smina`

## Summary

- DQ-Dock avg RMSD: `0.45062377154827116`
- DQ-Dock total time (s): `50.45769929885864`
- Vina avg top-pose RMSD: `4.824460482597351`
- Vina avg best-returned-mode RMSD: `4.495517778396606`
- Vina total time (s): `750.9497225284576`
- Total benchmark time (s): `915.2903304100037`

## DQ-Dock

| PDB | Target | RMSD | Time (s) | Energy | Gap Proof | Native Rank | Energy Gap |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1hvr | HIV-1 Protease | 0.215 | 8.470 | -11.069 | proved | 1 | 41.318159103393555 |
| 1fkf | FKBP12-FK506 | 0.475 | 10.867 | -7.263 | proved | 1 | 5.062362194061279 |
| 2rh1 | beta2-Adrenergic Receptor | 0.762 | 3.814 | -6.718 | proved | 1 | 5.8549840450286865 |
| 2xp9 | PIN1 | 0.287 | 3.480 | -5.257 | proved | 1 | 3.920843482017517 |
| 3p5o | BRD4 Bromodomain 1 | 0.647 | 3.443 | -6.475 | proved | 1 | 4.631722927093506 |
| 6w63 | SARS-CoV-2 Main Protease | 0.309 | 5.013 | -7.617 | proved | 1 | 5.824819087982178 |
| 1jvp | CDK2 | 1.144 | 3.602 | -12.650 | proved | 4 | 519.9891548156738 |
| 2hyy | ABL1 Kinase | 0.151 | 3.681 | -8.874 | proved | 1 | 35.34443283081055 |
| 1c5w | Urokinase-type Plasminogen Activator | 0.231 | 3.989 | -4.736 | proved | 1 | 25.21354103088379 |
| 4d83 | BACE-1 | 0.285 | 4.098 | -11.359 | proved | 1 | 1031.9862480163574 |

## Vina

| PDB | Target | Top RMSD | Best Returned RMSD | Time (s) | Affinity |
| --- | --- | ---: | ---: | ---: | ---: |
| 1hvr | HIV-1 Protease | 6.436 | 6.199 | 122.258 | -15.100 |
| 1fkf | FKBP12-FK506 | 5.890 | 5.596 | 125.732 | -10.300 |
| 2rh1 | beta2-Adrenergic Receptor | 1.675 | 1.636 | 25.296 | -10.200 |
| 2xp9 | PIN1 | 2.454 | 2.454 | 51.822 | -6.900 |
| 3p5o | BRD4 Bromodomain 1 | 5.176 | 5.047 | 33.546 | -9.400 |
| 6w63 | SARS-CoV-2 Main Protease | 6.705 | 4.998 | 78.516 | -8.000 |
| 1jvp | CDK2 | 4.046 | 3.964 | 106.758 | -14.000 |
| 2hyy | ABL1 Kinase | 6.355 | 6.179 | 120.538 | -12.100 |
| 1c5w | Urokinase-type Plasminogen Activator | 3.264 | 3.264 | 5.440 | -6.700 |
| 4d83 | BACE-1 | 6.244 | 5.619 | 81.044 | -11.700 |
