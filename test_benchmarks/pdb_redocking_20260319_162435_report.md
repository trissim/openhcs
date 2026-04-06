# PDB Redocking Benchmark

- Phase: `complete`
- Charge method: `simple`
- Requested complexes: `10`
- Completed DQ-Dock complexes: `10`
- Excluded complexes: `0`
- Poses: `1000`
- Optimization steps: `100`
- Pocket-guided sampling: `True`
- Competitors: `Vina`

## Summary

- DQ-Dock avg RMSD: `1.6469606518745423`
- DQ-Dock total time (s): `95.1854920387268`
- Total benchmark time (s): `1865.7154059410095`

## DQ-Dock

| PDB | Target | RMSD | Time (s) | Energy | Gap Proof | Native Rank | Energy Gap |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1hvr | HIV-1 Protease | 0.325 | 11.498 | -10.932 | proved | 1 | 10.181472539901733 |
| 1fkf | FKBP12-FK506 | 4.902 | 5.898 | -3.703 | proved | 1 | 5.42975640296936 |
| 2rh1 | beta2-Adrenergic Receptor | 0.619 | 6.762 | -6.895 | proved | 1 | 4.400501012802124 |
| 2xp9 | PIN1 | 4.009 | 30.288 | -3.214 | proved | 1 | 2.2629551887512207 |
| 3p5o | BRD4 Bromodomain 1 | 0.626 | 6.608 | -5.827 | proved | 1 | 4.441833019256592 |
| 6w63 | SARS-CoV-2 Main Protease | 1.087 | 6.110 | -6.692 | proved | 1 | 5.4581522941589355 |
| 1jvp | CDK2 | 1.762 | 6.915 | -8.446 | proved | 52 | 568.7809858322144 |
| 2hyy | ABL1 Kinase | 0.854 | 7.002 | -3.424 | proved | 1 | 7.188594460487366 |
| 1c5w | Urokinase-type Plasminogen Activator | 1.801 | 6.989 | -3.637 | proved | 1 | 2.093143939971924 |
| 4d83 | BACE-1 | 0.485 | 7.115 | -10.463 | proved | 1 | 8.058666706085205 |

## Vina

- Score: `affinity`
- Prep: `direct_pdb`
- Completed complexes: `10`
- Successful complexes: `10`
- Avg top RMSD: `6.606834983825683`
- Avg best-returned RMSD: `4.977611875534057`
- Total time (s): `1717.2260937690735`

| PDB | Target | Status | Top RMSD | Best Returned RMSD | Time (s) | Score | Error |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1hvr | HIV-1 Protease | success | 6.436139106750488 | 6.198639869689941 | 327.510 | -15.1 | None |
| 1fkf | FKBP12-FK506 | success | 5.890055179595947 | 5.596170425415039 | 349.963 | -10.3 | None |
| 2rh1 | beta2-Adrenergic Receptor | success | 6.399831295013428 | 3.771505832672119 | 65.057 | -10.2 | None |
| 2xp9 | PIN1 | success | 6.7812371253967285 | 4.781976222991943 | 112.415 | -7.2 | None |
| 3p5o | BRD4 Bromodomain 1 | success | 5.175532817840576 | 5.047098636627197 | 89.822 | -9.4 | None |
| 6w63 | SARS-CoV-2 Main Protease | success | 6.881319046020508 | 4.693138122558594 | 173.115 | -8.1 | None |
| 1jvp | CDK2 | success | 4.044555187225342 | 3.960338592529297 | 291.264 | -14.0 | None |
| 2hyy | ABL1 Kinase | success | 10.738680839538574 | 8.202057838439941 | 243.237 | -12.7 | None |
| 1c5w | Urokinase-type Plasminogen Activator | success | 4.144738674163818 | 3.896393299102783 | 13.203 | -7.0 | None |
| 4d83 | BACE-1 | success | 9.576260566711426 | 3.6287999153137207 | 51.641 | -8.8 | None |
