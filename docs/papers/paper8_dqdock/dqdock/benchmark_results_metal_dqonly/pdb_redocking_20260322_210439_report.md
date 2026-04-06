# PDB Redocking Benchmark

- Phase: `complete`
- Charge method: `gasteiger`
- Requested complexes: `170`
- Completed DQ-Dock complexes: `127`
- Successful DQ-Dock complexes: `127`
- Excluded complexes: `33`
- Poses: `10000`
- Optimization steps: `10`
- Pocket-guided sampling: `True`
- Competitors: ``

## Summary

- DQ-Dock avg RMSD: `1.1450529807489218`
- DQ-Dock total time (s): `7883.4976394228315`
- Total benchmark time (s): `8196.161312818527`

## Benchmark Diversity

### By Target Class

| Target Class | Count |
| --- | ---: |
| hydrolase | 61 |
| transferase | 31 |
| other | 14 |
| oxidoreductase | 9 |
| transport | 6 |
| lyase | 4 |
| transcription | 1 |
| isomerase | 1 |

### By Binding Mode

| Binding Mode | Count |
| --- | ---: |
| standard | 98 |
| metal_mediated | 23 |
| multi_residue | 6 |

## DQ-Dock

| PDB | Target | Class | Mode | Status | RMSD | Time (s) | Energy | Gap Proof | Native Rank | Energy Gap | Error |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 1hk4 | HUMAN SERUM ALBUMIN COMPLEXED WITH THYROXINE | other | standard | success | 0.179 | 68.506 | -11.735 | proved | 1 | 1.597 | n/a |
| 1gni | HUMAN SERUM ALBUMIN COMPLEXED WITH CIS-9-OCTADECENOIC ACID | other | standard | success | 0.330 | 41.183 | -10.888 | proved | 9 | 6.173 | n/a |
| 1nhu | HEPATITIS C VIRUS RNA POLYMERASE IN COMPLEX WITH | transferase | standard | success | 0.201 | 59.168 | -10.899 | proved | 21 | 5.899 | n/a |
| 2d3z | X-RAY CRYSTAL STRUCTURE OF HEPATITIS C VIRUS RNA-DEPENDENT | transferase | standard | success | 0.285 | 51.859 | -13.376 | proved | 11 | 6.529 | n/a |
| 2d3u | X-RAY CRYSTAL STRUCTURE OF HEPATITIS C VIRUS RNA DEPENDENT | transferase | standard | success | 0.410 | 48.210 | -13.821 | proved | 9 | 6.566 | n/a |
| 1ajp | PENICILLIN ACYLASE COMPLEXED WITH 2,5-DIHYDROXYPHENYLACETIC | hydrolase | standard | success | 0.283 | 44.797 | -8.196 | proved | 8 | 3.124 | n/a |
| 1ai5 | PENICILLIN ACYLASE COMPLEXED WITH M-NITROPHENYLACETIC ACID | hydrolase | standard | success | 0.329 | 44.915 | -9.441 | proved | 7 | 4.527 | n/a |
| 1ajq | PENICILLIN ACYLASE COMPLEXED WITH THIOPHENEACETIC ACID | hydrolase | standard | success | 0.273 | 48.289 | -8.647 | proved | 4 | 4.641 | n/a |
| 1gpk | STRUCTURE OF ACETYLCHOLINESTERASE COMPLEX WITH | hydrolase | standard | success | 0.399 | 61.936 | -12.233 | proved | 6 | 5.515 | n/a |
| 1h23 | STRUCTURE OF ACETYLCHOLINESTERASE (E.C. 3.1.1.7) COMPLEXED | hydrolase | standard | success | 0.195 | 64.112 | -16.813 | proved | 6 | 6.405 | n/a |
| 1e66 | STRUCTURE OF ACETYLCHOLINESTERASE COMPLEXED WITH | hydrolase | standard | success | 0.291 | 64.043 | -17.629 | proved | 4 | 6.129 | n/a |
| 2rkm | STRUCTURE OF OPPA COMPLEXED WITH LYS-LYS | transport | multi_residue | success | 0.109 | 49.634 | -13.007 | proved | 5 | 5.799 | n/a |
| 1u33 | IN SITU EXTENSION AS AN APPROACH FOR IDENTIFYING NOVEL | hydrolase | standard | success | 0.229 | 42.516 | -18.105 | proved | 4 | 9.676 | n/a |
| 1xd1 | ACARBOSE REARRANGEMENT MECHANISM IMPLIED BY THE KINETIC AND | hydrolase | standard | success | 0.305 | 38.452 | -19.665 | proved | 2 | 4.086 | n/a |
| 1uwt | STRUCTURE OF BETA-GLYCOSIDASE FROM SULFOLOBUS SOLFATARICUS | hydrolase | standard | success | 0.129 | 46.325 | -12.695 | proved | 9 | 6.803 | n/a |
| 2ceq | BETA-GLYCOSIDASE FROM SULFOLOBUS SOLFATARICUS IN COMPLEX | hydrolase | standard | success | 0.167 | 58.652 | -13.556 | proved | 25 | 6.334 | n/a |
| 2cer | BETA-GLYCOSIDASE FROM SULFOLOBUS SOLFATARICUS IN COMPLEX | hydrolase | standard | success | 0.142 | 66.422 | -16.850 | proved | 5 | 8.413 | n/a |
| 2qwb | THE X-RAY STRUCTURE OF A COMPLEX OF SIALIC ACID AND A DRUG | hydrolase | standard | success | 0.372 | 47.631 | -12.571 | proved | 16 | 8.735 | n/a |
| 2qwd | THE X-RAY STRUCTURE OF A COMPLEX OF 4-AMINO-NEU5AC2EN AND A | hydrolase | standard | success | 0.296 | 48.698 | -10.831 | proved | 17 | 8.029 | n/a |
| 2qwe | THE X-RAY STRUCTURE OF A COMPLEX OF 4-GUANIDINO-NEU5AC2EN | hydrolase | standard | success | 0.157 | 45.433 | -13.112 | proved | 4 | 7.202 | n/a |
| 2j77 | BETA-GLUCOSIDASE FROM THERMOTOGA MARITIMA IN COMPLEX WITH | hydrolase | standard | success | 0.137 | 51.304 | -12.054 | proved | 29 | 7.518 | n/a |
| 2j78 | BETA-GLUCOSIDASE FROM THERMOTOGA MARITIMA IN COMPLEX WITH | hydrolase | standard | success | 0.112 | 42.575 | -12.886 | proved | 5 | 7.309 | n/a |
| 2cet | BETA-GLUCOSIDASE FROM THERMOTOGA MARITIMA IN COMPLEX WITH | hydrolase | standard | success | 0.088 | 47.007 | -16.997 | proved | 6 | 6.424 | n/a |
| 2g94 | CRYSTAL STRUCTURE OF BETA-SECRETASE BOUND TO A POTENT AND | hydrolase | standard | success | 0.235 | 60.674 | -22.821 | proved | 2 | 3.170 | n/a |
| 1v16 | CROSSTALK BETWEEN COFACTOR BINDING AND THE PHOSPHORYLATION | oxidoreductase | metal_mediated | success | 0.266 | 57.201 | -68.981 | proved | 25 | 37.781 | n/a |
| 1olu | ROLES OF HIS291-ALPHA AND HIS146-BETA' IN THE REDUCTIVE | oxidoreductase | metal_mediated | success | 0.414 | 43.793 | -68.053 | proved | 25 | 37.848 | n/a |
| 1ols | ROLES OF HIS291-ALPHA AND HIS146-BETA' IN THE REDUCTIVE | oxidoreductase | metal_mediated | success | 0.076 | 71.450 | -69.517 | proved | 10 | 42.293 | n/a |
| 1n2v | CRYSTAL STRUCTURE OF TGT IN COMPLEX WITH | transferase | standard | success | 0.174 | 54.950 | -9.440 | proved | 16 | 5.640 | n/a |
| 1k4g | CRYSTAL STRUCTURE OF TRNA-GUANINE TRANSGLYCOSYLASE (TGT) | transferase | standard | success | 0.784 | 47.476 | -11.774 | proved | 21 | 6.529 | n/a |
| 1s39 | CRYSTAL STRUCTURE OF TGT IN COMPLEX WITH | transferase | standard | success | 0.153 | 45.990 | -11.240 | proved | 9 | 3.629 | n/a |
| 2bak | P38ALPHA MAP KINASE BOUND TO MPAQ | transferase | standard | success | 0.364 | 59.824 | -19.422 | proved | 3 | 7.893 | n/a |
| 2baj | P38ALPHA BOUND TO PYRAZOLOUREA | transferase | standard | success | 0.302 | 64.482 | -17.606 | proved | 4 | 9.660 | n/a |
| 1ndw | CRYSTAL STRUCTURE OF ADENOSINE DEAMINASE COMPLEXED WITH | hydrolase | metal_mediated | success | 0.226 | 67.095 | -12.071 | proved | 4 | 6.614 | n/a |
| 1ndy | CRYSTAL STRUCTURE OF ADENOSINE DEAMINASE COMPLEXED WITH | hydrolase | metal_mediated | success | 0.085 | 66.013 | -13.767 | proved | 4 | 5.686 | n/a |
| 1ndz | CRYSTAL STRUCTURE OF ADENOSINE DEAMINASE COMPLEXED WITH | hydrolase | metal_mediated | success | 0.456 | 69.439 | -19.320 | proved | 5 | 9.182 | n/a |
| 1l2s | X-RAY CRYSTAL STRUCTURE OF AMPC BETA-LACTAMASE FROM E. COLI | hydrolase | standard | success | 0.241 | 63.666 | -11.579 | proved | 30 | 4.748 | n/a |
| 1xgj | AMPC BETA-LACTAMASE IN COMPLEX WITH 3-(4-CARBOXY-2-HYDROXY- | hydrolase | standard | success | 0.358 | 51.117 | -10.694 | proved | 29 | 5.029 | n/a |
| 1q8t | THE CATALYTIC SUBUNIT OF CAMP-DEPENDENT PROTEIN KINASE | transferase | standard | success | 0.971 | 49.527 | -9.381 | proved | 9 | 4.280 | n/a |
| 1ydt | STRUCTURE OF CAMP-DEPENDENT PROTEIN KINASE, ALPHA-CATALYTIC | transferase | standard | success | 0.314 | 55.815 | -15.607 | proved | 2 | 2.997 | n/a |
| 1re8 | CRYSTAL STRUCTURE OF CAMP-DEPENDENT PROTEIN KINASE | transferase | standard | success | 0.308 | 64.430 | -19.670 | proved | 4 | 11.346 | n/a |
| 1m2q | CRYSTAL STRUCTURE OF 1,8-DI-HYDROXY-4-NITRO-XANTEN-9-ONE/CK2 | transferase | standard | success | 0.304 | 57.203 | -13.715 | proved | 14 | 4.749 | n/a |
| 1om1 | CRYSTAL STRUCTURE OF MAIZE CK2 ALPHA IN COMPLEX WITH IQA | transferase | standard | success | 0.449 | 61.472 | -13.805 | proved | 11 | 8.314 | n/a |
| 1zoe | CRYSTAL STRUCTURE OF PROTEIN KINASE CK2 IN COMPLEX WITH | transferase | standard | success | 0.165 | 62.290 | -11.046 | proved | 4 | 5.332 | n/a |
| 2azr | CRYSTAL STRUCTURE OF PTP1B WITH BICYCLIC THIOPHENE INHIBITOR | hydrolase | standard | success | 0.212 | 53.039 | -12.565 | proved | 4 | 6.651 | n/a |
| 1g7f | HUMAN PTP1B CATALYTIC DOMAIN COMPLEXED WITH PNU177496 | hydrolase | standard | success | 0.274 | 58.487 | -14.408 | proved | 3 | 3.811 | n/a |
| 1nny | POTENT, SELECTIVE PROTEIN TYROSINE PHOSPHATASE 1B INHIBITOR | hydrolase | standard | success | 0.213 | 60.672 | -21.871 | proved | 5 | 10.034 | n/a |
| 5er1 | A RATIONAL APPROACH TO THE DESIGN OF ANTIHYPERTENSIVES. | hydrolase | standard | success | 0.227 | 57.164 | -14.409 | proved | 1 | 0.456 | n/a |
| 4tln | BINDING OF HYDROXAMIC ACID INHIBITORS TO CRYSTALLINE | hydrolase | metal_mediated | success | 0.268 | 54.166 | -61.453 | proved | 36 | 85.225 | n/a |
| 1tmn | BINDING OF N-CARBOXYMETHYL DIPEPTIDE INHIBITORS TO | hydrolase | metal_mediated | success | 0.567 | 57.314 | -63.277 | proved | 29 | 67.448 | n/a |
| 4tmn | SLOW-AND FAST-BINDING INHIBITORS OF THERMOLYSIN DISPLAY | hydrolase | metal_mediated | success | 0.419 | 57.699 | -70.274 | proved | 13 | 35.447 | n/a |
| 1fh7 | CRYSTAL STRUCTURE OF THE XYLANASE CEX WITH | hydrolase | multi_residue | success | 0.147 | 58.198 | -13.840 | proved | 5 | 3.652 | n/a |
| 1fh9 | CRYSTAL STRUCTURE OF THE XYLANASE CEX WITH | hydrolase | multi_residue | success | 0.197 | 71.010 | -15.149 | proved | 11 | 8.828 | n/a |
| 1fh8 | CRYSTAL STRUCTURE OF THE XYLANASE CEX WITH | hydrolase | multi_residue | success | 0.110 | 57.030 | -12.246 | proved | 5 | 1.352 | n/a |
| 2ctc | THE HIGH RESOLUTION CRYSTAL STRUCTURE OF THE COMPLEX | hydrolase | metal_mediated | success | 3.379 | 58.660 | -21.666 | proved | 14 | 6.484 | n/a |
| 8cpa | COMPARISON OF THE STRUCTURES OF THREE CARBOXYPEPTIDASE | hydrolase | metal_mediated | success | 0.321 | 55.837 | -42.559 | proved | 6 | 68.253 | n/a |
| 7cpa | COMPARISON OF THE STRUCTURES OF THREE CARBOXYPEPTIDASE | hydrolase | metal_mediated | success | 0.409 | 52.378 | -47.038 | proved | 12 | 23.127 | n/a |
| 1e1v | HUMAN CYCLIN DEPENDENT KINASE 2 COMPLEXED WITH THE | transferase | standard | success | 1.075 | 59.182 | -9.929 | proved | 7 | 6.400 | n/a |
| 1b39 | HUMAN CYCLIN-DEPENDENT KINASE 2 PHOSPHORYLATED ON THR 160 | transferase | metal_mediated | success | 0.436 | 63.597 | -58.531 | proved | 8 | 51.862 | n/a |
| 1pxo | HUMAN CYCLIN DEPENDENT KINASE 2 COMPLEXED WITH THE | transferase | standard | success | 0.198 | 53.410 | -7.661 | proved | 19 | 2.580 | n/a |
| 1vzq | COMPLEX OF THROMBIN WITH DESIGNED INHIBITOR 7165 | hydrolase | standard | success | 0.300 | 61.044 | -18.183 | proved | 2 | 9.732 | n/a |
| 1sl3 | CRYSTAL STRUCTUE OF THROMBIN IN COMPLEX WITH A POTENT P1 | hydrolase | standard | success | 0.211 | 68.223 | -20.413 | proved | 2 | 4.440 | n/a |
| 2brb | STRUCTURE-BASED DESIGN OF NOVEL CHK1 INHIBITORS: INSIGHTS | transferase | standard | success | 0.437 | 57.839 | -12.589 | proved | 6 | 5.481 | n/a |
| 2brm | STRUCTURE-BASED DESIGN OF NOVEL CHK1 INHIBITORS: INSIGHTS | transferase | standard | success | 0.581 | 57.443 | -11.943 | proved | 12 | 4.862 | n/a |
| 1nvq | THE COMPLEX STRUCTURE OF CHECKPOINT KINASE CHK1/UCN-01 | transferase | standard | success | 0.450 | 51.952 | -18.852 | proved | 11 | 10.316 | n/a |
| 1jqd | CRYSTAL STRUCTURE ANALYSIS OF HUMAN HISTAMINE | transferase | standard | success | 0.287 | 56.888 | -13.274 | proved | 8 | 8.036 | n/a |
| 2aov | HISTAMINE METHYLTRANSFERASE COMPLEXED WITH THE ANTIFOLATE | transferase | standard | success | 0.310 | 68.804 | -9.838 | proved | 5 | 4.782 | n/a |
| 2aou | HISTAMINE METHYLTRANSFERASE COMPLEXED WITH THE ANTIMALARIAL | transferase | standard | success | 0.205 | 69.620 | -14.172 | proved | 4 | 6.820 | n/a |
| 1pbq | CRYSTAL STRUCTURE OF THE NR1 LIGAND BINDING CORE IN COMPLEX | transport | standard | success | 0.148 | 65.295 | -12.488 | proved | 5 | 5.858 | n/a |
| 1vfn | PURINE NUCLEOSIDE PHOSPHORYLASE | transferase | standard | success | 0.502 | 70.129 | -7.476 | proved | 25 | 4.331 | n/a |
| 1p1q | CRYSTAL STRUCTURE OF THE GLUR2 LIGAND BINDING CORE (S1S2J) | other | standard | success | 0.191 | 70.970 | -12.428 | proved | 3 | 5.625 | n/a |
| 1syh | X-RAY STRUCTURE OF THE GLUR2 LIGAND-BINDING CORE (S1S2J) IN | other | standard | success | 0.021 | 69.645 | -16.203 | proved | 2 | 10.163 | n/a |
| 1ftm | CRYSTAL STRUCTURE OF THE GLUR2 LIGAND BINDING CORE (S1S2J) | other | standard | success | 0.111 | 70.214 | -12.573 | proved | 3 | 4.863 | n/a |
| 1fcx | ISOTYPE SELECTIVITY OF THE HUMAN RETINOIC ACID NUCLEAR | other | standard | success | 1.181 | 52.988 | -5.866 | proved | 25 | 3.589 | n/a |
| 1fd0 | ISOTYPE SELECTIVITY OF THE HUMAN RETINOIC ACID NUCLEAR | other | standard | success | 0.258 | 71.302 | -18.279 | proved | 6 | 9.381 | n/a |
| 1fcz | ISOTYPE SELECTIVITY OF THE HUMAN RETINOIC ACID NUCLEAR | other | standard | success | 1.220 | 49.065 | -6.334 | proved | 22 | 3.894 | n/a |
| 1f4e | CRYSTAL STRUCTURE OF E. COLI THYMIDYLATE SYNTHASE COMPLEXED | transferase | standard | success | 0.299 | 56.149 | -8.843 | proved | 12 | 3.851 | n/a |
| 1f4f | CRYSTAL STRUCTURE OF E. COLI THYMIDYLATE SYNTHASE COMPLEXED | transferase | standard | success | 0.203 | 71.004 | -10.736 | proved | 8 | 4.176 | n/a |
| 1f4g | CRYSTAL STRUCTURE OF E. COLI THYMIDYLATE SYNTHASE COMPLEXED | transferase | standard | success | 0.253 | 66.347 | -14.198 | proved | 5 | 8.184 | n/a |
| 1sqa | SUBSTITUTED 2-NAPHTHAMIDINE INHIBITORS OF UROKINASE | hydrolase | standard | success | 0.066 | 81.482 | -14.645 | proved | 3 | 7.380 | n/a |
| 2b1v | HUMAN ESTROGEN RECEPTOR ALPHA LIGAND-BINDING DOMAIN IN | other | standard | success | 0.926 | 44.499 | -11.338 | proved | 4 | 5.544 | n/a |
| 2fai | HUMAN ESTROGEN RECEPTOR ALPHA LIGAND-BINDING DOMAIN IN | other | standard | success | 0.999 | 64.241 | -11.636 | proved | 3 | 5.971 | n/a |
| 2ayr | A SERM DESIGNED FOR THE TREATMENT OF UTERINE LEIOMYOMA WITH | transcription | standard | success | 0.272 | 65.664 | -19.602 | proved | 5 | 7.439 | n/a |
| 1ttm | HUMAN CARBONIC ANHYDRASE II COMPLEXED WITH 667-COUMATE | lyase | metal_mediated | success | 1.428 | 68.517 | -77.169 | proved | 26 | 78.407 | n/a |
| 1if7 | CARBONIC ANHYDRASE II COMPLEXED WITH | lyase | metal_mediated | success | 1.429 | 66.970 | -56.658 | proved | 30 | 72.762 | n/a |
| 2bok | FACTOR XA - CATION | hydrolase | standard | success | 0.313 | 68.639 | -16.471 | proved | 5 | 9.548 | n/a |
| 1nfy | CRYSTAL STRUCTURE OF HUMAN COAGULATION FACTOR XA COMPLEXED | hydrolase | standard | success | 0.358 | 63.042 | -17.884 | proved | 4 | 7.458 | n/a |
| 1mq6 | CRYSTAL STRUCTURE OF 3-CHLORO-N-[4-CHLORO-2-[[(5-CHLORO-2-PY | hydrolase | standard | success | 0.207 | 84.904 | -19.034 | proved | 4 | 8.109 | n/a |
| 2usn | CRYSTAL STRUCTURE OF THE CATALYTIC DOMAIN OF HUMAN | hydrolase | metal_mediated | success | 0.208 | 64.106 | -17.402 | proved | 25 | 19.719 | n/a |
| 2d1o | STROMELYSIN-1 (MMP-3) COMPLEXED TO A HYDROXAMIC ACID | hydrolase | metal_mediated | success | 0.266 | 69.509 | -71.371 | proved | 17 | 44.202 | n/a |
| 2flb | DISCOVERY OF A NOVEL HYDROXY PYRAZOLE BASED FACTOR IXA | hydrolase | standard | success | 3.400 | 78.015 | -6.768 | proved | 25 | 9.448 | n/a |
| 2b7d | FACTOR VIIA INHIBITORS: CHEMICAL OPTIMIZATION, PRECLINICAL | other | standard | success | 0.313 | 63.545 | -16.011 | proved | 9 | 8.271 | n/a |
| 1j17 | FACTOR XA SPECIFIC INHIBITOR IN COMPLEX WITH RAT TRYPSIN | hydrolase | standard | success | 0.369 | 75.230 | -18.603 | proved | 4 | 9.264 | n/a |
| 1v2o | TRYPSIN INHIBITOR IN COMPLEX WITH BOVINE TRYPSIN VARIANT | hydrolase | standard | success | 0.181 | 63.613 | -11.097 | proved | 1 | 0.259 | n/a |
| 1jys | CRYSTAL STRUCTURE OF E. COLI MTA/ADOHCY NUCLEOSIDASE | hydrolase | standard | success | 0.709 | 63.528 | -6.779 | proved | 21 | 3.065 | n/a |
| 1nc1 | CRYSTAL STRUCTURE OF E. COLI MTA/ADOHCY NUCLEOSIDASE | hydrolase | standard | success | 0.327 | 105.151 | -15.727 | proved | 1 | 0.971 | n/a |
| 1y6q | CYRSTAL STRUCTURE OF MTA/ADOHCY NUCLEOSIDASE COMPLEXED WITH | hydrolase | standard | success | 0.332 | 86.184 | -15.336 | proved | 4 | 8.044 | n/a |
| 1bma | BENZYL METHYL AMINIMIDE INHIBITOR COMPLEXED TO PORCINE | hydrolase | standard | success | 0.113 | 62.264 | -17.906 | proved | 6 | 8.356 | n/a |
| 1ela | ANALOGOUS INHIBITORS OF ELASTASE DO NOT ALWAYS BIND | hydrolase | standard | success | 0.166 | 66.941 | -16.346 | proved | 6 | 7.998 | n/a |
| 1elb | ANALOGOUS INHIBITORS OF ELASTASE DO NOT ALWAYS BIND | hydrolase | standard | success | 0.207 | 64.166 | -14.708 | proved | 5 | 6.230 | n/a |
| 1a69 | PURINE NUCLEOSIDE PHOSPHORYLASE IN COMPLEX WITH FORMYCIN B | transferase | standard | success | 0.285 | 85.433 | -14.431 | proved | 1 | 1.278 | n/a |
| 3pce | STRUCTURE OF PROTOCATECHUATE 3,4-DIOXYGENASE COMPLEXED WITH | oxidoreductase | metal_mediated | success | 0.350 | 77.849 | -36.272 | proved | 27 | 63.992 | n/a |
| 3pch | STRUCTURE OF PROTOCATECHUATE 3,4-DIOXYGENASE COMPLEXED WITH | oxidoreductase | metal_mediated | success | 0.353 | 149.395 | -37.020 | proved | 27 | 45.414 | n/a |
| 3pcj | STRUCTURE OF PROTOCATECHUATE 3,4-DIOXYGENASE COMPLEXED WITH | oxidoreductase | metal_mediated | success | 0.557 | 69.073 | -63.157 | proved | 29 | 86.289 | n/a |
| 2cgr | LOCAL AND TRANSMITTED CONFORMATIONAL CHANGES ON | other | standard | success | 0.527 | 66.935 | -17.845 | proved | 5 | 4.940 | n/a |
| 1flr | 4-4-20 FAB FRAGMENT | other | standard | success | 0.352 | 76.984 | -19.061 | proved | 6 | 9.955 | n/a |
| 2gss | HUMAN GLUTATHIONE S-TRANSFERASE P1-1 IN COMPLEX WITH | transferase | standard | success | 0.478 | 69.089 | -7.974 | proved | 22 | 4.042 | n/a |
| 3gss | HUMAN GLUTATHIONE S-TRANSFERASE P1-1 IN COMPLEX WITH | transferase | multi_residue | success | 0.230 | 75.815 | -17.635 | proved | 4 | 3.510 | n/a |
| 10gs | HUMAN GLUTATHIONE S-TRANSFERASE P1-1, COMPLEX WITH TER117 | transferase | standard | success | 0.143 | 63.301 | -16.251 | proved | 6 | 8.346 | n/a |
| 2std | SCYTALONE DEHYDRATASE COMPLEXED WITH TIGHT-BINDING | lyase | standard | success | 0.091 | 68.693 | -12.233 | proved | 3 | 6.129 | n/a |
| 3std | SCYTALONE DEHYDRATASE AND CYANOCINNOLINE INHIBITOR | lyase | standard | success | 0.184 | 55.290 | -16.749 | proved | 4 | 8.665 | n/a |
| 1jaq | COMPLEX OF 1-HYDROXYLAMINE-2-ISOBUTYLMALONYL-ALA-GLY-NH2 | hydrolase | metal_mediated | success | 0.782 | 48.609 | -61.541 | proved | 27 | 45.806 | n/a |
| 1zs0 | CRYSTAL STRUCTURE OF THE COMPLEX BETWEEN MMP-8 AND A | hydrolase | metal_mediated | success | 0.882 | 63.836 | -45.636 | proved | 22 | 65.140 | n/a |
| 1zvx | CRYSTAL STRUCTURE OF THE COMPLEX BETWEEN MMP-8 AND A | hydrolase | metal_mediated | success | 0.348 | 60.526 | -46.524 | proved | 28 | 33.456 | n/a |
| 2d0k | METHIONINE-FREE MUTANT OF ESCHERICHIA COLI DIHYDROFOLATE | oxidoreductase | standard | success | 0.381 | 60.448 | -13.296 | proved | 6 | 4.053 | n/a |
| 1dhi | LONG-RANGE STRUCTURAL EFFECTS IN A SECOND-SITE REVERTANT OF | oxidoreductase | standard | success | 0.431 | 67.890 | -12.680 | proved | 6 | 4.310 | n/a |
| 2drc | INVESTIGATION OF THE FUNCTIONAL ROLE OF TRYPTOPHAN-22 IN | oxidoreductase | standard | success | 0.280 | 68.244 | -12.735 | proved | 2 | 6.383 | n/a |
| 1df8 | S45A MUTANT OF STREPTAVIDIN IN COMPLEX WITH BIOTIN | transport | standard | success | 0.192 | 63.853 | -12.061 | proved | 4 | 4.653 | n/a |
| 2f01 | EPI-BIOTIN COMPLEX WITH CORE STREPTAVIDIN | transport | multi_residue | success | 0.117 | 62.179 | -12.551 | proved | 3 | 7.134 | n/a |
| 2g8r | THE CRYSTAL STRUCTURE OF THE RNASE A- | hydrolase | standard | success | 81.554 | 70.658 | 0.066 | proved | 1 | 153.531 | n/a |
| 1tyr | TRANSTHYRETIN COMPLEX WITH RETINOIC ACID | transport | standard | success | 5.807 | 49.789 | -5.080 | proved | 23 | 2.897 | n/a |
| 1e5a | STRUCTURE OF HUMAN TRANSTHYRETIN COMPLEXED WITH | transport | standard | success | 3.424 | 68.341 | -4.272 | proved | 28 | 2.145 | n/a |
| 2g5u | HUMAN TRANSTHYRETIN (TTR) COMPLEXED WITH HYDROXYLATED | other | standard | success | 5.960 | 66.658 | -6.420 | proved | 25 | 3.688 | n/a |
| 1sv3 | STRUCTURE OF THE COMPLEX FORMED BETWEEN PHOSPHOLIPASE A2 | hydrolase | standard | success | 0.558 | 68.040 | -7.008 | proved | 25 | 2.378 | n/a |
| 1q7a | CRYSTAL STRUCTURE OF THE COMPLEX FORMED BETWEEN RUSSELL'S | hydrolase | standard | success | 0.153 | 64.062 | -10.414 | proved | 11 | 3.161 | n/a |
| 2f80 | HIV-1 PROTEASE MUTANT D30N COMPLEXED WITH INHIBITOR TMC114 | hydrolase | standard | success | 0.331 | 61.406 | -20.035 | proved | 4 | 10.786 | n/a |
| 2i0d | CRYSTAL STRUCTURE OF AD-81 COMPLEXED WITH WILD TYPE HIV-1 | hydrolase | standard | success | 0.100 | 67.759 | -17.713 | proved | 4 | 8.170 | n/a |
| 1d7j | FKBP COMPLEXED WITH 4-HYDROXY-2-BUTANONE | isomerase | standard | success | 0.304 | 54.786 | -5.142 | proved | 26 | 2.831 | n/a |

## Excluded

| PDB | Reason |
| --- | --- |
| 1m0n | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1zc9 | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1m0q | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1ppm | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1bxo | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1nje | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1tsy | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 2fzc | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 2h3e | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1d09 | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1v48 | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1b8o | RDKit produced non-finite Gasteiger charge at atom index 213 |
| 1o3p | Ligand charge count mismatch for 1o3p: charges=45 atoms=25 |
| 1hfs | Unknown element for simple charges: F |
| 2bz6 | Ligand charge count mismatch for 2bz6: charges=54 atoms=30 |
| 1loq | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1lol | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1x1z | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 4tim | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1utp | Pocket charge count mismatch for 1utp: charges=297 atoms=168 |
| 1o3f | Ligand charge count mismatch for 1o3f: charges=41 atoms=25 |
| 1k9s | RDKit produced non-finite Gasteiger charge at atom index 209 |
| 6std | Ligand charge count mismatch for 6std: charges=22 atoms=21 |
| 1o0h | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1u1b | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 2c02 | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1hi4 | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 2bzz | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1fki | Pocket charge count mismatch for 1fki: charges=157 atoms=137 |
| 1fkb | Ligand charge count mismatch for 1fkb: charges=68 atoms=65 |
| 6rnt | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1det | RDKit produced non-finite Gasteiger charge at atom index 0 |
| 1rnt | RDKit produced non-finite Gasteiger charge at atom index 0 |
