# Response to Reviewer 2 (Major Comments)

We thank the reviewer for these careful comments. We address each in turn.

---

## On the stochastic row uniformity (Comment 1)

We agree that the stochastic regime contains two parallel questions with distinct complexity profiles. The regime matrix already presents these as separate rows (preservation vs. decisiveness), and we acknowledge that our presentation in the abstract and introduction did not sufficiently foreground this distinction. We have revised the manuscript to clarify that:

- **Preservation** (stochastic sufficiency) generalizes static sufficiency under probability, with explicit-state P + full-support inheritance of static classifications + partial support-sensitive lemmas
- **Decisiveness** is a separate predicate with explicit-state P + PP-hardness (succinct) + NP^PP membership (anchor/minimum variants)

The reviewer is correct that the escalation static → stochastic → sequential is not uniform across all predicates within the stochastic regime. We have amended the abstract and introduction to reflect this more accurately.

---

## On the full-support distinction (Comment 2)

We note that the manuscript already contains the qualifications the reviewer requests. The abstract states:

> "Under full support, the natural base case, we give complete classification. The obstruction to general-distribution classification is structural..."

And in the regime hierarchy section, Table 2 explicitly marks "open" entries for general-distribution preservation variants.

We appreciate the reviewer's insistence on clarity here, and we have further emphasized these qualifications in the abstract and introduction to ensure no reader misses them.

---

## On mechanization claims (Comment 3)

We similarly note that the manuscript already contains the exact language the reviewer requests. The abstract states:

> "Full oracle-machine formalization remains outside current formalized complexity capabilities; the artifact provides independent verification of the finite combinatorial core underlying the oracle-class arguments..."

The introduction and conclusion contain parallel language. We agree this framing is important and have ensured it is prominently visible.

---

## On engineering/simplicity-tax interpretation (Comment 4)

The reviewer correctly identifies that the trilemma is a direct corollary of coNP-hardness, and asks for more rigorous justification of the cost-model assumptions.

We note that the simplicity-tax section already contains the formalization the reviewer requests:
- Definitions for relevant support, intrinsic/central degrees of freedom, and simplicity tax (Section 10.3)
- The simplicity-tax identity (Proposition 10.3.1): centralDOF + simplicityTax = intrinsicDOF
- Linear externalization under per-site cost model (Proposition 10.3.2)
- Amortization threshold theorem (Theorem 10.3.3): N > H_central / (lambda * simplicityTax)

These results establish the mathematical framework. To address the reviewer's concern about the examples being thin, we have expanded Section 10.1 with two additional concrete case studies:
- A toy POMDP planning example showing how preservation/decisiveness manifests in classical planning
- A reinforcement learning hyperparameter tuning example showing the same structure in learning theory

These examples ground the abstract formalism in canonical engineering settings.

---

We have addressed Comments 1-3 in the revision. We are happy to discuss Comment 4 further and will provide a more formalized treatment in the next revision.
