# arXiv abstract (paper4)

Title: Decision Quotient: Complexity of Exact Relevance Certification

## Abstract (MathJax, arXiv-ready)

```text
We study exact relevance certification: which coordinates of a state space suffice to determine the optimal action, and how hard it is to certify that the remaining coordinates may be discarded without changing the optimizer. For a decision problem $\mathcal{D}=(A,S,U)$ with $S=X_1 \times \cdots \times X_n$, a coordinate set $I$ is sufficient if $s_I=s'_I$ implies $\operatorname{Opt}(s)=\operatorname{Opt}(s')$. The associated optimizer quotient records the coarsest abstraction that preserves optimal-action distinctions.

Under the encoding model fixed in Section, we prove a regime hierarchy. In the static regime, [Sufficiency-Check]{.smallcaps} and [Minimum-Sufficient-Set]{.smallcaps} are coNP-complete, while [Anchor-Sufficiency]{.smallcaps} is $\Sigma_{2}^{P}$-complete. In the stochastic regime, [Stochastic-Sufficiency-Check]{.smallcaps} is PP-complete. In the sequential regime, [Sequential-Sufficiency-Check]{.smallcaps} is PSPACE-complete. Under standard assumptions, this yields strict inclusions static $\subset$ stochastic $\subset$ sequential.

We also prove a witness-checking lower bound, approximation-hardness results consisting of a shifted-family gap theorem that turns uniform factor guarantees into tautology detection and an exact set-cover equivalence on a separate explicit gadget family with approximation-ratio transfer, an integrity-competence impossibility result for exact reliability under polynomial budgets, an encoding-sensitive dichotomy (explicit-state tractability versus succinct ETH-conditioned exponential lower bounds), and six tractable subcases (bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, and coordinate symmetry).

These results also yield direct corollaries for exact simplification tasks: in the general succinct regime, exact behavior-preserving minimization has no polynomial-time worst-case solution unless $P=coNP$; no polynomial-budget solver can be simultaneously integrity-preserving and fully competent on the hard exact regime; and conservative over-specification can be a rational response when exact pruning carries exponential cost. A final hardness-conservation section ties any chosen architecture to the canonical relevant-coordinate set and makes the resulting externalization cost explicit via $\mathrm{centralDOF}+\mathrm{simplicityTax}=\mathrm{intrinsicDOF}$.

The Lean 4 development mechanizes the core reductions and theorem statements, with 49744 lines, 2152 theorems/lemmas, and 189 proof files.
```

## Abstract (Unicode, Zenodo-ready)

```text
We study exact relevance certification: which coordinates of a state space suffice to determine the optimal action, and how hard it is to certify that the remaining coordinates may be discarded without changing the optimizer. For a decision problem 𝒟 = (A, S, U) with S = X₁ × ⋯ × Xₙ, a coordinate set I is sufficient if s_I = s′_(I) implies Opt (s) = Opt (s′). The associated optimizer quotient records the coarsest abstraction that preserves optimal-action distinctions.

Under the encoding model fixed in Section, we prove a regime hierarchy. In the static regime, SUFFICIENCY-CHECK and MINIMUM-SUFFICIENT-SET are coNP-complete, while ANCHOR-SUFFICIENCY is Σ₂ᴾ-complete. In the stochastic regime, STOCHASTIC-SUFFICIENCY-CHECK is PP-complete. In the sequential regime, SEQUENTIAL-SUFFICIENCY-CHECK is PSPACE-complete. Under standard assumptions, this yields strict inclusions static ⊂ stochastic ⊂ sequential.

We also prove a witness-checking lower bound, approximation-hardness results consisting of a shifted-family gap theorem that turns uniform factor guarantees into tautology detection and an exact set-cover equivalence on a separate explicit gadget family with approximation-ratio transfer, an integrity-competence impossibility result for exact reliability under polynomial budgets, an encoding-sensitive dichotomy (explicit-state tractability versus succinct ETH-conditioned exponential lower bounds), and six tractable subcases (bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, and coordinate symmetry).

These results also yield direct corollaries for exact simplification tasks: in the general succinct regime, exact behavior-preserving minimization has no polynomial-time worst-case solution unless P = coNP; no polynomial-budget solver can be simultaneously integrity-preserving and fully competent on the hard exact regime; and conservative over-specification can be a rational response when exact pruning carries exponential cost. A final hardness-conservation section ties any chosen architecture to the canonical relevant-coordinate set and makes the resulting externalization cost explicit via centralDOF + simplicityTax = intrinsicDOF.

The Lean 4 development mechanizes the core reductions and theorem statements, with 49744 lines, 2152 theorems/lemmas, and 189 proof files.
```
