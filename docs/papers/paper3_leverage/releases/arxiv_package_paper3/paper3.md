# Paper: Leverage, Structural Rank, and Thermodynamic Selection in Information-Processing Systems

**Status**: Draft-ready | **Lean**: 70700 lines, 3155 theorems

---

## Abstract

We study leverage as a structural invariant of information-processing systems. For any system $A$ with capabilities $\mathrm{Cap}(A) > 0$, we prove a Five-Way Equivalence: $$\begin{aligned}
\mathrm{DOF}(A) = 1
&\;\iff\; \text{max leverage } L = |\mathrm{Cap}|/\mathrm{DOF}
\;\iff\; \text{single source of truth} \\
&\;\iff\; \mathrm{srank} = 1
\;\iff\; \text{tractable sufficiency} \\
&\;\iff\; \text{minimum thermodynamic cost}.
\end{aligned}$$

We also prove that the England replication inequality, $$\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k,$$ reduces to counting: a $k$-variable system has $2^k$ states, and the finite inequality $k \leq 2^{k-1}$ yields the entropy gap under Landauer calibration. The thermodynamic bound $E_{\mathrm{decision}} \geq \mathrm{srank} \cdot k_B T \ln 2$ then follows from the same calibration (BA7).

Within the explicit thermodynamic model used in the paper (Landauer calibration, finite energy, finite signal speed, and nontrivial state space), we derive a physical no-go theorem for polynomial collapse (LP38). Engineering corollaries follow directly: $\mathrm{DOF} = n$ is isomorphic to a series reliability system with $n$ failure points; expected modification cost is proportional to $\mathrm{DOF}$; and minimum edit-energy equals $k_B T \ln 2 \cdot \mathrm{DOF}(A)$.

All theorems are machine-checked in Lean 4 with no `sorry` placeholders. The only explicit physics premises are Landauer calibration, the Second Law (non-negative entropy production), and the Thermodynamic Uncertainty Relation (Barato-Seifert 2015). An assumption ledger records the dependency structure of each physically grounded result.

Keywords: thermodynamics, Landauer principle, entropy production, information theory, structural rank, formal verification, degrees of freedom


_Failed to convert lean_stats.tex_

# Introduction

This paper proves thermodynamic bounds on information-processing systems from first principles. The central result is the Five-Way Equivalence (Theorem [\[thm:five-way\]](#thm:five-way){reference-type="ref" reference="thm:five-way"}): five independent scientific frameworks (engineering, epistemics, information theory, computational complexity, and statistical physics) all characterize the same structural property, $\mathrm{DOF} = 1$. Software architecture serves as the application domain; the underlying physics is general. All claims are verified in Lean 4 with zero `sorry` placeholders.

## Formalization Statistics

::: center
  **Metric**                                    **Value**
  ------------------------------------- -----------------
  Lines of Lean 4 (local layer)                      4120
  Theorems/lemmas (local layer)                       210
  `sorry` placeholders (local layer)                    0
  Proof files (local layer)                            17
  Total imported Lean lines                         70700
  Total theorem/lemma statements                     3155
  Total imported `sorry` placeholders                   0
  Total proof files in import closure                 244
  **Dependencies (live imports)**       
  `AbstractClassSystem`                   lines, theorems
  `Ssot`                                  lines, theorems
  `DecisionQuotient`                      lines, theorems
:::

All proofs build with `lake build`. Axiom dependencies verified via `#print axioms`: `propext`, `Quot.sound`, `Classical.choice`.

## Central Result

::: theorem
[]{#thm:five-way label="thm:five-way"} For any architecture $A$ with $\mathrm{Cap}(A) > 0$, the following are equivalent: $$\begin{aligned}
&\underbrace{\mathrm{DOF}(A) = 1}_{\text{single source}}
\;\iff\;
\underbrace{L(A) \text{ maximal}}_{\text{engineering}}
\;\iff\;
\underbrace{\mathrm{SSOT}(A)}_{\text{epistemics}} \\
&\iff\;
\underbrace{\mathrm{srank}(A) = 1}_{\text{information}}
\;\iff\;
\underbrace{\text{tractable sufficiency}}_{\text{complexity}}
\;\iff\;
\underbrace{\text{min thermo cost}}_{\text{physics}}
\end{aligned}$$
:::

Proved in Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"}. Machine-checked via `Leverage` $\to$ `DecisionQuotient` $\to$ `Ssot` $\to$ Mathlib.

## Contributions

1.  **DOF-Reliability Isomorphism (Theorem [\[thm:dof-reliability\]](#thm:dof-reliability){reference-type="ref" reference="thm:dof-reliability"}):** Architecture with $n$ DOF is isomorphic to a series reliability system with $n$ components.

2.  **Leverage-Error Tradeoff (Theorem [\[thm:leverage-error\]](#thm:leverage-error){reference-type="ref" reference="thm:leverage-error"}):** $L(A_1) > L(A_2) \implies P_{\mathrm{error}}(A_1) < P_{\mathrm{error}}(A_2)$ at equal capabilities.

3.  **Modification Complexity Gap (Theorem [\[thm:leverage-gap\]](#thm:leverage-gap){reference-type="ref" reference="thm:leverage-gap"}):** Expected modification cost is proportional to DOF at fixed capabilities.

4.  **Physical Edit-Energy Floor (Theorem [\[thm:physical-energy-floor\]](#thm:physical-energy-floor){reference-type="ref" reference="thm:physical-energy-floor"}):** Minimum edit-energy $= j_\mathcal{M} \cdot \mathrm{DOF}(A)$ under Landauer calibration. No P $\neq$ coNP assumption.

5.  **England Replication Inequality (Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}):** $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$. Proof: a $k$-variable system has $2^k$ states (counting); entropy gap follows from $k \leq 2^{k-1}$ (induction).

6.  **Optimal Architecture (Theorem [\[thm:optimal\]](#thm:optimal){reference-type="ref" reference="thm:optimal"}):** $f(R) = \arg\max_{A:\,\mathrm{Cap}(A) \supseteq R} L(A)$ minimizes expected error and is computable.

#### What is new.

`AbstractClassSystem` and `Ssot` establish two of the five equivalences. This paper adds leverage maximization and proves it equivalent to the remaining three. The England inequality is new: the only physics input is the Landauer constant $k_B$; the bound reduces to $k \leq 2^{k-1}$.

## Dependency Chain

1.  `AbstractClassSystem`: axis orthogonality $\to$ error independence (Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"})

2.  `Ssot`: DOF $= 1$ $\iff$ coherence $\to$ second equivalence

3.  `DecisionQuotient`: tractability boundary and thermodynamic lift $\to$ fourth and fifth equivalences

4.  `Leverage` (this paper): leverage maximization $\iff$ all of the above

## Scope

Leverage measures capability-to-DOF ratio. Performance, security, and other dimensions are orthogonal. Error independence is derived from axis orthogonality in `AbstractClassSystem`, not assumed. Physical claims use one explicit constant ($j_\mathcal{M} > 0$) and no additional axioms beyond the Landauer calibration.

## Organization

Section [\[foundations\]](#foundations){reference-type="ref" reference="foundations"} defines Architecture, DOF, Capabilities, and Leverage. Section [\[probability-model\]](#probability-model){reference-type="ref" reference="probability-model"} derives the error model. Section [\[main-theorems\]](#main-theorems){reference-type="ref" reference="main-theorems"} proves decidability and optimality. Section [\[instances\]](#instances){reference-type="ref" reference="instances"} gives leverage instances. Section [\[five-way-equivalence\]](#five-way-equivalence){reference-type="ref" reference="five-way-equivalence"} proves the five-way equivalence and the England inequality. Section [\[appendix-lean\]](#appendix-lean){reference-type="ref" reference="appendix-lean"} describes the Lean mechanization.


# Foundations

We formalize the core concepts: architecture state spaces, degrees of freedom, capabilities, and leverage.

## Architecture State Space

::: definition
[]{#def:architecture label="def:architecture"} An *architecture* is a tuple $A = (C, S, T, R)$ where:

-   $C$ is a finite set of *components* (modules, services, endpoints, etc.)

-   $S = \prod_{c \in C} S_c$ is the *state space* (product of component state spaces)

-   $T : S \to \mathcal{P}(S)$ defines valid *transitions* (state changes)

-   $R$ is a set of *requirements* the architecture must satisfy
:::

**Intuition:** An architecture consists of components, each with a state space. The total state space is the product of component spaces. Transitions define how the system can evolve.

::: example
-   $C = \{U, O, P\}$ where $U=\text{UserService}$, $O=\text{OrderService}$, $P=\text{PaymentService}$

-   $S_{\text{UserService}} = \text{UserDB} \times \text{Endpoints} \times \text{Config}$

-   Similar for other services

-   $S = S_{\text{UserService}} \times S_{\text{OrderService}} \times S_{\text{PaymentService}}$
:::

## Degrees of Freedom

::: definition
[]{#def:dof label="def:dof"} The *degrees of freedom* of architecture $A = (C, S, T, R)$ is a natural number $\text{DOF}(A) \in \mathbb{N}$ counting the independent axes of variation in the state space. Formally, it is the cardinality of a minimal complete orthogonal axis set for $S$ (as established in `AbstractClassSystem`). In the Lean formalization, `Architecture.dof` is a field of type $\mathbb{N}$.
:::

**Operational meaning:** DOF counts independent modification points. If $\text{DOF}(A) = n$, then $n$ independent changes can be made to the architecture.

::: proposition
[]{#prop:dof-additive label="prop:dof-additive"} For disjoint architectures $A_1 = (C_1, S_1, T_1, R_1)$ and $A_2 = (C_2, S_2, T_2, R_2)$ with $C_1 \cap C_2 = \emptyset$: $$\text{DOF}(A_1 \oplus A_2) = \text{DOF}(A_1) + \text{DOF}(A_2)$$ where $A_1 \oplus A_2 = (C_1 \cup C_2, S_1 \times S_2, T_1 \times T_2, R_1 \cup R_2)$.
:::

::: proof
*Proof.* Components are disjoint ($C_1 \cap C_2 = \emptyset$), so the axis sets of $A_1$ and $A_2$ are disjoint. The axis set of $A_1 \oplus A_2$ is their disjoint union, giving $\text{DOF}(A_1 \oplus A_2) = |\text{axes}(A_1)| + |\text{axes}(A_2)| = \text{DOF}(A_1) + \text{DOF}(A_2)$. ◻
:::

::: example
1.  **Monolith:** Single deployment unit $\to$ DOF $= 1$

2.  **$n$ Microservices:** $n$ independent services $\to$ DOF $= n$

3.  **Copied Code:** Code duplicated to $n$ locations $\to$ DOF $= n$ (each copy independent)

4.  **SSOT:** Single source, $n$ derived uses $\to$ DOF $= 1$ (only source is independent)

5.  **$k$ API Endpoints:** $k$ independent definitions $\to$ DOF $= k$

6.  **$m$ Config Parameters:** $m$ independent settings $\to$ DOF $= m$
:::

## Capabilities

::: definition
[]{#def:capabilities label="def:capabilities"} The *capability set* of architecture $A$ is: $$\text{Cap}(A) = \{r \in R \mid A \text{ satisfies } r\}$$
:::

**Examples of capabilities:**

-   "Support horizontal scaling"

-   "Provide type provenance"

-   "Enable independent deployment"

-   "Satisfy single source of truth for class definitions"

-   "Allow polyglot persistence"

::: definition
[]{#def:satisfies label="def:satisfies"} Architecture $A$ *satisfies* requirement $r$ (written $A \vDash r$) if there exists an execution trace in $(S, T)$ that meets $r$'s specification.
:::

## Leverage

::: definition
[]{#def:leverage label="def:leverage"} The *leverage* of architecture $A$ is: $$L(A) = \frac{|\text{Cap}(A)|}{\text{DOF}(A)}$$
:::

**Special cases:**

1.  **Unbounded Leverage:** As $|\text{Cap}(A)|$ grows for fixed $\text{DOF}(A) = 1$, leverage grows without bound. This is the metaprogramming regime: a single source with an unbounded number of derived capabilities. In any fixed finite model, $L$ is a positive rational; "$L = \infty$" is a shorthand for this limit.

2.  **Unit Leverage ($L = 1$):** Linear relationship (n capabilities from n DOF)

3.  **Sublinear Leverage ($L < 1$):** Antipattern (more DOF than capabilities)

::: example
-   **SSOT:** DOF $= 1$, Cap $= \{F, \text{uses of } F\}$ where each new use adds a capability\
    $\Rightarrow L$ grows without bound as derivations accumulate

-   **Scattered Code (n copies):** DOF $= n$, Cap $= \{F\}$\
    $\Rightarrow L = 1/n$ (antipattern!)

-   **Generic REST Endpoint:** DOF $= 1$, Cap $= \{\text{serve } n \text{ use cases}\}$\
    $\Rightarrow L = n$

-   **Specific Endpoints:** DOF $= n$, Cap $= \{\text{serve } n \text{ use cases}\}$\
    $\Rightarrow L = 1$
:::

::: definition
[]{#def:dominance label="def:dominance"} Architecture $A_1$ *dominates* $A_2$ (written $A_1 \succeq A_2$) if:

1.  $\text{Cap}(A_1) \supseteq \text{Cap}(A_2)$ (at least same capabilities)

2.  $L(A_1) \geq L(A_2)$ (at least same leverage)

$A_1$ *strictly dominates* $A_2$ (written $A_1 \succ A_2$) if $A_1 \succeq A_2$ with at least one inequality strict.
:::

## Modification Complexity

::: definition
[]{#def:mod-complexity label="def:mod-complexity"} For requirement change $\delta R$, the *modification complexity* is: $$M(A, \delta R) = \text{expected number of independent changes to implement } \delta R$$
:::

::: theorem
[]{#thm:mod-bound label="thm:mod-bound"} For all architectures $A$ and requirement changes $\delta R$: $$M(A, \delta R) \leq \text{DOF}(A)$$ with equality when $\delta R$ affects all components.
:::

::: proof
*Proof.* By Definition [\[def:dof\]](#def:dof){reference-type="ref" reference="def:dof"}, DOF counts independent axes. A requirement change $\delta R$ decomposes into sub-changes along individual axes; sub-changes along distinct axes are independent. Therefore the number of independent changes is at most the number of axes, which is $\text{DOF}(A)$. Equality holds when $\delta R$ has a non-trivial projection onto every axis. ◻
:::

::: example
Consider changing a structural fact $F$ with $n$ use sites:

-   **SSOT:** $M = 1$ (change at source, derivations update automatically)

-   **Scattered:** $M = n$ (must change each copy independently)
:::

## Formalization in Lean

All definitions in this section are formalized in `Leverage/Foundations.lean`:

-   `Architecture`: Structure with components, state, transitions, requirements

-   `Architecture.leverage`: Leverage metric

-   `Architecture.dominates`: Dominance relation

-   `compose_dof`: Proposition [\[prop:dof-additive\]](#prop:dof-additive){reference-type="ref" reference="prop:dof-additive"}

-   `modification_eq_dof`: Theorem [\[thm:mod-bound\]](#thm:mod-bound){reference-type="ref" reference="thm:mod-bound"}


# Probability Model

We derive the relationship between DOF and error probability from `AbstractClassSystem`'s axis independence theorem.

## Error Independence from Axis Orthogonality

The independence of errors is not an axiom; it is a consequence of axis orthogonality proved in `AbstractClassSystem` [@paper1_typing_discipline].

::: theorem
[]{#thm:error-independence label="thm:error-independence"} If axes $\{A_1, \ldots, A_n\}$ are orthogonal (`AbstractClassSystem`, Theorem `minimal_complete_unique_orthogonal`), then errors along each axis are statistically independent.
:::

::: proof
*Proof.* By Definition [\[def:architecture\]](#def:architecture){reference-type="ref" reference="def:architecture"}, the state space is the product $S = \prod_{c \in C} S_c$. `AbstractClassSystem`'s orthogonality theorem (`minimal_complete_unique_orthogonal`) establishes that a minimal complete axis set partitions the state space into independent dimensions: axis $A_i$ is a projection onto a factor of the product, and $A_i \perp A_j$ means the projection onto factor $i$ carries no information about factor $j$.

An *error along axis $A_i$* is the event $E_i$ that the value on factor $i$ deviates from specification. Since the factors are independent dimensions of the product space, the probability measure factorizes: $$P(E_i \mid \text{state of factors } \neq i) = P(E_i).$$ This is the standard characterization of independence for product probability spaces: the marginal on factor $i$ is independent of the marginal on any other factor. Therefore $P(E_i \cap E_j) = P(E_i) \cdot P(E_j)$ for all $i \neq j$, and by induction the $n$ events $E_1, \ldots, E_n$ are mutually independent. ◻
:::

::: corollary
[]{#cor:dof-errors label="cor:dof-errors"} DOF$(A) = n$ implies $n$ independent sources of error, each with probability $p$.
:::

::: proof
*Proof.* DOF counts independent axes (Section [\[foundations\]](#foundations){reference-type="ref" reference="foundations"}, Definition [\[def:dof\]](#def:dof){reference-type="ref" reference="def:dof"}). By Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"}, independent axes have independent errors. ◻
:::

::: theorem
[]{#thm:error-compound label="thm:error-compound"} For a system to be correct, all $n$ independent axes must be error-free. Errors compound multiplicatively.
:::

::: proof
*Proof.* By `Ssot`'s coherence theorem (`oracle_arbitrary`), incoherence in any axis violates system correctness. An error in axis $A_i$ introduces incoherence along $A_i$. Therefore, correctness requires $\bigwedge_{i=1}^{n} \neg\text{error}(A_i)$. By Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"}, this probability is $(1-p)^n$. ◻
:::

## Error Probability Formula

::: theorem
[]{#thm:error-prob label="thm:error-prob"} For architecture with $n$ DOF and per-component error rate $p$: $$P_{\text{error}}(n) = 1 - (1-p)^n$$
:::

::: proof
*Proof.* By Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"} (derived from `AbstractClassSystem`'s orthogonality), each DOF has independent error probability $p$, so each is correct with probability $(1-p)$. By Theorem [\[thm:error-compound\]](#thm:error-compound){reference-type="ref" reference="thm:error-compound"}, all $n$ DOF must be correct: $$P_{\text{correct}}(n) = (1-p)^n$$ Therefore: $$P_{\text{error}}(n) = 1 - P_{\text{correct}}(n) = 1 - (1-p)^n$$ ◻
:::

::: corollary
[]{#cor:linear-approx label="cor:linear-approx"} For fixed $p>0$, the linear expected-error model and the exact series model induce the same ordering on architectures by DOF.
:::

::: proof
*Proof.* By Theorem [\[thm:approx-bound\]](#thm:approx-bound){reference-type="ref" reference="thm:approx-bound"}, ordering equivalence is exact under the discrete model used in the mechanization. ◻
:::

::: corollary
[]{#cor:dof-monotone label="cor:dof-monotone"} For architectures $A_1, A_2$: $$\text{DOF}(A_1) < \text{DOF}(A_2) \implies P_{\text{error}}(A_1) < P_{\text{error}}(A_2)$$
:::

::: proof
*Proof.* $P_{\text{error}}(n) = 1 - (1-p)^n$ is strictly increasing in $n$ for $p \in (0,1)$. ◻
:::

## Expected Errors

::: theorem
[]{#thm:expected-errors label="thm:expected-errors"} Expected number of errors in architecture $A$: $$\mathbb{E}[\text{\# errors}] = p \cdot \text{DOF}(A)$$
:::

::: proof
*Proof.* By linearity of expectation: $$\mathbb{E}[\text{\# errors}] = \sum_{i=1}^{\text{DOF}(A)} P(\text{error in DOF}_i) = \sum_{i=1}^{\text{DOF}(A)} p = p \cdot \text{DOF}(A)$$ ◻
:::

::: example
Assume $p = 0.01$ (1% per-component error rate):

-   DOF $= 1$: $P_{\text{error}} = 1 - 0.99 = 0.01$ (1%)

-   DOF $= 10$: $P_{\text{error}} = 1 - 0.99^{10} \approx 0.096$ (9.6%)

-   DOF $= 100$: $P_{\text{error}} = 1 - 0.99^{100} \approx 0.634$ (63.4%)
:::

## Connection to Reliability Theory

The error model has a direct interpretation in classical reliability theory [@patterson2013computer], connecting software architecture to a mature mathematical framework with 60+ years of theoretical development.

::: theorem
[]{#thm:dof-reliability label="thm:dof-reliability"} An architecture with DOF $= n$ is *isomorphic* to a series reliability system with $n$ components. The isomorphism:

1.  **Preserves ordering:** If $\text{DOF}(A_1) < \text{DOF}(A_2)$, then $P_{\text{error}}(A_1) < P_{\text{error}}(A_2)$

2.  **Is invertible:** Round-trip mapping preserves DOF exactly

3.  **Connects domains:** $P_{\text{error}}(n) = 1 - R_{\text{series}}(n)$ where $R_{\text{series}}(n) = (1-p)^n$
:::

**Interpretation:** Each DOF is a "component" that must work correctly. This is the reliability analog of Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"}, which derives error independence from axis orthogonality.

::: theorem
[]{#thm:approx-bound label="thm:approx-bound"} For architectures $A_1,A_2$ with equal capabilities and error rate $p>0$, the linear model and the exact series model induce the same DOF ordering: $$\text{DOF}(A_1)\le \text{DOF}(A_2)
\iff
(\mathbb{E}[\text{errors}(A_1)]\cdot d)\le(\mathbb{E}[\text{errors}(A_2)]\cdot d),$$ where $d$ is the denominator of the discrete error-rate representation.
:::

::: proof
*Proof.* This is theorem `ordering_equivalence_exact` in `Leverage/Probability.lean`. ◻
:::

**Linear Approximation Justification:** For small $p$ (the software engineering regime where $p \approx 0.01$), the linear model $P_{\text{error}} \approx n \cdot p$ is:

1.  Order-equivalent to the exact series model in the mechanized discrete representation

2.  Monotone in DOF for fixed positive error rate

3.  Cleanly formalized in natural-number arithmetic

## Epistemic Grounding

The probability model is not axiomatic; it is derived from the epistemic foundations in `AbstractClassSystem` and `Ssot`:

1.  `AbstractClassSystem` proves axis orthogonality (`minimal_complete_unique_orthogonal`)

2.  **Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"}** derives error independence from orthogonality

3.  `Ssot` establishes that DOF = 1 guarantees coherence (`oracle_arbitrary`)

4.  **Theorem [\[thm:error-compound\]](#thm:error-compound){reference-type="ref" reference="thm:error-compound"}** connects errors to incoherence

This derivation chain ensures the probability model rests on proved foundations, not assumed axioms.

## Leverage Gap: Quantitative Predictions

The leverage framework provides *quantitative*, empirically testable predictions about modification costs.

::: theorem
[]{#thm:leverage-gap label="thm:leverage-gap"} For architectures with equal capabilities, the modification cost ratio equals the inverse leverage ratio: $$\frac{\text{ModCost}(A_2)}{\text{ModCost}(A_1)} = \frac{\text{DOF}(A_2)}{\text{DOF}(A_1)} = \frac{L(A_1)}{L(A_2)}$$
:::

::: theorem
[]{#thm:testable-prediction label="thm:testable-prediction"} If architecture $A_1$ has $n\times$ lower DOF than $A_2$ (for equal capabilities), then $A_1$ requires $n\times$ fewer expected modifications. This is empirically testable against PR/commit data.
:::

::: corollary
[]{#cor:dof-ratio label="cor:dof-ratio"} The ratio of expected errors between two architectures equals the ratio of their DOF: $$\frac{\mathbb{E}[\text{errors}(A_2)]}{\mathbb{E}[\text{errors}(A_1)]} = \frac{\text{DOF}(A_2)}{\text{DOF}(A_1)}$$
:::

These theorems transform architectural intuitions into testable hypotheses. A claim that "Pattern X is 3× better than Pattern Y" can be verified by comparing DOF and measuring modification frequency in real codebases.

## Formalization

Formalized in `Leverage/Probability.lean`:

-   `dof_reliability_isomorphism`: Theorem [\[thm:dof-reliability\]](#thm:dof-reliability){reference-type="ref" reference="thm:dof-reliability"} (the central isomorphism)

-   `isomorphism_preserves_failure_ordering`: Ordering preservation

-   `isomorphism_roundtrip`: Invertibility proof

-   `ordering_equivalence_exact`: Theorem [\[thm:approx-bound\]](#thm:approx-bound){reference-type="ref" reference="thm:approx-bound"} (exact ordering equivalence)

-   `linear_model_preserves_ordering`: Linear ordering support

-   `leverage_gap`: Theorem [\[thm:leverage-gap\]](#thm:leverage-gap){reference-type="ref" reference="thm:leverage-gap"}

-   `testable_modification_prediction`: Theorem [\[thm:testable-prediction\]](#thm:testable-prediction){reference-type="ref" reference="thm:testable-prediction"}

-   `dof_ratio_predicts_error_ratio`: Corollary [\[cor:dof-ratio\]](#cor:dof-ratio){reference-type="ref" reference="cor:dof-ratio"}

-   `lower_dof_lower_errors`: Corollary [\[cor:dof-monotone\]](#cor:dof-monotone){reference-type="ref" reference="cor:dof-monotone"}

-   `expected_errors_from_linearity`: Theorem [\[thm:expected-errors\]](#thm:expected-errors){reference-type="ref" reference="thm:expected-errors"}

-   `ssot_minimal_errors`: SSOT minimality


# Main Theorems

We prove the core results connecting leverage to error probability and architectural optimality. The theoretical structure is:

1.  **DOF-Reliability Isomorphism** (Theorem [\[thm:dof-reliability\]](#thm:dof-reliability){reference-type="ref" reference="thm:dof-reliability"}): Maps architecture to reliability theory

2.  **Leverage-Error Tradeoff** (Theorem [\[thm:leverage-error\]](#thm:leverage-error){reference-type="ref" reference="thm:leverage-error"}): Connects leverage to error probability

3.  **Physical Edit-Energy Floor** (Theorem [\[thm:physical-energy-floor\]](#thm:physical-energy-floor){reference-type="ref" reference="thm:physical-energy-floor"}): DOF controls minimum edit-energy

4.  **Optimality Criterion** (Theorem [\[thm:optimal\]](#thm:optimal){reference-type="ref" reference="thm:optimal"}): Correctness in a fixed capability class

5.  **Composition Stability** (Theorem [\[thm:composition\]](#thm:composition){reference-type="ref" reference="thm:composition"}): Composition preserves leverage lower bounds

## Recap: DOF-Reliability Isomorphism

The core theoretical contribution (stated in Section 1.3) is that DOF maps formally to series system components. This enables all subsequent results by connecting software architecture to the mature mathematical framework of reliability theory.

**Key properties of the isomorphism:**

-   **Preserves ordering:** If $\text{DOF}(A_1) < \text{DOF}(A_2)$, then $P_{\text{error}}(A_1) < P_{\text{error}}(A_2)$

-   **Invertible:** An architecture can be reconstructed from its series system representation

-   **Compositional:** $\text{DOF}(A_1 \oplus A_2) = \text{DOF}(A_1) + \text{DOF}(A_2)$ (series systems combine)

## The Leverage Maximization Principle

Given the DOF-Reliability Isomorphism, the following is a *corollary*:

::: theorem
[]{#thm:leverage-max label="thm:leverage-max"} For any architectural decision with alternatives $A_1, \ldots, A_n$ meeting capability requirements, the optimal choice maximizes leverage: $$A^* = \arg\max_{A_i} L(A_i) = \arg\max_{A_i} \frac{|\text{Capabilities}(A_i)|}{\text{DOF}(A_i)}$$
:::

**Note:** This is *not* the central theorem; it is a consequence of the DOF-Reliability Isomorphism. The isomorphism is the deep result; "maximize leverage" is the actionable heuristic derived from it.

## Leverage-Error Tradeoff

::: theorem
[]{#thm:leverage-error label="thm:leverage-error"} For architectures $A_1, A_2$ with equal capabilities: $$\text{Cap}(A_1) = \text{Cap}(A_2) \wedge L(A_1) > L(A_2) \implies P_{\text{error}}(A_1) < P_{\text{error}}(A_2)$$
:::

::: proof
*Proof.* Given: $\text{Cap}(A_1) = \text{Cap}(A_2)$ and $L(A_1) > L(A_2)$.

Since $L(A) = |\text{Cap}(A)|/\text{DOF}(A)$ and capabilities are equal: $$\frac{|\text{Cap}(A_1)|}{\text{DOF}(A_1)} > \frac{|\text{Cap}(A_2)|}{\text{DOF}(A_2)}$$

With $|\text{Cap}(A_1)| = |\text{Cap}(A_2)|$: $$\frac{1}{\text{DOF}(A_1)} > \frac{1}{\text{DOF}(A_2)} \implies \text{DOF}(A_1) < \text{DOF}(A_2)$$

By Corollary [\[cor:dof-monotone\]](#cor:dof-monotone){reference-type="ref" reference="cor:dof-monotone"}: $$\text{DOF}(A_1) < \text{DOF}(A_2) \implies P_{\text{error}}(A_1) < P_{\text{error}}(A_2)$$ ◻
:::

**Corollary:** Maximizing leverage minimizes error probability (for fixed capabilities).

## Physical Edit-Energy Floor

::: theorem
[]{#thm:physical-energy-floor label="thm:physical-energy-floor"} Fix a physical edit model $\mathcal{M}$ with per-edit conversion constant $j_{\mathcal{M}}>0$. For any architecture $A$: $$E_{\min}(A;\mathcal{M}) = j_{\mathcal{M}}\cdot \mathrm{DOF}(A).$$ Hence $\mathrm{DOF}(A_1) < \mathrm{DOF}(A_2)$ implies $E_{\min}(A_1;\mathcal{M}) < E_{\min}(A_2;\mathcal{M})$.
:::

::: proof
*Proof.* In the mechanized model, modification complexity is definitionally $\mathrm{DOF}$. Multiplying by a positive per-edit conversion constant gives the lower bound and strict monotonicity in DOF. ◻
:::

::: corollary
[]{#cor:leverage-energy label="cor:leverage-energy"} For architectures $A_1,A_2$ with equal capabilities, if $L(A_1)>L(A_2)$ then $$E_{\min}(A_1;\mathcal{M}) < E_{\min}(A_2;\mathcal{M}),$$ and the induced energy gap is strictly positive.
:::

::: theorem
[]{#thm:physical-budget-boundary label="thm:physical-budget-boundary"} For physical model $\mathcal{M}$ and budget $B$, implementation feasibility is equivalent to clearing the floor: $$E_{\min}(A;\mathcal{M}) \le B \iff \text{feasible}(A,B,\mathcal{M}).$$ Hence $B < E_{\min}(A;\mathcal{M})$ implies infeasibility. Also, for fixed $B$ and equal capabilities, if $L(A_1)>L(A_2)$ and $A_2$ is feasible under $B$, then $A_1$ is feasible under $B$.
:::

::: corollary
[]{#cor:physical-assumption-necessity label="cor:physical-assumption-necessity"} The physical layer uses two explicit premises for no-go style infeasibility claims: positive per-edit conversion and an external budget bound. Dropping positivity admits a zero-floor countermodel. Dropping the external budget-bound premise blocks infeasibility conclusions because a feasible budget witness always exists.
:::

## Metaprogramming Dominance

::: theorem
[]{#thm:metaprog label="thm:metaprog"} For any $N \in \mathbb{N}$, there exists a metaprogramming architecture $M$ with $\text{DOF}(M) = 1$ and $L(M) \geq N$.
:::

::: proof
*Proof.* Let $M$ be a metaprogramming architecture with a single source definition ($\text{DOF}(M) = 1$) and at least $N$ derived capabilities. Then $L(M) = |\text{Cap}(M)|/1 \geq N$.

**Modeling note.** In any fixed finite model, $L(M)$ is a positive rational. The colloquial claim "leverage $= \infty$" means: for any bound $N$, the architecture can be extended to achieve $L \geq N$ while keeping $\text{DOF} = 1$. The Lean formalization proves the for-all-$N$ version (L36). ◻
:::

## Architectural Decision Criterion

::: theorem
[]{#thm:optimal label="thm:optimal"} Given requirements $R$, architecture $A^*$ is optimal in its capability class if and only if:

1.  $\text{Cap}(A^*) \supseteq R$ (feasibility)

2.  $\forall A'$ with $\text{Cap}(A') = \text{Cap}(A^*)$ and $\text{Cap}(A') \supseteq R$: $L(A^*) \geq L(A')$ (maximality in fixed capability class)
:::

::: proof
*Proof.* ($\Leftarrow$) Suppose $A^*$ satisfies (1) and (2). Then $A^*$ is feasible and has maximum leverage among feasible architectures in the same capability class. By Theorem [\[thm:leverage-error\]](#thm:leverage-error){reference-type="ref" reference="thm:leverage-error"}, this minimizes error probability in that class.

($\Rightarrow$) If (1) fails, $A^*$ is infeasible. If (2) fails, there exists $A'$ in the same capability class with $L(A') > L(A^*)$, so $P_{\text{error}}(A') < P_{\text{error}}(A^*)$ by Theorem [\[thm:leverage-error\]](#thm:leverage-error){reference-type="ref" reference="thm:leverage-error"}, contradicting optimality. ◻
:::

**Decision Procedure:**

1.  Enumerate candidate architectures $\{A_1, \ldots, A_n\}$

2.  Filter: Keep only $A_i$ with $\text{Cap}(A_i) \supseteq R$

3.  Optimize: Choose $A^* = \arg\max_i L(A_i)$

## Leverage Composition

::: theorem
[]{#thm:composition label="thm:composition"} For modular architecture $A = A_1 \oplus A_2$ with disjoint components:

1.  $\text{DOF}(A) = \text{DOF}(A_1) + \text{DOF}(A_2)$

2.  if $L(A_1)\ge 1$ and $L(A_2)\ge 1$, then $L(A)\ge 1$
:::

::: proof
*Proof.* (1) By Proposition [\[prop:dof-additive\]](#prop:dof-additive){reference-type="ref" reference="prop:dof-additive"}.

\(2\) If $L(A_i)\ge 1$, then $|\text{Cap}(A_i)|\ge \text{DOF}(A_i)$ for $i=1,2$. Summing both inequalities gives $$|\text{Cap}(A_1)|+|\text{Cap}(A_2)|\ge \text{DOF}(A_1)+\text{DOF}(A_2),$$ which is exactly $L(A)\ge 1$ under additive composition. ◻
:::

**Interpretation:** Composition preserves a baseline leverage floor under the stated assumptions.

::: remark
[]{#rem:composition-breaks-ssot label="rem:composition-breaks-ssot"} Theorem [\[thm:composition\]](#thm:composition){reference-type="ref" reference="thm:composition"} preserves leverage *floors*, but composing two SSOT architectures breaks SSOT: if $\mathrm{DOF}(A_1) = \mathrm{DOF}(A_2) = 1$, then $\mathrm{DOF}(A_1 \oplus A_2) = 2$. This is formalized as `compose_breaks_ssot` in `Leverage/BridgeToDQ.lean`. The composition tax is unavoidable: distributed systems consisting of $k$ independent SSOT components have $\mathrm{DOF} = k$, placing them in the coNP-hard regime (srank $= k > 1$) with mandatory thermodynamic cost per decision cycle.
:::

## Formalization

Main optimization theorems are formalized in `Leverage/Theorems.lean`; physical edit-energy theorems are formalized in `Leverage/Physical.lean`:

-   L29: Theorem [\[thm:leverage-error\]](#thm:leverage-error){reference-type="ref" reference="thm:leverage-error"}

-   L1 and L5: Theorem [\[thm:physical-energy-floor\]](#thm:physical-energy-floor){reference-type="ref" reference="thm:physical-energy-floor"}

-   L3 and L6: Corollary [\[cor:leverage-energy\]](#cor:leverage-energy){reference-type="ref" reference="cor:leverage-energy"}

-   L2 and L4: Theorem [\[thm:physical-budget-boundary\]](#thm:physical-budget-boundary){reference-type="ref" reference="thm:physical-budget-boundary"}

-   L7 and L8: Corollary [\[cor:physical-assumption-necessity\]](#cor:physical-assumption-necessity){reference-type="ref" reference="cor:physical-assumption-necessity"}

-   L35 and L36: Theorem [\[thm:metaprog\]](#thm:metaprog){reference-type="ref" reference="thm:metaprog"}

-   L34 and L38: Theorem [\[thm:optimal\]](#thm:optimal){reference-type="ref" reference="thm:optimal"}

-   L18 and L19: Theorem [\[thm:composition\]](#thm:composition){reference-type="ref" reference="thm:composition"}

-   `Leverage/Physical.lean`: physical edit-energy floor theorems

## Integration with Dependencies

The leverage framework provides the unifying theory for results proved in `AbstractClassSystem` and `Ssot`:

::: theorem
[]{#thm:paper1-integration label="thm:paper1-integration"} Nominal typing dominance from `AbstractClassSystem` [@paper1_typing_discipline] is an instance of leverage maximization:

-   Nominal typing adds 4 B-dependent capabilities over duck typing

-   DOF remains fixed in the mechanized typing instance

-   Therefore nominal typing has strictly higher leverage
:::

::: theorem
[]{#thm:paper2-integration label="thm:paper2-integration"} SSOT dominance from `Ssot` [@paper2_ssot] is an instance of leverage maximization:

-   SSOT fixes DOF at 1 for a structural fact

-   Non-SSOT replication yields DOF $= n$ for the same fact

-   Therefore leverage improves by factor $n$
:::

These integration theorems are formalized in:

-   `Leverage/Typing.lean`

-   `Leverage/SSOT.lean`


# Five-Way Equivalence

This section establishes the central result: five independent characterizations of the same architectural property (DOF = 1 / SSOT) derived from five distinct first-principles frameworks. All proofs are machine-checked in Lean 4.

## Framework 1: Engineering Optimization (Leverage)

::: formal
**Theorem 5.1 (Maximum Leverage).** For any architecture $A$ with $\text{Cap}(A) > 0$, $A$ achieves maximum leverage among architectures with equal capabilities if and only if $\text{DOF}(A) = 1$.

Formally: $L(A) = \max_{A': \text{Cap}(A')=\text{Cap}(A)} L(A') \iff \text{DOF}(A) = 1$.
:::

::: informal
An architecture has the highest possible capability-to-DOF ratio exactly when it has a single degree of freedom.
:::

**Proof.**

-   (Forward) If $\text{DOF}(A) = 1$, then $L(A) = |\text{Cap}(A)|/1 = |\text{Cap}(A)|$. Any $A'$ with same capabilities has $\text{DOF}(A') \geq 1$, so $L(A') \leq |\text{Cap}(A)| = L(A)$.

-   (Backward) If $A$ has maximum leverage but $\text{DOF}(A) > 1$, construct $A'$ with $\text{DOF}(A') = 1$ and same capabilities. Then $L(A') = |\text{Cap}(A)| > |\text{Cap}(A)|/\text{DOF}(A) = L(A)$, contradiction.

## Framework 2: Architectural Epistemic Coherence (Ssot)

::: formal
**Theorem 5.2 (Coherence).** An architecture satisfies the Single Source of Truth property if and only if $\text{DOF}(A) = 1$.

Formally: $\text{SSOT}(A) \iff \text{DOF}(A) = 1$.
:::

::: informal
Epistemic coherence (one source of truth per structural fact) is equivalent to having a single degree of freedom.
:::

This is the DOF = 1 characterization from the Ssot coherence theorem.

## Framework 3: Information-Theoretic Structural Rank (DecisionQuotient)

::: formal
**Theorem 5.3 (DOF-Structural Rank Isomorphism).** For any architecture $A$, let $\text{canonicalDP}(n)$ be the canonical decision problem with $n$ boolean coordinates. Then: $$\text{srank}(\text{canonicalDP}(\text{DOF}(A))) = \text{DOF}(A)$$
:::

::: informal
The structural rank (number of relevant decision coordinates) equals the degrees of freedom. The canonical encoding uses: states as $n$ boolean coordinates, actions as either querying a coordinate or falling back, and utilities that reward correct coordinate identification.
:::

The canonical encoding is:

-   States: $\text{Fin } n \to \text{Bool}$ ($n$ binary coordinates)

-   Actions: $\text{Fin } n \oplus \text{Unit}$ (query coordinate $i$, or fallback)

-   Utility: $u(\text{inl } i, s) = 2$ if $s(i) = \text{true}$, else $0$; $u(\text{inr}\ (), s) = 1$

::: formal
**Corollary 5.4.** DOF $= 1$ if and only if srank $= 1$.

Formally: $\text{DOF}(A) = 1 \iff \text{srank}(\text{canonicalDP}(\text{DOF}(A))) = 1$.
:::

::: informal
An architecture has a single degree of freedom exactly when its canonical decision problem has minimal structural rank.
:::

## Framework 4: Computational Complexity (Tractability)

::: formal
**Theorem 5.5 (Tractability Boundary).** For the canonical decision problem with $n$ coordinates:

1.  If $n = 1$ (srank $= 1$), sufficiency checking is in P (polynomial time).

2.  If $n > 1$ (srank $> 1$), sufficiency checking is coNP-hard.
:::

::: informal
When the structural rank is 1, determining whether current information suffices for an optimal decision is computationally tractable. When the structural rank exceeds 1, the same problem is computationally intractable (coNP-hard).
:::

This connects the architectural property (DOF = 1) to computational feasibility of decision-making.

::: formal
**Corollary 5.6.** DOF $= 1$ (SSOT) is the unique architecture class for which sufficiency checking is tractable.
:::

::: informal
Among architectures, only those with a single degree of freedom admit tractable sufficiency checking.
:::

## Framework 5: Thermodynamic Selection (Statistical Physics)

::: formal
**Theorem 5.7 (Thermodynamic Selection).** Let $M$ be a thermodynamic model with Landauer calibration. For any architecture $A$ with DOF $> 1$:

1.  There exist decision instances where sufficiency checking requires $\Omega(2^{\text{DOF}})$ logical operations.

2.  Under the physical axioms of [@paper4_decision_quotient] (LP38): Landauer (EP1), nontrivial state space (EP4), finite signal speed (LP43), finite universe energy (LP44) --- P $\neq$ NP is proved, hence P $\neq$ coNP; no polynomial-time sufficiency certification exists.

3.  **(Unconditional)** Each sufficiency-check cycle incurs mandatory positive energy cost $\geq j_{\mathcal{M}} \cdot \mathrm{srank}(A) > j_{\mathcal{M}}$, proved from Landauer calibration and the DOF $=$ srank identity alone, with no complexity hypothesis.

For DOF $= 1$, the energy lower bound is $j_{\mathcal{M}}$ (the Landauer minimum).
:::

::: informal
Architectures with more than one degree of freedom necessarily incur thermodynamic costs per decision cycle. Point 3 holds unconditionally: the energy bound follows from Landauer and the DOF $=$ srank identity (L43). Point 2 holds under the physical axioms of [@paper4_decision_quotient] (LP38), which prove P $\neq$ NP, hence P $\neq$ coNP.
:::

Physical assumptions (point 3 only):

-   Positive Landauer constant ($j_{\mathcal{M}} > 0$) --- cf. Theorem [\[cor:leverage-energy\]](#cor:leverage-energy){reference-type="ref" reference="cor:leverage-energy"} (L3, L6)

-   Landauer calibration ($k_B T \ln 2$ joules per bit)

Physical edit-energy floor from Section 4.3 is a special case: DOF controls minimum edit-energy, and higher leverage implies lower energy in a fixed capability class (Theorem [\[thm:physical-energy-floor\]](#thm:physical-energy-floor){reference-type="ref" reference="thm:physical-energy-floor"}, L1, L5).

## The Five-Way Equivalence

::: formal
**Theorem 5.8 (Five-Way Equivalence).** For any architecture $A$ with $\text{Cap}(A) > 0$, the following are equivalent: $$\text{DOF}(A) = 1 \iff \text{max leverage} \iff \text{SSOT}(A) \iff \text{srank}(A) = 1 \iff \text{tractable sufficiency} \iff \text{minimum thermodynamic cost}$$
:::

::: informal
Five independent scientific frameworks---engineering optimization, epistemic coherence, information geometry, computational complexity, and statistical physics---all select the same architectural property: having exactly one degree of freedom.
:::

::: center
  **Domain**                 **Characterization of DOF $= 1$**
  -------------------------- --------------------------------------
  Engineering Optimization   Maximum leverage
  Architectural Epistemic    Single Source of Truth
  Information Geometry       Structural rank $= 1$
  Computational Complexity   Tractable sufficiency checking
  Statistical Physics        Minimum thermodynamic cost per cycle
:::

**Proof.** The equivalence follows from Theorems 5.1-5.7:

1.  $\text{DOF} = 1 \iff \text{max leverage}$: Theorem 5.1 (L31, L28)

2.  $\text{DOF} = 1 \iff \text{SSOT}$: Theorem 5.2 [@paper2_ssot]

3.  $\text{DOF} = \text{srank}$: Theorem 5.3, so $\text{DOF} = 1 \iff \text{srank} = 1$

4.  $\text{srank} = 1 \iff \text{tractable sufficiency}$: Theorem 5.5 [@paper4_decision_quotient]. Forward direction (srank $=1 \to$ polynomial) is unconditional. Backward direction (tractable $\to$ srank $=1$) holds under P $\neq$ coNP, which follows from the physical axioms of [@paper4_decision_quotient] (LP38).

5.  $\text{DOF} > 1 \iff \text{above-minimum thermodynamic cost}$: Theorem 5.7, so $\text{DOF} = 1 \iff \text{minimum thermodynamic cost}$

The transitivity of logical equivalence gives the five-way equivalence.

::: formal
**Corollary 5.9 (Convergence).** Within the `DecisionQuotient` canonical decision problem framework, the boolean-coordinate encoding (states as $n$ boolean coordinates, actions as coordinate queries or fallback, utilities rewarding correct identification) is the unique decision structure with $n = 1$ that simultaneously satisfies all five characterizations.
:::

::: informal
In the canonical decision problem representation, there is exactly one $n=1$ structure, and all five characterizations select it. This is a consequence of the equivalences, not an additional theorem. Note: there are many architectures with DOF $=1$ (monolith, single DB table, single endpoint); the uniqueness claim is scoped to the canonical decision problem encoding.
:::

**Proof.** Each characterization is equivalent to DOF $= 1$ (Theorems 5.1--5.7). Within the canonical encoding, DOF $= 1$ fixes $n = 1$, and $\mathrm{Fin}\ 1 \to \mathrm{Bool}$ has a unique element. All five requirements therefore select the same structure.

## Formalization

The following Lean 4 proofs establish the connections:

-   `Leverage/Foundations.lean`:

    -   L31, L28 (L50): DOF $= 1 \to$ max leverage

    -   L48: max leverage $\to$ DOF $= 1$

    -   L44: biconditional

-   `Leverage/BridgeToDQ.lean`:

    -   L43: DOF $=$ srank

    -   L51: DOF $= 1 \to$ srank $= 1$

    -   L46: DOF $> 1 \to$ srank $> 1$

    -   L54: DOF $> 1 \to$ mandatory energy

    -   L47: max leverage $\to$ tractable

-   `DecisionQuotient/Tractability/StructuralRank.lean`: srank $= 1 \to$ P (L56), srank $> 1 \to$ coNP-hard (L53)

-   `DecisionQuotient/ThermodynamicLift.lean`: energy lower bounds under Landauer calibration (L49)

The cross-module dependency chain is live: `Leverage` $\to$ `DecisionQuotient` $\to$ Mathlib.

## Relation to Prior Sections

This section subsumes and extends Section 4's results:

-   Theorem 4.1 (Leverage-Error Tradeoff, L29) is a consequence of the equivalence between leverage and DOF

-   Theorem 4.2 (Modification Complexity Gap, L30) follows from DOF $= 1$ minimizing modification cost

-   Theorem 4.3 (Optimal Architecture, L38, L34) is strengthened: the optimal architecture is now characterized by five independent properties

-   Theorem 4.4 (Metaprogramming Dominance, L36, L35) is a special case: unbounded derivations from a single source achieve DOF $= 1$

## England Replication Inequality (Proved)

All previously open conjectures are now proved. The England Replication Inequality is mechanized in `Leverage/BridgeToDQ.lean` (L45).

::: theorem
[]{#thm:england label="thm:england"} Let $\Delta S_{\min}(\text{srank}) = \text{srank} \cdot k_B \ln 2$ be the minimal entropy production under Landauer calibration. For a single-source architecture ($\text{srank} = 1$) and a $k$-copy replication architecture ($\text{srank} = k$): $$\Delta S_{\min}(1) + k_B \ln k \leq \Delta S_{\min}(k)$$ equivalently, $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$.
:::

::: proof
*Proof.* The gap is $(k-1) \cdot k_B \ln 2$. Since $k \leq 2^{k-1}$ (L52), taking logs gives $\ln k \leq (k-1) \ln 2$, so the gap is $\geq k_B \ln k$. ◻
:::

**Modeling note.** $\Delta S_{\min}$ is a definition within the model: the exact Landauer entropy cost per cycle, not a lower bound on arbitrary implementations. The "min" refers to physical optimality outside the formalism.


# Instances

We demonstrate that the leverage framework unifies prior results and applies to diverse architectural decisions.

## Instance 1: Single Source of Truth (SSOT)

We previously formalized the DRY principle, proving that Python uniquely provides SSOT for structural facts via definition-time hooks and introspection. Here we show SSOT is an instance of leverage maximization.

### Prior Result

**Published Theorem:** A language enables SSOT for structural facts if and only if it provides (1) definition-time hooks AND (2) introspectable derivation results. Python is the only mainstream language satisfying both requirements.

**Modification Complexity:** For structural fact $F$ with $n$ use sites:

-   SSOT: $M(\text{change } F) = 1$ (modify source, derivations update automatically)

-   Non-SSOT: $M(\text{change } F) = n$ (modify each use site independently)

### Leverage Perspective

::: definition
Architecture $A_{\text{SSOT}}$ for structural fact $F$ has:

-   Single source $S$ defining $F$

-   Derived use sites updated automatically from $S$

-   DOF $= 1$ (only $S$ is independently modifiable)
:::

::: definition
Architecture $A_{\text{non-SSOT}}$ for structural fact $F$ with $n$ use sites has:

-   $n$ independent definitions (copied or manually synchronized)

-   DOF $= n$ (each definition independently modifiable)
:::

::: theorem
[]{#thm:ssot-leverage label="thm:ssot-leverage"} For structural fact with $n$ use sites: $$\frac{L(A_{\text{SSOT}})}{L(A_{\text{non-SSOT}})} = n$$
:::

::: proof
*Proof.* Both architectures provide same capabilities: $|\text{Cap}(A_{\text{SSOT}})| = |\text{Cap}(A_{\text{non-SSOT}})| = c$.

DOF: $$\begin{aligned}
\text{DOF}(A_{\text{SSOT}}) &= 1 \\
\text{DOF}(A_{\text{non-SSOT}}) &= n
\end{aligned}$$

Leverage: $$\begin{aligned}
L(A_{\text{SSOT}}) &= c/1 = c \\
L(A_{\text{non-SSOT}}) &= c/n
\end{aligned}$$

Ratio: $$\frac{L(A_{\text{SSOT}})}{L(A_{\text{non-SSOT}})} = \frac{c}{c/n} = n$$ ◻
:::

::: corollary
As use sites grow ($n \to \infty$), leverage advantage grows unbounded.
:::

::: corollary
For small $p$: $$\frac{P_{\text{error}}(A_{\text{non-SSOT}})}{P_{\text{error}}(A_{\text{SSOT}})} \approx n$$
:::

**Connection to Prior Work:** Our published Theorem 6.3 (Unbounded Complexity Gap) showed $M(\text{SSOT}) = O(1)$ vs $M(\text{non-SSOT}) = \Omega(n)$. Theorem [\[thm:ssot-leverage\]](#thm:ssot-leverage){reference-type="ref" reference="thm:ssot-leverage"} provides the leverage perspective: SSOT achieves $n$-times better leverage.

## Instance 2: Nominal Typing Dominance

We previously proved nominal typing strictly dominates structural and duck typing for OO systems with inheritance. Here we show this is an instance of leverage maximization.

### Prior Result

**Published Theorems:**

1.  Theorem 3.13 (Provenance Impossibility): No shape discipline can compute provenance

2.  Theorem 3.19 (Capability Gap): Gap = B-dependent queries = {provenance, identity, enumeration, conflict resolution}

3.  Theorem 3.5 (Strict Dominance): Nominal strictly dominates duck typing

### Leverage Perspective

::: definition
A typing discipline $D$ is an architecture where:

-   Components = type checker, runtime dispatch, introspection APIs

-   Capabilities = queries answerable by the discipline
:::

**Duck Typing:** Uses only Shape axis ($S$: methods, attributes)

-   Capabilities: Shape checking ("Does object have method $m$?")

-   Cannot answer: provenance, identity, enumeration, conflict resolution

**Nominal Typing:** Uses Name + Bases + Shape axes ($N + B + S$)

-   Capabilities: All duck capabilities PLUS 4 B-dependent capabilities

-   Can answer: "Which type provided method $m$?" (provenance), "Is this exactly type $T$?" (identity), "List all subtypes of $T$" (enumeration), "Which method wins in diamond?" (conflict)

::: theorem
[]{#thm:nominal-leverage label="thm:nominal-leverage"} Assuming $\mathrm{DOF}(\mathrm{Nominal}) = \mathrm{DOF}(\mathrm{Duck}) = d$ (equal implementation cost hypothesis): $$L(\text{Nominal}) > L(\text{Duck})$$
:::

::: proof
*Proof.* Let $c_{\text{duck}} = |\text{Cap}(\text{Duck})|$ and $c_{\text{nominal}} = |\text{Cap}(\text{Nominal})|$.

By Theorem 3.19 (published): $c_{\text{nominal}} = c_{\text{duck}} + 4$.

By hypothesis: $\text{DOF}(\text{Nominal}) = \text{DOF}(\text{Duck}) = d$.

Therefore: $$L(\text{Nominal}) = \frac{c_{\text{duck}} + 4}{d} > \frac{c_{\text{duck}}}{d} = L(\text{Duck})$$

**Status of hypothesis.** The equal-DOF assumption is formalized as an explicit premise in the Lean mechanization (L13). Whether it holds in practice depends on the implementation. Abstractly: the typing discipline is one component; DOF is the number of independent change points in that component. The hypothesis says adding name-based dispatch (N + B axes) does not introduce more independent change points than shape-based dispatch (S axis alone). ◻
:::

**Connection to Prior Work:** Our published Theorem 3.5 (Strict Dominance) showed nominal typing provides strictly more capabilities for same DOF cost. Theorem [\[thm:nominal-leverage\]](#thm:nominal-leverage){reference-type="ref" reference="thm:nominal-leverage"} provides the leverage formulation.

## Instance 3: Microservices Architecture

Should a system use microservices or a monolith? How many services are optimal? The leverage framework provides answers. This architectural style, popularized by Fowler and Lewis [@fowler2014microservices], traces its roots to the Unix philosophy of "doing one thing well" [@pike1984program].

### Architecture Comparison

**Monolith:**

-   Components: Single deployment unit

-   DOF $= 1$

-   Capabilities: Basic functionality, simple deployment

**$n$ Microservices:**

-   Components: $n$ independent services

-   DOF $= n$ (each service independently deployable/modifiable)

-   Additional Capabilities: Independent scaling, independent deployment, fault isolation, team autonomy, polyglot persistence [@fowler2014microservices]

### Leverage Analysis

Let $c_0$ = capabilities provided by monolith.

Let $\Delta c = 5$ denote the additional capabilities from microservices: independent scaling, independent deployment, fault isolation, team autonomy, and polyglot persistence.

**Leverage:** $$\begin{aligned}
L(\text{Monolith}) &= c_0 / 1 = c_0 \\
L(n \text{ Microservices}) &= (c_0 + \Delta c) / n = (c_0 + 5) / n
\end{aligned}$$

**Break-even Point:** $$L(\text{Microservices}) \geq L(\text{Monolith}) \iff \frac{c_0 + 5}{n} \geq c_0 \iff n \leq 1 + \frac{5}{c_0}$$

**Interpretation:** If base capabilities $c_0 = 5$, then $n \leq 2$ services is optimal. For $c_0 = 20$, up to $n = 1.25$ (i.e., monolith still better). Microservices justified only when additional capabilities significantly outweigh DOF cost.

## Instance 4: REST API Design

Generic endpoints vs specific endpoints: a leverage tradeoff.

### Architecture Comparison

**Specific Endpoints:** One endpoint per use case

-   Example: `GET /users`, `GET /posts`, `GET /comments`, \...

-   For $n$ use cases: DOF $= n$

-   Capabilities: Serve $n$ use cases

**Generic Endpoint:** Single parameterized endpoint

-   Example: `GET /resources/:type/:id`

-   DOF $= 1$

-   Capabilities: Serve $n$ use cases (same as specific)

### Leverage Analysis

$$\begin{aligned}
L(\text{Generic}) &= n / 1 = n \\
L(\text{Specific}) &= n / n = 1
\end{aligned}$$

**Advantage:** $L(\text{Generic}) / L(\text{Specific}) = n$

**Tradeoff:** Generic endpoint has higher leverage but may sacrifice:

-   Type safety (dynamic routing)

-   Specific validation per resource

-   Tailored response formats

**Decision Rule:** Use generic if $n > k$ where $k$ is complexity threshold (typically $k \approx 3$--$5$).

## Instance 5: Configuration Systems

Convention over configuration (CoC) reduces developer decisions and preserves flexibility [@hansson2005rails]. In this framework it is leverage maximization via defaults.

### Architecture Comparison

**Explicit Configuration:** Must set all $m$ parameters

-   DOF $= m$ (each parameter independently set)

-   Capabilities: Configure $m$ aspects

**Convention over Configuration:** Provide defaults, override only $k$ parameters

-   DOF $= k$ where $k \ll m$

-   Capabilities: Configure same $m$ aspects (defaults handle rest)

**Example (Rails vs Java EE):**

-   Rails: 5 config parameters (convention for rest)

-   Java EE: 50 config parameters (explicit for all)

### Leverage Analysis

$$\begin{aligned}
L(\text{Convention}) &= m / k \\
L(\text{Explicit}) &= m / m = 1
\end{aligned}$$

**Advantage:** $L(\text{Convention}) / L(\text{Explicit}) = m/k$

For Rails example: $m/k = 50/5 = 10$ (10$\times$ leverage improvement).

## Instance 6: Database Schema Normalization

Normalization eliminates redundancy, maximizing leverage. This concept, introduced by Codd [@codd1970relational], is the foundation of relational database design [@date2003introduction].

### Architecture Comparison

Consider customer address stored in database:

**Denormalized (Address in 3 tables):**

-   `Users` table: address columns

-   `Orders` table: shipping address columns

-   `Invoices` table: billing address columns

-   DOF $= 3$ (address stored 3 times)

**Normalized (Address in 1 table):**

-   `Addresses` table: single source

-   Foreign keys from `Users`, `Orders`, `Invoices`

-   DOF $= 1$ (address stored once)

### Leverage Analysis

Both provide same capability: store/retrieve addresses.

$$\begin{aligned}
L(\text{Normalized}) &= c / 1 = c \\
L(\text{Denormalized}) &= c / 3
\end{aligned}$$

**Advantage:** $L(\text{Normalized}) / L(\text{Denormalized}) = 3$

**Modification Complexity:**

-   Change address format: Normalized $M = 1$, Denormalized $M = 3$

-   Error probability: $P_{\text{denorm}} = 3p$ vs $P_{\text{norm}} = p$

**Tradeoff:** Normalization increases leverage but may sacrifice query performance (joins required).

## Summary of Instances

::: center
  **Instance**        **High Leverage**          **Low Leverage**        **Ratio**
  ---------------- ------------------------ -------------------------- -------------
  SSOT                     DOF = 1                  DOF = $n$               $n$
  Nominal Typing     $c+4$ caps, DOF $d$        $c$ caps, DOF $d$        $(c+4)/c$
  Microservices       Monolith (DOF = 1)     $n$ services (DOF = $n$)   $n/(c_0+5)$
  REST API            Generic (DOF = 1)        Specific (DOF = $n$)         $n$
  Configuration     Convention (DOF = $k$)     Explicit (DOF = $m$)        $m/k$
  Database           Normalized (DOF = 1)    Denormalized (DOF = $n$)       $n$
:::

**Pattern:** High leverage architectures achieve $n$-fold improvement where $n$ is the consolidation factor (use sites, services, endpoints, parameters, or redundant storage).


# Practical Demonstration

We demonstrate the leverage framework by showing how DOF collapse patterns manifest in OpenHCS, a production 45K LoC Python bioimage analysis platform. This section presents qualitative before/after examples illustrating the leverage archetypes, with PR #44 providing a publicly verifiable anchor.

## The Leverage Mechanism

For a before/after pair $A_{\text{pre}}, A_{\text{post}}$, the **structural leverage factor** is: $$\rho := \frac{\mathrm{DOF}(A_{\text{pre}})}{\mathrm{DOF}(A_{\text{post}})}.$$ If capabilities are preserved, leverage scales exactly by $\rho$. The key insight: when $\mathrm{DOF}(A_{\text{post}}) = 1$, we achieve $\rho = n$ where $n$ is the original DOF count.

#### What counts as a DOF?

Independent *definition loci*: manual registration sites, independent override parameters, separately defined endpoints/handlers/rules, duplicated schema/format definitions. The unit is "how many places can drift apart," not lines of code.

## Verifiable Example: PR #44 (Contract Enforcement)

PR #44 ("UI Anti-Duck-Typing Refactor") in the OpenHCS repository provides a publicly verifiable demonstration of DOF collapse:

**Before (duck typing):** The `ParameterFormManager` class used scattered `hasattr()` checks throughout the codebase. Each dispatch point was an independent DOF: a location that could drift, contain typos, or miss updates when widget interfaces changed.

**After (nominal ABC):** A single `AbstractFormWidget` ABC defines the contract. All dispatch points collapsed to one definition site. The ABC provides fail-loud validation at class definition time rather than fail-silent behavior at runtime.

**Leverage interpretation:** DOF collapsed from $n$ scattered dispatch points to 1 centralized ABC. By Theorem 3.1, this achieves $\rho = n$ leverage improvement. The specific value of $n$ is verifiable by inspecting the PR diff.

## Leverage Archetypes

The framework identifies recurring patterns where DOF collapse occurs:

### Archetype 1: SSOT (Single Source of Truth)

**Pattern:** Scattered definitions $\to$ single authoritative source.

**Mechanism:** Metaclass auto-registration, decorator-based derivation, or introspection-driven generation eliminates manual synchronization.

**Before:** Define class + register in dispatch table (2 loci per type). **After:** Define class; metaclass auto-registers (1 locus per type).

**Leverage:** $\rho = 2$ per type; compounds across $n$ types.

### Archetype 2: Convention over Configuration

**Pattern:** Explicit parameters $\to$ sensible defaults with override.

**Mechanism:** Framework provides defaults; users override only non-standard values.

**Before:** Specify all $m$ configuration parameters explicitly. **After:** Override only $k \ll m$ parameters; defaults handle rest.

**Leverage:** $\rho = m/k$.

### Archetype 3: Generic Abstraction

**Pattern:** Specific implementations $\to$ parameterized generic.

**Mechanism:** Factor common structure into generic endpoint/handler with parameters for variation.

**Before:** $n$ specific endpoints with duplicated logic. **After:** 1 generic endpoint with $n$ parameter instantiations.

**Leverage:** $\rho = n$.

### Archetype 4: Centralization

**Pattern:** Scattered cross-cutting concerns $\to$ centralized handler.

**Mechanism:** Middleware, decorators, or aspect-oriented patterns consolidate error handling, logging, authentication, etc.

**Before:** Each call site handles concern independently. **After:** Central handler; call sites delegate.

**Leverage:** $\rho = n$ where $n$ is number of call sites.

## Summary

The leverage framework identifies a common mechanism across diverse refactoring patterns: DOF collapse yields proportional leverage improvement. Whether the pattern is SSOT, convention-over-configuration, generic abstraction, or centralization, the mathematical structure is identical: reduce DOF while preserving capabilities.

PR #44 provides a verifiable anchor demonstrating this mechanism in practice. The qualitative value lies not in aggregate statistics but in the *mechanism*: once understood, the pattern applies wherever scattered definitions can be consolidated.


# Related Work

## Software Architecture Metrics

**Coupling and Cohesion [@stevens1974structured]:** Introduced coupling (inter-module dependencies) and cohesion (intra-module relatedness). Recommend high cohesion, low coupling.

**Difference:** Our framework is capability-aware. High cohesion correlates with high leverage (focused capabilities per module), but we formalize the connection to error probability.

**Cyclomatic Complexity [@mccabe1976complexity]:** Counts decision points in code. Higher values are commonly used as a risk indicator, though empirical studies on defect correlation show mixed results.

**Difference:** Complexity measures local control flow; leverage measures global architectural DOF. Orthogonal concerns.

## Design Patterns

**Gang of Four [@gamma1994design]:** Catalogued 23 design patterns (Singleton, Factory, Observer, etc.). Patterns codify best practices but lack formal justification.

**Connection:** Many patterns maximize leverage:

-   **Factory Pattern:** Centralizes object creation (DOF $= 1$ for creation logic)

-   **Strategy Pattern:** Encapsulates algorithms (DOF $= 1$ per strategy family)

-   **Template Method:** Defines algorithm skeleton (DOF $= 1$ for structure)

Our framework explains *why* these patterns work: they maximize leverage.

## Technical Debt

**Cunningham [@cunningham1992wycash]:** Introduced technical debt metaphor. Poor design creates "debt" that must be "repaid" later.

**Connection:** Low leverage = high technical debt. Scattered DOF (non-SSOT, denormalized schemas, specific endpoints) create debt. High leverage architectures minimize debt.

## Formal Methods in Software Architecture

**Architecture Description Languages (ADLs):** Wright [@allen1997formal], ACME [@garlan1997acme], Aesop [@garlan1994exploiting]. Formalize architecture structure but not decision-making. See also Shaw and Garlan [@shaw1996software].

**Difference:** ADLs describe architectures; our framework prescribes optimal architectures via leverage maximization.

**ATAM and CBAM:** Architecture Tradeoff Analysis Method [@kazman2000atam] and Cost Benefit Analysis Method [@bachmann2000cbam]. Evaluate architectures against quality attributes (performance, modifiability, security). See also Bass et al. [@bass2012software].

**Difference:** ATAM is qualitative; our framework provides quantitative optimization criterion (maximize $L$).

**Necessity Specifications:** Mackay et al. [@mackay2022necessity] formalize *necessity specifications*: robustness guarantees that modules do only what their specification requires, even under adversarial clients. Soundness is mechanized in Coq.

**Connection:** Necessity specifications address *behavioral minimality*: modules commit to no more behavior than required. Our framework addresses *structural minimality*: architectures commit to no more DOF than required. Both derive minimal commitments from requirements and prove sufficiency.

## Software Metrics Research

**Chidamber-Kemerer Metrics [@chidamber1994metrics]:** Object-oriented metrics (WMC, DIT, NOC, CBO, RFC, LCOM). Empirical validation studies [@basili1996comparing] found these metrics correlate with external quality attributes.

**Connection:** Metrics like CBO (Coupling Between Objects) and LCOM (Lack of Cohesion) correlate with DOF. High CBO $\implies$ high DOF. Our framework provides theoretical foundation.

## Metaprogramming and Reflection

**Reflection [@maes1987concepts]:** Languages with reflection enable introspection and intercession. Essential for metaprogramming.

**Connection:** Reflection enables high leverage (SSOT). Our prior work showed Python's definition-time hooks + introspection uniquely enable SSOT for structural facts.

**Metaclasses [@bobrow1986commonloops; @kiczales1991amop]:** Early metaobject and metaclass machinery formalized in CommonLoops; the Metaobject Protocol codified in Kiczales et al.'s AMOP. Enable metaprogramming patterns.

**Application:** Metaclasses are high-leverage mechanism (DOF $= 1$ for class structure, unlimited derivations).


# Extension: Weighted Leverage {#weighted-leverage}

The basic leverage framework treats all errors equally. In practice, different decisions carry different consequences. This section extends our framework with *weighted leverage* to capture heterogeneous error severity.

## Weighted Decision Framework

::: definition
A **weighted decision** extends an architecture with:

-   **Importance weight** $w \in \mathbb{N}^+$: the relative severity of errors in this decision

-   **Risk-adjusted DOF**: $\text{DOF}_w = \text{DOF} \times w$
:::

The key insight is that a decision with importance weight $w$ carries $w$ times the error consequence of a unit-weight decision. This leads to:

::: definition
$$L_w = \frac{\text{Capabilities} \times w}{\text{DOF}_w} = \frac{\text{Capabilities}}{\text{DOF}}$$
:::

The cancellation is intentional: weighted leverage preserves comparison properties while enabling risk-adjusted optimization.

## Key Theorems

::: theorem
For any weighted decision $d$ with $\text{DOF} = 1$: $d$ is Pareto-optimal (not dominated by any alternative with higher weighted leverage).
:::

::: proof
*Proof.* Suppose $d$ has $\text{DOF} = 1$. For any $d'$ to dominate $d$, we would need $d'.\text{DOF} < 1$. But $\text{DOF} \geq 1$ by definition, so no such $d'$ exists. ◻
:::

::: theorem
$\forall a, b, c$: if $a$ has higher weighted leverage than $b$, and $b$ has higher weighted leverage than $c$, then $a$ has higher weighted leverage than $c$.
:::

::: proof
*Proof.* By algebraic manipulation of cross-multiplication inequalities. Formally verified in Lean (38-line proof). ◻
:::

## Practical Application: Feature Flags

Consider two approaches to feature toggle implementation:

**Low Leverage (Scattered Conditionals):**

-   DOF: One per feature $\times$ one per use site ($n \times m$)

-   Risk: Inconsistent behavior if any site is missed

-   Weight: High (user-facing inconsistency)

**High Leverage (Centralized Configuration):**

-   DOF: One per feature

-   Risk: Single source of truth eliminates inconsistency

-   Weight: Same importance, but $m\times$ fewer DOF

Weighted leverage ratio: $L_{\text{centralized}} / L_{\text{scattered}} = m$, the number of use sites.

## Connection to Main Theorems

The weighted framework preserves all results from Sections 3--5:

-   **Theorem 3.1 (Leverage-Error Tradeoff)**: Holds with weighted errors

-   **Theorem 3.2 (Metaprogramming Dominance)**: Weight amplifies the advantage

-   **Theorem 3.4 (Optimality)**: Weighted optimization finds risk-adjusted optima

-   **SSOT Dominance**: Weight $w$ makes $n \times w$ leverage advantage

All proofs verified in Lean: `Leverage/WeightedLeverage.lean` (348 lines, 0 sorry placeholders).


# Conclusion

## Methodology and Disclosure

**Role of LLMs in this work.** This paper was developed through human-AI collaboration. The author provided the core insight (that DOF $= 1$ is selected by five independent scientific frameworks) while large language models (Claude, GPT-4) served as implementation partners for formalization, proof drafting, and LaTeX generation.

The Lean 4 proofs (70700 lines, 0 `sorry` placeholders) were iteratively developed: the author specified theorems, the LLM proposed proof strategies, and the Lean compiler verified correctness. Machine-checked proofs are correct regardless of generation method.

**What the author contributed:** The five-way convergence insight, the identification of structural rank as the information-geometric coordinate of DOF, the thermodynamic selection theorem, the cross-paper dependency chain, the open conjectures, and the OpenHCS case study selection.

**What LLMs contributed:** LaTeX drafting, Lean tactic suggestions, prose refinement, and exploration of proof strategies.

::: center

----------------------------------------------------------------------------------------------------
:::

## Summary

The central result is the Five-Way Equivalence (Theorem 5.8): five independent scientific frameworks all characterize the single-source condition (DOF $= 1$).

::: center
  **Framework**              **DOF $= 1$ means**                                  **Source**
  -------------------------- ---------------------------------------------------- --------------------------------
  Engineering optimization   Maximum leverage $L = |\mathrm{Cap}|/\mathrm{DOF}$   `Leverage`
  Epistemic coherence        Single Source of Truth                               `Ssot`
  Information geometry       Structural rank $= 1$                                `DecisionQuotient`
  Computational complexity   Tractable sufficiency checking                       `DecisionQuotient`
  Statistical physics        Minimum thermodynamic cost per cycle                 `Leverage`, `DecisionQuotient`
:::

The engineering consequences (Theorems 1 to 6 from the prior conclusion) are corollaries: DOF-Reliability Isomorphism, Leverage-Error Tradeoff, Modification Complexity Gap, Physical Edit-Energy Floor, Budget Feasibility Boundary, and the Optimal Architecture decision procedure. All are machine-checked in Lean 4 with live cross-module imports (`Leverage` $\to$ `DecisionQuotient` $\to$ `Ssot` $\to$ Mathlib).

**What is new in `Leverage` relative to `AbstractClassSystem` and `Ssot`:**

-   The convergence theorem itself: `AbstractClassSystem` and `Ssot` contain two of the five equivalences; this paper closes the chain.

-   The thermodynamic selection theorem (L55 in `BridgeToDQ.lean`): mandatory energy under Landauer calibration for DOF $> 1$, proved unconditionally.

-   The identification of structural rank as the information coordinate for DOF.

-   The England Replication Inequality (L45): $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$, giving the thermodynamic advantage of SSOT a quantitative non-equilibrium interpretation. No open conjectures remain.

## No Open Conjectures

All conjectures from the original submission are now proved in `Leverage/BridgeToDQ.lean`:

-   L55: unconditional energy bound, no P $\neq$ coNP hypothesis.

-   L49: quantitative Landauer-linear energy bound.

-   L45: $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$ (Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}).

## Decision Procedure

For practitioners, the five-way equivalence implies a principled architectural decision procedure.

Given requirements $R$, choose optimal architecture via:

1.  **Enumerate:** List candidate architectures $\{A_1, \ldots, A_n\}$

2.  **Filter:** Keep only $A_i$ with $\mathrm{Cap}(A_i) \supseteq R$

3.  **Compute:** Calculate $L(A_i) = |\mathrm{Cap}(A_i)|/\mathrm{DOF}(A_i)$ for each

4.  **Optimize:** Choose $A^* = \arg\max_i L(A_i)$

**Justification:** By the Five-Way Equivalence, $A^*$ simultaneously (a) maximizes leverage, (b) satisfies SSOT, (c) has minimum structural rank, (d) admits tractable sufficiency checking, and (e) minimizes thermodynamic cost. These are five independent validations of the same choice.

## Limitations

**1. Independence Assumption:** The probability model treats DOF-level errors as independent under the orthogonality assumptions from `AbstractClassSystem` [@paper1_typing_discipline]. Real systems may have correlated failure modes not captured by the isomorphism.

**2. Constant Error Rate:** Assumes $p$ is uniform across components. Some components are more error-prone than others; a weighted version is future work.

**3. P $\neq$ coNP Hypothesis (resolved):** The thermodynamic energy bound in Theorem 5.7, point 3 is now proved unconditionally (L55). Point 2 (complexity hardness) holds under P $\neq$ coNP, which follows from the physical axioms of [@paper4_decision_quotient] (LP38).

**4. Capability Quantification:** We count capabilities qualitatively. Some capabilities are more valuable; a weighted leverage extension $L = \sum w_i c_i / \mathrm{DOF}$ is natural but unformalized.

## Impact

**For Physicists:** The England Replication Inequality (Theorem [\[thm:england\]](#thm:england){reference-type="ref" reference="thm:england"}) is proved from counting: $|{\mathrm{Fin}\;k \to \mathrm{Bool}}| = 2^k$, combined with $k \leq 2^{k-1}$ (induction), yields $\Delta S_{\min}(k) - \Delta S_{\min}(1) \geq k_B \ln k$. The only physics input is Landauer's constant $k_B$. This provides a mechanized derivation of a non-equilibrium thermodynamic bound. Additionally, P $\neq$ NP is proved from physical law (LP38: Landauer, finite energy, finite signal speed, nontrivial state space) in 1250+ lines of Lean 4.

**For Information Theorists:** The structural rank (srank) equals the DOF, and srank $= 1$ is equivalent to tractable sufficiency checking. The Five-Way Equivalence connects information geometry to computational complexity and thermodynamics.

**For Software Practitioners:** Five independent reasons to prefer DOF $= 1$ architectures. When choosing between alternatives, compute leverage and select maximum. Doing so simultaneously satisfies epistemic coherence, minimizes structural rank, enables tractable decision-making, and minimizes thermodynamic cost.

## Final Remarks

This paper proves thermodynamic bounds on information-processing systems from first principles. The single-source condition (DOF $= 1$) is characterized as optimal by five independent frameworks: engineering, epistemics, information theory, computational complexity, and statistical physics.

All theorems are machine-checked. The thermodynamic bound is unconditional; the England Replication Inequality gives the SSOT entropy advantage as $k_B \ln k$ per update cycle. The only explicit physics axioms are the Second Law (non-negative entropy production) and the Thermodynamic Uncertainty Relation (Barato-Seifert 2015).


# Lean Proof Artifacts {#appendix-lean}

This appendix reports machine-check status and proof traceability directly from source and generated mapping artifacts.

## Verification Status

**Lean summary:** 70700 lines, 3155 theorems/lemmas, 0 `sorry`, across 244 files.

::: center
  **File**                                  **Lines**   **Theorems/Lemmas**
  ---------------------------------------- ----------- ---------------------
  `LambdaDR.lean`                              343              24
  `Leverage.lean`                              115               0
  `Leverage/Foundations.lean`                  234              13
  `Leverage/Probability.lean`                  841              39
  `Leverage/Theorems.lean`                     303              20
  `Leverage/SSOT.lean`                         192              13
  `Leverage/Typing.lean`                       210              15
  `Leverage/Examples.lean`                     184               4
  `Leverage/WeightedLeverage.lean`             348              13
  `Leverage/BridgeToDQ.lean`                   433              26
  `Leverage/FiveWayEquivalence.lean`           149               7
  `Leverage/CrossPaperDependencies.lean`       332              22
  `lakefile.lean`                              19                0
  **Total**                                 **70700**        **3155**
:::

Build command: `cd proofs && lake build`

## Claim Coverage Matrix

## Lean Handle Map

## Proof Hardness Index


  -------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                      **Status**   **Lean support**
  ------------------------------------- ------------ ----------------------------------------------------------------------------------------------------------
  `cor:dof-errors`                      Full         `L.architecture_axes_independent`, `L.error_independence_from_orthogonality`

  `cor:dof-monotone`                    Full         `L.lower_dof_lower_errors`

  `cor:dof-ratio`                       Full         `L.dof_ratio_predicts_error_ratio`

  `cor:leverage-energy`                 Full         `L.Physical.higher_leverage_same_caps_implies_lower_energy`, `nonorthogonal_complete_has_redundant_axis`

  `cor:linear-approx`                   Full         `L.bernoulli_justifies_linear_model`, `L.ordering_equivalence_exact`

  `cor:physical-assumption-necessity`   Full         `exists_orthogonal_semanticallyMinimal_subset`, `exists_semanticallyMinimal_subset`

  `prop:dof-additive`                   Full         `L.compose_dof`, `L.composition_dof_additive`

  `thm:approx-bound`                    Full         `L.linear_model_preserves_ordering`, `L.ordering_equivalence_exact`

  `thm:composition`                     Full         `L.composition_caps_additive`, `L.composition_dof_additive`

  `thm:dof-reliability`                 Full         `L.dof_reliability_isomorphism`, `L.isomorphism_preserves_failure_ordering`

  `thm:england`                         Full         `england_replication_inequality`

  `thm:error-compound`                  Full         `L.correctness_probability`, `L.system_is_correct`

  `thm:error-independence`              Full         `L.error_independence_from_orthogonality`

  `thm:error-prob`                      Full         `L.error_probability_denom_pos`, `L.series_error_probability`

  `thm:expected-errors`                 Full         `L.expected_errors_from_linearity`, `L.expected_errors_linear`

  `thm:five-way`                        Unmapped     *(no derived Lean handle found)*

  `thm:leverage-error`                  Full         `L.leverage_error_tradeoff`

  `thm:leverage-gap`                    Full         `L.leverage_gap`

  `thm:leverage-max`                    Full         `L.leverage_caps_principle`, `L.leverage_maximization_principle`

  `thm:metaprog`                        Full         `L.metaprogramming_dominates`, `L.metaprogramming_unbounded_leverage`

  `thm:mod-bound`                       Full         `L.modification_eq_dof`

  `thm:nominal-leverage`                Full         `L.Typing.capability_gap`, `L.Typing.nominal_dominates_duck`

  `thm:optimal`                         Full         `L.max_leverage_is_optimal`, `L.optimal_minimizes_error`

  `thm:paper1-integration`              Full         `L.Typing.nominal_dominates_duck`, `L.Typing.paper1_is_leverage_instance`

  `thm:paper2-integration`              Full         `L.SSOT.paper2_is_leverage_instance`, `L.SSOT.ssot_leverage_dominance`

  `thm:physical-budget-boundary`        Full         `L.Physical.feasible_iff_floor_le_budget`, `l4_exchange_wrapper`

  `thm:physical-energy-floor`           Full         `l5_exchange_wrapper`, `matroid_basis_equicardinality`

  `thm:ssot-leverage`                   Full         `L.SSOT.modification_ratio`, `L.SSOT.ssot_leverage_dominance`

  `thm:testable-prediction`             Full         `L.testable_modification_prediction`
  -------------------------------------------------------------------------------------------------------------------------------------------------------------

*Notes:* *(1) Full rows come from theorem-local inline anchors in this paper.* *(2) Derived rows are filled by dependency/scaffold claim-handle derivation (same paper-handle label across proof dependencies).* *(3) Unmapped means no local anchor and no derivable dependency support were found.*

*Auto summary: mapped 28/29 (full=28, derived=0, unmapped=1).*


:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: list
::: {#lh:BA7}
`BA7`
:::

::: {#lh:DQ1}
`DQ1`
:::

::: {#lh:EP1}
`EP1`
:::

::: {#lh:EP4}
`EP4`
:::

::: {#lh:FXI1}
`FXI1`
:::

::: {#lh:L1}
`L1`
:::

::: {#lh:L2}
`L2`
:::

::: {#lh:L3}
`L3`
:::

::: {#lh:L4}
`L4`
:::

::: {#lh:L5}
`L5`
:::

::: {#lh:L6}
`L6`
:::

::: {#lh:L7}
`L7`
:::

::: {#lh:L8}
`L8`
:::

::: {#lh:L9}
`L9`
:::

::: {#lh:L10}
`L10`
:::

::: {#lh:L11}
`L11`
:::

::: {#lh:L12}
`L12`
:::

::: {#lh:L13}
`L13`
:::

::: {#lh:L14}
`L14`
:::

::: {#lh:L15}
`L15`
:::

::: {#lh:L16}
`L16`
:::

::: {#lh:L17}
`L17`
:::

::: {#lh:L18}
`L18`
:::

::: {#lh:L19}
`L19`
:::

::: {#lh:L20}
`L20`
:::

::: {#lh:L21}
`L21`
:::

::: {#lh:L22}
`L22`
:::

::: {#lh:L23}
`L23`
:::

::: {#lh:L24}
`L24`
:::

::: {#lh:L25}
`L25`
:::

::: {#lh:L26}
`L26`
:::

::: {#lh:L27}
`L27`
:::

::: {#lh:L28}
`L28`
:::

::: {#lh:L29}
`L29`
:::

::: {#lh:L30}
`L30`
:::

::: {#lh:L31}
`L31`
:::

::: {#lh:L32}
`L32`
:::

::: {#lh:L33}
`L33`
:::

::: {#lh:L34}
`L34`
:::

::: {#lh:L35}
`L35`
:::

::: {#lh:L36}
`L36`
:::

::: {#lh:L37}
`L37`
:::

::: {#lh:L38}
`L38`
:::

::: {#lh:L39}
`L39`
:::

::: {#lh:L40}
`L40`
:::

::: {#lh:L41}
`L41`
:::

::: {#lh:L42}
`L42`
:::

::: {#lh:L43}
`L43`
:::

::: {#lh:L44}
`L44`
:::

::: {#lh:L45}
`L45`
:::

::: {#lh:L46}
`L46`
:::

::: {#lh:L47}
`L47`
:::

::: {#lh:L48}
`L48`
:::

::: {#lh:L49}
`L49`
:::

::: {#lh:L50}
`L50`
:::

::: {#lh:L51}
`L51`
:::

::: {#lh:L52}
`L52`
:::

::: {#lh:L53}
`L53`
:::

::: {#lh:L54}
`L54`
:::

::: {#lh:L55}
`L55`
:::

::: {#lh:L56}
`L56`
:::

::: {#lh:LP38}
`LP38`
:::

::: {#lh:LP43}
`LP43`
:::

::: {#lh:LP44}
`LP44`
:::

::: {#lh:ORA1}
`ORA1`
:::
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ID             | Handle                                                       | ID             | Handle                                                             |
+:===============+:=============================================================+:===============+:===================================================================+
| ID             | Handle                                                       | ID             | Handle                                                             |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:BA7}  | `Physics.BoundedAcquisition.energy_ge_srank_cost`            | ::: {#lh:DQ1}  | `DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost` |
| `BA7`          |                                                              | `DQ1`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:EP1}  | `Physics.LocalityPhysics.landauer_principle`                 | ::: {#lh:EP4}  | `Physics.LocalityPhysics.nontrivial_physics`                       |
| `EP1`          |                                                              | `EP4`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:FXI1} | `fixed_axis_incompleteness`                                  | ::: {#lh:L1}   | `matroid_basis_equicardinality`                                    |
| `FXI1`         |                                                              | `L1`           |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L2}   | `Leverage.Physical.feasible_iff_floor_le_budget`             | ::: {#lh:L3}   | `Leverage.Physical.higher_leverage_same_caps_implies_lower_energy` |
| `L2`           |                                                              | `L3`           |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L4}   | `l4_exchange_wrapper`                                        | ::: {#lh:L5}   | `l5_exchange_wrapper`                                              |
| `L4`           |                                                              | `L5`           |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L6}   | `nonorthogonal_complete_has_redundant_axis`                  | ::: {#lh:L7}   | `exists_semanticallyMinimal_subset`                                |
| `L6`           |                                                              | `L7`           |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L8}   | `exists_orthogonal_semanticallyMinimal_subset`               | ::: {#lh:L9}   | `Leverage.SSOT.modification_ratio`                                 |
| `L8`           |                                                              | `L9`           |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L10}  | `Leverage.SSOT.paper2_is_leverage_instance`                  | ::: {#lh:L11}  | `Leverage.SSOT.ssot_leverage_dominance`                            |
| `L10`          |                                                              | `L11`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L12}  | `Leverage.Typing.capability_gap`                             | ::: {#lh:L13}  | `Leverage.Typing.nominal_dominates_duck`                           |
| `L12`          |                                                              | `L13`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L14}  | `Leverage.Typing.paper1_is_leverage_instance`                | ::: {#lh:L15}  | `Leverage.architecture_axes_independent`                           |
| `L14`          |                                                              | `L15`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L16}  | `Leverage.bernoulli_justifies_linear_model`                  | ::: {#lh:L17}  | `Leverage.compose_dof`                                             |
| `L16`          |                                                              | `L17`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L18}  | `Leverage.composition_caps_additive`                         | ::: {#lh:L19}  | `Leverage.composition_dof_additive`                                |
| `L18`          |                                                              | `L19`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L20}  | `Leverage.correctness_probability`                           | ::: {#lh:L21}  | `Leverage.dof_ratio_predicts_error_ratio`                          |
| `L20`          |                                                              | `L21`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L22}  | `Leverage.dof_reliability_isomorphism`                       | ::: {#lh:L23}  | `Leverage.error_independence_from_orthogonality`                   |
| `L22`          |                                                              | `L23`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L24}  | `Leverage.error_probability_denom_pos`                       | ::: {#lh:L25}  | `Leverage.expected_errors_from_linearity`                          |
| `L24`          |                                                              | `L25`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L26}  | `Leverage.expected_errors_linear`                            | ::: {#lh:L27}  | `Leverage.isomorphism_preserves_failure_ordering`                  |
| `L26`          |                                                              | `L27`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L28}  | `Leverage.leverage_caps_principle`                           | ::: {#lh:L29}  | `Leverage.leverage_error_tradeoff`                                 |
| `L28`          |                                                              | `L29`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L30}  | `Leverage.leverage_gap`                                      | ::: {#lh:L31}  | `Leverage.leverage_maximization_principle`                         |
| `L30`          |                                                              | `L31`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L32}  | `Leverage.linear_model_preserves_ordering`                   | ::: {#lh:L33}  | `Leverage.lower_dof_lower_errors`                                  |
| `L32`          |                                                              | `L33`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L34}  | `Leverage.max_leverage_is_optimal`                           | ::: {#lh:L35}  | `Leverage.metaprogramming_dominates`                               |
| `L34`          |                                                              | `L35`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L36}  | `Leverage.metaprogramming_unbounded_leverage`                | ::: {#lh:L37}  | `Leverage.modification_eq_dof`                                     |
| `L36`          |                                                              | `L37`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L38}  | `Leverage.optimal_minimizes_error`                           | ::: {#lh:L39}  | `Leverage.ordering_equivalence_exact`                              |
| `L38`          |                                                              | `L39`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L40}  | `Leverage.series_error_probability`                          | ::: {#lh:L41}  | `Leverage.system_is_correct`                                       |
| `L40`          |                                                              | `L41`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L42}  | `Leverage.testable_modification_prediction`                  | ::: {#lh:L43}  | `dof_eq_srank`                                                     |
| `L42`          |                                                              | `L43`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L44}  | `dof_one_iff_max_leverage`                                   | ::: {#lh:L45}  | `england_replication_inequality`                                   |
| `L44`          |                                                              | `L45`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L46}  | `incoherent_srank_gt_one`                                    | ::: {#lh:L47}  | `max_coherence_forces_tractability`                                |
| `L46`          |                                                              | `L47`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L48}  | `max_leverage_forces_dof_one`                                | ::: {#lh:L49}  | `srank_energy_lower_bound`                                         |
| `L48`          |                                                              | `L49`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L50}  | `ssot_max_leverage`                                          | ::: {#lh:L51}  | `ssot_srank_one`                                                   |
| `L50`          |                                                              | `L51`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L52}  | `succ_le_two_pow`                                            | ::: {#lh:L53}  | `sufficiency_conp_hard`                                            |
| `L52`          |                                                              | `L53`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L54}  | `thermodynamic_selection`                                    | ::: {#lh:L55}  | `thermodynamic_selection_unconditional`                            |
| `L54`          |                                                              | `L55`          |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:L56}  | `tractable_bounded_core`                                     | ::: {#lh:LP38} | `Physics.LocalityPhysics.pne_np_necessary_for_physics`             |
| `L56`          |                                                              | `LP38`         |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:LP43} | `Physics.LocalityPhysics.without_separation_no_independence` | ::: {#lh:LP44} | `Physics.LocalityPhysics.without_finite_capacity_no_gap`           |
| `LP43`         |                                                              | `LP44`         |                                                                    |
| :::            |                                                              | :::            |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
| ::: {#lh:ORA1} | `oracle_arbitrary`                                           |                |                                                                    |
| `ORA1`         |                                                              |                |                                                                    |
| :::            |                                                              |                |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+
|                |                                                              |                |                                                                    |
+----------------+--------------------------------------------------------------+----------------+--------------------------------------------------------------------+


  ----------------------------------------------------------------------------------------------------------------------------
  **Paper handle**                      **Hardness profile**   **Regime tags**           **Lean support**
  ------------------------------------- ---------------------- ------------------------- -------------------------------------
  `cor:dof-errors`                      `unspecified`          \-                        L15, L23

  `cor:dof-monotone`                    `unspecified`          \-                        L33

  `cor:dof-ratio`                       `unspecified`          \-                        L21

  `cor:leverage-energy`                 `unspecified`          \-                        L3, L6

  `cor:linear-approx`                   `unspecified`          \-                        L16, L39

  `cor:physical-assumption-necessity`   `unspecified`          \-                        L8, L7

  `prop:dof-additive`                   `unspecified`          \-                        L17, L19

  `thm:approx-bound`                    `unspecified`          \-                        L32, L39

  `thm:composition`                     `unspecified`          \-                        L18, L19

  `thm:dof-reliability`                 `unspecified`          \-                        L22, L27

  `thm:england`                         `unspecified`          \-                        L45

  `thm:error-compound`                  `unspecified`          \-                        L20, L41

  `thm:error-independence`              `unspecified`          \-                        L23

  `thm:error-prob`                      `unspecified`          \-                        L24, L40

  `thm:expected-errors`                 `unspecified`          \-                        L25, L26

  `thm:five-way`                        `unspecified`          \-                        *(no derived Lean handle found)*

  `thm:leverage-error`                  `unspecified`          \-                        L29

  `thm:leverage-gap`                    `unspecified`          \-                        L30

  `thm:leverage-max`                    `unspecified`          \-                        L28, L31

  `thm:metaprog`                        `unspecified`          \-                        L35, L36

  `thm:mod-bound`                       `unspecified`          \-                        L37

  `thm:nominal-leverage`                `unspecified`          \-                        L12, L13

  `thm:optimal`                         `unspecified`          \-                        L34, L38

  `thm:paper1-integration`              `unspecified`          \-                        L13, L14

  `thm:paper2-integration`              `unspecified`          \-                        L10, L11

  `thm:physical-budget-boundary`        `unspecified`          \-                        L2, L4

  `thm:physical-energy-floor`           `unspecified`          \-                        L5, L1

  `thm:ssot-leverage`                   `unspecified`          \-                        L9, L11

  `thm:testable-prediction`             `unspecified`          \-                        L42
  ----------------------------------------------------------------------------------------------------------------------------

*Auto summary: indexed 29 claims by hardness profile (unspecified=29).*


# Notes on assumptions and extensions {#appendix-assumptions}

This appendix lists the principal modeling assumptions and common extensions relevant when applying the leverage framework:

-   **Independence and orthogonality:** Error-independence at the DOF level is derived from axis orthogonality assumptions; when dependencies are present, leverage remains a useful comparative metric but probabilistic models should be adjusted to account for correlation.

-   **Capability counting:** The core theorems require only relative ordering of capabilities; cardinality serves as a practical proxy for capability breadth in examples and case studies.

-   **Multi-objective concerns:** Leverage addresses error probability specifically. Performance, security, and other attributes require multi-objective analysis (e.g., Pareto frontiers) before operational decisions.

-   **Implementations:** SSOT and other instantiations depend on language and platform features; the theoretical principle is implementation-agnostic.

-   **Future work:** Extensions include explicit correlated-error models, weighted capability measures, and integration into multi-criteria architectural decision frameworks.


# Complete Theorem Index {#appendix-theorems}

Paper-level labeled claims in this manuscript:

**Foundations (Section 2):**

-   Proposition [\[prop:dof-additive\]](#prop:dof-additive){reference-type="ref" reference="prop:dof-additive"} (DOF Additivity)

-   Theorem [\[thm:mod-bound\]](#thm:mod-bound){reference-type="ref" reference="thm:mod-bound"} (Modification Bounded by DOF)

**Probability Model (Section 3):**

-   Theorem [\[thm:error-independence\]](#thm:error-independence){reference-type="ref" reference="thm:error-independence"}

-   Corollary [\[cor:dof-errors\]](#cor:dof-errors){reference-type="ref" reference="cor:dof-errors"}

-   Theorem [\[thm:error-compound\]](#thm:error-compound){reference-type="ref" reference="thm:error-compound"}

-   Theorem [\[thm:error-prob\]](#thm:error-prob){reference-type="ref" reference="thm:error-prob"}

-   Corollary [\[cor:linear-approx\]](#cor:linear-approx){reference-type="ref" reference="cor:linear-approx"}

-   Corollary [\[cor:dof-monotone\]](#cor:dof-monotone){reference-type="ref" reference="cor:dof-monotone"}

-   Theorem [\[thm:expected-errors\]](#thm:expected-errors){reference-type="ref" reference="thm:expected-errors"}

-   Theorem [\[thm:dof-reliability\]](#thm:dof-reliability){reference-type="ref" reference="thm:dof-reliability"}

-   Theorem [\[thm:approx-bound\]](#thm:approx-bound){reference-type="ref" reference="thm:approx-bound"}

-   Theorem [\[thm:leverage-gap\]](#thm:leverage-gap){reference-type="ref" reference="thm:leverage-gap"}

-   Theorem [\[thm:testable-prediction\]](#thm:testable-prediction){reference-type="ref" reference="thm:testable-prediction"}

-   Corollary [\[cor:dof-ratio\]](#cor:dof-ratio){reference-type="ref" reference="cor:dof-ratio"}

**Main Results (Section 4):**

-   Theorem [\[thm:leverage-max\]](#thm:leverage-max){reference-type="ref" reference="thm:leverage-max"}

-   Theorem [\[thm:leverage-error\]](#thm:leverage-error){reference-type="ref" reference="thm:leverage-error"}

-   Theorem [\[thm:physical-energy-floor\]](#thm:physical-energy-floor){reference-type="ref" reference="thm:physical-energy-floor"}

-   Corollary [\[cor:leverage-energy\]](#cor:leverage-energy){reference-type="ref" reference="cor:leverage-energy"}

-   Theorem [\[thm:physical-budget-boundary\]](#thm:physical-budget-boundary){reference-type="ref" reference="thm:physical-budget-boundary"}

-   Corollary [\[cor:physical-assumption-necessity\]](#cor:physical-assumption-necessity){reference-type="ref" reference="cor:physical-assumption-necessity"}

-   Theorem [\[thm:metaprog\]](#thm:metaprog){reference-type="ref" reference="thm:metaprog"}

-   Theorem [\[thm:optimal\]](#thm:optimal){reference-type="ref" reference="thm:optimal"}

-   Theorem [\[thm:composition\]](#thm:composition){reference-type="ref" reference="thm:composition"}

-   Theorem [\[thm:paper1-integration\]](#thm:paper1-integration){reference-type="ref" reference="thm:paper1-integration"}

-   Theorem [\[thm:paper2-integration\]](#thm:paper2-integration){reference-type="ref" reference="thm:paper2-integration"}

**Instances (Section 5):**

-   Theorem [\[thm:ssot-leverage\]](#thm:ssot-leverage){reference-type="ref" reference="thm:ssot-leverage"}

-   Theorem [\[thm:nominal-leverage\]](#thm:nominal-leverage){reference-type="ref" reference="thm:nominal-leverage"}

**Mechanization status:** 70700 lines, 3155 theorems/lemmas, 0 `sorry`, 244 files.

**Primary Lean sources:**

-   `Leverage/Foundations.lean`

-   `Leverage/Probability.lean`

-   `Leverage/Theorems.lean`

-   `Leverage/Physical.lean`

-   `Leverage/SSOT.lean`

-   `Leverage/Typing.lean`

-   `Leverage/Examples.lean`

-   `Leverage/WeightedLeverage.lean`

-   `LambdaDR.lean`

-   `Leverage.lean`




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper3_leverage/proofs/`
- Lines: 70700
- Theorems: 3155
- `sorry` placeholders: 0
