# Paper: Exact Consistency in Multi-Location Encodings: Zero-Error Thresholds and Side Information Bounds

**Status**: IEEE Transactions on Information Theory-ready | **Lean**: 14898 lines, 703 theorems

---

## Abstract

_Abstract not available._

# Introduction

When one latent fact is represented at several modifiable locations, exact consistency becomes a zero-error inference problem for a deterministic multi-location encoding. The current system state either identifies one authoritative value or leaves a nontrivial ambiguity class of mutually incompatible latent states. The paper asks when the encoding architecture itself guarantees exact consistency and, when it does not, how much side information is needed for zero-error resolution. The resulting theory belongs to the intersection of zero-error information theory, side-information source coding, and finite converses [@shannon1956zero; @korner1973graphs; @lovasz1979shannon; @witsenhausen1976zero; @slepian1973noiseless; @cover2006elements].

## Problem Formulation {#sec:encoding-problem}

The model yields an exact-consistency analogue of zero-error resolvability for modifiable encodings. Rate is measured by the number of independently writable locations; we refer to this quantity as the *independent rate* and use *degrees of freedom* (DOF) only as shorthand. The central theorem family is a deterministic finite converse: once ambiguity survives the observation transcript, the same finite information budget reappears in several classical forms.

::: theorem
Let $X$ be a finite latent source, let $Y$ be the deterministic observation transcript available to a resolver, and let $T$ be a finite auxiliary tag. Exact zero-error recovery from $(Y,T)$ is possible exactly when the pair map $x \mapsto (Y(x),T(x))$ is injective on the surviving ambiguity class. This same injectivity requirement yields confusability and counting converses, conditional-entropy and decoder-output reformulations, finite alphabet-gap constraints, and a budgeted finite-error extension. In particular, exact $k$-way resolution requires at least $\log_2 k$ bits of side information.
:::

The structural threshold then follows as the special case in which exact zero incoherence is possible:

::: center
:::

where $C_0$ denotes the largest independent rate that guarantees zero incoherence.

The converse family and the threshold theorem therefore play different roles: the converse identifies the finite obstruction, and $C_0=1$ identifies the unique regime in which that obstruction is absent.

## Relation to Classical Information Theory {#sec:connection-it}

Three classical lines are particularly relevant.

**Zero-error information theory.** Shannon's zero-error framework and its graph-theoretic refinements describe exact communication under confusability constraints [@shannon1956zero; @korner1973graphs; @lovasz1979shannon]. Our model translates that perspective from one-shot channels to persistent encodings under edits.

**Source coding with side information.** In Slepian--Wolf theory and Witsenhausen's zero-error side-information problem, decoder side information lowers the rate or label budget needed for correct recovery [@slepian1973noiseless; @witsenhausen1976zero]. Here the side information is deterministic rather than stochastic, but it plays the same operational role: exact recovery is impossible unless the available tags or observations separate all remaining alternatives.

**Finite entropy converses.** Classical Fano inequalities and related entropy bounds convert decoding difficulty into information bounds [@fano1961transmission; @witsenhausen1975conditional; @cover2006elements]. Our setting yields a deterministic finite-source analogue in which counting, conditional entropy, decoder-output entropy, and output-gap formulations become different normal forms of the same obstruction.

## Realizability and Verification {#sec:realizability}

Beyond the abstract threshold, a second question is when a concrete host system can realize the rate-$1$ regime. We show that two structural properties are required:

1.  **Causal update propagation:** derived locations must update automatically when the source changes.

2.  **Provenance observability:** the system must expose enough structural information to verify which locations are authoritative and which are derived.

## Paper Organization {#overview}

Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"} formalizes the encoding model, the zero-incoherence threshold, and the base side-information lower bound. Section [\[sec:ssot\]](#sec:ssot){reference-type="ref" reference="sec:ssot"} interprets derivation as the dependence structure achieving rate $1$. Section [\[sec:requirements\]](#sec:requirements){reference-type="ref" reference="sec:requirements"} gives the realizability theorem. Section [\[sec:bounds\]](#sec:bounds){reference-type="ref" reference="sec:bounds"} develops the finite converse family and its rate-complexity consequence. Sections [\[sec:evaluation\]](#sec:evaluation){reference-type="ref" reference="sec:evaluation"} and [\[sec:empirical\]](#sec:empirical){reference-type="ref" reference="sec:empirical"} verify the realizability criterion on concrete host systems. A supplementary Lean 4 artifact machine-checks the converse family, threshold chain, realizability theorem, and rate-complexity arguments [@demoura2021lean4].

## Scope {#sec:scope}

The scope is finite facts represented at multiple locations under admissible edits. The focus is on *structural facts*, meaning facts whose encoding locations are fixed when an object, declaration, or artifact is created. This isolates the zero-error regime most clearly and captures a wide class of realizations without making the theory domain-specific.

## Contributions {#sec:contributions}

The main contributions are:

-   **A deterministic finite converse family** for exact consistency, presented through confusability, counting, conditional-entropy, decoder-output, and entropy-gap formulations of the same finite obstruction, together with a deterministic data-processing step and a budgeted finite-error extension.

-   **A sharp threshold corollary** showing that zero-incoherence capacity is exactly one independent source; this is the structural rate-$1$ consequence of the converse family.

-   **A realizability theorem** identifying causal propagation and provenance observability as the conditions needed to attain the rate-$1$ regime in concrete systems.

-   **An operational consequence** showing $O(1)$ manual update cost at rate $1$ and $\Omega(n)$ cost above it, with an unbounded asymptotic gap.

-   **A supplementary machine-checked artifact** verifying the converse family, threshold chain, realizability theorem, and rate-complexity arguments in Lean 4.


# Encoding Systems and Coherence {#sec:foundations}

We formalize exact consistency for multi-location encodings with modification constraints. Facts take values in finite alphabets, system states expose multiple representations of those facts, edits move the system among reachable states, and the observer must determine whether the represented fact is uniquely recoverable.

## Model assumptions and notation {#sec:assumptions}

We state the modeling assumptions used throughout the paper.

1.  **Fact value model (A1):** Each fact $F$ takes values in a finite set $\mathcal{V}_F$; $H(F)=\log_2|\mathcal{V}_F|$ denotes its entropy under a chosen prior. When we write "zero probability of incoherence" we mean $\Pr(\text{incoherent})=0$ under the model for edits and randomness specified below.

2.  **Update model (A2):** Modifications occur as discrete events. An edit $\delta_F$ changes the source value for $F$; derived locations update deterministically when causal propagation is present. We do not require a blocklength parameter; asymptotic statements are in the number of encoding locations $n$.

3.  **Adversary / randomness (A3):** When randomness is needed we assume a benign stochastic edit model for illustration. For impossibility and converse statements we make no cooperative assumptions: adversarial edit sequences that produce incoherence are allowed whenever the independent rate exceeds $1$.

4.  **Side-information channel (A4):** Derived locations and runtime queries collectively form side information $S$ available to any verifier. In the finite zero-error layer, the relevant quantity is the number of distinguishable tags that $S$ can expose.

References to these assumptions use labels (A1)--(A4). Full formal versions of the operational model are given in the mechanized supplement (Supplement A).

## The Encoding Model {#sec:epistemic}

We begin with the abstract encoding model: locations, values, reachable states, and ambiguity.

::: definition
[]{#def:encoding-system label="def:encoding-system"} An *encoding system* for a fact $F$ is a collection of locations $\{L_1, \ldots, L_n\}$, each capable of holding a value for $F$.
:::

::: definition
[]{#def:coherence label="def:coherence"} An encoding system is *coherent* iff all locations hold the same value: $$\forall i, j: \text{value}(L_i) = \text{value}(L_j)$$
:::

::: definition
[]{#def:incoherence label="def:incoherence"} An encoding system is *incoherent* iff some locations disagree: $$\exists i, j: \text{value}(L_i) \neq \text{value}(L_j)$$
:::

::: definition
[]{#def:ambiguity-class label="def:ambiguity-class"} For a system state $x$, the *ambiguity class* of $x$ is the set $$\mathcal A(x) = \{v : \exists L \text{ with } \text{value}(L)=v\}.$$ When $|\mathcal A(x)|>1$, the state presents multiple internally supported candidate values for the same fact.
:::

**The zero-error resolution problem.** Given a state $x$, a resolver must output the correct value of $F$ with zero error. When $|\mathcal A(x)|>1$, the internal state alone may fail to identify a unique answer.

::: theorem
[]{#thm:oracle-arbitrary label="thm:oracle-arbitrary"} For any incoherent encoding system state $x$ and any oracle $O$ that resolves $x$ to a value $v \in \mathcal A(x)$, there exists a value $v' \in \mathcal A(x)$ such that $v' \neq v$.
:::

::: proof
*Proof.* By incoherence, $|\mathcal A(x)|>1$, so there exist $v_1,v_2\in\mathcal A(x)$ with $v_1\neq v_2$. Either $O$ picks $v_1$ (then $v_2$ disagrees) or it does not pick $v_1$ (then $v_1$ disagrees). ◻
:::

::: definition
[]{#def:dof-epistemic label="def:dof-epistemic"} The *degrees of freedom* (DOF), or *independent rate*, of an encoding system is the number of locations that can be modified independently.
:::

::: theorem
[]{#thm:dof-one-coherence label="thm:dof-one-coherence"} If the independent rate is $1$, then the encoding system is coherent in all reachable states.
:::

::: proof
*Proof.* At independent rate $1$, exactly one location is independent. All other locations are derived (automatically updated when the source changes). Derived locations cannot diverge from their source. Therefore, all locations hold the value determined by the single independent source. Disagreement is impossible. ◻
:::

::: theorem
[]{#thm:dof-gt-one-incoherence label="thm:dof-gt-one-incoherence"} If the independent rate exceeds $1$, then incoherent states are reachable.
:::

::: proof
*Proof.* With DOF $> 1$, at least two locations are independent. Independent locations can be modified separately. A sequence of edits can set $L_1 = v_1$ and $L_2 = v_2$ where $v_1 \neq v_2$. This is an incoherent state. ◻
:::

::: corollary
[]{#cor:coherence-forces-ssot label="cor:coherence-forces-ssot"} If coherence must be guaranteed, then the independent rate $1$ is necessary and sufficient.
:::

This is the basic exact-consistency criterion in the model.

The following subsections refine the model and introduce the quantities used in the later converse statements.

## Host Systems {#sec:edit-space}

The abstract model applies to any host system that exposes a finite family of addressable locations, an admissible edit set, and a correctness condition that can be violated by inconsistent updates.

::: definition
A *host system* $C$ is a finite collection of addressable locations together with a set $E(C)$ of admissible edits. Each edit $\delta\in E(C)$ transforms the current state of $C$ into another reachable state.
:::

::: definition
A *location* $L\in C$ is an addressable component of the host system whose value may participate in the representation of a fact.
:::

::: definition
For host system $C$, the *modification space* $E(C)$ is the set of admissible edits on $C$.
:::

The modification space can be large, but only edits that change one designated fact are considered here.

## Facts: Atomic Units of Specification {#sec:facts}

::: definition
[]{#def:fact label="def:fact"} A *fact* $F$ is an atomic semantic attribute of the represented object that can change independently of other attributes.
:::

The granularity of facts is determined by correctness, not by storage layout. If two pieces of information must change together, they form one fact; if they can change independently, they are distinct facts.

::: definition
[]{#def:structural-fact label="def:structural-fact"} A fact $F$ is *structural* with respect to encoding system $C$ iff the locations encoding $F$ are fixed at definition time: $$\begin{aligned}
\text{structural}(F, C) \Longleftrightarrow {} &
\{L : \text{encodes}(L, F)\} \\
& \subseteq \text{DefinitionSyntax}(C)
\end{aligned}$$ where $\text{DefinitionSyntax}(C)$ denotes the part of the host-system description that fixes the encoding locations when the relevant object is created.
:::

Structural facts are fixed when the relevant object is created or declared. Once fixed, the structure cannot be repaired retroactively without a new structural edit. Non-structural facts may require reactive or feedback-based mechanisms not studied here.

## Encoding: The Correctness Relationship {#sec:encoding}

::: definition
[]{#def:encodes label="def:encodes"} Location $L$ *encodes* fact $F$, written $\text{encodes}(L, F)$, iff correctness requires updating $L$ when $F$ changes.

Formally: $$\begin{aligned}
\text{encodes}(L, F) \Longleftrightarrow {} &
\forall \delta_F:\; \neg\text{updated}(L, \delta_F) \\
& \rightarrow \text{incorrect}(\delta_F(C))
\end{aligned}$$

where $\delta_F$ is an edit targeting fact $F$.
:::

## Modification Complexity {#sec:mod-complexity}

::: definition
[]{#def:mod-complexity label="def:mod-complexity"} $$M(C, \delta_F) = |\{L \in C : \text{encodes}(L, F)\}|$$ The number of locations that must be updated when fact $F$ changes.
:::

Modification complexity is the central operational metric of the paper. It measures the cost of changing a fact in terms of locations that must be updated for correctness.

::: theorem
[]{#thm:correctness-forcing label="thm:correctness-forcing"} $M(C, \delta_F)$ is the **minimum** number of edits required for correctness. Fewer edits imply an incorrect system state.
:::

::: proof
*Proof.* Suppose $M(C, \delta_F) = k$, meaning $k$ locations encode $F$. By Definition [\[def:encodes\]](#def:encodes){reference-type="ref" reference="def:encodes"}, each encoding location must be updated when $F$ changes. If only $j < k$ locations are updated, then $k - j$ locations still reflect the old value of $F$. These locations create inconsistencies:

1.  The specification says $F$ has value $v'$ (new)

2.  Locations $L_1, \ldots, L_j$ reflect $v'$

3.  Locations $L_{j+1}, \ldots, L_k$ reflect $v$ (old)

By Definition [\[def:encodes\]](#def:encodes){reference-type="ref" reference="def:encodes"}, the resulting state is incorrect. Therefore, all $k$ locations must be updated, and $k$ is the minimum. ◻
:::

## Independence and Degrees of Freedom {#sec:dof}

Not all encoding locations are created equal. Some are *derived* from others.

::: definition
[]{#def:independent label="def:independent"} Locations $L_1, L_2$ are *independent* for fact $F$ iff they can diverge. Updating $L_1$ does not automatically update $L_2$, and vice versa.

Formally: $L_1$ and $L_2$ are independent iff there exists a sequence of edits that makes $L_1$ and $L_2$ encode different values for $F$.
:::

::: definition
[]{#def:derived label="def:derived"} Location $L_{\text{derived}}$ is *derived from* $L_{\text{source}}$ iff updating $L_{\text{source}}$ automatically updates $L_{\text{derived}}$. Derived locations are not independent of their sources.
:::

::: definition
[]{#def:dof label="def:dof"} $$\text{DOF}(C, F) = |\{L \in C : \text{encodes}(L, F) \land \text{independent}(L)\}|$$ The number of *independent* locations encoding fact $F$, i.e., the independent rate for that fact.
:::

The independent rate is the key metric. Modification complexity $M$ counts all encoding locations. DOF counts only the independent ones. If all but one encoding location is derived, the independent rate remains $1$ even though $M$ can be large.

::: theorem
[]{#thm:dof-inconsistency label="thm:dof-inconsistency"} $\text{DOF}(C, F) = k$ implies that up to $k$ distinct values for $F$ can coexist simultaneously. In particular, $k>1$ makes ambiguity reachable.
:::

::: proof
*Proof.* Each independent location can hold a different value. By Definition [\[def:independent\]](#def:independent){reference-type="ref" reference="def:independent"}, no constraint forces agreement between independent locations. Therefore, $k$ independent locations can simultaneously realize up to $k$ distinct values. ◻
:::

::: corollary
[]{#cor:dof-risk label="cor:dof-risk"} $\text{DOF}(C, F) > 1$ implies incoherent states are reachable.
:::

## Information-Theoretic Quantities {#sec:it-quantities}

Before establishing the main threshold result, we define the information-theoretic quantities used to express ambiguity, redundancy, and side-information requirements.

::: definition
[]{#def:encoding-rate label="def:encoding-rate"} The *independent rate* of system $C$ for fact $F$ is $R(C, F) = \text{DOF}(C, F)$, the number of independent encoding locations.
:::

::: definition
[]{#def:value-entropy label="def:value-entropy"} For a fact $F$ with value space $\mathcal{V}_F$, the *value entropy* is: $$H(F) = \log_2 |\mathcal{V}_F|$$ This is the information content of specifying $F$'s value.
:::

::: definition
[]{#def:redundancy label="def:redundancy"} The *encoding redundancy* of system $C$ for fact $F$ is: $$\rho(C, F) = \text{DOF}(C, F) - 1$$ Redundancy measures excess independent encodings beyond the minimum needed to represent $F$.
:::

::: theorem
[]{#thm:redundancy-incoherence label="thm:redundancy-incoherence"} Encoding redundancy $\rho > 0$ is necessary and sufficient for incoherence reachability: $$\rho(C, F) > 0 \Longleftrightarrow \text{incoherent states reachable}$$
:::

::: proof
*Proof.* $(\Rightarrow)$ If $\rho > 0$, then DOF $> 1$. By Theorem [\[thm:dof-gt-one-incoherence\]](#thm:dof-gt-one-incoherence){reference-type="ref" reference="thm:dof-gt-one-incoherence"}, incoherent states are reachable.

$(\Leftarrow)$ If incoherent states are reachable, then by contrapositive of Theorem [\[thm:dof-one-coherence\]](#thm:dof-one-coherence){reference-type="ref" reference="thm:dof-one-coherence"}, DOF $\neq 1$. Since the fact is encoded, DOF $\geq 1$, so DOF $> 1$, hence $\rho > 0$. ◻
:::

::: definition
[]{#def:incoherence-entropy label="def:incoherence-entropy"} For a state whose ambiguity class has size $k$, the *incoherence entropy* is: $$H_{\text{inc}} = \log_2 k$$ This quantifies the uncertainty about which value is authoritative.
:::

## Zero-Incoherence Capacity Theorem {#sec:capacity}

We now establish the central exact-consistency threshold. The argument is combinatorial, and its role is to expose the single-source regime on which the later converse and rate-complexity results are built.

**Background: Zero-Error Capacity.** Shannon [@shannon1956zero] introduced zero-error capacity: the maximum rate at which information can be transmitted with *zero* probability of error. Körner [@korner1973graphs] and Lovász [@lovasz1979shannon] linked such questions to confusability structure. Our zero-incoherence capacity is the corresponding exact-consistency threshold for modifiable encodings.

::: definition
[]{#def:coherence-capacity label="def:coherence-capacity"} The *zero-incoherence capacity* of an encoding system is: $$\begin{aligned}
C_0 = \sup\{R : {} & \text{independent rate } R \\
& \Rightarrow \text{incoherence probability} = 0\}
\end{aligned}$$ where "incoherence probability = 0" means no incoherent state is reachable under any modification sequence.
:::

::: theorem
[]{#thm:coherence-capacity label="thm:coherence-capacity"} The zero-incoherence capacity of any encoding system under independent modification is exactly 1: $$C_0 = 1$$
:::

We state the result in achievability/converse form because later sections attach the nontrivial converse and complexity consequences to the same threshold.

::: theorem
[]{#thm:capacity-achievability label="thm:capacity-achievability"} Independent rate $1$ achieves zero incoherence. Therefore $C_0 \geq 1$.
:::

::: proof
*Proof.* Let DOF$(C, F) = 1$. By Definition [\[def:dof\]](#def:dof){reference-type="ref" reference="def:dof"}, exactly one location $L_s$ is independent; all other locations $\{L_i\}$ are derived from $L_s$.

By Definition [\[def:derived\]](#def:derived){reference-type="ref" reference="def:derived"}, derived locations satisfy: $\text{update}(L_s) \Rightarrow \text{automatically\_updated}(L_i)$. Therefore, for any reachable state: $$\forall i: \text{value}(L_i) = f_i(\text{value}(L_s))$$ where $f_i$ is the derivation function. All values are determined by $L_s$. Disagreement requires two locations with different determining sources, but only $L_s$ exists. Therefore, no incoherent state is reachable. ◻
:::

::: theorem
[]{#thm:capacity-converse label="thm:capacity-converse"} Any independent rate greater than $1$ fails to achieve zero incoherence. Therefore $C_0 < R$ for all $R > 1$.
:::

::: proof
*Proof.* Let DOF$(C, F) = k > 1$. By Definition [\[def:independent\]](#def:independent){reference-type="ref" reference="def:independent"}, there exist locations $L_1, L_2$ that can be modified independently.

**Construction of incoherent state:** Consider the modification sequence:

1.  $\delta_1$: Set $L_1 = v_1$ for some $v_1 \in \mathcal{V}_F$

2.  $\delta_2$: Set $L_2 = v_2$ for some $v_2 \neq v_1$

Both modifications are valid (each location accepts values from $\mathcal{V}_F$). By independence, $\delta_2$ does not affect $L_1$. The resulting state has: $$\text{value}(L_1) = v_1 \neq v_2 = \text{value}(L_2)$$

By Definition [\[def:incoherence\]](#def:incoherence){reference-type="ref" reference="def:incoherence"}, this state is incoherent. Since it is reachable, zero incoherence is not achieved.

The converse is intentionally combinatorial: once two independently writable locations exist, one can construct a reachable disagreement state. Later sections refine the same obstruction using finite zero-error side-information counting arguments. ◻
:::

::: proof
*Proof of Theorem [\[thm:coherence-capacity\]](#thm:coherence-capacity){reference-type="ref" reference="thm:coherence-capacity"}.* By Theorem [\[thm:capacity-achievability\]](#thm:capacity-achievability){reference-type="ref" reference="thm:capacity-achievability"}, $C_0 \geq 1$.

By Theorem [\[thm:capacity-converse\]](#thm:capacity-converse){reference-type="ref" reference="thm:capacity-converse"}, $C_0 < R$ for all $R > 1$.

Therefore $C_0 = 1$ exactly. The bound is tight: achieved at DOF $= 1$, not achieved at any DOF $> 1$. ◻
:::

**Relation to Shannon capacity.** The analogy is structural rather than literal: Shannon studies zero-error communication over confusable channel outputs, whereas here the object is exact consistency under admissible edits. The role of "rate" is played by the number of independently writable locations.

::: corollary
[]{#cor:capacity-unique label="cor:capacity-unique"} Independent rate $1$ is the unique capacity-achieving encoding rate. No alternative achieves zero incoherence at higher rate.
:::

::: corollary
[]{#cor:redundancy-above label="cor:redundancy-above"} Any encoding with $\rho > 0$ (redundancy) operates above capacity and cannot guarantee coherence.
:::

## Side Information for Resolution {#sec:side-information}

When an encoding system is incoherent, exact resolution requires external side information.

::: theorem
[]{#thm:side-info label="thm:side-info"} Given a state whose ambiguity class has size $k$, resolving to the correct value requires at least $\log_2 k$ bits of side information.
:::

::: proof
*Proof.* The ambiguity class presents $k$ internally supported alternatives. By Theorem [\[thm:oracle-arbitrary\]](#thm:oracle-arbitrary){reference-type="ref" reference="thm:oracle-arbitrary"}, no internal rule distinguishes them.

Let $S \in \{1, \ldots, k\}$ denote the index of the correct source. The decoder must determine $S$ to resolve correctly. Since $S$ is uniformly distributed over $k$ alternatives (by symmetry of the incoherent state): $$H(S) = \log_2 k$$

Any resolution procedure is a function $R: \text{State} \to \mathcal{V}_F$. Without side information $Y$: $$H(S | R(\text{State})) = H(S) = \log_2 k$$ because $R$ can be computed from the state, which contains no information about which source is correct.

With side information $Y$ that identifies the correct source: $$H(S | Y) = 0 \Rightarrow I(S; Y) = H(S) = \log_2 k$$

Therefore, side information must expose at least $\log_2 k$ bits' worth of distinguishable tags. ◻
:::

::: corollary
[]{#cor:dof1-zero-side label="cor:dof1-zero-side"} At independent rate $1$, resolution requires 0 bits of side information.
:::

::: proof
*Proof.* With $k = 1$, there is one independent location. $H(S) = \log_2 1 = 0$. No uncertainty exists about the authoritative source. ◻
:::

::: corollary
[]{#cor:side-info-redundancy label="cor:side-info-redundancy"} The required side information equals $\log_2(1 + \rho)$ where $\rho$ is encoding redundancy: $$\text{Side information required} = \log_2(\text{DOF}) = \log_2(1 + \rho)$$
:::


# Derivation as the Capacity-Achieving Structure {#sec:ssot}

Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"} identifies the sharp threshold: exact consistency is guaranteed only at independent rate $1$. Here the corresponding structure is *derivation*: one location carries the independent value, while all other representations are deterministic views of that source.

## Rate 1 as the Exact-Consistency Regime {#sec:ssot-def}

::: definition
[]{#def:ssot label="def:ssot"} An encoding system $C$ achieves the *optimal encoding rate* for fact $F$ iff $$\text{DOF}(C,F)=1.$$
:::

Rate $1$ is the unique independent rate compatible with zero incoherence in the model. A single authoritative source with deterministic views is the corresponding structural form.

::: corollary
[]{#thm:ssot-determinate label="thm:ssot-determinate"} If $\text{DOF}(C,F)=1$, then the value of $F$ is determinate in every reachable state.
:::

::: corollary
[]{#thm:ssot-optimal label="thm:ssot-optimal"} If $\text{DOF}(C,F)=1$, then coherence restoration requires $O(1)$ manual updates.
:::

::: corollary
[]{#thm:ssot-unique label="thm:ssot-unique"} DOF $=1$ is the unique rate guaranteeing exact consistency.
:::

::: corollary
[]{#cor:no-redundancy label="cor:no-redundancy"} If $\text{DOF}(C,F)>1$, then incoherent states are reachable.
:::

::: proof
*Proof.* Immediate from Theorem [\[thm:dof-gt-one-incoherence\]](#thm:dof-gt-one-incoherence){reference-type="ref" reference="thm:dof-gt-one-incoherence"}. ◻
:::

## Counting Converse Interpretation {#sec:info-sketches}

The threshold and the converse fit together in a compact way.

-   When ambiguity classes are trivial, no auxiliary information is needed and rate $1$ suffices.

-   When a nontrivial ambiguity class survives the observation transcript, Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"} shows that exact recovery requires a sufficiently large side-information alphabet.

-   If that side information is unavailable but exact correctness is still demanded, the architecture must rely on additional independently accessible support, which moves the system above the rate-$1$ regime and leads to the update-cost lower bound of Section [\[sec:bounds\]](#sec:bounds){reference-type="ref" reference="sec:bounds"}.

Derivation is therefore the capacity-achieving dependence structure: it removes independent rate without removing observable views.

## Rate--Complexity Tradeoff {#sec:ssot-vs-m}

The benefit of derivation is operational. An encoding may expose many visible copies of a fact, but only the independent copies govern manual synchronization cost.

-   **Independent rate $1$:** one manual update changes the authoritative source; all derived views follow automatically.

-   **Independent rate above $1$:** each independent location must be synchronized manually to restore coherence.

This is the architectural meaning of the threshold: independent rate above $1$ creates both ambiguity and manual synchronization burden.

## Derivation: The Relevant Dependence Structure {#sec:derivation}

::: definition
[]{#def:derivation label="def:derivation"} Location $L_{\text{derived}}$ is *derived from* $L_{\text{source}}$ for fact $F$ iff an update to $L_{\text{source}}$ determines the value of $L_{\text{derived}}$ without an additional manual edit.
:::

Derivation may occur at creation time, build time, commit time, or query time depending on the host system. The abstract point is unchanged: a derived location does not contribute an additional independently writable value for the same fact.

::: theorem
[]{#thm:derivation-excludes label="thm:derivation-excludes"} If $L_{\text{derived}}$ is derived from $L_{\text{source}}$, then $L_{\text{derived}}$ cannot diverge from $L_{\text{source}}$ and does not contribute to DOF.
:::

::: proof
*Proof.* By Definition [\[def:derivation\]](#def:derivation){reference-type="ref" reference="def:derivation"}, the value at $L_{\text{derived}}$ is determined by the value at $L_{\text{source}}$ under admissible updates. There is therefore no reachable state in which the two locations carry incompatible values for the same fact. Since the derived location has no independently writable contribution, it does not increase DOF. ◻
:::

::: corollary
[]{#cor:metaprogramming label="cor:metaprogramming"} If all encodings of $F$ except one are derived from that one, then $\text{DOF}(C,F)=1$ and exact consistency is guaranteed.
:::

::: proof
*Proof.* Only one source remains independent; all other encodings are deterministic views. Apply Theorem [\[thm:dof-one-coherence\]](#thm:dof-one-coherence){reference-type="ref" reference="thm:dof-one-coherence"}. ◻
:::

The same dependence structure can be realized in many host systems, but the theorem is purely about deterministic dependence in the abstract encoding model.


# Realizability Requirements for the Capacity-Achieving Regime {#sec:requirements}

The threshold theorem identifies the unique independent rate compatible with exact consistency. A separate question is whether a concrete host system can realize that rate. This section isolates the two structural properties needed for that purpose.

A host system realizes the rate-$1$ regime when one location is authoritative and every secondary representation is maintained as a deterministic view. Two capabilities are required:

1.  **Causal update propagation**: updates to the authoritative source must automatically propagate to derived locations.

2.  **Provenance observability**: the system must expose enough structural information to verify which locations are authoritative and which are derived.

## Confusability and Side Information {#sec:confusability}

To connect realizability to information available at verification time, fix a fact $F$ and consider the alternatives that remain compatible with the available observations. These alternatives form the same zero-error obstruction analyzed in Section [\[sec:bounds\]](#sec:bounds){reference-type="ref" reference="sec:bounds"}: if the host cannot separate the surviving ambiguity class, exact verification requires additional structural side information.

::: lemma
[]{#lem:confusability-clique label="lem:confusability-clique"} If the confusability graph for $F$ contains a clique of size $k$, then any zero-error side-information scheme distinguishing those $k$ alternatives requires at least $k$ distinct tags, equivalently at least $\log_2 k$ bits of side information.
:::

::: proof
*Proof.* Within a $k$-clique, every pair of alternatives is confusable under the base observation. A zero-error tag assignment must therefore separate all $k$ states. Distinct states cannot share a tag, so at least $k$ tags are required. ◻
:::

If a host system does not expose enough structural information to separate the remaining alternatives, exact verification of the rate-$1$ regime is impossible.

## The Structural Timing Constraint {#sec:timing}

::: definition
[]{#def:structural-fact-req label="def:structural-fact-req"} A fact $F$ is *structural* if every location encoding $F$ is fixed when the underlying object, declaration, or artifact is created. After that moment, the encoding can be read and compared, but not retroactively re-created without a new edit event.
:::

Structural facts are the natural exact-consistency setting because they expose a clear creation moment. Examples include schema declarations, registry membership rules, dependency metadata, and interface-level constraints.

::: theorem
[]{#thm:timing-forces label="thm:timing-forces"} For structural facts, derivation must occur no later than the moment at which the relevant encoding locations become fixed.
:::

::: proof
*Proof.* Let $t_{\mathrm{fix}}$ denote the time at which the structural encoding becomes fixed. A derivation performed strictly before $t_{\mathrm{fix}}$ cannot depend on the completed source value; a derivation performed strictly after $t_{\mathrm{fix}}$ cannot alter the already fixed structural representation. Therefore structural derivation must occur at $t_{\mathrm{fix}}$ or as part of the same creation/update event. ◻
:::

## Requirement 1: Causal Update Propagation {#sec:causal-propagation}

::: definition
[]{#def:causal-propagation label="def:causal-propagation"} An encoding system has *causal update propagation* if every update to a source location automatically updates each location derived from that source, without requiring a separate manual synchronization step.
:::

This is the host-system manifestation of deterministic dependence. Without it, a temporal gap appears between source modification and derived-view repair.

::: theorem
[]{#thm:causal-necessary label="thm:causal-necessary"} Achieving independent rate $1$ for structural facts requires causal update propagation.
:::

::: proof
*Proof.* By Theorem [\[thm:timing-forces\]](#thm:timing-forces){reference-type="ref" reference="thm:timing-forces"}, structural derivation must occur at the moment the source-side structural encoding is fixed. If causal propagation is absent, then some derived location remains stale until a later manual action. During that interval the source and derived locations disagree, so the architecture does not realize exact consistency. Hence a verifiable rate-$1$ realization for structural facts requires causal propagation. ◻
:::

## Requirement 2: Provenance Observability {#sec:provenance-observability}

::: definition
[]{#def:provenance-observability label="def:provenance-observability"} An encoding system has *provenance observability* if it supports queries that reveal which locations encode a fact, which of those locations are authoritative, and which are derived.
:::

Provenance observability is the structural side information needed to certify that the system is genuinely in the single-source regime rather than merely appearing coherent on a small sample of states.

::: theorem
[]{#thm:provenance-necessary label="thm:provenance-necessary"} Verifying that independent rate $1$ holds requires provenance observability.
:::

::: proof
*Proof.* To verify independent rate $1$, one must enumerate the locations encoding the fact, identify the authoritative source, and confirm that every remaining location is derived from it. Without provenance queries, these checks cannot be completed from inside the host system. The architecture may be coherent on observed states, but the single-source claim is not verifiable. ◻
:::

## Independence of the Two Requirements {#sec:independence}

The two requirements are logically separate.

::: theorem
[]{#thm:independence label="thm:independence"}

1.  An encoding system can have causal propagation without provenance observability.

2.  An encoding system can have provenance observability without causal propagation.
:::

::: proof
*Proof.* For (1), consider a system that automatically refreshes all derived views after each source update but exposes no query interface for its derivation graph. Coherence may hold operationally, but the single-source claim is not inspectable. For (2), consider a system that records complete provenance metadata yet requires users to trigger propagation manually after each change. The derivation structure is visible, but stale views remain reachable. Hence neither property implies the other. ◻
:::

## The Realizability Theorem {#sec:realizability-theorem}

::: theorem
[]{#thm:ssot-iff label="thm:ssot-iff"} An encoding system can achieve verifiable independent rate $1$ for structural facts if and only if it provides both causal update propagation and provenance observability.
:::

::: proof
*Proof.* Necessity follows from Theorems [\[thm:causal-necessary\]](#thm:causal-necessary){reference-type="ref" reference="thm:causal-necessary"} and [\[thm:provenance-necessary\]](#thm:provenance-necessary){reference-type="ref" reference="thm:provenance-necessary"}. For sufficiency, assume both properties hold. Causal propagation ensures that every derived location is updated as part of the same structural event as the source; provenance observability then certifies that all secondary encodings are in fact derived from that source. The architecture therefore realizes a verifiable single-source encoding, i.e., independent rate $1$. ◻
:::

The application section specializes the theorem to computational examples.


# Verification of the Realizability Criterion {#sec:evaluation}

This section tests Theorem [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"} on representative host systems.

::: corollary
[]{#cor:lang-realizability label="cor:lang-realizability"} A computational platform realizes verifiable independent rate $1$ for structural facts if and only if it combines automatic propagation of derived representations with queryable provenance for those representations.
:::

The corollary is a direct restatement of Theorem [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"}. What changes across examples is the host mechanism that provides propagation and provenance. The tables below record whether the theorem's prediction is confirmed on concrete systems.

#### Programming-language runtimes (selected).

For language-level artifacts, the realizability question becomes concrete: does the host language/runtime itself provide both (i) automatic propagation from one structural definition to its derived runtime views and (ii) queryable provenance for those views, without relying on external build tooling? Table [\[tab:selected-pl\]](#tab:selected-pl){reference-type="ref" reference="tab:selected-pl"} records a compact comparison for representative mainstream runtimes drawn from widely used languages in current usage surveys and rankings [@stackoverflow2025; @tiobe2024].

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Language**                                             **Prop.**   **Prov.**   **Rate 1**   **Feature**        **Reason**
  -------------------------------------------------------- ----------- ----------- ------------ ------------------ --------------------------------------------------------------------------------------------------------
  Python [@pep487; @pythondatamodel2025]                   yes         yes         yes          \_\_init\_-\       Host runtime provides class-definition hooks and hierarchy introspection.
                                                                                                subclass\_\_\      
                                                                                                \_\_sub-\          
                                                                                                classes\_\_()      

  Java [@gosling2021java]                                  no          no          no           annotations only   Annotations/tooling are external; the JVM lacks subclass-enumeration provenance.

  Rust [@rustref2024]                                      partial     no          no           proc macros        Proc macros generate code at compile time, but generated provenance is unavailable at runtime.

  Go [@gospec2024]                                         no          no          no           reflection only    No definition-time hooks and no host-level registry of implementers.

  TypeScript/\                                             partial     no          no           decorators         Decorators attach metadata, but type erasure and missing subclass provenance block exact verification.
  JavaScript\                                                                                                      
  [@typescriptDecorators2025; @javascriptDecorators2025]                                                           
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table V.1.** Selected mainstream programming-language runtimes under Corollary [\[cor:lang-realizability\]](#cor:lang-realizability){reference-type="ref" reference="cor:lang-realizability"}. []{#tab:selected-pl label="tab:selected-pl"}

Corollary [\[cor:lang-realizability\]](#cor:lang-realizability){reference-type="ref" reference="cor:lang-realizability"} predicts verifiable rate $1$ exactly when both host-level conditions hold. In this selected runtime set, Python is the only positive case; the negative rows fail for the missing condition listed in the table. Table [\[tab:selected-db\]](#tab:selected-db){reference-type="ref" reference="tab:selected-db"} shows the same separation in a non-programming-language setting.

#### Database systems.

The same criterion separates native maintained views from external synchronization pipelines.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Database pattern**                                                                            **Prop.**   **Prov.**   **Rate 1**   **Reason**
  ----------------------------------------------------------------------------------------------- ----------- ----------- ------------ ----------------------------------------------------------------------------------------------------------------------------
  Engine-maintained materialized view [@postgresMaterializedViews2025; @postgresPgMatviews2025]   yes         yes         yes          The database engine maintains the derived relation and exposes catalog metadata for the maintained object.

  External ETL copy / duplicate summary table                                                     no          no          no           Synchronization can be skipped or forked, and provenance for the derived copy is no longer intrinsic to the host database.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table V.2.** Database realizations under the same criterion. []{#tab:selected-db label="tab:selected-db"}

Corollary [\[cor:lang-realizability\]](#cor:lang-realizability){reference-type="ref" reference="cor:lang-realizability"} again predicts the outcome exactly: engine-maintained views satisfy both conditions, whereas externalized synchronization does not.

#### Dependency-resolution systems.

A third everyday regime appears in dependency-resolution systems. Package managers often provide strong provenance observability for resolved dependency graphs, but they do not provide causal propagation in the sense of Corollary [\[cor:lang-realizability\]](#cor:lang-realizability){reference-type="ref" reference="cor:lang-realizability"}: editing the authoritative manifest does not itself update the lockfile or installed graph until a separate host operation is executed.

::: center
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **System**                                                  **Prop.**   **Prov.**   **Rate 1**   **Feature**          **Reason**
  ----------------------------------------------------------- ----------- ----------- ------------ -------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  npm\                                                        no          yes         no           lockfile + explain   The lockfile records the resolved tree and npm exposes dependency provenance, but manifest edits and lock/tree synchronization are separated by explicit npm operations.
  [@npmLock2026; @npmExplain2026; @npmLs2026]                                                                           

  Cargo\                                                      no          yes         no           lockfile + tree      Cargo exposes resolved-dependency provenance clearly, but `Cargo.lock` remains a distinct derived artifact synchronized through Cargo commands rather than at source-edit time.
  [@cargoLockGuide2026; @cargoTree2026; @cargoMetadata2026]                                                             

  Poetry\                                                     no          yes         no           `show –tree`         The resolved graph is queryable, but lock regeneration is an explicit step, so propagation is not intrinsic to the source edit itself.
  [@poetryCli2026]                                                                                                      

  pnpm\                                                       no          yes         no           lockfile + why       The host exposes dependency provenance, but the lockfile is still a separately maintained derived artifact rather than an automatically updated structural view at edit time.
  [@pnpmInstall2026; @pnpmWhy2026; @pnpmList2026]                                                                       
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table V.3.** Dependency-resolution systems are typically provenance-rich but propagation-incomplete under the realizability criterion. []{#tab:selected-deps label="tab:selected-deps"}
:::

Corollary [\[cor:lang-realizability\]](#cor:lang-realizability){reference-type="ref" reference="cor:lang-realizability"} also predicts the mixed regime in Table [\[tab:selected-deps\]](#tab:selected-deps){reference-type="ref" reference="tab:selected-deps"}: provenance can be present without host-native propagation, and then verifiable rate $1$ still fails. Supplement A contains additional case-study material.


# Finite Converse Family and Rate-Complexity Bounds {#sec:bounds}

This section states the deterministic finite converse family and then derives its operational rate-complexity consequence.

## Finite Converse Family {#sec:finite-converse-family}

The cleanest zero-error formulation starts from the joint observation-tag map itself.

::: theorem
[]{#thm:pair-injective label="thm:pair-injective"} Let $X$ range over a finite latent alphabet, let $Y$ be the deterministic observation transcript, and let $T$ be a finite auxiliary tag. Exact zero-error recovery from $(Y,T)$ is possible if and only if the pair map $x \mapsto (Y(x),T(x))$ is injective on the ambiguity class under consideration.
:::

::: proof
*Proof.* If two distinct latent states induce the same pair $(Y,T)$, then every deterministic decoder receives the same input on both states and must err on at least one of them. Conversely, if the pair map is injective, define the decoder by inversion on the image of $(Y,T)$; injectivity makes that inverse single-valued and exact. ◻
:::

::: theorem
[]{#thm:confusability-converse label="thm:confusability-converse"} Fix a deterministic observation transcript and an $L$-bit side-information tag. If $K$ latent states induce the same observation transcript, then exact zero-error decoding on that ambiguity class requires $K$ distinct tag outcomes. Equivalently, a $K$-way ambiguity class forms a confusability clique whose size cannot exceed the available tag alphabet.
:::

::: proof
*Proof.* Apply Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} on an ambiguity class on which the observation is constant. Then injectivity of $(Y,T)$ reduces to injectivity of the tag coordinate alone on that class. Since an $L$-bit tag provides at most $2^L$ outcomes, the clique size cannot exceed $2^L$. ◻
:::

::: corollary
[]{#cor:global-budget label="cor:global-budget"} If the observation alphabet has size $O$ and the auxiliary tag alphabet has size $T$, then exact zero-error recovery on a latent alphabet of size $K$ requires $$K \le O\,T.$$
:::

::: proof
*Proof.* Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} requires injectivity of the map into the product alphabet $\mathcal Y \times \mathcal T$. A product alphabet of size $O\,T$ cannot support an injective image of more than $O\,T$ latent states. ◻
:::

::: theorem
[]{#thm:fiber-cardinality label="thm:fiber-cardinality"} Fix an observation value $y$. On the observation fiber $$\mathcal F_y=\{x : Y(x)=y\},$$ exact zero-error recovery forces the tag map to be injective. Consequently, $$|\mathcal F_y| \le |\mathcal T|.$$
:::

::: proof
*Proof.* Restrict Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} to $\mathcal F_y$. Since $Y$ is constant on the fiber, injectivity of the pair map reduces to injectivity of the tag coordinate alone. An injective map from $\mathcal F_y$ into the tag alphabet immediately gives $|\mathcal F_y| \le |\mathcal T|$. ◻
:::

::: theorem
[]{#thm:fano-converse label="thm:fano-converse"} Let $F$ range over $K$ possible latent values. Suppose exact recovery is attempted from a fixed observation transcript together with an $L$-bit side-information tag. Then $$K \le 2^L.$$ Equivalently, exact zero-error recovery of $K$ ambiguous states requires at least $\log_2 K$ bits of side information.
:::

::: proof
*Proof.* This is Theorem [\[thm:confusability-converse\]](#thm:confusability-converse){reference-type="ref" reference="thm:confusability-converse"} specialized to a single $K$-way ambiguity class. ◻
:::

The counting bound is the coarsest form of the obstruction. The next step is to weight the same argument by source mass. Exact recovery still enforces injectivity on successful confusability classes, but now the question is how much entropy can remain once the source is partitioned into success and failure branches.

::: theorem
[]{#thm:conditional-entropy-converse label="thm:conditional-entropy-converse"} Let $X$ be uniform on a $K$-way ambiguity class and let $Y$ be the deterministic observation transcript. If $Y$ is constant on that class, then $H(X\mid Y)=\log_2 K$. In particular, any tag-observation pair $(Y,T)$ that resolves the class exactly must satisfy $$H(X\mid Y,T)=0
\qquad\text{and}\qquad
H(T\mid Y)\ge \log_2 K .$$
:::

::: proof
*Proof.* If $Y$ is constant on the class, conditioning on $Y$ leaves the uniform prior unchanged, hence $H(X\mid Y)=\log_2 K$. Exact recovery from $(Y,T)$ forces $H(X\mid Y,T)=0$, so the side information carried by $T$ must close the entire $\log_2 K$ gap. ◻
:::

::: theorem
[]{#thm:weighted-clique-entropy label="thm:weighted-clique-entropy"} Let $S$ be a success set for a zero-error decoder, and suppose $S$ forms a confusability clique under the observation transcript. If $p_S$ denotes the total probability mass of $S$, then the entropy carried by the successful states obeys $$\sum_{x\in S} p(x)\log_2 \frac{1}{p(x)}
\;\le\;
h_2(p_S) + p_S \log_2 |\mathcal T|,$$ where $h_2$ is binary entropy and $|\mathcal T|$ is the tag alphabet size.
:::

::: proof
*Proof.* Inside a confusability clique, exact recovery on the success set still forces tag injectivity. The successful states therefore contribute at most $\log_2 |\mathcal T|$ bits per unit success mass. Separating success from failure adds the Bernoulli term $h_2(p_S)$, which yields the stated bound. ◻
:::

::: theorem
[]{#thm:success-set-injective label="thm:success-set-injective"} Let $S$ be the decoder success set inside a confusability clique. Then the restriction of the observation-tag map to $S$ is injective, and the weighted entropy of the successful states is bounded by the logarithm of the available tag budget: $$\sum_{x\in S} p(x)\log_2 \frac{1}{p(x)}
\le
h_2(p_S)+ p_S \log_2 |\mathcal T|.$$
:::

::: proof
*Proof.* On the success set, exact decoding holds by definition, so Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} applies directly to the restricted map. Because the clique assumption makes the observation coordinate constant up to confusability, injectivity must be carried by the tag coordinate. The successful branch therefore occupies at most $|\mathcal T|$ distinguishable states, and weighting by its total mass $p_S$ yields the stated entropy bound. ◻
:::

::: theorem
[]{#thm:decoder-gap-converse label="thm:decoder-gap-converse"} In the same deterministic finite model, the obstruction can also be expressed as output-entropy and finite-budget gap constraints: any deterministic decoder output $\hat X$ satisfies $$H(\hat X)\le H(Y,T)\le H(X),$$ and the observation-tag entropy and decoded-output entropy are bounded by their corresponding finite alphabet ceilings, so the deficits $$\log_2 |\mathcal Y\times \mathcal T|-H(Y,T),
\qquad
\log_2 |\widehat{\mathcal X}|-H(\hat X)$$ are nonnegative and vanish only in the uniform saturation cases formalized in the supplement.
:::

::: proof
*Proof.* The decoder output is a deterministic function of $(Y,T)$, so Theorem [\[thm:deterministic-data-processing\]](#thm:deterministic-data-processing){reference-type="ref" reference="thm:deterministic-data-processing"} gives $H(\hat X)\le H(Y,T)$. The pair entropy is itself bounded by the source entropy and by the logarithm of the available pair alphabet, which yields the nonnegative gap constraints. ◻
:::

::: theorem
[]{#thm:deterministic-data-processing label="thm:deterministic-data-processing"} Let $\kappa$ be any deterministic coarsening of the observation-tag pair $(Y,T)$. Then $$H(\kappa(Y,T)) \le H(Y,T).$$ In particular, since any deterministic decoder output $\hat X$ is a coarsening of $(Y,T)$, $$H(\hat X)\le H(Y,T)\le \log_2 |\mathcal Y\times \mathcal T|.$$
:::

::: proof
*Proof.* Deterministic coarsening merges atoms of the distribution on $(Y,T)$ and therefore cannot increase entropy. A decoder is one such coarsening. The ceiling follows because $(Y,T)$ takes values in a finite alphabet of size $|\mathcal Y\times\mathcal T|$. ◻
:::

::: theorem
[]{#thm:equivalence-viewpoint label="thm:equivalence-viewpoint"} The confusability, counting, conditional-entropy, decoder-output, and finite-gap statements above are different normal forms of the same deterministic finite zero-error obstruction: a surviving $K$-way ambiguity class requires a budget of at least $\log_2 K$ bits to be resolved exactly.
:::

::: proof
*Proof.* Pair injectivity is the structural core. The confusability and counting theorems express the injectivity requirement combinatorially; the conditional-entropy and weighted-entropy theorems express it in source-mass coordinates; the decoder-output and gap theorems express it through deterministic coarsening and finite alphabet ceilings; and the finite-error theorem is the relaxed version obtained by allowing a nonzero failure branch. Each theorem therefore measures the same failure of exact isolation in a different coordinate system. ◻
:::

## Finite-error extension {#sec:finite-error-extension}

The main body of the paper studies exact zero error. The same deterministic finite model also supports a budgeted finite-error extension.

::: theorem
[]{#thm:finite-error-budgeted label="thm:finite-error-budgeted"} Let $P_e$ denote the decoder error probability on a finite source of size $K$, observed through a deterministic transcript alphabet of size $O$ together with a tag alphabet of size $T$. Then the decoded-output entropy obeys $$H(\hat X)
\;\le\;
h_2(P_e) + (1-P_e)\log_2 (OT) + P_e \log_2 (K-1).$$
:::

::: proof
*Proof.* Partition the source into success and failure events. On the success branch, the decoded output is constrained by the observation-tag budget and contributes at most $\log_2(OT)$. On the failure branch, the decoder can still output at most one of the remaining $K-1$ alternatives. The Bernoulli split between the two branches contributes the binary-entropy term. Equivalently, $$H(\hat X)=H(\hat X \mid \text{success/failure}) + H(\text{success/failure})$$ is bounded by combining the support bound on each branch with the binary entropy of the branch variable itself. ◻
:::

The zero-error theorems are the $P_e=0$ boundary of this inequality. In that regime the success branch occupies all probability mass, the Bernoulli term vanishes, the failure contribution disappears, and the budget collapses back to the finite converse family above.

## Rate-Complexity Consequence {#sec:rate-complexity-consequence}

The threshold $C_0=1$ has an operational consequence: independent rate governs manual synchronization cost in the same finite deterministic model.

## Cost Model {#sec:cost-model}

::: definition
[]{#def:cost-model label="def:cost-model"} Let $\delta_F$ be a modification to fact $F$ in encoding system $C$. The *effective modification complexity* $M_{\mathrm{effective}}(C,\delta_F)$ is the number of locations that must be edited manually to preserve correctness after applying $\delta_F$.
:::

Only manual edits are counted. Derived locations updated automatically by the host system contribute zero effective cost.

## Upper Bound at Unit Independent Rate {#sec:upper-bound}

::: theorem
[]{#thm:upper-bound label="thm:upper-bound"} If $\text{DOF}(C,F)=1$, then $$M_{\mathrm{effective}}(C,\delta_F)=O(1).$$
:::

::: proof
*Proof.* When $\text{DOF}(C,F)=1$, there is exactly one independently writable source location for $F$. Updating that location is one manual edit; every other representation of $F$ is derived and is updated automatically. The number of manual edits therefore stays bounded by a constant independent of the number of visible encodings. ◻
:::

## Lower Bound Above the Threshold {#sec:lower-bound}

::: theorem
[]{#thm:lower-bound label="thm:lower-bound"} If fact $F$ is encoded at $n$ independent locations, then $$M_{\mathrm{effective}}(C,\delta_F)=\Omega(n).$$
:::

::: proof
*Proof.* Each independent location can retain an outdated value unless it is edited directly. To restore exact consistency after changing $F$, each of the $n$ independent locations must therefore be synchronized manually. Hence the effective modification complexity grows at least linearly in $n$. ◻
:::

::: lemma
[]{#lem:info-dof label="lem:info-dof"} If the available side-information mechanism exposes at most $2^L$ distinguishable tags while the latent ambiguity class has size $K>2^L$, then no zero-error resolver can identify the true state from those tags alone. Any architecture that still guarantees exact correctness must therefore rely on additional independently accessible discriminating support, moving the system above the single-source regime.
:::

::: proof
*Proof.* Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"} rules out exact recovery from the available tag budget. If exact correctness is nevertheless required, the architecture must supplement the insufficient side-information channel with additional independently accessible information. In the present model, that means leaving the rate-$1$ regime and paying the synchronization cost of Theorem [\[thm:lower-bound\]](#thm:lower-bound){reference-type="ref" reference="thm:lower-bound"}. ◻
:::

## The Unbounded Gap {#sec:gap}

::: theorem
[]{#thm:unbounded-gap label="thm:unbounded-gap"} The ratio between the modification cost of an architecture with $n$ independent encodings and the modification cost of a rate-$1$ architecture diverges: $$\lim_{n\to\infty}\frac{M_{\mathrm{DOF}>1}(n)}{M_{\mathrm{DOF}=1}}=\infty.$$
:::

::: proof
*Proof.* By Theorem [\[thm:upper-bound\]](#thm:upper-bound){reference-type="ref" reference="thm:upper-bound"}, a rate-$1$ architecture has constant effective modification complexity. By Theorem [\[thm:lower-bound\]](#thm:lower-bound){reference-type="ref" reference="thm:lower-bound"}, an architecture with $n$ independent encodings incurs linear cost in $n$. The ratio therefore grows without bound. ◻
:::

Once a fact is duplicated across many independently writable locations, the cost of preserving exact consistency scales with the number of independent copies, not merely with the visibility of the fact.


# Application: Supplementary Case Study Pointer {#sec:empirical}

Supplement A contains case-study details, measurement methodology, and before/after traces. The examples realize independent rate $1$ by combining one authoritative representation with automatically maintained derived views, and they exhibit the $O(1)$ versus $\Omega(n)$ rate-complexity gap of Section [\[sec:bounds\]](#sec:bounds){reference-type="ref" reference="sec:bounds"}.


# Related Work {#sec:related}

## Zero-Error and Side-Information Lineage {#sec:related-source-coding}

Shannon's zero-error framework and its graph-theoretic refinements by Körner and Lovász provide the closest conceptual starting point [@shannon1956zero; @korner1973graphs; @lovasz1979shannon]. Witsenhausen's zero-error side-information problem is also directly relevant because it studies exact recovery under side information through graph coloring and label budgets [@witsenhausen1976zero]. Here the confusable objects are reachable states of a modifiable multi-location encoding rather than channel inputs. The threshold $C_0=1$ is therefore a storage/update analogue of a zero-error feasibility boundary.

Slepian--Wolf coding and classical entropy converses supply the second line of influence [@slepian1973noiseless; @fano1961transmission; @witsenhausen1975conditional; @cover2006elements]. Coding-for-computing and functional-compression results are also close neighbors because they study deterministic decoder targets under side information and graph constraints [@orlitsky2001coding; @doshi2010functional]. In our setting the side information is deterministic rather than stochastic, but it plays the same operational role: exact recovery becomes impossible when the available observation/tag alphabet is too small to separate the remaining alternatives.

## Consistency-Constrained Storage and Interaction {#sec:related-distributed}

Consistency-constrained storage problems, including multi-version coding and related distributed-storage converses, show that correctness requirements impose unavoidable information costs [@rashmi2015multiversion]. Here the object under study is a modifiable encoding architecture, and the main question is the exact-consistency cost of multiple independently writable representations of one latent fact.

## Computational Realizations {#sec:related-meta}

Reflection, metadata systems, and maintenance mechanisms in host platforms provide examples of how the realizability conditions can be instantiated in practice [@kiczales1991art; @smith1984reflection].

## Formal Verification {#sec:related-formal}

The Lean 4 formalization places the work in the tradition of mechanized mathematics and verified theory development [@demoura2021lean4; @leroy2009compcert].


# Conclusion {#sec:conclusion}

Exact consistency in modifiable multi-location encodings yields a deterministic finite converse family. Once the observation transcript leaves a nontrivial ambiguity class, the same finite obstruction can be stated as a confusability-clique bound, a counting converse, a conditional-entropy lower bound, and decoder-output or finite-gap constraints. The same model also supports deterministic data-processing and a budgeted finite-error extension.

The threshold $C_0=1$ is the structural corollary of that converse family: one authoritative location with deterministic views is the unique zero-incoherence regime, and any higher independent rate makes ambiguity reachable. The same obstruction then reappears operationally as manual update cost, yielding $O(1)$ cost at rate $1$ and $\Omega(n)$ cost above it, with an unbounded asymptotic gap. The realizability theorem identifies causal propagation and provenance observability as the host-level conditions under which the rate-$1$ regime is actually attained.

**Limitations and future work.** The present model is finite and exact. Natural extensions include probabilistic coherence, networked encodings with communication constraints, and partial-correlation regimes that move closer to classical stochastic side-information models.

## Artifacts {#sec:data-availability}

The Lean 4 formalization is included as supplementary material. Sections [\[sec:evaluation\]](#sec:evaluation){reference-type="ref" reference="sec:evaluation"} and [\[sec:empirical\]](#sec:empirical){reference-type="ref" reference="sec:empirical"} record theorem-level realizability checks and case-study material.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX/structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean/LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion/exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.


# Mechanized Verification (Lean 4) {#sec:lean .unnumbered}

The main theorem chains are machine-checked in Lean 4.

The artifact covers four groups of claims used in the paper:

-   the foundational threshold chain, including coherence, independent rate, the side-information lower bound, and the finite counting converse;

-   the realizability chain, including derivation, the timing constraint, causal propagation, provenance observability, and the realizability iff theorem;

-   the rate-complexity chain, including the $O(1)$ upper bound, the $\Omega(n)$ lower bound, and the unbounded-gap consequence;

-   the finite entropy layer supporting pair injectivity, clique/fiber budget bounds, conditional-entropy reformulations, deterministic data processing, decoder-output bounds, KL-gap reformulations, and the budgeted finite-error extension.

The supplementary artifact contains the theorem-to-declaration coverage matrix, proof inventory, and build instructions.




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper2_ssot/proofs/`
- Lines: 14898
- Theorems: 703
- `sorry` placeholders: 0
