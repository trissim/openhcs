# Paper: Exact Consistency Under Partial Views: Graph Colorability, Capacity, and Equality in Multi-Location Encodings

**Status**: IEEE Transactions on Information Theory-ready | **Lean**: 24527 lines, 1163 theorems

---

## Abstract

We construct a structural theory of failure for multi-location encodings. Admissible partial views induce a confusability graph on latent tuples; this graph serves as the architecture's exact failure map, recording which state pairs the system cannot structurally distinguish. In the exact coordinate-view model, this induced graph class is already structured: adjacency depends only on coordinate-agreement sets, and coordinatewise alphabet permutations act by graph automorphisms. In this model, exact recovery with a $T$-ary auxiliary tag is equivalent to $T$-colorability of the induced graph, exact recovery on a designated success set is equivalent to colorability of the induced subgraph, and the optimal finite weighted success value is the maximum source mass of a $T$-colorable induced subgraph.

Repeated composition preserves this failure geometry rather than erasing it: the full $n$-block confusability graph is the $n$-fold strong power of the one-shot graph, the normalized block-rate sequence converges to a real asymptotic Shannon-capacity value equal to the supremum of the finite block rates, and that value is bounded above both by the complement chromatic number and by a fixed Lovász-$\vartheta$ upper bound. The same upper invariant admits standard orthonormal-representation, primal-PSD, and dual-theta forms.

The paper also characterizes when this upper theory is sharp. On the cluster-collapse subclass, asymptotic Shannon capacity and the fixed Lovász-$\vartheta$ upper both equal the logarithm of the number of confusability components. For the original view-family model, transitivity of base confusability is the exact sharpness criterion for the current collapse mechanism; a meet-witness condition is sufficient; and fiber coherence is a stronger sufficient condition that sharpens the value to the logarithm of realized transcript fibers. Under an affine restriction on the realized state family, the same framework also yields a fact-side representable matroid: semantic determination is span membership of coordinate functionals, and the minimal determining fact sets are exactly the bases.

Beneath this graph-capacity arc lies a deterministic finite converse: exact zero-error recovery is equivalent to injectivity of the observation-tag map, and the same obstruction appears in counting and entropy forms together with deterministic data processing and a budgeted finite-error extension. This yields a formal integrity guarantee family inside the model: rate $1$ is exactly the structural-integrity regime, higher rate makes integrity violations reachable, and causal propagation together with provenance observability characterize verifiable structural integrity in concrete hosts. A supplementary Lean 4 artifact machine-checks the converse, graph, equality, affine, and realizability chains. **Index Terms:** zero-error information theory, side information, confusability, graph capacity, structural integrity


# Introduction

The mathematically trivial part of multi-location consistency is that once more than one location is independently writable, disagreement can occur. The substantive question is what exact ambiguity structure survives once a system exposes only partial views of its latent state, and what formal integrity guarantees are possible in that regime. This paper answers that question with a zero-error graph-capacity theory for deterministic partial views on latent tuples.

When one latent fact or tuple of facts is represented at several modifiable locations, the available observation need not identify a unique latent state. Under partial views, this surviving ambiguity is not merely a yes/no defect: the admissible view family induces a confusability graph on latent tuples. Its edges record the exact state pairs that the architecture cannot separate, proper colorings quantify the side-information budget needed for exact recovery, and repeated composition pushes the same structure to strong powers and asymptotic Shannon capacity. The paper therefore studies not just whether a multi-location system can fail, but the topology of those failures, the integrity guarantees excluded or permitted by that topology, and the recovery laws forced by it. The resulting theory lies at the intersection of zero-error information theory, side-information source coding, finite converses, and structural integrity analysis [@shannon1956zero; @korner1973graphs; @lovasz1979shannon; @witsenhausen1976zero; @slepian1973noiseless; @cover2006elements].

**Reading roadmap.** The paper has one main theorem arc and two consequence arcs. The main arc studies the failure geometry induced by partial views: graph characterization, block and asymptotic capacity, upper theory, and equality. The first consequence arc isolates the one-shot deterministic converse that this graph theory lifts from clique-shaped ambiguity, and the second uses the same model for a separate fact-side question and for boundary-case realizability consequences. The main invariants and their roles are summarized below.

::: center
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Layer**                    **Question**                                                **Key invariant / result**                                            **Section**
  ---------------------------- ----------------------------------------------------------- --------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  Graph/capacity arc           What exact failure topology do partial views induce?        Colorability, strong powers, asymptotic capacity, $\vartheta$ upper   §[\[sec:graph-characterization\]](#sec:graph-characterization){reference-type="ref" reference="sec:graph-characterization"}--§[\[sec:equality\]](#sec:equality){reference-type="ref" reference="sec:equality"}

  Finite converse              What obstruction survives before graph structure appears?   Injectivity, confusability, counting, entropy, finite error           §[\[sec:converse\]](#sec:converse){reference-type="ref" reference="sec:converse"}

  Fact-side structure          When is one fact semantically determined by others?         Affine representable matroid on fact indices                          §[\[sec:affine\]](#sec:affine){reference-type="ref" reference="sec:affine"}

  Threshold/cost corollaries   What follows at unit rate and above it?                     Unit-rate threshold and manual-update gap                             §[\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"}

  Realizability / readings     When can a host attain and certify unit rate?               Causal propagation + provenance observability                         §[\[sec:requirements\]](#sec:requirements){reference-type="ref" reference="sec:requirements"}--§[\[sec:evaluation\]](#sec:evaluation){reference-type="ref" reference="sec:evaluation"}
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::

**Novelty boundary.** Witsenhausen's zero-error side-information problem and related graph-based source-coding work study exact recovery once the relevant graph or side-information structure is fixed. The new step here is model-side and structural: a deterministic multi-location partial-view architecture on latent tuples generates its confusability graph from admissible views, and the paper identifies the exact recovery, success-set, weighted-success, block-composition, and equality consequences of that induced graph. Once the graph is fixed, the coloring, strong-power, Shannon-capacity, and Lovász-$\vartheta$ machinery is classical. The unit-rate threshold by itself is only a boundary corollary; the substantive contribution begins once the obstruction survives and acquires non-clique topology. The finite converse layer is the deterministic toolkit beneath that arc, while the affine and realizability sections are consequence arcs built on the same model rather than separate primary claims.

The graph-generation mechanism is now more sharply characterized. The exact tuple-space model is not graph-universal: confusability depends only on the coordinate-agreement set of a pair of latent tuples, so the induced graph is an agreement-set graph rather than an arbitrary finite graph. Coordinatewise alphabet permutations act by graph automorphisms, and in fact any latent tuple can be carried to any other by such an automorphism. Hence the induced graphs form a structured subclass of finite graphs. The model already generates non-clique graphs, includes the cluster-collapse subclass on which equality is explicit, and is closed under the block-composition operation that becomes strong product on confusability graphs. The binary square already yields a $4$-cycle rather than a clique, and its iterates give a concrete infinite non-clique strong-product family inside the class.

## Problem Formulation {#sec:encoding-problem}

The model yields an exact-consistency analogue of zero-error resolvability for modifiable encodings. Rate is measured by the number of independently writable locations; we refer to this quantity as the *independent rate* and use *degrees of freedom* (DOF) only as shorthand.

**Failure geometry from partial views.** The main jump of the paper is from clique-shaped ambiguity to architecture-specific failure structure. In the multi-fact partial-observation extension, the confusability graph is the structural object controlling exact consistency under partial views, exact recovery is equivalent to graph colorability, and exact finite weighted success is determined by the maximum $T$-colorable induced subgraph. The binary two-fact square already shows that the induced graph need not be a clique: it is a $4$-cycle, so failures above the unit-rate boundary are structured rather than uniform.

**Asymptotic rate and upper theory.** This graph characterization then lifts to strong powers and asymptotic zero-error rates. The full $n$-block confusability graph is the $n$-fold strong power of the one-shot graph; the normalized block-rate sequence converges to a real asymptotic Shannon-capacity value equal to the supremum of the finite block rates; and that value is bounded above by both the logarithm of the complement chromatic number and a fixed Lovász-$\vartheta$ upper convention. The one-shot upper object matches standard orthonormal-representation, primal-PSD, and dual-theta forms.

**Equality characterization.** The paper also proves an equality characterization. For the original clique-fiber subclass coming from a surjective label map, asymptotic Shannon capacity and the fixed Lovász-$\vartheta$ upper both collapse to $\log |\mathcal B|$, where $\mathcal B$ is the fiber-label alphabet. This equality for cluster-graph structure is classical; the new point here is that transitivity of base confusability is the exact sharpness criterion for the current collapse mechanism in the view-family model, while meet-witnessing and fiber coherence are stronger sufficient conditions. Under these conditions, the asymptotic Shannon capacity and the fixed Lovász-$\vartheta$ upper collapse to the logarithm of the number of connected components or realized transcript fibers, depending on the structure available.

**Finite converse foundation.** Let $X$ be a finite latent source, let $Y$ be the deterministic observation transcript available to a resolver, and let $T$ be a finite auxiliary tag. Exact zero-error recovery from $(Y,T)$ is possible exactly when the pair map $x \mapsto (Y(x),T(x))$ is injective on the surviving ambiguity class. The same obstruction appears as confusability, counting, conditional-entropy, decoder-output, and finite-gap formulations, and it admits a budgeted finite-error extension. In particular, exact $k$-way resolution requires at least $\log_2 k$ bits of side information.

**Boundary corollary.** For deterministic multi-location encodings, the zero-incoherence threshold equals $1$: exact zero-error recovery is possible exactly in the single-source regime. This identifies the unique trivial no-failure corner and, equivalently, the exact structural-integrity regime. The more detailed mathematics begins once the finite obstruction survives and acquires nontrivial confusability structure.

**Secondary fact-side question.** The same framework also supports a second structural question: which fact coordinates are determined by others across the realized state family? Under an affine restriction, that coordinate-dependence question becomes the standard span-membership condition on coordinate functionals, so the induced fact-closure is a representable matroid on fact indices. The value of this specialization here is operational rather than linear-algebraic novelty: the same realized-state model then carries two complementary invariants, a graph on latent states and, in the affine regime, a matroid on coordinates whose rank gives tractable upper bounds on confusability and capacity.

## Terminology and Analogies {#sec:terminology}

The paper introduces terms that have natural analogs in related fields. Table [\[tab:terminology\]](#tab:terminology){reference-type="ref" reference="tab:terminology"} maps the main concepts to help readers anchor the new terminology in familiar frameworks.

::: center
  ---------------------------------------------------------------------------------------------------------------------------
  **This paper**                   **Information theory analog**                    **Distributed systems analog**
  -------------------------------- ------------------------------------------------ -----------------------------------------
  Independent rate $k$ (DOF)       Degrees of freedom / source dimension            Number of "masters" / write quorum size

  Confusability graph $G$          Channel confusion graph / characteristic graph   Consistency violation topology

  Auxiliary tag $T$                Side information / decoder context               Version vector / logical clock

  Unit rate ($k=1$)                Zero-error capacity of 1                         Single source of truth (SSOT)

  Fiber / connected component      Confusion class / equivalence class              Replica group / consistency partition

  Strong power $G^{\boxtimes n}$   $n$-fold channel product                         Multi-round consistency protocol
  ---------------------------------------------------------------------------------------------------------------------------
:::

**Table I.** Terminology mapping across fields. The same mathematical structure appears under different names in zero-error information theory and distributed consistency. []{#tab:terminology label="tab:terminology"}

Readers familiar with any one column can use it as an anchor for the corresponding concepts in this paper.

## Relation to Classical Information Theory {#sec:connection-it}

Three classical lines are particularly relevant.

**Zero-error information theory.** Shannon's zero-error framework and its graph-theoretic refinements describe exact communication under confusability constraints [@shannon1956zero; @korner1973graphs; @lovasz1979shannon]. Our model translates that perspective from one-shot channels to persistent encodings under edits.

**Source coding with side information.** In Slepian--Wolf theory and Witsenhausen's zero-error side-information problem, decoder side information lowers the rate or label budget needed for correct recovery [@slepian1973noiseless; @witsenhausen1976zero]. Here the side information is deterministic rather than stochastic, but it plays the same operational role: exact recovery is impossible unless the available tags or observations separate all remaining alternatives.

**Finite entropy converses.** Classical Fano inequalities and related entropy bounds convert decoding difficulty into information bounds [@fano1961transmission; @witsenhausen1975conditional; @cover2006elements]. Our setting yields a deterministic finite-source analogue in which counting, conditional entropy, decoder-output entropy, and output-gap formulations become different normal forms of the same obstruction.

## Realizability and Verification {#sec:realizability}

The same deterministic model also raises a realizability question, but it is downstream of the main graph arc: when can a concrete host system realize the rate-$1$ boundary case and thereby certify structural integrity? We show that two structural properties are required:

1.  **Causal update propagation:** derived locations must update automatically when the source changes.

2.  **Provenance observability:** the system must expose enough structural information to verify which locations are authoritative and which are derived.

## Paper Organization {#overview}

The main theorem arc is: partial views induce a confusability graph; exact recovery becomes graph colorability; block composition turns that graph into strong powers; the resulting normalized rates converge to the classical Shannon-capacity value of the induced graph; a fixed Lovász-$\vartheta$ upper theory bounds that value; and a structural transitivity criterion identifies when the model-generated graph collapses to the classical cluster-graph equality case. The deterministic converse is the finite foundation of that arc. The affine and realizability results are consequence arcs built on the same model rather than separate primary claims.

Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"} gives the minimum model and threshold setup. Section [\[sec:converse\]](#sec:converse){reference-type="ref" reference="sec:converse"} develops the finite converse foundation. Sections [\[sec:graph-characterization\]](#sec:graph-characterization){reference-type="ref" reference="sec:graph-characterization"}--[\[sec:equality\]](#sec:equality){reference-type="ref" reference="sec:equality"} contain the main graph-capacity arc: graph characterization, block and asymptotic capacity, upper theory, and equality characterization. Section [\[sec:affine\]](#sec:affine){reference-type="ref" reference="sec:affine"} gives the affine fact-side specialization. Section [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"} records threshold and rate-complexity corollaries, Section [\[sec:requirements\]](#sec:requirements){reference-type="ref" reference="sec:requirements"} gives the operational realizability criterion, and Sections [\[sec:evaluation\]](#sec:evaluation){reference-type="ref" reference="sec:evaluation"} and [\[sec:empirical\]](#sec:empirical){reference-type="ref" reference="sec:empirical"} record representative host-level readings and supplementary case-study material. A supplementary Lean 4 artifact machine-checks the converse family, graph extension, theta upper theory, affine fact-matroid specialization, threshold chain, operational realizability criterion, and rate-complexity arguments [@demoura2021lean4].

## Scope {#sec:scope}

The scope is finite facts represented at multiple locations under admissible edits. The focus is on *structural facts*, meaning facts whose encoding locations are fixed when an object, declaration, or artifact is created. This isolates the zero-error regime most clearly and captures a wide class of realizations without making the theory domain-specific.

## Contributions {#sec:contributions}

The main contributions are grouped into the core graph-capacity arc and the consequence arcs built on the same model.

#### Core graph-capacity contributions.

-   **A multi-fact confusability-graph extension** in which partial observations on latent tuples produce non-clique confusability graphs, exact recovery becomes equivalent to graph colorability, success-set exactness becomes equivalent to induced-subgraph colorability, and the exact finite weighted-success value is characterized by the maximum $T$-colorable induced subgraph.

-   **A strong-power block law and asymptotic Shannon-capacity export** in which the $n$-block confusability graph is the $n$-fold strong power of the one-shot graph, so the classical Shannon-capacity framework applies directly to the induced graph family and the normalized block-rate sequence converges to the corresponding asymptotic capacity value.

-   **A fixed Lovász-$\vartheta$ upper theory and standard equivalent forms** in which the same asymptotic capacity is bounded above by complement-chromatic and fixed Lovász-$\vartheta$ upper objects, with standard orthonormal, primal-PSD, and dual-theta forms.

-   **A model-side equality characterization** in which the classical cluster-graph equality case is exported back to the original view-family model under transitivity of confusability, with a meet-witness criterion as a weaker sufficient condition and fiber coherence as a stronger one.

-   **A deterministic finite converse toolkit** for exact consistency, presented through pair injectivity together with confusability, counting, conditional-entropy, decoder-output, and entropy-gap formulations of the same finite obstruction, deterministic data processing, and a budgeted finite-error extension.

#### Consequence contributions.

-   **An affine fact-matroid specialization from the same latent-state framework** in which affine realized-state families induce a representable matroid on fact indices: semantic determination becomes span membership, and the resulting matroid rank yields tractable upper bounds for the main confusability/capacity problem.

-   **Threshold and realizability corollaries** showing that unit rate is the unique zero-incoherence regime, that unit rate is exactly the structural-integrity regime, that unit rate has $O(1)$ manual update cost while higher rates incur an $\Omega(n)$ lower bound, and that causal propagation together with provenance observability gives an operational criterion for realizable verifiable structural integrity in concrete hosts.

-   **A supplementary machine-checked artifact** verifying the theorem chain in Lean 4.


# Figures {#sec:figures}

<figure id="fig:failure-topology">

<figcaption>Two failure topologies. In (a), the observation provides no discrimination, so the confusability graph is a clique: every state is confusable with every other, and exact recovery requires a tag for each state. In (b), partial observations provide some discrimination: opposite corners are distinguishable even though adjacent states are not. The 4-cycle is 2-colorable, so exact recovery requires only 2 tags. This structural difference is invisible to the unit-rate threshold alone.</figcaption>
</figure>


# Model and Basic Quantities {#sec:foundations}

This section introduces only the minimum structure needed to state the deterministic converse foundation and its boundary threshold corollary: latent states, deterministic observations, auxiliary tags, ambiguity classes, and independent rate.

## Model assumptions and notation {#sec:assumptions}

We use the following standing assumptions.

1.  **Finite latent state:** each fact takes values in a finite alphabet.

2.  **Deterministic observations:** a system state exposes a deterministic observation transcript.

3.  **Independent rate:** the independent rate counts the number of independently writable source locations for the same latent fact.

4.  **Finite side information:** any auxiliary tag is drawn from a finite alphabet.

These are the only ingredients needed for the deterministic zero-error theory developed below.

## Latent states, ambiguity, and independent rate {#sec:epistemic}

An *encoding system* for a fact $F$ is a finite collection of locations $\{L_1,\dots,L_n\}$ whose current values jointly determine the observation transcript available to a resolver. For a system state $x$, the *ambiguity class* is the set of latent values still compatible with that transcript; when the ambiguity class has size greater than one, the transcript alone does not identify a unique authoritative value. In this single-fact setting, latent-state ambiguity and value ambiguity coincide because the latent state is just the fact value. In the later multi-fact extension, ambiguity is instead over full latent tuples compatible with the partial observations. A *fact* is an atomic semantic attribute of the represented object that can vary independently of other attributes, and a fact is *structural* if the locations encoding it are fixed when the underlying object, declaration, or artifact is created.

The admissible edit set $E(C)$ is part of the primitive specification of an encoding system $C$. It records which edit events or edit sequences are considered reachable by the model. The paper does not derive $E(C)$ from a particular concurrency policy, protocol, or consensus mechanism; those are downstream instantiations. Independence and independent rate are therefore always defined *relative to the chosen admissible edit model*.

::: definition
[]{#def:independent label="def:independent"} Let $E(C)$ denote the admissible edit set of encoding system $C$. Two locations encoding the same fact are *independent relative to $E(C)$* if some admissible edit sequence in $E(C)$ can change one without forcing the other to match it, equivalently if some admissible edit sequence reaches a state in which the two locations disagree.
:::

::: definition
[]{#def:derived label="def:derived"} A location is *derived* from a source location if updating the source determines the derived value without an additional manual edit.
:::

::: definition
[]{#def:dof label="def:dof"} The *degrees of freedom* (DOF), or *independent rate*, of an encoding system for fact $F$ is the number of pairwise independent source locations encoding $F$ under the specified admissible edit set $E(C)$. In the single-fact model, the remaining encoding locations are treated as derived views of those independent sources.
:::

::: definition
[]{#def:structural-integrity label="def:structural-integrity"} An encoding system has *structural integrity* for fact $F$ if every reachable state is coherent, so no reachable system state presents mutually incompatible encodings of $F$.
:::

::: definition
[]{#def:integrity-violation label="def:integrity-violation"} A reachable *integrity violation* is a reachable incoherent state for fact $F$.
:::

::: proposition
[]{#thm:dof-one-coherence label="thm:dof-one-coherence"} If the independent rate is $1$, then all reachable states are coherent.
:::

::: proof
*Proof.* At independent rate $1$, exactly one location is independent and all remaining locations are derived from it. Derived locations cannot diverge from their source, so disagreement is unreachable. ◻
:::

::: proposition
[]{#thm:dof-gt-one-incoherence label="thm:dof-gt-one-incoherence"} If the independent rate exceeds $1$, then incoherent states are reachable.
:::

::: proof
*Proof.* With at least two independent locations, admissible edits can assign different values to those locations, producing a reachable disagreement state. ◻
:::

## Zero-Incoherence Threshold {#sec:capacity}

::: definition
[]{#def:coherence-capacity label="def:coherence-capacity"} The *zero-incoherence threshold* is the largest independent rate for which incoherent states are unreachable.
:::

::: corollary
[]{#thm:coherence-capacity label="thm:coherence-capacity"} For deterministic multi-location encodings, the zero-incoherence threshold equals $1$.
:::

::: proof
*Proof.* *Achievability.* Suppose $\mathrm{DOF}(C,F)=1$, and let $L_s$ be the unique independent location. By Definition [\[def:dof\]](#def:dof){reference-type="ref" reference="def:dof"}, every other encoding location is non-independent and is therefore treated in this model as a derived view of $L_s$. By Definition [\[def:derived\]](#def:derived){reference-type="ref" reference="def:derived"}, updating $L_s$ determines the value of each such derived location without an additional manual edit. Hence all encoding locations agree in every reachable state, so zero incoherence is achieved.

*Converse.* Suppose $\mathrm{DOF}(C,F)=k>1$. By Definition [\[def:independent\]](#def:independent){reference-type="ref" reference="def:independent"}, there exist at least two independent locations $L_1,L_2$ and an admissible edit sequence that can make them disagree. Equivalently, choose values $v_1\neq v_2$ and apply edits setting $L_1\leftarrow v_1$ and then $L_2\leftarrow v_2$. By independence, the second edit does not force $L_1$ to change. The resulting reachable state satisfies $\mathrm{value}(L_1)\neq \mathrm{value}(L_2)$ and is therefore incoherent.

Thus zero-error recovery is possible exactly at independent rate $1$, so the zero-incoherence threshold equals $1$. ◻
:::

::: corollary
[]{#cor:integrity-threshold label="cor:integrity-threshold"} A deterministic multi-location encoding has structural integrity if and only if its independent rate equals $1$.
:::

::: proof
*Proof.* If the independent rate is $1$, Proposition [\[thm:dof-one-coherence\]](#thm:dof-one-coherence){reference-type="ref" reference="thm:dof-one-coherence"} shows that all reachable states are coherent, hence Definition [\[def:structural-integrity\]](#def:structural-integrity){reference-type="ref" reference="def:structural-integrity"} holds. If the independent rate exceeds $1$, Proposition [\[thm:dof-gt-one-incoherence\]](#thm:dof-gt-one-incoherence){reference-type="ref" reference="thm:dof-gt-one-incoherence"} gives a reachable incoherent state, so structural integrity fails. ◻
:::

This threshold is an early architectural consequence of the deterministic model. It identifies the unique trivial no-failure corner and, equivalently, the exact structural-integrity regime. Above that boundary, integrity violations are reachable by construction. The deeper theorems of the paper begin once one asks how much finite side information is needed whenever the transcript leaves a nontrivial ambiguity class.

::: corollary
[]{#thm:side-info label="thm:side-info"} If a surviving ambiguity class has size $k$, then exact zero-error resolution requires at least $\log_2 k$ bits of side information.
:::

::: proof
*Proof.* Exact resolution of $k$ surviving alternatives requires at least $k$ distinguishable tags, hence at least $\log_2 k$ bits. ◻
:::

The remainder of the paper sharpens this counting statement into a unified finite converse family, asks how that one-shot obstruction acquires non-clique topology under partial views, and then develops the corresponding asymptotic capacity and upper-bound theory.


# Unified Finite Converse {#sec:converse}

This section supplies the one-shot obstruction that the later graph-capacity theory lifts from clique-shaped ambiguity to non-clique failure geometry. Its role is foundational and unifying rather than graph-theoretically novel: in the single-fact setting, every surviving ambiguity class is effectively clique-shaped under the observation transcript, so the same deterministic obstruction can be stated as injectivity, clique counting, entropy, decoder-output, and finite-error bounds. The cleanest zero-error formulation starts from the joint observation-tag map itself. The first theorem gives the basic criterion; the remaining statements package the same obstruction in the forms reused later in the graph and realizability arcs.

::: theorem
[]{#thm:pair-injective label="thm:pair-injective"} Let $X$ range over a finite latent alphabet, let $Y$ be the deterministic observation transcript, and let $T$ be a finite auxiliary tag. Exact zero-error recovery from $(Y,T)$ is possible if and only if the pair map $x \mapsto (Y(x),T(x))$ is injective on the ambiguity class under consideration.
:::

::: proof
*Proof.* Let $\phi(x):=(Y(x),T(x))$.

*Necessity.* If $\phi$ is not injective, then there exist distinct latent states $x_1\neq x_2$ with $\phi(x_1)=\phi(x_2)$. Any deterministic decoder $D$ therefore receives the same input on both states and must satisfy $D(\phi(x_1))=D(\phi(x_2))$. Since $x_1\neq x_2$, the common output cannot equal both latent states, so $D$ errs on at least one of them.

*Sufficiency.* If $\phi$ is injective, define a decoder on the image of $\phi$ by $D(y,t)=\phi^{-1}(y,t)$, with arbitrary extension off the image. Injectivity makes $\phi^{-1}$ single-valued on the image, and for every latent state $x$ we then have $D(Y(x),T(x))=x$. Thus exact zero-error recovery is possible. ◻
:::

::: proposition
[]{#thm:confusability-converse label="thm:confusability-converse"} Fix a deterministic observation transcript and an $L$-bit side-information tag. If $K$ latent states induce the same observation transcript, then exact zero-error decoding on that ambiguity class requires $K$ distinct tag outcomes. Equivalently, a $K$-way ambiguity class forms a confusability clique whose size cannot exceed the available tag alphabet.
:::

::: proof
*Proof.* Apply Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} on an ambiguity class on which the observation is constant. Then injectivity of $(Y,T)$ reduces to injectivity of the tag coordinate alone on that class. Since an $L$-bit tag provides at most $2^L$ outcomes, the clique size cannot exceed $2^L$. ◻
:::

#### Global observation-tag budget.

If the observation alphabet has size $O$ and the auxiliary tag alphabet has size $T$, then exact zero-error recovery on a latent alphabet of size $K$ requires $$K \le O\,T.$$

::: proof
*Proof.* Let $\phi(x)=(Y(x),T(x))$. By Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"}, exact zero-error recovery requires $\phi$ to be injective on the latent alphabet under consideration. The codomain of $\phi$ is the finite product alphabet $\mathcal Y\times\mathcal T$, which has cardinality $|\mathcal Y\times\mathcal T|=OT$. An injective map from a set of size $K$ into a set of size $OT$ can exist only if $K\le OT$. This is exactly the claimed global budget bound. ◻
:::

#### Fiber-level injectivity and cardinality.

Fix an observation value $y$. On the observation fiber $$\mathcal F_y=\{x : Y(x)=y\},$$ exact zero-error recovery forces the tag map to be injective. Consequently, $$|\mathcal F_y| \le |\mathcal T|.$$

::: proof
*Proof.* Fix $y$ and restrict attention to the fiber $\mathcal F_y=\{x:Y(x)=y\}$. On this set the observation coordinate is constant, so for $x,x'\in\mathcal F_y$ one has $$(Y(x),T(x))=(Y(x'),T(x')) \iff T(x)=T(x').$$ Hence Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} implies that exact recovery on $\mathcal F_y$ is possible only if the tag map $x\mapsto T(x)$ is injective on that fiber. Since the image of an injective map cannot be larger than its codomain, one obtains $|\mathcal F_y|\le |\mathcal T|$. ◻
:::

::: proposition
[]{#thm:fano-converse label="thm:fano-converse"} Let $F$ range over $K$ possible latent values. Suppose exact recovery is attempted from a fixed observation transcript together with an $L$-bit side-information tag. Then $$K \le 2^L.$$ Equivalently, exact zero-error recovery of $K$ ambiguous states requires at least $\log_2 K$ bits of side information.
:::

::: proof
*Proof.* This is Theorem [\[thm:confusability-converse\]](#thm:confusability-converse){reference-type="ref" reference="thm:confusability-converse"} specialized to a single $K$-way ambiguity class. ◻
:::

The counting bound is the coarsest form of the obstruction. The next step is to weight the same argument by source mass. Exact recovery still enforces injectivity on successful confusability classes, but now the question is how much entropy can remain once the source is partitioned into success and failure branches.

#### Conditional-entropy formulation.

Let $X$ be uniform on a $K$-way ambiguity class and let $Y$ be the deterministic observation transcript. If $Y$ is constant on that class, then $H(X\mid Y)=\log_2 K$. In particular, any tag-observation pair $(Y,T)$ that resolves the class exactly must satisfy $$H(X\mid Y,T)=0
\qquad\text{and}\qquad
H(T\mid Y)\ge \log_2 K .$$

::: proof
*Proof.* If $Y$ is constant on the $K$-point ambiguity class, then conditioning on $Y$ does not refine the distribution of $X$. Because $X$ is uniform on that class, this gives $H(X\mid Y)=\log_2 K$. Exact recovery from $(Y,T)$ means that $X$ is a deterministic function of $(Y,T)$, so $H(X\mid Y,T)=0$. Using the chain rule, $$H(X\mid Y)=I(X;T\mid Y)+H(X\mid Y,T)=I(X;T\mid Y).$$ Therefore $I(X;T\mid Y)=\log_2 K$. Since conditional mutual information is bounded by conditional entropy, $I(X;T\mid Y)\le H(T\mid Y)$, and the lower bound $H(T\mid Y)\ge \log_2 K$ follows. ◻
:::

#### Mass-weighted clique entropy bound.

Let $S$ be a success set for a zero-error decoder, and suppose $S$ forms a confusability clique under the observation transcript. If $p_S$ denotes the total probability mass of $S$, then the entropy carried by the successful states obeys $$\sum_{x\in S} p(x)\log_2 \frac{1}{p(x)}
\;\le\;
h_2(p_S) + p_S \log_2 |\mathcal T|,$$ where $h_2$ is binary entropy and $|\mathcal T|$ is the tag alphabet size.

::: proof
*Proof.* Let $B$ be the indicator of the event $\{X\in S\}$. Then $\Pr[B=1]=p_S$, so $H(B)=h_2(p_S)$. Decompose the source entropy carried by the successful states by conditioning on $B$: $$H(X)=H(B)+H(X\mid B).$$ Restricting to the success branch $B=1$, exact recovery on the clique $S$ forces injectivity of the tag map on $S$, hence at most $|\mathcal T|$ successful states can be resolved. Therefore $$H(X\mid B=1) \le \log_2 |\mathcal T|.$$ Multiplying by the success probability gives a contribution of at most $p_S\log_2 |\mathcal T|$ from that branch. Combining this with the Bernoulli split term $h_2(p_S)$ yields the displayed inequality for the entropy mass carried by the successful states. ◻
:::

In particular, on any exact success set inside a confusability clique, the restricted observation-tag map is injective by Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"}, so the same entropy inequality applies directly to the successful branch rather than only to the whole clique.

#### Decoder-output and gap formulations.

In the same deterministic finite model, the obstruction can also be expressed as output-entropy and finite-budget gap constraints: any deterministic decoder output $\hat X$ satisfies $$H(\hat X)\le H(Y,T)\le H(X),$$ and the observation-tag entropy and decoded-output entropy are bounded by their corresponding finite alphabet ceilings, so the deficits $$\log_2 |\mathcal Y\times \mathcal T|-H(Y,T),
\qquad
\log_2 |\widehat{\mathcal X}|-H(\hat X)$$ are nonnegative and vanish only in the uniform saturation cases formalized in the supplement.

::: proof
*Proof.* Because the decoder is deterministic, $\hat X=D(Y,T)$ is a deterministic function of the pair $(Y,T)$. The deterministic data-processing statement below therefore gives $$H(\hat X)\le H(Y,T).$$ The pair $(Y,T)$ is itself a deterministic function of the latent source $X$, so another application of deterministic data processing yields $H(Y,T)\le H(X)$. Since $(Y,T)$ takes values in the finite alphabet $\mathcal Y\times\mathcal T$, one also has $$H(Y,T)\le \log_2 |\mathcal Y\times\mathcal T|.$$ Likewise $H(\hat X)\le \log_2 |\widehat{\mathcal X}|$. Subtracting these entropies from their respective alphabet ceilings gives the nonnegative gap quantities claimed in the statement. ◻
:::

#### Deterministic data processing.

Let $\kappa$ be any deterministic coarsening of the observation-tag pair $(Y,T)$. Then $$H(\kappa(Y,T)) \le H(Y,T).$$ In particular, since any deterministic decoder output $\hat X$ is a coarsening of $(Y,T)$, $$H(\hat X)\le H(Y,T)\le \log_2 |\mathcal Y\times \mathcal T|.$$

::: proof
*Proof.* Write $Z=(Y,T)$. If $\kappa$ is deterministic, then $\kappa(Z)$ is a function of $Z$, so the conditional entropy $H(\kappa(Z)\mid Z)$ is zero. By the chain rule, $$H(Z)=H(\kappa(Z),Z)=H(\kappa(Z))+H(Z\mid \kappa(Z)),$$ and since conditional entropy is nonnegative, $H(\kappa(Z))\le H(Z)$. Taking $\kappa$ to be the decoder map gives $H(\hat X)\le H(Y,T)$. Finally, because $Z$ ranges over the finite alphabet $\mathcal Y\times\mathcal T$, one has the standard finite-alphabet ceiling $H(Y,T)\le \log_2 |\mathcal Y\times\mathcal T|$. ◻
:::

This theorem is the structural reason the output and gap formulations are not independent embellishments of the counting converse. Once the observation is coarsened, every derived representation inherits the same obstruction through deterministic data processing rather than escaping it.

::: proposition
[]{#thm:equivalence-viewpoint label="thm:equivalence-viewpoint"} The confusability, counting, conditional-entropy, decoder-output, and finite-gap statements above are different normal forms of the same deterministic finite zero-error obstruction: a surviving $K$-way ambiguity class requires a budget of at least $\log_2 K$ bits to be resolved exactly.
:::

::: proof
*Proof.* Pair injectivity is the structural core. The confusability and counting theorems express the injectivity requirement combinatorially; the conditional-entropy and weighted-entropy theorems express it in source-mass coordinates; the decoder-output and gap theorems express it through deterministic coarsening and finite alphabet ceilings; and the finite-error theorem is the relaxed version obtained by allowing a nonzero failure branch. Each theorem therefore measures the same failure of exact isolation in a different coordinate system. ◻
:::

## Finite-error extension {#sec:finite-error-extension}

The main body of the paper studies exact zero error. The same deterministic finite model also supports a budgeted finite-error extension, which should be read as a deterministic finite analogue of the classical Fano line of argument rather than as a separate asymptotic coding theorem [@fano1961transmission; @witsenhausen1975conditional; @cover2006elements].

::: proposition
[]{#thm:finite-error-budgeted label="thm:finite-error-budgeted"} Let $P_e$ denote the decoder error probability on a finite source of size $K$, observed through a deterministic transcript alphabet of size $O$ together with a tag alphabet of size $T$. Then the decoded-output entropy obeys $$H(\hat X)
\;\le\;
h_2(P_e) + (1-P_e)\log_2 (OT) + P_e \log_2 (K-1).$$
:::

::: proof
*Proof.* Partition the source into success and failure events. On the success branch, the decoded output is constrained by the observation-tag budget and contributes at most $\log_2(OT)$. On the failure branch, the decoder can still output at most one of the remaining $K-1$ alternatives. The Bernoulli split between the two branches contributes the binary-entropy term. Equivalently, $$H(\hat X)=H(\hat X \mid \text{success/failure}) + H(\text{success/failure})$$ is bounded by combining the support bound on each branch with the binary entropy of the branch variable itself. ◻
:::

The zero-error theorems are the $P_e=0$ boundary of this inequality. In that regime the success branch occupies all probability mass, the Bernoulli term vanishes, the failure contribution disappears, and the budget collapses back to the unified finite converse above.

The next section asks what survives when admissible views separate some latent alternatives but not others. In that regime the same one-shot obstruction is no longer represented by a single clique; it becomes an induced confusability graph that can carry genuinely non-clique failure structure.


# Graph Characterization of Partial Views {#sec:graph-characterization}

Before defining the general confusability graph, we develop a concrete example that captures the key structural phenomenon: partial observations can induce nontrivial failure topology.

## A Running Example: The Binary Square {#sec:binary-square}

Consider a system with two binary facts $(x_1, x_2) \in \{0,1\}^2$, giving four latent states: $$(0,0), \quad (0,1), \quad (1,0), \quad (1,1).$$ Suppose the architecture exposes two partial views:

-   View 1 reveals only coordinate $x_1$

-   View 2 reveals only coordinate $x_2$

#### Confusability structure.

Two states are confusable when at least one view fails to separate them:

-   Through View 1: states $(0,0)$ and $(0,1)$ are confusable (both have $x_1=0$), and states $(1,0)$ and $(1,1)$ are confusable (both have $x_1=1$).

-   Through View 2: states $(0,0)$ and $(1,0)$ are confusable (both have $x_2=0$), and states $(0,1)$ and $(1,1)$ are confusable (both have $x_2=1$).

The confusability graph is therefore a 4-cycle: $$(0,0) \sim (0,1) \sim (1,1) \sim (1,0) \sim (0,0),$$ where opposite corners $(0,0)$ and $(1,1)$, as well as $(0,1)$ and $(1,0)$, are *not* adjacent. This is not a clique: the architecture can distinguish some pairs even though it cannot distinguish all pairs.

#### Exact recovery.

Because the graph is 2-colorable (e.g., color by parity $c(x_1,x_2) = x_1 \oplus x_2$), exact recovery is possible with a 2-ary auxiliary tag. However, the graph is not 1-colorable (it contains edges), so a 1-tag decoder must fail on at least one confusable pair. Under the uniform source, the best 1-tag exact success probability is $1/2$ (choose either diagonal as the success set).

#### What this example shows.

1.  Partial views induce *structured* confusability, not uniform ambiguity.

2.  The confusability graph captures exactly which state pairs the architecture cannot separate.

3.  Graph-theoretic properties (chromatic number, independence number) determine exact recovery costs.

Figure [\[fig:failure-topology\]](#fig:failure-topology){reference-type="ref" reference="fig:failure-topology"} contrasts this structured failure with the clique-shaped failure that arises when no observation provides any discrimination. The theorems below generalize this phenomenon to arbitrary multi-fact partial-view systems.

## General Graph Characterization

This section turns the one-shot obstruction into the paper's main structural object: the confusability graph induced by admissible partial views. The single-fact converse family treats ambiguity classes that are effectively clique-shaped under the observation transcript. A natural extension is to let the latent state be a tuple of facts and let each location reveal only a subset of coordinates. Failure then becomes structured: some latent pairs remain indistinguishable and others do not, so the resulting confusability graph need not be complete.

In this extension, ambiguity is now over full latent tuples rather than single fact values: two tuples are confusable when the allowed partial observations fail to separate them. The induced graph is therefore a failure map for the architecture, not merely a bookkeeping device.

The graph can be handled either explicitly or implicitly. If there are $d$ facts over an alphabet of size $q$, then the latent state space has cardinality $q^d$, so any explicit confusability graph materialization already starts from an exponentially large vertex set. A naive explicit construction ranges over at most $q^{2d}$ ordered state pairs. For that reason the natural algorithmic interface is implicit: the formalization packages adjacency as a Boolean confusability oracle on state pairs, and proves that this oracle is equivalent to the confusability relation itself. The theory below is therefore compatible with implicit graph access even when explicit graph materialization is impractical.

The deterministic restriction is also not fundamental at the graph-construction level. The Lean development formalizes a support-overlap extension in which each location may emit any observation from a finite support set, and two states are confusable when some admissible location has overlapping observation support on those states. The present deterministic partial-view model embeds into that broader framework as the singleton-support special case, so the graph construction itself already extends beyond exact deterministic views even though the paper develops only the deterministic architectural arc.

At the same time, the exact current coordinate-subset model is *not* graph-universal on the full tuple space. The formalization shows that equality at a view is equivalent to containment of that view inside the coordinate-agreement set of the two tuples, so confusability depends only on which coordinates agree, not on the actual symbols carried there. Equivalently, coordinatewise alphabet permutations preserve the confusability relation and induce graph automorphisms; in fact, for any two latent tuples there exists such an automorphism carrying one to the other. Thus the induced graphs in the exact tuple model form a structured subclass of finite graphs rather than an arbitrary graph family. In particular, non-vertex-transitive graphs such as a three-vertex path cannot arise as full confusability graphs in this exact model.

::: theorem
[]{#thm:multifact-colorability label="thm:multifact-colorability"} For a finite multi-fact latent tuple observed through a finite family of partial-coordinate views, exact zero-error recovery with a $T$-ary auxiliary tag exists if and only if the induced confusability graph is $T$-colorable.
:::

::: proof
*Proof.* Let $G_{\mathrm{conf}}$ be the confusability graph.

*Necessity.* Suppose exact recovery with $T$ tags is possible. If two latent tuples are adjacent in $G_{\mathrm{conf}}$, then by definition some allowed observation leaves them indistinguishable. Exact recovery therefore cannot assign them the same tag, because the joint observation-tag pair would then coincide on two distinct latent tuples and Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} would be violated. Hence the tag assignment is a proper $T$-coloring of $G_{\mathrm{conf}}$.

*Sufficiency.* Suppose $c$ is a proper $T$-coloring of $G_{\mathrm{conf}}$. Define the auxiliary tag of a latent tuple to be its color. If two latent tuples produce the same observation and the same color, then they are confusable and adjacent, contradicting proper colorability. Therefore the joint observation-color map is injective, and Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} yields an exact decoder. ◻
:::

An edge of $G_{\mathrm{conf}}$ records a specific pair of latent states that the view architecture cannot separate exactly, and a proper coloring is exactly a disambiguation budget that resolves those failures. Exact recovery cost is therefore determined by failure topology, not merely by the fact that more than one source exists.

This statement is packaged through an explicit simple confusability graph object, and for $n$ repeated copies the full block confusability graph is identified with the $n$-fold strong power of the one-shot confusability graph. The zero-error criterion is therefore literally a graph-colorability theorem on an explicit power family rather than only an analogy.

::: theorem
[]{#thm:multifact-success-set label="thm:multifact-success-set"} In the same model, exact decoding on a designated success set $S$ is equivalent to $T$-colorability of the induced confusability subgraph on $S$.
:::

::: proof
*Proof.* Restrict the argument of Theorem [\[thm:multifact-colorability\]](#thm:multifact-colorability){reference-type="ref" reference="thm:multifact-colorability"} to the success set. Exactness is required only on $S$, so the coloring condition is required only on confusable pairs inside $S$. ◻
:::

::: theorem
[]{#thm:multifact-max-success label="thm:multifact-max-success"} For each tag alphabet size $T>0$, there is a largest success-set size $$M_T := \max\{|S| : S \text{ induces a } T\text{-colorable confusability subgraph}\}.$$ An exact decoder with $T$ tags exists on a success set $S$ if and only if $|S|\le M_T$ after replacing $S$ by some $T$-colorable success set of size $M_T$. Under the uniform source, the optimal exact success probability is therefore $M_T/|\mathcal X|$.
:::

::: proof
*Proof.* Theorem [\[thm:multifact-success-set\]](#thm:multifact-success-set){reference-type="ref" reference="thm:multifact-success-set"} identifies exact success sets with induced $T$-colorable subgraphs. Since the latent alphabet is finite, a maximum-cardinality such set exists. The mechanized characterization exposes this optimum directly. In the binary four-cycle witness, the optimum at $T=1$ is exactly two states, so the best uniform exact success probability is $2/4=1/2$. ◻
:::

The same characterization is also available in weighted form. For any finite source weight $w$ on the latent alphabet, the maximum mass of a $T$-colorable induced subgraph is the exact finite success value under budget $T$. This is the weighted version of the same structural claim: under limited budget, the best exact decoder is the largest colorable safe subworld allowed by the induced failure graph.

Equivalently, this optimum can be packaged as an explicit exact finite rate-distortion value for the extended model at tag budget $T$: every exact success set has weighted mass at most this value, and some exact decoder attains it. Thus the extended model no longer has only a converse inequality; at the one-shot level it has an exact weighted success law rather than merely an upper bound.

::: theorem
[]{#thm:multifact-nonclique label="thm:multifact-nonclique"} There exists a binary two-fact system with single-coordinate observation views whose confusability graph is not a clique. In that system, two tags suffice for exact recovery, one tag is impossible, and any one-tag decoder can be exact on at most half of the four latent states under the uniform source.
:::

::: proof
*Proof.* The witness is the four-state binary square with one location revealing the first coordinate and the other revealing the second. States that agree on one coordinate are confusable through the corresponding location, but opposite corners are not confusable, so the graph is a $4$-cycle rather than a clique. An explicit exact two-tag decoder is obtained by the parity coloring $c(x_1,x_2)=x_1\oplus x_2$. With one tag, any exact success set must be an independent set of the cycle, hence has size at most two, yielding success probability at most $1/2$ under the uniform source. ◻
:::

::: theorem
[]{#thm:multifact-product label="thm:multifact-product"} If two partial-observation systems are colorable with tag alphabets of sizes $T_1$ and $T_2$, then their block-composed product system is colorable with a tag alphabet of size $T_1T_2$.
:::

For later use, recall the strong product definition: if $G$ and $H$ are graphs, then $G\boxtimes H$ has vertex set $V(G)\times V(H)$, and $(u,v)$ is adjacent to $(u',v')$ exactly when either $u=u'$ and $v\sim v'$, or $u\sim u'$ and $v=v'$, or $u\sim u'$ and $v\sim v'$. The block-composition theorem is model-specific because the admissible product views generate exactly these three adjacency cases.

::: proof
*Proof.* Take proper colorings of the two component confusability graphs and encode the pair of colors as one color in the product alphabet. If two product states are confusable, then some allowed product view fails to separate them. In the product system this means that either the first coordinates agree and the second coordinates are confusable, or the second coordinates agree and the first coordinates are confusable, or both coordinates are confusable. These are exactly the three adjacency cases in the strong product $G_1\boxtimes G_2$. Hence every confusable product pair is separated by at least one component coloring, and therefore by the paired color as well. ◻
:::

This product law now has both directions in the formal development. The upper direction is the multiplicative coloring construction above. More sharply, once both location alphabets are nonempty, the product confusability graph is exactly the strong product of the two component confusability graphs. The graph object is therefore not just analogous to zero-error graph products; it is literally a strong-product construction for the induced failure map.

The lower direction is clique-based: if the first component contains a confusability clique of size $c_1$ and the second contains one of size $c_2$, then the product system contains a confusability clique of size $c_1c_2$, so any exact product decoder requires at least $c_1c_2$ tags. The same mechanism also propagates to finite-error success sets: the largest exactly decodable success set in the product system is at least the product of the largest exactly decodable success sets in the two components. At the one-shot zero-error coding level, the maximum independent-set code size also obeys the same product lower bound. Thus the extended model now has an honest strong-product graph law together with matching lower/upper budget consequences.

At this point the induced graph class has four explicit structural features. It contains non-clique graphs, as witnessed by the binary square. It contains the cluster-collapse subclass on which the later equality theorem is exact. It is closed under the block-composition operation, which becomes strong product on confusability graphs. And, by the agreement-set characterization above, it is a structured subclass rather than an arbitrary graph family on the full tuple space. The iterated family below packages these features into a concrete infinite class.

#### Example family: iterated binary squares.

The four-state witness above is only the smallest member of a larger model-generated family. Let $W$ denote the binary square system of Theorem [\[thm:multifact-nonclique\]](#thm:multifact-nonclique){reference-type="ref" reference="thm:multifact-nonclique"}. Composing $m$ independent copies of $W$ yields a system with $2m$ binary fact coordinates and $4^m$ latent states whose confusability graph is the $m$-fold strong power $$C_4^{\boxtimes m}.$$ Because each copy of $W$ is exactly recoverable with two tags, repeated application of Theorem [\[thm:multifact-product\]](#thm:multifact-product){reference-type="ref" reference="thm:multifact-product"} gives exact recovery on the $m$-fold product with $2^m$ tags. Conversely, each copy contains a confusability clique of size $2$, so the clique lower bound in the same product theorem yields a confusability clique of size $2^m$ in the product system. Hence every exact decoder for this family requires at least $2^m$ tags, and therefore the exact tag budget is $$T_{\mathrm{exact}}(W^{\otimes m}) = 2^m.$$ This provides an infinite family of non-clique examples, generated directly by the model, on which the graph object, the exact budget, and the strong-product law all interact nontrivially rather than only in the single four-cycle case. At this point the paper has moved decisively beyond the trivial threshold story: above the unit-rate corner, not all failures are equally severe, because the architecture can induce a non-clique topology with a strictly more structured recovery law.

Once exact one-shot recovery is governed by graph colorability, repeated composition naturally lifts the theory to strong graph powers and asymptotic zero-error rates. The next subsection makes that lift explicit.


# Block Composition, Strong Powers, and Asymptotic Capacity {#sec:capacity-theory}

This section studies how the induced failure topology scales. Once exact recovery is governed by colorability of the one-shot confusability graph, repeated composition does not reset the ambiguity; it composes the same failure structure across blocks. The next results identify the correct strong-power object and the resulting asymptotic zero-error rate.

## Motivation: Repeated Composition

What happens if we repeat the partial-view experiment $n$ times? If we compose $n$ independent copies of the same encoding system, does the ambiguity explode uncontrollably, or does it grow at a predictable rate?

The answer depends on the failure topology:

-   If the system were *fully* incoherent (confusability graph is a clique), every block would multiply the confusion. The ambiguity would grow without any exploitable structure.

-   Because our confusability graph has *structure* (as in the 4-cycle example of Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"}), the ambiguity growth is controlled by that structure. Some state pairs remain distinguishable even across multiple blocks.

The key mathematical insight is that $n$-fold block composition turns the confusability graph into its $n$-fold *strong power*: $$G_n \;=\; G^{\boxtimes n}.$$ Independent sets in the strong power relate systematically to independent sets in the base graph, yielding a supermultiplicative growth law that converges to a well-defined asymptotic rate. The theorem below quantifies this rate.

## Block Composition Law

A genuine supermultiplicative block law holds: $$\alpha_{m+n}\;\ge\;\alpha_m\alpha_n,$$ where $\alpha_n$ denotes the largest one-shot zero-error code size in the $n$-block composed system. This is the correct discrete precursor to Shannon-capacity analysis: the admissible code-size sequence is already supermultiplicative at the graph level.

More sharply, the block-level graph itself factors correctly. Confusability between concatenated $(m+n)$-block states is equivalent to adjacency in the strong product of the $m$-block and $n$-block confusability graphs, packaged formally as an explicit graph isomorphism. Thus the repeated-block system is not only bounded by strong-product arguments at the codebook level; it is literally a strong-product graph-power object for the same induced failure map.

#### Why the strong product arises.

The strong product structure may appear surprising at first. One might expect that if block 1 is confusable and block 2 is not, the pair should still be confusable (an "OR-product" over blocks). The key is that block confusability requires a *single location tuple* $\ell = (\ell_1, \ell_2)$ that works *simultaneously* for both blocks:

-   Block 1: observe location $\ell_1$, must match

-   Block 2: observe location $\ell_2$, must match

Two $(m+n)$-block states are confusable iff there exists one location tuple making observations match across *all* blocks. This is not "confusable in block 1 OR confusable in block 2" but rather "$\exists$ location tuple, $\forall$ blocks: match." That universal quantification over blocks is exactly the strong product condition: for each block, either equal or confusable (matching on some location), with at least one block strictly confusable.

#### Concrete example.

Consider two copies of the binary square. State $((0,0), (0,0))$ vs $((0,1), (1,1))$:

-   Block 1: $(0,0) \sim (0,1)$ in $C_4$ (agree on $x_1$)

-   Block 2: $(0,0) \not\sim (1,1)$ in $C_4$ (disagree on both coordinates)

An "OR-product" would mark these confusable because block 1 matches. But the actual model requires *one* location pair $(\ell_1, \ell_2)$ matching *both* blocks. Block 1 can match via $\ell_1 = 1$ (observing $x_1$). Block 2 cannot match: location 1 gives $0$ vs $1$, location 2 gives $0$ vs $1$. No location pair works, so these states are not confusable, exactly as the strong product predicts.

Iterating the same construction yields the derived $n$-block lower bound. If $\alpha_1$ denotes the one-shot maximum independent-set code size, then the mechanized block-power theorem gives $$\alpha_1^n \;\le\; \alpha_n,$$ where $\alpha_n$ is the largest one-shot zero-error code size in the $n$-fold block-composed system. This is the first asymptotic-capacity scaffold in the extended model: repeated block composition already forces the expected exponential growth law before any Shannon-capacity or $\vartheta$-style upper theory is introduced. Once the architecture determines the one-shot failure graph, that same graph already controls the asymptotic lower law.

For notation, write $\Theta(G)$ for the Shannon capacity of a confusability graph $G$ in logarithmic units, and write $\vartheta_\infty(G)$ for the corresponding asymptotic Lovász-$\vartheta$ upper value. When the object is a view family $\mathcal V$ rather than an abstract graph, we write $\Theta(G_{\mathcal V})$ and $\vartheta_\infty(G_{\mathcal V})$ for the same quantities computed on the induced confusability graph $G_{\mathcal V}$.

These normalized block rates can be packaged into an explicit lower asymptotic capacity envelope $$C_* \;:=\; \sup_{n\ge 1} \frac{\log \alpha_n}{n},$$ implemented as an $\mathrm{ENNReal}$ supremum of the positive block-rate sequence. Formally, $\alpha_n$ is identified directly with the maximum finite independent-set size of the $n$-block confusability graph, the block graph is identified with the $n$-fold strong power of the one-shot confusability graph, and the same envelope is proved equal to the graph-side Shannon lower-capacity expression built from those strong powers.

The asymptotic scaffold is also not limited to a lower envelope statement. On the graph side itself, strong-product lower bounds for independent-set cardinality induce monotonicity of normalized strong-power rates along repeated multiples. Specializing those generic graph theorems back to the model-generated confusability graph yields the repeated-block statement below.

Repeating a fixed $m$-block system $k$ times yields $$\alpha_m^k \;\le\; \alpha_{km},$$ and hence $$\frac{\log \alpha_m}{m} \;\le\; \frac{\log \alpha_{km}}{km}.$$ Thus rates along repeated block multiples are monotone from below toward the same capacity envelope. This is still a lower theory rather than a full Shannon-capacity theorem, but it is already a genuine asymptotic graph-rate statement for the induced failure topology.

Once the block confusability graph has been identified with strong powers of the one-shot graph, the asymptotic rate theory is exactly the classical Shannon-capacity framework specialized back to the model-generated graph family.

::: theorem
[]{#thm:asymptotic-capacity label="thm:asymptotic-capacity"} Let $\alpha_n$ denote the largest one-shot zero-error code size in the $n$-block composed system. Then the normalized block rates $$\frac{\log \alpha_n}{n}$$ converge to a real asymptotic Shannon-capacity value, and that value is equal to the supremum of the finite block-rate sequence.
:::

::: proof
*Proof.* The argument rests on two observations: supermultiplicativity of independent-set sizes under strong product, and Fekete's lemma.

*Supermultiplicativity.* Recall that the $n$-block confusability graph $G_n$ is the $n$-fold strong power $G^{\boxtimes n}$ of the one-shot graph $G$. For any two graphs $H$ and $K$, an independent set in $H\boxtimes K$ can be constructed from independent sets in $H$ and $K$ as follows. If $I_H\subseteq V(H)$ and $I_K\subseteq V(K)$ are independent, then the Cartesian product $I_H\times I_K\subseteq V(H)\times V(K)$ is independent in $H\boxtimes K$: if $(u,v)$ and $(u',v')$ are adjacent in the strong product, then either $u=u'$ and $v\sim v'$, or $u\sim u'$ and $v=v'$, or $u\sim u'$ and $v\sim v'$. In each case at least one of $u,u'$ or $v,v'$ is an edge within the respective independent set, which is impossible. Hence $\alpha(H\boxtimes K)\ge\alpha(H)\alpha(K)$. Iterating gives $\alpha_{m+n}=\alpha(G^{\boxtimes(m+n)})\ge\alpha(G^{\boxtimes m})\alpha(G^{\boxtimes n})=\alpha_m\alpha_n$, so the sequence $\{\alpha_n\}$ is supermultiplicative.

*Fekete convergence.* The inequality $\alpha_{m+n}\ge\alpha_m\alpha_n$ is equivalent to subadditivity of the sequence $\{-\log\alpha_n\}$. The trivial bound $\alpha_n\le|V(G)|^n$ ensures that $-\log\alpha_n\ge -n\log|V(G)|$, so the sequence is bounded below. By Fekete's subadditivity lemma, the normalized limit $$\lim_{n\to\infty}\frac{\log\alpha_n}{n}=\sup_{n\ge1}\frac{\log\alpha_n}{n}$$ exists and equals the supremum of the finite block-rate sequence. This limit is the asymptotic Shannon capacity $\Theta(G)$ in logarithmic units.

The mechanized development packages the same argument with explicit sequence objects and convergence lemmas; the textual expansion above makes the independent-set product construction and Fekete application explicit for accessibility. ◻
:::

#### Running example: Binary square.

For the 4-cycle $C_4$ from Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"}, the independence number is $\alpha(C_4) = 2$ (choose either diagonal). The strong power $C_4^{\boxtimes n}$ has $\alpha(C_4^{\boxtimes n}) = 2^n$, giving normalized rate $(\log 2^n)/n = \log 2$. The asymptotic Shannon capacity is therefore $\Theta(C_4) = \log 2$, which matches the one-shot chromatic bound: the 4-cycle is 2-colorable, so exact recovery requires 2 tags at every blocklength.

#### Relation to graph capacity theory.

This places the extended model directly in the classical zero-error graph-capacity setting. The difference from the classical channel picture is generative rather than formal: here the confusability graphs arise from deterministic view families and partial observations on latent tuples. We do not claim a new general Shannon-capacity theorem for arbitrary graphs; the new content is that this partial-view model reaches the classical graph machinery in a non-clique regime, so the architecture determines the graph once and that graph then determines both one-shot and asymptotic recoverability. The clique-fiber regime recovers the classical cluster-graph equality case, while the transitivity criterion identifies when that case is produced by the view-family model. The non-clique witness already shows that the model also generates genuinely nontrivial graph behavior.


# Theta Upper Theory {#sec:theta-theory}

This section supplies the asymptotic upper law for the same induced failure graph. The previous section showed that repeated composition turns view-induced ambiguity into strong powers and hence into an asymptotic recoverability rate. The question now is how large that rate can be. The first upper bound is complement-chromatic. The stronger package identifies the same one-shot upper invariant with standard orthonormal, primal-PSD, and dual-theta forms and then lifts those bounds to strong powers.

The complement-chromatic bound is the first upper bound. The complement of the strong-power confusability graph is identified with the coordinatewise "there exists an offending coordinate" graph on the complement, and a coloring of the one-shot complement graph lifts to a coloring of every strong power by coloring each coordinate separately. Consequently, $$\alpha(G^{\boxtimes n}) \le \chi(\overline G)^n,
\qquad
\frac{\log \alpha(G^{\boxtimes n})}{n} \le \log \chi(\overline G),$$ and the asymptotic capacity is bounded above by $\log \chi(\overline G)$.

At the one-shot level, the same upper invariant is matched with standard witness, orthonormal-representation, primal-PSD, and dual-theta formulations by explicit feasible-object equivalences. Applying these one-shot bounds to strong powers yields subadditive upper-rate sequences, so Fekete's lemma gives asymptotic primal and dual upper values, each equal to the infimum of its finite power-rate bounds. The finite block rates are bounded above by either sequence, hence $$\Theta(G)\le \vartheta_{\infty}(G).$$ The same induced topology that governs exact recovery therefore also carries a classical asymptotic upper theory. The formal development proves these one-shot equivalences and the resulting asymptotic primal/dual upper laws.

What remains open is the sharpness of this upper theory for the graph classes generated by the extended model, not the existence of standard primal--dual theta packaging.


# Equality Characterization {#sec:equality}

This section asks when the upper theory of the induced failure graph is exactly sharp. The graph-theoretic equality facts used here are classical for cluster graphs; the model-specific question is when the partial-view architecture generates exactly that collapse regime. For the current collapse route, that criterion can be characterized exactly.

The original clique-fiber regime is now isolated as an equality subclass rather than only a threshold corollary. If a graph is generated by a surjective label map $x \mapsto b(x)$ with adjacency exactly between distinct vertices in the same fiber, then one representative from each fiber forms an independent set of size $|\mathcal B|$, while the complement graph is colorable by the same label map using $|\mathcal B|$ colors. This equality for cluster-graph structure is classical; the mechanized development packages the model-specific export needed here into a restricted-class theorem: $$C(G)=\vartheta_{\infty}(G)=\log |\mathcal B|.$$ Thus the extended theory now contains both a genuine non-clique graph regime with strict upper/lower separation and a clean equality regime recovering the original fiber picture inside the same asymptotic Lovász-$\vartheta$ framework. That equality is no longer confined to the abstract cluster-graph subclass. The current sharpness criterion is characterized exactly by transitivity of the base confusability relation: $$\text{confusability is transitive}
\iff
G_{\mathrm{conf}}=\mathrm{Cluster}(\Pi_{\mathrm{cc}}).$$ Hence, if confusability is transitive, the base confusability graph is exactly the cluster graph on connected components, and the model-specific equality theorem yields $$\Theta(G_{\mathcal V})=\vartheta_\infty(G_{\mathcal V})=\log |\Pi_{\mathrm{cc}}|,$$ where $\Pi_{\mathrm{cc}}$ is the connected-component partition of the base confusability graph. Between arbitrary base models and full fiber coherence, a checkable sufficient condition is that the view family be *meet-witnessed*: every pair of allowed views contains another allowed view inside their intersection. Under that condition, any two confusability witnesses can be pulled back to a common subview, so confusability is transitive and the same equality follows. Fiber coherence is then a stronger sufficient structural condition: if equality at one allowed location already fixes the full observation transcript, transitivity follows, the connected components are exactly the realized transcript fibers, and the equality sharpens further to $$\Theta(G_{\mathcal V})=\vartheta_\infty(G_{\mathcal V})=\log |\mathcal T_{\mathrm{real}}|.$$ At the witness level, this same collapse admits a local composition form: base confusability is the edge union of the single-view cluster graphs, and transitivity is equivalent to closure of those fixed-view witnesses under two-step composition through an intermediate state. This is the exact local criterion for the current equality mechanism: the collapse to a cluster graph is determined by how single-view witnesses compose, not by an opaque global coincidence of the full confusability graph. Thus the equality question is structural: it identifies when genuinely non-clique failure geometry collapses to exact cluster behavior.

#### Running example: Binary square.

The binary square of Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"} does *not* have transitive confusability: $(0,0) \sim (0,1)$ and $(0,1) \sim (1,1)$, but $(0,0) \not\sim (1,1)$. The 4-cycle is not a cluster graph, so the equality collapse does not apply. Indeed, the asymptotic Shannon capacity $\Theta(C_4) = \log 2$ is strictly less than the Lovász-$\vartheta$ upper bound $\vartheta(C_4) = \log(1+\sqrt{2})$, demonstrating the gap between lower and upper asymptotic theories in the non-transitive regime. This gap is invisible to the one-shot threshold analysis but becomes explicit in the asymptotic theory.

**Hierarchy summary.** This equality route has one exact criterion and two stronger sufficient conditions. The table below records only the logical structure and the resulting capacity value.

::: center
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Condition**        **Necessity?**                            **Structural consequence**                                    **Capacity value**
  -------------------- ----------------------------------------- ------------------------------------------------------------- -------------------------------------
  Fiber coherence      No; stronger sufficient condition         Connected components are realized transcript fibers           $\log |\mathcal T_{\mathrm{real}}|$

  Meet-witness         No; sufficient for collapse               Base confusability collapses to the component cluster graph   $\log |\Pi_{\mathrm{cc}}|$

  Transitivity         Yes, for the current equality mechanism   $G_{\mathrm{conf}}=\mathrm{Cluster}(\Pi_{\mathrm{cc}})$       $\log |\Pi_{\mathrm{cc}}|$
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::


# Affine Fact-Side Structure {#sec:affine}

This section records a secondary affine specialization of the same model, but it changes questions. The preceding sections study distinguishability of latent states from partial observations. The same latent-state model also has a fact-side structural question: which fact coordinates determine others? Under an affine restriction on the realized state family, that determination problem reduces to standard linear algebra and yields a useful rank geometry for the main graph/capacity problem.

::: proposition
[]{#thm:affine-fact-matroid label="thm:affine-fact-matroid"} Assume the realized state family is affine over a field, so that the valid latent tuples form an affine translate of a linear subspace of the ambient fact space. For any fact set $S$ and fact index $i$, semantic determination of $i$ by $S$ is equivalent to membership of the coordinate functional for $i$ in the linear span of the coordinate functionals indexed by $S$. Consequently the fact indices carry a representable matroid: a fact set is independent exactly when its coordinate functionals are linearly independent, and the minimal determining fact sets are exactly the bases, all of common cardinality equal to the rank.
:::

::: proof
*Proof.* Let the affine realized-state family be $A=a_0+V$, where $a_0$ is a fixed origin and $V$ is a linear subspace of the ambient fact space. Fix a fact set $S$ and a fact index $i$.

By definition, fact $i$ is semantically determined by $S$ exactly when for all $x,x'\in A$, agreement on every coordinate in $S$ forces agreement on coordinate $i$. Writing $d=x-x'$, this is equivalent to the statement that every direction vector $d\in V$ whose coordinates in $S$ vanish also satisfies $d_i=0$.

Let $e_j$ denote the coordinate functional for fact $j$. The condition above says precisely that $$V\cap \bigcap_{j\in S}\ker(e_j)\subseteq \ker(e_i).$$ In finite-dimensional linear algebra, this is equivalent to $e_i$ lying in the span of the coordinate functionals $\{e_j:j\in S\}$. Thus semantic determination by $S$ is exactly span membership of the corresponding coordinate functional.

The coordinate functionals therefore realize a representable matroid on fact indices. In that matroid, independence is linear independence of the coordinate functionals, bases are the maximal independent spanning sets, and every basis has the common rank cardinality. Since span corresponds exactly to semantic determination, the minimal determining fact sets are precisely those bases. ◻
:::

The matroid and the confusability graph are natural companions induced by the same realized-state model. The affine representable matroid is not merely an aside: under the common and transparent specialization to finite affine families with coordinate-projection views, the matroid rank provides *computationally tractable* upper bounds on confusability and capacity. The utility of this matroid structure is immediate: the matroid rank $t(S)$ provides a computationally tractable upper bound on the Shannon capacity of the induced confusability graph (Corollary [\[cor:matroid-capacity-bounds\]](#cor:matroid-capacity-bounds){reference-type="ref" reference="cor:matroid-capacity-bounds"}), bypassing the hardness of the general capacity computation. Computing Shannon capacity for general graphs is hard (the independence number is NP-hard, and even the Lovász-$\vartheta$ bound requires semidefinite programming). In contrast, matroid rank is computable by Gaussian elimination. The statements below show that $t(S)$, a simple linear-algebraic quantity, bounds $\alpha(G)$ and $\Theta(G)$ from above, giving polynomial-time capacity estimates for this structured subclass.

## View-fiber dimensions and capacity {#sec:affine-to-graph}

Let $A=a_0+V$ with $V$ an $r$-dimensional linear subspace over a finite field $\mathbb{F}_q$, and let admissible views be coordinate-subset projections. For a coordinate set $S$ write $t(S)$ for the rank of the corresponding coordinate functionals on $V$.

::: proposition
Let $A$ and $t(S)$ be as above. Each projection-fiber for view $S$ is an affine subspace of dimension $r-t(S)$ and size $q^{r-t(S)}$. The single-view confusability graph $G_S$ is therefore a disjoint union of $q^{t(S)}$ cliques each of size $q^{r-t(S)}$. In particular $$\alpha(G_S)=q^{t(S)},\qquad \Theta(G_S)=t(S)\log q,$$ where $\alpha$ is the independence number and $\Theta$ is Shannon capacity (logarithms in the same base used elsewhere).
:::

::: proof
*Proof.* The projection onto coordinates $S$ is an affine-linear map whose linear part restricted to $V$ has rank $t(S)$. Its kernel is therefore a linear subspace of $V$ of dimension $r-t(S)$, so every fiber is an affine translate of that kernel and has cardinality $q^{r-t(S)}$. The fibers partition $A$, and within each fiber every pair of states is confusable under $S$, so $G_S$ is a disjoint union of equal-size cliques. The number of fibers equals $|A|/q^{r-t(S)}=q^{t(S)}$, so one can select one representative per fiber to obtain an independent set of size $q^{t(S)}$, whence $\alpha(G_S)=q^{t(S)}$. The block-power identity for such clustered graphs gives $\alpha(G_S^{\boxtimes n})=q^{nt(S)}$, and normalization yields $\Theta(G_S)=t(S)\log q$. ◻
:::

::: corollary
[]{#cor:matroid-capacity-bounds label="cor:matroid-capacity-bounds"} Let $G$ be the full confusability graph generated by a family of coordinate-subset views $\mathcal S$. Then $$\alpha(G)\le \min_{S\in\mathcal S} q^{t(S)}
\qquad\text{and}\qquad
\Theta(G) \le \min_{S\in\mathcal S} t(S)\log q.$$ Thus the matroid rank function $t(\cdot)$ on coordinate sets supplies explicit upper bounds on one-shot and asymptotic exact-recovery rates for the induced confusability graph.
:::

::: proof
*Proof.* An independent set for $G$ must be independent in each single-view graph $G_S$, hence its size is at most $\alpha(G_S)=q^{t(S)}$ for every $S\in\mathcal S$. The inequalities follow from taking the minimum over $\mathcal S$ and applying the block-power argument from the proposition. ◻
:::

**Remark.** The matroid rank $t(S)$ is the "informational dimension" captured by coordinates in $S$: $t(S)=r$ iff $S$ determines the entire state, and any admissible view containing a basis removes confusability entirely (the induced graph is edgeless). These finite-affine, coordinate-view statements show the matroid is directly applicable to the graph/capacity arc; extensions beyond this specialization are natural directions for future work.

#### Running example: Binary square.

The binary square $\{0,1\}^2$ from Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"} is an affine space over $\mathbb{F}_2$ with $r=2$. Each single-coordinate view has matroid rank $t(\{1\}) = t(\{2\}) = 1$, giving fiber size $2^{2-1} = 2$ and single-view capacity bound $\Theta(G_{\{i\}}) \le 1 \cdot \log 2 = \log 2$. The full confusability graph (the 4-cycle) achieves this bound, so the matroid rank bound is tight in this case. The matroid viewpoint shows that the 4-cycle structure arises precisely because each view captures only half the informational dimension.


# Unit-Rate Structural Interpretation {#sec:ssot}

The main graph-capacity arc studies what happens once ambiguity survives and acquires nontrivial topology. This section returns to the opposite boundary case identified in Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"}: the unit-rate corner in which that ambiguity never appears. It records two direct interpretations of that boundary case: derivation realizes the unit-rate regime, and leaving it produces the manual-update gap proved later in Section [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"}.

## Derivation as the Single-Source Structure {#sec:ssot-def}

An encoding system has *unit independent rate* when $\text{DOF}(C,F)=1$. By Proposition [\[thm:dof-one-coherence\]](#thm:dof-one-coherence){reference-type="ref" reference="thm:dof-one-coherence"}, this is exactly the regime in which every reachable state remains coherent, so the represented fact is determinate and coherence restoration requires only $O(1)$ manual updates. By Proposition [\[thm:dof-gt-one-incoherence\]](#thm:dof-gt-one-incoherence){reference-type="ref" reference="thm:dof-gt-one-incoherence"}, any rate above $1$ admits reachable disagreement states. Thus unit rate is the unique rate guaranteeing exact consistency.

## Reading the Converse at Unit Rate {#sec:info-sketches}

When ambiguity classes are trivial, no auxiliary information is needed and unit rate suffices. When a nontrivial ambiguity class survives the observation transcript, Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"} shows that exact recovery requires a sufficiently large side-information alphabet. If that side information is unavailable but exact correctness is still demanded, additional independently accessible support is required, which is what moves the system above unit rate and leads to the update-cost lower bound of Section [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"}. Derivation is therefore the dependence structure that preserves visible views while removing independent rate.

## Rate and Manual Update Cost {#sec:ssot-vs-m}

An encoding may expose many visible copies of a fact, but only the independent copies govern manual synchronization cost. At unit rate, one manual update changes the authoritative source and derived views follow automatically; above unit rate, each independent location must be synchronized manually. This is the cost interpretation of the threshold.

## Derived Views {#sec:derivation}

::: definition
[]{#def:derivation label="def:derivation"} Location $L_{\text{derived}}$ is *derived from* $L_{\text{source}}$ for fact $F$ iff an update to $L_{\text{source}}$ determines the value of $L_{\text{derived}}$ without an additional manual edit.
:::

Derivation may occur at creation time, build time, commit time, or query time depending on the host system. The abstract point is unchanged: a derived location does not contribute an additional independently writable value for the same fact.

::: proposition
[]{#thm:derivation-excludes label="thm:derivation-excludes"} If $L_{\text{derived}}$ is derived from $L_{\text{source}}$, then $L_{\text{derived}}$ cannot diverge from $L_{\text{source}}$ and does not contribute to DOF.
:::

::: proof
*Proof.* By Definition [\[def:derivation\]](#def:derivation){reference-type="ref" reference="def:derivation"}, the value at $L_{\text{derived}}$ is determined by the value at $L_{\text{source}}$ under admissible updates. There is therefore no reachable state in which the two locations carry incompatible values for the same fact. Since the derived location has no independently writable contribution, it does not increase DOF. ◻
:::

If all encodings of $F$ except one are derived from that one, then Proposition [\[thm:derivation-excludes\]](#thm:derivation-excludes){reference-type="ref" reference="thm:derivation-excludes"} leaves exactly one independent source, hence $\text{DOF}(C,F)=1$ and exact consistency follows from Proposition [\[thm:dof-one-coherence\]](#thm:dof-one-coherence){reference-type="ref" reference="thm:dof-one-coherence"}.

The same dependence structure can be realized in many host systems. Its role here is only to prepare the realizability criterion.


# Threshold and Rate-Complexity Corollaries {#sec:rate-corollaries}

The main graph-capacity arc characterizes the structure of exact failure under partial views. This section records one downstream operational consequence at the opposite boundary: independent rate governs manual synchronization cost in the same finite deterministic model.

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
*Proof.* Each independent location can retain an outdated value unless it is edited directly. After a source update, every one of the $n$ independent locations can therefore witness a distinct stale copy of the same fact. Any exact repair strategy must bring each such location back into agreement, and in the worst case no single manual action can repair two independently writable locations at once. Hence the effective modification complexity grows at least linearly in $n$. ◻
:::

::: lemma
[]{#lem:info-dof label="lem:info-dof"} If the available side-information mechanism exposes at most $2^L$ distinguishable tags while the latent ambiguity class has size $K>2^L$, then no zero-error resolver can identify the true state from those tags alone. Any architecture that still guarantees exact correctness must therefore rely on additional independently accessible discriminating support, moving the system above unit rate.
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


# Operational Realizability Criterion {#sec:requirements}

The main graph-capacity arc characterizes what happens once ambiguity survives and must be resolved structurally. This section asks a downstream operational question about the opposite boundary case: when can a concrete host system realize the unit-rate corner isolated earlier, and what two structural properties are needed for that purpose? In integrity language, the question is when structural integrity can be both enforced and certified by the host itself. The resulting criterion is architectural rather than a new graph-capacity theorem.

A host system realizes the rate-$1$ regime when one location is authoritative and every secondary representation is maintained as a deterministic view. Two capabilities are required:

1.  **Causal update propagation**: updates to the authoritative source must automatically propagate to derived locations.

2.  **Provenance observability**: the system must expose enough structural information to verify which locations are authoritative and which are derived.

::: definition
[]{#def:verifiable-integrity label="def:verifiable-integrity"} An encoding system has *verifiable structural integrity* for structural facts if it both realizes structural integrity in reachable states and exposes enough host-native information to certify that this integrity-preserving dependence structure holds.
:::

## Confusability and Side Information {#sec:confusability}

To connect realizability to information available at verification time, fix a fact $F$ and consider the alternatives that remain compatible with the available observations. These alternatives form the same zero-error obstruction analyzed in Sections [\[sec:converse\]](#sec:converse){reference-type="ref" reference="sec:converse"}--[\[sec:equality\]](#sec:equality){reference-type="ref" reference="sec:equality"}: if the host cannot separate the surviving ambiguity class, exact verification requires additional structural side information.

::: lemma
[]{#lem:confusability-clique label="lem:confusability-clique"} If the confusability graph for $F$ contains a clique of size $k$, then any zero-error side-information scheme distinguishing those $k$ alternatives requires at least $k$ distinct tags, equivalently at least $\log_2 k$ bits of side information.
:::

::: proof
*Proof.* Within a $k$-clique, every pair of alternatives is confusable under the base observation. A zero-error tag assignment must therefore separate all $k$ states. Distinct states cannot share a tag, so at least $k$ tags are required. ◻
:::

If a host system does not expose enough structural information to separate the remaining alternatives, exact verification of the rate-$1$ regime is impossible. In security-adjacent terms, the host cannot certify that an apparently coherent state is genuinely authoritative rather than an undetected stale coincidence.

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
[]{#def:causal-propagation label="def:causal-propagation"} An encoding system has *causal update propagation* if every update to a source location is followed by updates to all locations derived from that source with no reachable intermediate state in which source and derived locations disagree. Equivalently, propagation is part of the same admissible update event for the purposes of reachable-state analysis, rather than an asynchronous repair step with observable stale states.
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

::: corollary
[]{#thm:ssot-iff label="thm:ssot-iff"} An encoding system can achieve verifiable independent rate $1$, equivalently verifiable structural integrity, for structural facts if and only if it provides both causal update propagation and provenance observability.
:::

::: proof
*Proof.* Necessity follows from Theorems [\[thm:causal-necessary\]](#thm:causal-necessary){reference-type="ref" reference="thm:causal-necessary"} and [\[thm:provenance-necessary\]](#thm:provenance-necessary){reference-type="ref" reference="thm:provenance-necessary"}. For sufficiency, assume both properties hold. Causal propagation ensures that every derived location is updated as part of the same structural event as the source; provenance observability then certifies that all secondary encodings are in fact derived from that source. The architecture therefore realizes a verifiable single-source encoding, i.e., independent rate $1$, which is exactly verifiable structural integrity in the sense of Definition [\[def:verifiable-integrity\]](#def:verifiable-integrity){reference-type="ref" reference="def:verifiable-integrity"}. ◻
:::

The next section records representative readings of the theorem on familiar host systems.


# Operational Implications of the Unit-Rate Boundary {#sec:evaluation}

The preceding sections characterized the graph-capacity tradeoffs for independent rate $k > 1$. We now return to the boundary case $k = 1$ (Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"}), which identified the necessary and sufficient conditions for verifiable structural integrity: causal propagation and provenance observability.

While these conditions appear abstract, they correspond to distinct architectural classes in real-world systems.

#### Blind Update (Pattern A).

Systems where independent locations can diverge lack causal propagation. Dependency managers such as npm, Cargo, Poetry, and pnpm are canonical examples: manifest edits do not automatically update the lock artifact. Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"} predicts these systems cannot guarantee structural integrity, which aligns with the observed consistency maintenance costs in practice.

#### Amnesiac Derivation (Pattern B).

Systems that compute derived values but discard the derivation graph lack provenance observability. Under the runtime-only interpretation used here, Rust and TypeScript/JavaScript fall into this pattern because compile-time or decorator-style generation does not survive as host-native provenance in the runtime artifact. Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"} predicts these systems cannot certify integrity, even if it holds accidentally.

#### Coherent Kernel (Pattern C).

Systems supplying both causal propagation and provenance observability realize the verifiable rate-$1$ regime. Engine-maintained materialized views and Python's runtime metaprogramming hooks are representative examples: the derivation is maintained automatically and the derivation structure is inspectable.

These observations suggest that the theoretical unit-rate regime is not merely a mathematical curiosity but a structural property that distinguishes verifiable architectures from unverifiable ones. The majority of common infrastructure falls into Pattern A, Pattern B, or their intersection, so structural incoherence or unverifiability is the norm rather than the exception. A detailed classification of representative systems is provided in Appendix [\[sec:appendix-classification\]](#sec:appendix-classification){reference-type="ref" reference="sec:appendix-classification"}.


# Supplementary Case-Study Pointer {#sec:empirical}

Supplement A contains case-study details, measurement methodology, and before/after traces. The examples realize independent rate $1$ by combining one authoritative representation with automatically maintained derived views, and they exhibit the $O(1)$ versus $\Omega(n)$ rate-complexity gap of Section [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"}.


# Related Work {#sec:related}

## Zero-Error and Side-Information Lineage {#sec:related-source-coding}

Shannon's zero-error framework and its graph-theoretic refinements by Körner and Lovász provide the closest conceptual starting point [@shannon1956zero; @korner1973graphs; @lovasz1979shannon]. Witsenhausen's zero-error side-information problem is directly relevant because it studies exact recovery under side information through graph coloring and label budgets [@witsenhausen1976zero]. Our one-shot coloring statements lie in that lineage, but the contribution here is earlier in the pipeline: admissible partial views on latent tuples generate the confusability graph from the encoding model itself rather than taking the graph as given.

#### Key distinction from Witsenhausen.

It is true that both settings induce confusability graphs from an observation law, so the distinction is not that one graph is "given" and the other is "induced." The real difference is architectural. In Witsenhausen's setting, the observation law is the primitive object and one studies coding for the resulting graph. In our setting, the primitive object is a partial-view architecture on latent tuples, and the paper studies what graph structure that architecture can generate and what recovery laws follow from it. This makes two model-side questions central that are not part of the classical fixed-channel formulation: how the admissible view family constrains the induced graph class, and how changing the view topology changes the resulting failure graph and therefore the recovery budget. In the exact coordinate-view model analyzed here, the induced graphs are already non-arbitrary: confusability depends only on coordinate-agreement sets and is invariant under coordinatewise alphabet permutations, so the architecture realizes a structured subclass rather than an unrestricted graph input. The operational contribution is therefore not a new coloring theorem for arbitrary graphs, but an architectural realization theory that connects partial-view design, induced graph structure, and exact zero-error recovery.

Slepian--Wolf coding and classical entropy converses supply the second line of influence [@slepian1973noiseless; @fano1961transmission; @witsenhausen1975conditional; @cover2006elements]. Coding-for-computing, functional compression, and graph-entropy/characteristic-graph viewpoints are also close neighbors because they study deterministic decoder targets and coding questions once an underlying graph or function structure has been specified [@orlitsky2001coding; @doshi2010functional; @korner1973graphs]. In our setting the side information is deterministic rather than stochastic, but it plays the same operational role: exact recovery becomes impossible when the available observation/tag alphabet is too small to separate the remaining alternatives.

After a confusability graph is fixed, the colorability, strong-power, Shannon-capacity, and Lovász-$\vartheta$ machinery used later is classical. In particular, capacity--$\vartheta$ equality for cluster graphs is classical; the novelty here is identifying transitivity of view-generated confusability as the model-side condition that produces this equality subclass, together with meet-witnessing and fiber coherence as structural sufficient conditions. The paper therefore does not claim a new theorem about arbitrary graphs, but a deterministic partial-view model that generates a structured agreement-set graph class from multi-location encodings, an exact recovery/success-set/weighted-success theory for that model-generated class, and a structural equality route inside it. The paper does not attempt a classification of all graph classes with capacity--$\vartheta$ equality.

## Consistency-Constrained Storage and Interaction {#sec:related-distributed}

Consistency-constrained storage problems, including multi-version coding and related distributed-storage converses, show that correctness requirements impose unavoidable information costs [@rashmi2015multiversion]. Here the object under study is a modifiable encoding architecture, and the main question is the exact-consistency cost of multiple independently writable representations of one latent fact.

## Computational Realizations {#sec:related-meta}

Reflection, metadata systems, and maintenance mechanisms in host platforms provide examples of how the realizability conditions can be instantiated in practice [@kiczales1991art; @smith1984reflection]. These examples are auxiliary to the zero-error theory rather than part of its main novelty claim.

## Formal Verification {#sec:related-formal}

The Lean 4 formalization places the work in the tradition of mechanized mathematics and verified theory development [@demoura2021lean4; @leroy2009compcert].


# Conclusion {#sec:conclusion}

The paper's main contribution is a structural theory of exact failure under deterministic partial views. Admissible view families induce confusability graphs on latent states; those graphs record which latent alternatives the architecture cannot separate, and they govern the exact recovery law. In the multi-fact partial-observation extension, exact recovery becomes colorability, exactness on a success set becomes induced-subgraph colorability, and the exact finite weighted-success value is determined by the largest colorable induced subgraph.

Repeated composition preserves that failure structure rather than erasing it: the block-rate sequence is supermultiplicative, its normalized rates converge by Fekete's lemma to a real asymptotic Shannon capacity equal to the supremum of the finite block rates, and the same capacity is bounded above by complement-chromatic and fixed Lovász-$\vartheta$ upper objects. The one-shot upper invariant is matched with standard orthonormal-representation, primal-PSD, and dual-theta forms.

The upper theory is sharp on a structurally characterized subclass. For the original clique-fiber subclass, asymptotic Shannon capacity, the fixed Lovász-$\vartheta$ upper, and the logarithm of the number of fibers coincide. The same equality exports back to the original view-family model through a broader criterion: transitivity of confusability is the exact sharpness criterion for the current collapse mechanism, a meet-witness condition is sufficient for that collapse, and fiber coherence is a stronger sufficient condition under which the connected components are exactly the realized transcript fibers.

The same latent-state framework also yields a fact-side affine specialization. The observation-side question asks which latent states are distinguishable from partial views; the fact-side question asks which coordinates determine others across the realized-state family. When the valid states form an affine family, the latter question becomes span membership of coordinate functionals, so the fact indices carry a representable matroid and the minimal determining fact sets are exactly the bases. In this affine regime, the same model therefore carries a graph invariant on states and a matroid invariant on coordinates, and the latter supplies tractable upper bounds for the former.

The deterministic finite converse remains the foundation of this theory. Once the observation transcript leaves a nontrivial ambiguity class, the same obstruction can be stated as confusability, counting, conditional-entropy, decoder-output, and finite-gap constraints, together with deterministic data processing and a budgeted finite-error extension. The paper does not claim a new general theorem about arbitrary confusability graphs; rather, it shows that a partial-view encoding model can induce genuinely non-clique failure geometry and reach the classical graph-capacity machinery there. Within that model, these theorems also yield a formal integrity guarantee family: rate $1$ is exactly the structural-integrity regime, higher rate makes integrity violations reachable, and realizability plus provenance characterize when that integrity can be certified by the host.

As downstream consequences of that foundation, unit rate is the unique zero-incoherence regime, any higher independent rate makes ambiguity reachable, and the same obstruction yields an $O(1)$ versus $\Omega(n)$ manual-update gap together with a realizability criterion based on causal propagation and provenance observability. The host-level classification table is included only as a representative reading of that boundary-case criterion.

This boundary-case classification also has an integrity reading. In Pattern A and Pattern B architectures, the host cannot reliably distinguish a valid derived state from an undetected stale state by its own structural interface. The resulting incoherence is therefore not merely a maintenance inconvenience; it is an integrity exposure that leaves stale but plausible states reachable and, in adversarial settings, creates latent attack surface inside the architecture itself.

**Computational representation.** The graph-theoretic formulation is structurally exponential in the number of facts: for $d$ facts over an alphabet of size $q$, the latent state space has cardinality $q^d$, so any explicit confusability graph materialization necessarily starts from an exponentially large vertex set. A naive explicit construction ranges over at most $q^{2d}$ ordered state pairs, formalized in the artifact through the corresponding product-cardinality and pair-count bounds. At the same time, the graph need not be materialized: adjacency is already packaged by an implicit oracle on state pairs, formalized as a Boolean confusability test equivalent to the confusability relation itself. This implicit representation is the natural algorithmic interface for large view families, while the affine matroid bounds provide tractable upper estimates on capacity without explicit graph construction.

**Limitations and future work.** The present paper develops only the finite deterministic architectural arc. For the exact tuple-space coordinate-view model, the induced graphs are already structurally constrained by agreement-set symmetry; what remains open is the sharpness of the upper theory on that subclass and on broader stochastic-support extensions, together with equality questions for those subclasses.

## Artifacts {#sec:data-availability}

The Lean 4 formalization is included as supplementary material. Sections [\[sec:evaluation\]](#sec:evaluation){reference-type="ref" reference="sec:evaluation"} and [\[sec:empirical\]](#sec:empirical){reference-type="ref" reference="sec:empirical"} record representative realizability readings and supplementary case-study material.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX/structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean/LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion/exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.


# Mechanized Verification (Lean 4) {#sec:lean .unnumbered}

The main theorem chains are machine-checked in Lean 4, and the cited proof modules compile with zero `sorry` placeholders. The paper keeps some proofs abbreviated for readability, but full formal proofs of the cited results appear in the supplementary Lean artifact.

For the asymptotic arguments, the artifact formalizes the relevant block power-rate sequences directly and proves the required supermultiplicative or subadditive inequalities on those sequences before invoking Fekete-style convergence lemmas. The asymptotic capacity and asymptotic upper values are therefore obtained inside the formal development from explicit sequence objects rather than only quoted as external limits. The cited asymptotic theorems use classical reasoning in the same way as the corresponding finite graph and entropy arguments.

The artifact covers the theorem packages used in the paper:

-   the threshold and unified-converse package: coherence, independent rate, the side-information lower bound, pair injectivity, clique/fiber/global budget bounds, conditional-entropy and decoder-output reformulations, deterministic data processing, and the budgeted finite-error extension;

-   the graph and asymptotic-capacity package: an explicit confusability graph object, exact recovery as colorability, exact success-set cardinality and weighted success mass via largest colorable induced subgraphs, strong-power identification for block confusability, supermultiplicative block growth, Fekete-style asymptotic Shannon capacity, complement-chromatic upper bounds, and a fixed Lovász-$\vartheta$ upper theory with standard orthonormal, primal-PSD, and dual forms;

-   the equality package: the clique-fiber equality theorem, export back to the original view-family model through transitivity of confusability, the meet-witness sufficient condition, and the stronger fiber-coherence collapse criterion;

-   the affine fact-matroid package: semantic determination as span membership of coordinate functionals, the representable matroid on fact indices, and the identification of minimal determining fact sets with the matroid bases;

-   the downstream consequence package: derivation, the timing constraint, causal propagation, provenance observability, the realizability iff theorem, and the $O(1)$ / $\Omega(n)$ rate-complexity gap.

The supplementary artifact contains the theorem-to-declaration coverage matrix, proof inventory, and build instructions.


# System Classification {#sec:appendix-classification}

This appendix provides a consolidated classification of representative systems according to the architectural patterns identified by Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"}. The pattern labels used in the table are: Pattern A (Blind Update), meaning missing causal propagation; Pattern B (Amnesiac Derivation), meaning missing provenance observability; and Pattern C (Coherent Kernel), meaning both conditions are present and verifiable rate $1$ is realizable.

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Host family**         **Host / Pattern**                                                                              **Class**   **Prop.**   **Prov.**   **Rate 1**   **Mechanism**
  ----------------------- ----------------------------------------------------------------------------------------------- ----------- ----------- ----------- ------------ ----------------------------------------------------------------------------------------------------------------------------------------
  Programming languages   Python [@pep487; @pythondatamodel2025]                                                          C           yes         yes         yes          For the structural facts considered here, class-definition hooks and runtime hierarchy introspection are host-native.

  Databases               Engine-maintained materialized view [@postgresMaterializedViews2025; @postgresPgMatviews2025]   C           yes         yes         yes          The engine maintains the derived relation and exposes catalog metadata for it.

  Programming languages   Rust [@rustref2024]                                                                             B           partial     no          no           Compile-time macro generation exists, but the compiled runtime artifact does not retain derivation provenance.

  Programming languages   TypeScript / JavaScript [@typescriptDecorators2025; @javascriptDecorators2025]                  B           partial     no          no           Decorators can attach metadata, but type erasure and missing subclass provenance block exact verification in the host runtime.

  Programming languages   Java [@gosling2021java]                                                                         A/B         no          no          no           Under the runtime-only host interpretation used here, annotations and related derivation tooling are external to the JVM runtime.

  Programming languages   Go [@gospec2024]                                                                                A/B         no          no          no           Reflection exists, but there are no definition-time hooks or host-native implementer registries.

  Databases               External ETL copy / duplicate summary table                                                     A/B         no          no          no           Synchronization is externalized, and provenance is no longer intrinsic to the host database.

  Dependencies            npm [@npmLock2026; @npmExplain2026; @npmLs2026]                                                 A           no          yes         no           Lockfile and query commands expose provenance, but manifest edits require explicit npm operations before the derived state is updated.

  Dependencies            Cargo [@cargoLockGuide2026; @cargoTree2026; @cargoMetadata2026]                                 A           no          yes         no           Resolved-dependency provenance is exposed, but `Cargo.lock` remains a distinct derived artifact synchronized through Cargo commands.

  Dependencies            Poetry [@poetryCli2026]                                                                         A           no          yes         no           The resolved graph is queryable, but lock regeneration is an explicit step rather than part of the source edit itself.

  Dependencies            pnpm [@pnpmInstall2026; @pnpmWhy2026; @pnpmList2026]                                            A           no          yes         no           Provenance is queryable, but the lockfile remains separately maintained rather than automatically updated with source edits.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table A.1.** Classification of representative systems. The same theorem-level coordinates apply uniformly across programming-language runtimes, databases, and dependency managers, so the systems analysis is a single validation table rather than a domain-by-domain survey. []{#tab:host-checks-a-appendix label="tab:host-checks-a-appendix"}




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper2_ssot/proofs/`
- Lines: 24527
- Theorems: 1163
- `sorry` placeholders: 0
