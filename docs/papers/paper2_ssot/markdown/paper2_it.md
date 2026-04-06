# Paper: Exact Recovery Under Deterministic Partial Views: Confusability Graphs, Strong Powers, and Capacity

**Status**: IEEE Transactions on Information Theory-ready | **Lean**: 25924 lines, 1252 theorems

---

## Abstract

We study exact recovery from deterministic partial views of a finite latent tuple. A family of admissible views induces a confusability graph on latent states, and this graph is the structural object governing zero-error recovery. In the exact coordinate-view model on the full labeled tuple space, we characterize the realizable confusability relations exactly: they are precisely those determined by upward-closed families of coordinate-agreement sets. We show that exact recovery with a $T$-ary auxiliary tag is equivalent to $T$-colorability of the induced graph, while exact recovery on a designated success set is equivalent to colorability of the corresponding induced subgraph. Under repeated composition, the block confusability graph is the strong power of the one-shot graph, so the normalized zero-error rates converge to the Shannon capacity of the induced graph and inherit the standard Lovász-$\vartheta$ upper theory. We also identify a structural equality route: when confusability is transitive, the induced graph collapses to a cluster graph, yielding capacity--$\vartheta$ equality, with meet-witnessing and fiber coherence as sufficient conditions. Finally, under an affine restriction on the realized state family, the coordinate side carries a representable matroid whose rank gives tractable upper bounds on confusability and capacity. A classification of representative channel families shows that the majority of widely deployed deterministic partial-view architectures operate above the zero-incoherence boundary, rendering the graph-capacity limits operationally unavoidable.


# Introduction

The substantive question is what exact ambiguity structure survives when a finite latent source is observed only through deterministic partial views, and what zero-error recovery laws are forced by that structure. This paper answers that question with a graph-capacity theory for deterministic partial views on latent tuples. When applied to representative channel families underlying widely deployed infrastructure, the resulting realizability criterion proves that the majority of deterministic partial-view architectures operate outside the verifiable zero-error regime, rendering the preceding capacity limits operationally unavoidable.

The theory is architectural but information-theoretic: once the admissible view family is fixed, the induced confusability graph, exact tag budget, block law, asymptotic capacity, and standard upper bounds are all determined by that structure.

When a latent fact or tuple of facts is exposed only through partial observations, the observation need not identify a unique latent state. The surviving ambiguity is not merely a cardinality obstruction: the admissible view family induces a confusability graph on latent tuples. Its edges record the exact state pairs that the architecture cannot separate, proper colorings quantify the side-information budget needed for exact recovery, and repeated composition lifts the same structure to strong powers and asymptotic Shannon capacity. The resulting theory lies at the intersection of zero-error information theory, side-information source coding, and graph-capacity theory [@shannon1956zero; @korner1973graphs; @lovasz1979shannon; @korner2002zero; @witsenhausen1976zero; @slepian1973noiseless; @cover2006elements; @alon2002source].

**Novelty.** Unlike Witsenhausen's setting where the observation law is given, here a deterministic multi-location partial-view architecture generates that law. The exact coordinate-view model realizes confusability relations determined by upward-closed families of coordinate-agreement sets, and conversely every upward-closed family is realizable. The model includes non-clique graphs, the cluster-collapse equality subclass, and is closed under block-composition (strong product).

## Distributed Source Coding Formulation {#sec:encoding-problem}

The model yields an exact-consistency analogue of zero-error resolvability for modifiable encodings. Rate is measured by the number of independently writable locations; we refer to this quantity as the *independent rate* and use *degrees of freedom* (DOF) only as shorthand.

**Failure geometry from partial views.** The main jump of the paper is from clique-shaped ambiguity to architecture-specific failure structure. In the multi-fact partial-observation extension, the confusability graph is the structural object controlling exact consistency under partial views, exact recovery is equivalent to graph colorability, and exact finite weighted success is determined by the maximum $T$-colorable induced subgraph. The binary two-fact square already shows that the induced graph need not be a clique: it is a $4$-cycle, so failures above the unit-rate boundary are structured rather than uniform.

**Asymptotic rate and upper theory.** This graph characterization then lifts to strong powers and asymptotic zero-error rates. The full $n$-block confusability graph is the $n$-fold strong power of the one-shot graph; the normalized block-rate sequence converges to a real asymptotic Shannon-capacity value equal to the supremum of the finite block rates; and that value is bounded above by both the logarithm of the complement chromatic number and a fixed Lovász-$\vartheta$ upper convention. The one-shot upper object matches standard orthonormal-representation, primal-PSD, and dual-theta forms.

**Equality characterization.** The paper also proves an equality characterization. For the original clique-fiber subclass coming from a surjective label map, asymptotic Shannon capacity and the fixed Lovász-$\vartheta$ upper both collapse to $\log |\mathcal B|$, where $\mathcal B$ is the fiber-label alphabet. This equality for cluster-graph structure is classical; the new point here is that transitivity of base confusability gives a model-side route from the view-family architecture to the cluster-graph equality case, while meet-witnessing and fiber coherence are stronger sufficient conditions. Under these conditions, the asymptotic Shannon capacity and the fixed Lovász-$\vartheta$ upper collapse to the logarithm of the number of connected components or realized transcript fibers, depending on the structure available.

**Finite converse foundation.** Let $X$ be a finite latent source, let $Y$ be the deterministic observation transcript available to a resolver, and let $T$ be a finite auxiliary tag. Exact zero-error recovery from $(Y,T)$ is possible exactly when the pair map $x \mapsto (Y(x),T(x))$ is injective on the surviving ambiguity class. The same obstruction appears as confusability, counting, conditional-entropy, decoder-output, and finite-gap formulations, and it admits a budgeted finite-error extension. In particular, exact $k$-way resolution requires at least $\log_2 k$ bits of side information. This converse family is the one-shot foundation on which the later graph-capacity theory is built.

**Boundary corollary.** For deterministic multi-location encodings, the zero-incoherence threshold equals $1$: the single-source regime is the unique no-failure corner, and it is exactly the structural-integrity regime. In coding-theoretic terms, this is the erasure-correcting regime where the syndrome uniquely determines the message. The richer mathematics begins once the finite obstruction survives and acquires nontrivial confusability structure.

**Secondary fact-side question.** The same framework also supports a second structural question: which fact coordinates are determined by others across the realized state family? Under an affine restriction, that coordinate-dependence question becomes the standard span-membership condition on coordinate functionals, so the induced fact-closure is a representable matroid on fact indices. The value of this specialization here is operational rather than linear-algebraic novelty: the same realized-state model then carries two complementary invariants, a graph on latent states and, in the affine regime, a matroid on coordinates whose rank gives upper bounds on confusability and capacity. Those bounds are algorithmically tractable once the affine family is presented explicitly by a basis or generator matrix for its direction space; under that representation, the relevant rank quantity is exactly the rank of the restricted coordinate projection and satisfies formal size bounds such as $t(S)\le |S|$, so the computation reduces to Gaussian elimination.

## Boundary Interpretation {#sec:realizability}

The same deterministic model also has a boundary interpretation at rate $1$: when can a concrete host system realize the unique no-ambiguity regime and certify that it has done so? We show that two structural properties are required:

1.  **Causal update propagation:** derived locations must update automatically when the source changes.

2.  **Provenance observability:** the system must expose enough structural information to verify which locations are authoritative and which are derived.

## Paper Organization {#overview}

The main theorem family is: partial views induce a confusability graph; exact recovery becomes graph colorability; block composition turns that graph into strong powers; the resulting normalized rates converge to the classical Shannon-capacity value of the induced graph; a fixed Lovász-$\vartheta$ upper theory bounds that value; and transitivity of confusability gives the model-side export to the classical cluster-graph equality case. The deterministic converse is the finite foundation of that family. The affine and realizability sections are later specializations and boundary consequences of the same model rather than separate primary claims.

Section [\[sec:foundations\]](#sec:foundations){reference-type="ref" reference="sec:foundations"} gives the minimum model and threshold setup. Section [\[sec:converse\]](#sec:converse){reference-type="ref" reference="sec:converse"} develops the finite converse foundation. Sections [\[sec:graph-characterization\]](#sec:graph-characterization){reference-type="ref" reference="sec:graph-characterization"}--[\[sec:equality\]](#sec:equality){reference-type="ref" reference="sec:equality"} contain the main graph-capacity arc: graph characterization, block and asymptotic capacity, upper theory, and equality characterization. Section [\[sec:affine\]](#sec:affine){reference-type="ref" reference="sec:affine"} records the affine specialization as a second lens on the same confusability structure. Sections [\[sec:ssot\]](#sec:ssot){reference-type="ref" reference="sec:ssot"}, [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"}, and [\[sec:requirements\]](#sec:requirements){reference-type="ref" reference="sec:requirements"} then interpret the unit-rate boundary and its structural consequences. Appendix [\[sec:appendix-classification\]](#sec:appendix-classification){reference-type="ref" reference="sec:appendix-classification"} records representative host-level classifications and companion case-study details. A supplementary Lean 4 artifact machine-checks the converse family, graph extension, theta upper theory, affine fact-matroid specialization, threshold chain, structural-consequence criterion, and rate-complexity arguments [@demoura2021lean4; @mathlib2020].

## Scope {#sec:scope}

The scope is finite facts represented at multiple locations under admissible edits. The focus is on *structural facts*, meaning facts whose encoding locations are fixed when an object, declaration, or artifact is created. This isolates the zero-error regime and captures a wide class of realizations without making the theory domain-specific.

## Contributions {#sec:contributions}

**Core graph-capacity:** Multi-fact confusability-graph extension where partial observations produce non-clique graphs, exact recovery equals graph colorability, success-set exactness equals induced-subgraph colorability. Strong-power block law: $n$-block confusability graph is the $n$-fold strong power of the one-shot graph. Fixed Lovász-$\vartheta$ upper theory. Model-side equality characterization via transitivity of confusability exporting cluster-graph equality. Deterministic finite converse toolkit (pair injectivity, confusability, counting, conditional-entropy, decoder-output, entropy-gap formulations).

**Boundary consequences:** Affine restriction yields representable matroid on coordinates with rank bounds. Unit rate is the unique zero-incoherence regime with $O(1)$ manual update cost vs $\Omega(n)$ for higher rates. Causal propagation plus provenance observability gives verifiable structural integrity criterion. Zero-incoherence classification of representative channel families (programming languages, databases, dependency managers). Supplementary Lean 4 artifact.


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
[]{#def:dof label="def:dof"} The *degrees of freedom* (DOF) is used as shorthand for the independent rate, defined as the number of pairwise independent source locations encoding $F$ under the specified admissible edit set $E(C)$. In the single-fact model, the remaining encoding locations are treated as derived views of those independent sources.
:::

::: definition
[]{#def:structural-integrity label="def:structural-integrity"} An encoding system has *structural integrity* for fact $F$ if every reachable state is coherent, so no reachable system state presents mutually incompatible encodings of $F$.
:::

::: definition
[]{#def:integrity-violation label="def:integrity-violation"} A reachable *integrity violation* is a reachable incoherent state for fact $F$.
:::

## Zero-Incoherence Threshold {#sec:capacity}

**Remark.** At independent rate $1$, all locations are derived from a single source, so disagreement is unreachable. At independent rate above $1$, admissible edits can assign different values to two independent locations, producing a reachable disagreement state.

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

**Remark.** Equivalently, structural integrity (Definition [\[def:structural-integrity\]](#def:structural-integrity){reference-type="ref" reference="def:structural-integrity"}) holds if and only if the independent rate equals $1$.

This threshold is an early architectural consequence of the deterministic model. It identifies the unique zero-error corner and, equivalently, the exact structural-integrity regime. Above that boundary, integrity violations are reachable by construction. The deeper theorems of the paper begin once one asks how much finite side information is needed whenever the transcript leaves a nontrivial ambiguity class.

The surviving ambiguity also forces an information-theoretic lower bound: exact resolution of a $k$-way class requires at least $\log_2 k$ bits of side information (Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"} below).

The remainder of the paper sharpens this counting statement into a unified finite converse family, asks how that one-shot obstruction acquires non-clique topology under partial views, and then develops the corresponding asymptotic capacity and upper-bound theory.


# Unified Finite Converse {#sec:converse}

This section supplies the one-shot obstruction that the later graph-capacity theory lifts from clique-shaped ambiguity to non-clique failure geometry. Its role is foundational and unifying rather than graph-theoretically novel: in the single-fact setting, every surviving ambiguity class is effectively clique-shaped under the observation transcript, so the same deterministic obstruction can be stated as injectivity, clique counting, entropy, decoder-output, and finite-error bounds. The cleanest zero-error formulation starts from the joint observation-tag map itself.

::: theorem
[]{#thm:pair-injective label="thm:pair-injective"} Let $X$ range over a finite latent alphabet, let $Y$ be the deterministic observation transcript, and let $T$ be a finite auxiliary tag. Zero-error recovery from $(Y,T)$ is possible if and only if the pair map $x \mapsto (Y(x),T(x))$ is injective on the ambiguity class under consideration.
:::

::: proof
*Proof.* Let $\phi(x):=(Y(x),T(x))$.

*Necessity.* If $\phi$ is not injective, then there exist distinct latent states $x_1\neq x_2$ with $\phi(x_1)=\phi(x_2)$. Any deterministic decoder $D$ therefore receives the same input on both states and must satisfy $D(\phi(x_1))=D(\phi(x_2))$. Since $x_1\neq x_2$, the common output cannot equal both latent states, so $D$ errs on at least one of them.

*Sufficiency.* If $\phi$ is injective, define a decoder on the image of $\phi$ by $D(y,t)=\phi^{-1}(y,t)$, with arbitrary extension off the image. Injectivity makes $\phi^{-1}$ single-valued on the image, and for every latent state $x$ we then have $D(Y(x),T(x))=x$. Thus zero-error recovery is possible. ◻
:::

::: theorem
[]{#thm:confusability-converse label="thm:confusability-converse"} Fix a deterministic observation transcript and an $L$-bit side-information tag. If $K$ latent states induce the same observation transcript, then zero-error decoding on that ambiguity class requires $K$ distinct tag outcomes. Equivalently, a $K$-way ambiguity class forms a confusability clique whose size cannot exceed the available tag alphabet.
:::

::: proof
*Proof.* Apply Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} on an ambiguity class on which the observation is constant. Then injectivity of $(Y,T)$ reduces to injectivity of the tag coordinate alone on that class. Since an $L$-bit tag provides at most $2^L$ outcomes, the clique size cannot exceed $2^L$. ◻
:::

::: corollary
[]{#thm:fano-converse label="thm:fano-converse"} Theorem [\[thm:confusability-converse\]](#thm:confusability-converse){reference-type="ref" reference="thm:confusability-converse"} specialized to a single $K$-way ambiguity class gives $K \le 2^L$, i.e., at least $\log_2 K$ bits of side information.
:::

**Proposition III.4 (Equivalent formulations).** The pair-injectivity obstruction of Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} admits the following equivalent formulations on a finite source observed through a deterministic transcript with a finite auxiliary tag:

-   *Global budget:* $K \le O\,T$ for latent size $K$, observation alphabet size $O$, and tag alphabet size $T$.

-   *Fiber injectivity:* $|\mathcal F_y| \le |\mathcal T|$ for each observation fiber $\mathcal F_y=\{x : Y(x)=y\}$.

-   *Conditional entropy:* $H(X \mid Y) = \log_2 K$ on a $K$-way class with constant observation, and $H(T \mid Y) \ge \log_2 K$.

-   *Decoder output:* $H(\hat X) \le H(Y,T) \le H(X)$, with nonnegative gap to alphabet ceilings.

-   *Mass-weighted clique:* $\sum_{x\in S} p(x)\log_2 \frac{1}{p(x)} \le h_2(p_S) + p_S \log_2 |\mathcal T|$ for a success-set clique $S$.

-   *Data processing:* $H(\kappa(Y,T)) \le H(Y,T)$ for any deterministic coarsening $\kappa$.

*Proof sketch.* Each formulation follows from Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} by restricting the injectivity condition to the relevant domain (fiber, clique, full source) and applying standard entropy identities. The confusability, counting, conditional-entropy, decoder-output, and finite-gap statements are therefore different normal forms of the same obstruction: a surviving $K$-way ambiguity class requires a budget of at least $\log_2 K$ bits to be resolved exactly.

## Finite-error extension {#sec:finite-error-extension}

The main body of the paper studies zero error. The same deterministic finite model also supports a budgeted finite-error extension, which should be read as a deterministic finite analogue of the classical Fano line of argument rather than as a separate asymptotic coding theorem [@fano1961transmission; @witsenhausen1975conditional; @cover2006elements; @kelly2011improved].

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

Before defining the general confusability graph, we develop a concrete example: partial observations can induce nontrivial failure topology.

<figure id="fig:failure-topology">

<figcaption>Two failure topologies. In (a), the observation provides no discrimination, so the confusability graph is a clique: every state is confusable with every other, and exact recovery requires a tag for each state. In (b), partial observations provide some discrimination: opposite corners are distinguishable even though adjacent states are not. The 4-cycle is 2-colorable, so exact recovery requires only 2 tags. This structural difference is invisible to the unit-rate threshold alone.</figcaption>
</figure>

## A Running Example: The Binary Square {#sec:binary-square}

Consider a system with two binary facts $(x_1, x_2) \in \{0,1\}^2$, giving four latent states: $$(0,0), \quad (0,1), \quad (1,0), \quad (1,1).$$ View 1 reveals $x_1$ and View 2 reveals $x_2$.

#### Confusability structure.

Two states are confusable when at least one view fails to separate them. View 1 reveals $x_1$, so $(0,0)\sim(0,1)$ and $(1,0)\sim(1,1)$. View 2 reveals $x_2$, so $(0,0)\sim(1,0)$ and $(0,1)\sim(1,1)$. The confusability graph is therefore a 4-cycle: $$(0,0) \sim (0,1) \sim (1,1) \sim (1,0) \sim (0,0),$$ where opposite corners are not adjacent.

#### Exact recovery.

The graph is 2-colorable (e.g., color by parity $c(x_1,x_2) = x_1 \oplus x_2$), so exact recovery is possible with a 2-ary auxiliary tag. With one tag, the best exact success probability under the uniform source is $1/2$.

**Coding-theoretic interpretation.** The parity coloring $c(x_1,x_2)=x_1\oplus x_2$ is a syndrome bit: with $T=2$ we decode perfectly, with $T=1$ we cannot distinguish adjacent vertices. The confusability graph is the confusion graph of a code with parity-check $H=[1\;1]$.

Figure [1](#fig:failure-topology){reference-type="ref" reference="fig:failure-topology"} contrasts this structured failure with clique-shaped failure. The theorems below generalize this to arbitrary multi-fact partial-view systems.

## General Graph Characterization

This section turns the one-shot obstruction into the paper's main structural object: the confusability graph induced by admissible partial views, where two tuples are confusable when allowed observations fail to separate them. For $d$ facts over alphabet size $q$, explicit graph materialization is exponentially large ($q^d$ vertices), so the formalization uses an implicit oracle interface. A naive explicit construction ranges over at most $q^{2d}$ ordered state pairs. The formalization packages adjacency as a Boolean confusability oracle and an equivalent agreement-set oracle, computable in $O(d + \sum_{\ell}|V_\ell|)$ time by computing the agreement set and scanning views.

The deterministic restriction is also not fundamental at the graph-construction level. The Lean development formalizes a support-overlap extension in which each location may emit any observation from a finite support set, and two states are confusable when some admissible location has overlapping observation support on those states.

The exact coordinate-subset model is not graph-universal. Equality at a view equals containment of that view in the coordinate-agreement set, so confusability depends only on which coordinates agree. From any view family we obtain an upward-closed family $\mathcal U := \{S \subseteq [F] : \exists \ell, V_\ell \subseteq S\}$ such that two tuples are adjacent iff their agreement set lies in $\mathcal U$. Conversely, every upward-closed family arises from some coordinate-view architecture. When the alphabet has at least two symbols, every proper coordinate subset occurs as the agreement set of some distinct pair of tuples. Coordinatewise alphabet permutations preserve the confusability relation and induce graph automorphisms. The model realizes vertex-transitive graphs on the full tuple space; non-vertex-transitive graphs such as a three-vertex path cannot arise as full confusability graphs. Classification up to unlabeled graph isomorphism remains open.

::: theorem
[]{#thm:multifact-colorability label="thm:multifact-colorability"} For a finite multi-fact latent tuple observed through a finite family of partial-coordinate views, exact zero-error recovery with a $T$-ary auxiliary tag exists if and only if the induced confusability graph is $T$-colorable [@west2001introduction; @diestel2017graph].
:::

::: proof
*Proof.* Let $G_{\mathrm{conf}}$ be the confusability graph.

*Necessity.* Suppose exact recovery with $T$ tags is possible. If two latent tuples are adjacent in $G_{\mathrm{conf}}$, then by definition some allowed observation leaves them indistinguishable. Exact recovery therefore cannot assign them the same tag, because the joint observation-tag pair would then coincide on two distinct latent tuples and Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} would be violated. Hence the tag assignment is a proper $T$-coloring of $G_{\mathrm{conf}}$.

*Sufficiency.* Suppose $c$ is a proper $T$-coloring of $G_{\mathrm{conf}}$. Define the auxiliary tag of a latent tuple to be its color. If two latent tuples produce the same observation and the same color, then they are confusable and adjacent, contradicting proper colorability. Therefore the joint observation-color map is injective, and Theorem [\[thm:pair-injective\]](#thm:pair-injective){reference-type="ref" reference="thm:pair-injective"} yields an exact decoder. ◻
:::

This statement is packaged through an explicit confusability graph object, and for $n$ repeated copies the full block confusability graph is identified with the $n$-fold strong power of the one-shot confusability graph.

::: theorem
[]{#thm:multifact-success-set label="thm:multifact-success-set"} In the same model, exact decoding on a designated success set $S$ is equivalent to $T$-colorability of the induced confusability subgraph on $S$.
:::

::: proof
*Proof.* Restrict the argument of Theorem [\[thm:multifact-colorability\]](#thm:multifact-colorability){reference-type="ref" reference="thm:multifact-colorability"} to the success set. Exactness is required only on $S$, so the coloring condition is required only on confusable pairs inside $S$. ◻
:::

::: theorem
[]{#thm:multifact-max-success label="thm:multifact-max-success"} For each tag alphabet size $T>0$, there is a largest success-set size $$M_T := \max\{|S| : S \text{ induces a $T$-colorable subgraph}\}.$$ An exact decoder with $T$ tags exists on a success set $S$ if and only if $|S|\le M_T$ after replacing $S$ by some $T$-colorable success set of size $M_T$. Under the uniform source, the optimal exact success probability is therefore $M_T/|\mathcal X|$.
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

The strong product of graphs $G$ and $H$, denoted $G\boxtimes H$, is defined as follows. The vertex set is $V(G)\times V(H)$. Two vertices $(u,v)$ and $(u',v')$ are adjacent exactly when either $u=u'$ and $v\sim v'$, or $u\sim u'$ and $v=v'$, or $u\sim u'$ and $v\sim v'$. The block-composition theorem is model-specific because the admissible product views generate exactly these three adjacency cases [@imrich2008product; @west2001introduction].

::: proof
*Proof.* Take proper colorings of the two component confusability graphs and encode the pair of colors as one color in the product alphabet. If two product states are confusable, then some allowed product view fails to separate them. In the product system this means that either the first coordinates agree and the second coordinates are confusable, or the second coordinates agree and the first coordinates are confusable, or both coordinates are confusable. These are exactly the three adjacency cases in the strong product $G_1\boxtimes G_2$. Hence every confusable product pair is separated by at least one component coloring, and therefore by the paired color as well. ◻
:::

The product confusability graph is exactly the strong product of the two component graphs. Clique-based lower bounds propagate to products: if the first component contains a confusability clique of size $c_1$ and the second contains one of size $c_2$, then the product contains a clique of size $c_1c_2$.

The induced graph class has four structural features: it contains non-clique graphs (binary square witness), the cluster-collapse subclass on which the equality theorem is exact, closure under block-composition (strong product), and is exactly a monotone agreement-set class.

#### Example family: iterated binary squares.

Let $W$ be the binary square from Theorem [\[thm:multifact-nonclique\]](#thm:multifact-nonclique){reference-type="ref" reference="thm:multifact-nonclique"}. The $m$-fold product has confusability graph $C_4^{\boxtimes m}$ and exact tag budget $T_{\mathrm{exact}}(W^{\otimes m}) = 2^m$. This gives an infinite family of non-clique examples on which the graph object, exact budget, and strong-product law interact nontrivially.

Once exact one-shot recovery is governed by graph colorability, repeated composition lifts the theory to strong graph powers and asymptotic zero-error rates. The next subsection makes that lift explicit.


# Block Composition, Strong Powers, and Asymptotic Capacity {#sec:capacity-theory}

This section studies how the induced failure topology scales. Once exact recovery is governed by colorability of the one-shot confusability graph, repeated composition does not reset the ambiguity; it composes the same failure structure across blocks. The next results identify the correct strong-power object and the resulting asymptotic zero-error rate.

## Motivation: Repeated Composition

Under $n$-fold block composition, the confusability graph becomes its $n$-fold *strong power*: $G_n = G^{\boxtimes n}$. Unlike the clique case where ambiguity explodes, structured graphs yield controlled growth via supermultiplicative independent-set behavior. The theorem below quantifies the resulting asymptotic rate.

## Block Composition Law

A genuine supermultiplicative block law holds: $$\alpha_{m+n}\;\ge\;\alpha_m\alpha_n,$$ where $\alpha_n$ denotes the largest one-shot zero-error code size in the $n$-block composed system. This is the correct discrete precursor to Shannon-capacity analysis: the admissible code-size sequence is already supermultiplicative at the graph level.

More sharply, the block-level graph itself factors correctly. Confusability between concatenated $(m+n)$-block states is equivalent to adjacency in the strong product of the $m$-block and $n$-block confusability graphs, packaged formally as an explicit graph isomorphism. Thus the repeated-block system is not only bounded by strong-product arguments at the codebook level; it is literally a strong-product graph-power object for the same induced failure map.

#### Strong-product quantification.

Block confusability requires one location tuple working *simultaneously* for both blocks. Iterating yields $\alpha_1^n \le \alpha_n$.

For notation, $\Theta(G)$ denotes Shannon capacity of confusability graph $G$ and $\vartheta_\infty(G)$ the asymptotic Lovász-$\vartheta$ upper value. The lower asymptotic capacity envelope $$C_* \;:=\; \sup_{n\ge 1} \frac{\log \alpha_n}{n},$$ is identified with the maximum independent-set size of the $n$-block confusability graph.

Strong-product lower bounds give monotonicity: $\alpha_m^k \le \alpha_{km}$ and $\frac{\log \alpha_m}{m} \le \frac{\log \alpha_{km}}{km}$.

::: theorem
[]{#thm:asymptotic-capacity label="thm:asymptotic-capacity"} Let $\alpha_n$ denote the largest one-shot zero-error code size in the $n$-block composed system. Then the normalized block rates $$\frac{\log \alpha_n}{n}$$ converge to a real asymptotic Shannon-capacity value, and that value is equal to the supremum of the finite block-rate sequence.
:::

::: proof
*Proof.* The $n$-block confusability graph is $G_n = G^{\boxtimes n}$. For any graphs $H,K$, the Cartesian product of independent sets $I_H\times I_K$ is independent in $H\boxtimes K$, giving $\alpha(H\boxtimes K)\ge\alpha(H)\alpha(K)$. Iterating yields supermultiplicativity $\alpha_{m+n}\ge\alpha_m\alpha_n$, equivalent to subadditivity of $\{-\log\alpha_n\}$. Boundedness follows from $\alpha_n\le|V(G)|^n$. Fekete's lemma [@fekete1923uber] gives $$\lim_{n\to\infty}\frac{\log\alpha_n}{n}=\sup_{n\ge1}\frac{\log\alpha_n}{n},$$ the asymptotic Shannon capacity $\Theta(G)$. ◻
:::

#### Running example: Binary square.

For the 4-cycle $C_4$ from Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"}, $\alpha(C_4) = 2$ and $\alpha(C_4^{\boxtimes n}) = 2^n$, giving normalized rate $\log 2$ and asymptotic Shannon capacity $\Theta(C_4) = \log 2$. This matches the iterated binary-square family: one-shot budget is $2$, $m$-block budget grows as $2^m$.

#### Relation to graph capacity theory.

This places the extended model in the classical zero-error graph-capacity setting. Confusability graphs arise from deterministic view families on latent tuples; the architecture determines the graph, which then determines recoverability. The clique-fiber regime recovers the cluster-graph equality case via transitivity of confusability.


# Theta Upper Theory {#sec:theta-theory}

This section supplies the asymptotic upper law for the same induced failure graph. The previous section showed that repeated composition turns view-induced ambiguity into strong powers and hence into an asymptotic recoverability rate. The question now is how large that rate can be. The first upper bound is complement-chromatic. The stronger package identifies the same one-shot upper invariant with standard orthonormal, primal-PSD, and dual-theta forms and then lifts those bounds to strong powers [@lovasz1979shannon; @grotschel1981ellipsoid; @knuth1994remark; @boyd2004convex].

The complement-chromatic bound is the first upper bound. The complement of the strong-power confusability graph is identified with the coordinatewise "there exists an offending coordinate" graph on the complement, and a coloring of the one-shot complement graph lifts to a coloring of every strong power by coloring each coordinate separately. Consequently, $$\alpha(G^{\boxtimes n}) \le \chi(\overline G)^n,
\qquad
\frac{\log \alpha(G^{\boxtimes n})}{n} \le \log \chi(\overline G),$$ and the asymptotic capacity is bounded above by $\log \chi(\overline G)$.

At the one-shot level, the same upper invariant is matched with standard witness, orthonormal-representation, primal-PSD, and dual-theta formulations by explicit feasible-object equivalences. Applying these one-shot bounds to strong powers yields subadditive upper-rate sequences, so Fekete's lemma [@fekete1923uber] gives asymptotic primal and dual upper values, each equal to the infimum of its finite power-rate bounds. The finite block rates are bounded above by either sequence, hence $$\Theta(G)\le \vartheta_{\infty}(G).$$ The same induced topology that governs exact recovery therefore also carries a classical asymptotic upper theory. The formal development proves these one-shot equivalences and the resulting asymptotic primal/dual upper laws.

Interpretationally, the complement-chromatic bound is the first combinatorial ceiling, while the Lovász-$\vartheta$ package supplies the sharper geometric and semidefinite upper theory attached to the same confusability graph. Once a deterministic partial-view architecture generates a failure graph, the standard zero-error upper hierarchy attaches to that graph without further model-specific modification [@lovasz1979shannon; @haemers1978upper; @cubitt2014bounds; @zuiddam2019asymptotic].

What remains open is the sharpness of this upper theory for the graph classes generated by the extended model, not the existence of standard primal--dual theta packaging.


# Equality Characterization {#sec:equality}

This section asks when the upper theory of the induced failure graph is exactly sharp. The graph-theoretic equality facts used here are classical for cluster graphs; the model-specific question is when the partial-view architecture generates that collapse regime. Transitivity of confusability is the key structural condition exporting the view-family model to the cluster-graph equality case.

**Open problem.** A precise open problem is to characterize which upward‑closed agreement‑set families produce transitive confusability, and therefore trigger the cluster‑graph equality collapse. We have identified boundary facts: the binary square is a small witness where transitivity fails, while trivial single‑view families and the fiber‑coherent/meet‑witnessed families provably produce transitive confusability and hence the equality collapse. A complete classification of upward‑closed families that yield transitivity remains open and is an attractive direction for further work.

The original clique-fiber regime is now isolated as an equality subclass rather than only a threshold corollary. If a graph is generated by a surjective label map $x \mapsto b(x)$ with adjacency exactly between distinct vertices in the same fiber, then one representative from each fiber forms an independent set of size $|\mathcal B|$, while the complement graph is colorable by the same label map using $|\mathcal B|$ colors. This equality for cluster-graph structure is classical; the mechanized development packages the model-specific export needed here into a restricted-class theorem: $$C(G)=\vartheta_{\infty}(G)=\log |\mathcal B|.$$ Thus the extended theory now contains both a genuine non-clique graph regime with strict upper/lower separation and a clean equality regime recovering the original fiber picture inside the same asymptotic Lovász-$\vartheta$ framework. In particular, transitivity of confusability exports the model-generated graph directly to the classical cluster-graph setting through the following mechanized implication: $$\text{confusability is transitive}
\Longrightarrow
G_{\mathrm{conf}}=\mathrm{Cluster}(\Pi_{\mathrm{cc}}).$$ Hence, if confusability is transitive, the base confusability graph is exactly the cluster graph on connected components, and the model-specific equality theorem yields $$\Theta(G_{\mathcal V})=\vartheta_\infty(G_{\mathcal V})=\log |\Pi_{\mathrm{cc}}|,$$ where $\Pi_{\mathrm{cc}}$ is the connected-component partition of the base confusability graph. Between arbitrary base models and full fiber coherence, a checkable sufficient condition is that the view family be *meet-witnessed*: every pair of allowed views contains another allowed view inside their intersection. Under that condition, any two confusability witnesses can be pulled back to a common subview, so confusability is transitive and the same equality follows. Fiber coherence is then a stronger sufficient structural condition: if equality at one allowed location already fixes the full observation transcript, transitivity follows, the connected components are exactly the realized transcript fibers, and the equality sharpens further to $$\Theta(G_{\mathcal V})=\vartheta_\infty(G_{\mathcal V})=\log |\mathcal T_{\mathrm{real}}|.$$ At the witness level, this same collapse admits a local composition form: base confusability is the edge union of the single-view cluster graphs, and transitivity is equivalent to closure of those fixed-view witnesses under two-step composition through an intermediate state. This is the exact local criterion for the current equality mechanism: the collapse to a cluster graph is determined by how single-view witnesses compose, not by an opaque global coincidence of the full confusability graph. Thus the equality question is structural: it identifies when genuinely non-clique failure geometry collapses to exact cluster behavior.

#### Running example: Binary square.

The binary square of Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"} does *not* have transitive confusability: $(0,0) \sim (0,1)$ and $(0,1) \sim (1,1)$, but $(0,0) \not\sim (1,1)$. The 4-cycle is not a cluster graph, so the transitivity-based equality collapse does not apply. In this special case the classical graph invariants still satisfy $\Theta(C_4)=\vartheta_\infty(C_4)=\log 2$, but that coincidence no longer comes from the cluster-collapse mechanism isolated in this section. The binary square therefore marks the boundary of the paper's equality route: it is the first non-clique witness, yet not a transitive one.

**Hierarchy summary.** This equality route has one transitivity-based sufficient condition and two stronger sufficient conditions. The table below records only the logical structure and the resulting capacity value.

::: center
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Condition**        **Logical status**                         **Structural consequence**                                    **Capacity value**
  -------------------- ------------------------------------------ ------------------------------------------------------------- -------------------------------------
  Fiber coherence      Stronger sufficient condition              Connected components are realized transcript fibers           $\log |\mathcal T_{\mathrm{real}}|$

  Meet-witness         Sufficient for transitivity and collapse   Base confusability collapses to the component cluster graph   $\log |\Pi_{\mathrm{cc}}|$

  Transitivity         Sufficient collapse condition used here    $G_{\mathrm{conf}}=\mathrm{Cluster}(\Pi_{\mathrm{cc}})$       $\log |\Pi_{\mathrm{cc}}|$
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------
:::


# Affine Dual: Coordinate Matroid {#sec:affine}

The preceding sections developed the main graph-capacity and equality arc for partial views. Computing the resulting graph quantities is hard in general. This section isolates a regime in which the coordinate side of the model becomes tractable: when the realized state family is affine, semantic determination reduces to standard linear algebra, and the resulting upper bounds can be computed by matrix rank rather than by solving a general graph-capacity problem.

The resulting object is a representable matroid on fact indices. It is the coordinate-side dual of the same realized-state structure that generates the confusability graph, and it provides computationally tractable upper bounds on confusability and capacity. The tractability claim is representation-sensitive: throughout this section, the affine family $A=a_0+V$ is assumed to be given by an explicit linear presentation of $V$, for example a basis or generator matrix over $\mathbb F_q$, together with the explicit coordinate-subset view family. Under that input model, the relevant quantity is just the rank of a restricted coordinate map and is computable by Gaussian elimination.

#### Warm-up example.

Before stating the general proposition, consider the binary square from Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"}. Here the realized state family is the full affine space $A=\mathbb F_2^2$, so we may take $a_0=(0,0)$ and direction space $V=\mathbb F_2^2$ with generator matrix $$G = \begin{bmatrix}1 & 0 \\\ 0 & 1\end{bmatrix}.$$ The coordinate functionals are $e_1=[1\ \ 0]$ and $e_2=[0\ \ 1]$. A single-coordinate view, say $S=\{1\}$, has rank $t(S)=1$: it captures one of the two available dimensions, so it reveals exactly half of the informational dimension of the state space. This is the finite-affine meaning of the later matroid rank bound: the coordinate-side quantity $t(S)$ measures how much of the state survives the view, and in the affine regime it is obtained by ordinary matrix rank.

::: proposition
[]{#thm:affine-fact-matroid label="thm:affine-fact-matroid"} Assume the realized state family is affine over a field, so that the valid latent tuples form an affine translate of a linear subspace of the ambient fact space. For any fact set $S$ and fact index $i$, semantic determination of $i$ by $S$ is equivalent to membership of the coordinate functional for $i$ in the linear span of the coordinate functionals indexed by $S$. Consequently the fact indices carry a representable matroid: a fact set is independent exactly when its coordinate functionals are linearly independent, and the minimal determining fact sets are exactly the bases, all of common cardinality equal to the rank.
:::

::: proof
*Proof.* Let the affine realized-state family be $A=a_0+V$, where $a_0$ is a fixed origin and $V$ is a linear subspace of the ambient fact space. Fix a fact set $S$ and a fact index $i$.

By definition, fact $i$ is semantically determined by $S$ exactly when for all $x,x'\in A$, agreement on every coordinate in $S$ forces agreement on coordinate $i$. Writing $d=x-x'$, this is equivalent to the statement that every direction vector $d\in V$ whose coordinates in $S$ vanish also satisfies $d_i=0$.

Let $e_j$ denote the coordinate functional for fact $j$. The condition above says precisely that $$V\cap \bigcap_{j\in S}\ker(e_j)\subseteq \ker(e_i).$$ In finite-dimensional linear algebra, this is equivalent to $e_i$ lying in the span of the coordinate functionals $\{e_j:j\in S\}$. Thus semantic determination by $S$ is exactly span membership of the corresponding coordinate functional.

The coordinate functionals therefore realize a representable matroid on fact indices. In that matroid, independence is linear independence of the coordinate functionals, bases are the maximal independent spanning sets, and every basis has the common rank cardinality. Since span corresponds exactly to semantic determination, the minimal determining fact sets are precisely those bases. ◻
:::

The matroid and the confusability graph are complementary objects induced by the same realized-state model. Under the common specialization to finite affine families with coordinate-projection views, the matroid rank provides *computationally tractable* upper bounds on confusability and capacity. For each coordinate set $S$ the quantity $t(S)$ is the rank of the restricted coordinate map $\pi_S|_V$. The formalization proves the basic size bounds $t(S)\le |S|$ and $t(S)\le \dim V$, and shows that $t(S)=|S|$ on linearly independent coordinate families while $t(S)$ reaches the full ambient determining rank exactly on determining sets. Hence each $t(S)$ is computable by Gaussian elimination on an explicit presentation of $V$, and the bounds of Corollary [\[cor:matroid-capacity-bounds\]](#cor:matroid-capacity-bounds){reference-type="ref" reference="cor:matroid-capacity-bounds"} are computable in polynomial time. The utility is immediate: $t(S)$ provides an explicit upper bound on the Shannon capacity of the induced confusability graph, bypassing the hardness of the general capacity computation. Computing Shannon capacity for general graphs is hard (the independence number is NP-hard, and even the Lovász-$\vartheta$ bound requires semidefinite programming [@pekosz2011complexity; @grotschel1981ellipsoid]). In contrast, once the affine family is explicitly presented, the relevant rank quantities reduce to standard linear algebra.

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

**Remark.** The matroid rank $t(S)$ is the "informational dimension" captured by coordinates in $S$, directly analogous to the rank function in representable matroids studied in combinatorial optimization [@oxley2011matroid; @welsh1976matroid; @recski1989matroid] and coding theory [@cover2006elements]. Here $t(S)=r$ iff $S$ determines the entire state, and any admissible view containing a basis removes confusability entirely (the induced graph is edgeless). These finite-affine, coordinate-view statements show the matroid is directly applicable to the graph/capacity arc; extensions beyond this specialization are directions for future work.

#### Running example: Binary square.

The binary square $\{0,1\}^2$ from Section [\[sec:binary-square\]](#sec:binary-square){reference-type="ref" reference="sec:binary-square"} is an affine space over $\mathbb{F}_2$ with $r=2$. Each single-coordinate view has matroid rank $t(\{1\}) = t(\{2\}) = 1$, giving fiber size $2^{2-1} = 2$ and single-view capacity bound $\Theta(G_{\{i\}}) \le 1 \cdot \log 2 = \log 2$. The full confusability graph (the 4-cycle) achieves this bound, so the matroid rank bound is tight in this case. The matroid viewpoint shows that the 4-cycle structure arises precisely because each view captures only half the informational dimension.


# Unit-Rate Boundary Characterization {#sec:ssot}

The preceding sections characterize the zero-error capacity, strong-power scaling laws, and upper bounds for partial views when ambiguity survives. This section turns to the operational boundary: what structural properties are necessary to remain at the zero-incoherence threshold, and what is the information-theoretic penalty for leaving it? The resulting criterion is not merely a structural curiosity. When a deterministic partial-view architecture operates above unit rate, it does not merely experience operational friction; it deterministically inherits the non-clique failure topologies and bounded exact-success rates established in Sections [\[sec:graph-characterization\]](#sec:graph-characterization){reference-type="ref" reference="sec:graph-characterization"}--[\[sec:equality\]](#sec:equality){reference-type="ref" reference="sec:equality"}. Section [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"} proves that this boundary violation strictly forces an $\Omega(n)$ manual-update lower bound, and Section [\[sec:requirements\]](#sec:requirements){reference-type="ref" reference="sec:requirements"} establishes the exact realizability criteria for the zero-error regime.

## Derivation as the Single-Source Structure {#sec:ssot-def}

An encoding system has *unit independent rate* when $\text{DOF}(C,F)=1$. By Corollary [\[thm:coherence-capacity\]](#thm:coherence-capacity){reference-type="ref" reference="thm:coherence-capacity"}, this is exactly the regime in which every reachable state remains coherent, and any rate above $1$ admits reachable disagreement states. Thus unit rate is the unique exact-consistency boundary, and the later rate-complexity statements are its operational cost interpretation.

## Reading the Converse at Unit Rate {#sec:info-sketches}

When ambiguity classes are trivial, no auxiliary information is needed and unit rate suffices. When a nontrivial ambiguity class survives the observation transcript, Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"} shows that exact recovery requires a sufficiently large side-information alphabet. If that side information is unavailable but exact correctness is still demanded, additional independently accessible support is required. That is exactly what moves the system above unit rate and leads to the update-cost lower bound of Section [\[sec:rate-corollaries\]](#sec:rate-corollaries){reference-type="ref" reference="sec:rate-corollaries"}. Derivation is therefore the dependence structure that preserves visible views while removing independent rate.

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

If all encodings of $F$ except one are derived from that one, then Proposition [\[thm:derivation-excludes\]](#thm:derivation-excludes){reference-type="ref" reference="thm:derivation-excludes"} leaves exactly one independent source, hence $\text{DOF}(C,F)=1$ and exact consistency follows from Corollary [\[thm:coherence-capacity\]](#thm:coherence-capacity){reference-type="ref" reference="thm:coherence-capacity"}.

The same dependence structure can be realized in many host systems. Its role here is only to prepare the realizability criterion.


# Threshold and Rate-Complexity Corollaries {#sec:rate-corollaries}

The graph-capacity arc characterizes exact failure under partial views. This section gives a complementary bound: independent rate governs the minimum synchronization cost required to restore zero-error states after source modifications. When an architecture maintains $n$ independent representations without zero-delay state synchronization, the communication cost to restore joint consistency scales as $\Omega(n)$.

## Cost Model {#sec:cost-model}

::: definition
[]{#def:cost-model label="def:cost-model"} Let $\delta_F$ be a modification to fact $F$ in encoding system $C$. The *synchronization cost* $S(C,\delta_F)$ is the minimum number of independent state interventions required to restore a zero-error joint state after applying $\delta_F$. Deterministic dependent nodes contribute zero cost, as their state is compelled by the source update.
:::

## Upper Bound at Unit Independent Rate {#sec:upper-bound}

::: theorem
[]{#thm:upper-bound label="thm:upper-bound"} If $\text{DOF}(C,F)=1$, then $$S(C,\delta_F)=O(1).$$
:::

::: proof
*Proof.* When $\text{DOF}(C,F)=1$, there is exactly one independently writable source location for $F$. Updating that location is one manual edit; every other representation of $F$ is derived and is updated automatically. The number of manual edits therefore stays bounded by a constant independent of the number of visible encodings. ◻
:::

## Lower Bound Above the Threshold {#sec:lower-bound}

::: theorem
[]{#thm:lower-bound label="thm:lower-bound"} If fact $F$ is encoded at $n$ independent locations without zero-delay state synchronization, then $$S(C,\delta_F)=\Omega(n).$$ Equivalently, independent rate $n$ without deterministic coupling requires linear per-update communication cost.
:::

::: proof
*Proof.* Each independent location can retain stale state unless acted upon directly. After a source update, every one of the $n$ independent encodings can witness a distinct stale copy. Any protocol restoring zero-error consistency must therefore address each channel separately. No single intervention can synchronize two independently writable nodes, so the cost grows at least linearly in $n$. ◻
:::

::: lemma
[]{#lem:info-dof label="lem:info-dof"} If the available side-information mechanism exposes at most $2^L$ distinguishable tags while the latent ambiguity class has size $K>2^L$, then no zero-error resolver can identify the true state from those tags alone. Any architecture that still guarantees exact correctness must therefore rely on additional independently accessible discriminating support, moving the system above unit rate.
:::

::: proof
*Proof.* Theorem [\[thm:fano-converse\]](#thm:fano-converse){reference-type="ref" reference="thm:fano-converse"} rules out exact recovery from the available tag budget. If exact correctness is nevertheless required, the architecture must supplement the insufficient side-information channel with additional independently accessible information. In the present model, that means leaving the rate-$1$ regime and paying the synchronization cost of Theorem [\[thm:lower-bound\]](#thm:lower-bound){reference-type="ref" reference="thm:lower-bound"}. ◻
:::

## The Unbounded Gap {#sec:gap}

::: theorem
[]{#thm:unbounded-gap label="thm:unbounded-gap"} The ratio between the synchronization cost of an architecture with $n$ independent encodings and the synchronization cost of a rate-$1$ architecture diverges: $$\lim_{n\to\infty}\frac{S_{\mathrm{DOF}>1}(n)}{S_{\mathrm{DOF}=1}}=\infty.$$
:::

::: proof
*Proof.* By Theorem [\[thm:upper-bound\]](#thm:upper-bound){reference-type="ref" reference="thm:upper-bound"}, a rate-$1$ architecture has constant synchronization cost. By Theorem [\[thm:lower-bound\]](#thm:lower-bound){reference-type="ref" reference="thm:lower-bound"}, an architecture with $n$ independent encodings incurs linear cost in $n$. The ratio therefore grows without bound. ◻
:::

Once a fact is duplicated across many independently writable locations, the cost of preserving exact consistency scales with the number of independent copies, not with the visibility of the fact.


# Zero-Error Realizability {#sec:requirements}

The graph-capacity arc establishes the failure topology that partial views induce. This section answers the complementary question: which channel architectures realize the verifiable zero-error regime, and which do not? The resulting realizability criterion is architectural: it identifies two necessary and sufficient structural properties. It is grounded in the same finite-obstruction and confusability theory developed above. Applied to representative deterministic partial-view channels, including programming-language runtimes, database engines, and dependency managers, the criterion yields a zero-incoherence classification that demonstrates the majority of widely deployed infrastructure operates outside the verifiable zero-error regime (Appendix [\[sec:appendix-classification\]](#sec:appendix-classification){reference-type="ref" reference="sec:appendix-classification"}).

A host system realizes the rate-$1$ regime when one location is authoritative and every secondary representation is maintained as a deterministic view. Verifiable decoding requires two capabilities:

1.  **Zero-delay state synchronization**: updates to the authoritative source propagate to derived locations as part of the same atomic update event, with no reachable intermediate disagreement.

2.  **Structural side-information**: the decoder has access to topology metadata distinguishing authoritative sources from dependent nodes.

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

Structural facts are the canonical exact-consistency setting because they expose a clear creation moment. Examples include schema declarations, registry membership rules, dependency metadata, and interface-level constraints.

::: theorem
[]{#thm:timing-forces label="thm:timing-forces"} For structural facts, derivation must occur no later than the moment at which the relevant encoding locations become fixed.
:::

::: proof
*Proof.* Let $t_{\mathrm{fix}}$ denote the time at which the structural encoding becomes fixed. A derivation performed strictly before $t_{\mathrm{fix}}$ cannot depend on the completed source value; a derivation performed strictly after $t_{\mathrm{fix}}$ cannot alter the already fixed structural representation. Therefore structural derivation must occur at $t_{\mathrm{fix}}$ or as part of the same creation/update event. ◻
:::

## Requirement 1: Causal Update Propagation {#sec:causal-propagation}

::: definition
[]{#def:causal-propagation label="def:causal-propagation"} An encoding system has *zero-delay state synchronization* if every update to a source location is followed by updates to all locations derived from that source with no reachable intermediate state in which source and derived locations disagree. Equivalently, propagation is part of the same admissible update event for the purposes of reachable-state analysis.
:::

This is the host-system manifestation of deterministic dependence. Without it, a temporal gap appears between source modification and derived-view repair.

::: theorem
[]{#thm:causal-necessary label="thm:causal-necessary"} Achieving independent rate $1$ for structural facts requires zero-delay state synchronization.
:::

::: proof
*Proof.* By Theorem [\[thm:timing-forces\]](#thm:timing-forces){reference-type="ref" reference="thm:timing-forces"}, structural derivation must occur at the moment the source-side structural encoding is fixed. If zero-delay synchronization is absent, then some derived location remains stale until a later manual action. During that interval the source and derived locations disagree, so the architecture does not realize exact consistency. Hence a verifiable rate-$1$ realization for structural facts requires zero-delay synchronization. ◻
:::

## Requirement 2: Structural Side-Information {#sec:provenance-observability}

::: definition
[]{#def:provenance-observability label="def:provenance-observability"} An encoding system has *structural side-information* if it supports queries that reveal which locations encode a fact, which of those locations are authoritative, and which are derived.
:::

Structural side-information is the topology metadata needed to certify that the system is genuinely in the single-source regime rather than merely appearing coherent on a small sample of states. Without structural side-information, the decoder cannot distinguish a genuine zero-error configuration from a coincidentally matching stale state.

::: theorem
[]{#thm:provenance-necessary label="thm:provenance-necessary"} Verifying that independent rate $1$ holds requires structural side-information.
:::

::: proof
*Proof.* To verify independent rate $1$, one must enumerate the locations encoding the fact, identify the authoritative source, and confirm that every remaining location is derived from it. Without structural side-information, these checks cannot be completed from inside the host system. The architecture may be coherent on observed states, but the single-source claim is not verifiable. ◻
:::

## Independence of the Two Requirements {#sec:independence}

The two requirements are logically separate.

::: theorem
[]{#thm:independence label="thm:independence"}

1.  An encoding system can have zero-delay state synchronization without structural side-information.

2.  An encoding system can have structural side-information without zero-delay state synchronization.
:::

::: proof
*Proof.* For (1), consider a system that automatically refreshes all derived views after each source update but exposes no query interface for its derivation graph. Coherence may hold operationally, but the single-source claim is not inspectable. For (2), consider a system that records complete provenance metadata yet requires users to trigger propagation manually after each change. The derivation structure is visible, but stale views remain reachable. Hence neither property implies the other. ◻
:::

## The Realizability Theorem {#sec:realizability-theorem}

::: corollary
[]{#thm:ssot-iff label="thm:ssot-iff"} An observer can verifiably decode the single-source zero-error regime if and only if the system provides both zero-delay state synchronization and structural side-information.
:::

::: proof
*Proof.* Necessity follows from Theorems [\[thm:causal-necessary\]](#thm:causal-necessary){reference-type="ref" reference="thm:causal-necessary"} and [\[thm:provenance-necessary\]](#thm:provenance-necessary){reference-type="ref" reference="thm:provenance-necessary"}. For sufficiency, assume both properties hold. Zero-delay synchronization ensures that every derived location is updated as part of the same structural event as the source; structural side-information then certifies that all secondary encodings are in fact derived from that source. The architecture therefore realizes a verifiable single-source encoding, i.e., independent rate $1$, which is exactly verifiable structural integrity in the sense of Definition [\[def:verifiable-integrity\]](#def:verifiable-integrity){reference-type="ref" reference="def:verifiable-integrity"}. ◻
:::

Applied to representative hosts, this criterion separates systems in which propagation and provenance are both host-native from systems missing one or both capabilities.

Table [\[tab:host-checks-a-appendix\]](#tab:host-checks-a-appendix){reference-type="ref" reference="tab:host-checks-a-appendix"} reports the zero-incoherence classification for twelve representative channel families spanning three infrastructure domains: programming-language runtimes, database engines, and dependency managers. Of these, only two achieve verifiable rate $1$ under the criterion above. The remaining ten fail at least one of the two necessary conditions. Every entry is verified against the mechanized sufficiency conditions of Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"}. The classification demonstrates that the majority of deterministic partial-view channels underlying widely deployed infrastructure operate above the zero-incoherence threshold, where the $\Omega(n)$ synchronization cost of Theorem [\[thm:lower-bound\]](#thm:lower-bound){reference-type="ref" reference="thm:lower-bound"} applies and the nontrivial confusability structure of Section [\[sec:graph-characterization\]](#sec:graph-characterization){reference-type="ref" reference="sec:graph-characterization"} governs exact recovery.


# Related Work {#sec:related}

## Zero-Error and Side-Information Lineage {#sec:related-source-coding}

Shannon's zero-error framework and its graph-theoretic refinements by Körner and Lovász provide the closest conceptual starting point [@shannon1956zero; @korner1973graphs; @lovasz1979shannon]. Körner--Orlitsky's zero-error synthesis is also a direct reference point for this lineage [@korner2002zero]. Witsenhausen's zero-error side-information problem is directly relevant because it studies exact recovery under side information through graph coloring and label budgets [@witsenhausen1976zero]. Our one-shot coloring statements lie in that lineage. As in Witsenhausen's formulation, an observation law induces the relevant confusability graph. The difference is that here the observation law is generated by a deterministic multi-fact tuple-space architecture with admissible coordinate-subset views, so the model itself constrains what graphs can arise.

#### Key distinction from Witsenhausen.

Witsenhausen's setting starts from a side-information law and studies the coding problem for the graph it induces. Our setting starts from a deterministic partial-view architecture on latent tuples, with observations obtained by admissible coordinate-subset projections, and asks what graph structure that architecture can generate and what recovery laws follow from it. In the exact full-tuple-space coordinate-view model analyzed here, that graph class can now be characterized exactly: the realizable confusability relations are precisely those determined by upward-closed families of coordinate-agreement sets. The contribution is therefore not a new coloring theorem for arbitrary graphs, but an architectural realization theory connecting multi-fact tuple-space views, a fully characterized model-generated graph class, and exact zero-error recovery.

Slepian--Wolf coding and classical entropy converses supply the second line of influence [@slepian1973noiseless; @fano1961transmission; @witsenhausen1975conditional; @cover2006elements]. Coding-for-computing, functional compression, and graph-entropy/characteristic-graph viewpoints are also close neighbors because they study deterministic decoder targets and coding questions once an underlying graph or function structure has been specified [@orlitsky2001coding; @alon2002source; @doshi2010functional; @korner1973graphs; @simonyi2001graph]. Recent zero-error function-compression work also sits in this neighborhood [@guang2023zero; @liu2021leakage].

**Characteristic‑graph vs. coordinate‑projection models.** Characteristic‑graph formalisms handle arbitrary deterministic functions of the source, while our model restricts attention to coordinate‑projection observations on labeled tuple spaces. That restriction enables a reduction of confusability to agreement‑set membership, yielding proofs that (a) realizable confusability relations are exactly those determined by upward‑closed agreement families, (b) coordinatewise alphabet permutations act by graph automorphisms, and (c) the induced graph class is monotone and closed under block‑composition (strong‑product). These specific structural consequences do not follow from the general characteristic‑graph viewpoint.

In our setting the side information is deterministic rather than stochastic, but it plays the same operational role: exact recovery becomes impossible when the available observation/tag alphabet is too small to separate the remaining alternatives.

After a confusability graph is fixed, the colorability, strong-power, Shannon-capacity, and Lovász-$\vartheta$ machinery used later is classical. The novelty here is identifying transitivity of view-generated confusability as the model-side condition that produces the cluster-graph equality subclass, together with meet-witnessing and fiber coherence as structural sufficient conditions. On the upper-bound side, this follows the same classical-to-modern arc that includes Haemers-type bounds and asymptotic-spectrum viewpoints [@haemers1978upper; @zuiddam2019asymptotic; @deboer2024asymptotic]. The paper therefore does not claim a new theorem about arbitrary graphs, but a deterministic partial-view model that generates a fully characterized monotone agreement-set graph class on the full labeled tuple space. What it does not attempt is the broader unlabeled-isomorphism classification, or the corresponding classifications for restricted realized-state families and stochastic support-overlap models.

## Consistency-Constrained Storage and Interaction {#sec:related-distributed}

Consistency-constrained storage problems, including multi-version coding and related distributed-storage converses, show that correctness requirements impose unavoidable information costs [@rashmi2015multiversion]. The consistency literature establishes fundamental tradeoffs between consistency, availability, and partition tolerance [@brewer2000cap; @gilbert2002cap; @gray1978notes], while correctness models such as linearizability specify formal guarantees for concurrent operations [@herlihy1990linearizability]. Here the object under study is a modifiable encoding architecture, and the main question is the exact-consistency cost of multiple independently writable representations of one latent fact.

## Computational Realizations {#sec:related-meta}

Reflection, metadata systems, and maintenance mechanisms in host platforms provide examples of how the boundary-case realizability conditions can be instantiated in practice [@kiczales1991art; @smith1984reflection]. These examples are secondary to the paper's main zero-error and graph-capacity contribution.

## Formal Verification {#sec:related-formal}

The Lean 4 formalization places the work in the tradition of mechanized mathematics and verified theory development [@demoura2021lean4; @mathlib2020; @leroy2009compcert].


# Conclusion {#sec:conclusion}

The paper's main contribution is a zero-error theory for deterministic partial views. Admissible view families induce confusability graphs on latent states; those graphs record exactly which latent alternatives the architecture cannot separate, and they govern the exact recovery law. In the exact full-tuple-space coordinate-view model, the realizable confusability relations are characterized exactly by upward-closed agreement-set families. In the multi-fact extension, exact recovery becomes graph colorability, exactness on a success set becomes induced-subgraph colorability, and the exact finite weighted-success value is determined by the largest colorable induced subgraph.

Repeated composition preserves that failure structure rather than erasing it: the block-rate sequence is supermultiplicative, its normalized rates converge by Fekete's lemma to a real asymptotic Shannon capacity equal to the supremum of the finite block rates, and the same capacity is bounded above by complement-chromatic and fixed Lovász-$\vartheta$ upper objects. The one-shot upper invariant is matched with standard orthonormal-representation, primal-PSD, and dual-theta forms.

The upper theory is sharp on a structurally characterized subclass. For the original clique-fiber subclass, asymptotic Shannon capacity, the fixed Lovász-$\vartheta$ upper, and the logarithm of the number of fibers coincide. The same equality exports back to the original view-family model through a broader structural route: transitivity of confusability yields the component-cluster collapse, meet-witnessing is a sufficient condition for that transitivity, and fiber coherence is a stronger sufficient condition under which the connected components are exactly the realized transcript fibers.

The same latent-state framework also yields a fact-side affine specialization. The observation-side question asks which latent states are distinguishable from partial views; the fact-side question asks which coordinates determine others across the realized-state family. When the valid states form an affine family, the latter question becomes span membership of coordinate functionals, so the fact indices carry a representable matroid and the minimal determining fact sets are exactly the bases. In this affine regime, the same model carries a graph invariant on states and a matroid invariant on coordinates, and the latter supplies tractable upper bounds for the former. More sharply, the relevant affine rank quantity is exactly the finrank of the image of the restricted coordinate projection, which closes the representation-level link between the matroid language and explicit linear algebra.

The deterministic finite converse remains the foundation of this theory. Once the observation transcript leaves a nontrivial ambiguity class, the same obstruction can be stated as confusability, counting, conditional-entropy, decoder-output, and finite-gap constraints, together with deterministic data processing and a budgeted finite-error extension. The paper does not claim a new theorem about arbitrary confusability graphs; rather, it identifies a deterministic partial-view model that induces a fully characterized nontrivial graph class and carries the classical zero-error graph-capacity machinery with it.

As boundary consequences, unit rate is the unique zero-incoherence regime, any higher independent rate makes ambiguity reachable, and the same obstruction yields an $O(1)$ versus $\Omega(n)$ manual-update gap together with a realizability criterion based on causal propagation and provenance observability. Appendix [\[sec:appendix-classification\]](#sec:appendix-classification){reference-type="ref" reference="sec:appendix-classification"} records representative host-level readings of that boundary-case criterion together with companion case-study details.

**Computational representation.** The graph-theoretic formulation is structurally exponential in the number of facts: for $d$ facts over an alphabet of size $q$, the latent state space has cardinality $q^d$, so any explicit confusability graph materialization necessarily starts from an exponentially large vertex set. A naive explicit construction ranges over at most $q^{2d}$ ordered state pairs, formalized in the artifact through the corresponding product-cardinality and pair-count bounds. At the same time, the graph need not be materialized: adjacency is already packaged by an implicit oracle on state pairs, formalized as a Boolean confusability test equivalent to the confusability relation itself, and also by an equivalent agreement-set oracle whose direct scan cost is bounded linearly in the total view size. Under the explicit coordinate-view representation used throughout the paper, that one-shot oracle is therefore polynomial in its direct input size, computable by scanning the coordinates and the admissible view family in $O(d+\sum_{\ell}|V_\ell|)$ time, hence $O(Ld)$. In the affine specialization, the upper bounds are likewise polynomial-time computable once the direction space is presented explicitly by a basis or generator matrix, because the formalized rank quantity is exactly the finrank of the image of the restricted coordinate projection, so each bound reduces to Gaussian elimination on that explicit map. The exponential barrier therefore lies in full graph materialization and global capacity computation, not in the local adjacency test or in the affine upper-bound surrogate.

**Limitations and future work.** The present paper develops only the finite deterministic architectural arc. For the exact tuple-space coordinate-view model, the full labeled-tuple-space graph class is now characterized exactly by monotone agreement-set families. Several directions remain open. First, the classification up to unlabeled graph isomorphism is incomplete: two distinct upward-closed families may yield isomorphic confusability graphs, and a complete invariant-theoretic characterization of the model-generated graph class is not yet available. Second, the restricted-state case, where the realized latent tuples form a proper subset of the full product space, admits a richer confusability structure whose graph-theoretic properties are not fully resolved. Third, the stochastic support-overlap extension, in which each location may emit any observation from a finite support set and two states are confusable when some location has overlapping support, is formalized in the Lean artifact but not developed textually; its graph-theoretic consequences and the sharpness of the Lovász-$\vartheta$ upper bound on the resulting broader graph class remain open. Finally, the sharpness of the capacity upper bounds on the deterministic model's non-clique subclass is not established: the binary square achieves capacity-$\vartheta$ equality by coincidence rather than by the transitivity mechanism of Section [\[sec:equality\]](#sec:equality){reference-type="ref" reference="sec:equality"}, and the general conditions under which the upper theory is tight for the model-generated class are not known.

## Artifacts {#sec:data-availability}

The Lean 4 formalization is included as supplementary material. Appendix [\[sec:appendix-classification\]](#sec:appendix-classification){reference-type="ref" reference="sec:appendix-classification"} records representative realizability readings and companion case-study material.

## Acknowledgment: AI-use Disclosure {#acknowledgment-ai-use-disclosure .unnumbered}

Generative AI tools (including Codex, Claude Code, Augment, Kilo, and OpenCode) were used throughout this manuscript, across all sections (Abstract, Introduction, theoretical development, proof sketches, applications, conclusion, and appendix) and across all stages from initial drafting to final revision. The tools were used for boilerplate generation, prose and notation refinement, LaTeX/structure cleanup, translation of informal proof ideas into candidate formal artifacts (Lean/LaTeX), and repeated adversarial reviewer-style critique passes to identify blind spots and clarity gaps.

The author retained full intellectual and editorial control, including problem selection, theorem statements, assumptions, novelty framing, acceptance criteria, and final inclusion/exclusion decisions. No technical claim was accepted solely from AI output. Formal claims reported as machine-verified were admitted only after Lean verification (no `sorry` in cited modules) and direct author review; Lean was used as an integrity gate for responsible AI-assisted research. The author is solely responsible for all statements, citations, and conclusions.


# System Classification {#sec:appendix-classification}

This appendix applies the realizability criterion of Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"} to representative channel witnesses. Table [\[tab:host-checks-a-appendix\]](#tab:host-checks-a-appendix){reference-type="ref" reference="tab:host-checks-a-appendix"} classifies systems by their structural side-information capabilities and zero-delay state synchronization properties. Pattern A: missing causal propagation. Pattern B: missing provenance observability. Pattern C: both conditions present (verifiable rate $1$).

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **Channel**       **Instance**                                                                                    **Class**   **Prop.**   **Prov.**   **Rate 1**   **Mechanism**
  ----------------- ----------------------------------------------------------------------------------------------- ----------- ----------- ----------- ------------ ----------------------------------------------------------------------------------------------------------------------------
  Prog. languages   Python [@pep487; @pythondatamodel2025]                                                          C           yes         yes         yes          Class-definition hooks and runtime hierarchy introspection are host-native.

  Databases         Engine-maintained materialized view [@postgresMaterializedViews2025; @postgresPgMatviews2025]   C           yes         yes         yes          Engine maintains the derived relation and exposes catalog metadata.

  Prog. languages   Rust [@rustref2024]                                                                             B           partial     no          no           Compile-time macro generation exists, but the compiled runtime artifact does not retain derivation provenance.

  Prog. languages   TypeScript / JavaScript [@typescriptDecorators2025; @javascriptDecorators2025]                  B           partial     no          no           Decorators can attach metadata, but type erasure and missing subclass provenance block exact verification.

  Prog. languages   Java [@gosling2021java]                                                                         A/B         no          no          no           Under the runtime-only host interpretation used here, annotations are external to the JVM runtime.

  Prog. languages   Go [@gospec2024]                                                                                A/B         no          no          no           Reflection exists, but there are no definition-time hooks or host-native implementer registries.

  Databases         External ETL copy / duplicate summary table                                                     A/B         no          no          no           Synchronization is externalized, and provenance is no longer intrinsic to the host database.

  Deps.             npm [@npmLock2026; @npmExplain2026; @npmLs2026]                                                 A           no          yes         no           Lockfile (auxiliary tag) and query commands expose provenance, but manifest edits require explicit out-of-band operations.

  Deps.             Cargo [@cargoLockGuide2026; @cargoTree2026; @cargoMetadata2026]                                 A           no          yes         no           Resolved-dependency provenance is exposed, but the auxiliary tag (`Cargo.lock`) remains a distinct derived artifact.

  Deps.             Poetry [@poetryCli2026]                                                                         A           no          yes         no           The resolved graph is queryable, but lock regeneration is an explicit step rather than part of the source edit itself.

  Deps.             pnpm [@pnpmInstall2026; @pnpmWhy2026; @pnpmList2026]                                            A           no          yes         no           Provenance is queryable, but the lockfile remains separately maintained rather than automatically updated.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**Table A.1.** Zero-incoherence classification of representative channel witnesses in the deterministic partial-view model. The same theorem-level coordinates apply uniformly across programming-language runtimes, databases, and dependency managers. Each entry is verified against the mechanized sufficiency conditions of Corollary [\[thm:ssot-iff\]](#thm:ssot-iff){reference-type="ref" reference="thm:ssot-iff"}. []{#tab:host-checks-a-appendix label="tab:host-checks-a-appendix"}




---

## Machine-Checked Proofs

All theorems are formalized in Lean 4:
- Location: `docs/papers/paper2_ssot/proofs/`
- Lines: 25924
- Theorems: 1252
- `sorry` placeholders: 0
