## Paper 4 intro/contributions reframing plan

Date: 2026-03-13
Status: planning only, not yet applied

### Goal

Reframe `paper4` so the optimizer quotient `Q` is visibly the paper's organizing object, while keeping every headline claim faithful to the current theorem package and Lean artifact.

The revised intro should read like this:

> The optimizer quotient is the canonical exact decision-preserving abstraction.
> The paper then classifies the cost of certifying that structure across static, stochastic, and sequential regimes.

This is stronger and cleaner than the current framing, but still honest. It uses the universal-property backbone already present in `paper4` and `paper4b` without overclaiming a fully general stochastic terminality theorem.

### What changes, in one sentence

Current framing: `paper4` reads primarily like a regime-sensitive complexity paper with a useful quotient object.

Target framing: `paper4` should read like a paper about the canonical object `Q`, with the complexity matrix explaining when exact certification of `Q`-relevant structure is easy, split, or hard.

### Guardrails

These points should stay explicit throughout the rewrite.

- Do say `Q` is the canonical/coarsest exact decision-preserving abstraction in `Set`.
- Do use the universal-property language already supported by `DecisionProblem.quotient_is_coarsest` and `DecisionProblem.quotient_has_unique_factorization`.
- Do say the stochastic preservation story bridges back to the same quotient object, especially under full support.
- Do not say the stochastic case already has a fully general terminal-object theorem.
- Do not collapse the stochastic row to `PP`. PP-hardness in `paper4` is for decisiveness and its variants, not for preservation.
- Do not replace the paper's actual six tractable cases with a different external list.
- Do keep the mechanization claim calibrated: finite combinatorial cores, deciders, witness/checking schemas, reduction correctness. Not full oracle-machine formalization.

### Source files to revise

- `docs/papers/paper4_decision_quotient/latex/content/01_introduction.tex`
- `docs/papers/paper4_decision_quotient/latex/content/abstract.tex`
- `docs/papers/paper4_decision_quotient/markdown/paper4.md`

The real source of record for the submission is the LaTeX in `latex/content`, but `paper4.md` should be kept in sync for review and future drafting.

## Integration of the `paper4b` proof spine

Yes, we should explicitly integrate the `paper4b` theorem spine into `paper4`'s framing.

The right way to do that is not to import `paper4b` as if it were a separate result set. The better move is to foreground the theorem package that is already present in the `paper4` artifact and use `paper4b`'s rhetoric to explain what those theorems mean.

### What is already literally proved

Static canonicality of the decision quotient is already theorem-level in the Lean artifact:

- `DecisionProblem.quotient_is_coarsest`
- `DecisionProblem.quotient_has_unique_factorization`
- `DecisionProblem.quotient_represents_opt_equiv`
- `DecisionProblem.quotientEquivOptRange`

This means the paper can safely say that the optimizer quotient is canonically characterized as the coarsest exact decision-preserving abstraction in `Set`, not just described that way informally.

### Lean handles to name explicitly in the plan

The currently wired paper-level handles that support this framing are:

- `QT1` = `DecisionProblem.quotient_is_coarsest`
- `QT7` = `DecisionProblem.quotient_has_unique_factorization`

Useful additional theorem aliases present in `HandleAliases.lean` and available to cite if we want a richer theorem-facing discussion are:

- `QT3` = `DecisionProblem.quotient_represents_opt_equiv`
- `QT5` = `DecisionProblem.quotientEquivOptRange`

Relevant stochastic bridge handles already present in the paper's handle map are:

- `DC96` = `StochasticSequential.static_sufficiency_iff_stochastic_preservation_of_full_support`
- `DC97` = `StochasticSequential.stochasticDecisionEquiv_iff_decisionEquiv_of_preservation`
- `DC98` = `StochasticSequential.stochasticDecisionEquiv_iff_decisionEquiv_of_full_support`
- `DC99` = `StochasticSequential.stochasticEquivSetoid_eq_decisionSetoid_of_full_support`

### How to integrate those proofs rhetorically

The revised introduction and contributions should say, in substance:

- `QT1` and `QT7` prove that the optimizer quotient is canonical in the precise universal-property sense.
- `QT3` and `QT5` explain what that quotient is concretely: it identifies optimizer-equivalent states and is canonically equivalent to the image/range of `Opt`.
- `DC96` through `DC99` show how the stochastic preservation theory reconnects to the same quotient object under preservation, and under full support to the original decision quotient itself.

That is the correct mathematical integration of `paper4b` into `paper4`.

### Important scope note

This integration supports the sentence

> The decision quotient has literally proved canonical status in the artifact.

but only in the static `Set`-level exact-abstraction sense, plus the existing stochastic bridge theorems.

It does **not** yet support the stronger sentence

> We have proved a fully general stochastic universal property for arbitrary stochastic abstractions.

The plan should preserve that distinction.

## Edit plan

### 1. Rebuild the opening around `Q`

#### Current problem

The current opening in `01_introduction.tex` starts from the question "which coordinates can be hidden" and only later identifies the optimizer quotient as the structural centerpiece.

That makes the paper read like a complexity survey first and a structural paper second.

#### Planned change

Open by introducing the optimizer quotient immediately after the base sufficiency definition, and state the central thesis before the three-regime tour:

- every exact decision-preserving abstraction must respect the distinctions encoded by `Q`
- `Q` is the coarsest such abstraction
- the paper studies how hard it is to certify that exact structure in static, stochastic, and sequential settings

#### Draft replacement for the opening block

Target file: `docs/papers/paper4_decision_quotient/latex/content/01_introduction.tex`

Replace the current first two paragraphs (currently beginning `Which coordinates of a decision problem...` and `In plain terms:`) with the following draft:

```tex
Which coordinates of a decision problem can be hidden without changing the optimal action, and what is the canonical exact abstraction that records all and only the distinctions that matter for that decision? We study this question for a decision problem $\mathcal{D}=(A,S,U)$ with $S = X_1 \times \cdots \times X_n$. A coordinate set $I$ is sufficient when agreement on $I$ forces agreement on the optimal-action set:
\[
s_I = s'_I \implies \Opt(s) = \Opt(s').
\]
The associated optimizer quotient identifies states exactly when they induce the same optimal-action set. In \textbf{Set}, this quotient is the coarsest abstraction through which $\Opt$ factors, equivalently the canonical lossless abstraction of the decision boundary.

This paper's main claim is that once that canonical object is fixed, exact relevance certification becomes a regime-sensitive complexity problem. The structural object is the same throughout, but the cost of certifying it changes with the ambient model. In the static regime, exact certification reduces to counterexample exclusion and collapses to relevance containment. In the stochastic regime, conditioning splits the exact question into preservation and decisiveness. In the sequential regime, temporal contingency lifts the same certification task to planning-style reasoning. The paper therefore reads in two layers: the optimizer quotient gives the canonical notion of exact decision-preserving compression, and the complexity matrix classifies when that structure can be certified exactly.
```

#### Reasoning

- This keeps the familiar opening question.
- It introduces `Q` before the complexity landscape.
- It turns the regime matrix into the consequence of certifying one object across richer settings.
- It reuses claims already supported by `paper4` and `paper4b`.

### 2. Tighten the interpretive guide so it supports the new thesis

#### Current problem

The current interpretive guide is vivid, but it appears before the paper's formal thesis fully lands and slightly diffuses the opening.

#### Planned change

Keep the guide, but shorten it and make `Q` the first interpretive term. Reduce the density of coined labels so the section reads as support rather than a second opening.

#### Draft replacement

Replace the current `\paragraph{Informal interpretive guide.}` paragraph with:

```tex
\paragraph{Informal interpretive guide.}
The formal objects studied here admit a compact intuitive reading. The optimizer quotient isolates the decision signal: it collapses exactly those state distinctions that do not change the optimal-action set. Relevant coordinates are points of leverage: changing them can change the decision. In the static regime, the resulting simplification is forced by structure rather than discovered by search. In the later consequence sections, the simplicity tax names the residual burden created when a system uses heuristic simplification in place of exact certification.
```

#### Reasoning

- Makes `Q` the first interpretive object.
- Preserves the nice language around leverage and simplicity tax.
- Removes some metaphor load from the top of the paper.

### 3. Rewrite the static-structure paragraph to make `Q` the theorem-level anchor

#### Current problem

The current paragraph beginning `This is a certification problem rather than a forward-evaluation problem` contains the right facts, but its center of gravity is still minimum-sufficiency collapse rather than the canonical abstraction.

#### Planned change

Keep the quantifier point, but make the order:

1. `Q` is canonical.
2. Static sufficiency identifies exactly which coordinates expose `Q`.
3. Minimum sufficiency collapses because of that structure.

#### Draft replacement

```tex
This is a certification problem rather than a forward-evaluation problem. Evaluating $U(a,s)$ or $\Opt(s)$ on a fully specified state is one task; proving that a family of coordinates is irrelevant requires ruling out every counterexample pair of states. That universal structure is exactly what makes the optimizer quotient central rather than cosmetic: it records the maximal lossless collapse of the state space, and every exact decision-preserving abstraction must refine it. In the static regime, this structural picture sharpens further. A coordinate set is sufficient exactly when it exposes all relevant coordinates, so the minimum sufficient set is not discovered by a separate alternating search. It is simply the relevant-coordinate set itself. By contrast, \textsc{Anchor-Sufficiency} retains a genuine existential choice because the anchor assignment must still satisfy a universal fiberwise condition.
```

#### Reasoning

- Keeps the quantifier story.
- Makes the universal property feel necessary, not decorative.
- Connects static collapse to the quotient story directly.

### 4. Reframe the stochastic paragraph as a split certification story around the same object

#### Current problem

The current stochastic paragraph is already careful, but it does not say strongly enough that preservation is the route back to the same quotient object.

#### Planned change

Keep the preservation/decisiveness split, but foreground that preservation is the stochastic route back to exact quotient preservation.

#### Draft replacement

```tex
\paragraph{Why probability changes the exact question.}
In stochastic decision problems, exact analysis separates into a preservation question and a decisiveness question. Preservation asks whether coarse information reproduces the full-information optimizer, so it is the direct stochastic analogue of exact quotient preservation. Decisiveness asks whether each observable fiber already determines a unique Bayes-optimal action, so it is a stronger decision-completeness demand. Formally, these are \emph{stochastic sufficiency} and \emph{stochastic decisiveness}. They are distinct, and the paper treats them separately on purpose. Preservation gives the bridge back to static sufficiency and, under full support, back to the same quotient structure as the static regime. Decisiveness yields the paper's strongest succinct-encoding hardness classification. The stochastic row is therefore not one theorem about one predicate, but a split exact theory with two different complexity behaviors.
```

#### Reasoning

- Preserves the careful scope of the existing stochastic claims.
- Explicitly prevents readers from flattening the row into a single PP story.
- Links preservation back to `Q` more clearly.

### 5. Reframe the regime-matrix paragraph as a cost-of-certifying-`Q` paragraph

#### Current problem

The current paragraph beginning `This systematic development reveals a regime-sensitive complexity landscape` is good, but can more explicitly state that the matrix is about certifying the same structural object.

#### Draft replacement

```tex
This systematic development reveals a regime-sensitive complexity landscape because the same structural target is being certified under richer models. Static certification is governed by counterexample exclusion. In the stochastic regime, preservation retains the quotient-preservation target while decisiveness introduces conditional-comparison hardness. In the sequential regime, omitted information can matter only through future contingencies, so exact certification becomes planning-hard. Read this way, the complexity matrix is not a list of unrelated classifications. It is the cost profile of certifying the exact decision-relevant distinctions encoded by the optimizer quotient.
```

#### Reasoning

- Makes the matrix feel conceptually unified.
- Avoids sounding like three adjacent papers stitched together.

## Contributions rewrite

### Current problem

The current contributions list starts with the static structural theorem, which is true, but does not foreground the optimizer quotient as the paper's organizing object.

### Planned change

Reorder the contributions so the first contribution is the canonical structure, followed by the regime-sensitive certification landscape, then tractability islands, then methodological consequences, then mechanization.

### Draft replacement for the contributions preamble

Replace the current preamble paragraph under `\subsection{Contributions}` with:

```tex
The paper is best read as a theory of exact decision-relevant information organized around one canonical object: the optimizer quotient. The quotient gives the coarsest exact abstraction that preserves optimal-action distinctions. The paper then classifies how the cost of certifying that structure changes across static, stochastic, and sequential regimes, identifies tractable structural subcases, and supports the finite combinatorial core with a Lean artifact.
```

Add a short sentence immediately after this preamble if we want the theorem support visible up front:

```tex
This canonical status is theorem-level rather than merely descriptive: the artifact proves the quotient's coarseness and unique factorization properties \leanmeta{\LH{QT1}, \LH{QT7}.}
```

### Draft replacement for the numbered list

Replace the current five contribution items with the following five items.

```tex
\begin{enumerate}
\item \textbf{Optimizer quotient as canonical structure.} For a finite decision problem, the optimizer quotient identifies states exactly when they induce the same optimal-action set. In \textbf{Set}, it is the coarsest exact decision-preserving abstraction, equivalently the coimage/image factorization of the optimizer map. This gives the paper a common structural object through which the later static, stochastic, and sequential results can be compared.

\item \textbf{Regime-sensitive certification landscape.} The paper studies how hard it is to certify exact preservation of decision-relevant structure across three regimes. In the static regime, sufficiency collapses to relevance containment, so \textsc{Minimum-Sufficient-Set} is coNP-complete rather than the expected $\Sigma_2^P$ search problem, while \textsc{Anchor-Sufficiency} remains $\Sigma_2^P$-complete. In the stochastic regime, exact analysis splits into preservation and decisiveness. Preservation is polynomial-time under explicit-state encoding, bridges back to static sufficiency and the optimizer quotient, and under full support yields inherited coNP and $\Sigma_2^P$ classifications for the minimum and anchor variants. Decisiveness is polynomial-time in explicit state and PP-hard under succinct encoding, with the anchor and minimum variants PP-hard and in $\textsf{NP}^{\textsf{PP}}$ at paper level. In the sequential regime, sufficiency, minimum, and anchor queries are PSPACE-complete.

\item \textbf{Tractability islands and encoding-sensitive contrast.} The paper proves an explicit-state versus succinct-encoding contrast and isolates six structural subcases in which exact certification remains polynomial-time: bounded actions, separable utility, low tensor rank, tree structure, bounded treewidth, and coordinate symmetry. These are not ad hoc exceptions. They are regimes in which the decision boundary admits enough structural alignment to make exact certification feasible.

\item \textbf{Consequences for exact certifiers.} In the hard exact regime, certifiers cannot in general be simultaneously sound, complete, and polynomial-budgeted. The same theory yields witness-checking lower bounds, approximation obstructions, and the practical consequence called the simplicity tax: when exact relevance is not certified, the unresolved burden is not removed but shifted elsewhere in the system.

\item \textbf{Mechanized finite-decision core.} A Lean 4 artifact mechanically checks the main finite deciders, search procedures, reduction-correctness lemmas, witness/checking schemas, and bridge results \leanmeta{\LHrng{DC}{91}{99}, \LHrng{OU}{1}{2}, \LHrng{OU}{8}{12}, \LHrng{EH}{1}{10}.} The artifact mechanizes the finite combinatorial core underlying the paper's stochastic upper bounds and reduction infrastructure. Full oracle-machine formalization remains outside the current state of formalized complexity theory, so the paper proves those oracle-class memberships in the text and uses the artifact as an independent check of the finite core.
\end{enumerate}
```

### Reasoning

- Contribution 1 puts `Q` first.
- Contribution 2 presents the complexity matrix as the cost of certifying the same structural object.
- Contribution 3 keeps the actual six tractable cases already proved in `paper4`.
- Contribution 5 preserves the current calibration of the Lean claim.

## Specific theorem-citation plan

To make the `paper4b` integration visible instead of implicit, use the following citation pattern.

### Introduction

If we add Lean handles in the introduction, keep them sparse. The best place is the sentence introducing canonical status of the quotient.

Recommended insertion:

```tex
The associated optimizer quotient identifies states exactly when they induce the same optimal-action set. In \textbf{Set}, this quotient is the coarsest abstraction through which $\Opt$ factors, equivalently the canonical lossless abstraction of the decision boundary. \leanmeta{\LH{QT1}, \LH{QT7}.}
```

Optional enriched version if we also want the concrete identification with the optimizer image/range visible:

```tex
... the coarsest abstraction through which $\Opt$ factors, equivalently the canonical lossless abstraction of the decision boundary. It is canonically equivalent to the image/range of $\Opt$. \leanmeta{\LH{QT1}, \LH{QT7}.}
```

If `QT5` is not currently exposed in the paper's handle map, do not cite it in the first pass. Add the theorem name to the prose, and only wire the handle later if needed.

### Formal setup / foundations

The canonical theorem already appears in `02_formal_setup.tex` with

- `\leanmeta{\LH{QT1}, \LH{QT7}.}` at the universal-property theorem
- `\leanmeta{\LH{QT7}, \LH{AB2}.}` at the canonical-status remark

This should stay. The intro rewrite should make those theorem references feel like the paper's backbone rather than later support.

### Stochastic section

If we want one sentence in the intro or contributions that explicitly says the stochastic preservation theory returns to the same quotient object, the clean support bundle is:

```tex
Under preservation, and under full support in particular, the stochastic fiber quotient collapses back to the original decision quotient. \leanmeta{\LH{DC97}, \LH{DC99}.}
```

Use `DC96` or `DC98` only if we need to mention the full-support bridge itself rather than just the quotient consequence.

## Abstract alignment

This is optional for the first pass, but likely worth doing so the paper's first page and introduction agree.

### Planned abstract adjustment

Keep the current abstract's factual content, but revise the first two sentences so the optimizer quotient is presented as the paper's structural centerpiece rather than as a side object appended to the complexity story.

### Draft abstract opening

Target file: `docs/papers/paper4_decision_quotient/latex/content/abstract.tex`

Suggested replacement for the first paragraph:

```tex
Which coordinates of a decision problem can you hide without changing the decision, and what is the canonical exact abstraction that records all and only the distinctions that matter for that choice? We study this as an exact relevance-certification problem. For a decision problem $\mathcal{D}=(A,S,U)$, the optimizer quotient identifies states exactly when they induce the same optimal-action set, yielding the coarsest abstraction that preserves optimal-action distinctions. The paper then asks how hard it is to certify that exact structure across static, stochastic, and sequential regimes.
```

### Reasoning

- Makes the abstract match the planned intro thesis.
- Keeps the complexity matrix as the paper's payoff, not its only identity.

## Concrete cleanup items while editing

- Remove the duplicated closing paragraph in `01_introduction.tex` after the contributions list.
- Keep `paper4.md` synchronized with the LaTeX edits after the wording is approved.
- Preserve the existing careful statements about support-sensitive stochastic preservation beyond full support.
- Do not import `paper4b` language wholesale if it blurs the stronger complexity-specific contributions of `paper4`.

## Review checklist for the next pass

- Does the first page make `Q` feel like the paper's central mathematical object?
- Does the intro still clearly explain why the stochastic row splits into preservation and decisiveness?
- Are we still accurately distinguishing explicit-state tractability from succinct hardness?
- Are the six tractable cases exactly the ones already proved in `paper4`?
- Does the mechanization paragraph remain calibrated and not overclaim oracle formalization?
- Does the revised text sound like one theory with one organizing object, rather than several adjacent theorem packages?

## Recommendation

Apply this in two steps.

1. Revise `01_introduction.tex` and `abstract.tex` only, then review the rhetorical balance.
2. Once that reads correctly, sync the same wording into `paper4.md` and only then do broader local consistency edits.
