# Paper 1 JSAIT Follow-Up Proof Outline

This note collects follow-up results that look worth attempting given the current paper-1 theorem stack and Lean infrastructure. The goal is not to commit to all of them, but to prioritize claims that are (i) structurally close to what is already formalized, and (ii) materially strengthen the connection to semantics-aware compression, learned representations, and privacy/security.

## Current infrastructure already in place

- Finite explicit observer model and profile map `\pi`
- Information barrier / observational equivalence theorems
- Nominal-tag sufficiency and ambiguity converse in terms of `A_\pi`
- Exact finite-block confusability laws
- Matroid structure and distinguishing dimension `d`
- Open-world extension instability and Rice-style certification limits

These ingredients make three kinds of follow-up theorems especially plausible:

1. monotonicity or comparison theorems on existing invariants,
2. representation/embedding reinterpretations of the same invariants,
3. disclosure-style statements that reuse `L`, `W`, and `d` without requiring a full new privacy formalism.

## Highest-value, lowest-risk candidates

### 1. Embedding-collapse theorem

Status: **mechanized in the finite explicit setting**.

- `\LH{GPH27}`: one-symbol zero-error feasibility is equivalent to injectivity.
- `\LH{GPH26}`: in the nonempty case, injectivity is equivalent to worst-case fiber size one.

Claim shape:

> If a learned representation or embedding map is used as the observable profile map `\pi`, then zero-error class identification is feasible without tags iff the representation is class-injective. If not, the exact additional tag budget is at least `\log_2 A_\pi`.

Why it fits:

- This is mostly a reinterpretation theorem, not a new mathematical object.
- It connects directly to the special issue by tying the ambiguity converse to learned representations.
- It can likely be stated with almost no new proof burden: it is an instantiation corollary of the existing converse plus feasibility theorem.

Lean effort:

- Low.
- Mostly packaging: define an embedding-instantiated observation map and invoke existing theorems.

Submission value:

- High.
- This is the cleanest bridge from the current theory to neural classification and semantics-aware compression.

### 2. Tag-vs-query disclosure comparison

Status: **partially available already**.

- Existing formal ingredients already prove the comparison needed for a disclosure-style statement: nominal tags give constant witness cost, while tag-free zero-error identification requires at least `d` primitive observations.
- In addition, a finite explicit adversarial-family disclosure theorem is now mechanized:
  - `\LH{PRIV1}`: nominal identity disclosure is constant,
  - `\LH{PRIV2}`: tag-free identity resolution needs at least `n-1` primitive disclosures in the adversarial family,
  - `\LH{PRIV3}`: formal disclosure separation theorem combining the two.
- A genuinely privacy-theoretic statement still requires a formal leakage model rather than witness cost or disclosure count alone.

Claim shape:

> Explicit tags reveal class identity in one access, whereas any tag-free zero-error identification strategy requires at least `d` primitive observations.

Why it fits:

- The ingredients are already present: nominal witness cost is `O(1)` and tag-free lower bound is `\Omega(d)`.
- It gives a concrete privacy/security interpretation without pretending to prove a full differential-privacy or PIR result.

Lean effort:

- Very low.
- Almost entirely corollary packaging from existing witness-cost theorems.

Submission value:

- Medium to high.
- Good for a short extension paragraph or proposition, especially if phrased as a disclosure-pattern comparison rather than a full privacy theorem.

### 3. Monotonicity under profile refinement

Status: **mechanized in a finite explicit form**.

- `\LH{GPH28}`: if one observation channel factors through another, the finer channel cannot have larger collision multiplicity.
- This captures the basic monotonicity needed for the “adding observations reduces ambiguity” interpretation.

Claim shape:

> If the observable family is refined so that the new profile map separates at least as many classes as the old one, then collision multiplicity cannot increase and the zero-error tag lower bound cannot worsen.

Why it fits:

- This is natural mathematically and operationally.
- It formalizes the intuition that adding observations can only reduce ambiguity.

Lean effort:

- Moderate.
- Needs a formal notion of profile refinement / factorization between observation maps.

Submission value:

- Medium.
- Strengthens the observer-model story and could support engineering interpretation.

## Medium-value candidates

### 4. Coarsening monotonicity for distinguishing dimension

Status: **not attempted beyond initial inspection**.

- This remains plausible, but it is not low-hanging fruit.
- The obstruction is structural rather than computational: `d` is defined through minimal distinguishing sets and matroid rank, so a clean monotonicity theorem requires a careful formal relation between two query families and their induced closures.

Claim shape:

> Under suitable coarsening of the observable family, the distinguishing dimension does not decrease.

Why it fits:

- This would connect `d` more tightly to observation quality.
- It complements the `A_\pi` monotonicity story.

Lean effort:

- Moderate to high.
- Needs careful formalization because `d` is defined through minimal distinguishing sets and matroid rank.

Submission value:

- Medium.
- Good mathematically, but not as immediately special-issue-facing as the embedding theorem.

### 5. Combined lossless frontier corollary

Status: **already effectively available from existing theorems**.

- No new mechanization appears necessary here.
- The result is a packaging corollary of the ambiguity converse, the exact block law, and the query lower bound.

Claim shape:

> In the tag-free corner, the minimum lossless query budget is `d`; in the zero-query corner, the minimum lossless tag budget is `\log_2 A_\pi`; together these are the two principal axes of the lossless semantic-compression frontier.

Why it fits:

- Almost already in the paper.
- Could be elevated into a cleaner named corollary if desired.

Lean effort:

- Very low.

Submission value:

- Medium.
- Mostly expositional; useful if we want a sharper “lossless frontier” slogan.

## Higher-risk candidates

### 6. Approximation or hardness of estimating `A_\pi`

Status: **explored, not attempted**.

- This would require a specific implicit representation model (circuits, programs, neural nets, etc.) and a reduction.
- It is not low-hanging fruit relative to the current finite explicit development.

Claim shape:

> For implicit representations of `\pi` (circuits, programs, learned models), computing or approximating `A_\pi` is hard under an appropriate encoding.

Why it fits:

- This would directly address the computational-theoretic gap around the converse.

Lean effort:

- High.
- Requires a concrete implicit model and a reduction.

Submission value:

- Medium to high, but only if done cleanly.
- Not a quick win.

### 7. Formal privacy theorem with leakage measure

Status: **explored, not attempted**.

- The present infrastructure supports disclosure-pattern remarks, but not a leakage theorem.
- A real privacy theorem would need a formal object such as transcript leakage, a privacy budget, or an explicit information-disclosure functional.

Claim shape:

> Zero-error identification without tags requires at least a specified amount of information disclosure under a formal leakage model.

Why it fits:

- Strong special-issue relevance.

Lean effort:

- High.
- Needs a new formal object: leakage, transcript entropy, privacy budget, or related measure.

Submission value:

- High if done well, but not a low-hanging-fruit extension.

## Recommended order of attack

1. **Embedding-collapse theorem**
   - Best mix of impact, fit, and low proof burden.
2. **Tag-vs-query disclosure comparison**
   - Easy to package, useful for privacy/security framing.
3. **Profile-refinement monotonicity**
   - Good mathematical extension once the first two are stable.
4. **Everything else only if there is time**

## Practical recommendation

If the goal is to strengthen the current JSAIT submission rather than start a second paper, the best use of effort is:

- add the embedding-collapse interpretation now in prose,
- optionally package the tag-vs-query disclosure contrast as a short corollary later,
- leave `A_\pi` estimation hardness and full privacy formalization for follow-on work.
