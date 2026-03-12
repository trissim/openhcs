# Decision Quotient (ToC) - Lean 4 Formalization

## Overview

This directory contains the complete Lean 4 formalization for Decision Quotient (ToC).

- **Lines:** 33311
- **Theorems:** 1378
- **`sorry` placeholders:** 0


## Requirements

- Lean 4 (see `lean-toolchain` for exact version)
- Mathlib4

## Building

```bash
lake update
lake build
```

## File Structure

| File | Module |
|------|--------|
| `AbstractionCollapse.lean` | AbstractionCollapse |
| `AlgorithmComplexity.lean` | AlgorithmComplexity |
| `Basic.lean` | Basic |
| `BayesFoundations.lean` | BayesFoundations |
| `BayesFromDQ.lean` | BayesFromDQ |
| `BayesOptimalityProof.lean` | BayesOptimalityProof |
| `BayesianBridge.lean` | BayesianBridge |
| `Bridges.lean` | Bridges |
| `CheckAxioms.lean` | CheckAxioms |
| `ClaimClosure.lean` | ClaimClosure |
| `Complexity.lean` | Complexity |
| `ComplexityMain.lean` | ComplexityMain |
| `Computation.lean` | Computation |
| `ComputationalDecisionProblem.lean` | ComputationalDecisionProblem |
| `CrossPaperDependencies.lean` | CrossPaperDependencies |
| `DecisionGeometry.lean` | DecisionGeometry |
| `Dichotomy.lean` | Dichotomy |
| `Approximation.lean` | Approximation |
| `Elicitation.lean` | Elicitation |
| `ValueOfInformation.lean` | ValueOfInformation |
| `Finite.lean` | Finite |
| `FunctionalInformation.lean` | FunctionalInformation |
| `GraphNontriviality.lean` | GraphNontriviality |
| `HandleAliases.lean` | HandleAliases |
| `ApproximationHardness.lean` | ApproximationHardness |
| `ConfigReduction.lean` | ConfigReduction |
| `CountingComplexity.lean` | CountingComplexity |
| `CoveringLowerBound.lean` | CoveringLowerBound |
| `ETH.lean` | ETH |
| `QBF.lean` | QBF |
| `SAT.lean` | SAT |
| `AnchorSufficiency.lean` | AnchorSufficiency |
| `Summary.lean` | Summary |
| `VectorE_CoNP.lean` | VectorE_CoNP |
| `Sigma2PHardness.lean` | Sigma2PHardness |
| `Hardness.lean` | Hardness |
| `HardnessDistribution.lean` | HardnessDistribution |
| `RDSrank.lean` | RDSrank |
| `RateDistortion.lean` | RateDistortion |
| `Information.lean` | Information |
| `Instances.lean` | Instances |
| `IntegrityCompetence.lean` | IntegrityCompetence |
| `InteriorVerification.lean` | InteriorVerification |
| `PhysicalBudgetCrossover.lean` | PhysicalBudgetCrossover |
| `AccessRegime.lean` | AccessRegime |
| `AnchorChecks.lean` | AnchorChecks |
| `AssumptionNecessity.lean` | AssumptionNecessity |
| `BoundedAcquisition.lean` | BoundedAcquisition |
| `ClaimTransport.lean` | ClaimTransport |
| `ConstraintForcing.lean` | ConstraintForcing |
| `Conversation.lean` | Conversation |
| `DecisionEquivalence.lean` | DecisionEquivalence |
| `DecisionTime.lean` | DecisionTime |
| `DiscreteSpacetime.lean` | DiscreteSpacetime |
| `HeisenbergStrong.lean` | HeisenbergStrong |
| `Instantiation.lean` | Instantiation |
| `IntegrityEquilibrium.lean` | IntegrityEquilibrium |
| `InvariantAgreement.lean` | InvariantAgreement |
| `LocalityPhysics.lean` | LocalityPhysics |
| `MeasureNecessity.lean` | MeasureNecessity |
| `MolecularIntegrity.lean` | MolecularIntegrity |
| `ObserverRelativeState.lean` | ObserverRelativeState |
| `PhysicalCore.lean` | PhysicalCore |
| `PhysicalHardness.lean` | PhysicalHardness |
| `PhysicalIncompleteness.lean` | PhysicalIncompleteness |
| `TUR.lean` | TUR |
| `TemporalCountingGap.lean` | TemporalCountingGap |
| `TransportCost.lean` | TransportCost |
| `Uncertainty.lean` | Uncertainty |
| `WassersteinIntegrity.lean` | WassersteinIntegrity |
| `WolpertConstraints.lean` | WolpertConstraints |
| `WolpertDecomposition.lean` | WolpertDecomposition |
| `WolpertMismatch.lean` | WolpertMismatch |
| `WolpertResidual.lean` | WolpertResidual |
| `PolynomialReduction.lean` | PolynomialReduction |
| `QueryComplexity.lean` | QueryComplexity |
| `Quotient.lean` | Quotient |
| `Reduction.lean` | Reduction |
| `Reduction_AllCoords.lean` | Reduction_AllCoords |
| `FisherInformation.lean` | FisherInformation |
| `ProbabilisticBridge.lean` | ProbabilisticBridge |
| `AlgorithmComplexity.lean` | AlgorithmComplexity |
| `Basic.lean` | Basic |
| `ClaimClosure.lean` | ClaimClosure |
| `Computation.lean` | Computation |
| `CrossRegime.lean` | CrossRegime |
| `Dichotomy.lean` | Dichotomy |
| `Economics.lean` | Economics |
| `Finite.lean` | Finite |
| `Hardness.lean` | Hardness |
| `HardnessDistribution.lean` | HardnessDistribution |
| `Hierarchy.lean` | Hierarchy |
| `Information.lean` | Information |
| `Instances.lean` | Instances |
| `PolynomialReduction.lean` | PolynomialReduction |
| `QueryComplexity.lean` | QueryComplexity |
| `Quotient.lean` | Quotient |
| `SubstrateCost.lean` | SubstrateCost |
| `Summary.lean` | Summary |
| `TemporalIntegrity.lean` | TemporalIntegrity |
| `TemporalLearning.lean` | TemporalLearning |
| `ThermodynamicLift.lean` | ThermodynamicLift |
| `Tractability.lean` | Tractability |
| `StochasticSequential.lean` | StochasticSequential |
| `Sufficiency.lean` | Sufficiency |
| `Summary.lean` | Summary |
| `ThermodynamicLift.lean` | ThermodynamicLift |
| `ToolCollapse.lean` | ToolCollapse |
| `BoundedActions.lean` | BoundedActions |
| `Dimensional.lean` | Dimensional |
| `FPT.lean` | FPT |
| `SeparableUtility.lean` | SeparableUtility |
| `StructuralRank.lean` | StructuralRank |
| `Tightness.lean` | Tightness |
| `TreeStructure.lean` | TreeStructure |
| `Tractable.lean` | Tractable |
| `UniverseObjective.lean` | UniverseObjective |
| `WitnessCheckingDuality.lean` | WitnessCheckingDuality |
| `DecisionQuotient.lean` | DecisionQuotient |
| `DeclInfoExport.lean` | DeclInfoExport |
| `DependencyGraph.lean` | DependencyGraph |
| `GraphExport.lean` | GraphExport |
| `lakefile.lean` | lakefile |

## Verification

All files compile with 0 `sorry` placeholders. All claims are machine-verified.

## License

MIT License - See main repository for details.

---
*Auto-generated by build_papers.py*
