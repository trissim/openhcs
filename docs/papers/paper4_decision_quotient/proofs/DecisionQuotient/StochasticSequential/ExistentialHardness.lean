/-
  Paper 4b: Stochastic and Sequential Regimes

  ExistentialHardness.lean - Honest hardness wrappers for stochastic existential
  queries.

  This file packages the mechanized existential-majority source problem,
  witness-over-verifier source characterization, and the reduction into the
  stochastic anchor query family. As elsewhere in this repository, these are
  honest internal hardness wrappers rather than end-to-end oracle-TM
  `NP^PP` theorems.
-/

import DecisionQuotient.StochasticSequential.NPPPHardness

namespace DecisionQuotient.StochasticSequential

abbrev HonestExistentialAnchorSourceFitsNPOverPPStyle : Prop :=
  FitsNPOverPPStyle ExistsMajorityInput PackedXWitness
    (fun q => q ∈ ExistsMajoritySourceLanguage)
    ExistsMajorityPackedWitnessRel

abbrev HonestExistentialAnchorHard : Prop :=
  HonestExistsMajorityStochasticAnchorHard

abbrev HonestExistentialAnchorQueryFamilyHard : Prop :=
  HonestExistsMajorityStochasticAnchorQueryFamilyHard

abbrev HonestExistentialAnchorNPOverPPStyleHard : Prop :=
  HonestNPOverPPStyleHard ExistsMajorityInput ExistsMajorityAnchorLanguage

abbrev HonestExistentialAnchorQueryFamilyNPOverPPStyleHard : Prop :=
  HonestNPOverPPStyleHard ExistsMajorityAnchorQueryInstance
    ExistsMajorityAnchorQueryLanguage

theorem existential_anchor_source_fits_np_over_ppstyle_honest :
    FitsNPOverPPStyle ExistsMajorityInput PackedXWitness
      (fun q => q ∈ ExistsMajoritySourceLanguage)
      ExistsMajorityPackedWitnessRel :=
  existsMajority_source_fits_np_over_ppstyle

theorem existential_anchor_hard_honest :
    HonestExistentialAnchorHard :=
  existsMajority_stochastic_anchor_hard

theorem existential_anchor_query_family_hard_honest :
    HonestExistentialAnchorQueryFamilyHard :=
  existsMajority_stochastic_anchor_query_family_hard

theorem existential_anchor_np_over_ppstyle_hard_honest :
    HonestExistentialAnchorNPOverPPStyleHard :=
  existsMajority_anchor_language_honest_np_over_ppstyle_hard

theorem existential_anchor_query_family_np_over_ppstyle_hard_honest :
    HonestExistentialAnchorQueryFamilyNPOverPPStyleHard :=
  existsMajority_anchor_query_family_honest_np_over_ppstyle_hard

end DecisionQuotient.StochasticSequential
