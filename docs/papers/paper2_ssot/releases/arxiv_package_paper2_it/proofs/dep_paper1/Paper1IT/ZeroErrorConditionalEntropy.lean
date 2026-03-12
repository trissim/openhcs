import Paper1IT.GraphEntropy

namespace Ssot
namespace GraphEntropy

theorem conditionalEntropy_div_log2_le_optimalExpectedAdaptiveBitLength
    {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (p : VertexDist α) (observe : α → β)
    [Entropy.ClassicalEntropyAssumptions] :
    conditionalEntropyGiven p observe / Real.log 2 ≤
      optimalExpectedAdaptiveBitLength observe (p.pushforward observe) := by
  have hbase := conditionalEntropyGiven_le_log2_mul_expectedAdaptiveBitLength
    p observe (optimalFiberBitLength observe) (optimalFiberBitLength_feasible observe)
  have hlog2pos : 0 < Real.log 2 := by
    exact Real.log_pos (by norm_num : (1 : ℝ) < 2)
  exact (div_le_iff₀ hlog2pos).2 <| by
    simpa [optimalExpectedAdaptiveBitLength, expectedAdaptiveBitLength, mul_comm, mul_left_comm,
      mul_assoc] using hbase

theorem exists_conditionalCodes_within_one_bit_of_conditionalEntropy
    {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (p : VertexDist α) (observe : α → β)
    [Entropy.ClassicalCodingAssumptions] :
    ∃ L : ∀ b, 0 < (p.pushforward observe).prob b → {a : α // observe a = b} → ℕ,
      conditionalExpectedCodeLength p observe L ≤
        conditionalEntropyGiven p observe / Real.log 2 + 1 := by
  exact exists_conditionalCodes_expectedLength_le_entropy_bits_plus_one p observe

theorem zeroErrorConditionalEntropySandwich
    {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (p : VertexDist α) (observe : α → β)
    [Entropy.ClassicalEntropyAssumptions] [Entropy.ClassicalCodingAssumptions] :
    conditionalEntropyGiven p observe / Real.log 2 ≤
      optimalExpectedAdaptiveBitLength observe (p.pushforward observe)
    ∧
    ∃ L : ∀ b, 0 < (p.pushforward observe).prob b → {a : α // observe a = b} → ℕ,
      conditionalExpectedCodeLength p observe L ≤
        conditionalEntropyGiven p observe / Real.log 2 + 1 := by
  exact ⟨
    conditionalEntropy_div_log2_le_optimalExpectedAdaptiveBitLength p observe,
    exists_conditionalCodes_within_one_bit_of_conditionalEntropy p observe
  ⟩

end GraphEntropy
end Ssot
