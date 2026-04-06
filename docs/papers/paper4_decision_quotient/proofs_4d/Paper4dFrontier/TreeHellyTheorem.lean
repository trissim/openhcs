import Paper4dFrontier.TreeHellyEquivFinset
import Mathlib.Data.Fintype.Option

namespace Paper4dFrontier

open Classical

theorem treeHellyFinset_empty : TreeHellyFinsetProp PEmpty := by
  intro F hF G hT hconn hpair
  rcases hF with ⟨s, hs⟩
  rcases (hconn s hs).nonempty with ⟨x⟩
  cases x.1

theorem treeHellyFinset_all (α : Type*) [Fintype α] : TreeHellyFinsetProp α := by
  classical
  refine Fintype.induction_empty_option (P := fun α _ => TreeHellyFinsetProp α)
    (fun α β _ e h => by
      letI : Fintype α := Fintype.ofEquiv β e.symm
      exact treeHellyFinset_of_equiv e h)
    treeHellyFinset_empty
    (fun α _ h => treeHelly_option_finset h)
    α

end Paper4dFrontier
