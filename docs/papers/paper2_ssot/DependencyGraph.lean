/-
  DependencyGraph.lean — Build-time dependency graph extraction

  Extracts every declaration in the environment and its transitive
  dependencies, outputs JSON for visualization and integrity analysis.

  Usage: Copied into each paper's Lean source tree by build_papers.py,
  then compiled as the `DependencyGraph` module by one package visible
  to the current paper build. GraphBridges.lean and GraphExport.lean
  import that module directly.
-/

import Lean
import Lean.Util.FoldConsts

open Lean System
open Lean Elab Command

/-- Collect all constant names referenced in a ConstantInfo's type and value. -/
def collectDeps (info : ConstantInfo) : Array Name :=
  let deps := info.type.foldConsts #[] fun n acc => acc.push n
  match info with
  | ConstantInfo.defnInfo v  => v.value.foldConsts deps fun n acc => acc.push n
  | ConstantInfo.thmInfo v   => v.value.foldConsts deps fun n acc => acc.push n
  | ConstantInfo.opaqueInfo v => v.value.foldConsts deps fun n acc => acc.push n
  | _ => deps

/-- Filter: keep only declarations from the project namespaces proper. -/
def isProjectName (n : Name) : Bool :=
  let s := n.toString
  s.startsWith "Leverage" ||
  s.startsWith "Ssot" ||
  s.startsWith "DecisionQuotient" ||
  s.startsWith "AbstractClassSystem"

/-- Collapse external dependencies into a small set of foundation witnesses. -/
def foundationBucket (n : Name) : Option String :=
  let s := n.toString
  if s == "Nat" || s.startsWith "Nat." ||
     s == "Int" || s.startsWith "Int." ||
     s == "Fin" || s.startsWith "Fin." then
    some "FOUNDATION.Counting"
  else if s == "List" || s.startsWith "List." ||
          s == "Array" || s.startsWith "Array." ||
          s == "Finset" || s.startsWith "Finset." then
    some "FOUNDATION.Collections"
  else if s == "Bool" || s.startsWith "Bool." then
    some "FOUNDATION.Boolean"
  else if s == "True" || s == "False" ||
          s == "And" || s == "Or" || s == "Not" || s == "Iff" ||
          s == "Exists" ||
          s == "Subtype" || s.startsWith "Subtype." ||
          s == "Prod" || s.startsWith "Prod." ||
          s == "Sum" || s.startsWith "Sum." ||
          s == "Sigma" || s.startsWith "Sigma." ||
          s == "PSigma" || s.startsWith "PSigma." then
    some "FOUNDATION.Logic"
  else if s == "Eq" || s.startsWith "Eq." ||
          s == "HEq" || s.startsWith "HEq." ||
          s == "Decidable" || s.startsWith "Decidable." ||
          s.startsWith "Classical." then
    some "FOUNDATION.Equality"
  else if s == "Quot" || s.startsWith "Quot." ||
          s == "propext" ||
          s == "Set" || s.startsWith "Set." ||
          s == "Membership" || s.startsWith "Membership." then
    some "FOUNDATION.SetTheory"
  else if s == "LT" || s.startsWith "LT." ||
          s == "LE" || s.startsWith "LE." ||
          s == "GT" || s.startsWith "GT." ||
          s == "GE" || s.startsWith "GE." ||
          s.startsWith "Preorder." ||
          s.startsWith "PartialOrder." ||
          s.startsWith "LinearOrder." ||
          s.startsWith "Add." ||
          s.startsWith "Sub." ||
          s.startsWith "Mul." ||
          s.startsWith "Div." then
    some "FOUNDATION.OrderAlgebra"
  else if s.startsWith "MeasureTheory." || s.startsWith "ProbabilityTheory." then
    some "FOUNDATION.MeasureTheory"
  else if s == "Real" || s.startsWith "Real." ||
          s == "NNReal" || s.startsWith "NNReal." ||
          s == "ENNReal" || s.startsWith "ENNReal." ||
          s == "Rat" || s.startsWith "Rat." ||
          s == "Complex" || s.startsWith "Complex." then
    some "FOUNDATION.Analysis"
  else
    none

/-- Persistent Lean-side declaration of graph-only semantic bridge edges. -/
initialize graphBridgeExt :
    SimplePersistentEnvExtension (String × String) (Array (String × String)) ←
  registerSimplePersistentEnvExtension {
    name := `graphBridgeExt
    addEntryFn := fun s e => s.push e
    addImportedFn := fun ss => Id.run do
      let mut out := #[]
      for s in ss do
        out := out ++ s
      pure out
  }

syntax (name := graphBridgeCmd) "#graph_bridge " str " => " str : command

elab_rules : command
  | `(#graph_bridge $src:str => $tgt:str) => do
      modifyEnv fun env => graphBridgeExt.addEntry env (src.getString, tgt.getString)

/-- JSON escape for strings -/
def jsonEscape (s : String) : String :=
  s.replace "\\" "\\\\" |>.replace "\"" "\\\"" |>.replace "\n" "\\n"

def foundationSummary (bucket : String) : String :=
  match bucket with
  | "FOUNDATION.Counting" =>
      "Collapsed witness for natural-number, integer, and finite-index counting structure."
  | "FOUNDATION.Collections" =>
      "Collapsed witness for lists, arrays, and finite-set collection structure."
  | "FOUNDATION.Boolean" =>
      "Collapsed witness for Boolean branching structure."
  | "FOUNDATION.Logic" =>
      "Collapsed witness for propositional, existential, and product/sum logical structure."
  | "FOUNDATION.Equality" =>
      "Collapsed witness for equality, decidability, and classical reasoning structure."
  | "FOUNDATION.SetTheory" =>
      "Collapsed witness for sets, membership, quotient, and extensional set structure."
  | "FOUNDATION.OrderAlgebra" =>
      "Collapsed witness for order, comparison, and algebraic operator structure."
  | "FOUNDATION.MeasureTheory" =>
      "Collapsed witness for measure-theoretic and probabilistic structure."
  | "FOUNDATION.Analysis" =>
      "Collapsed witness for real/complex analytic structure."
  | _ =>
      "Collapsed first-principle witness node."

def jsonStringOrNull : Option String → String
  | some s => s!"\"{jsonEscape s}\""
  | none => "null"

/-! ### JSON export for visualization pipeline -/

elab "#export_graph_json" path:str : command => do
  let env ← getEnv
  let filePath := path.getString

  let mut nodes : Array String := #[]
  let mut edges : Array String := #[]
  let mut foundationNodes : Array String := #[]

  let entries := (Lean.Environment.constants env).fold
    (init := #[]) fun acc name info => acc.push (name, info)

  for (name, info) in entries do
    if isProjectName name then
      let kind := match info with
        | ConstantInfo.thmInfo _    => "theorem"
        | ConstantInfo.defnInfo _   => "definition"
        | ConstantInfo.axiomInfo _  => "axiom"
        | ConstantInfo.opaqueInfo _ => "opaque"
        | _                         => "other"
      nodes := nodes.push
        s!"\{\"id\":\"{jsonEscape name.toString}\",\"kind\":\"{kind}\"}"

      let deps := collectDeps info
      let mut seenTargets : Array String := #[]
      for d in deps do
        if isProjectName d then
          let target := d.toString
          if !seenTargets.contains target then
            seenTargets := seenTargets.push target
            edges := edges.push
              s!"\{\"source\":\"{jsonEscape name.toString}\",\"target\":\"{jsonEscape target}\"}"
        else
          match foundationBucket d with
          | some bucket =>
              if !foundationNodes.contains bucket then
                foundationNodes := foundationNodes.push bucket
                nodes := nodes.push
                  s!"\{\"id\":\"{jsonEscape bucket}\",\"kind\":\"axiom\"}"
              if !seenTargets.contains bucket then
                seenTargets := seenTargets.push bucket
                edges := edges.push
                  s!"\{\"source\":\"{jsonEscape name.toString}\",\"target\":\"{jsonEscape bucket}\"}"
          | none => pure ()

  for (source, target) in graphBridgeExt.getState env do
    edges := edges.push
      s!"\{\"source\":\"{jsonEscape source}\",\"target\":\"{jsonEscape target}\"}"

  let json := s!"\{\"nodes\":[{",".intercalate nodes.toList}],\"edges\":[{",".intercalate edges.toList}]}"

  IO.FS.writeFile ⟨filePath⟩ json
  Lean.logInfo m!"Graph exported to {filePath}: {nodes.size} nodes, {edges.size} edges"

elab "#export_decl_info_json" path:str : command => do
  let env ← getEnv
  let filePath := path.getString

  let mut entries : Array String := #[]
  let mut seenFoundation : Array String := #[]

  let constants := (Lean.Environment.constants env).fold
    (init := #[]) fun acc name info => acc.push (name, info)

  for (name, info) in constants do
    if isProjectName name then
      let kind := match info with
        | ConstantInfo.thmInfo _    => "theorem"
        | ConstantInfo.defnInfo _   => "definition"
        | ConstantInfo.axiomInfo _  => "axiom"
        | ConstantInfo.opaqueInfo _ => "opaque"
        | _                         => "other"
      let signature ← liftTermElabM do
        let fmt ← Lean.Meta.ppExpr info.type
        pure (toString fmt)
      let doc? ← liftIO <| findDocString? env name (includeBuiltin := false)
      entries := entries.push
        ("\"" ++ jsonEscape name.toString ++ "\":{" ++
          "\"kind\":\"" ++ kind ++ "\"," ++
          "\"origin\":\"project\"," ++
          "\"signature\":\"" ++ jsonEscape signature ++ "\"," ++
          "\"doc\":" ++ jsonStringOrNull doc? ++
        "}")

      let deps := collectDeps info
      for d in deps do
        match foundationBucket d with
        | some bucket =>
            if !seenFoundation.contains bucket then
              seenFoundation := seenFoundation.push bucket
              entries := entries.push
                ("\"" ++ jsonEscape bucket ++ "\":{" ++
                  "\"kind\":\"axiom\"," ++
                  "\"origin\":\"foundation\"," ++
                  "\"signature\":\"" ++ jsonEscape (foundationSummary bucket) ++ "\"," ++
                  "\"doc\":null" ++
                "}")
        | none => pure ()

  let json := "{" ++ ",".intercalate entries.toList ++ "}"
  IO.FS.writeFile ⟨filePath⟩ json
  Lean.logInfo m!"Declaration metadata exported to {filePath}: {entries.size} entries"
