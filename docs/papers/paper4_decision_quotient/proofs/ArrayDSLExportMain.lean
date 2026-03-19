import DecisionQuotient.Computation.ArrayDSLExport

open DecisionQuotient.Computation.ArrayDSL

def main (args : List String) : IO Unit := do
  let outputPath := match args with
    | [] => "arraydsl_primitives.json"
    | path :: _ => path
  writeExportFile ⟨outputPath⟩
  IO.println s!"ArrayDSL primitive export written to {outputPath}: {exportPrimitives.length} primitives"
