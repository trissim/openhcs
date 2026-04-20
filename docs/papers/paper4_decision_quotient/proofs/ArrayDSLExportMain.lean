import DecisionQuotient.Computation.ArrayDSLExport

open DecisionQuotient.Computation.ArrayDSL

def main (args : List String) : IO Unit := do
  match args with
  | "--universal" :: path :: _ =>
      writeUniversalExportFile ⟨path⟩
      IO.println
        s!"ArrayDSL universal IR export written to {path}: {exportUniversalIRCatalog.operations.length} operations"
  | "--universal" :: [] =>
      let path := "arraydsl_universal_ir.json"
      writeUniversalExportFile ⟨path⟩
      IO.println
        s!"ArrayDSL universal IR export written to {path}: {exportUniversalIRCatalog.operations.length} operations"
  | "--program-ir" :: path :: _ =>
      writeProgramExportFile ⟨path⟩
      IO.println
        s!"ArrayDSL program IR export written to {path}: {primitiveCoverageProgram.requiredOps.length} required ops"
  | "--program-ir" :: [] =>
      let path := "arraydsl_program_ir.json"
      writeProgramExportFile ⟨path⟩
      IO.println
        s!"ArrayDSL program IR export written to {path}: {primitiveCoverageProgram.requiredOps.length} required ops"
  | "--jax-codegen-report" :: path :: _ =>
      writeJaxCodegenReportExportFile ⟨path⟩
      IO.println s!"ArrayDSL JAX codegen report export written to {path}"
  | "--jax-codegen-report" :: [] =>
      let path := "arraydsl_jax_codegen_report.json"
      writeJaxCodegenReportExportFile ⟨path⟩
      IO.println s!"ArrayDSL JAX codegen report export written to {path}"
  | _ =>
      let outputPath := match args with
        | [] => "arraydsl_primitives.json"
        | path :: _ => path
      writeExportFile ⟨outputPath⟩
      IO.println
        s!"ArrayDSL primitive export written to {outputPath}: {exportPrimitives.length} primitives"
