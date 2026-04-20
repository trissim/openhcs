import Lean
import DecisionQuotient.Computation.ArrayDSL

open Lean Elab Command
open DecisionQuotient.Computation.ArrayDSL

namespace DecisionQuotient
namespace Computation
namespace ArrayDSL

def jsonEscape (s : String) : String :=
  s.replace "\\" "\\\\" |>.replace "\"" "\\\"" |>.replace "\n" "\\n"

def exprKindToString : ExprKind → String
  | .scalar => "scalar"
  | .tensor => "tensor"
  | .callable => "callable"

def scalarTypeToString : ScalarType → String
  | .real => "real"
  | .boolean => "boolean"

def loweringKindToString : LoweringKind → String
  | .vmap => "vmap"
  | .reduceSum => "reduce_sum"
  | .elemBinaryAdd => "elem_binary_add"
  | .elemBinarySub => "elem_binary_sub"
  | .norm => "norm"
  | .distance => "distance"
  | .pairwiseDistances => "pairwise_distances"
  | .applyCutoff => "apply_cutoff"
  | .lennardJones => "lennard_jones"
  | .sumPairPotentials => "sum_pair_potentials"

def backendToString : Backend → String
  | .jax => "jax"
  | .torch => "torch"
  | .xla => "xla"
  | .onnx => "onnx"
  | .custom name => name

def jsonStringOrNull : Option String → String
  | some value => s!"\"{jsonEscape value}\""
  | none => "null"

def natOrNull : Option Nat → String
  | some n => toString n
  | none => "null"

def argSpecToJson (arg : ArgSpec) : String :=
  "{" ++
    "\"name\":\"" ++ jsonEscape arg.name ++ "\"," ++
    "\"kind\":\"" ++ exprKindToString arg.kind ++ "\"," ++
    "\"scalar_type\":" ++ jsonStringOrNull (arg.scalarType?.map scalarTypeToString) ++
  "}"

def valueTypeToJson (ty : ValueTypeIR) : String :=
  "{" ++
    "\"kind\":\"" ++ exprKindToString ty.kind ++ "\"," ++
    "\"scalar_type\":" ++ jsonStringOrNull (ty.scalarType?.map scalarTypeToString) ++ "," ++
    "\"rank\":" ++ natOrNull ty.rank? ++
  "}"

def portSpecToJson (port : PortSpecIR) : String :=
  "{" ++
    "\"name\":\"" ++ jsonEscape port.name ++ "\"," ++
    "\"type\":" ++ valueTypeToJson port.ty ++
  "}"

def backendLoweringToJson (lowering : BackendLoweringRef) : String :=
  "{" ++
    "\"backend\":\"" ++ backendToString lowering.backend ++ "\"," ++
    "\"module\":\"" ++ jsonEscape lowering.module ++ "\"," ++
    "\"symbol\":\"" ++ jsonEscape lowering.symbol ++ "\"," ++
    "\"lowering_kind\":\"" ++ loweringKindToString lowering.loweringKind ++ "\"" ++
  "}"

def primitiveToJson (primitive : PrimitiveIR) : String :=
  let argsJson := ",".intercalate (primitive.args.map argSpecToJson)
  "{" ++
    "\"name\":\"" ++ jsonEscape primitive.name ++ "\"," ++
    "\"args\":[" ++ argsJson ++ "]," ++
    "\"result_kind\":\"" ++ exprKindToString primitive.resultKind ++ "\"," ++
    "\"scalar_type\":" ++ jsonStringOrNull (primitive.scalarType?.map scalarTypeToString) ++ "," ++
    "\"lowering_kind\":\"" ++ loweringKindToString primitive.loweringKind ++ "\"," ++
    "\"jax_module\":\"" ++ jsonEscape primitive.jaxModule ++ "\"," ++
    "\"jax_symbol\":\"" ++ jsonEscape primitive.jaxSymbol ++ "\"," ++
    "\"supports_grad\":" ++ (if primitive.supportsGrad then "true" else "false") ++ "," ++
    "\"lean_symbol\":\"" ++ jsonEscape primitive.leanSymbol ++ "\"," ++
    "\"proof_ref\":" ++ jsonStringOrNull primitive.proofRef? ++ "," ++
    "\"proof_status\":" ++ jsonStringOrNull primitive.proofStatus? ++
  "}"

def opSchemaToJson (op : OpSchemaIR) : String :=
  let inputsJson := ",".intercalate (op.inputs.map portSpecToJson)
  let outputsJson := ",".intercalate (op.outputs.map portSpecToJson)
  let loweringsJson := ",".intercalate (op.lowerings.map backendLoweringToJson)
  "{" ++
    "\"op_name\":\"" ++ jsonEscape op.opName ++ "\"," ++
    "\"inputs\":[" ++ inputsJson ++ "]," ++
    "\"outputs\":[" ++ outputsJson ++ "]," ++
    "\"lowerings\":[" ++ loweringsJson ++ "]," ++
    "\"supports_grad\":" ++ (if op.supportsGrad then "true" else "false") ++ "," ++
    "\"lean_symbol\":\"" ++ jsonEscape op.leanSymbol ++ "\"," ++
    "\"proof_ref\":" ++ jsonStringOrNull op.proofRef? ++ "," ++
    "\"proof_status\":" ++ jsonStringOrNull op.proofStatus? ++
  "}"

def universalCatalogToJson (catalog : UniversalIRCatalog) : String :=
  let operationsJson := ",".intercalate (catalog.operations.map opSchemaToJson)
  "{" ++
    "\"schema_version\":\"" ++ jsonEscape catalog.schemaVersion ++ "\"," ++
    "\"dialect\":\"" ++ jsonEscape catalog.dialect ++ "\"," ++
    "\"operations\":[" ++ operationsJson ++ "]" ++
  "}"

def listNatToJson (xs : List Nat) : String :=
  "[" ++ ",".intercalate (xs.map toString) ++ "]"

def listBoolToJson (xs : List Bool) : String :=
  "[" ++ ",".intercalate (xs.map (fun b => if b then "true" else "false")) ++ "]"

def listRatToJson (xs : List Rat) : String :=
  "[" ++ ",".intercalate (xs.map (fun q => s!"\"{q}\"")) ++ "]"

def constValueToJson : ConstValueIR → String
  | .real value =>
      "{" ++
        "\"kind\":\"real\"," ++
        "\"value\":\"" ++ toString value ++ "\"" ++
      "}"
  | .boolean value =>
      "{" ++
        "\"kind\":\"boolean\"," ++
        "\"value\":" ++ (if value then "true" else "false") ++
      "}"
  | .realTensor shape values =>
      "{" ++
        "\"kind\":\"real_tensor\"," ++
        "\"shape\":" ++ listNatToJson shape ++ "," ++
        "\"values\":" ++ listRatToJson values ++
      "}"
  | .booleanTensor shape values =>
      "{" ++
        "\"kind\":\"boolean_tensor\"," ++
        "\"shape\":" ++ listNatToJson shape ++ "," ++
        "\"values\":" ++ listBoolToJson values ++
      "}"

def attrToJson (attr : String × String) : String :=
  "{" ++
    "\"key\":\"" ++ jsonEscape attr.1 ++ "\"," ++
    "\"value\":\"" ++ jsonEscape attr.2 ++ "\"" ++
  "}"

def programNodeToJson (node : ProgramNodeIR) : String :=
  let inputsJson := "[" ++ ",".intercalate (node.inputs.map (fun x => s!"\"{jsonEscape x}\"")) ++ "]"
  let outputsJson := "[" ++ ",".intercalate (node.outputs.map (fun x => s!"\"{jsonEscape x}\"")) ++ "]"
  let attrsJson := "[" ++ ",".intercalate (node.attrs.map attrToJson) ++ "]"
  "{" ++
    "\"node_id\":" ++ toString node.nodeId ++ "," ++
    "\"op_name\":\"" ++ jsonEscape node.opName ++ "\"," ++
    "\"inputs\":" ++ inputsJson ++ "," ++
    "\"outputs\":" ++ outputsJson ++ "," ++
    "\"attrs\":" ++ attrsJson ++
  "}"

def terminatorToJson : BlockTerminatorIR → String
  | .jump target =>
      "{" ++
        "\"kind\":\"jump\"," ++
        "\"target\":" ++ toString target ++
      "}"
  | .branch cond trueTarget falseTarget =>
      "{" ++
        "\"kind\":\"branch\"," ++
        "\"cond\":\"" ++ jsonEscape cond ++ "\"," ++
        "\"true_target\":" ++ toString trueTarget ++ "," ++
        "\"false_target\":" ++ toString falseTarget ++
      "}"
  | .ret values =>
      "{" ++
        "\"kind\":\"return\"," ++
        "\"values\":[" ++ ",".intercalate (values.map (fun x => s!"\"{jsonEscape x}\"")) ++ "]" ++
      "}"

def basicBlockToJson (block : BasicBlockIR) : String :=
  let nodesJson := "[" ++ ",".intercalate (block.nodes.map programNodeToJson) ++ "]"
  "{" ++
    "\"block_id\":" ++ toString block.blockId ++ "," ++
    "\"nodes\":" ++ nodesJson ++ "," ++
    "\"terminator\":" ++ terminatorToJson block.terminator ++
  "}"

def shapeConstraintToJson : ShapeConstraintIR → String
  | .equal lhs rhs =>
      "{" ++
        "\"kind\":\"equal\"," ++
        "\"lhs\":\"" ++ jsonEscape lhs ++ "\"," ++
        "\"rhs\":\"" ++ jsonEscape rhs ++ "\"" ++
      "}"
  | .fixed name shape =>
      "{" ++
        "\"kind\":\"fixed\"," ++
        "\"name\":\"" ++ jsonEscape name ++ "\"," ++
        "\"shape\":" ++ listNatToJson shape ++
      "}"

def edgeToJson (edge : NodeId × NodeId) : String :=
  "{" ++
    "\"from\":" ++ toString edge.1 ++ "," ++
    "\"to\":" ++ toString edge.2 ++
  "}"

def programIRToJson (program : ProgramIR) : String :=
  let inputsJson := "[" ++ ",".intercalate (program.inputs.map portSpecToJson) ++ "]"
  let constantsJson :=
    "[" ++
      ",".intercalate (program.constants.map (fun c =>
        "{" ++
          "\"name\":\"" ++ jsonEscape c.1 ++ "\"," ++
          "\"value\":" ++ constValueToJson c.2 ++
        "}")) ++
    "]"
  let blocksJson := "[" ++ ",".intercalate (program.blocks.map basicBlockToJson) ++ "]"
  let outputsJson := "[" ++ ",".intercalate (program.outputs.map (fun x => s!"\"{jsonEscape x}\"")) ++ "]"
  let edgesJson := "[" ++ ",".intercalate (program.dataEdges.map edgeToJson) ++ "]"
  let constraintsJson := "[" ++ ",".intercalate (program.shapeConstraints.map shapeConstraintToJson) ++ "]"
  "{" ++
    "\"program_name\":\"" ++ jsonEscape program.programName ++ "\"," ++
    "\"inputs\":" ++ inputsJson ++ "," ++
    "\"constants\":" ++ constantsJson ++ "," ++
    "\"blocks\":" ++ blocksJson ++ "," ++
    "\"entry_block\":" ++ toString program.entryBlock ++ "," ++
    "\"outputs\":" ++ outputsJson ++ "," ++
    "\"data_edges\":" ++ edgesJson ++ "," ++
    "\"shape_constraints\":" ++ constraintsJson ++
  "}"

def backendOpCodegenToJson (code : BackendOpCodegenIR) : String :=
  "{" ++
    "\"op_name\":\"" ++ jsonEscape code.opName ++ "\"," ++
    "\"module\":\"" ++ jsonEscape code.module ++ "\"," ++
    "\"symbol\":\"" ++ jsonEscape code.symbol ++ "\"," ++
    "\"lowering_kind\":\"" ++ loweringKindToString code.loweringKind ++ "\"" ++
  "}"

def programCodegenReportToJson (report : ProgramBackendCodegenReport) : String :=
  let requiredJson := "[" ++ ",".intercalate (report.requiredOps.map (fun x => s!"\"{jsonEscape x}\"")) ++ "]"
  let generatedJson := "[" ++ ",".intercalate (report.generated.map backendOpCodegenToJson) ++ "]"
  let missingJson := "[" ++ ",".intercalate (report.missingOps.map (fun x => s!"\"{jsonEscape x}\"")) ++ "]"
  "{" ++
    "\"backend\":\"" ++ backendToString report.backend ++ "\"," ++
    "\"required_ops\":" ++ requiredJson ++ "," ++
    "\"generated\":" ++ generatedJson ++ "," ++
    "\"missing_ops\":" ++ missingJson ++ "," ++
    "\"success\":" ++ (if report.success then "true" else "false") ++ "," ++
    "\"entry_symbol\":\"" ++ jsonEscape report.entrySymbol ++ "\"," ++
    "\"module_stub\":\"" ++ jsonEscape report.moduleStub ++ "\"" ++
  "}"

def exportPrimitivesJson : String :=
  "[" ++ ",".intercalate (exportPrimitives.map primitiveToJson) ++ "]"

def exportUniversalIRJson : String :=
  universalCatalogToJson exportUniversalIRCatalog

def exportPrimitiveCoverageProgramJson : String :=
  programIRToJson primitiveCoverageProgram

def exportPrimitiveCoverageJaxCodegenReportJson : String :=
  programCodegenReportToJson
    (exportUniversalIRCatalog.codegenProgramReport Backend.jax primitiveCoverageProgram)

def writeExportFile (filePath : System.FilePath) : IO Unit := do
  IO.FS.writeFile filePath exportPrimitivesJson

def writeUniversalExportFile (filePath : System.FilePath) : IO Unit := do
  IO.FS.writeFile filePath exportUniversalIRJson

def writeProgramExportFile (filePath : System.FilePath) : IO Unit := do
  IO.FS.writeFile filePath exportPrimitiveCoverageProgramJson

def writeJaxCodegenReportExportFile (filePath : System.FilePath) : IO Unit := do
  IO.FS.writeFile filePath exportPrimitiveCoverageJaxCodegenReportJson

elab "#export_arraydsl_json" path:str : command => do
  let filePath := path.getString
  writeExportFile ⟨filePath⟩
  Lean.logInfo m!"ArrayDSL primitive export written to {filePath}: {exportPrimitives.length} primitives"

elab "#export_arraydsl_universal_json" path:str : command => do
  let filePath := path.getString
  writeUniversalExportFile ⟨filePath⟩
  Lean.logInfo
    m!"ArrayDSL universal IR export written to {filePath}: {exportUniversalIRCatalog.operations.length} operations"

elab "#export_arraydsl_program_ir_json" path:str : command => do
  let filePath := path.getString
  writeProgramExportFile ⟨filePath⟩
  Lean.logInfo
    m!"ArrayDSL program IR export written to {filePath}: {primitiveCoverageProgram.requiredOps.length} required ops"

elab "#export_arraydsl_jax_codegen_report_json" path:str : command => do
  let filePath := path.getString
  writeJaxCodegenReportExportFile ⟨filePath⟩
  Lean.logInfo m!"ArrayDSL JAX codegen report export written to {filePath}"

end ArrayDSL
end Computation
end DecisionQuotient
