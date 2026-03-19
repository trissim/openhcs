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
  | .integer => "integer"

def loweringKindToString : LoweringKind → String
  | .vmap => "vmap"
  | .reduceSum => "reduce_sum"
  | .elemBinaryAdd => "elem_binary_add"
  | .elemBinarySub => "elem_binary_sub"
  | .norm => "norm"
  | .rowWiseNorm => "row_wise_norm"
  | .distance => "distance"
  | .rowWiseDistance => "row_wise_distance"
  | .supportConditioning => "support_conditioning"
  | .normalizeProbabilityVector => "normalize_probability_vector"
  | .uniformProbabilityVectorLike => "uniform_probability_vector_like"
  | .noopBiasedProbabilityVectorLike => "noop_biased_probability_vector_like"
  | .topKWithTiesMask => "top_k_with_ties_mask"
  | .ambiguityBandMask => "ambiguity_band_mask"
  | .stableArgmaxMasked => "stable_argmax_masked"
  | .axisAngleQuaternion => "axis_angle_quaternion"
  | .localTranslationStencil3D => "local_translation_stencil_3d"
  | .localRotationStencil3D => "local_rotation_stencil_3d"
  | .quaternionDictionary8 => "quaternion_dictionary_8"
  | .rigidTransform3D => "rigid_transform_3d"
  | .pairwiseDistances => "pairwise_distances"
  | .pairwiseDistances3D => "pairwise_distances_3d"
  | .minimumImagePairwiseDistances => "minimum_image_pairwise_distances"
  | .applyCutoff => "apply_cutoff"
  | .lennardJones => "lennard_jones"
  | .sumPairPotentials => "sum_pair_potentials"
  | .sumPairPotentialsMatrix => "sum_pair_potentials_matrix"
  | .sumPairPotentials3D => "sum_pair_potentials_3d"
  | .typedLennardJonesMatrix => "typed_lennard_jones_matrix"
  | .typedLennardJonesCutoff => "typed_lennard_jones_cutoff"
  | .coulombCutoff => "coulomb_cutoff"
  | .upperTriangleMaskedSum => "upper_triangle_masked_sum"
  | .ewaldRealSpaceKernel => "ewald_real_space_kernel"

def jsonStringOrNull : Option String → String
  | some value => s!"\"{jsonEscape value}\""
  | none => "null"

def argSpecToJson (arg : ArgSpec) : String :=
  "{" ++
    "\"name\":\"" ++ jsonEscape arg.name ++ "\"," ++
    "\"kind\":\"" ++ exprKindToString arg.kind ++ "\"," ++
    "\"scalar_type\":" ++ jsonStringOrNull (arg.scalarType?.map scalarTypeToString) ++
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

def exportPrimitivesJson : String :=
  "[" ++ ",".intercalate (exportPrimitives.map primitiveToJson) ++ "]"

def writeExportFile (filePath : System.FilePath) : IO Unit := do
  IO.FS.writeFile filePath exportPrimitivesJson

elab "#export_arraydsl_json" path:str : command => do
  let filePath := path.getString
  writeExportFile ⟨filePath⟩
  Lean.logInfo m!"ArrayDSL primitive export written to {filePath}: {exportPrimitives.length} primitives"

end ArrayDSL
end Computation
end DecisionQuotient
