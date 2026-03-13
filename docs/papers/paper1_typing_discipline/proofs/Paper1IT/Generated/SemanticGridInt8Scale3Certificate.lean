import Paper1IT.EmpiricalAuditBridge

namespace Paper1IT.Generated

open Ssot.Paper1IT

def SemanticGridInt8Scale3Certificate : EmpiricalCertificate :=
  { fiberSizes := [77, 11, 10, 5, 3, 2, 2, 2, 2, 2, 2, 1, 1]
  , fiberWeightsDescending := [[3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000], [1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000], [3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000], [3000000, 3000000, 3000000, 3000000, 3000000], [3000000, 3000000, 3000000], [3000000, 3000000], [3000000, 3000000], [3000000, 3000000], [2000000, 2000000], [1000000, 1000000], [1000000, 1000000], [2000000], [1000000]]
  , weightScale := 1000000
  , reportedMaxFiberCard := 77
  , reportedTotalItems := 120
  , reportedThresholdBits := 7
  , budgetRows := [
    { bits := 0, reportedTagAlphabet := 1, reportedWorstFiberFloorNumerator := 76, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 107, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 211000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 1, reportedTagAlphabet := 2, reportedWorstFiberFloorNumerator := 75, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 96, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 185000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 2, reportedTagAlphabet := 4, reportedWorstFiberFloorNumerator := 73, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 87, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 162000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 3, reportedTagAlphabet := 8, reportedWorstFiberFloorNumerator := 69, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 74, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 131000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 4, reportedTagAlphabet := 16, reportedWorstFiberFloorNumerator := 61, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 61, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 98000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 5, reportedTagAlphabet := 32, reportedWorstFiberFloorNumerator := 45, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 45, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 66000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 6, reportedTagAlphabet := 64, reportedWorstFiberFloorNumerator := 13, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 13, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 13000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 7, reportedTagAlphabet := 128, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := true },
    { bits := 8, reportedTagAlphabet := 256, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 77, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := true }
    ]
  }

theorem SemanticGridInt8Scale3Certificate_valid : ValidEmpiricalCertificate SemanticGridInt8Scale3Certificate := by
  native_decide

end Paper1IT.Generated
