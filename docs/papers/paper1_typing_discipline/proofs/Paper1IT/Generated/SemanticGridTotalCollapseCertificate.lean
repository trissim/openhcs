import Paper1IT.EmpiricalAuditBridge

namespace Paper1IT.Generated

open Ssot.Paper1IT

def SemanticGridTotalCollapseCertificate : EmpiricalCertificate :=
  { fiberSizes := [120]
  , fiberWeightsDescending := [[3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 2000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000]]
  , weightScale := 1000000
  , reportedMaxFiberCard := 120
  , reportedTotalItems := 120
  , reportedThresholdBits := 7
  , budgetRows := [
    { bits := 0, reportedTagAlphabet := 1, reportedWorstFiberFloorNumerator := 119, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 119, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 237000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 1, reportedTagAlphabet := 2, reportedWorstFiberFloorNumerator := 118, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 118, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 234000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 2, reportedTagAlphabet := 4, reportedWorstFiberFloorNumerator := 116, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 116, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 228000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 3, reportedTagAlphabet := 8, reportedWorstFiberFloorNumerator := 112, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 112, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 216000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 4, reportedTagAlphabet := 16, reportedWorstFiberFloorNumerator := 104, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 104, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 192000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 5, reportedTagAlphabet := 32, reportedWorstFiberFloorNumerator := 88, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 88, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 144000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 6, reportedTagAlphabet := 64, reportedWorstFiberFloorNumerator := 56, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 56, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 72000000, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := false },
    { bits := 7, reportedTagAlphabet := 128, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := true },
    { bits := 8, reportedTagAlphabet := 256, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 120, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 120, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 240000000, reportedZeroErrorFeasible := true }
    ]
  }

theorem SemanticGridTotalCollapseCertificate_valid : ValidEmpiricalCertificate SemanticGridTotalCollapseCertificate := by
  native_decide

end Paper1IT.Generated
