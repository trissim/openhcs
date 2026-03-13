import Paper1IT.EmpiricalAuditBridge

namespace Paper1IT.Generated

open Ssot.Paper1IT

def HashDemoCertificate : EmpiricalCertificate :=
  { fiberSizes := [13, 6, 5, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  , fiberWeightsDescending := [[1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000], [1000000, 1000000, 1000000, 1000000, 1000000, 1000000], [1000000, 1000000, 1000000, 1000000, 1000000], [1000000, 1000000], [1000000, 1000000], [1000000, 1000000], [1000000], [1000000], [1000000], [1000000], [1000000], [1000000], [1000000], [1000000], [1000000]]
  , weightScale := 1000000
  , reportedMaxFiberCard := 13
  , reportedTotalItems := 39
  , reportedThresholdBits := 4
  , budgetRows := [
    { bits := 0, reportedTagAlphabet := 1, reportedWorstFiberFloorNumerator := 12, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 24, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 24000000, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := false },
    { bits := 1, reportedTagAlphabet := 2, reportedWorstFiberFloorNumerator := 11, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 18, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 18000000, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := false },
    { bits := 2, reportedTagAlphabet := 4, reportedWorstFiberFloorNumerator := 9, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 12, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 12000000, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := false },
    { bits := 3, reportedTagAlphabet := 8, reportedWorstFiberFloorNumerator := 5, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 5, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 5000000, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := false },
    { bits := 4, reportedTagAlphabet := 16, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := true },
    { bits := 5, reportedTagAlphabet := 32, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := true },
    { bits := 6, reportedTagAlphabet := 64, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 13, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 39, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 39000000, reportedZeroErrorFeasible := true }
    ]
  }

theorem HashDemoCertificate_valid : ValidEmpiricalCertificate HashDemoCertificate := by
  native_decide

end Paper1IT.Generated
