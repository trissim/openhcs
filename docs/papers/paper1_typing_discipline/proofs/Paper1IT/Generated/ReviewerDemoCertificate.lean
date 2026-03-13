import Paper1IT.EmpiricalAuditBridge

namespace Paper1IT.Generated

open Ssot.Paper1IT

def ReviewerDemoCertificate : EmpiricalCertificate :=
  { fiberSizes := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  , fiberWeightsDescending := [[3000000], [3000000], [3000000], [3000000], [3000000], [3000000], [2000000], [2000000], [2000000], [2000000], [2000000], [2000000], [2000000], [2000000], [2000000], [2000000], [1000000], [1000000], [1000000], [1000000]]
  , weightScale := 1000000
  , reportedMaxFiberCard := 1
  , reportedTotalItems := 20
  , reportedThresholdBits := 0
  , budgetRows := [
    { bits := 0, reportedTagAlphabet := 1, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 1, reportedTagAlphabet := 2, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 2, reportedTagAlphabet := 4, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 3, reportedTagAlphabet := 8, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 4, reportedTagAlphabet := 16, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 5, reportedTagAlphabet := 32, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 6, reportedTagAlphabet := 64, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 7, reportedTagAlphabet := 128, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true },
    { bits := 8, reportedTagAlphabet := 256, reportedWorstFiberFloorNumerator := 0, reportedWorstFiberFloorDenominator := 1, reportedExactUniformErrorNumerator := 0, reportedExactUniformErrorDenominator := 20, reportedExactWeightedErrorNumeratorScaled := 0, reportedExactWeightedErrorDenominatorScaled := 42000000, reportedZeroErrorFeasible := true }
    ]
  }

theorem ReviewerDemoCertificate_valid : ValidEmpiricalCertificate ReviewerDemoCertificate := by
  native_decide

end Paper1IT.Generated
