from os import PathLike

from .classification import VerificationProfile
from .model import VerificationReport
from .stage_one import OracleStageResult
from .stage_two import TargetStageResult

def run_oracle_stage(profile: VerificationProfile) -> OracleStageResult: ...
def run_target_stage(
    profile: VerificationProfile, oracle_result: OracleStageResult
) -> TargetStageResult: ...
def verify_backend(
    target: str, *, output: str | PathLike[str] | None = None
) -> VerificationReport: ...
