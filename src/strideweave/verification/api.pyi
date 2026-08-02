from os import PathLike

from .model import VerificationReport

def test_backend(output: str | PathLike[str] | None = None) -> VerificationReport: ...
