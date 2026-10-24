from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import _run_isolated


def _large_result() -> bytes:
    return b"x" * (4 * 1024 * 1024)


def test_isolated_job_returns_large_result_without_pipe_deadlock():
    result = _run_isolated(_large_result, (), timeout=10)

    assert len(result) == 4 * 1024 * 1024
