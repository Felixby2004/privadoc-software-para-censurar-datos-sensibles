from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import _run_isolated


def _large_result() -> bytes:
    return b"x" * (4 * 1024 * 1024)


def test_isolated_job_returns_large_result_without_pipe_deadlock():
    result = _run_isolated(_large_result, (), timeout=10)

    assert len(result) == 4 * 1024 * 1024

import asyncio
import app.main as main


def _plus_one(value: int) -> int:
    return value + 1


def test_run_job_uses_reusable_thread_when_process_isolation_disabled(monkeypatch):
    monkeypatch.setattr(main, "IS_VERCEL", False)
    monkeypatch.setattr(main, "USE_ISOLATED_JOBS", False)

    def should_not_run(*args, **kwargs):
        raise AssertionError("No debe crear un proceso aislado en modo reutilizable")

    monkeypatch.setattr(main, "_run_isolated", should_not_run)

    result = asyncio.run(main._run_job(_plus_one, (41,)))

    assert result == 42


def test_default_hard_timeout_is_long_enough_for_slow_ocr():
    assert main.JOB_TIMEOUT_SECONDS >= 600


def _slow_but_valid_result() -> str:
    import time
    time.sleep(0.15)
    return "ok"


def test_reusable_thread_does_not_cancel_valid_ocr_at_hard_timeout(monkeypatch):
    monkeypatch.setattr(main, "IS_VERCEL", False)
    monkeypatch.setattr(main, "USE_ISOLATED_JOBS", False)
    monkeypatch.setattr(main, "JOB_TIMEOUT_SECONDS", 0.01)

    result = asyncio.run(main._run_job(_slow_but_valid_result, ()))

    assert result == "ok"
