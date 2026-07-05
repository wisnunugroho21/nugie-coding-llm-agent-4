"""
Real code execution + test validation.

OpenCoder's synthesis pipelines only keep synthesized samples whose code actually
runs and passes generated test cases (Sec. 2.3 "High Quality Code Snippet ...
LLM-synthesized with test validation"; Sec. 4.1 Educational/Diverse synthesis
"executed using a Python interpreter ... Only the data samples that successfully
pass the tests are retained"). This module does exactly that: it concatenates a
candidate solution with its tests and runs them in a subprocess with a timeout,
returning whether they passed.

This is a *timeout-guarded subprocess*, not a hardened sandbox. For untrusted
synthesis at scale, run it inside a container / seccomp / nsjail. The interface
(`validate_with_tests`) is unchanged either way.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import tempfile

# Where temp run-files go (kept off the user's project tree).
_SCRATCH = os.environ.get(
    "SYNTH_SCRATCH",
    "/private/tmp/claude-501/-Users-nugrohodewantoro-Documents-Projects-Machine-Learning-nugie-coding-llm-agent-4/"
    "b8934146-cfe4-4cba-9691-cc200a8fb741/scratchpad",
)


@dataclasses.dataclass
class ExecResult:
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_python(code: str, timeout: float = 10.0) -> ExecResult:
    """Run `code` as a standalone Python script in a subprocess with a timeout."""
    os.makedirs(_SCRATCH, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=".py", dir=_SCRATCH, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        try:
            proc = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True,
                timeout=timeout,
                # Minimal env; do not inherit the parent's PYTHON* customization.
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            )
        except subprocess.TimeoutExpired as e:
            return ExecResult(False, -1, e.stdout or "", (e.stderr or "") + "\n[timeout]", timed_out=True)
        return ExecResult(proc.returncode == 0, proc.returncode, proc.stdout, proc.stderr)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def validate_with_tests(solution_code: str, test_code: str, timeout: float = 10.0) -> ExecResult:
    """True iff `solution_code` + `test_code` run to completion with exit code 0.

    The tests are expected to raise (e.g. via `assert`) on failure, which yields a
    non-zero exit code and thus `passed == False`.
    """
    combined = solution_code.rstrip() + "\n\n" + test_code.rstrip() + "\n"
    return run_python(combined, timeout=timeout)
