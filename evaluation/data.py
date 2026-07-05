"""
Problem loaders + a tiny bundled sample so the harness runs offline.

`load_humaneval` / `load_mbpp` read the official JSONL formats:
  * HumanEval: fields `task_id`, `prompt`, `entry_point`, `test`, `canonical_solution`
    (openai/human-eval `HumanEval.jsonl`).
  * MBPP: fields `task_id`, `text`/`prompt`, `test_list`, `test_setup_code`, `code`
    (google-research `mbpp.jsonl`; the sanitized split uses `prompt` instead of `text`).

`sample_humaneval()` / `sample_mbpp()` return a handful of self-contained problems
(canonical solutions pass their tests) so the oracle self-test scores 1.0.
"""

from __future__ import annotations

import json

from .problems import HumanEvalProblem, MBPPProblem


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_humaneval(path: str) -> list[HumanEvalProblem]:
    out = []
    for r in _read_jsonl(path):
        out.append(HumanEvalProblem(
            task_id=r["task_id"], prompt=r["prompt"], entry_point=r["entry_point"],
            test=r["test"], canonical_solution=r.get("canonical_solution", ""),
        ))
    return out


def load_mbpp(path: str) -> list[MBPPProblem]:
    out = []
    for r in _read_jsonl(path):
        out.append(MBPPProblem(
            task_id=str(r.get("task_id", r.get("id", ""))),
            text=r.get("text") or r.get("prompt") or "",
            test_list=list(r.get("test_list", [])),
            test_setup_code=r.get("test_setup_code", "") or "",
            code=r.get("code", ""),
        ))
    return out


# --------------------------------------------------------------------------- #
#  Bundled offline sample (canonical solutions all pass -> oracle Pass@1 == 1.0)
# --------------------------------------------------------------------------- #
def sample_humaneval() -> list[HumanEvalProblem]:
    return [
        HumanEvalProblem(
            task_id="sample/add",
            prompt=(
                "def add(a: int, b: int) -> int:\n"
                '    """Return the sum of a and b.\n'
                "    >>> add(2, 3)\n    5\n    \"\"\"\n"
            ),
            entry_point="add",
            canonical_solution="    return a + b\n",
            test=(
                "def check(candidate):\n"
                "    assert candidate(2, 3) == 5\n"
                "    assert candidate(-1, 1) == 0\n"
                "    assert candidate(0, 0) == 0\n"
            ),
        ),
        HumanEvalProblem(
            task_id="sample/is_even",
            prompt=(
                "def is_even(n: int) -> bool:\n"
                '    """Return True if n is even.\n    >>> is_even(4)\n    True\n    """\n'
            ),
            entry_point="is_even",
            canonical_solution="    return n % 2 == 0\n",
            test=(
                "def check(candidate):\n"
                "    assert candidate(4) is True\n"
                "    assert candidate(7) is False\n"
                "    assert candidate(0) is True\n"
            ),
        ),
        HumanEvalProblem(
            task_id="sample/reverse",
            prompt=(
                "def reverse_string(s: str) -> str:\n"
                '    """Return s reversed.\n    >>> reverse_string(\'abc\')\n    \'cba\'\n    """\n'
            ),
            entry_point="reverse_string",
            canonical_solution="    return s[::-1]\n",
            test=(
                "def check(candidate):\n"
                "    assert candidate('abc') == 'cba'\n"
                "    assert candidate('') == ''\n"
                "    assert candidate('a') == 'a'\n"
            ),
        ),
    ]


def sample_mbpp() -> list[MBPPProblem]:
    return [
        MBPPProblem(
            task_id="mbpp/factorial",
            text="Write a function to compute the factorial of a non-negative integer.",
            code="def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
            test_list=[
                "assert factorial(0) == 1",
                "assert factorial(5) == 120",
                "assert factorial(3) == 6",
            ],
        ),
        MBPPProblem(
            task_id="mbpp/palindrome",
            text="Write a function to check whether a string is a palindrome.",
            code="def is_palindrome(s):\n    return s == s[::-1]\n",
            test_list=[
                "assert is_palindrome('racecar') == True",
                "assert is_palindrome('hello') == False",
            ],
        ),
        MBPPProblem(
            task_id="mbpp/max_two",
            text="Write a function that returns the maximum of two numbers.",
            code="def max_of_two(a, b):\n    return a if a > b else b\n",
            test_list=[
                "assert max_of_two(3, 7) == 7",
                "assert max_of_two(10, 2) == 10",
            ],
        ),
    ]
