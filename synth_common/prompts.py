"""
Prompt templates shared by the synthesis pipelines.

Each template embeds an `[intent:...]` marker. Real teacher models simply ignore
the marker and follow the instruction; `MockTeacher` keys off it to return the
right *kind* of executable output, so the offline demo/tests exercise the full
question -> answer -> tests -> refine flow. Keep the human-readable instructions
faithful to the paper's described procedure (Sec. 4.1).
"""

from __future__ import annotations


def question_prompt(seed: str, language: str, difficulty: str, task_type: str) -> str:
    return (
        "[intent:question]\n"
        "You are designing a programming exercise. Using the reference material "
        "below only as inspiration, write a single, self-contained "
        f"{difficulty} {task_type} problem in {language}.\n\n"
        f"Reference:\n{seed}\n"
    )


def answer_prompt(question: str, language: str) -> str:
    return (
        "[intent:answer]\n"
        f"Solve the following {language} problem. Return only the solution code "
        "in a fenced code block.\n\n"
        f"Problem:\n{question}\n"
    )


def tests_prompt(question: str, solution: str, language: str) -> str:
    return (
        "[intent:tests]\n"
        f"Write {language} assertion-based test cases for the solution below. "
        "Return only a fenced code block that raises on failure.\n\n"
        f"Problem:\n{question}\n\nSolution:\n```python\n{solution}\n```\n"
    )


def refine_prompt(question: str, solution: str) -> str:
    return (
        "[intent:refine]\n"
        "Improve the following solution by adding clear comments and a short "
        "explanation, without changing its behavior. Return a fenced code block.\n\n"
        f"Problem:\n{question}\n\nSolution:\n```python\n{solution}\n```\n"
    )


def educational_prompt(snippet: str) -> str:
    return (
        "[intent:educational]\n"
        "Turn the following code snippet into a concise educational passage that "
        "teaches the key concept it demonstrates.\n\n"
        f"Snippet:\n```python\n{snippet}\n```\n"
    )


def package_qa_prompt(library: str, api: str, signature: str, doc: str) -> str:
    return (
        "[intent:answer]\n"
        f"Using the *current* API below for the `{library}` library, write a "
        "question a developer might ask about `" + api + "` and answer it with "
        "correct, up-to-date code in a fenced code block.\n\n"
        f"Signature: {signature}\nDocumentation:\n{doc}\n"
    )
