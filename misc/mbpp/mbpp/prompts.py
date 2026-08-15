import hashlib

from mbpp.dataset import MbppExample

SYSTEM_PROMPT = """
Solve the task in Python. Respond with exactly one fenced ```python code block and nothing else — no prose before or after it.

Rules:
- Write the solution as a flat sequence of top-level statements, not a function definition that gets called at the end. Each statement must be independently executable as it is written, top to bottom.
- Do not define a function or class, and do not use `if __name__ == "__main__":`.
- Do not use `input()`, read from stdin, `import` anything, or call `eval`/`exec`/`compile`.
- Do not write an infinite or unbounded loop.
- Structure the code in this order: (1) assign the given input to variables, (2) the statements that compute the answer, (3) exactly one final `print(repr(answer))`.

<bad_example>
```python
def solve(n):
    total = n
    return total

print(solve(12))
```
</bad_example>

<good_example>
```python
n = 12
total = n
answer = total
print(repr(answer))
```
</good_example>
""".strip()

HUMAN_PROMPT = """
Task: {text}

Example — input: {example_test_input} -> output: {example_test_output}

Solve it for this input: {test_input}
""".strip()

# Fingerprint of the template pair actually in force, independent of any
# per-case content. Recorded alongside timing results so a template edit
# between runs is visible as a version change instead of silently producing
# incomparable data.
PROMPT_VERSION = hashlib.sha256(
    f"{SYSTEM_PROMPT}\0{HUMAN_PROMPT}".encode()
).hexdigest()[:12]


def render(case: MbppExample) -> list[dict[str, str]]:
    """Render a single MBPP example as a list of chat messages.

    Args:
        case: The MBPP row to render.

    Returns:
        A `[system, user]` message pair ready to pass to a chat completion.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": HUMAN_PROMPT.format(
                text=case.text,
                example_test_input=case.example_test_input,
                example_test_output=case.example_test_output,
                test_input=case.test_input,
            ),
        },
    ]
