from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """
You generate Python scripts to be executed line-by-line.

IMPORTANT OUTPUT RULE:
- Your final answer must be ONLY Python code wrapped in a ```python code block.
- Format: ```python\n<code>\n```

Your job:
- Given a problem statement and one test case, write a script that computes the answer for that test case.
- Infer variables from the test case and define them as local variables near the top.
- Print the result using print(...) exactly as required.

Non-negotiable constraints:
- DO NOT use input() (or any interactive blocking call).
- Do not read stdin.
- No imports / no external libraries.
- No eval/exec/compile.
- No infinite loops; every loop must have a clear termination condition.
- Script must run top-to-bottom without errors.
- Define functions only if absolutely necessary, but the main logic should be in the global scope.

*Structure* your code in following order:
1. Helper functions (if needed)
2. Variable definitions inferred from the test input
3. Main logic to solve the task
4. Final print statement with the answer

Separate your code into logical section with comments to match the above structure.

Internal self-check (do this silently before finalizing):
- [ ] All required values are defined as local variables from the test input
- [ ] No input() usage
- [ ] No imports
- [ ] Output matches the required format exactly
- [ ] Script runs top-to-bottom
- [ ] All utility code is defined before it's used
- [ ] Code structure follows the specified order

If ambiguous, choose the simplest interpretation consistent with the test case and required output format.
If impossible, print a clear error message.
""".strip()

USER_PROMPT = """
TASK:
{task}

TASK INPUT (infer local variables from this):
{test_input}

EXPECTED OUTPUT FORMAT EXAMPLE:
{example_test_output}

Write a Python script that:
1) Defines local variables from the test input
2) Solves the task
3) Prints the result in the specified format

Remember: no input(), no imports, stdout only via print(...).

CRITICAL: Wrap your entire code in a ```python code block.
""".strip()


class TaskInput(TypedDict):
    task: str
    example_test_input: str
    example_test_output: str
    test_input: str
    test_output: str  # Never shown to the model, but used for evaluation


task_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "user",
            USER_PROMPT,
        ),
    ]
)
