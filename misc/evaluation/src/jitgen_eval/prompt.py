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

TOP-LEVEL CODE RULE (most important):
Your script is executed statement by statement as you write it, so the work must happen
at the top level of the module - not inside a function body.
- Do NOT wrap the solution in a function and then call it at the end.
- Write the assignments, loops and conditionals that solve the task directly at the top
  level, operating on the variables inferred from the test input.
- Define a function ONLY when the same logic is genuinely reused: it is called from more
  than one place, or it has to be recursive. A function that is called exactly once is
  not reuse - inline its body instead.
- The task statement may ask for a "function". Ignore that framing: produce a top-level
  script that prints the answer for the given test input.

<bad_example reason="solution hidden in a function, nothing runs until the final call">
def f1(x, y):
    total = 0
    while x < y:
        total += x
        x += 1
    return total

result = f1(2, 6)
print(result)
</bad_example>

<good_example reason="same solution, executes as it streams">
# Variables from the test input
x = 2
y = 6

# Main logic
total = 0
while x < y:
    total += x
    x += 1

# Answer
print(total)
</good_example>

Non-negotiable constraints:
- DO NOT use input() (or any interactive blocking call).
- Do not read stdin.
- No imports / no external libraries.
- No eval/exec/compile.
- No infinite loops; every loop must have a clear termination condition.
- Script must run top-to-bottom without errors.
- No function may wrap the main logic (see TOP-LEVEL CODE RULE).

*Structure* your code in following order:
1. Variable definitions inferred from the test input
2. Main logic to solve the task, written at the top level
3. Final print statement with the answer

Only if a genuinely reusable or recursive helper is unavoidable, define it above step 1.

Separate your code into logical section with comments to match the above structure.

Internal self-check (do this silently before finalizing):
- [ ] All required values are defined as local variables from the test input
- [ ] The main logic runs at the top level; no def wraps it
- [ ] Every function I defined (if any) is called from more than one place, or is recursive
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
2) Solves the task using top-level statements (no function wrapping the solution)
3) Prints the result in the specified format

Remember: no input(), no imports, stdout only via print(...), main logic at the top level.

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
