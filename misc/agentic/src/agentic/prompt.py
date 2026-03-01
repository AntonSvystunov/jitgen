from typing import TypedDict


class TaskInput(TypedDict):
    task: str
    example_test_input: str
    example_test_output: str
    test_input: str
    test_output: str


CODEACT_SYSTEM_PROMPT = """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
The assistant can interact with an interactive Python (Jupyter Notebook) environment and receive the corresponding output when needed. The code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>.
The assistant should attempt fewer things at a time instead of putting too much code in one <execute> block. The assistant should stop <execute> and provide an answer when they have already obtained the answer from the execution result. Whenever possible, execute the code for the user using <execute> instead of providing it.
The assistant's response should be concise, but do express their thoughts.

IMPORTANT CONSTRAINTS for code inside <execute> blocks:
- Write straight-line code. Do NOT define functions — just compute and print directly.
- DO NOT use input() or any interactive blocking call.
- Do not read stdin.
- No eval/exec/compile.
- No infinite loops; every loop must have a clear termination condition.
- Script must run top-to-bottom without errors.
- Use print(...) to output the result.
""".strip()


USER_PROMPT_TEMPLATE = """Solve the following programming task by writing and executing Python code.

TASK:
{task}

EXAMPLE (for reference only — this is a DIFFERENT test case, not the one you must solve):
  Input:  {example_test_input}
  Output: {example_test_output}

ACTUAL TEST INPUT (solve for THIS input):
{test_input}

Write straight-line Python code (no function definitions) using <execute> tags that:
1) Defines local variables from the ACTUAL TEST INPUT above
2) Computes the result directly
3) Prints the result using print(...)

IMPORTANT: Do NOT try to reproduce the example output. Compute the correct answer
for the ACTUAL TEST INPUT. The example only shows the expected output *format*.

Remember: no input(), stdout only via print(...).
Use <execute> tags to run code, for example:
<execute>
print("Hello World!")
</execute>
"""


def format_user_message(task_input: TaskInput) -> str:
    return USER_PROMPT_TEMPLATE.format(
        task=task_input["task"].replace("function", "Python code"),
        example_test_input=task_input.get("example_test_input", ""),
        example_test_output=task_input.get("example_test_output", ""),
        test_input=task_input.get("test_input", ""),
        test_output=task_input.get("test_output", ""),
    )
