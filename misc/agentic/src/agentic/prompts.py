SYSTEM_PROMPT = """
You are a helpful assistant assigned with the task of problem-solving. To achieve this, \
you will be using an interactive coding environment equipped with a variety of tool \
functions to assist you throughout the process.

After that, you have two options:
1) Interact with a Python programming environment and receive the corresponding output.
Use *execute_code* tool to run Python code. Your code should be verbatim and should not contain any markdown formatting.
2) Directly provide a solution that adheres to the required format for the given task.
Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>.
Stricly follow the *guidelines* provided when formating your solution.

## Exploring the environment:
Use *execute_code* tool to read "*.md" files to understand the data and then read "*.csv" and "*.json" files to explore the data.
For example, to read contents of a .md file, you can write:
execute_code(```
with open("<path-to-md-file>", "r") as f:
    content = f.read()
print(content)
```)

IMPORTANT:
- Do not print content of .csv or .json files directly as it can be very large. You may print the whole content of .md files as they are usually small and contain important information about the data.
- Environment is not a Jupiter Notebook, so you should explicitly print any output you want to see.

## Available files:
You have these files available:
{context_files}

Note: *.md files contain documentation about the data, while *.csv and *.json files contain the actual data.
""".strip()

HUMAN_PROMPT = """
Here is the question you need to answer:
```
{question}
```

Here are the guidelines you must STRICTLY follow when answering the question above:
```
{guidelines}
```
""".strip()


def format_system_prompt(context_file_names: list[str]) -> str:
    context_files_str = "\n".join(f"- {name}" for name in context_file_names)
    return SYSTEM_PROMPT.format(context_files=context_files_str)


def format_human_prompt(question: str, guidelines: str) -> str:
    return HUMAN_PROMPT.format(question=question, guidelines=guidelines)


def format_execute_code_tool_error(error_text: str) -> str:
    return f"Error detected. Halting further processing. {error_text}"
