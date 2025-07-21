from jitgen.executors.python import InProcPythonExecutor


import pytest


@pytest.fixture()
def python_executor():
    return InProcPythonExecutor()


@pytest.mark.asyncio
async def test_hello_world(python_executor: InProcPythonExecutor):
    source_code = "print('Hello, World!')"
    result = await python_executor.aexecute(source_code)

    assert result.success
    assert result.output == "Hello, World!\n"
    assert result.error is None
    assert not result.has_timed_out
    assert result.timeout_value is None


@pytest.mark.asyncio
async def test_syntax_error(python_executor: InProcPythonExecutor):
    source_code = "print('Hello, World!')\nprint('This will cause a syntax error'"
    with pytest.raises(SyntaxError):
        await python_executor.aexecute(source_code)


@pytest.mark.asyncio
async def test_semantic_error(python_executor: InProcPythonExecutor):
    source_code = "print('Hello, World!')\nprint(undefined_variable)"
    result = await python_executor.aexecute(source_code)

    assert not result.success
    assert result.output == "Hello, World!\n"
    assert result.error is not None
    assert "undefined_variable" in result.error
    assert not result.has_timed_out
    assert result.timeout_value is None


@pytest.mark.asyncio
async def test_locals(python_executor: InProcPythonExecutor):
    source_code = "a = 10"
    result = await python_executor.aexecute(source_code)

    assert result.success
    assert result.output == ""
    assert result.error is None
    assert not result.has_timed_out
    assert result.timeout_value is None

    assert python_executor._locals["a"] == 10


@pytest.mark.asyncio
async def test_sequence_of_statements(python_executor: InProcPythonExecutor):
    source_code = ["a = 10", "b = 20", "print(a + b)"]

    for stmt in source_code:
        result = await python_executor.aexecute(stmt)

        assert result.success
        assert not result.has_timed_out
        assert result.timeout_value is None

    assert python_executor._locals["a"] == 10
    assert python_executor._locals["b"] == 20

    assert result.output == "30\n"
