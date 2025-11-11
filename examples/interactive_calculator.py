"""
Interactive Calculator Example

This example demonstrates JitGen with an interactive calculator scenario,
showing how code execution happens incrementally as code is "typed".
"""

import asyncio
from jitgen.prebuilt.python import create_python_jitgen


async def simulate_typing_calculator_code():
    """
    Simulates incremental code generation (like an LLM would produce).
    This demonstrates how JitGen handles incomplete input and executes
    complete statements as soon as they're detected.
    """
    # Simulate incremental code generation (fragment by fragment)
    code_parts = [
        "# Simple Calculator\n",
        "a = 15\n",
        "b = 27\n",
        "\n",
        "print(f'{a} + {b} = {a + b}')\n",
        "print(f'{a} * {b} = {a * b}')\n",
        "print(f'{a} / {b} = {a / b:.2f}')\n",
        "\n",
        "# Calculate power\n",
        "result = a ** 2\n",
        "print(f'{a} squared = {result}')\n",
    ]
    
    for part in code_parts:
        yield part
        await asyncio.sleep(0.1)  # Small delay to show incremental processing


async def main():
    """Interactive calculator example."""
    print("=" * 60)
    print("JitGen Interactive Calculator Example")
    print("=" * 60)
    print("\nSimulating incremental code generation...")
    print("JitGen will execute complete statements as soon as they're detected.\n")
    
    jitgen = create_python_jitgen()
    
    print("-" * 60)
    print("Output:")
    print("-" * 60)
    
    async for output in jitgen.arun_from_stream(simulate_typing_calculator_code()):
        print(output, end="", flush=True)
    
    print("-" * 60)
    print("\n✓ Example complete!")
    print("\nKey observations:")
    print("  • Code was executed incrementally as complete statements were detected")
    print("  • Variables persisted across statements (REPL-style behavior)")
    print("  • Output appeared immediately when statements completed")
    print("  • Each fragment was processed as it arrived")


if __name__ == "__main__":
    asyncio.run(main())

