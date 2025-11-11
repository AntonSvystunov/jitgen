"""
Error Handling Example

This example demonstrates how JitGen handles errors:
- Syntax errors halt generation immediately
- Runtime errors halt generation immediately
- Incomplete code is buffered until completion
"""

import asyncio
from jitgen.prebuilt.python import create_python_jitgen


async def simulate_code_with_error():
    """Simulates code generation that contains an error."""
    fragments = [
        "print('This will execute')\n",
        "x = 10\n",
        "print(f'x = {x}')\n",
        "print('This will also execute')\n",
        "y = undefined_variable  # This will cause a runtime error\n",
        "print('This will never execute')\n",
    ]
    
    for fragment in fragments:
        yield fragment
        await asyncio.sleep(0.1)


async def simulate_syntax_error():
    """Simulates code generation with a syntax error."""
    fragments = [
        "print('Valid code')\n",
        "x = 10\n",
        "print(x +  # Syntax error: incomplete expression\n",
        "print('This will never execute')\n",
    ]
    
    for fragment in fragments:
        yield fragment
        await asyncio.sleep(0.1)


async def simulate_incomplete_code():
    """Simulates incomplete code that gets completed later."""
    fragments = [
        "def greet(name):\n",
        "    return f'Hello, {name}!'\n",
        "\n",
        "print(greet('World'))\n",
    ]
    
    for fragment in fragments:
        yield fragment
        await asyncio.sleep(0.1)


async def main():
    """Demonstrate error handling scenarios."""
    print("=" * 60)
    print("JitGen Error Handling Example")
    print("=" * 60)
    
    jitgen = create_python_jitgen()
    
    # Example 1: Runtime error
    print("\n1. Runtime Error Example")
    print("-" * 60)
    print("Code with a runtime error (undefined variable):")
    print("-" * 60)
    try:
        async for output in jitgen.arun_from_stream(simulate_code_with_error()):
            print(output, end="", flush=True)
    except ValueError as e:
        print(f"\n✗ Error caught: {e}")
        print("  Generation halted immediately when error was detected.")
    
    # Reset for next example
    jitgen = create_python_jitgen()
    
    # Example 2: Incomplete code (should complete successfully)
    print("\n\n2. Incomplete Code Example")
    print("-" * 60)
    print("Code that starts incomplete but gets completed:")
    print("-" * 60)
    async for output in jitgen.arun_from_stream(simulate_incomplete_code()):
        print(output, end="", flush=True)
    print("\n✓ Code completed successfully!")
    print("  Incomplete statements were buffered until completion.")
    
    # Example 3: Syntax error
    print("\n\n3. Syntax Error Example")
    print("-" * 60)
    print("Code with a syntax error (incomplete expression):")
    print("-" * 60)
    jitgen = create_python_jitgen()
    try:
        async for output in jitgen.arun_from_stream(simulate_syntax_error()):
            print(output, end="", flush=True)
    except ValueError as e:
        print(f"\n✗ Syntax error caught: {e}")
        print("  Generation halted when syntax error was detected.")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("  • Runtime errors halt generation immediately")
    print("  • Syntax errors halt generation immediately")
    print("  • Incomplete statements are buffered until completion")
    print("  • This prevents wasted computation on invalid code")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

