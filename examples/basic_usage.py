"""
Basic JitGen Usage Example

This example demonstrates the core functionality of JitGen:
- Incremental code execution as fragments are generated
- Early output production
- State persistence across statements
"""

import asyncio
from jitgen.prebuilt.python import create_python_jitgen


async def simulate_code_generation():
    """
    Simulates an LLM generating code fragments incrementally.
    In a real scenario, these would come from an LLM stream.
    """
    fragments = [
        "# Calculate fibonacci numbers\n",
        "def fib(n):\n",
        "    if n <= 1:\n",
        "        return n\n",
        "    return fib(n-1) + fib(n-2)\n",
        "\n",
        "print('Fibonacci sequence:')\n",
        "for i in range(10):\n",
        "    print(f'fib({i}) = {fib(i)}')\n",
    ]
    
    for fragment in fragments:
        yield fragment
        # Simulate network delay
        await asyncio.sleep(0.1)


async def main():
    """Main example demonstrating JitGen incremental execution."""
    print("=" * 60)
    print("JitGen Basic Usage Example")
    print("=" * 60)
    print("\nSimulating incremental code generation...")
    print("Output will appear as complete statements are executed:\n")
    
    jitgen = create_python_jitgen()
    
    print("-" * 60)
    async for output in jitgen.arun_from_stream(simulate_code_generation()):
        print(output, end="", flush=True)
    print("-" * 60)
    
    print("\n✓ Execution complete! Notice how output appeared incrementally.")
    print("  Each complete statement was executed as soon as it was detected.")


if __name__ == "__main__":
    asyncio.run(main())

