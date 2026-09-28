# Saves the live PARCS MCP server's tool list (names, descriptions, input schemas)
# as the unit tests' fixture, so they check the schemas the live server serves.
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from jitgen_openai import McpToolBridge

OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "unit_tests"
    / "fixtures"
    / "parcs_tools.json"
)


async def main(url: str) -> None:
    async with McpToolBridge.sse(url) as bridge:
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool in bridge.tools.values()
        ]
    OUT.write_text(
        json.dumps(tools, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved {len(tools)} tools to {OUT}")


if __name__ == "__main__":
    load_dotenv()
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PARCS_SERVER_URL")
    if not url:
        sys.exit("pass the PARCS MCP SSE endpoint, or set PARCS_SERVER_URL in .env")
    asyncio.run(main(url))
