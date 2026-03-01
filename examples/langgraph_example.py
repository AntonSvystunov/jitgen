"""
LangGraph Integration Example with JitGen

This example demonstrates how to use JitGen with LangGraph to create
an agent that can execute Python code incrementally as it's generated.

The pipeline:
1. Agent receives a task and calls execute_task tool
2. Execute_task node sends task to LLM with system prompt
3. Code is streamed and executed incrementally using JitGen
4. Code blocks in LLM output are replaced with execution STDOUT
5. Result is returned as tool call output

Requirements:
- Google Generative AI API key set in GOOGLE_API_KEY environment variable
- langchain-google-genai package installed
"""

import asyncio
import os

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from dotenv import load_dotenv

from jitgen_langchain.python import create_python_jitgen_parser

_ = load_dotenv()


# System prompt from code_block.ipynb
SYSTEM_PROMPT = """
You are a English to Python translator. 
You will be provided with a task that you should code using Python to print the result of execution.
You have to generate a code snippet that covers user's request enclosed in "```python" code block.
Always add print() function call in python code snippet to print the result.

# Examples
Human: What is the sum of all natural numbers up to 10?
AI:
To calculate all natural numbers up to 10, I should use range() in the loop and accumulate the sum in the variable. Then I should use print() to print the result.
```python
total = 0
for number in range(10):
    total += number

print(total)
```
""".strip()


# Use MessagesState for proper message handling
# This automatically handles message merging and serialization
AgentState = MessagesState


def create_execute_task_tool():
    """Create the execute_task tool definition."""
    from langchain_core.tools import tool

    @tool
    def execute_task(task_description: str) -> str:
        """
        Execute a Python task. Provide a clear description of what needs to be done.

        Args:
            task_description: A description of the task to execute using Python code.

        Returns:
            The result of executing the task.
        """
        # This will be handled by the execute_task_node
        return task_description

    return execute_task


async def execute_task_node(state: AgentState) -> AgentState:
    """
    Execute task node that:
    1. Takes the tool call from agent
    2. Sends task to LLM with system prompt
    3. Streams and executes code using JitGen
    4. Replaces code blocks with execution output
    5. Returns tool call output
    """
    messages = state["messages"]

    # Find the last tool call (should be execute_task)
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        raise ValueError("Expected AIMessage with tool_calls")

    tool_call = last_message.tool_calls[0]
    if tool_call["name"] != "execute_task":
        raise ValueError(f"Expected execute_task tool call, got {tool_call['name']}")

    task_description = tool_call["args"]["task_description"]

    # Initialize LLM and JitGen parser
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    jitgen_parser = create_python_jitgen_parser()

    # Create prompt with system message
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), ("human", "{message}")]
    )

    # Build chain: prompt -> LLM -> JitGen parser
    chain = prompt | llm | jitgen_parser

    # Stream once through the chain and collect execution outputs
    execution_outputs = []
    async for output_chunk in chain.astream({"message": task_description}):
        if output_chunk and output_chunk.strip():
            execution_outputs.append(output_chunk)

    # Combine all execution outputs (STDOUT from executed code)
    execution_result = "".join(execution_outputs).strip()

    # Return just the execution stdout
    final_output = execution_result if execution_result else "No output produced."

    # Create tool message with the result
    tool_message = ToolMessage(
        content=final_output.strip(),
        tool_call_id=tool_call["id"],
    )

    # Return the tool message - LangGraph will append it to existing messages
    return {"messages": [tool_message]}


def agent_node(state: AgentState) -> AgentState:
    """
    Agent node that processes the task and decides to call execute_task tool.
    After receiving tool results, provides a final response.
    """
    # Initialize LLM with tools
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.2,
    )

    # Bind tools for tool calling
    execute_task_tool = create_execute_task_tool()
    llm_with_tools = llm.bind_tools([execute_task_tool])

    # Get agent response - MessagesState handles message merging automatically
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    Determine the next node based on the last message.
    """
    last_message = state["messages"][-1]

    # If the last message has tool calls, go to execute_task
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "execute_task"

    # Otherwise, end
    return END


def create_graph():
    """
    Create and compile the LangGraph graph.

    Graph structure:
    - agent: Processes user input and decides to call execute_task tool
    - execute_task: Executes Python code using JitGen and returns result
    - Flow: agent -> (conditional) -> execute_task -> agent -> END
    """
    # Create the graph with MessagesState
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("execute_task", execute_task_node)

    # Set entry point
    workflow.set_entry_point("agent")

    # Add conditional edges from agent
    # Routes to execute_task if tool calls are present, otherwise ends
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "execute_task": "execute_task",
            END: END,
        },
    )

    # After execute_task completes, return to agent to process tool result
    workflow.add_edge("execute_task", "agent")

    # Compile the graph (required before use)
    return workflow.compile()


async def main():
    """Main example function."""
    print("=" * 60)
    print("JitGen LangGraph Integration Example")
    print("=" * 60)

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("\n⚠ Error: GOOGLE_API_KEY environment variable not set")
        print("\nTo run this example:")
        print("1. Get a Google Generative AI API key")
        print("2. Set it: export GOOGLE_API_KEY='your-api-key'")
        print("3. Run this script again\n")
        return

    # Create the graph
    graph = create_graph()

    # Example task
    task = "How many vowels are in the sentence 'aaaabbbb bbabbab sbasb abyyu i'?"

    print(f"\nTask: {task}")
    print("\nExecuting with LangGraph and JitGen...")
    print("-" * 60)

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=task)],
    }

    try:
        # Stream the graph execution
        # Events contain updates from each node as they execute
        async for event in graph.astream(initial_state):
            # Process each node's output in the event
            for _node_name, node_state in event.items():
                if "messages" in node_state and node_state["messages"]:
                    last_message = node_state["messages"][-1]

                    # Display tool output
                    if isinstance(last_message, ToolMessage):
                        print("\n[Tool Output]")
                        content = last_message.content
                        print(content if isinstance(content, str) else str(content))

                    # Display agent messages
                    elif isinstance(last_message, AIMessage):
                        if last_message.tool_calls:
                            tool_call = last_message.tool_calls[0]
                            print(f"\n[Agent] Calling tool: {tool_call['name']}")
                            print(f"  Task: {tool_call['args']['task_description']}")
                        else:
                            content = last_message.content
                            print(
                                f"\n[Agent] {content if isinstance(content, str) else str(content)}"
                            )

        print("-" * 60)
        print("\n✓ Execution complete!")
        print("  Notice how code was executed incrementally using JitGen.")

    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
