import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from agent_graph import create_agent_graph


def convert_mcp_to_langchain(mcp_tools, session, loop):
    langchain_tools = []

    for t in mcp_tools:
        tool_name = t.name
        tool_desc = t.description

        if "retrieve" in tool_name:
            def make_retrieve(name, desc):
                @tool
                def custom_tool(query: str) -> str:
                    """Query the vector database to retrieve relevant document snippets."""
                    future = asyncio.run_coroutine_threadsafe(
                        session.call_tool(name, arguments={"query": query}), loop
                    )
                    result = future.result(timeout=30)
                    return str(result)
                custom_tool.name = name
                custom_tool.description = desc
                return custom_tool
            langchain_tools.append(make_retrieve(tool_name, tool_desc))

        else:
            def make_calc(name, desc):
                @tool
                def calc_tool(expression: str) -> str:
                    """Evaluate a safe mathematical expression."""
                    future = asyncio.run_coroutine_threadsafe(
                        session.call_tool(name, arguments={"expression": expression}), loop
                    )
                    result = future.result(timeout=30)
                    return str(result)
                calc_tool.name = name
                calc_tool.description = desc
                return calc_tool
            langchain_tools.append(make_calc(tool_name, tool_desc))

    return langchain_tools


async def chat_loop():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "mcp_server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=os.environ.copy()
    )

    print("Connecting to MCP Server and loading tools...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            loop = asyncio.get_event_loop()
            mcp_features = await session.list_tools()
            compiled_tools = convert_mcp_to_langchain(mcp_features.tools, session, loop)

            graph = create_agent_graph(compiled_tools)
            print("\nInitialization Complete! Type 'exit' or 'quit' to end.")
            print("-" * 50)

            while True:
                try:
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("\nYou: ").strip()
                    )
                    if not user_input:
                        continue
                    if user_input.lower() in ["exit", "quit"]:
                        break

                    state = {"messages": [HumanMessage(content=user_input)]}
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: graph.invoke(state)
                    )

                    print(f"\nAgent: {result['messages'][-1].content}")

                except (KeyboardInterrupt, EOFError):
                    break
                except Exception as e:
                    print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        sys.exit(1)
    asyncio.run(chat_loop())