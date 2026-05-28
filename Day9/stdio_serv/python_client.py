from os.path import join, dirname, abspath
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters, client
import asyncio

mcp_server_script = join(dirname(abspath(__file__)) , "server1.py")

server_params = StdioServerParameters(
    command = "python",
    args = [str(mcp_server_script)],
    env = {}
)


async def main():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools() 
            print("Available tools" , tools)

            result = await session.call_tool("fetch")
            print("\n"*2, result)

if __name__ == "__main__":
    asyncio.run(main())