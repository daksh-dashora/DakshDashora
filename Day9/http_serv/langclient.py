from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

load_dotenv()


async def main():

    client  = MultiServerMCPClient(
        {
        "data_fetch" :{
            "transport": "streamable-http",
            "url" : os.getenv("SERVER_URL"),
            
        },
        
       
        
    }
    )

    tools = await client.get_tools()
    # print([tool.name for tool in tools])

    agent = create_agent(
        
        model = ChatGoogleGenerativeAI(
            model = "models/gemini-2.5-flash"
        ),
        tools = tools
    )

    response = await agent.ainvoke(
        {
            "messages" :[
                {
                    "role" : "user",
                    "content" : "Calculate 4*12"
                }
            ]
        }
    )

    print(response["messages"][-1].content)

    
asyncio.run(main())