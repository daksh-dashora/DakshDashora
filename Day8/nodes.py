from dotenv import load_dotenv
from state import AgentState

from langchain_core.messages import AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import calculator
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model = "models/gemini-2.5-flash",
    temperature = 0
)

llm_with_tools = llm.bind_tools([calculator])

def agent_node(state: AgentState):

    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages" : messages+[response]
    }


tools = {
    "calculator" : calculator
}


def tool_node(state: AgentState):

    last_message = state["messages"][-1]

    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        result = tools[tool_name].invoke(tool_args)

        tool_message = ToolMessage(
            content=result,
            tool_call_id = tool_call["id"]
   
        )

        tool_messages.append(tool_message)

    return {
        "messages" : state["messages"] + tool_messages
    }



def should_continue(state: AgentState) -> str:

    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "tools"
    
    return "end"
