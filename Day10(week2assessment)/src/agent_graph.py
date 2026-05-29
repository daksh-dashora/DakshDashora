from typing import Annotated, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_agent_graph(mcp_tools):
    model = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)
    bound_model = model.bind_tools(mcp_tools)
    
    def call_model(state: AgentState):
        response = bound_model.invoke(state["messages"])
        return {"messages": [response]}
        
    def execute_tools(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        tool_outputs = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            matched_tool = next((t for t in mcp_tools if t.name == tool_name), None)
            if matched_tool:
                output = matched_tool.invoke(tool_args)
                tool_outputs.append({
                    "role": "tool",
                    "content": str(output),
                    "tool_call_id": tool_call["id"],
                    "name": tool_name
                })
        return {"messages": tool_outputs}

    def route_decision(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", execute_tools)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        route_decision,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()