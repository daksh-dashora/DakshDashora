from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from state import AgentState
from nodes import agent_node, tool_node, should_continue

load_dotenv()

graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end" : END
    }
)
graph.add_edge("tools", "agent")

graph = graph.compile()



with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# initial_state = {
#     "messages": [
#         HumanMessage(content="What is 457 multiplied by 83")
#     ]
# }

# result = graph.invoke(initial_state)
# print(result["messages"][-1].content)