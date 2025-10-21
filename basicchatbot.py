from typing import Annotated
import os
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition


os.environ["GOOGLE_API_KEY"] = "AIzaSyABhnUhDweDUlT2zCYziR_F67XCYhQn7bU"

llm = init_chat_model("google_genai:gemini-2.0-flash")

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    # print("Requesting human assistance for query:", query)
    human_response = interrupt({"query": query})
    return human_response["data"]
# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.

memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}

tools = [human_assistance]
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)
from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
#     png_bytes = graph.get_graph().draw_mermaid_png()
#     with open("graph.png", "wb") as f:
#         f.write(png_bytes)
#     print("Saved graph.png")
# except Exception:
#     pass

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
# config = {"configurable": {"thread_id": "1"}}

# events = graph.stream(
#     {"messages": [{"role": "user", "content": user_input}]},
#     config,
#     stream_mode="values",
# )
# for event in events:
#     if "messages" in event:
#         event["messages"][-1].pretty_print()


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        # stream_graph_updates(user_input)
        events = graph.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config,
            stream_mode="values",
        )
        for event in events:
            event["messages"][-1].pretty_print()
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break