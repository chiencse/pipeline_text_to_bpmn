from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage
from pydantic import BaseModel
class WeatherResponse(BaseModel):
    conditions: str
checkpointer = InMemorySaver()
def get_weather(city: str) -> str:  
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

model = init_chat_model(
    "anthropic:claude-3-7-sonnet-latest",
    temperature=0,
    api_key="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"
)
model_google = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",   # hoặc "gemini-1.5-flash"
    temperature=0.0,
    api_key="AIzaSyABhnUhDweDUlT2zCYziR_F67XCYhQn7bU",
    response_format=WeatherResponse
)

agent = create_react_agent(
    model=model_google,
    tools=[get_weather],  
    prompt="You are a helpful assistant",
    # checkpointer=checkpointer
)



def print_conversation(result, title=""):
    print(f"\n======= {title} =======")
    total_input = total_output = 0

    for msg in result["messages"]:
        if isinstance(msg, HumanMessage):
            print(f"👤 User: {msg.content}")
        elif isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for call in msg.tool_calls:
                    print(f"🤖 Agent: [calls tool '{call['name']}' with args {call['args']}]")
            else:
                print(f"🤖 Agent: {msg.content}")

            # Lấy token usage (nếu có)
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                total_input += usage.get("input_tokens", 0)
                total_output += usage.get("output_tokens", 0)

        else:
            print(f"🛠️ Tool '{msg.name}': {msg.content}")

    print(f"🧮 Tokens used → Input: {total_input}, Output: {total_output}, Total: {total_input + total_output}")
    print("=" * 40)

# Gọi hai lần


# Run the agent
# config = {"configurable": {"thread_id": "1"}}
# sf_response = agent.invoke(
#     {"messages": [{"role": "user", "content": "what is the weather in sf"}]},
#     config  
# )

# ny_response = agent.invoke(
#     {"messages": [{"role": "user", "content": "what about new york?"}]},
#     config
# )
# print_conversation(sf_response, "Lần 1: hỏi SF")
# print_conversation(ny_response, "Lần 2: hỏi New York")

response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)

print(response)
