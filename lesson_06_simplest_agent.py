import os
from dotenv import load_dotenv
from openai import OpenAI


#avem agentul - trebuie tools


load_dotenv()

os.getenv("OPEN_ROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
)
# lista de tools pentru agent
tools = [{
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run uv run pytest",
    }
}]
messages=[]
messages.append({
    "role": "system",
    "content": "Use run_tests tool when asked to run any tests"
})
messages.append({
    "role":"user",
    "content":input("You: ")
})
# lista de tools pentru agent
tools = [{
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run uv run pytest",
    }
}]

response = client.chat.completions.create(
    messages=messages,
    model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
    tools=tools,
    tool_choice="auto"
).choices[0].message

print(response)
