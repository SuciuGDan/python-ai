import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


def run_tests() -> str:
    try:
        result = subprocess.run(
            ["uv", "run", "pytest"],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        return (result.stdout or "No output") + "\n" + (result.stderr or "")
    except Exception as e:
        return f"Error running tests: {e}"

def complete(client:OpenAI,messages:list[dict],tools:list[dict]):
    response = client.chat.completions.create(
        messages=messages,
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        tools=tools,
        tool_choice="auto",
    ).choices[0].message
    return response



def main():
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )

    # avem agentul - trebuie tools
    # lista de tools pentru agent
    tools = [{
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run uv run pytest",
        },
    }]


    messages = []
    messages.append({
        "role": "system",
        "content": "Use run_tests tool when asked to run any tests",
    })

    while True:
        inp = input("You: ")
        if inp == "quit":
            break
        messages.append({
            "role": "user",
            "content": inp,
        })


        #cerem agentului un mesaj

        response = complete(client,messages,tools)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                output = run_tests()
                messages.append({
                    "role":"tool",
                    "tool_call_id": tool_call.id,
                    "content": output,
                })

            response = complete(client, messages, tools)

        print("Agent said: ", response.content)
        print(f"Message count: {len(messages)}")


        print(response)


if __name__ == "__main__":
    main()
    #print(run_tests())