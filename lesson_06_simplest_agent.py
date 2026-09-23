import os
import subprocess
from operator import add
from pathlib import Path
from pydantic import BaseModel,Field,ValidationError
from dotenv import load_dotenv
from openai import OpenAI



class FileNameValidator(BaseModel):
    name: str = Field(description="Name of the text file in this project")


class FileSummaryValidator(BaseModel):
    """"This validator can validate the output of an agent, the JSON returned by the summary agent"""
    title: str = Field(description="The generated for the file")
    summary: str = Field(description="The summary of the file")


#starts a new agent and summarizes a file
def summarize_file(client:OpenAI,args:dict):
    #will receive a filename, will read the file, provide it to an agent and return a summary
    try:
        #receives and validates file name
        req = FileNameValidator.model_validate(args)
        filename = req.name
        #makes sure the file exists
        path = Path(filename).resolve()
        #aici punem limita ca agentul sa poata citi doar fisiere din folderul curent
        #only allows files from current folder
        path.relative_to(Path.cwd().resolve())
        #reads file
        text = path.read_text()


        #starts our agent
        response = client.chat.completions.create(
            model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
            messages = [
            {"role":"system","content":"Return ONLY JSON, that contains a title and a short summary. "
                                       "Summarise the file you received, and put it in the JSON response"},
                {"role":"user", "content":f"Please summarize this file for me: {text}"}
        ],
            response_format={
                "type":"json_schema",
                "strict" : True,
                "json_schema":{"name":"file_summary", "schema":FileSummaryValidator.model_json_schema()}
            },
        ).choices[0].message
        #first, validates llm agent output. sometimes it can be messed up
        validated_model_json = FileSummaryValidator.model_validate_json(response.content)
        #returns json with title and summary
        return validated_model_json.model_dump_json()

    except Exception as exc:
        return f"Error: {exc}"



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

            #without this response, the agent stays behind with 1 step
            response = complete(client, messages, tools)
            messages.append(response.model_dump(exclude_none=True))


        print("Agent said: ", response.content)
        print(f"Message count: {len(messages)}")


        print(response)


if __name__ == "__main__":
    #main()
    #print(run_tests())
    load_dotenv()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPEN_ROUTER_API_KEY"),
    )

    print(summarize_file(client,args={"name":"lesson_06_simplest_agent.py"}))