import json
import os
from http.client import responses
from pathlib import Path
from typing import TypedDict, Annotated, Literal

from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from requests import request

from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_REPORT_FILE = "log_report.txt"
MAX_AGENT_STEPS = 6

SYSTEM_PROMPT = """
You are an incident report assistant. You will receive a list of errors from a log report.
You can use the create_incident tool once, to report them. IMPORTANT: USE THIS TOOL ONLY ONCE.
Try not to hallucinate or generate errors from nowhere. Only report errors that you have actually received from the log report.
After the tool returns, give a short final report.
"""

load_dotenv()
CLIENT = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API_KEY")
)

# reducer:
def append_messages(old_state_messages: list, new_messages: list) -> list:
    return old_state_messages + new_messages

# state:
class State(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_file: str = Field(default="", description="Name of the log report file.")
    # used for relevant context of the errors, for the llm.
    lines: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    # here we link the append_messages reducer to our state.messages property
    # append_messages is a reducer.
    messages: Annotated[list[dict], append_messages] = Field(default_factory=list)
    # start at 0, end at MAX_AGENT_STEPS
    agent_steps: int = 0
    # final text answer written by the agent, or by the all-ok branch.
    response: str = ""


class IncidentReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="A short title for the incident.")
    summary: str = Field(description="A short summary about what went wrong. Use 2-4 sentences max.")
    severity_level: Literal["low", "medium", "high"] = Field(description="This describes how serious the incident is.")


def create_incident(args: dict):
    try:
        report_request = IncidentReport.model_validate(args)
    except ValidationError as ex:
        return f"error: invalid arguments: {ex}"

    # SIMULATE a POST request.
    print("tool> POST req at https://sentry.io/incidents")
    print(f"tool> tool title: {report_request.title}")
    print(f"tool> SEVERITY: {report_request.severity_level}")

    response = {
        "status": "accepted",
        "incident_id": "INC-003",
        "message": "incident received!",
        "code": "202"
    }
    print(f"tool> response: {response}")
    return json.dumps(response)


TOOLS = [{
    "type": "function",
    "function": {
        "name": "create_incident",
        "description": "Create an incident for the errors found in the log report. Call this tool only once!",
        "parameters": IncidentReport.model_json_schema()
    }
}]


def read_log_report(state: State):
    path = PROJECT_ROOT / state.log_file
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    errors = []
    for line in lines:
        if "ERROR" in line:
            errors.append(line)

    error_text = "\n".join(errors)

    # the agent starts with the system prompt and the errors found in the log.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"These errors were found in {path.name}:\n\n{error_text}"},
    ]

    print("start> reading ", path.name)
    print("start> errors found: ", len(errors))

    # ii dam lui langgraph un state partial care va fi updatat in state-ul intreg.
    # reducer
    return {"lines": lines, "errors": errors, "messages": messages}

def check_for_errors(state: State):
    if len(state.errors) > 0:
        return "call_agent"
    return "report_all_ok"

def call_agent(state: State):
    message = CLIENT.chat.completions.create(
        model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
        messages=state.messages,
        tools=TOOLS,
        tool_choice="auto"
    ).choices[0].message

    # se face un pas fiindca intrebam agentul ce face. incrementam cu 1.
    steps = state.agent_steps + 1

    # [message.model_dump(exclude_none=True)] -> new_messages in the reducer.
    # reducer-ul se apeleaza automat cand un nod vrea sa faca un update pentru acea proprietate.
    # langgraph se ocupa automat de apelarea acelui reducer.
    update = {
        "messages": [message.model_dump(exclude_none=True)],
        "agent_steps": steps
    }

    # no tool calls means this is the final answer, keep the text for main().
    if not message.tool_calls:
        update["response"] = message.content or ""

    return update

def agent_next_steps(state: State):
    if state.agent_steps >= MAX_AGENT_STEPS:
        print("agent> stopping: step limit reached.")
        return END

    last_message = state.messages[-1]
    if last_message.get("tool_calls"):
        return "call_tools"

    # GOTO END if there are no tool calls.
    return END

def call_tools(state: State):
    last_message = state.messages[-1]
    tool_messages = []

    for tool_call in last_message.get("tool_calls"):
        tool_name = tool_call["function"]["name"]
        arguments = json.loads(tool_call["function"]["arguments"])

        if tool_name == "create_incident":
            # aici apelam tool-ul
            output = create_incident(arguments)
        else:
            output = f"error: unknown tool name: {tool_name}"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": output
        })

    # the reducer appends these to state.messages, so the state must be updated
    # with a dict keyed by the field name, not with a bare list.
    return {"messages": tool_messages}

def report_all_ok(state: State):
    print("action> Nothing to worry about, everything OK. No errors found in logs.")
    return {"response": "No errors found in logs."}


def build_graph():
    builder = StateGraph(State)

    builder.add_node("read_log_report", read_log_report)
    builder.add_node("call_agent", call_agent)
    builder.add_node("call_tools", call_tools)
    builder.add_node("report_all_ok", report_all_ok)

    builder.add_edge(START, "read_log_report")
    builder.add_conditional_edges("read_log_report", check_for_errors)
    builder.add_conditional_edges("call_agent", agent_next_steps)
    # de la call_tools trebuie sa avem legatura catre call agent.
    builder.add_edge("call_tools", "call_agent")
    builder.add_edge("report_all_ok", END)

    return builder.compile()


def main():
    graph = build_graph()
    initial_state = State(log_file=LOG_REPORT_FILE)

    print("======= run starts ========")
    result = graph.invoke(initial_state)
    print("errors found: ", result["errors"])
    print("response: ", result["response"])
    print(result["messages"])
    print("======= run ends   ========")



if __name__ == "__main__":
    main()