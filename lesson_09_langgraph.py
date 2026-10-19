from pathlib import Path
from typing import TypedDict

from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from requests import request

PROJECT_ROOT = Path(__file__).resolve().parent

# state:
class State(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_file: str = Field(default="", description="Name of the log report file.")
    errors: list[str] = Field(default_factory=list)
    response: str = ""


def read_log_report(state: State):
    path = PROJECT_ROOT / state.log_file
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    errors = []
    for line in lines:
        if "ERROR" in line:
            errors.append(line)

    print("start> reading ", path.name)
    print("start> errors found: ", len(errors))

    # ii dam lui langgraph un state partial care va fi updatat in state-ul intreg.
    # reducer
    return {"errors": errors}

def check_for_errors(state: State):
    if len(state.errors) > 0:
        return "call_incident_api"
    return "report_all_ok"

def call_incident_api(state: State):
    print("action> calling our incident api")
    req = request("POST", "https://jsonplaceholder.typicode.com/todos/")
    c = len(state.errors)
    print(f"info> sent {c} errors to our incident API.")

    return {"response": str(req)}

def report_all_ok(state: State):
    print("action> Nothing to worry about, everything OK. No errors found in logs.")
    return {"response": "No errors found in logs."}


def build_graph():
    builder = StateGraph(State)

    builder.add_node("read_log_report", read_log_report)
    builder.add_node("call_incident_api", call_incident_api)
    builder.add_node("report_all_ok", report_all_ok)

    builder.add_edge(START, "read_log_report")
    builder.add_conditional_edges("read_log_report", check_for_errors)
    builder.add_edge("call_incident_api", END)
    builder.add_edge("report_all_ok", END)

    return builder.compile()


def main():
    graph = build_graph()
    initial_state = State(log_file="log_report.txt")

    print("======= run starts ========")
    result = graph.invoke(initial_state)
    print("errors found: ", result["errors"])
    print("response: ", result["response"])
    print("======= run ends   ========")



if __name__ == "__main__":
    main()