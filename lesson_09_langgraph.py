from typing import TypedDict
from langgraph.graph import  StateGraph, START, END

# langgraph


#state:

class State(TypedDict):
    pass

BRANCH_CONDITION = False

def start_node(state:State):
    print("Node 1 takes an action and makes the first step. Start!")


#conditional function, picks the next node

def choose_branch(state:State):
    #poate fi o decizie deterministica, cu cod deterministic. sau poate fi o decizie luata de un model LLM

    #decizie deterministica
    if BRANCH_CONDITION:
        return "left_node"
    else:
        return "right_node"


def left_node(state:State):
    print("Node 2, we chose the left branch")

def right_node(state:State):
    print("Node 3, we chose the right branch")

def last_node(state:State):
    print("Last node byebye")

def build_graph():
    builder = StateGraph(State)

    builder.add_node("start_node", start_node)
    builder.add_node("left_node", left_node)
    builder.add_node("right_node", right_node)
    builder.add_node("last_node", last_node)

    builder.add_edge(START, "start_node")
    builder.add_conditional_edges("start_node",choose_branch)
    builder.add_edge("left_node","last_node")
    builder.add_edge("right_node","last_node")
    builder.add_edge("last_node", END)

    return builder.compile()


def main():
    graph = build_graph()

    print("====== run starts ======")
    graph.invoke({})
    print("====== run ends =======")



if __name__ == "__main__":
    main()
