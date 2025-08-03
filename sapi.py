# main.py
from langgraph_flow import graph

if __name__ == "__main__":
    state = {
        "username": "jane_hr",
        "query": "tell me everything i need to here"
    }

    response = graph.invoke(state)
    print(response["answer"])
