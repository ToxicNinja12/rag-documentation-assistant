from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    action: str


llm = ChatOpenAI(model="gpt-4o-mini")


def analyze_query(state: AgentState) -> AgentState:
    """This node handles queries"""
    system_message = (
        "You are a query analyzer, you will take the user's raw query and rewrite or expand the query to improve "
        "data retrieval quality from a small corpus of python technical documentation. You can add synonyms, or even "
        "send a prompt to clarify ambiguity. You should prepend your response with either 'rewrite:' or 'clarify:'. "
        "This output is not going back to the user it will go to another node for further processing."
    )
    prompt = [SystemMessage(content=system_message)] + state["messages"]
    raw_response = llm.invoke(prompt)
    response = str(raw_response.content).split(": ")

    state["action"] = response[0]
    state["messages"].append(AIMessage(content=response[1]))
    return state


def should_clarify(state: AgentState) -> str:
    """This function handles logic on whether the user should be prompted for clarification"""
    if state["action"] == 'clarify':
        print("AI thinks clarification is needed")
        return "dont_proceed"
    return "proceed"


graph = StateGraph(AgentState)
graph.add_node("query", analyze_query)

graph.set_entry_point("query")
graph.add_conditional_edges(
    "query",
    should_clarify,
    {
        "proceed": END,
        "dont_proceed": END
    }
)

agent = graph.compile()
