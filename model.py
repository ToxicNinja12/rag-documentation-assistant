from typing import TypedDict, List, Union, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from ingest import init_ingestion
from dotenv import load_dotenv

load_dotenv()

vector_store = init_ingestion()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    retrieved: Any
    action: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


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
        print("[ALERT] AI thinks clarification is needed")
        return "dont_proceed"
    return "proceed"


def retrieve_data(state: AgentState) -> AgentState:
    """This node retrieves data from the vector store"""
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    retrieved = retriever.invoke(state["messages"][-1].content)
    state["retrieved"] = retrieved
    return state


def grade_data(state: AgentState) -> AgentState:
    """This node grades the retrieved data"""
    system_message = (
        "You are a data evaluator, you will evaluate whether the retrieved data from a small corpus of python "
        "technical documentation is exactly what the user was asking. If it is not relevant, you will"
        " be honest and admit you don't know what the answer to the query should be. If the query is relevant just reply"
        " 'relevant: '. If it is not relevant reply with the apology and prepend the apology with 'irrelevant:'."
    )
    prompt = [SystemMessage(content=system_message), str(state["retrieved"]), HumanMessage(content=state["messages"][-1].content)]
    raw_response = llm.invoke(prompt)
    response = str(raw_response.content).split(": ")

    state["action"] = response[0]
    if state["action"] == "irrelevant":
        state["messages"].append(AIMessage(content=response[1]))
    return state


def should_apologize(state: AgentState) -> str:
    """This function handles logic on whether the user should be given an apology for the inability to find the
    relevant information"""
    if state["action"] == 'irrelevant':
        print("[ALERT] AI thinks data retrieved is irrelevant")
        return "dont_proceed"
    return "proceed"


def generate_output(state: AgentState) -> AgentState:
    """This node generates the final result"""
    system_message = (
        "You have retrieved the result of the query posed by the user from a small corpus of python technical "
        "documentation. Generate a clear, accurate and short answer grounded in retrieved context. Include citations or "
        "references to the source document in the response. Make sure to generate the output without any formatting or newline characters."
        "You're absolutely not allowed to use any '\\n' or '```'. The answer must not be more than two sentences."
    )
    prompt = [SystemMessage(content=system_message), str(state["retrieved"]),
              HumanMessage(content=state["messages"][-1].content)]
    response = llm.invoke(prompt)
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("query", analyze_query)
graph.add_node("retrieve", retrieve_data)
graph.add_node("data_grading", grade_data)
graph.add_node("generation", generate_output)

graph.set_entry_point("query")
graph.add_conditional_edges(
    "query",
    should_clarify,
    {
        "proceed": "retrieve",
        "dont_proceed": END
    }
)
graph.add_edge("retrieve", "data_grading")
graph.add_conditional_edges(
    "data_grading",
    should_apologize,
    {
        "proceed": "generation",
        "dont_proceed": END
    }
)
graph.set_finish_point("generation")

agent = graph.compile()
