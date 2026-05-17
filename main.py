from fastapi import FastAPI
from model import agent, HumanMessage
from ingest import get_files, init_ingestion

app = FastAPI()
init_ingestion()

@app.post("/query")
async def handle_question(q: str) -> dict[str, str]:
    """This function handles queries posed by the user"""
    response = agent.invoke({"messages": [HumanMessage(content=q)]})
    return {"response": response["messages"][-1].content}


@app.post("/ingest")
async def handle_upload():
    pass


@app.get("/documents")
async def fetch_documents():
    """This function returns list of files currently in corpus"""
    return {"corpus": get_files()}


@app.post("/feedback")
async def handle_feedback():
    pass
