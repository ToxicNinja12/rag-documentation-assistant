from fastapi import FastAPI
from model import agent, HumanMessage

app = FastAPI()


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
    pass


@app.post("/feedback")
async def handle_feedback():
    pass
