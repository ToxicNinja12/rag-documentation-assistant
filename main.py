from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/query")
async def handle_question():
    pass


@app.post("/ingest")
async def handle_upload():
    pass


@app.get("/documents")
async def fetch_documents():
    pass


@app.post("/feedback")
async def handle_feedback():
    pass
