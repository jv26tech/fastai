from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastai.qdrant import upload_website_to_collection
from fastai.rag import get_answer_and_docs

app = FastAPI(title="RAG Chat API", description="Simple RAG chat", version="0.1")


@app.post("/chat")
def chat(message: str):
    response = get_answer_and_docs(message)
    response_content = {
        "question": message,
        "answer": response["answer"],
        "documents": [doc.dict() for doc in response["context"]],
    }
    return JSONResponse(content=response_content, status_code=200)


@app.post("/indexing")
def indexing(url: str):
    upload_website_to_collection(url)
    return JSONResponse(content={"url": url})
