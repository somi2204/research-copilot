from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from agents.qa_agent import qa_agent
from agents.summary_agent import summary_agent
from agents.compare_agent import compare_agent

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama

import fitz
import os

app = FastAPI()


# MODELS

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)

llm = ChatOllama(
    model="llama3.2:1b"
)


# GLOBAL VARIABLES


chunks = []
embeddings = []

current_pdf_path = "data/current.pdf"

# PROCESS PDF FUNCTION

def process_pdf(current_pdf_path):

    global chunks
    global embeddings

    doc = fitz.open(current_pdf_path)

    full_text = ""

    for page in doc:
        full_text += page.get_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=150
    )

    chunks = text_splitter.split_text(full_text)

    embeddings = embedding_model.encode(chunks)

    print(f"\nProcessed PDF: {current_pdf_path}")
    print(f"Total Chunks: {len(chunks)}")

# REQUEST MODEL


class QueryRequest(BaseModel):
    query: str


# HOME ROUTE

@app.get("/")
def home():

    return {
        "message": "Research Copilot API Running"
    }

# PDF UPLOAD ROUTE

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    global current_pdf_path

    file_path = "data/current.pdf"

    contents = await file.read()

    with open(file_path, "wb") as f:
        f.write(contents)

    current_pdf_path = "data/current.pdf"

    print(f"\nNEW PDF UPLOADED: {file.filename}")

    process_pdf(current_pdf_path)

    return {
        "message": "PDF uploaded and processed successfully"
    }

# QUERY ROUTE

@app.post("/query")
def query_ai(request: QueryRequest):

    query = request.query

    # ROUTER

    if "summarize" in query.lower():

        answer = summary_agent(
            chunks,
            llm
        )

    elif "compare" in query.lower():

        answer = compare_agent(
            chunks,
            llm
        )

    else:

        answer = qa_agent(
            query,
            embedding_model,
            embeddings,
            chunks,
            llm
        )

    return {
        "query": query,
        "response": answer
    }