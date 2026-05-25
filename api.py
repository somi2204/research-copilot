from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File

from agents.qa_agent import qa_agent
from agents.summary_agent import summary_agent
from agents.compare_agent import compare_agent

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama

import fitz

app = FastAPI()
# Load PDF
doc = fitz.open("data/sample.pdf")

full_text = ""

for page in doc:
    full_text += page.get_text()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=150
)

chunks = text_splitter.split_text(full_text)

# Embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(chunks)

# Local LLM
llm = ChatOllama(model="llama3.2:1b")

# Request schema
class QueryRequest(BaseModel):
    query: str

# Root route
@app.get("/")
def home():

    return {
        "message": "Research Copilot API Running"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {
        "message": f"{file.filename} uploaded successfully"
    }

# Main query endpoint
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
            model,
            embeddings,
            chunks,
            llm
        )

    cleaned_answer = answer.replace("\n", "\n• ")

    return {
        "query": query,
        "response": cleaned_answer
    }

