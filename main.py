from agents.compare_agent import compare_agent
from agents.qa_agent import qa_agent
from agents.summary_agent import summary_agent
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import ChatOllama
import fitz
import numpy as np

# Open PDF
doc = fitz.open("data/sample.pdf")

# Extract text
full_text = ""

for page in doc:
    full_text += page.get_text()

print("TEXT EXTRACTED SUCCESSFULLY!\n")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=150
)

chunks = text_splitter.split_text(full_text)

print(f"Number of chunks created: {len(chunks)}")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(chunks)

print("\nEmbeddings created successfully!")

# LLM
llm = ChatOllama(model="llama3.2:1b")

# Prompt
query = input("\nAsk something: ")

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

print("\nAI RESPONSE:\n")
print(answer)