from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def qa_agent(query, model, embeddings, chunks, llm):

    query_embedding = model.encode([query])

    similarities = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = similarities.argsort()[-5:][::-1]

    retrieved_chunks = [chunks[i] for i in top_indices]

    retrieved_context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are a highly accurate AI resume assistant.

    Answer ONLY using the provided context.

    Do NOT guess dates, organizations, universities, or company names.

    If multiple dates or organizations exist, choose ONLY the one directly related to the question.

    If the answer is unclear, say:
    "The exact information was not clearly found in the document."

    Context:
    {retrieved_context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content