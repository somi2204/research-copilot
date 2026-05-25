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
   You are a helpful and accurate AI resume assistant.

    Answer the question using ONLY the provided context.

    Be concise and clear.

    Context:
    {retrieved_context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content