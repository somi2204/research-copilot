def summary_agent(chunks, llm):

    combined_text = " ".join(chunks[:3])

    prompt = f"""
    Summarize the following document:

    {combined_text}
    """

    response = llm.invoke(prompt)

    return response.content