def compare_agent(chunks, llm):

    combined_text = " ".join(chunks[:5])

    prompt = f"""
    Compare the important information in the document.

    Focus on:
    - skills
    - projects
    - experience
    - technologies

    Document:
    {combined_text}
    """

    response = llm.invoke(prompt)

    return response.content