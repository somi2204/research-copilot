import streamlit as st
import requests

st.set_page_config(
    page_title="Research Copilot",
    layout="wide"
)

st.title("🤖 Research Copilot")
st.write("Multi-Agent AI Research Assistant")

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Upload PDF"):

        files = {
            "file": uploaded_file.getvalue()
        }

        response = requests.post(
            "http://127.0.0.1:8000/upload",
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file,
                    "application/pdf"
                )
            }
        )

        st.success("PDF uploaded successfully!")

# User input
query = st.text_input(
    "Ask something about the document:"
)

# Button
if st.button("Submit"):

    response = requests.post(
        "http://127.0.0.1:8000/query",
        json={
            "query": query
        }
    )

    result = response.json()

    st.subheader("AI Response")

    st.write(result["response"])