import streamlit as st
import requests

# Title and Description
st.title("RAG Code Builder")
st.write("Generate code for Retrieval-Augmented Generation (RAG) with customizable configurations.")

# Initialize session state for options if it doesn't exist
if 'options' not in st.session_state:
    st.session_state.options = {
        "chunking": None,
        "embedding": None,
        "vectordb": None,
        "llm": None,
    }

# 1. Chunking Options
st.subheader("Chunking Method")
st.session_state.options["chunking"] = st.selectbox(
    "Select a chunking method:",
    ["Token-based", "Sentence-based", "Paragraph-based", "Fixed-size"],
    index=0  # Set default index to the first option
)

# 2. Embedding Options
st.subheader("Embedding Model")
st.session_state.options["embedding"] = st.selectbox(
    "Select an embedding model:",
    ["OpenAI Embeddings", "Hugging Face Transformers", "Custom Model"],
    index=0
)

# 3. Vector Database Options
st.subheader("Vector Database")
st.session_state.options["vectordb"] = st.selectbox(
    "Select a vector database:",
    ["Pinecone", "Weaviate", "FAISS", "ChromaDB"],
    index=0
)

# 4. LLM Options
st.subheader("Large Language Model")
st.session_state.options["llm"] = st.selectbox(
    "Select an LLM:",
    ["OpenAI GPT", "Google Gemini"],
    index=0
)

# Backend API URLs
generate_url = "http://127.0.0.1:8000/generate_code/"
download_url = "http://127.0.0.1:8000/download_code/"

# Generate Button
if st.button("Generate Code"):
    # Display loading message
    with st.spinner("Generating code..."):
        # Send selected options to FastAPI backend
        response = requests.post(generate_url, json=st.session_state.options)

        if response.status_code == 200:
            generated_code = response.json().get("generated_code")
            st.code(generated_code, language="python")
            st.session_state.generated_code = generated_code  # Store generated code in session state
        else:
            st.error("Failed to generate code. Please try again.")

# Check if generated code exists in session state
if 'generated_code' in st.session_state:
    if st.button("Download Code as .py"):
        response = requests.post(download_url, json={"options": st.session_state.options, "file_type": "py"})
        if response.status_code == 200:
            with open("rag_setup.py", "wb") as f:
                f.write(response.content)
            st.success("Code downloaded as rag_setup.py")
        else:
            st.error("Failed to download the file. Please try again.")

    if st.button("Download Code as .ipynb"):
        response = requests.post(download_url, json={"options": st.session_state.options, "file_type": "ipynb"})
        if response.status_code == 200:
            with open("rag_setup.ipynb", "wb") as f:
                f.write(response.content)
            st.success("Code downloaded as rag_setup.ipynb")
        else:
            st.error("Failed to download the file. Please try again.")
