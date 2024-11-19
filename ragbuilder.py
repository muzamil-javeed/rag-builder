import os
import streamlit as st

# Initialize session state for storing generated snippets
if "generated_snippets" not in st.session_state:
    st.session_state.generated_snippets = {}

def add_code_snippet(step_name, snippet):
    """Add a code snippet for a specific step."""
    st.session_state.generated_snippets[step_name] = snippet

def save_code_to_file():
    """Save all generated code snippets to a single Python file."""
    complete_code = "\n\n".join(st.session_state.generated_snippets.values())
    with open("rag_builder_code.py", "w") as f:
        f.write(complete_code)

# Step 1: Data Ingestion
st.header("Step 1: Data Ingestion")
data_source = st.selectbox("Select Data Source", ["PDF"])
if st.button("Generate Data Ingestion Code"):
    ingestion_code = """
from langchain.document_loaders import PyPDFLoader

# Load PDF documents
loader = PyPDFLoader("your_pdf_file.pdf")
documents = loader.load()
print("Loaded Documents:", documents)
"""
    st.code(ingestion_code, language="python")
    add_code_snippet("Data Ingestion", ingestion_code)

# Step 2: Text Chunking
st.header("Step 2: Text Chunking")
chunking_method = st.selectbox(
    "Select Chunking Method",
    ["Fixed-Size", "Sentence-Based", "Semantic-Based", "Recursive"],
)
if st.button("Generate Chunking Code"):
    if chunking_method == "Fixed-Size":
        chunking_code = """
def fixed_size_chunk(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# Example Usage
chunks = fixed_size_chunk("Sample text for chunking.", max_words=50)
print("Chunks:", chunks)
"""
    elif chunking_method == "Sentence-Based":
        chunking_code = """
import spacy
nlp = spacy.load("en_core_web_sm")

def sentence_chunk(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

# Example Usage
chunks = sentence_chunk("Sample text for sentence-based chunking.")
print("Chunks:", chunks)
"""
    elif chunking_method == "Semantic-Based":
        chunking_code = """
import spacy
nlp = spacy.load("en_core_web_sm")

def semantic_chunk(text, max_len=200):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    for sent in doc.sents:
        current_chunk.append(sent.text)
        if len(' '.join(current_chunk)) > max_len:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Example Usage
chunks = semantic_chunk("Sample text for semantic-based chunking.", max_len=150)
print("Chunks:", chunks)
"""
    elif chunking_method == "Recursive":
        chunking_code = """
def recursive_chunk(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    current_chunk = []
    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Example Usage
chunks = recursive_chunk("Sample text for recursive chunking.", max_tokens=50)
print("Chunks:", chunks)
"""
    st.code(chunking_code, language="python")
    add_code_snippet(f"Text Chunking: {chunking_method}", chunking_code)

# Step 3: Embeddings
st.header("Step 3: Embeddings")
embedding_model = st.selectbox("Select Embedding Model", ["OpenAI", "Hugging Face"])
if st.button("Generate Embeddings Code"):
    if embedding_model == "Hugging Face":
        embeddings_code = """
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Set Hugging Face API Key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "your_huggingface_api_key"

# Initialize Hugging Face Embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


"""
    elif embedding_model == "OpenAI":
        embeddings_code = """
from langchain.embeddings import OpenAIEmbeddings

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings()

# Example Usage
embedded_vector = embeddings.embed_query("Sample query text.")
print("Embedding Vector:", embedded_vector)
"""
    st.code(embeddings_code, language="python")
    add_code_snippet(f"Embeddings: {embedding_model}", embeddings_code)

# Step 4: Vector Database
st.header("Step 4: Vector Database")
vector_db = st.selectbox("Select Vector DB", ["FAISS", "ChromaDB", "Pinecone"])
if st.button("Generate Vector DB Code"):
    if vector_db == "FAISS":
        vector_db_code = """
from langchain.vectorstores import FAISS

# Example Documents
documents = ["Document 1", "Document 2"]

# Initialize FAISS and Add Documents
vector_store = FAISS.from_texts(documents, embeddings)
print("Vector Store Initialized")
"""
    elif vector_db == "ChromaDB":
        vector_db_code = """
from langchain.vectorstores import ChromaDB

# Initialize ChromaDB
vector_store = ChromaDB(embedding_function=embeddings.embed_query)

# Example Documents
documents = ["Document 1", "Document 2"]
for doc in documents:
    vector_store.add_texts([doc])
print("Documents Added to ChromaDB")
"""
    elif vector_db == "Pinecone":
        vector_db_code = """
import pinecone
from langchain.vectorstores import Pinecone

# Initialize Pinecone
pinecone.init(api_key="your_pinecone_api_key", environment="your_pinecone_env")

# Example Documents
documents = ["Document 1", "Document 2"]
vector_store = Pinecone.from_texts(documents, embeddings, index_name="example-index")
print("Documents Added to Pinecone")
"""
    st.code(vector_db_code, language="python")
    add_code_snippet(f"Vector DB: {vector_db}", vector_db_code)

# Step 5: Retrieval and Generation
st.header("Step 5: Retrieval and Generation")
llm_model = st.selectbox("Select LLM Model", ["OpenAI", "Hugging Face", "Gemini"])
if st.button("Generate Retrieval & Generation Code"):
    generation_code = f"""
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Initialize LLM
llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B", model_kwargs={{"temperature": 0.7, "max_length": 512}})

# Initialize RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever())

# Example Query
query = "Explain the concept of attention from the paper."
result = qa_chain.run(query)
print("Result:", result)
"""
    st.code(generation_code, language="python")
    add_code_snippet(f"Retrieval & Generation: {llm_model}", generation_code)

# Save and Download Code
if st.button("Download Complete RAG Code"):
    save_code_to_file()
    with open("rag_builder_code.py", "rb") as file:
        st.download_button("Download RAG Code", file, file_name="rag_builder_code.py")

# Display previously generated snippets
st.sidebar.header("Generated Snippets")
for step, snippet in st.session_state.generated_snippets.items():
    st.sidebar.subheader(step)
    st.sidebar.code(snippet, language="python")
