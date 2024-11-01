from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os

app = FastAPI()

class RAGOptions(BaseModel):
    chunking: str
    embedding: str
    vectordb: str
    llm: str

@app.post("/generate_code/")
async def generate_code(options: RAGOptions):
    # Templates for different chunking methods
    if options.chunking == "Token-based":
        chunking_logic = """
# Token-based Chunking
def chunk_text(text, chunk_size=100):
    tokens = tokenizer.encode(text)
    chunks = [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]
    return [tokenizer.decode(chunk) for chunk in chunks]
"""
    elif options.chunking == "Sentence-based":
        chunking_logic = """
# Sentence-based Chunking
from nltk.tokenize import sent_tokenize

def chunk_text(text, chunk_size=5):
    sentences = sent_tokenize(text)
    return [" ".join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
"""
    elif options.chunking == "Paragraph-based":
        chunking_logic = """
# Paragraph-based Chunking
def chunk_text(text, chunk_size=1):
    paragraphs = text.split("\\n\\n")
    return ["\\n\\n".join(paragraphs[i:i+chunk_size]) for i in range(0, len(paragraphs), chunk_size)]
"""
    elif options.chunking == "Fixed-size":
        chunking_logic = """
# Fixed-size Chunking
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
"""
    else:
        chunking_logic = "# No valid chunking method selected.\n"

    # Templates for different embedding models
    if options.embedding == "OpenAI Embeddings":
        embedding_logic = """
# OpenAI Embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']
"""
    elif options.embedding == "Hugging Face Transformers":
        embedding_logic = """
# Hugging Face Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).tolist()
"""
    elif options.embedding == "Custom Model":
        embedding_logic = """
# Custom Embedding
def get_embedding(text):
    # Custom embedding logic goes here
    pass
"""
    else:
        embedding_logic = "# No valid embedding model selected.\n"

    # Templates for vector database options
    if options.vectordb == "FAISS":
        vectordb_logic = """
# FAISS Vector Database
import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_embeddings(self, embeddings):
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query_embedding, k=5):
        distances, indices = self.index.search(np.array([query_embedding]).astype('float32'), k)
        return distances, indices
"""
    elif options.vectordb == "Pinecone":
        vectordb_logic = """
# Pinecone Vector Database
import pinecone

# Initialize Pinecone
pinecone.init(api_key='YOUR_API_KEY', environment='YOUR_ENVIRONMENT')

class PineconeDB:
    def __init__(self, index_name):
        self.index = pinecone.Index(index_name)
    
    def add_embeddings(self, embeddings, ids):
        self.index.upsert(vectors=list(zip(ids, embeddings)))
    
    def search(self, query_embedding, top_k=5):
        return self.index.query(queries=[query_embedding], top_k=top_k)
"""
    elif options.vectordb == "Weaviate":
        vectordb_logic = """
# Weaviate Vector Database
from weaviate import Client

client = Client("http://localhost:8080")

class WeaviateDB:
    def add_embeddings(self, embeddings, ids):
        for embedding, id in zip(embeddings, ids):
            client.data_object.create(
                data_object={"vector": embedding},
                class_name="YourClassName",
                uuid=id
            )
    
    def search(self, query_embedding, limit=5):
        return client.query.get("YourClassName", ["*"]) \
            .with_near_vector({"vector": query_embedding}) \
            .with_limit(limit) \
            .do()
"""
    else:
        vectordb_logic = "# No valid vector database selected.\n"

    # Templates for different LLMs
    if options.llm == "OpenAI GPT":
        llm_logic = """
# OpenAI GPT-3.5 API Call
def call_openai_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']
"""
    elif options.llm == "Google Gemini":
        llm_logic = """
# Gemini API Call
def call_gemini(prompt):
    response = requests.post(
        'https://api.gemini.com/v1/generate',
        headers={'Authorization': 'Bearer YOUR_API_KEY'},
        json={'prompt': prompt}
    )
    return response.json()['output']
"""
    else:
        llm_logic = "# No valid LLM selected.\n"

    # Combine all code snippets into the final generated code
    code_chunks = {
        "chunking": f"# Chunking method: {options.chunking}\n{chunking_logic}\n",
        "embedding": f"# Embedding model: {options.embedding}\n{embedding_logic}\n",
        "vectordb": f"# Vector Database: {options.vectordb}\n{vectordb_logic}\n",
        "llm": f"# LLM model: {options.llm}\n{llm_logic}\n",
    }

    # Final code structure to generate embeddings for each chunk, store in DB, and call LLM
    final_code = (
        "'''\n"
        "This code is generated for a RAG setup with the following configurations:\n"
        f"Chunking: {options.chunking}\n"
        f"Embedding Model: {options.embedding}\n"
        f"Vector Database: {options.vectordb}\n"
        f"LLM: {options.llm}\n"
        "'''\n\n"
        + code_chunks["chunking"]
        + code_chunks["embedding"]
        + code_chunks["vectordb"]
        + code_chunks["llm"] +
        """
# Example usage
text = "Your input text goes here."
chunks = chunk_text(text)  # Get the chunks
embeddings = [get_embedding(chunk) for chunk in chunks]  # Get embeddings for each chunk

# Storing embeddings in the vector database
vector_db = FAISSIndex(dimension=len(embeddings[0]))  # Replace with your vector DB class
vector_db.add_embeddings(embeddings)

# Querying the vector database for the nearest embeddings
query_text = "What information do we have about [TOPIC]?"
query_embedding = get_embedding(query_text)  # Get embedding for the query
distances, indices = vector_db.search(query_embedding)  # Get nearest embeddings

# Prepare the prompt for the LLM using the most relevant context
if indices.size > 0:
    relevant_chunks = [chunks[i] for i in indices.flatten()]
    context = " ".join(relevant_chunks)
    prompt = f"Using the following information, {query_text}: {context}"

    # Call the LLM with the prepared prompt
    response = call_openai_gpt(prompt)  # Replace with the selected LLM call
    print(response)
else:
    print("No relevant embeddings found.")
"""
    )

    return {"generated_code": final_code}




@app.post("/download_code/")
async def download_code(options: RAGOptions, file_type: str):
    # Generate code using the provided options
    generated_code_response = await generate_code(options)
    code = generated_code_response["generated_code"]

    # Create a filename based on user selection
    filename = f"rag_setup.{file_type}"
    
    # Save the code to a file
    if file_type == 'py':
        with open(filename, 'w') as f:
            f.write(code)
    elif file_type == 'ipynb':
        # Create a basic Jupyter Notebook structure
        notebook_content = {
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": code.splitlines(keepends=True)
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open(filename, 'w') as f:
            import json
            json.dump(notebook_content, f)

    else:
        raise HTTPException(status_code=400, detail="Invalid file type. Please select either 'py' or 'ipynb'.")

    # Return the file as a response
    return FileResponse(filename, media_type='application/octet-stream', filename=filename)