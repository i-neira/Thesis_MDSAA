from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
from uuid import uuid4
import shutil
import time

# Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --------------------- Data Import

# Paths in the project
base_folder = os.path.dirname(os.path.abspath(__file__))

# Name of the folder to store the DB
vector_dbs_path = os.path.abspath(os.path.join(base_folder, "../Vector_DBs"))

# Name of the folder with the documents
folder_path = os.path.abspath(os.path.join(base_folder, "../Corpus"))


# List to save the documents
documents = []

# Check all the .txt files in the folder

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()
            documents.append({
                "filename": filename,
                "content": content
            })

# --------------------- Vector database preparation

db_location = os.path.join(vector_dbs_path, 'chrome_db_1024')

# Delete Chroma DB if exists and initialize

if os.path.exists(db_location):
    shutil.rmtree(db_location)

vector_store = Chroma(
    collection_name="RAG_test",
    persist_directory=db_location,
    embedding_function=embeddings
)

# --------------------- Chunking, embedding and ingestion process

# Initialize text splitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],  # Include separators
    length_function=len,
    is_separator_regex=False,
)

for doc in documents:
    chunks = []
    chunks = text_splitter.create_documents(
        [doc['content']], metadatas=[{"title": doc['filename']}])

    uuids = [str(uuid4()) for _ in range(len(chunks))]

    vector_store.add_documents(documents=chunks, ids=uuids)


# --------------------- Define a retriever

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
