# Libraries
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os
from uuid import uuid4
import shutil
import json
import time
from typing import List, Dict, Any, Optional

# Global setup

# Paths in the project
base_folder = os.path.dirname(os.path.abspath(__file__))
# Adjusted paths for when called from Code folder
ai_prepositions_path = os.path.abspath(os.path.join(base_folder, "../AI_Prepositions"))
# DB stays in Vector_DBs folder
DB_LOCATION = os.path.join(base_folder, 'chrome_db_ia')
JSON_FILE = os.path.join(ai_prepositions_path, 'ai_chunk_process.json')



# Embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")


def safe_delete_db(db_path: str, max_retries: int = 3, delay: float = 1.0):
    """
    Delets the DB
    """
    if not os.path.exists(db_path):
        print(f"Base de datos no existe (no necesita eliminarse): {db_path}")
        return True

    for attempt in range(max_retries):
        try:
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
                # print(f"Database deleted: {db_path}")
                return True
        except PermissionError as e:
            print(
                f"Attempt {attempt + 1}: Cannot be deleted {db_path} - {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(
                    f"ERROR: Failed to delete database after {max_retries} attempts")
                return False
        except Exception as e:
            print(f"Unexpected error when deleting the database: {e}")
            return False
    return False


def process_loaded_data_to_chunks(data: Dict) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Processes JSON data 

    Args:
        data: Dictionary with loaded JSON data

    Returns:
        tuple: (texts, metadatas) - to use with create_documents()
    """
    texts = []
    metadatas = []

    # Process each file in the JSON
    for file_key, file_data in data.items():
        # Extract filename without extension to use as source
        parts = file_key.split('.')
        if len(parts) == 2:
            source_name = file_key.split('.')[0].strip()
        else:
            source_name = file_key.split('.')[1].strip()

        # Process each chunk_for_embedding
        chunks_for_embedding = file_data.get('chunks_for_embedding', [])

        for chunk in chunks_for_embedding:
            # Extract content
            content = chunk.get('content', '')
            if content.strip():  # Only add if there is content
                texts.append(content)

                # Create metadata
                metadata = {
                    'title': chunk.get('title', ''),
                    'summary': chunk.get('summary', ''),
                    'source': source_name
                }
                metadatas.append(metadata)

    return texts, metadatas


def create_vector_store(force_recreate: bool = False) -> Optional[Chroma]:
    """
    Create or load the vector store

    Args:
        force_recreate: If True, forces the recreation of the database even if it exists

    Returns:
        Chroma vector store instance or None if there's an error
    """
    try:
        # If the recreation is forced or does not exist, create a new one
        if force_recreate or not os.path.exists(DB_LOCATION):
            if not safe_delete_db(DB_LOCATION):
                return None

            # Load JSON
            if not os.path.exists(JSON_FILE):
                print(f"ERROR: File not found {JSON_FILE}")
                return None

            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                datos = json.load(f)

            # Data
            texts, metadatas = process_loaded_data_to_chunks(datos)

            if not texts:
                print("ERROR: No texts found for processing")
                return None

            # Create vector sotre
            vector_store = Chroma(
                collection_name="RAG_test",
                persist_directory=DB_LOCATION,
                embedding_function=embeddings
            )

            # Create documents
            documents = []
            for text, metadata in zip(texts, metadatas):
                doc = Document(page_content=text, metadata=metadata)
                documents.append(doc)

            # Generate UUIDs
            uuids = [str(uuid4()) for _ in range(len(documents))]

            # Add documents in batchs
            batch_size = 50
            total_batches = (len(documents) + batch_size - 1) // batch_size

            print(
                f"Processing {len(documents)} documents in {total_batches} batches...")

            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                batch_uuids = uuids[i:i + batch_size]

                batch_num = i // batch_size + 1
                print(f"Processing batch {batch_num}/{total_batches}")

                vector_store.add_documents(
                    documents=batch_docs, ids=batch_uuids)

            print("Vector store successfully created")
            return vector_store

        else:
            # Load existing vector store
            print("Loading existing vector store...")
            vector_store = Chroma(
                collection_name="RAG_test",
                persist_directory=DB_LOCATION,
                embedding_function=embeddings
            )
            return vector_store

    except Exception as e:
        # Handle any errors
        print(f"ERROR when creating/loading vector store: {e}")
        return None


def get_retriever(k: int = 5, force_recreate: bool = False):
    """
    Get a retriever instance

    Creates or loads the vector store if necessary

    Args:
        k: Number of most similar documents to retrieve for each query
        force_recreate: If True, forces recreation of the vector database

    Returns:
        Retriever or None if fails
    """

    # Create or load the vector store that contains embedded documents
    vector_store = create_vector_store(force_recreate=force_recreate)

    # Validate that vector store was successfully created/loaded
    if vector_store is None:
        return None
    # Convert the vector store into a retriever
    return vector_store.as_retriever(search_kwargs={"k": k})


# Initialize global retriever (only when imported)
retriever = None


def init_retriever(force_recreate: bool = False):
    """
    Initialize global retriever
    """
    global retriever
    retriever = get_retriever(force_recreate=force_recreate)
    return retriever

# Function to safely obtain the retriever


def get_safe_retriever():
    """
    Get retriever instance 

    Returns:
        Cached retriever instance, initializing if necessary

    """
    global retriever
    if retriever is None:
        print("Initializing retriever...")
        retriever = get_retriever()
    return retriever


if __name__ == "__main__":
    print("Running ...")

    # Ask if you want to recreate the database
    recreate = input(
        "¿Recreate the database? (y/n): ").lower().startswith('y')

    # Create retriever
    retriever = get_retriever(force_recreate=recreate)

    if retriever:
        print("Vector store and retriever created successfully")

        # Test
        try:
            test_query = "test"
            results = retriever.invoke(test_query)
            print(
                f"✅ Successful test: {len(results)} documents were found")
        except Exception as e:
            print(f"❌ Error in the test: {e}")
    else:
        print("❌ Error creating retriever")
