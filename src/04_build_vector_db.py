import os
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
from dotenv import load_dotenv

import chromadb
from google import genai
from google.genai import types
from models import EMBED_MODEL, EMBED_DIM


BASE_DIR = Path(__file__).resolve().parent.parent
CHUNKS_PATH = BASE_DIR / "data" / "processed" / "chunks.json"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"


def load_chunks(file_path: Path) -> List[Dict[str, Any]]:
    """
    Reads data from a JSON file and returns it as a list of dictionaries.

    Args:
        file_path (Path): The path to the JSON file to be read.

    Returns:
        List[Dict[str, Any]]: A list of parsed dictionaries containing the chunk data.
        
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"Fayl tapılmadı: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans metadata by converting unsupported types (None, list) into strings 
    for ChromaDB compatibility.

    Args:
        chunk (Dict[str, Any]): A single chunk dictionary containing metadata.

    Returns:
        Dict[str, Any]: A cleaned metadata dictionary safe for ChromaDB insertion.
    """
    meta = {}
    for k, v in chunk.items():
        if k in ("chunk_id", "text"):
            continue
        if v is None:
            meta[k] = ""
        elif isinstance(v, list):
            meta[k] = ", ".join(str(x) for x in v)
        else:
            meta[k] = v
    return meta


def generate_embeddings(chunks: List[Dict[str, Any]], client: genai.Client) -> Tuple[List[str], List[List[float]], List[str], List[Dict[str, Any]]]:
    """
    Sends text chunks to the API, generates vector embeddings, and prepares 
    separated data lists for ChromaDB ingestion.

    Args:
        chunks (List[Dict[str, Any]]): A list of chunk dictionaries to embed.
        client (genai.Client): An initialized Google GenAI client instance.

    Returns:
        Tuple[List[str], List[List[float]], List[str], List[Dict[str, Any]]]: 
        A tuple containing lists of chunk IDs, embeddings, documents (texts), 
        and metadata dictionaries, respectively.
    """
    ids = []
    embeddings = []
    documents = []
    metadatas = []

    total_chunks = len(chunks)
    print(f"Embedding process started. Total chunks: {total_chunks}")

    for i, chunk in enumerate(chunks):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=chunk["text"],
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBED_DIM,
                ),
            )
            
            ids.append(chunk["chunk_id"])
            embeddings.append(result.embeddings[0].values)
            documents.append(chunk["text"])
            metadatas.append(clean_metadata(chunk))

            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{total_chunks} chunks embedded...")

            time.sleep(0.5) 

        except Exception as e:
            print(f"ERROR (Chunk ID: {chunk.get('chunk_id', 'Unknown')}): {e}")

    print(f"API process finished. Successfully generated embeddings: {len(ids)}")
    return ids, embeddings, documents, metadatas


def save_to_chromadb(ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]], db_path: Path):
    """
    Writes vectors, documents, and metadata into a local ChromaDB database.

    Args:
        ids (List[str]): A list of unique chunk identifiers.
        embeddings (List[List[float]]): A list of generated vector embeddings.
        documents (List[str]): A list of original text chunks.
        metadatas (List[Dict[str, Any]]): A list of cleaned metadata dictionaries.
        db_path (Path): The local file path where the ChromaDB will be stored.

    Returns:
        None
    """

    chroma_client = chromadb.PersistentClient(path=str(db_path))

    collection = chroma_client.get_or_create_collection(
        name="family_code_az",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Total records (documents) in ChromaDB: {collection.count()}")
    print(f"Database successfully saved to: {db_path}")


def main():

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY not found! Please check your .env file.")

    client = genai.Client(api_key=api_key)

    chunks = load_chunks(CHUNKS_PATH)

    ids, embeddings, documents, metadatas = generate_embeddings(chunks, client)

    if ids:
        save_to_chromadb(ids, embeddings, documents, metadatas, CHROMA_DB_PATH)
    else:
        print("No data was successfully embedded. Database write operation aborted.")

if __name__ == "__main__":
    main()