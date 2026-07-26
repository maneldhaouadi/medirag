import chromadb
from pathlib import Path
from mistralai import Mistral
from dotenv import load_dotenv
import os
from .ocr import pdf_to_text, image_to_text

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents_medicaux",
    metadata={"hnsw:space": "cosine"}
)


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_embeddings(texts: list) -> list:
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=texts
    )
    return [item.embedding for item in response.data]


def ingest_document(file_path: str, doc_name: str = None) -> dict:
    path = Path(file_path)
    doc_name = doc_name or path.name

    print(f"Traitement de : {doc_name}")

    print("  Extraction du texte...")
    if path.suffix.lower() == ".pdf":
        text = pdf_to_text(file_path)
    else:
        text = image_to_text(file_path)

    if not text.strip():
        return {"error": "Aucun texte extrait"}

    print("  Découpe en chunks...")
    chunks = chunk_text(text)
    print(f"     {len(chunks)} chunks créés")

    print("  Création des embeddings...")
    all_embeddings = []
    batch_size = 10
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = get_embeddings(batch)
        all_embeddings.extend(embeddings)

    print("  Stockage dans ChromaDB...")
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]

    try:
        collection.delete(where={"source": doc_name})
    except:
        pass

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print(f"  Document '{doc_name}' ingere avec succes !")
    return {"chunks": len(chunks), "document": doc_name}


def list_documents() -> list:
    results = collection.get()
    if not results["metadatas"]:
        return []
    sources = list({m["source"] for m in results["metadatas"]})
    return sorted(sources)