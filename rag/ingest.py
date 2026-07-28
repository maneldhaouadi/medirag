import chromadb
from pathlib import Path
import os
from .ocr import pdf_to_text, image_to_text


def get_mistral_client():
    from mistralai import Mistral
    try:
        import streamlit as st
        api_key = st.secrets["MISTRAL_API_KEY"]
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("Clé API Mistral non trouvée. Vérifie .env ou Streamlit Secrets.")
    return Mistral(api_key=api_key)


# ChromaDB — /tmp fonctionne sur Streamlit Cloud ET en local
chroma_client = chromadb.PersistentClient(path="/tmp/chroma_db")
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
    client = get_mistral_client()
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
    except Exception:
        pass

    collection.add(
        ids=ids,
        embeddings=all_embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print(f"  Document '{doc_name}' ingéré avec succès !")
    return {"chunks": len(chunks), "document": doc_name}


def list_documents() -> list:
    results = collection.get()
    if not results["metadatas"]:
        return []
    sources = list({m["source"] for m in results["metadatas"]})
    return sorted(sources)