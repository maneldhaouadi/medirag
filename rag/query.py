import chromadb
from mistralai import Mistral
from dotenv import load_dotenv
import os

load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents_medicaux",
    metadata={"hnsw:space": "cosine"}
)

SYSTEM_PROMPT = """Tu es MediRAG, un assistant médical intelligent et bienveillant.
Tu aides les utilisateurs a comprendre leurs documents medicaux (analyses de sang,
ordonnances, comptes-rendus, imageries).

Regles importantes :
- Reponds TOUJOURS en te basant sur le contexte fourni
- Si une valeur est hors norme, signale-le clairement
- Utilise un langage clair et accessible, explique les termes medicaux
- Cite toujours tes sources (quel document, quelle section)
- Rappelle que tu n'es pas medecin et qu'une consultation reste necessaire
- Ne pose JAMAIS de diagnostic definitif"""


def search_context(question: str, n_results: int = 5) -> tuple:
    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[question]
    )
    question_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results
    )

    context_parts = []
    sources = []

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta["source"]
        if source not in sources:
            sources.append(source)
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def ask(question: str) -> dict:
    count = collection.count()
    if count == 0:
        return {
            "answer": "Aucun document medical n'a encore ete importe. Veuillez d'abord ajouter vos documents.",
            "sources": []
        }

    context, sources = search_context(question)

    user_prompt = f"""Voici des extraits de documents medicaux pertinents :

{context}

---

Question du patient : {question}

Reponds de facon claire, structuree et bienveillante en te basant sur ces extraits."""

    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }