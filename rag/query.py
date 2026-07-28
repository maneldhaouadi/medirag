import chromadb
import os


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


# Même chemin que ingest.py — important !
chroma_client = chromadb.PersistentClient(path="/tmp/chroma_db")
collection = chroma_client.get_or_create_collection(
    name="documents_medicaux",
    metadata={"hnsw:space": "cosine"}
)

SYSTEM_PROMPT = """Tu es MediRAG, un assistant médical intelligent et bienveillant.
Tu aides les utilisateurs à comprendre leurs documents médicaux.

RÈGLE ABSOLUE SUR LA LANGUE :
- Détecte la langue de la question posée par l'utilisateur
- Réponds TOUJOURS dans la même langue que la question
- Si la question est en arabe → réponds en arabe
- Si la question est en français → réponds en français
- Si la question est en anglais → réponds en anglais
- La langue du document PDF n'influence PAS la langue de ta réponse

Règles médicales :
- Réponds TOUJOURS en te basant sur le contexte fourni
- Si une valeur est hors norme, signale-le clairement
- Explique les termes médicaux en langage simple
- Cite toujours tes sources (quel document)
- Rappelle que tu n'es pas médecin
- Ne pose JAMAIS de diagnostic définitif"""


def detect_language(text: str) -> str:
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars > 2:
        return "arabe"
    french_words = ["est", "le", "la", "les", "un", "une", "que", "qui",
                    "comment", "pourquoi", "quoi", "quel", "mon", "ma"]
    if any(word in text.lower() for word in french_words):
        return "français"
    return "anglais"


def search_context(question: str, n_results: int = 5) -> tuple:
    client = get_mistral_client()

    response = client.embeddings.create(
        model="mistral-embed",
        inputs=[question]
    )
    question_embedding = response.data[0].embedding

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    context_parts = []
    sources = []
    distances = results["distances"][0]

    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta["source"]
        if source not in sources:
            sources.append(source)
        context_parts.append(f"[Source: {source}]\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    # Score de confiance
    avg_distance = sum(distances) / len(distances)
    confidence = round((1 - avg_distance) * 100, 1)
    if confidence > 75:
        confidence_label = "🟢 Élevée"
    elif confidence > 50:
        confidence_label = "🟡 Moyenne"
    else:
        confidence_label = "🔴 Faible — réponse peut être imprécise"

    return context, sources, confidence, confidence_label


def ask(question: str) -> dict:
    client = get_mistral_client()

    count = collection.count()
    if count == 0:
        return {
            "answer": "Aucun document médical n'a encore été importé. Veuillez d'abord ajouter vos documents.",
            "sources": [],
            "confidence": 0,
            "confidence_label": "—"
        }

    context, sources, confidence, confidence_label = search_context(question)
    langue = detect_language(question)

    user_prompt = f"""Voici des extraits de documents médicaux pertinents :

{context}

---

Question : {question}

IMPORTANT : Réponds obligatoirement en {langue}.
Réponds de façon claire et bienveillante en te basant sur ces extraits."""

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
        "sources": sources,
        "confidence": confidence,
        "confidence_label": confidence_label
    }