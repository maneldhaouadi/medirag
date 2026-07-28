import streamlit as st
from pathlib import Path
from datetime import datetime
import tempfile
import os
from rag.ingest import ingest_document, list_documents
from rag.query import ask

st.set_page_config(
    page_title="MediRAG",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediRAG — Assistant Médical Intelligent")
st.caption("Analysez vos documents médicaux en langage naturel · Propulsé par Mistral AI")

# ── Initialise l'historique de conversation ──
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar : gestion des documents ──
with st.sidebar:
    st.header("📁 Mes Documents Médicaux")

    uploaded = st.file_uploader(
        "Ajouter un document",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Ordonnance, analyse de sang, compte-rendu..."
    )

    if uploaded and st.button("📤 Importer", type="primary"):
        with st.spinner(f"Traitement de {uploaded.name}..."):
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            result = ingest_document(tmp_path, doc_name=uploaded.name)
            os.unlink(tmp_path)

            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"✅ {result['chunks']} sections indexées !")
                st.rerun()

    # Liste des documents chargés
    docs = list_documents()
    if docs:
        st.divider()
        st.subheader("Documents chargés")
        for doc in docs:
            st.markdown(f"📄 {doc}")
    else:
        st.info("Aucun document chargé pour l'instant.")

    # Bouton vider l'historique
    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Vider la conversation"):
            st.session_state.messages = []
            st.rerun()

# ── Onglets principaux ──
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Analyse", "📋 Rapport PDF"])

# ─────────────────────────────────────────────
# TAB 1 — Chat
# ─────────────────────────────────────────────
with tab1:
    st.subheader("💬 Posez votre question")

    # Exemples de questions cliquables
    cols = st.columns(3)
    examples = [
        "Quelles sont mes valeurs anormales ?",
        "Explique mon taux de créatinine",
        "Quels médicaments sont prescrits ?"
    ]
    for col, example in zip(cols, examples):
        if col.button(example, use_container_width=True):
            st.session_state.question_example = example

    st.divider()

    # Affiche l'historique de conversation
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "confidence_label" in msg:
                st.caption(f"Confiance : {msg['confidence_label']}")
                if msg.get("sources"):
                    st.caption(f"Sources : {', '.join(msg['sources'])}")

    # Zone de saisie
    question = st.chat_input("Posez votre question médicale...")

    # Gère le clic sur un exemple
    if "question_example" in st.session_state:
        question = st.session_state.pop("question_example")

    # Traitement de la question
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Analyse en cours..."):
                result = ask(question)

            st.markdown(result["answer"])
            st.caption(f"Confiance : {result['confidence_label']}")
            if result["sources"]:
                st.caption(f"Sources : {', '.join(result['sources'])}")

            st.warning(
                "⚕️ Cette analyse est fournie à titre informatif uniquement. "
                "Consultez toujours un professionnel de santé."
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "confidence_label": result["confidence_label"],
            "sources": result["sources"]
        })

# ─────────────────────────────────────────────
# TAB 2 — Analyse (placeholder, à compléter)
# ─────────────────────────────────────────────
with tab2:
    st.subheader("📊 Analyse des valeurs")
    docs = list_documents()
    if not docs:
        st.info("Importez d'abord un document depuis la barre latérale.")
    else:
        st.success(f"{len(docs)} document(s) chargé(s). Posez vos questions dans l'onglet Chat.")

# ─────────────────────────────────────────────
# TAB 3 — Rapport PDF
# ─────────────────────────────────────────────
with tab3:
    st.subheader("📋 Générer un rapport PDF")
    patient_name = st.text_input("Nom du patient (optionnel)")

    if st.button("📥 Générer le rapport PDF"):
        from rag.report import generate_pdf_report
        from rag.query import analyze_critical_values

        docs = list_documents()
        if not docs:
            st.warning("Aucun document chargé.")
        else:
            with st.spinner("Génération du rapport..."):
                alertes = analyze_critical_values(docs[0])
                pdf_bytes = generate_pdf_report(
                    patient_name=patient_name,
                    doc_name=docs[0],
                    alertes=alertes,
                    conversation=st.session_state.messages
                )

            st.download_button(
                label="⬇️ Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=f"rapport_medirag_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )