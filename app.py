import streamlit as st
from pathlib import Path
import tempfile
import os
from rag.ingest import ingest_document, list_documents
from rag.query import ask

st.set_page_config(
    page_title="MediRAG",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediRAG — Assistant Medical Intelligent")
st.caption("Analysez vos documents medicaux en langage naturel · Propulse par Mistral AI")

with st.sidebar:
    st.header("Mes Documents Medicaux")

    uploaded = st.file_uploader(
        "Ajouter un document",
        type=["pdf", "png", "jpg", "jpeg"],
        help="Ordonnance, analyse de sang, compte-rendu..."
    )

    if uploaded and st.button("Importer", type="primary"):
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
                st.success(f"{result['chunks']} sections indexees !")
                st.rerun()

    docs = list_documents()
    if docs:
        st.divider()
        st.subheader("Documents charges")
        for doc in docs:
            st.markdown(f"- {doc}")
    else:
        st.info("Aucun document charge pour l'instant.")

st.subheader("Posez votre question")

cols = st.columns(3)
examples = [
    "Quelles sont mes valeurs anormales ?",
    "Explique mon taux de creatinine",
    "Quels medicaments sont prescrits ?"
]
for col, example in zip(cols, examples):
    if col.button(example, use_container_width=True):
        st.session_state.question = example

question = st.text_area(
    "Votre question",
    value=st.session_state.get("question", ""),
    placeholder="Ex: Mon taux de cholesterol est-il normal ?",
    height=80,
    label_visibility="collapsed"
)

if st.button("Analyser", type="primary", disabled=not question.strip()):
    with st.spinner("Analyse en cours..."):
        result = ask(question)

    st.divider()
    st.markdown("### Reponse de MediRAG")
    st.markdown(result["answer"])

    if result["sources"]:
        st.divider()
        st.markdown("**Sources consultees :**")
        for src in result["sources"]:
            st.markdown(f"- {src}")

    st.warning("Cette analyse est fournie a titre informatif uniquement. Consultez toujours un professionnel de sante.")