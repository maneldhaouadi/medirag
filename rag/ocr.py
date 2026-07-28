import fitz
import base64
from pathlib import Path
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


def pdf_to_text(pdf_path: str) -> str:
    client = get_mistral_client()
    doc = fitz.open(pdf_path)
    all_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()

        if len(text) < 50:
            print(f"  → Page {page_num+1} scannée, OCR en cours...")
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode()

            response = client.chat.complete(
                model="pixtral-12b-2409",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Tu es un expert en documents médicaux.
Extrais TOUT le texte de cette image de document médical.
Respecte la mise en forme originale (tableaux, listes, valeurs).
Retourne uniquement le texte extrait, sans commentaire."""
                        }
                    ]
                }]
            )
            text = response.choices[0].message.content

        all_text.append(f"--- Page {page_num+1} ---\n{text}")

    doc.close()
    return "\n\n".join(all_text)


def image_to_text(image_path: str) -> str:
    client = get_mistral_client()

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = Path(image_path).suffix.lower().replace(".", "")
    mime = f"image/{ext if ext != 'jpg' else 'jpeg'}"

    response = client.chat.complete(
        model="pixtral-12b-2409",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": "Extrais tout le texte de ce document médical en préservant la structure."
                }
            ]
        }]
    )
    return response.choices[0].message.content