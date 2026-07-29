"""
voice.py — Interface vocale pour MediRAG
Correction : conversion webm→wav + gestion du chargement modèle + feedback utilisateur
"""

import io
import tempfile
import os
import streamlit as st
from gtts import gTTS


@st.cache_resource(show_spinner="Chargement du modèle vocal... (une seule fois)")
def load_whisper_model():
    """
    Charge le modèle Whisper UNE SEULE FOIS grâce au cache Streamlit.
    Sans ce cache, le modèle est retéléchargé à chaque question.
    """
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")


def convert_to_wav(audio_bytes: bytes) -> str:
    """
    Convertit n'importe quel format audio (webm, ogg, mp4...)
    en fichier WAV compatible faster-whisper.
    Nécessite ffmpeg installé sur le système.
    Retourne le chemin du fichier WAV temporaire.
    """
    import subprocess

    # Sauvegarde l'audio brut dans un fichier temporaire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    output_path = input_path.replace(".webm", ".wav")

    # Conversion via ffmpeg (disponible sur Streamlit Cloud et la plupart des systèmes)
    result = subprocess.run([
        "ffmpeg", "-y",           # -y = écraser sans confirmation
        "-i", input_path,         # fichier source
        "-ar", "16000",           # 16kHz — fréquence optimale pour Whisper
        "-ac", "1",               # mono
        "-f", "wav",              # format WAV
        output_path
    ], capture_output=True)

    os.unlink(input_path)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg échoué : {result.stderr.decode()}")

    return output_path


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcrit l'audio en texte.
    Retourne la transcription ou "" en cas d'erreur.
    """
    wav_path = None
    try:
        # Étape 1 — Conversion format
        wav_path = convert_to_wav(audio_bytes)

        # Étape 2 — Chargement modèle (caché, fait une seule fois)
        model = load_whisper_model()

        # Étape 3 — Transcription
        segments, info = model.transcribe(
            wav_path,
            language="fr",
            beam_size=3,           # compromis vitesse/précision
            vad_filter=True,       # filtre les silences automatiquement
        )

        transcription = " ".join(seg.text for seg in segments).strip()
        return transcription

    except FileNotFoundError:
        st.error("❌ ffmpeg n'est pas installé. Ajoutez `packages.txt` avec `ffmpeg`.")
        return ""
    except Exception as e:
        st.error(f"❌ Erreur transcription : {e}")
        return ""
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


def text_to_speech(text: str) -> bytes:
    """
    Convertit du texte en audio MP3 via gTTS.
    Limite à 500 caractères pour éviter les délais trop longs.
    """
    tts = gTTS(text=text[:500], lang="fr", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.getvalue()