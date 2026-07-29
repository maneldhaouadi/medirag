"""
voice.py — Interface vocale pour MediRAG
Utilise st.audio_input() (navigateur) + faster-whisper (transcription locale gratuite)
+ gTTS (synthèse vocale). Aucune clé API requise.
"""

import io
import tempfile
import os
import streamlit as st
from gtts import gTTS


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcrit un fichier audio en texte via faster-whisper (100% local, gratuit).
    audio_bytes : contenu brut du fichier audio (wav/webm).
    Retourne la transcription ou "" en cas d'erreur.
    """
    try:
        from faster_whisper import WhisperModel

        # Charge le modèle "tiny" — léger (~75 MB), rapide, suffisant pour du médical court
        # Options : "tiny", "base", "small" (plus précis mais plus lent)
        model = WhisperModel("tiny", device="cpu", compute_type="int8")

        # Sauvegarde l'audio dans un fichier temporaire (faster-whisper lit un fichier)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        segments, _ = model.transcribe(tmp_path, language="fr")
        os.unlink(tmp_path)

        transcription = " ".join(segment.text for segment in segments).strip()
        return transcription

    except Exception as e:
        st.error(f"Erreur de transcription : {e}")
        return ""


def text_to_speech(text: str) -> bytes:
    """
    Convertit du texte en audio MP3 via gTTS (Google Text-to-Speech).
    Limite à 500 caractères pour éviter les délais trop longs.
    Retourne les bytes MP3.
    """
    tts = gTTS(text=text[:500], lang="fr", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.getvalue()