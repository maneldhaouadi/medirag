"""
voice.py — Interface vocale pour MediRAG
Corrections appliquées :
  - Vérification de ffmpeg au démarrage
  - Extension d'entrée générique (.audio) pour compatibilité navigateurs
  - Validation du fichier WAV produit (taille > 0)
  - Logs d'erreur ffmpeg complets et visibles
  - Gestion propre des fichiers temporaires (finally)
"""

import io
import tempfile
import os
import streamlit as st
from gtts import gTTS


# ─────────────────────────────────────────────
# Vérification ffmpeg
# ─────────────────────────────────────────────

def check_ffmpeg() -> bool:
    """
    Vérifie que ffmpeg est installé et accessible dans le PATH.
    Appelé au début de transcribe_audio() pour un feedback immédiat.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except Exception:
        return False


# ─────────────────────────────────────────────
# Chargement du modèle Whisper (mis en cache)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle vocal... (une seule fois)")
def load_whisper_model():
    """
    Charge le modèle Whisper UNE SEULE FOIS grâce au cache Streamlit.
    Sans ce cache, le modèle est retéléchargé à chaque question.
    """
    from faster_whisper import WhisperModel
    return WhisperModel("base", device="cpu", compute_type="int8")


# ─────────────────────────────────────────────
# Conversion audio → WAV
# ─────────────────────────────────────────────

def convert_to_wav(audio_bytes: bytes) -> str:
    """
    Convertit n'importe quel format audio (webm, ogg, mp4...)
    en fichier WAV 16kHz mono, compatible faster-whisper.

    Nécessite ffmpeg installé sur le système (packages.txt : ffmpeg).
    Retourne le chemin du fichier WAV temporaire créé.
    Lève RuntimeError si la conversion échoue ou produit un fichier vide.
    """
    import subprocess

    # ✅ On utilise l'extension .audio (générique) au lieu de .webm
    # car certains navigateurs envoient du .ogg ou .mp4 selon la plateforme.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".audio") as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name

    output_path = input_path + ".wav"

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",       # -y = écraser sans confirmation
                "-i", input_path,     # fichier source (format auto-détecté)
                "-ar", "16000",       # 16 kHz — fréquence optimale pour Whisper
                "-ac", "1",           # mono
                "-f", "wav",          # format de sortie WAV
                output_path
            ],
            capture_output=True,
            timeout=30                # évite un blocage infini
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("ffmpeg a dépassé le délai de 30 secondes.")
    finally:
        # Toujours nettoyer le fichier d'entrée, même en cas d'erreur
        if os.path.exists(input_path):
            os.unlink(input_path)

    # ✅ Vérification du code de retour ffmpeg avec message complet
    if result.returncode != 0:
        error_msg = result.stderr.decode(errors="replace")
        raise RuntimeError(
            f"ffmpeg a échoué (code {result.returncode}) :\n{error_msg}"
        )

    # ✅ Vérification que le fichier WAV existe et n'est pas vide
    if not os.path.exists(output_path):
        raise RuntimeError("ffmpeg n'a pas créé le fichier WAV de sortie.")

    if os.path.getsize(output_path) == 0:
        os.unlink(output_path)
        raise RuntimeError("ffmpeg a produit un fichier WAV vide (audio trop court ?).")

    return output_path


# ─────────────────────────────────────────────
# Transcription audio → texte
# ─────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcrit un enregistrement audio en texte français.

    Étapes :
      1. Vérifie que ffmpeg est disponible
      2. Convertit l'audio en WAV 16kHz mono
      3. Charge le modèle Whisper (via cache)
      4. Transcrit et retourne le texte

    Retourne la transcription (str) ou "" en cas d'erreur.
    """
    # ✅ Vérification ffmpeg avant tout traitement
    if not check_ffmpeg():
        st.error(
            "❌ ffmpeg est introuvable sur ce système. "
            "Ajoutez un fichier `packages.txt` contenant `ffmpeg` à la racine du projet."
        )
        return ""

    wav_path = None
    try:
        # Étape 1 — Conversion format → WAV
        wav_path = convert_to_wav(audio_bytes)

        # Étape 2 — Chargement modèle (mis en cache, fait une seule fois)
        model = load_whisper_model()

        # Étape 3 — Transcription
        segments, info = model.transcribe(
            wav_path,
            language="fr",    # langue forcée en français
            beam_size=3,      # compromis vitesse / précision
            vad_filter=True,  # filtre les silences automatiquement
        )

        transcription = " ".join(seg.text for seg in segments).strip()

        if not transcription:
            st.warning("⚠️ Aucun texte détecté. Parlez plus fort ou réenregistrez.")

        return transcription

    except RuntimeError as e:
        # Erreur de conversion ffmpeg ou de fichier vide
        st.error(f"❌ Erreur de conversion audio : {e}")
        return ""

    except Exception as e:
        # Erreur inattendue (modèle, réseau, etc.)
        st.error(f"❌ Erreur lors de la transcription : {e}")
        return ""

    finally:
        # ✅ Toujours supprimer le fichier WAV temporaire
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


# ─────────────────────────────────────────────
# Synthèse vocale texte → MP3
# ─────────────────────────────────────────────

def text_to_speech(text: str) -> bytes:
    """
    Convertit du texte en audio MP3 via gTTS (Google Text-to-Speech).
    Limite à 500 caractères pour éviter des délais trop longs.
    Retourne les bytes MP3 prêts pour st.audio().
    """
    tts = gTTS(text=text[:500], lang="fr", slow=False)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.getvalue()