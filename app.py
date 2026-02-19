# app.py
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
import numpy as np
import soundfile as sf
import io
import json
from pathlib import Path

from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel

# =========================
# Init models
# =========================
encoder = VoiceEncoder()
whisper = WhisperModel("small", device="cpu", compute_type="int8")

app = FastAPI(title="MedVoice Backend")

# =========================
# Simple file persistence
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# In-memory cache (speed only)
VOICEPRINTS: dict[str, np.ndarray] = {}  # doctor_id -> embedding (np.ndarray)


def _safe_key(k: str) -> str:
    return "".join(c for c in k.strip() if c.isalnum() or c in ("_", "-", "."))


def _voiceprint_path(doctor_id: str) -> Path:
    return DATA_DIR / f"voiceprint_{_safe_key(doctor_id)}.json"


def _save_emb(doctor_id: str, emb: np.ndarray) -> None:
    f = _voiceprint_path(doctor_id)
    payload = {
        "doctor_id": doctor_id,
        "embedding": emb.astype(float).tolist(),
    }
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_emb(doctor_id: str):
    f = _voiceprint_path(doctor_id)
    if not f.exists():
        return None
    try:
        payload = json.loads(f.read_text(encoding="utf-8"))
        emb_list = payload.get("embedding")
        if not emb_list:
            return None
        return np.array(emb_list, dtype=np.float32)
    except Exception:
        return None


def _exists_emb(doctor_id: str) -> bool:
    f = _voiceprint_path(doctor_id)
    return f.exists() and f.stat().st_size > 0


def _delete_emb(doctor_id: str) -> None:
    f = _voiceprint_path(doctor_id)
    if f.exists():
        f.unlink()


def _get_emb(doctor_id: str):
    # cache first
    if doctor_id in VOICEPRINTS:
        return VOICEPRINTS[doctor_id]
    # file fallback
    emb = _load_emb(doctor_id)
    if emb is not None:
        VOICEPRINTS[doctor_id] = emb
    return emb


def read_audio(file_bytes: bytes):
    """
    Reads WAV/OGG/FLAC/WEBM if SoundFile can decode it.
    If your browser records WEBM/OPUS and SoundFile fails,
    then we switch to ffmpeg (later).
    """
    data, sr = sf.read(io.BytesIO(file_bytes))
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr


def cosine(a, b) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


# =========================
# Health
# =========================
@app.get("/health")
def health():
    return {"ok": True, "service": "medvoice"}


@app.get("/api/health")
def api_health():
    return {"ok": True, "service": "medvoice"}


# =========================
# Voiceprint management
# =========================
@app.get("/api/voiceprint/exists")
def voiceprint_exists(doctor_id: str = Query(..., min_length=1)):
    """
    GET /api/voiceprint/exists?doctor_id=12345
    Checks FILE existence (truth), not only memory.
    """
    return {"ok": True, "exists": bool(_exists_emb(doctor_id))}


@app.post("/api/voiceprint/delete")
def voiceprint_delete(doctor_id: str = Query(..., min_length=1)):
    """
    POST /api/voiceprint/delete?doctor_id=12345
    Deletes file + cache entry.
    """
    _delete_emb(doctor_id)
    VOICEPRINTS.pop(doctor_id, None)
    return {"ok": True, "deleted": True}


# Optional: still keep /api/voice/exists but it must check FILE, not memory
@app.get("/api/voice/exists")
def voice_exists(doctor_id: str = Query(..., min_length=1)):
    return {"ok": True, "exists": bool(_exists_emb(doctor_id))}


# =========================
# Enroll / Transcribe
# =========================
@app.post("/api/voice/enroll")
async def enroll(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
):
    """
    POST /api/voice/enroll
    FormData:
      - doctor_id
      - audio (file)
    """
    raw = await audio.read()
    wav, sr = read_audio(raw)
    wav_rs = preprocess_wav(wav, source_sr=sr)

    emb = encoder.embed_utterance(wav_rs)
    VOICEPRINTS[doctor_id] = emb
    _save_emb(doctor_id, emb)

    return {"ok": True, "message": "Voice enrolled", "doctor_id": doctor_id}


@app.post("/api/voice/transcribe")
async def transcribe(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
    verify_threshold: float = Form(0.75),
):
    """
    POST /api/voice/transcribe
    FormData:
      - doctor_id
      - audio (file)
      - verify_threshold (optional)
    Returns:
      ok, verified, score, text
    """
    raw = await audio.read()
    wav, sr = read_audio(raw)
    wav_rs = preprocess_wav(wav, source_sr=sr)

    # 1) verify speaker
    emb_saved = _get_emb(doctor_id)
    if emb_saved is None:
        return JSONResponse(
            {"ok": False, "message": "No voice enrolled for doctor_id"},
            status_code=400,
        )

    emb_now = encoder.embed_utterance(wav_rs)
    score = cosine(emb_saved, emb_now)
    verified = score >= float(verify_threshold)

    if not verified:
        return {"ok": True, "verified": False, "score": score, "text": ""}

    # 2) transcribe
    segments, _info = whisper.transcribe(wav, language="es")
    text = "".join([seg.text for seg in segments]).strip()

    return {"ok": True, "verified": True, "score": score, "text": text}
