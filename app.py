# app.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel

# =========================
# Config
# =========================
APP_TITLE = "MedVoice Backend"
DEFAULT_MODEL = "small"  # small / medium / large-v3 (según máquina)
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# GitHub Pages origin (tu repo)
GHP_ORIGIN = "https://mysiss-abap.github.io"

# =========================
# Init models
# =========================
encoder = VoiceEncoder()
whisper = WhisperModel(DEFAULT_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

app = FastAPI(title=APP_TITLE)

# CORS para que el HTML en GitHub Pages pueda llamar tu API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        GHP_ORIGIN,
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://localhost",
        "http://localhost:8000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Persistencia (archivos)
# =========================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# cache RAM (solo performance, la verdad está en archivo)
VOICEPRINTS: dict[str, np.ndarray] = {}  # doctor_id -> embedding


def _safe_key(k: str) -> str:
    return "".join(c for c in k.strip() if c.isalnum() or c in ("_", "-", "."))


def _voiceprint_path(doctor_id: str) -> Path:
    return DATA_DIR / f"voiceprint_{_safe_key(doctor_id)}.json"


def _save_emb(doctor_id: str, emb: np.ndarray) -> None:
    f = _voiceprint_path(doctor_id)
    payload = {"doctor_id": doctor_id, "embedding": emb.astype(float).tolist()}
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _load_emb(doctor_id: str) -> Optional[np.ndarray]:
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


def _get_emb(doctor_id: str) -> Optional[np.ndarray]:
    if doctor_id in VOICEPRINTS:
        return VOICEPRINTS[doctor_id]
    emb = _load_emb(doctor_id)
    if emb is not None:
        VOICEPRINTS[doctor_id] = emb
    return emb


# =========================
# Audio helpers
# =========================
def read_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    """
    Lee WAV/OGG/FLAC/WEBM si soundfile lo soporta en tu Windows.
    Si algún día falla WEBM/OPUS, lo cambiamos a ffmpeg.
    """
    data, sr = sf.read(io.BytesIO(file_bytes))
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
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
    # VERDAD por archivo, no por memoria
    return {"ok": True, "exists": bool(_exists_emb(doctor_id))}


@app.post("/api/voiceprint/delete")
def voiceprint_delete(doctor_id: str = Query(..., min_length=1)):
    _delete_emb(doctor_id)
    VOICEPRINTS.pop(doctor_id, None)
    return {"ok": True, "deleted": True}


# Compat: tu HTML setup lo etiqueta como /api/voice/exists a veces
@app.get("/api/voice/exists")
def voice_exists(doctor_id: str = Query(..., min_length=1)):
    return {"ok": True, "exists": bool(_exists_emb(doctor_id))}


# =========================
# (Opcional) Noise profile
# =========================
@app.post("/api/noise-profile")
async def noise_profile(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
):
    # No bloquea nada: lo dejamos como stub (puedes evolucionarlo luego)
    _ = await audio.read()
    return {"ok": True, "message": "Noise profile received", "doctor_id": doctor_id}


# =========================
# Enroll / Transcribe
# =========================
@app.post("/api/enroll")
async def enroll(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
):
    raw = await audio.read()
    wav, sr = read_audio(raw)
    wav_rs = preprocess_wav(wav, source_sr=sr)

    emb = encoder.embed_utterance(wav_rs)
    VOICEPRINTS[doctor_id] = emb
    _save_emb(doctor_id, emb)

    return {"ok": True, "message": "Voice enrolled", "doctor_id": doctor_id}


@app.post("/api/transcribe")
async def transcribe(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
    target_field: str = Form(""),
    verify_threshold: float = Form(0.75),
):
    raw = await audio.read()
    wav, sr = read_audio(raw)
    wav_rs = preprocess_wav(wav, source_sr=sr)

    emb_saved = _get_emb(doctor_id)
    if emb_saved is None:
        return JSONResponse(
            {"ok": False, "message": "No voice enrolled for doctor_id"},
            status_code=400,
        )

    emb_now = encoder.embed_utterance(wav_rs)
    score = cosine(emb_saved, emb_now)
    verify_ok = score >= float(verify_threshold)

    if not verify_ok:
        return {
            "ok": True,
            "doctor_id": doctor_id,
            "target_field": target_field,
            "verify_ok": False,
            "verify_score": score,
            "text": "",
        }

    segments, _info = whisper.transcribe(wav, language="es")
    text = "".join([seg.text for seg in segments]).strip()

    return {
        "ok": True,
        "doctor_id": doctor_id,
        "target_field": target_field,
        "verify_ok": True,
        "verify_score": score,
        "text": text,
    }
