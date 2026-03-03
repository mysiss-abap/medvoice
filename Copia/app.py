# app.py
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# from resemblyzer import VoiceEncoder, preprocess_wav
from faster_whisper import WhisperModel

APP_TITLE  = "MedVoice Backend"
DEFAULT_MODEL = "small"
DEVICE     = "cpu"
COMPUTE_TYPE = "int8"

BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR     = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

GHP_ORIGIN = "https://mysiss-abap.github.io"

whisper = WhisperModel(DEFAULT_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)

app = FastAPI(title=APP_TITLE)


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


app.mount("/static", NoCacheStaticFiles(directory=FRONTEND_DIR), name="static")

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

VOICEPRINTS: dict[str, np.ndarray] = {}
PUSH_RESULTS: list[dict] = []
CONTEXT_STORE: dict[str, dict] = {}
RESULT_BY_SESSION: dict[str, dict] = {}


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

def read_audio(file_bytes: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(file_bytes))
    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


# ── Health ────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "service": "medvoice"}

@app.get("/api/health")
def api_health():
    return {"ok": True, "service": "medvoice"}


# ── Guardar JSON en 2 rutas Windows + backup ──────────────
@app.post("/api/save-json")
async def save_json(payload: dict = Body(...)):
    """
    Rutas destino:
      B) C:\\MedVoice\\json_medvoice.json
    Crea directorios si no existen. Backup siempre en data/.
    """
    targets = [
        Path(r"C:\MedVoice\json_medvoice.json"),
    ]
    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    saved: list[str] = []
    errors: list[dict] = []

    for tp in targets:
        try:
            tp.parent.mkdir(parents=True, exist_ok=True)
            tp.write_text(json_str, encoding="utf-8")
            saved.append(str(tp))
        except Exception as exc:
            errors.append({"path": str(tp), "error": str(exc)})

    # Backup en servidor siempre
    try:
        (DATA_DIR / "json_medvoice.json").write_text(json_str, encoding="utf-8")
        saved.append(str(DATA_DIR / "json_medvoice.json"))
    except Exception:
        pass

    if saved:
        return {"ok": True, "saved": saved, "errors": errors}
    return JSONResponse({"ok": False, "saved": [], "errors": errors}, status_code=500)

# Alias legacy
@app.post("/api/medvoice/save-json")
async def save_json_legacy(payload: dict = Body(...)):
    return await save_json(payload)


# ── Voiceprint ────────────────────────────────────────────
@app.get("/api/voiceprint/exists")
def voiceprint_exists(doctor_id: str = Query(..., min_length=1)):
    return {"ok": True, "exists": bool(_exists_emb(doctor_id))}

@app.post("/api/voiceprint/delete")
def voiceprint_delete(doctor_id: str = Query(..., min_length=1)):
    _delete_emb(doctor_id)
    VOICEPRINTS.pop(doctor_id, None)
    return {"ok": True, "deleted": True}

@app.get("/api/voice/exists")
def voice_exists(doctor_id: str = Query(..., min_length=1)):
    return {"ok": True, "exists": bool(_exists_emb(doctor_id))}

@app.post("/api/noise-profile")
async def noise_profile(doctor_id: str = Form(...), audio: UploadFile = File(...)):
    _ = await audio.read()
    return {"ok": True, "message": "Noise profile received", "doctor_id": doctor_id}

@app.post("/api/enroll")
async def enroll(doctor_id: str = Form(...), audio: UploadFile = File(...)):
    return {"ok": False, "message": "Voice enrollment disabled (resemblyzer commented)"}

@app.post("/api/transcribe")
async def transcribe(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
    target_field: str = Form(""),
    verify_threshold: float = Form(0.75),
):
    raw = await audio.read()
    wav, sr = read_audio(raw)
    segments, _info = whisper.transcribe(wav, language="es")
    text = "".join([seg.text for seg in segments]).strip()
    return {
        "ok": True,
        "doctor_id": doctor_id,
        "target_field": target_field,
        "verify_ok": True,
        "verify_score": 1.0,
        "text": text,
    }

@app.post("/api/context")
async def set_context(payload: dict = Body(...)):
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return JSONResponse({"ok": False, "msg": "session_id required"}, status_code=400)
    item = dict(payload)
    item["session_id"] = session_id
    CONTEXT_STORE[session_id] = item
    return {"ok": True, "session_id": session_id, "stored": True}

@app.get("/api/context")
def get_context(session_id: str = Query(..., min_length=1)):
    sid = str(session_id).strip()
    item = CONTEXT_STORE.get(sid)
    if not item:
        return JSONResponse({"ok": False, "msg": "no context"}, status_code=404)
    return {"ok": True, "context": item}

@app.get("/api/result")
def get_result(session_id: str = Query(..., min_length=1)):
    sid = str(session_id).strip()
    item = RESULT_BY_SESSION.get(sid)
    if not item:
        return {"ok": False, "msg": "no result"}
    return {"ok": True, **item}

@app.post("/api/push_result")
async def push_result(payload: dict = Body(...)):
    item = {
        "session_id": str(payload.get("session_id", "")).strip(),
        "doctor_id": str(payload.get("doctor_id", "")).strip(),
        "patnr": str(payload.get("patnr", "")).strip(),
        "falnr": str(payload.get("falnr", "")).strip(),
        "field": str(payload.get("field", "")).strip(),
        "text": str(payload.get("text", "")).strip(),
    }
    required = ("session_id", "doctor_id", "patnr", "falnr", "field", "text")
    missing = [k for k in required if not item.get(k)]
    if missing:
        return JSONResponse({"ok": False, "msg": "missing fields", "missing": missing}, status_code=400)
    PUSH_RESULTS.append(item)
    RESULT_BY_SESSION[item["session_id"]] = {
        "session_id": item["session_id"],
        "patnr": item["patnr"],
        "falnr": item["falnr"],
        "field": item["field"],
        "text": item["text"],
        "doctor_id": item["doctor_id"],
    }
    return {"ok": True, "stored": True, "result": item}
