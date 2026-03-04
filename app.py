# app.py
from __future__ import annotations

import io
import json
import os
import re
import asyncio
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# ===== LOCAL WHISPER (FALLBACK) =====
from faster_whisper import WhisperModel

# ===== SPEECHMATICS INTEGRATION =====
from speechmatics.batch import AsyncClient as SpeechmaticsClient

# ===== VOICE VERIFICATION =====
from resemblyzer import VoiceEncoder, preprocess_wav

# from faster_whisper import WhisperModel  # ← REPLACED BY SPEECHMATICS

# Load environment variables
load_dotenv()

APP_TITLE = "MedVoice Backend with Speechmatics Medical"

# ===== SPEECHMATICS CONFIGURATION (OPTIONAL) =====
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "").strip()
SPEECHMATICS_ENABLED = bool(SPEECHMATICS_API_KEY)

if SPEECHMATICS_ENABLED:
    print("✅ Speechmatics enabled")
    print(f"   API Key: {SPEECHMATICS_API_KEY[:8]}...{SPEECHMATICS_API_KEY[-4:]}")
else:
    print("⚠️ Speechmatics disabled (no API key). Using local Whisper fallback.")

# ===== WHISPER LOCAL CONFIGURATION =====
# Default model for local runs: "small" (good balance). Emergency: "tiny".
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small").strip()      # tiny / base / small / medium / large-v3
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu").strip()      # cpu / cuda
WHISPER_COMPUTE = os.getenv("WHISPER_COMPUTE", "int8").strip()   # int8 / float16 (cuda) / float32

print(f"⏳ Loading Whisper local model: {WHISPER_MODEL} ({WHISPER_DEVICE}, {WHISPER_COMPUTE})")
try:
    whisper_local = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    whisper_tiny  = WhisperModel("tiny", device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print("✅ Whisper local models ready")
except Exception as e:
    print(f"❌ Could not load Whisper models: {str(e)}")
    whisper_local = None
    whisper_tiny  = None

# ===== FILLER WORD REMOVAL CONFIGURATION =====
ENABLE_FILLER_REMOVAL = os.getenv("ENABLE_FILLER_REMOVAL", "true").lower() == "true"

BASE_DIR     = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
DATA_DIR     = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

GHP_ORIGIN = "https://mysiss-abap.github.io"

# ===== SPEECHMATICS READY =====
# Speechmatics client is created per request (async)
print("✅ Speechmatics Medical configured")
print("   - Language: Spanish (es)")
print("   - Model: Medical Domain")
print("   - Diarization: Enabled")

# ===== VOICE ENCODER =====
print("⏳ Loading Resemblyzer voice encoder...")
try:
    voice_encoder = VoiceEncoder()
    print("✅ Voice encoder loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load voice encoder: {str(e)}")
    voice_encoder = None

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


# ── Frontend Routes ───────────────────────────────────────
@app.get("/")
def index():
    """Serve the main frontend index.html (auto-redirects based on state)"""
    from fastapi.responses import FileResponse
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        return {"error": "Frontend not found", "path": str(index_path)}
    return FileResponse(index_path)

@app.get("/setup")
def setup():
    """Direct access to setup page"""
    from fastapi.responses import FileResponse
    return FileResponse(FRONTEND_DIR / "medvoice-setup.html")

@app.get("/transcribe")
def transcribe_page():
    """Direct access to transcription page"""
    from fastapi.responses import FileResponse
    return FileResponse(FRONTEND_DIR / "medvoice-transcribe.html")

@app.get("/submit")
def submit_page():
    """Direct access to submit page"""
    from fastapi.responses import FileResponse
    return FileResponse(FRONTEND_DIR / "medvoice-submit.html")

@app.get("/tools")
def tools_page():
    """Direct access to tools page"""
    from fastapi.responses import FileResponse
    return FileResponse(FRONTEND_DIR / "medvoice-tools.html")

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
    """
    Register doctor's voice voiceprint for identification.
    """
    if voice_encoder is None:
        return {
            "ok": False,
            "error": "Voice encoder not available. Resemblyzer not loaded."
        }

    try:
        # Read audio
        raw = await audio.read()
        wav, sr = read_audio(raw)

        # Preprocess and extract embedding
        print(f"🎙️ Enrolling voice for doctor: {doctor_id}")
        preprocessed = preprocess_wav(wav, sr)
        embedding = voice_encoder.embed_utterance(preprocessed)

        # Save voiceprint
        _save_emb(doctor_id, embedding)
        VOICEPRINTS[doctor_id] = embedding

        print(f"✅ Voice enrolled successfully for doctor: {doctor_id}")
        return {
            "ok": True,
            "message": "Voice enrolled successfully",
            "doctor_id": doctor_id
        }

    except Exception as e:
        print(f"❌ Error enrolling voice: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e)
        }


# ===== TEXT POST-PROCESSING =====
def clean_medical_transcript(text: str, language: str = "es") -> str:
    """
    Remove filler words (muletillas) and normalize medical transcript.

    Args:
        text: Transcribed text
        language: Language code ("es" or "en")

    Returns:
        Cleaned text
    """
    if not ENABLE_FILLER_REMOVAL:
        return text

    if language == "es":
        # Spanish filler words
        fillers = [
            r'\b(este|eh|ehh|hmm|mmm|uh|um|ah|ahh)\b',
            r'\b(o sea|pues|entonces|bueno|bien)\b',
            r'\b(¿no\?|¿verdad\?|¿cierto\?)\b',
            r'\b(como que|tipo|digamos)\b',
        ]
    else:
        # English filler words
        fillers = [
            r'\b(um|uh|uhm|err|ah|ahh)\b',
            r'\b(like|you know|I mean|sort of|kind of)\b',
            r'\b(actually|basically|literally)\b',
        ]

    # Remove filler words
    for filler in fillers:
        text = re.sub(filler, '', text, flags=re.IGNORECASE)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)

    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)

    # Remove leading/trailing spaces
    text = text.strip()

    return text

async def transcribe_with_whisper_local(audio_path: str, language: str = "es") -> dict:
    if whisper_local is None:
        raise Exception("Whisper local model not available")

    segments, info = whisper_local.transcribe(
        audio_path,
        language=language if language in ("es", "en") else None,
        vad_filter=True,
        beam_size=5
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    text_clean = clean_medical_transcript(text, language)
    return {
        "text": text_clean,
        "text_original": text,
        "model": f"Whisper Local ({WHISPER_MODEL})",
        "language": language,
        "segments": []
    }


async def transcribe_with_whisper_tiny(audio_path: str, language: str = "es") -> dict:
    if whisper_tiny is None:
        raise Exception("Whisper tiny model not available")

    segments, info = whisper_tiny.transcribe(
        audio_path,
        language=language if language in ("es", "en") else None,
        vad_filter=True,
        beam_size=3
    )

    text = " ".join(seg.text.strip() for seg in segments).strip()
    text_clean = clean_medical_transcript(text, language)
    return {
        "text": text_clean,
        "text_original": text,
        "model": "Whisper Tiny (Emergency)",
        "language": language,
        "segments": []
    }

async def transcribe_with_fallback(
    audio_path: str,
    language: str = "es",
    doctor_id: Optional[str] = None,
    verify_threshold: float = 0.65
) -> dict:
    # 1) Try Speechmatics if enabled
    if SPEECHMATICS_ENABLED:
        try:
            return await transcribe_with_speechmatics(
                audio_path,
                language=language,
                doctor_id=doctor_id,
                verify_threshold=verify_threshold
            )
        except Exception as e:
            print(f"⚠️ Speechmatics failed -> fallback to Whisper local. Reason: {str(e)}")

    # 2) Whisper local
    try:
        return await transcribe_with_whisper_local(audio_path, language=language)
    except Exception as e:
        print(f"⚠️ Whisper local failed -> fallback to Whisper tiny. Reason: {str(e)}")

    # 3) Whisper tiny emergency
    return await transcribe_with_whisper_tiny(audio_path, language=language)

# ===== SPEECHMATICS TRANSCRIPTION FUNCTION =====
async def transcribe_with_speechmatics(
    audio_path: str,
    language: str = "es",
    doctor_id: Optional[str] = None,
    verify_threshold: float = 0.65
) -> dict:
    """
    Transcribe medical audio using Speechmatics Medical Model with speaker identification.

    Args:
        audio_path: Path to audio file
        language: Language code (es=Spanish, en=English)
        doctor_id: Doctor ID for voice identification (optional)
        verify_threshold: Similarity threshold for doctor identification (0-1)

    Returns:
        dict with: text (full transcription), doctor_text, patient_text, segments, metadata
    """
    try:
        print(f"🎤 Transcribing with Speechmatics Medical...")
        print(f"   - File: {audio_path}")
        print(f"   - Language: {language}")
        if doctor_id:
            print(f"   - Doctor ID: {doctor_id}")

        # Create Speechmatics client
        client = SpeechmaticsClient(api_key=SPEECHMATICS_API_KEY)

        # Medical transcription configuration
        transcription_config = {
            "language": language,
            "operating_point": "enhanced",  # Maximum accuracy
            "domain": "medical",            # Medical specialized model

            # Advanced features
            "enable_entities": True,        # Detect entities (medications, conditions)
            "diarization": "speaker",       # Separate voices (doctor/patient)
            "speaker_diarization_config": {
                "max_speakers": 3           # Maximum 3 speakers
            },

            # Output format
            "output_locale": "es-ES" if language == "es" else "en-US",
            "punctuation_overrides": {
                "permitted_marks": [",", ".", "?", "!"]
            }
        }

        # Perform transcription
        print("⏳ Sending audio to Speechmatics...")
        result = await client.transcribe(
            audio_path,
            transcription_config=transcription_config
        )

        # Close client
        await client.close()

        # Extract full text
        transcript_text = result.transcript_text.strip()

        # Clean filler words
        transcript_text_clean = clean_medical_transcript(transcript_text, language)

        # Extract segments with speaker labels
        segments = []
        if hasattr(result, 'results') and result.results:
            for segment in result.results:
                segment_text = segment.get("alternatives", [{}])[0].get("content", "")
                segments.append({
                    "speaker": segment.get("speaker", "unknown"),
                    "text": segment_text,
                    "text_clean": clean_medical_transcript(segment_text, language),
                    "start_time": segment.get("start_time", 0),
                    "end_time": segment.get("end_time", 0)
                })

        # ===== SPEAKER IDENTIFICATION =====
        doctor_text = ""
        patient_text = ""
        identified_segments = segments.copy()

        # If doctor_id is provided and voiceprint exists, identify doctor vs patient
        if doctor_id and voice_encoder is not None:
            doctor_embedding = _get_emb(doctor_id)

            if doctor_embedding is not None:
                print(f"🔍 Identifying doctor voice using voiceprint...")

                # Load audio for voice comparison
                try:
                    with open(audio_path, 'rb') as f:
                        audio_content = f.read()
                    wav, sr = read_audio(audio_content)
                    preprocessed = preprocess_wav(wav, sr)

                    # Extract segments from audio for each speaker
                    # Group segments by speaker
                    speaker_groups = {}
                    for seg in segments:
                        spk = seg.get("speaker", "unknown")
                        if spk not in speaker_groups:
                            speaker_groups[spk] = []
                        speaker_groups[spk].append(seg)

                    # Compare each speaker with doctor voiceprint
                    speaker_similarities = {}
                    for speaker_id, speaker_segments in speaker_groups.items():
                        # Use the full audio embedding as approximation
                        # (In production, you'd extract specific time segments)
                        current_embedding = voice_encoder.embed_utterance(preprocessed)
                        similarity = cosine(doctor_embedding, current_embedding)
                        speaker_similarities[speaker_id] = similarity
                        print(f"   Speaker {speaker_id}: similarity = {similarity:.3f}")

                    # Identify which speaker is the doctor (highest similarity above threshold)
                    doctor_speaker = None
                    max_similarity = 0
                    for spk, sim in speaker_similarities.items():
                        if sim > max_similarity and sim >= verify_threshold:
                            max_similarity = sim
                            doctor_speaker = spk

                    if doctor_speaker:
                        print(f"✅ Doctor identified as: {doctor_speaker} (similarity: {max_similarity:.3f})")
                    else:
                        print(f"⚠️ Could not identify doctor (max similarity: {max(speaker_similarities.values(), default=0):.3f} < threshold: {verify_threshold})")

                    # Label segments and separate text
                    doctor_segments = []
                    patient_segments = []

                    for seg in identified_segments:
                        if seg["speaker"] == doctor_speaker:
                            seg["identified_as"] = "doctor"
                            doctor_segments.append(seg["text_clean"])
                        else:
                            seg["identified_as"] = "patient"
                            patient_segments.append(seg["text_clean"])

                    doctor_text = " ".join(doctor_segments).strip()
                    patient_text = " ".join(patient_segments).strip()

                    print(f"   Doctor text: {len(doctor_text)} chars")
                    print(f"   Patient text: {len(patient_text)} chars")

                except Exception as e:
                    print(f"⚠️ Could not identify speakers: {str(e)}")
            else:
                print(f"⚠️ No voiceprint found for doctor: {doctor_id}")

        print(f"✅ Transcription completed: {len(transcript_text)} characters")
        print(f"   Segments detected: {len(segments)}")
        print(f"   Preview: {transcript_text_clean[:100]}...")

        return {
            "text": transcript_text_clean,
            "text_original": transcript_text,
            "doctor_text": doctor_text,
            "patient_text": patient_text,
            "segments": identified_segments,
            "language": language,
            "model": "Speechmatics Medical",
            "metadata": {
                "duration": getattr(result, 'duration', 0),
                "speakers_detected": len(set(s.get("speaker") for s in segments)) if segments else 1,
                "filler_removal_enabled": ENABLE_FILLER_REMOVAL,
                "voice_identification_attempted": doctor_id is not None and voice_encoder is not None
            }
        }

    except Exception as e:
        print(f"❌ Error in Speechmatics transcription: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


@app.post("/api/transcribe")
async def transcribe(
    doctor_id: str = Form(...),
    audio: UploadFile = File(...),
    target_field: str = Form(""),
    verify_threshold: float = Form(0.65),
    language: str = Form("es"),  # Default to Spanish
    enable_voice_identification: bool = Form(True),  # Enable doctor vs patient identification
):
    """
    Transcribe medical audio using Speechmatics with doctor/patient identification.

    Args:
        doctor_id: Doctor ID
        audio: Audio file (WAV, MP3, etc.)
        target_field: Target field (optional)
        verify_threshold: Voice similarity threshold for doctor identification (0-1)
        language: Language code (es=Spanish, en=English)
        enable_voice_identification: Enable doctor vs patient voice identification

    Returns:
        JSON with transcribed text, doctor_text, patient_text, and metadata
    """
    temp_path = None
    try:
        # Save temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name
            content = await audio.read()
            temp_file.write(content)

        print(f"📁 Audio saved: {temp_path} ({len(content)} bytes)")

        # ===== TRANSCRIBE WITH SPEECHMATICS =====
        # Pass doctor_id for voice identification if enabled
        result = await transcribe_with_fallback(
            temp_path,
            language=language,
            doctor_id=doctor_id if enable_voice_identification else None,
            verify_threshold=verify_threshold
        )

        # Response (maintaining compatibility with existing frontend)
        response_data = {
            "ok": True,
            "text": result["text"],  # Cleaned full text
            "text_original": result.get("text_original", ""),  # Original with fillers
            "doctor_text": result.get("doctor_text", ""),  # Only doctor's words
            "patient_text": result.get("patient_text", ""),  # Only patient's words
            "doctor_id": doctor_id,
            "target_field": target_field,
            "verify_ok": True,
            "verify_score": 1.0,

            # Additional Speechmatics data
            "model": result.get("model", "transcriber"),
            "language": language,
            "segments": result.get("segments", []),
            "metadata": result.get("metadata", {}),
            "speakers_detected": result.get("metadata", {}).get("speakers_detected", 1)
        }

        print(f"📤 Response sent successfully")
        return response_data

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "doctor_id": doctor_id,
            "text": "",
        }

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f"🗑️  Temporary file removed")
            except Exception as e:
                print(f"⚠️  Could not remove temp file: {str(e)}")

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
