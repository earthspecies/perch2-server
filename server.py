"""FastAPI server wrapping the Perch 2 bird sound classifier.

Exposes these endpoints:
  GET  /        — active configuration (labels, sample_rate, window_samples)
  GET  /health  — liveness check
  GET  /labels  — ordered list of scientific names matching logit indices
  POST /logits  — send {"audio": "<base64 float32 PCM bytes>"}, receive classification logits
  POST /embed   — send raw WAV bytes, receive backbone embedding

The model is downloaded automatically from Kaggle on first run and cached
locally by kagglehub. No separate labels file is needed — the class list is
loaded from the CSV files that ship with the model checkpoint.

Deploy with:
  uvicorn server:app --host 0.0.0.0 --port 8000

Environment variables:
  PERCH2_VERSION   integer version of the Perch v2 model on Kaggle (default: 1)
  PERCH2_CPU       set to "1" to load the CPU-only model variant (for machines without GPU)
"""

import base64
import io
import os
import threading
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from perch_hoplite.zoo import hub
from perch_hoplite.zoo.taxonomy_model_tf import TaxonomyModelTF
from ml_collections import config_dict

# ---------------------------------------------------------------------------
# Shared state (populated at startup)
# ---------------------------------------------------------------------------
_state: dict = {}
# Sync endpoints run in a thread pool. Concurrent TF calls were seen to mix up
# tensors between requests, so the model runs one request at a time.
_model_lock = threading.Lock()

SAMPLE_RATE = 32_000
WINDOW_SAMPLES = 5 * SAMPLE_RATE  # 160 000


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    version = int(os.environ.get("PERCH2_VERSION", "1"))
    use_cpu = os.environ.get("PERCH2_CPU", "0") == "1"
    tfhub_path = hub.PERCH_V2_CPU_SLUG if use_cpu else hub.PERCH_V2_SLUG
    cfg = config_dict.ConfigDict({
        "model_path": "",
        "sample_rate": SAMPLE_RATE,
        "window_size_s": 5.0,
        "hop_size_s": 5.0,
        "target_peak": 0.25,
        "tfhub_version": version,
        "tfhub_path": tfhub_path,
    })
    model = TaxonomyModelTF.from_tfhub(cfg)
    # Every path (batch_embed, embed) normalises through this method.
    model.normalize_audio = _normalize_audio

    # class_list is a dict[str, ClassList]; "labels" is the species head.
    # ClassList.classes is an ordered tuple of scientific names.
    _state["model"] = model
    _state["labels"] = list(model.class_list["labels"].classes)

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    device = f"{len(gpus)}x GPU" if gpus else "CPU"
    print(f"Perch 2 server ready. {len(_state['labels'])} classes on {device}.")
    yield
    # --- shutdown: nothing to clean up ---


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_audio(framed_audio: np.ndarray, target_peak: float | None) -> np.ndarray:
    """Hoplite's `normalize_audio`, but an all-zero clip stays zero.

    Hoplite calls `np.divide(..., where=peak > 0)` with no `out=`, so the rows it
    skips hold whatever memory numpy hands back (values up to ~1e18 were seen).
    """
    if target_peak is None:
        return framed_audio
    framed_audio = framed_audio - np.mean(framed_audio, axis=-1, keepdims=True)
    peak_norm = np.max(np.abs(framed_audio), axis=-1, keepdims=True)
    framed_audio = np.divide(framed_audio, peak_norm, out=np.zeros_like(framed_audio), where=(peak_norm > 0.0))
    return framed_audio * target_peak


def _decode_wav(body: bytes) -> np.ndarray:
    """Decode WAV bytes → mono float32 numpy array resampled to SAMPLE_RATE."""
    audio, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)  # stereo → mono
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def describe():
    """Return the active configuration of this adapter (used by HttpClient.describe())."""
    return {
        "labels": _state["labels"],
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def labels():
    """Return the ordered list of scientific names corresponding to logit indices."""
    return _state["labels"]


class LogitsRequest(BaseModel):
    audio: str  # base64-encoded float32 PCM bytes


@app.post("/logits")
def logits(body: LogitsRequest):
    """Accept a batch of mono audio clips as base64-encoded float32 PCM bytes.

    The decoded bytes must be N * WINDOW_SAMPLES * 4 bytes: N clips concatenated,
    each exactly WINDOW_SAMPLES float32 samples (5 s at 32 kHz, little-endian).
    Batch size N is inferred from the decoded payload length.
    Returns {"logits": [[float, ...]]} with shape [N, n_classes].
    """
    try:
        raw = base64.b64decode(body.audio)
    except Exception:
        raise HTTPException(status_code=422, detail="audio field is not valid base64.")

    n_clips, remainder = divmod(len(raw), WINDOW_SAMPLES * 4)
    if n_clips == 0 or remainder != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Decoded audio length {len(raw)} B is not a positive multiple of "
                f"{WINDOW_SAMPLES * 4} B (1 clip = {WINDOW_SAMPLES} float32 samples)."
            ),
        )

    audio_batch = np.frombuffer(raw, dtype=np.float32).reshape(n_clips, WINDOW_SAMPLES)

    with _model_lock:
        outputs = _state["model"].batch_embed(audio_batch)

    # outputs.logits["label"] shape: [B, n_frames, n_classes]
    # With window_size == hop_size == 5 s, n_frames == 1; mean over axis=1 is safe generally.
    label_logits = outputs.logits["label"].mean(axis=1)  # [B, n_classes]

    return {"logits": label_logits.tolist()}


@app.post("/embed")
async def embed(request: Request):
    """Same input as /logits. Returns {"embedding": [[float, ...]]} — the
    backbone embedding before the classification head, shape [1, embed_dim].
    """
    body = await request.body()
    try:
        audio = _decode_wav(body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not decode audio: {e}")

    if len(audio) != WINDOW_SAMPLES:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {WINDOW_SAMPLES} samples (5 s @ 32 kHz), got {len(audio)}.",
        )

    with _model_lock:
        outputs = _state["model"].embed(audio)

    # outputs.embeddings shape: [frames, channels, embed_dim]
    # Average over frames and channels to get one vector per clip.
    embedding = outputs.embeddings.mean(axis=(0, 1))

    return {"embedding": [embedding.tolist()]}
