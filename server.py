"""FastAPI server wrapping the Perch 2 bird sound classifier.

Exposes three endpoints:
  GET  /health  — liveness check
  GET  /labels  — ordered list of ebird codes matching logit indices
  POST /logits  — send raw WAV bytes, receive classification logits
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

import io
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from perch_hoplite.zoo import hub
from perch_hoplite.zoo.taxonomy_model_tf import TaxonomyModelTF
from ml_collections import config_dict

# ---------------------------------------------------------------------------
# Shared state (populated at startup)
# ---------------------------------------------------------------------------
_state: dict = {}

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

    # class_list is a dict[str, ClassList]; "labels" is the species head.
    # ClassList.classes is an ordered tuple of scientific names.
    _state["model"] = model
    _state["labels"] = list(model.class_list["labels"].classes)

    print(f"Perch 2 server ready. {len(_state['labels'])} classes.")
    yield
    # --- shutdown: nothing to clean up ---


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def labels():
    """Return the ordered list of ebird codes corresponding to logit indices."""
    return _state["labels"]


@app.post("/logits")
async def logits(request: Request):
    """Accept a mono WAV clip as raw bytes (Content-Type: application/octet-stream).

    The clip should be 5 seconds at 32 kHz. Other sample rates are resampled.
    Returns {"logits": [[float, ...]]} with shape [1, n_classes].
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

    outputs = _state["model"].embed(audio)

    # outputs.logits is a dict e.g. {"label": array[frames, n_classes]}
    # Note: class_list key is "labels" (plural) but inference key is "label" (singular).
    # Average over frames to get one vector per clip.
    label_logits = outputs.logits["label"].mean(axis=0)

    return {"logits": [label_logits.tolist()]}


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

    outputs = _state["model"].embed(audio)

    # outputs.embeddings shape: [frames, channels, embed_dim]
    # Average over frames and channels to get one vector per clip.
    embedding = outputs.embeddings.mean(axis=(0, 1))

    return {"embedding": [embedding.tolist()]}
