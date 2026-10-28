"""FastAPI server wrapping the Perch 2 bird sound classifier.

Exposes these endpoints:
  GET  /        — active configuration (labels, sample_rate, window_samples)
  GET  /health  — liveness check
  GET  /labels  — ordered list of scientific names matching logit indices
  POST /logits  — send {"audio": "<base64 float32 PCM bytes>"}, receive classification logits
  POST /frame_logits — same input as /logits, receive per-time-step logits
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
    # Every path (batch_embed, embed, _run_model) normalises through this method.
    model.normalize_audio = _normalize_audio

    # class_list is a dict[str, ClassList]; "labels" is the species head.
    # ClassList.classes is an ordered tuple of scientific names.
    _state["model"] = model
    _state["labels"] = list(model.class_list["labels"].classes)
    _load_head(model)

    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    device = f"{len(gpus)}x GPU" if gpus else "CPU"
    print(f"Perch 2 server ready. {len(_state['labels'])} classes on {device}.")
    yield
    # --- shutdown: nothing to clean up ---


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# Prototype head, applied per time step
# ---------------------------------------------------------------------------

# Checkpoint leaves holding the head's weights. The numbers are positions, not
# names, so `_load_head` checks the shapes and runs a self-check against the
# model's own `label` output.
_HEAD_LEAVES = {"bias": 494, "protop_kernel": 495, "prototypes": 496}
_SELF_CHECK_TOL = 1e-2


def _head_logits(spatial, pool_time: bool):
    """Apply the prototype head to `spatial_embedding` [B, T, F, D].

    The head scores each class as sum_k relu(kernel[c, k]) * max sim[c, k] + bias[c],
    where sim is the cosine similarity between a grid vector and prototype k
    of class c. With `pool_time` the max runs over (time, frequency), which is
    the model's own clip logit; without it the max runs over frequency only,
    giving one logit per time step.
    """
    import tensorflow as tf

    # Shapes: B clips, T=16 time steps, F=4 frequency bands, D=1536, C classes, K=4 prototypes.
    n_classes, n_protos = _state["protop_kernel"].shape
    x = tf.math.l2_normalize(spatial, axis=-1)  # [B, T, F, D]
    sims = tf.einsum("btfd,dk->btfk", x, _state["prototypes"])  # [D, C*K] -> [B, T, F, C*K]
    sims = tf.reshape(sims, tf.concat([tf.shape(sims)[:3], [n_classes, n_protos]], 0))  # [B, T, F, C, K]
    sims = tf.reduce_max(sims, axis=[1, 2] if pool_time else 2)  # [B, C, K], or [B, T, C, K] per step
    # [C, K] kernel broadcasts over the leading axes; sum over K -> [B, C] or [B, T, C].
    return tf.reduce_sum(sims * _state["protop_kernel"], axis=-1) + _state["bias"]


def _load_head(model) -> None:
    """Load the prototype head weights and check them against the model's `label` output."""
    import tensorflow as tf

    ckpt = tf.train.load_checkpoint(os.path.join(model.model_path, "variables", "variables"))
    head = {
        name: ckpt.get_tensor(f"_tf_var_leaves/{leaf}/.ATTRIBUTES/VARIABLE_VALUE")
        for name, leaf in _HEAD_LEAVES.items()
    }
    n_classes = len(_state["labels"])
    n_protos = head["protop_kernel"].shape[-1] if head["protop_kernel"].ndim == 2 else -1
    embed_dim = head["prototypes"].shape[1] if head["prototypes"].ndim == 3 else -1
    expected = {
        "bias": (n_classes,),
        "protop_kernel": (n_classes, n_protos),
        "prototypes": (n_classes, embed_dim, n_protos),
    }
    for name, shape in expected.items():
        if head[name].shape != shape:
            raise RuntimeError(f"Head leaf {name} has shape {head[name].shape}, expected {shape}.")

    # [C, D, K] -> L2-normalised over D -> [D, C*K], so all similarities are one matmul.
    protos = head["prototypes"] / np.linalg.norm(head["prototypes"], axis=1, keepdims=True)
    _state["prototypes"] = tf.constant(protos.transpose(1, 0, 2).reshape(embed_dim, n_classes * n_protos))
    # The stored kernel has small negative entries; Perch's head clamps them to 0
    # (`non_negative_kernel=True` in chirp/models/heads.py), so we do the same.
    _state["protop_kernel"] = tf.constant(np.maximum(head["protop_kernel"], 0))
    _state["bias"] = tf.constant(head["bias"])

    # Self-check on seeded noise plus a tone burst.
    t = np.arange(WINDOW_SAMPLES) / SAMPLE_RATE
    audio = 0.1 * np.random.default_rng(0).standard_normal(WINDOW_SAMPLES)
    audio += np.where((t >= 1.0) & (t < 1.5), 0.5 * np.sin(2 * np.pi * 3000 * t), 0.0)
    outputs = _run_model(audio[None].astype(np.float32))
    recon = _head_logits(outputs["spatial_embedding"], pool_time=True).numpy()
    diff = float(np.abs(recon - outputs["label"].numpy()).max())
    if not diff <= _SELF_CHECK_TOL:
        raise RuntimeError(f"Prototype head self-check failed: max |head - label| = {diff:.3e} > {_SELF_CHECK_TOL}.")
    _state["frame_logits_shape"] = list(_head_logits(outputs["spatial_embedding"], pool_time=False).shape[1:])
    print(f"Prototype head self-check passed: max |head - label| = {diff:.3e}; frame logits {_state['frame_logits_shape']}.")


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


def _run_model(audio_batch: np.ndarray) -> dict:
    """Normalise 5 s clips exactly as `batch_embed` does and run the SavedModel."""
    model = _state["model"]
    audio_batch = model.normalize_audio(audio_batch, model.target_peak)
    return model.model.signatures["serving_default"](inputs=audio_batch)


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

@app.get("/")
def describe():
    """Return the active configuration of this adapter (used by HttpClient.describe())."""
    return {
        "labels": _state["labels"],
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "frame_logits_shape": _state["frame_logits_shape"],
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


def _decode_clips(audio: str) -> np.ndarray:
    """Decode base64 float32 PCM into [N, WINDOW_SAMPLES] clips."""
    try:
        raw = base64.b64decode(audio)
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
    return np.frombuffer(raw, dtype=np.float32).reshape(n_clips, WINDOW_SAMPLES)


@app.post("/logits")
def logits(body: LogitsRequest):
    """Accept a batch of mono audio clips as base64-encoded float32 PCM bytes.

    The decoded bytes must be N * WINDOW_SAMPLES * 4 bytes: N clips concatenated,
    each exactly WINDOW_SAMPLES float32 samples (5 s at 32 kHz, little-endian).
    Batch size N is inferred from the decoded payload length.
    Returns {"logits": [[float, ...]]} with shape [N, n_classes].
    """
    audio_batch = _decode_clips(body.audio)

    outputs = _state["model"].batch_embed(audio_batch)

    # outputs.logits["label"] shape: [B, n_frames, n_classes]
    # With window_size == hop_size == 5 s, n_frames == 1; mean over axis=1 is safe generally.
    label_logits = outputs.logits["label"].mean(axis=1)  # [B, n_classes]

    return {"logits": label_logits.tolist()}


@app.post("/frame_logits")
def frame_logits(body: LogitsRequest):
    """Same input as /logits. Returns per-time-step logits from the prototype head.

    Each step's logit takes the best prototype match over frequency only, at that
    time step (the clip logit takes it over time and frequency). Returns
    {"frame_logits": "<base64 float32>", "shape": [N, n_frames, n_classes]}.
    """
    outputs = _run_model(_decode_clips(body.audio))
    logits = _head_logits(outputs["spatial_embedding"], pool_time=False).numpy().astype(np.float32)
    return {"frame_logits": base64.b64encode(logits.tobytes()).decode("ascii"), "shape": list(logits.shape)}


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
