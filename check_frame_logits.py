"""Checks for the /frame_logits endpoint; run on its own in a Slurm job.

Usage:
  .venv/bin/python check_frame_logits.py [clips.npz]

Loads the server's model (which runs the head self-check at startup), then:
  - optional: for each clip in clips.npz (array `clips` [N, 160000]), compares the
    head with the max over (time, frequency) against /logits' path (`batch_embed`);
  - places tone bursts at known times and prints which steps light up;
  - calls /frame_logits and checks the response shape.
"""

import asyncio
import base64
import sys
import time

import numpy as np

import server

TOL = 1e-2


async def main() -> None:
    async with server.lifespan(server.app):
        model = server._state["model"]
        n_classes = len(server._state["labels"])
        n_frames = server._state["frame_logits_shape"][0]

        if len(sys.argv) > 1:
            data = np.load(sys.argv[1])
            clips, names = data["clips"].astype(np.float32), data["names"]
            label = model.batch_embed(clips).logits["label"].mean(axis=1)
            outputs = server._run_model(clips)
            head = server._head_logits(outputs["spatial_embedding"], pool_time=True).numpy()
            diffs = np.abs(head - label).max(axis=1)
            for name, d in zip(names, diffs):
                print(f"clip {name}: max |head(max over t,f) - /logits| = {d:.3e}")
            assert diffs.max() <= TOL, f"real-clip check failed: {diffs.max():.3e}"

        sr, n = server.SAMPLE_RATE, server.WINDOW_SAMPLES
        t = np.arange(n) / sr
        rng = np.random.default_rng(1)
        for lo, hi in [(1.0, 1.5), (3.5, 4.0)]:
            audio = 0.02 * rng.standard_normal(n)
            audio += np.where((t >= lo) & (t < hi), 0.5 * np.sin(2 * np.pi * 3000 * t), 0.0)
            body = server.LogitsRequest(audio=base64.b64encode(audio.astype(np.float32).tobytes()).decode())
            resp = server.frame_logits(body)
            logits = np.frombuffer(base64.b64decode(resp["frame_logits"]), dtype=np.float32).reshape(resp["shape"])
            assert resp["shape"] == [1, n_frames, n_classes], resp["shape"]
            top = logits[0].max(axis=0).argmax()
            step = 5.0 / n_frames
            expected = [i for i in range(n_frames) if i * step < hi and (i + 1) * step > lo]
            per_step = logits[0, :, top]
            lit = [int(i) for i in np.flatnonzero(per_step > np.median(per_step) + 0.5 * (per_step.max() - np.median(per_step)))]
            print(f"tone {lo}-{hi} s: top class {server._state['labels'][top]}; lit steps {lit}, expected {expected}")
            print("  per-step logits:", np.round(per_step, 2).tolist())

        batch = rng.standard_normal((8, n)).astype(np.float32)
        body = server.LogitsRequest(audio=base64.b64encode(batch.tobytes()).decode())
        server.frame_logits(body)  # warm-up
        start = time.perf_counter()
        resp = server.frame_logits(body)
        elapsed = time.perf_counter() - start
        assert resp["shape"] == [8, n_frames, n_classes], resp["shape"]
        print(f"shape {resp['shape']} OK; batch of 8 windows in {elapsed:.2f} s ({8 * 5 / elapsed:.0f}x real time)")


asyncio.run(main())
