#!/usr/bin/env python3
"""Speaker similarity (ECAPA-TDNN, speechbrain spkrec-ecapa-voxceleb, CPU) of generated files to the voice reference,
for comparing weight types. Needs speechbrain; the model is downloaded on first use (or set ECAPA_DIR to a local copy).

    python similarity.py voice.wav out/asr  ->  mean cosine per model prefix (files <model>-<n>.wav)
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import librosa
from speechbrain.inference.speaker import EncoderClassifier

torch.set_num_threads(4)
ECAPA = os.environ.get("ECAPA_DIR", "speechbrain/spkrec-ecapa-voxceleb")
spk = EncoderClassifier.from_hparams(source=ECAPA, savedir=os.environ.get("ECAPA_DIR", "pretrained/spkrec-ecapa"),
                                     run_opts={"device": "cpu"})


def emb(p):
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    with torch.no_grad():
        e = spk.encode_batch(torch.from_numpy(x)[None])[0, 0].numpy()
    return e / np.linalg.norm(e)


ref = emb(sys.argv[1])
groups = defaultdict(list)
for f in sorted(Path(sys.argv[2]).glob("*.wav")):
    if f.name.endswith(".asr.wav"):
        continue
    groups[f.stem.rsplit("-", 1)[0]].append(float(emb(f) @ ref))
for k, v in groups.items():
    print(f"{k:28s} similarity to the voice {np.mean(v):.3f}  ({' '.join(f'{x:.3f}' for x in v)})")
