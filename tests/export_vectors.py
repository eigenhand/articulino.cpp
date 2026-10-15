#!/usr/bin/env python3
"""Convert the reference vectors (tests/vectors/<name>.npz, read only) into a GGUF container that the C++ parity
test reads, plus the sampler noise as .npy (4, frames, 96) for `articulino-cli --noise`.

    python tests/export_vectors.py tests/vectors/de_sentence.npz build/vectors/
"""
import json
import sys
from pathlib import Path

import numpy as np
import gguf

src, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
z = np.load(src)
meta = json.loads(src.with_suffix(".json").read_text(encoding="utf-8"))
w = gguf.GGUFWriter(out_dir / f"{src.stem}.vectors.gguf", "articulino-vectors")
w.add_string("vectors.sentence", meta["sentence"])
w.add_string("vectors.language", meta["language"])
w.add_uint32("vectors.frames_total", meta["frames_total"])
w.add_uint32("vectors.frames_new", meta["frames_new"])
for k in z.files:
    a = z[k]
    if a.dtype == np.bool_:
        a = a.astype(np.float32)
    if np.issubdtype(a.dtype, np.integer):
        a = a.astype(np.int32)
    else:
        a = a.astype(np.float32)
    w.add_tensor(k, np.ascontiguousarray(a))
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
noise = np.stack([z[f"noise_{k}"] for k in range(4)]).astype(np.float32)
np.save(out_dir / f"{src.stem}.noise.npy", noise)
print(out_dir / f"{src.stem}.vectors.gguf", noise.shape)
