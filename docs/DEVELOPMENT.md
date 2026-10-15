# Development notes: C++ / GGUF

State of 15 October 2026. The computation follows the inference specification in the PyTorch repository
([docs/SPEC.md](https://github.com/eigenhand/articulino/blob/main/docs/SPEC.md)).

## What is there

- **ggml** as a submodule in `third_party/ggml`, pinned to v0.26.0 (d7cb5741), linked statically, without OpenMP (own
  thread pool). **pocketfft** (C++ header, BSD-3) unchanged in `third_party/pocketfft` (commit c90e55b3). Licenses:
  `LICENSE` (Apache 2.0), `THIRD_PARTY_NOTICES.md` (ggml MIT, pocketfft BSD-3).
- **Converter** `tools/convert_to_gguf.py`: the two training checkpoints → one GGUF, 594 tensors, 109.5 M parameters;
  metadata `articulino.*` with every size, latent mean/std, timesteps, character set, prompt 125 frames, context 12
  frames, 24 kHz, post-processing and the calibration sentences. Types f32, f16, q8_0 (and q5_0, not evaluated);
  `articulino-quantize` (C++, ggml's quantizer) adds the K-quants q6_k and q5_k (not evaluated).
- **Engine** (`src/`): log-mel, latent encoder, voice encoder, voice cache (prompt keys/values, cross-attention
  keys/values and the AdaNorm parameters of all four timesteps precomputed per voice), text encoder and durations,
  rate equalization (on by default), frame expansion, the four DiT steps in one graph, decoder and inverse STFT, shelf
  +2 dB at 180 Hz and peak 0.95, outlier/whistle detection with one retry (seed + 1, like the original service),
  character normalization (a table generated from the Python code for every code point), sentence splitting,
  `[en]` markup, resampling, own random numbers (xoshiro256** + Box–Muller) and a test mode with noise from `.npy`.
- **C API** `include/articulino.h` (load, set voice, speak a sentence or text, per-sentence callback, text
  streaming), example `examples/stream_demo.c`, **CLI** `articulino-cli` (also `--stdin` and `-o -` for pipes,
  `--bench`, `--gpu`).
- **CPU tuning**: fused operations (depthwise convolution, GELU with a vectorized erf using PyTorch's formula), fused
  QKV, F16 widened to F32 at load time.
- **Vulkan** (optional, `-DARTICULINO_VULKAN=ON`, `--gpu`): the whole graph on the GPU through ggml's scheduler,
  tested on an AMD Radeon AI PRO R9700 (RADV), one split per graph.

## Equality (F32, `test-parity`, max|Δ| / max|ref| per block)

German / English: log-mel 1.6e-5; mu 5.5e-7; tokens 1.1e-6; vector 3.6e-7; cache k/v 6.1e-6 / 1.1e-6; text 3.3e-7 /
4.7e-7; log durations 5.1e-7 / 4.7e-7; frame expansion 7e-8; DiT steps ≤ 5.6e-6; four chained steps 1.7e-5 / 7.6e-6;
decoder 5.1e-6 / 2.0e-6 (121 / 119 dB); the whole chain from WAV and text with the reference noise: audio SNR
73.3 / 74.4 dB. Normalizer and sentence splitting: 22 of 22 cases identical to Python (`tests/check_text.py`).

## Precisions (SNR per block in dB, German; decoder: waveform SNR / mean log-mel deviation in dB)

| | F16 | Q8_0 | Vulkan F32 |
|---|---|---|---|
| mu | 73 | 47 | 66 |
| tokens | 63 | 63 | 60 |
| cache k | 76 | 43 | 72 |
| text | 77 | 44 | 57 |
| log duration | 81 | 47 | 64 |
| DiT step | 69 | 36 | 65 |
| 4 steps | 65 | 30 | 62 |
| decoder | 67 / 0.002 | 32 / 0.13 | 63 / 0.004 |

In the quantized files the voice encoder (`voice.*`) and `enc.mu` stay F16 (`enc.mu` has outlier inputs: 31 dB at
Q8_0). Measured with `tools/sensitivity.sh`, which quantizes one group of weights at a time.

## Listening checks

Parakeet-TDT 0.6B v3 (0.4 s of silence in front), four German sentences, seed 0, ECAPA similarity to the voice:
CER 0.0 % for F32, F16, Q8_0 and Vulkan F32; ECAPA 0.617 / 0.616 / 0.624, Vulkan 0.618 (±0.03 is noise). An English
sentence pair is recognised word for word (F32, Q8_0). A
short voice (3 s) and a 44.1 kHz stereo voice work. The large evaluation on unseen speakers is in
[MODEL_CARD_GGUF.md](../MODEL_CARD_GGUF.md).

## Speed

See the README. Measured next to other jobs on the same cores; `tools/wait_idle.sh` waits for idle cores, best of
three. With few threads ggml's F32 matmul is slower than PyTorch (MKL); at 2 threads F32 and Q8_0 are below PyTorch.

Sizes: f32 438 MB, f16 220 MB, q8_0 137 MB.

## 4-bit (dropped)

Q4_0 and IQ4_NL files (latent encoder, DiT output and decoder head in Q8_0, voice encoder F16) ran at 52× on 12
threads, and in the evaluation on unseen speakers they were understood as well as F32 (WER, CER and the pronunciation
suite unchanged). But German UTMOS was 0.066 (Q4_0) and 0.050 (IQ4_NL) lower, significant after Holm correction and
outside the equivalence margin of ±0.05, and Q4_0 also lost speaker similarity (−0.006 German, −0.014 suite). Their
renderings differed from F32 by 4 dB log-mel (two thirds of a seed change). After the MLX 4-bit version had failed a
listening test, the author decided to drop 4 bit everywhere; the converter and the quantizer no longer offer it.

## Open points

- The full German text frontend (numbers, abbreviations) stays in Python and runs before the engine.
- The same seed does not give the same audio as PyTorch (own random numbers); exact comparisons go through `--noise`.
- Vulkan is tested on one AMD card only; the GPU computes matmuls in fp16 internally (≈ 60–65 dB per block).
- The converter reads the training checkpoints; converting from the released safetensors would make it reproducible
  from public files.
