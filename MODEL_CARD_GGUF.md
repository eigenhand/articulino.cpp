---
language:
- de
- en
license: apache-2.0
pipeline_tag: text-to-speech
library_name: ggml
base_model: eigenhand/Articulino-DE-EN-Realtime-Stream-TTS
tags:
- text-to-speech
- tts
- gguf
- ggml
- german
- english
- voice-cloning
- zero-shot
- cpu
- quantized
---

# Articulino-DE-EN-Realtime-Stream-TTS-GGUF

**GGUF files of [Articulino-DE-EN-Realtime-Stream-TTS](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS)
for [articulino.cpp](https://github.com/eigenhand/articulino.cpp): German and English text to speech with zero-shot voice cloning, in C++, without
Python: 33 times faster than real time on 12 CPU threads with the recommended 8-bit file, 135 times with Vulkan.**

Each file holds the complete model in one piece: text encoder, duration predictor, voice encoder, the 4-step DMD2
student DiT, the latent encoder for the voice reference, and the decoder (Vocos-style, inverse STFT), with all settings
as GGUF metadata. The weights are the same as in the main repository; only their storage format and precision differ.

| File | Size | Weights | Speed, 12 CPU threads¹ | Notes |
|---|---|---|---|---|
| `articulino-q8_0.gguf` | 137 MB | 8 bit (Q8_0) | 33× | **recommended**: equivalent to F32 in every measure, a third of the size |
| `articulino-f16.gguf` | 220 MB | 16 bit float | 26× | half the download of F32, computed in F32 after loading |
| `articulino-f32.gguf` | 438 MB | 32 bit float | 26× | reference; matches PyTorch to float rounding |

¹ Times real time for the whole pipeline per sentence, on 12 efficiency cores of an Intel Core i9-14900K (in a VM).
The PyTorch release reaches 24.9× on the same cores. With 2 threads: Q8_0 6.9×, F32 6.0×
(PyTorch 8.3×). There are no 4-bit files: they were understood as well as F32 but sounded measurably worse
(UTMOS 0.05–0.07 lower), so 8 bits is the smallest precision released.

| File | 12 threads | 8 threads | 4 threads | 2 threads | First sentence (12 threads) |
|---|---|---|---|---|---|
| F32 | 25.9× | 20.7× | 11.5× | 6.0× | 0.18 s |
| F16 | 25.7× | 20.3× | 11.5× | 6.0× | 0.18 s |
| Q8_0 | 32.7× | 24.8× | 13.3× | 6.9× | 0.14 s |
| PyTorch F32, for comparison | 24.9× | 20.6× | 12.9× | 8.3× | 0.20 s |
| Any file, Vulkan on an AMD Radeon AI PRO R9700 | 135–138× | | | | 0.03 s |

In the Q8_0 file, norms, biases, embeddings and the voice encoder (which runs once per voice) keep 16 or 32 bits.
Everything that runs per sentence uses the file's type.

## Quickstart

```bash
git clone --recursive https://github.com/eigenhand/articulino.cpp
cmake -S articulino.cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
hf download eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-GGUF articulino-q8_0.gguf --local-dir .

build/articulino-cli -m articulino-q8_0.gguf --voice voice.wav --threads 8 \
    -t "Guten Morgen! Der Termin ist heute um fünfzehn Uhr dreißig." -o morgen.wav
```

`voice.wav` is 3 to 8 seconds of clean speech from someone who agreed to this (24 kHz mono is best; other rates are
resampled). Streaming from a language model:

```bash
my-llm | build/articulino-cli -m articulino-q8_0.gguf --voice voice.wav --stdin -o - | aplay -r 24000 -f S16_LE -c 1
```

Every sentence is spoken as soon as it is complete. The engine also has a C API (`articulino.h`) with a per-sentence
callback and a streaming text input, and an optional Vulkan GPU path (`--gpu`); see the engine's README.

**Text must be spelled out.** The engine lower-cases, removes accents and keeps the 52 letters and punctuation marks the
model knows. It does not turn digits, dates, times, money, units or abbreviations into words; the German text frontend
of the Python release does that. Words in the other language can be marked: `Das [en]Meeting[/en] ist verschoben.`

## Quality of the quantized files

**Unseen speakers.** Every file spoke the test sets of the main card's evaluation with **the same noise** as the
PyTorch reference: 320 German utterances from 160 speakers and 107 English utterances from 4 speakers that the model
never saw (three seeds each), and the 129-sentence German pronunciation suite. Whisper large-v3 (beam 5) for WER and
CER, ECAPA cosine to the voice prompt for similarity, UTMOS for quality; 95 % bootstrap intervals over utterances.

| German (320) | WER | CER | Similarity | UTMOS |
|---|---|---|---|---|
| PyTorch reference | 4.72 % [3.90, 5.59] | 1.84 % [1.46, 2.24] | 0.659 [0.649, 0.669] | 3.30 [3.27, 3.34] |
| F32 | 4.72 % [3.90, 5.59] | 1.84 % [1.46, 2.24] | 0.659 [0.649, 0.669] | 3.30 [3.27, 3.34] |
| F16 | 4.70 % [3.87, 5.56] | 1.83 % [1.45, 2.24] | 0.659 [0.649, 0.669] | 3.30 [3.27, 3.34] |
| **Q8_0** | **4.67 %** [3.85, 5.50] | **1.79 %** [1.42, 2.19] | **0.659** [0.649, 0.669] | **3.30** [3.26, 3.33] |
| F32, Vulkan (R9700) | 4.75 % [3.92, 5.60] | 1.81 % [1.44, 2.21] | 0.659 [0.649, 0.670] | 3.30 [3.27, 3.34] |

| English (107) | WER | CER | Similarity | UTMOS |
|---|---|---|---|---|
| PyTorch reference | 2.12 % [1.40, 2.92] | 0.85 % [0.55, 1.19] | 0.590 [0.577, 0.603] | 4.28 [4.26, 4.30] |
| F32 | 2.13 % [1.42, 2.94] | 0.85 % [0.55, 1.19] | 0.590 [0.577, 0.603] | 4.28 [4.26, 4.30] |
| F16 | 2.09 % [1.38, 2.89] | 0.85 % [0.54, 1.20] | 0.590 [0.577, 0.603] | 4.28 [4.26, 4.30] |
| **Q8_0** | **2.25 %** [1.52, 3.08] | **0.91 %** [0.58, 1.27] | **0.591** [0.578, 0.604] | **4.27** [4.25, 4.29] |

| Pronunciation suite (129) | CER, Whisper, 3 seeds | Correct in all seeds | Parakeet (seed 0): correct |
|---|---|---|---|
| PyTorch reference | 1.41 % | 116 | 122 |
| F32 / F16 | 1.41 % / 1.41 % | 116 / 116 | 122 / 123 |
| Q8_0 | 1.27 % | 118 | 121 |

**Equivalence to the reference.** A file counts as equivalent when the 90 % interval of every paired difference lies
inside the margins fixed before the measurement (±1.0 point WER, ±0.5 point CER, ±0.01 similarity, ±0.05 UTMOS);
"different" means a significant paired difference (Wilcoxon, Holm-corrected).

| File | ΔWER German (points) | ΔSimilarity German | ΔUTMOS German | German | English | Suite |
|---|---|---|---|---|---|---|
| F32 | 0.00 [0.00, 0.00] | 0.000 | 0.000 | equivalent | equivalent | equivalent |
| F16 | −0.02 [−0.08, 0.02] | 0.000 | −0.002 | equivalent | equivalent | equivalent |
| **Q8_0** | −0.06 [−0.26, 0.15] | 0.000 | −0.004 | **equivalent** | **equivalent** | **equivalent** |
| F32, Vulkan | +0.02 [−0.08, 0.13] | 0.000 | −0.000 | equivalent | equivalent | equivalent |

**What this means.** F16, Q8_0 and the Vulkan path speak like F32 on every set and every measure, so Q8_0 is the
recommended file. 4-bit versions (Q4_0, IQ4_NL) went through the same evaluation and were dropped: they were
understood as well as F32, but German UTMOS was 0.05–0.07 lower (significant after Holm correction), and Q4_0 also lost
some speaker similarity.

**Fidelity.** Median log-mel distance to the reference output; another seed of the reference itself differs by
5.9 dB (German), 5.5 dB (English) and 5.5 dB (suite).

| File | Log-mel distance, German / English / suite | Same number of frames (German) |
|---|---|---|
| F32 | 0.00 / 0.00 / 0.00 dB | 100 % |
| F16 | 0.07 / 0.10 / 0.06 dB | 100 % |
| Q8_0 | 1.34 / 1.42 / 1.24 dB | 81 % |
| F32, Vulkan (R9700) | 0.40 / 0.48 / 0.35 dB | 94 % |

**Building blocks.** Same inputs, signal-to-noise ratio of one DiT step against float32: F16 69 dB, Q8_0 36 dB; the
decoder's spectrum deviates by 0.002 / 0.13 dB on average. The full tables
are in the engine's README.

The F32 file reproduces the PyTorch reference: every building block within 1e-5 relative error, and the whole chain
from voice WAV and text to audio at 73 dB SNR with the same noise.

## Model

| Part | Parameters | Runs |
|---|---|---|
| Text encoder (letters → features, 6 layers, d 512) | 18.9 M | once per sentence |
| Duration predictor | 1.9 M | once per sentence |
| Voice encoder (32 voice tokens + voice vector) | 20.4 M | once per voice |
| Student DiT (8 blocks, d 384, 6 heads) | 25.1 M | 4× per sentence |
| Latent encoder (24 kHz → 96-dim latents at 25 Hz) | 21.5 M | once per voice |
| Decoder (latents → magnitude/phase → inverse STFT) | 21.7 M | once per sentence |

The first 5 seconds of the voice reference sit in front of every sentence as a prompt that attends only to itself, so
its keys and values are computed once per voice (the voice cache). Details on design, training data, training and
evaluation are in the [main model card](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS).

## Limitations

- About one in fifteen renderings has a small slip (a word swallowed or blurred); another seed usually fixes it.
- English is limited (51 h of audiobooks); German is the main language.
- Letters, not phonemes: names and foreign words can be read wrongly. Numbers must be spelled out first.
- Streaming is per sentence; sentences over 180 characters are split at a comma.
- The engine is made for CPUs; its Vulkan path (about 136× real time on an AMD Radeon AI PRO R9700) is tested on one
  GPU only.
- The engine's random generator differs from PyTorch's: the same seed gives a different rendering than the Python
  release.

## Responsible use

Articulino can copy a voice from a few seconds of audio. Only clone voices with the speaker's consent, tell listeners
that they hear synthetic speech, and do not use it to impersonate anyone, for fraud, or to get around voice
authentication. In the EU, Article 50 of the AI Act requires machine-readable marking of synthetic audio by providers
of such systems; these weights contain no watermark. See "Responsible use" in the main model card.

## License and attribution

Apache License 2.0 for the weights in these files and for articulino.cpp. The model was trained on corpora under CC0,
CC BY 4.0 and public domain; the CC BY 4.0 attribution (Emilia-YODAS, Hi-Fi TTS 2, CML-TTS) is in `NOTICE` and must be
kept when redistributing. articulino.cpp uses ggml (MIT) and pocketfft (BSD 3-Clause).

```bibtex
@misc{articulino2026,
  title        = {Articulino-DE-EN-Realtime-Stream-TTS: Small German and English Text to Speech with Zero-Shot Voice Cloning},
  author       = {eigenhand},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS}}
}
```
