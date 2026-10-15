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
Python, about 50 times faster than real time on CPU cores.**

Each file holds the complete model in one piece: text encoder, duration predictor, voice encoder, the 4-step DMD2
student DiT, the latent encoder for the voice reference, and the decoder (Vocos-style, inverse STFT), with all settings
as GGUF metadata. The weights are the same as in the main repository; only their storage format and precision differ.

| File | Size | Weights | Speed, 12 CPU threads¹ | Notes |
|---|---|---|---|---|
| `articulino-iq4_nl.gguf` | 104 MB | 4 bit (IQ4_NL) | 52× | fastest on CPUs; renders differ more from F32 (see Quality) |
| `articulino-q4_0.gguf` | 104 MB | 4 bit (Q4_0) | 52× | the classic 4-bit type |
| `articulino-q8_0.gguf` | 137 MB | 8 bit (Q8_0) | 33× | close to F32 at a third of the size |
| `articulino-f16.gguf` | 220 MB | 16 bit float | 26× | half the download of F32, computed in F32 after loading |
| `articulino-f32.gguf` | 438 MB | 32 bit float | 26× | reference; matches PyTorch to float rounding |

¹ Times real time for the whole pipeline per sentence, on 12 efficiency cores of an Intel Core i9-14900K (in a VM).
The PyTorch release reaches 24.9× on the same cores. With 2 threads: IQ4_NL 11.2×, Q8_0 6.9×, F32 6.0×
(PyTorch 8.3×).

| File | 12 threads | 8 threads | 4 threads | 2 threads | First sentence (12 threads) |
|---|---|---|---|---|---|
| F32 | 25.9× | 20.7× | 11.5× | 6.0× | 0.18 s |
| F16 | 25.7× | 20.3× | 11.5× | 6.0× | 0.18 s |
| Q8_0 | 32.7× | 24.8× | 13.3× | 6.9× | 0.14 s |
| Q4_0 | 52.0× | 38.9× | 21.5× | 11.4× | 0.08 s |
| IQ4_NL | 52.4× | 39.3× | 21.6× | 11.2× | 0.09 s |
| PyTorch F32, for comparison | 24.9× | 20.6× | 12.9× | 8.3× | 0.20 s |
| Any file, Vulkan on an AMD Radeon AI PRO R9700 | 135–138× | | | | 0.03 s |

In the quantized files, norms, biases, embeddings and the voice encoder (which runs once per voice) keep 16 or 32 bits;
in the 4-bit files the latent encoder, the DiT output layer and the decoder head keep 8 bits. Everything that runs per
sentence uses the file's type.

## Quickstart

```bash
git clone --recursive https://github.com/eigenhand/articulino.cpp
cmake -S articulino.cpp -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
huggingface-cli download eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-GGUF articulino-iq4_nl.gguf --local-dir .

build/articulino-cli -m articulino-iq4_nl.gguf --voice voice.wav --threads 8 \
    -t "Guten Morgen! Der Termin ist heute um fünfzehn Uhr dreißig." -o morgen.wav
```

`voice.wav` is 3 to 8 seconds of clean speech from someone who agreed to this (24 kHz mono is best; other rates are
resampled). Streaming from a language model:

```bash
my-llm | build/articulino-cli -m articulino-iq4_nl.gguf --voice voice.wav --stdin -o - | aplay -r 24000 -f S16_LE -c 1
```

Every sentence is spoken as soon as it is complete. The engine also has a C API (`articulino.h`) with a per-sentence
callback and a streaming text input, and an optional Vulkan GPU path (`--gpu`); see the engine's README.

**Text must be spelled out.** The engine lower-cases, removes accents and keeps the 52 letters and punctuation marks the
model knows. It does not turn digits, dates, times, money, units or abbreviations into words; the German text frontend
of the Python release does that. Words in the other language can be marked: `Das [en]Meeting[/en] ist verschoben.`

## Quality of the quantized files

**Unseen speakers.** Every file spoke the test sets of the main card's evaluation with **the same noise** as the
PyTorch reference: 320 German utterances from 160 speakers and 107 English utterances from 4 speakers that the model
never saw (three seeds each), and the 129-sentence German pronunciation suite. Fidelity is the median log-mel
distance to the reference output; for scale, another seed of the reference itself differs by 5.9 dB (German), 5.5 dB
(English) and 5.5 dB (suite). "Same length" is the share of utterances with exactly the same number of frames.

| File | Log-mel distance, German / English / suite | Same length (German) | Pronunciation suite, Parakeet: sentences correct | Suite CER |
|---|---|---|---|---|
| PyTorch reference | 0 | 100 % | 122 / 129 | 2.29 % |
| F32 | 0.00 / 0.00 / 0.00 dB | 100 % | 122 / 129 | 2.29 % |
| F16 | 0.07 / 0.10 / 0.06 dB | 100 % | 123 / 129 | 2.16 % |
| Q8_0 | 1.34 / 1.42 / 1.24 dB | 81 % | 121 / 129 | 2.22 % |
| Q4_0 | 4.10 / 3.93 / 3.94 dB | 24 % | 118 / 129 | 1.97 % |
| IQ4_NL | 4.03 / 4.12 / 3.78 dB | 28 % | 122 / 129 | 1.42 % |
| F32, Vulkan (R9700) | 0.40 / 0.48 / 0.35 dB | 94 % | 122 / 129 | 2.01 % |
| IQ4_NL, Vulkan (R9700) | 4.04 / 4.04 / 3.80 dB | 23 % | 123 / 129 | 1.16 % |

F32 and F16 give the reference output. Q8_0 changes it a little (a quarter of a seed change). The two 4-bit files
render most sentences noticeably differently (two thirds of a seed change, other phrase lengths in three of four
utterances), while the pronunciation suite stays as correct as with F32. **Word error rate, speaker similarity and
UTMOS of these files on the unseen speakers are being measured** with the same protocol as the ONNX and PyTorch
variants (which were all equivalent to the reference); this card will be updated with the results. Until then, use
Q8_0 or F16 where the voice must match F32 exactly, and listen before choosing a 4-bit file.

**Building blocks.** Same inputs, signal-to-noise ratio of one DiT step against float32: F16 69 dB, Q8_0 36 dB,
Q4_0 18 dB, IQ4_NL 19 dB; the decoder's spectrum deviates by 0.002 / 0.13 / 0.60 / 0.53 dB on average. The full tables
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
