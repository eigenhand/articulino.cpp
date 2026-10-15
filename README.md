# articulino.cpp

C++ inference engine for **Articulino-DE-EN-Realtime-Stream-TTS**, the small German and English text-to-speech model
with zero-shot voice cloning. It runs on [ggml](https://github.com/ggml-org/ggml), loads a single GGUF file, needs no
Python at run time, and is fast on ordinary CPUs: with 4-bit weights it speaks about 50 times faster than real time on
12 efficiency cores.

- the whole pipeline in C++: voice WAV → log-mel → latent encoder → voice tokens and prompt cache; text → letters →
  text encoder → durations → 4-step DiT sampler → Vocos-style decoder → inverse STFT → 24 kHz
- one GGUF file per weight type: F32, F16, Q8_0, Q4_0, IQ4_NL
- C API (`include/articulino.h`) with per-sentence streaming, and a command line tool
- matches the reference PyTorch code to float32 rounding (see [Parity](#parity))

Model weights: [`eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-GGUF`](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS-GGUF).
Model card, training and evaluation: [`eigenhand/Articulino-DE-EN-Realtime-Stream-TTS`](https://huggingface.co/eigenhand/Articulino-DE-EN-Realtime-Stream-TTS).

## Build

Requirements: a C++17 compiler, CMake ≥ 3.19, git. Nothing else; ggml is a pinned git submodule (v0.26.0) and the FFT
(pocketfft) is vendored.

```bash
git clone --recursive https://github.com/eigenhand/articulino.cpp
cd articulino.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This builds `build/articulino-cli`, `build/articulino-quantize`, the shared library `build/libarticulino.so` (only the C
API is exported; ggml is linked in statically) and the tests.

| CMake option | Default | Meaning |
|---|---|---|
| `GGML_NATIVE` / `ARTICULINO_NATIVE` | `ON` | optimize for the build machine (`-march=native`). Turn both off for portable binaries; keep `GGML_AVX2=ON` for any x86-64 CPU from 2013 on |
| `GGML_OPENMP` | `OFF` | ggml's own thread pool (no libgomp dependency); OpenMP was not faster here |
| `ARTICULINO_BUILD_SHARED` | `ON` | build `libarticulino` |
| `ARTICULINO_VULKAN` | `OFF` | also build ggml's Vulkan backend; `--gpu` / `use_gpu` then runs the graphs on the GPU (see [GPU](#gpu-vulkan)) |

## Command line

```bash
# a text with several sentences -> WAV
articulino-cli -m articulino-q8_0.gguf --voice voice.wav \
    -t "Guten Morgen! Der Termin ist heute um fünfzehn Uhr dreißig." -o out.wav --threads 8

# English
articulino-cli -m articulino-q8_0.gguf --voice voice.wav --lang en -t "The delivery arrives tomorrow." -o en.wav

# streaming: text arrives on stdin piece by piece, every finished sentence is spoken at once (raw PCM on stdout)
my-llm | articulino-cli -m articulino-q8_0.gguf --voice voice.wav --stdin -o - | aplay -r 24000 -f S16_LE -c 1
```

| Option | Meaning |
|---|---|
| `-m, --model` | GGUF file |
| `--voice` | reference recording, WAV (8/16/24/32-bit PCM or float; other sample rates are resampled). The first 8 s are used; 5–8 s of clean, dry speech work best, 3 s is the minimum that sounds reasonable |
| `-t, --text` / `-f, --file` / `--stdin` | text: argument, UTF-8 file, or stdin as it arrives |
| `-o, --out` | output WAV, 24 kHz mono 16 bit; `-` writes raw 16-bit PCM to stdout, sentence by sentence |
| `--threads N` | CPU threads (default 4) |
| `--gpu` | run on the first GPU of a Vulkan build (see [GPU](#gpu-vulkan)) |
| `--seed S` | noise seed (default 0). Every sentence starts from this seed, as in the reference service |
| `--speed X` | speaking rate 0.7–1.6 |
| `--lang de\|en` | base language; `[en]…[/en]` inside the text switches single words (`Das [en]Meeting[/en] ist verschoben.`) |
| `--pause MS` | silence between sentences |
| `--no-rate-eq` | keep the raw predicted durations (see below) |
| `--no-post` | skip the low shelf and peak limit |
| `--no-retry` | keep a sentence even if it came out broken (see below) |
| `--sentence` | speak the text as one sentence, no splitting |
| `--noise FILE.npy` | test mode: sampler noise from a float32 array (4, frames, 96) instead of the seed |
| `--normalize` | only print the sentence split and the letters the model will see |
| `--bench FILE` | real-time factor over the sentences in FILE (one per line), see [Speed](#speed) |

What happens per sentence, exactly as in the reference service: the text is split into sentences (sentences under 20
letters wait for the next one, sentences over 180 characters are split at a comma); each sentence is normalized to
letters; the duration predictor gives every letter a length, and by default every sentence is then brought to the
voice's mean letter rate (measured once per voice on six calibration sentences; `--no-rate-eq` turns this off); the
student DiT makes the new latents in 4 steps behind the 5 s voice prompt; the decoder renders them with 12 prompt frames
of lead-in that are cut off; a +2 dB low shelf at 180 Hz is applied and peaks above 0.95 are scaled down. A sentence
whose latents run away (std > 1.6 or |z| > 9) or that contains a narrow-band whistle above 1.8 kHz for 60 ms is made
once more with seed + 1, like the service does.

### Text: what the engine does and what it does not

The engine does the model's own normalization (`text.normalisiere()` of the reference code): lower case including
Ä Ö Ü ẞ, typographic quotes and dashes mapped, accents removed (é → e, ñ → n, ø → ö, æ → ä, þ → th), whitespace
collapsed, everything outside the 52 symbols dropped. The mapping is a table generated from the Python reference for all
Unicode code points (`tools/gen_text_table.py`), so it gives the same letters.

**It does not spell out numbers, dates, times, money, units or abbreviations.** The model has practically never seen
digits. For German, run the text frontend of the Python release first (it turns „15:30 Uhr“ into „fünfzehn Uhr dreißig“
and „PDF“ into „peh deh eff“), or spell numbers out yourself. A typical setup keeps the frontend in the process that
produces the text and pipes the result into `articulino-cli --stdin` or the C API.

## C API

```c
#include "articulino.h"

static int on_audio(const float * samples, size_t n, const char * sentence, void * user) {
    /* 24 kHz mono float in [-1, 1]; play or store it. Return non-zero to stop. */
    return 0;
}

int main(void) {
    articulino_params p = articulino_default_params();
    p.n_threads = 8;
    articulino_context * ctx = articulino_load("articulino-q8_0.gguf", &p);
    if (!ctx) { fprintf(stderr, "%s\n", articulino_last_error()); return 1; }
    articulino_set_voice_file(ctx, "voice.wav");          /* voice cache: computed once, reused for every sentence */

    articulino_options o = articulino_default_options();  /* de, speed 1, seed 0, rate equalization, post-processing */
    articulino_say(ctx, "Guten Morgen! Heute regnet es.", &o, on_audio, NULL);   /* one callback per sentence */

    /* streaming input, e.g. from a language model */
    articulino_stream * s = articulino_stream_begin(ctx, &o, on_audio, NULL);
    articulino_stream_push(s, "Guten Mor");
    articulino_stream_push(s, "gen! Heute regnet es ab achtzehn Uhr. Nimm ");   /* first sentence is spoken here */
    articulino_stream_push(s, "einen Schirm mit.");
    articulino_stream_end(s);                                                  /* speaks the rest, frees s */

    articulino_free(ctx);
}
```

| Function | |
|---|---|
| `articulino_load(path, params)` / `articulino_free(ctx)` | load a GGUF model / free everything |
| `articulino_set_voice_file(ctx, wav)` / `articulino_set_voice(ctx, samples, n, sample_rate)` | set the voice; encodes it and builds the voice cache |
| `articulino_say(ctx, text, opt, cb, user)` | split into sentences, callback after each |
| `articulino_say_sentence(ctx, sentence, opt, &samples, &n)` | one sentence as given; free with `articulino_free_audio` |
| `articulino_stream_begin / _push / _end` | text in pieces; finished sentences are spoken during `_push` |
| `articulino_normalize(text)` | the letters the model sees (free with `articulino_free_string`) |
| `articulino_set_threads`, `articulino_sample_rate`, `articulino_last_error` | |

A context is not thread safe; use one per thread (they can share nothing but the file). `examples/stream_demo.c` is a
complete program. Link against `libarticulino.so` (or the static `articulino_core` target from CMake).

## Model files

All files hold the complete model: text encoder, duration predictor, voice encoder, student DiT, latent encoder and
decoder, plus every hyper-parameter as GGUF metadata (`articulino.*`: sizes, latent mean/std, sampler times
0, ¼, ½, ¾, the 52-symbol character set, prompt length 125 frames, decoder lead-in 12 frames, 24 kHz, post-processing
settings, rate calibration sentences).

| File | Size | Matmul weights | Use |
|---|---|---|---|
| `articulino-f32.gguf` | 438 MB | F32 | reference; matches PyTorch to float rounding |
| `articulino-f16.gguf` | 220 MB | F16 | half the download; widened to F32 when loading (x86 has no F16 arithmetic), so it runs like F32 |
| `articulino-q8_0.gguf` | 137 MB | Q8_0 | **recommended**: 8 bit, equivalent to F32 on unseen speakers (WER, similarity, UTMOS) |
| `articulino-q4_0.gguf` | 104 MB | Q4_0 | 4 bit, written by the Python converter; lower quality and similarity than IQ4_NL |
| `articulino-iq4_nl.gguf` | 104 MB | IQ4_NL | fastest: 4 bit with non-linear levels, more accurate than Q4_0; UTMOS 0.05 below F32 |

Quantized files keep precision where it is cheap or needed: norms, biases, embeddings and depthwise kernels stay F32;
the voice encoder (`voice.*`, runs once per voice) stays F16, and the latent projection of the latent encoder
(`enc.mu`, outlier inputs) stays F16; in the 4-bit files the latent encoder (`enc.*`), the DiT output projection and
the decoder head stay Q8_0. Everything that runs per sentence (text encoder, durations, DiT, decoder) uses the file's
type. On x86 with AVX2 (and on ARM with dot-product instructions) the 4-bit weights are repacked at load time into
ggml's interleaved layout (`CPU_REPACK`), which is what makes them fast.

### Converting and quantizing

```bash
pip install torch numpy gguf
python tools/convert_to_gguf.py --student schueler-dmd1-020000.pt --vocoder vocoder-d1-020000.pt \
    --type f32 -o articulino-f32.gguf            # also f16, q8_0, q5_0, q4_0
build/articulino-quantize articulino-f32.gguf articulino-iq4_nl.gguf iq4_nl   # also q4_k, q5_k, q6_k, ...
```

The converter reads the two training checkpoints (student acoustic model and latent autoencoder). Tensor layout:
PyTorch `Linear` weights as they are (ggml reads `(out, in)` as `ne = [in, out]`); self-attention projections fused
into `*.attn_qkv`, cross-attention into `*_q` and `*_kv`; convolutions unfolded into matmuls (`(out, K·in)`); the
decoder's transposed convolution (k 8, stride 4) split into its two output phases `dec.up0`/`dec.up1`; depthwise
kernels as `(K, C)`. Tensor names are listed by `python -c "import gguf; print(*[t.name for t in gguf.GGUFReader('x.gguf').tensors], sep='\n')"`.

## Parity

`build/test-parity model.gguf vectors.gguf` runs every building block on the reference input of its stage from the
PyTorch vectors (`tests/vectors/*.npz`, exported with `tests/export_vectors.py`) and compares the output; at the end
the whole chain runs from the voice WAV and the sentence with the reference noise.

F32 file, German / English sentence (`max|Δ| / max|ref|`; the tolerance is 1e-3 per block and 40 dB SNR for audio):

| Stage | de | en |
|---|---|---|
| log-mel | 1.6e-5 | 1.6e-5 |
| latent encoder (mu) | 5.5e-7 | 5.5e-7 |
| voice tokens / vector | 1.1e-6 / 3.6e-7 | 1.1e-6 / 3.6e-7 |
| prompt cache k / v (8 blocks) | 6.1e-6 / 1.1e-6 | 6.1e-6 / 1.1e-6 |
| text encoder | 3.3e-7 | 4.7e-7 |
| log durations | 5.1e-7 | 4.7e-7 |
| durations / frame conditions | 1.0e-9 / 6.6e-8 | 1.9e-9 / 6.7e-8 |
| DiT velocity, steps 0–3 | ≤ 5.6e-6 | ≤ 5.4e-6 |
| 4 chained steps (latents) | 1.7e-5 | 7.6e-6 |
| decoder + inverse STFT | 5.1e-6 (121 dB) | 2.0e-6 (119 dB) |
| whole chain from WAV + text: audio | SNR 73 dB | SNR 74 dB |

The vectorized erf of the GELU uses the same Abramowitz–Stegun formula as PyTorch's CPU kernel, which is why the
decoder output agrees to 120 dB.

For the quantized files the SNR of every stage (same reference inputs, German sentence; the DiT and decoder rows are
what changes the sound):

| Stage | F16 | Q8_0 | Q4_0 | IQ4_NL |
|---|---|---|---|---|
| latent encoder (mu) | 73 dB | 47 dB | 47 dB | 47 dB |
| voice tokens | 63 dB | 63 dB | 63 dB | 63 dB |
| prompt cache k | 76 dB | 43 dB | 23 dB | 24 dB |
| text encoder | 77 dB | 44 dB | 24 dB | 25 dB |
| log durations | 81 dB | 47 dB | 34 dB | 33 dB |
| DiT velocity (one step) | 69 dB | 36 dB | 18 dB | 19 dB |
| 4 chained steps (latents) | 65 dB | 30 dB | 10 dB | 17 dB |
| decoder: waveform SNR / mean log-mel deviation | 67 dB / 0.002 dB | 32 dB / 0.13 dB | 16 dB / 0.60 dB | 19 dB / 0.53 dB |

F16 is computed in F32 after loading (the weights are widened), so only the 16-bit rounding of the weights remains.

Low waveform SNR after the sampler does not mean bad audio: with a 4-step sampler, small changes lead to a different but
equally valid rendering (different phase and micro-timing). What matters is whether the speech stays right; see
[Listening check](#listening-check).

### Listening check

Four German sentences per file, seed 0, voice `radiov`, recognised by Parakeet-TDT 0.6B v3 with 0.4 s of silence in
front; speaker similarity is the ECAPA cosine to the voice reference (speechbrain spkrec-ecapa-voxceleb):

| File | Character error rate | Similarity to the voice |
|---|---|---|
| F32 | 0.0 % | 0.617 |
| F16 | 0.0 % | 0.616 |
| Q8_0 | 0.0 % | 0.624 |
| Q4_0 | 0.0 % | 0.620 |
| IQ4_NL | 0.0 % | 0.644 |

An English sentence pair is recognised word for word from F32, Q8_0 and IQ4_NL. With four sentences, differences in
similarity of ±0.03 are noise.

## Speed

Real-time factor (seconds of audio per second of compute) like the reference measurement `rtf.py`: ten German
sentences (40.8 s of audio), voice cache computed beforehand, one warm-up sentence, then the whole chain per sentence
(text encoder, durations, 4 DiT passes, decoder, inverse STFT, low shelf), seed 42 + i, no rate equalization, like the
reference. Intel Core i9-14900K in a VM, efficiency cores only (`taskset -c 16-27`), best of three runs (other jobs
shared these cores; `tools/bench.sh` waits for idle cores before each run). Reproduce with
`articulino-cli -m MODEL --voice tests/data/radiov.wav --bench tests/data/sentences10.txt --threads N`.

| File | 12 threads | 8 threads | 4 threads | 2 threads | first sentence (12 threads) |
|---|---|---|---|---|---|
| F32 | 25.9× | 20.7× | 11.5× | 6.0× | 0.18 s |
| F16 | 25.7× | 20.3× | 11.5× | 6.0× | 0.18 s |
| Q8_0 | 32.7× | 24.8× | 13.3× | 6.9× | 0.14 s |
| Q4_0 | 52.0× | 38.9× | 21.5× | 11.4× | 0.08 s |
| **IQ4_NL** | **52.4×** | **39.3×** | **21.6×** | **11.2×** | 0.09 s |
| PyTorch F32 (reference) | 24.9× | 20.6× | 12.9× | 8.3× | 0.20 s |

At few threads ggml's float32 matmul reaches less of the cores' peak than PyTorch's (MKL/oneDNN); the 4-bit files make
up for it with less memory traffic and the repacked AVX2 kernels.

Where the time goes (IQ4_NL, 12 threads, per sentence of ~4 s): text encoder and durations ~6 ms, DiT 4 steps ~37 ms,
decoder ~34 ms, inverse STFT and filter < 1 ms; the artifact check of the default settings adds about 1 ms. Setting a
voice (latent encoder, voice encoder, voice cache, rate calibration) takes 0.1–0.15 s.

## GPU (Vulkan)

Built with `-DARTICULINO_VULKAN=ON` (needs the Vulkan headers and `glslc`), `--gpu` (C API: `params.use_gpu = 1`) puts
the weights and the voice cache into GPU memory and runs every graph on the first GPU through ggml's scheduler; the
log-mel, inverse STFT and text processing stay on the CPU. Tested on an AMD Radeon AI PRO R9700 with Mesa RADV
(`VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json` selects it when several drivers are installed); every graph
runs on the GPU in one piece.

| File | Real-time factor | First sentence |
|---|---|---|
| F32 / F16 | 138× | 0.03 s |
| Q8_0 | 135× | 0.03 s |
| IQ4_NL | 136× | 0.03 s |

On the GPU the weight type hardly matters: the model is small and the time goes into launching kernels (DiT ~19 ms,
decoder ~7 ms per sentence). ggml's Vulkan matmuls round to 16-bit floats internally, so even the F32 file agrees with
PyTorch only to about 60–65 dB per stage (decoder spectrum 0.004 dB); the four German test sentences are recognised
without errors (F32 and IQ4_NL, similarity to the voice 0.618 and 0.645). Other GPUs and backends are untested.

## How it is built

- `src/model.cpp`: GGUF loading, weight placement (default CPU buffer, or the repack buffer for 4-bit), thread pool
- `src/nn.cpp`: the networks as ggml graphs; activations are `[channels, time]`. The four sampler steps run in one
  graph; the prompt's keys/values, the voice tokens' cross-attention keys/values and the AdaNorm scale/shift of every
  block at the four sampler times are computed once per voice
- `src/ops_cpu.cpp`: two fused CPU ops (depthwise convolution, GELU with vectorized erf)
- `src/audio.cpp`: log-mel (Hann periodic, centered on 1024, reflect padding, HTK mel filters from the checkpoint),
  inverse STFT with window-sum normalization like `torch.istft`, WAV I/O, resampling, low shelf
- `src/text.cpp`: normalization table, `[en]` markup, sentence splitter (same results as the Python service on the
  cases in `tests/check_text.py`)
- `src/synth.cpp`: durations, rate equalization, upsampling of the letter features to frames, per-sentence pipeline

Development tools: `tests/test_parity.cpp`, `tests/check_text.py` (normalizer and splitter against Python),
`tests/asr_check.py` (Parakeet), `tests/similarity.py` (ECAPA), `tools/bench.sh`, `tools/sensitivity.sh` (per-layer
quantization sensitivity). `ARTICULINO_PROFILE=1` prints the time per ggml op (single-threaded runs are meaningful),
`ARTICULINO_NO_REPACK=1` keeps 4-bit weights in the plain layout, `ARTICULINO_GENERIC_OPS=1` uses plain ggml ops
instead of the fused ones.

## Limitations

- **The GPU path is young**: tested only with Vulkan on one AMD card; on the GPU the results agree with the reference
  to fp16 precision, not float32. CUDA and Metal builds of ggml should work through the same scheduler but were not
  tried.
- **Streaming is per sentence**, as in the model: a sentence's audio is ready at once, not inside the sentence.
- **No text frontend for numbers and abbreviations** (see above); without it, digits are not read.
- **Reproducibility.** The engine's noise generator (xoshiro256** + Box–Muller) is not PyTorch's, so the same seed gives
  a different (equally valid) rendering than the Python release; within the engine a seed always gives the same audio
  on the same build. `--noise` feeds external noise for exact comparisons.
- **English is limited** to what the model learned (51 h of audiobooks).
- Voices longer than 8 s are cut; voices shorter than 5 s use a shorter prompt.

## License

Apache License 2.0 (`LICENSE`), for the engine and the converted weights. Third-party code: ggml (MIT) and pocketfft
(BSD 3-Clause), see `THIRD_PARTY_NOTICES.md`. The training data attribution of the model (CC BY 4.0 corpora) is in the
NOTICE of the model repository and must be kept when redistributing the weights.

Only clone voices with the speaker's consent, and disclose synthetic speech; see "Responsible use" in the model card.
