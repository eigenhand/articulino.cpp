#!/usr/bin/env python3
"""Convert the Articulino PyTorch checkpoints to a single GGUF file for articulino.cpp.

    python convert_to_gguf.py --student schueler-dmd1-020000.pt --vocoder vocoder-d1-020000.pt \
        --type q8_0 -o articulino-q8_0.gguf

Inputs are the two training checkpoints (student acoustic model and latent autoencoder). The output holds every
tensor the C++ engine needs, under descriptive names, plus all hyper-parameters as metadata. Layout notes:

* Linear weights keep the PyTorch (out, in) layout, which ggml reads as ne = [in, out] (ready for ggml_mul_mat).
* Self-attention projections are fused into one `attn_qkv` matmul (rows q, k, v); cross-attention keeps `*_q` and
  a fused `*_kv` (rows k, v), because queries and keys come from different inputs.
* Convolutions are stored "unfolded" for a matmul: (out, in, K) -> (out, K * in), index k * in + c.
  The transposed convolution of the decoder (k 8, stride 4) is split into its two output phases (see dec.up0/up1).
* Depthwise ConvNeXt kernels are stored as (K, C) so that tap k is one row of C values.
* The latent encoder keeps only the `mu` half of its output projection.

Weight types: f32, f16, q8_0, q5_0, q4_0. Only large matmul weights are quantized (row length a multiple of 32);
norms, biases, embeddings, depthwise taps and the mel filterbank stay in F32, and matmul weights that cannot be
quantized fall back to F16.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import gguf
from gguf import GGMLQuantizationType as QT

ARCH = "articulino"
CHARSET = " abcdefghijklmnopqrstuvwxyzäöüß.,?!-'\":;()0123456789"
# sentences used by the reference service to calibrate a voice's mean speaking rate (TTSK_TEMPO_GLEICH)
RATE_CALIBRATION = [
    "Guten Morgen, die Systeme laufen stabil und es gibt nichts Dringendes.",
    "Ich habe die Unterlagen geprüft und fasse das Wichtigste kurz zusammen.",
    "Der Termin am Donnerstag wurde auf den Nachmittag verschoben.",
    "Soll ich die Nachricht jetzt abschicken oder noch einmal vorlesen?",
    "Das Training läuft seit gestern Abend ohne Unterbrechung weiter.",
    "Draußen sind es zwölf Grad, und am Abend soll es regnen.",
]
TYPES = {"f32": QT.F32, "f16": QT.F16, "q8_0": QT.Q8_0, "q5_0": QT.Q5_0, "q4_0": QT.Q4_0}
# llama.cpp style file type ids, for tools that display them
FILE_TYPE = {"f32": 0, "f16": 1, "q8_0": 7, "q5_0": 8, "q4_0": 2}


def np32(t) -> np.ndarray:
    return np.ascontiguousarray(t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t, np.float32))


class Collector:
    def __init__(self, wtype: str, keep_q8: set[str], keep_f16: str = "", only: str = "", voice_type: str = "",
                 enc_type: str = ""):
        self.wtype = wtype
        self.voice_type = voice_type
        self.enc_type = enc_type
        self.keep_q8 = keep_q8
        self.keep_f16 = re.compile(keep_f16) if keep_f16 else None
        self.only = re.compile(only) if only else None
        self.items: list[tuple[str, np.ndarray, QT]] = []
        self.n_params = 0

    def vec(self, name: str, a):
        """Norm weights, biases, gammas, embeddings, tables: always F32."""
        a = np32(a)
        self.items.append((name, a, QT.F32))
        self.n_params += a.size

    def mat(self, name: str, a):
        """A 2-D matmul weight (out, in)."""
        a = np32(a)
        assert a.ndim == 2, (name, a.shape)
        self.n_params += a.size
        want = TYPES[self.wtype]
        # the voice path runs once per voice, not per sentence: precision is cheap there
        if self.voice_type and name.startswith("voice."):
            want = TYPES[self.voice_type]
        if self.enc_type and name.startswith("enc."):
            want = TYPES[self.enc_type]
        if want in (QT.Q8_0, QT.Q5_0, QT.Q4_0):
            if a.shape[1] % 32 != 0 or a.shape[0] < 32:
                want = QT.F16                           # cannot be block-quantized (or too small to matter)
            elif want != QT.Q8_0 and TYPES[self.wtype] != QT.Q8_0 and any(name.startswith(p) for p in self.keep_q8):
                want = QT.Q8_0                          # sensitive layers keep 8 bits in the 4/5-bit files
            if (self.keep_f16 and self.keep_f16.search(name)) or (self.only and not self.only.search(name)):
                want = QT.F16
        self.items.append((name, a, want))

    def write(self, writer: gguf.GGUFWriter):
        for name, a, qt in self.items:
            if qt == QT.F32:
                writer.add_tensor(name, a.astype(np.float32))
            elif qt == QT.F16:
                writer.add_tensor(name, a.astype(np.float16))
            else:
                writer.add_tensor(name, gguf.quants.quantize(a, qt), raw_dtype=qt)


def linear(c: Collector, name: str, sd: dict, key: str, bias=True):
    c.mat(name + ".weight", sd[key + ".weight"])
    if bias:
        c.vec(name + ".bias", sd[key + ".bias"])


def norm(c: Collector, name: str, sd: dict, key: str):
    c.vec(name + ".weight", sd[key + ".weight"])
    c.vec(name + ".bias", sd[key + ".bias"])


def attention(c: Collector, name: str, sd: dict, key: str, prefix: str, fuse_q: bool):
    """Attention(q, kv, aus). fuse_q (self-attention): one {prefix}_qkv matmul [q; k; v];
    otherwise (cross-attention, q and k/v see different inputs): {prefix}_q and {prefix}_kv [k; v]."""
    wq, bq = np32(sd[key + ".q.weight"]), np32(sd[key + ".q.bias"])
    wkv, bkv = np32(sd[key + ".kv.weight"]), np32(sd[key + ".kv.bias"])    # rows: k then v
    if fuse_q:
        c.mat(f"{name}.{prefix}_qkv.weight", np.concatenate([wq, wkv], 0))
        c.vec(f"{name}.{prefix}_qkv.bias", np.concatenate([bq, bkv], 0))
    else:
        c.mat(f"{name}.{prefix}_q.weight", wq); c.vec(f"{name}.{prefix}_q.bias", bq)
        c.mat(f"{name}.{prefix}_kv.weight", wkv); c.vec(f"{name}.{prefix}_kv.bias", bkv)
    linear(c, f"{name}.{prefix}_out", sd, key + ".aus")


def ffn(c: Collector, name: str, sd: dict, key: str):
    linear(c, f"{name}.ffn_up", sd, key + ".a")
    linear(c, f"{name}.ffn_down", sd, key + ".b")


def tblock(c: Collector, name: str, sd: dict, key: str):
    norm(c, f"{name}.attn_norm", sd, key + ".n1")
    attention(c, name, sd, key + ".att", "attn", True)
    norm(c, f"{name}.ffn_norm", sd, key + ".n2")
    ffn(c, name, sd, key + ".ff")


def conv_unfold(w) -> np.ndarray:
    """Conv1d weight (out, in, K) -> (out, K * in) with column index k * in + c."""
    w = np32(w)
    o, i, k = w.shape
    return np.ascontiguousarray(w.transpose(0, 2, 1).reshape(o, k * i))


def convnext(c: Collector, name: str, sd: dict, key: str):
    dw = np32(sd[key + ".dw.weight"])                   # (C, 1, 7)
    c.vec(f"{name}.dw.weight", np.ascontiguousarray(dw[:, 0, :].T))     # (7, C)
    c.vec(f"{name}.dw.bias", sd[key + ".dw.bias"])
    norm(c, f"{name}.norm", sd, key + ".norm")
    linear(c, f"{name}.up", sd, key + ".up")
    linear(c, f"{name}.down", sd, key + ".down")
    c.vec(f"{name}.gamma", sd[key + ".gamma"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student", required=True, help="student checkpoint (schueler-*.pt)")
    ap.add_argument("--vocoder", required=True, help="autoencoder checkpoint (vocoder-*.pt)")
    ap.add_argument("--type", default="f16", choices=sorted(TYPES), help="weight type for large matmul weights")
    ap.add_argument("--keep-q8", default="dec.head,dit.out.",
                    help="comma-separated tensor name prefixes kept at Q8_0 in q4_0/q5_0 files ('' = none)")
    ap.add_argument("--keep-f16", default=r"^enc\.mu\.",
                    help="regex: matmul weights kept at F16 in quantized files (default: the latent projection of the "
                         "voice encoder, which has outlier inputs)")
    ap.add_argument("--voice-type", default=None, choices=sorted(TYPES),
                    help="type of the voice encoder (voice.*); default f16 in quantized files")
    ap.add_argument("--enc-type", default=None, choices=sorted(TYPES),
                    help="type of the latent encoder (enc.*); default q8_0 in q4_0/q5_0 files, else --type")
    ap.add_argument("--only", default="", help="regex: quantize only these matmul weights (others F16); for experiments")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    ck = torch.load(args.student, map_location="cpu", weights_only=False)
    vck = torch.load(args.vocoder, map_location="cpu", weights_only=False)
    cfg, vcfg = ck["cfg"], vck["cfg"]
    assert cfg.get("anfang_eigen") and "lehrer_dim" in cfg, "expected a student checkpoint with a self-contained prompt"
    sd, vsd = ck["ema"], vck["ae"]
    latent, dim, heads, layers = cfg["latent"], cfg["dim"], cfg["koepfe"], cfg["lagen"]
    text_dim, text_layers = cfg["text_dim"], cfg["text_lagen"]
    voice_out = cfg["lehrer_dim"]
    times = [float(t) for t in ck.get("zeiten", (0.0, 0.25, 0.5, 0.75))]
    assert len(CHARSET) + 1 == cfg["v"]

    keep = {p for p in args.keep_q8.split(",") if p}
    quant = args.type in ("q8_0", "q5_0", "q4_0")
    voice_type = args.voice_type or ("f16" if quant else "")
    enc_type = args.enc_type or ("q8_0" if args.type in ("q4_0", "q5_0") else "")
    c = Collector(args.type, keep, args.keep_f16, args.only, voice_type, enc_type)

    # ---------------- text encoder
    c.vec("text.tok_embd", sd["text.emb.weight"])
    c.vec("text.lang_embd", sd["text.sprache.weight"])
    for i in range(text_layers):
        tblock(c, f"text.blk.{i}", sd, f"text.bloecke.{i}")
    norm(c, "text.out_norm", sd, "text.norm")

    # ---------------- voice (reference) encoder
    voice_layers = len({k.split(".")[2] for k in sd if k.startswith("ref.bloecke.")})
    n_tokens = sd["ref.fragen"].shape[0]
    linear(c, "voice.in", sd, "ref.ein")
    for i in range(voice_layers):
        tblock(c, f"voice.blk.{i}", sd, f"ref.bloecke.{i}")
    c.vec("voice.queries", sd["ref.fragen"])
    for j in range(2):
        norm(c, f"voice.pool.{j}.attn_norm", sd, f"ref.pool.{j}.0")
        attention(c, f"voice.pool.{j}", sd, f"ref.pool.{j}.1", "attn", False)
        norm(c, f"voice.pool.{j}.ffn_norm", sd, f"ref.pool.{j}.2")
        ffn(c, f"voice.pool.{j}", sd, f"ref.pool.{j}.3")
    norm(c, "voice.out_norm", sd, "ref.norm")
    linear(c, "voice.tokens_out", sd, "ref.aus")
    linear(c, "voice.vec_fc1", sd, "ref.vektor.0")
    linear(c, "voice.vec_fc2", sd, "ref.vektor.2")

    # ---------------- duration predictor
    dur_width = sd["dauer.c1.weight"].shape[0]
    linear(c, "dur.voice_proj", sd, "dauer.stimme")
    for i in (1, 2, 3):
        c.mat(f"dur.conv{i}.weight", conv_unfold(sd[f"dauer.c{i}.weight"]))
        c.vec(f"dur.conv{i}.bias", sd[f"dauer.c{i}.bias"])
        norm(c, f"dur.norm{i}", sd, f"dauer.n{i}")
    c.vec("dur.out.weight", sd["dauer.aus.weight"])       # (1, 384): a dot product, kept in F32
    c.vec("dur.out.bias", sd["dauer.aus.bias"])

    # ---------------- DiT (student)
    linear(c, "dit.text_proj", sd, "text_zu")
    w_in = np32(sd["ein.weight"])                          # (dim, 2 * latent + 1): [x_t, prompt, prompt flag]
    c.mat("dit.in_noisy.weight", w_in[:, :latent])
    c.mat("dit.in_prompt.weight", w_in[:, latent:2 * latent])
    c.vec("dit.in_flag", w_in[:, 2 * latent])
    c.vec("dit.in.bias", sd["ein.bias"])
    linear(c, "dit.time_fc1", sd, "zeit.0")
    linear(c, "dit.time_fc2", sd, "zeit.2")
    linear(c, "dit.voice_tok_proj", sd, "tok_zu")
    linear(c, "dit.voice_vec_proj", sd, "g_zu")
    for i in range(layers):
        k = f"bloecke.{i}"
        n = f"dit.blk.{i}"
        linear(c, f"{n}.ada_attn", sd, k + ".n1.lin")
        attention(c, n, sd, k + ".att", "attn", True)
        norm(c, f"{n}.xattn_norm", sd, k + ".n2")
        attention(c, n, sd, k + ".kreuz", "xattn", False)
        linear(c, f"{n}.ada_ffn", sd, k + ".n3.lin")
        ffn(c, n, sd, k + ".ff")
    linear(c, "dit.out_ada", sd, "norm.lin")
    linear(c, "dit.out", sd, "aus")

    # ---------------- latent encoder (voice wav -> latents)
    c.vec("audio.mel_fb", vsd["encoder.mel.mel.mel_scale.fb"])      # (513, 100)
    c.vec("audio.hann", vsd["encoder.mel.mel.spectrogram.window"])  # periodic Hann, 960
    c.mat("enc.in.weight", conv_unfold(vsd["encoder.ein.weight"]))
    c.vec("enc.in.bias", vsd["encoder.ein.bias"])
    blk = down = 0
    n_stufen = len({k.split(".")[2] for k in vsd if k.startswith("encoder.stufen.")})
    for s in range(n_stufen):
        k = f"encoder.stufen.{s}"
        if k + ".gamma" in vsd:
            convnext(c, f"enc.blk.{blk}", vsd, k); blk += 1
        else:
            c.mat(f"enc.down.{down}.weight", conv_unfold(vsd[k + ".weight"]))
            c.vec(f"enc.down.{down}.bias", vsd[k + ".bias"]); down += 1
    enc_blocks = blk
    norm(c, "enc.out_norm", vsd, "encoder.norm")
    c.mat("enc.mu.weight", np32(vsd["encoder.aus.weight"])[:latent])
    c.vec("enc.mu.bias", np32(vsd["encoder.aus.bias"])[:latent])

    # ---------------- decoder (latents -> waveform)
    c.mat("dec.in.weight", conv_unfold(vsd["decoder.ein.weight"]))
    c.vec("dec.in.bias", vsd["decoder.ein.bias"])
    wt = np32(vsd["decoder.hoch.weight"])                  # ConvTranspose1d (in, out, K=8), stride 4, padding 2
    cin, cout, kk = wt.shape
    stride = 4
    assert kk == 2 * stride
    # output phase a (0/1) and offset r: tap k = 4a + r lands at output 4(t + a) + r - 2. Store per phase a the
    # matrix (r * cout + o, cin) so that one matmul gives all 4 offsets of a frame.
    for a in range(2):
        w_a = wt[:, :, a * stride:(a + 1) * stride]        # (cin, cout, 4) [c, o, r]
        c.mat(f"dec.up{a}.weight", np.ascontiguousarray(w_a.transpose(2, 1, 0).reshape(stride * cout, cin)))
    c.vec("dec.up.bias", vsd["decoder.hoch.bias"])
    dec_blocks = vcfg["dec_bloecke"]
    for i in range(dec_blocks):
        convnext(c, f"dec.blk.{i}", vsd, f"decoder.bloecke.{i}")
    norm(c, "dec.out_norm", vsd, "decoder.norm")
    linear(c, "dec.head", vsd, "decoder.kopf")

    # ---------------- metadata
    w = gguf.GGUFWriter(args.out, ARCH)
    w.add_name("Articulino-DE-EN-Realtime-Stream-TTS")
    w.add_description("German + English text to speech with zero-shot voice cloning; 4-step DMD2 student")
    w.add_license("apache-2.0")
    w.add_file_type(FILE_TYPE[args.type])
    A = ARCH + "."
    w.add_string(A + "weight_type", args.type)
    w.add_uint32(A + "sample_rate", 24000)
    w.add_string(A + "charset", CHARSET)
    w.add_uint32(A + "vocab_size", cfg["v"])
    w.add_array(A + "languages", ["de", "en"])
    w.add_uint32(A + "latent_dim", latent)
    w.add_uint32(A + "latent_rate", 25)
    w.add_uint32(A + "samples_per_latent", 960)
    w.add_uint32(A + "duration_units_per_latent", 4)          # durations are in 10 ms units
    w.add_float32(A + "duration_max", 400.0)
    w.add_uint32(A + "prompt_frames", 125)                     # 5 s voice prompt in front of every sentence
    w.add_uint32(A + "decoder_context_frames", 12)             # prompt frames decoded in front and cut off
    w.add_uint32(A + "voice_max_samples", 8 * 24000)
    w.add_array(A + "latent_mean", [float(x) for x in np32(ck["mittel"])])
    w.add_array(A + "latent_std", [float(x) for x in np32(ck["streuung"])])
    w.add_array(A + "sampler_times", times)
    w.add_uint32(A + "text.dim", text_dim)
    w.add_uint32(A + "text.layers", text_layers)
    w.add_uint32(A + "text.heads", 8)
    w.add_uint32(A + "voice.dim", text_dim)
    w.add_uint32(A + "voice.layers", voice_layers)
    w.add_uint32(A + "voice.heads", 8)
    w.add_uint32(A + "voice.tokens", n_tokens)
    w.add_uint32(A + "voice.out_dim", voice_out)
    w.add_uint32(A + "duration.width", dur_width)
    w.add_uint32(A + "dit.dim", dim)
    w.add_uint32(A + "dit.layers", layers)
    w.add_uint32(A + "dit.heads", heads)
    w.add_float32(A + "rope.base", 10000.0)
    w.add_uint32(A + "mel.n_fft", 1024)
    w.add_uint32(A + "mel.win_length", 960)
    w.add_uint32(A + "mel.hop", 240)
    w.add_uint32(A + "mel.n_mels", 100)
    w.add_uint32(A + "istft.n_fft", 960)
    w.add_uint32(A + "istft.hop", 240)
    w.add_uint32(A + "encoder.dim", vcfg["enc_dim"])
    w.add_uint32(A + "encoder.blocks", enc_blocks)
    w.add_uint32(A + "decoder.dim", vcfg["dec_dim"])
    w.add_uint32(A + "decoder.blocks", dec_blocks)
    w.add_float32(A + "post.shelf_db", 2.0)
    w.add_float32(A + "post.shelf_hz", 180.0)
    w.add_float32(A + "post.shelf_slope", 0.7)
    w.add_float32(A + "post.peak", 0.95)
    w.add_array(A + "rate_calibration", RATE_CALIBRATION)
    w.add_uint32(A + "source.student_step", int(ck.get("schritt", 0)))
    w.add_uint32(A + "source.vocoder_step", int(vck.get("schritt", 0)))
    c.write(w)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    counts = {}
    for _, a, qt in c.items:
        counts[qt.name] = counts.get(qt.name, 0) + a.size
    size = Path(args.out).stat().st_size
    print(f"{args.out}: {len(c.items)} tensors, {c.n_params / 1e6:.1f} M parameters, {size / 1e6:.1f} MB | "
          + ", ".join(f"{k} {v / 1e6:.1f} M" for k, v in sorted(counts.items())), file=sys.stderr)


if __name__ == "__main__":
    main()
