#!/usr/bin/env python3
"""Listening check through a speech recognizer (OpenAI-style /v1/audio/transcriptions; we used Parakeet-TDT 0.6B v3): speak sentences with articulino-cli for each model,
put 0.4 s of silence in front, transcribe, and report the character error rate against the input text.

    python tests/asr_check.py --asr http://127.0.0.1:8084 --cli build/articulino-cli --voice tests/data/radiov.wav --out out/asr models/articulino-*.gguf
"""
import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

SENTENCES = [
    "Guten Morgen, die Systeme laufen stabil und es gibt nichts Dringendes.",
    "Der Termin am Donnerstag wurde auf den Nachmittag verschoben.",
    "Draußen sind es zwölf Grad, und am Abend soll es regnen.",
    "Kurz zur Erinnerung: Du wolltest heute noch den Drucker reinigen.",
]
URL = ""   # set from --asr


NUMBER_WORDS = "null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn " \
         "siebzehn achtzehn neunzehn zwanzig".split()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"\b(\d{1,2})\b", lambda m: NUMBER_WORDS[int(m.group(1))] if int(m.group(1)) <= 20 else m.group(1), s)
    s = re.sub(r"[^a-zäöüß0-9 ]", " ", s)
    return " ".join(s.split())


def cer(ref: str, hyp: str) -> float:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def transcribe(wav: Path, lang: str = "de") -> str:
    y, sr = sf.read(wav, dtype="float32")
    padded = wav.with_suffix(".asr.wav")
    sf.write(padded, np.concatenate([np.zeros(int(0.4 * sr), np.float32), y]), sr)
    r = subprocess.run(["curl", "-s", URL, "-F", f"file=@{padded}", "-F", f"language={lang}"], capture_output=True,
                       timeout=120)
    padded.unlink()
    try:
        return json.loads(r.stdout.decode()).get("text", "")
    except json.JSONDecodeError:
        return "<error: " + r.stdout.decode()[:200] + ">"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", required=True, help="base URL of the speech recognizer")
    ap.add_argument("--cli", required=True)
    ap.add_argument("--voice", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", default="12")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--extra", default="", help="extra CLI arguments, e.g. --gpu")
    ap.add_argument("--tag", default="", help="suffix for the output names")
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    global URL
    URL = a.asr.rstrip("/") + "/v1/audio/transcriptions"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = []
    for m in a.models:
        name = Path(m).stem + a.tag
        cers = []
        for i, s in enumerate(SENTENCES):
            wav = out / f"{name}-{i + 1}.wav"
            subprocess.run([a.cli, "-m", m, "--voice", a.voice, "-t", s, "-o", str(wav), "--threads", a.threads,
                            "--seed", a.seed, "-q"] + a.extra.split(), check=True)
            hyp = transcribe(wav)
            c = cer(s, hyp)
            cers.append(c)
            print(f"{name} [{i + 1}] CER {c * 100:5.1f} %  {hyp}", flush=True)
        summary.append((name, float(np.mean(cers))))
    print()
    for name, c in summary:
        print(f"{name:28s} mean CER {c * 100:5.2f} %")


if __name__ == "__main__":
    sys.exit(main())
