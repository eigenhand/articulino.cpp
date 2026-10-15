#!/usr/bin/env python3
"""Compare the C++ normalizer and sentence splitter (build/test-text) with the PyTorch reference package
(articulino.text.normalize_characters and articulino.sentences.split_sentences, https://github.com/eigenhand/articulino),
which in turn matches the original service code.

    pip install "articulino @ git+https://github.com/eigenhand/articulino"   # or: export ARTICULINO_PYTHON_SRC=.../src
    python tests/check_text.py build/test-text
"""
import os
import subprocess
import sys

if os.environ.get("ARTICULINO_PYTHON_SRC"):
    sys.path.insert(0, os.environ["ARTICULINO_PYTHON_SRC"])
from articulino.sentences import split_sentences  # noqa: E402
from articulino.text import normalize_characters  # noqa: E402

CASES = [
    "Guten Morgen! Der Termin ist heute um fünfzehn Uhr dreißig.",
    "ÄÖÜ äöü ẞ ß Straße GROSS",
    "Café „Olé“ – naïve Ærø… Smørrebrød, Þór, Ðe",
    "It's 5 o'clock! «Bonjour» ‚quoted‘ ﬁnal ﬂow x² H₂O ½ №5",
    "  Mehrfach   Leerzeichen\tund\nZeilen  ",
    "Emoji 😀 und 中文 und Ελληνικά und Кириллица",
    "Wer? Wie! Was... Ja: nein; vielleicht (oder) [Klammer] {x} <y> @ # $ % & * + = / \\ | ~ ^ _",
    "Ça va? Où est l'hôtel? Ñandú, São Paulo, Kraków, Dvořák, Şişli, Tromsø",
    " geschütztes Leerzeichen schmal　breit",
]
SPLIT = [
    ("S", "Hallo. Wie geht es dir heute? Mir geht es gut!"),
    ("S", "Ja. Nein. Vielleicht doch, aber nur wenn es heute nicht regnet. Gut."),
    ("P", "Das ist ein Satz. Und hier kommt noch einer"),
    ("P", "Das ist ein ganz normaler Satz mit Ende."),
    ("P", "Das ist ein ganz normaler Satz mit Ende. "),
    ("S", "Er sagte: \"Komm her!\" Dann ging er. Das war es dann auch schon für heute, oder?"),
    ("S", "Zahlen wie 3.5 oder 1.000 sind schwierig... Und Abkürzungen z.B. auch."),
    ("S", "Das hier ist ein sehr langer Satz, der viele Kommas enthält, damit er geteilt werden muss, weil das Modell nur bis etwa zwanzig Sekunden kennt, und deshalb teilen wir ihn an einem Komma, das zwischen sechzig und hundertachtzig Zeichen liegt, wie im Original."),
    ("S", "Ein sehr langer Satz ohne ein einziges Komma der trotzdem immer weiter und weiter geht bis er endlich die Grenze von hundertachtzig Zeichen ueberschreitet und dann an einem Leerzeichen geteilt werden muss weil es keine Kommas gibt."),
    ("P", "Kurz. Auch kurz. "),
    ("S", "Ende ohne Punkt"),
    ("S", "Wirklich?!… Ja!!! (Echt.) 'So.' Fertig."),
    ("P", "Erster Satz ist lang genug für sich allein.\nZweiter Satz auch, er steht auf der nächsten Zeile.\n"),
]


def main():
    exe = sys.argv[1]
    lines = [f"N|{c}" for c in CASES] + [f"{m}|{t}" for m, t in SPLIT]
    inp = "\n".join(l.replace("\n", "\\n") for l in lines) + "\n"
    out = subprocess.run([exe], input=inp.encode(), capture_output=True, check=True).stdout.decode().splitlines()
    bad = 0
    for line, got in zip(lines, out):
        mode, text = line[0], line[2:]
        if mode == "N":
            want = normalize_characters(text)
        else:
            done, rest = split_sentences(text, mode == "S")
            want = "".join(f"[{s}]" for s in done) + f"|rest=[{rest}]"
        ok = want == got
        bad += not ok
        print(("ok  " if ok else "DIFF") + f" {mode} {text[:60]!r}")
        if not ok:
            print(f"     python: {want!r}\n     c++   : {got!r}")
    print(f"{len(lines) - bad}/{len(lines)} identical")
    sys.exit(1 if bad else 0)


main()
