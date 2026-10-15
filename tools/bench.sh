#!/bin/bash
# Real-time factor like the original evaluation script: 10 sentences, the test voice, voice cache beforehand, whole
# chain per sentence. Runs each (model, threads) pair REPEAT times when the cores are idle and keeps the best run.
# Optional: CPUS=first-last pins the runs to those cores (and tools/wait_idle.sh watches them).
#   tools/bench.sh "models/articulino-f32.gguf models/articulino-q8_0.gguf" "12 8 4 2" [repeat]
MODELS=$1; THREADS=${2:-"12 8 4 2"}; REPEAT=${3:-3}
S=tests/data/sentences10.txt; V=tests/data/radiov.wav
PIN=(); [ -n "$CPUS" ] && PIN=(taskset -c "$CPUS")
cd "$(dirname "$0")/.."
for t in $THREADS; do
  for m in $MODELS; do
    best=""; bestv=0
    for r in $(seq $REPEAT); do
      ./tools/wait_idle.sh || echo "(cores not idle, measuring anyway)" >&2
      line=$("${PIN[@]}" ./build/articulino-cli -m $m --voice $V --bench $S --threads $t -q 2>/dev/null)
      v=$(echo "$line" | sed -E 's/.*= ([0-9.]+)x real time.*/\1/')
      if awk "BEGIN{exit !($v > $bestv)}"; then bestv=$v; best=$line; fi
    done
    echo "$best"
  done
done
