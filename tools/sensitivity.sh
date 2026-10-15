#!/bin/bash
# Quantization sensitivity: quantize one group of matmul weights at a time (rest F16) and print the SNR of the
# stage it feeds. Usage: tools/sensitivity.sh q8_0|q5_0 [vectors]   (run from the repository root, after building)
# Needs the training checkpoints: CHECKPOINTS=<folder with schueler-dmd1-020000.pt and vocoder-d1-020000.pt>
T=${1:-q8_0}; VEC=${2:-build/vectors/de_sentence.vectors.gguf}
D=${CHECKPOINTS:?set CHECKPOINTS}; PY=${PYTHON:-python}
mkdir -p models/exp
while read -r name regex stage; do
  [ -z "$name" ] && continue
  f=models/exp/sens-$T-$name.gguf
  $PY tools/convert_to_gguf.py --student $D/schueler-dmd1-020000.pt --vocoder $D/vocoder-d1-020000.pt --type $T \
     --keep-q8 "" --voice-type $T --enc-type $T --keep-f16 "" --only "$regex" -o $f >/dev/null 2>&1
  out=$(./build/test-parity $f $VEC --threads 12 2>&1)
  snr=$(echo "$out" | grep -E "^$stage " | sed -E 's/.*SNR +([-0-9.]+) dB.*/\1/' | tr '\n' ' ')
  printf "%-16s %-40s %-16s %s\n" "$name" "$regex" "$stage" "$snr"
  rm -f $f
done <<'LIST'
enc_up      ^enc\.blk\.\d+\.up\.            voice_mu
enc_down    ^enc\.blk\.\d+\.down\.          voice_mu
enc_conv    ^enc\.(in|down)\.               voice_mu
enc_mu      ^enc\.mu\.                      voice_mu
voice_in    ^voice\.in\.                    voice_tokens
voice_attn  ^voice\.blk\.\d+\.attn_         voice_tokens
voice_ffn   ^voice\.blk\.\d+\.ffn_          voice_tokens
voice_pool  ^voice\.pool\.                  voice_tokens
voice_out   ^voice\.(tokens_out|vec_)       voice_tokens
text_attn   ^text\.blk\.\d+\.attn_          text_h
text_ffn    ^text\.blk\.\d+\.ffn_           text_h
dur         ^dur\.                          duration_log
dit_inproj  ^dit\.(text_proj|in_|time_|voice_)  step1_velocity
dit_ada     ^dit\.blk\.\d+\.ada_            step1_velocity
dit_attn    ^dit\.blk\.\d+\.attn_           step1_velocity
dit_xattn   ^dit\.blk\.\d+\.xattn_          step1_velocity
dit_ffn     ^dit\.blk\.\d+\.ffn_            step1_velocity
dit_out     ^dit\.out                       step1_velocity
dec_in      ^dec\.(in|up)                   decoder_out
dec_up      ^dec\.blk\.\d+\.up\.            decoder_out
dec_down    ^dec\.blk\.\d+\.down\.          decoder_out
dec_head    ^dec\.head                      decoder_out
LIST
