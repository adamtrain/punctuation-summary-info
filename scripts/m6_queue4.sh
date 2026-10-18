#!/bin/bash
# Remainder of the scaling queue (BRIEF.md decision log, 2026-10-18), replacing m6_queue3.sh after GPT-2's long step,
# which keeps going: long-distance for the other finished models (2.8B added: its long step ran out of memory), then
# Llama with split readout tokenization and the fused-vs-split check at the end. Appends to queue2.log.
cd /lambda/nfs/AT1/punctuation-summary-info || exit 1
PY=$HOME/venvs/punctuation-summary-info/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== $*"; "$@" || { echo "=== FAILED: $*"; return 1; }; }

while pgrep -f "m6_scale.py run gpt2 long" >/dev/null; do sleep 60; done
echo "=== queue 4 continues $(date -u)"

for m in gpt-neo-125m pythia-160m pythia-410m pythia-1.4b pythia-2.8b; do step $PY scripts/m6_scale.py run $m long; done
echo "=== LADDER2_DONE $(date -u)"

L=llama-3.1-8b
step $PY scripts/m6_scale.py verify $L && step $PY scripts/m6_scale.py bf16check $L && step $PY scripts/m6_scale.py screen $L \
  || { echo "=== LLAMA_STOPPED: setup failed"; exit 1; }
step $PY scripts/m6_ladder_items.py $L $L || { echo "=== LLAMA_STOPPED: items fail the bar"; exit 1; }
step $PY scripts/m6_long_pairs.py pairs $L && step $PY scripts/m6_token_check.py $L $L --long \
  || { echo "=== LLAMA_STOPPED: long pairs"; exit 1; }
step $PY scripts/m6_scale.py run $L || { echo "=== LLAMA_STOPPED: run failed"; exit 1; }
step $PY scripts/m6_scale.py fusedcheck $L
echo "=== LLAMA_DONE $(date -u)"
