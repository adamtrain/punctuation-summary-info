#!/bin/bash
# Second scaling queue (BRIEF.md decision log, 2026-10-18): reruns what failed in the first one, one GPU job at a time.
#   long-distance step for every model (PG19 edge-apostrophe fix), Pythia-2.8B/6.9B after calibration (out of memory
#   next to the GPT-2 rerun), then Llama-3.1-8B (device fix). Every failed step is logged as "=== FAILED".
cd /lambda/nfs/AT1/punctuation-summary-info || exit 1
PY=$HOME/venvs/punctuation-summary-info/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== $*"; "$@" || { echo "=== FAILED: $*"; return 1; }; }

while pgrep -f "scripts/m3_probes.py --all" >/dev/null; do sleep 60; done  # the GPT-2 hardware rerun holds the GPU
echo "=== GPU free, queue 2 starts $(date -u)"

for m in pythia-2.8b pythia-6.9b; do step $PY scripts/m6_scale.py run $m behavior knockout accept patch probe long; done
for m in gpt2 gpt-neo-125m pythia-160m pythia-410m pythia-1.4b; do step $PY scripts/m6_scale.py run $m long; done
echo "=== LADDER2_DONE $(date -u)"

L=llama-3.1-8b
step $PY scripts/m6_scale.py verify $L && step $PY scripts/m6_scale.py bf16check $L && step $PY scripts/m6_scale.py screen $L \
  || { echo "=== LLAMA_STOPPED: setup failed"; exit 1; }
step $PY scripts/m6_ladder_items.py $L $L || { echo "=== LLAMA_STOPPED: items fail the bar"; exit 1; }
step $PY scripts/m6_long_pairs.py pairs $L && step $PY scripts/m6_long_check.py $L $L \
  || { echo "=== LLAMA_STOPPED: long pairs"; exit 1; }
step $PY scripts/m6_scale.py run $L && echo "=== LLAMA_DONE $(date -u)" || echo "=== LLAMA_STOPPED: run failed"
