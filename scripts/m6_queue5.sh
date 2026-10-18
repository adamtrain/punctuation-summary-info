#!/bin/bash
# After m6_queue4.sh: rerun GPT-Neo-125M's knockout and acceptance steps, whose attention knockout was a silent no-op
# (TransformerBridge fires hook_attn_scores after GPT-Neo's attention; BRIEF.md decision log, 2026-10-18). Appends to
# queue2.log. The no-op results are kept as knockout_noop_*/accept_noop_* for the record.
cd /lambda/nfs/AT1/punctuation-summary-info || exit 1
PY=$HOME/venvs/punctuation-summary-info/bin/python
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
step() { echo "=== $*"; "$@" || { echo "=== FAILED: $*"; return 1; }; }

while pgrep -f "^bash scripts/m6_queue4.sh" >/dev/null; do sleep 60; done
echo "=== queue 5 starts $(date -u)"
d=outputs/models/gpt-neo-125m
for f in knockout.jsonl knockout_summary.json accept.json accept_mixed.jsonl; do
  [ -e $d/$f ] && [ ! -e $d/noop_$f ] && mv $d/$f $d/noop_$f
done
step $PY scripts/m6_scale.py run gpt-neo-125m knockout accept
echo "=== QUEUE5_DONE $(date -u)"
