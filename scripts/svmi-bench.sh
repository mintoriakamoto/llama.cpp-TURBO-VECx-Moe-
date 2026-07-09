#!/usr/bin/env bash
# SVMI benchmark harness: baseline vs pinned-host vs full weight streaming.
#
# Runs llama-bench three times with identical settings and prints a comparison:
#   1. baseline           - stock scheduler, synchronous weight uploads
#   2. pinned             - + page-locked host weights (GGML_CUDA_REGISTER_HOST)
#   3. streamed           - + SVMI upload queues and staging ring (--stream-weights)
#
# Usage:
#   scripts/svmi-bench.sh -m model.gguf [-ngl 99] [-ot 'blk\.(...)\.ffn_.*=CPU'] \
#                         [-p 2048] [-n 32] [-r 3] [extra llama-bench args...]
#
# For a 70B-under-20GB run, generate placement flags first:
#   python3 scripts/svmi-plan.py model.gguf --vram-budget 19
set -euo pipefail

BIN=${LLAMA_BENCH:-./build/bin/llama-bench}
if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found or not executable (set LLAMA_BENCH or build first)" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    grep '^#' "$0" | head -14 | sed 's/^# \{0,1\}//'
    exit 1
fi

run() {
    local label=$1; shift
    echo "=== $label ==="
    "$@" "$BIN" "${ARGS[@]}" 2>/dev/null | tee /tmp/svmi-bench-"$label".txt
    echo
}

ARGS=("$@")

run baseline env -u GGML_CUDA_REGISTER_HOST -u GGML_SCHED_STREAM_WEIGHTS
run pinned   env GGML_CUDA_REGISTER_HOST=1
run streamed env GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_STREAM_WEIGHTS=8

echo "=== summary (t/s columns) ==="
for label in baseline pinned streamed; do
    echo "--- $label"
    grep -E 'pp[0-9]+|tg[0-9]+' /tmp/svmi-bench-"$label".txt || true
done
