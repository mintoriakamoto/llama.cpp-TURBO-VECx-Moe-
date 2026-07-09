#!/usr/bin/env bash
# SVMI correctness check: prove weight streaming is token-identical to the baseline.
#
# The whole premise of SVMI is that only the *transport* of the weights changes —
# the same kernels run on the same data — so greedy decoding must produce the exact
# same tokens with and without streaming. This script runs the same prompt twice and
# diffs the output; optionally it also compares perplexity so a numeric equality check
# backs up the token diff.
#
# Usage:
#   scripts/svmi-verify.sh -m model.gguf [-ngl 99] [-ot 'blk\.(...)\.ffn_.*=CPU'] \
#                          [--ppl-file wiki.test.raw] [extra llama-cli args...]
#
# Notes:
#   * Uses greedy sampling (temp 0, fixed seed) so any divergence is a real bug.
#   * Runs on whatever backend you built; on CPU-only builds it still validates the
#     scheduler bookkeeping, though the interesting overlap only happens on GPU.
#   * Defaults to llama-completion (non-interactive); set LLAMA_CLI to override.
set -euo pipefail

BIN=${LLAMA_CLI:-./build/bin/llama-completion}
PPL_BIN=${LLAMA_PERPLEXITY:-./build/bin/llama-perplexity}
SLOTS=${SVMI_SLOTS:-8}

PPL_FILE=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --ppl-file) PPL_FILE=$2; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

if [ ${#ARGS[@]} -eq 0 ]; then
    grep '^#' "$0" | head -18 | sed 's/^# \{0,1\}//'
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found or not executable (set LLAMA_CLI or build first)" >&2
    exit 1
fi

PROMPT=${SVMI_PROMPT:-"The theory of relativity states that"}
GEN=(-n "${SVMI_NGEN:-64}" -p "$PROMPT" --seed 1234 --temp 0 --no-display-prompt --simple-io -no-cnv)

run_tokens() {
    # $1 = label; remaining = env assignments
    local label=$1; shift
    env "$@" "$BIN" "${ARGS[@]}" "${GEN[@]}" 2>/dev/null > "/tmp/svmi-verify-$label.txt"
}

echo "== token-identity check (greedy, seed 1234) =="
run_tokens baseline -u GGML_CUDA_REGISTER_HOST -u GGML_SCHED_STREAM_WEIGHTS
run_tokens streamed GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_STREAM_WEIGHTS="$SLOTS"

if diff -u /tmp/svmi-verify-baseline.txt /tmp/svmi-verify-streamed.txt > /tmp/svmi-verify.diff; then
    echo "PASS: streamed output is byte-identical to baseline"
else
    echo "FAIL: streamed output diverged from baseline" >&2
    cat /tmp/svmi-verify.diff >&2
    exit 1
fi

if [ -n "$PPL_FILE" ]; then
    if [ ! -x "$PPL_BIN" ]; then
        echo "warn: $PPL_BIN not found, skipping perplexity comparison" >&2
        exit 0
    fi
    echo
    echo "== perplexity check =="
    ppl() {
        local label=$1; shift
        env "$@" "$PPL_BIN" "${ARGS[@]}" -f "$PPL_FILE" 2>/dev/null \
            | grep -oE 'PPL = [0-9.]+' | tail -1 | awk '{print $3}'
    }
    P_BASE=$(ppl baseline -u GGML_CUDA_REGISTER_HOST -u GGML_SCHED_STREAM_WEIGHTS)
    P_STREAM=$(ppl streamed GGML_CUDA_REGISTER_HOST=1 GGML_SCHED_STREAM_WEIGHTS="$SLOTS")
    echo "baseline PPL = $P_BASE"
    echo "streamed PPL = $P_STREAM"
    if [ "$P_BASE" = "$P_STREAM" ]; then
        echo "PASS: perplexity identical"
    else
        echo "FAIL: perplexity differs ($P_BASE vs $P_STREAM)" >&2
        exit 1
    fi
fi
