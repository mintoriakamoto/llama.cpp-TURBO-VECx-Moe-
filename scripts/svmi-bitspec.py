#!/usr/bin/env python3
"""
SVMI BitSpec feasibility study — bit-plane self-speculative decoding (go/no-go)

BitSpec is a draft model you get for free: take the target model's own quantized
weights and truncate them to a few bits per value. The truncated weights are small
enough to stay *resident* in VRAM (a 3-bit copy of a model is ~19% of its f16 size),
so a draft forward pass needs zero PCIe traffic. The full-precision weights — the ones
that must be streamed over PCIe — are then touched only once per *verification* pass,
which accepts/rejects a whole block of drafted tokens at once. That amortizes the
dominant streaming cost across many tokens, which is the only way to move a
memory-streamed decoder off its `bytes / PCIe_bandwidth` floor without changing outputs.

Unlike Medusa/EAGLE (trained heads) or a separate small draft model (extra weights,
separate tokenizer risk), BitSpec's draft is bit-identical in architecture and vocab to
the target and requires no training — it is literally the target at lower precision.

This script answers the one empirical question that decides whether BitSpec is worth
building: **how often does the low-bit draft agree with full precision on the token
decision?** That agreement rate is the per-token speculative acceptance probability,
which sets the achievable speedup.

Method (GPU-free, uses real weights):
  * The decision that picks the next token is argmax over the output projection
    (lm_head): token = argmax(W_out @ h). We measure how often
    argmax(W_out_trunc @ h) == argmax(W_out_full @ h), and how often the full-precision
    winner is within the draft's top-k (top-k matters for *tree* speculation, where the
    draft proposes k candidates per step).
  * Input hidden states h are sampled from the model's own token-embedding rows — real
    vectors in the model's hidden space, a far better in-distribution proxy than random
    noise (argmax is scale-invariant, so no normalization is needed).
  * The draft weights are modeled as a per-output-row symmetric b-bit weight-only
    quantization of the dequantized full weights (a faithful stand-in for an aggressive
    low-bit resident draft).

Honesty notes:
  * This measures acceptance at the *final projection* only. A real BitSpec draft runs
    the whole low-bit network, so per-layer error compounds; the lm_head agreement here
    is an optimistic upper bound on end-to-end draft fidelity, but it is the single most
    decision-relevant layer and the right first go/no-go.
  * Greedy (argmax) speculation is assumed. Distribution-preserving sampling uses the
    standard accept/reject rule and tracks these numbers closely at low temperature.

Usage:
  python3 scripts/svmi-bitspec.py model.gguf [--bits 2,3,4] [--trials 512] \
                                   [--topk 1,2,4,8] [--pcie-gbps 25] [--model-gb 40]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402
import gguf.quants as gq  # noqa: E402


def dequant(tensor) -> np.ndarray:
    qt = GGMLQuantizationType(tensor.tensor_type)
    return gq.dequantize(tensor.data, qt).astype(np.float32)


def find_tensor(reader, *substrings):
    for t in reader.tensors:
        name = t.name.lower()
        if any(s in name for s in substrings):
            return t
    return None


def rowwise_quant(w: np.ndarray, bits: int) -> np.ndarray:
    """Per-row symmetric weight-only quantization to `bits` bits, dequantized back to
    f32. Models a low-bit resident draft weight. Row = one output neuron."""
    qmax = (1 << (bits - 1)) - 1  # e.g. bits=3 -> levels in [-3, 3]
    if qmax < 1:
        qmax = 1
    absmax = np.abs(w).max(axis=1, keepdims=True)
    absmax[absmax == 0] = 1.0
    scale = absmax / qmax
    q = np.round(w / scale).clip(-qmax, qmax)
    return (q * scale).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--bits", default="2,3,4", help="draft bit-widths to test (default 2,3,4)")
    ap.add_argument("--trials", type=int, default=512, help="number of hidden-state samples (default 512)")
    ap.add_argument("--topk", default="1,2,4,8", help="top-k containment levels to report (default 1,2,4,8)")
    ap.add_argument("--pcie-gbps", type=float, default=None,
                    help="effective PCIe H2D GB/s for the speedup model "
                    "(default 24; PCIe 3.0 x16 cards like the 2080/2080Ti/1660Ti are ~12)")
    ap.add_argument("--gpu", choices=["1660ti", "2060", "2070", "2080", "2080ti",
                                      "3060", "3070", "3090", "4090"],
                    help="preset that sets --pcie-gbps for a consumer GPU "
                         "(Turing cards are PCIe 3.0 ~12 GB/s; Ampere+ ~24)")
    ap.add_argument("--model-gb", type=float, default=40.0, help="streamed weight size (GiB) for the speedup model")
    ap.add_argument("--draft-len", type=int, default=8, help="max drafted tokens per verification pass (default 8)")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    bits_list = [int(b) for b in args.bits.split(",") if b.strip()]
    topk_list = sorted(int(k) for k in args.topk.split(",") if k.strip())
    if args.pcie_gbps is None:
        # Turing consumer cards are PCIe 3.0 (~12 GB/s pinned); everything else defaults to 24
        pcie3 = {"1660ti", "2060", "2070", "2080", "2080ti"}
        args.pcie_gbps = 12.0 if args.gpu in pcie3 else 24.0
    rng = np.random.default_rng(args.seed)

    reader = GGUFReader(args.model)

    out = find_tensor(reader, "output.weight", "lm_head")
    emb = find_tensor(reader, "token_embd", "tok_embeddings", "embed_tokens")
    if emb is None:
        print("could not find a token-embedding tensor")
        return 1
    # tied embeddings: many models drop output.weight and reuse the embedding matrix
    if out is None:
        out = emb
        tied = True
    else:
        tied = out.name == emb.name

    W = dequant(out)          # (vocab, hidden)
    E = dequant(emb)          # (vocab, hidden)
    vocab, hidden = W.shape
    print(f"model:   {Path(args.model).name}")
    print(f"lm_head: {out.name}  {GGMLQuantizationType(out.tensor_type).name}  "
          f"(vocab={vocab}, hidden={hidden}){'  [tied to embeddings]' if tied else ''}")
    print(f"inputs:  {args.trials} hidden states sampled from '{emb.name}' rows\n")

    # sample real hidden-space directions from the embedding rows
    idx = rng.choice(E.shape[0], size=min(args.trials, E.shape[0]), replace=False)
    H = E[idx]                                    # (trials, hidden)

    logits_full = H @ W.T                          # (trials, vocab)
    top1_full = logits_full.argmax(axis=1)
    maxk = max(topk_list)
    # indices of the draft's top-maxk, per trial, computed once per bit-width below

    print(f"{'bits':>4} {'draft MiB':>9} {'resident%':>9} "
          + " ".join(f'top{k:<2} acc' for k in topk_list))
    results = {}
    for bits in bits_list:
        Wq = rowwise_quant(W, bits)
        logits_draft = H @ Wq.T
        # top-k of the draft
        part = np.argpartition(-logits_draft, kth=maxk - 1, axis=1)[:, :maxk]
        row = np.arange(H.shape[0])[:, None]
        # order those maxk by score so top-k prefixes are correct
        order = np.argsort(-logits_draft[row, part], axis=1)
        draft_topk = part[row, order]              # (trials, maxk) sorted best-first

        accs = []
        for k in topk_list:
            hit = (draft_topk[:, :k] == top1_full[:, None]).any(axis=1)
            accs.append(hit.mean())
        results[bits] = dict(zip(topk_list, accs))
        draft_mib = vocab * hidden * bits / 8 / 1024**2
        resident_pct = bits / 16.0 * 100
        print(f"{bits:>4} {draft_mib:9.1f} {resident_pct:8.1f}% "
              + " ".join(f'{a:8.1%}' for a in accs))

    # translate top-1 acceptance into a decode-speedup estimate under the bandwidth model
    L = args.draft_len
    base_tps = args.pcie_gbps * 1e9 / (args.model_gb * 1024**3)
    print(f"\n-- decode speedup model (greedy chain, draft length {L}) --")
    print(f"PCIe floor: {args.model_gb} GiB / {args.pcie_gbps} GB/s = "
          f"{1/base_tps:.2f} s per streamed full-model pass ({base_tps:.2f} tok/s unspeculated)")
    print(f"{'bits':>4} {'accept(top1)':>12} {'E[tokens/pass]':>15} {'~speedup':>9} {'est tok/s':>10}")
    for bits in bits_list:
        a = results[bits][1]
        # one verification pass drafts up to L tokens; accepted run before the first
        # rejection is truncated-geometric, plus the always-correct verified token:
        #   E[tokens] = sum_{i=0..L} a^i = (1 - a^{L+1}) / (1 - a)
        e_tokens = (1.0 - a ** (L + 1)) / max(1e-6, 1.0 - a)
        # each pass streams the model once; the L resident draft steps are ~free next to PCIe
        print(f"{bits:>4} {a:11.1%} {e_tokens:15.2f} {e_tokens:8.2f}x {e_tokens*base_tps:10.2f}")

    best_bits = min(bits_list)
    best_acc = results[best_bits][1]
    print()
    if best_acc >= 0.6:
        print(f"verdict: PROCEED — even a {best_bits}-bit resident draft agrees with full "
              f"precision {best_acc:.0%} of the time at the token decision; "
              f"self-speculation should amortize streaming well.")
    else:
        for bits in bits_list:
            if results[bits][1] >= 0.6:
                print(f"verdict: PROCEED at >= {bits}-bit draft "
                      f"(top-1 {results[bits][1]:.0%}); {best_bits}-bit is too lossy.")
                break
        else:
            print("verdict: MARGINAL — top-1 agreement is low; prefer tree drafts (see "
                  "top-k columns) or a higher-bit draft, and re-check on real activations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
