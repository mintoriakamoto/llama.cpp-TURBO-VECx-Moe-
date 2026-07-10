#!/usr/bin/env python3
"""
SVMI fleet planner — many agents on one GPU (MAVM capacity / throughput / cost model)

Answers the multi-agent question: on a single small-VRAM GPU, how many concurrent
agents (sequences) can share one streamed model, what does each additional agent cost,
and how fast does the fleet run? It quantifies the "Multi-Agent Virtual Memory" (MAVM)
design from docs/svmi-research.md, mapping each goal to a number:

  * bigger models        -> weights streamed, only a resident hot set in VRAM
  * more agents          -> weights are shared once; a new agent costs only its KV
  * cheaper VRAM         -> shared-prefix KV is stored once for the whole fleet
  * cheaper tokens       -> the shared system prompt is prefilled once, not per agent
  * faster / no idle     -> stream-once-serve-many batches agents onto one PCIe pass,
                            and idle agents' KV spills to host to free VRAM
  * + speculation        -> BitSpec amortizes each streamed pass over many tokens

Everything modeled here is token-identical: shared weights, shared-prefix KV
(copy-on-write), batched streaming, KV spill, and speculative verification all produce
the same tokens a single-agent full-VRAM run would.

Method: reads the real model dimensions from the GGUF (layers, heads, kv-heads, embd),
then applies standard capacity/bandwidth formulas. Numbers are analytical estimates, not
a substitute for measuring on hardware, but they are grounded in the actual model shape.

Usage:
  python3 scripts/svmi-fleet.py model.gguf --gpu 3060 \
      --agents 1,2,4,8,16 --ctx 8192 --shared-prompt 2048 --gen 256
  python3 scripts/svmi-fleet.py --profile 70b --gpu 2080ti --agents 1,2,4 --ctx 4096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

from gguf import GGUFReader  # noqa: E402

GiB = 1024**3

PCIE_BW = {"3.0-x8": 6.0, "3.0-x16": 12.0, "4.0-x8": 12.0, "4.0-x16": 24.0, "5.0-x16": 48.0}
# (vram GiB, pcie link, copy engines)
GPU_PRESETS = {
    "1660ti": (6, "3.0-x16", 1), "2060": (6, "3.0-x16", 1), "2070": (8, "3.0-x16", 1),
    "2080": (8, "3.0-x16", 1), "2080ti": (11, "3.0-x16", 1),
    "3060": (12, "4.0-x16", 2), "3070": (8, "4.0-x16", 2), "3080": (10, "4.0-x16", 2),
    "3090": (24, "4.0-x16", 2), "4070": (12, "4.0-x16", 2), "4090": (24, "4.0-x16", 2),
}
KV_BPE = {"f16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q4_0": 0.5625}

# synthetic profiles (Q4_K_M-class weights) so a fleet can be planned without the GGUF:
# (n_layer, n_embd, n_head, n_head_kv, total weight GiB, ffn frac, attn frac)
MODEL_PROFILES = {
    "7b":  (32, 4096, 32, 32,  3.9, 0.62, 0.30),
    "8b":  (32, 4096, 32,  8,  4.6, 0.55, 0.22),
    "13b": (40, 5120, 40, 40,  7.4, 0.63, 0.30),
    "70b": (80, 8192, 64,  8, 39.6, 0.70, 0.25),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES),
                    help="synthetic Q4_K_M-class model profile instead of a GGUF")
    ap.add_argument("--gpu", choices=sorted(GPU_PRESETS), help="consumer GPU preset")
    ap.add_argument("--vram-budget", type=float, help="usable VRAM GiB (overrides --gpu)")
    ap.add_argument("--pcie", choices=sorted(PCIE_BW), help="PCIe link (overrides --gpu)")
    ap.add_argument("--display-reserve", type=float, default=1.0)
    ap.add_argument("--agents", default="1,2,4,8,16,32", help="agent counts to tabulate")
    ap.add_argument("--ctx", type=int, default=8192, help="per-agent context length")
    ap.add_argument("--shared-prompt", type=int, default=0,
                    help="tokens of a system prompt shared by all agents (prefix KV dedup)")
    ap.add_argument("--gen", type=int, default=256, help="tokens generated per agent (for cost model)")
    ap.add_argument("--kv-type", choices=list(KV_BPE), default="q8_0")
    ap.add_argument("--overhead", type=float, default=1.25, help="activation/compute reserve GiB")
    ap.add_argument("--stream-slots", type=int, default=8)
    ap.add_argument("--bitspec-tokens", type=float, default=7.0,
                    help="mean tokens accepted per streamed pass under BitSpec (see svmi-bitspec.py)")
    ap.add_argument("--idle-frac", type=float, default=0.5,
                    help="fraction of agents idle at a given instant (KV spillable to host)")
    args = ap.parse_args()

    # resolve GPU
    n_queues = 2
    if args.gpu:
        vram, link, n_queues = GPU_PRESETS[args.gpu]
        if args.vram_budget is None:
            args.vram_budget = max(0.0, vram - args.display_reserve)
        if args.pcie is None:
            args.pcie = link
    if args.vram_budget is None:
        ap.error("provide --vram-budget or --gpu")
    pcie_bw = PCIE_BW[args.pcie or "4.0-x16"]

    if args.profile:
        n_layer, n_embd, n_head, n_head_kv, w_gib, ffn_frac, attn_frac = MODEL_PROFILES[args.profile]
        arch = f"profile:{args.profile}"
        model_name = f"{args.profile} (synthetic Q4_K_M-class)"
        total_w = int(w_gib * GiB)
        ffn_bytes = int(total_w * ffn_frac)
        attn_bytes = int(total_w * attn_frac)
        # one FFN weight tensor ~ ffn bytes / (3 tensors per layer * layers)
        max_stream_tensor = ffn_bytes // (3 * n_layer)
    elif args.model:
        reader = GGUFReader(args.model)

        def fi(key: str, default: int = 0) -> int:
            f = reader.get_field(key)
            return int(f.parts[f.data[0]][0]) if f is not None else default

        arch = ""
        f = reader.get_field("general.architecture")
        if f is not None:
            arch = bytes(f.parts[f.data[0]]).decode("utf-8")

        n_layer   = fi(f"{arch}.block_count")
        n_embd    = fi(f"{arch}.embedding_length")
        n_head    = fi(f"{arch}.attention.head_count") or 1
        n_head_kv = fi(f"{arch}.attention.head_count_kv") or n_head
        model_name = Path(args.model).name

        total_w = sum(int(t.n_bytes) for t in reader.tensors)
        ffn_bytes = sum(int(t.n_bytes) for t in reader.tensors if ".ffn_" in t.name)
        attn_bytes = sum(int(t.n_bytes) for t in reader.tensors if ".attn_" in t.name)
        max_stream_tensor = max((int(t.n_bytes) for t in reader.tensors
                                 if ".ffn_" in t.name or ".attn_" in t.name), default=0)
    else:
        ap.error("provide a GGUF path or --profile")

    head_dim = n_embd // n_head if n_head else 0
    streamable_w = ffn_bytes + attn_bytes        # scheduler streams any big matmul weight
    min_resident = total_w - streamable_w        # embeddings / norms / output head

    # KV bytes per token (whole model) and per agent
    kv_per_tok = 2 * head_dim * n_head_kv * n_layer * KV_BPE[args.kv_type]
    kv_shared = int(kv_per_tok * args.shared_prompt)          # stored once for the fleet
    kv_unique = int(kv_per_tok * max(0, args.ctx - args.shared_prompt))  # per agent

    ring = args.stream_slots * max_stream_tensor
    budget = int(args.vram_budget * GiB)
    fixed = ring + int(args.overhead * GiB)

    print(f"model        : {model_name} ({arch}, {n_layer} layers, {total_w/GiB:.2f} GiB)")
    if args.gpu:
        print(f"gpu          : {args.gpu}  ({args.pcie}, {n_queues} H2D copy engine(s))")
    print(f"vram budget  : {budget/GiB:.2f} GiB   |  streamable {streamable_w/GiB:.2f} / "
          f"resident-min {min_resident/GiB:.2f} GiB")
    print(f"kv/token     : {kv_per_tok/1024:.1f} KiB (whole model, {args.kv_type})   "
          f"per-agent ctx {args.ctx} -> {kv_unique/GiB:.3f} GiB "
          f"(+{kv_shared/GiB:.3f} GiB shared prefix)")
    print(f"fixed reserve: ring {ring/GiB:.2f} + overhead {args.overhead:.2f} GiB\n")

    print(f"{'agents':>6} {'resident W':>10} {'streamed W':>10} {'KV total':>9} "
          f"{'fits?':>6} {'stream GiB/tok':>13} {'agg tok/s*':>10} {'per-agent':>9}")

    agent_counts = [int(a) for a in args.agents.split(",") if a.strip()]
    max_fit = 0
    for n in agent_counts:
        kv_total = kv_shared + n * kv_unique
        avail_for_w = budget - fixed - kv_total
        if avail_for_w < min_resident:
            print(f"{n:>6} {'—':>10} {'—':>10} {kv_total/GiB:8.2f}G {'NO':>6}  "
                  f"(KV+reserve alone exceed budget; need KV spill or fewer agents)")
            continue
        max_fit = n
        # everything that fits stays resident; the remainder of the streamable set streams
        resident_w = min(total_w, avail_for_w)
        streamed_w = total_w - resident_w
        # one batched streamed pass serves all n active agents (stream-once-serve-many)
        stream_per_tok = streamed_w  # bytes moved per *pass*, shared across the batch
        base_pass_per_s = pcie_bw * 1e9 / streamed_w if streamed_w else float("inf")
        # aggregate decode tok/s ~ passes/s * agents batched * accepted tokens/pass (BitSpec)
        if streamed_w == 0:
            agg = float("inf")
            per_agent = float("inf")
        else:
            agg = base_pass_per_s * n * args.bitspec_tokens
            per_agent = agg / n
        fits = "yes"
        print(f"{n:>6} {resident_w/GiB:9.2f}G {streamed_w/GiB:9.2f}G {kv_total/GiB:8.2f}G "
              f"{fits:>6} {stream_per_tok/GiB:12.3f}  "
              f"{'resident' if streamed_w==0 else f'{agg:8.1f}'} "
              f"{'—' if streamed_w==0 else f'{per_agent:8.1f}'}")

    print("\n* aggregate decode tok/s across the whole fleet, streaming-bound region, with "
          "stream-once-serve-many + BitSpec\n  (mean {:.0f} tokens/pass); it plateaus once "
          "compute, not PCIe, becomes the bottleneck.".format(args.bitspec_tokens))

    # marginal cost of one more agent, and the naive comparison
    print("\n-- economics --")
    print(f"marginal VRAM per additional agent : {kv_unique/GiB:.3f} GiB (its unique KV only; "
          f"weights are shared)")
    naive_per_agent = (total_w + kv_shared + kv_unique) / GiB
    print(f"naive (no sharing) per agent       : {naive_per_agent:.2f} GiB "
          f"(full model copy each) -> MAVM is ~{naive_per_agent/(kv_unique/GiB):.0f}x denser in agents/GiB")

    if args.shared_prompt > 0:
        big = max(agent_counts)
        print(f"prefix cache (shared prompt {args.shared_prompt} tok): prefilled once, not per "
              f"agent -> saves {(big-1)*args.shared_prompt:,} prompt tokens at {big} agents "
              f"({(big-1)*args.shared_prompt/max(1,big*(args.shared_prompt+args.gen)):.0%} of total token work)")

    reclaim = args.idle_frac * kv_unique
    print(f"idle reclamation: with {args.idle_frac:.0%} agents idle, spilling their KV frees "
          f"{args.idle_frac:.0%} x per-agent KV each = {reclaim/GiB:.3f} GiB/idle-agent back to "
          f"weights or new agents (no GPU idle: freed VRAM promotes streamed layers resident)")

    if max_fit:
        print(f"\nverdict: up to ~{max_fit} concurrent agents fit this budget at ctx {args.ctx}; "
              f"beyond that, enable KV host-spill (--idle-frac) or raise the shared-prompt fraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
