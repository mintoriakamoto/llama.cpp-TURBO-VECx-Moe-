#!/usr/bin/env python3
"""
MoE-EP validator — predictive expert paging (SVMI roadmap phase 6, §15).

MoE models activate top-k experts per token per layer, but expert popularity is
heavy-tailed and consecutive tokens reuse experts. That makes expert weights the
ideal streaming target: keep the popular set resident, page the tail, and prefetch
the predicted set from the router's early signal instead of stalling on demand.

This simulator draws expert activations with the structure real routers show
(Zipf popularity + temporal stickiness + topic drift) and compares residency
policies by PCIe bytes/token — the number that bounds streamed decode speed:

  static   pin the globally most popular experts, stream everything else
  lru      page experts in on use, evict least-recently-used
  lookahead lru + prefetch predicted experts off the critical path
           (router-logit lookahead; hit rate is the honesty knob)

Usage:
  python3 scripts/svmi-expertpage.py --profile mixtral-8x7b --gpu 3060
  python3 scripts/svmi-expertpage.py --profile qwen3-30b-a3b --resident-gib 4
  python3 scripts/svmi-expertpage.py --profile deepseek-v3 --lookahead-hit 0.8
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

GiB = 1024**3

# (n_layer, n_expert, top_k, expert MiB @Q4-class, attn+shared GiB resident anyway)
MOE_PROFILES = {
    "mixtral-8x7b":  (32,   8, 2, 168.0, 2.1),
    "qwen3-30b-a3b": (48, 128, 8,   2.8, 1.9),
    "deepseek-v3":   (61, 256, 8,  13.8, 8.2),
}
GPU_VRAM = {"1660ti": 6, "2080": 8, "2080ti": 11, "3060": 12, "3090": 24, "4090": 24}
PCIE_GBS = {"1660ti": 12.0, "2080": 12.0, "2080ti": 12.0, "3060": 24.0, "3090": 24.0, "4090": 24.0}


def simulate(n_layer, n_expert, top_k, args, rng):
    """Returns per-policy average experts *fetched on the critical path* per token."""
    # per-layer popularity: Zipf over a drifting permutation
    perms = [rng.permutation(n_expert) for _ in range(n_layer)]
    zipf = 1.0 / np.arange(1, n_expert + 1) ** args.zipf

    n_res = args.resident_experts
    static_set = [set(np.argsort(-zipf[np.argsort(p)])[:n_res].tolist()) for p in perms]
    lru = [list(range(min(n_res, n_expert))) for _ in range(n_layer)]
    prev = [set() for _ in range(n_layer)]

    fetch = {"static": 0, "lru": 0, "lookahead": 0}
    for _ in range(args.tokens):
        for layer in range(n_layer):
            if rng.random() < args.drift:
                perms[layer] = rng.permutation(n_expert)
            pop = np.empty(n_expert)
            pop[perms[layer]] = zipf
            if prev[layer]:                          # stickiness: recent experts recur
                idx = np.fromiter(prev[layer], dtype=int)
                pop[idx] += args.sticky * pop.max()
            pop /= pop.sum()
            used = rng.choice(n_expert, size=top_k, replace=False, p=pop)

            fetch["static"] += sum(1 for e in used if e not in static_set[layer])

            cache = lru[layer]
            miss = [e for e in used if e not in cache]
            fetch["lru"] += len(miss)
            # lookahead hides a fraction of LRU misses behind earlier layers' compute
            hidden = int(round(len(miss) * args.lookahead_hit))
            fetch["lookahead"] += len(miss) - hidden
            for e in used:                           # LRU update
                if e in cache:
                    cache.remove(e)
                cache.append(e)
            if len(cache) > n_res:
                del cache[:-n_res]
            prev[layer] = set(used.tolist())

    return {k: v / args.tokens for k, v in fetch.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", choices=sorted(MOE_PROFILES), default="mixtral-8x7b")
    ap.add_argument("--gpu", choices=sorted(GPU_VRAM), default="3060")
    ap.add_argument("--resident-gib", type=float, default=None,
                    help="VRAM for resident experts (default: what the GPU leaves free)")
    ap.add_argument("--tokens", type=int, default=2000, help="tokens simulated")
    ap.add_argument("--zipf", type=float, default=1.05, help="expert popularity skew")
    ap.add_argument("--sticky", type=float, default=0.35, help="token-to-token expert reuse")
    ap.add_argument("--drift", type=float, default=0.002, help="per-layer topic-switch prob/token")
    ap.add_argument("--lookahead-hit", type=float, default=0.7,
                    help="fraction of misses the router-logit prefetch hides (honesty knob)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    n_layer, n_expert, top_k, exp_mib, fixed_gib = MOE_PROFILES[args.profile]
    exp_bytes = exp_mib * 1024**2
    vram = GPU_VRAM[args.gpu]
    pcie = PCIE_GBS[args.gpu]
    if args.resident_gib is None:
        args.resident_gib = max(0.5, vram - fixed_gib - 2.0)   # kv + overhead reserve ~2 GiB
    total_res = int(args.resident_gib * GiB / exp_bytes)
    args.resident_experts = max(top_k, total_res // n_layer)

    total_experts_gib = n_layer * n_expert * exp_bytes / GiB
    rng = np.random.default_rng(args.seed)
    res = simulate(n_layer, n_expert, top_k, args, rng)

    print(f"model        : {args.profile}  ({n_layer} layers x {n_expert} experts, top-{top_k}, "
          f"{exp_mib:.0f} MiB/expert, {total_experts_gib:.0f} GiB of experts total)")
    print(f"gpu          : {args.gpu} ({vram} GiB, {pcie:.0f} GB/s eff PCIe)  |  resident budget "
          f"{args.resident_gib:.1f} GiB = {args.resident_experts}/{n_expert} experts/layer")
    print(f"trace        : zipf {args.zipf}, sticky {args.sticky}, drift {args.drift}, "
          f"lookahead hit {args.lookahead_hit:.0%}\n")

    demand_base = n_layer * top_k   # no residency at all
    print(f"{'policy':>10} {'fetched/tok':>12} {'MB/tok':>8} {'PCIe-bound tok/s':>17} {'vs none':>8}")
    for policy in ("static", "lru", "lookahead"):
        f = res[policy]
        mb = f * exp_bytes / 1e6
        tps = pcie * 1e9 / (f * exp_bytes) if f > 0 else float("inf")
        base_mb = demand_base * exp_bytes / 1e6
        print(f"{policy:>10} {f:>12.2f} {mb:>8.1f} "
              f"{('%16.1f' % tps) if f > 0 else '   all resident':>17} {base_mb / mb if mb else 0:>7.1f}x")
    print(f"{'none':>10} {demand_base:>12.2f} {demand_base * exp_bytes / 1e6:>8.1f} "
          f"{pcie * 1e9 / (demand_base * exp_bytes):>17.1f} {'1.0x':>8}")

    print("\nnotes: 'lookahead' assumes the router's early signal (previous layer's hidden")
    print("state) predicts the expert set well enough to prefetch — the measured knob is")
    print("--lookahead-hit; trace it on a real model before believing it. Fetches share")
    print("the SVMI upload queues, so hidden misses cost bandwidth but not latency.")
    print("PCIe-bound tok/s ignores compute; the real rate is min(this, compute rate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
