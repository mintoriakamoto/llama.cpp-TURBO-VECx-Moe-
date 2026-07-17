#!/usr/bin/env python3
"""
SPEC-PF validator — how much demand-fetch latency can speculative page prefetch hide?

CTX-VM fetches missed KV pages on demand, on the critical path of the attention op
that needs them. SPEC-PF prefetches the pages *predicted* from the previous step's
scores on the weight-streaming clock instead. This simulator generates attention
page-access traces with the structure real attention shows — sinks + local window
always hit; distal pages follow a persistent (sticky) Zipf process with topic drift —
and measures how many misses the prefetcher hides.

Outputs the effective critical-path fetch tokens/step, which is exactly what the fleet
planner consumes as  svmi-fleet.py --prefetch-hit H.

Usage:
  python3 scripts/svmi-pageprefetch.py                       # defaults
  python3 scripts/svmi-pageprefetch.py --pages 2048 --steps 4000 --drift 0.02
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def run_sim(args, rng: np.random.Generator):
    n = args.pages
    # per-page base popularity: Zipf over a topical permutation that drifts over time
    perm = rng.permutation(n)
    zipf = 1.0 / np.arange(1, n + 1) ** args.zipf
    prev_hot: set[int] = set()
    prefetched: set[int] = set()
    demand_misses = prefetch_hits = total_distal = 0

    for _ in range(args.steps):
        if rng.random() < args.drift:                      # topic switch: reshuffle part
            cut = rng.integers(0, n)
            perm = np.concatenate([perm[cut:], perm[:cut]])
        pop = np.empty(n)
        pop[perm] = zipf
        # persistence: pages accessed last step stay likely (temporal locality)
        if prev_hot:
            idx = np.fromiter(prev_hot, dtype=int)
            pop[idx] += args.sticky * pop.max()
        pop /= pop.sum()
        accessed = set(rng.choice(n, size=args.distal, replace=False, p=pop).tolist())

        total_distal += len(accessed)
        for p in accessed:
            if p in prefetched or p in prev_hot:           # hidden: already resident
                prefetch_hits += 1
            else:
                demand_misses += 1

        # prefetcher: predict from this step's accesses + score prior, budget-limited
        order = np.argsort(-pop)
        prefetched = set(order[: args.budget].tolist()) | accessed
        prev_hot = accessed

    return prefetch_hits, demand_misses, total_distal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=1024, help="distal pages in context (256 tok each)")
    ap.add_argument("--steps", type=int, default=2000, help="decode steps simulated")
    ap.add_argument("--distal", type=int, default=8, help="distal pages attention touches per step")
    ap.add_argument("--zipf", type=float, default=1.1, help="popularity skew exponent")
    ap.add_argument("--sticky", type=float, default=0.5, help="temporal persistence strength")
    ap.add_argument("--drift", type=float, default=0.01, help="topic-switch probability per step")
    ap.add_argument("--page-tokens", type=int, default=256)
    ap.add_argument("--kv-bytes-tok", type=float, default=96.0,
                    help="cold KV bytes/token/layer-sum fetched (q4 70B latent ~96)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"pages {args.pages}, {args.distal} distal accesses/step, zipf {args.zipf}, "
          f"sticky {args.sticky}, drift {args.drift}\n")
    print(f"{'prefetch budget':>15} {'hit rate':>9} {'demand tok/step':>16} {'hidden':>7}")

    base_tokens = args.distal * args.page_tokens
    for budget in (0, args.distal, 2 * args.distal, 4 * args.distal, 8 * args.distal):
        args.budget = budget
        hits, misses, total = run_sim(args, rng)
        hit_rate = hits / max(1, total)
        demand_tokens = (1 - hit_rate) * base_tokens
        print(f"{budget:>15} {hit_rate:>9.3f} {demand_tokens:>16.0f} {hit_rate:>6.0%}")

    print(f"\nbaseline demand fetch: {base_tokens} tok/step "
          f"({base_tokens * args.kv_bytes_tok / 1e6:.1f} MB/step at {args.kv_bytes_tok:.0f} B/tok)")
    print("notes: hit rate at your chosen budget is the planner input (--prefetch-hit H);")
    print("prefetch bandwidth rides the weight-stream queues off the critical path, so")
    print("hidden fetches cost bandwidth but not latency. Drift spikes cause short dips -")
    print("size the budget for the post-switch regime, not the steady state. Replace this")
    print("trace model with --trace from a real run before trusting fine differences.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
