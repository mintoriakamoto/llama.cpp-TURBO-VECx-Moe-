#!/usr/bin/env python3
"""
CTX-VM v2 validator — product-quantized landmarks: recall vs index size.

CTX-VM's page table keeps one f16 key landmark per (page, layer, kv-head); at 1M
tokens that index alone is 640 MiB for a 70B. This tool measures what compressing the
landmarks with product quantization costs in top-k page recall, on synthetic keys with
the clustered structure real semantic pages exhibit (mixture of Gaussians + drift),
and prints the index-size table that motivates the whole exercise.

Recall here is agreement with the *f16-landmark* ranking (the §8 baseline), not with
dense attention — PQ can only lose what the landmark abstraction already kept.

Usage:
  python3 scripts/svmi-pqindex.py                          # defaults: 512 pages, d=128
  python3 scripts/svmi-pqindex.py --pages 4096 --topk 16
  python3 scripts/svmi-pqindex.py --hard                   # near-isotropic keys (floor)
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

# (n_layer, n_embd, n_head, n_head_kv) — matches svmi-fleet.py profiles
PROFILES = {"8b": (32, 4096, 32, 8), "70b": (80, 8192, 64, 8)}
CTX_POINTS = (131072, 524288, 1048576)
PAGE_TOKENS = 256


def make_pages(n_pages: int, dim: int, hard: bool, rng: np.random.Generator):
    """Per-page landmark vectors: clustered with drift (default) or near-isotropic."""
    if hard:
        landmarks = rng.standard_normal((n_pages, dim))
    else:
        n_clusters = max(4, n_pages // 32)
        centers = rng.standard_normal((n_clusters, dim)) * 2.0
        assign = np.sort(rng.integers(0, n_clusters, n_pages))  # sorted = topical drift
        landmarks = centers[assign] + 0.6 * rng.standard_normal((n_pages, dim))
    return landmarks / np.linalg.norm(landmarks, axis=1, keepdims=True)


def pq_train(x: np.ndarray, m: int, rng: np.random.Generator, iters: int = 12):
    """Light k-means PQ: m sub-vectors, 256 centroids each."""
    n, d = x.shape
    dsub = d // m
    codebooks = np.empty((m, 256, dsub))
    codes = np.empty((n, m), dtype=np.uint8)
    for j in range(m):
        sub = x[:, j * dsub:(j + 1) * dsub]
        k = min(256, n)
        cent = sub[rng.choice(n, k, replace=False)].copy()
        for _ in range(iters):
            d2 = ((sub[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            a = d2.argmin(1)
            for c in range(k):
                mask = a == c
                if mask.any():
                    cent[c] = sub[mask].mean(0)
        codebooks[j, :k] = cent
        codebooks[j, k:] = cent[0]
        codes[:, j] = a
    return codebooks, codes


def pq_scores(q: np.ndarray, codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Asymmetric distance: query in float vs PQ-reconstructed landmarks."""
    m, _, dsub = codebooks.shape
    lut = np.einsum("md,mkd->mk", q.reshape(m, dsub), codebooks)  # per-sub dot tables
    return lut[np.arange(m)[:, None], codes.T].sum(0)


def recall_at_k(exact_rank: np.ndarray, approx_rank: np.ndarray, k: int) -> float:
    return len(set(exact_rank[:k]) & set(approx_rank[:k])) / k


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=512, help="pages in the simulated context")
    ap.add_argument("--dim", type=int, default=128, help="landmark dim (= head_dim)")
    ap.add_argument("--queries", type=int, default=256, help="query vectors to average over")
    ap.add_argument("--topk", type=int, default=8, help="pages fetched per step")
    ap.add_argument("--hard", action="store_true", help="near-isotropic keys (worst case)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    landmarks = make_pages(args.pages, args.dim, args.hard, rng)
    # queries live near existing content (attention looks things up, not random points)
    src = landmarks[rng.integers(0, args.pages, args.queries)]
    queries = src + 0.4 * rng.standard_normal((args.queries, args.dim))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    exact = queries @ landmarks.T
    exact_rank = np.argsort(-exact, axis=1)

    print(f"pages {args.pages} ({args.pages * PAGE_TOKENS:,} tokens), dim {args.dim}, "
          f"top-{args.topk}, {'HARD isotropic' if args.hard else 'clustered'} keys\n")
    print(f"{'index':>7} {'bytes/head/page':>15} {'recall@k':>9} {'recall@2k':>10}")
    for label, m in (("pq32", 32), ("pq16", 16), ("pq8", 8)):
        if args.dim % m:
            continue
        codebooks, codes = pq_train(landmarks, m, rng)
        r1 = r2 = 0.0
        for i in range(args.queries):
            s = pq_scores(queries[i], codebooks, codes)
            ar = np.argsort(-s)
            r1 += recall_at_k(exact_rank[i], ar, args.topk)
            r2 += recall_at_k(exact_rank[i], ar, 2 * args.topk)
        r1, r2 = r1 / args.queries, r2 / args.queries
        print(f"{label:>7} {m:>15} {r1:>9.3f} {r2:>10.3f}")
    print(f"{'f16':>7} {args.dim * 2:>15} {'1.000':>9} {'1.000':>10}   (baseline)")

    def fmt(b: float) -> str:
        return f"{b / 1024**2:7.0f}M"

    print("\n-- index VRAM at scale (per agent, landmarks only) --")
    print(f"{'model':>6} {'ctx':>10} {'f16':>9} {'pq16':>9} {'pq8':>9}")
    for pname, (n_layer, n_embd, n_head, n_head_kv) in PROFILES.items():
        head_dim = n_embd // n_head
        for ctx in CTX_POINTS:
            pages = ctx // PAGE_TOKENS
            f16 = pages * n_layer * n_head_kv * head_dim * 2
            pq16 = pages * n_layer * n_head_kv * 16
            pq8 = pages * n_layer * n_head_kv * 8
            print(f"{pname:>6} {ctx:>10,} {fmt(f16):>9} {fmt(pq16):>9} {fmt(pq8):>9}")

    print("\nnotes: recall is vs f16-landmark ranking; over-fetching (recall@2k) recovers")
    print("most residual misses at 2x fetch cost. Codebooks add ~256*dim*2 bytes/layer")
    print("(shared across pages - negligible). Feed the planner: svmi-fleet.py")
    print("--landmarks pq16. Floor-case check: rerun with --hard.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
