#!/usr/bin/env python3
"""
CTX-VM v2 validator — product-quantized landmarks: recall vs index size.

CTX-VM's page table keeps one f16 key landmark per (page, layer, kv-head); at 1M
tokens that index alone is 640 MiB for a 70B. This tool measures what compressing the
landmarks with product quantization costs in top-k page recall, on synthetic keys with
the clustered structure real semantic pages exhibit (mixture of Gaussians + drift),
and prints the index-size table that motivates the whole exercise.

Ground truth is the EXACT page ranking (max query-key dot over the page's real keys),
so the table shows both losses at once: what the landmark abstraction gives up, and
what PQ-compressing the landmark gives up on top. Two landmark kinds are compared:
  mean    one normalized mean key per page (§8 baseline)
  minmax  Quest-style per-dim min/max bound (2 vectors; upper-bounds the page score,
          which is why Quest-class systems report recall close to full attention)
PCA-rotating keys before PQ (--rotate, OPQ-lite) tightens the codes further.

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


KEYS_PAGE = 16   # keys simulated per page (subsampled: recall stabilizes fast)


def make_pages(n_pages: int, dim: int, hard: bool, rng: np.random.Generator):
    """Per-page KEY sets: clustered with drift (default) or near-isotropic."""
    if hard:
        keys = rng.standard_normal((n_pages, KEYS_PAGE, dim))
    else:
        n_clusters = max(4, n_pages // 32)
        centers = rng.standard_normal((n_clusters, dim)) * 2.0
        assign = np.sort(rng.integers(0, n_clusters, n_pages))  # sorted = topical drift
        keys = centers[assign][:, None, :] + 0.6 * rng.standard_normal((n_pages, KEYS_PAGE, dim))
    return keys / np.linalg.norm(keys, axis=2, keepdims=True)


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
    ap.add_argument("--rotate", action="store_true", help="PCA-rotate before PQ (OPQ-lite)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    keys = make_pages(args.pages, args.dim, args.hard, rng)          # (pages, keys, dim)
    mean_lm = keys.mean(axis=1)
    mean_lm /= np.linalg.norm(mean_lm, axis=1, keepdims=True)
    kmin, kmax = keys.min(axis=1), keys.max(axis=1)                  # Quest bounds

    # queries live near existing content (attention looks things up, not random points)
    src = keys[rng.integers(0, args.pages, args.queries), rng.integers(0, KEYS_PAGE, args.queries)]
    queries = src + 0.4 * rng.standard_normal((args.queries, args.dim))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    # EXACT ground truth: best key dot product in each page
    exact = np.einsum("qd,pkd->qpk", queries, keys).max(axis=2)
    exact_rank = np.argsort(-exact, axis=1)

    # landmark rankings
    s_mean = queries @ mean_lm.T
    # Quest bound: sum_d max(q_d*min_d, q_d*max_d) - vectorized over pages
    s_minmax = np.maximum(queries[:, None, :] * kmin[None], queries[:, None, :] * kmax[None]).sum(axis=2)

    if args.rotate:                                  # OPQ-lite: PCA basis before PQ
        _, _, vt = np.linalg.svd(mean_lm - mean_lm.mean(0), full_matrices=False)
        rot = vt.T
    else:
        rot = np.eye(args.dim)
    pq_input = mean_lm @ rot

    print(f"pages {args.pages} ({args.pages * PAGE_TOKENS:,} tokens), dim {args.dim}, "
          f"top-{args.topk}, {'HARD isotropic' if args.hard else 'clustered'} keys, "
          f"{'PCA-rotated' if args.rotate else 'unrotated'} PQ\n")
    print(f"{'index':>7} {'bytes/head/page':>15} {'recall@k':>9} {'recall@2k':>10}   (vs EXACT page ranking)")
    for label, sc, nbytes in (("mean", s_mean, args.dim * 2), ("minmax", s_minmax, args.dim * 4)):
        rank = np.argsort(-sc, axis=1)
        r1 = float(np.mean([recall_at_k(exact_rank[i], rank[i], args.topk) for i in range(args.queries)]))
        r2 = float(np.mean([recall_at_k(exact_rank[i], rank[i], 2 * args.topk) for i in range(args.queries)]))
        print(f"{label:>7} {nbytes:>15} {r1:>9.3f} {r2:>10.3f}   f16 landmark")
    for label, m in (("pq32", 32), ("pq16", 16), ("pq8", 8)):
        if args.dim % m:
            continue
        codebooks, codes = pq_train(pq_input, m, rng)
        r1 = r2 = 0.0
        for i in range(args.queries):
            s = pq_scores(queries[i] @ rot, codebooks, codes)
            ar = np.argsort(-s)
            r1 += recall_at_k(exact_rank[i], ar, args.topk)
            r2 += recall_at_k(exact_rank[i], ar, 2 * args.topk)
        r1, r2 = r1 / args.queries, r2 / args.queries
        print(f"{label:>7} {m:>15} {r1:>9.3f} {r2:>10.3f}   PQ of mean landmark")

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

    print("\nnotes: recall is vs the EXACT page ranking, so the mean/minmax rows show the")
    print("landmark abstraction's own loss - PQ costs almost nothing on top of it, and")
    print("over-fetch (recall@2k) recovers most of the rest at 2x fetch cost. On THIS")
    print("synthetic distribution the Quest-style minmax bound under-ranks (upper bounds")
    print("favor high-spread pages); Quest measures the opposite on real unnormalized")
    print("keys, so trace real K tensors before choosing a landmark kind. --rotate")
    print("(OPQ-lite) tightens the smallest codes. Planner: svmi-fleet.py --landmarks pq16.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
