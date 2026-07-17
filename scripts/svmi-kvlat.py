#!/usr/bin/env python3
"""
KV-LAT validator — how far can a latent (MLA-style) retrofit compress GQA KV?

Per layer, stack the K and V projection weights [W_K; W_V] and compute the singular
value spectrum. The rank needed to retain a given fraction of spectral energy bounds
the latent width r a lossless-ish retrofit could use; bytes/token follow directly:

  baseline (GQA):  2 * d_head * n_head_kv * bpe          per layer
  KV-LAT:          (r + d_rope) * bpe                    per layer
                   (one shared latent + a decoupled RoPE key)

Weight-space energy is an OPTIMISTIC proxy: production retrofits whiten by the input
activation covariance (TransMLA/X-MLA-style) and check perplexity. Treat these ranks
as the ceiling of what calibration can reach, not a promise. See
docs/svmi-research.md §10 for deployment rules (cold-tier-only, BitSpec verification).

Usage:
  python3 scripts/svmi-kvlat.py model.gguf                # real tensors (dequantized)
  python3 scripts/svmi-kvlat.py --profile 70b             # synthetic ensemble
  python3 scripts/svmi-kvlat.py --profile 8b --rope-dims 64 --kv-bpe 1.0625
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# (n_layer, n_embd, n_head, n_head_kv) — matches svmi-fleet.py profiles
MODEL_PROFILES = {
    "7b":  (32, 4096, 32, 32),
    "8b":  (32, 4096, 32, 8),
    "13b": (40, 5120, 40, 40),
    "70b": (80, 8192, 64, 8),
}

ENERGY_TARGETS = (0.95, 0.99, 0.999)


def synthetic_kv_stack(n_embd: int, kv_dim: int, rng: np.random.Generator) -> np.ndarray:
    """[W_K; W_V]-shaped matrix with the decaying spectrum trained projections show.

    Trained K/V projections are far from isotropic: their singular values decay
    as a power law; public MLA-retrofit measurements land 99%-energy ranks near
    kv_dim/2, which a ~i^-0.8 decay reproduces. --profile numbers are indicative.
    """
    rows, cols = 2 * kv_dim, n_embd
    k = min(rows, cols)
    u, _ = np.linalg.qr(rng.standard_normal((rows, k)))
    v, _ = np.linalg.qr(rng.standard_normal((cols, k)))
    s = (np.arange(1, k + 1, dtype=np.float64)) ** -0.8
    return (u * s) @ v.T


def layer_ranks(m: np.ndarray) -> dict[float, int]:
    s = np.linalg.svd(m.astype(np.float64), compute_uv=False)
    energy = np.cumsum(s * s) / np.sum(s * s)
    return {t: int(np.searchsorted(energy, t) + 1) for t in ENERGY_TARGETS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES), help="synthetic model shape")
    ap.add_argument("--layers", type=int, default=0, help="limit analyzed layers (0 = all)")
    ap.add_argument("--rope-dims", type=int, default=64, help="decoupled RoPE key dims kept exact")
    ap.add_argument("--kv-bpe", type=float, default=1.0625, help="bytes/element of the latent cache (q8_0 default)")
    ap.add_argument("--seed", type=int, default=7, help="rng seed for --profile mode")
    args = ap.parse_args()

    layers: list[np.ndarray] = []
    if args.model:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
        from gguf import GGUFReader
        from gguf.quants import dequantize

        reader = GGUFReader(args.model)
        arch_f = reader.get_field("general.architecture")
        arch = bytes(arch_f.parts[arch_f.data[0]]).decode("utf-8") if arch_f else "?"

        def fi(key: str, default: int = 0) -> int:
            f = reader.get_field(key)
            return int(f.parts[f.data[0]][0]) if f is not None else default

        n_layer = fi(f"{arch}.block_count")
        n_embd = fi(f"{arch}.embedding_length")
        n_head = fi(f"{arch}.attention.head_count") or 1
        n_head_kv = fi(f"{arch}.attention.head_count_kv") or n_head
        name = Path(args.model).name

        tensors = {t.name: t for t in reader.tensors}
        n_take = args.layers or n_layer
        for i in range(min(n_take, n_layer)):
            try:
                wk = tensors[f"blk.{i}.attn_k.weight"]
                wv = tensors[f"blk.{i}.attn_v.weight"]
            except KeyError:
                print(f"layer {i}: fused/missing attn_k|attn_v tensors, skipping", file=sys.stderr)
                continue
            k = dequantize(wk.data, wk.tensor_type).reshape(wk.shape[1], wk.shape[0])
            v = dequantize(wv.data, wv.tensor_type).reshape(wv.shape[1], wv.shape[0])
            layers.append(np.concatenate([k, v], axis=0))
    elif args.profile:
        n_layer, n_embd, n_head, n_head_kv = MODEL_PROFILES[args.profile]
        name = f"{args.profile} (synthetic spectrum, seed {args.seed})"
        rng = np.random.default_rng(args.seed)
        kv_dim = (n_embd // n_head) * n_head_kv
        # spectra vary little across layers; analyze a sample and reuse
        n_take = min(args.layers or 4, n_layer)
        for _ in range(n_take):
            layers.append(synthetic_kv_stack(n_embd, kv_dim, rng))
    else:
        ap.error("provide a GGUF path or --profile")

    if not layers:
        print("no analyzable layers found", file=sys.stderr)
        return 1

    head_dim = n_embd // n_head
    base_bytes = 2 * head_dim * n_head_kv * args.kv_bpe

    print(f"model     : {name}  ({n_layer} layers, d_head {head_dim}, GQA {n_head}/{n_head_kv})")
    print(f"baseline  : {base_bytes:.0f} KV bytes/token/layer at {args.kv_bpe} bpe "
          f"({base_bytes * n_layer / 1024:.1f} KiB/token all layers)")
    print(f"analyzed  : {len(layers)} layer(s); latent = shared rank-r + {args.rope_dims} RoPE dims\n")

    print(f"{'energy':>7} {'rank r (med)':>12} {'lat bytes/tok/layer':>20} {'vs GQA':>7} {'cold@131K q4/agent':>18}")
    ranks_by_target: dict[float, list[int]] = {t: [] for t in ENERGY_TARGETS}
    for m in layers:
        for t, r in layer_ranks(m).items():
            ranks_by_target[t].append(r)
    for t in ENERGY_TARGETS:
        r = int(np.median(ranks_by_target[t]))
        lat_bytes = (r + args.rope_dims) * args.kv_bpe
        ratio = base_bytes / lat_bytes
        # cold-tier size for THIS model at 131,072 ctx, q4 cold (bpe 0.5625)
        cold = (r + args.rope_dims) * 0.5625 * n_layer * 131072 / 1024**3
        print(f"{t:>7.3f} {r:>12} {lat_bytes:>20.0f} {ratio:>6.1f}x {cold:>16.2f} GiB")

    print("\nnotes: weight-space SVD is the optimistic bound — calibrate with activation")
    print("covariance before trusting a rank; keep the hot window exact and verify with")
    print("BitSpec when deploying latents beyond the cold tier (svmi-research.md §10).")
    print("Feed the chosen rank to the fleet planner:  svmi-fleet.py --kv-lat R")
    return 0


if __name__ == "__main__":
    sys.exit(main())
