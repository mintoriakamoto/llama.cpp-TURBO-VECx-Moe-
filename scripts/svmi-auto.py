#!/usr/bin/env python3
"""
svmi-auto — one command: what should THIS machine do with THIS model?

Reads the model (GGUF or --profile) and the GPU, decides the regime, and prints
ready-to-run llama.cpp flags plus the planner to consult for fine-tuning:

  resident   everything fits: plain flags, nothing clever needed
  offload    close: keep attention resident, stream the FFN tail (svmi-plan.py split)
  streamed   far: full SVMI weight streaming + BitSpec advice (svmi-plan / svmi-arbiter)
  fleet      many agents or long context: MAVM/CTX-VM planning (svmi-fleet.py)

Only flags that exist in this branch today are emitted (--stream-weights,
--stream-decode, -ngl, -ctk/-ctv); CTX-VM paging and expert lookahead are still
planner-level designs and are pointed at, not faked.

Usage:
  python3 scripts/svmi-auto.py model.gguf --gpu 3060
  python3 scripts/svmi-auto.py --profile 70b --gpu 3060 --ctx 32768
  python3 scripts/svmi-auto.py --profile 8b --gpu 1660ti --agents 8 --ctx 131072
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GiB = 1024**3

# (n_layer, n_embd, n_head, n_head_kv, weights GiB Q4_K_M-class) — svmi-fleet.py shapes
MODEL_PROFILES = {
    "7b":  (32, 4096, 32, 32,  3.9),
    "8b":  (32, 4096, 32,  8,  4.6),
    "13b": (40, 5120, 40, 40,  7.4),
    "70b": (80, 8192, 64,  8, 39.6),
}
GPU_PRESETS = {  # vram GiB, effective PCIe GB/s
    "1660ti": (6, 12.0), "2060": (6, 12.0), "2070": (8, 12.0), "2080": (8, 12.0),
    "2080ti": (11, 12.0), "3060": (12, 24.0), "3070": (8, 24.0), "3080": (10, 24.0),
    "3090": (24, 24.0), "4070": (12, 24.0), "4090": (24, 24.0),
}
Q8_BPE = 1.0625


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    ap.add_argument("--gpu", choices=sorted(GPU_PRESETS), required=True)
    ap.add_argument("--gpus", type=int, default=1, help="number of identical GPUs (layer split)")
    ap.add_argument("--ctx", type=int, default=8192, help="context length you actually need")
    ap.add_argument("--agents", type=int, default=1, help="concurrent agents/slots planned")
    ap.add_argument("--display-reserve", type=float, default=1.0)
    ap.add_argument("--overhead", type=float, default=1.25, help="activation reserve GiB")
    ap.add_argument("--host-ram", type=float, default=64.0, help="host RAM GiB (for the -nkvo ceiling)")
    args = ap.parse_args()

    if args.model:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
        from gguf import GGUFReader

        reader = GGUFReader(args.model)
        f = reader.get_field("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8") if f else "?"

        def fi(key: str, default: int = 0) -> int:
            fld = reader.get_field(key)
            return int(fld.parts[fld.data[0]][0]) if fld is not None else default

        n_layer = fi(f"{arch}.block_count")
        n_embd = fi(f"{arch}.embedding_length")
        n_head = fi(f"{arch}.attention.head_count") or 1
        n_head_kv = fi(f"{arch}.attention.head_count_kv") or n_head
        weights = sum(int(t.n_bytes) for t in reader.tensors)
        name = Path(args.model).name
        model_arg = args.model
    elif args.profile:
        n_layer, n_embd, n_head, n_head_kv, w_gib = MODEL_PROFILES[args.profile]
        weights = int(w_gib * GiB)
        name = f"{args.profile} (Q4_K_M-class profile)"
        model_arg = "model.gguf"
    else:
        ap.error("provide a GGUF path or --profile")

    vram, pcie = GPU_PRESETS[args.gpu]
    n_gpu = max(1, args.gpus)
    # desktop boards run two x16 slots at x8/x8: per-card link bandwidth halves,
    # but uploads run in parallel, so AGGREGATE stays ~one full link
    pcie_agg = pcie if n_gpu > 1 else pcie
    budget = (n_gpu * vram - args.display_reserve) * GiB
    head_dim = n_embd // n_head
    kv_tok = 2 * head_dim * n_head_kv * n_layer * Q8_BPE
    kv = kv_tok * args.ctx * args.agents
    fixed = args.overhead * GiB
    need = weights + kv + fixed
    fleet = args.agents > 1 or args.ctx > 32768

    print(f"model   : {name}  ({weights / GiB:.1f} GiB weights, {n_layer} layers)")
    gpu_lbl = f"{n_gpu}x {args.gpu}" if n_gpu > 1 else args.gpu
    print(f"gpu     : {gpu_lbl}  ({n_gpu * vram} GiB VRAM total, ~{pcie_agg:.0f} GB/s aggregate PCIe"
          + (", assumes x8/x8 slots" if n_gpu > 1 else "") + ")")
    print(f"need    : {weights / GiB:.1f} weights + {kv / GiB:.1f} KV (q8_0 @ {args.ctx:,} ctx "
          f"x {args.agents}) + {fixed / GiB:.1f} reserve = {need / GiB:.1f} GiB "
          f"vs {budget / GiB:.1f} budget\n")

    kvflags = "-fa on -ctk q8_0 -ctv q8_0"   # quantized V cache requires flash attention
    if n_gpu > 1:
        split = ",".join(["1"] * n_gpu)
        kvflags += f" --split-mode layer --tensor-split {split}"
    if need <= budget:
        print("regime  : RESIDENT — everything fits, no streaming needed")
        print(f"\n  llama-cli -m {model_arg} -ngl 999 {kvflags} -c {args.ctx}")
    elif weights * 0.55 + kv + fixed <= budget:
        print("regime  : OFFLOAD — attention + head fit; stream the FFN tail on demand")
        print(f"\n  llama-cli -m {model_arg} -ngl 999 --stream-weights 8 --stream-decode \\")
        print(f"            {kvflags} -c {args.ctx}")
        print("\n  exact resident/streamed split (-ot overrides), decode floor:")
        print(f"    python3 scripts/svmi-plan.py {model_arg} --gpu {args.gpu} --ctx {args.ctx}")
    else:
        streamed = max(weights * 0.05, need - budget)   # bytes/pass that do not fit resident
        floor = pcie_agg * 1e9 / streamed
        print("regime  : STREAMED — the model mostly lives in pinned host RAM (SVMI)")
        print(f"          PCIe decode floor ~{floor:.1f} tok/s, ~{floor * 7:.0f} with BitSpec-class speculation")
        print(f"\n  llama-cli -m {model_arg} -ngl 999 --stream-weights 8 --stream-decode \\")
        print(f"            {kvflags} -c {args.ctx}")
        print(f"\n  split + floor detail : python3 scripts/svmi-plan.py {model_arg} --gpu {args.gpu}")
        prof = args.profile or "70b"
        print(f"  CPU-assisted decode  : python3 scripts/svmi-arbiter.py --profile {prof} "
              f"--gpu {args.gpu} --cpu ddr5-2ch --cores 16")
    if fleet:
        print(f"\nfleet   : {args.agents} agent(s) @ {args.ctx:,} ctx — full-VRAM KV stops scaling here;")
        print("          plan capacity with CTX-VM paging (planner-level today, engine phase 5):")
        print(f"    python3 scripts/svmi-fleet.py {'--profile ' + args.profile if args.profile else model_arg} "
              f"--gpu {args.gpu} --agents {args.agents} --ctx {args.ctx} \\")
        print("        --cold-kv-type q4_0 --landmarks pq16 --prefetch-hit 0.7")
    # --- max context on this rig, per engine-real KV mode ---
    # weights the KV must coexist with: full model when resident, else the
    # non-streamable floor (embeddings/norms/head, ~5%) since SVMI pages the rest
    w_gpu = weights if weights + fixed <= budget else weights * 0.05
    kv_room_vram = max(0.0, budget - w_gpu - fixed)
    host_room = max(0.0, args.host_ram * GiB * 0.85 - (weights if w_gpu < weights else 0))
    kv_tok1 = kv_tok / Q8_BPE            # per-token bytes at 1.0 bpe, single agent basis
    per = args.agents
    print("\nmax ctx : engine-real modes on this rig"
          + (f" ({args.agents} agents)" if per > 1 else "") + ":")
    for label, bpe, room, flags in (
        ("f16 KV, VRAM",  2.0,    kv_room_vram, "-fa on"),
        ("q8_0 KV, VRAM", 1.0625, kv_room_vram, "-fa on -ctk q8_0 -ctv q8_0"),
        ("q4_0 KV, VRAM", 0.5625, kv_room_vram, "-fa on -ctk q4_0 -ctv q4_0"),
        ("q4_0 KV, host RAM", 0.5625, host_room, "-fa on -ctk q4_0 -ctv q4_0 -nkvo (slower/step)"),
    ):
        cmax = int(room / (kv_tok1 * bpe * per))
        print(f"  {label:<18} ~{cmax:>9,} tok   {flags}")
    print("  beyond training ctx: --rope-scaling yarn --yarn-orig-ctx <train-ctx>; long")
    print("  sessions reuse the window via --cache-reuse / context shift; SWA models cap")
    print("  KV at their window (--swa-full disables). CTX-VM paging past all of these")
    print("  is planner-level today (svmi-fleet.py).")

    if n_gpu > 1:
        print("\nmulti-gpu: layer split shown (safest); 2080 Ti-class pairs with an NVLink")
        print("bridge can try --split-mode row for tensor parallelism on the resident share.")
    print("\nverify any streamed setup with scripts/svmi-verify.sh (token identity) before trusting it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
