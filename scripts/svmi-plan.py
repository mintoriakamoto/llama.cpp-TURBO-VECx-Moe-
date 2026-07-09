#!/usr/bin/env python3
"""
SVMI residency planner

Given a GGUF model and a VRAM budget, decide which tensors should stay resident on the
GPU and which should be kept in host RAM and streamed on demand (--stream-weights).
Emits ready-to-use llama.cpp flags (-ngl / -ot overrides) implementing the plan.

Placement policy (latency mode, all matrix math on the GPU):
  1. token embeddings / output head / norms are always resident (touched every token,
     small, and on the critical path of speculative drafting)
  2. attention tensors of every layer are resident next: they are needed every token,
     feed the KV cache, and are small compared to the FFN
  3. FFN (or MoE expert) tensors are resident for as many layers as fit, allocated from
     both ends of the stack inward (first/last layers have the highest measured impact
     on output quality and are used by self-drafting); the remainder is streamed
  4. a streaming ring-buffer reserve and the KV cache are subtracted from the budget
     before placing weights

Usage:
  python3 scripts/svmi-plan.py model.gguf --vram-budget 20 [--ctx 32768] [--kv-type q8_0]

The printed command line can be passed directly to llama-cli / llama-server / llama-bench.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# use the in-tree gguf-py reader
sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

from gguf import GGUFReader  # noqa: E402

GiB = 1024**3
MiB = 1024**2

# effective *pinned* H2D bandwidth (GB/s), well below the theoretical link rate
PCIE_BW = {
    "3.0-x8":  6.0,
    "3.0-x16": 12.0,
    "4.0-x8":  12.0,
    "4.0-x16": 24.0,
    "5.0-x16": 48.0,
}

# consumer-GPU presets: (physical VRAM GiB, PCIe link, copy engines, note).
# The usable weight budget defaults to physical minus a display/driver reserve.
GPU_PRESETS = {
    "1660ti": (6,  "3.0-x16", 1, "Turing TU116, PCIe 3.0, no tensor cores"),
    "2060":   (6,  "3.0-x16", 1, "Turing TU106, PCIe 3.0"),
    "2070":   (8,  "3.0-x16", 1, "Turing TU106, PCIe 3.0"),
    "2080":   (8,  "3.0-x16", 1, "Turing TU104, PCIe 3.0"),
    "2080ti": (11, "3.0-x16", 1, "Turing TU102, PCIe 3.0"),
    "3060":   (12, "4.0-x16", 2, "Ampere GA106, PCIe 4.0"),
    "3060ti": (8,  "4.0-x16", 2, "Ampere GA104, PCIe 4.0"),
    "3070":   (8,  "4.0-x16", 2, "Ampere GA104, PCIe 4.0"),
    "3080":   (10, "4.0-x16", 2, "Ampere GA102, PCIe 4.0"),
    "3090":   (24, "4.0-x16", 2, "Ampere GA102, PCIe 4.0"),
    "4060":   (8,  "4.0-x8",  2, "Ada AD107, PCIe 4.0 x8 (narrow link!)"),
    "4070":   (12, "4.0-x16", 2, "Ada AD104, PCIe 4.0"),
    "4090":   (24, "4.0-x16", 2, "Ada AD102, PCIe 4.0"),
}


def tensor_layer(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def is_ffn(name: str) -> bool:
    # dense FFN and MoE expert tensors: the large, streamable part of a layer
    return ".ffn_" in name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="path to the GGUF model")
    ap.add_argument("--gpu", choices=sorted(GPU_PRESETS), help="consumer GPU preset: "
                    "sets VRAM budget, PCIe bandwidth, and upload-queue count "
                    "(e.g. 1660ti, 2080, 2080ti, 3060)")
    ap.add_argument("--vram-budget", type=float, help="usable VRAM budget in GiB "
                    "(overrides the --gpu preset; required if --gpu is not given)")
    ap.add_argument("--pcie", choices=sorted(PCIE_BW), help="PCIe link for the decode-floor "
                    "estimate (overrides the --gpu preset; default 4.0-x16)")
    ap.add_argument("--display-reserve", type=float, default=1.0,
                    help="VRAM held back for the OS/display when using --gpu (GiB, default 1.0)")
    ap.add_argument("--ctx", type=int, default=32768, help="planned context length (KV sizing)")
    ap.add_argument("--kv-type", choices=["f16", "q8_0", "q5_1", "q4_0"], default="q8_0",
                    help="KV cache quantization (default q8_0)")
    ap.add_argument("--stream-slots", type=int, default=8, help="streaming ring-buffer slots")
    ap.add_argument("--overhead", type=float, default=1.25,
                    help="activations/compute-buffer reserve in GiB (default 1.25)")
    args = ap.parse_args()

    # resolve GPU preset -> budget / pcie / queues
    n_queues = 2
    gpu_note = ""
    if args.gpu:
        vram_gib, link, n_queues, gpu_note = GPU_PRESETS[args.gpu]
        if args.vram_budget is None:
            args.vram_budget = max(0.0, vram_gib - args.display_reserve)
        if args.pcie is None:
            args.pcie = link
    if args.vram_budget is None:
        ap.error("provide --vram-budget or --gpu")
    if args.pcie is None:
        args.pcie = "4.0-x16"
    pcie_bw = PCIE_BW[args.pcie]

    reader = GGUFReader(args.model)

    def field_int(key: str, default: int = 0) -> int:
        f = reader.get_field(key)
        return int(f.parts[f.data[0]][0]) if f is not None else default

    arch = ""
    f = reader.get_field("general.architecture")
    if f is not None:
        arch = bytes(f.parts[f.data[0]]).decode("utf-8")

    n_layer     = field_int(f"{arch}.block_count")
    n_embd      = field_int(f"{arch}.embedding_length")
    n_head_kv   = field_int(f"{arch}.attention.head_count_kv") or field_int(f"{arch}.attention.head_count")
    n_head      = field_int(f"{arch}.attention.head_count") or 1
    head_dim    = n_embd // n_head if n_head else 0

    # per-tensor sizes
    layer_attn  : dict[int, int] = defaultdict(int)   # attention + norms per layer
    layer_ffn   : dict[int, int] = defaultdict(int)   # ffn / experts per layer
    global_size = 0                                    # embeddings, output head, final norm
    total_size  = 0

    for t in reader.tensors:
        size = int(t.n_bytes)
        total_size += size
        layer = tensor_layer(t.name)
        if layer is None:
            global_size += size
        elif is_ffn(t.name):
            layer_ffn[layer] += size
        else:
            layer_attn[layer] += size

    if n_layer == 0:
        n_layer = max(list(layer_attn.keys()) + list(layer_ffn.keys()), default=-1) + 1

    # KV cache estimate: 2 (K and V) * head_dim * n_head_kv * ctx * bytes-per-el * n_layer
    kv_bpe = {"f16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q4_0": 0.5625}[args.kv_type]
    kv_size = int(2 * head_dim * n_head_kv * args.ctx * kv_bpe * n_layer)

    # streaming ring reserve: slots * largest streamable tensor
    max_stream_tensor = max((int(t.n_bytes) for t in reader.tensors if is_ffn(t.name)), default=0)
    ring_size = args.stream_slots * max_stream_tensor

    budget = int(args.vram_budget * GiB)
    reserve = kv_size + ring_size + int(args.overhead * GiB)
    weight_budget = budget - reserve

    if weight_budget <= 0:
        print(f"error: budget {args.vram_budget:.1f} GiB is smaller than the fixed reserve "
              f"({reserve / GiB:.2f} GiB: KV {kv_size / GiB:.2f} + ring {ring_size / GiB:.2f} "
              f"+ overhead {args.overhead:.2f}); reduce --ctx or --stream-slots", file=sys.stderr)
        return 1

    # placement
    resident = global_size
    attn_total = sum(layer_attn.values())
    if resident + attn_total > weight_budget:
        print(f"error: embeddings+attention alone ({(global_size + attn_total) / GiB:.2f} GiB) "
              f"exceed the weight budget ({weight_budget / GiB:.2f} GiB); this model needs a "
              f"larger budget even with full FFN streaming", file=sys.stderr)
        return 1
    resident += attn_total

    # FFN layers from both ends inward
    order: list[int] = []
    lo, hi = 0, n_layer - 1
    while lo <= hi:
        order.append(lo)
        if hi != lo:
            order.append(hi)
        lo += 1
        hi -= 1

    ffn_resident: set[int] = set()
    for layer in order:
        size = layer_ffn.get(layer, 0)
        if resident + size <= weight_budget:
            resident += size
            ffn_resident.add(layer)

    streamed_layers = sorted(set(range(n_layer)) - ffn_resident)
    streamed_bytes = sum(layer_ffn.get(l, 0) for l in streamed_layers)

    print(f"model              : {Path(args.model).name} ({arch}, {n_layer} layers, "
          f"{total_size / GiB:.2f} GiB weights)")
    if args.gpu:
        print(f"gpu                : {args.gpu} ({gpu_note})")
    print(f"vram budget        : {budget / GiB:.2f} GiB")
    print(f"  kv cache ({args.kv_type:>5}) : {kv_size / GiB:6.2f} GiB @ ctx {args.ctx}")
    print(f"  stream ring      : {ring_size / GiB:6.2f} GiB ({args.stream_slots} slots x "
          f"{max_stream_tensor / MiB:.0f} MiB)")
    print(f"  overhead reserve : {args.overhead:6.2f} GiB")
    print(f"  resident weights : {resident / GiB:6.2f} GiB")
    print(f"streamed per token : {streamed_bytes / GiB:.2f} GiB "
          f"({len(streamed_layers)}/{n_layer} FFN layers streamed)")
    floor = pcie_bw * 1e9 / streamed_bytes if streamed_bytes else float("inf")
    print(f"decode floor       : {floor:.2f} tok/s @ {pcie_bw:.0f} GB/s ({args.pcie} pinned)")
    print(f"  with BitSpec (3-bit self-draft, ~7x): ~{floor * 7:.1f} tok/s "
          f"(see scripts/svmi-bitspec.py)")
    print()

    if not streamed_layers:
        print("# everything fits resident; streaming not needed:")
        print(f"-ngl 99 -c {args.ctx} -ctk {args.kv_type} -ctv {args.kv_type} -fa on")
        return 0

    # compact regex alternation for the streamed layer ids
    ids = "|".join(str(l) for l in streamed_layers)
    ot = f"blk\\.({ids})\\.ffn_.*=CPU"

    print("# flags implementing this plan:")
    print(f"-ngl 99 -ot '{ot}' \\")
    print(f"  --stream-weights {args.stream_slots} --stream-decode \\")
    print(f"  -c {args.ctx} -ctk {args.kv_type} -ctv {args.kv_type} -fa on")
    if n_queues == 1:
        print(f"# {args.gpu or 'this GPU'} exposes a single H2D copy engine: use one upload queue")
        print("export GGML_SCHED_STREAM_QUEUES=1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
