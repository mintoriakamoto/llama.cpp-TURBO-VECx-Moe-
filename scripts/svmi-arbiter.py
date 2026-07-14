#!/usr/bin/env python3
"""
ARBITER — Asymmetric-fabric Routing: Bandwidth-arbitrage Inference with
Tiered Elastic Residency (docs/svmi-research.md §9)

The inversion: a PC has three memory fabrics with silicon attached —
    VRAM   ~360-1000 GB/s   (GPU)
    DDR4/5 ~ 35-280  GB/s   (CPU, AVX/AMX)
    PCIe   ~ 12-48   GB/s   (the pipe between them)
and PCIe, the fabric every offloading engine funnels weights through, is the
*slowest*. For bandwidth-bound decode, moving a weight over PCIe to compute it
on the GPU is strictly worse than computing it where it already lives.

ARBITER therefore never moves weights on the critical path. Per verify pass:

    GPU : runs the resident draft (speculative tokens) + verifies the
          GPU-resident weight share at VRAM bandwidth
    CPU : concurrently verifies the host-resident share at RAM bandwidth,
          reading each weight once for the whole speculative batch
    PCIe: carries only activations (KBs/layer) and *background* residency
          migration (elastic promotion of hot layers into freed VRAM)

Decode becomes  tok/s = E / max(t_gpu_draft + t_gpu_verify, t_cpu_verify)
with E = accepted tokens per pass. The optimizer below picks the GPU-resident
fraction that balances the two sides — the bandwidth-arbitrage split — and
compares against pure streaming, stock partial offload, and pure CPU.

Exactness: verification computes the true model everywhere; routing changes
*where* math runs, not the math. Token-identical by construction.

Analytical model grounded in real shapes (GGUF or --profile); measure on
hardware before believing decimals.

Usage:
  python3 scripts/svmi-arbiter.py --profile 70b --gpu 3060 --cpu ddr5-2ch --cores 16
  python3 scripts/svmi-arbiter.py model.gguf --gpu 2080ti --cpu ddr4-2ch --cores 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

GiB = 1024**3
GB = 1e9

PCIE_BW = {"3.0-x8": 6.0, "3.0-x16": 12.0, "4.0-x8": 12.0, "4.0-x16": 24.0, "5.0-x16": 48.0}
# (vram GiB, pcie link, vram GB/s)
GPU_PRESETS = {
    "1660ti": (6, "3.0-x16", 288), "2060": (6, "3.0-x16", 336), "2070": (8, "3.0-x16", 448),
    "2080": (8, "3.0-x16", 448), "2080ti": (11, "3.0-x16", 616),
    "3060": (12, "4.0-x16", 360), "3070": (8, "4.0-x16", 448), "3080": (10, "4.0-x16", 760),
    "3090": (24, "4.0-x16", 936), "4070": (12, "4.0-x16", 504), "4090": (24, "4.0-x16", 1008),
}
# effective (not theoretical) CPU memory bandwidth for GEMV-like streaming, GB/s
CPU_PRESETS = {
    "ddr4-2ch": 35.0, "ddr4-4ch": 65.0,
    "ddr5-2ch": 70.0, "ddr5-4ch": 130.0, "ddr5-8ch": 280.0,
}
# effective quantized-GEMM throughput per core (AVX2/AVX512-VNNI mix), FLOP/s
FLOPS_PER_CORE = 75e9

# (n_layer, total weight GiB, params B) Q4_K_M-class
MODEL_PROFILES = {
    "7b": (32, 3.9, 7e9), "8b": (32, 4.6, 8e9), "13b": (40, 7.4, 13e9),
    "70b": (80, 39.6, 70e9),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES))
    ap.add_argument("--gpu", choices=sorted(GPU_PRESETS), required=True)
    ap.add_argument("--cpu", choices=sorted(CPU_PRESETS), default="ddr5-2ch")
    ap.add_argument("--cores", type=int, default=16, help="physical cores usable for verify GEMM")
    ap.add_argument("--pcie", choices=sorted(PCIE_BW), help="override GPU preset link")
    ap.add_argument("--vram-budget", type=float, help="usable VRAM GiB (default: preset - reserve)")
    ap.add_argument("--display-reserve", type=float, default=1.0)
    ap.add_argument("--kv-reserve", type=float, default=1.0,
                    help="GiB reserved for KV window / CTX-VM page table (see svmi-fleet.py)")
    ap.add_argument("--overhead", type=float, default=1.0, help="activations/scratch GiB")
    ap.add_argument("--draft-gib", type=float, default=1.0,
                    help="VRAM-resident draft size (small model or BitSpec low-bit share)")
    ap.add_argument("--spec-tokens", type=float, default=5.0,
                    help="E: mean accepted tokens per verify pass (svmi-bitspec.py measures ~7 "
                         "for a 3-bit self-draft; 4-5 is conservative for a small external draft)")
    args = ap.parse_args()

    vram_gib, link, vram_bw = GPU_PRESETS[args.gpu]
    pcie_bw = PCIE_BW[args.pcie or link]
    ram_bw = CPU_PRESETS[args.cpu]
    cpu_flops = args.cores * FLOPS_PER_CORE
    budget = (args.vram_budget if args.vram_budget is not None
              else max(0.0, vram_gib - args.display_reserve))

    if args.profile:
        n_layer, w_gib, params = MODEL_PROFILES[args.profile]
        model_name = f"{args.profile} (synthetic Q4_K_M-class)"
    elif args.model:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
        from gguf import GGUFReader
        reader = GGUFReader(args.model)
        f = reader.get_field("general.architecture")
        arch = bytes(f.parts[f.data[0]]).decode("utf-8") if f is not None else ""
        fld = reader.get_field(f"{arch}.block_count")
        n_layer = int(fld.parts[fld.data[0]][0]) if fld is not None else 0
        w_gib = sum(int(t.n_bytes) for t in reader.tensors) / GiB
        params = sum(int(t.n_elements) for t in reader.tensors)
        model_name = Path(args.model).name
    else:
        ap.error("provide a GGUF path or --profile")

    W = w_gib * GiB / GB               # weights, GB
    draft = args.draft_gib * GiB / GB  # GB
    E = args.spec_tokens
    resident_room = max(0.0, (budget - args.kv_reserve - args.overhead - args.draft_gib)) * GiB / GB

    def pass_time(gpu_share_gb: float) -> tuple[float, float, float]:
        """returns (t_gpu, t_cpu, tok/s) for one verify pass of E tokens"""
        cpu_share = W - gpu_share_gb
        t_gpu = E * draft / vram_bw + gpu_share_gb / vram_bw
        t_cpu_mem = cpu_share / ram_bw
        t_cpu_fl = 2.0 * (cpu_share / W * params) * E / cpu_flops
        t_cpu = max(t_cpu_mem, t_cpu_fl)
        # activations cross PCIe twice per contiguous split: negligible bytes, ~2 sync hops
        t = max(t_gpu, t_cpu) + 2 * 20e-6
        return t_gpu, t_cpu, E / t

    # --- modes ---
    # 1. ARBITER: optimal GPU-resident share within VRAM room
    best = None
    steps = 200
    for i in range(steps + 1):
        share = min(W, resident_room) * i / steps
        r = pass_time(share)
        if best is None or r[2] > best[2]:
            best = (*r, share)
    assert best is not None  # loop always runs at least once
    t_gpu_b, t_cpu_b, arb_tps, arb_share = best

    # 2. SVMI pure streaming + spec: draft on GPU, verify streams host share over PCIe
    gpu_res = min(W, resident_room)
    streamed = W - gpu_res
    t_stream = E * draft / vram_bw + gpu_res / vram_bw + streamed / pcie_bw
    stream_tps = E / t_stream

    # 3. stock partial offload (llama.cpp default: CPU computes host layers, no speculation)
    stock_tps = 1.0 / (streamed / ram_bw + gpu_res / vram_bw) if W > 0 else 0.0

    # 4. pure CPU + spec (no GPU at all; draft also reads from RAM)
    t_cpu_only = max((W + E * draft) / ram_bw, 2.0 * params * E / cpu_flops)
    cpu_tps = E / t_cpu_only

    # PCIe left free for background residency migration
    mig_bw = pcie_bw * 0.9  # activations use ~nothing

    print(f"model   : {model_name}  ({w_gib:.1f} GiB weights, {params/1e9:.0f}B params, {n_layer} layers)")
    print(f"gpu     : {args.gpu}  (VRAM {vram_bw:.0f} GB/s, budget {budget:.1f} GiB, "
          f"PCIe {pcie_bw:.0f} GB/s)")
    print(f"cpu     : {args.cpu} ({ram_bw:.0f} GB/s eff) x {args.cores} cores "
          f"({cpu_flops/1e12:.1f} TFLOP/s eff)")
    print(f"spec    : draft {args.draft_gib:.1f} GiB resident, E = {E:.0f} accepted tok/pass\n")

    print(f"fabric arbitrage (why ARBITER wins): moving 1 GB of weights costs "
          f"{1/pcie_bw*1e3:.0f} ms over PCIe,\n  {1/ram_bw*1e3:.0f} ms computed in-place on CPU, "
          f"{1/vram_bw*1e3:.1f} ms if already in VRAM.\n")

    rows = [
        ("ARBITER (cross-fabric verify)", arb_tps,
         f"GPU {arb_share:.1f} GB + CPU {W-arb_share:.1f} GB; "
         f"gpu {t_gpu_b*1e3:.0f} ms || cpu {t_cpu_b*1e3:.0f} ms/pass"),
        ("SVMI streaming + spec", stream_tps,
         f"streams {streamed:.1f} GB/pass over PCIe"),
        ("stock partial offload", stock_tps,
         "CPU share unbatched (no speculation)"),
        ("pure CPU + spec", cpu_tps,
         "no GPU; RAM-bandwidth bound"),
    ]
    print(f"{'mode':<32} {'tok/s':>7}  notes")
    for name, tps, note in rows:
        print(f"{name:<32} {tps:7.1f}  {note}")

    if W <= resident_room:
        print("\nnote: this model fits entirely in VRAM — plain resident decode wins; "
              "ARBITER is for models that don't fit.")
    print(f"\nARBITER speedup: {arb_tps/stock_tps:.1f}x vs stock partial offload, "
          f"{arb_tps/stream_tps:.1f}x vs streaming-only")
    print(f"PCIe on the critical path: 0 bytes of weights (activations only); "
          f"{mig_bw:.0f} GB/s free for\n  background residency migration -> promoting "
          f"{max(0.0, min(W, resident_room)):.1f} GB takes "
          f"{max(0.0, min(W, resident_room))/mig_bw:.1f} s, amortized across the session.")
    bind = "CPU flops" if (2.0 * ((W - arb_share) / W * params) * E / cpu_flops) > (W - arb_share) / ram_bw \
        else "CPU RAM bandwidth"
    print(f"binding resource at optimum: {bind} -> next win: more cores/AMX "
          f"or faster RAM, not a faster GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
