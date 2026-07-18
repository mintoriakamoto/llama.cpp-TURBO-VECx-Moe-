#!/usr/bin/env python3
"""
SVMI fleet planner — many agents, long context, one GPU (MAVM + CTX-VM model)

Answers the multi-agent question at real context lengths (131072 / 262144 tokens):
how many concurrent agents share one streamed model, what does each additional agent
cost in VRAM *and* host RAM, and how fast does the fleet run?

It quantifies two designs from docs/svmi-research.md:

  MAVM  (§7)  weights shared once (O(1) in agents), shared-prefix KV dedup,
              stream-once-serve-many batching, idle-KV spill.
  CTX-VM (§8) paged context virtual memory: full KV lives in pinned host RAM,
              a landmark page table + a hot window stay in VRAM, and missed
              pages are demand-fetched over the same DMA path as weights.
              This is what makes 131K–256K context per agent fit a consumer GPU:
              per-agent VRAM drops from O(ctx) to O(window + ctx/page_size).

Goal → mechanism map:
  bigger models     -> weights streamed; only a hot set resident
  more agents       -> new agent costs its hot KV window + page table, not 20+ GiB
  131K/256K context -> CTX-VM paging (full KV would be 21-80 GiB/agent for a 70B)
  cheaper VRAM      -> shared-prefix pages stored once fleet-wide
  cheaper tokens    -> shared corpus prefilled once, not per agent
  faster / no idle  -> batched streaming clock + BitSpec; idle agents' hot window
                       spills to host so the GPU never idles on their behalf

Numbers are analytical estimates grounded in the model's actual shape (read from the
GGUF, or a synthetic profile); measure on hardware before believing decimals.

Usage:
  python3 scripts/svmi-fleet.py --profile 70b --gpu 2080ti --agents 1,2,4,8 \
      --ctx 131072 --shared-prompt 98304 --host-ram 128
  python3 scripts/svmi-fleet.py model.gguf --gpu 3060 --agents 8,16,32 --ctx 262144
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

GiB = 1024**3

PCIE_BW = {"3.0-x8": 6.0, "3.0-x16": 12.0, "4.0-x8": 12.0, "4.0-x16": 24.0, "5.0-x16": 48.0}
# (vram GiB, pcie link, copy engines, effective dense fp16 TFLOPS)
GPU_PRESETS = {
    "1660ti": (6, "3.0-x16", 1, 5.0), "2060": (6, "3.0-x16", 1, 10.0), "2070": (8, "3.0-x16", 1, 12.0),
    "2080": (8, "3.0-x16", 1, 16.0), "2080ti": (11, "3.0-x16", 1, 22.0),
    "3060": (12, "4.0-x16", 2, 13.0), "3060ti": (8, "4.0-x16", 2, 16.0),
    "3070": (8, "4.0-x16", 2, 20.0), "3080": (10, "4.0-x16", 2, 30.0),
    "3090": (24, "4.0-x16", 2, 36.0),
    "4060": (8, "4.0-x8", 2, 15.0), "4060ti": (16, "4.0-x8", 2, 22.0),
    "4070": (12, "4.0-x16", 2, 29.0), "4070ti": (12, "4.0-x16", 2, 40.0),
    "4070tis": (16, "4.0-x16", 2, 44.0), "4080": (16, "4.0-x16", 2, 49.0),
    "4090": (24, "4.0-x16", 2, 82.0),
    "5060ti": (16, "4.0-x16", 2, 24.0), "5070": (12, "5.0-x16", 2, 31.0),
    "5070ti": (16, "5.0-x16", 2, 44.0), "5080": (16, "5.0-x16", 2, 57.0),
    "5090": (32, "5.0-x16", 2, 105.0),
}
Q4KM_BYTES_PARAM = 0.56       # ~bytes/param of a Q4_K_M-class quant (param count estimate)
GEMM_UTIL = 0.5               # sustained fraction of peak on batched verify GEMMs
KV_BPE = {"f16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q4_0": 0.5625}
# landmark bytes per (page, layer, kv-head): f16 vector vs product-quantized codes (§11)
LANDMARK_BYTES = {"f16": None, "pq32": 32, "pq16": 16, "pq8": 8}   # None -> head_dim*2
LAT_ROPE_DIMS = 64  # decoupled RoPE dims kept exact by KV-LAT (§10)

# synthetic profiles (Q4_K_M-class weights) so a fleet can be planned without the GGUF:
# (n_layer, n_embd, n_head, n_head_kv, total weight GiB, streamable frac)
MODEL_PROFILES = {
    "7b":  (32, 4096, 32, 32,  3.9, 0.92),
    "8b":  (32, 4096, 32,  8,  4.6, 0.77),
    "13b": (40, 5120, 40, 40,  7.4, 0.93),
    "70b": (80, 8192, 64,  8, 39.6, 0.95),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", nargs="?", help="GGUF file (or use --profile)")
    ap.add_argument("--profile", choices=sorted(MODEL_PROFILES),
                    help="synthetic Q4_K_M-class model profile instead of a GGUF")
    ap.add_argument("--gpu", choices=sorted(GPU_PRESETS), help="consumer GPU preset")
    ap.add_argument("--gpus", type=int, default=1,
                    help="number of identical GPUs (layer split: VRAM adds, PCIe aggregate "
                         "~one full link on x8/x8 desktop slots, compute adds)")
    ap.add_argument("--vram-budget", type=float, help="usable VRAM GiB (overrides --gpu)")
    ap.add_argument("--pcie", choices=sorted(PCIE_BW), help="PCIe link (overrides --gpu)")
    ap.add_argument("--display-reserve", type=float, default=1.0)
    ap.add_argument("--agents", default="1,2,4,8,16,32", help="agent counts to tabulate")
    ap.add_argument("--ctx", type=int, default=131072, help="per-agent context length")
    ap.add_argument("--shared-prompt", type=int, default=0,
                    help="tokens of a corpus/system prompt shared by all agents (prefix KV dedup)")
    ap.add_argument("--gen", type=int, default=1024, help="tokens generated per agent (for cost model)")
    ap.add_argument("--kv-type", choices=list(KV_BPE), default="q8_0", help="hot (VRAM) KV quant")
    ap.add_argument("--cold-kv-type", choices=list(KV_BPE), default=None,
                    help="host-resident cold-page KV quant (default: same as --kv-type)")
    ap.add_argument("--kv-lat", type=int, default=0, metavar="R",
                    help="KV-LAT latent rank per layer (0=off); see svmi-kvlat.py (§10)")
    ap.add_argument("--landmarks", choices=sorted(LANDMARK_BYTES), default="f16",
                    help="page-table landmark encoding (§11); pq16 = 16x smaller index")
    ap.add_argument("--prefetch-hit", type=float, default=0.0, metavar="H",
                    help="SPEC-PF prefetch hit rate 0..1 (§12); see svmi-pageprefetch.py")
    ap.add_argument("--prune-cold", type=float, default=0.0, metavar="F",
                    help="cold-tier fraction pruned by the attention-mass ledger (§13, lossy)")
    ap.add_argument("--kv-mode", choices=["auto", "full", "paged"], default="auto",
                    help="full: whole KV in VRAM; paged: CTX-VM demand paging; auto: paged "
                         "when ctx > 16K or full per-agent KV exceeds 1.5 GiB")
    ap.add_argument("--kv-window", type=int, default=4096,
                    help="paged mode: hot KV tokens resident in VRAM per agent "
                         "(attention sink + local window + hot-page cache)")
    ap.add_argument("--page", type=int, default=256, help="paged mode: KV page size in tokens")
    ap.add_argument("--kv-fetch", type=int, default=512,
                    help="paged mode: mean missed-page KV tokens fetched from host per decode step")
    ap.add_argument("--host-ram", type=float, default=64.0,
                    help="pinned host RAM budget GiB (weights + cold KV pool)")
    ap.add_argument("--overhead", type=float, default=1.25, help="activation/compute reserve GiB")
    ap.add_argument("--stream-slots", type=int, default=8)
    ap.add_argument("--bitspec-tokens", type=float, default=7.0,
                    help="mean tokens accepted per streamed pass under BitSpec (see svmi-bitspec.py)")
    ap.add_argument("--idle-frac", type=float, default=0.5,
                    help="fraction of agents idle at a given instant (hot window spillable)")
    args = ap.parse_args()

    # resolve GPU
    n_queues = 2
    gpu_tflops = 0.0
    n_gpu = max(1, args.gpus)
    if args.gpu:
        vram, link, n_queues, gpu_tflops = GPU_PRESETS[args.gpu]
        gpu_tflops *= n_gpu                       # compute adds across cards
        if args.vram_budget is None:
            args.vram_budget = max(0.0, n_gpu * vram - args.display_reserve)
        if args.pcie is None:
            args.pcie = link
    if args.vram_budget is None:
        ap.error("provide --vram-budget or --gpu")
    pcie_bw = PCIE_BW[args.pcie or "4.0-x16"]

    if args.profile:
        n_layer, n_embd, n_head, n_head_kv, w_gib, stream_frac = MODEL_PROFILES[args.profile]
        arch = f"profile:{args.profile}"
        model_name = f"{args.profile} (synthetic Q4_K_M-class)"
        total_w = int(w_gib * GiB)
        streamable_w = int(total_w * stream_frac)
        # one streamable tensor ~ streamable bytes / (~7 matmul tensors per layer)
        max_stream_tensor = streamable_w // (7 * n_layer)
    elif args.model:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))
        from gguf import GGUFReader

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
        streamable_w = sum(int(t.n_bytes) for t in reader.tensors
                           if ".ffn_" in t.name or ".attn_" in t.name)
        max_stream_tensor = max((int(t.n_bytes) for t in reader.tensors
                                 if ".ffn_" in t.name or ".attn_" in t.name), default=0)
    else:
        ap.error("provide a GGUF path or --profile")

    head_dim = n_embd // n_head if n_head else 0
    min_resident = total_w - streamable_w      # embeddings / norms / output head

    hot_bpe  = KV_BPE[args.kv_type]
    cold_bpe = KV_BPE[args.cold_kv_type or args.kv_type]
    if args.kv_lat > 0:   # KV-LAT (§10): one shared latent + decoupled RoPE key per layer
        kv_tok_hot  = (args.kv_lat + LAT_ROPE_DIMS) * n_layer * hot_bpe
        kv_tok_cold = (args.kv_lat + LAT_ROPE_DIMS) * n_layer * cold_bpe
    else:
        kv_tok_hot  = 2 * head_dim * n_head_kv * n_layer * hot_bpe    # bytes/token, all layers
        kv_tok_cold = 2 * head_dim * n_head_kv * n_layer * cold_bpe

    shared_ctx = min(args.shared_prompt, args.ctx)
    unique_ctx = args.ctx - shared_ctx

    full_kv_agent = int(kv_tok_hot * unique_ctx)   # full mode, per agent, VRAM
    paged = args.kv_mode == "paged" or (
        args.kv_mode == "auto" and (args.ctx > 16384 or full_kv_agent > 1.5 * GiB))

    lm_bytes = LANDMARK_BYTES[args.landmarks]
    landmark_page = n_layer * n_head_kv * (head_dim * 2 if lm_bytes is None else lm_bytes)
    if paged:
        window = min(args.kv_window, args.ctx)
        pages_unique = math.ceil(unique_ctx / args.page)
        pages_shared = math.ceil(shared_ctx / args.page)
        kv_vram_agent  = int(window * kv_tok_hot) + pages_unique * landmark_page
        kv_vram_shared = pages_shared * landmark_page             # landmarks only, stored once
        kv_host_agent  = int(unique_ctx * kv_tok_cold * (1.0 - args.prune_cold))
        kv_host_shared = int(shared_ctx * kv_tok_cold)            # shared corpus never pruned
        # per decode step per agent; SPEC-PF-hidden fetches leave the critical path (§12)
        kv_fetch_bytes = int(args.kv_fetch * kv_tok_cold * (1.0 - args.prefetch_hit))
    else:
        kv_vram_agent  = full_kv_agent
        kv_vram_shared = int(kv_tok_hot * shared_ctx)
        kv_host_agent = kv_host_shared = kv_fetch_bytes = 0

    ring = args.stream_slots * max_stream_tensor
    budget = int(args.vram_budget * GiB)
    host_budget = int(args.host_ram * GiB)
    fixed = ring + int(args.overhead * GiB)

    print(f"model        : {model_name} ({arch}, {n_layer} layers, {total_w/GiB:.2f} GiB, "
          f"streamable {streamable_w/GiB:.2f})")
    if args.gpu:
        print(f"gpu          : {(str(n_gpu) + 'x ') if n_gpu > 1 else ''}{args.gpu}  ({args.pcie}, "
              f"{n_queues} H2D copy engine(s), {pcie_bw:.0f} GB/s aggregate"
              + (", x8/x8 slots assumed" if n_gpu > 1 else "") + ")")
    print(f"context      : {args.ctx:,} tok/agent ({shared_ctx:,} shared + {unique_ctx:,} unique), "
          f"kv {args.kv_type}" + (f"/{args.cold_kv_type} cold" if args.cold_kv_type else ""))
    print(f"kv mode      : {'CTX-VM paged' if paged else 'full-in-VRAM'}"
          + (f"  (window {args.kv_window:,} tok, {args.page}-tok pages, "
             f"~{args.kv_fetch} tok fetched/step)" if paged else ""))
    wave2 = []
    if args.kv_lat > 0:
        wave2.append(f"KV-LAT r={args.kv_lat}+{LAT_ROPE_DIMS} rope")
    if args.landmarks != "f16":
        wave2.append(f"{args.landmarks} landmarks")
    if args.prefetch_hit > 0:
        wave2.append(f"SPEC-PF hit {args.prefetch_hit:.0%}")
    if args.prune_cold > 0:
        wave2.append(f"cold pruned {args.prune_cold:.0%} (lossy)")
    if wave2:
        print("second wave  : " + ", ".join(wave2))
    if paged:
        print(f"kv/agent     : VRAM {kv_vram_agent/GiB:.2f} GiB (vs {full_kv_agent/GiB:.1f} full "
              f"-> {full_kv_agent/max(1,kv_vram_agent):.0f}x smaller) + host {kv_host_agent/GiB:.2f} GiB")
    else:
        print(f"kv/agent     : VRAM {kv_vram_agent/GiB:.2f} GiB")
    print(f"vram/host    : budget {budget/GiB:.2f} GiB (ring {ring/GiB:.2f} + overhead "
          f"{args.overhead:.2f} reserved)  |  host budget {args.host_ram:.0f} GiB "
          f"(weights pinned: {total_w/GiB:.2f})\n")

    print(f"{'agents':>6} {'resident W':>10} {'streamed W':>10} {'KV vram':>8} {'host RAM':>8} "
          f"{'fits?':>6} {'agg tok/s*':>10} {'per-agent':>9}")

    agent_counts = [int(a) for a in args.agents.split(",") if a.strip()]
    max_fit = 0
    for n in agent_counts:
        kv_vram = kv_vram_shared + n * kv_vram_agent
        host = total_w + kv_host_shared + n * kv_host_agent
        avail_for_w = budget - fixed - kv_vram
        why = None
        if avail_for_w < min_resident:
            why = "VRAM: KV+reserve exceed budget"
        elif host > host_budget:
            why = f"host RAM: needs {host/GiB:.0f} GiB (raise --host-ram or --cold-kv-type q4_0)"
        if why:
            print(f"{n:>6} {'—':>10} {'—':>10} {kv_vram/GiB:7.2f}G {host/GiB:7.0f}G {'NO':>6}  ({why})")
            continue
        max_fit = n
        resident_w = min(total_w, avail_for_w)
        streamed_w = total_w - resident_w
        # one batched streamed pass serves all n agents; each accepted token also pays
        # its agent's missed-page KV fetch on the same PCIe link
        bytes_per_pass = streamed_w + n * args.bitspec_tokens * kv_fetch_bytes
        if bytes_per_pass <= 0:
            agg_s, per_s = "resident", "—"
        else:
            agg = pcie_bw * 1e9 / bytes_per_pass * n * args.bitspec_tokens
            mark = ""
            if gpu_tflops > 0:
                params_est = total_w / Q4KM_BYTES_PARAM
                ceiling = GEMM_UTIL * gpu_tflops * 1e12 / (2.0 * params_est)
                if agg > ceiling:                 # streaming stops binding; FLOPs do
                    agg, mark = ceiling, "c"
            agg_s, per_s = f"{agg:8.1f}{mark}", f"{agg/n:8.1f}"
        print(f"{n:>6} {resident_w/GiB:9.2f}G {streamed_w/GiB:9.2f}G {kv_vram/GiB:7.2f}G "
              f"{host/GiB:7.0f}G {'yes':>6} {agg_s:>10} {per_s:>9}")

    print("\n* aggregate decode tok/s, streaming-bound region: one PCIe pass over streamed "
          "weights serves all agents\n  (stream-once-serve-many), x BitSpec mean "
          f"{args.bitspec_tokens:.0f} tok/pass, minus paged-KV fetch traffic; 'c' marks rows\n"
          "  clamped at the GPU compute ceiling "
          + (f"(~{GEMM_UTIL * gpu_tflops * 1e12 / (2.0 * total_w / Q4KM_BYTES_PARAM):,.0f} tok/s "
             f"aggregate at {GEMM_UTIL:.0%} of {gpu_tflops:.0f} TF fp16)." if gpu_tflops > 0
             else "(unknown GPU: give --gpu for the FLOPs clamp)."))

    print("\n-- economics --")
    print(f"marginal agent : {kv_vram_agent/GiB:.2f} GiB VRAM + {kv_host_agent/GiB:.2f} GiB host "
          f"(weights shared, O(1) in agents)")
    naive = (total_w + kv_tok_hot * args.ctx) / GiB
    print(f"naive per agent: {naive:.1f} GiB VRAM (own model copy + full-VRAM KV) -> "
          f"{naive/max(1e-9, kv_vram_agent/GiB):.0f}x denser in agents/GiB of VRAM")
    if shared_ctx > 0 and len(agent_counts) > 0:
        big = max(agent_counts)
        saved = (big - 1) * shared_ctx
        total_work = big * (args.ctx - shared_ctx + args.gen) + shared_ctx
        print(f"prefix dedup   : {shared_ctx:,}-tok shared corpus prefilled once -> saves "
              f"{saved:,} prompt tokens at {big} agents "
              f"({saved/(saved+total_work):.0%} of fleet prefill+decode work)")
    print(f"idle spill     : at {args.idle_frac:.0%} idle, each parked agent frees "
          f"{kv_vram_agent/GiB:.2f} GiB VRAM (hot window + page table page out; "
          f"wake-up hides behind the first layers' compute)")
    if max_fit:
        print(f"\nverdict: ~{max_fit} concurrent agents at ctx {args.ctx:,} on this budget"
              + ("; scale further with --cold-kv-type q4_0, a smaller --kv-window, "
                 "or more --host-ram." if paged else "."))
    else:
        print("\nverdict: 0 agents fit — switch --kv-mode paged, shrink --kv-window, or use "
              "--cold-kv-type q4_0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
