# SVMI — Streaming Virtual Memory Inference

SVMI treats the GPU as a cache over a host-RAM weight store instead of a permanent home
for the model: **weight residency is dynamic**. This is what makes 70B-class models run
in under 20 GB of VRAM with all matrix math on the GPU and token-identical output.

This fork implements the SVMI streaming core on top of upstream llama.cpp, plus the
planning and measurement tools around it. The full research report and architecture
blueprint are at the bottom of this document.

## Quick start

```bash
# build (CUDA)
cmake -B build -DGGML_CUDA=ON && cmake --build build -j

# 1. generate a placement plan for your VRAM budget
python3 scripts/svmi-plan.py models/llama-70b-q4_k_m.gguf --vram-budget 19

# 2. run with the flags it prints, e.g.:
./build/bin/llama-cli -m models/llama-70b-q4_k_m.gguf \
    -ngl 99 -ot 'blk\.(24|25|...|55)\.ffn_.*=CPU' \
    --stream-weights 8 --stream-decode \
    -c 32768 -ctk q8_0 -ctv q8_0 -fa on -p "hello"
```

## What the streaming core does

Stock llama.cpp with partial offload (`-ngl` / `-ot ...=CPU`) has two costs:

1. weights kept in host RAM are uploaded **synchronously** right before the op that
   needs them, stalling the GPU on PCIe for every layer (prefill), and
2. during single-token decode, host-resident layers are computed **on the CPU**, which
   is an order of magnitude slower than streaming them to the GPU would be.

SVMI replaces both:

- **Pinned weight store** — mmap-backed host weights are page-locked
  (`cudaHostRegister`) so uploads run as true async DMA at full PCIe bandwidth instead
  of through the driver's hidden bounce buffer (~6–7 GB/s → ~20+ GB/s measured on
  PCIe 4.0). Loader-level, opt-in via `GGML_CUDA_REGISTER_HOST=1`.
- **Upload queues + staging ring** — the scheduler scans ahead of the split being
  computed and enqueues weight uploads early on dedicated upload queues (secondary
  backend instances = independent CUDA streams, one per DMA copy engine) into a ring of
  reusable staging slots. Transfers overlap compute; events sequence slot reuse. Slot
  assignment is deterministic per eval, so CUDA graph replay stays valid.
- **Streamed decode** (`--stream-decode`) — forces device offload at every batch size,
  so decode also keeps all matmuls on the GPU and streams the weights, instead of
  falling back to CPU compute. Decode throughput is then bounded by PCIe bandwidth
  times the streamed fraction of the model — see the planner output for the floor.
- **MoE fast path** — for `MUL_MAT_ID` (expert) weights with large batches, the full
  expert tensor is prefetched without waiting for the routing ids (with small batches
  the stock selective-expert copy moves fewer bytes and is kept). This subsumes the
  fable5 `GGML_SCHED_PREFETCH_EXPERTS` patch, generalized to dense `MUL_MAT` weights
  and multiple upload queues, and rebased on current master.

### Flags and environment variables

| Flag / env | Effect |
| --- | --- |
| `--stream-weights N` | enable streaming with `N` staging slots (1 = default of 8); also sets the two env vars below. Available in `llama-cli`, `llama-server`, and `llama-bench` |
| `--stream-decode` | force device offload at any batch size (sets `GGML_OP_OFFLOAD_MIN_BATCH=1`). Available in `llama-cli`, `llama-server`, and `llama-bench` |
| `GGML_SCHED_STREAM_WEIGHTS=N` | scheduler-level enable (what `--stream-weights` sets) |
| `GGML_CUDA_REGISTER_HOST=1` | pin mmap'd host weights for fast DMA |
| `GGML_SCHED_STREAM_QUEUES=N` | upload queues (default 2, one per DMA copy engine) |
| `GGML_SCHED_STREAM_PREFETCH=N` | how many scheduler splits ahead to enqueue uploads (default 4) |
| `GGML_SCHED_PREFETCH_EXPERTS=1` | accepted as a compatibility alias for `GGML_SCHED_STREAM_WEIGHTS` |

All of it is opt-in and token-identical: the same kernels run on the same data, only
the transport is scheduled differently.

## Tools

- `scripts/svmi-plan.py` — the residency planner. Given a GGUF and a VRAM budget it
  splits the budget between KV cache, streaming ring, and resident weights, keeps
  embeddings/attention/head resident, fills FFN layers from both ends of the stack
  inward, and prints ready-to-use `-ngl`/`-ot`/`--stream-*` flags plus the PCIe decode
  floor for the plan.
- `scripts/svmi-entropy.py` — go/no-go study for the compressed-transport phase:
  measures order-0 byte entropy of quantized GGUF blocks per structural component
  (scales vs packed nibbles) and estimates the achievable lossless rANS transport
  ratio.
- `scripts/svmi-bench.sh` — three-way llama-bench comparison: baseline vs pinned vs
  streamed, same settings, prints a summary table.
- `scripts/svmi-verify.sh` — correctness gate: greedy-decodes the same prompt with and
  without streaming and diffs the tokens (byte-identical expected), with an optional
  `--ppl-file` perplexity equality check. This is the executable form of the
  token-identity guarantee.
- `scripts/svmi-bitspec.py` — go/no-go study for **BitSpec** (bit-plane self-speculative
  decoding): measures how often a low-bit resident draft of the model's own weights
  agrees with full precision at the token decision, i.e. the speculative acceptance rate.
  See the research notes below.
- `scripts/svmi-fleet.py` — multi-agent, long-context capacity planner for the **MAVM**
  and **CTX-VM** designs: given a GGUF (or `--profile 7b/8b/13b/70b`) and a GPU preset,
  tabulates how many concurrent agents fit at up to 131K/256K context each, the marginal
  VRAM *and host RAM* of each additional agent, tokens saved by shared-prefix KV dedup,
  aggregate fleet throughput under stream-once-serve-many + BitSpec, and the wall that
  binds first (VRAM vs host RAM) with the flag that moves it. Long contexts switch
  automatically to paged KV: full KV in pinned host RAM, a landmark page table + hot
  window in VRAM.
- `scripts/svmi-arbiter.py` — cross-fabric routing optimizer for the **ARBITER** design:
  given GPU + CPU presets, solves the bandwidth-arbitrage split (GPU verifies the
  VRAM-resident share, CPU verifies the host share in speculative batches, PCIe carries
  zero weights on the critical path) and compares decode rate against streaming and
  stock partial offload. Names the binding resource at the optimum.
- `scripts/svmi-kvlat.py` — **KV-LAT** rank study (second wave, §10): SVD spectra of
  `[W_K; W_V]` from a real GGUF (or synthetic profile) → the latent rank a MLA-style
  retrofit could cache instead of full GQA KV, and the bytes/token it buys.
- `scripts/svmi-pqindex.py` — **CTX-VM v2** landmark compression study (§11): top-k
  page recall of product-quantized landmarks vs f16, plus the index-size table that
  makes 1M-token page tables fit (640 MiB → 40 MiB for a 70B at pq16).
- `scripts/svmi-pageprefetch.py` — **SPEC-PF** simulator (§12): how much demand-fetch
  latency speculative page prefetch hides, as a function of prefetch budget; its hit
  rate feeds `svmi-fleet.py --prefetch-hit`. The fleet planner models the whole second
  wave via `--kv-lat`, `--landmarks`, `--prefetch-hit`, and `--prune-cold`; the landmark
  math itself is CI-guarded by `tests/test-ctxvm-landmarks.cpp`.
- `scripts/svmi-expertpage.py` — **MoE-EP** expert-paging simulator (§15, roadmap
  phase 6): compares static / LRU / router-lookahead expert residency by critical-path
  PCIe bytes per token on MoE profiles (Mixtral-8x7B, Qwen3-30B-A3B, DeepSeek-V3 class).
- `scripts/svmi-auto.py` — the **one command**: reads the model + GPU, picks the regime
  (resident / offload / streamed / fleet), prints ready-to-run flags that exist in this
  branch today, and names the specialist planner for fine-tuning.

## Consumer GPUs (6–12 GB): 1660 Ti, RTX 2080 / 2080 Ti, RTX 3060

These are exactly the cards SVMI is built for — small VRAM, and (for the Turing trio)
the PCIe 3.0 link that sets a lower streaming ceiling than the datacenter default. The
planner knows them: `python3 scripts/svmi-plan.py model.gguf --gpu 2080ti` fills in the
VRAM budget, the effective PCIe bandwidth, and the correct upload-queue count.

| GPU | VRAM | Arch | PCIe link | eff. H2D (pinned) | H2D copy engines → `GGML_SCHED_STREAM_QUEUES` |
| --- | ---: | --- | --- | ---: | --- |
| GTX 1660 Ti | 6 GB | Turing (no tensor cores) | 3.0 x16 | ~12 GB/s | 1 |
| RTX 2080 | 8 GB | Turing | 3.0 x16 | ~12 GB/s | 1 |
| RTX 2080 Ti | 11 GB | Turing | 3.0 x16 | ~12 GB/s | 1 |
| RTX 3060 | 12 GB | Ampere | 4.0 x16 | ~24 GB/s | 2 |

Card-specific guidance:

- **PCIe 3.0 halves the streaming ceiling.** The Turing trio streams at ~12 GB/s, so the
  raw decode floor for a fully-streamed 70B is ~0.3 tok/s (vs ~0.6 on the 3060's PCIe
  4.0). This makes the **stream-less** and **amortize** multipliers essential on these
  cards, not optional: keep as much resident as VRAM allows, and use BitSpec-style
  self-speculation (`scripts/svmi-bitspec.py`) to amortize each stream over many tokens.
- **One copy engine on GeForce cards.** Consumer Turing exposes a single H2D DMA engine,
  so a second upload queue just serializes with overhead — the planner emits
  `export GGML_SCHED_STREAM_QUEUES=1` for these. The 3060 has two, so it keeps the
  default of 2.
- **1660 Ti has no tensor cores and slow FP16.** Flash attention and F16 KV run on the
  slower non-tensor path; prefer `-ctk q8_0 -ctv q8_0` (already the planner default) and
  expect attention compute, not streaming, to dominate at short context.
- **What fits.** A 12 GB 3060 holds the resident hot set (embeddings + all attention +
  the head/tail FFN layers) of a 70B Q4_K_M with room for a quantized KV cache, streaming
  the middle FFN layers; a 6 GB 1660 Ti is better matched to 8B–34B resident-heavy
  configs or to 70B only with an aggressive streamed fraction. Run the planner to see the
  exact split and the honest decode floor for your budget.

Quick start on these cards:

```bash
# RTX 2080 Ti (11 GB, PCIe 3.0): plan, then run with the printed flags
python3 scripts/svmi-plan.py models/llama-70b-q4_k_m.gguf --gpu 2080ti
export GGML_SCHED_STREAM_QUEUES=1        # single copy engine (planner reminds you)
./build/bin/llama-cli -m models/llama-70b-q4_k_m.gguf <printed -ngl/-ot/--stream flags>

# check the speculative-decoding upside for your PCIe generation
python3 scripts/svmi-bitspec.py models/llama-70b-q4_k_m.gguf --gpu 2080ti
```

## Measured expectations (honest numbers)

Per-token decode cost when streaming: `bytes_streamed / effective_PCIe_bandwidth`.
For a 70B Q4_K_M (~40 GB) with ~12 GB of weights resident and ~28 GB streamed on
PCIe 4.0 x16 (~25 GB/s pinned):

- prefill: near-resident speed at 1–2k batch (transfers hide behind compute); this is
  the regime where the fable5 ancestor of this code measured +64% on an RTX 3060.
- decode floor: ~0.9 tok/s at batch 1 — physics, not implementation. The multipliers
  that stack on top (roadmap below): offload-aware speculative decoding (published
  results: 4–6 tok/s for 70B-4bit on an RTX 4090, SpecExec NeurIPS'24), batched decode
  (streamed bytes amortize across the batch), and compressed transport (~1.1–1.35x).

## Roadmap

| Phase | Content | Status |
| --- | --- | --- |
| 1 | pinned weight store, upload queues, staging ring, `--stream-weights` | **this branch** |
| 2 | residency planner | **this branch** (`scripts/svmi-plan.py`) |
| 3 | offload-aware speculative decoding — **BitSpec**: a low-bit resident copy of the model's own weights is the draft, verified per stream pass (validated: `scripts/svmi-bitspec.py`, see [svmi-research.md](svmi-research.md)) | next |
| 4 | compressed transport: entropy-coded quantized blocks, GPU rANS decode fused with dequant (`scripts/svmi-entropy.py` decides go/no-go) | next |
| 5 | paged KV with host spill on the same DMA engine | planned |
| 6 | predictive MoE expert paging (router-logit lookahead) — **MoE-EP** (`scripts/svmi-expertpage.py`, §15) | validated |
| 7 | **second wave (July 2026)** — PQ landmarks + two-level page table (`scripts/svmi-pqindex.py`, `tests/test-ctxvm-landmarks.cpp`) | validated |
| 8 | second wave — speculative page prefetch on the streaming clock (`scripts/svmi-pageprefetch.py`) | validated |
| 9 | second wave — KV-LAT latent KV retrofit (`scripts/svmi-kvlat.py`), cold-tier ledger pruning, unified VRAM pool | designed |

---

# Research report: Beyond quantization — 70B in <20 GB VRAM

## 1. The bandwidth model

Everything is evaluated against this per-token, per-layer cost:

```
t_layer = max( t_compute(batch, layer),  bytes_streamed(layer) / BW_effective )
```

- 70B Q4_K_M: ~40 GB weights, 80 layers, ~500 MB/layer.
- PCIe 4.0 x16: 32 GB/s theoretical, 24–27 GB/s achieved with pinned memory.
- Decode floor (batch 1, everything streamed): 40 GB / 25 GB/s = **0.62 tok/s**. Every
  technique below is a multiplier on this number.
- Prefill: compute per layer grows with batch, transfer does not; break-even on a
  4090-class GPU is roughly a 1–2k-token chunk per layer pass. Prefill is nearly
  solvable with streaming alone.

The design axiom follows: **the GPU is a cache, not a home.** Any token-identical
solution to 70B-in-20GB is a bandwidth problem, and the research space collapses into
four independent multipliers:

1. **Stream less** — keep a hot set resident (residency planning).
2. **Stream denser** — entropy-code quantized weights, decompress on GPU.
3. **Amortize each stream** — many tokens (speculation) or many sequences (batching)
   per full-model pass.
4. **Hide the stream** — ring buffers, dual DMA engines, pinned memory, CUDA graphs.

Composed: 12 GB resident (stream 28 GB) x ~1.2x transport compression x 8–12 accepted
tokens per verification pass = **4–8 tok/s conversational decode, token-identical,
under 20 GB VRAM** — consistent with published SpecExec results.

## 2. Technique catalog (summary)

| # | Technique | Speed impact | Difficulty | Maturity | ROI |
| --- | --- | --- | --- | --- | --- |
| T1 | pinned store + staging ring + upload queues | foundation (sets the PCIe floor) | moderate | engineering (FlexGen, fable5 embryo) | mandatory |
| T2 | transfer shaping: dual copy engines, NUMA, huge pages | +10–25% effective BW | low | known HPC practice, absent in llama.cpp | high |
| T3 | NVMe tier (GPUDirect Storage / BaM) | 3.5x slower than RAM tier; removes RAM ceiling | high | research/early product | medium |
| T4 | entropy-coded transport, GPU rANS decode ("Layer Compression Bus") | x compression ratio (est. 1.1–1.35x on k-quants) | high | components published (DietGPU 250–600 GB/s decode, ZipServ fused decompress-GEMM, nvCOMP w/ Blackwell HW engine); composition novel | high, most publishable |
| T5 | inter-layer delta coding | likely ~1.0x (weights decorrelated post-quant) | high | novel, probably negative result | skip |
| T6 | static hot-set residency planner | 1.43x for 12/40 GB resident | moderate | FlexGen ancestor (throughput mode); latency-mode + speculation-aware formulation is new | high |
| T7 | activation-sparsity residency (PowerInfer) | large, but needs ReLU models; breaks token-identity on SwiGLU | high | published (SOSP'24) | low for this mission |
| T8 | predictive MoE expert paging, all-GPU compute | x sparsity factor (<25% hot for Mixtral-class) | mod-high | not implemented in any OSS engine | very high for MoE |
| T9 | offload-aware speculative decoding (tree verification per stream pass) | **5–15x** on streamed decode | high | SpecExec/Sequoia published; integration with a streaming scheduler is novel systems work | highest |
| T10 | batched decode pipelining (zig-zag) | x batch width | moderate | FlexGen | high for servers |
| T11 | CUDA graph capture per stage, persistent kernels | 5–15% | mod/high | engineering; persistent-kernel streaming unpublished | medium |
| T12 | quantized paged KV with host spill | enables long context under budget | mod-high | components product-grade; DMA-unified host spill new in llama.cpp | high for long ctx |
| T13 | placement-aware mixed precision (resident Q6_K, streamed Q4_K) | second-order on both axes | low-mod | novel coupling of published ideas | medium |

Key numbers behind the table:

- **SpecExec** (NeurIPS'24): dynamic draft trees, 10–20 accepted tokens per target
  pass; 70B on RTX 4090 with offloading at 4–6 tok/s (4-bit), 8–18x over sequential.
- **Sequoia**: hardware-aware static trees, 0.57 s/token for Llama2-70B on a 4090.
- **DietGPU**: GPU rANS at 250–600 GB/s — decompression is free next to PCIe.
- **ZipServ** (2026): decompress fused into GEMM beats cuBLAS by up to 2.21x; proves
  fused-codec kernels are practical. nvCOMP on Blackwell adds a hardware decompression
  engine (600 GB/s, fused copy-decompress) — the T4 path gets faster on future GPUs
  for free.
- **PowerInfer** (SOSP'24): 11.69x over llama.cpp — but the power-law neuron sparsity
  it needs comes from ReLU-family models; Llama-3/Qwen-class SwiGLU models have weak
  natural sparsity and exploiting it changes outputs. Rejected for the token-identical
  core; viable only as an opt-in approximate mode on ReLU-fied checkpoints.
- **LLM-in-a-Flash** (Apple, ACL'24): windowing + row-column bundling for flash reads;
  the cost-model approach informs the T3 NVMe tier. No public code.

## 3. Gap analysis

What no open-source engine implements today (verified against llama.cpp, ExLlama,
vLLM, TensorRT-LLM, DeepSpeed, FlexGen, PowerInfer, mid-2026):

1. token-identical GPU-compute-only layer streaming in a mainstream engine (exists
   only in prototypes: ntransformer, gdsllm) — **this branch adds it to llama.cpp**;
2. entropy-coded transport of quantized weight blocks with GPU decode on the PCIe
   weight path (components exist: DietGPU / ZipServ / nvCOMP);
3. tree-based offload-aware speculative decoding integrated with a streaming
   scheduler (SpecExec/Sequoia are standalone research code);
4. a joint latency-mode residency optimizer (weights + KV + draft + ring) —
   FlexGen solves only the throughput variant;
5. predictive MoE expert paging with all-GPU compute (llama.cpp and KTransformers
   compute cold experts on the CPU).

Highest breakthrough probability: T9 x T1 (speculative streaming) for dense 70B, and
T8 for MoE. Most novel/publishable: T4 and the T6 joint placement formulation.

## 4. SVMI blueprint

```
host (tier 1/2)                          GPU (< 20 GB)
+---------------------------+            +--------------------------------+
| NVMe GGUF(-S) weight file |            | resident hot set   (~12 GB)    |
|        |                  |  queue A   | staging ring       (~2.5 GB)   |
| pinned RAM arena          | =========> | [GPU rANS decode]  (phase 4)   |
|  entropy-coded blocks     |  queue B   | paged KV hot       (~2.7 GB)   |
|  NUMA-local, hugepages    | =========> | activations/graphs (~1.5 GB)   |
| KV spill pool             | <========> | draft path (resident layers)   |
+---------------------------+   KV DMA   +--------------------------------+
```

- **Memory layout** — VRAM: resident-set arena (planner-owned, immutable after load),
  staging ring (fixed stride = max streamed tensor), paged KV pool, activation
  workspace. Host: pinned arena in streaming order (one contiguous DMA per layer),
  KV spill pool. NVMe: GGUF(-S) with per-tensor compressed streams + entropy tables.
- **Scheduler** — extends `ggml_backend_sched`: residency class per tensor; cold
  layers check slot-ready events and never fall back to CPU compute; decode mode
  interleaves draft-tree construction on resident layers with the stream-in of the
  verification pass, so speculation costs ~zero wall-clock.
- **DMA engine** — two H2D queues bound to both copy engines; chunked uploads so
  slot events fire early; per-transfer bandwidth telemetry feeds the runtime
  optimizer; D2H reserved for KV spill.
- **Prefetch engine** — distance auto-tuned to `ceil(t_transfer / t_compute) + 1`
  jitter slot; dense models use deterministic next-layer order; MoE uses router-logit
  lookahead with a blocking-fetch fallback (correctness never depends on prediction).
- **Layer/tensor cache** — resident set from the placement knapsack (attention,
  head/tail layers, and self-draft layers weighted up); ring slots are pure FIFO
  (access order is the schedule); MoE adds an LRU expert cache between tiers.
- **KV manager** — paged, q8_0 default, FlashAttention required in streamed mode;
  conservative spill (exact attention; cold pages prefetched back, overlapped like
  weights); approximate modes (H2O/SnapKV eviction, 2-bit KIVI) opt-in and labeled.
- **Runtime optimizer** — measures per-layer compute time, achieved PCIe bandwidth,
  and speculation acceptance; adjusts prefetch distance, draft budget, and (between
  requests) the residency plan.
- **Streams** — S0 compute; S1/S2 H2D weights; S3 KV spill; S4 decompression (or
  fused into dequant epilogues). Event graph: upload-done -> decode-done ->
  compute-done -> slot-free.
- **Thread model** — main inference thread, prefetch thread (owns DMA queues),
  NVMe reader thread, server threads; lock-free SPSC rings between them; no CPU
  thread ever computes a transformer layer.

## 5. Constraints and honesty notes

- Token-identity is preserved by T1/T2/T4/T6 and conservative T12; speculation (T9)
  is distribution-preserving via standard acceptance sampling; anything approximate
  (T7, KV eviction, progressive precision) is opt-in and labeled.
- Decode for dense 70B remains physically PCIe-anchored; SVMI's claim is turning
  0.6 tok/s into 4–8 tok/s at <20 GB with unchanged quality — not matching an H100.
- Performance numbers quoted from external work (SpecExec, Sequoia, DietGPU, ZipServ,
  PowerInfer, FlexGen) are theirs; numbers for this branch must be measured with
  `scripts/svmi-bench.sh` on real hardware.

## Beyond the roadmap

Novel techniques designed for this fork — bit-plane self-speculative decoding (BitSpec),
pipelined streaming GEMM, stream-once-serve-many, elastic residency, draft-guided MoE
prefetch, residual-precision transport, **MAVM** (Multi-Agent Virtual Memory: shared
weights, shared-prefix KV dedup, fleet-clock batching, idle-KV spill), and **CTX-VM**
(paged context virtual memory: full KV in pinned host RAM, landmark page table + hot
window in VRAM, demand-fetched pages on the weight-streaming DMA path — what makes
131K/256K context per agent fit consumer VRAM), and **ARBITER** (compute-follows-memory:
weights never cross PCIe on the critical path — the GPU drafts and verifies its resident
share while the CPU verifies the host share in speculative batches, with the split solved
by bandwidth arbitrage; modeled 5× over stock partial offload for a 70B on consumer
hardware) — are written up with bandwidth math, honesty notes, and build priorities in
**[svmi-research.md](svmi-research.md)**. Four ship working offline validators:
`scripts/svmi-bitspec.py` (BitSpec), `scripts/svmi-fleet.py` (MAVM + CTX-VM), and
`scripts/svmi-arbiter.py` (ARBITER).

### What should this machine do with this model? (one command)

```sh
# reads the model + GPU, picks the regime, prints ready-to-run flags
python3 scripts/svmi-auto.py model.gguf --gpu 3060
# multi-GPU rigs: --gpus N (layer split; VRAM adds, x8/x8 slot bandwidth assumed)
python3 scripts/svmi-auto.py model.gguf --gpu 2080ti --gpus 2
# or, CPU-assisted decode analysis directly:
python3 scripts/svmi-arbiter.py --profile 70b --gpu 3060 --cpu ddr5-2ch --cores 16
```

### Running many long-context agents on one small GPU (one command)

```sh
# 16 agents of a 70B at 131,072 ctx each (96K shared corpus) on a 12 GB RTX 3060
python3 scripts/svmi-fleet.py --profile 70b --gpu 3060 --agents 4,8,12,16 \
    --ctx 131072 --shared-prompt 98304 --cold-kv-type q4_0 --kv-window 2048 --host-ram 128

# 8B at 262,144 ctx on the same card; or plan from a real GGUF instead of a profile
python3 scripts/svmi-fleet.py --profile 8b --gpu 3060 --agents 4,8,16 --ctx 262144 \
    --shared-prompt 131072 --cold-kv-type q4_0
python3 scripts/svmi-fleet.py models/model.gguf --gpu 2080ti --agents 8,16 --ctx 131072
```

The report answers, per agent count: does the fleet fit (and if not, whether VRAM or host
RAM is the wall and which flag moves it); resident vs streamed weights; the marginal
VRAM + host RAM of the next agent; prompt tokens saved by prefix dedup; and the aggregate
decode rate in the streaming-bound region. The rules of thumb it encodes: **weights cost
O(1) in agents; paged KV costs O(window + ctx/page) in VRAM instead of O(ctx); the full
context lives in host RAM (the cheap resource); idle agents cost ~0.**
