# SVMI research notes — novel techniques for streamed 70B inference

This document proposes techniques that are **not** in mainstream offloading engines
(llama.cpp, ExLlama, vLLM, TensorRT-LLM, DeepSpeed-Inference, FlexGen, PowerInfer as of
mid-2026) and are not already covered by the SVMI roadmap in [`svmi.md`](svmi.md). Each
entry states the idea, why it is new, the bandwidth math, an honesty note on limits, and
what to build. Everything is anchored to the SVMI axiom: for a model that does not fit in
VRAM, decode latency is `bytes_streamed / PCIe_bandwidth`, and the only ways to beat it
are **stream less**, **stream denser**, **amortize each stream**, or **hide the stream** —
plus, for multi-agent fleets, a fifth axis: **share every stream and every byte across
agents** (§7), for long context a sixth: **page the context itself** (§8), and the
endgame seventh: **don't stream at all — send compute to the memory** (§9).

Four of these ship with working, GPU-free validators: BitSpec
(`scripts/svmi-bitspec.py`), MAVM + CTX-VM (`scripts/svmi-fleet.py`), and ARBITER
(`scripts/svmi-arbiter.py`); measured results are inline.

---

## 1. BitSpec — bit-plane self-speculative decoding  *(amortize; validated)*

**Idea.** Speculative decoding needs a cheap draft model whose tokens the expensive model
verifies in one pass. Instead of training draft heads (Medusa/EAGLE) or shipping a
separate small model, **use the target's own weights truncated to a few bits per value as
the draft.** A 3-bit copy of the model is ~19% of its f16 size, small enough to keep
*resident* in VRAM. So the draft forward pass costs **zero PCIe traffic**, and the
full-precision weights — the ones that must be streamed — are touched only once per
*verification* pass, which accepts a whole run of drafted tokens at once. That divides the
dominant streaming cost by the mean accepted-run length.

**Why it's new.** Existing self-speculation skips *layers* (LayerSkip, draft-&-verify);
Medusa/EAGLE add *trained* heads; SpecExec/Sequoia use a *separate* draft model. BitSpec's
draft is the same architecture and vocabulary as the target, needs no training, and — the
part nobody exploits — is chosen specifically so the draft is the cheap-to-*store* copy in
an offloading system. The draft's residency is the whole point, not an afterthought.

**Bandwidth math.** With per-token acceptance `a` and a draft chain of length `L`, expected
tokens per streamed pass is `(1 − a^(L+1)) / (1 − a)`. Decode throughput becomes
`base_tps × E[tokens]`, where `base_tps = PCIe_BW / streamed_bytes`.

**Measured (real weights, `scripts/svmi-bitspec.py` on Qwen2.5-0.5B Q4_K_M).** Agreement
between the low-bit draft and full precision at the token decision (argmax of the output
projection over 1024 real hidden-state samples):

| draft bits | resident size | top-1 agree | → speedup (L=8) |
| ---: | ---: | ---: | ---: |
| 2-bit | 12.5% of f16 | 65% | 2.8x |
| 3-bit | 18.8% of f16 | **95%** | **7.4x** |
| 4-bit | 25.0% of f16 | 99% | 8.8x |

A 3-bit resident draft turns the ~0.6 tok/s streaming floor for a 70B into ~4 tok/s,
consistent with published SpecExec results (4–6 tok/s, 70B-4bit, RTX 4090) but with a
zero-training, zero-extra-model draft.

**Honesty.** The validator measures agreement at the *lm_head* only; a real draft runs the
whole low-bit network, so per-layer error compounds and end-to-end acceptance will be
lower than the single-layer number — treat the table as an optimistic bound and a go/no-go,
not a promise. Acceptance also depends on the sampler; greedy is assumed. Verification is
exact, so **outputs remain distribution-identical** regardless of draft quality — a bad
draft only costs speed, never correctness. Build order: resident low-bit weight buffers →
draft forward pass on the resident copy → tree/chain verification hooked into the existing
streaming scheduler (the verification pass *is* the normal streamed forward pass).

## 2. Pipelined streaming GEMM — intra-op transfer/compute overlap  *(hide)*

**Idea.** SVMI today streams a whole weight tensor, waits on its upload event, then
launches the matmul. Instead, tile the weight by output rows into chunks, and launch the
matmul on chunk *i* as soon as chunk *i* lands while chunk *i+1* is still in flight. The
op's wall-clock drops from `transfer(whole) + compute(whole)` to
`transfer(whole) + compute(one tile)` — i.e. compute is fully hidden except for the last
tile, and the first partial results appear after just one tile's transfer.

**Why it's new.** llama.cpp streams and computes at whole-tensor granularity; cuBLAS/cuDNN
pipelines assume weights already resident. Fusing the H2D copy schedule *into* the GEMM
tiling on the PCIe-bound path is not done in any OSS engine.

**Math.** Doesn't lower the `bytes/BW` floor (all bytes still move), so it is a *hide*
multiplier, not an *amortize* one. Its wins: (a) removes the compute tail from every
streamed op (meaningful at prefill, where compute per tensor is large), and (b) cuts
time-to-first-tile, which matters when many small consumers wait on one weight. Best used
*under* BitSpec, not instead of it.

**Honesty.** Requires a GEMM that accepts a "rows valid up to k" predicate or per-tile
launches; adds kernel-launch overhead that eats the benefit for small tensors (gate on
tensor size). Token-identical.

## 3. Stream-once, serve-many — cross-request weight broadcast  *(amortize, servers)*

**Idea.** In `llama-server` with concurrent requests, align the decode step of all active
sequences to a shared **layer-streaming clock**: when layer *L*'s weights are streamed in,
every waiting sequence consumes them in the same batched matmul before the slot is
recycled. One PCIe stream of the model then serves *N* sequences per pass instead of one.

**Why it's new.** Continuous batching (vLLM, llama.cpp server) batches *compute* but
assumes weights are resident; when weights are streamed, existing servers re-stream per
scheduling group. Making the *streaming schedule* the synchronization primitive that
concurrent requests rendezvous on is a new scheduling policy for offloaded serving.

**Math.** Per-token streamed bytes fall from `model_bytes` to `model_bytes / N_batched`.
This is the throughput-mode dual of BitSpec's latency-mode amortization, and the two
compose multiplicatively: `N` sequences × `E[tokens]` accepted each.

**Honesty.** Helps aggregate throughput, not single-stream latency; needs enough
concurrent traffic to fill the window, and a bounded wait so a lone request isn't stalled
waiting for company (fall back to solo streaming under a latency deadline). Token-identical.

## 4. Elastic residency — self-tuning hot set  *(stream less)*

**Idea.** Residency is correctness-neutral (every weight is used either way), so it can be
tuned online with zero risk. Track a per-tensor **stall contribution** = time the compute
stream spent blocked on that tensor's upload event. When VRAM frees up (KV cache shrinks,
requests drain), promote the highest-stall streamed tensors to resident; on memory
pressure, demote the lowest. The plan converges to the hot set that actually costs the most
to stream *for this workload*, rather than the static, offline guess `svmi-plan.py` makes.

**Why it's new.** FlexGen solves a *static, throughput-mode* placement; SVMI's planner is
static too. An online, latency-mode controller driven by measured stall — and safe because
residency never changes outputs — is not in any engine.

**Honesty.** Needs hysteresis to avoid promote/demote thrash, and promotion costs a
one-time upload. Gains taper once the genuinely hot layers are resident. Token-identical.

## 5. Draft-guided expert prefetch for MoE  *(stream less + hide, MoE)*

**Idea.** Couple BitSpec (§1) with MoE paging: the resident low-bit draft is run first, and
its router logits **predict which experts the next tokens will select**. The streamer
prefetches exactly those experts before the true (full-precision) router runs. On a
misprediction the engine blocks and fetches the correct expert — so correctness never
depends on the prediction, only speed does.

**Why it's new.** llama.cpp and KTransformers either compute cold experts on the CPU or
fetch reactively after the router runs (a serial stall). Using a resident draft's routing
as a *speculative prefetch oracle* for the weight streamer is unexplored.

**Honesty.** Value scales with routing predictability and MoE sparsity; dense models get
nothing. Router top-k agreement should be validated the same way §1 validates lm_head
agreement (extend `svmi-bitspec.py` to `blk.*.ffn_gate_inp`). Token-identical.

## 6. Residual-precision streaming — token-identical delta transport  *(stream denser)*

**Idea.** Keep a low-bit copy of *every* layer resident (as in §1), then to get exact
results stream only the **residual** `full_weight − dequant(resident_low_bit)` instead of
the full weight. If the residual codes smaller than the full quantized weight, you move
fewer bytes with **zero quality loss**.

**Why it's new / why it might not work.** It is a novel token-identical bandwidth reducer,
but it is honest-to-flag as *probably marginal*: a good quantizer already whitens its
residual, so the residual's entropy is often close to the original's. It only pays off when
`entropy(residual) + amortized(resident) < entropy(full)`. Decision procedure: extend
`scripts/svmi-entropy.py` to emit the residual stream and measure its bits/byte; proceed
only if the residual codes below ~0.8x of the full weight. Included here as a documented
negative-result candidate — the kind of idea worth *disproving* cheaply before building.

---

## 7. MAVM — Multi-Agent Virtual Memory  *(share everything, once)*

**Problem.** Agentic workloads don't run one sequence — they run a *fleet*: a planner, a
coder, a reviewer, tool-callers, all hitting the same model concurrently, mostly sharing a
system prompt, and mostly *idle* while they wait on tools or each other. Provisioning each
agent as if it were a standalone deployment (its own model copy, its own full-context KV,
its own prefill of the shared prompt) wastes VRAM, tokens, and time in direct proportion
to the fleet size.

**Idea.** Treat the SVMI substrate — resident weights, streaming ring, KV pool — as a
single **virtual memory system shared by all agents**, with four mechanisms:

1. **Shared weights, once.** All agents run against one resident hot set and one streaming
   ring. The marginal VRAM cost of an additional agent is *only its unique KV* — for a
   7B-class model at 8K context that is tens of MiB, not gigabytes. Weight VRAM is
   `O(1)` in the number of agents instead of `O(N)`.
2. **Prefix KV dedup (copy-on-write).** Agents sharing a system prompt share the KV pages
   for that prefix; a page is copied only when an agent's sequence diverges. This is exact
   (same tokens ⇒ same KV) and cuts both VRAM *and* token work: the shared prompt is
   prefilled **once** for the whole fleet, saving `(N−1) × prefix_tokens` of prompt
   processing — at 32 agents with a 2K shared prompt, ~86 % of the fleet's total token
   work disappears.
3. **Stream-once-serve-many as the fleet clock (§3).** Active agents decode in lock-step
   on the shared layer-streaming schedule, so one PCIe pass over the streamed weights
   serves the whole batch. Aggregate throughput scales with the number of *active* agents
   until compute, not PCIe, becomes the bottleneck; combined with BitSpec (§1) the
   streaming-bound region compounds to `passes/s × N_active × E[tokens]`.
4. **Idle-aware KV spill (elastic residency for sequences).** An agent blocked on a tool
   call contributes nothing but occupies KV. MAVM demotes idle agents' unique KV to pinned
   host RAM and uses the freed VRAM to either admit more active agents or promote streamed
   layers resident (faster decode for everyone). On wake, the KV pages back in behind the
   first layer's compute — the same latency-hiding trick SVMI already plays with weights.
   The GPU never idles on behalf of an idle agent.

**Why it's different.** Serving stacks (vLLM, SGLang) share prefixes and batch requests,
but they assume the *weights fit in VRAM*. MAVM's contribution is doing this on top of a
**streamed** model: the weight-streaming clock becomes the batching mechanism (not an
obstacle), and residency is arbitraged *across* weights and KV — idle KV is demoted so hot
weights can be promoted, which no fits-in-VRAM server needs to consider.

**Token identity.** Preserved throughout: shared weights are the same weights; shared
prefix KV is bit-identical to recomputed KV; batching and spill change scheduling, not
math; BitSpec verifies exactly.

**Validation.** `scripts/svmi-fleet.py` implements the capacity/throughput/cost model,
reading real GGUF dimensions (or `--profile 7b/8b/13b/70b` synthetic shapes). At long
context MAVM composes with CTX-VM (§8); headline runs are listed there.

---

## 8. CTX-VM — paged context virtual memory  *(share + stream the context itself)*

**Problem.** MAVM alone dies at real context lengths. A 70B GQA model's KV cache costs
~170 KiB/token (q8_0, all 80 layers): at 131,072 tokens that is **21 GiB per agent**, at
262,144 tokens **43 GiB** — more than the weights, and per agent. No amount of weight
streaming rescues a fleet whose *context* doesn't fit. Deep KV quant only divides by ~2–4;
the growth is linear in ctx and in agents.

**Idea.** Apply the SVMI move — virtual memory with a hot working set — to the KV cache:

1. **Cold tier (pinned host RAM).** The full KV for every agent lives in pinned host
   memory (optionally deeper-quantized, e.g. q4_0 cold vs q8_0 hot), DMA-reachable on the
   same upload queues as streamed weights. VRAM stops scaling with context length.
2. **Page table in VRAM (landmarks).** KV is chunked into fixed pages (default 256
   tokens). Per page, per layer, per KV head, a small **key landmark** (f16 summary
   vector) stays resident — ~160 KiB per page for a 70B, i.e. **80 MiB** of VRAM indexes
   131K tokens whose full KV is 21 GiB. Each decode step scores the query against
   landmarks to rank pages — the attention analogue of a TLB.
3. **Hot window (VRAM).** Attention sinks + the local window + the top-ranked hot pages
   stay resident (default 4,096 tokens ≈ 0.66 GiB for a 70B). Missed pages are fetched on
   demand; with temporal locality of attention, misses are a few hundred tokens' worth per
   step (~85 MB for a 70B at q4 cold — small next to the 30+ GiB weight pass it rides
   along with).
4. **Shared pages are fleet-global.** A 98K-token shared corpus is one set of cold pages
   and one set of landmarks for the whole fleet, composing directly with §7's prefix
   dedup: per-agent cost is only the *unique* context.

Per-agent VRAM drops from `O(ctx)` to `O(window + ctx / page_size)` — 8–30× smaller at
131K–256K in the runs below — and per-agent host RAM becomes the scaling wall, which is
the cheap resource ($/GiB roughly 20× below VRAM).

**Why it's different.** Paged KV (vLLM's PagedAttention) manages VRAM fragmentation, not
capacity — pages stay in VRAM. Offload engines that spill KV to host read it back
wholesale, paying full-context PCIe per token. Quest/InfLLM-style top-k attention selects
pages but keeps everything in VRAM. CTX-VM combines the three: host-resident capacity,
landmark-directed *selective* fetch, and integration with the weight-streaming clock so
KV fetches share the DMA engines and the batching schedule that SVMI already runs.

**Honesty note — token identity.** Landmark-directed page selection is the one mechanism
in this document that is **approximate by default**: if a relevant page is not fetched,
attention differs from the dense result. Two exact modes exist at a bandwidth price:
(a) *exact scoring* — keep all keys resident at 2-bit (≈ 2.8 GiB for 70B @131K), score
every token exactly, fetch only the values of the true top-k; (b) *BitSpec-style
verification* — periodically recompute a step with dense attention and roll back on
divergence. The planner models the default (approximate) mode; flag `--kv-fetch` upward
to trade bandwidth for recall.

**Validation** (`scripts/svmi-fleet.py`, analytical, real shapes):

* **70B @131,072 ctx on RTX 3060 (12 GB), 96K shared corpus, q4 cold, 128 GiB host** —
  **16 agents fit** (marginal agent: 0.35 GiB VRAM + 2.8 GiB host vs 61 GiB naive);
  modeled ~61 tok/s aggregate, ~3.8 tok/s per agent in the streaming-bound region.
* **70B @131,072 ctx on RTX 2080 Ti (11 GB), PCIe 3.0** — 8 agents fit; ~16 tok/s
  aggregate.
* **8B @262,144 ctx on RTX 3060** — 8 agents fit with weights fully resident (KV, not
  weights, is the binding constraint; host RAM is the wall at 45 GiB).
* **0.5B @131,072 ctx on GTX 1660 Ti (6 GB)** — **128 agents** fit; prefix dedup saves
  16.7M prompt tokens at 256 agents.

The `NO` rows in the planner output name the binding wall (VRAM vs host RAM) and the flag
that moves it (`--kv-window`, `--cold-kv-type q4_0`, `--host-ram`).

---

## 9. ARBITER — compute follows memory  *(don't stream at all)*

**The 2049 inversion.** Every offloading engine to date — including SVMI §1–§8 — treats
the GPU as *the* computer and PCIe as the road weights must travel to reach it. Look at
the machine the way a scheduler in 2049 would: it is three memory fabrics, each with
silicon already attached —

| fabric | bandwidth | attached compute |
| --- | --- | --- |
| VRAM | 360–1000 GB/s | GPU SMs |
| DDR4/DDR5 | 35–280 GB/s | CPU cores (AVX-VNNI, AMX) |
| PCIe | 12–48 GB/s | none — it's a pipe |

Bandwidth-bound decode touches every weight once per token. Moving 1 GB of weights over
PCIe 4.0 costs 42 ms; computing it *in place* on a DDR5 CPU costs 14 ms; in VRAM, 2.8 ms.
**PCIe — the fabric offloading engines funnel everything through — is the slowest path in
the machine.** The optimal policy is therefore: *weights never cross PCIe on the critical
path*. Compute goes to where each weight already lives.

**Mechanism.** ARBITER composes three pieces:

1. **Bandwidth-arbitrage routing.** Split the weights between VRAM-resident (GPU
   verifies) and host-resident (CPU verifies) shares, chosen not by layer-count
   heuristics (`-ngl`) but by solving for the split that equalizes the two fabrics'
   pass times — accounting for VRAM bandwidth, effective RAM bandwidth, the CPU's
   quantized-GEMM FLOP ceiling, and the speculative batch size `E`. The optimum moves
   when any of those change, which is why it must be solved, not hard-coded.
2. **Cross-fabric draft/verify pipelining.** The resident draft (BitSpec low-bit copy or
   a small model) generates the next speculative run on the GPU *while* the CPU verifies
   the previous batch — the two fabrics work concurrently instead of taking turns. The
   CPU reads each host weight **once per batch of E tokens** (speculative verification is
   a batch-E GEMM), which is what lets a 70 GB/s memory system behave like a 350 GB/s one
   at E = 5. Stock partial offload gets neither the batching nor the overlap.
3. **PCIe demoted to a background lane.** Freed of weight traffic, PCIe carries only
   per-layer activations (kilobytes) and **elastic residency migration**: hot layers
   promote into VRAM at ~90 % of link rate behind the compute, continuously re-balancing
   the arbitrage split as VRAM frees (§4 gets a dedicated free channel).

**Exactness.** Verification computes the true model everywhere; ARBITER changes *where*
the math runs, never the math. Token-identical by construction.

**Modeled results** (`scripts/svmi-arbiter.py`, real shapes, conservative CPU numbers):

| config | stock partial offload | SVMI stream+spec | **ARBITER** |
| --- | --- | --- | --- |
| 70B Q4_K_M, RTX 3060 + DDR5-2ch, 16 cores | 2.0 tok/s | 3.4 | **10.3 (5.2×)** |
| 70B Q4_K_M, RTX 2080 Ti + DDR4-2ch, 8 cores | 1.0 tok/s | 1.7 | **5.0 (5.1×)** |
| 8B Q4_K_M, RTX 3060 + DDR5-2ch | 72.9 tok/s | 174.6 | **208.1 (2.9×)** |

The tool also names the binding resource at the optimum (usually CPU RAM bandwidth or
FLOPs) — meaning the next dollar goes to RAM or cores, not a bigger GPU.

**Honesty notes.** (a) The CPU must actually be available — a busy host (MAVM fleets
doing tool work) shrinks the CPU share; the router re-solves with a utilization factor.
(b) Effective RAM bandwidth assumes NUMA-local pinned allocations; cross-socket traffic
halves it. (c) The draft must fit in VRAM next to the GPU share — for a 70B on a 12 GB
card that means a ~1 GiB external draft or a *partial* BitSpec copy, so `E ≈ 4–5` is the
conservative planning number, not the measured 7. (d) llama.cpp already runs host layers
on CPU at batch 1; the 5× comes from the *combination* of batched speculative
verification on the CPU, solved (not guessed) splits, and draft/verify overlap — each
piece alone is worth far less. (e) Laptop thermals can clip sustained CPU GEMM; derate
`--cores`.

---

## Composition and priority

The multipliers stack: `throughput ≈ base_tps × E[tokens] (BitSpec) × N (serve-many)`,
with elastic residency and pipelined GEMM lowering `base_tps`'s denominator and its tail.

Build priority by expected payoff per unit of engineering risk:

1. **BitSpec (§1)** — largest single win, already de-risked by `svmi-bitspec.py`; verification reuses the existing streamed forward pass.
2. **Elastic residency (§4)** — cheap, safe, immediate; pure host-side scheduler logic.
3. **Stream-once-serve-many (§3)** — high value for `llama-server` deployments.
4. **Pipelined streaming GEMM (§2)** — solid prefill/tail win; needs a tiling-aware kernel.
5. **ARBITER (§9)** — the largest modeled decode multiplier for models that don't fit
   (5× over stock partial offload); reuses §1's draft and existing CPU kernels, so the
   new engineering is the router and the overlap, not new math.
6. **MAVM (§7)** — the multiplier for agentic fleets; §3 + prefix dedup + KV spill composed into one substrate, planned with `svmi-fleet.py`.
7. **CTX-VM (§8)** — the prerequisite for 131K/256K context on consumer VRAM; extends §7, planned with the same tool.
8. **Draft-guided MoE prefetch (§5)** — high value but MoE-only; depends on §1.
9. **Residual-precision streaming (§6)** — validate-then-maybe; likely a documented negative.

All preserve token-identity (BitSpec/§5 via exact verification, CTX-VM/§8 via its exact
modes, ARBITER/§9 by routing-not-math; the rest by construction), which keeps SVMI's
core promise: same outputs, a fraction of the VRAM.

---

# Second wave — July 2026: shrinking the context itself

The first wave (§1–§9) moved weights and KV *around* — streamed, paged, shared. The
second wave shrinks what has to move at all. Four techniques plus one allocator change,
each with a validator; the fleet planner (`scripts/svmi-fleet.py`) models all of them
via `--kv-lat`, `--landmarks`, `--prefetch-hit`, and `--prune-cold`.

## 10. KV-LAT — latent KV compression (MLA-style retrofit)

**Problem.** CTX-VM (§8) made KV *capacity* cheap (host RAM) but every fetched page
still pays full GQA bytes over PCIe, and the cold tier still costs ~2.8 GiB/agent for a
70B @131K even at q4_0. The KV bytes themselves are the next wall.

**Idea.** DeepSeek's MLA showed attention can run from a small per-token *latent*
instead of full K/V: cache `c_t = x_t W_down` (rank `r`, shared across KV heads) and
absorb the up-projections into the attention matmuls, keeping only a small decoupled
RoPE key (`d_r` dims) exact. This fork already carries MLA-style cache plumbing for
DeepSeek architectures (`src/llama-kv-cache-dsv4.*`); KV-LAT is the **retrofit** of the
same trick onto stock GQA checkpoints: jointly factor each layer's `[W_K; W_V]` with a
truncated (activation-whitened) SVD, fine-tune nothing, verify everything.

* Bytes/token/layer drop from `2 · d_head · n_head_kv · bpe` to `(r + d_r) · bpe` —
  for a 70B (GQA-8, d_head 128): 2,048 → ~640 bytes at r 512 (q8), a **3.2×** cut that
  multiplies every CTX-VM number: cold tier, fetch traffic, and host-RAM wall.
* Composes with §8's landmarks (they index latents just as well) and §1's BitSpec
  (periodic dense verification bounds the approximation).

**Validator** (`scripts/svmi-kvlat.py`): per-layer SVD spectra of `[W_K; W_V]` from a
real GGUF (dequantized via gguf-py) or a synthetic ensemble; reports the rank needed for
95/99/99.9 % spectral energy and the resulting bytes/token vs the GQA baseline.

**Honesty notes.** (a) Weight-space energy is an *optimistic proxy*: production ranks
need activation-aware calibration (whiten by the input covariance, à la 2025's
TransMLA/X-MLA retrofits) and a held-out perplexity check. (b) Token identity is lost
at finite rank — recover it with BitSpec-style verification or reserve KV-LAT for the
*cold* tier only (hot window stays exact). (c) RoPE does not commute with the down
projection; the decoupled-RoPE dims (`d_r`, typically 32–64) are the price of admission.

## 11. CTX-VM v2 — product-quantized landmarks, two-level page table

**Problem.** §8's f16 landmarks cost `n_layer · n_head_kv · d_head · 2` bytes/page —
160 KiB/page for a 70B. At 131K that is a pleasant 80 MiB; at **1M tokens it is 640 MiB
per agent**, and the index itself becomes the VRAM wall it was built to remove.

**Idea.** Landmarks are lookup keys, not math operands — compress them like a vector
database would. Product-quantize each per-head landmark (M sub-vectors × 8-bit codes,
shared 256-entry codebooks per layer), and add a coarse level: super-pages of 16 pages
with one f16 centroid, scored first; only surviving super-pages have their PQ codes
scored at all.

* `pq16` (M=16): 16 bytes/head/page → **16×** smaller index; 1M-token index for a 70B
  drops 640 MiB → 40 MiB. `pq8` halves it again for recall-tolerant workloads.
* Scoring cost falls with the coarse level: `O(n_super + 16·k_super)` instead of
  `O(n_pages)` — at 1M ctx that is ~50× fewer landmark dot products per step.

**Validator** (`scripts/svmi-pqindex.py`): top-k page recall of PQ landmarks vs exact
f16 landmarks on clustered synthetic keys (mixture-of-Gaussians with drift — the
structure real semantic pages exhibit); reports recall@k for pq16/pq8 and index bytes
at 131K/512K/1M for the 8B and 70B profiles.

**Honesty note.** PQ recall on synthetic clusters is an upper bound for adversarially
uniform keys; the validator's `--hard` mode (near-isotropic keys) gives the floor. The
coarse level assumes locality in page relevance — true for prose and code, weaker for
random-access retrieval workloads.

## 12. SPEC-PF — speculative page prefetch

**Problem.** §8 fetches missed pages *on demand*: the fetch sits on the critical path
of the attention op that needs it. At PCIe 3.0 a 4-page miss burst is ~1 ms — real
latency once weights stop being the bottleneck (resident models, MAVM fleets).

**Idea.** Attention page relevance has strong temporal locality: the top-k pages at
step `t` predict most of the top-k at `t+1`. Keep the previous step's page scores as a
prior, prefetch the predicted set on the *weight-streaming clock* (the upload queues
already tick every layer), and demand-fetch only true surprises.

**Validator** (`scripts/svmi-pageprefetch.py`): a trace simulator — sinks + local
window always hit; distal accesses follow a persistent Zipf process with drift —
measures prefetch hit-rate vs prefetch budget and the resulting reduction in
critical-path fetch bytes/step. Typical regime: 70–90 % of misses hidden at a prefetch
budget of 2× the average working set.

**Honesty note.** Locality collapses on topic switches (hit-rate dips for a few steps)
and the simulator's drift parameter is a guess until traced on real attention scores;
wire `--trace` to a llama.cpp run before believing the third decimal.

## 13. Cold-tier pruning — the attention-mass ledger

**Problem.** CTX-VM keeps *every* token's KV in host RAM forever; at 256K × 32 agents
the host RAM wall (§8's runs) is mostly tokens no head has attended to in thousands of
steps (H2O/SnapKV observation: attention mass is heavy-tailed).

**Idea.** Per page, accumulate the attention mass it received over the last N decode
steps (4 bytes/page ledger — free). Pages below threshold demote: first to a q2 archive
(landmark stays, fetch re-quantizes up), then out entirely under `--prune-cold`, with
the page's landmark marked so exact modes know the history is incomplete.

* Models 2–4× cold-tier shrink at 131K+ on prose workloads; the planner's
  `--prune-cold F` shows the host-RAM wall moving accordingly.

**Honesty note.** This is the one *lossy-by-design* technique in SVMI: a pruned page is
unrecoverable. Ship it opt-in, default off, and never prune inside the exact-mode
window. Needs a task-level eval (long-doc QA recall) before any default flips.

## 14. Unified VRAM pool — one budget for ring and window

**Problem.** The staging ring (§1) and the CTX-VM hot windows are sized statically and
separately, but their demand is anti-correlated: prefill is ring-hungry (weights
stream hard, attention is local), decode is window-hungry (weights amortize across the
fleet, attention roams).

**Idea.** One VRAM pool, re-split at phase boundaries: prefill borrows window budget
for +2–4 ring slots (deeper overlap), decode returns it as +1–2K tokens of hot window
per agent. No new allocations at runtime — the pool is pre-carved into page-sized
blocks both consumers speak.

The planner models the split implicitly today (`--stream-slots`, `--kv-window` are the
static knobs); the engine change is an allocator, not new math, and reuses the slot
machinery already in `ggml_backend_sched` (`ggml-backend.cpp`).

## Second-wave priority

1. **PQ landmarks (§11)** — pure index-side change, no numerics risk, unlocks 1M-token
   page tables; validated by `svmi-pqindex.py` and the C++ reference test
   (`tests/test-ctxvm-landmarks.cpp`).
2. **SPEC-PF (§12)** — scheduler-only, composes with everything; simulator-validated.
3. **KV-LAT (§10)** — biggest bytes win (multiplies §8 everywhere) but needs
   calibration infrastructure; start with cold-tier-only deployment.
4. **Unified pool (§14)** — allocator refactor, do it when §12 lands (same code area).
5. **Cold pruning (§13)** — last: lossy, needs evals, and §10 shrinks the same bytes
   losslessly-with-verification first.

The composition target for the wave: a 70B fleet at **1M tokens of addressable context
per agent** on a 12 GB card — 40 MiB index + 0.7 GiB window + latent cold tier in host
RAM — with the same token-identity story as the first wave: approximate fast paths,
exact verification available everywhere.

## 15. MoE-EP — predictive expert paging (phase 6, validated)

**Problem.** MoE checkpoints put most of their bytes in experts the average token never
touches: a Mixtral-class 8x7B carries 42 GiB of experts and activates 2/8 per layer; a
DeepSeek-V3-class model carries ~200 GiB and activates 8/256. The generic SVMI streamer
(§1) already pages experts *by demand* — but demand arrives when the router fires, one
layer before the bytes are needed, which is exactly the window prefetch needs.

**Idea.** Three residency policies, in ascending order of intelligence:

1. **static** — pin the globally popular experts (Zipf head), stream the tail;
2. **LRU** — page experts on use, evict least-recently-used (temporal stickiness pays);
3. **lookahead** — LRU + prefetch the *predicted* set from the router's early signal
   (previous layer's hidden state is a strong predictor of the next layer's routing),
   riding the same upload queues as §1, off the critical path.

**Validation** (`scripts/svmi-expertpage.py`, trace-model simulator, honesty knob
`--lookahead-hit`, default 0.7):

* Mixtral-8x7B-class on a 3060: 6.6 experts fetched on the critical path per token
  under lookahead vs 64 with no residency — **9.7×** less critical-path PCIe.
* Qwen3-30B-A3B-class (128 fine experts): **24.9×**; fine-grained expert grids page
  dramatically better because each miss is only 2.8 MiB.

**Honesty notes.** (a) The lookahead hit rate is *assumed*, not measured — wire the
simulator's trace mode to real router logits before quoting the multiplier. (b) The
popularity/stickiness parameters are literature-shaped, not fitted; batch serving
flattens popularity (more distinct experts per step) and lowers all policies toward
`none`. (c) Engine-side, this is the same staging-ring machinery as §1 — the new code
is the predictor and the eviction policy, not the transport.
