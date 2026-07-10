# SVMI research notes — novel techniques for streamed 70B inference

This document proposes techniques that are **not** in mainstream offloading engines
(llama.cpp, ExLlama, vLLM, TensorRT-LLM, DeepSpeed-Inference, FlexGen, PowerInfer as of
mid-2026) and are not already covered by the SVMI roadmap in [`svmi.md`](svmi.md). Each
entry states the idea, why it is new, the bandwidth math, an honesty note on limits, and
what to build. Everything is anchored to the SVMI axiom: for a model that does not fit in
VRAM, decode latency is `bytes_streamed / PCIe_bandwidth`, and the only ways to beat it
are **stream less**, **stream denser**, **amortize each stream**, or **hide the stream** —
plus, for multi-agent fleets, a fifth axis: **share every stream and every byte across
agents** (§7), and for long context a sixth: **page the context itself** (§8).

Three of these ship with working, GPU-free validators: BitSpec
(`scripts/svmi-bitspec.py`), and MAVM + CTX-VM (`scripts/svmi-fleet.py`); measured
results are inline.

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

## Composition and priority

The multipliers stack: `throughput ≈ base_tps × E[tokens] (BitSpec) × N (serve-many)`,
with elastic residency and pipelined GEMM lowering `base_tps`'s denominator and its tail.

Build priority by expected payoff per unit of engineering risk:

1. **BitSpec (§1)** — largest single win, already de-risked by `svmi-bitspec.py`; verification reuses the existing streamed forward pass.
2. **Elastic residency (§4)** — cheap, safe, immediate; pure host-side scheduler logic.
3. **Stream-once-serve-many (§3)** — high value for `llama-server` deployments.
4. **Pipelined streaming GEMM (§2)** — solid prefill/tail win; needs a tiling-aware kernel.
5. **MAVM (§7)** — the multiplier for agentic fleets; §3 + prefix dedup + KV spill composed into one substrate, planned with `svmi-fleet.py`.
6. **CTX-VM (§8)** — the prerequisite for 131K/256K context on consumer VRAM; extends §7, planned with the same tool.
7. **Draft-guided MoE prefetch (§5)** — high value but MoE-only; depends on §1.
8. **Residual-precision streaming (§6)** — validate-then-maybe; likely a documented negative.

All preserve token-identity (BitSpec/§5 via exact verification, CTX-VM/§8 via its exact
modes; the rest by construction), which keeps SVMI's core promise: same outputs, a
fraction of the VRAM.
