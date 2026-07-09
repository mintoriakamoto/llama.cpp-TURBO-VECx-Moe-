# SVMI research notes — novel techniques for streamed 70B inference

This document proposes techniques that are **not** in mainstream offloading engines
(llama.cpp, ExLlama, vLLM, TensorRT-LLM, DeepSpeed-Inference, FlexGen, PowerInfer as of
mid-2026) and are not already covered by the SVMI roadmap in [`svmi.md`](svmi.md). Each
entry states the idea, why it is new, the bandwidth math, an honesty note on limits, and
what to build. Everything is anchored to the SVMI axiom: for a model that does not fit in
VRAM, decode latency is `bytes_streamed / PCIe_bandwidth`, and the only ways to beat it
are **stream less**, **stream denser**, **amortize each stream**, or **hide the stream**.

One of these (BitSpec) ships with a working, GPU-free validator
(`scripts/svmi-bitspec.py`); the measured result is below.

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

## Composition and priority

The multipliers stack: `throughput ≈ base_tps × E[tokens] (BitSpec) × N (serve-many)`,
with elastic residency and pipelined GEMM lowering `base_tps`'s denominator and its tail.

Build priority by expected payoff per unit of engineering risk:

1. **BitSpec (§1)** — largest single win, already de-risked by `svmi-bitspec.py`; verification reuses the existing streamed forward pass.
2. **Elastic residency (§4)** — cheap, safe, immediate; pure host-side scheduler logic.
3. **Stream-once-serve-many (§3)** — high value for `llama-server` deployments.
4. **Pipelined streaming GEMM (§2)** — solid prefill/tail win; needs a tiling-aware kernel.
5. **Draft-guided MoE prefetch (§5)** — high value but MoE-only; depends on §1.
6. **Residual-precision streaming (§6)** — validate-then-maybe; likely a documented negative.

All six preserve token-identity (BitSpec/§5 via exact verification; the rest by
construction), which keeps SVMI's core promise: same outputs, a fraction of the VRAM.
