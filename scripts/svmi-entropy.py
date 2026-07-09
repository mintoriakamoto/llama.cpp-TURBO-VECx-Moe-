#!/usr/bin/env python3
"""
SVMI compressed-transport feasibility study (go/no-go for the Layer Compression Bus)

Measures the entropy of quantized GGUF weight blocks, split by structural component
(block scales/mins vs packed quant indices), to estimate the achievable lossless
compression ratio of an rANS/FSE-coded PCIe transport stream.

A ratio r means streamed-weight PCIe bandwidth is effectively multiplied by r
(GPU-side rANS decode at 250+ GB/s is far faster than PCIe, so decode is free).
Rule of thumb: proceed with the GPU codec if the byte-level entropy estimate is
>= 1.12x on the dominant tensor types; below that the kernel complexity is not worth it.

Usage:
  python3 scripts/svmi-entropy.py model.gguf [--max-tensors 40] [--per-type]
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "gguf-py"))

from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

# block layouts: type -> (block_bytes, [(name, offset, length), ...])
# scales/mins are the high-redundancy part; nibble planes are the high-entropy part
BLOCK_LAYOUTS = {
    GGMLQuantizationType.Q4_0: (18, [("scale_f16", 0, 2), ("nibbles", 2, 16)]),
    GGMLQuantizationType.Q8_0: (34, [("scale_f16", 0, 2), ("int8", 2, 32)]),
    GGMLQuantizationType.Q4_K: (144, [("scales_d", 0, 4), ("scales_m", 4, 8), ("nibbles", 12, 132)]),
    GGMLQuantizationType.Q5_K: (176, [("scales_d", 0, 4), ("scales_m", 4, 8), ("high_bits", 12, 32), ("nibbles", 44, 132)]),
    GGMLQuantizationType.Q6_K: (210, [("low_bits", 0, 128), ("high_bits", 128, 64), ("scales", 192, 16), ("d", 208, 2)]),
}


def byte_entropy(data: np.ndarray) -> float:
    """Shannon entropy in bits/byte of the byte stream (order-0 model, rANS-achievable)."""
    counts = np.bincount(data.reshape(-1), minlength=256)
    total = counts.sum()
    if total == 0:
        return 8.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--max-tensors", type=int, default=40,
                    help="sample at most N large tensors (default 40)")
    ap.add_argument("--per-type", action="store_true", help="print per-quant-type breakdown only")
    args = ap.parse_args()

    reader = GGUFReader(args.model)

    # sample the largest tensors first: they dominate the stream
    tensors = sorted(reader.tensors, key=lambda t: -int(t.n_bytes))[: args.max_tensors]

    # aggregated bits per component per quant type
    agg: dict[tuple, list] = defaultdict(lambda: [0.0, 0])  # (qtype, component) -> [bits, bytes]
    type_bytes: Counter = Counter()

    for t in tensors:
        qtype = GGMLQuantizationType(t.tensor_type)
        if qtype not in BLOCK_LAYOUTS:
            continue
        block_bytes, components = BLOCK_LAYOUTS[qtype]
        raw = np.frombuffer(t.data.tobytes(), dtype=np.uint8)
        n_blocks = len(raw) // block_bytes
        if n_blocks == 0:
            continue
        blocks = raw[: n_blocks * block_bytes].reshape(n_blocks, block_bytes)
        type_bytes[qtype.name] += len(raw)
        for name, off, length in components:
            comp = blocks[:, off : off + length]
            h = byte_entropy(comp)
            agg[(qtype.name, name)][0] += h * comp.size
            agg[(qtype.name, name)][1] += comp.size

    if not agg:
        print("no supported quantized tensors found (need Q4_0/Q8_0/Q4_K/Q5_K/Q6_K)")
        return 1

    print(f"sampled {len(tensors)} tensors from {Path(args.model).name}\n")
    print(f"{'type':7} {'component':11} {'bits/byte':>9} {'share':>7} {'ratio':>7}")

    per_type_bits: dict = defaultdict(lambda: [0.0, 0])
    for (tname, comp), (bits, nbytes) in sorted(agg.items()):
        h = bits / nbytes
        share = nbytes / sum(v[1] for k, v in agg.items() if k[0] == tname)
        print(f"{tname:7} {comp:11} {h:9.3f} {share:6.1%} {8.0 / h:7.3f}x")
        per_type_bits[tname][0] += bits
        per_type_bits[tname][1] += nbytes

    print()
    total_bits = total_bytes = 0.0
    for tname, (bits, nbytes) in per_type_bits.items():
        h = bits / nbytes
        total_bits += bits
        total_bytes += nbytes
        print(f"{tname:7} overall     {h:9.3f}         {8.0 / h:7.3f}x  "
              f"({type_bytes[tname] / 1024**3:.2f} GiB in stream)")

    overall = total_bits / total_bytes
    ratio = 8.0 / overall
    print(f"\nweighted overall entropy: {overall:.3f} bits/byte -> "
          f"estimated transport compression {ratio:.3f}x")
    print("verdict:", "PROCEED with GPU rANS transport codec (>= 1.12x)" if ratio >= 1.12
          else "SKIP the codec for this model/quant (ratio below 1.12x threshold)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
