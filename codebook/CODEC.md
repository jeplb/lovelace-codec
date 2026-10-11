# S12 codec specification


## 1. Preserve magnitude and phase

For a float64 32×32 image, carry its mean μ and compute a unitary two-dimensional
Fourier transform of `image − μ`. Apply `fftshift` and retain exactly the centered
16×16 block `[8:24,8:24]`, frequencies −8 through +7 on each axis. This retains
256 complex coefficients, not just their magnitudes. The paired target is a
circular spatial shift of the first; their nearly identical Fourier magnitudes
and different phases make phase loss observable.

Set `s` to the maximum absolute real or imaginary coordinate within the retained
block. Independently quantize each coordinate `v` using

```text
q = clip(floor((v/s + 1) * 63/2 + 0.5), 0, 63)
v_hat = -s + 2*s*q/63
```

Real then imaginary components are packed in row-major coefficient order, six
bits each, most significant bit first. Thus 512 component labels use 384 bytes
per image. Quantization is endpoint-inclusive, has no exact zero level and is
not a learned audio quantizer. No coefficient is silently exempted.

Inverse reconstruction places the dequantized 16×16 complex block back in an
otherwise-zero 32×32 Fourier grid, applies `ifftshift` then unitary `ifft2`, and
returns **the real part plus the carried mean**. The crop is not Hermitian-complete;
discarding its imaginary part is the documented convention, not a new release fix.
Neither metric clipping nor spatial realignment is introduced. Exact coefficient
recovery reproduces the quantized S12 source ceiling, not the original image.

## 2. Carry a verifiable self-contained header

Each image carries the fixed big-endian 64-byte header `>4sBBBBBBHddI32s`:

| Field | Value / meaning |
|---|---|
| Magic / version | `LVIM` / 1 |
| Image height / width | 32 / 32 |
| Retained height / width | 16 / 16 |
| Component bits / flags | 6 / 1 |
| Mean / scale | Two float64 values μ and s |
| Payload length | 384 bytes |
| Carried SHA-256 | Digest of the preceding 32 header bytes concatenated with the 384-byte payload |

Header bounds are validated before physical reconstruction. The decoder receives
no expected payload hash; it must recover the digest from the reads. The evaluator
independently compares every claimed success with original header/payload bytes.
The checksum is integrity detection, **not** an outer error-correcting code.

## 3. Map packed bits to composition cells

Each byte is split into four two-bit digits, most significant first. Digits use
the frozen lexicographic physical mapping:

| Digit | A-run | C-run | T-run | Cell sequence form |
|---:|---:|---:|---:|---|
| 0 | 3 | 7 | 10 | A³C⁷T¹⁰ |
| 1 | 3 | 10 | 7 | A³C¹⁰T⁷ |
| 2 | 7 | 3 | 10 | A⁷C³T¹⁰ |
| 3 | 7 | 10 | 3 | A⁷C¹⁰T³ |

Each cell is 20 nt and represents two bits. Eight cells carry two bytes per
oligo. Base order is always A,C,T; only run lengths vary. The four state frequencies
are the corresponding lengths divided by 20. This is a **discrete mapping of
quantized coefficient bits**, not an assertion that composition coordinates equal
Fourier real/imaginary values or that a DNA strand is an oscillatory Fourier carrier.

Each image uses `(64 + 384)/2 = 224` oligos. Addresses 0–223 carry A, 224–447 carry
B, 448–471 are fitting pilots, and 472–479 are QC pilots. The first 32 oligos of
each image carry its existing header. Both images and all pilots share one read
pool; window/image identity is inferred from decoded global addresses.

## 4. Full 298-nt strands and addresses

```text
CCCAATTAATACGTTGCCTACTCCG
[16 transition-coded address runs × 5 nt]
GGGG
[eight 20-nt A/C/T cells]
GGGG
CGCTATCAGAACAAGAGAAGCACTA
```

The two 25-nt flanks and both delimiters are already present. The ordered reverse
PCR primer (5′→3′) is `TAGTGCTTCTCTTGTTCTGATAGCG`, the reverse complement of the
embedded right-hand region. Do not append duplicate primers or optimize away runs.

An 11-bit address message uses extended Hamming (16,11), parity positions 1,2,4,8
and overall parity position16. Message bits fill the other positions least
significant first. XOR with `0xF6CF`; process word bits least significant first.
Starting from implicit A in the cycle A,C,T, bit0 advances +1 and bit1 advances
+2. Expand each resulting base fivefold. This is a structured address code, not
outer payload ECC and not a claimed DNA edit-distance guarantee. The code space
contains 2,048 words; only 480 addresses are allocated in this pool.

## 5. Sequence channel and evidence

The generator consumes complete strands, coalesces maximal homopolymers first,
then uses separate `W_b(y | x)` for A/C/T address and payload runs. Axes are base
A,C,T; designed lengths 3–15; observed lengths 0–20. The saved models preserve C
undercall, length-dependent compression/spread and collapse. Approximate Figure2
digitization is not an empirical count table. Synthetic optimistic/central/
pessimistic models are sensitivity assumptions, not independent experiments.

The motivating assay is R10.4.1 with
`dna_r10.4.1_e8.2_400bps_sup@v5.2.0`; this is not a validated simulator of every
nanopore workflow. Short-primer perturbations, G-anchor errors, wrong-base edits,
truncation, orientation and structural stresses are separately labelled assumed
processes in [CHANNEL_PROVENANCE.md](CHANNEL_PROVENANCE.md). No uniform per-base
error rate replaces the run channel. FASTQ quality values are metadata only.

The checked-in ordinary pools have no specified whole-identity dropout or
chimeras, but multinomial coverage, damaged flanks, substitutions, insertions and
truncation still occur. The full archived stress results explicitly include
dropout, chimeras and abundance imbalance; their failures are not hidden.

## 6. Blind parsing and likelihood aggregation

The decoder locates and orients primers, uses a documented shortlist plus
run-profile scoring for addresses, and parses the payload pattern `(A,C,T)^8`
with reachable missing, inserted, split and merged-run paths. It does not group
every next three observed runs naively. A Viterbi maximum is a **hard alignment**,
not marginalization over alignment paths.

For each inferred cell, state `s` receives

```text
ell_s = log W_A(y_A | a_s) + log W_C(y_C | c_s) + log W_T(y_T | t_s).
```

Evidence is summed over accepted reads assigned to an address before final cell
decisions. One ambiguous read is not duplicated across address groups. Twenty-four
known pilot identities adapt the distributions using a 0.5-per-bin pseudocount
and 16 prior-equivalent observations; sparse rows with fewer than 8 observations
retain the prior. The eight QC pilots are not fitted. QC warnings/failures and
prior-based address-routing limitations remain as frozen in the implementation.

Cell probabilities yield digit/byte calls and confidence-based erasure flags.
No parity generation, RS, alternative outer code, repair, or true-payload filling
follows. Erasures remain flags: even flagged hard calls are retained for labelled
partial outputs. Header/checksum failure means unsuccessful full-file recovery.
No source mean/scale is secretly supplied to rescue a damaged header.

## 7. Isolation, metrics and complete costs

The Linux decoder's allowlist includes only public source-code/runtime files,
reads, format/physical-state configuration, channel prior and its output directory.
Simulator identities, source images, unknown sequences and expected checksums are
not readable. The parent evaluator and optional generator are separate from that
process. Every example replay tests truth-read denial.

Reports distinguish component-index and complex-coefficient errors, confidence
erasures, exact carried-header/payload success, unavailable reconstructions, and
distortion to both original targets and frozen S12 reconstructions. A complex
coefficient is wrong if either component label is wrong. Failed files are counted,
not averaged away. No additional enhancement is applied to reconstructed images.

One physical pool uses **480 × 298 = 143,040 designed nt**, including 448 data and
32 pilot oligos, 128 header bytes and 768 coefficient bytes. No payload parity or
padding is added. It represents 6,144 quantized-coefficient bits, a nominal
coefficient density of approximately 0.04293 bit per designed nt. Complete verified
recovery, not that nominal ratio, determines successful recovered user bits.

Depths 8/16/32/64 mean 3,840/7,680/15,360/30,720 attempted raw records. Rejected reads
and pilot reads remain in these totals; there is no accepted-read top-up. Actual
sequenced nucleotides vary with errors and are measured from each saved prefix.
The known public format, state mapping, channel priors and pilot templates are
preshared; this is not standalone self-discovering storage. The image codec does
not require the earlier 4,096-byte audio VQ model.

## 8. Claim boundary

The examples show a working, reproducible **model-qualified simulation pipeline**.
They are not exact empirical calibration or a biochemical feasibility guarantee.
Ordinary and severe-stress outcomes, Fourier-crop loss, quantization loss, parser
approximations, calibration limitations and unresolved primer accessibility are
all retained. A clean export check is neither a synthesis order nor a PCR pass.
