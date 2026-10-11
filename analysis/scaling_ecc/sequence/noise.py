"""complete-strand controlled noise with probabilities per original nucleotide."""
import numpy as np

BASES = np.asarray(list("ACGT"))
LOOKUP = {base: i for i, base in enumerate(BASES)}
ASCII = np.frombuffer(b"ACGT", dtype=np.uint8)
ASCII_TO_CODE = np.zeros(256, dtype=np.uint8)
ASCII_TO_CODE[ASCII] = np.arange(4)


def mutate(sequence, substitution, indel, rng):
    if not substitution and not indel:
        return sequence
    source = np.fromiter((LOOKUP[b] for b in sequence), dtype=np.int8)
    sub = rng.random(len(source)) < substitution
    if sub.any():
        source[sub] = (source[sub] + rng.integers(1, 4, size=sub.sum())) % 4
    if not indel:
        return "".join(BASES[source])
    events = rng.random(len(source)) < indel
    kind = rng.random(len(source)) < 0.5
    deleted = events & kind
    inserted = events & ~kind
    result = []
    insertion_bases = iter(BASES[rng.integers(0, 4, size=inserted.sum())])
    for i, value in enumerate(source):
        if not deleted[i]:
            result.append(BASES[value])
        if inserted[i]:
            result.append(next(insertion_bases))
    return "".join(result)


def allocate_reads(count, reference_count, mode, rng):
    total = count * 16 if mode == "per_strand" else reference_count * 16
    q, r = divmod(total, count)
    values = np.full(count, q, dtype=np.int32)
    values[rng.permutation(count)[:r]] += 1
    assert values.sum() == total
    return values


def mutate_many(sequence, count, substitution, indel, rng):
    """batch generation uses the same per-original-base noise law."""
    if count == 0:
        return []
    if not substitution and not indel:
        return [sequence] * count
    codes = ASCII_TO_CODE[np.frombuffer(sequence.encode(), dtype=np.uint8)]
    values = np.broadcast_to(codes, (count, len(codes))).copy()
    sub = rng.random(values.shape) < substitution
    values[sub] = (values[sub] + rng.integers(1, 4, size=sub.sum())) % 4
    bases = ASCII[values]
    if not indel:
        return [row.tobytes().decode() for row in bases]
    events = rng.random(values.shape) < indel
    deletions = events & (rng.random(values.shape) < 0.5)
    insertions = events & ~deletions
    inserts = ASCII[rng.integers(0, 4, size=values.shape)]
    paired = np.stack((bases, inserts), axis=-1).reshape(count, -1)
    keep = np.stack((~deletions, insertions), axis=-1).reshape(count, -1)
    return [row[mask].tobytes().decode() for row, mask in zip(paired, keep)]


def conditions():
    values = [("baseline", 0.0, 0.0, 0.0, False)]
    for p in (0.005, 0.01, 0.02, 0.05):
        values.append(("substitution", p, 0.0, 0.0, False))
    for p in (0.005, 0.01, 0.02, 0.05):
        values.append(("indel", 0.0, p, 0.0, False))
    for p in (0.05, 0.10, 0.20, 0.40):
        values.append(("dropout", 0.0, 0.0, p, False))
    values.append(("shared_template", 0.01, 0.0, 0.0, True))
    return values
