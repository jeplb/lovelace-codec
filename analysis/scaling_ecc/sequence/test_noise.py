import numpy as np
from .noise import mutate, allocate_reads, conditions


def test_mutually_exclusive_indels_and_substitution_alphabet():
    sequence = "A" * 1000
    sub = mutate(sequence, 1, 0, np.random.default_rng(1))
    assert len(sub) == len(sequence) and "A" not in sub
    indel = mutate(sequence, 0, 1, np.random.default_rng(2))
    # each nucleotide produces either zero or two bases at total indel probability one
    assert len(indel) % 2 == 0
    assert set(indel) <= set("ACGT")
    assert mutate(sequence, 0, 0, np.random.default_rng(3)) == sequence


def test_budget_is_exact():
    for count in (32, 36, 40, 48, 56):
        assert allocate_reads(count, 32, "total", np.random.default_rng(4)).sum() == 32 * 16
        assert allocate_reads(count, 32, "per_strand", np.random.default_rng(4)).sum() == count * 16
    assert len(conditions()) == 14
